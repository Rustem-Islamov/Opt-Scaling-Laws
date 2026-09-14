# Modified from: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging

from dataclasses import dataclass
from tqdm import tqdm

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Scion optimizer

def zeroth_power_via_svd(G):
   U, S, V = G.svd()
   return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError


class Spectral(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = g.shape
        g *= (d_out / d_in)**0.5
        return g


class Sign(Norm):
    def __init__(self, zero_init=False):
        self.zero_init = zero_init

    def lmo(self, g):
        out_channels, in_channels = g.shape     # in_channels=768
        return (1/in_channels)*torch.sign(g)    


norm_dict = {
    'Spectral': Spectral,
    'Sign': Sign
}



class ScionLight(torch.optim.Optimizer):
    """Memory-efficient variant of the Scion optimizer.
    
    This implementation saves memory by storing only the averaged gradient instead of 
    both the gradient and its average. Note that gradients should not be zeroed since
    p.grad is used directly to store the gradient average.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv', 
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)
        scale (float, optional): Scale factor for updates (default: 1.0)
        unconstrained (bool, optional): Whether to use unconstrained updates (default: False)
    
    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'scale': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'scale': radius*60.0,
        ... }]
        >>> optimizer = ScionLight(optim_groups, lr=2**-12, momentum=0.1)
    """
    def __init__(self, params, lr=1e-3, momentum=1.0, norm: str='Auto', norm_kwargs: dict=None, scale=1.0, unconstrained=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(lr=lr, momentum=momentum, scale=scale, unconstrained=unconstrained, norm=norm, norm_kwargs=norm_kwargs)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                G = p.grad
                if G is None:
                    continue

                update = scale * norm_backend.lmo(G)
                if not unconstrained:
                    p.data.mul_(1-lr)
                p.data.add_(update, alpha=-lr)
                
                if momentum != 1:
                    G.mul_(1-momentum)

