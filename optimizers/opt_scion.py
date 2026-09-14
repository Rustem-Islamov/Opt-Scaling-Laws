import torch

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


class OptimisticScion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,
        dual_momentum1=0, dual_momentum2=0, 
        radius=1.0, constrained=True,
        norm: str='Spectral', norm_kwargs: dict=None):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if dual_momentum1 < 0.0 or dual_momentum2 < 0.0:
            raise ValueError(f"Invalid momentum value: {dual_momentum1}, {dual_momentum2}")
        defaults = dict(lr=lr, radius=radius, constrained=constrained,
            dual_momentum1=dual_momentum1, dual_momentum2=dual_momentum2, 
            norm=norm, norm_kwargs=norm_kwargs)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            dual_momentum1 = group['dual_momentum1']
            dual_momentum2 = group['dual_momentum2']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            radius = group['radius']
            constrained = group['constrained']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                # Set states
                if 'mom_buff' not in state.keys():
                    state['mom_buff'] = torch.clone(g)

                mom_buff = state['mom_buff']

                # Dual averaging
                mom_buff.mul_(dual_momentum1).add_(g, alpha=1-dual_momentum1)
                if dual_momentum2 != 0: # nesterov momentum
                    mom_buff = mom_buff.mul(dual_momentum2).add(g, alpha=1-dual_momentum2)

                update = radius * norm_backend.lmo(mom_buff)

                if constrained:
                    # x_{k+1} = (1-lr)*x_k - lr * radius * lmo(m)
                    p.data.mul_(1-lr).add_(update, alpha=-lr)
                else:
                    # x_{k+1} = x_k - lr * radius * lmo(m)
                    p.data.add_(update, alpha=-lr)
