import math
import torch
import math
from typing import Optional, Dict
import torch.distributed as dist

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

def ddp_allreduce_scalar(x):
    """All-reduce a scalar across GPUs."""
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x.item() / dist.get_world_size()

class Norm(object):
    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError
    
    def norm(self, w):
        """Primal norm whose unit ball defines the LMO used in lmo()."""
        raise NotImplementedError


class ColNorm(Norm):
    """
    Column-wise normalization.

    Args:
        normalized (bool, optional): If True, normalizes by the input dimension. Use True only for non-input layers.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=False, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1) 
        rms_values = 1/math.sqrt(g.size(0))*torch.sqrt(torch.sum(g ** 2, dim=0, keepdim=True))
        if self.normalized:
            rms_values *= g.size(1)
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1) 
        return g

    def init(self, w):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=0, keepdim=True)
        w.data *= math.sqrt(w.size(0))
        if self.normalized:
            w.data /= w.size(1)
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return w
    
    def norm(self, w: torch.Tensor) -> torch.Tensor:
        x = w.transpose(0, 1) if self.transpose else w
        if x.ndim != 2:
            raise ValueError(f"ColNorm.norm expects 2D tensor, got {tuple(w.shape)}")

        # Column RMS: sqrt(mean over rows)
        col_rms = torch.sqrt(torch.mean(x.float() ** 2, dim=0))  # (d_in,)
        val = col_rms.max()

        if self.normalized:
            val = val * x.size(1)  # d_in

        return val.to(dtype=w.dtype)


class RowNorm(Norm):
    """
    Row-wise normalization.

    Args:
        normalized (bool, optional): If True, normalizes by the input dimension. Use False only for the input layer.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=True, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1) 
        rms_values = torch.sqrt(torch.sum(g ** 2, dim=-1, keepdim=True))
        if self.normalized:
            rms_values *= math.sqrt(g.size(-1))
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1) 
        return g

    def init(self, w):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=-1, keepdim=True)
        if self.normalized:
            w.data /= math.sqrt(w.size(-1))
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)       
        return w
        
    def norm(self, w: torch.Tensor) -> torch.Tensor:
        x = w.transpose(0, 1) if self.transpose else w
        if x.ndim != 2:
            raise ValueError(f"RowNorm.norm expects 2D tensor, got {tuple(w.shape)}")

        row_l2 = torch.sqrt(torch.sum(x.float() ** 2, dim=-1))  # (d_out,)
        val = row_l2.max()

        if self.normalized:
            val = val * math.sqrt(x.size(-1))  # sqrt(d_in)

        return val.to(dtype=w.dtype)


class BiasRMS(Norm):
    def lmo(self, g):
        eps = 1e-8
        rms_values = torch.sqrt(torch.mean(g ** 2, dim=0, keepdim=True))
        g = g / (rms_values + eps)
        return g

    def init(self, g):
        return torch.nn.init.zeros_(g)
        
    def norm(self, w: torch.Tensor) -> torch.Tensor:
        # BiasRMS is intended for 0D/1D tensors (bias/scale vectors)
        if w.ndim == 0:
            return w.abs()
        if w.ndim == 1:
            return torch.sqrt(torch.mean(w.float() ** 2)).to(dtype=w.dtype)
        raise ValueError(f"BiasRMS.norm expects 0D/1D tensor, got {tuple(w.shape)}")


class SpectralConv(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        if g.ndim == 3:    # Conv1d
            out_channels, in_channels, k = g.shape
            g *= (out_channels / in_channels)**0.5 / k
        elif g.ndim == 4:   # Conv2d
            out_channels, in_channels, k, _ = g.shape
            g *= (out_channels / in_channels)**0.5 / (k ** 2)
        return g
    
    def init(self, w):
        w_fp = w.data.double()
    
        if w.ndim == 3:  # Conv1d: (out_ch, in_ch, k)
            out_channels, in_channels, k = w_fp.shape
            for kx in range(k):
                torch.nn.init.orthogonal_(w_fp[:, :, kx])
            w_fp.mul_((out_channels / in_channels) ** 0.5 / k)
    
        elif w.ndim == 4:  # Conv2d: (out_ch, in_ch, kx, ky)
            out_channels, in_channels, kx, ky = w_fp.shape
            for i in range(kx):
                for j in range(ky):
                    torch.nn.init.orthogonal_(w_fp[:, :, i, j])
            w_fp.mul_((out_channels / in_channels) ** 0.5 / (kx * ky))
    
        else:
            raise ValueError(f"SpectralConv.init expected 3D/4D conv weight, got {tuple(w.shape)}")
    
        w.data = w_fp.to(dtype=w.data.dtype)
        return w
        
    def norm(self, w: torch.Tensor) -> torch.Tensor:
        if w.ndim not in (3, 4):
            raise ValueError(f"SpectralConv.norm expects 3D/4D tensor, got {tuple(w.shape)}")

        out_ch, in_ch = w.shape[0], w.shape[1]
        w_flat = w.reshape(out_ch, -1)

        # Match your lmo scaling exactly:
        if w.ndim == 3:
            k = w.shape[2]
            scale = math.sqrt(out_ch / in_ch) / k
        else:
            k = w.shape[2]   # matches your lmo's `k, _ = g.shape`
            scale = math.sqrt(out_ch / in_ch) / (k ** 2)

        sigma = torch.linalg.matrix_norm(w_flat.float(), ord=2)
        return (sigma / scale).to(dtype=w.dtype)


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = g.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        g *= scale

        return g

    def init(self, w):
        w_fp = w.data.double()
        torch.nn.init.orthogonal_(w_fp)
        d_out, d_in = w_fp.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        w_fp.mul_(scale)
    
        w.data = w_fp.to(dtype=w.data.dtype)
        return w

    def norm(self, w: torch.Tensor) -> torch.Tensor:
        if w.ndim != 2:
            raise ValueError(f"Spectral.norm expects 2D tensor, got {tuple(w.shape)}")

        d_out, d_in = w.shape
        if self.normalized:
            scale = (d_out / d_in) ** 0.5
        else:
            scale = d_out ** 0.5
        if self.max:
            scale = max(1.0, scale)

        # Exact spectral norm (can be expensive); cast to float for stability.
        sigma = torch.linalg.matrix_norm(w.float(), ord=2)
        return (sigma / scale).to(dtype=w.dtype)

class Sign(Norm):
    def __init__(self, zero_init: bool = False, normalized: bool = True):
        self.zero_init = zero_init
        self.normalized = normalized

    @staticmethod
    def _fan_in(t: torch.Tensor) -> int:
        # 0D/1D: fan_in = numel; >=2D: fan_in = product of dims 1:
        if t.ndim <= 1:
            return max(int(t.numel()), 1)
        return max(int(t.numel() // t.shape[0]), 1)

    def lmo(self, g: torch.Tensor) -> torch.Tensor:
        fan_in = self._fan_in(g)
        if self.normalized:
            return torch.sign(g) * (1.0 / fan_in)
        return torch.sign(g)

    def init(self, w: torch.Tensor) -> torch.Tensor:
        if self.zero_init:
            torch.nn.init.zeros_(w)
            return w

        w.data = (torch.randint(0, 2, w.shape, device=w.device) * 2 - 1).to(dtype=w.dtype)
        if self.normalized:
            w.data.mul_(1.0 / self._fan_in(w))
        return w
        
    def norm(self, w: torch.Tensor) -> torch.Tensor:
        max_abs = w.abs().max()
        if self.normalized:
            max_abs = max_abs * self._fan_in(w)
        return max_abs


class Auto(Norm):
    def lmo(self, g):
        if g.ndim in [3,4]:
            return SpectralConv().lmo(g)
        elif g.ndim == 2:
            return Spectral().lmo(g)
        elif g.ndim in [0,1]:
            return BiasRMS().lmo(g)

    def init(self, w):
        if w.ndim in [3,4]:
            return SpectralConv().init(w)
        elif w.ndim == 2:
            return Spectral().init(w)
        elif w.ndim in [0,1]:
            return BiasRMS().init(w)
            
    def norm(self, w: torch.Tensor) -> torch.Tensor:
        if w.ndim in (3, 4):
            return SpectralConv().norm(w)
        if w.ndim == 2:
            return Spectral().norm(w)
        if w.ndim in (0, 1):
            return BiasRMS().norm(w)
        raise ValueError(f"Auto.norm: unsupported tensor rank {w.ndim} for shape {tuple(w.shape)}")


norm_dict = {
    'ColNorm': ColNorm,
    'RowNorm': RowNorm,
    'BiasRMS': BiasRMS,
    'SpectralConv': SpectralConv,
    'Spectral': Spectral,
    'Sign': Sign,
    'Auto': Auto,
}


class ScionTrace(torch.optim.Optimizer):
    """
        Scion optimizer implementation.
    """
    def __init__(
            self, params, lr=1e-3, momentum=1.0, norm: str='Spectral', 
            norm_kwargs: dict=None, scale=1.0, unconstrained=False, weight_decay=0.0
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        self.trace_stats_every = 200
        defaults = dict(lr=lr, momentum=momentum, scale=scale, 
                        unconstrained=unconstrained, weight_decay=weight_decay)
        super().__init__(params, defaults)

        
    @torch.no_grad()
    def track_stats(self, cur_loss=None, store_dtype: torch.dtype = torch.bfloat16, store_on_cpu: bool = False):
        """Store the current stochastic gradient (one sample) inside the optimizer.
    
        This clones every p.grad (after grad-accumulation) so Engine can safely call
        optimizer.zero_grad(set_to_none=True) afterwards.
        """
        if not hasattr(self, "_gn_samples"):
            self._gn_samples = []          # List[List[Tensor|None]]; each sample aligned with param list
            self._gn_losses = []           # Optional loss values for the sampled batches
            self._gn_last_stats = None     # Dict[str, float] produced by report_stats()
    
        sample = []
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    sample.append(None)
                    continue
                gg = g.detach()
                if store_on_cpu:
                    gg = gg.to(device="cpu", dtype=store_dtype)
                else:
                    gg = gg.to(dtype=store_dtype)
                sample.append(gg.clone())
        
        self._gn_samples.append(sample)
        if cur_loss is not None:
            try:
                self._gn_losses.append(float(cur_loss))
            except Exception:
                pass
                
    @torch.no_grad()
    def report_stats(self, eps: float = 1e-12) -> Optional[Dict[str, float]]:
        """Compute gradient-noise statistics over the samples collected via track_stats().

        This function is DDP-safe: it all-reduces scalars (not full gradients).
        """
        if not hasattr(self, "_gn_samples") or len(self._gn_samples) == 0:
            return None

        samples = self._gn_samples
        m = len(samples)  # number of samples on this rank
        params = [p for group in self.param_groups for p in group["params"]]

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # ---------------------------
        # E ||g||_F^2
        # ---------------------------
        sum_g2 = 0.0
        for s in samples:
            for g in s:
                if g is None:
                    continue
                gf = g.to(dtype=torch.float32)
                sum_g2 += float(torch.sum(gf * gf).item())
        Eg2_local = sum_g2 / max(m, 1)

        # ---------------------------
        # || E[g] ||_F^2
        # ---------------------------
        mean_g2_local = 0.0
        for i in range(len(params)):
            acc = None
            for s in samples:
                g = s[i]
                if g is None:
                    continue
                gf = g.to(dtype=torch.float32)
                if acc is None:
                    acc = gf.clone()
                else:
                    acc.add_(gf)
            if acc is not None:
                mean_g2_local += float(torch.sum(acc * acc).item())
        mean_g2_local = mean_g2_local / (m * m)

        # ---------------------------
        # New: ||G_t - g_t||_* / ||G_t - g_t||_F
        # where G_t is the averaged gradient over the m samples, and g_t is one sampled mini-batch
        # gradient (we use the first sample).
        # ---------------------------
        # Build norm backends aligned with flattened parameter order (same order as track_stats()).
        norm_backends = []
        for group in self.param_groups:
            norm_name = group.get("norm", "Auto")
            norm_kwargs = group.get("norm_kwargs", {}) or {}
            nb = norm_dict[norm_name](**norm_kwargs)
            for _ in group["params"]:
                norm_backends.append(nb)

        first = samples[0]
        delta_star_local = 0.0
        delta_fro_sq_local = 0.0

        for i in range(len(params)):
            # Sum gradients for this parameter across samples (treat None as zero)
            acc = None
            for s in samples:
                g = s[i]
                if g is None:
                    continue
                gf = g.to(dtype=torch.float32)
                if acc is None:
                    acc = gf.clone()
                else:
                    acc.add_(gf)

            if acc is None:
                continue  # no gradients for this parameter in any sample

            mean_g = acc / float(m)

            g0 = first[i]
            if g0 is None:
                g0f = torch.zeros_like(mean_g)
            else:
                g0f = g0.to(dtype=torch.float32)

            delta = mean_g - g0f

            # ||Δ||_F^2 accumulation (layer-wise)
            delta_fro_sq_local += float(torch.sum(delta * delta).item())

            # ||Δ||_* accumulation (layer-wise):
            #   ||Δ||_* += <Δ, lmo(Δ)>
            # The LMO for some norms (e.g., Spectral) can be much faster on GPU; if samples are stored
            # on CPU but CUDA is available, move Δ to CUDA just for the LMO computation.
            delta_for_lmo = delta
            if device.type == "cuda" and delta_for_lmo.device.type != "cuda":
                # Heuristic: move only when CUDA is available; keeps Sign/Row/Col/etc. on CPU if desired.
                # (Still correct either way, just potentially slower on CPU for Spectral-based LMOs.)
                delta_for_lmo = delta_for_lmo.to(device=device)

            lmo_delta = norm_backends[i].lmo(delta_for_lmo)
            # Ensure float32 for stable dot product
            lmo_delta = lmo_delta.to(dtype=torch.float32)
            delta_star_local += float(torch.sum(delta_for_lmo.to(dtype=torch.float32) * lmo_delta).item())

        # ---------------------------
        # DDP synchronization of scalars
        # ---------------------------
        if dist.is_initialized():
            Eg2_tensor = torch.tensor(Eg2_local, device=device)
            mean_g2_tensor = torch.tensor(mean_g2_local, device=device)
            delta_star_tensor = torch.tensor(delta_star_local, device=device)
            delta_fro_sq_tensor = torch.tensor(delta_fro_sq_local, device=device)

            dist.all_reduce(Eg2_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mean_g2_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(delta_star_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(delta_fro_sq_tensor, op=dist.ReduceOp.SUM)

            world_size = dist.get_world_size()
            Eg2 = Eg2_tensor.item() / world_size
            mean_g2 = mean_g2_tensor.item() / world_size
            delta_star = delta_star_tensor.item() / world_size
            delta_fro_sq = delta_fro_sq_tensor.item() / world_size
        else:
            Eg2 = Eg2_local
            mean_g2 = mean_g2_local
            delta_star = delta_star_local
            delta_fro_sq = delta_fro_sq_local

        # Compute variance and SNR
        var_biased = max(Eg2 - mean_g2, 0.0)
        var_unbiased = var_biased * (m / (m - 1)) if m > 1 else 0.0

        mean_norm = math.sqrt(max(mean_g2, 0.0))
        std = math.sqrt(max(var_unbiased, 0.0))
        snr = mean_norm / (std + eps)

        delta_fro = math.sqrt(max(delta_fro_sq, 0.0))
        delta_star_over_fro = float(delta_star / (delta_fro + eps))

        stats: Dict[str, float] = {
            "grad/noise_samples": float(m),
            "grad/noise_E_grad_norm2": float(Eg2),
            "grad/noise_mean_grad_norm2": float(mean_g2),
            "grad/noise_sigma2": float(var_unbiased),
            "grad/noise_sigma": float(std),
            "grad/noise_snr": float(snr),
            "rho/delta_fro": float(delta_fro),
            "rho/delta_star": float(delta_star),
            "rho/noise_delta_star_over_fro": float(delta_star_over_fro),
        }

        if hasattr(self, "_gn_losses") and len(self._gn_losses) == m:
            lmean = float(sum(self._gn_losses) / m)
            lvar = float(sum((x - lmean) ** 2 for x in self._gn_losses) / max(m - 1, 1))
            stats["grad/noise_loss_mean"] = lmean
            stats["grad/noise_loss_var"] = lvar

        # store for logging, then clear stored grads
        self._gn_last_stats = stats
        self._gn_samples = []
        self._gn_losses = []

        return stats


    @torch.no_grad()
    def track_rhos(self) -> Dict[str, float]:
        """Track rho-style ratios for parameters (and momentum buffers when present).

        Returns a dict of scalars. This function is DDP-safe (reduces scalars).
        """
        rhos: Dict[str, float] = {}

        total_fro_sq = 0.0
        total_nuc = 0.0
        total_mom_fro_sq = 0.0
        total_mom_nuc = 0.0

        for group in self.param_groups:
            norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            momentum = group.get("momentum", 1.0)

            for p in group["params"]:
                if p is None:
                    continue

                # Parameter contribution
                total_nuc += float(torch.sum(p * norm_backend.lmo(p)).item())
                total_fro_sq += float(torch.sum(p * p).item())

                # Momentum buffer contribution (if momentum buffers exist)
                if momentum != 1.0:
                    state = self.state.get(p, None)
                    if state is None:
                        continue
                    buf = state.get("momentum_buffer", None)
                    if buf is None:
                        continue
                    total_mom_nuc += float(torch.sum(buf * norm_backend.lmo(buf)).item())
                    total_mom_fro_sq += float(torch.sum(buf * buf).item())

        # DDP-safe reductions
        if dist.is_initialized():
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            total_nuc_tensor = torch.tensor([total_nuc], device=device, dtype=torch.float32)
            total_fro_sq_tensor = torch.tensor([total_fro_sq], device=device, dtype=torch.float32)
            total_mom_nuc_tensor = torch.tensor([total_mom_nuc], device=device, dtype=torch.float32)
            total_mom_fro_sq_tensor = torch.tensor([total_mom_fro_sq], device=device, dtype=torch.float32)

            dist.all_reduce(total_nuc_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_fro_sq_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_mom_nuc_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_mom_fro_sq_tensor, op=dist.ReduceOp.SUM)

            total_nuc = float(total_nuc_tensor.item())
            total_fro_sq = float(total_fro_sq_tensor.item())
            total_mom_nuc = float(total_mom_nuc_tensor.item())
            total_mom_fro_sq = float(total_mom_fro_sq_tensor.item())

        rhos["rho/total"] = float(total_nuc / math.sqrt(total_fro_sq + 1e-12))
        rhos["rho/total_nuc"] = float(total_nuc)
        rhos["rho/total_fro"] = float(math.sqrt(total_fro_sq + 1e-12))
        rhos["rho/total_mom_nuc"] = float(total_mom_nuc)
        rhos["rho/total_mom_fro"] = float(math.sqrt(total_mom_fro_sq + 1e-12))

        return rhos


    @torch.no_grad()
    def step(self, step, loss, trace=False):
        # 1. Initialize
        num = den = grad_norm = dual_grad_norm = num_nuc = den_spec = 0.0
        if step == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state['prev_p'] = p.detach().clone()
                    state['prev_grad'] = p.grad.detach().clone()

        # 3. Update Params
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            weight_decay = group.get('weight_decay', 0.0)
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                # Keep raw grads and params
                raw_p = p.detach().clone()
                raw_g = g.detach().clone()

                # Apply momentum
                if momentum != 1:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                # Compute update
                LMO = norm_backend.lmo(g)
                update = scale * LMO

                # Apply weight decay & update
                #if weight_decay != 0:
                #    p.data.mul_(1 - lr * weight_decay)
                if unconstrained:
                    p.data.add_(update, alpha=-lr)
                else:
                    p.data.mul_(1 - lr).add_(update, alpha=-lr)

                # 4. Step tracing
                if trace:
                    grad_diff = state['prev_grad'].detach() - raw_g
                    num += grad_diff.pow(2).sum().item()

                    temp = grad_diff.clone()
                    num_nuc += (temp.mul_(norm_backend.lmo(grad_diff))).sum().item()

                    grad_norm += raw_g.pow(2).sum().item()
                    state['prev_grad'].copy_(raw_g)

                    delta_p = raw_p.detach() - state['prev_p']
                    den += delta_p.pow(2).sum().item()
                    den_spec = max(den_spec, norm_backend.norm(delta_p))
                    state['prev_p'].copy_(raw_p)

                    dual_grad_norm += (raw_g * LMO).sum().item()

        if trace:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            # Convert scalars to tensors for DDP-safe reductions
            num_tensor = torch.tensor([num], device=device, dtype=torch.float32)
            den_tensor = torch.tensor([den], device=device, dtype=torch.float32)
            num_nuc_tensor = torch.tensor([num_nuc], device=device, dtype=torch.float32)
            den_spec_tensor = torch.tensor([den_spec], device=device, dtype=torch.float32)
            grad_norm_tensor = torch.tensor([grad_norm], device=device, dtype=torch.float32)
            dual_grad_norm_tensor = torch.tensor([dual_grad_norm], device=device, dtype=torch.float32)

            if dist.is_initialized():
                dist.all_reduce(num_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(den_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_nuc_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(den_spec_tensor, op=dist.ReduceOp.MAX)  # max over GPUs
                dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(dual_grad_norm_tensor, op=dist.ReduceOp.SUM)

            # Convert back to floats
            num = float(torch.sqrt(num_tensor))
            den = float(torch.sqrt(den_tensor))
            num_nuc = float(num_nuc_tensor)
            den_spec = float(den_spec_tensor)
            grad_norm = float(grad_norm_tensor)
            dual_grad_norm = float(dual_grad_norm_tensor)

            local_smoothness_fro = num / (den + 1e-8)
            local_smoothness_spec = num_nuc / (den_spec + 1e-8)

            # Track rhos (already DDP-safe in previous fix)
            rhos = self.track_rhos()

            return {
                "stats/grad_norm_fro_power_1": grad_norm,
                "stats/grad_norm_nuc_power_1": dual_grad_norm,
                "stats/num_fro": num,
                "stats/den_fro": den,
                "stats/num_nuc": num_nuc,
                "stats/den_spec": den_spec,
                "stats/local_smooth_fro": local_smoothness_fro,
                "stats/local_smooth_spec": local_smoothness_spec,
                **rhos,
            }

        return {}

    def init(self):
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            init_func = norm_backend.init
            scale = group['scale']
            for p in group['params']:
                init_func(p)
                p.data *= scale
        


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    From: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt
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


def zeroth_power_via_svd(G):
   U, S, V = G.svd()
   return U @ V.T
