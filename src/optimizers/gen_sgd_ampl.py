import torch
from torch.optim.optimizer import Optimizer
from src.aggregators import CM, Mean, NNM


class SGDGen(Optimizer):
    r"""
    Generalized SGD with (optional) clipping, DP noise, error feedback variants,
    and robust aggregation across multiple workers.

    Memory-safety improvements vs. original:
      - Do not store unbounded per-step tensors (removed 'raw_grads').
      - 'updates' (per-round storage) is cleared right after aggregation.
      - Minimize clones and use in-place ops where appropriate.
    """

    def __init__(
        self,
        params,
        lr,
        n_workers,
        n_byzant_workers,
        attack,
        batch_size,
        momentum=0.0,
        beta=1.0,
        dampening=0.0,
        tau=None,                 # clipping radius (required)
        weight_decay=0.0,
        nesterov=False,
        comp=None,                # kept for API compatibility (unused here)
        master_comp=None,         # kept for API compatibility (unused here)
        DP=None,                  # Differential Privacy: bool
        noise=None,               # DP noise std (sigma)
        error_feedback="None",    # None | "Safe-DSHB" | "EF21M"
        device="cuda:0",
        normalize=False,
        robust_aggregator=None,   # None | "CWMedian" | "NNM"  (default: Mean)
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if beta < 0.0:
            raise ValueError(f"Invalid heavy-ball value beta: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta=beta,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            batch_size=batch_size,
            tau=tau,
            noise=noise,
            DP=DP,
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and zero dampening")

        super().__init__(params, defaults)

        if tau is None:
            raise ValueError("Clipping radius 'tau' can't be None")

        if DP and noise is None:
            raise ValueError("For DP, noise (std) must be provided")

        self.tau = tau
        self.noise = noise
        self.device = device
        self.DP = bool(DP)
        self.normalize = normalize

        self.error_feedback = None if error_feedback == "None" else error_feedback

        self.n_workers = int(n_workers)
        self.n_byzant_workers = int(n_byzant_workers)
        self.attack = attack

        # counts steps within a round of n_workers contributions
        self.grads_received = 0

        # Select robust aggregator
        if robust_aggregator is None:
            self.robust_aggregator = Mean()
        elif robust_aggregator == "CWMedian":
            self.robust_aggregator = CM()
        elif robust_aggregator == "NNM":
            self.robust_aggregator = NNM(f=self.n_byzant_workers)
        else:
            raise ValueError(f"Unknown robust aggregator: {robust_aggregator}")

        # Debug banner (optional)
        print(self.error_feedback, self.tau, self.DP, self.noise)
        for group in self.param_groups:
            print("mom:", group["momentum"], "beta:", group["beta"], "lr:", group["lr"], "tau:", self.tau)

    def __setstate__(self, state):
        print("WARNING OPTIMIZER __setstate__ (loading state)")
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    # ------------ helpers ------------

    @torch.no_grad()
    def _add_dp_noise_(self, update: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Optionally add DP Gaussian noise (in-place friendly)."""
        if not self.DP:
            return update
        # Allocate a noise tensor, then add; letting PyTorch reuse allocator.
        noise = self.noise * torch.randn_like(update, device=update.device)
        if scale != 1.0:
            noise.mul_(scale)
        return update.add_(noise)

    # Note: This function updates EF21M's v buffer in-place (original behavior),
    # because compute_clip_norm previously had side-effects in your code.
    @torch.no_grad()
    def compute_clip_norm(self, w_id: int) -> torch.Tensor:
        """Compute and return sqrt(sum ||·||^2) (later square-rooted)."""
        grad_norm_sq = torch.tensor(0.0, device=self.device)
        for group in self.param_groups:
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                d_p = p.grad.detach()

                if self.error_feedback is None or self.error_feedback == "Safe-DSHB":
                    grad_norm_sq += d_p.pow(2).sum() #torch.sum(d_p * d_p)

                elif self.error_feedback == "EF21M":
                    error_name_g = f"error_g_{w_id}"
                    error_name_v = f"error_v_{w_id}"

                    if error_name_g not in param_state:
                        # momentum^2 * ||g||^2
                        grad_norm_sq += (momentum * momentum) * d_p.pow(2).sum() #torch.sum(d_p * d_p)
                    else:
                        # v <- (1-m)*v + m*g  (create once, then in-place update)
                        if error_name_v not in param_state:
                            param_state[error_name_v] = torch.zeros_like(d_p, device=self.device)
                        v = param_state[error_name_v]
                        v.mul_(1 - momentum).add_(momentum, d_p)
                        #grad_norm_sq += torch.sum((v - param_state[error_name_g]) ** 2)
                        grad_norm_sq += (v - param_state[error_name_g]).pow(2).sum()
        return grad_norm_sq


    @torch.no_grad()
    def clip_indiv_grad(self, w_id: int) -> None:
        """
        Per-example L2 clipping to radius self.tau, then average across the batch
        and overwrite p.grad with the averaged, clipped gradient.
        Expects self.state[p][f'{w_id}_{i}_indiv_grad'] to exist for i=0..bs-1.
        """
        for group in self.param_groups:
            bs = group["batch_size"]

            # 1) Compute per-example global norms: ||g^(i)||^2 = sum_p sum(g_p^(i)^2)
            # Start with zeros on the right device/dtype
            per_ex_sqnorm = torch.zeros(bs, device=self.device, dtype=torch.float32) # tensor of gradient norms of size (bs,)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Accumulate squared norms across parameters
                # Each st[f'{w_id}_{i}_indiv_grad'] has same dtype/device as p
                for i in range(bs):
                    gi = state[f'{w_id}_{i}_indiv_grad']
                    per_ex_sqnorm[i] += (gi * gi).sum()

            # 2) Compute clipping coefficients (vectorized)
            per_ex_norm = per_ex_sqnorm.sqrt().clamp_min(1e-10)
            clip_coef = (self.tau / per_ex_norm).clamp(max=1.0)  # shape: (bs,)

            # 3) Scale each per-parameter, per-example gradient in-place
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                for i in range(bs):
                    state[f'{w_id}_{i}_indiv_grad'].mul_(clip_coef[i])

            # 4) Average across examples and overwrite p.grad
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                avg = torch.zeros_like(p)
                for i in range(bs):
                    avg.add_(state[f'{w_id}_{i}_indiv_grad'], alpha=1.0/bs)
                    state.pop(f'{w_id}_{i}_indiv_grad', None)

                if self.DP:
                    avg_clip_grad = self._add_dp_noise_(avg_clip_grad, scale=1.0)

                p.grad.detach().copy_(avg_clip_grad)

    # ------------ main step ------------

    @torch.no_grad()
    def step_local_global(self, w_id: int, closure=None):
        """Collect one worker's gradients, apply robust aggregation when all workers have contributed."""
        loss = None
        if closure is not None:
            loss = closure()

        self.grads_received += 1
        idx = self.grads_received - 1  # index into per-round updates

        for group in self.param_groups:
            momentum = group["momentum"]
            beta = group["beta"]
            lr = group["lr"]

            # compute clipping coefficient
            #clip_norm = torch.sqrt(self.compute_clip_norm(w_id)) + 1e-10
            #clip_coef = min(1.0, float(self.tau) / float(clip_norm))
            

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                g = p.grad.detach()  # current grad (no clone; we're in no_grad), this is overwritten averaged clipped gradient with DP noise


                # --- compute per-worker 'update' according to error feedback ---
                if self.error_feedback is None:
                    # plain clipped gradient
                    # we add DP noise to the averaged clipped gradient
                    update = g # tilde{g}_i^t

                elif self.error_feedback == "Safe-DSHB":
                    name_g = f"error_g_{w_id}"
                    if name_g not in param_state:
                        # m_i^0 = momentum * g
                        param_state[name_g] = g.mul(momentum)
                    else:
                        # m <- (1-m)*m + m*g
                        m = param_state[name_g]
                        m.mul_(1 - momentum).add_(momentum, g)
                    update = param_state[name_g]

                elif self.error_feedback == "EF21M":
                    clip_norm = torch.sqrt(self.compute_clip_norm(w_id)) + 1e-10
                    clip_coef = min(1.0, float(self.tau) / float(clip_norm))

                    name_g = f"error_g_{w_id}"
                    name_v = f"error_v_{w_id}"

                    # init persistent buffers once
                    if name_v not in param_state:
                        param_state[name_v] = torch.zeros_like(g, device=self.device)

                    if name_g not in param_state:
                        # g0 = beta * clip(momentum * grad)
                        update = g.mul(momentum * clip_coef * beta)
                        # store a persistent copy for future deltas
                        param_state[name_g] = update.clone()
                    else:
                        v = param_state[name_v]
                        # v <- (1-m)*v + m*grad
                        v.mul_(1 - momentum).add_(momentum, g)
                        # delta = beta * clip(v - g_prev)
                        update = (v - param_state[name_g]).mul(clip_coef * beta)
                        # g_prev <- g_prev + delta
                        param_state[name_g].add_(update)

                else:
                    raise ValueError(f"Unknown error_feedback mode: {self.error_feedback}")

                # --- store update into per-round slot for robust aggregation ---
                if 'updates' not in param_state:
                    param_state['updates'] = [0] * (self.n_workers + self.n_byzant_workers)
                    param_state['updates'][self.grads_received - 1] = update
                else:
                    if self.error_feedback == 'EF21M':
                        param_state['updates'][self.grads_received - 1] += update
                    else:
                        param_state['updates'][self.grads_received - 1] = update

                param_state["updates"][idx] = update

                # When all workers have contributed: aggregate and step
                if self.grads_received == self.n_workers:

                    corrupted_grad = self.attack(param_state['updates'][:self.n_workers])
                    for i in range(self.n_byzant_workers):
                        param_state['updates'][self.n_workers + i] = corrupted_grad # as byzant can devide on betahat and kill momentum

                    full_grad = self.robust_aggregator(param_state["updates"])  # tensor same shape as p
                    # (Optional) zero out stored full_grad for EF usage; not needed here
                    # Apply update: p <- p - lr * full_grad
                    p.add_(full_grad, alpha=-lr)

                    # FREE per-round storage to release memory immediately
                    param_state.pop("updates", None)
                    # If you had set/used 'full_grad' persistently, clear it:
                    # param_state.pop("full_grad", None)

        # reset round counter
        if self.grads_received == self.n_workers:
            self.grads_received = 0

        return loss
