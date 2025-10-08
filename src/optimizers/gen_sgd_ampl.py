import torch
from torch.optim.optimizer import Optimizer
from src.aggregators import CM, Mean, NNM

class SGDGen(Optimizer):
    r"""
        based on torch.optim.SGD implementation
    """

    def __init__(self, 
                 params, 
                 lr, 
                 n_workers,
                 n_byzant_workers,
                 attack,
                 batch_size,
                 momentum=0,
                 beta=1, 
                 dampening=0, 
                 inner_tau=None,      #Clipping every sample
                 outer_tau=None,      #Clipping ef
                 weight_decay=0, 
                 nesterov=False, 
                 comp=None, 
                 master_comp=None, 
                 DP=None,   #Differencial Privacy True/False parameter 
                 noise=None,  #Differential Privacy 
                 error_feedback='None',  
                 device='cuda:0', 
                 normalize=False,
                 robust_aggregator=None,
                 ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if beta < 0.0:
            raise ValueError("Invalid heavy-ball value: {}".format(beta))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, beta=beta, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        inner_tau=inner_tau, outer_tau=outer_tau, noise=noise, DP=DP, batch_size=batch_size)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDGen, self).__init__(params, defaults)

        self.inner_tau = inner_tau
        self.outer_tau = outer_tau
        self.noise = noise
        self.device = device
        self.DP = DP
        self.normalize = normalize
        if error_feedback == 'None':
            self.error_feedback = None
        else:
            self.error_feedback = error_feedback
                
        if self.inner_tau is None:
            raise ValueError("Clipping radius can't be None")
            
        if self.outer_tau is None:
            raise ValueError("Clipping radius can't be None")

        if self.DP and self.noise is None:
            raise ValueError("For DP noise variance can't be None")

        self.n_workers = n_workers
        self.n_byzant_workers = n_byzant_workers
        self.attack = attack

        self.grads_received = 0
        self.n_iters = 0

        if robust_aggregator is None:
            self.robust_aggregator = Mean()
        elif robust_aggregator == "CWMedian":
            self.robust_aggregator = CM()
        elif robust_aggregator == "NNM":
            self.robust_aggregator = NNM(f=self.n_byzant_workers)
        else:
            raise ValueError("Unknown robbust aggregator")    
        
        
        for group in self.param_groups:
            momentum = group['momentum']
            beta = group['beta']
            lr = group['lr']

    def __setstate__(self, state):
        print("WARNING OPTIMZER SETSTATE")
        super(SGDGen, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def _add_dp_noise_(self, update: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Optionally add DP Gaussian noise (in-place friendly)."""
        """
            scale = beta_hat
        """
        if not self.DP:
            return update
        # Allocate a noise tensor, then add; letting PyTorch reuse allocator.
        noise = self.noise * torch.randn_like(update, device=update.device)
        if scale != 1.0:
            noise.mul_(scale)
        return update.add_(noise)

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

                #print("CLIP IND GRAD", self.state[p].keys())

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
            clip_coef = (self.inner_tau / per_ex_norm).clamp(max=1.0)  # shape: (bs,)

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
                    avg = self._add_dp_noise_(avg, scale=1.0)

                p.grad.detach().copy_(avg)
    
       
    @torch.no_grad()
    def compute_clip_norm(self, w_id):

        assert self.error_feedback == "EF21M"

        """Computes and returns *squared* gradient norm."""
        grad_norm_sq = 0.  # we assume all params are on the same device
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                
                if p.grad is None:
                    continue
                    
                param_state = self.state[p]

                d_p = p.grad.data.clone()
                if self.error_feedback == "EF21M":
                    error_name_g = 'error_g_' + str(w_id)
                    error_name_v = 'error_v_' + str(w_id)
                    
                    if error_name_g not in param_state:
                        grad_norm_sq += momentum**2*torch.sum(d_p**2)
                    else:
                        ## v_i^{t+1} = (1-momentum)*v_i^t + momentum * nabla f_i(x^{t+1})
                        param_state[error_name_v] = (1-momentum)*param_state[error_name_v] + momentum*d_p
                        ## ||v_i^{t+1} - g_i^t||^2
                        grad_norm_sq += torch.sum((param_state[error_name_v] - param_state[error_name_g])**2)
                    
        return grad_norm_sq


    @torch.no_grad()
    def step_local_global(self, w_id, closure=None):
        """Performs a single optimization step.

        Arguments:
            w_id: integer, id of the worker
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.grads_received += 1

        for group in self.param_groups:
            momentum = group['momentum']
            beta = group['beta']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                d_p = p.grad.data.clone()


                if self.error_feedback == None:
                    update = d_p

                elif self.error_feedback == "Safe-DSHB":

                    error_name_g = 'error_g_' + str(w_id)

                    if error_name_g not in param_state:
                        param_state[error_name_g] = momentum*d_p.clone() 
                        update = param_state[error_name_g] # compute m_i^0 = momentum*tilde{g}_i^t
                    else:
                        param_state[error_name_g] = (1-momentum)*param_state[error_name_g] + momentum*d_p.clone() # compute m_i^t = (1-momentum)*m_i^{t-1} + momentum*tilde{g}_i^t
                        update = param_state[error_name_g]

                elif self.error_feedback == "EF21M":

                    clip_norm = torch.sqrt(self.compute_clip_norm(w_id)) + 1e-10
                    clip_coef = min(1.0, self.outer_tau / clip_norm)

                    error_name_g = 'error_g_' + str(w_id)
                    error_name_v = 'error_v_' + str(w_id)
                    
                    if error_name_v not in param_state:
                        ## v_i^0 = momentum * nabla f_i(x^0)
                        param_state[error_name_v] = momentum*d_p.clone()

                    if error_name_g not in param_state:
                        ## d_p = clip_tau(momentum * nabla f_i(x^0)) = g_i^0

                        d_p = beta * clip_coef * (momentum * d_p) 
                        param_state[error_name_g] = d_p
                        update = d_p
                    
                    else:
                        ## g_i^{t+1} += clip_tau(v_i^{t+1} - g_i^t)
                        update = beta * clip_coef * (param_state[error_name_v] - param_state[error_name_g]) 
                        param_state[error_name_g] += update

                if 'updates' not in param_state:
                    param_state['updates'] = [0] * (self.n_workers + self.n_byzant_workers)
                    param_state['updates'][self.grads_received - 1] = update
                else:
                    if self.error_feedback == 'EF21M':
                        param_state['updates'][self.grads_received - 1] += update
                    else:
                        param_state['updates'][self.grads_received - 1] = update

                if self.grads_received == self.n_workers:
                    # COMPARE AVG VS NNM + CWM
                    # BITFLIPPING

                    #print(len(param_state['updates']))
                    #orig_mean = torch.stack(param_state['updates'], 1).mean()
                    #print(orig_mean)

                    corrupted_grad = self.attack(param_state['updates'][:self.n_workers])
                    for i in range(self.n_byzant_workers):
                        param_state['updates'][self.n_workers + i] = corrupted_grad # as byzant can devide on betahat and kill momentum

                    # print([el.mean() for el in param_state['updates']])

                    #print(len(param_state['updates']))
                    #print(torch.stack(param_state['updates'], 1).mean())
                    #print((orig_mean * self.n_workers - orig_mean * self.n_byzant_workers) / (self.n_workers + self.n_byzant_workers))

                    grad = self.robust_aggregator(param_state['updates'])
                    
                    p.copy_(p - lr*grad)
        

        if self.grads_received == self.n_workers:
            self.grads_received = 0

        return loss
