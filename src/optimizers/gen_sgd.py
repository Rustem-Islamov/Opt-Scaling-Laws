import torch
from torch.optim.optimizer import Optimizer


class SGDGen(Optimizer):
    r"""
        based on torch.optim.SGD implementation
    """

    def __init__(self, params, lr, n_workers, momentum=0, beta=1, dampening=0, tau=None, noise=None,
                 weight_decay=0, nesterov=False, comp=None, master_comp=None, DP=None,
                 error_feedback=False, device='cuda:0', normalize=False):
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
                        tau=tau, noise=noise, DP=DP)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDGen, self).__init__(params, defaults)

        self.tau = tau
        self.noise = noise
        self.device = device
        self.DP = DP
        self.normalize = normalize
        if error_feedback == 'None':
            self.error_feedback = None
        else:
            self.error_feedback = error_feedback
        
        print(self.error_feedback, self.tau, self.DP, self.noise)
        
        if self.tau is None:
            raise ValueError("Clipping radius can't be None")
            
        if self.DP and self.noise is None:
            raise ValueError("For DP noise variance can't be None")

        self.n_workers = n_workers
        self.grads_received = 0
        self.n_iters = 0
        
        for group in self.param_groups:
            momentum = group['momentum']
            beta = group['beta']
            lr = group['lr']
            print('mom:', momentum, 'beta:', beta, 'lr:', lr, 'tau:', self.tau)

    def __setstate__(self, state):
        super(SGDGen, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    @torch.no_grad()
    def compute_update_norm(self):
        """Computes and returns *squared* update norm."""
        upd_norm_sq = 0.  # we assume all params are on the same device
        for group in self.param_groups:
            for p in group['params']:
                
                if p.grad is None:
                    continue
                    
                param_state = self.state[p]

                if self.error_feedback == "ANorm":
                    upd_norm_sq += torch.sum(param_state['full_grad']**2)

                        
        return upd_norm_sq
    
       
    @torch.no_grad()
    def compute_clip_norm(self, w_id):
        """Computes and returns *squared* gradient norm."""
        grad_norm_sq = 0.  # we assume all params are on the same device
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                
                if p.grad is None:
                    continue
                    
                param_state = self.state[p]

                d_p = p.grad.data.clone()
                    
                if self.error_feedback == None:
                    grad_norm_sq += torch.sum(d_p**2)

                if self.error_feedback == "ClipSGDM":
                    error_name = 'mom_' + str(w_id)
                    if error_name not in param_state:
                        grad_norm_sq += momentum**2*torch.sum(d_p**2)
                    else:
                        param_state[error_name] = (1-momentum)*param_state[error_name_v] + momentum*d_p
                        grad_norm_sq += torch.sum(param_state[error_name]**2)

                if self.error_feedback == "EF21":
                    error_name = 'error_g_' + str(w_id)
                    if error_name not in param_state:
                        grad_norm_sq += torch.sum(d_p**2)
                    else:
                        grad_norm_sq += torch.sum((d_p - param_state[error_name])**2)


                if self.error_feedback == "ANorm":
                    error_name = 'error_g_' + str(w_id)
                    if error_name not in param_state:
                        grad_norm_sq += torch.sum(d_p**2)
                    else:
                        grad_norm_sq += torch.sum((d_p - param_state[error_name])**2)
                
                        
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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            beta = group['beta']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            clip_norm = torch.sqrt(self.compute_clip_norm(w_id)) + 1e-10

            if self.error_feedback == 'ANorm':
                clip_coef = 1/(self.tau + clip_norm)
            else:
                clip_coef = min(1.0, self.tau / clip_norm)

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                d_p = p.grad.data.clone()
                
                if self.error_feedback == None:

                    d_p = d_p * clip_coef
                    update = d_p 

                elif self.error_feedback == "ClipSGDM":
                    error_name = 'mom_' + str(w_id)
                    if error_name not in param_state:
                        d_p = d_p * clip_coef
                        update = d_p
                    else:
                        update = clip_coef * param_state[error_name]
                        

                elif self.error_feedback == "EF21":

                    error_name = 'error_g_' + str(w_id)
                    if error_name not in param_state:
                        d_p = d_p * clip_coef
                        param_state[error_name] = d_p
                        update = d_p
                        
                    else:
                        update = clip_coef * (d_p - param_state[error_name])
                        param_state[error_name] += update

                elif self.error_feedback == "ANorm":

                    error_name = 'error_g_' + str(w_id)
                    if error_name not in param_state:
                        d_p = beta * clip_coef * d_p
                        param_state[error_name] = d_p
                        update = d_p
                    else:
                        update = beta * clip_coef * (d_p - param_state[error_name])
                        param_state[error_name] += update

                elif self.error_feedback == "EF21M":

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

                if self.DP:
                    if self.error_feedback is None:
                        gaussian_noise = self.noise * torch.randn_like(update).to(self.device)
                        update.data += gaussian_noise
                    if self.error_feedback == "ClipSGDM":
                        gaussian_noise = self.noise * torch.randn_like(update).to(self.device)
                        update.data += gaussian_noise
                    elif self.error_feedback == 'EF21':
                        gaussian_noise = self.noise * torch.randn_like(update).to(self.device)
                        update.data += gaussian_noise
                    elif self.error_feedback == 'ANorm':
                        gaussian_noise = self.noise * torch.randn_like(update).to(self.device)
                        update.data += beta * gaussian_noise
                    elif self.error_feedback == 'EF21M':
                        gaussian_noise = self.noise * torch.randn_like(update).to(self.device)
                        update.data += beta * gaussian_noise

                if 'full_grad' not in param_state:
                    #print('update full grad', self.n_iters, self.grads_received, w_id)
                    param_state['full_grad'] = update / self.n_workers
                else:
                    #print('update full grad', self.n_iters, self.grads_received, w_id)
                    param_state['full_grad'] += update / self.n_workers

                if self.grads_received == self.n_workers:
                    grad = param_state['full_grad']

                    if self.error_feedback == 'ANorm':
                        if self.normalize:
                            upd_norm = torch.sqrt(self.compute_update_norm()) + 1e-10
                            grad /= upd_norm 
                    
                    if self.error_feedback is None:
                        param_state['full_grad'] = torch.zeros_like(grad)

                    if weight_decay != 0:
                        grad.add(p, alpha=weight_decay)

                    #p.data.add_(grad, alpha=-1)
                    p.copy_(p - lr*grad)
        

        if self.grads_received == self.n_workers:
            self.grads_received = 0

        return loss
