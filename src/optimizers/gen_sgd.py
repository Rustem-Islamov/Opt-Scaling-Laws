import torch
from torch.optim.optimizer import Optimizer
from src.aggregators import CM, Mean

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
                 momentum=0,
                 beta=1, 
                 dampening=0, 
                 tau=None,      #Clipping
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
        self.n_byzant_workers = n_byzant_workers
        self.attack = attack

        self.grads_received = 0
        self.n_iters = 0

        if robust_aggregator is None:
            self.robust_aggregator = Mean()
        elif robust_aggregator == "CWMedian":
            self.robust_aggregator = CM()
        else:
            raise ValueError("Unknown robbust aggregator")    
        
        
        for group in self.param_groups:
            momentum = group['momentum']
            beta = group['beta']
            lr = group['lr']
            print('mom:', momentum, 'beta:', beta, 'lr:', lr, 'tau:', self.tau)

    def __setstate__(self, state):
        print("WARNING OPTIMZER SETSTATE")
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

            clip_norm = torch.sqrt(self.compute_clip_norm(w_id)) + 1e-10

            clip_coef = min(1.0, self.tau / clip_norm)

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                d_p = p.grad.data.clone()
                
                if 'raw_grads' not in param_state:
                    param_state['raw_grads'] = [d_p]
                else:
                    param_state['raw_grads'].append(d_p)

                if self.error_feedback == None:

                    d_p = d_p * clip_coef
                    update = d_p

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
                    elif self.error_feedback == 'EF21M':
                        gaussian_noise = self.noise * torch.randn_like(update).to(self.device)
                        update.data += beta * gaussian_noise

                if 'updates' not in param_state:
                    param_state['updates'] = [0] * (self.n_workers + self.n_byzant_workers)
                    param_state['updates'][self.grads_received - 1] = update
                else:
                    param_state['updates'][self.grads_received - 1] += update

                if self.grads_received == self.n_workers:
                    # COMPARE AVG VS NNM + CWM
                    # BITFLIPPING

                    #print(len(param_state['updates']))
                    #orig_mean = torch.stack(param_state['updates'], 1).mean()
                    #print(orig_mean)

                    # print("raw grad len", len(param_state['raw_grads']))

                    corrupted_grad = self.attack(param_state['raw_grads'])
                    for i in range(self.n_byzant_workers):
                        param_state['updates'][self.n_workers + i] += beta * corrupted_grad

                    # print([el.mean() for el in param_state['updates']])

                    param_state['raw_grads'] = []
                    #print(len(param_state['updates']))
                    #print(torch.stack(param_state['updates'], 1).mean())
                    #print((orig_mean * self.n_workers - orig_mean * self.n_byzant_workers) / (self.n_workers + self.n_byzant_workers))

                    if 'full_grad' not in param_state:
                        param_state['full_grad'] = self.robust_aggregator(param_state['updates'])
                    else:
                        param_state['full_grad'] += self.robust_aggregator(param_state['updates'])

                    grad = param_state['full_grad']
                    
                    if self.error_feedback is None:
                        param_state['full_grad'] = torch.zeros_like(grad)

                    p.copy_(p - lr*grad)
        

        if self.grads_received == self.n_workers:
            self.grads_received = 0

        return loss
