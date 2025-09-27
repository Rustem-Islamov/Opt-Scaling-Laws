import torch

class Attack():
    def __init__():
        pass
    def __call__(self, *args, **kwds):
        pass

    
class BitFlippingAttack(Attack):

    def __init__(self, n_byz_workers):
        self.n_byz_workers = n_byz_workers

    def __call__(self, good_gradients):
        stacked_gradients = torch.stack(good_gradients, 1)
        avaraged_gradient = torch.mean(stacked_gradients, 1)

        return self.n_byz_workers * [-avaraged_gradient]
    
    def __str__(self):
        return "BitFlippingAttack"


        
    