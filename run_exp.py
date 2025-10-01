from src.models.resnet_cifar import *
from src.models.mnist_models import Net
from torch.nn import CrossEntropyLoss
from train import run_exp
from src.utils.utils import create_exp
import numpy as np

from src.aggregators.NNM import NNM

n_workers = 20
n_byzant_workers = 5
model = 'mnist_net'
dataset_name = 'mnist'
dataset_train_size=60000
method = 'ByzClip21-SGD2M'
#bs = 250
ef = 'EF21M'
sch = None
DP = True
epochs = 60
delta = 4e-4
seed = 228


from src.attacks.attacks import BitFlippingAttack


# None, CWMedian

agg = "NNM"
attack = BitFlippingAttack(n_byzant_workers)

for beta in [0.1]: #momentum
    for hbeta in [0.01, 0.1]:  #beta hat from paper
        for tau in [1e-2, 1e-3, 1e-4, 1e-5]:  #clipping
            for lrs in [10, 1, 1e-1, 1e-2, 1e-3]: #1e-1, 1e-2, 1e-3, 1e-4
                for eps in [18]:

                    bs=256 #

                    # 8 13 18 23 28
                    #DP_noise = tau/eps*np.sqrt(epochs*np.log(1/delta))
                    DP_noise = tau/eps*np.sqrt(dataset_train_size/n_workers/bs*epochs*np.log(1/delta))
                
                    #net = _get_resnet(model, 10)
                    net = Net()

                    exp = create_exp(name=model, dataset=dataset_name, net=net, device='cuda', model_name=model,
                                        n_workers=n_workers, epochs=epochs, seed=seed, batch_size=bs,
                                        compression={'wrapper':False, 'compression':None},
                                        error_feedback=ef, criterion=CrossEntropyLoss(),
                                        beta=hbeta, momentum=beta, lrs=[lrs], tau=tau, noise=DP_noise, DP=DP,
                                        master_compression=None,  weight_decay=0,
                                        robust_aggregator=agg, n_byzant_workers=n_byzant_workers,
                                        delta=delta, eps=eps, attack=attack)
                    best_lr, best_acc_lr = run_exp(exp,
                                                   suffix=f'AISTATS2026_{method}_{model}_lr{lrs}_beta_{beta}_hbeta{hbeta}_tau{tau}_seed{seed}_eps{eps}_agg{str(agg)}_attack{str(attack)}',
                                                   schedule=sch)
