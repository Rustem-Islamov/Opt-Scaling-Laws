import argparse
import numpy as np
from torch.nn import CrossEntropyLoss

from src.attacks.attacks import BitFlippingAttack
from src.models.resnet_cifar import *
from src.models.mnist_models import Net
from src.models.cifar10_cnn import CifarNet
from src.utils.utils import create_exp
from train import run_exp

def get_attack_from_name(attack_name, n_byzant):
    if attack_name.lower() == 'bitflippingattack':
        return BitFlippingAttack(n_byzant)
    else:
        raise ValueError(f"Unknown attack type: {attack_name}")

def get_model_from_name(model_name):
    if model_name.lower() == 'cnn_cifar':
        return CifarNet()
    if model_name.lower() == 'cnn_mnist':
        return Net()
    
    raise ValueError(f"Unsupported model '{model_name}")

def get_train_dataset_size(dataset_name):
    if dataset_name in ['cifar10', 'cifar100']:
        return 50000
    elif dataset_name == 'mnist':
        return 60000
    else:
        raise Exception


def main(args):
    tau = args.tau     #clipping
    lrs = args.lr
    beta = args.beta
    hbeta = args.hbeta
    seed = args.seed
    eps = args.eps
    bs = args.bs
    epochs = args.epochs
    n_workers = args.n_workers
    n_byzant_workers = args.n_byzant_workers
    DP = args.DP
    ef = args.ef
    delta = args.delta
    agg = args.agg
    device = args.device
    
    model_name = args.model
    method_name = args.method

    dataset_name = args.dataset_name
    dataset_train_size = get_train_dataset_size(dataset_name)
    attack = get_attack_from_name(args.attack, n_byzant = n_byzant_workers)

    DP_noise = tau/eps*np.sqrt(dataset_train_size/n_workers/bs*epochs*np.log(1/delta))

    net = get_model_from_name(model_name)

    project_name = f"NAME_{method_name}_{dataset_name}_EPOCHS{epochs}_BS{bs}_NEWATTACK_{n_byzant_workers}BYZ_EPS{eps}"

    exp = create_exp(name=model_name, dataset=dataset_name, net=net, device=device, model_name=model_name,
                        n_workers=n_workers, epochs=epochs, seed=seed, batch_size=bs,
                        compression={'wrapper':False, 'compression':None},
                        error_feedback=ef, criterion=CrossEntropyLoss(),
                        beta=hbeta, momentum=beta, lrs=[lrs], inner_tau=tau, outer_tau=None, noise=DP_noise, DP=DP,
                        master_compression=None,  weight_decay=0,
                        robust_aggregator=agg, n_byzant_workers=n_byzant_workers,
                        delta=delta, eps=eps, attack=attack,
                        project_name = project_name)
    best_lr, best_acc_lr = run_exp(exp,
                                    suffix=f'AISTATS2026_{method_name}_{model_name}_lr{lrs}_beta_{beta}_hbeta{hbeta}_tau{tau}_seed{seed}_eps{eps}_agg{str(agg)}_attack{str(attack)}',
                                    schedule=None)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument("device", type=str)
    parser.add_argument("tau", type=float, help="clipping")
    
    parser.add_argument("--model", type=str, default='cnn_mnist', choices=['cnn_mnist', 'cnn_cifar'], help="name of architecture for logging")
    parser.add_argument("--dataset_name", type=str, default='mnist', choices=['cifar10', 'mnist'], help="dataset")
    parser.add_argument("--method", type=str, default='AIRBENCH_TEST', help="name of method for logging")
    parser.add_argument("--seed", type=int, default=228, help="random seed")

    parser.add_argument("--n_workers", type=int, default=20, help="number of good workers")
    parser.add_argument("--n_byzant_workers", type=int, default=5, help="number of byz workers")
    parser.add_argument("--agg", type=str, default="NNM", help="tobust aggregator")
    parser.add_argument("--attack", type=str, default='BitFlippingAttack', choices=['BitFlippingAttack', None], help="attack type")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1, help="momentum")
    parser.add_argument("--hbeta", type=float, default=0.01, help="beta_hat")
    parser.add_argument("--ef", type=str, default='EF21M', choices=[None, 'EF21M', 'Safe-DSHB'], help="ef mechanism")

    parser.add_argument("--DP", type=bool, default=True, help="enable differential privacy")
    parser.add_argument("--eps", type=float, default=18, help="epsilon privacy budget")
    parser.add_argument("--delta", type=float, default=4e-4, help="delta privacy budget")
    
    args = parser.parse_args()

    main(args)