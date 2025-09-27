import glob
import numpy as np
import os
import random
import torch

from pickle import load, dump



def set_worker_seed(worker_id):
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed):
    """
    Set random seed for model training or inference.

    Args:
        seed (int): defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)



def evaluate(model, loader, criterion, device):
    """
    Compute loss and accuracy
    """

    correct = 0
    total_loss = 0

    model.eval()
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()

        #preds = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        _, preds = torch.max(output.data, 1)
        correct += (preds == labels).sum().item()

    accuracy = 100. * correct / len(loader.dataset)
    total_loss = total_loss / len(loader)

    return total_loss, accuracy


SAVED_RUNS_PATH = 'saved_data/'
EXP_PATH = 'exps_setup/'


def save_run(suffix, run):
    if not os.path.isdir(SAVED_RUNS_PATH):
        os.mkdir(SAVED_RUNS_PATH)

    file = SAVED_RUNS_PATH + suffix + '.pickle'
    with open(file, 'wb') as f:
        dump(run, f)


def read_all_runs(exp, suffix=None):
    if suffix is None:
        suffix = exp['name']

    runs = list()
    runs_files = glob.glob(SAVED_RUNS_PATH + suffix + '_' + '[1-9]*.pickle')  # reads at most first ten runs
    for run_file in runs_files:
        runs.append(read_run(run_file))
    return runs


def read_run(file):
    with open(file, 'rb') as f:
        run = load(f)
    return run


def create_run():
    run = {'train_loss': [],
           'train_acc': [],
           'test_loss': [],
           'test_acc': []
           }
    return run


def update_run(train_loss, test_loss, test_acc, run):
    run['train_loss'].append(train_loss)
    run['test_loss'].append(test_loss)
    run['test_acc'].append(test_acc)


def save_exp(exp):
    if not os.path.isdir(EXP_PATH):
        os.mkdir(EXP_PATH)

    file = EXP_PATH + exp['name'] + '.pickle'
    with open(file, 'wb') as f:
        dump(exp, f)


def load_exp(exp_name):
    file = EXP_PATH + exp_name + '.pickle'
    with open(file, 'rb') as f:
        exp = load(f)
    return exp


def create_exp(name, dataset, net, model_name, n_workers, epochs, seed, batch_size, lrs, tau, noise, DP, compression, error_feedback, criterion, device,
               robust_aggregator, n_byzant_workers, attack, delta, eps, master_compression=None, momentum=0, beta=1, weight_decay=0, normalize=False):
    exp = {
        'name': name,
        'dataset_name': dataset,
        'net': net,
        'model_name':model_name,
        'n_workers': n_workers,
        'epochs': epochs,
        'seed': seed,
        'batch_size': batch_size,
        'lrs':  lrs,
        'tau': tau,
        'noise': noise,
        'DP': DP,
        'lr': None,
        'device':device,
        'compression': compression,
        'master_compression': master_compression,
        'error_feedback': error_feedback,
        'criterion': criterion,
        'momentum': momentum,
        'beta': beta,
        'weight_decay': weight_decay,
        'normalize': normalize,
        'robust_aggregator': robust_aggregator,
        'n_byzant_workers': n_byzant_workers,
        'attack': attack,
        'eps': eps,
        'delta': delta,
        }
    return exp