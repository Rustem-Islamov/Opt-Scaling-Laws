import numpy as np
import os
import random
import torch


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
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)