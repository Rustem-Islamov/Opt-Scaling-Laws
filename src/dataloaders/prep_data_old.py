import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def create_loaders(dataset_name, n_workers, batch_size, seed=42):

    train_data, test_data = load_data(dataset_name)

    train_loader_workers = dict()
    n = len(train_data)

    # preparing iterators for workers and validation set
    np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_val = int(np.floor(0.1 * n))
    val_data = Subset(train_data, indices=indices[:n_val])

    indices = indices[n_val:]
    #labels = [train_data[i][1] for i in indices]
    #indices = [x[1] for x in sorted(zip(labels, indices))]
    
    n = len(indices)
    a = int(np.floor(n / (n_workers)))
    top_ind = a * n_workers
    seq = range(a, top_ind, a)
    split = np.split(indices[:top_ind], seq)
    #split = [np.array([]).astype(int) for _ in range(n_workers)]
    
    
    for idx in indices[top_ind:]:
    #for idx in indices:
        cur_cl = train_data[idx][1]
        split[cur_cl] = np.append(split[cur_cl], idx)
        
    min_len = min([len(split[i]) for i in range(n_workers)])
    #print(min_len)
    
    for idx in range(n_workers):
        split[idx] = split[idx][:min_len]

    b = 0
    for ind in split:
        train_loader_workers[b] = DataLoader(Subset(train_data, ind), batch_size=batch_size, shuffle=True)
        b = b + 1

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader_workers, val_loader, test_loader


def load_data(dataset_name):

    if dataset_name == 'mnist':

        transform = transforms.ToTensor()

        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)

        test_data = datasets.MNIST(root='data', train=False,
                                   download=True, transform=transform)
    elif dataset_name == 'cifar10':

        normalize = transforms.Normalize(mean=[0.49154913, 0.4821251, 0.44642678],
                                         std = [0.24703223, 0.24348513, 0.26158784])
        train_transform = transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize,
                                             ])
        
        val_transform = transforms.Compose([transforms.ToTensor(),
                                            normalize,
                                            ])
        
        train_data = datasets.CIFAR10(root='data', train=True,
                                      download=True, transform=train_transform)

        test_data = datasets.CIFAR10(root='data', train=False,
                                     download=True, transform=val_transform)
    elif dataset_name == 'cifar100':
        normalize = transforms.Normalize(mean = [0.5071, 0.4866, 0.4409],
                                         std = [0.2673, 0.2564, 0.2762])
        
        train_transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                             ])
        
        val_transform = transforms.Compose([transforms.ToTensor(),
                                            normalize,
                                            ])
        
        train_data = datasets.CIFAR100(root='data', train=True,
                                       download=True, transform=train_transform)

        test_data = datasets.CIFAR100(root='data', train=False,
                                      download=True, transform=val_transform)
        
    else:
        raise ValueError(dataset_name + ' is not known.')

    return train_data, test_data
