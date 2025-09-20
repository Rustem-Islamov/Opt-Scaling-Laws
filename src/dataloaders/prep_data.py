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

    n_val = int(np.floor(0.001 * n))
    val_data = Subset(train_data, indices=indices[:n_val])

    indices = indices[n_val:]
    n = len(indices)
    a = int(np.floor(n / (2*n_workers)))
    top_ind = a * n_workers
    #top_ind = 0
    seq = range(a, top_ind, a)
    split = np.split(indices[:top_ind], seq)
    #split = [np.array([]).astype(int) for _ in range(n_workers)]
    
    
    for idx in indices[top_ind:]:
    #for idx in indices:
        cur_cl = train_data[idx][1]
        split[cur_cl] = np.append(split[cur_cl], idx)
        
    min_len = min([len(split[i]) for i in range(n_workers)])
    
    for idx in range(n_workers):
        split[idx] = split[idx][:min_len]


    b = 0
    for ind in split:
        train_loader_workers[b] = DataLoader(Subset(train_data, ind), batch_size=batch_size, shuffle=True)
        b = b + 1

    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader_workers, val_loader, test_loader


def load_data(dataset_name,
              horizontal_flip_p = 0,
              random_crop_padding = 0,
              ):

    if dataset_name == 'mnist':

        transform = transforms.ToTensor()

        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)

        test_data = datasets.MNIST(root='data', train=False,
                                   download=True, transform=transform)
    
    elif dataset_name in ['cifar10', 'cifar100']:


        if dataset_name == 'cifar10':

            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.2471, 0.2435, 0.2616])
        
        else:

            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                            std=[0.2675, 0.2565, 0.2761]) 


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p = horizontal_flip_p),
            transforms.RandomCrop(size = 32, padding = random_crop_padding),
            transforms.ToTensor(),
            normalize,
        ])

        dataset_class = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.CIFAR100

        train_data = dataset_class(root='data', train=True,
                                    download=True, transform=transform_train)

        test_data = dataset_class(root='data', train=False,
                                    download=True, transform=transform_test)
    else:
        raise ValueError(dataset_name + ' is not known.')

    return train_data, test_data
