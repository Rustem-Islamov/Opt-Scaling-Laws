import torch
import numpy as np

from utils import create_run, update_run, save_run, seed_everything
from prep_data import create_loaders
from optimizer.gen_sgd import SGDGen

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb

RUNS = 3


def train_workers(suffix, model, optimizer, criterion, epochs, train_loader_workers,device,
                  val_loader, test_loader, n_workers, hpo=False, scheduler=None):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    run = create_run()
    train_loss = np.inf

    best_val_loss = np.inf
    test_loss = np.inf
    test_acc = 0
    best_val_acc = 0
    ###
    val_loss, _ = accuracy_and_loss(model, val_loader, criterion, device)  ### Computing loss before training
    
    if val_loss < best_val_loss:
            test_loss, test_acc = accuracy_and_loss(model, test_loader, criterion, device)
            best_val_loss = val_loss
            
    update_run(train_loss, test_loss, test_acc, run)    
    ###

    
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        train_loader_iter = [iter(train_loader_workers[w]) for w in range(n_workers)]
        iter_steps = len(train_loader_workers[0])
        for _ in range(iter_steps):
            for w_id in range(n_workers):
                data, labels = next(train_loader_iter[w_id])
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                running_loss += loss.item()
                optimizer.step_local_global(w_id)
                optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
            print(optimizer.param_groups[0]["lr"])

        train_loss = running_loss/(iter_steps*n_workers)

        val_loss, val_acc = accuracy_and_loss(model, val_loader, criterion, device)

        #if val_loss < best_val_loss:
        #    test_loss, test_acc = accuracy_and_loss(model, test_loader, criterion, device)
        #    best_val_loss = val_loss
        
        #if val_acc > best_val_acc:
        test_loss, test_acc = accuracy_and_loss(model, test_loader, criterion, device)
        best_val_acc = val_acc
        best_val_loss = val_loss

        update_run(train_loss, test_loss, test_acc, run)

        print('Epoch: {}/{}.. Training Loss: {:.5f}, Test Loss: {:.5f}, Test accuracy: {:.2f}'.format(e + 1, epochs, train_loss, test_loss, test_acc), end='\n')
        wandb.log({"epoch":e+1, "train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc})

    print('')
    if not hpo:
        save_run(suffix, run)

    wandb.finish()

    return best_val_loss, best_val_acc


def accuracy_and_loss(model, loader, criterion, device):
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


def tune_step_size(exp, suffix=None, schedule=None):
    best_val_loss = np.inf
    best_lr = 0
    best_val_acc = 0
    best_acc_lr = 0
    
    seed = exp['seed']
    seed_everything(seed)
    hpo = False
    
    exp['val_losses'] = []
    exp['val_accs'] = []
    for idx, lr in enumerate(exp['lrs']):
        print('Learning rate {:2.4f}:'.format(lr))
        if schedule is not None:
            val_loss, val_acc = run_workers(lr, exp, suffix=suffix+'lr_{}'.format(lr), hpo=hpo, schedule=schedule)
        else:
            val_loss, val_acc = run_workers(lr, exp, suffix=suffix+'lr_{}'.format(lr), hpo=hpo)
        exp['val_losses'].append(val_loss)
        exp['val_accs'].append(val_acc)
        if val_loss < best_val_loss:
            best_lr = lr
            best_val_loss = val_loss
            
        if val_acc > best_val_acc:
            best_acc_lr = lr
            best_val_acc = val_acc
            
    return best_lr, best_acc_lr

def run_workers(lr, exp, suffix=None, hpo=False, schedule=None):
    dataset_name = exp['dataset_name']
    n_workers = exp['n_workers']
    batch_size = exp['batch_size']
    epochs = exp['epochs']
    criterion = exp['criterion']
    error_feedback = exp['error_feedback']
    momentum = exp['momentum']
    beta = exp['beta']
    tau = exp['tau']
    noise = exp['noise']
    device = exp['device']
    DP = exp['DP']
    seed = exp['seed']
    weight_decay = exp['weight_decay']
    compression = get_compression(**exp['compression'])
    master_compression = exp['master_compression']
    model_name = exp['model_name']
    normalize = exp['normalize']
    delta = 1e-3
    if error_feedback == 'ANorm':
        eps = round(1/noise*np.sqrt(epochs*np.log(1/delta)), 2)
    else:
        eps = round(tau/noise*np.sqrt(epochs*np.log(1/delta)), 2)
    

    wandb.init(
            # set the wandb project where this run will be logged
            project='NeurIPS' + dataset_name+model_name,
        
            # track hyperparameters and run metadata
            config={
            "ef":error_feedback,
            "dataset": dataset_name,
            "epochs": epochs,
            "batch size": batch_size,
            "seed":seed,
            "tau": tau,
            "DP": DP,
            "noise": noise,
            "lr": lr,
            "mom": momentum,
            "beta": beta,
            "eps": eps,
            "delta": delta,
            "normalize": normalize
            }
        )

    net = exp['net']
    model = net().to(device)

    train_loader_workers, val_loader, test_loader = create_loaders(dataset_name, n_workers, batch_size, seed)

    optimizer = SGDGen(model.parameters(), lr=lr, n_workers=n_workers, error_feedback=error_feedback,device=device,
                       comp=compression, momentum=momentum, beta=beta, tau=tau, noise=noise, DP=DP, weight_decay=weight_decay,
                       master_comp=master_compression, normalize=normalize)
    
    if schedule is not None:
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=2, epochs=epochs)
        #lambda1 = lambda epoch: lr * np.cos(np.pi/2 * epoch / (epochs + 1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.316)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        val_loss, val_acc = train_workers(suffix, model, optimizer, criterion, epochs, train_loader_workers,device,
                                 val_loader, test_loader, n_workers, hpo=hpo, scheduler=scheduler)
    else:
        val_loss, val_acc = train_workers(suffix, model, optimizer, criterion, epochs, train_loader_workers,device,
                                 val_loader, test_loader, n_workers, hpo=hpo)
                             
    return val_loss, val_acc


def run_tuned_exp(exp, runs=RUNS, suffix=None):
    if suffix is None:
        suffix = exp['name']

    lr = exp['lr']

    if lr is None:
        raise ValueError("Tune step size first")

    seed = exp['seed']
    seed_everything(seed)

    for i in range(runs):
        print('Run {:3d}/{:3d}, Name {}:'.format(i+1, runs, suffix))
        suffix_run = suffix + '_' + str(i+1)
        run_workers(lr, exp, suffix_run)


def get_single_compression(wrapper, compression, **kwargs):
    if wrapper:
        return compression(**kwargs)
    else:
        return compression


def get_compression(combine=None, **kwargs):
    if combine is None:
        return get_single_compression(**kwargs)
    else:
        compression_1 = get_single_compression(**combine['comp_1'])
        compression_2 = get_single_compression(**combine['comp_2'])
        return combine['func'](compression_1, compression_2)
