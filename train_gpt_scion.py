# Modified from: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt
import os
import sys
import uuid
import time
import wandb
import torch
import torch.distributed as dist
import torch._inductor.config as config

import random
import numpy as np

from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from utils.gpt import GPT, GPTConfig
from optimizers.scion import Scion
from utils.dataloader import DistributedDataLoader
from utils.configs import load_config

def seed_everything(seed: int, rank: int):
    seed = seed + rank  # IMPORTANT: different seed per rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# read the code of this file ASAP, for logging
with open(sys.argv[0]) as f:
    code = f.read() 

# -----------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
cli_args = parser.parse_args()
args = load_config(cli_args.config)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])

# Example usage
BASE_SEED = getattr(args, "seed", 1337)
seed_everything(BASE_SEED, ddp_rank)

device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

if master_process:
    print("======== Arguments ========")
    print(args)
    print("===========================")
    
# convenience variables
B, T = args.device_batch_size, args.sequence_length
T_eval = args.sequence_length_eval
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T_eval * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T_eval * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T_eval, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
import torch._dynamo
torch._dynamo.config.optimize_ddp = False

model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
optim_groups = [{
    'params': raw_model.transformer.h.parameters(), 
    'norm': 'Spectral', 
    'norm_kwargs': {'steps': 5}, 
    'scale': args.scale_embed,
    'weight_decay': args.weight_decay,
}, {
    'params': raw_model.lm_head.parameters(), 
    'norm': 'Sign', 
    'norm_kwargs': {}, 
    'scale': args.scale_matrix,
    'weight_decay': args.weight_decay,
}]
optimizer1 = Scion(optim_groups, lr=args.lr_embed, momentum=args.momentum, unconstrained=args.unconstrained)
optimizers = [optimizer1]

# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup
    if it < args.warmup_iters:
        ratio = (it+1) / args.warmup_iters
    # 2) constant
    elif it < args.num_iterations - args.warmdown_iters:
        ratio = 1.0
    # 3) linear warmdown
    else:
        ratio = (args.num_iterations - it) / args.warmdown_iters
    # clamp ratio so final LR = 1e-5
    min_ratio = 1e-8 / args.lr_embed
    return max(ratio, min_ratio)
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    wandb.init(
        project=args.project, 
        name=args.run, 
        config=vars(args)
    )
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# ----------------------------
# start global timer (optional, total training)
torch.cuda.synchronize()
global_start_time = time.time()

model.eval()
val_loader.reset()

val_loss = torch.zeros((), device=device)
for _ in range(val_steps):
    x_val, y_val = val_loader.next_batch()
    with ctx:
        _, loss = model(x_val, y_val, return_logits=False)
        val_loss += loss.detach()

dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
val_loss /= val_steps

if master_process:
    wandb.log({
        "val_loss": val_loss.item(),
    })
    
val_loader.reset()
train_loader.reset()
train_iter_start = time.time()  # timer for training iterations only

for step in tqdm(range(1, args.num_iterations + 1)):
    last_step = (step == args.num_iterations)

    # --------------- EVALUATION -----------------
    val_loss = None
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # validation does NOT affect training timer

    if last_step:
        break

    # --------------- TRAINING SECTION -----------------.
    train_iter_start = time.time()
    model.train()
    batch_tokens = B * T * ddp_world_size * train_accumulation_steps

    for i in range(train_accumulation_steps+1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        x, y = train_loader.next_batch()
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)

    # ----------------- TRAIN TIMING -----------------
    torch.cuda.synchronize()
    train_iter_end = time.time()
    iter_elapsed_sec = (train_iter_end - train_iter_start)
    tokens_per_sec = batch_tokens / iter_elapsed_sec

    # ----------------- LOGGING -----------------
    if master_process:
        lr = sched.get_last_lr()[0]
        wandb.log({
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item() if val_loss else None,
            "learning_rate": lr,
            "tokens/sec": tokens_per_sec,
        })
        
model.eval()
val_loader.reset()

val_loss = torch.zeros((), device=device)

for _ in range(val_steps):
    x_val, y_val = val_loader.next_batch()
    with ctx:
        _, loss = model(x_val, y_val, return_logits=False)
        val_loss += loss.detach()

# ALL ranks must participate
dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
val_loss /= val_steps

# ONLY rank 0 logs
if master_process:
    wandb.log({
        "run/finish_val_loss": val_loss.item(),
    })

if master_process:
    wandb.log({
        "train_loss": train_loss.item(),
        "val_loss": val_loss.item() if val_loss else None,
        "learning_rate": lr,
        "tokens/sec": tokens_per_sec,
    })
    wandb.finish()
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()