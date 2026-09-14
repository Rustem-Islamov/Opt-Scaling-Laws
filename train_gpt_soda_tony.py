# Training script for GPT with SODA optimizer
# Based on co-author's train_gpt_soda.py, corrected for Jean Zay
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
from optimizers.soda import SODA
from utils.dataloader import DistributedDataLoader
from utils.configs import load_config

def seed_everything(seed: int, rank: int):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

os.environ.setdefault("NCCL_TIMEOUT", "30")

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
cli_args = parser.parse_args()
args = load_config(cli_args.config)

assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])

BASE_SEED = getattr(args, "seed", 1337)
seed_everything(BASE_SEED, ddp_rank)

device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0)

try:
    if master_process:
        print("======== Arguments ========")
        print(args)
        print("===========================")

    B, T = args.device_batch_size, args.sequence_length
    T_eval = args.sequence_length_eval
    assert args.val_tokens % (B * T_eval * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T_eval * ddp_world_size)
    print(args.batch_size, B, ddp_world_size)
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T_eval, ddp_rank, ddp_world_size)
    if master_process:
        print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    x, y = train_loader.next_batch()

    num_vocab = 50304
    model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd))
    model = model.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True
    import torch._dynamo
    torch._dynamo.config.optimize_ddp = False
    model = DDP(model, device_ids=[ddp_local_rank])
    model = torch.compile(model)

    raw_model = model.module
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # --- SODA optimizer setup ---
    if master_process:
        tied = (raw_model.transformer.wte.weight is raw_model.lm_head.weight)
        print(f"embedding/lm_head weight tying (shared tensor): {tied}")

    optim_groups = [{
        'params': raw_model.transformer.h.parameters(),
        'norm': 'Spectral',
        'norm_kwargs': {'steps': 5},
        'radius': args.scale_embed,
    }, {
        'params': raw_model.lm_head.parameters(),
        'norm': 'Sign',
        'norm_kwargs': {},
        'radius': args.scale_matrix,
    }]

    n_hidden = sum(p.numel() for p in raw_model.transformer.h.parameters())
    n_embed = sum(p.numel() for p in raw_model.lm_head.parameters())
    if master_process:
        print(f"hidden params (Spectral, radius={args.scale_embed}): {n_hidden:,} | "
              f"embed params (Sign, radius={args.scale_matrix}): {n_embed:,}")
        print(f"dual_momentum1={args.d_mom1}, dual_momentum2={args.d_mom2}")
        print(f"lr={args.lr_embed}")

    optimizer1 = SODA(optim_groups, lr=args.lr_embed,
                       dual_momentum1=args.d_mom1, dual_momentum2=args.d_mom2)
    optimizers = [optimizer1]

    # Learning rate schedule: linear warmup and warmdown
    def get_lr(it):
        assert it <= args.num_iterations
        if args.warmup_iters > 0 and it < args.warmup_iters:
            ratio = (it + 1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            ratio = 1.0
        else:
            ratio = (args.num_iterations - it) / args.warmdown_iters
        min_ratio = 1e-8 / args.lr_embed
        return max(ratio, min_ratio)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # begin logging
    if master_process:
        run_id = str(uuid.uuid4())
        wandb_kwargs = dict(
            project=args.project,
            name=args.run,
            config=vars(args),
        )
        if hasattr(args, 'entity') and args.entity:
            wandb_kwargs['entity'] = args.entity
        wandb.init(**wandb_kwargs)
        logdir = 'logs/%s/' % run_id
        os.makedirs(logdir, exist_ok=True)
        logfile = 'logs/%s.txt' % run_id
        with open(logfile, "w") as f:
            f.write('='*100 + '\n')
            f.write(code)
            f.write('='*100 + '\n')
            f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(f'{result.stdout}\n')
            f.write('='*100 + '\n')

    training_time_ms = 0
    torch.cuda.synchronize()
    global_start_time = time.time()

    # --- Initial validation ---
    model.eval()
    val_loader.reset()

    val_loss = torch.zeros((), device=device)
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx:
            loss = model(x_val, y_val, return_logits=False)
            val_loss += loss.detach()

    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    val_loss /= val_steps

    if master_process:
        wandb.log({"val_loss": val_loss.item()})

    val_loader.reset()
    train_loader.reset()
    train_iter_start = time.time()

    for step in tqdm(range(1, args.num_iterations + 1)):
        last_step = (step == args.num_iterations)

        # --------------- EVALUATION -----------------
        val_loss = None
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            model.eval()
            val_loader.reset()
            val_loss = torch.zeros((), device=device)
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with ctx:
                    loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps

        if last_step:
            break

        # --------------- TRAINING SECTION -----------------
        train_iter_start = time.time()
        model.train()
        batch_tokens = B * T * ddp_world_size * train_accumulation_steps

        for i in range(1, train_accumulation_steps + 1):
            with ctx:
                loss = model(x, y, return_logits=False)
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
                "val_loss": val_loss.item() if val_loss is not None else None,
                "learning_rate": lr,
                "tokens/sec": tokens_per_sec,
                "step": step,
            })

    # --- Final validation ---
    model.eval()
    val_loader.reset()

    val_loss = torch.zeros((), device=device)
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx:
            loss = model(x_val, y_val, return_logits=False)
            val_loss += loss.detach()

    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    val_loss /= val_steps

    if master_process:
        wandb.log({"run/finish_val_loss": val_loss.item()})

    if master_process:
        wandb.log({
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item() if val_loss is not None else None,
            "learning_rate": lr,
            "tokens/sec": tokens_per_sec,
        })
        wandb.finish()
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    dist.destroy_process_group()

except Exception as e:
    print(f"[rank {ddp_rank}] Exception: {repr(e)}", flush=True)
    raise

finally:
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    finally:
        pass
