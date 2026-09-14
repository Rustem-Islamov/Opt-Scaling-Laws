#!/bin/bash
#SBATCH -J opt-scion-base-model-tuning
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH -o opt_scion_train_logs/%x_%A.out
#SBATCH -e opt_scion_train_logs/%x_%A.err
#SBATCH --exclude=gpu-39,gpu-59

# SBATCH -p cscc-gpu-p

# or

#SBATCH --partition=long
#SBATCH --qos=gpu-12


set -euo pipefail




# --- Conda setup for non-interactive Slurm jobs ---
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate plainLM

# --- Diagnose + force cache dirs to a path this user actually owns ---
echo "HOME=$HOME"
echo "TMPDIR=${TMPDIR:-<unset>}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME:-<unset>}"
echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-<unset>}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-<unset>}"

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="$HOME/.cache"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export TORCH_COMPILE_CACHE_DIR="$HOME/.cache/torch_compile"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_COMPILE_CACHE_DIR"

python -c "from torch._dynamo.package import cache_dir; print('dynamo cache_dir resolves to:', cache_dir())"

# Ensure relative paths work
cd "$SLURM_SUBMIT_DIR"

# --- Logging setup ---
OUTDIR="opt_scion_train_logs"
mkdir -p "$OUTDIR"

stamp="$(date +'%Y%m%d-%H%M%S')_${SLURM_JOB_ID}"
LOGFILE="${OUTDIR}/opt_scion_${stamp}.log"

echo "[$(date)] Job started"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Working dir: $(pwd)"
echo "Logging to: ${LOGFILE}"
echo "----------------------------------------"

# --- Config manipulation ---
ORIG_CFG="./configs/opt_scion.yaml"
NEW_CFG="./configs/temp/opt_scion_${stamp}.yaml"

mkdir -p "./configs/temp"

python - <<EOF
import yaml
from pathlib import Path

orig = Path("${ORIG_CFG}")
new = Path("${NEW_CFG}")

with orig.open("r") as f:
    cfg = yaml.safe_load(f)

# Update requested fields
cfg["seed"] = 103
d_mom1 = 0.95
d_mom2 = 0.95

cfg["adj_embed"] = 1
cfg["adj_matrix"] = 1

cfg["scale_embed"] = int(50 * cfg["adj_embed"])
cfg["scale_matrix"] = int(3000 * cfg["adj_matrix"])

cfg["batch_size"] = 1248
increase = 15

cfg["sequence_length"] = 1024
t_0 = 5100 * 256 * 1024

t_budget = increase * t_0
cfg["num_iterations"] = int(t_budget/cfg["batch_size"]/cfg["sequence_length"])

base_lr = 2.5e-4
cfg["lr_embed"] = base_lr
cfg["lr_matrix"] = base_lr

cfg["device_batch_size"] = 32
cfg["val_loss_every"] = 100
cfg["d_mom1"] = d_mom1
cfg["d_mom2"] = d_mom2

cfg["project"] = f"124M-CORRECT-OPT-SCION-budget{t_budget}"
cfg["warmdown_iters"] = int(0.28*cfg["num_iterations"])


run = cfg.get("run", "opt-scion")
cfg["run"] = f"{run}_{cfg['batch_size']}_{cfg['num_iterations']}_{d_mom1}_{d_mom2}_{base_lr}"


# Print all hyperparameters before writing/training
print("======== Config ========")
for k, v in sorted(cfg.items()):
    print(f"{k}: {v}")
print("=========================")

# Write new config
with new.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote updated config to {new}")
EOF

echo "Using config: ${NEW_CFG}"
echo "----------------------------------------"

# --- Run training ---
torchrun --standalone --nproc_per_node=1 \
    train_gpt_optimistic_scion.py \
    --config="${NEW_CFG}" \
    2>&1 | tee "$LOGFILE"

echo "----------------------------------------"
echo "[$(date)] Job finished"
