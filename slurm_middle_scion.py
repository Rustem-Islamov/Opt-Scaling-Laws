#!/bin/bash
#SBATCH -J scion-775-052
#SBATCH -p cscc-gpu-p
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH -o train_logs/%x_%A.out
#SBATCH -e train_logs/%x_%A.err

set -euo pipefail
# SBATCH -p cscc-gpu-p
#  SBATCH --account=cscc-users 

# or

# SBATCH -p long

# --- Conda setup for non-interactive Slurm jobs ---
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate plainLM

# Ensure relative paths work
cd "$SLURM_SUBMIT_DIR"

# --- Logging setup ---
OUTDIR="scion_train_logs"
mkdir -p "$OUTDIR"

stamp="$(date +'%Y%m%d-%H%M%S')_${SLURM_JOB_ID}"
LOGFILE="${OUTDIR}/scion_${stamp}.log"

echo "[$(date)] Job started"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Working dir: $(pwd)"
echo "Logging to: ${LOGFILE}"
echo "----------------------------------------"

# --- Config manipulation ---
ORIG_CFG="./configs/scion.yaml"
NEW_CFG="./configs/temp/scion_${stamp}.yaml"

mkdir -p "./configs/temp"

python - <<EOF
import yaml
from pathlib import Path

orig = Path("${ORIG_CFG}")
new = Path("${NEW_CFG}")

with orig.open("r") as f:
    cfg = yaml.safe_load(f)

# Update requested fields
base_lr = 0.5*2.4e-4
alpha=0.18
cfg["batch_size"] = 400
cfg["device_batch_size"] = 2
cfg["lr_embed"] = base_lr
cfg["lr_matrix"] = base_lr
cfg["sequence_length"] = 2048
cfg["val_loss_every"] = 1000
cfg["n_layer"] = 36
cfg["n_head"] = 20
cfg["n_embd"] = 1280
cfg["momentum"] = alpha
t_budget = 10200*400*2048
cfg["num_iterations"] = int(t_budget/cfg["sequence_length"]/cfg["batch_size"])

cfg["project"] += f"large-model-budget{t_budget}"
cfg["warmdown_iters"] = int(0.28*cfg["num_iterations"])

# Update run name
run = cfg.get("run", "scion")
cfg["run"] = f"{run}_{cfg['batch_size']}_{cfg["num_iterations"]}_{base_lr}"

# Write new config
with new.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote updated config to {new}")
EOF

echo "Using config: ${NEW_CFG}"
echo "----------------------------------------"

# --- Run training ---
torchrun --standalone --nproc_per_node=4 \
    train_gpt_scion.py \
    --config="${NEW_CFG}" \
    2>&1 | tee "$LOGFILE"

echo "----------------------------------------"
echo "[$(date)] Job finished"
