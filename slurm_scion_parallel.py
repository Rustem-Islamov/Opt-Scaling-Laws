#!/bin/bash
#SBATCH -J scion-64
#SBATCH -p cscc-gpu-p
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --array=0-4
#SBATCH -o train_logs/%x_%A_%a.out
#SBATCH -e train_logs/%x_%A_%a.err

set -euo pipefail

# -------------------------------
# Learning-rate sweep definition
# -------------------------------
LRS=(2.4e-4 3.6e-4 4.8e-4 6.0e-4 7.2e-4)
BASE_LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "Running LR = ${BASE_LR}"

# -------------------------------
# Conda setup (non-interactive)
# -------------------------------
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate plainLM

# Ensure relative paths work
cd "$SLURM_SUBMIT_DIR"

# -------------------------------
# Logging setup
# -------------------------------
OUTDIR="scion_train_logs"
mkdir -p "$OUTDIR" configs/temp train_logs

stamp="$(date +'%Y%m%d-%H%M%S')_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOGFILE="${OUTDIR}/scion_${stamp}.log"

echo "[$(date)] Job started"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Learning rate: ${BASE_LR}"
echo "Working dir: $(pwd)"
echo "Logging to: ${LOGFILE}"
echo "----------------------------------------"

# -------------------------------
# Config manipulation
# -------------------------------
ORIG_CFG="./configs/scion.yaml"
NEW_CFG="./configs/temp/scion_${stamp}.yaml"

python - <<EOF
import yaml
from pathlib import Path

orig = Path("${ORIG_CFG}")
new = Path("${NEW_CFG}")

with orig.open("r") as f:
    cfg = yaml.safe_load(f)

base_lr = float("${BASE_LR}")

# Training parameters
cfg["batch_size"] = 512
cfg["device_batch_size"] = 64
cfg["lr_embed"] = base_lr
cfg["lr_matrix"] = base_lr
cfg["sequence_length"] = 256
cfg["val_loss_every"] = 500

# Training length
t_budget = 2550 * 512 * 1024
cfg["num_iterations"] = int(1024 * 2550 / cfg["sequence_length"])
cfg["warmdown_iters"] = int(0.28 * cfg["num_iterations"])

# Naming
cfg["project"] += f"-budget{t_budget}"
run = cfg.get("run", "scion")
cfg["run"] = f"{run}_bs{cfg['batch_size']}_it{cfg['num_iterations']}_lr{base_lr:.1e}"

with new.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote updated config to {new}")
EOF

echo "Using config: ${NEW_CFG}"
echo "----------------------------------------"

# -------------------------------
# Run training
# -------------------------------
torchrun --standalone --nproc_per_node=4 \
    train_gpt_scion.py \
    --config="${NEW_CFG}" \
    2>&1 | tee "$LOGFILE"

echo "----------------------------------------"
echo "[$(date)] Job finished"
