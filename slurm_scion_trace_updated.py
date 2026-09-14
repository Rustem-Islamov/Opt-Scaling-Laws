#!/bin/bash
#SBATCH -J sciontrace-2-512
#SBATCH -o train_logs/%x_%A.out
#SBATCH -e train_logs/%x_%A.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=39G
#SBATCH --time=23:55:00
#SBATCH --partition=a100,a100-80g
#SBATCH --qos=gpu1day
#SBATCH --gres=gpu:1

set -euo pipefail

# --- Environment setup ---
eval "$(conda shell.bash hook)"
conda activate plainLM310

# Ensure relative paths work
cd "$SLURM_SUBMIT_DIR"

# --- Logging setup ---
OUTDIR="train_logs"
mkdir -p "$OUTDIR"

stamp="$(date +'%Y%m%d-%H%M%S')_${SLURM_JOB_ID}"
LOGFILE="${OUTDIR}/sciontrace_${stamp}.log"

echo "[$(date)] Job started"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Working dir: $(pwd)"
echo "Logging to: ${LOGFILE}"
echo "----------------------------------------"

# --- Config manipulation ---
ORIG_CFG="./configs/scion_trace.yaml"
NEW_CFG="./configs/temp/scion_trace_${stamp}.yaml"

python - <<EOF
import yaml
from pathlib import Path

orig = Path("${ORIG_CFG}")
new = Path("${NEW_CFG}")

with orig.open("r") as f:
    cfg = yaml.safe_load(f)

# Update requested fields
cfg["trace_m"] = 2
cfg["batch_size"] = 512

# Update run name
run = cfg.get("run", "sciontrace")
cfg["run"] = f"{run}_{cfg['trace_m']}_{cfg['batch_size']}"

# Write new config
with new.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote updated config to {new}")
EOF

echo "Using config: ${NEW_CFG}"
echo "----------------------------------------"

# --- Run training ---
torchrun --standalone --nproc_per_node=1 \
    train_gpt_sciontrace_updated.py \
    --config="${NEW_CFG}" \
    2>&1 | tee "$LOGFILE"

echo "----------------------------------------"
echo "[$(date)] Job finished"
