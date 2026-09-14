#!/bin/bash
#SBATCH -J sciontrace-1gpu
#SBATCH -o %x_%A.out
#SBATCH -e %x_%A.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=35G
#SBATCH --time=12:00:00
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

# --- Run training ---
torchrun --standalone --nproc_per_node=1 \
  train_gpt_sciontrace.py --config=./configs/scion_trace.yaml \
  2>&1 | tee "$LOGFILE"

echo "----------------------------------------"
echo "[$(date)] Job finished"
