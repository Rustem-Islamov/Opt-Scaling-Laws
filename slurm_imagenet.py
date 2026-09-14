#!/bin/bash
#SBATCH -J imagenet-download
#SBATCH -p cscc-gpu-p          # or change to a CPU / long partition if preferred
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -o imagenet_logs/%x_%A.out
#SBATCH -e imagenet_logs/%x_%A.err


set -euo pipefail

# ----------------------------
# Conda setup
# ----------------------------
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate plainLM

# Ensure relative paths work
cd "$SLURM_SUBMIT_DIR"

# ----------------------------
# Logging
# ----------------------------
OUTDIR="imagenet_logs"
mkdir -p "$OUTDIR"

stamp="$(date +'%Y%m%d-%H%M%S')_${SLURM_JOB_ID}"
LOGFILE="${OUTDIR}/imagenet_${stamp}.log"

echo "[$(date)] ImageNet download job started"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Logging to: ${LOGFILE}"
echo "----------------------------------------"

# ----------------------------
# (Optional but STRONGLY recommended)
# Put Hugging Face cache on a large disk
# ----------------------------
# export HF_HOME=/path/to/big_disk/huggingface

export HF_HOME=/scratch/$USER/huggingface
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN before running this job}

# ----------------------------
# Run download
# ----------------------------
python imagenet_download.py 2>&1 | tee "$LOGFILE"

echo "----------------------------------------"
echo "[$(date)] ImageNet download job finished"
