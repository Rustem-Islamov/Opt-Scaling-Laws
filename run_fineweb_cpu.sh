#!/bin/bash
#SBATCH -J fineweb_cache
#SBATCH -o fineweb_cache_%j.out
#SBATCH -e fineweb_cache_%j.err

#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

eval "$(conda shell.bash hook)"
conda activate plainLM

cd "$SLURM_SUBMIT_DIR"

python data/cached_fineweb100B.py

echo "Job finished at $(date)"