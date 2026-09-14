#!/bin/bash
#SBATCH -J fineweb_cache
#SBATCH -o train_logs/fineweb_cache_%j.out
#SBATCH -e train_logs/fineweb_cache_%j.err

#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00

# Activate conda
eval "$(conda shell.bash hook)"
conda activate plainLM

# Run the script
python data/cached_fineweb10B.py 
