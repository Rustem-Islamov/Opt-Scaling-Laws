#!/bin/bash
#SBATCH -J fineweb100B_cache
#SBATCH -o train_logs/fineweb100B_cache_%j.out
#SBATCH -e train_logs/fineweb100B_cache_%j.err

#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00

# Activate conda
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate plainLM

# Authenticate with Hugging Face Hub (avoids rate limits / slow downloads)
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)"

# Run the script
# Default downloads 407 chunks (~40B tokens, ~75GB). Pass a number as
# an argument to download fewer/more chunks, e.g. sbatch run_fineweb100B.py 100
python data/cached_fineweb100B.py "$@"
