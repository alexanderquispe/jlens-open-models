#!/bin/bash
#SBATCH --job-name=jlens
#SBATCH --account=safety_first
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Runs any command on one H100. Usage, from the repo root:
#   sbatch hpc/gpu.sbatch bash run_all.sh llama3.1-8b
#   sbatch hpc/gpu.sbatch python experiments/02_weekday_circle/weekday_circle.py --model llama3.1-8b --task engels --position day
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
export PATH="$PWD/.venv/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$@"
