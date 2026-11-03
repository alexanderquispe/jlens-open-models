#!/bin/bash
#SBATCH --job-name=jlens-cpu
#SBATCH --account=safety_first
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Runs any command on 32 CPU cores, no GPU (simulations that need no model). Usage, from the repo root:
#   sbatch hpc/cpu.sbatch python experiments/07_jspace_recovery/occupancy_theory.py --part reduced
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  # one thread per worker process
"$@"
