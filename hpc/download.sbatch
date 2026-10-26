#!/bin/bash
#SBATCH --job-name=jlens-download
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Downloads models and their lenses once into the shared cache, on CPU (no GPU hours spent on the network).
# Usage, from the repo root: sbatch hpc/download.sbatch llama3.1-8b gpt2-small
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
export PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=1
python -m jlens_open.download "$@"
