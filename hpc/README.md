# Running on the Caltech HPC cluster

The scripts here are a starting point and have not been run on the cluster yet: check the partition and GPU
names with `sinfo -o "%P %G"` and adjust the `#SBATCH` lines if needed.

## Once per person (login node)

```bash
git clone https://github.com/alexanderquispe/jlens-open-models && cd jlens-open-models
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv in your home; uv also fetches Python 3.11
uv sync                                           # .venv with the pinned versions (CUDA build of torch on Linux)
echo 'export JLENS_SHARED=/path/to/group/storage/jlens' >> ~/.bashrc   # the group's shared directory
.venv/bin/huggingface-cli login                   # your token, for gated models such as Llama 3.1
```

`JLENS_SHARED` must be a directory every member can read and write. `hpc/env.sh` puts the Hugging Face cache
there, so each model is downloaded once for the whole group, and keeps your token in your home directory.

## Once per model (for the whole group)

```bash
sbatch hpc/download.sbatch llama3.1-8b gpt2-small
```

This runs on CPU, so no GPU time is spent waiting on the network, and downloads only the weights, config and
tokenizer (for Llama 3.1, that skips Meta's 16 GB `original/` checkpoint), plus the lens. If the compute nodes
have no internet access, run the same thing on a login node:

```bash
source hpc/env.sh && HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/python -m jlens_open.download llama3.1-8b
```

## Jobs

```bash
sbatch hpc/gpu.sbatch bash run_all.sh llama3.1-8b
sbatch hpc/gpu.sbatch python experiments/02_weekday_circle/weekday_circle.py --model llama3.1-8b --task engels --position day
squeue -u $USER
tail -f logs/jlens-<jobid>.out
```

Jobs run with `HF_HUB_OFFLINE=1`: they read the shared cache and never download, so a model that is missing
fails right away instead of downloading inside a GPU job.

Interactive session on a GPU:

```bash
srun --gres=gpu:h100:1 --mem=64G --time=01:00:00 --pty bash
source hpc/env.sh && source .venv/bin/activate
```

Memory: Llama-3.1-8B in bf16 needs about 16 GB on the GPU plus 2 GB for its lens, so one H100 is plenty. Models
of 70B and more need several GPUs, which `LensModel` does not support yet (see the open tasks in the top-level
README).
