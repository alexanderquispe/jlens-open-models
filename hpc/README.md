# Running on the Caltech HPC cluster

Jobs are charged to the Slurm account `safety_first` and GPU jobs run in the `gpu` partition (the default
partition, `expansion`, has no GPUs); both are set in the `#SBATCH` lines. `download.sbatch` has been run on the
cluster; `gpu.sbatch` has not yet.

## Where things live

| What | Where |
|---|---|
| Models and lenses (Hugging Face cache, shared) | `/resnick/groups/Safety_First/jlens` (`$JLENS_SHARED`) |
| Your clone, `.venv`, logs | `/resnick/groups/Safety_First/$USER/jlens-open-models` |
| uv cache, temporary files | `/resnick/scratch/$USER` (files not accessed in 14 days are purged) |
| Hugging Face token | `~/.cache/huggingface/token` (private to you) |

Home has a 60 GB quota, so keep models and environments out of it. Group storage is free up to 20 TB.

## Once per person (login node)

```bash
cat >> ~/.bashrc <<'EOF'
export JLENS_SHARED=/resnick/groups/Safety_First/jlens
export UV_CACHE_DIR=/resnick/scratch/$USER/uv_cache
unset SSL_CERT_DIR   # the system profile sets it to a file, and uv then fails with "invalid peer certificate"
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
EOF
source ~/.bashrc && mkdir -p /resnick/scratch/$USER
cd /resnick/groups/Safety_First/$USER
git clone git@github.com:alexanderquispe/jlens-open-models.git && cd jlens-open-models
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv in your home; uv also fetches Python 3.11
uv sync                                           # .venv with the pinned versions (CUDA build of torch on Linux)
.venv/bin/huggingface-cli login                   # your token, for gated models such as Llama 3.1
```

Run `huggingface-cli login` in a real terminal (VS Code or ssh): it reads the token interactively. For pushing,
add an SSH key from the cluster (`ssh-keygen -t ed25519`) to your GitHub account.

`JLENS_SHARED` must be a directory every member can read and write. `hpc/env.sh` puts the Hugging Face cache
there, so each model is downloaded once for the whole group, and keeps your token in your home directory.

## Once per model (for the whole group)

```bash
sbatch hpc/download.sbatch llama3.1-8b gpt2-small
```

This runs on CPU, so no GPU time is spent waiting on the network, and downloads only the weights, config and
tokenizer (for Llama 3.1, that skips Meta's 16 GB `original/` checkpoint), plus the lens. Compute nodes have
internet access; we measured 20-40 MB/s per job, so an 8B model takes about 10 minutes. Models download one
after another within a job: for several large models, submit one job per model. To see what is already
there: `ls $JLENS_SHARED/hf_cache/hub`. The same thing also runs on a login node:

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
srun -A safety_first -p gpu --gres=gpu:h100:1 -c 8 --mem=64G --time=01:00:00 --pty bash
source hpc/env.sh && source .venv/bin/activate
```

An interactive session is billed until you `exit`, even while the GPU sits idle.

Memory: Llama-3.1-8B in bf16 needs about 16 GB on the GPU plus 2 GB for its lens, so one H100 is plenty. The
32B models need one H200 and more host memory than the default 64 GB to load:

```bash
sbatch --gres=gpu:nvidia_h200:1 --mem=192G hpc/gpu.sbatch env PYTHON=.venv-tf5/bin/python PYTHONPATH=. bash run_all.sh qwen3-32b
```

Larger models need several GPUs. When a job sees more than one GPU, `LensModel` splits the blocks across them
(the first GPU takes half a share, since it also holds the lenses, which are float32, and the activations the
experiments keep) and returns everything on the first, so the experiments run unchanged. Llama-3.3-70B-Instruct
(141 GB in bf16, 21 GB of lens) on two H200s:

```bash
sbatch --time=16:00:00 --gres=gpu:nvidia_h200:2 --mem=256G hpc/gpu.sbatch bash run_all.sh llama3.3-70b-it
```

The blocks run one after another, so two GPUs are not faster than one; they only add memory.

## GPUs and cost

| `--gres` | Memory | Compute units per hour |
|---|---|---|
| `gpu:nvidia_l40s:1` | 48 GB | 61 |
| `gpu:h100:1` | 80 GB | 120 |
| `gpu:nvidia_h200:1` | 141 GB | 156 |

Each CPU core adds 1 unit per hour; memory is free. A unit costs $0.012 up to $6,400 of group spending per
fiscal year, less above that ([rates](https://www.hpc.caltech.edu/docs/rates.html)), so `gpu.sbatch` (one H100,
8 cores) costs about $1.54 per hour of actual run time. Group usage so far:

```bash
sreport -t hours -T billing cluster AccountUtilizationByUser accounts=safety_first start=2026-10-01 end=now
```
