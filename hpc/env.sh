# Environment for jobs on the cluster. Sourced by hpc/*.sbatch; also useful in an interactive session.
# JLENS_SHARED: a directory in the group's storage that everyone can read and write, so that each model
# and lens is downloaded once for everyone. Set it once in ~/.bashrc (or replace the default below).
export JLENS_SHARED="${JLENS_SHARED:-/path/to/group/storage/jlens}"
if [[ "$JLENS_SHARED" == /path/to/* ]]; then
  echo "Set JLENS_SHARED to the group's shared directory (see hpc/env.sh)" >&2
  return 1 2>/dev/null || exit 1
fi
umask 002                                              # files in the shared cache stay writable by the group
export HF_HOME="$JLENS_SHARED/hf_cache"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"  # your HF token stays in your home, not in the shared cache
export HF_HUB_OFFLINE=1                                # jobs only read the cache; download.sbatch turns this off
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
