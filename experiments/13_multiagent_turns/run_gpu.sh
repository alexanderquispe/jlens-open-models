#!/bin/bash
# The GPU steps of experiment 13, in order. Each one skips what is already done, so the job can be resubmitted.
#   sbatch --gres=gpu:nvidia_h200:1 --mem=160G --time=03:00:00 hpc/gpu.sbatch bash experiments/13_multiagent_turns/run_gpu.sh
set -uo pipefail
export PYTHONPATH=.
PY=.venv-tf5/bin/python
CALLS=/resnick/scratch/aquisper/mamu_pilot/calls
ACTS=/resnick/scratch/aquisper/mamu_pilot/acts
# 1. The lens on the reference prompt sets (experiment 01), unless it has run
[ -f experiments/01_lens_eval/results/qwen3.8-27b.csv ] || $PY experiments/01_lens_eval/lens_eval.py --model qwen3.8-27b
# 2. Replay every extracted episode
for d in $CALLS/*/; do
  $PY experiments/13_multiagent_turns/replay.py $d $ACTS/$(basename $d)
done
