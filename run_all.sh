#!/usr/bin/env bash
# Every experiment, every configuration, for one model. Run from anywhere, with the environment active.
# Usage: bash run_all.sh <model>   (any key of jlens_open.MODELS; Llama 3.1 is gated: accept Meta's license on
# Hugging Face first). LAYERS=::2 patches every second layer in experiments 04 and 05 (default: all layers).
set -euo pipefail
cd "$(dirname "$0")"
model="${1:?usage: bash run_all.sh <model>}"
PY="${PYTHON:-python}"
LAYERS="${LAYERS:-::1}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p logs
"$PY" experiments/01_lens_eval/lens_eval.py --model "$model" 2>&1 | tee "logs/lens_eval_$model.log"
for cfg in "sequence last" "sequence day" "engels day" "engels last"; do
  set -- $cfg
  "$PY" experiments/02_weekday_circle/weekday_circle.py --model "$model" --task "$1" --position "$2" 2>&1 \
    | tee "logs/weekday_circle_${model}_$1_$2.log"
done
"$PY" experiments/03_jspace/jspace.py --model "$model" 2>&1 | tee "logs/jspace_$model.log"
"$PY" experiments/04_multihop_swap/multihop_swap.py --model "$model" --layers "$LAYERS" 2>&1 | tee "logs/multihop_swap_$model.log"
for cfg in "months start" "days start" "months last"; do
  set -- $cfg
  "$PY" experiments/05_cyclic_irreps/cyclic_irreps.py --model "$model" --concept "$1" --position "$2" --layers "$LAYERS" 2>&1 \
    | tee "logs/cyclic_irreps_${model}_$1_$2.log"
done
"$PY" experiments/06_false_context/false_context.py --model "$model" 2>&1 | tee "logs/false_context_$model.log"
