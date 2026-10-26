#!/usr/bin/env bash
# Both experiments, every configuration, for one model. Run from anywhere, with the environment active.
# Usage: bash run_all.sh gpt2-small|llama3.1-8b   (Llama 3.1 8B is gated: accept Meta's license on Hugging Face first)
set -euo pipefail
cd "$(dirname "$0")"
model="${1:?usage: bash run_all.sh gpt2-small|llama3.1-8b}"
PY="${PYTHON:-python}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p logs
"$PY" experiments/01_lens_eval/lens_eval.py --model "$model" 2>&1 | tee "logs/lens_eval_$model.log"
for cfg in "sequence last" "sequence day" "engels day" "engels last"; do
  set -- $cfg
  "$PY" experiments/02_weekday_circle/weekday_circle.py --model "$model" --task "$1" --position "$2" 2>&1 \
    | tee "logs/weekday_circle_${model}_$1_$2.log"
done
