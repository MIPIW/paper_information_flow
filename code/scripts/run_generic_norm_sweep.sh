#!/bin/bash
# Cross-family residual-norm profiling sweep (CPU), sequential to avoid
# thread contention. Continues past any single model's failure (e.g. gated
# repo without approved access) so one bad entry doesn't kill the batch.
set -uo pipefail
PY=/home/hyohyeongjang/.conda/envs/commonRegion/bin/python
mkdir -p logs results
export OMP_NUM_THREADS=40

declare -A MODELS=(
  [qwen2.5-0.5b]="Qwen/Qwen2.5-0.5B"
  [qwen2.5-1.5b]="Qwen/Qwen2.5-1.5B"
  [gemma-2b]="google/gemma-2b"
  [gemma2-2b]="google/gemma-2-2b"
  [tinyllama-1.1b]="TinyLlama/TinyLlama_v1.1"
  [llama3.2-1b]="meta-llama/Llama-3.2-1B"
  [tinymistral-248m]="Locutusque/TinyMistral-248M-v2"
  [minueza-32m]="Felladrin/Minueza-32M-Base"
)

for key in qwen2.5-0.5b qwen2.5-1.5b gemma-2b gemma2-2b tinyllama-1.1b llama3.2-1b tinymistral-248m minueza-32m; do
  name="${MODELS[$key]}"
  echo "=== $key ($name) ==="
  $PY scripts/generic_norm_profile.py --model-name "$name" \
      --languages en,zh --n-examples-per-lang 10 --max-length 48 \
      --out "results/norm_profile_${key}.json" \
      > "logs/norm_profile_${key}.log" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "  FAILED (exit $status) -- see logs/norm_profile_${key}.log"
  else
    echo "  OK -- results/norm_profile_${key}.json"
  fi
done
echo "ALL_DONE"
