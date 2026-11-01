#!/usr/bin/env bash
# Run manually in memelens-nf4 with physical GPU 0 free; no training is launched.
set -euo pipefail
sweep_data="${1:-data/processed/mami_patterns_100_all20_seed42}"
sweep_output="${2:-results/mami_lora_qlora/patterns-100-all20-v1}"
clean_output="${3:-results/mami_lora_qlora/test-1000-v1}"
for sweep_arm in fp16-base nf4-base fp16-lora nf4-qlora; do
  if [[ ! -f "$clean_output/$sweep_arm/complete.json" ]]; then
    echo "Completed clean/grid evaluation required: $clean_output/$sweep_arm/complete.json" >&2
    exit 1
  fi
done
for sweep_arm in fp16-base nf4-base fp16-lora nf4-qlora; do
  CUDA_VISIBLE_DEVICES=0 nice -n 19 python -u scripts/mami_lora_qlora/evaluate_emoji_sweep.py run \
    --arm "$sweep_arm" --data "$sweep_data" --output-root "$sweep_output" --reuse-complete
done
nice -n 19 python scripts/mami_lora_qlora/evaluate_emoji_sweep.py summarize \
  --data "$sweep_data" --output-root "$sweep_output" --clean-results "$clean_output"
