#!/usr/bin/env bash
# Finish the original grid evaluation first. Run with GPU 0 available.
set -euo pipefail
pattern_data="${1:-data/processed/mami_unseen_patterns_seed42}"
pattern_output="${2:-results/mami_lora_qlora/unseen-patterns-v1}"
clean_output="${3:-results/mami_lora_qlora/test-1000-v1}"
for pattern_arm in fp16-base nf4-base fp16-lora nf4-qlora; do
  if [[ ! -f "$clean_output/$pattern_arm/complete.json" ]]; then
    echo "Finish the original clean/grid evaluation first: $clean_output/$pattern_arm/complete.json missing" >&2
    exit 1
  fi
done
for pattern_arm in fp16-base nf4-base fp16-lora nf4-qlora; do
  CUDA_VISIBLE_DEVICES=0 nice -n 19 python -u scripts/mami_lora_qlora/evaluate_unseen_patterns.py run \
    --arm "$pattern_arm" --data "$pattern_data" --output-root "$pattern_output" --reuse-complete
 done
if [[ ! -e "$pattern_output/comparison.json" ]]; then
  nice -n 19 python scripts/mami_lora_qlora/evaluate_unseen_patterns.py summarize \
    --data "$pattern_data" --output-root "$pattern_output" --clean-results "$clean_output"
fi
