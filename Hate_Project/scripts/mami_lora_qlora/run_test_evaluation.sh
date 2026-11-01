#!/usr/bin/env bash
# Run from the project root, in memelens-nf4, with GPU 0 free.
set -euo pipefail
test_data="${1:-data/processed/mami_lora_qlora_test_seed42}"
test_output="${2:-results/mami_lora_qlora/test-1000-v1}"
for test_arm in fp16-base nf4-base fp16-lora nf4-qlora; do
  CUDA_VISIBLE_DEVICES=0 nice -n 19 python -u scripts/mami_lora_qlora/evaluate_test.py run \
    --arm "$test_arm" --data "$test_data" --output-root "$test_output" --reuse-complete
done
if [[ ! -e "$test_output/comparison.json" ]]; then
  nice -n 19 python scripts/mami_lora_qlora/evaluate_test.py summarize \
    --data "$test_data" --output-root "$test_output"
fi
