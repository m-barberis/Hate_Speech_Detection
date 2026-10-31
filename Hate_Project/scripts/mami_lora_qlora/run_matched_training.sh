#!/usr/bin/env bash
# Run manually from the project root in memelens-nf4, with GPU 0 available.
set -euo pipefail
for training_mode in fp16 nf4; do
  training_output="results/mami_lora_qlora/$training_mode-training-v2"
  if [[ -e "$training_output" ]]; then
    echo "Output already exists; preserve it and inspect before retrying: $training_output" >&2
    exit 1
  fi
done
for training_mode in fp16 nf4; do
  CUDA_VISIBLE_DEVICES=0 nice -n 19 python -u scripts/mami_lora_qlora/train_comparison.py \
    --mode "$training_mode" \
    --output "results/mami_lora_qlora/$training_mode-training-v2"
done
