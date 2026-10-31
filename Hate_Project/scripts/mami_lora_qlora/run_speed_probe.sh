#!/usr/bin/env bash
# Run manually from the project root in memelens-nf4, only once GPU 0 is free.
set -euo pipefail
probe_root="${1:-results/mami_lora_qlora/speed-probe-v1}"
for probe_mode in fp16 nf4; do
  for probe_variant in baseline cached answer-only batch2; do
    CUDA_VISIBLE_DEVICES=0 nice -n 19 python -u scripts/mami_lora_qlora/speed_probe.py run \
      --mode "$probe_mode" \
      --variant "$probe_variant" \
      --output "$probe_root/$probe_mode-$probe_variant"
  done
done
nice -n 19 python scripts/mami_lora_qlora/speed_probe.py summarize \
  --root "$probe_root" --output "$probe_root/summary.json"
