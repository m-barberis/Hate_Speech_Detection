#!/usr/bin/env bash
# Run from the project root with GPU 0 free; leave previous results untouched.
set -euo pipefail
probe_root="${1:-results/mami_lora_qlora/quick-batch2-v1}"
for probe_mode in fp16 nf4; do
  for probe_variant in cached batch2-full; do
    extra_args=(--repetitions 1)
    if [[ "$probe_variant" == batch2-full ]]; then
      extra_args+=(--stress-sample)
    fi
    CUDA_VISIBLE_DEVICES=0 nice -n 19 python -u scripts/mami_lora_qlora/speed_probe.py run \
      --mode "$probe_mode" --variant "$probe_variant" \
      --output "$probe_root/$probe_mode-$probe_variant" "${extra_args[@]}"
  done
done
nice -n 19 python scripts/mami_lora_qlora/speed_probe.py summarize \
  --root "$probe_root" --variants cached batch2-full --reference cached \
  --output "$probe_root/summary.json"
