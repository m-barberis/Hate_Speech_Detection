#!/usr/bin/env bash
# Run manually on the allocated V100. Reuse completed FP16; restart NF4 only.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
original_root="${1:-results/smolvlm-optimized-v1}"
recovery_root="${2:-$original_root/nf4-recovery-mb1-v1}"
if [[ -e "$recovery_root" ]]; then
    echo "Preserve existing output: $recovery_root. Pass a new recovery directory as the second argument." >&2
    exit 1
fi
# Read-only checkpoint/file verification before creating any recovery output.
python scripts/smolvlm/run.py verify-training --mode fp16 \
    --fp16-training "$original_root/fp16-training"
mkdir -p "$recovery_root"
flags=(--microbatch 1 --cpu-threads 2)
python scripts/smolvlm/run.py profile --mode nf4 "${flags[@]}" \
    --output "$recovery_root/profile-nf4"
# Tiny diagnostic checks in both formats; no FP16 full training is repeated.
for mode in fp16 nf4; do
    python scripts/smolvlm/run.py smoke --mode "$mode" "${flags[@]}" \
        --output "$recovery_root/smoke-$mode"
done
python scripts/smolvlm/run.py verify-recovery "${flags[@]}" \
    --fp16-training "$original_root/fp16-training" \
    --smoke-fp16 "$recovery_root/smoke-fp16" --smoke-nf4 "$recovery_root/smoke-nf4" \
    --allow-execution-differences --output "$recovery_root/recovery-check.json"
python scripts/smolvlm/run.py train --mode nf4 "${flags[@]}" \
    --smoke-fp16 "$recovery_root/smoke-fp16" --smoke-nf4 "$recovery_root/smoke-nf4" \
    --output "$recovery_root/nf4-training"
for arm in fp16-base nf4-base fp16-lora nf4-qlora; do
    python scripts/smolvlm/run.py evaluate --arm "$arm" --allow-execution-differences \
        --fp16-training "$original_root/fp16-training" \
        --nf4-training "$recovery_root/nf4-training" --output "$recovery_root/test/$arm"
done
python scripts/smolvlm/summarize.py --root "$recovery_root/test" \
    --output "$recovery_root/test/summary.json"
