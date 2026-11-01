#!/usr/bin/env bash
# Run from Hate_Project on an allocated, idle V100, in memelens-nf4.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
run_root="${1:-results/smolvlm-optimized-v1}"
if [[ -e "$run_root" ]]; then
    echo "Preserve existing output: $run_root. Pass a new directory as the first argument." >&2
    exit 1
fi
training_flags=(--microbatch "${SMOLVLM_MICROBATCH:-2}" --cpu-threads "${SMOLVLM_CPU_THREADS:-2}")
case "${SMOLVLM_CHECKPOINTING:-off}" in
    on) training_flags+=(--gradient-checkpointing) ;;
    off) ;;
    *) echo 'SMOLVLM_CHECKPOINTING must be on or off' >&2; exit 1 ;;
esac
mkdir -p "$run_root"
python scripts/smolvlm/run.py cache
python scripts/smolvlm/run.py check --output "$run_root/environment.json"
python -m unittest discover -s scripts/smolvlm -p 'test_*.py' -v
python scripts/smolvlm/run.py preflight "${training_flags[@]}" --output "$run_root/preflight.json"
for mode in fp16 nf4; do
    python scripts/smolvlm/run.py profile --mode "$mode" "${training_flags[@]}" --output "$run_root/profile-$mode"
    python scripts/smolvlm/run.py smoke --mode "$mode" "${training_flags[@]}" --output "$run_root/smoke-$mode"
done
for mode in fp16 nf4; do
    python scripts/smolvlm/run.py train --mode "$mode" "${training_flags[@]}" \
        --smoke-fp16 "$run_root/smoke-fp16" --smoke-nf4 "$run_root/smoke-nf4" \
        --output "$run_root/$mode-training"
done
for arm in fp16-base nf4-base fp16-lora nf4-qlora; do
    python scripts/smolvlm/run.py evaluate --arm "$arm" \
        --cpu-threads "${SMOLVLM_CPU_THREADS:-2}" \
        --fp16-training "$run_root/fp16-training" --nf4-training "$run_root/nf4-training" \
        --output "$run_root/test/$arm"
done
python scripts/smolvlm/summarize.py --root "$run_root/test" --output "$run_root/test/summary.json"
