#!/usr/bin/env bash
# Run from Hate_Project on an allocated, idle V100, in memelens-nf4.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
python scripts/smolvlm/run.py cache
python scripts/smolvlm/run.py check --output results/smolvlm-environment.json
python -m unittest discover -s scripts/smolvlm -p 'test_*.py' -v
python scripts/smolvlm/run.py preflight --output results/smolvlm-preflight.json
python scripts/smolvlm/run.py smoke --mode fp16 --output results/smolvlm/smoke-fp16
python scripts/smolvlm/run.py smoke --mode nf4 --output results/smolvlm/smoke-nf4
python scripts/smolvlm/run.py train --mode fp16 --output results/smolvlm/fp16-training
python scripts/smolvlm/run.py train --mode nf4 --output results/smolvlm/nf4-training
for arm in fp16-base nf4-base fp16-lora nf4-qlora; do
    python scripts/smolvlm/run.py evaluate --arm "$arm" --output "results/smolvlm/test/$arm"
done
python scripts/smolvlm/summarize.py
