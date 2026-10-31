# MAMI LoRA and QLoRA training feasibility gate on the V100

The prepared-data hashes and server imports have passed operator checks:
18,000 training records, 1,000 clean validation records, 1,000 overlay validation
records. Server versions are PyTorch 2.5.1, Transformers 4.57.1, Accelerate 1.10.1,
bitsandbytes 0.42.0, and PEFT 0.17.1. Actual model training remains unverified.

## Scope

`scripts/mami_lora_qlora/training_smoke.py` runs four fixed training-only
microbatches and one disposable AdamW optimizer step. It loads the same pinned
public checkpoint as the inference follow-up, in either FP16 or NF4. No model
adapter is saved. This is a compatibility and gradient-flow check, not training
of a model to be evaluated or evidence of converged quality.

All runs use physical GPU 0, exposed as logical CUDA device 0. They run in fresh
sequential processes on an otherwise available V100. The operator follows the
lab's allocation rules. No setup, driver, container, or package changes occur.

## Reused settings and implementation changes

- Read rank, alpha, dropout, target modules and learning rate directly from the
  literal config in the original `scripts/mami_finetuning/train_finetuning.py`.
  Record that source's hash and the actual modules matched at runtime.
- Load JSONL with ordinary Python; the Hugging Face datasets package is not
  needed. Training examples and their existing prompts are unchanged.
- Use the existing pixel budget (524,288 maximum, 4,096 minimum), SDPA, FP16
  autocast, no offload, safetensors only, cached checkpoint revision
  `a633183849df60a37251d29c3f1b0d379ae801f5`, and process-local forced dispatch
  hooks in both arms. NF4 uses eligible linear modules including vision layers,
  FP16 compute, and no double quantization.
- Freeze the base explicitly and train only FP32 LoRA parameters, as in the old
  adapter stability cast. Preserve the base's loaded dtypes. Do not blindly call
  PEFT's k-bit preparation helper on both arms: it upcasts nonquantized FP16
  parameters to FP32 and could turn the FP16 baseline into an FP32 model.
- Enable nonreentrant gradient checkpointing in both arms, disable the KV cache
  for training, and reset RNG seeds before adapter injection and before the
  microbatches. Compare initial adapter hashes to confirm matched initialization.
- Correct prompt masking: process the same image with prompt-only and complete
  messages, verify exact token-prefix equality after image-token expansion, then
  mask prompt and padding tokens. Decode/log only the supervised label suffix.
- Prefer a clean/overlay pair per class with differing transcription availability,
  chosen deterministically from training records. Never use test or validation
  examples to tune this gate.
- Accumulate four equally weighted losses for this one smoke optimizer step.
  This is a diagnostic batch of four, not a decision about full-run effective
  batch size. Use the original learning rate, AdamW with zero weight decay, a
  fixed initial gradient scale of 1024, and gradient-norm clipping at 1.0.

## Server commands

After syncing the new code, run from the server project root in memelens-nf4:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/mami_lora_qlora/training_smoke.py \
  --mode fp16 \
  --data data/processed/mami_lora_qlora_seed42 \
  --output results/mami_lora_qlora/smoke-fp16-v1

CUDA_VISIBLE_DEVICES=0 python scripts/mami_lora_qlora/training_smoke.py \
  --mode nf4 \
  --data data/processed/mami_lora_qlora_seed42 \
  --output results/mami_lora_qlora/smoke-nf4-v1
```

Stop at the first failure and inspect the error. Do not start the second command
if the first fails. Existing output directories are never overwritten. A complete
marker appears only after finite losses, finite nonzero adapter gradients,
frozen-base checks, a successful optimizer update, and finite updated parameters.
OOM, unsupported operations, prefix mismatch or numerical failures require review
before retrying with a changed configuration. Four examples cannot establish
memory feasibility for every longer training input.

## Evidence to inspect

Each output has request.json (settings/source/data hashes), model_inventory.json
(converted and adapted modules, trainable parameters, initial adapter hash),
microbatches.jsonl (input hashes, masks/label suffixes, losses and diagnostic
forward/backward durations), and complete.json on success. Timing includes cold
training effects and must not be reported as an efficiency benchmark.

Between arms compare selected examples, input signatures, initial adapter hashes,
adapter names/shapes/counts, processor settings and checkpoint/software identity.
Review trainable parameters without gradients rather than silently assuming
every adapter was exercised.

## Full training remains a separate next stage

No full-training launch command exists yet. The agreed schedule is two epochs,
with final-epoch evaluation in both arms. Full-run effective batch size must be
made explicit: the old script specified microbatch one and accumulation eight
per DDP worker, while its comments describe six workers. That implies a nominal
global batch of 48 only if six workers actually ran, which missing logs cannot
confirm. A single-GPU accumulation of eight is not automatically equivalent.

Transcription-present validation files are still absent. The recommended four
fixed validation conditions need confirmation/implementation before reporting
that validation covers both prompt modes. Existing augmentation does not need to
be repeated. No prepared records, splits or completed ten-token inference results
are modified by this gate.

Local checks run without GPU libraries:

```bash
python -m unittest discover -s scripts/mami_lora_qlora -p 'test_*.py'
```
