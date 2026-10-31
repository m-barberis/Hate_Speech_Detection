# MAMI robustness adaptation: matched FP16 LoRA and NF4 QLoRA training

The user approved full-run global batch 48 and four fixed validation conditions
on 31 October 2026, after both shared-initialization v2 smoke checks passed.
No full training has been run from the local implementation environment.

## Fixed experiment

Both arms use the prepared 9,000-original training split (18,000 clean/overlay
records), two epochs, microbatch two, and accumulation 24 on GPU 0 of eslsrv13.
Each epoch has 375 optimizer updates, for exactly 750 updates per mode.
Keep the original learning rate 2e-4 and literal original LoRA configuration.
Use AdamW with zero weight decay, betas 0.9/0.999, epsilon 1e-8, norm clipping
at 1.0, and linear learning-rate decay without warm-up. These optimizer/schedule
values match the original Trainer defaults where not explicitly configured.
Each record's mean supervised-token loss receives equal weight within a
48-record update. This explicit custom-loop weighting is shared across arms;
do not claim bitwise reproduction of historical Trainer loss handling.

The nominal historical global batch is 48 only if its commented six DDP workers
actually ran; missing logs prevent confirming that. The new two-arm comparison
has a fixed measured batch irrespective of historical uncertainty.

Training reuses the validated model-loading/initialization and image-expanded
prompt masking functions from `training_smoke.py`. The smoke helper was refactored
to expose that setup without changing its base/adapter dtype policies. Both arms
retain the cached immutable public checkpoint, FP16 autocast, nonreentrant gradient
checkpointing, SDPA, fixed image budget, and process-local forced dispatch hooks.
NF4 remains broad eligible-linear weight quantization, including vision layers,
with FP16 compute and double quantization disabled. Only FP32 adapters train.

The same CPU-created shared initialization is copied and verified before either
optimizer starts. Full-run preprocessing checks successful matching v2 gates,
all prepared/supplement file hashes and all referenced image bytes. No CPU/disk
model offload or automatic image-budget reductions are allowed.

## Four validation conditions

`prepare_validation_conditions.py` creates a separate supplement with 1,000
records per condition, always the same validation originals:

- `clean_image_only.jsonl`
- `overlay_image_only.jsonl`
- `clean_with_transcription.jsonl`
- `overlay_with_transcription.jsonl`

It reuses the existing image paths and byte hashes. Transcription-present records
use the old prompt function with dropout disabled. Overlays keep the original
transcription, so they test assistance from preserved text rather than OCR run on
corrupted pixels. Training dropout and original prepared files are unchanged.
The validation grid uses the ten training emoji identities; this validation
suite does not establish transfer to new emoji identities or geometries.

After each epoch, evaluate all four conditions with greedy generation, maximum
five new tokens, KV cache, the same leading-label parser, and separate strict
format compliance. Targets are removed from input IDs and attention masks before
generation. Invalid answers remain errors. Log raw answers, token counts,
input hashes, macro-F1, accuracy, invalid counts and confusion matrices. Also
report teacher-forced loss weighted by supervised token counts. Validation
preserves training RNG states and never updates weights.

Transcription-present clean and overlay results are primary validation readouts;
image-only results diagnose behavior without supplied text. Report conditions
separately. Validation does not select checkpoints or trigger early stopping.
The model for final evaluation is always the epoch-two adapter in each arm.
No test images are loaded by the trainer.

## Server commands after syncing code

In the existing memelens-nf4 environment, from the server project directory,
create the validation supplement on CPU once:

```bash
python scripts/mami_lora_qlora/prepare_validation_conditions.py
```

Then verify the complete experiment inputs on CPU, without creating run outputs:

```bash
python scripts/mami_lora_qlora/train_comparison.py \
  --mode fp16 --preflight-only \
  --output results/mami_lora_qlora/fp16-training-v2
```

Expected preflight: 18,000 training records, 375 updates per epoch, 750 updates
total, 1,000 records in each validation condition and the verified shared initial
adapter hash. The checker hashes every distinct referenced image file.

Use a persistent server terminal and an available GPU 0, following lab allocation
rules. Run the modes sequentially, each in a fresh process:

```bash
CUDA_VISIBLE_DEVICES=0 nice -n 19 python -u scripts/mami_lora_qlora/train_comparison.py \
  --mode fp16 --output results/mami_lora_qlora/fp16-training-v2

CUDA_VISIBLE_DEVICES=0 nice -n 19 python -u scripts/mami_lora_qlora/train_comparison.py \
  --mode nf4 --output results/mami_lora_qlora/nf4-training-v2
```

The default data, validation supplement, shared initialization and successful
v2 smoke paths match the commands used in this chat. CLI options allow explicit
paths when those directories differ. No package installations are required.

Do not start the NF4 run if FP16 fails. Stop for OOM, non-finite loss/gradients,
unexpected frozen-base gradients or integrity/masking failures; review the actual
error before changing settings. Do not silently lower an image budget for one
arm. Four-record smoke checks cannot guarantee all longer full-run inputs fit.
Existing run and supplement directories are not overwritten.

## Saved evidence and recovery

Each run records `request.json`, `model_inventory.json`, `training_plan.json`,
per-record data/mask/loss logs (with microbatch indices) and `training_steps.jsonl`. Epoch directories
contain adapter safetensors/config, processor files, optimizer/scheduler/scaler
and RNG state, a hash manifest, validation predictions and epoch summaries.
Frozen embedding/head matrices are not redundantly saved, while their LoRA
parameters remain in the adapter checkpoint. The immutable public base revision
is recorded for subsequent evaluation/reloading.

Save each epoch before validation so a validation failure does not lose its
trained adapter. Training state is retained for recovery; automatic resume is
not implemented in this version. Failed output directories remain inspectable,
with `failure.json` when possible, and no successful full-run completion marker.
Do not restart or remove failed runs without inspecting what completed.

A run-level `complete.json` appears only after all 750 updates and both full
validation passes. Its final adapter path points explicitly to `epoch-2/adapter`.
Each mode performs 8,000 validation generations across two epochs plus validation
loss forwards. Allow for that work in addition to the training estimate below.

Training wall time includes preprocessing, disk access, transfer, loss logging
and optimizer work, while excluding model loading, checkpoint saving and
validation. Peak allocated memory is the maximum across training updates;
loading and validation peaks are excluded. Reserved memory retains allocator
history, which may include earlier validation. These scopes differ from the
completed generation-only inference benchmark and must be labelled separately.
GPU contention and sequential-run drift remain operator-monitored limitations.

## Implementation verification and limitations

CPU tests cover stratified split separation, deterministic fixed augmentations,
validation image reuse and transcription templates, input corruption rejection,
shared initialization, image-expanded prefix boundaries, parser behavior, epoch
coverage and learning-rate endpoints. The local runtime has no Torch/GPU stack;
the full loop, checkpoint saving and validation generation have not been executed
locally. Server smoke checks validate the reused GPU training core, not the full
new loop. Runtime assertions guard the remaining server checks.

```bash
python -m unittest discover -s scripts/mami_lora_qlora -p 'test_*.py'
```

Sources for default optimizer/scheduler and adapter serialization behavior:

- https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/training_args.py
- https://github.com/huggingface/peft/blob/v0.17.1/src/peft/peft_model.py


## Approved batch-two implementation after the server probes

The first microbatch-one FP16 run was interrupted; retain its directory and use
fresh `fp16-training-v2` and `nf4-training-v2` outputs. Never combine its partial
logs with these new matched runs. The quick server check measured cached
microbatch-one versus cached full-logit microbatch-two updates at 45.64 versus
27.82 seconds for FP16, and 49.84 versus 30.69 seconds for NF4. Four-record
relative gradient differences were 3.13% and 2.78%. Both sample memory checks
passed, but the longest checked sequence was only 521 tokens. These establish
sample feasibility, not worst-case memory or equivalent final accuracy.

The full runner now shares the probe's pair-padding and full-logit loss functions
in `training_batches.py`. It calculates all sequence logits, averages supervised
causal token losses within each record, then averages the two record losses.
Divide each pair's mean by 24 before backpropagation. Exactly the original 48
ordered records contribute to every update; no bucketing, reordering, truncation,
image-budget change, validation change or automatic batch-size fallback is added.
Both modes retain identical data order hashes, shared initialization and schedule.

One CPU worker hashes and preprocesses upcoming pairs. Keep at most two pending
or prepared pairs plus the consumer's current pair, with transient preprocessing
workspace. There is no dataset cache, pinning requirement or new dependency.
Main-thread record identity assertions guard ordering; worker exceptions propagate
and close the queue on failure. Validation starts only after the epoch's producer
has closed and continues to use single-record generation. Log each update's
largest CPU input-pair tensor bytes, duration, loss and GPU allocation/reservation.
The source hash of the shared batching helper is saved in `training_plan.json`.

CPU prefetching overlaps preprocessing rather than eliminating it; cached-probe
timings are therefore estimates, not promised full-run throughput. At those
sampled rates, training alone would take roughly 5.8 and 6.4 hours for the 750
updates. Validation, checkpointing, loading and larger examples add time.
The full-dataset memory limit remains unverified. Stop and inspect any OOM;
never silently change just one comparison arm. Do not run another standalone
probe before the approved full runs; inspect initial full-run update timings.

After syncing the full scripts folder and docs, the already-prepared data and
validation supplement do not need rebuilding. For sequential launches that stop
on the first failure and reject existing output paths:

```bash
bash scripts/mami_lora_qlora/run_matched_training.sh
```

Use an available GPU 0 and a persistent server terminal. This script uses CPU
nice 19; GPU use remains exclusive by lab allocation rather than CPU priority.
The runner still saves only at epoch boundaries and has no automatic resume.
