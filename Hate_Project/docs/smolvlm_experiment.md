# SmolVLM-256M: matched NF4, LoRA and QLoRA experiment

Approved on 1 November 2026: use a generative SmolVLM, at most three epochs,
validate every 75 optimizer updates, early stopping after three consecutive
checks without improvement, select/restore the best validation checkpoint.
Environment compatibility must be checked before training. Existing MemeLens
code and outputs remain unchanged.

The user approved local performance changes after observing 84–91 seconds per
update. **Do not connect to the server through SSH, scp, rsync, or another remote
access method without first asking the user.** The optimized implementation is
verified locally; the previous server reports below describe the original code.

## Question and interpretation

Can a compact general VLM learn MAMI misogyny classification and emoji-overlay
robustness, and how does NF4 affect quality, memory and runtime before and after
adapter training? This model has not inherited MemeLens's meme-specific training.
There is no agreed accuracy threshold or claim that it will match MemeLens.
Report actual macro-F1/accuracy, invalid answers, confusion matrices, strict
format compliance, peak memory, runtime and training steps. Quality improvements
support task adaptation; measured FP16/NF4 differences describe the quantization
tradeoff. Lower memory does not imply faster execution.

## Immutable model and software

`HuggingFaceTB/SmolVLM-256M-Instruct`
at `7e3e67edbbed1bf9888184d9df282b700a323964`, 256,484,928 total parameters.
Uses `Idefics3ForConditionalGeneration` and `Idefics3Processor` in the existing
Transformers 4.57.1 installation. Only pinned safetensors weights are used;
remote model code is disabled. GPU runs require one visible V100 on physical
GPU 0, native sm_70 support, no competing compute processes and no offload.
Do not change packages just to bypass a failed check.

The native processor uses longest edge 2048, image splitting enabled, 512-pixel
crops, and 64 visual tokens per crop. Retain those defaults identically in all
four arms. These are SmolVLM settings, not the old Qwen pixel budget. Crop count
varies by image; both crop tensors and pixel masks are padded across records.
Inputs over 8192 tokens fail rather than silently truncate or change resolution.

The pinned native chat template expects list-valued text content. Preserve the
original system instruction verbatim at the start of the user text, followed by
the original user prompt. The supervised assistant label is a text content item.
Check exact prompt-prefix equality after visual-token expansion, mask prompt and
padding, and decode the supervised suffix to verify the label is really present.
Generation removes the target and labels, uses greedy decoding, KV cache, and
at most five new tokens. Invalid answers remain errors using the existing parser.

## Data and training

Reuse the immutable prepared data, not fresh splits or augmented images:

- `data/processed/mami_lora_qlora_seed42`: 9000 originals / 18000 training records.
- `data/processed/mami_lora_qlora_validation_seed42`: four conditions of 1000 each.
- `data/processed/mami_lora_qlora_test_seed42`: four conditions of 1000 each.

Clean/overlay crossed with image-only/original transcription are reported
separately. The stopping score is the equally weighted mean of the two
transcription-present validation macro-F1 scores. The original transcription
is preserved under overlays. Test data is loaded only by the evaluation command;
no test metric selects the recipe or a checkpoint.

Training: seed 42, microbatch 2, accumulation 24, effective batch 48,
375 updates per epoch, at most 1125 updates. Microbatch remains 2 by default;
explicit `--microbatch 4` or `--microbatch 8` derives accumulation 12 or 6,
respectively, preserving the same 48 ordered records and equal per-record loss
weight in every update. These larger settings require new FP16/NF4 smoke gates.
LoRA rank 16, alpha 32, dropout
0.05, AdamW learning rate 2e-4, weight decay 0, default betas/epsilon, gradient
clipping 1, FP16 autocast and FP32 adapters. Freeze base weights without upcasting
the whole base to FP32. Gradient checkpointing is disabled by default to avoid
activation recomputation. `--gradient-checkpointing` explicitly enables the
previous nonreentrant policy when memory requires it; apply the same setting to
both arms and regenerate their gates. No automatic OOM fallback. The learning
rate decays linearly over the maximum 1125 steps without warmup.

Target existing attention/MLP projections in text and vision plus embedding/head:
q_proj, k_proj, v_proj, o_proj, out_proj, gate_proj, up_proj, down_proj, fc1, fc2,
lm_head, embed_tokens. Record the actual matched modules. `out_proj` covers
SmolVLM's vision attention output. Initialize adapter tensors from sorted CPU
FP32 arrays identically in both arms; verify names/shapes, hashes and inputs.
NF4 converts eligible linear weights including vision, uses FP16 compute and
disables double quantization. Embeddings, norms and other excluded tensors are
not claimed to be four-bit. Keep adapters unmerged.

Validate **all four full conditions every 75 optimizer steps**, five checks per
epoch. Strict score improvement resets patience; ties count as no improvement.
Three consecutive non-improving checks stop the run; the earliest possible stop
is step 300. Save every improving adapter, preserve validation predictions at
every check, restore the best adapter, and verify its canonical tensor hash.
No epoch-end duplicate validation and no fixed epoch-two selection. Each arm
may stop at a different step. Report this when comparing runtime and quality.
The full validation grid can be a substantial part of runtime.

The native image processor runs once per record. A private shallow processor copy
owns a per-record cache; prompt and labeled calls still use native tokenization
and image-token expansion. Exact prefix/mask checks and input signatures remain.
The cache is discarded with the record; it is not a huge in-memory dataset cache.

One CPU producer prepares at most two pending microbatches plus the consumer.
It closes and joins every 75 optimizer steps **before** validation starts, so
validation never shares its processor with a live producer. No reordering,
augmentation change, cropping reduction or token truncation is introduced.
CPU tensor operations use two threads by default (`--cpu-threads` overrides).
This also prevents high thread counts from competing with the prefetch worker.

Update logs separately report CPU wait, overlapped CPU preprocessing, GPU transfer,
forward and backward time, optimizer time, total update wall time and peak memory.
The overlapped times must not be added together as if they were serial. Run-level
training wall time includes producer startup/shutdown and occupancy guards.
Do not attribute all cross-model speed differences to parameter count or NF4.

## Compatibility and execution gates

1. `check`: package/model-class inventory, actual cached processor, synthetic
   prompt/suffix masking and two-record batch, CUDA architecture and occupancy.
   It does not load model weights or certify GPU training. The readiness flag
   is a preliminary inventory, not a passed training gate.
2. CPU tests: stopping/ties/reset behavior, monitor conditions, order, actual
   cached processor, small Idefics3 forward/backward with variable crop counts,
   frozen-base gradients, adapter save/restore and tamper detection.
3. Fresh-process FP16 and NF4 `smoke`: four fixed training-only records, two
   microbatches, one optimizer update, finite losses/gradients/tensors and a
   target-free generation. Compare initialization, inputs, inventory, versions
   and GPU identity before full training. This is not worst-case memory proof.
4. Train each arm independently; require the two matching successful gates.
5. Evaluate unadapted FP16/NF4 and their corresponding best LoRA/QLoRA adapters
   on the frozen test set. Verify checkpoint file/tensor hashes and data links.

Stop on OOM, unsupported operations, non-finite numbers, missing gradients,
mask/prefix mismatch, changed inputs or competing GPU processes. Do not
automatically lower image resolution or silently change precision/hardware.

From the server's Hate_Project directory with `memelens-nf4` active:

```bash
# Cache the pinned checkpoint (about 513 MB of weights), without loading a GPU.
python scripts/smolvlm/run.py cache

# Readiness and CPU checks; no training starts.
CUDA_VISIBLE_DEVICES=0 python scripts/smolvlm/run.py check
python -m unittest discover -s scripts/smolvlm -p 'test_*.py' -v
python scripts/smolvlm/run.py preflight

# On the allocated idle V100, run the GPU gates in separate processes.
CUDA_VISIBLE_DEVICES=0 python scripts/smolvlm/run.py smoke --mode fp16 --output results/smolvlm-optimized-v1/smoke-fp16
CUDA_VISIBLE_DEVICES=0 python scripts/smolvlm/run.py smoke --mode nf4 --output results/smolvlm-optimized-v1/smoke-nf4

# Continue only after successful matching gates.
CUDA_VISIBLE_DEVICES=0 python scripts/smolvlm/run.py train --mode fp16 --output results/smolvlm-optimized-v1/fp16-training
CUDA_VISIBLE_DEVICES=0 python scripts/smolvlm/run.py train --mode nf4 --output results/smolvlm-optimized-v1/nf4-training

# Repeat for nf4-base, fp16-lora and nf4-qlora after both trainings finish.
CUDA_VISIBLE_DEVICES=0 python scripts/smolvlm/run.py evaluate --arm fp16-base --output results/smolvlm-optimized-v1/test/fp16-base

# After all four test arms complete, verify paired inputs and compare scores.
python scripts/smolvlm/summarize.py
```

`bash scripts/smolvlm/run_pipeline.sh` runs this entire sequence, stopping at the
first failure. The optimized launcher uses **`results/smolvlm-optimized-v1`**,
to keep original run outputs intact. It includes
two-update disposable profiles and matching smoke gates before either full
training. Pass a new output root as its first argument for another fresh run.
All outputs are new; existing directories are refused. Run in a
persistent terminal. The launcher expects prepared data to exist and uses the
current interpreter, so activate `memelens-nf4` first. Interrupted runs retain
their audit files; this first implementation does not resume training.

### Running the optimized version

After the user copies the local scripts and stops the old job themselves, run on
the allocated V100 in the existing environment:

```bash
bash scripts/smolvlm/run_pipeline.sh
```

That keeps microbatch 2 and disables recomputation. To investigate a larger
microbatch before committing to a full run, profile training-only records in
fresh directories (do not run alongside the old job):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smolvlm/run.py profile --mode fp16 \
  --microbatch 4 --output results/smolvlm-profile-fp16-mb4
CUDA_VISIBLE_DEVICES=0 python scripts/smolvlm/run.py profile --mode nf4 \
  --microbatch 4 --output results/smolvlm-profile-nf4-mb4
```

Profiles write `profile.json`, not a successful training `complete.json` gate.
They use two updates by default (2–5 configurable), never load test data, and
do not establish maximum-length memory safety or convergence. No server timing
or speedup is claimed from local CPU tests. After choosing a common configuration,
the launcher can use it consistently:

```bash
SMOLVLM_MICROBATCH=4 bash scripts/smolvlm/run_pipeline.sh results/smolvlm-optimized-mb4
# Optional memory-saving recomputation, identical in both arms:
SMOLVLM_CHECKPOINTING=on bash scripts/smolvlm/run_pipeline.sh results/smolvlm-optimized-checkpointed
```

Changing microbatch can change floating-point rounding and dropout draws; it is
not a claim of reproducing the old trajectory bit for bit. New code hashes and
execution settings invalidate the old smoke gates. Restart from the shared
initialization in new output folders; the current training process will not
pick up edits and is not resumed under a different recipe.

All 12 optimized tests passed locally in an isolated temporary environment using
Transformers 4.57.1 / PEFT 0.17.1 and the Mac's PyTorch 2.12.0 CPU build. The real
pinned processor produced bit-identical tensors, labels, masks and audit hashes
versus the original two-image-processing path, for both class labels. Tests also
cover the prefetch engine's order and equal record weighting at microbatches
2/4/8, producer shutdown before validation, step-300 early stopping in a flat-score
test, and best-step-75 checkpoint selection. That control-flow test uses simulated
training/validation; it is not a measured training result. The tiny real Idefics3
CPU backward/checkpoint test passed with recomputation disabled.

No SSH connection, server synchronization, GPU profiling or GPU training was
performed while implementing these optimizations. The V100's PyTorch 2.5.1 /
NF4 path must pass the new profiles and smoke gates before a new full run.

## Environment evidence from 1 November 2026

The local desktop is macOS arm64 with Python 3.14 / PyTorch 2.12.0, no CUDA,
and missing Transformers, PEFT and bitsandbytes. It cannot run the approved
CUDA/NF4 experiment. Standard-library control tests run locally.

Read-only server checks reached `eslsrv13.epfl.ch`. Its existing `memelens-nf4`
environment has Python 3.10.18, PyTorch 2.5.1 (CUDA 11.8, sm_70), Transformers
4.57.1, PEFT 0.17.1, Accelerate 1.10.1, bitsandbytes 0.42.0, NumPy 1.26.4,
Pillow 11.3.0 and pandas 2.3.2. The actual model class and pinned processor load.
At inspection the V100 32 GB was occupied by another compute process; no GPU
model run or full training was started. The T4 is not substituted for the V100.
Tests are staged in `/tmp/barberis-smolvlm-20261101`, separate from the live
project. Final verified results are recorded below as checks finish.

All eight final server unit/integration tests passed in the existing environment.
The cached real processor successfully preserves both class-label suffixes and
masks expanded prompts. A small real Idefics3 architecture passed CPU
forward/backward with all adapter gradients, frozen-base checks and a verified
checkpoint save/restore round trip. The actual processor creates 13 native
512-pixel crops for the synthetic 96x64 image. No packages were installed.
The result-comparison test rejects changed metrics and unpaired inputs. The
33 existing MemeLens pipeline tests also passed locally with bundled dependencies.
Both labels fit within the generation budget: ` clean` is one token and
` misogynous` is four tokens, leaving one token for the end-of-utterance marker.

The **actual pretrained checkpoint** also passed a CPU FP32 inference check with
two CPU threads: all 256,484,928 parameters loaded, 13 native crops processed,
and five tokens generated (`In this image, we`). Inference took 57.49 seconds;
this is a compatibility check on a synthetic solid-color image, not an accuracy
or GPU speed benchmark. Reproduce with `cpu_model_check.py --output <new.json>`
after caching the model. CUDA is hidden for this supplemental check.

The existing MAMI data preflight passed on the server: 18,000 training records,
1,000 records in each validation condition, 20,000 unique image files hashed,
and valid masked label suffixes in the four selected real training records.
These records produced 13 or 17 crops, so the pending GPU checks exercise the
native multi-crop path rather than a single reduced image.

**Still pending:** real-checkpoint FP16 and NF4 GPU optimizer/generation gates,
then full training and test evaluation. The occupied V100 was not interrupted.
No claim of QLoRA GPU compatibility or successful full training follows from
the CPU evidence. Local copies of compatibility reports are saved under
`results/smolvlm/compatibility/`.
