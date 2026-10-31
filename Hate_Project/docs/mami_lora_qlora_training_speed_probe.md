# Training speed probe and CPU niceness for the MAMI LoRA/QLoRA study

The user approved an optimization check while the first FP16 full run was in
progress. Update nine took 47.1 seconds for 48 sequential records. The selected
speed changes are diagnostic candidates; the full training runner stays unchanged.

## GPU sharing and nice

`nice -n 19` lowers CPU scheduling priority. It does not lower GPU scheduling
priority, free VRAM, reserve a GPU for another user, or automatically yield the
GPU when someone else needs it. It may make preprocessing slower under CPU load.
GPU sharing/preemption follows the lab's scheduler and allocation rules.

For a future training launch on available GPU 0, the syntax is:

```bash
CUDA_VISIBLE_DEVICES=0 nice -n 19 python -u scripts/mami_lora_qlora/train_comparison.py \
  --mode fp16 --output results/mami_lora_qlora/fp16-training-v2
```

Use the new output path only for a deliberate new run. This example does not
resume or replace the currently running v1 job. To change CPU priority of a live
owned process, first verify its PID/user/command with ps, then use
`renice -n 19 -p PID`. Do not renice or terminate other users' processes.
Suspending a CUDA process is not a way to return its VRAM to other users.

The current full runner only saves at epoch boundaries and has no automatic
resume. Exiting before its first epoch checkpoint loses that partial epoch's
progress. Do not stop the existing run just to change CPU niceness.

## Controlled probe

`speed_probe.py` uses only fixed training records and the same pinned public
base, adapters, shared initialization, image budget, SDPA, FP16 autocast and
nonreentrant checkpointing as the approved run. It does not change the live
training code, source data, validation split, adapter targets or output-token cap.

Every variant runs in a fresh process, first on FP16 and then on NF4:

1. `baseline`: microbatch one, original full-sequence output logits, recompute
   preprocessing and integrity checks inside the measured pass.
2. `cached`: same GPU computation, reuse verified preprocessed CPU tensors.
3. `answer-only`: cached inputs, output logits only at positions predicting
   supervised answer tokens, with the correct causal shift and FP32 cross entropy.
4. `batch2`: cached inputs and answer-only logits, right-pad pairs, microbatch two
   and accumulation 24. Preserve equal loss weight per record, not a token-weighted
   average that would favor longer class labels.

The answer-only method keeps the entire image/prompt transformer computation. It
reduces only output projection/loss work at otherwise unsupervised positions.
For two records it selects the union of required logit positions, ignores each
record's masked positions, and averages its token loss before averaging records.

For correctness, use the same four training-only smoke records with dropout off,
gradient scaling, no optimizer update, and saved FP32 unscaled gradients. Report
loss differences, relative L2 gradient differences and maximum absolute gradient
differences against the baseline. CPU tests verify the causal alignment algebra.
Server results must still establish numerical agreement under actual FP16/NF4
kernels; no automatic equivalence declaration or fastest-variant selection occurs.

For timing, restore the common initialization and create a fresh AdamW/GradScaler
for each of three repetitions. Each repetition processes the same 48 records and
performs one disposable optimizer update, preserving effective batch 48. Warm-up
originals are disjoint from timed originals. Dropout and checkpointing are active
during timing. No adapted checkpoint is saved.

Measured wall time includes CPU work inside that variant, transfers, forward/
backward and the optimizer step, with CUDA synchronization at boundaries. It
excludes model loading, correctness checks, warm-up, shared-weight restoration
and initial cache preparation. Cache preparation time is reported separately.
This measures disposable first optimizer steps, not a steady full-training loop.

The cache contains 52 records in host RAM. It is an upper-bound diagnostic for
preprocessing reuse, not approval to cache all 18,000 training records in RAM.
Full-run adoption needs a bounded cache/prefetch design or a measured storage/RAM
budget. Input hashes verify that cache use preserves the single-record tensors.
Padding means batch-two tensors have different shapes while retaining original
content, mask semantics and per-record weighting.

## Execution after the GPU becomes free

Do not run the probe alongside the active training job. Before loading any model,
and between repetitions, it checks physical GPU 0 for other compute processes
and refuses to proceed. Those snapshots cannot prevent a new competing job from
starting mid-pass or implement GPU preemption. Follow the lab's allocation rules.

After syncing code, from the project root in memelens-nf4:

```bash
bash scripts/mami_lora_qlora/run_speed_probe.sh
```

This launches eight fresh sequential processes using `nice -n 19`, then writes
`results/mami_lora_qlora/speed-probe-v1/summary.json`. Each has request/inventory
metadata, input hashes, three timing/memory records, correctness loss/gradients
and a completion marker. NPZ gradient archives require disk space proportional
to eight copies of the FP32 adapter gradients. No model weights are downloaded.

The launcher stops on the first failure, including OOM. Preserve the failed
directory and inspect the error rather than silently changing batch size or
the image budget. Output directories are never overwritten; a new root can be
passed as the launcher's first argument for an approved deliberate retry.

Nice values are recorded in probe metadata. Compare variant timing under the
same CPU contention/allocation conditions. Three passes and a fixed variant order
do not remove thermal drift, allocator/kernel warm-up effects or workload
representativeness limits.

## Approved short follow-up: full-logit microbatch two

The first server probe completed. Cached microbatch one took 43.55 seconds for
FP16 and 48.04 seconds for NF4 per 48 records. Cached answer-only microbatch two
took 27.08 and 29.36 seconds, respectively, but answer-only microbatch one did
not improve on caching alone and changed gradients in the four-record check.

The approved follow-up isolates microbatch two with full logits. On the user's
request to keep it short, run only cached microbatch one and `batch2-full`, in
both modes, with one timed 48-record update per configuration. Retain the same
four-record dropout-disabled gradient comparison. The full-logit batched loss
averages token losses within each record, then averages records, preserving the
original training objective with unequal label lengths. This is not token-mean
batch loss. Dropout remains enabled during timing; bitwise agreement is not an
acceptance requirement and numerical differences must be reviewed.

```bash
bash scripts/mami_lora_qlora/run_quick_batch2_probe.sh
```

The new launcher uses nice 19, four fresh sequential model loads, a separate
`results/mami_lora_qlora/quick-batch2-v1` root, and the existing GPU-occupancy guard.
Expect several minutes rather than the previous eight-run, three-repeat probe;
one timing pass has greater uncertainty. It does not rerun the old probe or
modify the full training runner. Summary speedups use `cached` as the reference.

After timing each batch-two configuration, perform two disposable training
updates using the two largest token sequences and the two largest vision-patch
inputs in the same 52-record sample. Report allocated/reserved memory and chosen
record metadata. These checks include optimizer state and stop on OOM or
nonfinite gradients. They are sample checks, not a worst-case full-dataset memory
guarantee. Report the sample's CPU tensor bytes; do not extrapolate this small
sample into permission to cache all 18,000 records. The exhaustive dataset scan
and full-cache allocation are deliberately omitted to keep this follow-up short.

Review speed, memory and gradients before selecting an implementation for the
full matched runs. A bounded full-run preprocessing strategy remains unresolved.

Source references:

- https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
- https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/loss/loss_utils.py
- https://man7.org/linux/man-pages/man1/nice.1.html
