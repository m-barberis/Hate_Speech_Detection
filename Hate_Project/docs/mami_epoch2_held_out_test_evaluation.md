# Fixed epoch-two LoRA/QLoRA evaluation on the 1,000 MAMI test memes

Approved scope: all 1,000 labelled test originals, one fixed overlay per meme,
balanced across the 20 evaluation emoji identities from the historical
`mami_finetuning/eval_finetuned.py`. Evaluate four arms: unadapted FP16,
unadapted NF4, epoch-two FP16 LoRA, and epoch-two NF4 QLoRA. Each sees four
conditions: clean/overlay crossed with image-only/original transcription.
There are 4,000 predictions per arm and 16,000 in total. No new training,
checkpoint selection, alpha sweep, circle patterns, or 20-emoji-per-meme sweep.

## Frozen data and augmentation

The preparation script reads test transcriptions from tab-separated Test.csv
and binary misogyny labels from tab-separated test_labels.txt (header optional).
It requires exactly matching unique filename sets and 500 examples per class;
missing labels, empty transcriptions, missing images and unreadable images stop
preparation. It supports the original jpg-to-png fallback. All test originals
are used; the 100-image benchmark manifest does not subsample this evaluation.

The script reuses `augment_train_set.py` for 4x4 overlays at nominal alpha 0.8,
original compositing semantics and Pillow JPEG defaults. The 20 emoji identities
are extracted from the historical evaluation script without importing its GPU
or distributed-training code. Asset hashes are frozen and identities must be
disjoint from the 10 training emojis. Exact Noto release remains unknown.

Sort filenames, shuffle separately within each true class using seed 42, and
cycle through the original ordered 20 emoji codes. This gives 25 memes per class
per emoji (50 total). This is a fixed class-balanced assignment, not a claim of
individual-emoji robustness from a 1,000-example-per-emoji evaluation.
Every model sees the same rendered overlay bytes. Clean originals are not
re-encoded. Original transcription is retained unchanged under overlays; no OCR
is rerun. The original prompt function is reused with dropout disabled for
transcription-present conditions and forced absence otherwise.

Image bytes are checked against the preparation manifest and exact byte
duplicates with all training/validation originals are rejected. Known equal
filenames with different images are retained and recorded, not treated as
identity matches. Cross-name RGB/semantic duplication is not exhaustively
audited. All 1,000 memes were held out of adapter training, but 100 were already
examined in the earlier inference benchmark: do not call this an entirely
previously unseen test set or use it to tune the training recipe retrospectively.

## Server preparation

Sync the entire updated scripts/mami_lora_qlora folder and this document. Existing
prepared training/validation data and completed training outputs are unchanged.
In memelens-nf4, from the project root:

```bash
nice -n 19 python scripts/mami_lora_qlora/prepare_test_conditions.py
```

By default, locate exactly one Test.csv and one test_labels.txt recursively
under data/raw, use Test.csv's parent for test images, and infer the emoji asset
folder from the existing training manifest (or EMOJI_DIR if set). Discovery stops
on ambiguity or missing files; it does not guess between datasets. Explicit
paths override discovery:

```bash
nice -n 19 python scripts/mami_lora_qlora/prepare_test_conditions.py \
  --texts /actual/test/Test.csv \
  --labels /actual/test_labels.txt \
  --images /actual/test/images \
  --emoji-dir /actual/emoji/512
```

The new immutable output is
`data/processed/mami_lora_qlora_test_seed42/`, with a manifest, 1,000 JPEG overlays,
four condition JSONL files and a completion marker. Existing directories are
never overwritten. There is no GPU work during preparation.

## Fixed inference procedure

```bash
bash scripts/mami_lora_qlora/run_test_evaluation.sh
```

Run manually with GPU 0 available and a persistent terminal. The launcher runs
four fresh sequential processes with CUDA_VISIBLE_DEVICES=0 and nice 19, stops
on the first failure, then performs CPU paired analysis. An occupancy guard
rejects competing GPU processes before loading and between conditions. Snapshot
checks do not implement GPU preemption or guarantee contention cannot arise.

The immutable public revision is
QCRI/MemeLens-VLM@a633183849df60a37251d29c3f1b0d379ae801f5. Reuse local cache,
SDPA, FP16 compute, min pixels 4096 and max pixels 524288. NF4 uses the training
eligible-linear quantization policy, no double quantization and no activation
quantization. Keep LoRA adapters unmerged in both adapted arms. PEFT loads the
fixed epoch-2 adapter on its corresponding base; the canonical FP32 hash of all
reloaded LoRA tensors must equal the completed training hash. Verify saved adapter
files against the epoch checkpoint manifest. Do not load optimizer pickle files.
Both completed training plans must have matching data order, initialization,
batch, schedule and implementation controls. The inference environment must match
the recorded training package versions. No new package installation is required.

Inference is single-example greedy generation, max_new_tokens=5, num_beams=1,
KV cache enabled, eval mode, no gradients or teacher-forced target inputs.
Use the existing leading-label parser; invalid/ambiguous answers remain errors.
Strict format compliance is reported separately. Hash actual prompt-only tensors
and verify signatures and true labels align across all four arms before paired
analysis. This is one quality pass, not a repeated controlled latency benchmark;
reported evaluation wall time and peak memory are descriptive only. No training
or test-data driven checkpoint selection is performed.

## Outputs and uncertainty

Default result root: `results/mami_lora_qlora/test-1000-v1/`. Each arm folder has:

- metadata.json: immutable base, software/hardware, processor, quantization,
  generation controls, trained-source hashes, adapter hash and code hashes.
- predictions.jsonl: 4,000 ordered predictions, raw answers, true/predicted labels,
  format compliance, generated/input token counts, emoji code and input hash.
- complete.json: metrics for each condition, file hashes, descriptive wall time
  and peak inference memory. Appears only after all predictions complete.

The final comparison.json contains all arm summaries and paired contrasts:
FP16 adaptation, NF4 adaptation, unadapted quantization, adapted quantization,
and overlay-minus-clean effects within each arm and transcription condition.
Report macro-F1, accuracy, invalid count, confusion matrices and strict format.

Each contrast uses the same resampled meme indices for both predictions, with
2,000 class-stratified bootstrap replicates, seed 42, and percentile 95% intervals.
Do not treat four conditions as 4,000 independent original memes. These are
unadjusted pointwise intervals, not multiplicity-corrected significance claims
or proof of equivalence. No equivalence margin is specified. Single training
seed/split results do not measure training-run variability. Transcription-present
conditions are primary; image-only conditions diagnose behavior without supplied
text. Keep each condition separate.

Rerunning the launcher reuses verified completed arms and refuses incomplete
existing arm directories; it does not silently overwrite partial predictions or
restart a failed arm. Preserve failures and inspect them before deciding on a
retry. There is no per-example automatic resume in this version. The comparison
file is also never overwritten. Positional launcher arguments can select a new
prepared-data path and result root for an explicitly intended retry.

The local environment has no GPU/Torch stack. CPU tests exercise exact label/text
joins, deterministic class-balanced emoji assignment, original augmentation and
prompt reuse, data corruption rejection, and paired interval calculations with
invalid predictions. Actual adapter reload/hash verification and inference remain
server runtime checks; CPU tests alone do not establish GPU compatibility.
