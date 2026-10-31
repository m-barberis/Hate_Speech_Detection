# MAMI overlay robustness: matched FP16 LoRA and NF4 QLoRA protocol

The user approved a stratified 90/10 split, seed 42, on 31 October 2026.
Reuse the historical augmentation method and two-epoch training schedule.
Both new arms use the same prepared records and evaluate the final epoch-two
checkpoint. Validation monitors training; it does not select a checkpoint.
Historical LoRA scores provide context rather than a controlled baseline.

## Data preparation

Inputs on eslsrv13:

- `data/raw/TRAINING/training.csv`: tab-separated, 10,000 originals, 5,000/class.
- Training images beside the table, e.g. `data/raw/TRAINING/1.jpg`.
- Noto 512x512 RGBA PNG folder, supplied through `EMOJI_DIR`.
- The existing 1,000-image MAMI test set stays separate.

Training has 9,000 originals (4,500/class). Validation has 1,000 originals
(500/class). Split membership is determined from sorted original filenames,
with a separate seeded random generator. Original CSV order determines the
augmentation draws, matching the old clean-dropout/overlay-dropout/emoji-choice
sequence. Split membership never depends on an augmentation or prediction.

New implementation and checks live in `scripts/mami_lora_qlora/`. This protocol
lives separately in `docs/mami_lora_qlora_robustness_protocol.md`.

`scripts/mami_lora_qlora/prepare_comparison_data.py` loads
`scripts/mami_finetuning/augment_train_set.py` by its location relative to the
new script, then calls its original `apply_4x4_grid` and `construct_prompts`
functions. The original script now has
import-safe initialization and accepts an optional random generator; its normal
standalone behavior is otherwise unchanged. No new renderer is substituted.

The original nominal opacity 0.8, grid geometry, resize/compositing operations,
and Pillow JPEG defaults remain unchanged. Nominal opacity is a renderer
parameter, not a claim that 80% of image information is occluded.

Outputs in a new directory:

- `train.jsonl`: 18,000 fixed records, clean plus overlay per training original,
  using the original independently drawn 50% transcription dropout.
- `validation_clean.jsonl`: 1,000 clean validation records, image-only prompt.
- `validation_overlay.jsonl`: 1,000 corresponding fixed grid overlays, using
  the ten training emoji assets and the same image-only prompt. This validation
  condition does not measure transfer to held-out emoji identities/geometries.
- `manifest.json`: split memberships, labels, source paths/hashes, emoji hashes,
  renderer/preparation source hashes and relevant library versions.
- `complete.json`: record counts and JSONL/manifest hashes. Absent on failure.

Target labels are `clean` and `misogynous`. Transcriptions retain the original
script's whitespace stripping and missing-value handling. No mojibake repair
or semantic text cleaning is introduced. Missing/invalid records abort rather
than being silently skipped. Repeated runs cannot overwrite an output folder.

Five training/test filename collisions contain different images according to
byte hashes, decoded-pixel hashes and the user's visual inspection. Retain those
training examples and use split-qualified IDs. A broader duplicate audit under
different filenames remains deferred. No exact historical Noto release is known;
asset hashes identify the supplied files for this follow-up.

## Server execution

Push/sync this checkout to the server, including the updated historical
`scripts/mami_finetuning/augment_train_set.py` helper and the new
`scripts/mami_lora_qlora/` directory. In the existing `memelens-nf4` shell, from
the project root, retain the verified `EMOJI_DIR`:

```bash
python scripts/mami_lora_qlora/prepare_comparison_data.py \
  --train-dir data/raw/TRAINING \
  --emoji-dir "$EMOJI_DIR" \
  --output data/processed/mami_lora_qlora_seed42
```

This command only prepares data on CPU. It neither loads model weights nor
launches training. It writes 10,000 augmented JPEGs, so allow disk space for
those images as well as the JSONL files. On interruption retain the partial
directory for inspection and use a new output path after diagnosing the issue.

## Subsequent model work

Use the same pinned public checkpoint for fresh FP16 LoRA and NF4 QLoRA.
Keep adapter targets and effective training batch comparable. Audit actual
adapter matches and image-expanded prompt masking before training. The old
training script assumes multi-GPU DDP and must be adapted before use on the
single V100. The old training implementation is not a launch command for this
follow-up. Check finite loss/gradients and memory before full training.

Future evaluations use five maximum new tokens, greedy decoding and the same
leading-label parser in all arms, retaining strict format compliance separately.
The existing ten-token efficiency runs remain a separate result. No GPU jobs
or five-token baseline reruns have been launched by this implementation step.

No model-training code or existing efficiency runner settings are changed by
this preparation step. Full-run hardware, quality and latency results cannot
be predicted from synthetic preparation tests.

## Local checks

```bash
python -m unittest discover -s scripts/mami_lora_qlora -p 'test_*.py'
```
