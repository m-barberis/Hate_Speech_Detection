# Approved 100-meme, all-20-emoji layout diagnostic

The user approved the bounded one-layout-per-meme, transcription-only evaluation.
This is a new exploratory test evaluation, with no new training or tuning.
It retains the current inference and JPEG pipeline; it is not an exact replay of
the historical grid/circle sweep. In particular, it does not restore historical
processor defaults, the old substring parser or unequal base/adapter token limits.

## Fixed scope

- Sample 100 of the 1,000 MAMI test originals, seed 42, without consulting outcomes.
- Sample within true-class/existing-pattern strata: 50 memes per true class;
  34 dense_random, 33 center, 33 sparse_random overall.
- Class-zero quotas are 17 dense, 16 center, 17 sparse. Class-one quotas are
  17 dense, 17 center, 16 sparse. Source IDs are sorted before deterministic draws.
- Retain each selected meme's already frozen layout and pixel placements from
  `data/processed/mami_unseen_patterns_seed42/manifest.json`.
- Render every selected meme with all 20 original evaluation emoji identities.
  Coordinates and sizes are identical across its emoji variants. No rotation;
  random positions fully inside the image; overlaps allowed; nominal alpha 0.8.
- Preserve the existing compositing semantics and Pillow JPEG defaults.
- Use the original supplied transcription without running OCR.
- Evaluate fp16-base, nf4-base, fp16-lora and nf4-qlora, sequentially on physical
  GPU 0. Fixed epoch-two adapters stay unmerged. No checkpoint selection.
- 100 memes x 20 emojis x one condition x four arms = 8,000 new predictions.

The dense recipe uses 16 emojis of side length 25% of the shorter image side;
center uses one emoji of side length 75%; sparse uses four emojis of side length
25%. Integer positions and sizes are inherited exactly from the previous data.

## Preparation and manual execution

Sync these files to the same server checkout, including the shared inference helper:

- `scripts/mami_lora_qlora/prepare_emoji_sweep.py`
- `scripts/mami_lora_qlora/evaluate_emoji_sweep.py`
- `scripts/mami_lora_qlora/run_emoji_sweep.sh`
- `scripts/mami_lora_qlora/evaluate_unseen_patterns.py` (shared inference helper)
- `scripts/mami_lora_qlora/test_emoji_sweep.py` (CPU checks)
- This protocol document.

From the server project root, in `memelens-nf4`, prepare on CPU:

```bash
nice -n 19 python scripts/mami_lora_qlora/prepare_emoji_sweep.py
```

Then, only with GPU 0 free and previous evaluations finished:

```bash
bash scripts/mami_lora_qlora/run_emoji_sweep.sh
```

No server commands are launched by the assistant. Preparation requires the
previously frozen full clean/grid and unseen-pattern datasets, raw original images
and all 20 emoji assets at their recorded server paths. The run requires the
matched completed training folders. Summarization additionally requires all four
completed original clean/grid result folders.

Defaults:

- Prepared data: `data/processed/mami_patterns_100_all20_seed42/`
- Results: `results/mami_lora_qlora/patterns-100-all20-v1/`
- Per arm: `metadata.json`, `predictions.jsonl`, `complete.json`
- Final summary: `comparison.json`

The launcher accepts prepared-data, result-root and clean-result-root as its
first, second and third optional arguments. Preparation refuses to overwrite an
existing folder. Completed evaluation arms are reused only after verification;
partial arm folders cause a stop and are preserved. There is no automatic partial
resume or silent restart. The summary is recomputed and an existing comparison is
accepted only if identical. Stop on OOM, competing GPU processes, source/hash or
control mismatches, or inference failures. Do not change the image budget or
default failures to class zero to finish a run.

## Model and inference controls

Reuse the existing unseen-pattern generation loop rather than duplicate model
loading: immutable checkpoint revision, local cached files only, FP16 compute,
NF4 eligible-linear weight quantization with no double quantization, single-example
SDPA inference, FP16 autocast, greedy decoding, five new tokens, one beam, KV cache,
minimum 4,096 and maximum 524,288 pixels. Hash generation inputs without targets.
Reuse the leading-label parser; invalid/ambiguous predictions remain errors and
strict formatting is reported separately. Keep raw responses.

The shared helper was parameterized only for dataset loading, completed-output
validation, source provenance and dynamic counts/progress. Its normal full
unseen-pattern entry point retains its existing conditions and inference controls.
The original grid evaluator is unchanged.

## Analysis and interpretation

Report macro-F1, accuracy, confusion matrices, invalid counts and strict-format
compliance for the full fixed mixture, each pattern and each emoji. Aggregate
macro-F1 comes from pooled confusion counts, not an average of emoji macro-F1s.
All memes have 20 variants, so each meme receives equal total weight.

For the aggregate and each pattern, compare adaptation within precision and
NF4 versus FP16 before and after adaptation. Compare corruptions against the same
selected memes' verified clean-with-transcription predictions. Repeat each meme's
clean label prediction only to match the 20-variant weight when forming corruption
contrasts; do not count these repeats as newly generated predictions or independent
memes. Existing grid predictions have only one emoji per meme and are not treated
as an identity-matched all-20 grid baseline.

Use paired 2,000-replicate percentile 95% bootstrap intervals, seed 42. Draw whole
memes within true-class/assigned-pattern strata, retaining every selected meme's
20 emoji variants and using identical draws across the arms in each contrast.
Per-meme confusion matrices allow efficient clustered resampling without treating
variants as independent observations.

Only 100 independent memes are represented; pattern subsets contain 33 or 34.
Intervals are pointwise and unadjusted, no equivalence margin was specified, and
one training seed/split does not measure training-run variability. A smaller drop
from a worse clean score is not sufficient evidence of greater robustness. Pattern
recipes differ in coverage/count/size and subsets, so raw pattern rankings do not
isolate geometry. Test results were already examined, and the sample may overlap
the 100 memes previously used for inference benchmarking. Treat this as diagnostic
evidence, not a fresh confirmatory test or an authorized model-selection loop.

## Resource estimate

Previous single-pass timings suggest about 86 minutes of evaluation across all
four arms, plus loading, preparation and saving; allow roughly two hours if input
token lengths are comparable. This is a planning estimate, not a latency benchmark.
No software installation, retraining or extra GPU job is part of this protocol.
