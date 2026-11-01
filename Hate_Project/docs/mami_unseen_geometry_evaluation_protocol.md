# Proposed unseen-circle robustness test for the fixed MemeLens adapters

Status: superseded, not implemented. The user instead approved dense random,
center and sparse random patterns; see mami_dense_center_sparse_test_protocol.md.
This supplements the existing 1,000-image clean/grid evaluation and does not
change its inputs, running processes, checkpoint selection, or saved outputs.

## Question and hypothesis

Does adaptation on 4x4 training-emoji grids improve classification on the
historical circle_12 corruption, which was absent from adapter augmentation?
Compare each fixed epoch-two adapter with its corresponding unadapted base.
A positive paired macro-F1 difference supports adaptation benefits on this
condition; an interval crossing zero leaves the direction uncertain. No minimum
practical effect or equivalence margin has been chosen. Unseen means absent from
this adapter augmentation, not proven absent from public-base pretraining.

## Recommended primary extension

Use all the same 1,000 test originals and exactly the same per-meme assignment of
20 evaluation emoji identities already frozen for the grid evaluation (25 memes
per true class per emoji). Keep the original transcriptions for both layouts.
Add circle_12 at nominal alpha 0.8, with and without supplied transcription.
Evaluate unadapted FP16, unadapted NF4, epoch-two FP16 LoRA and epoch-two NF4 QLoRA.
This adds 8,000 predictions, bringing clean/grid/circle evaluation to 24,000.
Reuse existing clean/grid predictions only after input, source, generation and
checkpoint controls are verified; do not regenerate them merely to add circles.

Reuse the actual apply_circle_pattern function from
scripts/mami_finetuning/eval_finetuned.py. Parameters are 12 emojis, emoji side
length int(0.15 * minimum image side), circle radius 0.35 * minimum image side,
centered on the image, with the historical angular placement, LANCZOS resize,
alpha scaling, masked paste and alpha-composite operations. Make it safely
importable or extract its unchanged function AST into a minimal namespace so
importing augmentation does not initialize distributed inference. Record original
source hashes and verify pixel equality against the unchanged function on CPU.
Use the same JPEG encoding as the frozen grid test and freeze output hashes.
Nominal alpha is a recipe parameter; preserve original compositing semantics
rather than claiming a uniform 80% effective pixel opacity.

The circle recipe changes location, emoji count, emoji size and occupied area.
It is a historical layout-recipe transfer test, not an area-matched experiment
isolating spatial geometry alone. Circle scores exceeding grid scores may simply
reflect less obstruction; they do not by themselves prove better generalization.
Because evaluation emojis are also absent from adapter augmentation, this tests
joint transfer to these identities and to the circle layout. Pairing the same
emoji identity per meme between grid and circle controls identity for that
layout-recipe contrast. It does not isolate geometry independently of all other
corruption parameters, or demonstrate behavior for every emoji on every meme.

## Controls, outputs and interpretation

Use the same immutable public base revision, unmerged adapters, pinned software,
processor/image budget, single-example greedy generation, max_new_tokens=5,
label parser and strict-format scoring as the current test. No new training,
learning-rate tuning, checkpoint switching, DDP setup or package installation.
Run sequentially on available V100 GPU 0 with CPU nice 19; refuse competing GPU
jobs, preserve completed results, and stop on OOM, missing/corrupt images,
adapter/hash/control mismatches or inference failures. Never default failures to
clean labels. New artifacts belong in scripts/mami_lora_qlora and docs, with new
prepared/result folders so the existing grid evaluation remains inspectable.

Report macro-F1, accuracy, confusion matrices, invalid counts and strict format
for each arm and prompt condition. Primary readout supplies original transcription;
image-only results are diagnostic. Do not pool conditions into a larger sample.

Paired model contrasts on circles: FP16 LoRA minus FP16 base; NF4 QLoRA minus NF4
base; NF4 base minus FP16 base; and QLoRA minus LoRA. Report within-arm circle
minus clean and circle minus grid effects. To assess whether the adaptation
benefit transfers from grid to circle, also report (adapted circle F1 minus base
circle F1) minus (adapted grid F1 minus base grid F1), separately by precision and
prompt condition. Use the same class-stratified resampled meme indices in every
term, 2,000 seed-42 bootstrap replicates and percentile 95% intervals. These are
pointwise exploratory intervals, not adjusted multiple-testing claims, formal
equivalence, or measurements of training-seed variability.

Runtime estimate: the circle extension contains half as many predictions as the
current 16,000-prediction clean/grid test. Use observed per-arm timings from that
test to estimate the extension; do not promise half the wall time because model
loading, token counts and image-processing costs may differ.

## Optional original alpha sweep, not part of the recommended first extension

The historical scripts used alpha values 0.0, 0.5, 0.8 and 1.0, grid_4x4 and
circle_12, and all 20 evaluation emojis for each meme. A bounded follow-up could
retain the current single-emoji assignment while adding grid alpha 0.5/1.0 and
circle alpha 0.5/0.8/1.0. Relative to the existing clean/grid-0.8 test, that adds
40,000 predictions across four arms and two prompt conditions, 56,000 overall.
The primary circle-0.8 extension accounts for 8,000 of those additions.
Alpha 0.0 reuses clean inputs and predictions. This reproduces the historical
layout and alpha settings but not the original full emoji-per-meme sweep.
Do not run this optional sweep without approval of its additional scope/cost.
