# Huawei interview preparation

Prepared 31 October 2026. This is a timed presentation script and technical backup, not a finished slide deck.

## Main message

“I studied two practical tradeoffs in multimodal classification. In our EPFL project, robustness-oriented LoRA adaptation improved performance on synthetic overlays while reducing clean performance in the saved predictions. In a separate follow-up using the same pinned public checkpoint in both arms, NF4 reduced peak allocated GPU memory by 60.7%, but increased median generation latency by 29.2% and reduced macro-F1 by 5.06 percentage points on a balanced 100-image pilot. The memory target passed and the quality-retention target failed.”

The historical scores are recomputed from local prediction CSVs. The full efficiency scores currently come from the user-supplied completed-run summary. Full efficiency prediction files are absent locally. Treat the latter as reported results until the read-only server check completes.

## Twenty-minute presentation

### 1. Background and motivation — 2 minutes

**Suggested title:** Multimodal classification under memory and robustness constraints

Introduce your actual education and relevant experience using your CV. The available project files establish the EPFL EE-559 collaboration but do not establish your full CV. Do not invent a biography.

**Speaking points:** “My interest is in how model choices affect useful performance under resource constraints. This project gave me experience selecting a multimodal model and data, designing augmentations, and interpreting evaluation results. I extended it with a small inference comparison to measure memory and latency directly.”

Connect motivation to resident model memory and response time. Do not claim measured energy, financial cost, deployment throughput, or training-memory savings.

### 2. Task, model and ownership — 2 minutes

**Suggested title:** Multimodal misogyny classification

- Task: binary MAMI classification, with image-only and supplied-transcription settings in the original study.
- Public model: QCRI/MemeLens-VLM, loaded by Qwen3VLForConditionalGeneration. The supplied provenance describes Qwen3-VL-8B-Instruct as its base.
- Original project: three-person EPFL EE-559 collaboration with Giovanni Pomati and Giorgio Panizzutti.
- Your stated contributions: model and dataset selection, augmentations for fine-tuning and evaluation, results analysis.

**Say:** “I worked on these areas within a collaborative project. I use ‘we’ for the joint training and evaluation work and explain my own decisions separately.”

Do not assign unverified individual responsibilities to teammates. Explain the separate follow-up and AI-assisted implementation honestly if asked. MAMI appears among the model's source datasets, so this is not evidence of unseen-dataset generalization.

### 3. Original adaptation and robustness method — 2 minutes

**Suggested title:** LoRA adaptation with synthetic overlays

Explain LoRA as training low-rank updates while retaining the underlying model weights. Original configuration: rank 16, alpha 32, dropout 0.05, two epochs, learning rate 2e-4, broad projection/MLP/output/embedding targets.

Training records included clean images and 4×4 emoji grids at nominal opacity 0.8, with 50% transcription dropout. Evaluation varied emoji identity, geometry and opacity. Say “synthetic obfuscation” rather than claiming real-world adversarial robustness. Grid/circle evaluation supports a question about new geometries, but pooled results alone do not isolate geometry transfer.

No historical training-memory, speed, or cost measurements survive. Missing adapters and training logs limit reproducibility.

### 4. Historical results and what did not work — 3 minutes

**Suggested title:** Overlay robustness improved as clean performance fell

| Saved result label | Clean macro-F1 | Opacity 0.8 | Opacity 1.0 |
|---|---:|---:|---:|
| Public checkpoint, image only | 0.8451 | 0.7816 | 0.7428 |
| LoRA-labelled, image only | 0.8242 | 0.7938 | 0.7560 |
| Public checkpoint, image + transcription | 0.8324 | 0.7599 | 0.7706 |
| LoRA-labelled, image + transcription | 0.8219 | 0.7967 | 0.7899 |

Each clean condition has 1,000 images. Each displayed overlay condition pools 40 variants per image, 40,000 rows from the same 1,000 images. These are repeated perturbations, not 40,000 independent test images. Macro-F1 here is calculated from pooled confusion counts, not the mean of per-variant F1 scores.

**Say:** “In the image-only records, clean macro-F1 fell by 2.09 points, while opacity-0.8 macro-F1 rose by 1.22 points. In the transcription setting, clean macro-F1 fell by 1.04 points and opacity-0.8 macro-F1 rose by 3.68 points. That is a useful tradeoff, but it did not preserve clean performance.”

The image-only clean-to-0.8 drop is 6.34 points before versus 3.03 after. With transcription it is 7.25 versus 2.53. The parent README's 5.9%/1.5% and 7.0%/1.1% figures do not follow this explicit aggregation. Omit those README figures until their exact definition is recovered. They may refer to another slice, so disagreement is not proof that the underlying calculation was wrong.

**Visible caveat:** historical evaluators differed in parsing/generation limits and often defaulted errors to clean. Checkpoint revision and exact environment are unknown. The adapter is unavailable. Describe an association in saved results, not a controlled causal estimate of the LoRA effect.

### 5. Controlled efficiency follow-up — 3 minutes

**Suggested title:** FP16 and NF4 under matched inference settings

**Say:** “I could not use the missing adapter for a reproducible systems comparison, so both arms use the same immutable public checkpoint. This isolates the comparison from the historical LoRA results.”

- Revision: a633183849df60a37251d29c3f1b0d379ae801f5.
- 100 fixed clean MAMI images, 50/class, seed 42, five disjoint warm-ups.
- Image plus task instruction, no supplied transcription or overlays.
- One V100-PCIE-32GB, batch one, same processor and pixel budget, SDPA, greedy generation, KV cache, at most 10 new tokens.
- NF4 for eligible linear modules with FP16 compute and no double quantization. This is not an all-tensors-at-4-bit claim.
- Three passes per mode in sequential fresh processes, quality from pass 0 only.

**Measurement boundaries:** CUDA-synchronized model.generate includes vision encoding, prefill and decoding. It excludes loading, disk reads, CPU preprocessing and transfer. Memory is absolute peak PyTorch allocated memory during requests, including weights and GPU inputs. Report loading peaks separately.

The environment uses PyTorch 2.5.1/CUDA 11.8, Transformers 4.57.1, Accelerate 1.10.1 and bitsandbytes 0.42.0. Both full arms are intended to use process-local forced dispatch hooks. The server metadata check must confirm equality. Do not interpret this as a benchmark of a modern optimized NF4 stack.

User reports GPU 0 had no competing workloads throughout both runs, monitored throughout. This is operator confirmation, not a continuous telemetry file inspected here.

### 6. Efficiency results — 3 minutes

**Suggested title:** NF4 saved memory and missed the quality target

| Metric | FP16 | NF4 | Comparison |
|---|---:|---:|---:|
| Peak allocated GPU memory | 16.56 GiB | 6.52 GiB | 60.7% reduction |
| Median generation latency | 480 ms | 620 ms | 29.2% slower |
| p95 generation latency | 555 ms | 727 ms | Descriptive request percentile |
| Macro-F1, 100 images | 0.8798 | 0.8292 | −5.06 percentage points |
| Accuracy | 88/100 | 83/100 | Five fewer correct |
| Strict label-only compliance | 100% | 95% | Leading-label scoring remains valid |
| Mean generated tokens | 3.62 | 4.11 | 13.5% more in NF4 |

**Say:** “The result met the proposed memory target of at least a 30% reduction. It failed the target of at most 0.02 absolute macro-F1 loss. On this V100 and software stack, reduced memory did not produce a latency improvement.”

Report the paired stratified bootstrap interval for NF4 minus FP16 as −0.0960 to −0.0102, with 2,000 resamples. It is a conditional estimate for this balanced pilot and saved pairings, not an equivalence test or a guarantee about deployment quality. The numeric interval is supplied by the user and awaits raw-record reproduction.

### 7. Failure analysis — 3 minutes

**Suggested title:** Classification and output format are separate failure modes

From the reported confusion matrices and five disagreements: four previously correct clean images became false positives and one previously correct misogynous image became a false negative. All five changes lost correct predictions. Their image IDs and visual causes remain unavailable locally.

A verified warm-up example is 15210.jpg, true clean:

- FP16: `clean`, two generated tokens.
- NF4: `clean\nThe meme features a character with a serious`, ten generated tokens.
- Both predict clean under the approved leading-label parser. NF4 violates strict one-label formatting.

This warm-up is disjoint from evaluation. It illustrates why format failure is not automatically a classification failure. Do not use it as one of the five evaluation disagreements. Smoke tests confirm 368 converted NF4 modules: 116 vision and 252 language. This prevents attributing failures solely to language quantization.

For the final failure slide, inspect all five evaluation disagreements and all five format violations. Select one false positive and the false negative as representative examples. Include filename, ground truth, both raw answers and token counts. Separate visible evidence from hypotheses about small text, image cues or label ambiguity. Do not claim an explanation identifies the model's internal reasoning. Check image redistribution rights before embedding dataset images.

### 8. Lessons and research fit — 2 minutes

**Suggested title:** Resource savings require task-level validation

**Say:** “The original study taught me to evaluate clean quality alongside robustness. The follow-up taught me to measure memory, latency and output behavior together. NF4 was effective for memory in this configuration, but it did not meet my quality-retention target and it was slower. I would keep FP16 for this pilot's quality-sensitive use case when memory permits. Further investigation should start from the changed examples and profiling, with a concrete plan agreed before running anything.”

Connect this to your interest in multimodal efficiency research. Do not claim the team uses any particular architecture or hardware unless they tell you.

## Technical questions to rehearse

**Why use NF4?** It offers a way to reduce storage for eligible weights while retaining FP16 computation. In this study, the measured question is whether that saving preserves task quality and response time. Do not claim it is optimal for this task.

**Why was NF4 slower?** The measurement shows slower full generation on this stack. NF4 generated more tokens, so output length contributes to the comparison. Dequantization/kernel overhead and dispatch behavior are possible contributors, but no profiler evidence separates them. Do not divide median latency by mean tokens and call the result kernel speed.

**Why does memory fall by 60.7% rather than exactly 75%?** Peak allocated memory includes unquantized tensors, quantization metadata, activations, caches and inference temporaries. Only eligible linear layers were converted. A 4-bit weight representation does not imply a fourfold reduction in total peak memory.

**Does it fit on the T4?** No T4 run has been performed. The V100 PyTorch allocation is not a T4 compatibility or capacity test. Leave performance on the T4 unclaimed.

**How is this related to LoRA?** LoRA was the original adaptation method. NF4 is a separate public-checkpoint inference follow-up. No quantized LoRA/QLoRA result was produced.

**Why only 100 images?** It is a fixed, balanced, paired pilot designed for a bounded follow-up. It detects the observed tradeoff but does not establish deployment performance. Repeated timing passes do not increase the quality sample size.

**Is the quality loss statistically significant?** The reported paired bootstrap interval excludes zero, but the change consists of only five examples in a small balanced sample. Say the pilot supports a negative quality difference under its sampling assumptions. Avoid treating the interval as population-wide proof or a formal test of the retention target.

**How robust is the latency difference?** Reported per-pass FP16 medians are 485/483/473 ms, and NF4 medians are 619/622/618 ms. The slowdown appears in every pass. These are descriptive repetitions within sequential runs, not randomized independent trials across hardware or time. Operator monitoring reports no competing workload.

**Why permit explanations if the prompt asks for one label?** The approved scoring rule accepts a standalone leading class, rejects missing/ambiguous first-line labels, retains raw answers and reports strict compliance separately. The rule changed before the 100-image evaluation and applies equally to both arms. It never silently maps an unrecognized response to clean.

**Can extra tokens explain every disagreement?** That is unknown until the overlap is inspected. The count of five disagreements and five format violations does not identify the same five images.

**What would you investigate next?** First inspect existing changed examples and token-length patterns. A future experiment could separate fixed-output-length timing from task generation or localize vision/language sensitivity, but each requires an agreed design. None has been run here.

**What did you personally do?** Model and dataset selection, augmentation design for adaptation/evaluation, and analysis in the original collaborative project. Explain your actual role in the follow-up and the use of AI assistance without claiming sole implementation of everything.

## Questions for the team — final 10-minute segment

- Which resource constraint most often determines your research priorities: GPU memory, latency, throughput or energy?
- How do you balance systems improvements with multimodal task-quality regressions?
- What would a successful internship contribution look like, and how do interns work with research and engineering colleagues?
- Which evaluation workloads and compute environments would the intern use?

## Evidence still needed before finalizing a failure slide

1. Recompute the full summary from saved server records and compare it with summary-eslsrv13.json.
2. Verify final settings, pinned revision, software, GPU identity, input hashes, manifest membership, parsing and pass stability in both full runs.
3. Recover the five changed IDs and their paired raw outputs, token counts and format overlap.
4. Inspect the associated images for qualitative discussion. Local data/raw currently contains no test images.

No setup, model download, new inference, training or package modification is required for these checks. The read-only command is in `scripts/efficiency/inspect_completed_runs.sh`. Paste its contents into the existing memelens-nf4 server shell. It recomputes the summary in temporary storage and leaves saved results untouched.

## Source record

- Interview format and requirements: /Users/matteobarberis/Downloads/Internship_Interview_Prep_Multimodal_Efficiency.pdf, page 1.
- Original contributions and completed efficiency numbers: user's pasted continuation request, 31 October 2026.
- Historical calculations: results/mami_massive_sweep_results.csv, mami_massive_sweep_with_text_results.csv, mami_finetuned_massive_no_text_results.csv, mami_finetuned_massive_results.csv.
- Local controls and warm-ups: results/efficiency_v100/manifest.json and smoke run metadata/warmup files.
- Scoring and measurement implementation: scripts/efficiency/compare.py. Actual server file may differ and remains the controlling implementation for those runs.
- GPU-idleness confirmation: user reply in this chat, 31 October 2026.

The pinned public model card could not be retrieved through the web tool during this review. Architecture/source-dataset statements above retain the supplied provenance; do not describe them as independently web-verified.
