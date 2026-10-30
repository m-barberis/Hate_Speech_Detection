# MemeLens FP16 versus NF4: approved interview follow-up

Approved in chat on 30 October 2026. This is a new inference-only follow-up to
the collaborative EPFL robustness/LoRA project. Both arms use one immutable
public `QCRI/MemeLens-VLM` checkpoint, without the lost project adapter.
Never calculate quantization loss against the old fine-tuned results.

Matteo's stated original contributions: model and dataset selection,
augmentations for fine-tuning and evaluation, and analysis of the results.
The original project was collaborative; distinguish this follow-up and describe
AI-assisted implementation honestly if asked.

## Scope and controls

- 100 clean MAMI test images, 50 per class, seed 42; five disjoint warm-ups.
- Image plus the original task instruction, no supplied transcription or overlays.
- Same model revision, processor, actual tensor inputs, SDPA attention, greedy
  generation, 10-token maximum, KV cache, batch one, one V100, three timed passes.
- Same maximum pixel budget: `512 * 32 * 32`, minimum `4 * 32 * 32`.
  Aspect ratio is preserved by the processor; log actual image grids and input
  token counts. This is a controlled new setting, not exact historical replication.
- FP16 versus bitsandbytes NF4 eligible linear layers; FP16 compute, no nested
  quantization. The module inventory records whether vision layers are converted
  too. This is not a claim of language-only quantization or all tensors being 4-bit.
- All weights on the GPU; no CPU/disk offload. No fine-tuning or training.
- Quality is computed once from pass 0. Repeated passes measure timing variation,
  not 300 independent quality observations. Summarization rejects unstable raw
  answers between passes rather than concealing nondeterminism.
- Exact `clean`/`misogynous` parser, invalid answers recorded as errors, exceptions
  abort. A malformed answer never silently becomes class 0.
- Macro-F1, accuracy, confusion matrix with invalid column, paired disagreement,
  2,000 stratified paired bootstrap resamples for the F1 difference.
- Synchronized generation latency (vision encoder + prefill + decoding), median
  and p95 over 300 requests, per-pass medians, generated tokens. Disk access,
  CPU preprocessing, CPU-to-GPU copy, loading, and five warm-ups excluded.
  This is generation latency, not end-to-end service latency or isolated kernel speed.
- Absolute peak PyTorch allocated/reserved GPU memory per request, including
  weights and GPU inputs. Maximum across requests is the headline. Loading peak
  and post-load allocation reported separately. Allocator reservation is not live
  tensor memory; these do not capture every byte of CUDA context/driver memory.
- No speedup assumed. Expected outcome: lower weight/resident memory; measured
  quality and latency may improve, remain similar, or worsen.
- Descriptive targets: 30% reduction in peak allocated memory and no more than
  0.02 absolute F1 loss. These are engineering targets, not equivalence tests.
  A balanced 100-example sample supports only a small pilot conclusion.
- MAMI is among MemeLens's source datasets. Do not call this unseen-task or
  zero-shot generalization. No choices should be tuned on the 100 outcomes.

## 1. First action on the server: inventory only

Copy this `efficiency` directory to the server. On an allocated GPU node, from
the directory where you intend to store the model and outputs, run:

```bash
bash /path/to/efficiency/preflight.sh
```

Share the output before choosing the environment. `nvidia-smi`'s displayed CUDA
version is the driver's supported CUDA level, not proof that a matching PyTorch
runtime or CUDA toolkit is installed. Old scripts show FP16/SDPA intent and
multi-GPU support; they cannot establish current driver, RAM, disk or quotas.
Do not share passwords, private keys or tokens.

Upload only the original MAMI **test images** and **test_labels.txt**. Training
images, emoji assets and transcription tables are unnecessary for this follow-up.
The label file is tab separated: filename, binary misogyny, then optional subtypes.

## 2. Candidate environment, after checking the inventory

**Current route (supersedes the container/native recipes below):** a separate
Conda environment named `memelens-nf4` was successfully created. Its imports show
Python 3.10, PyTorch 2.5.1, CUDA runtime 11.8, torchvision 0.20.1 and `sm_70`.
The solver used `pytorch`, `nvidia/label/cuda-11.8.0`, and `defaults`, with
NumPy 1.26.4 and networkx 3.3 pinned. An explicit SymPy Conda pin was unavailable
and removed. See `requirements-legacy.txt` for the remaining Python dependencies.
The old bitsandbytes 0.42.0 path remains provisional until its CUDA backend and
the actual MemeLens model smoke tests pass. No container was needed for this route.
The runner now explicitly requires safetensors and disables remote custom code.
This uses an older runtime for both arms; it does not change the comparison method.

**Server update:** host glibc is 2.17. Native bitsandbytes 0.48.1 wheels require
glibc >=2.24, so do not use the native installation block below on this server.
Apptainer 1.3.2 is installed. The current route is a newer container userland
using `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, checked before installing
extra packages. Use `--cleanenv` and the container's `/opt/conda/bin/python` so
the host virtual/Conda environment cannot supply the interpreter. The existing
host virtual environment cannot be reused inside the container.
GPU access uses `apptainer exec --nv`; actual container permissions and GPU
library compatibility remain to be tested. No model experiment settings change.

Container preflight from the server project directory:

```bash
mkdir -p containers .apptainer-cache .apptainer-tmp
export APPTAINER_CACHEDIR="$PWD/.apptainer-cache"
export APPTAINER_TMPDIR="$PWD/.apptainer-tmp"
apptainer pull containers/pytorch-2.6.0-cu124.sif docker://pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
apptainer exec --cleanenv --nv containers/pytorch-2.6.0-cu124.sif /opt/conda/bin/python - <<'PY'
import platform
import sys
import torch
print('Python:', sys.version)
print('Interpreter:', sys.executable)
print('glibc:', platform.libc_ver())
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU architectures:', torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
PY
```

The native installation recipe below is retained only for hosts with compatible
glibc. On eslsrv11, wait for the container check before installing dependencies.

Updated after the server inventory: Linux x86-64, Python 3.10.18,
PyTorch 2.6.0 CUDA 12.4, torchvision 0.21.0,
Transformers 4.57.1, Accelerate 1.10.1, bitsandbytes 0.48.1. This is a proposed
compatible stack, not a GPU-tested lockfile. Use an isolated environment.

```bash
python3 -m venv .venv-memelens-efficiency
source .venv-memelens-efficiency/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r /path/to/efficiency/requirements.txt
python -m pip check
python -m pip freeze > environment-freeze.txt
```

Do not install a plain latest CUDA 13 PyTorch wheel, BF16/FlashAttention defaults,
or replace the system driver. The existing server environment confirms
PyTorch 2.6.0+cu124 detects the V100 and includes `sm_70`; use those same versions
in a clean virtual environment so setup cannot alter the existing `tse` environment.
The full model/NF4 execution still needs smoke tests. Driver is 550.54.15;
330 GB host RAM and 1.6 TB filesystem space were available, but user quota is unknown.
Another user's GPU job was active: wait for an exclusive allocation for benchmarking.
Reserve approximately 25–40 GB disk and
32 GB host RAM minimum (64 GB preferred), plus one exclusive 32 GB V100.
These are estimates; available memory and loading behavior decide feasibility.

References: [PyTorch previous wheels](https://pytorch.org/get-started/previous-versions/),
[bitsandbytes 0.48.1 compatibility](https://huggingface.co/docs/bitsandbytes/v0.48.1/en/installation),
[Qwen3-VL Transformers requirement](https://github.com/QwenLM/Qwen3-VL).

## 3. Freeze the data and public weights

From a new working directory, activate the environment, then set these paths:

```bash
EFFICIENCY_DIR=/path/to/efficiency
MAMI_IMAGES=/path/to/MAMI/test
MAMI_LABELS=/path/to/MAMI/test_labels.txt
mkdir -p efficiency-results
python "$EFFICIENCY_DIR/compare.py" prepare --images "$MAMI_IMAGES" --labels "$MAMI_LABELS" --output efficiency-results/manifest.json
python "$EFFICIENCY_DIR/compare.py" download --output efficiency-results/checkpoint.json
```

The manifest validates all supplied label/image pairs before sampling, hashes
image bytes and rejects duplicate IDs/missing files. No correctness-based sampling.
The download step resolves the public checkpoint once, caches its inference files
and records its immutable commit. Later runs use only cached files. Download time
and bytes depend on network/cache; do not count them as inference latency.
Keep the same manifest and checkpoint JSON for both modes. Outputs cannot overwrite
existing files/directories; use a new run directory for a deliberate retry.

## 4. Compatibility gate: run each in a fresh process

Request an exclusive GPU from the scheduler first, following your server's rules.
On a single-GPU node set `CUDA_VISIBLE_DEVICES=0`. If the scheduler already sets
this variable, preserve its assignment instead. Do not assume physical GPU 0 is yours.

```bash
python "$EFFICIENCY_DIR/compare.py" run --mode fp16 --checkpoint efficiency-results/checkpoint.json --manifest efficiency-results/manifest.json --smoke --output efficiency-results/smoke-fp16
python "$EFFICIENCY_DIR/compare.py" run --mode nf4 --checkpoint efficiency-results/checkpoint.json --manifest efficiency-results/manifest.json --smoke --output efficiency-results/smoke-nf4
```

Inspect five raw answers, loading/peak memory and module inventories for both modes.
Both smoke tests must complete. Any invalid warm-up answer stops the run for review.
Stop for OOM, non-finite/numerically suspect output, unsupported kernels, or offload.
Do not loosen parsing or change the image budget for only one mode to make it pass.
Use the five examples to estimate runtime; do not tune on the evaluation subset.
Allow at most two hours for compatibility troubleshooting; then discuss the
approved alternative: two image-token budgets with the FP16 model. If FP16 itself
does not fit, do not silently switch weights or compare against offloaded FP16.

## 5. Full comparison, only after the smoke gate passes

Keep other GPU workloads off the device and power settings unchanged. Each command
loads one mode and exits before the next starts. The sequential order is a small
pilot limitation; review per-pass timings for thermal/load drift.

```bash
python "$EFFICIENCY_DIR/compare.py" run --mode fp16 --checkpoint efficiency-results/checkpoint.json --manifest efficiency-results/manifest.json --output efficiency-results/fp16
python "$EFFICIENCY_DIR/compare.py" run --mode nf4 --checkpoint efficiency-results/checkpoint.json --manifest efficiency-results/manifest.json --output efficiency-results/nf4
python "$EFFICIENCY_DIR/compare.py" summarize --fp16 efficiency-results/fp16 --nf4 efficiency-results/nf4 --output efficiency-results/summary.json
```

Each run saves metadata, raw warm-up answers, flushed per-example/per-pass JSONL,
and a completion marker. Summarization rejects incomplete/mismatched checkpoints,
software, hardware identity, preprocessing, manifests, or actual tensor inputs.
If evaluation answers are invalid, report their rate and inspect raw outputs;
do not filter examples out after seeing which mode failed.

## Schedule and presentation

Day 1: environment/data and smoke gate (1–2 hours), then comparison and inspection
(3–5 hours provisional; use the smoke timing estimate). Day 2 morning: verify
results and create one table plus memory/latency figures. Day 2 afternoon: slides,
technical questions and rehearsal. Keep the GPU study small enough to protect that time.

The original robustness results supply motivation and a real failure: clean quality
decreased after robustness-oriented adaptation. The follow-up supplies a separate
deployment tradeoff using public weights. If NF4 is slower, report it as measured;
do not invent a failed result to satisfy the interview prompt.

## Local validation

```bash
python -m unittest discover -s /path/to/efficiency -p 'test_*.py'
```

These checks cover invalid-output scoring, deterministic/disjoint sampling,
paired-summary arithmetic and rejection of mismatched checkpoint weights. They
do not substitute for either actual V100 smoke test. No GPU results exist yet.
