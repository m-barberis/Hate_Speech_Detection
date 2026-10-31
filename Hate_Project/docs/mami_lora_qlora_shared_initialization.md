# Shared FP32 adapter initialization for matched MAMI LoRA and QLoRA

The first server training smoke checks completed for both modes. Each exercised
four training records and one disposable optimizer step. All 508 trainable
parameter tensors received gradients, but their initial aggregate hashes differed.
No learned adapters were retained. These results establish basic compatibility,
not matched full training or a training-efficiency benchmark.

The user approved a common initialization on 31 October 2026. Existing v1 outputs
and prepared datasets remain unchanged.

## Explanation and remaining uncertainty

The v1 hash includes parameter names and values in traversal order. Differing
hashes do not identify the magnitude or cause of a numerical difference. The
old inventories permit comparison of names, shapes, order and input signatures,
but do not retain individual initial tensors for numerical diagnosis.

PEFT 0.17.1 initializes LoRA matrices and then moves/casts them to the base
layer's device/dtype. Floating-point base weights can cause a cast through FP16;
integer-stored quantized weights do not trigger that floating-point cast.
Subsequent conversion to FP32 cannot undo rounding. This is a plausible
explanation, not a demonstrated diagnosis of these particular server tensors.

Sources:

- https://github.com/huggingface/peft/blob/v0.17.1/src/peft/tuners/lora/layer.py
- https://github.com/huggingface/peft/blob/v0.17.1/src/peft/tuners/tuners_utils.py

## Implementation

`scripts/mami_lora_qlora/shared_adapter_init.py` first compares the successful
v1 smoke directories. It rejects different adapter names/shapes, source/data/
checkpoint/software controls, target modules, processors, GPUs, or input hashes.
Order-only differences are reported separately.

It then creates one CPU FP32 initialization using NumPy PCG64 seed 42 and sorted
parameter names. Linear LoRA A is uniform in [-1/sqrt(fan_in), +1/sqrt(fan_in)],
equivalent to default Kaiming-uniform initialization with a=sqrt(5). Linear LoRA B
and embedding LoRA A are zero. Embedding LoRA B is standard normal.

This retains default zero-update LoRA initialization distributions but uses a
documented new random sequence, not a reconstruction of either v1 run or
PyTorch's exact historical seed-42 values. No task outcomes enter initialization.

The output contains `adapter_init.npz` and `metadata.json`. NumPy loads arrays
with allow_pickle=False. Metadata records shapes, source controls, initializer,
file hash and canonical tensor hash. Existing outputs are never overwritten.
Creating this file needs no model weights or GPU.

The revised smoke runner requires `--adapter-init`. After PEFT constructs the
adapters and the runner converts them to FP32, it validates names/shapes and copies
the shared values. Before optimizer construction it hashes the actual GPU adapter
values and stops unless they match. Frozen base weights remain unchanged.

The new hash sorts by parameter name and includes names, shapes and little-endian
FP32 values. Its schema is `sorted-name-shape-fp32-le-v1`. These hashes cannot be
compared directly with v1's order-dependent hashes.

## Server commands after syncing code

Compare the old inventories and create the shared initialization on CPU:

```bash
python scripts/mami_lora_qlora/shared_adapter_init.py \
  --fp16 results/mami_lora_qlora/smoke-fp16-v1 \
  --nf4 results/mami_lora_qlora/smoke-nf4-v1 \
  --output results/mami_lora_qlora/shared_init_seed42
```

If successful and the GPU is available, repeat the disposable checks in fresh
sequential processes. Stop on a failure rather than changing settings:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/mami_lora_qlora/training_smoke.py \
  --mode fp16 \
  --adapter-init results/mami_lora_qlora/shared_init_seed42 \
  --output results/mami_lora_qlora/smoke-fp16-v2

CUDA_VISIBLE_DEVICES=0 python scripts/mami_lora_qlora/training_smoke.py \
  --mode nf4 \
  --adapter-init results/mami_lora_qlora/shared_init_seed42 \
  --output results/mami_lora_qlora/smoke-nf4-v2
```

Both runs must print the same verified initial adapter hash and retain matching
input signatures. Updated hashes may differ after the optimizer step. These checks
do not launch full training, change splits, regenerate overlays or extend validation.
