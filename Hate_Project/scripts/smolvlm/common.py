"""SmolVLM-specific model, masking, batching and matched-adapter controls.

Heavy dependencies are lazy so planning/help and early-stopping tests need no GPU.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct'
REVISION = '7e3e67edbbed1bf9888184d9df282b700a323964'
LEGACY = Path(__file__).resolve().parents[1] / 'mami_lora_qlora'
CONDITIONS = ('clean_image_only', 'overlay_image_only',
              'clean_with_transcription', 'overlay_with_transcription')
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj',
           'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2', 'lm_head', 'embed_tokens']
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS,
            bias='none', task_type='CAUSAL_LM')
PACKAGES = ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes',
            'numpy', 'Pillow', 'huggingface-hub', 'pandas')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def legacy(name):
    # Import existing data validators without modifying the MemeLens pipeline.
    if str(LEGACY) not in sys.path:
        sys.path.insert(0, str(LEGACY))
    return __import__(name)


def versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def gpu_free():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Use CUDA_VISIBLE_DEVICES=0 for the existing V100 allocation')
    output = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    others = [int(line.strip()) for line in output.splitlines()
              if line.strip().isdigit() and int(line.strip()) != os.getpid()]
    if others:
        raise RuntimeError(f'GPU 0 is occupied by processes {others}; wait for the allocation')


def seed():
    import torch
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def processor_load():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION,
        local_files_only=True, trust_remote_code=False)
    # Keep the checkpoint's native resizing/cropping, not Qwen's pixel controls.
    processor.tokenizer.padding_side = 'right'
    return processor


def model_load(mode):
    gpu_free()
    import torch
    from transformers import Idefics3ForConditionalGeneration, BitsAndBytesConfig
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('A single visible CUDA GPU is required')
    if torch.cuda.get_device_capability() != (7, 0) or 'sm_70' not in torch.cuda.get_arch_list():
        raise RuntimeError('This comparison targets the existing V100 with native sm_70 support')
    kwargs = dict(revision=REVISION, local_files_only=True, trust_remote_code=False,
                  use_safetensors=True, dtype=torch.float16, device_map={'': 0},
                  attn_implementation='sdpa')
    if mode == 'nf4':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False)
    elif mode != 'fp16':
        raise ValueError(mode)
    # Existing bnb 0.42 cannot .to() a quantized model; force dispatch hooks.
    from functools import partial
    from unittest.mock import patch
    from transformers import modeling_utils
    with patch.object(modeling_utils, 'dispatch_model',
                      partial(modeling_utils.dispatch_model, force_hooks=True)):
        model = Idefics3ForConditionalGeneration.from_pretrained(MODEL, **kwargs)
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('CPU/disk offload is outside the comparison')
    if any(p.dtype == torch.bfloat16 for p in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError('Unexpected BF16 tensors on V100')
    quantized = [n for n, m in model.named_modules() if m.__class__.__name__ == 'Linear4bit']
    if bool(quantized) != (mode == 'nf4'):
        raise RuntimeError('Quantization inventory differs from the requested mode')
    return model, quantized


def adapter_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if 'lora_' in n]


def adapter_hash(parameters):
    return legacy('shared_adapter_init').canonical_hash(
        (n, p.detach().float().cpu().numpy()) for n, p in parameters)


def attach_adapter(model):
    import torch
    from peft import LoraConfig, get_peft_model
    matched = [n for n, _ in model.named_modules() if n.rsplit('.', 1)[-1] in TARGETS]
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(**LORA))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, _ in trainable):
        raise RuntimeError('Only adapters should train')
    # Sample the same sorted CPU FP32 arrays regardless of base weight dtype.
    init = legacy('shared_adapter_init')
    arrays = init.initialize_arrays({n: tuple(p.shape) for n, p in trainable})
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.data = torch.from_numpy(arrays[name]).to(parameter.device)
    initial_hash = adapter_hash(trainable)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    return model, trainable, dict(initial_adapter_sha256=initial_hash,
        adapter_targets=matched, lora=LORA,
        trainable_shapes={n: list(p.shape) for n, p in trainable},
        trainable_parameter_count=sum(p.numel() for _, p in trainable))


def messages(row):
    # This checkpoint's native template drops string-valued system content.
    # Preserve both instructions verbatim in a supported user text item.
    return [dict(role='user', content=[dict(type='image'),
                 dict(type='text', text=row['system_prompt'] + '\n\n' + row['user_prompt'])])]


def prompt_inputs(processor, row):
    from PIL import Image
    if digest(row['image_path']) != row['image_sha256']:
        raise ValueError(f"Image bytes changed: {row['image_path']}")
    text = processor.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)
    with Image.open(row['image_path']) as source:
        image = source.convert('RGB')
    return processor(text=[text], images=[image], padding=True, return_tensors='pt'), image


def make_batch(processor, row):
    prompt, image = prompt_inputs(processor, row)
    text = processor.apply_chat_template(messages(row) + [dict(role='assistant',
        content=[dict(type='text', text=row['target'])])],
        tokenize=False, add_generation_prompt=False)
    full = processor(text=[text], images=[image], padding=True, return_tensors='pt')
    n = prompt['input_ids'].shape[1]
    ids = full['input_ids'][0].tolist()
    if n >= len(ids) or ids[:n] != prompt['input_ids'][0].tolist():
        raise ValueError('Image-expanded prompt is not a strict prefix of the supervised sequence')
    if len(ids) > 8192:
        raise ValueError('Input exceeds the native 8192-token context; no silent truncation')
    suffix = processor.decode(ids[n:], skip_special_tokens=False)
    if not suffix.lstrip().startswith(row['target']):
        raise ValueError('Supervised suffix does not start with the target')
    labels = full['input_ids'].clone()
    labels[:, :n] = -100
    labels[full['attention_mask'] == 0] = -100
    full['labels'] = labels
    signature = hashlib.sha256()
    for key, value in sorted(full.items()):
        signature.update(key.encode())
        signature.update(str((list(value.shape), str(value.dtype))).encode())
        signature.update(value.contiguous().numpy().tobytes())
    return full, dict(original_id=row['original_id'], is_augmented=row['is_augmented'],
        prompt_tokens=n, full_tokens=len(ids), supervised_tokens=int((labels != -100).sum()),
        decoded_supervision=suffix, input_signature=signature.hexdigest(),
        pixel_shape=list(full['pixel_values'].shape))


def combine_batches(entries, pad_token_id):
    """Right-pad text and the variable native image-crop axis independently."""
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    required = {'input_ids', 'attention_mask', 'labels', 'pixel_values', 'pixel_attention_mask'}
    if not entries or any(set(batch) != required for batch, _ in entries):
        raise ValueError('Unexpected Idefics3 processor fields')
    max_length = max(b['input_ids'].shape[1] for b, _ in entries)
    max_crops = max(b['pixel_values'].shape[1] for b, _ in entries)
    result = {}
    for key, fill in (('input_ids', pad_token_id), ('attention_mask', 0), ('labels', -100)):
        result[key] = torch.cat([torch.nn.functional.pad(b[key], (0, max_length-b[key].shape[1]),
            value=fill) for b, _ in entries])
    for key in ('pixel_values', 'pixel_attention_mask'):
        shape = entries[0][0][key].shape[2:]
        if any(b[key].shape[2:] != shape for b, _ in entries):
            raise ValueError('Native crop spatial dimensions differ')
        padded = []
        for batch, _ in entries:
            tensor = batch[key]
            zeros = tensor.new_zeros((1, max_crops-tensor.shape[1], *shape))
            padded.append(torch.cat((tensor, zeros), dim=1))
        result[key] = torch.cat(padded)
    return BatchFeature(result)


def load_data(data, validation):
    records, prepared = legacy('training_smoke').load_training_records(Path(data))
    conditions, validated = legacy('prepare_validation_conditions').load(data, validation)
    checked = {}
    for row in records + [r for rows in conditions.values() for r in rows]:
        path = row['image_path']
        if path not in checked:
            checked[path] = digest(path)
        if checked[path] != row['image_sha256']:
            raise ValueError(f'Image bytes changed: {path}')
    return records, conditions, dict(prepared_complete_sha256=digest(Path(data)/'complete.json'),
        validation_complete_sha256=digest(Path(validation)/'complete.json'),
        train_records=len(records), validation_counts={k: len(v) for k, v in conditions.items()},
        unique_image_files=len(checked))


def metrics():
    path = Path(__file__).resolve().parents[1] / 'efficiency' / 'compare.py'
    spec = importlib.util.spec_from_file_location('_smolvlm_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(model, processor, conditions, out):
    """Full fixed conditions, target-free greedy generation; preserve training RNG."""
    import torch
    gpu_free()
    out.mkdir(parents=True)
    scoring = metrics()
    was_training = model.training
    states = random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    summaries = {}
    model.eval()
    try:
        for name in CONDITIONS:
            truth, predicted, strict = [], [], []
            started = __import__('time').perf_counter()
            torch.cuda.reset_peak_memory_stats()
            with (out/f'{name}.jsonl').open('x') as stream:
                for index, row in enumerate(conditions[name], 1):
                    batch, audit = make_batch(processor, row)
                    n = audit['prompt_tokens']
                    inputs = {k: v[:, :n] if k in ('input_ids', 'attention_mask') else v
                              for k, v in batch.items() if k != 'labels'}
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        output = model.generate(**inputs, max_new_tokens=5,
                            do_sample=False, num_beams=1, use_cache=True)
                    answer_ids = output[0, n:]
                    raw = processor.decode(answer_ids, skip_special_tokens=True)
                    truth.append(int(row['target'] == 'misogynous'))
                    predicted.append(scoring.parse_answer(raw))
                    strict.append(scoring.strict_answer(raw))
                    stream.write(json.dumps(dict(original_id=row['original_id'], true_label=truth[-1],
                        pred_label=predicted[-1], raw_answer=raw, strict_format=strict[-1],
                        generated_tokens=len(answer_ids), **{k: v for k, v in audit.items()
                        if k not in ('original_id', 'decoded_supervision')})) + '\n')
                    del inputs, output, answer_ids, batch
                    if index % 250 == 0:
                        print(f'{name}: {index}/{len(conditions[name])}', flush=True)
            torch.cuda.synchronize()
            summaries[name] = dict(scoring.quality(truth, predicted),
                strict_format_compliance=sum(strict)/len(strict),
                wall_seconds=__import__('time').perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
            print(f'{name}: macro-F1={summaries[name]["macro_f1"]:.4f}', flush=True)
    finally:
        model.train(was_training)
        model.config.use_cache = not was_training
        random.setstate(states[0])
        torch.set_rng_state(states[1])
        torch.cuda.set_rng_state_all(states[2])
    save_json(out/'summary.json', summaries)
    return summaries


def monitor_score(summaries):
    score = sum(summaries[k]['macro_f1'] for k in
        ('clean_with_transcription', 'overlay_with_transcription')) / 2
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid validation macro-F1')
    return score
