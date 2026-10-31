"""One disposable adapter optimizer step on four training records, V100 only.

No full training, checkpoint saving, test-set inference, or package changes.
GPU libraries are imported only inside run(), so data/masking checks run on CPU.
"""
import argparse
import ast
from collections import Counter, defaultdict
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
from unittest.mock import patch


MODEL = 'QCRI/MemeLens-VLM'
REVISION = 'a633183849df60a37251d29c3f1b0d379ae801f5'
LEGACY_TRAINER = Path(__file__).resolve().parents[1] / 'mami_finetuning' / 'train_finetuning.py'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def original_recipe():
    """Read literal settings without importing the old DDP/datasets program."""
    tree = ast.parse(LEGACY_TRAINER.read_text())
    calls = {name: [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == name]
             for name in ('LoraConfig', 'TrainingArguments')}
    if any(len(nodes) != 1 for nodes in calls.values()):
        raise ValueError('Expected one original LoRA config and one training config')
    lora = {k.arg: ast.literal_eval(k.value) for k in calls['LoraConfig'][0].keywords}
    arguments = {k.arg: k.value for k in calls['TrainingArguments'][0].keywords}
    lr = ast.literal_eval(arguments['learning_rate'])
    return lora, lr


def load_training_records(root):
    complete = json.loads((root / 'complete.json').read_text())
    if digest(root / 'manifest.json') != complete['manifest_sha256']:
        raise ValueError('Prepared manifest changed')
    for key, expected in complete['jsonl_sha256'].items():
        if digest(root / f'{key}.jsonl') != expected:
            raise ValueError(f'Prepared records changed: {key}')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['seed'] != 42 or manifest['validation_fraction'] != 0.1:
        raise ValueError('Unexpected prepared split')
    originals = {f"TRAINING/{r['file_name']}": r for r in manifest['originals']}
    records = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    if len(records) != 18000 or complete['counts']['train'] != 18000:
        raise ValueError('Expected 18,000 frozen training records')
    seen = set()
    counts = Counter()
    for row in records:
        source = originals[row['original_id']]
        if source['split'] != 'train' or row['split'] != 'train':
            raise ValueError('Validation original found in training records')
        if row['target'] != ('misogynous' if source['label'] else 'clean'):
            raise ValueError('Record label differs from split manifest')
        key = row['original_id'], row['is_augmented']
        if key in seen:
            raise ValueError('Duplicate original/augmentation pair')
        seen.add(key)
        counts[source['label']] += 1
    if len({r['original_id'] for r in records}) != 9000 or counts != {0: 9000, 1: 9000}:
        raise ValueError('Expected 9,000 balanced training originals with two records each')
    return records, complete


def select_smoke_records(records):
    """Fixed training-only pairs, preferring both transcription conditions."""
    pairs = defaultdict(dict)
    for row in records:
        pairs[row['original_id']][row['is_augmented']] = row
    selected = []
    for target in ('clean', 'misogynous'):
        candidates = [pair for _, pair in sorted(pairs.items())
                      if set(pair) == {False, True} and pair[False]['target'] == target]
        if not candidates:
            raise ValueError(f'Missing training pair for {target}')
        mixed = [p for p in candidates if
                 p[False]['user_prompt'].startswith('Meme Text:') !=
                 p[True]['user_prompt'].startswith('Meme Text:')]
        pair = (mixed or candidates)[0]
        selected.extend((pair[False], pair[True]))
    return selected


def prompt_boundary(prompt_ids, full_ids):
    # These must be the processor's image-expanded IDs, not text-only token IDs.
    n = len(prompt_ids)
    if not n or n >= len(full_ids) or full_ids[:n] != prompt_ids:
        raise ValueError('Image-expanded prompt is not a strict prefix of the full sequence')
    return n


def make_batch(processor, row):
    from PIL import Image
    if digest(row['image_path']) != row['image_sha256']:
        raise ValueError(f"Image changed: {row['image_path']}")
    messages = [dict(role='system', content=row['system_prompt']),
                dict(role='user', content=[dict(type='image'),
                     dict(type='text', text=row['user_prompt'])])]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(
        messages + [dict(role='assistant', content=row['target'])],
        tokenize=False, add_generation_prompt=False)
    with Image.open(row['image_path']) as source:
        image = source.convert('RGB')
    full = processor(text=[full_text], images=[image], padding=True, return_tensors='pt')
    prompt = processor(text=[prompt_text], images=[image], padding=True, return_tensors='pt')
    ids = full['input_ids'][0].tolist()
    boundary = prompt_boundary(prompt['input_ids'][0].tolist(), ids)
    supervised = processor.decode(ids[boundary:], skip_special_tokens=False)
    if not supervised.lstrip().startswith(row['target']):
        raise ValueError('Supervised tokens do not start with the target class')
    labels = full['input_ids'].clone()
    labels[:, :boundary] = -100
    labels[full['attention_mask'] == 0] = -100
    full['labels'] = labels
    signature = hashlib.sha256()
    for key, value in sorted(full.items()):
        signature.update(key.encode())
        signature.update(str(value.shape).encode())
        signature.update(str(value.dtype).encode())
        signature.update(value.contiguous().numpy().tobytes())
    audit = dict(original_id=row['original_id'], target=row['target'],
                 is_augmented=row['is_augmented'],
                 has_transcription=row['user_prompt'].startswith('Meme Text:'),
                 prompt_tokens=boundary, full_tokens=len(ids),
                 supervised_tokens=int((labels != -100).sum().item()),
                 decoded_supervision=supervised, input_signature=signature.hexdigest(),
                 image_grid_thw=full['image_grid_thw'].tolist())
    return full, audit


def adapter_hash(parameters):
    h = hashlib.sha256()
    for name, value in parameters:
        h.update(name.encode())
        h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    root, out = Path(args.data).resolve(), Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'Use a new smoke output directory: {out}')
    records, complete = load_training_records(root)
    selected = select_smoke_records(records)
    lora_config, lr = original_recipe()
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, modeling_utils
    from peft import LoraConfig, get_peft_model
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose only GPU 0 with CUDA_VISIBLE_DEVICES=0')
    if torch.cuda.get_device_capability() != (7, 0) or 'sm_70' not in torch.cuda.get_arch_list():
        raise RuntimeError('This gate requires the approved V100 and sm_70 support')
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'peft', 'numpy', 'Pillow')}
    for key, expected in dict(torch='2.5.1', transformers='4.57.1', accelerate='1.10.1',
                              bitsandbytes='0.42.0', peft='0.17.1').items():
        if versions[key].split('+')[0] != expected:
            raise RuntimeError(f'Unexpected {key} version: {versions[key]}')
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                                             trust_remote_code=False)
    processor.image_processor.size = dict(longest_edge=524288, shortest_edge=4096)
    if hasattr(processor.image_processor, 'max_pixels'):
        processor.image_processor.max_pixels = 524288
        processor.image_processor.min_pixels = 4096
    out.mkdir(parents=True)
    save_json(out / 'request.json', dict(mode=args.mode, checkpoint=dict(model=MODEL, revision=REVISION),
        versions=versions, prepared_complete_sha256=digest(root / 'complete.json'),
        prepared_manifest_sha256=complete['manifest_sha256'], legacy_trainer_sha256=digest(LEGACY_TRAINER),
        runner_sha256=digest(__file__), lora=lora_config, learning_rate=lr,
        smoke_microbatches=len(selected), gradient_checkpointing='nonreentrant',
        max_pixels=524288, min_pixels=4096, dispatch_force_hooks=True,
        base_policy='frozen loaded dtype; FP32 adapters; FP16 autocast',
        selected=[dict(original_id=r['original_id'], is_augmented=r['is_augmented']) for r in selected]))
    kwargs = dict(revision=REVISION, local_files_only=True, use_safetensors=True,
                  trust_remote_code=False, dtype=torch.float16, device_map={'': 0},
                  attn_implementation='sdpa')
    if args.mode == 'nf4':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    print(f'Loading {args.mode} base for disposable training check', flush=True)
    torch.cuda.reset_peak_memory_stats()
    with patch.object(modeling_utils, 'dispatch_model',
                      partial(modeling_utils.dispatch_model, force_hooks=True)):
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, **kwargs)
    torch.cuda.synchronize()
    load_peak = torch.cuda.max_memory_allocated()
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('Offloaded parameters invalidate this gate')
    quantized = [n for n, m in model.named_modules() if m.__class__.__name__ == 'Linear4bit']
    if bool(quantized) != (args.mode == 'nf4'):
        raise RuntimeError('Unexpected quantization inventory')
    matched = [n for n, _ in model.named_modules() if
               any(n == t or n.endswith('.' + t) for t in lora_config['target_modules'])]
    # Explicitly freeze the base without casting all its weights to FP32. The
    # stock k-bit preparation helper would also upcast the FP16 baseline.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = get_peft_model(model, LoraConfig(**lora_config))
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('Adapter injection placed parameters outside the GPU')
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, _ in trainable):
        raise RuntimeError('Only LoRA parameters may be trainable')
    for _, parameter in trainable:
        parameter.data = parameter.data.to(torch.float32)
    if any(p.dtype == torch.bfloat16 for p in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError('Unexpected BF16 state on V100')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    initial_hash = adapter_hash(trainable)
    save_json(out / 'model_inventory.json', dict(quantized_modules=quantized,
        adapter_target_matches=matched, trainable_parameters=[dict(name=n, shape=list(p.shape),
        dtype=str(p.dtype)) for n, p in trainable], trainable_parameter_count=sum(p.numel() for _, p in trainable),
        parameter_dtype_counts=dict(Counter(str(p.dtype) for p in model.parameters())),
        initial_adapter_sha256=initial_hash, gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
        load_peak_allocated_bytes=load_peak, processor=processor.image_processor.to_dict()))
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=lr, weight_decay=0.0, foreach=False)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    with (out / 'microbatches.jsonl').open('x') as stream:
        for index, row in enumerate(selected, 1):
            inputs, audit = make_batch(processor, row)
            inputs = inputs.to('cuda')
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                output = model(**inputs)
                loss = output.loss
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite training loss')
            scaler.scale(loss / len(selected)).backward()
            torch.cuda.synchronize()
            audit.update(loss=float(loss.detach().item()), forward_backward_ms=(time.perf_counter()-start)*1000)
            stream.write(json.dumps(audit) + '\n')
            stream.flush()
            print(f"Microbatch {index}/4: loss={audit['loss']:.4f}, supervised={audit['supervised_tokens']} tokens", flush=True)
            del inputs, output, loss
    scaler.unscale_(optimizer)
    gradients = [(n, p.grad) for n, p in trainable if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all().item() for _, g in gradients):
        raise RuntimeError('Missing or non-finite adapter gradients')
    if not any(torch.count_nonzero(g).item() for _, g in gradients):
        raise RuntimeError('All adapter gradients are zero')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen base received gradients')
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    if any(not torch.isfinite(p).all().item() for _, p in trainable):
        raise RuntimeError('Non-finite adapter weights after update')
    final_hash = adapter_hash(trainable)
    if final_hash == initial_hash:
        raise RuntimeError('Optimizer did not change adapter weights')
    result = dict(mode=args.mode, microbatches=4, optimizer_steps=1,
                  gradient_norm=float(norm.item()), gradient_parameter_count=len(gradients),
                  trainable_without_gradient=[n for n, p in trainable if p.grad is None],
                  initial_adapter_sha256=initial_hash, updated_adapter_sha256=final_hash,
                  peak_allocated_bytes=peak, peak_reserved_bytes=reserved,
                  qualification='Four training records only; not full-run memory or speed evidence',
                  adapter_saved=False)
    save_json(out / 'complete.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fp16', 'nf4'], required=True)
    parser.add_argument('--data', default='data/processed/mami_lora_qlora_seed42')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
