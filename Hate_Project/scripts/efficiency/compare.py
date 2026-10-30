"""Controlled MemeLens FP16/NF4 inference comparison. GPU imports occur only in run()."""
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

MODEL = 'QCRI/MemeLens-VLM'
SYSTEM = ('You are an automated classification API. You must output EXACTLY ONE WORD. '
          'Do not output any full sentences, explanations, or prefixes.')
PROMPT = ("Task: Identify if this meme contains misogyny.\n"
          "Allowed Labels: 'misogynous' or 'clean'.\nAnswer:")
SETTINGS = dict(max_pixels=512 * 32 * 32, min_pixels=4 * 32 * 32,
                max_new_tokens=10, do_sample=False, num_beams=1, use_cache=True,
                batch_size=1, attention='sdpa', compute_dtype='float16',
                double_quant=False, system=SYSTEM, prompt=PROMPT)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    # Exclusive creation prevents accidentally overwriting a previous run.
    with open(path, 'x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


SETTINGS["answer_parser"] = "leading-label-v1"


def parse_answer(text):
    answer = text.strip().lower()
    match = re.match(r"^(clean|misogynous)(?=$|[\s.,:;!?])", answer)
    if not match:
        return -1
    labels = set(re.findall(r"\b(?:clean|misogynous)\b", answer.splitlines()[0]))
    if len(labels) != 1:
        return -1
    return {"clean": 0, "misogynous": 1}[match.group(1)]


def strict_answer(text):
    return text.strip().lower() in ("clean", "misogynous")


def quality(truth, predicted):
    if not truth or len(truth) != len(predicted):
        raise ValueError('Expected nonempty, aligned predictions')
    if any(t not in (0, 1) for t in truth) or any(p not in (-1, 0, 1) for p in predicted):
        raise ValueError('Unexpected label')
    # Invalid outputs (-1) count as FN for their true class, never as clean.
    f1 = []
    for c in (0, 1):
        tp = sum(t == c and p == c for t, p in zip(truth, predicted))
        fp = sum(t != c and p == c for t, p in zip(truth, predicted))
        fn = sum(t == c and p != c for t, p in zip(truth, predicted))
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d else 0.0)
    return dict(n=len(truth), macro_f1=sum(f1) / 2,
                accuracy=sum(t == p for t, p in zip(truth, predicted)) / len(truth),
                invalid_count=predicted.count(-1),
                confusion_rows_true_0_1_cols_pred_0_1_invalid=[
                    [sum(t == c and p == v for t, p in zip(truth, predicted))
                     for v in (0, 1, -1)] for c in (0, 1)])


def percentile(values, q):
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def prepare(args):
    root = Path(args.images).resolve()
    groups = {0: [], 1: []}
    seen = set()
    with open(args.labels, newline='', encoding='utf-8-sig') as stream:
        for row in csv.reader(stream, delimiter='\t'):
            if not row:
                continue
            if row[0].strip() == 'file_name':
                continue
            if len(row) < 2 or row[1].strip() not in ('0', '1'):
                raise ValueError(f'Invalid label row: {row[:2]}')
            name, label = row[0].strip(), int(row[1])
            if name in seen:
                raise ValueError(f'Duplicate image ID: {name}')
            seen.add(name)
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Image outside root: {name}')
            if not path.is_file():
                # Existing evaluators support jpg/png filename discrepancies.
                path = path.with_suffix('.png')
            if not path.is_file():
                raise FileNotFoundError(f'Missing image: {name}')
            groups[label].append(dict(file_name=name, label=label,
                                      image_path=str(path), sha256=digest(path)))
    rng = random.Random(42)
    for group in groups.values():
        group.sort(key=lambda row: row['file_name'])
        rng.shuffle(group)
        if len(group) < 53:
            raise ValueError('Need at least 53 images per class for evaluation + warm-up')
    evaluation = groups[0][:50] + groups[1][:50]
    rng.shuffle(evaluation)
    warmup = groups[0][50:53] + groups[1][50:52]
    save_json(args.output, dict(seed=42, labels_sha256=digest(args.labels),
                                evaluation=evaluation, warmup=warmup))
    print(f'Frozen 100 evaluation images and 5 disjoint warm-ups: {args.output}')


def download(args):
    from huggingface_hub import model_info, snapshot_download
    revision = model_info(MODEL).sha
    # Download once, then use the immutable commit for both modes.
    snapshot_download(MODEL, revision=revision,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    save_json(args.output, dict(model=MODEL, revision=revision))
    print(f'Cached {MODEL} at {revision}; recorded in {args.output}')


def run(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'Use a fresh output directory: {out}')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError('This approved comparison is for a Volta V100 (CC 7.0)')
    if 'sm_70' not in torch.cuda.get_arch_list():
        raise RuntimeError('PyTorch wheel lacks native sm_70 support')
    pin = json.loads(Path(args.checkpoint).read_text())
    if pin['model'] != MODEL or not re.fullmatch(r'[0-9a-f]{40}', pin['revision']):
        raise ValueError('Expected an immutable public MemeLens checkpoint commit')
    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest['warmup'] + manifest['evaluation']
    for row in rows:
        if digest(row['image_path']) != row['sha256']:
            raise ValueError(f"Image hash mismatch: {row['file_name']}")
    if len(set(r['file_name'] for r in rows)) != len(rows):
        raise ValueError('Warm-up/evaluation overlap or duplicate IDs')
    out.mkdir(parents=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    processor = AutoProcessor.from_pretrained(MODEL, revision=pin['revision'],
                                             local_files_only=True, trust_remote_code=False)
    processor.image_processor.size = dict(longest_edge=SETTINGS['max_pixels'],
                                          shortest_edge=SETTINGS['min_pixels'])
    if hasattr(processor.image_processor, 'max_pixels'):
        processor.image_processor.max_pixels = SETTINGS['max_pixels']
        processor.image_processor.min_pixels = SETTINGS['min_pixels']
    kwargs = dict(revision=pin['revision'], local_files_only=True, use_safetensors=True,
                  trust_remote_code=False, dtype=torch.float16,
                  device_map={'': 0}, attn_implementation='sdpa')
    if args.mode == 'nf4':
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    # Legacy dispatch compatibility fix, identical for FP16 and NF4.
    from functools import partial
    from unittest.mock import patch
    from transformers import modeling_utils
    SETTINGS["dispatch_force_hooks"] = True
    with patch.object(
        modeling_utils,
        "dispatch_model",
        partial(modeling_utils.dispatch_model, force_hooks=True),
    ):
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, **kwargs).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    load_peak = torch.cuda.max_memory_allocated()
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('CPU/disk offload invalidates this comparison')
    quantized = [name for name, module in model.named_modules()
                 if module.__class__.__name__ == 'Linear4bit']
    if (args.mode == 'nf4') != bool(quantized):
        raise RuntimeError('Quantized module inventory does not match requested mode')
    if any(p.dtype == torch.bfloat16 for p in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError('Unexpected BF16 parameters/buffers on V100')
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'torchvision', 'transformers', 'accelerate', 'bitsandbytes', 'Pillow', 'numpy')}
    try:
        gpu_status = subprocess.check_output(['nvidia-smi'], text=True)
    except (OSError, subprocess.CalledProcessError):
        gpu_status = 'unavailable'
    metadata = dict(mode=args.mode, smoke=args.smoke, checkpoint=pin, settings=SETTINGS,
                    manifest_sha256=digest(args.manifest), versions=versions,
                    torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
                    cudnn_version=torch.backends.cudnn.version(),
                    gpu_status=gpu_status, python=__import__('sys').version,
                    quantized_modules=quantized, device_map=str(model.hf_device_map),
                    parameter_dtypes=dict(collections.Counter(str(p.dtype) for p in model.parameters())),
                    processor=processor.image_processor.to_dict(),
                    load_seconds=load_seconds, load_peak_allocated_bytes=load_peak,
                    loaded_allocated_bytes=torch.cuda.memory_allocated(),
                    loaded_reserved_bytes=torch.cuda.memory_reserved(),
                    generation_config=model.generation_config.to_dict(), passes=3)
    save_json(out / 'metadata.json', metadata)

    def predict(row, validate_logits=False):
        with Image.open(row['image_path']) as source:
            image = source.convert('RGB')
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': [{'type': 'image'},
                                                {'type': 'text', 'text': PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True)
        # Hash the actual model inputs, not only filenames/settings.
        h = hashlib.sha256()
        for key, value in sorted(inputs.items()):
            h.update(key.encode())
            h.update(str(value.shape).encode())
            h.update(str(value.dtype).encode())
            h.update(value.contiguous().numpy().tobytes())
        signature = h.hexdigest()
        grid = inputs['image_grid_thw'].tolist()
        token_count = inputs['input_ids'].shape[1]
        inputs = inputs.to('cuda')
        # Includes model weights, inputs and inference temporaries; excludes CPU preprocessing.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                       num_beams=1, use_cache=True,
                                       return_dict_in_generate=validate_logits,
                                       output_scores=validate_logits)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        if validate_logits:
            # -inf can legitimately mask tokens; NaN or +inf indicates failure.
            for scores in generated.scores:
                if torch.isnan(scores).any().item() or torch.isposinf(scores).any().item():
                    raise RuntimeError('Non-finite generation scores during warm-up')
            sequences = generated.sequences
        else:
            sequences = generated
        answer_ids = sequences[0, token_count:]
        raw = processor.decode(answer_ids, skip_special_tokens=True)
        return dict(file_name=row['file_name'], true_label=row['label'],
                    pred_label=parse_answer(raw), raw_answer=raw,
                    strict_format=strict_answer(raw),
                    input_signature=signature, input_tokens=token_count,
                    image_grid_thw=grid, generated_tokens=len(answer_ids),
                    latency_ms=elapsed, peak_allocated_bytes=peak,
                    peak_reserved_bytes=reserved)

    # Always preserve raw warm-up outputs, including invalid answers.
    with open(out / 'warmup.jsonl', 'x') as stream:
        for row in manifest['warmup']:
            result = predict(row, validate_logits=True)
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            print('Warm-up:', row['file_name'], repr(result['raw_answer']), flush=True)
            if result['pred_label'] == -1:
                raise RuntimeError('Invalid warm-up answer; inspect before proceeding')
    if args.smoke:
        save_json(out / 'complete.json', dict(smoke=True, warmup_examples=5))
        return
    with open(out / 'predictions.jsonl', 'x') as stream:
        for pass_id in range(3):
            order = list(manifest['evaluation'])
            random.Random(42 + pass_id).shuffle(order)
            for index, row in enumerate(order):
                result = predict(row)
                result['pass_id'] = pass_id
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                print(f"{args.mode}: pass {pass_id + 1}/3, {index + 1}/100, "
                      f"{result['latency_ms']:.1f} ms", flush=True)
    # Exceptions abort instead of silently writing a clean prediction.
    save_json(out / 'complete.json', dict(smoke=False, evaluation_examples=100, passes=3))


def load_results(path):
    path = Path(path)
    complete = json.loads((path / 'complete.json').read_text())
    if complete['smoke']:
        raise ValueError('Smoke runs cannot be summarized as evaluation results')
    meta = json.loads((path / 'metadata.json').read_text())
    records = [json.loads(line) for line in (path / 'predictions.jsonl').read_text().splitlines()]
    indexed = {(r['pass_id'], r['file_name']): r for r in records}
    if len(records) != 300 or len(indexed) != 300:
        raise ValueError('Expected exactly 300 unique example/pass records')
    ids = sorted(name for p, name in indexed if p == 0)
    if len(ids) != 100 or set(p for p, _ in indexed) != {0, 1, 2}:
        raise ValueError('Expected 100 images and 3 passes')
    for name in ids:
        first = indexed[0, name]
        for p in (1, 2):
            row = indexed[p, name]
            for key in ('true_label', 'input_signature', 'raw_answer'):
                if first[key] != row[key]:
                    raise ValueError(f'Unstable {key} across repeated passes: {name}')
    return meta, records, indexed, ids


def summarize(args):
    a, ar, ai, ids = load_results(args.fp16)
    b, br, bi, bids = load_results(args.nf4)
    if a['mode'] != 'fp16' or b['mode'] != 'nf4' or ids != bids:
        raise ValueError('Incorrect modes or mismatched evaluation IDs')
    for key in ('checkpoint', 'settings', 'manifest_sha256', 'versions', 'torch_cuda',
                'gpu', 'gpu_uuid', 'cudnn_version', 'python', 'processor', 'generation_config', 'passes'):
        if a[key] != b[key]:
            raise ValueError(f'Comparison control differs: {key}')
    for name in ids:
        for key in ('true_label', 'input_signature'):
            if ai[0, name][key] != bi[0, name][key]:
                raise ValueError(f'Mismatched {key}: {name}')
    truth = [ai[0, name]['true_label'] for name in ids]
    ap = [ai[0, name]['pred_label'] for name in ids]
    bp = [bi[0, name]['pred_label'] for name in ids]
    result = dict(checkpoint=a['checkpoint'], n=100, quality_pass=0,
                  fp16=quality(truth, ap), nf4=quality(truth, bp),
                  disagreement_rate=sum(x != y for x, y in zip(ap, bp)) / 100,
                  delta_macro_f1_nf4_minus_fp16=quality(truth, bp)['macro_f1'] - quality(truth, ap)['macro_f1'])
    rng = random.Random(42)
    deltas = []
    # Stratified paired bootstrap preserves the deliberately balanced sampling design.
    groups = [[i for i, t in enumerate(truth) if t == c] for c in (0, 1)]
    if list(map(len, groups)) != [50, 50]:
        raise ValueError('Expected 50 examples per class')
    for _ in range(2000):
        chosen = [rng.choice(group) for group in groups for _ in range(len(group))]
        ts = [truth[i] for i in chosen]
        deltas.append(quality(ts, [bp[i] for i in chosen])['macro_f1'] -
                      quality(ts, [ap[i] for i in chosen])['macro_f1'])
    result['paired_stratified_bootstrap_delta_95_interval'] = [percentile(deltas, .025), percentile(deltas, .975)]
    for mode, meta, records in [('fp16', a, ar), ('nf4', b, br)]:
        result[mode].update(dict(
            strict_format_compliance=sum(strict_answer(r['raw_answer']) for r in records if r['pass_id'] == 0) / len(ids),
            median_latency_ms=statistics.median(r['latency_ms'] for r in records),
            p95_latency_ms=percentile([r['latency_ms'] for r in records], .95),
            pass_median_latency_ms=[statistics.median(r['latency_ms'] for r in records if r['pass_id'] == p)
                                    for p in (0, 1, 2)],
            mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in records),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in records),
            loaded_allocated_bytes=meta['loaded_allocated_bytes'],
            load_peak_allocated_bytes=meta['load_peak_allocated_bytes']))
    result['peak_allocated_reduction_fraction'] = 1 - result['nf4']['peak_allocated_bytes'] / result['fp16']['peak_allocated_bytes']
    result['latency_ratio_nf4_over_fp16'] = result['nf4']['median_latency_ms'] / result['fp16']['median_latency_ms']
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--images', required=True)
    prep.add_argument('--labels', required=True)
    prep.add_argument('--output', required=True)
    prep.set_defaults(func=prepare)
    cache = commands.add_parser('download')
    cache.add_argument('--output', required=True)
    cache.set_defaults(func=download)
    bench = commands.add_parser('run')
    bench.add_argument('--mode', choices=['fp16', 'nf4'], required=True)
    bench.add_argument('--checkpoint', required=True)
    bench.add_argument('--manifest', required=True)
    bench.add_argument('--output', required=True)
    bench.add_argument('--smoke', action='store_true')
    bench.set_defaults(func=run)
    summary = commands.add_parser('summarize')
    summary.add_argument('--fp16', required=True)
    summary.add_argument('--nf4', required=True)
    summary.add_argument('--output', required=True)
    summary.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
