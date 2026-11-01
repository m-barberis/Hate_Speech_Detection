"""Separate evaluator for approved unseen layouts; original grid evaluator unchanged."""
import argparse
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
from unittest.mock import patch
import numpy as np
from PIL import Image
import prepare_unseen_patterns as test_data
import prepare_test_conditions as grid_data
import evaluate_test as original
import train_comparison as training
import training_smoke as common

gpu_free = original.gpu_free
training_gates = original.training_gates
verify_test_training_link = original.verify_test_training_link
ARMS = original.ARMS
DEFAULT_ROOT = 'results/mami_lora_qlora/unseen-patterns-v1'


def read_completed(folder, conditions, preparation_sha):
    done = json.loads((folder/'complete.json').read_text())
    if (done['test_preparation_sha256'] != preparation_sha or
        done['predictions_sha256'] != common.digest(folder/'predictions.jsonl') or
        done['metadata_sha256'] != common.digest(folder/'metadata.json')):
        raise ValueError('Saved unseen-pattern evaluation changed')
    rows = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
    expected = [(c, r['original_id']) for c in test_data.CONDITIONS for r in conditions[c]]
    if [(r['condition'], r['original_id']) for r in rows] != expected:
        raise ValueError('Missing, duplicate or reordered pattern predictions')
    for row, src in zip(rows, (r for c in test_data.CONDITIONS for r in conditions[c])):
        if row['true_label'] != int(src['target'] == 'misogynous') or row['pattern'] != src['pattern']:
            raise ValueError('Prediction label or pattern differs')
    return done, rows


def run(args, *, data_module=test_data, completion_reader=read_completed, runner_path=__file__):
    conditions, _ = data_module.load(args.data)
    verify_test_training_link(args)
    gates = training_gates(args)
    arm, mode = args.arm, args.arm.split('-')[0]
    out = Path(args.output_root).resolve()/arm
    preparation_sha = common.digest(Path(args.data)/'complete.json')
    training_hashes = {m: g['completion_sha256'] for m, g in gates.items()}
    if out.exists():
        if not args.reuse_complete:
            raise FileExistsError(f'Preserve existing output: {out}')
        done, _ = completion_reader(out, conditions, preparation_sha)
        if done['arm'] != arm or done['training_completion_sha256'] != training_hashes:
            raise ValueError('Completed evaluation belongs to different trained runs')
        print(f'Reusing verified completed test arm: {arm}', flush=True)
        return
    gpu_free()
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, modeling_utils
    from peft import PeftModel
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError('Expose only the approved V100: CUDA_VISIBLE_DEVICES=0')
    if 'sm_70' not in torch.cuda.get_arch_list():
        raise RuntimeError('PyTorch lacks native V100 support')
    versions = {name: importlib.metadata.version(name) for name in gates[mode]['request']['versions']}
    if versions != gates[mode]['request']['versions']:
        raise RuntimeError('Evaluation software differs from the completed training runs')
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    processor = AutoProcessor.from_pretrained(common.MODEL, revision=common.REVISION,
        local_files_only=True, trust_remote_code=False)
    processor.image_processor.size = dict(longest_edge=524288, shortest_edge=4096)
    if hasattr(processor.image_processor, 'max_pixels'):
        processor.image_processor.max_pixels = 524288
        processor.image_processor.min_pixels = 4096
    kwargs = dict(revision=common.REVISION, local_files_only=True, use_safetensors=True,
        trust_remote_code=False, dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa')
    if mode == 'nf4':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    print(f'Loading {arm}', flush=True)
    with patch.object(modeling_utils, 'dispatch_model', partial(modeling_utils.dispatch_model, force_hooks=True)):
        model = Qwen3VLForConditionalGeneration.from_pretrained(common.MODEL, **kwargs)
    adapted = not arm.endswith('-base')
    if adapted:
        model = PeftModel.from_pretrained(model, gates[mode]['root']/'epoch-2/adapter',
            is_trainable=False, autocast_adapter_dtype=True)
        adapters = [(n, p) for n, p in model.named_parameters() if 'lora_' in n]
        if common.adapter_hash(adapters) != gates[mode]['complete']['final_adapter_sha256']:
            raise ValueError('Reloaded epoch-two adapter hash differs from final trained tensors')
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('CPU/offloaded parameters are not allowed')
    quantized = [n for n, m in model.named_modules() if m.__class__.__name__ == 'Linear4bit']
    if bool(quantized) != (mode == 'nf4'):
        raise RuntimeError('Quantization inventory differs from requested arm')
    if any(p.dtype == torch.bfloat16 for p in list(model.parameters())+list(model.buffers())):
        raise RuntimeError('Unexpected BF16 tensors on V100')
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = True
    out.mkdir(parents=True)
    metadata = dict(arm=arm, checkpoint=dict(model=common.MODEL, revision=common.REVISION),
        test_preparation_sha256=preparation_sha, training_completion_sha256=training_hashes,
        versions=versions, gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
        processor=processor.image_processor.to_dict(), quantized_modules=quantized,
        adapter_hash=gates[mode]['complete']['final_adapter_sha256'] if adapted else None,
        adapter_merged=False, runner_sha256=common.digest(runner_path),
        inference_source_sha256=common.digest(__file__),
        scoring_source_sha256=common.digest(Path(__file__).resolve().parents[1]/'efficiency/compare.py'),
        generation=dict(max_new_tokens=5, do_sample=False, num_beams=1, use_cache=True),
        image_budget=dict(max_pixels=524288, min_pixels=4096), inference_microbatch=1,
        generation_config=model.generation_config.to_dict(), cpu_nice=os.getpriority(os.PRIO_PROCESS, 0))
    common.save_json(out/'metadata.json', metadata)
    metrics = training.metric_functions()
    summaries = {}
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (out/'predictions.jsonl').open('x') as stream:
        for condition in data_module.CONDITIONS:
            gpu_free()
            saved = []
            for index, row in enumerate(conditions[condition], 1):
                # Hash actual prompt-only inputs; never construct a teacher-forced target.
                if common.digest(row['image_path']) != row['image_sha256']:
                    raise ValueError('Test image changed after preflight')
                with Image.open(row['image_path']) as source:
                    image = source.convert('RGB')
                messages = [dict(role='system', content=row['system_prompt']),
                    dict(role='user', content=[dict(type='image'), dict(type='text', text=row['user_prompt'])])]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch = processor(text=[text], images=[image], return_tensors='pt', padding=True)
                signature = hashlib.sha256()
                for key, value in sorted(batch.items()):
                    signature.update(key.encode()); signature.update(str(value.shape).encode())
                    signature.update(str(value.dtype).encode()); signature.update(value.contiguous().numpy().tobytes())
                n = batch['input_ids'].shape[1]
                batch = batch.to('cuda')
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    generated = model.generate(**batch, max_new_tokens=5, do_sample=False, num_beams=1, use_cache=True)
                raw = processor.decode(generated[0, n:], skip_special_tokens=True)
                result = dict(condition=condition, original_id=row['original_id'],
                    true_label=int(row['target'] == 'misogynous'), pred_label=metrics.parse_answer(raw),
                    raw_answer=raw, strict_format=metrics.strict_answer(raw),
                    generated_tokens=int(generated.shape[1]-n), input_tokens=n,
                    input_signature=signature.hexdigest(), emoji_code=row['emoji_code'], pattern=row['pattern'])
                stream.write(json.dumps(result, ensure_ascii=False)+'\n'); stream.flush()
                saved.append(result)
                del batch, generated
                if index % 250 == 0:
                    print(f'{arm} {condition}: {index}/{len(conditions[condition])}', flush=True)
            summary = metrics.quality([r['true_label'] for r in saved], [r['pred_label'] for r in saved])
            summary['strict_format_compliance'] = sum(r['strict_format'] for r in saved)/len(saved)
            summaries[condition] = summary
            print(f"{arm} {condition}: macro-F1={summary['macro_f1']:.4f}", flush=True)
    gpu_free()
    torch.cuda.synchronize()
    done = dict(arm=arm,
        n_originals=len({r['original_id'] for rows in conditions.values() for r in rows}),
        predictions=sum(len(rows) for rows in conditions.values()), validation_used_for_selection=False,
        test_preparation_sha256=preparation_sha, training_completion_sha256=training_hashes,
        predictions_sha256=common.digest(out/'predictions.jsonl'), metadata_sha256=common.digest(out/'metadata.json'),
        metrics=summaries, evaluation_wall_seconds=time.perf_counter()-start,
        peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_inference_reserved_bytes=torch.cuda.max_memory_reserved(),
        qualification='One quality pass; wall time is not a controlled repeated latency benchmark; adapters unmerged')
    common.save_json(out/'complete.json', done)
    print(json.dumps(done, indent=2))


def paired_delta(truth, a, b, patterns, repeats=2000):
    truth, a, b, patterns = map(np.asarray, (truth, a, b, patterns))
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero((truth == label) & (patterns == pattern))
              for label in (0, 1) for pattern in sorted(set(patterns.tolist()))]
    indices = np.concatenate([rng.choice(g, size=(repeats, len(g)), replace=True) for g in groups if len(g)], axis=1)
    deltas = original.macro_f1_array(truth[indices], b[indices])-original.macro_f1_array(truth[indices], a[indices])
    return dict(delta_macro_f1=float(original.macro_f1_array(truth, b)-original.macro_f1_array(truth, a)),
        paired_label_pattern_stratified_95_interval=np.quantile(deltas, [.025, .975]).tolist(), bootstrap_replicates=repeats)


def summarize(args):
    conditions, _ = test_data.load(args.data)
    verify_test_training_link(args)
    gates = training_gates(args)
    manifest = json.loads((Path(args.data)/'manifest.json').read_text())
    clean_conditions, _ = grid_data.load(manifest['source_data'])
    source_sha = common.digest(Path(manifest['source_data'])/'complete.json')
    pattern_sha = common.digest(Path(args.data)/'complete.json')
    all_records, clean_records, arm_summaries, metas, previous_metas = {}, {}, {}, {}, {}
    metric_functions = training.metric_functions()
    for arm in ARMS:
        folder = Path(args.output_root)/arm
        done, rows = read_completed(folder, conditions, pattern_sha)
        clean_folder = Path(args.clean_results)/arm
        clean_done, previous = original.read_completed(clean_folder, clean_conditions, source_sha)
        expected_training = {m: gate['completion_sha256'] for m, gate in gates.items()}
        if done['arm'] != arm or clean_done['arm'] != arm or done['training_completion_sha256'] != expected_training or clean_done['training_completion_sha256'] != expected_training:
            raise ValueError('Clean/pattern predictions belong to different models or training runs')
        meta = json.loads((folder/'metadata.json').read_text())
        previous_meta = json.loads((clean_folder/'metadata.json').read_text())
        # Different evaluator source hashes are intentional and both retained.
        for key in ('checkpoint', 'versions', 'processor', 'generation', 'image_budget',
                    'inference_microbatch', 'generation_config', 'adapter_hash', 'adapter_merged', 'scoring_source_sha256'):
            if meta[key] != previous_meta[key]:
                raise ValueError(f'Cannot reuse clean predictions; control differs: {arm} {key}')
        metas[arm] = meta
        previous_metas[arm] = previous_meta
        all_records[arm] = {(r['condition'], r['original_id']): r for r in rows}
        clean_records[arm] = {(r['condition'], r['original_id']): r for r in previous}
        by_pattern = {}
        for condition in test_data.CONDITIONS:
            by_pattern[condition] = {}
            for pattern in ('all',)+test_data.PATTERNS:
                group = [r for r in rows if r['condition'] == condition and (pattern == 'all' or r['pattern'] == pattern)]
                summary = metric_functions.quality([r['true_label'] for r in group], [r['pred_label'] for r in group])
                summary['strict_format_compliance'] = sum(r['strict_format'] for r in group)/len(group)
                by_pattern[condition][pattern] = summary
        arm_summaries[arm] = dict(by_pattern=by_pattern,
            pattern_completion_sha256=common.digest(folder/'complete.json'),
            reused_clean_completion_sha256=common.digest(clean_folder/'complete.json'),
            pattern_evaluator_sha256=meta['runner_sha256'], original_evaluator_sha256=previous_meta['runner_sha256'])
    for arm in ARMS:
        if previous_metas[arm]['runner_sha256'] != previous_metas[ARMS[0]]['runner_sha256']:
            raise ValueError('Original evaluations used different evaluator sources')
        for key in ('checkpoint', 'versions', 'gpu', 'gpu_uuid', 'processor', 'generation',
                    'image_budget', 'inference_microbatch', 'generation_config', 'runner_sha256', 'scoring_source_sha256'):
            if metas[arm][key] != metas[ARMS[0]][key]:
                raise ValueError(f'Pattern arms differ: {key}')
        for key, row in all_records[arm].items():
            for field in ('true_label', 'input_signature', 'pattern', 'emoji_code'):
                if row[field] != all_records[ARMS[0]][key][field]:
                    raise ValueError(f'Pattern test input differs across arms: {key} {field}')
        for key, row in clean_records[arm].items():
            for field in ('true_label', 'input_signature'):
                if row[field] != clean_records[ARMS[0]][key][field]:
                    raise ValueError(f'Reused clean/grid inputs differ across arms: {key} {field}')
    contrasts, corruption_effects = {}, {}
    for condition in test_data.CONDITIONS:
        contrasts[condition], corruption_effects[condition] = {}, {}
        source_rows = conditions[condition]
        for pattern in ('all',)+test_data.PATTERNS:
            selected = [r for r in source_rows if pattern == 'all' or r['pattern'] == pattern]
            ids = [r['original_id'] for r in selected]
            truth = [int(r['target'] == 'misogynous') for r in selected]
            patterns = [r['pattern'] for r in selected]
            predictions = {arm: [all_records[arm][condition, name]['pred_label'] for name in ids] for arm in ARMS}
            contrasts[condition][pattern] = {}
            for a, b in (('fp16-base', 'fp16-lora'), ('nf4-base', 'nf4-qlora'),
                         ('fp16-base', 'nf4-base'), ('fp16-lora', 'nf4-qlora')):
                contrasts[condition][pattern][f'{b}_minus_{a}'] = paired_delta(truth, predictions[a], predictions[b], patterns)
            corruption_effects[condition][pattern] = {}
            for arm in ARMS:
                clean_condition = condition.replace('overlay_', 'clean_')
                clean = [clean_records[arm][clean_condition, name]['pred_label'] for name in ids]
                grid = [clean_records[arm][condition, name]['pred_label'] for name in ids]
                corruption_effects[condition][pattern][arm] = dict(
                    pattern_minus_clean=paired_delta(truth, clean, predictions[arm], patterns),
                    pattern_minus_grid=paired_delta(truth, grid, predictions[arm], patterns))
    report = dict(n_originals=1000, new_predictions=8000, reused_clean_predictions=8000, reused_grid_predictions=8000,
        test_preparation_sha256=pattern_sha, pattern_counts=manifest['counts'],
        arms=arm_summaries, paired_model_contrasts=contrasts, paired_corruption_effects=corruption_effects,
        qualifications=['One fixed pattern and unseen emoji per meme; not all three patterns on every image.',
            'Primary aggregate is the fixed class-balanced mixture; individual patterns have about 333 examples.',
            'Counts, sizes, positions, overlaps and coverage differ, so this does not isolate pure spatial geometry.',
            'Paired seed-42 bootstrap, 2000 replicates, stratified by true label and assigned pattern.',
            'Unadjusted pointwise intervals, no equivalence margin; single training seed and split.',
            'Transcription-present conditions primary; original supplied transcription retained.',
            'Previously completed clean/grid predictions reused after integrity and inference-control verification.',
            'Test images held out of adapter training; 100 previously used in inference benchmarking.'])
    path = Path(args.output_root)/'comparison.json'
    common.save_json(path, report)
    print(f'Saved unseen-pattern comparison: {path}')
    print(json.dumps(arm_summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'summarize'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--data', default=test_data.DEFAULT_OUTPUT)
    parser.add_argument('--output-root', default=DEFAULT_ROOT)
    parser.add_argument('--clean-results', default=original.DEFAULT_ROOT)
    parser.add_argument('--training-data', default='data/processed/mami_lora_qlora_seed42')
    parser.add_argument('--fp16-training', default='results/mami_lora_qlora/fp16-training-v2')
    parser.add_argument('--nf4-training', default='results/mami_lora_qlora/nf4-training-v2')
    parser.add_argument('--reuse-complete', action='store_true')
    args = parser.parse_args()
    if args.command == 'run' and not args.arm:
        parser.error('--arm is required for run')
    (run if args.command == 'run' else summarize)(args)


if __name__ == '__main__':
    main()
