"""Evaluate fixed bases and epoch-two adapters on the frozen 1,000-meme test."""
import argparse
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import prepare_comparison_data as prep
import prepare_test_conditions as test_data
import train_comparison as training
import training_smoke as common

ARMS = ('fp16-base', 'nf4-base', 'fp16-lora', 'nf4-qlora')
DEFAULT_ROOT = 'results/mami_lora_qlora/test-1000-v1'


def gpu_free():
    output = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True)
    others = [int(line.strip()) for line in output.splitlines()
              if line.strip().isdigit() and int(line.strip()) != os.getpid()]
    if others:
        raise RuntimeError(f'GPU 0 has competing compute processes: {others}')


def training_gates(args):
    gates = {}
    for mode in ('fp16', 'nf4'):
        root = Path(getattr(args, mode+'_training')).resolve()
        done = json.loads((root/'complete.json').read_text())
        plan = json.loads((root/'training_plan.json').read_text())
        request = json.loads((root/'request.json').read_text())
        if done['mode'] != mode or done['optimizer_steps'] != 750 or done['epochs'] != 2 or done['final_checkpoint_policy'] != 'epoch_2_without_validation_selection':
            raise ValueError(f'Expected completed fixed epoch-two {mode} run')
        if common.digest(root/'training_plan.json') != done['training_plan_sha256']:
            raise ValueError('Training plan changed')
        if request['checkpoint'] != dict(model=common.MODEL, revision=common.REVISION):
            raise ValueError('Training base revision differs')
        marker = json.loads((root/'epoch-2/checkpoint_complete.json').read_text())
        if marker['epoch'] != 2 or marker['global_step'] != 750 or marker['initial_adapter_sha256'] != done['initial_adapter_sha256']:
            raise ValueError('Epoch-two checkpoint marker differs')
        files = {name: sha for name, sha in marker['files'].items() if name.startswith('adapter/')}
        if 'adapter/adapter_config.json' not in files or not any(n.endswith('.safetensors') for n in files):
            raise ValueError('Expected safetensors adapter checkpoint')
        for name, sha in files.items():
            if common.digest(root/'epoch-2'/name) != sha:
                raise ValueError(f'Adapter checkpoint changed: {name}')
        if common.digest(Path(args.training_data)/'complete.json') != plan['prepared_complete_sha256']:
            raise ValueError('Training data preparation differs')
        gates[mode] = dict(root=root, complete=done, plan=plan, request=request,
            completion_sha256=common.digest(root/'complete.json'), checkpoint_sha256=common.digest(root/'epoch-2/checkpoint_complete.json'))
    for key in ('training_order_sha256', 'prepared_complete_sha256', 'validation_complete_sha256',
                'initial_adapter_sha256', 'microbatch', 'gradient_accumulation', 'effective_global_batch',
                'runner_sha256', 'batching_source_sha256', 'training_logits', 'epochs', 'optimizer_updates'):
        if gates['fp16']['plan'][key] != gates['nf4']['plan'][key]:
            raise ValueError(f'Matched training control differs: {key}')
    if gates['fp16']['request']['versions'] != gates['nf4']['request']['versions']:
        raise ValueError('Training software versions differ')
    return gates


def record_key(row):
    return row['condition'], row['original_id']


def read_completed(folder, conditions, preparation_sha):
    done = json.loads((folder/'complete.json').read_text())
    if (done['test_preparation_sha256'] != preparation_sha or
            done['predictions_sha256'] != common.digest(folder/'predictions.jsonl') or
            done['metadata_sha256'] != common.digest(folder/'metadata.json')):
        raise ValueError('Completed test run or preparation changed')
    rows = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
    expected = [(c, r['original_id']) for c in test_data.CONDITIONS for r in conditions[c]]
    if [record_key(r) for r in rows] != expected:
        raise ValueError('Missing, duplicate or reordered test predictions')
    for row, source in zip(rows, (r for c in test_data.CONDITIONS for r in conditions[c])):
        if row['true_label'] != int(source['target'] == 'misogynous'):
            raise ValueError('Saved test labels differ')
    return done, rows


def verify_test_training_link(args):
    manifest = json.loads((Path(args.data)/'manifest.json').read_text())
    if manifest['training_manifest_sha256'] != common.digest(Path(args.training_data)/'manifest.json'):
        raise ValueError('Test preparation belongs to a different training split')


def run(args):
    conditions, _ = test_data.load(args.data)
    verify_test_training_link(args)
    gates = training_gates(args)
    arm, mode = args.arm, args.arm.split('-')[0]
    out = Path(args.output_root).resolve()/arm
    preparation_sha = common.digest(Path(args.data)/'complete.json')
    training_hashes = {m: g['completion_sha256'] for m, g in gates.items()}
    if out.exists():
        if not args.reuse_complete:
            raise FileExistsError(f'Preserve existing output: {out}')
        done, _ = read_completed(out, conditions, preparation_sha)
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
        adapter_merged=False, runner_sha256=common.digest(__file__),
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
        for condition in test_data.CONDITIONS:
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
                    input_signature=signature.hexdigest(), emoji_code=row['emoji_code'])
                stream.write(json.dumps(result, ensure_ascii=False)+'\n'); stream.flush()
                saved.append(result)
                del batch, generated
                if index % 250 == 0:
                    print(f'{arm} {condition}: {index}/1000', flush=True)
            summary = metrics.quality([r['true_label'] for r in saved], [r['pred_label'] for r in saved])
            summary['strict_format_compliance'] = sum(r['strict_format'] for r in saved)/len(saved)
            summaries[condition] = summary
            print(f"{arm} {condition}: macro-F1={summary['macro_f1']:.4f}", flush=True)
    gpu_free()
    torch.cuda.synchronize()
    done = dict(arm=arm, n_originals=1000, predictions=4000, validation_used_for_selection=False,
        test_preparation_sha256=preparation_sha, training_completion_sha256=training_hashes,
        predictions_sha256=common.digest(out/'predictions.jsonl'), metadata_sha256=common.digest(out/'metadata.json'),
        metrics=summaries, evaluation_wall_seconds=time.perf_counter()-start,
        peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_inference_reserved_bytes=torch.cuda.max_memory_reserved(),
        qualification='One quality pass; wall time is not a controlled repeated latency benchmark; adapters unmerged')
    common.save_json(out/'complete.json', done)
    print(json.dumps(done, indent=2))


def macro_f1_array(truth, predicted):
    """Last axis contains aligned observations; invalid predictions remain errors."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    result = 0.0
    for label in (0, 1):
        tp = ((truth == label) & (predicted == label)).sum(axis=-1)
        fp = ((truth != label) & (predicted == label)).sum(axis=-1)
        fn = ((truth == label) & (predicted != label)).sum(axis=-1)
        denominator = 2*tp+fp+fn
        result = result + np.divide(2*tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)/2
    return result


def paired_delta(truth, a, b, repeats=2000):
    truth, a, b = map(np.asarray, (truth, a, b))
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(truth == c) for c in (0, 1)]
    if any(len(g) == 0 for g in groups):
        raise ValueError('Both true classes required')
    indices = np.concatenate([rng.choice(g, size=(repeats, len(g)), replace=True) for g in groups], axis=1)
    deltas = macro_f1_array(truth[indices], b[indices])-macro_f1_array(truth[indices], a[indices])
    return dict(delta_macro_f1=float(macro_f1_array(truth, b)-macro_f1_array(truth, a)),
                paired_stratified_bootstrap_95_interval=np.quantile(deltas, [.025, .975]).tolist(), bootstrap_replicates=repeats)


def summarize(args):
    conditions, _ = test_data.load(args.data)
    verify_test_training_link(args)
    gates = training_gates(args)
    preparation_sha = common.digest(Path(args.data)/'complete.json')
    loaded, records, metadata = {}, {}, {}
    for arm in ARMS:
        folder = Path(args.output_root)/arm
        done, rows = read_completed(folder, conditions, preparation_sha)
        if done['arm'] != arm or done['training_completion_sha256'] != {m: g['completion_sha256'] for m, g in gates.items()}:
            raise ValueError('Evaluation belongs to different trained runs')
        if done['metadata_sha256'] != common.digest(folder/'metadata.json'):
            raise ValueError('Test metadata changed')
        metadata[arm] = json.loads((folder/'metadata.json').read_text())
        loaded[arm] = done
        records[arm] = {record_key(r): r for r in rows}
    reference = ARMS[0]
    for arm in ARMS:
        for key in ('checkpoint', 'versions', 'gpu', 'gpu_uuid', 'processor', 'generation',
                    'image_budget', 'inference_microbatch', 'generation_config', 'runner_sha256', 'scoring_source_sha256'):
            if metadata[arm][key] != metadata[reference][key]:
                raise ValueError(f'Test comparison control differs: {key}')
        for key, row in records[arm].items():
            for field in ('true_label', 'input_signature'):
                if row[field] != records[reference][key][field]:
                    raise ValueError(f'Test input alignment differs: {arm} {key} {field}')
    contrasts = {}
    for condition in test_data.CONDITIONS:
        ids = [r['original_id'] for r in conditions[condition]]
        truth = [records[reference][condition, name]['true_label'] for name in ids]
        pairs = [('fp16-base', 'fp16-lora'), ('nf4-base', 'nf4-qlora'),
                 ('fp16-base', 'nf4-base'), ('fp16-lora', 'nf4-qlora')]
        contrasts[condition] = {}
        for a, b in pairs:
            ap = [records[a][condition, name]['pred_label'] for name in ids]
            bp = [records[b][condition, name]['pred_label'] for name in ids]
            contrasts[condition][f'{b}_minus_{a}'] = paired_delta(truth, ap, bp)
    robustness = {}
    for arm in ARMS:
        robustness[arm] = {}
        for suffix in ('image_only', 'with_transcription'):
            ids = [r['original_id'] for r in conditions['clean_'+suffix]]
            truth = [records[arm]['clean_'+suffix, name]['true_label'] for name in ids]
            clean = [records[arm]['clean_'+suffix, name]['pred_label'] for name in ids]
            overlay = [records[arm]['overlay_'+suffix, name]['pred_label'] for name in ids]
            robustness[arm][suffix] = paired_delta(truth, clean, overlay)
    report = dict(n_originals=1000, total_predictions=16000, arms=loaded,
        paired_model_contrasts=contrasts, overlay_minus_clean=robustness,
        qualifications=['Fixed epoch-two adapters, no test checkpoint selection.',
            'Unadjusted pointwise paired 95% intervals; no multiple-comparison correction or equivalence margin.',
            'Single seed and split; intervals reflect test-example resampling, not training-run variability.',
            'Each meme has one fixed unseen-emoji overlay; not a full emoji or geometry sweep.',
            'Transcription-present conditions primary; supplied transcription remains unchanged by overlay.',
            'These images were held out of adapter training, but 100 were previously used for inference benchmarking.'])
    destination = Path(args.output_root)/'comparison.json'
    common.save_json(destination, report)
    print(f'Saved full paired comparison: {destination}')
    print(json.dumps({a: loaded[a]['metrics'] for a in ARMS}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'summarize'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--data', default=test_data.DEFAULT_OUTPUT)
    parser.add_argument('--training-data', default='data/processed/mami_lora_qlora_seed42')
    parser.add_argument('--fp16-training', default='results/mami_lora_qlora/fp16-training-v2')
    parser.add_argument('--nf4-training', default='results/mami_lora_qlora/nf4-training-v2')
    parser.add_argument('--output-root', default=DEFAULT_ROOT)
    parser.add_argument('--reuse-complete', action='store_true')
    args = parser.parse_args()
    if args.command == 'run' and not args.arm:
        parser.error('--arm is required for run')
    (run if args.command == 'run' else summarize)(args)


if __name__ == '__main__':
    main()
