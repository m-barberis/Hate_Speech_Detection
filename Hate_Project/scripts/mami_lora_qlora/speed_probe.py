"""Disposable, fixed-workload training speed checks; never edits a live run."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

import numpy as np
import shared_adapter_init as shared
import training_smoke as common
import train_comparison as training
from training_batches import combine_batches, full_record_losses


VARIANTS = ('baseline', 'cached', 'answer-only', 'batch2')
RUN_VARIANTS = VARIANTS + ('batch2-full',)


def other_gpu_processes():
    # Query physical GPU 0, matching the approved server commands. Refuse to
    # compete with a live training process, irrespective of its owner.
    output = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return [int(line.strip()) for line in output.splitlines()
            if line.strip().isdigit() and int(line.strip()) != os.getpid()]


def selected_positions(labels):
    """Select logit positions predicting all unmasked next-token labels."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or (labels[:, 0] != -100).any():
        raise ValueError('Expected masked first token and a batch/sequence label matrix')
    positions = np.flatnonzero((labels[:, 1:] != -100).any(axis=0))
    if not len(positions) or (labels[:, 1:] != -100).sum(axis=1).min() == 0:
        raise ValueError('Every record needs supervised answer tokens')
    return positions.tolist()


def forward_loss(model, batch, answer_only, positions=None):
    import torch
    if not answer_only:
        if batch['labels'].shape[0] == 1:
            return model(**batch).loss
        return full_record_losses(model, batch).mean()
    indices = torch.tensor(positions, dtype=torch.long, device=batch['input_ids'].device)
    inputs = {key: value for key, value in batch.items() if key != 'labels'}
    logits = model(**inputs, logits_to_keep=indices).logits
    targets = batch['labels'].index_select(1, indices + 1)
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1), ignore_index=-100, reduction='none').view_as(targets)
    counts = (targets != -100).sum(dim=1)
    return (loss.sum(dim=1) / counts).mean()


def copy_initial(parameters, arrays):
    import torch
    with torch.no_grad():
        for name, parameter in parameters:
            parameter.copy_(torch.from_numpy(arrays[name]).to(parameter.device))


def groups_for(rows, width):
    if len(rows) % width:
        raise ValueError('Benchmark records do not divide into complete microbatches')
    return [rows[i:i+width] for i in range(0, len(rows), width)]


def benchmark_rows(records, warmup):
    excluded = {r['original_id'] for r in warmup}
    order = training.epoch_order(len(records), 1)
    selected = [records[i] for i in order if records[i]['original_id'] not in excluded][:48]
    if len(selected) != 48:
        raise ValueError('Need 48 training records disjoint from warm-up originals')
    return selected


def run(args):
    if other_gpu_processes():
        raise RuntimeError('GPU 0 has another compute process. Do not benchmark alongside the live training run.')
    out, data = Path(args.output).resolve(), Path(args.data).resolve()
    if out.exists():
        raise FileExistsError('Use a new speed-probe output directory')
    rows, complete = common.load_training_records(data)
    warmup = common.select_smoke_records(rows)
    selected = benchmark_rows(rows, warmup)
    model, processor, parameters, lr = common.initialize_model(
        args, data, out, complete, selected, purpose='speed_probe')
    import torch
    arrays, initial_meta = shared.load(args.adapter_init)
    width = 2 if args.variant in ('batch2', 'batch2-full') else 1
    answer_only = args.variant in ('answer-only', 'batch2')
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError('Explicit tokenizer padding token required for batching')
    start = time.perf_counter()
    cached = {id(row): common.make_batch(processor, row) for row in selected + warmup}
    cache_seconds = time.perf_counter() - start
    signatures = [cached[id(r)][1]['input_signature'] for r in selected]

    def get_batch(group, use_cache):
        entries = [cached[id(row)] if use_cache else common.make_batch(processor, row) for row in group]
        if any(entry[1]['input_signature'] != cached[id(row)][1]['input_signature']
               for row, entry in zip(group, entries)):
            raise ValueError('Recomputed input differs from its verified cached counterpart')
        if len(entries) == 1:
            # .to() mutates BatchFeature; keep cached tensors on CPU.
            from transformers.feature_extraction_utils import BatchFeature
            batch = BatchFeature(dict(entries[0][0]))
        else:
            batch = combine_batches(entries, pad_id)
        positions = selected_positions(batch['labels'].numpy()) if answer_only else None
        return batch.to('cuda'), positions

    common.save_json(out / 'probe_configuration.json', dict(variant=args.variant,
        mode=args.mode, runner_sha256=common.digest(__file__), microbatch=width,
        accumulation=48 // width, records_per_timing_pass=48, timing_passes=args.repetitions,
        timing_input_signatures=signatures,
        correctness_policy='dropout disabled; same four training records; no optimizer step',
        timing_policy='dropout enabled; reset common initialization and fresh AdamW each pass',
        cache_policy='52-record CPU RAM cache; preparation excluded from timed passes',
        cpu_nice=os.getpriority(os.PRIO_PROCESS, 0)))

    # Equivalence diagnostic: dropout disabled and no optimizer step. Save
    # gradients for quantitative cross-process comparison, not model training.
    model.eval()
    model.zero_grad(set_to_none=True)
    correction_loss = 0.0
    for group in groups_for(warmup, width):
        batch, positions = get_batch(group, True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = forward_loss(model, batch, answer_only, positions)
        if not torch.isfinite(loss).item():
            raise RuntimeError('Non-finite correctness-check loss')
        correction_loss += float(loss.item()) * len(group) / len(warmup)
        (loss * (1024.0 * len(group) / len(warmup))).backward()
        del loss, batch
    gradient_arrays = {}
    for name, parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError('Missing/non-finite correctness gradient')
        gradient_arrays[name] = parameter.grad.detach().float().cpu().numpy().copy() / 1024.0
    with (out / 'correctness_gradients.npz').open('xb') as stream:
        np.savez(stream, **gradient_arrays)
    model.zero_grad(set_to_none=True)
    del gradient_arrays

    # Training-mode warm-up exercises checkpointing and the selected batch shape.
    model.train()
    for group in groups_for(warmup, width):
        batch, positions = get_batch(group, True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = forward_loss(model, batch, answer_only, positions)
        (loss * 1024.0).backward()
        del loss, batch
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    passes = []
    for repetition in range(args.repetitions):
        competing = other_gpu_processes()
        if competing:
            raise RuntimeError(f'Another GPU process appeared: {competing}')
        copy_initial(parameters, arrays)
        model.zero_grad(set_to_none=True)
        gc.collect()
        optimizer = torch.optim.AdamW([p for _, p in parameters], lr=lr, weight_decay=0.0, foreach=False)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
        random.seed(42 + repetition)
        torch.manual_seed(42 + repetition)
        torch.cuda.manual_seed_all(42 + repetition)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        mean_loss = 0.0
        for group in groups_for(selected, width):
            batch, positions = get_batch(group, args.variant != 'baseline')
            with torch.autocast('cuda', dtype=torch.float16):
                loss = forward_loss(model, batch, answer_only, positions)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite timing-pass loss')
            mean_loss += float(loss.item()) * len(group) / 48
            scaler.scale(loss * (len(group) / 48)).backward()
            del loss, batch
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        result = dict(repetition=repetition, wall_seconds=time.perf_counter()-start,
            mean_record_loss=mean_loss, gradient_norm=float(norm.item()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        passes.append(result)
        common.save_json(out / f'pass-{repetition}.json', result)
        print(json.dumps(result), flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler
    stress = []
    if args.stress_sample:
        if width != 2:
            raise ValueError('Sample memory check requires microbatch two')
        # Bounded check of the existing sample, not an exhaustive dataset scan.
        for metric in ('full_tokens', 'vision_patches'):
            if other_gpu_processes():
                raise RuntimeError('Another GPU process appeared before memory check')
            ranked = sorted(selected + warmup, reverse=True,
                key=lambda row: cached[id(row)][1]['full_tokens'] if metric == 'full_tokens'
                else cached[id(row)][0]['pixel_values'].shape[0])[:2]
            copy_initial(parameters, arrays)
            optimizer = torch.optim.AdamW([p for _, p in parameters], lr=lr, weight_decay=0.0, foreach=False)
            scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
            torch.cuda.reset_peak_memory_stats()
            batch, positions = get_batch(ranked, True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = forward_loss(model, batch, answer_only, positions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            stress.append(dict(metric=metric, records=[cached[id(r)][1] for r in ranked],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            optimizer.zero_grad(set_to_none=True)
            del optimizer, scaler, loss, batch
    if other_gpu_processes():
        raise RuntimeError('Another GPU process appeared; timings require review')
    result = dict(mode=args.mode, variant=args.variant, microbatch=width, records_per_pass=48,
        passes=passes, median_wall_seconds=statistics.median(p['wall_seconds'] for p in passes),
        cache_preparation_seconds=cache_seconds, initial_adapter_sha256=initial_meta['initial_adapter_sha256'],
        correctness_mean_record_loss=correction_loss,
        correctness_gradient_file_sha256=common.digest(out / 'correctness_gradients.npz'),
        timing_input_signatures=signatures,
        correctness_ids=[dict(original_id=r['original_id'], is_augmented=r['is_augmented']) for r in warmup],
        qualification='Disposable first optimizer steps, 48-record RAM cache, training-only inputs; not full-run throughput or bitwise equivalence',
        sample_memory_checks=stress,
        cached_sample_tensor_bytes=sum(t.numel()*t.element_size()
            for batch, _ in cached.values() for t in batch.values()),
        adapter_saved=False)
    common.save_json(out / 'complete.json', result)
    print(json.dumps(result, indent=2))


def summarize(args):
    root = Path(args.root)
    report = {}
    for mode in ('fp16', 'nf4'):
        loaded = {}
        inventories = {}
        requests = {}
        for variant in args.variants:
            folder = root / f'{mode}-{variant}'
            done = json.loads((folder / 'complete.json').read_text())
            if done['mode'] != mode or done['variant'] != variant:
                raise ValueError('Incorrect speed-probe mode or variant')
            if common.digest(folder / 'correctness_gradients.npz') != done['correctness_gradient_file_sha256']:
                raise ValueError('Correctness gradient archive changed')
            loaded[variant] = done
            requests[variant] = json.loads((folder / 'request.json').read_text())
            inventories[variant] = json.loads((folder / 'model_inventory.json').read_text())
        baseline = loaded[args.reference]
        reference = root / f'{mode}-{args.reference}' / 'correctness_gradients.npz'
        comparisons = {}
        with np.load(reference, allow_pickle=False) as original_gradients:
            denominator = sum(float(np.square(original_gradients[n].astype(np.float64)).sum()) for n in original_gradients.files)
            for variant, result in loaded.items():
                for key in ('initial_adapter_sha256', 'timing_input_signatures', 'correctness_ids'):
                    if result[key] != baseline[key]:
                        raise ValueError(f'Benchmark control differs: {key}')
                for key in ('checkpoint', 'versions', 'lora', 'prepared_complete_sha256',
                            'max_pixels', 'min_pixels', 'gradient_checkpointing', 'shared_init_weights_sha256'):
                    if requests[variant][key] != requests[args.reference][key]:
                        raise ValueError(f'Request control differs: {key}')
                for key in ('gpu', 'gpu_uuid', 'processor', 'quantized_modules', 'trainable_parameters'):
                    if inventories[variant][key] != inventories[args.reference][key]:
                        raise ValueError(f'Model inventory differs: {key}')
                with np.load(root / f'{mode}-{variant}' / 'correctness_gradients.npz', allow_pickle=False) as gradients:
                    if set(gradients.files) != set(original_gradients.files):
                        raise ValueError('Correctness gradient parameter names differ')
                    numerator = 0.0
                    maximum = 0.0
                    for name in gradients.files:
                        if gradients[name].shape != original_gradients[name].shape:
                            raise ValueError('Correctness gradient shapes differ')
                        difference = gradients[name].astype(np.float64) - original_gradients[name].astype(np.float64)
                        numerator += float(np.square(difference).sum())
                        maximum = max(maximum, float(np.abs(difference).max()))
                comparisons[variant] = dict(median_wall_seconds=result['median_wall_seconds'],
                    speedup_over_baseline=baseline['median_wall_seconds']/result['median_wall_seconds'],
                    peak_allocated_bytes=max(p['peak_allocated_bytes'] for p in result['passes']),
                    correctness_loss_delta=result['correctness_mean_record_loss']-baseline['correctness_mean_record_loss'],
                    gradient_relative_l2=(numerator/denominator)**.5 if denominator else None,
                    gradient_max_absolute_difference=maximum,
                    reference_variant=args.reference,
                    sample_memory_checks=result.get('sample_memory_checks', []),
                    cached_sample_tensor_bytes=result.get('cached_sample_tensor_bytes'),
                    qualification='Review numerical differences; no automatic equivalence claim or variant selection')
        report[mode] = comparisons
    common.save_json(Path(args.output), report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    probe = commands.add_parser('run')
    probe.add_argument('--mode', choices=['fp16', 'nf4'], required=True)
    probe.add_argument('--variant', choices=RUN_VARIANTS, required=True)
    probe.add_argument('--repetitions', type=int, choices=(1, 3), default=3)
    probe.add_argument('--stress-sample', action='store_true')
    probe.add_argument('--data', default='data/processed/mami_lora_qlora_seed42')
    probe.add_argument('--adapter-init', default='results/mami_lora_qlora/shared_init_seed42')
    probe.add_argument('--output', required=True)
    summary = commands.add_parser('summarize')
    summary.add_argument('--root', default='results/mami_lora_qlora/speed-probe-v1')
    summary.add_argument('--variants', nargs='+', choices=RUN_VARIANTS, default=VARIANTS)
    summary.add_argument('--reference', choices=RUN_VARIANTS, default='baseline')
    summary.add_argument('--output', default='results/mami_lora_qlora/speed-probe-v1/summary.json')
    args = parser.parse_args()
    (run if args.command == 'run' else summarize)(args)


if __name__ == '__main__':
    main()
