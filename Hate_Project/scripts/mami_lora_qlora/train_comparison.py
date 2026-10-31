"""Matched single-V100 FP16 LoRA / NF4 QLoRA training, final epoch selection."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import time
from contextlib import closing

import prepare_validation_conditions as validation_data
import shared_adapter_init as shared
import training_smoke as common
import training_batches as batching


EPOCHS = 2
MICROBATCH = 2
ACCUMULATION = 24
EFFECTIVE_BATCH = MICROBATCH * ACCUMULATION
SEED = 42


def epoch_order(count, epoch):
    order = list(range(count))
    random.Random(SEED + epoch - 1).shuffle(order)
    return order


def update_groups(order, accumulation=EFFECTIVE_BATCH):
    if not order or len(order) % accumulation:
        raise ValueError('Training epoch must contain complete accumulation groups')
    return [order[i:i + accumulation] for i in range(0, len(order), accumulation)]


def lr_multiplier(step, total):
    if total <= 0 or step < 0 or step > total:
        raise ValueError('Invalid linear learning-rate schedule position')
    return 1.0 - step / total


def order_hash(records, order):
    import hashlib
    identities = [(records[i]['original_id'], records[i]['is_augmented']) for i in order]
    return hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()


def metric_functions():
    path = Path(__file__).resolve().parents[1] / 'efficiency' / 'compare.py'
    spec = importlib.util.spec_from_file_location('_existing_efficiency_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generation_inputs_from_batch(batch, prompt_tokens):
    """Remove the teacher-forced answer and labels before free generation."""
    full_length = batch['input_ids'].shape[1]
    if not 0 < prompt_tokens < full_length or 'labels' not in batch:
        raise ValueError('Expected a supervised batch and strict prompt boundary')
    result = {k: v for k, v in batch.items() if k != 'labels'}
    for key in ('input_ids', 'attention_mask', 'token_type_ids', 'mm_token_type_ids'):
        if key in result:
            if result[key].shape[1] != full_length:
                raise ValueError(f'Unexpected sequence alignment: {key}')
            result[key] = result[key][:, :prompt_tokens]
    return result


def verify_gates(args, initial_meta):
    _, controls, _ = shared.compare_runs(args.smoke_fp16, args.smoke_nf4)
    if controls['checkpoint'] != dict(model=common.MODEL, revision=common.REVISION):
        raise ValueError('Smoke checkpoint differs')
    if controls['prepared_complete_sha256'] != common.digest(Path(args.data) / 'complete.json'):
        raise ValueError('Smoke gate belongs to different prepared data')
    if controls['lora'] != initial_meta['controls']['lora']:
        raise ValueError('Smoke adapter configuration differs')
    for folder in (args.smoke_fp16, args.smoke_nf4):
        done = json.loads((Path(folder) / 'complete.json').read_text())
        if (done['initial_adapter_sha256'] != initial_meta['initial_adapter_sha256'] or
                done['shared_init_weights_sha256'] != initial_meta['weights_sha256'] or
                done['adapter_hash_schema'] != shared.HASH_SCHEMA or
                done['optimizer_steps'] != 1 or done['microbatches'] != 4 or
                done['trainable_without_gradient']):
            raise ValueError('Matching successful v2 training smoke gates are required')


def preflight(args):
    root = Path(args.data).resolve()
    records, prepared = common.load_training_records(root)
    conditions, validated = validation_data.load(root, args.validation)
    _, initial_meta = shared.load(args.adapter_init)
    verify_gates(args, initial_meta)
    checked = {}
    for row in records + [r for rows in conditions.values() for r in rows]:
        path = row['image_path']
        if path not in checked:
            checked[path] = common.digest(path)
            if len(checked) % 2000 == 0:
                print(f'Preflight verified {len(checked)} image files', flush=True)
        if checked[path] != row['image_sha256']:
            raise ValueError(f'Image changed: {path}')
    orders = [epoch_order(len(records), epoch) for epoch in (1, 2)]
    updates = [len(update_groups(order)) for order in orders]
    if updates != [375, 375]:
        raise ValueError('Expected 375 optimizer updates per epoch')
    report = dict(mode=args.mode, original_training_images=9000, training_records=len(records),
        epochs=EPOCHS, microbatch=MICROBATCH, gradient_accumulation=ACCUMULATION,
        effective_global_batch=48, updates_per_epoch=updates, optimizer_updates=sum(updates),
        validation_counts={name: len(rows) for name, rows in conditions.items()},
        training_order_sha256=[order_hash(records, order) for order in orders],
        initial_adapter_sha256=initial_meta['initial_adapter_sha256'],
        unique_verified_image_files=len(checked),
        prepared_complete_sha256=common.digest(root / 'complete.json'),
        validation_complete_sha256=common.digest(Path(args.validation) / 'complete.json'),
        final_checkpoint_policy='epoch_2_without_validation_selection',
        training_logits='full_sequence', cpu_prefetch_microbatches=2,
        preprocessing_policy='one CPU producer; at most two pending pairs plus consumer; no dataset cache',
        validation_max_new_tokens=5, lr_schedule='linear_decay_without_warmup')
    return records, conditions, prepared, validated, report


def validate(model, processor, conditions, epoch_out):
    import torch
    metrics = metric_functions()
    saved_python_rng = random.getstate()
    saved_cpu_rng = torch.get_rng_state()
    saved_cuda_rng = torch.cuda.get_rng_state_all()
    model.eval()
    summaries = {}
    try:
        for condition in validation_data.CONDITIONS:
            truth, predicted, strict = [], [], []
            token_loss_sum, token_count = 0.0, 0
            with (epoch_out / f'validation_{condition}.jsonl').open('x') as stream:
                for index, row in enumerate(conditions[condition], 1):
                    batch, audit = common.make_batch(processor, row)
                    batch = batch.to('cuda')
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        output = model(**batch)
                        loss = float(output.loss.item())
                        if not torch.isfinite(output.loss).item():
                            raise RuntimeError('Non-finite validation loss')
                        del output
                        # The supervised target is removed before generation.
                        n = audit['prompt_tokens']
                        generation_inputs = generation_inputs_from_batch(batch, n)
                        generated = model.generate(**generation_inputs, max_new_tokens=5,
                            do_sample=False, num_beams=1, use_cache=True)
                    answer_ids = generated[0, n:]
                    raw = processor.decode(answer_ids, skip_special_tokens=True)
                    prediction = metrics.parse_answer(raw)
                    correct_label = 1 if row['target'] == 'misogynous' else 0
                    truth.append(correct_label)
                    predicted.append(prediction)
                    strict.append(metrics.strict_answer(raw))
                    token_loss_sum += loss * audit['supervised_tokens']
                    token_count += audit['supervised_tokens']
                    stream.write(json.dumps(dict(original_id=row['original_id'],
                        true_label=correct_label, pred_label=prediction, raw_answer=raw,
                        strict_format=strict[-1], generated_tokens=len(answer_ids),
                        supervised_tokens=audit['supervised_tokens'], teacher_forced_loss=loss,
                        input_signature=audit['input_signature'], input_tokens=n,
                        image_grid_thw=audit['image_grid_thw'])) + '\n')
                    stream.flush()
                    del batch, generation_inputs, generated, answer_ids
                    if index % 250 == 0:
                        print(f'Validation {condition}: {index}/{len(conditions[condition])}', flush=True)
            summary = metrics.quality(truth, predicted)
            summary.update(strict_format_compliance=sum(strict) / len(strict),
                           mean_teacher_forced_token_loss=token_loss_sum / token_count)
            summaries[condition] = summary
            print(f"Validation {condition}: macro-F1={summary['macro_f1']:.4f}, invalid={summary['invalid_count']}", flush=True)
    finally:
        model.train()
        model.config.use_cache = False
        random.setstate(saved_python_rng)
        torch.set_rng_state(saved_cpu_rng)
        torch.cuda.set_rng_state_all(saved_cuda_rng)
    common.save_json(epoch_out / 'validation_summary.json', summaries)
    return summaries


def save_checkpoint(model, processor, optimizer, scheduler, scaler, epoch, step, folder, plan):
    import torch
    folder.mkdir()
    # Embedding/head LoRA tensors are included; frozen base matrices are not.
    model.save_pretrained(folder / 'adapter', safe_serialization=True, save_embedding_layers=False)
    processor.save_pretrained(folder / 'processor')
    torch.save(dict(epoch=epoch, global_step=step, optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                    torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all(),
                    python_rng_state=random.getstate()), folder / 'training_state.pt')
    metadata = dict(epoch=epoch, global_step=step,
        base_checkpoint=dict(model=common.MODEL, revision=common.REVISION),
        prepared_complete_sha256=plan['prepared_complete_sha256'],
        initial_adapter_sha256=plan['initial_adapter_sha256'],
        files={str(p.relative_to(folder)): common.digest(p) for p in sorted(folder.rglob('*')) if p.is_file()})
    common.save_json(folder / 'checkpoint_complete.json', metadata)


def run(args):
    out = Path(args.output).resolve()
    if out.exists() and not args.preflight_only:
        raise FileExistsError(f'Use a new full-training output directory: {out}')
    records, conditions, prepared, _, plan = preflight(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.preflight_only:
        return
    root = Path(args.data).resolve()
    model, processor, trainable, lr = common.initialize_model(
        args, root, out, prepared, common.select_smoke_records(records), purpose='full_training')
    import torch
    metrics_source = Path(__file__).resolve().parents[1] / 'efficiency' / 'compare.py'
    plan.update(learning_rate=lr, optimizer='torch.AdamW', weight_decay=0.0,
        adam_betas=[0.9, 0.999], adam_epsilon=1e-8, max_gradient_norm=1.0,
        training_loss_weighting='equal weight per record across each 48-record accumulation group',
        runner_sha256=common.digest(__file__), shared_runner_sha256=common.digest(common.__file__),
        batching_source_sha256=common.digest(batching.__file__),
        validation_preparer_sha256=common.digest(validation_data.__file__),
        scoring_source_sha256=common.digest(metrics_source),
        smoke_completion_sha256={mode: common.digest(Path(folder) / 'complete.json')
            for mode, folder in (('fp16', args.smoke_fp16), ('nf4', args.smoke_nf4))})
    common.save_json(out / 'training_plan.json', plan)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=lr, weight_decay=0.0, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step: lr_multiplier(step, plan['optimizer_updates']))
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
    optimizer.zero_grad(set_to_none=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError('Explicit tokenizer padding token required for batching')

    def prepare_pair(indices):
        entries = [common.make_batch(processor, records[index]) for index in indices]
        batch = batching.combine_batches(entries, pad_id)
        cpu_bytes = sum(t.numel()*t.element_size() for t in batch.values())
        return batch, [audit for _, audit in entries], cpu_bytes

    peaks, reserved_peaks, epoch_summaries = [], [], []
    step = 0
    with (out / 'microbatches.jsonl').open('x') as micro_log, (out / 'training_steps.jsonl').open('x') as step_log:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            model.config.use_cache = False
            start_epoch = time.perf_counter()
            epoch_losses = []
            epoch_ordering = epoch_order(len(records), epoch)
            with closing(batching.prefetched(batching.paired_indices(epoch_ordering), prepare_pair)) as inputs:
                for group in update_groups(epoch_ordering):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start_update = time.perf_counter()
                    losses = []
                    max_cpu_pair_bytes = 0
                    for offset in range(0, len(group), MICROBATCH):
                        indices = group[offset:offset+MICROBATCH]
                        batch, audits, cpu_bytes = next(inputs)
                        max_cpu_pair_bytes = max(max_cpu_pair_bytes, cpu_bytes)
                        if [(a['original_id'], a['is_augmented']) for a in audits] != [
                                (records[i]['original_id'], records[i]['is_augmented']) for i in indices]:
                            raise RuntimeError('Prefetch changed training order')
                        batch = batch.to('cuda')
                        with torch.autocast('cuda', dtype=torch.float16):
                            record_losses = batching.full_record_losses(model, batch)
                            loss = record_losses.mean()
                        if not torch.isfinite(loss).item():
                            raise RuntimeError('Non-finite training loss; stop for review')
                        scaler.scale(loss / ACCUMULATION).backward()
                        values = record_losses.detach().float().cpu().tolist()
                        losses.extend(values)
                        for index, audit, value in zip(indices, audits, values):
                            audit.update(epoch=epoch, optimizer_step=step+1, training_record_index=index,
                                microbatch_in_update=offset//MICROBATCH+1, loss=value)
                            micro_log.write(json.dumps(audit) + '\n')
                        del batch, record_losses, loss
                    gradients = [p.grad for _, p in trainable if p.grad is not None]
                    if len(gradients) != len(trainable):
                        raise RuntimeError('Some adapter parameter tensors received no gradient')
                    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                        raise RuntimeError('Frozen base received gradients')
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
                    used_lr = optimizer.param_groups[0]['lr']
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    if not torch.stack([torch.isfinite(p).all() for _, p in trainable]).all().item():
                        raise RuntimeError('Non-finite adapter parameters after update')
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    step += 1
                    peaks.append(torch.cuda.max_memory_allocated())
                    reserved_peaks.append(torch.cuda.max_memory_reserved())
                    mean_loss = sum(losses) / len(losses)
                    epoch_losses.extend(losses)
                    item = dict(epoch=epoch, optimizer_step=step, records_in_update=len(group),
                        microbatches_in_update=ACCUMULATION, max_preprocessed_pair_tensor_bytes=max_cpu_pair_bytes,
                        mean_record_loss=mean_loss, gradient_norm_before_clipping=float(norm.item()),
                        learning_rate=used_lr, next_learning_rate=optimizer.param_groups[0]['lr'],
                        wall_seconds=time.perf_counter()-start_update,
                        peak_allocated_bytes=peaks[-1], peak_reserved_bytes=reserved_peaks[-1])
                    step_log.write(json.dumps(item) + '\n')
                    step_log.flush()
                    micro_log.flush()
                    if step == 1 or step % 10 == 0:
                        print(f"Epoch {epoch}/2, update {step}/750: loss={mean_loss:.4f}, lr={used_lr:.3g}, "
                              f"step={item['wall_seconds']:.1f}s, allocated={peaks[-1]/2**30:.2f} GiB", flush=True)
                    del gradients
            training_seconds = time.perf_counter() - start_epoch
            epoch_out = out / f'epoch-{epoch}'
            print(f'Saving epoch {epoch} adapter and training state', flush=True)
            save_checkpoint(model, processor, optimizer, scheduler, scaler, epoch, step, epoch_out, plan)
            start_validation = time.perf_counter()
            validation_summary = validate(model, processor, conditions, epoch_out)
            result = dict(epoch=epoch, global_step=step, mean_training_record_loss=sum(epoch_losses)/len(epoch_losses),
                training_wall_seconds=training_seconds, validation_wall_seconds=time.perf_counter()-start_validation,
                validation=validation_summary, checkpoint=f'epoch-{epoch}/adapter')
            epoch_summaries.append(result)
            common.save_json(epoch_out / 'epoch_complete.json', result)
    if step != 750:
        raise RuntimeError('Incorrect final optimizer step count')
    result = dict(mode=args.mode, epochs=2, optimizer_steps=step,
        final_adapter=str(out / 'epoch-2' / 'adapter'), final_checkpoint_policy='epoch_2_without_validation_selection',
        final_adapter_sha256=common.adapter_hash(trainable),
        peak_training_allocated_bytes=max(peaks), peak_training_reserved_bytes=max(reserved_peaks),
        training_wall_seconds=sum(r['training_wall_seconds'] for r in epoch_summaries),
        timing_scope='epoch training loops including preprocessing, transfer, logging and optimizer; excludes loading, checkpoint saves and validation',
        memory_scope='maximum training-update PyTorch allocation; excludes validation and loading peaks; reserved allocator history can include validation',
        epochs_summary=epoch_summaries, validation_complete_sha256=plan['validation_complete_sha256'],
        initial_adapter_sha256=plan['initial_adapter_sha256'], training_plan_sha256=common.digest(out / 'training_plan.json'))
    common.save_json(out / 'complete.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fp16', 'nf4'], required=True)
    parser.add_argument('--data', default='data/processed/mami_lora_qlora_seed42')
    parser.add_argument('--validation', default='data/processed/mami_lora_qlora_validation_seed42')
    parser.add_argument('--adapter-init', default='results/mami_lora_qlora/shared_init_seed42')
    parser.add_argument('--smoke-fp16', default='results/mami_lora_qlora/smoke-fp16-v2')
    parser.add_argument('--smoke-nf4', default='results/mami_lora_qlora/smoke-nf4-v2')
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight-only', action='store_true', help='Check all input hashes and gates on CPU; create no outputs')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    existed = out.exists()
    try:
        run(args)
    except BaseException as error:
        if not existed and (out / 'request.json').exists() and not (out / 'failure.json').exists():
            common.save_json(out / 'failure.json', dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
