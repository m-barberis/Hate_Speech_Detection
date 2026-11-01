"""Matched SmolVLM-256M feasibility checks, early-stopped training and evaluation."""
import argparse
from dataclasses import dataclass, asdict
import json
import platform
from pathlib import Path
import random
import time

import common as c

MAX_EPOCHS = 3
EVAL_STEPS = 75
PATIENCE = 3
MICROBATCH = 2
ACCUMULATION = 24
BATCH = 48
LR = 2e-4


@dataclass
class EarlyStopping:
    best_score: float = -1.0
    best_step: int = 0
    bad_checks: int = 0
    last_step: int = 0

    def update(self, step, score):
        import math
        if step <= self.last_step or step % EVAL_STEPS:
            raise ValueError('Validation steps must increase and be multiples of 75')
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Expected finite validation macro-F1 in [0, 1]')
        self.last_step = step
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_step, self.bad_checks = score, step, 0
        else:
            self.bad_checks += 1
        return improved, self.bad_checks >= PATIENCE


def order(count, epoch):
    if count % BATCH:
        raise ValueError('Training count must form complete 48-record updates')
    indices = list(range(count))
    random.Random(42 + epoch - 1).shuffle(indices)
    return indices


def control(data):
    return dict(model=c.MODEL, revision=c.REVISION, versions=c.versions(), lora=c.LORA,
        data=data, image_policy='native processor defaults, no truncation',
        prompt_policy='system instruction prepended verbatim to user text',
        quantization=dict(type='nf4', compute='float16', double_quant=False,
                          scope='eligible linear modules including vision'),
        seed=42, microbatch=MICROBATCH, accumulation=ACCUMULATION, effective_batch=BATCH,
        learning_rate=LR, max_epochs=MAX_EPOCHS, eval_steps=EVAL_STEPS, patience=PATIENCE,
        monitor='mean macro-F1 clean/overlay with transcription', min_delta=0.0,
        lr_schedule='linear decay over maximum 1125 updates, no warmup')


def check(args):
    import importlib.util
    result = dict(platform=platform.platform(), versions=c.versions(), model=c.MODEL,
                  revision=c.REVISION, cuda_training_ready=False)
    if not importlib.util.find_spec('transformers'):
        result['blocker'] = 'Transformers is missing from this interpreter'
    else:
        import torch
        from transformers import Idefics3ForConditionalGeneration  # noqa: F401
        result.update(model_class_import=True, cuda_available=torch.cuda.is_available(),
                      cuda_build=torch.version.cuda, cuda_arches=torch.cuda.get_arch_list())
        try:
            processor = c.processor_load()
            result.update(processor=processor.__class__.__name__,
                          image_processor=processor.image_processor.to_dict())
            # Only tiny synthetic preprocessing, no model weights or GPU allocation.
            import tempfile
            from PIL import Image
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/'synthetic.png'
                Image.new('RGB', (96, 64), (80, 120, 160)).save(path)
                row = dict(original_id='synthetic', is_augmented=False,
                    image_path=str(path), image_sha256=c.digest(path),
                    system_prompt='Classify this meme.', user_prompt='Return clean or misogynous.',
                    target='clean')
                batch, audit = c.make_batch(processor, row)
                combined = c.combine_batches([(batch, audit), (batch, audit)],
                    processor.tokenizer.pad_token_id)
                rendered = processor.apply_chat_template(c.messages(row), tokenize=False,
                    add_generation_prompt=True)
                if row['system_prompt'] not in rendered:
                    raise RuntimeError('System instruction disappeared from the rendered prompt')
                result.update(processor_smoke=True, audit=audit,
                              batch_shapes={k: list(v.shape) for k, v in combined.items()})
        except Exception as error:
            result['processor_error'] = f'{type(error).__name__}: {error}'
        try:
            c.gpu_free()
            result['gpu_idle'] = True
        except Exception as error:
            result['gpu_idle'] = False
            result['gpu_blocker'] = str(error)
        result['cuda_training_ready'] = bool(result.get('processor_smoke') and
            result.get('cuda_available') and result.get('gpu_idle') and
            'sm_70' in result.get('cuda_arches', []) and all(result['versions'].values()))
    print(json.dumps(result, indent=2))
    if args.output:
        c.save_json(args.output, result)
    # This is a readiness inventory, never a successful GPU smoke marker.


def cache(args):
    from huggingface_hub import snapshot_download
    path = snapshot_download(c.MODEL, revision=c.REVISION,
        allow_patterns=['*.json', '*.txt', '*.jinja', '*.model', '*.safetensors'])
    print(path)


def preflight(args):
    records, _, data = c.load_data(args.data, args.validation)
    processor = c.processor_load()
    selected = c.legacy('training_smoke').select_smoke_records(records)
    audits = []
    for offset in (0, 2):
        entries = [c.make_batch(processor, row) for row in selected[offset:offset+2]]
        c.combine_batches(entries, processor.tokenizer.pad_token_id)
        audits.extend(audit for _, audit in entries)
    result = dict(controls=control(data), selected_input_audits=audits,
        maximum_optimizer_steps=len(records)//BATCH*MAX_EPOCHS)
    print(json.dumps(result, indent=2))
    if args.output:
        c.save_json(args.output, result)


def setup(args, controls):
    import torch
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    c.save_json(out/'request.json', dict(command=args.command, mode=args.mode, controls=controls,
        source_sha256={p.name: c.digest(p) for p in Path(__file__).parent.glob('*.py')
                       if not p.name.startswith('test_')}))
    c.seed()
    processor = c.processor_load()
    model, quantized = c.model_load(args.mode)
    model, trainable, inventory = c.attach_adapter(model)
    inventory.update(quantized_modules=quantized, processor=processor.image_processor.to_dict(),
        gpu=torch.cuda.get_device_name(), gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),
        'uuid', 'unavailable')), versions=c.versions())
    c.save_json(out/'inventory.json', inventory)
    # Loading/quantization may consume different RNG draws; begin dropout identically.
    c.seed()
    return out, processor, model, trainable, inventory


def optimizer_step(model, trainable, optimizer, scaler):
    import torch
    gradients = [p.grad for _, p in trainable if p.grad is not None]
    if len(gradients) != len(trainable):
        missing = [n for n, p in trainable if p.grad is None]
        raise RuntimeError(f'Adapters without gradients: {missing}')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen base received gradients')
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
    if not float(norm) > 0:
        raise RuntimeError('No nonzero adapter gradients')
    scaler.step(optimizer)
    scaler.update()
    if any(not torch.isfinite(p).all().item() for _, p in trainable):
        raise RuntimeError('Non-finite updated adapter tensors')
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def smoke(args):
    import torch
    records, _, data = c.load_data(args.data, args.validation)
    selected = c.legacy('training_smoke').select_smoke_records(records)
    controls = control(data)
    out, processor, model, trainable, inventory = setup(args, controls)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=LR, weight_decay=0.0, foreach=False)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
    batching = c.legacy('training_batches')
    audits, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for offset in (0, 2):
        entries = [c.make_batch(processor, row) for row in selected[offset:offset+2]]
        audits.extend(audit for _, audit in entries)
        batch = c.combine_batches(entries, processor.tokenizer.pad_token_id).to('cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            loss = batching.full_record_losses(model, batch).mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError('Non-finite smoke loss')
        losses.append(float(loss))
        scaler.scale(loss / 2).backward()
        del batch, loss, entries
    norm = optimizer_step(model, trainable, optimizer, scaler)
    if c.adapter_hash(trainable) == inventory['initial_adapter_sha256']:
        raise RuntimeError('The smoke optimizer did not change adapter tensors')
    # Verify target-free inference works after backward, without using validation/test.
    batch, _ = c.prompt_inputs(processor, selected[0])
    batch = batch.to('cuda')
    n = batch['input_ids'].shape[1]
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        answer = model.generate(**batch, max_new_tokens=5, do_sample=False, use_cache=True)
    raw = processor.decode(answer[0, n:], skip_special_tokens=True)
    c.save_json(out/'complete.json', dict(mode=args.mode, controls=controls,
        inventory_sha256=c.digest(out/'inventory.json'), request_sha256=c.digest(out/'request.json'),
        initial_adapter_sha256=inventory['initial_adapter_sha256'],
        input_audits=audits, losses=losses, gradient_norm=norm, optimizer_steps=1,
        generated_answer=raw, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        qualification='Four training records only; does not prove worst-case memory or convergence'))


def verify_gates(args, controls):
    gates = []
    sources = {p.name: c.digest(p) for p in Path(__file__).parent.glob('*.py')
               if not p.name.startswith('test_')}
    for mode, folder in (('fp16', args.smoke_fp16), ('nf4', args.smoke_nf4)):
        root = Path(folder)
        done = c.read_json(root/'complete.json')
        inv = c.read_json(root/'inventory.json')
        request = c.read_json(root/'request.json')
        if (done['mode'] != mode or done['controls'] != controls or done['optimizer_steps'] != 1
                or request['source_sha256'] != sources):
            raise ValueError('Successful matching SmolVLM smoke gates are required for this code/data/environment')
        if (c.digest(root/'inventory.json') != done['inventory_sha256'] or
                c.digest(root/'request.json') != done['request_sha256']):
            raise ValueError('Smoke inventory/request changed')
        if bool(inv['quantized_modules']) != (mode == 'nf4'):
            raise ValueError('Smoke quantization inventory differs')
        gates.append((done, inv))
    for key in ('initial_adapter_sha256', 'input_audits'):
        if gates[0][0][key] != gates[1][0][key]:
            raise ValueError(f'Smoke mismatch: {key}')
    for key in ('trainable_shapes', 'adapter_targets', 'processor', 'gpu_uuid', 'versions'):
        if gates[0][1][key] != gates[1][1][key]:
            raise ValueError(f'Smoke inventory mismatch: {key}')
    return gates[0][0]['initial_adapter_sha256']


def checkpoint(model, processor, folder, step, score, controls):
    folder.mkdir()
    model.save_pretrained(folder/'adapter', safe_serialization=True, save_embedding_layers=False)
    processor.save_pretrained(folder/'processor')
    marker = dict(step=step, score=score, controls=controls,
        adapter_sha256=c.adapter_hash(c.adapter_parameters(model)),
        files={str(p.relative_to(folder)): c.digest(p) for p in folder.rglob('*') if p.is_file()})
    c.save_json(folder/'complete.json', marker)
    return marker


def restore(model, folder):
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    marker = c.read_json(folder/'complete.json')
    for name, sha in marker['files'].items():
        if c.digest(folder/name) != sha:
            raise ValueError(f'Checkpoint file changed: {name}')
    state = load_peft_weights(str(folder/'adapter'), device='cpu')
    result = set_peft_model_state_dict(model, state, adapter_name='default')
    if result.unexpected_keys:
        raise ValueError(f'Unexpected saved adapter keys: {result.unexpected_keys}')
    if c.adapter_hash(c.adapter_parameters(model)) != marker['adapter_sha256']:
        raise ValueError('Restored adapter tensor hash differs')
    return marker


def train(args):
    import torch
    records, conditions, data = c.load_data(args.data, args.validation)
    controls = control(data)
    initial_hash = verify_gates(args, controls)
    out, processor, model, trainable, inventory = setup(args, controls)
    if inventory['initial_adapter_sha256'] != initial_hash:
        raise ValueError('Training adapter initialization differs from matched smoke')
    maximum_steps = len(records)//BATCH * MAX_EPOCHS
    if maximum_steps != 1125:
        raise ValueError('Expected 375 updates per epoch, at most 1125 total')
    c.save_json(out/'training_plan.json', dict(controls=controls, maximum_steps=maximum_steps,
        initial_adapter_sha256=initial_hash, smoke_sha256={mode: c.digest(Path(folder)/'complete.json')
        for mode, folder in (('fp16', args.smoke_fp16), ('nf4', args.smoke_nf4))}))
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=LR, weight_decay=0.0, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1-step/maximum_steps)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
    stopping = EarlyStopping()
    batching = c.legacy('training_batches')
    step, training_seconds, validation_seconds, peak = 0, 0.0, 0.0, 0
    stopped = False
    # Synchronous preprocessing avoids simultaneous access to the processor during validation.
    with (out/'steps.jsonl').open('x') as log, (out/'validation.jsonl').open('x') as validation_log:
        for epoch in range(1, MAX_EPOCHS+1):
            indices = order(len(records), epoch)
            for start in range(0, len(indices), BATCH):
                c.gpu_free()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                losses = []
                for offset in range(0, BATCH, MICROBATCH):
                    chosen = indices[start+offset:start+offset+MICROBATCH]
                    entries = [c.make_batch(processor, records[i]) for i in chosen]
                    batch = c.combine_batches(entries, processor.tokenizer.pad_token_id).to('cuda')
                    with torch.autocast('cuda', dtype=torch.float16):
                        record_losses = batching.full_record_losses(model, batch)
                        loss = record_losses.mean()
                    if not torch.isfinite(loss).item():
                        raise RuntimeError('Non-finite training loss')
                    losses.extend(record_losses.detach().float().cpu().tolist())
                    scaler.scale(loss/ACCUMULATION).backward()
                    del batch, entries, record_losses, loss
                norm = optimizer_step(model, trainable, optimizer, scaler)
                used_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                step += 1
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                training_seconds += seconds
                peak = max(peak, torch.cuda.max_memory_allocated())
                log.write(json.dumps(dict(step=step, epoch=epoch, mean_loss=sum(losses)/len(losses),
                    gradient_norm=norm, learning_rate=used_lr, wall_seconds=seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())) + '\n')
                log.flush()
                if step == 1 or step % 10 == 0:
                    print(f'Epoch {epoch}/{MAX_EPOCHS}, step {step}/{maximum_steps}, loss={sum(losses)/len(losses):.4f}', flush=True)
                if step % EVAL_STEPS == 0:
                    started = time.perf_counter()
                    summary = c.evaluate(model, processor, conditions, out/f'validation-step-{step}')
                    validation_seconds += time.perf_counter()-started
                    score = c.monitor_score(summary)
                    improved, stopped = stopping.update(step, score)
                    if improved:
                        checkpoint(model, processor, out/f'checkpoint-step-{step}', step, score, controls)
                    validation_log.write(json.dumps(dict(**asdict(stopping), epoch=epoch,
                        improved=improved, stop=stopped, validation=summary)) + '\n')
                    validation_log.flush()
                    print(f'Validation step {step}: score={score:.4f}, best={stopping.best_score:.4f}, patience={stopping.bad_checks}/{PATIENCE}', flush=True)
                    if stopped:
                        break
            if stopped:
                break
    best_folder = out/f'checkpoint-step-{stopping.best_step}'
    best = restore(model, best_folder)
    c.save_json(out/'complete.json', dict(mode=args.mode, controls=controls,
        optimizer_steps=step, epochs_completed=step/(len(records)//BATCH), early_stopped=stopped,
        best_step=stopping.best_step, best_score=stopping.best_score,
        best_checkpoint=best_folder.name, best_adapter_sha256=best['adapter_sha256'],
        initial_adapter_sha256=initial_hash, checkpoint_policy='best_validation_mean_macro_f1',
        best_checkpoint_complete_sha256=c.digest(best_folder/'complete.json'),
        request_sha256=c.digest(out/'request.json'), training_plan_sha256=c.digest(out/'training_plan.json'),
        training_wall_seconds=training_seconds, validation_wall_seconds=validation_seconds,
        peak_training_allocated_bytes=peak, restored_best=True))


def training_gate(root, mode):
    root = Path(root).resolve()
    done = c.read_json(root/'complete.json')
    marker_path = root/done['best_checkpoint']/'complete.json'
    marker = c.read_json(marker_path)
    if (done['mode'] != mode or done['controls']['model'] != c.MODEL or
            done['controls']['revision'] != c.REVISION or done['controls']['versions'] != c.versions()
            or done['checkpoint_policy'] != 'best_validation_mean_macro_f1' or not done['restored_best']
            or marker['step'] != done['best_step'] or marker['score'] != done['best_score']
            or marker['adapter_sha256'] != done['best_adapter_sha256']
            or c.digest(marker_path) != done['best_checkpoint_complete_sha256']):
        raise ValueError('Expected completed best-validation SmolVLM training in the same environment')
    for name in ('request', 'training_plan'):
        if c.digest(root/f'{name}.json') != done[f'{name}_sha256']:
            raise ValueError(f'Training {name} changed')
    for name, sha in marker['files'].items():
        if c.digest(marker_path.parent/name) != sha:
            raise ValueError(f'Checkpoint changed: {name}')
    return root, done


def evaluate_test(args):
    from peft import PeftModel
    mode = args.arm.split('-')[0]
    gates = {m: training_gate(folder, m) for m, folder in
        (('fp16', args.fp16_training), ('nf4', args.nf4_training))}
    if gates['fp16'][1]['controls'] != gates['nf4'][1]['controls'] or gates['fp16'][1][
            'initial_adapter_sha256'] != gates['nf4'][1]['initial_adapter_sha256']:
        raise ValueError('Training arms have different controls or initialization')
    test = c.legacy('prepare_test_conditions')
    conditions, _ = test.load(args.test_data)
    test_manifest = c.read_json(Path(args.test_data)/'manifest.json')
    if test_manifest['training_manifest_sha256'] != c.digest(Path(args.data)/'manifest.json'):
        raise ValueError('Test data belongs to a different training split')
    if c.digest(Path(args.data)/'complete.json') != gates[mode][1]['controls']['data']['prepared_complete_sha256']:
        raise ValueError('Test/training preparation link differs')
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    c.seed()
    processor = c.processor_load()
    model, quantized = c.model_load(mode)
    root, trained = gates[mode]
    if not args.arm.endswith('-base'):
        model = PeftModel.from_pretrained(model, root/trained['best_checkpoint']/'adapter',
                                         is_trainable=False, autocast_adapter_dtype=True)
        if c.adapter_hash(c.adapter_parameters(model)) != trained['best_adapter_sha256']:
            raise ValueError('Reloaded best adapter differs')
    model.requires_grad_(False)
    model.eval()
    summary = c.evaluate(model, processor, conditions, out)
    c.save_json(out/'complete.json', dict(arm=args.arm, model=c.MODEL, revision=c.REVISION,
        versions=c.versions(), quantized_modules=quantized, summary=summary,
        best_step=trained['best_step'] if not args.arm.endswith('-base') else None,
        test_preparation_sha256=c.digest(Path(args.test_data)/'complete.json'),
        training_completion_sha256={m: c.digest(r/'complete.json') for m, (r, _) in gates.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'cache', 'preflight', 'smoke', 'train', 'evaluate'])
    parser.add_argument('--mode', choices=['fp16', 'nf4'])
    parser.add_argument('--output')
    parser.add_argument('--data', default='data/processed/mami_lora_qlora_seed42')
    parser.add_argument('--validation', default='data/processed/mami_lora_qlora_validation_seed42')
    parser.add_argument('--smoke-fp16', default='results/smolvlm/smoke-fp16')
    parser.add_argument('--smoke-nf4', default='results/smolvlm/smoke-nf4')
    parser.add_argument('--fp16-training', default='results/smolvlm/fp16-training')
    parser.add_argument('--nf4-training', default='results/smolvlm/nf4-training')
    parser.add_argument('--test-data', default='data/processed/mami_lora_qlora_test_seed42')
    parser.add_argument('--arm', choices=['fp16-base', 'nf4-base', 'fp16-lora', 'nf4-qlora'])
    args = parser.parse_args()
    if args.command in ('smoke', 'train', 'evaluate') and not args.output:
        parser.error('--output is required')
    if args.command in ('smoke', 'train') and not args.mode:
        parser.error('--mode is required')
    if args.command == 'evaluate' and not args.arm:
        parser.error('--arm is required')
    out = Path(args.output).resolve() if args.output else None
    if out and out.exists():
        raise FileExistsError(f'Preserve existing output: {out}')
    try:
        dict(check=check, cache=cache, preflight=preflight, smoke=smoke,
             train=train, evaluate=evaluate_test)[args.command](args)
    except BaseException as error:
        if out and out.is_dir() and (out/'request.json').exists():
            c.save_json(out/'failure.json', dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
