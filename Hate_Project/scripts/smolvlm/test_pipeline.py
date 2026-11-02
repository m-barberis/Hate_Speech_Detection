"""Control-flow tests plus optional real PyTorch/Idefics3 CPU integration checks."""
from pathlib import Path
import tempfile
import unittest
import importlib.util

import common as c
from run import EarlyStopping, order, execution_settings


class ControlTests(unittest.TestCase):
    def test_recovery_launcher_never_retrains_fp16_or_overwrites_existing_outputs(self):
        import json
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root/'original'
            original.mkdir()
            sentinel = original/'finished-fp16.txt'
            sentinel.write_text('keep this result')
            binary = root/'bin'
            binary.mkdir()
            calls = root/'calls.jsonl'
            fake = binary/'python'
            fake.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                'with open(os.environ["SMOLVLM_TEST_CALLS"], "a") as f:\n'
                '    f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'],
                       SMOLVLM_TEST_CALLS=str(calls))
            script = str(Path(__file__).with_name('recover_nf4.sh'))
            retry = root/'recovery'
            result = subprocess.run(['bash', script, str(original), str(retry)], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = [json.loads(line) for line in calls.read_text().splitlines()]
            training = [args for args in recorded if len(args) > 1 and args[1] == 'train']
            self.assertEqual(len(training), 1)
            self.assertEqual(training[0][training[0].index('--mode')+1], 'nf4')
            self.assertEqual(training[0][training[0].index('--microbatch')+1], '1')
            evaluations = [args for args in recorded if len(args) > 1 and args[1] == 'evaluate']
            self.assertEqual(len(evaluations), 4)
            self.assertTrue(all('--allow-execution-differences' in args for args in evaluations))
            self.assertEqual(sentinel.read_text(), 'keep this result')
            previous_calls = calls.read_bytes()
            result = subprocess.run(['bash', script, str(original), str(retry)], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(calls.read_bytes(), previous_calls)

    def test_validation_windows_and_effective_batch_are_preserved(self):
        from types import SimpleNamespace
        from training_engine import validation_windows
        indices = order(18000, 1)
        windows = validation_windows(indices)
        self.assertEqual([len(w) for w in windows], [3600]*5)
        self.assertEqual([i for window in windows for i in window], indices)
        for microbatch in (1, 2, 4, 8):
            settings = execution_settings(SimpleNamespace(microbatch=microbatch))
            self.assertEqual(settings['microbatch']*settings['accumulation'], 48)
            self.assertFalse(settings['gradient_checkpointing'])
        with self.assertRaises(ValueError):
            execution_settings(SimpleNamespace(microbatch=5))

    def test_recovery_allows_only_explicit_execution_differences(self):
        from copy import deepcopy
        from run import control, compare_training_controls
        fp16 = control({'manifest': 'same'})
        nf4 = deepcopy(fp16)
        nf4.update(microbatch=1, accumulation=48)
        with self.assertRaisesRegex(ValueError, 'allow-execution-differences'):
            compare_training_controls(fp16, nf4)
        comparison = compare_training_controls(fp16, nf4, True)
        self.assertFalse(comparison['strict_execution_match'])
        self.assertEqual(comparison['execution_differences']['microbatch'], dict(fp16=2, nf4=1))
        self.assertTrue(compare_training_controls(fp16, deepcopy(fp16))['strict_execution_match'])
        for key, value in [('learning_rate', 1e-3), ('seed', 123), ('eval_steps', 150),
                           ('revision', 'different'), ('lora', {}), ('data', {})]:
            changed = dict(nf4, **{key: value})
            with self.assertRaisesRegex(ValueError, 'experiment controls differ'):
                compare_training_controls(fp16, changed, True)
        with self.assertRaisesRegex(ValueError, 'effective batch'):
            compare_training_controls(fp16, dict(nf4, accumulation=24), True)

    def test_recovery_verifies_existing_fp16_without_modifying_it(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'fp16-training'
            folder = root/'checkpoint-step-525'
            adapter = folder/'adapter'
            adapter.mkdir(parents=True)
            (adapter/'adapter_model.safetensors').write_bytes(b'file-verification-fixture')
            controls = run.control({'manifest': 'same'})
            c.save_json(folder/'complete.json', dict(step=525, score=.8412741387829133,
                controls=controls, adapter_sha256='best', files={
                    'adapter/adapter_model.safetensors': c.digest(adapter/'adapter_model.safetensors')}))
            for name in ('request', 'training_plan'):
                c.save_json(root/f'{name}.json', dict(controls=controls))
            done = dict(mode='fp16', controls=controls, best_checkpoint=folder.name,
                optimizer_steps=750, best_step=525, best_score=.8412741387829133,
                best_adapter_sha256='best', initial_adapter_sha256='init',
                checkpoint_policy='best_validation_mean_macro_f1', restored_best=True,
                best_checkpoint_complete_sha256=c.digest(folder/'complete.json'),
                request_sha256=c.digest(root/'request.json'),
                training_plan_sha256=c.digest(root/'training_plan.json'))
            c.save_json(root/'complete.json', done)
            original = {str(p.relative_to(root)): c.digest(p) for p in root.rglob('*') if p.is_file()}
            smoke = Path(temp)/'smoke-nf4'
            smoke.mkdir()
            c.save_json(smoke/'complete.json', dict(controls=controls))
            args = SimpleNamespace(fp16_training=str(root), smoke_nf4=str(smoke),
                smoke_fp16='unused', microbatch=1, allow_execution_differences=True,
                output=str(Path(temp)/'recovery-check.json'))
            with patch.object(run, 'verify_gates', return_value='init'), patch.object(c, 'model_load',
                    side_effect=AssertionError('Recovery verification must not load a GPU')):
                run.verify_recovery(args)
            disclosure = c.read_json(args.output)['training_comparison']
            self.assertEqual(disclosure['execution_settings']['nf4']['accumulation'], 48)
            self.assertEqual(original, {str(p.relative_to(root)): c.digest(p)
                                       for p in root.rglob('*') if p.is_file()})
            args.output = None
            with patch.object(run, 'verify_gates', return_value='different'):
                with self.assertRaisesRegex(ValueError, 'initialization differs'):
                    run.verify_recovery(args)
            with (adapter/'adapter_model.safetensors').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Checkpoint changed'):
                run.training_gate(root, 'fp16')

    def test_summary_rejects_unpaired_inputs_and_changed_metrics(self):
        import json
        from summarize import ARMS, summarize
        scoring = c.metrics()
        disclosure = dict(strict_execution_match=False,
            execution_differences={'microbatch': dict(fp16=2, nf4=1)})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(original_id=str(i), true_label=i % 2, pred_label=i % 2,
                input_signature=str(i), raw_answer='misogynous' if i % 2 else 'clean')
                for i in range(1000)]
            quality = scoring.quality([r['true_label'] for r in rows], [r['pred_label'] for r in rows])
            for arm in ARMS:
                folder = root/arm
                folder.mkdir()
                for condition in c.CONDITIONS:
                    (folder/f'{condition}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                c.save_json(folder/'complete.json', dict(arm=arm, model=c.MODEL, revision=c.REVISION,
                    versions={}, test_preparation_sha256='same', training_completion_sha256={},
                    training_comparison=disclosure,
                    best_step=75 if not arm.endswith('base') else None,
                    summary={k: quality for k in c.CONDITIONS}))
            self.assertEqual(summarize(root)['conditions']['clean_image_only']['macro_f1_deltas']['qlora_minus_lora'], 0)
            self.assertEqual(summarize(root)['training_comparison'], disclosure)
            complete_path = root/ARMS[-1]/'complete.json'
            completion = c.read_json(complete_path)
            completion['training_comparison'] = None
            complete_path.write_text(json.dumps(completion))
            with self.assertRaisesRegex(ValueError, 'disclosure differs'):
                summarize(root)
            completion['training_comparison'] = disclosure
            complete_path.write_text(json.dumps(completion))
            path = root/ARMS[-1]/'clean_image_only.jsonl'
            changed = [dict(r) for r in rows]
            changed[0]['input_signature'] = 'different'
            path.write_text(''.join(json.dumps(r)+'\n' for r in changed))
            with self.assertRaisesRegex(ValueError, 'inputs differ'):
                summarize(root)
            changed[0] = dict(rows[0], pred_label=-1, raw_answer='invalid')
            path.write_text(''.join(json.dumps(r)+'\n' for r in changed))
            with self.assertRaisesRegex(ValueError, 'metrics differ'):
                summarize(root)

    def test_patience_counts_checks_and_restores_best_step(self):
        stopping = EarlyStopping()
        self.assertEqual(stopping.update(75, .5), (True, False))
        self.assertEqual(stopping.update(150, .49), (False, False))
        self.assertEqual(stopping.update(225, .5), (False, False))
        self.assertEqual(stopping.update(300, .48), (False, True))
        self.assertEqual(stopping.best_step, 75)

    def test_improvement_resets_patience(self):
        stopping = EarlyStopping()
        for step, score in [(75, .5), (150, .4), (225, .6)]:
            stopping.update(step, score)
        self.assertEqual(stopping.bad_checks, 0)
        self.assertEqual(stopping.best_step, 225)
        for step, score in [(225, .7), (250, .7), (300, float('nan'))]:
            with self.assertRaises(ValueError):
                stopping.update(step, score)

    def test_order_is_complete_repeatable_and_changes_between_epochs(self):
        self.assertEqual(order(18000, 1), order(18000, 1))
        self.assertEqual(set(order(18000, 1)), set(range(18000)))
        self.assertNotEqual(order(18000, 1), order(18000, 2))
        self.assertEqual(18000//48, 375)
        self.assertEqual([s for s in range(1, 1126) if s % 75 == 0], list(range(75, 1126, 75)))

    def test_monitor_uses_only_two_primary_conditions(self):
        summary = {k: {'macro_f1': v} for k, v in zip(c.CONDITIONS, [0, 0, .6, .8])}
        self.assertAlmostEqual(c.monitor_score(summary), .7)

    def test_instruction_preserved_without_target_in_prompt(self):
        row = dict(system_prompt='SYSTEM instruction', user_prompt='USER instruction', target='secret')
        message = c.messages(row)[0]
        self.assertEqual(message['role'], 'user')
        self.assertEqual(message['content'][1]['text'], 'SYSTEM instruction\n\nUSER instruction')
        self.assertNotIn('secret', message['content'][1]['text'])


HAS_TORCH_STACK = all(importlib.util.find_spec(p) for p in ('torch', 'transformers', 'peft', 'numpy'))


@unittest.skipUnless(HAS_TORCH_STACK, 'Requires the server PyTorch/Transformers/PEFT stack')
class ArchitectureTests(unittest.TestCase):
    def test_training_closes_worker_before_validation_and_stops_at_300(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        import run
        import training_engine as engine
        worker_active = [False]
        windows = []
        model = torch.nn.Linear(1, 1)
        records = [dict(original_id=str(i), is_augmented=False) for i in range(18000)]
        conditions = {name: [] for name in c.CONDITIONS}

        def updates(*args):
            indices, first_step = args[3], args[9]
            windows.append(list(indices))
            worker_active[0] = True
            try:
                for i in range(len(indices)//48):
                    yield dict(step=first_step+i+1, mean_loss=.5, wall_seconds=1.,
                               cpu_wait_seconds=0., peak_allocated_bytes=1)
            finally:
                worker_active[0] = False

        def evaluate(*args):
            self.assertFalse(worker_active[0])
            return {name: dict(macro_f1=.6) for name in c.CONDITIONS}

        def checkpoint(model, processor, folder, step, score, controls):
            folder.mkdir()
            c.save_json(folder/'complete.json', dict(step=step))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c.save_json(root/'request.json', {})
            for mode in ('fp16', 'nf4'):
                (root/mode).mkdir()
                c.save_json(root/mode/'complete.json', {})
            args = SimpleNamespace(data='unused', validation='unused', mode='fp16',
                smoke_fp16=str(root/'fp16'), smoke_nf4=str(root/'nf4'))
            with patch.object(c, 'load_data', return_value=(records, conditions, {})), patch.object(
                    run, 'verify_gates', return_value='init'), patch.object(run, 'setup',
                    return_value=(root, None, model, list(model.named_parameters()),
                                  dict(initial_adapter_sha256='init'))), patch.object(
                    engine, 'updates', side_effect=updates), patch.object(c, 'evaluate', side_effect=evaluate), patch.object(
                    run, 'checkpoint', side_effect=checkpoint), patch.object(run, 'restore',
                    return_value=dict(adapter_sha256='best')) as restore, patch.object(torch.amp, 'GradScaler'):
                run.train(args)
            done = c.read_json(root/'complete.json')
            self.assertEqual(done['optimizer_steps'], 300)
            self.assertEqual(done['best_step'], 75)
            self.assertTrue(done['early_stopped'])
            restore.assert_called_once_with(model, root/'checkpoint-step-75')
            self.assertEqual([i for w in windows for i in w], run.order(18000, 1)[:14400])

    def test_preprocessing_once_is_exactly_equal_to_original_two_calls(self):
        from unittest.mock import patch
        from PIL import Image
        import torch
        torch.set_num_threads(2)
        try:
            processor = c.processor_load()
        except OSError:
            self.skipTest('Pinned tokenizer/processor not cached')
        original = processor.image_processor
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'image.png'
            # Non-uniform pixels exercise resizing, splitting and normalization.
            image = Image.new('RGB', (96, 64), (60, 100, 150))
            image.paste((220, 30, 50), (20, 10, 70, 40))
            image.save(path)
            for target in ('clean', 'misogynous'):
                row = dict(original_id='example', is_augmented=False,
                    image_path=str(path), image_sha256=c.digest(path),
                    system_prompt='Classify this meme.', user_prompt='Return the label.', target=target)
                with patch.object(original, 'preprocess', wraps=original.preprocess) as calls:
                    optimized, optimized_audit = c.make_batch(processor, row)
                    self.assertEqual(calls.call_count, 1)
                with patch.object(c, 'record_processor', side_effect=lambda p: p), patch.object(
                        original, 'preprocess', wraps=original.preprocess) as calls:
                    reference, reference_audit = c.make_batch(processor, row)
                    self.assertEqual(calls.call_count, 2)
                self.assertIs(processor.image_processor, original)
                self.assertEqual(optimized_audit, reference_audit)
                self.assertEqual(set(optimized), set(reference))
                for key in optimized:
                    self.assertTrue(torch.equal(optimized[key], reference[key]), key)

    def test_prefetch_updates_preserve_record_weighting_and_close_producer(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch
        import threading
        import torch
        import training_engine as engine
        batching = c.legacy('training_batches')
        torch.set_num_threads(2)

        class Batch(dict):
            def to(self, device):
                return self

        class Event:
            def __init__(self, **kwargs):
                pass
            def record(self):
                pass
            def elapsed_time(self, other):
                return 0.0

        class Scaler:
            def scale(self, loss):
                return loss

        records = [dict(original_id=str(i), is_augmented=False, value=i/48) for i in range(48)]
        def prepare(processor, row):
            return {'x': torch.tensor([[row['value']]])}, row
        def combine(entries, pad_id):
            return Batch(x=torch.cat([b['x'] for b, _ in entries]))
        def record_losses(model, batch):
            return ((model(batch['x'])-.5*batch['x'])**2).flatten()
        def update(model, trainable, optimizer, scaler):
            norm = float(trainable[0][1].grad.norm())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return norm

        weights = []
        for microbatch in (1, 2, 4, 8):
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(.3)
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            with patch.object(c, 'make_batch', side_effect=prepare), patch.object(
                    c, 'combine_batches', side_effect=combine), patch.object(c, 'gpu_free'), patch.object(
                    c, 'legacy', return_value=SimpleNamespace(prefetched=batching.prefetched,
                    full_record_losses=record_losses)), patch.object(torch, 'autocast',
                    side_effect=lambda *a, **k: nullcontext()), patch.object(torch.cuda, 'synchronize'), patch.object(
                    torch.cuda, 'reset_peak_memory_stats'), patch.object(torch.cuda, 'Event', Event), patch.object(
                    torch.cuda, 'max_memory_allocated', return_value=0):
                rows = list(engine.updates(model, SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
                    records, list(range(48)), list(model.named_parameters()), optimizer, Scaler(),
                    scheduler, microbatch, 74, update))
            self.assertEqual(rows[0]['step'], 75)
            self.assertEqual(rows[0]['records_in_update'], 48)
            self.assertFalse(any(t.name.startswith('mami-input') for t in threading.enumerate()))
            weights.append(float(model.weight.detach()))
        for weight in weights[1:]:
            self.assertAlmostEqual(weight, weights[0], places=6)

    def test_cached_real_processor_supervision_and_crop_padding(self):
        from PIL import Image
        try:
            processor = c.processor_load()
        except OSError:
            self.skipTest('Pinned tokenizer/processor not cached; test never downloads')
        with tempfile.TemporaryDirectory() as temp:
            entries = []
            for index, size in enumerate(((96, 64), (64, 96))):
                path = Path(temp)/f'{index}.png'
                Image.new('RGB', size, (80, 120, 160)).save(path)
                row = dict(original_id=str(index), is_augmented=False,
                    image_path=str(path), image_sha256=c.digest(path),
                    system_prompt='Classify this meme.', user_prompt='Return clean or misogynous.',
                    target='clean' if index == 0 else 'misogynous')
                batch, audit = c.make_batch(processor, row)
                self.assertTrue(audit['decoded_supervision'].lstrip().startswith(row['target']))
                self.assertTrue((batch['labels'][:, :audit['prompt_tokens']] == -100).all())
                entries.append((batch, audit))
            combined = c.combine_batches(entries, processor.tokenizer.pad_token_id)
            self.assertEqual(combined['input_ids'].shape[0], 2)

    def test_variable_crop_batch_backward_and_checkpoint_roundtrip(self):
        import torch
        from transformers import Idefics3Config, Idefics3ForConditionalGeneration
        import run
        torch.set_num_threads(2)
        config = Idefics3Config(image_token_id=90, scale_factor=2, pad_token_id=0,
            text_config=dict(model_type='llama', vocab_size=100, hidden_size=32,
                intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                num_key_value_heads=2, pad_token_id=0, bos_token_id=1, eos_token_id=2),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=4, image_size=32, patch_size=16, num_channels=3))
        config._attn_implementation = 'sdpa'
        model, parameters, inventory = c.attach_adapter(Idefics3ForConditionalGeneration(config))
        initial = c.adapter_hash(parameters)

        def entry(crops):
            ids = torch.tensor([[1] + [90]*crops + [3, 4]])
            labels = ids.clone()
            labels[:, :-1] = -100
            return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                pixel_values=torch.randn(1, crops, 3, 32, 32),
                pixel_attention_mask=torch.ones(1, crops, 32, 32, dtype=torch.long)), {}

        batch = c.combine_batches([entry(1), entry(2)], 0)
        self.assertEqual(tuple(batch['pixel_values'].shape), (2, 2, 3, 32, 32))
        self.assertTrue((batch['pixel_attention_mask'][0, 1] == 0).all())
        self.assertEqual(batch['labels'][0, -1].item(), -100)
        losses = c.legacy('training_batches').full_record_losses(model, batch)
        self.assertEqual(tuple(losses.shape), (2,))
        self.assertTrue(torch.isfinite(losses).all())
        losses.mean().backward()
        self.assertTrue(all(p.grad is not None for _, p in parameters))
        self.assertFalse(any(p.grad is not None for p in model.parameters() if not p.requires_grad))
        optimizer = torch.optim.AdamW([p for _, p in parameters], lr=2e-4)
        optimizer.step()
        self.assertNotEqual(initial, c.adapter_hash(parameters))

        class DummyProcessor:
            def save_pretrained(self, folder):
                folder.mkdir()
                (folder/'stub.json').write_text('{}')

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'checkpoint'
            run.checkpoint(model, DummyProcessor(), folder, 75, .5, {})
            saved_hash = c.adapter_hash(parameters)
            with torch.no_grad():
                parameters[0][1].add_(1)
            self.assertNotEqual(saved_hash, c.adapter_hash(parameters))
            run.restore(model, folder)
            self.assertEqual(saved_hash, c.adapter_hash(parameters))
            with (folder/'adapter'/'adapter_config.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'file changed'):
                run.restore(model, folder)


if __name__ == '__main__':
    unittest.main()
