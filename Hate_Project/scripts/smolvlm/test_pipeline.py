"""Control-flow tests plus optional real PyTorch/Idefics3 CPU integration checks."""
from pathlib import Path
import tempfile
import unittest
import importlib.util

import common as c
from run import EarlyStopping, order


class ControlTests(unittest.TestCase):
    def test_summary_rejects_unpaired_inputs_and_changed_metrics(self):
        import json
        from summarize import ARMS, summarize
        scoring = c.metrics()
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
                    best_step=75 if not arm.endswith('base') else None,
                    summary={k: quality for k in c.CONDITIONS}))
            self.assertEqual(summarize(root)['conditions']['clean_image_only']['macro_f1_deltas']['qlora_minus_lora'], 0)
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
