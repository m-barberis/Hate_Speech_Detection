"""CPU-only checks of training boundaries, selection, and data integrity."""
import json
from pathlib import Path
import tempfile
import unittest

import training_smoke as smoke


class TrainingGateTests(unittest.TestCase):
    def test_prompt_mask_uses_full_expanded_prefix(self):
        prompt = [10, 11, 999, 999, 999, 12]
        self.assertEqual(smoke.prompt_boundary(prompt, prompt + [21, 22]), 6)
        for wrong in ([], [10, 11, 999, 12], prompt + [21, 22]):
            with self.assertRaises(ValueError):
                smoke.prompt_boundary(wrong, prompt + [21, 22])

    def test_recipe_reuses_actual_legacy_values(self):
        recipe, learning_rate = smoke.original_recipe()
        self.assertEqual(recipe['r'], 16)
        self.assertEqual(recipe['lora_alpha'], 32)
        self.assertEqual(recipe['lora_dropout'], .05)
        self.assertIn('embed_tokens', recipe['target_modules'])
        self.assertIn('lm_head', recipe['target_modules'])
        self.assertEqual(learning_rate, 2e-4)

    def test_training_only_pairs_and_integrity_guards(self):
        originals = [dict(file_name=f'{i}.jpg', label=i % 2,
                          split='train' if i < 9000 else 'validation') for i in range(10000)]
        rows = [dict(original_id=f'TRAINING/{i}.jpg', split='train',
                     target='misogynous' if i % 2 else 'clean',
                     is_augmented=aug, user_prompt='Task: label' if aug else 'Meme Text: transcript')
                for i in range(9000) for aug in (False, True)]
        selected = smoke.select_smoke_records(rows)
        self.assertEqual(selected, smoke.select_smoke_records(list(reversed(rows))))
        self.assertEqual(len(selected), 4)
        self.assertEqual({r['target'] for r in selected}, {'clean', 'misogynous'})
        self.assertEqual(sum(r['is_augmented'] for r in selected), 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps(dict(seed=42, validation_fraction=.1,
                                                              originals=originals)))
            path = root / 'train.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            complete = dict(manifest_sha256=smoke.digest(root / 'manifest.json'),
                            jsonl_sha256={'train': smoke.digest(path)}, counts={'train': 18000})
            marker = root / 'complete.json'
            marker.write_text(json.dumps(complete))
            loaded, _ = smoke.load_training_records(root)
            self.assertEqual(len(loaded), 18000)
            with path.open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'records changed'):
                smoke.load_training_records(root)
            rows[0]['original_id'] = 'TRAINING/9000.jpg'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            complete['jsonl_sha256']['train'] = smoke.digest(path)
            marker.write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, 'Validation original'):
                smoke.load_training_records(root)


if __name__ == '__main__':
    unittest.main()
