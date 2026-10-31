"""CPU-only checks for the MAMI LoRA/QLoRA split and frozen augmentation records."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import prepare_comparison_data as prep


class PreparationTests(unittest.TestCase):
    def test_split_is_stratified_disjoint_and_order_independent(self):
        rows = [dict(file_name=f'{label}_{i}.jpg', label=label)
                for label in (0, 1) for i in range(5000)]
        split = prep.stratified_split(rows, 42)
        self.assertEqual(split, prep.stratified_split(list(reversed(rows)), 42))
        for label in (0, 1):
            self.assertEqual(sum(r['label'] == label and split[r['file_name']] == 'train'
                                 for r in rows), 4500)
            self.assertEqual(sum(r['label'] == label and split[r['file_name']] == 'validation'
                                 for r in rows), 500)

    def test_preparation_retains_pair_identity_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, assets = root / 'TRAINING', root / 'assets'
            train.mkdir()
            assets.mkdir()
            lines = ['file_name\tmisogynous\tText Transcription\n']
            for label in (0, 1):
                for i in range(10):
                    name = f'{label}_{i}.jpg'
                    Image.new('RGB', (16, 16), (label*100, i*20, 80)).save(train / name)
                    lines.append(f'{name}\t{label}\tOriginal transcription {i}\n')
            (train / 'training.csv').write_text(''.join(lines))
            for code in prep.original.NEW_TRAIN_EMOJIS:
                Image.new('RGBA', (512, 512), (240, 200, 0, 220)).save(assets / f'emoji_u{code}.png')
            real_read = prep.read_training_data
            with self.assertRaisesRegex(ValueError, '5000'):
                real_read(train)
            results = []
            for destination in ('a', 'b'):
                out = root / destination
                args = argparse.Namespace(train_dir=str(train), emoji_dir=str(assets),
                                          output=str(out), seed=42)
                with patch.object(prep, 'read_training_data', side_effect=lambda p: real_read(p, 10)):
                    with contextlib.redirect_stdout(io.StringIO()):
                        prep.prepare(args)
                completion = json.loads((out / 'complete.json').read_text())
                self.assertEqual(completion['counts'],
                    {'train': 36, 'validation_clean': 2, 'validation_overlay': 2})
                tables = {key: [json.loads(line) for line in (out / f'{key}.jsonl').read_text().splitlines()]
                          for key in completion['counts']}
                train_ids = {r['original_id'] for r in tables['train']}
                clean_ids = {r['original_id'] for r in tables['validation_clean']}
                overlay_ids = {r['original_id'] for r in tables['validation_overlay']}
                self.assertFalse(train_ids & clean_ids)
                self.assertEqual(clean_ids, overlay_ids)
                self.assertEqual(len(train_ids), 18)
                for key in ('validation_clean', 'validation_overlay'):
                    self.assertTrue(all(r['user_prompt'].startswith('Task: ') and
                                        'Meme Text:' not in r['user_prompt'] for r in tables[key]))
                for key, table in tables.items():
                    for r in table:
                        self.assertEqual(prep.digest(r['image_path']), r['image_sha256'])
                        # Compare two output locations without their absolute path.
                        del r['image_path']
                results.append(tables)
                with self.assertRaises(FileExistsError):
                    prep.prepare(args)
            self.assertEqual(results[0], results[1])
            (assets / 'emoji_u1f47d.png').unlink()
            with self.assertRaises(FileNotFoundError):
                prep.load_assets(assets)


if __name__ == '__main__':
    unittest.main()
