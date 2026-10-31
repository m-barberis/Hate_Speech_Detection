"""CPU-only tests for the full-training schedule and four validation conditions."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import prepare_comparison_data as prep
import prepare_validation_conditions as validation
import train_comparison as train


class FullTrainingTests(unittest.TestCase):
    def test_generation_cannot_see_teacher_forced_target(self):
        batch = dict(input_ids=np.array([[1, 2, 3, 91, 92]]),
                     attention_mask=np.ones((1, 5)), labels=np.array([[-100, -100, -100, 91, 92]]),
                     mm_token_type_ids=np.zeros((1, 5)), pixel_values=np.ones((4, 8)))
        generated = train.generation_inputs_from_batch(batch, 3)
        self.assertNotIn('labels', generated)
        np.testing.assert_array_equal(generated['input_ids'], [[1, 2, 3]])
        self.assertEqual(generated['attention_mask'].shape, (1, 3))
        self.assertEqual(generated['mm_token_type_ids'].shape, (1, 3))
        self.assertIs(generated['pixel_values'], batch['pixel_values'])
        self.assertEqual(batch['input_ids'].shape, (1, 5))
        with self.assertRaises(ValueError):
            train.generation_inputs_from_batch(batch, 5)

    def test_schedule_has_exactly_750_updates_and_no_dropped_records(self):
        first = train.epoch_order(18000, 1)
        second = train.epoch_order(18000, 2)
        self.assertEqual(first, train.epoch_order(18000, 1))
        self.assertNotEqual(first, second)
        for order in (first, second):
            groups = train.update_groups(order)
            self.assertEqual(len(groups), 375)
            self.assertTrue(all(len(g) == 48 for g in groups))
            self.assertEqual(sorted(i for g in groups for i in g), list(range(18000)))
        self.assertEqual(train.lr_multiplier(0, 750), 1)
        self.assertEqual(train.lr_multiplier(375, 750), .5)
        self.assertEqual(train.lr_multiplier(750, 750), 0)
        with self.assertRaises(ValueError):
            train.update_groups(list(range(49)))
        with self.assertRaises(ValueError):
            train.lr_multiplier(751, 750)

    def test_four_conditions_reuse_original_images_and_transcriptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, assets, data, output = (root / x for x in ('TRAINING', 'assets', 'prepared', 'validation'))
            images.mkdir()
            assets.mkdir()
            lines = ['file_name\tmisogynous\tText Transcription\n']
            for label in (0, 1):
                for i in range(10):
                    name = f'{label}_{i}.jpg'
                    Image.new('RGB', (16, 16), (label*90, i*20, 70)).save(images / name)
                    lines.append(f'{name}\t{label}\tTranscript {label}/{i}\n')
            (images / 'training.csv').write_text(''.join(lines))
            for code in prep.original.NEW_TRAIN_EMOJIS:
                Image.new('RGBA', (512, 512), (200, 240, 30, 200)).save(assets / f'emoji_u{code}.png')
            real_read = prep.read_training_data
            with patch.object(prep, 'read_training_data', side_effect=lambda p: real_read(p, 10)):
                with contextlib.redirect_stdout(io.StringIO()):
                    prep.prepare(argparse.Namespace(train_dir=str(images), emoji_dir=str(assets),
                                                    output=str(data), seed=42))
            original_complete = (data / 'complete.json').read_bytes()
            original_train = (data / 'train.jsonl').read_bytes()
            image_hashes = {str(p): prep.digest(p) for p in data.rglob('*.jpg')}
            with contextlib.redirect_stdout(io.StringIO()):
                validation.create(argparse.Namespace(data=str(data), output=str(output)), expected_count=2)
            conditions, _ = validation.load(data, output, expected_count=2)
            self.assertEqual(set(conditions), set(validation.CONDITIONS))
            for kind in ('clean', 'overlay'):
                by_id = {r['original_id']: r for r in conditions[f'{kind}_image_only']}
                for row in conditions[f'{kind}_with_transcription']:
                    source = by_id[row['original_id']]
                    self.assertEqual(row['image_path'], source['image_path'])
                    self.assertEqual(row['image_sha256'], source['image_sha256'])
                    label, number = Path(row['original_id']).stem.split('_')
                    self.assertEqual(row['user_prompt'], f"Meme Text: 'Transcript {label}/{number}'\n\n" + source['user_prompt'])
            self.assertEqual((data / 'complete.json').read_bytes(), original_complete)
            self.assertEqual((data / 'train.jsonl').read_bytes(), original_train)
            self.assertEqual({str(p): prep.digest(p) for p in data.rglob('*.jpg')}, image_hashes)
            with self.assertRaises(FileExistsError):
                validation.create(argparse.Namespace(data=str(data), output=str(output)), expected_count=2)
            path = output / 'clean_with_transcription.jsonl'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'records changed'):
                validation.load(data, output, expected_count=2)

    def test_existing_parser_keeps_invalid_outputs_as_errors(self):
        metrics = train.metric_functions()
        self.assertEqual(metrics.parse_answer('clean\nexplanation'), 0)
        self.assertEqual(metrics.parse_answer('clean or misogynous'), -1)
        result = metrics.quality([0, 1], [-1, 1])
        self.assertEqual(result['invalid_count'], 1)
        self.assertEqual(result['accuracy'], .5)


if __name__ == '__main__':
    unittest.main()
