"""CPU checks for test membership, frozen overlays and paired uncertainty."""
import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import evaluate_test as evaluation
import prepare_test_conditions as prep


class TestEvaluationTests(unittest.TestCase):
    def test_original_emoji_split_and_balanced_deterministic_assignments(self):
        codes, _ = prep.evaluation_emojis()
        rows = [dict(file_name=f'{c}_{i}.jpg', label=c) for c in (0, 1) for i in range(500)]
        chosen = prep.assignments(rows, list(codes))
        self.assertEqual(chosen, prep.assignments(list(reversed(rows)), list(codes)))
        counts = Counter((r['label'], chosen[r['file_name']]) for r in rows)
        self.assertEqual(set(counts.values()), {25})
        self.assertEqual(len(counts), 40)

    def test_paired_bootstrap_keeps_pairs_and_invalid_outputs(self):
        truth = [0, 0, 1, 1]
        a, b = [-1, 0, 0, 1], [0, 0, 1, 1]
        metrics = evaluation.training.metric_functions()
        self.assertAlmostEqual(float(evaluation.macro_f1_array(truth, a)), metrics.quality(truth, a)['macro_f1'])
        same = evaluation.paired_delta(truth, a, a, repeats=100)
        self.assertEqual(same['paired_stratified_bootstrap_95_interval'], [0, 0])
        result = evaluation.paired_delta(truth, a, b, repeats=100)
        self.assertGreater(result['delta_macro_f1'], 0)
        reverse = evaluation.paired_delta(truth, b, a, repeats=100)
        np.testing.assert_allclose(reverse['paired_stratified_bootstrap_95_interval'],
                                   [-x for x in reversed(result['paired_stratified_bootstrap_95_interval'])])

    def test_preparation_preserves_test_text_labels_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, training, emojis, out = [root/n for n in ('raw', 'training', 'emojis', 'out')]
            for path in (raw, training, emojis):
                path.mkdir()
            text_lines = ['file_name\tText Transcription\n']
            label_lines = []
            for label in (0, 1):
                for index in range(20):
                    name = f'{label}_{index}.jpg'
                    Image.new('RGB', (16, 16), (label*80, index*8, 30)).save(raw/name)
                    text_lines.append(f'{name}\tOriginal text {label}/{index}\n')
                    label_lines.append(f'{name}\t{label}\t0\t0\t0\t0\n')
            (raw/'Test.csv').write_text(''.join(text_lines))
            (raw/'test_labels.txt').write_text(''.join(reversed(label_lines)))
            codes, _ = prep.evaluation_emojis()
            for code in codes:
                Image.new('RGBA', (512, 512), (200, 0, 50, 255)).save(emojis/f'emoji_u{code}.png')
            table = training/'training.csv'
            table.write_text('unused fixture training table')
            manifest = dict(train_table=dict(path=str(table), sha256=prep.prep.digest(table)),
                originals=[dict(file_name='training_only.jpg', sha256='different')],
                assets=[dict(path=str(emojis/'emoji_u1f480.png'))])
            prep.prep.save_json(training/'manifest.json', manifest)
            prep.prep.save_json(training/'complete.json', dict(manifest_sha256=prep.prep.digest(training/'manifest.json'), jsonl_sha256={}))
            args = argparse.Namespace(training_data=str(training), raw_root=str(raw), texts=None,
                labels=None, images=None, emoji_dir=None, output=str(out))
            with contextlib.redirect_stdout(io.StringIO()):
                prep.create(args, expected=40)
            conditions, _ = prep.load(out, expected=40)
            self.assertEqual(set(conditions), set(prep.CONDITIONS))
            for row in conditions['overlay_with_transcription']:
                label, index = Path(row['original_id']).stem.split('_')
                self.assertIn(f'Original text {label}/{index}', row['user_prompt'])
                self.assertEqual(row['target'], 'misogynous' if label == '1' else 'clean')
                self.assertIn(row['emoji_code'], codes)
            self.assertTrue(all(not r['user_prompt'].startswith('Meme Text:') for r in conditions['clean_image_only']))
            with self.assertRaises(FileExistsError):
                prep.create(args, expected=40)
            Path(conditions['overlay_image_only'][0]['image_path']).write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'Test image changed'):
                prep.load(out, expected=40)

    def test_labels_text_ids_must_match_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'Test.csv').write_text('file_name\tText Transcription\na.jpg\ttext\nb.jpg\ttext\n')
            (root/'labels.txt').write_text('a.jpg\t0\na.jpg\t1\n')
            with self.assertRaisesRegex(ValueError, 'duplicate test label'):
                prep.read_test(root/'Test.csv', root/'labels.txt', root, expected=2)
            (root/'labels.txt').write_text('a.jpg\t0\nc.jpg\t1\n')
            with self.assertRaisesRegex(ValueError, 'match exactly'):
                prep.read_test(root/'Test.csv', root/'labels.txt', root, expected=2)


if __name__ == '__main__':
    unittest.main()
