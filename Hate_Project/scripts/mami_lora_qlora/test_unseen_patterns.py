"""CPU checks for fixed layout assignment and original compositing semantics."""
from collections import Counter
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
import prepare_unseen_patterns as patterns
import evaluate_unseen_patterns as evaluation


class UnseenPatternTests(unittest.TestCase):
    def test_preparation_reuses_original_prompts_emojis_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, out = root/'source', root/'new'
            source.mkdir()
            base, asset_path = root/'base.jpg', root/'emoji.png'
            Image.new('RGB', (16, 16), (40, 80, 120)).save(base)
            Image.new('RGBA', (32, 32), (200, 50, 0, 180)).save(asset_path)
            conditions = {c: [] for c in patterns.source_data.CONDITIONS}
            for c in conditions:
                for i in range(1000):
                    conditions[c].append(dict(original_id=f'TEST/{i}.jpg', split='test',
                        image_path=str(base), image_sha256=patterns.prep.digest(base),
                        target='clean' if i < 500 else 'misogynous',
                        system_prompt='original system', user_prompt='Meme Text: original text' if c.endswith('with_transcription') else 'original instruction',
                        is_augmented=c.startswith('overlay_'), emoji_code='skull' if c.startswith('overlay_') else None))
            (source/'complete.json').write_text('{}')
            (source/'manifest.json').write_text(json.dumps(dict(training_manifest_sha256='fixture',
                assets=[dict(code='skull', path=str(asset_path), sha256=patterns.prep.digest(asset_path))])))
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            with patch.object(patterns.source_data, 'load', return_value=(conditions, {})):
                with contextlib.redirect_stdout(io.StringIO()):
                    patterns.create(argparse.Namespace(source_data=str(source), output=str(out)))
                converted, _ = patterns.load(out)
                self.assertEqual(set(converted), set(patterns.CONDITIONS))
                for c in converted:
                    self.assertEqual(len(converted[c]), 1000)
                    for row, old in zip(converted[c], conditions[c]):
                        for key in ('original_id', 'system_prompt', 'user_prompt', 'target', 'emoji_code'):
                            self.assertEqual(row[key], old[key])
                self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
                Path(converted[patterns.CONDITIONS[0]][0]['image_path']).write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'overlay bytes changed'):
                    patterns.load(out)

    def test_fixed_balanced_assignment_and_counts(self):
        rows = [dict(original_id=f'TEST/{c}_{i}.jpg', target='misogynous' if c else 'clean')
                for c in (0, 1) for i in range(500)]
        assigned = patterns.assign_patterns(rows)
        self.assertEqual(assigned, patterns.assign_patterns(list(reversed(rows))))
        self.assertEqual(Counter(assigned.values()), dict(dense_random=334, center=333, sparse_random=333))
        for name in patterns.PATTERNS:
            classes = Counter(r['target'] for r in rows if assigned[r['original_id']] == name)
            self.assertLessEqual(abs(classes['clean']-classes['misogynous']), 1)

    def test_placements_are_repeatable_inside_image_and_use_agreed_sizes(self):
        for shape in ((120, 80), (80, 120), (1, 1)):
            for name, count in (('dense_random', 16), ('center', 1), ('sparse_random', 4)):
                positions = patterns.placements(shape, name, 'TEST/example.jpg')
                self.assertEqual(positions, patterns.placements(shape, name, 'TEST/example.jpg'))
                self.assertEqual(len(positions), count)
                for x, y, side in positions:
                    self.assertGreaterEqual(x, 0); self.assertGreaterEqual(y, 0)
                    self.assertLessEqual(x+side, shape[0]); self.assertLessEqual(y+side, shape[1])
                    self.assertEqual(side, max(1, int(min(shape)*(.75 if name == 'center' else .25))))
        self.assertNotEqual(patterns.placements((120, 80), 'dense_random', 'a'),
                            patterns.placements((120, 80), 'dense_random', 'b'))

    def test_renderer_matches_original_grid_pixel_for_pixel(self):
        rng = np.random.default_rng(42)
        base = Image.fromarray(rng.integers(0, 256, (80, 120, 4), dtype=np.uint8), 'RGBA')
        asset = Image.fromarray(rng.integers(0, 256, (32, 32, 4), dtype=np.uint8), 'RGBA')
        w, h = base.size
        side = int(min(w, h)*.25)
        positions = [(int((i+.5)*(w/4)-side/2), int((j+.5)*(h/4)-side/2), side)
                     for i in range(4) for j in range(4)]
        for alpha in (.5, .8, 1.0):
            expected = patterns.prep.original.apply_4x4_grid(base, asset, alpha)
            actual = patterns.render(base, asset, positions, alpha)
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_pairing_stratification_and_invalid_predictions(self):
        truth = [0, 0, 0, 1, 1, 1]
        labels = ['dense_random', 'center', 'sparse_random']*2
        a, b = [-1, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1]
        same = evaluation.paired_delta(truth, a, a, labels, repeats=30)
        self.assertEqual(same['paired_label_pattern_stratified_95_interval'], [0, 0])
        forward = evaluation.paired_delta(truth, a, b, labels, repeats=30)
        backward = evaluation.paired_delta(truth, b, a, labels, repeats=30)
        self.assertGreater(forward['delta_macro_f1'], 0)
        np.testing.assert_allclose(backward['paired_label_pattern_stratified_95_interval'],
            [-x for x in reversed(forward['paired_label_pattern_stratified_95_interval'])])


if __name__ == '__main__':
    unittest.main()
