"""CPU-only checks of the 100-meme sample, 20-emoji pairing and clustered intervals."""
import argparse
from collections import Counter
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import prepare_emoji_sweep as preparation
import evaluate_emoji_sweep as evaluation


def source_rows():
    rows = [dict(original_id=f'TEST/{label}_{i}.jpg', target='misogynous' if label else 'clean',
        split='test', system_prompt='system', user_prompt=f'Meme Text: {label}/{i}',
        is_augmented=True, emoji_code='old') for label in (0, 1) for i in range(500)]
    assigned = preparation.patterns.assign_patterns(rows)
    return [dict(r, pattern=assigned[r['original_id']]) for r in rows]


def predictions(ids=range(12), codes=('a', 'b')):
    return [dict(condition=evaluation.CONDITION, original_id=str(i), emoji_code=code,
        true_label=i % 2, pattern=preparation.patterns.PATTERNS[i % 3],
        pred_label=(i % 2 if code == 'a' else -1), raw_answer='', strict_format=False)
        for i in ids for code in codes]


class EmojiSweepTests(unittest.TestCase):
    def test_sample_has_exact_quotas_and_is_order_independent(self):
        rows = source_rows()
        ids = preparation.select_ids(rows)
        self.assertEqual(ids, preparation.select_ids(list(reversed(rows))))
        self.assertEqual(len(ids), 100)
        selected = [r for r in rows if r['original_id'] in ids]
        self.assertEqual(Counter(r['target'] for r in selected), dict(clean=50, misogynous=50))
        self.assertEqual(Counter(r['pattern'] for r in selected), dict(dense_random=34, center=33, sparse_random=33))
        for label in (0, 1):
            self.assertEqual(Counter(r['pattern'] for r in selected
                if int(r['target'] == 'misogynous') == label), preparation.QUOTAS[label])

    def test_preparation_preserves_positions_prompts_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, grid, out = root/'source', root/'grid', root/'output'
            source.mkdir(); grid.mkdir()
            base = root/'original.jpg'
            Image.new('RGB', (12, 16), (40, 80, 120)).save(base)
            codes = sorted(preparation.grid_data.evaluation_emojis()[0])
            assets = []
            for index, code in enumerate(codes):
                path = root/f'{code}.png'
                Image.new('RGBA', (8, 8), (index*10, 50, 150, 180)).save(path)
                assets.append(dict(code=code, path=str(path), sha256=preparation.prep.digest(path)))
            rows = source_rows()
            clean = [dict(r, image_path=str(base), image_sha256=preparation.prep.digest(base)) for r in rows]
            geometry = {r['original_id']: dict(pattern=r['pattern'], emoji_code='old', nominal_alpha=.8,
                placements=preparation.patterns.placements((12, 16), r['pattern'], r['original_id'])) for r in rows}
            old_manifest = dict(source_data=str(grid), training_manifest_sha256='training', geometries=geometry)
            (source/'manifest.json').write_text(json.dumps(old_manifest))
            (source/'complete.json').write_text('{}')
            (grid/'complete.json').write_text('{}')
            (grid/'manifest.json').write_text(json.dumps(dict(assets=assets)))
            original = dict(clean_with_transcription=clean)
            before = (source/'manifest.json').read_bytes()
            with patch.object(preparation.patterns, 'load', return_value=({evaluation.CONDITION: rows}, {})), \
                 patch.object(preparation.grid_data, 'load', return_value=(original, {})):
                with contextlib.redirect_stdout(io.StringIO()):
                    preparation.create(argparse.Namespace(source_data=str(source), output=str(out)))
                converted, done = preparation.load(out)
                records = converted[evaluation.CONDITION]
                self.assertEqual(len(records), 2000)
                self.assertEqual(set(Counter(r['original_id'] for r in records).values()), {20})
                manifest = json.loads((out/'manifest.json').read_text())
                first = records[0]
                self.assertEqual(manifest['geometries'][first['original_id']],
                    json.loads(json.dumps(geometry[first['original_id']])))
                expected = root/'expected.jpg'
                with Image.open(base) as image, Image.open(assets[0]['path']) as asset:
                    preparation.patterns.render(image, asset.convert('RGBA'),
                        geometry[first['original_id']]['placements']).save(expected, format='JPEG')
                self.assertEqual(expected.read_bytes(), Path(first['image_path']).read_bytes())
                self.assertEqual(before, (source/'manifest.json').read_bytes())
                with self.assertRaises(FileExistsError):
                    preparation.create(argparse.Namespace(source_data=str(source), output=str(out)))
                Path(first['image_path']).write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'overlay bytes changed'):
                    preparation.load(out)

    def test_cluster_bootstrap_retains_variants_and_invalid_errors(self):
        a = predictions()
        b = [dict(r, pred_label=r['true_label']) for r in a]
        same = evaluation.paired_delta(a, a, repeats=100)
        self.assertEqual(same['paired_meme_cluster_label_pattern_stratified_95_interval'], [0, 0])
        result = evaluation.paired_delta(a, b, repeats=100)
        reverse = evaluation.paired_delta(b, a, repeats=100)
        self.assertGreater(result['delta_macro_f1'], 0)
        np.testing.assert_allclose(reverse['paired_meme_cluster_label_pattern_stratified_95_interval'],
            [-x for x in reversed(result['paired_meme_cluster_label_pattern_stratified_95_interval'])])
        doubled = [dict(r, emoji_code=r['emoji_code']+suffix) for r in a for suffix in ('1', '2')]
        doubled_b = [dict(r, pred_label=r['true_label']) for r in doubled]
        self.assertEqual(result, evaluation.paired_delta(doubled, doubled_b, repeats=100))
        _, counts, _, _ = evaluation.meme_counts(a)
        expected = evaluation.shared.original.macro_f1_array(
            [r['true_label'] for r in a], [r['pred_label'] for r in a])
        self.assertAlmostEqual(float(evaluation.f1_counts(counts.sum(axis=0))), float(expected))
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            evaluation.paired_delta(a, list(reversed(b)), repeats=10)

    def test_summary_uses_only_selected_clean_memes_and_checks_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out, clean_root = root/'results', root/'clean'
            chosen = preparation.select_ids(source_rows())
            lookup = {r['original_id']: r for r in source_rows()}
            codes = sorted(preparation.grid_data.evaluation_emojis()[0])
            source = [dict(lookup[i], emoji_code=c) for i in chosen for c in codes]
            records = [dict(condition=evaluation.CONDITION, original_id=r['original_id'], emoji_code=r['emoji_code'],
                pattern=r['pattern'], true_label=int(r['target'] == 'misogynous'),
                pred_label=int(r['target'] == 'misogynous'), strict_format=True,
                input_signature=r['original_id']+r['emoji_code']) for r in source]
            clean_rows = [dict(condition='clean_with_transcription', original_id=r['original_id'],
                true_label=int(r['target'] == 'misogynous'), pred_label=0,
                strict_format=True, input_signature=r['original_id']) for r in source_rows()]
            data_root = root/'data'; data_root.mkdir()
            manifest = dict(grid_data=str(root/'grid'), grid_preparation_sha256='grid',
                selected_ids=chosen, emoji_codes=codes, pattern_counts=dict(dense_random=34, center=33, sparse_random=33))
            (data_root/'manifest.json').write_text(json.dumps(manifest))
            (data_root/'complete.json').write_text('{}')
            gates = {mode: dict(completion_sha256=mode, complete=dict(final_adapter_sha256=mode+'-adapter'))
                for mode in ('fp16', 'nf4')}
            hashes = {mode: mode for mode in gates}
            for arm in evaluation.ARMS:
                for parent in (out, clean_root):
                    directory = parent/arm; directory.mkdir(parents=True)
                    meta = {k: 'same' for k in evaluation.CONTROLS+('gpu', 'gpu_uuid', 'runner_sha256', 'inference_source_sha256')}
                    meta['adapter_hash'] = None if arm.endswith('-base') else arm.split('-')[0]+'-adapter'
                    (directory/'metadata.json').write_text(json.dumps(meta))
                    (directory/'complete.json').write_text('{}')
            def completed(folder, *_):
                done = dict(arm=folder.name, training_completion_sha256=hashes, evaluation_wall_seconds=1,
                    peak_inference_allocated_bytes=2, peak_inference_reserved_bytes=3)
                return done, (records if folder.parent == out else clean_rows)
            args = argparse.Namespace(data=str(data_root), output_root=str(out), clean_results=str(clean_root))
            with patch.object(evaluation.data, 'load', return_value=({evaluation.CONDITION: source}, {})), \
                 patch.object(evaluation.shared, 'verify_test_training_link'), \
                 patch.object(evaluation.shared, 'training_gates', return_value=gates), \
                 patch.object(evaluation.data.grid_data, 'load', return_value=({}, {})), \
                 patch.object(evaluation, 'read_completed', side_effect=completed), \
                 patch.object(evaluation.shared.original, 'read_completed', side_effect=completed), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluation.summarize(args)
                report = json.loads((out/'comparison.json').read_text())
                self.assertEqual(report['total_predictions'], 8000)
                self.assertEqual(report['arms']['fp16-base']['matched_clean']['n'], 100)
                self.assertEqual(report['arms']['fp16-base']['by_pattern']['all']['n'], 2000)
                self.assertEqual(report['paired_model_contrasts']['all']['fp16-lora_minus_fp16-base']['delta_macro_f1'], 0)
                evaluation.summarize(args)  # verified idempotent summary
                meta_path = out/'nf4-base/metadata.json'
                meta = json.loads(meta_path.read_text()); meta['generation'] = 'different'
                meta_path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, 'Cannot reuse clean predictions'):
                    evaluation.summarize(args)

    def test_completed_reader_rejects_duplicate_variants_and_incorrect_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ids = preparation.select_ids(source_rows())
            lookup = {r['original_id']: r for r in source_rows()}
            codes = sorted(preparation.grid_data.evaluation_emojis()[0])
            source = [dict(lookup[i], emoji_code=c) for i in ids for c in codes]
            records = [dict(condition=evaluation.CONDITION, original_id=r['original_id'],
                emoji_code=r['emoji_code'], pattern=r['pattern'], true_label=int(r['target'] == 'misogynous'),
                pred_label=0, raw_answer='clean', strict_format=True) for r in source]
            meta = dict(arm='fp16-base', test_preparation_sha256='prepared', training_completion_sha256={},
                runner_sha256=evaluation.shared.common.digest(evaluation.__file__),
                inference_source_sha256=evaluation.shared.common.digest(evaluation.shared.__file__))
            (root/'metadata.json').write_text(json.dumps(meta))
            done = dict(arm='fp16-base', test_preparation_sha256='prepared', n_originals=100, predictions=2000,
                training_completion_sha256={}, metadata_sha256=evaluation.shared.common.digest(root/'metadata.json'))
            summary = evaluation.quality(records); summary.pop('n_originals')
            done['metrics'] = {evaluation.CONDITION: summary}
            def save():
                (root/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                done['predictions_sha256'] = evaluation.shared.common.digest(root/'predictions.jsonl')
                (root/'complete.json').write_text(json.dumps(done))
            save()
            evaluation.read_completed(root, {evaluation.CONDITION: source}, 'prepared')
            saved = records[1]; records[1] = records[0]; save()
            with self.assertRaisesRegex(ValueError, 'Missing, duplicate or reordered'):
                evaluation.read_completed(root, {evaluation.CONDITION: source}, 'prepared')
            records[1] = saved; records[0]['pred_label'] = 1; save()
            with self.assertRaisesRegex(ValueError, 'parsing differs'):
                evaluation.read_completed(root, {evaluation.CONDITION: source}, 'prepared')

    def test_launcher_dispatches_four_sequential_arms_and_refuses_missing_clean(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bin_dir, clean = root/'bin', root/'clean'
            bin_dir.mkdir()
            log = root/'calls'
            stub = bin_dir/'python'
            stub.write_text('#!/bin/sh\nprintf "%s|%s\\n" "${CUDA_VISIBLE_DEVICES:-cpu}" "$*" >> "$SWEEP_TEST_LOG"\n')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'], SWEEP_TEST_LOG=str(log))
            script = str(Path(evaluation.__file__).with_name('run_emoji_sweep.sh'))
            command = ['bash', script, str(root/'data'), str(root/'results'), str(clean)]
            missing = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertFalse(log.exists())
            for arm in evaluation.ARMS:
                (clean/arm).mkdir(parents=True)
                (clean/arm/'complete.json').write_text('{}')
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = log.read_text().splitlines()
            self.assertEqual(len(calls), 5)
            for call, arm in zip(calls[:4], evaluation.ARMS):
                self.assertTrue(call.startswith('0|'))
                self.assertIn(f'--arm {arm}', call)
                self.assertIn('--reuse-complete', call)
            self.assertIn('summarize', calls[-1])


if __name__ == '__main__':
    unittest.main()
