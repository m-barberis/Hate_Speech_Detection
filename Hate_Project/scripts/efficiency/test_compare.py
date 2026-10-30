"""CPU-only checks of evaluation validity; no model downloads or inference."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import compare


class ValidityTests(unittest.TestCase):
    def test_invalid_answers_are_errors_not_clean(self):
        self.assertEqual(compare.parse_answer(' clean\n'), 0)
        self.assertEqual(compare.parse_answer('MISOGYNOUS'), 1)
        for answer in ('not misogynous', 'Label: clean', '', 'clean or misogynous'):
            self.assertEqual(compare.parse_answer(answer), -1)
        result = compare.quality([0, 1], [-1, 1])
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['macro_f1'], .5)
        self.assertEqual(result['invalid_count'], 1)
        self.assertEqual(result['confusion_rows_true_0_1_cols_pred_0_1_invalid'], [[0, 0, 1], [0, 1, 0]])

    def test_manifest_is_reproducible_and_warmups_are_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / 'labels.tsv'
            with labels.open('w') as stream:
                writer = csv.writer(stream, delimiter='\t')
                for c in (0, 1):
                    for i in range(55):
                        name = f'{c}_{i}.jpg'
                        (root / name).write_bytes(f'{c}_{i}'.encode())
                        writer.writerow([name, c])
            for output in ('a.json', 'b.json'):
                compare.prepare(argparse.Namespace(images=str(root), labels=str(labels), output=str(root / output)))
            self.assertEqual((root / 'a.json').read_bytes(), (root / 'b.json').read_bytes())
            manifest = json.loads((root / 'a.json').read_text())
            self.assertEqual(len(manifest['evaluation']), 100)
            self.assertEqual(sum(r['label'] for r in manifest['evaluation']), 50)
            self.assertFalse({r['file_name'] for r in manifest['evaluation']} &
                             {r['file_name'] for r in manifest['warmup']})

    def make_run(self, root, mode):
        root.mkdir()
        metadata = dict(mode=mode, checkpoint={'revision': 'a' * 40}, settings=compare.SETTINGS,
                        manifest_sha256='fixed', versions={}, torch_cuda='12.6', gpu='V100',
                        gpu_uuid='fixed-gpu', cudnn_version=1, python='test', processor={},
                        generation_config={}, passes=3, loaded_allocated_bytes=100,
                        load_peak_allocated_bytes=110)
        compare.save_json(root / 'metadata.json', metadata)
        compare.save_json(root / 'complete.json', dict(smoke=False))
        with (root / 'predictions.jsonl').open('w') as stream:
            for p in range(3):
                for i in range(100):
                    stream.write(json.dumps(dict(pass_id=p, file_name=str(i), true_label=i % 2,
                        pred_label=i % 2, raw_answer='clean' if i % 2 == 0 else 'misogynous',
                        input_signature=str(i), latency_ms=10, generated_tokens=2,
                        peak_allocated_bytes=200 if mode == 'fp16' else 100,
                        peak_reserved_bytes=220 if mode == 'fp16' else 120)) + '\n')

    def test_summary_and_comparison_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root / 'fp16', 'fp16')
            self.make_run(root / 'nf4', 'nf4')
            args = argparse.Namespace(fp16=str(root / 'fp16'), nf4=str(root / 'nf4'), output=str(root / 'summary.json'))
            compare.summarize(args)
            result = json.loads((root / 'summary.json').read_text())
            self.assertEqual(result['delta_macro_f1_nf4_minus_fp16'], 0)
            self.assertEqual(result['paired_stratified_bootstrap_delta_95_interval'], [0, 0])
            self.assertEqual(result['peak_allocated_reduction_fraction'], .5)
            meta_path = root / 'nf4' / 'metadata.json'
            meta = json.loads(meta_path.read_text())
            meta['checkpoint']['revision'] = 'b' * 40
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                compare.summarize(args)


if __name__ == '__main__':
    unittest.main()
