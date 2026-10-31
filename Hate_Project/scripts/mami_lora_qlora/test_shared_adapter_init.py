"""Check common initialization and control comparisons without Torch or a GPU."""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import shared_adapter_init as shared


class SharedInitializationTests(unittest.TestCase):
    def setUp(self):
        self.shapes = {
            'base_model.model.layer.q_proj.lora_A.default.weight': (2, 8),
            'base_model.model.layer.q_proj.lora_B.default.weight': (4, 2),
            'base_model.model.embed_tokens.lora_embedding_A.default': (2, 10),
            'base_model.model.embed_tokens.lora_embedding_B.default': (8, 2),
        }

    def test_hash_ignores_order_and_preserves_names_shapes_values(self):
        arrays = shared.initialize_arrays(self.shapes)
        self.assertEqual(shared.canonical_hash(arrays), shared.canonical_hash(list(reversed(list(arrays.items())))))
        name = next(n for n in arrays if '.lora_A.' in n)
        altered = dict(arrays)
        altered[name] = arrays[name].copy()
        altered[name].flat[0] = np.nextafter(altered[name].flat[0], np.float32(np.inf))
        self.assertNotEqual(shared.canonical_hash(arrays), shared.canonical_hash(altered))
        altered[name] = arrays[name].reshape(4, 4)
        self.assertNotEqual(shared.canonical_hash(arrays), shared.canonical_hash(altered))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            shared.canonical_hash([(name, arrays[name]), (name, arrays[name])])

    def test_default_zero_update_and_initialization_distribution(self):
        arrays = shared.initialize_arrays(self.shapes)
        again = shared.initialize_arrays(dict(reversed(list(self.shapes.items()))))
        for name, array in arrays.items():
            np.testing.assert_array_equal(array, again[name])
            self.assertEqual(array.dtype, np.float32)
            if '.lora_B.' in name or '.lora_embedding_A.' in name:
                self.assertFalse(array.any())
            elif '.lora_A.' in name:
                self.assertTrue((np.abs(array) <= 1 / math.sqrt(array.shape[1])).all())
                self.assertTrue(array.any())
            else:
                self.assertTrue(array.any())
        shared.validate_arrays(arrays, self.shapes)
        with self.assertRaisesRegex(ValueError, 'names differ'):
            shared.validate_arrays({}, self.shapes)

    def make_runs(self, root):
        controls = dict(checkpoint={'model': 'QCRI/MemeLens-VLM', 'revision': 'a'*40},
            versions={}, prepared_complete_sha256='data', prepared_manifest_sha256='manifest',
            legacy_trainer_sha256='legacy', runner_sha256='old-runner',
            lora={'r': 2}, learning_rate=2e-4, smoke_microbatches=4,
            gradient_checkpointing='nonreentrant', max_pixels=524288, min_pixels=4096,
            dispatch_force_hooks=True, base_policy='fixed', selected=[])
        for mode in ('fp16', 'nf4'):
            out = root / mode
            out.mkdir()
            entries = [dict(name=n, shape=list(s), dtype='torch.float32') for n, s in self.shapes.items()]
            if mode == 'nf4':
                entries.reverse()
            inventory = dict(trainable_parameters=entries,
                trainable_parameter_count=sum(math.prod(s) for s in self.shapes.values()),
                processor={}, gpu='V100', gpu_uuid='fixed',
                quantized_modules=['vision.linear'] if mode == 'nf4' else [],
                adapter_target_matches=['layer.q_proj', 'embed_tokens'], initial_adapter_sha256=mode)
            (out / 'request.json').write_text(json.dumps(dict(mode=mode, **controls)))
            (out / 'model_inventory.json').write_text(json.dumps(inventory))
            (out / 'complete.json').write_text(json.dumps({'mode': mode}))
            (out / 'microbatches.jsonl').write_text(''.join(json.dumps(dict(original_id=str(i),
                is_augmented=bool(i % 2), input_signature=f'input-{i}'))+'\n' for i in range(4)))

    def test_saved_initialization_and_inventory_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_runs(root)
            _, _, diagnostics = shared.compare_runs(root / 'fp16', root / 'nf4')
            self.assertFalse(diagnostics['parameter_order_matches'])
            args = argparse.Namespace(fp16=str(root / 'fp16'), nf4=str(root / 'nf4'), output=str(root / 'shared'))
            with contextlib.redirect_stdout(io.StringIO()):
                shared.create(args)
            arrays, meta = shared.load(root / 'shared')
            self.assertEqual(meta['initial_adapter_sha256'], shared.canonical_hash(arrays))
            with self.assertRaises(FileExistsError):
                shared.create(args)
            path = root / 'nf4' / 'microbatches.jsonl'
            original = path.read_text()
            path.write_text(original.replace('input-0', 'different'))
            with self.assertRaisesRegex(ValueError, 'input signatures'):
                shared.compare_runs(root / 'fp16', root / 'nf4')
            path.write_text(original)
            inventory_path = root / 'nf4' / 'model_inventory.json'
            inventory = json.loads(inventory_path.read_text())
            inventory['trainable_parameters'][0]['shape'][0] += 1
            inventory_path.write_text(json.dumps(inventory))
            with self.assertRaises(ValueError):
                shared.compare_runs(root / 'fp16', root / 'nf4')
            weights = root / 'shared' / 'adapter_init.npz'
            weights.write_bytes(weights.read_bytes() + b'altered')
            with self.assertRaisesRegex(ValueError, 'file hash'):
                shared.load(root / 'shared')


if __name__ == '__main__':
    unittest.main()
