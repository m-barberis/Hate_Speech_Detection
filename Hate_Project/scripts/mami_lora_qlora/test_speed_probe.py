"""CPU checks of sparse causal-logit loss alignment and probe isolation."""
import unittest
from unittest.mock import patch

import numpy as np
import speed_probe as probe


class SpeedProbeTests(unittest.TestCase):
    def test_selected_logits_match_full_masked_causal_loss(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(2, 8, 11))
        # Right-padded second record and different answer starting positions.
        labels = np.array([[-100, -100, -100, -100, -100, 2, 3, 4],
                           [-100, -100, -100, 5, 6, -100, -100, -100]])
        positions = probe.selected_positions(labels)
        self.assertEqual(positions, [2, 3, 4, 5, 6])

        def losses(selected_logits, targets):
            shifted = selected_logits - selected_logits.max(axis=-1, keepdims=True)
            log_prob = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            means = []
            for row in range(2):
                mask = targets[row] != -100
                means.append(-log_prob[row, np.flatnonzero(mask), targets[row, mask]].mean())
            return np.array(means)

        full = losses(logits[:, :-1], labels[:, 1:])
        selected = losses(logits[:, positions], labels[:, np.array(positions)+1])
        np.testing.assert_allclose(full, selected, rtol=0, atol=1e-14)
        # Equal weighting of records differs from token-weighted mean when
        # answer lengths differ; batching must preserve the former.
        self.assertNotAlmostEqual(float(full.mean()), float((full * [3, 2]).sum()/5))
        with self.assertRaises(ValueError):
            probe.selected_positions([[1, -100, 2]])
        with self.assertRaises(ValueError):
            probe.selected_positions([[-100, -100]])

    def test_timing_records_are_disjoint_from_warmup_originals(self):
        records = [dict(original_id=f'TRAINING/{i//2}.jpg', is_augmented=bool(i % 2)) for i in range(120)]
        warmup = records[:4]
        selected = probe.benchmark_rows(records, warmup)
        self.assertEqual(len(selected), 48)
        self.assertFalse({r['original_id'] for r in selected} & {r['original_id'] for r in warmup})
        self.assertEqual(selected, probe.benchmark_rows(records, warmup))
        self.assertEqual(len(probe.groups_for(selected, 2)), 24)

    def test_gpu_inventory_excludes_only_own_process(self):
        with patch.object(probe.os, 'getpid', return_value=10), \
             patch.object(probe.subprocess, 'check_output', return_value='10\n20\n'):
            self.assertEqual(probe.other_gpu_processes(), [20])


if __name__ == '__main__':
    unittest.main()
