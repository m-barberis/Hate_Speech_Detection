"""CPU checks for bounded prefetch ordering, coverage and failure propagation."""
import threading
import unittest
from contextlib import closing

import training_batches as batching


class PrefetchTests(unittest.TestCase):
    def test_pairs_preserve_epoch_coverage(self):
        import train_comparison as training
        order = training.epoch_order(18000, 1)
        pairs = batching.paired_indices(order)
        self.assertEqual(len(pairs), 9000)
        self.assertEqual([i for pair in pairs for i in pair], order)
        self.assertEqual(training.MICROBATCH * training.ACCUMULATION, 48)
        self.assertEqual(len(training.update_groups(order)), 375)
        with self.assertRaises(ValueError):
            batching.paired_indices([1])

    def test_one_consumer_plus_two_pending_and_close(self):
        prepared = []
        ready = threading.Event()

        def prepare(item):
            prepared.append(item)
            if len(prepared) == 3:
                ready.set()
            return item

        with closing(batching.prefetched(range(100), prepare)) as stream:
            self.assertEqual(next(stream), 0)
            self.assertTrue(ready.wait(2))
            self.assertEqual(prepared, [0, 1, 2])
        self.assertEqual(prepared, [0, 1, 2])

    def test_order_and_worker_failures_are_not_silenced(self):
        self.assertEqual(list(batching.prefetched(range(10), lambda x: x*x)),
                         [x*x for x in range(10)])

        def fail(item):
            if item == 2:
                raise ValueError('corrupt input')
            return item

        with self.assertRaisesRegex(ValueError, 'corrupt input'):
            list(batching.prefetched(range(10), fail))
        with self.assertRaises(ValueError):
            list(batching.prefetched([], str, depth=0))


if __name__ == '__main__':
    unittest.main()
