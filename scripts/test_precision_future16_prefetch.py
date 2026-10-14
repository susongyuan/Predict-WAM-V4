import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
import numpy as np
from scripts.run_precision_future16_prefetch import BatchPrefetch, ParallelRamFeatures

class IndexDataset:
    def __len__(self): return 1000
    def __getitem__(self, index): return {'index': index, 'value': index*3}

class PrefetchTests(unittest.TestCase):
    def test_same_samples_and_saved_rng_with_speculative_next_batch(self):
        expected = np.random.default_rng(42)
        actual = BatchPrefetch(IndexDataset(), np.random.default_rng(42), 23, 4, np.random.default_rng)
        try:
            for _ in range(4):
                for _ in range(23):
                    index = actual.integers(1000)
                    self.assertEqual(index, int(expected.integers(1000)))
                    self.assertEqual(actual.get(index), {'index':index, 'value':index*3})
                before = copy.deepcopy(actual.bit_generator.state)
                actual.prefetch_next()
                self.assertEqual(actual.bit_generator.state, before)
                self.assertEqual(actual.bit_generator.state, expected.bit_generator.state)
                resumed = np.random.default_rng(0)
                resumed.bit_generator.state = copy.deepcopy(actual.bit_generator.state)
                shadow = np.random.default_rng(0)
                shadow.bit_generator.state = copy.deepcopy(expected.bit_generator.state)
                self.assertEqual(int(resumed.integers(1000)), int(shadow.integers(1000)))
            self.assertEqual(actual.get(10), {'index':10,'value':30})
            self.assertEqual(actual.bit_generator.state, expected.bit_generator.state)
        finally:
            actual.close()

    def test_sampling_protocol_rejects_duplicate_or_out_of_order_read(self):
        actual = BatchPrefetch(IndexDataset(), np.random.default_rng(42), 23, 2, np.random.default_rng)
        try:
            index = actual.integers(1000)
            with self.assertRaises(ValueError): actual.integers(1000)
            with self.assertRaises(ValueError): actual.get(index+1)
            self.assertEqual(actual.get(index)['index'], index)
        finally:
            actual.close()

    def test_parallel_lru_preserves_features_and_resident_cap(self):
        class Episode: sha256='a'*64
        class Store:
            def get(self, episode, frame):
                return np.full((512,),frame,dtype=np.int32),np.array([frame],np.float32)
        cache = ParallelRamFeatures(Store(), budget_bytes=16000)
        frames = [i%30 for i in range(200)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            results=list(pool.map(lambda frame:cache.get(Episode(),frame),frames))
        for frame,value in zip(frames,results):
            np.testing.assert_array_equal(value[0],np.full((512,),frame,np.int32))
        self.assertLessEqual(cache.stats['peak_resident_bytes'],16000)
        self.assertLessEqual(cache.stats['resident_bytes'],16000)
        self.assertGreater(cache.stats['evictions'],0)

if __name__ == '__main__': unittest.main()
