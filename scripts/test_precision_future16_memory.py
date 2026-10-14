"""Verify the user-authorized swap threshold without GPU or real memory pressure."""
import unittest
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts import run_precision_future16_memory as runtime

class MemoryReserveTests(unittest.TestCase):
    def check_fraction(self, fraction):
        with patch.object(runtime, 'ORIGINAL_RESERVE', side_effect=RuntimeError('95% swap reserve reached')), \
             patch.object(runtime, 'memory', return_value={'SwapTotal': 1000, 'SwapFree': 1000-fraction}):
            return runtime.reserve(object())

    def test_swap_95_to_below_98_allowed(self):
        for fraction in (950, 970, 979):
            self.check_fraction(fraction)

    def test_swap_98_and_above_stops(self):
        for fraction in (980, 990, 1000):
            with self.assertRaisesRegex(RuntimeError, '98% swap reserve reached'):
                self.check_fraction(fraction)

    def test_other_reserves_remain_enforced(self):
        for message in ('RAM reserve reached', 'Disk reserve reached', 'Wrong GPU owner'):
            with patch.object(runtime, 'ORIGINAL_RESERVE', side_effect=RuntimeError(message)):
                with self.assertRaisesRegex(RuntimeError, message):
                    runtime.reserve(object())

    def test_original_startup_check_is_called(self):
        args = object()
        with patch.object(runtime, 'ORIGINAL_RESERVE') as original:
            runtime.reserve(args, startup=True)
            original.assert_called_once_with(args, startup=True)

class RuntimeProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from harness.zhuhai_hybrid_cache import HybridFeatureCache
        cls.original_cache = HybridFeatureCache
        cls.profile = dict(swap_stop_fraction=.98, torch_cpu_threads=8,
            cuda_memory_fraction=.95, ram_feature_cache_bytes=8*1024**3)
        runtime.install_profile(cls.profile)

    def test_read_hit_does_not_write_index_or_use_generator(self):
        from harness.zhuhai_hybrid_cache import HybridFeatureCache
        from scripts.test_zhuhai_hybrid_cache import identity
        with tempfile.TemporaryDirectory() as directory:
            initial = self.original_cache(directory, budget_bytes=100000, min_free_bytes=0)
            initial.read_bytes(identity(), lambda: b'valid')
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            cache = HybridFeatureCache(directory, budget_bytes=100000, min_free_bytes=0)
            self.assertEqual(cache.read_bytes(identity(), lambda: self.fail('unexpected regeneration')), b'valid')
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})
            self.assertEqual(cache.stats['directory_scans'], 0)

    def test_corrupt_payload_is_rebuilt_with_original_verification(self):
        from harness.zhuhai_hybrid_cache import HybridFeatureCache
        from scripts.test_zhuhai_hybrid_cache import identity
        with tempfile.TemporaryDirectory() as directory:
            self.original_cache(directory, budget_bytes=100000, min_free_bytes=0).read_bytes(identity(), lambda:b'valid')
            next(Path(directory).glob('*.payload')).write_bytes(b'corrupt')
            cache = HybridFeatureCache(directory, budget_bytes=100000, min_free_bytes=0)
            self.assertEqual(cache.read_bytes(identity(), lambda:b'rebuilt'), b'rebuilt')
            self.assertGreater(cache.stats['corruptions'], 0)

    def test_ram_cap_and_contract_binding(self):
        from harness.zhuhai_training import BoundedRamFeatures
        cache = BoundedRamFeatures(object(), budget_bytes=8*1024**3)
        self.assertEqual(cache.budget_bytes, 8*1024**3)
        raw = {'source_files': {}, 'hjepa': {}}
        bound = runtime.bind_contract(raw, self.profile)
        self.assertEqual(raw, {'source_files': {}, 'hjepa': {}})
        self.assertEqual(bound['runtime_performance_profile'], self.profile)
        self.assertEqual(bound['source_files'], bound['hjepa']['code_hashes'])

if __name__ == '__main__':
    unittest.main()
