#!/usr/bin/env python3
"""CPU-only correctness tests; no DINO, robot, training or recording."""
import multiprocessing as mp
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.zhuhai_hybrid_cache import HybridFeatureCache, HARD_DISK_BUDGET_BYTES, MINIMUM_FREE_BYTES


def identity(frame=0):
    return dict(source_sha256='a'*64, encoder_sha256='b'*64,
                preprocess_sha256='c'*64, checkpoint_sha256='d'*64,
                frame_index=frame, feature_schema_version='test-v1')


def concurrent_worker(root, calls, results):
    cache = HybridFeatureCache(root, budget_bytes=100000, min_free_bytes=0)
    def generate():
        with calls.get_lock(): calls.value += 1
        time.sleep(.05)
        return b'shared'
    results.put(cache.read_bytes(identity(), generate))


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'cache'
    def tearDown(self): self.tmp.cleanup()
    def cache(self, budget=70000, **kw):
        return HybridFeatureCache(self.root, budget_bytes=budget, min_free_bytes=0, **kw)

    def test_hit_and_identity_binding(self):
        c=self.cache(); calls=[]
        def gen(): calls.append(1); return b'a'*5000
        self.assertEqual(c.read_bytes(identity(),gen),b'a'*5000)
        self.assertEqual(c.read_bytes(identity(),gen),b'a'*5000)
        self.assertEqual(len(calls),1)
        changed=identity(); changed['encoder_sha256']='e'*64
        c.read_bytes(changed,gen); self.assertEqual(len(calls),2)
        self.assertLessEqual(c.usage_bytes(),c.budget_bytes)

    def test_lru_and_budget_including_metadata(self):
        c=self.cache()
        for i in range(3):
            c.read_bytes(identity(i), lambda i=i:bytes([i])*5000)
            self.assertLessEqual(c.usage_bytes(),c.budget_bytes)
        self.assertEqual(len(list(self.root.glob('*.payload'))),1)
        self.assertGreaterEqual(c.stats['evictions'],2)

    def test_corruption_rebuild(self):
        c=self.cache();c.read_bytes(identity(),lambda:b'first')
        next(self.root.glob('*.payload')).write_bytes(b'corrupt')
        self.assertEqual(c.read_bytes(identity(),lambda:b'rebuilt'),b'rebuilt')
        self.assertEqual(c.stats['corruptions'],1)

    def test_exception_preserves_old_and_raw(self):
        c=self.cache(); c.read_bytes(identity(),lambda:b'old')
        raw=Path(self.tmp.name)/'raw.hdf5';raw.write_bytes(b'original')
        before={p.name:p.read_bytes() for p in self.root.iterdir()}
        def fail(): raise RuntimeError('producer failed')
        with self.assertRaises(RuntimeError):c.read_bytes(identity(1),fail)
        after={p.name:p.read_bytes() for p in self.root.iterdir()}
        self.assertEqual(before,after);self.assertEqual(raw.read_bytes(),b'original')
        self.assertFalse(list(self.root.glob('.write-*')))

    def test_zero_and_oversize_online(self):
        for budget in (0,4096):
            c=self.cache(budget)
            self.assertEqual(c.read_bytes(identity(),lambda:b'x'*5000),b'x'*5000)
            self.assertFalse(list(self.root.glob('*.payload')))
            with self.assertRaisesRegex(OSError, 'bounded disk lease'):
                with c.get_or_compute(identity(),lambda:b'online'*1000):pass
            self.assertFalse(list(Path(self.tmp.name).glob('online-feature-*')))
            self.assertLessEqual(c.usage_bytes(),budget)

    def test_same_key_concurrent_once(self):
        ctx=mp.get_context('fork');calls=ctx.Value('i',0);results=ctx.Queue()
        workers=[ctx.Process(target=concurrent_worker,args=(str(self.root),calls,results)) for _ in range(4)]
        for w in workers:w.start()
        for w in workers:w.join(5);self.assertEqual(w.exitcode,0)
        self.assertEqual([results.get(timeout=1) for _ in workers],[b'shared']*4)
        self.assertEqual(calls.value,1)

    def test_free_floor_online_no_disk_write(self):
        c=HybridFeatureCache(self.root,budget_bytes=100000,min_free_bytes=10**18)
        self.assertEqual(c.read_bytes(identity(),lambda:b'data'),b'data')
        self.assertFalse(list(self.root.glob('*.payload')))
        with self.assertRaises(OSError):
            with c.get_or_compute(identity(),lambda:b'data'):pass

    def test_shrunken_budget_and_unknown_not_evicted(self):
        c=self.cache(100000);c.read_bytes(identity(),lambda:b'x'*5000)
        c=self.cache(4096);c.read_bytes(identity(2),lambda:b'online')
        self.assertLessEqual(c.usage_bytes(),4096)
        unknown=self.root/'raw.txt';unknown.write_bytes(b'preserve')
        c=self.cache(0);self.assertEqual(c.read_bytes(identity(3),lambda:b'yes'),b'yes')
        self.assertEqual(unknown.read_bytes(),b'preserve')
        self.assertEqual(c.usage_bytes(),0)

    def test_zero_shrink_and_orphan_metadata(self):
        c=self.cache(100000);c.read_bytes(identity(),lambda:b'x'*5000)
        next(self.root.glob('*.payload')).unlink()
        c=self.cache(0);self.assertEqual(c.read_bytes(identity(1),lambda:b'new'),b'new')
        self.assertEqual(c.usage_bytes(),0)
        self.assertFalse(list(self.root.glob('*.json')))

    def test_cache_is_not_prefilled_and_source_is_not_counted(self):
        c=self.cache(10**12)
        baseline=c.usage_bytes()
        self.assertLess(baseline,100000)
        raw=self.root/'source_episode.hdf5';raw.write_bytes(b'raw'*10000)
        self.assertEqual(c.usage_bytes(),baseline)
        c.read_bytes(identity(),lambda:b'feature')
        self.assertLess(c.usage_bytes(),baseline+20000)
        self.assertEqual(raw.read_bytes(),b'raw'*10000)

    def test_thousand_entries_hits_and_misses_do_not_scan_directory(self):
        c=self.cache(20000000)
        for i in range(1000): c.read_bytes(identity(i),lambda i=i:str(i).encode())
        scans=c.stats['directory_scans']
        with patch.object(Path,'iterdir',side_effect=AssertionError('hot path directory scan')):
            for i in (0,17,999):
                self.assertEqual(c.read_bytes(identity(i),lambda:b'wrong'),str(i).encode())
            self.assertEqual(c.read_bytes(identity(1001),lambda:b'new'),b'new')
            self.assertLessEqual(c.usage_bytes(),c.budget_bytes)
        self.assertEqual(c.stats['directory_scans'],scans)
        self.assertEqual(c.audit()['entries'],1001)

    def test_lru_touch_retains_recent_entry(self):
        c=self.cache(70000)
        c.read_bytes(identity(0),lambda:b'zero'*1000)
        c.read_bytes(identity(1),lambda:b'one'*1000)
        c.read_bytes(identity(0),lambda:b'wrong')
        c.read_bytes(identity(2),lambda:b'two'*1000)
        calls=[]
        self.assertEqual(c.read_bytes(identity(0),lambda:(calls.append(1) or b'wrong')),b'zero'*1000)
        self.assertEqual(calls,[])
        self.assertGreaterEqual(c.stats['evictions'],1)
        self.assertEqual(c.read_bytes(identity(1),lambda:b'rebuilt-one'),b'rebuilt-one')

    def test_corrupt_sqlite_rebuilds_from_owned_metadata(self):
        c=self.cache(100000);c.read_bytes(identity(),lambda:b'original-feature')
        (self.root/'index.sqlite3').write_bytes(b'broken SQLite')
        rebuilt=self.cache(100000)
        self.assertEqual(rebuilt.read_bytes(identity(),lambda:b'wrong'),b'original-feature')
        self.assertLessEqual(rebuilt.usage_bytes(),rebuilt.budget_bytes)

    def test_restart_recovers_index_and_lru(self):
        c=self.cache(100000)
        for i in range(3):c.read_bytes(identity(i),lambda i=i:bytes([i])*5000)
        before=c.usage_bytes()
        restarted=self.cache(100000)
        self.assertEqual(restarted.read_bytes(identity(2),lambda:b'wrong'),bytes([2])*5000)
        self.assertEqual(restarted.usage_bytes(),before)
        self.assertEqual(restarted.stats['directory_scans'],1)

    def test_identity_required_and_symlink_rejected(self):
        c=self.cache()
        with self.assertRaises(ValueError):c.read_bytes({},lambda:b'bad')
        raw=Path(self.tmp.name)/'raw';raw.write_bytes(b'raw')
        (self.root/'link').symlink_to(raw)
        with self.assertRaises(ValueError):c.audit()
        self.assertEqual(raw.read_bytes(),b'raw')

    def test_decimal_terabyte_is_a_hard_ceiling(self):
        self.assertEqual(HARD_DISK_BUDGET_BYTES, 1_000_000_000_000)
        self.assertEqual(MINIMUM_FREE_BYTES, 107_374_182_400)
        c = HybridFeatureCache(self.root)
        self.assertEqual(c.budget_bytes, 10**12)
        self.assertEqual(c.min_free_bytes, 100*1024**3)
        for size in (10**12+1, 1024**4, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                HybridFeatureCache(self.root, budget_bytes=size)

    def test_low_free_disk_does_not_create_sqlite_or_spill(self):
        from collections import namedtuple
        usage = namedtuple('usage', 'total used free')(10**13, 0, MINIMUM_FREE_BYTES)
        c = HybridFeatureCache(self.root)
        with patch('harness.zhuhai_hybrid_cache.shutil.disk_usage', return_value=usage):
            self.assertEqual(c.read_bytes(identity(), lambda:b'feature'), b'feature')
            self.assertFalse((self.root/'index.sqlite3').exists())
            with self.assertRaises(OSError):
                with c.get_or_compute(identity(), lambda:b'feature'):pass
            self.assertEqual(c.usage_bytes(), 0)
        self.assertFalse(list(self.root.parent.glob('online-feature-*')))

    def test_successful_lease_stays_within_cache_and_budget(self):
        c = self.cache()
        with c.get_or_compute(identity(), lambda:b'admitted') as path:
            self.assertEqual(path.parent, self.root)
            self.assertEqual(path.read_bytes(), b'admitted')
            self.assertLessEqual(c._usage(), c.budget_bytes)
        self.assertTrue(path.exists())

    def test_unknown_hash_named_evidence_and_orphans_are_preserved(self):
        c = self.cache(100000)
        key, _ = c._identity(identity())
        protected = {self.root/(key+'.payload'):b'original evidence',
                     self.root/(key+'.json'):b'not cache metadata',
                     self.root/('.write-'+'e'*64+'-orphan'):b'unknown interrupted evidence',
                     self.root/'source.hdf5':b'source', self.root/'checkpoint.pt':b'weights',
                     self.root/'evaluation.json':b'physical result'}
        for path, value in protected.items():path.write_bytes(value)
        self.assertEqual(c.read_bytes(identity(), lambda:b'online'), b'online')
        for path, value in protected.items():self.assertEqual(path.read_bytes(), value)
        self.assertGreater(c.stats['preserved_unowned'], 0)
        self.assertLessEqual(c.usage_bytes(), c.budget_bytes)

    def test_unowned_index_is_never_recreated(self):
        c = self.cache()
        path = self.root/'index.sqlite3'
        path.write_bytes(b'original non-cache evidence')
        with self.assertRaisesRegex(ValueError, 'Unowned cache index'):
            c.read_bytes(identity(), lambda:b'x')
        self.assertEqual(path.read_bytes(), b'original non-cache evidence')

    def test_post_startup_symlink_and_hardlink_fail_closed(self):
        c = self.cache(100000)
        c.read_bytes(identity(), lambda:b'feature')
        payload = next(self.root.glob('*.payload'))
        raw = self.root.parent/'source.hdf5'; raw.write_bytes(b'preserve source')
        payload.unlink(); payload.symlink_to(raw)
        with self.assertRaises(ValueError):c.read_bytes(identity(), lambda:b'replacement')
        self.assertEqual(raw.read_bytes(), b'preserve source')
        payload.unlink()
        import os
        os.link(raw, payload)
        with self.assertRaises(ValueError):c.read_bytes(identity(), lambda:b'replacement')
        self.assertEqual(raw.read_bytes(), b'preserve source')

    def test_publication_collision_never_overwrites_external_file(self):
        c = self.cache(100000)
        key, _ = c._identity(identity())
        target = self.root/(key+'.json')
        import os
        real_link = os.link
        def collision(source, destination):
            if Path(destination) == target:
                target.write_bytes(b'external evidence')
            return real_link(source, destination)
        with patch('harness.zhuhai_hybrid_cache.os.link', side_effect=collision):
            with self.assertRaises(FileExistsError):c.read_bytes(identity(), lambda:b'feature')
        self.assertEqual(target.read_bytes(), b'external evidence')
        self.assertFalse(list(self.root.glob('.write-*')))

if __name__=='__main__':unittest.main(verbosity=2)
