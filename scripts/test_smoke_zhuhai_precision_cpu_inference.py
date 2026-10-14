#!/usr/bin/env python3
"""Tiny readonly-backend tests, no real checkpoints or datasets."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

spec=importlib.util.spec_from_file_location('cpu_smoke',Path(__file__).with_name('smoke_zhuhai_precision_cpu_inference.py'))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


class ReadOnlyBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.identity=dict(source_sha256='a'*64,encoder_sha256='b'*64,checkpoint_sha256='b'*64,
                           preprocess_sha256='c'*64,frame_index=2,feature_schema_version='cpu-test')
        self.binding={'original':'binding'}
        self.backend=s.ReadOnlyFrameBackend(self.root,self.binding,'a'*64,32)
        encoded=json.dumps(dict(self.identity,hjepa_binding=self.binding),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.key=s.sha(encoded); self.payload=self.root/(self.key+'.payload'); self.meta=self.root/(self.key+'.json')
        self.data=b'small CPU fixture'
        self.payload.write_bytes(self.data)
        self.meta.write_text(json.dumps(dict(key=self.key,identity=json.loads(encoded),size=len(self.data),payload_sha256=s.sha(self.data))))
        self.producer=Mock(side_effect=AssertionError('producer must not execute'))

    def test_identity_matches_original_static_key_and_no_writes(self):
        from harness.zhuhai_hybrid_cache import HybridFeatureCache
        key,_=HybridFeatureCache._identity(dict(self.identity,hjepa_binding=self.binding))
        self.assertEqual(key,self.key)  # Never instantiate the shared cache.
        before={p:p.stat() for p in self.root.iterdir()}
        self.assertEqual(self.backend.read_bytes(self.identity,self.producer),self.data)
        for p,old in before.items():
            current=p.stat()
            self.assertEqual((current.st_mtime_ns,current.st_atime_ns,current.st_ctime_ns),
                             (old.st_mtime_ns,old.st_atime_ns,old.st_ctime_ns))
        self.producer.assert_not_called()
        self.assertEqual(set(before),set(self.root.iterdir()))

    def test_missing_payload_fails_without_generation(self):
        self.payload.unlink()
        with self.assertRaisesRegex(FileNotFoundError,'no generation'):
            self.backend.read_bytes(self.identity,self.producer)
        self.producer.assert_not_called()
        self.assertFalse(self.payload.exists())

    def test_corrupt_hash_and_future_or_other_source_rejected(self):
        self.payload.write_bytes(b'x'*len(self.data))
        with self.assertRaisesRegex(ValueError,'SHA'):
            self.backend.read_bytes(self.identity,self.producer)
        for identity in (dict(self.identity,frame_index=33),dict(self.identity,source_sha256='d'*64)):
            with self.assertRaises(ValueError): self.backend.read_bytes(identity,self.producer)
        self.producer.assert_not_called()

    def test_symlink_and_hardlink_payload_rejected(self):
        other=self.root/'original'; other.write_bytes(self.data)
        self.payload.unlink(); self.payload.symlink_to(other)
        with self.assertRaises(ValueError): self.backend.read_bytes(self.identity,self.producer)
        self.payload.unlink(); os.link(other,self.payload)
        with self.assertRaises(ValueError): self.backend.read_bytes(self.identity,self.producer)
        self.assertEqual(other.read_bytes(),self.data)
        self.producer.assert_not_called()


if __name__=='__main__': unittest.main(verbosity=2)
