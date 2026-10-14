#!/usr/bin/env python3
"""CPU-only regression checks; no GPU/simulator or original evidence writes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repeat', BASE / 'runner/repeat.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

class SafetyTests(unittest.TestCase):
    def test_bundle(self):
        self.assertEqual(len(r.verify_bundle()), 64)

    def test_seed0_forbidden(self):
        for seeds in ([0], [1, 1], [2, 1], [10], []):
            with self.assertRaises(RuntimeError):
                r.validate_seeds(seeds)

    def test_new_seeds_unexecuted(self):
        self.assertEqual([r.receipt_state(s)[0] for s in range(1, 10)], ['new'] * 9)

    def test_execute_blocks_before_lock_or_probe(self):
        with patch.object(r, 'chain_lock', side_effect=AssertionError('lock reached')), patch.object(r, 'dry_run', side_effect=AssertionError('probe reached')):
            with self.assertRaisesRegex(RuntimeError, 'GPU blocked'):
                r.main(['--execute'])

    def test_gpu_api_blocked_before_process_scan(self):
        with patch.object(r, 'nonempty_training_processes', side_effect=AssertionError('scan reached')):
            with self.assertRaisesRegex(RuntimeError, 'GPU blocked'):
                r.acquire_gpu('test')

    def test_run_api_blocked_before_receipt_or_launch(self):
        with patch.object(r, 'receipt_state', side_effect=AssertionError('receipt reached')):
            with self.assertRaisesRegex(RuntimeError, 'GPU blocked'):
                r.run_one(1, launch=lambda *a, **k: self.fail('launch reached'))

    def test_preparation_refuses_overwrite(self):
        paths = [BASE/'protocol.json', BASE/'runner/runner_integrity.json']
        before = [p.read_bytes() for p in paths]
        result = subprocess.run(['/usr/bin/python3','-B',str(BASE/'runner/prepare_protocol.py')], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refusing automatic replacement', result.stderr)
        self.assertEqual(before, [p.read_bytes() for p in paths])

    def test_no_admission_claim(self):
        p = r.read(r.PROTOCOL)
        self.assertFalse(p['execution_admission']['allowed'])
        self.assertEqual(p['execution_admission']['resource_limit_percent'], 85)
        self.assertIsNone(p['launch_draft'])
        self.assertNotIn('resource_policy', p)
        self.assertFalse(p['reused_seed0']['rerun_allowed'])

    def test_frozen_hash_drift_fails(self):
        original = r.sha
        manifest = r.read(r.INVENTORY)['baseline_manifest']
        with patch.object(r, 'sha', side_effect=lambda p: '0'*64 if str(p) == manifest else original(p)):
            with self.assertRaisesRegex(RuntimeError, 'Original manifest changed'):
                r.verify_image()

if __name__ == '__main__':
    unittest.main(verbosity=2)
