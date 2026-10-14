#!/usr/bin/env python3
"""CPU-only entry/handoff regression tests; no model load, decoding or tracking."""
import ast
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import check_task090_early_baseline as entry


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='entry_test_',dir=entry.RUN_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.out=self.root/'attempt';self.out.mkdir()
        for name,value in (('RUN_ROOT',self.root),('OUT',self.out),('np',np),('SOURCES',{})):
            p=patch.object(entry,name,value,create=True);p.start();self.addCleanup(p.stop)

    def test_required_explicit_stage_and_output(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):entry.parse_args([])

    def test_old_output_refused(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            entry.parse_args(['--stage','prepare','--out',str(ROOT/'outputs/old'),'--check-only'])

    def test_execution_needs_preflight(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            entry.parse_args(['--stage','prepare','--out',str(self.out)])

    def test_features_needs_prepared(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            entry.parse_args(['--stage','features','--out',str(self.out),'--preflight',str(self.root)])

    def test_no_alternate_frame(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            entry.parse_args(['--stage','features','--out',str(self.out),'--check-only','--frame','870'])

    def test_check_only_has_no_handoff_requirement(self):
        args=entry.parse_args(['--stage','features','--out',str(self.out),'--check-only'])
        self.assertTrue(args.check_only)

    def test_exclusive_write_preserves_completed_file(self):
        path=self.out/'stage_result.json'
        entry.write_json(path,{'sentinel':1})
        with self.assertRaises(FileExistsError):entry.write_json(path,{'sentinel':2})
        self.assertEqual(json.loads(path.read_text()),{'sentinel':1})

    def test_forward_claim_cannot_be_bypassed_by_new_attempt(self):
        entry.claim_execution('dino_forward_started')
        with patch.object(entry,'OUT',self.root/'attempt02'),self.assertRaises(FileExistsError):
            entry.claim_execution('dino_forward_started')

    def test_handoff_pins_content_version_and_completion(self):
        payload=self.out/'frames.jsonl';payload.write_text('test-only\n')
        report=dict(stage='prepare',check_only=False,status='complete',handoff_version=entry.HANDOFF_VERSION,
            dataset_revision=entry.REVISION,verified=False,admitted=False,files_sha256={'frames.jsonl':entry.sha(payload)})
        entry.write_json(self.out/'stage_result.json',report)
        entry.read_completed(self.out,'prepare')
        payload.write_text('tampered\n')
        with self.assertRaises(AssertionError):entry.read_completed(self.out,'prepare')

    def test_wrong_handoff_version_refused(self):
        entry.write_json(self.out/'stage_result.json',dict(stage='prepare',check_only=False,status='complete',
            handoff_version='wrong',dataset_revision=entry.REVISION,verified=False,admitted=False,files_sha256={}))
        with self.assertRaises(AssertionError):entry.read_completed(self.out,'prepare')

    def test_partial_output_is_not_completed_handoff(self):
        entry.write_json(self.out/'failure.json',{'status':'failed'})
        with self.assertRaises(FileNotFoundError):entry.read_completed(self.out,'prepare')

    def test_feature_phase_has_no_prep_or_continuous_dependency(self):
        tree=ast.parse(Path(entry.__file__).read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='exact_initial_features')
        self.assertFalse({'prep','continuous'} & {n.id for n in ast.walk(fn) if isinstance(n,ast.Name)})

    def test_resources_preserve_real_counts_over_budget_snapshot(self):
        class Budget:
            def snapshot(self):return {'counts':{'incorrect':999}}
        with patch.object(entry,'budget',Budget(),create=True),patch.object(entry,'COUNTS',{'dino_forward_completed':1}):
            self.assertEqual(entry.resources()['counts'],{'dino_forward_completed':1})


if __name__=='__main__':
    unittest.main(verbosity=2)
