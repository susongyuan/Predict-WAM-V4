#!/usr/bin/env python3
"""CPU-only50k handoff/readiness tests; no service controls or GPU calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import handoff_robotwin_multitask_50k as handoff
from scripts import run_robotwin_multitask_extended_pipeline as pipeline
from scripts.run_robotwin_multitask_pipeline import atomic,read,sha

class HandoffTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.out=self.root/'extension_50k';self.out.mkdir()
  atomic(self.root/'pipeline_manifest.json',{'source_sha256':{}})
  atomic(self.root/'train/status.json',{'status':'running','step':4975})
  atomic(self.root/'train/checkpoint_last.pt.sha256.json',{'step':4000,'model_only':False,'sha256':'a'*64})
 def test_wait_requires5000_checkpoint_and_finished_val(self):
  self.assertIsNone(handoff.inspect_ready(self.root))
  atomic(self.root/'train/status.json',{'status':'running','step':5000})
  self.assertIsNone(handoff.inspect_ready(self.root))
  atomic(self.root/'train/checkpoint_last.pt.sha256.json',{'step':5000,'model_only':False})
  self.assertIsNone(handoff.inspect_ready(self.root))
  atomic(self.root/'train/validation_step_005000.json',{'selection':'none'})
  self.assertEqual(handoff.inspect_ready(self.root)['step'],5000)
 def test_wrong_or_missed_checkpoint_refused(self):
  atomic(self.root/'train/checkpoint_last.pt.sha256.json',{'step':6000,'model_only':False})
  with self.assertRaisesRegex(RuntimeError,'missed'):handoff.inspect_ready(self.root)
 def test_unexpected_original_training_stop_not_admitted(self):
  atomic(self.root/'train/status.json',{'status':'failed','step_observed':4999})
  with self.assertRaisesRegex(RuntimeError,'unexpectedly'):handoff.inspect_ready(self.root)
 def test_wrapper_waits_then_verifies_parent(self):
  self.assertFalse(pipeline.ready(self.root,self.out))
  path=self.root/'safe_checkpoint';path.write_bytes(b'CPU fixture')
  h={'status':'complete','parent_step':5000,'new_total':50000,'source_changes':[], 'no_native_evaluation_started':True,'old_unit_properties':'MainPID=0\nActiveState=failed', 'parent_checkpoint':str(path),'parent_sha256':sha(path)}
  atomic(self.out/'handoff.json',h);self.assertTrue(pipeline.ready(self.root,self.out))
  path.write_bytes(b'changed')
  with self.assertRaisesRegex(RuntimeError,'immutable'):pipeline.ready(self.root,self.out)
 def test_wrapper_refuses_prior_evaluation_or_wrong_budget(self):
  h={'status':'complete','parent_step':5000,'new_total':20000,'source_changes':[], 'no_native_evaluation_started':True,'old_unit_properties':'MainPID=0'}
  atomic(self.out/'handoff.json',h)
  with self.assertRaises(RuntimeError):pipeline.ready(self.root,self.out)
  h['new_total']=50000;h['no_native_evaluation_started']=False;atomic(self.out/'handoff.json',h)
  with self.assertRaisesRegex(RuntimeError,'already evaluated'):pipeline.ready(self.root,self.out)
 def test_handoff_failure_is_not_silently_waited_through(self):
  atomic(self.out/'handoff_status.json',{'status':'failed','error':'synthetic'})
  with self.assertRaisesRegex(RuntimeError,'handoff failed'):pipeline.ready(self.root,self.out)

if __name__=='__main__':unittest.main(verbosity=2)
