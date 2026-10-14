#!/usr/bin/env python3
"""CPU-only continuation gates; never signal/train/load real weights or acquire GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import continue_robotwin_multitask_evaluation as c

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'run';self.root.mkdir()
        self.original=self.root/'pipeline_manifest.json';c.atomic(self.original,{'source_sha256':{}})
        self.completed={'status':'failed','exit_code':1,'source_changes':[]}
        self.training={'status':'completed_fixed_budget','step':20000,'schedule_budget':20000,'heldout_used':False}
        self.error={'status':'infrastructure_error','phase':'preflight','error':"KeyError: 'episodes'",'scored_episodes':0,'simulator_started':False}
        self.write_ready()
        self.ckpt=self.root/'train/checkpoint_step_020000.pt';self.ckpt.write_bytes(b'CPU fixture, not weights')
        c.atomic(str(self.ckpt)+'.sha256.json',{'step':20000,'model_only':True,'size':self.ckpt.stat().st_size,'sha256':c.sha(self.ckpt)})
        self.source=self.base/'unchanged.py';self.source.write_text('x=1\n')
        self.manifest={'run_root':str(self.root),'original_manifest':str(self.original),'original_unit':'fake-never-systemctl',
          'source_sha256':{str(self.source):c.sha(self.source)},'wait_timeout_seconds':60,'eval_timeout_seconds':60,
          'gpu_lock_owner':'fake-cpu','python':'NEVER_RUN','evaluator':'NEVER_RUN'}
        self.path=self.root/'continuation_manifest.json';c.atomic(self.path,self.manifest)
        self.names=['put_bottles_dustbin','place_cans_plasticbox','open_laptop','click_bell','press_stapler']
    def write_ready(self):
        c.atomic(self.root/'completion.json',self.completed)
        c.atomic(self.root/'pipeline_status.json',{'stage':'cpu_eval_preflight'})
        c.atomic(self.root/'train/status.json',self.training)
        c.atomic(self.root/'eval_cpu_preflight/preflight_error.json',self.error)
    def fake_stage(self,name,command,timeout,root,out,manifest,state):
        self.commands.append((name,command))
        if name=='eval':
            c.atomic(root/'eval/summary.json',{'status':'completed','verified_complete':True,'scored_episodes':50,
                'record_video':False,'third_view_camera_created':False,'tasks':{t:{'native_successes':2,'planned_episodes':10} for t in self.names}})
    def execute(self,stage=None):
        self.commands=[];self.lock=Mock();self.lock.owner.return_value=None
        with patch.object(c,'ROOT',self.base),patch.object(c,'gpu_lock',self.lock),patch.object(c,'unit_stopped',return_value=True),\
             patch.object(c,'append_progress'),patch.object(c,'update_doc'),patch.object(c,'stage',side_effect=stage or self.fake_stage),\
             patch.object(c.signal,'signal'):
            return c.run(self.path)
    def test_wait_before_original_completion_and_reject_other_stage(self):
        (self.root/'completion.json').unlink();self.assertIsNone(c.readiness(self.root))
        self.write_ready();c.atomic(self.root/'pipeline_status.json',{'stage':'train'})
        with self.assertRaisesRegex(c.ContinuationError,'registered CPU'):c.readiness(self.root)
    def test_exact_known_metadata_failure_is_required(self):
        for key,value in [('error',"KeyError: 'another'"),('scored_episodes',1),('simulator_started',True)]:
            old=self.error.copy();self.error[key]=value;self.write_ready()
            with self.assertRaises(c.ContinuationError):c.readiness(self.root)
            self.error=old
    def test_no_training_restart_no_budget_change(self):
        for key,value in [('step',19999),('schedule_budget',50000),('heldout_used',True),('status','running')]:
            old=self.training.copy();self.training[key]=value;self.write_ready()
            with self.assertRaises(c.ContinuationError):c.readiness(self.root)
            self.training=old
    def test_original_source_error_or_prior_eval_never_retried(self):
        self.completed['source_changes']=['bad'];self.write_ready()
        with self.assertRaises(c.ContinuationError):c.readiness(self.root)
        self.completed['source_changes']=[];self.write_ready();(self.root/'eval').mkdir()
        with self.assertRaisesRegex(c.ContinuationError,'already exists'):c.readiness(self.root)
    def test_full_handoff_uses_only_original_final_and_preserves_old_receipts(self):
        old=(self.root/'completion.json').read_bytes();weights=self.ckpt.read_bytes()
        self.assertEqual(self.execute(),0)
        self.assertEqual([x[0] for x in self.commands],['compat_cpu_preflight','eval'])
        for _,cmd in self.commands:
            self.assertEqual(cmd[cmd.index('--checkpoint')+1],str(self.ckpt))
            self.assertNotIn('--resume',cmd);self.assertNotIn('--record-video',cmd)
        self.assertEqual((self.root/'completion.json').read_bytes(),old);self.assertEqual(self.ckpt.read_bytes(),weights)
        done=c.read(self.root/'evaluation_continuation/completion.json')
        self.assertTrue(done['verified_complete']);self.assertFalse(done['training_restarted'])
        self.lock.acquire.assert_called_once();self.lock.release.assert_called_once()
    def test_corrupt_final_checkpoint_fails_before_gpu_acquisition(self):
        self.ckpt.write_bytes(b'changed')
        self.assertEqual(self.execute(),1);self.lock.acquire.assert_not_called()
        self.assertFalse(c.read(self.root/'evaluation_continuation/completion.json')['verified_complete'])
    def test_changed_source_stops_before_any_stage(self):
        self.source.write_text('changed')
        self.assertEqual(self.execute(),1);self.assertEqual(self.commands,[]);self.lock.acquire.assert_not_called()
    def test_stage_failure_no_retry_and_lock_released(self):
        def failed(*args):raise RuntimeError('synthetic stage failure')
        self.assertEqual(self.execute(failed),1);self.lock.release.assert_called_once()
        self.assertFalse(c.read(self.root/'experiment_status.json')['verified_complete'])
    def test_existing_continuation_refuses_overwrite(self):
        p=self.root/'evaluation_continuation';p.mkdir();(p/'status.json').write_text('old')
        with self.assertRaises(FileExistsError):self.execute()
        self.assertEqual((p/'status.json').read_text(),'old')

if __name__=='__main__':unittest.main(verbosity=2)
