"""CPU fixtures for explicit continuation only; no real GPU, checkpoints or log writes."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_zhuhai_wam_precision_v2_continuation as q


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.origin=self.root/'origin';self.out=self.root/'continuation';self.audit=self.root/'audit';self.audit.mkdir()
        self.train=self.origin/'train';self.train.mkdir(parents=True)
        self.source=self.root/'runtime.py';self.source.write_text('# frozen fixture\n')
        self.eval=self.root/'eval.py';self.eval.write_text('# fixed attempt fixture\n')
        self.profile=self.root/'prior_launch.json';lib=self.root/'libstdcpp';lib.write_bytes(b'old used fixture library')
        self.put(self.profile,{'plan':{'child_env':{'OMNI_KIT_ACCEPT_EULA':'YES','LD_PRELOAD':str(lib)}}})
        stages=[dict(name='tiny_admission',kind='train',step=2),dict(name='train_to_20000',kind='train',step=20000)]
        for step in(20000,25000,30000):
            if step>20000:stages.append(dict(name=f'train_to_{step}',kind='train',mode='train',step=step,out=str(self.train),command=['python','train.py','--out',str(self.train),'--steps','30000','--batch-size','4','--gpu-lock-owner','old','--max-updates',str(step),'--resume']))
            stages.append(dict(name=f'development_{step}',kind='eval',step=step,seeds=q.SEEDS,out=str(self.origin/f'development_{step}'),command=['old-eval']))
        self.old=dict(schema='zhuhai-precision-wam-task2-30k-pipeline-v2',source_hashes={str(self.source):q.h.sha256_file(self.source)},stages=stages,checkpoint_selection='fixed30k')
        self.put(self.origin/'plan.json',self.old)
        self.put(self.origin/'status.json',dict(status='failed',active_stage='development_20000',completed_stages=['tiny_admission','train_to_20000']))
        self.put(self.train/'status.json',dict(status='stopped_at_safe_boundary',step=20000,target_steps=30000))
        self.bad=self.origin/'development_20000/zhuhai_task2/seed_932101/result.json'
        self.put(self.bad,dict(control_steps=0,full_task_success=None,error='SystemExit: EULA not accepted'))
        last=self.train/'checkpoint_last.pt';last.write_bytes(b'fixture20k')
        receipt=dict(step=20000,model_only=False,size=last.stat().st_size,sha256=q.h.sha256_file(last))
        self.put(last.with_name(last.name+'.sha256.json'),receipt)
        self.put(self.audit/'checkpoint_020000_audit.json',dict(status='verified',full_state=receipt,model_only=dict(step=20000)))
        self.put(self.audit/'bootstrap_recovery_authorization.json',dict(control_steps=0))
        self.put(self.audit/'fixed_attempts_protocol.json',dict(checkpoints=[20000,25000,30000],seeds=q.SEEDS,scoring_source_unchanged=True,postprocessing_source_unchanged=True))
        for name,value in [('ORIGIN',self.origin),('OUT',self.out),('AUDIT',self.audit),('PROFILE',self.profile),('EVAL',self.eval)]:
            p=patch.object(q,name,value);p.start();self.addCleanup(p.stop)
        self.args=q.parser().parse_args([])
    def put(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def plan(self):return q.build_plan(self.args)
    def test_five_stage_schedule_keeps_train_out_and_budget_and_new_owner(self):
        p=self.plan();self.assertEqual([s['step']for s in p['stages']],[20000,25000,25000,30000,30000])
        self.assertEqual(p['training_origin'],str(self.origin));self.assertFalse(p['record_video']);self.assertFalse(p['collection_parallel'])
        for s in p['stages']:
            self.assertEqual(s['command'][s['command'].index('--gpu-lock-owner')+1],q.OWNER)
            if s['kind']=='train':
                self.assertEqual(s['out'],str(self.train));self.assertIn('--resume',s['command']);self.assertEqual(s['command'][s['command'].index('--steps')+1],'30000')
            else:self.assertEqual(s['seeds'],q.SEEDS)
    def test_first_eval_adopts_exact_two_physical_results_never_zero_interpreter_error(self):
        first=self.plan()['stages'][0];self.assertTrue(first['adopt_existing'])
        self.assertEqual(first['command'].count('--adopt-result'),2)
        self.assertIn('seed932102_r2',first['adopted_results'][1]);self.assertIn('bootstrap_recovery',first['adopted_results'][0])
    def test_reject_nonzero_bootstrap_case(self):
        self.put(self.bad,dict(control_steps=1,full_task_success=None,error='EULA not accepted'))
        with self.assertRaisesRegex(ValueError,'zero physical'):self.plan()
    def test_reject_other_error_signature(self):
        self.put(self.bad,dict(control_steps=0,full_task_success=None,error='different'))
        with self.assertRaises(ValueError):self.plan()
    def test_reject_wrong_original_stage(self):
        self.put(self.origin/'status.json',dict(status='failed',active_stage='other',completed_stages=[]))
        with self.assertRaises(ValueError):self.plan()
    def test_reject_unverified_fullstate_or_wrong_training_boundary(self):
        self.put(self.train/'status.json',dict(status='training',step=19999,target_steps=30000))
        with self.assertRaises(ValueError):self.plan()
    def test_reject_resume_bytes_drift(self):
        (self.train/'checkpoint_last.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.plan()
    def test_reject_origin_code_drift(self):
        self.source.write_text('# changed')
        with self.assertRaises(ValueError):self.plan()
    def test_reject_shared_standalone_owner(self):
        self.args.gpu_lock_owner='old'
        with self.assertRaises(ValueError):self.plan()
    def test_default_plan_does_not_execute(self):
        with patch.object(q,'execute',side_effect=AssertionError('not allowed')),patch('sys.stdout',io.StringIO()):self.assertEqual(q.main([]),0)
    def test_successful_attempt_failure_scores_do_not_block_completion_or_repeat_work(self):
        plan=self.plan();calls=[]
        with patch.object(q.gpu_lock,'acquire')as acquire,patch.object(q.gpu_lock,'release')as release,patch.object(q.h,'verify_owner'),patch.object(q,'verify_stage',return_value={'success_rate':0}),patch.object(q,'summarize'),patch.object(q,'log'):
            self.assertEqual(q.execute(plan,lambda p,s:calls.append(s['name'])or 0),0)
            acquire.assert_called_once();release.assert_called_once_with(q.OWNER)
        self.assertEqual(calls,[s['name']for s in plan['stages']]);self.assertEqual(q.h.read_json(self.out/'status.json')['status'],'completed')
    def test_infrastructure_failure_stops_and_releases_without_retry(self):
        plan=self.plan();calls=[]
        with patch.object(q.gpu_lock,'acquire'),patch.object(q.gpu_lock,'release')as release,patch.object(q.h,'verify_owner'),patch.object(q,'verify_stage'),patch.object(q,'summarize'),patch.object(q,'log'):
            with self.assertRaises(RuntimeError):q.execute(plan,lambda p,s:calls.append(s['name'])or 2)
            release.assert_called_once_with(q.OWNER)
        self.assertEqual(calls,['development_20000']);self.assertEqual(q.h.read_json(self.out/'status.json')['status'],'failed')
    def test_gpu_wait_receipt_published_before_acquisition_and_no_release_if_not_acquired(self):
        plan=self.plan()
        def fail(*a,**k):
            self.assertEqual(q.h.read_json(self.out/'status.json')['status'],'waiting_for_exclusive_gpu');raise TimeoutError('fixture')
        with patch.object(q.gpu_lock,'acquire',side_effect=fail),patch.object(q.gpu_lock,'release')as release,patch.object(q,'summarize'):
            with self.assertRaises(TimeoutError):q.execute(plan)
            release.assert_not_called()
    def test_summary_exception_cannot_leave_our_gpu_owner(self):
        plan=self.plan()
        with patch.object(q.gpu_lock,'acquire'),patch.object(q.gpu_lock,'release')as release,patch.object(q.h,'verify_owner'),patch.object(q,'verify_stage'),patch.object(q,'summarize',side_effect=ValueError('fixture summary')),patch.object(q,'log'):
            with self.assertRaises(ValueError):q.execute(plan,lambda p,s:0)
            release.assert_called_once_with(q.OWNER)
    def test_nonempty_controller_rejected_without_gpu_acquire(self):
        plan=self.plan();self.out.mkdir();(self.out/'failure_guard').write_text('keep')
        with patch.object(q.gpu_lock,'acquire')as acquire:
            with self.assertRaises(ValueError):q.execute(plan)
            acquire.assert_not_called()
        self.assertEqual((self.out/'failure_guard').read_text(),'keep')


if __name__=='__main__':unittest.main(verbosity=2)
