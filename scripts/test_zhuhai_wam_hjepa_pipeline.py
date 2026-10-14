#!/usr/bin/env python3
"""No simulator/model imports: verify fixed schedules, failures and duplicate writes."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
from scripts import run_zhuhai_wam_hjepa_pipeline as p

class PipelineTests(unittest.TestCase):
    def plan(self,directory):
        args=p.parser().parse_args(['--out',str(p.ROOT/'outputs/hjepa_test_unused'),'--gpu-lock-owner','cpu-test-owner'])
        plan=p.build_plan(args);plan['out']=str(directory)
        for stage in plan['stages']:
            stage['out']=str(directory/Path(stage['out']).name)
        return plan
    def fake(self,stage):
        out=Path(stage['out']);out.mkdir(parents=True,exist_ok=True)
        if stage['kind']=='train':
            p.h.write_json(out/'status.json',dict(status='completed' if stage['step'] in (2,30000) else 'stopped_at_safe_boundary',step=stage['step'],target_steps=2 if stage['mode']=='tiny' else 30000,auxiliary_gradient_admission_passed=stage['mode']=='tiny'))
            ck=out/f"checkpoint_step_{stage['step']:06d}.pt";ck.write_bytes(b'cpu-test')
            p.h.write_json(ck.with_name(ck.name+'.sha256.json'),dict(step=stage['step'],model_only=True,size=ck.stat().st_size))
        else:
            p.h.write_json(out/'summary.json',dict(status='completed',checkpoint_step=stage['step'],planned_episodes=len(stage['seeds']),episodes=[dict(task=p.h.TASK,seed=s,status='completed',full_task_success=False) for s in stage['seeds']]))
        return 0
    def test_exact_30k_schedule_and_independent_final(self):
        with tempfile.TemporaryDirectory() as d:
            plan=self.plan(Path(d))
            self.assertEqual(len(plan['stages']),8)
            trains=[s for s in plan['stages'] if s['kind']=='train']
            self.assertEqual([s['step'] for s in trains],[2,20000,25000,30000])
            self.assertEqual(['--resume' in s['command'] for s in trains],[False,False,True,True])
            for s in trains:self.assertEqual(s['command'][s['command'].index('--steps')+1],'30000')
            self.assertFalse(set(plan['development_seeds']) & set(plan['final_seeds']))
            self.assertFalse(plan['record_video']);self.assertFalse(plan['collection_parallel'])
    def test_all_policy_failures_are_kept_and_queue_finishes_without_retry(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p.h,'verify_owner'),patch.object(p,'log'):
            plan=self.plan(Path(d));calls=[]
            def run(stage):calls.append(stage['name']);return self.fake(stage)
            self.assertEqual(p.execute(plan,run),0)
            self.assertEqual(calls,[s['name'] for s in plan['stages']])
            with self.assertRaises(FileExistsError):p.execute(plan,run)
    def test_infrastructure_failure_stops_every_following_stage(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p.h,'verify_owner'),patch.object(p,'log'):
            plan=self.plan(Path(d));calls=[]
            def run(stage):calls.append(stage['name']);return 2
            with self.assertRaises(RuntimeError):p.execute(plan,run)
            self.assertEqual(calls,['tiny_admission'])
            self.assertEqual(p.h.read_json(Path(d)/'status.json')['status'],'failed')
    def test_wrong_step_seed_or_nonboolean_score_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            plan=self.plan(Path(d));s=plan['stages'][2];self.fake(s)
            path=Path(s['out'])/'summary.json';value=p.h.read_json(path);value['episodes'][0]['full_task_success']=1;p.h.write_json(path,value)
            with self.assertRaises(ValueError):p.verify_result(s)
            s=plan['stages'][0];self.fake(s);path=Path(s['out'])/'status.json';value=p.h.read_json(path);value['auxiliary_gradient_admission_passed']=False;p.h.write_json(path,value)
            with self.assertRaises(ValueError):p.verify_result(s)

if __name__=='__main__':unittest.main()
