#!/usr/bin/env python3
"""Fixed Precision queue contracts; fake receipts only, no subprocess/GPU execution."""
import copy
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_zhuhai_wam_precision_pipeline as p


class PipelineTests(unittest.TestCase):
    def plan(self,directory):
        p.configure();args=p.base.parser().parse_args(['--out',str(ROOT/'outputs/precision_cpu_unused'),'--gpu-lock-owner','cpu-test-owner'])
        plan=p.build_plan(args);plan['out']=str(directory)
        for stage in plan['stages']:stage['out']=str(directory/Path(stage['out']).name)
        return plan
    def fake(self,stage):
        out=Path(stage['out']);out.mkdir(parents=True,exist_ok=True)
        if stage['kind']=='train':
            p.precision.write_json(out/'status.json',dict(status='completed' if stage['step'] in (2,30000) else 'stopped_at_safe_boundary',step=stage['step'],target_steps=2 if stage['mode']=='tiny' else 30000,auxiliary_gradient_admission_passed=stage['mode']=='tiny',precision_gradient_admission_passed=stage['mode']=='tiny'))
            ck=out/f"checkpoint_step_{stage['step']:06d}.pt";ck.write_bytes(b'cpu-test-receipt')
            p.precision.write_json(ck.with_name(ck.name+'.sha256.json'),dict(step=stage['step'],model_only=True,size=ck.stat().st_size,sha256=p.precision.sha256_file(ck)))
        else:
            p.precision.write_json(out/'summary.json',dict(status='completed',checkpoint_step=stage['step'],planned_episodes=len(stage['seeds']),episodes=[dict(task=p.precision.TASK,seed=s,status='completed',full_task_success=False) for s in stage['seeds']]))
        return 0
    def test_schedule_precision_scripts_fixed_seeds_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            plan=self.plan(Path(d));stages=plan['stages']
            self.assertEqual(plan['schema'],'zhuhai-precision-wam-task2-30k-pipeline-v1')
            self.assertEqual([s['step'] for s in stages],[2,20000,20000,25000,25000,30000,30000,30000])
            self.assertFalse(plan['record_video']);self.assertFalse(plan['third_view']);self.assertFalse(plan['collection_parallel'])
            for stage in stages:
                self.assertIn('precision.py',stage['command'][2])
                if stage['kind']=='train':
                    self.assertEqual(stage['command'][stage['command'].index('--steps')+1],'30000')
                    self.assertEqual('--resume' in stage['command'],stage['step'] in (25000,30000))
                    self.assertEqual(stage['command'][stage['command'].index('--cache-root')+1],str(ROOT/'data/zhuhai_hjepa_hybrid_features_20261009'))
                elif stage['name'].startswith('development'):
                    self.assertEqual(stage['seeds'],list(range(932101,932106)))
            self.assertEqual(stages[-1]['seeds'],list(range(932001,932011)))
            self.assertFalse(set(plan['development_seeds']) & set(plan['final_seeds']))
            self.assertTrue(all(str(ROOT/name) in plan['source_hashes'] for name in p.precision.EXTENSION_SOURCES))
    def test_all_policy_failures_retained_and_duplicate_writer_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p.precision,'verify_owner'),patch.object(p.base,'log'):
            plan=self.plan(Path(d));calls=[]
            def run(stage):calls.append(stage['name']);return self.fake(stage)
            self.assertEqual(p.base.execute(plan,run),0)
            self.assertEqual(calls,[s['name'] for s in plan['stages']])
            with self.assertRaises(FileExistsError):p.base.execute(plan,run)
            self.assertEqual(p.precision.read_json(Path(d)/'status.json')['status'],'completed')
    def test_infrastructure_failure_stops_following_stages(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p.precision,'verify_owner'),patch.object(p.base,'log'):
            plan=self.plan(Path(d));calls=[]
            def run(stage):calls.append(stage['name']);return 2
            with self.assertRaises(RuntimeError):p.base.execute(plan,run)
            self.assertEqual(calls,['tiny_admission'])
            self.assertEqual(p.precision.read_json(Path(d)/'status.json')['status'],'failed')
    def test_wrong_seed_checkpoint_step_and_nonboolean_score_fail(self):
        with tempfile.TemporaryDirectory() as d:
            stage=self.plan(Path(d))['stages'][2];self.fake(stage)
            path=Path(stage['out'])/'summary.json';original=p.precision.read_json(path)
            for change in ('seed','score','step'):
                value=copy.deepcopy(original)
                if change=='seed':value['episodes'][0]['seed']+=1
                elif change=='score':value['episodes'][0]['full_task_success']=1
                else:value['checkpoint_step']-=1
                p.precision.write_json(path,value)
                with self.assertRaises(ValueError):p.base.verify_result(stage)
    def test_plan_default_never_spawns_processes(self):
        import io
        with patch.object(p.base.subprocess,'Popen',side_effect=AssertionError('plan must not execute')),patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(p.main(['--out',str(ROOT/'outputs/precision_cpu_unused')]),0)
    def test_precision_gradient_admission_and_milestone_hash_required(self):
        with tempfile.TemporaryDirectory() as d:
            stage=self.plan(Path(d))['stages'][0];self.fake(stage)
            status=Path(stage['out'])/'status.json';value=p.precision.read_json(status)
            value['precision_gradient_admission_passed']=False;p.precision.write_json(status,value)
            with self.assertRaisesRegex(ValueError,'dense-refiner'):p.verify_result(stage)
            value['precision_gradient_admission_passed']=True;p.precision.write_json(status,value)
            checkpoint=Path(stage['out'])/'checkpoint_step_000002.pt';checkpoint.write_bytes(b'altered-contents!')
            with self.assertRaises(ValueError):p.verify_result(stage)


if __name__=='__main__':unittest.main(verbosity=2)
