"""Synthetic CPU tests; no Isaac, CUDA, or production data mutation."""
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import collect_isaac_zhuhai_recycle as c
from scripts import run_zhuhai_task2_collection_only as p
from scripts.test_zhuhai_200_handoff import QuotaTests
from scripts.test_zhuhai_pipeline import fixture_row, FakeCollectors


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.out=self.root/'run';self.out.mkdir()
        self.args=SimpleNamespace(collection_workers=3,resume=False,gpu_lock_owner='CPU-fixture',source_root=self.root/'source',collection_root=self.out,prior_inventory=self.root/'prior.json')
        self.jobs=[]
        for i in (2,):
            for index in range(3):
                task=f'zhuhai_task{i}';target=self.out/task/'randomized/shards'/f'shard_{index:02d}'
                cmd=['CPU_FAKE','--task',task,'--mode','randomized','--count','100','--shard-index',str(index),
                     '--shard-count','3','--seed-start',str(943000+i*10000+1000),'--out',str(target),
                     '--source-manifest',str(self.args.source_root/task/'source/manifest.json'),'--prior-inventory',str(self.args.prior_inventory),'--gpu-lock-owner','CPU-fixture']
                receipt=dict(successful_episodes=34 if index==0 else 33,source_families=34 if index==0 else 33)
                self.jobs.append(dict(name=f'randomized_task{i}_shard{index}',task=task,shard_index=index,command=cmd,
                    manifest=target/'manifest.json',validate=lambda receipt=receipt:receipt))

    def run_scheduler(self, fallback=False, recycle=False):
        system=FakeCollectors({f'randomized_task{i}':[dict(duration=60.,code=0,ready_after=None) for _ in range(3)] for i in (2,)})
        if recycle: system.scenarios['randomized_task2'][0]['code']=c.RECYCLE_EXIT_CODE
        good=dict(admitted=True,memory=dict(available_bytes=12*p.GIB),gpu_free_bytes=6*p.GIB,swap_fraction=.4)
        triggered=False
        def admission():
            nonlocal triggered
            active=sum(child.poll() is None for child in system.children)
            if fallback and active==3 and not triggered:
                triggered=True;return dict(admitted=False,memory=dict(available_bytes=2*p.GIB),gpu_free_bytes=6*p.GIB,swap_fraction=.4)
            return good
        state=dict(stages={});plan=dict(source_hashes={},robot_binding={'asset_sha256':{}})
        with ExitStack() as stack:
            for name,value in [('log',lambda *a,**k:None),('verify_owner',lambda *a:None),('check_disk',lambda *a:None),
                               ('collection_resource_admission',admission),('_new_initial_evidence',lambda *a:False)]:
                stack.enter_context(patch.object(p,name,value))
            stack.enter_context(patch.object(p,'validate_recycle_boundary',return_value=dict(quota_complete=False,successful_episodes=3)))
            stack.enter_context(patch.object(p.subprocess,'Popen',system.spawn))
            stack.enter_context(patch.object(p.time,'monotonic',lambda:system.now))
            stack.enter_context(patch.object(p.time,'sleep',system.sleep))
            stack.enter_context(patch.object(p.os,'killpg',system.killpg))
            p.run_collection_group(self.out,'randomized',self.jobs,self.args,state,plan)
        return system,state

    def test_planned_recycle_requeues_same_shard_with_resume(self):
        system,state=self.run_scheduler(recycle=True)
        self.assertEqual(len(system.launches),4)
        self.assertEqual(system.signals,[])
        self.assertIn('--resume',system.launches[-1]['command'])
        self.assertEqual(system.launches[-1]['command'][system.launches[-1]['command'].index('--out')+1],system.launches[0]['command'][system.launches[0]['command'].index('--out')+1])
        self.assertTrue(all(row['status']=='completed' for row in state['stages'].values()))
        self.assertTrue(any(row['decision']=='planned_worker_recycle' for row in state['collection_groups']['randomized']['decisions']))

    def test_three_workers_all_three_shards_complete(self):
        system,state=self.run_scheduler()
        self.assertEqual(state['collection_groups']['randomized']['peak_active_workers'],3)
        self.assertEqual(len(system.launches),3);self.assertEqual(system.signals,[])
        self.assertTrue(all(x['status']=='completed' for x in state['stages'].values()))

    def test_remaining_three_not_stopped_by_swap_90percent(self):
        report=dict(ram=dict(total_bytes=32*p.GIB,available_bytes=4*p.GIB,swap_total_bytes=16*p.GIB,swap_used_bytes=.91*16*p.GIB),
            gpu=dict(devices=[dict(total_bytes=16*p.GIB,used_bytes=8*p.GIB)]),disk=dict(free_bytes=300*p.GIB),task_owned_processes=[])
        with patch('scripts.zhuhai_resource_watch.snapshot',return_value=report):
            value=c.resource_snapshot()
        self.assertEqual(value['advice'],'resources_within_declared_reserves')
        self.assertEqual(value['thresholds']['fourth_worker_swap_exit_fraction'],.90)
        self.assertEqual(value['thresholds']['maximum_swap_fraction'],.95)

    def test_four_worker_swap90_fallback_stops_only_newest_and_resumes(self):
        self.args.collection_workers=4;self.jobs=[]
        for i in (2,):
            for index in range(4):
                task=f'zhuhai_task{i}';target=self.out/task/'randomized/shards'/f'shard_{index:02d}'
                cmd=['CPU_FAKE','--task',task,'--mode','randomized','--count','100','--shard-index',str(index),
                    '--shard-count','4','--seed-start',str(943000+i*10000+1000),'--out',str(target),
                    '--source-manifest',str(self.args.source_root/task/'source/manifest.json'),'--prior-inventory',str(self.args.prior_inventory),'--gpu-lock-owner','CPU-fixture']
                self.jobs.append(dict(name=f'randomized_task{i}_shard{index}',task=task,shard_index=index,command=cmd,
                    manifest=target/'manifest.json',validate=lambda:dict(successful_episodes=25,source_families=25)))
        system=FakeCollectors({f'randomized_task{i}':[dict(duration=90.,code=0,ready_after=None) for _ in range(4)] for i in (2,)})
        triggered=False
        def admission():
            nonlocal triggered
            active=sum(x.poll() is None for x in system.children)
            if active==4 and not triggered:
                triggered=True;return dict(admitted=False,memory=dict(available_bytes=4*p.GIB,total_bytes=32*p.GIB),gpu_free_bytes=6*p.GIB,swap_fraction=.90)
            return dict(admitted=True,memory=dict(available_bytes=12*p.GIB,total_bytes=32*p.GIB),gpu_free_bytes=6*p.GIB,swap_fraction=.4)
        state=dict(stages={});plan=dict(source_hashes={},robot_binding={'asset_sha256':{}})
        with ExitStack() as stack:
            for name,value in [('log',lambda *a,**k:None),('verify_owner',lambda *a:None),('check_disk',lambda *a:None),
                ('collection_resource_admission',admission),('_new_initial_evidence',lambda *a:False)]:stack.enter_context(patch.object(p,name,value))
            stack.enter_context(patch.object(p.subprocess,'Popen',system.spawn));stack.enter_context(patch.object(p.time,'monotonic',lambda:system.now))
            stack.enter_context(patch.object(p.time,'sleep',system.sleep));stack.enter_context(patch.object(p.os,'killpg',system.killpg))
            p.run_collection_group(self.out,'randomized',self.jobs,self.args,state,plan)
        self.assertEqual(state['collection_groups']['randomized']['peak_active_workers'],4)
        self.assertEqual(len(system.signals),1);self.assertEqual(system.signals[0][0],system.launches[3]['pid'])
        self.assertTrue(all(x['status']=='completed' for x in state['stages'].values()))

    def test_resource_fallback_preserves_partial_and_resumes_only_owned_shard(self):
        system,state=self.run_scheduler(True)
        self.assertEqual(len(system.signals),1)
        self.assertEqual(system.signals[0][0],system.launches[2]['pid'])
        self.assertTrue(any('--resume' in x['command'] for x in system.launches))
        self.assertTrue(list(self.out.rglob('partial_cpu_evidence_*')))
        self.assertTrue(all(x['status']=='completed' for x in state['stages'].values()))

    def test_collection_import_paths_are_bound_and_originals_preserved(self):
        borrowed=self.root/'preserved_v7';self.args.collection_root=borrowed
        with self.assertRaisesRegex(ValueError,'isolation'):p._collection_jobs(self.out,'randomized',self.jobs,self.args)
        for job in self.jobs:
            destination=borrowed/job['task']/'randomized/shards'/f"shard_{job['shard_index']:02d}"
            job['command'][job['command'].index('--out')+1]=str(destination);job['manifest']=destination/'manifest.json'
        p._collection_jobs(self.out,'randomized',self.jobs,self.args)

    def test_wrong_parent_and_duplicate_shard_rejected_before_spawn(self):
        jobs=deepcopy(self.jobs);jobs[0]['command'][jobs[0]['command'].index('--source-manifest')+1]=str(self.root/'other.json')
        with self.assertRaisesRegex(ValueError,'isolation'):p._collection_jobs(self.out,'randomized',jobs,self.args)
        jobs=deepcopy(self.jobs);jobs[0]['shard_index']=1
        with self.assertRaisesRegex(ValueError,'disjoint'):p._collection_jobs(self.out,'randomized',jobs,self.args)

    def test_prior_inventory_must_be_bound_exactly_before_spawn(self):
        jobs=deepcopy(self.jobs)
        offset=jobs[0]['command'].index('--prior-inventory')
        jobs[0]['command'][offset+1]=str(self.root/'wrong_prior.json')
        with self.assertRaisesRegex(ValueError,'isolation'):
            p._collection_jobs(self.out,'randomized',jobs,self.args)
        del jobs[0]['command'][offset:offset+2]
        with self.assertRaisesRegex(ValueError,'Exactly one --prior-inventory'):
            p._collection_jobs(self.out,'randomized',jobs,self.args)


class CollectionOnlyTests(unittest.TestCase):
    def test_collection_barrier_ends_pipeline_without_training_or_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'run'
            args=p.parser().parse_args(['--out',str(out),'--collection-root',str(out),
                '--prior-inventory',str(root/'prior.json'),'--source-root',str(root/'source'),
                '--deferred-root',str(root/'deferred'),'--life-mapping','body_text',
                '--collection-workers','4','--gpu-lock-owner','CPU-fixture','--execute'])
            plan=dict(scene_ready=True,source_hashes={},robot_binding=dict(asset_sha256={}),
                training_authorized=False,automatic_training=False,automatic_evaluation=False)
            jobs=[]
            def collect(out,mode,items,args,state,plan):
                jobs.extend(items)
                for item in items:
                    state['stages'][item['name']]=dict(status='completed')
            with patch.object(p,'build_plan',return_value=plan),patch.object(p,'log'),patch.object(p,'verify_owner'),\
                    patch.object(p,'snapshot_sources',return_value={}),\
                    patch.object(p,'collection_receipt',return_value=dict(successful_episodes=100,source_families=100)),\
                    patch.object(p,'run_collection_group',side_effect=collect),\
                    patch('scripts.collect_isaac_zhuhai_task2_repair.aggregate_shards'),\
                    patch.object(p.subprocess,'Popen',side_effect=AssertionError('No trainer or evaluator may launch')):
                self.assertEqual(p.pipeline(args),0)
            self.assertEqual(len(jobs),4)
            completion=json.loads((out/'completion.json').read_text())
            state=json.loads((out/'status.json').read_text())
            self.assertEqual(state['status'],'completed_collection_only')
            for value in (completion,state):
                self.assertIs(value['training_authorized'],False)
                self.assertIs(value['automatic_training'],False)
                self.assertIs(value['automatic_evaluation'],False)
            self.assertFalse((out/'training').exists())
            self.assertFalse((out/'tiny_training').exists())
            self.assertFalse((out/'evaluations').exists())

    def test_training_and_evaluation_cli_flags_not_supported(self):
        required=['--out','run','--collection-root','run','--prior-inventory','prior',
            '--source-root','source','--deferred-root','deferred','--life-mapping','body_text']
        from contextlib import redirect_stderr
        import io
        for forbidden in ('--train-steps','--evaluation-count'):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                p.parser().parse_args([*required,forbidden,'1'])


if __name__=='__main__':unittest.main()


