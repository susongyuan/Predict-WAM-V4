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
from scripts import collect_isaac_zhuhai_parallel as c
from scripts import run_zhuhai_pipeline_parallel as p
from scripts.test_zhuhai_200_handoff import QuotaTests
from scripts.test_zhuhai_pipeline import fixture_row, FakeCollectors


class ShardTests(unittest.TestCase):
    def setUp(self):
        fixture=QuotaTests('test_original_source_keeps_old_quota_and_hash');fixture.setUp()
        self.addCleanup(fixture.tmp.cleanup);self.fixture=fixture;self.root=fixture.root
        self.contract=deepcopy(fixture.contract)
        self.contract.update(seed_start=10000,resource_policy=dict(c.RESOURCE_POLICY),
            collection_implementation=dict(path=str(Path(c.__file__).resolve()),sha256=c.sha256(c.__file__)),
            collection_sharding=dict(shard_count=3,family_assignment='source seeds ascending; index modulo shard_count',
                global_randomized_count=200,randomized_per_family=2,disjoint_seed_stride=1000))
        self.data_root=self.root/'parallel_fixture_data';self.data_root.mkdir()
        self.components=[]
        for index in range(3):
            parents=[1000+j for j in range(100) if j%3==index]
            rows=[fixture_row(self.data_root,self.contract,10000+index*1000+j,parent=parents[j%len(parents)]) for j in range(2*len(parents))]
            path=self.root/f'shard_{index}.json'
            c.write_once(path,dict(contract=self.contract,requested_success_count=len(rows),status='complete',
                success_count=len(rows),attempts=rows))
            self.components.append(path)

    def test_uneven_three_shards_exact_200_preserve_source_and_rows(self):
        before=self.fixture.source.read_bytes()
        counts=[len(c.validate_manifest(x,task='zhuhai_task1',mode='randomized',shard_index=j,shard_count=3)) for j,x in enumerate(self.components)]
        self.assertEqual(counts,[68,66,66])
        target=self.root/'aggregate.json';c.aggregate_shards(target,self.components,task='zhuhai_task1')
        self.assertEqual(len(c.validate_manifest(target,task='zhuhai_task1',mode='randomized')),200)
        self.assertEqual(self.fixture.source.read_bytes(),before)
        self.assertEqual(json.loads(target.read_text())['attempts'],sorted([r for x in self.components for r in json.loads(x.read_text())['attempts']],key=lambda r:r['seed']))
        # Idempotent publication preserves immutable index bytes.
        original=target.read_bytes();c.aggregate_shards(target,self.components,task='zhuhai_task1');self.assertEqual(target.read_bytes(),original)

    def test_wrong_family_shard_rejected(self):
        data=json.loads(self.components[0].read_text());row=data['attempts'][0]
        row['family_id']='life:zhuhai_task1:source:1001';row['source_parent']=1001
        row['metadata'].update(family_id=row['family_id'],parent_source_seed=1001)
        self.components[0].write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'assigned source family'):
            c.validate_manifest(self.components[0],task='zhuhai_task1',mode='randomized',shard_index=0,shard_count=3)

    def test_four_shards_each_50_complete_exact_quota(self):
        contract=deepcopy(self.contract);contract['collection_sharding']['shard_count']=4
        directory=self.root/'four_shard_data';directory.mkdir();components=[]
        for index in range(4):
            parents=[1000+j for j in range(100) if j%4==index]
            rows=[fixture_row(directory,contract,10000+index*1000+j,parent=parents[j%25]) for j in range(50)]
            path=directory/f'shard_{index}.json'
            c.write_once(path,dict(contract=contract,requested_success_count=50,status='complete',success_count=50,attempts=rows))
            components.append(path)
        target=c.aggregate_shards(directory/'aggregate.json',components,task='zhuhai_task1')
        self.assertEqual(len(c.validate_manifest(target,task='zhuhai_task1',mode='randomized')),200)

    def test_incomplete_shard_never_publishes_aggregate(self):
        data=json.loads(self.components[0].read_text());data['attempts'].pop();data['success_count']-=1
        self.components[0].write_text(json.dumps(data));target=self.root/'aggregate.json'
        with self.assertRaises(ValueError):c.aggregate_shards(target,self.components,task='zhuhai_task1')
        self.assertFalse(target.exists())

    def test_changed_component_rejected_even_with_same_success_count(self):
        target=self.root/'aggregate.json';c.aggregate_shards(target,self.components,task='zhuhai_task1')
        self.components[0].write_bytes(self.components[0].read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Shard manifest changed'):
            c.validate_manifest(target,task='zhuhai_task1',mode='randomized')


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.out=self.root/'run';self.out.mkdir()
        self.args=SimpleNamespace(collection_workers=3,resume=False,gpu_lock_owner='CPU-fixture',source_root=self.root/'source')
        self.jobs=[]
        for i in (1,2):
            for index in range(3):
                task=f'zhuhai_task{i}';target=self.out/task/'randomized/shards'/f'shard_{index:02d}'
                cmd=['CPU_FAKE','--task',task,'--mode','randomized','--count','200','--shard-index',str(index),
                     '--shard-count','3','--seed-start',str(933000+i*10000+1000),'--out',str(target),
                     '--source-manifest',str(self.args.source_root/task/'source/manifest.json'),'--gpu-lock-owner','CPU-fixture']
                receipt=dict(successful_episodes=68 if index==0 else 66,source_families=34 if index==0 else 33)
                self.jobs.append(dict(name=f'randomized_task{i}_shard{index}',task=task,shard_index=index,command=cmd,
                    manifest=target/'manifest.json',validate=lambda receipt=receipt:receipt))

    def run_scheduler(self, fallback=False):
        system=FakeCollectors({f'randomized_task{i}':[dict(duration=60.,code=0,ready_after=None) for _ in range(3)] for i in (1,2)})
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
            stack.enter_context(patch.object(p.subprocess,'Popen',system.spawn))
            stack.enter_context(patch.object(p.time,'monotonic',lambda:system.now))
            stack.enter_context(patch.object(p.time,'sleep',system.sleep))
            stack.enter_context(patch.object(p.os,'killpg',system.killpg))
            p.run_collection_group(self.out,'randomized',self.jobs,self.args,state,plan)
        return system,state

    def test_three_workers_all_six_shards_complete(self):
        system,state=self.run_scheduler()
        self.assertEqual(state['collection_groups']['randomized']['peak_active_workers'],3)
        self.assertEqual(len(system.launches),6);self.assertEqual(system.signals,[])
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
        for i in (1,2):
            for index in range(4):
                task=f'zhuhai_task{i}';target=self.out/task/'randomized/shards'/f'shard_{index:02d}'
                cmd=['CPU_FAKE','--task',task,'--mode','randomized','--count','200','--shard-index',str(index),
                    '--shard-count','4','--seed-start',str(933000+i*10000+1000),'--out',str(target),
                    '--source-manifest',str(self.args.source_root/task/'source/manifest.json'),'--gpu-lock-owner','CPU-fixture']
                self.jobs.append(dict(name=f'randomized_task{i}_shard{index}',task=task,shard_index=index,command=cmd,
                    manifest=target/'manifest.json',validate=lambda:dict(successful_episodes=50,source_families=25)))
        system=FakeCollectors({f'randomized_task{i}':[dict(duration=90.,code=0,ready_after=None) for _ in range(4)] for i in (1,2)})
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

    def test_wrong_parent_and_duplicate_shard_rejected_before_spawn(self):
        jobs=deepcopy(self.jobs);jobs[0]['command'][jobs[0]['command'].index('--source-manifest')+1]=str(self.root/'other.json')
        with self.assertRaisesRegex(ValueError,'isolation'):p._collection_jobs(self.out,'randomized',jobs,self.args)
        jobs=deepcopy(self.jobs);jobs[0]['shard_index']=1
        with self.assertRaisesRegex(ValueError,'disjoint'):p._collection_jobs(self.out,'randomized',jobs,self.args)

if __name__=='__main__':unittest.main()
