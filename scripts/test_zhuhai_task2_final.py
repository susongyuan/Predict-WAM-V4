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
from scripts import run_zhuhai_task2_final as p
from scripts.test_zhuhai_200_handoff import QuotaTests
from scripts.test_zhuhai_pipeline import fixture_row, FakeCollectors


class ShardTests(unittest.TestCase):
    def setUp(self):
        fixture=QuotaTests('test_original_source_keeps_old_quota_and_hash');fixture.setUp()
        self.addCleanup(fixture.tmp.cleanup);self.fixture=fixture;self.root=fixture.root
        self.contract=deepcopy(fixture.contract)
        self.contract.update(requested_success_count=100,production_quota=dict(c.PRODUCTION_QUOTAS,total_per_task=200,total_all_tasks=400),
            source_quota_transition=dict(from_quota=c.source_protocol.PRODUCTION_QUOTAS,to_quota=c.PRODUCTION_QUOTAS,source_relabelled=False),
            seed_start=10000,resource_policy=dict(c.RESOURCE_POLICY),worker_recycling=dict(c.WORKER_RECYCLING),
            collection_implementation=dict(path=str(Path(c.__file__).resolve()),sha256=c.sha256(c.__file__)),
            collection_sharding=dict(shard_count=3,family_assignment='source seeds ascending; index modulo shard_count',
                global_randomized_count=100,randomized_per_family=1,disjoint_seed_stride=1000))
        self.data_root=self.root/'parallel_fixture_data';self.data_root.mkdir()
        self.components=[]
        for index in range(3):
            parents=[1000+j for j in range(100) if j%3==index]
            rows=[fixture_row(self.data_root,self.contract,10000+index*1000+j,parent=parents[j%len(parents)]) for j in range(len(parents))]
            path=self.root/f'shard_{index}.json'
            c.write_once(path,dict(contract=self.contract,requested_success_count=len(rows),status='complete',
                success_count=len(rows),attempts=rows))
            self.components.append(path)

    def test_uneven_three_shards_exact_100_preserve_source_and_rows(self):
        before=self.fixture.source.read_bytes()
        counts=[len(c.validate_manifest(x,task='zhuhai_task1',mode='randomized',shard_index=j,shard_count=3)) for j,x in enumerate(self.components)]
        self.assertEqual(counts,[34,33,33])
        target=self.root/'aggregate.json';c.aggregate_shards(target,self.components,task='zhuhai_task1')
        self.assertEqual(len(c.validate_manifest(target,task='zhuhai_task1',mode='randomized')),100)
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

    def test_training_admits_100_plus_100_and_keeps_family_splits(self):
        from harness.zhuhai_training import task_sources
        target=c.aggregate_shards(self.root/'aggregate.json',self.components,task='zhuhai_task1')
        tasks=self.root/'tasks.json'
        tasks.write_text(json.dumps(dict(tasks=[dict(task='zhuhai_task1',source_manifest=str(self.fixture.source),randomized_manifest=str(target))])))
        from harness import zhuhai_training as zt
        actual_sha=zt.sha256_file
        with patch.object(zt,'sha256_file',side_effect=lambda path:self.contract['source_scene']['sha256'] if str(path).endswith('/cpu-only') else actual_sha(path)):
            splits,sources=task_sources(tasks,source_count=100,randomized_count=100)
        self.assertEqual({key:len(rows) for key,rows in splits['zhuhai_task1'].items()},dict(train=160,val=20,heldout=20))
        families={key:{row['family_id'] for row in rows} for key,rows in splits['zhuhai_task1'].items()}
        self.assertFalse(families['train'] & families['val'] or families['train'] & families['heldout'] or families['val'] & families['heldout'])
        self.assertEqual(sources['zhuhai_task1']['quotas'],dict(source=100,randomized=100))

    def test_four_shards_each_25_complete_exact_quota(self):
        contract=deepcopy(self.contract);contract['collection_sharding']['shard_count']=4
        directory=self.root/'four_shard_data';directory.mkdir();components=[]
        for index in range(4):
            parents=[1000+j for j in range(100) if j%4==index]
            rows=[fixture_row(directory,contract,10000+index*1000+j,parent=parents[j%25]) for j in range(25)]
            path=directory/f'shard_{index}.json'
            c.write_once(path,dict(contract=contract,requested_success_count=25,status='complete',success_count=25,attempts=rows))
            components.append(path)
        target=c.aggregate_shards(directory/'aggregate.json',components,task='zhuhai_task1')
        self.assertEqual(len(c.validate_manifest(target,task='zhuhai_task1',mode='randomized')),100)

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
        self.args=SimpleNamespace(collection_workers=3,resume=False,gpu_lock_owner='CPU-fixture',source_root=self.root/'source',collection_root=self.out)
        self.jobs=[]
        for i in (2,):
            for index in range(3):
                task=f'zhuhai_task{i}';target=self.out/task/'randomized/shards'/f'shard_{index:02d}'
                cmd=['CPU_FAKE','--task',task,'--mode','randomized','--count','100','--shard-index',str(index),
                     '--shard-count','3','--seed-start',str(933000+i*10000+1000),'--out',str(target),
                     '--source-manifest',str(self.args.source_root/task/'source/manifest.json'),'--gpu-lock-owner','CPU-fixture']
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
                    '--shard-count','4','--seed-start',str(933000+i*10000+1000),'--out',str(target),
                    '--source-manifest',str(self.args.source_root/task/'source/manifest.json'),'--gpu-lock-owner','CPU-fixture']
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


class MilestoneTests(unittest.TestCase):
    def test_continuous_20k_schedule_with_one_final_test_and_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary); calls=[]
            args=SimpleNamespace(out=out,resume=False,train_steps=20000,train_batch_size=4,
                gpu_lock_owner='CPU-fixture',evaluation_max_steps=2400)
            plan=dict(training_milestones=[20000],evaluation_seeds=[932001,932002],milestone_protocol={},training_tasks=['zhuhai_task2'])
            def stage(out,name,command,args,state,validate=None):
                calls.append((name,command))
                if name.startswith('evaluation_'):
                    destination=Path(command[command.index('--out')+1]);destination.mkdir(parents=True)
                    episodes=[dict(task=task,seed=seed,full_task_success=seed==932001) for task in ('zhuhai_task2',) for seed in plan['evaluation_seeds']]
                    (destination/'summary.json').write_text(json.dumps(dict(episodes=episodes)))
            with patch.object(p,'run_stage',stage),patch.object(p,'verify_plan_sources'),patch.object(p,'log'):
                results=p.run_training_milestones(args,{},plan,['WAM','trainer','--tasks','fixture'],'ISAAC','evaluator')
            self.assertEqual([name for name,_ in calls],[name for step in plan['training_milestones'] for name in (f'training_{step:06d}',f'evaluation_{step:06d}')])
            training=calls[::2];evaluations=calls[1::2]
            for index,(_,command) in enumerate(training):
                self.assertEqual(command[command.index('--steps')+1],'20000')
                self.assertEqual(command[command.index('--max-updates')+1],str(plan['training_milestones'][index]))
                self.assertEqual(command[command.index('--save-every')+1],'5000')
                self.assertEqual('--resume' in command,index>0)
            self.assertEqual(len({tuple(cmd[cmd.index('--seeds')+1:]) for _,cmd in evaluations}),1)
            self.assertEqual(len({cmd[cmd.index('--checkpoint')+1] for _,cmd in evaluations}),1)
            self.assertTrue(all(cmd[cmd.index('--task')+1]=='zhuhai_task2' for _,cmd in evaluations))
            self.assertTrue(all(row['by_task']['zhuhai_task2']==dict(completed_episodes=2,full_successes=1) for row in results))
            self.assertEqual(len(json.loads((out/'milestone_results.json').read_text())['milestones']),1)

    def test_partial_eval_is_protected_and_completed_boundary_skips(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);target=out/'evaluations/step_005000';target.mkdir(parents=True)
            (target/'partial.json').write_text('{}')
            state=dict(stages={'evaluation_005000':dict(status='failed',command=['test'])})
            with self.assertRaisesRegex(RuntimeError,'retained unchanged'):
                p.stage_needs_run(out,'evaluation_005000',['test'],SimpleNamespace(resume=True),state)


class Reuse100RequestTests(unittest.TestCase):
    def test_exact_quota_and_only_task2_required(self):
        from scripts.train_zhuhai_task2_reuse import validate_request
        with tempfile.TemporaryDirectory() as temporary:
            tasks=Path(temporary)/'tasks.json'
            tasks.write_text(json.dumps(dict(tasks=[dict(task=f'zhuhai_task{i}',source_manifest='source',randomized_manifest='randomized') for i in (2,)])))
            args=SimpleNamespace(tasks=str(tasks),source_count=100,randomized_count=100,allow_small_data=False,mode='train',execute_independent=True)
            validate_request(args)
            tasks.write_text(json.dumps(dict(tasks=[dict(task=t,source_manifest='s',randomized_manifest='r') for t in ('zhuhai_task1','zhuhai_task2')])))
            with self.assertRaises(ValueError):validate_request(args)
            args.randomized_count=200
            with self.assertRaises(ValueError):validate_request(args)

    def test_garbage_only_cannot_enter_training(self):
        from scripts.train_zhuhai_task2_reuse import validate_request
        with tempfile.TemporaryDirectory() as temporary:
            tasks=Path(temporary)/'tasks.json';tasks.write_text(json.dumps(dict(tasks=[dict(task='zhuhai_task1',source_manifest='s',randomized_manifest='r')])))
            args=SimpleNamespace(tasks=str(tasks),source_count=100,randomized_count=100,allow_small_data=False,mode='train',execute_independent=True)
            with self.assertRaises(ValueError):validate_request(args)

    def test_small_data_quota_bypass_is_rejected(self):
        from scripts.train_zhuhai_task2_reuse import validate_request
        with self.assertRaises(ValueError):validate_request(SimpleNamespace(allow_small_data=True,source_count=100,randomized_count=100))

class RecyclingBoundaryTests(ShardTests):
    def test_recycle_only_between_small_batches_before_quota(self):
        self.assertFalse(c.recycle_due(2,2,25))
        self.assertTrue(c.recycle_due(3,3,25))
        self.assertFalse(c.recycle_due(3,25,25))

    def partial(self):
        path=self.components[0];data=json.loads(path.read_text())
        data.update(status='worker_recycle_boundary',attempts=data['attempts'][:3],success_count=3)
        path.write_text(json.dumps(data));return path,data

    def test_partial_boundary_cannot_be_training_admitted(self):
        path,data=self.partial()
        result=c.validate_recycle_boundary(path,task='zhuhai_task1')
        self.assertEqual(result['successful_episodes'],3);self.assertFalse(result['quota_complete'])
        with self.assertRaises(ValueError):c.validate_manifest(path,task='zhuhai_task1',mode='randomized',shard_index=0,shard_count=3)

    def test_running_attempt_rejected(self):
        path,data=self.partial();data['attempts'].append(dict(status='running',seed=123))
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'between physical attempts'):c.validate_recycle_boundary(path,task='zhuhai_task1')

    def test_changed_success_file_rejected(self):
        path,data=self.partial();Path(data['attempts'][0]['data_path']).write_bytes(b'corrupted fixture')
        with self.assertRaisesRegex(ValueError,'changed'):c.validate_recycle_boundary(path,task='zhuhai_task1')

    def test_failure_streak_is_preserved_across_boundary(self):
        path,data=self.partial();data['attempts'].extend([dict(status='failed',seed=101),dict(status='failed',seed=102)])
        path.write_text(json.dumps(data));c.validate_recycle_boundary(path,task='zhuhai_task1')
        self.assertEqual(c.consecutive_failures(json.loads(path.read_text())['attempts']),2)
        data['attempts'].append(dict(status='failed',seed=103));self.assertEqual(c.consecutive_failures(data['attempts']),3)

if __name__=='__main__':unittest.main()
