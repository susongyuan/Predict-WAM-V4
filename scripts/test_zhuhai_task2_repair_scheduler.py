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
from scripts import run_zhuhai_task2_repair as p
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


class RepairPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        source=self.root/'original/zhuhai_task2/source/manifest.json';source.parent.mkdir(parents=True)
        source.write_text(json.dumps(dict(contract=dict(engine=dict(name='Isaac Sim',version='6.1.0.0')))))
        prior=self.root/'prior.json';prior.write_text(json.dumps(dict(components=[dict(path=str(self.root/'old/manifest.json'))])))
        self.args=SimpleNamespace(source_root=self.root/'original',collection_root=self.root/'corrected',prior_inventory=prior,collection_workers=4)

    def import_receipt(self, source_count=100, prior_count=13, residual_count=87):
        remaining=[list(range(residual_count//4+(index<residual_count%4))) for index in range(4)]
        def residual(binding,index,count,verify_files=True):
            self.assertEqual(binding['source_manifest']['engine'],dict(name='Isaac Sim',version='6.1.0.0'))
            self.assertEqual(binding['collection_sharding'],dict(shard_count=4,
                family_assignment='source seeds ascending; index modulo shard_count',
                global_randomized_count=100,randomized_per_family=1,disjoint_seed_stride=1000))
            return remaining[index]
        with patch.object(p,'validate_manifest',return_value=[{}]*source_count),patch.object(p,'prior_rows',return_value=[{}]*prior_count),patch.object(p,'residual_families',side_effect=residual):
            return p.validate_prior_import(self.args)

    def test_13_preserved_and_87_residual_are_explicit(self):
        receipt=self.import_receipt()
        self.assertEqual(receipt['successful_episodes'],13)
        self.assertEqual(receipt['new_successful_episode_target'],87)
        self.assertEqual(sum(receipt['residual_shard_targets']),87)

    def test_changed_prior_quota_or_missing_source_rejected(self):
        for source_count,prior_count in [(99,13),(100,12),(100,14)]:
            with self.assertRaisesRegex(ValueError,'100 original sources and 13'):
                self.import_receipt(source_count,prior_count)
        with self.assertRaisesRegex(ValueError,'87 disjoint'):
            self.import_receipt(residual_count=86)

    def test_original_source_output_cannot_be_reused(self):
        self.args.collection_root=self.args.source_root
        with self.assertRaisesRegex(ValueError,'original source evidence'):
            self.import_receipt()

    def test_prior_stopped_shard_output_cannot_be_reused(self):
        prior_shard=self.args.collection_root/'zhuhai_task2/randomized/shards/shard_00/manifest.json'
        self.args.prior_inventory.write_text(json.dumps(dict(components=[dict(path=str(prior_shard))])))
        with self.assertRaisesRegex(ValueError,'prior stopped shards'):
            self.import_receipt()

    def test_default_authorized_final_evaluation(self):
        args=p.parser().parse_args(['--out','run','--collection-root','run','--prior-inventory','prior',
            '--source-root','source','--deferred-root','deferred','--life-mapping','body_text'])
        self.assertEqual((args.train_steps,args.evaluation_count,args.evaluation_max_steps),(20000,10,2400))


class RepairStatusTests(unittest.TestCase):
    def setUp(self):
        from scripts import progress1_log_task2_repair as logger
        self.logger=logger
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.run=self.root/'outputs/repair';self.run.mkdir(parents=True)
        self.prior=self.root/'prior.json';self.prior.write_text(json.dumps(dict(successful_episodes=13)))
        task='zhuhai_task2';self.random=self.run/task/'randomized/manifest.json'
        source=self.root/'source.json';source.write_text(json.dumps(dict(success_count=100)))
        (self.run/'tasks.json').write_text(json.dumps(dict(tasks=[dict(task=task,source_manifest=str(source),randomized_manifest=str(self.random))])))
        (self.run/'plan.json').write_text(json.dumps(dict(prior_DR_import=dict(path=str(self.prior),sha256=p.sha(self.prior),successful_episodes=13))))
        self.text='<!-- active_zhuhai_pipeline_200: outputs/repair -->\n- **生产数据：旧值**\n- **活动流水线：旧值**\n'

    def refresh(self):
        with patch.object(self.logger,'ROOT',self.root):
            return self.logger.refresh_active_run(self.text)

    def leaf(self, number=2, probe=False):
        path=self.random.parent/'shards/shard_00/manifest.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(success_count=number,contract=dict(task='zhuhai_task2',mode='randomized',source_probe=probe))))

    def test_partial_status_counts_13_prior_plus_only_new_successes(self):
        self.leaf()
        (self.run/'status.json').write_text(json.dumps(dict(status='running',active_stage='randomized_collection')))
        text=self.refresh()
        self.assertIn('task2源100/100、随机化15/100',text)
        self.assertIn('running / randomized_collection',text)

    def test_completed_aggregate_not_counted_twice(self):
        self.leaf()
        self.random.write_text(json.dumps(dict(success_count=100,contract=dict(source_probe=False))))
        self.assertIn('task2源100/100、随机化100/100',self.refresh())

    def test_probe_not_counted_as_production(self):
        self.leaf(probe=True)
        self.assertIn('task2源100/100、随机化?/100',self.refresh())

    def test_changed_prior_cannot_claim_preserved_count(self):
        self.leaf();self.prior.write_bytes(self.prior.read_bytes()+b' ')
        self.assertIn('task2源100/100、随机化?/100',self.refresh())


if __name__=='__main__':unittest.main()
