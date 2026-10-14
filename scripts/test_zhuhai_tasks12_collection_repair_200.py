#!/usr/bin/env python3
"""Meaningful CPU-only DR200 evidence and scheduler tests; never physical results.

Synthetic episodes exercise actual provenance/strict-score/aggregate validators.
Subprocess, GPU ownership, resource readings and simulator are never invoked.
"""
from collections import Counter
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_isaac_zhuhai_tasks12_repair_200 as c
from scripts import run_zhuhai_tasks12_collection_repair_200 as s
from scripts import run_zhuhai_tasks12_collection_repair as old_s
from scripts import collect_isaac_zhuhai_100_parallel as old1
from scripts.test_zhuhai_tasks12_collection_repair_scheduler import CpuEvidence, cli, publish, good_resources
from scripts.test_zhuhai_pipeline import fixture_row, FakeCollectors


def fixture200(root, workers=4):
    base = CpuEvidence(root)  # Unchanged imported task2 full CPU transaction.
    source = base.sources['zhuhai_task1']; sc = old_s.read_json(source)['contract']
    parents = [fixture_row(source.parent, sc, 943000+i) for i in range(100)]
    for i, row in enumerate(parents):
        row['split'] = 'train' if i < 80 else 'val' if i < 90 else 'heldout'
        row['metadata'].update(task='zhuhai_task1', split=row['split'])
    publish(source, dict(contract=sc, requested_success_count=100, success_count=100, status='complete', attempts=parents))
    prior_path, inventory = root/'old1/manifest.json', root/'prior1.json'
    oc = deepcopy(base.contracts['zhuhai_task1'])
    oc.update(source_manifest=dict(**old_s.reference(source), engine=sc['engine']), collection_implementation=old_s.reference(old1.__file__))
    old_rows = []
    for i, parent in enumerate(parents[:48]):
        row = fixture_row(prior_path.parent, oc, 920000+i, parent=parent['seed'])
        row['split'] = parent['split']; row['metadata'].update(task='zhuhai_task1', split=row['split'])
        old_rows.append(row)
    failures = [dict(seed=920100+i, status='failed', failure_reason='CPU preserved original failed attempt') for i in range(3)]
    publish(prior_path, dict(contract=oc, requested_success_count=100, success_count=48,
                            status='stopped_consecutive_failures', attempts=[*old_rows, *failures]))
    publish(inventory, dict(task='zhuhai_task1', old_collector=old_s.reference(old1.__file__), attempts=old_rows,
        successful_episodes=48, components=[dict(**old_s.reference(prior_path), successful_episodes=48)]))
    dr = deepcopy(oc)
    dr.update(requested_success_count=200, max_attempts=2000, seed_start=s.TASK1_SEED_START,
        production_quota=dict(source=100, randomized=200, total_per_task=300, total_all_tasks=500),
        source_quota_transition=dict(from_quota=c.source_protocol.PRODUCTION_QUOTAS, to_quota=c.PRODUCTION_QUOTAS, source_relabelled=False),
        collection_implementation=old_s.reference(c.__file__), inherited_collection_implementation=c.inherited_binding(),
        collection_contract_version=c.CONTRACT_VERSION, expert_variant=old_s.reference(c.TASK_EXPERT_PATHS['zhuhai_task1']),
        expert_dependency_sha256=c.expert_dependency_binding('zhuhai_task1'), collection_sharding=c.sharding(workers),
        prior_inventory=old_s.reference(inventory))
    parts = []
    for index in range(workers):
        directory = root/'new/zhuhai_task1/randomized/shards'/f'shard_{index:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for j, parent in enumerate(c.residual_families(dr, index, workers)):
            row = fixture_row(directory, dr, s.TASK1_SEED_START + index*c.SEED_STRIDE+j, parent=parent['seed'])
            row['split'] = parent['split']; row['metadata'].update(task='zhuhai_task1', split=row['split'])
            rows.append(row)
        path = directory/'manifest.json'
        publish(path, dict(contract=dr, requested_success_count=len(rows), success_count=len(rows), status='complete', attempts=rows))
        parts.append(path)
    # Real frozen strict v2 admission predicates against synthetic evidence.
    original = old_s.read_json(base.probes[0])['attempts'][0]
    by_seed = {r['seed']: r for r in parents}
    candidate, entry = c.TASK_EXPERT_PATHS['zhuhai_task1'], ROOT/'scripts/probe_zhuhai_task1_expert_repair_v2.py'
    pc = deepcopy(oc)
    pc.update(expert_variant=old_s.reference(candidate), collection_implementation=old_s.reference(c.frozen.__file__))
    pc['scene_dependency_sha256'].update({str(p): old_s.sha(p) for p in (candidate, entry, ROOT/'harness/isaac_zhuhai_task1_expert_repair_v1.py')})
    bindings = {**pc['scene_dependency_sha256'], **pc['robot_binding']['asset_sha256'], str(source): old_s.sha(source),
        str(Path(c.frozen.__file__).resolve()): old_s.sha(c.frozen.__file__),
        **{by_seed[seed]['data_path']: by_seed[seed]['data_sha256'] for seed in set(old_s.V2_PROBE_CASES.values())}}
    fingerprint = old_s.json_digest(bindings)
    suite = root/'v2_suite'; suite.mkdir()
    publish(suite/'suite_plan.json', dict(schema=old_s.V2_PROBE_SCOPE,
        required_cases=[dict(seed=a, parent=b) for a,b in old_s.V2_PROBE_CASES.items()], minimum_strict_physical_passes=4,
        candidate_sha256=old_s.sha(candidate), bindings_sha256=fingerprint, bindings=bindings,
        admitted_to_production=False, production_quota_credit=0, record_video=False, third_view=False))
    probes = []
    for seed, parent in old_s.V2_PROBE_CASES.items():
        path = suite/f'seed_{seed}'/'manifest.json'; path.parent.mkdir()
        contract = deepcopy(pc)
        contract.update(source_probe=True, allow_unvalidated_probe=True, collection_scope=old_s.V2_PROBE_SCOPE,
            admitted_to_production=False, production_quota_credit=0, requested_success_count=1, max_attempts=1,
            probe_implementation=old_s.reference(entry), probe_bindings_sha256=fingerprint)
        row = deepcopy(original); row.update(seed=seed, source_parent=parent, family_id=by_seed[parent]['family_id'])
        row['metadata'].update(seed=seed, parent_source_seed=parent, family_id=row['family_id'], contract=contract)
        data = path.parent/'episode.cpu_fixture'; data.write_text(f'CPU synthetic v2 {seed}')
        receipt_path = path.parent/'expert.json'; receipt = old_s.read_json(original['expert_receipt_path'])
        receipt['expert_class'] = 'harness.'+candidate.stem+'.SmoothExpert'; publish(receipt_path, receipt)
        row.update(data_path=str(data), data_sha256=old_s.sha(data), expert_receipt_path=str(receipt_path),
                   expert_receipt_sha256=old_s.sha(receipt_path))
        publish(path, dict(contract=contract, requested_success_count=1, success_count=1, status='complete', attempts=[row]))
        publish(path.parent/'plan.json', dict(contract=contract, seed=seed, source_parent=parent,
            suite_bindings_sha256=fingerprint, admitted_to_production=False, production_quota_credit=0))
        probes.append(path)
    admission = root/'admission.json'
    publish(admission, dict(status='passed', task='zhuhai_task1', collector=old_s.reference(c.frozen.__file__),
        expert=old_s.reference(candidate), probe_manifests=[old_s.reference(p) for p in probes]))
    args = s.parser().parse_args(cli(root, workers))
    return args, dr, parents, old_rows, parts, probes


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dr200-cpu-'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args, self.contract, self.parents, self.prior, self.parts, self.probes = fixture200(self.root)

    def test_two_slots_minus_original48_and_all_shards(self):
        slots = [r for i in range(4) for r in c.residual_families(self.contract, i, 4)]
        count = Counter(r['family_id'] for r in slots)
        self.assertEqual(len(slots), 152)
        self.assertEqual(count[self.parents[0]['family_id']], 1)
        self.assertEqual(count[self.parents[48]['family_id']], 2)
        self.assertEqual([len(c.residual_families(self.contract, i, 4)) for i in range(4)], [38]*4)
        self.assertEqual(c.frozen.PRODUCTION_QUOTAS, dict(source=100, randomized=100))
        self.assertIsNot(c._p.__dict__, c.frozen.__dict__)
        self.assertIs(c._p.collect.__code__, c.frozen.collect.__code__)
        self.assertIs(s.run_collection_group.__code__, old_s.run_collection_group.__code__)
        self.assertFalse(c.parser().parse_args(['--task','zhuhai_task1','--mode','randomized','--count','200',
            '--seed-start','990000','--out',str(self.root/'plan_only'),'--prior-inventory',str(self.args.task1_prior_inventory)]).execute)

    def test_complete200_family_counts_split_and_original_evidence_preserved(self):
        preserved = {p: p.read_bytes() for directory in ('source', 'old1', 'old2', 'v11', 'v2_suite')
                     for p in (self.root/directory).rglob('*') if p.is_file()}
        target = self.root/'new/zhuhai_task1/randomized/manifest.json'
        c.aggregate_shards(target, self.parts, task='zhuhai_task1')
        rows = c.validate_manifest(target, task='zhuhai_task1', mode='randomized')
        result = s.family_integrity(self.parents, rows, randomized_per_family=2)
        self.assertEqual(result['dr_split_counts'], dict(train=160, val=20, heldout=20))
        self.assertEqual(len(rows), 200)
        self.assertTrue(all(Counter(r['family_id'] for r in rows)[p['family_id']] == 2 for p in self.parents))
        self.assertEqual({r['data_path'] for r in self.prior}, {r['data_path'] for r in rows if r['seed'] < s.TASK1_SEED_START})
        self.assertTrue(any(r['status'] == 'failed' for r in old_s.read_json(target)['attempts']))
        self.assertTrue(all(p.read_bytes() == value for p, value in preserved.items()))

    def test_original_unlabeled_sources_keep_original_future_split_rule(self):
        parents = deepcopy(self.parents)
        rows = deepcopy([*self.prior, *[r for p in self.parts for r in old_s.read_json(p)['attempts']]])
        for row in [*parents, *rows]:
            row.pop('split', None); row['metadata'].pop('split', None)
        result = s.family_integrity(parents, rows, randomized_per_family=2)
        self.assertEqual(result['source_split_counts'], {'unassigned':100})
        self.assertEqual(result['planned_training_split']['seed'], 20261008)
        self.assertEqual(result['planned_training_split']['successful_episodes_with_two_DR'], {'train':240,'val':30,'heldout':30})
        self.assertFalse(result['training_split_materialized'])
        rows[0]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'split'):
            s.family_integrity(parents, rows, randomized_per_family=2)

    def test_unequal_prior_quota_cannot_overfill_next_parent(self):
        slots = c.residual_families(self.contract, 0, 4); attempts = []
        while len(attempts) < len(slots):
            parent = c.next_parent(slots, attempts)
            attempts.append(dict(status='success', family_id=parent['family_id']))
        self.assertEqual(Counter(r['family_id'] for r in attempts), Counter(r['family_id'] for r in slots))
        with self.assertRaisesRegex(ValueError, 'No remaining'):
            c.next_parent(slots, attempts)
        attempts.append(attempts[0])
        with self.assertRaisesRegex(ValueError, 'quota exceeded'):
            c.next_parent(slots, attempts)

    def test_source_drift_cross_split_and_family_overfill_rejected(self):
        rows = [*self.prior, *[r for p in self.parts for r in old_s.read_json(p)['attempts']]]
        bad = deepcopy(rows); bad[-1]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'split'):
            s.family_integrity(self.parents, bad, randomized_per_family=2)
        bad = deepcopy(rows); bad[-1]['family_id'] = bad[0]['family_id']; bad[-1]['source_parent'] = bad[0]['source_parent']
        bad[-1]['metadata'].update(family_id=bad[0]['family_id'], parent_source_seed=bad[0]['source_parent'], split='train'); bad[-1]['split']='train'
        with self.assertRaisesRegex(ValueError, 'two DR'):
            s.family_integrity(self.parents, bad, randomized_per_family=2)
        source = Path(self.contract['source_manifest']['path']); source.write_bytes(source.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            c.residual_families(self.contract, 0, 4)

    def test_collector_default_plan_and_execute_without_probe_receipt_refused(self):
        argv = ['collector', '--task','zhuhai_task1','--mode','randomized','--count','200','--seed-start','990000',
            '--out',str(self.root/'default_plan'),'--source-manifest',str(self.args.source_root/'zhuhai_task1/source/manifest.json'),
            '--prior-inventory',str(self.args.task1_prior_inventory), '--shard-count','4']
        verified_slots = c.residual_families(self.contract,0,4)
        safe = deepcopy(self.contract)
        safe['robot_binding']['assets_ready'] = True; safe['runtime_interface'] = {'ready':True}
        with patch.object(sys,'argv',argv), patch.object(c,'contract',return_value=safe), \
                patch.object(c,'residual_families',return_value=verified_slots), \
                patch.object(c,'_guarded_owner',side_effect=AssertionError('Default must not touch GPU owner')), \
                patch.object(c._p,'collect',side_effect=AssertionError('Default must not run physical collector')):
            self.assertEqual(c.main(),0)
        argv[argv.index('--out')+1] = str(self.root/'execute_no_receipt');argv.append('--execute')
        with patch.object(sys,'argv',argv), patch.object(c,'contract',return_value=safe), \
                patch.object(c,'residual_families',return_value=verified_slots), \
                patch.object(c,'_guarded_owner',side_effect=AssertionError('Missing receipt must stop before GPU owner')), \
                patch.object(c._p,'collect',side_effect=AssertionError('Missing receipt must not run collector')):
            with self.assertRaisesRegex(ValueError,'four-v2-probe acceptance'):
                c.main()

    def test_frozen_resource_guard_version_and_quota_never_relaxed(self):
        c.validate_header(self.contract, 'zhuhai_task1')
        for key, value in [('max_consecutive_failures', 4), ('max_attempts', 3000),
                ('record_video', True), ('requested_success_count', 100), ('inherited_collection_implementation', {})]:
            bad = deepcopy(self.contract); bad[key] = value
            with self.assertRaises(ValueError):
                c.validate_header(bad, 'zhuhai_task1')
        self.assertEqual(c.RESOURCE_POLICY, c.frozen.RESOURCE_POLICY)

    def test_four_strict_probes_required_and_old_receipt_not_mutated(self):
        before = self.args.task1_acceptance.read_bytes()
        proof = s.accepted_task1_expert(self.args.task1_acceptance)
        self.assertEqual(proof['seeds'], sorted(old_s.V2_PROBE_CASES))
        self.assertEqual(before, self.args.task1_acceptance.read_bytes())
        value = old_s.read_json(self.args.task1_acceptance); value['probe_manifests'] = value['probe_manifests'][:3]
        publish(self.args.task1_acceptance, value)
        with self.assertRaisesRegex(ValueError, 'four'):
            s.accepted_task1_expert(self.args.task1_acceptance)

    def test_failed_or_boolean_probe_never_admitted(self):
        path = self.probes[0]; value = old_s.read_json(path); value['attempts'][0]['status'] = 'failed'; publish(path, value)
        admission = old_s.read_json(self.args.task1_acceptance); admission['probe_manifests'][0] = old_s.reference(path)
        publish(self.args.task1_acceptance, admission)
        with self.assertRaises(ValueError):
            s.accepted_task1_expert(self.args.task1_acceptance)
        publish(self.args.task1_acceptance, dict(status='passed', task='zhuhai_task1', all_four_strict_passed=True))
        with self.assertRaises(ValueError):
            s.accepted_task1_expert(self.args.task1_acceptance)

    def test_real_dry_plan_binds_both_frozen_files_and_no_process(self):
        self.args.out.mkdir()
        with patch.object(s._p.subprocess, 'Popen', side_effect=AssertionError('No CPU physical process')):
            plan = s.build_plan(self.args)
        self.assertEqual(plan['total_all_tasks'], 500)
        self.assertEqual(plan['new_successful_episode_target'], 152)
        self.assertEqual(plan['prior_DR_import']['zhuhai_task1']['residual_shard_targets'], [38]*4)
        self.assertIn(str(Path(c.frozen.__file__).resolve()), plan['source_hashes'])
        self.assertIn(str(Path(old_s.__file__).resolve()), plan['source_hashes'])
        self.assertTrue(all(plan[k] is False for k in s.DISABLED))
        self.assertFalse(plan['recording']['video'])
        self.assertFalse(plan['recording']['third_view'])

    def test_full_mock_execution_publishes500_and_never_train(self):
        self.args.out.mkdir(); self.args.execute=True; self.args.resume=True
        plan=s.build_plan(self.args); plan_path=self.args.out/'plan.json'; publish(plan_path, plan)
        def completed_group(out, mode, jobs, args, state, plan):
            for job in jobs:
                state['stages'][job['name']]=dict(status='completed', validation=job['validate'](), process_exited=True)
        with patch.object(s._p, 'run_collection_group', completed_group), patch.object(s._p, 'log'):
            self.assertEqual(s.execute_collection(self.args, plan, plan_path), 0)
        completion=old_s.read_json(self.args.out/'completion.json')
        self.assertEqual(completion['total_successful_episodes'],500)
        self.assertEqual(completion['randomized_count_per_task'],{'zhuhai_task1':200,'zhuhai_task2':100})
        self.assertTrue(all(completion[k] is False for k in s.DISABLED))

    def test_recycle_accepts_two_slots_and_preserves_guard(self):
        path=self.parts[0]; value=old_s.read_json(path)
        # Pick both relatives of a previously empty family, a legal partial quota.
        grouped={}
        for row in value['attempts']:grouped.setdefault(row['family_id'],[]).append(row)
        rows=next(rows for rows in grouped.values() if len(rows)==2)
        value.update(status='worker_recycle_boundary',success_count=2,attempts=rows);publish(path,value)
        self.assertEqual(c.validate_recycle_boundary(path,task='zhuhai_task1')['successful_episodes'],2)
        value['attempts'].extend(dict(seed=rows[-1]['seed']+20+i,status='failed') for i in range(3));publish(path,value)
        with self.assertRaisesRegex(ValueError,'Unsafe recycle'):
            c.validate_recycle_boundary(path,task='zhuhai_task1')


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='dr200-queue-cpu-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.args=s.parser().parse_args(cli(self.root));self.args.out.mkdir()
        self.plan=dict(source_hashes={},robot_binding=dict(asset_sha256={}))
        self.state=dict(stages={});self.jobs=[]
        for i in range(4):
            command=s.collector_command(self.args,i);target=Path(s._p._command_value(command,'--out'))
            self.jobs.append(dict(name=f'randomized_task1_shard{i}',task='zhuhai_task1',shard_index=i,
                command=command,manifest=target/'manifest.json',validate=lambda:dict(successful_episodes=38,source_families=25)))

    def run_group(self,resources=good_resources,scenarios=None):
        self.system=FakeCollectors({'randomized_task1':scenarios or [dict(duration=60.,code=0,ready_after=None) for _ in range(4)]})
        with ExitStack() as stack:
            for name,value in [('log',Mock()),('verify_owner',Mock()),('check_disk',Mock()),('collection_resource_admission',resources)]:
                stack.enter_context(patch.object(s._p,name,value))
            stack.enter_context(patch.object(s._p.subprocess,'Popen',self.system.spawn))
            stack.enter_context(patch.object(s._p.time,'monotonic',lambda:self.system.now))
            stack.enter_context(patch.object(s._p.time,'sleep',self.system.sleep))
            stack.enter_context(patch.object(s._p.os,'killpg',self.system.killpg))
            s.run_collection_group(self.args.out,'randomized',self.jobs,self.args,self.state,self.plan)
        return self.system

    def test_only_four_disjoint_dr200_writers_and_stagger(self):
        system=self.run_group()
        self.assertEqual(len(system.launches),4)
        self.assertEqual(self.state['collection_groups']['randomized']['peak_active_workers'],4)
        self.assertTrue(all(b['time']-a['time']>=15-1e-7 for a,b in zip(system.launches,system.launches[1:])))
        self.assertTrue(all(row['command'][row['command'].index('--count')+1]=='200' for row in system.launches))
        self.assertTrue(all('--execute' in row['command'] for row in system.launches))

    def test_unknown_resources_refuse_and_low_headroom_serializes(self):
        with self.assertRaisesRegex(RuntimeError,'First worker'):
            self.run_group(resources=lambda:dict(admitted=True))
        self.assertEqual(self.system.launches,[])
        resource=good_resources();resource['memory']['available_bytes']=4*s.GIB
        self.state=dict(stages={});self.args.out=self.root/'serialrun';self.args.out.mkdir();self.run_group(resources=lambda:resource)
        self.assertEqual(self.state['collection_groups']['randomized']['peak_active_workers'],1)
        resource['swap_fraction']=.90
        self.assertFalse(s.resource_ready(resource,fourth=True))
        resource['swap_fraction']=.95
        self.assertFalse(s.resource_ready(resource))

    def test_duplicate_writer_live_pid_and_noncollection_flags_refused(self):
        self.state['stages'][self.jobs[0]['name']]=dict(command=self.jobs[0]['command'],status='running',pid=123,process_exited=False)
        with patch.object(s._p.os,'kill',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'still alive'):
                self.run_group()
        self.state=dict(stages={})
        for flag in ('--train','--tiny','--eval','--source-probe'):
            jobs=deepcopy(self.jobs);jobs[0]['command'].append(flag)
            with self.assertRaises(ValueError):s._collection_jobs(self.args.out,'randomized',jobs,self.args)

    def test_failure_guard_and_partial_never_reset(self):
        job=self.jobs[0];job['manifest'].parent.mkdir(parents=True)
        publish(job['manifest'],dict(contract={},attempts=[dict(status='failed') for _ in range(3)],status='stopped_consecutive_failures'))
        self.args.resume=True
        with patch.object(c,'validate_header'),patch.object(c,'_success_rows'):
            with self.assertRaisesRegex(RuntimeError,'Failure guard'):
                s.resume_target_guard(job,self.args,c)
        job['manifest'].unlink();(job['manifest'].parent/'original.partial').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'Unindexed partial'):
            s.resume_target_guard(job,self.args,c)

    def test_no_lock_alias_or_more_than_four_workers(self):
        self.args.collection_workers=5
        with self.assertRaises(ValueError):s.validate_args(self.args)
        self.args.collection_workers=4;self.args.collection_root=self.args.source_root
        with self.assertRaisesRegex(ValueError,'overlaps'):s.protect_output_paths(self.args)
        directory=self.root/'exclusive';directory.mkdir()
        with s.exclusive_lock(directory/'.pipeline.lock'):
            with self.assertRaisesRegex(RuntimeError,'Live writer lock'):
                with s.exclusive_lock(directory/'.pipeline.lock'):pass


if __name__=='__main__':unittest.main()
