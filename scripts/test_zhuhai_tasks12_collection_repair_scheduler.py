#!/usr/bin/env python3
"""Stdlib-only integration tests. Every archive/receipt here is a CPU fixture.

Never import torch/Isaac, query the GPU, run a collector, or write production
outputs. Main-entry execution, resource samples, Popen and GPU ownership are
mocked; manifest/physical-evidence validation itself is real unless noted.
"""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
import builtins
import io
import json
import os
from pathlib import Path
import signal
import symtable
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_zhuhai_tasks12_collection_repair as p
from scripts import collect_isaac_zhuhai_tasks12_repair as c
from scripts import collect_isaac_zhuhai_100_parallel as old1
from scripts import collect_isaac_zhuhai_recycle as old2
from scripts.test_zhuhai_pipeline import fixture_contract, fixture_row, FakeCollectors
from harness.zhuhai_task1_rules import Task1Judge, DESTINATIONS


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def good_resources():
    return dict(admitted=True, memory=dict(total_bytes=32*p.GIB, available_bytes=12*p.GIB),
                gpu_free_bytes=6*p.GIB, swap_fraction=.4)


def cli(root, workers=4):
    return ['--out', str(root/'run'), '--collection-root', str(root/'new'),
            '--source-root', str(root/'source'), '--deferred-root', str(root/'deferred'),
            '--task2-collection-run', str(root/'v11'), '--task1-prior-inventory', str(root/'prior1.json'),
            '--task2-prior-inventory', str(root/'prior2.json'), '--task1-admission', str(root/'admission.json'),
            '--life-mapping', 'body_text', '--collection-workers', str(workers), '--gpu-lock-owner', 'CPU_ONLY']


class CpuEvidence:
    """Small complete synthetic manifests, original48+52/task1 and13+87/task2."""
    def __init__(self, root):
        self.root = root; self.argv = cli(root); self.args = p.parser().parse_args(self.argv)
        self.sources, self.contracts, self.parts = {}, {}, {}
        for number in (1, 2):
            task = f'zhuhai_task{number}'
            source = root/'source'/task/'source/manifest.json'; source.parent.mkdir(parents=True)
            sc = fixture_contract(task=task)
            sc.update(source_probe=False, allow_unvalidated_probe=False, collection_scope='production')
            rows = [fixture_row(source.parent, sc, number*100000+1000+i) for i in range(100)]
            for i, row in enumerate(rows):
                label = 'train' if i < 80 else 'val' if i < 90 else 'heldout'
                row['split'] = label; row['metadata'].update(task=task, split=label)
            publish(source, dict(contract=sc, status='complete', success_count=100,
                                 requested_success_count=100, attempts=rows))
            self.sources[task] = source
            dr = deepcopy(sc)
            dr.update(mode='randomized', requested_success_count=100, seed_start=974000 if number==1 else 964000,
                max_consecutive_failures=3, max_attempts=1000, production_quota=dict(source=100, randomized=100, total_per_task=200, total_all_tasks=400),
                source_quota_transition=dict(from_quota=c.source_protocol.PRODUCTION_QUOTAS, to_quota=c.PRODUCTION_QUOTAS, source_relabelled=False),
                source_manifest=dict(path=str(source), sha256=p.sha(source), engine=sc['engine']),
                resource_policy=dict(c.RESOURCE_POLICY), worker_recycling=dict(c.WORKER_RECYCLING),
                collection_sharding=dict(shard_count=4, family_assignment='source seeds ascending; index modulo shard_count',
                    global_randomized_count=100, randomized_per_family=1, disjoint_seed_stride=1000),
                collection_implementation=p.reference(c.__file__), expert_variant=p.reference(c.TASK_EXPERT_PATHS[task]))
            prior_count, old = (48, old1) if number==1 else (13, old2)
            old_path = root/f'old{number}'/'manifest.json'; old_path.parent.mkdir()
            old_contract = deepcopy(dr); old_contract['collection_implementation'] = p.reference(old.__file__)
            old_rows = []
            for i, parent in enumerate(rows[:prior_count]):
                row = fixture_row(old_path.parent, old_contract, number*100000+10000+i, parent=parent['seed'])
                row['split'] = parent['split']; row['metadata'].update(task=task, split=row['split'])
                old_rows.append(row)
            failed = [dict(seed=number*100000+19000+i, status='failed', failure_reason='preserved CPU failed attempt') for i in range(3)]
            publish(old_path, dict(contract=old_contract, status='stopped_consecutive_failures',
                requested_success_count=100, success_count=prior_count, attempts=[*old_rows, *failed]))
            (old_path.parent/'original.partial').write_bytes(b'preserve this CPU partial')
            prior = root/f'prior{number}.json'
            publish(prior, dict(task=task, old_collector=p.reference(old.__file__), attempts=old_rows,
                successful_episodes=prior_count, components=[dict(**p.reference(old_path), successful_episodes=prior_count)]))
            dr['prior_inventory'] = p.reference(prior); self.contracts[task] = dr
            self.parts[task] = []
            for index in range(4):
                directory = (root/'new' if number==1 else root/'task2_data')/task/'randomized/shards'/f'shard_{index:02d}'
                directory.mkdir(parents=True)
                parents = c.residual_families(dr, index, 4)
                new_rows = []
                for j, parent in enumerate(parents):
                    row = fixture_row(directory, dr, dr['seed_start']+index*1000+j, parent=parent['seed'])
                    row['split'] = parent['split']; row['metadata'].update(task=task, split=row['split'])
                    new_rows.append(row)
                part = directory/'manifest.json'
                publish(part, dict(contract=dr, status='complete', success_count=len(new_rows),
                                   requested_success_count=len(new_rows), attempts=new_rows))
                self.parts[task].append(part)
        randomized = root/'task2_data/zhuhai_task2/randomized/manifest.json'
        c.aggregate_shards(randomized, self.parts['zhuhai_task2'], task='zhuhai_task2')
        v11 = root/'v11'; v11.mkdir()
        plan = dict(source_count=100, randomized_count=100, stages=['source_task2', 'randomized_task2'], **p.DISABLED)
        publish(v11/'plan.json', plan)
        sr = p.collection_receipt(self.sources['zhuhai_task2'], 'zhuhai_task2', 'source')
        dr = p.collection_receipt(randomized, 'zhuhai_task2', 'randomized')
        publish(v11/'tasks.json', dict(tasks=[dict(task='zhuhai_task2', source_manifest=str(self.sources['zhuhai_task2']), randomized_manifest=str(randomized))]))
        publish(v11/'status.json', dict(status='completed_collection_only', plan_sha256=p.sha(v11/'plan.json'), active_stage=None,
            stages={'source_task2': dict(status='completed', validation=sr), 'randomized_task2': dict(status='completed', validation=dr)}, **p.DISABLED))
        publish(v11/'completion.json', dict(status='completed_collection_only', task='zhuhai_task2',
            source_successful_episodes=100, randomized_successful_episodes=100, randomized_validation=dr, **p.DISABLED))
        self.probes = []
        for i in range(4):
            probe = root/'probes'/f'seed_{950000+i}'/'manifest.json'; probe.parent.mkdir(parents=True)
            contract = deepcopy(self.contracts['zhuhai_task1'])
            contract.update(source_probe=True, collection_scope='expert_repair_probe_not_production',
                            probe_implementation=p.reference(ROOT/'scripts/probe_zhuhai_task1_expert_repair_v1.py'))
            parent = p.read_json(self.sources['zhuhai_task1'])['attempts'][i]
            row = fixture_row(probe.parent, contract, 950000+i, parent=parent['seed'])
            score = Task1Judge().report()
            score.update(base_score=30, full_task_success=True, bins_fixed_throughout=True)
            for name, destination in DESTINATIONS.items():
                score['items'][name].update(destination=destination, grasp_awarded=True, placement_valid=True,
                                          passed_open_mouth=True, stable_seconds=1.1, base_points=8)
            for lid in score['lids'].values():
                lid.update(opened_awarded=True, open_now=True, base_points=3)
            row['final_rule_evidence'].update(score=score, physics_time_seconds=6.,
                objects={name: dict(released=True, touching_robot=False, CPU_fixture_only=True) for name in DESTINATIONS})
            row.update(observation_count=61, accepted_physics_times=[n/10 for n in range(61)])
            for stats in row['episode_statistics'].values():
                stats['count'] = 60
            trace = [dict(phase=name+':CPU_SYNTHETIC_FULL_TASK') for name in DESTINATIONS]
            receipt = dict(expert_class='harness.'+Path(contract['expert_variant']['path']).stem+'.SmoothExpert',
                expert_info=dict(plan_success=True, full_task_success=True, failure=None, trace=trace, command_count=60),
                trace=trace, motion_commands=[[0.]*16 for _ in range(60)], plan_success=True, state_dim=16, action_dim=16,
                state_semantics=p.STATE_SEMANTICS, data_contract_sha256=p.data_contract_hash(),
                robot_contract_sha256=contract['robot_contract_sha256'])
            publish(Path(row['expert_receipt_path']), receipt); row['expert_receipt_sha256'] = p.sha(row['expert_receipt_path'])
            publish(probe, dict(contract=contract, status='complete', success_count=1, requested_success_count=1, attempts=[row]))
            publish(probe.parent/'plan.json', dict(contract=contract, failed_seed=row['seed'], source_parent=parent['seed'], admitted_to_production=False))
            self.probes.append(probe)
        self.refresh_admission()

    def refresh_admission(self):
        publish(self.root/'admission.json', dict(status='passed', task='zhuhai_task1', collector=p.reference(c.__file__),
            expert=p.reference(c.TASK_EXPERT_PATHS['zhuhai_task1']), probe_manifests=[p.reference(x) for x in self.probes]))

    def main(self, *extra, group=None, **patches):
        """Real main, real preflight/manifests; execution side effects are mocked."""
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO())); stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(patch.object(p, 'log'))
            stack.enter_context(patch.object(p, 'verify_owner'))
            stack.enter_context(patch.object(p, 'check_disk'))
            stack.enter_context(patch.object(p, 'robot_binding', return_value=dict(assets_ready=True, asset_sha256={})))
            stack.enter_context(patch.object(p, 'runtime_interface', return_value=dict(ready=True)))
            stack.enter_context(patch.object(p.subprocess, 'Popen', side_effect=AssertionError('CPU test may not spawn real processes')))
            if group is not None:
                stack.enter_context(patch.object(p, 'run_collection_group', side_effect=group))
            for key, value in patches.items():
                stack.enter_context(patch.object(p, key, value))
            return p.main([*self.argv, *extra])


class EvidenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tasks12-cpu-'); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.fixture = CpuEvidence(self.root)

    def new_v2_suite(self):
        """Use the actual new probe's scope/plan/suite format, CPU artifacts only."""
        source = self.root/'v2_source/manifest.json'; source.parent.mkdir()
        sc = p.read_json(self.fixture.sources['zhuhai_task1'])['contract']
        parents = [fixture_row(source.parent, sc, 943000+i) for i in range(100)]
        publish(source, dict(contract=sc, status='complete', requested_success_count=100, success_count=100, attempts=parents))
        by_seed = {row['seed']:row for row in parents}
        candidate = ROOT/'harness/isaac_zhuhai_task1_expert_repair_v2.py'
        entry = ROOT/'scripts/probe_zhuhai_task1_expert_repair_v2.py'
        implementation = self.root/'collect_isaac_zhuhai_tasks12_repair_v2.py'
        implementation.write_text('# explicitly mocked CPU production collector\n')
        module = SimpleNamespace(__file__=str(implementation), TASK_EXPERT_PATHS={'zhuhai_task1':candidate},
                                 validate_manifest=c.validate_manifest, _success_rows=c._success_rows)
        extra = [candidate, entry, ROOT/'harness/isaac_zhuhai_task1_expert_repair_v1.py']
        base = deepcopy(self.fixture.contracts['zhuhai_task1'])
        base.update(source_manifest=dict(**p.reference(source), engine=sc['engine']),
                    collection_implementation=p.reference(old2.__file__), expert_variant=p.reference(candidate))
        base['scene_dependency_sha256'].update({str(path):p.sha(path) for path in extra})
        bindings = {**base['scene_dependency_sha256'], **base['robot_binding']['asset_sha256'],
                    str(source):p.sha(source), str(Path(old2.__file__).resolve()):p.sha(old2.__file__),
                    **{by_seed[seed]['data_path']:by_seed[seed]['data_sha256'] for seed in set(p.V2_PROBE_CASES.values())}}
        fingerprint = p.json_digest(bindings)
        directory = self.root/'v2_suite'; directory.mkdir()
        suite = dict(schema=p.V2_PROBE_SCOPE, required_cases=[dict(seed=s,parent=r) for s,r in p.V2_PROBE_CASES.items()],
            minimum_strict_physical_passes=4, candidate_sha256=p.sha(candidate), bindings_sha256=fingerprint,
            bindings=bindings, admitted_to_production=False, production_quota_credit=0, record_video=False, third_view=False)
        publish(directory/'suite_plan.json', suite)
        probes = []
        original = p.read_json(self.fixture.probes[0])['attempts'][0]
        for seed, parent in p.V2_PROBE_CASES.items():
            path = directory/f'seed_{seed}'/'manifest.json'; path.parent.mkdir()
            contract = deepcopy(base)
            contract.update(source_probe=True, allow_unvalidated_probe=True, collection_scope=p.V2_PROBE_SCOPE,
                admitted_to_production=False, production_quota_credit=0, requested_success_count=1,
                probe_implementation=p.reference(entry), probe_bindings_sha256=fingerprint,
                max_attempts=1, max_consecutive_failures=3)
            row = deepcopy(original); family = by_seed[parent]['family_id']
            row.update(seed=seed, family_id=family, source_parent=parent)
            row['metadata'].update(seed=seed, family_id=family, parent_source_seed=parent, contract=contract)
            data = path.parent/'episode.cpu_fixture'; data.write_text(f'CPU synthetic v2 evidence {seed}')
            row.update(data_path=str(data), data_sha256=p.sha(data))
            receipt = p.read_json(original['expert_receipt_path']); receipt['expert_class'] = 'harness.'+candidate.stem+'.SmoothExpert'
            receipt_path = path.parent/'expert.json'; publish(receipt_path, receipt)
            row.update(expert_receipt_path=str(receipt_path), expert_receipt_sha256=p.sha(receipt_path))
            publish(path, dict(contract=contract, requested_success_count=1, success_count=1, status='complete', attempts=[row]))
            publish(path.parent/'plan.json', dict(contract=contract, seed=seed, source_parent=parent,
                suite_bindings_sha256=fingerprint, admitted_to_production=False, production_quota_credit=0))
            probes.append(path)
        admission = directory/'admission.json'
        publish(admission, dict(status='passed', task='zhuhai_task1', collector=p.reference(implementation),
                                expert=p.reference(candidate), probe_manifests=[p.reference(x) for x in probes]))
        return module, admission, directory, probes

    def test_real_v2_probe_suite_schema_bindings_and_required_cases(self):
        module, admission, directory, probes = self.new_v2_suite()
        result = p.accepted_task1_expert(admission, module)
        self.assertEqual(result['seeds'], sorted(p.V2_PROBE_CASES))
        self.assertEqual(result['probe_success_count'], 4)
        self.assertEqual(result['collector'], p.reference(module.__file__))
        self.assertIn(str(directory/'suite_plan.json'), result['evidence_hashes'])
        self.assertIn(str(ROOT/'harness/isaac_zhuhai_task1_expert_repair_v1.py'), result['evidence_hashes'])

    def test_v2_boolean_suite_result_incomplete_manifest_and_binding_drift_fail_closed(self):
        module, admission, directory, probes = self.new_v2_suite()
        original = p.read_json(admission)
        publish(admission, dict(all_four_strict_passed=True, manifests=[str(p) for p in probes], probe_success_count=4))
        with self.assertRaises(ValueError): p.accepted_task1_expert(admission, module)
        publish(admission, original)
        preserved_manifest = p.read_json(probes[0]); manifest = deepcopy(preserved_manifest)
        manifest['attempts'] = []; publish(probes[0], manifest)
        original['probe_manifests'] = [p.reference(x) for x in probes]; publish(admission, original)
        with self.assertRaises(ValueError): p.accepted_task1_expert(admission, module)
        publish(probes[0], preserved_manifest)
        original['probe_manifests'] = [p.reference(x) for x in probes]; publish(admission, original)
        suite_path = directory/'suite_plan.json'; suite = p.read_json(suite_path)
        suite['bindings'][str(ROOT/'harness/isaac_zhuhai_task1_expert_repair_v1.py')] = '0'*64
        publish(suite_path, suite)
        with self.assertRaises(ValueError): p.accepted_task1_expert(admission, module)

    def test_real_main_dry_run_no_process_or_training(self):
        self.assertEqual(self.fixture.main(), 0)
        plan = p.read_json(self.root/'run/plan.json')
        self.assertEqual((plan['source_count'], plan['randomized_count'], plan['total_all_tasks']), (100, 100, 400))
        self.assertEqual(plan['prior_DR_import']['zhuhai_task1']['residual_shard_targets'], [13]*4)
        self.assertEqual(plan['task1_expert_acceptance']['probe_success_count'], 4)
        self.assertFalse((self.root/'run/status.json').exists())
        self.assertFalse((self.root/'run/completion.json').exists())
        self.assertFalse((self.root/'new/.collection_repair.lock').exists())
        self.assertTrue(all(plan[key] is False for key in p.DISABLED))
        self.assertFalse(any(word in stage for stage in plan['stages'] for word in ('train', 'tiny', 'eval')))

    def test_real_main_mock_execution_48_plus_52_retains_failure_evidence(self):
        preserved = {path: path.read_bytes() for directory in ('source', 'old1', 'old2', 'v11', 'probes')
                     for path in (self.root/directory).rglob('*') if path.is_file()}
        commands = []
        def group(out, mode, jobs, args, state, plan):
            p._collection_jobs(out, mode, jobs, args)
            commands.extend(j['command'] for j in jobs)
            for job in jobs:
                state['stages'][job['name']] = dict(status='completed', command=job['command'], validation=job['validate']())
        self.assertEqual(self.fixture.main('--execute', '--resume', group=group), 0)
        self.assertEqual(len(commands), 4)
        self.assertTrue(all(cmd[2] == str(Path(c.__file__).resolve()) and p._command_value(cmd, '--count') == '100' for cmd in commands))
        self.assertTrue(all(p._command_value(cmd, '--task') == 'zhuhai_task1' for cmd in commands))
        target = self.root/'new/zhuhai_task1/randomized/manifest.json'
        rows = c.validate_manifest(target, task='zhuhai_task1', mode='randomized')
        self.assertEqual(len(rows), 100); self.assertEqual(sum(r['seed'] < 974000 for r in rows), 48)
        self.assertEqual({label: sum(r['split']==label for r in rows) for label in ('train', 'val', 'heldout')}, dict(train=80, val=10, heldout=10))
        self.assertEqual(p.read_json(self.root/'run/completion.json')['total_successful_episodes'], 400)
        for path, data in preserved.items():
            self.assertEqual(path.read_bytes(), data)

    def test_explicit_v2_path_through_real_main_with_mock_versioned_module(self):
        # Clone only CPU functions into an isolated mocked versioned module;
        # never monkey-patch the running/frozen generic collector itself.
        import types
        version = self.root/'versioned'; version.mkdir()
        collector_path = version/'collect_isaac_zhuhai_tasks12_repair_v2.py'
        collector_path.write_text('# CPU mock v2 implementation; not a real collector\n')
        expert_path = version/'isaac_zhuhai_task1_expert_repair_v2.py'
        expert_path.write_text('# CPU mock v2 expert; no physical success claim\n')
        namespace = dict(vars(c)); namespace['__file__'] = str(collector_path)
        namespace['TASK_EXPERT_PATHS'] = {**c.TASK_EXPERT_PATHS, 'zhuhai_task1': expert_path}
        for key, value in vars(c).items():
            if isinstance(value, types.FunctionType) and value.__globals__ is vars(c):
                function = types.FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
                function.__kwdefaults__ = value.__kwdefaults__; namespace[key] = function
        module = SimpleNamespace(**namespace)
        for path in [*self.fixture.parts['zhuhai_task1'], *self.fixture.probes]:
            value = p.read_json(path); contract = value['contract']
            contract.update(collection_implementation=p.reference(collector_path), expert_variant=p.reference(expert_path))
            for row in value['attempts']:
                row['metadata']['contract'] = deepcopy(contract)
                if path in self.fixture.probes:
                    receipt_path = Path(row['expert_receipt_path']); receipt = p.read_json(receipt_path)
                    receipt['expert_class'] = 'harness.'+expert_path.stem+'.SmoothExpert'
                    publish(receipt_path, receipt); row['expert_receipt_sha256'] = p.sha(receipt_path)
            publish(path, value)
            if path in self.fixture.probes:
                plan_path = path.parent/'plan.json'; plan = p.read_json(plan_path); plan['contract'] = contract; publish(plan_path, plan)
        admission = self.root/'admission.json'
        publish(admission, dict(status='passed', task='zhuhai_task1', collector=p.reference(collector_path),
            expert=p.reference(expert_path), probe_manifests=[p.reference(x) for x in self.fixture.probes]))
        launches = []
        def group(out, mode, jobs, args, state, plan):
            p._collection_jobs(out, mode, jobs, args)
            for job in jobs:
                self.assertEqual(job['command'][2], str(collector_path))
                launches.append(job['command'])
                state['stages'][job['name']] = dict(status='completed', command=job['command'], validation=job['validate']())
        self.assertEqual(self.fixture.main('--execute', '--resume', '--task1-collector', str(collector_path), group=group,
            selected_collector=Mock(return_value=module), snapshot_sources=Mock(return_value=dict(CPU_mock=True))), 0)
        self.assertEqual(len(launches), 4)
        plan = p.read_json(self.root/'run/plan.json')
        self.assertEqual(plan['task1_collector'], p.reference(collector_path))
        self.assertEqual(plan['task1_expert_acceptance']['expert'], p.reference(expert_path))
        self.assertEqual(plan['expert_dependency_sha256'][str(expert_path)], p.sha(expert_path))
        inherited = str(ROOT/'harness/isaac_zhuhai_task1_expert_repair_v1.py')
        self.assertEqual(plan['source_hashes'][inherited], p.sha(inherited))
        self.assertEqual(p.read_json(self.root/'run/completion.json')['total_successful_episodes'], 400)

    def test_uneven_worker_residuals_keep_final_100_quota(self):
        for workers, counts in ((1,[52]), (2,[26,26]), (3,[18,17,17]), (4,[13,13,13,13])):
            self.fixture.args.collection_workers = workers
            with self.subTest(workers=workers):
                result = p.task1_prior_import(self.fixture.args)
                self.assertEqual(result['residual_shard_targets'], counts)
                self.assertEqual(result['successful_episodes']+sum(counts), 100)

    def test_completed_probe_data_cannot_be_relabelled_as_production_quota(self):
        path = self.fixture.parts['zhuhai_task1'][0]; value = p.read_json(path)
        probe = p.read_json(self.fixture.probes[0])['attempts'][0]
        value['attempts'][0].update(data_path=probe['data_path'], data_sha256=probe['data_sha256'])
        publish(path, value)
        self.assertEqual(self.fixture.main('--execute', '--resume', group=lambda *a:None), 2)
        self.assertFalse((self.root/'new/zhuhai_task1/randomized/manifest.json').exists())
        self.assertFalse((self.root/'run/completion.json').exists())

    def test_running_v11_is_waiting_and_never_loads_v2_or_freezes_plan(self):
        status = self.root/'v11/status.json'; value = p.read_json(status); value.update(status='running', success_count=62)
        publish(status, value)
        select = Mock(side_effect=AssertionError('must not select collector while v11 is incomplete'))
        self.assertEqual(self.fixture.main('--execute', selected_collector=select), 3)
        select.assert_not_called(); self.assertFalse((self.root/'run/plan.json').exists())
        self.assertEqual(p.read_json(self.root/'run/readiness.json')['waiting_for'], ['task2_collection'])

    def test_missing_admission_waits_but_boolean_admission_is_rejected(self):
        admission = self.root/'admission.json'; admission.unlink()
        self.assertEqual(self.fixture.main('--execute'), 3)
        publish(admission, dict(status='passed', success=True, probe_success_count=4))
        self.assertEqual(self.fixture.main('--execute'), 2)
        self.assertFalse((self.root/'run/plan.json').exists())

    def test_task2_status_completion_receipt_and_counts_not_sufficient(self):
        path = self.fixture.parts['zhuhai_task2'][0]
        manifest = p.read_json(path); manifest['attempts'].pop(); manifest['success_count'] -= 1
        publish(path, manifest)
        with self.assertRaises(ValueError): p.completed_task2_import(self.root/'v11')
        self.assertEqual(self.fixture.main('--execute'), 2)
        self.assertFalse((self.root/'run/plan.json').exists())

    def test_task2_completion_transaction_and_task_paths_hash_bound(self):
        status = self.root/'v11/status.json'; original = p.read_json(status)
        for change in (dict(plan_sha256='0'*64), dict(active_stage='randomized_task2'), dict(automatic_training=True)):
            publish(status, {**original, **change})
            with self.subTest(change=change), self.assertRaises(ValueError): p.completed_task2_import(self.root/'v11')
        publish(status, original)
        tasks = self.root/'v11/tasks.json'; value = p.read_json(tasks)
        value['tasks'][0]['source_manifest'] = str(self.fixture.sources['zhuhai_task1']); publish(tasks, value)
        with self.assertRaises(ValueError): p.completed_task2_import(self.root/'v11')

    def test_probe_three_duplicate_failed_and_changed_are_rejected(self):
        admission = self.root/'admission.json'; original = p.read_json(admission)
        for refs in (original['probe_manifests'][:3], [original['probe_manifests'][0]]*4):
            publish(admission, {**original, 'probe_manifests': refs})
            with self.assertRaises(ValueError): p.accepted_task1_expert(admission)
        self.fixture.refresh_admission()
        path = self.fixture.probes[0]; manifest = p.read_json(path)
        for status in ('running', 'stopped_attempt_limit', 'worker_recycle_boundary'):
            publish(path, {**manifest, 'status': status}); self.fixture.refresh_admission()
            with self.subTest(status=status), self.assertRaises(ValueError): p.accepted_task1_expert(admission)
        publish(path, manifest); self.fixture.refresh_admission(); path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError): p.accepted_task1_expert(admission)

    def test_probe_fake_expert_receipt_and_partial_task_rejected_after_rehash(self):
        path = self.fixture.probes[0]; manifest = p.read_json(path)
        row = manifest['attempts'][0]; receipt = Path(row['expert_receipt_path'])
        publish(receipt, dict(plan_success=True, full_task_success=True, successful_episodes=1))
        row['expert_receipt_sha256'] = p.sha(receipt); publish(path, manifest); self.fixture.refresh_admission()
        with self.assertRaisesRegex(ValueError, 'trace'): p.accepted_task1_expert(self.root/'admission.json')
        row['final_rule_evidence']['score']['items']['squeeze_bottle']['grasp_awarded'] = False
        publish(path, manifest); self.fixture.refresh_admission()
        with self.assertRaisesRegex(ValueError, 'three physical grasps'): p.accepted_task1_expert(self.root/'admission.json')

    def test_probe_global_data_duplicates_and_missing_hash_evidence_rejected(self):
        first, second = (p.read_json(x) for x in self.fixture.probes[:2])
        for key in ('data_path', 'data_sha256'):
            second['attempts'][0][key] = first['attempts'][0][key]
        publish(self.fixture.probes[1], second); self.fixture.refresh_admission()
        with self.assertRaisesRegex(ValueError, 'Duplicate physical'): p.accepted_task1_expert(self.root/'admission.json')

    def test_admission_collector_hash_binding_is_mandatory(self):
        path = self.root/'admission.json'; value = p.read_json(path)
        value['collector']['sha256'] = 'f'*64; publish(path, value)
        self.assertEqual(self.fixture.main('--execute'), 2)
        self.assertFalse((self.root/'run/plan.json').exists())

    def test_changed_evidence_after_dry_plan_fails_closed(self):
        self.assertEqual(self.fixture.main(), 0)
        before = (self.root/'run/plan.json').read_bytes()
        probe = self.fixture.probes[0]; probe.write_bytes(probe.read_bytes()+b' ')
        self.assertEqual(self.fixture.main('--execute', '--resume'), 2)
        self.assertEqual((self.root/'run/plan.json').read_bytes(), before)

    def test_old48_not_counts_only_and_residual_families_cannot_overlap(self):
        path = self.root/'old1/manifest.json'; value = p.read_json(path)
        Path(value['attempts'][0]['data_path']).write_bytes(b'changed old physical data')
        with self.assertRaises(ValueError): p.task1_prior_import(self.fixture.args)

    def test_failed_main_group_halts_no_aggregate_or_completion(self):
        self.assertEqual(self.fixture.main('--execute', '--resume', group=Mock(side_effect=RuntimeError('CPU group failure'))), 2)
        self.assertEqual(p.read_json(self.root/'run/status.json')['status'], 'halted_stage_failure')
        self.assertFalse((self.root/'run/completion.json').exists())
        self.assertFalse((self.root/'new/zhuhai_task1/randomized/manifest.json').exists())

    def test_source_family_split_and_data_replay_are_rejected(self):
        sources = c.validate_manifest(self.fixture.sources['zhuhai_task1'], task='zhuhai_task1', mode='source')
        old = c.prior_rows(self.root/'prior1.json')
        wrong = deepcopy(old); wrong[0]['split'] = wrong[0]['metadata']['split'] = 'heldout'
        with self.assertRaisesRegex(ValueError, 'cross split'): p.family_integrity(sources, wrong, complete=False)
        wrong = deepcopy(old); wrong[0]['data_sha256'] = sources[0]['data_sha256']
        with self.assertRaisesRegex(ValueError, 'Duplicate physical'): p.family_integrity(sources, wrong, complete=False)
        wrong = deepcopy(old); wrong[1]['family_id'] = wrong[0]['family_id']
        with self.assertRaisesRegex(ValueError, 'Duplicate randomized'): p.family_integrity(sources, wrong, complete=False)

    def test_v11_live_lock_and_alias_outputs_refused_without_modification(self):
        before = (self.root/'v11/status.json').read_bytes()
        with p.exclusive_lock(self.root/'v11/.pipeline.lock'):
            with self.assertRaisesRegex(RuntimeError, 'Live writer lock'): p.completed_task2_import(self.root/'v11')
        self.assertEqual((self.root/'v11/status.json').read_bytes(), before)
        self.fixture.args.out = self.root/'v11'/'nested'
        with self.assertRaisesRegex(ValueError, 'overlaps'): p.pipeline(self.fixture.args)
        self.assertFalse(self.fixture.args.out.exists())

    def test_two_coordinators_cannot_share_collection_root(self):
        with p.exclusive_lock(self.root/'new/.collection_repair.lock'):
            self.assertEqual(self.fixture.main('--execute', '--resume'), 2)
        self.assertFalse((self.root/'run/status.json').exists())


class SchedulerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tasks12-scheduler-cpu-'); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.args = p.parser().parse_args(cli(self.root))
        self.out = self.args.out; self.out.mkdir()
        self.plan = dict(source_hashes={}, robot_binding=dict(asset_sha256={}))
        self.state = dict(stages={})
        self.jobs = self.make_jobs()

    def make_jobs(self):
        jobs = []
        for index in range(self.args.collection_workers):
            cmd = p.collector_command(self.args, index); target = Path(p._command_value(cmd, '--out'))
            jobs.append(dict(name=f'randomized_task1_shard{index}', task='zhuhai_task1', shard_index=index,
                command=cmd, manifest=target/'manifest.json', validate=lambda: dict(successful_episodes=13, source_families=13)))
        return jobs

    def run_group(self, *, scenarios=None, resources=None, recycle=None):
        self.system = FakeCollectors({'randomized_task1': scenarios or [dict(duration=60., code=0, ready_after=None) for _ in range(4)]})
        with ExitStack() as stack:
            for name, value in [('log', Mock()), ('verify_owner', Mock()), ('check_disk', Mock()),
                                ('collection_resource_admission', resources or good_resources)]:
                stack.enter_context(patch.object(p, name, value))
            stack.enter_context(patch.object(p.subprocess, 'Popen', self.system.spawn))
            stack.enter_context(patch.object(p.time, 'monotonic', lambda: self.system.now))
            stack.enter_context(patch.object(p.time, 'sleep', self.system.sleep))
            stack.enter_context(patch.object(p.os, 'killpg', self.system.killpg))
            if recycle is not None:
                stack.enter_context(patch.object(c, 'validate_recycle_boundary', recycle))
            p.run_collection_group(self.out, 'randomized', self.jobs, self.args, self.state, self.plan)
        return self.system

    def test_four_workers_only_collection_and_15_second_stagger(self):
        system = self.run_group()
        self.assertEqual(self.state['collection_groups']['randomized']['peak_active_workers'], 4)
        self.assertEqual(len(system.launches), 4)
        self.assertTrue(all(b['time']-a['time'] >= 15-1e-7 for a,b in zip(system.launches, system.launches[1:])))
        self.assertEqual(system.signals, [])
        self.assertTrue(all(s['status']=='completed' for s in self.state['stages'].values()))

    def test_low_headroom_serial_and_unknown_first_worker_refused(self):
        resources = good_resources(); resources['memory']['available_bytes'] = 4*p.GIB
        system = self.run_group(resources=lambda: resources)
        self.assertEqual(self.state['collection_groups']['randomized']['peak_active_workers'], 1)
        self.assertEqual(len(system.launches), 4)
        self.assertFalse(p.resource_ready(dict(admitted=True)))

    def test_unknown_or_95percent_swap_never_starts_even_first_worker(self):
        with self.assertRaisesRegex(RuntimeError, 'First worker'):
            self.run_group(resources=lambda: dict(admitted=True))
        self.assertEqual(self.system.launches, [])
        bad = good_resources(); bad['swap_fraction'] = .95
        self.assertFalse(p.resource_ready(bad))

    def test_fourth_90percent_swap_fallback_retains_partial_and_resumes_same_shard(self):
        triggered = False
        def resource():
            nonlocal triggered
            answer = good_resources()
            if len([x for x in self.system.children if x.poll() is None]) == 4 and not triggered:
                triggered = True; answer['swap_fraction'] = .90
            return answer
        system = self.run_group(resources=resource)
        self.assertEqual(len(system.signals), 1)
        self.assertEqual(system.signals[0][0], system.launches[3]['pid'])
        self.assertTrue(any('--resume' in row['command'] for row in system.launches))
        self.assertTrue(list(self.root.rglob('partial_cpu_evidence_*')))
        self.assertLessEqual(self.state['collection_groups']['randomized']['peak_active_workers'], 4)
        reserve = good_resources(); reserve['swap_fraction'] = .91
        self.assertTrue(p.resource_ready(reserve)); self.assertFalse(p.resource_ready(reserve, fourth=True))

    def test_planned_recycle_uses_validator_and_does_not_reset_failure_history(self):
        scenarios = [dict(duration=1., code=75, ready_after=None), *[dict(duration=1., code=0, ready_after=None) for _ in range(4)]]
        boundary = Mock(return_value=dict(quota_complete=False, successful_episodes=3))
        system = self.run_group(scenarios=scenarios, recycle=boundary)
        boundary.assert_called_once(); self.assertEqual(len(system.launches), 5)
        self.assertIn('--resume', system.launches[-1]['command'])
        stage = self.state['stages'][self.jobs[0]['name']]
        self.assertEqual(stage['previous_attempt']['exit_code'], 75)
        self.assertTrue(list(self.out.rglob('result.json')))

    def test_failure_stops_expansion_and_terminates_only_owned_children(self):
        scenarios = [dict(duration=40., code=2, ready_after=None), *[dict(duration=100., code=0, ready_after=None) for _ in range(3)]]
        with self.assertRaisesRegex(RuntimeError, 'exited 2'): self.run_group(scenarios=scenarios)
        self.assertEqual(len(self.system.launches), 3)
        self.assertTrue(all(pid in {x['pid'] for x in self.system.launches} for pid, sig in self.system.signals))
        self.assertTrue(all(sig == signal.SIGTERM for pid, sig in self.system.signals))
        self.assertTrue(list(self.root.rglob('partial_cpu_evidence_*')))
        self.assertEqual(self.state['status'], 'halted_stage_failure')

    def test_spawn_failure_and_interruption_leave_attempt_results(self):
        with self.assertRaises(OSError): self.run_group(scenarios=[dict(spawn_error=True)])
        self.assertTrue(list(self.out.rglob('result.json')))
        self.assertEqual(self.state['stages'][self.jobs[0]['name']]['status'], 'failed')

    def test_signal_during_spawn_reaps_owned_process(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_group(scenarios=[dict(duration=100., code=0, signal_during_spawn=True)])
        self.assertEqual(len(self.system.signals), 1)
        self.assertEqual(self.state['status'], 'halted_interrupted')

    def test_queue_rejects_train_eval_tiny_executable_flags_and_task2_jobs(self):
        originals = deepcopy(self.jobs)
        mutations = [lambda j:j['command'].__setitem__(2, str(ROOT/'scripts/train_zhuhai_wam.py')),
                     lambda j:j['command'].extend(['--eval']), lambda j:j['command'].extend(['--source-probe']),
                     lambda j:j.update(name='tiny_training'), lambda j:j.update(task='zhuhai_task2'),
                     lambda j:j['command'].__setitem__(j['command'].index('--max-consecutive-failures')+1, '99'),
                     lambda j:j['command'].__setitem__(j['command'].index('--count')+1, '52')]
        for mutation in mutations:
            jobs = deepcopy(originals); mutation(jobs[0])
            with self.assertRaises(ValueError): p._collection_jobs(self.out, 'randomized', jobs, self.args)
        with self.assertRaises(ValueError): p.stage_needs_run(self.out, 'training', [], self.args, self.state)

    def test_live_previous_pid_and_completed_fake_receipt_rejected(self):
        job = self.jobs[0]
        self.state['stages'][job['name']] = dict(status='running', command=job['command'], pid=os.getpid())
        self.args.resume = True
        with self.assertRaisesRegex(RuntimeError, 'still alive'):
            p.stage_needs_run(self.out, job['name'], job['command'], self.args, self.state, job['validate'])
        self.state['stages'][job['name']].update(status='completed', validation=dict(successful_episodes=100))
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            p.stage_needs_run(self.out, job['name'], job['command'], self.args, self.state, job['validate'])

    def test_existing_failure_guard_and_unindexed_partial_never_reset(self):
        job = self.jobs[0]; job['manifest'].parent.mkdir(parents=True)
        partial = job['manifest'].parent/'episode.partial'; partial.write_bytes(b'CPU retained partial')
        with self.assertRaisesRegex(ValueError, 'Unindexed partial'): p.resume_target_guard(job, self.args, c)
        manifest = dict(status='stopped_consecutive_failures', contract={}, requested_success_count=13, attempts=[dict(status='failed')]*3)
        publish(job['manifest'], manifest); self.args.resume = True
        with patch.object(c, 'validate_header'), patch.object(c, '_success_rows', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'Failure guard'): p.resume_target_guard(job, self.args, c)
        self.assertEqual(p.read_json(job['manifest']), manifest); self.assertEqual(partial.read_bytes(), b'CPU retained partial')

    def test_fixed_quota_and_max_workers_at_cli_and_direct_entry(self):
        for flag, value in (('--source-count','99'), ('--randomized-count','52'), ('--collection-workers','5')):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): p.parser().parse_args(cli(self.root)+[flag,value])
        self.args.source_count = 52
        with self.assertRaises(ValueError): p.pipeline(self.args)
        self.args.source_count = 100; self.args.collection_workers = True
        with self.assertRaises(ValueError): p.validate_args(self.args)

    def test_forged_counts_cannot_satisfy_barrier(self):
        state = dict(stages={f'source_task{i}': dict(status='completed', validation=dict(successful_episodes=100, source_families=100)) for i in (1,2)})
        with self.assertRaisesRegex(RuntimeError, 'validators required'): p.collection_barrier(state, 'source')

    def test_output_lock_and_symlink_alias_exclusion(self):
        with p.exclusive_lock(self.out/'.pipeline.lock'):
            with self.assertRaisesRegex(RuntimeError, 'Live writer lock'): p.pipeline(self.args)
        alias = self.root/'alias'; alias.symlink_to(self.out, target_is_directory=True); self.args.out = alias
        with self.assertRaisesRegex(ValueError, 'symlink'): p.pipeline(self.args)

    def test_wrong_external_gpu_owner_is_rejected_read_only(self):
        with patch('scripts.gpu_lock.owner', return_value='RUNNING_V11'):
            with self.assertRaisesRegex(RuntimeError, 'matching external GPU lock'): p.verify_owner('CPU_ONLY')

    def test_explicit_v2_collector_selection_and_command_routing_without_importing_isaac(self):
        workspace = self.root/'workspace'; scripts = workspace/'scripts'; scripts.mkdir(parents=True)
        selected = scripts/'collect_isaac_zhuhai_tasks12_repair_v2.py'; selected.write_text('# synthetic import-safe v2 collector\n')
        expert = workspace/'harness/isaac_zhuhai_task1_expert_repair_v2.py'; expert.parent.mkdir(); expert.write_text('# synthetic v2\n')
        module = SimpleNamespace(**{key:getattr(c,key) for key in ('PRODUCTION_QUOTAS','RESOURCE_POLICY','WORKER_RECYCLING',
            'RECYCLE_EXIT_CODE','MINIMUM_FREE_BYTES','validate_manifest','_success_rows','prior_rows','residual_families',
            'aggregate_shards','validate_header','validate_recycle_boundary','consecutive_failures')})
        module.__file__ = str(selected); module.TASK_EXPERT_PATHS = {'zhuhai_task1': expert}
        self.args.task1_collector = selected
        with patch.object(p, 'ROOT', workspace), patch.object(p.importlib, 'import_module', return_value=module) as load:
            self.assertIs(p.selected_collector(self.args), module)
            load.assert_called_once_with('scripts.collect_isaac_zhuhai_tasks12_repair_v2')
            self.jobs = self.make_jobs(); p._collection_jobs(self.out, 'randomized', self.jobs, self.args)
            self.assertTrue(all(job['command'][2] == str(selected) for job in self.jobs))
            # Explicit selection cannot reuse an admission for generic/v1.
            admission = self.root/'admission.json'
            publish(admission, dict(status='passed', task='zhuhai_task1', collector=p.reference(c.__file__),
                                    expert=p.reference(expert), probe_success_count=4))
            with self.assertRaisesRegex(ValueError, 'selected collector'): p.accepted_task1_expert(admission, module)

    def test_nested_collection_alias_cannot_write_preserved_tree(self):
        target = self.root/'old'; target.mkdir()
        task = self.args.collection_root/'zhuhai_task1'; task.mkdir(parents=True)
        (task/'randomized').symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            p._collection_jobs(self.out, 'randomized', self.jobs, self.args)
        self.assertEqual(list(target.iterdir()), [])

    def test_invalid_recycle_receipt_halts_without_retry(self):
        scenarios = [dict(duration=1., code=75, ready_after=None)]
        with self.assertRaisesRegex(ValueError, 'invalid recycle'):
            self.run_group(scenarios=scenarios, recycle=Mock(side_effect=ValueError('invalid recycle')))
        self.assertEqual(len(self.system.launches), 1)
        self.assertEqual(self.state['status'], 'halted_stage_failure')

    def test_actual_v2_collector_api_and_both_experts_in_immutable_snapshot(self):
        self.args.task1_collector = ROOT/'scripts/collect_isaac_zhuhai_tasks12_repair_v2.py'
        selected = p.selected_collector(self.args)  # CPU import only; never main()/collect().
        self.assertEqual(Path(selected.__file__).resolve(), self.args.task1_collector)
        dependencies = p.expert_dependency_hashes(selected)
        expected = {str(ROOT/'harness'/f'isaac_zhuhai_task1_expert_repair_v{version}.py') for version in (1,2)}
        self.assertEqual(set(dependencies), expected)
        self.assertEqual(dependencies, selected.expert_dependency_binding('zhuhai_task1'))
        plan = dict(source_hashes={**dependencies, str(self.args.task1_collector):p.sha(self.args.task1_collector)},
                    robot_binding=dict(asset_sha256={}), expert_dependency_sha256=dependencies)
        with patch.object(p, 'check_disk'):
            receipt = p.snapshot_sources(self.out, plan)
        files = p.read_json(receipt['path'])['files']
        for path, digest in dependencies.items():
            relative = Path(path).relative_to(ROOT)
            self.assertEqual(files[str(relative)]['sha256'], digest)
            self.assertEqual(p.sha(self.out/'source_snapshot'/relative), digest)
        before = {path:Path(path).read_bytes() for path in dependencies}
        p.verify_plan_sources(plan)
        for path, data in before.items(): self.assertEqual(Path(path).read_bytes(), data)

    def test_no_unresolved_globals_or_heavy_runtime_imports(self):
        declared = set(vars(p)) | set(vars(builtins)); missing = set()
        def inspect(scope):
            for symbol in scope.get_symbols():
                if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in declared:
                    missing.add(symbol.get_name())
            for child in scope.get_children(): inspect(child)
        inspect(symtable.symtable(Path(p.__file__).read_text(), p.__file__, 'exec'))
        self.assertEqual(missing, set())
        self.assertFalse(any(name in sys.modules for name in ('torch', 'isaacsim', 'numpy', 'h5py')))


if __name__ == '__main__':
    unittest.main()
