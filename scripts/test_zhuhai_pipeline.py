#!/usr/bin/env python3
"""CPU-only coordinator/admission regression; all archives here are synthetic.

No Isaac imports, model construction, CUDA operations, or real collection.
"""
from copy import deepcopy
import json
import os
import signal
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_isaac_zhuhai as collect
from scripts import run_zhuhai_pipeline as pipeline


def rule_state(success=False, time=0.):
    return dict(physics_time_seconds=time,
        physics_safety=dict(passed=True, events=[], checks={'CPU_fixture_only': True}), objects={'synthetic_fixture': {'physical_test': False}},
        score=dict(base_score=30 if success else 0, full_task_success=success,
            items={'synthetic_fixture': {}}, simulation_tolerances={}, rule_source='CPU fixture, not physical evidence'))


def fixture_contract(task='zhuhai_task1', mode='source', count=None):
    robot = collect.robot_spec.robot_contract()
    return dict(engine=collect.ENGINE, task=task, track='life', mode=mode,
        state_dim=16, action_dim=16, state_semantics=collect.STATE_SEMANTICS, action_semantics=collect.ACTION_SEMANTICS,
        state_contract=collect.robot_spec.STATE_CONTRACT, action_contract=collect.robot_spec.ACTION_CONTRACT,
        robot_contract=robot, robot_contract_sha256=collect.json_digest(robot),
        data_contract=collect.data_contract(), data_contract_sha256=collect.data_contract_hash(),
        requested_success_count=collect.PRODUCTION_QUOTAS[mode] if count is None else count,
        production_quota=dict(collect.PRODUCTION_QUOTAS, total_per_task=600, total_all_tasks=1200),
        life_mapping='body_text', robot=robot['profile'],
        robot_binding={'asset_sha256': {}}, source_scene={'path': 'cpu-only', 'sha256': 'a'*64},
        scene_dependency_sha256={}, record_video=False, third_view=False, camera_count=3,
        camera_names=list(collect.ONBOARD_CAMERAS), frequency_hz=10., physics_dt_seconds=.004,
        minimum_free_bytes=collect.MINIMUM_FREE_BYTES)


def fixture_row(directory, contract, seed, parent=None):
    path = directory / f'episode_{seed}.cpu_fixture'
    path.write_bytes(f'CPU synthetic data only: {seed}'.encode())
    family = f'life:{contract["task"]}:source:{seed if parent is None else parent}'
    receipt = directory/f'expert_{seed}.json'
    collect.write_once(receipt, dict(CPU_fixture_only=True, seed=seed, action_dim=16, state_dim=16, trace=[]))
    def stats(semantics):
        home = list(collect.robot_spec.HOME_ACTION)
        return dict(dimension=16, semantics=semantics, count=1, mean=home, std=[0.]*16, minimum=home, maximum=home)
    return dict(seed=seed, status='success', family_id=family, source_parent=parent,
        state_dim=16, action_dim=16, state_semantics=collect.STATE_SEMANTICS, action_semantics=collect.ACTION_SEMANTICS,
        data_contract_sha256=collect.data_contract_hash(), robot_contract_sha256=contract['robot_contract_sha256'],
        episode_statistics=dict(state=stats(collect.STATE_SEMANTICS), action=stats(collect.ACTION_SEMANTICS)),
        expert_receipt_path=str(receipt), expert_receipt_sha256=collect.sha256(receipt), expert_receipt_valid=True,
        physics_safety_all_observations_passed=True,
        initial_camera_observations={name: dict(nonempty_rgb=True, CPU_fixture_only=True) for name in collect.ONBOARD_CAMERAS},
        initial_head_camera_evidence=dict(all_geometry_in_frame=True, all_named_entities_visible=True),
        metadata=dict(seed=seed, family_id=family, contract=contract, parent_source_seed=parent),
        plan_success=True, strict_predicate=True, observation_count=2,
        initial_rule_evidence=rule_state(), final_rule_evidence=rule_state(True, .1),
        accepted_physics_times=[0., .1], data_path=str(path), data_sha256=collect.sha256(path))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.log_patch = patch.object(pipeline, 'log')
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def source(self):
        directory = self.root/'source'; directory.mkdir()
        contract = fixture_contract()
        rows = [fixture_row(directory, contract, 1000+i) for i in range(100)]
        path = directory/'manifest.json'
        collect.write_once(path, dict(contract=contract, status='complete', requested_success_count=100,
                                      success_count=100, attempts=rows))
        return path

    def randomized(self, source):
        directory = self.root/'randomized'; directory.mkdir()
        contract = fixture_contract(mode='randomized')
        contract['source_manifest'] = dict(path=str(source), sha256=collect.sha256(source))
        rows = [fixture_row(directory, contract, 10000+i, parent=1000+i%100) for i in range(500)]
        path = directory/'manifest.json'
        collect.write_once(path, dict(contract=contract, status='complete', requested_success_count=500,
                                      success_count=500, attempts=rows))
        return path

    def alter(self, path, change):
        data = json.loads(path.read_text()); change(data)
        path.write_text(json.dumps(data))  # Only deliberately corrupted temporary CPU fixtures.

    def args(self, *extra):
        return pipeline.parser().parse_args(['--out', str(self.root/'run'), '--life-mapping', 'body_text', *extra])

    def test_defaults_plan_and_new_totals_without_simulator(self):
        args = self.args()
        self.assertEqual((args.source_count, args.randomized_count), (100, 500))
        self.assertEqual(args.collection_workers, 1)
        with patch.object(pipeline.subprocess, 'check_output', return_value='{}'), \
             patch.object(pipeline, 'robot_binding', return_value=dict(asset_sha256={}, assets_ready=True,
                 robot_contract=collect.robot_spec.robot_contract(), robot_contract_sha256=collect.json_digest(collect.robot_spec.robot_contract()))):
            plan = pipeline.build_plan(args)
        self.assertEqual((plan['total_per_task'], plan['total_all_tasks']), (600, 1200))
        self.assertEqual(plan['family_split']['episodes_per_task'], dict(train=480, val=60, heldout=60))
        logger = ROOT / 'scripts/progress1_log.py'
        self.assertEqual(plan['source_hashes'].get(str(logger)), collect.sha256(logger))
        self.assertEqual(plan['collection_workers'], 1)
        self.assertEqual(plan['collection_scheduling']['minimum_available_ram_bytes'], 12*1024**3)
        self.assertEqual(plan['collection_scheduling']['minimum_free_gpu_bytes'], 6*1024**3)
        self.assertEqual(plan['collection_scheduling']['training_and_evaluation_workers'], 1)
        self.assertTrue(plan['source_snapshot_policy']['weights_and_HDF_excluded'])
        self.assertEqual(plan['cache']['disk_budget_bytes'], 10**12)
        self.assertEqual(plan['cache']['minimum_free_bytes'], 100*1024**3)
        self.assertEqual(plan['recording']['camera_count'], 3)
        self.assertIs(plan['recording']['video'], False)
        self.assertIs(plan['recording']['third_view'], False)
        self.assertFalse(any(name == 'isaacsim' or name.startswith('omni.') for name in sys.modules))

    def test_400_production_rejected_before_output(self):
        with self.assertRaisesRegex(ValueError, '600 per task'):
            pipeline.pipeline(self.args('--randomized-count', '400'))
        self.assertFalse((self.root/'run').exists())
        args = ['collect', '--task', 'zhuhai_task1', '--mode', 'randomized', '--count', '400',
                '--seed-start', '2000', '--out', str(self.root/'collection'), '--plan-only']
        with patch.object(sys, 'argv', args), self.assertRaisesRegex(ValueError, 'another 500'):
            collect.main()
        self.assertFalse((self.root/'collection').exists())

    def test_all_pipeline_commands_pass_explicit_quotas_and_reserves(self):
        args = self.args('--execute', '--gpu-lock-owner', 'CPU-MOCK-ONLY')
        plan = dict(scene_ready=True, source_hashes={}, robot_binding={'asset_sha256': {}},
            robot_contract_sha256=collect.json_digest(collect.robot_spec.robot_contract()),
            cache=dict(root=str(self.root/'cache'), disk_budget_bytes=10**12, ram_budget_bytes=2*1024**3),
            evaluation_seeds=[932001], total_per_task=600, total_all_tasks=1200)
        calls = {}
        def fake_stage(out, name, command, args, state, validate=None):
            calls[name] = command
            state['stages'][name] = {'status': 'completed'}
            if name.startswith(('source_', 'randomized_')):
                mode = name.split('_task')[0]
                state['stages'][name]['validation'] = dict(successful_episodes=collect.PRODUCTION_QUOTAS[mode], source_families=100)
            if name == 'fresh_seed_physical_evaluation':
                (out/'evaluation').mkdir()
                collect.write_once(out/'evaluation/summary.json', {'status': 'CPU MOCK ONLY'})
        with patch.object(pipeline, 'build_plan', return_value=plan), \
             patch.object(pipeline, 'verify_owner'), patch.object(pipeline, 'run_stage', side_effect=fake_stage):
            self.assertEqual(pipeline.pipeline(args), 0)
        for mode, count in collect.PRODUCTION_QUOTAS.items():
            for task in (1, 2):
                cmd = calls[f'{mode}_task{task}']
                self.assertEqual(cmd[cmd.index('--count')+1], str(count))
        for name in ('tiny_training', 'training'):
            cmd = calls[name]
            for flag, value in (('--source-count', 100), ('--randomized-count', 500),
                                ('--cache-budget-bytes', 10**12), ('--cache-ram-budget-bytes', 2*1024**3),
                                ('--minimum-free-bytes', 100*1024**3)):
                self.assertEqual(cmd[cmd.index(flag)+1], str(value))
        completion = json.loads((args.out/'completion.json').read_text())
        self.assertEqual((completion['total_per_task'], completion['total_all_tasks']), (600, 1200))

    def test_old_plan_is_never_upgraded_or_overwritten(self):
        args = self.args(); args.out.mkdir()
        path = args.out/'plan.json'
        collect.write_once(path, {'randomized_count': 400, 'total_per_task': 500})
        before = path.read_bytes()
        with patch.object(pipeline, 'build_plan', return_value={'randomized_count': 500}), self.assertRaises(ValueError):
            pipeline.pipeline(args)
        self.assertEqual(path.read_bytes(), before)

    def test_write_once_preserves_plan_evidence_and_timestamp(self):
        path = self.root/'result.json'; collect.write_once(path, {'failed': True})
        before, stamp = path.read_bytes(), path.stat().st_mtime_ns
        collect.write_once(path, {'failed': True}, identical_ok=True)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        with self.assertRaises(FileExistsError):collect.write_once(path, {'failed': False})
        self.assertEqual(path.read_bytes(), before)

    def test_full_100_plus_500_source_family_admission(self):
        source = self.source(); dr = self.randomized(source)
        families = collect.validate_manifest(source, task='zhuhai_task1', mode='source')
        rows = collect.validate_manifest(dr, task='zhuhai_task1', mode='randomized')
        self.assertEqual((len(families), len(rows)), (100, 500))
        self.assertTrue(all(sum(r['family_id'] == f['family_id'] for r in rows) == 5 for f in families))

    def test_success_flag_does_not_override_failed_expert_or_score(self):
        source = self.source()
        original = json.loads(source.read_text())
        for field, value in [('plan_success', False), ('plan_success', 'true'), ('strict_predicate', False),
                             ('observation_count', 1)]:
            data = deepcopy(original); data['attempts'][0][field] = value
            source.write_text(json.dumps(data))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                collect.validate_manifest(source, task='zhuhai_task1', mode='source')
        data = deepcopy(original)
        data['attempts'][0]['final_rule_evidence'] = rule_state(False, .1)
        source.write_text(json.dumps(data))
        with self.assertRaises(ValueError):collect.validate_manifest(source, task='zhuhai_task1', mode='source')

    def test_failure_and_partial_attempts_do_not_fill_success_quota(self):
        source = self.source()
        self.alter(source, lambda d: d['attempts'][0].update(status='failed'))
        with self.assertRaisesRegex(ValueError, 'success_count'):
            collect.validate_manifest(source, task='zhuhai_task1', mode='source')
        self.alter(source, lambda d: d.update(success_count=99))
        with self.assertRaisesRegex(ValueError, '100 strictly successful'):
            collect.validate_manifest(source, task='zhuhai_task1', mode='source')

    def test_changed_source_content_rejected_without_manifest_rewrite(self):
        source = self.source(); data = json.loads(source.read_text())
        Path(data['attempts'][0]['data_path']).write_bytes(b'changed CPU fixture')
        before = source.read_bytes()
        with self.assertRaises(ValueError):collect.initial_manifest(source, data['contract'], 100)
        self.assertEqual(source.read_bytes(), before)

    def test_resume_does_not_retarget_quota_in_place(self):
        source = self.source(); data = json.loads(source.read_text()); before = source.read_bytes()
        with self.assertRaisesRegex(ValueError, 'quota changed'):
            collect.initial_manifest(source, data['contract'], 500)
        self.assertEqual(source.read_bytes(), before)

    def test_duplicate_source_family_or_file_is_rejected(self):
        source = self.source(); original = json.loads(source.read_text())
        data = deepcopy(original)
        data['attempts'][1]['family_id'] = data['attempts'][0]['family_id']
        data['attempts'][1]['metadata']['family_id'] = data['attempts'][0]['family_id']
        source.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'distinct canonical'):
            collect.validate_manifest(source, task='zhuhai_task1', mode='source')
        data = deepcopy(original)
        for key in ('data_path', 'data_sha256'):data['attempts'][1][key] = data['attempts'][0][key]
        source.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Duplicate physical'):
            collect.validate_manifest(source, task='zhuhai_task1', mode='source')

    def test_wrong_dr_parent_seed_or_unregistered_family_is_rejected(self):
        source = self.source(); dr = self.randomized(source); original = json.loads(dr.read_text())
        for field, value in [('source_parent', 9999), ('family_id', 'life:zhuhai_task1:source:9999')]:
            data = deepcopy(original); data['attempts'][0][field] = value
            if field == 'family_id':data['attempts'][0]['metadata'][field] = value
            dr.write_text(json.dumps(data))
            with self.subTest(field=field), self.assertRaises(ValueError):
                collect.validate_manifest(dr, task='zhuhai_task1', mode='randomized')

    def test_dr_cannot_reuse_source_episode_bytes(self):
        source = self.source(); dr = self.randomized(source)
        parent = json.loads(source.read_text())['attempts'][0]
        self.alter(dr, lambda d: d['attempts'][0].update(data_path=parent['data_path'], data_sha256=parent['data_sha256']))
        with self.assertRaisesRegex(ValueError, 'additional physical rollouts'):
            collect.validate_manifest(dr, task='zhuhai_task1', mode='randomized')

    def test_source_manifest_hash_and_robot_mapping_are_bound(self):
        source = self.source(); dr = self.randomized(source)
        self.alter(source, lambda d: d['contract'].update(robot='different_cpu_fixture'))
        with self.assertRaisesRegex(ValueError, 'source manifest changed'):
            collect.validate_manifest(dr, task='zhuhai_task1', mode='randomized')

    def test_old_dr_400_quota_rejected(self):
        source = self.source(); dr = self.randomized(source)
        self.alter(dr, lambda d: d.update(requested_success_count=400, success_count=400, attempts=d['attempts'][:400]))
        before = dr.read_bytes()
        with self.assertRaisesRegex(ValueError, 'production quota'):
            collect.validate_manifest(dr, task='zhuhai_task1', mode='randomized')
        self.assertEqual(dr.read_bytes(), before)

    def test_five_successful_relatives_not_five_attempts(self):
        families = [{'family_id': str(i), 'seed': i} for i in range(100)]
        attempts = []
        for i in range(500):
            parent = collect.next_parent(families, attempts)
            attempts.append(dict(status='failed', family_id=parent['family_id']))
            self.assertEqual(collect.next_parent(families, attempts), parent)
            attempts.append(dict(status='success', family_id=parent['family_id']))
        self.assertTrue(all(sum(r['status'] == 'success' and r['family_id'] == f['family_id'] for r in attempts) == 5 for f in families))

    def test_unbalanced_family_count_rejected_even_at_total_500(self):
        source = self.source(); dr = self.randomized(source)
        def alter(data):
            row = data['attempts'][0]
            row.update(family_id='life:zhuhai_task1:source:1001', source_parent=1001)
            row['metadata'].update(family_id=row['family_id'], parent_source_seed=1001)
        self.alter(dr, alter)
        with self.assertRaisesRegex(ValueError, 'five successful relatives'):
            collect.validate_manifest(dr, task='zhuhai_task1', mode='randomized')

    def test_camera_and_recording_protocol_are_strict(self):
        source = self.source(); original = json.loads(source.read_text())
        for field, value in [('record_video', True), ('third_view', True), ('camera_count', 4),
                             ('camera_names', ['head_camera']), ('minimum_free_bytes', 10**9)]:
            data = deepcopy(original); data['contract'][field] = value
            source.write_text(json.dumps(data))
            with self.subTest(field=field), self.assertRaises(ValueError):
                collect.validate_manifest(source, task='zhuhai_task1', mode='source')

    def test_resume_does_not_reset_three_failure_stop(self):
        directory = self.root/'stopped'; directory.mkdir()
        c = fixture_contract()
        collect.write_once(directory/'manifest.json', dict(contract=c, requested_success_count=100,
            success_count=0, status='stopped_consecutive_failures', attempts=[dict(seed=i, status='failed') for i in range(3)]))
        args = SimpleNamespace(out=directory, count=100, seed_start=0, max_attempts=None, max_consecutive_failures=3)
        self.assertEqual(collect.collect(args, c, []), 2)
        self.assertNotIn('harness.isaac_zhuhai_runtime', sys.modules)

    def test_nonuniform_timestamps_are_not_successful_demonstrations(self):
        source = self.source()
        self.alter(source, lambda d: d['attempts'][0].update(accepted_physics_times=[0., .2]))
        with self.assertRaisesRegex(ValueError, 'uniform 10Hz'):
            collect.validate_manifest(source, task='zhuhai_task1', mode='source')

    def test_unique_stage_attempt_logs_and_failure_halt(self):
        args = self.args(); args.out.mkdir(); args.gpu_lock_owner = 'CPU-MOCK-ONLY'
        state = {'stages': {}}
        command = [sys.executable, '-B', '-c', 'print("synthetic failure"); raise SystemExit(2)']
        with patch.object(pipeline, 'verify_owner'), patch.object(pipeline, 'check_disk'):
            with self.assertRaises(RuntimeError):pipeline.run_stage(args.out, 'source_task1', command, args, state)
            first = args.out/'stage_attempts/source_task1/attempt_0001/process.log'
            before = first.read_bytes()
            args.resume = True
            with self.assertRaises(RuntimeError):pipeline.run_stage(args.out, 'source_task1', command, args, state)
        self.assertEqual(first.read_bytes(), before)
        self.assertTrue((args.out/'stage_attempts/source_task1/attempt_0002/result.json').is_file())
        self.assertEqual(state['status'], 'halted_stage_failure')

    def test_exit_zero_without_valid_artifact_is_failed_stage(self):
        args = self.args(); args.out.mkdir(); state = {'stages': {}}
        def invalid():raise ValueError('completed quota absent')
        with patch.object(pipeline, 'verify_owner'), patch.object(pipeline, 'check_disk'), self.assertRaises(ValueError):
            pipeline.run_stage(args.out, 'source_task1', [sys.executable, '-B', '-c', 'pass'], args, state, validate=invalid)
        self.assertEqual(state['stages']['source_task1']['status'], 'failed')

    def test_completed_stage_is_revalidated_without_rerunning(self):
        args = self.args(); args.out.mkdir(); state = {'stages': {}}
        cmd = [sys.executable, '-B', '-c', 'pass']
        with patch.object(pipeline, 'verify_owner'), patch.object(pipeline, 'check_disk'):
            pipeline.run_stage(args.out, 'stage', cmd, args, state, validate=lambda:{'sha256': 'original'})
        with patch.object(pipeline.subprocess, 'Popen', side_effect=AssertionError('must not rerun')), self.assertRaises(ValueError):
            pipeline.run_stage(args.out, 'stage', cmd, args, state, validate=lambda:{'sha256': 'changed'})

    def test_evaluation_keeps_failed_policy_seeds_in_denominator(self):
        out = self.root/'eval'; rows = []
        for task in ('zhuhai_task1', 'zhuhai_task2'):
            target = out/task/'seed_1'; target.mkdir(parents=True)
            row = dict(task=task, seed=1, status='completed', full_task_success=False)
            collect.write_once(target/'result.json', row); rows.append(row)
        collect.write_once(out/'summary.json', dict(status='completed', episodes=rows))
        receipt = pipeline.evaluation_receipt(out, {'evaluation_seeds': [1]})
        self.assertEqual((receipt['completed_episodes'], receipt['full_successes']), (2, 0))
        self.alter(out/'summary.json', lambda d: d.update(episodes=d['episodes'][:1]))
        with self.assertRaises(ValueError):pipeline.evaluation_receipt(out, {'evaluation_seeds': [1]})

    def test_incomplete_core_stage_cannot_overwrite_prior_artifacts(self):
        args = self.args('--resume'); args.out.mkdir()
        target = args.out/'training'; target.mkdir()
        path = target/'plan.json'; path.write_bytes(b'original training configuration')
        cmd = [sys.executable, '-B', '-c', 'pass']
        state = {'stages': {'training': dict(status='failed', command=cmd)}}
        with patch.object(pipeline.subprocess, 'Popen', side_effect=AssertionError('must not spawn')), \
             self.assertRaisesRegex(RuntimeError, 'transactional resume'):
            pipeline.run_stage(args.out, 'training', cmd, args, state)
        self.assertEqual(path.read_bytes(), b'original training configuration')
        self.assertFalse((args.out/'stage_attempts').exists())

    def test_running_stage_cannot_start_a_second_writer(self):
        import os
        args = self.args('--resume'); args.out.mkdir()
        cmd = [sys.executable, '-B', '-c', 'pass']
        state = {'stages': {'source_task1': dict(status='running', command=cmd, pid=os.getpid())}}
        with self.assertRaisesRegex(RuntimeError, 'still alive'):
            pipeline.run_stage(args.out, 'source_task1', cmd, args, state)

    def test_spec_contract_and_missing_assets_are_cpu_safe(self):
        with patch.object(collect.robot_spec, 'ASSET', self.root/'missing_robot.usda'), \
             patch.object(collect.robot_spec, 'ASSET_MANIFEST', self.root/'missing_manifest.json'), \
             patch.object(collect.robot_spec, 'KINEMATIC_CHAIN', self.root/'missing_chain.json'):
            binding = collect.robot_binding()
        self.assertFalse(binding['assets_ready'])
        self.assertEqual(binding['robot_contract']['action_dim'], 16)
        self.assertEqual(binding['robot_contract']['state_dim'], 16)
        self.assertEqual(len(binding['missing_assets']), 3)
        self.assertNotIn('isaacsim', sys.modules)

    def test_native14_or_commanded_state_contract_is_rejected(self):
        source = self.source(); original = json.loads(source.read_text())
        for field, value in [('state_dim', 14), ('action_dim', 14), ('state_semantics', 'commanded_joint_gripper')]:
            data = deepcopy(original); data['contract'][field] = value
            source.write_text(json.dumps(data))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Native16'):
                collect.validate_manifest(source, task='zhuhai_task1', mode='source')

    def test_success_requires_16d_measured_statistics_and_expert_receipt(self):
        source = self.source(); original = json.loads(source.read_text())
        data = deepcopy(original); data['attempts'][0]['episode_statistics']['state']['dimension'] = 14
        source.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, '16D'):
            collect.validate_manifest(source, task='zhuhai_task1', mode='source')
        source.write_text(json.dumps(original))
        receipt = Path(original['attempts'][0]['expert_receipt_path'])
        receipt.write_bytes(b'changed fixture receipt')
        with self.assertRaisesRegex(ValueError, 'trace/receipt'):
            collect.validate_manifest(source, task='zhuhai_task1', mode='source')

    def test_physics_safety_false_missing_or_not_continuous_rejects_success_manifest(self):
        source = self.source(); original = json.loads(source.read_text())
        for case in ('initial_false', 'final_missing', 'not_continuous'):
            data = deepcopy(original); row = data['attempts'][0]
            if case == 'initial_false':row['initial_rule_evidence']['physics_safety']['passed'] = False
            elif case == 'final_missing':row['final_rule_evidence'].pop('physics_safety')
            else:row['physics_safety_all_observations_passed'] = False
            source.write_text(json.dumps(data))
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, 'physics_safety'):
                collect.validate_manifest(source, task='zhuhai_task1', mode='source')

    def test_visibility_flag_alone_cannot_admit_missing_rgb_readback(self):
        source = self.source()
        self.alter(source, lambda d: d['attempts'][0]['initial_camera_observations']['right_camera'].update(nonempty_rgb=False))
        with self.assertRaisesRegex(ValueError, 'actual nonempty RGB'):
            collect.validate_manifest(source, task='zhuhai_task1', mode='source')

    def test_checkpoint_receipt_rejects_legacy14_contract(self):
        directory = self.root/'checkpoint'; directory.mkdir()
        path = directory/'checkpoint_step_000002.pt'; path.write_bytes(b'CPU fixture, not model weights')
        collect.write_once(path.with_name(path.name+'.sha256.json'), dict(step=2, size=path.stat().st_size, sha256=collect.sha256(path)))
        contract = dict(format='zhuhai-isaac-native16-measured-10hz-v1',
            config={'common': dict(state_dim=16, action_dim=16)},
            data_contract=collect.data_contract(), robot_contract=collect.robot_spec.robot_contract())
        collect.write_once(directory/'manifest.json', {'contract': contract})
        receipt = pipeline.checkpoint_receipt(directory, 2)
        self.assertEqual(receipt['action_dim'], 16)
        self.alter(directory/'manifest.json', lambda d: d['contract']['config']['common'].update(action_dim=14))
        with self.assertRaisesRegex(ValueError, 'native16'):
            pipeline.checkpoint_receipt(directory, 2)

    def test_explicit_one_source_probe_plan_keeps_production_quota(self):
        target = self.root/'one_source_probe'
        argv = ['collect', '--task', 'zhuhai_task2', '--mode', 'source', '--source-probe',
                '--count', '1', '--seed-start', '921002', '--out', str(target), '--plan-only']
        with patch.object(sys, 'argv', argv):self.assertEqual(collect.main(), 0)
        c = json.loads((target/'plan.json').read_text())['contract']
        self.assertEqual(c['requested_success_count'], 1)
        self.assertEqual(c['max_attempts'], 1)
        self.assertTrue(c['source_probe'])
        self.assertTrue(c['allow_unvalidated_probe'])
        self.assertEqual(c['production_quota']['source'], 100)
        self.assertEqual(c['production_quota']['randomized'], 500)
        self.assertNotIn('isaacsim', sys.modules)

    def test_probe_cannot_relax_formal_or_randomized_source_pool(self):
        source = self.source()
        self.alter(source, lambda d: d['contract'].update(source_probe=True, allow_unvalidated_probe=True))
        with self.assertRaisesRegex(ValueError, 'not production'):
            collect.validate_manifest(source, task='zhuhai_task1', mode='source')
        for mode, count in (('randomized', '1'), ('source', '100')):
            argv = ['collect', '--task', 'zhuhai_task2', '--mode', mode, '--source-probe',
                    '--count', count, '--seed-start', '921002', '--out', str(self.root/'invalid')]
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'source/count=1'):
                collect.main()


class FakeCollectors:
    """Deterministic CPU fake Popen/process groups; no real GPU/collector spawn."""
    def __init__(self, scenarios=None):
        self.now = 0.
        self.scenarios = scenarios or {}
        self.children = []
        self.signals = []
        self.launches = []

    def sleep(self, seconds):
        self.now += seconds

    def spawn(self, command, **kwargs):
        if kwargs.get('start_new_session') is not True:
            raise AssertionError('collector must own an isolated new process group')
        name = pipeline._command_value(command, '--mode') + '_' + pipeline._command_value(command, '--task').replace('zhuhai_', '')
        choices = self.scenarios.get(name, [])
        scenario = choices.pop(0) if choices else dict(duration=3., code=0, ready_after=None)
        if scenario.get('spawn_error'):
            raise OSError('CPU fake spawn failure')
        child = self.Child(self, command, name, scenario)
        self.children.append(child)
        self.launches.append(dict(name=name, time=self.now, pid=child.pid, command=list(command)))
        if scenario.get('signal_during_spawn'):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return child

    def killpg(self, pid, sig):
        child = next((c for c in self.children if c.pid == pid), None)
        if child is None:
            raise AssertionError('attempt to signal an unowned/non-child process group')
        if sig != signal.SIGTERM:
            raise AssertionError('only SIGTERM is authorized')
        self.signals.append((pid, sig))
        child.returncode = -sig
        child.output.mkdir(parents=True, exist_ok=True)
        (child.output/f'partial_cpu_evidence_{pid}').write_bytes(b'preserve resumable synthetic evidence')

    class Child:
        def __init__(self, system, command, name, scenario):
            self.system, self.command, self.name, self.scenario = system, command, name, scenario
            self.pid = 500000 + len(system.children)
            self.started = system.now
            self.returncode = None
            self.ready_written = self.interrupted = False
            self.output = Path(pipeline._command_value(command, '--out'))
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output/f'initial_cpu_evidence_{self.pid}').write_bytes(b'CPU fixture, not physical data')
        def poll(self):
            elapsed = self.system.now-self.started
            if self.returncode is not None:
                return self.returncode
            if self.scenario.get('interrupt_after') is not None and elapsed >= self.scenario['interrupt_after'] and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt('CPU fake coordinator interruption')
            ready = self.scenario.get('ready_after')
            if ready is not None and elapsed >= ready and not self.ready_written:
                path = self.output/'manifest.json'
                value = json.loads(path.read_text()) if path.exists() else {'attempts': []}
                value['attempts'].append(dict(initial_rule_evidence={'CPU_fixture_only': True}))
                path.write_text(json.dumps(value))
                self.ready_written = True
            if elapsed >= self.scenario.get('duration', 3.):
                self.returncode = self.scenario.get('code', 0)
            return self.returncode
        def wait(self, timeout=None):
            if self.returncode is None:
                self.system.now = max(self.system.now, self.started+self.scenario.get('duration', 3.))
            return self.poll()


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root/'run'; self.out.mkdir()
        self.args = pipeline.parser().parse_args(['--out', str(self.out), '--life-mapping', 'body_text',
            '--collection-workers', '2', '--gpu-lock-owner', 'CPU-fixture-owner'])
        self.state = {'stages': {}}
        self.plan = dict(source_hashes={}, robot_binding={'asset_sha256': {}})
        self.patches = ExitStack(); self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(pipeline, 'log'))
        self.patches.enter_context(patch.object(pipeline, 'verify_owner'))
        self.patches.enter_context(patch.object(pipeline, 'check_disk'))
        self.good_resource = dict(admitted=True, available_ram_bytes=12*pipeline.GIB, gpu_free_bytes=6*pipeline.GIB)

    def jobs(self, mode='source'):
        result = []
        for i in (1, 2):
            directory = self.out/f'zhuhai_task{i}'/mode
            command = ['CPU_FAKE_COLLECTOR', '--task', f'zhuhai_task{i}', '--mode', mode,
                '--count', str(collect.PRODUCTION_QUOTAS[mode]), '--seed-start', str(933000+i*10000+(1000 if mode=='randomized' else 0)),
                '--out', str(directory), '--gpu-lock-owner', self.args.gpu_lock_owner]
            if mode == 'randomized':
                command += ['--source-manifest', str(self.out/f'zhuhai_task{i}'/'source/manifest.json')]
            receipt = dict(successful_episodes=collect.PRODUCTION_QUOTAS[mode], source_families=100,
                           manifest=str(directory/'manifest.json'))
            result.append(dict(name=f'{mode}_task{i}', command=command, manifest=directory/'manifest.json',
                               validate=lambda receipt=receipt: dict(receipt)))
        return result

    def run_fake(self, system, jobs=None, admission=None, delay=.6):
        jobs = self.jobs() if jobs is None else jobs
        mode = jobs[0]['name'].split('_task')[0]
        with patch.object(pipeline.subprocess, 'Popen', side_effect=system.spawn), \
             patch.object(pipeline.os, 'killpg', side_effect=system.killpg), \
             patch.object(pipeline.time, 'monotonic', side_effect=lambda: system.now), \
             patch.object(pipeline.time, 'sleep', side_effect=system.sleep), \
             patch.object(pipeline, 'SECOND_START_DELAY_SECONDS', delay), \
             patch.object(pipeline, 'COLLECTION_POLL_SECONDS', .2), \
             patch.object(pipeline, 'collection_resource_admission', side_effect=admission or (lambda:self.good_resource)):
            return pipeline.run_collection_group(self.out, mode, jobs, self.args, self.state, self.plan)

    def test_two_workers_success_stagger_owner_isolation_and_status_history(self):
        system = FakeCollectors()
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        self.run_fake(system)
        self.assertEqual(len(system.launches), 2)
        self.assertGreaterEqual(system.launches[1]['time'], .6)
        self.assertLess(system.launches[1]['time'], 3.)
        self.assertEqual(system.signals, [])
        self.assertEqual(self.state['collection_groups']['source']['peak_active_workers'], 2)
        self.assertEqual(before, {sig: signal.getsignal(sig) for sig in before})
        for i in (1, 2):
            name=f'source_task{i}'; entry=self.state['stages'][name]
            self.assertEqual(entry['status'], 'completed'); self.assertEqual(entry['exit_code'], 0)
            self.assertEqual(entry['external_gpu_lock_owner'], self.args.gpu_lock_owner)
            self.assertEqual(entry['validation']['successful_episodes'], 100)
            folder=self.out/'stage_attempts'/name/'attempt_0001'
            for filename in ('intent.json','status.json','result.json','process.log'):
                self.assertTrue((folder/filename).is_file())
        pipeline.collection_barrier(self.state, 'source')

    def test_first_new_initial_evidence_allows_early_resource_check(self):
        system=FakeCollectors({'source_task1':[dict(duration=5., code=0, ready_after=.2)]})
        self.run_fake(system, delay=15.)
        self.assertLess(system.launches[1]['time'], 15.)
        decision=self.state['collection_groups']['source']['decisions'][0]
        self.assertEqual(decision['warmup_trigger'], 'new_initial_rule_evidence')

    def test_stale_initial_evidence_does_not_bypass_15_second_warmup(self):
        path=self.out/'zhuhai_task1/source/manifest.json'; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'attempts':[{'initial_rule_evidence':{'old':True}}]}))
        system=FakeCollectors({'source_task1':[dict(duration=20., code=0, ready_after=None)]})
        self.run_fake(system, delay=15.)
        self.assertGreaterEqual(system.launches[1]['time'],15.)
        self.assertEqual(self.state['collection_groups']['source']['decisions'][0]['warmup_trigger'],'15s_startup_delay')

    def test_low_or_unavailable_resources_fall_back_to_serial(self):
        system=FakeCollectors()
        self.run_fake(system, admission=lambda:dict(admitted=False, reason='RAM below 12GiB'))
        self.assertGreaterEqual(system.launches[1]['time'],3.)
        group=self.state['collection_groups']['source']
        self.assertEqual(group['peak_active_workers'],1)
        self.assertEqual(group['decisions'][0]['decision'],'serial_resource_fallback')
        self.assertEqual(system.signals,[])

    def test_resource_probe_exception_is_serial_not_blind_parallel(self):
        system=FakeCollectors()
        def fail():raise OSError('CPU mocked unavailable inventory')
        self.run_fake(system, admission=fail)
        self.assertEqual(self.state['collection_groups']['source']['peak_active_workers'],1)
        self.assertIn('unavailable',self.state['collection_groups']['source']['decisions'][0]['admission']['error'])

    def test_first_completed_before_delay_needs_no_second_worker_resource_probe(self):
        system=FakeCollectors({'source_task1':[dict(duration=.2,code=0)]})
        def never():raise AssertionError('no overlap means no resource query')
        self.run_fake(system, admission=never,delay=15.)
        self.assertEqual(self.state['collection_groups']['source']['peak_active_workers'],1)
        self.assertEqual(self.state['collection_groups']['source']['decisions'][0]['decision'],'serial_first_finished_before_overlap')

    def test_failure_terminates_only_known_live_sibling_group(self):
        protected=self.root/'unrelated_result.json';protected.write_bytes(b'not ours')
        system=FakeCollectors({'source_task1':[dict(duration=2.,code=2)],'source_task2':[dict(duration=20.,code=0)]})
        with self.assertRaises(RuntimeError):self.run_fake(system)
        self.assertEqual(system.signals,[(system.children[1].pid,signal.SIGTERM)])
        self.assertEqual(protected.read_bytes(),b'not ours')
        self.assertEqual(self.state['stages']['source_task1']['status'],'failed')
        self.assertEqual(self.state['stages']['source_task2']['status'],'interrupted')
        self.assertEqual(self.state['stages']['source_task2']['exit_code'],-signal.SIGTERM)
        self.assertTrue(list((self.out/'zhuhai_task2/source').glob('partial_cpu_evidence_*')))
        self.assertEqual(self.state['status'],'halted_stage_failure')
        with self.assertRaises(RuntimeError):pipeline.collection_barrier(self.state,'source')

    def test_exit_zero_but_invalid_manifest_stops_sibling(self):
        jobs=self.jobs()
        def fail():raise ValueError('CPU fake quota incomplete')
        jobs[0]['validate']=fail
        system=FakeCollectors({'source_task1':[dict(duration=2.,code=0)],'source_task2':[dict(duration=20.,code=0)]})
        with self.assertRaisesRegex(ValueError,'quota'):self.run_fake(system,jobs)
        self.assertEqual(len(system.signals),1)
        self.assertEqual(self.state['stages']['source_task1']['exit_code'],0)
        self.assertEqual(self.state['stages']['source_task1']['status'],'failed')

    def test_first_failure_before_warmup_never_starts_second(self):
        system=FakeCollectors({'source_task1':[dict(duration=.2,code=2)]})
        with self.assertRaises(RuntimeError):self.run_fake(system,delay=15.)
        self.assertEqual(len(system.launches),1)
        self.assertNotIn('source_task2',self.state['stages'])
        self.assertFalse((self.out/'zhuhai_task2').exists())
        self.assertEqual(system.signals,[])

    def test_second_spawn_failure_stops_first_and_saves_both_receipts(self):
        system=FakeCollectors({'source_task2':[dict(spawn_error=True)]})
        with self.assertRaises(OSError):self.run_fake(system)
        self.assertEqual(system.signals,[(system.children[0].pid,signal.SIGTERM)])
        self.assertEqual(self.state['stages']['source_task2']['status'],'failed')
        self.assertIsNone(self.state['stages']['source_task2']['exit_code'])
        self.assertTrue((self.out/'stage_attempts/source_task2/attempt_0001/result.json').exists())

    def test_interrupt_restores_handlers_and_only_SIGTERMs_owned_children(self):
        before={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        system=FakeCollectors({'source_task1':[dict(duration=20.,interrupt_after=1.)],
                               'source_task2':[dict(duration=20.)]})
        with self.assertRaises(KeyboardInterrupt):self.run_fake(system)
        self.assertEqual({p for p,s in system.signals},{c.pid for c in system.children})
        self.assertEqual(before,{sig:signal.getsignal(sig) for sig in before})
        self.assertEqual(self.state['status'],'halted_interrupted')

    def test_resume_completed_child_skipped_failed_child_gets_new_attempt(self):
        jobs=self.jobs()
        first=FakeCollectors({'source_task1':[dict(duration=1.,code=0)],'source_task2':[dict(duration=2.,code=2)]})
        with self.assertRaises(RuntimeError):self.run_fake(first,jobs)
        prior={p:p.read_bytes() for p in self.out.rglob('*') if p.is_file() and p.name not in ('status.json',)}
        self.args.resume=True
        jobs[1]['command'].append('--resume')
        second=FakeCollectors()
        self.run_fake(second,jobs)
        self.assertEqual([x['name'] for x in second.launches],['source_task2'])
        for path,content in prior.items():self.assertEqual(path.read_bytes(),content)
        entry=self.state['stages']['source_task2']
        self.assertEqual(entry['attempt_number'],2)
        self.assertEqual(entry['previous_attempt']['status'],'failed')
        self.assertEqual(self.state['stages']['source_task1']['attempt_number'],1)
        pipeline.collection_barrier(self.state,'source')

    def test_resume_interrupted_data_preserved_and_both_commands_explicit(self):
        jobs=self.jobs()
        first=FakeCollectors({'source_task1':[dict(duration=2.,code=2)],'source_task2':[dict(duration=20.)]})
        with self.assertRaises(RuntimeError):self.run_fake(first,jobs)
        partial=next((self.out/'zhuhai_task2/source').glob('partial_cpu_evidence_*'));before=partial.read_bytes()
        self.args.resume=True
        for job in jobs:job['command'].append('--resume')
        self.run_fake(FakeCollectors(),jobs)
        self.assertEqual(partial.read_bytes(),before)
        self.assertEqual(self.state['stages']['source_task2']['previous_attempt']['status'],'interrupted')
        self.assertTrue((self.out/'stage_attempts/source_task1/attempt_0002/result.json').exists())

    def test_completed_evidence_changed_on_resume_never_relaunches(self):
        jobs=self.jobs();self.run_fake(FakeCollectors(),jobs);self.args.resume=True
        jobs[0]['validate']=lambda:dict(successful_episodes=99,source_families=100)
        system=FakeCollectors()
        with self.assertRaisesRegex(ValueError,'evidence changed'):self.run_fake(system,jobs)
        self.assertEqual(system.launches,[])

    def test_running_pid_is_not_signaled_or_replaced_on_resume(self):
        jobs=self.jobs();self.args.resume=True
        self.state['stages']['source_task2']=dict(status='running',pid=os.getpid(),command=jobs[1]['command'])
        system=FakeCollectors()
        with self.assertRaisesRegex(RuntimeError,'still alive'):self.run_fake(system,jobs)
        self.assertEqual(system.launches,[]);self.assertEqual(system.signals,[])

    def test_outside_output_wrong_owner_and_wrong_DR_parent_rejected(self):
        for kind in ('outside','owner','DR_parent'):
            jobs=self.jobs('randomized' if kind=='DR_parent' else 'source')
            flag='--source-manifest' if kind=='DR_parent' else '--out' if kind=='outside' else '--gpu-lock-owner'
            jobs[1]['command'][jobs[1]['command'].index(flag)+1]=str(self.root/'foreign')
            system=FakeCollectors()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.run_fake(system,jobs)
            self.assertEqual(system.launches,[]);self.assertFalse((self.root/'foreign').exists())

    def test_signal_during_Popen_is_deferred_until_child_is_registered(self):
        system=FakeCollectors({'source_task1':[dict(duration=20.,signal_during_spawn=True)]})
        with self.assertRaises(KeyboardInterrupt):self.run_fake(system)
        self.assertEqual(len(system.children),1)
        self.assertEqual(system.signals,[(system.children[0].pid,signal.SIGTERM)])
        self.assertEqual(self.state['stages']['source_task1']['status'],'interrupted')
        self.assertTrue(self.state['stages']['source_task1']['process_exited'])

    def test_collection_workers_choices_and_immutable_plan(self):
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                pipeline.parser().parse_args(['--out',str(self.out),'--life-mapping','body_text','--collection-workers','3'])
        self.args.collection_workers=1
        one=dict(collection_workers=1,scene_ready=True)
        with patch.object(pipeline,'build_plan',return_value=one),patch('builtins.print'):
            self.assertEqual(pipeline.pipeline(self.args),0)
        path=self.out/'plan.json';before=(path.read_bytes(),path.stat().st_mtime_ns)
        self.args.collection_workers=2
        with patch.object(pipeline,'build_plan',return_value=dict(one,collection_workers=2)),self.assertRaises(ValueError):
            pipeline.pipeline(self.args)
        self.assertEqual((path.read_bytes(),path.stat().st_mtime_ns),before)

    def test_worker1_keeps_serial_stage_calls_without_resource_queries(self):
        self.args.collection_workers=1;called=[]
        with patch.object(pipeline,'run_stage',side_effect=lambda out,name,*a,**kw:called.append(name)), \
             patch.object(pipeline,'collection_resource_admission',side_effect=AssertionError('must not query')):
            pipeline.run_collection_group(self.out,'source',self.jobs(),self.args,self.state,self.plan)
        self.assertEqual(called,['source_task1','source_task2'])

    def test_resource_thresholds_exact_and_fail_closed_without_GPU_calls(self):
        from scripts import zhuhai_resource_watch as watcher
        for available,free,expected in ((12*pipeline.GIB,6*pipeline.GIB,True),
                (12*pipeline.GIB-1,6*pipeline.GIB,False),(12*pipeline.GIB,6*pipeline.GIB-1,False)):
            gpu=dict(status='read_only_inventory',devices=[dict(total_bytes=16*pipeline.GIB,used_bytes=16*pipeline.GIB-free)])
            with patch.object(watcher,'memory_info',return_value={'available_bytes':available}), \
                 patch.object(watcher,'gpu_info',return_value=gpu):
                self.assertIs(pipeline.collection_resource_admission()['admitted'],expected)
        with patch.object(watcher,'memory_info',return_value={'available_bytes':32*pipeline.GIB}), \
             patch.object(watcher,'gpu_info',return_value={'status':'unavailable'}):
            self.assertFalse(pipeline.collection_resource_admission()['admitted'])

    def test_group_barriers_stop_DR_and_training_on_incomplete_quotas(self):
        args=self.args;args.execute=True
        plan=dict(self.plan,scene_ready=True,cache=dict(root=str(self.root/'cache'),disk_budget_bytes=10**12,ram_budget_bytes=2*1024**3),
                  robot_contract_sha256='a'*64,evaluation_seeds=[1])
        group_calls=[];later=[]
        def incomplete(out,mode,jobs,args,state,plan):
            group_calls.append(mode)
            for i in (1,2):state['stages'][f'{mode}_task{i}']=dict(status='completed',
                validation=dict(successful_episodes=100 if i==1 else 99,source_families=100))
        with patch.object(pipeline,'build_plan',return_value=plan), \
             patch.object(pipeline,'run_collection_group',side_effect=incomplete), \
             patch.object(pipeline,'run_stage',side_effect=lambda *a,**k:later.append(a)),self.assertRaises(RuntimeError):
            pipeline.pipeline(args)
        self.assertEqual(group_calls,['source']);self.assertEqual(later,[])

    def test_both_source_validated_before_DR_and_all1200_before_serial_training(self):
        args=self.args;args.execute=True
        plan=dict(self.plan,scene_ready=True,cache=dict(root=str(self.root/'cache'),disk_budget_bytes=10**12,ram_budget_bytes=2*1024**3),
                  robot_contract_sha256='a'*64,evaluation_seeds=[1])
        events=[]
        def group(out,mode,jobs,args,state,plan):
            if mode=='randomized':pipeline.collection_barrier(state,'source')
            events.append(mode)
            for job in jobs:state['stages'][job['name']]=dict(status='completed',validation=dict(
                successful_episodes=collect.PRODUCTION_QUOTAS[mode],source_families=100))
        def stage(out,name,command,args,state,validate=None):
            pipeline.collection_barrier(state,'source');pipeline.collection_barrier(state,'randomized')
            events.append(name);state['stages'][name]={'status':'completed'}
            if name=='fresh_seed_physical_evaluation':
                (out/'evaluation').mkdir();collect.write_once(out/'evaluation/summary.json',{'CPU_fixture':True})
        with patch.object(pipeline,'build_plan',return_value=plan),patch.object(pipeline,'run_collection_group',side_effect=group), \
             patch.object(pipeline,'run_stage',side_effect=stage):
            self.assertEqual(pipeline.pipeline(args),0)
        self.assertEqual(events,['source','randomized','tiny_training','training','fresh_seed_physical_evaluation'])
        self.assertTrue((self.out/'source_snapshot/receipt.json').exists())


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.workspace=self.root/'workspace';self.workspace.mkdir()
        self.out=self.root/'run';self.out.mkdir()
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(pipeline,'ROOT',self.workspace))
        self.stack.enter_context(patch.object(pipeline,'check_disk'))
        self.code=self.workspace/'scripts/task.py';self.code.parent.mkdir();self.code.write_bytes(b'print("frozen code")\n')
        self.config=self.workspace/'config/small.json';self.config.parent.mkdir();self.config.write_bytes(b'{"frozen":true}')
        self.logger=self.workspace/'scripts/progress1_log.py';self.logger.write_bytes(b'# stdlib progress refresh\n')
        self.plan=dict(source_hashes={str(p):collect.sha256(p) for p in (self.code,self.config,self.logger)},collection_workers=2)

    def test_snapshot_relative_layout_hashes_and_idempotent_resume(self):
        receipt=pipeline.snapshot_sources(self.out,self.plan)
        folder=self.out/'source_snapshot'
        self.assertEqual((folder/'scripts/task.py').read_bytes(),self.code.read_bytes())
        self.assertEqual((folder/'config/small.json').read_bytes(),self.config.read_bytes())
        saved={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file()}
        self.assertEqual(pipeline.snapshot_sources(self.out,self.plan),receipt)
        self.assertEqual(saved,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in saved})
        info=json.loads((folder/'receipt.json').read_text())
        self.assertEqual(info['plan_sha256'],collect.json_digest(self.plan))
        self.assertEqual(set(info['files']),{'scripts/task.py','config/small.json','scripts/progress1_log.py'})
        self.assertEqual((folder/'scripts/progress1_log.py').read_bytes(),self.logger.read_bytes())

    def test_hash_preflight_before_any_snapshot_copy(self):
        self.config.write_bytes(b'changed before copy')
        with self.assertRaisesRegex(ValueError,'changed before snapshot'):pipeline.snapshot_sources(self.out,self.plan)
        self.assertFalse((self.out/'source_snapshot').exists())

    def test_live_drift_does_not_overwrite_frozen_source_snapshot(self):
        pipeline.snapshot_sources(self.out,self.plan)
        copied=self.out/'source_snapshot/scripts/task.py';old=copied.read_bytes()
        self.code.write_bytes(b'new ongoing development')
        with self.assertRaises(ValueError):pipeline.snapshot_sources(self.out,self.plan)
        self.assertEqual(copied.read_bytes(),old)

    def test_partial_snapshot_resume_and_collision_protection(self):
        folder=self.out/'source_snapshot/scripts';folder.mkdir(parents=True)
        copied=folder/'task.py';copied.write_bytes(self.code.read_bytes())
        stamp=copied.stat().st_mtime_ns
        pipeline.snapshot_sources(self.out,self.plan);self.assertEqual(copied.stat().st_mtime_ns,stamp)
        copied.write_bytes(b'prior evidence, do not overwrite')
        with self.assertRaises(ValueError):pipeline.snapshot_sources(self.out,self.plan)
        self.assertEqual(copied.read_bytes(),b'prior evidence, do not overwrite')

    def test_weights_HDF_and_large_files_never_copied_or_deleted(self):
        for filename in ('checkpoint.pt','episode.hdf5','oversize.py'):
            path=self.workspace/filename
            path.write_bytes(b'protected artifact')
            self.plan['source_hashes'][str(path)]=collect.sha256(path)
        with patch.object(pipeline,'SNAPSHOT_FILE_LIMIT',30):
            # The oversize source remains a small CPU fixture; mocked stat size
            # is not needed, just exceed the test's deliberately tiny cap.
            path=self.workspace/'oversize.py';path.write_bytes(b'x'*31)
            self.plan['source_hashes'][str(path)]=collect.sha256(path)
            pipeline.snapshot_sources(self.out,self.plan)
        receipt=json.loads((self.out/'source_snapshot/receipt.json').read_text())
        self.assertEqual(set(receipt['skipped']),{'checkpoint.pt','episode.hdf5','oversize.py'})
        for name in receipt['skipped']:
            self.assertFalse((self.out/'source_snapshot'/name).exists());self.assertTrue((self.workspace/name).exists())

    def test_snapshot_destination_symlink_or_external_source_cannot_escape(self):
        foreign=self.root/'foreign';foreign.mkdir()
        (self.out/'source_snapshot').symlink_to(foreign,target_is_directory=True)
        with self.assertRaises(ValueError):pipeline.snapshot_sources(self.out,self.plan)
        self.assertEqual(list(foreign.iterdir()),[])
        (self.out/'source_snapshot').unlink()
        path=foreign/'evidence.py';path.write_bytes(b'foreign evidence')
        plan=dict(source_hashes={str(path):collect.sha256(path)})
        with self.assertRaisesRegex(ValueError,'outside workspace'):pipeline.snapshot_sources(self.out,plan)
        self.assertEqual(path.read_bytes(),b'foreign evidence')


if __name__ == '__main__':
    unittest.main(verbosity=2)
