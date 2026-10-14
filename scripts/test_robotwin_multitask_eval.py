#!/usr/bin/env python3
"""CPU-only admission/evidence/queue tests; no GPU, encoder, training, or Scene."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from harness import robotwin_multitask_policy_worker as worker
from harness.robotwin_policy_worker import InfrastructureError, ChunkPolicy, atomic_json, digest, identity, sha256
from scripts import eval_robotwin_multitask as evaluation

TASKS = ['put_bottles_dustbin', 'place_cans_plasticbox', 'open_laptop', 'click_bell', 'press_stapler']


def fixture(root):
    train, provenance, specs, files = {}, {}, [], []
    for index, task in enumerate(TASKS):
        directory = root / task
        directory.mkdir()
        source_file = directory / 'raw' / 'episode.hdf5'
        source_file.parent.mkdir()
        (source_file.parent / 'seed.txt').write_text('1\n2\n3\n')
        cache = {'source': {'task': task, 'archive': 'demo_clean.zip', 'episodes': [{'source': str(source_file)}]}}
        cache_id = worker.data_digest(cache)
        cache_dir = directory / 'cache' / cache_id
        cache_dir.mkdir(parents=True)
        atomic_json(cache_dir / 'cache_config.json', cache)
        files.append(identity(cache_dir / 'cache_config.json'))
        split = {'data_root': str(directory / 'data'), 'splits': {'train': ['e.hdf5'], 'val': ['v.hdf5'], 'heldout': ['h.hdf5']}}
        atomic_json(directory / 'split.json', split)
        specs.append({'task': task, 'split_manifest': str(directory / 'split.json'), 'feature_cache': str(directory / 'cache')})
        train[task] = [{'episode': 'e.hdf5', 'config_id': cache_id, 'length': 2}]
        provenance[task] = {'episodes': train[task], 'episode_count': 1, 'frame_count': 2,
            'split_manifest_sha256': worker.data_digest(split), 'cache_config_id': cache_id}
    prov = {'fit_split': 'train', 'heldout_used': False, 'tasks': provenance}
    condition = {task: np.random.default_rng(i).normal(size=1024).astype(np.float32).tolist() for i, task in enumerate(TASKS)}
    stats = {'normalization': {'action_repr': 'absolute', 'provenance': prov,
                'robotwin': {key: {'min': [-1.] * 14, 'max': [1.] * 14} for key in ('action', 'state')}},
        'temporal_stats': {'views': list(worker.MODEL_VIEWS), 'proprio_lags': [2], 'memory_lags': [2],
            'proprio_diff_std': np.ones((2, 14)).tolist(), 'memory_diff_std': np.ones((3, 2, 2048)).tolist(),
            'provenance': prov}}
    cfg = {'common': {'action_dim': 14, 'state_dim': 14, 'action_chunk_size': 32, 'num_inference_steps': 10},
        'dataset': {'cond_mode': 'train_task_vtt'}, 'training': {'max_steps': 20000},
        'model': {'views': list(worker.MODEL_VIEWS), 'use_task_cond': True,
                  'v3': {'proprio_lags': [2], 'memory_lags': [2]}}}
    contract = {'format': worker.FORMAT, 'tasks': specs, 'config': cfg, 'statistics': stats,
        'condition_bank': {'format': worker.BANK_FORMAT, 'source_split': 'train', 'provenance': prov, 'conditions': condition},
        'episodes': {'train': train}, 'cache_files': files, 'source_hashes': {},
        'heldout_used': False, 'encoder_loaded_for_training': False}
    checkpoint = root / 'checkpoint_step_020000.pt'
    checkpoint.write_bytes(b'CPU metadata fixture, not model tensors')
    atomic_json(checkpoint.with_name(checkpoint.name + '.sha256.json'), {'sha256': sha256(checkpoint),
        'size': checkpoint.stat().st_size, 'step': 20000, 'model_only': True})
    atomic_json(root / 'manifest.json', {'contract': contract, 'contract_sha256': digest(contract)})
    atomic_json(root / 'tasks.json', {'tasks': specs})
    return checkpoint, contract


class AdmissionTests(unittest.TestCase):
    def test_archive_derivation_and_conflicts(self):
        for source in ({'archive': 'demo_clean.zip'}, {'task_config': 'demo_clean'},
                       {'archive': 'demo_clean.zip', 'task_config': 'demo_clean'}):
            self.assertEqual(evaluation.source_task_config(source), 'demo_clean')
        self.assertEqual(evaluation.source_task_config({'archive': 'demo_randomized.zip'}), 'demo_randomized')
        for source in ({}, {'archive': 'unknown.zip'}, {'task_config': 'demo_randomized', 'archive': 'demo_clean.zip'}):
            with self.assertRaises(InfrastructureError):
                evaluation.source_task_config(source)

    def test_real_five_sources_and_official_limits(self):
        path = ROOT / 'outputs/runs/multitask5_wam_v3_vtt_20261008/tasks.json'
        if not path.exists():
            self.skipTest('registered task data not present')
        for row in json.loads(path.read_text())['tasks']:
            root = Path(json.loads(Path(row['split_manifest']).read_text())['data_root'])
            source = next(p / 'SOURCE.json' for p in (root, *root.parents) if (p / 'SOURCE.json').is_file())
            self.assertEqual(evaluation.source_task_config(json.loads(source.read_text())), 'demo_clean')
        self.assertEqual([evaluation.official_step_limit(t) for t in TASKS], [1700, 800, 700, 400, 400])

    def test_metadata_contract_and_task_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, contract = fixture(root)
            bundle = worker.load_contract(checkpoint, root / 'tasks.json')
            self.assertEqual(bundle['contract'], contract)
            for task in TASKS:
                value = worker.select_condition(contract, task)
                np.testing.assert_array_equal(value, np.asarray(contract['condition_bank']['conditions'][task], np.float32))
                value[:] = 0
                self.assertGreater(np.linalg.norm(worker.select_condition(contract, task)), 1)
            with self.assertRaises(InfrastructureError):
                worker.select_condition(contract, 'unknown_task')
            receipt = checkpoint.with_name(checkpoint.name + '.sha256.json')
            data = json.loads(receipt.read_text()); data['step'] = 19000; atomic_json(receipt, data)
            with self.assertRaises(InfrastructureError):
                worker.load_contract(checkpoint, root / 'tasks.json')

    def test_manifest_stats_and_split_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint, contract = fixture(root)
            invalid = copy.deepcopy(contract)
            invalid['statistics']['normalization']['provenance']['fit_split'] = 'val'
            with self.assertRaises(InfrastructureError):
                worker.validate_condition_statistics(invalid)
            invalid = copy.deepcopy(contract)
            invalid['condition_bank']['conditions'][TASKS[1]] = invalid['condition_bank']['conditions'][TASKS[0]]
            with self.assertRaises(InfrastructureError):
                worker.validate_condition_statistics(invalid)
            split = Path(contract['tasks'][0]['split_manifest']); split.write_text('{}')
            with self.assertRaises(InfrastructureError):
                worker.load_contract(checkpoint, root / 'tasks.json')

    def test_all_task_plans_archive_derived_and_seed_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint, _ = fixture(root)
            args = evaluation.parse_args(['--checkpoint', str(checkpoint), '--multitask-manifest', str(root / 'tasks.json'), '--out', str(root / 'out')])
            bundle = worker.load_contract(checkpoint, root / 'tasks.json')
            with patch.object(evaluation.native, 'load_task_config', return_value=({}, [])), patch.object(evaluation, 'asset_evidence', return_value={}):
                plan = evaluation.queue_plan(args, bundle)
                self.assertEqual(len(plan['queue']), 50)
                self.assertTrue(all(v['task_config'] == 'demo_clean' for v in plan['tasks'].values()))
                self.assertTrue(all(v['onboard_archive']['enabled'] for v in plan['tasks'].values()))
                self.assertTrue(all(not v['recording']['enabled'] for v in plan['tasks'].values()))
                (root / TASKS[0] / 'raw' / 'seed.txt').write_text('922001\n')
                with self.assertRaises(InfrastructureError):
                    evaluation.queue_plan(args, bundle)

    def test_import_is_torch_and_simulator_free(self):
        code = 'import sys; import scripts.eval_robotwin_multitask; assert "torch" not in sys.modules; assert "sapien" not in sys.modules'
        run = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_cli_no_video_or_seed_cherry_picking(self):
        base = ['--checkpoint', 'c', '--multitask-manifest', 'm', '--out', 'o']
        args = evaluation.parse_args(base)
        self.assertEqual(args.seeds, list(evaluation.SEEDS))
        self.assertEqual(args.camera_readback, 'serial')
        self.assertEqual(args.render_denoiser, 'none')
        for extra in (['--record-video'], ['--seeds', '1'], ['--execute']):
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                evaluation.parse_args(base + extra)
        args.gpu_lock_owner = 'pipeline'
        command = evaluation.child_command(args, 'open_laptop', evaluation.SEEDS[0], Path('/tmp/no-write'))
        self.assertIn('--gpu-lock-owner', command)
        self.assertNotIn('gpu_lock.py', command)
        self.assertEqual(command[command.index('--episode-task') + 1], 'open_laptop')

    def test_history_and_action_queue_reset_each_episode(self):
        class Context:
            def reset(self): self.rows = []
            def push(self, state, pool): self.rows.append(state.copy())
            def inputs(self): return self.rows
        ctx = Context()
        policy = ChunkPolicy(ctx, lambda rgb: (None, None), lambda f, states, seed, step: np.ones((32,14)))
        observation = {'rgb': {}, 'commanded_joint_gripper_state': np.zeros(14)}
        policy.reset('first', 1)
        policy.observe('first', 0, observation)
        policy.observe('first', 1, observation)
        self.assertEqual(len(ctx.rows), 2)
        self.assertGreater(len(policy.queue), 0)
        policy.reset('second', 2)
        self.assertEqual(ctx.rows, [])
        self.assertEqual(len(policy.queue), 0)
        self.assertEqual(policy.next_step, 0)
        with self.assertRaises(InfrastructureError): policy.observe('first', 2, observation)


class EvidenceTests(unittest.TestCase):
    def make_native_rollout(self, directory):
        from scripts.test_robotwin_eval_bridge import InjectedEnv, InjectedPolicy
        args = SimpleNamespace(checkpoint='/unused', multitask_manifest='/unused', gpu_lock_owner='test',
                               startup_timeout=10, policy_timeout=2, camera_readback='native', render_denoiser='none')
        task, seed = 'click_bell', evaluation.SEEDS[0]
        condition_hash = 'a' * 64
        runtime = {'task': task, 'checkpoint_sha256': 'b' * 64, 'contract_sha256': 'c' * 64,
            'statistics_sha256': 'd' * 64, 'condition_bank_sha256': 'e' * 64,
            'task_condition_sha256': condition_hash, 'dino_frozen': True, 'strict_load': True}
        policy = InjectedPolicy()
        policy.ready = {'runtime': runtime}
        original = policy.action
        def action(*a):
            return {**original(*a), 'task_condition': {'task': task, 'sha256': condition_hash}}
        policy.action = action
        plan = {'task': task, 'seeds': [seed], 'task_kwargs': {}, 'limits': {'control_steps': 3},
            'checkpoint_receipt': {'sha256': runtime['checkpoint_sha256']},
            **{key: runtime[key] for key in ('contract_sha256', 'statistics_sha256', 'condition_bank_sha256', 'task_condition_sha256')},
            'native_success_predicate': 'envs.click_bell.click_bell.check_success',
            'recording': {'enabled': False}, 'onboard_archive': {'enabled': True},
            'policy': {'views': ['head_camera', 'left_camera', 'right_camera']}}
        atomic_json(directory / 'plan.json', plan)
        class Guard:
            peaks = {}
            def check(self): return {}
        settings = evaluation.native_args(args, task, seed)
        settings.max_steps = 3
        with patch.object(evaluation.native, 'make_camera', side_effect=AssertionError('third camera forbidden')):
            result = evaluation.native.run_evaluation(settings, plan, directory, client_factory=lambda a, p: policy,
                env_factory=InjectedEnv, video_factory=evaluation.disabled_factory,
                owner_check=lambda owner: None, guard_factory=Guard)
        self.assertEqual(result['status'], 'completed', result)
        return result['episodes'][0]

    def test_native_evidence_verified_without_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.make_native_rollout(root)
            verification = evaluation.verify_native_evidence(root, row)
            self.assertTrue(verification['verified_complete'])
            self.assertEqual(verification['onboard_frames_verified'], 4)
            result = evaluation.collect_episode('click_bell', evaluation.SEEDS[0], root, {'returncode': 0, 'external_timeout': False})
            self.assertIs(result['native_success'], False)
            self.assertTrue(evaluation.score_rows([result])['verified_complete'])
            frame = root / row['episode'] / 'onboard' / 'observation_000001.npz'
            frame.unlink()
            result = evaluation.collect_episode('click_bell', evaluation.SEEDS[0], root, {'returncode': 0, 'external_timeout': False})
            self.assertIsNone(result['native_success'])
            self.assertEqual(result['status'], 'infrastructure_error')
            self.assertFalse(evaluation.score_rows([result])['verified_complete'])

    def test_timeout_and_action_corruption_stay_unscored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = self.make_native_rollout(root)
            result = evaluation.collect_episode('click_bell', evaluation.SEEDS[0], root, {'returncode': -9, 'external_timeout': True})
            self.assertEqual(result['status'], 'external_timeout')
            self.assertIsNone(result['native_success'])
            actions = root / row['episode'] / 'actions.jsonl'
            actions.write_text(actions.read_text().splitlines()[0] + '\n')
            result = evaluation.collect_episode('click_bell', evaluation.SEEDS[0], root, {'returncode': 0, 'external_timeout': False})
            self.assertIsNone(result['native_success'])
            self.assertIn('action pairs', result['error'])

    def test_serial_queue_never_retries_or_counts_infra_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(out=str(root), checkpoint='/unused', multitask_manifest='/unused',
                gpu_lock_owner='test', camera_readback='serial', render_denoiser='none', startup_timeout=1, policy_timeout=1)
            plan = {'checkpoint_sha256': 'a' * 64, 'contract_sha256': 'b' * 64,
                'tasks': {task: {'limits': {'external_wall_seconds': 1}} for task in TASKS},
                'queue': [{'task': task, 'seed': seed} for task in TASKS for seed in evaluation.SEEDS]}
            seen = []
            def launcher(command, log, timeout):
                task = command[command.index('--episode-task') + 1]
                seed = int(command[command.index('--episode-seed') + 1])
                seen.append((task, seed))
                return {'returncode': 2, 'external_timeout': len(seen) == 3}
            summary = evaluation.run_queue(args, plan, launcher=launcher)
            self.assertEqual(len(seen), 3)
            self.assertEqual(len(set(seen)), 3)
            self.assertEqual(summary['native_failures'], 0)
            self.assertEqual(summary['unscored_episodes'], 50)
            self.assertIsNone(summary['native_success_rate'])
            self.assertFalse(summary['verified_complete'])
            self.assertEqual(summary['status_counts']['external_timeout'], 1)
            self.assertEqual(summary['status'], 'stopped_after_three_unscored')
            self.assertEqual(summary['status_counts']['not_started'], 47)

    def test_wilson_and_valid_failures_do_not_trigger_infrastructure_stop(self):
        self.assertAlmostEqual(evaluation.wilson95(0, 10)[1], 0.2775328, places=6)
        self.assertIsNone(evaluation.score_rows([{'status': 'infrastructure_error', 'native_success': None}])['wilson_95'])
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=tmp, checkpoint='/unused', multitask_manifest='/unused', gpu_lock_owner='test',
                camera_readback='serial', render_denoiser='none', startup_timeout=1, policy_timeout=1)
            plan = {'checkpoint_sha256':'a'*64, 'contract_sha256':'b'*64,
                'tasks': {task: {'limits': {'external_wall_seconds':1}} for task in TASKS},
                'queue':[{'task':task,'seed':seed} for task in TASKS for seed in evaluation.SEEDS]}
            def collected(task, seed, out, execution):
                return dict(task=task, seed=seed, status='step_limit', native_success=False,
                            verification={'verified_complete':True})
            from unittest.mock import patch
            with patch.object(evaluation, 'collect_episode', side_effect=collected):
                result = evaluation.run_queue(args, plan, launcher=lambda *args: {'returncode':0})
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['native_failures'], 50)
            self.assertTrue(result['verified_complete'])
            self.assertEqual(result['native_success_rate'], 0)


if __name__ == '__main__':
    unittest.main()
