"""CPU-only task dispatch, strict pure policy, training admission and queue accounting."""
import contextlib
import io
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import eval_robotwin_wam as evaluation
from scripts import run_robotwin_checkpoint_queue as queue
from scripts import run_robotwin_posttrain_eval_pipeline as pipeline
from harness.robotwin_policy_worker import InfrastructureError, atomic_json, sha256


class TaskTests(unittest.TestCase):
    def args(self, *extra):
        return evaluation.parse_args(['--run-dir', '/unused/new', '--checkpoint', '/unused/c.pt',
            '--training-run-dir', '/unused', '--plan-only', *extra])

    def test_non_cabinet_config_and_path_validation(self):
        cfg, _ = evaluation.load_task_config('demo_clean', 'place_cans_plasticbox')
        self.assertEqual(cfg['task_name'], 'place_cans_plasticbox')
        self.assertEqual(cfg['planner_mode'], 'joint_targets')
        self.assertFalse(cfg['collect_data'])
        for task in ('../escape', 'place_cans_plasticbox.foo', 'no_such_task', None):
            with self.assertRaises(InfrastructureError):
                evaluation.validate_task(task)

    def test_pure_rejects_every_advisory_or_feedback_path(self):
        for flags in (['--target-head', 'x'], ['--vtt-bank', 'x'],
                      ['--vtt-adapter', 'x', '--vtt-bank', 'y'], ['--execution-mode', 'contact_refresh']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*flags)
        args = self.args()
        args.target_head = 'injected'
        with self.assertRaises(InfrastructureError):
            evaluation.make_client(args, Path('/unused'))

    def test_generic_snapshot_does_not_read_cabinet_actor_gt(self):
        class Env:
            def __getattr__(self, name):
                raise AssertionError('unexpected GT read ' + name)
        with patch.object(evaluation, 'snapshot', side_effect=AssertionError('cabinet diagnostics')):
            self.assertFalse(evaluation.task_snapshot(Env(), 'place_cans_plasticbox')['diagnostics_available'])
        evidence = evaluation.task_evidence('place_cans_plasticbox')
        self.assertNotIn('cabinet_joint_qpos', evidence)
        self.assertEqual(evidence['native_predicate'], 'envs.place_cans_plasticbox.place_cans_plasticbox.check_success')

    def test_native_environment_dispatch_uses_selected_module_and_class(self):
        env = SimpleNamespace(check_success=lambda: False)
        module = SimpleNamespace(place_cans_plasticbox=lambda: env)
        with patch.dict('sys.modules', {'torch': SimpleNamespace(set_num_threads=lambda n: None),
                                      'numpy': SimpleNamespace(__version__='1.26.4'),
                                      'cv2': SimpleNamespace(setNumThreads=lambda n: None)}), \
             patch.object(evaluation.importlib, 'import_module', return_value=module) as load:
            self.assertIs(evaluation.make_env('place_cans_plasticbox'), env)
        load.assert_called_once_with('envs.place_cans_plasticbox')

    def test_plan_task_is_from_training_contract(self):
        args = self.args()
        contract = {'config': {'dataset': {'task_name': 'place_cans_plasticbox', 'task_config': 'demo_clean'}}}
        with patch.object(evaluation, 'training_contract', return_value=(contract, {'source': {}}, {}, {})), \
             patch.object(evaluation, 'load_task_config', side_effect=RuntimeError('configuration seam')) as load:
            with self.assertRaisesRegex(RuntimeError, 'configuration seam'):
                evaluation.plan_evaluation(args)
        load.assert_called_once_with('demo_clean', 'place_cans_plasticbox')
        args.task_config = 'demo_randomized'
        with patch.object(evaluation, 'training_contract', return_value=(contract, {'source': {}}, {}, {})):
            with self.assertRaisesRegex(InfrastructureError, 'differs'):
                evaluation.plan_evaluation(args)

    def test_generic_pure_rollout_native_predicate_no_gt_policy(self):
        from scripts.test_robotwin_eval_bridge import InjectedEnv, InjectedPolicy, InjectedVideo
        env, policy = InjectedEnv(), InjectedPolicy()
        del env.cabinet
        del env.object
        args = self.args('--seeds', '9000001', '--max-steps', '2')
        with tempfile.TemporaryDirectory() as d, \
             patch.object(evaluation, 'snapshot', side_effect=AssertionError('cabinet diagnostic used')):
            result = evaluation.run_evaluation(args, {'task': 'place_cans_plasticbox', 'task_kwargs': {}}, Path(d),
                client_factory=lambda *a: policy, env_factory=lambda: env,
                video_factory=lambda *a: (InjectedVideo(), None), owner_check=lambda *a: None,
                guard_factory=lambda: SimpleNamespace(check=lambda: {}, peaks={}))
        self.assertEqual(result['task'], 'place_cans_plasticbox')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['scored_episodes'], 1)
        self.assertEqual(result['native_failures'], 1)
        self.assertNotIn('cabinet_joint_qpos', result['episodes'][0]['evidence'])

    def test_worker_task_mismatch_never_creates_scene(self):
        from scripts.test_robotwin_eval_bridge import InjectedPolicy
        policy = InjectedPolicy()
        policy.ready = {'runtime': {'task': 'put_object_cabinet'}}
        with tempfile.TemporaryDirectory() as d, patch.object(evaluation, 'make_env') as scene:
            result = evaluation.run_evaluation(self.args(), {'task': 'place_cans_plasticbox'}, Path(d),
                client_factory=lambda *a: policy, env_factory=scene, owner_check=lambda *a: None,
                guard_factory=lambda: SimpleNamespace(check=lambda: {}, peaks={}))
        scene.assert_not_called()
        self.assertIn('runtime task differs', result['error'])
        self.assertEqual(result['scored_episodes'], 0)

    def test_first_frame_plan_without_checkpoint_or_policy(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d)/'cfg.yaml'
            cfg.write_text('dataset:\n  task_name: place_cans_plasticbox\n  task_config: demo_clean\n')
            args = evaluation.parse_args(['--run-dir', str(Path(d)/'smoke'), '--training-config', str(cfg),
                 '--first-frame-only', '--seeds', '9000100', '--plan-only'])
            with patch.object(evaluation, 'make_client', side_effect=AssertionError('worker')):
                plan = evaluation.smoke_plan(args)
            self.assertEqual(plan['task'], 'place_cans_plasticbox')
            self.assertFalse(plan['policy_constructed'])
            self.assertEqual(plan['control_actions'], 0)

    def test_first_frame_runs_no_policy_or_actions(self):
        import numpy as np
        from scripts.test_robotwin_eval_bridge import InjectedEnv
        env = InjectedEnv()
        env._update_render = lambda: None
        env.take_action = lambda *a: self.fail("smoke attempted control")
        args = self.args('--seeds', '9000100')
        camera = SimpleNamespace(take_picture=lambda: None, get_picture=lambda name: np.ones((4, 6, 4)))
        with tempfile.TemporaryDirectory() as d, \
             patch.object(evaluation, 'make_client', side_effect=AssertionError('smoke constructed worker')), \
             patch.object(evaluation, 'make_camera', return_value=(camera, {})):
            result = evaluation.run_first_frame(args, {'task': 'place_cans_plasticbox', 'task_kwargs': {}}, Path(d),
                env_factory=lambda task: env, owner_check=lambda owner: None,
                guard_factory=lambda: SimpleNamespace(check=lambda: {}, peaks={}))
            self.assertEqual(len(list(Path(d).glob('initial_*.png'))), 4)
        self.assertEqual(result['status'], 'native_first_frame_complete')
        self.assertFalse(result['policy_constructed'])
        self.assertEqual(result['control_actions'], 0)
        self.assertEqual(result['scored_episodes'], 0)
        self.assertTrue(env.closed)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clean = dict(LoadState='loaded', ActiveState='inactive', SubState='dead', Result='success',
                          ExecMainCode='1', ExecMainStatus='0', MainPID='0')
        self.status(status='bounded_complete', step=30000, target=30000)

    def status(self, **data):
        atomic_json(self.root/'status.json', data)

    def gate(self, state=None, lock=None):
        return pipeline.training_gate(self.root, 'train-unit', state_reader=lambda u: state or self.clean,
                                      owner_reader=lambda: lock)

    def test_requires_all_three_success_exit_and_lock_conditions(self):
        self.assertTrue(self.gate()[0])
        self.assertTrue(self.gate(state={**self.clean, 'ActiveState': 'active', 'SubState': 'exited'})[0])
        self.assertFalse(self.gate(lock='training')[0])
        self.assertFalse(self.gate(state={**self.clean, 'ActiveState': 'active'})[0])
        self.assertFalse(self.gate(state={**self.clean, 'LoadState': 'not-found'})[0])
        self.assertFalse(self.gate(state={**self.clean, 'ExecMainCode': '0'})[0])
        self.assertFalse(self.gate(state={**self.clean, 'ExecMainStatus': '1'})[0])

    def test_30k_checkpoint_alone_not_training_success(self):
        (self.root/'checkpoint_step_030000.pt').touch()
        for status in ({}, {'status': 'running', 'step': 30000, 'target': 30000}):
            self.status(**status)
            self.assertFalse(self.gate()[0])

    def test_failed_interrupted_partial_and_wrong_targets_rejected(self):
        for data in ({'status': 'stopped'}, {'status': 'failed'},
                     {'status': 'bounded_complete', 'step': 25000, 'target': 25000},
                     {'status': 'complete', 'step': 50000, 'target': 50000}):
            self.status(**data)
            with self.assertRaises(pipeline.TrainingFailed):
                self.gate()
        self.status(status='bounded_complete', step=30000, target=30000)
        with self.assertRaises(pipeline.TrainingFailed):
            self.gate(state={**self.clean, 'Result': 'signal'})

    def test_selection_audit_complete_and_disjoint(self):
        audit = self.root/'selection.json'
        cfg = {'training': {'seed': 123, 'validation_seed': 456}}
        for data in ({}, {'complete': True, 'selection_seeds': []},
                     {'complete': True, 'selection_seeds': [789], 'note': 'all prior smoke runs'}):
            atomic_json(audit, data)
            with self.assertRaises(ValueError):
                pipeline.seed_audit(cfg, [789, 790, 791, 792, 793], audit)
        atomic_json(audit, dict(complete=True, selection_seeds=[10], note='all prior smoke/development'))
        result = pipeline.seed_audit(cfg, [789, 790, 791, 792, 793], audit)
        self.assertEqual(result['excluded_seeds'], [10, 123, 456])

    def test_milestone_admission_checks_dataset_overlap_and_hashes(self):
        (self.root/'seed.txt').write_text('11 12 13')
        cfg = {'config': {'dataset': {'task_name': 'place_cans_plasticbox'}}, 'config_sha256': 'config-hash'}
        cache = {'source': {'episodes': [{'source': str(self.root/'data/ep.hdf5')}]}}
        manifest = dict(training_run_dir=str(self.root), task='place_cans_plasticbox', seeds=[20,21,22,23,24],
                        selection_audit={'excluded_seeds': []}, training_config={'sha256': 'config-hash'})
        for step in (25000, 30000):
            (self.root/f'checkpoint_step_{step:06d}.pt').write_text(f'CPU synthetic checkpoint {step}')
        def contract(run, ck):
            step = int(ck.stem.split('_')[-1])
            return cfg, cache, {}, {'step': step, 'model_only': True, 'sha256': sha256(ck)}
        with patch.object(pipeline, 'training_contract', side_effect=contract):
            admitted = pipeline.checkpoint_admission(manifest)
            self.assertEqual(len(admitted['checkpoints']), 2)
            manifest['seeds'][0] = 11
            with self.assertRaisesRegex(ValueError, 'overlap'):
                pipeline.checkpoint_admission(manifest)
            manifest['seeds'][0] = 20
            cfg['config_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                pipeline.checkpoint_admission(manifest)

    def test_wait_loop_rechecks_sources_status_and_lock_before_admission(self):
        args = SimpleNamespace(wait_seconds=100, poll_seconds=1)
        manifest = dict(training_run_dir=str(self.root), training_unit='train')
        answers = [(False, {'reason': 'training'}), (False, {'reason': 'lock'}), (True, {'ready': True})]
        with patch.object(pipeline, 'verify_frozen') as verify, \
             patch.object(pipeline, 'training_gate', side_effect=answers) as gate, \
             patch.object(pipeline.time, 'sleep') as sleep:
            admitted = pipeline.wait_ready(args, manifest, {}, self.root)
        self.assertEqual(admitted, {'ready': True})
        self.assertEqual(gate.call_count, 3)
        self.assertEqual(verify.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_interruption_while_waiting_never_launches_evaluation(self):
        cfg = self.root/'cfg.yml'
        cfg.write_text('dataset:\n  task_name: place_cans_plasticbox\n')
        audit = self.root/'audit.json'
        atomic_json(audit, dict(complete=True, selection_seeds=[], note='CPU-only fixture'))
        out = self.root/'queue'
        argv = ['--training-config', str(cfg), '--training-run-dir', str(self.root), '--training-unit', 'train',
                '--run-dir', str(out), '--selection-seeds-file', str(audit), '--seeds', '31','32','33','34','35', '--execute']
        with patch.object(pipeline, 'training_gate', return_value=(False, {'reason': 'training'})), \
             patch.object(pipeline.time, 'sleep', side_effect=queue.StopQueue('synthetic interrupt')), \
             patch.object(queue, 'run_manifest') as launch:
            self.assertEqual(pipeline.main(argv), 1)
        launch.assert_not_called()
        result = json.loads((out/'pipeline_status.json').read_text())
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['counts']['25000']['not_started'], 5)
        self.assertEqual(result['counts']['30000']['native_failures'], 0)
        self.assertTrue((out/'archive_index.json').is_file())


class QueueAccountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ck = self.root/'checkpoint.pt'
        self.ck.write_text('synthetic CPU test only')

    def job(self, step=25000, seed=9000001):
        return dict(unit=f'cpu-{step}-{seed}', run_dir=str(self.root/f'{step}-{seed}'),
            checkpoint=str(self.ck), checkpoint_step=step, seed=seed,
            training_run_dir='/unused/new-training', task='place_cans_plasticbox', system_mode='pure')

    def result(self, job, success=False):
        return dict(status='completed', task=job['task'], scored_episodes=1, native_successes=int(success),
                    episodes=[dict(episode='episode_000', seed=job['seed'],
                        status='native_success' if success else 'step_limit',
                        evidence={'native_benchmark_success': success})])

    def test_counts_do_not_convert_invalid_or_interrupt_to_failure(self):
        jobs = [self.job(seed=s) for s in range(10, 15)]
        rows = [dict(jobs[0], status='invalid_initialization', native_successes=None),
                dict(jobs[0], status='valid_terminal', native_successes=1),
                dict(jobs[1], status='valid_terminal', native_successes=0),
                dict(jobs[2], status='interrupted', native_successes=None)]
        count = queue.counts(jobs, rows)['25000']
        self.assertEqual(count['attempted_episodes'], 4)
        self.assertEqual(count['native_failures'], 1)
        self.assertEqual(count['invalid_initializations'], 1)
        self.assertEqual(count['incomplete_attempts'], 1)
        self.assertEqual(count['not_started'], 2)
        self.assertIsNone(count['native_success_rate'])

    def test_real_video_required_for_scoring(self):
        job = self.job()
        result = self.result(job, True)
        self.assertEqual(queue.classify_result(job, result, {'Result': 'success'}), ('valid_terminal', 1))
        self.assertEqual(queue.classify_result(job, result, {'Result': 'success'}, True)[0], 'infrastructure_incomplete')
        directory = Path(job['run_dir'])/'episode_000'
        directory.mkdir(parents=True)
        (directory/'third_view.mp4').write_bytes(b'cpu-synthetic-file-validation-fixture')
        atomic_json(directory/'third_view_video.json', dict(complete=True, synthetic=False, seed=job['seed'], frames=3))
        self.assertEqual(queue.classify_result(job, result, {'Result': 'success'}, True), ('valid_terminal', 1))
        self.assertEqual(queue.classify_result(job, result, {'Result': 'signal'}, True)[0], 'infrastructure_incomplete')

    def test_task_seed_status_and_native_count_must_agree(self):
        job = self.job()
        for changes in ({'task': 'put_object_cabinet'}, {'native_successes': 1}, {'status': 'infrastructure_error'}):
            result = {**self.result(job), **changes}
            self.assertEqual(queue.classify_result(job, result, {'Result': 'success'})[0], 'infrastructure_incomplete')
        result = self.result(job)
        result['episodes'][0]['seed'] += 1
        self.assertEqual(queue.classify_result(job, result, {'Result': 'success'})[0], 'infrastructure_incomplete')

    def simulate(self, jobs, launch, **extra):
        manifest = {'jobs': jobs, 'source_sha256': {}, **extra}
        calls = {}
        def states(unit):
            calls[unit] = calls.get(unit, 0) + 1
            return ({'LoadState': 'not-found', 'ActiveState': 'inactive'} if calls[unit] == 1
                    else {'LoadState': 'loaded', 'ActiveState': 'inactive', 'Result': 'success'})
        with patch.object(queue, 'unit_state', side_effect=states), \
             patch.object(queue.subprocess, 'run', side_effect=launch), \
             patch.object(queue, 'log_progress'):
            code = queue.run_manifest(manifest, self.root/'manifest.json')
        return code, json.loads((self.root/'queue_status.json').read_text())

    def test_invalid_retry_retains_same_seed_and_both_records(self):
        job = self.job()
        launches = []
        def launch(cmd, **kw):
            launches.append(cmd)
            run = Path(cmd[cmd.index('--run-dir') + 1])
            run.mkdir(parents=True)
            result = self.result(job)
            if len(launches) == 1:
                result.update(scored_episodes=0, native_successes=0,
                              episodes=[dict(seed=job['seed'], status='instance_setup_failed')])
            atomic_json(run/'result.json', result)
            return SimpleNamespace(returncode=0)
        code, state = self.simulate([job], launch, max_invalid_retries=1)
        self.assertEqual(code, 0)
        self.assertEqual([x[x.index('--seeds')+1] for x in launches], [str(job['seed'])]*2)
        self.assertNotEqual(launches[0][launches[0].index('--run-dir')+1], launches[1][launches[1].index('--run-dir')+1])
        self.assertEqual(len(state['jobs']), 2)
        self.assertEqual(state['counts']['25000']['invalid_initializations'], 1)
        self.assertEqual(len(list(self.root.glob('*_receipt.json'))), 2)

    def test_interrupt_does_not_retry_and_keeps_receipt(self):
        job = self.job()
        calls = []
        def launch(cmd, **kw):
            calls.append(cmd)
            if cmd[0] == 'systemd-run':
                raise queue.StopQueue('synthetic interruption')
            return SimpleNamespace(returncode=0)
        code, state = self.simulate([job, self.job(seed=9000002)], launch)
        self.assertEqual(code, 1)
        self.assertEqual(state['jobs'][0]['status'], 'interrupted')
        self.assertEqual(state['counts']['25000']['native_failures'], 0)
        self.assertEqual(state['counts']['25000']['not_started'], 1)
        self.assertTrue(any(c[:3] == ['systemctl', '--user', 'stop'] for c in calls))
        self.assertTrue((self.root/(job['unit']+'_receipt.json')).exists())

    def test_two_infrastructure_failures_stop_remaining_jobs(self):
        jobs = [self.job(seed=s) for s in range(10, 15)]
        code, state = self.simulate(jobs, lambda *a, **kw: SimpleNamespace(returncode=0))
        self.assertEqual(code, 1)
        self.assertEqual(state['counts']['25000']['incomplete_attempts'], 2)
        self.assertEqual(state['counts']['25000']['not_started'], 3)
        self.assertEqual(state['counts']['25000']['native_failures'], 0)

    def test_five_by_two_serial_order_and_shared_seeds(self):
        seeds = list(range(901, 906))
        audit, cfg = self.root/'audit.json', self.root/'cfg.yaml'
        atomic_json(audit, dict(complete=True, selection_seeds=[20], note='all CPU fixture selection seeds'))
        cfg.write_text('dataset:\n  task_name: place_cans_plasticbox\n')
        args = pipeline.parse_args(['--training-config', str(cfg), '--training-run-dir', str(self.root),
            '--training-unit', 'train', '--run-dir', str(self.root/'archive'), '--selection-seeds-file', str(audit),
            '--seeds', *map(str, seeds)])
        manifest = pipeline.build_manifest(args, {'dataset': {'task_name': 'place_cans_plasticbox'}})
        self.assertEqual([(j['checkpoint_step'], j['seed']) for j in manifest['jobs']],
                         [(step, seed) for step in (25000, 30000) for seed in seeds])
        for job in manifest['jobs']:
            self.assertTrue(job['record_onboard'])
            self.assertEqual(job['max_steps'], 800)
            self.assertEqual(job['system_mode'], 'pure')
            self.assertEqual(job['settle_steps'], 0)
            self.assertEqual(job['training_run_dir'], str(self.root))
        with patch.object(pipeline, 'training_gate', side_effect=AssertionError('must not query unit in dry-run')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(['--training-config', str(cfg), '--training-run-dir', str(self.root),
                '--training-unit', 'train', '--run-dir', str(self.root/'archive'), '--selection-seeds-file', str(audit),
                '--seeds', *map(str, seeds)]), 0)
        self.assertFalse((self.root/'archive').exists())

    def test_retained_child_exit_is_archived_then_stopped_without_waiting(self):
        job = self.job()
        calls = []
        states = [{'LoadState': 'not-found', 'ActiveState': 'inactive'},
                  {'LoadState': 'loaded', 'ActiveState': 'active', 'SubState': 'exited',
                   'Result': 'success', 'ExecMainStatus': '0', 'MainPID': '0'}]
        def run(cmd, **kw):
            calls.append(cmd)
            if cmd[0] == 'systemd-run':
                path = Path(job['run_dir'])
                path.mkdir()
                atomic_json(path/'result.json', self.result(job))
            else:
                self.assertTrue((self.root/(job['unit']+'_receipt.json')).exists())
            return SimpleNamespace(returncode=0)
        with patch.object(queue, 'unit_state', side_effect=states), \
             patch.object(queue.subprocess, 'run', side_effect=run), patch.object(queue, 'log_progress'), \
             patch.object(queue.time, 'sleep', side_effect=AssertionError('completed unit was treated as running')):
            self.assertEqual(queue.run_manifest({'jobs': [job], 'source_sha256': {}}, self.root/'m.json'), 0)
        self.assertIn('--property=RemainAfterExit=yes', calls[0])
        self.assertEqual(calls[1], ['systemctl', '--user', 'stop', job['unit']])


if __name__ == '__main__':
    unittest.main()
