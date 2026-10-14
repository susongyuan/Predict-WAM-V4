#!/usr/bin/env python3
"""CPU-only supervisor/interface regression tests; no real trainer/evaluator imports.

    python3 -B -m unittest scripts.test_robotwin_multitask_pipeline -v

All tests are ordinary regressions, including the previously reproduced defects.
The legacy --strict-known-risks CLI flag remains a no-op compatibility alias;
normal execution already enforces every assertion. Progress/document I/O errors
are advisory: evaluation continues and reporting_errors preserves the exception.
Machine-readable evidence writes remain strict and fail-closed.
Only temporary files are written; subprocesses, locks, signals, document/progress
updates and native dependencies are mocked. Simulator/trainer/evaluator
implementations are never imported.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_robotwin_multitask_pipeline as pipeline

# The checker changes display/thread environment on import; restore it in tests.
with patch.dict(os.environ):
    interface = importlib.import_module('scripts.check_robotwin_multitask_interface')


class Clock:
    def __init__(self):
        self.current = 0.0
        self.on_sleep = None

    def monotonic(self):
        return self.current

    def sleep(self, seconds):
        self.current += seconds
        if self.on_sleep:
            self.on_sleep()


class Child:
    """Bounded fake Popen child; None polls mean alive, no process is spawned."""
    def __init__(self, polls=(0,), on_exit=None, wait_timeout=False):
        self.pid = 424242
        self.polls = iter(polls)
        self.last_poll = None
        self.returncode = None
        self.on_exit = on_exit
        self.wait_timeout = wait_timeout
        self.wait_calls = []

    def poll(self):
        self.last_poll = next(self.polls, self.last_poll)
        self.returncode = self.last_poll
        if self.returncode is not None and self.on_exit:
            callback, self.on_exit = self.on_exit, None
            callback()
        return self.returncode

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        if self.wait_timeout and len(self.wait_calls) == 1:
            raise subprocess.TimeoutExpired('mock-child', timeout)
        self.returncode = -signal.SIGTERM
        return self.returncode


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='multitask-supervisor-cpu-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'run'
        self.out.mkdir()
        self.source = self.root / 'frozen_source.py'
        self.source.write_text('# synthetic immutable source\n')
        self.clock = Clock()
        self.lock_name = None
        self.lock = SimpleNamespace(acquire=Mock(side_effect=self.acquire),
                                    release=Mock(side_effect=self.release),
                                    owner=Mock(side_effect=lambda: self.lock_name))
        self.progress = Mock()
        self.docs = Mock()
        self.popen = Mock(side_effect=self.default_child)
        self.killpg = Mock()
        self.disk = Mock(return_value=SimpleNamespace(free=1000 * 1024**3))
        self.publications = []
        real_atomic = pipeline.atomic

        def capture_atomic(path, value):
            self.publications.append((Path(path), copy.deepcopy(value)))
            real_atomic(path, value)

        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        for name, value in [('ROOT', self.root), ('gpu_lock', self.lock),
                            ('append_progress', self.progress), ('update_doc', self.docs),
                            ('time', self.clock), ('atomic', capture_atomic)]:
            self.patches.enter_context(patch.object(pipeline, name, value))
        self.patches.enter_context(patch.object(pipeline.subprocess, 'Popen', self.popen))
        # Deny subprocess.run too: accidental GPU/systemd probes must fail the test.
        self.patches.enter_context(patch.object(pipeline.subprocess, 'run',
            side_effect=AssertionError('real subprocess.run is forbidden')))
        self.patches.enter_context(patch.object(pipeline.os, 'killpg', self.killpg))
        self.patches.enter_context(patch.object(pipeline.signal, 'signal'))
        self.patches.enter_context(patch.object(pipeline.shutil, 'disk_usage', self.disk))
        self.manifest = dict(out=str(self.out), gpu_lock_owner='cpu-test',
                             source_sha256={str(self.source): pipeline.sha(self.source)},
                             reserve_disk_bytes=300 * 1024**3,
                             environment={'CPU_TEST_SENTINEL': '1'},
                             stages=[self.stage(n) for n in ('cache', 'train', 'eval')])
        self.manifest_path = self.root / 'manifest.json'
        self.summary = {'verified_complete': True, 'total_episodes': 50}

    def acquire(self, name, **kwargs):
        if self.lock_name is not None:
            raise AssertionError('fake GPU lock already owned')
        self.lock_name = name

    def release(self, name):
        if self.lock_name == name:
            self.lock_name = None
            return True
        return False

    def stage(self, name, **extra):
        return dict(name=name, command=['NEVER_EXECUTE', name],
                    log=str(self.out / 'logs' / (name + '.log')),
                    timeout_seconds=60, **extra)

    def write_json(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def default_child(self, command, **kwargs):
        def complete():
            if command[-1] == 'train':
                self.write_json(self.out/'train/status.json', {'status': 'completed'})
            elif command[-1] == 'eval':
                self.write_json(self.out/'eval/summary.json', self.summary)
        return Child(on_exit=complete)

    def run_pipeline(self):
        self.write_json(self.manifest_path, self.manifest)
        return pipeline.run(self.manifest_path)

    def run_stage(self, stage=None):
        self.state = dict(status='running', stages=[], _started=0.0)
        return pipeline.run_stage(stage or self.manifest['stages'][0],
                                  self.manifest, self.state, self.out)

    def completion(self):
        return pipeline.read(self.out/'completion.json')

    def assert_failed(self):
        value = self.completion()
        self.assertEqual(value['status'], 'failed')
        self.assertNotEqual(value['exit_code'], 0)
        self.assertIs(value['verified_complete'], False)
        self.assertIs(value['gpu_lock_released'], True)

    def test_train_automatically_hands_off_to_eval_and_publishes_complete(self):
        self.assertEqual(self.run_pipeline(), 0)
        self.assertEqual([c.args[0][-1] for c in self.popen.call_args_list],
                         ['cache', 'train', 'eval'])
        result = self.completion()
        self.assertEqual(result['status'], 'completed')
        self.assertIs(result['verified_complete'], True)
        self.assertIs(result['gpu_lock_released'], True)
        status = pipeline.read(self.out/'pipeline_status.json')
        self.assertEqual([s['status'] for s in status['stages']], ['complete'] * 3)
        self.assertNotIn('_started', status)
        self.assertGreaterEqual(self.progress.call_count, 8)
        self.assertEqual(self.docs.call_count, 2)
        for item in self.popen.call_args_list:
            self.assertTrue(item.kwargs['start_new_session'])
            self.assertEqual(item.kwargs['env']['CPU_TEST_SENTINEL'], '1')
            self.assertEqual(item.kwargs['cwd'], self.root)

    def test_changed_source_before_launch_never_acquires_gpu_or_spawns(self):
        self.source.write_text('# changed\n')
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()
        self.popen.assert_not_called()
        self.lock.acquire.assert_not_called()
        self.assertEqual(self.completion()['source_changes'], [str(self.source)])

    def test_deleted_source_is_detected(self):
        self.source.unlink()
        self.assertEqual(pipeline.changes(self.manifest), [str(self.source)])

    def test_source_mutation_during_stage_stops_child(self):
        child = Child(polls=(None,))
        self.popen.side_effect = None
        self.popen.return_value = child
        self.clock.on_sleep = lambda: self.source.write_text('# mutation\n')
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            self.run_stage()
        self.killpg.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertEqual(self.state['stages'][0]['status'], 'failed')

    def test_source_mutation_after_training_blocks_eval(self):
        def create(command, **kwargs):
            if command[-1] == 'train':
                return Child(on_exit=lambda: self.source.write_text('# changed during train exit\n'))
            return self.default_child(command, **kwargs)
        self.popen.side_effect = create
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()
        self.assertEqual([c.args[0][-1] for c in self.popen.call_args_list], ['cache', 'train'])

    def test_final_source_recheck_rejects_mutation_on_eval_exit(self):
        def create(command, **kwargs):
            child = self.default_child(command, **kwargs)
            if command[-1] == 'eval':
                complete = child.on_exit
                def finish():
                    complete()
                    self.source.write_text('# mutation at final exit\n')
                child.on_exit = finish
            return child
        self.popen.side_effect = create
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()

    def test_timeout_terminates_process_group_and_publishes_failure(self):
        child = Child(polls=(None,))
        self.popen.side_effect = None
        self.popen.return_value = child
        self.manifest['stages'][0]['timeout_seconds'] = 1
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()
        self.killpg.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertEqual(child.wait_calls, [45])
        status = pipeline.read(self.out/'pipeline_status.json')
        self.assertEqual(status['stages'][0]['status'], 'failed')
        self.assertIn('TimeoutError', status['stages'][0]['error'])

    def test_timeout_escalates_to_sigkill_when_term_ignored(self):
        child = Child(polls=(None,), wait_timeout=True)
        pipeline.stop_child(child)
        self.assertEqual(self.killpg.call_args_list,
                         [call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGKILL)])
        self.assertEqual(child.wait_calls, [45, 15])

    def test_already_exited_child_not_signalled(self):
        pipeline.stop_child(Child())
        self.killpg.assert_not_called()

    def test_risk_child_exit_between_poll_and_term_does_not_mask_cleanup(self):
        self.killpg.side_effect = ProcessLookupError('child already exited')
        pipeline.stop_child(Child(polls=(None,)))

    def test_disk_reserve_boundary_and_default(self):
        reserve = 300 * 1024**3
        self.disk.return_value.free = reserve
        self.assertEqual(pipeline.check_disk({}), reserve)
        self.disk.return_value.free = reserve - 1
        with self.assertRaisesRegex(RuntimeError, 'disk reserve violated'):
            pipeline.check_disk({})
        self.assertEqual(pipeline.check_disk({'reserve_disk_bytes': reserve-1}), reserve-1)

    def test_low_disk_before_stage_never_spawns(self):
        self.disk.return_value.free = 299 * 1024**3
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()
        self.popen.assert_not_called()

    def test_disk_exhaustion_during_stage_terminates_child(self):
        child = Child(polls=(None,))
        self.popen.side_effect = None
        self.popen.return_value = child
        self.clock.on_sleep = lambda: setattr(self.disk.return_value, 'free', 1)
        with self.assertRaisesRegex(RuntimeError, 'disk reserve violated'):
            self.run_stage()
        self.killpg.assert_called_once_with(child.pid, signal.SIGTERM)

    def test_nonzero_train_exit_never_runs_eval(self):
        def create(command, **kwargs):
            return Child(polls=(7,)) if command[-1] == 'train' else self.default_child(command, **kwargs)
        self.popen.side_effect = create
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()
        self.assertEqual([c.args[0][-1] for c in self.popen.call_args_list], ['cache', 'train'])

    def test_required_training_json_gate_blocks_eval_on_mismatch(self):
        self.manifest['stages'][1]['required_json'] = [dict(
            path=str(self.out/'train/status.json'), equals={'status': 'not-the-emitted-status'})]
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()
        self.assertEqual(self.popen.call_count, 2)
        self.assertIn('completion gate failed', self.completion()['error'])

    def test_required_training_json_success_permits_eval(self):
        self.manifest['stages'][1]['required_json'] = [dict(
            path=str(self.out/'train/status.json'), equals={'status': 'completed'})]
        self.assertEqual(self.run_pipeline(), 0)
        self.assertEqual(self.popen.call_count, 3)

    def test_missing_final_summary_fails_closed_even_when_children_exit_zero(self):
        self.popen.side_effect = lambda *a, **k: Child()
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()

    def test_false_verified_complete_fails_closed(self):
        self.summary['verified_complete'] = False
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()

    def test_absent_verified_complete_fails_closed(self):
        self.summary = {'status': 'completed', 'total_episodes': 50}
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()

    def test_risk_string_false_is_not_verified_complete(self):
        self.summary['verified_complete'] = 'false'
        self.assertNotEqual(self.run_pipeline(), 0)
        self.assert_failed()

    def test_risk_numeric_one_is_not_boolean_completion_gate(self):
        self.summary['verified_complete'] = 1
        self.manifest['stages'][2]['required_json'] = [dict(
            path=str(self.out/'eval/summary.json'), equals={'verified_complete': True})]
        self.assertNotEqual(self.run_pipeline(), 0)
        self.assert_failed()

    def test_malformed_final_summary_fails_closed(self):
        def create(command, **kwargs):
            if command[-1] == 'eval':
                def corrupt():
                    target = self.out/'eval/summary.json'
                    target.parent.mkdir(parents=True)
                    target.write_text('{truncated')
                return Child(on_exit=corrupt)
            return self.default_child(command, **kwargs)
        self.popen.side_effect = create
        self.assertEqual(self.run_pipeline(), 1)
        self.assert_failed()

    def test_existing_claim_refuses_restart_without_overwriting_old_result(self):
        self.write_json(self.out/'pipeline_claim.json', {'pid': 99})
        prior = {'status': 'completed', 'evidence': 'must-preserve'}
        self.write_json(self.out/'completion.json', prior)
        before = (self.out/'completion.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_pipeline()
        self.assertEqual((self.out/'completion.json').read_bytes(), before)
        self.popen.assert_not_called()
        self.lock.acquire.assert_not_called()

    def test_existing_stage_log_is_never_truncated(self):
        path = Path(self.manifest['stages'][0]['log'])
        path.parent.mkdir()
        path.write_bytes(b'original evidence\n')
        with self.assertRaises(FileExistsError):
            self.run_stage()
        self.assertEqual(path.read_bytes(), b'original evidence\n')
        self.popen.assert_not_called()

    def test_risk_existing_completion_without_claim_is_preserved(self):
        self.write_json(self.out/'completion.json', {'status': 'old-run', 'evidence': 42})
        before = (self.out/'completion.json').read_bytes()
        try:
            self.run_pipeline()
        except (FileExistsError, RuntimeError):
            pass
        self.assertEqual((self.out/'completion.json').read_bytes(), before)
        self.popen.assert_not_called()

    def test_risk_stale_verified_summary_is_not_reused_as_current_eval(self):
        self.write_json(self.out/'eval/summary.json', self.summary)
        # All commands return zero but none produces current evaluation evidence.
        self.popen.side_effect = lambda *a, **k: Child()
        self.manifest['stages'][2]['required_json'] = [dict(
            path=str(self.out/'eval/summary.json'), equals={'verified_complete': True})]
        try:
            code = self.run_pipeline()
        except (FileExistsError, RuntimeError):
            return  # Rejecting contaminated output before acquisition is also safe.
        self.assertNotEqual(code, 0)
        self.assert_failed()

    def test_risk_terminal_doc_exception_cannot_publish_failed_but_verified(self):
        def publish(text):
            if '训练与评测已完成' in text:
                raise OSError('synthetic status document write failure')
        self.docs.side_effect = publish
        code = self.run_pipeline()
        result = self.completion()
        self.assertEqual(result['status'] == 'completed', result['verified_complete'])
        self.assertEqual(code == 0, result['status'] == 'completed')

    def test_progress_failure_after_training_continues_auto_eval_and_records_error(self):
        # Advisory reporting must not block train -> eval or lose the I/O exception.
        error = OSError('synthetic progress.md failure after completed training')
        def append(message):
            if '完成 `train`' in message:
                raise error
        self.progress.side_effect = append
        self.assertEqual(self.run_pipeline(), 0)
        self.assertEqual([c.args[0][-1] for c in self.popen.call_args_list],
                         ['cache', 'train', 'eval'])
        self.assertEqual(pipeline.read(self.out/'train/status.json')['status'], 'completed')
        self.assertIs(pipeline.read(self.out/'eval/summary.json')['verified_complete'], True)
        result = self.completion()
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['status'], 'completed')
        self.assertIs(result['verified_complete'], True)
        self.assertIs(result['gpu_lock_released'], True)
        status = pipeline.read(self.out/'pipeline_status.json')
        self.assertEqual(status['status'], 'completed')
        self.assertEqual([stage['status'] for stage in status['stages']], ['complete'] * 3)
        self.assertEqual(status['reporting_errors'], result['reporting_errors'])
        self.assertEqual(len(result['reporting_errors']), 1)
        self.assertEqual(result['reporting_errors'][0]['error'], repr(error))

    def test_live_state_is_published_and_large_episode_detail_omitted(self):
        self.write_json(self.out/'train/status.json',
                        {'step': 12000, 'episodes': ['large'], 'commands': ['private'], 'loss': .2})
        child = Child(polls=(None, 0))
        self.popen.side_effect = None
        self.popen.return_value = child
        self.run_stage()
        live = [value for path, value in self.publications
                if path.name == 'pipeline_status.json' and 'disk_free_gib' in value]
        self.assertTrue(live)
        self.assertEqual(live[0]['detail']['training'], {'step': 12000, 'loss': .2})
        self.assertEqual(live[0]['stages'][0]['pid'], child.pid)
        self.assertNotIn('_started', live[0])
        self.assertFalse(list(self.out.rglob('*.tmp')))

    def test_malformed_detail_is_ignored(self):
        target = self.out/'train/status.json'
        target.parent.mkdir()
        target.write_text('{truncated')
        self.assertEqual(pipeline.summarize_detail(self.out), {})

    def test_risk_valid_json_nonobject_status_does_not_kill_training(self):
        self.write_json(self.out/'train/status.json', [])
        self.assertEqual(pipeline.summarize_detail(self.out), {})

    def test_risk_release_error_does_not_prevent_terminal_status_publication(self):
        self.lock.release.side_effect = OSError('synthetic lock unlink failure')
        try:
            self.run_pipeline()
        except OSError:
            pass
        self.assertTrue((self.out/'completion.json').exists())
        result = self.completion()
        self.assertIs(result['gpu_lock_released'], False)


class Array:
    """Tiny numpy-free object supporting only the checker's CPU contract."""
    def __init__(self, shape, *, dtype='uint8', std=10.0, finite=True):
        self.shape, self.dtype, self.finite = shape, dtype, finite
        self.ndim = len(shape)
        self._std = std

    def std(self):
        return self._std

    def tolist(self):
        return [0.0] * self.shape[0]


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='multitask-interface-cpu-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root/'admission'
        self.rgb = {name: Array((8, 8, 3)) for name in ('head', 'left', 'right')}
        self.command = Array((14,), dtype='float32')
        self.updated = self.command
        self.action_delta = 1
        self.observation_calls = 0
        self.config = {'render_denoiser': 'default'}
        self.env = SimpleNamespace(cameras=object(), robot=object(), take_action_cnt=0,
                                   step_lim=100, setup_demo=Mock(), get_obs=Mock(return_value='raw'),
                                   check_success=Mock(side_effect=[False, False]))
        self.close_env = Mock()
        self.make_env = Mock(return_value=self.env)
        self.owner = Mock(return_value='cpu-test')
        self.atomic_json = Mock(side_effect=lambda path, value: Path(path).write_text(json.dumps(value)))
        self.imread_write = Mock(return_value=True)
        self.adapter = SimpleNamespace(step=Mock(side_effect=self.hold),
                                       last_topp_diagnostics={'cpu_mock': True})
        self.fake_modules = {
            'scripts.eval_robotwin_wam': SimpleNamespace(
                load_task_config=Mock(return_value=(self.config, None)), make_env=self.make_env,
                close_env=self.close_env, ROBOTWIN=self.root),
            'harness.robotwin_qpos_adapter': SimpleNamespace(
                NativeQposAdapter=Mock(return_value=self.adapter), policy_observation=self.observe),
            'harness.robotwin_eval_evidence': SimpleNamespace(joint_bounds=Mock(return_value=([], None))),
            'harness.robotwin_policy_worker': SimpleNamespace(atomic_json=self.atomic_json),
            'harness.robotwin_camera_readback': SimpleNamespace(SerialColorReadback=Mock()),
            'scripts.gpu_lock': SimpleNamespace(owner=self.owner),
            'numpy': SimpleNamespace(uint8='uint8', isfinite=lambda a: SimpleNamespace(all=lambda: a.finite)),
            'cv2': SimpleNamespace(imwrite=self.imread_write, cvtColor=lambda a, _: a, COLOR_RGB2BGR=123),
        }
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.dict(sys.modules, self.fake_modules))
        self.chdir = self.patches.enter_context(patch.object(interface.os, 'chdir'))
        self.patches.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.patches.enter_context(patch.object(subprocess, 'Popen',
            side_effect=AssertionError('real subprocess forbidden in interface test')))
        self.patches.enter_context(patch.object(subprocess, 'run',
            side_effect=AssertionError('real subprocess forbidden in interface test')))

    def observe(self, raw):
        self.observation_calls += 1
        return {'rgb': self.rgb, 'commanded_joint_gripper_state':
                self.command if self.observation_calls == 1 else self.updated}

    def hold(self, command):
        self.assertIs(command, self.command)
        self.env.take_action_cnt += self.action_delta

    def run_interface(self):
        return interface.main(['--task', 'click_bell', '--out', str(self.out),
                               '--seed', '922000', '--gpu-lock-owner', 'cpu-test'])

    def result(self):
        return json.loads((self.out/'result.json').read_text())

    def test_native14_three_rgb_one_hold_success_and_cleanup(self):
        self.assertEqual(self.run_interface(), 0)
        result = self.result()
        self.assertEqual(result['status'], 'native_interface_smoke_complete')
        self.assertEqual(result['native_take_action_count_delta'], 1)
        self.assertEqual(len(result['views']), 3)
        self.assertEqual(self.imread_write.call_count, 3)
        self.assertFalse(result['record_video'])
        self.assertFalse(result['third_view_camera_created'])
        self.assertFalse(result['expert_called'])
        self.assertFalse(result['policy_constructed'])
        self.assertEqual(result['scored_episodes'], 0)
        self.assertEqual(self.env.setup_demo.call_args.kwargs['render_denoiser'], 'none')
        self.assertTrue(self.env.setup_demo.call_args.kwargs['is_test'])
        self.close_env.assert_called_once_with(self.env)

    def test_wrong_external_gpu_owner_fails_before_environment_creation(self):
        self.owner.return_value = 'foreign-job'
        self.assertEqual(self.run_interface(), 1)
        self.assertEqual(self.result()['status'], 'failed')
        self.make_env.assert_not_called()
        self.close_env.assert_called_once_with(None)

    def test_initially_successful_episode_rejected_without_hold(self):
        self.env.check_success.side_effect = [True]
        self.assertEqual(self.run_interface(), 1)
        self.adapter.step.assert_not_called()
        self.assertTrue(self.result()['native_initial_predicate'])
        self.close_env.assert_called_once_with(self.env)

    def test_not_three_rgb_views_rejected(self):
        del self.rgb['right']
        self.assertEqual(self.run_interface(), 1)
        self.adapter.step.assert_not_called()

    def test_blank_rgb_rejected(self):
        self.rgb['head']._std = 0
        self.assertEqual(self.run_interface(), 1)
        self.assertIn('blank onboard RGB', self.result()['error'])

    def test_bad_rgb_dtype_rejected(self):
        self.rgb['head'].dtype = 'float32'
        self.assertEqual(self.run_interface(), 1)
        self.assertIn('RGB format', self.result()['error'])

    def test_onboard_still_write_error_fails_and_closes_environment(self):
        self.imread_write.return_value = False
        self.assertEqual(self.run_interface(), 1)
        self.assertIn('onboard still', self.result()['error'])
        self.close_env.assert_called_once_with(self.env)

    def test_initial_non_native14_state_rejected(self):
        self.command.shape = (13,)
        self.assertEqual(self.run_interface(), 1)
        self.adapter.step.assert_not_called()

    def test_initial_nonfinite_state_rejected(self):
        self.command.finite = False
        self.assertEqual(self.run_interface(), 1)
        self.adapter.step.assert_not_called()

    def test_hold_must_advance_exactly_one_native_action(self):
        self.action_delta = 2
        self.assertEqual(self.run_interface(), 1)
        self.assertIn('HOLD interface failed', self.result()['error'])

    def test_post_hold_nonfinite_state_rejected(self):
        self.updated = Array((14,), finite=False)
        self.assertEqual(self.run_interface(), 1)
        self.assertIn('HOLD interface failed', self.result()['error'])

    def test_risk_post_hold_wrong_dimension_is_not_native14_admission(self):
        self.updated = Array((13,), dtype='float32')
        self.assertEqual(self.run_interface(), 1)
        self.assertEqual(self.result()['status'], 'failed')

    def test_cleanup_error_fails_admission(self):
        self.close_env.side_effect = OSError('synthetic cleanup failure')
        self.assertEqual(self.run_interface(), 1)
        self.assertEqual(self.result()['status'], 'failed')
        self.assertIn('cleanup_error', self.result())

    def test_existing_output_is_preserved(self):
        self.out.mkdir()
        target = self.out/'result.json'
        target.write_text('{"old_evidence": true}')
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_interface()
        self.assertEqual(target.read_bytes(), before)
        self.make_env.assert_not_called()


if __name__ == '__main__':
    # Backward compatibility: every regression is strict by default now.
    if '--strict-known-risks' in sys.argv:
        sys.argv.remove('--strict-known-risks')
    unittest.main(verbosity=2)
