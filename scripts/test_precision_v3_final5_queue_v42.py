#!/usr/bin/env python3
"""Independent CPU mock tests for the V3 final-five waiting queue.

Never executes a GPU command, evaluator, simulator, trainer, or background unit.
All subprocess/process ownership and resource admission paths are mocked. Real
CPU plan validation only reads frozen declarations and hashes existing files.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_precision_v3_final5_v42 as q

GOOD_STATUS = dict(status='completed', step=30000, target_steps=30000)
GOOD_UNIT = dict(ActiveState='inactive', SubState='dead', MainPID='0', Result='success')


class Child:
    def __init__(self, polls=None, code=0):
        self.pid = 765432
        self.returncode = code
        self.polls = list(polls) if polls is not None else None
        self.alive = True
        self.wait_calls = []
    def poll(self):
        if self.polls is not None:
            return self.polls.pop(0) if self.polls else self.returncode
        return None if self.alive else self.returncode
    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self.alive = False
        return self.returncode


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v3-final5-mock-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.out = self.base / 'new-evaluation'
        self.train = self.base / 'train'
        self.train.mkdir()
        self.checkpoint = self.train / 'checkpoint_step_030000.pt'
        self.checkpoint.write_bytes(b'CPU mock placeholder, not a checkpoint')
        (self.train / 'manifest.json').write_text(json.dumps({'contract': {'mock': True}}))
        self.events, self.writes = [], []
        self.plan = dict(seeds=list(q.e.SEEDS), planned_episodes=5, out=str(self.out),
            checkpoint_expected_path=str(self.checkpoint), child_env={
                'LD_LIBRARY_PATH': '/pinned/nvidia_5959107/lib', 'CUDA_VISIBLE_DEVICES': '0',
                'P2_V3_THIRD_VIEW_SEED': '932126'}, frozen_sources={},
            extension_sources={str(q.RUNTIME): 'a' * 64}, preregistration={},
            preserved_original_results={}, action_postprocessing_sha256='a' * 64,
            interpretation='CPU mock; no policy success claim', commands=[
                dict(seed=s, out=str(self.out/q.e.TASK/f'seed_{s}'), command=['mock-evaluator', str(s)])
                for s in q.e.SEEDS])

    def write(self, path, value):
        self.writes.append((Path(path).name, deepcopy(value)))

    def execute_context(self, *, rows=None, failed_audit=False, leftover_gpu=False):
        """Mock every actionful seam; filesystem activity stays in a temporary directory."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        owner = {'name': None}
        jobs_calls = {'n': 0}
        def gpu_jobs(_):
            jobs_calls['n'] += 1
            return [{'pid': 444444}] if leftover_gpu and jobs_calls['n'] > 2 else []
        def acquire(name, **kwargs):
            self.events.append(('acquire', name)); owner['name'] = name
        def release(name):
            self.events.append(('release', name)); owner['name'] = None
        def wait(plan):
            self.events.append(('wait',)); return None
        def audit(plan):
            self.events.append(('audit',))
            if failed_audit:
                raise q.e.EvidenceError('Independent final audit is not evaluation_ready')
            return {'path': str(self.out/'audit.json'), 'sha256': 'a' * 64}
        def run(item, plan):
            self.events.append(('episode', item['seed']))
            row = rows.get(item['seed'], {}) if rows else {}
            return 2 if isinstance(row, dict) and row.get('status', '') in ('control_safety_abort', 'physics_safety_abort') else 0
        def classify(path, *args):
            seed = int(Path(path).parent.name.split('_')[-1])
            if rows and rows.get(seed) == 'unknown':
                raise q.e.EvidenceError('Unknown infrastructure result cannot be a valid task failure')
            row = dict(seed=seed, status='completed', full_task_success=False, observed_base_score=0)
            if rows and seed in rows:
                row.update(rows[seed])
            return row
        scheduler = SimpleNamespace(collection_resource_admission=Mock(return_value={'mock': True}),
                                    resource_ready=Mock(return_value=True))
        admission = SimpleNamespace(gpu_jobs=gpu_jobs, processes=lambda: [], scheduler_module=lambda: scheduler)
        stack.enter_context(patch.dict(sys.modules, {'scripts.run_zhuhai_task1_dr200_concurrent': admission}))
        stack.enter_context(patch.object(sys.modules['scripts'], 'run_zhuhai_task1_dr200_concurrent', admission, create=True))
        stack.enter_context(patch.object(q, 'OUT', self.out))
        stack.enter_context(patch.object(q, 'TRAIN', self.train))
        stack.enter_context(patch.object(q, 'verify', side_effect=lambda p: self.events.append(('verify',))))
        stack.enter_context(patch.object(q, 'wait_for_training', side_effect=wait))
        stack.enter_context(patch.object(q, 'audit_final_training', side_effect=audit))
        stack.enter_context(patch.object(q, 'checkpoint_metadata', return_value=dict(path=str(self.checkpoint), step=30000,
            sha256='a' * 64, size=self.checkpoint.stat().st_size)))
        stack.enter_context(patch.object(q, 'actual_trainers', return_value=[]))
        stack.enter_context(patch.object(q.gpu_lock, 'owner', side_effect=lambda: owner['name']))
        stack.enter_context(patch.object(q.gpu_lock, 'acquire', side_effect=acquire))
        stack.enter_context(patch.object(q.gpu_lock, 'release', side_effect=release))
        stack.enter_context(patch.object(q, 'run_once', side_effect=run))
        stack.enter_context(patch.object(q.e, 'classify', side_effect=classify))
        stack.enter_context(patch.object(q.e, 'write_json', side_effect=self.write))
        stack.enter_context(patch.object(q, 'log'))
        return admission, scheduler

    def test_finished_gate_requires_every_release_condition(self):
        self.assertTrue(q.finished_and_released(GOOD_STATUS, GOOD_UNIT, [], None))
        variants = [('status', 'training'), ('step', 29999), ('target_steps', 5000)]
        for key, value in variants:
            with self.subTest(status_field=key):
                self.assertFalse(q.finished_and_released(dict(GOOD_STATUS, **{key: value}), GOOD_UNIT, [], None))
        variants = [('ActiveState', 'active'), ('SubState', 'exited'), ('MainPID', '99'), ('Result', 'exit-code')]
        for key, value in variants:
            with self.subTest(unit_field=key):
                self.assertFalse(q.finished_and_released(GOOD_STATUS, dict(GOOD_UNIT, **{key: value}), [], None))
        self.assertFalse(q.finished_and_released(GOOD_STATUS, {'query_error': 'timeout'}, [], None))
        self.assertFalse(q.finished_and_released(GOOD_STATUS, GOOD_UNIT, [123], None))
        self.assertFalse(q.finished_and_released(GOOD_STATUS, GOOD_UNIT, [], 'training-owner'))

    def test_systemctl_observation_timeout_is_unknown(self):
        import subprocess
        with patch.object(q.subprocess, 'run', side_effect=subprocess.TimeoutExpired('systemctl', 10)):
            result = q.unit_state(q.TRAIN_UNIT)
        self.assertIn('query_error', result)
        self.assertNotIn('ActiveState', result)

    def test_wait_repolls_transient_status_read_failure_without_action(self):
        error = json.JSONDecodeError('partial transient', '', 0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(q, 'OUT', self.out))
            stack.enter_context(patch.object(q, 'verify'))
            read = stack.enter_context(patch.object(q.e, 'read_json', side_effect=[error, GOOD_STATUS]))
            stack.enter_context(patch.object(q, 'unit_state', return_value=GOOD_UNIT))
            stack.enter_context(patch.object(q, 'actual_trainers', return_value=[]))
            stack.enter_context(patch.object(q.gpu_lock, 'owner', return_value=None))
            acquire = stack.enter_context(patch.object(q.gpu_lock, 'acquire', side_effect=AssertionError('waiting must not take GPU')))
            stack.enter_context(patch.object(q.e, 'write_json', side_effect=self.write))
            stack.enter_context(patch.object(q.time, 'monotonic', return_value=0.))
            sleep = stack.enter_context(patch.object(q.time, 'sleep'))
            stack.enter_context(patch.object(q, 'log'))
            kill = stack.enter_context(patch.object(q.os, 'killpg', side_effect=AssertionError('waiting must never signal trainer')))
            q.wait_for_training(self.plan)
        self.assertEqual(read.call_count, 2)
        self.assertGreaterEqual(sleep.call_count, 1)
        acquire.assert_not_called(); kill.assert_not_called()

    def test_training_live_waits_then_returns_only_after_release(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(q, 'OUT', self.out))
            stack.enter_context(patch.object(q, 'verify'))
            stack.enter_context(patch.object(q.e, 'read_json', side_effect=[dict(status='training', step=4999, target_steps=30000), GOOD_STATUS]))
            stack.enter_context(patch.object(q, 'unit_state', side_effect=[dict(ActiveState='active', SubState='running', MainPID='123'), GOOD_UNIT]))
            stack.enter_context(patch.object(q, 'actual_trainers', side_effect=[[123], []]))
            stack.enter_context(patch.object(q.gpu_lock, 'owner', side_effect=['trainer-owner', None]))
            stack.enter_context(patch.object(q.e, 'write_json', side_effect=self.write))
            stack.enter_context(patch.object(q.time, 'monotonic', return_value=0.))
            sleep = stack.enter_context(patch.object(q.time, 'sleep'))
            stack.enter_context(patch.object(q, 'log'))
            kill = stack.enter_context(patch.object(q.os, 'killpg', side_effect=AssertionError('must not signal trainer')))
            q.wait_for_training(self.plan)
        self.assertEqual(sleep.call_count, 1); kill.assert_not_called()

    def test_owner_read_error_is_unknown_not_released(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(q, 'OUT', self.out))
            stack.enter_context(patch.object(q, 'verify'))
            stack.enter_context(patch.object(q.e, 'read_json', return_value=GOOD_STATUS))
            stack.enter_context(patch.object(q, 'unit_state', return_value=GOOD_UNIT))
            stack.enter_context(patch.object(q, 'actual_trainers', return_value=[]))
            owner = stack.enter_context(patch.object(q.gpu_lock, 'owner', side_effect=[OSError('transient lock read'), None]))
            stack.enter_context(patch.object(q.e, 'write_json', side_effect=self.write))
            stack.enter_context(patch.object(q.time, 'monotonic', return_value=0.))
            sleep = stack.enter_context(patch.object(q.time, 'sleep'))
            stack.enter_context(patch.object(q, 'log'))
            q.wait_for_training(self.plan)
        self.assertEqual(owner.call_count, 2)
        self.assertEqual(sleep.call_count, 1)

    def test_actual_trainer_scan_recognizes_script_and_module_commands(self):
        class Entry:
            def __init__(self, name, argv, unreadable=False):
                self.name, self.argv, self.unreadable = name, argv, unreadable
            def __truediv__(self, other):
                return self
            def read_bytes(self):
                if self.unreadable:
                    raise OSError('process disappeared during snapshot')
                return '\0'.join(self.argv).encode()
        entries = [Entry('123', ['python', '/work/scripts/train_zhuhai_wam_precision_v3.py']),
                   Entry('456', ['python', '-m', 'scripts.train_zhuhai_wam_precision_v2']),
                   Entry('789', ['python', '/work/scripts/monitor_precision_v3_training_cpu_v42.py']),
                   Entry('888', [], unreadable=True)]
        original_path = q.Path
        def path(value):
            return SimpleNamespace(iterdir=lambda: iter(entries)) if value == '/proc' else original_path(value)
        with patch.object(q, 'Path', side_effect=path):
            self.assertEqual(q.actual_trainers(), [123, 456])

    def test_failed_training_halts_without_restart_or_evaluation(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(q, 'OUT', self.out))
            stack.enter_context(patch.object(q, 'verify'))
            stack.enter_context(patch.object(q.e, 'read_json', return_value=dict(status='failed', step=77, target_steps=30000)))
            stack.enter_context(patch.object(q, 'unit_state', return_value=dict(GOOD_UNIT, ActiveState='failed', Result='exit-code')))
            stack.enter_context(patch.object(q, 'actual_trainers', return_value=[]))
            stack.enter_context(patch.object(q.gpu_lock, 'owner', return_value=None))
            stack.enter_context(patch.object(q.e, 'write_json', side_effect=self.write))
            run = stack.enter_context(patch.object(q.subprocess, 'Popen', side_effect=AssertionError('no restart/eval')))
            kill = stack.enter_context(patch.object(q.os, 'killpg', side_effect=AssertionError('no signals')))
            with self.assertRaises(q.e.EvidenceError):
                q.wait_for_training(self.plan)
        run.assert_not_called(); kill.assert_not_called()

    def test_audit_ready_required_before_gpu_or_evaluation(self):
        self.execute_context(failed_audit=True)
        with self.assertRaises(q.e.EvidenceError):
            q.execute(self.plan)
        self.assertEqual([x for x in self.events if x[0] in ('wait', 'audit', 'acquire', 'episode')], [('wait',), ('audit',)])
        self.assertTrue(any(name == 'failure_guard.json' for name, value in self.writes))

    def test_final_audit_requires_ready_status_and_exit_code(self):
        cases = [({'evaluation_ready': False, 'status': 'passed_final_training_and_gpu_released'}, 0, False),
                 ({'evaluation_ready': True, 'status': 'waiting_for_training_completion'}, 0, False),
                 ({'evaluation_ready': True, 'status': 'passed_final_training_and_gpu_released'}, 2, False),
                 ({'evaluation_ready': 1, 'status': 'passed_final_training_and_gpu_released'}, 0, False),
                 ({'evaluation_ready': True, 'status': 'passed_final_training_and_gpu_released'}, 0, True)]
        for index, (report, code, accepted) in enumerate(cases):
            with self.subTest(index=index):
                directory = self.base/f'audit-{index}'; directory.mkdir()
                def run(command, **kwargs):
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                    self.assertEqual(command[0], str(q.CPU_PYTHON))
                    self.assertTrue(command[2].endswith('audit_precision_v3_training_final_v42.py'))
                    Path(command[command.index('--out')+1]).write_text(json.dumps(report))
                    return SimpleNamespace(returncode=code)
                with patch.object(q, 'OUT', directory), patch.object(q.e, 'write_json', side_effect=self.write), \
                        patch.object(q.subprocess, 'run', side_effect=run):
                    if accepted:
                        result = q.audit_final_training(self.plan)
                        self.assertEqual(result['sha256'], q.e.sha256_file(result['path']))
                    else:
                        with self.assertRaisesRegex(q.e.EvidenceError, 'independent audit failed'):
                            q.audit_final_training(self.plan)

    def test_fixed_five_valid_failures_are_not_retried(self):
        self.execute_context(rows={q.e.SEEDS[1]: {'status': 'physics_safety_abort'}, q.e.SEEDS[3]: {'status': 'control_safety_abort'}})
        with patch.dict(os.environ, {}, clear=False):
            self.assertEqual(q.execute(self.plan), 0)
        phases = [x[0] for x in self.events]
        self.assertLess(phases.index('wait'), phases.index('audit'))
        self.assertLess(phases.index('audit'), phases.index('acquire'))
        self.assertLess(phases.index('acquire'), phases.index('episode'))
        self.assertEqual([x[1] for x in self.events if x[0] == 'episode'], list(q.e.SEEDS))
        summaries = [value for name, value in self.writes if name == 'summary.json']
        self.assertEqual(summaries[-1]['status'], 'completed')
        self.assertEqual(summaries[-1]['success_rate'], 0.)
        self.assertEqual([row['seed'] for row in summaries[-1]['episodes']], list(q.e.SEEDS))
        self.assertEqual(summaries[-1]['recorded_seeds'], [932126])
        self.assertEqual(sum(name.startswith('attempt_seed_') for name, value in self.writes), 5)
        self.assertIn(('release', q.OWNER), self.events)

    def test_unknown_failure_stops_remaining_seeds_and_preserves_attempt(self):
        self.execute_context(rows={q.e.SEEDS[0]: 'unknown'})
        with self.assertRaises(q.e.EvidenceError):
            q.execute(self.plan)
        self.assertEqual([x[1] for x in self.events if x[0] == 'episode'], [932126])
        self.assertEqual(sum(name.startswith('attempt_seed_') for name, value in self.writes), 1)
        self.assertTrue(any(name == 'failure_guard.json' for name, value in self.writes))
        self.assertIn(('release', q.OWNER), self.events)

    def test_existing_target_is_never_reused_by_duplicate_writer(self):
        self.execute_context()
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            q.execute(self.plan)
        self.assertFalse(self.events)
        self.assertFalse(self.writes)

    def test_owner_retained_if_gpu_child_remains(self):
        self.execute_context(leftover_gpu=True)
        self.assertEqual(q.execute(self.plan), 0)
        self.assertNotIn(('release', q.OWNER), self.events)
        self.assertTrue(any(name == 'owner_retained_guard.json' for name, value in self.writes))

    def run_once_context(self, *, child, observations=None, write_error=False, health_ok=True):
        self.out.mkdir(exist_ok=True)
        item = self.plan['commands'][0]
        directory = Path(item['out']); directory.mkdir(parents=True, exist_ok=True)
        result_file = directory/'result.json'
        result_file.write_text('{"status":"running","control_steps":7}')
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(q, 'OUT', self.out))
        popen = stack.enter_context(patch.object(q.subprocess, 'Popen', return_value=child))
        health = SimpleNamespace(returncode=0 if health_ok else 1, stdout='595.91.07\n' if health_ok else '', stderr='')
        stack.enter_context(patch.object(q.subprocess, 'run', return_value=health))
        def write(path, value):
            if write_error and Path(path).name == 'active_participant.json':
                raise OSError('injected active-participant write failure')
            self.write(path, value)
        stack.enter_context(patch.object(q.e, 'write_json', side_effect=write))
        stack.enter_context(patch.object(q.e, 'read_json', side_effect=observations if observations is not None else [dict(status='running', control_steps=7)]*10))
        stack.enter_context(patch.object(q.time, 'monotonic', return_value=0.))
        stack.enter_context(patch.object(q.time, 'time', return_value=result_file.stat().st_mtime+1))
        stack.enter_context(patch.object(q.time, 'sleep'))
        kill = stack.enter_context(patch.object(q.os, 'killpg'))
        return item, popen, kill

    def test_result_observation_error_repolls_same_live_child(self):
        child = Child(polls=[None, None, 0, 0])
        item, popen, kill = self.run_once_context(child=child, observations=[json.JSONDecodeError('transient', '', 0), dict(status='running', control_steps=8)])
        self.assertEqual(q.run_once(item, self.plan), 0)
        self.assertEqual(popen.call_count, 1)
        kill.assert_not_called()
        env = popen.call_args.kwargs['env']
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
        self.assertIn('nvidia_5959107', env['LD_LIBRARY_PATH'])
        self.assertEqual(env['P2_V3_THIRD_VIEW_SEED'], '932126')

    def test_cleanup_covers_failure_immediately_after_popen(self):
        child = Child()
        item, popen, kill = self.run_once_context(child=child, write_error=True)
        with self.assertRaises(OSError):
            q.run_once(item, self.plan)
        kill.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertEqual(child.wait_calls, [15])

    def test_watchdog_signals_only_its_own_eval_group(self):
        child = Child()
        item, popen, kill = self.run_once_context(child=child, health_ok=False)
        with self.assertRaisesRegex(q.e.EvidenceError, 'driver failed3'):
            q.run_once(item, self.plan)
        kill.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertNotEqual(child.pid, 123)  # mocked trainer PID in other tests is never signalled

    def test_cli_without_execute_is_cpu_plan_only(self):
        with patch.object(q, 'make_plan', return_value=self.plan), patch.object(q, 'verify'), \
                patch.object(q, 'execute', side_effect=AssertionError('plan CLI must not create a queue')) as execute, \
                patch.object(sys, 'argv', [str(Path(q.__file__))]), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(q.main(), 0)
        execute.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['status'], 'cpu_plan_only_no_simulation')
        self.assertFalse(self.out.exists())

    def test_parent_admission_uses_pinned_environment(self):
        admission, scheduler = self.execute_context()
        def check_parent_env():
            self.assertIn('nvidia_5959107', os.environ.get('LD_LIBRARY_PATH', ''))
            return {'mock': True}
        scheduler.collection_resource_admission.side_effect = check_parent_env
        with patch.dict(os.environ, {'LD_LIBRARY_PATH': '', 'CUDA_VISIBLE_DEVICES': ''}, clear=False):
            self.assertEqual(q.execute(self.plan), 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24/precision_v3_final5_queue_cpu_tests_v42.json')
    args = parser.parse_args(argv)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QueueTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    plan = None
    report = dict(status='passed' if result.wasSuccessful() else 'failed',
        verification_kind='precision-v3-final-five-queue-cpu-gate-and-mocks-v42',
        tests_run=result.testsRun, failure_count=len(result.failures), error_count=len(result.errors),
        failures=[dict(test=str(test), trace=trace) for test, trace in result.failures+result.errors],
        runner_source_sha256=q.e.sha256_file(q.__file__), test_source_sha256=q.e.sha256_file(__file__),
        physical_evaluation_started=False, gpu_command_executed=False, trainer_signalled=False,
        actual_cpu_plan_verified=plan is not None,
        plan_seeds=plan['seeds'] if plan else None,
        recording=plan['recording'] if plan else None,
        new_source_count=len(plan['extension_sources']) if plan else None,
        scope='mock control-flow and real CPU plan binding; not actual future training completion or evaluation success')
    report['file_sha256'] = {
        str(Path(q.__file__).resolve()): report['runner_source_sha256'],
        str(Path(__file__).resolve()): report['test_source_sha256']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def write_report():
        temp = args.out.with_name(args.out.name+'.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        temp.replace(args.out)
    # Bootstrap the queue's own mock-test gate only after every mock has passed.
    # The queue is not started by this test. Real plan verification then adds its
    # explicit evidence before the root freezes this report or starts a unit.
    write_report()
    if result.wasSuccessful():
        try:
            plan = q.make_plan()
            q.verify(plan)
            assert plan['seeds'] == [932126, 932127, 932128, 932129, 932130]
            assert plan['recording']['recorded_seeds'] == [932126]
            assert plan['child_env']['CUDA_VISIBLE_DEVICES'] == '0'
            assert 'nvidia_5959107' in plan['child_env']['LD_LIBRARY_PATH']
            report.update(actual_cpu_plan_verified=True, plan_seeds=plan['seeds'],
                          recording=plan['recording'], new_source_count=len(plan['extension_sources']))
        except BaseException as error:
            report.update(status='failed', actual_cpu_plan_verified=False,
                          plan_error=f'{type(error).__name__}: {error}')
        write_report()
    print(json.dumps({k:report[k] for k in ('status', 'tests_run', 'failure_count', 'error_count')}, ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
