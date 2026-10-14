#!/usr/bin/env python3
"""CPU-only in-memory Popen/clock/evidence tests. No filesystem test artifacts.
Run with python -B; default CLI smoke test is also read-only. No Isaac/GPU.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib.abc
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'isaacsim', 'omni', 'carb', 'pxr', 'torch', 'fcl'}:
            raise AssertionError('CPU queue tests cannot import ' + name)
sys.meta_path.insert(0, NoRuntime())

import copy
from contextlib import redirect_stdout
import io
import json
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from scripts import run_zhuhai_task1_repair_v2_probes as q


def plan(workers=4):
    out = q.probe.VERIFICATION/'task1_expert_repair_v2_QUEUE_CPU_TEST_NOT_EXECUTED'
    cases = []
    for seed, parent in q.probe.REQUIRED_CASES.items():
        commands = {str(slot): ['/fake/python', '-B', str(q.probe.ENTRY), '--seed', str(seed),
            '--source-parent', str(parent), '--out', str(out/f'seed_{seed}'), '--worker-slot', str(slot),
            '--execute', '--gpu-lock-owner', 'zhuhai-task1-repair-v2-test',
            '--expected-candidate-sha256', 'a'*64, '--expected-bindings-sha256', 'b'*64]
            for slot in range(workers)}
        cases.append(dict(seed=seed, parent=parent, out=str(out/f'seed_{seed}'), commands_by_slot=commands))
    return dict(schema=q.SCHEMA, out=str(out), gpu_lock_owner='zhuhai-task1-repair-v2-test', workers=workers,
        cases=cases, candidate_sha256='a'*64, bindings_sha256='b'*64, child_env={'PYTHONDONTWRITEBYTECODE': '1'},
        resource_wait_seconds=100., worker_timeout_seconds=1000., stagger_seconds=15.,
        suite={'bindings': {'/fake/expert.py': 'a'*64}}, queue_bindings={'/fake/queue.py': 'a'*64},
        file_stamps={'/fake/expert.py': [1]*5, '/fake/queue.py': [1]*5})


def resources(ram=8, gpu=4, swap=.1, total=32):
    g = q.probe.GIB
    return dict(ram=dict(total_bytes=total*g, available_bytes=ram*g, swap_total_bytes=10*g,
        swap_used_bytes=swap*10*g), disk={'free_bytes': 200*g},
        gpu={'devices': [dict(total_bytes=16*g, used_bytes=(16-gpu)*g)]})


class MemoryEvidence:
    def __init__(self): self.files, self.events, self.logs = {}, [], []
    def write(self, name, value):
        if name in self.files: raise FileExistsError(name)
        self.files[name] = copy.deepcopy(value)
    def event(self, value): self.events.append(copy.deepcopy(value))
    def log(self, seed):
        handle = io.BytesIO(); self.logs.append(handle); return handle


class FakeProcess:
    def __init__(self, world, pid, duration, code):
        self.world, self.pid, self.end, self.code, self.stopped = world, pid, world.t + duration, code, None
    def poll(self):
        return self.stopped if self.stopped is not None else self.code if self.world.t >= self.end else None


class World:
    def __init__(self, durations=None, codes=None):
        self.t = 0.; self.children = []; self.starts = []; self.signals = []; self.peak = 0
        self.durations = durations or [80]*4; self.codes = codes or [0]*4
        self.evidence = MemoryEvidence(); self.cap = 4; self.foreign_at = None; self.orphan = False
        self.guard = SimpleNamespace(check=Mock())
    def now(self): return self.t
    def sleep(self, seconds): self.t += seconds
    def live_count(self): return sum(c.poll() is None for c in self.children)
    def popen(self, argv, **kwargs):
        seed = int(argv[argv.index('--seed')+1])
        assert f'command_{seed}.json' in self.evidence.files, 'command must freeze BEFORE Popen'
        assert kwargs['start_new_session'] is True and 'shell' not in kwargs
        assert kwargs['stderr'] == q.subprocess.STDOUT and kwargs['stdin'] == q.subprocess.DEVNULL
        i = len(self.children)
        child = FakeProcess(self, 50000+i, self.durations[i], self.codes[i])
        self.children.append(child); self.starts.append((self.t, list(argv), kwargs))
        self.peak = max(self.peak, self.live_count())
        return child
    def resource(self): return resources(ram=8 if self.live_count() < self.cap else 6)
    def processes(self):
        rows = [dict(pid=c.pid, pgid=c.pid, ppid=1, start_ticks=c.pid, state='R',
            argv=['python', str(q.probe.ENTRY)]) for c in self.children if c.poll() is None]
        if self.orphan and self.children and self.children[0].poll() is not None:
            rows.append(dict(pid=60000, pgid=self.children[0].pid, ppid=1, state='R', argv=['kit']))
        if self.foreign_at is not None and self.t >= self.foreign_at:
            rows.append(dict(pid=90000, pgid=90000, ppid=1, state='R', argv=['python', 'collect_isaac_zhuhai_recycle.py']))
        return rows
    def kill(self, pgid, sig):
        self.signals.append((pgid, sig))
        child = next(c for c in self.children if c.pid == pgid)
        child.stopped = -int(sig)
        if child is self.children[0]: self.orphan = False
    def verify(self, directory):
        return dict(all_four_strict_passed=True, candidate_sha256='a'*64, bindings_sha256='b'*64,
                    admitted_to_production=False, production_quota_credit=0)
    def services(self):
        return q.Services(popen=self.popen, now=self.now, sleep=self.sleep, resources=self.resource,
            processes=self.processes, owner=Mock(), verify=Mock(side_effect=self.verify), signal_group=self.kill)
    def run(self, value=None, services=None):
        return q.run_queue(value or plan(), services or self.services(), self.evidence, self.guard)


class SchedulerTests(unittest.TestCase):
    def test_four_fixed_cases_max_four_and_15_second_stagger(self):
        w = World(); services = w.services(); code, result = w.run(services=services)
        self.assertEqual(code, 0); self.assertEqual(w.peak, 4)
        self.assertEqual([int(argv[argv.index('--seed')+1]) for _, argv, _ in w.starts], list(q.probe.REQUIRED_CASES))
        self.assertTrue(all(b[0]-a[0] >= 15 for a, b in zip(w.starts, w.starts[1:])))
        self.assertFalse(result['admitted_to_production']); self.assertEqual(result['production_quota_credit'], 0)
        self.assertTrue(all(log.closed for log in w.evidence.logs)); self.assertFalse(w.signals)
        services.verify.assert_called_once_with(Path(plan()['out']))

    def test_single_worker_still_has_stagger(self):
        w = World(durations=[1]*4); code, _ = w.run(plan(1))
        self.assertEqual(code, 0); self.assertEqual(w.peak, 1)
        self.assertEqual([x[0] for x in w.starts], [0, 15, 30, 45])

    def test_resource_adaptive_concurrency_runs_all_without_retries(self):
        w = World(); w.cap = 2; code, result = w.run()
        self.assertEqual(code, 0); self.assertEqual(w.peak, 2); self.assertEqual(len(w.starts), 4)
        self.assertTrue(any(e['event'] == 'resource_wait_lower_concurrency' for e in w.evidence.events))
        self.assertEqual(len({r['seed'] for r in result['attempts']}), 4)

    def test_fourth_swap_gate_uses_lower_slot_when_capacity_returns(self):
        w = World(); services = w.services()
        services.resources = lambda: resources(swap=.90 if w.live_count() == 3 else .80)
        code, _ = w.run(services=services)
        self.assertEqual(code, 0); self.assertEqual(w.peak, 3)
        self.assertEqual(w.starts[-1][1][w.starts[-1][1].index('--worker-slot')+1], '0')

    def test_physical_failure_preserved_finish_all_and_suite_unadmitted(self):
        w = World(codes=[2, 0, 2, 0]); services = w.services()
        services.verify = Mock(side_effect=ValueError('strict physical probe failed'))
        code, result = w.run(services=services)
        self.assertEqual(code, 2); self.assertEqual(len(w.starts), 4)
        self.assertEqual([x['exit_code'] for x in result['attempts']], [2, 0, 2, 0])
        self.assertFalse(result['all_four_strict_passed']); services.verify.assert_called_once()

    def test_exit_zero_cannot_replace_strict_physical_verification(self):
        w = World(); services = w.services(); services.verify = Mock(side_effect=ValueError('not strict'))
        code, result = w.run(services=services)
        self.assertEqual(code, 2); self.assertIn('not strict', result['verification']['error'])

    def test_failed_exit_cannot_be_masked_by_true_verifier(self):
        w = World(codes=[0, 0, 2, 0]); code, result = w.run()
        self.assertEqual(code, 2); self.assertFalse(result['all_four_strict_passed'])

    def test_source_drift_stops_immediately_only_kills_owned_children(self):
        w = World()
        def guard(**kw):
            if w.t >= 16: raise RuntimeError('source drift')
        w.guard.check.side_effect = guard
        code, result = w.run()
        self.assertEqual(code, 2); self.assertEqual(len(w.starts), 2)
        self.assertIn('source drift', result['fatal_error'])
        self.assertEqual({pid for pid, _ in w.signals}, {50000, 50001})

    def test_owner_loss_stops_pending_launches(self):
        w = World(); services = w.services()
        def owner(value):
            if w.t >= 1: raise RuntimeError('owner lost')
        services.owner = owner
        code, result = w.run(services=services)
        self.assertEqual(code, 2); self.assertEqual(len(w.starts), 1)
        self.assertIn('owner lost', result['fatal_error'])

    def test_foreign_orphan_never_signalled(self):
        w = World(); w.foreign_at = 5; code, result = w.run()
        self.assertEqual(code, 2); self.assertEqual(len(w.starts), 1)
        self.assertTrue(w.signals); self.assertNotIn(90000, [x[0] for x in w.signals])
        self.assertIn('Foreign', result['fatal_error'])

    def test_parent_exit_with_live_descendants_cannot_finish_successfully(self):
        w = World(durations=[1, 80, 80, 80]); w.orphan = True
        code, result = w.run()
        self.assertEqual(code, 2); self.assertIn('live descendants', result['fatal_error'])
        self.assertIn(50000, [x[0] for x in w.signals])

    def test_inventory_failure_still_signals_only_known_groups(self):
        w = World(); services = w.services(); real = services.processes
        def unavailable():
            if w.children: raise RuntimeError('proc inventory unavailable')
            return real()
        services.processes = unavailable
        code, result = w.run(services=services)
        self.assertEqual(code, 2); self.assertTrue(w.signals)
        self.assertEqual({x[0] for x in w.signals}, {50000})
        self.assertIsNotNone(result['cleanup_error'])
        self.assertFalse(result['external_gpu_lock_released'])

    def test_no_resources_timeout_launches_nothing(self):
        w = World(); w.cap = 0; value = plan(); value['resource_wait_seconds'] = 3
        code, result = w.run(value)
        self.assertEqual(code, 2); self.assertFalse(w.starts)
        self.assertTrue(all(x['status'] == 'not_started' for x in result['attempts']))

    def test_worker_timeout_stops_only_its_group(self):
        w = World(); value = plan(); value['worker_timeout_seconds'] = 2
        code, result = w.run(value)
        self.assertEqual(code, 2); self.assertEqual(w.signals[0][0], 50000)
        self.assertIn('timeout', result['fatal_error'])

    def test_popen_failure_preserves_command_and_stops(self):
        w = World(); services = w.services(); services.popen = Mock(side_effect=OSError('spawn failed'))
        code, result = w.run(services=services)
        self.assertEqual(code, 2); self.assertEqual(result['attempts'][0]['status'], 'launch_failed')
        self.assertIn('command_944025.json', w.evidence.files); self.assertTrue(w.evidence.logs[0].closed)

    def test_existing_partial_case_never_replayed(self):
        w = World()
        with patch.object(Path, 'exists', return_value=True): code, result = w.run()
        self.assertEqual(code, 2); self.assertFalse(w.starts); self.assertIn('never overwrite', result['fatal_error'])

    def test_keyboard_interrupt_uses_owned_cleanup(self):
        w = World(); services = w.services(); base_sleep = services.sleep
        def interrupted(seconds):
            if not w.signals: raise KeyboardInterrupt()
            base_sleep(seconds)
        services.sleep = interrupted
        code, result = w.run(services=services)
        self.assertEqual(code, 2); self.assertIn('KeyboardInterrupt', result['fatal_error'])
        self.assertEqual(len(w.signals), 1)


class GateTests(unittest.TestCase):
    def test_resource_exact_boundaries(self):
        self.assertTrue(q.launch_allowed(resources(ram=7, gpu=3), 0))
        self.assertFalse(q.launch_allowed(resources(ram=6.99), 0))
        self.assertFalse(q.launch_allowed(resources(gpu=2.99), 0))
        self.assertFalse(q.launch_allowed(resources(swap=.90), 3))
        self.assertTrue(q.launch_allowed(resources(swap=.90), 2))
        self.assertFalse(q.launch_allowed(resources(total=80, ram=8), 3))
        self.assertTrue(q.launch_allowed(resources(total=80, ram=8), 0))

    def test_foreign_known_jobs_and_wrong_owner_same_script_rejected(self):
        for argv in (['python', 'collect_isaac_zhuhai_recycle.py'], ['python', 'collect_other_dataset.py'], ['kit'],
                     ['python', str(q.probe.ENTRY), '--gpu-lock-owner', 'zhuhai-task1-repair-v2-test'],
                     ['python', '-m', 'isaacsim'], ['python', '-c', 'from isaacsim import SimulationApp']):
            row = dict(pid=9000, pgid=9000, state='S', argv=argv)
            with self.assertRaises(RuntimeError): q.reject_foreign([row], own_groups=[50000], self_pid=1)
        q.reject_foreign([dict(pid=5, pgid=5, state='R', argv=['kit'])], [5], self_pid=1)
        q.reject_foreign([dict(pid=5, pgid=5, state='Z', argv=['kit'])], [], self_pid=1)

    def test_source_and_launch_command_drift(self):
        value = plan(); guard = q.FrozenGuard(value)
        with patch.object(q, 'file_stamp', return_value=[1]*5), patch.object(q.probe, 'sha256', return_value='a'*64):
            guard.check(full=True)
            value['cases'][0]['commands_by_slot']['0'].append('--evil')
            with self.assertRaisesRegex(RuntimeError, 'launch plan changed'): guard.check()
        value = plan(); guard = q.FrozenGuard(value)
        with patch.object(q, 'file_stamp', return_value=[2]*5):
            with self.assertRaisesRegex(RuntimeError, 'changed'): guard.check()
        with patch.object(q, 'file_stamp', return_value=[1]*5), patch.object(q.probe, 'sha256', return_value='c'*64):
            with self.assertRaisesRegex(RuntimeError, 'hash drift'): guard.check()

    def test_defaults_main_is_read_only_no_process_inventory(self):
        with patch.object(q, 'build_plan', return_value=plan()), patch.object(q, 'execute_plan') as execute, \
                patch.object(q, 'process_snapshot') as inventory, patch.object(q.probe, 'verify_owner') as owner, \
                patch.object(Path, 'mkdir') as mkdir, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(q.main([]), 0)
        self.assertEqual(json.loads(output.getvalue())['mode'], 'CPU_PLAN_NO_GPU_NO_WRITES')
        execute.assert_not_called(); inventory.assert_not_called(); owner.assert_not_called(); mkdir.assert_not_called()
        self.assertEqual(q.parser().parse_args([]).workers, 4)

    def test_execute_requires_both_frozen_hashes_and_external_v2_owner(self):
        base = q.parser().parse_args([])
        for updates in (dict(execute=True), dict(execute=True, gpu_lock_owner='zhuhai-task1-repair-v2-test'),
                        dict(gpu_lock_owner='zhuhai-task2-collection-only-v2-20261009')):
            value = copy.deepcopy(base); vars(value).update(updates)
            with self.assertRaises(ValueError): q.validate_args(value)
        with patch.object(q.probe, 'verify_owner', side_effect=RuntimeError('wrong owner')), patch.object(q, 'build_plan') as build:
            with self.assertRaisesRegex(RuntimeError, 'wrong owner'):
                q.main(['--execute', '--gpu-lock-owner', 'zhuhai-task1-repair-v2-test',
                    '--expected-candidate-sha256', 'a'*64, '--expected-bindings-sha256', 'b'*64])
            build.assert_not_called()

    def test_old_directory_v1_production_and_workers5_rejected(self):
        for out in (q.probe.VERIFICATION/'task1_expert_repair_v1', ROOT/'outputs/production_test'):
            args = q.parser().parse_args(['--out', str(out)])
            with self.assertRaises((ValueError, FileExistsError)): q.validate_args(args)
        with patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit): q.parser().parse_args(['--workers', '5'])
        with patch.object(Path, 'exists', return_value=True):
            with self.assertRaises(FileExistsError): q.validate_args(q.parser().parse_args([]))

    def test_build_plan_freezes_all_slot_commands_and_hashes(self):
        args = q.parser().parse_args(['--gpu-lock-owner', 'zhuhai-task1-repair-v2-test'])
        suite = dict(bindings={}, candidate_sha256='a'*64, bindings_sha256='b'*64)
        with patch.object(q.probe, 'make_plan', return_value=(None, None, None, suite)) as make, \
                patch.object(q.probe, 'sha256', return_value='c'*64), patch.object(q, 'file_stamp', return_value=[1]*5), \
                patch.object(Path, 'mkdir') as mkdir:
            value = q.build_plan(args)
        self.assertEqual(make.call_count, 4); mkdir.assert_not_called()
        self.assertEqual(len(value['cases']), 4); self.assertFalse(value['record_video'])
        for case in value['cases']:
            self.assertEqual(len(case['commands_by_slot']), 4)
            for command in case['commands_by_slot'].values():
                self.assertEqual(command[2], str(q.probe.ENTRY))
                self.assertIn('a'*64, command); self.assertIn('b'*64, command)
                self.assertNotIn('--resume', command)

    def test_mismatching_per_case_fingerprints_rejected(self):
        suite = dict(bindings={}, candidate_sha256='a'*64, bindings_sha256='b'*64)
        with patch.object(q.probe, 'make_plan', side_effect=[(None, None, None, suite), (None, None, None, {})]):
            with self.assertRaisesRegex(RuntimeError, 'same frozen'): q.build_plan(q.parser().parse_args([]))

    def test_mutex_rejects_duplicate_queue_without_unlinking_gpu_lock(self):
        with patch.object(q.probe, 'verify_owner'), patch.object(q.os, 'open', return_value=12), \
                patch.object(q.os, 'close') as close, patch.object(q.fcntl, 'flock', side_effect=BlockingIOError()), \
                patch.object(Path, 'unlink') as unlink:
            with self.assertRaisesRegex(RuntimeError, 'Another v2 probe queue'):
                with q.queue_mutex('zhuhai-task1-repair-v2-test'): self.fail('duplicate entered')
            close.assert_called_once_with(12); unlink.assert_not_called()

    def test_no_simulation_runtime_imported(self):
        self.assertNotIn('harness.isaac_zhuhai_runtime', sys.modules)
        self.assertFalse(any(x.split('.')[0] in {'isaacsim', 'omni', 'carb', 'pxr', 'torch'} for x in sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
