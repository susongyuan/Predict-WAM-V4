#!/usr/bin/env python3
"""CPU-only checks for the authorized evaluator resource consumer."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import behavior_resources as resources
from wam_v3_resource_authorization import RECEIPT, validate_authorization, resource_environment


def snapshot(cpu=10, ram=40, vram=1500, utilization=10):
    return {'memory_used_percent': ram, 'cpu_used_percent': cpu,
            'memory_total_bytes': 32 * 1024**3, 'memory_available_bytes': 24 * 1024**3,
            'cpu_logical': 32, 'gpu': [{'vram_total_mib': 16311,
            'vram_used_mib': vram, 'utilization_percent': utilization}]}


class FakeChild:
    def __init__(self, polls):
        self.polls = iter(polls)
        self.returncode = None

    def poll(self):
        self.returncode = next(self.polls, self.returncode)
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.env = resource_environment(validate_authorization(str(RECEIPT)))

    def test_receipt_resolves_independent_budgets(self):
        original = copy.deepcopy(resources.CONFIG)
        policy, contract = resources.resource_policy(self.env)
        self.assertEqual(policy['monitor_exit_percent'], 85)
        self.assertEqual(policy['ram_exit_percent'], 85)
        self.assertEqual(policy['gpu_total_vram_limit_mib'], 15300)
        self.assertEqual(policy['gpu_vram_exit_percent'], 100)
        self.assertEqual(contract['authorization'], str(RECEIPT))
        self.assertEqual(resources.CONFIG, original)

    def test_numeric_environment_cannot_authorize(self):
        for key in self.env:
            if key != 'GPU_AUTHORIZATION':
                with self.subTest(key=key), self.assertRaises(ValueError):
                    resources.resource_policy({key: self.env[key]})

    def test_wrong_receipt_or_conflicting_numbers_rejected(self):
        with self.assertRaises(ValueError):
            resources.resource_policy({'GPU_AUTHORIZATION': '/tmp/not-the-user-receipt.json'})
        for key in self.env:
            if key != 'GPU_AUTHORIZATION':
                with self.subTest(key=key), self.assertRaises(ValueError):
                    resources.resource_policy({**self.env, key: 'untrusted'})

    def test_boundaries_are_not_coupled(self):
        policy, _ = resources.resource_policy(self.env)
        self.assertIsNone(resources.violation(snapshot(cpu=84.9, ram=84.9, vram=14580, utilization=100), policy))
        self.assertIsNone(resources.violation(snapshot(vram=15299, utilization=100), policy))
        self.assertEqual(resources.violation(snapshot(vram=15300), policy), 'global VRAM threshold reached')
        self.assertEqual(resources.violation(snapshot(cpu=85), policy), 'global CPU threshold reached')
        self.assertEqual(resources.violation(snapshot(ram=85), policy), 'global RAM threshold reached')
        s = snapshot()
        s['gpu'] = []
        self.assertEqual(resources.violation(s, policy), 'GPU monitoring unavailable')

    def test_legacy_config_is_not_changed(self):
        policy, contract = resources.resource_policy({})
        self.assertEqual(policy, resources.CONFIG['resource_policy'])
        self.assertIsNone(contract)
        resources.self_test()

    def test_invalid_authorization_precedes_output_and_process(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
                patch.object(resources.subprocess, 'Popen') as popen, \
                patch.object(resources, 'hardware_snapshot') as hardware:
            out = Path(tmp) / 'should-not-exist'
            with self.assertRaises(ValueError):
                resources.run_guarded(['never-run'], {'P2_TOTAL_VRAM_ABORT_MIB': '15300'}, out)
            self.assertFalse(out.exists())
            popen.assert_not_called()
            hardware.assert_not_called()

    def guarded_case(self, during, expected, polls):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
                patch.object(resources, 'hardware_snapshot', side_effect=[snapshot(), during]), \
                patch.object(resources.subprocess, 'Popen', return_value=FakeChild(polls)) as popen, \
                patch.object(resources.subprocess, 'run') as kill, \
                patch.object(resources.time, 'sleep'), patch.object(resources.os, 'sched_getaffinity', return_value=set(range(32))):
            result = resources.run_guarded(['never-run'], self.env, tmp)
            summary = json.loads((Path(tmp) / 'resource_summary.json').read_text())
            self.assertEqual(summary['abort_reason'], expected)
            self.assertEqual(summary['monitor_policy']['monitor_exit_percent'], 85)
            self.assertEqual(summary['monitor_policy']['gpu_total_vram_limit_mib'], 15300)
            self.assertEqual(summary['resource_authorization']['authorization'], str(RECEIPT))
            self.assertFalse(summary['guarantees_global_vram_cap'])
            self.assertIn('MemorySwapMax=0', popen.call_args.args[0])
            if expected:
                self.assertNotEqual(result, 0)
                kill.assert_called_once()
            else:
                self.assertEqual(result, 0)
                kill.assert_not_called()

    def test_guard_allows_authorized_peak(self):
        self.guarded_case(snapshot(vram=14580, utilization=100), None, [None, 0, 0])

    def test_guard_uses_cpu85_during_execution(self):
        self.guarded_case(snapshot(cpu=85), 'global CPU threshold reached', [None, None])

    def test_guard_uses_ram85_during_execution(self):
        self.guarded_case(snapshot(ram=85), 'global RAM threshold reached', [None, None])

    def test_guard_uses_vram15300_during_execution(self):
        self.guarded_case(snapshot(vram=15300), 'global VRAM threshold reached', [None, None])

    def test_manifest_binds_authorization_and_helper(self):
        tree = ast.parse((ROOT / 'scripts/behavior_closed_loop.py').read_text())
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'runtime_artifact_paths')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self.env, clear=True):
            folder = Path(tmp)
            stats = folder / 'stats.json'
            stats.write_text('{}')
            scope = {'ROOT': ROOT, 'CONFIG': resources.CONFIG, 'Path': Path, 'json': json, 'os': os,
                     'task_profile': SimpleNamespace(VTT_DIR=folder, TASK_ID=0)}
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual_manifest_function>', 'exec'), scope)
            args = SimpleNamespace(checkpoint=str(folder / 'ck.pt'), orchestrate=None, program=None, memory_vtt=None)
            cfg = {'dataset': {'stats_path': str(stats)}, 'model': {'vision_encoder': {'checkpoint_path': str(folder)}}}
            paths = scope['runtime_artifact_paths'](args, ['--robot-config', str(folder / 'robot.yaml')], cfg)
            self.assertIn(RECEIPT.resolve(), paths)
            self.assertIn((ROOT / 'scripts/wam_v3_resource_authorization.py').resolve(), paths)


if __name__ == '__main__':
    unittest.main(verbosity=2)
