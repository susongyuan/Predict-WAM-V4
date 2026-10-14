#!/usr/bin/env python3
"""Focused CPU-only receipt/launcher tests; never invoke a GPU executable."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_wam_v3_stages as S
import wam_v3_resource_authorization as R

HERE = ROOT / 'outputs/validation/task0_continuation_gpu_authorization_20260930'
OLD_PLAN = ROOT / 'outputs/validation/task0_continuation_20k_20260929/continuation_plan.json'
OLD_APPROVAL = ROOT / 'outputs/validation/task0_continuation_20k_20260929/completion_20260930/no_gpu_approval.json'


class AuthorizationTests(unittest.TestCase):
    def args(self, authorization=None):
        cp = json.loads(OLD_PLAN.read_text())
        return SimpleNamespace(prefix=cp['run_prefix'] + 'auth_cpu', budget=50000,
            config=cp['parent']['config'], resume_source=cp['parent']['checkpoint'],
            cache_ledger='data/dino_cache_task000/sha256_ledger_micro16_30k_v1.json',
            approval=str(OLD_APPROVAL), continuation_plan=str(OLD_PLAN), instance=10,
            gpu_authorization=authorization)

    def test_exact_receipt_and_budget(self):
        contract = R.validate_authorization(str(R.RECEIPT))
        self.assertEqual(R.validate_resource_contract(contract), contract)
        self.assertEqual([contract[k] for k in ('gpu_utilization_percent_allowed', 'total_vram_abort_mib',
                                              'cpu_percent_ceiling', 'ram_percent_ceiling')], [100, 15300, 85, 85])
        self.assertFalse(contract['training_simulation_concurrency'])
        self.assertFalse(contract['training_approval_implied'])

    def test_no_receipt_or_boolean_environment_can_authorize(self):
        with patch.dict(os.environ, GPU_PEAK_ALLOWED='1', GPU_AUTHORIZATION='true',
                        P2_GPU_UTILIZATION_ALLOWED='100', P2_TOTAL_VRAM_ABORT_MIB='999999'):
            with self.assertRaises(ValueError): R.validate_authorization(None)
            with self.assertRaises(ValueError): R.validate_authorization('true')
            plan = S.new_plan(self.args())
            with patch.object(S.subprocess, 'run', side_effect=AssertionError('no processes')):
                with self.assertRaisesRegex(ValueError, 'resource-window'): S.require_execution_authorization(plan)

    def test_copied_receipt_wrong_path_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'authorization.json'; p.write_bytes(R.RECEIPT.read_bytes())
            with self.assertRaisesRegex(ValueError, 'substitute'): R.validate_authorization(str(p))

    def test_edited_receipt_rejected_before_json_use(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'authorization.json'
            receipt = json.loads(R.RECEIPT.read_text()); receipt['operational_total_vram_abort_mib'] = 99999
            p.write_text(json.dumps(receipt))
            with patch.object(R, 'RECEIPT', p):
                with self.assertRaisesRegex(ValueError, 'SHA256'): R.validate_authorization(str(p))

    def test_changed_budget_or_lock_rejected(self):
        contract = R.validate_authorization(str(R.RECEIPT))
        for key, value in [('gpu_utilization_percent_allowed', 999), ('total_vram_abort_mib', 99999),
                           ('cpu_percent_ceiling', 100), ('ram_percent_ceiling', 100),
                           ('training_simulation_concurrency', True), ('gpu_lock', '/tmp/not-project.lock')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                R.validate_resource_contract({**contract, key: value})

    def test_controller_held_lock_rejected_without_processes(self):
        a = self.args(str(R.RECEIPT)); plan = S.new_plan(a)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); lock = out / 'gpu.lock'; lock.write_text('other-owner')
            with patch.object(S, 'GPU_LOCK', lock), patch.object(S.subprocess, 'run', side_effect=AssertionError('no processes')):
                self.assertEqual(S.execute(plan, a, out), 3)
            self.assertEqual(lock.read_text(), 'other-owner')
            self.assertIn('GPU lock is already held', S.read(out / 'status.json')['error'])

    def test_valid_resource_is_not_training_approval(self):
        a = self.args(str(R.RECEIPT)); plan = S.new_plan(a)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            with patch.object(S, 'GPU_LOCK', out / 'absent.lock'), patch.object(S.subprocess, 'run', side_effect=AssertionError('no processes')):
                self.assertEqual(S.execute(plan, a, out), 3)
            self.assertIn('training approval is not admitted', S.read(out / 'status.json')['error'])

    def test_sanitized_environment_and_rederived_budget(self):
        expected = R.resource_environment(R.validate_authorization(str(R.RECEIPT)))
        with patch.dict(os.environ, {key: 'malicious-stale-value' for key in expected}):
            clean = S.clean_env()
            self.assertFalse(set(expected) & set(clean))
            clean.update(R.resource_environment(R.validate_authorization(str(R.RECEIPT))))
            self.assertEqual({key: clean[key] for key in expected}, expected)

    def test_four_targets_and_pinned_last_remain(self):
        a = self.args(str(R.RECEIPT)); plan = S.new_plan(a)
        self.assertEqual(plan['checkpoints'], [35000, 40000, 45000, 50000])
        self.assertEqual(plan['start_step'], 30000)
        self.assertEqual(plan['training']['resume_source'], json.loads(OLD_PLAN.read_text())['parent']['checkpoint'])
        self.assertTrue(plan['training']['resume_source'].endswith('/checkpoint_last.pt'))
        self.assertEqual(S.require_execution_authorization(plan), plan['resource'])
        self.assertEqual(plan['training']['approval_sha256'], S.digest(OLD_APPROVAL))

    def launcher_env(self, dry=True):
        a = self.args()
        env = dict(os.environ, NAME=a.prefix, MODE='long', MAX_STEPS='35000', CONFIG=a.config,
                   CACHE_LEDGER=a.cache_ledger, APPROVAL=a.approval, RESUME=a.resume_source,
                   CONTINUATION_PLAN=str(OLD_PLAN), DRY_RUN='1' if dry else '0',
                   CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        for key in ('INIT', 'RUN_NAME', 'RESUME_SHA256', 'GPU_AUTHORIZATION'):
            env.pop(key, None)
        return env

    def test_launcher_unauthorized_denial_before_service(self):
        env = self.launcher_env(False)
        env.update(GPU_PEAK_ALLOWED='1', P2_GPU_UTILIZATION_ALLOWED='100')
        run = subprocess.run(['bash', str(ROOT / 'scripts/run_wam_v3_unit.sh')], env=env,
                             capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn('explicit GPU authorization receipt required', run.stderr)
        self.assertNotIn('Launching', run.stdout)
        self.assertFalse((ROOT / 'outputs/training' / env['NAME']).exists())

    def test_launcher_held_lock_refuses_in_isolated_cpu_fixture(self):
        env = self.launcher_env(False); env['GPU_AUTHORIZATION'] = str(R.RECEIPT)
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / 'gpu.lock'; lock.write_text('fixture-other-owner')
            source = (ROOT / 'scripts/run_wam_v3_unit.sh').read_text()
            # Redirect only the fixture lock/root, never create or replace the real GPU lock.
            source = source.replace('ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)',
                                    f'ROOT={ROOT}')
            source = source.replace('LOCK_FILE=/tmp/project2_gpu.lock', f'LOCK_FILE={lock}')
            fixture = Path(td) / 'launcher.sh'; fixture.write_text(source)
            run = subprocess.run(['bash', str(fixture)], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn('GPU lock is already held', run.stderr)
            self.assertEqual(lock.read_text(), 'fixture-other-owner')
            self.assertNotIn('Launching', run.stdout)

    def test_entrypoint_source_binding_not_relaxed(self):
        plan = S.new_plan(self.args(str(R.RECEIPT)))
        plan['training']['entrypoint_source_sha256']['scripts/run_wam_v3_unit.sh'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'entrypoint sources changed'):
            S.require_execution_authorization(plan)

    def test_launcher_authorized_dryrun_four_targets(self):
        env = self.launcher_env(); env['GPU_AUTHORIZATION'] = str(R.RECEIPT)
        logs = []
        for step in (35000, 40000, 45000, 50000):
            env['MAX_STEPS'] = str(step)
            run = subprocess.run(['bash', str(ROOT / 'scripts/run_wam_v3_unit.sh')], env=env,
                                 capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn(f'--max-steps {step}', run.stdout)
            self.assertIn('"total_vram_abort_mib": 15300', run.stdout)
            self.assertIn('"gpu_utilization_percent_allowed": 100', run.stdout)
            self.assertIn('checkpoint_last.pt', run.stdout)
            logs.append(run.stdout)
        self.assertFalse((ROOT / 'outputs/training' / env['NAME']).exists())
        print('AUTHORIZED_LAUNCHER_CPU_PREVIEW_ONLY\n' + '\n'.join(logs))

    def test_existing_four_stage_last_resume_chain_cpu_fixture(self):
        # Run just this existing bounded integration, not the completed 63-test audit.
        # Its process/eval/lock/video boundaries are mocked; no GPU work is possible.
        path = ROOT / 'outputs/validation/task0_continuation_20k_20260929/completion_20260930/test_integration.py'
        spec = importlib.util.spec_from_file_location('prior_cpu_integration', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        case = module.Integration('test_cpu_four_stage_train_eval_select_resume')
        try:
            case.test_cpu_four_stage_train_eval_select_resume()
        finally:
            case.doCleanups()


if __name__ == '__main__':
    unittest.main(verbosity=2)
