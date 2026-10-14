#!/usr/bin/env python3
"""CPU-only tests. No real subprocess/systemctl/GPU; all fixtures are temporary.
Run: python3 -B scripts/test_wam_v3_stages.py -v
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).with_name('run_wam_v3_stages.py')


class Harness:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.out = root / 'outputs/training/audit_stages'
        (root / 'config.yaml').write_text('fixture: true')
        (root / 'approval.json').write_text('{}')
        self.now = 0
        self.release_at = None
        self.sleeps = []
        self.train_envs = []
        self.phys_envs = []
        self.states = []
        self.query_error = False
        self.training_status = 'budget_complete'
        self.delta = 0
        self.make_last = True
        self.make_sidecar = True
        self.make_interval = True
        self.last_mutation = None
        self.physical_rc = 0
        self.lock_after_training = False
        self.lock_after_physical = False
        self.mutate = lambda manifest, plan, episode: None
        self.proc_tokens = {}
        self.m = types.ModuleType('stages_under_test')
        self.m.__file__ = str(SOURCE)
        exec(compile(SOURCE.read_text(), str(SOURCE), 'exec'), self.m.__dict__)
        self.m.verify_video = Mock(return_value='mock_cpu_video_report.json')
        self.m.publish_best = Mock()
        self.m.ROOT = root
        self.m.GPU_LOCK = root / 'gpu.lock'
        self.m.time = types.SimpleNamespace(time=lambda: self.now, monotonic=lambda: self.now, sleep=self.sleep)
        self.m.process_token = lambda pid: self.proc_tokens.get(pid)
        self.m.subprocess = types.SimpleNamespace(
            run=Mock(side_effect=self.run), Popen=Mock(side_effect=self.popen),
            STDOUT=subprocess.STDOUT, SubprocessError=subprocess.SubprocessError)

    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.release_at is not None and self.now >= self.release_at:
            self.m.GPU_LOCK.unlink(missing_ok=True)
        if seconds == 15:
            self.proc_tokens.clear()

    def make_train(self, step=5000):
        run = self.root / 'outputs/training' / f'audit_s{step:06d}'
        run.mkdir(parents=True, exist_ok=False)
        self.put(run / 'status.json', {'status': self.training_status})
        self.put(run / 'summary.json', {'completed_updates': step + self.delta})
        if self.make_interval:
            (run / f'checkpoint_step_{step:06d}.pt').write_bytes(b'interval')
        if self.make_last:
            (run / 'checkpoint_last.pt').write_bytes(b'last')
        if self.make_sidecar:
            receipt = {'sha256': hashlib.sha256(b'last').hexdigest(), 'step': step,
                       'size': 4, 'model_only': False}
            if self.last_mutation:
                self.last_mutation(receipt)
            self.put(run / 'checkpoint_last.pt.sha256.json', receipt)
        return run

    def run(self, args, **kw):
        if args[0] == 'systemctl':
            if self.query_error:
                return subprocess.CompletedProcess(args, 1, stdout='', stderr='manager unavailable')
            state = self.states.pop(0) if self.states else 'inactive'
            pid = '100' if state in self.m.LIVE else '0'
            result = 'exit-code' if state == 'failed' else 'success'
            return subprocess.CompletedProcess(args, 0, stdout=(f'LoadState=loaded\nActiveState={state}\n'
                    f'MainPID={pid}\nResult={result}\nExecMainStatus=0\n'), stderr='')
        if args[0].endswith('/run_wam_v3_unit.sh'):
            assert not self.m.GPU_LOCK.exists(), 'launcher encountered held GPU lock'
            env = kw['env']
            self.train_envs.append(dict(env))
            self.make_train(int(env['MAX_STEPS']))
            if self.lock_after_training:
                self.m.GPU_LOCK.touch()
                self.release_at = self.now + 4
            return subprocess.CompletedProcess(args, 0)
        raise AssertionError('UNMOCKED subprocess blocked: ' + repr(args))

    def make_physical(self, step=5000):
        name = f'audit_s{step:06d}_dev10_k1'
        ck = self.root / 'outputs/training' / f'audit_s{step:06d}' / f'checkpoint_step_{step:06d}.pt'
        phys = self.root / 'outputs/behavior/closed_loop' / name
        identity = {'task': self.m.TASK, 'instance_id': 42, 'rollout_id': 0}
        episode = {**identity, 'steps': 100, 'success': False, 'q_score': {'final': 0.0},
                   'effective_policy_seed': 0, 'effective_evaluator_seed': 0}
        plan = {'task': self.m.TASK, 'mode': 'public_test', 'instance_indices': [10],
                'instance_ids': [42], 'seed': 0, 'num_rollouts': 1}
        man = {'run_name': name, 'profile': 'development', 'task_id': 0, 'task_name': self.m.TASK,
               'checkpoint': str(ck), 'checkpoint_sha256': self.m.digest(ck),
               'instances_override': [10], 'samples_averaged': 1, 'exec_horizon': 8,
               'seed': 0, 'effective_policy_seed': 0, 'effective_evaluator_seed': 0,
               'evaluation_complete': True, 'evaluator_returncode': 0, 'launcher_returncode': 0,
               'official_score_valid': False, 'expected_episodes': 1,
               'planned_episodes': [identity], 'completed_episodes': [{**episode, 'q_score': 0.0}]}
        self.mutate(man, plan, episode)
        self.put(phys / 'manifest.json', man)
        self.put(phys / 'eval/evaluation_plan.json', plan)
        self.put(phys / 'eval/json/result.json', episode)
        return phys

    def popen(self, args, **kw):
        assert args == [str(self.root / 'scripts/run_task0_closed_loop.sh')]
        assert not self.m.GPU_LOCK.exists(), 'physical overlap'
        self.phys_envs.append(dict(kw['env']))
        step = int(kw['env']['NAME'].split('_s')[1].split('_')[0])
        self.make_physical(step)
        def wait():
            if self.lock_after_physical:
                self.m.GPU_LOCK.touch()
                self.release_at = self.now + 4
            return self.physical_rc
        return types.SimpleNamespace(pid=123, wait=wait)

    def invoke(self, *extra, inherited=None, bare=False):
        args = [str(SOURCE), '--prefix', 'audit']
        if not bare:
            args += ['--config', 'config.yaml', '--cache-ledger', 'ledger.json', '--approval', 'approval.json']
        with patch.object(sys, 'argv', args + list(extra)), patch.dict(os.environ, inherited or {}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return self.m.main()

    def read(self, name='status.json'):
        return json.loads((self.out / name).read_text())

    def setup_continue(self):
        self.invoke('--plan-only')
        self.make_train()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='wam-stages-tests-')
        self.addCleanup(self.tmp.cleanup)
        self.h = Harness(Path(self.tmp.name))

    def test_video_failure_stops_before_next_training_and_no_best(self):
        self.h.m.verify_video.side_effect = ValueError('video evidence not verified')
        self.assertEqual(self.h.invoke(), 3)
        self.assertEqual(len(self.h.train_envs), 1)
        self.assertEqual(len(self.h.phys_envs), 1)
        self.h.m.publish_best.assert_not_called()
        self.assertEqual(self.h.read()['failed_phase'], 'verify_video')
        self.assertFalse((self.h.out / 'stage_005000.json').exists())

    def test_best_index_failure_preserves_completed_stage_and_no_next_train(self):
        self.h.m.publish_best.side_effect = ValueError('bad best index')
        self.assertEqual(self.h.invoke(), 3)
        self.assertEqual(len(self.h.train_envs), 1)
        self.assertEqual(self.h.read('stage_005000.json')['phase'], 'physical_complete')

    def test_default_plan_fixed_30k_5k_dev_k1(self):
        self.assertEqual(self.h.invoke('--plan-only'), 0)
        p = self.h.read('plan.json')
        self.assertEqual(p['checkpoints'], list(range(5000, 30001, 5000)))
        self.assertEqual(p['total_optimizer_steps'], 30000)
        for k, v in {'profile': 'development', 'instance': 10, 'seed': 0, 'samples': 1,
                     'exec_horizon': 8, 'task_id': 0, 'attempts_per_interval': 1,
                     'video_view': 'viewer', 'max_attempts': 3}.items():
            self.assertEqual(p['physical'][k], v)
        self.h.m.subprocess.run.assert_not_called()
        self.h.m.subprocess.Popen.assert_not_called()

    def test_six_stages_and_official_false_policy_failure_continue(self):
        self.assertEqual(self.h.invoke(), 0)
        self.assertEqual([e['MAX_STEPS'] for e in self.h.train_envs],
                         [str(n) for n in range(5000, 30001, 5000)])
        self.assertEqual(len(self.h.phys_envs), 6)
        row = self.h.read('stage_005000.json')
        self.assertIs(row['physical_manifest']['official_score_valid'], False)
        self.assertIs(row['physical_manifest']['completed_episodes'][0]['success'], False)
        self.assertFalse(self.h.read()['task_capability_claimed'])
        for first, following in zip(self.h.train_envs, self.h.train_envs[1:]):
            self.assertIn(first['NAME'] + '/checkpoint_last.pt', following['RESUME'])
            self.assertEqual(following['RESUME_SHA256'], hashlib.sha256(b'last').hexdigest())

    def test_continue_completed_idempotent_original_plan_unchanged(self):
        self.assertEqual(self.h.invoke(), 0)
        before = (self.h.out / 'plan.json').read_bytes()
        self.assertEqual(self.h.invoke('--continue-existing', bare=True), 0)
        self.assertEqual(len(self.h.train_envs), 6)
        self.assertEqual(len(self.h.phys_envs), 6)
        self.assertEqual((self.h.out / 'plan.json').read_bytes(), before)

    def test_attach_existing_active_training_never_relaunch(self):
        self.h.setup_continue()
        self.h.states = ['active', 'deactivating', 'inactive']
        self.assertEqual(self.h.invoke('--continue-existing', bare=True), 0)
        self.assertEqual(len(self.h.train_envs), 5)
        self.assertEqual(self.h.train_envs[0]['MAX_STEPS'], '10000')
        self.assertEqual(self.h.sleeps[:2], [15, 15])
        self.assertEqual(len(self.h.phys_envs), 6)

    def test_completed_stages_skip_lock_held_by_later_active_trainer(self):
        self.assertEqual(self.h.invoke(), 0)
        self.h.m.GPU_LOCK.touch()  # A later running unit must not block already-valid stage skipping.
        self.h.states = ['active']
        self.h.m.subprocess.run.reset_mock()
        self.assertEqual(self.h.invoke('--continue-existing'), 0)
        self.h.m.subprocess.run.assert_not_called()
        self.assertEqual(self.h.sleeps, [])
        self.assertTrue(self.h.m.GPU_LOCK.exists())
        self.assertEqual(len(self.h.train_envs), 6)

    def test_completed_training_only_first_physical(self):
        self.h.setup_continue()
        self.assertEqual(self.h.invoke('--continue-existing'), 0)
        self.assertEqual(len(self.h.train_envs), 5)
        self.assertEqual(self.h.phys_envs[0]['NAME'], 'audit_s005000_dev10_k1')

    def test_existing_valid_physical_without_stage_is_recovered(self):
        self.h.setup_continue()
        self.h.make_physical()
        self.assertEqual(self.h.invoke('--continue-existing'), 0)
        self.assertEqual(len(self.h.phys_envs), 5)

    def test_existing_incomplete_physical_never_retested(self):
        self.h.setup_continue()
        self.h.mutate = lambda m, p, e: m.update(evaluation_complete=False)
        self.h.make_physical()
        self.assertEqual(self.h.invoke('--continue-existing'), 3)
        self.assertEqual(self.h.train_envs, [])
        self.assertEqual(self.h.phys_envs, [])
        self.assertIn('evaluation incomplete', self.h.read()['error'])

    def test_existing_physical_active_pid_attach_no_retest(self):
        self.h.setup_continue()
        self.h.make_physical()
        self.h.put(self.h.out / 'status.json', {'step_target': 5000, 'physical_launcher_pid': 123,
                                              'physical_launcher_start': '456'})
        self.h.proc_tokens[123] = '456'
        self.assertEqual(self.h.invoke('--continue-existing'), 0)
        self.assertIn(15, self.h.sleeps)
        self.assertEqual(len(self.h.phys_envs), 5)

    def test_previous_attempt_log_without_directory_fails_closed(self):
        self.h.setup_continue()
        (self.h.out / 'physical_005000.log').touch()
        self.assertEqual(self.h.invoke('--continue-existing'), 3)
        self.assertFalse(self.h.phys_envs)

    def test_wait_release_both_directions(self):
        self.h.lock_after_training = True
        self.h.lock_after_physical = True
        self.assertEqual(self.h.invoke(), 0)
        self.assertEqual(self.h.sleeps.count(2), 24)

    def test_release_timeout_persisted_no_launch(self):
        self.h.m.GPU_LOCK.touch()
        self.assertEqual(self.h.invoke(), 3)
        self.assertEqual(self.h.now, 90)
        self.assertEqual(self.h.read()['phase'], 'stopped_infrastructure_failure')
        self.assertFalse(self.h.train_envs)
        self.assertFalse(self.h.phys_envs)

    def test_dangling_lock_symlink_fails_closed(self):
        self.h.m.GPU_LOCK.symlink_to(self.h.root / 'absent')
        self.assertEqual(self.h.invoke(), 3)
        self.assertFalse(self.h.train_envs)

    def test_systemctl_query_error_persisted(self):
        self.h.query_error = True
        self.assertEqual(self.h.invoke(), 3)
        self.assertIn('systemctl query failed', self.h.read()['error'])
        self.assertFalse(self.h.train_envs)
        self.assertFalse(self.h.phys_envs)

    def test_failed_unit_not_success_even_with_complete_json(self):
        self.h.setup_continue()
        self.h.states = ['failed']
        self.assertEqual(self.h.invoke('--continue-existing'), 3)
        self.assertFalse(self.h.phys_envs)

    def test_training_failure_and_wrong_steps_cannot_succeed(self):
        for i, change in enumerate(({'training_status': 'failed'}, {'delta': -1})):
            with self.subTest(change=change):
                h = Harness(self.h.root / str(i))
                for k, v in change.items():
                    setattr(h, k, v)
                self.assertEqual(h.invoke(), 2)
                self.assertFalse(h.phys_envs)

    def test_checkpoint_receipt_and_exact_interval_fail_closed(self):
        changes = [{'make_last': False}, {'make_sidecar': False}, {'make_interval': False},
                   {'last_mutation': lambda r: r.update(sha256='0' * 64)},
                   {'last_mutation': lambda r: r.update(step=4999)},
                   {'last_mutation': lambda r: r.update(model_only=True)},
                   {'last_mutation': lambda r: r.update(size=99)}]
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                h = Harness(self.h.root / str(i))
                for k, v in change.items():
                    setattr(h, k, v)
                self.assertEqual(h.invoke(), 3)
                self.assertFalse(h.phys_envs)

    def test_strict_physical_evidence_failures(self):
        mutations = [lambda m, p, e: m.update(evaluation_complete=1),
                     lambda m, p, e: m.update(evaluator_returncode=False),
                     lambda m, p, e: m.update(launcher_returncode=1),
                     lambda m, p, e: m.update(expected_episodes=2),
                     lambda m, p, e: m.update(planned_episodes=[]),
                     lambda m, p, e: m.update(completed_episodes=[]),
                     lambda m, p, e: m.update(checkpoint_sha256='bad'),
                     lambda m, p, e: p.update(seed=1),
                     lambda m, p, e: p.update(instance_indices=[11]),
                     lambda m, p, e: e.update(task='wrong_task'),
                     lambda m, p, e: e.update(instance_id=10),
                     lambda m, p, e: e.update(rollout_id=1),
                     lambda m, p, e: e.update(effective_policy_seed=1),
                     lambda m, p, e: e.update(success='false'),
                     lambda m, p, e: e.update(q_score={'final': True}),
                     lambda m, p, e: e.update(q_score={'final': float('nan')}),
                     lambda m, p, e: e.update(q_score={'final': 0.5})]
        for i, mutation in enumerate(mutations):
            with self.subTest(case=i):
                h = Harness(self.h.root / str(i))
                h.mutate = mutation
                self.assertEqual(h.invoke(), 3)
                self.assertEqual(len(h.train_envs), 1)
                self.assertNotEqual(h.read()['phase'], 'budget_complete')

    def test_physical_nonzero_never_resume(self):
        self.h.physical_rc = 1
        self.assertEqual(self.h.invoke(), 3)
        self.assertEqual(len(self.h.train_envs), 1)

    def test_recorded_nonzero_launcher_failure_survives_multiple_restarts(self):
        self.h.setup_continue()
        self.h.make_physical()
        self.h.put(self.h.out / 'status.json', {'step_target': 5000, 'returncode': 1})
        for _ in range(2):
            self.assertEqual(self.h.invoke('--continue-existing'), 3)
            self.assertEqual(self.h.read()['returncode'], 1)
        self.assertFalse(self.h.phys_envs)

    def test_active_physical_before_directory_creation_attaches(self):
        self.h.setup_continue()
        self.h.put(self.h.out / 'status.json', {'step_target': 5000, 'physical_launcher_pid': 123,
                                              'physical_launcher_start': '456'})
        self.h.proc_tokens[123] = '456'
        original_sleep = self.h.sleep
        def finish_physical(seconds):
            if self.h.proc_tokens:
                self.h.make_physical()
            original_sleep(seconds)
        self.h.m.time.sleep = finish_physical
        self.assertEqual(self.h.invoke('--continue-existing'), 0)
        self.assertEqual(len(self.h.phys_envs), 5)

    def test_duplicate_json_not_accepted(self):
        self.h.setup_continue()
        phys = self.h.make_physical()
        (phys / 'eval/json/duplicate.json').write_bytes((phys / 'eval/json/result.json').read_bytes())
        self.assertEqual(self.h.invoke('--continue-existing'), 3)
        self.assertFalse(self.h.phys_envs)

    def test_systemctl_timeout_persisted(self):
        self.h.m.subprocess.run.side_effect = subprocess.TimeoutExpired('systemctl', 30)
        self.assertEqual(self.h.invoke(), 3)
        self.assertIn('TimeoutExpired', self.h.read()['error'])

    def test_collected_missing_unit_is_valid_exit_evidence(self):
        original = self.h.run
        def collected(args, **kwargs):
            if args[0] == 'systemctl':
                return subprocess.CompletedProcess(args, 0, stdout=(
                    'LoadState=not-found\nActiveState=inactive\nMainPID=0\n'), stderr='')
            return original(args, **kwargs)
        self.h.m.subprocess.run.side_effect = collected
        self.assertEqual(self.h.invoke(), 0)

    def test_unknown_manager_output_never_proves_exit(self):
        self.h.m.subprocess.run.side_effect = lambda *a, **k: subprocess.CompletedProcess(
            a, 0, stdout='', stderr='')
        self.assertEqual(self.h.invoke(), 3)
        self.assertIn('unknown unit state', self.h.read()['error'])

    def test_resume_10k_does_not_launch_completed_targets(self):
        source = self.h.make_train(10000) / 'checkpoint_last.pt'
        self.assertEqual(self.h.invoke('--resume-source', str(source)), 0)
        self.assertEqual([e['MAX_STEPS'] for e in self.h.train_envs],
                         ['15000', '20000', '25000', '30000'])

    def test_fixed_environment_and_resume_hash_not_stale(self):
        bad = {k: 'stale' for k in ('RESUME', 'INIT', 'RUN_NAME', 'RESUME_SHA256', 'DRY_RUN',
                                   'MAXSTEPS', 'ORCH_EXTRA', 'VIDEO_VIEW', 'MAX_ATTEMPTS')}
        self.assertEqual(self.h.invoke(inherited=bad), 0)
        self.assertNotIn('RESUME_SHA256', self.h.train_envs[0])
        for env in self.h.phys_envs:
            self.assertEqual(env['VIDEO_VIEW'], 'viewer')
            self.assertEqual(env['MAX_ATTEMPTS'], '3')
            for key in ('RESUME_SHA256', 'DRY_RUN', 'MAXSTEPS', 'ORCH_EXTRA'):
                self.assertNotIn(key, env)

    def test_conflicting_continue_plan_rejected(self):
        self.h.invoke('--plan-only')
        before = (self.h.out / 'plan.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'budget differs'):
            self.h.invoke('--continue-existing', '--budget', '50000')
        self.assertEqual((self.h.out / 'plan.json').read_bytes(), before)
        self.h.m.subprocess.run.assert_not_called()

    def test_legacy_plan_without_viewer_fields_is_compatible(self):
        self.h.invoke('--plan-only')
        plan = self.h.read('plan.json')
        plan['physical'].pop('video_view')
        plan['physical'].pop('max_attempts')
        self.h.put(self.h.out / 'plan.json', plan)
        self.assertEqual(self.h.invoke('--continue-existing', bare=True), 0)
        self.assertEqual(self.h.read('plan.json'), plan)

    def test_no_continue_flag_refuses_existing_output(self):
        self.h.invoke('--plan-only')
        with self.assertRaises(FileExistsError):
            self.h.invoke()
        self.h.m.subprocess.run.assert_not_called()

    def test_plan_only_continue_no_process_or_plan_write(self):
        self.h.invoke('--plan-only')
        before = (self.h.out / 'plan.json').read_bytes()
        self.assertEqual(self.h.invoke('--continue-existing', '--plan-only', bare=True), 0)
        self.h.m.subprocess.run.assert_not_called()
        self.assertEqual((self.h.out / 'plan.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
