#!/usr/bin/env python3
"""CPU-only phase-journal tests; all artifacts live in temporary directories."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_robotwin_wam as evaluation


class PhaseJournalTests(unittest.TestCase):
    def test_written_before_call_and_no_payload_serialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = evaluation.PhaseJournal(tmp)
            journal.context('episode_001_seed10001', 10001, 141)
            def observe(secret):
                state = json.loads(journal.path.read_text())
                self.assertEqual((state['phase'], state['status']), ('get_obs', 'running'))
                self.assertEqual((state['episode'], state['seed'], state['step']),
                                 ('episode_001_seed10001', 10001, 141))
                self.assertEqual(state['pid'], os.getpid())
                self.assertIsInstance(state['timestamp'], float)
                return secret
            secret = {'object_GT': 'not-for-policy'}
            self.assertIs(journal.call('get_obs', observe, secret), secret)
            state = json.loads(journal.path.read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['sequence'], 2)
            self.assertNotIn('object_GT', journal.path.read_text())
            self.assertFalse(journal.path.with_suffix('.json.tmp').exists())

    def test_baseexception_survives_cleanup_and_terminal_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = evaluation.PhaseJournal(tmp)
            journal.context('ep', 10001, 141)
            error = KeyboardInterrupt('synthetic interrupt')
            def fail():
                raise error
            with self.assertRaises(KeyboardInterrupt) as caught:
                journal.call('get_obs', fail)
            self.assertIs(caught.exception, error)
            self.assertEqual(json.loads(journal.path.read_text())['status'], 'error')
            journal.call('cleanup_scene', lambda: None)
            journal.write('run_finished', 'error')
            state = json.loads(journal.path.read_text())
            self.assertEqual(state['failures'][0]['phase'], 'get_obs')
            self.assertEqual(state['failures'][0]['step'], 141)
            self.assertIn('KeyboardInterrupt', state['failures'][0]['error'])

    def test_atomic_replace_failure_leaves_previous_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = evaluation.PhaseJournal(tmp)
            journal.write('native_action', 'completed')
            previous = journal.path.read_bytes()
            with patch.object(Path, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaises(OSError):
                    journal.call('get_obs', lambda: self.fail('must not enter call'))
            self.assertEqual(journal.path.read_bytes(), previous)
            self.assertEqual(json.loads(previous)['phase'], 'native_action')

    def run_fake(self, fail_observation=False, fail_cleanup=False):
        # Existing CPU-only fixtures; importing them does not execute their suite.
        from scripts.test_robotwin_eval_bridge import InjectedEnv, InjectedPolicy, InjectedVideo
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = evaluation.parse_args(['--run-dir', tmp, '--checkpoint', '/unused',
                '--training-run-dir', '/unused', '--plan-only', '--seeds', '10000', '10001'])
            journal = evaluation.PhaseJournal(out)
            states, environments = [], []
            original_write = journal.write
            def write(*a, **kw):
                original_write(*a, **kw)
                states.append(json.loads(journal.path.read_text()))
            journal.write = write
            def factory():
                env = InjectedEnv()
                original_obs = env.get_obs
                def get_obs():
                    state = json.loads(journal.path.read_text())
                    self.assertEqual((state['phase'], state['status']), ('get_obs', 'running'))
                    if fail_observation and len(environments) == 2 and env.take_action_cnt == 2:
                        raise RuntimeError('synthetic native observation failure')
                    return original_obs()
                env.get_obs = get_obs
                if fail_cleanup:
                    def close():
                        raise RuntimeError('synthetic cleanup failure')
                    env.close_env = close
                environments.append(env)
                return env
            policy = InjectedPolicy()
            result = evaluation.run_evaluation(args, {'task_kwargs': {}}, out,
                client_factory=lambda *a: policy, env_factory=factory,
                video_factory=lambda *a: (InjectedVideo(), None), owner_check=lambda *a: None,
                guard_factory=lambda: SimpleNamespace(check=lambda: {}, peaks={}), journal=journal)
            return result, states, copy.deepcopy(journal.state), environments, policy

    def test_two_episode_progression_and_unchanged_scoring(self):
        result, states, final, envs, policy = self.run_fake()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['native_success_rate'], 0)
        self.assertEqual([len(env.actions) for env in envs], [3, 3])
        self.assertTrue(policy.closed)
        self.assertEqual(final['phase'], 'run_finished')
        self.assertEqual(final['failures'], [])
        running = [s for s in states if s['status'] == 'running']
        for phase in ('policy_action', 'native_action', 'get_obs', 'gt_snapshot',
                      'gt_check_success', 'video_capture', 'cleanup_scene', 'cleanup_video', 'cleanup_policy'):
            self.assertIn(phase, [s['phase'] for s in running])
        for i, state in enumerate(states[:-1]):
            if state['phase'] == 'get_obs' and state['status'] == 'running':
                self.assertEqual(states[i+1]['phase'], 'get_obs')
                self.assertEqual(states[i+1]['status'], 'completed')
        self.assertEqual([s['step'] for s in running if s['phase'] == 'native_action'], [0, 1, 2, 0, 1, 2])

    def test_second_episode_observation_failure_preserved_after_cleanup(self):
        result, states, final, envs, policy = self.run_fake(fail_observation=True)
        self.assertEqual(result['status'], 'infrastructure_error')
        failure = final['failures'][0]
        self.assertEqual((failure['phase'], failure['seed'], failure['step']), ('get_obs', 10001, 1))
        self.assertEqual(final['status'], 'error')
        self.assertIsNone(result['episodes'][1]['evidence']['native_benchmark_success'])
        self.assertTrue(all(env.closed for env in envs) and policy.closed)

    def test_cleanup_failure_recorded(self):
        result, states, final, envs, policy = self.run_fake(fail_cleanup=True)
        self.assertEqual(result['status'], 'infrastructure_error')
        self.assertEqual(final['failures'][0]['phase'], 'cleanup_scene')
        self.assertEqual(result['episodes'][1]['status'], 'not_started')
        self.assertTrue(policy.closed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
