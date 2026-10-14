"""CPU fake-client/driver tests. No model, Scene, GPU, or physical success claim."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from harness.core import Memory
from harness.runtime import Outcome
from harness.robotwin_policy_worker import ChunkPolicy, VIEWS, envelope
from harness.robotwin_system_adapter import RobotwinSystemAdapter, CapabilityUnavailable


def observation(step=0):
    return {'rgb': {v: np.full((2, 3, 3), step, np.uint8) for v in VIEWS},
            'commanded_joint_gripper_state': np.arange(14, dtype=np.float32) / 20}


class Context:
    def reset(self):
        self.states = []
    def push(self, state, pool):
        self.states.append(np.asarray(state).copy())
    def inputs(self):
        return len(self.states)


class FakeClient:
    """Same reset/action signatures and actual ChunkPolicy's CPU queue/history logic."""
    def __init__(self):
        self.calls, self.samples, self.encodes = [], [], []
        self.context = Context()
        self.policy = ChunkPolicy(self.context, self.encode, self.sample)
        self.corrupt = None
    def encode(self, rgb):
        self.encodes.append(tuple(int(rgb[v][0, 0, 0]) for v in VIEWS))
        return None, np.zeros((3, 2))
    def sample(self, features, count, seed, step):
        self.samples.append((count, seed, step))
        # Includes out-of-bound grippers: adapter must not alter the raw prediction.
        return np.arange(32 * 14, dtype=np.float32).reshape(32, 14) / 100 + step
    def reset(self, episode, seed):
        self.calls.append(('reset', episode, seed))
        self.policy.reset(episode, seed)
        return envelope('reset_ok', episode, 0)
    def action(self, episode, step, obs):
        self.calls.append(('action', episode, step))
        reply = envelope('action', episode, step, **self.policy.observe(episode, step, obs))
        if self.corrupt:
            reply.update(self.corrupt)
        return reply


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.now, self.executed = 10., []
        self.client = FakeClient()
        self.adapter = RobotwinSystemAdapter(self.client, self.driver, max_age_s=2., clock=lambda: self.now)
        self.adapter.reset('ep', 123)
    def driver(self, raw):
        self.executed.append(list(raw))
        return True
    def prepare(self, step=0, **kw):
        args = dict(episode='ep', step=step, observation_id=f'obs{step}', observed_at=self.now,
                    observation=observation(step))
        args.update(kw)
        return self.adapter.prepare(**args)

    def test_transparent_trace_matches_unwrapped_chunk8(self):
        baseline = FakeClient()
        baseline.reset('ep', 123)
        baseline_actions = []
        for step in range(17):
            expected = baseline.action('ep', step, observation(step))
            actual = self.prepare(step)
            self.assertEqual(actual, expected)
            baseline_actions.append(expected['raw_action'])
            self.assertTrue(self.adapter.execute('ep', step, f'obs{step}'))
            self.now += .1
        self.assertEqual(self.executed, baseline_actions)
        self.assertEqual(self.client.calls, baseline.calls)
        self.assertEqual(self.client.encodes, baseline.encodes)
        self.assertEqual(self.client.samples, [(1, 123, 0), (9, 123, 8), (17, 123, 16)])
        self.assertEqual(len(self.client.context.states), 17)
        np.testing.assert_array_equal(self.client.context.states, baseline.context.states)
        self.assertEqual(sum(c[0] == 'reset' for c in self.client.calls), 1)
        self.assertFalse(hasattr(self.adapter, 'memory'))

    def test_returned_reply_mutation_cannot_change_dispatch(self):
        reply = self.prepare()
        expected = deepcopy(reply['raw_action'])
        reply['raw_action'][0] = 999
        self.adapter.execute('ep', 0, 'obs0')
        self.assertEqual(self.executed, [expected])

    def test_stale_future_or_nan_capture_never_calls_policy(self):
        for captured in (7., 11., float('nan')):
            with self.subTest(captured=captured):
                a = RobotwinSystemAdapter(self.client, self.driver, max_age_s=2., clock=lambda: 10.)
                a.reset(str(captured), 123)
                before = len(self.client.calls)
                with self.assertRaises(ValueError):
                    a.prepare(str(captured), 0, 'x', captured, observation())
                self.assertEqual(len(self.client.calls), before)
        self.assertEqual(self.executed, [])

    def test_expired_after_inference_burns_transaction_without_reset(self):
        self.prepare()
        self.now = 13.
        with self.assertRaises(ValueError):
            self.adapter.execute('ep', 0, 'obs0')
        self.now = 10.
        with self.assertRaises(ValueError):
            self.adapter.execute('ep', 0, 'obs0')
        with self.assertRaises(ValueError):
            self.prepare(1)
        self.assertEqual(self.executed, [])
        self.assertEqual(len(self.client.context.states), 1)
        self.assertEqual(self.client.calls, [('reset', 'ep', 123), ('action', 'ep', 0)])

    def test_wrong_request_ids_and_duplicate_observation(self):
        for kw in ({'episode': 'other'}, {'step': 1}, {'step': False}):
            with self.assertRaises(ValueError):
                self.prepare(**kw)
        self.prepare()
        with self.assertRaises(ValueError):
            self.prepare(1)
        for args in [('other', 0, 'obs0'), ('ep', 1, 'obs0'), ('ep', 0, 'wrong'), ('ep', False, 'obs0')]:
            with self.assertRaises(ValueError):
                self.adapter.execute(*args)
        self.assertEqual(self.executed, [])
        self.adapter.execute('ep', 0, 'obs0')
        with self.assertRaises(ValueError):
            self.prepare(1, observation_id='obs0')

    def test_wrong_worker_reply_poisoned_no_retry(self):
        self.client.corrupt = {'episode': 'wrong'}
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.client.corrupt = None
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(len(self.client.context.states), 1)
        self.assertEqual(self.executed, [])

    def test_execute_ack_exactly_once(self):
        self.prepare()
        self.assertIs(self.adapter.execute('ep', 0, 'obs0'), True)
        with self.assertRaises(ValueError):
            self.adapter.execute('ep', 0, 'obs0')
        self.assertEqual(len(self.executed), 1)

    def test_native_failure_never_retries_or_resets(self):
        for failure in (False, None, 1, OSError('native submission uncertain')):
            with self.subTest(failure=failure):
                client, calls = FakeClient(), []
                def driver(raw):
                    calls.append(raw)
                    if isinstance(failure, Exception):
                        raise failure
                    return failure
                a = RobotwinSystemAdapter(client, driver, max_age_s=2., clock=lambda: 10.)
                a.reset('ep', 123)
                a.prepare('ep', 0, 'x', 10., observation())
                with self.assertRaises((RuntimeError, OSError)):
                    a.execute('ep', 0, 'x')
                with self.assertRaises(ValueError):
                    a.execute('ep', 0, 'x')
                with self.assertRaises(ValueError):
                    a.prepare('ep', 1, 'y', 10., observation())
                self.assertEqual(len(calls), 1)
                self.assertEqual(client.calls, [('reset', 'ep', 123), ('action', 'ep', 0)])

    def test_recursive_gt_fields_and_provenance_rejected(self):
        for key in ('gt_pose', 'groundTruth', 'native_success', 'eval_success', 'arm_tag',
                    'object_pose_world', 'cabinet_joint_qpos', 'reward', 'third_view', 'bddl'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.adapter.record_visible_outcome({'nested': [{'deeper': {key: True}}]})
        for source in ('evaluation_gt.jsonl', 'simulator_predicates', 'native_score', 'ground_truth'):
            with self.assertRaises(ValueError):
                self.adapter.record_visible_outcome(Outcome({'whole_task': True}, True, source,
                                                           'perception_predicates', 'obs0'))
        obs = observation()
        obs['rgb']['head_camera'] = {'gt_pose': [1, 2, 3]}
        with self.assertRaises(ValueError):
            self.prepare(observation=obs)
        self.assertEqual(self.client.calls, [('reset', 'ep', 123)])

    def test_unknown_never_records_memory_or_trusts_visible_label(self):
        visible = Outcome({'whole_task': True}, True, 'onboard_review', 'perception_predicates', 'obs0')
        with patch.object(Memory, 'record', side_effect=AssertionError('must not write')) as write:
            self.assertIsNone(self.adapter.record_visible_outcome())
            self.assertIsNone(self.adapter.record_visible_outcome(visible))
            write.assert_not_called()

    def test_pending_replan_terminates_without_resetting_worker_history(self):
        self.prepare()
        old_queue = len(self.client.policy.queue)
        with self.assertRaises(ValueError):
            self.adapter.replan()
        with self.assertRaises(ValueError):
            self.adapter.execute('ep', 0, 'obs0')
        self.assertEqual(len(self.client.policy.queue), old_queue)
        self.assertEqual(len(self.client.context.states), 1)
        self.assertEqual(self.client.calls, [('reset', 'ep', 123), ('action', 'ep', 0)])

    def test_acknowledged_replan_clears_queue_not_history(self):
        self.prepare()
        self.adapter.execute('ep', 0, 'obs0')
        def clear(episode, step, reason):
            result = self.client.policy.clear_actions(episode, step, reason)
            from harness.robotwin_policy_worker import envelope
            return envelope('cleared', episode, step, **result)
        self.client.clear_actions = clear
        result = self.adapter.replan('independent_reobserve')
        self.assertEqual(result['discarded_actions'], 7)
        self.assertTrue(result['history_preserved'])
        self.assertEqual(len(self.client.context.states), 1)
        self.assertEqual(self.adapter.next_step, 1)

    def test_acknowledged_replan_missing_capability_fails_closed(self):
        self.prepare()
        self.adapter.execute('ep', 0, 'obs0')
        with self.assertRaises(CapabilityUnavailable):
            self.adapter.replan()
        self.assertTrue(self.adapter.failed)

    def test_reset_is_explicit_new_episode_only(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.adapter.reset('new', 456)
        self.adapter.execute('ep', 0, 'obs0')
        with self.assertRaises(ValueError):
            self.adapter.reset('ep', 123)
        self.adapter.reset('new', 456)
        self.assertEqual(len(self.client.context.states), 0)
        self.assertEqual(self.adapter.next_step, 0)

    def test_bad_modalities_rejected_before_policy(self):
        for mutate in (lambda o: o.update(actual_qpos=o.pop('commanded_joint_gripper_state')),
                       lambda o: o.update(task_cond=np.zeros(1024)),
                       lambda o: o['rgb'].pop('right_camera'),
                       lambda o: o.update(commanded_joint_gripper_state=np.zeros(61)),
                       lambda o: o['rgb'].update(head_camera=np.zeros((2, 2, 3)))):
            client = FakeClient()
            a = RobotwinSystemAdapter(client, self.driver, max_age_s=2., clock=lambda: 10.)
            a.reset('ep', 123)
            obs = observation()
            mutate(obs)
            with self.assertRaises((ValueError, RuntimeError)):
                a.prepare('ep', 0, 'x', 10., obs)
            self.assertEqual(client.calls, [('reset', 'ep', 123)])

    def test_no_model_or_scene_imports(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('sapien', sys.modules)
        with self.assertRaises(ValueError):
            self.adapter.record_visible_outcome(np.array([{'gt_score': 1}], dtype=object))

    def test_adapter_remains_bounded_with_replan_extension(self):
        path = Path(__file__).resolve().parents[1] / 'harness/robotwin_system_adapter.py'
        self.assertLessEqual(len(path.read_text().splitlines()), 130)


if __name__ == '__main__':
    unittest.main()
