#!/usr/bin/env python3
"""CPU execution of policy lifecycle methods; no model, simulator, or success labels."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'harness/policy_server.py'


def method(name):
    tree = ast.parse(POLICY.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WamPolicy')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = {'np': np, 'time': time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(POLICY), 'exec'), namespace)
    return namespace[name]


def step(skill, *objects):
    return SimpleNamespace(skill=skill, objects=objects, key=lambda: f'{skill}({",".join(objects)})')


def program(target='object_a'):
    return [step('move to', target), step('pick up from', target, 'support_a'),
            step('move to', 'support_b'), step('place on next to', target, 'support_b', 'landmark'),
            step('move to', 'object_b'), step('pick up from', 'object_b', 'support_c')]


def policy(steps=None, index=0, status='running'):
    steps = program() if steps is None else steps
    producer = Mock()
    producer.requires_applied_action = True
    producer.update.return_value = {'verified': False, 'admitted': False, 'status': 'unknown'}
    p = SimpleNamespace(orch=SimpleNamespace(candidates=[SimpleNamespace(steps=steps)],
                        index=index, status=status, contract=None),
                        onboard_measurement=producer, onboard_measurement_every=1,
                        _onboard_measurement_binding=None, episode=1, step_in_episode=0,
                        _last_action=None, _emit=Mock())
    for name in ('_onboard_observation_request', '_log_task090_measurement'):
        setattr(p, name, MethodType(method(name), p))
    return p


def observe(p):
    p._log_task090_measurement({'sensor': 'sentinel'}, np.zeros(61), 'features')
    p.step_in_episode += 1
    return p._emit.call_args.args[0]


class LifecycleTests(unittest.TestCase):
    def test_navigation_requests_next_manipulated_object_not_destination(self):
        p = policy(index=2)
        request = p._onboard_observation_request()
        self.assertEqual(request['object_id'], 'object_a')
        self.assertEqual(request['measurement_contract_key'], program()[3].key())
        self.assertFalse(request['observed_identity'])

    def test_first_frame_observed_without_existing_contract_or_action(self):
        p = policy()
        event = observe(p)
        self.assertEqual(p.onboard_measurement.update.call_args.kwargs['object_id'], 'object_a')
        self.assertIsNone(p.onboard_measurement.update.call_args.kwargs['action'])
        self.assertTrue(event['diagnostic_only'])
        self.assertFalse(event['measurement']['verified'])

    def test_navigation_pick_transport_place_share_observer_lifetime(self):
        p = policy()
        for index in range(4):
            p.orch.index = index
            observe(p)
        self.assertEqual(p.onboard_measurement.reset.call_count, 1)
        self.assertEqual(p.onboard_measurement.update.call_count, 4)
        self.assertEqual([c.kwargs['step'] for c in p.onboard_measurement.update.call_args_list], list(range(4)))

    def test_changed_target_resets_before_measurement(self):
        p = policy()
        observe(p)
        p.orch.index = 4
        observe(p)
        self.assertEqual(p.onboard_measurement.reset.call_count, 2)
        self.assertEqual(p.onboard_measurement.update.call_args.kwargs['object_id'], 'object_b')

    def test_episode_change_never_reuses_previous_binding(self):
        p = policy()
        observe(p)
        p.episode += 1
        p.step_in_episode = 0
        observe(p)
        self.assertEqual(p.onboard_measurement.reset.call_count, 2)

    def test_hold_continues_observation_without_claiming_success(self):
        p = policy(index=1, status='held')
        for _ in range(3):
            event = observe(p)
            self.assertFalse(event['measurement']['verified'])
        self.assertEqual(p.onboard_measurement.update.call_count, 3)
        self.assertEqual(p.orch.index, 1)
        self.assertEqual(p.orch.status, 'held')

    def test_ambiguous_request_invalidates_old_target_without_guessing(self):
        p = policy()
        observe(p)
        p.orch.candidates.append(SimpleNamespace(steps=program('different_object')))
        event = observe(p)
        self.assertIsNone(event['observation_request'])
        self.assertIsNone(p._onboard_measurement_binding)
        self.assertEqual(p.onboard_measurement.update.call_count, 1)
        self.assertEqual(p.onboard_measurement.reset.call_count, 2)

    def test_same_target_different_skill_is_not_resolved_by_guessing(self):
        p = policy()
        alternative = program()
        alternative[1] = step('place in', 'object_a', 'support_a')
        p.orch.candidates.append(SimpleNamespace(steps=alternative))
        self.assertIsNone(p._onboard_observation_request())

    def test_consensus_does_not_require_choosing_between_equivalent_requests(self):
        p = policy()
        p.orch.candidates.append(SimpleNamespace(steps=program()))
        self.assertEqual(p._onboard_observation_request()['object_id'], 'object_a')

    def test_consecutive_navigation_cannot_borrow_later_manipulation(self):
        p = policy([step('move to', 'waypoint'), *program()])
        self.assertIsNone(p._onboard_observation_request())

    def test_end_of_program_and_no_program_are_unknown(self):
        for p in (policy(index=6), policy(status='done'), policy([]), policy([step('move to', 'goal')])):
            with self.subTest(p=p):
                self.assertIsNone(p._onboard_observation_request())

    def test_stale_contract_cannot_rebind_to_another_program_step(self):
        p = policy()
        p.orch.contract = SimpleNamespace(index=1, key=program()[1].key())
        self.assertIsNone(p._onboard_observation_request())
        p.orch.contract = SimpleNamespace(index=0, key='different program')
        self.assertIsNone(p._onboard_observation_request())

    def test_matching_live_contract_is_accepted_as_a_request_only(self):
        p = policy(index=1)
        p.orch.contract = SimpleNamespace(index=1, key=program()[1].key())
        self.assertEqual(p._onboard_observation_request()['object_id'], 'object_a')

    def test_interval_and_disabled_diagnostics_do_not_call_producer(self):
        p = policy()
        p.onboard_measurement_every = 5
        p.step_in_episode = 1
        p._log_task090_measurement({}, np.zeros(61), None)
        p.onboard_measurement.update.assert_not_called()
        p.onboard_measurement = None
        p.step_in_episode = 5
        p._log_task090_measurement({}, np.zeros(61), None)
        p._emit.assert_not_called()

    def test_action_and_state_are_copies_not_policy_mutations(self):
        p = policy()
        p._last_action = np.arange(23, dtype=np.float32)
        state = np.arange(61, dtype=float)
        def mutate(obs, s, **kw):
            s[:] = -5
            kw['action'][:] = -6
            return {'verified': False, 'admitted': False}
        p.onboard_measurement.update.side_effect = mutate
        p._log_task090_measurement({}, state, None)
        np.testing.assert_array_equal(state, np.arange(61))
        np.testing.assert_array_equal(p._last_action, np.arange(23))

    def test_producer_success_claim_is_rejected_not_returned(self):
        p = policy()
        p.onboard_measurement.update.return_value = {'verified': True, 'admitted': True, 'satisfied': True}
        event = observe(p)
        self.assertEqual(event['measurement']['status'], 'diagnostic_error')
        self.assertFalse(event['measurement']['verified'])
        self.assertFalse(event['measurement']['admitted'])
        self.assertEqual(p.orch.index, 0)

    def test_producer_exception_is_logged_not_success(self):
        p = policy()
        p.onboard_measurement.update.side_effect = ValueError('missing depth')
        event = observe(p)
        self.assertEqual(event['measurement']['reason'], 'ValueError:missing depth')
        self.assertFalse(event['measurement']['verified'])

    def test_reset_method_resets_observer_and_request(self):
        p = policy()
        observe(p)
        p.finish_episode = Mock()
        p._reset_v3_contexts = Mock()
        p._proprio_buf = []
        p.program = None
        method('reset')(p)
        self.assertIsNone(p._onboard_measurement_binding)
        self.assertEqual(p.onboard_measurement.reset.call_count, 2)
        self.assertEqual(p.episode, 2)
        self.assertEqual(p.step_in_episode, 0)

    def test_real_pre_tick_path_observes_once_in_navigation_manipulation_and_hold(self):
        class AtTick(Exception):
            pass
        for index, status in ((0, 'running'), (1, 'running'), (2, 'running'), (1, 'held')):
            with self.subTest(index=index, status=status):
                p = policy(index=index, status=status)
                s = program()[index]
                p.orch.contract = SimpleNamespace(index=index, key=s.key(), step=s,
                    is_navigation=s.skill == 'move to', precondition_key=None)
                p._grounded = True
                p.postcondition_mode = 'task090_observed'
                p._find = lambda *a: np.zeros((4, 4, 3), np.uint8)
                p.post_task1 = Mock()
                p._active_contract = 'active'
                p._contract_id = lambda c: 'active'
                post = {'verified': False, 'satisfied': False, 'source': 'unchanged_postcondition_gate'}
                p._task090_post_update = Mock(return_value=post)
                p.orch.tick = Mock(side_effect=AtTick)
                with self.assertRaises(AtTick):
                    method('_act_orchestrated')(p, {}, np.zeros(61), 'features', None, time.monotonic())
                self.assertEqual(p.onboard_measurement.update.call_count, 1)
                forwarded = p.orch.tick.call_args.kwargs['postcondition']
                if s.skill == 'move to':
                    self.assertIsNone(forwarded)
                    p._task090_post_update.assert_not_called()
                else:
                    self.assertIs(forwarded, post)
                    p._task090_post_update.assert_called_once()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = time.monotonic()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleTests))
    report = {'scope': 'CPU policy method/lifecycle regression, no physical task execution',
              'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'passed': result.wasSuccessful(), 'wall_seconds': time.monotonic()-started,
              'gpu_used': False, 'simulator_executed': False, 'physical_success': None,
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (POLICY, Path(__file__))}}
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'unittest.log').write_text(stream.getvalue())
    print(stream.getvalue())
    print(json.dumps(report, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
