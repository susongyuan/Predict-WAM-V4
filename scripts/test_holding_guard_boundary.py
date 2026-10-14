#!/usr/bin/env python3
"""CPU instance-boundary regressions for the opt-in holding guard; no model or simulation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from types import SimpleNamespace as NS
import unittest

import numpy as np
from harness.holding_guard import HoldingGuard
from harness.policy_server import WamPolicy
from harness.task0_predicates import GUARDED_SOURCE


class HoldingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.policy = WamPolicy.__new__(WamPolicy)
        self.policy.episode = 2
        self.policy.step_in_episode = 22
        self.policy.holding_guard = HoldingGuard(episode=2)
        self.policy.orch = NS(contract=None)
        self.events = []
        self.policy._emit = self.events.append
        self.state = np.zeros(61, dtype=np.float32)
        self.state[49:51] = .019
        self.action = np.zeros(23, dtype=np.float32)
        self.action[14] = self.action[22] = 1.0

    def contract(self, skill='press', objects=('radio',)):
        self.policy.orch.contract = NS(step=NS(skill=skill, objects=objects), started_step=21)

    def latch(self):
        evidence = dict(source=GUARDED_SOURCE, verified=True,
                        verifier_kind='controller_postconditions', outcome_observable=True,
                        satisfied=True, failed=False, reason='held_and_lifted', episode=2,
                        observation_id=20, started_step=0, key='pick up from(radio,coffee_table)',
                        objects=['radio', 'coffee_table'], hand='right')
        self.assertTrue(self.policy.holding_guard.accept_pick(
            evidence, episode=2, step=20, started_step=0, key=evidence['key'],
            objects=('radio', 'coffee_table'), hand='right'))

    def test_enabled_empty_guard_execute_instance(self):
        self.contract()
        out = self.policy._return_action(self.action, self.state, executing=True)
        np.testing.assert_array_equal(out.numpy(), self.action)

    def test_enabled_empty_guard_hold_instance(self):
        out = self.policy._return_action(self.action, self.state, executing=False)
        np.testing.assert_array_equal(out.numpy(), self.action)

    def test_held_press_rewrites_only_held_hand_and_last_action(self):
        self.latch()
        self.contract()
        out = self.policy._return_action(self.action, self.state, executing=True)
        expected = self.action.copy()
        expected[22] = -1.0
        np.testing.assert_array_equal(out.numpy(), expected)
        np.testing.assert_array_equal(self.policy._last_action, expected)
        self.assertEqual(self.action[22], 1.0)
        self.assertTrue(any(e['event'] == 'holding_guard_rewrite' for e in self.events))

    def test_place_permission_does_not_release_during_hold(self):
        self.latch()
        self.contract('place on', ('radio', 'coffee_table'))
        out = self.policy._return_action(self.action, self.state, executing=False)
        self.assertEqual(float(out[22]), -1.0)

    def test_matching_executing_place_allows_release(self):
        self.latch()
        self.contract('place on', ('radio', 'coffee_table'))
        out = self.policy._return_action(self.action, self.state, executing=True)
        self.assertEqual(float(out[22]), 1.0)
        np.testing.assert_array_equal(self.policy._last_action, self.action)

    def test_wrong_object_place_cannot_release_held_object(self):
        self.latch()
        self.contract('place on', ('other_radio', 'coffee_table'))
        out = self.policy._return_action(self.action, self.state, executing=True)
        self.assertEqual(float(out[22]), -1.0)

    def test_disabled_guard_preserves_old_boundary(self):
        self.policy.holding_guard = None
        out = self.policy._return_action(self.action, self.state, executing=True)
        np.testing.assert_array_equal(out.numpy(), self.action)
        np.testing.assert_array_equal(self.policy._last_action, self.action)


if __name__ == '__main__':
    unittest.main()
