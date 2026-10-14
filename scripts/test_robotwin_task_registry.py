"""CPU-only multi-target truth and strict dispatch checks using explicit fakes."""
from dataclasses import replace
from pathlib import Path
import sys
import json
import hashlib
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from harness.core import Scope, context_fingerprint
from harness.runtime import Outcome
from harness.robotwin_policy_worker import VIEWS, envelope
from harness.robotwin_task_registry import TaskDefinition, TaskRegistry, aggregate_target_predicates
from harness.robotwin_task_supervisor import CapabilityCandidate, ModelContract, ProviderBinding, GroundingResult, VisibleResult
from harness.robotwin_strict_middle_kernel import StrictMiddleKernel, ActionRiskEvidence, RiskCalibrationBinding, command_hash


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.definition = TaskDefinition('put_bottles_dustbin', ('a', 'b', 'c'), ('inside', 'released', 'stable'))

    def test_all_three_targets_required(self):
        with self.assertRaises(ValueError):
            aggregate_target_predicates(self.definition, {'a': dict.fromkeys(self.definition.predicates, True)})

    def test_unknown_does_not_become_success(self):
        rows = {key: dict.fromkeys(self.definition.predicates, True) for key in self.definition.targets}
        rows['b']['inside'] = None
        self.assertIsNone(aggregate_target_predicates(self.definition, rows)['inside'])
        rows['c']['inside'] = False
        self.assertIs(aggregate_target_predicates(self.definition, rows)['inside'], False)

    def test_gt_and_non_boolean_rejected(self):
        rows = {key: dict.fromkeys(self.definition.predicates, True) for key in self.definition.targets}
        rows['a']['inside'] = 1
        with self.assertRaises(ValueError): aggregate_target_predicates(self.definition, rows)
        with self.assertRaises(ValueError): aggregate_target_predicates(self.definition, {'object_GT': {}})

    def test_duplicate_task_registration_rejected(self):
        registry = TaskRegistry(); registry.register(self.definition)
        self.assertIs(registry.get(self.definition.task), self.definition)
        with self.assertRaises(ValueError): registry.register(self.definition)


class FakeClient:
    def __init__(self): self.actions = self.clears = 0
    def reset(self, episode, seed): return envelope('reset_ok', episode, 0)
    def action(self, episode, step, observation):
        self.actions += 1
        return envelope('action', episode, step, raw_action=[0.] * 14)
    def clear_actions(self, episode, step, reason):
        self.clears += 1
        return envelope('cleared', episode, step, history_preserved=True, next_step=step)


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.; self.dispatched = []; self.events = []; self.client = FakeClient()
        self.definition = TaskDefinition('put_bottles_dustbin', ('a', 'b', 'c'), ('inside', 'released', 'stable'), max_age_s=2., robot_version='fake-robot')
        self.scope = Scope(self.definition.task, self.definition.task, context_fingerprint({'test': 'synthetic_not_physical'}),
                           'fake-robot', 'fake-controller', 'fake-wam', 'fake-provider', 'development', 'fake-run')
        self.candidate = CapabilityCandidate(self.definition.task, 'fake-sensor-bin-track')
        self.contract = ModelContract(self.definition.task, 'fake-wam', 'fake-robot', 'fake-controller')
        self.terminal = False; self.predicates = dict.fromkeys(self.definition.predicates)
        self.risk_unknown = False; self.wrong_action = False
        self.bound = lambda action: list(action)
        self.grounder = ProviderBinding('fake-grounder', 'v1', self.scope.key(), self.ground, trusted=True, admitted=True)
        self.verifier = ProviderBinding('fake-verifier', 'v1', self.scope.key(), self.verify, trusted=True, admitted=True, independent=True)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        receipt = Path(self.temp.name) / 'synthetic_test_calibration.json'
        receipt.write_text(json.dumps(dict(format='native14-risk-calibration-v1', admitted=True,
            provider_id='fake-test-risk', provider_version='v1', scope_key=self.scope.key(),
            calibration_id='fake-calibration', robot_version='fake-robot', action_dim=14,
            dispatch_horizon=1, evaluator_gt_used_at_runtime=False,
            validation_evidence='SYNTHETIC CPU FIXTURE, NOT PHYSICAL ADMISSION')))
        self.risk_binding = RiskCalibrationBinding(ProviderBinding('fake-test-risk', 'v1', self.scope.key(),
            self.risk, trusted=True, admitted=True, independent=True), 'fake-calibration', str(receipt),
            hashlib.sha256(receipt.read_bytes()).hexdigest())
        for name in ('harness.robotwin_full_middle_bridge.robot_feedback', 'harness.robotwin_full_middle_bridge.hold_measured', 'harness.robotwin_strict_middle_kernel.hold_measured'):
            p = patch(name, return_value={'test_only': True}); p.start(); self.addCleanup(p.stop)
        self.kernel = self.build()
        self.kernel.reset('ep', 1, self.candidate)

    def ground(self, frame):
        return GroundingResult('fake-grounder', 'v1', self.scope.key(), frame.episode_id, frame.observation.observation_id,
                              frame.observation.observed_at, 'execute', (self.candidate,))

    def verify(self, frame):
        return VisibleResult('fake-verifier', 'v1', self.scope.key(), frame.episode_id, frame.observation.observation_id,
                             frame.observation.observed_at, Outcome(self.predicates, self.terminal, 'fake-test-sensor',
                             'perception_predicates', frame.observation.observation_id), candidate=self.candidate)

    def risk(self, action, frame):
        oid, at = frame.observation.observation_id, frame.observation.observed_at
        return ActionRiskEvidence(oid, 'wrong' if self.wrong_action else command_hash(action), at, 'fake-calibration',
                                  True, None if self.risk_unknown else .01, 1, 'fake-test-risk')

    def build(self, **overrides):
        args = dict(definition=self.definition, scope=self.scope, contract=self.contract, client=self.client,
                    driver=lambda command: self.dispatched.append(command) or True, robot=lambda: object(),
                    visible_tracks=lambda sensors: (self.candidate.object_id,), grounder=self.grounder, verifier=self.verifier,
                    bound_command=self.bound, risk_binding=self.risk_binding, max_risk=.05,
                    event=self.events.append, clock=lambda: self.now)
        args.update(overrides)
        return StrictMiddleKernel(**args)

    def prepare(self):
        return self.kernel.prepare('ep', 0, 'frame0', self.now, {'rgb': {v: np.zeros((8, 8, 3), np.uint8) for v in VIEWS},
                                   'commanded_joint_gripper_state': [0.] * 14})

    def test_missing_or_unadmitted_verifier_fails_before_model(self):
        for verifier in (None, replace(self.verifier, admitted=False), replace(self.verifier, independent=False)):
            with self.assertRaises(ValueError): self.build(verifier=verifier)
        self.assertEqual(self.client.actions, 0)

    def test_unadmitted_risk_provider_or_changed_receipt_rejected(self):
        invalid = replace(self.risk_binding, provider=replace(self.risk_binding.provider, admitted=False))
        with self.assertRaises(ValueError): self.build(risk_binding=invalid)
        Path(self.risk_binding.receipt_path).write_text('{}')
        with self.assertRaises(ValueError): self.build()

    def test_semantic_target_set_is_not_a_physical_track(self):
        with self.assertRaises(ValueError):
            self.kernel.reset('other', 2, CapabilityCandidate(self.definition.task, self.definition.target_set_key))

    def test_unknown_outcome_executes_without_success(self):
        result = self.prepare(); self.assertEqual(result.mode, 'EXECUTE'); self.assertIsNone(result.verified_outcome)
        self.assertTrue(self.kernel.execute('ep', 0, 'frame0')); self.assertEqual(len(self.dispatched), 1)
        with self.assertRaises(ValueError): self.kernel.execute('ep', 0, 'frame0')

    def test_success_is_controlled_done_without_wam(self):
        self.terminal = True; self.predicates = dict.fromkeys(self.definition.predicates, True)
        result = self.prepare(); self.assertEqual(result.mode, 'DONE'); self.assertEqual(self.client.actions, 0)

    def test_unknown_risk_revokes_without_dispatch(self):
        self.risk_unknown = True; result = self.prepare()
        self.assertEqual(result.mode, 'HOLD'); self.assertTrue(result.transaction_revoked_terminal)
        self.assertFalse(self.dispatched); self.assertIsNone(self.kernel.adapter.pending); self.assertTrue(self.kernel.adapter.failed)

    def test_wrong_final_action_hash_rejected(self):
        self.wrong_action = True; self.assertEqual(self.prepare().mode, 'HOLD'); self.assertFalse(self.dispatched)

    def test_expired_after_prepare_never_dispatches(self):
        self.prepare(); self.now += 3.
        result = self.kernel.execute('ep', 0, 'frame0'); self.assertEqual(result.mode, 'HOLD'); self.assertFalse(self.dispatched)
        self.assertFalse(result.dispatch_ack); self.assertFalse(result)

    def test_driver_bounds_changed_after_check_rejected(self):
        self.prepare(); self.kernel.bound_command = lambda action: [1.] * 14
        with self.assertRaises(ValueError): self.kernel.execute('ep', 0, 'frame0')
        self.assertFalse(self.dispatched)

    def test_provider_exception_never_dispatches(self):
        def failed(*args): raise ValueError('test provider error')
        self.kernel.risk_binding = replace(self.risk_binding, provider=replace(self.risk_binding.provider, callback=failed)); result = self.prepare()
        self.assertEqual(result.reason, 'command_or_risk_provider_error'); self.assertFalse(self.dispatched)


if __name__ == '__main__': unittest.main()
