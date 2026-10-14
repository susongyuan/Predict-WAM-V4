"""Synthetic integration checks; fake providers do not validate physical sensing."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.core import Scope, context_fingerprint
from harness.robotwin_full_middle_bridge import FullMiddleBridge, SupervisorStop, TrainedVTTContract
from harness.robotwin_policy_worker import envelope, VIEWS, sha256
from harness.robotwin_system_adapter import RobotwinSystemAdapter
from harness.robotwin_task_supervisor import (
    CapabilityCandidate, GroundingResult, ModelContract, ProviderBinding,
    TaskSupervisor, VisibleResult,
)
from harness.runtime import Outcome


class Client:
    def __init__(self):
        self.calls = []
    def reset(self, episode, seed):
        return envelope('reset_ok', episode, 0)
    def action(self, episode, step, observation):
        self.calls.append(('action', step))
        return envelope('action', episode, step, raw_action=[0.] * 14)
    def clear_actions(self, episode, step, reason):
        self.calls.append(('clear', step))
        return envelope('cleared', episode, step, history_preserved=True, next_step=step)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.client, self.dispatched, self.events = Client(), [], []
        self.scope = Scope('put_object_cabinet', 'put_object_cabinet',
            context_fingerprint({'test': 'synthetic_not_physical'}),
            'fake-robot', 'fake-controller', 'fake-wam', 'fake-binding', 'development', 'fake-run')
        self.target = CapabilityCandidate('put_object_cabinet', 'fake-visible-track')
        self.terminal, self.predicates = False, {'inside': None, 'released': None, 'stable': None}
        self.admitted_ground, self.admitted_verify = True, True
        self.sensor_calls, self.recoveries = [], []
        self.patches = [patch('harness.robotwin_full_middle_bridge.robot_feedback', return_value={'arms': {}}),
                        patch('harness.robotwin_full_middle_bridge.hold_measured', return_value={'measured': True})]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        self.build()

    def ground(self, frame):
        return GroundingResult('fake-grounder', 'test', self.scope.key(), frame.episode_id,
            frame.observation.observation_id, frame.observation.observed_at, 'execute', (self.target,))

    def verify(self, frame):
        outcome = Outcome(self.predicates.copy(), self.terminal, 'fake-sensor-test',
                          'perception_predicates', frame.observation.observation_id)
        return VisibleResult('fake-verifier', 'test', self.scope.key(), frame.episode_id,
            frame.observation.observation_id, frame.observation.observed_at, outcome,
            recovery='replan', recovery_permitted=True, candidate=self.target)

    def build(self):
        contract = ModelContract(self.target.capability, self.scope.wam_version,
                                 self.scope.robot_version, self.scope.controller_version)
        self.supervisor = TaskSupervisor(contract=contract, scope=self.scope,
            required_predicates=tuple(self.predicates), max_age_s=2., ask_timeout_s=2.,
            grounder=ProviderBinding('fake-grounder', 'test', self.scope.key(), self.ground,
                                    trusted=True, admitted=self.admitted_ground),
            verifier=ProviderBinding('fake-verifier', 'test', self.scope.key(), self.verify,
                                     trusted=True, admitted=self.admitted_verify, independent=True),
            max_questions=0, max_recoveries=1, clock=lambda: self.now,
            request_replan=lambda request: self.recoveries.append(request) or True)
        adapter = RobotwinSystemAdapter(self.client,
            lambda action: self.dispatched.append(action) or True, max_age_s=2., clock=lambda: self.now)
        self.bridge = FullMiddleBridge(adapter, self.supervisor, robot=lambda: object(),
            visible_tracks=lambda sensors: self.sensor_calls.append(set(sensors)) or (self.target.object_id,),
            event=self.events.append, clock=lambda: self.now)
        self.bridge.reset('fake-episode', 1, self.target)

    def prepare(self, **extra):
        obs = {'rgb': {v: np.zeros((8, 8, 3), np.uint8) for v in VIEWS},
               'commanded_joint_gripper_state': [0.] * 14, **extra}
        return self.bridge.prepare('fake-episode', 0, 'frame-0', 10., obs)

    def test_unknown_completion_executes_without_inventing_success(self):
        self.prepare()
        self.bridge.execute('fake-episode', 0, 'frame-0')
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.sensor_calls, [{'rgb', 'proprioception'}])
        self.assertIsNone(self.events[0]['verified_outcome'])
        with self.assertRaises(ValueError): self.bridge.execute('fake-episode', 0, 'frame-0')

    def test_unadmitted_grounder_stops_before_wam(self):
        self.admitted_ground = False; self.build()
        with self.assertRaises(SupervisorStop): self.prepare()
        self.assertFalse(self.dispatched)
        self.assertFalse(any(kind == 'action' for kind, _ in self.client.calls))

    def test_unadmitted_verifier_cannot_write_success(self):
        self.admitted_verify = False; self.build()
        self.terminal = True; self.predicates = dict.fromkeys(self.predicates, True)
        self.prepare()
        self.assertEqual(self.supervisor.decision().mode, 'EXECUTE')
        self.assertIsNone(self.events[0]['verified_outcome'])

    def test_complete_verified_success_stops_before_wam(self):
        self.terminal = True; self.predicates = dict.fromkeys(self.predicates, True)
        with self.assertRaises(SupervisorStop) as stop: self.prepare()
        self.assertEqual(stop.exception.directive.mode, 'DONE')
        self.assertFalse(self.dispatched)
        self.assertFalse(any(kind == 'action' for kind, _ in self.client.calls))

    def test_verified_failure_requires_bound_recovery_and_new_observation(self):
        self.terminal = True; self.predicates['inside'] = False
        self.predicates['released'] = self.predicates['stable'] = True
        with self.assertRaises(SupervisorStop): self.prepare()
        self.assertFalse(self.dispatched)
        self.bridge.request_recovery('replan')
        self.bridge.request_recovery('replan')
        self.assertEqual(len(self.recoveries), 1)
        self.assertEqual(self.supervisor.decision().mode, 'HOLD')

    def test_expiry_after_prepare_revokes_and_prevents_resume(self):
        self.prepare(); self.now = 13.
        with self.assertRaises(SupervisorStop): self.bridge.execute('fake-episode', 0, 'frame-0')
        self.assertFalse(self.dispatched)
        self.assertIsNone(self.bridge.adapter.pending)
        self.assertTrue(self.bridge.adapter.failed)

    def test_gt_ingress_rejected_before_any_provider(self):
        with self.assertRaises(ValueError): self.prepare(object_GT={'pose': [0, 0, 0]})
        self.assertFalse(self.sensor_calls)
        self.assertFalse(self.client.calls)

    def test_ask_is_disabled(self):
        self.supervisor.max_questions = 1
        with self.assertRaises(ValueError):
            FullMiddleBridge(self.bridge.adapter, self.supervisor, robot=lambda: object(),
                             visible_tracks=lambda s: ())


class VTTContractTests(unittest.TestCase):
    def test_real_trained_receipt_contract_and_mismatch(self):
        artifact = ROOT / 'outputs/training/cabinet_vtt_adapter_1000_20261006/adapter_step_001000.pt'
        if not artifact.exists(): self.skipTest('local trained artifact unavailable')
        protocol = json.loads((artifact.parent / 'protocol.json').read_text())
        base, adapter = protocol['base_checkpoint_sha256'], sha256(artifact)
        version = 'cabinet-vtt:' + base + ':' + adapter
        scope = Scope('put_object_cabinet', 'put_object_cabinet', context_fingerprint({'test': 'contract_only'}),
                      'aloha', 'controller', version, 'test', 'development', 'test')
        contract = TrainedVTTContract('put_object_cabinet', version, 'aloha', 'controller',
            adapter_path=str(artifact), base_checkpoint_sha256=base, adapter_sha256=adapter)
        contract.validate(scope)
        with self.assertRaises(ValueError): replace(contract, adapter_sha256='wrong').validate(scope)
        with self.assertRaises(ValueError): replace(contract, action_dim=23).validate(scope)


if __name__ == '__main__': unittest.main()
