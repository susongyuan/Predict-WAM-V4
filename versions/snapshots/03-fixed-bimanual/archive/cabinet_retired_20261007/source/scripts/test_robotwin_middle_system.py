"""Controller lifecycle and trained VTT tests; no synthetic task success claims."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.robotwin_middle_system import MiddleSystem, MotionWindow
from harness.robotwin_policy_worker import envelope, VIEWS, InfrastructureError
from harness.robotwin_vtt_runtime import VTTMonitor, adapter_metadata
from harness.robotwin_condition_memory import DemoConditionMemory

ADAPTER = ROOT / 'outputs/training/cabinet_vtt_adapter_1000_20261006/adapter_step_001000.pt'
BANK = ROOT / 'outputs/validation/cabinet_vtt_bank_20261006'

def feedback(value=0):
    return {'arms': {s: {'actual_qpos': [value]*6, 'drive_targets': [value]*6}
                     for s in ('left', 'right')}}

def observation():
    return {'rgb': {v: np.zeros((12,16,3), np.uint8) for v in VIEWS},
            'commanded_joint_gripper_state': np.zeros(14)}

class Client:
    def __init__(self):
        self.clears = self.selections = self.actions = 0
    def reset(self, ep, seed):
        return envelope('reset_ok', ep, 0)
    def action(self, ep, step, obs):
        self.actions += 1
        return envelope('action', ep, step, raw_action=[0.]*14,
                        vtt={'episode': ep, 'observation_step': step, 'progress': 0.,
                             'closure_candidate': False, 'task_success_verified': None})
    def clear_actions(self, ep, step, reason):
        self.clears += 1
        return envelope('cleared', ep, step, history_preserved=True, next_step=step)
    def request(self, message, kind):
        self.selections += 1
        return envelope(kind, message['episode'], message['step'],
                        history_preserved=True, next_step=message['step'],
                        condition_source='TRAIN-demo:test_fixture')

class MiddleTests(unittest.TestCase):
    def test_measured_motion_is_not_grasp(self):
        w = MotionWindow()
        for i in range(41):
            w.push(i, feedback(), 0.)
        self.assertTrue(w.metrics()['controller_stalled'])
        self.assertNotIn('grasp_verified', w.metrics())
        w.push(41, feedback(.1), 0.)
        self.assertFalse(w.metrics()['controller_stalled'])

    def test_plateau_does_not_trigger_while_moving(self):
        w = MotionWindow()
        for i in range(41):
            w.push(i, feedback(i*.005), 0.)
        self.assertFalse(w.metrics()['controller_stalled'])

    def test_gaps_and_bad_feedback_rejected(self):
        w = MotionWindow();w.push(0, feedback())
        with self.assertRaises(ValueError):w.push(2, feedback())
        with self.assertRaises(ValueError):w.push(1, feedback(float('nan')))
        with self.assertRaises(ValueError):w.push(1, {'object_GT':{}, **feedback()})

    def run_controller(self, with_vtt, resumes):
        with tempfile.TemporaryDirectory() as td:
            c = Client()
            system = MiddleSystem(c, lambda a: True, robot=lambda: object(),
                    refresh=lambda: (observation(), time.monotonic()),
                    out=td, episode='episode', with_vtt=with_vtt)
            system.reset('episode', 12)
            try:
                for step in range(201):
                    value = max(0, step-180)*.005 if resumes else 0.
                    with patch('harness.robotwin_middle_system.robot_feedback', return_value=feedback(value)), \
                         patch('harness.robotwin_middle_system.hold_measured', return_value={'test_hold': True}):
                        system.prepare('episode',step,str(step),time.monotonic(),observation())
                        # Pending action prohibits another supervisory mutation.
                        with self.assertRaises(ValueError):
                            system.prepare('episode',step,str(step)+'bad',time.monotonic(),observation())
                        system.execute('episode',step,str(step))
                records = system.memory.recall(system.scope, policy_only=True)
                self.assertTrue(records)
                self.assertTrue(all(set(r['predicates']) == {'controller_motion_resumed'} for r in records))
                self.assertIsNone(system.report['task_success_verified'])
                self.assertEqual(system.report['task_success_memory_records'],0)
                if resumes:
                    self.assertEqual(c.clears,1)
                    self.assertTrue(any(r['outcome']=='recovery' for r in records))
                else:
                    self.assertLessEqual(c.clears,2)
                    self.assertFalse(any(r['outcome']=='recovery' for r in records))
                self.assertEqual(c.selections,c.clears if with_vtt else 0)
            finally:
                system.close()

    def test_vtt_retrieval_only_after_ack_clear(self):self.run_controller(True, True)
    def test_no_vtt_controller_runs_without_condition_switch(self):self.run_controller(False, True)
    def test_failed_recovery_is_not_promoted(self):self.run_controller(True, False)

class VTTTests(unittest.TestCase):
    def test_completed_artifact_receipt_and_non_admission(self):
        protocol = json.loads((ADAPTER.parent/'protocol.json').read_text())
        _, meta = adapter_metadata(ADAPTER, protocol['base_checkpoint_sha256'])
        self.assertFalse(meta['physical_success_verified'])
        with self.assertRaises(InfrastructureError):adapter_metadata(ADAPTER,'wrong_base')

    def test_train_retrieval_distinct_candidates_and_no_fake_wins(self):
        memory=DemoConditionMemory(BANK, ADAPTER)
        vector, source=memory.select(memory.contexts[0])
        self.assertEqual(source,'TRAIN-demo:'+memory.names[0])
        np.testing.assert_array_equal(vector,memory.instances[0])
        _, next_source=memory.select(memory.contexts[0]);self.assertNotEqual(source,next_source)
        memory.reset();_, reset_source=memory.select(memory.contexts[0]);self.assertEqual(source,reset_source)

    def test_vtt_candidate_never_certifies_completion(self):
        m=VTTMonitor(BANK)
        zero=np.zeros(1024,np.float32)
        self.assertFalse(m.observe(zero,0)['closure_candidate'])
        candidate=m.observe(m.prototype.copy(),30)
        self.assertTrue(candidate['closure_candidate'])
        self.assertIsNone(candidate['task_success_verified'])
        m.reset()
        with self.assertRaises(InfrastructureError):m.observe(zero,1)

if __name__=='__main__':unittest.main()
