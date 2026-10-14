"""CPU-only startup checks; no weights, scene, labels or successful operations."""
import ast
import copy
from pathlib import Path
import unittest

from harness.completion_readiness import completion_readiness, require_completion_readiness

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(postconditions='task090_observed', manip_executor='wam', candidate_outcome_policy='hold')


class Readiness(unittest.TestCase):
    def test_default_known_unadmitted_gate_is_blocked(self):
        result = completion_readiness(90, CFG)
        self.assertFalse(result['launch_allowed'])
        self.assertTrue(result['cannot_complete_by_design'])
        self.assertIn('missing_relation_calibration', result['blockers'])

    def test_explicit_diagnostic_is_allowed_but_not_admitted(self):
        result = require_completion_readiness(90, {**CFG, 'run_purpose': 'diagnostic_only'})
        self.assertTrue(result['launch_allowed'])
        self.assertFalse(result['gate_admitted'])
        self.assertFalse(result['proves_task_capability'])
        self.assertFalse(result['proves_current_preconditions'])

    def test_config_admitted_flag_cannot_override_real_gate(self):
        with self.assertRaisesRegex(ValueError, 'unadmitted completion gate'):
            require_completion_readiness(90, {**CFG, 'admitted': True, 'verified': True})

    def test_invalid_or_conflicting_purpose_rejected(self):
        for config, purpose in ((CFG, 'test'), ({**CFG, 'run_purpose': 'diagnostic_only'}, 'capability_validation')):
            with self.subTest(purpose=purpose), self.assertRaises(ValueError):
                completion_readiness(90, config, purpose=purpose)

    def test_wrong_task_cannot_reuse_task90_gate(self):
        with self.assertRaises(ValueError):
            completion_readiness(1, CFG, purpose='diagnostic_only')

    def test_unchecked_legacy_does_not_claim_admission(self):
        result = completion_readiness(1, {'postconditions': 'task1_physical'})
        self.assertEqual(result['status'], 'not_checked')
        self.assertIsNone(result['gate_admitted'])
        self.assertIsNone(result['cannot_complete_by_design'])

    def test_empty_task90_orchestrator_cannot_bypass_mode_validation(self):
        with self.assertRaises(ValueError):
            completion_readiness(90, {}, purpose='diagnostic_only')
        self.assertEqual(completion_readiness(90, None)['status'], 'not_checked')

    def test_input_is_unchanged(self):
        config = copy.deepcopy(CFG)
        completion_readiness(90, config)
        self.assertEqual(config, CFG)

    def test_policy_gate_runs_before_device_and_model_loading(self):
        tree = ast.parse((ROOT / 'harness/policy_server.py').read_text())
        policy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WamPolicy')
        init = next(n for n in policy.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
        check = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'require_completion_readiness')
        device = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'device')
        self.assertLess(check.lineno, device.lineno)


if __name__ == '__main__':
    unittest.main()
