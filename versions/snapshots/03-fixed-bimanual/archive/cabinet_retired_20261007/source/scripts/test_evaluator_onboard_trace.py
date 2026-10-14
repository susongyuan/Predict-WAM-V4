"""CPU tests for read-only, step-aligned executed-action evidence."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Poison:
    def __iter__(self):
        raise AssertionError('forbidden value was read')


class OnboardTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / 'third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/eval/evaluator.py'
        tree = ast.parse(source.read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_onboard_trace_evidence')
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
        cls.read = staticmethod(ns[fn.name])

    def make(self, step=9):
        return SimpleNamespace(_pose_log_step=step, robot_action=[0.1]*23,
            robot=SimpleNamespace(name='robot_r1', action_dim=23),
            obs={'robot_r1::proprio': [0.2]*61, 'robot_r1::cam_rel_poses': [0.3]*21,
                 'privbase_nav::ashcan.n.01_1::ifzxzj': Poison(), 'other::proprio': Poison()})

    def test_action_precedes_observation_by_one(self):
        row = self.read(self.make())
        self.assertEqual((row['action_step'], row['observation_step']), (8, 9))
        self.assertEqual(len(row['action']), 23)

    def test_reset_does_not_read_stale_action(self):
        fake = self.make(0)
        fake.robot_action = Poison()
        row = self.read(fake)
        self.assertEqual(row['status'], 'available')
        self.assertIsNone(row['action'])
        self.assertIsNone(row['action_step'])

    def test_only_onboard_vectors_not_nav_truth(self):
        row = self.read(self.make())
        self.assertEqual(set(row['onboard_vectors']), {'robot_r1::proprio', 'robot_r1::cam_rel_poses'})

    def test_no_mutation(self):
        fake = self.make()
        before = copy.deepcopy((fake.robot_action, fake.obs['robot_r1::proprio']))
        row = self.read(fake)
        row['action'][0] = 999
        self.assertEqual(before, (fake.robot_action, fake.obs['robot_r1::proprio']))

    def test_nonfinite_is_unavailable(self):
        fake = self.make()
        fake.robot_action[0] = float('nan')
        self.assertEqual(self.read(fake)['status'], 'unavailable')

    def test_wrong_action_dimension_is_unavailable(self):
        fake = self.make()
        fake.robot_action = [1.0]
        self.assertEqual(self.read(fake)['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
