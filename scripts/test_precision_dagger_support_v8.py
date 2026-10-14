"""Real USD/FCL regression of the archived support collision, CPU only."""
from copy import deepcopy
from pathlib import Path
import json
import unittest
import numpy as np
from pxr import Usd, UsdGeom
from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
from harness.isaac_zhuhai_expert import FrankaFK
from harness.isaac_zhuhai_collision import PandaSelfCollision
from harness.isaac_zhuhai_support_collision_v8 import PandaSupportCollision
from harness.isaac_zhuhai_segmented_expert_v8 import SegmentedExpert, ExpertPathBlocked
from harness.zhuhai_robot_spec import HOME_ACTION
from harness.zhuhai_official_braking import BrakingRateLimiter, protocol


def scene():
    env = IsaacZhuhaiEnv.__new__(IsaacZhuhaiEnv)
    env.stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(env.stage, 1.)
    env._build_fixed_frame()
    env.kinematics = FrankaFK()
    return env


def recorded_commands():
    p = Path('outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger50_segmented_20261010_v6/episodes/seed_1014003/actions.jsonl')
    return [json.loads(line) for line in p.read_text().splitlines()]


class Tests(unittest.TestCase):
    def test_real_recorded_fault_is_rejected_and_home_remains_reachable(self):
        env = scene()
        old = PandaSelfCollision(env.kinematics)
        new = PandaSupportCollision(env.kinematics, env.stage, env.support_contract['static_collision_paths'])
        for arm, sl in [('left', slice(0, 7)), ('right', slice(8, 15))]:
            self.assertTrue(new.collision_free(arm, np.asarray(HOME_ACTION)[sl]))
        fault = np.asarray(recorded_commands()[-1]['dispatched_action'])[:7]
        self.assertTrue(old.collision_free('left', fault))
        self.assertFalse(new.collision_free('left', fault))
        self.assertEqual(new.last_pair[:2], ('left/panda_link6', '/World/FixedSupport/mast'))
        self.assertEqual(len(new.geometry), 16)

    def test_real_recorded_pregrasp_path_is_blocked_before_its_end(self):
        env = scene()
        checker = PandaSupportCollision(env.kinematics, env.stage, env.support_contract['static_collision_paths'])
        rows = recorded_commands()
        start = np.asarray(rows[1553]['dispatched_action'])[:7]
        end = np.asarray(rows[1562]['dispatched_action'])[:7]
        self.assertTrue(checker.collision_free('left', start))
        self.assertFalse(checker.path_free('left', start, end))

    def test_dispatch_rejection_sends_no_action_and_keeps_inherited_braking_state(self):
        env = scene()
        state = np.asarray(recorded_commands()[-1]['dispatched_action'])
        env.manifest = dict(objects={})
        env.actual_state = lambda: state.copy()
        env.get_action = lambda: state.copy()
        dispatched = []
        env.step_action = lambda action, interval: dispatched.append(action)
        limiter = BrakingRateLimiter(protocol()['rate_limit'])
        limiter.reset(state)
        before = deepcopy(limiter.__dict__)
        expert = SegmentedExpert(env, limiter)
        with self.assertRaises(ExpertPathBlocked):
            expert._drive(state)
        self.assertEqual(dispatched, [])
        for name in ('previous', 'velocity', 'acceleration'):
            np.testing.assert_array_equal(limiter.__dict__[name], before[name])

    def test_missing_or_changed_support_geometry_cannot_silently_pass(self):
        env = scene()
        paths = env.support_contract['static_collision_paths']
        a = PandaSupportCollision(env.kinematics, env.stage, paths)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            PandaSupportCollision(env.kinematics, env.stage, [])
        prim = env.stage.GetPrimAtPath('/World/FixedSupport/mast')
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set((.1, -.68, .95))
        b = PandaSupportCollision(env.kinematics, env.stage, paths)
        self.assertNotEqual(a.geometry_sha256, b.geometry_sha256)


if __name__ == '__main__':
    unittest.main()
