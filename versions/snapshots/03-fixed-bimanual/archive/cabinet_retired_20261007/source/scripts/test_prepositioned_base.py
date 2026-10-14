"""CPU geometry and protocol checks; not physical prepositioning success."""
import ast
import json
import math
from pathlib import Path
import unittest

import numpy as np
from harness.docking_controller import rot
from harness.privileged_base_navigation import ASSET, PREPOSITION_WRAPPER, WRAPPER, pose_matrix, validate_launch

ROOT = Path(__file__).resolve().parents[1]


class PrepositionGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / 'harness/prepositioned_base_rgbd_wrapper.py').read_text()
        tree = ast.parse(cls.source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'candidate_stances')
        ns = dict(np=np, math=math, pose_matrix=pose_matrix, rot=rot)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<geometry>', 'exec'), ns)
        cls.candidates = staticmethod(ns['candidate_stances'])
        box = json.loads(ASSET.read_text())['link_bounding_boxes']['base_link']['visual']['axis_aligned']
        cls.center = np.array(box['transform'])[:3, 3]
        cls.extent = np.array(box['extent'])

    def values(self, obstacles=()):
        return self.candidates([0, -3, .005, 0, 0, 0, 1], [0, 0, .333, 0, 0, 0, 1],
                               np.ones(3), self.center, self.extent, np.array([.602, -.193]), obstacles)

    def test_24_candidate_limit_and_exact_relative_target(self):
        candidates = self.values()
        self.assertEqual(len(candidates), 24)
        clear = [x for x in candidates if not x['blocked_by']]
        self.assertTrue(clear)
        for row in clear:
            rim = np.array([0, 0, .333]) + row['rim_local']
            np.testing.assert_allclose(rot(-row['yaw']) @ (rim[:2] - row['xy']), [.602, -.193], atol=1e-10)

    def test_obstacle_rejects_all_not_move_object(self):
        rows = self.values([('solid', [-10, -10, 0], [10, 10, 1])])
        self.assertTrue(all('solid' in row['blocked_by'] for row in rows))

    def test_translation_invariance(self):
        a = self.values()
        b = self.candidates([7, -5, .005, 0, 0, 0, 1], [7, -2, .333, 0, 0, 0, 1],
                            np.ones(3), self.center, self.extent, np.array([.602, -.193]), [])
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(y['xy']) - x['xy'], [7, -2], atol=1e-10)
            self.assertEqual(x['blocked_by'], y['blocked_by'])

    def test_preposition_requires_matching_opt_in_and_label(self):
        cfg = dict(navigation='docking', postconditions='task090_observed', privileged_base_navigation=PREPOSITION_WRAPPER)
        self.assertTrue(validate_launch(cfg, PREPOSITION_WRAPPER, 'privbase_preposition_test', 'development', 90))
        for wrapper, name in ((WRAPPER, 'privbase_preposition_test'), (PREPOSITION_WRAPPER, 'privbase_only')):
            with self.assertRaises(ValueError):
                validate_launch(cfg, wrapper, name, 'development', 90)

    def test_already_at_dock_never_leaves_for_new_waypoint(self):
        from harness.docking_controller import DockingController, DockConfig, ObjectMap, Odometry, Track, load_calibration
        calibration = load_calibration(ROOT / 'outputs/validation/task090_docking_calibration_train_20260930')
        ctl = DockingController(calibration, DockConfig())
        odom, objects = Odometry(), ObjectMap()
        target = calibration['dock']['trash_can']['target']
        objects.tracks = [Track('trash_can', target.copy(), hits=3, last_seen=0, track_id=1, z=.67)]
        ctl.start('trash_can', 'before_bin', odom)
        def forbidden_plan(*args, **kwargs):
            raise AssertionError('must not replan away from an already measured valid dock')
        ctl._plan_pose = forbidden_plan
        action, status = ctl.step(np.zeros(61), objects, odom, obstacles=np.empty((0, 2)))
        np.testing.assert_allclose(action[:3], 0., atol=1e-7)
        self.assertEqual(status['mode'], 'align')
        self.assertFalse(status['docked'])  # Posture/settling still required; not a timeout success.

    def test_warmup_reset_is_not_prepositioned(self):
        tree = ast.parse(self.source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        class Parent:
            def reset(self):
                return {'warmup': True}, {'unchanged': True}
        ns = dict(PrivilegedBaseRGBDWrapper=Parent)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<wrapper>', 'exec'), ns)
        wrapper = ns['PrepositionedBaseRGBDWrapper']()
        self.assertEqual(wrapper.reset(), ({'warmup': True}, {'unchanged': True}))
        self.assertIn('task090_docking_calibration_train_20260930', self.source)
        self.assertIn('load_calibration(path=calibration', self.source)

    def test_no_object_or_joint_position_setters_and_interval_cleared(self):
        tree = ast.parse(self.source)
        setters = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr.startswith('set_')]
        self.assertEqual([n.func.attr for n in setters], ['set_position_orientation', 'set_joint_velocities'])
        self.assertTrue(all(isinstance(n.func.value, ast.Name) and n.func.value.id == 'robot' for n in setters))
        self.assertIn('self._interval_velocity = self._interval_end = None', self.source)
        self.assertIn('target moved during base-only prepositioning', self.source)


if __name__ == '__main__':
    unittest.main()
