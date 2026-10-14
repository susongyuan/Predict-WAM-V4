#!/usr/bin/env python3
"""Native16 numeric-only smoothing regressions; no simulator/model/GPU."""
from copy import deepcopy
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from harness import zhuhai_action_smoothing as smooth
from harness import zhuhai_robot_spec as spec


class Context:
    def __init__(self):self.states=[]; self.pools=[]
    def reset(self):self.states.clear(); self.pools.clear()
    def push(self, state, pool):self.states.append(np.array(state, copy=True)); self.pools.append(pool)
    def inputs(self):return {'state': self.states[-1].copy()}


class Native16SmoothingTests(unittest.TestCase):
    def setUp(self):
        self.settings = smooth.protocol()
        self.home = np.asarray(spec.HOME_ACTION, np.float64)
        self.rgb = {name: np.ones((4, 4, 3), np.uint8) for name in self.settings['data_contract']['camera_names']}

    def chunk(self):
        values = np.tile(self.home, (32, 1))
        t = np.arange(32, dtype=float)
        values[:, smooth.ARM] += (.025*np.sin(t*.7)+.01*np.cos(t*1.7))[:, None]
        values[:, 7] = t < 12
        values[:, 15] = (t >= 3) & (t < 25)
        return values

    def test_spec_dimensions_limits_and_contract(self):
        self.assertEqual((self.settings['state_dim'], self.settings['action_dim']), (16, 16))
        self.assertEqual(self.settings['state_semantics'], 'measured_joint_gripper')
        self.assertEqual(self.settings['preserved_gripper_dimensions'], [7, 15])
        self.assertEqual(self.settings['smoothed_dimensions'], [*range(7), *range(8, 15)])
        self.assertEqual(self.settings['robot_contract'], spec.robot_contract())
        bounds = np.asarray(spec.DUAL_JOINT_LIMITS)
        np.testing.assert_array_equal(self.settings['rate_limit']['position_lower_rad'], bounds[:, 0])
        np.testing.assert_array_equal(self.settings['rate_limit']['position_upper_rad'], bounds[:, 1])

    def test_exact_upstream_spline_on_14_arm_dimensions_only(self):
        from scipy.interpolate import make_lsq_spline
        source = smooth.UPSTREAM.read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'bspline_smooth')
        namespace = dict(np=np, make_lsq_spline=make_lsq_spline)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(smooth.UPSTREAM), 'exec'), namespace)
        raw = self.chunk()
        result = smooth.smooth_chunk(raw, self.settings)
        expected = namespace['bspline_smooth'](raw[:, smooth.ARM], degree=3, num_ctrl_pts=8)
        np.testing.assert_array_equal(result[:, smooth.ARM], expected)
        np.testing.assert_array_equal(result[:, [7, 15]], raw[:, [7, 15]])
        self.assertFalse(np.array_equal(result[:, 6], raw[:, 6]))  # old gripper index is now the seventh left joint
        self.assertFalse(np.array_equal(result[:, 14], raw[:, 14]))
        np.testing.assert_array_equal(raw, self.chunk())

    def test_old14_and_nonfinite_chunks_or_protocols_are_rejected(self):
        for value in (np.zeros((32, 14)), np.full((32, 16), np.nan)):
            with self.assertRaises(ValueError):smooth.smooth_chunk(value, self.settings)
        old = deepcopy(self.settings); old['action_dim'] = 14
        with self.assertRaises(ValueError):smooth.smooth_chunk(self.chunk(), old)

    def test_declared_velocity_acceleration_and_position_bounds(self):
        rate = smooth.PhysicalRateLimiter(self.settings['rate_limit'])
        rate.reset(self.home)
        previous = self.home.copy(); velocity = np.zeros(14)
        for i in range(24):
            raw = self.home.copy(); raw[smooth.ARM] += .25 if i < 12 else -.15
            raw[7], raw[15] = float(i % 2), float((i+1) % 2)
            actual, stats = rate.step(raw)
            v = (actual[smooth.ARM]-previous[smooth.ARM])/.1
            a = (v-velocity)/.1
            self.assertLessEqual(float(np.abs(v).max()), 1.2+1e-10)
            self.assertLessEqual(float(np.abs(a).max()), 4.+1e-10)
            spec.validate_action(actual)
            np.testing.assert_array_equal(actual[[7, 15]], raw[[7, 15]])
            self.assertEqual(stats['action_dim'], 16)
            self.assertEqual(stats['gripper_indices'], [7, 15])
            self.assertEqual(stats['state_semantics'], 'measured_joint_gripper')
            previous, velocity = actual, v

    def test_old_aloha_bounds_or_invalid_initial_target_fail_closed(self):
        settings = deepcopy(self.settings['rate_limit'])
        settings['position_lower_rad'] = [-10.]*14; settings['position_upper_rad'] = [10.]*14
        with self.assertRaises(ValueError):smooth.PhysicalRateLimiter(settings)
        rate = smooth.PhysicalRateLimiter(self.settings['rate_limit'])
        for value in (np.zeros(14), np.zeros(16)):
            with self.assertRaises(ValueError):rate.reset(value)

    def test_policy_history_uses_measured_state_not_drive_target(self):
        context = Context(); calls=[]
        def encode(rgb):self.assertEqual(set(rgb), set(self.rgb)); return None, np.ones((3, 2048), np.float32)
        def sample(features, history, seed, step):
            calls.append((step, history['state'].copy()))
            raw = np.tile(self.home, (32, 1)); raw[:, 0] += .4
            return raw
        policy = smooth.SmoothedChunkPolicy(context, encode, sample, self.settings)
        policy.reset('cpu_native16', 20261009)
        measured = self.home.copy(); measured[0] -= .2; measured[7] = .8
        previous = self.home.copy(); velocity = np.zeros(14)
        for step in range(17):
            obs = dict(rgb=self.rgb, measured_joint_gripper_state=measured.copy(),
                       commanded_joint_gripper_state=previous.copy())
            result = policy.observe('cpu_native16', step, obs)
            actual = np.asarray(result['limited_action'])
            v = (actual[smooth.ARM]-previous[smooth.ARM])/.1
            self.assertLessEqual(float(np.abs((v-velocity)/.1).max()), 4.+1e-10)
            self.assertEqual(result['state_dim'], 16)
            np.testing.assert_array_equal(context.states[-1], measured)
            previous, velocity = actual, v
        self.assertEqual([step for step, state in calls], [0, 8, 16])
        self.assertAlmostEqual(calls[0][1][0], self.home[0]-.2)
        self.assertEqual(len(context.states), 17)
        self.assertNotIn('isaacsim', sys.modules)

    def test_command_only_observation_cannot_be_used_as_model_state(self):
        policy = smooth.SmoothedChunkPolicy(Context(), lambda rgb:(None, None), lambda *args:self.chunk(), self.settings)
        policy.reset('cpu_native16', 1)
        with self.assertRaisesRegex(ValueError, 'measured native16'):
            policy.observe('cpu_native16', 0, dict(rgb=self.rgb, commanded_joint_gripper_state=self.home))
        with self.assertRaises(ValueError):
            policy.observe('cpu_native16', 0, dict(rgb=self.rgb, measured_joint_gripper_state=np.zeros(14),
                commanded_joint_gripper_state=self.home))

    def test_no_extra_camera_and_queue_ids_are_strict(self):
        policy = smooth.SmoothedChunkPolicy(Context(), lambda rgb:(None, None), lambda *args:self.chunk(), self.settings)
        policy.reset('cpu_native16', 1)
        rgb = dict(self.rgb, third_view=np.ones((4, 4, 3), np.uint8))
        with self.assertRaises(ValueError):
            policy.observe('cpu_native16', 0, dict(rgb=rgb, measured_joint_gripper_state=self.home,
                commanded_joint_gripper_state=self.home))
        with self.assertRaises(ValueError):policy.clear_actions('other_episode', 0, 'test')


if __name__ == '__main__':
    unittest.main(verbosity=2)
