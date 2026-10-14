#!/usr/bin/env python3
"""CPU safety regressions; no model, simulator or GPU."""
from pathlib import Path
import sys, unittest
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from harness import zhuhai_official_braking as b
from harness import zhuhai_action_smoothing as old
from harness import zhuhai_robot_spec as spec


class BrakingTests(unittest.TestCase):
    def setUp(self):
        self.settings = b.protocol(); self.home = np.asarray(spec.HOME_ACTION, dtype=np.float64)
        self.limiter = b.BrakingRateLimiter(self.settings['rate_limit']); self.limiter.reset(self.home)

    def verify_step(self, limiter, target):
        prev, velocity, acceleration = limiter.previous.copy(), limiter.velocity.copy(), limiter.acceleration.copy()
        actual, stats = limiter.step(target)
        spec.validate_action(actual)
        v = (actual[old.ARM]-prev)/.1; a = (v-velocity)/.1
        self.assertTrue(np.all(np.abs(v) <= limiter.vmax+1e-9))
        self.assertTrue(np.all(np.abs(a) <= limiter.amax+1e-8))
        self.assertTrue(np.all(np.abs((a-acceleration)/.1) <= limiter.jmax+1e-7))
        self.assertGreaterEqual(stats['minimum_braking_surplus_rad'], -1e-10)
        np.testing.assert_array_equal(actual[old.GRIPPER], np.clip(target[old.GRIPPER], 0, 1))
        lo, hi = limiter.feasible_interval()
        self.assertTrue(np.all(lo <= hi+1e-10))
        return actual, stats

    def test_official_limits_are_per_axis_and_angle_unchanged(self):
        np.testing.assert_array_equal(self.limiter.vmax, spec.VELOCITY_LIMITS*2)
        np.testing.assert_array_equal(self.limiter.amax, b.PANDA_ACCELERATION*2)
        np.testing.assert_array_equal(self.limiter.lower, np.asarray(spec.DUAL_JOINT_LIMITS)[:, 0])
        np.testing.assert_array_equal(self.limiter.upper, np.asarray(spec.DUAL_JOINT_LIMITS)[:, 1])

    def test_brakes_at_both_bounds_under_persistent_outward_requests(self):
        for direction in (1, -1):
            self.limiter.reset(self.home)
            events = 0
            for t in range(160):
                target = self.home.copy(); target[old.ARM] = direction*1000
                target[7], target[15] = float(t%2), float((t+1)%2)
                _, stats = self.verify_step(self.limiter, target)
                events += stats['braking_limited_dimensions']
            self.assertGreater(events, 0)
            np.testing.assert_allclose(self.limiter.velocity, 0., atol=1e-9)

    def test_discrete_braking_tail_matches_explicit_deceleration(self):
        for fraction in (.01, .21, .73, 1.):
            v = self.limiter.vmax*fraction; remaining = v.copy(); distance = np.zeros(14)
            for _ in range(20):
                remaining = np.maximum(remaining-self.limiter.amax*.1, 0)
                distance += remaining*.1
            np.testing.assert_allclose(self.limiter.future_stop_distance(v), distance, atol=1e-14)

    def test_random_reversals_always_leave_a_feasible_stop(self):
        rng = np.random.default_rng(20261010)
        for t in range(2000):
            target = rng.uniform(-8, 8, 16); target[[7,15]] = rng.uniform(-.1, 1.1, 2)
            self.verify_step(self.limiter, target)
        clone = b.BrakingRateLimiter(self.settings['rate_limit'])
        clone.previous = self.limiter.previous.copy(); clone.velocity = self.limiter.velocity.copy()
        clone.acceleration = self.limiter.acceleration.copy()
        for _ in range(10):
            target = self.home.copy(); target[old.ARM] = clone.previous
            self.verify_step(clone, target)
        np.testing.assert_allclose(clone.velocity, 0., atol=1e-9)

    def test_invalid_or_already_unstoppable_state_is_rejected(self):
        with self.assertRaises(ValueError): self.limiter.step(np.full(16, np.nan))
        self.limiter.previous[1] = self.limiter.upper[1]-.001
        self.limiter.velocity[1] = self.limiter.vmax[1]
        with self.assertRaisesRegex(ValueError, b.SIGNATURE): self.limiter.step(self.home)

    def test_explicit_eval_overlay_keeps_checkpoint_contract_untouched(self):
        from scripts import eval_zhuhai_precision_official_braking as evaluator
        original = dict(action_postprocessing=old.protocol(), action_postprocessing_sha256='original',
                        source_files={'immutable': 'x'}, model={'weights': 'same'})
        changed = evaluator.overlay(original)
        self.assertEqual(original['action_postprocessing'], old.protocol())
        self.assertEqual(original['action_postprocessing_sha256'], 'original')
        self.assertEqual(changed['source_files'], original['source_files'])
        self.assertEqual(changed['model'], original['model'])
        self.assertEqual(changed['action_postprocessing']['rate_limit']['braking'], b.VERSION)


if __name__ == '__main__': unittest.main(verbosity=2)
