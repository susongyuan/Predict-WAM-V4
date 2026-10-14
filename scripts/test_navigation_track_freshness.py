#!/usr/bin/env python3
"""CPU-only regressions for navigation evidence expiry, not object absence."""
import ast
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.docking_controller import ObjectMap, Odometry, DockingController


class FreshnessTests(unittest.TestCase):
    def observed_map(self, age=90):
        m, odom = ObjectMap(max_age_steps=age), Odometry()
        for step in range(51, 56):
            m.update([{'label': 'trash_can', 'xyz_base': [2.467, .028, .3]}], odom, step, visible_fn=None)
        return m, odom

    def test_old_counterexample_and_new_rejection(self):
        m, odom = self.observed_map()
        m.update([], odom, 1020, visible_fn=None)
        old_source = ROOT/'outputs/validation/task090_diagnostic_repairs_20261001/tracking/source_before/docking_controller.py'
        tree = ast.parse(old_source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ObjectMap')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'confirmed')
        scope = {}; exec(compile(ast.Module(body=[fn], type_ignores=[]), str(old_source), 'exec'), scope)
        self.assertEqual(len(scope['confirmed'](m, 'trash_can')), 1)
        self.assertEqual(m.confirmed('trash_can'), [])
        self.assertEqual(len(m.tracks), 1)  # conservative history is not erased as observed absence
        self.assertEqual(m.tracks[0].last_seen, 55)

    def test_boundary_and_detector_gap(self):
        m, _ = self.observed_map()
        m.advance(145)
        self.assertEqual(len(m.confirmed('trash_can')), 1)
        m.advance(146)  # no detector invocation is needed to age the evidence
        self.assertEqual(m.confirmed('trash_can'), [])

    def test_repeated_fresh_static_detections_remain_eligible(self):
        m, odom = self.observed_map()
        for step in (100, 150, 200):
            m.update([{'label': 'trash_can', 'xyz_base': [2.467, .028, .3]}], odom, step, visible_fn=None)
            self.assertEqual(len(m.confirmed('trash_can')), 1)

    def test_reset_retains_limit_and_resets_episode_clock(self):
        m, odom = self.observed_map()
        m.advance(1020); m.reset()
        self.assertEqual(m.max_age_steps, 90)
        self.assertIsNone(m._step)
        self.assertEqual(m.tracks, [])
        m.update([{'label': 'trash_can', 'xyz_base': [1., 0., .3]}], odom, 0)
        self.assertEqual(m.tracks[0].track_id, 1)

    def test_cached_target_and_plan_are_dropped(self):
        m, odom = self.observed_map()
        ctl = DockingController.__new__(DockingController)
        ctl.label, ctl.track_id = 'trash_can', m.tracks[0].track_id
        ctl._plan, ctl._plan_age = {'track': ctl.track_id, 'W': [1, 2]}, 10
        self.assertIsNotNone(ctl._target(m, odom))
        m.advance(1020)
        self.assertIsNone(ctl._target(m, odom))
        self.assertIsNone(ctl.track_id)
        self.assertIsNone(ctl._plan)
        self.assertEqual(ctl._plan_age, 0)

    def test_legacy_default_unchanged(self):
        m, odom = self.observed_map(None)
        m.update([], odom, 1020)
        self.assertEqual(len(m.confirmed('trash_can')), 1)

    def test_consumed_never_navigation_target(self):
        m, _ = self.observed_map()
        m.tracks[0].consumed = True
        self.assertEqual(m.confirmed('trash_can'), [])

    def test_bad_age_and_clock_rejected(self):
        for bad in (0, -1, True, 1.5):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                ObjectMap(max_age_steps=bad)
        m, _ = self.observed_map()
        with self.assertRaises(ValueError): m.advance(54)

    def test_policy_wiring_and_training_sources(self):
        tree = ast.parse((ROOT/'harness/policy_server.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WamPolicy')
        act = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_act_orchestrated')
        calls = [n for n in ast.walk(act) if isinstance(n, ast.Call)]
        self.assertTrue(any(isinstance(n.func, ast.Attribute) and n.func.attr == 'advance'
                            and ast.unparse(n.func.value) == 'self.omap' for n in calls))
        cfg = json.loads((ROOT/'configs/task090_orchestrate.json').read_text())
        self.assertEqual(cfg['track_max_age_steps'], 90)
        manifest = json.loads((ROOT/'outputs/training/task090_v3_50k_20260930_s010000/manifest.json').read_text())
        for name in ('harness/docking_controller.py', 'harness/policy_server.py', 'configs/task090_orchestrate.json'):
            self.assertNotIn(str(ROOT/name), manifest['source_sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
