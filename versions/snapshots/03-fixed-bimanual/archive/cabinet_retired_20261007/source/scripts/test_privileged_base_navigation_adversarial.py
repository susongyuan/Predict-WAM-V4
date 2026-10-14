#!/usr/bin/env python3
"""Adversarial CPU privbase boundary tests; synthetic transforms, never physical success evidence."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["P2_TASK_ID"] = "90"
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.test_privileged_base_navigation import contract, navigator, observation, packet
from harness.policy_server import WamPolicy
from harness.privileged_base_navigation import KEY, pose_matrix
from harness.docking_controller import rot


class AdversarialTests(unittest.TestCase):
    def test_bad_pose_values_and_geometry_changes_latch_closed(self):
        # Every vector slot is finite-checked; pose quaternions and model scale are additionally constrained.
        for field, value in [(i, float("nan")) for i in range(22)] + [(8, 2.), (15, 2.), (19, 0.), (20, -1.)]:
            with self.subTest(field=field, value=value):
                n = navigator(); bad = packet(); bad[field] = value
                with self.assertRaises(ValueError): n.prepare(bad, 0)
                self.assertTrue(n.blocked)
                with self.assertRaises(ValueError): n.prepare(packet(), 0)
                self.assertEqual(n.omap.tracks, [])
        n = navigator(); n.prepare(packet(), 0)
        bad = packet(1); bad[19] *= 1.1
        with self.assertRaises(ValueError): n.prepare(bad, 1)
        self.assertTrue(n.blocked)

    def test_freshness_cannot_be_faked_with_step_only(self):
        for bad in (packet(1, epoch=2), packet(2)):
            n = navigator(); n.prepare(packet(), 0)
            with self.assertRaises(ValueError): n.prepare(bad, int(bad[3]))
            self.assertTrue(n.blocked)
        n = navigator(); n.prepare(packet(), 0)
        bad = packet(1); bad[4] = packet()[4]
        with self.assertRaises(ValueError): n.prepare(bad, 1)
        self.assertTrue(n.blocked)

    def test_sidechannel_is_not_accepted_under_similar_or_nested_keys(self):
        for key in (KEY + "::rgb", "robot_r1::" + KEY, "robot_r1::object_poses", "nav"):
            p = WamPolicy.__new__(WamPolicy)
            p.privbase = navigator(); p.step_in_episode = 0; p.orch = NS(contract=None)
            with self.subTest(key=key), self.assertRaises(ValueError):
                p._split_navigation_observation({**observation(), key: packet()})
            self.assertTrue(p.privbase.blocked)

    def test_default_ignores_even_unconvertible_privileged_value(self):
        class Poison:
            def __array__(self, *args, **kwargs):
                raise AssertionError("default path inspected privileged data")
        p = WamPolicy.__new__(WamPolicy); p.privbase = None
        obs = observation(); clean = p._split_navigation_observation({**obs, KEY: Poison()})
        self.assertEqual(clean.keys(), obs.keys())
        for key in obs: self.assertIs(clean[key], obs[key])

    def test_act_failure_discards_queued_actions_and_privileged_evidence(self):
        p = WamPolicy.__new__(WamPolicy)
        p.privbase = navigator(); p.privbase.prepare(packet(), 0)
        p.privbase.start(contract(), 0); p.privbase.controller.last_status = {"docked": True}
        p.step_in_episode = 0; p.queue = [np.ones(23)]; p.orch = NS(contract=contract())
        p._act = lambda obs: p._split_navigation_observation(obs)
        with self.assertRaises(ValueError): p.act(observation())
        self.assertEqual(p.queue, [])
        self.assertTrue(p.privbase.blocked)
        self.assertEqual(p.privbase.controller.last_status, {})
        with self.assertRaises(ValueError): p.act({**observation(), KEY: packet()})

    def test_rebinding_and_post_handoff_navigation_are_rejected(self):
        n = navigator(); n.prepare(packet(), 0); n.start(contract(), 0)
        replacement = contract(); replacement.started_step = 1
        with self.assertRaises(ValueError): n.evidence(replacement, 0)
        self.assertTrue(n.blocked)
        n = navigator(); n.prepare(packet(), 0); n.start(contract(), 0); n.finish_navigation()
        with self.assertRaises(ValueError): n.start(contract(), 0)
        self.assertEqual(n.omap.tracks, [])

    def test_common_world_transform_does_not_change_docking_geometry(self):
        a, b = navigator(), navigator()
        original = packet(base=(0., 0., .2))
        transformed = original.copy()
        angle = .7
        translation = np.array([3.2, -1.8, .4])  # synthetic frame change, not a deployment coordinate
        r = rot(angle)
        for start in (5, 12):
            transformed[start:start+2] = r @ original[start:start+2] + translation[:2]
            transformed[start+2] += translation[2]
            yaw = math.atan2(pose_matrix(original[start:start+7])[1, 0],
                             pose_matrix(original[start:start+7])[0, 0]) + angle
            transformed[start+3:start+7] = [0, 0, math.sin(yaw/2), math.cos(yaw/2)]
        a.prepare(original, 0); b.prepare(transformed, 0)
        np.testing.assert_allclose(a.offset, b.offset, atol=1e-12)
        np.testing.assert_allclose(a.odom.to_base(a.omap.tracks[0].pos),
                                   b.odom.to_base(b.omap.tracks[0].pos), atol=1e-12)
        self.assertAlmostEqual(a.omap.tracks[0].z, b.omap.tracks[0].z)

    def test_previous_docked_flag_cannot_override_live_distance(self):
        n = navigator(); n.prepare(packet(), 0); n.start(contract(), 0)
        n.controller.last_status = {"docked": True, "mode": "docked"}
        ev = n.evidence(contract(), 0)
        self.assertGreater(ev["error_norm"], n.controller.cfg.tol_xy)
        self.assertFalse(ev["docked"])
        self.assertEqual(ev["observation_id"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
