#!/usr/bin/env python3
"""CPU-only privbase isolation/default-path/binding/handoff checks; no simulation or weights."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["P2_TASK_ID"] = "90"
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.docking_controller import BASE_SCALE, DT, load_calibration, rot, task_dock_config
from harness.privileged_base_navigation import (ASSET, GOAL, KEY, OPTION, WRAPPER, PrivbaseNavigator,
                                                validate_config, validate_launch)
from harness.policy_server import WamPolicy, PolicyModel, HEAD_CAMERA
from harness.task1_perception_adapter import TaskPerception


def config():
    return {**json.loads((ROOT / "configs/task090_orchestrate.json").read_text()), OPTION: WRAPPER}


def contract(key="move to(trash_can)", objects=("trash_can",), navigation=True):
    return NS(key=key, step=NS(objects=objects), is_navigation=navigation, started_step=0)


def packet(step=0, base=(0., 0., 0.), target=(2., -.2, .34), epoch=1):
    x, y, yaw = base
    return np.r_[1, 90, epoch, step, 100+step*DT, x, y, 0, 0, 0, np.sin(yaw/2), np.cos(yaw/2),
                 target, 0, 0, 0, 1, 1, 1, 1].astype(np.float64)


def observation(state=None):
    return {"robot_r1::proprio": np.zeros(61, np.float32) if state is None else state.copy(),
            "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros((32, 32, 3), np.uint8),
            "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": np.ones((32, 32), np.float32),
            "robot_r1::cam_rel_poses": np.tile([0, 0, 1, 0, 0, 0, 1], 3)}


def navigator():
    return PrivbaseNavigator(load_calibration(), task_dock_config())


class PrivbaseTests(unittest.TestCase):
    def test_opt_in_and_default(self):
        self.assertFalse(validate_config({}, 90))
        self.assertFalse(validate_launch({}, None, "normal", "dev", 90))
        self.assertTrue(validate_launch(config(), WRAPPER, "s015000_privbase_seed0", "dev", 90))
        for cfg, wrapper, name, profile, tid in [({}, WRAPPER, "privbase", "dev", 90),
                (config(), None, "privbase", "dev", 90), (config(), WRAPPER, "normal", "dev", 90),
                (config(), WRAPPER, "privbase", "report", 90), (config(), WRAPPER, "privbase", "dev", 1),
                ({**config(), "manip_executor": "primitive"}, WRAPPER, "privbase", "dev", 90)]:
            with self.assertRaises(ValueError): validate_launch(cfg, wrapper, name, profile, tid)

    def test_default_allowlist_unchanged_and_enabled_strips_gt(self):
        obs = observation()
        raw = {**obs, KEY: packet(), "reward": 1, "object_poses": {"gt": 1}, "task_id": 90}
        p = WamPolicy.__new__(WamPolicy); p.step_in_episode = 0; p.orch = NS(contract=None)
        p.privbase = None
        clean = p._split_navigation_observation(raw)
        self.assertEqual(clean.keys(), obs.keys())
        self.assertTrue(all(clean[k] is obs[k] for k in obs))
        p.privbase = navigator()
        clean2 = p._split_navigation_observation(raw)
        self.assertEqual(clean.keys(), clean2.keys())
        self.assertTrue(all(clean[k] is clean2[k] for k in clean))
        self.assertNotIn(KEY, clean2)
        self.assertIn(KEY, raw)  # input must not be mutated

    def test_missing_bad_stale_and_wrong_goal_fail_closed(self):
        for bad in (None, np.zeros(22), np.full(22, np.nan), packet(1)):
            n = navigator()
            with self.assertRaises((ValueError, TypeError)): n.prepare(bad, 0)
            self.assertTrue(n.blocked)
        n = navigator(); n.prepare(packet(), 0); n.start(contract(), 0)
        n.controller.last_status = {"docked": True}
        with self.assertRaises(ValueError): n.prepare(packet(), 1)
        self.assertFalse(n.controller.last_status.get("docked", False))
        for c, index in [(contract(objects=("other_bin",)), 0), (contract(), 2),
                         (contract(key="move to(furniture_sink)", objects=("furniture_sink",)), 0)]:
            n = navigator(); n.prepare(packet(), 0)
            with self.assertRaises(ValueError): n.start(c, index)
        p = WamPolicy.__new__(WamPolicy); p.step_in_episode = 0; p.orch = NS(contract=None); p.privbase = navigator()
        with self.assertRaises(ValueError):
            p._split_navigation_observation({**observation(), KEY.replace(GOAL, "ashcan.n.01_2"): packet()})

    def test_reset_and_handoff_remove_gt(self):
        n = navigator(); n.prepare(packet(), 0); n.finish_navigation()
        self.assertEqual(n.omap.tracks, []); self.assertIsNone(n.offset)
        p = WamPolicy.__new__(WamPolicy); p.step_in_episode = 10
        p.orch = NS(contract=contract("pick up from(trash_can,floors)", navigation=False)); p.privbase = n
        clean = p._split_navigation_observation({**observation(), KEY: "malformed GT must be ignored in manipulation"})
        self.assertNotIn(KEY, clean)
        n.reset()
        with self.assertRaises(ValueError): n.prepare(packet(epoch=1), 0)
        n.reset(); n.prepare(packet(epoch=2), 0)
        self.assertFalse(n.blocked)

    def test_static_rim_not_center_and_no_live_gt_upper_body(self):
        state = np.zeros(61)
        a, b = navigator(), navigator()
        a.prepare(packet(), 0); b.prepare(packet(target=(7., 1., .34)), 0)
        a.start(contract(), 0); b.start(contract(), 0)
        self.assertGreater(np.linalg.norm(a.offset[:2]), .1)
        np.testing.assert_array_equal(a.offset, b.offset)
        perception = TaskPerception(fake=lambda *_: [], fake_obstacles=lambda *_: np.zeros((0, 2)))
        aa, _ = a.action(state, observation(state), perception, contract(), 0)
        ab, _ = b.action(state, observation(state), perception, contract(), 0)
        np.testing.assert_array_equal(aa[3:], ab[3:])
        previous = a.offset.copy(); a.prepare(packet(1, base=(1., .8, .1)), 1)
        np.testing.assert_array_equal(a.offset, previous)

    def test_depth_required_and_backoff_shielded(self):
        n = navigator(); n.prepare(packet(), 0); n.start(contract(), 0)
        perception = TaskPerception(fake=lambda *_: [], fake_obstacles=lambda *_: np.zeros((0, 2)))
        n.controller._backoff = 2
        with patch.object(n.controller, "_safe", return_value=(np.zeros(3), {"test": "stopped"})) as shield:
            a, st = n.action(np.zeros(61), observation(), perception, contract(), 0)
            self.assertTrue(shield.called); np.testing.assert_array_equal(a[:3], 0)
            self.assertEqual(st["privbase_recovery_safety"], {"test": "stopped"})
        obs = observation(); obs.pop("robot_r1::robot_r1:zed_link:Camera:0::depth_linear")
        with self.assertRaises(ValueError): n.action(np.zeros(61), obs, perception, contract(), 0)

    def test_physical_velocity_rollout_reaches_dock(self):
        n = navigator(); c = contract(); base = np.zeros(3); state = np.zeros(61)
        perception = TaskPerception(fake=lambda *_: [], fake_obstacles=lambda *_: np.zeros((0, 2)))
        for step in range(1000):
            n.prepare(packet(step, base=base), step)
            if step == 0: n.start(c, 0)
            a, st = n.action(state, observation(state), perception, c, 0)
            state[:3] = a[:3] * BASE_SCALE
            state[53:57], state[3:10], state[28:35] = a[3:7], a[7:14], a[15:22]
            base[:2] += rot(base[2] + .5*state[2]*DT) @ state[:2] * DT
            base[2] += state[2]*DT
            if st["docked"]: break
        self.assertTrue(st["docked"], st)
        self.assertGreater(step, 30)
        self.assertTrue(n.evidence(c, 0)["docked"])

    def test_real_policy_handoff_and_wam_inputs(self):
        # Real orchestrator and policy execution paths; only expensive model heads replaced by CPU stubs.
        from omegaconf import OmegaConf
        import torch
        cfg = OmegaConf.create({"dataset": {"image_size": [32, 32], "cond_mode": "skill_proto"},
                                "model": {"vision_encoder": {"checkpoint_path": "cpu-stub"}},
                                "common": {"proprio_len": 1}})
        wrapper = NS(views=[HEAD_CAMERA], vision_encoder=None, action_repr_name="absolute", proprio_mask_name="positions_only")
        def loader(name, checkpoint, device, cache):
            return PolicyModel(name, checkpoint, 0, cfg, "skill_proto", wrapper)
        with tempfile.TemporaryDirectory() as tmp:
            p = WamPolicy("cpu-stub", device="cpu", log_dir=tmp, orchestrate=config(), model_loader=loader,
                          skill="move to", exec_horizon=8, num_steps=10, cuda_fraction=.3, seed=0)
            feats = [torch.zeros(1, 8, 1024)] * 3
            p._encode = lambda pixels: (feats, np.zeros(1024, np.float32))
            p._last_patch = torch.zeros(8, 1024)
            seen = []
            def sample(feats, state, cond, **kwargs):
                seen.append((np.asarray(state).copy(), cond.clone()))
                return np.zeros((32, 23), np.float32)
            p._sample = sample
            p.perception = TaskPerception(fake=lambda obs, state: self._sensor_only(obs),
                                          fake_obstacles=lambda obs, state: np.zeros((0, 2)))
            base = np.zeros(3); state = np.zeros(61, np.float32)
            for step in range(1000):
                obs = observation(state); obs[KEY] = packet(step, base=base)
                a = p.act(obs).numpy()
                if seen: break
                state[:3] = a[:3]*BASE_SCALE
                state[53:57], state[3:10], state[28:35] = a[3:7], a[7:14], a[15:22]
                base[:2] += rot(base[2]+.5*state[2]*DT) @ state[:2]*DT; base[2] += state[2]*DT
            self.assertTrue(seen, "real policy never sampled the WAM after docking")
            self.assertFalse(p.orch.contract.is_navigation)
            self.assertTrue(p.privbase.finished)
            self.assertEqual(p.omap.tracks, [])  # privileged tracks never entered sensor map
            self.assertEqual(seen[0][0].shape, (61,))
            self.assertEqual(p.stats["chunks"], 1)
            # Manipulation must run without GT and use only the original observable predicate producer.
            for _ in range(2): p.act(observation(state))
            self.assertEqual(p.postcondition_mode, "task090_observed")
            self.assertFalse(p._last_post_t1.get("satisfied", False))
            p.trace.close()

    def _sensor_only(self, obs):
        self.assertNotIn(KEY, obs)
        self.assertFalse(any(k.startswith("privbase") for k in obs))
        return []


if __name__ == "__main__":
    unittest.main(verbosity=2)
