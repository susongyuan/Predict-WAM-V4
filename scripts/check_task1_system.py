#!/usr/bin/env python3
"""CPU integration check of the Task 1 system path (no simulator, no weights, no GPU): WamPolicy in orchestrator
mode with `navigation: docking` + `postconditions: task1_physical`, stub 3-view WAM, fake perception and a small
kinematic world. It checks wiring, not physics or model quality:

  docking controller drives move-to contracts from base-frame detections and hands over at the calibrated pose;
  WAM chunks run only manipulation contracts; calibrated proprio predicates complete picks/places;
  an empty grasp triggers re-approach, a can released outside the bin triggers rollback to re-pick it,
  a can that keeps failing is skipped (replanner) and the others are still finished.

    P2_TASK_ID=1 .conda-envs/lilawam/bin/python scripts/check_task1_system.py
"""
import hashlib
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("P2_TASK_ID", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from harness import task_profile  # noqa: E402
from harness.docking_controller import BASE_SCALE, DT, rot, wrap  # noqa: E402
from harness.policy_server import PolicyModel, WamPolicy  # noqa: E402
from harness.task1_perception_adapter import TaskPerception  # noqa: E402
from harness.wam_behavior import HEAD_CAMERA, LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA  # noqa: E402

assert task_profile.TASK_ID == 1, "run with P2_TASK_ID=1"
TARGETS = "data/skill_vtt_task001/contract_targets.json"
TARGETS_SHA = hashlib.sha256((ROOT / TARGETS).read_bytes()).hexdigest()
CONTRACTS = json.loads((ROOT / TARGETS).read_text())["contracts"]
KEYS = [k for k in CONTRACTS if not k.startswith("move to")]
VECS = np.stack([np.asarray(CONTRACTS[k]["vector"], np.float32) for k in KEYS])
VIEWS = [HEAD_CAMERA, LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA]


class Encoder(torch.nn.Module):
    def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
        B = pixel_values.shape[0]
        h = torch.zeros(B, 201, 1024) + pixel_values.float().mean()
        return types.SimpleNamespace(hidden_states=[h] * 25, last_hidden_state=h)


class Action(torch.nn.Module):
    """Drives the flow toward a contract-dependent chunk: picks close both grippers, place-in opens the left."""
    action_len = 32

    def forward(self, t, noisy_actions, qpos_history, dino_features_list, task_cond):
        key = KEYS[int(np.argmax(VECS @ task_cond[0].cpu().numpy()))]
        target = torch.zeros_like(noisy_actions)
        if key.startswith("pick up from"):
            target[..., 14], target[..., 22] = -1.0, -1.0
        elif key.startswith("place in"):
            target[..., 14], target[..., 22] = 1.0, -1.0
        i = float(t[0]) * 10
        return {"final_pred": (target - noisy_actions) * 10 / (10 - i)}


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder, self.action_model, self.views = Encoder(), Action(), VIEWS
        self.device, self.encoder_dtype = torch.device("cpu"), torch.float32
        self.feat_layers, self.include_cls_register, self.num_register_tokens = [-12, -8, -4], True, 4
        self.action_min = torch.zeros(23)
        self.action_repr_name, self.proprio_mask_name = "absolute", "positions_only"
        self.proprio_history = None

    def normalize_state(self, s): return s
    def denormalize_action(self, x): return x
    def to_absolute_action(self, a, s): return a

    def get_vision_features(self, px):
        B = px.shape[0]
        out = self.vision_encoder(pixel_values=px.reshape(-1, *px.shape[-3:]))
        return [out.hidden_states[l].reshape(B, -1, 201, 1024) for l in self.feat_layers]


def loader(name, checkpoint, device, cache):
    cfg = OmegaConf.create({"dataset": {"image_size": [224, 224], "cond_mode": "contract_proto", "contract_targets_sha256": TARGETS_SHA},
                            "model": {"views": VIEWS, "vision_encoder": {"checkpoint_path": "models/dinov3-vitl16-pretrain-lvd1689m"}},
                            "common": {"proprio_len": 1}})
    return PolicyModel(name, checkpoint, 0, cfg, "contract_proto", Wrapper())


class World:
    """Holonomic base + static floor objects; grasp/release outcomes decided from the docking geometry."""

    def __init__(self, fail_grasp=(), drop_release=(), unreachable=()):
        self.pose = np.array([0.0, 0.0, 0.3])
        self.bin = {"pos": np.array([2.5, 1.0]), "held": False}
        self.cans = [{"pos": p, "state": "floor"} for p in (np.array([1.0, 4.0]), np.array([-1.5, 3.5]), np.array([0.5, 5.5]))]
        self.state = np.zeros(61, np.float32)
        self.state[24:26] = self.state[49:51] = 0.05
        self.state[17:20], self.state[42:45] = (0.45, 0.25, 0.25), (0.55, -0.3, 0.25)
        self.fail_grasp, self.drop_release, self.unreachable = list(fail_grasp), list(drop_release), set(unreachable)
        self.pick_counter, self.release_counter = 0, 0
        self.closed_for = {"left": 0, "right": 0}
        self.holding_can = None
        self.events = []

    def rel(self, p):
        return rot(-self.pose[2]) @ (p - self.pose[:2])

    def detections(self, obs, state):
        out = []
        items = [] if self.bin["held"] else [("trash_can", self.bin["pos"])]
        items += [("can_of_soda", c["pos"]) for i, c in enumerate(self.cans) if c["state"] == "floor" and i not in self.unreachable]
        for label, p in items:
            r = self.rel(p)
            if 0.35 < np.linalg.norm(r) < 7.0 and abs(math.atan2(r[1], r[0])) < math.radians(50):
                out.append({"label": label, "xyz_base": np.r_[r, 0.0], "score": 1.0})
        return out

    def step(self, action):
        vel = np.clip(action[:3], -1, 1) * BASE_SCALE
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.pose = np.array([self.pose[0] + (vel[0] * c - vel[1] * s) * DT, self.pose[1] + (vel[0] * s + vel[1] * c) * DT,
                              wrap(self.pose[2] + vel[2] * DT)])
        self.state[0:3] = vel
        # ideal joint position tracking (trunk, left arm, right arm); end effectors are simulated separately below
        self.state[53:57], self.state[3:10], self.state[28:35] = action[3:7], action[7:14], action[15:22]
        for hand, gi in (("left", 14), ("right", 22)):
            self.closed_for[hand] = self.closed_for[hand] + 1 if action[gi] < 0 else 0
        # right hand grasps the bin when it is at the docking pose (front-right) and the hand closes
        if not self.bin["held"] and self.closed_for["right"] == 15:
            r = self.rel(self.bin["pos"])
            if np.linalg.norm(r - np.array([0.626, -0.304])) < 0.2:
                self.bin["held"] = True
                self.state[49:51] = 0.006
                self.events.append("bin_grasped")
            else:
                self.state[49:51] = 0.0
        if self.bin["held"] and self.closed_for["right"] > 15:
            self.state[44] = min(0.6, self.state[44] + 0.01)
        # left hand grasps the nearest floor can at the docking pose (front-left)
        if self.holding_can is None and self.closed_for["left"] == 15:
            self.pick_counter += 1
            cand = [(i, cn) for i, cn in enumerate(self.cans) if cn["state"] == "floor"]
            best = min(cand, key=lambda ic: np.linalg.norm(self.rel(ic[1]["pos"]) - np.array([0.541, 0.259])), default=None)
            ok = (best is not None and np.linalg.norm(self.rel(best[1]["pos"]) - np.array([0.541, 0.259])) < 0.2
                  and self.pick_counter not in self.fail_grasp)
            if ok:
                self.holding_can = best[0]
                self.cans[best[0]]["state"] = "hand"
                self.state[24:26] = 0.037
                self.state[19] = 0.12
                self.events.append(f"can{best[0]}_grasped")
            else:
                self.state[24:26] = 0.0
                self.events.append("empty_grasp")
        if self.holding_can is not None and self.closed_for["left"] > 15:
            self.state[19] = min(0.6, self.state[19] + 0.01)
            # carry pose: over the bin unless this release is injected as a drop
            over = (self.release_counter + 1) not in self.drop_release
            self.state[17:19] = self.state[42:44] + (np.array([-0.05, 0.15]) if over else np.array([0.0, 0.6]))
            self.state[19] = self.state[44] + 0.05
        if action[14] > 0:
            if self.holding_can is not None:
                self.release_counter += 1
                over = float(np.linalg.norm(self.state[17:19] - self.state[42:44])) <= 0.3
                cn = self.cans[self.holding_can]
                if over:
                    cn["state"] = "bin"
                    self.events.append(f"can{self.holding_can}_in_bin")
                else:
                    cn["state"] = "floor"
                    cn["pos"] = self.pose[:2] + rot(self.pose[2]) @ np.array([0.6, 0.5])
                    self.events.append(f"can{self.holding_can}_dropped")
                self.holding_can = None
            self.state[24:26] = 0.05
        if action[14] < 0 and self.holding_can is None and self.closed_for["left"] > 15:
            self.state[24:26] = 0.0

    def obs(self):
        img = np.zeros((224, 224, 3), np.uint8)
        return {"robot_r1::proprio": self.state.copy(), "robot_r1::robot_r1:zed_link:Camera:0::rgb": img,
                "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": img,
                "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": img}


def run(name, world, max_steps=9000):
    orch = {"vlm": "none", "program": "configs/task001_fixed_program.json", "frozen": True, "train_targets": TARGETS,
            "navigation": "docking", "postconditions": "task1_physical", "rollback_on_release_failure": True,
            "skip_failed_objects": True, "reapproach_on_retry": True, "max_attempts": 2, "precondition": False,
            "visible_objects": ["trash_can", "can_of_soda", "floors"], "perceive_every": 5,
            "min_budget_steps": {"move to": 1500, "pick up from": 450, "place in": 300}}
    with tempfile.TemporaryDirectory() as tmp:
        p = WamPolicy("stub://task1", skill="move to", exec_horizon=8, num_steps=10, device="cpu", cuda_fraction=0.5,
                      seed=0, log_dir=tmp, orchestrate=orch, samples=1, model_loader=loader)
        p.perception = TaskPerception(fake=world.detections)
        for t in range(max_steps):
            a = p.act(world.obs()).numpy()
            world.step(a)
            if p.orch.status in ("done", "held"):
                break
        p.finish_episode()
        trace = [json.loads(l) for f in Path(tmp).glob("policy_trace_*.jsonl") for l in f.read_text().splitlines()]
    ev = [e["event"] for e in trace]
    in_bin = sum(c["state"] == "bin" for c in world.cans)
    print(f"[{name}] steps={t + 1} status={p.orch.status} cans_in_bin={in_bin} world={world.events}")
    print(f"   orch events: contract_end={ev.count('orch_contract_end')} rollback={ev.count('orch_recovery_rollback')} "
          f"reapproach={ev.count('orch_recovery_reapproach')} replan={ev.count('orch_replan')} skip={ev.count('skip_failed_object')}")
    return p, world, trace


def main():
    _, w, _ = run("nominal", World())
    assert w.bin["held"] and sum(c["state"] == "bin" for c in w.cans) == 3, "nominal run must put all three cans in the bin"
    _, w, tr = run("empty grasp once + one drop", World(fail_grasp=[2], drop_release=[2]))
    ev = [e["event"] for e in tr]
    assert "orch_recovery_reapproach" in ev and "orch_recovery_rollback" in ev, "recovery paths not exercised"
    assert sum(c["state"] == "bin" for c in w.cans) == 3, "recovery must still finish all three cans"
    _, w, tr = run("one unreachable can -> skip", World(unreachable={1}))
    assert sum(c["state"] == "bin" for c in w.cans) == 2 and "skip_failed_object" in [e["event"] for e in tr]
    print("PASS check_task1_system (CPU wiring check; not a physics or model-quality result)")


if __name__ == "__main__":
    main()
