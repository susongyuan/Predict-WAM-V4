#!/usr/bin/env python3
"""CPU WIRING CHECK of the Task 0 (turning_on_radio) system path. No simulator, no GPU, no model weights.

It checks WIRING ONLY. It is not a physics, perception-accuracy or model-quality result, and it never demonstrates task
success (only the evaluator's Q/BDDL can):

  * WamPolicy is built exactly as the real policy server builds it under P2_TASK_ID=0: the orchestrate JSON comes from
    `DRY_RUN=1 scripts/run_task0_closed_loop.sh`, and the program, skill-program, VTT, contract-target and docking
    calibration files are the real ones. The PolicyModel contract (views, image size, cond mode, training
    contract-target sha) comes from the Task 0 checkpoint's embedded config. Only the network is a stub, a scripted
    joint-space stand-in for the WAM. This catches start-up crashes such as a missing 'press' closure entry.
  * move to(radio) is run by the docking controller from base-frame radio detections. It must stop at the TRAIN
    grasp-time relative pose next to a coffee table (0.42 m top, legs) under the Task 0 height-aware shield, without the
    base or torso touching the table. A control run with the Task 1 2-D shield on the same table must fail to dock,
    which shows the test can tell the two apart.
  * the stub WAM runs only the manipulation contracts, with the base held: the right hand picks and holds the radio,
    and the left closed fist presses.
  * Task 0 proprio predicates complete pick and place. press is an unverified contact candidate: the program advances
    without a second press. A failed press is retried only when retry_safe (the fist never came near); a possible
    contact without a candidate holds instead of re-pressing.
  * injected failures: missed grasp -> re-approach + retry -> success; radio released away from the table -> safe HOLD,
    and no success is ever recorded.

    .conda-envs/lilawam/bin/python scripts/check_task0_system.py        (sets P2_TASK_ID=0; ~CPU minutes, <2 GB RAM)
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("P2_TASK_ID", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from harness import task_profile  # noqa: E402
from harness.docking_controller import BASE_SCALE, DT, DockConfig, load_calibration, rot, wrap  # noqa: E402
from harness.policy_server import PolicyModel, WamPolicy  # noqa: E402
from harness.r1pro_kinematics import R1ProKinematics  # noqa: E402
from harness.task1_perception_adapter import TaskPerception  # noqa: E402
from harness.wam_behavior import HEAD_CAMERA, LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA  # noqa: E402

assert task_profile.TASK_ID == 0, "run with P2_TASK_ID=0"
torch.set_num_threads(2)
LAUNCHER = ROOT / "scripts/run_task0_closed_loop.sh"
CKPT = ROOT / "outputs/training/task0_manip_v1_20260927/checkpoint_final.pt"
KIN = R1ProKinematics()
POST = json.loads((task_profile.DOCK_CALIB / "posture_and_gripper_stats_train.json").read_text())["postures"]
DOCK = np.asarray(load_calibration(target_at="grasp")["dock"]["radio"]["target"], float)   # radio at grasp, base frame


def posture(key):
    return {g: np.asarray(POST[key][g]["median"], float) for g in ("trunk", "larm", "rarm")}


def fk(arm, trunk, q7):
    return KIN.fk(arm, np.r_[trunk, q7])[0]


# ---- scripted joint-space stand-in for the WAM (targets from TRAIN demo postures + IK, NOT a model) --------------------
P_PICK0, P_PICK1, P_PRESS = posture("pick up from_start|radio"), posture("pick up from_end|radio"), posture("press_start|radio")


def fist_pose(dist):
    """press_start posture with the left arm moved by IK so the left EEF sits at the demo press offset from the right
    EEF (TRAIN median (-0.05, +0.12, -0.03)), rescaled to |offset| = dist."""
    right = fk("right", P_PRESS["trunk"], P_PRESS["rarm"])
    off = np.array([-0.0502, 0.1234, -0.0318])
    goal = right + off / np.linalg.norm(off) * dist
    q, info = KIN.ik("left", np.r_[P_PRESS["trunk"], P_PRESS["larm"]], goal, iters=600)
    assert info["pos_err"] < 0.06, info            # arm-crossing goal: only the resulting distance matters here
    got = float(np.linalg.norm(fk("left", P_PRESS["trunk"], q[4:]) - right))
    return {**P_PRESS, "larm": q[4:]}, got


P_FIST, FIST_DIST = fist_pose(0.133)        # demo-like fist at the held radio (inside the predicate's 0.10-0.23 m)
P_NEAR, NEAR_DIST = fist_pose(0.27)         # stops short: possible contact (< 0.30 m) but never a candidate
assert 0.10 <= FIST_DIST <= 0.20 and 0.235 <= NEAR_DIST <= 0.29, (FIST_DIST, NEAR_DIST)
STUB = {}                                    # test-double side channel: scenario knobs + the stub's current contract


def stub_reset(**kw):
    STUB.clear()
    STUB.update(key=None, press_pose=P_FIST, place_pose=P_PICK0,
                place_open_z=fk("right", P_PICK0["trunk"], P_PICK0["rarm"])[2] + 0.05)
    STUB.update(kw)


def _action_for(key, s):
    a = np.zeros(23, np.float32)
    rfs = float(s[49:51].sum())
    holding = 0.02 <= rfs <= 0.085
    if key.startswith("pick up from"):
        p, lg, rg = (P_PICK1 if holding else P_PICK0), 1.0, -1.0
    elif key.startswith("press"):
        p, lg, rg = STUB["press_pose"], -1.0, -1.0
    elif key.startswith("place on"):
        p, lg = STUB["place_pose"], 1.0   # lower the radio back toward the grasp height, then open the right hand
        rg = -1.0 if float(s[44]) > STUB["place_open_z"] else 1.0
    else:
        raise AssertionError(f"stub WAM asked to run {key!r} (navigation must never reach the WAM)")
    a[3:7], a[7:14], a[14], a[15:22], a[22] = p["trunk"], p["larm"], lg, p["rarm"], rg
    return a


class Encoder(torch.nn.Module):
    def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
        B = pixel_values.shape[0]
        h = torch.zeros(B, 201, 1024) + pixel_values.float().mean()
        return types.SimpleNamespace(hidden_states=[h] * 25, last_hidden_state=h)


class Action(torch.nn.Module):
    action_len = 32

    def __init__(self, keys, vecs):
        super().__init__()
        self.keys, self.vecs = keys, vecs

    def forward(self, t, noisy_actions, qpos_history, dino_features_list, task_cond):
        key = self.keys[int(np.argmin(np.linalg.norm(self.vecs - task_cond[0].cpu().numpy()[None], axis=1)))]   # exact vector
        STUB["key"] = key
        target = torch.from_numpy(_action_for(key, qpos_history[0, 0].cpu().numpy()))[None, None].expand_as(noisy_actions)
        i = float(t[0]) * 10
        return {"final_pred": (target - noisy_actions) * 10 / (10 - i)}


class Wrapper(torch.nn.Module):
    def __init__(self, views, lags, keys, vecs):
        super().__init__()
        self.vision_encoder, self.action_model, self.views = Encoder(), Action(keys, vecs), list(views)
        self.device, self.encoder_dtype = torch.device("cpu"), torch.float32
        self.feat_layers, self.include_cls_register, self.num_register_tokens = [-12, -8, -4], True, 4
        self.action_min = torch.zeros(23)
        self.action_repr_name, self.proprio_mask_name = "absolute", "positions_no_gripper"
        self.proprio_history = None if list(lags) == [0] else {"lags": list(lags)}

    def normalize_state(self, s): return s
    def denormalize_action(self, x): return x
    def to_absolute_action(self, a, s): return a

    def get_vision_features(self, px):
        B = px.shape[0]
        out = self.vision_encoder(pixel_values=px.reshape(-1, *px.shape[-3:]))
        return [out.hidden_states[l].reshape(B, -1, 201, 1024) for l in self.feat_layers]


def checkpoint_config():
    """The Task 0 checkpoint's embedded config (mmap, CPU; weights never used), or a stand-in if it is absent."""
    if CKPT.exists():
        ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
        cfg, source = OmegaConf.create(ck["config"]), str(CKPT.relative_to(ROOT))
        del ck
        return cfg, source
    sha = hashlib.sha256((ROOT / "data/skill_vtt_task000/contract_targets.json").read_bytes()).hexdigest()
    return OmegaConf.create({"dataset": {"image_size": [224, 224], "cond_mode": "contract_proto", "contract_targets_sha256": sha},
                             "model": {"views": [HEAD_CAMERA, LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA],
                                       "vision_encoder": {"checkpoint_path": "models/dinov3-vitl16-pretrain-lvd1689m"}},
                             "common": {"proprio_len": 1}}), "synthetic stand-in (checkpoint absent)"


def launcher_orchestrate():
    env = {**os.environ, "DRY_RUN": "1", "NAME": "cpu_wiring_check", "CKPT": str(CKPT)}
    out = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise SystemExit(f"launcher dry run failed:\n{out.stdout}\n{out.stderr}")
    line = next(l for l in out.stdout.splitlines() if l.startswith("ORCHESTRATE="))
    launch = next(l for l in out.stdout.splitlines() if l.startswith("LAUNCH:"))
    return json.loads(line.split("=", 1)[1]), launch


# ---- a small kinematic world: holonomic base, coffee table, radio, ideal joint tracking, FK end effectors ---------------
class World:
    TABLE_TOP = 0.42
    RADIO_GRASP_Z = 0.64              # handle grasp point (TRAIN median 0.641); body 0.42..0.66
    BASE_BOX = (-0.41, 0.25, -0.355, 0.355)       # R1Pro base_link (curobo spheres), what physically hits table legs
    TORSO_BOX = (-0.18, 0.02, -0.10, 0.10)        # torso column (collides with the table top if driven under it)

    def __init__(self, fail_grasps=(), press_block_steps=0, start_yaw=-0.9):
        self.pose = np.array([0.0, 0.0, float(start_yaw)])
        self.radio = np.array([2.6, 1.2])
        self.radio_z = self.RADIO_GRASP_Z
        self.radio_state = "table"
        x0, y0 = self.radio[0] - 0.36, self.radio[1] - 0.60       # radio 0.36 m behind the near table edge (inst 65)
        self.table = (x0, x0 + 0.80, y0, y0 + 1.20)
        tx = np.arange(self.table[0], self.table[1] + 1e-9, 0.08)
        ty = np.arange(self.table[2], self.table[3] + 1e-9, 0.08)
        self.top = np.stack(np.meshgrid(tx, ty), -1).reshape(-1, 2)
        legs = [(self.table[i] + s * 0.04, self.table[j] + t * 0.04) for i, s in ((0, 1), (1, -1)) for j, t in ((2, 1), (3, -1))]
        g = np.linspace(-0.02, 0.02, 3)
        self.legs = np.array([(lx + dx, ly + dy) for lx, ly in legs for dx in g for dy in g])
        self.state = np.zeros(61, np.float32)
        self.state[53:57] = posture("nav_mid|before_bin")["trunk"]
        self.state[24:26] = self.state[49:51] = 0.05
        self._eef()
        self.fail_grasps, self.press_block = set(fail_grasps), int(press_block_steps)
        self.grasp_attempts, self.closed_right, self.holding, self.grasp_z = 0, 0, False, None
        self.contact_n, self.toggles, self.toggled = 0, 0, False
        self.press_steps = 0
        self.leg_contacts = self.torso_contacts = 0
        self.events, self.dock_errors = [], []
        self.rng = np.random.default_rng(0)

    def _eef(self):
        s = self.state
        s[17:20] = fk("left", s[53:57], s[3:10])
        s[42:45] = fk("right", s[53:57], s[28:35])

    def rel(self, p):
        return rot(-self.pose[2]) @ (np.asarray(p, float) - self.pose[:2])

    def _in_fov(self, r, fov=50, rmin=0.35, rmax=7.0):
        return rmin < float(np.linalg.norm(r)) < rmax and abs(math.degrees(math.atan2(r[1], r[0]))) < fov

    def radio_world(self):
        if self.holding:
            return self.pose[:2] + rot(self.pose[2]) @ self.state[42:44]
        return self.radio

    def detections(self, obs, state):
        r = self.rel(self.radio_world())
        if not self._in_fov(r):
            return []
        z = float(self.state[44]) if self.holding else self.radio_z
        return [{"label": "radio", "xyz_base": np.r_[r + self.rng.normal(0, 0.01, 2), z], "score": 0.9, "held": self.holding}]

    def bands(self, obs, state, edges):
        """Height-banded head-depth points of the table top, legs and the (unmasked) radio body, base frame, FOV-limited."""
        pts = [(p, self.TABLE_TOP) for p in self.top]
        pts += [(p, z) for p in self.legs for z in (0.10, 0.25, 0.36)]
        rw = self.radio_world()
        pts += [(rw + np.array([dx, dy]), z) for dx in (-0.06, 0.06) for dy in (-0.15, 0.0, 0.15) for z in (0.5, 0.62)]
        out = []
        for lo, hi, sup in edges:
            P = [self.rel(p) for p, z in pts if lo <= z < hi]
            P = np.array([q for q in P if self._in_fov(q, fov=45, rmin=0.2, rmax=4.0)]).reshape(-1, 2)
            out.append((lo, hi, sup, P))
        return out

    def _collides(self, pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        R = np.array([[c, s], [-s, c]])
        def inside(P, box):
            if not len(P):
                return False
            Q = (P - pose[:2]) @ R.T
            x0, x1, y0, y1 = box
            return bool(np.any((Q[:, 0] > x0) & (Q[:, 0] < x1) & (Q[:, 1] > y0) & (Q[:, 1] < y1)))
        return inside(self.legs, self.BASE_BOX), inside(self.top, self.TORSO_BOX)

    def step(self, action):
        a = np.asarray(action, float)
        vel = np.clip(a[:3], -1, 1) * BASE_SCALE
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        new = np.array([self.pose[0] + (vel[0] * c - vel[1] * s) * DT, self.pose[1] + (vel[0] * s + vel[1] * c) * DT,
                        wrap(self.pose[2] + vel[2] * DT)])
        leg, torso = self._collides(new)
        if leg or torso:                       # the table would be pushed: the motion does not happen, it is counted
            self.leg_contacts += int(leg)
            self.torso_contacts += int(torso)
            vel = np.zeros(3)
        else:
            self.pose = new
        self.state[0:3] = vel
        # ideal joint tracking (the press injection blocks the left arm for the first press steps)
        pressing = STUB.get("key", "") and STUB["key"].startswith("press") and a[14] < 0
        self.press_steps += int(bool(pressing))
        self.state[53:57], self.state[28:35] = a[3:7], a[15:22]
        if not (pressing and self.press_steps <= self.press_block):
            self.state[3:10] = a[7:14]
        self._eef()
        # left hand: open or a closed fist (never grasps anything)
        self.state[24:26] = 0.0 if a[14] < 0 else 0.05
        # right hand: grasp the radio when closed for 15 steps with the radio at the calibrated grasp-time pose
        if a[22] < 0:
            self.closed_right += 1
            if not self.holding and self.closed_right == 15:
                self.grasp_attempts += 1
                err = float(np.linalg.norm(self.rel(self.radio) - DOCK)) if self.radio_state in ("table", "placed") else 9.0
                if err < 0.12 and self.grasp_attempts not in self.fail_grasps:
                    self.holding, self.radio_state, self.grasp_z = True, "held", float(self.state[44])
                    self.events.append(f"grasped(err={err:.3f})")
                else:
                    self.events.append(f"empty_grasp(err={err:.3f})")
            self.state[49:51] = 0.019 if self.holding else (0.0 if self.closed_right >= 15 else 0.05)
        else:
            self.closed_right = 0
            if self.holding:
                dz = float(self.state[44]) - self.grasp_z
                self.radio = self.radio_world()
                self.radio_state = "placed" if -0.05 <= dz <= 0.10 else "dropped"
                self.radio_z = self.RADIO_GRASP_Z if self.radio_state == "placed" else 0.12
                self.holding = False
                self.events.append(f"released_{self.radio_state}(dz={dz:.3f})")
            self.state[49:51] = 0.05
        # simulator truth (never shown to the policy): ToggledOn flips once per continuous 5-step fist contact
        d = float(np.linalg.norm(self.state[17:20] - self.state[42:45]))
        contact = self.holding and a[14] < 0 and d <= 0.20
        self.contact_n = self.contact_n + 1 if contact else 0
        if self.contact_n == 5:
            self.toggled, self.toggles = not self.toggled, self.toggles + 1
            self.events.append(f"TOGGLE->{'on' if self.toggled else 'off'}")

    def obs(self):
        img = np.zeros((224, 224, 3), np.uint8)
        return {"robot_r1::proprio": self.state.copy(), "robot_r1::robot_r1:zed_link:Camera:0::rgb": img,
                "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": img,
                "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": img}


def make_policy(orch, cfg, tmp, dock_config=None):
    targets = json.loads((ROOT / orch["train_targets"]).read_text())["contracts"]
    keys = [k for k in targets if not k.startswith("move to")]
    vecs = np.stack([np.asarray(targets[k]["vector"], np.float32) for k in keys])

    def loader(name, checkpoint, device, cache):
        from harness.wam_behavior import model_proprio_lags
        return PolicyModel(name, checkpoint, 4000, cfg, cfg.dataset.get("cond_mode"),
                           Wrapper(cfg.model.views, model_proprio_lags(cfg), keys, vecs))
    p = WamPolicy(str(CKPT), skill="move to", exec_horizon=8, num_steps=10, device="cpu", cuda_fraction=0.5, seed=0,
                  log_dir=tmp, orchestrate=orch, samples=1, model_loader=loader)
    if dock_config is not None:
        p.docking.cfg = dock_config
    return p


def run(name, world, orch, cfg, max_steps=3224, dock_config=None, **stub):
    stub_reset(**stub)
    with tempfile.TemporaryDirectory() as tmp:
        p = make_policy(orch, cfg, tmp, dock_config)
        # fake perception replaces ONLY the detector / depth (a real run uses harness.radio_perception on head RGB-D)
        p.perception = TaskPerception(fake=world.detections, fake_bands=world.bands)
        docked_err = []
        for t in range(max_steps):
            a = p.act(world.obs()).numpy()
            st = p.docking.last_status if p.docking is not None else {}
            if st.get("docked") and p.orch.contract is not None and p.orch.contract.is_navigation:
                docked_err.append(float(np.linalg.norm(world.rel(world.radio_world()) - DOCK)))
            world.step(a)
            if p.orch.status in ("done", "held"):
                break
        p.finish_episode()
        trace = [json.loads(l) for f in Path(tmp).glob("policy_trace_*.jsonl") for l in f.read_text().splitlines()]
    ev = [e for e in trace if e["event"].startswith("orch_")]
    starts = [e["key"] for e in ev if e["event"] == "orch_contract_start"]
    ends = [e for e in ev if e["event"] == "orch_contract_end"]
    holds = [e for e in ev if e["event"] == "orch_hold"]
    res = {"name": name, "steps": t + 1, "status": p.orch.status, "hold_reason": p.orch.pending_hold_reason,
           "starts": starts, "ends": [(e["key"], e["success"], e.get("predicate"), e.get("verified")) for e in ends],
           "holds": [e.get("reason") for e in holds], "events": [e["event"] for e in ev], "world": world.events,
           "toggles": world.toggles, "radio_on": world.toggled, "leg_contacts": world.leg_contacts,
           "torso_contacts": world.torso_contacts, "dock_err_at_docked": round(min(docked_err), 3) if docked_err else None,
           "chunks": [e.get("arm_key") for e in trace if e["event"] == "chunk"], "trace": trace}
    print(f"[{name}] steps={res['steps']} status={res['status']} hold={res['hold_reason']} toggles={world.toggles} "
          f"radio_on={world.toggled} radio={world.radio_state} leg/torso contacts={world.leg_contacts}/{world.torso_contacts} "
          f"dock_err={res['dock_err_at_docked']}")
    print(f"   contracts: {[(k, s, pr) for k, s, pr, _ in res['ends']]}")
    print(f"   world: {world.events}")
    return res


def press_ends(res):
    return [e for e in res["ends"] if e[0] == "press(radio)"]


def main():
    orch, launch = launcher_orchestrate()
    cfg, cfg_source = checkpoint_config()
    print(f"orchestrate (from the launcher dry run): {json.dumps(orch)}")
    print(f"model contract from: {cfg_source}; views={list(cfg.model.views)}; fist offsets {FIST_DIST:.3f} / {NEAR_DIST:.3f} m")
    assert orch["postconditions"] == "task0_physical" and orch["manip_executor"] == "wam" and orch["dock_target_at"] == "grasp"
    assert orch["base_control"] == "hold_during_manipulation" and orch["navigation"] == "docking"
    assert "--log-object-poses" in launch and "HeadRGBDWrapper" in launch and "P2_TASK_ID=0" in launch
    results = {}

    # 1) nominal: dock at the table, pick (right), press (left fist) -> unverified candidate, place, program done
    r = results["nominal"] = run("nominal", World(), orch, cfg)
    assert r["status"] == "done", r["status"]
    assert r["starts"].count("press(radio)") == 1 and press_ends(r) == [("press(radio)", True, "unverified_outcome_candidate", False)]
    assert ("place on(radio,coffee_table)", True, "calibrated_postcondition", True) in r["ends"]
    assert ("pick up from(radio,coffee_table)", True, "calibrated_postcondition", True) in r["ends"]
    assert r["toggles"] == 1 and r["radio_on"] and r["leg_contacts"] == 0 and r["torso_contacts"] == 0
    assert r["dock_err_at_docked"] is not None and r["dock_err_at_docked"] < 0.08, r["dock_err_at_docked"]
    press_i = max(i for i, k in enumerate(r["chunks"]) if k == "press(radio)")
    assert all(k != "press(radio)" for k in r["chunks"][press_i + 1:])
    assert all(k is not None and not k.startswith("move to") for k in r["chunks"]), "navigation reached the WAM"
    # the policy never records task success: the press ended unverified; bank frozen, no memory arm, no reliability update
    assert not any(e["event"] == "orch_contract_end" and e["key"] == "press(radio)" and e.get("verified") for e in r["trace"])

    # 2) missed grasp once -> re-approach + retry -> success
    r = results["missed_grasp"] = run("missed grasp once", World(fail_grasps={1}), orch, cfg)
    assert r["status"] == "done" and "orch_recovery_reapproach" in r["events"], (r["status"], r["events"])
    assert r["starts"].count("pick up from(radio,coffee_table)") == 2 and r["starts"].count("move to(radio)") == 2
    assert any(e[0].startswith("pick up") and e[2] == "calibrated_postcondition_failed" for e in r["ends"])
    assert r["toggles"] == 1 and r["starts"].count("press(radio)") == 1 and r["leg_contacts"] == 0

    # 3) press attempt 1 never approaches (no possible contact: retry_safe) -> retried -> candidate -> advance
    r = results["press_retry_safe"] = run("press blocked, retry_safe", World(press_block_steps=650), orch, cfg)
    assert r["status"] == "done" and r["starts"].count("press(radio)") == 2, (r["status"], r["starts"])
    first = next(e for e in r["trace"] if e["event"] == "orch_contract_end" and e["key"] == "press(radio)")
    assert not first["success"] and first["postcondition"]["retry_safe"] is True, first
    assert r["toggles"] == 1

    # 4) press stops 0.26 m from the held radio: possible contact, no candidate -> HOLD, never a second press
    r = results["press_unknown"] = run("press possible contact, no candidate", World(), orch, cfg, press_pose=P_NEAR)
    assert r["status"] == "held" and r["hold_reason"] == "irreversible_outcome_unknown", (r["status"], r["hold_reason"])
    assert r["starts"].count("press(radio)") == 1 and "place on(radio,coffee_table)" not in r["starts"]
    assert r["toggles"] == 0

    # 5) the radio is released 0.3 m above the table -> verified place failure -> in-place retry sees it is not in hand
    #    -> HOLD; no success claimed, no second press
    r = results["place_drop"] = run("released above the table", World(), orch, cfg, place_pose=P_PRESS,
                                  place_open_z=float("inf"))
    assert r["status"] == "held" and r["hold_reason"] == "required_skill_failed", (r["status"], r["hold_reason"])
    places = [e for e in r["ends"] if e[0].startswith("place on")]
    assert places and not any(e[1] for e in places) and places[0][2] == "calibrated_postcondition_failed", places
    assert any(e["event"] == "orch_contract_end" and e["key"].startswith("place on")
               and e["postcondition"]["reason"] == "released_away_from_surface" for e in r["trace"])
    assert "orch_recovery_in_place" in r["events"] and r["starts"].count("press(radio)") == 1

    # 6) control: the Task 1 2-D shield (symmetric footprint, no height bands) on the same table does not dock
    r = results["control_2d_shield"] = run("control: Task-1 2-D shield", World(), orch, cfg, max_steps=1500,
                                           dock_config=DockConfig())
    assert "pick up from(radio,coffee_table)" not in r["starts"], "2-D shield docked: the table test lost its power"
    assert r["leg_contacts"] == 0

    summary = {k: {kk: v[kk] for kk in ("steps", "status", "hold_reason", "starts", "ends", "toggles", "radio_on",
                                        "leg_contacts", "torso_contacts", "dock_err_at_docked", "world")}
               for k, v in results.items()}
    print(json.dumps({"check_task0_system": "PASS", "model_contract_source": cfg_source,
                      "note": "CPU wiring check only (stub WAM, fake perception, kinematic world); not a physics, "
                              "perception or model-quality result and never task success"}, indent=1))
    return summary


if __name__ == "__main__":
    main()
    print("PASS check_task0_system (CPU WIRING check only; not a physics, perception or model-quality result)")
