#!/usr/bin/env python3
"""Open-loop demo-ACTION replay: environment / evaluator validity diagnostic (BEHAVIOR-1K, Isaac Sim 6.1 port).

Task-aware through harness/task_profile.py: environment variable P2_TASK_ID, default 13 = Task 13 loading_the_car with
its original paths and behaviour unchanged; P2_TASK_ID=1 = Task 1 picking_up_trash (see "Task 1" below).

DIAGNOSTIC ONLY -- label `diagnostic: demo_action_replay`, `policy_result: false`, `official_score_valid: false`.
The recorded 23-D actions of ONE TRAIN demonstration are served step by step, open loop, by a minimal websocket
"policy" speaking the evaluator protocol of harness/policy_server.py (msgpack + numpy extension, /healthz, reset
messages), on the TRAIN task instance the demo was recorded on (never public_test dev/report instances 301-320).
It is a replay of a recorded answer trajectory, never a policy/task/skill result, and nothing here is WAM input.

Questions: (a) does the evaluator's car joint j_trunk exceed the Open threshold (5 % of range) and when?
(b) success / Q of the evaluator for the (full) replayed episode; (c) if the lid does not open: drift (replayed
proprio / odometry diverges from the demo before contact) or physics (tracking fine, lid does not lift)?

Task 1 (P2_TASK_ID=1, picking_up_trash; goal: the 3 cans inside the ashcan, Q = fraction inside at the end):
(a) evaluator success / Q / steps, and are they consistent with the BDDL goal literals inside(can_i, ashcan) logged
per step by the evaluator (--log-object-poses, opt-in port addition)? (b) per can: first step inside / over the bin
opening, final position relative to the bin, drops / misses, bin final pose, physics anomalies (evaluator-side pose
log); (c) if it fails: drift vs physics. The run also captures sim-native sensor data for perception development
(official RGBDFullResWrapper: head RGB-D 720x720, wrist RGB 480x480, every N steps; proprio + cam_rel_poses every
step) aligned with the ground-truth pose log. Mapping criterion: dataset metadata task_instance_id, and the demo
odometry from that instance's R1Pro start pose puts the trash can within 1.2 m at the 'pick up from(trash_can)'
start while the id +/- 1 alternatives do not (lead calibration 2026-09-26).

Task 0 (P2_TASK_ID=0, turning_on_radio; goal: toggled_on(radio), the evaluator ends the episode at the first step it
holds; OmniGibson ToggledOn flips when a robot finger touches the radio inside its `togglebutton` marker for 5 steps):
(a) open-loop replay: evaluator success / Q / step vs the demo's first reward frame, radio pose, per-step ToggledOn
value + hand_in_marker_steps (evaluator pose log, opt-in port field); (b) demo-prefix -> WAM handoffs on one TRAIN
demo: 'pick up from(radio,coffee_table)' (criterion: radio lowest point lifted >= 0.10 m, still >= 0.05 m at the
budget end) and 'press(radio)' (criterion: ToggledOn becomes true within the budget; step reported). Base held
(hold_during_manipulation) during the WAM contract by default. Mapping criterion: metadata task_instance_id, and the
demo odometry from that instance's start pose puts the radio within 1.2 m at the 'pick up from(radio)' start (TRAIN
calibration 2026-09-27: 88 % metadata vs 20-22 % for id +/- 1). Evaluator-side logs are never policy inputs.

V3 handoff (explicit parameters; old V2 defaults are unchanged):
  P2_TASK_ID=0 ... run ... --wam-checkpoint <V3.pt> --wam-samples 1 --wam-exec-horizon 8
  Add --wam-fixed-noise-seed <seed> only to reuse the same initial noise at EVERY replan.
  V3 history starts at the actual handoff frame, not at the demonstration prefix; each post-handoff
  control frame encodes fresh pooled features, including frames that only consume an action queue.

Subcommands (env):
  mapping   (lilawam)  demo -> train instance evidence: dataset metadata, upstream episode-id convention,
                       odometry docking check vs. alternative conventions over all 200 demos
  serve     (lilawam)  replay websocket server, CPU only, optional GPU-utilization throttle (wall-clock only)
  run       (python3)  GPU lock wait -> replay server unit -> guarded evaluator (--mode train) -> manifest -> analyze
  analyze   (behavior6 for plots; any env for numbers)  j_trunk, success/Q, tracking error, key frames, verdict
  self-test (lilawam)  CPU only: protocol round trip against fake observations + analysis on synthetic logs
"""
import argparse
import asyncio
import functools
import hashlib
import http
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from harness import task_profile as TP  # noqa: E402  P2_TASK_ID (default 13 = the original Task 13 paths, unchanged)

TASK_ID, TASK, SCENE = TP.TASK_ID, TP.TASK_NAME, "house_double_floor_lower"
DEMOS = TP.DATA
LOWDIM = TP.LOWDIM
FRAMES = TP.FRAMES
SEGMENTS = TP.SEGMENTS
SPLIT = TP.SPLIT
EPISODES_META = TP.EPISODES_META / "file-000.parquet"
INST_DIR = ROOT / f"data/omnigibson/2026-challenge-task-instances/scenes/{SCENE}/json/{SCENE}_task_{TASK}_instances"
EVAL_UTILS = ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/eval/utils/eval_utils.py"
ID_CONVENTION_SRC = ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/scripts/learning/generate_gcs_rawdata_manifest.py"
OUT_ROOT = ROOT / {13: "outputs/validation/demo_replay_feasibility_20260924",
                   1: "outputs/validation/task1_demo_replay_20260926",
                   0: "outputs/validation/task0_handoff_20260927"}[TASK_ID]
CONTRACT_TARGETS = TP.VTT_DIR / "contract_targets.json"
T1_HANDOFF_ROOT = ROOT / ("outputs/validation/task0_handoff_20260927" if TASK_ID == 0 else
                          "outputs/validation/task1_handoff_20260926")  # handoff run dirs (launch scripts live here too)
# Task 1 (picking_up_trash): BDDL scope names; docking radius from the lead calibration 2026-09-26 (metadata
# task_instance_id puts the trash can within 1.2 m at the 'pick up from(trash_can)' start in 89 % of train demos vs
# 15-17 % for id +/- 1; outputs/validation/task1_docking_calibration_20260926/).
T1_BIN = "ashcan.n.01_1"
T1_CANS = ("can__of__soda.n.01_1", "can__of__soda.n.01_2", "can__of__soda.n.01_3")
T1_DOCK_M = 1.2
T1_WRAPPERS = {"rgbd": "omnigibson.eval.wrappers.RGBDFullResWrapper", "default": "omnigibson.eval.wrappers.DefaultWrapper",
               "headrgbd": "harness.head_rgbd_wrapper.HeadRGBDWrapper"}   # head 720 RGB-D, wrists 224 RGB (VRAM, 2026-09-27)
DEPTH_SUFFIX = {"head": ":zed_link:Camera:0::depth_linear"}
DEMO_CAM_POSE_COLS = {"lwrist": "observation.robot2cam_pose.left_realsense_link_camera_0",
                      "rwrist": "observation.robot2cam_pose.right_realsense_link_camera_0",
                      "head": "observation.robot2cam_pose.zed_link_camera_0"}
# omnigibson/eval/utils/eval_utils.py CAMERA_INTRINSICS["R1Pro"] (head 720x720, wrists 480x480)
OFFICIAL_INTRINSICS = {"head": [[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]],
                       "left_wrist": [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]],
                       "right_wrist": [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]]}
HANDOFF_LABEL = {"diagnostic": "demo_prefix_handoff", "policy_result": "post-handoff segment only (train instance, diagnostic)",
                 "official_score_valid": False, "fed_to_wam": False,
                 "statement": ("Recorded demo actions are replayed for steps t < N on the demo's own TRAIN instance, then the WAM "
                               "executes the 'open lid(car)' contract. Isolates the manipulation executor from navigation/docking; "
                               "a train-instance diagnostic, never a task/skill result; nothing is used for training or selection.")}
if TASK_ID != 13:  # Task 1: one manipulation contract per run (manifest wam_config.contract); Task 13 label unchanged
    HANDOFF_LABEL = {**HANDOFF_LABEL, "statement": (
        "Recorded demo actions are replayed for steps t < N on the demo's own TRAIN instance, then the multi-view WAM "
        "executes ONE manipulation contract (manifest wam_config.contract) with no further demo input. Isolates the "
        "manipulation executor from navigation/docking; the physical outcome comes from the evaluator-side pose log only "
        "(never a policy input); a train-instance diagnostic, never a task/skill result; nothing is used for training or "
        "selection.")}
# Task 1 handoff (demo prefix -> ONE manipulation contract, lead spec 2026-09-26). Physical criteria are evaluated on the
# evaluator pose log only and written into the run manifest (handoff_criterion) BEFORE the run starts.
T1_HANDOFF_LIFT_M = 0.10        # pick up from: object lowest point >= 0.10 m above its step-0 height at some step >= N
T1_HANDOFF_END_M = 0.05         # ... and still >= 0.05 m at the last logged step (budget end or early stop)
T1_HANDOFF_BUDGET_FACTOR = 2    # default budget = 2 x (demo segment end - N)
T1_HANDOFF_EXEC_HORIZON, T1_HANDOFF_SAMPLES, T1_EARLY_STOP_STEPS = 8, 8, 60
# Task 0 (turning_on_radio, 2026-09-27): BDDL scope names; docking radius from the TRAIN calibration (metadata instance
# puts the radio within 1.2 m at the 'pick up from(radio)' start in 88 % of train demos vs 22 % / 20 % for id -1 / +1).
T0_RADIO = "radio_receiver.n.01_1"
T0_DOCK_M = 1.2
T0_BASE_CONTROL = "hold_during_manipulation"   # default WAM base ownership in Task-0 handoffs (WAM base collapses to ~0)
LOCK = Path("/tmp/project2_gpu.lock")
LILAWAM = ROOT / ".conda-envs/lilawam/bin/python"
BEHAVIOR6 = ROOT / ".conda-envs/behavior6/bin/python"
CAMS = {"head": "observation.rgb.zed_link_camera_0", "lwrist": "observation.rgb.left_realsense_link_camera_0",
        "rwrist": "observation.rgb.right_realsense_link_camera_0"}
LIVE_SUFFIX = {"head": ":zed_link:Camera:0::rgb", "lwrist": ":left_realsense_link:Camera:0::rgb",
               "rwrist": ":right_realsense_link:Camera:0::rgb"}
DT = 1.0 / 30.0
OPEN_FRACTION = 0.05  # omnigibson/object_states/open_state.py JOINT_THRESHOLD_BY_TYPE (revolute)
LABEL = {"diagnostic": "demo_action_replay", "policy_result": False, "official_score_valid": False,
         "fed_to_wam": False,
         "statement": ("Open-loop replay of a recorded TRAIN demonstration's actions on its own train task instance, to test "
                       "environment/evaluator validity of the Isaac Sim 6.1 port. Not a policy result; never report as task "
                       "or skill success; nothing from it is used for WAM training, conditioning or selection.")}
# Attribution thresholds for verdict (c); documented in the summary.
DRIFT_POS_M, DRIFT_YAW_DEG = 0.10, 5.0
ARM_OK_RAD, EEF_OK_M = 0.05, 0.03
# State layout (harness/wam_behavior.STATE_SLICES)
S = {"base_qvel": (0, 3), "arm_left_qpos": (3, 10), "eef_left_pos": (17, 20), "gripper_left_qpos": (24, 26),
     "arm_right_qpos": (28, 35), "eef_right_pos": (42, 45), "gripper_right_qpos": (49, 51), "trunk_qpos": (53, 57)}


def sl(name):
    return slice(*S[name])


def sha256_file(path, block=1 << 23):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------------------------
# demo / instance data
# ---------------------------------------------------------------------------------------------------------------
def demo_segments(episode):
    return json.loads(SEGMENTS.read_text())["episodes"][str(episode)]["segments"]


def openlid_segment(episode):
    seg = next(s for s in demo_segments(episode) if s["skill"] == "open lid")
    return int(seg["start"]), int(seg["end"])


def instance_file(k):
    return INST_DIR / f"{SCENE}_task_{TASK}_0_{k}_template-tro_state.json"


def test_instance_ids():
    m = re.search(r"TEST_INSTANCE_IDS\s*=\s*list\(range\((\d+),\s*(\d+)\)\)", EVAL_UTILS.read_text())
    if m is None:
        raise ValueError("cannot read TEST_INSTANCE_IDS from eval_utils.py")
    return list(range(int(m[1]), int(m[2])))


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_instance(k):
    d = json.loads(instance_file(k).read_text())
    pose = d["robot_poses"]["R1Pro"][0]
    start = (float(pose["position"][0]), float(pose["position"][1]), quat_yaw(pose["orientation"]))
    if TASK_ID == 0:  # Task 0: the radio (radio_receiver) on the coffee table
        return {"start": start, "radio": np.array(d[T0_RADIO]["root_link"]["pos"][:2], float),
                "radio_pose": [float(v) for v in d[T0_RADIO]["root_link"]["pos"] + d[T0_RADIO]["root_link"]["ori"]]}
    if TASK_ID != 13:  # Task 1: trash can (ashcan) and cans instead of the car
        return {"start": start, "bin": np.array(d[T1_BIN]["root_link"]["pos"][:2], float),
                "bin_pose": [float(v) for v in d[T1_BIN]["root_link"]["pos"] + d[T1_BIN]["root_link"]["ori"]],
                "cans": {c: [float(v) for v in d[c]["root_link"]["pos"]] for c in T1_CANS}}
    return {"start": start,
            "car": np.array(d["car.n.01_1"]["root_link"]["pos"][:2], float),
            "car_yaw": quat_yaw(d["car.n.01_1"]["root_link"]["ori"])}


def all_instances():
    out = {}
    for p in INST_DIR.glob(f"{SCENE}_task_{TASK}_0_*_template-tro_state.json"):
        k = int(re.search(r"_0_(\d+)_template", p.name)[1])
        out[k] = load_instance(k)
    return out


def integrate_odometry(qvel, start, dt=DT):
    """Robot-frame base velocity (vx, vy, wz) -> world (x, y, yaw); row t = pose before step t."""
    qvel = np.asarray(qvel, np.float64)
    x, y, yaw = start
    out = np.empty((len(qvel) + 1, 3))
    out[0] = start
    for i, (vx, vy, wz) in enumerate(qvel):
        c, s = math.cos(yaw), math.sin(yaw)
        x += (vx * c - vy * s) * dt
        y += (vx * s + vy * c) * dt
        yaw += wz * dt
        out[i + 1] = (x, y, yaw)
    return out


def docking(qvel_prefix, inst):
    """Where the demo's own base odometry puts the robot at 'open lid' start, if it started at this instance's pose."""
    p = integrate_odometry(qvel_prefix, inst["start"])[-1]
    d = inst["car"] - p[:2]
    dist = float(np.linalg.norm(d))
    dx, dy = float(p[0] - inst["car"][0]), float(p[1] - inst["car"][1])
    herr = math.degrees(abs((math.atan2(d[1], d[0]) - p[2] + math.pi) % (2 * math.pi) - math.pi))
    ok = bool(1.5 <= dist <= 4.5 and dx < -1.5 and abs(dy) < 1.2 and herr < 35.0)
    return {"behind_car_facing_it": ok, "distance_to_car_center_m": round(dist, 3), "dx_m": round(dx, 3),
            "dy_m": round(dy, 3), "heading_error_deg": round(herr, 1)}


def episode_table():
    import pyarrow.parquet as pq
    rows = pq.read_table(EPISODES_META, columns=["episode_index", "length", "raw_episode_id", "task_instance_id",
                                                  "annotation_path"]).to_pylist()
    return {int(r["episode_index"]): {"length": int(r["length"]), "raw_episode_id": int(r["raw_episode_id"]),
                                      "task_instance_id": int(r["task_instance_id"]),
                                      "annotation_path": r["annotation_path"]} for r in rows}


def demo_rewards(episode):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    for f in sorted((DEMOS / "data" / TP.CHUNK).glob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "next.reward", "next.terminated"])
        t = t.filter(pc.equal(t.column("episode_index"), episode))
        if t.num_rows:
            fi = np.array(t.column("frame_index").to_pylist())
            r = np.array(t.column("next.reward").to_pylist())
            term = np.array(t.column("next.terminated").to_pylist())
            return {"reward_frames": [int(fi[i]) for i in np.nonzero(r)[0]],
                    "reward_values": [round(float(r[i]), 4) for i in np.nonzero(r)[0]],
                    "first_terminated_frame": int(fi[np.argmax(term)]) if term.any() else None,
                    "source": f"{f.relative_to(ROOT)} next.reward / next.terminated"}
    raise KeyError(f"episode {episode} not found in demo parquet files")


# ---------------------------------------------------------------------------------------------------------------
# mapping evidence
# ---------------------------------------------------------------------------------------------------------------
def mapping_evidence(episode, instance):
    if TASK_ID == 0:
        return mapping_evidence_task0(episode, instance)
    if TASK_ID != 13:
        return mapping_evidence_task1(episode, instance)
    table = episode_table()
    split = json.loads(SPLIT.read_text())
    meta = table[episode]
    insts = all_instances()
    test_ids = test_instance_ids()
    seg_open = openlid_segment(episode)
    conventions = {"metadata task_instance_id": lambda r: r["task_instance_id"],
                   "metadata id - 1 (off-by-one)": lambda r: r["task_instance_id"] - 1,
                   "metadata id + 1": lambda r: r["task_instance_id"] + 1,
                   "raw_episode_id % 10000 (download_gcs_rawdata task<50 formula)": lambda r: r["raw_episode_id"] % 10000}
    population = {c: [0, 0] for c in conventions}
    this = {}
    for ep, r in sorted(table.items()):
        st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        f0 = openlid_segment(ep)[0]
        qv = np.asarray(st[:f0, :3], np.float64)
        for name, fn in conventions.items():
            k = fn(r)
            if k not in insts:
                if ep == episode:
                    this[name] = {"instance": k, "instance_file_exists": False}
                continue
            res = docking(qv, insts[k])
            population[name][1] += 1
            population[name][0] += int(res["behind_car_facing_it"])
            if ep == episode:
                this[name] = {"instance": k, **res}
    # chance level: how many of all train instances would pass for this episode
    qv = np.asarray(np.load(LOWDIM / f"episode_{episode}_state.npy", mmap_mode="r")[:seg_open[0], :3], np.float64)
    passing = sorted(k for k, inst in insts.items() if docking(qv, inst)["behind_car_facing_it"])
    raw_expected = int(f"{TASK_ID:04d}{instance:03d}0")
    pop = {c: {"pass": a, "n": n, "rate": round(a / n, 3) if n else None} for c, (a, n) in population.items()}
    alt_rates = [v["rate"] for c, v in pop.items() if c != "metadata task_instance_id" and v["rate"] is not None]
    checks = {
        "episode_in_train_split": episode in split["train_episode_indices"] and episode not in split["validation_episode_indices"],
        "metadata_task_instance_id_equals_requested": meta["task_instance_id"] == instance,
        "raw_episode_id_matches_upstream_convention": meta["raw_episode_id"] == raw_expected,
        "train_instance_file_exists": instance_file(instance).is_file(),
        "not_a_public_or_hidden_test_instance": instance not in test_ids,
        "odometry_docks_behind_car_for_metadata_instance": bool(this["metadata task_instance_id"].get("behind_car_facing_it")),
        "odometry_rejects_off_by_one_alternatives": all(not this[c].get("behind_car_facing_it", False)
                                                        for c in conventions if c != "metadata task_instance_id"),
        "population_metadata_rate_ge_0.5_and_alternatives_le_0.1": (pop["metadata task_instance_id"]["rate"] >= 0.5
                                                                     and all(r <= 0.1 for r in alt_rates)),
    }
    return {**LABEL, "created_utc": utc(), "episode": episode, "requested_train_instance": instance,
            "dataset_metadata": {"file": str(EPISODES_META.relative_to(ROOT)), **meta},
            "split": {"file": str(SPLIT.relative_to(ROOT)), "train": checks["episode_in_train_split"],
                      "closed_loop_test": split.get("closed_loop_test")},
            "upstream_id_convention": {
                "source": str(ID_CONVENTION_SRC.relative_to(ROOT)),
                "rule": "episode_id = f'{task_id:04d}{instance_id:03d}0'",
                "expected_raw_episode_id": raw_expected,
                "note": ("scripts/learning/download_gcs_rawdata.py uses f'{task:04d}{instance:04d}' for task<50 GCS raw files; "
                         "that reading (instance = raw % 10000) is tested below and rejected by the odometry check")},
            "instance_file": str(instance_file(instance).relative_to(ROOT)),
            "instance_file_sha256": sha256_file(instance_file(instance)) if instance_file(instance).is_file() else None,
            "test_instance_ids_excluded": [test_ids[0], test_ids[-1]],
            "dev_report_note": "development/report are public_test indices 10-19 / 0-9 -> instance ids 311-320 / 301-310; never used",
            "odometry_docking_check": {
                "method": ("integrate the demo's recorded robot-frame base velocity (observation.state[0:3], 30 Hz) from a candidate "
                           "instance's presampled R1Pro start pose up to the annotated 'open lid' start frame; the correct instance "
                           "must put the robot behind the car's rear (world -x side, car yaw ~0) 1.5-4.5 m from its centre, "
                           "|dy|<1.2 m, facing it within 35 deg"),
                "open_lid_start_frame": seg_open[0],
                "this_episode": this,
                "train_instances_passing_for_this_episode": {"count": len(passing), "of": len(insts), "ids": passing},
                "population_all_200_demos": pop},
            "checks": checks, "established": all(checks.values()),
            "demo_outcome": demo_rewards(episode)}


# --- Task 1 (picking_up_trash) mapping ------------------------------------------------------------------------
def trash_pick_segment(episode):
    """Task 1 anchor: the demo's 'pick up from(trash_can, floor)' segment (skill 2 of 12 in every demo)."""
    seg = next(s for s in demo_segments(episode) if s["skill"] == "pick up from"
               and s["objects"] and s["objects"][0][0].startswith("trash_can"))
    return int(seg["start"]), int(seg["end"])


def docking_task1(qvel_prefix, inst):
    """Where the demo's own base odometry puts the robot at the trash-can pick start, from this instance's start pose."""
    p = integrate_odometry(qvel_prefix, inst["start"])[-1]
    d = inst["bin"] - p[:2]
    dist = float(np.linalg.norm(d))
    bearing = math.degrees((math.atan2(d[1], d[0]) - p[2] + math.pi) % (2 * math.pi) - math.pi)
    return {"trash_can_within_1.2m": bool(dist <= T1_DOCK_M), "distance_to_trash_can_m": round(dist, 3),
            "trash_can_bearing_in_robot_frame_deg": round(bearing, 1)}


def demo_hands_task1(episode):
    """Gripper that closes (commanded open >= 0 at the segment start -> closed < 0 at its end) in each pick segment."""
    act = np.load(LOWDIM / f"episode_{episode}_action.npy", mmap_mode="r")
    out = []
    for s in demo_segments(episode):
        if s["skill"] == "pick up from":
            a, e = int(s["start"]), min(int(s["end"]), len(act) - 1)
            out.append({"object": s["objects"][0][0], "segment": [a, int(s["end"])],
                        "hand_closing": [h for h, i in (("left", 14), ("right", 22)) if act[a, i] >= 0 > act[e, i]]})
    return out


def t1_handoff_spec(episode, handoff_step, contract=None, skill=None, budget=None):
    """Task 1 handoff: the demo segment containing N must BE the contract (key built like the training contracts,
    harness.vtt_monitor.precondition_key). Returns the pre-registered physical criterion (target object, kind,
    thresholds), the demo's gripper events in that segment and the budget (default 2 x (segment end - N))."""
    from harness.vtt_monitor import precondition_key
    N = int(handoff_step)
    seg = next((s for s in demo_segments(episode) if int(s["start"]) <= N < int(s["end"])), None)
    if seg is None or not seg["objects"]:
        raise ValueError(f"handoff step {N} is not inside an annotated skill segment of episode {episode}")
    a, e = int(seg["start"]), int(seg["end"])
    key = precondition_key(seg["skill"], seg["objects"])
    contract = key if contract is None else str(contract)
    if contract != key:
        raise ValueError(f"contract {contract!r} differs from the demo segment [{a}, {e}) at step {N}: {key!r}")
    skill = seg["skill"] if skill is None else str(skill)
    if skill != contract.partition("(")[0]:
        raise ValueError(f"skill {skill!r} differs from the contract skill {contract.partition('(')[0]!r}")
    obj = seg["objects"][0]
    if skill == "pick up from":
        kind, container = "lift", None
        text = (f"{obj[0]}: lowest point of its oriented bounding box (pose log) >= {T1_HANDOFF_LIFT_M} m above its step-0 "
                f"height at some step >= N, and still >= {T1_HANDOFF_END_M} m above it at the last logged step (budget end "
                "or early stop)")
    elif skill == "place in":
        kind, container = "inside", obj[1]
        text = (f"BDDL goal literal inside({obj[0]}, {obj[1]}) (evaluator pose-log goal_literals, same predicate as the "
                "evaluator's goal/Q) true at the last logged step (budget end or early stop)")
    else:
        raise ValueError(f"no Task-1 handoff criterion for skill {skill!r}")
    act = np.load(LOWDIM / f"episode_{episode}_action.npy", mmap_mode="r")
    hands = {}
    for hand, i in (("left", 14), ("right", 22)):
        c0, c1 = float(act[a, i]), float(act[min(e, len(act)) - 1, i])
        hands[hand] = {"cmd_at_segment_start": round(c0, 3), "cmd_at_segment_end": round(c1, 3),
                       "closes_in_segment": bool(c0 >= 0 > c1), "opens_in_segment": bool(c0 < 0 <= c1)}
    return {"contract": contract, "skill": skill, "demo_segment": [a, e], "handoff_step": N, "handoff_at_segment_start": N == a,
            "target_object": obj[0], "container": container, "kind": kind, "criterion": text,
            "thresholds": ({"lift_m": T1_HANDOFF_LIFT_M, "end_m": T1_HANDOFF_END_M} if kind == "lift" else {"inside_at_end": True}),
            "lift_measure": ("min over the 8 corners of the object's oriented bounding box (pose-log header native bbox x scale, "
                             "centred at the reset AABB centre) of world z, minus its step-0 value; tipping over cannot "
                             "raise it" if kind == "lift" else None),
            "budget_steps": T1_HANDOFF_BUDGET_FACTOR * (e - N) if budget is None else int(budget),
            "budget_rule": f"{T1_HANDOFF_BUDGET_FACTOR} x (demo segment end - N)" if budget is None else "CLI override (--budget)",
            "demo_gripper_commands": hands, "gripper_convention": "action dims 14 (left) / 22 (right): +1 open, -1 closed",
            "privileged_data_use": "evaluator-side pose log only; never a policy input"}


def t1_checkpoint_summary(path):
    """Checkpoint config fields the Task-1 handoff relies on (mmap load, no weights materialised); None without torch."""
    try:
        import torch
    except ImportError:
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["config"]
    m, d = cfg.get("model", {}), cfg.get("dataset", {})
    out = {"step": int(ck.get("step", -1)), "task_id": int(d.get("task_id", 13)), "cond_mode": ck.get("cond_mode", d.get("cond_mode")),
           "views": list(m.get("views") or ["observation.rgb.zed_link_camera_0"]), "proprio_mask": m.get("proprio_mask"),
           "proprio_lags": list(m.get("proprio_lags") or [0]), "action_repr": m.get("action_repr") or "absolute",
           "image_size": list(d.get("image_size") or []), "contract_targets_sha256": d.get("contract_targets_sha256")}
    del ck
    return out


def mapping_evidence_task1(episode, instance):
    table = episode_table()
    split = json.loads(SPLIT.read_text())
    meta = table[episode]
    insts = all_instances()
    test_ids = test_instance_ids()
    anchor = trash_pick_segment(episode)
    key = "trash_can_within_1.2m"
    pm1 = ("metadata id - 1 (off-by-one)", "metadata id + 1")
    conventions = {"metadata task_instance_id": lambda r: r["task_instance_id"],
                   pm1[0]: lambda r: r["task_instance_id"] - 1,
                   pm1[1]: lambda r: r["task_instance_id"] + 1,
                   "raw_episode_id % 10000 (download_gcs_rawdata task<50 formula)": lambda r: r["raw_episode_id"] % 10000}
    train = set(split["train_episode_indices"])
    population = {c: [0, 0] for c in conventions}
    population_train = {c: [0, 0] for c in conventions}
    this = {}
    for ep, r in sorted(table.items()):
        st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        qv = np.asarray(st[:trash_pick_segment(ep)[0], :3], np.float64)
        for name, fn in conventions.items():
            k = fn(r)
            if k not in insts:
                if ep == episode:
                    this[name] = {"instance": k, "instance_file_exists": False}
                continue
            res = docking_task1(qv, insts[k])
            for pop, use in ((population, True), (population_train, ep in train)):
                if use:
                    pop[name][1] += 1
                    pop[name][0] += int(res[key])
            if ep == episode:
                this[name] = {"instance": k, **res}
    qv = np.asarray(np.load(LOWDIM / f"episode_{episode}_state.npy", mmap_mode="r")[:anchor[0], :3], np.float64)
    passing = sorted(k for k, inst in insts.items() if docking_task1(qv, inst)[key])
    raw_expected = int(f"{TASK_ID:04d}{instance:03d}0")

    def rates(counts):
        return {c: {"pass": a, "n": n, "rate": round(a / n, 3) if n else None} for c, (a, n) in counts.items()}
    pop, pop_train = rates(population), rates(population_train)
    meta_rate = pop_train["metadata task_instance_id"]["rate"] or 0.0
    checks = {
        "episode_in_train_split": episode in split["train_episode_indices"] and episode not in split["validation_episode_indices"],
        "metadata_task_instance_id_equals_requested": meta["task_instance_id"] == instance,
        "raw_episode_id_matches_upstream_convention": meta["raw_episode_id"] == raw_expected,
        "train_instance_file_exists": instance_file(instance).is_file(),
        "not_a_public_or_hidden_test_instance": instance not in test_ids,
        "odometry_puts_trash_can_within_1.2m_for_metadata_instance": bool(this.get("metadata task_instance_id", {}).get(key)),
        "odometry_rejects_id_pm1_alternatives": all(not this.get(c, {}).get(key, False) for c in pm1),
        "population_train_metadata_rate_ge_0.8_and_id_pm1_le_0.25": bool(
            meta_rate >= 0.8 and all((pop_train[c]["rate"] if pop_train[c]["rate"] is not None else 1.0) <= 0.25 for c in pm1)),
    }
    return {**LABEL, "created_utc": utc(), "task_id": TASK_ID, "task": TASK, "episode": episode,
            "requested_train_instance": instance,
            "dataset_metadata": {"file": str(EPISODES_META.relative_to(ROOT)), **meta},
            "split": {"file": str(SPLIT.relative_to(ROOT)), "train": checks["episode_in_train_split"],
                      "closed_loop_test": split.get("closed_loop_test")},
            "upstream_id_convention": {
                "source": str(ID_CONVENTION_SRC.relative_to(ROOT)),
                "rule": "episode_id = f'{task_id:04d}{instance_id:03d}0'",
                "expected_raw_episode_id": raw_expected},
            "instance_file": str(instance_file(instance).relative_to(ROOT)),
            "instance_file_sha256": sha256_file(instance_file(instance)) if instance_file(instance).is_file() else None,
            "test_instance_ids_excluded": [test_ids[0], test_ids[-1]],
            "dev_report_note": "development/report are public_test indices 10-19 / 0-9 -> instance ids 311-320 / 301-310; never used",
            "odometry_docking_check": {
                "method": ("integrate the demo's recorded robot-frame base velocity (observation.state[0:3], 30 Hz) from a candidate "
                           "instance's presampled R1Pro start pose (robot_poses.R1Pro) up to the annotated 'pick up from(trash_can)' "
                           f"start frame; the correct instance must put the trash can (ashcan root link) within {T1_DOCK_M} m of the "
                           "robot, and the id +/- 1 alternatives must not"),
                "calibration": "lead 2026-09-26, outputs/validation/task1_docking_calibration_20260926/",
                "anchor": "pick up from(trash_can) start", "anchor_frame": anchor[0], "radius_m": T1_DOCK_M,
                "this_episode": this,
                "train_instances_passing_for_this_episode": {"count": len(passing), "of": len(insts), "ids": passing},
                "population_all_200_demos": pop, "population_train_demos": pop_train},
            "checks": checks, "established": all(checks.values()),
            "demo_outcome": demo_rewards(episode), "demo_picks": demo_hands_task1(episode)}


# --- Task 0 (turning_on_radio) ------------------------------------------------------------------------------------
def radio_pick_segment(episode):
    """Task 0 anchor: the demo's 'pick up from(radio, coffee_table)' segment (skill 2 of 4 in every demo)."""
    seg = next(s for s in demo_segments(episode) if s["skill"] == "pick up from"
               and s["objects"] and s["objects"][0][0].startswith("radio"))
    return int(seg["start"]), int(seg["end"])


def docking_task0(qvel_prefix, inst):
    """Where the demo's own base odometry puts the robot at the radio pick start, from this instance's start pose."""
    p = integrate_odometry(qvel_prefix, inst["start"])[-1]
    d = inst["radio"] - p[:2]
    dist = float(np.linalg.norm(d))
    bearing = math.degrees((math.atan2(d[1], d[0]) - p[2] + math.pi) % (2 * math.pi) - math.pi)
    return {f"radio_within_{T0_DOCK_M}m": bool(dist <= T0_DOCK_M), "distance_to_radio_m": round(dist, 3),
            "radio_bearing_in_robot_frame_deg": round(bearing, 1)}


def demo_hands_task0(episode):
    """Per segment: which gripper closes / opens (demo command sign, +1 open / -1 closed) -> holding and pressing hand."""
    act = np.load(LOWDIM / f"episode_{episode}_action.npy", mmap_mode="r")
    out = []
    for s in demo_segments(episode):
        a, e = int(s["start"]), min(int(s["end"]), len(act)) - 1
        out.append({"skill": s["skill"], "object": s["objects"][0][0] if s["objects"] else None, "segment": [int(s["start"]), int(s["end"])],
                    "closes": [h for h, i in (("left", 14), ("right", 22)) if act[a, i] >= 0 > act[e, i]],
                    "opens": [h for h, i in (("left", 14), ("right", 22)) if act[a, i] < 0 <= act[e, i]],
                    "closed_at_start": [h for h, i in (("left", 14), ("right", 22)) if act[a, i] < 0]})
    return out


def t0_handoff_spec(episode, handoff_step, contract=None, skill=None, budget=None):
    """Task 0 handoff: like t1_handoff_spec (the demo segment containing N must BE the contract), criteria pre-registered
    2026-09-27: pick up from -> radio lowest point lift (T1 thresholds 0.10 / 0.05 m, judged at the last logged step =
    budget end, no early stop by default); press -> the radio's ToggledOn (BDDL goal literal toggled_on, evaluator pose
    log) becomes true at some step >= N within the budget (the evaluator then ends the episode: goal reached)."""
    from harness.vtt_monitor import precondition_key
    N = int(handoff_step)
    seg = next((s for s in demo_segments(episode) if int(s["start"]) <= N < int(s["end"])), None)
    if seg is None or not seg["objects"]:
        raise ValueError(f"handoff step {N} is not inside an annotated skill segment of episode {episode}")
    a, e = int(seg["start"]), int(seg["end"])
    key = precondition_key(seg["skill"], seg["objects"])
    contract = key if contract is None else str(contract)
    if contract != key:
        raise ValueError(f"contract {contract!r} differs from the demo segment [{a}, {e}) at step {N}: {key!r}")
    skill = seg["skill"] if skill is None else str(skill)
    if skill != contract.partition("(")[0]:
        raise ValueError(f"skill {skill!r} differs from the contract skill {contract.partition('(')[0]!r}")
    obj = seg["objects"][0]
    if skill == "pick up from":
        kind = "lift"
        text = (f"{obj[0]}: lowest point of its oriented bounding box (pose log) >= {T1_HANDOFF_LIFT_M} m above its step-0 "
                f"height at some step >= N, and still >= {T1_HANDOFF_END_M} m above it at the last logged step (budget end)")
        thresholds = {"lift_m": T1_HANDOFF_LIFT_M, "end_m": T1_HANDOFF_END_M}
    elif skill == "press":
        kind = "toggled_on"
        text = (f"ToggledOn({obj[0]}) (BDDL goal literal toggled_on, logged per step by the evaluator pose log; the evaluator's "
                "own goal predicate) becomes true at some step >= N within the budget; the first such step is reported. "
                "The evaluator ends the episode at that step (goal reached), so 'true at the end' is the same event")
        thresholds = {"toggled_on_within_budget": True}
    else:
        raise ValueError(f"no Task-0 handoff criterion for skill {skill!r} (registered: pick up from, press)")
    act = np.load(LOWDIM / f"episode_{episode}_action.npy", mmap_mode="r")
    hands = {}
    for hand, i in (("left", 14), ("right", 22)):
        c0, c1 = float(act[a, i]), float(act[min(e, len(act)) - 1, i])
        hands[hand] = {"cmd_at_segment_start": round(c0, 3), "cmd_at_segment_end": round(c1, 3),
                       "closes_in_segment": bool(c0 >= 0 > c1), "opens_in_segment": bool(c0 < 0 <= c1)}
    rw = demo_rewards(episode)
    first_reward = rw["reward_frames"][0] if rw["reward_frames"] else None
    return {"contract": contract, "skill": skill, "demo_segment": [a, e], "handoff_step": N, "handoff_at_segment_start": N == a,
            "target_object": obj[0], "container": obj[1] if len(obj) > 1 else None, "kind": kind, "criterion": text,
            "thresholds": thresholds,
            "lift_measure": ("min over the 8 corners of the object's oriented bounding box (pose-log header native bbox x scale, "
                             "centred at the reset AABB centre) of world z, minus its step-0 value; tipping over cannot "
                             "raise it" if kind == "lift" else None),
            "budget_steps": T1_HANDOFF_BUDGET_FACTOR * (e - N) if budget is None else int(budget),
            "budget_rule": f"{T1_HANDOFF_BUDGET_FACTOR} x (demo segment end - N)" if budget is None else "CLI override (--budget)",
            "demo_gripper_commands": hands, "gripper_convention": "action dims 14 (left) / 22 (right): +1 open, -1 closed",
            "demo_first_reward_frame": first_reward,
            "demo_toggle_in_segment": bool(first_reward is not None and a <= first_reward < e),
            "privileged_data_use": "evaluator-side pose log only; never a policy input"}


def handoff_spec(episode, handoff_step, contract=None, skill=None, budget=None):
    return (t0_handoff_spec if TASK_ID == 0 else t1_handoff_spec)(episode, handoff_step, contract, skill, budget)


def mapping_evidence_task0(episode, instance):
    """Task 0 demo -> train instance evidence (same checks as Task 1, radio instead of the trash can)."""
    table = episode_table()
    split = json.loads(SPLIT.read_text())
    meta = table[episode]
    insts = all_instances()
    test_ids = test_instance_ids()
    anchor = radio_pick_segment(episode)
    key = f"radio_within_{T0_DOCK_M}m"
    pm1 = ("metadata id - 1 (off-by-one)", "metadata id + 1")
    conventions = {"metadata task_instance_id": lambda r: r["task_instance_id"],
                   pm1[0]: lambda r: r["task_instance_id"] - 1,
                   pm1[1]: lambda r: r["task_instance_id"] + 1,
                   "raw_episode_id % 10000 (download_gcs_rawdata task<50 formula)": lambda r: r["raw_episode_id"] % 10000}
    train = set(split["train_episode_indices"])
    population = {c: [0, 0] for c in conventions}
    population_train = {c: [0, 0] for c in conventions}
    this = {}
    for ep, r in sorted(table.items()):
        st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        qv = np.asarray(st[:radio_pick_segment(ep)[0], :3], np.float64)
        for name, fn in conventions.items():
            k = fn(r)
            if k not in insts:
                if ep == episode:
                    this[name] = {"instance": k, "instance_file_exists": False}
                continue
            res = docking_task0(qv, insts[k])
            for pop, use in ((population, True), (population_train, ep in train)):
                if use:
                    pop[name][1] += 1
                    pop[name][0] += int(res[key])
            if ep == episode:
                this[name] = {"instance": k, **res}
    qv = np.asarray(np.load(LOWDIM / f"episode_{episode}_state.npy", mmap_mode="r")[:anchor[0], :3], np.float64)
    passing = sorted(k for k, inst in insts.items() if docking_task0(qv, inst)[key])
    raw_expected = int(f"{TASK_ID:04d}{instance:03d}0")

    def rates(counts):
        return {c: {"pass": a, "n": n, "rate": round(a / n, 3) if n else None} for c, (a, n) in counts.items()}
    pop, pop_train = rates(population), rates(population_train)
    meta_rate = pop_train["metadata task_instance_id"]["rate"] or 0.0
    checks = {
        "episode_in_train_split": episode in split["train_episode_indices"] and episode not in split["validation_episode_indices"],
        "metadata_task_instance_id_equals_requested": meta["task_instance_id"] == instance,
        "raw_episode_id_matches_upstream_convention": meta["raw_episode_id"] == raw_expected,
        "train_instance_file_exists": instance_file(instance).is_file(),
        "not_a_public_or_hidden_test_instance": instance not in test_ids,
        f"odometry_puts_radio_within_{T0_DOCK_M}m_for_metadata_instance": bool(this.get("metadata task_instance_id", {}).get(key)),
        "odometry_rejects_id_pm1_alternatives": all(not this.get(c, {}).get(key, False) for c in pm1),
        "population_train_metadata_rate_ge_0.8_and_id_pm1_le_0.25": bool(
            meta_rate >= 0.8 and all((pop_train[c]["rate"] if pop_train[c]["rate"] is not None else 1.0) <= 0.25 for c in pm1)),
    }
    return {**LABEL, "created_utc": utc(), "task_id": TASK_ID, "task": TASK, "episode": episode,
            "requested_train_instance": instance,
            "dataset_metadata": {"file": str(EPISODES_META.relative_to(ROOT)), **meta},
            "split": {"file": str(SPLIT.relative_to(ROOT)), "train": checks["episode_in_train_split"],
                      "closed_loop_test": split.get("closed_loop_test")},
            "upstream_id_convention": {"source": str(ID_CONVENTION_SRC.relative_to(ROOT)),
                                       "rule": "episode_id = f'{task_id:04d}{instance_id:03d}0'",
                                       "expected_raw_episode_id": raw_expected},
            "instance_file": str(instance_file(instance).relative_to(ROOT)),
            "instance_file_sha256": sha256_file(instance_file(instance)) if instance_file(instance).is_file() else None,
            "test_instance_ids_excluded": [test_ids[0], test_ids[-1]],
            "dev_report_note": "development/report are public_test indices 10-19 / 0-9 -> instance ids 311-320 / 301-310; never used",
            "odometry_docking_check": {
                "method": ("integrate the demo's recorded robot-frame base velocity (observation.state[0:3], 30 Hz) from a candidate "
                           "instance's presampled R1Pro start pose (robot_poses.R1Pro) up to the annotated 'pick up from(radio)' "
                           f"start frame; the correct instance must put the radio ({T0_RADIO} root link) within {T0_DOCK_M} m of the "
                           "robot, and the id +/- 1 alternatives must not"),
                "calibration": "TRAIN 2026-09-27 (t0-admission-handoff): metadata 0.883, id-1 0.222, id+1 0.197 within 1.2 m",
                "anchor": "pick up from(radio) start", "anchor_frame": anchor[0], "radius_m": T0_DOCK_M,
                "this_episode": this,
                "train_instances_passing_for_this_episode": {"count": len(passing), "of": len(insts), "ids": passing},
                "population_all_200_demos": pop, "population_train_demos": pop_train},
            "checks": checks, "established": all(checks.values()),
            "demo_outcome": demo_rewards(episode), "demo_hands": demo_hands_task0(episode)}


def demo_reference_task1(episode, out_path):
    """Demo camera poses in the robot base frame (observation.robot2cam_pose.*, per frame) for the cam_rel_poses check;
    extracted here (lilawam has pyarrow) so the behavior6 analysis needs no parquet reader."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    for f in sorted((DEMOS / "data" / TP.CHUNK).glob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", *DEMO_CAM_POSE_COLS.values()])
        t = t.filter(pc.equal(t.column("episode_index"), episode))
        if t.num_rows:
            order = np.argsort(np.array(t.column("frame_index").to_pylist()))
            poses = np.stack([np.array(t.column(c).to_pylist(), np.float32)[order] for c in DEMO_CAM_POSE_COLS.values()], 1)
            np.savez(out_path, robot2cam_pose=poses, cameras=np.array(list(DEMO_CAM_POSE_COLS)),
                     columns=np.array(list(DEMO_CAM_POSE_COLS.values())), source=str(f.relative_to(ROOT)))
            return poses.shape
    raise KeyError(f"episode {episode} not found in demo parquet files")


def cmd_mapping(args):
    ev = mapping_evidence(args.episode, args.instance)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, ev)
    if TASK_ID != 13:
        demo_reference_task1(args.episode, out.parent / "demo_reference.npz")
    print(json.dumps({"established": ev["established"], "checks": ev["checks"],
                      "population": ev["odometry_docking_check"]["population_all_200_demos"]}, indent=1))
    return 0 if ev["established"] else 3


# ---------------------------------------------------------------------------------------------------------------
# replay websocket server (protocol framing identical to harness/policy_server.py / eval network_utils)
# ---------------------------------------------------------------------------------------------------------------
def pack_data(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _msgpack():
    import msgpack
    return (functools.partial(msgpack.Packer, default=pack_data),
            functools.partial(msgpack.unpackb, object_hook=unpack_data, strict_map_key=False))


class GpuUtilization(threading.Thread):
    """1 Hz nvidia-smi utilization sampler (NVML ignores CUDA_VISIBLE_DEVICES; this process never touches CUDA)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.ema, self.last = None, None

    def run(self):
        while True:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.split()
                u = float(out[0])
                self.last = u
                self.ema = u if self.ema is None else 0.3 * u + 0.7 * self.ema
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            time.sleep(1.0)


class Throttle:
    """Wall-clock delay before answering an action (the evaluator is synchronous, so physics is unaffected).
    adaptive: keep GPU utilization EMA below `high` (85 % project ceiling) by stretching the sim duty cycle."""

    def __init__(self, mode="adaptive", high=75.0, low=60.0, max_ms=150.0, fixed_ms=0.0):
        self.mode, self.high, self.low, self.max_ms = mode, high, low, max_ms
        self.ms = float(fixed_ms)
        self.gpu = GpuUtilization() if mode == "adaptive" else None
        self._t = 0.0
        if self.gpu:
            self.gpu.start()

    def delay_s(self):
        if self.gpu is not None and self.gpu.ema is not None and time.monotonic() - self._t >= 1.0:
            self._t = time.monotonic()
            if self.gpu.ema > self.high:
                self.ms = min(self.ms + 5.0, self.max_ms)
            elif self.gpu.ema < self.low:
                self.ms = max(self.ms - 2.0, 0.0)
        return self.ms / 1000.0

    def state(self):
        return {"throttle_ms": round(self.ms, 1), "gpu_util_ema": None if self.gpu is None or self.gpu.ema is None
                else round(self.gpu.ema, 1)}


def keyframes_for(episode, n):
    """Even frames only (the demo frame cache has stride 2): segment starts, open-lid quartiles, last step."""
    if TASK_ID != 13:  # Task 1: start and end of every skill segment, first and last step
        ks = {0, n - 1}
        for s in demo_segments(episode):
            ks.update((int(s["start"]), int(s["end"])))
        return sorted({k - k % 2 for k in ks if 0 <= k < n})
    ks = {0, n - 1}
    for s in demo_segments(episode):
        ks.add(int(s["start"]))
    a, b = openlid_segment(episode)
    ks.update(int(a + q * (b - a)) for q in (0.25, 0.5, 0.75))
    ks.add(b - 1)
    return sorted({k - k % 2 for k in ks if 0 <= k < n})


def camera_roles():
    """(role, sensor) in evaluator cam_rel_poses order: r1pro.yaml eval.camera_sensor_names, iterated in file order by
    Evaluator._preprocess_obs (every configured sensor exists on R1Pro)."""
    lines = (EVAL_UTILS.parents[1] / "r1pro.yaml").read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.strip() == "camera_sensor_names:")
    indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
    roles = []
    for line in lines[i + 1:]:
        if not line.strip() or len(line) - len(line.lstrip()) != indent:
            break
        role, sensor = (x.strip() for x in line.split(":", 1))
        roles.append((role, sensor))
    return roles


def native_capture_steps(episode, n, mode="default"):
    """Lead request (paired sim/demo frames for the image-domain check): 20 steps at 10-step spacing from the episode
    start, and every 10 steps from ~60 steps before the demo 'open lid' start to its end. The first step is rounded
    down to an even frame so every capture also pairs with the stride-2 training frame cache."""
    a, b = openlid_segment(episode)
    s0 = max(0, a - 60)
    s0 -= s0 % 2
    if mode == "segment":  # calibration replays (lead): every 10 steps within the open-lid segment +/-60
        return sorted(k for k in range(s0, b + 61, 10) if 0 <= k < n)
    return sorted(k for k in set(range(0, 200, 10)) | set(range(s0, b + 1, 10)) if 0 <= k < n)


class ReplayPolicy:
    def __init__(self, episode, out_dir, steps=None, seed=0, tail_hold=True, head_stride=30, keyframes=None,
                 throttle=None, native_dir=None, native_steps=None, native_mode="default", perception_every=0):
        self.episode_id = episode
        self.action_path = LOWDIM / f"episode_{episode}_action.npy"
        actions = np.load(self.action_path).astype(np.float32)
        self.n_demo = len(actions)
        self.n = self.n_demo if steps is None else min(int(steps), self.n_demo)
        self.actions = actions[:self.n]
        self.hold = self.actions[-1].copy()
        self.hold[:3] = 0.0  # stop the base; keep the last arm/trunk/gripper targets
        self.tail_hold = tail_hold
        self.seed = int(seed)
        self.out = Path(out_dir)
        (self.out / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.out / "head_1hz").mkdir(parents=True, exist_ok=True)
        self.head_stride = int(head_stride)
        self.keyframes = set(keyframes if keyframes is not None else keyframes_for(episode, self.n))
        self.native_dir = Path(native_dir) if native_dir else None
        # perception capture (Task 1): every N steps RGB (all cameras) + head depth, cam_rel_poses every step
        self.perception_every = int(perception_every or 0)
        if self.perception_every:
            self.native_steps = set()
        else:
            self.native_steps = set(native_steps if native_steps is not None else native_capture_steps(episode, self.n, native_mode))
        self.native_meta = None
        if self.native_dir:
            for cam in LIVE_SUFFIX:
                (self.native_dir / cam).mkdir(parents=True, exist_ok=True)
            if self.perception_every:
                for cam in DEPTH_SUFFIX:
                    (self.native_dir / f"{cam}_depth").mkdir(parents=True, exist_ok=True)
        self.throttle = throttle or Throttle("off")
        self.sha = sha256_file(self.action_path)
        self.trace = (self.out / "trace.jsonl").open("a")
        self.episode = -1
        self.step = 0
        cap = self.n + 20000
        self.proprio = np.full((cap, 61), np.nan, np.float32)
        self.sent = np.full((cap, 23), np.nan, np.float32)
        self.t_recv = np.full(cap, np.nan)
        self.cam_rel = np.full((cap, 21), np.nan, np.float32) if self.perception_every else None
        self.delay = np.full(cap, np.nan) if self.perception_every else None
        self._pool, self._pending, self._capture_errors, self._captures, self._alpha_kept = None, [], [], 0, []
        if self.perception_every:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="capture")  # PNG encoding off the step path
        self._flushed_steps = 0
        self.emit({"event": "server_start", **LABEL, "episode_replayed": episode, "action_sha256": self.sha,
                   "replay_steps": self.n, "keyframes": sorted(self.keyframes)})

    def metadata(self):
        return {"policy": "demo_action_replay_diagnostic", **LABEL, "episode_replayed": self.episode_id,
                "action_file": str(self.action_path.relative_to(ROOT)), "action_sha256": self.sha,
                "replay_steps": self.n, "tail_action": ("hold: last demo arm/trunk/gripper targets, zero base velocity"
                                                        if self.tail_hold else "none (error past end)"),
                "seed": self.seed, "rng_reset_behavior": "deterministic open-loop replay; no RNG used",
                "action_dim": 23, "state_dim": 61}

    def emit(self, rec):
        rec["time"] = time.time()
        self.trace.write(json.dumps(rec) + "\n")
        self.trace.flush()

    def flush(self, final=False):
        if self.step == 0:
            return
        np.save(self.out / "proprio.npy.tmp.npy", self.proprio[:self.step])
        os.replace(self.out / "proprio.npy.tmp.npy", self.out / "proprio.npy")
        np.save(self.out / "actions_sent.npy.tmp.npy", self.sent[:self.step])
        os.replace(self.out / "actions_sent.npy.tmp.npy", self.out / "actions_sent.npy")
        np.save(self.out / "recv_time.npy.tmp.npy", self.t_recv[:self.step])
        os.replace(self.out / "recv_time.npy.tmp.npy", self.out / "recv_time.npy")
        # demo frame paired with each step: observation t is the state before action t = demo frame t; -1 = tail hold
        frame_index = np.where(np.arange(self.step) < self.n, np.arange(self.step), -1).astype(np.int64)
        targets = [self.out] + ([self.native_dir] if self.native_dir else [])
        for d in targets:
            np.save(d / "frame_index.npy.tmp.npy", frame_index)
            os.replace(d / "frame_index.npy.tmp.npy", d / "frame_index.npy")
        if self.native_dir:
            np.save(self.native_dir / "proprio.npy.tmp.npy", self.proprio[:self.step])
            os.replace(self.native_dir / "proprio.npy.tmp.npy", self.native_dir / "proprio.npy")
            if self.cam_rel is not None:
                np.save(self.native_dir / "cam_rel_poses.npy.tmp.npy", self.cam_rel[:self.step].reshape(-1, 3, 7))
                os.replace(self.native_dir / "cam_rel_poses.npy.tmp.npy", self.native_dir / "cam_rel_poses.npy")
        if self.delay is not None:
            np.save(self.out / "throttle_delay_s.npy.tmp.npy", self.delay[:self.step])
            os.replace(self.out / "throttle_delay_s.npy.tmp.npy", self.out / "throttle_delay_s.npy")
        if self._capture_errors:
            errors, self._capture_errors = self._capture_errors, []
            for err in errors:
                self.emit({"event": "capture_error", **err})
        after = ({"wam_steps_served": max(0, self.step - self.n)} if isinstance(self, HandoffPolicy)
                 else {"hold_steps_served": max(0, self.step - self.n)})
        write_json(self.out / "replay_status.json", {"episode": self.episode, "steps_served": self.step,
                                                      "demo_steps_served": min(self.step, self.n), **after,
                                                      "final": final, "updated_utc": utc(), **self.throttle.state()})
        self._flushed_steps = self.step

    def reset(self):
        self.emit({"event": "reset", "previous_episode": self.episode, "previous_steps": self.step})
        if self.step > 0:  # a second scored rollout is not expected; keep the first one's logs
            self.flush(final=True)
            os.replace(self.out / "proprio.npy", self.out / f"proprio_episode{self.episode}.npy")
            os.replace(self.out / "actions_sent.npy", self.out / f"actions_sent_episode{self.episode}.npy")
        self.episode += 1
        self.step = 0

    @staticmethod
    def _find(obs, suffix):
        for k in obs:
            if isinstance(k, str) and k.endswith(suffix):
                return obs[k]
        raise KeyError(f"no observation key ending with {suffix!r}; keys={sorted(map(str, obs))}")

    def _save_image(self, rgb, path, quality=None):
        from PIL import Image
        rgb = np.asarray(rgb)[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        img = Image.fromarray(np.ascontiguousarray(rgb))
        img.save(path, quality=quality) if quality else img.save(path)

    def _save_native(self, arr, cam, t):
        """Evidence only (never a policy input): the observation array exactly as received, lossless."""
        from PIL import Image
        arr = np.asarray(arr)
        if self.native_meta is None:
            self.native_meta = {**LABEL, "purpose": ("paired sim-native vs demo frames at the same demo frame index for the "
                                                     "image-domain check (native 224 render vs 720->INTER_AREA->224->JPEG q90 "
                                                     "training frames); evidence only, never fed to any policy"),
                                "camera_keys": {c: f"robot_r1::robot_r1{sfx[:-5]}::rgb" for c, sfx in LIVE_SUFFIX.items()},
                                "received_shape": list(arr.shape), "received_dtype": str(arr.dtype),
                                "format": ("PNG, all received channels (RGBA if 4), no resize/conversion" if arr.dtype == np.uint8
                                           and arr.ndim == 3 and arr.shape[2] in (3, 4) else "npy (non-uint8 array)"),
                                "capture_steps": sorted(self.native_steps),
                                "step_semantics": "step t = observation before action t = demo frame t (frame_index.npy; -1 = tail hold)",
                                "demo_pairs": {"episode": self.episode_id,
                                               "training_frame_cache": {c: f"data/frame_cache/{CAMS[c]}/episode_{self.episode_id}.jpgpack (stride 2)"
                                                                        for c in CAMS},
                                               "demo_video": "data/behavior-1k/2026-challenge-demos/videos/<camera>/chunk-013 (see frame cache meta.json)"},
                                "proprio": "proprio.npy: received 61-D robot_r1::proprio at every step (float32)"}
            write_json(self.native_dir / "capture_meta.json", self.native_meta)
        if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] in (3, 4):
            Image.fromarray(np.ascontiguousarray(arr), "RGBA" if arr.shape[2] == 4 else "RGB").save(
                self.native_dir / cam / f"step_{t:06d}.png")
        else:
            np.save(self.native_dir / cam / f"step_{t:06d}.npy", arr)

    def _write_perception_meta(self, jobs):
        roles = camera_roles()
        self.native_meta = {
            **LABEL, "task": TASK, "episode_replayed": self.episode_id,
            "purpose": ("sim-native sensor data aligned with the evaluator's ground-truth pose log, for perception development; "
                        "evidence only, never fed to any policy"),
            "capture_rule": (f"every {self.perception_every} environment steps (t % {self.perception_every} == 0), demo and "
                             "tail-hold steps alike; proprio and cam_rel_poses at every step"),
            "step_semantics": "step t = observation before action t = demo frame t (frame_index.npy; -1 = tail hold)",
            "received": {f"{kind}:{cam}": {"shape": list(np.shape(a)), "dtype": str(np.asarray(a).dtype)} for kind, cam, a in jobs},
            "files": {
                "head|lwrist|rwrist/step_XXXXXX.png": ("RGB exactly as rendered, lossless PNG; the alpha channel is dropped only "
                                                       "when it is constant 255 (frames keeping RGBA are listed in alpha_kept)"),
                "head_depth/step_XXXXXX.npy": ("head depth_linear = distance to the image plane (z-depth) in metres, stored "
                                               "float16 (relative precision ~5e-4, i.e. ~2 mm at 4 m; inf kept as rendered)"),
                "proprio.npy": "(T, 61) float32 robot_r1::proprio, every step (configs/behavior_environment.json state_slices)",
                "cam_rel_poses.npy": ("(T, 3, 7) float32 robot_r1::cam_rel_poses, every step: per camera [x, y, z, qx, qy, qz, qw] "
                                      "in the robot base frame (base footprint link), camera order = cam_rel_poses_order"),
                "frame_index.npy": "(T,) demo frame paired with each step; -1 = tail hold"},
            "camera_keys": {c: f"robot_r1::robot_r1{sfx[:-5]}::rgb" for c, sfx in LIVE_SUFFIX.items()}
                           | {f"{c}_depth": f"robot_r1::robot_r1{sfx.rsplit('::', 1)[0]}::depth_linear" for c, sfx in DEPTH_SUFFIX.items()},
            "cam_rel_poses_order": [r for r, _ in roles], "cam_rel_poses_sensors": [x for _, x in roles],
            "camera_convention": ("pose axes follow the USD camera convention (-Z forward, +Y up, +X right); OpenCV camera "
                                  "rotation = R_cam @ diag(1, -1, -1) (180 deg about X, as in omnigibson/eval/utils/obs_utils.depth_to_pcd)"),
            "intrinsics": {"official_eval_utils_CAMERA_INTRINSICS_R1Pro": OFFICIAL_INTRINSICS,
                           "rendered": "eval/object_poses/<task>_<instance>_0.jsonl header 'cameras' (sensor.intrinsic_matrix)"},
            "ground_truth": ("eval/object_poses/<task>_<instance>_<rollout>.jsonl (evaluator --log-object-poses): per-step world "
                             "poses of every task-scope object and the robot base + BDDL goal literals; row step t aligns "
                             "with step t here")}
        write_json(self.native_dir / "capture_meta.json", self.native_meta)

    def _write_capture(self, kind, cam, arr, t):
        """Capture-thread body; errors are collected and emitted by the main thread at the next flush."""
        try:
            from PIL import Image
            if kind == "depth":
                d = np.asarray(arr, np.float32)
                if d.ndim == 3 and d.shape[-1] == 1:
                    d = d[..., 0]
                np.save(self.native_dir / f"{cam}_depth" / f"step_{t:06d}.npy", d.astype(np.float16))
            else:
                a = np.ascontiguousarray(arr)
                if a.ndim == 3 and a.shape[2] == 4:
                    if (a[..., 3] == 255).all():
                        a = np.ascontiguousarray(a[..., :3])
                    else:
                        self._alpha_kept.append([cam, t])
                Image.fromarray(a).save(self.native_dir / cam / f"step_{t:06d}.png")  # uint8 HxWx3 -> RGB, HxWx4 -> RGBA
        except Exception as e:  # evidence capture must never stop the replay
            self._capture_errors.append({"step": t, "kind": kind, "camera": cam, "error": repr(e)})

    def _capture_perception(self, obs, t):
        jobs = []
        for kind, table in (("rgb", LIVE_SUFFIX), ("depth", DEPTH_SUFFIX)):
            for cam, suffix in table.items():
                try:
                    jobs.append((kind, cam, self._find(obs, suffix)))
                except KeyError as e:
                    self.emit({"event": "native_missing", "step": t, "camera": cam, "kind": kind, "error": str(e)})
        if self.native_meta is None:
            self._write_perception_meta(jobs)
        self._pending = [f for f in self._pending if not f.done()]
        while len(self._pending) > 16:  # bound memory: a few captures in flight at most
            self._pending.pop(0).result()
        for kind, cam, arr in jobs:
            self._pending.append(self._pool.submit(self._write_capture, kind, cam, arr, t))
        self._captures += 1

    def _drain_captures(self):
        if self._pool is not None:
            for f in self._pending:
                f.result()
            self._pending = []
            if self.native_meta is not None:
                self.native_meta.update(captures=self._captures, alpha_kept=self._alpha_kept,
                                        capture_errors_total=len(self._capture_errors))
                write_json(self.native_dir / "capture_meta.json", self.native_meta)

    def note_delay(self, delay_s):
        if self.delay is not None and self.step > 0:
            self.delay[self.step - 1] = delay_s

    def act(self, obs):
        t = self.step
        if t >= len(self.proprio):
            raise RuntimeError("replay buffer exhausted")
        self.proprio[t] = np.asarray(self._find(obs, "::proprio"), np.float32).reshape(-1)[:61]
        self.t_recv[t] = time.time()
        if self.cam_rel is not None:
            try:
                self.cam_rel[t] = np.asarray(self._find(obs, "::cam_rel_poses"), np.float32).reshape(-1)[:21]
            except KeyError as e:
                if t == 0:
                    self.emit({"event": "cam_rel_poses_missing", "error": str(e)})
        if self.native_dir and self.perception_every and t % self.perception_every == 0:
            self._capture_perception(obs, t)
        if t in self.keyframes:
            for cam, suffix in LIVE_SUFFIX.items():
                try:
                    self._save_image(self._find(obs, suffix), self.out / "snapshots" / f"sim_step{t:05d}_{cam}.png")
                except KeyError as e:
                    self.emit({"event": "snapshot_missing", "step": t, "camera": cam, "error": str(e)})
        if self.native_dir and t in self.native_steps:
            for cam, suffix in LIVE_SUFFIX.items():
                try:
                    self._save_native(self._find(obs, suffix), cam, t)
                except KeyError as e:
                    self.emit({"event": "native_missing", "step": t, "camera": cam, "error": str(e)})
        if self.head_stride and t % self.head_stride == 0:
            self._save_image(self._find(obs, LIVE_SUFFIX["head"]), self.out / "head_1hz" / f"sim_step{t:05d}.jpg", quality=90)
        a = self.next_action(t, obs)
        self.sent[t] = a
        self.step += 1
        if self.step % 300 == 0:
            self.flush()
            self.emit({"event": "progress", "episode": self.episode, "step": self.step, **self.throttle.state()})
        return np.array(a, dtype=np.float32, copy=True)

    def next_action(self, t, obs):
        if t < self.n:
            return self.actions[t]
        if self.tail_hold:
            return self.hold
        raise RuntimeError(f"replay exhausted at step {t}")

    def close(self):
        self._drain_captures()
        self.flush(final=True)
        self.emit({"event": "connection_closed", "episode": self.episode, "steps": self.step})


class HandoffPolicy(ReplayPolicy):
    """Demo actions for t < N, then the WAM on ONE contract (Task 13: 'open lid(car)', lead spec 2026-09-25; Task 1: a
    manipulation contract of the multi-view WAM, lead spec 2026-09-26). The WAM's own preprocessing + proprio buffer are
    fed at EVERY step (prefix included), exactly like deployment act(). The WAM only ever sees the proprio vector and the
    RGB views its checkpoint declares (never depth, cam_rel_poses, task id or evaluator logs)."""

    def __init__(self, episode, out_dir, handoff_step, wam_checkpoint, *, wam_device="cuda", wam_samples=8, wam_num_steps=10,
                 cuda_fraction=0.4, seed=0, native_every=15, budget=3084, contract="open lid(car)", skill="open lid",
                 wam_factory=None, fixed_noise_seed=None, exec_horizon=32, base_from_demo=False, base_control=None, **kw):
        n = int(handoff_step)
        native = list(range(n, n + int(budget) + 1, int(native_every)))
        super().__init__(episode, out_dir, steps=n, seed=seed, tail_hold=False, keyframes=[0, n - n % 2],
                         native_steps=native, **kw)
        import torch
        self.torch = torch
        self.N = n
        # base_from_demo (Task 1 diagnostic, 2026-09-27): base dims 0:3 replay the demo's base actions after N (zero past
        # the demo end); trunk/arms/grippers stay the WAM's. Isolates the upper-body executor from the WAM's collapsed
        # base output; never a deployable policy.
        self.base_from_demo = bool(base_from_demo)
        # base_control (orchestrate.base_control of the WAM): None keeps the WamPolicy default (Task 1 / 13 unchanged);
        # Task 0 defaults to hold_during_manipulation (base dims of every manipulation chunk zeroed, progress.md 2026-09-27).
        if base_control is None and TASK_ID == 0:
            base_control = T0_BASE_CONTROL
        self.base_control = base_control
        factory = wam_factory
        if factory is None:
            from harness.policy_server import WamPolicy as factory
        self.wam_kwargs = {"skill": "move to", "exec_horizon": int(exec_horizon), "num_steps": int(wam_num_steps), "device": wam_device,
                           "cuda_fraction": float(cuda_fraction), "seed": int(seed), "log_dir": str(Path(out_dir) / "wam"),
                           "orchestrate": {"vlm": "none", "frozen": True, "train_targets": str(CONTRACT_TARGETS.relative_to(ROOT))},
                           "samples": int(wam_samples)}
        if base_control is not None:
            self.wam_kwargs["orchestrate"]["base_control"] = str(base_control)
        self.wam = factory(str(wam_checkpoint), **self.wam_kwargs)
        vec = np.asarray(self.wam.bank.select(contract, skill).vector, np.float32)
        ref = np.asarray(json.loads(CONTRACT_TARGETS.read_text())["contracts"][contract]["vector"], np.float32)
        if not np.array_equal(vec, ref):
            raise ValueError("bank condition differs from the training contract vector")
        self.cond = torch.from_numpy(vec)[None]
        self.cond_info = {"contract": contract, "source": "frozen ConditionBank.select (train-contract-vtt-v1)",
                          "sha256": hashlib.sha256(vec.tobytes()).hexdigest(), "dim": int(vec.shape[0])}
        # Views the checkpoint declares (harness.policy_server.PolicyModel): head first; a head-only model keeps the
        # original single-key policy observation and the original chunk path below (Task 13 unchanged).
        model = self.wam.model
        self.exec_horizon, self.skill = int(exec_horizon), skill
        self.multiview = bool(getattr(model, "multiview", False))
        self.is_v3 = bool(getattr(model, "is_v3", False))
        self._v3_last_step = None
        self._v3_start_step = None
        self.view_suffixes = list(getattr(model, "view_suffixes", None) or [LIVE_SUFFIX["head"]])
        self.queue, self.chunk_id, self.handoff_record = [], 0, None
        self.fixed_noise_seed = None if fixed_noise_seed is None else int(fixed_noise_seed)
        self.noise_rule = ("policy.gen.manual_seed(%d) immediately before every _sample() call (same initial noise per chunk)"
                           % self.fixed_noise_seed if self.fixed_noise_seed is not None
                           else "generator seeded once at construction; fresh noise every chunk")
        self.demo_actions_full = np.load(self.action_path).astype(np.float32)
        self.chunk_log = (self.out / "chunks.jsonl").open("a")
        extra = {}
        if self.multiview:
            w = self.wam.wrapper
            extra = {"views": list(model.views), "view_suffixes": self.view_suffixes, "exec_horizon": self.exec_horizon,
                     "samples": int(self.wam.samples), "checkpoint_step": int(model.step), "skill": skill,
                     "proprio_mask": getattr(w, "proprio_mask_name", None), "action_repr": getattr(w, "action_repr_name", None),
                     "image_size": int(model.image_size), "base_control": getattr(self.wam, "base_control_mode", None),
                     "chunk_path": ("_preprocess(head) -> _model_feats(model, obs, head_px) = get_vision_features(_view_pixels) "
                                    "-> _sample(K averaged) -> _bounded_chunk -> _apply_base_control/_apply_arm_gain -> "
                                    "first exec_horizon actions (policy_server._act_orchestrated, multi-view branch)")}
        if self.is_v3:
            extra.update(architecture="v3", history_rule="reset at handoff; fresh pooled every control frame; no demo-prefix history",
                         chunk_path="per-frame _record_v3_frame -> _proprio_input/_model_feats -> _sample(K=1) -> exec_horizon actions")
        self.emit({"event": "handoff_config", **HANDOFF_LABEL, "handoff_step": n, "wam_checkpoint": str(wam_checkpoint),
                   "wam_kwargs": self.wam_kwargs, "cond": self.cond_info, "noise_rule": self.noise_rule,
                   "proprio_lags": list(getattr(self.wam.model, "proprio_lags", [0])),
                   "history_maxlen": self.wam._proprio_buf.maxlen, **extra})

    def metadata(self):
        extra = ({"contract": self.cond_info["contract"], "exec_horizon": self.exec_horizon, "samples": int(self.wam.samples),
                  "views": list(self.wam.model.views)} if self.multiview else {})
        return {**super().metadata(), **HANDOFF_LABEL, "policy": "demo_prefix_handoff_diagnostic", "handoff_step": self.N,
                "wam_checkpoint": str(getattr(self.wam, "checkpoint", "")), "cond": self.cond_info, "noise_rule": self.noise_rule,
                **extra}

    def reset(self):
        super().reset()
        self.wam.reset()          # deployment: every evaluator reset message resets the WAM episode (history cleared)
        self.queue = []
        self._v3_last_step = None
        self._v3_start_step = None

    def policy_obs(self, obs):
        """What the WAM may read: the 61-D proprio and the RGB views of its checkpoint, under the live evaluator key
        names (head-only model: exactly the original two keys). Nothing else from the evaluator observation."""
        out = {"robot_r1::proprio": self._find(obs, "::proprio")}
        for suffix in self.view_suffixes:
            out[f"robot_r1::robot_r1{suffix}"] = self._find(obs, suffix)
        return out

    def next_action(self, t, obs):
        try:
            return self._next_handoff_action(t, obs)
        except Exception:
            if getattr(self, "is_v3", False):
                # This adapter bypasses WamPolicy.act(), so it must fail closed itself too.
                self.queue = []
                self.wam._reset_v3_contexts("handoff_error")
                self._v3_last_step = None
                self._v3_start_step = None
            raise

    def _next_handoff_action(self, t, obs):
        obs_pol = self.policy_obs(obs)
        state, pixels = self.wam._preprocess(obs_pol)
        self.wam._record_proprio(state)
        if t < self.N:
            return self.actions[t]
        frame_started = time.monotonic()
        if self.is_v3:
            # Only the live post-handoff observations define this skill's history. The legacy episode
            # proprio buffer above may contain prefix frames, but it is NEVER the V3 model's input.
            if self._v3_last_step is None or t != self._v3_last_step + 1:
                self.wam._reset_v3_contexts("handoff_start_or_gap")
                self.queue = []
                self._v3_start_step = t
            self.wam.step_in_episode = t
            self.wam._v3_frame = {}  # allow reuse within this control frame only
            self.wam._head_encode_ms = 0.0  # frozen-condition handoff has no separate VTT/head pass
            self.wam._record_v3_frame(self.wam.model, obs_pol, state, pixels,
                                      (self.wam.episode, "handoff", self.cond_info["contract"], self.N, self._v3_start_step))
            self._v3_last_step = t
        if not self.queue:
            t0 = frame_started if self.is_v3 else time.monotonic()
            if self.multiview or self.is_v3:  # deployment path of a multi-view checkpoint (policy_server._act_orchestrated)
                st = self.wam._proprio_input(self.wam.model, state)
                if self.fixed_noise_seed is not None:
                    self.wam.gen.manual_seed(self.fixed_noise_seed)
                self.wam.step_in_episode = t      # attribution of the WAM's own action_bounds events only
                feats = self.wam._model_feats(self.wam.model, obs_pol, pixels, None)   # head_feats unused by multi-view
                chunk = self.wam._bounded_chunk(self.wam._sample(feats, st, self.cond, model=self.wam.model))
                chunk = self.wam._apply_arm_gain(self.wam._apply_base_control(chunk, self.skill), state, self.skill)
            else:
                feats, _ = self.wam._encode(pixels)
                st = self.wam._proprio_input(self.wam.model, state)
                if self.fixed_noise_seed is not None:
                    self.wam.gen.manual_seed(self.fixed_noise_seed)
                chunk = self.wam._bounded_chunk(self.wam._sample(feats, st, self.cond))
            infer_ms = (time.monotonic() - t0) * 1000
            self.queue = [chunk[i] for i in range(self.exec_horizon)]
            self.chunk_id += 1
            buf = np.stack(self.wam._proprio_buf)
            if self.handoff_record is None:
                ours = self.proprio[max(0, t + 1 - len(buf)):t + 1]
                self.handoff_record = {"step": t, "proprio_frames": int(self.wam._proprio_frames),
                                       "history_buffer_len": int(len(buf)), "history_maxlen": int(self.wam._proprio_buf.maxlen),
                                       "buffer_equals_received_states": bool(np.array_equal(buf, ours)),
                                       "proprio_input_shape": list(np.shape(st)),
                                       "state_vs_demo_frame_max_abs_upper_body": float(np.abs(
                                           np.r_[state[3:10], state[28:35], state[53:57]]
                                           - np.r_[self.demo_state_row(t)[3:10], self.demo_state_row(t)[28:35], self.demo_state_row(t)[53:57]]).max())}
                if self.multiview:
                    self.handoff_record.update(policy_obs_keys=sorted(obs_pol), views=list(self.wam.model.views),
                                               received_rgb_shapes={s: list(np.shape(obs_pol[f"robot_r1::robot_r1{s}"]))
                                                                    for s in self.view_suffixes},
                                               exec_horizon=self.exec_horizon, samples=int(self.wam.samples))
                if self.is_v3:
                    self.handoff_record.update(v3_context_steps=self.wam.model.context.steps,
                                               v3_history_start_step=self._v3_start_step,
                                               v3_prefix_history_used=False)
                write_json(self.out / "handoff_record.json", self.handoff_record)
            demo_a = self.demo_actions_full[t] if t < len(self.demo_actions_full) else None
            rec = {"step": t, "chunk": self.chunk_id, "infer_ms": round(infer_ms, 1),
                   "proprio_frames": int(self.wam._proprio_frames), "history_buffer_len": int(len(buf)),
                   "q_meas": [round(float(x), 4) for x in np.r_[state[53:57], state[3:10], state[28:35]]],
                   "q_cmd_first": [round(float(x), 4) for x in np.r_[chunk[0, 3:7], chunk[0, 7:14], chunk[0, 15:22]]],
                   "q_cmd_end": [round(float(x), 4) for x in np.r_[chunk[-1, 3:7], chunk[-1, 7:14], chunk[-1, 15:22]]],
                   "base_cmd": [round(float(x), 4) for x in chunk[0, :3]],
                   "base_cmd_mean": [round(float(x), 4) for x in chunk[:, :3].mean(0)],
                   "grip": [round(float(chunk[0, 14]), 3), round(float(chunk[0, 22]), 3)],
                   "demo_action_same_step_upper_body": (None if demo_a is None else
                                                        [round(float(x), 4) for x in np.r_[demo_a[3:7], demo_a[7:14], demo_a[15:22]]])}
            if self.multiview:  # executed part only (exec_horizon < 32): gripper commands and the last executed targets
                h = self.exec_horizon
                rec.update(exec_horizon=h,
                           grip_exec_left=[round(float(x), 3) for x in chunk[:h, 14]],
                           grip_exec_right=[round(float(x), 3) for x in chunk[:h, 22]],
                           q_cmd_exec_end=[round(float(x), 4) for x in np.r_[chunk[h - 1, 3:7], chunk[h - 1, 7:14], chunk[h - 1, 15:22]]],
                           demo_grip_same_step=None if demo_a is None else [round(float(demo_a[14]), 3), round(float(demo_a[22]), 3)])
            if self.is_v3:
                rec.update(self.wam._v3_chunk_log(self.wam.model))
                rec.update(v3_history_start_step=self._v3_start_step, samples=int(self.wam.samples))
            self.chunk_log.write(json.dumps(rec) + "\n")
            self.chunk_log.flush()
        a = self.queue.pop(0)
        if self.base_from_demo:
            a = np.array(a, copy=True)
            a[:3] = self.demo_actions_full[t, :3] if t < len(self.demo_actions_full) else 0.0
        return a

    def demo_state_row(self, t):
        if not hasattr(self, "_demo_state"):
            self._demo_state = np.load(LOWDIM / f"episode_{self.episode_id}_state.npy")
        return self._demo_state[min(t, len(self._demo_state) - 1)]


class PrimitiveHandoffPolicy(ReplayPolicy):
    """Task 1 diagnostic (2026-09-27): demo actions for t < N, then the perception+IK manipulation primitive
    (harness/grasp_primitive.py) executes ONE contract from proprio only, given the target object's base-frame position
    at N (`target`, here from the deterministic reference replay's evaluator pose log = oracle position: a geometry /
    controller check of the primitive, never a perception or task result). Optional odometry fine alignment of the base
    (`align_to`, docking calibration target_at="grasp")."""

    def __init__(self, episode, out_dir, handoff_step, *, kind, target=None, align=False, budget=600, seed=0,
                 native_every=15, **kw):
        n = int(handoff_step)
        native = list(range(n, n + int(budget) + 1, int(native_every)))
        super().__init__(episode, out_dir, steps=n, seed=seed, tail_hold=False, keyframes=[0, n - n % 2],
                         native_steps=native, **kw)
        from harness.docking_controller import load_calibration
        # kind may chain contracts ("pick_can,place_in"): the next starts once the previous primitive is done
        self.kinds = kind.split(",")
        kind = self.kinds[0]
        self.N, self.kind, self.target = n, kind, None if target is None else np.asarray(target, float)
        cal = load_calibration(target_at="grasp")
        lab = {"pick_can": "can_of_soda", "pick_bin": "trash_can"}.get(kind)
        self.align_to = cal["dock"][lab]["target"] if (align and lab) else None
        self.carry = cal["posture"]["nav_carrying_bin"]
        self.prim, self.prim_log = None, (self.out / "primitive.jsonl").open("a")
        self.emit({"event": "primitive_handoff_config", **HANDOFF_LABEL, "handoff_step": n, "kind": kind,
                   "target_base_at_N": None if target is None else list(map(float, target)),
                   "align_to": None if self.align_to is None else self.align_to.tolist(),
                   "target_source": "oracle: evaluator pose log of the deterministic reference demo replay at step N"})

    def metadata(self):
        return {**super().metadata(), **HANDOFF_LABEL, "policy": "demo_prefix_primitive_handoff_diagnostic",
                "handoff_step": self.N, "primitive": self.kind}

    def next_action(self, t, obs):
        if t < self.N:
            return self.actions[t]
        state = np.asarray(self._find(obs, "::proprio"), np.float64).reshape(-1)[:61]
        from harness.grasp_primitive import Primitive
        if self.prim is None:
            self.prim = Primitive(self.kind, state, self.target, carry_posture=self.carry, align_to=self.align_to)
        elif self.prim.done and len(self.kinds) > 1:
            self.kinds.pop(0)
            self.kind = self.kinds[0]
            self.prim = Primitive(self.kind, state, None, carry_posture=self.carry)
        phase = self.prim.phase
        a = self.prim.step(state)
        if phase != getattr(self, "_last_phase", None) or t % 30 == 0:
            self.prim_log.write(json.dumps({"step": t, "phase": phase, "feasible": self.prim.feasible,
                                            "plan": self.prim.plan_info, "grip_cmd": [float(a[14]), float(a[22])],
                                            "grip_meas": [float(state[24:26].sum()), float(state[49:51].sum())],
                                            "eef_left": state[17:20].round(4).tolist(), "eef_right": state[42:45].round(4).tolist()}) + "\n")
            self.prim_log.flush()
            self._last_phase = phase
        return a


def _health(connection, request):
    if getattr(request, "path", None) == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def serve_async(policy, host, port, stop_event=None, ready=None):
    import websockets
    import websockets.asyncio.server as ws_server
    Packer, unpackb = _msgpack()

    async def handler(websocket):
        packer = Packer()
        policy.emit({"event": "connect", "peer": str(websocket.remote_address)})
        await websocket.send(packer.pack(policy.metadata()))
        while True:
            try:
                msg = unpackb(await websocket.recv())
            except websockets.ConnectionClosed:
                policy.close()
                return
            if isinstance(msg, dict) and "status" in msg:
                await websocket.send(packer.pack({"episode": max(policy.episode, 0), "step": policy.step,
                                                  "status": "demo_action_replay"}))
                continue
            if isinstance(msg, dict) and "reset" in msg:
                policy.reset()
                continue
            t0 = time.monotonic()
            action = policy.act(msg)
            delay = policy.throttle.delay_s()
            policy.note_delay(delay)
            if delay > 0:
                await asyncio.sleep(delay)
            await websocket.send(packer.pack({"action": action, "server_timing": {
                "infer_ms": (time.monotonic() - t0) * 1000, "throttle_ms": delay * 1000}}))

    async with ws_server.serve(handler, host, port, compression=None, max_size=None, process_request=_health):
        if ready is not None:
            ready.set()
        if stop_event is None:
            await asyncio.Future()
        else:
            while not stop_event.is_set():
                await asyncio.sleep(0.05)


def cmd_serve(args):
    throttle = Throttle(args.throttle, fixed_ms=args.fixed_ms)
    if args.handoff_step is not None and args.primitive:
        tgt = None if not args.primitive_target else [float(v) for v in args.primitive_target.split(",")]
        policy = PrimitiveHandoffPolicy(args.episode, args.out, args.handoff_step, kind=args.primitive, target=tgt,
                                        align=args.primitive_align, budget=args.budget, seed=args.seed, throttle=throttle,
                                        native_dir=args.native_dir, head_stride=args.head_stride)
    elif args.handoff_step is not None:
        extra = {k: v for k, v in (("contract", args.wam_contract), ("skill", args.wam_skill),
                                   ("exec_horizon", args.wam_exec_horizon), ("base_control", args.wam_base_control)) if v is not None}
        if TASK_ID != 13 and "contract" not in extra:
            raise SystemExit("Task 1 handoff: --wam-contract is required (e.g. 'pick up from(can_of_soda,floors)')")
        policy = HandoffPolicy(args.episode, args.out, args.handoff_step, args.wam_checkpoint, wam_device=args.wam_device,
                               wam_samples=args.wam_samples, wam_num_steps=args.wam_num_steps,
                               cuda_fraction=args.wam_cuda_fraction, seed=args.seed, budget=args.budget,
                               fixed_noise_seed=args.wam_fixed_noise_seed, base_from_demo=args.wam_base_from_demo,
                               throttle=throttle, native_dir=args.native_dir, head_stride=args.head_stride, **extra)
    else:
        policy = ReplayPolicy(args.episode, args.out, steps=args.steps, seed=args.seed, throttle=throttle, native_dir=args.native_dir,
                              head_stride=args.head_stride, native_mode=args.native_mode, perception_every=args.perception_every)
    print(json.dumps({"serving": f"{args.host}:{args.port}", **policy.metadata()}), flush=True)

    def _term(*_):
        policy.close()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _term)
    asyncio.run(serve_async(policy, args.host, args.port))


# ---------------------------------------------------------------------------------------------------------------
# launcher (system python3): GPU lock, replay server unit, guarded evaluator, stricter 85 % watchdog, manifest
# ---------------------------------------------------------------------------------------------------------------
def active_gpu_units():
    out = subprocess.run(["systemctl", "--user", "list-units", "--type=service", "--state=active,activating",
                          "--no-legend", "--plain", "wam-*", "behavior-run-*"], capture_output=True, text=True).stdout
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.split()
    return int(out[0])


def gpu_total_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.split()
    return int(out[0])


def gpu_free_reason(max_used_mib=2600):
    if LOCK.exists():
        return f"lock held: {LOCK.read_text(errors='ignore').strip()!r}"
    used = gpu_used_mib()
    if used >= max_used_mib:
        return f"GPU memory used {used} MiB >= {max_used_mib}"
    units = active_gpu_units()
    if units:
        return f"active GPU units {units}"
    return None


def acquire_gpu_lock(run_name, log_path, poll=10.0, max_wait=None):
    started = time.monotonic()
    last = None
    with open(log_path, "a") as log:
        while True:
            reason = gpu_free_reason()
            if reason is None:
                try:
                    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                except FileExistsError:
                    reason = "lock appeared during acquisition"
                else:
                    os.write(fd, f"{run_name}\n".encode())
                    os.close(fd)
                    units = active_gpu_units()  # re-check after taking the lock (race with a unit start)
                    if units:
                        release_gpu_lock(run_name)
                        reason = f"active GPU units after lock {units}"
                    else:
                        log.write(json.dumps({"time": utc(), "event": "acquired", "waited_s": round(time.monotonic() - started)}) + "\n")
                        return time.monotonic() - started
            if reason != last:
                log.write(json.dumps({"time": utc(), "event": "waiting", "reason": reason}) + "\n")
                log.flush()
                print(f"[{utc()}] waiting for GPU: {reason}", flush=True)
                last = reason
            if max_wait is not None and time.monotonic() - started > max_wait:
                raise TimeoutError(f"GPU not free after {max_wait} s: {reason}")
            time.sleep(poll)


def release_gpu_lock(run_name):
    try:
        if LOCK.read_text().strip() == run_name:
            LOCK.unlink()
            return True
    except FileNotFoundError:
        pass
    return False


def unit_active(unit):
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0


class Watchdog(threading.Thread):
    """Project ceiling (85 %) on top of behavior_resources.run_guarded (whose configured limits are 90/95 %)."""

    def __init__(self, unit, log_path, limit=85.0, gpu_util_window_s=60.0, vram_limit_mib=None):
        super().__init__(daemon=True)
        self.unit, self.log_path, self.limit, self.window = unit, log_path, limit, gpu_util_window_s
        self.vram_limit_mib = vram_limit_mib
        self.stop = threading.Event()
        self.reason = None
        self.peaks = {"cpu": 0.0, "ram": 0.0, "vram_mib": 0, "vram_pct": 0.0, "gpu_util": 0}

    def run(self):
        from behavior_preflight import hardware_snapshot
        hot_since, vram_hits, cpu_since = None, 0, None
        with open(self.log_path, "a") as log:
            while not self.stop.is_set():
                s = hardware_snapshot()
                now = time.monotonic()
                gpu = s["gpu"][0] if s["gpu"] else None
                self.peaks["cpu"] = max(self.peaks["cpu"], s["cpu_used_percent"])
                self.peaks["ram"] = max(self.peaks["ram"], s["memory_used_percent"])
                reason = None
                if s["memory_used_percent"] >= self.limit:
                    reason = f"RAM {s['memory_used_percent']}% >= {self.limit}%"
                cpu_since = (cpu_since or now) if s["cpu_used_percent"] >= self.limit else None
                if cpu_since is not None and now - cpu_since >= 5.0:
                    reason = f"CPU >= {self.limit}% for 5 s"
                if gpu is None:
                    reason = reason or "GPU telemetry unavailable"
                else:
                    pct = 100.0 * gpu["vram_used_mib"] / gpu["vram_total_mib"]
                    self.peaks["vram_mib"] = max(self.peaks["vram_mib"], gpu["vram_used_mib"])
                    self.peaks["vram_pct"] = max(self.peaks["vram_pct"], round(pct, 2))
                    self.peaks["gpu_util"] = max(self.peaks["gpu_util"], gpu["utilization_percent"])
                    vlim = self.vram_limit_mib if self.vram_limit_mib else gpu["vram_total_mib"] * self.limit / 100
                    vram_hits = vram_hits + 1 if gpu["vram_used_mib"] >= vlim else 0
                    if vram_hits >= 2:
                        reason = reason or f"VRAM {gpu['vram_used_mib']} MiB ({pct:.1f}%) >= limit {vlim:.0f} MiB twice"
                    hot_since = (hot_since or now) if gpu["utilization_percent"] >= self.limit else None
                    if hot_since is not None and now - hot_since >= self.window:
                        reason = reason or f"GPU utilization >= {self.limit}% for {self.window:.0f} s"
                log.write(json.dumps({"t": round(now, 2), "cpu": s["cpu_used_percent"], "ram": s["memory_used_percent"],
                                      "vram_mib": gpu and gpu["vram_used_mib"], "gpu_util": gpu and gpu["utilization_percent"]}) + "\n")
                log.flush()
                if reason:
                    self.reason = reason
                    log.write(json.dumps({"event": "abort", "reason": reason, "time": utc()}) + "\n")
                    subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", self.unit], capture_output=True)
                    for _ in range(20):
                        if not unit_active(self.unit):
                            break
                        time.sleep(0.5)
                    subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", self.unit], capture_output=True)
                    return
                self.stop.wait(0.25)


def wait_health(port, timeout=60.0):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


class EarlyStop(threading.Thread):
    """Evaluator-side stop rule for handoff runs (privileged joint log; never reaches the policy): SIGINT the evaluator
    (its own Ctrl-C path finalizes video + joint log) once j_trunk >= thr for `hold` consecutive steps after the handoff."""

    def __init__(self, joint_dir, unit, start_step, thr=1.0, hold=300):
        super().__init__(daemon=True)
        self.joint_dir, self.unit, self.start_step, self.thr, self.hold = Path(joint_dir), unit, start_step, thr, hold
        self.stop = threading.Event()
        self.triggered = None

    def run(self):
        offset, buf, consec, path = 0, "", 0, None
        while not self.stop.is_set():
            if path is None:
                files = sorted(self.joint_dir.glob("*.jsonl")) if self.joint_dir.exists() else []
                path = files[0] if files else None
            if path is not None:
                with open(path) as f:
                    f.seek(offset)
                    buf += f.read()
                    offset = f.tell()
                *lines, buf = buf.split("\n")
                for line in lines:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if "step" not in rec or rec["step"] < self.start_step:
                        continue
                    val = next((v[0] for k, v in rec["joints"].items() if k.startswith("car")), None)
                    consec = consec + 1 if val is not None and val >= self.thr else 0
                    if consec >= self.hold and self.triggered is None:
                        self.triggered = {"step": rec["step"], "time": utc(), "rule": f"j_trunk >= {self.thr} for {self.hold} steps"}
                        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGINT", self.unit], capture_output=True)
                        for _ in range(120):
                            if not unit_active(self.unit):
                                return
                            time.sleep(1)
                        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", self.unit], capture_output=True)
                        return
            self.stop.wait(1.0)


class T1HandoffCriterion:
    """Pre-registered Task-1 handoff criterion (t1_handoff_spec) on the evaluator pose log, row by row. Evaluator-side
    evidence only: nothing here ever reaches the policy. lift = lowest oriented-bbox point minus its step-0 value (the
    first row fed must be step 0); inside = the BDDL goal literal inside(target, container) as logged per step."""

    def __init__(self, header, spec):
        self.kind = spec["kind"]
        names = {info.get("name"): inst for inst, info in header["entities"].items()}
        if spec["target_object"] not in names:
            raise ValueError(f"pose log has no entity named {spec['target_object']!r}: {sorted(names)}")
        self.inst = names[spec["target_object"]]
        self.info = header["entities"][self.inst]
        self.lit = self.literal = None
        if self.kind == "toggled_on":  # Task 0 press: the evaluator's own goal literal, logged per step
            self.literal = f"(toggled_on {self.inst})"
            options = (header.get("goal_literals") or {}).get("options") or []
            if not options or self.literal not in options[0]:
                raise ValueError(f"pose log does not log the goal literal {self.literal}")
            self.lit = options[0].index(self.literal)
        if self.kind == "inside":
            self.literal = f"(inside {self.inst} {names[spec['container']]})"
            options = (header.get("goal_literals") or {}).get("options") or []
            if not options or self.literal not in options[0]:
                raise ValueError(f"pose log does not log the goal literal {self.literal}")
            self.lit = options[0].index(self.literal)
        self.corners, self.ref = None, None

    def lowest_z(self, poses7):
        poses7 = np.atleast_2d(np.asarray(poses7, np.float64))
        if self.corners is None:  # object-frame box from the header (native bbox x scale, reset AABB centre) at step 0
            geo = t1_bin_geometry(self.info, poses7[0])
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64)
            self.corners = geo["centre_in_bin_frame"] + signs * geo["extent_m"] / 2
        R = quat_mat(poses7[:, 3:7])
        return (np.einsum("tij,kj->tki", R, self.corners)[..., 2] + poses7[:, None, 2]).min(1)

    def values(self, rows):
        if self.kind in ("inside", "toggled_on"):
            return np.array([float(bool(r["goal_literals"][0][self.lit])) for r in rows])
        z = self.lowest_z([r["objects"][self.inst] for r in rows])
        if self.ref is None:
            if int(rows[0]["step"]) != 0:
                raise ValueError("the lift reference is the pose-log row of step 0")
            self.ref = float(z[0])
        return z - self.ref

    def holds(self, value):
        return bool(value >= T1_HANDOFF_LIFT_M) if self.kind == "lift" else bool(value >= 0.5)

    def marker_steps(self, rows):
        """Task 0: the evaluator's ToggledOn hand_in_marker_steps per row (pose-log field 'toggled_on'), NaN if absent."""
        out = []
        for r in rows:
            v = (r.get("toggled_on") or {}) if isinstance(r.get("toggled_on"), dict) else {}
            out.append(float(v[self.inst][1]) if self.inst in v else float("nan"))
        return np.array(out)


class EarlyStopTask1(threading.Thread):
    """Task-1 handoff stop rule (evaluator-side pose log, never a policy input): SIGINT the evaluator (its Ctrl-C path
    finalizes the video and logs) once the pre-registered criterion holds for `hold` consecutive steps >= start_step."""

    def __init__(self, pose_dir, unit, start_step, spec, hold=T1_EARLY_STOP_STEPS):
        super().__init__(daemon=True)
        self.pose_dir, self.unit, self.start_step, self.spec, self.hold = Path(pose_dir), unit, int(start_step), spec, int(hold)
        self.stop = threading.Event()
        self.triggered, self.error = None, None
        self._crit, self._consec = None, 0

    def rule(self):
        what = (f"{self.spec['target_object']} lowest-point lift >= {T1_HANDOFF_LIFT_M} m" if self.spec["kind"] == "lift"
                else f"BDDL toggled_on({self.spec['target_object']}) true" if self.spec["kind"] == "toggled_on"
                else f"BDDL inside({self.spec['target_object']}, {self.spec['container']}) true")
        return f"{what} for {self.hold} consecutive steps after the handoff"

    def feed(self, rec):
        """One decoded pose-log record (header first, then rows in step order) -> True when the rule fires."""
        if rec.get("type") == "header":
            self._crit = T1HandoffCriterion(rec, self.spec)
            return False
        if self._crit is None or rec.get("type") == "end" or "step" not in rec:
            return False
        value = float(self._crit.values([rec])[0])   # the first row (step 0) sets the lift reference
        if rec["step"] < self.start_step:
            return False
        self._consec = self._consec + 1 if self._crit.holds(value) else 0
        if self._consec >= self.hold and self.triggered is None:
            self.triggered = {"step": int(rec["step"]), "time": utc(), "value": round(value, 4), "rule": self.rule()}
            return True
        return False

    def run(self):
        offset, buf, path = 0, "", None
        while not self.stop.is_set():
            if path is None:
                files = sorted(self.pose_dir.glob("*.jsonl")) if self.pose_dir.exists() else []
                path = files[0] if files else None
            if path is not None:
                with open(path) as f:
                    f.seek(offset)
                    buf += f.read()
                    offset = f.tell()
                *lines, buf = buf.split("\n")
                for line in lines:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    try:
                        fired = self.feed(rec)
                    except Exception as e:  # an evidence problem never stops the run; the budget still ends it
                        self.error = repr(e)
                        return
                    if fired:
                        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGINT", self.unit], capture_output=True)
                        for _ in range(120):
                            if not unit_active(self.unit):
                                return
                            time.sleep(1)
                        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", self.unit], capture_output=True)
                        return
            self.stop.wait(1.0)


def cmd_run(args):
    from behavior_closed_loop import wait_for_policy_capacity, wait_for_policy_request
    from behavior_eval import command as eval_command
    from behavior_preflight import CONFIG, child_environment, collect
    from behavior_resources import run_guarded

    handoff = args.handoff_step is not None
    label = HANDOFF_LABEL if handoff else LABEL
    run_dir = (T1_HANDOFF_ROOT if handoff and TASK_ID != 13 else OUT_ROOT) / args.run_name
    if args.dry_run:  # CPU-only launch check: everything up to the manifest, no GPU lock, no units, run dir in /tmp
        import shutil
        run_dir = Path("/tmp/p2_handoff_dryrun") / args.run_name
        shutil.rmtree(run_dir, ignore_errors=True)
    if TASK_ID != 13 and args.stop_after_openlid is not None:
        raise SystemExit("--stop-after-openlid is a Task 13 (open lid) option")
    spec = ckpt_cfg = None
    if handoff and TASK_ID == 13:  # Task 13 handoff: fixed open lid(car) / exec 32 / budget 3084 / early stop 300 steps
        if any(v is not None for v in (args.wam_contract, args.wam_skill, args.wam_exec_horizon)):
            raise SystemExit("the Task 13 handoff is fixed to 'open lid(car)' with exec horizon 32")
        args.budget = 3084 if args.budget is None else args.budget
        args.early_stop_steps = 300 if args.early_stop_steps is None else args.early_stop_steps
    elif handoff:  # Task 1: the contract must be the demo segment at N; criterion pre-registered in the manifest
        try:
            spec = handoff_spec(args.episode, args.handoff_step, args.wam_contract, args.wam_skill, args.budget)
        except (ValueError, KeyError) as e:
            raise SystemExit(f"Task {TASK_ID} handoff: {e}")
        args.budget = spec["budget_steps"]
        args.wam_exec_horizon = T1_HANDOFF_EXEC_HORIZON if args.wam_exec_horizon is None else args.wam_exec_horizon
        if TASK_ID == 0:  # criteria judged at the budget end (pick) / ended by the evaluator's own success (press)
            args.early_stop_steps = 0 if args.early_stop_steps is None else args.early_stop_steps
            args.wam_base_control = T0_BASE_CONTROL if args.wam_base_control is None else args.wam_base_control
        args.early_stop_steps = T1_EARLY_STOP_STEPS if args.early_stop_steps is None else args.early_stop_steps
        if not args.wam_checkpoint or not Path(args.wam_checkpoint).is_file():
            raise SystemExit(f"missing WAM checkpoint {args.wam_checkpoint!r}")
        ckpt_cfg = t1_checkpoint_summary(args.wam_checkpoint)
        if ckpt_cfg is not None and (ckpt_cfg["task_id"] != TASK_ID or ckpt_cfg["cond_mode"] != "contract_proto"
                                     or ckpt_cfg["contract_targets_sha256"] != sha256_file(CONTRACT_TARGETS)):
            raise SystemExit(f"checkpoint is not a Task-{TASK_ID} contract_proto model on {CONTRACT_TARGETS.relative_to(ROOT)}: {ckpt_cfg}")
    if args.tail_hold_steps is None:  # objects settle after the last demo action (Task 1: ~10 s)
        args.tail_hold_steps = 90 if TASK_ID == 13 else 300
    table_ok = LOWDIM / f"episode_{args.episode}_action.npy"
    if not table_ok.is_file():
        raise SystemExit(f"missing {table_ok}")
    index = json.loads((LOWDIM / "index.json").read_text())["episodes"][str(args.episode)]
    if index["split"] != "train":
        raise SystemExit(f"episode {args.episode} is not a TRAIN demo ({index['split']})")
    if args.instance in test_instance_ids():
        raise SystemExit("refusing a public/hidden test instance")
    action_sha = sha256_file(table_ok)  # SHA256 of the .npy file
    array_sha = hashlib.sha256(np.load(table_ok).tobytes()).hexdigest()  # index.json convention (array bytes)
    if array_sha != index["action_sha256"]:
        raise SystemExit(f"action array SHA256 differs from {(LOWDIM / 'index.json').relative_to(ROOT)}")
    n_demo = int(index["length"])
    if handoff:
        if not 0 < args.handoff_step < n_demo or not args.wam_checkpoint:
            raise SystemExit("--handoff-step must be inside the demo and --wam-checkpoint is required")
        steps, max_steps = args.handoff_step, args.handoff_step + args.budget
    else:
        steps = n_demo if args.steps is None else min(args.steps, n_demo)
        if args.stop_after_openlid is not None:
            steps = min(n_demo, openlid_segment(args.episode)[1] + args.stop_after_openlid)
        max_steps = steps + args.tail_hold_steps
    total_vram = gpu_total_mib()
    vram_limit_mib = (int(CONFIG["resource_policy"]["gpu_total_vram_limit_mib"]) if handoff or TASK_ID != 13  # lead decisions 2026-09-25/26
                      else int(total_vram * 0.85))
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a previous run
    evidence_path = run_dir / "mapping_evidence.json"
    rc = subprocess.run([str(LILAWAM), str(Path(__file__).resolve()), "mapping", "--episode", str(args.episode),
                         "--instance", str(args.instance), "--out", str(evidence_path)],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}).returncode
    evidence = json.loads(evidence_path.read_text())
    if rc != 0 or not evidence.get("established"):
        write_json(run_dir / "manifest.json", {**label, "status": "stopped_mapping_not_established", "checks": evidence.get("checks")})
        raise SystemExit("demo -> train instance mapping NOT established; stopping (no guess)")
    blockers = collect()["blockers"]
    if blockers:
        write_json(run_dir / "manifest.json", {**label, "status": "blocked_preflight", "blockers": blockers})
        raise SystemExit(f"preflight blockers: {blockers}")

    pid = os.getpid()
    eval_unit = f"behavior-run-{pid}"
    server_unit = f"wam-handoff-{pid}" if handoff else f"demo-replay-srv-{pid}"
    argv = eval_command("development", "127.0.0.1", args.port)
    argv[argv.index("--mode") + 1] = "train"
    i, j = argv.index("--instance-indices") + 1, argv.index("--num-rollouts")
    argv[i:j] = [str(args.instance)]
    argv[argv.index("--output-dir") + 1] = str(run_dir / "eval")
    argv += ["--max-steps", str(max_steps), "--seed", str(args.seed), "--log-object-joints"]
    if TASK_ID != 13:  # evaluator-side ground-truth poses (opt-in port addition) + the official RGB-D wrapper
        argv += ["--log-object-poses", "--env-wrapper", T1_WRAPPERS[args.env_wrapper]]
    if not args.no_video:
        argv += ["--write-video", "--video-view", "viewer", "--video-width", str(args.video_width),
                 "--video-height", str(args.video_height), "--video-framing", "auto"]
    assert argv[argv.index("--mode") + 1] == "train" and argv[i] == str(args.instance)
    server_cmd = [str(LILAWAM), str(Path(__file__).resolve()), "serve", "--episode", str(args.episode),
                  "--steps", str(steps), "--port", str(args.port), "--seed", str(args.seed),
                  "--out", str(run_dir / "replay"), "--throttle", args.throttle,
                  "--native-dir", str(run_dir / "native_frames")]
    server_cmd += (["--native-mode", args.native_mode] if TASK_ID == 13 else [] if handoff  # Task-1 handoff: own capture after N
                   else ["--perception-every", str(args.capture_every)])
    ckpt_sha = None
    if handoff:
        server_cmd += ["--handoff-step", str(args.handoff_step), "--wam-checkpoint", str(Path(args.wam_checkpoint).resolve()),
                       "--wam-device", "cuda", "--wam-samples", str(args.wam_samples), "--wam-num-steps", str(args.wam_num_steps),
                       "--wam-cuda-fraction", str(args.wam_cuda_fraction), "--budget", str(args.budget)]
        if args.wam_fixed_noise_seed is not None:
            server_cmd += ["--wam-fixed-noise-seed", str(args.wam_fixed_noise_seed)]
        if args.wam_base_from_demo:
            server_cmd += ["--wam-base-from-demo"]
        if args.primitive:
            server_cmd += ["--primitive", args.primitive, "--primitive-target", args.primitive_target or ""]
            if args.primitive_align:
                server_cmd += ["--primitive-align"]
        if TASK_ID != 13:
            server_cmd += ["--wam-contract", spec["contract"], "--wam-skill", spec["skill"],
                           "--wam-exec-horizon", str(args.wam_exec_horizon)]
        if args.wam_base_control is not None:
            server_cmd += ["--wam-base-control", args.wam_base_control]
        ckpt_sha = sha256_file(args.wam_checkpoint)
    artifacts = [Path(__file__).resolve(), ROOT / "scripts/behavior_eval.py", ROOT / "scripts/behavior_preflight.py",
                 ROOT / "scripts/behavior_resources.py", ROOT / "configs/behavior_environment.json",
                 EVAL_UTILS, EVAL_UTILS.parents[1] / "eval.py", EVAL_UTILS.parents[1] / "evaluator.py",
                 EVAL_UTILS.parents[1] / "r1pro.yaml", instance_file(args.instance), SEGMENTS, SPLIT, EPISODES_META,
                 LOWDIM / "index.json", LOWDIM / f"episode_{args.episode}_state.npy"]
    if handoff:
        artifacts += [ROOT / "harness/policy_server.py", ROOT / "harness/wam_behavior.py", ROOT / "harness/orchestrator.py",
                      CONTRACT_TARGETS, ROOT / "configs/r1pro_action_bounds.json"]
    if TASK_ID != 13:
        artifacts += [ROOT / "harness/task_profile.py", ROOT / "third_party/BEHAVIOR-1K-isaac6/PORT_PROVENANCE.json",
                      EVAL_UTILS.parents[1] / "wrappers/rgbd_full_res_wrapper.py", EVAL_UTILS.parents[1] / "wrappers/default_wrapper.py"]
    if handoff and TASK_ID != 13:  # ConditionBank inputs + the contract-key helper used by t1_handoff_spec
        artifacts += [TP.VTT_DIR / "meta.json", TP.VTT_DIR / "prototypes.npy", ROOT / "harness/vtt_monitor.py"]
    if TASK_ID == 0:
        artifacts += [ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/object_states/toggle.py"]
    segment_info = ({"open_lid_segment_frames": list(openlid_segment(args.episode))} if TASK_ID == 13 else
                    {"task_id": TASK_ID, "task": TASK, "radio_pick_segment_frames": list(radio_pick_segment(args.episode)),
                     "skill_segments": [[sg["skill"], sg["objects"][0][0] if sg["objects"] else None, int(sg["start"]), int(sg["end"])]
                                        for sg in demo_segments(args.episode)]} if TASK_ID == 0 else
                    {"task_id": TASK_ID, "task": TASK, "trash_can_pick_segment_frames": list(trash_pick_segment(args.episode)),
                     "skill_segments": [[sg["skill"], sg["objects"][0][0] if sg["objects"] else None, int(sg["start"]), int(sg["end"])]
                                        for sg in demo_segments(args.episode)]})
    manifest = {**label, "run_name": args.run_name, "status": "waiting_for_gpu_lock", "created_utc": utc(),
                "demo_episode": args.episode, "raw_episode_id": evidence["dataset_metadata"]["raw_episode_id"],
                "train_instance_id": args.instance, "evaluator_mode": "train",
                "mapping_established": True, "mapping_evidence": "mapping_evidence.json",
                "action_file": str(table_ok.relative_to(ROOT)), "action_file_sha256": action_sha,
                "action_array_sha256": array_sha, "action_array_sha256_matches_lowdim_index": True, "demo_length_frames": n_demo, "control_rate_hz": 30,
                "replay_steps_demo_actions": steps, "evaluator_max_steps": max_steps,
                **segment_info,
                "demo_outcome_in_dataset": evidence["demo_outcome"],
                "seeds": {"evaluator_seed": args.seed, "policy_seed_reported": args.seed,
                          "note": ("WAM Generator seeded once at construction" if handoff else "replay is deterministic (no RNG)")
                                  + "; Isaac/PhysX/RTX are not bitwise deterministic"},
                "isaac_sim_deviation": "Isaac Sim 6.1 port (third_party/BEHAVIOR-1K-isaac6/PORT_PROVENANCE.json); not the official 5.1 pin",
                "evaluator_argv": argv, "evaluator_env": child_environment(), "evaluator_unit": eval_unit,
                "server_unit": server_unit, "server_argv": server_cmd, "launcher_pid": pid,
                "throttle": args.throttle, "gpu_lock": str(LOCK), "viewer_video": not args.no_video,
                "resource_rules": {"ram_cpu_percent": 85, "vram_limit_mib": vram_limit_mib,
                                   "vram_rule": ("configs/behavior_environment.json gpu_total_vram_limit_mib (lead decision for handoff runs)"
                                                 if handoff else "85% of total" if TASK_ID == 13 else
                                                 "configs/behavior_environment.json gpu_total_vram_limit_mib (lead decision 2026-09-26, Task 1)"),
                                   "gpu_util": "85% sustained 60 s aborts; adaptive per-step throttle keeps EMA <= 75%"},
                "artifact_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in artifacts}}
    if handoff and TASK_ID == 13:
        manifest.update(handoff_step=args.handoff_step, budget_steps=args.budget,
                        wam_checkpoint=str(Path(args.wam_checkpoint)), wam_checkpoint_sha256=ckpt_sha,
                        wam_config={"skill": "move to", "exec_horizon": 32, "num_steps": args.wam_num_steps, "device": "cuda",
                                    "cuda_fraction": args.wam_cuda_fraction, "seed": args.seed, "samples": args.wam_samples,
                                    "orchestrate": {"vlm": "none", "frozen": True, "train_targets": str(CONTRACT_TARGETS.relative_to(ROOT))},
                                    "cond": "open lid(car) contract vector via frozen ConditionBank.select", "whole_body": True,
                                    "numerics": "bf16 autocast on cuda (deployment path)",
                                    "noise_rule": ("policy.gen.manual_seed(%d) immediately before every _sample() call "
                                                   "(fixed initial noise per chunk; mode commitment)" % args.wam_fixed_noise_seed
                                                   if args.wam_fixed_noise_seed is not None
                                                   else "Generator seeded once (seed) at construction; fresh noise every chunk")},
                        early_stop={"rule": f"j_trunk >= {args.early_stop_jtrunk} for {args.early_stop_steps} consecutive steps after handoff",
                                    "mechanism": "evaluator-side joint log -> SIGINT (evaluator Ctrl-C path); never a policy input"},
                        wam_start={"after": "evaluator health-check request (scene imported)", "reserve_mib": args.wam_reserve_mib,
                                   "limit_mib": vram_limit_mib},
                        native_frames={"dir": "native_frames", "every": 15, "from_step": args.handoff_step,
                                       "content": "raw head/wrist rgb as received (lossless PNG) after the handoff + proprio.npy + frame_index.npy (-1 = WAM step)"})
    elif handoff:  # Task 1: one manipulation contract of the multi-view WAM; criterion pre-registered here, before the run
        noise = ("policy.gen.manual_seed(%d) immediately before every _sample() call (fixed initial noise per chunk)"
                 % args.wam_fixed_noise_seed if args.wam_fixed_noise_seed is not None
                 else "Generator seeded once (seed) at construction; fresh noise every chunk")
        manifest.update(handoff_step=args.handoff_step, budget_steps=args.budget,
                        wam_checkpoint=str(Path(args.wam_checkpoint)), wam_checkpoint_sha256=ckpt_sha,
                        wam_checkpoint_config=ckpt_cfg,
                        wam_config={"contract": spec["contract"], "skill": spec["skill"], "exec_horizon": args.wam_exec_horizon,
                                    "samples": args.wam_samples, "num_steps": args.wam_num_steps, "device": "cuda",
                                    "cuda_fraction": args.wam_cuda_fraction, "seed": args.seed,
                                    "base_from_demo": bool(args.wam_base_from_demo),
                                    "executor": ("primitive:" + args.primitive) if args.primitive else "wam",
                                    "primitive_target_oracle": args.primitive_target or None,
                                    "primitive_align": bool(args.primitive_align),
                                    "orchestrate": {"vlm": "none", "frozen": True, "train_targets": str(CONTRACT_TARGETS.relative_to(ROOT))},
                                    "cond": (f"{spec['contract']} contract vector via frozen ConditionBank.select, equal to "
                                             f"{CONTRACT_TARGETS.relative_to(ROOT)} (checked at server start)"),
                                    "policy_obs": ("robot_r1::proprio + the checkpoint's RGB views under their live keys (head "
                                                   "':zed_link:Camera:0::rgb', wrists ':left/right_realsense_link:Camera:0::rgb'); no depth, "
                                                   "cam_rel_poses, task id or evaluator log"),
                                    "image_chain": ("policy_server.prep_rgb per view: RGB(A) -> uint8 -> cv2 INTER_AREA to the checkpoint "
                                                    "image_size (224) -> ImageNet normalisation"),
                                    "chunk_path": ("policy_server._act_orchestrated multi-view branch: _model_feats (one DINO pass over "
                                                   "the V views) -> _sample (K flow samples averaged, num_steps Euler) -> _bounded_chunk "
                                                   "-> _apply_base_control/_apply_arm_gain (identity: wam_whole_body, gains 1) -> first "
                                                   "exec_horizon actions, then re-plan"),
                                    "whole_body": True, "numerics": "bf16 autocast on cuda (deployment path)", "noise_rule": noise},
                        handoff_criterion={**spec, "registered_utc": utc(),
                                           "registration": "written into this manifest before the GPU lock / evaluator start"},
                        early_stop={"rule": EarlyStopTask1(run_dir, "", args.handoff_step, spec, args.early_stop_steps).rule(),
                                    "hold_steps": args.early_stop_steps,
                                    "mechanism": "evaluator-side pose log -> SIGINT (evaluator Ctrl-C path); never a policy input"},
                        wam_start={"after": "evaluator health-check request (scene imported)", "reserve_mib": args.wam_reserve_mib,
                                   "limit_mib": vram_limit_mib},
                        env_wrapper=T1_WRAPPERS[args.env_wrapper],
                        pose_log=f"eval/object_poses/{TASK}_{args.instance}_0.jsonl",
                        native_frames={"dir": "native_frames", "every": 15, "from_step": args.handoff_step,
                                       "content": ("raw head/left/right wrist RGB as received (lossless PNG, native wrapper resolution) "
                                                   "after the handoff + proprio.npy + frame_index.npy (-1 = WAM step); evidence only")},
                        evaluator_termination_note=("the evaluator terminates only when all 3 cans are inside (not reachable in one "
                                                    "contract); the run ends at N + budget or at the early stop"))
        if TASK_ID == 0:  # Task 0 wording / base ownership (the Task-1 fields above are left untouched)
            manifest["wam_config"].update(
                base_control=args.wam_base_control,
                chunk_path=("policy_server._act_orchestrated multi-view branch: _model_feats (one DINO pass over the V views) -> "
                            "_sample (K flow samples averaged, num_steps Euler) -> _bounded_chunk -> _apply_base_control "
                            f"({args.wam_base_control}: base dims zeroed for this manipulation contract"
                            + (", then replaced by the demo base actions (--wam-base-from-demo)" if args.wam_base_from_demo else "")
                            + ") / _apply_arm_gain (gain 1 = identity) -> first exec_horizon actions, then re-plan"),
                whole_body=args.wam_base_control == "wam_whole_body")
            manifest["early_stop"] = ({"rule": "disabled: the criterion is judged at the budget end (pick) or the evaluator ends "
                                               "the episode itself when toggled_on(radio) holds (press)", "hold_steps": 0}
                                      if not args.early_stop_steps else manifest["early_stop"])
            manifest["evaluator_termination_note"] = ("the evaluator ends the episode at the first step toggled_on(radio) holds "
                                                      "(BDDL goal; success/Q=1); otherwise the run ends at N + budget")
            manifest["evaluator_side_logging"] = ("pose log per step: radio root pose, BDDL goal literal toggled_on(radio), and "
                                                  "'toggled_on': [value, hand_in_marker_steps] (evaluator.py port field); evidence "
                                                  "only, never a policy input; finger contacts not logged")
    elif TASK_ID != 13:
        manifest.update(tail_hold_steps=args.tail_hold_steps,
                        tail_action="zero base velocity, last demo arm/trunk/gripper targets (lets objects settle; not a demo action)",
                        env_wrapper=T1_WRAPPERS[args.env_wrapper],
                        pose_log=f"eval/object_poses/{TASK}_{args.instance}_0.jsonl",
                        native_frames={"dir": "native_frames", "every": args.capture_every,
                                       "content": ("head/left/right wrist RGB as rendered (lossless PNG) + head depth_linear (float16 "
                                                   ".npy, metres) every N steps; proprio.npy + cam_rel_poses.npy (T,3,7) + "
                                                   "frame_index.npy every step; capture_meta.json; evidence only, never fed to any policy"),
                                       "ground_truth": f"eval/object_poses/{TASK}_{args.instance}_0.jsonl"},
                        evaluator_termination_note=("the evaluator ends the episode at the first step the BDDL goal holds "
                                                    "(terminated=success); the tail hold only runs if that never happens"))
    else:
        manifest.update(tail_hold_steps=args.tail_hold_steps,
                        tail_action="zero base velocity, last demo arm/trunk/gripper targets (lets objects settle; not a demo action)",
                        native_frames={"dir": "native_frames", "capture_steps": native_capture_steps(args.episode, steps, args.native_mode),
                                       "content": ("raw head/left/right wrist rgb exactly as received (lossless PNG) at the capture "
                                                   "steps + proprio.npy (every step) + frame_index.npy; lead request; evidence only, "
                                                   "never fed to any policy")})
    write_json(run_dir / "manifest.json", manifest)
    if args.dry_run:
        manifest["status"] = "dry_run_stopped_before_gpu_lock"
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"dry_run_manifest": str(run_dir / "manifest.json"), "would_write_to": str(
            (T1_HANDOFF_ROOT if handoff and TASK_ID != 13 else OUT_ROOT) / args.run_name), "evaluator_unit": eval_unit,
            "server_unit": server_unit, "evaluator_argv": argv, "server_argv": server_cmd}, indent=1))
        return 0

    lock_held = False
    server_log = None
    watchdog = early = None
    cancelled = threading.Event()
    starter_state = {}

    def start_server(env_extra):
        subprocess.Popen(["systemd-run", "--user", "--quiet", "--pipe", "--collect", f"--unit={server_unit}",
                          f"--working-directory={ROOT}", "-p", "CPUAffinity=12 13 14 15",
                          "-p", f"MemoryMax={'10G' if handoff else '4G'}", "-p", "MemorySwapMax=0",
                          "env", f"P2_TASK_ID={TASK_ID}", *env_extra, *server_cmd], stdout=server_log, stderr=subprocess.STDOUT)

    def deferred_start():
        """Handoff: the WAM loads only after the scene import (evaluator health request) and with measured headroom."""
        if not wait_for_policy_request(run_dir / "process.log", cancelled, min(args.timeout, 1500)):
            starter_state["result"] = "no_health_request"
            return
        ok = wait_for_policy_capacity(cancelled, "cuda", reserve_mib=args.wam_reserve_mib)
        from behavior_preflight import hardware_snapshot
        snap = hardware_snapshot()
        starter_state.update(vram_at_start_mib=snap["gpu"][0]["vram_used_mib"] if snap["gpu"] else None,
                             capacity_ok=bool(ok), time=utc())
        if not ok or cancelled.is_set():
            starter_state["result"] = "insufficient_vram_headroom_or_cancelled"
            subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", eval_unit], capture_output=True)
            return
        start_server(["OMP_NUM_THREADS=2"])
        starter_state["result"] = "wam_server_started"

    def _term(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, _term)
    rc = None
    starter = None
    try:
        waited = acquire_gpu_lock(args.run_name, run_dir / "gpu_lock_wait.jsonl", max_wait=args.max_lock_wait)
        lock_held = True
        manifest.update(status="running", gpu_lock_acquired_utc=utc(), gpu_lock_wait_s=round(waited))
        if TASK_ID != 13:
            from behavior_preflight import hardware_snapshot
            snap = hardware_snapshot()
            manifest["baseline_at_lock"] = {"vram_used_mib": snap["gpu"][0]["vram_used_mib"] if snap["gpu"] else None,
                                            "ram_used_percent": snap["memory_used_percent"],
                                            "ram_total_gib": round(snap["memory_total_bytes"] / 2**30, 2),
                                            "monotonic": time.monotonic()}
        write_json(run_dir / "manifest.json", manifest)
        server_log = (run_dir / "server.log").open("w")
        if handoff:
            starter = threading.Thread(target=deferred_start, daemon=True)
            starter.start()
            if TASK_ID == 13:
                early = EarlyStop(run_dir / "eval/object_joints", eval_unit, args.handoff_step, args.early_stop_jtrunk,
                                  args.early_stop_steps)
            else:
                early = (EarlyStopTask1(run_dir / "eval/object_poses", eval_unit, args.handoff_step, spec, args.early_stop_steps)
                         if args.early_stop_steps else None)   # Task 0: 0 = disabled
            if early is not None:
                early.start()
        else:
            start_server(["OMP_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES="])
            if not wait_health(args.port):
                raise RuntimeError("replay server did not become healthy")
        watchdog = Watchdog(eval_unit, run_dir / "watchdog.jsonl", vram_limit_mib=vram_limit_mib)
        watchdog.start()
        print("evaluator:", " ".join(argv), flush=True)
        rc = run_guarded(argv, child_environment(), run_dir, timeout=args.timeout)
    finally:
        cancelled.set()
        for th in (watchdog, early):
            if th is not None:
                th.stop.set()
        if starter is not None:
            starter.join(10)
        subprocess.run(["systemctl", "--user", "stop", server_unit], capture_output=True)
        if server_log:
            server_log.close()
        for _ in range(120):  # the lock is released only after the simulator unit is fully gone
            if not unit_active(eval_unit) and not unit_active(server_unit):
                break
            time.sleep(1)
        if lock_held:
            for _ in range(60):
                try:
                    if gpu_used_mib() < 2600:
                        break
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    break
                time.sleep(1)
            manifest["gpu_lock_released"] = release_gpu_lock(args.run_name)
            manifest["gpu_lock_released_utc"] = utc()
        try:
            res = json.loads((run_dir / "resource_summary.json").read_text())
            manifest["peak_vram_mib"] = max(res.get("peak_vram_mib", 0), watchdog.peaks["vram_mib"] if watchdog else 0)
        except (OSError, ValueError):
            manifest["peak_vram_mib"] = watchdog.peaks["vram_mib"] if watchdog else None
        manifest.update(evaluator_returncode=rc, watchdog_abort_reason=watchdog.reason if watchdog else None,
                        watchdog_peaks=watchdog.peaks if watchdog else None, finished_utc=utc(),
                        wam_start_result=starter_state or None,
                        early_stop_triggered=early.triggered if early else None,
                        status=("finished" if rc == 0 and not (watchdog and watchdog.reason)
                                and starter_state.get("result", "wam_server_started") == "wam_server_started"
                                else "failed_or_aborted"))
        if TASK_ID != 13:
            manifest["performance"] = run_performance(run_dir, manifest, watchdog)
            if handoff and early is not None and getattr(early, "error", None):
                manifest["early_stop_error"] = early.error
        try:
            plan = json.loads((run_dir / "eval/evaluation_plan.json").read_text())
            manifest["evaluation_plan"] = plan
            manifest["plan_matches_request"] = (plan.get("mode") == "train" and plan.get("instance_ids") == [args.instance]
                                                and plan.get("seed") == args.seed)
        except (OSError, ValueError) as e:
            manifest["evaluation_plan_error"] = str(e)
        write_json(run_dir / "manifest.json", manifest)
    ana = subprocess.run([str(BEHAVIOR6), str(Path(__file__).resolve()), "analyze", "--run-dir", str(run_dir)],
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"})
    manifest["analysis_returncode"] = ana.returncode
    write_json(run_dir / "manifest.json", manifest)
    return rc or 0


# ---------------------------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------------------------
def read_joint_log(path):
    header, rows, end = None, [], None
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "header":
                header = rec
            elif rec.get("type") == "end":
                end = rec
            else:
                rows.append(rec)
    return header, rows, end


def trunk_series(header, rows):
    inst = next(k for k in header["entities"] if k.startswith("car"))
    ent = header["entities"][inst]
    names = ent["joints"]
    idx = next((i for i, n in enumerate(names) if "trunk" in n.lower()), 0)
    lo, hi = ent["lower"][idx], ent["upper"][idx]
    steps = np.array([r["step"] for r in rows])
    vals = np.array([r["joints"][inst][idx] for r in rows], float)
    return {"entity": inst, "name": ent["name"], "joint": names[idx], "lower": lo, "upper": hi,
            "threshold": lo + OPEN_FRACTION * (hi - lo)}, steps, vals


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def demo_image(camera, episode, frame, size=None):
    from PIL import Image
    import io
    from harness.frame_cache import FramePack
    img = Image.open(io.BytesIO(FramePack(FRAMES / camera, episode).jpeg(frame))).convert("RGB")
    return img.resize((size, size)) if size else img


def gray_small(img, size=56):
    return np.asarray(img.convert("L").resize((size, size)), np.float64)


def segment_stats(values, segs, t_max):
    out = []
    for s in segs:
        a, b = int(s["start"]), min(int(s["end"]), t_max)
        if a >= b:
            continue
        v = values[a:b]
        v = v[np.isfinite(v)]
        if len(v):
            out.append({"skill": s["skill"], "start": a, "end": b, "median": round(float(np.median(v)), 4),
                        "p95": round(float(np.percentile(v, 95)), 4), "max": round(float(v.max()), 4)})
    return out


def make_plots(adir, j_info, j_steps, j_vals, segs, openlid, tr, success_step, handoff_step=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped: matplotlib unavailable in this interpreter (use .conda-envs/behavior6)"
    ink, ink2, band, band2, blue, orange = "#0b0b0b", "#52514e", "#ecebe7", "#d9d8d2", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.formatter.useoffset": False})
    files = []
    if j_info is not None:
        thr = j_info["threshold"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=130, gridspec_kw={"width_ratios": [2, 1]})
        lo, hi = openlid
        views = [(axes[0], None, "whole replay"), (axes[1], (max(0, lo - 300), hi + 300), "zoom: demo 'open lid' segment +/-300")]
        for ax, xlim, sub in views:
            ax.axvspan(lo, hi, color=band2, lw=0)
            ax.plot(j_steps, j_vals, color=blue, lw=1.5)
            ax.axhline(thr, color=ink2, lw=1, ls="--")
            ax.text(0.995, thr, f"Open threshold {thr:.4f} rad", transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", color=ink2, fontsize=8)
            ax.text((lo + hi) / 2, 0.97, "demo 'open lid'", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    color=ink2, fontsize=8)
            if handoff_step is not None:
                ax.axvline(handoff_step, color=orange, lw=1.2)
                ax.text(handoff_step, 0.5, f"WAM handoff (N={handoff_step}) ", transform=ax.get_xaxis_transform(), ha="right",
                        color=ink2, fontsize=8, rotation=90, va="center")
            if success_step is not None:
                ax.axvline(success_step, color=ink2, lw=1, ls=":")
                ax.text(success_step, 0.5, "evaluator terminated ", transform=ax.get_xaxis_transform(), ha="right",
                        color=ink2, fontsize=8, rotation=90, va="center")
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_ylim(min(-0.02, float(np.nanmin(j_vals)) - 0.02), max(thr * 1.6, float(np.nanmax(j_vals)) * 1.08))
            ax.set_xlabel("environment step (30 Hz = demo frame)")
            ax.set_title(sub, fontsize=8, color=ink2, loc="left")
            ax.grid(axis="y", color=band, lw=0.8)
        axes[0].set_ylabel(f"{j_info['joint']} [rad]")
        mode = (f"demo-prefix (t<{handoff_step}) -> WAM handoff" if handoff_step is not None
                else "open-loop demo-action replay")
        fig.suptitle(f"{j_info['name']} trunk joint (evaluator object-joint log) under {mode} "
                     "- diagnostic, not a policy result", fontsize=9, color=ink, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(adir / "j_trunk.png")
        plt.close(fig)
        files.append("j_trunk.png")
    if tr is not None:
        t = np.arange(len(tr["arm_left_rad"]))
        panels = [("Arm joints: max |sim - demo|", "rad", [("left arm", tr["arm_left_rad"], blue), ("right arm", tr["arm_right_rad"], orange)]),
                  ("Trunk joints: max |sim - demo|", "rad", [("trunk", tr["trunk_rad"], blue)]),
                  ("End-effector position (robot frame): |sim - demo|", "m", [("left eef", tr["eef_left_m"], blue), ("right eef", tr["eef_right_m"], orange)]),
                  ("Base velocity (vx, vy, wz): |sim - demo|", "m/s, rad/s", [("base qvel", tr["base_vel"], blue)]),
                  ("Odometry divergence |xy_sim - xy_demo| (both integrated from the instance start pose)", "m",
                   [("odometry", tr["odom_pos_m"], blue)])]
        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 10), dpi=130, sharex=True)
        for ax, (title, unit, series) in zip(axes, panels):
            for k, s in enumerate(segs):
                if s["skill"] == "open lid":
                    ax.axvspan(s["start"], s["end"], color=band2, lw=0)
                elif k % 2:
                    ax.axvspan(s["start"], s["end"], color=band, lw=0)
            for name, y, c in series:
                ax.plot(t, y, color=c, lw=1.0, label=name)
            if len(series) > 1:
                ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=len(series))
            ax.set_title(title, fontsize=8, color=ink, loc="left")
            ax.set_ylabel(unit, fontsize=8)
            ax.grid(axis="y", color=band, lw=0.8)
        axes[-1].set_xlabel("environment step (30 Hz = demo frame); shaded bands = demo skill segments, darker = 'open lid'")
        fig.suptitle("Replayed proprio vs recorded demo state - " + (f"demo prefix t<{handoff_step}, WAM afterwards (divergence after the handoff is expected)"
                     if handoff_step is not None else "open-loop demo-action replay") + " (diagnostic, not a policy result)",
                     fontsize=9, color=ink, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(adir / "tracking_error.png")
        plt.close(fig)
        files.append("tracking_error.png")
    return files


def montage(adir, run_dir, episode, steps_available):
    from PIL import Image, ImageDraw
    snap = run_dir / "replay/snapshots"
    out = []
    kdir = adir / "keyframes"
    kdir.mkdir(exist_ok=True)
    frames = sorted({int(p.name[8:13]) for p in snap.glob("sim_step*_head.png")})
    for f in frames:
        if f >= steps_available:
            continue
        canvas = Image.new("RGB", (3 * 224, 2 * 224 + 36), "white")
        for c, cam in enumerate(("head", "lwrist", "rwrist")):
            p = snap / f"sim_step{f:05d}_{cam}.png"
            if p.exists():
                canvas.paste(Image.open(p).convert("RGB").resize((224, 224)), (c * 224, 18))
            try:
                canvas.paste(demo_image(CAMS[cam], episode, f, 224), (c * 224, 18 + 224 + 18))
            except (KeyError, ValueError, OSError):
                pass
        d = ImageDraw.Draw(canvas)
        d.text((4, 3), f"SIM replay step {f} (head | left wrist | right wrist)", fill=(0, 0, 0))
        d.text((4, 224 + 21), f"DEMO episode {episode} frame {f}", fill=(0, 0, 0))
        name = f"keyframe_{f:05d}.png"
        canvas.save(kdir / name)
        out.append(f"keyframes/{name}")
    return out


def handoff_summary(run_dir, manifest, N, j_info, j_steps, j_vals, demo_state):
    h = {"handoff_step": N, "budget_steps": manifest.get("budget_steps")}
    if j_info is not None:
        thr = j_info["threshold"]
        post = j_steps >= N
        sp, jp = j_steps[post], j_vals[post]
        if len(jp):
            above = np.nonzero(jp > thr)[0]
            h.update(j_trunk_at_handoff=round(float(jp[0]), 4), j_trunk_max_after_handoff=round(float(jp.max()), 4),
                     j_trunk_max_step_after_handoff=int(sp[np.argmax(jp)]),
                     first_step_above_threshold_after_handoff=int(sp[above[0]]) if len(above) else None,
                     steps_from_handoff_to_open=int(sp[above[0]] - N) if len(above) else None,
                     steps_after_handoff_logged=int(len(jp) - 1), steps_with_j_trunk_ge_1=int((jp >= 1.0).sum()))
    rec = run_dir / "replay/handoff_record.json"
    h["handoff_record"] = json.loads(rec.read_text()) if rec.exists() else None
    cpath = run_dir / "replay/chunks.jsonl"
    chunks = [json.loads(line) for line in open(cpath)] if cpath.exists() else []
    if chunks:
        h["chunks"] = {"count": len(chunks), "infer_ms_mean": round(float(np.mean([c["infer_ms"] for c in chunks])), 1),
                       "first_chunk": {k: chunks[0][k] for k in ("step", "q_meas", "q_cmd_first", "q_cmd_end", "base_cmd",
                                                                 "base_cmd_mean", "grip", "demo_action_same_step_upper_body",
                                                                 "history_buffer_len", "proprio_frames")},
                       "base_cmd_abs_mean": [round(float(x), 4) for x in np.abs([c["base_cmd_mean"] for c in chunks]).mean(0)]}
    ppath = run_dir / "replay/proprio.npy"
    if ppath.exists():
        prop = np.load(ppath).astype(np.float64)
        rows = []
        for i, c in enumerate(chunks[:6]):
            end = c["step"] + 32  # observation after the chunk's 32 actions
            meas = prop[end, 28:32] if end < len(prop) else None
            tgt = np.asarray(c["q_cmd_end"][11:15])
            rows.append({"chunk": i + 1, "start_step": c["step"], "target_end_right_j0_j3": [round(float(x), 4) for x in tgt],
                         "measured_at_step": end,
                         "measured_right_j0_j3": None if meas is None else [round(float(x), 4) for x in meas],
                         "error": None if meas is None else [round(float(x), 4) for x in tgt - meas]})
        h["right_arm_chunk_end_target_vs_measured_first6"] = rows
        h["right_arm_j0_at_steps"] = {str(t): {"sim": round(float(prop[t, 28]), 4) if t < len(prop) else None,
                                               "demo": round(float(demo_state[t, 28]), 4)} for t in (800, 900, 1000)}
        if len(prop) > N:
            groups = {"trunk": slice(53, 57), "left_arm": slice(3, 10), "right_arm": slice(28, 35)}
            q_sim = prop[N:]
            q_demo = demo_state[N:N + len(q_sim)].astype(np.float64)
            h["upper_body_max_excursion_from_handoff_rad"] = {
                g: {"sim_wam": round(float(np.abs(q_sim[:, sl_] - q_sim[0, sl_]).max()), 4),
                    "demo_same_window": round(float(np.abs(q_demo[:, sl_] - q_demo[0, sl_]).max()), 4)}
                for g, sl_ in groups.items()}
            h["base_speed_mean_after_handoff"] = {"sim_wam": round(float(np.linalg.norm(q_sim[:, :2], axis=1).mean()), 4),
                                                  "demo_same_window": round(float(np.linalg.norm(q_demo[:, :2], axis=1).mean()), 4)}
            h["prefix_state_error_at_handoff"] = {
                "arm_left_rad": round(float(np.abs(prop[N, 3:10] - demo_state[N, 3:10]).max()), 4),
                "arm_right_rad": round(float(np.abs(prop[N, 28:35] - demo_state[N, 28:35]).max()), 4),
                "trunk_rad": round(float(np.abs(prop[N, 53:57] - demo_state[N, 53:57]).max()), 4)}
    return h


def cmd_analyze(args):
    if TASK_ID == 0:
        return analyze_task0(Path(args.run_dir))
    if TASK_ID != 13:
        return analyze_task1(Path(args.run_dir))
    run_dir = Path(args.run_dir)
    adir = run_dir / "analysis"
    adir.mkdir(exist_ok=True)
    ev = json.loads((run_dir / "mapping_evidence.json").read_text())
    mpath = run_dir / "manifest.json"
    run_manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    handoff = run_manifest.get("handoff_step")
    episode, instance = ev["episode"], ev["requested_train_instance"]
    segs = demo_segments(episode)
    openlid = openlid_segment(episode)
    demo_state = np.load(LOWDIM / f"episode_{episode}_state.npy")
    demo_act = np.load(LOWDIM / f"episode_{episode}_action.npy")
    summary = {**LABEL, "analysed_utc": utc(), "episode": episode, "train_instance": instance,
               "open_lid_segment_frames": list(openlid), "demo_outcome_in_dataset": ev.get("demo_outcome")}
    # evaluator result
    res_files = sorted((run_dir / "eval/json").glob("*.json"))
    result = json.loads(res_files[0].read_text()) if res_files else None
    summary["evaluator_result"] = None if result is None else {
        k: result.get(k) for k in ("task", "instance_id", "rollout_id", "steps", "success", "terminated", "truncated",
                                   "q_score", "time", "effective_policy_seed", "effective_evaluator_seed")}
    # object joint log
    jl = sorted((run_dir / "eval/object_joints").glob("*.jsonl"))
    j_info = j_steps = j_vals = None
    if jl:
        header, rows, end = read_joint_log(jl[0])
        j_info, j_steps, j_vals = trunk_series(header, rows)
        thr = j_info["threshold"]
        above = np.nonzero(j_vals > thr)[0]
        in_seg = (j_steps >= openlid[0]) & (j_steps < openlid[1])
        summary["j_trunk"] = {**{k: j_info[k] for k in ("entity", "name", "joint", "lower", "upper", "threshold")},
                              "rows": int(len(j_vals)), "end_record": end,
                              "max": round(float(j_vals.max()), 4), "max_step": int(j_steps[np.argmax(j_vals)]),
                              "first_step_above_threshold": int(j_steps[above[0]]) if len(above) else None,
                              "max_within_open_lid_segment": round(float(j_vals[in_seg].max()), 4) if in_seg.any() else None,
                              "value_at_open_lid_end": (round(float(j_vals[j_steps == openlid[1]][0]), 4)
                                                        if (j_steps == openlid[1]).any() else None),
                              "final": round(float(j_vals[-1]), 4)}
        np.savetxt(adir / "j_trunk.csv", np.stack([j_steps, j_vals], 1), delimiter=",", header="step,j_trunk_rad",
                   comments="", fmt=["%d", "%.6f"])
    # replayed proprio vs demo
    tr = None
    prop_path = run_dir / "replay/proprio.npy"
    if prop_path.exists():
        prop = np.load(prop_path).astype(np.float64)
        sent = np.load(run_dir / "replay/actions_sent.npy")
        T = min(len(prop), len(demo_state))
        p, d = prop[:T], demo_state[:T].astype(np.float64)
        status_path = run_dir / "replay/replay_status.json"
        served = json.loads(status_path.read_text()) if status_path.exists() else {}
        n_cmp = min(len(sent), len(demo_act), int(served.get("demo_steps_served", len(sent))))
        sent_matches = bool(np.array_equal(sent[:n_cmp], demo_act[:n_cmp].astype(np.float32)))
        inst = load_instance(instance)
        odo_sim = integrate_odometry(p[:, :3], inst["start"])[:T]
        odo_demo = integrate_odometry(d[:, :3], inst["start"])[:T]
        yaw_diff = np.degrees((odo_sim[:, 2] - odo_demo[:, 2] + np.pi) % (2 * np.pi) - np.pi)
        tr = {"arm_left_rad": np.abs(p[:, sl("arm_left_qpos")] - d[:, sl("arm_left_qpos")]).max(1),
              "arm_right_rad": np.abs(p[:, sl("arm_right_qpos")] - d[:, sl("arm_right_qpos")]).max(1),
              "trunk_rad": np.abs(p[:, sl("trunk_qpos")] - d[:, sl("trunk_qpos")]).max(1),
              "gripper_m": np.maximum(np.abs(p[:, sl("gripper_left_qpos")] - d[:, sl("gripper_left_qpos")]).max(1),
                                      np.abs(p[:, sl("gripper_right_qpos")] - d[:, sl("gripper_right_qpos")]).max(1)),
              "eef_left_m": np.linalg.norm(p[:, sl("eef_left_pos")] - d[:, sl("eef_left_pos")], axis=1),
              "eef_right_m": np.linalg.norm(p[:, sl("eef_right_pos")] - d[:, sl("eef_right_pos")], axis=1),
              "base_vel": np.linalg.norm(p[:, :3] - d[:, :3], axis=1),
              "odom_pos_m": np.linalg.norm(odo_sim[:, :2] - odo_demo[:, :2], axis=1),
              "odom_yaw_deg": np.abs(yaw_diff)}
        cols = list(tr)
        np.savetxt(adir / "tracking_error.csv", np.column_stack([np.arange(T)] + [tr[c] for c in cols]), delimiter=",",
                   header="step," + ",".join(cols), comments="", fmt="%.6f")
        def at(name, f):
            return round(float(tr[name][f]), 4) if f < T else None
        a, b = openlid
        win = slice(a, min(b, T))
        summary["tracking"] = {
            "steps_compared": T, "replayed_actions_equal_demo_actions": sent_matches, "demo_actions_compared": n_cmp,
            "steps_served": int(len(prop)), "hold_steps_served": int(max(0, len(prop) - n_cmp)),
            "definition": ("row t = observation before action t (evaluator proprio vs demo observation.state[t]); "
                           "odometry integrates robot-frame base velocity of each from the instance start pose"),
            "at_open_lid_start": {k: at(k, a) for k in cols},
            "open_lid_window_p95": {k: round(float(np.percentile(tr[k][win], 95)), 4) for k in cols} if a < T else None,
            "open_lid_window_max": {k: round(float(tr[k][win].max()), 4) for k in cols} if a < T else None,
            "whole_replay_p95": {k: round(float(np.percentile(tr[k], 95)), 4) for k in cols},
            "whole_replay_max": {k: round(float(tr[k].max()), 4) for k in cols},
            "at_end": {k: at(k, T - 1) for k in cols},
            "per_segment": {k: segment_stats(tr[k], segs, T) for k in ("arm_left_rad", "arm_right_rad", "trunk_rad",
                                                                          "base_vel", "odom_pos_m", "odom_yaw_deg")},
            "base_command_vs_measured_note": "demo base_qvel[t+1] ~= 0.75*action[t,0:2], 1.0*action[t,2] (controller output limits)"}
    # visual alignment (head camera, same frame index) and the step-0 instance check
    vis = {}
    head_dir = run_dir / "replay/head_1hz"
    if head_dir.exists():
        from PIL import Image
        pairs = []
        for pth in sorted(head_dir.glob("sim_step*.jpg")):
            f = int(pth.stem[8:])
            if f < len(demo_state) and f % 2 == 0:
                try:
                    pairs.append((f, ncc(gray_small(Image.open(pth)), gray_small(demo_image(CAMS["head"], episode, f)))))
                except (KeyError, ValueError, OSError):
                    pass
        if pairs:
            vis["head_ncc_same_frame"] = [(f, round(v, 3)) for f, v in pairs]
            vis["head_ncc_open_lid_window_mean"] = (round(float(np.mean([v for f, v in pairs if openlid[0] <= f < openlid[1]])), 3)
                                                    if any(openlid[0] <= f < openlid[1] for f, _ in pairs) else None)
        s0 = run_dir / "replay/snapshots/sim_step00000_head.png"
        if s0.exists():
            sim0 = gray_small(Image.open(s0))
            scores = []
            for ep in sorted(json.loads(SEGMENTS.read_text())["episodes"], key=int):
                try:
                    scores.append((ncc(sim0, gray_small(demo_image(CAMS["head"], int(ep), 0))), int(ep)))
                except (KeyError, ValueError, OSError):
                    pass
            scores.sort(reverse=True)
            rank = [e for _, e in scores].index(episode) + 1 if any(e == episode for _, e in scores) else None
            vis["step0_head_vs_all_demo_first_frames"] = {"rank_of_replayed_demo": rank, "of": len(scores),
                                                          "ncc_replayed_demo": next((round(s, 3) for s, e in scores if e == episode), None),
                                                          "top5": [(e, round(s, 3)) for s, e in scores[:5]]}
    summary["visual_alignment"] = vis or None
    success_step = None
    if result is not None and result.get("terminated"):
        success_step = int(result["steps"])
    if handoff is not None:
        summary.update(HANDOFF_LABEL)
        summary["handoff"] = handoff_summary(run_dir, run_manifest, handoff, j_info, j_steps, j_vals, demo_state)
    summary["plots"] = make_plots(adir, j_info, j_steps, j_vals, segs, openlid, tr, success_step, handoff)
    summary["keyframe_montages"] = montage(adir, run_dir, episode, len(demo_state)) if (run_dir / "replay/snapshots").exists() else []
    # verdict
    v = {}
    if "j_trunk" in summary:
        jt = summary["j_trunk"]
        v["a_trunk_exceeds_threshold"] = jt["first_step_above_threshold"] is not None
        v["a_first_step_above_threshold"] = jt["first_step_above_threshold"]
        v["a_j_trunk_max"], v["a_j_trunk_max_step"] = jt["max"], jt["max_step"]
        v["a_demo_open_lid_segment"] = list(openlid)
    if result is not None:
        n_demo = len(demo_state)
        v["b_steps_executed"] = result.get("steps")
        v["b_full_episode_replayed"] = bool(result.get("steps", 0) >= n_demo or result.get("terminated"))
        v["b_evaluator_success"] = result.get("success")
        v["b_evaluator_q_score"] = (result.get("q_score") or {}).get("final")
        v["b_demo_first_terminated_frame"] = (ev.get("demo_outcome") or {}).get("first_terminated_frame")
    if "j_trunk" in summary and not v.get("a_trunk_exceeds_threshold") and "tracking" in summary:
        t0 = summary["tracking"]["at_open_lid_start"]
        w = summary["tracking"]["open_lid_window_p95"] or {}
        drift = (t0["odom_pos_m"] is not None and (t0["odom_pos_m"] > DRIFT_POS_M or t0["odom_yaw_deg"] > DRIFT_YAW_DEG))
        arm_ok = max(w.get("arm_left_rad", 9), w.get("arm_right_rad", 9), w.get("trunk_rad", 9)) <= ARM_OK_RAD
        eef_ok = max(w.get("eef_left_m", 9), w.get("eef_right_m", 9)) <= EEF_OK_M
        if t0["odom_pos_m"] is None or not w:
            c = "inconclusive: replay logs do not reach the demo open-lid segment"
        elif drift:
            c = "drift: base odometry diverged from the demo before contact (open-lid start)"
        elif arm_ok and eef_ok:
            c = "physics: base on the demo path and arm/trunk/eef track the demo, but the lid does not lift"
        else:
            c = "physics/contact: base on the demo path at open-lid start, but upper-body tracking deviates during the contact window"
        v["c_attribution"] = c
        v["c_thresholds"] = {"drift_pos_m": DRIFT_POS_M, "drift_yaw_deg": DRIFT_YAW_DEG, "arm_ok_rad_p95": ARM_OK_RAD,
                             "eef_ok_m_p95": EEF_OK_M}
        v["c_evidence"] = {"at_open_lid_start": t0, "open_lid_window_p95": w}
    elif v.get("a_trunk_exceeds_threshold"):
        v["c_attribution"] = "not needed: the lid opened"
    if handoff is not None:  # replay questions (a)-(c) do not apply to a handoff run
        hs = summary["handoff"]
        v = {"wam_opened_lid_after_handoff": hs.get("first_step_above_threshold_after_handoff") is not None,
             **{k: hs.get(k) for k in ("handoff_step", "j_trunk_at_handoff", "j_trunk_max_after_handoff",
                                        "j_trunk_max_step_after_handoff", "first_step_above_threshold_after_handoff",
                                        "steps_from_handoff_to_open", "steps_after_handoff_logged")},
             "early_stop": run_manifest.get("early_stop_triggered"),
             "reference_full_demo_replay": "open-loop demo actions opened the lid at step 943 (run_ep2603_inst4_full)"}
    summary["verdict"] = v
    write_json(adir / "summary.json", summary)
    print(json.dumps({"verdict": v, "plots": summary["plots"]}, indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# Task 1 (picking_up_trash) analysis: evaluator result + evaluator-side pose log (--log-object-poses)
# ---------------------------------------------------------------------------------------------------------------
CV_FROM_USD = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes = USD camera axes rotated 180 deg about X
T1_LIFT_M, T1_FLOOR_M, T1_GEOM_TOL_M = 0.05, 0.03, 0.02
T1_COLORS = {"ink": "#0b0b0b", "ink2": "#52514e", "band": "#ecebe7", "band2": "#d9d8d2",
             "cans": ("#2a78d6", "#eb6834", "#1baf7a")}  # categorical slots 1-3 (all-pairs CVD-safe); bin = neutral ink


def read_log_tolerant(path):
    """header, rows, end, undecodable-line count (a killed evaluator can leave a partial last line)."""
    header, rows, end, bad = None, [], None, 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            kind = rec.get("type")
            if kind == "header":
                header = rec
            elif kind == "end":
                end = rec
            else:
                rows.append(rec)
    return header, rows, end, bad


def quat_mat(q):
    """Quaternion(s) [x, y, z, w] -> rotation matrix(es)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def project_world_points(points, base7, camrel7, K):
    """World points -> pixel (u, v) and OpenCV depth z for one camera: world <- robot base (pose-log robot_base)
    <- camera (cam_rel_poses entry, USD axes); OpenCV axes = USD axes @ diag(1, -1, -1)."""
    Rb, pb = quat_mat(base7[3:7]), np.asarray(base7[:3], float)
    Rc, pc = quat_mat(camrel7[3:7]), np.asarray(camrel7[:3], float)
    Rwc, pwc = Rb @ Rc @ CV_FROM_USD, pb + Rb @ pc
    xc = (np.atleast_2d(np.asarray(points, float)) - pwc) @ Rwc  # rows = Rwc^T (X - p)
    K = np.asarray(K, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * xc[:, 0] / xc[:, 2] + K[0, 2]
        v = K[1, 1] * xc[:, 1] / xc[:, 2] + K[1, 2]
    return u, v, xc[:, 2]


def t1_bin_geometry(info, bin7_0):
    """Bin frame geometry from the pose-log header: extents (native bbox x scale; fallback world AABB at reset),
    bbox centre offset from the root link in the bin frame, bottom/top heights in the bin frame."""
    R0, p0 = quat_mat(bin7_0[3:7]), np.asarray(bin7_0[:3], float)
    aabb = info.get("aabb_at_start_world")
    lo = hi = None
    if isinstance(aabb, list) and len(aabb) == 2 and all(isinstance(a, list) for a in aabb):
        lo, hi = np.asarray(aabb[0], float), np.asarray(aabb[1], float)
    nb, sc = info.get("native_bbox_object_frame"), info.get("scale")
    if isinstance(nb, list) and isinstance(sc, list):
        ext, source = np.asarray(nb, float) * np.asarray(sc, float), "native_bbox_object_frame x scale"
    elif lo is not None:
        ext, source = hi - lo, "world AABB at reset (yaw inflates x/y)"
    else:
        raise ValueError("pose-log header has no bin geometry")
    centre = R0.T @ ((lo + hi) / 2 - p0) if lo is not None else np.zeros(3)
    return {"extent_m": ext, "half_footprint_m": 0.5 * float(min(ext[0], ext[1])), "centre_in_bin_frame": centre,
            "bottom_z": float(centre[2] - ext[2] / 2), "top_z": float(centre[2] + ext[2] / 2), "source": source}


def t1_can_bin_relation(can7, bin7, geo):
    """Per step: horizontal distance from the bin axis (r), height above the bin bottom (h), both in the bin frame."""
    R = quat_mat(bin7[:, 3:7])
    rel = np.einsum("tji,tj->ti", R, can7[:, :3] - bin7[:, :3])
    r = np.linalg.norm(rel[:, :2] - geo["centre_in_bin_frame"][:2], axis=1)
    h = rel[:, 2] - geo["bottom_z"]
    H = geo["top_z"] - geo["bottom_z"]
    inside = (r < geo["half_footprint_m"]) & (h >= -T1_GEOM_TOL_M) & (h <= H + T1_GEOM_TOL_M)
    over = (r < geo["half_footprint_m"]) & (h > H + T1_GEOM_TOL_M)
    return rel, r, h, inside, over


def first_true(mask, steps):
    idx = np.nonzero(mask)[0]
    return int(steps[idx[0]]) if len(idx) else None


def t1_cam_rel_check(cam_rel, demo_ref, roles):
    """Live cam_rel_poses slots vs the demo's observation.robot2cam_pose cameras at the same frame index: which demo
    camera each slot matches (median position error) and the rotation convention (raw vs 180 deg about X)."""
    ref = demo_ref["robot2cam_pose"].astype(np.float64)
    names = [str(c) for c in demo_ref["cameras"]]
    T = min(len(cam_rel), len(ref))
    live = cam_rel[:T].astype(np.float64)
    ok = np.isfinite(live).all(axis=(1, 2))
    out = {"frames_compared": int(ok.sum()), "slots": {}}
    expected = {"left_wrist": "lwrist", "right_wrist": "rwrist", "head": "head"}
    flip = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion of 180 deg about X

    def qmul(a, b):
        ax, ay, az, aw = np.moveaxis(a, -1, 0)
        bx, by, bz, bw = np.moveaxis(b, -1, 0)
        return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)

    def ang(a, b):
        d = np.abs(np.sum(a / np.linalg.norm(a, axis=-1, keepdims=True) * b / np.linalg.norm(b, axis=-1, keepdims=True), -1))
        return np.degrees(2 * np.arccos(np.clip(d, 0, 1)))
    for i, role in enumerate(roles):
        errs = {names[j]: float(np.median(np.linalg.norm(live[ok, i, :3] - ref[:T][ok, j, :3], axis=1))) for j in range(ref.shape[1])}
        best = min(errs, key=errs.get)
        j = names.index(best)
        out["slots"][role] = {"slot": i, "best_matching_demo_camera": best, "median_pos_err_mm": round(1000 * errs[best], 1),
                              "median_pos_err_mm_all": {k: round(1000 * e, 1) for k, e in errs.items()},
                              "median_rot_err_deg_raw": round(float(np.median(ang(live[ok, i, 3:], ref[:T][ok, j, 3:]))), 2),
                              "median_rot_err_deg_after_x180": round(float(np.median(ang(qmul(live[ok, i, 3:], flip), ref[:T][ok, j, 3:]))), 2),
                              "matches_expected_role": best == expected.get(role)}
    out["order_verified"] = bool(out["slots"]) and all(s["matches_expected_role"] for s in out["slots"].values())
    return out


def t1_projection_check(run_dir, adir, steps, base, objs, cam_rel, roles, K, names, max_images=4):
    """Project the ground-truth object centres into the saved head frames with cam_rel_poses + the pose-log base pose
    and compare with the saved head depth at the projected pixel (evidence of frame/intrinsics conventions)."""
    from PIL import Image, ImageDraw
    head_i = roles.index("head") if "head" in roles else None
    nd = run_dir / "native_frames"
    caps = sorted(int(p.stem[5:]) for p in (nd / "head").glob("step_*.png")) if (nd / "head").exists() else []
    if head_i is None or not caps or cam_rel is None:
        return {"skipped": "no head captures / cam_rel_poses"}
    index = {int(s): k for k, s in enumerate(steps)}
    records, visible_steps = [], {}
    for t in caps:
        if t not in index or t >= len(cam_rel) or not np.isfinite(cam_rel[t, head_i]).all():
            continue
        k = index[t]
        pts = np.stack([objs[n][k, :3] for n in names])
        u, v, z = project_world_points(pts, base[k], cam_rel[t, head_i], K)
        dpath = nd / "head_depth" / f"step_{t:06d}.npy"
        depth = np.load(dpath).astype(np.float32) if dpath.exists() else None
        H, W = (depth.shape if depth is not None else (int(2 * K[1][2]), int(2 * K[0][2])))
        for n, uu, vv, zz in zip(names, u, v, z):
            if zz > 0.1 and 0 <= uu < W and 0 <= vv < H:
                d = float(depth[int(vv), int(uu)]) if depth is not None else None
                records.append({"step": t, "object": n, "u": round(float(uu), 1), "v": round(float(vv), 1),
                                "z_cv_m": round(float(zz), 3), "depth_at_pixel_m": None if d is None else round(d, 3),
                                "z_minus_depth_m": None if d is None or not np.isfinite(d) else round(float(zz) - d, 3)})
                visible_steps.setdefault(t, []).append(n)
    cans = [r["z_minus_depth_m"] for r in records if r["object"] != T1_BIN and r["z_minus_depth_m"] is not None]
    # overlays: first frame with the bin visible, first with a can visible, the frame with most objects, last capture
    picks = []
    for cond in (lambda ns: T1_BIN in ns, lambda ns: any(n != T1_BIN for n in ns)):
        t = next((t for t in sorted(visible_steps) if cond(visible_steps[t])), None)
        if t is not None:
            picks.append(t)
    if visible_steps:
        picks.append(max(visible_steps, key=lambda t: (len(visible_steps[t]), t)))
    picks.append(caps[-1])
    odir = adir / "gt_projection"
    odir.mkdir(exist_ok=True)
    files = []
    for t in sorted(set(picks))[:max_images]:
        img = Image.open(nd / "head" / f"step_{t:06d}.png").convert("RGB")
        draw = ImageDraw.Draw(img)
        for r in (r for r in records if r["step"] == t):
            c = T1_COLORS["cans"][T1_CANS.index(r["object"])] if r["object"] in T1_CANS else "#ffffff"  # bin: white
            x, y = r["u"], r["v"]
            draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=c, width=3)
            draw.line([x - 14, y, x + 14, y], fill=c, width=1)
            draw.line([x, y - 14, x, y + 14], fill=c, width=1)
            draw.text((x + 12, y - 22), f"{r['object'].split('.')[0]}{r['object'][-2:]} z={r['z_cv_m']}m", fill=c,
                      stroke_width=2, stroke_fill="#000000")
        draw.text((6, 6), f"step {t}: GT object centres projected with cam_rel_poses[head] + pose-log base + K", fill="#ffffff",
                  stroke_width=2, stroke_fill="#000000")
        name = f"gt_projection/step_{t:06d}.png"
        img.save(adir / name)
        files.append(name)
    return {"method": ("object root-link centres (pose log) -> head pixel via robot_base pose (pose log) and cam_rel_poses "
                       "head slot (USD axes -> OpenCV by diag(1,-1,-1)); K = pose-log rendered intrinsics if present"),
            "K": np.asarray(K, float).tolist(), "captures_checked": len(caps), "visible_projections": len(records),
            "cans_z_minus_depth_m": ({"n": len(cans), "median": round(float(np.median(cans)), 3),
                                      "p10": round(float(np.percentile(cans, 10)), 3), "p90": round(float(np.percentile(cans, 90)), 3),
                                      "expected": "~0 to +0.05 (can centre behind its visible surface) when unoccluded"}
                                     if cans else None),
            "overlays": files, "records_first_50": records[:50]}


def make_plots_task1(adir, steps, objs, base, lits, lit_names, segs, reward_frames, end_step, odo_demo, tr, names_short,
                     mode="open-loop demo-action replay", handoff_step=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped: matplotlib unavailable in this interpreter (use .conda-envs/behavior6)"
    C = T1_COLORS
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
                         "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.formatter.useoffset": False})
    files = []

    def bands(ax):
        for k, s in enumerate(segs):
            if k % 2:
                ax.axvspan(s["start"], s["end"], color=C["band"], lw=0)
    # 1) object heights
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=130)
    bands(ax)
    ax.plot(steps, objs[T1_BIN][:, 2], color=C["ink2"], lw=2, label="trash can (ashcan) root z")
    for i, c in enumerate(T1_CANS):
        if c in objs:
            ax.plot(steps, objs[c][:, 2], color=C["cans"][i], lw=1.5, label=f"{names_short[c]} root z")
            if lits is not None:
                k = next((j for j, n in enumerate(lit_names) if c in n), None)
                if k is not None and lits[:, k].any():
                    t0 = int(np.argmax(lits[:, k]))
                    ax.plot([steps[t0]], [objs[c][t0, 2]], "o", ms=8, mfc=C["cans"][i], mec="#fcfcfb", mew=2)
                    ax.annotate(f"inside (BDDL) @ {int(steps[t0])}", (steps[t0], objs[c][t0, 2]), xytext=(6, 10 + 14 * i),
                                textcoords="offset points", color=C["ink"], fontsize=8)
    for f in reward_frames or []:
        ax.axvline(f, color=C["ink2"], lw=0.9, ls=":")
    if reward_frames:
        ax.text(reward_frames[0], 0.02, "dotted: demo reward frames (can inside, dataset) ", transform=ax.get_xaxis_transform(),
                ha="right", va="bottom", color=C["ink2"], fontsize=8)
    ax.set_xlim(0, 1.02 * max([end_step, int(steps[-1])] + list(reward_frames or [])))
    ax.axvline(end_step, color=C["ink2"], lw=1, ls="--")
    ax.text(end_step, 0.5, " evaluator episode end", transform=ax.get_xaxis_transform(), rotation=90, va="center",
            ha="right", color=C["ink2"], fontsize=8)
    ax.set_xlabel("environment step (30 Hz = demo frame); shaded = alternate demo skill segments")
    ax.set_ylabel("height z [m] (world)")
    ax.grid(axis="y", color=C["band"], lw=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=4)
    if handoff_step is not None:
        ax.axvline(handoff_step, color=C["cans"][1], lw=1.2)
    ax.set_title(f"Task 1 {mode}: object heights from the evaluator pose log (diagnostic, not a policy result)",
                 fontsize=9, loc="left", color=C["ink"])
    fig.tight_layout()
    fig.savefig(adir / "task1_heights.png")
    plt.close(fig)
    files.append("task1_heights.png")
    # 2) top-down paths
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=130)
    ax.plot(base[:, 0], base[:, 1], color=C["ink"], lw=1.2, label="robot base (sim ground truth)")
    if odo_demo is not None:
        ax.plot(odo_demo[:, 0], odo_demo[:, 1], color=C["ink2"], lw=1.0, ls="--", label="demo odometry (from instance start)")
    ax.plot(objs[T1_BIN][:, 0], objs[T1_BIN][:, 1], color=C["ink2"], lw=3, alpha=0.55, label="trash can path")
    for i, c in enumerate(T1_CANS):
        if c in objs:
            ax.plot(objs[c][:, 0], objs[c][:, 1], color=C["cans"][i], lw=1.5, label=names_short[c])
            ax.plot(objs[c][0, 0], objs[c][0, 1], "o", ms=8, mfc=C["cans"][i], mec="#fcfcfb", mew=2)
            ax.plot(objs[c][-1, 0], objs[c][-1, 1], "s", ms=8, mfc=C["cans"][i], mec="#fcfcfb", mew=2)
    ax.plot(base[0, 0], base[0, 1], "o", ms=9, mfc=C["ink"], mec="#fcfcfb", mew=2)
    ax.plot(objs[T1_BIN][-1, 0], objs[T1_BIN][-1, 1], "s", ms=10, mfc=C["ink2"], mec="#fcfcfb", mew=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.grid(color=C["band"], lw=0.8)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.set_title("Top-down: circles = start, squares = end (evaluator pose log)", fontsize=9, loc="left", color=C["ink"])
    fig.tight_layout()
    fig.savefig(adir / "task1_topdown.png")
    plt.close(fig)
    files.append("task1_topdown.png")
    # 3) tracking vs demo
    if tr is not None:
        t = np.arange(len(tr["arm_left_rad"]))
        panels = [("Arm joints: max |sim - demo|", "rad", [("left arm", tr["arm_left_rad"], C["cans"][0]), ("right arm", tr["arm_right_rad"], C["cans"][1])]),
                  ("Trunk joints: max |sim - demo|", "rad", [("trunk", tr["trunk_rad"], C["cans"][0])]),
                  ("End-effector position (robot frame): |sim - demo|", "m", [("left eef", tr["eef_left_m"], C["cans"][0]), ("right eef", tr["eef_right_m"], C["cans"][1])]),
                  ("Base velocity (vx, vy, wz): |sim - demo|", "m/s, rad/s", [("base qvel", tr["base_vel"], C["cans"][0])]),
                  ("Base position vs demo odometry (both from the instance start pose)", "m",
                   [("sim odometry", tr["odom_pos_m"], C["cans"][0])] + ([("sim ground truth", tr["gt_vs_demo_odom_m"], C["cans"][1])]
                                                                         if "gt_vs_demo_odom_m" in tr else []))]
        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 10), dpi=130, sharex=True)
        for ax, (title, unit, series) in zip(axes, panels):
            bands(ax)
            for name, y, c in series:
                ax.plot(t[:len(y)], y, color=c, lw=1.0, label=name)
            if len(series) > 1:
                ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=len(series))
            ax.set_title(title, fontsize=8, color=C["ink"], loc="left")
            ax.set_ylabel(unit, fontsize=8)
            ax.grid(axis="y", color=C["band"], lw=0.8)
        axes[-1].set_xlabel("environment step (30 Hz = demo frame); shaded = alternate demo skill segments")
        fig.suptitle(f"Replayed proprio / base vs recorded demo - {mode} (diagnostic, not a policy result)",
                     fontsize=9, color=C["ink"], x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(adir / "tracking_error.png")
        plt.close(fig)
        files.append("tracking_error.png")
    return files


def t1_handoff_summary(run_dir, manifest, header, rows, steps, objs, geo, demo_state, demo_act, adir):
    """Task-1 handoff section: the pre-registered criterion (manifest handoff_criterion) on the evaluator pose log, the
    gripper commands actually sent after the handoff, and the robot's upper body / end effectors vs the demo over the
    demo segment. Evaluator-side evidence only (nothing here was ever a policy input)."""
    spec = manifest["handoff_criterion"]
    N, budget = int(manifest["handoff_step"]), int(manifest.get("budget_steps") or 0)
    a, e = (int(x) for x in spec["demo_segment"])
    crit = T1HandoffCriterion(header, spec)
    val = crit.values(rows)
    post = steps >= N
    last = int(steps[-1])
    early = manifest.get("early_stop_triggered")
    h = {"contract": spec["contract"], "skill": spec["skill"], "handoff_step": N, "budget_steps": budget, "demo_segment": [a, e],
         "target_object": {"name": spec["target_object"], "bddl_instance": crit.inst},
         "criterion": {"text": spec["criterion"], "kind": spec["kind"], "thresholds": spec["thresholds"],
                       "registered_utc": spec.get("registered_utc")},
         "last_logged_step": last, "steps_after_handoff_logged": max(0, last - N),
         "run_ended_by": ("early_stop" if early else "budget" if last >= N + budget else
                          f"other (manifest status {manifest.get('status')!r}, evaluator rc {manifest.get('evaluator_returncode')!r})"),
         "early_stop": early}
    at = {int(s): k for k, s in enumerate(steps)}
    k_N = at.get(N)
    tgt = objs[crit.inst]
    h["target_prefix_displacement_m"] = None if k_N is None else round(float(np.linalg.norm(tgt[k_N, :3] - tgt[0, :3])), 4)
    met, reason = None, "no pose-log rows after the handoff"
    if spec["kind"] == "lift" and post.any():
        vp, sp = val[post], steps[post]
        above = np.nonzero(vp >= T1_HANDOFF_LIFT_M)[0]
        first = int(sp[above[0]]) if len(above) else None
        dz = tgt[:, 2] - tgt[0, 2]
        h["lift"] = {"measure": spec.get("lift_measure"), "at_handoff_m": None if k_N is None else round(float(val[k_N]), 4),
                     "max_after_handoff_m": round(float(vp.max()), 4), "max_step": int(sp[np.argmax(vp)]),
                     f"first_step_ge_{T1_HANDOFF_LIFT_M}m": first, "steps_from_handoff_to_lift": None if first is None else first - N,
                     "at_last_logged_step_m": round(float(val[-1]), 4),
                     "root_link_dz_max_after_handoff_m": round(float(dz[post].max()), 4), "root_link_dz_at_end_m": round(float(dz[-1]), 4)}
        met = bool(first is not None and val[-1] >= T1_HANDOFF_END_M)
        reason = (f"max lift {vp.max():.3f} m (>= {T1_HANDOFF_LIFT_M} needed), end {val[-1]:.3f} m (>= {T1_HANDOFF_END_M} needed)")
    elif spec["kind"] == "inside" and post.any():
        vp, sp = val[post] >= 0.5, steps[post]
        first = int(sp[np.argmax(vp)]) if vp.any() else None
        rel, r, hb, inside_g, _ = t1_can_bin_relation(tgt, objs[T1_BIN], geo)
        lz = crit.lowest_z(tgt)  # all rows from step 0: the box corners are anchored at the step-0 pose
        h["inside"] = {"literal": crit.literal, "at_handoff": None if k_N is None else bool(val[k_N] >= 0.5),
                       "first_step_after_handoff": first, "steps_from_handoff": None if first is None else first - N,
                       "true_steps_after_handoff": int(vp.sum()), "transitions_after_handoff": int(np.count_nonzero(np.diff(vp.astype(int)))),
                       "at_last_logged_step": bool(val[-1] >= 0.5),
                       "final_relative_to_bin": {"dist_from_bin_axis_m": round(float(r[-1]), 3), "height_above_bin_bottom_m": round(float(hb[-1]), 3),
                                                 "geom_inside": bool(inside_g[-1])},
                       "target_lowest_point_above_step0_at_end_m": round(float(lz[-1] - lz[0]), 4)}
        met = bool(val[-1] >= 0.5)
        reason = f"inside literal at the last logged step ({last}): {bool(val[-1] >= 0.5)}"
    h["criterion"].update(met=met, reason=reason)
    if crit.inst != T1_BIN and T1_BIN in objs:  # the bin the other hand holds during can picks / places
        bname = header["entities"][T1_BIN].get("name")
        bc = T1HandoffCriterion(header, {"kind": "lift", "target_object": bname})
        bl = bc.values(rows)
        h["bin_side_effect"] = {"name": bname, "lowest_point_lift_at_handoff_m": None if k_N is None else round(float(bl[k_N]), 4),
                                "lowest_point_lift_at_end_m": round(float(bl[-1]), 4), "min_after_handoff_m": round(float(bl[post].min()), 4) if post.any() else None}
    # gripper commands actually sent (replay/actions_sent.npy) vs the demo segment
    sent_p, prop_p = run_dir / "replay/actions_sent.npy", run_dir / "replay/proprio.npy"
    sent = np.load(sent_p) if sent_p.exists() else None
    if sent is not None and len(sent) > N:
        grip = {}
        for hand, i in (("left", 14), ("right", 22)):
            c, d = sent[:, i].astype(np.float64), demo_act[:, i].astype(np.float64)
            w, prev = c[N:], float(c[N - 1]) if N else float("nan")
            dseg = d[a:e]
            grip[hand] = {"last_prefix_cmd": round(prev, 3), "state_at_handoff": "open" if prev >= 0 else "closed",
                          "first_wam_cmd": round(float(w[0]), 3), "wam_min": round(float(w.min()), 3), "wam_max": round(float(w.max()), 3),
                          "wam_closed_fraction": round(float((w < 0).mean()), 3),
                          "grasp_attempted": bool(prev >= 0 and (w < 0).any()),
                          "first_close_cmd_step": int(N + np.argmax(w < 0)) if prev >= 0 and (w < 0).any() else None,
                          "release_commanded": bool(prev < 0 and (w >= 0).any()),
                          "first_open_cmd_step": int(N + np.argmax(w >= 0)) if prev < 0 and (w >= 0).any() else None,
                          "demo_segment": {"start_cmd": round(float(dseg[0]), 3), "end_cmd": round(float(dseg[-1]), 3),
                                           "first_close_step": int(a + np.argmax(dseg < 0)) if dseg[0] >= 0 and (dseg < 0).any() else None,
                                           "first_open_step": int(a + np.argmax(dseg >= 0)) if dseg[0] < 0 and (dseg >= 0).any() else None}}
        h["gripper"] = grip
        h["grasp_attempted"] = {hand: g["grasp_attempted"] for hand, g in grip.items()}
        h["release_commanded"] = {hand: g["release_commanded"] for hand, g in grip.items()}
    cpath = run_dir / "replay/chunks.jsonl"
    chunks = [json.loads(line) for line in open(cpath)] if cpath.exists() else []
    if chunks:
        h["chunks"] = {"count": len(chunks), "infer_ms_mean": round(float(np.mean([c["infer_ms"] for c in chunks])), 1),
                       "exec_horizon": chunks[0].get("exec_horizon"),
                       "grip_first_cmd_by_chunk": [[c["step"], *c["grip"]] for c in chunks]}
    rec = run_dir / "replay/handoff_record.json"
    h["handoff_record"] = json.loads(rec.read_text()) if rec.exists() else None
    # upper body / end effectors vs the demo over the demo segment (hold baseline: stay at the handoff pose)
    series = None
    if prop_p.exists():
        prop = np.load(prop_p).astype(np.float64)
        hi = min(len(prop), e, len(demo_state))
        if len(prop) > N:
            h["prefix_state_error_at_handoff"] = {g: round(float(np.abs(prop[N, s0:s1] - demo_state[N, s0:s1]).max()), 4)
                                                  for g, (s0, s1) in (("arm_left_rad", (3, 10)), ("arm_right_rad", (28, 35)), ("trunk_rad", (53, 57)))}
        if hi > N:
            dem = demo_state[N:hi].astype(np.float64)
            sim = prop[N:hi]
            out = {"window": [N, hi], "definition": ("same step index; joints: mean |sim - demo| [rad]; eef: mean robot-frame distance "
                                                     "[m]; hold = the pose at the handoff kept constant (a do-nothing baseline)")}
            for g, (s0, s1) in (("trunk", (53, 57)), ("left_arm", (3, 10)), ("right_arm", (28, 35)),
                                ("left_eef_pos", (17, 20)), ("right_eef_pos", (42, 45))):
                ds, dh = sim[:, s0:s1] - dem[:, s0:s1], sim[:1, s0:s1] - dem[:, s0:s1]
                eef = g.endswith("eef_pos")
                err = np.linalg.norm(ds, axis=1) if eef else np.abs(ds).mean(1)
                herr = np.linalg.norm(dh, axis=1) if eef else np.abs(dh).mean(1)
                out[g] = {"sim_vs_demo_mean": round(float(err.mean()), 4), "hold_vs_demo_mean": round(float(herr.mean()), 4),
                          "sim_vs_demo_at_window_end": round(float(err[-1]), 4), "hold_vs_demo_at_window_end": round(float(herr[-1]), 4),
                          "sim_excursion_max": round(float(np.abs(sim[:, s0:s1] - sim[0, s0:s1]).max()), 4),
                          "demo_excursion_max": round(float(np.abs(dem[:, s0:s1] - dem[0, s0:s1]).max()), 4)}
            h["upper_body_vs_demo_over_segment"] = out
            series = (prop, sent)
        cols = ["step"] + [f"{k}_{s}{j}" for k in ("sim", "demo") for s, n in (("trunk", 4), ("larm", 7), ("rarm", 7), ("leef", 3), ("reef", 3))
                           for j in range(n)] + ["grip_left_sent", "grip_right_sent", "grip_left_demo", "grip_right_demo"]
        T = min(len(prop), len(demo_state), len(sent) if sent is not None else len(prop))
        if T > N:
            idx = np.r_[53:57, 3:10, 28:35, 17:20, 42:45]
            tab = np.column_stack([np.arange(N, T), prop[N:T][:, idx], demo_state[N:T][:, idx].astype(np.float64),
                                   (sent[N:T][:, [14, 22]] if sent is not None else np.full((T - N, 2), np.nan)),
                                   demo_act[N:T][:, [14, 22]].astype(np.float64)])
            np.savetxt(adir / "handoff_trajectories.csv", tab, delimiter=",", header=",".join(cols), comments="", fmt="%.5f")
            h["trajectories_csv"] = "analysis/handoff_trajectories.csv"
    h["plot"] = make_plot_handoff_task1(adir, h, steps, val, sent, demo_act, series, demo_state)
    return h


def make_plot_handoff_task1(adir, h, steps, val, sent, demo_act, series, demo_state):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped: matplotlib unavailable in this interpreter (use .conda-envs/behavior6)"
    C = T1_COLORS
    N, (a, e) = h["handoff_step"], h["demo_segment"]
    lo = max(0, N - 60)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), dpi=130, sharex=True)
    for ax in axes:
        ax.axvspan(a, e, color=C["band"], lw=0)
        ax.axvline(N, color=C["cans"][1], lw=1.2)
        ax.grid(axis="y", color=C["band"], lw=0.8)
    m = steps >= lo
    ax = axes[0]
    if h["criterion"]["kind"] == "lift":
        ax.plot(steps[m], val[m], color=C["cans"][0], lw=1.5, label=f"{h['target_object']['name']} lowest-point lift")
        for thr, txt in ((T1_HANDOFF_LIFT_M, "lift threshold"), (T1_HANDOFF_END_M, "end threshold")):
            ax.axhline(thr, color=C["ink2"], lw=1, ls="--")
            ax.text(0.995, thr, f"{txt} {thr} m", transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=C["ink2"], fontsize=8)
        ax.set_ylabel("m above step-0 height", fontsize=8)
    else:
        ax.step(steps[m], val[m], where="post", color=C["cans"][0], lw=1.5, label=f"BDDL inside({h['target_object']['name']}, bin)")
        ax.set_yticks([0, 1], ["false", "true"])
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    met = h["criterion"].get("met")
    ax.set_title(f"{h['contract']}: criterion {'MET' if met else 'not met' if met is not None else 'n/a'} "
                 f"({h['criterion'].get('reason')}); run ended by {h['run_ended_by']}", fontsize=8, loc="left", color=C["ink"])
    ax = axes[1]
    if sent is not None:
        t = np.arange(len(sent))
        mm = t >= lo
        for (hand, i), c in zip((("left", 14), ("right", 22)), (C["cans"][0], C["cans"][1])):
            ax.plot(t[mm], sent[mm, i], color=c, lw=1.4, label=f"{hand} gripper cmd sent (demo prefix, then WAM)")
            td = np.arange(lo, min(len(demo_act), int(t[-1]) + 1))
            ax.plot(td, demo_act[td, i], color=c, lw=1.0, ls=":", label=f"{hand} demo cmd")
    ax.set_ylabel("cmd (+1 open, -1 closed)", fontsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=7, ncol=2)
    ax = axes[2]
    if series is not None:
        prop, _ = series
        hi = min(len(prop), len(demo_state))
        for (g, s0), c in zip((("left eef", 17), ("right eef", 42)), (C["cans"][0], C["cans"][1])):
            tt = np.arange(N, hi)
            d_sim = np.linalg.norm(prop[N:hi, s0:s0 + 3] - demo_state[N:hi, s0:s0 + 3], axis=1)
            d_hold = np.linalg.norm(prop[N, s0:s0 + 3][None] - demo_state[N:hi, s0:s0 + 3], axis=1)
            ax.plot(tt, d_sim, color=c, lw=1.4, label=f"{g}: |sim - demo|")
            ax.plot(tt, d_hold, color=c, lw=1.0, ls="--", label=f"{g}: |hold at handoff - demo|")
    ax.set_ylabel("m (robot frame)", fontsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=7, ncol=2)
    axes[-1].set_xlabel("environment step (30 Hz = demo frame); band = demo segment; vertical line = handoff N")
    fig.suptitle("Task 1 demo-prefix -> WAM handoff (evaluator pose log; diagnostic, not a policy/task result)",
                 fontsize=9, color=C["ink"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(adir / "handoff_task1.png")
    plt.close(fig)
    return "handoff_task1.png"


def analyze_task1(run_dir):
    adir = run_dir / "analysis"
    adir.mkdir(exist_ok=True)
    ev = json.loads((run_dir / "mapping_evidence.json").read_text())
    if int(ev.get("task_id", 13)) != TASK_ID:
        raise SystemExit(f"run dir task {ev.get('task_id', 13)} differs from P2_TASK_ID={TASK_ID}")
    mpath = run_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    episode, instance = ev["episode"], ev["requested_train_instance"]
    segs = demo_segments(episode)
    demo_state = np.load(LOWDIM / f"episode_{episode}_state.npy")
    demo_act = np.load(LOWDIM / f"episode_{episode}_action.npy")
    reward_frames = (ev.get("demo_outcome") or {}).get("reward_frames")
    summary = {**LABEL, "analysed_utc": utc(), "task": TASK, "episode": episode, "train_instance": instance,
               "demo_outcome_in_dataset": ev.get("demo_outcome"), "demo_picks": ev.get("demo_picks")}
    res_files = sorted((run_dir / "eval/json").glob("*.json"))
    result = json.loads(res_files[0].read_text()) if res_files else None
    summary["evaluator_result"] = None if result is None else {
        k: result.get(k) for k in ("task", "instance_id", "rollout_id", "steps", "success", "terminated", "truncated",
                                   "q_score", "time", "effective_policy_seed", "effective_evaluator_seed")}
    inst = load_instance(instance)
    # ---- pose log
    pl = sorted((run_dir / "eval/object_poses").glob("*.jsonl"))
    objs = base = lits = steps = header = rows = None
    lit_names, geo, names_short, per_can = [], None, {}, {}
    anomalies = []
    if pl:
        header, rows, end, bad = read_log_tolerant(pl[0])
        steps = np.array([r["step"] for r in rows])
        base = np.array([r["robot_base"] for r in rows], np.float64)
        objs = {n: np.array([r["objects"][n] for r in rows], np.float64) for n in header["entities"]}
        opts = (header.get("goal_literals") or {}).get("options") or []
        if opts and "goal_literals" in rows[0]:
            lit_names = opts[0]
            lits = np.array([r["goal_literals"][0] for r in rows], bool)
        names_short = {n: header["entities"][n].get("name") or n for n in header["entities"]}
        summary["pose_log"] = {"file": str(pl[0].relative_to(run_dir)), "rows": len(rows), "undecodable_lines": bad,
                               "end_record": end, "entities": {n: header["entities"][n].get("name") for n in header["entities"]},
                               "goal_literals": lit_names, "cameras": header.get("cameras"),
                               "steps_contiguous": bool(len(steps) and np.array_equal(steps, np.arange(steps[0], steps[0] + len(steps))))}
        # instance check: step-0 poses equal the instance file
        d0 = {T1_BIN: float(np.linalg.norm(objs[T1_BIN][0, :3] - np.asarray(inst["bin_pose"][:3])))}
        d0.update({c: float(np.linalg.norm(objs[c][0, :3] - np.asarray(inst["cans"][c]))) for c in T1_CANS if c in objs})
        yaw0 = quat_yaw(base[0, 3:7])
        summary["instance_loaded_check"] = {
            "object_pos_diff_step0_vs_instance_file_m": {k: round(v, 4) for k, v in d0.items()},
            "robot_start_diff_m": round(float(np.hypot(base[0, 0] - inst["start"][0], base[0, 1] - inst["start"][1])), 4),
            "robot_start_yaw_diff_deg": round(math.degrees((yaw0 - inst["start"][2] + math.pi) % (2 * math.pi) - math.pi), 2),
            "matches": bool(max(d0.values()) < 0.05)}
        geo = t1_bin_geometry(header["entities"][T1_BIN], objs[T1_BIN][0])
        H = geo["top_z"] - geo["bottom_z"]
        binR = quat_mat(objs[T1_BIN][:, 3:7])
        tilt = np.degrees(np.arccos(np.clip(binR[:, 2, 2], -1, 1)))
        # can <-> demo pick/place segments by simulator object name
        seg_by_obj = {}
        for s in segs:
            if s["objects"]:
                seg_by_obj.setdefault(s["objects"][0][0], []).append([s["skill"], int(s["start"]), int(s["end"])])
        for i, c in enumerate(T1_CANS):
            if c not in objs:
                continue
            z = objs[c][:, 2]
            z0 = z[0]
            rel, r, h, inside_g, over_g = t1_can_bin_relation(objs[c], objs[T1_BIN], geo)
            k = next((j for j, n in enumerate(lit_names) if c in n), None)
            bddl = lits[:, k] if lits is not None and k is not None else None
            lifted = z > z0 + T1_LIFT_M
            lift_step = first_true(lifted, steps)
            drops = []
            if lift_step is not None:
                grounded = (z < z0 + T1_FLOOR_M) & ~inside_g & (~bddl if bddl is not None else True)
                after = steps > lift_step
                entering = grounded & after & ~np.r_[False, grounded[:-1]]
                drops = [{"step": int(steps[j]), "xy": [round(float(objs[c][j, 0]), 3), round(float(objs[c][j, 1]), 3)],
                          "dist_to_bin_axis_m": round(float(r[j]), 3)} for j in np.nonzero(entering)[0]]
            disp = np.linalg.norm(np.diff(objs[c][:, :3], axis=0), axis=1)
            if z.min() < -0.05:
                anomalies.append({"object": c, "type": "below_floor", "min_z": round(float(z.min()), 3), "step": int(steps[np.argmin(z)])})
            if len(disp) and disp.max() > 0.15:
                anomalies.append({"object": c, "type": "jump_gt_0.15m_per_step", "max_m": round(float(disp.max()), 3),
                                  "step": int(steps[np.argmax(disp) + 1])})
            per_can[c] = {
                "name": names_short.get(c), "demo_segments": seg_by_obj.get(names_short.get(c), []),
                "initial_pos": [round(float(x), 3) for x in objs[c][0, :3]], "final_pos": [round(float(x), 3) for x in objs[c][-1, :3]],
                "lifted_first_step": lift_step, "max_height_m": round(float(z.max()), 3),
                "bddl_inside_first_step": first_true(bddl, steps) if bddl is not None else None,
                "bddl_inside_final": bool(bddl[-1]) if bddl is not None else None,
                "bddl_inside_at_step0": bool(bddl[0]) if bddl is not None else None,
                "bddl_inside_transitions": int(np.count_nonzero(np.diff(bddl.astype(int)))) if bddl is not None else None,
                "geom_over_bin_footprint_first_step": first_true(over_g, steps),
                "geom_inside_bin_first_step": first_true(inside_g, steps),
                "final_relative_to_bin": {"bin_frame_xyz_from_root_m": [round(float(x), 3) for x in rel[-1]],
                                          "dist_from_bin_axis_m": round(float(r[-1]), 3),
                                          "height_above_bin_bottom_m": round(float(h[-1]), 3),
                                          "bin_height_m": round(float(H), 3), "bin_half_footprint_m": round(geo["half_footprint_m"], 3),
                                          "geom_inside": bool(inside_g[-1])},
                "drops_to_floor_outside_bin_after_lift": drops,
                "missed": (not bool(bddl[-1])) if bddl is not None else None}
        bz = objs[T1_BIN][:, 2]
        bdisp = np.linalg.norm(np.diff(objs[T1_BIN][:, :3], axis=0), axis=1)
        if bz.min() < -0.05:
            anomalies.append({"object": T1_BIN, "type": "below_floor", "min_z": round(float(bz.min()), 3)})
        if len(bdisp) and bdisp.max() > 0.15:
            anomalies.append({"object": T1_BIN, "type": "jump_gt_0.15m_per_step", "max_m": round(float(bdisp.max()), 3),
                              "step": int(steps[np.argmax(bdisp) + 1])})
        if tilt[-1] > 30:
            anomalies.append({"object": T1_BIN, "type": "bin_tipped_over_at_end", "tilt_deg": round(float(tilt[-1]), 1)})
        summary["bin"] = {"name": names_short.get(T1_BIN), "geometry": {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else
                                                                           (round(v, 4) if isinstance(v, float) else v)) for k, v in geo.items()},
                          "initial_pose": [round(float(x), 4) for x in objs[T1_BIN][0]],
                          "final_pose": [round(float(x), 4) for x in objs[T1_BIN][-1]],
                          "final_yaw_deg": round(math.degrees(quat_yaw(objs[T1_BIN][-1, 3:7])), 2),
                          "final_tilt_deg": round(float(tilt[-1]), 2), "max_tilt_deg": round(float(tilt.max()), 2),
                          "max_height_m": round(float(bz.max()), 3), "lifted_first_step": first_true(bz > bz[0] + T1_LIFT_M, steps),
                          "final_displacement_from_start_m": round(float(np.linalg.norm(objs[T1_BIN][-1, :3] - objs[T1_BIN][0, :3])), 3),
                          "demo_segments": seg_by_obj.get(names_short.get(T1_BIN), [])}
        summary["cans"] = per_can
        summary["physics_anomalies"] = anomalies
        # evaluator consistency with the logged goal literals
        if lits is not None and result is not None:
            fin, ini = lits[-1], lits[0]
            q_lit = float(np.sum(fin & ~ini) / len(fin))
            q_eval = (result.get("q_score") or {}).get("final")
            summary["evaluator_consistency"] = {
                "success_equals_all_literals_true_at_last_row": bool(result.get("success")) == bool(fin.all()),
                "q_from_literals": round(q_lit, 4), "q_evaluator": q_eval,
                "q_consistent": q_eval is not None and abs(q_lit - float(q_eval)) < 1e-6,
                "rows_equal_steps_plus_1": len(rows) == int(result.get("steps", -1)) + 1,
                "end_record_steps_equal_result_steps": bool(end) and end.get("environment_steps") == result.get("steps")}
        np.savetxt(adir / "object_heights.csv", np.column_stack([steps, objs[T1_BIN][:, 2]] + [objs[c][:, 2] for c in T1_CANS if c in objs]
                                                               + ([lits.astype(int)] if lits is not None else [])),
                   delimiter=",", header="step,bin_z," + ",".join(f"{c}_z" for c in T1_CANS if c in objs)
                   + ("," + ",".join(f"lit:{n}" for n in lit_names) if lits is not None else ""), comments="", fmt="%.5f")
    # ---- replayed proprio vs demo, base GT vs odometry
    tr = None
    odo_demo = None
    prop_path = run_dir / "replay/proprio.npy"
    if prop_path.exists():
        prop = np.load(prop_path).astype(np.float64)
        sent = np.load(run_dir / "replay/actions_sent.npy")
        T = min(len(prop), len(demo_state))
        p, d = prop[:T], demo_state[:T].astype(np.float64)
        status_path = run_dir / "replay/replay_status.json"
        served = json.loads(status_path.read_text()) if status_path.exists() else {}
        n_cmp = min(len(sent), len(demo_act), int(served.get("demo_steps_served", len(sent))))
        odo_sim = integrate_odometry(p[:, :3], inst["start"])[:T]
        odo_demo = integrate_odometry(d[:, :3], inst["start"])[:T]
        yaw_diff = np.degrees((odo_sim[:, 2] - odo_demo[:, 2] + np.pi) % (2 * np.pi) - np.pi)
        tr = {"arm_left_rad": np.abs(p[:, sl("arm_left_qpos")] - d[:, sl("arm_left_qpos")]).max(1),
              "arm_right_rad": np.abs(p[:, sl("arm_right_qpos")] - d[:, sl("arm_right_qpos")]).max(1),
              "trunk_rad": np.abs(p[:, sl("trunk_qpos")] - d[:, sl("trunk_qpos")]).max(1),
              "gripper_m": np.maximum(np.abs(p[:, sl("gripper_left_qpos")] - d[:, sl("gripper_left_qpos")]).max(1),
                                      np.abs(p[:, sl("gripper_right_qpos")] - d[:, sl("gripper_right_qpos")]).max(1)),
              "eef_left_m": np.linalg.norm(p[:, sl("eef_left_pos")] - d[:, sl("eef_left_pos")], axis=1),
              "eef_right_m": np.linalg.norm(p[:, sl("eef_right_pos")] - d[:, sl("eef_right_pos")], axis=1),
              "base_vel": np.linalg.norm(p[:, :3] - d[:, :3], axis=1),
              "odom_pos_m": np.linalg.norm(odo_sim[:, :2] - odo_demo[:, :2], axis=1),
              "odom_yaw_deg": np.abs(yaw_diff)}
        if base is not None:
            Tg = min(T, len(base))
            tr["gt_vs_demo_odom_m"] = np.linalg.norm(base[:Tg, :2] - odo_demo[:Tg, :2], axis=1)
            tr["gt_vs_sim_odom_m"] = np.linalg.norm(base[:Tg, :2] - odo_sim[:Tg, :2], axis=1)
        cols = list(tr)
        Tmin = min(len(v) for v in tr.values())
        np.savetxt(adir / "tracking_error.csv", np.column_stack([np.arange(Tmin)] + [tr[c][:Tmin] for c in cols]), delimiter=",",
                   header="step," + ",".join(cols), comments="", fmt="%.6f")

        def at(name, f):
            return round(float(tr[name][f]), 4) if f < len(tr[name]) else None
        summary["tracking"] = {
            "steps_compared": T, "replayed_actions_equal_demo_actions": bool(np.array_equal(sent[:n_cmp], demo_act[:n_cmp].astype(np.float32))),
            "demo_actions_compared": n_cmp, "steps_served": int(len(prop)),
            "definition": ("row t = observation before action t (evaluator proprio vs demo observation.state[t]); odometry "
                           "integrates robot-frame base velocity from the instance start pose; gt = evaluator pose-log base"),
            "at_segment_starts": [{"skill": s["skill"], "object": s["objects"][0][0] if s["objects"] else None, "start": int(s["start"]),
                                   **{k: at(k, int(s["start"])) for k in cols}} for s in segs if int(s["start"]) < T],
            "whole_replay_p95": {k: round(float(np.percentile(tr[k], 95)), 4) for k in cols},
            "whole_replay_max": {k: round(float(tr[k].max()), 4) for k in cols},
            "at_end": {k: at(k, T - 1) for k in cols},
            "per_segment": {k: segment_stats(tr[k], segs, T) for k in ("arm_left_rad", "arm_right_rad", "trunk_rad",
                                                                          "base_vel", "odom_pos_m", "odom_yaw_deg")}}
    # ---- cam_rel_poses: format / order vs the demo's robot2cam_pose, and GT projection into the head frames
    roles = [r for r, _ in camera_roles()]
    crp = run_dir / "native_frames/cam_rel_poses.npy"
    cam_rel = np.load(crp) if crp.exists() else None
    ref = run_dir / "demo_reference.npz"
    if cam_rel is not None:
        summary["cam_rel_poses"] = {"file": "native_frames/cam_rel_poses.npy", "shape": list(cam_rel.shape), "order_from_r1pro_yaml": roles,
                                    "format": "[x, y, z, qx, qy, qz, qw] per camera in the robot base frame (USD camera axes)",
                                    "finite_rows": int(np.isfinite(cam_rel).all(axis=(1, 2)).sum())}
        if ref.exists():
            summary["cam_rel_poses"]["vs_demo_robot2cam_pose"] = t1_cam_rel_check(cam_rel, np.load(ref), roles)
    if objs is not None and cam_rel is not None:
        cams = (summary.get("pose_log") or {}).get("cameras") or {}
        K = cams.get("head", {}).get("intrinsic_matrix")
        K = K if isinstance(K, list) else OFFICIAL_INTRINSICS["head"]
        summary["gt_projection_check"] = t1_projection_check(run_dir, adir, steps, base, objs, cam_rel, roles, K,
                                                             [T1_BIN, *[c for c in T1_CANS if c in objs]])
        summary["gt_projection_check"]["K_source"] = ("pose-log header cameras.head.intrinsic_matrix (rendered)"
                                                      if isinstance(cams.get("head", {}).get("intrinsic_matrix"), list)
                                                      else "eval_utils CAMERA_INTRINSICS (official)")
    # ---- visual alignment (head, same frame) + montages
    vis = {}
    head_dir = run_dir / "replay/head_1hz"
    if head_dir.exists():
        from PIL import Image
        pairs = []
        for pth in sorted(head_dir.glob("sim_step*.jpg")):
            f = int(pth.stem[8:])
            if f < len(demo_state) and f % 2 == 0:
                try:
                    pairs.append((f, ncc(gray_small(Image.open(pth)), gray_small(demo_image(CAMS["head"], episode, f)))))
                except (KeyError, ValueError, OSError):
                    pass
        if pairs:
            vis["head_ncc_same_frame_median"] = round(float(np.median([v for _, v in pairs])), 3)
            vis["head_ncc_same_frame"] = [(f, round(v, 3)) for f, v in pairs]
    summary["visual_alignment"] = vis or None
    end_step = int(result["steps"]) if result else (int(steps[-1]) if steps is not None else 0)
    hstep = manifest.get("handoff_step")
    mode = ("open-loop demo-action replay" if hstep is None else
            f"demo prefix t<{hstep} -> WAM '{(manifest.get('wam_config') or {}).get('contract')}' (divergence after the handoff is expected)")
    summary["plots"] = (make_plots_task1(adir, steps, objs, base, lits, lit_names, segs, reward_frames, end_step, odo_demo, tr, names_short,
                                         mode=mode, handoff_step=hstep)
                        if objs is not None else "skipped: no pose log")
    summary["keyframe_montages"] = montage(adir, run_dir, episode, len(demo_state)) if (run_dir / "replay/snapshots").exists() else []
    # ---- verdict
    v = {}
    if result is not None:
        v.update(a_evaluator_success=result.get("success"), a_q_score=(result.get("q_score") or {}).get("final"),
                 a_steps=result.get("steps"), a_terminated=result.get("terminated"), a_truncated=result.get("truncated"),
                 a_demo_length=len(demo_state), a_demo_can_inside_reward_frames=reward_frames)
    if "evaluator_consistency" in summary:
        v["a_evaluator_consistent_with_goal_literals"] = bool(summary["evaluator_consistency"]["q_consistent"]
                                                              and summary["evaluator_consistency"]["success_equals_all_literals_true_at_last_row"])
    if per_can:
        v["b_first_inside_step_bddl"] = {c: per_can[c]["bddl_inside_first_step"] for c in per_can}
        v["b_first_over_bin_footprint_step_geom"] = {c: per_can[c]["geom_over_bin_footprint_first_step"] for c in per_can}
        v["b_final_inside_bddl"] = {c: per_can[c]["bddl_inside_final"] for c in per_can}
        v["b_dropped"] = {c: len(per_can[c]["drops_to_floor_outside_bin_after_lift"]) for c in per_can}
        v["b_bin_final_pose"] = summary["bin"]["final_pose"]
        v["b_bin_final_tilt_deg"] = summary["bin"]["final_tilt_deg"]
        v["b_physics_anomalies"] = anomalies
        v["b_instance_loaded_matches_file"] = summary["instance_loaded_check"]["matches"]
    if result is not None and not result.get("success") and per_can and tr is not None:
        causes = {}
        for c, pc_ in per_can.items():
            if not pc_["missed"]:
                continue
            picks = [s for s in pc_["demo_segments"] if s[0] == "pick up from"]
            places = [s for s in pc_["demo_segments"] if s[0] == "place in"]
            ref_step = (picks[0][1] if picks and (pc_["lifted_first_step"] is None or pc_["drops_to_floor_outside_bin_after_lift"])
                        else places[0][1] if places else None)
            drift = (ref_step is not None and ref_step < len(tr["odom_pos_m"])
                     and (tr["odom_pos_m"][ref_step] > DRIFT_POS_M or tr["odom_yaw_deg"][ref_step] > DRIFT_YAW_DEG))
            kind = ("never lifted (grasp not achieved)" if pc_["lifted_first_step"] is None else
                    "dropped after lift" if pc_["drops_to_floor_outside_bin_after_lift"] else "carried but not inside at the end")
            causes[c] = {"what": kind, "reference_step": ref_step,
                         "attribution": ("drift: base odometry diverged from the demo before this skill" if drift else
                                         "physics/contact: base on the demo path at this skill; the object did not follow the demo"),
                         "odom_pos_m_at_ref": None if ref_step is None or ref_step >= len(tr["odom_pos_m"]) else round(float(tr["odom_pos_m"][ref_step]), 4),
                         "odom_yaw_deg_at_ref": None if ref_step is None or ref_step >= len(tr["odom_yaw_deg"]) else round(float(tr["odom_yaw_deg"][ref_step]), 2)}
        v["c_attribution"] = causes
        v["c_thresholds"] = {"drift_pos_m": DRIFT_POS_M, "drift_yaw_deg": DRIFT_YAW_DEG}
    elif result is not None and result.get("success"):
        v["c_attribution"] = "not needed: the evaluator reported success"
    handoff = manifest.get("handoff_step")
    if handoff is not None:  # replay questions (a)-(c) do not apply to a handoff run: the pre-registered criterion does
        summary.update(HANDOFF_LABEL)
        if objs is None or not manifest.get("handoff_criterion"):
            summary["handoff"] = {"error": "no pose log or no pre-registered handoff_criterion in the manifest"}
            v = {"criterion_met": None, "error": summary["handoff"]["error"]}
        else:
            hs = summary["handoff"] = t1_handoff_summary(run_dir, manifest, header, rows, steps, objs, geo, demo_state, demo_act, adir)
            v = {"criterion_met": hs["criterion"]["met"], "criterion": hs["criterion"]["text"], "reason": hs["criterion"]["reason"],
                 **{k: hs.get(k) for k in ("contract", "handoff_step", "budget_steps", "demo_segment", "last_logged_step",
                                           "run_ended_by", "lift", "inside", "grasp_attempted", "release_commanded", "bin_side_effect")},
                 "evaluator_result": summary.get("evaluator_result"), "physics_anomalies": anomalies,
                 "reference_full_demo_replay": ("run_ep389_inst290_rgbd (open-loop demo actions, Q=1.0): bin root +0.05 m at 1368, "
                                                "can_of_soda_113 root +0.05 m at 2635, inside at 2751")}
    summary["verdict"] = v
    write_json(adir / "summary.json", summary)
    print(json.dumps({"verdict": v, "plots": summary["plots"]}, indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# Task 0 (turning_on_radio) analysis: evaluator-side pose log only (radio pose, toggled_on literal, ToggledOn
# hand_in_marker_steps); never a policy input.
# ---------------------------------------------------------------------------------------------------------------
def t0_upper_body_vs_demo(prop, demo_state, N, hi):
    out = {"window": [N, hi], "definition": ("same step index; joints: mean |sim - demo| [rad]; eef: mean robot-frame distance [m]; "
                                             "hold = the pose at the handoff kept constant (a do-nothing baseline)")}
    dem, sim = demo_state[N:hi].astype(np.float64), prop[N:hi]
    for g, (s0, s1) in (("trunk", (53, 57)), ("left_arm", (3, 10)), ("right_arm", (28, 35)),
                        ("left_eef_pos", (17, 20)), ("right_eef_pos", (42, 45))):
        ds, dh = sim[:, s0:s1] - dem[:, s0:s1], sim[:1, s0:s1] - dem[:, s0:s1]
        eef = g.endswith("eef_pos")
        err = np.linalg.norm(ds, axis=1) if eef else np.abs(ds).mean(1)
        herr = np.linalg.norm(dh, axis=1) if eef else np.abs(dh).mean(1)
        out[g] = {"sim_vs_demo_mean": round(float(err.mean()), 4), "hold_vs_demo_mean": round(float(herr.mean()), 4),
                  "sim_vs_demo_at_window_end": round(float(err[-1]), 4), "hold_vs_demo_at_window_end": round(float(herr[-1]), 4)}
    # press geometry: left eef relative to the right (holding) eef, robot frame
    rel_sim = sim[:, 17:20] - sim[:, 42:45]
    rel_dem = dem[:, 17:20] - dem[:, 42:45]
    d = np.linalg.norm(rel_sim - rel_dem, axis=1)
    out["left_minus_right_eef"] = {"sim_vs_demo_mean_m": round(float(d.mean()), 4), "sim_vs_demo_min_m": round(float(d.min()), 4),
                                   "hold_vs_demo_mean_m": round(float(np.linalg.norm(rel_sim[:1] - rel_dem, axis=1).mean()), 4)}
    return out


def analyze_task0(run_dir):
    adir = run_dir / "analysis"
    adir.mkdir(exist_ok=True)
    ev = json.loads((run_dir / "mapping_evidence.json").read_text())
    if int(ev.get("task_id", 13)) != TASK_ID:
        raise SystemExit(f"run dir task {ev.get('task_id', 13)} differs from P2_TASK_ID={TASK_ID}")
    mpath = run_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    handoff = manifest.get("handoff_step") is not None
    episode, instance = ev["episode"], ev["requested_train_instance"]
    demo_state = np.load(LOWDIM / f"episode_{episode}_state.npy")
    demo_act = np.load(LOWDIM / f"episode_{episode}_action.npy")
    reward_frames = (ev.get("demo_outcome") or {}).get("reward_frames") or []
    summary = {**(HANDOFF_LABEL if handoff else LABEL), "analysed_utc": utc(), "task": TASK, "episode": episode,
               "train_instance": instance, "demo_outcome_in_dataset": ev.get("demo_outcome"), "demo_hands": ev.get("demo_hands")}
    res_files = sorted((run_dir / "eval/json").glob("*.json"))
    result = json.loads(res_files[0].read_text()) if res_files else None
    summary["evaluator_result"] = None if result is None else {
        k: result.get(k) for k in ("task", "instance_id", "rollout_id", "steps", "success", "terminated", "truncated",
                                   "q_score", "time", "effective_policy_seed", "effective_evaluator_seed")}
    q = None if result is None else (result.get("q_score") or {}).get("final")
    verdict = {"a_evaluator_success": None if result is None else bool(result.get("success")), "a_q_score": q}
    pl = sorted((run_dir / "eval/object_poses").glob("*.jsonl"))
    steps = tog = lift = marker = None
    if pl:
        header, rows, end, bad = read_log_tolerant(pl[0])
        steps = np.array([r["step"] for r in rows])
        base = np.array([r["robot_base"] for r in rows], np.float64)
        radio = np.array([r["objects"][T0_RADIO] for r in rows], np.float64)
        rname = header["entities"][T0_RADIO].get("name")
        tog_c = T1HandoffCriterion(header, {"kind": "toggled_on", "target_object": rname})
        lift_c = T1HandoffCriterion(header, {"kind": "lift", "target_object": rname})
        tog, lift, marker = tog_c.values(rows) >= 0.5, lift_c.values(rows), tog_c.marker_steps(rows)
        inst = load_instance(instance)
        d0 = float(np.linalg.norm(radio[0, :3] - np.asarray(inst["radio_pose"][:3])))
        yaw0 = quat_yaw(base[0, 3:7])
        summary["pose_log"] = {"file": str(pl[0].relative_to(run_dir)), "rows": len(rows), "undecodable_lines": bad, "end_record": end,
                               "radio_entity": {"bddl": T0_RADIO, "name": rname}, "goal_literal": tog_c.literal,
                               "object_states_header": header.get("object_states"),
                               "toggled_on_field_logged": bool(np.isfinite(marker).any()),
                               "steps_contiguous": bool(len(steps) and np.array_equal(steps, np.arange(steps[0], steps[0] + len(steps))))}
        summary["instance_loaded_check"] = {
            "radio_pos_diff_step0_vs_instance_file_m": round(d0, 4),
            "robot_start_diff_m": round(float(np.hypot(base[0, 0] - inst["start"][0], base[0, 1] - inst["start"][1])), 4),
            "robot_start_yaw_diff_deg": round(math.degrees((yaw0 - inst["start"][2] + math.pi) % (2 * math.pi) - math.pi), 2),
            "matches": bool(d0 < 0.05)}
        R = quat_mat(radio[:, 3:7])
        tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
        disp = np.linalg.norm(np.diff(radio[:, :3], axis=0), axis=1)
        anomalies = []
        if radio[:, 2].min() < -0.05:
            anomalies.append({"type": "below_floor", "min_z": round(float(radio[:, 2].min()), 3), "step": int(steps[np.argmin(radio[:, 2])])})
        if len(disp) and disp.max() > 0.15:
            anomalies.append({"type": "jump_gt_0.15m_per_step", "max_m": round(float(disp.max()), 3), "step": int(steps[np.argmax(disp) + 1])})
        first_on = first_true(tog, steps)
        flips = int(np.count_nonzero(np.diff(tog.astype(int))))
        demo_first = reward_frames[0] if reward_frames else None
        summary["radio"] = {"initial_pose": [round(float(x), 4) for x in radio[0]], "final_pose": [round(float(x), 4) for x in radio[-1]],
                            "lowest_point_lift_max_m": round(float(lift.max()), 4), "lowest_point_lift_max_step": int(steps[np.argmax(lift)]),
                            "lowest_point_lift_at_end_m": round(float(lift[-1]), 4), "tilt_at_end_deg": round(float(tilt[-1]), 1),
                            "tilt_max_deg": round(float(tilt.max()), 1)}
        summary["toggle"] = {"first_true_row_step": first_on, "true_at_step0": bool(tog[0]), "true_at_end": bool(tog[-1]),
                             "value_flips": flips, "turned_off_again": bool(first_on is not None and not tog[-1]),
                             "hand_in_marker_steps_max": None if not np.isfinite(marker).any() else int(np.nanmax(marker)),
                             "steps_with_hand_in_marker": None if not np.isfinite(marker).any() else int(np.nansum(marker > 0)),
                             "first_hand_in_marker_step": (first_true(np.nan_to_num(marker) > 0, steps) if np.isfinite(marker).any() else None),
                             "demo_first_reward_frame": demo_first,
                             "row_semantics": ("pose-log row t = state after action t-1, demo next.reward at frame f = state after action f; "
                                               "a perfect replay toggles at row f + 1"),
                             "row_minus_demo_reward_frame": None if first_on is None or demo_first is None else int(first_on - demo_first)}
        summary["physics_anomalies"] = anomalies
        verdict.update(a_first_toggled_on_row=first_on, a_demo_first_reward_frame=demo_first,
                       a_evaluator_consistent_with_goal_literal=(None if result is None else bool(bool(result.get("success")) == bool(tog[-1]))),
                       b_radio_lift_max_m=summary["radio"]["lowest_point_lift_max_m"], b_physics_anomalies=anomalies,
                       b_instance_loaded_matches_file=summary["instance_loaded_check"]["matches"])
    sent_p, prop_p = run_dir / "replay/actions_sent.npy", run_dir / "replay/proprio.npy"
    sent = np.load(sent_p) if sent_p.exists() else None
    prop = np.load(prop_p).astype(np.float64) if prop_p.exists() else None
    st_p = run_dir / "replay/replay_status.json"
    n_demo_served = json.loads(st_p.read_text()).get("demo_steps_served") if st_p.exists() else None
    if sent is not None and n_demo_served:
        n = min(int(n_demo_served), len(sent), len(demo_act))
        summary["tracking"] = {"demo_steps_served": int(n_demo_served),
                               "replayed_actions_equal_demo_actions": bool(np.array_equal(sent[:n], demo_act[:n].astype(np.float32)))}
        if prop is not None:
            m = min(n, len(prop), len(demo_state))
            summary["tracking"]["prefix_max_abs_err"] = {g: round(float(np.abs(prop[:m, s0:s1] - demo_state[:m, s0:s1]).max()), 4)
                                                         for g, (s0, s1) in (("arm_left_rad", (3, 10)), ("arm_right_rad", (28, 35)),
                                                                             ("trunk_rad", (53, 57)))} if m else None
    if handoff and steps is not None:
        spec = manifest["handoff_criterion"]
        N, budget = int(manifest["handoff_step"]), int(manifest.get("budget_steps") or 0)
        a, e = (int(x) for x in spec["demo_segment"])
        post = steps >= N
        last = int(steps[-1])
        at = {int(s_): k for k, s_ in enumerate(steps)}
        k_N = at.get(N)
        h = {"contract": spec["contract"], "skill": spec["skill"], "handoff_step": N, "budget_steps": budget, "demo_segment": [a, e],
             "criterion": {"text": spec["criterion"], "kind": spec["kind"], "thresholds": spec["thresholds"],
                           "registered_utc": spec.get("registered_utc")},
             "last_logged_step": last, "steps_after_handoff_logged": max(0, last - N),
             "run_ended_by": ("evaluator_success" if result is not None and result.get("success") else
                              "budget" if last >= N + budget else
                              f"other (manifest status {manifest.get('status')!r}, evaluator rc {manifest.get('evaluator_returncode')!r})"),
             "base_control": (manifest.get("wam_config") or {}).get("base_control"),
             "base_from_demo": (manifest.get("wam_config") or {}).get("base_from_demo")}
        met, reason = None, "no pose-log rows after the handoff"
        if spec["kind"] == "lift" and post.any():
            vp, sp = lift[post], steps[post]
            above = np.nonzero(vp >= T1_HANDOFF_LIFT_M)[0]
            first = int(sp[above[0]]) if len(above) else None
            h["lift"] = {"at_handoff_m": None if k_N is None else round(float(lift[k_N]), 4), "max_after_handoff_m": round(float(vp.max()), 4),
                         f"first_step_ge_{T1_HANDOFF_LIFT_M}m": first, "steps_from_handoff_to_lift": None if first is None else first - N,
                         "at_last_logged_step_m": round(float(lift[-1]), 4)}
            met = bool(first is not None and lift[-1] >= T1_HANDOFF_END_M)
            reason = f"max lift {vp.max():.3f} m (>= {T1_HANDOFF_LIFT_M} needed), end {lift[-1]:.3f} m (>= {T1_HANDOFF_END_M} needed)"
        elif spec["kind"] == "toggled_on" and post.any():
            on_at_N = bool(tog[k_N]) if k_N is not None else None
            vp, sp = tog[post], steps[post]
            within = sp <= N + budget
            first = int(sp[np.argmax(vp & within)]) if (vp & within).any() else None
            mk = marker[post]
            h["toggle"] = {"on_at_handoff": on_at_N, "first_step_on_after_handoff": first,
                           "steps_from_handoff": None if first is None else first - N,
                           "demo_first_reward_frame": spec.get("demo_first_reward_frame"),
                           "demo_steps_from_segment_start_to_reward": (None if spec.get("demo_first_reward_frame") is None
                                                                       else spec["demo_first_reward_frame"] - a),
                           "hand_in_marker_steps_max_after_handoff": None if not np.isfinite(mk).any() else int(np.nanmax(mk)),
                           "steps_with_hand_in_marker_after_handoff": None if not np.isfinite(mk).any() else int(np.nansum(mk > 0)),
                           "on_at_last_logged_step": bool(tog[-1])}
            met = bool(first is not None and on_at_N is False)
            reason = ("radio already on at the handoff (criterion undefined)" if on_at_N else
                      f"toggled_on first true at step {first} ({first - N} steps after N)" if first is not None else
                      f"toggled_on never true within N + {budget}")
        h["criterion"].update(met=met, reason=reason)
        h["radio_lowest_point_lift_at_end_m"] = round(float(lift[-1]), 4)
        if sent is not None and len(sent) > N:
            grip = {}
            for hand, i in (("left", 14), ("right", 22)):
                c, d = sent[:, i].astype(np.float64), demo_act[:, i].astype(np.float64)
                w, prev = c[N:], float(c[N - 1])
                dseg = d[a:e]
                grip[hand] = {"last_prefix_cmd": round(prev, 3), "first_wam_cmd": round(float(w[0]), 3),
                              "wam_closed_fraction": round(float((w < 0).mean()), 3),
                              "first_close_cmd_step": int(N + np.argmax(w < 0)) if prev >= 0 and (w < 0).any() else None,
                              "first_open_cmd_step": int(N + np.argmax(w >= 0)) if prev < 0 and (w >= 0).any() else None,
                              "demo_first_close_step": int(a + np.argmax(dseg < 0)) if dseg[0] >= 0 and (dseg < 0).any() else None,
                              "demo_first_open_step": int(a + np.argmax(dseg >= 0)) if dseg[0] < 0 and (dseg >= 0).any() else None}
            h["gripper"] = grip
            h["base_cmd_sent_abs_mean_after_handoff"] = [round(float(x), 4) for x in np.abs(sent[N:, :3]).mean(0)]
        if prop is not None and min(len(prop), e, len(demo_state)) > N:
            h["upper_body_vs_demo_over_segment"] = t0_upper_body_vs_demo(prop, demo_state, N, min(len(prop), e, len(demo_state)))
        cpath = run_dir / "replay/chunks.jsonl"
        chunks = [json.loads(line) for line in open(cpath)] if cpath.exists() else []
        if chunks:
            h["chunks"] = {"count": len(chunks), "infer_ms_mean": round(float(np.mean([c["infer_ms"] for c in chunks])), 1),
                           "exec_horizon": chunks[0].get("exec_horizon")}
        rec = run_dir / "replay/handoff_record.json"
        h["handoff_record"] = json.loads(rec.read_text()) if rec.exists() else None
        summary["handoff"] = h
        verdict.update(criterion_met=met, criterion_reason=reason)
    summary["verdict"] = verdict
    summary["plots"] = make_plot_task0(adir, summary, steps, lift, tog, marker, sent, demo_act, manifest)
    summary["interpretation"] = ("diagnostic on a TRAIN instance; evaluator-side evidence only; never a policy/skill/task success claim"
                                 if handoff else "open-loop demo-action replay: environment/evaluator validity only, not a policy result")
    write_json(adir / "summary.json", summary)
    print(json.dumps({"verdict": verdict}, indent=1, default=str))
    return 0


def make_plot_task0(adir, summary, steps, lift, tog, marker, sent, demo_act, manifest):
    if steps is None:
        return "skipped: no pose log"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped: matplotlib unavailable in this interpreter (use .conda-envs/behavior6)"
    C = T1_COLORS
    N = manifest.get("handoff_step")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=130, sharex=True)
    for ax in axes:
        if N is not None:
            ax.axvline(N, color=C["cans"][1], lw=1.2)
        ax.grid(axis="y", color=C["band"], lw=0.8)
    axes[0].plot(steps, lift, color=C["cans"][0], lw=1.5, label="radio lowest-point lift (m above step 0)")
    axes[0].axhline(T1_HANDOFF_LIFT_M, color=C["ink2"], lw=1, ls="--")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[1].step(steps, tog.astype(float), where="post", color=C["cans"][2], lw=1.5, label="toggled_on(radio) (BDDL goal literal)")
    if np.isfinite(marker).any():
        axes[1].plot(steps, marker / 5.0, color=C["cans"][0], lw=1.0, label="hand_in_marker_steps / 5")
    rf = (summary.get("demo_outcome_in_dataset") or {}).get("reward_frames") or []
    if rf:
        axes[1].axvline(rf[0] + 1, color=C["ink2"], lw=1, ls=":", label="demo first reward frame + 1")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    if sent is not None:
        t = np.arange(len(sent))
        for (hand, i), c in zip((("left", 14), ("right", 22)), (C["cans"][0], C["cans"][1])):
            axes[2].plot(t, sent[:, i], color=c, lw=1.3, label=f"{hand} gripper cmd sent")
            td = np.arange(min(len(demo_act), len(sent)))
            axes[2].plot(td, demo_act[td, i], color=c, lw=0.9, ls=":", label=f"{hand} demo cmd")
    axes[2].legend(loc="upper left", frameon=False, fontsize=7, ncol=2)
    axes[-1].set_xlabel("environment step (30 Hz = demo frame); vertical line = handoff N")
    v = summary.get("verdict", {})
    fig.suptitle(f"Task 0 {'handoff' if N is not None else 'demo replay'} (evaluator pose log; diagnostic): success={v.get('a_evaluator_success')} "
                 f"Q={v.get('a_q_score')} criterion_met={v.get('criterion_met')}", fontsize=9, color=C["ink"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(adir / "task0_radio.png")
    plt.close(fig)
    return "task0_radio.png"


def run_performance(run_dir, manifest, watchdog):
    """Task 1 run cost: simulator steps/s from the replay server's receive times (with and without the adaptive
    throttle delay), evaluator wall time, peak VRAM / RAM (system-wide; baseline at lock acquisition)."""
    out = {}
    try:
        t = np.load(run_dir / "replay/recv_time.npy")
        t = t[np.isfinite(t)]
        dpath = run_dir / "replay/throttle_delay_s.npy"
        delay = np.load(dpath) if dpath.exists() else None
        if len(t) > 1:
            span = float(t[-1] - t[0])
            dsum = float(np.nansum(delay[:len(t) - 1])) if delay is not None else 0.0
            out.update(steps_logged=int(len(t)), episode_wall_s=round(span, 1), sim_steps_per_s=round((len(t) - 1) / span, 2),
                       throttle_delay_total_s=round(dsum, 1),
                       sim_steps_per_s_excluding_throttle=round((len(t) - 1) / max(span - dsum, 1e-9), 2),
                       step_time_median_ms=round(1000 * float(np.median(np.diff(t))), 1))
        rs = json.loads((run_dir / "resource_summary.json").read_text())
        out["evaluator_process_wall_s"] = round(float(rs.get("elapsed_seconds", float("nan"))), 1)
        if len(t):
            import calendar
            start = calendar.timegm(time.strptime(rs["started_utc"], "%Y-%m-%dT%H:%M:%SZ"))
            out["evaluator_start_to_first_step_s"] = round(float(t[0]) - start, 1)
        out["peak_vram_mib_system"] = max(rs.get("peak_vram_mib", 0), watchdog.peaks["vram_mib"] if watchdog else 0)
        out["peak_ram_percent_system"] = max(rs.get("peak_ram_percent", 0), watchdog.peaks["ram"] if watchdog else 0)
        base = manifest.get("baseline_at_lock") or {}
        if base.get("ram_total_gib"):
            out["peak_ram_gib_system"] = round(out["peak_ram_percent_system"] * base["ram_total_gib"] / 100, 2)
        if base.get("vram_used_mib") is not None:
            out["vram_baseline_mib_at_lock"] = base["vram_used_mib"]
            out["peak_vram_mib_above_baseline"] = out["peak_vram_mib_system"] - base["vram_used_mib"]
        out["peak_gpu_util_percent"] = max(rs.get("peak_gpu_percent", 0), watchdog.peaks["gpu_util"] if watchdog else 0)
        out["peak_cpu_percent_system"] = max(rs.get("peak_cpu_percent", 0), watchdog.peaks["cpu"] if watchdog else 0)
    except (OSError, ValueError, KeyError) as e:
        out["error"] = repr(e)
    out["note"] = ("steps/s = (N-1)/(t_last - t_first) of observation receive times at the replay server (excludes scene "
                   "load); the adaptive throttle adds wall-clock delay only (physics unaffected); VRAM/RAM are system-wide "
                   "(nvidia-smi / host) and include the desktop")
    return out


# ---------------------------------------------------------------------------------------------------------------
# CPU self-test
# ---------------------------------------------------------------------------------------------------------------
def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cmd_self_test(args):
    if args.episode is None or args.instance is None:  # per-task defaults (Task 13: ep 2603 / instance 4)
        args.episode, args.instance = {13: (2603, 4), 1: (389, 290), 0: (61, 65)}[TASK_ID]
    if TASK_ID == 0:
        return self_test_task0(args)
    if TASK_ID != 13:
        return self_test_task1(args)
    import requests
    from websockets.sync.client import connect
    Packer, unpackb = _msgpack()
    report = {"self_test": "demo_action_replay CPU-only (no simulator, no GPU)", **LABEL}
    ep, inst = args.episode, args.instance
    # 1) mapping: established for the metadata instance, NOT for off-by-one
    ev = mapping_evidence(ep, inst)
    assert ev["established"], ev["checks"]
    bad = mapping_evidence(ep, inst - 1)
    assert not bad["established"] and not bad["checks"]["metadata_task_instance_id_equals_requested"]
    report["mapping"] = {"established": True, "off_by_one_rejected": True,
                         "population": ev["odometry_docking_check"]["population_all_200_demos"]}
    # 2) protocol round trip against fake observations
    demo_state = np.load(LOWDIM / f"episode_{ep}_state.npy")
    demo_act = np.load(LOWDIM / f"episode_{ep}_action.npy").astype(np.float32)
    n_serve, n_hold = 40, 6
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir = tmp / "run"
        (run_dir / "replay").mkdir(parents=True)
        policy = ReplayPolicy(ep, run_dir / "replay", steps=n_serve, seed=0, head_stride=30, keyframes=[0, 10, 38],
                              throttle=Throttle("fixed", fixed_ms=1.0), native_dir=run_dir / "native_frames",
                              native_steps=[0, 10, 20, 42])
        sent_images = {}
        port = free_port()
        stop, ready = threading.Event(), threading.Event()
        th = threading.Thread(target=lambda: asyncio.run(serve_async(policy, "127.0.0.1", port, stop, ready)), daemon=True)
        th.start()
        assert ready.wait(10)
        assert requests.get(f"http://127.0.0.1:{port}/healthz", timeout=2).ok
        packer = Packer()
        rng = np.random.default_rng(0)
        with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as ws:
            meta = unpackb(ws.recv())
            assert meta["policy"] == "demo_action_replay_diagnostic" and meta["policy_result"] is False
            assert meta["seed"] == 0 and meta["diagnostic"] == "demo_action_replay"
            ws.send(packer.pack({"reset": True}))  # evaluator: reset before load_task_instance
            ws.send(packer.pack({"reset": True}))  # evaluator: reset before the rollout
            got = []
            for t in range(n_serve + n_hold):
                imgs = {c: rng.integers(0, 256, (224, 224, 4), dtype=np.uint8) for c in LIVE_SUFFIX}
                sent_images[t] = imgs
                obs = {"robot_r1::proprio": demo_state[t].astype(np.float32),
                       **{f"robot_r1::robot_r1{sfx[:-5]}::rgb": imgs[c] for c, sfx in LIVE_SUFFIX.items()},
                       "robot_r1::cam_rel_poses": np.zeros(21, np.float32), "task_id": np.array([13])}
                ws.send(packer.pack(obs))
                resp = unpackb(ws.recv())
                assert "action" in resp and resp["action"].shape == (23,) and resp["action"].dtype == np.float32
                got.append(np.array(resp["action"]))
            ws.send(packer.pack({"status": True}))
            status = unpackb(ws.recv())
            assert status["step"] == n_serve + n_hold and status["episode"] == 1
        time.sleep(0.3)
        stop.set()
        th.join(5)
        got = np.stack(got)
        assert np.array_equal(got[:n_serve], demo_act[:n_serve]), "served actions differ from the demo"
        hold = demo_act[n_serve - 1].copy()
        hold[:3] = 0
        assert np.array_equal(got[n_serve:], np.repeat(hold[None], n_hold, 0)), "tail hold wrong"
        prop = np.load(run_dir / "replay/proprio.npy")
        assert prop.shape == (n_serve + n_hold, 61) and np.allclose(prop, demo_state[:n_serve + n_hold])
        assert (run_dir / "replay/snapshots/sim_step00010_head.png").exists()
        assert (run_dir / "replay/head_1hz/sim_step00030.jpg").exists()
        from PIL import Image
        nd = run_dir / "native_frames"
        for t in (0, 10, 20, 42):
            for c in LIVE_SUFFIX:
                back = np.asarray(Image.open(nd / c / f"step_{t:06d}.png"))
                assert back.shape == (224, 224, 4) and np.array_equal(back, sent_images[t][c]), f"native {c} {t} not lossless"
        assert not (nd / "head" / "step_000030.png").exists()
        fidx = np.load(nd / "frame_index.npy")
        assert np.array_equal(fidx, np.r_[np.arange(n_serve), -np.ones(n_hold, int)]), fidx
        assert np.array_equal(np.load(nd / "proprio.npy"), prop)
        meta = json.loads((nd / "capture_meta.json").read_text())
        assert meta["received_dtype"] == "uint8" and meta["received_shape"] == [224, 224, 4]
        caps = native_capture_steps(ep, 9296)
        assert caps[:20] == list(range(0, 200, 10)) and caps[20] == 698 and caps[-1] == 1708 and all(c % 2 == 0 for c in caps)
        report["native_frames"] = {"lossless_rgba_png_roundtrip": True, "frame_index_pairs_demo_frames": True,
                                   "hold_steps_marked_minus1": True, "capture_steps_2603": [caps[0], caps[19], caps[20], caps[-1], len(caps)]}
        report["protocol"] = {"health": True, "metadata_label": True, "two_resets_then_steps": True,
                              "actions_bitwise_equal_demo": True, "tail_hold": True, "status_rpc": True,
                              "proprio_logged": list(prop.shape), "snapshots": True}
        # 3) analysis on synthetic evaluator outputs (no simulator): lid opens at step 25
        write_json(run_dir / "mapping_evidence.json", ev)
        (run_dir / "eval/json").mkdir(parents=True)
        (run_dir / "eval/object_joints").mkdir(parents=True)
        write_json(run_dir / "eval/json/loading_the_car_4_0.json",
                   {"task": TASK, "instance_id": inst, "rollout_id": 0, "steps": n_serve + n_hold, "success": False,
                    "terminated": False, "truncated": True, "q_score": {"final": 0.0}})
        def joint_log(values):
            with open(run_dir / "eval/object_joints/loading_the_car_4_0.jsonl", "w") as f:
                f.write(json.dumps({"type": "header", "entities": {"car.n.01_1": {"name": "car_ssxsje_0", "joints": ["j_trunk"],
                                                                                   "lower": [0.0], "upper": [2.53]}}}) + "\n")
                for s, val in enumerate(values):
                    f.write(json.dumps({"step": s, "joints": {"car.n.01_1": [float(val)]}}) + "\n")
                f.write(json.dumps({"type": "end", "environment_steps": len(values) - 1}) + "\n")
        joint_log([0.0] * 25 + [0.3] * (n_serve + n_hold - 24))
        ns = argparse.Namespace(run_dir=str(run_dir))
        cmd_analyze(ns)
        s = json.loads((run_dir / "analysis/summary.json").read_text())
        assert s["verdict"]["a_trunk_exceeds_threshold"] and s["verdict"]["a_first_step_above_threshold"] == 25
        assert abs(s["j_trunk"]["threshold"] - 0.1265) < 1e-9
        assert s["tracking"]["replayed_actions_equal_demo_actions"] and s["tracking"]["whole_replay_max"]["arm_left_rad"] < 1e-6
        # 4) lid never opens + perfect tracking -> physics; perturbed base velocity -> drift (synthetic open-lid at frame 20)
        orig_openlid = globals()["openlid_segment"]
        globals()["openlid_segment"] = lambda e: (20, 40)
        try:
            joint_log([0.0] * (n_serve + n_hold + 1))
            cmd_analyze(ns)
            s = json.loads((run_dir / "analysis/summary.json").read_text())
            assert s["verdict"]["c_attribution"].startswith("physics:"), s["verdict"]
            prop2 = prop.copy()
            prop2[:, 0] += 0.3  # 0.3 m/s forward bias -> ~0.2 m by frame 20
            np.save(run_dir / "replay/proprio.npy", prop2)
            cmd_analyze(ns)
            s = json.loads((run_dir / "analysis/summary.json").read_text())
            assert s["verdict"]["c_attribution"].startswith("drift:"), s["verdict"]
        finally:
            globals()["openlid_segment"] = orig_openlid
        report["analysis"] = {"opened_detected_at_step": 25, "threshold_rad": 0.1265, "physics_vs_drift_rules": True}
        if args.plots_dir:
            import shutil
            dst = Path(args.plots_dir)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(run_dir, dst)
            report["synthetic_run_copy"] = str(dst)
    report["result"] = "PASS"
    print(json.dumps(report, indent=1))
    if args.report:
        write_json(Path(args.report), report)
    return 0


def self_test_task0(args):
    """Task 0 CPU-only self-test (no simulator, no GPU): paths, demo -> instance mapping, the evaluator pose-log
    toggled_on field (AST-executed evaluator methods on fakes), replay protocol round trip, analyze_task0 on synthetic
    replay logs (toggle / no toggle), handoff specs (pick / press), criteria on synthetic pose logs and the handoff
    analysis sections. Plumbing only, not a robot result."""
    import ast
    import requests
    from types import SimpleNamespace
    from unittest.mock import patch
    from websockets.sync.client import connect
    Packer, unpackb = _msgpack()
    ep, inst = args.episode, args.instance
    report = {"self_test": "Task 0 demo_action_replay / handoff CPU-only (no simulator, no GPU)", **LABEL, "episode": ep, "instance": inst}
    # 0) task_profile paths (Task 0 artifacts only)
    assert TASK == "turning_on_radio" and LOWDIM == ROOT / "data/lowdim_cache_task000" and SEGMENTS == DEMOS / "skill_segments_task000.json"
    assert SPLIT == DEMOS / "split_task_000.json" and EPISODES_META == DEMOS / "meta/episodes/chunk-000/file-000.parquet"
    assert OUT_ROOT == T1_HANDOFF_ROOT == ROOT / "outputs/validation/task0_handoff_20260927" and INST_DIR.is_dir()
    # 1) mapping: established for the metadata instance, rejected for id +/- 1
    ev = mapping_evidence(ep, inst)
    assert ev["established"], ev["checks"]
    for wrong in (inst - 1, inst + 1):
        bad = mapping_evidence(ep, wrong)
        assert not bad["established"] and not bad["checks"]["metadata_task_instance_id_equals_requested"]
    od = ev["odometry_docking_check"]
    report["mapping"] = {"established": True, "id_pm1_rejected": True, "this_episode": od["this_episode"],
                         "chance_instances_passing": od["train_instances_passing_for_this_episode"]["count"],
                         "population_train": od["population_train_demos"], "demo_hands": ev["demo_hands"]}
    # 2) evaluator pose log: toggled_on field [value, hand_in_marker_steps] only for ToggledOn objects
    tree = ast.parse((EVAL_UTILS.parents[1] / "eval.py").read_text())
    ns = {"argparse": argparse, "DEFAULT_EVAL_SEED": 0, "__doc__": "cpu test"}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args"],
                            type_ignores=[]), "eval.py", "exec"), ns)
    with patch.object(sys, "argv", ["eval.py", "--task-name", TASK]):
        assert ns["parse_args"]().log_object_poses is False
    Probe = _pose_log_probe()
    ToggledOn = type("ToggledOn", (), {})
    tstate = SimpleNamespace(value=False, robot_can_toggle_steps=0)

    class Ent:
        def __init__(self, name, pos, states=None):
            self.name, self.category, self.pos, self.quat = name, "thing", list(pos), [0.0, 0.0, 0.0, 1.0]
            self.aabb = (np.array(pos) - 0.1, np.array(pos) + 0.1)
            self.native_bbox, self.scale = np.array([0.3, 0.12, 0.2]), np.array([1.0, 1.0, 1.0])
            if states is not None:
                self.states = states

        def get_position_orientation(self):
            return np.array(self.pos), np.array(self.quat)
    pred = SimpleNamespace(body=["toggled_on", T0_RADIO], evaluate=lambda fn: bool(tstate.value))
    robot = Ent("robot_r1", [1.0, 2.0, 0.0])
    robot.sensors = {}
    radio, table = Ent("radio_89", [3.4, 4.6, 0.53], {ToggledOn: tstate}), Ent("coffee_table_koagbh_0", [3.5, 5.3, 0.36], {})
    probe = Probe()
    probe.robot, probe.robot_camera_names = robot, {}
    probe.env = SimpleNamespace(task=SimpleNamespace(object_scope={T0_RADIO: radio, "table.n.02_1": table, "agent.n.01_1": robot,
                                                                   "floor.n.01_1": Ent("floor", [0, 0, 0])},
                                                     ground_goal_state_options=[[pred]], _evaluate_predicate=lambda *a: None))
    probe._pose_log, probe._pose_log_entities, probe._pose_log_goal, probe._pose_log_step = None, {}, [], 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "object_poses" / f"{TASK}_{inst}_0.jsonl"
        path.parent.mkdir()
        probe.start_pose_log(str(path), episode_identity={"task": TASK, "instance_id": inst, "rollout_id": 0})
        tstate.robot_can_toggle_steps = 3
        probe._write_pose_log()
        tstate.value, tstate.robot_can_toggle_steps = True, 5
        probe._write_pose_log()
        probe.stop_pose_log()
        header, rows, end, badlines = read_log_tolerant(path)
    assert set(header["entities"]) == {T0_RADIO, "table.n.02_1"} and badlines == 0
    assert header["object_states"]["toggled_on"]["instances"] == [T0_RADIO]
    assert header["goal_literals"]["options"] == [[f"(toggled_on {T0_RADIO})"]]
    assert [r["toggled_on"][T0_RADIO] for r in rows] == [[False, 0], [False, 3], [True, 5]], rows
    assert [r["goal_literals"] for r in rows] == [[[False]], [[False]], [[True]]] and end == {"type": "end", "environment_steps": 2}
    # an evaluator run without ToggledOn objects keeps the previous row/header format (Task 1 / 13)
    probe.env.task.object_scope = {"table.n.02_1": table}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.jsonl"
        probe.start_pose_log(str(path))
        probe._write_pose_log()
        probe.stop_pose_log()
        h2, rows2, _, _ = read_log_tolerant(path)
    assert "object_states" not in h2 and all("toggled_on" not in r for r in rows2)
    report["evaluator_pose_log"] = {"toggled_on_value_and_hand_in_marker_steps_per_step": True, "absent_without_toggle_objects": True,
                                    "opt_in_default_off": True}
    # 3) protocol round trip (replay server) with perception capture every 10 steps
    demo_state = np.load(LOWDIM / f"episode_{ep}_state.npy")
    demo_act = np.load(LOWDIM / f"episode_{ep}_action.npy").astype(np.float32)
    n_serve, n_hold, every = 30, 4, 10
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        (run_dir / "replay").mkdir(parents=True)
        policy = ReplayPolicy(ep, run_dir / "replay", steps=n_serve, seed=0, head_stride=30, keyframes=[0, 10],
                              throttle=Throttle("fixed", fixed_ms=1.0), native_dir=run_dir / "native_frames", perception_every=every)
        port = free_port()
        stop, ready = threading.Event(), threading.Event()
        th = threading.Thread(target=lambda: asyncio.run(serve_async(policy, "127.0.0.1", port, stop, ready)), daemon=True)
        th.start()
        assert ready.wait(10) and requests.get(f"http://127.0.0.1:{port}/healthz", timeout=2).ok
        packer, got = Packer(), []
        with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as ws:
            meta = unpackb(ws.recv())
            assert meta["policy"] == "demo_action_replay_diagnostic" and meta["policy_result"] is False
            ws.send(packer.pack({"reset": True}))
            ws.send(packer.pack({"reset": True}))
            for t in range(n_serve + n_hold):
                img = rng.integers(0, 256, (224, 224, 4), dtype=np.uint8)
                obs = {"robot_r1::proprio": demo_state[t].astype(np.float32),
                       **{f"robot_r1::robot_r1{sfx}": img for sfx in LIVE_SUFFIX.values()},
                       "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": np.ones((224, 224), np.float32),
                       "robot_r1::cam_rel_poses": np.zeros(21, np.float32), "task_id": np.array([TASK_ID])}
                ws.send(packer.pack(obs))
                got.append(np.array(unpackb(ws.recv())["action"]))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            st = run_dir / "replay/replay_status.json"
            if st.exists() and json.loads(st.read_text()).get("final"):
                break
            time.sleep(0.1)
        stop.set()
        th.join(5)
        got = np.stack(got)
        assert np.array_equal(got[:n_serve], demo_act[:n_serve]), "served actions differ from the demo"
        hold = demo_act[n_serve - 1].copy()
        hold[:3] = 0
        assert np.array_equal(got[n_serve:], np.repeat(hold[None], n_hold, 0)), "tail hold wrong"
        assert (run_dir / "native_frames/head_depth/step_000020.npy").exists()
        report["protocol_capture"] = {"actions_bitwise_equal_demo": True, "tail_hold": True, "capture_every": every}
    # 4) analyze_task0 on synthetic replay logs (toggle at row T_on; then a run that never toggles)
    instd = load_instance(inst)
    segs = demo_segments(ep)
    rname = segs[1]["objects"][0][0]
    rf = ev["demo_outcome"]["reward_frames"][0]
    hdr = {"type": "header", "privileged_data_use": "evaluator-side evidence only; never policy input",
           "entities": {T0_RADIO: {"name": rname, "aabb_at_start_world": [[p - 0.1 for p in instd["radio_pose"][:3]],
                                                                           [p + 0.1 for p in instd["radio_pose"][:3]]],
                                   "native_bbox_object_frame": [0.3, 0.12, 0.2], "scale": [1.0, 1.0, 1.0]}},
           "goal_literals": {"options": [[f"(toggled_on {T0_RADIO})"]]},
           "object_states": {"toggled_on": {"instances": [T0_RADIO]}}, "cameras": {}}

    def pose_rows(last, on_from=None, lift_fn=None, on_until=None):
        out = []
        for t in range(last + 1):
            p = list(instd["radio_pose"])
            if lift_fn is not None:
                p[2] += lift_fn(t)
            on = on_from is not None and t >= on_from and (on_until is None or t <= on_until)
            mk = 0 if on_from is None else (min(5, max(0, t - (on_from - 5))) if t <= on_from else 0)
            out.append({"step": t, "robot_base": [instd["start"][0], instd["start"][1], 0, 0, 0, 0, 1],
                        "objects": {T0_RADIO: p}, "goal_literals": [[on]], "toggled_on": {T0_RADIO: [on, mk]}})
        return out

    def write_run(rd, rows_, success, manifest_, sent=None, prop=None, served=None, end=True):
        (rd / "replay").mkdir(parents=True)
        (rd / "eval/object_poses").mkdir(parents=True)
        (rd / "eval/json").mkdir(parents=True)
        write_json(rd / "mapping_evidence.json", ev)
        write_json(rd / "manifest.json", manifest_)
        last = rows_[-1]["step"]
        with open(rd / f"eval/object_poses/{TASK}_{inst}_0.jsonl", "w") as f:
            for r in [hdr, *rows_] + ([{"type": "end", "environment_steps": last}] if end else []):
                f.write(json.dumps(r) + "\n")
        write_json(rd / f"eval/json/{TASK}_{inst}_0.json", {"task": TASK, "instance_id": inst, "rollout_id": 0, "steps": last,
                                                            "success": success, "terminated": success, "truncated": not success,
                                                            "q_score": {"final": 1.0 if success else 0.0}})
        np.save(rd / "replay/actions_sent.npy", demo_act[:last] if sent is None else sent)
        np.save(rd / "replay/proprio.npy", (demo_state[:last] if prop is None else prop).astype(np.float32))
        write_json(rd / "replay/replay_status.json", {"demo_steps_served": last if served is None else served, "steps_served": last})
    ana = {}
    with tempfile.TemporaryDirectory(prefix="p2_t0_ana_") as tmp:
        rd = Path(tmp) / "replay_ok"
        write_run(rd, pose_rows(rf + 1, on_from=rf + 1), True, {"task_id": TASK_ID})
        with patch("builtins.print"):
            analyze_task0(rd)
        s = json.loads((rd / "analysis/summary.json").read_text())
        v = s["verdict"]
        assert v["a_evaluator_success"] and v["a_q_score"] == 1.0 and v["a_first_toggled_on_row"] == rf + 1, v
        assert v["a_evaluator_consistent_with_goal_literal"] and s["toggle"]["row_minus_demo_reward_frame"] == 1
        assert s["toggle"]["hand_in_marker_steps_max"] == 5 and s["tracking"]["replayed_actions_equal_demo_actions"]
        assert s["instance_loaded_check"]["matches"] and v["b_physics_anomalies"] == []
        rd2 = Path(tmp) / "replay_fail"
        write_run(rd2, pose_rows(rf + 300), False, {"task_id": TASK_ID})
        with patch("builtins.print"):
            analyze_task0(rd2)
        v2 = json.loads((rd2 / "analysis/summary.json").read_text())["verdict"]
        assert not v2["a_evaluator_success"] and v2["a_first_toggled_on_row"] is None and v2["a_evaluator_consistent_with_goal_literal"]
        ana["replay_success_case"] = {"first_toggled_on_row": v["a_first_toggled_on_row"], "demo_reward_frame": rf, "plots": s["plots"]}
        ana["replay_no_toggle_case"] = {"success": v2["a_evaluator_success"], "consistent": v2["a_evaluator_consistent_with_goal_literal"]}
    # 5) handoff specs: pick (lift) / press (toggled_on); mismatches and place-on refused
    pick = next(sg for sg in segs if sg["skill"] == "pick up from")
    press = next(sg for sg in segs if sg["skill"] == "press")
    place = next(sg for sg in segs if sg["skill"] == "place on")
    sp_p = handoff_spec(ep, pick["start"], "pick up from(radio,coffee_table)")
    sp_t = handoff_spec(ep, press["start"], "press(radio)")
    assert sp_p == handoff_spec(ep, pick["start"]) and sp_p["kind"] == "lift" and sp_t["kind"] == "toggled_on"
    assert sp_p["budget_steps"] == 2 * (pick["end"] - pick["start"]) and sp_t["budget_steps"] == 2 * (press["end"] - press["start"])
    assert sp_t["demo_toggle_in_segment"] and sp_t["demo_first_reward_frame"] == rf
    if ep == 61:
        assert (sp_p["handoff_step"], sp_p["budget_steps"], sp_t["handoff_step"], sp_t["budget_steps"], rf) == (650, 1396, 1348, 576, 1573)
        assert sp_p["demo_gripper_commands"]["right"]["closes_in_segment"] and sp_t["demo_gripper_commands"]["left"]["closes_in_segment"]
    for bad in ((pick["start"], "press(radio)"), (press["start"], "pick up from(radio,coffee_table)"),
                (place["start"], "place on(radio,coffee_table)"), (pick["start"] - 1, "pick up from(radio,coffee_table)")):
        try:
            handoff_spec(ep, *bad)
            raise AssertionError(f"handoff accepted: {bad}")
        except ValueError:
            pass
    # 6) criteria on synthetic pose logs + analyze_task0 handoff sections
    Np, Nt = sp_p["handoff_step"], sp_t["handoff_step"]
    lift = lambda t0, cap: (lambda t: 0.03 * min(max(t - t0 + 1, 0), cap))  # noqa: E731  0.12 m at t0 + 3
    cases = {"p_lift_held": (sp_p, pose_rows(Np + 200, lift_fn=lift(Np + 50, 6)), False, True, Np + 53),
             "p_lift_then_drop": (sp_p, pose_rows(Np + 200, lift_fn=lambda t: lift(Np + 50, 6)(t) if t < Np + 150 else 0.0), False, False, Np + 53),
             "t_toggle": (sp_t, pose_rows(Nt + 120, on_from=Nt + 120), True, True, Nt + 120),
             "t_no_toggle": (sp_t, pose_rows(Nt + sp_t["budget_steps"]), False, False, None),
             "t_on_before_handoff": (sp_t, pose_rows(Nt + 40, on_from=Nt - 10), True, False, Nt)}
    results = {}
    for name, (sp, rows_, success, met, first) in cases.items():
        crit = T1HandoffCriterion(hdr, sp)
        val = crit.values(rows_)
        results[name] = {"max_value": round(float(val.max()), 4)}
        with tempfile.TemporaryDirectory(prefix="p2_t0_hana_") as tmp:
            rd = Path(tmp) / "run"
            N, last = sp["handoff_step"], rows_[-1]["step"]
            sent = demo_act[:last].copy()
            if sp["kind"] == "toggled_on":
                sent[N:, 14] = 1.0
                sent[N + 25:, 14] = -1.0      # left fist closes 25 steps after the handoff
            write_run(rd, rows_, success, {"task_id": TASK_ID, "handoff_step": N, "budget_steps": sp["budget_steps"], "handoff_criterion": sp,
                                           "wam_config": {"contract": sp["contract"], "base_control": T0_BASE_CONTROL, "base_from_demo": False},
                                           "status": "finished", "evaluator_returncode": 0}, sent=sent, served=N)
            with patch("builtins.print"):
                analyze_task0(rd)
            s = json.loads((rd / "analysis/summary.json").read_text())
            h = s["handoff"]
            assert s["diagnostic"] == "demo_prefix_handoff" and h["criterion"]["met"] == met, (name, h["criterion"])
            if sp["kind"] == "lift":
                assert h["lift"][f"first_step_ge_{T1_HANDOFF_LIFT_M}m"] == first, (name, h["lift"])
                assert h["run_ended_by"] == "budget" or last < N + sp["budget_steps"]
            else:
                assert h["toggle"]["first_step_on_after_handoff"] == first, (name, h["toggle"])
                assert h["gripper"]["left"]["first_close_cmd_step"] == N + 25
                if name == "t_toggle":
                    assert h["run_ended_by"] == "evaluator_success" and h["toggle"]["steps_from_handoff"] == 120
                    assert h["toggle"]["hand_in_marker_steps_max_after_handoff"] == 5
            assert h["upper_body_vs_demo_over_segment"]["left_arm"]["sim_vs_demo_mean"] == 0.0  # sim = demo here
            results[name].update(met=h["criterion"]["met"], reason=h["criterion"]["reason"], run_ended_by=h["run_ended_by"])
    es = EarlyStopTask1("/nonexistent", "none", Nt, sp_t, 5)
    assert es.rule().startswith(f"BDDL toggled_on({sp_t['target_object']})")
    fired = next((r["step"] for r in [hdr, *pose_rows(Nt + 130, on_from=Nt + 120)] if es.feed(r)), None)
    assert fired == Nt + 124, fired
    report["analysis"] = ana
    report["handoff"] = {"specs": {k: {x: sp[x] for x in ("contract", "handoff_step", "budget_steps", "target_object", "kind")}
                                   for k, sp in (("p", sp_p), ("t", sp_t))}, "mismatch_and_place_on_refused": True,
                         "criteria_and_analysis": results}
    report["result"] = "PASS"
    print(json.dumps(report, indent=1, default=str))
    if args.report:
        write_json(Path(args.report), report)
    return 0


def _mat_quat(R):
    """Rotation matrix -> quaternion [x, y, z, w] (test helper)."""
    R = np.asarray(R, float)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def _pose_log_probe():
    """The evaluator's pose-log methods, executed from evaluator.py source (no simulator import)."""
    import ast
    src = EVAL_UTILS.parents[1] / "evaluator.py"
    cls = next(n for n in ast.parse(src.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Evaluator")
    names = {"_pose7", "_floats", "_goal_literal", "_pose_log_entity_info", "_pose_log_camera_info", "start_pose_log",
             "_write_pose_log", "stop_pose_log"}
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == names, "evaluator.py pose-log methods missing"
    mod = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="PoseLogProbe", bases=[], keywords=[], body=body,
                                                                  decorator_list=[])], type_ignores=[]))
    ns = {"os": os, "json": json, "np": np, "is_system_bddl_inst": lambda inst: inst.startswith("water")}
    exec(compile(mod, str(src), "exec"), ns)
    return ns["PoseLogProbe"]


def self_test_task1(args):
    import ast
    import requests
    from types import SimpleNamespace
    from unittest.mock import patch
    from websockets.sync.client import connect
    from PIL import Image
    Packer, unpackb = _msgpack()
    ep, inst = args.episode, args.instance
    report = {"self_test": "Task 1 demo_action_replay CPU-only (no simulator, no GPU)", **LABEL, "episode": ep, "instance": inst}
    # 0) task_profile paths (Task 1 artifacts only; Task 13 paths untouched)
    assert TASK == "picking_up_trash" and LOWDIM == ROOT / "data/lowdim_cache_task001" and SEGMENTS == DEMOS / "skill_segments_task001.json"
    assert SPLIT == DEMOS / "split_task_001.json" and EPISODES_META == DEMOS / "meta/episodes/chunk-001/file-000.parquet"
    assert OUT_ROOT == ROOT / "outputs/validation/task1_demo_replay_20260926" and INST_DIR.is_dir()
    assert [r for r, _ in camera_roles()] == ["left_wrist", "right_wrist", "head"]
    # 1) mapping: established for the metadata instance, rejected for id +/- 1
    ev = mapping_evidence(ep, inst)
    assert ev["established"], ev["checks"]
    for wrong in (inst - 1, inst + 1):
        bad = mapping_evidence(ep, wrong)
        assert not bad["established"] and not bad["checks"]["metadata_task_instance_id_equals_requested"]
    od = ev["odometry_docking_check"]
    report["mapping"] = {"established": True, "id_pm1_rejected": True, "this_episode": od["this_episode"],
                         "chance_instances_passing": od["train_instances_passing_for_this_episode"]["count"],
                         "population_train": od["population_train_demos"], "demo_picks": ev["demo_picks"]}
    # 2) evaluator: --log-object-poses is opt-in (default off); pose-log methods on fake entities
    tree = ast.parse((EVAL_UTILS.parents[1] / "eval.py").read_text())
    ns = {"argparse": argparse, "DEFAULT_EVAL_SEED": 0, "__doc__": "cpu test"}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args"],
                            type_ignores=[]), "eval.py", "exec"), ns)
    with patch.object(sys, "argv", ["eval.py", "--task-name", TASK]):
        a = ns["parse_args"]()
    assert a.log_object_poses is False and a.log_object_joints is False
    with patch.object(sys, "argv", ["eval.py", "--task-name", TASK, "--log-object-poses"]):
        assert ns["parse_args"]().log_object_poses is True
    Probe = _pose_log_probe()

    class Ent:
        def __init__(self, name, pos, category="thing"):
            self.name, self.category, self.pos, self.quat = name, category, list(pos), [0.0, 0.0, 0.0, 1.0]
            self.aabb = (np.array(pos) - 0.1, np.array(pos) + 0.1)
            self.native_bbox, self.scale = np.array([0.2, 0.2, 0.3]), np.array([1.0, 1.0, 1.0])

        def get_position_orientation(self):
            return np.array(self.pos), np.array(self.quat)
    state = {"in": False}
    pred = SimpleNamespace(body=["inside", "can__of__soda.n.01_1", "ashcan.n.01_1"], evaluate=lambda fn: state["in"])
    robot = Ent("robot_r1", [1.0, 2.0, 0.0], "agent")
    robot.sensors = {"robot_r1:zed_link:Camera:0": SimpleNamespace(image_height=720, image_width=720, horizontal_aperture=40.0,
                                                                   focal_length=17.0, modalities={"rgb", "depth_linear"},
                                                                   intrinsic_matrix=np.array([[306.0, 0, 360], [0, 306, 360], [0, 0, 1]]))}
    can, bin_ = Ent("can_of_soda_1", [2.0, 3.0, 0.07]), Ent("trash_can_1", [4.0, 1.0, 0.13])
    probe = Probe()
    probe.robot, probe.robot_camera_names = robot, {"head": "robot_r1::robot_r1:zed_link:Camera:0"}
    probe.env = SimpleNamespace(task=SimpleNamespace(
        object_scope={"ashcan.n.01_1": bin_, "can__of__soda.n.01_1": can, "floor.n.01_1": Ent("floor", [0, 0, 0]),
                      "agent.n.01_1": robot, "water.n.06_1": Ent("water", [0, 0, 0]), "none.n.01_1": None},
        ground_goal_state_options=[[pred]], _evaluate_predicate=lambda *a: None))
    probe._pose_log, probe._pose_log_entities, probe._pose_log_goal, probe._pose_log_step = None, {}, [], 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "object_poses" / f"{TASK}_{inst}_0.jsonl"
        path.parent.mkdir()
        probe.start_pose_log(str(path), episode_identity={"task": TASK, "instance_id": inst, "rollout_id": 0})
        can.pos, state["in"] = [4.0, 1.0, 0.2], True
        probe._write_pose_log()
        probe.stop_pose_log()
        header, rows, end, badlines = read_log_tolerant(path)
        try:
            probe.start_pose_log(str(path))
            raise AssertionError("pose log overwrite not refused")
        except FileExistsError:
            pass
    assert set(header["entities"]) == {"ashcan.n.01_1", "can__of__soda.n.01_1"}, header["entities"]  # no floor/agent/system
    assert header["privileged_data_use"] == "evaluator-side evidence only; never policy input" and badlines == 0
    assert header["goal_literals"]["options"] == [["(inside can__of__soda.n.01_1 ashcan.n.01_1)"]]
    assert header["cameras"]["cam_rel_poses_order"] == ["head"] and header["cameras"]["head"]["intrinsic_matrix"][0][0] == 306
    assert header["entities"]["ashcan.n.01_1"]["native_bbox_object_frame"] == [0.2, 0.2, 0.3]
    assert [r["step"] for r in rows] == [0, 1] and rows[0]["goal_literals"] == [[False]] and rows[1]["goal_literals"] == [[True]]
    assert rows[1]["objects"]["can__of__soda.n.01_1"] == [4.0, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0] and rows[0]["robot_base"][:3] == [1.0, 2.0, 0.0]
    assert end == {"type": "end", "environment_steps": 1}
    report["evaluator_pose_log"] = {"opt_in_default_off": True, "skips_floor_agent_system_none": True, "header_rows_end": True,
                                    "goal_literals_per_step": True, "refuses_overwrite": True}
    # 3) projection math on a known camera: base at origin, camera 1 m up looking along world +x (USD axes)
    Rc = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # columns: USD x (right) = -y, y (up) = +z, z (back) = -x
    cam7 = np.r_[[0.0, 0.0, 1.0], _mat_quat(Rc)]
    u, v, z = project_world_points([[2, 0, 1], [2, -0.5, 1], [2, 0, 1.5]], np.r_[[0.0, 0.0, 0.0], [0, 0, 0, 1.0]], cam7,
                                   OFFICIAL_INTRINSICS["head"])
    assert np.allclose([u[0], v[0], z[0]], [360, 360, 2]) and u[1] > 360 and abs(v[1] - 360) < 1e-6 and v[2] < 360
    report["projection_math"] = {"centre": [round(float(u[0]), 3), round(float(v[0]), 3)], "right_point_u": round(float(u[1]), 1),
                                 "up_point_v": round(float(v[2]), 1)}
    # 4) protocol round trip at the official RGB-D resolution with perception capture every 10 steps
    demo_state = np.load(LOWDIM / f"episode_{ep}_state.npy")
    demo_act = np.load(LOWDIM / f"episode_{ep}_action.npy").astype(np.float32)
    n_serve, n_hold, every = 40, 6, 10
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        (run_dir / "replay").mkdir(parents=True)
        policy = ReplayPolicy(ep, run_dir / "replay", steps=n_serve, seed=0, head_stride=30, keyframes=[0, 10, 38],
                              throttle=Throttle("fixed", fixed_ms=1.0), native_dir=run_dir / "native_frames", perception_every=every)
        port = free_port()
        stop, ready = threading.Event(), threading.Event()
        th = threading.Thread(target=lambda: asyncio.run(serve_async(policy, "127.0.0.1", port, stop, ready)), daemon=True)
        th.start()
        assert ready.wait(10) and requests.get(f"http://127.0.0.1:{port}/healthz", timeout=2).ok
        packer, got, sent = Packer(), [], {}
        with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as ws:
            meta = unpackb(ws.recv())
            assert meta["policy"] == "demo_action_replay_diagnostic" and meta["policy_result"] is False
            ws.send(packer.pack({"reset": True}))
            ws.send(packer.pack({"reset": True}))
            for t in range(n_serve + n_hold):
                head = rng.integers(0, 256, (720, 720, 4), dtype=np.uint8)
                head[..., 3] = 255 if t != 20 else head[..., 3]  # step 20: non-constant alpha must be kept (RGBA)
                depth = rng.uniform(0.2, 9.0, (720, 720)).astype(np.float32)
                depth[:4, :4] = np.inf
                wl, wr = (np.concatenate([rng.integers(0, 256, (480, 480, 3), dtype=np.uint8), np.full((480, 480, 1), 255, np.uint8)], 2)
                          for _ in range(2))
                crp = rng.normal(size=21).astype(np.float32)
                sent[t] = {"head": head, "head_depth": depth, "lwrist": wl, "rwrist": wr, "cam_rel": crp}
                obs = {"robot_r1::proprio": demo_state[t].astype(np.float32),
                       "robot_r1::robot_r1:zed_link:Camera:0::rgb": head, "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": depth,
                       "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": wl,
                       "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": wr,
                       "robot_r1::cam_rel_poses": crp, "task_id": np.array([TASK_ID])}
                ws.send(packer.pack(obs))
                resp = unpackb(ws.recv())
                assert resp["action"].shape == (23,) and resp["action"].dtype == np.float32
                got.append(np.array(resp["action"]))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            st = run_dir / "replay/replay_status.json"
            if st.exists() and json.loads(st.read_text()).get("final"):
                break
            time.sleep(0.1)
        stop.set()
        th.join(5)
        got = np.stack(got)
        assert np.array_equal(got[:n_serve], demo_act[:n_serve]), "served actions differ from the demo"
        hold = demo_act[n_serve - 1].copy()
        hold[:3] = 0
        assert np.array_equal(got[n_serve:], np.repeat(hold[None], n_hold, 0)), "tail hold wrong"
        nd = run_dir / "native_frames"
        T = n_serve + n_hold
        caps = list(range(0, T, every))
        for t in caps:
            for c in ("lwrist", "rwrist"):
                back = np.asarray(Image.open(nd / c / f"step_{t:06d}.png"))
                assert back.shape == (480, 480, 3) and np.array_equal(back, sent[t][c][..., :3]), f"{c} {t} not lossless"
            back = np.asarray(Image.open(nd / "head" / f"step_{t:06d}.png"))
            want = sent[t]["head"] if t == 20 else sent[t]["head"][..., :3]
            assert back.shape == want.shape and np.array_equal(back, want), f"head {t} not lossless"
            dep = np.load(nd / "head_depth" / f"step_{t:06d}.npy")
            assert dep.dtype == np.float16 and np.array_equal(dep, sent[t]["head_depth"].astype(np.float16)) and np.isinf(dep[0, 0])
        assert not (nd / "head" / "step_000005.png").exists()
        crp = np.load(nd / "cam_rel_poses.npy")
        assert crp.shape == (T, 3, 7) and np.array_equal(crp.reshape(T, 21), np.stack([sent[t]["cam_rel"] for t in range(T)]))
        prop = np.load(nd / "proprio.npy")
        assert prop.shape == (T, 61) and np.allclose(prop, demo_state[:T])
        assert np.array_equal(np.load(nd / "frame_index.npy"), np.r_[np.arange(n_serve), -np.ones(n_hold, int)])
        delay = np.load(run_dir / "replay/throttle_delay_s.npy")
        assert delay.shape == (T,) and np.allclose(delay, 0.001)
        cmeta = json.loads((nd / "capture_meta.json").read_text())
        assert cmeta["captures"] == len(caps) and cmeta["alpha_kept"] == [["head", 20]] and cmeta["capture_errors_total"] == 0
        assert cmeta["cam_rel_poses_order"] == ["left_wrist", "right_wrist", "head"]
        assert cmeta["received"]["rgb:head"]["shape"] == [720, 720, 4] and cmeta["received"]["depth:head"]["dtype"] == "float32"
        assert (run_dir / "replay/snapshots/sim_step00010_head.png").exists() and (run_dir / "replay/head_1hz/sim_step00030.jpg").exists()
        report["protocol_capture"] = {"actions_bitwise_equal_demo": True, "tail_hold": True, "capture_steps": caps,
                                      "rgb_png_lossless": True, "constant_alpha_dropped_nonconstant_kept": True,
                                      "head_depth_float16_exact": True, "cam_rel_poses_every_step": list(crp.shape),
                                      "proprio_every_step": list(prop.shape), "throttle_delay_logged": True}
        # 5) analysis on synthetic evaluator outputs (success, then a dropped can -> Q = 2/3)
        write_json(run_dir / "mapping_evidence.json", ev)
        write_json(run_dir / "manifest.json", {"task_id": TASK_ID})
        demo_ref = np.stack([sent[t]["cam_rel"].reshape(3, 7) for t in range(T)])
        np.savez(run_dir / "demo_reference.npz", robot2cam_pose=demo_ref[:, [2, 0, 1]].astype(np.float32),
                 cameras=np.array(["head", "lwrist", "rwrist"]))  # permuted on purpose: the check must recover the order
        instd = load_instance(inst)
        segs = demo_segments(ep)
        seg_names = [s["objects"][0][0] for s in segs if s["skill"] == "pick up from"]
        bin_name, can_names = seg_names[0], seg_names[1:4]
        bx, by, bz = instd["bin_pose"][:3]
        (run_dir / "eval/json").mkdir(parents=True)
        (run_dir / "eval/object_poses").mkdir(parents=True)
        lit_order = ["can__of__soda.n.01_1", "can__of__soda.n.01_3", "can__of__soda.n.01_2"]

        def schedule():
            """Per can: lift 0.6 m in 8 steps -> carry to the bin axis in <= 0.08 m steps -> descend 0.08 m/step into
            the bin. Expected geometric events are computed analytically here (independent of the analysis code)."""
            out, t = {}, 10
            for c in T1_CANS:
                x, y, z0 = instd["cans"][c]
                dist = math.hypot(bx - x, by - y)
                n = int(math.ceil(dist / 0.08)) + 1
                over = next(tt for tt in range(t + 8, t + 8 + n) if dist * (1 - (tt - t - 7) / n) < 0.15)
                k = int(math.ceil((z0 + 0.6 - 0.42) / 0.08 - 1e-9))
                out[c] = {"lift": t, "carry": n, "over": over, "in": t + 7 + n + k, "reach": t + 7 + n}
                t += 8 + n + 12
            return out, t + 5

        sched, T_pose = schedule()

        def synth(drop_third):
            rows_ = []
            for t in range(T_pose):
                objs_ = {T1_BIN: list(instd["bin_pose"])}
                lits_ = {}
                for i, c in enumerate(T1_CANS):
                    x, y, z0 = instd["cans"][c]
                    sc = sched[c]
                    pos = [x, y, z0]
                    if sc["lift"] <= t < sc["lift"] + 8:
                        pos = [x, y, z0 + 0.075 * (t - sc["lift"] + 1)]
                    elif sc["lift"] + 8 <= t < sc["lift"] + 8 + sc["carry"]:
                        f = (t - sc["lift"] - 7) / sc["carry"]
                        pos = [x + f * (bx - x), y + f * (by - y), z0 + 0.6]
                    elif t >= sc["lift"] + 8 + sc["carry"]:
                        pos = [bx, by, max(0.12, z0 + 0.6 - 0.08 * (t - sc["reach"]))]
                    if drop_third and i == 2 and t >= sc["lift"] + 11:  # slips out while carried, falls to the floor
                        f = 3 / sc["carry"]
                        pos = [x + f * (bx - x), y + f * (by - y), max(z0, z0 + 0.6 - 0.08 * (t - sc["lift"] - 10))]
                    objs_[c] = pos + [0.0, 0.0, 0.0, 1.0]
                    lits_[c] = t >= sc["in"] and not (drop_third and i == 2)
                rows_.append({"step": t, "robot_base": [instd["start"][0], instd["start"][1], 0.0, 0.0, 0.0,
                                                        math.sin(instd["start"][2] / 2), math.cos(instd["start"][2] / 2)],
                              "objects": objs_, "goal_literals": [[lits_[c] for c in lit_order]]})
            hdr = {"type": "header", "privileged_data_use": "evaluator-side evidence only; never policy input",
                   "entities": {T1_BIN: {"name": bin_name, "aabb_at_start_world": [[bx - 0.15, by - 0.15, 0.0], [bx + 0.15, by + 0.15, 0.40]],
                                         "native_bbox_object_frame": [0.3, 0.3, 0.4], "scale": [1.0, 1.0, 1.0]},
                                **{c: {"name": can_names[i]} for i, c in enumerate(T1_CANS)}},
                   "goal_literals": {"options": [[f"(inside {c} ashcan.n.01_1)" for c in lit_order]]},
                   "cameras": {"cam_rel_poses_order": ["left_wrist", "right_wrist", "head"],
                               "head": {"intrinsic_matrix": OFFICIAL_INTRINSICS["head"]}}}
            with open(run_dir / f"eval/object_poses/{TASK}_{inst}_0.jsonl", "w") as f:
                for rec in [hdr, *rows_, {"type": "end", "environment_steps": T_pose - 1}]:
                    f.write(json.dumps(rec) + "\n")
            write_json(run_dir / f"eval/json/{TASK}_{inst}_0.json",
                       {"task": TASK, "instance_id": inst, "rollout_id": 0, "steps": T_pose - 1, "success": not drop_third,
                        "terminated": not drop_third, "truncated": drop_third, "q_score": {"final": 2 / 3 if drop_third else 1.0}})
        synth(False)
        with patch("builtins.print"):
            analyze_task1(run_dir)
        s = json.loads((run_dir / "analysis/summary.json").read_text())
        v = s["verdict"]
        assert v["a_evaluator_success"] and v["a_q_score"] == 1.0 and v["a_evaluator_consistent_with_goal_literals"], v
        assert v["b_first_inside_step_bddl"] == {c: sched[c]["in"] for c in T1_CANS}, (v, sched)
        assert v["b_first_over_bin_footprint_step_geom"] == {c: sched[c]["over"] for c in T1_CANS}, (v, sched)
        assert s["cans"][T1_CANS[0]]["geom_inside_bin_first_step"] == sched[T1_CANS[0]]["in"]
        assert s["cans"][T1_CANS[0]]["lifted_first_step"] == sched[T1_CANS[0]]["lift"]  # z0 + 0.075 > z0 + 0.05
        assert v["b_dropped"] == {c: 0 for c in T1_CANS} and v["b_physics_anomalies"] == [] and v["b_instance_loaded_matches_file"]
        assert s["cans"][T1_CANS[0]]["demo_segments"] and s["bin"]["demo_segments"], "demo segments not paired by object name"
        cr = s["cam_rel_poses"]["vs_demo_robot2cam_pose"]
        assert cr["order_verified"] and cr["slots"]["head"]["best_matching_demo_camera"] == "head" and cr["slots"]["head"]["median_pos_err_mm"] == 0
        assert s["tracking"]["replayed_actions_equal_demo_actions"] and isinstance(s["plots"], (list, str))
        assert s["evaluator_consistency"]["rows_equal_steps_plus_1"] and s["evaluator_consistency"]["end_record_steps_equal_result_steps"]
        for f in (run_dir / "eval/json").glob("*.json"):
            f.unlink()
        (run_dir / f"eval/object_poses/{TASK}_{inst}_0.jsonl").unlink()
        synth(True)
        with patch("builtins.print"):
            analyze_task1(run_dir)
        s2 = json.loads((run_dir / "analysis/summary.json").read_text())
        v2 = s2["verdict"]
        assert not v2["a_evaluator_success"] and abs(v2["a_q_score"] - 2 / 3) < 1e-9 and v2["a_evaluator_consistent_with_goal_literals"], v2
        assert v2["b_dropped"][T1_CANS[2]] == 1 and s2["cans"][T1_CANS[2]]["missed"] and not s2["cans"][T1_CANS[0]]["missed"]
        assert v2["c_attribution"][T1_CANS[2]]["what"] == "dropped after lift" and set(v2["c_attribution"]) == {T1_CANS[2]}, v2
        report["analysis"] = {"success_case": {"first_inside_bddl": v["b_first_inside_step_bddl"], "q_consistent": True,
                                               "cam_rel_order_recovered_from_permuted_demo_reference": True},
                              "drop_case": {"q": round(v2["a_q_score"], 4), "dropped": v2["b_dropped"], "attribution": v2["c_attribution"]},
                              "plots": s["plots"]}
        if args.plots_dir:
            import shutil
            dst = Path(args.plots_dir)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(run_dir, dst)
            report["synthetic_run_copy"] = str(dst)
    report["handoff"] = _self_test_task1_handoff(ev, ep, inst, keep=args.plots_dir)
    report["result"] = "PASS"
    print(json.dumps(report, indent=1, default=str))
    if args.report:
        write_json(Path(args.report), report)
    return 0


def _self_test_task1_handoff(ev, ep, inst, keep=None):
    """CPU only, no simulator: Task-1 handoff spec (contract == demo segment, budgets), criteria on synthetic pose logs
    (lift, tipped-over bin, lift-then-drop, inside, inside-then-out), the early-stop trigger step, and analyze_task1's
    handoff section on synthetic run dirs (evaluator-side logs only)."""
    from unittest.mock import patch
    segs = demo_segments(ep)
    pick_bin = next(s for s in segs if s["skill"] == "pick up from" and s["objects"][0][0].startswith("trash_can"))
    pick_can = next(s for s in segs if s["skill"] == "pick up from" and s["objects"][0][0].startswith("can_of_soda"))
    place = next(s for s in segs if s["skill"] == "place in" and s["objects"][0][0] == pick_can["objects"][0][0])
    spec = {"a": t1_handoff_spec(ep, pick_bin["start"], "pick up from(trash_can,floors)"),
            "b": t1_handoff_spec(ep, pick_can["start"], "pick up from(can_of_soda,floors)"),
            "c": t1_handoff_spec(ep, place["start"], "place in(can_of_soda,trash_can)")}
    assert spec["b"] == t1_handoff_spec(ep, pick_can["start"])  # default contract = the segment's own key
    for k, s in (("a", pick_bin), ("b", pick_can), ("c", place)):
        assert spec[k]["budget_steps"] == 2 * (s["end"] - s["start"]) and spec[k]["handoff_at_segment_start"], spec[k]
    assert spec["a"]["kind"] == spec["b"]["kind"] == "lift" and spec["c"]["kind"] == "inside"
    assert spec["c"]["container"] == pick_bin["objects"][0][0] and spec["c"]["target_object"] == pick_can["objects"][0][0]
    if ep == 389:  # lead numbers 2026-09-26 (analysis/summary.json demo_picks / segments)
        assert [spec[k]["handoff_step"] for k in "abc"] == [920, 2341, 2658] and [spec[k]["budget_steps"] for k in "abc"] == [1156, 634, 342]
        assert spec["a"]["demo_gripper_commands"]["right"]["closes_in_segment"] and spec["b"]["demo_gripper_commands"]["left"]["closes_in_segment"]
        assert spec["c"]["demo_gripper_commands"]["left"]["opens_in_segment"]
    for bad in ((pick_can["start"], "pick up from(trash_can,floors)"), (pick_bin["start"] - 5, None), (place["start"], "pick up from(can_of_soda,floors)")):
        try:
            t1_handoff_spec(ep, *bad)
            raise AssertionError(f"mismatched handoff accepted: {bad}")
        except ValueError:
            pass
    assert t1_handoff_spec(ep, pick_can["start"], budget=77)["budget_rule"].startswith("CLI")
    # synthetic evaluator pose logs on the real instance poses (BDDL -> name as logged for instance 290)
    instd = load_instance(inst)
    others = [s["objects"][0][0] for s in segs if s["skill"] == "pick up from" and s["objects"][0][0].startswith("can_of_soda")
              and s["objects"][0][0] != pick_can["objects"][0][0]]
    names = {T1_BIN: pick_bin["objects"][0][0], T1_CANS[2]: pick_can["objects"][0][0], T1_CANS[0]: others[0], T1_CANS[1]: others[1]}
    lit_order = [T1_CANS[1], T1_CANS[2], T1_CANS[0]]
    bext, cext = np.array([0.2302, 0.2301, 0.2789]), np.array([0.0762, 0.0762, 0.1269])

    def info(pos, ext):
        c = np.array([pos[0], pos[1], ext[2] / 2])
        return {"name": None, "aabb_at_start_world": [(c - [0.2, 0.2, ext[2] / 2]).tolist(), (c + [0.2, 0.2, ext[2] / 2]).tolist()],
                "native_bbox_object_frame": ext.tolist(), "scale": [1.0, 1.0, 1.0]}
    header = {"type": "header", "privileged_data_use": "evaluator-side evidence only; never policy input",
              "entities": {T1_BIN: {**info(instd["bin_pose"], bext), "name": names[T1_BIN]},
                           **{c: {**info(instd["cans"][c], cext), "name": names[c]} for c in T1_CANS}},
              "goal_literals": {"options": [[f"(inside {c} {T1_BIN})" for c in lit_order]]}, "cameras": {}}
    half_x = np.array([math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)])  # 90 deg about x (bin on its side)

    def rows_for(last, target=None, dz=None, quat=None, inside=None):
        out = []
        for t in range(last + 1):
            objs_ = {T1_BIN: list(instd["bin_pose"])}
            objs_.update({c: list(instd["cans"][c]) + [0.0, 0.0, 0.0, 1.0] for c in T1_CANS})
            if target is not None:
                p = list(objs_[target])
                p[2] += dz(t)
                if quat is not None and quat(t) is not None:
                    p[3:7] = list(quat(t))
                objs_[target] = p
            lit = [bool(inside(t)) if (inside is not None and c == T1_CANS[2]) else False for c in lit_order]
            out.append({"step": t, "robot_base": [instd["start"][0], instd["start"][1], 0, 0, 0, 0, 1], "objects": objs_,
                        "goal_literals": [lit]})
        return out
    Na, Nb, Nc = (spec[k]["handoff_step"] for k in "abc")
    lift = lambda t0, cap: (lambda t: 0.03 * min(max(t - t0 + 1, 0), cap))  # noqa: E731  0.12 m at t0 + 3
    cases = {
        "a_lift_held": (spec["a"], rows_for(Na + 150, T1_BIN, lift(Na + 40, 10)), True, Na + 43, Na + 102),
        "a_bin_tipped_over": (spec["a"], rows_for(Na + 150, T1_BIN, lambda t: (0.115 - 0.1348) if t >= Na + 40 else 0.0,
                                                  lambda t: half_x if t >= Na + 40 else None), False, None, None),
        "b_lift_then_drop": (spec["b"], rows_for(Nb + 120, T1_CANS[2], lambda t: lift(Nb + 20, 5)(t) if t <= Nb + 80 else 0.0),
                             False, Nb + 23, None),
        "c_inside_held": (spec["c"], rows_for(Nc + 130, inside=lambda t: t >= Nc + 50), True, Nc + 50, Nc + 109),
        "c_inside_then_out": (spec["c"], rows_for(Nc + 130, inside=lambda t: Nc + 50 <= t <= Nc + 90), False, Nc + 50, None)}
    results = {}
    for name, (sp, rows_, met, first, fire) in cases.items():
        crit = T1HandoffCriterion(header, sp)
        val = crit.values(rows_)
        steps_ = np.array([r["step"] for r in rows_])
        post = steps_ >= sp["handoff_step"]
        hit = np.nonzero(np.array([crit.holds(v) for v in val]) & post)[0]
        got_first = int(steps_[hit[0]]) if len(hit) else None
        got_met = (bool(len(hit)) and val[-1] >= T1_HANDOFF_END_M) if sp["kind"] == "lift" else bool(val[-1] >= 0.5)
        es = EarlyStopTask1("/nonexistent", "none", sp["handoff_step"], sp, T1_EARLY_STOP_STEPS)
        fired = next((r["step"] for r in [header, *rows_] if es.feed(r)), None)
        assert (got_met, got_first, fired) == (met, first, fire), (name, got_met, got_first, fired, val.max())
        results[name] = {"met": got_met, "first_step": got_first, "early_stop_step": fired, "max_value": round(float(val.max()), 4)}
    # analyze_task1 handoff section on synthetic run dirs (b: budget end, evaluator JSON; c: early stop, no JSON)
    demo_state = np.load(LOWDIM / f"episode_{ep}_state.npy")
    demo_act = np.load(LOWDIM / f"episode_{ep}_action.npy").astype(np.float32)
    ana = {}
    for name, k, early in (("b_lift_then_drop", "b", None), ("c_inside_held", "c", {"step": Nc + 109, "rule": "synthetic"}),
                           ("a_lift_held", "a", {"step": Na + 102, "rule": "synthetic"})):
        sp, rows_ = cases[name][0], cases[name][1]
        N, last = sp["handoff_step"], rows_[-1]["step"]
        with tempfile.TemporaryDirectory(prefix="p2_handoff_t1ana_") as tmp:
            rd = Path(tmp) / "run"
            (rd / "replay").mkdir(parents=True)
            (rd / "eval/object_poses").mkdir(parents=True)
            write_json(rd / "mapping_evidence.json", ev)
            write_json(rd / "manifest.json", {"task_id": TASK_ID, "handoff_step": N, "budget_steps": last - N if early is None else sp["budget_steps"],
                                             "handoff_criterion": sp, "wam_config": {"contract": sp["contract"]},
                                             "early_stop_triggered": early, "status": "finished", "evaluator_returncode": 0})
            np.save(rd / "replay/proprio.npy", demo_state[:last].astype(np.float32))
            sent = demo_act[:last].copy()
            if k == "b":
                sent[N:, 14] = 1.0
                sent[N + 15:, 14] = -1.0  # left hand: open at the handoff, WAM closes it 15 steps later
            if k == "c":
                sent[N:, 14] = -1.0
                sent[N + 30:, 14] = 1.0   # left hand holds the can at the handoff, WAM opens it 30 steps later
            np.save(rd / "replay/actions_sent.npy", sent)
            write_json(rd / "replay/replay_status.json", {"demo_steps_served": N, "steps_served": last})
            with open(rd / f"eval/object_poses/{TASK}_{inst}_0.jsonl", "w") as f:
                for r in [header, *rows_] + ([{"type": "end", "environment_steps": last}] if early is None else []):
                    f.write(json.dumps(r) + "\n")
            if early is None:
                (rd / "eval/json").mkdir(parents=True)
                write_json(rd / f"eval/json/{TASK}_{inst}_0.json", {"task": TASK, "instance_id": inst, "rollout_id": 0, "steps": last,
                                                                    "success": False, "terminated": False, "truncated": True,
                                                                    "q_score": {"final": 0.0}})
            with patch("builtins.print"):
                analyze_task1(rd)
            s = json.loads((rd / "analysis/summary.json").read_text())
            hs, v = s["handoff"], s["verdict"]
            assert v["criterion_met"] == cases[name][2] and hs["criterion"]["met"] == cases[name][2], (name, hs["criterion"])
            assert s["diagnostic"] == "demo_prefix_handoff" and hs["run_ended_by"] == ("budget" if early is None else "early_stop")
            if k in "ab":
                assert hs["lift"][f"first_step_ge_{T1_HANDOFF_LIFT_M}m"] == cases[name][3], hs["lift"]
            if k == "b":
                assert hs["grasp_attempted"] == {"left": True, "right": False} and hs["gripper"]["left"]["first_close_cmd_step"] == N + 15
                assert hs["bin_side_effect"]["name"] == names[T1_BIN] and abs(hs["bin_side_effect"]["lowest_point_lift_at_end_m"]) < 1e-9
            if k == "c":
                assert hs["inside"]["first_step_after_handoff"] == Nc + 50 and hs["release_commanded"]["left"]
                assert hs["gripper"]["left"]["first_open_cmd_step"] == N + 30 and not hs["grasp_attempted"]["left"]
            ub = hs["upper_body_vs_demo_over_segment"]
            assert ub["window"] == [N, min(last, sp["demo_segment"][1])] and ub["left_arm"]["sim_vs_demo_mean"] == 0.0  # sim = demo here
            assert (rd / "analysis/handoff_trajectories.csv").exists()
            if keep:  # copy for a behavior6 (matplotlib) re-analysis / plot check
                import shutil
                dst = Path(f"{keep}_handoff_{name}")
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(rd, dst)
            ana[name] = {"criterion_met": v["criterion_met"], "run_ended_by": hs["run_ended_by"], "reason": hs["criterion"]["reason"],
                         "grasp_attempted": hs.get("grasp_attempted"), "release_commanded": hs.get("release_commanded"),
                         "plot": hs["plot"]}
    return {"specs": {k: {x: spec[k][x] for x in ("contract", "handoff_step", "budget_steps", "target_object", "kind")} for k in spec},
            "mismatch_refused": True, "criteria_and_early_stop": results, "analysis": ana}


def cmd_self_test_handoff(args):
    """CPU-only plumbing test of the handoff server with the REAL checkpoint (samples/num_steps reduced for speed)."""
    if TASK_ID != 13:
        return self_test_handoff_task1(args)
    import requests
    from websockets.sync.client import connect
    from PIL import Image
    import io
    from harness.frame_cache import FramePack
    Packer, unpackb = _msgpack()
    ep, N, post = args.episode, args.handoff_step, args.post_steps
    demo_state = np.load(LOWDIM / f"episode_{ep}_state.npy")
    demo_act = np.load(LOWDIM / f"episode_{ep}_action.npy").astype(np.float32)
    bounds = json.loads((ROOT / "configs/r1pro_action_bounds.json").read_text())
    lo, hi = np.asarray(bounds["action_low"], np.float32), np.asarray(bounds["action_high"], np.float32)
    head = FramePack(FRAMES / CAMS["head"], ep)
    t_start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        policy = HandoffPolicy(ep, tmp / "replay", N, args.wam_checkpoint, wam_device="cpu", wam_samples=args.wam_samples,
                               wam_num_steps=args.wam_num_steps, seed=0, budget=post, throttle=Throttle("off"), native_dir=tmp / "native_frames",
                               fixed_noise_seed=args.fixed_noise_seed)
        load_s = time.monotonic() - t_start
        port = free_port()
        stop, ready = threading.Event(), threading.Event()
        th = threading.Thread(target=lambda: asyncio.run(serve_async(policy, "127.0.0.1", port, stop, ready)), daemon=True)
        th.start()
        assert ready.wait(30) and requests.get(f"http://127.0.0.1:{port}/healthz", timeout=2).ok
        packer, got = Packer(), []
        with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as ws:
            meta = unpackb(ws.recv())
            assert meta["diagnostic"] == "demo_prefix_handoff" and meta["handoff_step"] == N and meta["seed"] == 0
            ws.send(packer.pack({"reset": True}))
            ws.send(packer.pack({"reset": True}))
            for t in range(N + post):
                rgb = np.asarray(Image.open(io.BytesIO(head.jpeg(t - t % 2))).convert("RGB"))
                rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], 2)
                obs = {"robot_r1::proprio": demo_state[t].astype(np.float32),
                       **{f"robot_r1::robot_r1{sfx[:-5]}::rgb": rgba for sfx in LIVE_SUFFIX.values()}}
                ws.send(packer.pack(obs))
                got.append(np.array(unpackb(ws.recv())["action"]))
        time.sleep(0.3)
        stop.set()
        th.join(5)
        got = np.stack(got)
        assert np.array_equal(got[:N], demo_act[:N]), "prefix actions differ from the demo"
        wam = got[N:]
        assert np.isfinite(wam).all() and (wam >= lo - 1e-6).all() and (wam <= hi + 1e-6).all()
        rec = json.loads((tmp / "replay/handoff_record.json").read_text())
        maxlen = policy.wam._proprio_buf.maxlen
        assert rec["step"] == N and rec["proprio_frames"] == N + 1 and rec["history_buffer_len"] == min(N + 1, maxlen)
        assert rec["buffer_equals_received_states"] and rec["proprio_input_shape"] == [len(policy.wam.model.proprio_lags), 61]
        chunks = [json.loads(line) for line in open(tmp / "replay/chunks.jsonl")]
        assert [c["step"] for c in chunks] == list(range(N, N + post, 32))
        for t in range(N, N + post, 15):
            assert (tmp / "native_frames/head" / f"step_{t:06d}.png").exists()
        fidx = np.load(tmp / "native_frames/frame_index.npy")
        assert (fidx[:N] == np.arange(N)).all() and (fidx[N:] == -1).all()
        if args.fixed_noise_seed is not None:
            assert policy.wam.gen.initial_seed() == args.fixed_noise_seed
        report = {"self_test_handoff": (f"PASS (CPU, real checkpoint, samples={args.wam_samples}, num_steps={args.wam_num_steps}; "
                                        "plumbing only, not a robot result)"),
                  "checkpoint": str(args.wam_checkpoint), "load_s": round(load_s, 1), "total_s": round(time.monotonic() - t_start, 1),
                  "handoff_record": rec, "chunks": len(chunks), "first_chunk_infer_ms": chunks[0]["infer_ms"],
                  "wam_action_within_bounds": True, "prefix_actions_bitwise_demo": True, "cond": policy.cond_info,
                  "proprio_lags": list(policy.wam.model.proprio_lags), "history_maxlen": maxlen,
                  "noise_rule": policy.noise_rule}
    print(json.dumps(report, indent=1))
    if args.report:
        write_json(Path(args.report), report)
    return 0


def self_test_handoff_task1(args):
    """Task 1: CPU plumbing check of the handoff server with the REAL multi-view checkpoint on recorded demo observations
    (frame cache: 224, stride 2, all three cameras). Checks: prefix = demo actions bitwise; the WAM gets exactly proprio +
    the checkpoint's RGB views (extra depth/cam_rel_poses/task_id keys in the fake observation must not reach it); DINO
    input = prep_rgb of each camera's frame in the checkpoint's view order; frozen contract vector = contract_targets.json;
    chunks every exec_horizon steps, finite and inside the action bounds; the executed actions equal the deployment formula
    recomputed from the recorded inputs; lags [0] handled. Plumbing only, not a robot result."""
    import io
    import requests
    import torch
    from websockets.sync.client import connect
    from PIL import Image
    from harness.frame_cache import FramePack
    from harness.policy_server import prep_rgb
    Packer, unpackb = _msgpack()
    ep, N, post = args.episode, args.handoff_step, args.post_steps
    spec = handoff_spec(ep, N, args.wam_contract)
    h = T1_HANDOFF_EXEC_HORIZON if args.wam_exec_horizon is None else int(args.wam_exec_horizon)
    demo_state = np.load(LOWDIM / f"episode_{ep}_state.npy")
    demo_act = np.load(LOWDIM / f"episode_{ep}_action.npy").astype(np.float32)
    bounds = json.loads((ROOT / "configs/r1pro_action_bounds.json").read_text())
    lo, hi = np.asarray(bounds["action_low"], np.float32), np.asarray(bounds["action_high"], np.float32)
    packs = {cam: FramePack(FRAMES / CAMS[cam], ep) for cam in CAMS}

    def rgba(cam, t):
        rgb = np.asarray(Image.open(io.BytesIO(packs[cam].jpeg(t - t % 2))).convert("RGB"))
        return np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], 2)

    def observation(t, imgs):  # evaluator-style keys + privileged extras that must NOT reach the WAM
        return {"robot_r1::proprio": demo_state[t].astype(np.float32),
                **{f"robot_r1::robot_r1{LIVE_SUFFIX[c]}": imgs[c] for c in CAMS},
                "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": np.ones((224, 224), np.float32),
                "robot_r1::cam_rel_poses": np.zeros(21, np.float32), "task_id": np.array([TASK_ID])}
    t_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="p2_handoff_selftest_") as tmp:
        tmp = Path(tmp)
        policy = HandoffPolicy(ep, tmp / "replay", N, args.wam_checkpoint, wam_device="cpu", wam_samples=args.wam_samples,
                               wam_num_steps=args.wam_num_steps, seed=0, budget=post, throttle=Throttle("off"),
                               native_dir=tmp / "native_frames", fixed_noise_seed=args.fixed_noise_seed,
                               contract=spec["contract"], skill=spec["skill"], exec_horizon=h)
        if args.threads:
            torch.set_num_threads(int(args.threads))  # offline speed only (WamPolicy sets 2); recompute below uses the same
        load_s = time.monotonic() - t_start
        wam, model = policy.wam, policy.wam.model
        views = [CAMS["head"], CAMS["lwrist"], CAMS["rwrist"]]
        assert policy.multiview and list(model.views) == views, model.views
        lags = list(model.proprio_lags)
        want_keys = sorted(["robot_r1::proprio"] + [f"robot_r1::robot_r1{s}" for s in model.view_suffixes])
        seen = {"px": [], "obs_keys": []}
        gvf, vpx = model.wrapper.get_vision_features, wam._view_pixels

        def spy_gvf(px, *a, **k):
            seen["px"].append(px.detach().clone())
            return gvf(px, *a, **k)

        def spy_view_pixels(obs, m, head_px):
            seen["obs_keys"].append(sorted(obs))
            return vpx(obs, m, head_px)
        model.wrapper.get_vision_features, wam._view_pixels = spy_gvf, spy_view_pixels
        port = free_port()
        stop, ready = threading.Event(), threading.Event()
        th = threading.Thread(target=lambda: asyncio.run(serve_async(policy, "127.0.0.1", port, stop, ready)), daemon=True)
        th.start()
        assert ready.wait(30) and requests.get(f"http://127.0.0.1:{port}/healthz", timeout=2).ok
        packer, got, kept = Packer(), [], {}
        with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as ws:
            meta = unpackb(ws.recv())
            assert meta["diagnostic"] == "demo_prefix_handoff" and meta["handoff_step"] == N and meta["seed"] == 0
            assert meta["contract"] == spec["contract"] and meta["exec_horizon"] == h and meta["views"] == views
            ws.send(packer.pack({"reset": True}))
            ws.send(packer.pack({"reset": True}))
            for t in range(N + post):
                imgs = {c: rgba(c, t) for c in CAMS}
                if t >= N and (t - N) % h == 0:
                    kept[t] = imgs
                ws.send(packer.pack(observation(t, imgs)))
                got.append(np.array(unpackb(ws.recv())["action"]))
        time.sleep(0.3)
        stop.set()
        th.join(5)
        run_s = time.monotonic() - t_start - load_s
        got = np.stack(got)
        assert np.array_equal(got[:N], demo_act[:N]), "prefix actions differ from the demo"
        wam_a = got[N:]
        assert np.isfinite(wam_a).all() and (wam_a >= lo - 1e-6).all() and (wam_a <= hi + 1e-6).all()
        rec = json.loads((tmp / "replay/handoff_record.json").read_text())
        maxlen = wam._proprio_buf.maxlen
        assert rec["step"] == N and rec["proprio_frames"] == N + 1 and rec["history_buffer_len"] == min(N + 1, maxlen)
        assert rec["buffer_equals_received_states"]
        assert rec["proprio_input_shape"] == ([61] if lags == [0] else [len(lags), 61]), rec["proprio_input_shape"]
        assert rec["policy_obs_keys"] == want_keys and rec["views"] == views, rec
        chunks = [json.loads(line) for line in open(tmp / "replay/chunks.jsonl")]
        assert [c["step"] for c in chunks] == list(range(N, N + post, h)), [c["step"] for c in chunks]
        assert all(c["exec_horizon"] == h and len(c["grip_exec_left"]) == h for c in chunks)
        # view order + preprocessing: each DINO input equals prep_rgb of the frame sent for that camera, head first
        assert len(seen["px"]) == len(chunks) and all(k == want_keys for k in seen["obs_keys"]), (len(seen["px"]), seen["obs_keys"][:1])
        for c, px in zip(chunks, seen["px"]):
            assert tuple(px.shape) == (1, 3, 3, model.image_size, model.image_size), tuple(px.shape)
            for v, cam in enumerate(("head", "lwrist", "rwrist")):
                assert np.array_equal(px[0, v].numpy(), prep_rgb(kept[c["step"]][cam], model.image_size)), (c["step"], cam)
        distinct_views = bool(not torch.equal(seen["px"][0][0, 1], seen["px"][0][0, 2]) and not torch.equal(seen["px"][0][0, 0], seen["px"][0][0, 1]))
        # frozen contract vector
        ref = np.asarray(json.loads(CONTRACT_TARGETS.read_text())["contracts"][spec["contract"]]["vector"], np.float32)
        assert np.array_equal(policy.cond.numpy()[0], ref)
        assert wam.train_targets_sha256 == sha256_file(CONTRACT_TARGETS) == model.cfg.dataset.get("contract_targets_sha256")
        # the executed first chunk == deployment formula on the recorded inputs at N with the same generator state
        recompute = None
        if lags == [0]:
            wam.gen.manual_seed(0 if args.fixed_noise_seed is None else args.fixed_noise_seed)
            obs_pol = policy.policy_obs(observation(N, kept[N]))
            state, head_px = wam._preprocess(obs_pol)
            chunk = wam._bounded_chunk(wam._sample(wam._model_feats(model, obs_pol, head_px, None), state, policy.cond, model=model))
            post_proc = wam._apply_arm_gain(wam._apply_base_control(chunk, spec["skill"]), state, spec["skill"])
            recompute = {"bitwise_equal_first_exec": bool(np.array_equal(post_proc[:h], wam_a[:h])),
                         "max_abs_diff": float(np.abs(post_proc[:h] - wam_a[:h]).max()),
                         "base_control_and_arm_gain_identity": bool(np.array_equal(post_proc, chunk))}
            if policy.base_control == "hold_during_manipulation":   # Task 0 default: base zeroed, upper body untouched
                expect = np.array(chunk, copy=True)
                expect[:, :3] = 0.0
                recompute.update(base_control=policy.base_control, base_zeroed_upper_body_identity=bool(np.array_equal(post_proc, expect)),
                                 base_control_and_arm_gain_identity=None)
                assert recompute["bitwise_equal_first_exec"] and recompute["base_zeroed_upper_body_identity"], recompute
            else:
                assert recompute["bitwise_equal_first_exec"] and recompute["base_control_and_arm_gain_identity"], recompute
        for t in range(N, N + post, 15):
            for cam in CAMS:
                assert (tmp / "native_frames" / cam / f"step_{t:06d}.png").exists(), (cam, t)
        fidx = np.load(tmp / "native_frames/frame_index.npy")
        assert (fidx[:N] == np.arange(N)).all() and (fidx[N:] == -1).all()
        # deployment resolution (RGBDFullResWrapper head 720 / wrists 480): prep_rgb path on sim renders of the replay run
        native = {}
        nd = OUT_ROOT / "run_ep389_inst290_rgbd/native_frames"
        tn = N - N % 10
        if ep == 389 and all((nd / c / f"step_{tn:06d}.png").exists() for c in CAMS):
            imgs = {c: np.asarray(Image.open(nd / c / f"step_{tn:06d}.png")) for c in CAMS}
            obs_pol = policy.policy_obs(observation(tn, imgs))
            _, head_px = wam._preprocess(obs_pol)
            V = vpx(obs_pol, model, head_px)
            assert V.shape == (3, 3, model.image_size, model.image_size)
            assert all(np.array_equal(V[v], prep_rgb(imgs[c], model.image_size)) for v, c in enumerate(("head", "lwrist", "rwrist")))
            native = {"frames": f"{nd.relative_to(ROOT)}/<cam>/step_{tn:06d}.png",
                      "received_shapes": {c: list(imgs[c].shape) for c in CAMS}, "model_input_shape": list(V.shape),
                      "per_view_equals_prep_rgb": True}
        w = model.wrapper
        report = {"self_test_handoff_task1": (f"PASS (CPU, real checkpoint, K={args.wam_samples}, num_steps={args.wam_num_steps}, "
                                              f"exec_horizon={h}; recorded demo frames; plumbing only, not a robot result)"),
                  "checkpoint": str(args.wam_checkpoint), "checkpoint_step": int(model.step), "views": views,
                  "proprio_mask": w.proprio_mask_name, "proprio_lags": lags, "action_repr": w.action_repr_name,
                  "contract": spec["contract"], "criterion_that_a_sim_run_would_use": spec["criterion"],
                  "budget_default": spec["budget_steps"], "cond": policy.cond_info,
                  "cond_equals_contract_targets": True, "train_targets_sha256": wam.train_targets_sha256,
                  "load_s": round(load_s, 1), "run_s": round(run_s, 1), "torch_threads": torch.get_num_threads(),
                  "chunks": len(chunks), "chunk_steps": [c["step"] for c in chunks],
                  "infer_ms": [c["infer_ms"] for c in chunks],
                  "grip_exec_by_chunk": [[c["step"], c["grip_exec_left"], c["grip_exec_right"], c["demo_grip_same_step"]] for c in chunks],
                  "prefix_actions_bitwise_demo": True, "wam_action_within_bounds": True,
                  "policy_obs_keys": want_keys, "privileged_keys_filtered": True,
                  "dino_input_equals_prep_rgb_per_view_in_order": True, "views_distinct": distinct_views,
                  "deployment_formula_recompute": recompute, "handoff_record": rec, "history_maxlen": maxlen,
                  "deployment_resolution_path": native or "skipped (no replay native frames for this episode/step)",
                  "noise_rule": policy.noise_rule}
    print(json.dumps(report, indent=1, default=str))
    if args.report:
        write_json(Path(args.report), report)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mapping")
    m.add_argument("--episode", type=int, required=True)
    m.add_argument("--instance", type=int, required=True)
    m.add_argument("--out", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--episode", type=int, required=True)
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8010)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--throttle", choices=("adaptive", "fixed", "off"), default="adaptive")
    s.add_argument("--fixed-ms", type=float, default=0.0)
    s.add_argument("--head-stride", type=int, default=30)
    s.add_argument("--native-dir", default=None, help="lossless raw camera captures + proprio.npy + frame_index.npy")
    s.add_argument("--native-mode", choices=("default", "segment"), default="default")
    s.add_argument("--perception-every", type=int, default=0,
                   help="Task 1 perception capture: RGB + head depth every N steps, cam_rel_poses every step (0 = off)")
    s.add_argument("--handoff-step", type=int, default=None,
                   help="demo actions for t < N, then the WAM (Task 13: open lid(car); Task 1: --wam-contract)")
    s.add_argument("--wam-checkpoint", default=None)
    s.add_argument("--wam-device", default="cuda")
    s.add_argument("--wam-samples", type=int, default=8)
    s.add_argument("--wam-num-steps", type=int, default=10)
    s.add_argument("--wam-cuda-fraction", type=float, default=0.4)
    s.add_argument("--budget", type=int, default=3084)
    s.add_argument("--wam-fixed-noise-seed", type=int, default=None)
    s.add_argument("--wam-base-from-demo", action="store_true")
    s.add_argument("--primitive", default=None, help="pick_can | pick_bin | place_in, or a comma chain")
    s.add_argument("--primitive-target", default="")
    s.add_argument("--primitive-align", action="store_true")
    s.add_argument("--wam-contract", default=None, help="Task 1 handoff contract key, e.g. 'pick up from(can_of_soda,floors)'")
    s.add_argument("--wam-skill", default=None, help="VTT skill of the contract (default: the handoff default / contract prefix)")
    s.add_argument("--wam-exec-horizon", type=int, default=None, help="actions executed per chunk (default 32 = Task 13)")
    s.add_argument("--wam-base-control", choices=("wam_whole_body", "hold_during_manipulation"), default=None,
                   help="WAM orchestrate.base_control (default: WamPolicy default; Task 0: hold_during_manipulation)")
    r = sub.add_parser("run")
    r.add_argument("--episode", type=int, required=True)
    r.add_argument("--instance", type=int, required=True)
    r.add_argument("--run-name", required=True)
    r.add_argument("--steps", type=int, default=None, help="demo actions to replay (default: whole episode)")
    r.add_argument("--tail-hold-steps", type=int, default=None, help="default 90 (Task 13) / 300 (Task 1)")
    r.add_argument("--port", type=int, default=8010)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--timeout", type=int, default=5400)
    r.add_argument("--max-lock-wait", type=float, default=None)
    r.add_argument("--throttle", choices=("adaptive", "off"), default="adaptive")
    r.add_argument("--no-video", action="store_true", help="no viewer video (only if VRAM headroom requires it)")
    r.add_argument("--stop-after-openlid", type=int, default=None, help="replay only until open-lid end + K steps")
    r.add_argument("--native-mode", choices=("default", "segment"), default="default")
    r.add_argument("--handoff-step", type=int, default=None, help="demo-prefix -> WAM handoff diagnostic at step N")
    r.add_argument("--wam-checkpoint", default="outputs/training/stage7b_prophistdiff_manip_20260924/checkpoint_best.pt")
    r.add_argument("--wam-samples", type=int, default=8)
    r.add_argument("--wam-num-steps", type=int, default=10)
    r.add_argument("--wam-cuda-fraction", type=float, default=0.4)
    r.add_argument("--wam-reserve-mib", type=int, default=2300)
    r.add_argument("--wam-fixed-noise-seed", type=int, default=None, help="manual_seed before every _sample (mode commitment)")
    r.add_argument("--wam-base-from-demo", action="store_true", help="Task 1 diagnostic: base dims replay the demo after N")
    r.add_argument("--primitive", default=None,
                   help="Task 1 diagnostic: perception+IK primitive instead of the WAM after N (no WAM loaded)")
    r.add_argument("--primitive-target", default="", help="x,y,z of the target object in the base frame at N (oracle)")
    r.add_argument("--primitive-align", action="store_true", help="odometry fine alignment to the grasp-time dock pose")
    r.add_argument("--budget", type=int, default=None,
                   help="post-handoff steps (default: Task 13 3084 = open-lid budget; Task 1 2 x (demo segment end - N))")
    r.add_argument("--early-stop-jtrunk", type=float, default=1.0)
    r.add_argument("--early-stop-steps", type=int, default=None,
                   help="consecutive steps the criterion must hold (default: Task 13 300 (j_trunk); Task 1 60 (pose log))")
    r.add_argument("--wam-contract", default=None,
                   help="Task 1: contract key; must equal the demo segment at N (default: derived from that segment)")
    r.add_argument("--wam-skill", default=None, help="Task 1: VTT skill (default: the contract prefix)")
    r.add_argument("--wam-exec-horizon", type=int, default=None, help="Task 1: actions per chunk before re-planning (default 8)")
    r.add_argument("--wam-base-control", choices=("wam_whole_body", "hold_during_manipulation"), default=None,
                   help="WAM base ownership during the contract (default: WamPolicy default; Task 0: hold_during_manipulation)")
    r.add_argument("--video-width", type=int, default=1280)
    r.add_argument("--video-height", type=int, default=720)
    r.add_argument("--env-wrapper", choices=tuple(T1_WRAPPERS), default="rgbd",
                   help="Task 1: official RGBDFullResWrapper (head 720x720, wrists 480x480, RGB+depth) or the default 224 RGB")
    r.add_argument("--capture-every", type=int, default=10, help="Task 1 perception capture period (steps)")
    r.add_argument("--dry-run", action="store_true",
                   help="CPU-only launch check: mapping, preflight, spec, checkpoint, manifest in /tmp/p2_handoff_dryrun; no GPU lock/units")
    a = sub.add_parser("analyze")
    a.add_argument("--run-dir", required=True)
    t = sub.add_parser("self-test")
    t.add_argument("--episode", type=int, default=None, help="default: 2603 (Task 13) / 389 (Task 1)")
    t.add_argument("--instance", type=int, default=None, help="default: 4 (Task 13) / 290 (Task 1)")
    t.add_argument("--report", default=None)
    t.add_argument("--plots-dir", default=None, help="copy the synthetic run dir here (for a behavior6 plot check)")
    h = sub.add_parser("self-test-handoff")
    h.add_argument("--episode", type=int, default=2603)
    h.add_argument("--handoff-step", type=int, default=70)
    h.add_argument("--post-steps", type=int, default=40)
    h.add_argument("--wam-checkpoint", default="outputs/training/stage7b_prophistdiff_manip_20260924/checkpoint_best.pt")
    h.add_argument("--report", default=None)
    h.add_argument("--fixed-noise-seed", type=int, default=None)
    h.add_argument("--wam-samples", type=int, default=1, help="K flow samples averaged per chunk")
    h.add_argument("--wam-num-steps", type=int, default=2, help="Euler steps (deployment: 10)")
    h.add_argument("--wam-contract", default=None, help="Task 1 (default: the demo segment at N)")
    h.add_argument("--wam-exec-horizon", type=int, default=None, help="default 32 (Task 13) / 8 (Task 1)")
    h.add_argument("--threads", type=int, default=None, help="Task 1 check: torch CPU threads after load (WamPolicy sets 2)")
    args = ap.parse_args()
    return {"mapping": cmd_mapping, "serve": cmd_serve, "run": cmd_run, "analyze": cmd_analyze,
            "self-test": cmd_self_test, "self-test-handoff": cmd_self_test_handoff}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
