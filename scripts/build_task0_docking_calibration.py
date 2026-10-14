#!/usr/bin/env python3
"""Task 0 turning_on_radio docking calibration from TRAIN demonstrations (CPU; same schema as Task 1's
outputs/validation/task1_docking_calibration_20260926/, read by harness/docking_controller.load_calibration).

grasp_relative_poses_train.json: one row per train demo with an unambiguous pick hand
  (last open->close transition that stays closed at the pick-segment end):
  rel_x/rel_y = grasp point (eef of the holding hand at closure) expressed in the base frame at pick START,
  base_shift  = base odometry [x, y, yaw] from pick start to closure (so grasp_time_rel() gives the pose at closure),
  eef_z, trunk_start/trunk_grasp.
posture_and_gripper_stats_train.json: median/p10/p90 of trunk / left arm / right arm joint positions at
  'nav_mid|before_bin' (middle of move to; hands free - the Task 1 key name is kept so the controller needs no
  task branch), 'pick up from_start|radio', 'pick up from_end|radio', 'press_start|radio', 'place on_start|radio';
  finger sums (m) while navigating and while holding.
The point convention is the RIGHT-HAND GRASP POINT (eef at closure); a detector returning the radio centre must add
its measured centre->grasp offset (harness/radio_perception.py).
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/behavior-1k/2026-challenge-demos"
LOW = ROOT / "data/lowdim_cache_task000"
OUT = ROOT / "outputs/validation/task0_docking_calibration_20260927"
GRIP_ACTION = {"left": 14, "right": 22}
EEF = {"left": slice(17, 20), "right": slice(42, 45)}
FING = {"left": slice(24, 26), "right": slice(49, 51)}
TRUNK, LARM, RARM = slice(53, 57), slice(3, 10), slice(28, 35)
DT = 1.0 / 30.0
LABEL = "radio"


def integrate(v):
    th = np.concatenate([[0.0], np.cumsum(v[:, 2] * DT)])
    thm = th[:-1] + 0.5 * v[:, 2] * DT
    dx = (v[:, 0] * np.cos(thm) - v[:, 1] * np.sin(thm)) * DT
    dy = (v[:, 0] * np.sin(thm) + v[:, 1] * np.cos(thm)) * DT
    return np.concatenate([[0.0], np.cumsum(dx)]), np.concatenate([[0.0], np.cumsum(dy)]), th


def q(rows):
    a = np.asarray(rows, float)
    return {"median": np.median(a, 0).round(4).tolist(), "p10": np.percentile(a, 10, 0).round(4).tolist(),
            "p90": np.percentile(a, 90, 0).round(4).tolist(), "n": int(len(a))}


def main():
    split = json.loads((DEMO / "split_task_000.json").read_text())
    segs_all = json.loads((DEMO / "skill_segments_task000.json").read_text())["episodes"]
    rows, skipped = [], []
    post = {k: {"trunk": [], "larm": [], "rarm": []} for k in
            ("nav_mid|before_bin", f"pick up from_start|{LABEL}", f"pick up from_end|{LABEL}", f"press_start|{LABEL}",
             f"place on_start|{LABEL}")}
    fing = {"left_open_nav": [], "right_open_nav": [], "right_holding": [], "left_holding": []}
    for ep in split["train_episode_indices"]:
        st = np.load(LOW / f"episode_{ep}_state.npy"); act = np.load(LOW / f"episode_{ep}_action.npy")
        segs = segs_all[str(ep)]["segments"]
        if [s["skill"] for s in segs][:3] != ["move to", "pick up from", "press"]:
            skipped.append({"ep": ep, "why": "program"}); continue
        nav, pick, press = segs[0], segs[1], segs[2]
        en = min(pick["end"], len(act) - 1)
        cands = []
        for hand, gi in GRIP_ACTION.items():
            ga = act[:, gi]
            tr = [t for t in range(max(pick["start"], 1), en + 1) if ga[t - 1] > 0 and ga[t] <= 0]
            if tr and ga[en] <= 0:
                cands.append((hand, tr[-1]))
        if len(cands) != 1:
            skipped.append({"ep": ep, "why": "ambiguous_grasp_hand", "cands": cands}); continue
        hand, gf = cands[0]
        s0 = pick["start"]
        x, y, th = integrate(st[s0 + 1:gf + 1, 0:3].astype(np.float64))
        e = st[gf, EEF[hand]].astype(np.float64)
        c, s = math.cos(th[-1]), math.sin(th[-1])
        rel = [x[-1] + c * e[0] - s * e[1], y[-1] + s * e[0] + c * e[1]]
        rows.append({"ep": ep, "seg": 1, "cat": LABEL, "hand": hand, "grasp_frame": int(gf), "t_from_start": int(gf - s0),
                     "rel_x": float(rel[0]), "rel_y": float(rel[1]), "eef_z": float(e[2]),
                     "trunk_start": st[s0, TRUNK].tolist(), "trunk_grasp": st[gf, TRUNK].tolist(),
                     "base_shift": [float(x[-1]), float(y[-1]), float(th[-1])]})
        if hand != "right":
            continue          # postures only from the standard-hand demos (right holds, left presses)
        mid = (nav["start"] + nav["end"]) // 2
        for key, f in (("nav_mid|before_bin", mid), (f"pick up from_start|{LABEL}", s0),
                       (f"pick up from_end|{LABEL}", pick["end"] - 1), (f"press_start|{LABEL}", press["start"]),
                       *(((f"place on_start|{LABEL}", segs[3]["start"]),) if len(segs) > 3 and segs[3]["skill"] == "place on" else ())):
            post[key]["trunk"].append(st[f, TRUNK]); post[key]["larm"].append(st[f, LARM]); post[key]["rarm"].append(st[f, RARM])
        fing["left_open_nav"].append(st[mid, FING["left"]].sum()); fing["right_open_nav"].append(st[mid, FING["right"]].sum())
        fing["right_holding"].append(st[press["start"], FING["right"]].sum())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grasp_relative_poses_train.json").write_text(json.dumps(rows, indent=1))

    def fq(v):
        a = np.asarray(v, float)
        return {"median": float(np.median(a)), "p1": float(np.percentile(a, 1)), "p5": float(np.percentile(a, 5)),
                "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}
    stats = {"postures": {k: {g: q(v) for g, v in d.items()} for k, d in post.items() if d["trunk"]},
             "gripper_finger_sum_m": {k: fq(v) for k, v in fing.items() if v},
             "source": ("TRAIN split_task_000.json, standard-hand demos (right hand holds the radio, left presses); trunk 53:57, "
                        "left arm 3:10, right arm 28:35, finger qpos sum (m); scripts/build_task0_docking_calibration.py"),
             "hands": dict(Counter(r["hand"] for r in rows)), "skipped": skipped}
    (OUT / "posture_and_gripper_stats_train.json").write_text(json.dumps(stats, indent=1))
    right = [r for r in rows if r["hand"] == "right"]
    start = np.array([[r["rel_x"], r["rel_y"]] for r in right])
    shift = np.array([r["base_shift"] for r in right])
    grasp = np.array([[math.cos(-b[2]) * (r["rel_x"] - b[0]) - math.sin(-b[2]) * (r["rel_y"] - b[1]),
                       math.sin(-b[2]) * (r["rel_x"] - b[0]) + math.cos(-b[2]) * (r["rel_y"] - b[1])] for r, b in zip(right, shift)])
    print(json.dumps({"n_rows": len(rows), "hands": stats["hands"], "n_skipped": len(skipped),
                      "start_rel": q(start), "grasp_rel": q(grasp), "base_shift_norm": q(np.hypot(shift[:, 0], shift[:, 1])),
                      "eef_z": q([r["eef_z"] for r in right]),
                      "pick_start_trunk": stats["postures"][f"pick up from_start|{LABEL}"]["trunk"]["median"],
                      "nav_trunk": stats["postures"]["nav_mid|before_bin"]["trunk"]["median"],
                      "fingers": {k: round(v["median"], 4) for k, v in stats["gripper_finger_sum_m"].items()}}, indent=1))


if __name__ == "__main__":
    main()
