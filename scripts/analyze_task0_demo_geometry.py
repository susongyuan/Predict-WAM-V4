#!/usr/bin/env python3
"""Task 0 turning_on_radio: TRAIN-only demonstration geometry (CPU, offline; nothing here is fed to a policy).

Per train episode:
  * toggle frame = first frame with next.reward > 0 (the evaluator's toggled_on(radio) flip);
    which skill segment contains it, offset from segment start;
  * pick: which hand closes (last open->close gripper transition that stays closed at pick end), grasp frame,
    radio grasp point in the base frame at pick START and at grasp (eef at closure moved by odometry);
  * press: which hand moves, its gripper command at the toggle frame, its eef relative to the holding eef
    (in the base frame) at the toggle frame -> button location relative to the grasp;
  * base displacement during pick / press / place; trunk + arm postures at pick start and at toggle.

Output: outputs/validation/task0_demo_geometry_20260927/{rows_train.json,summary.json}
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/behavior-1k/2026-challenge-demos"
LOW = ROOT / "data/lowdim_cache_task000"
OUT = ROOT / "outputs/validation/task0_demo_geometry_20260927"
GRIP_ACTION = {"left": 14, "right": 22}
EEF = {"left": slice(17, 20), "right": slice(42, 45)}
FING = {"left": slice(24, 26), "right": slice(49, 51)}
ARM_Q = {"left": slice(7, 14), "right": slice(15, 22)}   # action slices (absolute joint targets)
TRUNK_A = slice(3, 7)
DT = 1.0 / 30.0


def integrate(v):
    th = np.concatenate([[0.0], np.cumsum(v[:, 2] * DT)])
    thm = th[:-1] + 0.5 * v[:, 2] * DT
    dx = (v[:, 0] * np.cos(thm) - v[:, 1] * np.sin(thm)) * DT
    dy = (v[:, 0] * np.sin(thm) + v[:, 1] * np.cos(thm)) * DT
    return np.concatenate([[0.0], np.cumsum(dx)]), np.concatenate([[0.0], np.cumsum(dy)]), th


def rel_at(state, a, b, p_b):
    """Point p_b (base frame at frame b) expressed in the base frame at frame a (a <= b)."""
    if b == a:
        return np.asarray(p_b[:2], float)
    x, y, th = integrate(state[a + 1:b + 1, 0:3].astype(np.float64))
    c, s = math.cos(th[-1]), math.sin(th[-1])
    return np.array([x[-1] + c * p_b[0] - s * p_b[1], y[-1] + s * p_b[0] + c * p_b[1]])


def disp(state, a, b):
    x, y, th = integrate(state[a + 1:b + 1, 0:3].astype(np.float64))
    return float(math.hypot(x[-1], y[-1])), float(th[-1])


def rewards():
    out = {}
    for f in sorted((DEMO / "data/chunk-000").glob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "next.reward"])
        ep = np.array(t.column("episode_index")); fr = np.array(t.column("frame_index")); r = np.array(t.column("next.reward"))
        for e in np.unique(ep):
            m = (ep == e) & (r > 0)
            out[int(e)] = int(fr[m].min()) if m.any() else None
    return out


def main():
    split = json.loads((DEMO / "split_task_000.json").read_text())
    segs_all = json.loads((DEMO / "skill_segments_task000.json").read_text())["episodes"]
    train = split["train_episode_indices"]
    rew = rewards()
    rows, problems = [], []
    for ep in train:
        state = np.load(LOW / f"episode_{ep}_state.npy"); act = np.load(LOW / f"episode_{ep}_action.npy")
        segs = segs_all[str(ep)]["segments"]
        names = [s["skill"] for s in segs]
        if names[:3] != ["move to", "pick up from", "press"]:
            problems.append({"ep": ep, "why": "program", "skills": names}); continue
        pick, press = segs[1], segs[2]
        tf = rew.get(ep)
        cands = []
        for hand, gi in GRIP_ACTION.items():
            ga = act[:, gi]; st, en = pick["start"], min(pick["end"], len(ga) - 1)
            tr = [t for t in range(max(st, 1), en + 1) if ga[t - 1] > 0 and ga[t] <= 0]
            if tr and ga[en] <= 0:
                cands.append((hand, tr[-1]))
        if len(cands) != 1:
            problems.append({"ep": ep, "why": "grasp_hand", "cands": cands}); continue
        hold, gf = cands[0]
        other = "left" if hold == "right" else "right"
        g_eef = state[gf, EEF[hold]].astype(np.float64)
        row = {"ep": ep, "hold_hand": hold, "grasp_frame": gf, "pick_start": pick["start"], "pick_end": pick["end"],
               "press_start": press["start"], "press_end": press["end"], "toggle_frame": tf,
               "radio_rel_pick_start": rel_at(state, pick["start"], gf, g_eef).tolist(),
               "radio_rel_grasp": g_eef[:2].tolist(), "radio_z_grasp": float(g_eef[2]),
               "base_disp_pick": disp(state, pick["start"], gf), "base_disp_press": disp(state, press["start"], press["end"]),
               "trunk_pick_start": act[pick["start"], TRUNK_A].tolist()}
        if tf is not None:
            row["toggle_in"] = next((s["skill"] for s in segs if s["start"] <= tf < s["end"]), "after_last")
            row["toggle_minus_press_start"] = tf - press["start"]
            row["toggle_minus_grasp"] = tf - gf
            h, o = state[tf, EEF[hold]].astype(np.float64), state[tf, EEF[other]].astype(np.float64)
            row["press_eef_minus_hold_eef"] = (o - h).tolist()
            row["press_eef_base"] = o.tolist(); row["hold_eef_base"] = h.tolist()
            row["press_grip_cmd"] = float(act[tf, GRIP_ACTION[other]])
            row["press_fingers"] = float(state[tf, FING[other]].sum()); row["hold_fingers"] = float(state[tf, FING[hold]].sum())
            # press-hand eef travel during press segment (does the other hand do the pressing?)
            seg_o = state[press["start"]:press["end"], EEF[other]]; seg_h = state[press["start"]:press["end"], EEF[hold]]
            row["press_travel_other"] = float(np.linalg.norm(np.diff(seg_o, axis=0), axis=1).sum())
            row["press_travel_hold"] = float(np.linalg.norm(np.diff(seg_h, axis=0), axis=1).sum())
            row["hold_fingers_press_min"] = float(state[press["start"]:tf + 1, FING[hold]].sum(1).min())
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows_train.json").write_text(json.dumps(rows, indent=1))

    def q(vals):
        v = np.asarray(vals, float)
        return {"n": int(len(v)), "median": np.median(v, 0).round(4).tolist(), "p10": np.percentile(v, 10, 0).round(4).tolist(),
                "p90": np.percentile(v, 90, 0).round(4).tolist()}
    from collections import Counter
    tg = [r for r in rows if r["toggle_frame"] is not None]
    summ = {"source": "TRAIN only (split_task_000.json), lowdim_cache_task000 + parquet next.reward", "n_rows": len(rows),
            "problems": problems, "hold_hand": Counter(r["hold_hand"] for r in rows),
            "toggle_in_segment": Counter(r["toggle_in"] for r in tg), "no_toggle": [r["ep"] for r in rows if r["toggle_frame"] is None]}
    for hand in ("left", "right"):
        rr = [r for r in tg if r["hold_hand"] == hand]
        if not rr:
            continue
        summ[f"hold_{hand}"] = {
            "radio_rel_pick_start": q([r["radio_rel_pick_start"] for r in rr]),
            "radio_rel_grasp": q([r["radio_rel_grasp"] for r in rr]),
            "radio_z_grasp": q([r["radio_z_grasp"] for r in rr]),
            "base_disp_pick_m": q([r["base_disp_pick"][0] for r in rr]),
            "base_disp_press_m": q([r["base_disp_press"][0] for r in rr]),
            "press_eef_minus_hold_eef": q([r["press_eef_minus_hold_eef"] for r in rr]),
            "press_eef_base": q([r["press_eef_base"] for r in rr]),
            "hold_eef_base_at_toggle": q([r["hold_eef_base"] for r in rr]),
            "press_grip_cmd": q([r["press_grip_cmd"] for r in rr]),
            "press_fingers": q([r["press_fingers"] for r in rr]), "hold_fingers": q([r["hold_fingers"] for r in rr]),
            "press_travel_other_m": q([r["press_travel_other"] for r in rr]),
            "press_travel_hold_m": q([r["press_travel_hold"] for r in rr]),
            "toggle_minus_press_start": q([r["toggle_minus_press_start"] for r in rr]),
            "toggle_minus_grasp": q([r["toggle_minus_grasp"] for r in rr]),
            "trunk_pick_start": q([r["trunk_pick_start"] for r in rr]),
        }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=int))
    print(json.dumps(summ, indent=1, default=int)[:6000])


if __name__ == "__main__":
    sys.exit(main())
