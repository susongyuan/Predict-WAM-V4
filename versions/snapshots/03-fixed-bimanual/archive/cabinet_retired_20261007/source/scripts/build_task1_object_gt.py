#!/usr/bin/env python3
"""Offline ground truth for Task 1 object perception (DEVELOPMENT / VALIDATION ONLY, never fed to a policy).

Per demo episode (200-399) this writes, for every frame, the base-frame positions of the four task objects
(trash can + 3 soda cans), their state (floor / held / in bin) and a label-uncertainty estimate:

* Initial world poses of the robot and the objects come from the task-instance `tro_state` JSON
  (`task_instance_id` in meta/episodes). Projection of these at frame 0 lands exactly on the objects.
* Robot trajectory = measured base linear velocity + commanded yaw rate, integrated at 30 Hz.  Heading drifts
  by several percent of the accumulated rotation, so each segment between pick starts is corrected with a
  MAP fit (yaw-rate scale k, linear scale s; priors 0.15 / 0.03) to the grasp anchor of the object picked at
  the end of the segment: object position in the base frame at pick start = eef position at the grasp frame
  (last open->close transition of the gripper that is closed at the end of the pick segment), moved back to
  the pick-start base frame with the odometry in between (same rule as
  outputs/validation/task1_docking_calibration_20260926/grasp_relative_poses_train.json, 720/720 identical).
  The trash-can anchor is the rim, so the median rim->centre offset (0.016, +/-0.088) m is added.
* Label uncertainty per frame/object: 0.03 m + calibrated growth with accumulated rotation (rad) and travel (m)
  since the nearest trusted reference (t=0 or an anchor); calibrated with a skip-one-anchor test (see summary).

Output: outputs/validation/task1_perception_20260926/gt/episode_<id>.npz + gt_summary.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/behavior-1k/2026-challenge-demos"
INST = (ROOT / "data/omnigibson/2026-challenge-task-instances/scenes/house_double_floor_lower/json/"
        "house_double_floor_lower_task_picking_up_trash_instances/"
        "house_double_floor_lower_task_picking_up_trash_0_{}_template-tro_state.json")
OUT = ROOT / "outputs/validation/task1_perception_20260926/gt"
OBJECTS = ["trash_can_116", "can_of_soda_115", "can_of_soda_114", "can_of_soda_113"]
NAME2INST = {"trash_can_116": "ashcan.n.01_1", "can_of_soda_115": "can__of__soda.n.01_1",
             "can_of_soda_114": "can__of__soda.n.01_2", "can_of_soda_113": "can__of__soda.n.01_3"}
MODEL = {"trash_can_116": "wkxtxh", "can_of_soda_115": "itolcg", "can_of_soda_114": "itolcg", "can_of_soda_113": "lugwcz"}
# object half extents (radius, half height) from the asset metadata bbox_size (scale 1 in the template scene)
HALF = {"wkxtxh": (0.115, 0.139), "itolcg": (0.0376, 0.0673), "lugwcz": (0.0381, 0.0635)}
RIM_OFFSET = np.array([0.016, 0.088])  # median bin centre - right-hand rim grasp point, base frame (22 short-travel eps)
CAMS = ["zed_link_camera_0", "right_realsense_link_camera_0", "left_realsense_link_camera_0"]
STATE = {"floor": 0, "held": 1, "in_bin": 2, "unknown": 3}
GRIP_ACTION = {"left": 14, "right": 22}
EEF_POS = {"left": 17, "right": 42}
EEF_QUAT = {"left": 20, "right": 45}
DT = 1.0 / 30.0


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def integrate(v, x0, y0, th0):
    """v: (T-1, 3) robot-frame [vx, vy, wz] driving frame t -> t+1. Returns x, y, th of length T."""
    th = th0 + np.concatenate([[0.0], np.cumsum(v[:, 2] * DT)])
    thm = th[:-1] + 0.5 * v[:, 2] * DT
    dx = (v[:, 0] * np.cos(thm) - v[:, 1] * np.sin(thm)) * DT
    dy = (v[:, 0] * np.sin(thm) + v[:, 1] * np.cos(thm)) * DT
    return x0 + np.concatenate([[0.0], np.cumsum(dx)]), y0 + np.concatenate([[0.0], np.cumsum(dy)]), th


def to_base(w_xy, x, y, th):
    d = np.asarray(w_xy) - np.array([x, y])
    c, s = math.cos(th), math.sin(th)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def load_meta():
    rows = pq.read_table(DEMO / "meta/episodes/chunk-001/file-000.parquet").to_pylist()
    return {r["episode_index"]: r for r in rows}


def cam_pose_table(meta, ep):
    r = meta[ep]
    f = DEMO / f"data/chunk-001/file-{r['data/file_index']:03d}.parquet"
    cols = ["episode_index", "frame_index"] + [f"observation.robot2cam_pose.{c}" for c in CAMS]
    t = pq.read_table(f, columns=cols, filters=[("episode_index", "=", ep)])
    order = np.argsort(np.array(t.column("frame_index")))
    return {c: np.stack(t.column(f"observation.robot2cam_pose.{c}").to_numpy(zero_copy_only=False))[order]
            for c in CAMS}


def find_picks(segs, act):
    picks = []
    for i, seg in enumerate(segs):
        if seg["skill"] != "pick up from":
            continue
        st, en = seg["start"], seg["end"]
        cands = []
        for hand, gi in GRIP_ACTION.items():
            ga = act[:, gi]
            tr = [t for t in range(max(st, 1), min(en, len(ga))) if ga[t - 1] > 0 and ga[t] <= 0]
            if tr and ga[min(en, len(ga) - 1)] <= 0:
                cands.append((hand, tr[-1]))
        if len(cands) != 1:
            raise ValueError(f"ambiguous grasp in segment {i}: {cands}")
        picks.append({"seg": i, "object": seg["objects"][0][0], "start": st, "end": en,
                      "hand": cands[0][0], "grasp": cands[0][1]})
    return picks


def anchor_rel(state, v, pick):
    """Object position (x, y) in the base frame at pick start, from grasp kinematics."""
    st, gf, hand = pick["start"], pick["grasp"], pick["hand"]
    x, y, th = integrate(v[st:gf], 0.0, 0.0, 0.0)
    e = state[gf, EEF_POS[hand]:EEF_POS[hand] + 2].astype(np.float64)
    c, s = math.cos(th[-1]), math.sin(th[-1])
    rel = np.array([x[-1] + c * e[0] - s * e[1], y[-1] + s * e[0] + c * e[1]])
    if pick["object"].startswith("trash_can"):
        rel = rel + (RIM_OFFSET if hand == "right" else RIM_OFFSET * np.array([1, -1]))
    return rel


def fit_segment(v_seg, pose_a, w_obj, rel_obj, sigma_anchor, sigma_k=0.15, sigma_s=0.03):
    """MAP (k, s) so that the object's predicted base-frame position at the segment end equals the anchor."""
    def resid(p):
        k, s = p
        vv = v_seg * np.array([s, s, 1.0 + k])
        x, y, th = integrate(vv, *pose_a)
        pr = to_base(w_obj, x[-1], y[-1], th[-1])
        return np.concatenate([(pr - rel_obj) / sigma_anchor, [k / sigma_k, (s - 1.0) / sigma_s]])
    p = np.array([0.0, 1.0])
    for _ in range(12):
        r = resid(p)
        J = np.zeros((4, 2))
        for j in range(2):
            dp = np.zeros(2); dp[j] = 1e-5
            J[:, j] = (resid(p + dp) - r) / 1e-5
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        p = p + step
        if np.abs(step).max() < 1e-7:
            break
    return p


def build(ep, meta, segs_all, write=True):
    r = meta[ep]
    ipath = Path(str(INST).format(r["task_instance_id"]))
    if not ipath.exists():
        return {"episode": ep, "skipped": f"missing instance {r['task_instance_id']}"}
    inst = json.loads(ipath.read_text())
    state = np.load(ROOT / f"data/lowdim_cache_task001/episode_{ep}_state.npy").astype(np.float64)
    act = np.load(ROOT / f"data/lowdim_cache_task001/episode_{ep}_action.npy").astype(np.float64)
    T = len(state)
    segs = segs_all[str(ep)]["segments"]
    v = state[1:, 0:3].copy()
    v[:, 2] = act[:-1, 2]  # commanded yaw rate integrates more consistently than the measured one
    rp = inst["robot_poses"]["R1Pro"][0]
    pose0 = (rp["position"][0], rp["position"][1], yaw_of(rp["orientation"]))
    world = {o: np.array(inst[NAME2INST[o]]["root_link"]["pos"], dtype=np.float64) for o in OBJECTS}
    picks = find_picks(segs, act)
    by_obj = {p["object"]: p for p in picks}
    anchors = sorted(((p["start"], p["object"], anchor_rel(state, v, p)) for p in picks), key=lambda a: a[0])

    # raw (uncorrected) forward error at each anchor, for the summary
    xr, yr, thr = integrate(v, *pose0)
    raw_err = {o: float(np.hypot(*(to_base(world[o][:2], xr[t], yr[t], thr[t]) - rel))) for t, o, rel in anchors}

    # sequential per-segment correction
    vc = v.copy()
    pose_a, t_a, params = pose0, 0, []
    for t_b, o, rel in anchors:
        if t_b <= t_a:
            continue
        sig = 0.07 if o.startswith("trash_can") else 0.03
        k, s = fit_segment(v[t_a:t_b], pose_a, world[o][:2], rel, sig)
        vc[t_a:t_b] = v[t_a:t_b] * np.array([s, s, 1.0 + k])
        x, y, th = integrate(vc[t_a:t_b], *pose_a)
        pose_a, t_a = (x[-1], y[-1], th[-1]), t_b
        params.append({"object": o, "t": int(t_b), "k": float(k), "s": float(s),
                       "resid": float(np.hypot(*(to_base(world[o][:2], x[-1], y[-1], th[-1]) - rel)))})
    x, y, th = integrate(vc, *pose0)

    # skip-one-anchor test: refit with one interior anchor removed and measure its prediction error
    skip = []
    for i in range(len(anchors) - 1):
        keep = [a for j, a in enumerate(anchors) if j != i]
        vt = v.copy(); pa, ta = pose0, 0
        for t_b, o, rel in keep:
            sig = 0.07 if o.startswith("trash_can") else 0.03
            k, s = fit_segment(v[ta:t_b], pa, world[o][:2], rel, sig)
            vt[ta:t_b] = v[ta:t_b] * np.array([s, s, 1.0 + k])
            xx, yy, tt = integrate(vt[ta:t_b], *pa)
            pa, ta = (xx[-1], yy[-1], tt[-1]), t_b
        xx, yy, tt = integrate(vt, *pose0)
        t_i, o_i, rel_i = anchors[i]
        prev_t = anchors[i - 1][0] if i > 0 else 0
        next_t = anchors[i + 1][0]
        rot = np.abs(v[:, 2]) * DT; trav = np.hypot(v[:, 0], v[:, 1]) * DT
        dr = min(rot[prev_t:t_i].sum(), rot[t_i:next_t].sum()); dl = min(trav[prev_t:t_i].sum(), trav[t_i:next_t].sum())
        skip.append({"object": o_i, "t": int(t_i), "err": float(np.hypot(*(to_base(world[o_i][:2], xx[t_i], yy[t_i], tt[t_i]) - rel_i))),
                     "raw_err": raw_err[o_i], "rot_to_ref": float(dr), "trav_to_ref": float(dl)})

    # per-frame object states and base-frame positions
    n_obj = len(OBJECTS)
    pos_b = np.full((T, n_obj, 3), np.nan, dtype=np.float32)
    st_arr = np.full((T, n_obj), STATE["unknown"], dtype=np.uint8)
    rot_ref = np.full((T, n_obj), np.nan, dtype=np.float32)
    trav_ref = np.full((T, n_obj), np.nan, dtype=np.float32)
    cum_rot = np.concatenate([[0.0], np.cumsum(np.abs(v[:, 2]) * DT)])
    cum_trav = np.concatenate([[0.0], np.cumsum(np.hypot(v[:, 0], v[:, 1]) * DT)])
    ref_times = np.array([0] + [a[0] for a in anchors])
    c, s_ = np.cos(th), np.sin(th)
    place_end = {}
    for seg in segs:
        if seg["skill"] == "place in":
            place_end[seg["objects"][0][0]] = seg["end"]
    place_on = [seg for seg in segs if seg["skill"] == "place on"]
    for j, o in enumerate(OBJECTS):
        p = by_obj[o]
        g = p["grasp"]
        dxw, dyw = world[o][0] - x, world[o][1] - y
        bx, by = c * dxw + s_ * dyw, -s_ * dxw + c * dyw
        pos_b[:g, j, 0], pos_b[:g, j, 1], pos_b[:g, j, 2] = bx[:g], by[:g], world[o][2]
        st_arr[:g, j] = STATE["floor"]
        # distance (rotation / travel) to the nearest trusted reference frame
        idx = np.clip(np.searchsorted(ref_times, np.arange(g)), 0, len(ref_times) - 1)
        prev_ref = ref_times[np.clip(idx - 1, 0, None)]
        prev_ref = np.where(ref_times[idx] == np.arange(g), ref_times[idx], prev_ref)
        next_ref = ref_times[idx]
        rot_ref[:g, j] = np.minimum(cum_rot[np.arange(g)] - cum_rot[prev_ref], np.abs(cum_rot[next_ref] - cum_rot[np.arange(g)]))
        trav_ref[:g, j] = np.minimum(cum_trav[np.arange(g)] - cum_trav[prev_ref], np.abs(cum_trav[next_ref] - cum_trav[np.arange(g)]))
        hand = p["hand"]
        if o.startswith("can"):
            pe = place_end.get(o, T)
            st_arr[g:pe, j] = STATE["held"]
            st_arr[pe:, j] = STATE["in_bin"]
            pos_b[g:pe, j] = state[g:pe, EEF_POS[hand]:EEF_POS[hand] + 3]
        else:
            end_hold = place_on[0]["end"] if place_on else T
            st_arr[g:end_hold, j] = STATE["held"]
            pos_b[g:end_hold, j] = state[g:end_hold, EEF_POS[hand]:EEF_POS[hand] + 3]
    bin_hand = by_obj["trash_can_116"]["hand"]
    for j, o in enumerate(OBJECTS):
        if o.startswith("can"):
            m = st_arr[:, j] == STATE["in_bin"]
            pos_b[m, j] = state[m, EEF_POS[bin_hand]:EEF_POS[bin_hand] + 3]
    cams = cam_pose_table(meta, ep)
    info = {"episode": ep, "instance": r["task_instance_id"], "length": T, "picks": picks, "bin_hand": bin_hand,
            "corrections": params, "raw_anchor_err": raw_err, "skip_one": skip}
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"episode_{ep}.npz", pos_base=pos_b, state=st_arr, rot_to_ref=rot_ref,
                            trav_to_ref=trav_ref, robot_xyth=np.stack([x, y, th], 1).astype(np.float32),
                            cam_zed=cams["zed_link_camera_0"].astype(np.float32),
                            cam_right=cams["right_realsense_link_camera_0"].astype(np.float32),
                            cam_left=cams["left_realsense_link_camera_0"].astype(np.float32),
                            eef_left=state[:, 17:20].astype(np.float32), eef_right=state[:, 42:45].astype(np.float32),
                            objects=np.array(OBJECTS), models=np.array([MODEL[o] for o in OBJECTS]),
                            world=np.stack([world[o] for o in OBJECTS]).astype(np.float32),
                            info=json.dumps(info))
    return info


def _worker(args):
    ep, = args
    meta = load_meta()
    segs = json.loads((DEMO / "skill_segments_task001.json").read_text())["episodes"]
    try:
        return build(ep, meta, segs)
    except Exception as exc:  # report, never silently drop
        return {"episode": ep, "error": repr(exc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    meta = load_meta()
    eps = args.episodes or sorted(meta)
    with mp.Pool(args.workers) as pool:
        infos = pool.map(_worker, [(e,) for e in eps])
    split = json.loads((DEMO / "split_task_001.json").read_text())
    val = set(split["validation_episode_indices"])
    ok = [i for i in infos if "skip_one" in i]
    skip = [s | {"episode": i["episode"]} for i in ok for s in i["skip_one"]]
    corr = [c for i in ok for c in i["corrections"]]

    def stats(vals):
        vals = np.asarray(vals, dtype=float)
        return {"n": int(len(vals)), "median": float(np.median(vals)), "p75": float(np.percentile(vals, 75)),
                "p90": float(np.percentile(vals, 90))} if len(vals) else {"n": 0}
    rot = np.array([s["rot_to_ref"] for s in skip]); err = np.array([s["err"] for s in skip])
    raw = np.array([s["raw_err"] for s in skip])
    bins = {}
    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99)]:
        m = (rot >= lo) & (rot < hi)
        bins[f"rot_{lo}-{hi}"] = {"corrected": stats(err[m]), "raw_forward": stats(raw[m])}
    summary = {"episodes": len(eps), "built": len(ok), "validation_built": sum(i["episode"] in val for i in ok),
               "skipped_or_error": [i for i in infos if "skip_one" not in i],
               "skip_one_anchor_err_m": stats(err), "skip_one_raw_forward_err_m": stats(raw),
               "skip_one_by_rotation_to_reference": bins,
               "fit_resid_m": stats([c["resid"] for c in corr]),
               "fit_k": stats([c["k"] for c in corr]), "fit_s": stats([c["s"] for c in corr])}
    (OUT.parent / "gt_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "skipped_or_error"}, indent=1))
    print("skipped/error:", summary["skipped_or_error"])


if __name__ == "__main__":
    main()
