#!/usr/bin/env python3
"""Offline ground truth for Task 0 (turning_on_radio) radio perception. DEVELOPMENT / EVALUATION ONLY - never a policy
input (it uses the simulator's task-instance poses and future frames of the demo).

Per demo episode (0-199, chunk-000) this writes, for every frame, the radio's base-frame centre, its base-frame yaw, its
state (on the table before the grasp / held / placed back) and a label-uncertainty estimate:

* Initial world poses of the robot and the radio come from the task-instance `tro_state` JSON
  (`radio_receiver.n.01_1`, `robot_poses`->`R1Pro`), instance = meta/episodes `task_instance_id`. The mapping is verified
  on frame 0 of every episode's head video: the radio projected with the instance poses must land on red pixels, and the
  neighbouring instance ids (id-1, id+1) must not (see `mapping_check` in gt_summary.json).
* GT point = radio BBOX CENTRE = root_link pos + R_radio @ base_link_offset (asset wxnicr metadata; the root link
  origin is 2.2 cm off the visual centre along the radio's x axis). Radio yaw = yaw of the radio's x axis (the short,
  0.138 m axis; the long 0.321 m axis is its y axis, the handle runs along it).
* Robot trajectory = measured base linear velocity state[t+1, 0:2] + commanded yaw rate action[t, 2], integrated at
  30 Hz. Raw forward odometry misses the grasp anchor by 0.235 m median (TRAIN, right-hand grasps); a single global
  yaw-rate scale K0 = -0.10 (commanded yaw rate over-predicts by 10 %; grid search on TRAIN only, see
  /tmp exploration logged in gt_summary `odometry`) brings that to 0.072 m median / 0.21 m p90.
  Per episode a MAP fit (yaw scale k ~ N(K0, 0.05), linear scale s ~ N(1, 0.03)) then pins the radio to the grasp
  anchor at the grasp frame (sigma 0.05 m).
* Grasp anchor: the grasp frame is the last open->close transition of the gripper that is still closed at the end of the
  `pick up from` segment (same rule as scripts/analyze_task0_demo_geometry.py). Radio centre at the grasp frame =
  EEF position - GRASP_OFFSET, GRASP_OFFSET = median (EEF - radio centre) in the RADIO frame over TRAIN right-hand
  grasps with the K0 odometry (handle on top, ~centred). Episodes with an ambiguous grasp hand (both grippers close in
  the pick segment) are anchored with the hand that is still closed at the start of `press`; if that is also ambiguous
  the episode keeps the K0 odometry without anchor (flag `anchored` False).
* Held: radio pose rigidly attached to the holding EEF (EEF quaternions state[20:24] left / [45:49] right, xyzw,
  base frame). Placed: first close->open of the holding gripper after the grasp; the radio stays at the pose it had in
  the hand at that frame, fixed in the world (odometry frame) afterwards.
* Label uncertainty (m, 1-sigma-ish): pre-grasp frames: UNC0 + growth with accumulated |rotation| (rad) and travel (m)
  to the nearest trusted reference (frame 0 = exact instance pose, grasp frame = anchor). Growth rates are fitted on
  TRAIN only as the 75th-percentile ratio of the K0-odometry anchor miss to the rotation/travel since frame 0.
  Held frames: 0.04 m (grasp spread). Placed frames: anchor uncertainty 0.06 m + growth since the release.

Output: outputs/validation/task0_perception_20260927/gt/episode_<id>.npz + ../gt_summary.json
`nice -n 10 .conda-envs/lilawam/bin/python scripts/build_task0_object_gt.py [--episodes ...] [--workers 4]`
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/behavior-1k/2026-challenge-demos"
LOW = ROOT / "data/lowdim_cache_task000"
INST = (ROOT / "data/omnigibson/2026-challenge-task-instances/scenes/house_double_floor_lower/json/"
        "house_double_floor_lower_task_turning_on_radio_instances/"
        "house_double_floor_lower_task_turning_on_radio_0_{}_template-tro_state.json")
ASSET = ROOT / "data/omnigibson/behavior-1k-assets/objects/radio/wxnicr/misc/metadata.json"
OUT = ROOT / "outputs/validation/task0_perception_20260927/gt"
HEAD_KEY = "observation.rgb.zed_link_camera_0"
STATE = {"table": 0, "held": 1, "placed": 2, "unknown": 3}
GRIP_ACTION = {"left": 14, "right": 22}
EEF_POS = {"left": slice(17, 20), "right": slice(42, 45)}
EEF_QUAT = {"left": slice(20, 24), "right": slice(45, 49)}
FINGERS = {"left": slice(24, 26), "right": slice(49, 51)}
DT = 1.0 / 30.0
K0, SIGMA_K, SIGMA_S, SIGMA_ANCHOR = -0.10, 0.05, 0.03, 0.05
UNC0, UNC_HELD, UNC_PLACED = 0.02, 0.04, 0.06


def asset_geometry():
    m = json.loads(ASSET.read_text())
    return np.array(m["bbox_size"], float), np.array(m["base_link_offset"], float)


BBOX_SIZE, BASE_LINK_OFFSET = asset_geometry()        # (0.138, 0.321, 0.235), (-0.022, -0.002, 0.0065)


def quat_to_mat(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def integrate(v, x0, y0, th0):
    """v: (N, 3) robot-frame [vx, vy, wz] driving frame t -> t+1. Returns x, y, th of length N+1."""
    th = th0 + np.concatenate([[0.0], np.cumsum(v[:, 2] * DT)])
    thm = th[:-1] + 0.5 * v[:, 2] * DT
    dx = (v[:, 0] * np.cos(thm) - v[:, 1] * np.sin(thm)) * DT
    dy = (v[:, 0] * np.sin(thm) + v[:, 1] * np.cos(thm)) * DT
    return x0 + np.concatenate([[0.0], np.cumsum(dx)]), y0 + np.concatenate([[0.0], np.cumsum(dy)]), th


def velocities(state, act):
    """(T-1, 3): measured linear velocity of frame t+1 and commanded yaw rate of frame t (drive t -> t+1)."""
    return np.c_[state[1:, 0:2], act[:-1, 2]].astype(np.float64)


def load_meta():
    rows = pq.read_table(DEMO / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    return {r["episode_index"]: r for r in rows}


def load_instance(iid):
    p = Path(str(INST).format(iid))
    if not p.exists():
        return None
    inst = json.loads(p.read_text())
    rp = inst["robot_poses"]["R1Pro"][0]
    rl = inst["radio_receiver.n.01_1"]["root_link"]
    Rr = quat_to_mat(rl["ori"])
    root = np.array(rl["pos"], float)
    return {"pose0": (rp["position"][0], rp["position"][1], yaw_of(rp["orientation"])), "root": root, "R": Rr,
            "centre": root + Rr @ BASE_LINK_OFFSET, "yaw": math.atan2(Rr[1, 0], Rr[0, 0]),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(Rr[2, 2], -1, 1))))}


def cam_pose_table(meta, ep):
    r = meta[ep]
    f = DEMO / f"data/chunk-000/file-{r['data/file_index']:03d}.parquet"
    cols = ["frame_index", "observation.robot2cam_pose.zed_link_camera_0"]
    t = pq.read_table(f, columns=cols, filters=[("episode_index", "=", ep)])
    order = np.argsort(np.array(t.column("frame_index")))
    return np.stack(t.column(cols[1]).to_numpy(zero_copy_only=False))[order].astype(np.float64)


def grasp(segs, act, state=None):
    """(hand, grasp frame, ambiguous?, pick segment).

    Primary rule: the only gripper with an open->close transition in the pick segment that is still closed at the pick
    end. Ambiguous cases (9/199 episodes: the free hand is closed into a fist for pressing, or the closure falls after
    the annotated pick end) are resolved with the finger width 20 frames after each candidate closure: a gripper
    blocked by the handle stays at 0.02-0.085 m finger sum (median 0.038), an empty fist closes to < 0.02."""
    pick = next(s for s in segs if s["skill"] == "pick up from")
    cands = []
    for hand, gi in GRIP_ACTION.items():
        ga = act[:, gi]
        en = min(pick["end"], len(ga) - 1)
        tr = [t for t in range(max(pick["start"], 1), en + 1) if ga[t - 1] > 0 and ga[t] <= 0]
        if tr and ga[en] <= 0:
            cands.append((hand, tr[-1]))
    if len(cands) == 1:
        return cands[0][0], cands[0][1], False, pick
    if state is None:
        return None, None, True, pick
    press = next((s for s in segs if s["skill"] == "press"), None)
    hi = min(press["end"] if press else pick["end"], len(act) - 1)
    cands = []
    for hand, gi in GRIP_ACTION.items():
        ga = act[:, gi]
        tr = [t for t in range(max(pick["start"], 1), hi + 1) if ga[t - 1] > 0 and ga[t] <= 0]
        if tr:
            # last closure that is blocked by an object (a re-grasp supersedes an earlier attempt)
            for t in reversed(tr):
                f = float(state[min(t + 20, len(state) - 1), FINGERS[hand]].sum())
                if 0.02 <= f <= 0.085:
                    cands.append((hand, t))
                    break
    if len(cands) == 1:
        return cands[0][0], cands[0][1], True, pick
    return None, None, True, pick


def release(act, hand, gf):
    ga = act[:, GRIP_ACTION[hand]]
    tr = [t for t in range(gf + 1, len(ga)) if ga[t - 1] <= 0 and ga[t] > 0]
    return tr[0] if tr else None


def anchor_miss(v, inst, state, hand, gf, k, s):
    """Radio-frame (EEF - radio centre) at the grasp frame with odometry scales (k, s)."""
    x, y, th = integrate(v[:gf] * np.array([s, s, 1.0 + k]), *inst["pose0"])
    c_b = rotz(-th[-1]) @ (inst["centre"] - np.array([x[-1], y[-1], 0.0]))
    e = state[gf, EEF_POS[hand]].astype(np.float64)
    R_rb = rotz(-th[-1]) @ inst["R"]                     # radio -> base at gf
    return R_rb.T @ (e - c_b), (x, y, th)


def fit(v_seg, inst, target_base_xy, gf):
    """MAP (k, s): radio centre predicted in the base frame at gf equals the anchor."""
    w = inst["centre"][:2]

    def resid(p):
        k, s = p
        x, y, th = integrate(v_seg * np.array([s, s, 1.0 + k]), *inst["pose0"])
        d = w - np.array([x[-1], y[-1]])
        c, s_ = math.cos(th[-1]), math.sin(th[-1])
        pr = np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]])
        return np.concatenate([(pr - target_base_xy) / SIGMA_ANCHOR, [(k - K0) / SIGMA_K, (s - 1.0) / SIGMA_S]])
    p = np.array([K0, 1.0])
    for _ in range(15):
        r = resid(p)
        J = np.zeros((4, 2))
        for j in range(2):
            dp = np.zeros(2)
            dp[j] = 1e-5
            J[:, j] = (resid(p + dp) - r) / 1e-5
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        p = p + step
        if np.abs(step).max() < 1e-8:
            break
    return p, float(np.linalg.norm(resid(p)[:2]) * SIGMA_ANCHOR)


def build(ep, meta, segs_all, cal, write=True):
    r = meta[ep]
    inst = load_instance(r["task_instance_id"])
    if inst is None:
        return {"episode": ep, "skipped": f"missing task instance {r['task_instance_id']}"}
    state = np.load(LOW / f"episode_{ep}_state.npy").astype(np.float64)
    act = np.load(LOW / f"episode_{ep}_action.npy").astype(np.float64)
    T = len(state)
    segs = segs_all[str(ep)]["segments"]
    v = velocities(state, act)
    hand, gf, ambiguous, pick = grasp(segs, act, state)
    info = {"episode": ep, "instance": int(r["task_instance_id"]), "length": T, "hand": hand, "grasp_frame": gf,
            "grasp_hand_ambiguous": ambiguous, "pick_start": pick["start"], "radio_tilt_deg": inst["tilt_deg"]}
    k, s, anchored = K0, 1.0, False
    if hand is not None:
        miss_raw, _ = anchor_miss(v, inst, state, hand, gf, 0.0, 1.0)
        miss_k0, _ = anchor_miss(v, inst, state, hand, gf, K0, 1.0)
        info.update(anchor_miss_raw_radio=miss_raw.tolist(), anchor_miss_k0_radio=miss_k0.tolist())
        if cal is not None:
            # anchor = EEF - R_radio_base @ GRASP_OFFSET (radio frame); R_radio_base from the K0 odometry heading
            _, (_, _, th0) = anchor_miss(v, inst, state, hand, gf, K0, 1.0)
            R_rb = rotz(-th0[-1]) @ inst["R"]
            target = state[gf, EEF_POS[hand]] - R_rb @ np.asarray(cal["grasp_offset_radio"])
            (k, s), res = fit(v[:gf], inst, target[:2], gf)
            anchored = True
            info.update(fit_k=float(k), fit_s=float(s), fit_resid=res)
    vc = v * np.array([s, s, 1.0 + k])
    x, y, th = integrate(vc, *inst["pose0"])
    rot = np.concatenate([[0.0], np.cumsum(np.abs(v[:, 2]) * DT)])
    trav = np.concatenate([[0.0], np.cumsum(np.hypot(v[:, 0], v[:, 1]) * DT)])
    rate_rot, rate_trav = (cal["unc_rate_rot"], cal["unc_rate_trav"]) if cal else (0.05, 0.05)

    pos = np.full((T, 3), np.nan, np.float32)
    yaw_b = np.full(T, np.nan, np.float32)
    st = np.full(T, STATE["unknown"], np.uint8)
    unc = np.full(T, np.nan, np.float32)
    c, s_ = np.cos(th), np.sin(th)

    def world_to_base(pw, idx):
        dx, dy = pw[0] - x[idx], pw[1] - y[idx]
        return np.stack([c[idx] * dx + s_[idx] * dy, -s_[idx] * dx + c[idx] * dy, np.full(len(idx), pw[2])], 1)

    g_end = gf if gf is not None else T
    idx = np.arange(g_end)
    pos[:g_end] = world_to_base(inst["centre"], idx)
    yaw_b[:g_end] = np.angle(np.exp(1j * (inst["yaw"] - th[:g_end])))
    st[:g_end] = STATE["table"]
    grow0 = rate_rot * rot[:g_end] + rate_trav * trav[:g_end]
    if anchored:
        growa = rate_rot * (rot[gf] - rot[:g_end]) + rate_trav * (trav[gf] - trav[:g_end]) + SIGMA_ANCHOR
        unc[:g_end] = UNC0 + np.minimum(grow0, growa)
    else:
        unc[:g_end] = UNC0 + grow0
    rel = None
    if gf is not None:
        # rigid attachment to the holding EEF, from the (corrected) radio pose at the grasp frame
        cb = world_to_base(inst["centre"], np.array([gf]))[0]
        R_rb = rotz(-th[gf]) @ inst["R"]
        Re_g = quat_to_mat(state[gf, EEF_QUAT[hand]])
        e_g = state[gf, EEF_POS[hand]]
        o_e = Re_g.T @ (cb - e_g)
        R_re = Re_g.T @ R_rb
        rel = release(act, hand, gf)
        h_end = rel if rel is not None else T
        for t in range(gf, h_end):
            Re = quat_to_mat(state[t, EEF_QUAT[hand]])
            pos[t] = state[t, EEF_POS[hand]] + Re @ o_e
            Rr = Re @ R_re
            yaw_b[t] = math.atan2(Rr[1, 0], Rr[0, 0])
        st[gf:h_end] = STATE["held"]
        unc[gf:h_end] = UNC_HELD
        info.update(release_frame=rel, grasp_offset_eef=o_e.tolist())
        if rel is not None:
            Re = quat_to_mat(state[rel, EEF_QUAT[hand]])
            pb = state[rel, EEF_POS[hand]] + Re @ o_e
            pw = np.array([x[rel] + c[rel] * pb[0] - s_[rel] * pb[1], y[rel] + s_[rel] * pb[0] + c[rel] * pb[1], pb[2]])
            Rr = Re @ R_re
            yw = math.atan2(Rr[1, 0], Rr[0, 0]) + th[rel]
            idx = np.arange(rel, T)
            pos[rel:] = world_to_base(pw, idx)
            yaw_b[rel:] = np.angle(np.exp(1j * (yw - th[rel:])))
            st[rel:] = STATE["placed"]
            unc[rel:] = UNC_PLACED + rate_rot * (rot[rel:] - rot[rel]) + rate_trav * (trav[rel:] - trav[rel])
            info.update(placed_z=float(pb[2]), placed_tilt_deg=float(np.degrees(np.arccos(np.clip(Rr[2, 2], -1, 1)))))
    info.update(anchored=anchored, rot_to_grasp=None if gf is None else float(rot[gf]),
                trav_to_grasp=None if gf is None else float(trav[gf]))
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"episode_{ep}.npz", pos_base=pos, yaw_base=yaw_b, state=st, unc=unc,
                            robot_xyth=np.stack([x, y, th], 1).astype(np.float32), cam_zed=cam_pose_table(meta, ep).astype(np.float32),
                            eef_left=state[:, 17:20].astype(np.float32), eef_right=state[:, 42:45].astype(np.float32),
                            fingers=np.c_[state[:, FINGERS["left"]].sum(1), state[:, FINGERS["right"]].sum(1)].astype(np.float32),
                            world_centre=inst["centre"].astype(np.float32), world_yaw=np.float32(inst["yaw"]),
                            bbox_size=BBOX_SIZE.astype(np.float32), info=json.dumps(info))
    return info


# ---- calibration (TRAIN only) ------------------------------------------------------------------------------------------
def calibrate(meta, segs_all, train):
    offs, ratios = [], []
    for ep in train:
        inst = load_instance(meta[ep]["task_instance_id"])
        if inst is None:
            continue
        state = np.load(LOW / f"episode_{ep}_state.npy").astype(np.float64)
        act = np.load(LOW / f"episode_{ep}_action.npy").astype(np.float64)
        hand, gf, amb, _ = grasp(segs_all[str(ep)]["segments"], act, state)
        if hand != "right" or amb:
            continue
        v = velocities(state, act)
        miss, _ = anchor_miss(v, inst, state, hand, gf, K0, 1.0)
        rot = float(np.abs(v[:gf, 2]).sum() * DT)
        trav = float(np.hypot(v[:gf, 0], v[:gf, 1]).sum() * DT)
        offs.append(miss)
        ratios.append((rot, trav))
    offs = np.array(offs)
    off = np.median(offs, 0)
    err = np.linalg.norm(offs[:, :2] - off[:2], axis=1)
    rt = np.array(ratios)
    # uncertainty growth: 75th percentile of err / (rotation + travel) split proportionally by a least-squares fit
    A = rt
    coef = np.linalg.lstsq(A, err, rcond=None)[0]
    coef = np.clip(coef, 0.0, None)
    scale = float(np.percentile(err / np.maximum(A @ coef, 1e-3), 75))
    return {"grasp_offset_radio": off.tolist(), "n": int(len(offs)),
            "grasp_offset_radio_p25": np.percentile(offs, 25, 0).tolist(), "grasp_offset_radio_p75": np.percentile(offs, 75, 0).tolist(),
            "k0_anchor_miss_xy_m": {"median": float(np.median(err)), "p75": float(np.percentile(err, 75)),
                                    "p90": float(np.percentile(err, 90))},
            "unc_rate_rot": float(coef[0] * scale), "unc_rate_trav": float(coef[1] * scale),
            "unc_model": "unc = UNC0 + rate_rot * |rotation| + rate_trav * travel to the nearest reference; rates = LSQ fit "
                         "of the K0 anchor miss vs (rotation, travel) since frame 0, scaled to its 75th percentile"}


# ---- mapping check (frame 0 of the head video) ---------------------------------------------------------------------------
def frame0(meta, ep):
    import av
    row = meta[ep]
    p = f"videos/{HEAD_KEY}"
    path = DEMO / f"videos/{HEAD_KEY}/chunk-{row[p + '/chunk_index']:03d}/file-{row[p + '/file_index']:03d}.mp4"
    start = float(row[p + "/from_timestamp"])
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        c.seek(int(max(0.0, start - 1.0) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            if fr.pts is not None and float(fr.pts * fr.time_base) >= start - 0.5 / 30:
                return fr.to_ndarray(format="rgb24")
    return None


def red_fraction(img, cam7, centre_base):
    """Fraction of saturated red pixels in a 15x15 window at the projected radio centre (None: not in the image)."""
    import cv2
    R = quat_to_mat(cam7[3:7]) @ np.diag([1.0, -1.0, -1.0])
    pc = R.T @ (centre_base - cam7[:3])
    if pc[2] < 0.1:
        return None
    u, v = 306.0 * pc[0] / pc[2] + 360.0, 306.0 * pc[1] / pc[2] + 360.0
    if not (8 <= u < 712 and 8 <= v < 712):
        return None
    hsv = cv2.cvtColor(img[int(v) - 7:int(v) + 8, int(u) - 7:int(u) + 8], cv2.COLOR_RGB2HSV)
    h, s_, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float((((h <= 8) | (h >= 172)) & (s_ >= 120) & (vv >= 70)).mean())


def mapping_check(ep):
    meta = load_meta()
    img = frame0(meta, ep)
    cam7 = cam_pose_table(meta, ep)[0]
    iid = meta[ep]["task_instance_id"]
    out = {"episode": ep, "instance": iid}
    for name, j in (("id", iid), ("id_minus_1", iid - 1), ("id_plus_1", iid + 1)):
        inst = load_instance(j)
        if inst is None or img is None:
            out[name] = None
            continue
        x0, y0, th0 = inst["pose0"]
        cb = rotz(-th0) @ (inst["centre"] - np.array([x0, y0, 0.0]))
        out[name] = red_fraction(img, cam7, cb)
    return out


def _worker(args):
    ep, cal = args
    meta = load_meta()
    segs = json.loads((DEMO / "skill_segments_task000.json").read_text())["episodes"]
    try:
        return build(ep, meta, segs, cal)
    except Exception as exc:  # report, never silently drop
        return {"episode": ep, "error": repr(exc)}


def stats(vals):
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    return {"n": int(len(vals)), "median": float(np.median(vals)), "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)), "max": float(vals.max())} if len(vals) else {"n": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-mapping-check", action="store_true")
    args = ap.parse_args()
    meta = load_meta()
    segs_all = json.loads((DEMO / "skill_segments_task000.json").read_text())["episodes"]
    split = json.loads((DEMO / "split_task_000.json").read_text())
    train, val = [int(e) for e in split["train_episode_indices"]], [int(e) for e in split["validation_episode_indices"]]
    cal = calibrate(meta, segs_all, train)
    print("calibration (TRAIN):", json.dumps(cal, indent=1), flush=True)
    eps = args.episodes or sorted(meta)
    with mp.Pool(min(args.workers, 4)) as pool:
        infos = pool.map(_worker, [(e, cal) for e in eps])
        checks = [] if args.skip_mapping_check else pool.map(mapping_check, eps)
    ok = [i for i in infos if "anchored" in i]
    trn = [i for i in ok if i["episode"] in set(train)]

    def miss_norm(i, key):
        m = i.get(key)
        return None if m is None else float(np.hypot(m[0] - cal["grasp_offset_radio"][0], m[1] - cal["grasp_offset_radio"][1]))
    summary = {
        "statement": "Offline labels for development/evaluation only; never a policy input.",
        "episodes": len(eps), "built": len(ok), "train_built": len(trn), "validation_built": sum(i["episode"] in set(val) for i in ok),
        "skipped_or_error": [i for i in infos if "anchored" not in i],
        "point_convention": "pos_base = radio bbox centre (root + R @ base_link_offset); yaw_base = yaw of the radio x axis "
                            "(short 0.138 m axis) in the base frame; state 0 table / 1 held / 2 placed / 3 unknown",
        "radio_asset": {"model": "wxnicr", "bbox_size_m": BBOX_SIZE.tolist(), "base_link_offset_m": BASE_LINK_OFFSET.tolist()},
        "odometry": {"velocity": "state[t+1, 0:2] measured vx, vy + action[t, 2] commanded yaw rate",
                     "global_yaw_scale_K0": K0, "K0_source": "grid search on TRAIN right-hand grasps (k in -0.14..-0.06, s in "
                     "0.98..1.02): median anchor miss 0.235 m raw -> 0.072 m at k=-0.10, s=1.00 (p90 0.21 m)",
                     "per_episode_prior": {"sigma_k": SIGMA_K, "sigma_s": SIGMA_S, "sigma_anchor_m": SIGMA_ANCHOR}},
        "calibration_train_only": cal,
        "anchor_miss_raw_m_train": stats([miss_norm(i, "anchor_miss_raw_radio") for i in trn if i["hand"] == "right" and not i["grasp_hand_ambiguous"]]),
        "anchor_miss_k0_m_train": stats([miss_norm(i, "anchor_miss_k0_radio") for i in trn if i["hand"] == "right" and not i["grasp_hand_ambiguous"]]),
        "anchor_miss_k0_m_validation": stats([miss_norm(i, "anchor_miss_k0_radio") for i in ok if i["episode"] in set(val) and i["hand"] == "right"]),
        "fit_k": stats([i.get("fit_k") for i in ok]), "fit_s": stats([i.get("fit_s") for i in ok]),
        "fit_resid_m": stats([i.get("fit_resid") for i in ok]),
        "hold_hand": {h: sum(i["hand"] == h for i in ok) for h in ("left", "right", None)},
        "ambiguous_grasp_resolved_by_press_start": [i["episode"] for i in ok if i["grasp_hand_ambiguous"] and i["hand"]],
        "not_anchored": [i["episode"] for i in ok if not i["anchored"]],
        "placed_z_m": stats([i.get("placed_z") for i in ok]), "placed_tilt_deg": stats([i.get("placed_tilt_deg") for i in ok]),
        "initial_radio_tilt_deg": stats([i["radio_tilt_deg"] for i in ok]),
    }
    if checks:
        good = [c for c in checks if c["id"] is not None]
        summary["mapping_check"] = {
            "rule": "frame-0 head image, 15x15 window at the projected radio centre, fraction of red pixels (H<=8|H>=172, "
                    "S>=120, V>=70); mapping = meta task_instance_id; alternatives id-1 / id+1 for contrast",
            "n_projected_in_image": len(good), "not_in_image": [c["episode"] for c in checks if c["id"] is None],
            "id_red_fraction": stats([c["id"] for c in good]),
            "id_red_fraction_ge_0.3": sum(c["id"] >= 0.3 for c in good),
            "id_minus_1_red_fraction": stats([c["id_minus_1"] for c in checks]),
            "id_plus_1_red_fraction": stats([c["id_plus_1"] for c in checks]),
            "low_red_episodes": [c for c in good if c["id"] < 0.3]}
    (OUT.parent / "gt_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("skipped_or_error",)}, indent=1, default=float)[:5000])


if __name__ == "__main__":
    sys.exit(main())
