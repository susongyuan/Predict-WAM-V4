#!/usr/bin/env python3
"""Offline accuracy of harness/object_perception.py (Task 1 onboard detector). DEVELOPMENT EVIDENCE ONLY.

This measures offline perception accuracy on recorded frames; it is not a docking or task result.

Data (nothing from the report / test instances 301-310 is used):
  A. Simulator RGB-D replay of TRAIN demo episode 389 on its own train instance 290 (Isaac Sim 6.1 port):
     outputs/validation/task1_demo_replay_20260926/run_ep389_inst290_rgbd/ (native 720x720 head RGB + depth_linear +
     proprio cam_rel_poses every 10 steps; evaluator-side ground-truth world poses of objects and robot base).
     Split by 250-step blocks: TUNING = steps with (step // 250) % 2 == 0, HELD-OUT = the odd blocks. (A first
     500-block split was abandoned before final tuning because its tuning half contained no mid-range trash-can view;
     that v1 held-out result is recorded in the report.) Step 0 is excluded (post-reset garbage camera pose; used to
     check the plausibility guard).
     Ground truth: object centre in the base frame; an object is a POSITIVE when it stands on the floor (cans not in
     the bin, not within 0.25 m of a closed gripper), its projected centre is inside the image and its depth-verified
     mask (depth points inside the object's cylinder) covers >= 3 px and >= 15 % of its projected box. Other objects with
     a visible mask are IGNORE (detections on them are counted separately, not as false positives).
  B. Demo videos (HEVC 720x720 head RGB, NO depth -> depth-free fallback mode) with the offline labels of
     scripts/build_task1_object_gt.py / scripts/task1_perception_labels.py. Reported: the 20 VALIDATION episodes of
     data/behavior-1k/2026-challenge-demos/split_task_001.json. Train episodes 200, 201, 202, 203, 206 were used to
     check the depth-free colour gates (tuning). Label noise: the odometry-based GT has ~3-10 cm error (gt_summary).
  C. obstacle_points on the replay held-out frames, scored against the scene layout map (walls + furniture).
  D. End-to-end: harness/task1_perception_adapter.TaskPerception on observation-shaped dicts built from the replay
     frames (RGBA uint8, depth_linear, flat 21-D cam_rel_poses, 61-D proprio) feeding docking_controller.ObjectMap.

Output: outputs/validation/task1_perception_20260926/report_detector.json + det_*.png examples.
CPU only. `python3 scripts/eval_object_perception.py [--demo-stride 15] [--workers 6]`
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")               # CPU-only evaluation; keep the machine share small

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from harness import object_perception as op  # noqa: E402

RUN = ROOT / "outputs/validation/task1_demo_replay_20260926/run_ep389_inst290_rgbd"
NF = RUN / "native_frames"
OUT = ROOT / "outputs/validation/task1_perception_20260926"
DEMO = ROOT / "data/behavior-1k/2026-challenge-demos"
BINS = [(0.0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, math.inf)]
BIN_NAMES = ["0-1.5", "1.5-3", "3-5", ">5"]
LABELS = ["can_of_soda", "trash_can"]
GATE = {"can_of_soda": 0.30, "trash_can": 0.35}          # m, xy association gate (detection centre vs GT centre)
HALF = {"wkxtxh": (0.115, 0.139), "itolcg": (0.0376, 0.0673), "lugwcz": (0.0381, 0.0635)}
MODEL = {"can_of_soda_114": "itolcg", "can_of_soda_115": "itolcg", "can_of_soda_113": "lugwcz", "trash_can_116": "wkxtxh"}
DEMO_TUNING_EPISODES = [200, 201, 202, 203, 206]


def bin_of(r):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= r < hi:
            return i
    return len(BINS) - 1


def pct(v, q):
    return None if len(v) == 0 else round(float(np.percentile(v, q)), 4)


# ---- A. replay ground truth ----------------------------------------------------------------------------------------
class Replay:
    def __init__(self):
        self.cams = np.load(NF / "cam_rel_poses.npy").astype(np.float64)       # (T, 3, 7)
        self.prop = np.load(NF / "proprio.npy").astype(np.float64)             # (T, 61)
        self.steps = sorted(int(p.stem.split("_")[1]) for p in (NF / "head").glob("step_*.png"))
        self.rows, self.hdr = {}, None
        with open(RUN / "eval/object_poses/picking_up_trash_290_0.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") == "header":
                    self.hdr = r
                elif "step" in r:
                    self.rows[r["step"]] = r
        self.ents = self.hdr["entities"]

    def frame(self, s):
        bgr = cv2.imread(str(NF / f"head/step_{s:06d}.png"), cv2.IMREAD_UNCHANGED)
        rgb = np.ascontiguousarray(bgr[:, :, :3][:, :, ::-1])
        depth = np.load(NF / f"head_depth/step_{s:06d}.npy").astype(np.float32)
        return rgb, depth

    def base7(self, s):
        return np.array(self.rows[s]["robot_base"], np.float64)

    def objects(self, s, depth, cam):
        """GT objects at step s with positive / ignore flags (see module docstring)."""
        b7 = self.base7(s)
        Rb = op.quat_to_mat(b7[3:7])
        vv, uu = np.mgrid[0:720, 0:720]
        z = np.where(np.isfinite(depth) & (depth > 0.05), depth, 0.0)
        Pb = cam.backproject(uu, vv, z).reshape(-1, 3)
        Pw = Pb @ Rb.T + b7[:3]
        held_xy = [e[:2] for e in op._held_eefs(self.prop[s], op.DEFAULT)]
        out = []
        objs = []
        for key, e in self.ents.items():
            o7 = np.array(self.rows[s]["objects"][key], np.float64)
            objs.append((key, e, o7, (o7[:3] - b7[:3]) @ Rb))
        bin_c = next(c for (k, e, o7, c) in objs if e["category"] == "trash_can")
        for key, e, o7, c in objs:
            m = MODEL[e["name"]]
            Ro = op.quat_to_mat(o7[3:7])
            Po = (Pw - o7[:3]) @ Ro
            ext = np.array(e["native_bbox_object_frame"]) / 2
            inside = (np.hypot(Po[:, 0], Po[:, 1]) <= ext[0] + 0.01) & (np.abs(Po[:, 2]) <= ext[2] + 0.015)
            npx = int((inside.reshape(720, 720) & (z > 0)).sum())
            floor_z = 0.135 if m == "wkxtxh" else 0.068
            in_bin = e["category"] == "can_of_soda" and float(np.hypot(*(c[:2] - bin_c[:2]))) < 0.13
            floor = abs(o7[2] - floor_z) < 0.03 and not in_bin
            held = any(float(np.hypot(*(c[:2] - h))) < 0.25 for h in held_xy)
            r_, hh = HALF[m]
            a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            pts = np.concatenate([np.c_[c[0] + r_ * np.cos(a), c[1] + r_ * np.sin(a), np.full(16, c[2] + dz)] for dz in (-hh, hh)])
            u, v, zz = cam.project(pts)
            if (zz > 0.05).all():
                area = max((np.clip(u.max(), 0, 720) - np.clip(u.min(), 0, 720)) * (np.clip(v.max(), 0, 720) - np.clip(v.min(), 0, 720)), 1.0)
            else:
                area = 1e9
            uc, vc, zc = cam.project(c[None])
            in_img = bool(zc[0] > 0.05 and 10 <= uc[0] < 710 and 10 <= vc[0] < 710)
            pos = floor and not held and in_img and npx >= 3 and npx >= 0.15 * area
            out.append({"name": e["name"], "label": e["category"], "centre": c, "range": float(np.hypot(c[0], c[1])),
                        "npx": npx, "floor": floor, "held": held, "in_bin": in_bin, "in_img": in_img, "pos": pos,
                        "ignore": (not pos) and npx >= 3, "uv": (float(uc[0]), float(vc[0]))})
        return out


def det_point(d):
    """Detection point comparable with the GT object centre (bin: fitted centre; the rim point differs by RIM_OFFSET)."""
    return np.asarray(d.get("center_base", d["xyz_base"]), np.float64)


def match(dets, objs):
    """Greedy nearest association of detections to POSITIVE objects of the same label; returns (pairs, unmatched)."""
    cand = []
    for i, d in enumerate(dets):
        for j, o in enumerate(objs):
            if o["pos"] and o["label"] == d["label"]:
                e = float(np.linalg.norm(det_point(d)[:2] - o["centre"][:2]))
                if e <= GATE[d["label"]]:
                    cand.append((e, i, j))
    pairs, used_d, used_o = [], set(), set()
    for e, i, j in sorted(cand):
        if i not in used_d and j not in used_o:
            pairs.append((i, j))
            used_d.add(i)
            used_o.add(j)
    return pairs, [i for i in range(len(dets)) if i not in used_d]


def classify_fp(d, objs):
    p = det_point(d)[:2]
    if any(o["ignore"] and o["label"] == d["label"] and np.linalg.norm(p - o["centre"][:2]) <= 0.35 for o in objs):
        return "ignore_object"            # partially visible / held / in-bin object of the same label: not an FP
    binc = next(o for o in objs if o["label"] == "trash_can")["centre"][:2]
    if d["label"] == "can_of_soda" and np.linalg.norm(p - binc) < 0.25:
        return "bin_labelled_can"
    if any((o["held"] or o["in_bin"] or not o["floor"]) and np.linalg.norm(p - o["centre"][:2]) <= 0.35 for o in objs):
        return "held_or_in_bin_object"
    if any(o["label"] != d["label"] and np.linalg.norm(p - o["centre"][:2]) <= 0.2 for o in objs):
        return "wrong_label"
    return "background"


def new_acc():
    return {lab: {"gt": [0] * 4, "tp": [0] * 4, "det": [0] * 4, "det_tp": [0] * 4, "err_xy": [[] for _ in BINS],
                  "err_3d": [[] for _ in BINS], "vis_pos": [0] * 4, "fp": {}} for lab in LABELS}


def summarise(acc):
    out = {}
    for lab in LABELS:
        a = acc[lab]
        rows = {}
        for b, name in enumerate(BIN_NAMES):
            rows[name] = {"n_gt": a["gt"][b], "recall": None if not a["gt"][b] else round(a["tp"][b] / a["gt"][b], 3),
                          "n_det": a["det"][b], "precision": None if not a["det"][b] else round(a["det_tp"][b] / a["det"][b], 3),
                          "err_xy_median_m": pct(a["err_xy"][b], 50), "err_xy_p90_m": pct(a["err_xy"][b], 90),
                          "err_3d_median_m": pct(a["err_3d"][b], 50), "err_3d_p90_m": pct(a["err_3d"][b], 90),
                          "visible_fn_true_on_positives": None if not a["gt"][b] else round(a["vis_pos"][b] / a["gt"][b], 3)}
        allxy = sum(a["err_xy"], [])
        all3 = sum(a["err_3d"], [])
        out[lab] = {"by_distance": rows, "recall_all": None if not sum(a["gt"]) else round(sum(a["tp"]) / sum(a["gt"]), 3),
                    "precision_all": None if not sum(a["det"]) else round(sum(a["det_tp"]) / sum(a["det"]), 3),
                    "err_xy_median_m": pct(allxy, 50), "err_xy_p90_m": pct(allxy, 90),
                    "err_3d_median_m": pct(all3, 50), "err_3d_p90_m": pct(all3, 90), "false_positives": a["fp"]}
    return out


def eval_replay(rp, steps, depth_mode=True, examples=None):
    acc = new_acc()
    vis = {"occluded_or_out_of_view_in_range": 0, "visible_fn_true_on_those": 0}
    lat = []
    fp_list = []
    for s in steps:
        rgb, depth = rp.frame(s)
        R, t = op.camera_from_pose(rp.cams[s, 2])
        cam = op.Camera(R, t, 720, 720)
        objs = rp.objects(s, depth, cam)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        t0 = time.perf_counter()
        dets, vfn = op.detect_from_observation(rgba, depth if depth_mode else None, rp.cams[s].reshape(-1), rp.prop[s])
        lat.append((time.perf_counter() - t0) * 1e3)
        pairs, unmatched = match(dets, objs)
        matched_o = {j for _, j in pairs}
        for j, o in enumerate(objs):
            if o["pos"]:
                b = bin_of(o["range"])
                acc[o["label"]]["gt"][b] += 1
                acc[o["label"]]["vis_pos"][b] += int(bool(vfn(o["centre"])))
                acc[o["label"]]["tp"][b] += int(j in matched_o)
            elif o["floor"] and o["npx"] == 0 and op.DEFAULT.vis_range[0] <= o["range"] <= op.DEFAULT.vis_range[1]:
                vis["occluded_or_out_of_view_in_range"] += 1
                vis["visible_fn_true_on_those"] += int(bool(vfn(o["centre"])))
        for i, j in pairs:
            d, o = dets[i], objs[j]
            b = bin_of(o["range"])
            p = det_point(d)
            acc[d["label"]]["err_xy"][b].append(float(np.linalg.norm(p[:2] - o["centre"][:2])))
            acc[d["label"]]["err_3d"][b].append(float(np.linalg.norm(p - o["centre"])))
        for i, d in enumerate(dets):
            b = bin_of(float(np.hypot(*det_point(d)[:2])))
            if i in unmatched:
                cat = classify_fp(d, objs)
                if cat == "ignore_object":
                    continue
                acc[d["label"]]["fp"][cat] = acc[d["label"]]["fp"].get(cat, 0) + 1
                fp_list.append({"step": s, "label": d["label"], "xy": np.round(det_point(d)[:2], 2).tolist(), "category": cat})
            else:
                acc[d["label"]]["det_tp"][b] += 1
            acc[d["label"]]["det"][b] += 1
        if examples is not None and s in examples:
            draw_example(rgb, cam, dets, objs, OUT / f"det_replay_step{s:05d}{'' if depth_mode else '_nodepth'}.png",
                         f"replay inst290 step {s} ({'RGB-D' if depth_mode else 'RGB only'})")
    return {"frames": len(steps), "metrics": summarise(acc),
            "visible_fn_false_visible": {**vis, "rate": None if not vis["occluded_or_out_of_view_in_range"] else
                                         round(vis["visible_fn_true_on_those"] / vis["occluded_or_out_of_view_in_range"], 3)},
            "latency_ms": {"median": pct(lat, 50), "p95": pct(lat, 95), "max": pct(lat, 100)},
            "false_positive_list_head": fp_list[:40]}


def draw_example(rgb, cam, dets, objs, path, title):
    img = rgb[:, :, ::-1].copy()
    for o in objs:
        u, v = o["uv"]
        if 0 <= u < 720 and 0 <= v < 720:
            col = (0, 255, 0) if o["pos"] else (0, 160, 255)
            cv2.circle(img, (int(u), int(v)), 10, col, 2)
    for d in dets:
        for key, col in (("center_base", (255, 0, 255)), ("xyz_base", (0, 0, 255))):
            if key not in d:
                continue
            u, v, z = cam.project(np.asarray(d[key], float)[None])
            if z[0] > 0.05:
                cv2.drawMarker(img, (int(u[0]), int(v[0])), col, cv2.MARKER_CROSS, 18, 2)
        u, v, z = cam.project(det_point(d)[None])
        if z[0] > 0.05:
            txt = f"{'can' if d['label'] == 'can_of_soda' else 'bin'} {np.hypot(*det_point(d)[:2]):.1f}m"
            cv2.putText(img, txt, (int(u[0]) + 8, int(v[0]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "green=GT positive, orange=GT ignore, red+=detection (bin: magenta=centre, red=rim grasp pt)",
                (8, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)


# ---- B. demo videos (depth-free, 2-D) -------------------------------------------------------------------------------
def demo_frames(ep, stride):
    import av
    import pyarrow.parquet as pq
    cam_key = "observation.rgb.zed_link_camera_0"
    row = None
    for f in sorted((DEMO / "meta/episodes/chunk-001").glob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            if r["episode_index"] == ep:
                row = r
    p = f"videos/{cam_key}"
    path = DEMO / f"videos/{cam_key}/chunk-{row[p + '/chunk_index']:03d}/file-{row[p + '/file_index']:03d}.mp4"
    start, length, fps = float(row[p + "/from_timestamp"]), int(row["length"]), 30.0
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        c.seek(int(max(0.0, start - 1.0) / st.time_base), stream=st, backward=True)
        k = 0
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            ts = float(fr.pts * fr.time_base)
            if ts < start + k / fps - 0.5 / fps:
                continue
            if k % stride == 0:
                yield k, fr.to_ndarray(format="rgb24")
            k += 1
            if k >= length:
                break


def demo_episode(args):
    ep, stride, example_frames = args
    cv2.setNumThreads(1)
    from task1_perception_labels import EpisodeLabels, SceneMaps
    L = EpisodeLabels(ep, SceneMaps())
    state = np.load(ROOT / f"data/lowdim_cache_task001/episode_{ep}_state.npy").astype(np.float64)
    recs, fps_, lat, ex = [], [], [], []
    for k, img in demo_frames(ep, stride):
        objs = L.objects(k, 720, img)
        t0 = time.perf_counter()
        dets, _ = op.detect_from_observation(img, None, L.cam[k], state[k])
        lat.append((time.perf_counter() - t0) * 1e3)
        cans = [d for d in dets if d["label"] == "can_of_soda"]
        used = set()
        for o in objs:
            if o["cls"] == 0 or not o["visible"] or o["uv"] is None:
                continue
            b = o["box"]
            tol = max(10.0, 0.6 * math.hypot(b[2] - b[0], b[3] - b[1]))
            best = None
            for j, d in enumerate(cans):
                e = math.hypot(d["uv"][0] - o["uv"][0], d["uv"][1] - o["uv"][1])
                if j not in used and e < tol and (best is None or e < best[1]):
                    best = (j, e)
            if best:
                used.add(best[0])
            recs.append({"dist": o["dist"], "hit": best is not None, "colour_verified": bool(o["colour_verified"]),
                         "px_err": None if not best else best[1],
                         "xy_err": None if not best else float(np.linalg.norm(cans[best[0]]["xyz_base"][:2] - o["xyz"][:2]))})
        held = L.held_objects(k)
        binpos = [o["xyz"] for o in objs if o["cls"] == 0] + [p for (j, st, p) in held if L.models[j] == "wkxtxh"]
        for j, d in enumerate(cans):
            if j in used:
                continue
            u, v = d["uv"]
            if any(o["box"] is not None and o["cls"] > 0 and o["box"][0] - 15 <= u <= o["box"][2] + 15
                   and o["box"][1] - 15 <= v <= o["box"][3] + 15 for o in objs):
                continue                                  # on an in-frustum can that is not a positive label: ignore
            p = d["xyz_base"][:2]
            if any(np.linalg.norm(p - bp[:2]) < 0.3 for bp in binpos):
                cat = "bin_labelled_can"
            elif held:
                # carried bin / held can: plane-projected position is meaningless; use the image instead
                cat = "held_or_in_bin_object" if any(_near_held_px(L, k, pp, u, v) for (_, _, pp) in held) else "background"
            else:
                cat = "background"
            fps_.append({"k": k, "cat": cat, "dist": float(np.hypot(*p))})
        if k in example_frames:
            ex.append((k, img, dets, objs))
    return {"ep": ep, "recs": recs, "fps": fps_, "lat": lat, "frames": len(lat), "examples": [(k, img, dets) for k, img, dets, _ in ex]}


def _near_held_px(L, k, p_base, u, v, radius=90):
    from task1_perception_labels import intrinsics, project_points
    uv, z = project_points(np.asarray(p_base, float)[None], L.cam[k], intrinsics(720))
    return bool(z[0] > 0.05 and math.hypot(uv[0, 0] - u, uv[0, 1] - v) < radius)


def eval_demos(episodes, stride, workers, example_eps=()):
    jobs = [(ep, stride, {600, 1800} if ep in example_eps else set()) for ep in episodes]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(demo_episode, jobs)
    rows = {}
    for name, (lo, hi) in zip(BIN_NAMES, BINS):
        r = [x for res_ in res for x in res_["recs"] if lo <= x["dist"] < hi]
        rv = [x for x in r if x["colour_verified"]]
        rows[name] = {"n_gt_visible": len(r), "recall": None if not r else round(float(np.mean([x["hit"] for x in r])), 3),
                      "n_gt_colour_verified": len(rv),
                      "recall_colour_verified": None if not rv else round(float(np.mean([x["hit"] for x in rv])), 3),
                      "px_err_median": pct([x["px_err"] for x in r if x["hit"]], 50),
                      "xy_err_median_m_vs_noisy_gt": pct([x["xy_err"] for x in r if x["hit"]], 50),
                      "xy_err_p90_m_vs_noisy_gt": pct([x["xy_err"] for x in r if x["hit"]], 90)}
    fps_ = [f for r in res for f in r["fps"]]
    frames = sum(r["frames"] for r in res)
    n_tp = sum(x["hit"] for r in res for x in r["recs"])
    cats = {}
    for f in fps_:
        cats[f["cat"]] = cats.get(f["cat"], 0) + 1
    lat = [x for r in res for x in r["lat"]]
    for r in res:
        for k, img, dets in r["examples"]:
            out = img[:, :, ::-1].copy()
            for d in dets:
                u, v = d.get("uv", (None, None))
                if u is not None:
                    cv2.drawMarker(out, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(out, f"demo val episode {r['ep']} frame {k} (RGB only, depth-free mode)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(OUT / f"det_demo_ep{r['ep']}_f{k:05d}.png"), out)
    return {"episodes": episodes, "frame_stride": stride, "frames": frames, "mode": "depth-free (videos have no depth)",
            "can_of_soda": {"by_distance": rows, "matched_detections": int(n_tp), "false_positives": len(fps_),
                            "false_positives_per_frame": round(len(fps_) / max(frames, 1), 3),
                            "precision_approx": round(n_tp / max(n_tp + len(fps_), 1), 3), "fp_categories": cats},
            "trash_can": "not evaluated: the depth-free bin detector is disabled by default (bin_without_depth=False) "
                         "because it fires on wood furniture/floor; with depth the geometric detector is used",
            "latency_ms": {"median": pct(lat, 50), "p95": pct(lat, 95)}}


# ---- C. obstacle points ---------------------------------------------------------------------------------------------
def eval_obstacles(rp, steps):
    from task1_perception_labels import SceneMaps, MAP_RES
    maps = SceneMaps()
    occ = (maps.walls | maps.furniture).astype(np.uint8)
    dist_cells = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 5)
    res = {"with_self_filter": [], "no_proprio_box_only": [], "no_filter": []}
    lat, counts = [], []
    for s in steps:
        _, depth = rp.frame(s)
        b7 = rp.base7(s)
        Rb = op.quat_to_mat(b7[3:7])
        objs_w = [np.array(rp.rows[s]["objects"][k][:2]) for k in rp.ents]
        cam21 = rp.cams[s].reshape(-1)
        t0 = time.perf_counter()
        P = op.obstacle_points(depth, cam21, rp.prop[s])
        lat.append((time.perf_counter() - t0) * 1e3)
        counts.append(len(P))
        variants = {"with_self_filter": P, "no_proprio_box_only": op.obstacle_points(depth, cam21, None)}
        # no filter at all: same pipeline with the self mask disabled
        orig = op._self_mask
        op._self_mask = lambda Q, st, ct: np.zeros(len(Q), bool)
        variants["no_filter"] = op.obstacle_points(depth, cam21, rp.prop[s])
        op._self_mask = orig
        for name, Q in variants.items():
            if not len(Q):
                continue
            W = np.c_[Q, np.zeros(len(Q))] @ Rb.T + b7[:3]
            r, c = maps.rc(W[:, 0], W[:, 1])
            ok = (r >= 0) & (r < maps.size) & (c >= 0) & (c < maps.size)
            d = np.full(len(Q), 99.0)
            d[ok] = dist_cells[r[ok], c[ok]] * MAP_RES
            near_obj = np.min([np.linalg.norm(W[:, :2] - o, axis=1) for o in objs_w], 0) < 0.25
            consistent = (d <= 0.10) | near_obj
            close = np.hypot(Q[:, 0], Q[:, 1]) < 1.0
            res[name].append((consistent.mean(), int((close & ~consistent).sum()), int(close.sum())))
    out = {"frames": len(steps), "points_per_frame_median": pct(counts, 50), "latency_ms": {"median": pct(lat, 50), "p95": pct(lat, 95)},
           "note": "consistent = within 0.10 m of a wall/furniture cell of the scene layout map (floor_trav) or within "
                   "0.25 m of a task object; inconsistent points within 1 m of the robot are mostly its own arms / "
                   "grippers / carried bin (the map has no small clutter, so this is an upper bound on self leakage)"}
    for name, v in res.items():
        if v:
            out[name] = {"map_consistent_fraction_median": round(float(np.median([x[0] for x in v])), 3),
                         "inconsistent_points_within_1m_per_frame_mean": round(float(np.mean([x[1] for x in v])), 1),
                         "points_within_1m_per_frame_mean": round(float(np.mean([x[2] for x in v])), 1)}
    return out


# ---- D. adapter end-to-end ------------------------------------------------------------------------------------------
def eval_adapter(rp):
    from harness.task1_perception_adapter import TaskPerception, HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX
    from harness.docking_controller import ObjectMap, Odometry
    tp = TaskPerception()
    omap, odom = ObjectMap(), Odometry()
    captured = set(rp.steps)
    log = []
    step0 = None
    checkpoints = {90, 900, 1490, 2340, 2830, 3120}
    for s in range(0, max(rp.steps) + 1):
        if s > 0:
            odom.update(rp.prop[s - 1, 0:3])     # measured local base velocity (proprio base_qvel) driving s-1 -> s
        if s not in captured:
            continue
        rgb, depth = rp.frame(s)
        obs = {"robot_r1::robot_r1" + HEAD_RGB_SUFFIX: np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]),
               "robot_r1::robot_r1" + HEAD_DEPTH_SUFFIX: depth,
               "robot_r1" + CAM_POSE_SUFFIX: rp.cams[s].reshape(-1).astype(np.float32)}
        dets, vfn = tp.detect(obs, rp.prop[s].astype(np.float32))
        obst = tp.obstacles(obs, rp.prop[s].astype(np.float32))
        if s == 0:
            step0 = {"n_detections": len(dets), "visible_fn_is_none": vfn is None, "n_obstacle_points": int(len(obst))}
        omap.update(dets, odom, s, visible_fn=vfn)
        if s in checkpoints:
            R, t = op.camera_from_pose(rp.cams[s, 2])
            objs = rp.objects(s, depth, op.Camera(R, t, 720, 720))
            rows = []
            for o in objs:
                if not o["floor"]:
                    continue
                tr = [x for x in omap.confirmed(o["label"])]
                best = min(tr, key=lambda x: np.linalg.norm(odom.to_base(x.pos) - o["centre"][:2]), default=None)
                gt_pt = o["centre"][:2] - (op.RIM_OFFSET if o["label"] == "trash_can" else 0.0)
                rows.append({"object": o["name"], "gt_range_m": round(o["range"], 2),
                             "nearest_confirmed_track_err_m": None if best is None else
                             round(float(np.linalg.norm(odom.to_base(best.pos) - gt_pt)), 3)})
            log.append({"step": s, "n_confirmed": {lab: len(omap.confirmed(lab)) for lab in LABELS},
                        "n_tracks": len(omap.tracks), "n_obstacle_points": int(len(obst)), "floor_objects": rows})
    return {"note": "observation dicts with the adapter's suffix keys (RGBA uint8 720x720, depth_linear float32, flat "
                    "21-D cam_rel_poses, 61-D proprio); ObjectMap in an odometry frame integrated from proprio base "
                    "velocity (so track errors include odometry drift); trash_can tracks hold the rim grasp point, "
                    "compared with GT centre - RIM_OFFSET. Detections every 10 steps (capture rate).",
            "step0_guard": step0, "checkpoints": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-stride", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-demos", action="store_true")
    args = ap.parse_args()
    cv2.setNumThreads(1)                     # latency measured on one core
    rp = Replay()
    tune = [s for s in rp.steps if s > 0 and (s // 250) % 2 == 0]
    held = [s for s in rp.steps if s > 0 and (s // 250) % 2 == 1]
    report = {
        "statement": "Offline perception accuracy of harness/object_perception.py on recorded frames. Not a docking, "
                     "skill or task result; no simulator was run for this evaluation.",
        "detector_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(op.DEFAULT).items()},
        "point_convention": {"can_of_soda": "xyz_base = can centre (z fixed 0.066 m)",
                             "trash_can": "xyz_base = right-hand rim grasp point = bin centre - RIM_OFFSET "
                                          f"{op.RIM_OFFSET.tolist()} (base frame), z = rim 0.279 m; the fitted centre is "
                                          "'center_base'. Errors below compare center_base with the GT centre (the rim "
                                          "point error is identical by construction; the rim offset itself is the train-"
                                          "demo median with its own spread)."},
        "replay": {"run": str(RUN.relative_to(ROOT)), "split": "250-step blocks: tuning (step//250)%2==0, held-out odd",
                   "tuning_steps": [tune[0], tune[-1], len(tune)], "heldout_steps": [held[0], held[-1], len(held)],
                   "heldout_blocks": sorted({s // 250 * 250 for s in held}),
                   "v1_split_note": "First split (500-step blocks) was dropped before the geometric bin detector was "
                                    "built: its tuning half had no mid-range trash-can view; on its held-out half the "
                                    "v1 colour-only detector had can recall 1.00 (0-5 m) but trash-can recall 0.47 "
                                    "(0-1.5 m) and 0.00 (1.5-5 m)."},
    }
    ex = {800, 1900, 2400, 3300}
    print("replay held-out (RGB-D)...", flush=True)
    report["replay"]["heldout_rgbd"] = eval_replay(rp, held, True, examples=ex)
    print("replay tuning (RGB-D)...", flush=True)
    report["replay"]["tuning_rgbd"] = eval_replay(rp, tune, True)
    print("replay held-out (RGB only)...", flush=True)
    report["replay"]["heldout_rgb_only"] = eval_replay(rp, held, False, examples={1900})
    print("obstacles...", flush=True)
    report["obstacle_points"] = eval_obstacles(rp, held)
    print("adapter end-to-end...", flush=True)
    report["adapter_end_to_end"] = eval_adapter(rp)
    if not args.skip_demos:
        split = json.loads((DEMO / "split_task_001.json").read_text())
        val = [int(e) for e in split["validation_episode_indices"]]
        print(f"demo validation episodes {val} ...", flush=True)
        report["demo_videos_validation"] = eval_demos(val, args.demo_stride, args.workers, example_eps=val[:2])
        report["demo_videos_validation"]["tuning_episodes_used_for_depth_free_gates"] = DEMO_TUNING_EPISODES
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report_detector.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({"heldout_rgbd": {lab: report["replay"]["heldout_rgbd"]["metrics"][lab]["by_distance"] for lab in LABELS}},
                     indent=1, default=float))
    print("wrote", OUT / "report_detector.json")


if __name__ == "__main__":
    main()
