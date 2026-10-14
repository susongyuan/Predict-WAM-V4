#!/usr/bin/env python3
"""CPU check of the evaluator-only observer camera (recording_utils.ObserverFraming); no simulator, no GPU.

Synthetic axis-aligned box world (walls, ceiling, door jamb, car) with an exact slab ray test stands in for the PhysX
scene query. Checks: the legacy fixed offset is occluded where the 2026-09-23 smoke video was (robot in a doorway /
against the garage wall), auto framing finds an unobstructed, fully framed pose, the car rear is kept in view at the
trunk, motion is smooth (no jitter) apart from logged cuts, and results are deterministic.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORDING_UTILS = ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/eval/utils/recording_utils.py"
spec = importlib.util.spec_from_file_location("recording_utils", RECORDING_UTILS)
ru = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ru)

OFFSET, TARGET = [-2.5, -2.5, 2.1], [0.6, 0.0, 0.9]  # evaluator defaults
W, H, APERTURE, FOCAL = 1280, 720, 28.0, 17.0

# Garage x in [0, 6], y in [0, 5], ceiling 2.6 m; wall at x=0 has a 1.0 m doorway (y in [2, 3]) into a 1.2 m corridor
# running to x=-6. Car occupies x in [3.4, 5.9], y in [1.2, 3.1] (rear bumper faces -x).
T = 0.15
BOXES = {
    "floor": ([-6.5, -0.5, -0.1], [6.5, 5.5, 0.0]),
    "ceiling": ([-6.5, -0.5, 2.6], [6.5, 5.5, 2.8]),
    "garage_wall_x0_low": ([-T, -0.5, 0.0], [0.0, 2.0, 2.6]),
    "garage_wall_x0_high": ([-T, 3.0, 0.0], [0.0, 5.5, 2.6]),
    "garage_wall_x6": ([6.0, -0.5, 0.0], [6.0 + T, 5.5, 2.6]),
    "garage_wall_y0": ([0.0, -T, 0.0], [6.0, 0.0, 2.6]),
    "garage_wall_y5": ([0.0, 5.0, 0.0], [6.0, 5.0 + T, 2.6]),
    "corridor_wall_south": ([-6.5, 1.9 - T, 0.0], [0.0, 1.9, 2.6]),
    "corridor_wall_north": ([-6.5, 3.1, 0.0], [0.0, 3.1 + T, 2.6]),
    "car": ([3.4, 1.2, 0.0], [5.9, 3.1, 1.4]),
}
ROBOT = "robot"


def ray_box(start, end, lo, hi):
    d = end - start
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if start[k] < lo[k] or start[k] > hi[k]:
                return False
            continue
        a, b = (lo[k] - start[k]) / d[k], (hi[k] - start[k]) / d[k]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return False
    return True


def make_blocked(boxes, counter):
    def blocked(start, end, ignore):
        counter[0] += 1
        start, end = np.asarray(start, float), np.asarray(end, float)
        return any(name not in ignore and ray_box(start, end, np.asarray(lo), np.asarray(hi))
                   for name, (lo, hi) in boxes.items())
    return blocked


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def subject(base, yaw, arm_reach=0.35, car_poi=True):
    """R1Pro-like point layout: base 0.7 x 0.7 m, torso/head up to 1.6 m, two end effectors ahead of the torso."""
    rot = yaw_matrix(yaw)
    base = np.asarray(base, float)
    local_links = [[0, 0, 0.1], [0, 0, 0.5], [0.05, 0, 1.0], [0.05, 0, 1.45], [0.1, 0, 1.6],
                   [0.2, 0.25, 1.3], [0.2, -0.25, 1.3], [arm_reach, 0.25, 1.0], [arm_reach, -0.25, 1.0]]
    links = np.array([base + rot @ np.asarray(p, float) for p in local_links])
    top = links[np.argmax(links[:, 2])]
    lower = base + [0, 0, 0.25]
    footprint = [base + rot @ np.array([0.35 * sx, 0.35 * sy, 0.05]) for sx in (-1, 1) for sy in (-1, 1)]
    optional = [(links[-2], (ROBOT,)), (links[-1], (ROBOT,))]
    lo, hi = map(np.asarray, BOXES["car"])
    poi = np.clip(base + [0, 0, 1.0], lo, hi)
    if car_poi and np.linalg.norm((poi - base)[:2]) <= 1.5:
        optional.append((poi, (ROBOT, "car")))
    return {"required": np.stack([lower, 0.5 * (lower + top), top]),
            "frustum": np.vstack([links, top + [0, 0, 0.15], np.stack(footprint)]),
            "robot_ignore": (ROBOT,), "optional": optional}


def legacy_visible(base, yaw, blocked, subj):
    pos, rot = ru.viewer_pose(np.asarray(base, float), yaw_matrix(yaw), OFFSET, TARGET)
    th = 0.5 * APERTURE / FOCAL
    clear = all(not blocked(pos, p, (ROBOT,)) for p in subj["required"])
    framed = bool(ru.in_frustum(subj["frustum"], pos, rot, th, th * H / W).all())
    return {"required_rays_clear": clear, "framed": framed, "position": np.round(pos, 3).tolist()}


def run(trajectory, boxes, arm_reach=0.35):
    counter = [0]
    blocked = make_blocked(boxes, counter)
    framing = ru.ObserverFraming(OFFSET, TARGET, W, H, APERTURE, FOCAL, blocked=blocked)
    records, positions = [], []
    for base, yaw in trajectory:
        pos, rot, rec = framing.update(np.asarray(base, float), yaw_matrix(yaw), subject(base, yaw, arm_reach))
        # Independent re-check of the returned pose (no reuse of the framing object's own verdict).
        subj = subject(base, yaw, arm_reach)
        th = 0.5 * APERTURE / FOCAL
        rec["independent_clear"] = all(not blocked(pos, p, (ROBOT,)) for p in subj["required"])
        rec["independent_framed"] = bool(ru.in_frustum(subj["frustum"], pos, rot, th, th * H / W).all())
        records.append(rec)
        positions.append(pos)
    positions = np.array(positions)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cut_frames = {r["frame"] for r in records if r["cut"]}
    smooth_steps = [s for i, s in enumerate(steps, start=1) if i not in cut_frames]
    return records, {"frames": len(records), "stats": framing.stats, "rays": counter[0],
                     "all_valid": all(r["valid"] for r in records),
                     "all_independent_ok": all(r["independent_clear"] and r["independent_framed"] for r in records),
                     "cuts_after_first": len(cut_frames - {0}),
                     "max_step_without_cut_m": round(float(max(smooth_steps, default=0.0)), 4),
                     "mean_step_without_cut_m": round(float(np.mean(smooth_steps)) if smooth_steps else 0.0, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/validation/viewer_framing_20260924/cpu_check.json"))
    args = ap.parse_args()
    blocked = make_blocked(BOXES, [0])
    results, failures = {}, []

    # Robot speeds follow the action bound (<= 0.75 m/s, 1 rad/s) at 30 Hz.
    v, w = 0.75 / 30, 1.0 / 30
    scenarios = {
        # frame 0 analogue: robot in the corridor facing the garage; legacy camera lands behind the corridor wall.
        "corridor_start": [((-3.0, 2.5, 0.0), 0.0)] * 5,
        # frame 301 analogue: robot docked at the car rear with its back to the south wall (legacy camera outside).
        "trunk_dock_near_wall": [((2.7, 2.15, 0.0), 0.0)] * 5,
        # 4.5 m drive through the doorway into the garage, then a 90 deg turn and a short drive to the car rear.
        "corridor_to_trunk": ([((-3.0 + v * i, 2.5, 0.0), 0.0) for i in range(int(4.5 / v))]
                              + [((1.5, 2.5, 0.0), -w * i) for i in range(int((np.pi / 2) / w))]
                              + [((1.5, 2.5 - v * i, 0.0), -np.pi / 2) for i in range(int(0.3 / v))]
                              + [((1.5, 2.2, 0.0), -np.pi / 2 + w * i) for i in range(int((np.pi / 2) / w))]
                              + [((1.5 + v * i, 2.2, 0.0), 0.0) for i in range(int(1.2 / v))]
                              + [((2.7, 2.2, 0.0), 0.0)] * 150),
        # in-place 180 deg spin in the garage (camera must orbit smoothly, not jitter).
        "spin_in_garage": [((2.0, 3.8, 0.0), w * i) for i in range(int(np.pi / w))],
    }
    for name, trajectory in scenarios.items():
        records, summary = run(trajectory, BOXES, arm_reach=0.6 if "trunk" in name else 0.35)
        base, yaw = trajectory[0]
        summary["legacy_first_frame"] = legacy_visible(base, yaw, blocked, subject(base, yaw))
        base, yaw = trajectory[-1]
        summary["legacy_last_frame"] = legacy_visible(base, yaw, blocked, subject(base, yaw))
        summary["first_record"] = records[0]
        summary["last_record"] = records[-1]
        summary["car_poi_frames"] = sum(1 for r in records if len(subject(*trajectory[r["frame"]], 0.6)["optional"]) > 2)
        summary["car_poi_frames_all_optional_visible"] = sum(
            1 for r in records if len(subject(*trajectory[r["frame"]], 0.6)["optional"]) > 2 and r["optional_missing"] == 0)
        records_again, summary_again = run(trajectory, BOXES, arm_reach=0.6 if "trunk" in name else 0.35)
        summary["deterministic"] = records == records_again
        results[name] = summary
        if not (summary["all_valid"] and summary["all_independent_ok"] and summary["deterministic"]):
            failures.append(name)

    # Legacy offset must actually reproduce the reported failure mode in the two analogues (test is meaningful).
    for name, key in (("corridor_start", "legacy_first_frame"), ("trunk_dock_near_wall", "legacy_first_frame")):
        legacy = results[name][key]
        if legacy["required_rays_clear"] and legacy["framed"]:
            failures.append(f"{name}: legacy camera unexpectedly unobstructed (scenario not representative)")
    if results["trunk_dock_near_wall"]["last_record"]["optional_missing"] != 0:
        failures.append("trunk_dock_near_wall: car rear or an end effector not visible")
    if results["spin_in_garage"]["max_step_without_cut_m"] > 0.25:
        failures.append("spin_in_garage: camera moves > 0.25 m per frame without a cut")

    # Degenerate case: robot boxed in a 1 m closet -> no valid pose exists; must not crash, must be logged as fallback.
    closet = dict(BOXES, closet_w=([-0.8, 3.6, 0], [-0.7, 4.8, 2.6]), closet_e=([0.3, 3.6, 0], [0.4, 4.8, 2.6]),
                  closet_s=([-0.8, 3.6, 0], [0.4, 3.7, 2.6]), closet_n=([-0.8, 4.7, 0], [0.4, 4.8, 2.6]))
    _, closet_summary = run([((-0.2, 4.2, 0.0), 0.0)] * 3, closet)
    results["closet_no_valid_pose"] = {k: closet_summary[k] for k in ("frames", "stats", "all_valid")}
    if closet_summary["stats"]["frames_fallback_best_effort"] != 3:
        failures.append("closet_no_valid_pose: fallback not reported")

    report = {"module": str(RECORDING_UTILS.relative_to(ROOT)), "offset": OFFSET, "target": TARGET,
              "resolution": [W, H], "n_candidates": len(ru.ObserverFraming(OFFSET, TARGET, W, H, APERTURE, FOCAL,
                                                                          blocked=blocked).candidates),
              "scenarios": results, "failures": failures, "passed": not failures,
              "scope": "synthetic box world + exact slab ray test; checks selection logic only, not Isaac rendering"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for name, s in results.items():
        if name == "closet_no_valid_pose":
            print(f"{name}: {s}")
            continue
        print(f"{name}: frames={s['frames']} valid={s['all_valid']} independent={s['all_independent_ok']} "
              f"det={s['deterministic']} selections={s['stats']['selections']} glides={s['stats']['glides']} cuts={s['cuts_after_first']} "
              f"max_step={s['max_step_without_cut_m']} legacy0={s['legacy_first_frame']['required_rays_clear']}/"
              f"{s['legacy_first_frame']['framed']} first_cand={s['first_record']['candidate']} "
              f"last_cand={s['last_record']['candidate']} last_optional_missing={s['last_record']['optional_missing']} rays={s['rays']}")
    print("PASS" if not failures else f"FAIL {failures}")
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
