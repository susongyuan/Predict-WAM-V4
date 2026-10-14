#!/usr/bin/env python3
"""Task90 TRAIN-only docking statistics and optional raw-RGB calibration samples.

Reuses the established lowdim grasp/odometry recipe and docking JSON schema.
No simulator, object pose truth, validation episode, fitted color rule, or policy
weights are used. EEF-at-closure is a grasp reference, NOT an object centroid.
The resulting docking file does not by itself calibrate a fruit detector.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.setdefault("P2_TASK_ID", "90") != "90":
    raise ValueError("This entrypoint is Task90 only")
from harness import task_profile as TP
from scripts.build_task0_docking_calibration import integrate, q, GRIP_ACTION, EEF, FING, TRUNK, LARM, RARM

LABELS = {"trash_can", "half_banana", "half_pomegranate"}
DEFAULT_SAMPLES = ROOT / "outputs/validation/task090_preparation_20260930/data_runtime/train_observations"


def write_new(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"Refusing to replace different calibration evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build():
    split = json.loads(TP.SPLIT.read_text())
    segments = json.loads(TP.SEGMENTS.read_text())
    index = json.loads((TP.LOWDIM / "index.json").read_text())
    train = split["train_episode_indices"]
    assert set(train).isdisjoint(split["validation_episode_indices"])
    assert index["split_sha256"] == hashlib.sha256(TP.SPLIT.read_bytes()).hexdigest()
    rows, skipped = [], []
    post = defaultdict(lambda: {"trunk": [], "larm": [], "rarm": []})
    fingers = defaultdict(list)

    def pose(key, state, frame):
        for g, sl in (("trunk", TRUNK), ("larm", LARM), ("rarm", RARM)):
            post[key][g].append(state[frame, sl])

    for ep in train:
        assert segments["episodes"][str(ep)]["split"] == index["episodes"][str(ep)]["split"] == "train"
        st = np.load(TP.LOWDIM / f"episode_{ep}_state.npy", allow_pickle=False)
        act = np.load(TP.LOWDIM / f"episode_{ep}_action.npy", allow_pickle=False)
        assert st.shape == (len(act), 61) and act.shape[1] == 23
        segs = segments["episodes"][str(ep)]["segments"]
        for i, pick in enumerate(segs):
            if pick["skill"] != "pick up from":
                continue
            label = pick["objects"][0][0]
            assert label in LABELS
            start, end = pick["start"], min(pick["end"] - 1, len(act) - 1)
            candidates = []
            for hand, gi in GRIP_ACTION.items():
                ga = act[:, gi]
                transitions = [t for t in range(max(start, 1), end + 1) if ga[t - 1] > 0 and ga[t] <= 0]
                if transitions and np.all(ga[transitions[-1]:end + 1] <= 0):
                    candidates.append((hand, transitions[-1]))
            if len(candidates) != 1:
                skipped.append({"ep": ep, "seg": i, "cat": label, "reason": "ambiguous_or_missing_close", "candidates": candidates})
                continue
            hand, frame = candidates[0]
            x, y, th = integrate(st[start + 1:frame + 1, :3].astype(np.float64))
            eef = st[frame, EEF[hand]].astype(np.float64)
            c, s = math.cos(th[-1]), math.sin(th[-1])
            rows.append({"ep": ep, "seg": i, "cat": label, "hand": hand,
                         "grasp_frame": frame, "t_from_start": frame - start,
                         "rel_x": float(x[-1] + c * eef[0] - s * eef[1]),
                         "rel_y": float(y[-1] + s * eef[0] + c * eef[1]),
                         "eef_z": float(eef[2]), "base_shift": [float(x[-1]), float(y[-1]), float(th[-1])],
                         "trunk_start": st[start, TRUNK].tolist(), "trunk_grasp": st[frame, TRUNK].tolist()})
    # Match controller's majority-hand choice; do not mix left/right pick-start poses.
    hand_counts = {label: Counter(r["hand"] for r in rows if r["cat"] == label) for label in sorted(LABELS)}
    if any(not count or len(count) > 1 and count.most_common()[0][1] == count.most_common()[1][1] for count in hand_counts.values()):
        raise ValueError("Missing category or ambiguous majority hand; calibration not ready")
    majority = {label: count.most_common(1)[0][0] for label, count in hand_counts.items()}
    for r in rows:
        if r["hand"] != majority[r["cat"]]:
            continue
        ep, i, label = r["ep"], r["seg"], r["cat"]
        st = np.load(TP.LOWDIM / f"episode_{ep}_state.npy", allow_pickle=False)
        segs = segments["episodes"][str(ep)]["segments"]
        pick = segs[i]
        pose(f"pick up from_start|{label}", st, pick["start"])
        pose(f"pick up from_end|{label}", st, pick["end"] - 1)
        # Empty-hands navigation before the very first pick only; not fruit-carrying navigation.
        if i == 1 and segs[0]["skill"] == "move to":
            mid = (segs[0]["start"] + segs[0]["end"] - 1) // 2
            pose("nav_mid|before_bin", st, mid)
            for hand in GRIP_ACTION:
                fingers[f"{hand}_open_nav"].append(float(st[mid, FING[hand]].sum()))
        fingers[f"{r['hand']}_holding"].append(float(st[pick["end"] - 1, FING[r["hand"]]].sum()))
    stats = {"postures": {k: {g: q(v) for g, v in groups.items()} for k, groups in post.items()},
             "gripper_finger_sum_m": {k: q(v) for k, v in fingers.items()},
             "source": "Task90 TRAIN lowdim only; majority-hand pick poses; EEF closure reference, not object center",
             "hands_by_category": {k: dict(v) for k, v in hand_counts.items()}, "skipped": skipped}
    source_paths = [TP.SPLIT, TP.SEGMENTS, TP.LOWDIM / "index.json", Path(__file__),
                    ROOT / "scripts/build_task0_docking_calibration.py"]
    manifest = {"task_id": 90, "scene": "Rs_int", "split": "train", "train_episodes": train,
                "rows": len(rows), "counts_by_category": dict(Counter(r["cat"] for r in rows)),
                "majority_hand": majority, "skipped_count": len(skipped),
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "uses_dev_or_report": False, "uses_hidden_object_positions": False,
                "controller": "unchanged harness.docking_controller.load_calibration",
                "reference": "End-effector grasp point at closure transported to base frame at pick start",
                "limitations": ["Not a Task90 detector calibration: centroid-to-grasp reference must match live RGB-D detector",
                                "No fruit-carrying navigation posture or bin-placement docking reference calibrated",
                                "Dataset closure is a kinematic proxy, not independently verified physical grasp",
                                "No simulator or WAM capability tested"]}
    for name, obj in (("grasp_relative_poses_train.json", rows), ("posture_and_gripper_stats_train.json", stats),
                      ("calibration_manifest.json", manifest)):
        write_new(TP.DOCK_CALIB / name, obj)
    return rows, manifest


def export_observations(rows, episodes, output):
    from harness.behavior_data import BehaviorData, decode_reference
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    data = BehaviorData(split="train")
    episodes = episodes or list(data.episode_ids[:3])
    if not episodes or not set(episodes) <= set(data.episode_ids):
        raise ValueError("Observation calibration accepts only Task90 TRAIN episodes")
    segments = json.loads(TP.SEGMENTS.read_text())["episodes"]
    records = []
    tables = {}
    for ep in episodes:
        meta = data.episodes[ep]
        path = data._data_path(meta)
        if path not in tables:
            cols = ["episode_index", "frame_index", "observation.state"] + [
                camera.replace("observation.rgb.", "observation.robot2cam_pose.") for camera in data.cameras]
            tables = {path: pq.read_table(path, columns=cols, use_threads=False)}
        table = tables[path]
        table = table.filter(pc.equal(table["episode_index"], ep))
        selected = {0: {"role": "initial_navigation", "object": None}}
        for r in rows:
            if r["ep"] == ep:
                start = segments[str(ep)]["segments"][r["seg"]]["start"]
                selected[start] = {"role": "pick_start", "object": r["cat"]}
                selected[r["grasp_frame"]] = {"role": "grasp_reference", "object": r["cat"], "hand": r["hand"]}
        for frame, tags in sorted(selected.items()):
            row = table.slice(frame, 1).to_pylist()[0]
            assert row["frame_index"] == frame
            for camera in data.cameras:
                reference = data.video_reference(ep, frame, camera)
                image, timestamp = decode_reference(reference)
                name = f"ep{ep}_f{frame:05d}_{camera.removeprefix('observation.rgb.')}.png"
                output.mkdir(parents=True, exist_ok=True)
                image.save(output / name)
                records.append({"episode": ep, "split": "train", "train_instance_id": meta["task_instance_id"],
                                "frame": frame, "camera": camera, "rgb": str(output / name),
                                "image_size": list(image.size), "decoded_pts": timestamp,
                                "source": reference, "state61": row["observation.state"],
                                "camera_pose7": row[camera.replace("observation.rgb.", "observation.robot2cam_pose.")],
                                "annotation_hint_not_label": tags, "boxes_xyxy": None,
                                "depth": None, "centroid_to_grasp_offset": None})
    manifest = {"task_id": 90, "split": "train", "episodes": episodes, "samples": records,
                "fitted_detector": False, "uses_dev_or_report": False,
                "missing": ["Train RGB boxes/masks for fruit, bin and hard-negative background",
                            "Matched head depth observations and camera intrinsics for metric/reference-point validation",
                            "Train detector fit and held-out calibration checks; do not reuse Task1 can thresholds for fruit"],
                "runtime_interface": "orchestrate.perception.module -> detect_from_observation(rgb, depth, cam_rel_pose, state); obstacle_points reused from harness.object_perception",
                "note": "RGB/pose samples only, no generated labels, no arbitrary color thresholds, not a baseline result"}
    write_new(output / "manifest.json", manifest)
    return {"manifest": str(output / "manifest.json"), "images": len(records), "episodes": episodes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-observations", action="store_true")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--observations-out", type=Path, default=DEFAULT_SAMPLES)
    args = ap.parse_args()
    rows, manifest = build()
    result = {"docking_dir": str(TP.DOCK_CALIB), **{k: manifest[k] for k in ("rows", "counts_by_category", "majority_hand", "skipped_count")}}
    if args.export_observations:
        result["observations"] = export_observations(rows, args.episodes, args.observations_out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
