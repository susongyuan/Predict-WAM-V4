#!/usr/bin/env python3
"""Audit temporal coverage and motion consistency of Task-13 open-lid demonstrations.

This is a data-only diagnostic.  It reads the cached public demonstrations and reports
how much data exists, where trunk/arm motion occurs, whether both arms move together,
and whether short-horizon joint-motion directions cancel across demonstrations at the
same normalized phase.  It does not use simulator state, run a policy, or measure task
success.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "data/behavior-1k/2026-challenge-demos/skill_segments_task013.json"
INDEX = ROOT / "data/lowdim_cache/index.json"
LOWDIM = ROOT / "data/lowdim_cache"

STATE_GROUPS = {
    "left_arm": (3, 10),
    "right_arm": (28, 35),
    "trunk": (53, 57),
}
VELOCITY_GROUPS = {
    "left_arm": (10, 17),
    "right_arm": (35, 42),
    "trunk": (57, 61),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {name: float(np.quantile(values, q)) for name, q in
            (("min", 0.0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("max", 1.0))}


def direction_stats(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    moving = norms > 1e-6
    if not moving.any():
        return {"n": int(len(vectors)), "moving": 0, "mean_norm": 0.0,
                "coherence": None, "mean_direction_cos": None, "fraction_positive": None}
    unit = vectors[moving] / norms[moving, None]
    mean_unit = unit.mean(axis=0)
    coherence = float(np.linalg.norm(mean_unit))
    if coherence > 1e-9:
        direction = mean_unit / coherence
        cosines = unit @ direction
        mean_cos = float(cosines.mean())
        fraction_positive = float((cosines > 0).mean())
    else:
        mean_cos = 0.0
        fraction_positive = float("nan")
    return {"n": int(len(vectors)), "moving": int(moving.sum()), "mean_norm": float(norms.mean()),
            "coherence": coherence, "mean_direction_cos": mean_cos,
            "fraction_positive": fraction_positive,
            "cancellation_ratio": float(np.linalg.norm(vectors[moving].mean(axis=0)) / norms[moving].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase-bins", type=int, default=10)
    parser.add_argument("--horizons", default="32,64,128,256")
    parser.add_argument("--arm-active-rad-s", type=float, default=0.10)
    parser.add_argument("--trunk-active-rad-s", type=float, default=0.05)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    args.out.mkdir(parents=True)

    segment_data = json.loads(SEGMENTS.read_text())
    index = json.loads(INDEX.read_text())
    horizons = [int(value) for value in args.horizons.split(",")]
    rows = []
    for episode_text, episode_info in segment_data["episodes"].items():
        episode = int(episode_text)
        split = index["episodes"][episode_text]["split"]
        for segment_index, segment in enumerate(episode_info["segments"]):
            if segment["skill"] == "open lid":
                rows.append({"episode": episode, "split": split, "segment_index": segment_index,
                             "start": int(segment["start"]), "end": int(segment["end"]),
                             "length": int(segment["end"] - segment["start"])})

    phase_records = {split: {str(i): [] for i in range(args.phase_bins)} for split in ("train", "validation")}
    horizon_vectors = {split: {str(h): {str(i): {group: [] for group in STATE_GROUPS}
                                             for i in range(args.phase_bins)} for h in horizons}
                       for split in ("train", "validation")}
    episode_summaries = []
    for row in rows:
        state = np.load(LOWDIM / f"episode_{row['episode']}_state.npy", mmap_mode="r")
        lo, hi = row["start"], row["end"]
        segment_state = np.asarray(state[lo:hi], dtype=np.float32)
        split = row["split"]
        summary = dict(row)
        summary["net_displacement"] = {}
        summary["path_length"] = {}
        for group, (start, end) in STATE_GROUPS.items():
            q = segment_state[:, start:end]
            summary["net_displacement"][group] = float(np.linalg.norm(q[-1] - q[0]))
            summary["path_length"][group] = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
        episode_summaries.append(summary)

        for phase in range(args.phase_bins):
            b0 = int(round(phase * len(segment_state) / args.phase_bins))
            b1 = int(round((phase + 1) * len(segment_state) / args.phase_bins))
            b1 = max(b1, b0 + 1)
            velocities = {}
            for group, (start, end) in VELOCITY_GROUPS.items():
                velocities[group] = np.linalg.norm(segment_state[b0:b1, start:end], axis=1)
            left = velocities["left_arm"] > args.arm_active_rad_s
            right = velocities["right_arm"] > args.arm_active_rad_s
            trunk = velocities["trunk"] > args.trunk_active_rad_s
            phase_records[split][str(phase)].append({
                "frames": int(b1 - b0),
                "left_active": float(left.mean()),
                "right_active": float(right.mean()),
                "both_active": float((left & right).mean()),
                "left_only": float((left & ~right).mean()),
                "right_only": float((~left & right).mean()),
                "neither_arm": float((~left & ~right).mean()),
                "trunk_active": float(trunk.mean()),
            })
            anchor = lo + b0
            for horizon in horizons:
                target = min(anchor + horizon, hi - 1)
                for group, (start, end) in STATE_GROUPS.items():
                    vector = np.asarray(state[target, start:end] - state[anchor, start:end], dtype=np.float64)
                    horizon_vectors[split][str(horizon)][str(phase)][group].append(vector)

    report = {
        "diagnostic": "open-lid demonstration phase and horizon audit",
        "policy_inference": False,
        "closed_loop": False,
        "task_success_evaluated": False,
        "sources": {str(SEGMENTS.relative_to(ROOT)): sha256(SEGMENTS), str(INDEX.relative_to(ROOT)): sha256(INDEX)},
        "thresholds": {"arm_active_rad_s": args.arm_active_rad_s, "trunk_active_rad_s": args.trunk_active_rad_s},
        "phase_bins": args.phase_bins,
        "horizons_frames": horizons,
        "splits": {},
        "episodes": episode_summaries,
    }
    for split in ("train", "validation"):
        split_rows = [row for row in episode_summaries if row["split"] == split]
        result = {
            "segments": len(split_rows),
            "total_frames": int(sum(row["length"] for row in split_rows)),
            "length_frames": quantiles([row["length"] for row in split_rows]),
            "length_seconds_at_30hz": quantiles([row["length"] / 30.0 for row in split_rows]),
            "net_displacement": {group: quantiles([row["net_displacement"][group] for row in split_rows])
                                 for group in STATE_GROUPS},
            "path_length": {group: quantiles([row["path_length"][group] for row in split_rows])
                            for group in STATE_GROUPS},
            "phase_activity": {},
            "horizon_direction_consistency": {},
        }
        for phase, records in phase_records[split].items():
            weights = np.asarray([record["frames"] for record in records], dtype=np.float64)
            result["phase_activity"][phase] = {
                key: float(np.average([record[key] for record in records], weights=weights))
                for key in ("left_active", "right_active", "both_active", "left_only", "right_only", "neither_arm", "trunk_active")
            }
        for horizon in horizons:
            result["horizon_direction_consistency"][str(horizon)] = {}
            for phase in range(args.phase_bins):
                result["horizon_direction_consistency"][str(horizon)][str(phase)] = {
                    group: direction_stats(horizon_vectors[split][str(horizon)][str(phase)][group])
                    for group in STATE_GROUPS
                }
        report["splits"][split] = result

    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    train = report["splits"]["train"]
    lines = [
        "# Open-lid phase audit",
        "",
        "Data-only diagnostic; not policy inference or task success evidence.",
        "",
        f"- Train segments: {train['segments']}; frames: {train['total_frames']}; median length: "
        f"{train['length_frames']['median']:.0f} frames ({train['length_seconds_at_30hz']['median']:.1f} s).",
        f"- Validation segments: {report['splits']['validation']['segments']}.",
        "- Motion activity by normalized phase (train):",
        "",
        "| Phase | left | right | both | left-only | right-only | trunk |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase, values in train["phase_activity"].items():
        lines.append(f"| {int(phase) / args.phase_bins:.1f}-{(int(phase) + 1) / args.phase_bins:.1f} | "
                     f"{values['left_active']:.2f} | {values['right_active']:.2f} | {values['both_active']:.2f} | "
                     f"{values['left_only']:.2f} | {values['right_only']:.2f} | {values['trunk_active']:.2f} |")
    lines += ["", "- Cross-demonstration direction coherence (train; 1=fully aligned, 0=cancelling):", "",
              "| Horizon | Phase | trunk | left arm | right arm |", "|---:|---:|---:|---:|---:|"]
    for horizon in horizons:
        for phase in range(args.phase_bins):
            values = train["horizon_direction_consistency"][str(horizon)][str(phase)]
            lines.append(f"| {horizon} | {phase / args.phase_bins:.1f} | "
                         f"{values['trunk']['coherence'] or 0:.2f} | {values['left_arm']['coherence'] or 0:.2f} | "
                         f"{values['right_arm']['coherence'] or 0:.2f} |")
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(args.out), "train_segments": train["segments"],
                      "train_frames": train["total_frames"],
                      "median_seconds": train["length_seconds_at_30hz"]["median"]}, indent=2))


if __name__ == "__main__":
    main()
