#!/usr/bin/env python3
"""CPU-only, hash-bound description of archived policy commands and scored motion.

Gripper commands/contact are never classified as a successful physical grasp.
This does not mutate a policy, replay a simulator, or alter the evaluation cohort.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_precision_dagger_milestones_v19 import episode

ARMS = (*range(7), *range(8, 15))
GRIPPERS = {"left": 7, "right": 15}
CLOSE, OPEN = 0.2, 0.8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def runs(flags):
    result, start = [], None
    for index, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            result.append([start, index - 1])
            start = None
    return result


def gripper_statistics(values, steps):
    closed = [value <= CLOSE for value in values]
    opened = [value >= OPEN for value in values]
    intervals = runs(closed)
    return {
        "minimum": min(values), "maximum": max(values),
        "closed_frames": sum(closed), "open_frames": sum(opened),
        "open_fraction": sum(opened) / len(values),
        "first_close_control_step": next((s for s, flag in zip(steps, closed) if flag), None),
        "close_intervals_control_steps": [[steps[a], steps[b]] for a, b in intervals],
        "longest_close_run_frames": max((b - a + 1 for a, b in intervals), default=0),
        "longest_open_run_frames": max((b - a + 1 for a, b in runs(opened)), default=0),
    }


def command_episode(row):
    physical = episode(row)  # Independently verifies scored result/rule hashes.
    action_binding = row["evidence_files"]["actions.jsonl"]
    action_path = Path(row["raw_result_path"]).parent / "actions.jsonl"
    require(str(action_path.resolve()) == action_binding["path"], "Wrong action archive path")
    require(sha256(action_path) == action_binding["sha256"], "Archived action hash changed")
    with action_path.open() as stream:
        actions = [json.loads(line) for line in stream]
    require(bool(actions), "No dispatched actions")
    steps = [a["control_step"] for a in actions]
    require(steps == list(range(len(actions))), "Nonconsecutive action archive")
    require(len(actions) == row["control_steps"], "Action/result count disagreement")
    raw = [a["policy"]["model_raw_action"] for a in actions]
    smooth = [a["policy"]["smoothed_action"] for a in actions]
    sent = [a["dispatched_action"] for a in actions]
    require(all(len(v) == 16 and all(math.isfinite(x) for x in v)
                for sequence in (raw, smooth, sent) for v in sequence),
            "Nonfinite or wrong-dimensional native action")
    close_plans = {side: {"replans": 0, "no_close_in_32": 0,
                         "first_close_in_executed_prefix_8": 0,
                         "first_close_only_in_discarded_tail_24": 0,
                         "first_close_offset_histogram": {}}
                   for side in GRIPPERS}
    for a in actions:
        if not a["policy"]["replanned"]:
            continue
        chunk = a["policy"]["raw_denormalized_chunk"]
        require(len(chunk) == 32 and all(len(v) == 16 for v in chunk), "Unexpected plan chunk")
        require(a["policy"]["execution_prefix"] == 8, "Different execution horizon")
        for side, index in GRIPPERS.items():
            plan = close_plans[side]
            plan["replans"] += 1
            offset = next((t for t, v in enumerate(chunk) if v[index] <= CLOSE), None)
            if offset is None:
                plan["no_close_in_32"] += 1
            else:
                key = str(offset)
                plan["first_close_offset_histogram"][key] = plan["first_close_offset_histogram"].get(key, 0) + 1
                if offset < 8:
                    plan["first_close_in_executed_prefix_8"] += 1
                else:
                    plan["first_close_only_in_discarded_tail_24"] += 1
    grippers = {}
    for side, index in GRIPPERS.items():
        expected = [max(0.0, min(1.0, v[index])) for v in raw]
        observed = [v[index] for v in sent]
        grippers[side] = {
            "model_raw": gripper_statistics([v[index] for v in raw], steps),
            "dispatched": gripper_statistics(observed, steps),
            "smoothing_gripper_max_abs_change": max(abs(r[index] - s[index]) for r, s in zip(raw, smooth)),
            "dispatched_vs_clamped_model_max_abs_change": max(abs(a - b) for a, b in zip(expected, observed)),
            "close_threshold_preserved_every_step": all((r[index] <= CLOSE) == (s[index] <= CLOSE)
                                                        for r, s in zip(raw, sent)),
            "open_threshold_preserved_every_step": all((r[index] >= OPEN) == (s[index] >= OPEN)
                                                       for r, s in zip(raw, sent)),
            "planned_close_timing": close_plans[side],
        }
    pairs = [(r, s) for r, s in zip(raw, smooth)]
    smooth_sent = [(s, a) for s, a in zip(smooth, sent)]

    def adjustment(sequence):
        differences = [abs(a[i] - b[i]) for a, b in sequence for i in ARMS]
        return {"rms_rad": math.sqrt(sum(v * v for v in differences) / len(differences)),
                "max_rad": max(differences),
                "changed_frames_above_1e_6_rad": sum(any(abs(a[i] - b[i]) > 1e-6 for i in ARMS)
                                                       for a, b in sequence)}

    return {
        "physical_progress": physical,
        "actions_sha256": action_binding["sha256"], "action_frames": len(actions),
        "braking_frame_fraction": row["braking_steps"] / len(actions),
        "model_to_smoothing_arm_adjustment": adjustment(pairs),
        "smoothed_to_dispatched_arm_adjustment": adjustment(smooth_sent),
        "grippers": grippers,
        "raw_action_source": "policy.model_raw_action; outer raw_action is a protocol alias and is not used",
        "planning_note": "Every replan executes at most the first8 commands. Tail-only closing can describe postponement, not its neural cause.",
    }


def analyze(paths, allow_partial=False):
    result, bindings = {}, {}
    for milestone, path in paths.items():
        path = Path(path)
        # A single loaded snapshot is used if a live cohort is still growing.
        snapshot = path.read_bytes()
        cohort = json.loads(snapshot)
        require(cohort["status"] == "completed" or allow_partial, "Incomplete evaluation cohort")
        rows = [command_episode(row) for row in cohort["episodes"]]
        require(len({r["physical_progress"]["seed"] for r in rows}) == len(rows), "Duplicate cases")
        result[str(milestone)] = {
            "cohort_status_at_snapshot": cohort["status"], "episode_count": len(rows),
            "full_successes": sum(r["physical_progress"]["full_task_success"] for r in rows),
            "scores": [r["physical_progress"]["observed_base_score"] for r in rows],
            "episodes": rows,
        }
        bindings[str(milestone)] = {"path": str(path.resolve()),
                                   "snapshot_sha256": hashlib.sha256(snapshot).hexdigest()}
    return {
        "schema": "precision-archived-execution-command-diagnostic-v23",
        "time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "thresholds": {"closed_le": CLOSE, "open_ge": OPEN},
        "milestones": result, "summary_snapshots": bindings,
        "analyzer_sha256": sha256(__file__),
        "interpretation": "Offline descriptive evidence. A command to close or physical contact does not establish a scored grasp. No isolated architecture/weight ablation, so these statistics do not identify causal model changes or under/overfitting.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", required=True, help="milestone=summary.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    paths = dict(item.split("=", 1) for item in args.cohort)
    out = Path(args.out)
    require(not out.exists(), "Preserve the existing diagnostic; use a fresh receipt")
    value = analyze(paths, args.allow_partial)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"out": str(out.resolve()), "milestones": {
        key: {"count": value["episode_count"], "scores": value["scores"],
              "left_first_close": [r["grippers"]["left"]["dispatched"]["first_close_control_step"]
                                   for r in value["episodes"]],
              "failure_stages": [r["physical_progress"]["failure_stage"] for r in value["episodes"]]}
        for key, value in value["milestones"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
