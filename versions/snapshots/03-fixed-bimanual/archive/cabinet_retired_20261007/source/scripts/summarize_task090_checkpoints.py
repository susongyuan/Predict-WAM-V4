#!/usr/bin/env python3
"""Summarize Task90 checkpoints without mixing navigation failures with grasp trials."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PICK = "pick up from(trash_can,floors)"
LIFT_METRES = 0.03
STABLE_STEPS = 15


def read_jsonl(path):
    rows = []
    malformed = 0
    with path.open() as file:
        for line in file:
            try:
                rows.append(json.loads(line))
            except ValueError:
                malformed += 1
    return rows, malformed


def grasp_result(score, trace, poses):
    """Post-hoc contact-assisted lift only; never an online completion signal."""
    episode = score.get("policy_episode")
    trace = [row for row in trace if row.get("episode") == episode]
    starts = [row for row in trace if row.get("event") == "orch_contract_start" and row.get("key") == PICK]
    chunks = [row for row in trace if row.get("event") == "chunk"]
    result = {"status": "not_reached_manipulation", "success": None,
              "lift_threshold_m": LIFT_METRES, "required_consecutive_steps": STABLE_STEPS,
              "longest_held_lift_steps": 0, "first_grasp_step": None}
    if not starts or not chunks:
        return result
    result["status"] = "unknown"
    if score.get("policy_step_alignment") is not True:
        return result
    headers = [row for row in poses if row.get("type") == "header"]
    if len(headers) != 1:
        return result
    header = headers[0]
    identity = header.get("episode_identity", {})
    if any(identity.get(key) != score.get(key) for key in ("task", "instance_id", "rollout_id", "policy_episode")):
        return result
    targets = [(key, value) for key, value in header.get("entities", {}).items() if value.get("category") == "trash_can"]
    if len(targets) != 1:
        return result
    target, info = targets[0]
    try:
        baseline = float(info["aabb_at_start_world"][0][2])
    except (KeyError, TypeError, IndexError, ValueError):
        return result
    start = starts[0]
    end = next((row["step"] for row in trace if row.get("event") == "orch_contract_end"
                and row.get("index") == start.get("index") and row.get("step", -1) >= start["step"]), score.get("steps", 0))
    if not any(start["step"] <= row.get("step", -1) < end for row in chunks):
        result["status"] = "not_reached_manipulation"
        return result
    # Action t produces observation t+1; the contract-start observation precedes WAM.
    selected = [row for row in poses if type(row.get("step")) is int and start["step"] < row["step"] <= end]
    previous, run, valid, unknown = None, 0, 0, 0
    for row in selected:
        step = row["step"]
        if previous is not None and step != previous + 1:
            run = 0
            unknown += 1
        previous = step
        evidence = row.get("manipulation", {})
        hands = evidence.get("assisted_grasp")
        if evidence.get("status") != "available" or evidence.get("grasping_mode") != "assisted" or not isinstance(hands, dict) or not hands:
            run = 0
            unknown += 1
            continue
        valid += 1
        held_lift = False
        for hand in hands.values():
            z = hand.get("aabb_min_z")
            if (hand.get("object_instance") == target and hand.get("constraint_active") is True
                    and hand.get("release_pending") is False and type(z) in (int, float)
                    and baseline + LIFT_METRES <= z < float("inf")):
                held_lift = True
        run = run + 1 if held_lift else 0
        result["longest_held_lift_steps"] = max(result["longest_held_lift_steps"], run)
        if run == STABLE_STEPS and result["first_grasp_step"] is None:
            result["first_grasp_step"] = step - STABLE_STEPS + 1
    success = result["first_grasp_step"] is not None
    complete = bool(selected) and selected[0]["step"] == start["step"] + 1 and selected[-1]["step"] == end and unknown == 0
    result.update(status="verified_assisted_lift" if success else ("no_qualified_lift" if valid and complete else "unknown"),
                  success=True if success else (False if valid and complete else None),
                  target=target, baseline_bottom_z_m=baseline, valid_evidence_steps=valid, unknown_evidence_steps=unknown,
                  evidence_window="post_action_observations_v2", first_action_step=start["step"],
                  observation_start_step=start["step"] + 1, observation_end_step=end)
    return result


def summarize_run(run):
    manifest_path = run / "manifest.json"
    if not manifest_path.exists():
        return [{"run": run.name, "status": "missing_manifest"}]
    try:
        manifest = json.loads(manifest_path.read_text())
        orch = json.loads(manifest["orchestrate"]) if isinstance(manifest.get("orchestrate"), str) else manifest.get("orchestrate", {})
    except (ValueError, TypeError):
        return [{"run": run.name, "status": "invalid_manifest"}]
    trace, malformed = [], 0
    for path in sorted(run.glob("policy_trace*.jsonl")):
        rows, bad = read_jsonl(path)
        trace.extend(rows)
        malformed += bad
    runtime = {key: manifest.get(key) for key in ("env_wrapper", "samples_averaged", "exec_horizon", "num_inference_steps", "max_steps_override")}
    runtime["orchestrate"] = orch
    runtime["source_sha"] = {key: value for key, value in manifest.get("artifact_sha256", {}).items() if not key.endswith(".pt")}
    version = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    checkpoint = Path(manifest.get("checkpoint", "")).stem
    try:
        checkpoint_step = int(checkpoint.rsplit("_", 1)[-1])
    except ValueError:
        checkpoint_step = None
    resources = json.loads((run / "resource_summary.json").read_text()) if (run / "resource_summary.json").exists() else {}
    base = {"run": run.name, "checkpoint_step": checkpoint_step, "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "interface_sha256": version, "wrapper": manifest.get("env_wrapper"),
            "assistance": ("prepositioned_base" if "preposition" in str(manifest.get("env_wrapper", "")).lower()
                           else "privileged_base" if "privileged_base" in str(manifest.get("env_wrapper", "")) else "observed_navigation"),
            "evaluation_complete": manifest.get("evaluation_complete", False), "evaluator_returncode": manifest.get("evaluator_returncode"),
            "trace_malformed_rows": malformed,
            **{key: resources.get(key) for key in ("elapsed_seconds", "peak_vram_mib", "peak_ram_percent", "peak_cpu_percent")}}
    scores = sorted((run / "eval/json").glob("*.json"))
    if not scores:
        return [{**base, "status": "not_scored"}]
    results = []
    for path in scores:
        try:
            score = json.loads(path.read_text())
        except ValueError:
            results.append({**base, "status": "invalid_score", "score_path": str(path.relative_to(ROOT))})
            continue
        ep_trace = [row for row in trace if row.get("episode") == score.get("policy_episode")]
        ends = [row for row in ep_trace if row.get("event") == "orch_contract_end"]
        pose_file = run / "eval/object_poses" / (path.stem + ".jsonl")
        poses, pose_bad = read_jsonl(pose_file) if pose_file.exists() else ([], 0)
        grasp = grasp_result(score, trace, poses)
        if malformed or pose_bad:
            grasp.update(status="unknown_corrupt_evidence", success=None)
        results.append({**base, "status": "scored", "instance_id": score.get("instance_id"),
                        "seed": score.get("effective_policy_seed"), "policy_episode": score.get("policy_episode"),
                        "steps": score.get("steps"), "task_success": score.get("success"),
                        "q_score": score.get("q_score", {}).get("final"),
                        "wam_chunks": sum(row.get("event") == "chunk" for row in ep_trace),
                        "first_failure": next((row for row in ends if row.get("success") is False), None),
                        "first_grasp": grasp, "first_grasp_status": grasp["status"],
                        "first_grasp_success": grasp["success"],
                        "policy_step_alignment": score.get("policy_step_alignment"),
                        "score_path": str(path.relative_to(ROOT))})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted((ROOT / "outputs/behavior/closed_loop").glob("task090_wam_*"))
    rows = [row for run in runs for row in summarize_run(run)]
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "Task90: every fixed 5k checkpoint, three matched seeds 0/1/2; privilege groups and interface versions separate",
              "first_grasp_scope": "post-hoc evaluator-only contact-assisted bin lift, not online admission or full task success",
              "runs": rows}
    (args.out / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    fields = ("run", "checkpoint_step", "assistance", "interface_sha256", "instance_id", "seed", "steps", "wam_chunks",
              "first_grasp_status", "first_grasp_success", "policy_step_alignment",
              "task_success", "q_score", "evaluation_complete", "evaluator_returncode", "elapsed_seconds", "peak_vram_mib")
    with (args.out / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
