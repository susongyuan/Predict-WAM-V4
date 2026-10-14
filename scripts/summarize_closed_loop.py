#!/usr/bin/env python3
"""Summarise scored episodes, policy candidates and evaluator-only raw object evidence.

    python3 scripts/summarize_closed_loop.py outputs/behavior/closed_loop/<run_name> --read-only
"""
import argparse
from collections import Counter, defaultdict
import glob
import json
from statistics import median, quantiles
from pathlib import Path


def episode_key(row):
    if (not isinstance(row, dict) or not isinstance(row.get("task"), str) or not row["task"]
            or any(type(row.get(k)) is not int or row[k] < 0 for k in ("instance_id", "rollout_id"))):
        return None
    return row["task"], row["instance_id"], row["rollout_id"]


def binding_key(row):
    key = episode_key(row)
    ep = row.get("policy_episode") if isinstance(row, dict) else None
    return (*key, ep) if key is not None and type(ep) is int and ep >= 0 else None


def main(run_dir, *, write=True):
    run = Path(run_dir)
    out = {"run": run.name}
    man = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
    for key in ("official_score_valid", "evaluator_returncode", "launcher_returncode", "seed",
                "effective_policy_seed", "effective_evaluator_seed", "seed_behavior", "artifact_changes"):
        out[key] = man.get(key)
    out["evaluation_complete"] = man.get("evaluation_complete", False)
    res = json.loads((run / "resource_summary.json").read_text()) if (run / "resource_summary.json").exists() else {}
    out["resources"] = {k: res.get(k) for k in ("abort_reason", "elapsed_seconds", "peak_ram_percent", "peak_gpu_percent", "peak_vram_mib")}
    episodes = []
    # Only evaluator JSONs are rollouts. Never infer task success from a video sidecar or a policy closure.
    for f in sorted((run / "eval/json").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and ("q_score" in d or "success" in d):
            episodes.append({"file": str(f.relative_to(run)),
                             **{k: d.get(k) for k in ("task", "instance_id", "rollout_id", "policy_episode", "steps", "success",
                                                     "policy_step_alignment", "effective_policy_seed", "effective_evaluator_seed")},
                             "q_score": d.get("q_score", {}).get("final") if isinstance(d.get("q_score"), dict) else None})
    out["episodes"] = episodes
    by_policy_episode, by_binding = defaultdict(list), defaultdict(list)
    for episode in episodes:
        key = binding_key(episode)
        if key is not None:
            by_policy_episode[episode["policy_episode"]].append(episode)
            by_binding[key].append(episode)

    rows, trace_sessions = [], defaultdict(set)
    for trace in sorted(run.glob("policy_trace_*.jsonl")):
        for number, line in enumerate(trace.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.update(trace_file=str(trace.relative_to(run)), trace_line=number)
            rows.append(row)
            if type(row.get("episode")) is int:
                trace_sessions[row["episode"]].add(str(trace))
    contract_ends = [dict(r) for r in rows if r.get("event") == "orch_contract_end"]

    def is_manipulation(record):
        return not (str(record.get("key", "")).startswith("move to(") or record.get("skill") == "move to")

    # Even legacy `verified: true` is a policy claim, not an independent physical verification.
    for r in contract_ends:
        r.update(physical_verified=None, evidence_status="unknown", evidence_reason="missing_episode_or_object_log_identity")
    perceptions = [r for r in contract_ends if r.get("success") is True and is_manipulation(r)]
    for r in contract_ends:
        ep = r.get("episode")
        matches = by_policy_episode.get(ep, []) if type(ep) is int else []
        r["episode_identity"] = None
        if len(matches) == 1 and len(trace_sessions[ep]) == 1:
            r["episode_identity"] = {k: matches[0][k] for k in ("task", "instance_id", "rollout_id", "policy_episode")}
    if rows:
        chunks = [r for r in rows if r.get("event") == "chunk"]
        lat = [r["infer_ms"] for r in chunks]
        out["policy"] = {"chunks": len(chunks), "infer_ms_p50": median(lat) if lat else None,
                         "infer_ms_p95": quantiles(lat, n=20, method="inclusive")[-1] if len(lat) > 1 else (lat[0] if lat else None),
                         "action_bounds_events": sum(r.get("event") == "action_bounds" for r in rows),
                         "clipped_components": sum(r.get("clipped_components", 0) for r in rows),
                         "resets": sum(r.get("event") == "reset" for r in rows),
                         "closure_events": [r for r in rows if r.get("event") == "closure"],
                         "heuristic_closure_candidates": [r for r in rows if r.get("event") == "orch_closure_candidate"],
                         "orchestrator_contract_ends": contract_ends,
                         "perception_candidates": perceptions,
                         "verified_manipulation_successes": None,
                         "unverified_manipulation_advancements": len(perceptions),
                         "last_progress": [r for r in rows if r.get("event") == "progress"][-3:]}
    out["record_status"] = {"run_record_complete": bool(out["evaluation_complete"]),
                            "task_successes": sum(episode.get("success") is True for episode in episodes),
                            "task_failures": sum(episode.get("success") is False for episode in episodes),
                            "physical_verification": "unknown: no independently calibrated manipulation verifier",
                            "note": "record completeness, perception candidates, raw joint evidence and task success are distinct"}

    joint_logs = sorted((run / "eval/object_joints").glob("*.jsonl"))
    headers = {}
    for f in joint_logs:
        with f.open() as file:
            first = file.readline()
        header = json.loads(first) if first.strip() else {}
        headers[f] = header if header.get("type") == "header" else {}
    log_bindings = Counter(binding_key(h.get("episode_identity")) for h in headers.values())
    if joint_logs:
        out["object_joints"] = {}
    for f in joint_logs:
        data = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        header = headers[f]
        steps = [r for r in data if "joints" in r]
        ids = [r.get("step") for r in steps]
        valid_steps = (all(type(i) is int and i >= 0 for i in ids)
                       and len(set(ids)) == len(ids) and ids == sorted(ids))
        at_step = {r["step"]: r for r in steps} if valid_steps else {}
        entities = {}
        for inst, info in header.get("entities", {}).items():
            observed = [r for r in steps if inst in r["joints"]]
            if not observed:
                continue
            vals = [r["joints"][inst] for r in observed]
            n = len(vals[0])
            entities[inst] = {"joints": info["joints"], "lower": info["lower"], "upper": info["upper"],
                              "initial": vals[0], "final": vals[-1],
                              "min": [min(v[j] for v in vals) for j in range(n)],
                              "max": [max(v[j] for v in vals) for j in range(n)],
                              "argmax_step": [max(observed, key=lambda r: r["joints"][inst][j]).get("step") for j in range(n)]}
        key = binding_key(header.get("episode_identity"))
        matches = by_binding.get(key, [])
        bound = key is not None and log_bindings[key] == 1 and len(matches) == 1
        audit = []
        if bound:
            result = matches[0]
            for r in contract_ends:
                if binding_key(r["episode_identity"]) != key:
                    continue
                st = r.get("step")
                post = r.get("postcondition") if isinstance(r.get("postcondition"), dict) else {}
                obs_id = post.get("observation_id")
                reason = None
                if result.get("policy_step_alignment") is not True:
                    reason = "policy_step_alignment_unconfirmed"
                elif not valid_steps:
                    reason = "invalid_or_duplicate_joint_steps"
                elif type(st) is not int or st < 0 or type(result.get("steps")) is not int or st > result["steps"]:
                    reason = "invalid_or_out_of_range_event_step"
                elif obs_id is not None and (type(obs_id) is not int or obs_id != st):
                    reason = "postcondition_observation_step_mismatch"
                elif st not in at_step:
                    reason = "exact_joint_step_missing"
                r.update(evidence_status="unknown" if reason else "matched_raw_joint_evidence",
                         evidence_reason=reason, evidence_file=str(f.relative_to(run)))
                audit.append({"episode": r["episode"], "key": r.get("key"), "step": st, "success": r.get("success"),
                              "perception_candidate": r.get("success") is True and is_manipulation(r),
                              "postcondition": r.get("postcondition"), "physical_verified": None,
                              "evidence_status": r["evidence_status"], "evidence_reason": reason,
                              "object_joints_at_step": None if reason else at_step[st]["joints"]})
        end = next((r for r in reversed(data) if r.get("type") == "end"), {})
        out["object_joints"][str(f.relative_to(run))] = {
            "episode_identity": header.get("episode_identity"),
            "binding_status": "matched" if bound else "unknown_missing_or_ambiguous_identity",
            "environment_steps": end.get("environment_steps"), "entities": entities,
            "privileged_data_use": header.get("privileged_data_use"), "contract_event_audit": audit,
            "perception_candidate_audit": [r for r in audit if r["perception_candidate"]]}
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if write:
        (run / "summary.json").write_text(text + "\n")
    print(text)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--read-only", action="store_true", help="Print without rewriting historical summary.json")
    args = parser.parse_args()
    main(args.run_dir or sorted(glob.glob("outputs/behavior/closed_loop/*"))[-1], write=not args.read_only)
