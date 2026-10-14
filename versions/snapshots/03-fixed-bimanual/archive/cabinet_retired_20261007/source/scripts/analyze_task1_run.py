#!/usr/bin/env python3
"""Offline audit of a Task 1 closed-loop run against evaluator-side ground truth (never a policy input).

Reads the policy trace (contracts, docking, perception, calibrated predicates) and the evaluator's pose log
(`--log-object-poses`: world poses + per-step BDDL goal literals) and reports:
  - outcome: success / Q / steps from the evaluator JSON;
  - contract timeline with the policy's own completion verdicts;
  - predicate audit: for every manipulation verdict, what the ground truth says (object lifted / literal
    `inside(can, ashcan)` true) -> agreement table; this is how the calibrated predicates are checked;
  - docking audit: true base-frame position of the nearest target-category object when docking reported success,
    versus the calibrated docking target;
  - perception audit: logged detections versus ground-truth objects in the base frame (error, false positives).

    python3 scripts/analyze_task1_run.py outputs/behavior/closed_loop/<run>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_base(robot7, p_world):
    yaw = quat_yaw(robot7[3:7])
    d = np.asarray(p_world[:2]) - np.asarray(robot7[:2])
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def category(inst):
    return "can_of_soda" if inst.startswith("can__of__soda") else "trash_can" if inst.startswith("ashcan") else inst


def main(run):
    run = Path(run)
    out = {"run": str(run)}
    results = [json.loads(p.read_text()) for p in sorted((run / "eval/json").glob("*.json"))]
    out["episodes"] = [{k: r.get(k) for k in ("instance_id", "steps", "success", "terminated", "truncated")}
                       | {"q_score": (r.get("q_score") or {}).get("final")} for r in results]
    trace = [e for p in sorted(run.glob("policy_trace_*.jsonl")) for e in load_jsonl(p)]
    poses = sorted((run / "eval/object_poses").glob("*.jsonl"))
    gt = load_jsonl(poses[0]) if poses else []
    header = gt[0] if gt and gt[0].get("type") == "header" else {}
    rows = {r["step"]: r for r in gt if "step" in r}
    literals = [l for opt in header.get("goal_literals", {}).get("options", [[]])[:1] for l in opt]

    def literal_true(step, text_part):
        r = rows.get(step)
        if r is None or "goal_literals" not in r:
            return None
        vals = r["goal_literals"][0]
        return any(v for l, v in zip(literals, vals) if text_part in json.dumps(l))

    # contract timeline
    starts = {(e["index"], e.get("attempt")): e for e in trace if e["event"] == "orch_contract_start"}
    ends = [e for e in trace if e["event"] == "orch_contract_end"]
    timeline = []
    for e in ends:
        timeline.append({"index": e["index"], "key": e["key"], "success": e["success"], "end_step": e["step"],
                         "steps": e.get("steps"), "predicate": e.get("predicate"),
                         "reason": (e.get("postcondition") or {}).get("reason")})
    out["timeline"] = timeline
    out["recovery_events"] = [{k: e.get(k) for k in ("event", "step", "key", "object", "reason")} for e in trace
                              if e["event"] in ("orch_recovery_rollback", "orch_recovery_reapproach", "orch_replan",
                                                "skip_failed_object", "orch_hold")]

    # predicate audit against ground truth
    audit = []
    for e in ends:
        key, step = e["key"], e["step"]
        if key.startswith("move to") or step not in rows:
            continue
        r = rows[step]
        objs = r["objects"]
        robot = r["robot_base"]
        if key.startswith("pick up from(can_of_soda"):
            lifted = [inst for inst, p in objs.items() if category(inst) == "can_of_soda" and p[2] > 0.15]
            truth = bool(lifted)
        elif key.startswith("pick up from(trash_can"):
            truth = any(p[2] > 0.2 for inst, p in objs.items() if category(inst) == "trash_can")
        elif key.startswith("place in"):
            later = [rows[s] for s in range(step, step + 90) if s in rows and "goal_literals" in rows[s]]
            before = rows.get(max(0, e["step"] - (e.get("steps") or 0)))
            n_before = sum(before["goal_literals"][0]) if before and "goal_literals" in before else None
            n_after = max((sum(x["goal_literals"][0]) for x in later), default=None)
            truth = None if n_before is None or n_after is None else n_after > n_before
        else:
            truth = None
        audit.append({"key": key, "step": step, "policy_verdict": e["success"], "ground_truth": truth,
                      "reason": (e.get("postcondition") or {}).get("reason")})
    out["predicate_audit"] = audit
    known = [a for a in audit if a["ground_truth"] is not None]
    out["predicate_agreement"] = (sum(a["policy_verdict"] == a["ground_truth"] for a in known), len(known))

    # docking audit
    dock_audit = []
    for e in trace:
        if e["event"] == "dock" and e.get("docked") and e.get("step") in rows:
            r = rows[e["step"]]
            label = e.get("label")
            cands = [(inst, to_base(r["robot_base"], p)) for inst, p in r["objects"].items()
                     if category(inst) == label and p[2] < 0.15 + (0.3 if label == "trash_can" else 0)]
            if not cands:
                continue
            believed = np.asarray(e.get("p_base") or [np.nan, np.nan])
            inst, true_rel = min(cands, key=lambda c: np.linalg.norm(c[1] - believed))
            target = np.asarray(e.get("dock_target"))
            dock_audit.append({"step": e["step"], "label": label, "object": inst, "believed_base": believed.round(3).tolist(),
                               "true_base": true_rel.round(3).tolist(), "perception_error_m": round(float(np.linalg.norm(true_rel - believed)), 3),
                               "true_dock_error_m": round(float(np.linalg.norm(true_rel - target)), 3)})
    out["docking_audit"] = dock_audit

    # perception audit
    errs, fps, n_det = {"can_of_soda": [], "trash_can": []}, 0, 0
    for e in trace:
        if e["event"] != "perception" or e.get("step") not in rows:
            continue
        r = rows[e["step"]]
        for d in e.get("detections", []):
            n_det += 1
            truth = [to_base(r["robot_base"], p) for inst, p in r["objects"].items() if category(inst) == d["label"]]
            if not truth:
                fps += 1
                continue
            err = min(float(np.linalg.norm(t - np.asarray(d["xyz_base"][:2]))) for t in truth)
            (errs.setdefault(d["label"], [])).append(err)
            if err > 0.5:
                fps += 1
    out["perception_audit"] = {"detections": n_det, "far_or_false(>0.5 m)": fps,
                               **{f"{k}_median_error_m": (round(float(np.median(v)), 3) if v else None) for k, v in errs.items()}}
    (run / "task1_audit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("episodes", "predicate_agreement", "perception_audit")}, indent=1))
    for t in timeline:
        print(f"  [{t['end_step']:5d}] {'OK ' if t['success'] else 'FAIL'} {t['key']:40s} {t['predicate']} {t['reason'] or ''}")
    for a in dock_audit:
        print(f"  dock@{a['step']}: {a['label']} believed {a['believed_base']} true {a['true_base']} -> true dock error {a['true_dock_error_m']} m")
    for a in audit:
        print(f"  audit {a['key']} @{a['step']}: policy={a['policy_verdict']} truth={a['ground_truth']} ({a['reason']})")


if __name__ == "__main__":
    main(sys.argv[1])
