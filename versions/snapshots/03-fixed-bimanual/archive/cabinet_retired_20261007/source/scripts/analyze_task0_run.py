#!/usr/bin/env python3
"""Offline audit of a Task 0 (turning_on_radio) closed-loop run against evaluator-side ground truth (never a policy input).

Reads the evaluator JSON, the evaluator pose log (`--log-object-poses`: world poses of the robot base, the radio and the
coffee table + per-step BDDL goal literal + the radio's ToggledOn value / hand_in_marker_steps) and, when present, the
policy trace (contract starts/ends, docking, perception). Reports:
  - outcome: success / Q / steps;
  - physical timeline: first step the radio is lifted >= 0.10 m (root z above its step-0 height), first step
    toggled_on is true, max hand_in_marker_steps (5 consecutive = toggle), toggles back off;
  - contract timeline with the policy's own verdicts, and a predicate audit (policy verdict vs the evaluator truth at the
    verdict step: pick = radio lifted >= 0.05 m; press = toggled_on true within 30 steps; place on = radio released within
    0.05 m of its initial height, i.e. back on a support);
  - docking audit: at every 'dock' event with docked=True, the true radio position in the base frame vs what the policy
    believed and vs the calibrated docking target;
  - perception audit: logged radio detections vs the true radio position in the base frame.
Evaluator-side data is evidence for the audit only.

    python3 scripts/analyze_task0_run.py outputs/behavior/closed_loop/<run>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RADIO, TABLE = "radio_receiver.n.01_1", "table.n.02_1"


def load_jsonl(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_tilt_deg(q):
    x, y, z, w = q
    zz = 1 - 2 * (x * x + y * y)          # world z of the object's z axis
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def to_base(robot7, p_world):
    yaw = quat_yaw(robot7[3:7])
    d = np.asarray(p_world[:2], float) - np.asarray(robot7[:2], float)
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def main(run):
    run = Path(run)
    out = {"run": str(run), "interpretation": "evaluator-side audit only; never a policy input"}
    results = [json.loads(p.read_text()) for p in sorted((run / "eval/json").glob("*.json"))]
    out["episodes"] = [{k: r.get(k) for k in ("instance_id", "steps", "success", "terminated", "truncated")}
                       | {"q_score": (r.get("q_score") or {}).get("final")} for r in results]
    poses = sorted((run / "eval/object_poses").glob("*.jsonl"))
    gt = load_jsonl(poses[0]) if poses else []
    rows = {r["step"]: r for r in gt if "step" in r}
    steps = sorted(rows)
    phys = {"pose_rows": len(rows)}
    if steps:
        r0 = rows[steps[0]]
        z0 = r0["objects"][RADIO][2]
        lift = {s: rows[s]["objects"][RADIO][2] - z0 for s in steps}
        tog = {s: rows[s].get("toggled_on", {}).get(RADIO) for s in steps}
        on = [s for s in steps if tog[s] and tog[s][0]]
        marker = [tog[s][1] for s in steps if tog[s]]
        flips = sum(1 for a, b in zip(steps, steps[1:]) if tog[a] and tog[b] and bool(tog[a][0]) != bool(tog[b][0]))
        phys.update({
            "radio_root_lift_max_m": round(float(max(lift.values())), 3),
            "first_step_lift_ge_0.10m": next((s for s in steps if lift[s] >= 0.10), None),
            "radio_root_lift_at_end_m": round(float(lift[steps[-1]]), 3),
            "radio_tilt_max_deg": round(max(quat_tilt_deg(rows[s]["objects"][RADIO][3:7]) for s in steps), 1),
            "first_step_toggled_on": on[0] if on else None, "toggled_on_at_end": bool(tog[steps[-1]] and tog[steps[-1]][0]),
            "toggle_value_flips": flips, "hand_in_marker_steps_max": int(max(marker)) if marker else None,
            "steps_with_hand_in_marker": int(sum(1 for m in marker if m and m > 0)),
            "radio_base_xy_at_step0": to_base(r0["robot_base"], r0["objects"][RADIO]).round(3).tolist(),
            "radio_distance_at_step0_m": round(float(np.linalg.norm(to_base(r0["robot_base"], r0["objects"][RADIO]))), 3)})
    out["physical"] = phys

    trace = [e for p in sorted(run.glob("policy_trace_*.jsonl")) for e in load_jsonl(p)]
    out["trace_events"] = len(trace)
    ends = [e for e in trace if e.get("event") == "orch_contract_end"]
    out["timeline"] = [{"index": e.get("index"), "key": e.get("key"), "success": e.get("success"), "end_step": e.get("step"),
                        "steps": e.get("steps"), "predicate": e.get("predicate"),
                        "verifier_kind": (e.get("postcondition") or {}).get("verifier_kind"),
                        "reason": (e.get("postcondition") or {}).get("reason")} for e in ends]
    out["recovery_events"] = [{k: e.get(k) for k in ("event", "step", "key", "object", "reason")} for e in trace
                              if e.get("event") in ("orch_recovery_rollback", "orch_recovery_reapproach", "orch_replan",
                                                    "skip_failed_object", "orch_hold")]
    audit = []
    for e in ends:
        key, step = str(e.get("key")), e.get("step")
        if key.startswith("move to") or step not in rows:
            continue
        if key.startswith("pick up from"):
            truth = bool(rows[step]["objects"][RADIO][2] - rows[steps[0]]["objects"][RADIO][2] >= 0.05)
        elif key.startswith("press"):
            window = [rows[s] for s in range(step - 30, step + 31) if s in rows]
            truth = any((w.get("toggled_on", {}).get(RADIO) or [False])[0] for w in window)
        elif key.startswith("place on"):
            later = [rows[s] for s in range(step, step + 60) if s in rows]
            truth = bool(later) and abs(later[-1]["objects"][RADIO][2] - rows[steps[0]]["objects"][RADIO][2]) < 0.05
        else:
            truth = None
        audit.append({"key": key, "step": step, "policy_verdict": e.get("success"), "ground_truth": truth,
                      "reason": (e.get("postcondition") or {}).get("reason")})
    out["predicate_audit"] = audit
    known = [a for a in audit if a["ground_truth"] is not None]
    out["predicate_agreement"] = [sum(a["policy_verdict"] == a["ground_truth"] for a in known), len(known)]

    dock = []
    for e in trace:
        if e.get("event") == "dock" and e.get("docked") and e.get("step") in rows:
            r = rows[e["step"]]
            true_rel = to_base(r["robot_base"], r["objects"][RADIO])
            believed = np.asarray(e.get("p_base") or [np.nan, np.nan], float)[:2]
            target = np.asarray(e.get("dock_target") or [np.nan, np.nan], float)[:2]
            dock.append({"step": e["step"], "label": e.get("label"), "believed_base": believed.round(3).tolist(),
                         "true_radio_root_base": true_rel.round(3).tolist(), "dock_target": target.round(3).tolist(),
                         "perception_error_m (root vs believed point)": round(float(np.linalg.norm(true_rel - believed)), 3)})
    out["docking_audit"] = dock

    errs, far, n_det = [], 0, 0
    for e in trace:
        if e.get("event") != "perception" or e.get("step") not in rows:
            continue
        r = rows[e["step"]]
        truth = to_base(r["robot_base"], r["objects"][RADIO])
        for d in e.get("detections", []):
            if d.get("label") != "radio":
                continue
            n_det += 1
            err = float(np.linalg.norm(truth - np.asarray(d["xyz_base"][:2], float)))
            errs.append(err)
            far += err > 0.5
    out["perception_audit"] = {"radio_detections": n_det, "far_or_false(>0.5 m)": int(far),
                               "median_xy_error_vs_root_m": round(float(np.median(errs)), 3) if errs else None,
                               "note": "detector reports the grasp point (handle, ~2-3 cm from the root in xy); the error "
                                       "here is against the radio ROOT, so ~0.03 m is the expected floor"}
    (run / "task0_audit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("episodes", "physical", "predicate_agreement", "perception_audit")}, indent=1))
    for t in out["timeline"]:
        print(f"  [{t['end_step']}] {'OK ' if t['success'] else 'FAIL'} {t['key']:40s} {t['verifier_kind']} {t['reason'] or ''}")
    for a in dock:
        print(f"  dock@{a['step']}: believed {a['believed_base']} true root {a['true_radio_root_base']} target {a['dock_target']}")
    for a in audit:
        print(f"  audit {a['key']} @{a['step']}: policy={a['policy_verdict']} truth={a['ground_truth']} ({a['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
