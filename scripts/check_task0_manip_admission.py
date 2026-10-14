#!/usr/bin/env python3
"""Offline K-sample admission of a Task-0 (turning_on_radio) WAM manipulation checkpoint on FIXED validation anchors.

Same rule family, thresholds, anchor rules, sampler and seed rule as the Task-1 gate: this script IMPORTS
scripts/check_task1_manip_admission.py (anchor_sets / anchor_metrics / event_metrics / evaluate_gate / load_model /
sample_k / view_pixels are reused unchanged; that file is not modified) and only supplies the Task-0 contract keys,
the Task-0 gate file format and ONE press-specific criterion:

  press_reach (press(radio) only; graded, pre-registered). In the TRAIN demos (outputs/validation/
  task0_demo_geometry_20260927) the RIGHT hand holds the radio and the LEFT hand forms a closed fist and moves it onto
  the radio's toggle button; the evaluator flips toggled_on when a finger link touches the radio inside the
  `togglebutton` marker for 5 steps. The physically relevant quantity is therefore the left fist position RELATIVE TO
  THE HOLDING (right) HAND, r(q) = R_right(q)^T (p_left(q) - p_right(q)), from URDF FK of trunk+arm joint positions
  (harness/r1pro_kinematics.py, eef offsets fitted on TRAIN). Three parts:
    approach : fixed press anchors of standard segments whose chunk ends at or before the toggle frame
               (a + 31 <= t_toggle); Cartesian intent d = r(cmd at chunk step 31) - r(current state), graded on anchors
               with |d_demo| >= 0.03 m with the Task-1 intent thresholds (cos mean > 0, frac positive >= 0.6,
               norm ratio >= 0.5; >= 20 moving anchors, else not certifiable = FAIL);
    arrival  : toggle anchors a = even(t_toggle - 16) (same placement as a gripper event), error of r(K-mean cmd) vs
               r(demo cmd) at the toggle step; Task-1 MAE-rule analogue: mean e_pred - mean e_hold <= noise floor
               = mean_anchor mean_k |r(S_k) - r(K-mean)| / sqrt(K); >= 5 anchors, else FAIL;
    fist     : K-mean left gripper command closed (sign -1) at the toggle step on >= 0.9 of toggle anchors
               (Task-1 steady-sign threshold).
  Why not only a gripper-switch timing rule: during pressing the fist is closed, so the switch that matters is not a
  grasp. The Task-1 gripper rules are still applied unchanged to press (the fist-forming close flip lies inside 19/20
  validation press segments, 25-182 frames after the start, so the event rule is defined), and press_reach adds the
  reach/hold-a-closed-fist-at-the-button event. admitted(press) = primary AND gripper AND press_reach.

Standard press segment (demo-only rule): at the press segment start the demo right gripper command is closed and the
left one open, and the toggle frame (first parquet next.reward > 0) lies inside the segment. Toggle frames are demo
labels only; nothing here is fed to the policy except the usual observation + contract vector.

Offline prediction diagnostic only: not closed loop, not physical skill success, not task success.

  P2_TASK_ID=0 python scripts/check_task0_manip_admission.py --write-gate DIR/gate.json        # CPU, before results
  P2_TASK_ID=0 python scripts/check_task0_manip_admission.py --self-test                       # CPU, synthetic predictors
  P2_TASK_ID=0 python scripts/check_task0_manip_admission.py --gate DIR/gate.json \
      --checkpoint outputs/training/task0_manip_v1_20260927/checkpoint_final.pt --out DIR/v1_final --duty-cycle 0.8
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import task_profile  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402
from harness.wam_behavior import FRAMES, LOWDIM, VTT_DIR, hold_baseline  # noqa: E402

_spec = importlib.util.spec_from_file_location("check_task1_manip_admission", ROOT / "scripts/check_task1_manip_admission.py")
T1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T1)

TASK_ID = 0
GATE_FORMAT = "task0-manip-admission-gate-v1"
PICK, PRESS, PLACE = "pick up from(radio,coffee_table)", "press(radio)", "place on(radio,coffee_table)"
CONTRACTS = [PICK, PRESS, PLACE]
ADMITTED_CHECKPOINTS = ["outputs/training/task0_manip_v1_20260927/checkpoint_final.pt"]
TOGGLE_OFFSET = T1.EVENT_OFFSET      # the toggle sits at chunk step ~16, like a gripper event
LEFT_GRIP_COL = T1.GRIPPER_COLUMNS["left"]
A_TRUNK, A_LARM, A_RARM = slice(3, 7), slice(7, 14), slice(15, 22)          # action: absolute joint targets
S_TRUNK, S_LARM, S_RARM = slice(53, 57), slice(3, 10), slice(28, 35)        # proprio qpos
DEMO_DATA = task_profile.DATA / "data" / task_profile.CHUNK

PRESS_REACH = {
    "contract": PRESS,
    "executing_hand": "left",
    "reference_hand": "right (holds the radio in 165/173 TRAIN demos)",
    "quantity": "r(q) = R_right(q)^T (p_left(q) - p_right(q)): left eef position in the right eef frame, URDF FK of "
                "trunk(4)+arm(7) joint values (harness/r1pro_kinematics.py, TRAIN-fitted eef offsets); predicted/demo "
                "chunks use the absolute joint-target actions, the current point uses proprio qpos",
    "standard_segment_rule": "demo right gripper command < 0 and left > 0 at the press segment start, and the toggle "
                             "frame (first parquet next.reward > 0 of the episode) inside [start, end)",
    "approach": {"anchors": "fixed press anchors of standard segments with a + 31 <= t_toggle",
                 "intent": "d = r(cmd at chunk step 31) - r(state at a); K-mean for the prediction",
                 "moving_anchor_min_intent_norm_m": 0.03, "min_moving_anchors_to_grade": 20,
                 "intent_cos_mean_gt": 0.0, "intent_frac_positive_min": 0.6, "intent_norm_ratio_min": 0.5,
                 "ungraded_means": "FAIL (reach cannot be certified)"},
    "arrival": {"anchors": f"one per standard segment: a = even(t_toggle - {TOGGLE_OFFSET}); lo <= a and a+32 < hi",
                "step": "chunk step t_toggle - a",
                "rule": "mean e_pred - mean e_hold <= noise floor; e = |r(cmd at step) - r(demo cmd at step)| (m), hold = "
                        "current qpos; floor = mean over anchors of mean_k |r(S_k) - r(K-mean)| / sqrt(K)",
                "min_anchors": 5, "too_few_means": "FAIL"},
    "fist": {"rule": "K-mean left gripper command sign == -1 (closed) at the toggle step", "fraction_min": 0.9},
    "pass_if": "approach AND arrival AND fist",
}


# ----------------------------------------------------------------------------------------------------------- anchors
def toggle_frames():
    """Episode -> first frame index with next.reward > 0 (demo label of the evaluator's toggled_on flip)."""
    import pyarrow.parquet as pq
    out = {}
    for f in sorted(DEMO_DATA.glob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "next.reward"])
        ep, fr, r = (np.asarray(t.column(c)) for c in ("episode_index", "frame_index", "next.reward"))
        for e in np.unique(ep):
            m = (ep == e) & (r > 0)
            out[int(e)] = int(fr[m].min()) if m.any() else None
    return out


def press_reach_sets(sets, toggles):
    """Standard press segments -> approach anchors (subset of the fixed anchors) and toggle-arrival anchors."""
    approach, arrival, excluded, segs = [], [], [], {}
    for rec in sets[PRESS]["fixed"]:
        segs.setdefault((rec["episode"], rec["segment"]), rec["bounds"])
    # segments without any fixed anchor still count (none expected: every press segment is >= 60 frames)
    for (ep, i), (lo, hi) in sorted(segs.items()):
        act = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
        t = toggles.get(ep)
        base = {"episode": ep, "segment": i, "bounds": [lo, hi], "toggle_frame": t}
        if not (act[lo, T1.GRIPPER_COLUMNS["right"]] < 0 and act[lo, LEFT_GRIP_COL] > 0):
            excluded.append({**base, "reason": "non-standard hands at press start (right not closed or left not open)"})
            continue
        if t is None or not lo <= t < hi:
            excluded.append({**base, "reason": "toggle frame not inside the press segment"})
            continue
        approach += [{**rec, "toggle_frame": t} for rec in sets[PRESS]["fixed"]
                     if (rec["episode"], rec["segment"]) == (ep, i) and rec["anchor"] + 31 <= t]
        a = T1.even_up(t - TOGGLE_OFFSET)
        if lo <= a and a + 32 < hi:
            arrival.append({**base, "anchor": a, "step": t - a})
        else:
            excluded.append({**base, "reason": "toggle chunk outside segment"})
    return {"approach": approach, "arrival": arrival, "excluded": excluded}


def all_anchor_sets():
    episodes, sets = T1.anchor_sets(CONTRACTS)
    return episodes, sets, press_reach_sets(sets, toggle_frames())


def reach_digest(pr):
    canon = {k: [{kk: r[kk] for kk in ("episode", "segment", "anchor", "toggle_frame")} for r in pr[k]]
             for k in ("approach", "arrival")}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------- metrics
_KIN = None


def kin():
    global _KIN
    if _KIN is None:
        from harness.r1pro_kinematics import R1ProKinematics
        _KIN = R1ProKinematics()
    return _KIN


def rel_fist(trunk, larm, rarm):
    k = kin()
    pl, _ = k.fk("left", np.r_[trunk, larm])
    pr, Rr = k.fk("right", np.r_[trunk, rarm])
    return Rr.T @ (pl - pr)


def rel_action(a):
    return rel_fist(a[A_TRUNK], a[A_LARM], a[A_RARM])


def rel_state(s):
    return rel_fist(s[S_TRUNK], s[S_LARM], s[S_RARM])


def approach_metrics(S, M, demo, state):
    r0 = rel_state(state)
    d_demo, d_pred, d_k1 = rel_action(demo[-1]) - r0, rel_action(M[-1]) - r0, rel_action(S[0, -1]) - r0
    return {"intent_norm_demo_m": float(np.linalg.norm(d_demo)), "intent_norm_pred_m": float(np.linalg.norm(d_pred)),
            "intent_cos": T1.cosine(d_demo, d_pred), "intent_cos_k1": T1.cosine(d_demo, d_k1),
            "intent_norm_k1_m": float(np.linalg.norm(d_k1))}


def arrival_metrics(S, M, demo, state, step):
    target = rel_action(demo[step])
    rm = rel_action(M[step])
    return {"e_pred_m": float(np.linalg.norm(rm - target)),
            "e_hold_m": float(np.linalg.norm(rel_state(state) - target)),
            "e_k1_m": float(np.linalg.norm(rel_action(S[0, step]) - target)),
            "noise_m": float(np.mean([np.linalg.norm(rel_action(S[k, step]) - rm) for k in range(len(S))])),
            "fist_closed_pred": bool(np.sign(M[step, LEFT_GRIP_COL]) < 0),
            "fist_closed_demo": bool(np.sign(demo[step, LEFT_GRIP_COL]) < 0),
            "fist_closed_hold": bool(np.sign(hold_baseline(torch.from_numpy(np.array(state, np.float32))[None])[0]
                                             .numpy()[step, LEFT_GRIP_COL]) < 0)}


def evaluate_press_reach(g, approach_rows, arrival_rows, k):
    ap, ar, fi = g["approach"], g["arrival"], g["fist"]
    moving = [r for r in approach_rows if r["reach"]["intent_norm_demo_m"] >= ap["moving_anchor_min_intent_norm_m"]]
    cos = np.array([r["reach"]["intent_cos"] for r in moving], np.float64)
    nd = float(np.mean([r["reach"]["intent_norm_demo_m"] for r in moving])) if moving else float("nan")
    npred = float(np.mean([r["reach"]["intent_norm_pred_m"] for r in moving])) if moving else float("nan")
    a = {"anchors": len(approach_rows), "moving_anchors": len(moving),
         "graded": len(moving) >= ap["min_moving_anchors_to_grade"], "intent_cos_mean": T1._nanmean(cos),
         "intent_frac_positive": float(np.mean(np.nan_to_num(cos, nan=-1.0) > 0)) if len(cos) else float("nan"),
         "intent_norm_demo_m": nd, "intent_norm_pred_m": npred,
         "intent_norm_ratio": npred / max(nd, 1e-12) if moving else float("nan"),
         "intent_cos_mean_k1": T1._nanmean([r["reach"]["intent_cos_k1"] for r in moving])}
    a["ok"] = bool(a["graded"] and a["intent_cos_mean"] > ap["intent_cos_mean_gt"]
                   and a["intent_frac_positive"] >= ap["intent_frac_positive_min"]
                   and a["intent_norm_ratio"] >= ap["intent_norm_ratio_min"])
    n = len(arrival_rows)
    col = lambda key: [r["arrival"][key] for r in arrival_rows]  # noqa: E731
    b = {"anchors": n, "graded": n >= ar["min_anchors"]}
    if n:
        b.update({"e_pred_mean_m": float(np.mean(col("e_pred_m"))), "e_hold_mean_m": float(np.mean(col("e_hold_m"))),
                  "e_k1_mean_m": float(np.mean(col("e_k1_m"))), "e_pred_median_m": float(np.median(col("e_pred_m"))),
                  "e_hold_median_m": float(np.median(col("e_hold_m"))),
                  "frac_e_pred_below_0.05m_report_only": float(np.mean(np.array(col("e_pred_m")) < 0.05)),
                  "noise_single_sample_m": float(np.mean(col("noise_m")))})
        b["noise_floor_m"] = b["noise_single_sample_m"] / math.sqrt(k)
        b["e_pred_minus_hold_m"] = b["e_pred_mean_m"] - b["e_hold_mean_m"]
        b["ok"] = bool(b["graded"] and b["e_pred_minus_hold_m"] <= b["noise_floor_m"])
    else:
        b["ok"] = False
    c = {"anchors": n, "fist_closed_frac": float(np.mean(col("fist_closed_pred"))) if n else float("nan"),
         "fist_closed_frac_demo": float(np.mean(col("fist_closed_demo"))) if n else float("nan"),
         "fist_closed_frac_hold": float(np.mean(col("fist_closed_hold"))) if n else float("nan")}
    c["ok"] = bool(n and c["fist_closed_frac"] >= fi["fraction_min"])
    return {"approach": a, "arrival": b, "fist": c, "pass": bool(a["ok"] and b["ok"] and c["ok"])}


def evaluate_contract(gate, key, fixed_rows, event_rows, k, approach_rows=(), arrival_rows=()):
    res = T1.evaluate_gate(gate, fixed_rows, event_rows, k)
    if key == PRESS:
        res["press_reach"] = evaluate_press_reach(gate["press_reach"], list(approach_rows), list(arrival_rows), k)
        res["admitted"] = bool(res["admitted"] and res["press_reach"]["pass"])
    return res


# -------------------------------------------------------------------------------------------------------------- gate
def write_gate(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists; a pre-registered gate is never overwritten")
    episodes, sets, pr = all_anchor_sets()
    targets = VTT_DIR / "contract_targets.json"
    tj = json.loads(targets.read_text())
    missing = [c for c in CONTRACTS if c not in tj["contracts"]]
    if missing:
        raise SystemExit(f"contract keys missing from {targets}: {missing}")
    gate = {"format": GATE_FORMAT,
            "registered_local_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "registered_before_results": True,
            "task_id": task_profile.TASK_ID, "task_name": task_profile.TASK_NAME, "split": "validation",
            "episodes": episodes, "contracts": CONTRACTS,
            "contract_targets": str(targets.relative_to(ROOT)), "contract_targets_sha256": T1.file_sha256(targets),
            "anchors": {"fixed_fractions": T1.FRACTIONS,
                        "fixed_rule": "a = even(lo + f*(hi-lo-33)); lo <= a and a+32 < hi (Task-1 gate rule)",
                        "event_rule": f"every demo gripper sign flip f in [lo,hi): a = even(f-{T1.EVENT_OFFSET}); lo <= a "
                                      "and a+32 < hi (Task-1 gate rule)",
                        "counts": {k: {"segments": v["segments"], "fixed": len(v["fixed"]),
                                       "event": {h: sum(e["hand"] == h for e in v["event"]) for h in T1.GRIPPER_COLUMNS},
                                       "event_directions": {d: sum(e["direction"] == d for e in v["event"])
                                                            for d in ("close", "open")},
                                       "skipped": len(v["skipped"])} for k, v in sets.items()},
                        "anchor_list_sha256": T1.anchor_digest(episodes, sets),
                        "press_reach_counts": {"approach": len(pr["approach"]), "arrival": len(pr["arrival"]),
                                               "excluded_segments": pr["excluded"]},
                        "press_reach_anchor_sha256": reach_digest(pr)},
            "sampling": {"K": 8, "flow_steps": "checkpoint common.num_inference_steps (10)",
                         "seed_rule": "torch.Generator(device).manual_seed(1000*episode + anchor), one batched "
                                      "wrapper.sample_actions call of K per anchor", "estimate": "K-sample mean"},
            **T1.GATE_THRESHOLDS,
            "press_reach": PRESS_REACH,
            "verdict": "per contract: primary_pass (upper body), gripper_pass (graded separately), admitted = primary_pass "
                       "AND gripper_pass; for press(radio) additionally AND press_reach.pass. Offline candidate admission "
                       "only; closed-loop physical checks still required.",
            "first_closed_loop_requires": "all three contracts admitted (they are all in configs/task000_fixed_program.json)",
            "checkpoints_to_evaluate": ADMITTED_CHECKPOINTS,
            "checkpoint_rule": "only checkpoint_final.pt (training step 4000) is admitted; the trainer's MAE-picked "
                               "checkpoint_best.pt must not be used (MAE rewards shrinking the motion, Task-13 stage8 / "
                               "Task-1 v2 lesson); the script refuses any other checkpoint unless --smoke",
            "rationale": {
                "rule_family": "Primary upper-body, gripper steady/event rules, K=8, anchors and seed rule are copied "
                               "unchanged from outputs/validation/task1_manip_admission_20260926/gate.json (05:33 "
                               "2026-09-26); contract keys match configs/lilawam_task000_multiview_manip_v1_train.yaml "
                               "(cond_mode contract_proto) and data/skill_vtt_task000/contract_targets.json.",
                "press_event": "The press hand's gripper does switch inside the segment (left fist forms 25-182 frames "
                               "after press start in 19/20 validation segments; demo-only count), so the Task-1 event "
                               "rule is defined and kept. But the press outcome depends on bringing the closed fist to "
                               "the button on the held radio, which no gripper-switch timing measures -> press_reach.",
                "press_reach_thresholds": "Intent thresholds are the Task-1 ones. 0.03 m moving threshold ~ 0.05 rad "
                                          "joint threshold x ~0.6 m lever arm. TRAIN demo statistics only (165 right-hold "
                                          "demos, no model output): fist offset from the right eef at the toggle varies "
                                          "8.0 cm (median distance to the median offset), hold error at a = t-16 is "
                                          "3.6 cm median, so the arrival rule is graded against hold with the "
                                          "candidate's own K-sample noise floor, like the MAE rule; the fist rule uses "
                                          "the steady-sign threshold 0.9. Validation demo-only counts: 138 approach "
                                          "anchors (130 moving), 18 arrival anchors; val eps 77, 153 excluded as "
                                          "non-standard hands.",
                "not_seen_before_registration": "No admission output of any Task-0 checkpoint existed when this gate was "
                                                "written. The trainer's own single-sample validation log (log.jsonl "
                                                "final_val) existed; thresholds were not tuned on it (all are the Task-1 "
                                                "values or TRAIN-demo derived)."},
            "interpretation": "Offline candidate admission of a manipulation executor; not closed loop, not task success."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate, indent=1) + "\n")
    print(json.dumps({"written": str(path), "sha256": T1.file_sha256(path), "counts": gate["anchors"]["counts"],
                      "press_reach_counts": {k: v for k, v in gate["anchors"]["press_reach_counts"].items()
                                             if k != "excluded_segments"},
                      "anchor_list_sha256": gate["anchors"]["anchor_list_sha256"],
                      "press_reach_anchor_sha256": gate["anchors"]["press_reach_anchor_sha256"]}, indent=1))


# --------------------------------------------------------------------------------------------------------- self-test
def self_test():
    """CPU only, no checkpoint: real validation anchors and demos, synthetic predictors with known verdicts."""
    episodes, sets, pr = all_anchor_sets()
    assert len(episodes) == 20 and all(sets[k]["fixed"] for k in CONTRACTS), "20 validation episodes, all contracts"
    assert len(pr["approach"]) >= 20 and len(pr["arrival"]) >= 5, pr["excluded"]
    rng = np.random.default_rng(0)
    gate = {**T1.GATE_THRESHOLDS, "press_reach": PRESS_REACH}
    approach_keys = {(r["episode"], r["anchor"]) for r in pr["approach"]}
    verdicts = {}

    def predictors(demo, state):
        hold = hold_baseline(torch.from_numpy(state)[None])[0].numpy()
        oracle = demo[None] + rng.normal(0, 0.01, (8, 32, 23)).astype(np.float32)
        late = oracle.copy()
        for col in T1.GRIPPER_COLUMNS.values():
            late[:, :, col] = np.concatenate([np.repeat(oracle[:, :1, col], 20, axis=1), oracle[:, :-20, col]], axis=1)
        openfist = oracle.copy()
        openfist[:, :, LEFT_GRIP_COL] = np.abs(openfist[:, :, LEFT_GRIP_COL]) + 0.5
        return {"oracle": oracle, "hold": hold[None] + rng.normal(0, 0.01, (8, 32, 23)).astype(np.float32),
                "late": late, "openfist": openfist}

    for key in CONTRACTS:
        rows = {n: {"fixed": [], "event": [], "approach": [], "arrival": []} for n in ("oracle", "hold", "late", "openfist")}
        todo = [("fixed", r) for r in sets[key]["fixed"]] + [("event", r) for r in sets[key]["event"]]
        if key == PRESS:
            todo += [("arrival", r) for r in pr["arrival"]]
        for kind, rec in todo:
            ep, a = rec["episode"], rec["anchor"]
            act = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
            st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
            demo, state = np.array(act[a:a + 32], np.float32), np.array(st[a], np.float32)
            for name, S in predictors(demo, state).items():
                row, M = T1.anchor_metrics(S, demo, state)
                if kind == "fixed":
                    rows[name]["fixed"].append(row)
                    if key == PRESS and (ep, a) in approach_keys:
                        rows[name]["approach"].append({"reach": approach_metrics(S, M, demo, state)})
                elif kind == "event":
                    rows[name]["event"].append({**rec, "event": T1.event_metrics(M, demo, rec["hand"])})
                else:
                    rows[name]["arrival"].append({"arrival": arrival_metrics(S, M, demo, state, rec["step"])})
        verdicts[key] = {n: evaluate_contract(gate, key, r["fixed"], r["event"], 8, r["approach"], r["arrival"])
                         for n, r in rows.items()}
        v = verdicts[key]
        assert v["oracle"]["admitted"], (key, "oracle", v["oracle"])
        assert not v["hold"]["primary_pass"], (key, "hold must fail intent")
        assert all(v["hold"]["groups"][g]["mae_ok"] for g in T1.UPPER), (key, "hold is its own MAE baseline")
        assert v["late"]["primary_pass"] and not v["late"]["gripper_pass"], (key, "late gripper must fail the event rule")
        if key == PRESS:
            assert v["oracle"]["press_reach"]["pass"], v["oracle"]["press_reach"]
            assert not v["hold"]["press_reach"]["approach"]["ok"], "hold must fail the Cartesian approach intent"
            assert v["hold"]["press_reach"]["arrival"]["ok"], "hold is its own arrival baseline"
            assert not v["openfist"]["press_reach"]["fist"]["ok"] and not v["openfist"]["admitted"], "open fist must fail"
    # the fist-close flip of the press hand is an event anchor (the Task-1 event rule is defined for press)
    assert sum(e["hand"] == "left" and e["direction"] == "close" for e in sets[PRESS]["event"]) >= 5
    rec = sets[PICK]["fixed"][0]
    views = ["observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
             "observation.rgb.right_realsense_link_camera_0"]
    px = T1.view_pixels([FramePack(FRAMES / v, rec["episode"]) for v in views], rec["anchor"], 224)
    assert px.shape == (3, 3, 224, 224) and px.dtype == np.float32 and not np.array_equal(px[0], px[1])
    pv = verdicts[PRESS]
    print(json.dumps({"passed": True, "cpu_only": True,
                      "anchor_counts_fixed_event": {k: (len(sets[k]["fixed"]), {h: sum(e["hand"] == h for e in sets[k]["event"])
                                                                                for h in T1.GRIPPER_COLUMNS}) for k in CONTRACTS},
                      "press_reach_counts": {"approach": len(pr["approach"]), "arrival": len(pr["arrival"]),
                                             "excluded": [(e["episode"], e["reason"]) for e in pr["excluded"]]},
                      "anchor_list_sha256": T1.anchor_digest(episodes, sets), "press_reach_anchor_sha256": reach_digest(pr),
                      "press_reach_synthetic": {n: {"approach_cos": round(pv[n]["press_reach"]["approach"]["intent_cos_mean"], 3)
                                                    if not math.isnan(pv[n]["press_reach"]["approach"]["intent_cos_mean"]) else None,
                                                    "arrival_e_pred_m": round(pv[n]["press_reach"]["arrival"]["e_pred_mean_m"], 4),
                                                    "arrival_e_hold_m": round(pv[n]["press_reach"]["arrival"]["e_hold_mean_m"], 4),
                                                    "fist": pv[n]["press_reach"]["fist"]["fist_closed_frac"],
                                                    "pass": pv[n]["press_reach"]["pass"]} for n in pv}}, indent=1))


# -------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-gate", type=Path, help="compute the fixed anchors and write the pre-registered gate, then exit")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gate", type=Path)
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", type=int, default=0, help="N>0: only the first N anchors per set; never an admission result")
    ap.add_argument("--duty-cycle", type=float, default=1.0,
                    help="GPU busy fraction: sleep busy*(1-d)/d after each anchor (85%% utilization ceiling); no effect on results")
    args = ap.parse_args()
    if task_profile.TASK_ID != TASK_ID:
        ap.error("run with P2_TASK_ID=0")
    if args.self_test:
        return self_test()
    if args.write_gate:
        return write_gate(args.write_gate)
    if not (args.gate and args.checkpoint and args.out):
        ap.error("--gate, --checkpoint and --out are required")
    gate = json.loads(args.gate.read_text())
    if gate.get("format") != GATE_FORMAT or gate["task_id"] != task_profile.TASK_ID:
        ap.error("not a Task-0 admission gate")
    if not 0.1 <= args.duty_cycle <= 1.0:
        ap.error("--duty-cycle must be in [0.1, 1]")
    if not args.smoke and args.samples != gate["sampling"]["K"]:
        ap.error(f"--samples must equal the registered K={gate['sampling']['K']}")
    ck_rel = str(Path(args.checkpoint).resolve().relative_to(ROOT)) if Path(args.checkpoint).resolve().is_relative_to(ROOT) \
        else str(Path(args.checkpoint).resolve())
    if not args.smoke and ck_rel not in gate["checkpoints_to_evaluate"]:
        ap.error(f"{ck_rel} is not a registered admission checkpoint {gate['checkpoints_to_evaluate']}")
    episodes, sets, pr = all_anchor_sets()
    if T1.anchor_digest(episodes, sets) != gate["anchors"]["anchor_list_sha256"] or episodes != gate["episodes"]:
        raise ValueError("Recomputed anchors differ from the pre-registered anchor list")
    if reach_digest(pr) != gate["anchors"]["press_reach_anchor_sha256"]:
        raise ValueError("Recomputed press_reach anchors differ from the pre-registered list")
    targets_path = ROOT / gate["contract_targets"]
    targets_sha = T1.file_sha256(targets_path)
    if targets_sha != gate["contract_targets_sha256"]:
        raise ValueError("contract targets changed since the gate was registered")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cfg, wrapper, step, views = T1.load_model(args.checkpoint, args.device)
    if cfg.dataset.get("contract_targets_sha256") != targets_sha:
        raise ValueError("checkpoint was trained with different contract targets")
    num_steps = int(cfg.common.num_inference_steps)
    size = int(cfg.dataset.image_size[0])
    targets = json.loads(targets_path.read_text())
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": step, "checkpoint_sha256": T1.file_sha256(args.checkpoint),
               "gate": str(args.gate), "gate_sha256": T1.file_sha256(args.gate),
               "anchor_list_sha256": gate["anchors"]["anchor_list_sha256"],
               "press_reach_anchor_sha256": gate["anchors"]["press_reach_anchor_sha256"],
               "views": views, "samples_K": args.samples, "flow_steps": num_steps, "smoke": bool(args.smoke),
               "duty_cycle": args.duty_cycle, "device": args.device,
               "admission_valid": not args.smoke, "closed_loop": False, "task_success_evaluated": False,
               "source_sha256": {str(p.relative_to(ROOT)): T1.file_sha256(p) for p in
                                 (Path(__file__).resolve(), ROOT / "scripts/check_task1_manip_admission.py",
                                  ROOT / "harness/wam_behavior.py", ROOT / "harness/r1pro_kinematics.py",
                                  ROOT / "configs/r1pro_eef_offsets.json", LOWDIM / "index.json", targets_path)},
               "contracts": {}}
    approach_keys = {(r["episode"], r["anchor"]) for r in pr["approach"]}
    all_rows = {}
    for key in gate["contracts"]:
        vector = torch.tensor(targets["contracts"][key]["vector"], dtype=torch.float32)[None]
        rows = {"fixed": [], "event": [], "approach": [], "arrival": []}
        kinds = ["fixed", "event"] + (["arrival"] if key == PRESS else [])
        for kind in kinds:
            anchors = pr["arrival"] if kind == "arrival" else sets[key][kind]
            anchors = anchors[:args.smoke] if args.smoke else anchors
            packs = {}
            for rec in anchors:
                ep, a = rec["episode"], rec["anchor"]
                if ep not in packs:
                    packs = {ep: [FramePack(FRAMES / v, ep) for v in views]}
                act = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
                st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
                demo, state = np.array(act[a:a + 32], np.float32), np.array(st[a], np.float32)
                busy = time.monotonic()
                S = T1.sample_k(wrapper, T1.view_pixels(packs[ep], a, size), state, vector, args.samples, 1000 * ep + a,
                                num_steps)
                if args.duty_cycle < 1.0:
                    time.sleep((time.monotonic() - busy) * (1.0 - args.duty_cycle) / args.duty_cycle)
                if not np.isfinite(S).all():
                    raise ValueError(f"non-finite samples at {key} episode {ep} anchor {a}")
                row, M = T1.anchor_metrics(S, demo, state)
                if kind == "fixed":
                    rows["fixed"].append({**rec, **row})
                    if key == PRESS and (ep, a) in approach_keys:
                        rows["approach"].append({**rec, "reach": approach_metrics(S, M, demo, state)})
                elif kind == "event":
                    rows["event"].append({**rec, **row, "event": T1.event_metrics(M, demo, rec["hand"])})
                else:
                    rows["arrival"].append({**rec, "arrival": arrival_metrics(S, M, demo, state, rec["step"])})
        verdict = evaluate_contract(gate, key, rows["fixed"], rows["event"], args.samples, rows["approach"], rows["arrival"])
        verdict["by_fraction"] = {str(f): {g: {"mae_pred": float(np.mean([r["groups"][g]["mae_pred"] for r in rs])),
                                                "mae_hold": float(np.mean([r["groups"][g]["mae_hold"] for r in rs])),
                                                "intent_cos": T1._nanmean([r["groups"][g]["intent_cos"] for r in rs])}
                                            for g in T1.UPPER}
                                  for f in T1.FRACTIONS for rs in [[r for r in rows["fixed"] if r["fraction"] == f]] if rs}
        summary["contracts"][key] = verdict
        all_rows[key] = rows
        g = verdict["groups"]
        line = (f"[{key}] fixed {len(rows['fixed'])} event {len(rows['event'])} | primary={'PASS' if verdict['primary_pass'] else 'FAIL'} "
                f"gripper={'PASS' if verdict['gripper_pass'] else 'FAIL'} | " + " ".join(
                    f"{n}: cos {g[n]['intent_cos_mean']:.2f} fp {g[n]['intent_frac_positive']:.2f} ratio {g[n]['intent_norm_ratio']:.2f} "
                    f"dMAE {g[n]['mae_pred_minus_hold']:+.4f}/{g[n]['noise_floor']:.4f}" for n in T1.UPPER) + " | " + " ".join(
                    f"{h}: steady {v['steady_sign_agree']:.3f} ev {v['event_both_endpoints']:.2f} (n={v['event_single_flip']})"
                    for h, v in verdict["grippers"].items()))
        if key == PRESS:
            p = verdict["press_reach"]
            line += (f" | press_reach={'PASS' if p['pass'] else 'FAIL'} approach cos {p['approach']['intent_cos_mean']:.2f} "
                     f"fp {p['approach']['intent_frac_positive']:.2f} ratio {p['approach']['intent_norm_ratio']:.2f} "
                     f"(n={p['approach']['moving_anchors']}) arrival dE {p['arrival'].get('e_pred_minus_hold_m', float('nan')):+.4f}/"
                     f"{p['arrival'].get('noise_floor_m', float('nan')):.4f} m (n={p['arrival']['anchors']}) "
                     f"fist {p['fist']['fist_closed_frac']:.2f}")
        print(line + f" | admitted={verdict['admitted']}", flush=True)
    summary["all_contracts_primary_pass"] = all(v["primary_pass"] for v in summary["contracts"].values())
    summary["all_contracts_gripper_pass"] = all(v["gripper_pass"] for v in summary["contracts"].values())
    summary["all_contracts_admitted"] = all(v["admitted"] for v in summary["contracts"].values())
    summary["seconds"] = round(time.monotonic() - started, 1)
    if torch.device(args.device).type == "cuda":
        summary["torch_peak_mib"] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    (args.out / "rows.json").write_text(json.dumps(all_rows) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"out": str(args.out), "checkpoint_step": step, "seconds": summary["seconds"], "smoke": bool(args.smoke),
                      "admitted": {k: v["admitted"] for k, v in summary["contracts"].items()},
                      "primary": {k: v["primary_pass"] for k, v in summary["contracts"].items()},
                      "gripper": {k: v["gripper_pass"] for k, v in summary["contracts"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
