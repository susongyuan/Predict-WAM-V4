#!/usr/bin/env python3
"""Offline K-sample admission of a Task-1 (picking_up_trash) WAM manipulation checkpoint on FIXED validation anchors.

Adapted from scripts/check_openlid_replay.py (same in-segment anchor rule, multi-view frame stacking, intent =
chunk-end target minus current joint position so a copy-state hold scores zero, per-anchor seed 1000*episode+anchor),
extended to several segments per episode, to every Task-1 executor contract, to gripper grading, and to a gate that is
READ from a pre-registered gate file (written before any result by --write-gate; this script never changes it).

Anchors (held-out validation demos only; P2_TASK_ID must be 1):
  fixed : every validation segment whose key equals the contract, 10 in-segment fractions
          (a = even(lo + f*(hi-lo-33)), lo <= a, a+32 < hi) -> upper-body intent / MAE and gripper steady-state sign;
  event : every demonstrated gripper sign flip at frame f inside such a segment, a = even(f-16), so the flip sits in the
          middle of the 32-step chunk -> grasp (close) / release (open) timing.
Each anchor feeds the real demo observation (head + left/right wrist cached frames in the checkpoint's view order, 61D
proprio) and the train-only contract vector through `wrapper.sample_actions` with K flow samples in ONE batch; the
K-sample mean is the deployed `--samples K` estimate. The checkpoint is built with harness.wam_behavior.build_models
(the model path of policy_server.load_policy_model: strict load, eval), without importing the policy server.

Offline prediction diagnostic only: not closed loop, not physical skill success, not task success.

  P2_TASK_ID=1 python scripts/check_task1_manip_admission.py --write-gate DIR/gate.json        # CPU, before results
  P2_TASK_ID=1 python scripts/check_task1_manip_admission.py --self-test                       # CPU, synthetic predictors
  P2_TASK_ID=1 python scripts/check_task1_manip_admission.py --gate DIR/gate.json --checkpoint CKPT --out DIR/best
"""
import argparse
import hashlib
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
from harness.vtt_monitor import precondition_key  # noqa: E402
from harness.wam_behavior import (ACTION_GROUPS, FRAMES, LOWDIM, STATE_SLICES, VTT_DIR, SkillTable,  # noqa: E402
                                  hold_baseline, normalize_image)

CONTRACTS = ["pick up from(trash_can,floors)", "pick up from(can_of_soda,floors)",
             "place in(can_of_soda,trash_can)", "place on(trash_can,floors)"]
FRACTIONS = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EVENT_OFFSET = 16
UPPER = ("trunk", "left_arm", "right_arm")
POSITION_PAIRS = {"trunk": ((3, 7), STATE_SLICES["trunk_qpos"]),
                  "left_arm": ((7, 14), STATE_SLICES["arm_left_qpos"]),
                  "right_arm": ((15, 22), STATE_SLICES["arm_right_qpos"])}
GRIPPER_COLUMNS = {"left": ACTION_GROUPS["left_gripper"][0], "right": ACTION_GROUPS["right_gripper"][0]}

GATE_THRESHOLDS = {
    "primary_upper_body": {
        "groups": list(UPPER),
        "moving_anchor_min_intent_norm_rad": 0.05,
        "min_moving_anchors_to_grade_intent": 20,
        "intent_cos_mean_gt": 0.0,
        "intent_frac_positive_min": 0.6,
        "intent_norm_ratio_min": 0.5,
        "mae_rule": "mean over ALL fixed anchors of K-mean MAE minus mean copy-hold MAE <= K-sample noise floor, "
                    "floor = mean over anchors of mean_k|S_k - K-mean| (group dims, 32 steps) / sqrt(K)",
        "pass_if": "every group: MAE rule AND (intent criteria on its moving anchors, or fewer than "
                   "min_moving_anchors_to_grade_intent moving anchors = static in this contract, graded by the MAE rule only)",
    },
    "grippers_graded_separately": {
        "sign": "+1 open / -1 closed demo command; prediction sign of the K-mean",
        "steady_sign_agreement_min": 0.9,
        "steady_rule": "per hand, mean over fixed anchors of the fraction of the 32 chunk steps whose sign matches the demo",
        "event_both_endpoints_min": 0.6,
        "event_rule": "per hand with >= min_event_anchors single-flip event anchors: fraction whose K-mean sign matches the "
                      "demo at chunk step 0 AND at step 31, i.e. the predicted switch lands inside the +-16-step window",
        "min_event_anchors": 5,
        "pass_if": "every hand: steady rule AND (event rule, or too few event anchors)",
    },
    "verdict": "per contract: primary_pass (upper body), gripper_pass (graded separately), admitted = primary_pass AND "
               "gripper_pass. Offline candidate admission only; closed-loop physical checks still required.",
}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else float("nan")


def file_sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def even_up(x):
    return (int(x) + 1) // 2 * 2


def anchor_sets(keys=CONTRACTS, split="validation", max_episodes=20):
    """Deterministic fixed + event anchors per contract from the annotations and cached demo actions only."""
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    episodes = [ep for ep in sorted(table.episodes) if index["episodes"][str(ep)]["split"] == split][:max_episodes]
    sets = {}
    for key in keys:
        fixed, events, skipped, segments = [], [], [], 0
        for ep in episodes:
            action = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
            for i, seg in enumerate(table.episodes[ep]["segments"]):
                if precondition_key(seg["skill"], seg["objects"]) != key:
                    continue
                segments += 1
                lo, hi = int(seg["start"]), int(seg["end"])
                for frac in FRACTIONS:
                    a = even_up(lo + frac * (hi - lo - 33))
                    rec = {"episode": ep, "segment": i, "bounds": [lo, hi], "fraction": frac, "anchor": a}
                    (fixed if lo <= a and a + 32 < hi else skipped).append(rec if lo <= a and a + 32 < hi else
                                                                           {**rec, "reason": "chunk outside segment"})
                for hand, col in GRIPPER_COLUMNS.items():
                    sign = np.sign(np.asarray(action[lo:hi, col], np.float32))
                    for rel in np.nonzero(sign[1:] != sign[:-1])[0] + 1:
                        flip = lo + int(rel)
                        a = even_up(flip - EVENT_OFFSET)
                        rec = {"episode": ep, "segment": i, "bounds": [lo, hi], "hand": hand, "flip_frame": flip,
                               "direction": "close" if sign[rel] < 0 else "open", "anchor": a}
                        if lo <= a and a + 32 < hi:
                            events.append(rec)
                        else:
                            skipped.append({**rec, "reason": "event chunk outside segment"})
        sets[key] = {"segments": segments, "fixed": fixed, "event": events, "skipped": skipped}
    return episodes, sets


def anchor_digest(episodes, sets):
    canon = {"episodes": episodes, "contracts": {k: {"fixed": v["fixed"], "event": v["event"]} for k, v in sets.items()}}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def anchor_metrics(S, demo, state):
    """S: (K, 32, 23) samples, demo: (32, 23), state: (61,) current frame. Returns (row, K-mean)."""
    M = S.mean(0)
    hold = hold_baseline(torch.from_numpy(np.array(state, np.float32))[None])[0].numpy()
    row = {"groups": {}, "grippers": {}}
    for name, (lo, hi) in ACTION_GROUPS.items():
        row["groups"][name] = {"mae_pred": float(np.abs(M[:, lo:hi] - demo[:, lo:hi]).mean()),
                               "mae_hold": float(np.abs(hold[:, lo:hi] - demo[:, lo:hi]).mean()),
                               "mae_k1": float(np.abs(S[0, :, lo:hi] - demo[:, lo:hi]).mean()),
                               "noise_mae": float(np.abs(S[:, :, lo:hi] - M[None, :, lo:hi]).mean())}
    for name, ((alo, ahi), (slo, shi)) in POSITION_PAIRS.items():
        q = np.asarray(state[slo:shi], np.float64)
        d_demo, d_pred, d_k1 = demo[-1, alo:ahi] - q, M[-1, alo:ahi] - q, S[0, -1, alo:ahi] - q
        row["groups"][name].update({"intent_norm_demo": float(np.linalg.norm(d_demo)),
                                    "intent_norm_pred": float(np.linalg.norm(d_pred)), "intent_cos": cosine(d_demo, d_pred),
                                    "intent_norm_k1": float(np.linalg.norm(d_k1)), "intent_cos_k1": cosine(d_demo, d_k1)})
    for hand, col in GRIPPER_COLUMNS.items():
        d = np.sign(demo[:, col])
        row["grippers"][hand] = {"sign_agree": float((np.sign(M[:, col]) == d).mean()),
                                 "sign_agree_hold": float((np.sign(hold[:, col]) == d).mean()),
                                 "sign_agree_k1": float((np.sign(S[0, :, col]) == d).mean()),
                                 "demo_sign": [int(d[0]), int(d[-1])]}
    return row, M


def event_metrics(M, demo, hand):
    col = GRIPPER_COLUMNS[hand]
    d, p = np.sign(demo[:, col]), np.sign(M[:, col])
    s0, s1 = d[0], d[-1]
    out = {"double_flip": bool(s0 == s1), "start_ok": bool(p[0] == s0), "end_ok": bool(p[-1] == s1),
           "both_endpoints": bool(p[0] == s0 and p[-1] == s1), "t_demo": int(np.argmax(d != s0))}
    settled = None
    if p[-1] == s1:
        bad = np.nonzero(p != s1)[0]
        settled = int(bad[-1] + 1) if len(bad) else 0
    out["t_pred_settled"] = settled
    out["timing_error_steps"] = (settled - out["t_demo"]) if settled is not None and out["start_ok"] and not out["double_flip"] else None
    return out


def _nanmean(values):
    arr = np.asarray(values, np.float64)
    return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def evaluate_gate(gate, fixed_rows, event_rows, k):
    """Pure function of the pre-registered thresholds and per-anchor rows of ONE contract."""
    pu, gg = gate["primary_upper_body"], gate["grippers_graded_separately"]
    res = {"fixed_anchors": len(fixed_rows), "event_anchors": len(event_rows), "groups": {}, "grippers": {}}
    for g in pu["groups"]:
        moving = [r for r in fixed_rows if r["groups"][g]["intent_norm_demo"] >= pu["moving_anchor_min_intent_norm_rad"]]
        cos = np.array([r["groups"][g]["intent_cos"] for r in moving], np.float64)
        graded = len(moving) >= pu["min_moving_anchors_to_grade_intent"]
        norm_demo = float(np.mean([r["groups"][g]["intent_norm_demo"] for r in moving])) if moving else float("nan")
        norm_pred = float(np.mean([r["groups"][g]["intent_norm_pred"] for r in moving])) if moving else float("nan")
        m = {"moving_anchors": len(moving), "moving_fraction": len(moving) / max(len(fixed_rows), 1),
             "intent_graded": graded, "intent_cos_mean": _nanmean(cos),
             "intent_frac_positive": float(np.mean(np.nan_to_num(cos, nan=-1.0) > 0)) if len(cos) else float("nan"),
             "intent_norm_demo": norm_demo, "intent_norm_pred": norm_pred,
             "intent_norm_ratio": norm_pred / max(norm_demo, 1e-12) if moving else float("nan"),
             "intent_cos_mean_k1": _nanmean([r["groups"][g]["intent_cos_k1"] for r in moving]),
             "intent_cos_mean_all_anchors": _nanmean([r["groups"][g]["intent_cos"] for r in fixed_rows]),
             "mae_pred": float(np.mean([r["groups"][g]["mae_pred"] for r in fixed_rows])),
             "mae_hold": float(np.mean([r["groups"][g]["mae_hold"] for r in fixed_rows])),
             "mae_k1": float(np.mean([r["groups"][g]["mae_k1"] for r in fixed_rows])),
             "noise_mae_single_sample": float(np.mean([r["groups"][g]["noise_mae"] for r in fixed_rows]))}
        m["noise_floor"] = m["noise_mae_single_sample"] / math.sqrt(k)
        m["mae_pred_minus_hold"] = m["mae_pred"] - m["mae_hold"]
        m["mae_ok"] = bool(m["mae_pred_minus_hold"] <= m["noise_floor"])
        m["intent_ok"] = (bool(m["intent_cos_mean"] > pu["intent_cos_mean_gt"]
                               and m["intent_frac_positive"] >= pu["intent_frac_positive_min"]
                               and m["intent_norm_ratio"] >= pu["intent_norm_ratio_min"]) if graded else None)
        m["pass"] = bool(m["mae_ok"] and m["intent_ok"] is not False)
        res["groups"][g] = m
    for name in ("base",):   # report only: the WAM base during manipulation contracts
        res.setdefault("report_only", {})[name] = {
            "mae_pred": float(np.mean([r["groups"][name]["mae_pred"] for r in fixed_rows])),
            "mae_hold": float(np.mean([r["groups"][name]["mae_hold"] for r in fixed_rows]))}
    for hand in GRIPPER_COLUMNS:
        steady = float(np.mean([r["grippers"][hand]["sign_agree"] for r in fixed_rows]))
        ev = [e for e in event_rows if e["hand"] == hand]
        single = [e for e in ev if not e["event"]["double_flip"]]
        graded = len(single) >= gg["min_event_anchors"]
        both = float(np.mean([e["event"]["both_endpoints"] for e in single])) if single else float("nan")
        timing = [e["event"]["timing_error_steps"] for e in single if e["event"]["timing_error_steps"] is not None]
        h = {"steady_sign_agree": steady,
             "steady_sign_agree_hold": float(np.mean([r["grippers"][hand]["sign_agree_hold"] for r in fixed_rows])),
             "steady_sign_agree_k1": float(np.mean([r["grippers"][hand]["sign_agree_k1"] for r in fixed_rows])),
             "steady_ok": bool(steady >= gg["steady_sign_agreement_min"]),
             "event_anchors": len(ev), "event_single_flip": len(single), "event_graded": graded,
             "event_directions": sorted({e["direction"] for e in single}),
             "event_both_endpoints": both,
             "event_end_ok": float(np.mean([e["event"]["end_ok"] for e in single])) if single else float("nan"),
             "event_start_ok": float(np.mean([e["event"]["start_ok"] for e in single])) if single else float("nan"),
             "timing_error_steps_median": float(np.median(timing)) if timing else float("nan"),
             "timing_error_steps_abs_median": float(np.median(np.abs(timing))) if timing else float("nan"),
             "event_ok": (bool(both >= gg["event_both_endpoints_min"]) if graded else None)}
        h["pass"] = bool(h["steady_ok"] and h["event_ok"] is not False)
        res["grippers"][hand] = h
    res["primary_pass"] = all(res["groups"][g]["pass"] for g in pu["groups"])
    res["gripper_pass"] = all(h["pass"] for h in res["grippers"].values())
    res["admitted"] = bool(res["primary_pass"] and res["gripper_pass"])
    return res


def view_pixels(packs, frame_index, size):
    """Same anchor frame from every view -> (V, 3, H, W), policy-server preprocessing per view (check_openlid_replay)."""
    import cv2
    out = []
    for pack in packs:
        rgb = np.asarray(pack.rgb(frame_index))[..., :3]
        if rgb.shape[0] != size or rgb.shape[1] != size:
            rgb = cv2.resize(np.ascontiguousarray(rgb), (size, size), interpolation=cv2.INTER_AREA)
        out.append(normalize_image(rgb))
    return np.stack(out)


def load_model(checkpoint, device):
    from omegaconf import OmegaConf
    from harness.wam_behavior import build_models, model_proprio_lags, model_views
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = OmegaConf.create(ck["config"])
    if int(cfg.dataset.get("task_id", 13)) != task_profile.TASK_ID:
        raise ValueError(f"checkpoint task_id {cfg.dataset.get('task_id', 13)} != P2_TASK_ID {task_profile.TASK_ID}")
    if ck.get("cond_mode", cfg.dataset.get("cond_mode")) != "contract_proto" or model_proprio_lags(cfg) != [0]:
        raise ValueError("expected a contract_proto checkpoint without proprio history")
    wrapper, _ = build_models(cfg, device, dino_dtype=torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32)
    wrapper.action_model.load_state_dict(ck["model_state_dict"], strict=True)
    wrapper.eval()
    step = int(ck.get("step", -1))
    del ck
    views = model_views(cfg)
    if list(getattr(wrapper, "views", views)) != views:
        raise ValueError("wrapper.views differs from the checkpoint config")
    return cfg, wrapper, step, views


@torch.no_grad()
def sample_k(wrapper, pixels, state, cond, k, seed, num_steps):
    """K flow samples in one batch through the deployed sampler; per-anchor generator seed."""
    gen = torch.Generator(device=wrapper.device).manual_seed(int(seed))
    px = torch.from_numpy(pixels)[None].repeat(k, 1, 1, 1, 1)
    st = torch.from_numpy(np.array(state, np.float32))[None].repeat(k, 1)
    out = wrapper.sample_actions(px, st, task_cond=cond.repeat(k, 1), num_steps=num_steps, generator=gen)
    return out[0].float().cpu().numpy()


def write_gate(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists; a pre-registered gate is never overwritten")
    episodes, sets = anchor_sets()
    targets = VTT_DIR / "contract_targets.json"
    gate = {"format": "task1-manip-admission-gate-v1",
            "registered_local_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "registered_before_results": True,
            "task_id": task_profile.TASK_ID, "task_name": task_profile.TASK_NAME, "split": "validation",
            "episodes": episodes, "contracts": CONTRACTS,
            "contract_targets": str(targets.relative_to(ROOT)), "contract_targets_sha256": file_sha256(targets),
            "anchors": {"fixed_fractions": FRACTIONS, "fixed_rule": "a = even(lo + f*(hi-lo-33)); lo <= a and a+32 < hi "
                        "(real annotation [start,end), the check_openlid_replay.py rule)",
                        "event_rule": f"every demo gripper sign flip f in [lo,hi): a = even(f-{EVENT_OFFSET}); lo <= a and a+32 < hi",
                        "counts": {k: {"segments": v["segments"], "fixed": len(v["fixed"]),
                                       "event": {h: sum(e["hand"] == h for e in v["event"]) for h in GRIPPER_COLUMNS},
                                       "skipped": len(v["skipped"])} for k, v in sets.items()},
                        "anchor_list_sha256": anchor_digest(episodes, sets)},
            "sampling": {"K": 8, "flow_steps": "checkpoint common.num_inference_steps (10)",
                         "seed_rule": "torch.Generator(device).manual_seed(1000*episode + anchor), one batched "
                                      "wrapper.sample_actions call of K per anchor", "estimate": "K-sample mean"},
            **GATE_THRESHOLDS,
            "checkpoints_to_evaluate": ["outputs/training/task1_manip_mv_20260926/checkpoint_best.pt",
                                        "outputs/training/task1_manip_mv_20260926/checkpoint_last.pt"],
            "rationale": {
                "moving_anchors": "Task-1 contracts move one arm/trunk at a time: on these validation anchors e.g. the left "
                                  "arm moves >=0.05 rad in only 20% of pick up from(trash_can) anchors and the trunk in 20% "
                                  "of place in anchors (demo-only statistics, no model output). Intent cosine on a near-"
                                  "static group is sampling noise, so intent is graded on anchors whose demo intent norm "
                                  ">= 0.05 rad (~3x the K8 intent noise); static behaviour is graded by the MAE rule.",
                "thresholds_reused": "cos_mean > 0, frac_positive >= 0.6, norm ratio >= 0.5 and the sigma/sqrt(K) noise-"
                                     "floor MAE rule are the Task-13 stage6/stage7b pre-registered criteria "
                                     "(scripts/compare_manipulation_predictions.py, progress archive 2026-09-24/25); the noise "
                                     "floor is measured on the candidate's own K samples instead of a borrowed Task-13 value.",
                "grippers": "picking success depends on grasp/release timing: event anchors put the demonstrated flip at chunk "
                            "step ~16, so both endpoints right means the predicted switch lands within +-16 steps (0.53 s)."},
            "interpretation": "Offline candidate admission of a manipulation executor; not closed loop, not task success."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate, indent=1) + "\n")
    print(json.dumps({"written": str(path), "sha256": file_sha256(path), "counts": gate["anchors"]["counts"],
                      "anchor_list_sha256": gate["anchors"]["anchor_list_sha256"]}, indent=1))


def self_test():
    """CPU only, no checkpoint: real validation anchors and demos, synthetic predictors with known verdicts."""
    episodes, sets = anchor_sets()
    assert len(episodes) == 20 and all(sets[k]["fixed"] for k in CONTRACTS)
    rng = np.random.default_rng(0)
    gate = GATE_THRESHOLDS
    verdicts = {}
    for key in CONTRACTS:
        rows = {"oracle": ([], []), "hold": ([], []), "late": ([], [])}
        for kind, anchors in (("fixed", sets[key]["fixed"]), ("event", sets[key]["event"])):
            for rec in anchors:
                ep, a = rec["episode"], rec["anchor"]
                act = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
                st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
                demo, state = np.array(act[a:a + 32], np.float32), np.array(st[a], np.float32)
                hold = hold_baseline(torch.from_numpy(state)[None])[0].numpy()
                oracle = demo[None] + rng.normal(0, 0.01, (8, 32, 23)).astype(np.float32)
                late = oracle.copy()
                for col in GRIPPER_COLUMNS.values():   # grasp/release 20 steps late: the flip leaves the window
                    late[:, :, col] = np.concatenate([np.repeat(oracle[:, :1, col], 20, axis=1), oracle[:, :-20, col]], axis=1)
                for name, S in (("oracle", oracle), ("hold", hold[None] + rng.normal(0, 0.01, (8, 32, 23)).astype(np.float32)),
                                ("late", late)):
                    row, M = anchor_metrics(S, demo, state)
                    if kind == "fixed":
                        rows[name][0].append(row)
                    else:
                        rows[name][1].append({**rec, "event": event_metrics(M, demo, rec["hand"])})
        verdicts[key] = {name: evaluate_gate(gate, f, e, 8) for name, (f, e) in rows.items()}
        v = verdicts[key]
        assert v["oracle"]["admitted"], (key, "oracle", v["oracle"])
        assert not v["hold"]["primary_pass"], (key, "hold must fail intent")
        assert all(v["hold"]["groups"][g]["mae_ok"] for g in UPPER), (key, "hold is its own MAE baseline")
        assert v["late"]["primary_pass"] and not v["late"]["gripper_pass"], (key, "late gripper must fail the event rule")
    # frame path: one anchor, three views, (V,3,224,224)
    rec = sets[CONTRACTS[1]]["fixed"][0]
    views = ["observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
             "observation.rgb.right_realsense_link_camera_0"]
    px = view_pixels([FramePack(FRAMES / v, rec["episode"]) for v in views], rec["anchor"], 224)
    assert px.shape == (3, 3, 224, 224) and px.dtype == np.float32 and not np.array_equal(px[0], px[1])
    counts = {k: (len(sets[k]["fixed"]), {h: sum(e["hand"] == h for e in sets[k]["event"]) for h in GRIPPER_COLUMNS})
              for k in CONTRACTS}
    print(json.dumps({"passed": True, "cpu_only": True, "anchor_counts_fixed_event": counts,
                      "anchor_list_sha256": anchor_digest(episodes, sets),
                      "oracle_left_arm_cos": {k: round(verdicts[k]["oracle"]["groups"]["left_arm"]["intent_cos_mean"], 3) for k in CONTRACTS},
                      "hold_frac_positive": {k: {g: round(verdicts[k]["hold"]["groups"][g]["intent_frac_positive"], 2) for g in UPPER}
                                             for k in CONTRACTS}}, indent=1))


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
    if task_profile.TASK_ID != 1:
        ap.error("run with P2_TASK_ID=1")
    if args.self_test:
        return self_test()
    if args.write_gate:
        return write_gate(args.write_gate)
    if not (args.gate and args.checkpoint and args.out):
        ap.error("--gate, --checkpoint and --out are required")
    gate = json.loads(args.gate.read_text())
    if gate.get("format") != "task1-manip-admission-gate-v1" or gate["task_id"] != task_profile.TASK_ID:
        ap.error("not a Task-1 admission gate")
    if not 0.1 <= args.duty_cycle <= 1.0:
        ap.error("--duty-cycle must be in [0.1, 1]")
    if not args.smoke and args.samples != gate["sampling"]["K"]:
        ap.error(f"--samples must equal the registered K={gate['sampling']['K']}")
    args.out.mkdir(parents=True, exist_ok=False)
    episodes, sets = anchor_sets(gate["contracts"])
    digest = anchor_digest(episodes, sets)
    if digest != gate["anchors"]["anchor_list_sha256"] or episodes != gate["episodes"]:
        raise ValueError("Recomputed anchors differ from the pre-registered anchor list")
    targets_path = ROOT / gate["contract_targets"]
    targets_sha = file_sha256(targets_path)
    if targets_sha != gate["contract_targets_sha256"]:
        raise ValueError("contract targets changed since the gate was registered")
    started = time.monotonic()
    cfg, wrapper, step, views = load_model(args.checkpoint, args.device)
    if cfg.dataset.get("contract_targets_sha256") != targets_sha:
        raise ValueError("checkpoint was trained with different contract targets")
    num_steps = int(cfg.common.num_inference_steps)
    size = int(cfg.dataset.image_size[0])
    targets = json.loads(targets_path.read_text())
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": step, "checkpoint_sha256": file_sha256(args.checkpoint),
               "gate": str(args.gate), "gate_sha256": file_sha256(args.gate), "anchor_list_sha256": digest,
               "views": views, "samples_K": args.samples, "flow_steps": num_steps, "smoke": bool(args.smoke),
               "duty_cycle": args.duty_cycle,
               "admission_valid": not args.smoke, "closed_loop": False, "task_success_evaluated": False,
               "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in
                                 (Path(__file__).resolve(), ROOT / "harness/wam_behavior.py", LOWDIM / "index.json", targets_path)},
               "contracts": {}}
    all_rows = {}
    for key in gate["contracts"]:
        vector = torch.tensor(targets["contracts"][key]["vector"], dtype=torch.float32)[None]
        fixed_rows, event_rows = [], []
        for kind in ("fixed", "event"):
            anchors = sets[key][kind][:args.smoke] if args.smoke else sets[key][kind]
            packs = {}
            for rec in anchors:
                ep, a = rec["episode"], rec["anchor"]
                if ep not in packs:
                    packs = {ep: [FramePack(FRAMES / v, ep) for v in views]}
                act = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
                st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
                demo, state = np.array(act[a:a + 32], np.float32), np.array(st[a], np.float32)
                busy = time.monotonic()
                S = sample_k(wrapper, view_pixels(packs[ep], a, size), state, vector, args.samples, 1000 * ep + a, num_steps)
                if args.duty_cycle < 1.0:
                    time.sleep((time.monotonic() - busy) * (1.0 - args.duty_cycle) / args.duty_cycle)
                if not np.isfinite(S).all():
                    raise ValueError(f"non-finite samples at {key} episode {ep} anchor {a}")
                row, M = anchor_metrics(S, demo, state)
                if kind == "fixed":
                    fixed_rows.append({**rec, **row})
                else:
                    event_rows.append({**rec, **row, "event": event_metrics(M, demo, rec["hand"])})
        verdict = evaluate_gate(gate, fixed_rows, event_rows, args.samples)
        verdict["by_fraction"] = {str(f): {g: {"mae_pred": float(np.mean([r["groups"][g]["mae_pred"] for r in rs])),
                                                "mae_hold": float(np.mean([r["groups"][g]["mae_hold"] for r in rs])),
                                                "intent_cos": _nanmean([r["groups"][g]["intent_cos"] for r in rs])}
                                            for g in UPPER}
                                  for f in FRACTIONS for rs in [[r for r in fixed_rows if r["fraction"] == f]] if rs}
        summary["contracts"][key] = verdict
        all_rows[key] = {"fixed": fixed_rows, "event": event_rows}
        g = verdict["groups"]
        print(f"[{key}] fixed {len(fixed_rows)} event {len(event_rows)} | primary={'PASS' if verdict['primary_pass'] else 'FAIL'} "
              f"gripper={'PASS' if verdict['gripper_pass'] else 'FAIL'} | " + " ".join(
                  f"{n}: cos {g[n]['intent_cos_mean']:.2f} fp {g[n]['intent_frac_positive']:.2f} ratio {g[n]['intent_norm_ratio']:.2f} "
                  f"dMAE {g[n]['mae_pred_minus_hold']:+.4f}/{g[n]['noise_floor']:.4f}" for n in UPPER) + " | " + " ".join(
                  f"{h}: steady {v['steady_sign_agree']:.3f} ev {v['event_both_endpoints']:.2f} (n={v['event_single_flip']})"
                  for h, v in verdict["grippers"].items()), flush=True)
    summary["all_contracts_primary_pass"] = all(v["primary_pass"] for v in summary["contracts"].values())
    summary["all_contracts_gripper_pass"] = all(v["gripper_pass"] for v in summary["contracts"].values())
    summary["all_contracts_admitted"] = all(v["admitted"] for v in summary["contracts"].values())
    summary["seconds"] = round(time.monotonic() - started, 1)
    if torch.device(args.device).type == "cuda":
        summary["torch_peak_mib"] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    (args.out / "rows.json").write_text(json.dumps(all_rows) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"out": str(args.out), "checkpoint_step": step, "seconds": summary["seconds"],
                      "admitted": {k: v["admitted"] for k, v in summary["contracts"].items()},
                      "primary": {k: v["primary_pass"] for k, v in summary["contracts"].items()},
                      "gripper": {k: v["gripper_pass"] for k, v in summary["contracts"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
