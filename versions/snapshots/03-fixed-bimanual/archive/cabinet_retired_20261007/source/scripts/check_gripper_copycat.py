#!/usr/bin/env python3
"""Does the Task-1 WAM gripper command follow the gripper OPENING in the proprio input rather than the wrist/head images?

Offline counterfactual on the pre-registered admission event anchors (demonstrated gripper sign flips at frame f,
held-out validation demos; `check_task1_manip_admission.anchor_sets`). For each event, two real observations:
  pre  = frame a  = even(f-16)  (command not yet flipped; the admission event anchor, same seed 1000*ep+a)
  post = frame a2 = even(f+16)  (command flipped 16 steps ago; must lie inside the segment)
and four K-sample predictions of the acting hand's gripper command:
  A  pre images  + pre proprio                       (reproduces the admission row)
  B  pre images  + pre proprio, gripper qpos := post (only the finger opening says "already switched")
  C  post images + post proprio
  D  post images + post proprio, gripper qpos := pre (only the finger opening says "not switched yet")
"follows proprio" = K-mean sign at chunk step 0 equals the sign implied by the substituted opening (B: post, D: pre).
Registered rule (progress.md 2026-09-26 11:00, before results): pooled follow fraction over B and D >= 0.5 -> the
gripper command is driven mainly by the finger-opening input (copycat).

Offline diagnostic only: not closed loop, not physical grasp success.

  P2_TASK_ID=1 python scripts/check_gripper_copycat.py --checkpoint CKPT --out DIR
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from harness import task_profile  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402
from harness.wam_behavior import FRAMES, LOWDIM, STATE_SLICES, VTT_DIR  # noqa: E402
from check_task1_manip_admission import (GRIPPER_COLUMNS, anchor_sets, even_up, file_sha256, load_model,  # noqa: E402
                                         sample_k, view_pixels)

QPOS = {"left": STATE_SLICES["gripper_left_qpos"], "right": STATE_SLICES["gripper_right_qpos"]}
RULE_FOLLOW_MIN = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--duty-cycle", type=float, default=0.8)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    if task_profile.TASK_ID != 1:
        ap.error("run with P2_TASK_ID=1")
    args.out.mkdir(parents=True, exist_ok=False)
    targets_path = VTT_DIR / "contract_targets.json"
    targets = json.loads(targets_path.read_text())
    episodes, sets = anchor_sets()
    cfg, wrapper, step, views = load_model(args.checkpoint, args.device)
    if cfg.dataset.get("contract_targets_sha256") != file_sha256(targets_path):
        raise ValueError("checkpoint was trained with different contract targets")
    num_steps, size = int(cfg.common.num_inference_steps), int(cfg.dataset.image_size[0])
    started = time.monotonic()
    rows, skipped = [], []
    for key, s in sets.items():
        cond = torch.tensor(targets["contracts"][key]["vector"], dtype=torch.float32)[None]
        events = s["event"][:args.smoke] if args.smoke else s["event"]
        for rec in events:
            ep, a, f, hand = rec["episode"], rec["anchor"], rec["flip_frame"], rec["hand"]
            lo, hi = rec["bounds"]
            a2 = even_up(f + 16)
            if a2 >= hi:
                skipped.append({**rec, "reason": "post frame outside segment"})
                continue
            act = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
            st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
            col, (qlo, qhi) = GRIPPER_COLUMNS[hand], QPOS[hand]
            pre_sign, post_sign = int(np.sign(act[a, col])), int(np.sign(act[a2, col]))
            if pre_sign == post_sign or pre_sign == 0:
                skipped.append({**rec, "reason": "no single flip between pre and post frames"})
                continue
            packs = [FramePack(FRAMES / v, ep) for v in views]
            s_pre, s_post = np.array(st[a], np.float32), np.array(st[a2], np.float32)
            px_pre, px_post = view_pixels(packs, a, size), view_pixels(packs, a2, size)
            s_b, s_d = s_pre.copy(), s_post.copy()
            s_b[qlo:qhi], s_d[qlo:qhi] = s_post[qlo:qhi], s_pre[qlo:qhi]
            out = {}
            for tag, px, state, seed in (("A", px_pre, s_pre, 1000 * ep + a), ("B", px_pre, s_b, 1000 * ep + a),
                                         ("C", px_post, s_post, 1000 * ep + a2), ("D", px_post, s_d, 1000 * ep + a2)):
                busy = time.monotonic()
                S = sample_k(wrapper, px, state, cond, args.samples, seed, num_steps)
                if args.duty_cycle < 1.0:
                    time.sleep((time.monotonic() - busy) * (1.0 - args.duty_cycle) / args.duty_cycle)
                M = S[:, :, col].mean(0)
                out[tag] = {"kmean_step0": float(M[0]), "sign_step0": int(np.sign(M[0])),
                            "frac_post_sign": float((np.sign(M) == post_sign).mean()),
                            "samples_post_at_step0": int((np.sign(S[:, 0, col]) == post_sign).sum())}
            # demo lag: frames from the command flip until the finger opening covers half of its pre->post change
            q = np.asarray(st[f:min(f + 40, len(st)), qlo:qhi], np.float64).mean(1)
            q0, q1 = float(np.asarray(st[a, qlo:qhi]).mean()), float(np.asarray(st[a2, qlo:qhi]).mean())
            half = np.nonzero(np.abs(q - q0) >= 0.5 * abs(q1 - q0))[0] if abs(q1 - q0) > 1e-4 else []
            rows.append({**rec, "post_anchor": a2, "pre_sign": pre_sign, "post_sign": post_sign,
                         "qpos_pre": q0, "qpos_post": q1, "demo_lag_half_frames": int(half[0]) if len(half) else None,
                         "pred": out,
                         "follow_B": out["B"]["sign_step0"] == post_sign, "follow_D": out["D"]["sign_step0"] == pre_sign,
                         "A_start_ok": out["A"]["sign_step0"] == pre_sign, "C_steady_ok": out["C"]["sign_step0"] == post_sign})

    def frac(rs, k):
        return float(np.mean([r[k] for r in rs])) if rs else float("nan")

    groups = {}
    for name, sel in (("all", lambda r: True), ("close", lambda r: r["direction"] == "close"),
                      ("open", lambda r: r["direction"] == "open")):
        rs = [r for r in rows if sel(r)]
        lags = [r["demo_lag_half_frames"] for r in rs if r["demo_lag_half_frames"] is not None]
        groups[name] = {"n": len(rs), "A_start_ok": frac(rs, "A_start_ok"), "C_steady_ok": frac(rs, "C_steady_ok"),
                        "follow_B": frac(rs, "follow_B"), "follow_D": frac(rs, "follow_D"),
                        "follow_pooled": float(np.mean([r["follow_B"] for r in rs] + [r["follow_D"] for r in rs])) if rs else float("nan"),
                        "A_frac_post": float(np.mean([r["pred"]["A"]["frac_post_sign"] for r in rs])) if rs else float("nan"),
                        "B_frac_post": float(np.mean([r["pred"]["B"]["frac_post_sign"] for r in rs])) if rs else float("nan"),
                        "C_frac_post": float(np.mean([r["pred"]["C"]["frac_post_sign"] for r in rs])) if rs else float("nan"),
                        "D_frac_post": float(np.mean([r["pred"]["D"]["frac_post_sign"] for r in rs])) if rs else float("nan"),
                        "demo_lag_half_frames_median": float(np.median(lags)) if lags else None}
    pooled = groups["all"]["follow_pooled"]
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": step, "checkpoint_sha256": file_sha256(args.checkpoint),
               "samples_K": args.samples, "smoke": bool(args.smoke), "views": views, "events_evaluated": len(rows),
               "skipped": skipped, "groups": groups, "rule": f"follow_pooled >= {RULE_FOLLOW_MIN} -> copycat",
               "copycat": bool(pooled >= RULE_FOLLOW_MIN), "seconds": round(time.monotonic() - started, 1),
               "interpretation": "offline counterfactual diagnostic; not closed loop, not grasp success"}
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("checkpoint_step", "events_evaluated", "groups", "copycat", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
