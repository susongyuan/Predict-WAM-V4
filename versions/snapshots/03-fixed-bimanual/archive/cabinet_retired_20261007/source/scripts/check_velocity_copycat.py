#!/usr/bin/env python3
"""Does the WAM's base command copy the measured base velocity in the proprio state ("copycat")?

Offline diagnostic on cached validation demonstrations (first navigation segment only): the same
frame/condition/noise is sampled with (a) the true 61D state, (b) base velocity zeroed, (c) base
velocity set to a typical forward speed. If the predicted base command follows the injected velocity
instead of the image, a closed-loop stall (stop once, never restart) is expected.

This is an input-contract diagnostic on offline data; it is not task success or a system comparison.
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
from harness.policy_server import WamPolicy
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, SkillTable
from harness.frame_cache import FramePack


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fractions", default="0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95")
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    targets = json.loads(Path(args.targets).read_text())
    vector = torch.tensor(targets["navigation"]["move to(car)"]["open lid(car)"]["vector"], dtype=torch.float32)[None]
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    episodes = [ep for ep, row in sorted(table.episodes.items()) if index["episodes"][str(ep)]["split"] == "validation"]
    episodes = episodes[:args.max_episodes]
    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device=args.device,
                       cuda_fraction=0.4, seed=0, log_dir=args.out,
                       orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})
    fractions = [float(x) for x in args.fractions.split(",")]
    variants = {"true_state": None, "zero_base_vel": np.zeros(3, np.float32), "forward_0.2": np.array([0.2, 0.0, 0.0], np.float32)}
    rows = []
    started = time.monotonic()
    for ep in episodes:
        seg = table.episodes[ep]["segments"][0]
        if seg["skill"] != "move to":
            continue
        action = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
        state_all = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        lo, hi = seg["fill_start"], seg["fill_end"]
        for frac in fractions:
            a = int(lo + frac * (hi - lo - 33))
            a = (a + 1) // 2 * 2
            state = np.array(state_all[a], np.float32)
            demo = np.array(action[a:a + 32], np.float32)
            rgb = pack.rgb(a)
            _, pixels = policy._preprocess({"r::proprio": state, "r::zed_link::rgb": rgb})
            feats, _ = policy._encode(pixels)
            row = {"episode": ep, "anchor": a, "fraction": frac, "state_base_vel": state[:3].tolist(),
                   "demo_base_mean": demo[:, :3].mean(0).tolist(), "demo_base_first8": demo[:8, :3].mean(0).tolist(),
                   "pred": {}}
            for name, inject in variants.items():
                s = state.copy()
                if inject is not None:
                    s[:3] = inject
                policy.gen.manual_seed(1000 * ep + a)
                chunk = policy._sample(feats, s, vector)
                row["pred"][name] = {"base_mean": chunk[:, :3].mean(0).tolist(), "base_first8": chunk[:8, :3].mean(0).tolist(),
                                     "base_x_profile": [round(float(x), 3) for x in chunk[:, 0]]}
            rows.append(row)
    # summary
    def arr(fn):
        return np.array([fn(r) for r in rows])
    vel = arr(lambda r: r["state_base_vel"][0])
    demo_x = arr(lambda r: r["demo_base_mean"][0])
    demo_yaw = arr(lambda r: r["demo_base_mean"][2])
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": policy.checkpoint_step, "n_anchors": len(rows),
               "episodes": episodes, "seconds": time.monotonic() - started, "task_success_evaluated": False}
    for name in variants:
        px = arr(lambda r: r["pred"][name]["base_mean"][0])
        pyaw = arr(lambda r: r["pred"][name]["base_mean"][2])
        summary[name] = {"mae_base_x_vs_demo": float(np.abs(px - demo_x).mean()),
                         "mae_base_yaw_vs_demo": float(np.abs(pyaw - demo_yaw).mean()),
                         "corr_pred_x_vs_demo_x": float(np.corrcoef(px, demo_x)[0, 1]),
                         "corr_pred_x_vs_state_vel_x": float(np.corrcoef(px, vel)[0, 1]) if np.std(vel) > 0 else None,
                         "mean_pred_x": float(px.mean())}
    # the two regimes that matter in closed loop
    moving = [r for r in rows if r["demo_base_mean"][0] > 0.15]
    at_rest_should_move = [r for r in moving if abs(r["state_base_vel"][0]) < 0.03]
    moving_state = [r for r in moving if r["state_base_vel"][0] > 0.12]
    def regime(rs):
        if not rs:
            return {"n": 0}
        out = {"n": len(rs), "demo_x": float(np.mean([r["demo_base_mean"][0] for r in rs]))}
        for name in variants:
            out[name + "_pred_x"] = float(np.mean([r["pred"][name]["base_mean"][0] for r in rs]))
        return out
    summary["regime_demo_moving_state_at_rest"] = regime(at_rest_should_move)
    summary["regime_demo_moving_state_moving"] = regime(moving_state)
    summary["copycat_verdict"] = None
    a, b = summary["regime_demo_moving_state_moving"], summary["regime_demo_moving_state_at_rest"]
    if a.get("n", 0) >= 5 and b.get("n", 0) >= 5:
        drop = a["true_state_pred_x"] - a["zero_base_vel_pred_x"]
        summary["copycat_verdict"] = {"pred_x_drop_when_velocity_zeroed": drop,
                                      "pred_x_at_rest_demo_moving": b["true_state_pred_x"],
                                      "interpretation": ("velocity-copying: zeroing base velocity removes most of the forward command"
                                                         if drop > 0.5 * a["true_state_pred_x"] and b["true_state_pred_x"] < 0.5 * b["demo_x"]
                                                         else "no strong velocity-copying on this sample")}
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    policy.trace.close()
    for path in args.out.glob("policy_trace_*.jsonl"):
        if path.stat().st_size == 0:
            path.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
