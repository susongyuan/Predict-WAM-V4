#!/usr/bin/env python3
"""Chunk-to-chunk consistency of the WAM under re-planning: does fresh sampling noise make absolute joint
targets jump between consecutive chunks (arm flailing), and does a fixed noise seed remove the jump?

Offline diagnostic on cached validation demonstrations. For each anchor a: sample chunks at a and at a+8
(the demo state/frame 8 steps later) with fresh noise and with the same noise; the replan jump is
|chunk_{a+8}[0] - chunk_a[8]| per action group. Compared with the demonstration's own change over 8 steps.
Across-seed spread at one observation is also reported. Not task success; not a system comparison.
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
from harness.wam_behavior import ACTION_GROUPS, FRAMES, HEAD_CAMERA, LOWDIM, SkillTable
from harness.frame_cache import FramePack
from harness.orchestrator import ConditionBank
from harness.vtt_monitor import precondition_key


def group_abs(delta):
    return {name: float(np.abs(delta[..., lo:hi]).mean()) for name, (lo, hi) in ACTION_GROUPS.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-episodes", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    targets = json.loads(Path(args.targets).read_text())
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    episodes = [ep for ep, row in sorted(table.episodes.items()) if index["episodes"][str(ep)]["split"] == "validation"][:args.max_episodes]
    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device=args.device,
                       cuda_fraction=0.4, seed=0, log_dir=args.out,
                       orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})
    bank = policy.bank
    started = time.monotonic()
    rows = []
    for ep in episodes:
        segments = table.episodes[ep]["segments"]
        action = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
        state_all = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        for si, seg in enumerate(segments[:4]):  # first navigation, open lid, second navigation, open door
            key = precondition_key(seg["skill"], seg["objects"])
            nxt = segments[si + 1] if si + 1 < len(segments) else None
            next_key = precondition_key(nxt["skill"], nxt["objects"]) if nxt and nxt["skill"] != "move to" else None
            skey = bank.storage_key(key, seg["skill"], next_key)
            if bank.targets is None or skey not in bank.targets:
                continue
            vector = torch.from_numpy(np.asarray(bank.targets[skey], np.float32))[None]
            lo, hi = seg["fill_start"], seg["fill_end"]
            for frac in (0.2, 0.5, 0.8):
                a = int(lo + frac * (hi - lo - 48))
                a = (a + 1) // 2 * 2
                b = a + 8
                obs = {}
                for t in (a, b):
                    state = np.array(state_all[t], np.float32)
                    _, pixels = policy._preprocess({"r::proprio": state, "r::zed_link::rgb": pack.rgb(t)})
                    feats, _ = policy._encode(pixels)
                    obs[t] = (state, feats)
                demo_a = np.array(action[a:a + 32], np.float32)
                demo_b = np.array(action[b:b + 32], np.float32)
                # across-seed spread at a
                samples = []
                for s in range(args.seeds):
                    policy.gen.manual_seed(s)
                    samples.append(policy._sample(obs[a][1], obs[a][0], vector))
                samples = np.stack(samples)
                spread0 = {name: float(samples[:, 0, lo_:hi_].std(0).mean()) for name, (lo_, hi_) in ACTION_GROUPS.items()}
                # replan jump, fresh noise (seed s at a, seed s+100 at b) vs same noise (seed s at both)
                fresh, same = [], []
                for s in range(args.seeds):
                    ca = samples[s]
                    policy.gen.manual_seed(s + 100)
                    cb_fresh = policy._sample(obs[b][1], obs[b][0], vector)
                    policy.gen.manual_seed(s)
                    cb_same = policy._sample(obs[b][1], obs[b][0], vector)
                    fresh.append(cb_fresh[0] - ca[8])
                    same.append(cb_same[0] - ca[8])
                rows.append({"episode": ep, "segment": si, "key": skey, "anchor": a, "fraction": frac,
                             "spread_step0_across_seeds": spread0,
                             "replan_jump_fresh_noise": group_abs(np.stack(fresh)),
                             "replan_jump_same_noise": group_abs(np.stack(same)),
                             "demo_change_over_8_steps": group_abs(demo_b[0] - demo_a[0]),
                             "demo_within_chunk_step_change": group_abs(np.diff(demo_a, axis=0)),
                             "pred_error_step0_vs_demo": group_abs(samples[:, 0] - demo_a[0]),
                             "pred_error_step0_vs_demo_arms_by_seed": [float(np.abs(samples[s, 0, 7:14] - demo_a[0, 7:14]).mean()) for s in range(args.seeds)]})
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": policy.checkpoint_step, "n_anchors": len(rows), "seeds": args.seeds,
               "episodes": episodes, "seconds": time.monotonic() - started, "task_success_evaluated": False, "by_skill": {}}
    for skill in sorted({r["key"].partition("(")[0] for r in rows}):
        rs = [r for r in rows if r["key"].startswith(skill + "(")]
        agg = {}
        for field in ("spread_step0_across_seeds", "replan_jump_fresh_noise", "replan_jump_same_noise",
                      "demo_change_over_8_steps", "demo_within_chunk_step_change", "pred_error_step0_vs_demo"):
            agg[field] = {g: float(np.mean([r[field][g] for r in rs])) for g in ACTION_GROUPS}
        agg["n"] = len(rs)
        summary["by_skill"][skill] = agg
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    policy.trace.close()
    for path in args.out.glob("policy_trace_*.jsonl"):
        if path.stat().st_size == 0:
            path.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
