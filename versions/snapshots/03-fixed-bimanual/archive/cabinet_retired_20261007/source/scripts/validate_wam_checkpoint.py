#!/usr/bin/env python3
"""CPU checkpoint check on real held-out frames, one middle-of-skill chunk per episode / class.

Offline, annotation-conditioned diagnostics only: no simulator, VLM, task score or RL comparison.
Run with CUDA_VISIBLE_DEVICES= and the lilawam Python in a CPU/RAM-limited cgroup.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from harness.frame_cache import FramePack
from harness.policy_server import WamPolicy
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, SkillTable, group_errors, hold_baseline
from behavior_preflight import hardware_snapshot


def select_samples(per_skill):
    index = json.loads((LOWDIM / "index.json").read_text())
    table = SkillTable()
    selected = []
    for skill, cid in table.classes.items():
        count = 0
        for ep, meta in sorted(index["episodes"].items(), key=lambda item: int(item[0])):
            if meta["split"] != "validation":
                continue
            for seg in table.episodes[int(ep)]["segments"]:
                if seg["class"] != cid or seg["end"] - seg["start"] < 64:
                    continue
                anchor = ((seg["start"] + seg["end"] - 32) // 2) // 2 * 2
                assert seg["start"] <= anchor and anchor + 32 <= seg["end"] <= meta["length"]
                selected.append({"episode": int(ep), "anchor": anchor, "skill": skill, "class": cid})
                count += 1
                break
            if count == per_skill:
                break
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/training/stage1_skillproto/checkpoint_best.pt")
    ap.add_argument("--per-skill", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.per_skill < 1:
        ap.error("--per-skill must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device="cpu",
                       cuda_fraction=0, seed=0, log_dir=args.out)
    rows = []
    report = {"checkpoint": str(args.checkpoint), "checkpoint_step": policy.checkpoint_step,
              "device": "cpu", "split": "validation", "seed": 0, "inference_steps": 10,
              "condition_source": "ground-truth skill class from held-out annotation; training-only VTT prototype",
              "sampling": "first two eligible validation episodes per class; one segment midpoint each",
              "physical_bounds_checked": False, "closed_loop": False, "rows": rows,
              "limitations": ["Small diagnostic sample, not the full validation set or a success rate",
                              "No VLM grounding or object-specific condition validation",
                              "Hold comparison only for base/trunk/arms; legacy gripper zeros are not a physical hold",
                              "No RL baseline; no replacement claim"]}
    for sample in select_samples(args.per_skill):
        snapshot = hardware_snapshot()
        if max(snapshot["cpu_used_percent"], snapshot["memory_used_percent"]) >= 80:
            raise RuntimeError("Global CPU/RAM headroom lost; CPU validation stopped")
        ep, anchor = sample["episode"], sample["anchor"]
        state = np.array(np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")[anchor], dtype=np.float32)
        target = torch.from_numpy(np.array(np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")[anchor:anchor+32]))
        frame = FramePack(FRAMES / HEAD_CAMERA, ep).rgb(anchor)
        obs = {"robot_r1::proprio": state, "robot_r1::zed_link::rgb": frame}
        policy.reset()
        policy.skill = sample["class"]
        t0 = time.monotonic()
        # The actual policy act path produces the first action and queues the next seven.
        first = policy.act(obs).numpy()
        prefix = np.stack([first, *policy.queue])
        elapsed = time.monotonic() - t0
        assert prefix.shape == (8, 23) and np.isfinite(prefix).all()
        mae = group_errors(torch.from_numpy(prefix)[None], target[:8][None])
        hold = group_errors(hold_baseline(torch.from_numpy(state)[None], chunk_len=8), target[:8][None])
        hold = {k: v for k, v in hold.items() if "gripper" not in k}
        # Error/HOLD invalidates the unexecuted model queue, and base motion stops.
        holding = policy.hold(obs)
        assert not policy.queue and torch.equal(holding[:3], torch.zeros(3))
        rows.append({**sample, "finite": True, "executed_prefix_shape": list(prefix.shape),
                     "inference_seconds": round(elapsed, 3), "mae_native_units": mae,
                     "hold_mae_non_gripper": hold})
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)
    report["status"] = "passed_finite_action_and_queue_contract"
    report["samples"] = len(rows)
    report["latency_seconds_p50_p95"] = np.percentile([r["inference_seconds"] for r in rows], [50, 95]).tolist()
    report["mae_native_units"] = {k: float(np.mean([r["mae_native_units"][k] for r in rows])) for k in rows[0]["mae_native_units"]}
    report["hold_mae_non_gripper"] = {k: float(np.mean([r["hold_mae_non_gripper"][k] for r in rows])) for k in rows[0]["hold_mae_non_gripper"]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(rows)} real held-out observations; result at {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
