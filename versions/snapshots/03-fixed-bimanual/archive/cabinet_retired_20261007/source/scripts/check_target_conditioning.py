#!/usr/bin/env python3
"""Check whether the frozen WAM responds to distinct targets under identical observation/noise.

Uses one real validation navigation observation. This is an input-contract diagnostic,
not task success, a memory experiment or a comparison between complete systems.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/training/stage1_skillproto/checkpoint_best.pt")
    parser.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    targets = json.loads(Path(args.targets).read_text())
    table = SkillTable()
    ep = next(ep for ep, row in sorted(table.episodes.items()) if row["split"] == "validation")
    frame = table.episodes[ep]["segments"][0]["start"]
    frame = (frame + 1) // 2 * 2
    state = np.array(np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")[frame], np.float32)
    rgb = FramePack(FRAMES / HEAD_CAMERA, ep).rgb(frame)
    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10,
                       device="cuda", cuda_fraction=0.4, seed=0, log_dir=args.out,
                       orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})
    _, pixels = policy._preprocess({"r::proprio": state, "r::zed_link::rgb": rgb})
    features, _ = policy._encode(pixels)
    rows, chunks = [], []
    for key, next_key in (("move to(car)", "open lid(car)"), ("move to(car)", "place in(toy_box,car)"),
                          ("move to(door)", "open door(door)"),
                          ("move to(digital)", "pick up from(digital,coffee_table)")):
        vector = targets["navigation"][key][next_key]["vector"]
        policy.gen.manual_seed(0)
        started = time.monotonic()
        chunk = policy._sample(features, state, torch.tensor(vector, dtype=torch.float32)[None])
        assert chunk.shape == (32, 23) and np.isfinite(chunk).all()
        rows.append({"key": key, "next_key": next_key, "seconds": time.monotonic() - started,
                     "first_base_input": chunk[0, :3].tolist(), "mean_base_input": chunk[:, :3].mean(0).tolist()})
        chunks.append(chunk)
    differences = [{"a": rows[0]["next_key"], "b": rows[i]["next_key"],
                    "mae_by_group": {name: float(np.abs(chunks[0][:, lo:hi] - chunks[i][:, lo:hi]).mean())
                                     for name, (lo, hi) in ACTION_GROUPS.items()}}
                   for i in range(1, len(chunks))]
    responsive = all(not np.array_equal(chunks[0], chunk) for chunk in chunks[1:])
    report = {"checkpoint": args.checkpoint, "checkpoint_step": policy.checkpoint_step,
              "episode": ep, "frame": frame, "split": "validation", "seed": 0,
              "fixed_observation_and_noise": True, "targets": args.targets, "responsive": responsive,
              "rows": rows, "differences": differences, "task_success_evaluated": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    policy.trace.close()
    for path in args.out.glob("policy_trace_*.jsonl"):
        if path.stat().st_size == 0:
            path.unlink()
    print(json.dumps(report, indent=2))
    assert responsive, "The frozen WAM ignored changes in target condition"


if __name__ == "__main__":
    main()
