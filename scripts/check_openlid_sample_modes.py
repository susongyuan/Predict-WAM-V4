#!/usr/bin/env python3
"""Measure whether stochastic WAM samples for open-lid disagree and cancel when averaged.

This offline validation diagnostic uses held-out demonstration observations.  A per-anchor
"best sample" score is reported only as a non-deployable capacity upper bound; it must not
be used to select actions in closed-loop evaluation because that would require the answer
trajectory.
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
from harness.frame_cache import FramePack  # noqa: E402
from harness.policy_server import WamPolicy  # noqa: E402
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, STATE_SLICES, SkillTable  # noqa: E402


POSITION_PAIRS = {
    "trunk": ((3, 7), STATE_SLICES["trunk_qpos"]),
    "left_arm": ((7, 14), STATE_SLICES["arm_left_qpos"]),
    "right_arm": ((15, 22), STATE_SLICES["arm_right_qpos"]),
}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-8 and nb > 1e-8 else None


def mean_finite(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fractions", default="0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    args.out.mkdir(parents=True)

    targets = json.loads((ROOT / args.targets).read_text())
    condition = torch.tensor(targets["contracts"]["open lid(car)"]["vector"], dtype=torch.float32)[None]
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    episodes = [episode for episode in sorted(table.episodes)
                if index["episodes"][str(episode)]["split"] == "validation"][:args.max_episodes]
    fractions = [float(value) for value in args.fractions.split(",")]
    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device=args.device,
                       cuda_fraction=0.4, seed=0, log_dir=args.out,
                       orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})

    rows = []
    started = time.monotonic()
    for episode in episodes:
        segment = next(segment for segment in table.episodes[episode]["segments"] if segment["skill"] == "open lid")
        action = np.load(LOWDIM / f"episode_{episode}_action.npy", mmap_mode="r")
        state_all = np.load(LOWDIM / f"episode_{episode}_state.npy", mmap_mode="r")
        pack = FramePack(FRAMES / HEAD_CAMERA, episode)
        lo, hi = segment["start"], segment["end"]
        for fraction in fractions:
            anchor = int(lo + fraction * (hi - lo - 33))
            anchor = (anchor + 1) // 2 * 2
            state = np.asarray(state_all[anchor], dtype=np.float32)
            demo = np.asarray(action[anchor:anchor + 32], dtype=np.float32)
            _, pixels = policy._preprocess({"r::proprio": state, "r::zed_link::rgb": pack.rgb(anchor)})
            features, _ = policy._encode(pixels)
            policy.gen.manual_seed(1000 * episode + anchor)
            samples = np.stack([policy._sample(features, state, condition) for _ in range(args.samples)])
            row = {"episode": episode, "anchor": anchor, "fraction": fraction, "groups": {}}
            for group, ((action_start, action_end), (state_start, state_end)) in POSITION_PAIRS.items():
                q = state[state_start:state_end]
                demo_intent = demo[-1, action_start:action_end] - q
                sample_intents = samples[:, -1, action_start:action_end] - q[None]
                sample_norms = np.linalg.norm(sample_intents, axis=1)
                mean_intent = sample_intents.mean(axis=0)
                sample_cosines = [cosine(intent, demo_intent) for intent in sample_intents]
                row["groups"][group] = {
                    "demo_intent_norm": float(np.linalg.norm(demo_intent)),
                    "sample_intent_norm_mean": float(sample_norms.mean()),
                    "mean_intent_norm": float(np.linalg.norm(mean_intent)),
                    "sample_cancellation_ratio": (float(np.linalg.norm(mean_intent) / sample_norms.mean())
                                                  if sample_norms.mean() > 1e-8 else None),
                    "sample_demo_cos_mean": mean_finite(sample_cosines),
                    "sample_demo_cos_max_oracle": (max(value for value in sample_cosines if value is not None)
                                                   if any(value is not None for value in sample_cosines) else None),
                    "mean_demo_cos": cosine(mean_intent, demo_intent),
                }
            rows.append(row)

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": policy.checkpoint_step,
        "key": "open lid(car)",
        "split": "validation",
        "samples": args.samples,
        "episodes": episodes,
        "n_anchors": len(rows),
        "seconds": time.monotonic() - started,
        "policy_inference": True,
        "closed_loop": False,
        "task_success_evaluated": False,
        "oracle_warning": "best-sample cosine uses demonstrated future actions and is diagnostic only; never deploy it",
        "groups": {},
        "by_fraction": {},
        "rows": rows,
    }
    metrics = ("demo_intent_norm", "sample_intent_norm_mean", "mean_intent_norm",
               "sample_cancellation_ratio", "sample_demo_cos_mean", "sample_demo_cos_max_oracle", "mean_demo_cos")
    for group in POSITION_PAIRS:
        report["groups"][group] = {metric: mean_finite([row["groups"][group][metric] for row in rows])
                                    for metric in metrics}
    for fraction in fractions:
        subset = [row for row in rows if row["fraction"] == fraction]
        report["by_fraction"][str(fraction)] = {
            group: {metric: mean_finite([row["groups"][group][metric] for row in subset]) for metric in metrics}
            for group in POSITION_PAIRS
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Open-lid stochastic sample audit", "",
             "Offline held-out prediction diagnostic; not closed-loop task success. Oracle best-sample scores are not deployable.", "",
             "| Group | demo norm | sample norm | averaged norm | cancellation | sample cos | averaged cos | oracle best cos |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, values in report["groups"].items():
        lines.append(f"| {group} | {values['demo_intent_norm']:.3f} | {values['sample_intent_norm_mean']:.3f} | "
                     f"{values['mean_intent_norm']:.3f} | {values['sample_cancellation_ratio']:.3f} | "
                     f"{values['sample_demo_cos_mean']:.3f} | {values['mean_demo_cos']:.3f} | "
                     f"{values['sample_demo_cos_max_oracle']:.3f} |")
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(args.out), "n_anchors": len(rows), "seconds": report["seconds"],
                      "groups": report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
