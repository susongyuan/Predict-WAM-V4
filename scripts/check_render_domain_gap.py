#!/usr/bin/env python3
"""Paired render-domain check: does the closed-loop image pipeline change the WAM's actions?

Training / offline admission see head frames rendered at 720x720 by the original data-collection stack, then
cv2 INTER_AREA -> 224 -> JPEG q90 (data/frame_cache). The closed loop sees the Isaac Sim 6.1 port rendering the
head camera natively at 224x224 (no anti-aliased downsampling, different renderer build). The open-loop demo-action
replay (scripts/replay_demo_actions_feasibility.py) saves native frames at demo frame indices, so for each paired
step t we run the SAME checkpoint with the SAME demo state history, contract vector and flow noise on
  A: native sim frame (closed-loop domain)      B: cached demo frame (training domain)
and a noise-floor control B2 = B with a different seed. |A-B| >> |B-B2| means the image domain alone moves the
action. Evidence only: nothing here is fed back to training, and demo frames/actions are train-split data.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.frame_cache import FramePack  # noqa: E402
from harness.policy_server import WamPolicy  # noqa: E402
from harness.vtt_monitor import precondition_key  # noqa: E402
from harness.wam_behavior import (ACTION_GROUPS, FRAMES, HEAD_CAMERA, LOWDIM, STATE_SLICES, SkillTable,  # noqa: E402
                                  model_proprio_lags, stack_proprio_history)

UPPER = {"trunk": ((3, 7), STATE_SLICES["trunk_qpos"]), "left_arm": ((7, 14), STATE_SLICES["arm_left_qpos"]),
         "right_arm": ((15, 22), STATE_SLICES["arm_right_qpos"])}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def segment_key_at(segments, t):
    for s in segments:
        if s["start"] <= t < s["end"]:
            return precondition_key(s["skill"], s["objects"]), s
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="outputs/training/stage7b_prophistdiff_manip_20260924/checkpoint_best.pt")
    ap.add_argument("--native-dir", required=True, type=Path, help="replay native_frames dir (head/step_XXXXXX.png)")
    ap.add_argument("--episode", type=int, default=2603)
    ap.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tracking-rad", type=float, default=0.05,
                    help="skip pairs whose replayed upper-body joints differ from the demo state by more (mean abs)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    contracts = json.loads(Path(args.targets).read_text())["contracts"]
    table = SkillTable()
    segments = table.episodes[args.episode]["segments"]
    demo_state = np.load(LOWDIM / f"episode_{args.episode}_state.npy", mmap_mode="r")
    demo_action = np.load(LOWDIM / f"episode_{args.episode}_action.npy", mmap_mode="r")
    pack = FramePack(FRAMES / HEAD_CAMERA, args.episode)
    sim_proprio = np.load(args.native_dir / "proprio.npy", mmap_mode="r")
    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device=args.device,
                       cuda_fraction=0.4, seed=0, log_dir=args.out,
                       orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})
    lags = model_proprio_lags(policy.cfg)
    steps = sorted(int(p.stem.split("_")[1]) for p in (args.native_dir / "head").glob("step_*.png"))
    rows, skipped = [], {"no_cache": 0, "no_segment": 0, "tracking": 0, "no_future": 0}
    started = time.monotonic()
    for t in steps:
        if not pack.has(t):
            skipped["no_cache"] += 1
            continue
        key, seg = segment_key_at(segments, t)
        if key is None or key not in contracts:
            skipped["no_segment"] += 1
            continue
        if t + 32 > seg["end"]:
            skipped["no_future"] += 1
            continue
        track = float(np.abs(sim_proprio[t][[*range(53, 57), *range(3, 10), *range(28, 35)]]
                             - demo_state[t][[*range(53, 57), *range(3, 10), *range(28, 35)]]).mean())
        if track > args.max_tracking_rad:
            skipped["tracking"] += 1
            continue
        cond = torch.tensor(contracts[key]["vector"], dtype=torch.float32)[None]
        state = np.array(demo_state[t], np.float32)
        state_in = state if lags == [0] else stack_proprio_history(demo_state, t, lags)
        native = np.asarray(Image.open(args.native_dir / "head" / f"step_{t:06d}.png"))
        cached = pack.rgb(t)
        preds, cls = {}, {}
        for name, rgb, seed in (("A_native", native, 1000 * args.episode + t), ("B_cached", cached, 1000 * args.episode + t),
                                ("B2_cached_seed2", cached, 1000 * args.episode + t + 7919)):
            _, px = policy._preprocess({"r::proprio": state, "r::zed_link::rgb": rgb})
            feats, c = policy._encode(px)
            cls[name] = c
            policy.gen.manual_seed(seed)
            preds[name] = np.mean([policy._sample(feats, state_in, cond) for _ in range(args.samples)], axis=0)
        demo = np.array(demo_action[t:t + 32], np.float32)
        row = {"t": t, "key": key, "phase": (t - seg["start"]) / max(1, seg["end"] - seg["start"]), "tracking_rad": track,
               "cls_cos_native_vs_cached": cosine(cls["A_native"], cls["B_cached"]), "groups": {}}
        for g, (lo, hi) in ACTION_GROUPS.items():
            row["groups"][g] = {"img_effect": float(np.abs(preds["A_native"][:, lo:hi] - preds["B_cached"][:, lo:hi]).mean()),
                                "noise_floor": float(np.abs(preds["B2_cached_seed2"][:, lo:hi] - preds["B_cached"][:, lo:hi]).mean()),
                                "err_native": float(np.abs(preds["A_native"][:, lo:hi] - demo[:, lo:hi]).mean()),
                                "err_cached": float(np.abs(preds["B_cached"][:, lo:hi] - demo[:, lo:hi]).mean())}
        for g, ((lo, hi), (slo, shi)) in UPPER.items():
            q = state[slo:shi]
            d = demo[-1, lo:hi] - q
            row["groups"][g].update({"intent_cos_native": cosine(preds["A_native"][-1, lo:hi] - q, d),
                                     "intent_cos_cached": cosine(preds["B_cached"][-1, lo:hi] - q, d),
                                     "intent_norm_native": float(np.linalg.norm(preds["A_native"][-1, lo:hi] - q)),
                                     "intent_norm_cached": float(np.linalg.norm(preds["B_cached"][-1, lo:hi] - q)),
                                     "intent_norm_demo": float(np.linalg.norm(d))})
        row["base_vx"] = {k: float(v[:, 0].mean()) for k, v in preds.items()} | {"demo": float(demo[:, 0].mean())}
        rows.append(row)
    summary = {"checkpoint": args.checkpoint, "episode": args.episode, "native_dir": str(args.native_dir),
               "samples": args.samples, "pairs": len(rows), "skipped": skipped, "seconds": time.monotonic() - started,
               "task_success_evaluated": False, "note": "paired offline check; A native Isaac-6.1 224 render of the "
               "replayed demo state, B training-cache frame of the same demo frame; same state/cond/noise", "by_key": {},
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for key in sorted({r["key"] for r in rows}):
        rs = [r for r in rows if r["key"] == key]
        s = summary["by_key"][key] = {"pairs": len(rs), "cls_cos_mean": float(np.mean([r["cls_cos_native_vs_cached"] for r in rs])),
                                      "base_vx_mean": {k: float(np.mean([r["base_vx"][k] for r in rs]))
                                                       for k in ("A_native", "B_cached", "demo")}, "groups": {}}
        for g in ACTION_GROUPS:
            s["groups"][g] = {m: float(np.mean([r["groups"][g][m] for r in rs])) for m in
                              ("img_effect", "noise_floor", "err_native", "err_cached")}
            if g in UPPER:
                s["groups"][g].update({m: float(np.nanmean([r["groups"][g][m] for r in rs])) for m in
                                       ("intent_cos_native", "intent_cos_cached", "intent_norm_native",
                                        "intent_norm_cached", "intent_norm_demo")})
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    policy.trace.close()
    for p in args.out.glob("policy_trace_*.jsonl"):
        if p.stat().st_size == 0:
            p.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
