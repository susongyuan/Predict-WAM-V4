#!/usr/bin/env python3
"""Fast-reach metric: does the WAM reproduce the demo's large upper-body motion at the start of a manipulation segment?

Anchors: every 8 frames in the first 15% of each validation episode's `--key` segment (training condition vector,
train-only stats). Per anchor: demo intent = a[t+31, upper] - q_t (18D trunk+arms, rad); prediction intent from the
K-sample mean and from each single sample (same seeds for every checkpoint: 1000*episode+anchor). "Fast" anchors =
demo 18D intent norm >= --fast-rad. Reports pred/demo norm ratio and cosine for the K-mean and for single samples.
Offline prediction only; not a task or skill success.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.frame_cache import FramePack  # noqa: E402
from harness.policy_server import WamPolicy  # noqa: E402
from harness.vtt_monitor import precondition_key  # noqa: E402
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, STATE_SLICES, SkillTable, model_proprio_lags, stack_proprio_history  # noqa: E402

A_UP = np.r_[3:7, 7:14, 15:22]
S_UP = np.r_[STATE_SLICES["trunk_qpos"][0]:STATE_SLICES["trunk_qpos"][1], STATE_SLICES["arm_left_qpos"][0]:STATE_SLICES["arm_left_qpos"][1],
             STATE_SLICES["arm_right_qpos"][0]:STATE_SLICES["arm_right_qpos"][1]]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--key", default="open lid(car)")
    ap.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--first-frac", type=float, default=0.15)
    ap.add_argument("--anchor-stride", type=int, default=8)
    ap.add_argument("--fast-rad", type=float, default=0.3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    vec = torch.tensor(json.loads(Path(args.targets).read_text())["contracts"][args.key]["vector"], dtype=torch.float32)[None]
    table = SkillTable(); index = json.loads((LOWDIM / "index.json").read_text())
    eps = [ep for ep, _ in sorted(table.episodes.items()) if index["episodes"][str(ep)]["split"] == "validation"][:20]
    pol = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device=args.device, cuda_fraction=0.4,
                    seed=0, log_dir=args.out, orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})
    lags = model_proprio_lags(pol.cfg)
    rows, t0 = [], time.monotonic()
    for ep in eps:
        segs = [s for s in table.episodes[ep]["segments"] if precondition_key(s["skill"], s["objects"]) == args.key]
        if len(segs) != 1:
            continue
        lo, hi = segs[0]["start"], segs[0]["end"]
        st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r"); ac = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
        pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        for a in range(lo + (lo % 2), int(lo + args.first_frac * (hi - lo)), args.anchor_stride):
            if a + 32 > hi or not pack.has(a):
                continue
            state = np.array(st[a], np.float32); q = state[S_UP]
            demo = np.array(ac[a + 31], np.float32)[A_UP] - q
            _, px = pol._preprocess({"r::proprio": state, "r::zed_link::rgb": pack.rgb(a)})
            feats, _ = pol._encode(px)
            s_in = state if lags == [0] else stack_proprio_history(st, a, lags)
            pol.gen.manual_seed(1000 * ep + a)
            singles = [pol._sample(feats, s_in, vec)[-1, A_UP] - q for _ in range(args.samples)]
            mean = np.mean(singles, axis=0)
            rows.append({"episode": ep, "anchor": a, "frac": (a - lo) / (hi - lo), "demo_norm": float(np.linalg.norm(demo)),
                         "mean_norm": float(np.linalg.norm(mean)), "mean_cos": cos(mean, demo),
                         "single_norms": [float(np.linalg.norm(x)) for x in singles], "single_cos": [cos(x, demo) for x in singles],
                         "demo_rarm_j0": float(demo[11]), "mean_rarm_j0": float(mean[11])})
    def summ(rs):
        if not rs:
            return None
        dn = np.array([r["demo_norm"] for r in rs])
        return {"anchors": len(rs), "demo_norm_mean": float(dn.mean()),
                "kmean_norm_ratio": float(np.mean([r["mean_norm"] for r in rs]) / dn.mean()),
                "single_norm_ratio": float(np.mean([np.mean(r["single_norms"]) for r in rs]) / dn.mean()),
                "kmean_cos": float(np.nanmean([r["mean_cos"] for r in rs])),
                "single_cos": float(np.nanmean([np.nanmean(r["single_cos"]) for r in rs])),
                "rarm_j0_ratio": float(np.mean([r["mean_rarm_j0"] for r in rs]) / np.mean([r["demo_rarm_j0"] for r in rs]))}
    fast = [r for r in rows if r["demo_norm"] >= args.fast_rad]
    summary = {"checkpoint": args.checkpoint, "key": args.key, "samples": args.samples, "episodes": eps,
               "first_frac": args.first_frac, "anchor_stride": args.anchor_stride, "fast_rad": args.fast_rad,
               "seed_rule": "1000*episode+anchor", "all": summ(rows), "fast": summ(fast), "seconds": time.monotonic() - t0,
               "task_success_evaluated": False, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n"); (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pol.trace.close()
    for p in args.out.glob("policy_trace_*.jsonl"):
        if p.stat().st_size == 0:
            p.unlink()
    print(json.dumps({k: summary[k] for k in ("all", "fast", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
