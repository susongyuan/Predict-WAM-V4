#!/usr/bin/env python3
"""Does WAM prediction error grow along the 32-step chunk?  (offline, validation demos, no simulator)

For held-out validation segments of a contract key, sample K flow chunks at anchors spread through the segment
and compare, per chunk index i = 0..31, the error of (a) a single sample and (b) the K-sample mean against the
demonstrated action a[t+i], with the hold baseline (current measured position for trunk/arms, zero velocity for
the base).  Position groups use absolute joint targets (stage3 representation), so a flat error curve means a
long chunk is not what degrades the prediction; a steep late rise supports a shorter execution horizon.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from harness.frame_cache import FramePack  # noqa: E402
from harness.policy_server import WamPolicy  # noqa: E402
from harness.vtt_monitor import precondition_key  # noqa: E402
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, STATE_SLICES, SkillTable  # noqa: E402
from check_sample_selection import sample_with_future  # noqa: E402

GROUPS = {"trunk": ((3, 7), STATE_SLICES["trunk_qpos"]),
          "left_arm": ((7, 14), STATE_SLICES["arm_left_qpos"]),
          "right_arm": ((15, 22), STATE_SLICES["arm_right_qpos"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt")
    ap.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    ap.add_argument("--keys", default="open lid(car);move to(car)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/validation/chunk_index_error_20260923")
    ap.add_argument("--fractions", default="0.05,0.2,0.35,0.5,0.65,0.8")
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    args.out.mkdir(parents=True)
    t0 = time.time()
    targets = json.loads((ROOT / args.targets).read_text())
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device="cuda",
                       cuda_fraction=0.4, seed=0, log_dir=args.out,
                       orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})
    fr = [float(x) for x in args.fractions.split(",")]
    report = {"checkpoint": args.checkpoint, "samples": args.samples, "keys": {}}
    for key in args.keys.split(";"):
        cond = torch.tensor(targets["contracts"][key]["vector"], dtype=torch.float32)[None]
        err = {g: {"single": [], "mean": [], "hold": []} for g in list(GROUPS) + ["base"]}
        n_eps = 0
        for ep in sorted(table.episodes):
            if index["episodes"][str(ep)]["split"] != "validation":
                continue
            segs = table.episodes[ep]["segments"]
            for j, s in enumerate(segs):
                skey = precondition_key(s["skill"], s["objects"])
                if s["skill"] == "move to":
                    nxt = segs[j + 1] if j + 1 < len(segs) else None
                    skey = f"move to({s['objects'][0]})" if s.get("objects") else None
                    if key == "move to(car)" and (nxt is None or precondition_key(nxt["skill"], nxt["objects"]) != "open lid(car)"):
                        continue
                if skey != key or s["end"] - s["start"] < 64:
                    continue
                act = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
                st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
                pack = FramePack(FRAMES / HEAD_CAMERA, ep)
                for f in fr:
                    a = int(s["start"] + f * (s["end"] - s["start"] - 33)) // pack.stride * pack.stride
                    state = np.array(st[a], np.float32)
                    demo = np.array(act[a:a + 32], np.float32)
                    _, px = policy._preprocess({"r::proprio": state, "r::zed_link::rgb": pack.rgb(a)})
                    feats, _ = policy._encode(px)
                    policy.gen.manual_seed(1000 * ep + a)
                    S = np.stack([sample_with_future(policy, feats, state, cond)[0] for _ in range(args.samples)])
                    M = S.mean(0)
                    for g, ((a0, a1), (s0, s1)) in GROUPS.items():
                        err[g]["single"].append(np.abs(S[:, :, a0:a1] - demo[None, :, a0:a1]).mean((0, 2)))
                        err[g]["mean"].append(np.abs(M[:, a0:a1] - demo[:, a0:a1]).mean(1))
                        err[g]["hold"].append(np.abs(state[s0:s1][None] - demo[:, a0:a1]).mean(1))
                    err["base"]["single"].append(np.abs(S[:, :, :3] - demo[None, :, :3]).mean((0, 2)))
                    err["base"]["mean"].append(np.abs(M[:, :3] - demo[:, :3]).mean(1))
                    err["base"]["hold"].append(np.abs(demo[:, :3]).mean(1))
                n_eps += 1
                break
            if n_eps >= args.max_episodes:
                break
        out = {"n_segments": n_eps, "n_anchors": len(err["trunk"]["mean"]), "groups": {}}
        for g, d in err.items():
            curves = {m: np.mean(v, 0) for m, v in d.items()}
            bins = {m: [float(c[i:i + 8].mean()) for i in (0, 8, 16, 24)] for m, c in curves.items()}
            out["groups"][g] = {"curve": {m: [round(float(x), 5) for x in c] for m, c in curves.items()},
                                "mean_by_8step_bin": bins,
                                "late_over_early_mean": bins["mean"][3] / max(bins["mean"][0], 1e-9),
                                "mean_minus_hold_by_bin": [round(bins["mean"][k] - bins["hold"][k], 5) for k in range(4)]}
            print(key, g, {m: [round(x, 4) for x in b] for m, b in bins.items()}, flush=True)
        report["keys"][key] = out
    report["runtime_s"] = round(time.time() - t0, 1)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print("runtime", report["runtime_s"])


if __name__ == "__main__":
    main()
