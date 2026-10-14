#!/usr/bin/env python3
"""Normalization statistics for the `upper_body_delta` action representation (train split only).

Motivation (outputs/validation/openlid_replay_*, progress.md 2026-09-22 17:55): with absolute joint
targets and joint min-max normalization, the hold baseline already explains all but ~8% of the
normalized action error, so the WAM learned "copy the current joints" for trunk/arms.  Here the 18
position-controlled upper-body dimensions are re-expressed as (target - current joint position) and
normalized with SYMMETRIC bounds (zero motion maps to exactly 0), taken from a high percentile of the
training-set |delta| so the typical 32-step motion spans a comparable normalized range to the base.

The output keeps the upstream `behavior1k.action.min/max` layout (delta dims replaced by -/+bound) so
the unchanged LiLa-WAM loader can read it; `action_repr` in the file must match the model config.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import ACTION_GROUPS, ACTION_REPRS, LOWDIM, STATE_SLICES  # noqa: E402

DATA = ROOT / "data/behavior-1k/2026-challenge-demos"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DATA / "train_stats_task013.json", type=Path)
    ap.add_argument("--out", default=DATA / "train_stats_task013_delta.json", type=Path)
    ap.add_argument("--percentile", type=float, default=99.5, help="symmetric bound = this percentile of |delta|")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="verify an existing output instead of writing")
    args = ap.parse_args()
    src = json.loads(args.source.read_text())
    index = json.loads((LOWDIM / "index.json").read_text())
    episodes = sorted(int(k) for k, v in index["episodes"].items() if v["split"] == "train")
    assert episodes == sorted(src["provenance"]["episode_indices"]), "train episodes differ from the absolute stats"
    pairs = ACTION_REPRS["upper_body_delta"]
    deltas = {name: [] for name, _ in pairs}
    anchors = 0
    for ep in episodes:
        action = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
        state = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        starts = np.arange(0, len(action) - args.chunk + 1, args.stride)
        anchors += len(starts)
        for name, sname in pairs:
            alo, ahi = ACTION_GROUPS[name]
            slo, shi = STATE_SLICES[sname]
            a = np.lib.stride_tricks.sliding_window_view(np.asarray(action[:, alo:ahi]), args.chunk, axis=0)[starts]  # (n,k,32)
            d = a - np.asarray(state[starts, slo:shi])[:, :, None]
            deltas[name].append(np.abs(d).reshape(-1, ahi - alo))
    amin = np.array(src["behavior1k"]["action"]["min"], np.float64)
    amax = np.array(src["behavior1k"]["action"]["max"], np.float64)
    bounds = {}
    for name, _ in pairs:
        alo, ahi = ACTION_GROUPS[name]
        allabs = np.concatenate(deltas[name])
        b = np.percentile(allabs, args.percentile, axis=0)
        b[b < 1e-6] = 0.0  # constant joints keep a zero range (loader maps denominator 0 -> 1)
        bounds[name] = {"bound": b.tolist(), "abs_delta_std": allabs.std(0).tolist(), "abs_delta_max": allabs.max(0).tolist(),
                        "frac_beyond_bound": (allabs > b).mean(0).tolist()}
        amin[alo:ahi] = -b
        amax[alo:ahi] = b
    out = {"action_repr": "upper_body_delta",
           "delta_groups": [[name, sname] for name, sname in pairs],
           "delta_bounds": bounds,
           "provenance": {**src["provenance"], "absolute_stats_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                          "delta_percentile": args.percentile, "chunk": args.chunk, "stride": args.stride, "anchors": int(anchors),
                          "note": "delta dims: symmetric +/-percentile(|target - current qpos|) over all train chunks; other dims copied"},
           "behavior1k": {"action": {**src["behavior1k"]["action"], "min": amin.tolist(), "max": amax.tolist()},
                          "state": src["behavior1k"]["state"]},
           "note": "Training-split statistics for the upper_body_delta WAM action representation; validation/report reuse these."}
    text = json.dumps(out, indent=1) + "\n"
    if args.check:
        existing = json.loads(args.out.read_text())
        assert existing["behavior1k"]["action"]["min"] == out["behavior1k"]["action"]["min"], "min differs"
        assert existing["behavior1k"]["action"]["max"] == out["behavior1k"]["action"]["max"], "max differs"
        print("check ok", args.out)
        return
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; delete explicitly or use --check")
    args.out.write_text(text)
    print(json.dumps({"out": str(args.out), "anchors": anchors, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "bounds": {k: [round(x, 4) for x in v["bound"]] for k, v in bounds.items()},
                      "frac_beyond": {k: [round(x, 4) for x in v["frac_beyond_bound"]] for k, v in bounds.items()}}, indent=1))


if __name__ == "__main__":
    main()
