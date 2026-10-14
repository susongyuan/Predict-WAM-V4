#!/usr/bin/env python3
"""Train-only normalization statistics for WAM v3 history tokens (harness/wam_v3.py).

proprio_diff_std[j]   std over TRAIN anchors of s_t - s_{h_j} (61-D), h_j = history_frames(t, segment start, lags)[j]
memory_diff_std[v, j] std over TRAIN anchors of pool_t - pool_{h_j} (2048-D) per view, h_j from the even-rounded start
Both use exactly the dataset's frame rule (WamV3Dataset.__getitem__), every `--every`-th anchor. Near-constant dims
get std 1.0 (their diffs are ~0 anyway). Validation/report data are never read.

  P2_TASK_ID=0 .conda-envs/lilawam/bin/python scripts/build_wam_v3_stats.py --config configs/<v3>.yaml [--proprio-only]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import task_profile  # noqa: E402
from harness.wam_behavior import SkillTable, model_views  # noqa: E402
from harness.wam_v3 import FeatureCache, ceil_even, history_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--proprio-only", action="store_true")
    args = ap.parse_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / args.config)
    v3 = cfg.model.v3
    plags, mlags = [int(v) for v in v3.proprio_lags], [int(v) for v in v3.memory_lags]
    views = model_views(cfg)
    split = json.loads(task_profile.SPLIT.read_text())
    train = sorted(int(e) for e in split["train_episode_indices"])
    skills = SkillTable()
    index = json.loads((task_profile.LOWDIM / "index.json").read_text())
    cache = None if args.proprio_only else FeatureCache(ROOT / cfg.dataset.feature_cache, views)
    p_sum = np.zeros((len(plags) + 1, 61)); p_sq = np.zeros_like(p_sum)
    m_sum = np.zeros((len(views), len(mlags) + 1, 2048)); m_sq = np.zeros_like(m_sum)
    n = 0
    t0 = time.time()
    for ep in train:
        state = np.load(task_profile.LOWDIM / f"episode_{ep}_state.npy").astype(np.float64)
        length = index["episodes"][str(ep)]["length"]
        for seg in skills.episodes[ep]["segments"]:
            if seg["skill"] == "move to":
                continue
            start, end = int(seg["start"]), int(seg["end"])
            # anchors: even frames with the whole chunk/future inside the segment (dataset 'safe' rule)
            first = ceil_even(start)
            anchors = np.arange(first, min(end, length) - 32, 2)[::args.every]
            for a in anchors:
                a = int(a)
                pf = history_frames(a, start, plags)
                d = state[a][None] - state[pf]
                p_sum += d; p_sq += d * d
                if cache is not None:
                    mf = history_frames(a, ceil_even(start), mlags)
                    for vi, view in enumerate(views):
                        pool = cache.pool(view, ep, [a] + mf).reshape(len(mf) + 1, -1).astype(np.float64)
                        dm = pool[:1] - pool[1:]
                        m_sum[vi] += dm; m_sq[vi] += dm * dm
                n += 1
    p_std = np.sqrt(np.maximum(p_sq / n - (p_sum / n) ** 2, 0.0))
    p_std = np.where(p_std < 1e-6, 1.0, p_std)
    out = {"task_id": task_profile.TASK_ID, "split": "train only", "train_episodes": len(train), "anchors_used": n,
           "every": args.every, "proprio_lags": plags, "memory_lags": mlags, "views": views,
           "rule": "diff_j = x_t - x_{history_frames(t, start, lags)[j]} (last j = contract start); token = clip(diff/std, +-5)",
           "proprio_diff_std": p_std.tolist(),
           "split_sha256": hashlib.sha256(task_profile.SPLIT.read_bytes()).hexdigest()}
    if cache is not None:
        m_std = np.sqrt(np.maximum(m_sq / n - (m_sum / n) ** 2, 0.0))
        m_std = np.where(m_std < 1e-4, 1.0, m_std)
        out["memory_diff_std"] = m_std.tolist()
        out["memory_diff_std_median"] = {v: [float(np.median(m_std[i, j])) for j in range(m_std.shape[1])] for i, v in enumerate(views)}
        out["feature_cache"] = str(cfg.dataset.feature_cache)
    path = ROOT / v3.stats_path
    if args.proprio_only:
        path = path.with_name(path.stem + ".proprio_only.json")
    path.write_text(json.dumps(out))
    print(json.dumps({"path": str(path), "anchors": n, "seconds": round(time.time() - t0, 1),
                      "trunk_std_by_lag": [np.round(p_std[j, 53:57], 4).tolist() for j in range(len(plags) + 1)],
                      "memory_median": out.get("memory_diff_std_median")}, indent=1))


if __name__ == "__main__":
    main()
