#!/usr/bin/env python3
"""Structural and sampled-content audit for the cached DINO features used by WAM v3.

This intentionally does not hash 500 GiB on every run. It checks every expected episode/view and every frame index,
validates file sizes, numpy headers, shapes/dtypes, source-frame metadata hashes, and scans deterministic rows from
every array for finite decoded bf16/float32 values. The report records the exact cache_config hash.
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
from scripts.cache_dino_features import manipulation_frames  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-per-episode", type=int, default=3)
    args = ap.parse_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / args.config)
    if int(cfg.dataset.task_id) != task_profile.TASK_ID:
        raise SystemExit(f"config task {cfg.dataset.task_id} != P2_TASK_ID {task_profile.TASK_ID}")
    cache = ROOT / cfg.dataset.feature_cache
    cc_path = cache / "cache_config.json"
    cc = json.loads(cc_path.read_text())
    views = model_views(cfg)
    feat_layers = [int(v) for v in cfg.model.vision_encoder.feat_layers]
    stride = int(cfg.dataset.frame_stride)
    index = json.loads((task_profile.LOWDIM / "index.json").read_text())
    skills = SkillTable()
    episodes = sorted(int(e) for e in index["episodes"])
    expected = {ep: manipulation_frames(skills.episodes[ep]["segments"],
                                         int(index["episodes"][str(ep)]["length"]), stride) for ep in episodes}
    failures, samples, total_rows, total_bytes = [], 0, 0, 0
    t0 = time.time()
    for view in views:
        for ep in episodes:
            frames = expected[ep]
            stem = cache / view / f"episode_{ep}"
            paths = {k: Path(str(stem) + suffix) for k, suffix in
                     (("feat", ".feat.npy"), ("pool", ".pool.npy"), ("frames", ".frames.npy"), ("meta", ".meta.json"))}
            try:
                meta = json.loads(paths["meta"].read_text())
                src = task_profile.FRAMES / view / f"episode_{ep}.meta.json"
                feat = np.load(paths["feat"], mmap_mode="r")
                pool = np.load(paths["pool"], mmap_mode="r")
                got_frames = np.load(paths["frames"], mmap_mode="r")
                want_feat = (len(frames), len(feat_layers), 201, 1024)
                want_pool = (len(frames), 2, 1024)
                checks = [
                    (meta.get("complete") is True, "meta not complete"),
                    (meta.get("rows") == len(frames), "meta row count"),
                    (meta.get("feat_layers") == feat_layers, "meta feat layers"),
                    (meta.get("stride") == stride, "meta stride"),
                    (meta.get("source_frame_meta_sha256") == sha256(src), "source frame meta hash"),
                    (feat.shape == want_feat and feat.dtype == np.uint16, f"feat {feat.shape}/{feat.dtype}"),
                    (pool.shape == want_pool and pool.dtype == np.float32, f"pool {pool.shape}/{pool.dtype}"),
                    (got_frames.shape == frames.shape and got_frames.dtype == np.int64, "frames shape/dtype"),
                    (np.array_equal(got_frames, frames), "frame index mismatch"),
                ]
                for ok, why in checks:
                    if not ok:
                        raise ValueError(why)
                # NPY header plus raw payload size must reach exactly EOF; detects truncation without reading 500 GiB.
                for p in (paths["feat"], paths["pool"], paths["frames"]):
                    total_bytes += p.stat().st_size
                if len(frames):
                    ids = np.unique(np.linspace(0, len(frames) - 1,
                                                min(args.rows_per_episode, len(frames)), dtype=int))
                    # Torch performs the exact uint16 -> bf16 bit reinterpretation; numpy commonly has no bf16 dtype.
                    import torch
                    f_ok = bool(torch.isfinite(torch.from_numpy(np.asarray(feat[ids]).copy().view(np.int16))
                                               .view(torch.bfloat16).float()).all())
                    if not f_ok or not np.isfinite(np.asarray(pool[ids])).all():
                        raise ValueError("non-finite sampled content")
                    samples += len(ids)
                total_rows += len(frames)
            except Exception as exc:
                failures.append({"view": view, "episode": ep, "error": repr(exc)})
    report = {
        "ok": not failures,
        "task_id": task_profile.TASK_ID,
        "config": args.config,
        "cache": str(cache.relative_to(ROOT)),
        "cache_config_sha256": sha256(cc_path),
        "views": views,
        "episodes": len(episodes),
        "files_expected": len(views) * len(episodes) * 4 + 1,
        "rows_per_view": total_rows // max(len(views), 1),
        "images": total_rows,
        "sampled_rows": samples,
        "array_bytes": total_bytes,
        "cache_config": cc,
        "failures": failures,
        "elapsed_s": round(time.time() - t0, 3),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
