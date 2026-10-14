#!/usr/bin/env python3
"""Cache BEHAVIOR Task 13 RGB frames as 224x224 packed JPEG per episode/camera, PTS-verified.

Sequential HEVC decode starting at each episode's video `from_timestamp`. Every decoded
frame PTS must match frame_index/fps within half a frame, otherwise the episode/camera is
rejected (no silent frame drops). Keeping every `stride`-th frame preserves anchors and
anchor+32 futures whenever both are multiples of the stride. Resumable and atomic.

Output layout: data/frame_cache/<camera>/episode_<id>.jpgpack  (+ .idx.npy uint64 offsets,
+ .meta.json). Read with FramePack in harness/frame_cache.py. Decode only; no policy data.
"""
import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.behavior_data import BehaviorData  # noqa: E402

SIZE = 224
QUALITY = 90


def episode_video(data, row, camera):
    prefix = f"videos/{camera}"
    path = data.root / data.info["video_path"].format(
        video_key=camera, chunk_index=row[f"{prefix}/chunk_index"], file_index=row[f"{prefix}/file_index"])
    return path, float(row[f"{prefix}/from_timestamp"]), int(row["length"])


def cache_one(args):
    episode_id, camera, stride, out_dir = args
    import av
    from PIL import Image
    data = BehaviorData(split="train")
    row = data.episodes[episode_id]
    path, start, length = episode_video(data, row, camera)
    out_dir = Path(out_dir)
    final = out_dir / f"episode_{episode_id}.jpgpack"
    meta_path = final.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("complete") and meta.get("stride") == stride and meta.get("size") == SIZE:
            return {"episode": episode_id, "camera": camera, "skipped": True}
    tmp_bin = final.with_suffix(".jpgpack.tmp")
    fps = data.fps
    tol = 0.5 / fps
    offsets = [0]
    max_err = 0.0
    kept = 0
    t0 = time.time()
    with av.open(str(path)) as container, tmp_bin.open("wb") as out:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        container.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream, backward=True)
        k = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ts = float(frame.pts * frame.time_base)
            expected = start + k / fps
            if ts < expected - tol:
                continue  # still before this episode's first frame
            if ts > expected + tol:
                raise ValueError(f"Missing frame: episode {episode_id} {camera} frame {k} expected {expected:.4f} got {ts:.4f}")
            max_err = max(max_err, abs(ts - expected))
            if k % stride == 0:
                small = frame.reformat(width=SIZE, height=SIZE, format="rgb24", interpolation="AREA").to_ndarray()
                buf = io.BytesIO()
                Image.fromarray(small).save(buf, format="JPEG", quality=QUALITY)
                out.write(buf.getvalue())
                offsets.append(out.tell())
                kept += 1
            k += 1
            if k >= length:
                break
        if k != length:
            raise ValueError(f"Short episode: {episode_id} {camera} decoded {k} of {length}")
    digest = hashlib.sha256()
    with tmp_bin.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    np.save(final.with_suffix(".idx.npy"), np.asarray(offsets, dtype=np.uint64))
    meta = {"episode_index": episode_id, "camera": camera, "video_path": str(path.relative_to(data.root)),
            "from_timestamp": start, "length": length, "stride": stride, "frame_indices": f"range(0,{length},{stride})",
            "count": kept, "size": SIZE, "jpeg_quality": QUALITY, "resample": "swscale AREA from native",
            "max_pts_error_s": max_err, "bytes": offsets[-1], "sha256": digest.hexdigest(),
            "seconds": round(time.time() - t0, 1), "complete": True}
    tmp_bin.replace(final)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return {"episode": episode_id, "camera": camera, "count": kept, "max_pts_error_s": max_err,
            "mib": round(offsets[-1] / 2**20, 1), "seconds": meta["seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", nargs="+", default=["observation.rgb.zed_link_camera_0"])
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "data/frame_cache")
    parser.add_argument("--episodes", type=int, nargs="*", help="Subset for testing")
    args = parser.parse_args()
    data = BehaviorData(split="train")
    episodes = args.episodes or sorted(data.episodes)
    tasks = []
    for camera in args.cameras:
        (args.out / camera).mkdir(parents=True, exist_ok=True)
        tasks.extend((ep, camera, args.stride, str(args.out / camera)) for ep in episodes)
    print(f"{len(tasks)} episode/camera jobs, stride {args.stride}, {args.workers} workers, cpus {sorted(os.sched_getaffinity(0))}", flush=True)
    log = (args.out / "cache_log.jsonl").open("a")
    done = 0
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(cache_one, tasks):
            done += 1
            result["elapsed"] = round(time.time() - t0)
            log.write(json.dumps(result) + "\n"); log.flush()
            if done % 10 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} done, {result}", flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
