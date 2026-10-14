#!/usr/bin/env python3
"""Export two TRAIN first-grasp RGB-D clips for inspection, never outcome labels.

Reuses prepare_task090_depth_samples.decode and the installed official depth codec.
No policy, simulator, detector, action replay, downloads, or GPU calls are made.
The output directory must not exist; partial evidence is retained on failure.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import av
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# Existing, read-only extraction code; its main() is deliberately not invoked.
from prepare_task090_depth_samples import decode, dequantize_depth, codec, DEPTH, RGB

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/behavior-1k/2026-challenge-demos"
LOWDIM = ROOT / "data/lowdim_cache_task090"
OUT = ROOT / "outputs/validation/task090_first_grasp_evidence_20261001"
EPISODES = (18000, 18001)
POSE = "observation.robot2cam_pose.zed_link_camera_0"
REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def rgb_frame(path, timestamp, tolerance):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(int(max(0., timestamp - .5) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * stream.time_base)
            if pts < timestamp - tolerance:
                continue
            if abs(pts - timestamp) > tolerance:
                raise ValueError(f"RGB PTS mismatch: {timestamp} vs {pts}")
            return frame.to_ndarray(format="rgb24"), pts
    raise ValueError(f"RGB frame missing: {timestamp} in {path}")


def video_reference(meta, info, key):
    prefix = "videos/" + key
    path = DATA / info["video_path"].format(video_key=key,
        chunk_index=meta[prefix + "/chunk_index"], file_index=meta[prefix + "/file_index"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, float(meta[prefix + "/from_timestamp"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=int, choices=(2, 5), default=2)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {args.out}")
    start_time = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only extraction requires CUDA to be uninitialized")

    split_path = DATA / "split_task_090.json"
    segment_path = DATA / "skill_segments_task090.json"
    info_path = DATA / "meta/info.json"
    index_path = LOWDIM / "index.json"
    split = json.loads(split_path.read_text())
    segments = json.loads(segment_path.read_text())
    info = json.loads(info_path.read_text())
    index = json.loads(index_path.read_text())
    train, val = set(split["train_episode_indices"]), set(split["validation_episode_indices"])
    assert train.isdisjoint(val) and set(EPISODES) <= train
    assert index["revision"] == segments["dataset_revision"] == REVISION
    assert index["split_sha256"] == segments["split_sha256"] == sha256(split_path)
    source_fps = int(info["fps"])
    enc = info["features"][DEPTH]["info"]
    assert source_fps == enc["video.fps"] == info["features"][RGB]["info"]["video.fps"] == 30
    assert enc["video.pix_fmt"] == "gray12le" and enc["video.use_log"]
    params = {key: enc["video." + key] for key in ("depth_min", "depth_max", "shift", "use_log", "pix_fmt")}
    tolerance = .5 / source_fps
    source_paths = {split_path, segment_path, info_path, index_path, Path(__file__).resolve(),
                    ROOT / "scripts/prepare_task090_depth_samples.py", Path(codec.__file__),
                    ROOT / "harness/wam_behavior.py",
                    ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/envs/lerobot_data_wrapper.py",
                    ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/robots/robot.py",
                    DATA / "DOWNLOAD_PROVENANCE_task0090_head_depth.json",
                    DATA / "DOWNLOAD_PROVENANCE_task0090_rgb.json"}
    metadata = {}
    for p in sorted((DATA / "meta/episodes/chunk-090").glob("*.parquet")):
        rows = pq.read_table(p, filters=[("episode_index", "in", list(EPISODES))], use_threads=False).to_pylist()
        if rows:
            source_paths.add(p)
        for row in rows:
            assert row["episode_index"] not in metadata
            metadata[row["episode_index"]] = row
    assert set(metadata) == set(EPISODES)

    # Resolve the raw interval before selecting frames; no cached closure proxy is used.
    plans = []
    for ep in EPISODES:
        meta = metadata[ep]
        assert meta["task_index"] == 90
        annotation_path = DATA / meta["annotation_path"]
        annotation = json.loads(annotation_path.read_text())
        raw = next(s for s in annotation["skill_annotation"]
                   if s["skill_description"] == ["pick up from"]
                   and s["object_id"] == [["trash_can", "floors"]])
        cached = segments["episodes"][str(ep)]
        assert cached["split"] == index["episodes"][str(ep)]["split"] == "train"
        seg = next(s for s in cached["segments"] if s["skill_idx"] == raw["skill_idx"])
        lo, hi = raw["frame_duration"]
        assert (lo, hi) == (seg["start"], seg["end"])
        assert seg["skill"] == "pick up from" and seg["objects"] == raw["object_id"]
        assert 0 <= lo < hi <= meta["length"] == cached["length"]
        begin, end = max(0, lo - 2 * source_fps), min(meta["length"], hi + 2 * source_fps)
        frames = list(range(begin, end, source_fps // args.fps))
        raw_neighbors = annotation["skill_annotation"][max(0, raw["skill_idx"] - 1):raw["skill_idx"] + 2]
        plans.append({"episode": ep, "raw_episode_id": meta["raw_episode_id"],
                      "train_instance_id": meta["task_instance_id"], "annotation_source": str(annotation_path),
                      "first_grasp_interval_half_open": [lo, hi], "raw_skill_annotation": raw,
                      "neighbor_annotations": raw_neighbors, "requested_window_half_open": [begin, end],
                      "sample_frames": frames, "sample_count": len(frames)})
        source_paths.add(annotation_path)

    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "extraction_plan.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": plans, "fps": args.fps, "padding_seconds_each_side": 2,
        "physical_success_label": None, "annotation_boundaries_are_not_outcome_labels": True})
    records, alignment, previews = [], [], []
    for plan in plans:
        ep = plan["episode"]
        meta = metadata[ep]
        frames = plan["sample_frames"]
        data_path = DATA / info["data_path"].format(chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
        state_path = LOWDIM / f"episode_{ep}_state.npy"
        state = np.load(state_path, mmap_mode="r", allow_pickle=False)
        # cache_behavior_lowdim.py hashes array payload, not the .npy header.
        assert hashlib.sha256(state.tobytes(order="C")).hexdigest() == index["episodes"][str(ep)]["state_sha256"]
        assert state.shape == (meta["length"], 61)
        cols = ["episode_index", "frame_index", "timestamp", "index", "observation.state", POSE]
        table = pq.read_table(data_path, columns=cols,
            filters=[("episode_index", "=", ep), ("frame_index", "in", frames)], use_threads=False)
        rows = sorted(table.to_pylist(), key=lambda r: r["frame_index"])
        assert [r["frame_index"] for r in rows] == frames
        rgb_path, rgb_offset = video_reference(meta, info, RGB)
        depth_path, depth_offset = video_reference(meta, info, DEPTH)
        source_paths.update((data_path, state_path, rgb_path, depth_path))
        ep_dir = args.out / f"episode_{ep}"
        ep_dir.mkdir()
        preview_path = ep_dir / "head_rgb_2fps.mp4" if args.fps == 2 else ep_dir / "head_rgb_5fps.mp4"
        max_lowdim_time_error = max_rgb_error = max_depth_error = max_pair_error = max_codec_error = 0.
        sampled_states, sampled_poses, timestamps = [], [], []
        with av.open(str(preview_path), mode="w") as video:
            stream = video.add_stream("libx264", rate=args.fps)
            stream.width = stream.height = 720
            stream.pix_fmt = "yuv420p"
            stream.codec_context.thread_count = 1
            stream.options = {"crf": "15", "preset": "fast"}
            for i, row in enumerate(rows):
                frame = row["frame_index"]
                assert row["episode_index"] == ep and row["index"] == meta["dataset_from_index"] + frame
                st = np.asarray(row["observation.state"], dtype=np.float32)
                pose = np.asarray(row[POSE], dtype=np.float32)
                assert np.array_equal(st, state[frame])
                assert st.shape == (61,) and pose.shape == (7,) and np.isfinite(st).all() and np.isfinite(pose).all()
                assert abs(np.linalg.norm(pose[3:]) - 1) < .001
                assert abs(np.linalg.norm(st[45:49]) - 1) < .001
                episode_time = frame / source_fps
                time_error = abs(row["timestamp"] - episode_time)
                assert time_error < 1e-4
                rgb, rgb_pts = rgb_frame(rgb_path, rgb_offset + episode_time, tolerance)
                q, depth_pts, original_pixel_format = decode(depth_path, depth_offset + episode_time, tolerance)
                depth = dequantize_depth(q, **params, output_unit="m").squeeze(-1)
                reference = np.exp(q.astype(np.float64) / 4095 *
                    (math.log(params["depth_max"] + params["shift"]) - math.log(params["depth_min"] + params["shift"]))
                    + math.log(params["depth_min"] + params["shift"])) - params["shift"]
                reference = np.clip(reference, params["depth_min"], params["depth_max"])
                codec_error = float(np.max(np.abs(depth - reference)))
                assert rgb.shape == (720, 720, 3) and depth.shape == (720, 720)
                assert depth.dtype == np.float32 and np.isfinite(depth).all() and codec_error < 1e-5
                pair_error = abs((rgb_pts - rgb_offset) - (depth_pts - depth_offset))
                assert pair_error < 1e-6
                png = ep_dir / f"f{frame:05d}_head_rgb.png"
                npy = ep_dir / f"f{frame:05d}_head_depth_m.npy"
                Image.fromarray(rgb).save(png)
                np.save(npy, depth, allow_pickle=False)
                vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(vf):
                    video.mux(packet)
                lo, hi = plan["first_grasp_interval_half_open"]
                records.append({"episode": ep, "raw_episode_id": meta["raw_episode_id"], "split": "train",
                    "train_instance_id": meta["task_instance_id"], "frame_index": frame, "dataset_index": row["index"],
                    "timestamp_s": row["timestamp"], "frame_time_s": episode_time, "preview_frame_index": i,
                    "preview_time_s": i / args.fps, "rgb": str(png), "depth_m": str(npy),
                    "rgb_sha256": sha256(png), "depth_m_sha256": sha256(npy),
                    "rgb_source": str(rgb_path), "depth_source": str(depth_path), "lowdim_source": str(data_path),
                    "rgb_stream_episode_offset_s": rgb_offset, "depth_stream_episode_offset_s": depth_offset,
                    "rgb_requested_pts_s": rgb_offset + episode_time, "rgb_decoded_pts_s": rgb_pts,
                    "depth_requested_pts_s": depth_offset + episode_time, "depth_decoded_pts_s": depth_pts,
                    "depth_original_pixel_format": original_pixel_format, "depth_unit": "m", "depth_shape": [720, 720],
                    "depth_range_m": [float(depth.min()), float(depth.max())],
                    "depth_saturated_min_pixels": int((q == 0).sum()), "depth_saturated_max_pixels": int((q == 4095).sum()),
                    "codec_reference_max_error_m": codec_error, "state61": st.tolist(), "camera_pose7": pose.tolist(),
                    "right_eef_pos_m": st[42:45].tolist(), "right_eef_quat_xyzw": st[45:49].tolist(),
                    "right_gripper_qpos_m": st[49:51].tolist(),
                    "annotation_temporal_position_not_outcome": "before" if frame < lo else "during" if frame < hi else "after",
                    "physical_success_label": None, "grasp_relation_label": None, "floor_contact_label": None})
                sampled_states.append(st)
                sampled_poses.append(pose)
                timestamps.append(row["timestamp"])
                max_lowdim_time_error = max(max_lowdim_time_error, time_error)
                max_rgb_error = max(max_rgb_error, abs(rgb_pts - rgb_offset - episode_time))
                max_depth_error = max(max_depth_error, abs(depth_pts - depth_offset - episode_time))
                max_pair_error = max(max_pair_error, pair_error)
                max_codec_error = max(max_codec_error, codec_error)
            for packet in stream.encode():
                video.mux(packet)
        lowdim_out = ep_dir / "observations.npz"
        np.savez(lowdim_out, frame_index=np.asarray(frames, dtype=np.int64), timestamp_s=np.asarray(timestamps),
                 state61=np.stack(sampled_states), camera_pose7=np.stack(sampled_poses),
                 right_eef_pose7=np.stack(sampled_states)[:, 42:49])
        with av.open(str(preview_path)) as video:
            stream = video.streams.video[0]
            stream.codec_context.thread_count = 1
            decoded_count = sum(1 for _ in video.decode(stream))
            assert decoded_count == len(frames)
        previews.append({"episode": ep, "path": str(preview_path), "sha256": sha256(preview_path),
                         "resolution": [720, 720], "fps": args.fps, "frames": decoded_count,
                         "cropped": False, "overlaid": False, "playback": "real-time sampled at requested fps",
                         "lowdim_npz": str(lowdim_out), "lowdim_npz_sha256": sha256(lowdim_out)})
        alignment.append({"episode": ep, "samples": len(rows), "frame_indices_exact": True,
                          "global_dataset_indices_exact": True, "parquet_equals_cached_state61": True,
                          "max_lowdim_timestamp_error_s": max_lowdim_time_error,
                          "max_rgb_pts_error_s": max_rgb_error, "max_depth_pts_error_s": max_depth_error,
                          "max_rgb_depth_relative_pts_error_s": max_pair_error,
                          "max_depth_codec_reference_error_m": max_codec_error})
        print(json.dumps({"episode_complete": ep, "samples": len(rows), "preview": str(preview_path)}), flush=True)

    frames_path = args.out / "frames.jsonl"
    with frames_path.open("x") as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    provenance = {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                  for p in sorted(source_paths)}
    assert not torch.cuda.is_initialized()
    report = {"status": "extraction_complete_not_admission", "task_id": 90, "split": "TRAIN only",
        "dataset_revision": REVISION, "episodes": list(EPISODES), "sample_count": len(records),
        "sample_fps": args.fps, "source_fps": source_fps, "frame_manifest": str(frames_path),
        "frame_manifest_sha256": sha256(frames_path), "plans": plans, "previews": previews,
        "alignment_checks": alignment, "source_provenance": provenance,
        "codec_parameters": enc, "depth_output_unit": "m", "depth_is_original_log_quantized_data": True,
        "state_schema": {"state61": "unmodified dataset observation.state; runtime slice schema source hashed",
                         "right_eef_pos": [42, 45], "right_eef_quat_xyzw": [45, 49], "right_gripper_qpos": [49, 51],
                         "eef_pose_frame": "robot base, raw observation; no world transformation",
                         "camera_pose7": "raw robot2cam_pose: camera xyz (m) and quaternion xyzw in robot frame; not inverted"},
        "physical_success_label": None, "uses_dev_or_report": False, "uses_hidden_object_truth": False,
        "uses_model_generated_labels": False, "simulator_executed": False, "policy_executed": False,
        "limitations": ["No outcome label is inferred from skill boundary, hand height, gripper or annotation.",
                        "2fps/5fps temporal subsampling can miss short contacts or slips.",
                        "Head view only; occluded hand-object contact and metric floor clearance require independent evidence.",
                        "Depth saturation is counted, not interpreted as a physical relationship.",
                        "Release and stability are outside this first-grasp window; full task and generalization untested.",
                        "This is evidence preparation, not predicate admission or a successful policy rollout."],
        "resources": {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "cuda_initialized": torch.cuda.is_initialized(),
                      "torch_threads": torch.get_num_threads(), "decoder_threads": 1, "arrow_cpu_threads": pa.cpu_count(),
                      "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      "elapsed_seconds": time.monotonic() - start_time},
        "environment": {"python": sys.executable, "python_version": sys.version, "av": av.__version__,
                        "numpy": np.__version__, "pyarrow": pa.__version__, "torch": torch.__version__,
                        "argv": sys.argv}, "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.out / "manifest.json", report)
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "samples": len(records),
                      "resources": report["resources"]}, indent=2))


if __name__ == "__main__":
    main()
