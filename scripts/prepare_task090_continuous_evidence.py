#!/usr/bin/env python3
"""Prepare exact 30Hz TRAIN RGB-D windows, never outcome labels or admission.

Existing image/depth payloads are verified and referenced, not copied. Each raw
stream is traversed once per episode for missing frames: inter-frame compression
may require decoding cached timestamps as dependencies, but they are never
converted, saved, or substituted for verified cache payloads. No DINO is loaded.
The destination must be new; partial data and failure metadata are retained.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

import av
import numpy as np
from PIL import Image
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from prepare_task090_grasp_evidence import (
    DATA, LOWDIM, RGB, DEPTH, POSE, REVISION, codec, dequantize_depth,
    sha256, video_reference, write_json,
)
# The existing helper sets two-thread defaults; this sequential exporter uses one.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/validation/task090_completion_20261003_0156/continuous_train"
DEFAULT_OUT = BASE / "extracted"
WINDOW_STARTS = {18000: 480, 18001: 870}
FPS = 30
CACHES = (
    ROOT / "outputs/validation/task090_takeover_20261002_1319/rigid_sampling_15fps",
    ROOT / "outputs/validation/task090_first_grasp_evidence_20261001/extracted",
)
# Only these already-verified TRAIN producers may supply cached sensor payloads.
PAYLOAD_ROOTS = CACHES + (ROOT / "outputs/validation/task090_onboard_grasp_20261002/dense_train_v1",)
WRAPPER = ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/envs/lerobot_data_wrapper.py"
DATA_WRAPPER = WRAPPER.with_name("data_wrapper.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_window(raw, segment, start, length):
    require(raw["skill_description"] == ["pick up from"] and
            raw["object_id"] == [["trash_can", "floors"]], "Wrong source annotation")
    lo, hi = raw["frame_duration"]
    require((lo, hi) == (segment["start"], segment["end"]), "Raw/cached annotation disagreement")
    require(segment["objects"] == raw["object_id"] and segment["skill"] == "pick up from",
            "Annotation identity disagreement")
    require(lo == start and 0 < lo < hi and hi + 60 <= length, "Window cannot be completed without truncation")
    return list(range(start, hi + 60))


def exact_rows(rows, episode, wanted, dataset_from_index):
    require([r["frame_index"] for r in rows] == wanted, "Missing, duplicate or unordered lowdim rows")
    for row in rows:
        frame = row["frame_index"]
        require(row["episode_index"] == episode and row["index"] == dataset_from_index + frame,
                "Wrong episode/global index")
        require(math.isfinite(row["timestamp"]) and abs(row["timestamp"] - frame / FPS) < 1e-4,
                "Lowdim timestamp mismatch")
    return {r["frame_index"]: r for r in rows}


def checked_state_action(row, prior, states, actions):
    frame = row["frame_index"]
    state = np.asarray(row["observation.state"], dtype=np.float32)
    pose = np.asarray(row[POSE], dtype=np.float32)
    action = np.asarray(prior["action"], dtype=np.float32)
    require(prior["frame_index"] == frame - 1 and prior["episode_index"] == row["episode_index"],
            "Previous action must be from same episode at frame-1")
    require(state.shape == (61,) and pose.shape == (7,) and action.shape == (23,), "Lowdim dimension mismatch")
    require(np.isfinite(state).all() and np.isfinite(pose).all() and np.isfinite(action).all(), "Nonfinite lowdim")
    require(np.array_equal(state, states[frame]) and np.array_equal(action, actions[frame - 1]),
            "Raw parquet differs from hashed state/action cache")
    require(abs(np.linalg.norm(pose[3:]) - 1) < .001 and abs(np.linalg.norm(state[45:49]) - 1) < .001,
            "Quaternion is not normalized")
    return state, pose, action


def action_pairing_contract(source):
    """Fail closed if the inspected writer no longer has obs[t], action[t] pairs."""
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process_traj_to_dataset"]
    require(len(functions) == 1, "Action writer function missing/ambiguous")
    assignments = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "frame" for t in n.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict), "Action frame writer changed")
    frame = assignments[0].value
    pairs = list(zip(frame.keys, frame.values))
    action = [v for k, v in pairs if isinstance(k, ast.Constant) and k.value == "action"]
    obs = [v for k, v in pairs if k is None]
    expected_action = ast.parse('traj_step["action"]', mode="eval").body
    expected_obs = ast.parse('traj_data[frame_idx - 1]["obs"]', mode="eval").body
    require(len(action) == len(obs) == 1 and ast.dump(action[0]) == ast.dump(expected_action) and
            ast.dump(obs[0]) == ast.dump(expected_obs), "Unverified action-to-observation alignment")
    return {"previous_action_row_offset": -1, "dimension": 23,
            "semantics": "recorded TRAIN demonstration environment command at parquet frame-1; never replayed",
            "writer_pairing": "dataset row f contains obs[f] paired with the command leading to obs[f+1]",
            "actuator_feedback_verified": False,
            "limitation": "source writer semantics plus raw/cached action equality, not real-robot motor telemetry"}


def pts_error(pts, target):
    require(pts is not None and math.isfinite(pts), "Missing/nonfinite source PTS")
    error = abs(pts - target)
    require(error < 1e-6, f"Missing exact frame / PTS mismatch: wanted {target}, got {pts}")
    return error


class MissingFrameReader:
    """One seek and one forward decode pass. Only missing frames are materialized."""
    def __init__(self, path, offset, missing, kind, counts):
        require(bool(missing) and missing == sorted(set(missing)), "Missing-frame list must be strictly ordered")
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = 1
        require(float(self.stream.average_rate) == FPS, "Source stream is not 30fps")
        self.container.seek(int(max(0., offset + missing[0] / FPS - .5) / self.stream.time_base),
                            stream=self.stream, backward=True)
        self.iterator = iter(self.container.decode(self.stream))
        self.offset, self.kind, self.counts = offset, kind, counts
        self.last_request = -1
        self.last_pts = -math.inf
        self.counts[f"{kind}_stream_opens"] += 1
        self.counts[f"{kind}_seeks"] += 1

    def close(self):
        self.container.close()

    def get(self, index):
        require(index > self.last_request, "A decoded timestamp must not be requested twice")
        self.last_request = index
        target = self.offset + index / FPS
        for frame in self.iterator:
            require(frame.pts is not None, "Decoder returned a frame without PTS")
            pts = float(frame.pts * self.stream.time_base)
            require(pts > self.last_pts, "Nonmonotonic source PTS")
            self.last_pts = pts
            self.counts[f"{self.kind}_compressed_frames_decoded"] += 1
            if pts < target - 1e-6:
                self.counts[f"{self.kind}_dependency_or_preroll_frames"] += 1
                continue
            pts_error(pts, target)
            self.counts[f"{self.kind}_missing_frames_materialized"] += 1
            if self.kind == "rgb":
                return frame.to_ndarray(format="rgb24"), pts, frame.format.name
            # Exact stride-aware gray12le conversion from the existing depth decoder.
            converted = frame.reformat(format="gray12le")
            plane = converted.planes[0]
            q = np.frombuffer(plane, dtype="<u2", count=converted.height * (plane.line_size // 2))
            q = q.reshape(converted.height, plane.line_size // 2)[:, :converted.width].copy()
            require(q.max() <= 4095, "Depth is not unscaled 12-bit code")
            return q, pts, frame.format.name
        raise ValueError(f"Source frame absent: {self.kind}, frame={index}, target={target}")


def metric_depth(q, params):
    require(q.dtype == np.uint16 and q.ndim == 2 and int(q.max()) <= 4095, "Invalid log-depth code")
    depth = dequantize_depth(q, **params, output_unit="m").squeeze(-1)
    reference = np.exp(q.astype(np.float64) / 4095 *
                       (math.log(params["depth_max"] + params["shift"]) -
                        math.log(params["depth_min"] + params["shift"])) +
                       math.log(params["depth_min"] + params["shift"])) - params["shift"]
    reference = np.clip(reference, params["depth_min"], params["depth_max"])
    error = float(np.max(np.abs(depth - reference)))
    require(depth.dtype == np.float32 and depth.shape == q.shape and np.isfinite(depth).all() and error < 1e-5,
            "Official metric depth failed independent log-codec check")
    return depth, error


def validate_cached_path(path, digest, roots=PAYLOAD_ROOTS):
    path = Path(path).resolve()
    require(any(path.is_relative_to(root.resolve()) for root in roots), "Unapproved cached sensor producer")
    require(path.is_file() and sha256(path) == digest, f"Cached payload SHA mismatch: {path}")
    return path


def merge_cached(cache, record, origin):
    require(record["split"] == "train" and record.get("physical_success_label") is None,
            "Non-TRAIN or outcome-labeled cache is not input")
    key = record["episode"], record["frame_index"]
    if key in cache:
        prior = cache[key][0]
        require(all(prior[k] == record[k] for k in ("rgb_sha256", "depth_m_sha256")),
                f"Conflicting cached payloads: {key}")
        cache[key][1].append(str(origin))
    else:
        cache[key] = (record, [str(origin)])


class Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.process = psutil.Process()
        self.process.cpu_affinity([min(self.process.cpu_affinity())])
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        self.peak_ram = self.peak_cpu = 0.
        self.counts = Counter()
        psutil.cpu_percent(interval=None)

    def guard(self):
        cpu, ram = psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
        self.peak_cpu, self.peak_ram = max(self.peak_cpu, cpu), max(self.peak_ram, ram)
        require(cpu < 85 and ram < 85, f"CPU/RAM85 guard: CPU={cpu}, RAM={ram}")
        require(time.monotonic() - self.started < 900, "CPU extraction exceeded 900 seconds")
        require(self.process.memory_info().rss < 2 * 1024 ** 3, "CPU extraction exceeded 2GiB RSS")
        require(not torch.cuda.is_initialized(), "CUDA must remain uninitialized")

    def snapshot(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return dict(pid=os.getpid(), wall_seconds=time.monotonic() - self.started,
                    cpu_seconds=usage.ru_utime + usage.ru_stime, peak_rss_mib=usage.ru_maxrss / 1024,
                    peak_system_cpu_percent=self.peak_cpu, peak_system_ram_percent=self.peak_ram,
                    cpu_affinity=self.process.cpu_affinity(), torch_threads=torch.get_num_threads(),
                    torch_interop_threads=torch.get_num_interop_threads(), arrow_threads=pa.cpu_count(),
                    decoder_threads=1, encoder_threads=1, cuda_initialized=torch.cuda.is_initialized(),
                    cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], process_io=self.process.io_counters()._asdict(),
                    counts=dict(self.counts))


def identity(path):
    path = Path(path)
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), f"Source changed during hash: {path}")
    return dict(sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns, inode=after.st_ino,
                full_payload_rehashed=True)


def run(out, budget):
    info_path, split_path = DATA / "meta/info.json", DATA / "split_task_090.json"
    index_path, segments_path = LOWDIM / "index.json", DATA / "skill_segments_task090.json"
    info, split, index, segments = [json.loads(p.read_text()) for p in (info_path, split_path, index_path, segments_path)]
    require(index["revision"] == segments["dataset_revision"] == REVISION, "Dataset revision mismatch")
    require(index["split_sha256"] == segments["split_sha256"] == sha256(split_path), "Split SHA mismatch")
    train, val = set(split["train_episode_indices"]), set(split["validation_episode_indices"])
    require(train.isdisjoint(val) and set(WINDOW_STARTS) <= train, "Episodes must be TRAIN only")
    enc = info["features"][DEPTH]["info"]
    require(info["fps"] == enc["video.fps"] == info["features"][RGB]["info"]["video.fps"] == FPS,
            "Source FPS mismatch")
    require(enc["video.pix_fmt"] == "gray12le" and enc["video.use_log"], "Wrong depth codec")
    params = {k: enc["video." + k] for k in ("depth_min", "depth_max", "shift", "use_log", "pix_fmt")}
    contract = action_pairing_contract(WRAPPER.read_text())
    sources = {info_path, split_path, index_path, segments_path, Path(__file__).resolve(),
               ROOT / "scripts/test_task090_continuous_evidence.py", WRAPPER, DATA_WRAPPER,
               ROOT / "scripts/cache_behavior_lowdim.py", ROOT / "scripts/prepare_task090_grasp_evidence.py",
               ROOT / "scripts/prepare_task090_depth_samples.py", Path(codec.__file__)}
    metadata, plans, cache, old_provenance = {}, [], {}, []
    for path in sorted((DATA / "meta/episodes/chunk-090").glob("*.parquet")):
        rows = pq.read_table(path, filters=[("episode_index", "in", list(WINDOW_STARTS))], use_threads=False).to_pylist()
        if rows:
            sources.add(path)
        for row in rows:
            require(row["episode_index"] not in metadata, "Duplicate episode metadata")
            metadata[row["episode_index"]] = row
    require(set(metadata) == set(WINDOW_STARTS), "Missing metadata")
    for ep, start in WINDOW_STARTS.items():
        meta = metadata[ep]
        require(meta["task_index"] == 90 and index["episodes"][str(ep)]["split"] == "train" and
                segments["episodes"][str(ep)]["split"] == "train", "Wrong task or split")
        annotation_path = DATA / meta["annotation_path"]
        annotation = json.loads(annotation_path.read_text())
        raw = next(s for s in annotation["skill_annotation"] if s["skill_description"] == ["pick up from"]
                   and s["object_id"] == [["trash_can", "floors"]])
        seg = next(s for s in segments["episodes"][str(ep)]["segments"] if s["skill_idx"] == raw["skill_idx"])
        require(meta["length"] == index["episodes"][str(ep)]["length"] == segments["episodes"][str(ep)]["length"],
                "Episode lengths differ")
        frames = checked_window(raw, seg, start, meta["length"])
        plans.append(dict(episode=ep, frames=frames, sample_count=len(frames),
                          requested_window_half_open=[frames[0], frames[-1] + 1],
                          first_grasp_interval_half_open=raw["frame_duration"], annotation_source=str(annotation_path),
                          raw_skill_annotation=raw, physical_success_label=None))
        sources.add(annotation_path)
    wanted_keys = {(p["episode"], f) for p in plans for f in p["frames"]}
    for base in CACHES:
        manifest_path = base / ("provenance.json" if (base / "provenance.json").exists() else "manifest.json")
        report = json.loads(manifest_path.read_text())
        frames_path = base / "frames.jsonl"
        require(report["dataset_revision"] == REVISION and sha256(frames_path) == report["frame_manifest_sha256"],
                "Frozen frame manifest SHA/revision mismatch")
        old_provenance.append(report["source_provenance"])
        sources.update((manifest_path, frames_path))
        for text in frames_path.read_text().splitlines():
            record = json.loads(text)
            if (record["episode"], record["frame_index"]) in wanted_keys:
                # Verify even duplicate cached rows; conflicts are not silently replaced.
                for key in ("rgb", "depth_m"):
                    validate_cached_path(record[key], record[key + "_sha256"])
                    budget.counts["cached_payload_hash_checks_preflight"] += 1
                merge_cached(cache, record, frames_path)
    missing_count = len(wanted_keys - cache.keys())
    per_new = 720 * 720 * (3 + 4) + 65536
    # Conservative uncompressed RGB+float depth, overhead, previews/manifests and margin.
    estimate = int((missing_count * per_new + len(wanted_keys) * 65536 + 64 * 1024**2) * 1.2)
    free = shutil.disk_usage(out).free
    require(free > estimate + 1024**3, "Insufficient disk space for continuous evidence")
    disk = dict(missing_rgbd_frames=missing_count, reused_rgbd_frames=len(cache), estimated_max_new_bytes=estimate,
                free_bytes_before=free, reserve_bytes=1024**3, cached_payloads_copied=False)
    for p in plans:
        p["reused_rgbd_frames"] = sum((p["episode"], f) in cache for f in p["frames"])
        p["missing_rgbd_frames"] = p["sample_count"] - p["reused_rgbd_frames"]
    write_json(out / "extraction_plan.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), plans=plans,
               sample_count=len(wanted_keys), sample_fps=FPS, frame_gap=1, disk=disk,
               cpu_only=True, no_dino=True, physical_success_label=None, annotation_is_not_outcome=True))
    print(json.dumps(dict(stage="preflight", samples=len(wanted_keys), disk=disk)), flush=True)
    for name in ("DOWNLOAD_PROVENANCE_task0090_head_depth.json", "DOWNLOAD_PROVENANCE_task0090_rgb.json"):
        path = DATA / name
        require(json.loads(path.read_text())["revision"] == REVISION, "Download revision mismatch")
        sources.add(path)
    for ep, meta in metadata.items():
        sources.update(video_reference(meta, info, key)[0] for key in (RGB, DEPTH))
        sources.add(DATA / info["data_path"].format(chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]))
        sources.update(LOWDIM / f"episode_{ep}_{key}.npy" for key in ("state", "action"))
    provenance = {}
    for path in sorted(sources):
        budget.guard()
        provenance[str(path)] = identity(path)
        for prior in old_provenance:
            if str(path) in prior and path.is_relative_to(DATA):
                prior_sha = prior[str(path)].get("sha256", prior[str(path)].get("prior_sha256"))
                require(prior_sha == provenance[str(path)]["sha256"], f"Original dataset source SHA changed: {path}")
    write_json(out / "source_provenance.json", provenance)
    contract.update(writer_source=str(WRAPPER), writer_source_sha256=provenance[str(WRAPPER)]["sha256"],
                    collector_source=str(DATA_WRAPPER), collector_source_sha256=provenance[str(DATA_WRAPPER)]["sha256"],
                    collector_reference="DataWrapper.step: env.step(action), then _record_step_trajectory(action, next_obs, ...)")
    previews, checks = [], []
    manifest_path = out / "frames.jsonl"
    with manifest_path.open("x") as sensor_out:
        for plan in plans:
            budget.guard()
            ep, frames = plan["episode"], plan["frames"]
            meta = metadata[ep]
            states = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r", allow_pickle=False)
            actions = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r", allow_pickle=False)
            for kind, arr, dim in (("state", states, 61), ("action", actions, 23)):
                require(arr.shape == (meta["length"], dim) and arr.dtype == np.float32, "Cached lowdim shape/dtype mismatch")
                require(hashlib.sha256(arr.tobytes(order="C")).hexdigest() == index["episodes"][str(ep)][kind + "_sha256"],
                        "Lowdim payload hash differs from index")
            data_path = DATA / info["data_path"].format(chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
            wanted = list(range(frames[0] - 1, frames[-1] + 1))
            raw_rows = pq.read_table(data_path, columns=["episode_index", "frame_index", "timestamp", "index", "observation.state", POSE, "action"],
                                    filters=[("episode_index", "=", ep), ("frame_index", "in", wanted)], use_threads=False).to_pylist()
            raw_rows.sort(key=lambda r: r["frame_index"])
            rows = exact_rows(raw_rows, ep, wanted, meta["dataset_from_index"])
            rgb_source, rgb_offset = video_reference(meta, info, RGB)
            depth_source, depth_offset = video_reference(meta, info, DEPTH)
            missing = [f for f in frames if (ep, f) not in cache]
            directory = out / f"episode_{ep}"
            directory.mkdir()
            preview_path = directory / "head_rgb_30fps_preview.mp4"
            max_rgb = max_depth = max_pair = max_codec = max_time = 0.
            sampled_states, sampled_poses, sampled_actions = [], [], []
            with ExitStack() as stack:
                if missing:
                    rgb_reader = MissingFrameReader(rgb_source, rgb_offset, missing, "rgb", budget.counts)
                    stack.callback(rgb_reader.close)
                    depth_reader = MissingFrameReader(depth_source, depth_offset, missing, "depth", budget.counts)
                    stack.callback(depth_reader.close)
                video = stack.enter_context(av.open(str(preview_path), mode="w"))
                stream = video.add_stream("libx264", rate=FPS)
                stream.width = stream.height = 360
                stream.pix_fmt = "yuv420p"
                stream.codec_context.thread_count = 1
                stream.options = {"crf": "24", "preset": "veryfast", "threads": "1"}
                for i, frame in enumerate(frames):
                    budget.guard()
                    row, prior = rows[frame], rows[frame - 1]
                    state, pose, action = checked_state_action(row, prior, states, actions)
                    reused = (ep, frame) in cache
                    if reused:
                        old, origins = cache[(ep, frame)]
                        require(old["rgb_source"] == str(rgb_source) and old["depth_source"] == str(depth_source) and
                                old["lowdim_source"] == str(data_path) and old["dataset_index"] == row["index"],
                                "Cached source identity mismatch")
                        require(np.array_equal(np.asarray(old["state61"], np.float32), state) and
                                np.array_equal(np.asarray(old["camera_pose7"], np.float32), pose), "Cached observation mismatch")
                        for kind, offset in (("rgb", rgb_offset), ("depth", depth_offset)):
                            require(old[kind + "_stream_episode_offset_s"] == offset, "Cached per-stream offset mismatch")
                            pts_error(old[kind + "_requested_pts_s"], offset + frame / FPS)
                        png = validate_cached_path(old["rgb"], old["rgb_sha256"])
                        npy = validate_cached_path(old["depth_m"], old["depth_m_sha256"])
                        with Image.open(png) as image:
                            rgb = np.asarray(image).copy()
                        depth = np.load(npy, allow_pickle=False)
                        rgb_pts, depth_pts = old["rgb_decoded_pts_s"], old["depth_decoded_pts_s"]
                        codec_error, pix = old["codec_reference_max_error_m"], old["depth_original_pixel_format"]
                        require(old["depth_unit"] == "m" and math.isfinite(codec_error) and codec_error < 1e-5,
                                "Cached metric codec proof missing")
                        rgb_sha, depth_sha = old["rgb_sha256"], old["depth_m_sha256"]
                        budget.counts["reused_rgbd_frames"] += 1
                        depth_counts = {k: old[k] for k in ("depth_saturated_min_pixels", "depth_saturated_max_pixels") if k in old}
                    else:
                        origins = []
                        rgb, rgb_pts, _ = rgb_reader.get(frame)
                        q, depth_pts, pix = depth_reader.get(frame)
                        depth, codec_error = metric_depth(q, params)
                        png, npy = directory / f"f{frame:05d}_head_rgb.png", directory / f"f{frame:05d}_head_depth_m.npy"
                        with png.open("xb") as file:
                            Image.fromarray(rgb).save(file, format="PNG", compress_level=3)
                        with npy.open("xb") as file:
                            np.save(file, depth, allow_pickle=False)
                        rgb_sha, depth_sha = sha256(png), sha256(npy)
                        budget.counts["newly_decoded_rgbd_frames"] += 1
                        depth_counts = dict(depth_saturated_min_pixels=int((q == 0).sum()), depth_saturated_max_pixels=int((q == 4095).sum()))
                    require(rgb.shape == (720, 720, 3) and rgb.dtype == np.uint8 and depth.shape == (720, 720) and
                            depth.dtype == np.float32 and np.isfinite(depth).all(), "Invalid sensor dimensions/dtypes")
                    rgb_error = pts_error(rgb_pts, rgb_offset + frame / FPS)
                    depth_error = pts_error(depth_pts, depth_offset + frame / FPS)
                    pair_error = pts_error(rgb_pts - rgb_offset, depth_pts - depth_offset)
                    require(float(depth.min()) >= params["depth_min"] - 1e-5 and float(depth.max()) <= params["depth_max"] + 1e-5,
                            "Metric depth outside codec range")
                    small = np.asarray(Image.fromarray(rgb).resize((360, 360), Image.Resampling.BILINEAR))
                    preview_frame = av.VideoFrame.from_ndarray(small, format="rgb24")
                    for packet in stream.encode(preview_frame):
                        video.mux(packet)
                    lo, hi = plan["first_grasp_interval_half_open"]
                    record = dict(episode=ep, raw_episode_id=meta["raw_episode_id"], train_instance_id=meta["task_instance_id"],
                        split="train", dataset_revision=REVISION, frame_index=frame, dataset_index=row["index"],
                        timestamp_s=row["timestamp"], frame_time_s=frame / FPS, sample_fps=FPS, frame_gap=1 if i else None,
                        rgb=str(png), depth_m=str(npy), rgb_sha256=rgb_sha, depth_m_sha256=depth_sha,
                        rgb_source=str(rgb_source), depth_source=str(depth_source), lowdim_source=str(data_path),
                        rgb_stream_episode_offset_s=rgb_offset, depth_stream_episode_offset_s=depth_offset,
                        rgb_requested_pts_s=rgb_offset + frame / FPS, depth_requested_pts_s=depth_offset + frame / FPS,
                        rgb_decoded_pts_s=rgb_pts, depth_decoded_pts_s=depth_pts,
                        depth_original_pixel_format=pix, depth_unit="m", depth_shape=[720, 720],
                        depth_range_m=[float(depth.min()), float(depth.max())], codec_reference_max_error_m=codec_error,
                        state61=state.tolist(), camera_pose7=pose.tolist(), applied_action_frame=frame - 1,
                        applied_action=action.tolist(), applied_action_verified=True,
                        action_semantics=contract["semantics"], action_source_row_index=prior["index"],
                        action_source_sha256=provenance[str(data_path)]["sha256"],
                        reused_existing_rgbd=reused, reused_from_frame_manifests=origins,
                        preview=str(preview_path), preview_frame_index=i, preview_time_s=i / FPS,
                        annotation_temporal_position_not_outcome="during" if lo <= frame < hi else "after",
                        physical_success_label=None, grasp_relation_label=None, floor_contact_label=None, **depth_counts)
                    # Never inherit feature rows, old previews or model/physical labels from cached rows.
                    sensor_out.write(json.dumps(record, allow_nan=False) + "\n")
                    sensor_out.flush()
                    budget.counts["output_frames"] += 1
                    sampled_states.append(state)
                    sampled_poses.append(pose)
                    sampled_actions.append(action)
                    max_rgb, max_depth, max_pair = max(max_rgb, rgb_error), max(max_depth, depth_error), max(max_pair, pair_error)
                    max_codec, max_time = max(max_codec, codec_error), max(max_time, abs(row["timestamp"] - frame / FPS))
                    if budget.counts["output_frames"] % 100 == 0:
                        print(json.dumps(dict(stage="extracting", episode=ep, frame=frame, resources=budget.snapshot())), flush=True)
                for packet in stream.encode():
                    video.mux(packet)
            with av.open(str(preview_path)) as verify:
                verify.streams.video[0].codec_context.thread_count = 1
                decoded = 0
                for preview_frame in verify.decode(verify.streams.video[0]):
                    pts_error(float(preview_frame.pts * preview_frame.time_base), decoded / FPS)
                    decoded += 1
                require(decoded == len(frames), "Preview dropped or duplicated frames")
            lowdim_out = directory / "observations.npz"
            with lowdim_out.open("xb") as file:
                np.savez(file, frame_index=np.asarray(frames, np.int64), timestamp_s=np.asarray([rows[f]["timestamp"] for f in frames]),
                         state61=np.stack(sampled_states), camera_pose7=np.stack(sampled_poses),
                         applied_action_frame=np.asarray(frames, np.int64) - 1, applied_action=np.stack(sampled_actions))
            previews.append(dict(episode=ep, path=str(preview_path), sha256=sha256(preview_path), frames=decoded,
                                 fps=FPS, duration_s=decoded / FPS, resolution=[360, 360], source_resolution=[720, 720],
                                 cropped=False, overlaid=False, lossy=True, purpose="display_only_never_model_input",
                                 lowdim_npz=str(lowdim_out), lowdim_npz_sha256=sha256(lowdim_out)))
            checks.append(dict(episode=ep, samples=len(frames), frame_gap=1, missing_frames=[], duplicate_frames=[],
                               source_frame_indices_exact=True, previous_actions_raw_and_cache_equal=True,
                               state_raw_and_cache_equal=True, max_lowdim_timestamp_error_s=max_time,
                               max_rgb_pts_error_s=max_rgb, max_depth_pts_error_s=max_depth,
                               max_rgb_depth_relative_pts_error_s=max_pair, max_depth_codec_reference_error_m=max_codec))
            print(json.dumps(dict(stage="episode_complete", episode=ep, samples=len(frames), preview=str(preview_path))), flush=True)
    require(budget.counts["output_frames"] == len(wanted_keys), "Incomplete continuous export")
    emitted = [(r["episode"], r["frame_index"]) for r in map(json.loads, manifest_path.read_text().splitlines())]
    expected = [(p["episode"], f) for p in plans for f in p["frames"]]
    require(emitted == expected, "Final manifest has missing, duplicated or reordered frames")
    for path, record in provenance.items():
        stat = Path(path).stat()
        require((stat.st_size, stat.st_mtime_ns, stat.st_ino) == (record["bytes"], record["mtime_ns"], record["inode"]),
                f"Input changed during extraction: {path}")
    budget.guard()
    output_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    report = dict(status="extraction_complete_not_admission", task_id=90, split="TRAIN only", dataset_revision=REVISION,
        episodes=list(WINDOW_STARTS), sample_count=len(emitted), sample_fps=FPS, source_fps=FPS, continuous=True,
        frame_gap=1, missing_frames=[], frame_manifest=str(manifest_path), frame_manifest_sha256=sha256(manifest_path),
        plans=plans, previews=previews, alignment_checks=checks, disk={**disk, "actual_new_bytes_before_manifest": output_bytes,
                                                                  "free_bytes_after": shutil.disk_usage(out).free},
        source_provenance=str(out / "source_provenance.json"), source_provenance_sha256=sha256(out / "source_provenance.json"),
        codec_parameters=enc, depth_output_unit="m", action_contract=contract,
        state_schema={"state61": "unmodified raw observation.state, exactly equal to hashed lowdim cache",
                      "camera_pose7": "raw robot2cam_pose: camera xyz metres and xyzw quaternion in robot frame; not inverted"},
        decoder_strategy="one seek/forward pass per stream per episode; convert/save missing frames only; cached inter-frame dependencies unavoidable",
        dino={"loaded": False, "computed": False, "cache_modified": False,
              "consumer_hint": "initialize at 18000:480 or 18001:870 using existing exact DINO cache only; do not infer features for odd frames"},
        physical_success_label=None, admitted=False, uses_dev_or_report_inputs=False, uses_hidden_object_truth=False,
        simulator_executed=False, policy_executed=False,
        limitations=["Temporal annotation boundaries are window hints, not grasp/success labels.",
                     "Finger or EEF state is not object pose, physical grasp, floor clearance or success evidence.",
                     "Source command provenance is not measured hardware actuation or a real-robot guarantee.",
                     "Head view may occlude object/hand contact; no physical relation labels are supplied.",
                     "Continuous TRAIN observations are preparation, not predicate admission or task completion.",
                     "Preview is downscaled/lossy display-only; use exact RGB/depth paths for algorithms."],
        resources=budget.snapshot(), environment=dict(python=sys.executable, python_version=sys.version,
                  av=av.__version__, numpy=np.__version__, pyarrow=pa.__version__, torch=torch.__version__, argv=sys.argv),
        created_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "manifest.json", report)
    print(json.dumps(dict(manifest=str(out / "manifest.json"), samples=len(emitted), missing_frames=0,
                          resources=report["resources"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(BASE.resolve()) and out != BASE.resolve(), "Output must be a new child of continuous_train")
    out.mkdir(parents=True, exist_ok=False)
    budget = None
    try:
        budget = Budget()
        budget.guard()
        write_json(out / "execution.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                   argv=sys.argv, output=str(out), cpu_only=True, output_must_not_exist=True))
        run(out, budget)
    except BaseException as error:
        write_json(out / "failure.json", dict(status="failed_partial_evidence_retained", error=repr(error),
                   traceback=traceback.format_exc(), resources=budget.snapshot() if budget else None,
                   physical_success_label=None, admitted=False))
        raise


if __name__ == "__main__":
    main()
