#!/usr/bin/env python3
"""Create faithful CPU-only GIF previews from a single existing evaluation video."""
import os

# Never initialize GPU libraries. Bound native worker pools and CPU affinity.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import av
from PIL import Image

SOURCE = Path("/home/user/Robotic/project2/outputs/behavior/closed_loop/task0_v3_30k_20260929_s025000_dev10_k1/eval/videos/turning_on_radio_311_0.mp4")
OUTPUT = Path(__file__).resolve().parent
SIZE = (768, 432)
GIF_FRAME_MS = 250
OVERVIEW_STRIDE = 60
DETAIL_START, DETAIL_END, DETAIL_STRIDE = 1796, 2500, 15


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(frame):
    return frame.to_image().resize(SIZE, Image.Resampling.LANCZOS).quantize(
        colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )


def save_gif(name, frames, indices, timestamps, stride, source_fps):
    destination = OUTPUT / name
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(destination) as image:
        encoded_frames = image.n_frames
        duration_ms = 0
        encoded_sizes = set()
        for index in range(encoded_frames):
            image.seek(index)
            duration_ms += image.info.get("duration", 0)
            encoded_sizes.add(image.size)
        assert encoded_sizes == {SIZE}
        assert duration_ms == len(frames) * GIF_FRAME_MS
    coverage_seconds = timestamps[-1] - timestamps[0] + 1 / source_fps
    record = {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "width": SIZE[0],
        "height": SIZE[1],
        "sampled_frames": len(frames),
        "encoded_gif_frames": encoded_frames,
        "frame_duration_ms": GIF_FRAME_MS,
        "playback_duration_seconds": duration_ms / 1000,
        "nominal_gif_fps": 1000 / GIF_FRAME_MS,
        "nominal_source_frame_stride": stride,
        "nominal_speed_multiplier": stride * (1000 / GIF_FRAME_MS) / source_fps,
        "coverage_duration_seconds": coverage_seconds,
        "effective_speed_multiplier_including_endpoint_holds": coverage_seconds / (duration_ms / 1000),
        "source_frame_indices_zero_based": indices,
        "source_timestamps_seconds": timestamps,
        "coverage_source_frames_zero_based_inclusive": [indices[0], indices[-1]],
        "coverage_source_timestamps_seconds": [timestamps[0], timestamps[-1]],
        "note": "Uniform chronological sampling with both coverage endpoints retained; no splicing, retiming within the source, or success claim. Final interval may be shorter than the nominal stride.",
    }
    print(json.dumps({"ready": str(destination), "bytes": record["bytes"]}), flush=True)
    return record


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stat_before = SOURCE.stat()
    input_sha = sha256(SOURCE)
    overview, overview_indices, overview_times = [], [], []
    detail, detail_indices, detail_times = [], [], []
    with av.open(str(SOURCE)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate)
        source_metadata = {
            "path": str(SOURCE),
            "bytes": stat_before.st_size,
            "sha256": input_sha,
            "width": stream.width,
            "height": stream.height,
            "average_rate_rational": str(stream.average_rate),
            "fps": fps,
            "time_base_rational": str(stream.time_base),
            "stream_duration_seconds": float(stream.duration * stream.time_base) if stream.duration is not None else None,
        }
        last_frame = None
        last_index = -1
        last_timestamp = None
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else index / fps
            overview_selected = index % OVERVIEW_STRIDE == 0
            detail_selected = DETAIL_START <= index <= DETAIL_END and (
                (index - DETAIL_START) % DETAIL_STRIDE == 0 or index == DETAIL_END
            )
            if overview_selected or detail_selected:
                image = convert(frame)
                if overview_selected:
                    overview.append(image)
                    overview_indices.append(index)
                    overview_times.append(timestamp)
                if detail_selected:
                    detail.append(image)
                    detail_indices.append(index)
                    detail_times.append(timestamp)
            last_frame, last_index, last_timestamp = frame, index, timestamp
        if last_frame is None:
            raise RuntimeError("Source video has no decodable frames")
        if overview_indices[-1] != last_index:
            overview.append(convert(last_frame))
            overview_indices.append(last_index)
            overview_times.append(last_timestamp)
        source_metadata["decoded_frames"] = last_index + 1
        source_metadata["first_timestamp_seconds"] = overview_times[0]
        source_metadata["last_timestamp_seconds"] = last_timestamp
    assert overview_indices[0] == 0 and overview_indices[-1] == last_index
    products = [save_gif("25k_full_task_overview_8x.gif", overview, overview_indices, overview_times, OVERVIEW_STRIDE, fps)]
    products[0]["coverage"] = "Entire source video, including beginning and final failure/end state."
    if detail:
        products.append(save_gif("25k_action_attempt_frames1796-2500_2x.gif", detail, detail_indices, detail_times, DETAIL_STRIDE, fps))
        products[-1]["coverage"] = "Source video frames 1796–2500 inclusive, zero-based; action-attempt detail only, not evidence of a successful press."
        products[-1]["step_alignment"] = "Frame indices, not verified simulator-step indices. Evaluation reports 3225 simulator steps; no exact per-frame step mapping is assumed."
    stat_after = SOURCE.stat()
    assert (stat_before.st_size, stat_before.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns), "Source video changed during generation"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "source": source_metadata,
        "outputs": products,
        "libraries": {"pyav": av.__version__, "pillow": Image.__version__},
        "compute": {"device": "CPU", "decoder_threads": 1, "native_worker_pool_threads": 1, "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        "editorial_policy": "One original run only. Overview preserves full start-to-finish coverage. Uniform frame sampling is for playback speed only. No claim of task or press success.",
    }
    metadata_path = OUTPUT / "25k_gif_manifest.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(metadata_path), "bytes": metadata_path.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
