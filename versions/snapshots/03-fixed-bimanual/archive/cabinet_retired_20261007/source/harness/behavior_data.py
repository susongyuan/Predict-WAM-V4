"""BEHAVIOR v3 samples, bounded parquet cache, PTS-checked RGB and train stats."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CHUNK_SIZE = 32
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data/behavior-1k/2026-challenge-demos"
COLUMNS = ["action", "observation.state", "episode_index", "frame_index", "index",
           "task_index", "timestamp", "next.terminated", "next.truncated"]


def decode_reference(reference, fps=30):
    """Decode the nearest video PTS, rejecting errors above half a frame."""
    import av
    target, best = reference["timestamp_s"], None
    with av.open(reference["path"]) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            distance = abs(timestamp - target)
            if best is None or distance < best[0]:
                best = distance, timestamp, frame
            if timestamp >= target:
                break
        if best is None or best[0] > 0.5 / fps + 1e-6:
            raise ValueError(f"No frame matches requested PTS: {reference}")
        return best[2].to_image(), best[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix(array, dimension):
    if (array.null_count or np.any(pc.list_value_length(array).to_numpy() != dimension)):
        raise ValueError(f"Expected non-null {dimension}D vectors")
    values = pc.list_flatten(array)
    if values.null_count:
        raise ValueError("Null vector values are invalid")
    result = values.to_numpy(zero_copy_only=False).reshape(-1, dimension)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite vector values are invalid")
    return result


class BehaviorData:
    def __init__(self, root=DEFAULT_ROOT, *, split="train", split_path=None, future_frame_offset=32):
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        self.root = Path(root).resolve()
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        if type(future_frame_offset) is not int or future_frame_offset < 1:
            raise ValueError("future_frame_offset must be a positive integer")
        self.split, self.future_frame_offset = split, future_frame_offset
        self.info = json.loads((self.root / "meta/info.json").read_text())
        from harness import task_profile  # P2_TASK_ID selects the task; Task 13 keeps DOWNLOAD_PROVENANCE.json
        self.provenance = json.loads((self.root / task_profile.PROVENANCE.name).read_text())
        if self.provenance["task_id"] != task_profile.TASK_ID:
            raise ValueError(f"Provenance task {self.provenance['task_id']} != P2_TASK_ID {task_profile.TASK_ID}")
        self.task_id = self.provenance["task_id"]
        self.split_path = Path(split_path) if split_path else self.root / f"split_task_{self.task_id:03d}.json"
        self.split_sha256 = _sha(self.split_path)
        specification = json.loads(self.split_path.read_text())
        if (self.info["codebase_version"] != "v3.0" or self.info["fps"] != 30
                or self.info["features"]["action"]["shape"] != [23]
                or self.info["features"]["observation.state"]["shape"] != [61]
                or specification["task_id"] != self.task_id):
            raise ValueError("Expected the fixed R1Pro 30 Hz / 23D action / 61D state v3 contract")
        self.fps = self.info["fps"]
        self.cameras = tuple(key for key in self.info["features"] if key.startswith("observation.rgb."))
        metadata_paths = sorted((self.root / "meta/episodes" / specification["chunk"]).glob("*.parquet"))
        rows = [row for path in metadata_paths for row in pq.read_table(path, use_threads=False).to_pylist()]
        self.episodes = {row["episode_index"]: row for row in rows}
        if not rows or len(self.episodes) != len(rows):
            raise ValueError("Missing or duplicate episode metadata")
        self.splits = {}
        for name in ("train", "validation"):
            ids = specification[f"{name}_episode_indices"]
            if not ids or any(type(value) is not int for value in ids) or len(set(ids)) != len(ids):
                raise ValueError("Split episode IDs must be unique integers")
            self.splits[name] = frozenset(ids)
        if (self.splits["train"] & self.splits["validation"]
                or self.splits["train"] | self.splits["validation"] != self.episodes.keys()):
            raise ValueError("Train and validation must partition all task episodes without overlap")
        self.episode_ids = tuple(sorted(self.splits[split]))
        self.files = {}
        for row in rows:
            if (row["task_index"] != self.task_id or row["length"] <= 0
                    or row["dataset_to_index"] - row["dataset_from_index"] != row["length"]):
                raise ValueError("Invalid task/episode bounds in metadata")
            self.files.setdefault(self._data_path(row), []).append(row)
        for path, episodes in self.files.items():
            episodes.sort(key=lambda row: row["dataset_from_index"])
            if (any(left["dataset_to_index"] != right["dataset_from_index"]
                    for left, right in zip(episodes, episodes[1:]))
                    or pq.ParquetFile(path).metadata.num_rows != sum(row["length"] for row in episodes)):
                raise ValueError("This adapter requires complete contiguous episodes within each parquet")
        self.metadata_sha256 = {str(path.relative_to(self.root)): _sha(path) for path in metadata_paths}
        self._cached_path = self._cached_table = None

    def _data_path(self, row):
        return self.root / self.info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])

    def _episode(self, episode_id):
        if type(episode_id) is not int or episode_id not in self.splits[self.split]:
            raise ValueError(f"Episode does not belong to {self.split}")
        return self.episodes[episode_id]

    def valid_anchors(self, episode_id):
        length = self._episode(episode_id)["length"]
        return range(max(0, length - max(CHUNK_SIZE - 1, self.future_frame_offset)))

    def video_reference(self, episode_id, frame_index, camera):
        row = self._episode(episode_id)
        if camera not in self.cameras or type(frame_index) is not int or not 0 <= frame_index < row["length"]:
            raise ValueError("Unknown RGB camera or out-of-episode frame")
        prefix = f"videos/{camera}"
        start, end = row[f"{prefix}/from_timestamp"], row[f"{prefix}/to_timestamp"]
        if (not np.isfinite([start, end]).all() or start < 0
                or not np.isclose((end - start) * self.fps, row["length"], rtol=0, atol=0.02)):
            raise ValueError("Video timestamps disagree with episode length")
        path = self.root / self.info["video_path"].format(
            video_key=camera, chunk_index=row[f"{prefix}/chunk_index"], file_index=row[f"{prefix}/file_index"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return {"path": str(path), "timestamp_s": start + frame_index / self.fps,
                "episode_timestamp_s": frame_index / self.fps, "frame_index": frame_index,
                "episode_index": episode_id, "camera": camera}

    def sample(self, episode_id, anchor):
        row = self._episode(episode_id)
        if type(anchor) is not int or anchor not in self.valid_anchors(episode_id):
            raise ValueError("Anchor would cross the episode boundary; no tail padding is applied")
        path = self._data_path(row)
        local_start = row["dataset_from_index"] - self.files[path][0]["dataset_from_index"] + anchor
        width = max(CHUNK_SIZE - 1, self.future_frame_offset) + 1
        # ponytail: one-file cache bounds RAM; row-group cache if parquet files grow.
        if self._cached_path != path:
            self._cached_table = None
            self._cached_table = pq.ParquetFile(path).read(columns=COLUMNS, use_threads=False)
            self._cached_path = path
        table = self._cached_table.slice(local_start, width).combine_chunks()
        expected_frames = np.arange(anchor, anchor + width)
        if (len(table) != width or np.any(table["episode_index"].to_numpy() != episode_id)
                or np.any(table["task_index"].to_numpy() != self.task_id)
                or not np.array_equal(table["frame_index"].to_numpy(), expected_frames)
                or not np.array_equal(table["index"].to_numpy(), row["dataset_from_index"] + expected_frames)
                or not np.allclose(table["timestamp"].to_numpy(), expected_frames / self.fps, rtol=0, atol=0.001)):
            raise ValueError("Parquet sample does not preserve episode/time boundaries")
        state = _matrix(table["observation.state"].chunk(0), 61)[:1].copy()
        actions = _matrix(table["action"].chunk(0), 23)[:CHUNK_SIZE].copy()
        return {"state": state, "action_sequence": actions, "action_mask": np.ones(CHUNK_SIZE, dtype=bool),
                "frames": {camera: self.video_reference(episode_id, anchor, camera) for camera in self.cameras},
                "future_frames": {camera: self.video_reference(episode_id, anchor + self.future_frame_offset, camera)
                                  for camera in self.cameras},
                "audit_only": {"next.terminated": table["next.terminated"].to_numpy()[:CHUNK_SIZE].copy(),
                               "next.truncated": table["next.truncated"].to_numpy()[:CHUNK_SIZE].copy()},
                "episode_index": episode_id, "anchor": anchor, "split": self.split,
                "task_id": self.task_id, "future_frame_offset": self.future_frame_offset}

    def write_train_stats(self, output):
        """Scan files once, select train rows before vector conversion/aggregation."""
        if self.split != "train":
            raise ValueError("Statistics can only be generated by the train split")
        accumulators = {"action": _Moments(23), "state": _Moments(61)}
        counts = {episode_id: 0 for episode_id in self.episodes}
        file_provenance = []
        train_values = pa.array(sorted(self.splits["train"]), type=pa.int64())
        for path, episodes in sorted(self.files.items()):
            parquet = pq.ParquetFile(path)
            file_ids = {row["episode_index"] for row in episodes}
            for batch in parquet.iter_batches(batch_size=8192, columns=COLUMNS, use_threads=False):
                ids = batch.column("episode_index").to_numpy()
                frames = batch.column("frame_index").to_numpy()
                indices = batch.column("index").to_numpy()
                if np.any(batch.column("task_index").to_numpy() != self.task_id):
                    raise ValueError("Unexpected task in parquet")
                for episode_id in np.unique(ids):
                    if int(episode_id) not in file_ids:
                        raise ValueError("Episode belongs to another parquet file")
                    mask = ids == episode_id
                    expected = np.arange(counts[int(episode_id)], counts[int(episode_id)] + np.count_nonzero(mask))
                    if (not np.array_equal(frames[mask], expected)
                            or not np.array_equal(indices[mask], self.episodes[int(episode_id)]["dataset_from_index"] + expected)):
                        raise ValueError("Missing, repeated or misindexed episode rows")
                    counts[int(episode_id)] += len(expected)
                selected = batch.filter(pc.is_in(batch.column("episode_index"), value_set=train_values))
                if selected.num_rows:
                    accumulators["action"].update(_matrix(selected.column("action"), 23))
                    accumulators["state"].update(_matrix(selected.column("observation.state"), 61))
            file_provenance.append({"path": str(path.relative_to(self.root)), "size_bytes": path.stat().st_size,
                                    "num_rows": parquet.metadata.num_rows})
        if any(counts[index] != row["length"] for index, row in self.episodes.items()):
            raise ValueError("Observed episode counts disagree with metadata")
        train_counts = {str(index): counts[index] for index in sorted(self.splits["train"])}
        provenance = {"repo_id": self.provenance["repo_id"], "revision": self.provenance["revision"],
                      "task_id": self.task_id, "split": "train", "split_sha256": self.split_sha256,
                      "split_file": self.split_path.name, "episode_indices": sorted(self.splits["train"]),
                      "episode_row_counts": train_counts, "num_rows": sum(train_counts.values()),
                      "metadata_sha256": self.metadata_sha256, "data_files": file_provenance,
                      "created_at_utc": datetime.now(timezone.utc).isoformat(),
                      "population": "all train episode rows once, including rows excluded as chunk anchors",
                      "algorithm": "streaming float64 merged centered moments; population std; exact min/max"}
        payload = {"behavior1k": {name: accumulator.result() for name, accumulator in accumulators.items()},
                   "provenance": provenance,
                   "note": "Training normalization statistics, not physical controller safety bounds. No clipping applied."}
        output = Path(output)
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, suffix=".tmp", delete=False) as temporary:
            try:
                json.dump(payload, temporary, indent=2, allow_nan=False)
                temporary.write("\n")
                temporary.close()
                Path(temporary.name).replace(output)
            finally:
                Path(temporary.name).unlink(missing_ok=True)
        return payload

    def read_train_stats(self, path):
        payload = json.loads(Path(path).read_text())
        provenance = payload["provenance"]
        if (provenance["split"] != "train" or provenance["task_id"] != self.task_id
                or provenance["repo_id"] != self.provenance["repo_id"]
                or provenance["split_sha256"] != self.split_sha256
                or provenance["revision"] != self.provenance["revision"]
                or provenance["metadata_sha256"] != self.metadata_sha256
                or provenance["episode_indices"] != sorted(self.splits["train"])
                or provenance["num_rows"] != sum(self.episodes[index]["length"] for index in self.splits["train"])):
            raise ValueError("Normalization statistics do not match this dataset's train split")
        for name, dimension in (("action", 23), ("state", 61)):
            values = payload["behavior1k"][name]
            for key in ("min", "max", "mean", "std"):
                array = np.asarray(values[key])
                if array.shape != (dimension,) or not np.isfinite(array).all():
                    raise ValueError("Invalid normalization dimensions/values")
            if np.any(np.asarray(values["min"]) > values["max"]) or np.any(np.asarray(values["std"]) < 0):
                raise ValueError("Invalid normalization bounds or standard deviation")
        return payload


class _Moments:
    def __init__(self, dimension):
        self.count = 0
        self.mean, self.m2 = np.zeros(dimension), np.zeros(dimension)
        self.minimum, self.maximum = np.full(dimension, np.inf), np.full(dimension, -np.inf)

    def update(self, values):
        values = values.astype(np.float64)
        count = len(values)
        mean = values.mean(axis=0)
        centered = values - mean
        delta = mean - self.mean
        total = self.count + count
        self.m2 += np.sum(centered * centered, axis=0) + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def result(self):
        if not self.count:
            raise ValueError("No training rows were accumulated")
        return {"min": self.minimum.tolist(), "max": self.maximum.tolist(), "mean": self.mean.tolist(),
                "std": np.sqrt(np.maximum(self.m2 / self.count, 0)).tolist(), "count": self.count,
                "constant_dimensions": np.flatnonzero(self.minimum == self.maximum).tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stats-output", type=Path)
    args = parser.parse_args()
    data = BehaviorData(args.root)
    output = args.stats_output or args.root / f"train_stats_task{data.task_id:03d}.json"
    result = data.write_train_stats(output)
    print(json.dumps({"output": str(output), "train_episodes": len(data.episode_ids),
                      "train_rows": result["provenance"]["num_rows"], "split_sha256": data.split_sha256}))
