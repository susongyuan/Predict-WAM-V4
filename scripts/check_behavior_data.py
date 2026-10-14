#!/usr/bin/env python3
"""Check real Task 13 samples, boundaries and previously generated train stats."""

import copy
import json
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
from harness.behavior_data import BehaviorData, DEFAULT_ROOT, _matrix, _Moments


def raises(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


def main():
    train, validation = BehaviorData(), BehaviorData(split="validation")
    assert len(train.episode_ids) == 180 and len(validation.episode_ids) == 20
    assert set(train.episode_ids).isdisjoint(validation.episode_ids)
    first = train.episode_ids[0]
    sample = train.sample(first, 0)
    assert sample["state"].shape == (1, 61) and sample["action_sequence"].shape == (32, 23)
    assert sample["action_mask"].all() and sample["action_sequence"].dtype == np.float32
    for camera in train.cameras:
        assert sample["frames"][camera]["episode_index"] == first
        assert sample["future_frames"][camera]["frame_index"] == 32
        assert np.isclose(sample["future_frames"][camera]["timestamp_s"] - sample["frames"][camera]["timestamp_s"], 32 / 30)
    final_anchor = train.valid_anchors(first)[-1]
    end = train.sample(first, final_anchor)
    assert next(iter(end["future_frames"].values()))["frame_index"] == train.episodes[first]["length"] - 1
    assert end["audit_only"]["next.terminated"].all()  # True flags are not row resets in this real episode.
    raises(ValueError, train.sample, first, final_anchor + 1)
    raises(ValueError, train.sample, first, -1)
    raises(ValueError, train.sample, first, True)
    raises(ValueError, train.sample, validation.episode_ids[0], 0)
    val_sample = validation.sample(validation.episode_ids[0], 7)
    assert val_sample["split"] == "validation" and val_sample["action_sequence"].shape == (32, 23)
    # Exercise a nonzero global parquet origin and a video-file-local episode offset.
    late = train.episode_ids[-1]
    late_sample = train.sample(late, 5)
    assert late_sample["episode_index"] == late and late_sample["anchor"] == 5
    second = train.episode_ids[1]
    camera = "observation.rgb.zed_link_camera_0"
    reference = train.video_reference(second, 0, camera)
    assert reference["timestamp_s"] == train.episodes[second][f"videos/{camera}/from_timestamp"] > 0
    raises(ValueError, train.video_reference, first, train.episodes[first]["length"], camera)
    raises(ValueError, train.video_reference, first, 0, "observation.depth_linear.zed_link_camera_0")
    longer_future = BehaviorData(future_frame_offset=64)
    assert len(longer_future.valid_anchors(first)) == train.episodes[first]["length"] - 64
    raises(ValueError, BehaviorData, future_frame_offset=0)
    raises(ValueError, BehaviorData, split="test")
    raises(ValueError, _matrix, pa.array([[0.0] * 22]), 23)
    raises(ValueError, _matrix, pa.array([[float("nan")] * 23]), 23)
    raises(ValueError, _matrix, pa.array([[None] * 23], type=pa.list_(pa.float32())), 23)

    moments = _Moments(2)
    values = np.array([[1e9, -3.0], [1e9 + 2, 1.0], [1e9 + 4, 8.0]])
    moments.update(values[:1])
    moments.update(values[1:])
    result = moments.result()
    assert result["count"] == 3
    assert np.allclose(result["mean"], values.mean(axis=0))
    assert np.allclose(result["std"], values.std(axis=0))
    assert np.array_equal(result["min"], values.min(axis=0))

    stats_path = DEFAULT_ROOT / "train_stats_task013.json"
    stats = train.read_train_stats(stats_path)
    assert validation.read_train_stats(stats_path) == stats  # Validation uses TRAIN statistics.
    expected_count = sum(train.episodes[index]["length"] for index in train.episode_ids)
    assert stats["provenance"]["num_rows"] == expected_count
    assert stats["provenance"]["episode_indices"] == list(train.episode_ids)
    assert set(map(int, stats["provenance"]["episode_row_counts"])) == set(train.episode_ids)
    assert sum(stats["provenance"]["episode_row_counts"].values()) == expected_count
    for key, sample_key in (("action", "action_sequence"), ("state", "state")):
        stat = stats["behavior1k"][key]
        assert stat["count"] == expected_count
        assert np.all(sample[sample_key] >= stat["min"]) and np.all(sample[sample_key] <= stat["max"])
    with TemporaryDirectory() as directory:
        temporary = Path(directory) / "changed.json"
        split = json.loads(train.split_path.read_text())
        overlap = copy.deepcopy(split)
        overlap["train_episode_indices"].append(split["validation_episode_indices"][0])
        temporary.write_text(json.dumps(overlap))
        raises(ValueError, BehaviorData, split_path=temporary)
        missing = copy.deepcopy(split)
        missing["train_episode_indices"].pop()
        temporary.write_text(json.dumps(missing))
        raises(ValueError, BehaviorData, split_path=temporary)
        raises(ValueError, validation.write_train_stats, temporary)
        assert json.loads(temporary.read_text()) == missing  # Refusal must not overwrite.
        bad_stats = copy.deepcopy(stats)
        bad_stats["provenance"]["episode_indices"].append(validation.episode_ids[0])
        temporary.write_text(json.dumps(bad_stats))
        raises(ValueError, train.read_train_stats, temporary)
        bad_stats = copy.deepcopy(stats)
        bad_stats["provenance"]["split_sha256"] = "wrong-split"
        temporary.write_text(json.dumps(bad_stats))
        raises(ValueError, validation.read_train_stats, temporary)
    print(json.dumps({"status": "PASS", "train_episodes": len(train.episode_ids),
                      "validation_episodes": len(validation.episode_ids), "train_rows": expected_count,
                      "real_samples_checked": 4, "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                      "rgb_decode": "not implemented; three-camera timestamps and file references checked"}))


if __name__ == "__main__":
    main()
