#!/usr/bin/env python3
"""Small CPU-only contract tests; synthetic fixtures are never extraction inputs."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

from collections import Counter
import copy
from fractions import Fraction
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import prepare_task090_continuous_evidence as exporter


class ContinuousEvidenceTests(unittest.TestCase):
    def test_annotation_window_is_half_open_and_not_truncated(self):
        raw = dict(skill_description=["pick up from"], object_id=[["trash_can", "floors"]], frame_duration=[480, 832])
        seg = dict(start=480, end=832, skill="pick up from", objects=raw["object_id"])
        frames = exporter.checked_window(raw, seg, 480, 3425)
        self.assertEqual((frames[0], frames[-1], len(frames)), (480, 891, 412))
        self.assertTrue(np.all(np.diff(frames) == 1))
        with self.assertRaises(ValueError):
            exporter.checked_window(raw, seg, 480, 850)
        with self.assertRaises(ValueError):
            exporter.checked_window(raw, dict(seg, end=831), 480, 3425)

    def test_previous_action_alignment_and_exact_state(self):
        states = np.zeros((4, 61), np.float32)
        states[:, 48] = 1
        actions = np.arange(4 * 23, dtype=np.float32).reshape(4, 23)
        row = {"episode_index": 18000, "frame_index": 2, "observation.state": states[2].tolist(),
               exporter.POSE: [0, 0, 1, 0, 0, 0, 1]}
        prior = dict(episode_index=18000, frame_index=1, action=actions[1].tolist())
        _, _, action = exporter.checked_state_action(row, prior, states, actions)
        np.testing.assert_array_equal(action, actions[1])
        for bad in (dict(prior, frame_index=2), dict(prior, episode_index=18001), dict(prior, action=actions[2].tolist())):
            with self.assertRaises(ValueError):
                exporter.checked_state_action(row, bad, states, actions)
        bad_row = copy.deepcopy(row)
        bad_row["observation.state"][0] = 1
        with self.assertRaises(ValueError):
            exporter.checked_state_action(bad_row, prior, states, actions)

    def test_missing_duplicate_cross_episode_lowdim_refused(self):
        rows = [dict(episode_index=18000, frame_index=i, index=100 + i, timestamp=i / 30) for i in range(3)]
        self.assertEqual(list(exporter.exact_rows(rows, 18000, [0, 1, 2], 100)), [0, 1, 2])
        for bad in (rows[::2], rows + [rows[-1]], [dict(rows[0], episode_index=18001)] + rows[1:]):
            with self.assertRaises(ValueError):
                exporter.exact_rows(bad, 18000, [0, 1, 2], 100)

    def test_pts_requires_exact_frame_not_nearest_or_gap(self):
        self.assertEqual(exporter.pts_error(1.0, 1.0), 0)
        for bad in (None, float("nan"), 1 + 1 / 30, 1 + .001):
            with self.assertRaises(ValueError):
                exporter.pts_error(bad, 1.0)

    def test_official_depth_matches_independent_log_codec(self):
        params = dict(depth_min=.01, depth_max=10., shift=1., use_log=True, pix_fmt="gray12le")
        q = np.asarray([[0, 1, 2048], [4094, 4095, 512]], dtype=np.uint16)
        depth, error = exporter.metric_depth(q, params)
        self.assertEqual(depth.dtype, np.float32)
        self.assertLess(error, 1e-5)
        self.assertAlmostEqual(float(depth[0, 0]), .01, places=5)
        self.assertAlmostEqual(float(depth[1, 1]), 10., places=5)
        with self.assertRaises(ValueError):
            exporter.metric_depth(np.asarray([[4096]], np.uint16), params)

    def test_cache_conflict_and_nontrain_or_label_refused(self):
        record = dict(episode=18000, frame_index=480, split="train", physical_success_label=None,
                      rgb_sha256="a", depth_m_sha256="b")
        cache = {}
        exporter.merge_cached(cache, record, "first")
        exporter.merge_cached(cache, dict(record), "second")
        self.assertEqual(cache[(18000, 480)][1], ["first", "second"])
        for bad in (dict(record, split="validation"), dict(record, physical_success_label=True),
                    dict(record, rgb_sha256="c"), dict(record, depth_m_sha256="d")):
            with self.assertRaises(ValueError):
                exporter.merge_cached(cache, bad, "bad")

    def test_cache_sha_and_allowed_producer_enforced(self):
        exporter.BASE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test_fixture_", dir=exporter.BASE) as directory:
            root = Path(directory)
            path = root / "fixture.bin"
            path.write_bytes(b"synthetic fixture, never evidence")
            digest = exporter.sha256(path)
            self.assertEqual(exporter.validate_cached_path(path, digest, (root,)), path)
            with self.assertRaises(ValueError):
                exporter.validate_cached_path(path, "0" * 64, (root,))
            with self.assertRaises(ValueError):
                exporter.validate_cached_path(path, digest, (root / "unapproved",))

    def test_writer_pairing_changed_mapping_fails_closed(self):
        source = '''def process_traj_to_dataset(traj_data):
    for frame_idx, traj_step in enumerate(traj_data):
        frame = {"action": traj_step["action"], **traj_data[frame_idx - 1]["obs"]}
'''
        self.assertEqual(exporter.action_pairing_contract(source)["previous_action_row_offset"], -1)
        with self.assertRaises(ValueError):
            exporter.action_pairing_contract(source.replace("frame_idx - 1", "frame_idx"))
        with self.assertRaises(ValueError):
            exporter.action_pairing_contract(source.replace('traj_step["action"]', 'traj_step["desired_action"]'))

    def test_sequential_decoder_only_materializes_missing_frames(self):
        calls = []
        class FakeFrame:
            def __init__(self, pts):
                self.pts = pts
                self.format = SimpleNamespace(name="rgb24")
            def to_ndarray(self, format):
                calls.append(self.pts)
                return np.zeros((2, 2, 3), np.uint8)
        reader = exporter.MissingFrameReader.__new__(exporter.MissingFrameReader)
        reader.stream = SimpleNamespace(time_base=Fraction(1, 30))
        reader.iterator = iter(FakeFrame(i) for i in range(5))
        reader.offset = 0.
        reader.kind = "rgb"
        reader.counts = Counter()
        reader.last_request, reader.last_pts = -1, float("-inf")
        reader.get(1)
        reader.get(3)
        self.assertEqual(calls, [1, 3])
        self.assertEqual(reader.counts["rgb_dependency_or_preroll_frames"], 2)
        with self.assertRaises(ValueError):
            reader.get(3)
        with self.assertRaises(ValueError):
            reader.get(8)

    def test_sequential_decoder_missing_middle_frame_is_fatal(self):
        reader = exporter.MissingFrameReader.__new__(exporter.MissingFrameReader)
        reader.stream = SimpleNamespace(time_base=Fraction(1, 30))
        reader.iterator = iter([SimpleNamespace(pts=0), SimpleNamespace(pts=2)])
        reader.offset, reader.kind, reader.counts = 0., "rgb", Counter()
        reader.last_request, reader.last_pts = -1, float("-inf")
        with self.assertRaises(ValueError):
            reader.get(1)
        self.assertNotIn("rgb_missing_frames_materialized", reader.counts)

    def test_cpu_only(self):
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
