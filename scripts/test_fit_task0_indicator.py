#!/usr/bin/env python3
"""Small CPU-only calibration regressions; fixtures are NOT physical labels.

Run with python3 -B -m unittest discover -s scripts -p test_fit_task0_indicator.py -v.
TASK0_INDICATOR_TEST_SOURCE can select the preserved source_before for the
ParserTests, StreakTests and RewardBoundaryTests ONLY. Never run the old main/fit.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = Path(os.environ.get("TASK0_INDICATOR_TEST_SOURCE", ROOT / "scripts/fit_task0_indicator.py"))
spec = importlib.util.spec_from_file_location("task0_fit_under_test", SOURCE)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)
PARAMS = {"min_area": 12, "ring_green": 0.35, "ring_red": 0.6}


def nested():
    # Shape and ranges from the existing Task0 episode 0 metadata, no RGB/rewards.
    return {"task_id": 0, "episodes": {"0": {"split": "train", "length": 1956,
            "segments": [{"skill": skill, "start": start, "end": end}
                         for skill, start, end in [("move to", 0, 265),
                                                  ("pick up from", 265, 1162),
                                                  ("press", 1162, 1434),
                                                  ("place on", 1434, 1776)]]}}}


def rows(frames, ep=0, truth="off", prediction="on"):
    return [{"ep": ep, "frame": f, "truth": truth,
             "g": [(50, 1.0)] if prediction == "on" else [],
             "r": [(50, 1.0)] if prediction == "off" else []} for f in frames]


class ParserTests(unittest.TestCase):
    def parse(self, doc=None, raw=None):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "skill_segments_task000.json"
            path.write_text(raw if raw is not None else json.dumps(doc))
            with patch.object(fit, "DEMO", Path(td)):
                return fit.press_segments()

    def test_real_nested_shape(self):
        self.assertEqual(self.parse(nested()), {0: (1162, 1434)})

    def test_two_episodes_do_not_collapse(self):
        doc = nested()
        doc["episodes"]["4"] = copy.deepcopy(doc["episodes"]["0"])
        doc["episodes"]["4"]["split"] = "validation"
        self.assertEqual(self.parse(doc), {0: (1162, 1434), 4: (1162, 1434)})

    def test_legacy_flat_list_and_wrapper(self):
        segments = [{"episode": 0, "skill": "move to", "start": 0, "end": 3},
                    {"episode": "0", "skill": "press", "start": 3, "end": 6}]
        for doc in (segments, {"segments": segments}, {"task_id": 0, "segments": segments}):
            with self.subTest(doc=doc):
                self.assertEqual(self.parse(doc), {0: (3, 6)})

    def test_reject_invalid_top_level(self):
        for doc in (None, True, 3, "segments", [], {}, {"episodes": {}},
                    {"segments": {}}, {"segments": ["press"]},
                    {"episodes": {}, "segments": []}, {"task_id": 0, "episodes": []}):
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                self.parse(doc)

    def test_reject_missing_or_wrong_task(self):
        for task in (None, True, "0", 0.0, 1):
            doc = nested()
            if task is None:
                del doc["task_id"]
            else:
                doc["task_id"] = task
            with self.subTest(task=task), self.assertRaises(ValueError):
                self.parse(doc)

    def test_reject_episode_id_aliases_and_types(self):
        for ep in ("00", "-1", "1.0", " true", "１", "", True, 0.0, -1, None):
            doc = [{"episode": ep, "skill": "press", "start": 0, "end": 1}]
            with self.subTest(ep=ep), self.assertRaises(ValueError):
                self.parse(doc)
        doc = nested()
        doc["episodes"]["00"] = doc["episodes"]["0"]
        with self.assertRaises(ValueError):
            self.parse(doc)

    def test_reject_duplicate_json_keys(self):
        for raw in ('{"task_id":0,"episodes":{"0":{},"0":{}}}',
                    '{"segments":[],"segments":[]}',
                    '{"segments":[{"episode":0,"start":0,"start":1,"end":3,"skill":"press"}]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.parse(raw=raw)

    def test_reject_bad_episode_records(self):
        for record in (None, [], "press", {}, {"length": 20, "segments": []}):
            doc = nested(); doc["episodes"]["0"] = record
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.parse(doc)
        for key, values in (("length", [None, True, "1956", 1956.0, 0, -1, 1000]),
                            ("split", [None, "dev", "report", True]),
                            ("segments", [None, {}, [], ["press"]])):
            for value in values:
                doc = nested(); doc["episodes"]["0"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.parse(doc)
        for key in ("length", "split", "segments"):
            doc = nested(); del doc["episodes"]["0"][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.parse(doc)

    def test_reject_bad_or_missing_segment_fields(self):
        for key, values in (("start", [None, True, "1162", 1162.0, -1, 1434, 1500]),
                            ("end", [None, True, "1434", 1434.0, -1, 1162, 3000]),
                            ("skill", [None, True, 3, "", " "]),
                            ("episode", [1, None, False])):
            for value in values:
                doc = nested(); doc["episodes"]["0"]["segments"][2][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.parse(doc)
        for key in ("start", "end", "skill"):
            doc = nested(); del doc["episodes"]["0"]["segments"][2][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.parse(doc)

    def test_reject_duplicate_overlapping_unordered_or_missing_press(self):
        originals = nested()["episodes"]["0"]["segments"]
        cases = [originals[:3] + [originals[2]] + originals[3:],
                 list(reversed(originals)),
                 [originals[0], {"skill": "press", "start": 200, "end": 300}],
                 [originals[0]],
                 [{"skill": "press", "start": 0, "end": 2},
                  {"skill": "press", "start": 2, "end": 4}]]
        for segments in cases:
            for flat in (False, True):
                doc = nested(); doc["episodes"]["0"]["segments"] = segments
                if flat:
                    doc = [dict(s, episode=0) for s in segments]
                with self.subTest(segments=segments, flat=flat), self.assertRaises(ValueError):
                    self.parse(doc)

    def test_reject_nonstandard_json_numbers(self):
        with self.assertRaises(ValueError):
            self.parse(raw='{"segments":[{"episode":0,"start":0,"end":NaN,"skill":"press"}]}')


class StreakTests(unittest.TestCase):
    def test_contiguous_cache_proxy_preserves_existing_counts(self):
        self.assertEqual(fit.episode_level(rows([0, 2, 4]), PARAMS)["on_run_before"], 1)
        self.assertEqual(fit.episode_level(rows([0, 2]), PARAMS)["on_run_before"], 0)

    def test_live_stride_requires_five_observed_frames(self):
        self.assertEqual(fit.episode_level(rows([0, 1, 2, 3, 4]), PARAMS, stride=1)["on_run_before"], 1)
        self.assertEqual(fit.episode_level(rows([0, 1, 2, 3]), PARAMS, stride=1)["on_run_before"], 0)

    def test_sparse_or_missing_frames_cannot_accumulate(self):
        for frames in ([0, 100, 200], [0, 2, 6], [0, 4, 6], [0, 1, 2]):
            with self.subTest(frames=frames):
                self.assertEqual(fit.episode_level(rows(frames), PARAMS)["on_run_before"], 0)

    def test_contiguous_suffix_after_gap_can_start_new_run(self):
        self.assertEqual(fit.episode_level(rows([0, 100, 102, 104]), PARAMS)["on_run_before"], 1)

    def test_duplicate_or_reverse_frames_are_rejected_not_sorted(self):
        for frames in ([0, 0, 0], [4, 2, 0], [0, 4, 2], [0, 2, 2, 4], [0, 2, 4, 4]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                fit.episode_level(rows(frames), PARAMS)

    def test_episode_boundaries_do_not_bridge_runs(self):
        result = fit.episode_level(rows([0, 2], ep=0) + rows([4], ep=1), PARAMS)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["on_run_before"], 0)

    def test_interleaved_episodes_preserve_per_episode_order(self):
        a, b = rows([0, 2, 4]), rows([0, 2, 4], ep=1)
        result = fit.episode_level([x for pair in zip(a, b) for x in pair], PARAMS)
        self.assertEqual(result["on_run_before"], 2)

    def test_truth_changes_and_unknown_reset(self):
        for middle in (rows([2], truth="on"), rows([2], prediction="unknown")):
            result = fit.episode_level(rows([0]) + middle + rows([4]), PARAMS)
            self.assertEqual(result["on_run_before"], 0)
        result = fit.episode_level(rows([0, 2, 4], truth="on", prediction="off"), PARAMS)
        self.assertEqual(result["off_run_after"], 1)
        self.assertEqual(result["off_run_before"], 0)

    def test_invalid_stride_or_row_scope(self):
        for stride in (0, -1, True, 2.0, "2"):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                fit.episode_level(rows([0, 2, 4]), PARAMS, stride=stride)
        for key, value in (("ep", True), ("ep", -1), ("frame", 0.0),
                           ("frame", -1), ("truth", "unknown")):
            rs = rows([0]); rs[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                fit.episode_level(rs, PARAMS)

    def test_empty_diagnostics_do_not_claim_an_episode(self):
        self.assertEqual(fit.episode_level([], PARAMS),
                         dict(episodes=0, on_run_after=0, on_run_before=0, off_run_before=0, off_run_after=0))


class RewardBoundaryTests(unittest.TestCase):
    def test_positive_then_negative_rewards_cannot_create_toggle_labels(self):
        # No dataset access: the old implementation reads only this fake table.
        rewards = [0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0]
        table = SimpleNamespace(column=lambda c: {"episode_index": [0] * 12,
                                "frame_index": list(range(12)), "next.reward": rewards}[c])
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data/chunk-000"; data.mkdir(parents=True)
            (data / "fixture.parquet").touch()
            with patch.object(fit, "DEMO", Path(td)), \
                 patch.object(fit, "pq", SimpleNamespace(read_table=lambda *a, **k: table), create=True):
                with self.assertRaises(ValueError):
                    fit.toggle_frames()

    def test_legacy_collector_refuses_permanent_on_labels(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "episode_0.jpgpack").touch()
            fp = SimpleNamespace(length=12, has=lambda f: True, rgb=lambda f: None)
            with patch.object(fit, "CACHE", Path(td)), \
                 patch.object(fit, "FramePack", return_value=fp, create=True), \
                 patch.object(fit, "comps", return_value=([], [])):
                with self.assertRaises(ValueError):
                    fit.collect([0], {0: 2}, {0: (0, 12)}, None)

    def test_even_empty_collection_cannot_be_mistaken_for_trusted_zero_errors(self):
        with self.assertRaises(ValueError):
            fit.collect([], {}, {}, None)


@unittest.skipIf(SOURCE.resolve() != (ROOT / "scripts/fit_task0_indicator.py").resolve(),
                 "Never execute the original main: it would fit thresholds")
class CliTests(unittest.TestCase):
    def test_default_refuses_before_reading_or_fitting(self):
        with tempfile.TemporaryDirectory() as td, \
             patch.object(fit, "DEMO", Path(td)), patch.object(fit, "OUT", Path(td) / "out"), \
             patch.object(fit, "toggle_frames", side_effect=AssertionError("must not read rewards")), \
             patch.object(fit, "press_segments", side_effect=AssertionError("must not read metadata")), \
             patch.object(fit, "collect", side_effect=AssertionError("must not read cache")):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(fit.main([]), 2)
            result = json.loads(stderr.getvalue())
            self.assertFalse(result["admitted"])
            self.assertEqual(result["status"], "blocked")
            self.assertIn("initial state", result["reason"])
            self.assertEqual(list(Path(td).iterdir()), [])

    def cli(self, *args):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        return subprocess.run([sys.executable, "-B", str(SOURCE), *map(str, args)],
                              capture_output=True, text=True, env=env, timeout=10)

    def test_default_cli_exit_and_help(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(json.loads(result.stderr)["admitted"])
        result = self.cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--check-segments", result.stdout)

    def test_metadata_cli_is_not_admission_even_with_verified_true(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "segments.json"
            doc = nested(); doc["verified"] = True
            path.write_text(json.dumps(doc))
            result = self.cli("--check-segments", "--segments", path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "segments_valid")
            self.assertFalse(json.loads(result.stdout)["admitted"])
            self.assertEqual(json.loads(result.stdout)["fit_status"], "blocked")
            self.assertEqual(list(Path(td).iterdir()), [path])
            result = self.cli("--labels", path)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unrecognized arguments", result.stderr)

    def test_invalid_missing_or_ambiguous_cli_input(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "segments.json"
            for text in ("not json", '{"segments":[],"segments":[]}', '{}'):
                path.write_text(text)
                result = self.cli("--check-segments", "--segments", path)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stderr)["status"], "invalid_segments")
            path.unlink()
            result = self.cli("--check-segments", "--segments", path)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(json.loads(result.stderr)["admitted"])
            self.assertEqual(self.cli("--segments", path).returncode, 2)


if __name__ == "__main__":
    unittest.main()
