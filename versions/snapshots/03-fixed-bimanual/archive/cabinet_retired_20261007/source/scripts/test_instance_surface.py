#!/usr/bin/env python3
"""Deterministic synthetic CPU contracts; no task data or threshold fitting.

Run with the existing NumPy/OpenCV interpreter, e.g.
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/test_instance_surface.py
Optional --json PATH writes a machine-readable execution record, not predictions.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest

# Set before importing NumPy; these tests deliberately use one CPU thread.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from harness.instance_surface import SurfaceThresholds, refine_target_surfaces
from harness.object_perception import Camera

cv2.setNumThreads(1)

# Analytical fixture settings, not a calibration for recordings or real objects.
THRESHOLDS = SurfaceThresholds(
    positive_min=0.6, margin_min=0.25, depth_min_m=0.05, depth_max_m=5.0,
    neighbor_distance_m=1.0, neighbor_color_linf=12, min_component_pixels=2,
)
PROVENANCE = {
    "score_source": "analytically constructed synthetic positive/negative fields",
    "threshold_source": "synthetic fixture specification, not fitted or deployable",
    "alignment_source": "synthetic full-frame grid, floor(v*Gh/H), floor(u*Gw/W)",
    "robot_self_mask_source": "synthetic bool mask, not a calibrated robot model",
}


def scene(h=12, w=12):
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float64)
    camera = Camera(np.eye(3), np.array([0.2, -0.1, 0.7]), w, h)
    positive = np.zeros((h, w), dtype=np.float64)
    negative = np.zeros((h, w), dtype=np.float64)
    return rgb, depth, camera, positive, negative


def refine(inputs, **kwargs):
    kwargs.setdefault("thresholds", THRESHOLDS)
    kwargs.setdefault("provenance", PROVENANCE)
    return refine_target_surfaces(*inputs, **kwargs)


class InstanceSurfaceTests(unittest.TestCase):
    def assert_observed_samples(self, result, inputs):
        _, depth, camera, _, _ = inputs
        self.assertFalse(result["object_presence_verified"])
        self.assertFalse(result["physical_success_verified"])
        self.assertEqual(result["visibility"], "unknown")
        for candidate in result["candidates"]:
            uv, cloud = candidate["pixel_uv"], candidate["cloud"]
            u, v = uv.T
            np.testing.assert_allclose(cloud, camera.backproject(u, v, depth[v, u]), rtol=0, atol=0)
            np.testing.assert_array_equal(result["labels"][v, u], candidate["component_id"])
            sample = np.array(candidate["surface_pixel_uv"])
            member = np.flatnonzero(np.all(uv == sample, axis=1))
            self.assertEqual(len(member), 1)
            np.testing.assert_array_equal(candidate["xyz_base"], cloud[member[0]])
            self.assertEqual(candidate["surface_depth_m"], depth[sample[1], sample[0]])
            self.assertFalse(candidate["object_presence_verified"])
            self.assertFalse(candidate["center_valid"])
            self.assertFalse(candidate["grasp_reference_valid"])
            self.assertNotIn("confidence", candidate)

    def test_disconnected_same_color_objects_do_not_merge(self):
        inputs = scene()
        rgb, _, _, pos, _ = inputs
        # Coarse semantic support covers both dark lookalikes and their background.
        pos[:] = 0.9
        rgb[2:5, 1:4] = 0
        rgb[7:10, 8:11] = 0
        result = refine(inputs)
        self.assertEqual(len(result["candidates"]), 3)  # both dark surfaces + background ambiguity
        boxes = [x["bbox_xyxy"] for x in result["candidates"]]
        self.assertIn([1, 2, 4, 5], boxes)
        self.assertIn([8, 7, 11, 10], boxes)
        self.assertNotEqual(result["labels"][2, 1], result["labels"][7, 8])
        self.assert_observed_samples(result, inputs)

    def test_disconnected_semantic_support_not_top_five(self):
        inputs = scene(16, 16)
        rgb, _, _, pos, _ = inputs
        rgb[:] = 0
        for r, c in ((1, 1), (1, 6), (1, 11), (7, 1), (7, 6), (7, 11)):
            pos[r:r+2, c:c+2] = 0.8
        result = refine(inputs)
        self.assertEqual(len(result["candidates"]), 6)
        self.assertTrue(all(x["pixels"] == 4 for x in result["candidates"]))

    def test_robot_exclusion_splits_connected_same_color_surface(self):
        inputs = scene(10, 10)
        inputs[3][3:6, 1:9] = 0.9
        mask = np.zeros((10, 10), dtype=bool)
        mask[:, 4:6] = True
        self.assertEqual(len(refine(inputs)["candidates"]), 1)
        result = refine(inputs, robot_self_mask=mask)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["diagnostics"]["excluded_robot_pixels"], 6)
        self.assertTrue(np.all(result["labels"][mask] == 0))
        self.assertTrue(all(not np.any(mask[x["pixel_uv"][:, 1], x["pixel_uv"][:, 0]])
                            for x in result["candidates"]))
        self.assert_observed_samples(result, inputs)

    def test_all_robot_support_is_unknown_not_absence(self):
        inputs = scene()
        inputs[3][:] = 0.9
        result = refine(inputs, robot_self_mask=np.ones((12, 12), dtype=bool))
        self.assertEqual(result["status"], "all_depth_support_excluded_by_robot")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["visibility"], "unknown")

    def test_invalid_depth_pixels_are_excluded(self):
        inputs = scene()
        inputs[3][1:7, 1:7] = 0.9
        inputs[1][2, 1:7] = [np.nan, np.inf, -1, 0, 0.05, 5.0]
        result = refine(inputs)
        self.assertEqual(result["diagnostics"]["surface_support_pixels"], 30)
        self.assertTrue(np.all(result["labels"][2, 1:7] == 0))
        self.assert_observed_samples(result, inputs)

    def test_all_invalid_depth_returns_empty(self):
        inputs = scene()
        inputs[3][:] = 0.9
        inputs[1][:] = np.nan
        result = refine(inputs)
        self.assertEqual(result["status"], "no_valid_depth_support")
        self.assertEqual(result["candidates"], [])

    def test_no_margin_support_despite_high_positive_scores(self):
        inputs = scene()
        inputs[3][:] = 0.95
        inputs[4][:] = 0.9
        result = refine(inputs)
        self.assertEqual(result["status"], "no_semantic_support")
        self.assertEqual(result["diagnostics"]["semantic_support_pixels"], 0)
        self.assertEqual(result["candidates"], [])

    def test_negative_field_changes_support(self):
        inputs = scene()
        inputs[3][2:5, 1:4] = inputs[3][7:10, 8:11] = 0.9
        self.assertEqual(len(refine(inputs)["candidates"]), 2)
        inputs[4][7:10, 8:11] = 0.85
        result = refine(inputs)
        self.assertEqual([c["bbox_xyxy"] for c in result["candidates"]], [[1, 2, 4, 5]])

    def test_absolute_positive_gate_and_nonfinite_scores(self):
        inputs = scene()
        inputs[3][:] = 0.4
        inputs[4][:] = -1.0
        self.assertEqual(refine(inputs)["status"], "no_semantic_support")
        inputs[3][:] = 0.9
        inputs[3][:6] = np.nan
        inputs[4][6:] = np.inf
        self.assertEqual(refine(inputs)["status"], "no_semantic_support")

    def test_metric_discontinuity_splits_touching_same_color(self):
        inputs = scene(16, 16)
        inputs[3][4:8, 3:11] = 0.9
        inputs[1][4:8, 7:11] = 2.0
        result = refine(inputs, thresholds=replace(THRESHOLDS, neighbor_distance_m=0.4))
        self.assertEqual([c["bbox_xyxy"] for c in result["candidates"]],
                         [[3, 4, 7, 8], [7, 4, 11, 8]])
        self.assert_observed_samples(result, inputs)

    def test_color_discontinuity_splits_touching_metric_surface(self):
        inputs = scene()
        inputs[3][3:7, 2:10] = 0.9
        inputs[0][3:7, 2:6] = 0
        inputs[0][3:7, 6:10] = 255
        result = refine(inputs)
        self.assertEqual(len(result["candidates"]), 2)
        disabled = refine(inputs, thresholds=replace(THRESHOLDS, neighbor_color_linf=255))
        self.assertEqual(len(disabled["candidates"]), 1)

    def test_diagonal_contact_does_not_create_an_edge(self):
        inputs = scene()
        inputs[3][2:4, 2:4] = inputs[3][4:6, 4:6] = 0.9
        self.assertEqual(len(refine(inputs)["candidates"]), 2)

    def test_coarse_grid_nondivisible_resolution_edges(self):
        rgb, depth, camera, _, _ = scene(7, 11)
        positive = np.zeros((2, 3))
        positive[1, 2] = 0.9
        inputs = rgb, depth, camera, positive, np.zeros((2, 3))
        result = refine(inputs)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["bbox_xyxy"], [8, 4, 11, 7])
        self.assertEqual(result["candidates"][0]["pixels"], 9)
        self.assert_observed_samples(result, inputs)

    def test_single_pixel_row_and_column_resolutions(self):
        thresholds = replace(THRESHOLDS, min_component_pixels=1, neighbor_distance_m=20.0)
        for h, w in ((1, 1), (1, 5), (5, 1)):
            with self.subTest(shape=(h, w)):
                inputs = scene(h, w)
                inputs[3][:] = 0.9
                result = refine(inputs, thresholds=thresholds)
                self.assertEqual(len(result["candidates"]), 1)
                self.assertEqual(result["candidates"][0]["pixels"], h*w)
                self.assert_observed_samples(result, inputs)

    def test_resolution_and_input_contracts_reject_silent_resampling(self):
        rgb, depth, camera, pos, neg = scene()
        for inputs, options in (
            ((rgb, depth[:-1], camera, pos, neg), {}),
            ((rgb, depth, Camera(np.eye(3), np.zeros(3), 10, 10), pos, neg), {}),
            ((rgb, depth, camera, pos.ravel(), neg.ravel()), {}),
            ((rgb, depth, camera, np.ones((13, 1)), np.zeros((13, 1))), {}),
            ((rgb, depth, camera, pos, neg[:5]), {}),
            ((rgb.astype(float)/255, depth, camera, pos, neg), {}),
            ((rgb, depth, camera, pos, neg), {"robot_self_mask": np.zeros((12, 12), np.uint8)}),
            ((rgb, depth, camera, pos, neg), {"robot_self_mask": np.zeros((10, 10), bool)}),
            ((rgb, depth, camera, pos, neg), {"max_image_pixels": 100}),
        ):
            with self.subTest(options=options.keys(), depth_shape=np.shape(inputs[1])):
                with self.assertRaises(ValueError):
                    refine(inputs, **options)

    def test_thresholds_and_provenance_are_required(self):
        for updates in ({"margin_min": 0}, {"margin_min": np.nan},
                        {"depth_max_m": 0.04}, {"neighbor_distance_m": 0},
                        {"neighbor_color_linf": 256}, {"min_component_pixels": True}):
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):
                    replace(THRESHOLDS, **updates)
        inputs = scene()
        for missing in ("score_source", "threshold_source", "alignment_source"):
            provenance = dict(PROVENANCE)
            del provenance[missing]
            with self.assertRaises(ValueError):
                refine(inputs, provenance=provenance)
        provenance = dict(PROVENANCE)
        del provenance["robot_self_mask_source"]
        with self.assertRaises(ValueError):
            refine(inputs, provenance=provenance, robot_self_mask=np.zeros((12, 12), bool))
        with self.assertRaises(TypeError):
            refine_target_surfaces(*inputs)

    def test_nonfinite_projection_is_excluded_before_connectivity(self):
        inputs = list(scene())
        inputs[3][2:5, 2:5] = 0.9
        original = inputs[2]
        class PartialCamera:
            W, H = original.W, original.H
            def backproject(self, u, v, z):
                cloud = original.backproject(u, v, z)
                cloud[u == 3] = np.nan
                return cloud
        inputs[2] = PartialCamera()
        result = refine(inputs)
        self.assertEqual(result["diagnostics"]["nonfinite_backprojection_pixels"], 3)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertTrue(np.all(result["labels"][:, 3] == 0))

    def test_component_budget_preserves_ambiguity_by_refusing_subset(self):
        inputs = scene()
        inputs[3][1:3, 1:3] = inputs[3][7:9, 7:9] = 0.9
        result = refine(inputs, max_candidates=1)
        self.assertEqual(result["status"], "candidate_budget_exceeded")
        self.assertEqual(result["diagnostics"]["eligible_components"], 2)
        self.assertEqual(result["candidates"], [])
        self.assertFalse(result["labels"].any())

    def test_small_components_discarded_explicitly(self):
        inputs = scene()
        inputs[3][1, 1] = 0.9
        result = refine(inputs)
        self.assertEqual(result["status"], "no_components_above_min_pixels")
        self.assertEqual(result["diagnostics"]["small_components"], 1)
        self.assertEqual(result["candidates"], [])

    def test_determinism_spatial_ids_and_input_immutability(self):
        inputs = scene()
        inputs[3][2:5, 2:5] = 0.8
        inputs[3][7:10, 7:10] = 0.95  # stronger component must not reorder spatial IDs
        snapshot = [x.copy() if isinstance(x, np.ndarray) else x for x in inputs]
        expected = refine(inputs)
        for _ in range(3):
            actual = refine(inputs)
            np.testing.assert_array_equal(actual["labels"], expected["labels"])
            self.assertEqual(actual["diagnostics"], expected["diagnostics"])
            for a, b in zip(actual["candidates"], expected["candidates"]):
                for key in a:
                    if isinstance(a[key], np.ndarray):
                        np.testing.assert_array_equal(a[key], b[key])
                    else:
                        self.assertEqual(a[key], b[key])
        self.assertEqual([c["bbox_xyxy"] for c in expected["candidates"]],
                         [[2, 2, 5, 5], [7, 7, 10, 10]])
        for a, b in zip(inputs, snapshot):
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)

    def test_graph_matches_independent_seeded_four_neighbor_oracle(self):
        # This is an implementation oracle on tiny generated arrays, not an
        # object/presence label or an empirical threshold selection procedure.
        from collections import deque
        rng = np.random.default_rng(74291)
        for case in range(24):
            inputs = scene(7, 9)
            rgb, depth, camera, pos, neg = inputs
            rgb[:] = rng.integers(0, 4, size=rgb.shape, dtype=np.uint8) * 40
            depth[:] = rng.choice([1.0, 1.2, 2.0, np.nan], size=depth.shape)
            pos[:] = rng.choice([0.2, 0.8, np.nan], size=pos.shape)
            neg[:] = rng.choice([0.0, 0.7], size=neg.shape)
            robot = rng.random(pos.shape) < 0.1
            thresholds = replace(THRESHOLDS, neighbor_distance_m=0.7,
                                 neighbor_color_linf=80, min_component_pixels=1)
            result = refine(inputs, thresholds=thresholds, robot_self_mask=robot)
            support = (np.isfinite(pos) & (pos >= thresholds.positive_min)
                       & (pos-neg >= thresholds.margin_min) & np.isfinite(depth)
                       & (depth > thresholds.depth_min_m) & (depth < thresholds.depth_max_m)
                       & ~robot)
            expected = np.zeros(pos.shape, np.int32)
            component_id = 0
            for v in range(7):
                for u in range(9):
                    if not support[v, u] or expected[v, u]:
                        continue
                    component_id += 1
                    expected[v, u] = component_id
                    queue = deque([(v, u)])
                    while queue:
                        y, x = queue.popleft()
                        point = camera.backproject(x, y, depth[y, x])
                        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                            if not (0 <= yy < 7 and 0 <= xx < 9):
                                continue
                            if not support[yy, xx] or expected[yy, xx]:
                                continue
                            other = camera.backproject(xx, yy, depth[yy, xx])
                            if np.linalg.norm(point-other) > thresholds.neighbor_distance_m:
                                continue
                            if max(abs(rgb[y, x].astype(int)-rgb[yy, xx].astype(int))) > thresholds.neighbor_color_linf:
                                continue
                            expected[yy, xx] = component_id
                            queue.append((yy, xx))
            with self.subTest(case=case):
                np.testing.assert_array_equal(result["labels"], expected)
                self.assertEqual(len(result["candidates"]), component_id)
                self.assert_observed_samples(result, inputs)

    def test_720_resolution_cpu_smoke(self):
        rgb, depth, camera, _, _ = scene(720, 720)
        inputs = rgb, depth, camera, np.full((14, 14), 0.9), np.zeros((14, 14))
        result = refine(inputs)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["pixels"], 720*720)
        self.assertEqual(result["candidates"][0]["bbox_xyxy"], [0, 0, 720, 720])
        self.assertEqual(result["diagnostics"]["surface_support_pixels"], 720*720)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "outcome": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"test": test.id(), "outcome": "failed"})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"test": test.id(), "outcome": "error"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="optional machine-readable test result")
    args = parser.parse_args()
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InstanceSurfaceTests)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    record = {
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "duration_seconds": time.perf_counter() - start,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "python": sys.executable, "numpy": np.__version__, "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "data_source": "synthetic RGB-D only; no task recordings or labels",
        "thresholds": "analytical fixtures only; not calibrated/deployable thresholds",
        "gpu_used": False, "object_presence_verified": False, "physical_success_verified": False,
        "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (ROOT / "harness/instance_surface.py", Path(__file__).resolve())},
        "tests": result.records,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
