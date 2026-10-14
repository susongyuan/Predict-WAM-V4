#!/usr/bin/env python3
"""CPU temporal / search-origin counterexamples; no sensor file or GPU required."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from harness.object_perception import Camera
from harness.dense_rgbd_tracking import DenseSurfaceTracker
from harness.dense_rgbd_registration import register, surface, transform, CAPTURE, TOL
from harness.robot_self_geometry import sha256
from scripts.test_dense_rgbd_registration import analytic_depth
from scripts import check_task090_rigid_train_v3 as audit

BASE = ROOT/'outputs/validation/task090_completion_20261003_0156/dense_tracking'
EVIDENCE = {}


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = Camera(np.eye(3), np.zeros(3), 480, 480)
        cls.depth = analytic_depth(cls.camera, np.eye(4), 'balanced')
        cls.rgb = np.full((480, 480, 3), 128, np.uint8)
        cls.state = np.zeros(7)
        cls.cloud = surface(cls.depth, cls.camera, cls.depth > 0, np.zeros_like(cls.depth, bool), 6.)

    def setUp(self):
        audit.guard()

    def tracker(self, **kwargs):
        obj = DenseSurfaceTracker(depth_max=6., self_filter=lambda p, s: np.ones(len(p), bool), max_age_s=.2, **kwargs)
        obj.reset('episode-A', 'surface-A')
        report, _ = obj.initialize(self.rgb, self.depth, self.camera, self.state, self.depth > 0,
            binding_id='external-candidate-A', **self.meta(0))
        self.assertEqual(report['status'], 'reference_initialized', report)
        self.assertIsNone(report['cumulative_error_bound_m'])
        return obj

    def meta(self, step):
        return dict(episode='episode-A', target='surface-A', step=step, timestamp_s=step/30.,
                    now_s=step/30., observation_id=f'frame-{step}')

    def moved(self, x=.025):
        pose = np.eye(4); pose[0, 3] = x
        return pose, analytic_depth(self.camera, pose, 'balanced')

    def update(self, obj, step, depth=None, mask=None, **changes):
        depth = self.moved()[1] if depth is None else depth
        meta = self.meta(step); meta.update(changes)
        return obj.update(self.rgb, depth, self.camera, self.state, depth > 0 if mask is None else mask, **meta)

    def assertUnknownUnchanged(self, obj, call):
        reference, pose, step = obj.reference, obj.accepted_pose, obj.last_accepted_step
        report, trace = call()
        self.assertEqual(report['status'], 'unknown', report)
        self.assertFalse(report['pose_updated'])
        self.assertIs(obj.reference, reference)
        self.assertIs(obj.accepted_pose, pose)
        self.assertEqual(obj.last_accepted_step, step)
        return report, trace

    def test_continuous_known_motion_exceeds_absolute_capture(self):
        obj = self.tracker(); reference = obj.reference
        errors, translations = [], []
        for step in range(1, 7):
            truth, depth = self.moved(.025*step)
            previous = obj.accepted_pose.copy()
            report, trace = self.update(obj, step, depth)
            self.assertEqual(report['status'], 'surface_motion_measured', report)
            self.assertIs(obj.reference, reference)
            np.testing.assert_array_equal(report['search_initial_pose'], previous)
            np.testing.assert_array_equal(trace['fit_source_points'], reference.cloud['points'][trace['source_fit_indices']])
            error = np.linalg.norm(transform(reference.cloud['points'], obj.accepted_pose)-transform(reference.cloud['points'], truth), axis=1).max()
            self.assertLessEqual(error, TOL)
            self.assertLessEqual(report['measurement']['local_surface_displacement_m'], CAPTURE)
            self.assertIsNone(report['cumulative_error_bound_m'])
            self.assertIsNone(report['measurement']['cumulative_error_bound_m'])
            self.assertFalse(any(report[k] for k in ('verified', 'admitted', 'grasp')))
            errors.append(float(error)); translations.append(float(obj.accepted_pose[0, 3]))
        self.assertGreater(translations[-1], CAPTURE)
        EVIDENCE['known_motion'] = dict(truth_total_translation_m=.15, max_surface_errors_m=errors, estimated_x_m=translations)

    def test_explicit_origin_does_not_replace_fit(self):
        truth, depth = self.moved(.15)
        current = surface(depth, self.camera, depth > 0, np.zeros_like(depth, bool), 6.)
        origin, _ = self.moved(.125)
        result, trace = register(self.cloud, current, initial_pose=origin)
        self.assertEqual(result['status'], 'feasible_measurement_candidate', result)
        self.assertGreater(np.linalg.norm(np.asarray(result['transform_source_to_current_base'])-origin), .01)
        error = np.linalg.norm(transform(self.cloud['points'], np.asarray(result['transform_source_to_current_base']))-transform(self.cloud['points'], truth), axis=1).max()
        self.assertLess(error, TOL)
        for seed in trace['search_seeds']:
            displacement = np.sqrt(np.mean(np.sum((transform(trace['fit_source_points'], seed)-transform(trace['fit_source_points'], origin))**2, axis=1)))
            self.assertLessEqual(displacement, CAPTURE+1e-12)
        legacy, _ = register(self.cloud, current)
        self.assertEqual(legacy['status'], 'unobservable')
        self.assertIn('outside_declared_local_capture_basin', legacy['reasons'])

    def test_default_preserves_before_hypotheses_and_result(self):
        before = BASE/'source_before/dense_rgbd_registration.py'
        spec = importlib.util.spec_from_file_location('dense_before', before)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        _, depth = self.moved(.025)
        current = surface(depth, self.camera, depth > 0, np.zeros_like(depth, bool), 6.)
        old, oldtrace = module.register(self.cloud, current)
        new, newtrace = register(self.cloud, current, initial_pose=None)
        for key, value in oldtrace.items():
            np.testing.assert_array_equal(newtrace[key], value)
        for key, value in old.items():
            self.assertEqual(new[key], value, key)

    def test_invalid_initial_poses_rejected(self):
        wrong = [np.zeros((3, 3)), np.full((4, 4), np.nan)]
        for index, value in ((0, -1.), (0, 2.), (3, 0.)):
            p = np.eye(4); p[index, index] = value; wrong.append(p)
        for pose in wrong:
            with self.assertRaisesRegex(ValueError, 'invalid_initial_pose_se3'):
                register(self.cloud, self.cloud, initial_pose=pose)

    def test_wrong_origin_cannot_invent_surface(self):
        bad = np.eye(4); bad[0, 3] = 2.
        result, _ = register(self.cloud, self.cloud, initial_pose=bad)
        self.assertEqual(result['status'], 'unobservable')
        empty = surface(np.zeros_like(self.depth), self.camera, self.depth > 0, np.zeros_like(self.depth, bool), 6.)
        result, trace = register(self.cloud, empty, initial_pose=bad)
        self.assertEqual(result['status'], 'unobservable')
        self.assertIsNone(result['transform_source_to_current_base'])
        self.assertNotIn('hypotheses', trace)

    def test_current_hold_only_does_not_change_fits(self):
        first, second = self.tracker(), self.tracker()
        _, depth = self.moved(); changed = depth.copy(); changed[1::2, 1::2] += .03
        baseline, a = self.update(first, 1, depth)
        altered, b = self.update(second, 1, changed, mask=depth > 0)
        self.assertEqual(baseline['status'], 'surface_motion_measured')
        self.assertEqual(altered['status'], 'unknown')
        for key in ('hypotheses', 'search_seeds', 'fit_scores', 'fit_centre', 'fit_source_points', 'fit_target_points', 'fit_distance_m', 'fit_inliers'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(baseline['measurement']['selected_hypothesis'], altered['measurement']['selected_hypothesis'])
        np.testing.assert_array_equal(baseline['measurement']['transform_source_to_current_base'], altered['measurement']['transform_source_to_current_base'])
        EVIDENCE['current_hold_only_all_fits_unchanged'] = True

    def test_missing_hold_never_updates(self):
        obj = self.tracker(); _, depth = self.moved(); depth[1::2, 1::2] = 0
        report, _ = self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1, depth))
        self.assertIn('insufficient_held_surface_support', report['reasons'])

    def test_empty_surface_failure_and_recovery_keep_reference(self):
        obj = self.tracker()
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1, np.zeros_like(self.depth)))
        result, _ = self.update(obj, 2)
        self.assertEqual(result['status'], 'surface_motion_measured', result)
        self.assertEqual(result['search_origin_step'], 0)

    def test_no_overlap_keeps_accepted_state(self):
        obj = self.tracker(); depth = np.where(self.depth > 0, self.depth+1., 0.)
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1, depth))

    def test_rank_degenerate_current_does_not_update(self):
        obj = self.tracker()
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1, np.full_like(self.depth, .95)))

    def test_failed_frame_cannot_become_search_origin(self):
        obj = self.tracker(); self.update(obj, 1)
        self.assertEqual(obj.last_accepted_step, 1)
        pose = obj.accepted_pose.copy()
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 2, np.zeros_like(self.depth)))
        report, _ = self.update(obj, 3, self.moved(.05)[1])
        self.assertEqual(report['status'], 'surface_motion_measured', report)
        self.assertEqual(report['search_origin_step'], 1)
        np.testing.assert_array_equal(report['search_initial_pose'], pose)

    def test_immutable_initial_arrays_and_ids(self):
        obj = self.tracker(); reference = obj.reference
        arrays = [reference.rgb, reference.depth, reference.state, *reference.camera.values(), *reference.cloud.values(), obj.accepted_pose]
        for array in arrays:
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(TypeError):
            reference.cloud['points'] = np.zeros((1, 3))
        self.update(obj, 1)
        self.assertIs(obj.reference, reference)
        u, v = reference.cloud['uv'].T
        np.testing.assert_array_equal(reference.cloud['raw_stencil_indices'][:, 0], v*self.camera.W+u)
        np.testing.assert_allclose(reference.cloud['points'], self.camera.backproject(u, v, reference.depth[v, u]))
        self.assertIsNone(obj._report()['cumulative_error_bound_m'])

    def test_caller_mutation_does_not_change_reference(self):
        obj = DenseSurfaceTracker(depth_max=6., self_filter=lambda p,s: np.ones(len(p), bool), max_age_s=.2)
        obj.reset('episode-A', 'surface-A')
        depth, rgb, state = self.depth.copy(), self.rgb.copy(), self.state.copy()
        camera = Camera(np.eye(3), np.zeros(3), 480, 480)
        obj.initialize(rgb, depth, camera, state, depth > 0, binding_id='A', **self.meta(0))
        depth[:] = 0; rgb[:] = 0; state[:] = 1; camera.t[:] = 1
        np.testing.assert_array_equal(obj.reference.depth, self.depth)
        np.testing.assert_array_equal(obj.reference.rgb, self.rgb)
        np.testing.assert_array_equal(obj.reference.camera['t'], np.zeros(3))

    def test_duplicate_and_nonmonotonic_inputs(self):
        obj = self.tracker(); self.update(obj, 1)
        for step in (1, 0):
            self.assertUnknownUnchanged(obj, lambda step=step: self.update(obj, step))
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 2, observation_id='frame-1'))
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 3, timestamp_s=1/30.))

    def test_stale_sensor_does_not_update(self):
        obj = self.tracker()
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1, now_s=2.))
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1, timestamp_s=.1, now_s=.05))

    def test_scope_mismatch_and_explicit_reset(self):
        obj = self.tracker()
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1, episode='episode-B'))
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1, target='surface-B'))
        old = obj.reference
        obj.reset('episode-B', 'surface-B')
        self.assertIsNone(obj.reference)
        self.assertIsNone(obj.accepted_pose)
        report, _ = obj.update(self.rgb, self.depth, self.camera, self.state, self.depth > 0,
                              **{**self.meta(0), 'episode':'episode-B', 'target':'surface-B'})
        self.assertEqual(report['status'], 'unknown')
        report, _ = obj.initialize(self.rgb, self.depth, self.camera, self.state, self.depth > 0, binding_id='B',
                                  **{**self.meta(1), 'episode':'episode-B', 'target':'surface-B'})
        self.assertEqual(report['status'], 'reference_initialized')
        self.assertNotEqual(obj.reference.reference_id, old.reference_id)

    def test_gap_six_can_measure_seven_expires(self):
        obj = self.tracker()
        result, _ = self.update(obj, 6)
        self.assertEqual(result['status'], 'surface_motion_measured', result)
        obj = self.tracker()
        result, _ = self.assertUnknownUnchanged(obj, lambda: self.update(obj, 7))
        self.assertTrue(obj.expired)
        self.assertIn('accepted_reference_expired_no_automatic_rebind', result['reasons'])
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 8))

    def test_many_failures_expire_from_last_accepted_not_last_seen(self):
        obj = self.tracker()
        for step in range(1, 7):
            self.assertUnknownUnchanged(obj, lambda step=step: self.update(obj, step, np.zeros_like(self.depth)))
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 7))
        self.assertTrue(obj.expired)

    def test_cannot_rebind_after_loss_or_failed_initialization(self):
        obj = self.tracker(); self.update(obj, 7)
        self.assertUnknownUnchanged(obj, lambda: obj.initialize(self.rgb, self.depth, self.camera,
            self.state, self.depth > 0, binding_id='different-target', **self.meta(8)))
        obj.reset('episode-A', 'surface-A')
        first, _ = obj.initialize(self.rgb, np.zeros_like(self.depth), self.camera, self.state, self.depth > 0,
                                 binding_id='A', **self.meta(0))
        self.assertEqual(first['status'], 'unknown')
        self.assertUnknownUnchanged(obj, lambda: obj.initialize(self.rgb, self.depth, self.camera,
            self.state, self.depth > 0, binding_id='B', **self.meta(1)))

    def test_self_filter_rejection_and_bad_contract(self):
        obj = self.tracker(); obj.self_filter = lambda p, s: np.zeros(len(p), bool)
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1))
        obj = self.tracker(); obj.self_filter = lambda p, s: np.ones(len(p), int)
        self.assertUnknownUnchanged(obj, lambda: self.update(obj, 1))
        for kwargs in ({'max_frame_gap':7}, {'max_frame_gap':0}):
            with self.assertRaises(ValueError):
                self.tracker(**kwargs)

    def test_camera_motion_and_base_coordinate_convention(self):
        obj = self.tracker()
        camera = Camera(Rotation.from_rotvec([.025, -.01, .015]).as_matrix(), np.array([.008, -.009, .004]), 480, 480)
        depth = analytic_depth(camera, np.eye(4), 'balanced')
        report, _ = obj.update(self.rgb, depth, camera, self.state, depth > 0, **self.meta(1))
        self.assertEqual(report['status'], 'surface_motion_measured', report)
        error = np.linalg.norm(transform(obj.reference.cloud['points'], obj.accepted_pose)-obj.reference.cloud['points'], axis=1).max()
        self.assertLess(error, TOL)
        self.assertIsNone(report['world_pose'])

    def test_wrong_units_and_invalid_camera_safely_unknown(self):
        obj = self.tracker()
        self.assertUnknownUnchanged(obj, lambda: obj.update(self.rgb, self.depth, self.camera, self.state,
            self.depth > 0, unit='mm', **self.meta(1)))
        camera = Camera(np.eye(3), np.zeros(3), 480, 480); camera.f = np.nan
        self.assertUnknownUnchanged(obj, lambda: obj.update(self.rgb, self.depth, camera, self.state,
            self.depth > 0, **self.meta(2)))

    def test_failed_cpu_report_blocks_checker(self):
        from scripts.check_task090_dense_tracking import require_cpu_reports
        with self.assertRaises(RuntimeError):
            require_cpu_reports({'failed':1}, {'failed':0})
        with self.assertRaises(RuntimeError):
            require_cpu_reports({'failed':0}, {'failed':0, 'cases':[]})


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.rows = []
    def addSuccess(self, test):
        super().addSuccess(test); self.rows.append(dict(case=test._testMethodName, passed=True))
    def addFailure(self, test, error):
        super().addFailure(test, error); self.rows.append(dict(case=test._testMethodName, passed=False, error=self._exc_info_to_string(error, test)))
    def addError(self, test, error):
        super().addError(test, error); self.rows.append(dict(case=test._testMethodName, passed=False, error=self._exc_info_to_string(error, test)))


def run(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    audit.START = time.monotonic(); audit.PROCESS.cpu_affinity([min(audit.PROCESS.cpu_affinity())]); cv2.setNumThreads(1)
    paths = [Path(__file__), ROOT/'harness/dense_rgbd_tracking.py', ROOT/'harness/dense_rgbd_registration.py',
             ROOT/'scripts/check_task090_dense_tracking.py', ROOT/'scripts/test_dense_rgbd_registration.py']
    hashes = {str(p):sha256(p) for p in paths}
    with (out/'tests.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrackingTests))
    report = dict(passed=sum(row['passed'] for row in result.rows), failed=len(result.failures)+len(result.errors), cases=result.rows,
        leakage_checks_passed=EVIDENCE.get('current_hold_only_all_fits_unchanged', False), evidence=EVIDENCE,
        source_sha256=hashes, resources=audit.resources(), physical_admission=False)
    assert all(sha256(path)==value for path,value in hashes.items())
    audit.write_json(out/'results.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    report = run(parser.parse_args().out)
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, indent=2))
    sys.exit(bool(report['failed']))
