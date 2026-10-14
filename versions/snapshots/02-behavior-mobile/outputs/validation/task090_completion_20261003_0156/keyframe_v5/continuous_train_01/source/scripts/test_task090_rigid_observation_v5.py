#!/usr/bin/env python3
"""Explicit v5 CPU suite; imported old tracker classes are never discovered."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
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
from harness import task090_rigid_observation_v3 as gates
from harness.task090_rigid_observation_v5 import Task090RigidObservationV5, VERSION, AppearanceKeyframe
from harness.task090_rigid_observation_v4 import immutable_array
from scripts import test_task090_rigid_observation_v4 as old
from scripts import test_robot_self_geometry as self_tests

CURRENT = ROOT/'outputs/validation/task090_completion_20261003_0156/keyframe_v5'


class KeyframeTests(unittest.TestCase):
    obs = old.OriginalBankTests.obs
    update = old.OriginalBankTests.update
    initialize = old.OriginalBankTests.initialize
    raw_update = old.OriginalBankTests.raw_update
    assert_unknown = old.OriginalBankTests.assert_unknown

    def setUp(self):
        old.OriginalBankTests.setUp(self)
        self.tracker = Task090RigidObservationV5(old.CALIBRATION, initializer=self.initializer)

    def snapshot(self):
        return {key: getattr(self.tracker, key, None) for key in self.tracker._FRAME_FIELDS}

    def unchanged(self, before):
        for key, value in before.items():
            self.assertIs(getattr(self.tracker, key, None), value, key)

    def test_failure_is_atomic_including_late_height(self):
        self.initialize()
        before = self.snapshot()
        with patch.object(gates, 'height_change_bound', side_effect=ValueError('invalid_geometry_margin')):
            self.assert_unknown(self.update(1, 1))
        self.unchanged(before)
        self.assertTrue(self.tracker.lost)
        self.assertEqual(self.tracker.candidate_count, 0)
        self.assertEqual(self.tracker.last_step, 1)

    def test_nonfinite_height_rolls_back_after_parent_commit(self):
        self.initialize()
        before = self.snapshot()
        real = gates.height_change_bound
        def invalid(*args):
            return dict(real(*args), conditional_min_height_gain_m=float('nan'))
        with patch.object(gates, 'height_change_bound', side_effect=invalid):
            report = self.update(1, 1)
        self.assert_unknown(report)
        self.assertIn('invalid_height', report['reason'])
        self.assertIsNone(report['rigid_tracking']['conditional_min_height_gain_m'])
        self.unchanged(before)

    def test_floor_invalid_and_expired_frames_do_not_commit(self):
        self.initialize()
        before = self.snapshot()
        with patch.object(gates, 'observed_floor_plane', return_value=dict(coefficients=np.zeros(3), margin_m=-.0001)):
            self.assert_unknown(self.update(1, 1))
        self.unchanged(before)
        self.assert_unknown(self.update(2, 10))
        self.unchanged(before)
        self.assertEqual(self.initializer.calls, 1)

    def test_invalid_initial_floor_leaves_no_bank(self):
        with patch.object(gates, 'observed_floor_plane', return_value=dict(coefficients=np.full(3, np.nan), margin_m=.001)):
            self.assert_unknown(self.update(0, 0))
        self.assertIsNone(self.tracker.original_bank)
        self.assertIsNone(self.tracker.appearance_keyframe)
        self.assertIsNone(self.tracker.previous_pose)
        self.assertEqual(self.initializer.calls, 0)

    def test_closed_hand_tracks_but_does_not_initialize(self):
        self.initialize()
        track = self.tracker.track_id
        self.state[49:51] = .02
        report = self.update(1, 1)
        self.assertEqual(report['rigid_tracking']['status'], 'rigid_motion_measured', report)
        self.assertEqual(self.tracker.track_id, track)
        self.assertEqual(self.initializer.calls, 1)
        self.tracker.reset()
        self.assert_unknown(self.update(2, 2))
        self.assertEqual(self.initializer.calls, 1)

    def test_keyframe_is_measured_immutable_and_bounded(self):
        self.initialize()
        original = self.tracker.original_bank
        for step in range(1, 5):
            previous = self.tracker.appearance_keyframe
            report = self.update(step, step)
            self.assertEqual(report['rigid_tracking']['status'], 'rigid_motion_measured', report)
            kf = self.tracker.appearance_keyframe
            self.assertEqual(kf.step, step)
            self.assertIsNot(kf, previous)
            np.testing.assert_array_equal(kf.gray, cv2.cvtColor(self.obs(step)['robot::zed_link:Camera:0::rgb'], cv2.COLOR_RGB2GRAY))
            np.testing.assert_array_equal(kf.uv, self.tracker.anchor_uv)
            np.testing.assert_array_equal(self.tracker.anchor_reference, original.reference[kf.ids])
            for array in (kf.gray, kf.uv, kf.ids):
                with self.assertRaises(ValueError):
                    array.setflags(write=True)
            self.assertEqual(report['rigid_tracking']['appearance_keyframe_count'], 1)
            self.assertIs(self.tracker.original_bank, original)

    def test_prediction_only_cannot_measure_or_rebind_after_total_loss(self):
        self.initialize()
        before = self.snapshot()
        self.state[42:45] += [0., 0., .2]
        def no_matches(old_gray, gray, uv, seeds):
            return seeds.copy(), np.zeros(len(uv), bool)
        with patch('harness.task090_rigid_observation_v5.seeded_flow_matches', side_effect=no_matches):
            for step in range(1, 8):
                self.assert_unknown(self.update(step, step))
                self.unchanged(before)
        self.assert_unknown(self.update(8, 8))  # no fresh semantic binding after expiry
        self.assertEqual(self.initializer.calls, 1)
        self.assertEqual(self.tracker.total_added, 0)

    def test_short_source_is_real_image_not_pose_seed(self):
        self.initialize()
        self.update(1, 1)
        previous = self.tracker.appearance_keyframe
        from harness.task090_rigid_observation_v4 import seeded_flow_matches
        seen = []
        def traced(old_gray, gray, uv, seeds):
            seen.append((old_gray, uv.copy(), seeds.copy()))
            return seeded_flow_matches(old_gray, gray, uv, seeds)
        with patch('harness.task090_rigid_observation_v5.seeded_flow_matches', side_effect=traced):
            report = self.update(3, 2)
        self.assertEqual(report['rigid_tracking']['status'], 'rigid_motion_measured', report)
        self.assertIs(seen[0][0], previous.gray)
        np.testing.assert_array_equal(seen[0][1], previous.uv)
        self.assertGreater(float(np.linalg.norm(self.tracker.anchor_uv-previous.uv, axis=1).mean()), 1.)
        self.assertEqual(self.tracker.appearance_keyframe.step, 2)

    def test_original_fallback_reobserves_ids_without_creating_landmarks(self):
        self.initialize()
        kf = self.tracker.appearance_keyframe
        self.tracker.appearance_keyframe = AppearanceKeyframe(kf.gray, immutable_array(kf.uv[:6]), immutable_array(kf.ids[:6]), kf.step)
        report = self.update(1, 1)
        self.assertEqual(report['rigid_tracking']['status'], 'rigid_motion_measured', report)
        self.assertGreater(report['rigid_tracking']['correspondence_sources']['original_fallback']['after_self'], 0)
        self.assertGreater(len(self.tracker.anchor_uv), 6)
        self.assertEqual(self.tracker.total_added, 0)
        self.assertEqual(len(set(self.tracker.appearance_keyframe.ids.tolist())), len(self.tracker.anchor_uv))

    def test_depth_jump_is_not_averaged_and_outlier_is_excluded(self):
        self.initialize()
        bank = self.tracker.original_bank
        obs = self.obs(1)
        uv = bank.uv + [1., 0.]
        jump_id = int(np.flatnonzero((uv[:, 0] > 40) & (uv[:, 0] < 120) & (uv[:, 1] > 40) & (uv[:, 1] < 120))[0])
        u, v = np.rint(uv[jump_id]).astype(int)
        obs['robot::zed_link:Camera:0::depth_linear'][v, u] = 1.15
        def exact(old_gray, gray, source_uv, seeds):
            return source_uv + [1., 0.], np.ones(len(source_uv), bool)
        with patch('harness.task090_rigid_observation_v5.seeded_flow_matches', side_effect=exact):
            report = self.raw_update(obs, 1)
        diag = report['rigid_tracking']
        if diag['status'] == 'unknown':
            self.assert_unknown(report)
            self.assertIn('rejected_frame_reason', diag['depth_samples'])
        else:
            self.assertFalse(self.tracker.last_accepted_mask[jump_id])
            self.assertIn(jump_id, diag['depth_samples']['rejected_original_ids'])
            self.assertEqual(diag['depth_samples']['rejection_gate'], 'original_rigid_consensus_1cm')
        self.assertAlmostEqual(diag['depth_samples']['local_span_5x5_m'][jump_id], .15, places=6)

    def test_fewer_than_six_current_originals_fail(self):
        self.initialize()
        before = self.snapshot()
        def five(old_gray, gray, uv, seeds):
            # Only five globally distinct IDs, even when the fallback is queried.
            valid = np.array([any(np.array_equal(q, p) for p in self.tracker.original_bank.uv[:5]) for q in uv])
            return uv + [1., 0.], valid
        with patch('harness.task090_rigid_observation_v5.seeded_flow_matches', side_effect=five):
            self.assert_unknown(self.update(1, 1))
        self.unchanged(before)

    def test_explicit_self_geometry_rejects_tracking_atomically(self):
        self.tracker.self_filter = lambda p, s: np.ones(len(p), bool)
        self.initialize()
        before = self.snapshot()
        self.tracker.self_filter = lambda p, s: np.zeros(len(p), bool)
        self.assert_unknown(self.update(1, 1))
        self.unchanged(before)

    def test_v5_cannot_relax_gap(self):
        with self.assertRaises(ValueError):
            Task090RigidObservationV5(old.CALIBRATION, initializer=old.Initializer(), max_frame_gap=7)


# Each borrowed tracker test is bound to KeyframeTests.setUp, which constructs v5.
# Exclude old tests whose patch targets or v4 direct-only mechanism do not apply.
BORROWED = (
    'test_occluded_originals_reappear_via_real_images',
    'test_full_occlusion_cannot_be_replaced_by_projected_pose',
    'test_bank_is_immutable_and_not_the_current_visibility',
    'test_wrong_and_invalid_depth_fail',
    'test_final_original_residual_is_recomputed_from_measured_depth',
    'test_no_replenishment_or_reset_of_existing_used_margin',
    'test_episode_reset_discards_bank_and_closed_hand_cannot_bind',
    'test_target_mismatch_does_not_rebind_or_replace_bank',
    'test_static_background_and_hand_motion_not_a_grasp',
    'test_repeated_texture_is_not_semantic_or_grasp_admission',
    'test_pure_rotation_cannot_prove_a_height_gain',
    'test_track_does_not_rerun_initializer',
    'test_contract_reset_preserves_reference_and_floor',
    'test_expired_anchor_never_reinitializes_open_hand',
    'test_recovery_needs_three_original_overlap_observations',
    'test_missing_floor_repeated_frame_and_forbidden_truth_fail_closed',
    'test_sparse_observations_never_become_dense_evidence',
    'test_eef_motion_cannot_replace_missing_object_measurements',
)
for name in BORROWED:
    setattr(KeyframeTests, name, getattr(old.OriginalBankTests, name))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(loader.loadTestsFromTestCase(cls) for cls in (
        KeyframeTests, old.SeededFlowTests, old.ConsistentGeometryTests,
        old.RigidGeometryTests, self_tests.GeometryTests, self_tests.CheckerContractTests))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=CURRENT)
    args = parser.parse_args()
    if args.output.resolve() != CURRENT.resolve():
        raise ValueError('only the isolated keyframe_v5 output is allowed')
    CURRENT.mkdir(parents=True, exist_ok=False)
    os.environ['TMPDIR'] = str(CURRENT)
    cv2.setNumThreads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import psutil
    assert psutil.virtual_memory().percent < 85
    start = time.monotonic()
    paths = [Path(__file__), ROOT/'scripts/check_task090_keyframe_tracking.py',
             ROOT/'harness/task090_rigid_observation_v5.py', ROOT/'scripts/check_task090_continuous_tracking.py',
             ROOT/'harness/task090_rigid_observation_v4.py', ROOT/'harness/task090_rigid_observation_v3.py',
             ROOT/'harness/task090_rigid_observation_v2.py', ROOT/'harness/robot_self_geometry.py',
             Path(old.__file__), ROOT/'scripts/test_task090_rigid_observation_v3.py',
             ROOT/'scripts/test_task090_rigid_observation_v2.py', Path(self_tests.__file__), old.CALIBRATION]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    suite = load_tests(unittest.defaultTestLoader, None, None)
    with (CURRENT/'cpu_tests.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    value = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                 failures=len(result.failures), errors=len(result.errors), tracker_version=VERSION,
                 v5_tracker_tests=unittest.defaultTestLoader.loadTestsFromTestCase(KeyframeTests).countTestCases(),
                 reused_tracker_test_names=list(BORROWED), shared_primitive_and_geometry_tests_separate=True,
                 source_sha256=hashes, seconds=time.monotonic()-start, gpu_used=False,
                 physical_labels_available=0, admitted=False, verified=False)
    with (CURRENT/'cpu_tests.json').open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
    print(json.dumps(value), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
