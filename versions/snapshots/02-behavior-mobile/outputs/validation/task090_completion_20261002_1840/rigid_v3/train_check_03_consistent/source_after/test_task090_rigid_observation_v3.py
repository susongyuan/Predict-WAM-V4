"""CPU counterexamples for v3; synthetic images are not physical task evidence."""
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from harness.object_perception import Camera
from harness.task090_predicates import GRIP_ACTION
from harness.task090_rigid_observation_v3 import (
    Task090RigidObservationV3, original_anchor_check, nearest_surface_distance,
    hand_separated, consistent_rigid, RIGID_TOLERANCE_M, FB_TOLERANCE_PX,
)
from harness.task090_rigid_observation_v2 import fit_rigid, robust_rigid
from scripts.test_task090_rigid_observation_v2 import RigidGeometryTests

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'configs/task090_perception_geometry_v2.json'


class Initializer:
    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        self.calls += 1
        return dict(status='measurements_available', detections=[
            dict(object_id=kwargs['object_id'], bbox_xyxy=[35, 35, 125, 125])])


class ConsistentGeometryTests(unittest.TestCase):
    def test_real_v2_refit_counterexample_is_finally_consistent(self):
        # Fixed synthetic fixture, unrelated to TRAIN or physical success labels.
        rng = np.random.default_rng(3)
        source = rng.uniform(-.1, .1, (24, 3))
        target = source + rng.normal(0., .005, (24, 3))
        before_source, before_target = source.copy(), target.copy()
        old_r, old_t, _, old_keep = robust_rigid(source, target)
        old_error = np.linalg.norm(source @ old_r.T + old_t - target, axis=1)
        self.assertEqual(int(old_keep.sum()), 16)
        self.assertGreater(float(old_error[old_keep].max()), .01)
        with patch('harness.task090_rigid_observation_v3.fit_rigid', wraps=fit_rigid) as fits:
            r, t, rms, keep = consistent_rigid(source, target)
        self.assertGreaterEqual(fits.call_count, 2)
        self.assertEqual(int(keep.sum()), 15)
        self.assertTrue(np.all(~keep | old_keep))  # Never re-admit a discarded point.
        residual = np.linalg.norm(source @ r.T + t - target, axis=1)
        self.assertLessEqual(float(residual[keep].max()), .01)
        self.assertAlmostEqual(rms, float(np.sqrt(np.mean(residual[keep]**2))), places=14)
        final_r, final_t, _ = fit_rigid(source[keep], target[keep])
        np.testing.assert_allclose(r, final_r, atol=1e-12)
        np.testing.assert_allclose(t, final_t, atol=1e-12)
        np.testing.assert_array_equal(source, before_source)
        np.testing.assert_array_equal(target, before_target)

    def test_refit_cannot_return_fewer_than_six(self):
        source = np.random.default_rng(7).uniform(-.1, .1, (6, 3))
        target = source.copy()
        target[-1] += [.1, 0., 0.]
        # Isolate the finalizer from RANSAC: even an optimistic upstream mask
        # cannot make a sub-six final set valid.
        initial = (np.eye(3), np.zeros(3), 0., np.ones(6, bool))
        with patch('harness.task090_rigid_observation_v3.robust_rigid', return_value=initial):
            with self.assertRaisesRegex(ValueError, 'too_few_final_consistent'):
                consistent_rigid(source, target)

    def test_final_refit_preserves_collinearity_rejection(self):
        source = np.c_[np.arange(6), np.zeros(6), np.ones(6)]
        initial = (np.eye(3), np.zeros(3), 0., np.ones(6, bool))
        with patch('harness.task090_rigid_observation_v3.robust_rigid', return_value=initial):
            with self.assertRaisesRegex(ValueError, 'collinear'):
                consistent_rigid(source, source)

    def test_exact_six_noncollinear_points_recover_pose(self):
        source = np.random.default_rng(12).uniform(-.1, .1, (6, 3))
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = np.array([.02, -.03, .04])
        target = source @ rotation.T + translation
        r, t, rms, keep = consistent_rigid(source, target)
        self.assertTrue(keep.all())
        np.testing.assert_allclose(r, rotation, atol=1e-12)
        np.testing.assert_allclose(t, translation, atol=1e-12)
        self.assertLess(rms, 1e-12)

    def test_invalid_geometry_cannot_enter_consensus(self):
        source = np.zeros((6, 3))
        for target in (np.full((6, 3), np.nan), np.zeros((5, 3))):
            with self.subTest(shape=target.shape):
                with self.assertRaisesRegex(ValueError, 'invalid_correspondences'):
                    consistent_rigid(source, target)


class AnchoredTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.initializer = Initializer()
        self.tracker = Task090RigidObservationV3(CALIBRATION, initializer=self.initializer)
        self.state = np.zeros(61)
        self.state[17:20] = [-3., 0., 0.]
        self.state[42:45] = [3., 0., 0.]
        self.state[24:26] = self.state[49:51] = .05
        rng = np.random.default_rng(118)
        self.texture = rng.integers(0, 61, size=(90, 90, 3), dtype=np.uint8)
        self.camera_patch = patch('harness.task090_rigid_observation_v3.camera_from_pose',
                                  return_value=(np.eye(3), np.zeros(3)))
        self.floor_patch = patch('harness.task090_rigid_observation_v3.observed_floor_plane',
                                 return_value=dict(coefficients=np.zeros(3), margin_m=.001, points=100))
        self.camera_patch.start()
        self.floor_patch.start()
        self.addCleanup(self.camera_patch.stop)
        self.addCleanup(self.floor_patch.stop)

    def obs(self, shift=0):
        rgb = np.full((160, 160, 3), 180, np.uint8)
        rgb[35:125, 35+shift:125+shift] = self.texture
        depth = np.full((160, 160), 2., np.float32)
        depth[35:125, 35+shift:125+shift] = 1.
        return {'robot::zed_link:Camera:0::rgb': rgb,
                'robot::zed_link:Camera:0::depth_linear': depth,
                'robot::cam_rel_poses': np.array([0., 0., 1., 0., 0., 0., 1.])}

    def update(self, shift, step, **kwargs):
        return self.tracker.update(self.obs(shift), self.state, episode=0, step=step,
                                   head_features=None, action=np.ones(23), **kwargs)

    def initialize(self):
        result = self.update(0, 0)
        self.assertEqual(result['rigid_tracking']['status'], 'tracking_initialized', result)

    def test_frozen_v2_thresholds_not_relaxed(self):
        self.assertEqual(RIGID_TOLERANCE_M, .01)
        self.assertEqual(FB_TOLERANCE_PX, 1.5)
        with self.assertRaises(ValueError):
            Task090RigidObservationV3(CALIBRATION, max_frame_gap=7)

    def test_track_does_not_rerun_initializer(self):
        self.initialize()
        result = self.update(1, 1)
        self.assertEqual(result['rigid_tracking']['status'], 'rigid_motion_measured', result)
        self.assertEqual(self.initializer.calls, 1)
        self.assertFalse(result['admitted'])
        self.assertFalse(result['verified'])

    def test_contract_reset_preserves_reference_and_floor(self):
        self.initialize()
        reference = self.tracker.reference.copy()
        floor = self.tracker.origin_floor.copy()
        track = self.tracker.track_id
        self.tracker.candidate_count = 16
        self.tracker.reset_contract('next_skill')
        np.testing.assert_array_equal(reference, self.tracker.reference)
        np.testing.assert_array_equal(floor, self.tracker.origin_floor)
        self.assertEqual(track, self.tracker.track_id)
        self.assertEqual(self.tracker.candidate_count, 0)

    def test_added_points_cannot_outvote_original_pose(self):
        ref = np.random.default_rng(8).normal(size=(12, 3))
        with self.assertRaisesRegex(ValueError, 'original_anchor_pose_disagreement'):
            original_anchor_check(ref, ref, np.eye(3), np.array([.011, 0., 0.]))

    def test_added_majority_is_only_validated_never_fits_original_pose(self):
        self.initialize()
        reference = np.random.default_rng(29).uniform(-.1, .1, (72, 3))
        original = reference[:12].copy()
        self.tracker.reference = reference.copy()
        self.tracker.anchor_reference = original.copy()
        self.tracker.uv = np.full((72, 2), 50., np.float32)
        self.tracker.anchor_uv = self.tracker.uv[:12].copy()
        self.tracker.errors = np.r_[np.zeros(12), np.full(60, .004)]
        gray = np.zeros((160, 160), np.uint8)
        depth = np.ones((160, 160), np.float32)
        for offset, expected in ((.009, 72), (.011, 12)):
            with self.subTest(added_offset=offset):
                points = reference.copy()
                points[12:, 0] += offset
                diagnostic = {}
                with patch('harness.task090_rigid_observation_v3.flow_matches',
                           side_effect=lambda old, new, uv: (uv.copy(), np.ones(len(uv), bool))), \
                     patch('harness.task090_rigid_observation_v3.sample_points',
                           side_effect=[original.copy(), points]), \
                     patch('harness.task090_rigid_observation_v3.robust_rigid', wraps=robust_rigid) as consensus:
                    result = self.tracker._tracked(gray, depth, self.tracker.previous_camera,
                                                   np.full_like(gray, 255), diagnostic)
                self.assertEqual(consensus.call_count, 1)
                np.testing.assert_array_equal(consensus.call_args[0][0], original)
                np.testing.assert_allclose(result[4], np.eye(3), atol=1e-12)
                np.testing.assert_allclose(result[5], np.zeros(3), atol=1e-12)
                self.assertEqual(diagnostic['pose_source'], 'original_anchors_only')
                self.assertEqual(diagnostic['overlap_inliers'], expected)
                self.assertLessEqual(diagnostic['overlap_pose_error_max_m'], .01)
                np.testing.assert_array_equal(result[1], reference[:expected])
                np.testing.assert_array_equal(result[3], self.tracker.errors[:expected])

    def test_fewer_than_six_original_anchors_rejected(self):
        ref = np.random.default_rng(8).normal(size=(5, 3))
        with self.assertRaisesRegex(ValueError, 'too_few'):
            original_anchor_check(ref, ref, np.eye(3), np.zeros(3))

    def test_replenishment_inherits_error_and_reference(self):
        self.initialize()
        origin = self.tracker.origin_surface.copy()
        self.tracker.uv = self.tracker.uv[:10]
        self.tracker.reference = self.tracker.reference[:10]
        self.tracker.errors = np.full(10, .003)
        self.tracker.error_high_water = .003
        result = self.update(1, 1)
        diag = result['rigid_tracking']
        self.assertEqual(diag['status'], 'rigid_motion_measured', result)
        self.assertGreater(diag['replenished_points'], 0, result)
        self.assertGreater(diag['cumulative_reference_error_m'], .003)
        self.assertTrue(np.all(self.tracker.errors[10:] >= .003))
        np.testing.assert_array_equal(origin, self.tracker.origin_surface)
        error = self.tracker.error_high_water
        self.tracker._unknown({}, 'synthetic_occlusion')
        self.assertEqual(error, self.tracker.error_high_water)

    def test_no_refill_or_rebind_without_original_overlap(self):
        self.initialize()
        self.tracker.anchor_uv = self.tracker.anchor_uv[:5]
        self.tracker.anchor_reference = self.tracker.anchor_reference[:5]
        self.tracker.candidate_count = 16
        result = self.update(1, 1)
        self.assertEqual(result['rigid_tracking']['status'], 'unknown', result)
        self.assertEqual(self.tracker.candidate_count, 0)
        self.assertEqual(result['rigid_tracking']['replenished_points'], 0)
        self.update(2, 2)
        self.assertEqual(self.initializer.calls, 1)

    def test_expired_anchor_never_reinitializes_open_hand(self):
        self.initialize()
        result = self.update(1, 10)
        self.assertEqual(result['rigid_tracking']['status'], 'unknown')
        result = self.update(2, 11)
        self.assertIn('original_anchor_keyframe_expired_no_rebind', result['reason'])
        self.assertEqual(self.initializer.calls, 1)

    def test_recovery_needs_three_original_overlap_observations(self):
        self.initialize()
        self.tracker._unknown({}, 'synthetic_occlusion')
        for step in (1, 2):
            result = self.update(step, step)
            self.assertEqual(result['rigid_tracking']['status'], 'anchor_recovery_pending', result)
            self.assertFalse(result['rigid_tracking']['candidate_grasp'])
            self.assertEqual(result['rigid_tracking']['replenished_points'], 0)
        result = self.update(3, 3)
        self.assertEqual(result['rigid_tracking']['status'], 'rigid_motion_measured', result)
        self.assertEqual(result['rigid_tracking']['recovery_observations'], 3)
        self.assertFalse(result['rigid_tracking']['candidate_grasp'])
        self.assertEqual(self.initializer.calls, 1)

    def test_missing_floor_repeated_frame_and_forbidden_truth_fail_closed(self):
        self.initialize()
        result = self.update(0, 1)
        self.assertIn('repeated_sensor_frame', result['reason'])
        observation = self.obs(2)
        observation['q_score'] = 1.
        result = self.tracker.update(observation, self.state, episode=0, step=2, head_features=None)
        self.assertIn('forbidden_non_sensor_input', result['reason'])
        with patch('harness.task090_rigid_observation_v3.observed_floor_plane', return_value=None):
            result = self.update(3, 3)
        self.assertIn('floor_not_observed', result['reason'])
        self.assertEqual(self.tracker.candidate_count, 0)

    def test_other_target_requires_explicit_initializer(self):
        with self.assertRaisesRegex(ValueError, 'own_initializer'):
            Task090RigidObservationV3(CALIBRATION, target='half')
        other = Task090RigidObservationV3(CALIBRATION, target='half', hand='left', initializer=Initializer())
        self.assertEqual(other.action_index, GRIP_ACTION['left'])
        self.assertNotEqual(other.diameter, self.tracker.diameter)

    def test_sparse_observations_never_become_dense_evidence(self):
        self.initialize()
        result = self.update(1, 2)
        diag = result['rigid_tracking']
        self.assertEqual(diag['sample_gap_steps'], 2)
        self.assertTrue(diag['sparse_sampling'])
        self.assertFalse(diag['candidate_grasp'])
        self.assertEqual(diag['candidate_observations'], 0)

    def test_membership_rejects_wrong_surface_and_both_hands(self):
        surface = np.array([[0., 0., 1.], [.1, 0., 1.]])
        distance = nearest_surface_distance(np.array([[0., 0., 1.], [0., 0., 1.02]]), surface)
        self.assertLessEqual(distance[0], .01)
        self.assertGreater(distance[1], .01)
        keep = hand_separated(surface, [[0., 0., 1.], [4., 4., 4.]])
        self.assertFalse(keep.any())

    def test_refill_rejects_occluder_depth(self):
        self.initialize()
        rgb = self.obs(1)['robot::zed_link:Camera:0::rgb']
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        depth = self.obs(1)['robot::zed_link:Camera:0::depth_linear'].copy()
        depth[35:125, 36:126] += .02
        camera = self.tracker.previous_camera
        r, t = np.eye(3), np.array([1/camera.f, 0., 0.])
        uv = self.tracker.uv[:10] + [1., 0.]
        reference = self.tracker.reference[:10]
        points = reference + t
        result = self.tracker._replenish(gray, depth, camera, self.tracker._appearance(rgb, depth),
                                         self.tracker.previous_hands, uv, reference, points,
                                         np.zeros(10), r, t, 0.)
        self.assertEqual(result[-1]['replenished_points'], 0)

    def test_eef_motion_cannot_replace_missing_object_measurements(self):
        self.initialize()
        obs = self.obs(1)
        obs['robot::zed_link:Camera:0::rgb'][:] = 180
        self.state[42:45] += [0., 0., .2]
        result = self.tracker.update(obs, self.state, episode=0, step=1, head_features=None)
        self.assertEqual(result['rigid_tracking']['status'], 'unknown')
        self.assertFalse(result['rigid_tracking']['candidate_grasp'])
        self.assertEqual(self.initializer.calls, 1)

    def test_original_error_available_on_failure(self):
        ref = np.random.default_rng(8).normal(size=(12, 3))
        diagnostic = {}
        with self.assertRaises(ValueError):
            original_anchor_check(ref, ref, np.eye(3), np.array([.011, 0., 0.]), diagnostic)
        self.assertEqual(diagnostic['original_anchor_inliers'], 12)
        self.assertGreater(diagnostic['original_anchor_pose_error_max_m'], .01)
        self.assertEqual(diagnostic['original_anchor_pose_inliers_at_1cm'], 0)

    def test_episode_and_target_changes_do_not_reuse_candidate(self):
        self.initialize()
        result = self.update(1, 1, object_id='half')
        self.assertIn('track_target_mismatch', result['reason'])
        self.assertFalse(result['rigid_tracking']['candidate_grasp'])


if __name__ == '__main__':
    unittest.main()
