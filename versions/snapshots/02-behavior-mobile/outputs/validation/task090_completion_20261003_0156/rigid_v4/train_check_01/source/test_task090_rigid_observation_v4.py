#!/usr/bin/env python3
"""CPU synthetic counterexamples only; none supplies physical identity/grasp labels."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from harness.task090_rigid_observation_v4 import Task090RigidObservationV4, seeded_flow_matches
from harness.task090_rigid_observation_v3 import sample_points
from harness.task090_rigid_observation_v2 import height_change_bound
from scripts.test_task090_rigid_observation_v3 import (
    AnchoredTests, CALIBRATION, Initializer, ConsistentGeometryTests,
)
from scripts.test_task090_rigid_observation_v2 import RigidGeometryTests


class OriginalBankTests(unittest.TestCase):
    obs = AnchoredTests.obs
    update = AnchoredTests.update
    initialize = AnchoredTests.initialize

    def setUp(self):
        AnchoredTests.setUp(self)
        self.tracker = Task090RigidObservationV4(CALIBRATION, initializer=self.initializer)

    def raw_update(self, observation, step, episode=0):
        return self.tracker.update(observation, self.state, episode=episode, step=step,
                                   head_features=None, action=np.ones(23))

    def assert_unknown(self, report):
        self.assertEqual(report['rigid_tracking']['status'], 'unknown', report)
        self.assertFalse(report['verified'])
        self.assertFalse(report['admitted'])
        self.assertFalse(report['rigid_tracking']['candidate_grasp'])

    def test_occluded_originals_reappear_via_real_images(self):
        self.initialize()
        bank = self.tracker.original_bank
        partial = self.obs(1)
        partial['robot::zed_link:Camera:0::rgb'][35:125, 36:80] = 180
        partial['robot::zed_link:Camera:0::depth_linear'][35:125, 36:80] = .5
        report = self.raw_update(partial, 1)
        self.assertEqual(report['rigid_tracking']['status'], 'rigid_motion_measured', report)
        mask = self.tracker.last_accepted_mask.copy()
        self.assertGreater(int((~mask).sum()), 0)
        self.assertGreaterEqual(int(mask.sum()), 6)
        report = self.update(2, 2)
        self.assertEqual(report['rigid_tracking']['status'], 'rigid_motion_measured', report)
        self.assertGreater(report['rigid_tracking']['reobserved_original_anchors'], 0)
        self.assertTrue(np.any(self.tracker.last_accepted_mask & ~mask))
        self.assertIs(bank, self.tracker.original_bank)
        self.assertEqual(self.initializer.calls, 1)

    def test_full_occlusion_cannot_be_replaced_by_projected_pose(self):
        self.initialize()
        bank, mask = self.tracker.original_bank, self.tracker.last_accepted_mask.copy()
        pose = tuple(x.copy() for x in self.tracker.previous_pose)
        obs = self.obs(1)
        obs['robot::zed_link:Camera:0::rgb'][:] = 180
        self.state[42:45] += [0., 0., .2]
        self.assert_unknown(self.raw_update(obs, 1))
        self.assertIs(bank, self.tracker.original_bank)
        np.testing.assert_array_equal(mask, self.tracker.last_accepted_mask)
        for before, after in zip(pose, self.tracker.previous_pose):
            np.testing.assert_array_equal(before, after)

    def test_bank_is_immutable_and_not_the_current_visibility(self):
        self.initialize()
        bank = self.tracker.original_bank
        before = [x.tobytes() for x in (bank.gray, bank.uv, bank.reference, bank.errors)]
        for array in (bank.gray, bank.uv, bank.reference, bank.errors):
            with self.assertRaises(ValueError):
                array.flat[0] = 0
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        self.update(1, 1)
        self.assertEqual(before, [x.tobytes() for x in (bank.gray, bank.uv, bank.reference, bank.errors)])
        self.assertFalse(np.shares_memory(bank.uv, self.tracker.anchor_uv))
        self.assertFalse(np.shares_memory(bank.reference, self.tracker.anchor_reference))
        self.tracker.reset_contract('next_skill')
        self.assertIs(bank, self.tracker.original_bank)

    def test_less_than_six_measurements_fail_without_deleting_bank(self):
        self.initialize()
        bank = self.tracker.original_bank
        def only_five(old, new, uv, seed):
            valid = np.arange(len(uv)) < 5
            return uv + [1., 0.], valid
        with patch('harness.task090_rigid_observation_v4.seeded_flow_matches', side_effect=only_five):
            report = self.update(1, 1)
        self.assert_unknown(report)
        self.assertEqual(report['rigid_tracking']['original_anchors_after_hand_separation'], 5)
        self.assertIs(bank, self.tracker.original_bank)
        self.assertEqual(len(bank.uv), int(self.tracker.last_accepted_mask.sum()))

    def test_degenerate_depth_geometry_fails(self):
        self.initialize()
        def collapsed(camera, depth, uv):
            return np.c_[np.linspace(-.1, .1, len(uv)), np.zeros(len(uv)), np.ones(len(uv))]
        with patch('harness.task090_rigid_observation_v4.sample_points', side_effect=collapsed):
            self.assert_unknown(self.update(1, 1))

    def test_wrong_and_invalid_depth_fail(self):
        for value in (2., np.nan, 0.):
            with self.subTest(depth=value):
                self.tracker.reset()
                self.initialize()
                obs = self.obs(1)
                obs['robot::zed_link:Camera:0::depth_linear'][35:125, 36:126] = value
                self.assert_unknown(self.raw_update(obs, 1))

    def test_final_original_residual_is_recomputed_from_measured_depth(self):
        self.initialize()
        report = self.update(2, 1)
        self.assertEqual(report['rigid_tracking']['status'], 'rigid_motion_measured', report)
        tracker = self.tracker
        measured = sample_points(tracker.previous_camera,
                                 self.obs(2)['robot::zed_link:Camera:0::depth_linear'], tracker.anchor_uv)
        r, t = tracker.previous_pose
        error = np.linalg.norm(tracker.anchor_reference @ r.T + t - measured, axis=1)
        self.assertGreaterEqual(len(error), 6)
        self.assertLessEqual(float(error.max()), .01)
        np.testing.assert_array_equal(tracker.anchor_reference,
                                      tracker.original_bank.reference[tracker.last_accepted_mask])

    def test_current_camera_and_accepted_pose_seed_not_eef(self):
        self.initialize()
        tracker = self.tracker
        bank = tracker.original_bank
        camera = tracker.previous_camera
        camera.t = np.array([.05, .02, 0.])
        tracker.previous_pose = (np.eye(3), np.array([.1, -.01, 0.]))
        tracker._current_hands = np.array([[-3., 0., 0.], [3., 0., 0.]])
        points = bank.reference + [.1, -.01, 0.]
        projected = (points-camera.t) @ camera.R
        projected = projected[:, :2] / projected[:, 2, None] * camera.f + [camera.cx, camera.cy]
        seen = []
        def reject(old, new, uv, seed):
            seen.append((old.copy(), uv.copy(), seed.copy()))
            return seed.copy(), np.zeros(len(seed), bool)
        with patch('harness.task090_rigid_observation_v4.seeded_flow_matches', side_effect=reject):
            with self.assertRaises(ValueError):
                tracker._tracked(bank.gray, tracker.previous_depth, camera,
                                 np.full_like(bank.gray, 255), {})
        np.testing.assert_array_equal(seen[0][0], bank.gray)
        np.testing.assert_array_equal(seen[0][1], bank.uv)
        np.testing.assert_allclose(seen[0][2], projected, atol=1e-5)

    def test_both_current_hand_neighborhoods_are_excluded(self):
        for hand_slice in (slice(17, 20), slice(42, 45)):
            with self.subTest(hand=hand_slice):
                self.tracker.reset()
                self.state[17:20], self.state[42:45] = [-3., 0., 0.], [3., 0., 0.]
                self.initialize()
                self.state[hand_slice] = [0., 0., 1.]
                point = np.array([[0., 0., 1.]])
                with patch('harness.task090_rigid_observation_v4.sample_points',
                           side_effect=lambda camera, depth, uv: np.repeat(point, len(uv), axis=0)):
                    report = self.update(1, 1)
                self.assert_unknown(report)
                self.assertEqual(report['rigid_tracking']['original_anchors_after_hand_separation'], 0)

    def test_no_replenishment_or_reset_of_existing_used_margin(self):
        self.initialize()
        tracker = self.tracker
        tracker.error_high_water = .003
        report = self.update(1, 1)
        diag = report['rigid_tracking']
        self.assertEqual(diag['status'], 'rigid_motion_measured', report)
        self.assertEqual(diag['replenished_points'], 0)
        self.assertEqual(diag['total_replenished_points'], 0)
        self.assertEqual(tracker.error_high_water, .003)
        self.assertGreaterEqual(diag['assumed_measurement_allowance_m'], .010)
        tracker._unknown({}, 'occlusion')
        self.assertEqual(tracker.error_high_water, .003)

    def test_episode_reset_discards_bank_and_closed_hand_cannot_bind(self):
        self.initialize()
        bank = self.tracker.original_bank
        self.state[49:51] = .02
        report = self.raw_update(self.obs(1), 0, episode=1)
        self.assert_unknown(report)
        self.assertIn('cannot_rebind_target_after_gripper_closes', report['reason'])
        self.assertIsNone(self.tracker.original_bank)
        self.assertIsNone(self.tracker.last_accepted_mask)
        self.assertEqual(self.initializer.calls, 1)
        self.state[49:51] = .05
        report = self.raw_update(self.obs(2), 1, episode=1)
        self.assertEqual(report['rigid_tracking']['status'], 'tracking_initialized', report)
        self.assertIsNot(self.tracker.original_bank, bank)
        self.assertTrue(self.tracker.track_id.startswith('anchored:1:'))

    def test_target_mismatch_does_not_rebind_or_replace_bank(self):
        self.initialize()
        bank, mask = self.tracker.original_bank, self.tracker.last_accepted_mask.copy()
        self.assert_unknown(self.update(1, 1, object_id='half'))
        self.assertIs(bank, self.tracker.original_bank)
        np.testing.assert_array_equal(mask, self.tracker.last_accepted_mask)
        self.assertEqual(self.initializer.calls, 1)

    def test_static_background_and_hand_motion_not_a_grasp(self):
        self.initialize()
        obs = self.obs(0)
        obs['robot::zed_link:Camera:0::rgb'][0, 0] = 181  # new sensor hash, same object pixels
        self.state[42:45] += [0., 0., .2]
        report = self.raw_update(obs, 1)
        diag = report['rigid_tracking']
        self.assertEqual(diag['status'], 'rigid_motion_measured', report)
        self.assertLess(diag['conditional_min_height_gain_m'], 0.)
        self.assertGreater(diag['hand_motion_residual_m'], .02)
        self.assertFalse(diag['candidate_grasp'])
        self.assertFalse(report['admitted'])
        self.assertFalse(report['verified'])

    def test_repeated_texture_is_not_semantic_or_grasp_admission(self):
        # A repeated material pattern can be geometrically ambiguous. The tracker
        # must not claim to solve identity simply because LK finds a local match.
        tile = np.random.default_rng(118).integers(0, 61, (15, 15, 3), dtype=np.uint8)
        self.texture = np.tile(tile, (6, 6, 1))
        self.initialize()
        report = self.update(1, 1)
        self.assertFalse(report['admitted'])
        self.assertFalse(report['verified'])
        self.assertFalse(report['rigid_tracking']['candidate_grasp'])
        self.assertEqual(self.initializer.calls, 1)

    def test_pure_rotation_cannot_prove_a_height_gain(self):
        points = np.random.default_rng(31).uniform(-.1, .1, (30, 3)) + [0., 0., 1.]
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        bound = height_change_bound(points, points @ rotation.T, rotation,
                                    np.zeros(3), np.zeros(3), .5, 0., .002)
        self.assertLessEqual(bound['conditional_min_height_gain_m'], 0.)

    def test_v4_cannot_relax_gap(self):
        with self.assertRaises(ValueError):
            Task090RigidObservationV4(CALIBRATION, initializer=Initializer(), max_frame_gap=7)


# Reuse unchanged v3 safety behaviors, not its obsolete shrink/refill implementation tests.
for _name in (
    'test_track_does_not_rerun_initializer',
    'test_contract_reset_preserves_reference_and_floor',
    'test_expired_anchor_never_reinitializes_open_hand',
    'test_recovery_needs_three_original_overlap_observations',
    'test_missing_floor_repeated_frame_and_forbidden_truth_fail_closed',
    'test_sparse_observations_never_become_dense_evidence',
    'test_eef_motion_cannot_replace_missing_object_measurements',
):
    setattr(OriginalBankTests, _name, getattr(AnchoredTests, _name))


class SeededFlowTests(unittest.TestCase):
    def test_both_lk_directions_use_initial_flow_and_fb_gate(self):
        image = np.zeros((80, 80), np.uint8)
        uv = np.array([[20., 20.], [25., 25.]], np.float32)
        seed = uv + [2., 0.]
        forward = (uv + [1., 0.]).astype(np.float32)[:, None]
        back = uv.copy()
        back[0, 0] += 1.5
        back[1, 0] += 1.5001
        status = np.ones((2, 1), np.uint8)
        with patch('harness.task090_rigid_observation_v4.cv2.calcOpticalFlowPyrLK',
                   side_effect=[(forward, status, None), (back[:, None], status, None)]) as lk:
            measured, valid = seeded_flow_matches(image, image, uv, seed)
        self.assertEqual(lk.call_count, 2)
        for call in lk.call_args_list:
            self.assertEqual(call.kwargs['flags'], cv2.OPTFLOW_USE_INITIAL_FLOW)
            self.assertEqual(call.kwargs['winSize'], (31, 31))
            self.assertEqual(call.kwargs['maxLevel'], 3)
        np.testing.assert_allclose(lk.call_args_list[0].args[3][:, 0], seed)
        np.testing.assert_allclose(lk.call_args_list[1].args[3][:, 0], uv)
        np.testing.assert_allclose(measured, forward[:, 0])
        np.testing.assert_array_equal(valid, [True, False])

    def test_missing_optical_flow_never_returns_prediction(self):
        image = np.zeros((80, 80), np.uint8)
        uv = np.array([[20., 20.]], np.float32)
        with patch('harness.task090_rigid_observation_v4.cv2.calcOpticalFlowPyrLK',
                   return_value=(None, None, None)):
            with self.assertRaisesRegex(ValueError, 'optical_flow_missing'):
                seeded_flow_matches(image, image, uv, uv)

    def test_dimension_change_fails_closed(self):
        uv = np.array([[20., 20.]], np.float32)
        with self.assertRaisesRegex(ValueError, 'invalid_original_flow_inputs'):
            seeded_flow_matches(np.zeros((80, 80), np.uint8), np.zeros((90, 90), np.uint8), uv, uv)


def load_tests(loader, tests, pattern):
    # Do not accidentally rerun the imported v3 tracker class as v4 evidence.
    return unittest.TestSuite(loader.loadTestsFromTestCase(cls) for cls in (
        OriginalBankTests, SeededFlowTests, ConsistentGeometryTests, RigidGeometryTests))


if __name__ == '__main__':
    cv2.setNumThreads(1)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    unittest.main(verbosity=2)
