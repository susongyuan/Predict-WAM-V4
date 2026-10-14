#!/usr/bin/env python3
"""CPU real-USD containment and synthetic integration counterexamples, no replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from harness import r1pro_kinematics as fk
from harness.robot_self_geometry import RobotSelfGeometry, EXPORT, SENSOR_MARGIN_M, sha256
from harness.task090_rigid_observation_v3 import hand_separated
from harness.task090_rigid_observation_v4 import Task090RigidObservationV4
from scripts.test_task090_rigid_observation_v3 import AnchoredTests, CALIBRATION

OUT = EXPORT/'implementation'


def synthetic_state(geometry, fraction=.5, finger=.02):
    """Synthetic SI state; EEF fields computed by unchanged FK, never TRAIN input."""
    state = np.zeros(61)
    for name, binding in geometry._bindings.items():
        lo, hi = geometry._limits[name]
        state[binding['state61_index']] = lo + fraction*(hi-lo)
    state[24:26] = state[49:51] = finger
    for arm, (position, quaternion) in fk.EEF.items():
        p, rotation = geometry.kinematics.fk(arm, fk.q11(state, arm))
        state[position], state[quaternion] = p, fk.mat_to_quat(rotation)
    return state


def usd_poses(document, state):
    """Independent exported joint0 @ motion @ inv(joint1), not URDF traversal."""
    poses = {'base_link': np.eye(4)}
    poses['base_link'][:3, 3] = document['existing_calibration']['root_xyz']
    pending = dict(document['joints'])
    while pending:
        ready = [name for name, j in pending.items() if j['parent'] in poses]
        if not ready:
            raise AssertionError('disconnected USD fixture')
        for name in ready:
            j = pending.pop(name)
            motion = np.eye(4)
            if j['qpos'] is not None:
                q = state[j['qpos']['state61_index']]
                axis = np.asarray(j['axis_joint'])
                if j['type'] == 'prismatic':
                    motion[:3, 3] = q*axis
                else:
                    # Explicit Rodrigues rather than runtime axis_angle.
                    x, y, z = axis
                    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                    motion[:3, :3] += np.sin(q)*cross + (1-np.cos(q))*(cross@cross)
            poses[j['child']] = (poses[j['parent']] @ np.asarray(j['T_parent_joint0']) @
                                  motion @ np.linalg.inv(j['T_child_joint1']))
    return poses


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = RobotSelfGeometry()
        cls.document = json.loads((EXPORT/'geometry.json').read_text())
        validation = json.loads((EXPORT/'validation.json').read_text())
        path = EXPORT/'collision_vertices.npz'
        if sha256(path) != validation['sources'][str(path)]['sha256']:
            raise ValueError('vertex_fixture_sha_mismatch')
        cls.vertices = np.load(path, allow_pickle=False)

    @classmethod
    def tearDownClass(cls):
        cls.vertices.close()

    def test_all_4852_real_vertices_at_three_articulated_poses(self):
        for fraction, finger in ((.25, 0.), (.5, .02), (.75, .05)):
            state = synthetic_state(self.geometry, fraction, finger)
            independent = usd_poses(self.document, state)
            runtime = self.geometry.link_poses(state)
            clouds = []
            for name, boxes in self.document['links'].items():
                np.testing.assert_allclose(runtime[name], independent[name], atol=1e-6, rtol=0)
                for box in boxes:
                    vertices = self.vertices[box['id']+'__points_link_m']
                    self.assertTrue(((vertices >= box['min_m']) & (vertices <= box['max_m'])).all())
                    transform = independent[name]
                    clouds.append(vertices@transform[:3, :3].T + transform[:3, 3])
            cloud = np.concatenate(clouds)
            self.assertEqual(len(cloud), 4852)
            self.assertFalse(self.geometry(cloud, state).any())

    def test_boundary_and_entire_one_cm_skin_are_rejected(self):
        state = synthetic_state(self.geometry)
        poses = self.geometry.link_poses(state)
        points = []
        for name, (lo, hi) in self.geometry.boxes.items():
            for a, b in zip(lo, hi):
                for axis in range(3):
                    for value in (a[axis]-.01, a[axis], b[axis], b[axis]+.01):
                        local = (a+b)/2
                        local[axis] = value
                        points.append(local@poses[name][:3, :3].T + poses[name][:3, 3])
        self.assertEqual(SENSOR_MARGIN_M, .01)
        self.assertFalse(self.geometry(np.asarray(points), state).any())

    def test_near_eef_external_points_not_automatically_excluded(self):
        state = synthetic_state(self.geometry)
        # Fixed 15cm directions; analytically outside every expanded box is the
        # witness condition, not a semantic label or a fitted exclusion threshold.
        directions = np.r_[np.eye(3), -np.eye(3)] * .15
        poses = usd_poses(self.document, state)
        for arm, (position, _) in fk.EEF.items():
            points = state[position] + directions
            independent_outside = np.ones(len(points), bool)
            for link, boxes in self.document['links'].items():
                local = (points-poses[link][:3, 3])@poses[link][:3, :3]
                for box in boxes:
                    independent_outside &= ~((local >= np.asarray(box['min_m'])-.01) &
                                              (local <= np.asarray(box['max_m'])+.01)).all(axis=1)
            self.assertTrue(independent_outside.any(), arm)
            np.testing.assert_array_equal(self.geometry(points, state), independent_outside)
            self.assertFalse(hand_separated(points, [state[17:20], state[42:45]]).any())

    def test_both_finger_joint2_axes_are_minus_y_without_q_sign_flip(self):
        closed = synthetic_state(self.geometry, finger=0.)
        opened = synthetic_state(self.geometry, finger=.05)
        before, after = self.geometry.link_poses(closed), self.geometry.link_poses(opened)
        for arm in ('left', 'right'):
            rotation = before[arm+'_gripper_link'][:3, :3]
            for finger, sign in ((1, 1), (2, -1)):
                name = f'{arm}_gripper_finger_link{finger}'
                delta = rotation.T@(after[name][:3, 3]-before[name][:3, 3])
                np.testing.assert_allclose(delta, [0., sign*.05, 0.], atol=1e-12)
                np.testing.assert_allclose(before[name][:3, :3], after[name][:3, :3], atol=1e-12)
        # Velocity slots must not be used as finger qpos.
        altered = opened.copy()
        altered[26:28] = altered[51:53] = 100.
        for name, pose in after.items():
            np.testing.assert_array_equal(self.geometry.link_poses(altered)[name], pose)

    def test_eef_offset_and_root_applied_once(self):
        state = synthetic_state(self.geometry)
        for arm, (position, quaternion) in fk.EEF.items():
            expected = fk.R1ProKinematics._T(fk.quat_to_mat(state[quaternion]), state[position])
            offset = fk.R1ProKinematics._T(self.geometry.kinematics.eef_rot[arm],
                                          self.geometry.kinematics.eef_offset[arm])
            np.testing.assert_allclose(self.geometry.link_poses(state)[arm+'_gripper_link'],
                                       expected@np.linalg.inv(offset), atol=1e-12)
            wrong = state.copy()
            wrong[position] += [.0, .0, .06]
            with self.assertRaisesRegex(ValueError, 'eef_fk_frame_mismatch'):
                self.geometry(np.zeros((1, 3)), wrong)

    def test_invalid_shape_nonfinite_units_and_pose_fail_closed(self):
        state = synthetic_state(self.geometry)
        for points in (np.zeros(3), np.zeros((2, 4)), np.array([[np.nan, 0., 0.]]), [[np.inf, 0., 0.]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.geometry(points, state)
        for wrong in (np.zeros(60), np.zeros((1, 61)), np.full(61, np.nan)):
            with self.assertRaises(ValueError):
                self.geometry(np.empty((0, 3)), wrong)
        for index, value in ((49, 50.), (24, -.01), (3, 180.), (45, np.nan)):
            wrong = state.copy()
            wrong[index] = value
            with self.assertRaises(ValueError):
                self.geometry(np.zeros((1, 3)), wrong)
        wrong = state.copy()
        wrong[45:49] = 0
        with self.assertRaisesRegex(ValueError, 'eef_xyzw'):
            self.geometry(np.zeros((1, 3)), wrong)
        wrong = state.copy()
        wrong[45:49] = np.roll(wrong[45:49], 1)
        with self.assertRaisesRegex(ValueError, 'eef_fk_frame_mismatch'):
            self.geometry(np.zeros((1, 3)), wrong)

    def test_empty_and_far_points_keep_shape_and_do_not_mutate(self):
        state = synthetic_state(self.geometry)
        points = np.array([[10., 10., 10.], [-10., -10., -10.]])
        before = state.copy(), points.copy()
        self.assertTrue(self.geometry(points, state).all())
        self.assertEqual(self.geometry(np.empty((0, 3)), state).shape, (0,))
        np.testing.assert_array_equal(state, before[0])
        np.testing.assert_array_equal(points, before[1])
        for lo, hi in self.geometry.boxes.values():
            with self.assertRaises(ValueError):
                lo.setflags(write=True)
            self.assertFalse(hi.flags.writeable)

    def test_tampered_geometry_units_mapping_or_source_never_load(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            path = Path(directory)/'invalid.json'
            for key, value in (('source_usd_sha256', '0'*64), ('units', {'meters_per_unit': .001}),
                               ('state61_mapping', {}), ('links', {})):
                document = dict(self.document)
                document[key] = value
                path.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, 'source_sha256_mismatch'):
                    RobotSelfGeometry(geometry_path=path)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'source_sha256_mismatch'):
                RobotSelfGeometry(validation_path=path)
        with patch('harness.robot_self_geometry.sha256', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'source_sha256_mismatch'):
                RobotSelfGeometry()


class InjectionTests(unittest.TestCase):
    obs = AnchoredTests.obs

    def setUp(self):
        AnchoredTests.setUp(self)

    def update(self, step=0):
        return self.tracker.update(self.obs(step), self.state, episode=0, step=step,
                                   head_features=None, action=np.ones(23))

    def test_default_path_keeps_old_radius(self):
        self.tracker = Task090RigidObservationV4(CALIBRATION, initializer=self.initializer)
        self.state[42:45] = [0., 0., 1.]
        with patch('harness.task090_rigid_observation_v3.sample_points',
                   side_effect=lambda c, d, uv: np.tile([0., 0., 1.], (len(uv), 1))):
            report = self.update()
        self.assertEqual(report['rigid_tracking']['self_filter_version'], 'legacy_eef_radius_0.2m')
        self.assertEqual(report['rigid_tracking']['status'], 'unknown')

    def test_explicit_filter_replaces_radius_in_init_surface_and_tracking(self):
        calls = []
        def keep(points, state):
            calls.append((points.copy(), state.copy()))
            return np.ones(len(points), bool)
        self.tracker = Task090RigidObservationV4(CALIBRATION, initializer=self.initializer, self_filter=keep)
        self.state[42:45] = [0., 0., 1.]
        report = self.update()
        self.assertEqual(report['rigid_tracking']['status'], 'tracking_initialized', report)
        self.assertGreater(report['rigid_tracking']['initial_surface_after_self_filter'], 0)
        report = self.update(1)
        self.assertEqual(report['rigid_tracking']['status'], 'rigid_motion_measured', report)
        self.assertGreater(report['rigid_tracking']['original_anchors_after_self_filter'], 5)
        self.assertEqual(self.initializer.calls, 1)
        self.assertEqual(len(calls), 4)
        for _, state in calls:
            np.testing.assert_array_equal(state, self.state)
        self.assertFalse(report['admitted'])
        self.assertFalse(report['verified'])

    def test_rejecting_or_invalid_callback_is_unknown_not_default_fallback(self):
        callbacks = [lambda p, s: np.zeros(len(p), bool), lambda p, s: np.ones((len(p), 1), bool),
                     lambda p, s: np.ones(len(p)), lambda p, s: np.full(len(p), np.nan)]
        def broken(p, s):
            raise RuntimeError('unavailable_geometry')
        for callback in callbacks + [broken]:
            self.tracker = Task090RigidObservationV4(CALIBRATION, initializer=self.initializer,
                                                     self_filter=callback)
            report = self.update()
            self.assertEqual(report['rigid_tracking']['status'], 'unknown', report)
            self.assertFalse(report['verified'])
            self.assertFalse(report['admitted'])
            self.assertIsNone(self.tracker.original_bank)

    def test_tracking_filter_failure_preserves_original_bank_and_pose(self):
        self.tracker = Task090RigidObservationV4(CALIBRATION, initializer=self.initializer,
                                                self_filter=lambda p, s: np.ones(len(p), bool))
        self.update()
        bank, pose = self.tracker.original_bank, self.tracker.previous_pose
        self.tracker.self_filter = lambda p, s: np.zeros(len(p), bool)
        report = self.update(1)
        self.assertEqual(report['rigid_tracking']['status'], 'unknown')
        self.assertEqual(report['rigid_tracking']['original_anchors_after_self_filter'], 0)
        self.assertIs(self.tracker.original_bank, bank)
        self.assertIs(self.tracker.previous_pose, pose)


class CheckerContractTests(unittest.TestCase):
    def test_exact_initial_cache_no_neighbor_or_reextraction_fallback(self):
        import hashlib
        from scripts.check_task090_continuous_tracking import initial_features
        features = np.ones((3, 201, 1024), np.float32)
        identity = dict(episode=18000, frame_index=480,
                        feature_float32_sha256=hashlib.sha256(features.tobytes()).hexdigest())
        with patch('scripts.check_task090_continuous_tracking.cached_frame', return_value=features) as cache:
            actual, audit = initial_features(18000, 480, identity)
            self.assertIs(actual, features)
            self.assertEqual(audit['status'], 'exact_cached_frame')
            cache.assert_called_once_with(18000, 480)
        for failure in (ValueError('requested frame is not in the frozen cache'), FileNotFoundError('missing')):
            with patch('scripts.check_task090_continuous_tracking.cached_frame', side_effect=failure) as cache:
                actual, audit = initial_features(18000, 480, identity)
                self.assertIsNone(actual)
                self.assertEqual(audit['status'], 'unknown_no_matching_initial_cache')
                cache.assert_called_once_with(18000, 480)
        with patch('scripts.check_task090_continuous_tracking.cached_frame', return_value=features):
            actual, _ = initial_features(18000, 480, dict(identity, feature_float32_sha256='0'*64))
            self.assertIsNone(actual)
        with patch('scripts.check_task090_continuous_tracking.cached_frame') as cache:
            actual, _ = initial_features(18000, 481, identity)
            self.assertIsNone(actual)
            cache.assert_not_called()

    def test_gap1_previous_action_si_and_unlabeled_contract(self):
        from scripts.check_task090_continuous_tracking import checked_row
        row = dict(episode=18000, frame_index=481, split='train', dataset_revision='fixture',
                   sample_fps=30, frame_gap=1, depth_unit='m', applied_action_frame=480,
                   applied_action_verified=True, action_source_row_index=100, dataset_index=101,
                   physical_success_label=None, grasp_relation_label=None, floor_contact_label=None,
                   state61=np.zeros(61), camera_pose7=[0., 0., 0., 0., 0., 0., 1.], applied_action=np.zeros(23))
        checked_row(row, 18000, 481, False, 'fixture')
        for key, value in (('frame_gap', 2), ('applied_action_frame', 481), ('depth_unit', 'mm'),
                           ('split', 'validation'), ('physical_success_label', True), ('state61', np.zeros(60)),
                           ('applied_action', np.full(23, np.nan)), ('camera_pose7', [0.]*7)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                checked_row(dict(row, **{key: value}), 18000, 481, False, 'fixture')


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(loader.loadTestsFromTestCase(cls) for cls in
                              (GeometryTests, InjectionTests, CheckerContractTests))


if __name__ == '__main__':
    import resource
    import sys
    import time
    import psutil
    from scripts import test_task090_rigid_observation_v4 as legacy
    cv2.setNumThreads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = time.monotonic()
    if psutil.virtual_memory().percent >= 85 or psutil.cpu_percent(interval=.1) >= 85:
        raise RuntimeError('CPU/RAM85 guard')
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [fk.ROOT/p for p in (
        'harness/robot_self_geometry.py', 'harness/task090_rigid_observation_v4.py',
        'scripts/test_robot_self_geometry.py', 'scripts/check_task090_continuous_tracking.py',
        'scripts/test_task090_rigid_observation_v4.py', 'scripts/test_task090_rigid_observation_v3.py',
        'scripts/test_task090_rigid_observation_v2.py')]
    sources = {str(path): sha256(path) for path in paths}
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([loader.loadTestsFromModule(sys.modules[__name__]),
                               loader.loadTestsFromModule(legacy)])
    with (OUT/'cpu_tests.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), legacy_v4_tests=38,
                  real_collision_vertices=4852, synthetic_articulated_poses=3,
                  continuous_train_frames_read=0, gpu_used=False, source_sha256=sources,
                  wall_seconds=time.monotonic()-started,
                  max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  admitted=False, verified=False)
    with (OUT/'cpu_tests.json').open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps(report))
    sys.exit(0 if result.wasSuccessful() else 1)
