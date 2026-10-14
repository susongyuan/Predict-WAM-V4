"""CPU contract tests for onboard measurements; no physical-admission claim."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from harness.task090_observation import Task090ObservationProducer


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.producer = Task090ObservationProducer('unused-test-calibration')
        self.point = np.array([.4, -.2, .6])
        self.count = 1
        self.state = np.zeros(61)
        self.state[42:45] = [.3, -.2, .6]
        self.state[49:51] = [.02, .02]
        self.camera = SimpleNamespace(backproject=lambda u, v, z: np.c_[u*.01, v*.01, z*.6])
        self.patches = [
            patch('harness.task090_observation.load_calibration', return_value={'depth_max_m':6.}),
            patch('harness.task090_observation.camera_from_pose', return_value=(None, None)),
            patch('harness.task090_observation.head_pose', return_value=None),
            patch('harness.task090_observation.Camera', return_value=self.camera),
            patch('harness.task090_observation.observed_floor_plane', return_value={
                'coefficients': np.zeros(3), 'margin_m': .006, 'points': 100}),
            patch('harness.task090_observation.detect_from_observation', side_effect=lambda *a, **k: (
                [{'label':'trash_can','bbox_xyxy':[0,0,8,8], 'xyz_base':self.point.tolist(),
                  'source':'rgbd', 'calibration_id':'test', 'point_semantics':'observed_surface_sample'}]*self.count, None)),
            patch('harness.task090_observation.appearance_candidates', return_value=[{
                'label':'trash_can','bbox_xyxy':[0,0,8,8], 'pixel_x':np.arange(8), 'pixel_y':np.arange(8)}])]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def obs(self, frame):
        return {'robot_r1::robot_r1:zed_link:Camera:0::rgb': np.full((8,8,3),frame,np.uint8),
                'robot_r1::robot_r1:zed_link:Camera:0::depth_linear': np.ones((8,8)),
                'robot_r1::cam_rel_poses': np.array([0,0,1,0,0,0,1])}

    def update(self, frame, step=None, episode=1):
        return self.producer.update(self.obs(frame), self.state, episode=episode, step=frame if step is None else step)

    def test_real_fields_but_not_success(self):
        row=self.update(0)
        self.assertEqual(row['status'],'measurements_available')
        d=row['detections'][0]
        self.assertEqual(d['hand_observation']['finger_sum_m'],.04)
        self.assertNotIn('support_clearance_m',d['relation'])
        self.assertFalse(d['whole_object_geometry_observed'])
        self.assertFalse(row['verified'])

    def test_co_motion_measured_without_self_certification(self):
        a=self.update(0)
        self.point[2]+=.03; self.state[44]+=.03
        b=self.update(1)
        self.assertEqual(a['detections'][0]['track_id'],b['detections'][0]['track_id'])
        self.assertAlmostEqual(b['detections'][0]['object_hand_offset_change_m'],0.)
        self.assertFalse(b['admitted'])

    def test_sparse_frames_reset_temporal_track(self):
        a=self.update(0); b=self.update(1,step=15)
        self.assertNotEqual(a['detections'][0]['track_id'],b['detections'][0]['track_id'])
        self.assertFalse(b['detections'][0]['temporal_continuity_supported'])
        self.assertTrue(b['detail']['sparse_gap'])

    def test_repeated_pixels_are_not_new_evidence(self):
        self.update(0)
        self.assertEqual(self.update(0,step=1)['reason'],'repeated_sensor_frame')

    def test_ambiguity_returns_unknown(self):
        self.count=2
        self.assertEqual(self.update(0)['reason'],'target_missing_or_ambiguous')

    def test_missing_target_returns_unknown(self):
        self.count=0
        self.assertEqual(self.update(0)['detections'],[])

    def test_gt_poison_rejected_without_read(self):
        class Poison:
            def __array__(self,*a,**k): raise AssertionError('GT read')
        obs=self.obs(0); obs['privbase_nav::ashcan.n.01_1::ifzxzj']=Poison()
        self.assertEqual(self.producer.update(obs,self.state,episode=1,step=0)['reason'],'forbidden_non_sensor_input')

    def test_episode_regression_not_a_fresh_reset(self):
        self.update(0,episode=2)
        self.assertEqual(self.update(1,episode=1)['reason'],'episode_regression')

    def test_new_episode_drops_history(self):
        a=self.update(0); b=self.update(0,episode=2)
        self.assertNotEqual(a['detections'][0]['track_id'],b['detections'][0]['track_id'])
        self.assertFalse(b['detections'][0]['temporal_continuity_supported'])


if __name__=='__main__':
    unittest.main()
