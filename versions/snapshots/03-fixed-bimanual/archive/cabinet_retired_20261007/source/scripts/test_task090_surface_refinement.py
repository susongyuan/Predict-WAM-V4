"""CPU integration of separated RGB-D candidates; no physical-success labels."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from harness.instance_surface import SurfaceThresholds
from harness.task090_dino_localizer import Task090DinoLocalizer
from harness.task090_observation import Task090ObservationProducer
from scripts.build_task090_dino_localizer import cached_frame

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz'
CALIBRATION = ROOT / 'configs/task090_perception_geometry_v2.json'
SETTINGS = dict(thresholds=dict(positive_min=.64, margin_min=.1, depth_min_m=.05,
    depth_max_m=3., neighbor_distance_m=.02, neighbor_color_linf=24., min_component_pixels=64),
    provenance=dict(score_source='frozen TRAIN appearance prototypes',
        threshold_source='synthetic integration fixture, NOT physical admission',
        alignment_source='full-frame 14x14 DINO spatial cells'), use_robot_self_geometry=False)


class Camera:
    W = H = 14

    def backproject(self, u, v, z):
        return np.column_stack((u * .001, v * .001, z))


class RefinementIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / 'outputs/validation/task090_first_grasp_evidence_20261001/extracted/frames.jsonl'
        with source.open() as file:
            cls.sample = next(r for r in map(json.loads, file)
                if r['episode'] == 18001 and r['frame_index'] == 870)
        cls.features = cached_frame(18001, 870)

    def observation(self):
        r = self.sample
        return {'robot_r1::robot_r1:zed_link:Camera:0::rgb': np.asarray(Image.open(r['rgb'])),
                'robot_r1::robot_r1:zed_link:Camera:0::depth_linear': np.load(r['depth_m'], allow_pickle=False),
                'robot_r1::cam_rel_poses': np.asarray(r['camera_pose7'])}

    def producer(self):
        return Task090ObservationProducer(CALIBRATION, dino_artifact=ARTIFACT,
                                         surface_refinement=copy.deepcopy(SETTINGS))

    def update(self, producer):
        return producer.update(self.observation(), self.sample['state61'],
            episode=18001, step=870, head_features=self.features)

    def test_scores_use_negative_and_do_not_merge_disconnected_candidates(self):
        localizer = Task090DinoLocalizer(ARTIFACT)
        pos, neg = np.zeros(196), np.zeros(196)
        pos[[15, 20, 25]] = .9
        neg[20] = .9
        thresholds = SurfaceThresholds(**{**SETTINGS['thresholds'], 'min_component_pixels': 1})
        with patch.object(localizer, 'scores', return_value=(pos, neg)):
            result = localizer.surfaces(np.zeros((14, 14, 3), np.uint8), np.ones((14, 14)),
                Camera(), None, thresholds=thresholds, provenance=SETTINGS['provenance'])
        self.assertEqual(len(result['candidates']), 2)
        self.assertEqual(result['labels'][1, 6], 0)
        self.assertEqual([c['surface_pixel_uv'] for c in result['candidates']], [[1, 1], [11, 1]])
        self.assertTrue(all(c['label'] == 'trash_can' for c in result['candidates']))
        self.assertFalse(result['physical_success_verified'])

    def test_extreme_numeric_domains_are_rejected_before_rounding_or_overflow(self):
        from harness.instance_surface import refine_target_surfaces
        thresholds = SurfaceThresholds(**{**SETTINGS['thresholds'], 'min_component_pixels': 1})
        rgb, depth = np.zeros((14, 14, 3), np.uint8), np.ones((14, 14))
        for pos, neg in ((9007199254740995, 9007199254740994), (1e308, 0.)):
            with self.subTest(pos=pos), self.assertRaisesRegex(ValueError, 'numeric magnitude'):
                refine_target_surfaces(rgb, depth, Camera(), np.full((14, 14), pos),
                    np.full((14, 14), neg), thresholds=thresholds, provenance=SETTINGS['provenance'])
        class ExtremeCamera(Camera):
            def backproject(self, u, v, z):
                return np.column_stack((np.full(len(u), 1e308), v, z))
        with self.assertRaisesRegex(ValueError, 'numeric magnitude'):
            refine_target_surfaces(rgb, depth, ExtremeCamera(), np.ones((14, 14)),
                np.zeros((14, 14)), thresholds=thresholds, provenance=SETTINGS['provenance'])

    def test_no_refinement_without_dino(self):
        with self.assertRaises(ValueError):
            Task090ObservationProducer(CALIBRATION, surface_refinement=SETTINGS)

    def test_invalid_settings_rejected(self):
        for change in ({'use_robot_self_geometry': 'false'}, {'extra': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Task090ObservationProducer(CALIBRATION, dino_artifact=ARTIFACT,
                    surface_refinement={**SETTINGS, **change})

    def test_multiple_candidates_stay_unknown_without_legacy_fallback(self):
        p = self.producer()
        candidate = dict(component_id=1, bbox_xyxy=[0, 0, 1, 1], xyz_base=[.5, 0., .3],
                         pixels=64, score_statistics={})
        reply = dict(candidates=[candidate, {**candidate, 'component_id': 2}],
                     status='candidates_available', diagnostics={'returned_components': 2})
        with patch.object(p.dino, 'surfaces', return_value=reply), patch.object(p.dino, 'surface') as legacy:
            result = self.update(p)
        legacy.assert_not_called()
        self.assertEqual(result['reason'], 'target_surface_missing_or_ambiguous')
        self.assertEqual(result['detail']['proposals'], 2)
        self.assertFalse(result['verified'])
        self.assertIsNone(p.track_id)

    def test_candidate_overflow_is_not_absence(self):
        p = self.producer()
        reply = dict(candidates=[], status='candidate_budget_exceeded', diagnostics={'eligible_components': 300})
        with patch.object(p.dino, 'surfaces', return_value=reply):
            result = self.update(p)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['detail']['surface_status'], 'candidate_budget_exceeded')
        self.assertFalse(result['admitted'])

    def test_single_surface_is_not_object_presence_or_clearance(self):
        p = self.producer()
        candidate = dict(component_id=1, bbox_xyxy=[0, 0, 1, 1], xyz_base=[.5, 0., .3],
            pixels=64, score_statistics={}, cloud=np.array([[.5, 0., .3]]), pixel_uv=np.array([[0, 0]]),
            label='trash_can', source='rgbd', object_presence_verified=False)
        with patch.object(p.dino, 'surfaces', return_value=dict(candidates=[candidate])):
            result = self.update(p)
        self.assertEqual(result['status'], 'measurements_available')
        self.assertFalse(result['verified'])
        measurement = result['detections'][0]
        self.assertNotIn('cloud', measurement)
        self.assertNotIn('support_clearance_m', measurement.get('relation', {}))
        self.assertFalse(measurement['object_presence_verified'])
        json.dumps(result)

    def test_self_geometry_error_never_falls_back(self):
        p = self.producer()
        def fail(points, state):
            raise ValueError('self_geometry_eef_fk_frame_mismatch')
        p.self_geometry = fail
        with patch.object(p.dino, 'surface') as legacy:
            result = self.update(p)
        legacy.assert_not_called()
        self.assertIn('self_geometry_eef_fk_frame_mismatch', result['reason'])
        self.assertFalse(result['verified'])

    def test_sparse_self_geometry_matches_full_mask(self):
        localizer = Task090DinoLocalizer(ARTIFACT)
        thresholds = SurfaceThresholds(**{**SETTINGS['thresholds'], 'min_component_pixels': 1})
        pos, neg = np.zeros(196), np.zeros(196)
        pos[[15, 20, 25]] = .9
        neg[20] = .9
        rgb, depth = np.zeros((14, 14, 3), np.uint8), np.ones((14, 14))
        provenance = {**SETTINGS['provenance'], 'robot_self_mask_source': 'synthetic FK exclusion'}
        full = np.zeros((14, 14), bool)
        full[1, 1] = True
        sizes = []
        def keep(points, state):
            sizes.append(len(points))
            return points[:, 0] > .005
        with patch.object(localizer, 'scores', return_value=(pos, neg)):
            sparse = localizer.surfaces(rgb, depth, Camera(), None, thresholds=thresholds,
                provenance=provenance, self_geometry=keep, state=np.zeros(61))
            dense = localizer.surfaces(rgb, depth, Camera(), None, thresholds=thresholds,
                provenance=provenance, robot_self_mask=full)
        self.assertEqual(sizes, [2])
        np.testing.assert_array_equal(sparse['labels'], dense['labels'])
        self.assertEqual(sparse['diagnostics'], dense['diagnostics'])

    def test_real_train_frame_keeps_inputs_and_evidence_boundary(self):
        p = self.producer()
        obs = self.observation()
        saved = {k: v.copy() for k, v in obs.items()}
        state = np.asarray(self.sample['state61'])
        state_before, features_before = state.copy(), self.features.copy()
        result = p.update(obs, state, episode=18001, step=870, head_features=self.features)
        for k, v in saved.items():
            np.testing.assert_array_equal(obs[k], v)
        np.testing.assert_array_equal(state, state_before)
        np.testing.assert_array_equal(self.features, features_before)
        self.assertFalse(result['verified'])
        self.assertFalse(result['admitted'])
        json.dumps(result)


if __name__ == '__main__':
    unittest.main()
