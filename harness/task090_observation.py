"""Task90 onboard measurement producer; observations, not admitted success labels.

Only first-bin grasp is covered. A visible surface sample is not the object center
or its complete lower boundary. Missing support geometry deliberately stays unknown.
"""
from __future__ import annotations

import numpy as np

from harness.task090_perception import (appearance_candidates, detect_from_observation,
    observed_floor_plane, Camera, camera_from_pose, head_pose, load_calibration)
from harness.task090_predicates import EEF, FINGERS, sensor_frame_sha256
from harness.task1_perception_adapter import find, HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX


class Task090ObservationProducer:
    def __init__(self, calibration_path, max_frame_gap=5, dino_artifact=None, surface_refinement=None):
        self.calibration_path = str(calibration_path)
        self.max_frame_gap = int(max_frame_gap)
        self.dino = None
        if dino_artifact is not None:
            from harness.task090_dino_localizer import Task090DinoLocalizer
            self.dino = Task090DinoLocalizer(dino_artifact)
        self.surface_thresholds = self.surface_provenance = self.self_geometry = None
        if surface_refinement is not None:
            from harness.instance_surface import SurfaceThresholds
            if self.dino is None or not isinstance(surface_refinement, dict):
                raise ValueError('surface refinement requires DINO and explicit settings')
            if set(surface_refinement) != {'thresholds', 'provenance', 'use_robot_self_geometry'}:
                raise ValueError('invalid surface refinement settings')
            self.surface_thresholds = SurfaceThresholds(**surface_refinement['thresholds'])
            self.surface_provenance = dict(surface_refinement['provenance'])
            if type(surface_refinement['use_robot_self_geometry']) is not bool:
                raise ValueError('use_robot_self_geometry must be explicit bool')
            if surface_refinement['use_robot_self_geometry']:
                from harness.robot_self_geometry import RobotSelfGeometry
                self.self_geometry = RobotSelfGeometry()
                self.surface_provenance['robot_self_mask_source'] = self.self_geometry.version
        self.reset()

    def reset(self):
        self.episode = self.last_step = self.previous = None
        self.seen = set()
        self.track_number = 0
        self.track_id = None

    def _unknown(self, reason, episode, step, **detail):
        self.previous = None
        self.track_id = None
        return {'status': 'unknown', 'verified': False, 'admitted': False, 'reason': reason,
                'episode': episode, 'observation_id': step, 'detections': [], 'detail': detail}

    def update(self, observation, state, *, episode, step, object_id='trash_can', head_features=None):
        if type(episode) is not int or type(step) is not int or episode < 0 or step < 0:
            return self._unknown('invalid_episode_or_step', episode, step)
        if self.episode is not None and episode < self.episode:
            return self._unknown('episode_regression', episode, step)
        if episode != self.episode:
            self.reset()
            self.episode = episode
        if self.last_step is not None and step <= self.last_step:
            return self._unknown('nonmonotonic_observation', episode, step)
        gap = None if self.last_step is None else step - self.last_step
        self.last_step = step
        if object_id != 'trash_can':
            return self._unknown('binding_not_implemented', episode, step)
        if not isinstance(observation, dict):
            return self._unknown('missing_sensor_dictionary', episode, step)
        forbidden = ('privbase_nav', 'object_scope', 'object_poses', 'goal_literals', 'q_score', 'reward', 'success')
        if any(any(str(key).startswith(prefix) for prefix in forbidden) for key in observation):
            return self._unknown('forbidden_non_sensor_input', episode, step)
        try:
            rgb, depth, cam_pose = (find(observation, suffix) for suffix in
                                   (HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX))
            if rgb is None or depth is None or cam_pose is None:
                return self._unknown('missing_rgbd_or_camera_pose', episode, step)
            rgb, depth, state = np.asarray(rgb), np.asarray(depth), np.asarray(state, dtype=float)
            if (rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or not np.isfinite(rgb).all()
                    or depth.squeeze().shape != rgb.shape[:2] or state.shape != (61,) or not np.isfinite(state).all()):
                return self._unknown('invalid_sensor_schema', episode, step)
            frame = sensor_frame_sha256(rgb, depth)
            if frame in self.seen:
                return self._unknown('repeated_sensor_frame', episode, step)
            self.seen.add(frame)
            camera = Camera(*camera_from_pose(head_pose(cam_pose)), rgb.shape[1], rgb.shape[0])
            dmap = depth.squeeze()
            floor = observed_floor_plane(dmap, camera)
            if self.dino is not None:
                if head_features is None:
                    return self._unknown('missing_current_dino_features', episode, step)
                if self.surface_thresholds is None:
                    proposed = self.dino.surface(rgb, dmap, camera, head_features, self.calibration_path)
                    if proposed is None:
                        return self._unknown('no_dino_refined_metric_surface', episode, step)
                    detected, cloud = proposed['detection'], proposed['cloud']
                else:
                    proposed = self.dino.surfaces(rgb, dmap, camera, head_features,
                        thresholds=self.surface_thresholds, provenance=self.surface_provenance,
                        self_geometry=self.self_geometry, state=state)
                    candidates = proposed['candidates']
                    if len(candidates) != 1:
                        return self._unknown('target_surface_missing_or_ambiguous', episode, step,
                            proposals=len(candidates), surface_status=proposed['status'],
                            surface_diagnostics=proposed['diagnostics'],
                            candidates=[{k: c[k] for k in ('component_id', 'bbox_xyxy', 'xyz_base',
                                'pixels', 'score_statistics')} for c in candidates])
                    candidate = candidates[0]
                    detected = {k: v for k, v in candidate.items() if k not in ('cloud', 'pixel_uv')}
                    cloud = candidate['cloud']
            else:
                detections, _ = detect_from_observation(rgb, depth, cam_pose, state,
                                                      calibration_path=self.calibration_path)
                matches = [d for d in detections if d['label'] == object_id]
                if len(matches) != 1:
                    return self._unknown('target_missing_or_ambiguous', episode, step,
                                         proposals=len(matches), floor_observed=floor is not None,
                                         sample_gap_steps=gap)
                detected = matches[0]
                proposals = appearance_candidates(rgb, self.calibration_path)
                pixels = next(p for p in proposals if p['label'] == object_id and
                              p['bbox_xyxy'] == detected['bbox_xyxy'])
                u, v = pixels['pixel_x'], pixels['pixel_y']
                z = dmap[v, u]
                valid = np.isfinite(z) & (z > .05) & (z < load_calibration(self.calibration_path)['depth_max_m'])
                cloud = camera.backproject(u[valid], v[valid], z[valid])
            if not len(cloud):
                return self._unknown('no_metric_surface', episode, step)
            point, eef = np.asarray(detected['xyz_base'], float), state[EEF['right']].copy()
            relative = point - eef
            contiguous = gap is not None and gap <= self.max_frame_gap and self.previous is not None
            # This is bounded candidate association, not identity certification or odometry extrapolation.
            associated = contiguous and np.linalg.norm(point - self.previous['point']) <= .04 + 1.5 * gap / 30.
            if not associated:
                self.track_number += 1
                self.track_id = f'candidate:{episode}:trash_can:{self.track_number}'
            measurement = {**detected, 'object_id': object_id, 'track_id': self.track_id,
                'episode': episode, 'observation_id': step, 'sensor_frame_sha256': frame,
                'hand': 'right', 'visible': True, 'metric_valid': True,
                'binding_status': 'unadmitted_visual_candidate',
                'hand_observation': {'eef_xyz_base': eef.tolist(),
                    'finger_positions_m': state[FINGERS['right']].tolist(),
                    'finger_sum_m': float(state[FINGERS['right']].sum())},
                'surface_extent_m': (np.percentile(cloud, 98, axis=0)-np.percentile(cloud, 2, axis=0)).tolist(),
                'object_hand_offset_m': relative.tolist(), 'sample_gap_steps': gap,
                'temporal_continuity_supported': bool(associated),
                'whole_object_geometry_observed': False}
            if associated:
                measurement['surface_motion_m'] = float(np.linalg.norm(point-self.previous['point']))
                measurement['eef_motion_m'] = float(np.linalg.norm(eef-self.previous['eef']))
                measurement['object_hand_offset_change_m'] = float(np.linalg.norm(relative-self.previous['relative']))
            if floor is not None:
                coef = floor['coefficients']
                clearance = cloud[:, 2] - np.c_[cloud[:, :2], np.ones(len(cloud))] @ coef
                reference = [float(point[0]), float(point[1]), float(np.r_[point[:2], 1.] @ coef)]
                measurement['relation'] = {'kind': 'support', 'reference': 'floors',
                    'reference_track_id': f'candidate:{episode}:floor-plane', 'reference_visible': True,
                    'reference_xyz_base': reference, 'plane_coefficients': coef.tolist(),
                    'plane_margin_m': float(floor['margin_m']), 'plane_points': int(floor['points']),
                    'visible_surface_clearance_p02_m': float(np.percentile(clearance, 2)),
                    'visible_surface_clearance_p98_m': float(np.percentile(clearance, 98)),
                    'whole_object_clearance_status': 'unknown_partial_surface'}
                # No support_clearance_m: the visible lower sample is not a bound on the whole object's bottom.
            self.previous = {'point': point, 'eef': eef, 'relative': relative}
            return {'status': 'measurements_available', 'verified': False, 'admitted': False,
                'reason': 'identity_and_whole_object_support_not_admitted', 'episode': episode,
                'observation_id': step, 'sensor_frame_sha256': frame, 'detections': [measurement],
                'detail': {'floor_observed': floor is not None, 'sparse_gap': gap is not None and gap > self.max_frame_gap}}
        except (ValueError, TypeError, KeyError, IndexError, StopIteration, np.linalg.LinAlgError) as error:
            return self._unknown(type(error).__name__ + ':' + str(error), episode, step)
