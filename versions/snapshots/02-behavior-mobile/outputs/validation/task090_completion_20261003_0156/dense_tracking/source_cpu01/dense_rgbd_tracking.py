"""Task-neutral, sensor-only local dense tracking; diagnostic candidates, not admission.

One explicit initial binding owns immutable RGB-D / camera / raw-pixel IDs / cloud.
The pose maps THAT reference base to the current Camera base, not to world. Only
accepted current surface fits advance it. Prediction selects a local ROI; it is
never an observation, a complete object envelope, or a cumulative error bound.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np
from scipy.spatial import cKDTree

from harness.dense_rgbd_registration import (
    CAPTURE, MIN_FIT, MIN_HOLD, checked_pose, partition, register, separated,
    support_report, surface, transform,
)


def immutable(value):
    """Bytes-backed arrays cannot be made writeable again by a caller."""
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class SensorReference:
    episode: object
    target: str
    binding_id: str
    step: int
    timestamp_s: float
    observation_id: str
    rgb: np.ndarray
    depth: np.ndarray
    state: np.ndarray
    camera: object
    cloud: object
    reference_id: str


class DenseSurfaceTracker:
    """No semantic rebind, model, action, gripper inference, odometry or file access.

    reset(episode, target) is an explicit external lifecycle operation, never a
    recovery action. initialize is allowed once per reset; an integration must
    authorize its initial binding separately (e.g. not after a closed-gripper loss).
    self_filter(points_base, state) returns a bool KEEP mask and must be pointwise;
    current_mask must depend only on current sensor data, not a saved trajectory.
    Timestamp/now share the caller's clock; replay timestamps cannot prove hardware
    freshness. A gap > max_frame_gap from last accepted observation latches unknown.
    """
    def __init__(self, *, depth_max, self_filter, max_age_s, max_frame_gap=6):
        if (type(max_frame_gap) is not int or not 1 <= max_frame_gap <= 6 or
                not np.isfinite(depth_max) or depth_max <= .05 or
                not np.isfinite(max_age_s) or max_age_s <= 0 or not callable(self_filter)):
            raise ValueError('invalid_tracker_contract')
        self.depth_max, self.self_filter = float(depth_max), self_filter
        self.max_frame_gap, self.max_age_s = max_frame_gap, float(max_age_s)
        self.reset(None, None)

    def reset(self, episode, target):
        self.episode, self.target = episode, target
        self._reference = self._accepted_pose = None
        self.last_accepted_step = self.last_accepted_timestamp = None
        self.last_seen_step = self.last_seen_timestamp = None
        self._seen_ids = set()
        self._attempted = self.expired = False

    @property
    def reference(self):
        return self._reference

    @property
    def accepted_pose(self):
        return self._accepted_pose

    def _report(self, status='unknown', reasons=(), **extra):
        return dict(status=status, reasons=list(reasons), episode=self.episode, target=self.target,
            reference_id=None if self.reference is None else self.reference.reference_id,
            accepted_pose_reference_to_last_accepted_base=None if self.accepted_pose is None else self.accepted_pose.tolist(),
            last_accepted_step=self.last_accepted_step, pose_updated=False, search_initial_pose=None,
            search_origin_step=None, cumulative_error_bound_m=None, part_identity='unknown',
            instance_scale='unknown', world_pose=None, verified=False, admitted=False, grasp=False,
            **extra)

    def _frame(self, *, episode, target, step, timestamp_s, now_s, observation_id):
        if episode != self.episode or target != self.target:
            raise ValueError('scope_mismatch_requires_explicit_reset_and_binding')
        if episode is None or not isinstance(target, str) or not target:
            raise ValueError('missing_episode_or_target')
        if type(step) is not int or step < 0 or not isinstance(observation_id, str) or not observation_id:
            raise ValueError('invalid_frame_identity')
        if not np.isfinite([timestamp_s, now_s]).all() or now_s < timestamp_s or now_s-timestamp_s > self.max_age_s:
            raise ValueError('stale_or_invalid_sensor_timestamp')
        if (self.last_seen_step is not None and
                (step <= self.last_seen_step or timestamp_s <= self.last_seen_timestamp)):
            raise ValueError('nonmonotonic_or_duplicate_frame')
        if observation_id in self._seen_ids:
            raise ValueError('repeated_observation_id')
        self.last_seen_step, self.last_seen_timestamp = step, timestamp_s
        self._seen_ids.add(observation_id)
        if self.last_accepted_step is not None and (
                step-self.last_accepted_step > self.max_frame_gap or
                timestamp_s-self.last_accepted_timestamp > self.max_age_s+1e-12):
            self.expired = True
        if self.expired:
            raise ValueError('accepted_reference_expired_no_automatic_rebind')

    def _cloud(self, rgb, depth, camera, state, mask, initial_pose=None, unit='m'):
        rgb, depth, state, mask = map(np.asarray, (rgb, depth, state, mask))
        if (rgb.shape != (camera.H, camera.W, 3) or rgb.dtype != np.uint8 or
                depth.shape != rgb.shape[:2] or mask.shape != depth.shape or mask.dtype != bool or
                state.ndim != 1 or not state.size or not np.isfinite(state).all()):
            raise ValueError('invalid_aligned_rgbd_state_or_mask')
        camera_pose = np.eye(4)
        camera_pose[:3, :3], camera_pose[:3, 3] = camera.R, camera.t
        checked_pose(camera_pose)
        if not np.isfinite([camera.f, camera.cx, camera.cy]).all() or camera.f <= 0 or unit != 'm':
            raise ValueError('invalid_camera_or_depth_unit')
        roi = mask & np.isfinite(depth) & (depth > .05) & (depth < self.depth_max)
        v, u = np.nonzero(roi)
        points = camera.backproject(u, v, depth[v, u])
        if initial_pose is not None:
            # Fit-fold reference only: a held-only depth perturbation cannot alter
            # ROI membership of any fit pixel. Current tests remain per-pixel.
            fit_ids, _ = partition(self.reference.cloud)
            prediction = transform(self.reference.cloud['points'][fit_ids], initial_pose)
            keep = cKDTree(prediction).query(points)[0] <= CAPTURE if len(prediction) else np.zeros(len(points), bool)
            roi[v[~keep], u[~keep]] = False
            v, u, points = v[keep], u[keep], points[keep]
        keep = np.asarray(self.self_filter(points, state))
        if keep.dtype != bool or keep.shape != (len(points),):
            raise ValueError('invalid_self_filter_keep_mask')
        robot = np.zeros(depth.shape, bool)
        robot[v, u] = ~keep
        return surface(depth, camera, roi, robot, self.depth_max, unit)

    def initialize(self, rgb, depth, camera, state, initial_mask, *, binding_id, episode,
                   target, step, timestamp_s, now_s, observation_id, unit='m'):
        if self._attempted:
            return self._report(reasons=['initial_binding_already_attempted_no_rebind']), {}
        trace = {}
        try:
            self._frame(episode=episode, target=target, step=step, timestamp_s=timestamp_s,
                        now_s=now_s, observation_id=observation_id)
            self._attempted = True
            if not isinstance(binding_id, str) or not binding_id:
                raise ValueError('missing_explicit_initial_binding')
            cloud = self._cloud(rgb, depth, camera, state, initial_mask, unit=unit)
            camera_record = {k: immutable(getattr(camera, k)) for k in ('R', 't', 'f', 'cx', 'cy', 'W', 'H')}
            owned = MappingProxyType({key: immutable(value) for key, value in cloud.items()})
            digest = hashlib.sha256(json.dumps([str(episode), target, binding_id, step, timestamp_s, observation_id]).encode())
            for value in (rgb, depth, state, *camera_record.values(), *owned.values()):
                digest.update(np.ascontiguousarray(value).tobytes())
            self._reference = SensorReference(episode, target, binding_id, step, timestamp_s, observation_id,
                immutable(rgb), immutable(depth), immutable(state), MappingProxyType(camera_record), owned, digest.hexdigest())
            fit, hold = partition(cloud)
            fit, hold = fit[separated(cloud['points'][fit])], hold[separated(cloud['points'][hold])]
            trace = dict(source_fit_indices=fit, source_hold_indices=hold)
            if len(fit) < MIN_FIT or len(hold) < MIN_HOLD:
                raise ValueError('insufficient_initial_surface_support')
            # Identity defines a coordinate reference to this REAL sensor cloud;
            # it is not a measured zero motion, known true pose, or zero error.
            self._accepted_pose = immutable(np.eye(4))
            self.last_accepted_step, self.last_accepted_timestamp = step, timestamp_s
            return self._report('reference_initialized', source_support=support_report(cloud['points'][fit], cloud['normals'][fit]),
                initial_binding='external_unadmitted_candidate', reference_coordinate_identity=True), trace
        except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as error:
            return self._report(reasons=[str(error)]), trace

    def update(self, rgb, depth, camera, state, current_mask, *, episode, target, step,
               timestamp_s, now_s, observation_id, unit='m'):
        trace, origin, origin_step = {}, None, None
        try:
            self._frame(episode=episode, target=target, step=step, timestamp_s=timestamp_s,
                        now_s=now_s, observation_id=observation_id)
            if self.reference is None or self.accepted_pose is None:
                raise ValueError('no_accepted_initial_reference_no_automatic_rebind')
            origin, origin_step = self.accepted_pose.copy(), self.last_accepted_step
            current = self._cloud(rgb, depth, camera, state, current_mask, initial_pose=origin, unit=unit)
            result, trace = register(self.reference.cloud, current, initial_pose=origin)
            trace.update({f'current_{key}': value for key, value in current.items()})
            updated = result['status'] == 'feasible_measurement_candidate'
            if updated:
                self._accepted_pose = immutable(checked_pose(result['transform_source_to_current_base']))
                self.last_accepted_step, self.last_accepted_timestamp = step, timestamp_s
            report = self._report('surface_motion_measured' if updated else 'unknown', result['reasons'], measurement=result)
            report.update(pose_updated=updated, search_initial_pose=origin.tolist(), search_origin_step=origin_step,
                roi_pixels=int(current['candidate_mask'].sum()), self_excluded_pixels=int(current['robot_mask'].sum()),
                current_surface_pixels=len(current['points']))
            return report, trace
        except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as error:
            report = self._report(reasons=[str(error)])
            report.update(search_initial_pose=None if origin is None else origin.tolist(), search_origin_step=origin_step)
            return report, trace
