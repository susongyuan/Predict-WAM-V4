"""Immutable original-landmark reobservation; CPU diagnostic, never admission.

Reuse v3's initialization, final 1cm consensus, floor/height margins, expiry,
recovery and candidate gates. Only correspondence maintenance changes: each
observation is measured against the INITIAL image, seeded by the last accepted
pose in the CURRENT camera. A seed is not a measurement. Temporarily invisible
landmarks remain in the bank; no dependent/replenished landmarks are introduced.
Initial semantic identity, repeated-texture identity and grasp are unadmitted.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from harness.task090_predicates import EEF, FINGERS
from harness.task090_rigid_observation_v3 import (
    Task090RigidObservationV3, FB_TOLERANCE_PX, MIN_INLIERS,
    original_anchor_check, sample_points, hand_separated,
)


def immutable_array(value):
    """Bytes-backed storage: even setflags(write=True) cannot mutate the bank."""
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class OriginalLandmarks:
    gray: np.ndarray
    uv: np.ndarray
    reference: np.ndarray
    errors: np.ndarray


def seeded_flow_matches(original_gray, gray, original_uv, projected_uv):
    """Forward AND reverse LK measure real images; neither seed is accepted alone."""
    uv = np.asarray(original_uv, np.float32)
    seed = np.asarray(projected_uv, np.float32)
    if (gray.shape != original_gray.shape or uv.shape != seed.shape or
            uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all() or
            not np.isfinite(seed).all()):
        raise ValueError('invalid_original_flow_inputs')
    if not len(uv):
        return uv.copy(), np.zeros(0, bool)
    options = dict(winSize=(31, 31), maxLevel=3,
                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    current, ok, _ = cv2.calcOpticalFlowPyrLK(
        original_gray, gray, uv[:, None].copy(), seed[:, None].copy(),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01), **options)
    if current is None or ok is None:
        raise ValueError('optical_flow_missing')
    # Do not pass invalid forward outputs into OpenCV's reverse call.
    finite = np.isfinite(current[:, 0]).all(axis=1)
    reverse_input = current.copy()
    reverse_input[~finite, 0] = seed[~finite]
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(
        gray, original_gray, reverse_input, uv[:, None].copy(), **options)
    if back is None or back_ok is None:
        raise ValueError('backward_flow_missing')
    measured = current[:, 0]
    valid = (ok[:, 0].astype(bool) & back_ok[:, 0].astype(bool) & finite &
             np.isfinite(back[:, 0]).all(axis=1) &
             (np.linalg.norm(back[:, 0] - uv, axis=1) <= FB_TOLERANCE_PX))
    h, w = gray.shape
    valid &= ((measured[:, 0] >= 1) & (measured[:, 0] < w - 1) &
              (measured[:, 1] >= 1) & (measured[:, 1] < h - 1))
    return measured, valid


class Task090RigidObservationV4(Task090RigidObservationV3):
    """Original-only direct tracking with independent bank and accepted visibility."""

    def __init__(self, *args, self_filter=None, **kwargs):
        # Explicit opt-in only. None preserves the original 0.2m default path.
        if self_filter is not None and not callable(self_filter):
            raise ValueError('invalid_self_filter')
        self.self_filter = self_filter
        self._current_state = None
        super().__init__(*args, **kwargs)

    def _self_keep(self, points, state, diagnostic, stage):
        try:
            keep = np.asarray(self.self_filter(points.copy(), np.asarray(state).copy()))
        except Exception as error:
            raise ValueError('self_filter_failed:' + str(error)) from error
        if keep.shape != (len(points),) or keep.dtype != np.dtype(bool):
            raise ValueError('self_filter_invalid_keep_mask')
        diagnostic[stage + '_before_self_filter'] = len(points)
        diagnostic[stage + '_after_self_filter'] = int(keep.sum())
        diagnostic[stage + '_self_rejected'] = int((~keep).sum())
        return keep

    def reset(self):
        super().reset()
        self.original_bank = None
        self.last_accepted_mask = None
        self._pending_mask = None
        self._current_hands = None

    def _initialize(self, observation, state, episode, step, features, gray, depth,
                    camera, floor, mask, diagnostic):
        if self.self_filter is None:
            super()._initialize(observation, state, episode, step, features, gray, depth,
                                camera, floor, mask, diagnostic)
        else:
            # Mirror the frozen parent's initializer, changing only self-exclusion.
            # Do not run its 0.2m filter first: it would erase near-hand targets.
            if float(state[FINGERS[self.hand]].sum()) < .095:
                raise ValueError('cannot_rebind_target_after_gripper_closes')
            self._self_keep(np.empty((0, 3)), state, diagnostic, 'state_validation')
            proposal = self.measure.update(observation, state, episode=episode, step=step,
                                          object_id=self.target, head_features=features)
            if proposal['status'] != 'measurements_available':
                raise ValueError(proposal['reason'])
            matches = [d for d in proposal['detections'] if d['object_id'] == self.target]
            if len(matches) != 1:
                raise ValueError('target_missing_or_ambiguous')
            box = matches[0]['bbox_xyxy']
            owned = np.zeros_like(mask)
            owned[box[1]:box[3], box[0]:box[2]] = mask[box[1]:box[3], box[0]:box[2]]
            found = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=.005,
                                           minDistance=6, mask=owned, blockSize=5)
            if found is None or len(found) < MIN_INLIERS:
                raise ValueError('not_enough_object_features')
            uv = found[:, 0]
            points = sample_points(camera, depth, uv)
            keep = self._self_keep(points, state, diagnostic, 'initial_anchors')
            uv, points = uv[keep], points[keep]
            if len(points) < MIN_INLIERS:
                raise ValueError('features_not_separate_from_robot')
            v, u = np.nonzero(owned)
            surface = camera.backproject(u, v, depth[v, u])
            surface = surface[self._self_keep(surface, state, diagnostic, 'initial_surface')]
            self.origin_surface = surface.copy()
            self.uv, self.reference = uv.copy(), points.copy()
            self.errors = np.zeros(len(points))
            self.anchor_uv, self.anchor_reference = uv.copy(), points.copy()
            self.anchor_total = len(points)
            self.origin_floor = floor['coefficients'].copy()
            self.origin_margin = float(floor['margin_m'])
            self.previous_pose = (np.eye(3), np.zeros(3))
            self.track_id = f'anchored:{episode}:{self.target}:{step}'
            self.keyframes = 1
            diagnostic.update(status='tracking_initialized', points=len(points),
                              original_anchor_inliers=len(points), overlap_inliers=len(points),
                              origin_surface_points=len(surface), replenished_points=0,
                              cumulative_reference_error_m=0., target_binding='unadmitted_initial_candidate')
        self.original_bank = OriginalLandmarks(
            immutable_array(gray), immutable_array(self.anchor_uv),
            immutable_array(self.anchor_reference), immutable_array(self.errors))
        self._pending_mask = np.ones(self.anchor_total, bool)

    def _tracked(self, gray, depth, camera, appearance, diagnostic):
        bank = self.original_bank
        if bank is None:
            raise ValueError('original_landmark_bank_missing')
        r0, t0 = self.previous_pose
        predicted = bank.reference @ r0.T + t0
        camera_points = (predicted - camera.t) @ camera.R
        positive = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > .05)
        seeds = np.zeros((len(bank.uv), 2), np.float32)
        seeds[positive] = (camera_points[positive, :2] / camera_points[positive, 2, None]
                           * camera.f + [camera.cx, camera.cy])
        h, w = gray.shape
        visible = (positive & np.isfinite(seeds).all(axis=1) &
                   (seeds[:, 0] >= 1) & (seeds[:, 0] < w-1) &
                   (seeds[:, 1] >= 1) & (seeds[:, 1] < h-1))
        ids = np.flatnonzero(visible)
        diagnostic.update(original_bank_size=len(bank.uv), original_projection_seeds=len(ids),
                          original_anchors_after_lk_depth_appearance=0,
                          original_anchors_after_hand_separation=0,
                          reobserved_original_anchors=0,
                          pose_source='original_anchors_only')
        uv, valid = seeded_flow_matches(bank.gray, gray, bank.uv[ids], seeds[ids])
        diagnostic['original_anchors_after_lk'] = int(valid.sum())
        ids, uv = ids[valid], uv[valid]
        u, v = np.rint(uv).astype(int).T
        depth_valid = (np.isfinite(depth[v, u]) & (depth[v, u] > .05) &
                       (depth[v, u] < self.calibration['depth_max_m']))
        diagnostic['original_anchors_after_depth'] = int(depth_valid.sum())
        valid = depth_valid & (appearance[v, u] > 0)
        diagnostic['original_anchors_after_appearance'] = int(valid.sum())
        ids, uv = ids[valid], uv[valid]
        diagnostic['original_anchors_after_lk_depth_appearance'] = len(ids)
        points = sample_points(camera, depth, uv)
        # Hands can exclude observations, never produce the object's pose or UV.
        if self.self_filter is None:
            hands = self._current_hands
            if hands is None:
                raise ValueError('current_hand_observations_missing')
            keep = hand_separated(points, hands)
        else:
            keep = self._self_keep(points, self._current_state, diagnostic, 'original_anchors')
        ids, uv, points = ids[keep], uv[keep], points[keep]
        diagnostic['original_anchors_after_hand_separation'] = len(ids)
        reference = bank.reference[ids]
        r, t, rms, keep, maximum = original_anchor_check(reference, points, diagnostic=diagnostic)
        ids, uv, points, reference = ids[keep], uv[keep], points[keep], reference[keep]
        errors = bank.errors[ids].copy()
        # Original references are never inverted from prior estimated poses. The
        # unchanged v3 height bound already includes sensor/floor/fit uncertainty.
        # Preserve any actually-used reference margin, including across recovery;
        # never zero a live high-water value to obtain a positive height bound.
        self.error_high_water = max(self.error_high_water, float(np.max(errors)))
        self._pending_mask = np.zeros(len(bank.uv), bool)
        self._pending_mask[ids] = True
        reobserved = 0 if self.last_accepted_mask is None else int(
            np.sum(self._pending_mask & ~self.last_accepted_mask))
        diagnostic.update(overlap_after_lk_depth_appearance=len(reference),
                          overlap_inliers=len(reference), fit_rms_m=rms,
                          overlap_pose_error_max_m=maximum,
                          reobserved_original_anchors=reobserved,
                          accepted_original_ids=ids.tolist())
        return uv, reference, points, errors, r, t, rms, uv.copy(), reference.copy()

    def _replenish(self, gray, depth, camera, appearance, hands, uv, reference,
                   points, errors, r, t, rms):
        """Explicit no-op: no new point, dependent error or pose pseudo-observation."""
        return uv, reference, points, errors, dict(
            replenished_points=0, refill_candidates=0, refill_owned_depth=0,
            refill_bidirectional_rigid=0)

    def update(self, observation, state, *, episode, step, head_features, action=None, object_id=None):
        self._pending_mask = None
        self._current_state = state
        # The parent validates the complete schema before _tracked is called.
        # It resets on a new episode, which only takes the initialization path.
        try:
            state_array = np.asarray(state, float)
            self._current_hands = ([state_array[EEF[h]].copy() for h in ('left', 'right')]
                                   if state_array.shape == (61,) and np.isfinite(state_array).all() else None)
        except (TypeError, ValueError):
            self._current_hands = None
        report = super().update(observation, state, episode=episode, step=step,
                                head_features=head_features, action=action, object_id=object_id)
        diagnostic = report['rigid_tracking']
        accepted = diagnostic['status'] in (
            'tracking_initialized', 'rigid_motion_measured', 'anchor_recovery_pending')
        if accepted and self._pending_mask is not None:
            self.last_accepted_mask = self._pending_mask.copy()
        diagnostic.update(self_filter_version='legacy_eef_radius_0.2m' if self.self_filter is None else
                          getattr(self.self_filter, 'version', 'explicit_callable'),
                          tracker_version='original_bank_rgbd_v4',
                          original_bank_size=0 if self.original_bank is None else len(self.original_bank.uv),
                          last_accepted_original_count=0 if self.last_accepted_mask is None else int(self.last_accepted_mask.sum()),
                          dependent_reference_points_used=0,
                          reference_uncertainty_source='initial_sensor_and_direct_original_fit_only')
        self._pending_mask = self._current_hands = self._current_state = None
        return report
