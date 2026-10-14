"""Anchored RGB-D rigid tracking, diagnostic only; no manipulation admission.

A track owns one immutable object reference and one observed origin floor. Candidate
lifetimes are separate from the track. Initialization reuses the existing TRAIN
appearance proposal; subsequent frames never rerun navigation/grounded-object gates.

New points need >=6 original anchors AND >=6 overlapping rigid inliers. They must
match the initial observed target surface, current target appearance, depth and a
backward/forward image correspondence, and be outside both hands. Reference errors
are inherited, accumulated and never reset by replenishment or recovery. None of
these checks certifies initial semantic identity or absolute support clearance.
"""
from __future__ import annotations

import cv2
import numpy as np

from harness.object_perception import Camera, camera_from_pose, head_pose
from harness.task090_observation import Task090ObservationProducer
from harness.task090_perception import load_calibration, observed_floor_plane
from harness.task090_predicates import EEF, FINGERS, GRIP_ACTION, sensor_frame_sha256
from harness.task090_rigid_observation_v2 import (
    fit_rigid, robust_rigid, height_change_bound, commanded_finger_sum,
)
from harness.task1_perception_adapter import (
    find, HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX,
)

RIGID_TOLERANCE_M = .01
FB_TOLERANCE_PX = 1.5
MIN_INLIERS = 6


def flow_matches(old_gray, gray, uv):
    """Fixed v2 forward/backward LK thresholds, without moving DINO ROI clipping."""
    current, ok, _ = cv2.calcOpticalFlowPyrLK(
        old_gray, gray, np.asarray(uv, np.float32)[:, None], None,
        winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    if current is None:
        raise ValueError('optical_flow_missing')
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(
        gray, old_gray, current, None, winSize=(31, 31), maxLevel=3)
    if back is None:
        raise ValueError('backward_flow_missing')
    current = current[:, 0]
    valid = (ok[:, 0].astype(bool) & back_ok[:, 0].astype(bool)
             & np.isfinite(current).all(axis=1)
             & (np.linalg.norm(back[:, 0] - uv, axis=1) <= FB_TOLERANCE_PX))
    h, w = gray.shape
    valid &= ((current[:, 0] >= 1) & (current[:, 0] < w - 1)
              & (current[:, 1] >= 1) & (current[:, 1] < h - 1))
    return current, valid


def sample_points(camera, depth, uv):
    u, v = np.rint(uv).astype(int).T
    return camera.backproject(uv[:, 0], uv[:, 1], depth[v, u])


def nearest_surface_distance(points, surface):
    """Bounded temporary arrays; all distances remain in the original reference."""
    if not len(surface):
        return np.full(len(points), np.inf)
    distances = []
    for start in range(0, len(points), 16):
        delta = points[start:start + 16, None] - surface[None]
        distances.extend(np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
    return np.asarray(distances)


def hand_separated(points, hands, radius=.2):
    return np.all(np.linalg.norm(points[:, None] - np.asarray(hands)[None], axis=2) > radius, axis=1)


def consistent_rigid(reference, points):
    """Refine the v2 consensus monotonically at the unchanged 1cm threshold.

    v2 returns the RANSAC mask from *before* its last refit. Each refit here can
    only remove members, and every return certifies that its >=6 non-collinear
    members satisfy the returned pose. Removed points never re-enter the fit.
    """
    reference, points = np.asarray(reference, float), np.asarray(points, float)
    if (reference.shape != points.shape or reference.ndim != 2 or reference.shape[1] != 3
            or not np.isfinite(reference).all() or not np.isfinite(points).all()):
        raise ValueError('invalid_correspondences')
    _, _, _, keep = robust_rigid(reference, points, RIGID_TOLERANCE_M)
    keep = keep.copy()
    while True:
        if int(keep.sum()) < MIN_INLIERS:
            raise ValueError('too_few_final_consistent_correspondences')
        # Reuse the original singular-value degeneracy check without relaxing it.
        r, t, _ = fit_rigid(reference[keep], points[keep])
        residual = np.linalg.norm(reference @ r.T + t - points, axis=1)
        retained = keep & (residual <= RIGID_TOLERANCE_M)
        if np.array_equal(retained, keep):
            return r, t, float(np.sqrt(np.mean(residual[keep]**2))), keep
        keep = retained


def original_anchor_check(reference, points, rotation=None, translation=None, diagnostic=None):
    """Authoritative original-only fit; optionally check a separately supplied pose."""
    r, t, rms, keep = consistent_rigid(reference, points)
    if rotation is None and translation is None:
        rotation, translation = r, t
    elif rotation is None or translation is None:
        raise ValueError('incomplete_comparison_pose')
    discrepancy = np.linalg.norm(reference[keep] @ rotation.T + translation - points[keep], axis=1)
    if diagnostic is not None:
        diagnostic.update(original_anchor_inliers=int(keep.sum()), original_anchor_fit_rms_m=rms,
                          original_anchor_pose_error_max_m=float(np.max(discrepancy)),
                          original_anchor_pose_error_rms_m=float(np.sqrt(np.mean(discrepancy**2))),
                          original_anchor_pose_inliers_at_1cm=int(np.sum(discrepancy <= RIGID_TOLERANCE_M)))
    if len(discrepancy) < MIN_INLIERS or np.max(discrepancy) > RIGID_TOLERANCE_M:
        raise ValueError('original_anchor_pose_disagreement')
    return r, t, rms, keep, float(np.max(discrepancy))


class Task090RigidObservationV3:
    """Target/hand-parameterized adapter with a conservative original-anchor bank.

    The bundled DINO initializer only supports trash_can. Other targets require an
    explicit measurement initializer with the same update contract; there is no
    silent fallback to the bin prototype. The tracker mechanics are target-agnostic.
    """
    requires_applied_action = True

    def __init__(self, calibration_path, dino_artifact=None, max_frame_gap=6,
                 *, target='trash_can', hand='right', initializer=None):
        self.calibration_path = str(calibration_path)
        self.calibration = load_calibration(self.calibration_path)
        if hand not in EEF or target not in self.calibration['classes']:
            raise ValueError('unknown_target_or_hand')
        if initializer is None and target != 'trash_can':
            raise ValueError('target_requires_its_own_initializer')
        if not 1 <= max_frame_gap <= 6:
            raise ValueError('cannot_relax_v2_frame_gap')
        self.target, self.hand = target, hand
        self.other_hand = 'left' if hand == 'right' else 'right'
        self.action_index = GRIP_ACTION[hand]
        self.max_frame_gap = max_frame_gap
        self.measure = initializer if initializer is not None else Task090ObservationProducer(
            self.calibration_path, max_frame_gap=max_frame_gap, dino_artifact=dino_artifact)
        prior = self.calibration['classes'][target]['geometry_prior']
        self.diameter = float(np.linalg.norm(prior['native_bbox_m']))
        self.reset()

    def reset_contract(self, contract=None):
        """A new skill resets only its evidence count, never the object reference."""
        self.contract = contract
        self.candidate_start = None
        self.candidate_count = 0

    def reset(self):
        self.episode = self.last_step = self.last_good_step = None
        self.seen = set()
        self.uv = self.reference = self.errors = None
        self.anchor_uv = self.anchor_reference = None
        self.origin_floor = self.origin_margin = self.origin_surface = None
        self.previous_gray = self.previous_depth = self.previous_camera = None
        self.previous_eef = self.previous_width = self.previous_pose = None
        self.anchor_total = 0
        self.error_high_water = 0.
        self.total_added = self.keyframes = 0
        self.lost = False
        self.recovery_count = 0
        self.track_id = None
        self.reset_contract()
        self.measure.reset()

    def _unknown(self, diagnostic, reason):
        # Keep the immutable reference and last *accepted* anchor frame. Never
        # automatically initialize a different object, even while the hand is open.
        self.lost = self.origin_surface is not None
        self.recovery_count = 0
        self.reset_contract(self.contract)
        diagnostic.update(status='unknown', reason=reason, candidate_grasp=False,
                          candidate_observations=0, recovery_observations=0,
                          cumulative_reference_error_m=self.error_high_water,
                          total_replenished_points=self.total_added,
                          original_anchors_retained=0 if self.anchor_uv is None else len(self.anchor_uv))

    def _appearance(self, rgb, depth):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = np.zeros(depth.shape, np.uint8)
        for lower, upper in self.calibration['classes'][self.target]['hsv_boxes']:
            mask |= cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8))
        mask[~np.isfinite(depth) | (depth <= .05) | (depth >= self.calibration['depth_max_m'])] = 0
        return mask

    def _initialize(self, observation, state, episode, step, features, gray, depth, camera, floor, mask, diagnostic):
        if float(state[FINGERS[self.hand]].sum()) < .095:
            raise ValueError('cannot_rebind_target_after_gripper_closes')
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
        hands = [state[EEF[h]] for h in ('left', 'right')]
        keep = hand_separated(points, hands)
        uv, points = uv[keep], points[keep]
        if len(points) < MIN_INLIERS:
            raise ValueError('features_not_separate_from_hand')
        # An immutable dense *observed* surface, not a simulator mesh. No plane or
        # support-height assumption is applied to it once the track has begun.
        v, u = np.nonzero(owned)
        surface = camera.backproject(u, v, depth[v, u])
        surface = surface[hand_separated(surface, hands)]
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

    def _tracked(self, gray, depth, camera, appearance, diagnostic):
        # Original anchors alone determine pose. Inverted/replenished references
        # are dependent evidence: they may agree with this pose, but never fit it.
        auv, avalid = flow_matches(self.previous_gray, gray, self.anchor_uv)
        u, v = np.rint(np.nan_to_num(auv)).astype(int).T
        u, v = np.clip(u, 0, gray.shape[1] - 1), np.clip(v, 0, gray.shape[0] - 1)
        avalid &= appearance[v, u] > 0
        diagnostic['original_anchors_after_lk_depth_appearance'] = int(avalid.sum())
        apoints = sample_points(camera, depth, auv[avalid])
        aref = self.anchor_reference[avalid]
        r, t, arms, akeep, _ = original_anchor_check(aref, apoints, diagnostic=diagnostic)
        diagnostic['pose_source'] = 'original_anchors_only'

        uv, valid = flow_matches(self.previous_gray, gray, self.uv)
        u, v = np.rint(np.nan_to_num(uv)).astype(int).T
        u, v = np.clip(u, 0, gray.shape[1] - 1), np.clip(v, 0, gray.shape[0] - 1)
        valid &= appearance[v, u] > 0
        diagnostic['overlap_after_lk_depth_appearance'] = int(valid.sum())
        points = sample_points(camera, depth, uv[valid])
        reference, errors = self.reference[valid], self.errors[valid]
        residual = np.linalg.norm(reference @ r.T + t - points, axis=1)
        keep = residual <= RIGID_TOLERANCE_M
        diagnostic['overlap_inliers'] = int(keep.sum())
        if int(keep.sum()) < MIN_INLIERS:
            raise ValueError('too_few_anchor_consistent_overlap')
        # Validate overlap geometry, but discard this auxiliary fit's pose: even
        # many near-consistent added points must not bias the independent anchors.
        fit_rigid(reference[keep], points[keep])
        rms = float(np.sqrt(np.mean(residual[keep]**2)))
        diagnostic.update(fit_rms_m=rms, overlap_pose_error_max_m=float(np.max(residual[keep])))
        uv, points, reference, errors = uv[valid][keep], points[keep], reference[keep], errors[keep]
        return (uv, reference, points, errors, r, t, max(rms, arms),
                auv[avalid][akeep], aref[akeep])

    def _replenish(self, gray, depth, camera, appearance, hands, uv, reference, points, errors, r, t, rms):
        """Only extend an accepted pose, never restart from DINO or the EEF."""
        stats = dict(replenished_points=0, refill_candidates=0, refill_owned_depth=0,
                     refill_bidirectional_rigid=0)
        if len(points) < MIN_INLIERS or len(uv) >= 80:
            return uv, reference, points, errors, stats
        transformed = self.origin_surface @ r.T + t
        cpoints = (transformed - camera.t) @ camera.R
        positive = cpoints[:, 2] > .05
        projected = np.zeros((len(cpoints), 2))
        projected[positive] = cpoints[positive, :2] / cpoints[positive, 2, None] * camera.f + [camera.cx, camera.cy]
        pix = np.rint(projected).astype(int)
        visible = positive & (pix[:, 0] >= 1) & (pix[:, 0] < gray.shape[1] - 1) & (pix[:, 1] >= 1) & (pix[:, 1] < gray.shape[0] - 1)
        ids = np.flatnonzero(visible)
        ids = ids[np.abs(depth[pix[ids, 1], pix[ids, 0]] - cpoints[ids, 2]) <= RIGID_TOLERANCE_M]
        owned = np.zeros_like(appearance)
        owned[pix[ids, 1], pix[ids, 0]] = 255
        # 1px rasterization support only. Metric membership still requires <=1cm.
        owned = cv2.dilate(owned, np.ones((3, 3), np.uint8)) & appearance
        for x, y in np.rint(uv).astype(int):
            cv2.circle(owned, (int(x), int(y)), 6, 0, -1)
        found = cv2.goodFeaturesToTrack(gray, maxCorners=80-len(uv), qualityLevel=.005,
                                       minDistance=6, mask=owned, blockSize=5)
        if found is None:
            return uv, reference, points, errors, stats
        new_uv = found[:, 0]
        new_points = sample_points(camera, depth, new_uv)
        new_reference = (new_points - t) @ r
        distance = nearest_surface_distance(new_reference, self.origin_surface)
        valid = (distance <= RIGID_TOLERANCE_M) & hand_separated(new_points, hands)
        stats['refill_candidates'] = len(new_uv)
        stats['refill_owned_depth'] = int(valid.sum())
        if not valid.any():
            return uv, reference, points, errors, stats
        new_uv, new_points, new_reference = new_uv[valid], new_points[valid], new_reference[valid]
        distance = distance[valid]
        # A new corner must also have been a rigidly corresponding target pixel in
        # the previous accepted frame. Newly revealed/occluded surfaces are unknown.
        back_uv, good = flow_matches(gray, self.previous_gray, new_uv)
        u, v = np.rint(np.nan_to_num(back_uv)).astype(int).T
        u, v = np.clip(u, 0, gray.shape[1]-1), np.clip(v, 0, gray.shape[0]-1)
        previous_z = self.previous_depth[v, u]
        good &= np.isfinite(previous_z) & (previous_z > .05) & (previous_z < self.calibration['depth_max_m'])
        ids = np.flatnonzero(good)
        if not len(ids):
            return uv, reference, points, errors, stats
        previous_points = sample_points(self.previous_camera, self.previous_depth, back_uv[ids])
        old_r, old_t = self.previous_pose
        residual = np.linalg.norm(previous_points - (new_reference[ids] @ old_r.T + old_t), axis=1)
        old_hands = self.previous_hands
        accepted = (residual <= RIGID_TOLERANCE_M) & hand_separated(previous_points, old_hands)
        ids, residual = ids[accepted], residual[accepted]
        stats['refill_bidirectional_rigid'] = len(ids)
        if not len(ids):
            return uv, reference, points, errors, stats
        # Never reset a newly inverted reference to zero uncertainty. The inherited
        # high-water bound also survives dropping old points, failures and recovery.
        inherited = max(self.error_high_water, float(np.max(errors)))
        new_error = inherited + 3*rms + distance[ids] + residual
        self.error_high_water = max(self.error_high_water, float(np.max(new_error)))
        self.total_added += len(ids)
        self.keyframes += 1
        stats['replenished_points'] = len(ids)
        return (np.concatenate([uv, new_uv[ids]]), np.concatenate([reference, new_reference[ids]]),
                np.concatenate([points, new_points[ids]]), np.concatenate([errors, new_error]), stats)

    def update(self, observation, state, *, episode, step, head_features, action=None, object_id=None):
        diagnostic = dict(status='unknown', candidate_grasp=False, verified=False, admitted=False,
                          absolute_support_clearance='unknown', tracker_version='anchored_rgbd_v3',
                          target=self.target, hand=self.hand,
                          assumptions=['correct initial material-point/target identity', 'rigid tracked body',
                                       'valid diameter and sensor margins'], replenished_points=0)
        report = dict(status='unknown', verified=False, admitted=False, episode=episode,
                      observation_id=step, detections=[], rigid_tracking=diagnostic)
        try:
            if type(episode) is not int or type(step) is not int or min(episode, step) < 0:
                raise ValueError('invalid_episode_or_step')
            if self.episode is not None and episode < self.episode:
                raise ValueError('episode_regression')
            if episode != self.episode:
                self.reset()
                self.episode = episode
            if object_id is not None and object_id != self.target:
                raise ValueError('track_target_mismatch')
            if self.last_step is not None and step <= self.last_step:
                raise ValueError('nonmonotonic_observation')
            gap = None if self.last_step is None else step - self.last_step
            self.last_step = step
            if gap is not None and gap > self.max_frame_gap:
                raise ValueError('observation_gap_exceeds_limit')
            if not isinstance(observation, dict):
                raise ValueError('missing_sensor_dictionary')
            forbidden = ('privbase_nav', 'object_scope', 'object_poses', 'goal_literals', 'q_score', 'reward', 'success')
            if any(str(k).startswith(forbidden) for k in observation):
                raise ValueError('forbidden_non_sensor_input')
            rgb, depth, pose = [find(observation, suffix) for suffix in
                                (HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX)]
            if rgb is None or depth is None or pose is None:
                raise ValueError('missing_rgbd_or_camera_pose')
            rgb, depth, state = np.asarray(rgb), np.asarray(depth).squeeze(), np.asarray(state, float)
            if (rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or not np.isfinite(rgb).all()
                    or depth.shape != rgb.shape[:2] or state.shape != (61,) or not np.isfinite(state).all()):
                raise ValueError('invalid_sensor_schema')
            frame_hash = sensor_frame_sha256(rgb, depth)
            if frame_hash in self.seen:
                raise ValueError('repeated_sensor_frame')
            self.seen.add(frame_hash)
            report['sensor_frame_sha256'] = frame_hash
            rgb = rgb[..., :3].astype(np.uint8)
            camera_pose = camera_from_pose(head_pose(pose))
            if camera_pose is None:
                raise ValueError('invalid_camera_pose')
            camera = Camera(*camera_pose, rgb.shape[1], rgb.shape[0])
            floor = observed_floor_plane(depth, camera)
            if floor is None:
                raise ValueError('floor_not_observed')
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            appearance = self._appearance(rgb, depth)
            eef = state[EEF[self.hand]]
            hands = [state[EEF[h]].copy() for h in ('left', 'right')]
            width = float(state[FINGERS[self.hand]].sum())
            if self.origin_surface is None:
                self._initialize(observation, state, episode, step, head_features, gray, depth,
                                 camera, floor, appearance, diagnostic)
            else:
                if step - self.last_good_step > self.max_frame_gap:
                    raise ValueError('original_anchor_keyframe_expired_no_rebind')
                result = self._tracked(gray, depth, camera, appearance, diagnostic)
                uv, reference, points, errors, r, t, rms, auv, aref = result
                was_lost = self.lost
                if was_lost:
                    self.recovery_count += 1
                    self.reset_contract(self.contract)
                    if self.recovery_count >= 3:
                        self.lost = False
                # During recovery, prove original overlap over consecutive observations
                # first; no speculative bank growth. The recovered frame itself cannot
                # contribute to a held-object candidate.
                if not was_lost:
                    uv, reference, points, errors, refill = self._replenish(
                        gray, depth, camera, appearance, hands, uv, reference, points, errors, r, t, rms)
                    diagnostic.update(refill)
                # Compute all geometry against the initial object coordinates and floor.
                # Added points cannot improve the origin span or erase original errors:
                # use original-only correspondences for the height bound.
                original_current = sample_points(camera, depth, auv)
                bound = height_change_bound(aref, original_current, r, self.origin_floor,
                                           floor['coefficients'], self.diameter, rms,
                                           self.origin_margin + float(floor['margin_m']))
                bound['assumed_measurement_allowance_m'] += self.error_high_water
                bound['conditional_min_height_gain_m'] -= self.error_high_water
                old_r, old_t = self.previous_pose
                delta_r = r @ old_r.T
                delta_t = t - delta_r @ old_t
                hand_error = float(np.linalg.norm(delta_r @ self.previous_eef + delta_t - eef))
                near = float(np.min(np.linalg.norm(points - eef, axis=1)))
                applied = np.asarray(action, float) if action is not None else None
                closing = bool(applied is not None and applied.shape == (23,) and np.isfinite(applied).all()
                               and applied[self.action_index] < 0)
                requested = commanded_finger_sum(applied[self.action_index]) if closing else None
                closure_error = None if requested is None else width-requested
                obstructed = bool(closing and closure_error > .004)
                stable = .01 <= width <= .09 and abs(width-self.previous_width) <= .001
                other_open = float(state[FINGERS[self.other_hand]].sum()) >= .095
                candidate = bool(not was_lost and obstructed and stable and other_open and near <= .25
                                 and hand_error <= .02 and bound['conditional_min_height_gain_m'] >= .03 and gap == 1)
                if candidate:
                    self.candidate_start = step if self.candidate_start is None else self.candidate_start
                    self.candidate_count += 1
                else:
                    self.reset_contract(self.contract)
                held = bool(candidate and self.candidate_count >= 3 and step-self.candidate_start >= 15)
                diagnostic.update(status='anchor_recovery_pending' if self.lost else 'rigid_motion_measured',
                                  points=len(points), candidate_grasp=held, candidate_observations=self.candidate_count,
                                  recovery_observations=self.recovery_count, recovery_frame=was_lost,
                                  sample_gap_steps=gap, sparse_sampling=gap != 1,
                                  hand_motion_residual_m=hand_error, nearest_tracked_point_to_eef_m=near,
                                  gripper_contact_candidate=bool(obstructed and stable),
                                  commanded_finger_sum_m=requested, closure_tracking_error_m=closure_error,
                                  assumed_rigid_body_diameter_m=self.diameter,
                                  cumulative_reference_error_m=self.error_high_water, **bound)
                self.uv, self.reference, self.errors = uv, reference, errors
                self.anchor_uv, self.anchor_reference = auv, aref
                self.previous_pose = (r, t)
            self.previous_gray = gray.copy()
            self.previous_depth = depth.copy()
            self.previous_camera = camera
            self.previous_eef, self.previous_width = eef.copy(), width
            self.previous_hands = hands
            self.last_good_step = step
            diagnostic.update(track_id=self.track_id, origin_anchor_count=self.anchor_total,
                              original_anchors_retained=len(self.anchor_uv),
                              total_replenished_points=self.total_added, keyframes=self.keyframes)
            report.update(status='measurements_available' if not self.lost else 'unknown',
                          reason='diagnostic_not_admitted')
        except (ValueError, TypeError, KeyError, IndexError, cv2.error, np.linalg.LinAlgError) as error:
            self._unknown(diagnostic, type(error).__name__ + ':' + str(error))
            report['reason'] = diagnostic['reason']
        return report
