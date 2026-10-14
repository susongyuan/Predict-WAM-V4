"""Unvalidated task1 v2 bottle candidate; v1 lids/carton/peel/placement unchanged.

Evidence: v1 944025/945024 stopped BEFORE closing (19.52/20.07 mm TCP
error). PhysX reported panda_hand--bottle, no finger contact. The scaled
nozzle projects 40.48/40.59 mm above the old TCP, but the cached hand hull
has only 37.438 mm TCP standoff. Smaller 944026/947000 passed. This version
pinches the rigid cap, not the nozzle, with geometry-sized aperture and
measured bilateral contact. No score, drive, collision guard or scene edits.
CPU geometry/IK is NOT physical validation; rerun all four physical probes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from harness import isaac_zhuhai_expert as original
from harness.isaac_zhuhai_task1_expert_repair_v1 import SmoothExpert as V1Expert
from harness.zhuhai_robot_spec import GRIPPER_FINGER_TRAVEL, TOOL_FROM_HAND

# Read-only measurement of cached payloads/geometries.usd:/Geometries/hand_c.
# Finger pad dimensions/offsets come from payloads/base.usda. No USD/Isaac
# loading is needed here. The probe binds these assets as well as the code.
PALM_FRONT_HAND_Z_M = 0.06596215069293976
PALM_STANDOFF_M = float(TOOL_FROM_HAND[2][3]) - PALM_FRONT_HAND_Z_M
PAD_HALF_HEIGHT_M = 0.00825
PAD_CENTER_FORWARD_M = 0.0584 + 0.04525 - float(TOOL_FROM_HAND[2][3])
PAD_INNER_OFFSET_M = 0.00758 - 0.00760
MAX_GAP_M = 2 * (GRIPPER_FINGER_TRAVEL + PAD_INNER_OFFSET_M)
APPROACH_ERROR_M = 0.006  # SAME v1 measured-approach gate, not relaxed.
PALM_CLEARANCE_M = 0.008
MIN_CAP_OVERLAP_M = 0.011
APERTURE_RESERVE_M = 0.016
MAX_RETRIES = 3
CAP_FRACTIONS = (0.62, 0.74, 0.50)
YAWS_DEG = (0., 20., -20., 90., -90., 180.)


@dataclass(frozen=True)
class BottleCandidate:
    candidate_id: str
    pose: np.ndarray
    open_fraction: float
    cap_span_m: float
    palm_clearance_m: float
    pad_overlap_m: float


def gap_from_fraction(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1.02:
        raise ValueError('Invalid measured finger opening')
    return 2 * (GRIPPER_FINGER_TRAVEL * value + PAD_INNER_OFFSET_M)


def fraction_for_gap(gap):
    if not math.isfinite(gap) or not 0 < gap <= MAX_GAP_M:
        raise ValueError('Requested aperture exceeds the unchanged Panda travel')
    return (gap / 2 - PAD_INNER_OFFSET_M) / GRIPPER_FINGER_TRAVEL


def _part_interval(part, body, axis):
    """Conservative support interval; cylinders do not grow spuriously at yaw45."""
    dims = np.asarray(part['dimensions_m'], dtype=float)
    center = body[:3, :3] @ np.asarray(part['position_m']) + body[:3, 3]
    local_axis = body[:3, :3].T @ np.asarray(axis)
    if part['kind'] in ('cylinder', 'cone'):
        radius = math.hypot(dims[0] * local_axis[0], dims[1] * local_axis[1]) / 2
        radius += abs(dims[2] * local_axis[2]) / 2
    else:
        yaw = float(part.get('yaw_rad', 0.))
        local_axis = Rotation.from_euler('z', -yaw).apply(local_axis)
        radius = float(np.abs(local_axis) @ dims / 2)
    midpoint = float(center @ axis)
    return midpoint - radius, midpoint + radius


def palm_clearance(spec, body, tool):
    # Separating plane of the hand hull in actual tool coordinates. A positive
    # margin screens the whole bottle; it does not certify all robot collisions.
    axis = tool[:3, 0]
    near = min(_part_interval(p, body, axis)[0] for p in spec['parts'])
    return PALM_STANDOFF_M + near - float(tool[:3, 3] @ axis)


def bottle_candidates(spec, body, table_surface):
    """Recompute cap pinch poses from measured body geometry; never edit spec."""
    body = np.asarray(body, dtype=float)
    if (body.shape != (4, 4) or not np.isfinite(body).all()
            or not np.allclose(body[3], [0, 0, 0, 1])
            or not np.allclose(body[:3, :3].T @ body[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(body[:3, :3]), 1., atol=1e-5)):
        raise ValueError('Bottle needs a finite measured rigid transform')
    if body[2, 2] < math.cos(math.radians(12)):
        raise RuntimeError('Bottle tipped beyond upright cap-pinch envelope')
    parts = {p['name']: p for p in spec['parts']}
    cap, barrel = parts['cap'], parts['bottle_body']
    for part in spec['parts']:
        dims = np.asarray(part['dimensions_m'], dtype=float)
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
            raise ValueError('Invalid bottle dimensions')
    if cap['kind'] != 'cylinder' or barrel['kind'] != 'cylinder':
        raise ValueError('Only the unchanged rigid capped-bottle proxy is supported')
    cap_low, cap_high = _part_interval(cap, body, np.array([0., 0., 1.]))
    barrel_high = _part_interval(barrel, body, np.array([0., 0., 1.]))[1]
    result = []
    for tier, fraction in enumerate(CAP_FRACTIONS):
        local = np.array(cap['position_m'], dtype=float)
        local[2] += (fraction - .5) * cap['dimensions_m'][2]
        point = body[:3, :3] @ local + body[:3, 3]
        point[2] += PAD_CENTER_FORWARD_M
        pad_low = point[2] - PAD_CENTER_FORWARD_M - PAD_HALF_HEIGHT_M
        pad_high = pad_low + 2 * PAD_HALF_HEIGHT_M
        overlap = min(cap_high, pad_high) - max(cap_low, pad_low)
        # Keep the pads above the wide shoulder; do not pinch a cone/nozzle.
        if (overlap < MIN_CAP_OVERLAP_M or pad_low < barrel_high + .0005
                or pad_low < table_surface + .02):
            continue
        for yaw_index, yaw in enumerate(YAWS_DEG):
            angle = math.radians(yaw)
            jaw = np.array([math.cos(angle), math.sin(angle), 0.])
            pose = np.eye(4)
            pose[:3, :3] = original.top_down_rotation(jaw)
            pose[:3, 3] = point
            clearance = palm_clearance(spec, body, pose)
            if clearance < PALM_CLEARANCE_M:
                continue
            # Open for the full swept bottle envelope, not merely the cap.
            mid = float(point @ jaw)
            radius = max(max(abs(x - mid) for x in _part_interval(p, body, jaw))
                         for p in spec['parts'])
            gap = 2 * radius + APERTURE_RESERVE_M
            if gap > MAX_GAP_M:
                continue  # No clamping of an infeasible aperture.
            low, high = _part_interval(cap, body, jaw)
            result.append(BottleCandidate(f'cap{tier}:yaw{yaw_index}', pose,
                fraction_for_gap(gap), high - low, clearance, overlap))
    if not result:
        raise RuntimeError('No cap pose with hand/pad/aperture clearance')
    return result


def contact_feedback(report, body_path, arm):
    fingers, other = set(), set()
    for pair in report['contacts']:
        a, b = pair['body0_path'], pair['body1_path']
        robot = b if a == body_path else a if b == body_path else ''
        if not robot.startswith(f'/World/Robot/{arm}/'):
            continue
        leaf = robot.rsplit('/', 1)[-1]
        (fingers if leaf in ('panda_leftfinger', 'panda_rightfinger') else other).add(leaf)
    return dict(bilateral=fingers == {'panda_leftfinger', 'panda_rightfinger'},
                fingers=sorted(fingers), nonfinger=sorted(other))


class SmoothExpert(V1Expert):
    """Only bottle acquisition differs. All actuation/guards stay inherited."""

    def _cap_choice(self, spec, arm, tried):
        body = self._body_matrix('squeeze_bottle')
        for candidate in bottle_candidates(spec, body, self.manifest['table_surface_world_m']):
            if candidate.candidate_id in tried:
                continue
            pre = candidate.pose.copy(); pre[2, 3] += .10
            reach = self.fk_model.ik(arm, pre, seed=self.current[self._arm_slice(arm)])
            valid = bool(reach.success)
            seed = reach.joints
            # Verify descent AND lift before touching. Detours/branch jumps are
            # forbidden near contact, using the original IK tolerances/checker.
            for height in (.075, .050, .025, 0., .035, .080, .140, .200):
                if not valid:
                    break
                target = candidate.pose.copy(); target[2, 3] += height
                solved = self.fk_model.ik(arm, target, seed=seed)
                valid = (solved.success and not getattr(solved, 'joint_waypoints', None)
                         and np.max(np.abs(solved.joints - seed)) <= .5)
                seed = solved.joints
            if valid:
                return candidate
            tried.add(candidate.candidate_id)
            self.trace.append(dict(phase='squeeze_bottle:v2_strict_path_rejected',
                                   candidate_id=candidate.candidate_id))
        raise RuntimeError('squeeze_bottle: no remaining strict cap descent/lift path')

    def _cap_line(self, arm, target, phase, seconds=2.5):
        start = self._actual_tool_pose(arm)
        poses = []
        for t in np.linspace(0., 1., 9):
            pose = target.copy()
            pose[:3, 3] = (1 - t) * start[:3, 3] + t * target[:3, 3]
            poses.append(pose)
        self.move_pose_path(arm, poses, f'squeeze_bottle:{phase}', seconds)

    def _cap_feedback(self, spec, arm, candidate):
        report = self.env.rule_state()
        feedback = contact_feedback(report, spec['body_path'], arm)
        gap = gap_from_fraction(self.env.actual_state()[self._gripper_index(arm)])
        feedback.update(gap_m=gap, expected_cap_span_m=candidate.cap_span_m,
            touching_gripper=report['objects']['squeeze_bottle']['touching_gripper'])
        feedback['pinched'] = (feedback['bilateral'] and not feedback['nonfinger']
            and feedback['touching_gripper']
            and candidate.cap_span_m * .75 <= gap <= candidate.cap_span_m + .006)
        return feedback

    def _cap_retry_exit(self, spec, arm, release_pose):
        actual = self._actual_tool_pose(arm)
        contacts = contact_feedback(self.env.rule_state(), spec['body_path'], arm)
        if contacts['fingers'] and actual[2, 3] > release_pose[2, 3] + .005:
            lower = actual.copy(); lower[2, 3] = release_pose[2, 3]
            self._cap_line(arm, lower, 'v2_retry_lower_before_release', 3.)
        self.gripper(arm, 1., 'squeeze_bottle:v2_retry_release')
        exit_pose = self._actual_tool_pose(arm).copy()
        exit_pose[2, 3] = max(exit_pose[2, 3], release_pose[2, 3] + .10)
        self._cap_line(arm, exit_pose, 'v2_retry_vertical_exit')
        self.hold(.6, 'squeeze_bottle:v2_retry_object_settle')

    def _acquire_garbage(self, key, spec, arm):
        if key != 'squeeze_bottle':
            return super()._acquire_garbage(key, spec, arm)
        if self.dof != 16:
            raise RuntimeError('v2 cap acquisition requires measured native16')
        tried, reason = set(), 'no candidate'
        for retry in range(MAX_RETRIES):
            candidate = self._cap_choice(spec, arm, tried)
            tried.add(candidate.candidate_id)
            self.trace.append(dict(phase=f'{key}:v2_cap_candidate', retry=retry,
                candidate_id=candidate.candidate_id, palm_clearance_m=candidate.palm_clearance_m,
                pad_overlap_m=candidate.pad_overlap_m, cap_span_m=candidate.cap_span_m,
                open_fraction=candidate.open_fraction, target_tool_world_matrix=candidate.pose.tolist()))
            self.gripper(arm, 1., f'{key}:v2_open_for_transit')
            pre = candidate.pose.copy(); pre[2, 3] += .10
            self.move_pose(arm, pre, f'{key}:v2_cap_pregrasp')
            self.gripper(arm, candidate.open_fraction, f'{key}:v2_geometry_preshape')
            self.hold(.2, f'{key}:v2_preshape_settle')
            measured_gap = gap_from_fraction(self.env.actual_state()[self._gripper_index(arm)])
            required_gap = gap_from_fraction(candidate.open_fraction)
            self.trace.append(dict(phase=f'{key}:v2_preshape_feedback',
                                   measured_gap_m=measured_gap, required_gap_m=required_gap))
            if measured_gap < required_gap - .002:
                reason = 'measured aperture did not open BEFORE descent'
                self._cap_retry_exit(spec, arm, candidate.pose)
                continue
            self._cap_line(arm, candidate.pose, 'v2_cap_approach')
            self.hold(.5, f'{key}:v2_approach_settle')
            actual, body = self._actual_tool_pose(arm), self._body_matrix(key)
            refreshed = next((c for c in bottle_candidates(spec, body,
                self.manifest['table_surface_world_m']) if c.candidate_id == candidate.candidate_id), None)
            tracking_error = float(np.linalg.norm(actual[:3, 3] - candidate.pose[:3, 3]))
            error = (float(np.linalg.norm(actual[:3, 3] - refreshed.pose[:3, 3]))
                     if refreshed is not None else None)
            angle = float(Rotation.from_matrix(candidate.pose[:3, :3] @ actual[:3, :3].T).magnitude())
            clearance = palm_clearance(spec, body, actual)
            contacts = contact_feedback(self.env.rule_state(), spec['body_path'], arm)
            self.trace.append(dict(phase=f'{key}:v2_measured_approach', candidate_id=candidate.candidate_id,
                position_error_m=tracking_error, live_cap_error_m=error, orientation_error_rad=angle,
                palm_clearance_m=clearance, contacts=contacts))
            if (error is None or error > APPROACH_ERROR_M
                    or tracking_error > APPROACH_ERROR_M or angle > .035
                    or clearance < PALM_CLEARANCE_M or contacts['nonfinger']):
                reason = 'measured approach/hand clearance rejected BEFORE closing'
                self._cap_retry_exit(spec, arm, candidate.pose)
                continue
            candidate = refreshed
            # A slow two-stage close reduces impulse; the final target remains
            # fully closed on the original force-limited Panda finger drives.
            self.gripper(arm, fraction_for_gap(candidate.cap_span_m + .004), f'{key}:v2_cap_near_close')
            self.gripper(arm, 0., f'{key}:v2_cap_close')
            self.hold(.4, f'{key}:v2_close_settle')
            feedback = self._cap_feedback(spec, arm, candidate)
            self.trace.append(dict(phase=f'{key}:v2_close_feedback', **feedback))
            release_pose = self._actual_tool_pose(arm).copy()
            if not feedback['pinched']:
                reason = 'missing bilateral cap contact or incorrect measured aperture'
                self._cap_retry_exit(spec, arm, release_pose)
                continue
            before = self._body_matrix(key).copy()
            lift = release_pose.copy(); lift[2, 3] += .035
            self._cap_line(arm, lift, 'v2_low_test_lift', 2.)
            self.hold(.4, f'{key}:v2_low_lift_settle')
            feedback = self._cap_feedback(spec, arm, candidate)
            body_delta = self._body_matrix(key)[:3, 3] - before[:3, 3]
            tool_delta = self._actual_tool_pose(arm)[:3, 3] - release_pose[:3, 3]
            drift = float(np.linalg.norm(body_delta - tool_delta))
            self.trace.append(dict(phase=f'{key}:v2_low_lift_feedback', body_lift_m=float(body_delta[2]),
                                   relative_drift_m=drift, **feedback))
            if not feedback['pinched'] or body_delta[2] < .020 or drift > .006:
                reason = 'cap failed measured low-lift follow/contact check'
                self._cap_retry_exit(spec, arm, release_pose)
                continue
            lift = release_pose.copy(); lift[2, 3] += .20
            self._cap_line(arm, lift, 'v2_full_lift', 3.)
            self.hold(5.2, f'{key}:physical_continuous_hold')
            item = self.env.rule_state()['score']['items'][key]
            feedback = self._cap_feedback(spec, arm, candidate)
            self.trace.append(dict(phase=f'{key}:v2_hold_feedback', score=dict(item), **feedback))
            if item['grasp_awarded'] and feedback['pinched']:
                return
            reason = 'unchanged physical5s award or retained bilateral contact missing'
            self._cap_retry_exit(spec, arm, release_pose)
        raise RuntimeError(f'{key}: v2 cap candidates exhausted ({reason}); no guard/score reset')
