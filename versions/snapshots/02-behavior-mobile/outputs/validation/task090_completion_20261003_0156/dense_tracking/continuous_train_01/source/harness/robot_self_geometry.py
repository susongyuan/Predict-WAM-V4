"""Pinned static USD collision boxes placed by current state61 / checked URDF FK.

Diagnostic robot-only exclusion, NOT a grasp/identity/contact certificate. No pxr,
IK, simulator, object pose, held-object geometry, or measured-EEF pose substitution.
An invalid source/state raises ValueError; the injected v4 caller returns unknown.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from harness import r1pro_kinematics as fk

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'outputs/validation/task090_completion_20261003_0156/self_geometry_usd'
GEOMETRY_SHA256 = '29fe253b44da3d48c95d63bc3e5046c921f7ebb51a88a361fd1bc2f46b565b69'
VALIDATION_SHA256 = 'a90e9f08cea6762f95a4f5cd10495f0b671714025ff93a1a556e57adbe59e7ee'
USD_SHA256 = '6029617cdd3aefce981428058a1c82cffe6af20ae61dc5be1da7334342c3bc52'
SENSOR_MARGIN_M = .01
# Schema/consistency checks, fixed before the single TRAIN run, not fitted gates.
Q_ROUNDOFF = 1e-6
EEF_POSITION_CHECK_M = 1e-4
EEF_ROTATION_CHECK_RAD = 1e-4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_json(path, expected):
    try:
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('self_geometry_source_sha256_mismatch:' + str(path))
        return json.loads(data)
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('self_geometry_source_unavailable') from error


def immutable(value):
    value = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)


class RobotSelfGeometry:
    """Callable(points_base_m[N,3], current_state61) -> bool[N] KEEP mask.

    Twelve distal links only: arm6/7, gripper, fingers and wrist camera on each
    side. Each source collision AABB is expanded by 1cm on every local axis:
    conservative also at corners (a superset of a Euclidean 1cm skin). Boundary
    points are rejected. Merely being within 0.2m of an EEF is not exclusion.
    Callers own time/episode pairing; this cannot certify hardware freshness.
    """
    version = 'usd_robot_aabb_state61_v1'

    def __init__(self, geometry_path=EXPORT/'geometry.json', validation_path=EXPORT/'validation.json'):
        geometry = checked_json(geometry_path, GEOMETRY_SHA256)
        validation = checked_json(validation_path, VALIDATION_SHA256)
        if (validation['status'] != 'PASS_STATIC_USD_COLLISION_EXPORT' or
                validation['geometry_sha256'] != GEOMETRY_SHA256 or
                geometry['source_usd_sha256'] != USD_SHA256):
            raise ValueError('self_geometry_unvalidated_source')
        # Verify the actual static source and all inputs used for runtime FK once
        # at construction, not once per point or frame. No USD parsing is needed.
        self.source_sha256 = {str(Path(geometry_path)): GEOMETRY_SHA256,
                              str(Path(validation_path)): VALIDATION_SHA256}
        paths = [fk.URDF, fk.OFFSETS, Path(fk.__file__),
                 ROOT/'data/omnigibson/omnigibson-robot-assets/models/r1pro/r1pro.yaml',
                 ROOT/'data/omnigibson/omnigibson-robot-assets/models/r1pro/usd/r1pro.usda']
        for path in paths:
            expected = validation['sources'][str(path)]['sha256']
            try:
                actual = sha256(path)
            except OSError as error:
                raise ValueError('self_geometry_source_unavailable:' + str(path)) from error
            if actual != expected:
                raise ValueError('self_geometry_source_sha256_mismatch:' + str(path))
            self.source_sha256[str(path)] = actual
        self.kinematics = fk.R1ProKinematics()
        self.boxes = {link: (immutable([box['min_m'] for box in boxes]),
                             immutable([box['max_m'] for box in boxes]))
                      for link, boxes in geometry['links'].items()}
        self._bindings = {name: joint['qpos'] for name, joint in geometry['joints'].items()
                          if joint['qpos'] is not None}
        self._limits = {name: (min(joint['limits_si'][0], self.kinematics.joints[name]['lim'][0]),
                              max(joint['limits_si'][1], self.kinematics.joints[name]['lim'][1]))
                        for name, joint in geometry['joints'].items() if name in self._bindings}
        # Only the 28 checked URDF joints, not unrelated base/head joints.
        pending = {name: self.kinematics.joints[name] for name in geometry['joints']
                   if name in self.kinematics.joints}
        self._ordered = []
        known = {'base_link'}
        while pending:
            ready = [name for name, joint in pending.items() if joint['parent'] in known]
            if not ready:
                raise ValueError('self_geometry_disconnected_kinematics')
            for name in ready:
                joint = pending.pop(name)
                self._ordered.append((name, joint))
                known.add(joint['child'])
        if not self.boxes.keys() <= known:
            raise ValueError('self_geometry_unplaced_link')

    def link_poses(self, state):
        """All modeled link poses in articulation-root metres, no world transform."""
        state = np.asarray(state, float)
        if state.shape != (61,) or not np.isfinite(state).all():
            raise ValueError('self_geometry_invalid_state61')
        for name, binding in self._bindings.items():
            value = state[binding['state61_index']]
            lo, hi = self._limits[name]
            if not lo - Q_ROUNDOFF <= value <= hi + Q_ROUNDOFF:
                raise ValueError('self_geometry_joint_outside_si_limits:' + name)
        poses = {'base_link': fk.R1ProKinematics._T(np.eye(3), self.kinematics.root_offset)}
        for name, joint in self._ordered:
            motion = np.eye(4)
            if joint['type'] != 'fixed':
                value = state[self._bindings[name]['state61_index']]
                if joint['type'] == 'prismatic':
                    motion[:3, 3] = joint['axis'] * value
                elif joint['type'] == 'revolute':
                    motion[:3, :3] = fk.axis_angle(joint['axis'], value)
                else:
                    raise ValueError('self_geometry_unsupported_joint')
            poses[joint['child']] = poses[joint['parent']] @ joint['T'] @ motion
        # EEF is a schema/FK cross-check only, never the object's position. Use
        # the unchanged calibrated -6cm / Y-pi offset exactly once, no refit.
        for arm, (position, quaternion) in fk.EEF.items():
            q = state[quaternion]
            norm = np.linalg.norm(q)
            if abs(norm - 1.) > .001:
                raise ValueError('self_geometry_invalid_eef_xyzw')
            gripper = poses[arm + '_gripper_link']
            predicted = gripper @ fk.R1ProKinematics._T(
                self.kinematics.eef_rot[arm], self.kinematics.eef_offset[arm])
            measured_rotation = fk.quat_to_mat(q / norm)
            difference = predicted[:3, :3] @ measured_rotation.T
            angle = np.arctan2(np.linalg.norm([difference[2, 1]-difference[1, 2],
                difference[0, 2]-difference[2, 0], difference[1, 0]-difference[0, 1]]) / 2,
                (np.trace(difference)-1) / 2)
            if (np.linalg.norm(predicted[:3, 3] - state[position]) > EEF_POSITION_CHECK_M or
                    abs(angle) > EEF_ROTATION_CHECK_RAD):
                raise ValueError('self_geometry_eef_fk_frame_mismatch:' + arm)
        return {name: poses[name] for name in self.boxes}

    def __call__(self, points, state):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('self_geometry_invalid_points_m')
        poses = self.link_poses(state)  # Even empty input must not bypass validation.
        rejected = np.zeros(len(points), bool)
        for name, (lo, hi) in self.boxes.items():
            pose = poses[name]
            # Bounded blocks for the dense initialization surface.
            for start in range(0, len(points), 4096):
                local = (points[start:start+4096] - pose[:3, 3]) @ pose[:3, :3]
                inside = ((local[:, None] >= lo - SENSOR_MARGIN_M - 1e-12) &
                          (local[:, None] <= hi + SENSOR_MARGIN_M + 1e-12)).all(axis=2).any(axis=1)
                rejected[start:start+4096] |= inside
        return ~rejected
