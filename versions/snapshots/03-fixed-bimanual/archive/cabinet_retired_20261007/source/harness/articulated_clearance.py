"""Conditional single-hinge clearance against ONE externally observed oriented plane.

Column-vector SE(3), metres/radians, positive dimensionless uniform scale s.
Root parts: x=t+R*s*v. Child parts: x=t+R*s*(o+Rot(axis,q)*(C*v-o));
C is root_from_child_zero (including translation), axis/o are root coordinates.
Plane clearance is n.(x-a), with unit normal pointing toward the free halfspace.
All vertices of each complete part, or a conservative enclosing convex hull, are
required. Completeness/provenance are caller assertions, not authenticated here.
No static pose, scene ground, residual-to-error conversion or runtime I/O exists.
"""
from collections.abc import Mapping
import math
import numpy as np


class _Unknown(Exception):
    pass


_LIMITS = (
    "Conditional geometry only; source/completeness assertions are not verified.",
    "One oriented infinite plane only; other supports, grasp and success are unknown.",
    "Joint interval and scale interval must include all applicable uncertainty.",
    "geometry_m bounds world-space enclosure/kinematic error uniformly over q and s.",
    "Float64 with outward numerical allowance; not formal interval arithmetic.",
)


def _need(record, key):
    if not isinstance(record, Mapping):
        raise ValueError("Expected a mapping")
    if key not in record or record[key] is None:
        raise _Unknown("Missing " + key)
    return record[key]


def _source(record, key="source"):
    value = _need(record, key)
    if not isinstance(value, str) or not value.strip():
        raise _Unknown("Missing provenance: " + key)
    return value


def _array(value, shape=None):
    a = np.asarray(value, dtype=float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError("Invalid shape or nonfinite numeric input")
    if np.any(np.abs(a) > 1e6):
        raise ValueError("Outside supported numerical domain: absolute input > 1e6")
    return a


def _se3(value):
    t = _array(value, (4, 4))
    r = t[:3, :3]
    if (not np.array_equal(t[3], [0, 0, 0, 1])
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0)
            or abs(np.linalg.det(r) - 1) > 1e-12):
        raise ValueError("Transform must be column-vector SE(3), without scale")
    return t


def _interval(value, positive=False):
    lo, hi = _array(value, (2,))
    if lo > hi or (positive and lo <= 0):
        raise ValueError("Invalid closed interval")
    return float(lo), float(hi)


def _trig_min(a, b, c, lo, hi):
    """Analytic endpoints and minimum stationary phase; never angle sampling."""
    radius = np.hypot(b, c)
    if hi - lo >= math.tau:
        return a - radius
    endpoint = np.minimum(a + b * math.cos(lo) + c * math.sin(lo),
                          a + b * math.cos(hi) + c * math.sin(hi))
    phase = np.arctan2(c, b) + math.pi
    stationary = phase + math.tau * np.ceil((lo - phase) / math.tau)
    # Other stationary phases are maxima and cannot reduce the minimum.
    return np.where((stationary >= lo) & (stationary <= hi),
                    np.minimum(endpoint, a - radius), endpoint)


def conservative_clearance(*, root_pose, parts, model, joints, scale, plane,
                           errors, units):
    """Return a geometric candidate/bound, or Unknown; malformed data raises ValueError.

    Required mappings (all source strings must identify external evidence):
      root_pose: transform, kind ('measurement' or 'synthetic_test'), source.
      plane: normal_world, point_world_m, kind as above, source.
      model: complete=True, part_ids (full inventory), joint_count, source.
      parts: list of {id, frame ('root'/'child'), vertices, complete=True,
                     representation ('vertices'/'convex_hull'), source}.
      joints: [] or [{axis_root, origin_root_m, root_from_child_zero,
                     angle_interval_rad, includes_error=True, source}].
      scale: interval, includes_error=True, source. Entire possible actual scale.
      errors: translation_m, rotation_rad, plane_point_m, plane_normal_rad,
              geometry_m, source. All five nonnegative, explicitly supplied.
      units: {length:'m', angle:'rad', scale:'dimensionless'}.

    Translation/plane-point errors are Euclidean balls. Rotation/plane-normal
    errors are angular balls. geometry_m is a world-space enclosure inflation
    including shape, joint-frame and cooking errors not otherwise covered.
    No independence assumption is needed. The error envelope subtracts
      et + ea + eg + 2*smax*rho*sin(er/2)
      + 2*(||t-a||+smax*rho)*sin(en/2), angles capped at pi.
    rho bounds every part's distance from the root throughout the angle range.
    Missing measurements cannot be replaced with authored/default asset poses.
    """
    result = dict(status="Unknown", lower_bound_m=None,
                  positive_clearance_candidate=None, verified=False,
                  admitted=False, physical_label=None, limitations=list(_LIMITS))
    try:
        if any(x is None for x in (root_pose, parts, model, joints, scale,
                                   plane, errors, units)):
            raise _Unknown("Missing required measurement/model/uncertainty input")
        for key, expected in (("length", "m"), ("angle", "rad"),
                              ("scale", "dimensionless")):
            if _need(units, key) != expected:
                raise ValueError("Wrong units: " + key)
        provenance = {k: _source(v) for k, v in
                      (("root", root_pose), ("plane", plane), ("model", model),
                       ("scale", scale), ("errors", errors))}
        kinds = [_need(v, "kind") for v in (root_pose, plane)]
        if any(k not in ("measurement", "synthetic_test") for k in kinds):
            raise _Unknown("Root pose and support plane must be externally observed")
        if _need(model, "complete") is not True:
            raise _Unknown("Partial model")
        ids = _need(model, "part_ids")
        if not isinstance(ids, (list, tuple)) or not ids or any(
                not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid complete part inventory")
        if not isinstance(joints, (list, tuple)) or not isinstance(parts, (list, tuple)):
            raise ValueError("Parts and joints must be sequences")
        count = _need(model, "joint_count")
        if type(count) is not int or count < 0:
            raise ValueError("Invalid joint count")
        if len(joints) > 1 or count > 1:
            raise _Unknown("Multiple joints are outside supported scope")
        if count != len(joints):
            raise _Unknown("Missing joint model")
        actual_ids = [_need(p, "id") for p in parts]
        if len(actual_ids) != len(ids) or set(actual_ids) != set(ids):
            raise _Unknown("Missing, duplicate or unexpected part")
        pose = _se3(_need(root_pose, "transform"))
        n = _array(_need(plane, "normal_world"), (3,))
        if abs(np.linalg.norm(n) - 1) > 1e-12:
            raise ValueError("Plane normal must be unit length with explicit orientation")
        n = n / np.linalg.norm(n)
        anchor = _array(_need(plane, "point_world_m"), (3,))
        slo, shi = _interval(_need(scale, "interval"), positive=True)
        if _need(scale, "includes_error") is not True:
            raise _Unknown("Scale uncertainty not enclosed")
        keys = ("translation_m", "rotation_rad", "plane_point_m",
                "plane_normal_rad", "geometry_m")
        e = {k: float(_array(_need(errors, k), ())) for k in keys}
        if min(e.values()) < 0:
            raise ValueError("Error bounds must be nonnegative")
        direction = pose[:3, :3].T @ n
        offset = pose[:3, 3] - anchor
        lo = hi = 0.0  # No joint: there is no unobserved angle to default.
        if joints:
            joint = joints[0]
            provenance["joint"] = _source(joint)
            if _need(joint, "includes_error") is not True:
                raise _Unknown("Joint interval does not enclose uncertainty")
            axis = _array(_need(joint, "axis_root"), (3,))
            length = np.linalg.norm(axis)
            if length < 1e-12:
                raise ValueError("Zero or numerically unusable hinge axis")
            axis = axis / length
            origin = _array(_need(joint, "origin_root_m"), (3,))
            child = _se3(_need(joint, "root_from_child_zero"))
            lo, hi = _interval(_need(joint, "angle_interval_rad"))
        minima, rho, condition, frames = {}, 0.0, 0.0, set()
        provenance["parts"] = {}
        for part in parts:
            pid = _need(part, "id")
            provenance["parts"][pid] = _source(part)
            if _need(part, "complete") is not True:
                raise _Unknown("Incomplete geometry for " + pid)
            if _need(part, "representation") not in ("vertices", "convex_hull"):
                raise ValueError("Unsupported geometry representation")
            v = _array(_need(part, "vertices"))
            if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
                raise ValueError("Expected nonempty N by 3 complete vertices")
            frame = _need(part, "frame")
            frames.add(frame)
            vnorm = float(np.linalg.norm(v, axis=1).max())
            if frame == "root":
                f, radius = float((v @ direction).min()), vnorm
            elif frame == "child" and joints:
                p = v @ child[:3, :3].T + child[:3, 3] - origin
                parallel = np.outer(p @ axis, axis)
                a = (origin + parallel) @ direction
                b = (p - parallel) @ direction
                c = np.cross(axis, p) @ direction
                f = float(_trig_min(a, b, c, lo, hi).min())
                radius = float(np.linalg.norm(origin) + np.linalg.norm(p, axis=1).max())
                condition = max(condition, vnorm + np.linalg.norm(child[:3, 3])
                                + 2 * np.linalg.norm(origin))
            else:
                raise ValueError("Frame must be root, or child with one hinge")
            rho = max(rho, radius)
            condition = max(condition, vnorm, radius)
            minima[pid] = float(n @ offset + min(slo * f, shi * f))
        if frames != ({"root", "child"} if joints else {"root"}):
            raise _Unknown("Missing root or articulated geometry")
        chord = lambda angle: 2 * math.sin(min(angle, math.pi) / 2)
        envelope = dict(translation_m=e["translation_m"], plane_point_m=e["plane_point_m"],
                        geometry_m=e["geometry_m"],
                        rotation_m=shi * rho * chord(e["rotation_rad"]),
                        plane_normal_m=(np.linalg.norm(offset) + shi * rho)
                        * chord(e["plane_normal_rad"]))
        # Includes transform tolerance, cancellation, phase reduction and libm rounding.
        envelope["numerical_m"] = float(1e-10 * (1 + np.linalg.norm(pose[:3, 3])
            + np.linalg.norm(anchor) + shi * condition) * (1 + max(abs(lo), abs(hi))))
        lower = float(np.nextafter(min(minima.values()) - sum(envelope.values()), -np.inf))
        result.update(status="geometry_only", lower_bound_m=lower,
                      positive_clearance_candidate=lower > 0, nominal_min_m=min(minima.values()),
                      per_part_nominal_min_m=minima, root_radius_m=rho,
                      error_envelope_m=envelope, sources=provenance,
                      synthetic_inputs="synthetic_test" in kinds)
    except _Unknown as exc:
        result["reason"] = str(exc)
    return result
