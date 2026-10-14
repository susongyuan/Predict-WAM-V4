#!/usr/bin/env python3
"""Offline CPU-only robot I/O declaration preflight (stdlib; no device access).

Usage: python3 -B scripts/check_robot_io_contract.py contract.json \
  --clock-domain unix_epoch --now 1790970000 --model-profile model.json
Exit 0: schema/readiness declarations pass; 2: blocked; 3: bad CLI/input.
Model compatibility is separate and never changes hardware admission (always false).
All timestamps must use the explicit --clock-domain; --now defaults to time.time()
only for unix_epoch. A custom/monotonic clock requires --now and a boot/clock ID.
This checks supplied metadata, not sensors, calibration accuracy or physical safety.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

UNITS = {"angular_position": "rad", "angular_velocity": "rad/s",
         "linear_position": "m", "linear_velocity": "m/s",
         "quaternion": "1", "gripper_aperture": "m",
         "angular_effort": "N*m", "linear_effort": "N"}
MAX_BYTES = 2 * 1024 * 1024


def number(x):
    if type(x) not in (int, float):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:
        return False


def text(x):
    return isinstance(x, str) and bool(x.strip()) and not x.strip().upper().startswith("TODO")


def vector(x, n):
    return isinstance(x, list) and len(x) == n and all(number(v) for v in x)


def interval(x):
    return vector(x, 2) and x[0] < x[1]


def matrix(x, n):
    return isinstance(x, list) and len(x) == n and all(vector(row, n) for row in x)


def rigid(x):
    if not matrix(x, 4) or any(abs(a-b) > 1e-6 for a, b in zip(x[3], [0, 0, 0, 1])):
        return False
    r = [row[:3] for row in x[:3]]
    if any(abs(sum(r[k][i]*r[k][j] for k in range(3)) - (i == j)) > 1e-5
           for i in range(3) for j in range(3)):
        return False
    det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
           - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
           + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    return abs(det-1) <= 1e-5


class Check:
    def __init__(self):
        self.issues = []

    def need(self, ok, path, message, kind="schema"):
        if not ok:
            self.issues.append({"kind": kind, "path": path, "reason": message})
        return bool(ok)

    def obj(self, x, path):
        self.need(isinstance(x, dict), path, "required object; null/TODO is blocked")
        return x if isinstance(x, dict) else {}

    def rows(self, x, path, maximum=512):
        ok = isinstance(x, list) and 0 < len(x) <= maximum
        self.need(ok, path, "required nonempty bounded list")
        return x if ok else []

    def strings(self, x, path):
        rows = self.rows(x, path)
        self.need(all(text(v) for v in rows), path, "explicit non-TODO names required")
        self.need(len({v for v in rows if isinstance(v, str)}) == len(rows), path, "names must be unique")
        return rows

    def positive(self, x, path):
        return self.need(number(x) and x > 0, path, "finite positive value required")


def compare_model(contract, profile):
    """Exact boundary matching; equal sizes never imply equal layouts or hardware."""
    if profile is None:
        return {"status": "NOT_CHECKED", "reason": "no model profile supplied", "hardware_validated": False}
    issues = []
    if not isinstance(profile, dict) or profile.get("profile_version") != 1:
        return {"status": "BLOCKED", "issues": ["invalid model profile"], "hardware_validated": False}
    # These are compared literally: no implicit reordering, unit conversion or camera aliasing.
    for key in ("frames", "action", "proprio", "cameras", "timing", "model_input"):
        expected = profile.get("expects", {}).get(key) if isinstance(profile.get("expects"), dict) else None
        actual = contract.get(key)
        if not isinstance(expected, dict) or not expected:
            issues.append("profile.expects." + key + " is missing")
            continue
        if not isinstance(actual, dict) and key != "cameras":
            issues.append(key + " is missing")
            continue
        if key == "cameras":
            if not isinstance(actual, list):
                issues.append("cameras is missing")
                continue
            actual = {"names": [v.get("name") if isinstance(v, dict) else None for v in actual],
                      "depth_enabled": [(v.get("depth") or {}).get("enabled") if isinstance(v, dict) and isinstance(v.get("depth"), dict) else None for v in actual]}
        for field, value in expected.items():
            if actual.get(field) != value:
                issues.append(key + "." + field + " differs (explicit adapter/recalibration required)")
    return {"status": "COMPATIBLE_DECLARED" if not issues else "INCOMPATIBLE_OR_UNKNOWN",
            "profile": profile.get("name"), "issues": issues, "hardware_validated": False,
            "scope": "exact declared tensor/semantic boundary only; no inference or task-performance test"}


def validate(contract, *, now, clock_domain, model_profile=None):
    c = Check()
    root = c.obj(contract, "$")
    c.need(type(root.get("schema_version")) is int and root.get("schema_version") == 1,
           "schema_version", "must be integer 1")
    robot = c.obj(root.get("robot"), "robot")
    c.need(text(robot.get("id")), "robot.id", "robot identity required")
    c.need(robot.get("evidence") == "declared", "robot.evidence", "this tool accepts declarations, never measured certification")
    frames = c.obj(root.get("frames"), "frames")
    base = frames.get("base")
    c.need(text(base), "frames.base", "named base frame required")
    c.need(frames.get("base_axes") == "x_forward_y_left_z_up", "frames.base_axes", "explicit right-handed canonical boundary required")
    c.need(frames.get("handedness") == "right", "frames.handedness", "right handed boundary required")
    c.need(frames.get("quaternion_order") in ("xyzw", "wxyz"), "frames.quaternion_order", "xyzw or wxyz required")
    c.need(frames.get("transform_convention") == "p_base=T_base_camera@p_camera",
           "frames.transform_convention", "column-vector camera-to-base convention required")
    timing = c.obj(root.get("timing"), "timing")
    c.need(number(now) and text(clock_domain), "check_clock", "finite reference time and explicit clock domain required")
    c.need(timing.get("clock_domain") == clock_domain, "timing.clock_domain", "reference time domain differs", "readiness")
    c.need(timing.get("unit") == "s", "timing.unit", "timestamps and ages must be seconds")
    for key in ("action_hz", "observation_hz", "max_age_s", "max_skew_s", "max_command_age_s"):
        c.positive(timing.get(key), "timing." + key)
    sensor_times = []

    def stamp(value, path, limit, sensor=True):
        v = c.obj(value, path)
        c.need(v.get("unit") == "s", path + ".unit", "timestamp must be seconds")
        c.need(v.get("clock_domain") == clock_domain, path + ".clock_domain", "time domain/boot identity mismatch", "readiness")
        t = v.get("value")
        if c.need(number(t), path + ".value", "finite timestamp required") and number(now) and number(limit):
            # A future timestamp is not silently treated as fresh.
            c.need(0 <= now-t <= limit, path, "timestamp is stale or in the future", "readiness")
            if sensor and v.get("clock_domain") == clock_domain:
                sensor_times.append(t)

    devices = {}
    for i, item in enumerate(c.rows(root.get("devices"), "devices")):
        p = "devices[" + str(i) + "]"
        row = c.obj(item, p)
        name = row.get("id")
        if c.need(text(name), p + ".id", "explicit device ID required"):
            c.need(name not in devices, p + ".id", "duplicate device")
            devices[name] = row
        c.need(row.get("kind") in ("camera", "controller", "estop"), p + ".kind", "camera/controller/estop required")
        c.need(row.get("available") is True, p + ".available", "device availability must be declared true", "readiness")
        c.need(row.get("evidence") == "declared", p + ".evidence", "availability is a declaration, not a hardware test")

    def device(value, kind, path):
        d = devices.get(value, {}) if isinstance(value, str) else {}
        c.need(d.get("kind") == kind, path, "missing matching declared device")

    joints = {}
    for i, item in enumerate(c.rows(root.get("joints"), "joints")):
        p = "joints[" + str(i) + "]"
        row = c.obj(item, p)
        name = row.get("name")
        if c.need(text(name), p + ".name", "explicit joint name required"):
            c.need(name not in joints, p + ".name", "duplicate joint")
            joints[name] = row
        kind = row.get("kind")
        kind = kind if isinstance(kind, str) else None
        c.need(kind in ("revolute", "prismatic"), p + ".kind", "revolute/prismatic required (bounded interface)")
        c.need(row.get("unit") == {"revolute": "rad", "prismatic": "m"}.get(kind) and kind in ("revolute", "prismatic"),
               p + ".unit", "joint limits must use radians/metres")
        c.need(interval(row.get("limits")), p + ".limits", "finite increasing joint limits required")
        c.positive(row.get("max_velocity"), p + ".max_velocity")
        c.need(row.get("velocity_unit") == {"revolute": "rad/s", "prismatic": "m/s"}.get(kind)
               and kind in ("revolute", "prismatic"), p + ".velocity_unit", "joint velocity unit mismatch")

    channels = {}
    for section in ("proprio", "action"):
        obj = c.obj(root.get(section), section)
        rows = c.rows(obj.get("layout"), section + ".layout")
        c.need(type(obj.get("dim")) is int and obj.get("dim") == len(rows), section + ".dim", "dimension must equal ordered layout length")
        stamp(obj.get("stamp"), section + ".stamp", timing.get("max_command_age_s" if section == "action" else "max_age_s"), section != "action")
        if section == "action":
            c.need(obj.get("position_command_mode") == "absolute", "action.position_command_mode",
                   "absolute position boundary required; delta controller needs an explicit adapter")
            device(obj.get("device"), "controller", "action.device")
            c.need(obj.get("queue_flushed_on_stop") is True, "action.queue_flushed_on_stop", "stop must flush queued commands", "readiness")
        names = set()
        channels[section] = {}
        for i, item in enumerate(rows):
            p = section + ".layout[" + str(i) + "]"
            row = c.obj(item, p)
            name, quantity, unit = row.get("name"), row.get("quantity"), row.get("unit")
            if c.need(text(name), p + ".name", "explicit dimension name required"):
                c.need(name not in names, p + ".name", "duplicate dimension name")
                names.add(name)
                channels[section][name] = row
            expected_unit = UNITS.get(quantity) if isinstance(quantity, str) else None
            c.need(expected_unit is not None, p + ".quantity", "unsupported/unknown physical quantity")
            c.need(row.get("frame") in (base, "joint") and text(row.get("frame")), p + ".frame", "named base frame or joint frame required")
            lowhigh = row.get("limits")
            if section == "action":
                c.need(interval(lowhigh), p + ".limits", "finite increasing command limits required")
            physical_limits = lowhigh
            if unit == "normalized" and section == "action":
                scale = c.obj(row.get("affine_to_si"), p + ".affine_to_si")
                c.need(scale.get("unit") == expected_unit and expected_unit is not None, p + ".affine_to_si.unit", "explicit physical unit required")
                if c.need(number(scale.get("scale")) and scale.get("scale") != 0 and number(scale.get("offset")), p + ".affine_to_si", "finite nonzero scale and offset required") and interval(lowhigh):
                    physical_limits = sorted(scale["scale"]*v+scale["offset"] for v in lowhigh)
                    c.need(interval(physical_limits), p + ".affine_to_si", "scaled bounds must remain finite and increasing")
            else:
                c.need(unit == expected_unit and expected_unit is not None, p + ".unit", "SI unit/quantity mismatch; no implicit conversion")
                c.need(row.get("affine_to_si") is None, p + ".affine_to_si", "SI channels must not hide a second scaling")
            c.need("joint" in row, p + ".joint", "explicit joint binding or null for non-joint channel required")
            joint_name = row.get("joint")
            if joint_name is not None:
                joint = joints.get(joint_name) if isinstance(joint_name, str) else None
                if c.need(joint is not None, p + ".joint", "joint binding absent from joint limits"):
                    family = "angular" if joint.get("kind") == "revolute" else "linear"
                    c.need(isinstance(quantity, str) and quantity.startswith(family + "_"), p + ".quantity", "joint type and channel quantity disagree")
                    if section == "action" and interval(physical_limits):
                        if quantity in ("angular_position", "linear_position") and interval(joint.get("limits")):
                            c.need(joint["limits"][0] <= physical_limits[0] < physical_limits[1] <= joint["limits"][1], p + ".limits", "command exceeds joint position bounds", "readiness")
                        if quantity in ("angular_velocity", "linear_velocity") and number(joint.get("max_velocity")):
                            c.need(max(abs(v) for v in physical_limits) <= joint["max_velocity"], p + ".limits", "command exceeds joint velocity bound", "readiness")
            elif row.get("frame") == "joint" and quantity != "gripper_aperture":
                c.need(False, p + ".joint", "joint-frame channel needs an explicit joint binding")

    for name in joints:
        c.need(any(row.get("joint") == name and row.get("quantity") in ("angular_position", "linear_position")
                   for row in channels["proprio"].values()), "joints." + name,
               "every controlled joint requires named position feedback")
    camera_names, keys = [], []
    depth_count = 0
    allowlist = c.strings(root.get("camera_allowlist"), "camera_allowlist")
    for i, item in enumerate(c.rows(root.get("cameras"), "cameras", 8)):
        p = "cameras[" + str(i) + "]"
        cam = c.obj(item, p)
        name = cam.get("name")
        c.need(text(name), p + ".name", "explicit camera name required")
        camera_names.append(name)
        device(cam.get("device"), "camera", p + ".device")
        rgb = c.obj(cam.get("rgb"), p + ".rgb")
        shape = rgb.get("shape")
        shape_ok = (isinstance(shape, list) and len(shape) == 3
                    and all(type(v) is int and 0 < v <= 16384 for v in shape)
                    and shape[2] == 3)
        c.need(shape_ok, p + ".rgb.shape", "explicit HxWx3 RGB shape required")
        c.need(rgb.get("layout") == "HWC" and rgb.get("encoding") == "rgb8", p + ".rgb", "HWC rgb8 boundary required")
        stamp(rgb.get("stamp"), p + ".rgb.stamp", timing.get("max_age_s"))
        k = c.obj(cam.get("intrinsics"), p + ".intrinsics")
        c.need(k.get("model") == "pinhole_rectified", p + ".intrinsics.model", "rectified pinhole boundary required; do not ignore distortion")
        c.need(k.get("unit") == "pixel", p + ".intrinsics.unit", "intrinsics must use pixels")
        c.need(shape_ok and k.get("resolution_hw") == shape[:2], p + ".intrinsics.resolution_hw", "intrinsics calibration resolution must match RGB")
        km = k.get("K")
        ok = matrix(km, 3)
        if ok:
            ok = (km[0][0] > 0 and km[1][1] > 0 and km[0][1] == 0 and km[1][0] == 0
                  and km[2] == [0, 0, 1] and shape_ok and 0 <= km[0][2] < shape[1] and 0 <= km[1][2] < shape[0])
        c.need(ok, p + ".intrinsics.K", "finite canonical K with positive focal lengths and in-image principal point required")
        c.need(text(k.get("calibration_id")), p + ".intrinsics.calibration_id", "calibration identity required (not verified by tool)")
        ex = c.obj(cam.get("extrinsics"), p + ".extrinsics")
        c.need(text(ex.get("from_frame")) and ex.get("from_frame") != base and ex.get("to_frame") == base,
               p + ".extrinsics", "distinct named camera frame -> named base frame required")
        c.need(ex.get("camera_axes") in ("opencv_x_right_y_down_z_forward", "usd_x_right_y_up_z_backward"),
               p + ".extrinsics.camera_axes", "explicit camera axes required; USD is not OpenCV")
        c.need(ex.get("translation_unit") == "m", p + ".extrinsics.translation_unit", "extrinsic translation must be metres")
        c.need(rigid(ex.get("matrix")), p + ".extrinsics.matrix", "finite rigid 4x4: orthogonal R, det +1, homogeneous last row required")
        c.need(text(ex.get("calibration_id")), p + ".extrinsics.calibration_id", "extrinsic/FK calibration identity required")
        c.need(ex.get("kind") in ("static", "dynamic"), p + ".extrinsics.kind", "static or dynamic required")
        if ex.get("kind") == "dynamic":
            stamp(ex.get("stamp"), p + ".extrinsics.stamp", timing.get("max_age_s"))
        depth = c.obj(cam.get("depth"), p + ".depth")
        c.need(type(depth.get("enabled")) is bool, p + ".depth.enabled", "explicit depth enabled/disabled declaration required")
        if depth.get("enabled") is False:
            c.need(text(rgb.get("key")), p + ".rgb.key", "explicit allowlisted RGB key required")
            keys.append(rgb.get("key"))
            continue
        if depth.get("enabled") is True:
            depth_count += 1
        c.need(depth.get("unit") == "m" and depth.get("quantity") == "z_depth", p + ".depth.unit", "metric z-depth required, not mm/range/depth-buffer")
        c.need(shape_ok and depth.get("shape") == shape[:2], p + ".depth.shape", "aligned depth shape must match RGB")
        c.need(depth.get("aligned_to_rgb") is True, p + ".depth.aligned_to_rgb", "RGB/depth alignment must be declared", "readiness")
        depth_range = depth.get("valid_range_m")
        c.need(interval(depth_range) and depth_range[0] > 0, p + ".depth.valid_range_m", "positive increasing usable metric range required")
        stamp(depth.get("stamp"), p + ".depth.stamp", timing.get("max_age_s"))
        mask = c.obj(depth.get("valid_mask"), p + ".depth.valid_mask")
        c.need(shape_ok and mask.get("shape") == shape[:2] and mask.get("dtype") == "bool", p + ".depth.valid_mask", "boolean mask matching depth shape required")
        c.need(mask.get("valid_when") == "finite_and_in_range", p + ".depth.valid_mask.valid_when", "mask must reject nonfinite, zero and out-of-range depths")
        c.need(mask.get("invalid_value") == "null", p + ".depth.valid_mask.invalid_value", "JSON boundary uses null for missing depth")
        for key, obj in (("rgb", rgb), ("depth", depth), ("mask", mask)):
            c.need(text(obj.get("key")), p + "." + key + ".key", "explicit allowlisted stream key required")
            keys.append(obj.get("key"))
        # Optional tiny offline value probe. Never claimed to be a measured frame.
        probe = depth.get("probe")
        if probe is not None:
            probe = c.obj(probe, p + ".depth.probe")
            values, valid = probe.get("values_m"), probe.get("valid_mask")
            ok = isinstance(values, list) and 0 < len(values) <= 4096 and isinstance(valid, list) and len(values) == len(valid)
            c.need(ok, p + ".depth.probe", "nonempty equal-length depth/mask probe required")
            if ok and interval(depth_range):
                for j, (v, m) in enumerate(zip(values, valid)):
                    expected = number(v) and depth_range[0] <= v <= depth_range[1]
                    c.need((v is None or number(v)) and type(m) is bool and m == expected,
                           p + ".depth.probe[" + str(j) + "]", "depth/mask disagreement or nonfinite value", "readiness")
    c.need(depth_count > 0, "cameras.depth", "at least one explicitly enabled RGB-D stream is required for this boundary")
    c.need(all(text(v) for v in camera_names) and len(set(v for v in camera_names if isinstance(v, str))) == len(camera_names), "cameras", "camera names must be unique")
    c.need(all(text(v) for v in keys) and len(set(v for v in keys if isinstance(v, str))) == len(keys), "camera_allowlist", "stream keys must be unique")
    c.need(sorted(v for v in keys if isinstance(v, str)) == sorted(v for v in allowlist if isinstance(v, str)),
           "camera_allowlist", "exact declared RGB/depth/mask keys required; no extra privileged/object streams")
    if sensor_times and number(timing.get("max_skew_s")):
        c.need(max(sensor_times)-min(sensor_times) <= timing["max_skew_s"], "timing.max_skew_s", "sensor/extrinsic timestamps are unsynchronized", "readiness")

    grip_actions = set()
    for i, item in enumerate(c.rows(root.get("grippers"), "grippers", 8)):
        p = "grippers[" + str(i) + "]"
        g = c.obj(item, p)
        action_name = g.get("action_name")
        action = channels["action"].get(action_name, {}) if isinstance(action_name, str) else {}
        c.need(action.get("quantity") == "gripper_aperture", p + ".action_name", "binding must name a gripper-aperture command")
        if isinstance(action_name, str):
            c.need(action_name not in grip_actions, p + ".action_name", "duplicate gripper binding")
            grip_actions.add(action_name)
        op, cl = g.get("open_command"), g.get("closed_command")
        valid_commands = number(op) and number(cl) and op != cl
        c.need(valid_commands, p, "distinct finite open/closed commands required")
        if valid_commands:
            c.need(g.get("larger_command") == ("open" if op > cl else "close"), p + ".larger_command", "gripper open/close sign disagrees")
            limits = action.get("limits")
            c.need(interval(limits) and limits[0] <= min(op, cl) < max(op, cl) <= limits[1], p, "gripper commands outside action bounds")
            scale = action.get("affine_to_si")
            if isinstance(scale, dict) and number(scale.get("scale")):
                c.need((op-cl)*scale["scale"] > 0, p, "open command must produce larger aperture")
            elif action.get("unit") == "m":
                c.need(op > cl, p, "metric aperture increases when opening")
        feedback = c.strings(g.get("feedback_joints"), p + ".feedback_joints")
        c.need(g.get("feedback_aggregation") == "mean", p + ".feedback_aggregation",
               "aperture boundary is explicit mean finger position in metres")
        feedback_limits = []
        for name in feedback:
            j = joints.get(name, {}) if isinstance(name, str) else {}
            if interval(j.get("limits")):
                feedback_limits.append(j["limits"])
            c.need(j.get("kind") == "prismatic" and j.get("unit") == "m", p + ".feedback_joints", "explicit prismatic finger joints in metres required")
            c.need(any(row.get("joint") == name and row.get("quantity") == "linear_position" for row in channels["proprio"].values()),
                   p + ".feedback_joints", "finger position missing from proprio layout")
        if valid_commands and feedback_limits and len(feedback_limits) == len(feedback):
            scale = action.get("affine_to_si")
            si_open, si_closed = None, None
            if isinstance(scale, dict) and number(scale.get("scale")) and number(scale.get("offset")):
                si_open, si_closed = (scale["scale"]*v+scale["offset"] for v in (op, cl))
            elif action.get("unit") == "m":
                si_open, si_closed = op, cl
            if number(si_open) and number(si_closed):
                low, high = (sum(limits[k] for limits in feedback_limits)/len(feedback_limits) for k in (0, 1))
                c.need(low <= si_closed < si_open <= high, p + ".feedback_joints",
                       "open/closed physical aperture exceeds mean finger limits", "readiness")
        c.need(text(g.get("hold_semantics")), p + ".hold_semantics", "explicit gripper hold semantics required; not force assurance")
    c.need(grip_actions == {name for name, row in channels["action"].items() if row.get("quantity") == "gripper_aperture"}, "grippers", "every aperture command must have one sign/feedback declaration")

    safety = c.obj(root.get("safety"), "safety")
    watchdog = c.obj(safety.get("watchdog"), "safety.watchdog")
    for field in ("enabled", "independent_of_policy"):
        c.need(watchdog.get(field) is True, "safety.watchdog." + field, "device-level watchdog must be declared", "readiness")
    c.need(watchdog.get("location") == "device", "safety.watchdog.location", "policy/network loop is not a device watchdog", "readiness")
    c.need(watchdog.get("on_expiry") == "stop", "safety.watchdog.on_expiry", "expired commands must stop, not persist", "readiness")
    timeout_ok = c.positive(watchdog.get("timeout_s"), "safety.watchdog.timeout_s")
    if timeout_ok and number(timing.get("action_hz")) and timing["action_hz"] > 0 and number(timing.get("max_command_age_s")):
        c.need(1/timing["action_hz"] <= watchdog["timeout_s"] <= timing["max_command_age_s"],
               "safety.watchdog.timeout_s", "watchdog must cover a control period but not exceed command age budget", "readiness")
    for section in ("hold", "stop", "estop"):
        obj = c.obj(safety.get(section), "safety." + section)
        c.need(obj.get("supported") is True, "safety." + section + ".supported", "required safety function undeclared", "readiness")
        for field in ("interface", "semantics"):
            c.need(text(obj.get(field)), "safety." + section + "." + field, "explicit interface/behavior required; tool never invokes it")
    hold = safety.get("hold") if isinstance(safety.get("hold"), dict) else {}
    c.need(hold.get("on_stale_proprio") == "stop", "safety.hold.on_stale_proprio", "must not synthesize copy-state hold from stale proprio", "readiness")
    estop = safety.get("estop") if isinstance(safety.get("estop"), dict) else {}
    for field in ("physical", "latched", "manual_reset"):
        c.need(estop.get(field) is True, "safety.estop." + field, "physical latched manual-reset emergency stop declaration required", "readiness")
    device(estop.get("device"), "estop", "safety.estop.device")
    schema_ok = not any(i["kind"] == "schema" for i in c.issues)
    ready = not c.issues
    return {"report_version": 1, "schema_status": "PASS" if schema_ok else "BLOCKED",
            "readiness_status": "PASS" if ready else "BLOCKED", "pass_scope": "offline_schema_and_declared_readiness_only",
            "evidence": "declared_only", "reference_clock": {"now": now, "domain": clock_domain},
            "hardware_execution": {"status": "NOT_TESTED", "admitted": False}, "real_robot_admission": False,
            "model_compatibility": compare_model(root, model_profile), "issues": c.issues,
            "limitations": ["No device access, commands, model inference or simulator ground truth.",
                            "No verification of image pixels, measured calibration, transport latency, forces or stopping distance.",
                            "Availability, safety behavior and timestamps are declarations; even PASS is not real-robot clearance."]}


def load_json(path):
    p = Path(path)
    if p.stat().st_size > MAX_BYTES:
        raise ValueError("input exceeds 2 MiB contract limit")
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out:
                raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("contract", help="explicit local JSON declaration, never a device URL")
    ap.add_argument("--model-profile", help="optional explicit expected model-boundary JSON")
    ap.add_argument("--clock-domain", required=True, help="unix_epoch or explicit clock/boot identifier")
    ap.add_argument("--now", type=float, help="reference seconds in that domain; required for non-unix clocks")
    ap.add_argument("--output", help="also save machine-readable report locally")
    args = ap.parse_args(argv)
    try:
        now = args.now
        if now is None:
            if args.clock_domain != "unix_epoch":
                raise ValueError("non-unix clock requires explicit --now")
            now = time.time()
        if not number(now):
            raise ValueError("reference time must be finite")
        report = validate(load_json(args.contract), now=now, clock_domain=args.clock_domain,
                          model_profile=load_json(args.model_profile) if args.model_profile else None)
        rc = 0 if report["readiness_status"] == "PASS" else 2
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        report = {"report_version": 1, "schema_status": "BLOCKED", "readiness_status": "BLOCKED",
                  "pass_scope": "offline_schema_and_declared_readiness_only", "input_error": str(exc),
                  "hardware_execution": {"status": "NOT_TESTED", "admitted": False}, "real_robot_admission": False}
        rc = 3
    data = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        try:
            Path(args.output).write_text(data, encoding="utf-8")
        except OSError as exc:
            print("cannot write report: " + str(exc), file=sys.stderr)
            return 3
    print(data, end="")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
