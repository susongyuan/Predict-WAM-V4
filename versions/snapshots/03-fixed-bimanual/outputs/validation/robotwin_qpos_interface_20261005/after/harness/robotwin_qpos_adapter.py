"""CPU-only boundary for RoboTwin native joint-target policies.

Policy input is three onboard RGB images plus commanded drive-target/gripper
state, NOT measured qpos. No task/actor poses, rewards, GT, or extra views are
read. This module does not render, plan IK, call an expert, or run a policy.

Configure the environment with planner_mode="joint_targets" in setup kwargs.
Use policy_observation(existing_raw_obs) and NativeQposAdapter(env).step(action).
The latter runs the existing native simulator action only when called by the
application; importing this module and validating data do not start simulation.
"""

from contextlib import contextmanager

import numpy as np


RGB_VIEWS = ("head_camera", "left_camera", "right_camera")
ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


class InvalidAction(ValueError):
    """Invalid input is retained, never clipped, rounded, or replaced."""

    def __init__(self, message, original):
        super().__init__(message)
        self.original = original


def _finite_array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def validate_action(action, joint_limits=None):
    """Return an unchanged-value/dtype copy of a single native 14D action.

    Order: left arm[6], left gripper, right arm[6], right gripper.
    Grippers must lie in [0, 1]. Optional externally supplied joint_limits has
    shape (12, 2): lower/upper bounds for left[6] then right[6], in native units.
    No joint limits are inferred, guessed, or obtained from object/task state.
    Raises InvalidAction (including for invalid limits) before any actuation.
    """
    try:
        values = _finite_array(action, (14,), "action")
        if np.any(values[[6, 13]] < 0) or np.any(values[[6, 13]] > 1):
            raise ValueError("gripper commands must be in [0, 1]")
        if joint_limits is not None:
            limits = _finite_array(joint_limits, (12, 2), "joint_limits")
            if np.any(limits[:, 0] > limits[:, 1]):
                raise ValueError("joint_limits lower bound exceeds upper bound")
            joints = values[list(ARM_INDICES)]
            if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
                raise ValueError("joint commands exceed explicit joint_limits")
    except (ValueError, TypeError, OverflowError) as exc:
        raise InvalidAction(str(exc), action) from exc
    return values.copy()


def policy_observation(raw_obs):
    """Strict allowlist; extra raw fields/views are ignored without reading them.

    Native get_*_arm_jointState uses get_drive_target plus stored normalized
    gripper commands. This is commanded state even if the robot did not move.
    Require native uint8 HWC RGB; do not normalize, resize, or synthesize views.
    """
    cameras = raw_obs["observation"]
    rgb = {}
    for name in RGB_VIEWS:
        image = np.asarray(cameras[name]["rgb"])
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
            raise ValueError(f"{name} RGB must have nonempty HWC shape with 3 channels")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} RGB must be native uint8")
        rgb[name] = image.copy()

    joints = raw_obs["joint_action"]
    left = _finite_array(joints["left_arm"], (6,), "left commanded joints")
    right = _finite_array(joints["right_arm"], (6,), "right commanded joints")
    lg = _finite_array(joints["left_gripper"], (), "left commanded gripper")
    rg = _finite_array(joints["right_gripper"], (), "right commanded gripper")
    state = validate_action(np.concatenate((left, lg.reshape(1), right, rg.reshape(1))))
    return {"rgb": rgb, "commanded_joint_gripper_state": state}


@contextmanager
def _observe_topp(robot, records):
    """Observe existing TOPP calls, re-raise unchanged, restore even on errors.

    Native take_action swallows TOPP exceptions/empty paths and continues with
    old arm drive targets plus grippers (50 fallback samples for a failed arm).
    Diagnostics here detect that fallback, NOT collision safety, measured
    motion, task success, or whether an early-success return executed all steps.
    Single-threaded use only: do not share an env across simultaneous steps.
    """
    originals = []

    def wrap(topp, arm):
        def observed(*args, **kwargs):
            record = {"arm": arm, "status": "called"}
            records.append(record)
            try:
                result = topp(*args, **kwargs)
            except Exception as exc:
                record.update(status="exception", error_type=type(exc).__name__, error=str(exc))
                raise
            try:
                steps = int(result[1].shape[0])
                record.update(status="empty" if steps == 0 else "returned", steps=steps)
            except (AttributeError, IndexError, TypeError, ValueError):
                record["status"] = "unrecognized_result"
            return result
        return observed

    try:
        for arm in ("left", "right"):
            planner = getattr(robot, f"{arm}_mplib_planner")
            original = planner.TOPP
            originals.append((planner, original))
            planner.TOPP = wrap(original, arm)
        yield
    finally:
        for planner, original in reversed(originals):
            planner.TOPP = original


class NativeQposAdapter:
    """Validated native qpos handoff; diagnostics stay outside policy input.

    last_topp_diagnostics is reset per attempt. An empty list means no TOPP
    call was observed (e.g. terminal env or invalid input), never success.
    Validation is fail-closed; valid values reach native take_action unchanged.
    Native gripper slew/drive behavior and TOPP timing are intentionally intact.
    """

    def __init__(self, env, joint_limits=None):
        if getattr(env.robot, "planner_mode", None) != "joint_targets":
            raise ValueError("NativeQposAdapter requires explicit planner_mode='joint_targets'")
        self.env = env
        self.joint_limits = None if joint_limits is None else np.array(joint_limits, copy=True)
        self.last_topp_diagnostics = []

    def step(self, action):
        self.last_topp_diagnostics = []
        values = validate_action(action, self.joint_limits)
        # Do not let a differently sized embodiment reinterpret the 14D layout.
        _finite_array(self.env.robot.get_left_arm_jointState(), (7,), "left commanded state")
        _finite_array(self.env.robot.get_right_arm_jointState(), (7,), "right commanded state")
        with _observe_topp(self.env.robot, self.last_topp_diagnostics):
            return self.env.take_action(values, action_type="qpos")
