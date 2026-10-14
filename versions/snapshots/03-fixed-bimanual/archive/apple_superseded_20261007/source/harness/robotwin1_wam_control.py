"""RoboTwin 1.0 commanded-qpos execution; CPU-safe until explicitly constructed.

This is NOT RoboTwin2 take_action/native8. The worker may queue eight actions,
but each action here forms two [current command, target] paths for official
MPLib TOPP at 250 Hz. Drive/passive-force/progress synchronization follows
Base_task.apply_policy_demo/apply_pi. Grippers use apply_pi's current-to-target
interpolation, directly in metres, without its policy-side 0..1 conversion.
No IK, collision planner, expert, recovery or measured-qpos feedback.
"""
from __future__ import annotations

import numpy as np
from harness.robotwin_policy_worker import InfrastructureError

EXECUTION_MODE = "native1_qpos8"
RGB_VIEWS = ("head_camera", "left_camera", "right_camera")
ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
TIMESTEP = 1 / 250
GRIPPER_MAX = 0.045


def physical14(value):
    """Validate, do not scale/clip model outputs into another embodiment."""
    a = np.asarray(value)
    if a.shape != (14,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise InfrastructureError("RoboTwin1 requires finite real 14D commanded qpos")
    # Float32's representation of .045 is slightly above Python's float64 .045.
    upper = np.asarray(GRIPPER_MAX, dtype=a.dtype) if a.dtype.kind == "f" else GRIPPER_MAX
    if np.any(a[[6, 13]] < 0) or np.any(a[[6, 13]] > upper):
        raise InfrastructureError("RoboTwin1 grippers must be physical 0..0.045 m, not normalized 0..1")
    return a.copy()


def commanded_state(env):
    """Official getters read drive targets + stored physical gripper commands."""
    return physical14(np.concatenate((env.get_left_arm_jointState(), env.get_right_arm_jointState())))


def capture_observation(env):
    """Original camera RGB path, not get_obs (which also computes GT/PCD).

    Render/read only: no hidden scene.step, depth, segmentation, end pose,
    front view, actor state or task geometry. Original D435 is 320x240.
    """
    env._update_render()
    rgb = {}
    for name in RGB_VIEWS:
        camera = getattr(env, name)
        camera.take_picture()
        image = np.asarray(env._get_camera_rgba(camera))
        if image.dtype != np.uint8 or image.shape != (240, 320, 4):
            raise InfrastructureError(f"{name}: original D435 uint8 240x320 RGBA required")
        rgb[name] = image[..., :3].copy()  # original RGB, no RGB/BGR swap
    return {"rgb": rgb, "commanded_joint_gripper_state": commanded_state(env)}


class ToppOnlyPlanner:
    """Expose only time parameterization, explicitly reject IK/planning APIs."""
    __slots__ = ("__topp",)

    def __init__(self, planner):
        self.__topp = planner.TOPP

    def TOPP(self, *args, **kwargs):
        return self.__topp(*args, **kwargs)

    def __getattr__(self, name):
        raise InfrastructureError(f"pure WAM rejects planner operation {name}; TOPP only")


def setup_topp_planners(env, mplib_module=None, **kwargs):
    """Official constructor/base pose with MPLib 0.2.1 Pose, never a raw list."""
    if mplib_module is None:
        import mplib as mplib_module
    # 0.2.1 exposes Pose at mplib.Pose (and mplib.pymp.Pose).
    pose_type = getattr(mplib_module, "Pose", None)
    if pose_type is None:
        pose_type = mplib_module.pymp.Pose
    for side, group in (("left", "fl_link6"), ("right", "fr_link6")):
        planner = mplib_module.Planner(
            urdf=kwargs.get("urdf_path", "./aloha_maniskill_sim/urdf/arx5_description_isaac.urdf"),
            srdf=kwargs.get("srdf_path", "./aloha_maniskill_sim/srdf/arx5_description_isaac.srdf"),
            move_group=group)
        # MPLib normalizes the quaternion as did the old API. Same world pose.
        planner.set_base_pose(pose_type(p=[0, -0.65, 0], q=[1, 0, 0, 1]))
        setattr(env, side + "_planner", ToppOnlyPlanner(planner))


def _forbidden(*args, **kwargs):
    raise InfrastructureError("pure WAM rejects expert/IK/pose planning")


def qpos_environment_class(native_class):
    """Keep original setup/load_camera/pre_move/check_success/close intact."""
    namespace = {"setup_planner": setup_topp_planners}
    for name in ("play_once", "apply_policy_demo", "apply_pi", "get_grasp_pose_w_given_direction",
                 "left_move_to_pose", "right_move_to_pose", "together_move_to_pose",
                 "left_move_to_pose_with_screw", "right_move_to_pose_with_screw",
                 "left_follow_path", "right_follow_path", "together_follow_path"):
        namespace[name] = _forbidden
    return type("PureWamRoboTwin1", (native_class,), namespace)


class Native1QposController:
    def __init__(self, env, check=None, max_topp_samples=10000):
        self.env = env
        self.check = check or (lambda: None)
        self.max_topp_samples = max_topp_samples
        self.control_steps = 0
        self.physics_steps = 0
        if not np.isclose(env.scene.get_timestep(), TIMESTEP, rtol=0, atol=1e-10):
            raise InfrastructureError("RoboTwin1 TOPP requires original 1/250 scene timestep")
        if getattr(env, "fix_gripper", False):
            raise InfrastructureError("fixed grippers forbidden for pure WAM")
        limits = []
        for index in list(env.left_arm_joint_id) + list(env.right_arm_joint_id):
            limit = np.asarray(env.active_joints[index].get_limits())
            if limit.shape != (1, 2) or not np.isfinite(limit).all() or limit[0, 0] > limit[0, 1]:
                raise InfrastructureError("invalid native arm joint limits")
            limits.append(limit[0])
        self.limits = np.asarray(limits)

    def _trajectory(self, planner, current, target):
        self.check()
        # No straight-line/hold fallback if TOPP errors or returns an empty path.
        _, pos, vel, _, duration = planner.TOPP(np.vstack((current, target)), TIMESTEP, verbose=True)
        self.check()
        pos, vel = np.asarray(pos), np.asarray(vel)
        if (pos.ndim != 2 or pos.shape[1:] != (6,) or vel.shape != pos.shape
                or not 1 <= len(pos) <= self.max_topp_samples
                or not np.isfinite(pos).all() or not np.isfinite(vel).all()
                or not np.isfinite(duration) or duration < 0):
            raise InfrastructureError("invalid/empty/over-budget RoboTwin1 TOPP trajectory")
        # Never label a partially parameterized target as executed.
        if not np.allclose(pos[-1], target, rtol=0, atol=1e-5):
            raise InfrastructureError("RoboTwin1 TOPP did not reach commanded target")
        return pos, vel

    def step(self, raw_action):
        env = self.env
        self.check()
        target = physical14(raw_action)
        joints = target[list(ARM_INDICES)]
        if np.any(joints < self.limits[:, 0]) or np.any(joints > self.limits[:, 1]):
            raise InfrastructureError("model qpos exceeds native joint limits (not clipped)")
        current = commanded_state(env)  # intentionally NOT robot.get_qpos()
        left = self._trajectory(env.left_planner, current[:6], target[:6])
        right = self._trajectory(env.right_planner, current[7:13], target[7:13])
        nl, nr = len(left[0]), len(right[0])
        # Single-action version of official apply_pi's gripper interpolation;
        # its normalization is deliberately absent, both endpoints are metres.
        grips = (np.linspace(current[6], target[6], nl + 1)[1:],
                 np.linspace(current[13], target[13], nr + 1)[1:])
        il = ir = steps = 0
        success = False

        def drive(trajectory, sample, ids, finger_ids, grip, side):
            for j, joint_id in enumerate(ids):
                env.active_joints[joint_id].set_drive_target(trajectory[0][sample, j])
                env.active_joints[joint_id].set_drive_velocity_target(trajectory[1][sample, j])
            for joint_id in finger_ids:
                env.active_joints[joint_id].set_drive_target(grip[sample])
                env.active_joints[joint_id].set_drive_velocity_target(0.05)
            setattr(env, side + "_gripper_val", float(grip[sample]))

        while il < nl or ir < nr:
            self.check()
            env.robot.set_qf(env.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            # Preserve official sequential progress comparison, not zip/truncate.
            if il < nl and il / nl <= ir / nr:
                drive(left, il, env.left_arm_joint_id, range(34, 36), grips[0], "left")
                il += 1
            if ir < nr and ir / nr <= il / nl:
                drive(right, ir, env.right_arm_joint_id, range(36, 38), grips[1], "right")
                ir += 1
            env.scene.step()
            steps += 1
            self.physics_steps += 1
            self.check()
            # Scoring only: no GT coordinates/actions leave the native predicate.
            success = bool(env.check_success())
            if success:
                break
        self.control_steps += 1
        return {"execution_mode": EXECUTION_MODE, "physics_steps": steps,
                "left_topp_samples": nl, "right_topp_samples": nr,
                "left_executed_samples": il, "right_executed_samples": ir,
                "target_fully_executed": il == nl and ir == nr,
                "native_benchmark_success": success, "raw_action": target.tolist(),
                "commanded_state_after": commanded_state(env).tolist()}
