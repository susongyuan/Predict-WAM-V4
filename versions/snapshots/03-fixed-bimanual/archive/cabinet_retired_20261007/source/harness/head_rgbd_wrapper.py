"""Evaluator env wrapper for the Task 1 closed loop: head camera RGB + depth_linear at the data-collection resolution
(720x720, as omnigibson.eval.wrappers.RGBDFullResWrapper), wrist cameras RGB only at 224x224 (the WAM input size, as
omnigibson.eval.wrappers.DefaultWrapper).

Why (progress.md 2026-09-27 01:10): with RGBDFullResWrapper the scene steady state is 13.2 GB (desktop included), and
the measured 3-view K8 WAM process peak is 1964 MiB, so simulator + policy exceed the 15300 MiB guard and the policy
server never starts. Perception only needs head RGB-D; the WAM resizes every view to 224, so rendering the wrists at
224 without depth removes the unused wrist depth/high-res buffers. Only camera resolution/modalities change (same
information class the challenge allows: RGB + depth + proprio); protocol deviation recorded in the run manifest.

Used via `--env-wrapper harness.head_rgbd_wrapper.HeadRGBDWrapper` (evaluator cwd = project root).
"""
import os

import omnigibson as og
from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import HEAD_RESOLUTION, get_robot_camera_names, set_sensor_modalities
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)
WRIST_RES = int(os.environ.get("P2_WRIST_RES", "224"))
HEAD_RES = int(os.environ.get("P2_HEAD_RES", str(HEAD_RESOLUTION[0])))
if HEAD_RES < 448 or HEAD_RES > int(HEAD_RESOLUTION[0]):
    raise ValueError(f"P2_HEAD_RES must be in [448, {HEAD_RESOLUTION[0]}], got {HEAD_RES}")


class HeadRGBDWrapper(EnvironmentWrapper):
    def __init__(self, env: Environment):
        super().__init__(env=env)
        robot = env.robots[0]
        roles = {name.split("::")[1]: cid
                 for cid, name in get_robot_camera_names(robot.name, getattr(env, "_eval_robot_config", {})).items()}
        for sensor_name, sensor in robot.sensors.items():
            if not hasattr(sensor, "image_height") or not hasattr(sensor, "image_width"):
                continue
            if roles.get(sensor_name) == "head":
                set_sensor_modalities(sensor, {"rgb", "depth_linear"})
                sensor.image_height = sensor.image_width = HEAD_RES
            else:
                set_sensor_modalities(sensor, {"rgb"})
                sensor.image_height = sensor.image_width = WRIST_RES
        og.sim.update_handles()
        env.load_observation_space()
        logger.info("HeadRGBDWrapper: head RGB-D %dx%d, wrists RGB %dx%d", HEAD_RES, HEAD_RES, WRIST_RES, WRIST_RES)
