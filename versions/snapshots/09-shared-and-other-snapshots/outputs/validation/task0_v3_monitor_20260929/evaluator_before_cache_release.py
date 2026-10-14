import cv2
import hashlib
import json
import logging
import os
import sys
import traceback
from signal import SIGINT, signal
from typing import Any, List, Tuple

import numpy as np
import torch as th
from av.container import Container
from av.stream import Stream
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import omnigibson as og
import omnigibson.utils.transform_utils as T
from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.utils.og_teleop_utils import load_available_tasks
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    EVAL_TIMEOUT_MULTIPLIER,
    NUM_HIDDEN_TEST_INSTANCES,
    NUM_PUBLIC_TEST_INSTANCES,
    TASK_NAMES_TO_INDICES,
    TASK_NAMES_TO_ROOMS,
    TEST_INSTANCE_IDS,
    flatten_obs_dict,
    generate_basic_environment_config,
    get_robot_camera_names,
)
from omnigibson.eval.utils.obs_utils import create_video_writer, write_video
from omnigibson.eval.utils import recording_utils
from omnigibson.eval.utils.recording_utils import ObserverFraming, native_rgb_frame, validate_viewer_config, viewer_pose
from omnigibson.eval.utils.score_utils import load_human_stats
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import Robot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.eval.utils.light_utils import LightToggleSynchronizer, set_light_control_toggles
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import create_module_logger

TORCH_NUM_THREADS = None
TORCH_NUM_INTEROP_THREADS = None

if TORCH_NUM_THREADS is not None:
    th.set_num_threads(TORCH_NUM_THREADS)
if TORCH_NUM_INTEROP_THREADS is not None:
    th.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)

LIGHT_EVAL_TASKS = {"turning_out_all_lights_before_sleep"}
EVAL_BASE_LINK_MASS = 250.0
EVAL_HEAD_HORIZONTAL_APERTURE = 40.0
DEFAULT_ROBOT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "r1pro.yaml")
EVAL_MODES = ("train", "public_test", "hidden_test")

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def _to_plain_dict(cfg: Any) -> dict | None:
    if cfg is None:
        return None
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def resolve_instance_ids(task_name: str, instance_indices: list[int], mode: str = "public_test") -> list[int]:
    assert mode in EVAL_MODES, f"Mode must be one of {EVAL_MODES}, got {mode}"
    if mode == "train":
        return [int(instance_id) for instance_id in instance_indices]

    test_instances = (
        TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES]
        if mode == "public_test"
        else TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:]
    )
    num_split_instances = NUM_PUBLIC_TEST_INSTANCES if mode == "public_test" else NUM_HIDDEN_TEST_INSTANCES
    assert set(instance_indices).issubset(
        set(range(num_split_instances))
    ), f"Instance indices must be in range({num_split_instances}) for mode {mode}"
    return [int(test_instances[i]) for i in instance_indices]


class Evaluator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()
        self.robot_name = None
        self.robot_eval_config = {}
        self.robot_camera_names = {}

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.robot = self.load_robot()
        self.robot_camera_names = get_robot_camera_names(self.robot.name, self.robot_eval_config)
        self._validate_robot_eval_config()
        self._apply_robot_eval_settings()
        self.policy = self.load_policy()
        self.metrics = self.load_metrics()
        self.obs = None
        self.light_synchronizer = None

        # Initialize the physics views so the first reset() (which eval.py issues before the first
        # load_task_instance) can read/restore robot joint state.
        og.sim.update_handles()

        self.env._current_episode = 0
        self._video_writer = None
        self._video_path = None
        self._video_rate = 30
        self._video_frames = 0
        self._video_metadata = None
        self._framing = None          # evaluator-only observer camera selection; never policy input
        self._framing_log = None
        self._framing_pois = []
        self._joint_log = None       # evaluator-side physical evidence (object joint angles); never policy input
        self._joint_log_entities = {}
        self._joint_log_step = 0
        self._pose_log = None        # evaluator-side ground-truth poses (opt-in --log-object-poses); never policy input
        self._pose_log_entities = {}
        self._pose_log_goal = []
        self._pose_log_step = 0

    def start_joint_log(self, fpath: str, episode_identity: dict | None = None) -> None:
        """Log articulated task-object joint positions every environment step (privileged, evaluator-side).

        Purpose: independent physical post-condition evidence (e.g. did the car trunk lid actually open),
        separate from the policy's own visual completion heuristics. Nothing here reaches the policy.
        """
        self.stop_joint_log()
        if os.path.exists(fpath):
            raise FileExistsError(f"Refusing to overwrite joint log: {fpath}")
        entities = {}
        for inst, entity in self.env.task.object_scope.items():
            if is_system_bddl_inst(inst) or entity is None or inst.startswith("agent.") or not hasattr(entity, "n_joints"):
                continue
            if entity.n_joints > 0:
                entities[inst] = entity
        self._joint_log_entities = entities
        self._joint_log_step = 0
        self._joint_log = open(fpath, "w")
        header = {"type": "header", "episode_identity": episode_identity,
                  "step_semantics": "step N is the observation before policy action N; reset is step 0",
                  "privileged_data_use": "object joint state for evaluator-side evidence only; never policy input",
                  "entities": {inst: {"name": entity.name, "joints": list(entity.joints.keys()),
                                      "lower": [float(x) for x in entity.joint_lower_limits],
                                      "upper": [float(x) for x in entity.joint_upper_limits]}
                               for inst, entity in entities.items()}}
        self._joint_log.write(json.dumps(header) + "\n")
        self._write_joint_log()

    def _write_joint_log(self) -> None:
        if self._joint_log is None:
            return
        row = {"step": self._joint_log_step,
               "joints": {inst: [float(x) for x in entity.get_joint_positions()] for inst, entity in self._joint_log_entities.items()}}
        self._joint_log.write(json.dumps(row) + "\n")
        self._joint_log_step += 1

    def stop_joint_log(self) -> None:
        if self._joint_log is not None:
            self._joint_log.write(json.dumps({"type": "end", "environment_steps": self._joint_log_step - 1}) + "\n")
            self._joint_log.close()
        self._joint_log = None
        self._joint_log_entities = {}

    @staticmethod
    def _pose7(entity) -> list:
        position, orientation = entity.get_position_orientation()
        return [float(x) for x in position] + [float(x) for x in orientation]

    @staticmethod
    def _floats(value):
        if value is None:
            return None
        value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        return np.asarray(value, dtype=float).tolist()

    @staticmethod
    def _goal_literal(pred) -> str:
        body = getattr(pred, "body", None)
        if isinstance(body, (list, tuple)) and all(isinstance(term, str) for term in body):
            return "(" + " ".join(body) + ")"
        return json.dumps(body, default=str)

    def _pose_log_entity_info(self, entity) -> dict:
        info = {"name": getattr(entity, "name", None), "category": getattr(entity, "category", None),
                "pose_source": "entity.get_position_orientation() (root link, world frame)"}
        for key, getter in (("aabb_at_start_world", lambda: list(entity.aabb)),
                            ("native_bbox_object_frame", lambda: entity.native_bbox),
                            ("scale", lambda: entity.scale)):
            try:
                value = getter()
                info[key] = [self._floats(v) for v in value] if key == "aabb_at_start_world" else self._floats(value)
            except Exception as error:  # evidence metadata only; never fail an evaluation over it
                info[key] = f"unavailable: {type(error).__name__}"
        return info

    def _pose_log_camera_info(self) -> dict:
        roles = [role for role, name in self.robot_camera_names.items() if name.split("::")[1] in self.robot.sensors]
        cameras = {"cam_rel_poses_order": roles,
                   "cam_rel_poses_format": ("policy obs '<robot>::cam_rel_poses' = concat over cam_rel_poses_order of "
                                            "[x, y, z, qx, qy, qz, qw] of each camera in the robot base frame "
                                            "(robot_base pose below); camera axes follow the USD convention "
                                            "(-Z forward, +Y up)")}
        for role in roles:
            sensor = self.robot.sensors[self.robot_camera_names[role].split("::")[1]]
            entry = {"sensor": self.robot_camera_names[role]}
            for key, getter in (("image_hw", lambda: [int(sensor.image_height), int(sensor.image_width)]),
                                ("intrinsic_matrix", lambda: self._floats(sensor.intrinsic_matrix)),
                                ("horizontal_aperture", lambda: float(sensor.horizontal_aperture)),
                                ("focal_length", lambda: float(sensor.focal_length)),
                                ("modalities", lambda: sorted(sensor.modalities))):
                try:
                    entry[key] = getter()
                except Exception as error:
                    entry[key] = f"unavailable: {type(error).__name__}"
            cameras[role] = entry
        return cameras

    def start_pose_log(self, fpath: str, episode_identity: dict | None = None) -> None:
        """Log world poses of task-scope objects and of the robot base every environment step (privileged, evaluator-side).

        Opt-in (--log-object-poses), mirrors start_joint_log. Purpose: ground truth for offline perception checks and
        physical post-condition evidence (e.g. did each object end inside its container). Systems, floors and the
        agent entry are skipped; the robot base pose is logged separately. The BDDL ground goal literals are
        evaluated per step when there are at most 64 of them (object states cache per simulator step). Nothing here
        reaches the policy.
        """
        self.stop_pose_log()
        if os.path.exists(fpath):
            raise FileExistsError(f"Refusing to overwrite pose log: {fpath}")
        entities = {}
        for inst, entity in self.env.task.object_scope.items():
            if (entity is None or is_system_bddl_inst(inst) or inst.split(".")[0] in ("agent", "floor")
                    or entity is self.robot or not hasattr(entity, "get_position_orientation")):
                continue
            entities[inst] = entity
        options = list(getattr(self.env.task, "ground_goal_state_options", None) or [])
        n_literals = sum(len(option) for option in options)
        # ToggledOn objects (e.g. Task 0 radio): log value + hand_in_marker_steps (robot_can_toggle_steps) per step.
        # Matched by class name (no import); only present when such an object is logged. Evaluator-side evidence only.
        toggles = {}
        for inst, entity in entities.items():
            for state_cls, state in (getattr(entity, "states", None) or {}).items():
                if getattr(state_cls, "__name__", "") == "ToggledOn":
                    toggles[inst] = state
        self._pose_log_toggles = toggles
        self._pose_log_entities = entities
        self._pose_log_goal = options if 0 < n_literals <= 64 else []
        self._pose_log_step = 0
        self._pose_log = open(fpath, "w")
        header = {"type": "header", "episode_identity": episode_identity,
                  "step_semantics": "step N is the observation before policy action N; reset is step 0",
                  "privileged_data_use": "evaluator-side evidence only; never policy input",
                  "frame": "world", "pose_format": "[x, y, z, qx, qy, qz, qw] (metres; quaternion x, y, z, w)",
                  "robot_base": {"name": self.robot.name,
                                 "pose_source": "robot.get_position_orientation() (base footprint link; the base pose "
                                                "used for cam_rel_poses)"},
                  "entities": {inst: self._pose_log_entity_info(entity) for inst, entity in entities.items()},
                  "goal_literals": {"source": ("env.task.ground_goal_state_options: BDDL ground goal literals, evaluated "
                                               "with env.task._evaluate_predicate after each step; the q_score counts "
                                               "literals false at reset and true at the end"),
                                    "logged": bool(self._pose_log_goal), "count": n_literals,
                                    "options": [[self._goal_literal(pred) for pred in option] for option in self._pose_log_goal]},
                  "cameras": self._pose_log_camera_info()}
        if toggles:
            header["object_states"] = {"toggled_on": {
                "instances": sorted(toggles),
                "row_field": "toggled_on: {instance: [value, hand_in_marker_steps]}",
                "source": ("object_states/toggle.py ToggledOn.value and .robot_can_toggle_steps (the state's own "
                           "hand_in_marker_steps; the value flips when it reaches CAN_TOGGLE_STEPS), read after each step")}}
        self._pose_log.write(json.dumps(header) + "\n")
        self._write_pose_log()

    def _write_pose_log(self) -> None:
        if self._pose_log is None:
            return
        row = {"step": self._pose_log_step, "robot_base": self._pose7(self.robot),
               "objects": {inst: self._pose7(entity) for inst, entity in self._pose_log_entities.items()}}
        if self._pose_log_goal:
            row["goal_literals"] = [[bool(pred.evaluate(self.env.task._evaluate_predicate)) for pred in option]
                                    for option in self._pose_log_goal]
        toggles = getattr(self, "_pose_log_toggles", None)
        if toggles:
            try:  # plain attribute reads (no state update); evidence only, never fail an evaluation over it
                row["toggled_on"] = {inst: [bool(st.value), int(st.robot_can_toggle_steps)] for inst, st in toggles.items()}
            except Exception as error:
                row["toggled_on"] = f"unavailable: {type(error).__name__}"
        self._pose_log.write(json.dumps(row) + "\n")
        self._pose_log_step += 1
        if self._pose_log_step % 30 == 0:
            self._pose_log.flush()

    def stop_pose_log(self) -> None:
        if self._pose_log is not None:
            self._pose_log.write(json.dumps({"type": "end", "environment_steps": self._pose_log_step - 1}) + "\n")
            self._pose_log.close()
        self._pose_log = None
        self._pose_log_entities = {}
        self._pose_log_goal = []
        self._pose_log_toggles = {}

    @property
    def should_sync_lights(self) -> bool:
        return self.env.task.activity_name in LIGHT_EVAL_TASKS

    def _reset_light_synchronizer(self) -> None:
        if self.should_sync_lights:
            self.light_synchronizer = LightToggleSynchronizer(self.env.scene)
            self.light_synchronizer.reset_from_current_state()
        else:
            self.light_synchronizer = None

    def _sync_lights_and_get_obs(self, obs: dict | None = None) -> dict:
        if not self.should_sync_lights:
            return obs

        if self.light_synchronizer is None:
            self._reset_light_synchronizer()
        else:
            self.light_synchronizer.sync_from_current_state()
        for _ in range(3):
            og.sim.render()
        obs, _ = self.env.get_obs()
        return obs

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"

        self.human_stats = load_human_stats(task_name)

        task_cfg = available_tasks[task_name][0]
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            cfg["scene"]["load_room_instances"] = TASK_NAMES_TO_ROOMS[task_name]

        robot_cfg = self._build_robot_config(task_name=task_name, task_cfg=task_cfg)
        self.robot_name = robot_cfg["name"]
        cfg["robots"] = [robot_cfg]

        if self.cfg.max_steps is None:
            max_steps = int(self.human_stats["length"] * EVAL_TIMEOUT_MULTIPLIER)
            logger.info(
                f"Setting timeout to be {EVAL_TIMEOUT_MULTIPLIER}x the average length of human demos: {max_steps}"
            )
            cfg["task"]["termination_config"]["max_steps"] = max_steps
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False

        if self.cfg.get("write_video", False) and self.cfg.get("video_view", "robot") == "viewer":
            validate_viewer_config(self.cfg.video_width, self.cfg.video_height,
                                   self.cfg.video_camera_offset, self.cfg.video_camera_target)
            gm.RENDER_VIEWER_CAMERA = True
            cfg.setdefault("render", {}).update(viewer_width=self.cfg.video_width, viewer_height=self.cfg.video_height)

        env = og.Environment(configs=cfg)
        env._eval_robot_config = self.robot_eval_config
        return instantiate(env_wrapper, env=env)

    def load_robot(self) -> Robot:
        if self.robot_name is not None:
            return self.env.scene.object_registry("name", self.robot_name)
        return self.env.robots[0]

    def _validate_robot_eval_config(self) -> None:
        missing_camera_sensors = []
        for camera_id, camera_name in self.robot_camera_names.items():
            sensor_name = camera_name.split("::", 1)[1]
            if sensor_name not in self.robot.sensors:
                missing_camera_sensors.append(f"{camera_id}: {sensor_name}")
        if missing_camera_sensors:
            raise ValueError(
                "Configured eval.camera_sensor_names entries were not found in robot.sensors: "
                f"{missing_camera_sensors}"
            )

        if self.cfg.get("write_video", False) and self.cfg.get("video_view", "robot") == "robot":
            required_camera_ids = {"left_wrist", "right_wrist", "head"}
            missing_camera_ids = sorted(required_camera_ids - set(self.robot_camera_names))
            if missing_camera_ids:
                raise ValueError(
                    "--write-video requires eval.camera_sensor_names roles "
                    f"{sorted(required_camera_ids)}; missing {missing_camera_ids}"
                )

    def _build_robot_config(self, task_name: str, task_cfg: dict) -> dict:
        robot_cfg = _to_plain_dict(OmegaConf.select(self.cfg, "robot"))
        if robot_cfg is None:
            robot_cfg = _to_plain_dict(OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH))
        else:
            robot_cfg = dict(robot_cfg)
        assert "model" in robot_cfg, "Robot config must include canonical 'model'"
        assert "type" not in robot_cfg, "Robot config must use canonical 'model', not 'type'"
        robot_cfg["model"] = robot_cfg["model"].lower()
        assert "name" in robot_cfg, "Robot config must include 'name'"
        self.robot_eval_config = _to_plain_dict(robot_cfg.pop("eval", None)) or {}
        robot_cfg["position"] = task_cfg["robot_start_position"]
        robot_cfg["orientation"] = task_cfg["robot_start_orientation"]

        return robot_cfg

    def _apply_robot_eval_settings(self) -> None:
        if self.robot.model in ("r1", "r1pro"):
            og.sim.stop()
            self.robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
            og.sim.play()

        head_camera_name = self.robot_camera_names.get("head")
        if head_camera_name is not None:
            head_sensor_name = head_camera_name.split("::")[1]
            if head_sensor_name in self.robot.sensors:
                self.robot.sensors[head_sensor_name].horizontal_aperture = EVAL_HEAD_HORIZONTAL_APERTURE

    def load_policy(self) -> Any:
        policy = instantiate(self.cfg.model)
        if hasattr(policy, "set_action_dim"):
            policy.set_action_dim(self.robot.action_dim)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def policy_evidence(self) -> dict:
        """Read seed metadata and the live episode/step counter; never send task/object truth to the policy."""
        client = getattr(self.policy, "policy", None)
        metadata = client.get_server_metadata() if hasattr(client, "get_server_metadata") else {}
        metadata = metadata or {}
        evidence = {"effective_policy_seed": metadata.get("seed"),
                    "policy_rng_reset_behavior": metadata.get("rng_reset_behavior"),
                    "policy_episode": None, "policy_step": None}
        if metadata.get("policy") == "lilawam_behavior_task13":
            from omnigibson.eval.utils.network_utils import unpackb
            # ponytail: use the existing WAM status RPC; expose a public client method if other policies adopt it.
            # Same connection orders this query after reset; status does not reset, sample, or execute an action.
            client._ws.send(client._packer.pack({"status": True}))
            status = unpackb(client._ws.recv(timeout=10))
            if (not isinstance(status, dict) or type(status.get("episode")) is not int
                    or type(status.get("step")) is not int or status["episode"] < 0 or status["step"] < 0):
                raise ValueError("Invalid policy episode/step status")
            evidence.update(policy_episode=status["episode"], policy_step=status["step"])
        return evidence

    def load_metrics(self) -> List[MetricBase]:
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self) -> Tuple[bool, bool]:
        self.robot_action = self.policy.forward(obs=self.obs)
        obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
        obs = self._sync_lights_and_get_obs(obs)
        self.obs = self._preprocess_obs(obs)

        if self._video_path is not None:
            self._write_video()
        self._write_joint_log()
        self._write_pose_log()

        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        for metric in self.metrics:
            metric.step(self.env, self.robot_action, obs, 0.0, terminated, truncated, info)
        return terminated, truncated

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            container, stream = self._video_writer
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._video_writer = video_writer

    def load_task_instance(self, instance_id: int) -> None:
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        mode = self.cfg.get("mode", "public_test")
        tro_file_path = get_task_instance_path(
            scene_model,
            f"{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state",
            mode=mode,
        )
        if tro_file_path is None:
            raise FileNotFoundError(
                f"Could not find 2026 {mode} task instance {instance_id} for "
                f"{self.env.task.activity_name} in scene {scene_model}."
            )

        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = {key.lower(): value for key, value in tro_state.items()}
                if "robot" in presampled_robot_poses:
                    available_poses = presampled_robot_poses["robot"]
                elif self.robot.model in presampled_robot_poses:
                    logger.info("No generic presampled robot pose found, using robot-specific pose.")
                    available_poses = presampled_robot_poses[self.robot.model]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose found for {self.robot.model}!")
                self.robot.set_position_orientation(available_poses[0]["position"], available_poses[0]["orientation"])
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        if self.should_sync_lights:
            set_light_control_toggles(self.env.task.object_scope.values(), True)

        # Keep all task-relevant entities (including the robot/agent) still while the scene settles,
        # so the snapshotted per-instance initial state is stable. The robot must already be in a clean
        # configuration when this is called -- eval.py resets before load_task_instance for this reason.
        og.sim.update_handles()
        for _ in range(25):
            og.sim.step_physics()
            for inst, entity in self.env.task.object_scope.items():
                if not is_system_bddl_inst(inst) and entity is not None:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()
        self._reset_light_synchronizer()

    def _preprocess_obs(self, obs: dict) -> dict:
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in self.robot_camera_names.values():
            sensor_name = camera_name.split("::")[1]
            if sensor_name not in self.robot.sensors:
                continue
            camera = self.robot.sensors[sensor_name]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        if cam_rel_poses:
            obs[f"{self.robot.name}::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self) -> None:
        if self.cfg.get("video_view", "robot") == "viewer":
            camera = og.sim.viewer_camera
            base_position, base_orientation = self.robot.get_position_orientation()
            if self._framing is not None:
                subject, poi_names = self._viewer_subject()
                position, rotation, record = self._framing.update(base_position.detach().cpu().numpy(),
                                                                  T.quat2mat(base_orientation).detach().cpu().numpy(), subject)
                record["pois"] = poi_names
                self._framing_log.write(json.dumps(record) + "\n")
            else:
                position, rotation = viewer_pose(base_position.detach().cpu().numpy(),
                                                T.quat2mat(base_orientation).detach().cpu().numpy(),
                                                self.cfg.video_camera_offset, self.cfg.video_camera_target)
            camera.set_position_orientation(th.tensor(position, dtype=th.float32),
                                            T.mat2quat(th.tensor(rotation, dtype=th.float32)))
            og.sim.render()  # Render-only: do not advance physics or add this camera to self.obs.
            rgb = camera.get_obs()[0]["rgb"].detach().cpu().numpy()
            frame = native_rgb_frame(rgb, self.cfg.video_width, self.cfg.video_height)[None]
        else:
            frame = self._robot_video_frame()
        if self._video_writer is None:
            self.video_writer = create_video_writer(
                self._video_path, resolution=frame.shape[1:3], rate=self._video_rate,
                stream_options={"crf": "18", "preset": "fast", "g": str(self._video_rate)},
                container_options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
            )
        write_video(frame, video_writer=self.video_writer, batch_size=1, mode="rgb")
        self._video_frames += 1

    @staticmethod
    def _viewer_ray_blocked(start, end, ignore_prefixes) -> bool:
        """Read-only PhysX scene query for the recording camera; hits on ignored prim subtrees are see-through."""
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end - start))
        if distance < 1e-6:
            return False
        blocked = [False]

        def report(hit):
            path = hit.rigid_body or hit.collision or ""
            if any(path == p or path.startswith(p + "/") for p in ignore_prefixes):
                return True
            blocked[0] = True
            return False

        og.sim.psqi.raycast_all(origin=start.tolist(), dir=((end - start) / distance).tolist(),
                                distance=distance, reportFn=report)
        return blocked[0]

    def _viewer_points_of_interest(self) -> list:
        """Task-object AABBs (not agent/floors/systems) snapshotted at recording start, for framing only."""
        pois = []
        for inst, entity in self.env.task.object_scope.items():
            if entity is None or is_system_bddl_inst(inst) or inst.split(".")[0] in ("agent", "floor") or entity is self.robot:
                continue
            try:
                lo, hi = entity.aabb
                pois.append((inst, lo.detach().cpu().numpy().astype(float), hi.detach().cpu().numpy().astype(float),
                             entity.prim_path))
            except Exception as error:  # recording-only convenience; never fail an evaluation over framing
                logger.warning(f"Viewer framing: skipping point of interest {inst}: {error}")
        return pois

    def _viewer_subject(self) -> tuple:
        robot = self.robot
        base_position, base_orientation = robot.get_position_orientation()
        base = base_position.detach().cpu().numpy().astype(float)
        rotation = T.quat2mat(base_orientation).detach().cpu().numpy().astype(float)
        links = np.stack([link.get_position_orientation()[0].detach().cpu().numpy() for link in robot.links.values()]).astype(float)
        top = links[np.argmax(links[:, 2])]
        lower = base + np.array([0.0, 0.0, 0.25])
        forward, left = rotation[:, 0], rotation[:, 1]
        footprint = [base + 0.35 * sx * forward + 0.35 * sy * left + np.array([0.0, 0.0, 0.05])
                     for sx in (-1, 1) for sy in (-1, 1)]
        ignore = (robot.prim_path,)
        optional = []
        if robot.is_manipulation:
            optional += [(link.get_position_orientation()[0].detach().cpu().numpy().astype(float), ignore)
                         for link in robot.eef_links.values()]
        names = []
        for name, lo, hi, path in self._framing_pois:
            point = np.clip(base + np.array([0.0, 0.0, 1.0]), lo, hi)  # nearest AABB point at torso height (car rear)
            if np.linalg.norm((point - base)[:2]) <= 1.5:
                optional.append((point, ignore + (path,)))
                names.append(name)
        subject = {"required": np.stack([lower, 0.5 * (lower + top), top]),
                   "frustum": np.vstack([links, top + np.array([0.0, 0.0, 0.15]), np.stack(footprint)]),
                   "robot_ignore": ignore, "optional": optional}
        return subject, names

    def _robot_video_frame(self) -> np.ndarray:
        required_camera_ids = ("left_wrist", "right_wrist", "head")
        if not all(camera_id in self.robot_camera_names for camera_id in required_camera_ids):
            raise ValueError("Robot recording requires head and both wrist cameras")
        if self.robot_camera_names["head"] + "::rgb" not in self.obs:
            raise ValueError("Robot recording requires RGB observations")
        left_wrist_rgb = cv2.resize(
            self.obs[self.robot_camera_names["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[self.robot_camera_names["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[self.robot_camera_names["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        return np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0)

    def start_recording(self, fpath: str, rate: int = 30) -> None:
        if type(rate) is not int or rate <= 0:
            raise ValueError("Video rate must be a positive integer")
        self.stop_recording()
        if os.path.exists(fpath) or os.path.exists(fpath + ".json"):
            raise FileExistsError(f"Refusing to overwrite recording: {fpath}")
        self._video_path = fpath
        self._video_rate = rate
        self._video_frames = 0
        viewer = self.cfg.get("video_view", "robot") == "viewer"
        self._video_metadata = {"status": "recording", "view": "viewer" if viewer else "robot",
                                "fps": rate, "sim_step_seconds": og.sim.get_sim_step_dt(),
                                "frames": 0, "initial_frame_included": True,
                                "one_frame_per_environment_step": True, "policy_inputs_unchanged": True,
                                "frame_width": self.cfg.video_width if viewer else 672,
                                "frame_height": self.cfg.video_height if viewer else 448,
                                "native_high_resolution": viewer, "framing_visually_verified": False}
        if viewer:
            camera = og.sim.viewer_camera
            if camera is None:
                raise RuntimeError("Viewer recording requires gm.RENDER_VIEWER_CAMERA before scene creation")
            camera.horizontal_aperture = 28.0
            camera.focal_length = 17.0
            self._video_metadata.update(camera_offset=list(self.cfg.video_camera_offset),
                                        camera_target=list(self.cfg.video_camera_target),
                                        camera_coordinates="metres in robot yaw frame",
                                        horizontal_aperture=28.0, focal_length=17.0,
                                        privileged_data_use="robot base pose for recording camera only; never policy input")
            framing = self.cfg.get("video_framing", "auto")
            self._video_metadata["framing"] = framing
            if framing == "auto":
                camera_log = fpath + ".camera.jsonl"
                if os.path.exists(camera_log):
                    raise FileExistsError(f"Refusing to overwrite recording: {camera_log}")
                self._framing = ObserverFraming(self.cfg.video_camera_offset, self.cfg.video_camera_target,
                                                self.cfg.video_width, self.cfg.video_height, 28.0, 17.0,
                                                blocked=self._viewer_ray_blocked)
                self._framing_pois = self._viewer_points_of_interest()
                self._framing_log = open(camera_log, "w")
                with open(recording_utils.__file__, "rb") as file:
                    recording_sha = hashlib.sha256(file.read()).hexdigest()
                self._video_metadata.update(
                    framing_description=("occlusion-aware follow camera: deterministic robot-yaw-relative candidate grid "
                                         "around camera_offset; keep current while PhysX rays to robot base/torso/top are "
                                         "unobstructed and all robot links are in frame; periodic upgrade with hysteresis; "
                                         "world-frame smoothing with cuts on switch; nearby task objects (e.g. car rear) "
                                         "and end effectors only lower the cost"),
                    framing_candidates=[{"cost": round(c, 4), "offset": o.round(4).tolist()} for c, o in self._framing.candidates],
                    framing_parameters={"smoothing": self._framing.smoothing, "upgrade_period": self._framing.upgrade_period,
                                        "hold_after_switch": self._framing.hold_after_switch,
                                        "switch_margin": self._framing.switch_margin, "clearance": self._framing.clearance},
                    framing_points_of_interest=[name for name, *_ in self._framing_pois],
                    framing_camera_log=os.path.basename(camera_log),
                    recording_utils_sha256=recording_sha,
                    privileged_data_use=("robot link poses, task-object AABBs (snapshot at reset) and PhysX scene raycasts "
                                         "for the recording camera only; never policy input"))
        self._save_video_metadata()
        self._write_video()  # Include the reset state before the first policy action.

    def _save_video_metadata(self) -> None:
        with open(self._video_path + ".json", "w") as file:
            json.dump(self._video_metadata, file, indent=2)

    def stop_recording(self, completed: bool = False, success: bool | None = None, steps: int | None = None) -> None:
        self.video_writer = None
        if self._video_path is not None:
            self._video_metadata.update(status="completed" if completed else "interrupted",
                                        frames=self._video_frames, success=success, environment_steps=steps,
                                        complete_frame_coverage=completed and steps is not None and self._video_frames == steps + 1)
            if self._framing is not None:
                # Geometric check only (collision rays + frustum); visual-only geometry is not detected, so
                # framing_visually_verified stays False until a human/native-frame inspection.
                self._video_metadata["framing_stats"] = dict(self._framing.stats)
            self._save_video_metadata()
        if self._framing_log is not None:
            self._framing_log.close()
        self._framing, self._framing_log, self._framing_pois = None, None, []
        self._video_path = None

    def reset(self) -> None:
        obs = self.env.reset()[0]
        self._reset_light_synchronizer()
        obs = self._sync_lights_and_get_obs(obs)
        self.obs = self._preprocess_obs(obs)
        for metric in self.metrics:
            metric.reset(self.env)
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.stop_recording()
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


__all__ = ["Evaluator", "resolve_instance_ids"]
