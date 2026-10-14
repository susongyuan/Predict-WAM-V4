#!/usr/bin/env python3
"""CPU/stub interface tests: no real SAPIEN, CuRobo, Torch, renderer or IK.

Run with .venvs/robotwin-sim/bin/python scripts/test_robotwin_qpos_adapter.py.
Native take_action is AST-extracted unchanged and runs against counting stubs;
these tests assert control flow, not physics or task success.
"""
import ast
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import types
from typing import Literal
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.robotwin_qpos_adapter import (
    InvalidAction, NativeQposAdapter, RGB_VIEWS, policy_observation, validate_action,
)

ROBOT_DIR = ROOT / "third_party/RoboTwin/envs/robot"
BASE_TASK = ROOT / "third_party/RoboTwin/envs/_base_task.py"


def module(name, **attrs):
    value = types.ModuleType(name)
    value.__dict__.update(attrs)
    return value


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@contextmanager
def robot_modules():
    """Block heavy dependencies before executing production source imports."""
    constructors = []
    worlds = []

    class Mplib:
        def __init__(self, *args, **kwargs):
            constructors.append((args, kwargs))
            self.TOPP = Mock(name="native_TOPP")
            self.set_base_pose = Mock()

        def plan_pose(self, *args, **kwargs):
            raise AssertionError("MPlib IK must never execute")

        plan_screw = plan_pose

    def world(*args):
        worlds.append(args)
        return ("world", args)

    mp = module("torch.multiprocessing", Pipe=Mock(side_effect=AssertionError("unexpected IPC")),
                Process=Mock(side_effect=AssertionError("unexpected worker")))
    core = module("sapien.core")
    mplib = module("mplib", Planner=Mplib)
    mplib.planner = module("mplib.planner")
    mods = {
        "mplib": mplib, "mplib.planner": mplib.planner,
        "mplib.sapien_utils": module("mplib.sapien_utils", SapienPlanner=Mplib, SapienPlanningWorld=world),
        "sapien": module("sapien", core=core), "sapien.core": core,
        "toppra": module("toppra", setup_logging=Mock()),
        "transforms3d": module("transforms3d"), "yaml": module("yaml"),
        "envs": module("envs", __path__=[]), "envs.robot": module("envs.robot", __path__=[]),
        "envs._GLOBAL_CONFIGS": module("envs._GLOBAL_CONFIGS", ROOT_PATH="/stub", ROTATE_NUM=10),
        "envs.utils": module("envs.utils", transforms=types.SimpleNamespace()),
        "torch": module("torch", multiprocessing=mp), "torch.multiprocessing": mp,
        "curobo": None,  # force optional-import failure independent of installed libraries
    }
    with patch.dict(sys.modules, mods):
        planner = load("envs.robot.planner", ROBOT_DIR / "planner.py")
        robot = load("envs.robot.robot", ROBOT_DIR / "robot.py")
        yield planner, robot, constructors, worlds, mp


def bare_robot(cls, need_topp=True, distinct_configs=False):
    robot = cls.__new__(cls)
    robot.need_topp = need_topp
    robot.is_dual_arm = False
    for arm in ("left", "right"):
        fields = dict(urdf_path=f"{arm}.urdf", srdf_path=f"{arm}.srdf", move_group=arm,
                      entity_origion_pose=object(), planner_type="mplib_RRT",
                      arm_joints_name=[f"{arm}{n}" for n in range(6)],
                      curobo_yml_path=f"{arm if distinct_configs else 'same'}/curobo.yml")
        for name, value in fields.items():
            setattr(robot, f"{arm}_{name}", value)
        name = types.SimpleNamespace(get_name=lambda: "joint")
        entity = types.SimpleNamespace(get_links=lambda: [name], get_active_joints=lambda: [name])
        setattr(robot, f"{arm}_entity", entity)
    return robot


class PlannerTests(unittest.TestCase):
    def test_missing_dependency_importable_but_full_fails_before_ipc(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            self.assertIsNotNone(p._CUROBO_IMPORT_ERROR)
            for distinct in (False, True):
                bot = bare_robot(r.Robot, distinct_configs=distinct)
                with self.assertRaisesRegex(RuntimeError, "full.*requires working CuRobo") as err:
                    bot.set_planner()
                self.assertIs(err.exception.__cause__, p._CUROBO_IMPORT_ERROR)
                self.assertEqual(bot.planner_mode, "full")
            with self.assertRaisesRegex(RuntimeError, "requires working CuRobo"):
                p.CuroboPlanner()
            self.assertFalse(calls)
            mp.Pipe.assert_not_called()
            mp.Process.assert_not_called()

    def test_default_full_preserves_curobo_and_mplib_arguments(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            fake_curobo = Mock()
            with patch.object(p, "_CUROBO_IMPORT_ERROR", None), patch.object(r, "CuroboPlanner", fake_curobo):
                bot = bare_robot(r.Robot)
                bot.set_planner()
                self.assertEqual(bot.planner_mode, "full")
                self.assertFalse(bot.communication_flag)
                self.assertEqual(fake_curobo.call_count, 2)
                self.assertEqual(len(calls), 2)
                for arm, call, (_, kwargs) in zip(("left", "right"), fake_curobo.call_args_list, calls):
                    self.assertEqual(call.args, (getattr(bot, arm+"_entity_origion_pose"),
                                                getattr(bot, arm+"_arm_joints_name"), ["joint"]))
                    self.assertEqual(call.kwargs["yml_path"], "/stub/same/curobo.yml")
                    self.assertEqual(kwargs, dict(urdf=arm+".urdf", srdf=arm+".srdf", move_group=arm,
                                                 user_link_names=["joint"], user_joint_names=["joint"],
                                                 use_convex=False))
                bot = bare_robot(r.Robot, need_topp=False)
                bot.set_planner()
                self.assertFalse(hasattr(bot, "left_mplib_planner"))
                self.assertEqual(len(calls), 2)

    def test_full_distinct_configs_keep_worker_branch(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            mp.Pipe.side_effect = [(Mock(), Mock()), (Mock(), Mock())]
            workers = [Mock(), Mock()]
            mp.Process.side_effect = workers
            with patch.object(p, "_CUROBO_IMPORT_ERROR", None), patch.object(r, "CuroboPlanner", Mock()) as curobo:
                bot = bare_robot(r.Robot, distinct_configs=True)
                bot.set_planner()
                self.assertTrue(bot.communication_flag)
                self.assertEqual(mp.Process.call_count, 2)
                self.assertEqual(len(calls), 2)
                for worker in workers:
                    worker.start.assert_called_once()
                    self.assertTrue(worker.daemon)
                curobo.assert_not_called()

    def test_joint_mode_reuses_mplib_and_never_creates_curobo(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            for scene in (None, object()):  # opaque sentinel, never a real Scene
                bot = bare_robot(r.Robot, need_topp=False, distinct_configs=True)
                with patch.object(r, "CuroboPlanner", Mock(side_effect=AssertionError("CuRobo"))):
                    bot.set_planner(scene, planner_mode="joint_targets")
                self.assertFalse(bot.communication_flag)
                self.assertIs(bot.left_planner, bot.left_mplib_planner)
                self.assertIsInstance(bot.left_planner, p.JointTargetsPlanner)
                self.assertFalse(hasattr(bot.left_planner, "planner"))
                self.assertIsInstance(bot.left_planner.TOPP, Mock)
                for arm in ("left", "right"):
                    result = getattr(bot, arm+"_plan_grippers")(0.2, 0.9)
                    self.assertEqual(result["num_step"], 200)
                    self.assertEqual(result["per_step"], (0.9-0.2)/200)
                    np.testing.assert_array_equal(result["result"], np.linspace(0.2, 0.9, 200))
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(worlds), 2)
            self.assertEqual(calls[-2][0][1], "left")
            self.assertEqual(calls[-1][0][1], "right")
            mp.Pipe.assert_not_called()
            mp.Process.assert_not_called()

    def test_joint_mode_rejects_cartesian_before_input_access(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            bot = bare_robot(r.Robot)
            bot.set_planner(planner_mode="joint_targets")
            for name in ("left_plan_path", "right_plan_path", "left_plan_multi_path", "right_plan_multi_path",
                         "update_world_pcd", "get_constraint_pose"):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "forbids Cartesian"):
                    getattr(bot, name)(object())
            with self.assertRaisesRegex(RuntimeError, "forbids Cartesian"):
                bot.create_target_pose_list(object(), object())
            for name in ("plan_path", "plan_pose", "plan_screw", "plan_batch", "update_point_cloud"):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "forbids Cartesian"):
                    getattr(bot.left_mplib_planner, name)(object())

    def test_mode_validation_and_immutability_precede_loading(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            bot = bare_robot(r.Robot)
            with self.assertRaisesRegex(ValueError, "Unknown planner_mode"):
                bot._init_robot_(None, planner_mode="auto")
            bot.set_planner(planner_mode="joint_targets")
            with self.assertRaisesRegex(ValueError, "Cannot change planner_mode"):
                bot._init_robot_(None, planner_mode="full")
            with self.assertRaisesRegex(ValueError, "Cannot change planner_mode"):
                bot.set_planner(planner_mode="full")

    def test_reset_rebuilds_joint_topp_without_curobo(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            bot = bare_robot(r.Robot)
            bot.set_planner(planner_mode="joint_targets")
            previous = bot.left_mplib_planner
            bot._init_robot_ = Mock()
            bot.init_joints = Mock()
            bot.reset(None, need_topp=True, planner_mode="joint_targets")
            self.assertIsNot(bot.left_mplib_planner, previous)
            self.assertEqual(len(calls), 4)
            self.assertEqual(bot.planner_mode, "joint_targets")
            bot.init_joints.assert_called_once()
            mp.Pipe.assert_not_called()

    def test_native_commanded_state_is_not_measured_qpos(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            bot = bare_robot(r.Robot)
            for arm, offset in (("left", 0), ("right", 10)):
                joints = [types.SimpleNamespace(get_drive_target=lambda n=n: np.array([n+offset], dtype=float))
                          for n in range(6)]
                setattr(bot, arm+"_arm_joints", joints)
                setattr(bot, arm+"_gripper", [])
                setattr(bot, arm+"_gripper_val", 0.5)
                # bare entities intentionally have no get_qpos implementation.
                self.assertEqual(getattr(bot, "get_"+arm+"_arm_jointState")(),
                                 list(np.arange(6)+offset)+[0.5])


    def test_constructor_kwargs_select_mode_without_real_scene(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            r.sapien.Pose = lambda pos, quat: types.SimpleNamespace(p=np.array(pos, dtype=float), q=np.array(quat))
            entity = types.SimpleNamespace(set_root_pose=Mock())
            loader = types.SimpleNamespace(load=Mock(return_value=entity))
            scene = types.SimpleNamespace(create_urdf_loader=Mock(return_value=loader))
            config = dict(urdf_path="stub.urdf", move_group=["left", "right"],
                          ee_joints=["left_ee", "right_ee"], arm_joints_name=[["j"]*6]*2,
                          gripper_name=[{}, {}], gripper_bias=0., gripper_scale=[0., 1.],
                          homestate=[[0.]*6]*2)
            for mode in (None, "joint_targets"):
                kwargs = dict(left_embodiment_config=config, right_embodiment_config=config,
                              left_robot_file="/stub", right_robot_file="/stub", dual_arm_embodied=True)
                if mode is not None:
                    kwargs["planner_mode"] = mode
                bot = r.Robot(scene, True, **kwargs)
                self.assertEqual(bot.planner_mode, mode or "full")
            self.assertFalse(calls)  # constructor loads stub entities, never a planner
            mp.Process.assert_not_called()

    def test_native_ee_request_rejected_by_robot_mode(self):
        with robot_modules() as (p, r, calls, worlds, mp):
            env, topp_calls = fake_env()
            for name in ("_require_cartesian_planner", "left_plan_path", "right_plan_path"):
                setattr(env.robot, name, types.MethodType(getattr(r.Robot, name), env.robot))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "forbids Cartesian"):
                env.take_action(np.zeros(16), action_type="ee")
            self.assertFalse(topp_calls)
            env.robot.set_arm_joints.assert_not_called()
            env.robot.set_gripper.assert_not_called()
            env.scene.step.assert_not_called()


def valid_action():
    return np.array([.1, .2, .3, .4, .5, .6, .25, -.1, -.2, -.3, -.4, -.5, -.6, .75], dtype=np.float32)


class ValidationTests(unittest.TestCase):
    def test_valid_action_preserves_dtype_bytes_and_input(self):
        action = valid_action()
        result = validate_action(action)
        self.assertEqual(result.dtype, action.dtype)
        self.assertEqual(result.tobytes(), action.tobytes())
        self.assertFalse(np.shares_memory(result, action))

    def test_invalid_actions_fail_closed_and_retain_original(self):
        cases = [np.zeros(13), np.zeros((1, 14)), np.zeros(15), np.full(14, np.nan),
                 np.full(14, np.inf), np.zeros(14, dtype=complex), ["0"]*14, [True]*14]
        for index, value in ((6, -0.01), (13, 1.01), (0, float("-inf"))):
            case = valid_action(); case[index] = value; cases.append(case)
        for action in cases:
            original = np.array(action, copy=True)
            with self.subTest(shape=original.shape, dtype=original.dtype), self.assertRaises(InvalidAction) as err:
                validate_action(action)
            self.assertIs(err.exception.original, action)
            np.testing.assert_array_equal(action, original)

    def test_external_limits_and_boundaries(self):
        action = valid_action()
        limits = np.tile([-1., 1.], (12, 1))
        validate_action(action, limits)
        action[6], action[13] = 0, 1
        validate_action(action, limits)
        action[12] = 1.01
        with self.assertRaisesRegex(InvalidAction, "exceed"):
            validate_action(action, limits)
        validate_action(action)  # no hidden joint limits
        for bad in (np.zeros((14, 2)), np.tile([1., -1.], (12, 1)), np.full((12, 2), np.nan)):
            with self.assertRaises(InvalidAction):
                validate_action(valid_action(), bad)

    def raw_obs(self):
        return {"observation": {name: {"rgb": np.full((2, 3, 3), i, dtype=np.uint8)}
                                for i, name in enumerate(RGB_VIEWS)},
                "joint_action": {"left_arm": np.arange(6), "left_gripper": 0.25,
                                 "right_arm": np.arange(10, 16), "right_gripper": 0.75}}

    def test_policy_allowlist_and_two_arm_order(self):
        class OnlyKeys(dict):
            def __getitem__(self, key):
                if key not in self.allowed:
                    raise AssertionError("Forbidden read: "+key)
                return super().__getitem__(key)
        raw = self.raw_obs()
        raw.update(object_pose=object(), reward=object(), GT=object(), endpose=object())
        raw["observation"].update(observer_camera=object(), unknown_view=object())
        raw = OnlyKeys(raw); raw.allowed = {"observation", "joint_action"}
        raw["observation"] = OnlyKeys(raw["observation"])
        raw["observation"].allowed = set(RGB_VIEWS)
        raw["joint_action"] = OnlyKeys(raw["joint_action"])
        raw["joint_action"].allowed = {"left_arm", "left_gripper", "right_arm", "right_gripper"}
        output = policy_observation(raw)
        self.assertEqual(set(output), {"rgb", "commanded_joint_gripper_state"})
        self.assertEqual(tuple(output["rgb"]), RGB_VIEWS)
        np.testing.assert_array_equal(output["commanded_joint_gripper_state"],
                                      [0, 1, 2, 3, 4, 5, .25, 10, 11, 12, 13, 14, 15, .75])
        output["rgb"]["head_camera"][:] = 255
        self.assertTrue((raw["observation"]["head_camera"]["rgb"] == 0).all())

    def test_observation_rejects_bad_dims_nan_and_missing_view(self):
        for field, value in (("left_arm", np.zeros(7)), ("right_arm", np.full(6, np.nan)),
                             ("right_gripper", [0.5]), ("left_gripper", np.nan)):
            raw = self.raw_obs(); raw["joint_action"][field] = value
            with self.assertRaises(ValueError):
                policy_observation(raw)
        for image in (np.zeros((2, 3, 4), dtype=np.uint8), np.full((2, 3, 3), np.nan),
                      np.zeros((0, 3, 3), dtype=np.uint8)):
            raw = self.raw_obs(); raw["observation"]["head_camera"]["rgb"] = image
            with self.assertRaises(ValueError):
                policy_observation(raw)
        raw = self.raw_obs(); del raw["observation"]["right_camera"]
        raw["observation"]["unknown_view"] = {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)}
        with self.assertRaises(KeyError):
            policy_observation(raw)


def native_take_action():
    tree = ast.parse(BASE_TASK.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Base_Task")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "take_action")
    ns = {"np": np, "Literal": Literal}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(BASE_TASK), "exec"), ns)
    return ns["take_action"]


def fake_env(outcomes=("ok", "ok")):
    env = types.SimpleNamespace(take_action_cnt=0, step_lim=10, eval_success=False,
                                eval_video_path=None, render_freq=0)
    env.scene = types.SimpleNamespace(step=Mock())  # counter only; no physics
    env._update_render = Mock()  # counter only; no renderer
    env.check_success = Mock(return_value=False)
    env.robot = types.SimpleNamespace(planner_mode="joint_targets",
                                     get_left_arm_jointState=lambda: [0.]*6+[0.],
                                     get_right_arm_jointState=lambda: [0.]*6+[0.],
                                     get_left_gripper_val=lambda: 0., get_right_gripper_val=lambda: 0.,
                                     set_arm_joints=Mock(), set_gripper=Mock())
    topp_calls = []

    def topp(arm, outcome):
        def call(path, dt, verbose):
            topp_calls.append((arm, path.copy(), dt, verbose))
            if outcome == "exception":
                raise ValueError("injected native TOPP failure")
            pos = np.empty((0, 6)) if outcome == "empty" else np.repeat(path[-1:], 2, axis=0)
            return np.arange(len(pos)), pos, np.zeros_like(pos), np.zeros_like(pos), .008
        return call

    for arm, outcome in zip(("left", "right"), outcomes):
        setattr(env.robot, arm+"_mplib_planner", types.SimpleNamespace(TOPP=topp(arm, outcome)))
    env.take_action = types.MethodType(native_take_action(), env)
    return env, topp_calls


class NativeBoundaryTests(unittest.TestCase):
    def test_native_topp_receives_original_joint_targets_and_timing(self):
        env, calls = fake_env()
        adapter = NativeQposAdapter(env)
        action = valid_action(); before = action.copy()
        originals = [env.robot.left_mplib_planner.TOPP, env.robot.right_mplib_planner.TOPP]
        with redirect_stdout(io.StringIO()):
            adapter.step(action)
        self.assertEqual(len(calls), 2)
        for (arm, path, dt, verbose), expected in zip(calls, (action[:6], action[7:13])):
            np.testing.assert_array_equal(path[1], expected)
            self.assertEqual(dt, 1/250); self.assertIs(verbose, True)
        np.testing.assert_array_equal(action, before)
        self.assertEqual([x["status"] for x in adapter.last_topp_diagnostics], ["returned", "returned"])
        self.assertEqual(env.robot.set_arm_joints.call_count, 4)
        self.assertIs(env.robot.left_mplib_planner.TOPP, originals[0])
        self.assertIs(env.robot.right_mplib_planner.TOPP, originals[1])
        grippers = env.robot.set_gripper.call_args_list
        self.assertEqual(grippers[-2].args, (action[6], "left"))
        self.assertEqual(grippers[-1].args, (action[13], "right"))

    def test_native_hold_plus_gripper_is_observable_not_changed(self):
        for outcomes in (("exception", "empty"), ("empty", "ok"), ("ok", "exception")):
            env, calls = fake_env(outcomes)
            adapter = NativeQposAdapter(env)
            with redirect_stdout(io.StringIO()):
                adapter.step(valid_action())
            self.assertEqual([x["status"] for x in adapter.last_topp_diagnostics],
                             ["returned" if x == "ok" else x for x in outcomes])
            arm_calls = env.robot.set_arm_joints.call_args_list
            grip_calls = env.robot.set_gripper.call_args_list
            for arm, outcome in zip(("left", "right"), outcomes):
                self.assertEqual(sum(c.args[2] == arm for c in arm_calls), 2 if outcome == "ok" else 0)
                self.assertEqual(sum(c.args[1] == arm for c in grip_calls), 2 if outcome == "ok" else 50)
            if "exception" in outcomes:
                record = adapter.last_topp_diagnostics[outcomes.index("exception")]
                self.assertEqual(record["error_type"], "ValueError")

    def test_invalid_action_never_calls_native_or_topp(self):
        env, calls = fake_env()
        adapter = NativeQposAdapter(env)
        action = valid_action(); action[13] = 2
        with self.assertRaises(InvalidAction):
            adapter.step(action)
        self.assertFalse(calls)
        self.assertEqual(action[13], 2)
        self.assertEqual(adapter.last_topp_diagnostics, [])
        env._update_render.assert_not_called(); env.scene.step.assert_not_called()
        env.robot.set_gripper.assert_not_called(); env.robot.set_arm_joints.assert_not_called()

    def test_wrong_embodiment_is_rejected_before_native(self):
        env, calls = fake_env()
        env.robot.get_right_arm_jointState = lambda: [0.]*8
        with self.assertRaises(ValueError):
            NativeQposAdapter(env).step(valid_action())
        self.assertFalse(calls)
        env.scene.step.assert_not_called()

    def test_requires_explicit_joint_mode(self):
        env, _ = fake_env(); env.robot.planner_mode = "full"
        with self.assertRaisesRegex(ValueError, "explicit planner_mode"):
            NativeQposAdapter(env)

    def test_terminal_native_has_no_false_topp_success(self):
        env, calls = fake_env(); env.eval_success = True
        adapter = NativeQposAdapter(env)
        adapter.step(valid_action())
        self.assertFalse(calls)
        self.assertEqual(adapter.last_topp_diagnostics, [])

    def test_topp_restored_if_native_raises(self):
        env, calls = fake_env()
        original = env.robot.left_mplib_planner.TOPP
        env.take_action = Mock(side_effect=RuntimeError("outside TOPP"))
        adapter = NativeQposAdapter(env)
        with self.assertRaisesRegex(RuntimeError, "outside TOPP"):
            adapter.step(valid_action())
        self.assertIs(env.robot.left_mplib_planner.TOPP, original)


if __name__ == "__main__":
    print("CPU-only stub tests; NumPy="+np.__version__+"; no simulator/model execution")
    unittest.main(verbosity=2)
