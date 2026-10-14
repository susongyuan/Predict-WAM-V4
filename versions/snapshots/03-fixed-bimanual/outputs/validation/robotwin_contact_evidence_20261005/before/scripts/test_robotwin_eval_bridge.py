#!/usr/bin/env python3
"""CPU-only regression tests. Fakes exist ONLY in this test executable.

No real model, CUDA, Scene, renderer or download. Optional ffmpeg smoke writes
synthetic-prefixed media and cannot establish real-scene framing.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
from harness.robotwin_policy_worker import (Channel, ChunkPolicy, InfrastructureError, PolicyClient,
    VIEWS, check_reply, decode_observation, encode_observation, envelope)


def fake_worker(mode, fd):
    """Private test harness path; no production argument can select these fakes."""
    ch = Channel(socket.socket(fileno=fd))
    if mode == "startup_exit":
        return 3
    if mode == "startup_timeout":
        ch.recv(10)
        return 4
    ch.send(envelope("ready", "startup", 0, runtime={"synthetic_test_only": True}), 2)
    while True:
        message = ch.recv(10)
        if mode == "death":
            os._exit(7)
        if mode == "timeout":
            ch.recv(10)
            return 5
        episode, step = message["episode"], message["step"]
        if mode == "wrong_id":
            episode = "wrong_episode"
        if message["kind"] == "reset":
            ch.send(envelope("reset_ok", episode, step), 2)
        else:
            raw = [0.0] * (13 if mode == "bad_shape" else 14)
            ch.send(envelope("action", episode, step, raw_action=raw), 2)


if "--fake-worker" in sys.argv:
    p = argparse.ArgumentParser()
    p.add_argument("--fake-worker")
    p.add_argument("--ipc-fd", type=int, required=True)
    a = p.parse_args()
    raise SystemExit(fake_worker(a.fake_worker, a.ipc_fd))

import numpy as np
from harness.robotwin_qpos_adapter import policy_observation
from harness.robotwin_eval_evidence import (VideoRecorder, bound_action, empty_evidence, joint_bounds,
    settle_diagnostic)
from scripts import eval_robotwin_wam as evaluation

OUT = ROOT / "outputs/validation/robotwin_eval_bridge_20261005"


def observation():
    return {"rgb": {v: np.full((12, 16, 3), i + 10, np.uint8) for i, v in enumerate(VIEWS)},
            "commanded_joint_gripper_state": np.zeros(14)}


def native_observation():
    return {"observation": {v: {"rgb": im} for v, im in observation()["rgb"].items()},
            "joint_action": {"left_arm": [0.] * 6, "right_arm": [0.] * 6,
                             "left_gripper": 0., "right_gripper": 0.},
            "object_GT": {"secret": [9, 9]}, "score": 100, "third_view_rgb": "DO_NOT_SEND"}


class ProtocolTests(unittest.TestCase):
    def test_three_rgb_state_roundtrip_without_gt(self):
        original = native_observation()
        wire = encode_observation(policy_observation(original))
        text = json.dumps(wire)
        for token in ("object_GT", "secret", "score", "third_view", "DO_NOT_SEND"):
            self.assertNotIn(token, text)
        restored = decode_observation(json.loads(text))
        for v in VIEWS:
            np.testing.assert_array_equal(restored["rgb"][v], original["observation"][v]["rgb"])
        self.assertEqual(restored["commanded_joint_gripper_state"].shape, (14,))

    def test_reject_extra_keys_and_views(self):
        for key in ("score", "object_pose", "third_view_rgb"):
            x = observation()
            x[key] = 1
            with self.assertRaises(InfrastructureError):
                encode_observation(x)
        x = observation()
        x["rgb"]["observer_camera"] = x["rgb"][VIEWS[0]]
        with self.assertRaises(InfrastructureError):
            encode_observation(x)

    def test_bad_state_nan_shape(self):
        for state in ([float("nan")] * 14, [float("inf")] * 14, [0.] * 13, [[0.] * 14], [True] * 14):
            x = observation()
            x["commanded_joint_gripper_state"] = state
            with self.assertRaises(InfrastructureError):
                encode_observation(x)

    def test_bad_rgb(self):
        for image in (np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3)), np.zeros((0, 2, 3), np.uint8)):
            x = observation()
            x["rgb"][VIEWS[0]] = image
            with self.assertRaises(InfrastructureError):
                encode_observation(x)
        for corrupt in ({"shape": [999999, 999999, 3], "bytes": ""},
                        {"shape": [1, 1, 3], "bytes": "not!base64"},
                        {"shape": [1, 1, 3], "bytes": "AA=="}):
            wire = encode_observation(observation())
            wire["rgb"][VIEWS[0]] = corrupt
            with self.assertRaises(InfrastructureError):
                decode_observation(wire)

    def test_wrong_id(self):
        for reply in (envelope("action", "different", 0), envelope("action", "ep", 1),
                      envelope("reset_ok", "ep", 0)):
            with self.assertRaises(InfrastructureError):
                check_reply(reply, "action", "ep", 0)

    def test_bounded_frames_nan_and_deadline(self):
        for payload in (struct.pack("!I", 32 * 1024 * 1024 + 1),
                        struct.pack("!I", 9) + b'{"x":NaN}', struct.pack("!I", 100) + b"partial"):
            a, b = socket.socketpair()
            try:
                a.sendall(payload)
                started = time.monotonic()
                with self.assertRaises(InfrastructureError):
                    Channel(b).recv(0.05)
                self.assertLess(time.monotonic() - started, 1)
            finally:
                a.close()
                b.close()


class BoundsEvidenceTests(unittest.TestCase):
    def test_clips_physical_only_keeps_raw(self):
        raw = np.full(14, 3.0)
        raw[6], raw[13] = 1.2, -0.2
        limits = np.tile([-0.75, 0.8], (12, 1))
        record = bound_action(raw, limits)
        np.testing.assert_array_equal(record["raw_action"], raw)
        self.assertEqual(record["bounded_action"][6], 1)
        self.assertEqual(record["bounded_action"][13], 0)
        self.assertEqual(record["bounded_action"][0], 0.8)
        self.assertTrue(all(record["clip_mask"]))
        self.assertEqual(raw[6], 1.2)

    def test_invalid_action_never_reaches_clip(self):
        for raw in ([float("nan")] * 14, [float("inf")] * 14, [0.] * 13):
            with self.assertRaises(InfrastructureError):
                bound_action(raw, np.tile([-1, 1], (12, 1)))
        with self.assertRaises(InfrastructureError):
            bound_action([0.] * 14, np.tile([2, -2], (12, 1)))

    def test_bounds_from_robot_not_task(self):
        joints = [SimpleNamespace(get_limits=lambda: [[-2., 2.]], get_name=lambda: "native_joint") for _ in range(6)]
        bounds, records = joint_bounds(SimpleNamespace(left_arm_joints=joints, right_arm_joints=joints))
        self.assertEqual(bounds.shape, (12, 2))
        self.assertEqual(records[0]["joint"], "native_joint")
        joints[0] = SimpleNamespace(get_limits=lambda: [[-float("inf"), float("inf")]])
        with self.assertRaises(InfrastructureError):
            joint_bounds(SimpleNamespace(left_arm_joints=joints, right_arm_joints=joints))

    def test_missing_evidence_null(self):
        evidence = empty_evidence()
        for key in ("release_verified", "full_containment", "competition_success", "five_second_hold",
                    "scene_timestamp", "native_benchmark_success", "stable_after_settle"):
            self.assertIsNone(evidence[key])
        self.assertIsNone(settle_diagnostic([], 5, 0)["stable_after_settle"])
        self.assertIsNone(settle_diagnostic([None] * 6, 5, 5)["stable_after_settle"])
        poses = [[0, 0, 0, 1, 0, 0, 0]] * 3
        self.assertTrue(settle_diagnostic(poses, 2, 2)["stable_after_settle"])
        self.assertIsNone(settle_diagnostic(poses, 2, 2)["scene_seconds"])

    def test_native_set_gripper_not_a_stage_override(self):
        # Execute exactly the native method AST with fake drive joints (no native imports).
        source = ROOT / "third_party/RoboTwin/envs/robot/robot.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Robot")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_gripper")
        module = ast.Module(body=[method], type_ignores=[])
        ns = {"np": np, "sapien": SimpleNamespace(physx=SimpleNamespace(PhysxArticulationJoint=object))}
        exec(compile(module, str(source), "exec"), ns)
        joint = SimpleNamespace(drive_target=np.array([0.0]), set_drive_target=lambda x: setattr(joint, "target", x),
                                set_drive_velocity_target=lambda x: setattr(joint, "velocity", x))
        robot = SimpleNamespace(left_entity=None, right_entity=None, _entity_qf=lambda entity: None,
            left_gripper=[(joint, 1, 0)], left_gripper_scale=[0, 1], get_normal_real_gripper_val=lambda: [0, 0])
        ns["set_gripper"](robot, 1.3, "left")
        self.assertEqual(robot.left_gripper_val, 1.0)  # stored command is clipped
        self.assertAlmostEqual(joint.target, 0.1)     # drive target still native slew-limited
        self.assertNotEqual(robot.left_gripper_val, joint.target)


class HistoryTests(unittest.TestCase):
    def context(self):
        # Actual V3Context and shared history_frames, extracted to avoid torch import.
        tree = ast.parse((ROOT / "harness/wam_v3.py").read_text())
        selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and node.name in ("V3Context", "history_frames")]
        ns = {"np": np}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "wam_v3_context_cpu", "exec"), ns)
        return ns["V3Context"]([2, 4, 8, 16, 32], [4, 16], VIEWS, state_dim=14)

    def test_every_frame_encoded_while_queue_nonempty_and_reset(self):
        calls, samples = [], []
        ctx = self.context()
        def encode(rgb):
            calls.append(len(rgb))
            return None, np.full((3, 2, 4), len(calls), np.float32)
        def sample(feats, inputs, seed, step):
            samples.append((seed, step, inputs[0].copy()))
            return np.arange(32 * 14).reshape(32, 14)
        policy = ChunkPolicy(ctx, encode, sample)
        policy.reset("one", 10000)
        for step in range(10):
            obs = observation()
            obs["commanded_joint_gripper_state"][:] = step
            result = policy.observe("one", step, obs)
            self.assertEqual(result["history_steps"], step + 1)
        self.assertEqual(calls, [3] * 10)
        self.assertEqual([s[1] for s in samples], [0, 8])
        self.assertEqual(ctx.steps, 10)
        self.assertEqual(samples[1][2][0, 0], 8)
        self.assertEqual(samples[1][2][1, 0], 6)
        policy.reset("two", 10001)
        self.assertEqual(len(policy.queue), 0)
        self.assertEqual(ctx.steps, 0)
        with self.assertRaises(InfrastructureError):
            policy.observe("one", 0, observation())
        policy.observe("two", 0, observation())
        with self.assertRaises(InfrastructureError):
            policy.observe("two", 0, observation())

    def test_reject_wrong_chunk_shape_and_nan(self):
        for chunk in (np.zeros((8, 14)), np.full((32, 14), np.nan)):
            p = ChunkPolicy(self.context(), lambda rgb: (None, np.zeros((3, 2, 4))), lambda *args: chunk)
            p.reset("ep", 0)
            with self.assertRaises(InfrastructureError):
                p.observe("ep", 0, observation())


class ProcessTests(unittest.TestCase):
    def setUp(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=OUT, prefix="synthetic_process_")
        self.directory = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def client(self, mode):
        return PolicyClient([sys.executable, "-B", str(Path(__file__)), "--fake-worker", mode],
                            self.directory / f"{mode}.log", startup_timeout=1, timeout=0.15)

    def test_startup_missing_executable(self):
        with self.assertRaises(FileNotFoundError):
            PolicyClient(["/does/not/exist/python"], self.directory / "missing.log", .1, .1)

    def test_startup_exit_and_timeout_cleaned(self):
        original = subprocess.Popen
        for mode in ("startup_exit", "startup_timeout"):
            children = []
            def spawn(*args, **kwargs):
                p = original(*args, **kwargs)
                children.append(p)
                return p
            started = time.monotonic()
            with patch("harness.robotwin_policy_worker.subprocess.Popen", side_effect=spawn):
                with self.assertRaises(InfrastructureError):
                    self.client(mode)
            self.assertLess(time.monotonic() - started, 4)
            self.assertIsNotNone(children[0].poll())

    def test_runtime_death_timeout_wrong_id_cleanup(self):
        for mode in ("death", "timeout", "wrong_id"):
            client = self.client(mode)
            process = client.proc
            started = time.monotonic()
            with self.assertRaises(InfrastructureError):
                client.reset("ep", 10000)
            self.assertLess(time.monotonic() - started, 3)
            self.assertIsNotNone(process.poll())
            self.assertIsNone(client.proc)

    def test_bad_shape_closes_worker(self):
        client = self.client("bad_shape")
        process = client.proc
        client.reset("ep", 10000)
        with self.assertRaises(InfrastructureError):
            client.action("ep", 0, observation())
        self.assertIsNotNone(process.poll())

    def test_good_protocol_and_cleanup(self):
        client = self.client("good")
        process = client.proc
        client.reset("ep", 0)
        self.assertEqual(len(client.action("ep", 0, observation())["raw_action"]), 14)
        client.close()
        self.assertIsNotNone(process.poll())


class InjectedRobot:
    """Test-only native interface; no Scene construction."""
    def __init__(self):
        self.planner_mode = "joint_targets"
        self.left_arm_joints = [SimpleNamespace(get_limits=lambda: [[-1., 1.]], get_name=lambda: "joint") for _ in range(6)]
        self.right_arm_joints = list(self.left_arm_joints)
        self.left_mplib_planner = SimpleNamespace(TOPP=lambda: (None, np.zeros((1, 6))))
        self.right_mplib_planner = SimpleNamespace(TOPP=lambda: (None, np.zeros((0, 6))))
    def get_left_arm_jointState(self): return [0.] * 7
    def get_right_arm_jointState(self): return [0.] * 7
    def get_left_gripper_val(self): return 0.
    def get_right_gripper_val(self): return 0.
    def get_normal_real_gripper_val(self): return [0., 0.]


class InjectedEnv:
    def __init__(self):
        self.robot, self.closed, self.actions = InjectedRobot(), False, []
        self.take_action_cnt, self.step_lim, self.eval_success = 0, 3, False
        self.object = SimpleNamespace(get_pose=lambda: SimpleNamespace(p=np.zeros(3), q=np.array([1, 0, 0, 0])))
        self.cabinet = SimpleNamespace(get_qpos=lambda: np.zeros(1))
        self.scene = SimpleNamespace(step=lambda: None)
    def setup_demo(self, **kwargs): self.kwargs = kwargs
    def get_obs(self): return native_observation()
    def take_action(self, action, action_type):
        self.actions.append(action)
        self.take_action_cnt += 1
        self.robot.left_mplib_planner.TOPP()
        self.robot.right_mplib_planner.TOPP()
    def check_success(self): return self.eval_success
    def close_env(self): self.closed = True


class InjectedPolicy:
    def __init__(self, fail_step=None):
        self.ready, self.closed, self.fail_step = {"test_only": True}, False, fail_step
    def reset(self, episode, seed): self.episode = episode
    def check_alive(self): pass
    def action(self, episode, step, obs):
        if step == self.fail_step:
            raise InfrastructureError("synthetic worker runtime death")
        assert set(obs) == {"rgb", "commanded_joint_gripper_state"}
        return {"raw_action": [2.] * 14}
    def close(self): self.closed = True


class InjectedVideo:
    def capture(self, *args): pass
    def close(self, **kwargs): pass


class EntryTests(unittest.TestCase):
    def setUp(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=OUT, prefix="synthetic_entry_")
        self.directory = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def args(self, out=None):
        return evaluation.parse_args(["--run-dir", str(out or self.directory / "new"), "--checkpoint", "/unused/c.pt",
                                      "--training-run-dir", "/unused", "--plan-only"])

    def test_reject_overwrite(self):
        args = self.args(self.directory)
        marker = self.directory / "keep"
        marker.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            evaluation.main(["--run-dir", args.run_dir, "--checkpoint", args.checkpoint,
                             "--training-run-dir", args.training_run_dir, "--plan-only"])
        self.assertEqual(marker.read_text(), "unchanged")
        self.assertFalse((self.directory / "result.json").exists())

    def test_real_requires_lock(self):
        with self.assertRaises(SystemExit):
            evaluation.parse_args(["--run-dir", "x", "--checkpoint", "y", "--training-run-dir", "z"])
        with patch("scripts.gpu_lock.owner", return_value="training_owned"):
            with self.assertRaises(InfrastructureError):
                evaluation.verify_gpu_owner("eval_owned")

    def test_total_wall_budget_covers_policy_startup(self):
        args = self.args()
        args.total_wall_seconds = 0.03
        received = []
        def slow_start(startup_args, out):
            received.append(startup_args.startup_timeout)
            time.sleep(0.15)
            return InjectedPolicy()
        guard = lambda: SimpleNamespace(check=lambda: None, peaks={})
        began = time.monotonic()
        with patch.object(evaluation, "make_env", side_effect=AssertionError("Scene constructed")) as scene:
            result = evaluation.run_evaluation(args, {}, self.directory, client_factory=slow_start,
                env_factory=scene, owner_check=lambda owner: None, guard_factory=guard)
        self.assertLess(time.monotonic() - began, 0.12)
        self.assertTrue(received and 0 < received[0] <= args.total_wall_seconds)
        self.assertEqual(args.startup_timeout, 180)
        self.assertFalse(scene.called)
        self.assertEqual(result["status"], "infrastructure_error")
        self.assertEqual(result["episodes"][0]["failure_phase"], "policy_startup")
        self.assertEqual(result["not_started"], 2)
        self.assertIsNone(result["native_success_rate"])

    def test_startup_failure_never_constructs_scene_marks_remaining(self):
        args = self.args()
        def fail(*args):
            raise InfrastructureError("synthetic startup failure")
        with patch.object(evaluation, "make_env", side_effect=AssertionError("Scene constructed")) as scene:
            result = evaluation.run_evaluation(args, {}, self.directory, client_factory=fail,
                env_factory=scene, owner_check=lambda owner: None)
        self.assertFalse(scene.called)
        self.assertEqual(result["status"], "infrastructure_error")
        self.assertEqual([e["status"] for e in result["episodes"]], ["infrastructure_error", "not_started", "not_started"])
        self.assertIsNone(result["native_success_rate"])

    def test_dev_seed_overlap_rejected(self):
        args = self.args()
        (self.directory / "seed.txt").write_text("74 10000 999")
        cache = {"source": {"episodes": [{"source": str(self.directory / "data/episode0.hdf5")}]}}
        with patch.object(evaluation, "training_contract", return_value=({}, cache, {}, {})), \
             patch.object(evaluation, "load_task_config", return_value=({}, [])):
            with self.assertRaisesRegex(InfrastructureError, "overlap"):
                evaluation.plan_evaluation(args)

    def test_wall_timeout_is_not_success(self):
        args = self.args()
        policy, env = InjectedPolicy(), InjectedEnv()
        def timeout(action, action_type):
            raise evaluation.WallTimeout("synthetic native wall timeout")
        env.take_action = timeout
        result = evaluation.run_evaluation(args, {"task_kwargs": {}}, self.directory,
            client_factory=lambda *a: policy, env_factory=lambda: env,
            video_factory=lambda *a: (InjectedVideo(), None), owner_check=lambda x: None)
        self.assertEqual(result["episodes"][0]["status"], "wall_timeout")
        self.assertIsNone(result["episodes"][0]["evidence"]["native_benchmark_success"])
        self.assertEqual(result["not_started"], 2)
        self.assertTrue(policy.closed and env.closed)

    def test_native_loop_bounds_fallback_and_no_success_invention(self):
        args = self.args()
        args.seeds, args.max_steps, args.settle_steps = [10000], 2, 2
        policy, env = InjectedPolicy(), InjectedEnv()
        result = evaluation.run_evaluation(args, {"task_kwargs": {}}, self.directory,
            client_factory=lambda *a: policy, env_factory=lambda: env,
            video_factory=lambda *a: (InjectedVideo(), None), owner_check=lambda x: None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["native_successes"], 0)
        self.assertEqual(result["native_success_rate"], 0)
        row = result["episodes"][0]
        self.assertEqual(row["status"], "step_limit")
        self.assertEqual(row["control_steps"], 2)
        self.assertIsNone(row["evidence"]["full_containment"])
        self.assertIsNone(row["evidence"]["release_verified"])
        self.assertTrue(row["evidence"]["stable_after_settle"])
        self.assertTrue(policy.closed and env.closed)
        lines = [json.loads(x) for x in (self.directory / row["episode"] / "actions.jsonl").read_text().splitlines()]
        self.assertEqual(lines[0]["raw_action"], [2.] * 14)
        self.assertEqual(lines[0]["bounded_action"], [1.] * 14)
        self.assertTrue(lines[1]["topp_fallback_observed"])

    def test_already_satisfied_initial_goal_has_no_policy_credit(self):
        args = self.args()
        policy, env = InjectedPolicy(), InjectedEnv()
        env.check_success = lambda: True
        result = evaluation.run_evaluation(args, {"task_kwargs": {}}, self.directory,
            client_factory=lambda *a: policy, env_factory=lambda: env,
            video_factory=lambda *a: (InjectedVideo(), None), owner_check=lambda x: None)
        self.assertEqual(result["status"], "infrastructure_error")
        self.assertEqual(result["episodes"][0]["control_steps"], 0)
        self.assertIsNone(result["episodes"][0]["evidence"]["native_benchmark_success"])
        self.assertTrue(policy.closed and env.closed)

    def test_latched_goal_is_separate_from_goal_after_settling(self):
        args = self.args()
        args.seeds, args.max_steps, args.settle_steps = [10000], 2, 2
        policy, env = InjectedPolicy(), InjectedEnv()
        original = env.take_action
        def action(*a, **k):
            original(*a, **k)
            env.eval_success = True
        env.take_action = action
        env.check_success = unittest.mock.Mock(side_effect=[False, True, False])
        result = evaluation.run_evaluation(args, {"task_kwargs": {}}, self.directory,
            client_factory=lambda *a: policy, env_factory=lambda: env,
            video_factory=lambda *a: (InjectedVideo(), None), owner_check=lambda x: None)
        evidence = result["episodes"][0]["evidence"]
        self.assertTrue(evidence["native_benchmark_success"])
        self.assertTrue(evidence["native_terminal_predicate"])
        self.assertFalse(evidence["native_predicate_after_settle"])
        self.assertIsNone(evidence["full_containment"])

    def test_runtime_failure_stops_queue_closes_scene(self):
        args = self.args()
        policy, env = InjectedPolicy(fail_step=1), InjectedEnv()
        result = evaluation.run_evaluation(args, {"task_kwargs": {}}, self.directory,
            client_factory=lambda *a: policy, env_factory=lambda: env,
            video_factory=lambda *a: (InjectedVideo(), None), owner_check=lambda x: None)
        self.assertEqual(result["status"], "infrastructure_error")
        self.assertEqual([x["status"] for x in result["episodes"]], ["infrastructure_error", "not_started", "not_started"])
        self.assertEqual(result["episodes"][0]["control_steps"], 1)
        self.assertIsNone(result["episodes"][0]["evidence"]["native_benchmark_success"])
        self.assertTrue(policy.closed and env.closed)

    def test_training_preprocess_cross_process(self):
        import hashlib
        payload = encode_observation(policy_observation(native_observation()))
        code = """
import hashlib, json, sys, numpy as np, cv2
from harness.robotwin_policy_worker import decode_observation, VIEWS
from harness.robotwin_training_data import preprocess_rgb
obs = decode_observation(json.loads(sys.stdin.read()))
rows = []
for view in VIEWS:
 raw = obs['rgb'][view]
 px = preprocess_rgb(raw)
 rows.append({'raw_sha256': hashlib.sha256(raw.tobytes()).hexdigest(), 'shape': list(px.shape),
              'dtype': str(px.dtype), 'preprocess_sha256': hashlib.sha256(px.tobytes()).hexdigest()})
print(json.dumps({'numpy': np.__version__, 'cv2': cv2.__version__, 'rows': rows}))
"""
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        child = subprocess.run([str(ROOT / ".venvs/wam-fixed/bin/python"), "-B", "-c", code],
            input=json.dumps(payload), capture_output=True, text=True, cwd=ROOT, env=env, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        result = json.loads(child.stdout)
        self.assertEqual(result["numpy"], "2.2.6")
        self.assertEqual(result["cv2"], "4.11.0")
        for row, view in zip(result["rows"], VIEWS):
            self.assertEqual(row["raw_sha256"], hashlib.sha256(observation()["rgb"][view].tobytes()).hexdigest())
            self.assertEqual(row["shape"], [3, 224, 224])
            self.assertEqual(row["dtype"], "float32")
        from harness.robotwin_policy_worker import atomic_json
        atomic_json(OUT / f"cross_abi_parent_numpy_{np.__version__}.json", {"synthetic": True,
            "parent_numpy": np.__version__, "child": result, "model_loaded": False, "cuda_visible_devices": ""})

    def test_plan_import_is_lazy_no_torch_scene_cuda(self):
        code = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch', 'sapien', 'envs', 'cv2', 'numpy'):
   raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Block())
from scripts import eval_robotwin_wam
assert not any(n in sys.modules for n in ('torch','sapien','envs','cv2','numpy'))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
