#!/usr/bin/env python3
"""CPU-only capture/lifecycle/transport tests; no model or simulator imports."""
import ast
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.onboard_recording import OnboardRecorder, SENSORS, recording_config, recording_manifest

POLICY = ROOT / "harness/policy_server.py"


def policy_method(name):
    tree = ast.parse(POLICY.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WamPolicy")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    namespace = {"np": np, "time": __import__("time"), "re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(POLICY), "exec"), namespace)
    return namespace[name]


def sensors():
    return {key: (np.arange(80, dtype=np.uint8).reshape(4, 5, 4) if name.endswith("rgb")
                  else np.arange(20, dtype=np.float32).reshape(4, 5, 1) if "depth" in name
                  else np.arange(61 if name == "proprio" else 21, dtype=np.float64))
            for name, key in SENSORS.items()}


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "capture"
        self.emit = Mock()
        self.recorder = OnboardRecorder(self.directory, emit=self.emit)

    def record(self, step=0, episode=1, obs=None, **kw):
        self.recorder.record(sensors() if obs is None else obs, episode=episode,
                             observation_step=step, **kw)

    def archive(self, index=0):
        with np.load(self.directory / f"received_{index:08d}.npz", allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        return arrays, json.loads(arrays.pop("metadata_json").item())

    def events(self):
        return [json.loads(line) for line in (self.directory / "events.jsonl").read_text().splitlines()]

    def test_lossless_all_sensors_and_no_input_mutation(self):
        obs = sensors()
        before = {key: value.tobytes() for key, value in obs.items()}
        self.record(obs=obs, contract_key="pick up|target", policy_status="EXECUTE")
        arrays, meta = self.archive()
        for name, key in SENSORS.items():
            self.assertEqual(arrays[name].dtype, obs[key].dtype)
            self.assertEqual(arrays[name].shape, obs[key].shape)
            self.assertEqual(arrays[name].tobytes(), before[key])
            self.assertEqual(obs[key].tobytes(), before[key])
        self.assertEqual(meta["contract_key_before_decision"], "pick up|target")
        self.assertFalse(meta["previous_sent_action"]["available"])
        self.assertEqual(meta["missing_required"], [])
        self.assertGreater(self.events()[-1]["bytes"], 0)
        self.assertGreaterEqual(self.events()[-1]["capture_ms"], 0)

    def test_gt_not_archived_even_from_unsanitized_caller(self):
        obs = sensors()
        sentinel = "PRIVATE_GT_SENTINEL"
        obs.update({"object_poses": sentinel, "reward": sentinel,
                    "robot_r1::nav_gt": sentinel, "phase": sentinel,
                    "robot_r1::robot_r1:gt:Camera:0::rgb": np.array([23]),
                    "robot_r1::robot_r1:zed_link:Camera:0::seg_semantic": np.array([23])})
        self.record(obs=obs)
        arrays, meta = self.archive()
        self.assertEqual(set(arrays), set(SENSORS))
        self.assertNotIn(sentinel, json.dumps(meta))
        self.assertNotIn(sentinel, (self.directory / "events.jsonl").read_text())
        self.assertNotIn("object_poses", json.dumps(meta))

    def test_nan_and_missing_preserved_and_visible(self):
        obs = sensors()
        obs[SENSORS["head_depth_m"]][0, 0, 0] = np.nan
        obs[SENSORS["proprio"]][0] = np.inf
        del obs[SENSORS["camera_pose"]]
        del obs[SENSORS["left_wrist_rgb"]]
        self.record(obs=obs)
        arrays, meta = self.archive()
        self.assertTrue(np.isnan(arrays["head_depth_m"][0, 0, 0]))
        self.assertEqual(arrays["head_depth_m"].tobytes(), obs[SENSORS["head_depth_m"]].tobytes())
        self.assertIn("camera_pose", meta["missing_required"])
        self.assertIn("left_wrist_rgb", meta["absent_optional"])
        self.assertEqual(meta["arrays"]["proprio"]["nonfinite_count"], 1)
        self.assertEqual(self.events()[-1]["status"], "partial")

    def test_empty_and_object_arrays_never_pickle(self):
        obs = sensors()
        obs[SENSORS["camera_pose"]] = {"hidden_object_truth": 1}
        obs[SENSORS["proprio"]] = np.array([], np.float32)
        self.record(obs=obs)
        arrays, meta = self.archive()
        self.assertNotIn("camera_pose", arrays)
        self.assertEqual(meta["invalid_sensors"]["camera_pose"], "non_numeric_sensor_rejected")
        self.assertEqual(meta["invalid_sensors"]["proprio"], "empty_sensor")
        self.assertNotIn("hidden_object_truth", json.dumps(meta))

    def test_all_missing_is_a_received_record_not_silent_skip(self):
        self.record(obs={})
        arrays, meta = self.archive()
        self.assertEqual(arrays, {})
        self.assertEqual(len(meta["missing_required"]), 4)
        self.assertEqual(self.recorder.received, 1)

    def test_duplicate_reversed_gap_and_explicit_episode_reset(self):
        for step in (0, 0, 3, 1):
            self.record(step)
        flags = [self.archive(index)[1]["sequence_flags"] for index in range(4)]
        self.assertEqual(flags, [[], ["duplicate_policy_step"], ["policy_step_gap"], ["out_of_order_policy_step"]])
        self.recorder.action_sent(np.arange(23, dtype=np.float32), episode=1, observation_step=1)
        self.recorder.reset(1)  # even a reused episode label is a distinct reset generation
        self.record()
        _, meta = self.archive(4)
        self.assertEqual(meta["generation"], 1)
        self.assertEqual(meta["sequence_flags"], [])
        self.assertFalse(meta["previous_sent_action"]["available"])
        self.assertEqual(len(list(self.directory.glob("received_*.npz"))), 5)

    def test_unannounced_episode_change_is_visible_error(self):
        self.record()
        self.record(episode=2)
        self.assertEqual(self.events()[-1]["event"], "onboard_recording_error")
        self.assertFalse((self.directory / "received_00000001.npz").exists())

    def test_previous_action_and_final_result_boundary(self):
        self.record()
        sent = np.arange(23, dtype=np.float32)
        self.recorder.action_sent(sent, episode=1, observation_step=0)
        sent[:] = -10
        self.record(1)
        arrays, meta = self.archive(1)
        np.testing.assert_array_equal(arrays["previous_sent_action"], np.arange(23, dtype=np.float32))
        self.assertEqual(meta["previous_sent_action"]["observation_receive_id"], 0)
        self.recorder.finish()
        self.assertTrue(self.events()[-1]["last_sent_action_has_later_received_observation"])
        self.recorder.action_sent(sent, episode=1, observation_step=1)
        self.recorder.finish()
        self.assertFalse(self.events()[-1]["last_sent_action_has_later_received_observation"])
        self.assertIn("unknown", self.events()[-1]["terminal_observation_coverage"])
        count = len(self.events())
        self.recorder.finish()
        self.assertEqual(len(self.events()), count)

    def test_directory_and_file_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            OnboardRecorder(self.directory)
        path = self.directory / "received_00000000.npz"
        path.write_bytes(b"prior evidence")
        self.record()
        self.assertEqual(path.read_bytes(), b"prior evidence")
        self.assertEqual(self.events()[-1]["reason"], "FileExistsError")

    def test_write_failure_visible_and_later_capture_recovers(self):
        with patch("harness.onboard_recording.np.savez_compressed", side_effect=OSError("disk full")):
            self.record()
        self.assertEqual(self.events()[-1]["event"], "onboard_recording_error")
        self.assertEqual(self.emit.call_args.args[0]["event"], "onboard_recording_error")
        self.record(1)
        self.assertEqual(self.archive(1)[1]["observation_step"], 1)
        self.assertEqual(self.recorder.errors, 1)

    def test_event_sink_failure_does_not_escape(self):
        with patch.object(Path, "open", side_effect=OSError("storage unavailable")):
            self.record()
        self.assertEqual(self.emit.call_args.args[0]["event_log_error"], "OSError")

    def test_configuration_is_explicit_and_defaults_disabled(self):
        self.assertFalse(recording_config(None)["enabled"])
        for config in (True, False, {}, {"enabled": 1}, {"enabled": True, "every": 2}):
            with self.assertRaises(ValueError):
                recording_config(config)
        self.assertTrue(recording_config({"enabled": True})["compressed"])
        self.assertFalse(recording_config({"enabled": False})["enabled"])
        self.assertIn("NOT guaranteed", recording_manifest({"enabled": True})["terminal_frame"])

    def test_uncompressed_mode_is_lossless_too(self):
        self.recorder.compressed = False
        self.record()
        np.testing.assert_array_equal(self.archive()[0]["head_rgb"], sensors()[SENSORS["head_rgb"]])

    def policy(self, recorder=True):
        p = SimpleNamespace(episode=1, step_in_episode=0, queue=[], program=None,
                            orch=SimpleNamespace(contract=SimpleNamespace(key="navigate|target"), status="EXECUTE"),
                            _encode=lambda pixels: ([], np.zeros(3)), _record_proprio=lambda state: None,
                            _preprocess=lambda obs: (np.zeros(61), np.zeros((3, 4, 5))),
                            _proprio_buf=[], _reset_v3_contexts=lambda reason: None,
                            _emit=lambda event: None, _orch_finish_episode=lambda: None)
        if recorder:
            p.onboard_recorder = self.recorder
        p._policy_observation = policy_method("_policy_observation")
        for name in ("_split_navigation_observation", "_act", "act", "reset", "finish_episode"):
            setattr(p, name, MethodType(policy_method(name), p))
        def decision(obs, state, feats, cls, started):
            if recorder:
                self.assertTrue((self.directory / f"received_{self.recorder.receive_id - 1:08d}.npz").exists())
            self.assertNotIn("nav_truth", obs)
            p.step_in_episode += 1
            return np.arange(23, dtype=np.float32)
        p._act_orchestrated = decision
        return p

    def test_real_act_hook_navigation_hold_contract_switch_and_reset(self):
        p = self.policy()
        obs = {**sensors(), "nav_truth": "must not archive"}
        for key, status in (("navigate|target", "EXECUTE"), ("pick|target", "EXECUTE"), (None, "HOLD")):
            p.orch.contract = None if key is None else SimpleNamespace(key=key)
            p.orch.status = status
            np.testing.assert_array_equal(p.act(obs), np.arange(23, dtype=np.float32))
        self.assertEqual([self.archive(i)[1]["contract_key_before_decision"] for i in range(3)],
                         ["navigate|target", "pick|target", None])
        self.assertEqual(self.archive(2)[1]["policy_status_before_decision"], "HOLD")
        p.reset()
        p.act(obs)
        self.assertEqual(self.archive(3)[1]["episode"], 2)
        self.assertEqual(self.archive(3)[1]["observation_step"], 0)

    def test_navigation_prepare_error_still_records_once_without_truth(self):
        p = self.policy()
        p.orch.contract.is_navigation = True
        p.privbase = SimpleNamespace(finished=False, prepare=Mock(side_effect=ValueError("invalid nav packet")),
                                     invalidate=Mock())
        with self.assertRaises(ValueError):
            p.act({**sensors(), "nav_truth": "PRIVATE_GT_SENTINEL"})
        self.assertEqual(self.recorder.received, 1)
        self.assertEqual(set(self.archive()[0]), set(SENSORS))
        self.assertNotIn("PRIVATE_GT_SENTINEL", json.dumps(self.archive()[1]))
        p.privbase.invalidate.assert_called_once()

    def test_contract_metadata_precedes_decision_switch(self):
        p = self.policy()
        def switch(*args):
            self.assertEqual(self.archive()[1]["contract_key_before_decision"], "navigate|target")
            p.orch.contract = SimpleNamespace(key="next|target")
            p.step_in_episode += 1
            return np.ones(23, np.float32)
        p._act_orchestrated = switch
        np.testing.assert_array_equal(p.act(sensors()), np.ones(23, np.float32))
        self.assertEqual(self.archive()[1]["contract_key_before_decision"], "navigate|target")

    def test_hook_before_preprocessing_failure(self):
        p = self.policy()
        p._preprocess = Mock(side_effect=ValueError("missing proprio"))
        with self.assertRaises(ValueError):
            p.act({})
        self.assertEqual(self.archive()[1]["missing_required"], ["head_rgb", "proprio", "camera_pose", "head_depth"])

    def test_disabled_hook_preserves_action_and_creates_no_archive(self):
        p = self.policy(recorder=False)
        np.testing.assert_array_equal(p.act(sensors()), np.arange(23, dtype=np.float32))
        self.assertFalse(list(self.directory.glob("received_*.npz")))

    def test_capture_error_does_not_change_decision_action(self):
        p = self.policy()
        with patch("harness.onboard_recording.np.savez_compressed", side_effect=OSError):
            np.testing.assert_array_equal(p.act(sensors()), np.arange(23, dtype=np.float32))
        self.assertEqual(self.events()[-1]["event"], "onboard_recording_error")

    def test_websocket_receipt_occurs_only_after_successful_send(self):
        tree = ast.parse(POLICY.read_text())
        serve = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "serve")
        handler = next(node for node in serve.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "handler")
        class Closed(Exception):
            pass
        class Action:
            def cpu(self):
                return self
            def numpy(self):
                return np.arange(23, dtype=np.float32)
        def run(fail_send):
            capture = Mock()
            p = SimpleNamespace(metadata=lambda: {}, stats={}, onboard_recorder=capture, episode=1,
                                step_in_episode=1, _action_step=0, act=lambda msg: Action(), finish_episode=Mock())
            ws = SimpleNamespace(remote_address="cpu-test", receives=0, sends=0)
            async def recv():
                ws.receives += 1
                if ws.receives > 1:
                    raise Closed()
                return sensors()
            async def send(packet):
                ws.sends += 1
                self.assertFalse(capture.action_sent.called)
                if fail_send and ws.sends == 2:
                    raise Closed()
            ws.recv, ws.send = recv, send
            namespace = {"policy": p, "log": Mock(), "Packer": lambda: SimpleNamespace(pack=lambda value: value),
                         "unpackb": lambda value: value, "websockets": SimpleNamespace(ConnectionClosed=Closed),
                         "control_message": lambda policy, msg: None, "time": __import__("time"),
                         "hold_on_error": False, "np": np, "observation_owner": [None]}
            exec(compile(ast.Module(body=[handler], type_ignores=[]), str(POLICY), "exec"), namespace)
            if fail_send:
                with self.assertRaises(Closed):
                    asyncio.run(namespace["handler"](ws))
                capture.action_sent.assert_not_called()
            else:
                asyncio.run(namespace["handler"](ws))
                capture.action_sent.assert_called_once()
                np.testing.assert_array_equal(capture.action_sent.call_args.args[0], np.arange(23, dtype=np.float32))
            p.finish_episode.assert_called_once()
        run(False)
        run(True)

    def test_second_observation_client_cannot_mutate_active_episode(self):
        tree = ast.parse(POLICY.read_text())
        serve = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "serve")
        handler = next(n for n in serve.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "handler")
        owner = object()
        p = SimpleNamespace(metadata=lambda: {}, act=Mock(), reset=Mock(), finish_episode=Mock())
        ws = SimpleNamespace(remote_address="second-client")
        async def recv():
            return {"reset": True}
        async def send(packet):
            pass
        ws.recv, ws.send = recv, send
        namespace = {"policy": p, "log": Mock(), "Packer": lambda: SimpleNamespace(pack=lambda x: x),
                     "unpackb": lambda x: x, "websockets": SimpleNamespace(ConnectionClosed=EOFError),
                     "control_message": lambda p, m: None, "observation_owner": [owner]}
        exec(compile(ast.Module(body=[handler], type_ignores=[]), str(POLICY), "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "owns the policy observation stream"):
            asyncio.run(namespace["handler"](ws))
        p.act.assert_not_called()
        p.reset.assert_not_called()
        p.finish_episode.assert_not_called()
        self.assertIs(namespace["observation_owner"][0], owner)

    def test_runtime_artifact_paths_include_new_module(self):
        source = ROOT / "scripts/behavior_closed_loop.py"
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "runtime_artifact_paths")
        root = Path(self.tmp.name)
        (root / "stats.json").write_text("{}")
        namespace = {"ROOT": root, "CONFIG": {"source_dir": "sim", "task": {"id": 0}}, "os": SimpleNamespace(environ={}),
                     "Path": Path, "json": json, "task_profile": SimpleNamespace(TASK_ID=0, VTT_DIR=root / "vtt")}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        cfg = {"dataset": {"stats_path": "stats.json"}, "model": {"vision_encoder": {"checkpoint_path": "dino"}}}
        args = SimpleNamespace(checkpoint=str(root / "model.pt"), orchestrate=None, program=None,
                               memory_vtt=None, env_wrapper=None)
        for config in (None, {"onboard_recording": {"enabled": True}}):
            args.orchestrate = json.dumps(config) if config is not None else None
            paths = namespace["runtime_artifact_paths"](args, ["--robot-config", str(root / "robot.yaml")], cfg)
            self.assertIn(root / "harness/onboard_recording.py", paths)


if __name__ == "__main__":
    unittest.main()
