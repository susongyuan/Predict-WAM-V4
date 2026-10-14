"""Opt-in, passive evidence of policy-received sensors, before any decision.

No evaluator IDs are supplied by the current wire protocol. ``receive_id`` counts
act() calls, while ``observation_step`` is the policy's pre-decision counter. A
repeated/reversed counter is recorded again, explicitly flagged, never silently
overwritten or treated as a new episode. Only reset() starts a new episode.

NPZ arrays retain source shape/dtype/bytes (including RGBA and nonfinite values).
No image resize, float cast, missing-value imputation or object pickle is used.
Synchronous I/O cost and failures are logged; recording never gates actions.
"""
import json
import logging
from pathlib import Path
import time

import numpy as np


SCHEMA = "policy_received_sensors_v1"
# A second, exact allowlist: even a caller accidentally passing unsanitized obs
# cannot archive object truth, navigation packets, scores, task labels or replies.
SENSORS = {
    "head_rgb": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "head_depth_m": "robot_r1::robot_r1:zed_link:Camera:0::depth_linear",
    "head_depth": "robot_r1::robot_r1:zed_link:Camera:0::depth",
    "left_wrist_rgb": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "right_wrist_rgb": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    "proprio": "robot_r1::proprio",
    "camera_pose": "robot_r1::cam_rel_poses",
}
LOG = logging.getLogger(__name__)


def recording_config(config):
    """Explicit enablement only; reject misspellings instead of silently omitting evidence."""
    if config is None:
        return {"enabled": False, "compressed": True}
    if not isinstance(config, dict) or set(config) - {"enabled", "compressed"}:
        raise ValueError("onboard_recording requires only enabled/compressed boolean options")
    if "enabled" not in config or any(type(value) is not bool for value in config.values()):
        raise ValueError("onboard_recording.enabled must be explicit; options must be booleans")
    return {"compressed": True, **config}


def recording_manifest(config):
    cfg = recording_config(config)
    return {
        **cfg, "schema": SCHEMA, "directory": "onboard_received",
        "coverage": "policy received observations only, all phases, before preprocessing/decisions",
        "cadence": "one record per act call; control/reset/status messages are not observations",
        "identity": "process receive_id plus reset generation, policy episode and pre-decision step; no evaluator observation ID",
        "duplicates": "preserved with unique receive_id; duplicate/out-of-order policy steps flagged",
        "sensors": SENSORS,
        "camera_pose_semantics": "native cam_rel_poses vector, original order unchanged; no world object/base truth",
        "format": "lossless numeric NPZ arrays, source shape/dtype; embedded metadata_json and events.jsonl",
        "previous_action": "only after successful websocket.send; transport receipt is not simulator execution acknowledgement",
        "terminal_frame": "NOT guaranteed: evaluator may terminate without sending the last action result to policy",
        "episode_end": "events report actions without a subsequent received observation, never infer missing physical results",
        "io": "synchronous; capture_ms and bytes per observation; errors visible in events, policy trace and stderr",
        "changes_wam_inputs_or_actions": False, "changes_stage_decisions": False,
        "physical_success_admitted": False,
    }


class OnboardRecorder:
    def __init__(self, directory, *, compressed=True, emit=None):
        self.directory = Path(directory)
        # Never append to a prior run, including an empty pre-existing directory.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events = self.directory / "events.jsonl"
        with self.events.open("x"):
            pass
        self.compressed, self.emit = compressed, emit
        with (self.directory / "manifest.json").open("x") as stream:
            json.dump(recording_manifest({"enabled": True, "compressed": compressed}), stream, indent=2)
            stream.write("\n")
        self.receive_id = 0
        self.generation = 0
        self.episode = None
        self._clear_episode()

    def _clear_episode(self):
        self.seen_steps = set()
        self.max_step = None
        self.last_receive_id = None
        self.last_action = None
        self.action_receive_id = None
        self.action_step = None
        self.received = self.saved = self.errors = 0
        self._finished = False

    def _event(self, record):
        record = {"time": time.time(), **record}
        try:
            with self.events.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        except Exception as error:
            # Independent sink: disk-full/permission failure must not disappear.
            LOG.error("onboard_recording event write failed: %s", error, exc_info=True)
            record = {**record, "event_log_error": type(error).__name__}
        if self.emit is not None:
            try:
                self.emit(dict(record))
            except Exception:
                LOG.exception("onboard_recording policy trace write failed")
        if record.get("event") == "onboard_recording_error":
            LOG.error("onboard_recording error: %s", record)

    def reset(self, episode):
        self.finish()
        self.generation += 1
        self.episode = int(episode)
        self._clear_episode()
        self._event({"event": "onboard_recording_reset", "episode": self.episode,
                     "generation": self.generation})

    def record(self, obs, *, episode, observation_step, contract_key=None, policy_status=None):
        started = time.monotonic()
        receive_id = self.receive_id
        self.receive_id += 1
        meta = {"schema": SCHEMA, "receive_id": receive_id, "episode": int(episode),
                "generation": self.generation, "observation_step": int(observation_step),
                "contract_key_before_decision": contract_key, "policy_status_before_decision": policy_status,
                "received_time": time.time(), "source_keys": {}, "arrays": {}}
        self.received += 1
        self.last_receive_id = receive_id
        self._finished = False
        try:
            if self.episode is None:
                self.episode = int(episode)
            # Do not guess episode resets from an out-of-order observation.
            if self.episode != episode:
                raise ValueError("episode changed without recorder.reset")
            flags = []
            if observation_step in self.seen_steps:
                flags.append("duplicate_policy_step")
            if self.max_step is not None and observation_step < self.max_step:
                flags.append("out_of_order_policy_step")
            if self.max_step is not None and observation_step > self.max_step + 1:
                flags.append("policy_step_gap")
            self.seen_steps.add(observation_step)
            self.max_step = observation_step if self.max_step is None else max(self.max_step, observation_step)
            meta["sequence_flags"] = flags
            arrays, invalid = {}, {}
            for name, key in SENSORS.items():
                if key not in obs:
                    continue
                value = np.asarray(obs[key])
                if value.dtype.kind not in "buif":
                    invalid[name] = "non_numeric_sensor_rejected"
                    continue
                arrays[name] = value.copy()
                meta["source_keys"][name] = key
                meta["arrays"][name] = {"shape": list(value.shape), "dtype": value.dtype.str,
                                        "nonfinite_count": int(np.size(value) - np.isfinite(value).sum())}
                if not value.size:
                    invalid[name] = "empty_sensor"
                elif meta["arrays"][name]["nonfinite_count"]:
                    invalid[name] = "nonfinite_sensor_preserved"
            missing = [name for name in ("head_rgb", "proprio", "camera_pose") if name not in arrays]
            if "head_depth_m" not in arrays and "head_depth" not in arrays:
                missing.append("head_depth")
            meta.update(missing_required=missing, invalid_sensors=invalid,
                        absent_optional=[name for name in ("left_wrist_rgb", "right_wrist_rgb") if name not in arrays])
            if self.last_action is not None:
                arrays["previous_sent_action"] = self.last_action.copy()
            meta["previous_sent_action"] = {
                "available": self.last_action is not None, "observation_receive_id": self.action_receive_id,
                "policy_step": self.action_step, "source": "websocket.send completed; simulator execution not acknowledged"}
            path = self.directory / f"received_{receive_id:08d}.npz"
            arrays["metadata_json"] = np.asarray(json.dumps(meta, allow_nan=False))
            with path.open("xb") as stream:
                (np.savez_compressed if self.compressed else np.savez)(stream, **arrays)
            self.saved += 1
            self._event({"event": "onboard_recording", **meta, "path": path.name,
                         "status": "partial" if missing or invalid else "saved",
                         "bytes": path.stat().st_size, "capture_ms": (time.monotonic() - started) * 1000})
        except Exception as error:
            self.errors += 1
            self._event({"event": "onboard_recording_error", "receive_id": receive_id,
                         "episode": int(episode), "generation": self.generation,
                         "observation_step": int(observation_step), "reason": type(error).__name__,
                         "capture_ms": (time.monotonic() - started) * 1000})

    def action_sent(self, action, *, episode, observation_step):
        """Called AFTER send, including hold-on-error; never use a queued/planned action."""
        try:
            if episode != self.episode or self.last_receive_id is None:
                raise ValueError("sent action has no received observation in this episode")
            value = np.asarray(action)
            if value.dtype.kind not in "buif" or value.shape != (23,) or not np.isfinite(value).all():
                raise ValueError("invalid sent action")
            self.last_action = value.copy()
            self.action_receive_id, self.action_step = self.last_receive_id, int(observation_step)
            self._finished = False
            self._event({"event": "onboard_recording_action_sent", "episode": int(episode),
                         "generation": self.generation, "receive_id": self.last_receive_id,
                         "observation_step": int(observation_step), "action": value.tolist()})
        except Exception as error:
            self.last_action = None
            self.action_receive_id = self.action_step = None
            self._event({"event": "onboard_recording_error", "operation": "action_sent",
                         "reason": type(error).__name__})

    def finish(self):
        if self._finished or self.episode is None:
            return
        self._event({"event": "onboard_recording_episode_end", "episode": self.episode,
                     "generation": self.generation, "received": self.received, "saved": self.saved,
                     "capture_errors": self.errors, "last_receive_id": self.last_receive_id,
                     "last_sent_action_receive_id": self.action_receive_id,
                     "last_sent_action_has_later_received_observation": (
                         None if self.action_receive_id is None else self.last_receive_id > self.action_receive_id),
                     "terminal_observation_coverage": "unknown: policy cannot archive observations evaluator never sent"})
        self._finished = True
