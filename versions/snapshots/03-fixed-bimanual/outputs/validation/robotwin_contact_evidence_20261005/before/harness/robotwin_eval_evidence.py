"""Evaluation-side boundaries, evidence and recording; never policy inputs.

No import-time simulator, torch, rendering, or model activity. Unknown evidence
stays null. A native predicate is not full containment or competition success.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import resource
import select
import signal
import subprocess
import time

from harness.robotwin_policy_worker import InfrastructureError, atomic_json, vector


class WallTimeout(BaseException):
    """Not an Exception: native TOPP's broad Exception handler must not swallow it."""


@contextmanager
def wall_limit(seconds):
    def expired(signum, frame):
        raise WallTimeout("episode wall limit expired")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class ResourceGuard:
    def __init__(self):
        self.previous = self.ticks()
        self.peaks = {}

    @staticmethod
    def ticks():
        values = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        return sum(values), values[3] + values[4]

    def check(self):
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        total, available = (int(info[k].split()[0]) for k in ("MemTotal", "MemAvailable"))
        now = self.ticks()
        dt, di = now[0] - self.previous[0], now[1] - self.previous[1]
        self.previous = now
        values = {"system_ram_percent": 100 * (1 - available / total),
                  "system_cpu_percent": 100 * (1 - di / dt) if dt else None,
                  "sim_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        for key, value in values.items():
            if value is not None:
                self.peaks[key] = max(self.peaks.get(key, 0), value)
        if values["system_ram_percent"] >= 85 or (dt >= 10 and values["system_cpu_percent"] >= 85):
            raise InfrastructureError(f"CPU/RAM85 resource boundary exceeded: {values}")
        return values


def joint_bounds(robot):
    """Read only this robot's native URDF-loaded joint limits, in policy order."""
    import numpy as np
    records = []
    for arm in ("left", "right"):
        joints = getattr(robot, f"{arm}_arm_joints")
        if len(joints) != 6:
            raise InfrastructureError("expected six native arm joints per side")
        for joint in joints:
            limits = np.asarray(joint.get_limits(), dtype=np.float64)
            if limits.shape != (1, 2) or not np.isfinite(limits).all() or limits[0, 0] > limits[0, 1]:
                raise InfrastructureError("missing/unbounded robot joint limit; no guessed bounds allowed")
            records.append({"arm": arm, "joint": joint.get_name(), "limits": limits[0].tolist()})
    return np.asarray([r["limits"] for r in records]), records


def bound_action(raw, limits):
    import numpy as np
    from harness.robotwin_qpos_adapter import ARM_INDICES, validate_action
    values = vector(raw, 14, "raw action")
    bounds = np.asarray(limits)
    if (bounds.shape != (12, 2) or bounds.dtype.kind not in "fiu" or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > bounds[:, 1])):
        raise InfrastructureError("invalid physical joint bounds")
    # float64 avoids integer truncation and preserves all float32 model values.
    bounded = values.astype(np.float64)
    bounded[list(ARM_INDICES)] = np.clip(bounded[list(ARM_INDICES)], bounds[:, 0], bounds[:, 1])
    bounded[[6, 13]] = np.clip(bounded[[6, 13]], 0, 1)
    validate_action(bounded, bounds)
    return {"raw_action": values.tolist(), "bounded_action": bounded.tolist(),
            "clip_mask": (values != bounded).tolist()}


def empty_evidence():
    return {"native_benchmark_success": None, "native_terminal_predicate": None,
            "native_predicate_after_settle": None,
            "object_pose_world": None, "cabinet_joint_qpos": None,
            "release_verified": None, "stable_after_settle": None,
            "full_containment": None, "competition_success": None, "five_second_hold": None,
            "scene_timestamp": None,
            "evidence_limitations": ["release/contact criterion not implemented",
                "complete object-in-cabinet volume test not implemented",
                "scene clock and five-second hold not validated"]}


def snapshot(env):
    import numpy as np
    result = {"object_pose_world": None, "cabinet_joint_qpos": None,
              "commanded_grippers": None, "gripper_drive_target_normalized": None, "errors": {}}
    def capture(name, fn):
        try:
            value = fn()
            array = np.asarray(value)
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError("nonfinite evidence")
            result[name] = array.tolist()
        except Exception as exc:
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
    capture("object_pose_world", lambda: np.concatenate((env.object.get_pose().p, env.object.get_pose().q)))
    capture("cabinet_joint_qpos", lambda: env.cabinet.get_qpos())
    capture("commanded_grippers", lambda: [env.robot.get_left_gripper_val(), env.robot.get_right_gripper_val()])
    # This accessor returns drive targets, NOT measured jaw separation/contact.
    capture("gripper_drive_target_normalized", env.robot.get_normal_real_gripper_val)
    result["object_model"] = str(getattr(env, "selected_modelname", "unknown"))
    result["object_model_id"] = str(getattr(env, "selected_model_id", "unknown"))
    return result


def settle_diagnostic(poses, requested_steps, completed_steps):
    """Measured pose drift across the fixed-step window, not physical time/inside."""
    result = {"stable_after_settle": None, "requested_physics_steps": requested_steps,
              "completed_physics_steps": completed_steps, "max_translation_m": None,
              "max_rotation_rad": None, "translation_threshold_m": 0.002,
              "rotation_threshold_rad": 0.02, "scene_seconds": None}
    if requested_steps <= 0 or completed_steps != requested_steps or len(poses) != completed_steps + 1 or any(p is None for p in poses):
        return result
    import numpy as np
    array = np.asarray(poses, dtype=np.float64)
    if array.shape != (completed_steps + 1, 7) or not np.isfinite(array).all():
        return result
    norms = np.linalg.norm(array[:, 3:], axis=1)
    if np.any(norms < 1e-9):
        return result
    quats = array[:, 3:] / norms[:, None]
    distance = float(np.max(np.linalg.norm(array[:, :3] - array[0, :3], axis=1)))
    rotation = float(np.max(2 * np.arccos(np.clip(np.abs(quats @ quats[0]), 0, 1))))
    result.update(max_translation_m=distance, max_rotation_rad=rotation,
                  stable_after_settle=distance <= 0.002 and rotation <= 0.02)
    return result


def jsonl(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()


def make_camera(env):
    """Evaluation-only 1080p camera; fixed design, actual framing not yet validated."""
    import numpy as np
    import sapien
    position = np.array([1.25, -1.45, 1.85])
    target = np.array([0.0, 0.0, 0.95])
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack((forward, left, up), axis=1)
    matrix[:3, 3] = position
    camera = env.scene.add_camera(name="evaluation_only_third_view", width=1920, height=1080,
                                  fovy=math.radians(65), near=0.05, far=100)
    camera.entity.set_pose(sapien.Pose(matrix))
    return camera, {"width": 1920, "height": 1080, "position": position.tolist(),
        "look_at": target.tolist(), "fovy_degrees": 65, "framing_verified": None,
        "policy_access": False, "full_scene_coverage_verified": None}


def resolve_ffmpeg(explicit=None):
    """Reuse a local executable only; never import another environment or download."""
    import shutil
    from harness.robotwin_policy_worker import ROOT
    candidates = [explicit] if explicit else [shutil.which("ffmpeg")]
    if not explicit:
        candidates += sorted(ROOT.glob(".conda-envs/*/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise InfrastructureError("no local ffmpeg executable; provide --ffmpeg, no automatic downloads")


class VideoRecorder:
    def __init__(self, directory, episode, seed, camera_metadata, fps=30, synthetic=False, ffmpeg=None):
        self.directory, self.episode, self.seed = Path(directory), episode, seed
        self.prefix = "synthetic" if synthetic else "third_view"
        self.frames, self.closed = 0, False
        self.meta = {"episode": episode, "seed": seed, "synthetic": synthetic, "nominal_video_fps": fps,
            "scene_clock_validated": False, "camera": camera_metadata, "frames": 0, "complete": False,
            "sampling": "initial frame, after each native control call, then each requested settle physics step; not continuous physics-rate capture"}
        executable = resolve_ffmpeg(ffmpeg)
        self.meta["ffmpeg_executable"] = executable
        self.err = (self.directory / f"{self.prefix}_ffmpeg.log").open("xb")
        try:
            self.proc = subprocess.Popen([executable, "-nostdin", "-n", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", "1920x1080", "-framerate", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                str(self.directory / f"{self.prefix}.mp4")], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)
            os.set_blocking(self.proc.stdin.fileno(), False)
        except BaseException:
            self.err.close()
            raise

    def append(self, rgb, control_step, settle_step=0):
        import numpy as np
        image = np.asarray(rgb)
        if image.shape != (1080, 1920, 3) or image.dtype != np.uint8:
            raise InfrastructureError("third-view recording requires native 1920x1080 uint8 RGB, no upscaling")
        data = memoryview(image.tobytes())
        deadline = time.monotonic() + 20
        while data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.proc.poll() is not None:
                raise InfrastructureError("ffmpeg timeout/exit")
            _, writable, _ = select.select([], [self.proc.stdin], [], remaining)
            if writable:
                try:
                    data = data[os.write(self.proc.stdin.fileno(), data[:65536]):]
                except BlockingIOError:
                    continue
        jsonl(self.directory / f"{self.prefix}_frames.jsonl", {"episode": self.episode, "seed": self.seed,
            "frame_index": self.frames, "control_step": control_step, "settle_step": settle_step,
            "scene_timestamp": None, "nominal_video_fps": self.meta["nominal_video_fps"]})
        self.frames += 1

    def capture(self, env, camera, control_step, settle_step=0):
        import numpy as np
        env._update_render()
        camera.take_picture()
        rgb = (camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
        self.append(rgb, control_step, settle_step)

    def close(self, complete=False):
        if self.closed:
            return
        self.closed = True
        try:
            self.proc.stdin.close()
            try:
                code = self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
                raise InfrastructureError("ffmpeg finalize timeout")
            if code != 0:
                raise InfrastructureError(f"ffmpeg exited {code}")
            self.meta["complete"] = bool(complete)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=2)
            self.err.close()
            self.meta["frames"] = self.frames
            atomic_json(self.directory / f"{self.prefix}_video.json", self.meta)
