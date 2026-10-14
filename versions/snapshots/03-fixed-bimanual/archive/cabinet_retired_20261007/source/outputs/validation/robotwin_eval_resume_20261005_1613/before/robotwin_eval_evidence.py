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


class EvaluationCancelled(BaseException):
    """Cancellation must bypass native TOPP's except Exception fallback."""


class RunStop:
    """Per-run sticky stop state; cooperative boundaries, not a native hard kill.

    A bare except may swallow the exception, but cannot clear this latch. Real
    execution still requires an external cgroup runtime/stop timeout watchdog.
    """
    def __init__(self, total_seconds):
        self.deadline = time.monotonic() + total_seconds
        self.episode_deadline = None
        self.reason = None
        self.previous_handlers = {}

    def terminate(self, signum, frame):
        if self.reason is None:
            self.reason = EvaluationCancelled(f"evaluator received signal {signum}")
        raise self.reason

    def install(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.previous_handlers[sig] = signal.signal(sig, self.terminate)

    def restore(self):
        for sig, handler in self.previous_handlers.items():
            signal.signal(sig, handler)

    def expire(self, message="wall limit expired"):
        if self.reason is None:
            self.reason = WallTimeout(message)
        raise self.reason

    def check(self):
        if self.reason is not None:
            raise self.reason
        if time.monotonic() >= min(self.deadline, self.episode_deadline or self.deadline):
            self.expire("absolute evaluation wall deadline expired")

    def call(self, fn, *args, **kwargs):
        self.check()
        value = fn(*args, **kwargs)
        self.check()
        return value


@contextmanager
def wall_limit(seconds, stop=None):
    def expired(signum, frame):
        if stop is not None:
            stop.expire()
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


def contact_snapshot(env):
    """Read-only SAPIEN 3 contact evidence, never a grasp/support/goal predicate.

    Public mapping: Actor.actor -> Entity; articulation.get_links() and
    PhysxContact.bodies -> PhysxRigidBaseComponent.entity. IDs are accepted only
    inside this scene's complete Entity registry. Names are never identities.
    A nonempty target contact is required to witness the reporter/target mapping;
    an empty list cannot establish separation. No Scene is constructed here.
    """
    result = {"status": "unknown", "reason": None, "api": None,
              "read_valid": False, "identity_valid": False, "counts_valid": False,
              "reported_contact_records": None, "target_robot_contact_count": None,
              "target_other_contact_count": None, "target_contact_observed": None,
              "target_other_nonpositive_separation_count": None,
              "target_entity_ids": None, "robot_entity_ids": None, "records": [],
              "release_verified": None, "count_unit": "reported_contact_records",
              "scope": "current reported contacts only; no support force, duration, or task success"}
    phase = "contact_api"
    try:
        scene = env.scene
        reporter = getattr(scene, "get_contacts", None)
        if reporter is None:
            reporter = getattr(getattr(scene, "physx_system", None), "get_contacts", None)
            result["api"] = "scene.physx_system.get_contacts"
        else:
            result["api"] = "scene.get_contacts"
        if not callable(reporter):
            raise ValueError("public contact reporter unavailable")
        phase = "contact_read"
        contacts = reporter()
        if not isinstance(contacts, (list, tuple)):
            raise ValueError("contact reporter did not return a list")
        result["reported_contact_records"] = len(contacts)
        result["read_valid"] = True
        phase = "identity_mapping"
        import sapien
        entity_type = sapien.Entity
        rigid_type = sapien.physx.PhysxRigidBaseComponent
        articulation_type = sapien.physx.PhysxArticulation
        entities = scene.get_entities()
        if not isinstance(entities, (list, tuple)) or not entities:
            raise ValueError("scene entity registry unavailable/empty")

        def entity_id(entity):
            if not isinstance(entity, entity_type) or entity.scene is not scene:
                raise ValueError("body is not an Entity in the evaluated scene")
            key = entity.per_scene_id
            if type(key) is not int or key < 0:
                raise ValueError("invalid per_scene_id")
            return key

        registry = {}
        for entity in entities:
            key = entity_id(entity)
            if key in registry:
                raise ValueError("duplicate per_scene_id in scene registry")
            registry[key] = entity

        def body_id(body):
            # Native contact bodies and robot links are rigid components;
            # the target's Actor wrapper instead holds an Entity.
            body = getattr(body, "actor", body)
            if isinstance(body, rigid_type):
                body = body.entity
            key = entity_id(body)
            if key not in registry:
                raise ValueError("body Entity absent from scene registry")
            return key

        def link_ids(articulation):
            if not isinstance(articulation, articulation_type):
                raise ValueError("robot/target articulation type unconfirmed")
            links = articulation.get_links()
            if not isinstance(links, (list, tuple)) or not links:
                raise ValueError("articulation links unavailable/empty")
            if any(not isinstance(link, rigid_type) for link in links):
                raise ValueError("unrecognized articulation link")
            return {body_id(link) for link in links}

        target = getattr(env.object, "actor", env.object)
        target_ids = link_ids(target) if isinstance(target, articulation_type) else {body_id(target)}
        robot_ids = set()
        for arm in ("left", "right"):
            robot_ids.update(link_ids(getattr(env.robot, f"{arm}_entity")))
        if target_ids & robot_ids:
            raise ValueError("target and robot identities overlap")
        result.update(target_entity_ids=sorted(target_ids), robot_entity_ids=sorted(robot_ids))
        robot_count = other_count = other_witness_count = 0
        for contact in contacts:
            bodies = contact.bodies
            if not isinstance(bodies, (list, tuple)) or len(bodies) != 2:
                raise ValueError("contact must contain two recognized bodies")
            keys = [body_id(body) for body in bodies]
            phase = "contact_points"
            points = contact.points
            if not isinstance(points, (list, tuple)) or not points:
                raise ValueError("contact has no observable points; not a contact witness")
            separations = [float(point.separation) for point in points]
            if not all(math.isfinite(value) for value in separations):
                raise ValueError("nonfinite contact point separation")
            minimum_separation = min(separations)
            phase = "identity_mapping"
            targets = [key in target_ids for key in keys]
            kind = "unrelated"
            if all(targets):
                kind = "target_internal"
            elif any(targets):
                other = keys[1] if targets[0] else keys[0]
                if other in robot_ids:
                    robot_count += 1
                    kind = "target_robot"
                else:
                    other_count += 1
                    other_witness_count += int(minimum_separation <= 0)
                    kind = "target_other"
            result["records"].append({"body_entity_ids": keys, "point_count": len(points), "kind": kind,
                                      "min_separation_m": minimum_separation})
        result.update(identity_valid=True, counts_valid=True,
                      target_robot_contact_count=robot_count, target_other_contact_count=other_count,
                      target_other_nonpositive_separation_count=other_witness_count,
                      target_contact_observed=bool(robot_count + other_count))
        if robot_count:
            result.update(status="valid", release_verified=False, reason="target_robot_contact_observed")
        elif other_witness_count:
            result.update(status="valid", release_verified=True,
                          reason="target_nonrobot_contact_observed_and_no_robot_contact")
        else:
            result["reason"] = "no_target_contact_witness; empty/irrelevant/positive-gap reports cannot prove separation"
    except Exception as exc:
        if phase in ("contact_read", "contact_points"):
            result["read_valid"] = False
        result["reason"] = f"{phase}: {type(exc).__name__}: {exc}"
    return result


def empty_evidence():
    return {"native_benchmark_success": None, "native_terminal_predicate": None,
            "native_predicate_after_settle": None,
            "object_pose_world": None, "cabinet_joint_qpos": None,
            "release_verified": None, "contact_evidence": None, "stable_after_settle": None,
            "full_containment": None, "competition_success": None, "five_second_hold": None,
            "scene_timestamp": None,
            "evidence_limitations": ["release means final reported contact separation only; real-scene API validation pending",
                "contact counts do not establish support force, grasp duration, or full containment",
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
    result["contact_evidence"] = contact_snapshot(env)
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
