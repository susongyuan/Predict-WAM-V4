#!/usr/bin/env python3
"""Native RoboTwin/WAM bridge, not the official seed-skipping expert evaluator.

Run --plan-only under any CPU Python with PyYAML; no torch or Scene import.
Real execution must use .venvs/robotwin-sim/bin/python after training releases
its GPU lock. Policy uses .venvs/wam-fixed/bin/python in a separate process.
Immutable milestone checkpoint required. No expert/IK/replay/mock fallback.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import random
import re
import signal
import sys
import time
import faulthandler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.robotwin_policy_worker import (InfrastructureError, PolicyClient, atomic_json,
    digest, identity, sha256, training_contract, verify_gpu_owner)
from harness.robotwin_eval_evidence import (ResourceGuard, RunStop, VideoRecorder, WallTimeout, EvaluationCancelled, bound_action,
    empty_evidence, joint_bounds, jsonl, make_camera, resolve_ffmpeg, settle_diagnostic, snapshot, wall_limit)

ROBOTWIN = ROOT / "third_party/RoboTwin"
CONFIG_DIR = ROBOTWIN / "env_cfg/task_config"
ASSET_RECEIPT = ROOT / "outputs/validation/robotwin_assets_20261005/asset_manifest.json"


class PhaseJournal:
    """Atomic latest-call evidence; never passed to policy or used for scoring.

    step is the zero-based action index (initialization uses 0). Observation
    after action N retains N. timestamp is Unix seconds; sequence orders writes.
    Arguments/returns are not serialized. Failures survive subsequent cleanup.
    This is not a heartbeat, watchdog, or a native-call timeout mechanism.
    """

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "live_state.json"
        self.state = dict(episode=None, seed=None, step=None, phase="created",
                          timestamp=time.time(), pid=os.getpid(), sequence=0,
                          status="completed", failures=[])

    def context(self, episode=None, seed=None, step=None):
        self.state.update(episode=episode, seed=seed, step=step)

    def write(self, phase, status, **extra):
        self.state.update(phase=phase, status=status, timestamp=time.time(),
                          sequence=self.state["sequence"] + 1, **extra)
        atomic_json(self.path, self.state)

    def failed(self, exc):
        failure = {key: self.state[key] for key in
                   ("episode", "seed", "step", "phase", "timestamp", "pid", "sequence")}
        failure["error"] = f"{type(exc).__name__}: {exc}"
        self.state["failures"].append(failure)
        self.write(self.state["phase"], "error")

    def call(self, phase, function, *args, **kwargs):
        self.write(phase, "running")
        # A Python deadline cannot interrupt a stalled native Vulkan call.
        # Emit a traceback for diagnosis; the external cgroup remains the hard
        # stop. The watchdog does not fabricate observations or retry actions.
        faulthandler.dump_traceback_later(60, repeat=True)
        try:
            result = function(*args, **kwargs)
        except BaseException as exc:
            self.failed(exc)
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()
        self.write(phase, "completed")
        return result


def validate_task(task):
    """Resolve a native task without importing simulator modules in CPU preflight."""
    if not isinstance(task, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", task):
        raise InfrastructureError("invalid native task name")
    if not (ROBOTWIN / "envs" / f"{task}.py").is_file():
        raise InfrastructureError(f"native environment not found: {task}")
    return task


def validate_pure(args):
    if getattr(args, "system_mode", "pure") != "pure":
        return
    if any(getattr(args, name, None) for name in ("target_head", "vtt_adapter", "vtt_bank")):
        raise InfrastructureError("pure WAM forbids target heads, VTT adapters and banks")
    if getattr(args, "execution_mode", "native8") != "native8":
        raise InfrastructureError("pure WAM requires native8; no feedback/recovery actions")


def task_evidence(task):
    evidence = empty_evidence()
    if task != "put_object_cabinet":
        evidence.pop("cabinet_joint_qpos")
        evidence["evidence_limitations"] = [
            "Only the selected task's native predicate/latch is scored; no generic grasp/release/hold claim",
            "Task-specific object/contact diagnostics are not implemented and remain null"]
    evidence["native_predicate"] = f"envs.{task}.{task}.check_success"
    return evidence


def task_snapshot(env, task):
    # Cabinet diagnostics assume env.object/env.cabinet. Never run them on other
    # tasks, or expose actor GT to policy. Nulls mean unimplemented, not failure.
    if task == "put_object_cabinet":
        return snapshot(env)
    return {"object_pose_world": None, "contact_evidence": {"release_verified": None},
            "task": task, "diagnostics_available": False}


def load_task_config(name, task):
    import yaml
    validate_task(task)
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise InfrastructureError("invalid native task config name")
    config = yaml.safe_load((CONFIG_DIR / f"{name}.yml").read_text())
    registry = yaml.safe_load((CONFIG_DIR / "_embodiment_config.yml").read_text())
    if config["embodiment"] != ["aloha-agilex"]:
        raise InfrastructureError("native14 bridge expects original dual-arm aloha-agilex")
    robot_dir = (ROBOTWIN / registry["aloha-agilex"]["file_path"]).resolve()
    embodiment = yaml.safe_load((robot_dir / "config.yml").read_text())
    files = [CONFIG_DIR / f"{name}.yml", CONFIG_DIR / "_embodiment_config.yml", CONFIG_DIR / "_camera_config.yml",
             CONFIG_DIR / "_eval_step_limit.yml", robot_dir / "config.yml", robot_dir / embodiment["urdf_path"]]
    config.update(task_name=task, task_config=name, eval_mode=True, planner_mode="joint_targets",
        left_robot_file=str(robot_dir), right_robot_file=str(robot_dir), dual_arm_embodied=True,
        left_embodiment_config=copy.deepcopy(embodiment), right_embodiment_config=copy.deepcopy(embodiment),
        render_freq=0, save_data=False, collect_data=False, eval_video_save_dir=None)
    # Keep original policy RGB + qpos flags. No extra modality enters the worker.
    if not config["data_type"]["rgb"] or not config["data_type"]["qpos"]:
        raise InfrastructureError("native RGB/qpos observation disabled")
    return config, files


def dataset_seed_inventory(cache):
    seed_files = set()
    for ep in cache["source"]["episodes"]:
        source = Path(ep["source"]).resolve()
        candidates = [p / "seed.txt" for p in source.parents if (p / "seed.txt").is_file()]
        if not candidates:
            raise InfrastructureError(f"original dataset seed.txt missing above {source}")
        seed_files.add(candidates[0])
    original_seeds = set()
    for file in seed_files:
        original_seeds.update(int(s) for s in file.read_text().split())
    if not seed_files or not original_seeds:
        raise InfrastructureError("dataset seed provenance is empty")
    return original_seeds, seed_files


def validate_system_task(args, task):
    """Explicit admission: retain cabinet modes; dustbin measured middle only.

    No VTT training/admission exists for the frozen zero-condition dustbin model.
    The full strict task supervisor/physical verifier remains unavailable here.
    """
    mode = getattr(args, "system_mode", "pure")
    if task == "put_object_cabinet" or mode == "pure":
        return
    if task == "put_bottles_dustbin" and mode == "middle":
        if any(getattr(args, k, None) for k in ("vtt_adapter", "vtt_bank", "target_head")):
            raise InfrastructureError("dustbin middle admits measured feedback only, no task-specific VTT/target head")
        if getattr(args, "execution_mode", "native8") != "native8":
            raise InfrastructureError("dustbin middle paired comparison requires original native8 execution")
        return
    raise InfrastructureError("task-generic bridge admits pure WAM; only dustbin measured middle is additionally admitted")


def plan_evaluation(args):
    validate_pure(args)
    contract, cache, stats, receipt = training_contract(args.training_run_dir, args.checkpoint)
    task = validate_task(contract["config"]["dataset"]["task_name"])
    recorded_config = contract["config"]["dataset"].get("task_config") or cache["source"].get("task_config")
    if args.task_config and recorded_config and args.task_config != recorded_config:
        raise InfrastructureError("task config differs from training provenance")
    config_name = args.task_config or recorded_config or "demo_randomized"
    validate_system_task(args, task)
    config, config_files = load_task_config(config_name, task)
    config["render_denoiser"] = getattr(args, "render_denoiser", "oidn")
    original_seeds, seed_files = dataset_seed_inventory(cache)
    overlap = set(args.seeds) & original_seeds
    if overlap:
        raise InfrastructureError(f"development seeds overlap original dataset seed.txt: {sorted(overlap)}")
    assets = json.loads(ASSET_RECEIPT.read_text())
    if assets.get("status") != "complete":
        raise InfrastructureError("downloaded asset receipt is not complete")
    asset_links = {str(p): str(p.resolve()) for p in (ROBOTWIN / "assets").iterdir() if p.is_symlink()}
    for path in asset_links.values():
        if not Path(path).is_dir():
            raise InfrastructureError(f"missing downloaded asset target: {path}")
    inventory = []
    # Check extracted inventory path/size, not a claim of fresh full asset hashes.
    for archive in assets["archives"].values():
        if not archive.get("file_inventory"):
            continue
        file = Path(archive["file_inventory"])
        inventory.append(identity(file))
        for line in file.read_text().splitlines():
            item = json.loads(line)
            path = Path(assets["asset_root"]) / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise InfrastructureError(f"asset inventory size/file mismatch: {path}")
    source_files = [ROOT / "harness/robotwin_policy_worker.py", ROOT / "harness/robotwin_eval_evidence.py", ROOT / "harness/robotwin_robot_feedback.py", ROOT / "harness/robotwin_wrist_target.py", ROOT / "harness/robotwin_camera_readback.py",
        ROOT / "harness/robotwin_vtt_runtime.py", ROOT / "harness/robotwin_vtt_adapter.py", ROOT / "harness/robotwin_condition_memory.py", ROOT / "harness/robotwin_middle_system.py", ROOT / "harness/core.py",
        ROOT / "harness/robotwin_qpos_adapter.py", ROOT / "harness/robotwin_native_diagnostics.py", ROOT / "harness/robotwin_onboard_recorder.py", ROOT / "harness/robotwin_system_adapter.py", ROOT / "harness/robotwin_observation_schedule.py", ROOT / "harness/robotwin_integrated_system.py", ROOT / "harness/robotwin_integrated_vlm.py", ROOT / "harness/robotwin_atomic_visual.py", ROOT / "harness/robotwin_vlm_worker.py", Path(__file__)] + sorted((ROBOTWIN / "envs").rglob("*.py"))
    vtt_artifacts = {}
    if getattr(args, 'vtt_adapter', None):
        from harness.robotwin_vtt_runtime import adapter_metadata
        _, vtt_artifacts['adapter'] = adapter_metadata(args.vtt_adapter, receipt['sha256'])
    if getattr(args, 'vtt_bank', None):
        vtt_artifacts['bank'] = [identity(Path(args.vtt_bank) / name) for name in
            ('prototypes.npy', 'result.json', 'meta.json', 'action_vtt_prototype.npy', 'vtt_instances.npy', 'context_cls.npy')]
    return {"status": "cpu_plan_ready_not_physical_validation", "task": task, "task_config": config_name,
        "native_success_predicate": f"envs.{task}.{task}.check_success",
        "model_constructed": False, "simulator_started": False, "cuda_initialized": False,
        "checkpoint": identity(args.checkpoint, content=False), "checkpoint_receipt": receipt,
        "advisory_target_head": identity(args.target_head) if getattr(args,"target_head",None) else None,
        "vtt_artifacts": vtt_artifacts,
        "camera_readback": getattr(args,"camera_readback","native"),
        "checkpoint_assurance": "plan: receipt+stat only; worker verifies fresh SHA256 and strict tensors before ready",
        "training_manifest": identity(Path(args.training_run_dir) / "manifest.json"), "contract_sha256": digest(contract),
        "train_stats": {k: identity(Path(args.training_run_dir) / f"{k}.json") for k in stats},
        "cache_config": identity(contract["cache_config"]["path"]), "encoder_revision": cache["encoder_revision"],
        "system": {"mode": getattr(args, "system_mode", "pure"), "full_system_verified": False, "visible_verifier_admitted": False, "memory_writes_enabled": False, "execution_mode": getattr(args, "execution_mode", "native8")},
        "onboard_archive": {"enabled": bool(getattr(args, "record_onboard", False)), "lossless_native_policy_inputs": True, "settle_extra_rgb_reads": bool(getattr(args, "record_onboard", False))},
        "seeds": args.seeds, "seed_files": [identity(p) for p in sorted(seed_files)],
        "seed_overlap": [], "renderer": {"shader": "rt", "samples_per_pixel": 32, "path_depth": 8,
            "denoiser": config["render_denoiser"], "native_default": "oidn",
            "protocol_deviation": config["render_denoiser"] != "oidn"}, "task_kwargs": config, "config_files": [identity(p) for p in config_files],
        "source_files": [identity(p) for p in source_files], "asset_receipt": identity(ASSET_RECEIPT),
        "asset_revision": assets["revision"], "asset_inventories": inventory, "asset_links": asset_links,
        "ffmpeg": identity(resolve_ffmpeg(args.ffmpeg)),
        "asset_assurance": "existing downloaded SHA/CRC receipts + live path/size checks; no fresh full asset hash",
        "limits": {"control_steps": args.max_steps, "episode_wall_seconds": args.episode_wall_seconds,
            "total_wall_seconds": args.total_wall_seconds, "policy_timeout_seconds": args.policy_timeout,
            "startup_timeout_seconds": args.startup_timeout, "settle_physics_steps": args.settle_steps},
        "policy": {"views": ["head_camera", "left_camera", "right_camera"], "state_dim": 14,
            "state_semantics": "commanded drive targets + stored gripper commands, not measured qpos",
            "K": 1, "euler_steps": 10, "chunk": 32, "execute_prefix": 8, "gt_control": False, "supervised_actions": False,
            "vlm_enabled": getattr(args, "system_mode", "pure") == "integrated",
            "recovery_enabled": getattr(args, "system_mode", "pure") != "pure", "task_cond": 'trained_VTT_residual' if getattr(args, 'vtt_adapter', None) else 0,
            "history": "encode three views and push state on every control observation, including queued actions",
            "bounds": "robot native joint limits + gripper [0,1], raw/bounded/mask logged; offline metrics unchanged"},
        "protocol_deviations": ["fixed development seeds, no expert filtering and no replacement of failed instances",
            "K1/10Euler/chunk32/exec8; not a claim of official benchmark sampling recipe",
            "explicit physical action bounds (raw values retained); original TOPP/gripper control unchanged",
            "1080p third-view camera added only for evaluation; framing not yet GPU-verified",
            "video nominal FPS is not scene time; one frame per control observation plus optional settle frames",
            "native predicate is not full containment, verified release, competition success, or a 5-second hold"],
        "gpu_validation_pending": ["strict real checkpoint inference and history", "native Scene joint-target rollout",
            "cross-environment ABI end-to-end", "1080p camera framing/render/video", "GPU memory coexistence of policy and sim",
            "real Scene contact reporting and target/robot identity mapping (static interfaces and CPU stubs only so far)"]}


def make_client(args, out):
    validate_pure(args)
    extra=["--target-head",str(Path(args.target_head).resolve())] if getattr(args,"target_head",None) else []
    for name in ('vtt_adapter', 'vtt_bank'):
        if getattr(args, name, None):
            extra.extend(['--' + name.replace('_', '-'), str(Path(getattr(args, name)).resolve())])
    return PolicyClient([str(ROOT / ".venvs/wam-fixed/bin/python"), "-B", str(ROOT / "harness/robotwin_policy_worker.py"),
        "--training-run-dir", str(Path(args.training_run_dir).resolve()), "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--gpu-lock-owner", args.gpu_lock_owner, "--execution-mode", getattr(args, "execution_mode", "native8"),*extra], out / "policy_worker.log", args.startup_timeout, args.policy_timeout)


def make_env(task):
    # First called AFTER successful worker-ready. No simulator import above here.
    validate_task(task)
    import numpy as np
    if np.__version__ != "1.26.4":
        raise InfrastructureError("real simulator requires robotwin-sim NumPy1.26.4; policy preprocessing stays in wam-fixed")
    sys.path.insert(0, str(ROBOTWIN))
    import torch
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(2)
    env = getattr(importlib.import_module(f"envs.{task}"), task)()
    if not callable(getattr(env, "check_success", None)):
        raise InfrastructureError(f"native task has no success predicate: {task}")
    return env


def make_video(env, directory, episode, seed, fps, ffmpeg=None):
    camera, metadata = make_camera(env)
    return VideoRecorder(directory, episode, seed, metadata, fps=fps, ffmpeg=ffmpeg), camera


def close_env(env):
    if env is None:
        return
    try:
        env.close_env()
    finally:
        viewer = getattr(env, "viewer", None)
        if viewer is not None:
            viewer.close()
        robot = getattr(env, "robot", None)
        if robot is not None and hasattr(robot, "close"):
            robot.close()


def run_evaluation(args, plan, out, *, client_factory=make_client, env_factory=make_env,
                   video_factory=make_video, owner_check=verify_gpu_owner, guard_factory=ResourceGuard, journal=None):
    """Dependency injection is test-only; production has no mock success path."""
    validate_pure(args)
    task = plan.get("task")
    import numpy as np
    from harness.robotwin_qpos_adapter import NativeQposAdapter, policy_observation
    out = Path(out)
    journal = journal or PhaseJournal(out)
    rows = [{"episode": f"episode_{i:03d}_seed{seed}", "seed": seed, "status": "not_started",
             "control_steps": 0, "native_take_action_count": None, "evidence": task_evidence(task)}
            for i, seed in enumerate(args.seeds)]
    summary = {"status": "running", "task": task, "episodes": rows, "policy_ready": None,
               "native_successes": 0, "scored_episodes": 0, "competition_success": None}
    client, visual, current = None, None, rows[0]
    guard = guard_factory()
    stop = RunStop(args.total_wall_seconds)
    total_deadline = stop.deadline
    phase = "policy_startup"
    try:
        stop.install()
        journal.call("gpu_owner_check", owner_check, args.gpu_lock_owner)
        journal.call("resource_check", guard.check)
        remaining = total_deadline - time.monotonic()
        if remaining <= 0:
            raise WallTimeout("total wall limit before policy startup")
        startup_args = copy.copy(args)
        startup_args.startup_timeout = min(args.startup_timeout, remaining)
        summary["policy_startup_timeout_applied"] = startup_args.startup_timeout
        with wall_limit(remaining, stop):
            client = journal.call("policy_startup", client_factory, startup_args, out)
            stop.check()
        summary["policy_ready"] = client.ready
        runtime_task = client.ready.get("runtime", {}).get("task")
        if runtime_task is not None and runtime_task != task:
            raise InfrastructureError("worker runtime task differs from evaluation task")
        if getattr(args, "system_mode", "pure") == "integrated":
            from harness.robotwin_policy_worker import PolicyClient
            visual = journal.call("visual_startup", PolicyClient,
                [str(ROOT / ".venvs/wam-fixed/bin/python"), "-B", str(ROOT / "harness/robotwin_integrated_vlm.py"),
                 "--gpu-lock-owner", args.gpu_lock_owner], out / "visual_worker.log", 180, 60)
            summary["visual_ready"] = visual.ready
        for index, row in enumerate(rows):
            current = row
            journal.context(row["episode"], row["seed"], 0)
            phase = "episode_reset"
            env = video = camera = adapter = raw_obs = onboard = system_adapter = None
            started = time.monotonic()
            directory = out / row["episode"]
            directory.mkdir(exist_ok=False)
            row["status"] = "starting"
            journal.call("result_write", atomic_json, out / "result.json", summary)
            try:
                remaining = min(args.episode_wall_seconds, total_deadline - time.monotonic())
                if remaining <= 0:
                    raise WallTimeout("total wall limit")
                stop.episode_deadline = time.monotonic() + remaining
                with wall_limit(remaining, stop):
                    journal.call("gpu_owner_check", owner_check, args.gpu_lock_owner)
                    journal.call("resource_check", guard.check)
                    if getattr(args, "system_mode", "pure") in ("transparent", "integrated", "middle"):
                        from harness.robotwin_system_adapter import RobotwinSystemAdapter
                        def dispatch_once(raw_action):
                            previous = int(env.take_action_cnt)
                            adapter.step(bound_action(raw_action, limits)["bounded_action"])
                            return int(env.take_action_cnt) == previous + 1
                        if getattr(args, "system_mode", "pure") == "integrated":
                            from harness.robotwin_integrated_system import IntegratedSystem
                            def refresh_visible():
                                fresh = env.get_obs()
                                return policy_observation(fresh), time.monotonic()
                            system_adapter = IntegratedSystem(client, dispatch_once, visual=visual,
                                robot=lambda: env.robot, refresh=refresh_visible, out=directory, episode=row["episode"])
                        elif args.system_mode == 'middle':
                            from harness.robotwin_middle_system import MiddleSystem
                            def refresh_visible():
                                fresh = env.get_obs()
                                return policy_observation(fresh), time.monotonic()
                            system_adapter = MiddleSystem(client, dispatch_once,
                                robot=lambda: env.robot, refresh=refresh_visible, out=directory,
                                episode=row['episode'], with_vtt=bool(getattr(args, 'vtt_adapter', None)),
                                task=task, checkpoint_sha256=client.ready.get('runtime', {}).get('checkpoint_sha256'))
                        else:
                            system_adapter = RobotwinSystemAdapter(client, dispatch_once, max_age_s=30)
                        journal.call("policy_reset", stop.call, system_adapter.reset, row["episode"], row["seed"])
                    else:
                        journal.call("policy_reset", stop.call, client.reset, row["episode"], row["seed"])
                    phase = "scene_setup"
                    random.seed(row["seed"])
                    np.random.seed(row["seed"])
                    env = journal.call("scene_create", env_factory, task) if env_factory is make_env else journal.call("scene_create", env_factory)
                    stop.check()
                    # Preserve the original setup/physics/success, not expert play_once.
                    journal.call("scene_setup", stop.call, env.setup_demo, now_ep_num=index, seed=row["seed"], is_test=True, **copy.deepcopy(plan["task_kwargs"]))
                    if getattr(args,"camera_readback","native")=="serial":
                        from harness.robotwin_camera_readback import SerialColorReadback
                        env._serial_color_readback=SerialColorReadback(env.cameras)
                    journal.call("policy_check_alive", client.check_alive)
                    limits, limit_records = journal.call("joint_bounds", joint_bounds, env.robot)
                    row["joint_bounds"] = {"source": "native robot joint.get_limits (URDF loaded)", "joints": limit_records}
                    adapter = journal.call("adapter_setup", NativeQposAdapter, env, limits)
                    row["native_step_limit"] = env.step_lim
                    row["status"] = "running"
                    phase = "recording_setup"
                    video, camera = journal.call("video_setup", video_factory, env, directory, row["episode"], row["seed"], args.video_fps,
                                                  plan.get("ffmpeg", {}).get("path"))
                    stop.check()
                    raw_obs = journal.call("get_obs", stop.call, env.get_obs)
                    observed_at = time.monotonic()
                    journal.call("video_capture", stop.call, video.capture, env, camera, 0)
                    # Save exactly the allowlisted policy images; evaluation artifacts only.
                    import cv2
                    initial_allowed = policy_observation(raw_obs)
                    if getattr(args, "record_onboard", False):
                        from harness.robotwin_onboard_recorder import OnboardRecorder
                        onboard = journal.call("onboard_setup", OnboardRecorder, directory, row["episode"], row["seed"])
                        journal.call("onboard_write", stop.call, onboard.append, initial_allowed, 0)
                    image_stats = {}
                    for name, image in initial_allowed["rgb"].items():
                        if not journal.call("initial_image_write", cv2.imwrite, str(directory / f"initial_{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                            raise InfrastructureError(f"failed to save initial policy image: {name}")
                        image_stats[name] = {"shape": list(image.shape), "min": int(image.min()),
                            "max": int(image.max()), "std": float(image.std())}
                    journal.call("episode_artifact_write", atomic_json, directory / "initial_observation.json", {"images": image_stats,
                        "commanded_state": initial_allowed["commanded_joint_gripper_state"].tolist()})
                    initial = journal.call("gt_snapshot", stop.call, task_snapshot, env, task)
                    journal.call("gt_record_write", jsonl, directory / "evaluation_gt.jsonl", {"control_step": 0, **initial})
                    phase = "initial_validation"
                    if bool(journal.call("gt_check_success", stop.call, env.check_success)):
                        row["initialization_invalid"] = True
                        raise InfrastructureError("initial state already satisfies native goal; no policy credit")
                    phase = "rollout"
                    cap = min(args.max_steps, int(env.step_lim))
                    if cap < 1:
                        raise InfrastructureError("native task step limit must be positive")
                    for step in range(cap):
                        journal.context(row["episode"], row["seed"], step)
                        stop.check()
                        journal.call("gpu_owner_check", owner_check, args.gpu_lock_owner)
                        resources = journal.call("resource_check", guard.check)
                        start_step = time.monotonic()
                        # The ONLY parent-to-worker observation handoff.
                        allowed = policy_observation(raw_obs)
                        observation_id = f"{row['episode']}:observation:{step}"
                        if system_adapter is not None:
                            reply = journal.call("system_prepare", stop.call, system_adapter.prepare, row["episode"], step, observation_id, observed_at, allowed)
                        else:
                            reply = journal.call("policy_action", stop.call, client.action, row["episode"], step, allowed)
                        record = bound_action(reply["raw_action"], limits)
                        record.update(episode=row["episode"], step=step, policy=reply,
                                      resources=resources, status="submitted")
                        journal.call("action_record_write", jsonl, directory / "actions.jsonl", record)
                        before = int(env.take_action_cnt)
                        if system_adapter is not None:
                            ack = journal.call("system_execute", stop.call, system_adapter.execute, row["episode"], step, observation_id)
                            journal.call("system_event_write", jsonl, directory / "system_events.jsonl", {"episode": row["episode"], "step": step,
                                "observation_id": observation_id, "driver_ack": ack, "visible_outcome": None,
                                "mode": args.system_mode, "memory_written": False, "full_system": False})
                        else:
                            journal.call("native_action", stop.call, adapter.step, record["bounded_action"])
                        after = int(env.take_action_cnt)
                        row["control_steps"] = step + 1
                        row["native_take_action_count"] = after
                        if after != before + 1:
                            raise InfrastructureError("native action was not consumed exactly once")
                        record.update(status="native_returned", native_take_action_count=after,
                            topp=adapter.last_topp_diagnostics, control_seconds=time.monotonic() - start_step,
                            topp_fallback_observed=any(r["status"] in ("exception", "empty", "unrecognized_result")
                                                      for r in adapter.last_topp_diagnostics),
                            fallback_semantics="old arm targets + native gripper fallback, NOT safety stop or success")
                        journal.call("action_record_write", jsonl, directory / "actions.jsonl", record)
                        raw_obs = journal.call("get_obs", stop.call, env.get_obs)
                        observed_at = time.monotonic()
                        if onboard is not None:
                            journal.call("onboard_write", stop.call, onboard.append, policy_observation(raw_obs), step + 1)
                        last = journal.call("gt_snapshot", stop.call, task_snapshot, env, task)
                        journal.call("gt_record_write", jsonl, directory / "evaluation_gt.jsonl", {"control_step": step + 1, **last})
                        journal.call("video_capture", stop.call, video.capture, env, camera, step + 1)
                        journal.call("policy_check_alive", client.check_alive)
                        if env.eval_success:
                            break
                    stop.check()
                    row["evidence"].update(native_benchmark_success=bool(env.eval_success),
                        native_terminal_predicate=bool(journal.call("gt_check_success", stop.call, env.check_success)),
                        object_pose_world=last["object_pose_world"])
                    if "cabinet_joint_qpos" in last:
                        row["evidence"]["cabinet_joint_qpos"] = last["cabinet_joint_qpos"]
                    row["status"] = "native_success" if env.eval_success else "step_limit"
                    # Optional fixed-step diagnostic does not change commands or award success.
                    phase = "settle_diagnostic"
                    poses = [last["object_pose_world"]]
                    for settle in range(args.settle_steps):
                        stop.check()
                        journal.call("gpu_owner_check", owner_check, args.gpu_lock_owner)
                        journal.call("resource_check", guard.check)
                        journal.call("policy_check_alive", client.check_alive)
                        journal.call("settle_scene_step", stop.call, env.scene.step)
                        if onboard is not None:
                            settle_obs = journal.call("settle_get_obs", stop.call, env.get_obs)
                            journal.call("onboard_write", stop.call, onboard.append, policy_observation(settle_obs), row["control_steps"], settle + 1)
                        last = journal.call("gt_snapshot", stop.call, task_snapshot, env, task)
                        poses.append(last["object_pose_world"])
                        journal.call("gt_record_write", jsonl, directory / "evaluation_gt.jsonl", {"control_step": row["control_steps"],
                            "settle_step": settle + 1, **last})
                        journal.call("video_capture", stop.call, video.capture, env, camera, row["control_steps"], settle + 1)
                    row["evidence"]["native_predicate_after_settle"] = bool(journal.call("gt_check_success", stop.call, env.check_success))
                    diagnostic = settle_diagnostic(poses, args.settle_steps, args.settle_steps)
                    row["settle_diagnostic"] = diagnostic
                    row["evidence"]["stable_after_settle"] = diagnostic["stable_after_settle"]
                    # Final snapshot only: no temporal hold/support claim and no
                    # changes to native scoring, commands or policy observations.
                    row["evidence"]["contact_evidence"] = last["contact_evidence"]
                    row["evidence"]["release_verified"] = last["contact_evidence"]["release_verified"]
                    row["post_settle_snapshot"] = last
                    phase = "video_finalize"
                    journal.call("video_finalize", stop.call, video.close, complete=True)
                    if onboard is not None:
                        journal.call("onboard_finalize", stop.call, onboard.close, complete=True)
            except BaseException as exc:
                if journal.state["status"] != "error":
                    journal.failed(exc)
                row.update(status=("wall_timeout" if isinstance(exc, WallTimeout) else
                                   "cancelled" if isinstance(exc, EvaluationCancelled) else "infrastructure_error"),
                           failure_phase=phase, error=f"{type(exc).__name__}: {exc}")
                if env is not None:
                    row["native_take_action_count"] = getattr(env, "take_action_cnt", None)
                if (type(exc).__name__ == "UnStableError" and phase == "scene_setup") or row.get("initialization_invalid"):
                    row["initialization_invalid"] = True
                    row["status"] = "instance_setup_failed"
                    # This exact seed is retained; never replace it with an easier seed.
                else:
                    # Unscored errors remain null even if a partial native latch was seen.
                    row["native_observed_before_error"] = row["evidence"]["native_benchmark_success"]
                    row["evidence"]["native_benchmark_success"] = None
                    raise
            finally:
                cleanup_errors = []
                if getattr(args, "system_mode", "pure") == "integrated" and system_adapter is not None:
                    row["integrated_system"] = system_adapter.report
                    system_adapter.close()
                if getattr(args, 'system_mode', 'pure') == 'middle' and system_adapter is not None:
                    row['middle_system'] = system_adapter.report
                    system_adapter.close()
                if onboard is not None and not onboard.manifest["complete"]:
                    try:
                        journal.call("cleanup_onboard", onboard.close, complete=False)
                    except BaseException as exc:
                        cleanup_errors.append(f"onboard: {type(exc).__name__}: {exc}")
                if video is not None:
                    try:
                        journal.call("cleanup_video", video.close, complete=False)
                    except BaseException as exc:
                        cleanup_errors.append(f"video: {type(exc).__name__}: {exc}")
                try:
                    journal.call("cleanup_scene", close_env, env)
                except BaseException as exc:
                    cleanup_errors.append(f"scene: {type(exc).__name__}: {exc}")
                # The native close_env currently delegates to gym.Env.close;
                # release adapter/camera references too before the next Scene.
                adapter = camera = raw_obs = env = None
                import gc
                journal.call("cleanup_gc", gc.collect)
                # Even a swallowed cleanup signal/deadline cannot leave a scored episode.
                try:
                    stop.check()
                except (WallTimeout, EvaluationCancelled) as exc:
                    journal.failed(exc)
                    row.update(status="wall_timeout" if isinstance(exc, WallTimeout) else "cancelled",
                               failure_phase=phase, error=f"{type(exc).__name__}: {exc}")
                    row["evidence"]["native_benchmark_success"] = None
                row["wall_seconds"] = time.monotonic() - started
                if cleanup_errors:
                    row.update(status="infrastructure_error", cleanup_errors=cleanup_errors)
                    row["evidence"]["native_benchmark_success"] = None
                journal.call("episode_artifact_write", atomic_json, directory / "episode.json", row)
                journal.write("episode_finished", "completed" if row["status"] in
                              ("native_success", "step_limit") else "error",
                              episode_status=row["status"])
                if cleanup_errors:
                    raise InfrastructureError("episode cleanup failed; remaining seeds not started")
            stop.check()
            stop.episode_deadline = None
        stop.check()
        summary["status"] = "completed"
    except BaseException as exc:
        if not journal.state["failures"]:
            journal.failed(exc)
        if current["status"] in ("not_started", "starting", "running"):
            current.update(status="infrastructure_error", failure_phase=phase, error=f"{type(exc).__name__}: {exc}")
        summary.update(status="infrastructure_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if visual is not None:
            visual.close()
        if client is not None:
            try:
                journal.call("cleanup_policy", client.close)
            except BaseException as exc:
                summary.update(status="infrastructure_error", cleanup_error=str(exc))
        stop.restore()
        summary["cancellation_reason"] = str(stop.reason) if stop.reason else None
        summary["peak_resources"] = guard.peaks
        summary["scored_episodes"] = sum(r["evidence"]["native_benchmark_success"] is not None for r in rows)
        summary["native_successes"] = sum(r["evidence"]["native_benchmark_success"] is True for r in rows)
        summary["attempted_episodes"] = sum(r["status"] != "not_started" for r in rows)
        summary["invalid_initializations"] = sum(r.get("initialization_invalid", False) for r in rows)
        summary["native_failures"] = sum(r["evidence"]["native_benchmark_success"] is False for r in rows)
        summary["incomplete_episodes"] = sum(r["status"] not in ("native_success", "step_limit", "not_started", "instance_setup_failed") for r in rows)
        summary["not_started"] = sum(r["status"] == "not_started" for r in rows)
        summary["native_success_rate"] = (summary["native_successes"] / len(rows)
            if summary["scored_episodes"] == len(rows) else None)
        journal.call("result_write", atomic_json, out / "result.json", summary)
        journal.write("run_finished", "completed" if summary["status"] == "completed" else "error",
                      result_status=summary["status"])
    return summary


def smoke_plan(args):
    import yaml
    cfg = yaml.safe_load(Path(args.training_config).read_text())
    task = validate_task(cfg["dataset"]["task_name"])
    recorded = cfg["dataset"].get("task_config")
    if args.task_config and recorded and args.task_config != recorded:
        raise InfrastructureError("task config differs from training provenance")
    name = args.task_config or recorded or "demo_randomized"
    config, files = load_task_config(name, task)
    config["render_denoiser"] = args.render_denoiser
    return {"status": "native_first_frame_plan_only", "task": task, "task_config": name,
            "task_kwargs": config, "config_files": [identity(p) for p in files],
            "training_config": identity(args.training_config), "policy_constructed": False,
            "control_actions": 0, "scored_episodes": 0, "seeds": args.seeds}


def run_first_frame(args, plan, out, *, env_factory=make_env, owner_check=verify_gpu_owner,
                    guard_factory=ResourceGuard, journal=None):
    """Native observation smoke only: no worker, policy, actions, expert or GT control."""
    import numpy as np
    import cv2
    from harness.robotwin_qpos_adapter import policy_observation
    journal = journal or PhaseJournal(out)
    stop, guard, env = RunStop(args.total_wall_seconds), guard_factory(), None
    result = dict(status="infrastructure_error", task=plan["task"], policy_constructed=False,
                  control_actions=0, scored_episodes=0, native_success_rate=None, seed=args.seeds[0])
    try:
        stop.install()
        with wall_limit(args.total_wall_seconds, stop):
            journal.call("gpu_owner_check", owner_check, args.gpu_lock_owner)
            journal.call("resource_check", guard.check)
            random.seed(args.seeds[0])
            np.random.seed(args.seeds[0])
            env = journal.call("scene_create", env_factory, plan["task"])
            journal.call("scene_setup", stop.call, env.setup_demo, now_ep_num=0, seed=args.seeds[0],
                         is_test=True, **copy.deepcopy(plan["task_kwargs"]))
            if args.camera_readback == "serial":
                from harness.robotwin_camera_readback import SerialColorReadback
                env._serial_color_readback = SerialColorReadback(env.cameras)
            allowed = policy_observation(journal.call("get_obs", stop.call, env.get_obs))
            for name, image in allowed["rgb"].items():
                if not cv2.imwrite(str(out / f"initial_{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise InfrastructureError("failed to save native first frame")
            camera, metadata = journal.call("third_view_setup", make_camera, env)
            journal.call("update_render", stop.call, env._update_render)
            journal.call("third_view_capture", stop.call, camera.take_picture)
            rgb = (journal.call("third_view_readback", stop.call, camera.get_picture, "Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
            if not cv2.imwrite(str(out / "initial_third_view.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise InfrastructureError("failed to save third-view first frame")
            result.update(status="native_first_frame_complete", camera=metadata,
                native_initial_predicate=bool(journal.call("gt_check_success", stop.call, env.check_success)),
                framing_verified=False, native_step_limit=env.step_lim)
            stop.check()
    except BaseException as exc:
        result.update(status="infrastructure_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            journal.call("cleanup_scene", close_env, env)
        except BaseException as exc:
            result.update(status="infrastructure_error", cleanup_error=str(exc))
        stop.restore()
        atomic_json(out / "result.json", result)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="new output directory, never overwritten")
    p.add_argument("--checkpoint", help="explicit immutable model-only milestone in training-run-dir")
    p.add_argument("--training-run-dir")
    p.add_argument("--first-frame-only", action="store_true", help="native scene/RGB smoke, no worker or action")
    p.add_argument("--training-config", help="YAML task source for first-frame-only before checkpoints exist")
    p.add_argument("--task-config", help="native config; defaults to training provenance or demo_randomized")
    p.add_argument("--seeds", nargs="+", type=int, default=[10000, 10001, 10002])
    p.add_argument("--render-denoiser", choices=("oidn", "none", "optix"), default="oidn",
                   help="explicit renderer compatibility choice; non-oidn is a recorded protocol deviation")
    p.add_argument("--gpu-lock-owner")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--episode-wall-seconds", type=float, default=900)
    p.add_argument("--total-wall-seconds", type=float, default=3600)
    p.add_argument("--startup-timeout", type=float, default=180)
    p.add_argument("--policy-timeout", type=float, default=30)
    p.add_argument("--settle-steps", type=int, default=0)
    p.add_argument("--system-mode", choices=("pure", "transparent", "integrated", "middle"), default="pure", help="middle excludes VLM/ASK; physical task verification remains unadmitted")
    p.add_argument("--execution-mode", choices=("native8", "contact_refresh"), default="native8")
    p.add_argument("--record-onboard", action="store_true", help="lossless policy observation archive, evaluation side only")
    p.add_argument("--target-head", help="optional appearance head; unadmitted advisory in worker replies")
    p.add_argument('--vtt-adapter', help='completed validated residual condition adapter')
    p.add_argument('--vtt-bank', help='TRAIN-only monitoring and demonstration condition bank')
    p.add_argument("--camera-readback", choices=("native","serial"), default="native")
    p.add_argument("--ffmpeg", help="optional explicit local encoder executable; never downloaded")
    p.add_argument("--video-fps", type=int, default=30, help="nominal playback FPS only, never a scene timestamp")
    args = p.parse_args(argv)
    try:
        validate_pure(args)
    except InfrastructureError as exc:
        p.error(str(exc))
    if args.first_frame_only:
        if not args.training_config or len(args.seeds) != 1 or args.system_mode != "pure":
            p.error("first-frame-only requires training-config, one seed, and pure mode")
    elif not args.checkpoint or not args.training_run_dir:
        p.error("evaluation requires checkpoint and training-run-dir")
    if args.vtt_adapter and not args.vtt_bank:
        p.error('trained VTT adapter requires its TRAIN bank')
    if not args.plan_only and not args.gpu_lock_owner:
        p.error("real evaluation requires --gpu-lock-owner held externally after training ends")
    if (not 1 <= len(args.seeds) <= 100 or len(set(args.seeds)) != len(args.seeds)
            or any(not 0 <= s < 2**32 for s in args.seeds)):
        p.error("seeds must be 1..100 unique uint32 values")
    if not 1 <= args.max_steps <= 2000 or not 0 <= args.settle_steps <= 500:
        p.error("max-steps must be [1,2000]; settle-steps [0,500]")
    for name in ("episode_wall_seconds", "total_wall_seconds", "startup_timeout", "policy_timeout"):
        if not 0 < getattr(args, name) <= 86400:
            p.error(f"invalid {name}")
    if not 1 <= args.video_fps <= 60:
        p.error("video FPS must be [1,60]")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Bind caller-relative input paths before entering RoboTwin's native cwd.
    for name in ("checkpoint", "training_run_dir", "training_config"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if args.target_head:
        args.target_head = str(Path(args.target_head).resolve())
    for name in ('vtt_adapter', 'vtt_bank'):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if args.ffmpeg:
        args.ffmpeg = str(Path(args.ffmpeg).resolve())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    out = Path(args.run_dir).resolve()
    # Intentional failure outside the catch: never touch any existing output.
    out.mkdir(parents=True, exist_ok=False)
    journal = PhaseJournal(out)
    try:
        plan = journal.call("preflight_plan", smoke_plan if args.first_frame_only else plan_evaluation, args)
        atomic_json(out / "plan.json", plan)
        if args.plan_only:
            journal.write("plan_finished", "completed")
            print(json.dumps({"status": plan["status"], "run_dir": str(out), "model_constructed": False,
                              "simulator_started": False, "seeds": args.seeds}, allow_nan=False))
            return 0
        journal.call("preflight_gpu_owner", verify_gpu_owner, args.gpu_lock_owner)
        # Native source uses a few relative paths; only the dedicated evaluator changes cwd.
        os.chdir(ROBOTWIN)
        result = (run_first_frame if args.first_frame_only else run_evaluation)(args, plan, out, journal=journal)
        print(json.dumps({"status": result["status"], "run_dir": str(out),
                          "native_success_rate": result["native_success_rate"]}, allow_nan=False))
        return 0 if result["status"] in ("completed", "native_first_frame_complete") else 2
    except Exception as exc:
        if journal.state["status"] != "error":
            journal.failed(exc)
        result = {"status": "infrastructure_error", "phase": "preflight", "error": f"{type(exc).__name__}: {exc}",
            "episodes": [{"seed": s, "status": "not_started", "evidence": empty_evidence()} for s in args.seeds]}
        atomic_json(out / "result.json", result)
        print(json.dumps(result, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
