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
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.robotwin_policy_worker import (TASK, InfrastructureError, PolicyClient, atomic_json,
    digest, identity, sha256, training_contract, verify_gpu_owner)
from harness.robotwin_eval_evidence import (ResourceGuard, VideoRecorder, WallTimeout, bound_action,
    empty_evidence, joint_bounds, jsonl, make_camera, resolve_ffmpeg, settle_diagnostic, snapshot, wall_limit)

ROBOTWIN = ROOT / "third_party/RoboTwin"
CONFIG_DIR = ROBOTWIN / "env_cfg/task_config"
ASSET_RECEIPT = ROOT / "outputs/validation/robotwin_assets_20261005/asset_manifest.json"


def load_task_config(name):
    import yaml
    if name != "demo_randomized":
        raise InfrastructureError("this checkpoint bridge requires original demo_randomized")
    config = yaml.safe_load((CONFIG_DIR / f"{name}.yml").read_text())
    registry = yaml.safe_load((CONFIG_DIR / "_embodiment_config.yml").read_text())
    if config["embodiment"] != ["aloha-agilex"]:
        raise InfrastructureError("native14 bridge expects original dual-arm aloha-agilex")
    robot_dir = (ROBOTWIN / registry["aloha-agilex"]["file_path"]).resolve()
    embodiment = yaml.safe_load((robot_dir / "config.yml").read_text())
    files = [CONFIG_DIR / f"{name}.yml", CONFIG_DIR / "_embodiment_config.yml", CONFIG_DIR / "_camera_config.yml",
             CONFIG_DIR / "_eval_step_limit.yml", robot_dir / "config.yml", robot_dir / embodiment["urdf_path"]]
    config.update(task_name=TASK, task_config=name, eval_mode=True, planner_mode="joint_targets",
        left_robot_file=str(robot_dir), right_robot_file=str(robot_dir), dual_arm_embodied=True,
        left_embodiment_config=copy.deepcopy(embodiment), right_embodiment_config=copy.deepcopy(embodiment),
        render_freq=0, save_data=False, collect_data=False, eval_video_save_dir=None)
    # Keep original policy RGB + qpos flags. No extra modality enters the worker.
    if not config["data_type"]["rgb"] or not config["data_type"]["qpos"]:
        raise InfrastructureError("native RGB/qpos observation disabled")
    return config, files


def plan_evaluation(args):
    contract, cache, stats, receipt = training_contract(args.training_run_dir, args.checkpoint)
    config, config_files = load_task_config(args.task_config)
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
    source_files = [ROOT / "harness/robotwin_policy_worker.py", ROOT / "harness/robotwin_eval_evidence.py",
        ROOT / "harness/robotwin_qpos_adapter.py", Path(__file__)] + sorted((ROBOTWIN / "envs").rglob("*.py"))
    return {"status": "cpu_plan_ready_not_physical_validation", "task": TASK, "task_config": args.task_config,
        "model_constructed": False, "simulator_started": False, "cuda_initialized": False,
        "checkpoint": identity(args.checkpoint, content=False), "checkpoint_receipt": receipt,
        "checkpoint_assurance": "plan: receipt+stat only; worker verifies fresh SHA256 and strict tensors before ready",
        "training_manifest": identity(Path(args.training_run_dir) / "manifest.json"), "contract_sha256": digest(contract),
        "train_stats": {k: identity(Path(args.training_run_dir) / f"{k}.json") for k in stats},
        "cache_config": identity(contract["cache_config"]["path"]), "encoder_revision": cache["encoder_revision"],
        "seeds": args.seeds, "seed_files": [identity(p) for p in sorted(seed_files)],
        "seed_overlap": [], "task_kwargs": config, "config_files": [identity(p) for p in config_files],
        "source_files": [identity(p) for p in source_files], "asset_receipt": identity(ASSET_RECEIPT),
        "asset_revision": assets["revision"], "asset_inventories": inventory, "asset_links": asset_links,
        "ffmpeg": identity(resolve_ffmpeg(args.ffmpeg)),
        "asset_assurance": "existing downloaded SHA/CRC receipts + live path/size checks; no fresh full asset hash",
        "limits": {"control_steps": args.max_steps, "episode_wall_seconds": args.episode_wall_seconds,
            "total_wall_seconds": args.total_wall_seconds, "policy_timeout_seconds": args.policy_timeout,
            "startup_timeout_seconds": args.startup_timeout, "settle_physics_steps": args.settle_steps},
        "policy": {"views": ["head_camera", "left_camera", "right_camera"], "state_dim": 14,
            "state_semantics": "commanded drive targets + stored gripper commands, not measured qpos",
            "K": 1, "euler_steps": 10, "chunk": 32, "execute_prefix": 8, "task_cond": 0,
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
    return PolicyClient([str(ROOT / ".venvs/wam-fixed/bin/python"), "-B", str(ROOT / "harness/robotwin_policy_worker.py"),
        "--training-run-dir", str(Path(args.training_run_dir).resolve()), "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--gpu-lock-owner", args.gpu_lock_owner], out / "policy_worker.log", args.startup_timeout, args.policy_timeout)


def make_env():
    # First called AFTER successful worker-ready. No simulator import above here.
    import numpy as np
    if np.__version__ != "1.26.4":
        raise InfrastructureError("real simulator requires robotwin-sim NumPy1.26.4; policy preprocessing stays in wam-fixed")
    sys.path.insert(0, str(ROBOTWIN))
    import torch
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(2)
    return getattr(importlib.import_module(f"envs.{TASK}"), TASK)()


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
                   video_factory=make_video, owner_check=verify_gpu_owner, guard_factory=ResourceGuard):
    """Dependency injection is test-only; production has no mock success path."""
    import numpy as np
    from harness.robotwin_qpos_adapter import NativeQposAdapter, policy_observation
    out = Path(out)
    rows = [{"episode": f"episode_{i:03d}_seed{seed}", "seed": seed, "status": "not_started",
             "control_steps": 0, "native_take_action_count": None, "evidence": empty_evidence()}
            for i, seed in enumerate(args.seeds)]
    summary = {"status": "running", "task": TASK, "episodes": rows, "policy_ready": None,
               "native_successes": 0, "scored_episodes": 0, "competition_success": None}
    client, current = None, rows[0]
    guard = guard_factory()
    total_deadline = time.monotonic() + args.total_wall_seconds
    phase = "policy_startup"
    try:
        owner_check(args.gpu_lock_owner)
        guard.check()
        remaining = total_deadline - time.monotonic()
        if remaining <= 0:
            raise WallTimeout("total wall limit before policy startup")
        startup_args = copy.copy(args)
        startup_args.startup_timeout = min(args.startup_timeout, remaining)
        summary["policy_startup_timeout_applied"] = startup_args.startup_timeout
        with wall_limit(remaining):
            client = client_factory(startup_args, out)
        summary["policy_ready"] = client.ready
        for index, row in enumerate(rows):
            current = row
            phase = "episode_reset"
            env = video = camera = adapter = raw_obs = None
            started = time.monotonic()
            directory = out / row["episode"]
            directory.mkdir(exist_ok=False)
            row["status"] = "starting"
            atomic_json(out / "result.json", summary)
            try:
                remaining = min(args.episode_wall_seconds, total_deadline - time.monotonic())
                if remaining <= 0:
                    raise WallTimeout("total wall limit")
                with wall_limit(remaining):
                    owner_check(args.gpu_lock_owner)
                    guard.check()
                    client.reset(row["episode"], row["seed"])
                    phase = "scene_setup"
                    random.seed(row["seed"])
                    np.random.seed(row["seed"])
                    env = env_factory()
                    # Preserve the original setup/physics/success, not expert play_once.
                    env.setup_demo(now_ep_num=index, seed=row["seed"], is_test=True, **copy.deepcopy(plan["task_kwargs"]))
                    client.check_alive()
                    limits, limit_records = joint_bounds(env.robot)
                    row["joint_bounds"] = {"source": "native robot joint.get_limits (URDF loaded)", "joints": limit_records}
                    adapter = NativeQposAdapter(env, limits)
                    row["native_step_limit"] = env.step_lim
                    row["status"] = "running"
                    phase = "recording_setup"
                    video, camera = video_factory(env, directory, row["episode"], row["seed"], args.video_fps,
                                                  plan.get("ffmpeg", {}).get("path"))
                    raw_obs = env.get_obs()
                    video.capture(env, camera, 0)
                    initial = snapshot(env)
                    jsonl(directory / "evaluation_gt.jsonl", {"control_step": 0, **initial})
                    if bool(env.check_success()):
                        raise InfrastructureError("initial state already satisfies native goal; no policy credit")
                    phase = "rollout"
                    cap = min(args.max_steps, int(env.step_lim))
                    for step in range(cap):
                        owner_check(args.gpu_lock_owner)
                        resources = guard.check()
                        start_step = time.monotonic()
                        # The ONLY parent-to-worker observation handoff.
                        allowed = policy_observation(raw_obs)
                        reply = client.action(row["episode"], step, allowed)
                        record = bound_action(reply["raw_action"], limits)
                        record.update(episode=row["episode"], step=step, policy=reply,
                                      resources=resources, status="submitted")
                        jsonl(directory / "actions.jsonl", record)
                        before = int(env.take_action_cnt)
                        adapter.step(record["bounded_action"])
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
                        jsonl(directory / "actions.jsonl", record)
                        raw_obs = env.get_obs()
                        last = snapshot(env)
                        jsonl(directory / "evaluation_gt.jsonl", {"control_step": step + 1, **last})
                        video.capture(env, camera, step + 1)
                        client.check_alive()
                        if env.eval_success:
                            break
                    row["evidence"].update(native_benchmark_success=bool(env.eval_success),
                        native_terminal_predicate=bool(env.check_success()),
                        object_pose_world=last["object_pose_world"], cabinet_joint_qpos=last["cabinet_joint_qpos"])
                    row["status"] = "native_success" if env.eval_success else "step_limit"
                    # Optional fixed-step diagnostic does not change commands or award success.
                    phase = "settle_diagnostic"
                    poses = [last["object_pose_world"]]
                    for settle in range(args.settle_steps):
                        owner_check(args.gpu_lock_owner)
                        guard.check()
                        client.check_alive()
                        env.scene.step()
                        last = snapshot(env)
                        poses.append(last["object_pose_world"])
                        jsonl(directory / "evaluation_gt.jsonl", {"control_step": row["control_steps"],
                            "settle_step": settle + 1, **last})
                        video.capture(env, camera, row["control_steps"], settle + 1)
                    row["evidence"]["native_predicate_after_settle"] = bool(env.check_success())
                    diagnostic = settle_diagnostic(poses, args.settle_steps, args.settle_steps)
                    row["settle_diagnostic"] = diagnostic
                    row["evidence"]["stable_after_settle"] = diagnostic["stable_after_settle"]
                    # Final snapshot only: no temporal hold/support claim and no
                    # changes to native scoring, commands or policy observations.
                    row["evidence"]["contact_evidence"] = last["contact_evidence"]
                    row["evidence"]["release_verified"] = last["contact_evidence"]["release_verified"]
                    row["post_settle_snapshot"] = last
                    phase = "video_finalize"
                    video.close(complete=True)
            except BaseException as exc:
                row.update(status="wall_timeout" if isinstance(exc, WallTimeout) else "infrastructure_error",
                           failure_phase=phase, error=f"{type(exc).__name__}: {exc}")
                if env is not None:
                    row["native_take_action_count"] = getattr(env, "take_action_cnt", None)
                if type(exc).__name__ == "UnStableError" and phase == "scene_setup":
                    row["status"] = "instance_setup_failed"
                    # This exact seed is retained; never replace it with an easier seed.
                else:
                    # Unscored errors remain null even if a partial native latch was seen.
                    row["native_observed_before_error"] = row["evidence"]["native_benchmark_success"]
                    row["evidence"]["native_benchmark_success"] = None
                    raise
            finally:
                cleanup_errors = []
                if video is not None:
                    try:
                        video.close(complete=False)
                    except BaseException as exc:
                        cleanup_errors.append(f"video: {type(exc).__name__}: {exc}")
                try:
                    close_env(env)
                except BaseException as exc:
                    cleanup_errors.append(f"scene: {type(exc).__name__}: {exc}")
                # The native close_env currently delegates to gym.Env.close;
                # release adapter/camera references too before the next Scene.
                adapter = camera = raw_obs = env = None
                import gc
                gc.collect()
                row["wall_seconds"] = time.monotonic() - started
                if cleanup_errors:
                    row.update(status="infrastructure_error", cleanup_errors=cleanup_errors)
                    row["evidence"]["native_benchmark_success"] = None
                atomic_json(directory / "episode.json", row)
                if cleanup_errors:
                    raise InfrastructureError("episode cleanup failed; remaining seeds not started")
        summary["status"] = "completed"
    except BaseException as exc:
        if current["status"] in ("not_started", "starting", "running"):
            current.update(status="infrastructure_error", failure_phase=phase, error=f"{type(exc).__name__}: {exc}")
        summary.update(status="infrastructure_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if client is not None:
            try:
                client.close()
            except BaseException as exc:
                summary.update(status="infrastructure_error", cleanup_error=str(exc))
        summary["peak_resources"] = guard.peaks
        summary["scored_episodes"] = sum(r["evidence"]["native_benchmark_success"] is not None for r in rows)
        summary["native_successes"] = sum(r["evidence"]["native_benchmark_success"] is True for r in rows)
        summary["not_started"] = sum(r["status"] == "not_started" for r in rows)
        summary["native_success_rate"] = (summary["native_successes"] / len(rows)
            if summary["scored_episodes"] == len(rows) else None)
        atomic_json(out / "result.json", summary)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="new output directory, never overwritten")
    p.add_argument("--checkpoint", required=True, help="explicit immutable model-only milestone in training-run-dir")
    p.add_argument("--training-run-dir", required=True)
    p.add_argument("--task-config", choices=("demo_randomized",), default="demo_randomized")
    p.add_argument("--seeds", nargs="+", type=int, default=[10000, 10001, 10002])
    p.add_argument("--gpu-lock-owner")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--episode-wall-seconds", type=float, default=900)
    p.add_argument("--total-wall-seconds", type=float, default=3600)
    p.add_argument("--startup-timeout", type=float, default=180)
    p.add_argument("--policy-timeout", type=float, default=30)
    p.add_argument("--settle-steps", type=int, default=0)
    p.add_argument("--ffmpeg", help="optional explicit local encoder executable; never downloaded")
    p.add_argument("--video-fps", type=int, default=30, help="nominal playback FPS only, never a scene timestamp")
    args = p.parse_args(argv)
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
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.training_run_dir = str(Path(args.training_run_dir).resolve())
    if args.ffmpeg:
        args.ffmpeg = str(Path(args.ffmpeg).resolve())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    out = Path(args.run_dir).resolve()
    # Intentional failure outside the catch: never touch any existing output.
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan = plan_evaluation(args)
        atomic_json(out / "plan.json", plan)
        if args.plan_only:
            print(json.dumps({"status": plan["status"], "run_dir": str(out), "model_constructed": False,
                              "simulator_started": False, "seeds": args.seeds}, allow_nan=False))
            return 0
        verify_gpu_owner(args.gpu_lock_owner)
        # Native source uses a few relative paths; only the dedicated evaluator changes cwd.
        os.chdir(ROBOTWIN)
        def terminate(signum, frame):
            raise InfrastructureError(f"evaluator received signal {signum}")
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        result = run_evaluation(args, plan, out)
        print(json.dumps({"status": result["status"], "run_dir": str(out),
                          "native_success_rate": result["native_success_rate"]}, allow_nan=False))
        return 0 if result["status"] == "completed" else 2
    except Exception as exc:
        result = {"status": "infrastructure_error", "phase": "preflight", "error": f"{type(exc).__name__}: {exc}",
            "episodes": [{"seed": s, "status": "not_started", "evidence": empty_evidence()} for s in args.seeds]}
        atomic_json(out / "result.json", result)
        print(json.dumps(result, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
