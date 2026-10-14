#!/usr/bin/env python3
"""Closed-loop BEHAVIOR run: WAM policy server (lilawam env) + official evaluator (Isaac 6.1 port env).

Both processes run in their own cgroup with CPU affinity; the evaluator is admission-checked and
monitored by behavior_resources.run_guarded (global RAM/CPU/VRAM/GPU-utilization abort). The policy
server is a background systemd unit that is always stopped at the end.

    python3 scripts/behavior_closed_loop.py --profile smoke --max-steps 300 --skill "move to"

Never run while training holds the GPU. Outputs land in outputs/behavior/closed_loop/<run_name>/.
"""
import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behavior_eval import command as eval_command  # noqa: E402
from behavior_preflight import CONFIG, ROOT, child_environment, collect, hardware_snapshot  # noqa: E402
from behavior_resources import run_guarded  # noqa: E402
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness import task_profile  # noqa: E402  P2_TASK_ID (default 13); passed on to the policy server

LILAWAM = ROOT / ".conda-envs/lilawam/bin/python"


def wait_for_policy_request(log_path, cancelled, timeout):
    """The official client's health request occurs after scene setup; 'Imported scene' is too early."""
    deadline = time.monotonic() + timeout
    while not cancelled.is_set() and time.monotonic() < deadline:
        if log_path.exists() and "Health check failed, waiting for server at" in log_path.read_text(errors="ignore"):
            return True
        cancelled.wait(0.25)
    return False


def wait_for_policy_capacity(cancelled, device, timeout=120, sample=hardware_snapshot, reserve_mib=None):
    """Start only with measured CUDA headroom; CPU needs no CUDA reservation. reserve_mib overrides the
    configured single-model reserve (a second manipulation checkpoint adds its own weights on top)."""
    deadline = time.monotonic() + timeout
    while not cancelled.is_set() and time.monotonic() < deadline:
        if device == "cpu":
            return True
        gpu = sample().get("gpu", [])
        if gpu:
            policy = CONFIG["resource_policy"]
            limit = min(policy["gpu_total_vram_limit_mib"],
                        gpu[0]["vram_total_mib"] * policy["monitor_exit_percent"] / 100)
            reserve = policy["policy_vram_reserve_mib"] if reserve_mib is None else reserve_mib
            if gpu[0]["vram_used_mib"] + reserve < limit:
                return not cancelled.is_set()
        cancelled.wait(0.5)
    return False


def planned_episodes(output_dir, task, instance_indices, seed, num_rollouts=1):
    """The evaluator writes its existing resolve_instance_ids mapping before constructing the environment."""
    plan = json.loads((Path(output_dir) / "eval/evaluation_plan.json").read_text())
    requested = {"task": task, "mode": "public_test", "instance_indices": instance_indices,
                 "num_rollouts": num_rollouts, "seed": seed}
    if (not isinstance(plan, dict) or type(plan.get("seed")) is not int
            or type(plan.get("num_rollouts")) is not int or any(plan.get(k) != v for k, v in requested.items())):
        raise ValueError("Evaluator plan differs from the requested task/instances/rollouts/seed")
    ids = plan.get("instance_ids")
    if (not isinstance(ids, list) or len(ids) != len(instance_indices) or not ids
            or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Invalid or duplicate resolved instance IDs in evaluator plan")
    return [{"task": task, "instance_id": i, "rollout_id": r} for i in ids for r in range(num_rollouts)]


def completed_results(output_dir, expected_episodes):
    """Require the exact planned identities, not an equal number of arbitrary scored episodes."""
    expected = {(r["task"], r["instance_id"], r["rollout_id"]) for r in expected_episodes}
    if len(expected) != len(expected_episodes):
        raise ValueError("Duplicate planned episode identities")
    results, seen = [], set()
    for path in sorted((Path(output_dir) / "eval/json").glob("*.json")):
        row = json.loads(path.read_text())
        if not isinstance(row, dict):
            raise ValueError(f"Invalid episode result: {path}")
        key = (row.get("task"), row.get("instance_id"), row.get("rollout_id"))
        score = row.get("q_score", {}).get("final") if isinstance(row.get("q_score"), dict) else None
        if (not isinstance(key[0], str) or any(type(x) is not int for x in key[1:])
                or type(row.get("success")) is not bool or type(row.get("steps")) is not int or row["steps"] <= 0
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                or key in seen or key not in expected):
            raise ValueError(f"Incomplete, duplicate or unexpected episode result: {path}")
        seen.add(key)
        results.append({**{k: row.get(k) for k in ("task", "instance_id", "rollout_id", "policy_episode", "steps", "success",
                                                "effective_policy_seed", "effective_evaluator_seed", "policy_rng_reset_behavior")},
                        "q_score": score})
    if seen != expected:
        raise ValueError(f"Missing planned episode results: {sorted(expected - seen)}")
    return results


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def checkpoint_config(path):
    """Read the embedded config, not a possibly stale YAML sidecar; never construct a model or initialize CUDA."""
    code = ("import json,sys,torch; from omegaconf import OmegaConf; "
            "cfg=torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['config']; "
            "print(json.dumps(OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)))")
    return json.loads(subprocess.check_output([str(LILAWAM), "-B", "-c", code, str(path)], text=True,
                                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}))


def checkpoint_action_mib(path):
    """fp32 action-model weight size (MiB) of a checkpoint's model_state_dict; mmap load, no CUDA, no model build."""
    code = ("import sys,torch; sd=torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['model_state_dict']; "
            "print(sum(v.numel() * v.element_size() for v in sd.values()) / 2**20)")
    return float(subprocess.check_output([str(LILAWAM), "-B", "-c", code, str(path)], text=True,
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}))


# Headroom for a second (manipulation) action model on top of the single-model reserve: its fp32 weights plus
# activations of the batched multi-view DINO pass / extra visual tokens (~30 MiB measured-by-size for 3 views).
MANIP_ACTIVATION_MARGIN_MIB = 64


def merge_manip_checkpoint(orchestrate, manip_checkpoint):
    """--manip-checkpoint is sugar for orchestrate.manip_checkpoint (the policy server option); reject conflicts."""
    if not manip_checkpoint:
        return orchestrate
    if not orchestrate:
        raise ValueError("--manip-checkpoint routes by orchestrator contract and requires --orchestrate")
    cfg = json.loads(orchestrate)
    if cfg.get("manip_checkpoint") not in (None, manip_checkpoint):
        raise ValueError("--manip-checkpoint conflicts with orchestrate.manip_checkpoint")
    cfg["manip_checkpoint"] = manip_checkpoint
    return json.dumps(cfg)


def manip_policy_reserve_mib(main_cfg, manip_cfg, manip_action_mib):
    """Configured single-model reserve + second action model; + a second DINOv3 only if the encoders differ."""
    extra = math.ceil(manip_action_mib) + MANIP_ACTIVATION_MARGIN_MIB
    main_enc, manip_enc = (ROOT / c["model"]["vision_encoder"]["checkpoint_path"] for c in (main_cfg, manip_cfg))
    if main_enc.resolve() != manip_enc.resolve():
        extra += math.ceil(sum(f.stat().st_size for f in manip_enc.glob("*.safetensors")) / 2**20)
    return CONFIG["resource_policy"]["policy_vram_reserve_mib"] + extra


def runtime_artifact_paths(args, argv, cfg, manip_cfg=None):
    """Fingerprint the files actually selected by the checkpoint and runtime gates; no model imports."""
    og = ROOT / CONFIG["source_dir"] / "OmniGibson/omnigibson"
    paths = [ROOT / p for p in (
        "scripts/behavior_closed_loop.py", "scripts/summarize_closed_loop.py", "scripts/behavior_eval.py",
        "scripts/behavior_preflight.py", "scripts/behavior_resources.py", "harness/policy_server.py",
        "harness/privileged_base_navigation.py",
        "harness/wam_behavior.py", "harness/orchestrator.py", "harness/vtt_monitor.py", "harness/docking.py",
        "harness/grounding.py", "harness/memory_vtt.py", "harness/core.py", "harness/task_profile.py",
        "configs/behavior_environment.json",
        "configs/r1pro_action_bounds.json", "third_party/LiLa-WAM/models/model_runner.py",
        "third_party/LiLa-WAM/models/vla_model_fm.py")]
    from harness.privileged_base_navigation import WRAPPERS, PREPOSITION_WRAPPER, ASSET
    if getattr(args, "env_wrapper", None) in WRAPPERS:
        paths += [ROOT / "harness/privileged_base_navigation.py", ROOT / "harness/privileged_base_rgbd_wrapper.py", ASSET]
    if getattr(args, "env_wrapper", None) == PREPOSITION_WRAPPER:
        paths += [ROOT / "harness/prepositioned_base_rgbd_wrapper.py"]
    if getattr(args, "env_wrapper", None) in ("harness.interval_base_rgbd_wrapper.IntervalBaseRGBDWrapper", *WRAPPERS):
        paths += [ROOT / "harness/interval_base_rgbd_wrapper.py", ROOT / "harness/base_encoder_velocity.py",
                  ROOT / "harness/head_rgbd_wrapper.py"]
    if os.environ.get("GPU_AUTHORIZATION"):
        from wam_v3_resource_authorization import validate_authorization
        contract = validate_authorization(os.environ["GPU_AUTHORIZATION"])
        paths += [ROOT / "scripts/wam_v3_resource_authorization.py", Path(contract["authorization"])]
    paths += [og / p for p in ("eval/eval.py", "eval/evaluator.py", "eval/utils/eval_utils.py", "eval/policies.py",
                              "eval/utils/network_utils.py", "eval/wrappers/default_wrapper.py", "metrics/task_metric.py")]
    stats = ROOT / cfg["dataset"]["stats_path"]
    provenance = json.loads(stats.read_text()).get("provenance", {})
    split = stats.parent / provenance.get("split_file", f"split_task_{task_profile.TASK_ID:03d}.json")
    paths += [Path(args.checkpoint), stats, split, Path(argv[argv.index("--robot-config") + 1])]
    vtt = task_profile.VTT_DIR  # WamPolicy uses the task's VTT_DIR (harness.task_profile), not the checkpoint's.
    paths += [vtt / p for p in ("meta.json", "prototypes.npy", "task_level.npy")]
    orch = json.loads(args.orchestrate) if args.orchestrate else {}
    detector_calibration = orch.get("perception", {}).get("detector", {}).get("calibration_path")
    if detector_calibration:
        paths.append(ROOT / detector_calibration)
    if args.program or args.orchestrate:
        paths += [ROOT / "configs" / p for p in (f"task{task_profile.TASK_ID:03d}_skill_program.json", "vtt_stall_calibration.json",
                                                "motion_stall_calibration.json")]
    if args.orchestrate:
        if orch.get("vlm", "none") == "none":
            paths.append(ROOT / orch["program"] if orch.get("program") else ROOT / "configs/task013_fixed_schedule.json")
        if orch.get("navigation") == "docking":
            # per-task docking calibration (harness/task_profile.DOCK_CALIB; Task 1 = the former fixed directory)
            calib = task_profile.DOCK_CALIB or ROOT / "outputs/validation/task1_docking_calibration_20260926"
            paths += [ROOT / "harness/docking_controller.py", ROOT / "harness/task1_perception_adapter.py",
                      ROOT / "harness/object_perception.py", calib / "grasp_relative_poses_train.json",
                      calib / "posture_and_gripper_stats_train.json"]
            from harness.task1_perception_adapter import detector_module_name
            detector = ROOT / (detector_module_name(orch.get("perception"), task_profile.TASK_ID).replace(".", "/") + ".py")
            if detector not in paths:          # Task 0: harness/radio_perception.py (re-exports object_perception)
                paths.append(detector)
            if task_profile.TASK_ID == 0:      # height-aware shield: shoulder heights by URDF FK (docking_controller)
                paths += [ROOT / "harness/r1pro_kinematics.py", ROOT / "configs/r1pro_eef_offsets.json",
                          ROOT / "data/omnigibson/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"]
        if orch.get("postconditions") == "task1_physical":
            paths.append(ROOT / "harness/task1_predicates.py")
        if orch.get("postconditions") == "task0_physical":
            paths.append(ROOT / "harness/task0_predicates.py")
        if orch.get("manip_executor") == "primitive":
            paths += [ROOT / "harness/grasp_primitive.py", ROOT / "harness/r1pro_kinematics.py",
                      ROOT / "configs/r1pro_eef_offsets.json",
                      ROOT / "outputs/validation/task1_docking_calibration_20260926/grasp_geometry_train.json"]
        if args.env_wrapper and args.env_wrapper.startswith("harness."):
            paths.append(ROOT / (args.env_wrapper.rsplit(".", 1)[0].replace(".", "/") + ".py"))
        if orch.get("precondition", True) and orch.get("navigation") != "docking":
            paths += [vtt / p for p in ("precond_prototypes.npy", "precond_meta.json")]
        for option, prefix in (("arrival_gate", "precond"), ("postcondition_gate", "postcond")):
            if orch.get(option) == "patch":
                paths += [vtt / f"{prefix}_patch_prototypes.npz", vtt / f"{prefix}_patch_meta.json"]
    for path in (args.program, args.memory_vtt, orch.get("train_targets"), orch.get("bank_in")):
        if path:
            paths.append(ROOT / path)
    model_cfgs = [cfg]
    if orch.get("manip_checkpoint"):
        if manip_cfg is None:
            raise ValueError("manip checkpoint config required to fingerprint its runtime artifacts")
        model_cfgs.append(manip_cfg)
        manip_stats = ROOT / manip_cfg["dataset"]["stats_path"]
        manip_split = manip_stats.parent / json.loads(manip_stats.read_text()).get("provenance", {}).get(
            "split_file", f"split_task_{CONFIG['task']['id']:03d}.json")
        paths += [Path(orch["manip_checkpoint"]), manip_stats, manip_split]
    for model_cfg in model_cfgs:
        v3 = model_cfg["model"].get("v3")
        if v3 is not None:
            paths += [ROOT / "harness/wam_v3.py", ROOT / v3["stats_path"]]
        if model_cfg["model"].get("proprio_history_stats"):   # train-only per-lag diff std used by normalize_state
            paths.append(ROOT / model_cfg["model"]["proprio_history_stats"])
        dino = ROOT / model_cfg["model"]["vision_encoder"]["checkpoint_path"]
        paths += [dino / "config.json", *dino.glob("*.safetensors"), *dino.glob("*.bin")]
    return sorted({p.resolve() for p in paths})


def artifact_hashes(paths):
    # Record absent optional stall calibrations too, so adding one during a run cannot go unnoticed.
    optional = {ROOT / "configs/vtt_stall_calibration.json", ROOT / "configs/motion_stall_calibration.json"}
    return {str(p): (None if p in optional and not p.exists() else sha256_file(p)) for p in paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(CONFIG["profiles"]), default="smoke")
    ap.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs/training/stage1_skillproto/checkpoint_best.pt"))
    ap.add_argument("--manip-checkpoint", default=None,
                    help="optional second checkpoint for all non-navigation orchestrator contracts "
                         "(sets orchestrate.manip_checkpoint; navigation keeps --checkpoint)")
    ap.add_argument("--skill", default="move to")
    ap.add_argument("--skill-schedule", default=None)
    ap.add_argument("--program", default=None, help="skill program JSON advanced by VTT closure")
    ap.add_argument("--budget-factor", type=float, default=1.5)
    ap.add_argument("--memory-vtt", default=None, help="harness.memory_vtt export JSON (verified realized VTTs as conditions)")
    ap.add_argument("--orchestrate", default=None, help="JSON config for harness.orchestrator mode (see policy_server --orchestrate)")
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=1, help="flow samples averaged per WAM chunk")
    ap.add_argument("--seed", type=int, default=0, help="Shared policy/eval seed; RNGs are not reseeded per episode")
    ap.add_argument("--max-steps", type=int, default=None, help="override episode step timeout (smoke/dev only)")
    ap.add_argument("--instances", type=int, nargs="*", default=None, help="override public instance indices")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=int, default=3000, help="evaluator wall timeout (s)")
    ap.add_argument("--write-video", action="store_true")
    ap.add_argument("--video-view", choices=("robot", "viewer"), default="robot",
                    help="robot: tiled policy cameras (672x448); viewer: third-person follow camera at --video-width x --video-height")
    ap.add_argument("--video-width", type=int, default=1280)
    ap.add_argument("--video-height", type=int, default=720)
    ap.add_argument("--video-framing", choices=("auto", "fixed"), default="auto",
                    help="viewer only. auto: occlusion-aware evaluator-side observer camera (PhysX rays + frustum over a "
                         "candidate grid, per-frame choice in videos/*.mp4.camera.jsonl); fixed: legacy follow offset. Never a policy input")
    ap.add_argument("--log-object-joints", action="store_true",
                    help="evaluator writes per-step task-object joint angles (physical post-condition evidence; never policy input)")
    ap.add_argument("--log-object-poses", action="store_true",
                    help="evaluator writes per-step task-object/robot poses (evaluator-side evidence only; never policy input)")
    ap.add_argument("--env-wrapper", default=None,
                    help="evaluator observation wrapper, e.g. omnigibson.eval.wrappers.RGBDFullResWrapper (official RGB-D track)")
    ap.add_argument("--policy-cuda-fraction", type=float, default=0.30)
    ap.add_argument("--policy-device", choices=("cpu", "cuda"), default="cuda",
                    help="CPU avoids overlap between WAM weights and Isaac VRAM; simulator still needs GPU authorization")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.seed < 2**32:
        ap.error("--seed must be an integer in [0, 2**32)")
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "process.log"
            cancelled = threading.Event()
            path.write_text("Imported scene\n")
            assert not wait_for_policy_request(path, cancelled, 0.01)
            path.write_text("Health check failed, waiting for server at http://127.0.0.1:8000...\n")
            assert wait_for_policy_request(path, cancelled, 0.01)
            cancelled.set()
            assert not wait_for_policy_request(path, cancelled, 0.01)
            assert not wait_for_policy_capacity(cancelled, "cpu")
            cancelled.clear()
            assert wait_for_policy_capacity(cancelled, "cpu")
            assert wait_for_policy_capacity(cancelled, "cuda", sample=lambda: {"gpu": [
                {"vram_total_mib": 16311, "vram_used_mib": 12682}]})
            assert not wait_for_policy_capacity(cancelled, "cuda", timeout=0.01, sample=lambda: {"gpu": []})
            # two-model reserve (stage3 + 212M-param fp32 manip model): blocks where the single-model reserve passes
            two = manip_policy_reserve_mib({"model": {"vision_encoder": {"checkpoint_path": "models/dinov3-vitl16-pretrain-lvd1689m"}}},
                                           {"model": {"vision_encoder": {"checkpoint_path": "models/dinov3-vitl16-pretrain-lvd1689m"}}}, 808.49)
            assert two == CONFIG["resource_policy"]["policy_vram_reserve_mib"] + 809 + MANIP_ACTIVATION_MARGIN_MIB
            sim = lambda used: (lambda: {"gpu": [{"vram_total_mib": 16311, "vram_used_mib": used}]})
            lim = min(CONFIG["resource_policy"]["gpu_total_vram_limit_mib"], 16311 * CONFIG["resource_policy"]["monitor_exit_percent"] / 100)
            edge = int(lim - two)          # boundary derived from the configured reserve (was hard-coded for 2200)
            assert wait_for_policy_capacity(cancelled, "cuda", timeout=0.01, sample=sim(edge))
            assert not wait_for_policy_capacity(cancelled, "cuda", timeout=0.01, sample=sim(edge + 1), reserve_mib=two)
            assert wait_for_policy_capacity(cancelled, "cuda", timeout=0.01, sample=sim(edge - 1), reserve_mib=two)
            orch = merge_manip_checkpoint('{"vlm": "none"}', "m.pt")
            assert json.loads(orch) == {"vlm": "none", "manip_checkpoint": "m.pt"}
            assert merge_manip_checkpoint(orch, "m.pt") == orch and merge_manip_checkpoint('{"a": 1}', None) == '{"a": 1}'
            for bad in ((None, "m.pt"), ('{"manip_checkpoint": "x.pt"}', "m.pt")):
                try:
                    merge_manip_checkpoint(*bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"accepted {bad}")
            assert not wait_for_policy_capacity(cancelled, "cuda", timeout=0.01, sample=lambda: {"gpu": [
                {"vram_total_mib": 16311, "vram_used_mib": 15000}]})
            assert completed_results(tmp, []) == []
            results_dir = Path(tmp) / "eval/json"
            results_dir.mkdir(parents=True)
            expected = [{"task": "synthetic", "instance_id": 17, "rollout_id": 0}]
            result = {"task": "synthetic", "instance_id": 17, "rollout_id": 0, "steps": 300, "success": False, "q_score": {"final": 0.0}}
            (results_dir / "episode.json").write_text(json.dumps(result))
            assert len(completed_results(tmp, expected)) == 1  # A completed task failure is still valid evidence.
            result["steps"] = 0
            (results_dir / "episode.json").write_text(json.dumps(result))
            try:
                completed_results(tmp, expected)
            except ValueError:
                pass
            else:
                raise AssertionError("Zero-step result accepted")
        print("PASS: startup capacity/cancellation (incl. two-model reserve), manip-checkpoint option merge, "
              "and real scored-episode completion")
        return

    if args.profile == "report" and (args.max_steps is not None or args.instances is not None):
        raise SystemExit("report profile must use the official instances and timeout")
    if args.instances is not None and (not args.instances or len(set(args.instances)) != len(args.instances)
                                       or not set(args.instances) <= set(CONFIG["profiles"][args.profile])):
        raise SystemExit("instance override must stay inside the selected profile")
    if sum(x is not None for x in (args.program, args.skill_schedule, args.orchestrate)) > 1:
        raise SystemExit("choose exactly one of program, skill-schedule, or orchestrate")
    try:
        args.orchestrate = merge_manip_checkpoint(args.orchestrate, args.manip_checkpoint)
        from harness.privileged_base_navigation import validate_launch, WRAPPERS, PREPOSITION_WRAPPER, PROTOCOL
        privbase_enabled = validate_launch(json.loads(args.orchestrate or "{}"), args.env_wrapper,
                                           args.run_name, args.profile, task_profile.TASK_ID)
    except ValueError as error:
        raise SystemExit(str(error))
    manip_path = json.loads(args.orchestrate).get("manip_checkpoint") if args.orchestrate else None
    if args.orchestrate and args.profile == "report":
        cfg = json.loads(args.orchestrate)
        cfg["frozen"] = True
        args.orchestrate = json.dumps(cfg)
    blockers = collect()["blockers"]
    if blockers:
        print(json.dumps({"blocked": blockers}, indent=2))
        raise SystemExit(2)
    manip_cfg, policy_reserve = None, CONFIG["resource_policy"]["policy_vram_reserve_mib"]
    if manip_path:  # second action model: read its embedded config and weight size before any process starts
        manip_cfg = checkpoint_config(manip_path)
        policy_reserve = manip_policy_reserve_mib(checkpoint_config(args.checkpoint), manip_cfg, checkpoint_action_mib(manip_path))
        if json.loads(args.orchestrate or "{}").get("offload_inactive"):
            # only one action model is ever resident (policy_server offload_inactive): single-model reserve + margin,
            # plus a second DINOv3 only if the encoders differ (same rule as manip_policy_reserve_mib)
            policy_reserve -= math.ceil(checkpoint_action_mib(manip_path))
    out = ROOT / "outputs/behavior/closed_loop" / args.run_name
    out.mkdir(parents=True, exist_ok=False)  # preserve prior evidence instead of mixing or deleting runs

    unit = f"wam-policy-{os.getpid()}"
    server_cmd = [str(LILAWAM), str(ROOT / "harness/policy_server.py"), "--checkpoint", args.checkpoint,
                  "--skill", args.skill, "--exec-horizon", str(args.exec_horizon), "--num-steps", str(args.num_steps), "--samples", str(args.samples),
                  "--port", str(args.port), "--cuda-fraction", str(args.policy_cuda_fraction),
                  "--device", args.policy_device, "--seed", str(args.seed),
                  "--log-dir", str(out), "--hold-on-error"]
    if args.skill_schedule:
        server_cmd += ["--skill-schedule", args.skill_schedule]
    if args.program:
        server_cmd += ["--program", args.program, "--budget-factor", str(args.budget_factor)]
    if args.memory_vtt:
        server_cmd += ["--memory-vtt", args.memory_vtt]
    if args.orchestrate:
        server_cmd += ["--orchestrate", args.orchestrate]
    server_log = (out / "policy_server.log").open("w")

    def start_server_after_scene_load():
        if not wait_for_policy_request(out / "process.log", cancelled, min(args.timeout, 1500)):
            return
        ready = wait_for_policy_capacity(cancelled, args.policy_device, reserve_mib=policy_reserve)
        (out / "policy_startup.json").write_text(json.dumps({"ready": ready, "device": args.policy_device,
            "policy_vram_reserve_mib": policy_reserve,
            "reason": "capacity_available" if ready else "cancelled_or_insufficient_cuda_headroom"}) + "\n")
        if not ready and not cancelled.is_set():
            # The evaluator would otherwise idle until --timeout waiting for a policy that never starts.
            refused.reason = "policy start refused (insufficient CUDA headroom)"
            refused.set()
        if not ready or cancelled.is_set():
            return
        subprocess.Popen(["systemd-run", "--user", "--quiet", "--pipe", "--collect", f"--unit={unit}",
                          f"--working-directory={ROOT}", "-p", "CPUAffinity=12 13 14 15", "-p", "MemoryMax=10G",
                          "-p", "MemorySwapMax=0", "env", "OMP_NUM_THREADS=2", f"P2_TASK_ID={task_profile.TASK_ID}", *server_cmd],
                         stdout=server_log, stderr=subprocess.STDOUT)
        print("policy server started after scene load", flush=True)

    cancelled, refused = threading.Event(), threading.Event()
    starter = threading.Thread(target=start_server_after_scene_load, daemon=True)
    starter.start()
    try:
        argv = eval_command(args.profile, "127.0.0.1", args.port) + ["--seed", str(args.seed)]
        argv[argv.index("--output-dir") + 1] = str(out / "eval")
        if args.instances is not None:
            i = argv.index("--instance-indices") + 1
            j = argv.index("--num-rollouts")
            argv[i:j] = [str(x) for x in args.instances]
        if args.max_steps is not None:
            if "--max-steps" in argv:
                argv[argv.index("--max-steps") + 1] = str(args.max_steps)
            else:
                argv += ["--max-steps", str(args.max_steps)]
        if args.write_video:
            argv += ["--write-video", "--video-view", args.video_view]
            if args.video_view == "viewer":
                argv += ["--video-width", str(args.video_width), "--video-height", str(args.video_height),
                         "--video-framing", args.video_framing]
        if args.log_object_joints:
            argv += ["--log-object-joints"]
        if args.log_object_poses:
            argv += ["--log-object-poses"]
        if args.env_wrapper:
            argv += ["--env-wrapper", args.env_wrapper]
        artifact_paths = (runtime_artifact_paths(args, argv, checkpoint_config(args.checkpoint), manip_cfg) if manip_path
                          else runtime_artifact_paths(args, argv, checkpoint_config(args.checkpoint)))
        hashes = artifact_hashes(artifact_paths)
        manifest = {"run_name": args.run_name, "profile": args.profile, "checkpoint": args.checkpoint,
                    "object_joint_log": ("eval/object_joints/*.jsonl: evaluator-side privileged joint angles for evidence only; never policy input"
                                         if args.log_object_joints else None),
                    "object_pose_log": ("eval/object_poses/*.jsonl: evaluator-side privileged object/robot poses for evidence only; never policy input"
                                        if args.log_object_poses else None),
                    "task_id": task_profile.TASK_ID, "task_name": task_profile.TASK_NAME, "env_wrapper": args.env_wrapper,
                    "checkpoint_sha256": hashes[str(Path(args.checkpoint).resolve())],
                    # navigation contracts -> checkpoint; every other contract -> manip_checkpoint (when set)
                    "manip_checkpoint": manip_path,
                    "manip_checkpoint_sha256": (hashes.get(str(Path(manip_path).resolve())) or sha256_file(manip_path))
                                               if manip_path else None,
                    "policy_vram_reserve_mib": policy_reserve,
                    "artifact_sha256": hashes, "policy_argv": server_cmd,
                    "seed": args.seed, "effective_policy_seed": None, "effective_evaluator_seed": None,
                    "seed_behavior": {"evaluator": "seed once before environment construction; no per-episode reseed",
                                      "policy": "Generator seeded once at construction; reset does not reseed",
                                      "policy_metadata_rng_reset_behavior": None,
                                      "physics_rendering_fully_deterministic": False,
                                      "note": "Python/NumPy/Torch RNG seeding does not guarantee identical RTX/PhysX trajectories or RGB; "
                                              "Warp rand_init returns a local state, not a global simulator seed."},
                    "skill": args.skill, "skill_schedule": args.skill_schedule, "program": args.program, "orchestrate": args.orchestrate, "budget_factor": args.budget_factor, "memory_vtt": args.memory_vtt, "exec_horizon": args.exec_horizon,
                    "num_inference_steps": args.num_steps, "samples_averaged": args.samples, "max_steps_override": args.max_steps,
                    "instances_override": args.instances, "evaluator_argv": argv,
                    "evaluator_env": child_environment(), "source": CONFIG["source_dir"], "environment": CONFIG["environment_dir"],
                    "policy_device": args.policy_device,
                    "official_score_valid": False,
                    "isaac_sim_deviation": "Isaac Sim 6.1 port (see third_party/BEHAVIOR-1K-isaac6/PORT_PROVENANCE.json); not the official 5.1 pin",
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if privbase_enabled:
            manifest["privileged_base_navigation"] = {**PROTOCOL, "wrapper": args.env_wrapper}
        if args.env_wrapper == PREPOSITION_WRAPPER:
            manifest["base_prepositioning"] = {
                "explicit_user_authorization": "2026-10-01: user selected allow base prepositioning for separate manipulation diagnosis",
                "source": "named object geometry plus TRAIN dock target; base-only pose initialization after reset",
                "obstacle_check": "oriented PhysX overlap_box against actual collision shapes, base only; AABBs logged; not full-body collision certification",
                "objects_moved": False, "upper_body_joints_set": False, "navigation_success_claimed": False,
                "receipt": "P2_PREPOSITION JSON in process.log; no physics setup steps; real posture settling thereafter"}
        if args.env_wrapper in ("harness.interval_base_rgbd_wrapper.IntervalBaseRGBDWrapper", *WRAPPERS):
            manifest["base_velocity_adapter"] = {
                "source": "robot six virtual-base joint encoder increments over the actual control interval",
                "output": "local midpoint-equivalent mean vx/vy/wz in the existing 61D proprio",
                "world_pose_used_for_control": False, "object_truth_used_for_control": False,
                "absolute_base_position_exposed": False, "gain_fitted_to_evaluator": False,
                "protocol_boundary": "explicit simulated proprioception adaptation; not native instantaneous qvel or real wheel-odometry validation"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("evaluator:", shlex.join(argv), flush=True)
        rc = run_guarded(argv, child_environment(), out, timeout=args.timeout, abort=refused)
        manifest["evaluator_returncode"] = rc
        indices = args.instances if args.instances is not None else CONFIG["profiles"][args.profile]
        manifest["expected_episodes"] = len(indices)
        manifest["evaluation_complete"] = False
        try:
            expected = planned_episodes(out, task_profile.TASK_NAME, indices, args.seed)
            manifest["planned_episodes"] = expected
            manifest["effective_evaluator_seed"] = args.seed  # Confirmed by the evaluator's pre-episode plan.
            receipt = json.loads((out / "eval/policy_metadata.json").read_text())
            manifest["effective_policy_seed"] = receipt.get("effective_policy_seed")
            manifest["seed_behavior"]["policy_metadata_rng_reset_behavior"] = receipt.get("policy_rng_reset_behavior")
            if type(manifest["effective_policy_seed"]) is not int or manifest["effective_policy_seed"] != args.seed:
                raise ValueError("Policy handshake did not confirm the requested seed")
            results = completed_results(out, expected)
            manifest["completed_episodes"] = results
            if any(type(r[k]) is not int or r[k] != args.seed for r in results
                   for k in ("effective_policy_seed", "effective_evaluator_seed")):
                raise ValueError("Missing or mismatched effective policy/evaluator seed evidence")
            behaviors = {r["policy_rng_reset_behavior"] for r in results}
            if (len(behaviors) != 1 or not all(isinstance(b, str) and b for b in behaviors)
                    or receipt.get("policy_rng_reset_behavior") not in behaviors):
                raise ValueError("Missing or inconsistent policy RNG reset metadata")
            manifest["effective_policy_seed"] = results[0]["effective_policy_seed"]
            manifest["seed_behavior"]["policy_metadata_rng_reset_behavior"] = results[0]["policy_rng_reset_behavior"]
            after = artifact_hashes(artifact_paths)
            manifest["artifact_changes"] = [p for p in hashes if hashes[p] != after[p]]
            if manifest["artifact_changes"]:
                raise ValueError("Runtime code/data artifacts changed during evaluation")
            manifest["evaluation_complete"] = rc == 0
        except (OSError, ValueError, TypeError, KeyError) as error:
            manifest["evaluation_complete"] = False
            manifest["result_error"] = str(error)
        if rc == 0 and not manifest["evaluation_complete"]:
            rc = 2
        manifest["launcher_returncode"] = rc
        manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise SystemExit(rc)
    finally:
        cancelled.set()
        starter.join()
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
        server_log.close()


if __name__ == "__main__":
    main()
