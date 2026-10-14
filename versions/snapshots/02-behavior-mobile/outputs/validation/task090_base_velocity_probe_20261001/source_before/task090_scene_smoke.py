#!/usr/bin/env python3
"""Task90 scene/sensor + ten copy-state hold steps, never a WAM capability test.

Only --execute launches GPU work, under the existing resource monitor and lock.
Does not overwrite shared robot bounds or change scene objects / BDDL goals.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if os.environ.setdefault("P2_TASK_ID", "90") != "90":
    raise ValueError("Task90 smoke requires P2_TASK_ID=90")
from harness import task_profile as TP
from behavior_preflight import ENV, SOURCE, child_environment, collect

SCOPES = {"half__banana.n.01_1": ("half_banana", "xytkre"),
          "half__pomegranate.n.01_1": ("half_pomegranate", "vqybii"),
          "ashcan.n.01_1": ("trash_can", "ifzxzj")}


def child_run(out, steps, instance):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import seed_everything
    from harness.wam_behavior import hold_baseline

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.35)
    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_HQ_RENDERING = False
    gm.DEFAULT_VIEWER_WIDTH = gm.DEFAULT_VIEWER_HEIGHT = 256
    seed_everything(0)
    robot = OmegaConf.load(str(SOURCE / "OmniGibson/omnigibson/eval/r1pro.yaml"))
    robot.sensor_config.VisionSensor.sensor_kwargs.image_height = 224
    robot.sensor_config.VisionSensor.sensor_kwargs.image_width = 224
    cfg = OmegaConf.create({"task": {"name": TP.TASK_NAME}, "robot": robot,
        "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
        "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
        "policy_name": "task090_hold_control_smoke_only", "headless": True, "partial_scene_load": True,
        "max_steps": max(steps + 1, 10), "write_video": False, "mode": "public_test", "seed": 0})
    instance_id = resolve_instance_ids(TP.TASK_NAME, [instance], "public_test")[0]
    with Evaluator(cfg) as evaluator:
        evaluator.reset()
        evaluator.load_task_instance(instance_id)
        evaluator.reset()
        assert evaluator.robot.action_dim == 23
        assert evaluator.env.task.scene_name == "Rs_int"
        objects = {}
        for scope, (category, model) in SCOPES.items():
            obj = evaluator.env.task.object_scope[scope]
            assert obj is not None and obj.category == category and obj.model == model
            objects[scope] = {"name": obj.name, "category": obj.category, "model": obj.model}
        state_key = f"{evaluator.robot.name}::proprio"
        state0 = evaluator.obs[state_key].detach().cpu()
        assert tuple(state0.shape) == (61,) and torch.isfinite(state0).all()
        cameras = {}
        for role, name in evaluator.robot_camera_names.items():
            rgb = evaluator.obs[name + "::rgb"].detach().cpu().numpy()
            assert rgb.shape[:2] == (224, 224) and rgb.dtype == np.uint8
            Image.fromarray(rgb[..., :3]).save(out / f"{role}_rgb.png")
            cameras[role] = {"key": name + "::rgb", "shape": list(rgb.shape)}
        assert len(cameras) == 3
        actions, states, ended = [], [state0.numpy()], False
        # Deliberately never call LocalPolicy.forward (its default all-zero action is unsafe for absolute joints).
        hold = hold_baseline(state0[None], chunk_len=1)[0, 0]
        for _ in range(steps):
            obs, _, terminated, truncated, _ = evaluator.env.step(hold, n_render_iterations=1)
            evaluator.obs = evaluator._preprocess_obs(evaluator._sync_lights_and_get_obs(obs))
            state = evaluator.obs[state_key].detach().cpu()
            assert state.shape == (61,) and torch.isfinite(state).all()
            actions.append(hold.numpy()); states.append(state.numpy())
            if terminated or truncated:
                ended = True
                break
        np.save(out / "states.npy", np.asarray(states, dtype=np.float32))
        np.save(out / "actions.npy", np.asarray(actions, dtype=np.float32).reshape(-1, 23))
        result = {"task_id": 90, "task_name": TP.TASK_NAME, "scene": "Rs_int", "objects": objects,
                  "public_instance_index": instance, "instance_id": instance_id,
                  "scene_reset_verified": True, "action_dimension": 23, "state_dimension": 61,
                  "actions_sent": len(actions), "action_source": "initial copy-state hold; fixed joint targets, zero base",
                  "cameras": cameras, "environment_ended_early": ended, "policy_evaluated": False,
                  "official_score_valid": False, "deviation": "Isaac Sim6.1 port, assisted grasp",
                  "note": "Scene and hold-control smoke only; no task success, WAM or navigation capability claim"}
        (out / "scene_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/validation/task090_preparation_20260930/data_runtime/scene_smoke")
    ap.add_argument("--hold-steps", type=int, default=10)
    ap.add_argument("--instance", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()
    if not 0 <= args.hold_steps <= 30 or args.instance not in range(10, 20):
        ap.error("smoke permits 0..30 hold steps and development indices 10..19 only")
    out = args.out.resolve()
    if args.child:
        child_run(out, args.hold_steps, args.instance)
        return
    if not args.execute:
        print(json.dumps({"task_id": 90, "scene": "Rs_int", "gpu_started": False,
            "out": str(out), "command": f"P2_TASK_ID=90 GPU_AUTHORIZATION={ROOT}/outputs/validation/task0_gpu_peak_authorization_20260930/authorization.json {ROOT}/.conda-envs/lilawam/bin/python {Path(__file__).resolve()} --execute",
            "note": "Main session only: serial GPU lock + existing behavior_resources guard; no model/VTT dependency"}, indent=2))
        return
    from wam_v3_resource_authorization import validate_authorization, resource_environment, require_unlocked
    from behavior_resources import run_guarded
    from gpu_lock import acquire, release
    contract = validate_authorization(os.environ.get("GPU_AUTHORIZATION"))
    require_unlocked()
    blockers = collect()["blockers"]
    if blockers:
        raise SystemExit(json.dumps({"blocked": blockers}))
    out.mkdir(parents=True, exist_ok=False)
    owner = f"task090-scene-smoke-{os.getpid()}"
    acquire(owner, max_wait=0)
    try:
        env = child_environment({"P2_TASK_ID": "90", "PYTHONPATH": str(ROOT), **resource_environment(contract)})
        started = time.time()
        rc = run_guarded([str(ENV / "bin/python"), str(Path(__file__).resolve()), "--child", "--out", str(out),
                          "--hold-steps", str(args.hold_steps), "--instance", str(args.instance)], env, out, timeout=args.timeout)
        marker = out / "scene_smoke.json"
        if rc == 0 and (not marker.exists() or marker.stat().st_mtime < started):
            rc = 2
        raise SystemExit(rc)
    finally:
        release(owner)


if __name__ == "__main__":
    main()
