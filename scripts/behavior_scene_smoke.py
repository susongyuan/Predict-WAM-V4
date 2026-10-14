#!/usr/bin/env python3
"""Reset the official Task 13 instance and verify sensors; never execute a policy action."""
import argparse
import json
import os
import time
from pathlib import Path

from behavior_preflight import CONFIG, ENV, ROOT, SOURCE, child_environment, collect


def child_run(output_dir):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import seed_everything

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    # This limits only torch allocations; the outer process monitors Vulkan/RTX VRAM as well.
    torch.cuda.set_per_process_memory_fraction(0.35)
    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_HQ_RENDERING = False
    gm.DEFAULT_VIEWER_WIDTH = 256
    gm.DEFAULT_VIEWER_HEIGHT = 256
    seed_everything(0)
    robot = OmegaConf.load(str(SOURCE / "OmniGibson/omnigibson/eval/r1pro.yaml"))
    robot.sensor_config.VisionSensor.sensor_kwargs.image_height = 224
    robot.sensor_config.VisionSensor.sensor_kwargs.image_width = 224
    cfg = OmegaConf.create({"task": {"name": CONFIG["task"]["name"]}, "robot": robot,
        "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
        "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
        "policy_name": "unused_during_scene_reset", "headless": True, "partial_scene_load": True,
        "max_steps": 10, "write_video": False, "mode": "public_test", "seed": 0})
    instance_id = resolve_instance_ids(CONFIG["task"]["name"], [10], "public_test")[0]
    print(f"Loading official task={CONFIG['task']['name']} public index=10 / instance={instance_id}", flush=True)
    with Evaluator(cfg) as evaluator:
        evaluator.reset()
        evaluator.load_task_instance(instance_id)
        evaluator.reset()
        assert evaluator.robot.action_dim == 23
        state = evaluator.obs[f"{evaluator.robot.name}::proprio"]
        assert tuple(state.shape) == (61,) and torch.isfinite(state).all()
        cameras = {}
        for role, name in evaluator.robot_camera_names.items():
            rgb = evaluator.obs[name + "::rgb"].detach().cpu().numpy()
            assert rgb.shape[:2] == (224, 224) and rgb.shape[-1] in (3, 4) and rgb.dtype == np.uint8
            Image.fromarray(rgb[..., :3]).save(output_dir / f"{role}_rgb.png")
            cameras[role] = {"observation_key": name + "::rgb", "shape": list(rgb.shape), "dtype": str(rgb.dtype)}
        # Physical action bounds in controller units for the decision kernel (from the robot model, not data).
        robot = evaluator.robot
        lower = robot.joint_lower_limits.detach().cpu().numpy().tolist()
        upper = robot.joint_upper_limits.detach().cpu().numpy().tolist()
        names = list(robot.joints.keys())
        idx = {"trunk": [int(i) for i in robot.trunk_control_idx],
               "left_arm": [int(i) for i in robot.arm_control_idx["left"]],
               "right_arm": [int(i) for i in robot.arm_control_idx["right"]]}
        from omnigibson.controllers import ControllerView
        base_limits = ControllerView.get_command_input_limits(robot.controllers["base"][0])
        base_lim = [list(map(float, base_limits[0])), list(map(float, base_limits[1]))]
        bounds = {"layout": "base(3) trunk(4) left_arm(7) left_gripper(1) right_arm(7) right_gripper(1)",
                  "base_command_input_limits": base_lim, "gripper_command_limits": [-1.0, 1.0], "joint_names": names}
        for part, ids in idx.items():
            bounds[part] = {"names": [names[i] for i in ids], "lower": [lower[i] for i in ids], "upper": [upper[i] for i in ids]}
        low = base_lim[0] + bounds["trunk"]["lower"] + bounds["left_arm"]["lower"] + [-1.0] + bounds["right_arm"]["lower"] + [-1.0]
        high = base_lim[1] + bounds["trunk"]["upper"] + bounds["left_arm"]["upper"] + [1.0] + bounds["right_arm"]["upper"] + [1.0]
        assert len(low) == 23 and len(high) == 23, (len(low), len(high))
        bounds["action_low"], bounds["action_high"] = low, high
        (ROOT / "configs/r1pro_action_bounds.json").write_text(json.dumps(bounds, indent=1) + "\n")
        result = {"scene_reset_verified": True, "policy_evaluated": False, "actions_sent": 0,
                  "action_bounds_file": "configs/r1pro_action_bounds.json",
                  "task": CONFIG["task"]["name"], "instance_id": instance_id, "public_instance_index": 10,
                  "action_dimension": evaluator.robot.action_dim, "state_dimension": int(state.numel()),
                  "controller_order": list(evaluator.robot.controller_order), "cameras": cameras,
                  "note": "Scene/sensor/schema verification only; no task success or Q-score claim."}
        (output_dir / "scene_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    output_dir = ROOT / "outputs/behavior/scene_smoke"
    if args.child:
        child_run(output_dir)
        return
    if not args.execute:
        print("Run with --execute to launch one guarded Task 13 scene reset. No policy actions are sent.")
        return
    blockers = collect()["blockers"]
    if blockers:
        print(json.dumps({"blocked": blockers}, indent=2))
        raise SystemExit(2)
    from behavior_resources import run_guarded
    env = child_environment()
    started = time.time()
    rc = run_guarded([str(ENV / "bin/python"), str(Path(__file__).resolve()), "--child"],
                     env, output_dir, timeout=args.timeout)
    marker = output_dir / "scene_smoke.json"
    # Kit may exit zero after a Python exception. Only a fresh success artifact proves this run passed.
    if rc == 0 and (not marker.exists() or marker.stat().st_mtime < started
                    or not json.loads(marker.read_text()).get("action_bounds_file")):
        print(json.dumps({"error": "scene smoke produced no fresh success artifact; inspect process.log"}))
        rc = 2
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
