"""Websocket evaluation runner for the BEHAVIOR-1K challenge.

Drives the OmniGibson ``Evaluator`` against a policy served over a websocket
(e.g. the openpi or GR00T ``scripts/b1k/serve_b1k.py`` server). For each test instance of a
task it runs a rollout and writes a per-rollout result JSON compatible with
``omnigibson/eval/utils/score_utils.py`` (``q_score``, ``time``,
``agent_distance`` / ``normalized_agent_distance``).

Example:
    python -m omnigibson.eval.eval \
        --task-name turning_on_radio \
        --robot-config omnigibson/eval/r1pro.yaml \
        --mode public_test \
        --host 127.0.0.1 --port 8000 \
        --instance-indices 0 --max-steps 500 \
        --output-dir outputs/b1k_eval --write-video
"""

import argparse
import json
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="BEHAVIOR task name, e.g. turning_on_radio.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy websocket server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy websocket server port.")
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help=(
            "Optional path to YAML/JSON file containing one complete robot config dictionary with canonical "
            "'model' and 'name' fields. Add eval.camera_sensor_names to configure eval camera roles."
        ),
    )
    parser.add_argument(
        "--instance-indices",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Instance indices for the selected mode. For train these are direct train instance IDs; "
            "for public_test / hidden_test these index into that 20-instance split."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("train", "public_test", "hidden_test"),
        default="public_test",
        help="Instance split to evaluate. Default: public_test.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED,
                        help="Seed eval RNGs once before environment construction; not a physics/rendering determinism guarantee.")
    parser.add_argument("--num-rollouts", type=int, default=1, help="Rollouts per instance.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode timeout in steps. Default (None) = 1.5x mean human-demo length.",
    )
    parser.add_argument(
        "--env-wrapper",
        default="omnigibson.eval.wrappers.DefaultWrapper",
        help="Target path of the EnvironmentWrapper to apply.",
    )
    parser.add_argument(
        "--policy",
        choices=("websocket", "local"),
        default="websocket",
        help="Policy backend to use. local emits zero actions and is intended for eval smoke tests.",
    )
    parser.add_argument("--output-dir", default="/tmp/b1k_eval", help="Where to write result JSONs.")
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save every attempted rollout from reset to termination/interruption under <output-dir>/videos.",
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Frame rate for saved rollout videos.")
    parser.add_argument("--video-view", choices=("robot", "viewer"), default="robot",
                        help="viewer records native HD third-person RGB independently of policy cameras.")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-camera-offset", type=float, nargs=3, default=[-2.5, -2.5, 2.1],
                        help="Recording camera XYZ in metres relative to robot yaw; inspect for walls/occlusion.")
    parser.add_argument("--video-camera-target", type=float, nargs=3, default=[0.6, 0.0, 0.9])
    parser.add_argument("--video-framing", choices=("auto", "fixed"), default="auto",
                        help="auto: occlusion-aware observer camera (PhysX rays + frustum, candidates around "
                             "--video-camera-offset, logged to <video>.camera.jsonl); fixed: raw yaw-relative offset. "
                             "Recording only; never a policy input.")
    parser.add_argument("--log-object-joints", action=argparse.BooleanOptionalAction, default=False,
                        help="Write per-step task-object joint angles under <output-dir>/object_joints (evaluator-side evidence; never policy input).")
    parser.add_argument("--log-object-poses", action=argparse.BooleanOptionalAction, default=False,
                        help="Write per-step world poses of task-scope objects (not systems/floors/agent), the robot base "
                             "and the BDDL goal literals under <output-dir>/object_poses (evaluator-side evidence; never policy input).")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OmniGibson headless (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gm.HEADLESS = args.headless

    if not 0 <= args.seed < 2**32 or args.num_rollouts < 1:
        raise ValueError("seed must be uint32 and num-rollouts must be positive")
    if len(set(args.instance_indices)) != len(args.instance_indices):
        raise ValueError("Duplicate instance indices are not separate rollouts")
    seed = seed_everything(args.seed)
    logger.info(f"Seeded Python, NumPy, and Torch with seed={seed}")

    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, mode=args.mode)
    logger.info(f"Resolved {args.mode} instance ids for {args.task_name}: {instance_ids}")

    # Resolve with the existing official mapping BEFORE any episode; the launcher checks this plan, not just a count.
    plan_dir = Path(args.output_dir).expanduser()
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan = {"task": args.task_name, "mode": args.mode, "instance_indices": args.instance_indices,
            "instance_ids": [int(i) for i in instance_ids], "num_rollouts": args.num_rollouts, "seed": seed}
    with (plan_dir / "evaluation_plan.json").open("x") as file:
        json.dump(plan, file, indent=2)

    robot_config = None
    if args.robot_config is not None:
        robot_config_path = Path(args.robot_config).expanduser()
        robot_config = OmegaConf.load(str(robot_config_path))
        logger.info(f"Loaded robot config from {robot_config_path}")

    if args.policy == "websocket":
        model_cfg = {
            "_target_": "omnigibson.eval.policies.WebsocketPolicy",
            "host": args.host,
            "port": args.port,
        }
    else:
        model_cfg = {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}

    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": args.env_wrapper},
            "policy_name": args.policy,
            "model": model_cfg,
            "headless": args.headless,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": args.write_video,
            "video_view": args.video_view,
            "video_width": args.video_width,
            "video_height": args.video_height,
            "video_camera_offset": args.video_camera_offset,
            "video_camera_target": args.video_camera_target,
            "video_framing": args.video_framing,
            "mode": args.mode,
            "seed": seed,
            "task": {"name": args.task_name},
            "robot": robot_config,
        }
    )

    json_dir = os.path.join(os.path.expanduser(args.output_dir), "json")
    os.makedirs(json_dir, exist_ok=True)
    video_dir = os.path.join(os.path.expanduser(args.output_dir), "videos")
    if args.write_video:
        os.makedirs(video_dir, exist_ok=True)

    results = []
    with Evaluator(cfg) as evaluator:
        for instance_id in instance_ids:
            try:
                preposition = args.env_wrapper == "harness.prepositioned_base_rgbd_wrapper.PrepositionedBaseRGBDWrapper"
                if preposition:
                    evaluator.env.set_preposition_enabled(False)
                evaluator.reset()
                evaluator.load_task_instance(int(instance_id))
                if preposition:
                    evaluator.env.set_preposition_enabled(True)
            except Exception:
                logger.exception(f"Failed to load task instance {instance_id}.")
                raise
            for rollout_id in range(args.num_rollouts):
                video_path = os.path.join(video_dir, f"{args.task_name}_{instance_id}_{rollout_id}.mp4")
                terminated = truncated = False
                steps, success = 0, None
                try:
                    evaluator.reset()
                    policy_start = evaluator.policy_evidence()
                    # Persist the handshake even if this rollout later crashes and produces no scored JSON.
                    receipt = {k: policy_start[k] for k in ("effective_policy_seed", "policy_rng_reset_behavior")}
                    with (plan_dir / "policy_metadata.json").open("w") as file:
                        json.dump(receipt, file, indent=2)
                    identity = {"task": args.task_name, "instance_id": int(instance_id), "rollout_id": rollout_id,
                                "policy_episode": policy_start["policy_episode"]}
                    if (policy_start["effective_policy_seed"] is not None
                            and (type(policy_start["effective_policy_seed"]) is not int or policy_start["effective_policy_seed"] != seed)):
                        raise ValueError("Policy seed differs from the requested evaluator seed")
                    if args.write_video:
                        evaluator.start_recording(video_path, rate=args.video_fps)
                    if args.log_object_joints:
                        joint_dir = os.path.join(args.output_dir, "object_joints")
                        os.makedirs(joint_dir, exist_ok=True)
                        evaluator.start_joint_log(os.path.join(joint_dir, f"{args.task_name}_{instance_id}_{rollout_id}.jsonl"),
                                                  episode_identity=identity)
                    if args.log_object_poses:
                        pose_dir = os.path.join(args.output_dir, "object_poses")
                        os.makedirs(pose_dir, exist_ok=True)
                        evaluator.start_pose_log(os.path.join(pose_dir, f"{args.task_name}_{instance_id}_{rollout_id}.jsonl"),
                                                 episode_identity=identity)
                    while not (terminated or truncated):
                        terminated, truncated = evaluator.step()
                        steps += 1

                    success = bool(evaluator.env.task.success)
                    metrics = {}
                    for metric in evaluator.metrics:
                        metrics.update(metric.aggregate(evaluator.env))

                    policy_end = evaluator.policy_evidence()
                    result = {
                        **identity,
                        "effective_evaluator_seed": seed,
                        "effective_policy_seed": policy_start["effective_policy_seed"],
                        "policy_rng_reset_behavior": policy_start["policy_rng_reset_behavior"],
                        "policy_step_alignment": (policy_start["policy_episode"] is not None
                                                  and policy_end["policy_episode"] == policy_start["policy_episode"]
                                                  and policy_start["policy_step"] == 0 and policy_end["policy_step"] == steps),
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "steps": steps,
                        "success": success,
                        **metrics,
                    }
                    out_path = os.path.join(json_dir, f"{args.task_name}_{instance_id}_{rollout_id}.json")
                    with open(out_path, "x") as f:
                        json.dump(result, f, indent=2, default=float)
                    q_score = metrics.get("q_score", {}).get("final")
                    video_msg = f" | video -> {video_path}" if args.write_video else ""
                    logger.info(
                        f"Result: instance={instance_id} rollout={rollout_id} steps={steps} "
                        f"success={success} q_score={q_score} -> {out_path}{video_msg}"
                    )
                    results.append(result)
                except Exception:
                    logger.exception(f"Instance {instance_id} rollout {rollout_id} failed.")
                    raise
                finally:
                    if args.write_video:
                        evaluator.stop_recording(completed=terminated or truncated, success=success, steps=steps)
                    evaluator.stop_joint_log()
                    if args.log_object_poses:
                        evaluator.stop_pose_log()

    n = len(results)
    n_success = sum(r["success"] for r in results)
    mean_q = (sum(r.get("q_score", {}).get("final", 0.0) for r in results) / n) if n else 0.0
    logger.info(f"Eval summary: {n_success}/{n} success | mean q_score={mean_q:.3f} | task={args.task_name}")


if __name__ == "__main__":
    main()
