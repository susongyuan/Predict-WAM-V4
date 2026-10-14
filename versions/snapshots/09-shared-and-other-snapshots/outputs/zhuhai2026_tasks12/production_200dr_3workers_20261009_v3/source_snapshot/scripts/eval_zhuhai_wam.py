#!/usr/bin/env python3
"""Fixed fresh-seed Zhuhai physical scoring with an explicit 10Hz driver.

The simulator and frozen-DINO WAM run in separate ABI environments, sharing
only allowlisted onboard RGB, native16 measured feedback and previous targets. No expert actions,
object poses, rule state, videos or third-view camera reach the policy. Rule
state is saved independently by the evaluator. A cutoff/error is preserved.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.robotwin_policy_worker import (Channel, InfrastructureError, MAX_IMAGE_BYTES,
    MODEL_VIEWS, VIEWS, PolicyClient as TransportClient, atomic_json, digest, envelope, identity,
    vector, strict_keys, ids, PROTOCOL)
from harness import zhuhai_training as zt
from harness import zhuhai_robot_spec as robot_spec
from scripts.progress1_log import log


OBSERVATION_FIELDS = ("rgb", "measured_joint_gripper_state", "commanded_joint_gripper_state")


def validate_observation(observation):
    """No legacy14 fallback, no task labels, object poses or rule-state leakage."""
    import numpy as np
    strict_keys(observation, OBSERVATION_FIELDS)
    strict_keys(observation["rgb"], VIEWS)
    result = dict(rgb={})
    for key in OBSERVATION_FIELDS[1:]:
        result[key] = vector(observation[key], robot_spec.ROBOT_DOF, key)
        if (np.abs(result[key]) > np.finfo(np.float32).max).any():
            raise InfrastructureError("feedback/command cannot be represented as float32")
    # Measured gripper overshoot is evidence, not a command to silently clip.
    commanded = result["commanded_joint_gripper_state"]
    grip = commanded[list(robot_spec.GRIPPER_INDICES)]
    if np.any(grip < 0) or np.any(grip > 1):
        raise InfrastructureError("target grippers must use [0,1], 1=open")
    for view in VIEWS:
        image = np.asarray(observation["rgb"][view])
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
                or min(image.shape[:2]) <= 0 or image.nbytes > MAX_IMAGE_BYTES):
            raise InfrastructureError("RGB must be bounded uint8 HWC3")
        result["rgb"][view] = image
    return result


def encode_observation(observation):
    observation = validate_observation(observation)
    return dict(rgb={v: dict(shape=list(image.shape), bytes=base64.b64encode(image.tobytes()).decode("ascii"))
                     for v, image in observation["rgb"].items()},
                **{k: observation[k].tolist() for k in OBSERVATION_FIELDS[1:]})


def decode_observation(value):
    import numpy as np
    strict_keys(value, OBSERVATION_FIELDS)
    strict_keys(value["rgb"], VIEWS)
    result = {k: value[k] for k in OBSERVATION_FIELDS[1:]}
    result["rgb"] = {}
    for view in VIEWS:
        item = value["rgb"][view]
        strict_keys(item, ("shape", "bytes"))
        shape = item["shape"]
        if (not isinstance(shape, list) or len(shape) != 3
                or any(type(x) is not int or x <= 0 for x in shape) or shape[2] != 3
                or math.prod(shape) > MAX_IMAGE_BYTES or not isinstance(item["bytes"], str)
                or len(item["bytes"]) > 4 * ((MAX_IMAGE_BYTES + 2) // 3)):
            raise InfrastructureError("invalid RGB geometry/size")
        try:
            raw = base64.b64decode(item["bytes"], validate=True)
        except (ValueError, TypeError) as exc:
            raise InfrastructureError("invalid RGB base64") from exc
        if len(raw) != math.prod(shape):
            raise InfrastructureError("RGB byte count mismatch")
        result["rgb"][view] = np.frombuffer(raw, np.uint8).reshape(shape).copy()
    return validate_observation(result)


def bound_action(raw, limits):
    """Bounds audit only; caller rejects any clipping, never applies a second limiter."""
    import numpy as np
    value = vector(raw, robot_spec.ROBOT_DOF, "native16 action").astype(np.float64)
    bounds = np.asarray(limits, np.float64)
    if bounds.shape != (len(robot_spec.ARM_INDICES), 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("native16 requires fourteen finite ordered physical arm bounds")
    result = value.copy()
    indices = list(robot_spec.ARM_INDICES)
    result[indices] = np.clip(value[indices], bounds[:, 0], bounds[:, 1])
    indices = list(robot_spec.GRIPPER_INDICES)
    result[indices] = np.clip(value[indices], 0, 1)
    return dict(raw_action=value.tolist(), bounded_action=result.tolist(), clip_mask=(result != value).tolist())


def policy_observation(env):
    from harness.zhuhai_demo_recording import native_observation
    result = native_observation(env.policy_observation(), measured_state=env.actual_state(), target_action=env.get_action())
    return validate_observation(result)


def runtime_gate_results(env, observation=None):
    """Adapt actual main-thread safety/camera receipts, never certify from a name."""
    actual_robot = getattr(env, "robot_contract", None)
    actual_robot = actual_robot() if callable(actual_robot) else actual_robot
    if actual_robot != zt.robot_contract():
        raise InfrastructureError("runtime instance robot contract differs from loaded native16 policy")
    results = getattr(env, "gate_results", {})
    results = results() if callable(results) else results
    answer = {}
    for kind in ("mechanical", "camera"):
        value = results.get(kind) if isinstance(results, dict) else None
        if value is None:
            for name in (kind + "_gate_result", kind + "_gate"):
                value = getattr(env, name, None)
                if value is not None:
                    value = value() if callable(value) else value
                    break
        if value is None and kind == "mechanical" and callable(getattr(env, "physics_safety", None)):
            value = dict(env.physics_safety(), source="runtime.physics_safety")
        if value is None and kind == "camera":
            evidence = getattr(env, "camera_evidence", None)
            if isinstance(evidence, dict) and observation is not None:
                observed = validate_observation(observation)
                value = dict(passed=(evidence.get("all_geometry_in_frame") is True
                                    and evidence.get("all_named_entities_visible") is True),
                    source="actual initial camera_evidence + three onboard RGB readbacks", evidence=evidence,
                    rgb_shapes={v: list(image.shape) for v, image in observed["rgb"].items()})
        if isinstance(value, dict):
            value = dict(value)
            # Instance identity above is independently checked before attaching it.
            value.setdefault("robot_contract", actual_robot)
            value.setdefault("robot_contract_sha256", digest(actual_robot))
        answer[kind] = value
    return answer


def validate_runtime_gates(gates):
    binding = zt.protocol_binding()
    for kind in ("mechanical", "camera"):
        gate = gates.get(kind)
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            raise InfrastructureError(f"explicit passed {kind} gate required before evaluation")
        fields = [key for key in ("robot_contract", "robot_contract_sha256") if key in gate]
        if not fields or any(gate[key] != binding[key] for key in fields):
            raise InfrastructureError(f"{kind} gate is not bound to the actual native16 robot contract")
        if kind == "mechanical" and gate.get("events"):
            raise InfrastructureError("mechanical gate contradicts retained safety events")
    return gates


def validate_physics_safety(evidence):
    safety = evidence.get("physics_safety") if isinstance(evidence, dict) else None
    if (not isinstance(safety, dict) or safety.get("passed") is not True
            or safety.get("events") != [] or not safety.get("checks")):
        raise InfrastructureError("unsafe/missing actual physics_safety; never assign policy success")
    return safety


class Native16PolicyClient(TransportClient):
    """Reuse process/socket ownership, never legacy14 observation/action codecs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        runtime = self.ready.get("runtime", {})
        if (runtime.get("format") != zt.FORMAT
                or runtime.get("robot_contract_sha256") != zt.protocol_binding()["robot_contract_sha256"]):
            self.close()
            raise InfrastructureError("worker is not the bound native16 measured-feedback policy")

    def action(self, episode, step, observation):
        try:
            reply = self.request(envelope("observe", episode, step, observation=encode_observation(observation)), "action")
            for key in ("raw_action", "model_raw_action", "smoothed_action", "limited_action"):
                vector(reply[key], robot_spec.ROBOT_DOF, key)
            if reply.get("raw_action") != reply.get("limited_action"):
                raise InfrastructureError("native16 driver action alias differs from limited command")
            return reply
        except BaseException:
            self.close()
            raise


def serve_native16(channel, policy):
    channel.send(envelope("ready", "startup", 0, runtime=policy.runtime, pid=os.getpid()), 10)
    while True:
        message = channel.recv(600)
        episode, step = message.get("episode"), message.get("step")
        ids(episode, step)
        if message.get("version") != PROTOCOL:
            raise InfrastructureError("unsupported IPC protocol")
        started = time.monotonic()
        if message.get("kind") == "reset":
            strict_keys(message, ("version", "kind", "episode", "step", "seed"))
            if step != 0:
                raise InfrastructureError("reset step must be zero")
            policy.reset(episode, message["seed"])
            reply = envelope("reset_ok", episode, step)
        elif message.get("kind") == "clear_actions":
            strict_keys(message, ("version", "kind", "episode", "step", "reason"))
            reply = envelope("cleared", episode, step, **policy.clear_actions(episode, step, message["reason"]))
        elif message.get("kind") == "observe":
            strict_keys(message, ("version", "kind", "episode", "step", "observation"))
            result = policy.observe(episode, step, decode_observation(message["observation"]))
            for key in ("raw_action", "model_raw_action", "smoothed_action", "limited_action"):
                vector(result[key], robot_spec.ROBOT_DOF, key)
            reply = envelope("action", episode, step, **result, policy_seconds=time.monotonic() - started)
        else:
            raise InfrastructureError("unknown native16 worker request")
        channel.send(reply, 30)


class Native16OnboardRecorder:
    """Lossless onboard sensor arrays, NOT videos; no evaluator truth is archived here."""
    def __init__(self, directory, episode, seed):
        self.path = Path(directory) / "onboard"
        self.path.mkdir(exist_ok=False)
        self.count = 0
        self.manifest = dict(schema=zt.FORMAT + "/onboard", episode=episode, seed=seed,
            frames=0, complete=False, lossless=True, fields=list(VIEWS) + list(OBSERVATION_FIELDS[1:]),
            time_semantics="actual physics_time_seconds at every 10Hz control observation",
            record_video=False, third_view=False, **zt.protocol_binding())
        atomic_json(self.path / "manifest.json", self.manifest)

    def append(self, observation, control_step, physics_time):
        import numpy as np
        from harness.robotwin_eval_evidence import jsonl
        observation = validate_observation(observation)
        if control_step != self.count or not math.isfinite(physics_time) or physics_time < 0:
            raise ValueError("onboard observation index/clock mismatch")
        cadence_ok = not self.count or math.isclose(physics_time - self.last_time, .1, rel_tol=0, abs_tol=1e-6)
        final = self.path / f"observation_{self.count:06d}.npz"
        with final.open("xb") as handle:
            np.savez_compressed(handle, **observation["rgb"],
                **{k: observation[k] for k in OBSERVATION_FIELDS[1:]})
        jsonl(self.path / "frames.jsonl", dict(index=self.count, file=final.name, control_step=control_step,
                                               physics_time_seconds=physics_time, wall_monotonic=time.monotonic()))
        self.count += 1
        self.last_time = physics_time
        if not cadence_ok:
            raise ValueError("off-cadence onboard sample preserved; expected 10Hz")

    def close(self, complete=False):
        self.manifest.update(frames=self.count, complete=complete)
        atomic_json(self.path / "manifest.json", self.manifest)


def verify_owner(expected):
    from scripts.gpu_lock import owner
    if not expected or owner() != expected:
        raise InfrastructureError('matching externally owned GPU lock required')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(Path(arg).name == 'train_zhuhai_wam.py' for arg in argv if arg):
            raise InfrastructureError('training process still present; cannot overlap evaluation')


def read_contract(path):
    path = Path(path).resolve()
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    contract = manifest['contract']
    receipt = json.loads(path.with_name(path.name + '.sha256.json').read_text())
    if (contract.get('format') != zt.FORMAT or manifest['contract_sha256'] != digest(contract)
            or receipt.get('model_only') is not True or receipt['size'] != path.stat().st_size):
        raise InfrastructureError('immutable native16 Zhuhai checkpoint/contract required')
    zt.validate_training_contract(contract)
    for name, expected in contract['source_files'].items():
        if identity(name)['sha256'] != expected:
            raise InfrastructureError('training implementation changed after checkpoint')
    for source in contract['sources'].values():
        if source.get('engine', {}).get('name') != 'Isaac Sim':
            raise InfrastructureError('Isaac Sim provenance required; legacy SAPIEN evaluation is disabled')
        if identity(source['scene']['path'])['sha256'] != source['scene']['sha256']:
            raise InfrastructureError('scene changed since collection; new data and run required')
        for row in source['manifests']:
            if identity(row['path'])['sha256'] != row['sha256']:
                raise InfrastructureError('collection manifest changed since training')
            for name, expected in row['contract'].get('config_hashes', {}).items():
                if identity(name)['sha256'] != expected:
                    raise InfrastructureError('head/wrist camera or task configuration changed')
            for name, expected in row['contract'].get('scene_dependency_sha256', {}).items():
                if identity(name)['sha256'] != expected:
                    raise InfrastructureError('onboard camera helper or rule implementation changed')
    from harness.zhuhai_action_smoothing import protocol, protocol_hash
    zt.validate_postprocessing(contract['action_postprocessing'])
    if (contract.get('action_postprocessing') != protocol()
            or contract.get('action_postprocessing_sha256') != protocol_hash(protocol())):
        raise InfrastructureError('Isaac run must bind the enabled upstream-spline and physical rate-limit protocol')
    return contract, receipt


def worker(args):
    from harness.robotwin_multitask_policy_worker import die_with_parent
    die_with_parent()
    channel = Channel(socket.socket(fileno=args.ipc_fd))
    try:
        verify_owner(args.gpu_lock_owner)
        contract, receipt = read_contract(args.checkpoint)
        if identity(args.checkpoint)['sha256'] != receipt['sha256']:
            raise InfrastructureError('checkpoint SHA256 differs')
        import cv2
        import numpy as np
        import torch
        from importlib.metadata import version
        from harness import zhuhai_training as zt
        from harness.robotwin_training_data import preprocess_rgb
        from harness.wam_behavior import load_upstream
        from harness.wam_v3 import V3Context
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        cv2.setNumThreads(2)
        wrapper, condition, actual = zt.load_policy(args.checkpoint, args.task, 'cuda')
        if actual != contract:
            raise InfrastructureError('loaded checkpoint differs from manifest')
        cfg = contract['config']
        if version('scipy') != contract['action_postprocessing']['worker_scipy_version']:
            raise InfrastructureError('postprocessing SciPy runtime version differs from bound protocol')
        _, runner = load_upstream()
        encoder_path = ROOT / cfg['model']['vision_encoder']['checkpoint_path']
        if identity(encoder_path / 'model.safetensors')['sha256'] != contract['cache']['encoder_sha256']:
            raise InfrastructureError('frozen DINO SHA differs')
        encoder, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(
            str(encoder_path), dtype=torch.bfloat16, device='cuda')
        if (hidden, registers, patch) != (1024, 4, 16):
            raise InfrastructureError('unexpected encoder geometry')
        wrapper.vision_encoder = encoder.eval().requires_grad_(False)
        wrapper.eval()
        def encode(rgb):
            verify_owner(args.gpu_lock_owner)
            with torch.inference_mode():
                pixels = np.stack([preprocess_rgb(rgb[v]) for v in VIEWS])
                feats, pool = wrapper.encode(torch.from_numpy(pixels)[None])
                return feats, pool[0].float().cpu().numpy()
        def sample(feats, inputs, seed, step):
            state, pool = inputs
            seed_value = int(hashlib.sha256(f'{seed}:{step}'.encode()).hexdigest()[:15], 16)
            generator = torch.Generator(device='cuda').manual_seed(seed_value)
            with torch.inference_mode():
                actions, _ = wrapper.sample_actions(feats, torch.from_numpy(state)[None],
                    torch.from_numpy(pool)[None], torch.from_numpy(condition)[None], num_steps=10,
                    generator=generator)
                return actions[0].float().cpu().numpy()
        v3 = cfg['model']['v3']
        from harness.zhuhai_action_smoothing import SmoothedChunkPolicy
        policy = SmoothedChunkPolicy(V3Context(v3['proprio_lags'], v3['memory_lags'], MODEL_VIEWS, state_dim=robot_spec.ROBOT_DOF),
                             encode, sample, contract['action_postprocessing'])
        policy.runtime = dict(format=zt.FORMAT, task=args.task, **zt.protocol_binding(), checkpoint_sha256=receipt['sha256'],
            checkpoint_step=receipt['step'], strict_load=True, projections_reset=False,
            task_condition_source='new_successful_TRAIN_endpoints_only',
            task_condition_sha256=hashlib.sha256(condition.tobytes()).hexdigest(),
            condition_source_split='train', numerical_normalization=contract['numerical_normalization'],
            time=contract['time'], action_postprocessing=contract['action_postprocessing'],
            action_postprocessing_sha256=contract['action_postprocessing_sha256'],
            raw_action_protocol_alias='limited_native_driver_action; original model outputs archived as model_raw_action',
            dino_frozen=True, system_mode='pure_wam',
            vlm=False, high_level_planner=False, ask=False, memory_writes=False)
        serve_native16(channel, policy)
        return 0
    except BaseException as exc:
        try:
            channel.send(envelope('infrastructure_error', 'startup', 0,
                                  error=f'{type(exc).__name__}: {exc}'), 1)
        except BaseException:
            pass
        raise
    finally:
        channel.sock.close()


def validate_rule(evidence):
    if not isinstance(evidence, dict) or not isinstance(evidence.get('score'), dict):
        raise ValueError('independent Isaac physical rule evidence required')
    clock = evidence.get('physics_time_seconds')
    score = evidence['score']
    if (not isinstance(clock, (float, int)) or not math.isfinite(clock) or clock < 0
            or type(score.get('full_task_success')) is not bool
            or not isinstance(score.get('base_score'), (float, int)) or not math.isfinite(score['base_score'])
            or not score.get('rule_source') or not evidence.get('objects')):
        raise ValueError('invalid physical rule clock/score/evidence')
    json.dumps(evidence, allow_nan=False)
    validate_physics_safety(evidence)
    return float(clock), score


def episode(args, contract, receipt):
    import numpy as np
    from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
    from harness.robotwin_eval_evidence import jsonl
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed = args.episode_seed
    name = f'{args.task}_seed_{seed}'
    row = dict(task=args.task, seed=seed, status='starting', control_steps=0,
               full_task_success=None, base_score=None, record_video=False,
               engine=dict(name='Isaac Sim', version='6.1.0.0'))
    atomic_json(out / 'result.json', row)
    env = client = onboard = None
    started = time.monotonic()
    try:
        log(f'Isaac固定未见种子闭环：{args.task} seed={seed}；10Hz、LiLa样条+跨chunk限幅、仅头部+双腕', actor='eval_zhuhai_wam')
        verify_owner(args.gpu_lock_owner)
        source = contract['sources'][args.task]
        env = IsaacZhuhaiEnv(gpu_lock_owner=args.gpu_lock_owner, output_dir=out)
        env.setup(task=args.task, seed=seed, randomized=False, mapping=source.get('life_mapping'))
        allowed = policy_observation(env)
        row.update(admission_gates=runtime_gate_results(env, allowed), **zt.protocol_binding())
        validate_runtime_gates(row['admission_gates'])
        initial = env.rule_state()
        initial_time, initial_score = validate_rule(initial)
        jsonl(out / 'rule_evidence.jsonl', dict(control_step=0, **initial))
        if initial_score['full_task_success'] or initial_score['base_score'] != 0:
            raise ValueError('invalid initial state already scores; no policy credit')
        limits = getattr(env, 'joint_limits', None)
        if limits is None and callable(getattr(env, 'joint_bounds', None)):
            limits = env.joint_bounds()
        limits = np.asarray(limits, dtype=np.float64)
        if limits.shape != (len(robot_spec.ARM_INDICES), 2) or not np.isfinite(limits).all():
            raise ValueError('Isaac runtime must expose all fourteen Panda arm joint limits')
        if not np.allclose(limits, robot_spec.DUAL_JOINT_LIMITS, atol=1e-6, rtol=0):
            raise ValueError('runtime joint travel differs from official dual-Panda spec')
        rate = contract['action_postprocessing']['rate_limit']
        declared = np.stack([rate['position_lower_rad'], rate['position_upper_rad']], axis=1)
        if not np.allclose(limits, declared, atol=1e-6, rtol=0):
            raise ValueError('Isaac physical joint bounds differ from bound postprocessing contract; new protocol required')
        command = [str(ROOT / '.venvs/wam-fixed/bin/python'), '-B', str(Path(__file__).resolve()),
                   '--worker', '--checkpoint', args.checkpoint, '--task', args.task,
                   '--gpu-lock-owner', args.gpu_lock_owner]
        client = Native16PolicyClient(command, out / 'policy_worker.log', args.startup_timeout, args.policy_timeout)
        client.reset(name, seed)
        row.update(policy_ready=client.ready, joint_limits=limits.tolist(),
            max_control_steps=args.max_steps, maximum_physics_seconds=args.max_steps / 10.,
            simulation_proxy=True, competition_certification=False, system_mode='pure_wam',
            action_postprocessing_sha256=contract['action_postprocessing_sha256'])
        onboard = Native16OnboardRecorder(out, name, seed)
        onboard.append(allowed, 0, initial_time)
        final = initial
        delta_sum, delta_count, max_velocity, max_acceleration = 0., 0, 0., 0.
        for step in range(args.max_steps):
            verify_owner(args.gpu_lock_owner)
            if time.monotonic() - started > args.wall_seconds:
                raise TimeoutError('fixed wall-clock budget reached')
            import shutil
            if shutil.disk_usage(out).free < 100 * 1024**3:
                raise InfrastructureError('100GiB free reserve reached; preserve source evidence')
            reply = client.action(name, step, allowed)
            bounded = bound_action(reply['limited_action'], limits)
            record = dict(control_step=step, raw_action=reply['model_raw_action'],
                smoothed_action=reply['smoothed_action'], limited_action=reply['limited_action'],
                dispatched_action=None if any(bounded['clip_mask']) else bounded['bounded_action'],
                clip_mask=bounded['clip_mask'], policy=reply,
                physics_time_before=final['physics_time_seconds'],
                action_postprocessing_sha256=contract['action_postprocessing_sha256'])
            jsonl(out / 'actions.jsonl', record)
            if any(bounded['clip_mask']):
                raise ValueError('post-limiter physical range changed; rejected action retained, no second limiter')
            before = float(final['physics_time_seconds'])
            env.step_action(np.asarray(bounded['bounded_action']), interval_seconds=.1)
            allowed = policy_observation(env)
            final = env.rule_state()
            jsonl(out / 'rule_evidence.jsonl', dict(control_step=step + 1, **final))
            row['last_rule_evidence'] = final
            onboard.append(allowed, step + 1, float(final['physics_time_seconds']))
            now, score = validate_rule(final)
            if not math.isclose(now - before, .1, abs_tol=1e-6):
                raise ValueError('Isaac runtime physical control clock diverged from fixed 10Hz')
            stats = reply['rate_limit_stats']
            delta_sum += float(reply['smooth_delta_rms_rad']) ** 2
            delta_count += 1
            max_velocity = max(max_velocity, stats['max_abs_velocity_rad_s'])
            max_acceleration = max(max_acceleration, stats['max_abs_acceleration_rad_s2'])
            row.update(status='running', control_steps=step + 1,
                elapsed_physics_seconds=now - initial_time, elapsed_wall_seconds=time.monotonic() - started,
                partial_base_score=score['base_score'], postprocessing_statistics=dict(
                    rms_spline_delta_rad=math.sqrt(delta_sum / delta_count),
                    max_abs_command_velocity_rad_s=max_velocity,
                    max_abs_command_acceleration_rad_s2=max_acceleration))
            if step % 10 == 0:
                atomic_json(out / 'result.json', row)
            if score['full_task_success']:
                break
        score = final['score']
        row.update(status='completed', full_task_success=bool(score['full_task_success']),
                   base_score=score['base_score'], final_rule_evidence=final,
                   termination='full_rule_success' if score['full_task_success'] else 'fixed_step_limit')
        onboard.close(complete=True)
        atomic_json(out / 'result.json', row)
        return 0
    except BaseException as exc:
        row.update(status='infrastructure_error' if not isinstance(exc, TimeoutError) else 'wall_timeout',
                   error=f'{type(exc).__name__}: {exc}', full_task_success=None, base_score=None,
                   elapsed_wall_seconds=time.monotonic() - started)
        atomic_json(out / 'result.json', row)
        return 2
    finally:
        if onboard is not None and not onboard.manifest['complete']:
            onboard.close(complete=False)
        if client is not None:
            client.close()
        if env is not None:
            env.close()


def evaluate(args):
    contract, receipt = read_contract(args.checkpoint)
    tasks = [args.task] if args.task else list(contract['conditions'])
    for task in tasks:
        if task not in contract['conditions']:
            raise ValueError('unknown trained task')
    seeds = args.seeds or list(range(932001, 932021))
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('unique nonnegative seeds required')
    training_seeds = {row['seed'] for splits in contract['splits'].values()
                      for rows in splits.values() for row in rows}
    if training_seeds & set(seeds):
        raise ValueError('physical evaluation seeds overlap admitted expert seeds')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(checkpoint=identity(args.checkpoint, content=False), receipt=receipt,
        contract_sha256=digest(contract), tasks=tasks, seeds=seeds, **zt.protocol_binding(),
        scoring='independent strict scene rule_state.score.full_task_success, no expert filtering',
        recording=dict(video=False, third_view=False, onboard_sensor_archive=True),
        engine=dict(name='Isaac Sim', version='6.1.0.0'),
        action_postprocessing=contract['action_postprocessing'],
        action_postprocessing_sha256=contract['action_postprocessing_sha256'],
        control=dict(frequency_hz=10, fixed_seconds_per_action=0.1, max_steps=args.max_steps,
                     maximum_physics_seconds=args.max_steps / 10., chunk_frames=32, execution_prefix=8,
                     driver='Isaac official dual-Panda native16 targets + measured feedback at 10Hz; spline once then cross-chunk physical limits'),
        scope='fresh-seed rigid simulation proxy, not competition certification',
        source_files={str(Path(__file__).resolve()): identity(__file__)['sha256']})
    if (out / 'plan.json').exists() and json.loads((out / 'plan.json').read_text()) != plan:
        raise ValueError('evaluation contract changed; use a new output directory')
    atomic_json(out / 'plan.json', plan)
    if not args.execute:
        print(json.dumps(dict(status='cpu_plan_completed', plan=plan), ensure_ascii=False, indent=2))
        return 0
    verify_owner(args.gpu_lock_owner)
    rows = []
    for task in tasks:
        for seed in seeds:
            target = out / task / f'seed_{seed}'
            if target.exists():
                result = json.loads((target / 'result.json').read_text())
                if result['status'] != 'completed':
                    raise RuntimeError('previous incomplete/error seed is retained; inspect before a new run')
                rows.append(result)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            command = [str(ROOT / '.conda-envs/behavior6/bin/python'), '-B', str(Path(__file__).resolve()),
                '--checkpoint', args.checkpoint, '--out', str(target), '--task', task,
                '--gpu-lock-owner', args.gpu_lock_owner, '--episode-seed', str(seed),
                '--max-steps', str(args.max_steps), '--wall-seconds', str(args.wall_seconds),
                '--startup-timeout', str(args.startup_timeout), '--policy-timeout', str(args.policy_timeout)]
            with (out / f'{task}_seed_{seed}.log').open('xb') as handle:
                proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT,
                                        start_new_session=True)
                try:
                    code = proc.wait(timeout=args.wall_seconds + args.startup_timeout + 60)
                except BaseException:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    raise
            result = json.loads((target / 'result.json').read_text()) if (target / 'result.json').exists() else dict(
                task=task, seed=seed, status='infrastructure_error', full_task_success=None, base_score=None,
                error=f'child exited {code} before result file')
            rows.append(result)
            summary = dict(status='evaluating', planned_episodes=len(tasks) * len(seeds), episodes=rows,
                completed_episodes=sum(r['status'] == 'completed' for r in rows),
                full_successes=sum(r.get('full_task_success') is True for r in rows))
            atomic_json(out / 'summary.json', summary)
            print(json.dumps(dict(task=task, seed=seed, status=result['status'],
                                  full_task_success=result.get('full_task_success'), base_score=result.get('base_score'))), flush=True)
            if code != 0 or result['status'] != 'completed':
                summary['status'] = 'halted_infrastructure_error'
                atomic_json(out / 'summary.json', summary)
                return 2
    by_task = {}
    for task in tasks:
        selected = [r for r in rows if r['task'] == task]
        successes = sum(r['full_task_success'] for r in selected)
        by_task[task] = dict(episodes=len(selected), full_successes=successes,
                            completion_rate=successes / len(selected),
                            mean_base_score=sum(r['base_score'] for r in selected) / len(selected))
    atomic_json(out / 'summary.json', dict(status='completed', planned_episodes=len(tasks) * len(seeds),
        episodes=rows, by_task=by_task, scope='fixed-seed simulation rule proxy', checkpoint_step=receipt['step']))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out')
    ap.add_argument('--task', choices=['zhuhai_task1', 'zhuhai_task2'])
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--gpu-lock-owner')
    ap.add_argument('--max-steps', type=int, default=12000)
    ap.add_argument('--wall-seconds', type=float, default=3600)
    ap.add_argument('--startup-timeout', type=float, default=180)
    ap.add_argument('--policy-timeout', type=float, default=60)
    ap.add_argument('--episode-seed', type=int)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--ipc-fd', type=int)
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if args.worker:
        if args.ipc_fd is None or not args.task:
            ap.error('worker needs --ipc-fd and --task')
        return worker(args)
    if not args.out or not 1 <= args.max_steps <= 12000:
        ap.error('out required and max-steps must be 1..12000 (at most 1200 simulation seconds)')
    if args.episode_seed is not None:
        contract, receipt = read_contract(args.checkpoint)
        return episode(args, contract, receipt)
    return evaluate(args)


if __name__ == '__main__':
    raise SystemExit(main())
