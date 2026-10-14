#!/usr/bin/env python3
"""Read-only evidence classification and serial fixed-attempt orchestration.

No model/scorer imports. Truncated-log verification lazily replays the ORIGINAL
CPU PhysicalRateLimiter (NumPy required; use behavior6 Python). Default CLI is a
non-writing CPU plan. Only
--execute may launch the frozen single-episode evaluator. No resume or retry.
The reported metric is system completion, not neural-policy-only performance.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_30k_20261010_v1'
AUDIT = ROOT / 'outputs/zhuhai2026_tasks12/verification/precision_v2_overnight_20261009'
PROTOCOL = AUDIT / 'fixed_attempts_protocol.json'
ENV_LAUNCH = ROOT / 'outputs/zhuhai2026_tasks12/verification/tasks12_collection_repair/task1_expert_repair_v2_20261009_c/launch.json'
EVAL = ROOT / 'scripts/eval_zhuhai_wam_precision_v2.py'
PYTHON = ROOT / '.conda-envs/behavior6/bin/python'
SEEDS = (932101, 932102, 932103, 932104, 932105)
STEPS = (20000, 25000, 30000)
TASK = 'zhuhai_task2'
FORMAT = 'zhuhai-precision-wam-native16-measured-10hz-v2'
SCHEMA = 'precision-v2-fixed-attempt-summary-v1'
SIGNATURE = 'no feasible simultaneous position/velocity/acceleration command; stop, never violate a limit'
SCOPE = 'fixed-attempt system completion rate including original controller safety aborts; not attributable solely to the neural policy'
FIELDS = ('head_camera', 'left_camera', 'right_camera', 'measured_joint_gripper_state', 'commanded_joint_gripper_state')


class EvidenceError(ValueError):
    """Fail closed: never reinterpret unknown errors as task failures."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(value):
    # Identical serialization to the frozen manifest's digest, not a new contract.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        before = os.fstat(f.fileno())
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
        after = os.fstat(f.fileno())
    require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns), f'file changed while reading: {path}')
    return h.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise EvidenceError(f'nonfinite JSON number: {value}')
    value = json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)
    json.dumps(value, allow_nan=False)
    return value


def read_json(path):
    return loads(Path(path).read_bytes())


def snapshot(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), f'active evidence: {path}')
    return raw, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def fixed_request(step, seeds):
    require(type(step) is int and step in STEPS, 'only fixed 20k/25k/30k milestones are admitted')
    require(list(seeds) == list(SEEDS) and all(type(s) is int for s in seeds), 'exact fixed seed order 932101..932105 required')


def known_adoptions():
    return {
        932101: RUN / 'development_20000_bootstrap_recovery_20261010_r1/zhuhai_task2/seed_932101/result.json',
        932102: RUN / 'development_20000_fixed_attempts_seed932102_r2/zhuhai_task2/seed_932102/result.json',
    }


def checkpoint_metadata(path):
    """CPU bytes/metadata checks only; never deserialize a model or initialize CUDA."""
    path = Path(path).resolve(strict=True)
    receipt = read_json(path.with_name(path.name + '.sha256.json'))
    step = receipt.get('step')
    fixed_request(step, SEEDS)
    require(path.name == f'checkpoint_step_{step:06d}.pt', 'checkpoint filename/step mismatch')
    require(receipt.get('model_only') is True and receipt.get('size') == path.stat().st_size,
            'immutable model-only receipt/size required')
    require(valid_sha(receipt.get('sha256')) and sha256_file(path) == receipt['sha256'], 'checkpoint SHA mismatch')
    manifest_path = path.parent / 'manifest.json'
    manifest = read_json(manifest_path)
    contract = manifest['contract']
    require(contract.get('format') == FORMAT and digest(contract) == manifest.get('contract_sha256'), 'checkpoint contract mismatch')
    source_files = contract.get('source_files', {})
    require(source_files and str(EVAL) in source_files, 'frozen evaluation source binding missing')
    for name, expected in source_files.items():
        require(sha256_file(name) == expected, f'frozen source changed: {name}')
    return dict(path=str(path), step=step, sha256=receipt['sha256'], size=receipt['size'],
                manifest_sha256=sha256_file(manifest_path), receipt_sha256=sha256_file(path.with_name(path.name + '.sha256.json'))), contract


def validate_rule(evidence):
    require(isinstance(evidence, dict), 'original rule evidence missing')
    score = evidence.get('score', {})
    require(finite(evidence.get('physics_time_seconds')) and evidence['physics_time_seconds'] >= 0,
            'invalid original physics clock')
    require(type(score.get('full_task_success')) is bool and finite(score.get('base_score')) and
            score.get('rule_source') and evidence.get('objects'), 'invalid original rule score/evidence')
    safety = evidence.get('physics_safety', {})
    require(safety.get('passed') is True and safety.get('events') == [] and safety.get('checks'), 'physics safety evidence not passed')
    require(evidence.get('video_recording') is False, 'rule evidence must declare video off')
    return score


def validate_binding(raw, step, seed, sha, contract=None):
    require(raw.get('task') == TASK and type(raw.get('seed')) is int and raw['seed'] == seed, 'task/seed mismatch')
    ready = raw.get('policy_ready', {})
    runtime = ready.get('runtime', {})
    require(ready.get('kind') == 'ready' and ready.get('version') == 1 and ready.get('episode') == 'startup', 'policy_ready missing/invalid')
    require(runtime.get('checkpoint_sha256') == sha and runtime.get('checkpoint_step') == step,
            'policy_ready checkpoint SHA/step mismatch')
    require(runtime.get('format') == FORMAT and runtime.get('task') == TASK and runtime.get('strict_load') is True and
            runtime.get('projections_reset') is False, 'original strict Precision V2 policy binding required')
    require(raw.get('record_video') is False and raw.get('max_control_steps') == 2400 and
            raw.get('maximum_physics_seconds') == 240.0, 'recording/control budget changed')
    for field in ('vlm', 'high_level_planner', 'ask', 'memory_writes', 'future_inputs_used_online'):
        require(runtime.get(field) is False, f'frozen pure-WAM scope changed: {field}')
    for key in ('robot_contract', 'data_contract', 'action_postprocessing'):
        binding = runtime.get(key)
        bound_sha = runtime.get(key + '_sha256')
        require(isinstance(binding, dict) and valid_sha(bound_sha) and digest(binding) == bound_sha,
                f'invalid runtime {key} binding')
        require(raw.get(key + '_sha256') == bound_sha, f'result/runtime {key} SHA mismatch')
        if key != 'action_postprocessing':
            require(raw.get(key) == binding, f'result/runtime {key} mismatch')
        if contract is not None:
            require(contract.get(key) == binding and contract.get(key + '_sha256') == bound_sha, f'checkpoint/runtime {key} mismatch')
    dc = runtime['data_contract']
    require(dc.get('record_video') is False and dc.get('third_view') is False and dc.get('camera_count') == 3,
            'extra camera/video forbidden')
    gates = raw.get('admission_gates', {})
    for name in ('mechanical', 'camera'):
        gate = gates.get(name, {})
        require(gate.get('passed') is True, f'{name} admission gate did not pass')
        bindings = [k for k in ('robot_contract', 'robot_contract_sha256') if k in gate]
        require(bindings and all(gate[k] == runtime[k] for k in bindings), f'{name} gate robot binding mismatch')
        if name == 'mechanical':
            require(gate.get('events') == [], 'mechanical admission has safety events')
    return runtime


def archive_evidence(directory, raw, runtime, terminal):
    """Verify dispatched count against action, rule and actual onboard archives."""
    count = raw['control_steps']
    name = f'{TASK}_seed_{raw["seed"]}'
    identities = {}

    def document(relative, lines=False):
        data, identity = snapshot(directory / relative)
        identities[relative] = identity
        return [loads(line) for line in data.splitlines()] if lines else loads(data)

    actions = document('actions.jsonl', True)
    frames = document('onboard/frames.jsonl', True)
    manifest = document('onboard/manifest.json')
    rules = document('rule_evidence.jsonl', True)
    require(len(actions) == count and len(frames) == len(rules) == count + 1, 'control/action/onboard/rule count mismatch')
    require(manifest.get('frames') == count + 1 and manifest.get('seed') == raw['seed'] and manifest.get('episode') == name,
            'onboard identity/count mismatch')
    require(manifest.get('schema') == FORMAT + '/onboard' and manifest.get('lossless') is True and
            manifest.get('fields') == list(FIELDS) and manifest.get('record_video') is False and manifest.get('third_view') is False,
            'onboard archive contract changed')
    require(manifest.get('complete') is (raw['status'] == 'completed'), 'onboard completion marker contradicts raw status')
    for key in ('robot_contract', 'robot_contract_sha256', 'data_contract', 'data_contract_sha256'):
        require(manifest.get(key) == runtime[key], f'onboard binding mismatch: {key}')
    require(len(list((directory / 'onboard').glob('observation_*.npz'))) == count + 1, 'unexpected/duplicate onboard archives')
    for i, (frame, rule) in enumerate(zip(frames, rules)):
        require(frame.get('index') == frame.get('control_step') == rule.get('control_step') == i, 'nonsequential evidence indices')
        filename = f'observation_{i:06d}.npz'
        require(frame.get('file') == filename and finite(frame.get('physics_time_seconds')), 'invalid onboard frame reference')
        require(math.isclose(frame['physics_time_seconds'], rule.get('physics_time_seconds', -1), abs_tol=1e-6, rel_tol=0), 'onboard/rule clock mismatch')
        if i:
            require(math.isclose(frame['physics_time_seconds'] - frames[i-1]['physics_time_seconds'], .1, abs_tol=1e-6, rel_tol=0), 'control clock not 10Hz')
        score = validate_rule(rule)
        if i < count:
            require(score['full_task_success'] is False, 'actions continued after original full success')
        if i == 0:
            require(score['base_score'] == 0, 'initial state already scores')
        path = directory / 'onboard' / filename
        require(path.is_file() and not path.is_symlink(), 'missing/redirected onboard observation')
        with zipfile.ZipFile(path) as z:
            require(sorted(z.namelist()) == sorted(k + '.npy' for k in FIELDS), 'invalid onboard npz fields')
            for field in FIELDS:
                with z.open(field + '.npy') as f:
                    require(f.read(6) == b'\x93NUMPY', 'invalid onboard NPY header')
                    version = f.read(2)
                    require(version in (b'\x01\x00', b'\x02\x00'), 'unknown onboard NPY version')
                    n = int.from_bytes(f.read(2 if version[0] == 1 else 4), 'little')
                    require(0 < n <= 65536, 'invalid onboard NPY header length')
                    header = ast.literal_eval(f.read(n).decode('latin1'))
                    shape = header.get('shape')
                    if field.endswith('_camera'):
                        require(isinstance(shape, tuple) and len(shape) == 3 and shape[2] == 3 and min(shape) > 0 and
                                header.get('descr') == '|u1', 'invalid onboard RGB geometry/type')
                    else:
                        require(shape == (16,) and header.get('descr') in ('<f4', '<f8', '=f4', '=f8'), 'invalid onboard native16 state')
        identities['onboard/' + filename] = dict(path=str(path.resolve()), sha256=sha256_file(path), size=path.stat().st_size)
        if i == count:
            require({k: v for k, v in rule.items() if k != 'control_step'} == terminal, 'terminal rule archive/result mismatch')
    for i, action in enumerate(actions):
        require(action.get('control_step') == i and action.get('clip_mask') == [False] * 16, 'action count/clipping mismatch')
        for key in ('raw_action', 'smoothed_action', 'limited_action', 'dispatched_action'):
            v = action.get(key)
            require(isinstance(v, list) and len(v) == 16 and all(finite(x) for x in v), f'invalid {key}')
        require(action['dispatched_action'] == action['limited_action'], 'limited action was not dispatched unchanged')
        require(action.get('action_postprocessing_sha256') == runtime['action_postprocessing_sha256'], 'action limiter SHA mismatch')
        policy = action.get('policy', {})
        require(policy.get('kind') == 'action' and policy.get('episode') == name and policy.get('step') == i and
                policy.get('limited_action') == action['limited_action'], 'action policy reply identity mismatch')
        require(finite(action.get('physics_time_before')) and
                math.isclose(action['physics_time_before'], frames[i]['physics_time_seconds'], abs_tol=1e-6, rel_tol=0), 'action clock mismatch')
    return identities


def replay_limiter_evidence(directory, raw, runtime):
    """Deterministic CPU witness, not a physical retry or an invented next action.

    Re-executes the original limiter on every archived smoothed command, starting
    at the original onboard commanded state. The terminal feasible interval is
    independent of the unknown next policy target. Never trust the analysis JSON.
    """
    settings = runtime['action_postprocessing']
    source = ROOT / 'harness/zhuhai_action_smoothing.py'
    require(settings.get('implementation_sha256') == sha256_file(source), 'original limiter source SHA mismatch')
    try:
        import numpy as np
        from harness import zhuhai_action_smoothing as original
    except ImportError as exc:
        raise EvidenceError('CPU NumPy required for direct original limiter replay; use behavior6 Python; checker never spawns a process') from exc
    require(settings == original.protocol(), 'original limiter/postprocessing contract changed')
    require(Path(original.__file__).resolve() == source.resolve(), 'original limiter module path mismatch')
    action_bytes, action_id = snapshot(directory / 'actions.jsonl')
    actions = [loads(line) for line in action_bytes.splitlines()]
    require(len(actions) == raw['control_steps'] and len(actions) > 0, 'limiter replay action count mismatch')
    initial = directory / 'onboard/observation_000000.npz'
    initial_sha = sha256_file(initial)
    with np.load(initial, allow_pickle=False) as observation:
        commanded = observation['commanded_joint_gripper_state'].copy()
    limiter = original.PhysicalRateLimiter(settings['rate_limit'])
    limiter.reset(commanded)
    for index, action in enumerate(actions):
        require(action.get('control_step') == index, 'limiter replay index mismatch')
        smoothed = np.asarray(action['smoothed_action'], dtype=np.float64)
        expected = np.asarray(action['limited_action'], dtype=np.float64)
        require(smoothed.shape == expected.shape == (16,) and np.isfinite(smoothed).all() and np.isfinite(expected).all(),
                'limiter replay nonfinite/invalid trajectory')
        actual, _ = limiter.step(smoothed)
        require(np.array_equal(actual, expected), f'original limiter trajectory mismatch at control_step={index}')
        require(np.array_equal(expected, np.asarray(action['dispatched_action'], dtype=np.float64)),
                f'limiter replay dispatched action mismatch at control_step={index}')
    # Same state/caps and same tolerance as the original step(), independent of
    # requested_v / any unknown, unreturned next smoothed policy output.
    v_lo = np.maximum(-limiter.vmax, limiter.velocity - limiter.amax * limiter.dt)
    v_hi = np.minimum(limiter.vmax, limiter.velocity + limiter.amax * limiter.dt)
    v_lo = np.maximum(v_lo, (limiter.lower - limiter.previous) / limiter.dt)
    v_hi = np.minimum(v_hi, (limiter.upper - limiter.previous) / limiter.dt)
    conflicts = np.flatnonzero(v_lo > v_hi + 1e-10)
    require(len(conflicts) > 0, 'original limiter terminal state is feasible; not a proven safety abort')
    try:
        limiter.step(np.asarray(actions[-1]['limited_action'], dtype=np.float64))
    except ValueError as exc:
        require(str(exc) == SIGNATURE, 'original limiter witness raised a different error')
    else:
        raise EvidenceError('original limiter witness did not reject')
    require(sha256_file(initial) == initial_sha and sha256_file(directory / 'actions.jsonl') == action_id['sha256'],
            'limiter replay evidence changed while reading')
    return dict(method='original_PhysicalRateLimiter_exact_cpu_replay_v1',
                reason='original IPC error precedes client close/terminate; incomplete original worker traceback retained unchanged',
                replayed_control_steps=len(actions), all_limited_actions_exactly_equal=True,
                initial_commanded_observation_sha256=initial_sha, actions_sha256=action_id['sha256'],
                original_limiter_source_sha256=settings['implementation_sha256'],
                infeasible_intervals=[dict(arm_index=int(i), native16_index=int(original.ARM[i]),
                    previous_target=float(limiter.previous[i]), previous_velocity=float(limiter.velocity[i]),
                    feasible_v_lower=float(v_lo[i]), feasible_v_upper=float(v_hi[i])) for i in conflicts],
                witness_target='last recorded limited_action, finite witness only; not the unknown unreturned policy action',
                unknown_next_action_reconstructed=False, rejection_signature=SIGNATURE,
                physical_execution=False, analysis_json_used_as_proof=False)


def classify_result(path, checkpoint_step, seed, checkpoint_sha256, *, contract=None):
    """Classify immutable raw evidence, never edit it; unknown errors raise."""
    fixed_request(checkpoint_step, SEEDS)
    require(seed in SEEDS and valid_sha(checkpoint_sha256), 'invalid expected seed/SHA')
    path = Path(path).resolve(strict=True)
    data, identity = snapshot(path)
    raw = loads(data)
    status = raw.get('status')
    require(status in ('completed', 'infrastructure_error'), f'unrecognized/nonterminal raw status: {status}')
    # Recognize only the exact original IPC error at the rejected control step.
    safety_abort = status == 'infrastructure_error'
    if safety_abort:
        require(type(raw.get('control_steps')) is int and raw['control_steps'] > 0, 'zero-control infrastructure failure is not a task attempt')
        prefix = f"InfrastructureError: wrong worker reply/ID, expected action/{TASK}_seed_{seed}/{raw['control_steps']}: "
        error = raw.get('error')
        require(isinstance(error, str) and error.startswith(prefix), 'unknown infrastructure error; HALT')
        try:
            reply = ast.literal_eval(error[len(prefix):])
        except (SyntaxError, ValueError) as exc:
            raise EvidenceError('invalid original limiter IPC error') from exc
        require(reply == dict(version=1, kind='infrastructure_error', episode='startup', step=0, error='ValueError: ' + SIGNATURE),
                'not the exact known limiter rejection; HALT')
        require(raw.get('base_score') is None and raw.get('full_task_success') is None, 'abort raw score/status must remain unaltered')
    else:
        require(not raw.get('error'), 'completed raw result contains an error')
    require(type(raw.get('control_steps')) is int and 0 < raw['control_steps'] <= 2400, 'invalid dispatched control count')
    runtime = validate_binding(raw, checkpoint_step, seed, checkpoint_sha256, contract)
    terminal = raw.get('last_rule_evidence' if safety_abort else 'final_rule_evidence')
    score = validate_rule(terminal)
    if safety_abort:
        require(score['full_task_success'] is False, 'safety abort requires original last full_task_success=False')
        log_bytes, log_id = snapshot(path.parent / 'policy_worker.log')
        text = log_bytes.decode('utf-8', errors='replace')
        full_traceback = bool('Traceback (most recent call last):' in text and
                re.search(r'File "[^"\n]*[/\\]zhuhai_action_smoothing\.py", line \d+, in step', text) and
                text.rstrip().endswith('ValueError: ' + SIGNATURE))
    else:
        require(type(raw.get('full_task_success')) is bool and raw['full_task_success'] == score['full_task_success'] and
                finite(raw.get('base_score')) and raw['base_score'] == score['base_score'], 'raw completed score differs from final_rule_evidence')
        require((raw['full_task_success'] and raw.get('termination') == 'full_rule_success') or
                (not raw['full_task_success'] and raw.get('termination') == 'fixed_step_limit' and raw['control_steps'] == 2400),
                'completed result has invalid termination/budget')
    evidence = archive_evidence(path.parent, raw, runtime, terminal)
    replay = None
    if safety_abort:
        evidence['policy_worker.log'] = log_id
        if not full_traceback:
            try:
                replay = replay_limiter_evidence(path.parent, raw, runtime)
            except (ValueError, KeyError, OSError) as exc:
                raise EvidenceError(f'truncated original traceback; deterministic limiter proof failed: {exc}') from exc
            require(replay['actions_sha256'] == evidence['actions.jsonl']['sha256'] and
                    replay['initial_commanded_observation_sha256'] == evidence['onboard/observation_000000.npz']['sha256'],
                    'replay/archive identity mismatch')
        require(sha256_file(path.parent / 'policy_worker.log') == log_id['sha256'], 'original worker log changed during classification')
    require(sha256_file(path) == identity['sha256'], 'raw result changed during classification')
    row = dict(task=TASK, seed=seed, status='control_safety_abort' if safety_abort else 'completed',
                full_task_success=False if safety_abort else raw['full_task_success'],
                base_score=raw['base_score'], raw_base_score=raw['base_score'], observed_base_score=score['base_score'],
                raw_status=status, raw_error=raw.get('error'), raw_full_task_success=raw['full_task_success'],
                raw_result_path=identity['path'], raw_result_sha256=identity['sha256'], control_steps=raw['control_steps'],
                checkpoint_step=checkpoint_step, checkpoint_sha256=checkpoint_sha256,
                attempt_count=1, evidence_files=evidence)
    if replay is not None:
        row['safety_abort_evidence'] = replay
        row['original_worker_traceback_truncated'] = True
    return row


def build_summary(metadata, rows, status, *, error=None):
    rows = sorted(rows, key=lambda row: SEEDS.index(row['seed']))
    normal = sum(r['status'] == 'completed' for r in rows)
    aborted = sum(r['status'] == 'control_safety_abort' for r in rows)
    successes = sum(r.get('full_task_success') is True for r in rows if r['status'] == 'completed')
    complete = len(rows) == 5 and [r['seed'] for r in rows] == list(SEEDS) and normal + aborted == 5
    require(status != 'completed' or complete, 'cannot complete an incomplete/invalid cohort')
    value = dict(schema=SCHEMA, status=status, checkpoint_step=metadata['step'], checkpoint_sha256=metadata['sha256'],
                 checkpoint=metadata, task=TASK, seeds=list(SEEDS), planned_episodes=5, episodes=rows,
                 normal_completed=normal, safety_aborted=aborted, full_successes=successes,
                 success_rate=successes / 5 if complete and status == 'completed' else None,
                 record_video=False, third_view=False, interpretation=SCOPE)
    if error is not None:
        value['error'] = error
    return value


def verify_summary(path, checkpoint_step, seeds, checkpoint_sha256=None):
    """Return verified completed summary or raise ValueError. Read-only, no processes."""
    fixed_request(checkpoint_step, seeds)
    path = Path(path)
    if path.is_dir():
        path /= 'summary.json'
    data, _ = snapshot(path)
    value = loads(data)
    require(value.get('schema') == SCHEMA and value.get('status') == 'completed', 'fixed-attempt cohort is not completed')
    sha = checkpoint_sha256 if checkpoint_sha256 is not None else value.get('checkpoint_sha256')
    require(valid_sha(sha) and value.get('checkpoint_sha256') == sha and value.get('checkpoint_step') == checkpoint_step,
            'summary checkpoint SHA/step mismatch')
    require(value.get('seeds') == list(seeds) and value.get('planned_episodes') == 5 and value.get('task') == TASK,
            'summary cohort metadata mismatch')
    require(value.get('record_video') is False and value.get('third_view') is False and value.get('interpretation') == SCOPE,
            'summary scope/recording mismatch')
    rows = value.get('episodes', [])
    require(len(rows) == 5 and [r.get('seed') for r in rows] == list(seeds), 'missing/duplicate/reordered fixed seed')
    require(len({r.get('raw_result_path') for r in rows}) == 5, 'duplicate raw result adoption')
    metadata = value.get('checkpoint', {})
    require(metadata.get('step') == checkpoint_step and metadata.get('sha256') == sha, 'summary checkpoint metadata mismatch')
    require(isinstance(metadata.get('path'), str), 'checkpoint path missing from summary metadata')
    actual_metadata, contract = checkpoint_metadata(metadata['path'])
    require(actual_metadata == metadata, 'summary checkpoint receipt/content changed')
    for row in rows:
        require(row.get('status') in ('completed', 'control_safety_abort') and type(row.get('full_task_success')) is bool,
                'invalid episode outcome')
        actual = classify_result(row['raw_result_path'], checkpoint_step, row['seed'], sha, contract=contract)
        require(row == actual, 'derived row/evidence SHA mismatch')
    expected = build_summary(metadata, rows, 'completed')
    require(value == expected, 'summary counts/rate/metadata mismatch')
    return value


def load_protocol():
    protocol = read_json(PROTOCOL)
    require(protocol.get('schema') == 'precision-v2-fixed-attempt-evaluation-protocol-v1' and
            protocol.get('checkpoints') == list(STEPS) and protocol.get('seeds') == list(SEEDS) and
            protocol.get('planned_attempts_each') == 5 and protocol.get('safety_abort_signature') == SIGNATURE,
            'pre-registered fixed-attempt protocol differs')
    for key in ('scoring_source_unchanged', 'postprocessing_source_unchanged', 'do_not_rerun_failed_seed', 'raw_result_files_must_not_be_modified'):
        require(protocol.get(key) is True, f'protocol changed: {key}')
    require(protocol.get('record_video') is False and protocol.get('third_view') is False, 'protocol recording changed')
    return protocol


def build_plan(args):
    require(args.task == TASK and list(args.seeds) == list(SEEDS), 'only the fixed task2 cohort is admitted')
    out = Path(args.out).absolute()
    require(not out.exists() and not out.is_symlink(), 'out must be fresh; no resume/duplicate writer')
    metadata, contract = checkpoint_metadata(args.checkpoint)
    protocol = load_protocol()
    adopted = [str(Path(p).resolve()) for p in args.adopt_result]
    require(len(adopted) == len(set(adopted)), 'duplicate --adopt-result')
    expected = {str(p.resolve()) for p in known_adoptions().values()} if metadata['step'] == 20000 else set()
    require(set(adopted) == expected, '20k requires BOTH original 932101/932102 paths; 25k/30k forbid adoption')
    launch = read_json(ENV_LAUNCH)
    env = launch['plan']['child_env']
    require(env.get('OMNI_KIT_ACCEPT_EULA') == 'YES' and all(env.get(k) == '2' for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')) and env.get('PYTHONDONTWRITEBYTECODE') == '1',
            'historical v2_c environment is not the required EULA/2-thread environment')
    require(env.get('LD_PRELOAD') == str(ROOT / '.conda-envs/behavior6/lib/libstdc++.so.6.0.36') and Path(env['LD_PRELOAD']).is_file(),
            'historical LD_PRELOAD binding mismatch')
    adopted_seeds = set(known_adoptions()) if adopted else set()
    commands = []
    for seed in SEEDS:
        if seed in adopted_seeds:
            continue
        target = out / TASK / f'seed_{seed}'
        command = [str(PYTHON), '-B', str(EVAL), '--checkpoint', metadata['path'], '--out', str(target), '--task', TASK,
                   '--gpu-lock-owner', args.gpu_lock_owner or '', '--episode-seed', str(seed), '--max-steps', '2400',
                   '--wall-seconds', '3600', '--startup-timeout', '180', '--policy-timeout', '60']
        commands.append(dict(seed=seed, out=str(target), command=command))
    return dict(schema='precision-v2-fixed-attempt-plan-v1', status='plan', out=str(out), checkpoint=metadata,
                task=TASK, seeds=list(SEEDS), planned_episodes=5, gpu_lock_owner=args.gpu_lock_owner,
                adopt_results=adopted, adopt_existing=bool(adopted), commands=commands, child_env=env,
                record_video=False, third_view=False, interpretation=SCOPE,
                protocol_sha256=sha256_file(PROTOCOL), env_launch_sha256=sha256_file(ENV_LAUNCH),
                wrapper_sha256=sha256_file(__file__), frozen_sources=contract['source_files'],
                pinned_original_abort=protocol['original_failed_physical_attempt'])


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def run_episode(item, env):
    """Exactly one Popen, no retries, only terminate our own child group."""
    with Path(item['out']).parent.parent.joinpath(f'seed_{item["seed"]}.log').open('xb') as log:
        process = subprocess.Popen(item['command'], cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=3600 + 180 + 60)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def rejected_row(path, seed, exc):
    row = dict(task=TASK, seed=seed, status='evidence_rejected', full_task_success=None,
               raw_result_path=str(path), validation_error=f'{type(exc).__name__}: {exc}')
    if Path(path).is_file():
        try:
            data, identity = snapshot(path)
            raw = loads(data)
            row.update(raw_status=raw.get('status'), raw_error=raw.get('error'), base_score=raw.get('base_score'),
                       raw_base_score=raw.get('base_score'), raw_full_task_success=raw.get('full_task_success'),
                       raw_result_sha256=identity['sha256'], control_steps=raw.get('control_steps'))
        except (OSError, ValueError):
            pass
    return row


def verify_external_owner(expected):
    require(Path('/tmp/project2_gpu.lock').read_text().strip() == expected, 'external GPU owner mismatch')


def reject_prior_wrapper_attempts(plan):
    # Check existing durable attempt intents, including children that never wrote
    # result.json. Merely choosing a fresh directory is NOT retry authorization.
    out = Path(plan['out'])
    for parent in {out.parent, RUN}:
        if not parent.is_dir():
            continue
        for previous in parent.glob('*/plan.json'):
            if previous.parent == out:
                continue
            old = read_json(previous)
            if old.get('schema') != 'precision-v2-fixed-attempt-plan-v1':
                continue
            if old.get('checkpoint', {}).get('sha256') == plan['checkpoint']['sha256']:
                require(not list(previous.parent.glob('attempt_seed_*.json')),
                        f'previous fixed-attempt launch intent exists; no retry: {previous.parent}')


def execute(plan):
    """Exclusive fresh output. Validate ALL adoptions before any physical launch."""
    require(plan.get('gpu_lock_owner'), '--execute requires --gpu-lock-owner')
    out = Path(plan['out'])
    # Lock the checkpoint READ ONLY: serialize this wrapper across distinct outs,
    # without modifying the original checkpoint, its receipt, or the GPU lock.
    with Path(plan['checkpoint']['path']).open('rb') as checkpoint_lock:
        fcntl.flock(checkpoint_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True, exist_ok=False)
        with (out / '.writer.lock').open('x') as writer:
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(out / 'plan.json', plan)
            rows = []
            metadata = plan['checkpoint']
            try:
                actual, contract = checkpoint_metadata(metadata['path'])
                require(actual == metadata, 'checkpoint metadata changed after plan')
                require(sha256_file(PROTOCOL) == plan['protocol_sha256'] and sha256_file(ENV_LAUNCH) == plan['env_launch_sha256'] and
                        sha256_file(__file__) == plan['wrapper_sha256'], 'protocol/launch/wrapper changed after plan')
                reject_prior_wrapper_attempts(plan)
                adopted = {str(p.resolve()): seed for seed, p in known_adoptions().items()}
                errors = []
                for path in plan['adopt_results']:
                    seed = adopted[path]
                    try:
                        row = classify_result(path, metadata['step'], seed, metadata['sha256'], contract=contract)
                        if seed == 932101:
                            require(row['raw_result_sha256'] == plan['pinned_original_abort']['result_sha256'], 'registered original abort result SHA changed')
                        rows.append(row)
                    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                        rows.append(rejected_row(path, seed, exc))
                        errors.append(f'seed {seed}: {exc}')
                require(not errors, '; '.join(errors))
                env = os.environ.copy()
                env.update(plan['child_env'])
                for item in plan['commands']:
                    verify_external_owner(plan['gpu_lock_owner'])
                    for name, expected in plan['frozen_sources'].items():
                        require(sha256_file(name) == expected, f'frozen source changed: {name}')
                    target = Path(item['out'])
                    require(not target.exists(), 'duplicate attempted seed output; no retry')
                    target.parent.mkdir(exist_ok=True)
                    # Intent is durable BEFORE Popen; failure here is never retried.
                    with (out / f'attempt_seed_{item["seed"]}.json').open('x') as f:
                        json.dump(item, f, allow_nan=False)
                        f.flush()
                        os.fsync(f.fileno())
                    path = target / 'result.json'
                    try:
                        code = run_episode(item, env)
                        row = classify_result(path, metadata['step'], item['seed'], metadata['sha256'], contract=contract)
                        require(code == (2 if row['status'] == 'control_safety_abort' else 0), 'child exit/result disagreement; HALT')
                        rows.append(row)
                    except BaseException as exc:
                        rows.append(rejected_row(path, item['seed'], exc))
                        raise
                    write_json(out / 'summary.json', build_summary(metadata, rows, 'evaluating'))
                write_json(out / 'summary.json', build_summary(metadata, rows, 'completed'))
                verify_summary(out / 'summary.json', metadata['step'], SEEDS, metadata['sha256'])
                return 0
            except BaseException as exc:
                write_json(out / 'summary.json', build_summary(metadata, rows, 'halted_infrastructure_error', error=f'{type(exc).__name__}: {exc}'))
                print(f'HALT: {type(exc).__name__}: {exc}', file=sys.stderr)
                return 2


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--task', default=TASK, choices=[TASK])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    ap.add_argument('--gpu-lock-owner')
    ap.add_argument('--adopt-result', action='append', default=[])
    ap.add_argument('--execute', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = build_plan(args)
        if args.execute:
            return execute(plan)
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'HALT: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
