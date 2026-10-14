#!/usr/bin/env python3
"""Read-only evidence classification and serial fixed-attempt orchestration.

Independent v36 archive checks: explicit first-case observer recording allowed. No model/scorer imports. Truncated-log verification lazily replays the ORIGINAL
CPU PhysicalRateLimiter (NumPy required; use behavior6 Python). Default CLI is a
non-writing CPU plan. Only
--execute may launch the frozen single-episode evaluator. No resume or retry.
The reported metric is system completion, not neural-policy-only performance.
"""
from __future__ import annotations
from scripts.eval_zhuhai_precision_fixed_attempts import write_json, verify_external_owner

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
    require(type(evidence.get('video_recording')) is bool and evidence.get('observer_video_only') is evidence['video_recording'], 'explicit observer recording rule binding required')
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
    require(raw.get('record_video') is (seed == 932106) and raw.get('third_view') is (seed == 932106) and raw.get('observer_policy_input') is False and raw.get('max_control_steps') == 2400 and
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
            'the three onboard policy input contract must remain unchanged')
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
        require(rule.get('video_recording') is raw['record_video'], 'case observer flag mismatch')
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

