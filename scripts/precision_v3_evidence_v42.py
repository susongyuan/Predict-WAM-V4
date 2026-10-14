"""Independent V3 archive admission, original official braking and physical rules.

Derived strict evidence checks retain all original action/observation/clock/hashes.
Only V3 identity and the explicitly requested first observer recording differ.
No simulator, policy inference, retry, source mutation or score reassignment.
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
SEEDS = (932126, 932127, 932128, 932129, 932130)
STEPS = (30000,)
TASK = 'zhuhai_task2'
FORMAT = 'zhuhai-precision-wam-native16-measured-10hz-v3'
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



from copy import deepcopy
from harness import zhuhai_official_braking as braking

e = sys.modules[__name__]
ACTUAL_PRECISION_BINDING = dict(version=3, refined_flow_steps=32, dense_kv_tokens=588,
    queries_per_view=16, global_visual_tokens=48, patch_projection_input_dim=1029,
    ray_scope='fixed_approximate_camera_frame_direction_no_depth_or_extrinsics', executed_prefix=8)


def changed_contract(frozen):
    result=deepcopy(frozen)
    result['action_postprocessing']=braking.protocol()
    result['action_postprocessing_sha256']=digest(result['action_postprocessing'])
    return result


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def validate_rule(evidence):
    require(isinstance(evidence, dict), 'original rule evidence missing')
    score = evidence.get('score', {})
    require(finite(evidence.get('physics_time_seconds')) and evidence['physics_time_seconds'] >= 0,
            'invalid original physics clock')
    require(type(score.get('full_task_success')) is bool and finite(score.get('base_score')) and
            score.get('rule_source') and evidence.get('objects'), 'invalid original rule score/evidence')
    safety = evidence.get('physics_safety', {})
    require(safety.get('passed') is True and safety.get('events') == [] and safety.get('checks'), 'physics safety evidence not passed')
    require(type(evidence.get('video_recording')) is bool, 'explicit recording flag required')
    require(evidence.get('observer_video_only', False) is evidence['video_recording'], 'observer-only flag differs')
    return score


def validate_binding(raw, step, seed, sha, contract=None):
    require(raw.get('task') == TASK and type(raw.get('seed')) is int and raw['seed'] == seed, 'task/seed mismatch')
    ready = raw.get('policy_ready', {})
    runtime = ready.get('runtime', {})
    require(ready.get('kind') == 'ready' and ready.get('version') == 1 and ready.get('episode') == 'startup', 'policy_ready missing/invalid')
    require(runtime.get('checkpoint_sha256') == sha and runtime.get('checkpoint_step') == step,
            'policy_ready checkpoint SHA/step mismatch')
    require(runtime.get('format') == FORMAT and runtime.get('task') == TASK and runtime.get('strict_load') is True and
            runtime.get('projections_reset') is False, 'strict V3 policy binding required')
    require(raw.get('record_video') is (seed == SEEDS[0]) and raw.get('third_view') is (seed == SEEDS[0])
            and raw.get('observer_policy_input') is False and raw.get('max_control_steps') == 2400 and
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
    require(runtime.get('precision_v3_actual') == ACTUAL_PRECISION_BINDING, 'actual V3 full32 architecture mismatch')
    require(runtime.get('system_mode') == 'precision_wam_v3_official_braking', 'unexpected policy system mode')
    for field in ('dino_frozen', 'precision_refiner_used_online', 'event_clock_used_online'):
        require(runtime.get(field) is True, f'required V3 mechanism missing: {field}')
    for field in ('hjepa_auxiliary_heads_used_online', 'event_clock_is_verified_grasp'):
        require(runtime.get(field) is False, f'privileged/misrepresented online mechanism: {field}')
    dc = runtime['data_contract']
    require(dc.get('record_video') is False and dc.get('third_view') is False and dc.get('camera_count') == 3
            and dc.get('camera_names') == list(FIELDS[:3]),
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
        require(rule.get('video_recording') is raw['record_video'], 'case observer recording flag changed')
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
    verify_native_payloads(directory, raw, count)
    return identities


def verify_native_payloads(directory, raw, count):
    """Validate actual payloads and dtype-exact previous-command receipts, not only headers."""
    import numpy as np
    actions = [loads(line) for line in (Path(directory)/'actions.jsonl').read_text().splitlines()]
    require(len(actions) == count, 'payload/action interval count differs')
    for index in range(count + 1):
        with np.load(Path(directory)/'onboard'/f'observation_{index:06d}.npz', allow_pickle=False) as z:
            require(set(z.files) == set(FIELDS), 'extra or missing sensor payload')
            measured = z['measured_joint_gripper_state']
            commanded = z['commanded_joint_gripper_state']
            for vector in (measured, commanded):
                require(vector.shape == (16,) and vector.dtype.kind == 'f' and np.isfinite(vector).all(),
                        'nonfinite or invalid native16 payload')
            require(np.all((commanded[[7, 15]] >= 0) & (commanded[[7, 15]] <= 1)), 'commanded gripper range differs')
            for view in FIELDS[:3]:
                rgb = z[view]
                require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
                        and min(rgb.shape) > 0, 'invalid RGB payload')
            if index:
                expected = np.asarray(actions[index-1]['dispatched_action'], dtype=commanded.dtype)
                require(np.array_equal(commanded, expected), 'actual driver target differs from archived dispatch')



def video_evidence(directory,raw):
    directory=Path(directory)
    if raw['seed']!=932126:
        e.require(not list(directory.glob('third_person*')),'unexpected observer artifacts on unrecorded case')
        return {}
    receipt=e.read_json(directory/'third_person_video_receipt.json')
    e.require((raw.get('observer_video_receipt')==receipt) and receipt['seed']==932126
        and receipt['frames']==len((directory/'actions.jsonl').read_text().splitlines())+1
        and receipt['fps']==10 and receipt['policy_reads_observer'] is False
        and receipt['physics_advanced_by_recording'] is False and receipt['success_selection'] is False,
        'requested observer binding/count differs')
    e.require(receipt.get('status') == 'encoded'
        and receipt.get('path') == str((directory/'third_person_full.mp4').resolve())
        and receipt.get('camera_path') == '/World/RequestedObserverCamera'
        and receipt.get('resolution_wh') == [960, 720]
        and receipt.get('policy_inputs_unchanged') == list(FIELDS[:3])
        and receipt.get('duration_seconds') == receipt['frames']/10,
        'observer identity, policy inputs, resolution or duration differs')
    video=directory/'third_person_full.mp4'
    e.require(receipt['video_sha256']==e.sha256_file(video),'observer video SHA differs')
    frames=directory/'third_person_frames.jsonl'
    e.require(receipt['frame_index_sha256']==e.sha256_file(frames),'observer frame index SHA differs')
    entries=[e.loads(line) for line in frames.read_text().splitlines()]
    e.require(len(entries)==receipt['frames'] and all(r['index']==i and abs(r['physics_time_seconds']-i*.1)<1e-6
        for i,r in enumerate(entries)),'observer physical frame clock differs')
    ffmpeg=next((e.ROOT/'.conda-envs/behavior6/lib/python3.12/site-packages/imageio_ffmpeg/binaries').glob('ffmpeg-linux-*'))
    result=subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-i',str(video),'-map','0:v:0',
        '-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True,timeout=180)
    decoded=[int(line.split('=',1)[1]) for line in result.stdout.splitlines() if line.startswith('frame=')]
    e.require(result.returncode==0 and decoded and max(decoded)==receipt['frames'],'full observer video decode failed')
    evidence={}
    for filename in ('third_person_full.mp4','third_person_frames.jsonl','third_person_video_receipt.json','third_person_encoder.log'):
        p=directory/filename;evidence[filename]=dict(path=str(p.resolve()),sha256=e.sha256_file(p),size=p.stat().st_size)
    return evidence

def classify_noncontact(path, metadata, frozen, source_hash):
    path = Path(path)
    raw = e.read_json(path)
    abort = raw.get('status') == 'infrastructure_error'
    if abort:
        prefix = f"InfrastructureError: wrong worker reply/ID, expected action/{e.TASK}_seed_{raw.get('seed')}/{raw.get('control_steps')}: "
        error = raw.get('error', '')
        e.require(error.startswith(prefix), 'Unknown infrastructure error; preserve and HALT')
        reply = ast.literal_eval(error[len(prefix):])
        e.require(reply == dict(version=1, kind='infrastructure_error', episode='startup', step=0,
            error='ValueError: '+braking.SIGNATURE), 'Unknown worker rejection; preserve and HALT')
    else:
        e.require(raw.get('status') == 'completed' and not raw.get('error'), 'Nonterminal or infrastructure result')
    seed = raw.get('seed')
    e.require(seed in SEEDS and 0 < raw.get('control_steps', 0) <= 2400, 'Invalid physical attempt')
    changed = changed_contract(frozen)
    runtime = e.validate_binding(raw, metadata['step'], seed, metadata['sha256'], changed)
    e.require(runtime.get('evaluation_version') == braking.VERSION
        and runtime.get('evaluation_source_sha256') == source_hash
        and runtime.get('training_contract_sha256') == e.digest(frozen), 'Evaluation overlay identity mismatch')
    terminal = raw.get('last_rule_evidence' if abort else 'final_rule_evidence')
    score = e.validate_rule(terminal)
    if abort:
        e.require(raw.get('full_task_success') is None and raw.get('base_score') is None
            and score['full_task_success'] is False, 'Unsuccessful abort raw evidence changed')
    else:
        e.require(raw.get('full_task_success') == score['full_task_success']
            and raw.get('base_score') == score['base_score'], 'Raw score differs from physical evidence')
        e.require((score['full_task_success'] and raw.get('termination') == 'full_rule_success')
            or (not score['full_task_success'] and raw.get('termination') == 'fixed_step_limit'
                and raw['control_steps'] == 2400), 'Invalid termination/budget')
    files = e.archive_evidence(path.parent, raw, runtime, terminal)
    import numpy as np
    with np.load(path.parent/'onboard/observation_000000.npz', allow_pickle=False) as z:
        initial = z['commanded_joint_gripper_state'].copy()
    limiter = braking.BrakingRateLimiter(changed['action_postprocessing']['rate_limit']); limiter.reset(initial)
    brake_steps = 0; max_delta = 0.
    for line in (path.parent/'actions.jsonl').read_text().splitlines():
        action = e.loads(line)
        actual, stats = limiter.step(action['smoothed_action'])
        e.require(np.array_equal(actual, action['dispatched_action']), 'Discrete braking replay mismatch')
        e.require(stats == action['policy']['rate_limit_stats'], 'Limiter statistics replay mismatch')
        brake_steps += int(stats['braking_limited_dimensions'] > 0)
        max_delta = max(max_delta, stats['arm_limit_delta_max_rad'])
    if abort:
        lo, hi = limiter.feasible_interval()
        e.require(np.any(lo > hi+1e-10), 'Cannot prove terminal braking safety abort')
    files.update(video_evidence(path.parent,raw))
    return dict(task=e.TASK, seed=seed, status='control_safety_abort' if abort else 'completed',
        full_task_success=False if abort else raw['full_task_success'], base_score=raw['base_score'],
        observed_base_score=score['base_score'], control_steps=raw['control_steps'], attempt_count=1,
        raw_result_path=str(path.resolve()), raw_result_sha256=e.sha256_file(path),
        evidence_files=files, braking_steps=brake_steps, max_command_adjustment_rad=max_delta,
        deterministic_command_replay_verified=True)


ERROR = 'InfrastructureError: unsafe/missing actual physics_safety; never assign policy success'
REASONS = frozenset(('inter_arm_collision', 'non_adjacent_self_collision',
    'robot_support_collision', 'non_finger_table_collision', 'robot_ground_collision'))
CHECKS = ['standard manufacturer position/velocity/effort limits', 'full-arm convex colliders',
    'gravity enabled', 'non-adjacent self/inter-arm contacts',
    'non-finger table/ground/support contacts']


def terminal_contact(rule, previous):
    e.validate_rule(previous)
    e.require(isinstance(rule, dict) and type(rule.get('video_recording')) is bool
        and rule.get('observer_video_only', False) is rule['video_recording'],
        'Contact terminal evidence missing or video changed')
    now = rule.get('physics_time_seconds'); before = previous['physics_time_seconds']
    e.require(e.finite(now) and math.isclose(now-before, .1, abs_tol=1e-6, rel_tol=0),
        'Contact terminal clock must follow the last safe control interval')
    score = rule.get('score', {})
    e.require(type(score.get('full_task_success')) is bool and e.finite(score.get('base_score'))
        and score.get('rule_source') == previous['score']['rule_source'] and rule.get('objects'),
        'Invalid contact terminal rule score')
    safety = rule.get('physics_safety', {})
    e.require(safety.get('passed') is False and safety.get('checks') == CHECKS
        and previous['physics_safety'].get('checks') == CHECKS
        and safety.get('scope') == previous['physics_safety'].get('scope'),
        'Contact monitor identity or failure flag differs')
    events = safety.get('events')
    e.require(isinstance(events, list) and events, 'Missing actual contact event')
    unique = set()
    for event in events:
        e.require(isinstance(event, dict) and set(event) == {'time_seconds','bodies','reason'},
            'Unknown actual contact event schema')
        bodies = event['bodies']; reason = event['reason']; when = event['time_seconds']
        e.require(reason in REASONS and isinstance(bodies, list) and len(bodies) == 2
            and all(isinstance(b, str) and b.startswith('/World/') for b in bodies)
            and bodies[0] != bodies[1] and any(b.startswith('/World/Robot/') for b in bodies),
            'Unknown contact reason or actual body identity')
        e.require(e.finite(when) and before-1e-6 <= when <= now+1e-6,
            'Contact event lies outside the terminal control interval')
        key = (tuple(bodies), reason)
        e.require(key not in unique, 'Duplicate actual contact event'); unique.add(key)
    return score


def classify_contact(path, metadata, frozen, source_hash, seeds):
    path = Path(path)
    data, identity = e.snapshot(path); raw = e.loads(data)
    e.require(raw.get('status') == 'infrastructure_error' and raw.get('error') == ERROR,
        'Only the exact actual physics-safety stop may be classified as a contact failure')
    reported = raw.get('control_steps'); seed = raw.get('seed')
    e.require(type(reported) is int and 0 <= reported < 2400 and seed in seeds,
        'Invalid physical contact attempt/budget')
    count = reported+1
    changed = changed_contract(frozen)
    runtime = e.validate_binding(raw, metadata['step'], seed, metadata['sha256'], changed)
    e.require(runtime.get('evaluation_version') == braking.VERSION
        and runtime.get('evaluation_source_sha256') == source_hash
        and runtime.get('training_contract_sha256') == e.digest(frozen),
        'Evaluation overlay identity mismatch')
    rules = [e.loads(line) for line in (path.parent/'rule_evidence.jsonl').read_text().splitlines()]
    e.require(len(rules) == count+1, 'Missing terminal contact observation')
    terminal = raw.get('last_rule_evidence')
    e.require({k:v for k,v in rules[-1].items() if k != 'control_step'} == terminal,
        'Actual contact terminal differs from raw result')
    terminal_contact(terminal, rules[-2])
    score = e.validate_rule(rules[-2])
    e.require(raw.get('full_task_success') is None and raw.get('base_score') is None
        and score['full_task_success'] is False
        and (raw.get('partial_base_score') == score['base_score']
            or (reported == 0 and raw.get('partial_base_score') is None and score['base_score'] == 0)),
        'Contact abort must keep raw success unknown and last safe score unchanged')
    if reported:
        e.require(e.finite(raw.get('elapsed_physics_seconds')) and math.isclose(
            raw['elapsed_physics_seconds'], rules[-2]['physics_time_seconds']-rules[0]['physics_time_seconds'],
            abs_tol=1e-6, rel_tol=0), 'Raw safe-step clock differs from archive')
    files = archive_contact_evidence(path.parent, raw, runtime, terminal, count)
    import numpy as np
    with np.load(path.parent/'onboard/observation_000000.npz', allow_pickle=False) as z:
        initial = z['commanded_joint_gripper_state'].copy()
    limiter = braking.BrakingRateLimiter(changed['action_postprocessing']['rate_limit'])
    limiter.reset(initial); brake_steps = 0; max_delta = 0.
    for line in (path.parent/'actions.jsonl').read_text().splitlines():
        action = e.loads(line); actual, stats = limiter.step(action['smoothed_action'])
        e.require(np.array_equal(actual, action['dispatched_action']), 'Discrete braking replay mismatch')
        e.require(stats == action['policy']['rate_limit_stats'], 'Limiter statistics replay mismatch')
        brake_steps += int(stats['braking_limited_dimensions'] > 0)
        max_delta = max(max_delta, stats['arm_limit_delta_max_rad'])
    e.require(e.sha256_file(path) == identity['sha256'], 'Raw contact result changed during admission')
    files.update(video_evidence(path.parent,raw))
    return dict(task=e.TASK, seed=seed, status='physics_safety_abort', full_task_success=False,
        base_score=None, observed_base_score=score['base_score'], control_steps=count,
        raw_reported_safe_control_steps=reported, attempt_count=1,
        raw_result_path=str(path.resolve()), raw_result_sha256=identity['sha256'], evidence_files=files,
        braking_steps=brake_steps, max_command_adjustment_rad=max_delta,
        deterministic_command_replay_verified=True, safety_events=terminal['physics_safety']['events'],
        elapsed_physics_seconds=terminal['physics_time_seconds']-rules[0]['physics_time_seconds'],
        score_policy='last safe frame only; contact stop always counts as one failure',
        count_policy='archive proves one dispatched interval before safety exception/bookkeeping')


def classify(path, metadata, frozen, source_hash, seeds, prior_classifier):
    raw = e.read_json(path)
    if raw.get('status') == 'infrastructure_error' and raw.get('error') == ERROR:
        return classify_contact(path, metadata, frozen, source_hash, seeds)
    return prior_classifier(path, metadata, frozen, source_hash)


def archive_contact_evidence(directory, raw, runtime, terminal, count):
    """Verify dispatched count against action, rule and actual onboard archives."""
    require(count == raw['control_steps']+1, 'Physical exception bookkeeping differs')
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
        require(rule.get('video_recording') is raw['record_video'], 'case observer flag differs')
        score = terminal_contact(rule, rules[i-1]) if i == count else validate_rule(rule)
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
    verify_native_payloads(directory, raw, count)
    return identities
