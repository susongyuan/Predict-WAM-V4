"""Admit proven physical-contact stops as failures, keeping runtime safety intact."""
from pathlib import Path
import ast
import math
import zipfile
from scripts import precision_continuity_evidence_v37 as e
from scripts import eval_zhuhai_precision_continuity_v37 as evaluator
from harness import zhuhai_chunk_continuity_v37 as braking

ERROR = 'InfrastructureError: unsafe/missing actual physics_safety; never assign policy success'
REASONS = frozenset(('inter_arm_collision', 'non_adjacent_self_collision',
    'robot_support_collision', 'non_finger_table_collision', 'robot_ground_collision'))
CHECKS = ['standard manufacturer position/velocity/effort limits', 'full-arm convex colliders',
    'gravity enabled', 'non-adjacent self/inter-arm contacts',
    'non-finger table/ground/support contacts']


def terminal_contact(rule, previous):
    e.validate_rule(previous)
    e.require(isinstance(rule, dict) and type(rule.get('video_recording')) is bool and rule.get('observer_video_only') is rule['video_recording'] and rule['video_recording'] is previous['video_recording'],
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
    changed = evaluator.overlay(frozen)
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
    from scripts.precision_continuity_replay_v37 import replay,video_evidence
    replayed=replay(path.parent,changed['action_postprocessing'])
    brake_steps=replayed['braking_steps'];max_delta=replayed['max_command_adjustment_rad']
    files.update(video_evidence(path.parent,raw))
    e.require(e.sha256_file(path) == identity['sha256'], 'Raw contact result changed during admission')
    return dict(task=e.TASK, seed=seed, status='physics_safety_abort', full_task_success=False,
        base_score=None, observed_base_score=score['base_score'], control_steps=count,
        raw_reported_safe_control_steps=reported, attempt_count=1,
        raw_result_path=str(path.resolve()), raw_result_sha256=identity['sha256'], evidence_files=files,
        braking_steps=brake_steps, max_command_adjustment_rad=max_delta,
        deterministic_command_replay_verified=True, fine_commands_verified=count*25, independent_finite_difference_limits_verified=True, actual_endpoint_archives_verified=True, safety_events=terminal['physics_safety']['events'],
        elapsed_physics_seconds=terminal['physics_time_seconds']-rules[0]['physics_time_seconds'],
        score_policy='last safe frame only; contact stop always counts as one failure',
        count_policy='archive proves one dispatched interval before safety exception/bookkeeping')


def classify(path, metadata, frozen, source_hash, seeds, prior_classifier):
    raw = e.read_json(path)
    if raw.get('status') == 'infrastructure_error' and raw.get('error') == ERROR:
        return classify_contact(path, metadata, frozen, source_hash, seeds)
    return prior_classifier(path, metadata, frozen, source_hash)


# Preserve all original archive, NPZ, action, clock, binding and hash checks.
from scripts.precision_continuity_evidence_v37 import (TASK, FORMAT, FIELDS, require,
    snapshot, loads, finite, validate_rule, sha256_file)

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
        require(rule.get('video_recording') is raw['record_video'], 'case observer flag mismatch')
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
    return identities
