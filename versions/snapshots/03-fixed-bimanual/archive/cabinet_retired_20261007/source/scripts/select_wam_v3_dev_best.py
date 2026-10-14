#!/usr/bin/env python3
"""Index fixed-plan task0 dev10 checkpoints, without torch, GPU or trainer writes.

API: update_best(stages_dir, root=ROOT), after the controller persists a stage.
CLI: python3 -B scripts/select_wam_v3_dev_best.py --stages-dir PATH

Rank ALL valid physical results by success desc, final Q desc, step asc. If all
fail, the link denotes only the highest-Q failed development candidate, NEVER a
successful checkpoint. No valid results => null best and no link. Invalid stages
are recorded, excluded, and cause CLI exit 2 (task failures do not).

JSON and symlink replacements are individually atomic, not a two-file transaction.
The JSON is authoritative; a repeat invocation revalidates and repairs the link.
Only selection outputs are written; no weights are loaded, copied or removed.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
TASK = 'turning_on_radio'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, expected):
    return type(value) is int and value == expected


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def read(path):
    value = json.loads(path.read_text(), object_pairs_hook=_object, parse_constant=_bad_constant)
    require(isinstance(value, dict), f'expected JSON object: {path}')
    return value


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def path_at(root, value):
    require(isinstance(value, str) and bool(value), 'missing/non-string evidence path')
    return (root / value).resolve()


def atomic_json(path, value):
    require(not path.is_symlink(), f'refusing JSON symlink: {path}')
    data = encoded(value)
    if path.exists() and path.read_text() == data:
        return  # Byte- and mtime-idempotent for unchanged evidence.
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_link(path, target):
    require(path.is_symlink() or not path.exists(), f'refusing to overwrite non-symlink: {path}')
    if target is None:
        if path.is_symlink():
            path.unlink()  # Remove only this index link, NEVER the checkpoint target.
        return
    target = str(target)
    if path.is_symlink() and os.readlink(path) == target:
        return
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        os.symlink(target, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_plan(plan, root):
    require(integer(plan.get('version'), 1), 'unsupported plan version')
    budget, steps = plan.get('total_optimizer_steps'), plan.get('checkpoints')
    require(type(budget) is int and budget in (30000, 50000), 'invalid plan budget')
    p, training = plan.get('physical'), plan.get('training')
    require(isinstance(p, dict) and isinstance(training, dict), 'invalid plan sections')
    continuation = training.get('continuation_plan')
    start = 30000 if continuation else 0
    require(integer(plan.get('start_step', 0), start), 'invalid plan start_step')
    if continuation:
        require(budget == 50000, 'invalid continuation budget')
        cp_path = path_at(root, continuation)
        require(digest(cp_path) == training.get('continuation_plan_sha256'),
                'continuation plan SHA256 mismatch')
        cp = read(cp_path)
        require(cp.get('format') == 'wam-v3-continuation-20k-v1'
                and encoded(cp.get('schedule')) == encoded({
                    'start_step': 30000, 'additional_steps': 20000, 'warmup_steps': 500,
                    'start_lr': 5e-6, 'peak_lr': 2e-5, 'final_lr': 5e-6,
                    'kind': 'linear_then_cosine_completed_global_updates'}),
                'invalid continuation schedule')
        parent = cp.get('parent', {})
        require(path_at(root, training.get('resume_source')) == path_at(root, parent.get('checkpoint'))
                and training.get('config_sha256') == parent.get('config_sha256'),
                'continuation parent/config mismatch')
    require(isinstance(steps, list) and all(type(s) is int for s in steps)
            and steps == list(range(start + 5000, budget + 1, 5000)), 'invalid original plan checkpoints')
    for key, value in {'profile': 'development', 'instance': 10, 'seed': 0, 'samples': 1,
                       'exec_horizon': 8, 'task_id': 0, 'attempts_per_interval': 1,
                       'fixed_program': True, 'video': [1280, 720]}.items():
        require(encoded(p.get(key)) == encoded(value), f'fixed physical plan mismatch: {key}')
    require(p.get('video_view', 'viewer') == 'viewer' and integer(p.get('max_attempts', 3), 3),
            'fixed viewer/attempts mismatch')
    require(training.get('optimizer_resume') is True and training.get('early_stopping') is False
            and plan.get('automatic_extension') is False and plan.get('no_test_feedback_to_training') is True,
            'training protocol mismatch')
    require(digest(path_at(root, training.get('config'))) == training.get('config_sha256'),
            'original config SHA256 mismatch')


def episode_key(row):
    require(isinstance(row, dict) and row.get('task') == TASK
            and type(row.get('instance_id')) is int and row['instance_id'] >= 0
            and integer(row.get('rollout_id'), 0), 'invalid episode identity')
    return row['task'], row['instance_id'], row['rollout_id']


def validate_candidate(row, stage_file, step, prefix, root):
    """Mirror controller validate_physical, plus stage/interval-sidecar binding.

    Instance index 10 need not resolve to instance ID 10. Eval-plan resolved ID
    binds all three episode records. official_score_valid is NOT a ranking gate.
    The controller owns video decoding/sidecar gating; preserve its evidence path.
    """
    name = f'{prefix}_s{step:06d}'
    physical = f'{name}_dev10_k1'
    ck = root / 'outputs/training' / name / f'checkpoint_step_{step:06d}.pt'
    phys = root / 'outputs/behavior/closed_loop' / physical
    require(integer(row.get('step_target'), step) and row.get('training_name') == name
            and row.get('unit') == f'wam-train-{name}.service'
            and row.get('physical_name') == physical, 'stage original-plan binding mismatch')
    require(integer(row.get('returncode'), 0), 'stage returncode not zero')
    require(row.get('physical_success_not_inferred') is True, 'missing evaluator-only success receipt')
    require(ck.is_file() and not ck.is_symlink() and ck.resolve() == ck,
            'missing/redirected exact interval checkpoint')
    require(path_at(root, row.get('checkpoint')) == ck, 'stage checkpoint path mismatch')
    receipt_file = ck.with_name(ck.name + '.sha256.json')
    receipt = read(receipt_file)
    require(integer(receipt.get('step'), step) and receipt.get('model_only') is True
            and integer(receipt.get('size'), ck.stat().st_size), 'interval checkpoint sidecar mismatch')
    sha = digest(ck)
    require(receipt.get('sha256') == sha and row.get('checkpoint_sha256') == sha,
            'interval/stage checkpoint SHA256 mismatch')
    man_file = phys / 'manifest.json'
    man = read(man_file)
    require(encoded(row.get('physical_manifest')) == encoded(man), 'stage manifest snapshot mismatch')
    require(man.get('evaluation_complete') is True, 'evaluation incomplete')
    require(all(integer(man.get(k), 0) for k in ('evaluator_returncode', 'launcher_returncode')),
            'evaluation/launcher returncode not zero')
    require(man.get('run_name') == physical and man.get('profile') == 'development'
            and integer(man.get('task_id'), 0) and man.get('task_name') == TASK,
            'manifest run/task/profile mismatch')
    require(path_at(root, man.get('checkpoint')) == ck and man.get('checkpoint_sha256') == sha,
            'manifest checkpoint mismatch')
    require(encoded(man.get('instances_override')) == encoded([10])
            and integer(man.get('samples_averaged'), 1) and integer(man.get('exec_horizon'), 8),
            'manifest instance/K/horizon mismatch')
    require(all(integer(man.get(k), 0) for k in ('seed', 'effective_policy_seed', 'effective_evaluator_seed')),
            'manifest seed mismatch')
    require(type(man.get('official_score_valid')) is bool, 'invalid official_score_valid flag')
    ep_file = phys / 'eval/evaluation_plan.json'
    ep = read(ep_file)
    require(ep.get('task') == TASK and ep.get('mode') == 'public_test'
            and encoded(ep.get('instance_indices')) == encoded([10])
            and integer(ep.get('seed'), 0) and integer(ep.get('num_rollouts'), 1), 'evaluator plan mismatch')
    ids = ep.get('instance_ids')
    require(isinstance(ids, list) and len(ids) == 1 and type(ids[0]) is int and ids[0] >= 0,
            'invalid resolved instance IDs')
    expected = (TASK, ids[0], 0)
    require(integer(man.get('expected_episodes'), 1), 'wrong expected episode count')
    for key in ('planned_episodes', 'completed_episodes'):
        rows = man.get(key)
        require(isinstance(rows, list) and len(rows) == 1 and episode_key(rows[0]) == expected,
                f'{key} binding mismatch')
    files = sorted((phys / 'eval/json').glob('*.json'))
    require(len(files) == 1, 'missing/duplicate evaluator JSON')
    raw = read(files[0])
    require(episode_key(raw) == expected, 'evaluator result binding mismatch')
    q = raw.get('q_score', {}).get('final') if isinstance(raw.get('q_score'), dict) else None
    require(type(raw.get('success')) is bool and type(q) in (int, float)
            and math.isfinite(q) and 0 <= q <= 1, 'invalid evaluator success/final Q')
    require(type(raw.get('steps')) is int and raw['steps'] > 0, 'invalid evaluator steps')
    result = man['completed_episodes'][0]
    require(type(result.get('success')) is bool and result['success'] == raw['success']
            and type(result.get('q_score')) in (int, float) and result['q_score'] == q
            and integer(result.get('steps'), raw['steps']), 'manifest/evaluator score mismatch')
    for source in (raw, result):
        require(all(integer(source.get(k), 0) for k in ('effective_policy_seed', 'effective_evaluator_seed')),
                'episode seed mismatch')
    return {'step': step, 'checkpoint': str(ck), 'checkpoint_sha256': sha,
            'physical_name': physical, 'success': raw['success'], 'final_q': q,
            'video_evidence': row.get('video_evidence'),
            'official_score_valid': man['official_score_valid'],
            'episode': {'task': TASK, 'instance_id': ids[0], 'rollout_id': 0, 'steps': raw['steps']},
            'evidence': {key: {'path': str(path), 'sha256': digest(path)} for key, path in (
                ('stage', stage_file), ('checkpoint_receipt', receipt_file), ('manifest', man_file),
                ('evaluation_plan', ep_file), ('evaluation_json', files[0]))}}


def update_best(stages_dir, root=ROOT):
    """Return/persist full audit index. Bad plan raises; bad stages are excluded.

    This uses a separate CPU selection mutex, never controller.lock or GPU locks.
    Intended for invocation after an atomic stage write, not during evidence edits.
    """
    root = Path(root).resolve()
    out = (root / stages_dir).resolve()
    require(out.is_dir() and out.name.endswith('_stages'), 'expected existing PREFIX_stages directory')
    prefix = out.name[:-len('_stages')]
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,60}', prefix) is not None, 'unsafe stage prefix')
    require(out == root / 'outputs/training' / f'{prefix}_stages', 'stages directory outside original layout')
    with (out / 'selection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan_file = out / 'plan.json'
        plan = read(plan_file)
        validate_plan(plan, root)
        protocol_file = out / 'selection_protocol.json'
        protocol = {'version': 1, 'plan_sha256': digest(plan_file), 'prefix': prefix,
                    'checkpoints': plan['checkpoints'],
                    'eligibility': 'original-plan physical_complete; task0 dev10 seed0 K1 exec8; one episode',
                    'ranking': ['success descending', 'final Q descending', 'step ascending'],
                    'all_failed': 'index highest-Q failed candidate; explicitly no successful checkpoint',
                    'no_results': 'best=null; no checkpoint_best_task.pt symlink',
                    'official_score_valid_is_ranking_gate': False,
                    'rule_origin': 'user predeclared before first physical result',
                    'scope': 'development selection only; not official reporting or training feedback'}
        require(not protocol_file.is_symlink(), 'refusing protocol symlink')
        if protocol_file.exists():
            stored = read(protocol_file)
            require({k: v for k, v in stored.items() if k != 'registered_unix'} == protocol,
                    'selection protocol/original plan changed; refusing reinterpretation')
        else:
            atomic_json(protocol_file, {**protocol, 'registered_unix': time.time()})
        candidates, rejected, pending = [], [], []
        expected_files = {f'stage_{s:06d}.json': s for s in plan['checkpoints']}
        found = {p.name: p for p in out.glob('stage_*.json')}
        for filename in sorted(expected_files.keys() | found.keys()):
            path = out / filename
            step = expected_files.get(filename)
            if filename not in found:
                pending.append({'step': step, 'stage_file': str(path), 'reason': 'no stage receipt'})
                continue
            try:
                require(step is not None, 'stage not bound to original plan')
                row = read(path)
                require(integer(row.get('step_target'), step), 'stage step/filename mismatch')
                if row.get('phase') != 'physical_complete':
                    pending.append({'step': step, 'stage_file': str(path), 'phase': row.get('phase'),
                                    'reason': 'not physical_complete'})
                    continue
                candidates.append(validate_candidate(row, path, step, prefix, root))
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
                rejected.append({'step': step, 'stage_file': str(path),
                                 'error': f'{type(error).__name__}: {error}'})
        best = max(candidates, key=lambda c: (c['success'], c['final_q'], -c['step']), default=None)
        has_success = any(c['success'] for c in candidates)
        status = 'successful_checkpoint' if has_success else ('no_successful_checkpoint' if best else 'no_valid_results')
        index = {'version': 1, 'plan_sha256': protocol['plan_sha256'],
                 'selection_protocol_sha256': digest(protocol_file), 'status': status,
                 'message': '已有成功开发checkpoint（非官方报告）' if has_success else '尚无成功checkpoint',
                 'has_successful_checkpoint': has_success, 'best': best,
                 'scope': protocol['scope'], 'candidate_count': len(candidates), 'rejected_count': len(rejected),
                 'candidates': candidates, 'rejected': rejected, 'pending': pending}
        require(not (out / 'best_task.json').is_symlink(), 'refusing best JSON symlink')
        atomic_link(out / 'checkpoint_best_task.pt', best['checkpoint'] if best else None)
        atomic_json(out / 'best_task.json', index)
        return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages-dir', required=True)
    parser.add_argument('--root', type=Path, default=ROOT, help='project root (also useful for CPU fixtures)')
    args = parser.parse_args(argv)
    try:
        index = update_best(args.stages_dir, root=args.root)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        print(f'selection refused: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    best = index['best']
    print(json.dumps({'status': index['status'], 'message': index['message'],
                      'best_step': best['step'] if best else None, 'candidate_count': index['candidate_count'],
                      'rejected_count': index['rejected_count']}, ensure_ascii=False))
    return 2 if index['rejected'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
