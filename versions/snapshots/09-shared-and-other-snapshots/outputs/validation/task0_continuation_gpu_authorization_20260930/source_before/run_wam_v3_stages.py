#!/usr/bin/env python3
"""GPU-serial 5k training -> fixed development evaluation -> optimizer resume.

Restart ONLY with --continue-existing --prefix ORIGINAL (the original plan is immutable).
The controller never kills a trainer, steals a GPU lock, or retries a physical episode.
Task failure is valid evaluation evidence, not an infrastructure failure or a success claim.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GPU_LOCK = Path('/tmp/project2_gpu.lock')
TASK = 'turning_on_radio'
LIVE = {'active', 'activating', 'deactivating', 'reloading'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, expected):
    return type(value) is int and value == expected


def wait_release(timeout=90):
    deadline = time.monotonic() + timeout
    while GPU_LOCK.exists() or GPU_LOCK.is_symlink():
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU lock not released; refusing overlap (no stealing)')
        time.sleep(2)


def unit_live(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
                             '-p', 'LoadState', '-p', 'ActiveState', '-p', 'MainPID',
                             '-p', 'Result', '-p', 'ExecMainStatus'],
                            capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, f'systemctl query failed: {unit}: {result.stderr}')
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    loaded, state, pid = (fields.get(k, '') for k in ('LoadState', 'ActiveState', 'MainPID'))
    require(loaded in ('loaded', 'not-found') and pid.isdecimal()
            and state in LIVE | {'inactive', 'failed'}, f'unknown unit state: {unit}: {fields}')
    if loaded == 'not-found':
        require(state == 'inactive' and int(pid) == 0, f'inconsistent missing unit: {fields}')
        return False
    if int(pid) > 0 or state in LIVE:
        return True
    require(state != 'failed' and fields.get('Result') == 'success'
            and fields.get('ExecMainStatus') == '0', f'training unit failed: {fields}')
    return False


def validate_last(path, step=None):
    """Hash before any downstream unpickling; schema/optimizer boundary delegated to trainer.

    train_wam_v3.save_checkpoint writes these exact metadata fields in its sidecar.
    No torch import, GPU work, or checkpoint deserialization occurs here.
    """
    require(path.is_file(), f'missing last checkpoint: {path}')
    receipt = read(path.with_name(path.name + '.sha256.json'))
    require(isinstance(receipt, dict), 'invalid checkpoint sidecar object')
    require(type(receipt.get('step')) is int and receipt['step'] >= 0,
            'invalid checkpoint sidecar step')
    require(step is None or receipt['step'] == step, 'last checkpoint step mismatch')
    require(receipt.get('model_only') is False, 'last checkpoint is model-only')
    require(integer(receipt.get('size'), path.stat().st_size), 'last checkpoint size mismatch')
    sha = digest(path)
    require(receipt.get('sha256') == sha, 'last checkpoint SHA256 mismatch')
    return receipt


def episode_key(row):
    require(isinstance(row, dict) and row.get('task') == TASK
            and type(row.get('instance_id')) is int and row['instance_id'] >= 0
            and integer(row.get('rollout_id'), 0), 'invalid episode identity')
    return row['task'], row['instance_id'], row['rollout_id']


def validate_physical(phys, physical, ck, ck_sha, instance):
    man = read(phys / 'manifest.json')
    require(isinstance(man, dict), 'invalid manifest object')
    require(man.get('evaluation_complete') is True, 'evaluation incomplete')
    require(all(integer(man.get(k), 0) for k in ('evaluator_returncode', 'launcher_returncode')),
            'evaluation/launcher returncode not zero')
    require(man.get('run_name') == physical and man.get('profile') == 'development'
            and integer(man.get('task_id'), 0) and man.get('task_name') == TASK,
            'manifest run/task/profile mismatch')
    require(Path(man.get('checkpoint', '')).resolve() == ck.resolve()
            and man.get('checkpoint_sha256') == ck_sha, 'manifest checkpoint mismatch')
    require(man.get('instances_override') == [instance]
            and integer(man.get('samples_averaged'), 1) and integer(man.get('exec_horizon'), 8),
            'manifest instance/K/horizon mismatch')
    require(all(integer(man.get(k), 0) for k in
                ('seed', 'effective_policy_seed', 'effective_evaluator_seed')), 'manifest seed mismatch')
    plan = read(phys / 'eval/evaluation_plan.json')
    require(isinstance(plan, dict), 'invalid evaluator plan object')
    require(plan.get('task') == TASK and plan.get('mode') == 'public_test'
            and plan.get('instance_indices') == [instance] and integer(plan.get('seed'), 0)
            and integer(plan.get('num_rollouts'), 1), 'evaluator plan mismatch')
    ids = plan.get('instance_ids')
    require(isinstance(ids, list) and len(ids) == 1 and type(ids[0]) is int and ids[0] >= 0,
            'invalid resolved instance IDs')
    expected = {(TASK, ids[0], 0)}  # Instance index is not necessarily the resolved instance ID.
    require(integer(man.get('expected_episodes'), 1), 'wrong expected episode count')
    planned, completed = man.get('planned_episodes'), man.get('completed_episodes')
    require(isinstance(planned, list) and len(planned) == 1
            and {episode_key(r) for r in planned} == expected, 'planned episode binding mismatch')
    require(isinstance(completed, list) and len(completed) == 1
            and {episode_key(r) for r in completed} == expected, 'completed episode binding mismatch')
    files = sorted((phys / 'eval/json').glob('*.json'))
    require(len(files) == 1, 'missing/duplicate evaluator JSON')
    raw = read(files[0])
    require(episode_key(raw) in expected, 'evaluator result binding mismatch')
    q = raw.get('q_score', {}).get('final') if isinstance(raw.get('q_score'), dict) else None
    require(type(raw.get('success')) is bool and type(q) in (int, float)
            and math.isfinite(q) and 0 <= q <= 1, 'invalid evaluator success/final Q')
    require(type(raw.get('steps')) is int and raw['steps'] > 0, 'invalid evaluator steps')
    result = completed[0]
    require(type(result.get('success')) is bool and result['success'] == raw['success']
            and type(result.get('q_score')) in (int, float) and result['q_score'] == q
            and type(result.get('steps')) is int and result['steps'] == raw['steps'],
            'manifest completed score differs from evaluator JSON')
    for source in (raw, result):
        require(all(integer(source.get(k), 0) for k in
                    ('effective_policy_seed', 'effective_evaluator_seed')), 'episode seed mismatch')
    # official_score_valid is deliberately preserved; Isaac6 is NOT the official pin.
    return man


def verify_video(phys, out, step):
    """CPU full decode plus exact episode/sidecar coverage; not a task-success test."""
    from check_closed_loop_video import check_run
    report = check_run(phys, pyav_python=ROOT / '.conda-envs/behavior6/bin/python')
    target = out / f'video_{step:06d}.json'
    write(target, report)
    require(report.get('all_verified') is True, f'video evidence not verified: {target}')
    return str(target)


def publish_best(out):
    # Separate from optimizer resume; only the user-facing development checkpoint index.
    from select_wam_v3_dev_best import update_best
    return update_best(out, root=ROOT)


def process_token(pid):
    """Read-only process identity: avoids attaching a reused PID; no kill(0)/signals."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def clean_env():
    env = dict(os.environ)
    for key in ('RESUME', 'INIT', 'RUN_NAME', 'RESUME_SHA256', 'DRY_RUN', 'MAXSTEPS',
                'ORCH_EXTRA', 'VIDEO_VIEW', 'MAX_ATTEMPTS', 'CONTINUATION_PLAN'):
        env.pop(key, None)
    return env


def new_plan(a):
    continuation = getattr(a, 'continuation_plan', None)
    return {'version': 1, 'created_unix': time.time(), 'total_optimizer_steps': a.budget or 30000,
            'checkpoints': list(range(35000 if continuation else 5000, (a.budget or 30000) + 1, 5000)),
            'start_step': 30000 if continuation else 0,
            'training': {'config': a.config, 'config_sha256': digest(ROOT / a.config),
                         'approval': a.approval, 'cache_ledger': a.cache_ledger,
                         'continuation_plan': continuation,
                         'continuation_plan_sha256': digest(ROOT / continuation) if continuation else None,
                         'resume_source': a.resume_source, 'optimizer_resume': True, 'early_stopping': False},
            'physical': {'profile': 'development', 'instance': 10 if a.instance is None else a.instance,
                         'seed': 0, 'samples': 1, 'exec_horizon': 8, 'task_id': 0,
                         'checkpoint_rule': 'exact interval checkpoint, NOT retrospectively best',
                         'attempts_per_interval': 1, 'video': [1280, 720], 'video_view': 'viewer',
                         'max_attempts': 3, 'fixed_program': True,
                         'success': 'only evaluator success/BDDL and final Q; no timeout or program-end success',
                         'scope': 'development checkpoint tracking, not official report'},
            'resource': 'exclusive GPU; wait before and after each stage; never steal lock',
            'physical_failure': 'record and continue unchanged training',
            'infra_failure': 'stop chain, preserve all evidence', 'no_test_feedback_to_training': True,
            'automatic_extension': False}


def validate_plan(plan, a):
    t, p = plan['training'], plan['physical']
    budget = plan['total_optimizer_steps']
    continuation = t.get('continuation_plan')
    start = 30000 if continuation else 0
    require(type(budget) is int and budget in ((50000,) if continuation else (30000, 50000))
            and integer(plan.get('start_step', 0), start)
            and all(type(step) is int for step in plan['checkpoints'])
            and plan['checkpoints'] == list(range(start + 5000, budget + 1, 5000)), 'invalid stage plan')
    if continuation:
        cp = read(ROOT / continuation)
        require(digest(ROOT / continuation) == t['continuation_plan_sha256'], 'continuation plan changed')
        require(cp['format'] == 'wam-v3-continuation-20k-v1'
                and cp['schedule'] == {'start_step': 30000, 'additional_steps': 20000,
                    'warmup_steps': 500, 'start_lr': 5e-6, 'peak_lr': 2e-5, 'final_lr': 5e-6,
                    'kind': 'linear_then_cosine_completed_global_updates'},
                'invalid continuation budget/schedule')
        require(a.prefix.startswith(cp['run_prefix']), 'independent continuation prefix required')
        require(Path(ROOT / t['resume_source']).resolve() == Path(ROOT / cp['parent']['checkpoint']).resolve(),
                'controller must start from pinned parent last')
        require(digest(ROOT / t['resume_source']) == cp['parent']['sha256'], 'parent SHA mismatch')
        require(t['config_sha256'] == cp['parent']['config_sha256'], 'continuation config differs')
    for key in ('config', 'cache_ledger', 'approval', 'resume_source', 'continuation_plan'):
        require(getattr(a, key, None) is None or getattr(a, key, None) == t.get(key), f'original plan {key} differs')
    require(a.budget is None or a.budget == budget, 'original budget differs')
    require(a.instance is None or a.instance == p['instance'], 'original instance differs')
    require(type(p['instance']) is int and p['instance'] in range(10, 20), 'development instances only')
    for key, value in {'profile': 'development', 'seed': 0, 'samples': 1, 'exec_horizon': 8,
                       'task_id': 0, 'attempts_per_interval': 1, 'fixed_program': True,
                       'video': [1280, 720]}.items():
        require(p.get(key) == value and type(p.get(key)) is type(value), f'fixed plan mismatch: {key}')
    require(p.get('video_view', 'viewer') == 'viewer' and integer(p.get('max_attempts', 3), 3),
            'fixed viewer/attempts mismatch')
    require(t.get('optimizer_resume') is True and t.get('early_stopping') is False
            and plan.get('automatic_extension') is False and plan.get('no_test_feedback_to_training') is True,
            'training protocol differs')
    require(digest(ROOT / t['config']) == t['config_sha256'], 'original config digest changed')


def require_execution_authorization(plan):
    """Unconditional continuation hard stop; no CLI/env override exists."""
    if plan['training'].get('continuation_plan'):
        raise ValueError('continuation GPU execution blocked: resource-window authorization unresolved')


def execute(plan, a, out):
    require_execution_authorization(plan)
    t, p = plan['training'], plan['physical']
    row = {'phase': 'preflight', 'time': time.time()}
    try:
        require((ROOT / t['approval']).is_file(), 'reviewed approval missing')
        resume = t.get('resume_source')
        initial_step = validate_last(ROOT / resume)['step'] if resume else 0
        for step in plan['checkpoints']:
            if step <= initial_step:
                continue
            name = f'{a.prefix}_s{step:06d}'
            unit = f'wam-train-{name}.service'
            physical = f'{name}_dev{p["instance"]}_k1'
            run = ROOT / 'outputs/training' / name
            phys = ROOT / 'outputs/behavior/closed_loop' / physical
            stage_file = out / f'stage_{step:06d}.json'
            old = read(stage_file) if stage_file.exists() else {}
            status = read(out / 'status.json') if (out / 'status.json').exists() else {}
            prior = {**old, **status} if status.get('step_target') == step else old
            row = {'phase': 'inspect_training', 'step_target': step, 'training_name': name,
                   'unit': unit, 'physical_name': physical, 'time': time.time()}
            for key in ('physical_launcher_pid', 'physical_launcher_start', 'returncode'):
                if key in prior:
                    row[key] = prior[key]
            # Completed stages must not wait on a lock owned by a LATER, already-active trainer.
            if old.get('phase') == 'physical_complete':
                row['phase'] = 'validate_completed_stage'
                require(read(run / 'status.json').get('status') == 'budget_complete'
                        and integer(read(run / 'summary.json').get('completed_updates'), step),
                        'completed stage training evidence mismatch')
                last = run / 'checkpoint_last.pt'
                validate_last(last, step)
                ck = run / f'checkpoint_step_{step:06d}.pt'
                ck_sha = digest(ck)
                require(old.get('checkpoint_sha256') == ck_sha and old.get('physical_name') == physical
                        and integer(old.get('step_target'), step) and integer(old.get('returncode'), 0),
                        'completed stage receipt binding mismatch')
                validate_physical(phys, physical, ck, ck_sha, p['instance'])
                verify_video(phys, out, step)
                publish_best(out)
                resume = str(last)
                continue
            # Query BEFORE considering any launch. Existing active runs are attached, never relaunched.
            active = unit_live(unit)
            if not run.exists() and not active:
                require(not phys.exists() and not old, 'orphan stage evidence; refusing retraining')
                row['phase'] = 'wait_gpu_before_training'
                write(out / 'status.json', row)
                wait_release()
                env = clean_env()
                env.update(NAME=name, MODE='long', MAX_STEPS=str(step), CONFIG=t['config'],
                           CACHE_LEDGER=t['cache_ledger'], APPROVAL=t['approval'], P2_TASK_ID='0')
                if t.get('continuation_plan'):
                    env['CONTINUATION_PLAN'] = str(ROOT / t['continuation_plan'])
                if resume:
                    env['RESUME'] = str(ROOT / resume)
                    env['RESUME_SHA256'] = validate_last(ROOT / resume)['sha256']
                row['phase'] = 'launch_training'
                write(out / 'status.json', row)
                subprocess.run([str(ROOT / 'scripts/run_wam_v3_unit.sh')], cwd=ROOT, env=env, check=True)
                active = unit_live(unit)
            row['phase'] = 'attach_training' if active else 'validate_training'
            write(out / 'status.json', row)
            while active:
                time.sleep(15)
                active = unit_live(unit)
            summary, train_status = read(run / 'summary.json'), read(run / 'status.json')
            if train_status.get('status') != 'budget_complete' or not integer(summary.get('completed_updates'), step):
                row.update(phase='stopped_training_failure', observed_status=train_status)
                write(out / 'status.json', row)
                return 2
            wait_release()
            last = run / 'checkpoint_last.pt'
            validate_last(last, step)
            ck = run / f'checkpoint_step_{step:06d}.pt'
            require(ck.is_file(), f'exact interval checkpoint absent: {ck}')
            ck_sha = digest(ck)
            row.update(checkpoint=str(ck), checkpoint_sha256=ck_sha, phase='physical_test')
            pid, token = prior.get('physical_launcher_pid'), prior.get('physical_launcher_start')
            physical_active = type(pid) is int and token is not None and process_token(pid) == token
            if phys.exists() or physical_active:
                # A completed directory can recover a crash before stage receipt persistence.
                # An incomplete one is NEVER re-launched. Attach only an exact persisted PID identity.
                if type(pid) is int and token is not None:
                    row.update(physical_launcher_pid=pid, physical_launcher_start=token)
                    while process_token(pid) == token:
                        write(out / 'status.json', {**row, 'phase': 'attach_physical'})
                        time.sleep(15)
                wait_release()
                require('returncode' not in prior or integer(prior['returncode'], 0),
                        'recorded physical launcher failure; refusing to accept success')
                man = validate_physical(phys, physical, ck, ck_sha, p['instance'])
                if old.get('phase') == 'physical_complete':
                    require(old.get('checkpoint_sha256') == ck_sha and old.get('physical_name') == physical
                            and integer(old.get('step_target'), step), 'stage receipt binding mismatch')
            else:
                require(not old and not (out / f'physical_{step:06d}.log').exists(),
                        'prior physical attempt exists; refusing automatic re-test')
                wait_release()
                env = clean_env()
                env.update(NAME=physical, CKPT=str(ck), INST=str(p['instance']), PROFILE='development',
                           SAMPLES='1', EXEC='8', VIDEO_W='1280', VIDEO_H='720', VIDEO_VIEW='viewer',
                           MAX_ATTEMPTS='3', P2_TASK_ID='0')
                write(out / 'status.json', row)
                with (out / f'physical_{step:06d}.log').open('x') as log:
                    proc = subprocess.Popen([str(ROOT / 'scripts/run_task0_closed_loop.sh')],
                                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    row.update(physical_launcher_pid=proc.pid, physical_launcher_start=process_token(proc.pid))
                    write(out / 'status.json', row)
                    rc = proc.wait()
                row['returncode'] = rc
                write(out / 'status.json', row)
                wait_release()
                require(integer(rc, 0), f'physical launcher failed: {rc}')
                man = validate_physical(phys, physical, ck, ck_sha, p['instance'])
            row.update(phase='verify_video', returncode=0)
            write(out / 'status.json', row)
            video_evidence = verify_video(phys, out, step)
            row.update(phase='physical_complete', returncode=0, physical_manifest=man,
                       video_evidence=video_evidence,
                       physical_success_not_inferred=True, time=time.time())
            # No summarizer subprocess: evaluator JSON + manifest are authoritative.
            write(stage_file, row)
            write(out / 'status.json', row)
            publish_best(out)
            resume = str(last)
        write(out / 'status.json', {'phase': 'budget_complete', 'steps': plan['total_optimizer_steps'],
                                   'automatic_extension': False, 'task_capability_claimed': False,
                                   'time': time.time()})
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, subprocess.SubprocessError) as error:
        row.update(failed_phase=row.get('phase'), phase='stopped_infrastructure_failure',
                   error=f'{type(error).__name__}: {error}', time=time.time())
        write(out / 'status.json', row)
        print(row['error'], file=sys.stderr)
        return 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    for key in ('config', 'cache-ledger', 'approval', 'resume-source', 'continuation-plan'):
        parser.add_argument('--' + key)
    parser.add_argument('--budget', type=int, choices=[30000, 50000])
    parser.add_argument('--instance', type=int)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--continue-existing', action='store_true')
    a = parser.parse_args()
    require(a.prefix.replace('_', '').replace('-', '').isalnum() and len(a.prefix) <= 60, 'safe prefix required')
    out = ROOT / 'outputs/training' / f'{a.prefix}_stages'
    if a.continue_existing:
        plan = read(out / 'plan.json')
    else:
        require(all((a.config, a.cache_ledger, a.approval)), 'new plan requires config/cache-ledger/approval')
        plan = new_plan(a)
    validate_plan(plan, a)
    if a.continue_existing and a.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if not a.continue_existing:
        out.mkdir(parents=True, exist_ok=False)
        write(out / 'plan.json', plan)
    if a.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    # Separate CPU-controller mutex; does not touch the GPU lock or any trainer.
    with (out / 'controller.lock').open('a') as controller_lock:
        try:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another controller owns this prefix; refusing concurrent continuation', file=sys.stderr)
            return 3
        return execute(plan, a, out)


if __name__ == '__main__':
    raise SystemExit(main())
