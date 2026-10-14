#!/usr/bin/env python3
"""One-shot, offline RoboTwin cache -> CPU preflight -> CUDA smoke -> 30k train.

No execution, writes, GPU queries or systemd calls without --execute. Use the WAM
Python environment for YAML (OmegaConf); JSON configurations need only stdlib.
The original config is passed unchanged: max_steps=schedule_steps=50000;
only the train CLI stop boundary is 30000. Smoke is a separate, fresh run.

Admission JSON is a strict *operator attestation*, not automated visual inspection.
Print the contract with --print-admission-template; an operator (human or explicitly
identified assistant visual reviewer) must replace the
placeholders, set all four booleans true, and supply SHA256-bound RGB/native
preflight evidence. Paths in config/admission are project-root relative (not
relative to their JSON/YAML file). SOURCE.json uses the cache's ancestor lookup.
The split hash binds the exact split bytes, not a new scan of all episode bytes;
the existing cache and trainer remain responsible for their data validation.

--out MUST NOT EXIST, even if empty. Its train/ directory is the posttrain
--training-run-dir; use the same retained --unit as --training-unit. A single
systemd user service owns the entire chain (Restart=no, RemainAfterExit=yes).
Linger must already be enabled for logout persistence; this script never enables
it, resets/stops other units, steals locks, retries or resumes training. Each GPU
stage releases only its own lock in finally; ExecStopPost is a SIGKILL fallback.
A permanent exclusive worker claim also forbids manual unit restart/replay.
Logs: logs/{01_feature_cache,02_cpu_preflight,03_cuda_smoke,04_train}.log;
pipeline_status.json + events.jsonl; trainer's train/status.json is never edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
TARGET = 30000
SCHEDULE = 50000
FLAGS = ('rgb_real_image_visible', 'native_preflight_passed', 'interface_accepted',
         'operator_confirmed')
ADMISSION_KEYS = {'schema_version', 'task', 'source_split_sha256', *FLAGS,
                  'operator', 'confirmed_at', 'note', 'evidence'}


def resolve(path):
    return (ROOT / Path(path).expanduser()).resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique, parse_constant=invalid)


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def write_new(path, value):
    with Path(path).open('x') as f:
        f.write(json_text(value))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json_text(value))
    tmp.replace(path)


def identity(path):
    path = resolve(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('required nonempty file: ' + str(path))
    return {'path': str(path), 'sha256': sha256(path)}


def load_config(path):
    if Path(path).suffix.lower() == '.json':
        return read_json(path)
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def admission_template():
    return {'schema_version': 1, 'task': '<exact dataset.task_name>',
            'source_split_sha256': '<SHA256 of the source split_manifest.json>',
            **{key: False for key in FLAGS}, 'operator': '<operator name>',
            'confirmed_at': '<timezone-aware ISO-8601 timestamp>',
            'note': '<what RGB first frame and native interface were verified>',
            'evidence': {'first_frame_rgb': {'path': '<saved real RGB image>', 'sha256': '<SHA256>'},
                         'native_preflight': {'path': '<native preflight report>', 'sha256': '<SHA256>'}}}


def validate_admission(path, task, split_sha):
    audit = read_json(path)
    if not isinstance(audit, dict) or set(audit) != ADMISSION_KEYS:
        raise ValueError('admission keys must exactly match --print-admission-template')
    if type(audit['schema_version']) is not int or audit['schema_version'] != 1:
        raise ValueError('admission schema_version must be integer 1')
    if audit['task'] != task or audit['source_split_sha256'] != split_sha:
        raise ValueError('admission task/source split SHA mismatch')
    for key in FLAGS:
        if audit[key] is not True:
            raise ValueError('admission requires ' + key + '=true')
    for key in ('operator', 'confirmed_at', 'note'):
        if not isinstance(audit[key], str) or not audit[key].strip() or '<' in audit[key]:
            raise ValueError('admission requires a filled ' + key)
    date = datetime.fromisoformat(audit['confirmed_at'].replace('Z', '+00:00'))
    if date.tzinfo is None or date.utcoffset() is None:
        raise ValueError('confirmed_at must include a timezone')
    evidence = audit['evidence']
    if not isinstance(evidence, dict) or set(evidence) != {'first_frame_rgb', 'native_preflight'}:
        raise ValueError('requires RGB first-frame and native preflight evidence')
    records = []
    for key, value in evidence.items():
        if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
            raise ValueError('evidence requires exactly path and sha256: ' + key)
        record = identity(value['path'])
        if not isinstance(value['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', value['sha256']):
            raise ValueError('invalid evidence SHA256: ' + key)
        if value['sha256'] != record['sha256']:
            raise ValueError('evidence SHA mismatch: ' + key)
        records.append(record)
    return audit, records


def absent(path):
    if os.path.lexists(path):
        raise FileExistsError('refusing existing output (no retry/resume/overwrite): ' + str(path))


def build_plan(args):
    # Check the lexical leaf BEFORE resolving: a dangling symlink also exists.
    absent(ROOT / Path(args.out).expanduser())
    config_path, admission_path, out = map(resolve, (args.config, args.admission_json, args.out))
    absent(out)
    config_record, admission_record = identity(config_path), identity(admission_path)
    cfg = load_config(config_path)
    dataset, training = cfg['dataset'], cfg['training']
    task = dataset['task_name']
    if not isinstance(task, str) or not re.fullmatch('[a-z0-9_]+', task):
        raise ValueError('invalid task name')
    for key in ('max_steps', 'schedule_steps'):
        if type(training[key]) is not int or training[key] != SCHEDULE:
            raise ValueError('keep training.' + key + '=50000; stop is CLI-only')
    split = identity(dataset['split_manifest'])
    if split['sha256'] != dataset['split_sha256']:
        raise ValueError('config source split SHA mismatch')
    manifest = read_json(split['path'])
    data_root = resolve(manifest['data_root'])
    source_path = next((p / 'SOURCE.json' for p in (data_root, *data_root.parents)
                        if (p / 'SOURCE.json').is_file()), None)
    if source_path is None:
        raise ValueError('dataset SOURCE.json task mismatch or missing')
    source_record = identity(source_path)
    if read_json(source_path).get('task') != task:
        raise ValueError('dataset SOURCE.json task mismatch or missing')
    audit, evidence = validate_admission(admission_path, task, split['sha256'])
    cache = resolve(dataset['feature_cache'])
    for protected in (cache, data_root):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError('pipeline output must be separate from cache/data root')
    unit = args.unit or 'rw-train-' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    if not re.fullmatch('[A-Za-z0-9_-]{1,120}', unit):
        raise ValueError('unit must be a bare safe name, no .service suffix')
    # Do NOT resolve a venv Python symlink: doing so silently loses the venv.
    python = os.path.abspath(os.path.expanduser(args.python))
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError('Python is not executable: ' + python)
    if not math.isfinite(args.gpu_lock_wait_seconds) or args.gpu_lock_wait_seconds < 0:
        raise ValueError('GPU lock wait must be finite and nonnegative')
    if args.cache_batch <= 0:
        raise ValueError('cache batch must be positive')
    lock_owner = unit + '-' + uuid.uuid4().hex
    base = [python, '-B', str(ROOT / 'scripts/train_robotwin_wam.py'), '--config', str(config_path)]
    stages = []
    commands = [
        ('feature_cache', True, None,
         [python, '-B', str(ROOT / 'scripts/cache_robotwin_features.py'),
          '--split-manifest', split['path'], '--out', str(cache), '--splits', 'train', 'val',
          '--batch', str(args.cache_batch), '--device', 'cuda']),
        ('cpu_preflight', False, out / 'cpu_preflight', base + ['--mode', 'plan', '--device', 'cpu']),
        ('cuda_smoke', True, out / 'cuda_smoke', base + ['--mode', 'smoke', '--device', 'cuda',
          '--smoke-steps', '2', '--smoke-batch-size', '1', '--gpu-lock-owner', lock_owner]),
        ('train', True, out / 'train', base + ['--mode', 'train', '--device', 'cuda',
          '--max-steps', str(TARGET), '--gpu-lock-owner', lock_owner]),
    ]
    for number, (name, gpu, stage_out, command) in enumerate(commands, 1):
        stages.append({'name': name, 'gpu': gpu, 'out': str(stage_out) if stage_out else None,
                       'command': command + (['--out', str(stage_out)] if stage_out else []),
                       'log': str(out / 'logs' / f'{number:02d}_{name}.log')})
    files = [SELF, ROOT / 'scripts/gpu_lock.py',
             ROOT / 'scripts/cache_robotwin_features.py', ROOT / 'scripts/train_robotwin_wam.py']
    plan = {'schema_version': 1, 'out': str(out), 'unit': unit, 'python': python,
            'config': str(config_path), 'task': task, 'admission': audit, 'lock_owner': lock_owner,
            'inputs': [config_record, admission_record, source_record, split] + [identity(p) for p in files] + evidence,
            'target': TARGET, 'schedule_budget': SCHEDULE,
            'gpu_lock_wait_seconds': args.gpu_lock_wait_seconds, 'stages': stages,
            'posttrain': {'training_run_dir': str(out / 'train'), 'training_unit': unit}}
    verify_inputs(plan)
    return plan


def systemd_command(plan, manifest_sha):
    out, unit = Path(plan['out']), plan['unit']
    # systemd parses ExecStopPost as an Exec* value, not shell syntax.
    cleanup = '/usr/bin/python3 ' + str(ROOT / 'scripts/gpu_lock.py') + ' release --owner ' + plan['lock_owner']
    return ['systemd-run', '--user', '--quiet', '--unit=' + unit, '--service-type=exec',
            '--property=WorkingDirectory=' + str(ROOT), '--property=Restart=no',
            '--property=RemainAfterExit=yes', '--property=KillMode=control-group',
            '--property=TimeoutStopSec=120', '--property=ExecStopPost=' + cleanup,
            '--setenv=PYTHONUNBUFFERED=1', '--setenv=PYTHONDONTWRITEBYTECODE=1',
            '--setenv=HF_HUB_OFFLINE=1', '--setenv=TRANSFORMERS_OFFLINE=1',
            plan['python'], '-B', str(SELF), '--execute',
            '--_worker-manifest', str(out / 'pipeline_manifest.json'), '--_manifest-sha', manifest_sha]


def verify_inputs(plan):
    for record in plan['inputs']:
        if identity(record['path']) != record:
            raise ValueError('admitted input changed: ' + record['path'])


def run_child(command, log, *, gpu):
    """Wait serially; terminate/reap only this child's group before releasing lock."""
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    if not gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    with Path(log).open('x') as stream:
        stream.write('$ ' + shlex.join(command) + '\n')
        stream.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait()
        except BaseException:
            # A second stop signal cannot interrupt reaping and release too early.
            previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            raise


def stage_success(stage):
    if stage['name'] == 'feature_cache':
        return  # The following CPU preflight validates complete cached payloads.
    out = Path(stage['out'])
    if stage['name'] == 'cpu_preflight':
        value = read_json(out / 'plan.json')
        if (value.get('status') != 'cache_preflight_passed_not_a_smoke'
                or value.get('mode') != 'plan' or value.get('model_constructed') is not False
                or value.get('schedule_budget') != SCHEDULE):
            raise ValueError('CPU preflight did not certify cached data')
    else:
        value = read_json(out / 'status.json')
        mode, target = ('smoke', 2) if stage['name'] == 'cuda_smoke' else ('train', TARGET)
        if (value.get('status') not in ('complete', 'bounded_complete') or value.get('mode') != mode
                or type(value.get('step')) is not int or value['step'] != target
                or type(value.get('target')) is not int or value['target'] != target
                or value.get('schedule_budget') != SCHEDULE):
            raise ValueError('stage did not complete its exact target: ' + stage['name'])
        if mode == 'train':
            for step in (25000, 30000):
                ck = out / f'checkpoint_step_{step:06d}.pt'
                receipt = ck.with_name(ck.name + '.sha256.json')
                if not ck.is_file() or ck.stat().st_size == 0 or not receipt.is_file():
                    raise ValueError('missing posttrain checkpoint/receipt: ' + str(ck))
                record = read_json(receipt)
                if (record.get('step') != step or record.get('model_only') is not True
                        or record.get('size') != ck.stat().st_size
                        or not re.fullmatch('[0-9a-f]{64}', str(record.get('sha256', '')))):
                    raise ValueError('invalid posttrain checkpoint receipt: ' + str(receipt))
                # The evaluator does the full checkpoint hash; no tensor load here.


class StopPipeline(BaseException):
    pass


def stop_signal(signum, frame):
    raise StopPipeline('signal ' + str(signum))


def run_pipeline(plan, *, runner=run_child, lock=None):
    """Internal worker; injectable runner/lock keep independent unit tests CPU-only."""
    out = Path(plan['out'])
    # Never erase this claim: even a failed attempt cannot replay training.
    write_new(out / 'worker_claim.json', {'pid': os.getpid(), 'started_at': time.time()})
    state = {'status': 'running', 'stage': 'admission', 'unit': plan['unit'],
             'target': TARGET, 'schedule_budget': SCHEDULE, 'pid': os.getpid(),
             'stages': {s['name']: {'status': 'pending', 'log': s['log']} for s in plan['stages']}}

    def emit(status, **extra):
        state.update(status=status, updated_at=time.time(), **extra)
        atomic_json(out / 'pipeline_status.json', state)
        with (out / 'events.jsonl').open('a') as f:
            f.write(json.dumps(state, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps({'stage': state['stage'], 'status': status, **extra}), flush=True)

    previous = {s: signal.signal(s, stop_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    current = None
    try:
        verify_inputs(plan)
        for stage in plan['stages']:
            current = state['stages'][stage['name']]
            state['stage'] = stage['name']
            verify_inputs(plan)
            if stage['out']:
                absent(stage['out'])
            current.update(status='waiting_gpu' if stage['gpu'] else 'running', started_at=time.time())
            emit(current['status'])

            def execute_stage():
                verify_inputs(plan)  # also recheck after waiting for someone else's GPU
                if stage['out']:
                    absent(stage['out'])
                current['status'] = 'running'
                emit('running')
                code = runner(stage['command'], stage['log'], gpu=stage['gpu'])
                current['returncode'] = code
                if code != 0:
                    raise RuntimeError(f"{stage['name']} exited {code}; see {stage['log']}")
                verify_inputs(plan)
                stage_success(stage)

            if stage['gpu']:
                if lock is None:
                    if str(ROOT) not in sys.path:
                        sys.path.insert(0, str(ROOT))
                    from scripts import gpu_lock as lock
                acquired = False
                try:
                    lock.acquire(plan['lock_owner'], max_wait=plan['gpu_lock_wait_seconds'])
                    acquired = True
                    execute_stage()
                finally:
                    # Owner-checked even if acquisition is interrupted just after creation.
                    released = lock.release(plan['lock_owner'])
                    if acquired and not released:
                        raise RuntimeError('GPU lock ownership lost; no foreign lock touched')
            else:
                execute_stage()
            current.update(status='complete', finished_at=time.time())
            emit('running')
        emit('complete', stage='complete', finished_at=time.time())
        return 0
    except BaseException as exc:
        terminal = 'stopped' if isinstance(exc, (StopPipeline, KeyboardInterrupt)) else 'failed'
        if current is not None:
            current.update(status=terminal, finished_at=time.time())
        emit(terminal, error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        return 130 if terminal == 'stopped' else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def submit(plan):
    """Reserve once, then submit once. An ambiguous submission is NEVER retried."""
    linger = subprocess.run(['loginctl', 'show-user', str(os.getuid()), '-p', 'Linger', '--value'],
                            check=True, capture_output=True, text=True, timeout=15)
    if linger.stdout.strip() != 'yes':
        raise RuntimeError('user Linger must already be yes for persistent execution; no settings changed')
    existing = subprocess.run(['systemctl', '--user', 'show', plan['unit'], '-p', 'LoadState', '--value'],
                              capture_output=True, text=True, timeout=15)
    if existing.stdout.strip() != 'not-found':
        raise RuntimeError('unit already exists or user manager unavailable; no reset/stop attempted')
    verify_inputs(plan)
    out = Path(plan['out'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    (out / 'logs').mkdir()
    manifest = out / 'pipeline_manifest.json'
    write_new(manifest, plan)
    write_new(out / 'pipeline_status.json', {'status': 'submitted', 'stage': 'admission', 'unit': plan['unit']})
    command = systemd_command(plan, sha256(manifest))
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except BaseException as exc:
        # systemd may have accepted the job before the client failed. Do not race
        # its worker by replacing pipeline_status, and do not submit a second job.
        write_new(out / 'submission.json', {'status': 'submission_unconfirmed_no_retry',
                  'error': f'{type(exc).__name__}: {exc}', 'command': command})
        raise
    write_new(out / 'submission.json', {'status': 'accepted', 'command': command,
                                       'stdout': result.stdout, 'stderr': result.stderr})
    return {'status': 'submitted_not_training_complete', 'out': str(out), 'unit': plan['unit'],
            'posttrain': plan['posttrain'], 'status_file': str(out / 'pipeline_status.json')}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', help='existing, unchanged 50k training configuration')
    parser.add_argument('--admission-json', help='completed strict operator/native first-frame attestation')
    parser.add_argument('--out', help='NEW pipeline directory, not the trainer directory')
    parser.add_argument('--unit', help='fresh systemd bare unit name (default derived from out)')
    parser.add_argument('--python', default=sys.executable, help='WAM interpreter; venv path preserved')
    parser.add_argument('--cache-batch', type=int, default=16)
    parser.add_argument('--gpu-lock-wait-seconds', type=float, default=86400)
    parser.add_argument('--execute', action='store_true', help='submit exactly one persistent systemd unit')
    parser.add_argument('--print-admission-template', action='store_true')
    parser.add_argument('--_worker-manifest', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest-sha', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker_manifest:
        if not args.execute or not args._manifest_sha:
            parser.error('internal worker requires --execute and manifest SHA')
    elif not args.print_admission_template and not all((args.config, args.admission_json, args.out)):
        parser.error('--config, --admission-json and --out are required')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.print_admission_template:
        print(json_text(admission_template()), end='')
        return 0
    if args._worker_manifest:
        if not os.environ.get('INVOCATION_ID'):
            raise ValueError('internal worker must run under systemd; use public --execute')
        path = resolve(args._worker_manifest)
        if sha256(path) != args._manifest_sha:
            raise ValueError('submitted pipeline manifest changed')
        plan = read_json(path)
        if path != Path(plan['out']) / 'pipeline_manifest.json':
            raise ValueError('pipeline manifest/output mismatch')
        return run_pipeline(plan)
    plan = build_plan(args)
    if not args.execute:
        print(json_text({'status': 'plan_only_no_jobs_started', 'plan': plan,
                         'systemd_command': systemd_command(plan, '<manifest SHA set at submission>')}), end='')
        return 0
    print(json_text(submit(plan)), end='')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)
