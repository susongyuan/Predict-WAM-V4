#!/usr/bin/env python3
"""Serial, externally bounded native evaluations; never count interruption as failure.

The dedicated post-training pipeline supplies a training gate and a preregistered
five-seed protocol. Legacy manifests remain usable with an explicit training run.
No queue process acquires a GPU lock: each child acquires it only after admission.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.robotwin_policy_worker import atomic_json, sha256


class StopQueue(BaseException):
    pass


def stop_signal(signum, frame):
    raise StopQueue(f'signal {signum}')


def evaluation_wall_budgets(job):
    """Predeclared action-proportional limits; preserve legacy <=700-step defaults."""
    steps = job.get('max_steps', 700)
    if type(steps) is not int or not 1 <= steps <= 2000:
        raise ValueError('max_steps must be integer [1,2000]')
    episode = job.get('episode_wall_seconds', 1200 if steps <= 700 else max(1200, steps * 3))
    total = job.get('total_wall_seconds', episode + 300)
    external = job.get('external_wall_seconds', total + 60)
    if (any(type(x) is not int for x in (episode, total, external))
            or not 1 <= episode < total < external <= 86400):
        raise ValueError('invalid predeclared episode/total/external wall budgets')
    return {'episode_wall_seconds': episode, 'total_wall_seconds': total,
            'external_wall_seconds': external}


def command(job):
    owner = job['unit']
    budgets = evaluation_wall_budgets(job)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', owner):
        raise ValueError('unsafe systemd unit name')
    mode = job.get('system_mode', 'pure')
    if mode not in ('pure', 'transparent', 'integrated', 'middle'):
        raise ValueError('unsupported system_mode: ' + str(mode))
    if mode == 'pure' and (any(job.get(k) for k in ('vtt_adapter', 'vtt_bank', 'target_head'))
                           or job.get('execution_mode', 'native8') != 'native8'):
        raise ValueError('pure WAM forbids VTT, target heads and recovery/feedback execution')
    training_run = job.get('training_run_dir')
    if not training_run:
        raise ValueError('explicit training_run_dir required; no task-specific default')
    entry = job.get('evaluator_entry', 'scripts/eval_robotwin_wam.py')
    if entry not in ('scripts/eval_robotwin_wam.py', 'scripts/eval_robotwin_offscreen.py', 'scripts/eval_robotwin_novideo.py'):
        raise ValueError('unsupported evaluator entry')
    extra = ['--system-mode', mode]
    if job.get('record_onboard', False):
        extra.append('--record-onboard')
    for key in ('vtt_adapter', 'vtt_bank', 'target_head', 'camera_readback', 'execution_mode', 'task_config'):
        if job.get(key):
            extra.extend(['--' + key.replace('_', '-'), str(job[key])])
    return ['systemd-run', '--user', '--quiet', '--unit=' + owner,
        '--property=WorkingDirectory=' + str(ROOT), '--property=RuntimeMaxSec=' + str(budgets['external_wall_seconds']),
        '--property=TimeoutStopSec=30', '--property=KillMode=control-group', '--property=RemainAfterExit=yes',
        '--property=ExecStopPost=/usr/bin/python3 ' + str(ROOT/'scripts/gpu_lock.py') + ' release --owner ' + owner,
        '--setenv=PYTHONUNBUFFERED=1', '--setenv=OMP_NUM_THREADS=2', '--setenv=MKL_NUM_THREADS=2',
        '/usr/bin/python3', 'scripts/gpu_lock.py', 'run', '--owner', owner, '--max-wait', '10', '--',
        str(ROOT/'.venvs/robotwin-sim/bin/python'), '-B', entry,
        '--run-dir', job['run_dir'], '--training-run-dir', str(training_run),
        '--checkpoint', job['checkpoint'], '--gpu-lock-owner', owner, '--render-denoiser', 'none',
        '--seeds', str(job['seed']), '--max-steps', str(job.get('max_steps', 700)),
        '--episode-wall-seconds', str(budgets['episode_wall_seconds']), '--total-wall-seconds', str(budgets['total_wall_seconds']),
        '--settle-steps', str(job.get('settle_steps', 120)), *extra]


def unit_state(unit):
    p = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState',
        '-p', 'Result', '-p', 'MainPID', '-p', 'LoadState', '-p', 'ExecMainCode', '-p', 'ExecMainStatus'],
        capture_output=True, text=True, timeout=15)
    return dict(line.split('=', 1) for line in p.stdout.splitlines() if '=' in line)


def log_progress(text):
    with (ROOT/'progress.md').open('a') as f:
        f.write('\n- ' + datetime.datetime.now().astimezone().isoformat(timespec='seconds') + ' 队列：' + text + '\n')


def read_result(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def classify_result(job, result, unit, require_video=False):
    """Only complete native terminal episodes are scored; errors always stay null."""
    episodes = result.get('episodes', [])
    episode = episodes[0] if len(episodes) == 1 else {}
    clean = (result.get('status') == 'completed' and episode.get('seed') == job['seed']
             and (not job.get('task') or result.get('task') == job['task']) and unit.get('Result') == 'success'
             and unit.get('ExecMainStatus', '0') == '0')
    if clean and episode.get('status') == 'instance_setup_failed' and result.get('scored_episodes') == 0:
        return 'invalid_initialization', None
    native = episode.get('evidence', {}).get('native_benchmark_success')
    valid = (clean and episode.get('seed') == job['seed'] and result.get('scored_episodes') == 1
             and type(native) is bool and result.get('native_successes') == int(native)
             and episode.get('status') == ('native_success' if native else 'step_limit'))
    if job.get('task') and result.get('task') != job['task']:
        valid = False
    if valid and require_video:
        directory = Path(job['run_dir']) / str(episode.get('episode', ''))
        meta = read_result(directory/'third_view_video.json')
        video = directory/'third_view.mp4'
        valid = (meta.get('complete') is True and meta.get('synthetic') is False
                 and meta.get('seed') == job['seed'] and meta.get('frames', 0) > 0
                 and video.is_file() and video.stat().st_size > 0)
    return ('valid_terminal', int(native)) if valid else ('infrastructure_incomplete', None)


def counts(jobs, attempts):
    report = {}
    for step in sorted({j['checkpoint_step'] for j in jobs}):
        planned = [j for j in jobs if j['checkpoint_step'] == step]
        rows = [r for r in attempts if r['checkpoint_step'] == step]
        valid = [r for r in rows if r['status'] == 'valid_terminal']
        successes = sum(r['native_successes'] for r in valid)
        report[str(step)] = dict(requested_episodes=len(planned), attempted_episodes=len(rows),
            scored_episodes=len(valid), native_successes=successes, native_failures=len(valid)-successes,
            invalid_initializations=sum(r['status'] == 'invalid_initialization' for r in rows),
            incomplete_attempts=sum(r['status'] not in ('valid_terminal', 'invalid_initialization') for r in rows),
            not_started=sum(not any(r['seed'] == j['seed'] for r in rows) for j in planned),
            unresolved_seeds=[j['seed'] for j in planned if not any(r['seed'] == j['seed'] for r in valid)],
            native_success_rate=successes/len(planned) if len(valid) == len(planned) else None)
    return report


def validate_manifest(manifest):
    jobs = manifest['jobs']
    seen, paths = set(), set()
    for j in jobs:
        if j['unit'] in seen:
            raise ValueError('duplicate unit')
        seen.add(j['unit'])
        if Path(j['run_dir']).exists():
            raise ValueError('refuse existing run ' + j['run_dir'])
        resolved = str(Path(j['run_dir']).resolve())
        if resolved in paths:
            raise ValueError('duplicate run directory')
        paths.add(resolved)
        if not Path(j['checkpoint']).is_file():
            raise ValueError('missing checkpoint')
    status_name = manifest.get('queue_status_filename', 'queue_status.json')
    if (not isinstance(status_name, str) or Path(status_name).name != status_name
            or not status_name.endswith('.json')):
        raise ValueError('queue_status_filename must be a local JSON filename')
    for j in jobs:
        if not j.get('training_run_dir'):
            j['training_run_dir'] = manifest.get('training_run_dir')
        command(j)
    if manifest.get('protocol') == 'pure_wam_30k_paired5_v1':
        seeds = manifest['seeds']
        if (len(seeds) != 5 or len(set(seeds)) != 5
                or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
            raise ValueError('paired protocol requires five distinct seeds')
        if [(j['checkpoint_step'], j['seed']) for j in jobs] != [(s, seed) for s in (25000, 30000) for seed in seeds]:
            raise ValueError('paired protocol must be serial 25k then 30k on identical seeds')
        if (manifest.get('require_video') is not True
                or any(j.get('system_mode', 'pure') != 'pure' or not j.get('record_onboard') for j in jobs)):
            raise ValueError('paired protocol requires pure WAM with onboard recording')
    return status_name


def run_manifest(manifest, path, *, before_job=None):
    status_path = Path(path).parent / validate_manifest(manifest)
    if status_path.exists():
        raise ValueError('refuse existing queue state; never auto-replay completed jobs')
    if manifest.get('protocol') == 'pure_wam_30k_paired5_v1' and before_job is None:
        raise ValueError('post-training protocol must run through its training-gated pipeline')
    state = {'status': 'running', 'pid': os.getpid(), 'jobs': [], 'current_unit': None}
    jobs, failures, current = manifest['jobs'], 0, None
    previous = {sig: signal.signal(sig, stop_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    def save():
        state['counts'] = counts(jobs, state['jobs'])
        atomic_json(status_path, state)
    def receipt(row):
        atomic_json(Path(path).parent / (row['unit'] + '_receipt.json'), row)
    try:
        save()
        for base in jobs:
            retries = manifest.get('max_invalid_retries', 0)
            if type(retries) is not int or not 0 <= retries <= 3:
                raise ValueError('max_invalid_retries must be 0..3 (same seed only)')
            for attempt in range(retries + 1):
                job = dict(base, attempt=attempt)
                if attempt:
                    job.update(unit=base['unit'] + f'-initretry{attempt}', run_dir=base['run_dir'] + f'_initretry{attempt}')
                if before_job:
                    before_job()
                for f, h in manifest['source_sha256'].items():
                    if sha256(f) != h:
                        raise ValueError('source changed; queue stopped: ' + f)
                if Path(job['run_dir']).exists():
                    raise ValueError('refuse existing run ' + job['run_dir'])
                u = unit_state(job['unit'])
                if u.get('LoadState') not in (None, 'not-found') or u.get('ActiveState') in ('active', 'activating', 'deactivating'):
                    raise ValueError('unit already exists; choose a new queue directory')
                current = dict(job, started_at=time.time(), status='launching', native_successes=None)
                state['jobs'].append(current)
                state['current_unit'] = job['unit']
                save()
                subprocess.run(command(job), check=True, timeout=30)
                started = time.monotonic()
                while True:
                    u = unit_state(job['unit'])
                    current['unit_state'] = u
                    live = Path(job['run_dir'])/'live_state.json'
                    if live.exists():
                        current['last_call'] = read_result(live)
                    save()
                    if (u.get('ActiveState') not in ('active', 'activating', 'deactivating')
                            or (u.get('ActiveState') == 'active' and u.get('SubState') == 'exited'
                                and u.get('MainPID') == '0')):
                        break
                    if time.monotonic() - started > evaluation_wall_budgets(job)['external_wall_seconds'] + 60:
                        raise RuntimeError('child unit exceeded external deadline')
                    time.sleep(10)
                result_path = Path(job['run_dir'])/'result.json'
                for f, h in manifest['source_sha256'].items():
                    if sha256(f) != h:
                        raise ValueError('source changed during episode; not scored: ' + f)
                result = read_result(result_path)
                status, successes = classify_result(job, result, u, manifest.get('require_video', False))
                current.update(status=status, native_successes=successes, finished_at=time.time(),
                               error=result.get('error'), result_path=str(result_path))
                receipt(current)
                # Retain successful transient units until Result/exit status is archived.
                if u.get('ActiveState') == 'active' and u.get('SubState') == 'exited':
                    subprocess.run(['systemctl', '--user', 'stop', job['unit']], timeout=45, check=True)
                state['current_unit'] = None
                save()
                log_progress(f"结束 {job['unit']}：{status}；native_successes={successes}；证据 {result_path}。未完成不计policy失败。")
                failures = failures + 1 if status == 'infrastructure_incomplete' else 0
                if failures >= 2:
                    raise RuntimeError('two consecutive infrastructure failures; diagnose before further GPU spending')
                if status != 'invalid_initialization':
                    break
        state['status'] = 'completed' if all(not c['unresolved_seeds'] for c in counts(jobs, state['jobs']).values()) else 'incomplete'
    except BaseException as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        if state.get('current_unit'):
            try:
                subprocess.run(['systemctl', '--user', 'stop', state['current_unit']], timeout=45, check=False)
            except BaseException as cleanup:
                state['cleanup_error'] = str(cleanup)
            if current is not None:
                current.update(status='interrupted' if isinstance(exc, (StopQueue, KeyboardInterrupt)) else 'infrastructure_incomplete',
                               native_successes=None, error=state['error'], finished_at=time.time())
                receipt(current)
    finally:
        state['current_unit'] = None
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0 if state['status'] == 'completed' else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    path = Path(args.manifest).resolve()
    manifest = json.loads(path.read_text())
    validate_manifest(manifest)
    if not args.execute:
        print(json.dumps({'jobs': len(manifest['jobs']), 'commands': [command(j) for j in manifest['jobs']]}, indent=2))
        return 0
    return run_manifest(manifest, path)


if __name__ == '__main__':
    raise SystemExit(main())
