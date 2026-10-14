#!/usr/bin/env python3
"""CPU queue and fixed-five V3 evaluation after verified final training.

No training signals, restarts, resume, physical retries or result selection.
Only --execute creates a queue. It never acquires a GPU while the trainer lives.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import precision_v3_evidence_v42 as e
from scripts import gpu_lock
from scripts.progress1_log_v41 import log

BASE = ROOT / 'outputs/zhuhai2026_tasks12'
TRAIN = BASE / 'wam_precision_v2_task2_v3fork80k_30k_20261012/train'
TRAIN_UNIT = 'p2-zhuhai-v3fork80k-train-20261012.service'
OWNER = 'zhuhai-v3-final5-20261012-v42'
UNIT = 'p2-zhuhai-v3-final5-aftertrain-20261012-v42.service'
SCHEMA = 'precision-v3-fork80k-final30k-fixed5-v42'
RUNTIME = ROOT / 'scripts/eval_zhuhai_wam_precision_v3_v42.py'
PREREGISTRATION = TRAIN.parent / 'evaluation_preregistration_v42.json'
OUT = BASE / 'wam_precision_v3_task2_final30k_fixed5_20261012_v42'
CPU_PYTHON = ROOT / '.venvs/wam-fixed/bin/python'
TRAINER_NAMES = {'train_zhuhai_wam_precision_v3.py', 'train_zhuhai_wam_precision_v2.py',
                 'train_zhuhai_wam_precision.py', 'train_zhuhai_wam_hjepa.py', 'train_zhuhai_wam.py'}
VERIFICATION = BASE/'verification/dagger80k_extra20_20261011_v24'


def verified_cpu_receipts():
    names = ['precision_v3_loading_cpu_v42.json', 'precision_v3_final_training_auditor_cpu_tests_v42.json',
        'v3_eval_analyzer_cpu_validation_v42.json', 'v3_runtime_evidence_independent_cpu_tests_v42_r4.json',
        'precision_v3_final5_queue_cpu_tests_v42.json']
    reports = {}
    for name in names:
        path = VERIFICATION/name
        report = e.read_json(path)
        e.require(str(report.get('status', '')).startswith('passed'), 'CPU preflight not passed: '+name)
        e.require(report.get('failure_count', 0) == 0 and report.get('error_count', 0) == 0
            and not report.get('failures', []), 'CPU preflight failures: '+name)
        bindings = dict(report.get('file_sha256', report.get('bindings', {})))
        if name == names[0]:
            bindings.update({str(ROOT/'harness/zhuhai_precision_v3_evaluation_v42.py'):report['loader_source_sha256'],
                str(ROOT/'scripts/test_precision_v3_loading_v42.py'):report['test_source_sha256']})
            e.require(report.get('cuda_initialized') is False and report['actual_checkpoint']['all_tensors_bit_exact'] is True,
                'Actual CPU checkpoint loading evidence missing')
        if name == names[1]:
            bindings.update({str(ROOT/'scripts/audit_precision_v3_training_final_v42.py'):report['auditor_source_sha256'],
                str(ROOT/'scripts/test_audit_precision_v3_training_final_v42.py'):report['test_source_sha256']})
        e.require(bindings, 'CPU report source bindings missing: '+name)
        for filename, sha in bindings.items():
            p = Path(filename) if Path(filename).is_absolute() else ROOT/filename
            e.require(e.sha256_file(p) == sha, 'CPU preflight/source drift: '+str(p))
        reports[str(path)] = dict(sha256=e.sha256_file(path), status=report['status'])
    return reports


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def unit_state(unit):
    try:
        result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState',
            '-p', 'SubState', '-p', 'MainPID', '-p', 'Result'], text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'query_error': f'{type(exc).__name__}: {exc}'}
    if result.returncode:
        return {'query_error': result.stderr.strip()}
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def observe_json(path):
    try:
        return e.read_json(path)
    except (OSError, ValueError) as exc:
        return {'observation_error': f'{type(exc).__name__}: {exc}'}


def observe_owner():
    try:
        return gpu_lock.owner()
    except (OSError, ValueError) as exc:
        return {'observation_error': f'{type(exc).__name__}: {exc}'}


def actual_trainers():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(Path(arg).name in TRAINER_NAMES or arg in {'scripts.'+Path(n).stem for n in TRAINER_NAMES}
               for arg in argv if arg):
            found.append(int(proc.name))
    return found


def finished_and_released(status, unit, trainers, owner):
    return (status.get('status') == 'completed' and status.get('step') == 30000
        and status.get('target_steps') == 30000 and unit.get('ActiveState') == 'inactive'
        and unit.get('SubState') == 'dead' and unit.get('MainPID') == '0'
        and unit.get('Result') == 'success' and not trainers and owner is None)


def make_plan():
    prereg = e.read_json(PREREGISTRATION)
    e.require(prereg['seeds'] == list(e.SEEDS) and prereg['planned_episodes'] == 5
        and prereg['recorded_seeds'] == [e.SEEDS[0]] and prereg['checkpoint_step'] == 30000
        and prereg['one_attempt_per_seed'] is True and prereg['retry_valid_failures'] is False
        and prereg['randomized'] is False and prereg['max_steps'] == 2400,
        'Preregistered protocol or seeds changed')
    manifest = e.read_json(TRAIN/'manifest.json')
    contract = manifest['contract']
    e.require(contract['format'] == e.FORMAT and e.digest(contract) == manifest['contract_sha256'],
        'V3 training contract differs')
    # Reuse the successfully verified pinned driver/Isaac environment, not global mismatched libraries.
    prior = e.read_json(BASE/'wam_precision_v2_task2_80k_extra20_20261011_v24/plan.json')
    environment = deepcopy(prior['child_env'])
    environment.update(prior['runtime_environment'])
    environment.update(CUDA_VISIBLE_DEVICES='0', P2_V3_THIRD_VIEW_SEED=str(e.SEEDS[0]))
    e.require('nvidia_5959107' in environment.get('LD_LIBRARY_PATH', ''), 'Pinned driver missing')
    checkpoint = str(TRAIN/'checkpoint_step_030000.pt')
    commands = [dict(seed=seed, out=str(OUT/e.TASK/f'seed_{seed}'), command=[
        str(e.PYTHON), '-B', str(RUNTIME), '--checkpoint', checkpoint,
        '--out', str(OUT/e.TASK/f'seed_{seed}'), '--task', e.TASK,
        '--gpu-lock-owner', OWNER, '--episode-seed', str(seed), '--max-steps', '2400',
        '--wall-seconds', '3600', '--startup-timeout', '180', '--policy-timeout', '60']) for seed in e.SEEDS]
    paths = ['scripts/eval_precision_v3_final5_v42.py', 'scripts/eval_zhuhai_wam_precision_v3_v42.py',
        'scripts/precision_v3_evidence_v42.py', 'harness/zhuhai_precision_v3_evaluation_v42.py',
        'harness/zhuhai_observer_video_v42.py', 'scripts/audit_precision_v3_training_final_v42.py',
        'scripts/analyze_precision_v3_eval_v42.py', 'scripts/test_precision_v3_loading_v42.py',
        'scripts/test_precision_v3_runtime_evidence_v42.py', 'scripts/test_audit_precision_v3_training_final_v42.py',
        'scripts/test_precision_v3_final5_queue_v42.py',
        'harness/zhuhai_official_braking.py', 'scripts/progress1_log_v41.py']
    extensions = {str(ROOT/p):e.sha256_file(ROOT/p) for p in paths}
    preserved = {}
    for directory in ('wam_precision_v2_task2_dagger_80k_fixed5_20261011_v23',
                      'wam_precision_v2_task2_80k_extra20_20261011_v24',
                      'wam_precision_v2_task2_official_braking30k_20261010_v1'):
        path = BASE/directory/'summary.json'
        preserved[str(path)] = e.sha256_file(path)
    return dict(schema=SCHEMA, time=now(), out=str(OUT), training=str(TRAIN),
        training_unit=TRAIN_UNIT, systemd_unit=UNIT, gpu_lock_owner=OWNER,
        checkpoint_expected_path=checkpoint, checkpoint_step=30000, seeds=list(e.SEEDS),
        planned_episodes=5, recording=dict(recorded_seeds=[e.SEEDS[0]], other_seeds=False,
        policy_inputs_unchanged=list(e.FIELDS[:3]), observer_policy_input=False),
        randomized=False, action_postprocessing=e.braking.protocol(),
        action_postprocessing_sha256=e.digest(e.braking.protocol()), commands=commands,
        child_env=environment, frozen_sources=contract['source_files'], extension_sources=extensions,
        preregistration=dict(path=str(PREREGISTRATION), sha256=e.sha256_file(PREREGISTRATION)),
        train_manifest_sha256=e.sha256_file(TRAIN/'manifest.json'), preserved_original_results=preserved,
        cpu_preflight_receipts=verified_cpu_receipts(),
        interpretation='Five predeclared new source scenes; final V3 candidate, original strict rules and official braking. '
        'No matched same-seed V2 baseline or component-causality claim; first observer only displays, never controls.')


def verify(plan):
    for path, sha in {**plan['frozen_sources'], **plan['extension_sources']}.items():
        e.require(e.sha256_file(path) == sha, 'Frozen source drift: '+path)
    e.require(e.sha256_file(PREREGISTRATION) == plan['preregistration']['sha256'], 'Preregistration drift')
    e.require(e.sha256_file(TRAIN/'manifest.json') == plan['train_manifest_sha256'], 'Training manifest drift')
    for path, sha in plan['preserved_original_results'].items():
        e.require(e.sha256_file(path) == sha, 'Original result changed: '+path)
    for path, receipt in plan['cpu_preflight_receipts'].items():
        e.require(e.sha256_file(path) == receipt['sha256'], 'Frozen CPU validation receipt drift: '+path)


def checkpoint_metadata(plan):
    path = Path(plan['checkpoint_expected_path'])
    receipt = e.read_json(path.with_name(path.name+'.sha256.json'))
    e.require(receipt['step'] == 30000 and receipt['model_only'] is True
        and receipt['size'] == path.stat().st_size and receipt['sha256'] == e.sha256_file(path),
        'Final immutable milestone differs')
    return dict(path=str(path), step=30000, sha256=receipt['sha256'], size=receipt['size'],
        manifest_sha256=e.sha256_file(TRAIN/'manifest.json'),
        receipt_sha256=e.sha256_file(path.with_name(path.name+'.sha256.json')))


def publish(plan, rows, status, error=None):
    complete = status == 'completed'
    if complete:
        e.require([r['seed'] for r in rows] == plan['seeds'], 'Exact five attempt order required')
    value = dict(schema=SCHEMA, status=status, time=now(), planned_episodes=5,
        preregistered_seeds=plan['seeds'], seeds=plan['seeds'], episodes=rows,
        full_successes=sum(r['full_task_success'] for r in rows),
        success_rate=sum(r['full_task_success'] for r in rows)/5 if complete else None,
        checkpoint=plan.get('checkpoint'), checkpoint_step=30000,
        checkpoint_sha256=plan.get('checkpoint', {}).get('sha256'), checkpoint_format=e.FORMAT,
        system_mode='precision_wam_v3_official_braking', evaluation_version=e.braking.VERSION,
        action_postprocessing_sha256=plan['action_postprocessing_sha256'], randomized=False,
        recorded_seeds=[e.SEEDS[0]], record_video=True, third_view=True,
        recording_scope='first case only; other four off', interpretation=plan['interpretation'])
    if error:
        value['error'] = error
    e.write_json(OUT/'summary.json', value)
    return value


def wait_for_training(plan):
    started = time.monotonic()
    last_summary = 0.
    while True:
        verify(plan)
        status = observe_json(TRAIN/'status.json')
        unit = unit_state(TRAIN_UNIT)
        trainers = actual_trainers()
        owner = observe_owner()
        e.write_json(OUT/'watchdog_status.json', dict(time=now(), status='waiting_for_training',
            training=status, unit=unit, actual_trainer_pids=trainers, gpu_owner=owner,
            monitor_cpu_only=True, evaluation_started=False))
        if finished_and_released(status, unit, trainers, owner):
            return
        # Query failures or nonterminal unit transitions are observations, never reasons to restart.
        if unit.get('ActiveState') in ('inactive', 'failed') and not trainers and 'observation_error' not in status:
            e.require(status.get('status') == 'completed' and unit.get('Result') == 'success',
                'Training terminal unsuccessful/incomplete; preserve and do not evaluate or restart')
        if time.monotonic()-last_summary >= 1800:
            log(f'V3五测CPU队列等待训练：{status.get("step")}/30000，unit={unit.get("ActiveState")}; '
                '未占GPU、未启动物理测试，不中断或重启训练。', 'v3_fixed5_queue')
            last_summary = time.monotonic()
        e.require(time.monotonic()-started < 43200, 'Training queue12h deadline exceeded; no restart')
        time.sleep(30)


def audit_final_training(plan):
    out = OUT/'training_final_independent_audit.json'
    e.require(not out.exists(), 'Never overwrite a final audit')
    command = [str(CPU_PYTHON), '-B', str(ROOT/'scripts/audit_precision_v3_training_final_v42.py'),
        '--train-root', str(TRAIN), '--out', str(out)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    e.write_json(OUT/'watchdog_status.json', dict(time=now(), status='independent_training_audit', command=command))
    with (OUT/'training_final_independent_audit.log').open('xb') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
    report = e.read_json(out)
    e.require(result.returncode == 0 and report.get('evaluation_ready') is True
        and report.get('status') == 'passed_final_training_and_gpu_released', 'Final training independent audit failed')
    return dict(path=str(out), sha256=e.sha256_file(out))


def run_once(item, plan):
    with (OUT/f'seed_{item["seed"]}.log').open('xb') as stream:
        child = subprocess.Popen(item['command'], cwd=ROOT, env=dict(os.environ, **plan['child_env']),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            e.write_json(OUT/'active_participant.json', dict(time=now(), pid=child.pid, pgid=child.pid,
                seed=item['seed'], command=item['command']))
            started = time.monotonic()
            bad_nvml = 0
            while child.poll() is None:
                e.require(time.monotonic()-started < 3840, 'Physical episode wall deadline')
                result_file = Path(item['out'])/'result.json'
                state = observe_json(result_file) if result_file.exists() else {}
                try:
                    age = time.time()-result_file.stat().st_mtime if result_file.exists() else time.monotonic()-started
                except OSError as exc:
                    age = time.monotonic()-started
                    state = {'observation_error': f'{type(exc).__name__}: {exc}'}
                e.require(not ('observation_error' not in state and state.get('status') in (None, 'starting') and age > 300), 'Startup exceeds300s')
                e.require(not (state.get('status') == 'running' and age > 600), 'Action heartbeat stale600s')
                try:
                    health = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                        env=dict(os.environ, **plan['child_env']), text=True, capture_output=True, timeout=10)
                    healthy = health.returncode == 0 and health.stdout.strip() == '595.91.07'
                except subprocess.TimeoutExpired:
                    healthy = False
                bad_nvml = 0 if healthy else bad_nvml+1
                e.require(bad_nvml < 3, 'Pinned driver failed3 consecutive checks')
                e.write_json(OUT/'watchdog_status.json', dict(time=now(), status='watching_evaluation',
                    seed=item['seed'], pid=child.pid, phase=state.get('status'), control_steps=state.get('control_steps'),
                    heartbeat_age_seconds=age, pinned_driver_healthy=healthy))
                time.sleep(30)
            return child.returncode
        finally:
            if child.poll() is None:
                # Only this queue's own evaluator process group; never any trainer or foreign process.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def execute(plan):
    queue_check = e.read_json(VERIFICATION/'precision_v3_final5_queue_cpu_tests_v42.json')
    e.require(queue_check.get('actual_cpu_plan_verified') is True, 'Independent production CPU plan not verified')
    OUT.mkdir(parents=True, exist_ok=False)
    rows, held = [], False
    e.write_json(OUT/'plan.json', plan)
    publish(plan, rows, 'waiting_for_training')
    with (OUT/'.writer.lock').open('a') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_for_training(plan)
            plan['training_final_audit'] = audit_final_training(plan)
            verify(plan)
            plan['checkpoint'] = checkpoint_metadata(plan)
            e.write_json(OUT/'plan.json', plan)
            # Resource/NVML/EGL checks use this process too, not only the future child worker.
            os.environ.update(plan['child_env'])
            from scripts import run_zhuhai_task1_dr200_concurrent as admission
            e.require(gpu_lock.owner() is None and not actual_trainers(), 'Training/GPU not released')
            e.require(not admission.gpu_jobs(admission.processes()), 'Foreign GPU participant exists')
            resource = admission.scheduler_module().collection_resource_admission()
            e.require(admission.scheduler_module().resource_ready(resource, expansion=True), 'GPU/RAM admission insufficient')
            gpu_lock.acquire(OWNER, max_wait=0)
            held = True
            e.require(not admission.gpu_jobs(admission.processes()), 'Foreign GPU participant appeared')
            contract = e.read_json(TRAIN/'manifest.json')['contract']
            with Path(plan['checkpoint']['path']).open('rb') as checkpoint:
                fcntl.flock(checkpoint, fcntl.LOCK_SH | fcntl.LOCK_NB)
                publish(plan, rows, 'evaluating')
                for item in plan['commands']:
                    verify(plan)
                    e.require(gpu_lock.owner() == OWNER and not actual_trainers(), 'Exclusive evaluation owner differs')
                    directory = Path(item['out'])
                    directory.parent.mkdir(exist_ok=True)
                    e.require(not directory.exists(), 'No physical retry or resume permitted')
                    e.write_json(OUT/f'attempt_seed_{item["seed"]}.json', item)
                    code = run_once(item, plan)
                    e.write_json(OUT/f'exit_seed_{item["seed"]}.json', dict(returncode=code))
                    row = e.classify(directory/'result.json', plan['checkpoint'], contract,
                        plan['extension_sources'][str(RUNTIME)], e.SEEDS, e.classify_noncontact)
                    e.require(code in (0, 2) if row['status'] in ('control_safety_abort', 'physics_safety_abort') else code == 0,
                        'Physical result and actual exit code differ')
                    rows.append(row)
                    publish(plan, rows, 'evaluating')
                    log(f'V3正式五测{len(rows)}/5 seed{item["seed"]}：{row["status"]}，'
                        f'末分{row["observed_base_score"]}，整任务成功{row["full_task_success"]}。', 'v3_fixed5_queue')
            verify(plan)
            publish(plan, rows, 'completed')
            e.write_json(OUT/'final_receipt.json', dict(status='completed', time=now(), tests=5,
                full_successes=sum(r['full_task_success'] for r in rows),
                scores=[r['observed_base_score'] for r in rows], attempts_not_retried=True,
                prior_results_preserved=True, training_final_audit=plan['training_final_audit']))
            return 0
        except BaseException as error:
            message = f'{type(error).__name__}: {error}'
            publish(plan, rows, 'halted', message)
            e.write_json(OUT/'failure_guard.json', dict(time=now(), error=message, all_attempts_preserved=True, retries=0))
            raise
        finally:
            if held:
                from scripts import run_zhuhai_task1_dr200_concurrent as admission
                jobs = admission.gpu_jobs(admission.processes())
                if jobs:
                    e.write_json(OUT/'owner_retained_guard.json', dict(owner=OWNER, jobs=jobs))
                else:
                    gpu_lock.release(OWNER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = make_plan()
    verify(plan)
    if args.execute:
        return execute(plan)
    print(json.dumps(dict(status='cpu_plan_only_no_simulation', seeds=plan['seeds'], out=plan['out'],
        waiting_for_final_training=True, recorded_seeds=[e.SEEDS[0]], extension_sources=len(plan['extension_sources']))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
