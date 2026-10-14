"""One CPU-only audit per invocation; no training changes or polling loop."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

VERIFY = Path(__file__).resolve().parent
ROOT = VERIFY.parents[3]

def read(path):
    return json.loads(path.read_text())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def checkpoint(path):
    receipt_path = path.with_name(path.name + '.sha256.json')
    for attempt in range(3):
        before = receipt_path.read_bytes()
        receipt = json.loads(before)
        stat_before = path.stat()
        actual_sha = sha(path)
        stat_after = path.stat()
        if before == receipt_path.read_bytes() and (
                stat_before.st_ino, stat_before.st_size, stat_before.st_mtime_ns) == (
                stat_after.st_ino, stat_after.st_size, stat_after.st_mtime_ns):
            return dict(path=str(path), receipt=receipt,
                        verified=actual_sha == receipt['sha256'] and stat_after.st_size == receipt['size'])
        if attempt < 2:
            time.sleep(3)
    return dict(path=str(path), verified=False, publication_race=True)

def main():
    cadence = read(VERIFY / 'assistant_monitor_cadence.json')
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    due = datetime.fromisoformat(cadence['next_assistant_check_due'])
    if now < due:
        raise SystemExit('Hourly audit is not due; no runtime observations performed.')
    train, evaluation, delivery = map(Path, (cadence['training_directory'],
        cadence['evaluation_directory'], cadence['delivery_directory']))
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', XDG_RUNTIME_DIR='/run/user/1000',
        DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    issues, units = [], {}
    training = read(train / 'status.json')
    summary = read(evaluation / 'summary.json') if (evaluation / 'summary.json').exists() else {}
    for unit in (cadence['current_training_unit'], cadence['current_monitor_unit'],
            cadence.get('current_evaluation_unit', 'p2-v4-future16-fast-eval-20261013.service'),
            cadence.get('current_finalizer_unit', 'p2-v4-future16-fast-finalize-20261013.service')):
        result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState',
            '-p', 'SubState', '-p', 'MainPID', '-p', 'ExecMainStatus', '-p', 'MemorySwapMax'],
            env=environment, capture_output=True, text=True, timeout=15)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        state['live_pid'] = Path('/proc/' + state.get('MainPID', '0')).exists()
        units[unit] = state
        if state.get('ActiveState') == 'failed' or state.get('ExecMainStatus', '0') != '0':
            issues.append('service failure: ' + unit)
    trainers = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(Path(arg).name in ('with_project_gpu.py', 'gpu_lock.py') for arg in argv):
                continue
            if not any(Path(arg).name == 'run_precision_future16_fastnorm.py' for arg in argv):
                continue
            if '--out' not in argv or argv[argv.index('--out') + 1] != str(train):
                continue
            if '--entry' not in argv:
                continue
            cgroup_path = Path('/sys/fs/cgroup' + next(line.split(':', 2)[2] for line in
                (proc / 'cgroup').read_text().splitlines() if line.startswith('0::')))
            memory = {name: (cgroup_path / name).read_text().strip() for name in
                ('memory.current', 'memory.swap.current', 'memory.swap.max', 'memory.events')}
            events = dict(line.split() for line in memory['memory.events'].splitlines())
            if int(events.get('oom', 0)) or int(events.get('oom_kill', 0)):
                issues.append('trainer cgroup OOM: ' + proc.name)
            trainers[proc.name] = dict(batch_size=argv[argv.index('--batch-size') + 1],
                                      command=argv, cgroup=memory)
        except (OSError, ValueError, StopIteration):
            continue
    if training.get('status') == 'training' and len(trainers) != 1:
        issues.append('expected one actual trainer, got ' + str(len(trainers)))
    monitor = read(VERIFY / 'monitor/latest.json')
    for _ in range(5):
        if datetime.fromisoformat(monitor['time']) >= due - timedelta(seconds=5):
            break
        time.sleep(2)
        monitor = read(VERIFY / 'monitor/latest.json')
    if datetime.fromisoformat(monitor['time']) < due - timedelta(seconds=5):
        issues.append('hourly monitor snapshot is stale')
    guards = [str(directory / 'failure_guard.json') for directory in (train, evaluation, delivery)
              if (directory / 'failure_guard.json').exists()]
    issues.extend('failure guard: ' + guard for guard in guards)
    journal = subprocess.run(['journalctl', '-k', '-b', '--since',
        datetime.fromisoformat(cadence['last_assistant_check_at']).strftime('%Y-%m-%d %H:%M:%S'),
        '--no-pager', '-o', 'short-iso'], env=environment, capture_output=True, text=True, timeout=20)
    kernel_matches = [line for line in journal.stdout.splitlines()
        if re.search(r'NVRM|Xid|out of memory|oom-kill|watchdog', line, re.I)]
    issues.extend('kernel: ' + line for line in kernel_matches)
    if journal.returncode:
        issues.append('kernel journal unavailable')
    bindings = read(train / 'manifest.json')['contract']['source_files']
    final_plan = read(delivery / 'plan.json')
    changed = [path for mapping in (bindings, final_plan['source_bindings'])
               for path, expected in mapping.items() if sha(Path(path)) != expected]
    issues.extend('frozen source changed: ' + path for path in changed)
    if final_plan['record_evaluation_video'] is not False or final_plan['extra_trials'] != 0:
        issues.append('evaluation recording or trial-count protocol differs')
    checkpoints = [checkpoint(train / 'checkpoint_last.pt')]
    models = list(train.glob('checkpoint_step_*.pt'))
    if models:
        checkpoints.append(checkpoint(max(models, key=lambda path:
            read(path.with_name(path.name + '.sha256.json'))['step'])))
    issues.extend('checkpoint integrity: ' + item['path'] for item in checkpoints if not item['verified'])
    previous = cadence['last_training_status']
    elapsed = (now - datetime.fromisoformat(cadence['last_observation_at'])).total_seconds()
    advanced = training.get('updates', 0) - previous.get('updates', 0)
    rate = elapsed / advanced if advanced > 0 else None
    if training.get('status') == 'training' and advanced <= 0:
        issues.append('training has not advanced between hourly checks')
    remaining_hours = ((50000 - training['updates']) * rate / 3600) if rate else None
    snapshot_time = datetime.fromisoformat(monitor['time'])
    next_due = snapshot_time + timedelta(seconds=3600)
    receipt = dict(time=now.isoformat(), status='issues_found' if issues else 'passed',
        issues=issues, snapshot_warnings=monitor['warnings'], units=units, actual_trainers=trainers,
        training=training, evaluation=summary, monitor_time=monitor['time'],
        gpu_snapshot=monitor['gpu_nvml'], memory_snapshot_kib=monitor['memory_kib'],
        metric_scope='resource snapshots at check time; not hourly means/minima',
        previous_30s_RAM_warnings_preserved=True, interval_seconds=3600,
        updates_advanced=advanced, wall_seconds_per_update_including_saves=rate,
        training_remaining_hours=remaining_hours, disk_free_GiB=shutil.disk_usage(ROOT).free / 1024**3,
        kernel_audit=dict(returncode=journal.returncode, matches=kernel_matches, stderr=journal.stderr),
        guards=guards, source_bindings_verified=dict(training=len(bindings), finalizer=len(final_plan['source_bindings'])),
        checkpoints=checkpoints, record_evaluation_video=final_plan['record_evaluation_video'],
        extra_physical_trials=final_plan['extra_trials'], parameters_changed=False, next_due=next_due.isoformat())
    name = 'hour_check_' + now.strftime('%Y%m%d_%H%M') + '.json'
    (VERIFY / name).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    cadence.update(last_assistant_check_at=now.isoformat(), last_observation_at=now.isoformat(),
        last_training_status=training, next_assistant_check_due=next_due.isoformat(),
        last_hourly_audit=name, last_goal_turn_classification='one scheduled hourly check; ' + receipt['status'])
    (VERIFY / 'assistant_monitor_cadence.json').write_text(json.dumps(cadence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(receipt=name, status=receipt['status'], training=training,
        issues=issues, warnings=monitor['warnings'], seconds_per_update=rate,
        remaining_hours=remaining_hours, next_due=next_due.isoformat()), ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
