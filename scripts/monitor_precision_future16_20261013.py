#!/usr/bin/env python3
"""CPU-only Future16 train/eval telemetry with configurable hourly checks."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.progress1_log_v41 import log
BASE = ROOT / 'outputs/zhuhai2026_tasks12'
TRAIN = BASE / 'wam_precision_v4_task2_future16_50k_20261013/train'
EVAL = BASE / 'wam_precision_v4_task2_future16_fixed5_20261013'
OUT = BASE / 'verification/precision_v4_20261013_future16/monitor'
UNITS = ('p2-v4-future16-train-resume-20261013.service',
         'p2-v4-future16-eval-20261013.service')
POLL_SECONDS = 3600
SUMMARY_SECONDS = 3600

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def snapshot():
    gpu = {}
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    try:
        util, used, total, power, temperature = map(float, result.stdout.strip().split(','))
        gpu = dict(utilization_percent=util, memory_used_mib=used, memory_total_mib=total,
            power_watts=power, temperature_celsius=temperature)
    except ValueError:
        gpu = dict(query_error=result.stderr.strip())
    units = {}
    for unit in UNITS:
        result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState',
            '-p', 'SubState', '-p', 'MainPID', '-p', 'ExecMainStatus'],
            capture_output=True, text=True, timeout=10)
        units[unit] = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if result.returncode:
            units[unit]['error'] = result.stderr.strip()
    processes = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any(Path(arg).name in ('train_zhuhai_wam_precision_v4_future16.py',
                    'eval_precision_v4_future16_final5.py', 'eval_zhuhai_wam_precision_v4_future16.py')
                    for arg in argv):
                continue
            state = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines()
                         if ':' in line)
            io = dict(line.split(':', 1) for line in (proc / 'io').read_text().splitlines())
            processes[proc.name] = dict(command=argv, memory_kib={k: int(state.get(k, '0').split()[0])
                for k in ('VmRSS', 'VmSwap', 'VmSize', 'VmHWM')}, io={k: int(v) for k, v in io.items()})
        except (OSError, ValueError):
            continue
    mem = dict((key, int(value.split()[0])) for key, value in
        (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()))
    training = read(TRAIN / 'status.json')
    evaluation = read(EVAL / 'summary.json')
    recent = []
    try:
        with (TRAIN / 'log.jsonl').open('rb') as stream:
            stream.seek(max(0, (TRAIN / 'log.jsonl').stat().st_size - 250000))
            for line in stream.read().splitlines():
                try:
                    recent.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    speed = sum(r['seconds_per_step'] for r in recent) / len(recent) if recent else None
    warnings = []
    if mem['MemAvailable'] < 5 * 1024**2:
        warnings.append('RAM available below 5GiB')
    if mem['SwapTotal'] and 1 - mem['SwapFree'] / mem['SwapTotal'] > .85:
        warnings.append('swap usage above 85%')
    for path in (TRAIN / 'failure_guard.json', EVAL / 'failure_guard.json'):
        if path.exists():
            warnings.append(str(path))
    for unit, state in units.items():
        if state.get('ActiveState') == 'failed' or state.get('error'):
            warnings.append(unit + ': ' + str(state))
    row = dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        training=training, evaluation=evaluation, units=units, processes=processes,
        memory_kib={k: mem[k] for k in ('MemAvailable', 'MemTotal', 'SwapFree', 'SwapTotal')},
        warnings=warnings, poll_seconds=POLL_SECONDS, summary_seconds=SUMMARY_SECONDS, monitor_uses_gpu=False, gpu_nvml=gpu,
        record_evaluation_video=False, seconds_per_update_recent=speed,
        estimated_training_remaining_seconds=(training.get('target_updates', 50000) -
            training.get('updates', 0)) * speed if speed else None)
    return row

def main():
    global TRAIN, EVAL, UNITS, POLL_SECONDS, SUMMARY_SECONDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--training', type=Path, default=TRAIN)
    parser.add_argument('--evaluation', type=Path, default=EVAL)
    parser.add_argument('--train-unit', default=UNITS[0])
    parser.add_argument('--eval-unit', default=UNITS[1])
    parser.add_argument('--poll-seconds', type=int, default=POLL_SECONDS)
    parser.add_argument('--summary-seconds', type=int, default=SUMMARY_SECONDS)
    args = parser.parse_args()
    if args.summary_seconds <= 0:
        parser.error('--summary-seconds must be positive')
    if args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive')
    POLL_SECONDS = args.poll_seconds
    SUMMARY_SECONDS = args.summary_seconds
    TRAIN, EVAL, UNITS = args.training.resolve(), args.evaluation.resolve(), (args.train_unit, args.eval_unit)
    runtime = f'/run/user/{os.getuid()}'
    os.environ.update(CUDA_VISIBLE_DEVICES='', XDG_RUNTIME_DIR=runtime,
                      DBUS_SESSION_BUS_ADDRESS='unix:path=' + runtime + '/bus')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_summary, previous_warning = 0, None
        while True:
            row = snapshot()
            temporary = OUT / 'latest.json.tmp'
            temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(OUT / 'latest.json')
            with (OUT / 'checks.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            now = time.monotonic()
            warning = json.dumps(row['warnings'])
            completed = row['evaluation'].get('status') == 'completed'
            if now - last_summary >= SUMMARY_SECONDS or warning != previous_warning or completed:
                message = (f"Future16监控：新增{row['training'].get('updates')}/50000，"
                    f"训练{row['training'].get('status')}，测试{row['evaluation'].get('status', '等待')}，"
                    f"完成回合{len(row['evaluation'].get('episodes', []))}/10；"
                    f"可用RAM {row['memory_kib']['MemAvailable']/1024**2:.2f}GiB；告警{row['warnings']}。")
                print(message, flush=True)
                log(message, 'future16_monitor')
                last_summary, previous_warning = now, warning
            if args.once or completed:
                return 0
            time.sleep(POLL_SECONDS)

if __name__ == '__main__':
    main()
