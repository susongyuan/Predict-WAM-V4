#!/usr/bin/env python3
"""CPU-only, read-only training supervision; never signals or restarts the trainer."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.progress1_log_v41 import log

TRAIN = ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_v3fork80k_30k_20261012/train'
UNIT = 'p2-zhuhai-v3fork80k-train-20261012.service'
OUT = TRAIN.parent / 'monitor'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {'read_error': f'{type(exc).__name__}: {exc}'}


def unit_state():
    result = subprocess.run(
        ['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '-p', 'SubState',
         '-p', 'ExecMainStatus', '-p', 'MainPID'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        return {'query_error': result.stderr.strip()}
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def trainer_io():
    rows = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any(Path(arg).name == 'train_zhuhai_wam_precision_v3.py' for arg in argv):
                continue
            if str(TRAIN) not in argv:
                continue
            data = dict(line.split(': ', 1) for line in (entry / 'io').read_text().splitlines())
            rows[entry.name] = {key: int(data[key]) for key in ('read_bytes', 'write_bytes', 'rchar', 'wchar')}
        except (OSError, ValueError):
            continue
    return rows


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def snapshot():
    status = read_json(TRAIN / 'status.json')
    unit = unit_state()
    io = trainer_io()
    stage = status.get('status', 'unknown')
    if stage == 'building' and (TRAIN / 'checkpoint_step_000000.pt.sha256.json').exists():
        stage = 'initial_diagnostic_before_optimizer'
    checkpoints = [path.name for path in sorted(TRAIN.glob('checkpoint_step_*.pt.sha256.json'))]
    row = dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
               training=status, effective_stage=stage, unit=unit, trainer_io=io,
               checkpoint_receipts=checkpoints, poll_seconds=30, summary_seconds=1800,
               video_recording=False, monitor_uses_gpu=False)
    elapsed = status.get('elapsed_seconds')
    step = status.get('step', 0)
    if isinstance(elapsed, (int, float)) and step > 0:
        row['seconds_per_update_including_cache_io'] = elapsed / step
        row['estimated_remaining_seconds'] = (status.get('target_steps', 30000) - step) * elapsed / step
    return row


def describe(row, reasons):
    status = row['training']
    return (f"V3 fork80k监控（{','.join(reasons)}）：正式更新{status.get('step', '?')}/30000，"
            f"阶段{row['effective_stage']}，unit={row['unit'].get('ActiveState', 'unknown')}/"
            f"{row['unit'].get('SubState', 'unknown')}；loss={status.get('last_loss')}；"
            f"告警={row.get('warning') or status.get('error') or row['unit'].get('query_error') or '无'}。"
            '仅CPU观察，不中断/重启训练；未评测，不宣称成功率。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', XDG_RUNTIME_DIR='/run/user/1000',
                      DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read_json(OUT / 'latest.json')
        seen = set(previous.get('reported_milestones', []))
        last_summary = 0.0
        last_signature = None
        last_activity = time.monotonic()
        last_warning = None
        while True:
            now = time.monotonic()
            row = snapshot()
            signature = json.dumps([row['training'], row['trainer_io']], sort_keys=True)
            if signature != last_signature:
                last_activity = now
                last_signature = signature
            row['seconds_without_status_or_io_change'] = round(now - last_activity)
            if now - last_activity >= 600:
                row['warning'] = 'no_status_or_io_change_for_10_minutes'
            exited = row['unit'].get('ActiveState') in ('inactive', 'failed') and not row['trainer_io']
            terminal_status = row['training'].get('status') in ('completed', 'failed', 'stopped')
            if exited and not terminal_status:
                row['warning'] = 'trainer_exited_without_terminal_status'
            reasons = []
            step = row['training'].get('step', 0)
            for milestone in range(5000, 30001, 5000):
                if step >= milestone and milestone not in seen:
                    reasons.append(f'milestone_{milestone}')
                    seen.add(milestone)
            if now - last_summary >= 1800:
                reasons.append('half_hour')
                last_summary = now
            if row.get('warning') != last_warning:
                reasons.append('warning_changed')
                last_warning = row.get('warning')
            if exited:
                reasons.append('terminal')
            row['reported_milestones'] = sorted(seen)
            row['monitor_status'] = 'finished' if exited else 'watching'
            write(OUT / 'latest.json', row)
            append(OUT / 'checks.jsonl', row)
            if reasons:
                append(OUT / 'events.jsonl', dict(row, reasons=reasons))
                message = describe(row, reasons)
                print(message, flush=True)
                log(message, 'v3_cpu_monitor')
            if args.once or exited:
                return 0 if not exited or (row['training'].get('status') == 'completed' and
                                          row['unit'].get('ExecMainStatus') == '0' and step == 30000) else 1
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
