#!/usr/bin/env python3
"""Read-only task resource receipt. No daemon, process kill or cache mutation.

GPU inventory uses nvidia-smi only, without importing torch/Isaac or creating
a CUDA context. Advice concerns pausing this task; other applications and
global OS/driver caches are never modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OWNED_SCRIPTS = {'collect_isaac_zhuhai.py', 'train_zhuhai_wam.py',
                 'eval_zhuhai_wam.py', 'run_zhuhai_pipeline.py',
                 'inspect_isaac_zhuhai.py', 'prepare_isaac_aloha.py'}


def memory_info():
    result = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, rest = line.split(':', 1)
        value = rest.split()
        if value and value[0].isdigit():
            result[name] = int(value[0]) * (1024 if len(value) > 1 and value[1] == 'kB' else 1)
    return dict(total_bytes=result['MemTotal'], available_bytes=result['MemAvailable'],
                used_percent=100 * (1 - result['MemAvailable'] / result['MemTotal']),
                swap_total_bytes=result.get('SwapTotal', 0),
                swap_used_bytes=result.get('SwapTotal', 0) - result.get('SwapFree', 0))


def owned_processes():
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [part for part in (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0') if part]
            scripts = {Path(arg).name for arg in args} & OWNED_SCRIPTS
            cwd = (entry / 'cwd').resolve()
            if not scripts or not (str(ROOT) in ' '.join(args) or cwd == ROOT or ROOT in cwd.parents):
                continue
            status = {}
            for line in (entry / 'status').read_text().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    status[key] = value.strip()
            rss = int(status.get('VmRSS', '0 kB').split()[0]) * 1024
            swap = int(status.get('VmSwap', '0 kB').split()[0]) * 1024
            rows.append(dict(pid=int(entry.name), script=sorted(scripts)[0],
                             state=status.get('State'), rss_bytes=rss, swap_bytes=swap))
        except (OSError, ValueError, PermissionError):
            continue
    return sorted(rows, key=lambda row: row['pid'])


def gpu_info():
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return dict(status='unavailable', reason='nvidia-smi not on PATH')
    try:
        result = subprocess.run([executable,
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=10)
        if result.returncode:
            return dict(status='unavailable', reason=result.stderr.strip()[:300])
        rows = []
        for fields in csv.reader(result.stdout.splitlines()):
            fields = [value.strip() for value in fields]
            rows.append(dict(index=int(fields[0]), name=fields[1],
                used_bytes=int(float(fields[2])) * 1024**2,
                total_bytes=int(float(fields[3])) * 1024**2,
                utilization_percent=float(fields[4])))
        return dict(status='read_only_inventory', devices=rows, cuda_context_created=False)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return dict(status='unavailable', reason=f'{type(exc).__name__}: {exc}')


def snapshot(*, minimum_ram_bytes=4 * 1024**3, minimum_disk_bytes=100 * 1024**3):
    ram = memory_info()
    processes = owned_processes()
    gpu = gpu_info()
    disk = shutil.disk_usage(ROOT)
    reasons = []
    if ram['available_bytes'] < minimum_ram_bytes or ram['used_percent'] >= 85:
        reasons.append('RAM reserve/resource boundary reached; pause this task at a safe checkpoint')
    if ram['swap_total_bytes'] and ram['swap_used_bytes'] / ram['swap_total_bytes'] >= .85:
        reasons.append('swap occupancy >=85%; pause this task and inspect its owned RSS')
    if disk.free < minimum_disk_bytes:
        reasons.append('100GiB disk reserve reached; pause this task and audit only reconstructible task caches')
    for row in gpu.get('devices', []):
        if row['total_bytes'] - row['used_bytes'] < 512 * 1024**2:
            reasons.append(f'GPU {row["index"]} has <512MiB free; pause new task allocations')
    lock = Path('/tmp/project2_gpu.lock')
    return dict(schema='zhuhai-readonly-resource-v1',
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        task_owned_processes=processes, task_owned_rss_bytes=sum(row['rss_bytes'] for row in processes),
        ram=ram, gpu=gpu, gpu_lock_owner=lock.read_text().strip() if lock.exists() else None,
        disk=dict(path=str(ROOT), total_bytes=disk.total, used_bytes=disk.used, free_bytes=disk.free),
        thresholds=dict(minimum_available_ram_bytes=minimum_ram_bytes, minimum_free_disk_bytes=minimum_disk_bytes),
        advice='pause_this_task' if reasons else 'resources_within_declared_reserves', reasons=reasons,
        actions_taken=[], user_processes_terminated=False, global_caches_cleared=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path,
        default=ROOT / 'outputs/validation/zhuhai_architecture_audit_20261008/resource_snapshot.json')
    ap.add_argument('--history', type=Path, help='Optional existing-run JSONL audit trail')
    ap.add_argument('--minimum-ram-bytes', type=int, default=4 * 1024**3)
    ap.add_argument('--check', action='store_true', help='Return 2 with pause advice; never stop a process')
    args = ap.parse_args()
    if args.minimum_ram_bytes < 1:
        ap.error('minimum RAM must be positive')
    sys.path.insert(0, str(ROOT))
    from scripts.progress1_log import log
    log('只读资源快照：采集本任务RSS、可用RAM/swap、GPU和磁盘；仅给暂停建议', actor='zhuhai_resource_watch')
    report = snapshot(minimum_ram_bytes=args.minimum_ram_bytes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + '.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(args.out)
    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        with args.history.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(report, ensure_ascii=False) + '\n')
    print(json.dumps(dict(advice=report['advice'], reasons=report['reasons'],
        task_owned_rss_bytes=report['task_owned_rss_bytes'], available_ram_bytes=report['ram']['available_bytes'],
        free_disk_bytes=report['disk']['free_bytes'], gpu=report['gpu']), ensure_ascii=False))
    return 2 if args.check and report['reasons'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
