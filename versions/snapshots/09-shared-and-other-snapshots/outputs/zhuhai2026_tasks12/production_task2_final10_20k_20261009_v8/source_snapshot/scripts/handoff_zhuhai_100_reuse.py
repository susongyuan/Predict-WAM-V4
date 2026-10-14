#!/usr/bin/env python3
"""Freeze only the old scheduler; finish its sources, then launch versioned 100 DR.

Linux pidfds bind signals to the inspected process, avoiding PID reuse. The two
existing Isaac children continue unchanged; original plans/HDF are not edited.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_zhuhai_pipeline_100_reuse as pipeline
from scripts import collect_isaac_zhuhai as original
from scripts.progress1_log_100_parallel import log
from scripts.gpu_lock import owner


def identity(pid):
    path = Path('/proc')/str(pid)
    stat = (path/'stat').read_text().split(') ', 1)[1].split()
    return dict(pid=pid, start_ticks=int(stat[19]), state=stat[0],
                command=(path/'cmdline').read_bytes().decode().split('\0')[:-1])


def live(expected):
    try:
        actual = identity(expected['pid'])
        return actual['start_ticks'] == expected['start_ticks'] and actual['state'] != 'Z'
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--coordinator-pid', type=int, required=True)
    ap.add_argument('--wrapper-pid', type=int, required=True)
    args = ap.parse_args()
    args.old_root = args.old_root.resolve(); args.out = args.out.resolve()
    plan = json.loads((args.out/'plan.json').read_text())
    old_plan = json.loads((args.old_root/'plan.json').read_text())
    pipeline.verify_plan_sources(plan); pipeline.verify_plan_sources(old_plan)
    if plan['source_import']['root'] != str(args.old_root) or plan['source_import']['original_plan_sha256'] != pipeline.sha(args.old_root/'plan.json'):
        raise ValueError('Source import provenance mismatch')
    coordinator, wrapper = identity(args.coordinator_pid), identity(args.wrapper_pid)
    command = coordinator['command']
    if ('scripts/run_zhuhai_pipeline.py' not in command or '--out' not in command
            or Path(command[command.index('--out')+1]).resolve() != args.old_root
            or '--randomized-count' not in command or command[command.index('--randomized-count')+1] != '500'
            or owner() != 'zhuhai-production-20261009'):
        raise ValueError('Old coordinator/owner identity does not match authorized handoff')
    state = json.loads((args.old_root/'status.json').read_text())
    if state.get('active_stage') != 'source_collection':
        raise ValueError('Handoff allowed only during original source collection')
    children = []
    for name in ('source_task1', 'source_task2'):
        entry = state['stages'][name]
        child = identity(entry['pid'])
        if 'source' not in child['command'] or '--out' not in child['command'] or not str(args.old_root) in child['command'][child['command'].index('--out')+1]:
            raise ValueError('Unexpected source worker identity')
        children.append(child)
    admission = pipeline.collection_resource_admission()
    intent = dict(schema='zhuhai-100dr-handoff-v1', source_root=str(args.old_root),
        new_root=str(args.out), old_coordinator=coordinator, old_wrapper=wrapper,
        original_workers=children, target=dict(source_per_task=100, randomized_per_task=100, total=400),
        new_plan_sha256=pipeline.sha(args.out/'plan.json'), third_worker_admission=admission,
        preserve_sources=True, preserve_original_plan=True, record_video=False, started_unix=time.time())
    pipeline.write_once(args.out/'handoff_intent.json', intent)
    coordinator_fd = os.pidfd_open(coordinator['pid'])
    stopped = False
    status = dict(status='starting_handoff', active_stage='source_collection', handoff_pid=os.getpid())
    def persist():
        pipeline.atomic(args.out/'handoff_status.json', status)
    def release_old():
        # SIGTERM is queued while stopped; SIGCONT runs the original handler,
        # which halts before its next scheduling decision and reaps its children.
        if live(coordinator):
            signal.pidfd_send_signal(coordinator_fd, signal.SIGTERM)
            signal.pidfd_send_signal(coordinator_fd, signal.SIGCONT)
    try:
        signal.pidfd_send_signal(coordinator_fd, signal.SIGSTOP); stopped = True
        for _ in range(100):
            if identity(coordinator['pid'])['state'] == 'T':break
            time.sleep(.02)
        else:raise RuntimeError('Old scheduler did not stop')
        status.update(status='waiting_original_sources', old_scheduler_stopped=True)
        persist(); log('用户授权每任务100DR：旧500DR调度器已暂停，仅原两源采集worker继续；源齐后自动版本化交接', actor='handoff_100')
        last_log = time.monotonic()
        while True:
            pipeline.verify_plan_sources(plan)
            complete = True; counts = {}
            for i, child in enumerate(children, 1):
                task = f'zhuhai_task{i}'
                manifest = json.loads((args.old_root/task/'source/manifest.json').read_text())
                counts[task] = manifest.get('success_count', 0)
                if manifest['status'] != 'complete':
                    complete = False
                    if not live(child):raise RuntimeError(f'{task} exited before its source quota; no automatic bypass')
                if manifest['status'].startswith(('stopped', 'paused')):
                    raise RuntimeError(f'{task} source halted: {manifest["status"]}')
            status.update(source_counts=counts, checked_unix=time.time()); persist()
            if complete:break
            if not live(coordinator) or identity(coordinator['pid'])['state'] != 'T':
                raise RuntimeError('Old scheduler no longer paused')
            if time.monotonic()-last_log >= 60:
                log('100DR交接待源完成：'+str(counts)+'；当前两源worker继续，旧500DR不得启动', actor='handoff_100')
                last_log = time.monotonic()
            time.sleep(10)
        receipts = {}
        for i in (1, 2):
            task=f'zhuhai_task{i}'; path=args.old_root/task/'source/manifest.json'
            rows=original.validate_manifest(path, task=task, mode='source')
            receipts[task]=dict(path=str(path), sha256=pipeline.sha(path), successful_episodes=len(rows))
        pipeline.write_once(args.out/'source_import_receipt.json', receipts)
        status.update(status='releasing_old_scheduler'); persist(); release_old()
        deadline=time.monotonic()+180
        while live(coordinator) or live(wrapper):
            if time.monotonic() > deadline:raise RuntimeError('Old coordinator/wrapper did not exit; new writers not launched')
            time.sleep(1)
        launch = json.loads((args.out/'launch.json').read_text())
        status.update(status='launching_100dr_pipeline', old_scheduler_exited=True); persist()
        with (args.out/'coordinator.log').open('x') as handle:
            proc=subprocess.Popen(launch['command'], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        pipeline.write_once(args.out/'activated.json', dict(pid=proc.pid, command=launch['command'], activated_unix=time.time(), source_import=receipts))
        status.update(status='activated_100dr_pipeline', pipeline_wrapper_pid=proc.pid); persist()
        log('每任务100源+100DR版本已启动；旧source保持原合同，以hash引用继承，不复制或改判', actor='handoff_100')
        code=proc.wait()
        status.update(status='pipeline_exited', exit_code=code, finished_unix=time.time()); persist()
        return code
    except BaseException as exc:
        status.update(status='halted_handoff', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        persist()
        if stopped:release_old()
        raise
    finally:
        os.close(coordinator_fd)


if __name__ == '__main__':
    # One handoff controller; no second owner or manifest writer.
    target = Path(sys.argv[sys.argv.index('--out')+1]); target.mkdir(parents=True, exist_ok=True)
    with (target/'.handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(main())
