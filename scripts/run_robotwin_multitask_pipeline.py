#!/usr/bin/env python3
"""Supervise preregistered cache/train/eval stages; persist status/progress, never pick scores.

The immutable JSON manifest contains absolute commands/timeouts and source hashes.
The caller starts this CPU supervisor as a bounded user systemd service. It owns
one external GPU lock for the entire serial pipeline; children must not acquire
it again. No automatic retry, checkpoint selection, or source mutation is allowed.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import gpu_lock


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def changes(manifest):
    return [p for p, digest in manifest['source_sha256'].items()
            if not Path(p).is_file() or sha(p) != digest]


def append_progress(message):
    with (ROOT/'progress.md').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(f'\n- {now()} — 多任务5测流水线：{message}\n')
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


def update_doc(snapshot):
    """Replace just the live block, retain old descriptions/results unchanged."""
    path = ROOT/'docs/current_status.md'
    start, end = '<!-- multitask-live:start -->', '<!-- multitask-live:end -->'
    text = path.read_text()
    block = start+'\n'+snapshot+'\n'+end
    if start in text:
        before, tail = text.split(start, 1)
        _, after = tail.split(end, 1)
        text = before+block+after
    else:
        text = block+'\n\n'+text
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def summarize_detail(out):
    detail = {}
    for name, path in [('training', out/'train/status.json'), ('evaluation', out/'eval/summary.json')]:
        if path.exists():
            try:
                value = read(path)
                if not isinstance(value, dict):
                    continue
                detail[name] = {k:v for k,v in value.items() if k not in ('episodes','verification','source_sha256','commands')}
            except (ValueError, OSError):
                pass
    return detail


def check_disk(manifest):
    reserve = manifest.get('reserve_disk_bytes', 300*1024**3)
    free = shutil.disk_usage(ROOT).free
    if free < reserve:
        raise RuntimeError(f'disk reserve violated: {free/1024**3:.1f} GiB < {reserve/1024**3:.1f} GiB')
    return free


def stop_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=45)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=15)


def report(state, callback, *args):
    """Human-readable progress is advisory; machine evidence writes remain strict."""
    try:
        callback(*args)
    except Exception as exc:
        state.setdefault('reporting_errors', []).append(dict(at=now(), operation=getattr(callback, '__name__', 'report'), error=repr(exc)))


def run_stage(stage, manifest, state, out):
    record = dict(name=stage['name'], status='running', started_at=now(), command=stage['command'], log=stage['log'])
    state['stages'].append(record)
    state.update(stage=stage['name'], updated_at=now())
    atomic(out/'pipeline_status.json', state)
    report(state, append_progress, f"开始 `{stage['name']}`；日志 `{Path(stage['log']).relative_to(ROOT)}`。")
    Path(stage['log']).parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    child = None
    last_report = started
    try:
        if changes(manifest):
            raise RuntimeError('frozen source changed: '+str(changes(manifest)))
        check_disk(manifest)
        with Path(stage['log']).open('xb') as log:
            env = os.environ.copy()
            env.update(manifest.get('environment', {}))
            child = subprocess.Popen(stage['command'], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            while child.poll() is None:
                if time.monotonic()-started > stage['timeout_seconds']:
                    raise TimeoutError('registered stage wall deadline exceeded')
                if changes(manifest):
                    raise RuntimeError('frozen source changed while stage running')
                free = check_disk(manifest)
                state.update(updated_at=now(), elapsed_seconds=time.monotonic()-state['_started'],
                             disk_free_gib=round(free/1024**3, 1), detail=summarize_detail(out))
                record['elapsed_seconds'] = time.monotonic()-started
                atomic(out/'pipeline_status.json', {k:v for k,v in state.items() if not k.startswith('_')})
                if time.monotonic()-last_report >= 1800:
                    detail = state['detail']
                    report(state, append_progress, f"`{stage['name']}`运行中，已{record['elapsed_seconds']/3600:.2f}小时；"
                                    f"状态摘要 `{json.dumps(detail, ensure_ascii=False)[:900]}`；非最终成绩。")
                    last_report = time.monotonic()
                time.sleep(15)
            record['exit_code'] = child.returncode
        if child.returncode != 0:
            raise RuntimeError(f"stage exited {child.returncode}; inspect {stage['log']}")
        if 'required_json' in stage:
            for requirement in stage['required_json']:
                value = read(requirement['path'])
                for key, expected in requirement.get('equals', {}).items():
                    if (value.get(key) is not expected if type(expected) is bool else value.get(key) != expected):
                        raise RuntimeError(f'completion gate failed {requirement["path"]}:{key}')
        record.update(status='complete', finished_at=now(), elapsed_seconds=time.monotonic()-started)
        report(state, append_progress, f"完成 `{stage['name']}`，耗时{record['elapsed_seconds']/60:.1f}分钟；证据保留。")
    except BaseException as exc:
        if child is not None:
            stop_child(child)
        record.update(status='failed', finished_at=now(), elapsed_seconds=time.monotonic()-started, error=repr(exc))
        raise
    finally:
        state['updated_at'] = now()
        atomic(out/'pipeline_status.json', {k:v for k,v in state.items() if not k.startswith('_')})


def run(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    out = Path(manifest['out'])
    for protected in ('completion.json', 'pipeline_status.json', 'train/status.json', 'eval/summary.json'):
        if (out/protected).exists():
            raise FileExistsError('refusing prior pipeline/evaluation evidence: '+str(out/protected))
    marker = out/'pipeline_claim.json'
    with marker.open('x') as stream:
        json.dump(dict(pid=os.getpid(), started_at=now(), manifest_sha256=sha(manifest_path)), stream)
    state = dict(status='running', stage='acquiring_gpu_lock', stages=[], started_at=now(), updated_at=now(),
                 _started=time.monotonic(), manifest=str(manifest_path))
    acquired = False
    def terminate(signum, frame):
        raise InterruptedError(f'pipeline received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminate)
    code = 1
    try:
        if changes(manifest):
            raise RuntimeError('source changed before pipeline launch')
        atomic(out/'pipeline_status.json', {k:v for k,v in state.items() if not k.startswith('_')})
        report(state, append_progress, '长流水线已登记，等待既有冻结DINO缓存释放GPU锁；不会同时训练/仿真。')
        gpu_lock.acquire(manifest['gpu_lock_owner'], max_wait=manifest.get('gpu_lock_wait_seconds', 10800))
        acquired = True
        report(state, append_progress, '已取得GPU互斥锁；正式执行已锁定的缓存→训练→最终权重评测，默认无视频。')
        report(state, update_doc, f"**5任务流水线已启动（{now()}）**：缓存→WAM V3＋VTT联合训练20k→每任务10次无视频评测；"
                   f"尚无最终多任务成绩。实时证据：`{out.relative_to(ROOT)}/pipeline_status.json`。")
        for stage in manifest['stages']:
            run_stage(stage, manifest, state, out)
        changed = changes(manifest)
        summary = read(out/'eval/summary.json')
        if changed or not isinstance(summary, dict) or summary.get('verified_complete') is not True:
            raise RuntimeError('final evidence/source verification failed')
        code = 0
        state.update(status='completed', stage='finished', detail=summarize_detail(out))
        scores = '；'.join(f"{task}: {row.get('native_successes')}/{row.get('planned_episodes')}" for task, row in summary.get('tasks', {}).items())
        report(state, append_progress, '全部预定阶段完成。'+scores+'。逐回合证据见run的`eval/summary.json`；'
                        '仅代表固定5任务训练后能力，不证明泛化或VTT因果增益。')
        report(state, update_doc, f"**5任务训练与评测已完成（{now()}）**。{scores}。最终逐任务成功率与核验见"
                   f"`{out.relative_to(ROOT)}/eval/summary.json`；单份最终权重、每任务10测、无视频。")
    except BaseException as exc:
        code = 1
        state.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        report(state, append_progress, f"在`{state['stage']}`停止：{exc}。没有自动换任务/换种子/选权重；保留日志和所有已有结果。")
        report(state, update_doc, f"**5任务流水线需要修复（{now()}）**：阶段`{state['stage']}`，`{str(exc)[:500]}`。"
                   f"已停止，不将未完成样本算policy失败；证据见`{out.relative_to(ROOT)}/pipeline_status.json`。")
    finally:
        if acquired:
            try:
                gpu_lock.release(manifest['gpu_lock_owner'])
            except BaseException as exc:
                code = 1
                state.update(status='failed', lock_release_error=repr(exc))
        state.update(finished_at=now(), updated_at=now(), elapsed_seconds=time.monotonic()-state['_started'])
        atomic(out/'pipeline_status.json', {k:v for k,v in state.items() if not k.startswith('_')})
        atomic(out/'completion.json', dict(exit_code=code, status=state['status'], finished_at=now(),
            verified_complete=code==0, error=state.get('error'), reporting_errors=state.get('reporting_errors', []), source_changes=changes(manifest),
            gpu_lock_released=gpu_lock.owner()!=manifest['gpu_lock_owner']))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(run(args.manifest))
    manifest = read(args.manifest)
    print(json.dumps(dict(stages=[s['name'] for s in manifest['stages']], source_changes=changes(manifest)), indent=2))
