#!/usr/bin/env python3
"""Continue final-checkpoint evaluation after one precisely identified metadata failure.

Never stop/restart training or alter the original frozen pipeline. Wait for its
CPU seed-manifest preflight to fail before any scored episode, preserve that
receipt, then run the explicitly versioned SOURCE compatibility evaluator once.
Only the original final20k, five tasks and 50 seeds are authorized. No retries.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import gpu_lock
from scripts.run_robotwin_multitask_pipeline import atomic, now, read, sha, changes, append_progress, update_doc, stop_child


class ContinuationError(RuntimeError):
    pass


def source_check(manifest):
    changed = [p for p, expected in manifest['source_sha256'].items()
               if not Path(p).is_file() or sha(p) != expected]
    changed += changes(read(manifest['original_manifest']))
    if changed:
        raise ContinuationError('frozen source changed: '+str(sorted(set(changed))))


def readiness(root):
    """Read-only, fail-closed handoff; no broad suppression of original errors."""
    if not (root/'completion.json').is_file():
        return None
    completed = read(root/'completion.json')
    status = read(root/'pipeline_status.json')
    training = read(root/'train/status.json')
    error_path = root/'eval_cpu_preflight/preflight_error.json'
    if completed.get('source_changes') != []:
        raise ContinuationError('original pipeline source integrity failed')
    if (completed.get('status') != 'failed' or completed.get('exit_code') != 1
            or status.get('stage') != 'cpu_eval_preflight'):
        raise ContinuationError('original exit is not the registered CPU seed-metadata failure')
    if (training.get('status') != 'completed_fixed_budget' or training.get('step') != 20000
            or training.get('schedule_budget') != 20000 or training.get('heldout_used') is not False):
        raise ContinuationError('full fixed-budget training is not complete')
    error = read(error_path)
    if (error.get('status') != 'infrastructure_error' or error.get('phase') != 'preflight'
            or error.get('error') != "KeyError: 'episodes'"
            or error.get('scored_episodes') != 0 or error.get('simulator_started') is not False):
        raise ContinuationError('original preflight error differs from the explicitly audited SOURCE incompatibility')
    if (root/'eval').exists():
        raise ContinuationError('original evaluation directory already exists; do not duplicate/replace any episode')
    return dict(original_completion=completed, original_preflight=error,
                original_completion_sha256=sha(root/'completion.json'),
                original_preflight_sha256=sha(error_path), training_status_sha256=sha(root/'train/status.json'))


def unit_stopped(unit):
    r = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p', 'ActiveState'],
                       capture_output=True, text=True, timeout=10, check=True)
    fields = dict(line.split('=', 1) for line in r.stdout.splitlines() if '=' in line)
    return fields.get('MainPID') == '0' and fields.get('ActiveState') in ('inactive','failed')


def report(state, callback, message):
    try:
        callback(message)
    except Exception as exc:
        state.setdefault('reporting_errors', []).append(dict(at=now(), error=repr(exc)))


def publish(root, out, state):
    state['updated_at'] = now()
    atomic(out/'status.json', state)
    atomic(root/'experiment_status.json', state)


def stage(name, command, timeout, root, out, manifest, state):
    state.update(phase=name, stage_started_at=now())
    publish(root, out, state)
    log = out/(name+'.log')
    report(state, append_progress, '五任务评测兼容接续：开始`'+name+'`，原训练/旧失败回执不修改。')
    started, child = time.monotonic(), None
    last_report = started
    try:
        if shutil.disk_usage(root).free < manifest.get('reserve_disk_bytes', 300*1024**3):
            raise ContinuationError('disk reserve below 300GiB')
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, env={**os.environ, **manifest.get('environment',{})})
            while child.poll() is None:
                source_check(manifest)
                if shutil.disk_usage(root).free < manifest.get('reserve_disk_bytes', 300*1024**3):
                    raise ContinuationError('disk reserve below 300GiB')
                if time.monotonic()-started > timeout:
                    raise TimeoutError(name+' exceeded registered wall budget')
                if (root/'eval/summary.json').is_file():
                    summary=read(root/'eval/summary.json')
                    state['evaluation']={k:v for k,v in summary.items() if k!='episodes'}
                publish(root, out, state)
                if time.monotonic()-last_report >= 1800:
                    ev=state.get('evaluation',{})
                    report(state,append_progress,f"五任务兼容评测`{name}`运行中：有效{ev.get('scored_episodes',0)}/50，原生成功{ev.get('native_successes',0)}；未完成，非最终成功率。")
                    last_report=time.monotonic()
                time.sleep(15)
            code=child.returncode
        atomic(out/(name+'_receipt.json'), dict(command=command, exit_code=code,
               finished_at=now(), elapsed_seconds=time.monotonic()-started, log_sha256=sha(log)))
        if code:
            raise ContinuationError(name+' exited '+str(code))
    finally:
        if child is not None and child.poll() is None:
            stop_child(child)


def run(manifest_path):
    manifest_path=Path(manifest_path).resolve();manifest=read(manifest_path)
    root=Path(manifest['run_root']);out=root/'evaluation_continuation'
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',phase='training_under_original_pipeline_then_compatible_eval',
               started_at=now(),verified_complete=False,manifest=str(manifest_path),
               manifest_sha256=sha(manifest_path),repair_scope='SOURCE seed inventory metadata only; no policy/protocol change',
               original_pipeline_unit=manifest['original_unit'],training_restarted=False,record_video=False)
    acquired=False;code=1
    def terminate(signum, frame):
        raise InterruptedError('continuation signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,terminate)
    try:
        source_check(manifest)
        publish(root,out,state)
        report(state,append_progress,'已注册独立CPU接续监督器：不干预正在训练的20k；仅在原pipeline因已确认SOURCE缺少episodes字段而于CPU评测预检停止、且尚无任何原生测试后，才接手同一final20k的原定50测。旧失败回执保留，新总状态见experiment_status.json。')
        start=time.monotonic()
        while True:
            source_check(manifest)
            ready=readiness(root)
            if ready is not None and unit_stopped(manifest['original_unit']):break
            if time.monotonic()-start > manifest['wait_timeout_seconds']:
                raise TimeoutError('waiting for original frozen training/preflight timed out')
            p=root/'train/status.json'
            if p.is_file():state['training']=read(p)
            publish(root,out,state)
            time.sleep(30)
        state.update(status='running',training_verified_completed=True,training_step=20000)
        atomic(out/'handoff.json',ready)
        source_check(manifest)
        checkpoint=root/'train/checkpoint_step_020000.pt'
        receipt=read(checkpoint.with_name(checkpoint.name+'.sha256.json'))
        if (receipt.get('step')!=20000 or receipt.get('model_only') is not True
                or receipt.get('size')!=checkpoint.stat().st_size or receipt.get('sha256')!=sha(checkpoint)):
            raise ContinuationError('fixed final20k checkpoint receipt failed')
        state['checkpoint_sha256']=receipt['sha256']
        gpu_lock.acquire(manifest['gpu_lock_owner'],max_wait=600)
        acquired=True
        report(state,update_doc,'**五任务20k训练已完成，评测正在兼容接续。** 原CPU预检因SOURCE清单字段格式停止，没有执行测试；保持原权重、任务、922001–922010种子及无视频协议，兼容入口只补全来源种子读取。实时总状态：`'+str((root/'experiment_status.json').relative_to(ROOT))+'`。')
        base=[manifest['python'],'-B',manifest['evaluator'],'--checkpoint',str(checkpoint),
              '--multitask-manifest',str(root/'tasks.json'),'--gpu-lock-owner',manifest['gpu_lock_owner'],
              '--camera-readback','serial','--render-denoiser','none']
        stage('compat_cpu_preflight',base+['--out',str(out/'cpu_preflight')],600,root,out,manifest,state)
        stage('eval',base+['--out',str(root/'eval'),'--execute'],manifest['eval_timeout_seconds'],root,out,manifest,state)
        source_check(manifest)
        summary=read(root/'eval/summary.json')
        if (summary.get('verified_complete') is not True or summary.get('scored_episodes')!=50
                or summary.get('record_video') is not False or summary.get('third_view_camera_created') is not False):
            raise ContinuationError('final50 evidence did not all pass')
        state.update(status='completed',phase='finished',verified_complete=True,
                     evaluation={k:v for k,v in summary.items() if k!='episodes'})
        scores='；'.join(f"{name}: {row['native_successes']}/{row['planned_episodes']}" for name,row in summary['tasks'].items())
        report(state,append_progress,'5任务最终评测完成并全部核验：'+scores+'。同一20k权重、原定50种子、无视频；源清单兼容修复未重训/未改策略/未替换测试。最终接续回执evaluation_continuation/completion.json；原pipeline的CPU预检失败回执单独保留。')
        report(state,update_doc,'**5任务20k训练与50次评测已完成并核验。** '+scores+'。结果见`'+str((root/'eval/summary.json').relative_to(ROOT))+'`；整轮完成门禁以`experiment_status.json`及`evaluation_continuation/completion.json`为准。旧pipeline因来源清单格式而发生的CPU预检失败回执未覆盖，不是模型失败。')
        code=0
    except BaseException as exc:
        state.update(status='failed',verified_complete=False,error=repr(exc),traceback=traceback.format_exc())
        report(state,append_progress,'五任务评测兼容接续停止：'+str(exc)+'；没有重训、重试或替换种子，记录见evaluation_continuation/status.json。')
        report(state,update_doc,'**五任务接续需要检查。** '+str(exc)+'。以`'+str((root/'experiment_status.json').relative_to(ROOT))+'`为准，未完成样本不计policy失败。')
    finally:
        if acquired:
            try:gpu_lock.release(manifest['gpu_lock_owner'])
            except Exception as exc:
                code=1;state.update(status='failed',verified_complete=False,lock_release_error=repr(exc))
        state['finished_at']=now()
        publish(root,out,state)
        atomic(out/'completion.json',dict(exit_code=code,verified_complete=state['verified_complete'],
            status=state['status'],error=state.get('error'),finished_at=state['finished_at'],
            checkpoint_sha256=state.get('checkpoint_sha256'),original_pipeline_receipt_preserved=True,
            training_restarted=False,reporting_errors=state.get('reporting_errors',[]),
            gpu_lock_released=gpu_lock.owner()!=manifest['gpu_lock_owner']))
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:raise SystemExit(run(a.manifest))
    m=read(a.manifest);source_check(m);print(json.dumps({'source_check':'passed','run_root':m['run_root'],'ready':readiness(Path(m['run_root'])) is not None}))
