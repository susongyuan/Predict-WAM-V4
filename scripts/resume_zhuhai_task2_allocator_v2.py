#!/usr/bin/env python3
"""Drain owned finite-lived collectors; resume frozen task2 queue with arena cap.

No source, failure policy, manifest contract, dataset quota, or task algorithm
changes. Only glibc allocation settings for new child processes are changed.
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_zhuhai_task2_final as p
from scripts.handoff_zhuhai_100_reuse import identity,live
from scripts.gpu_lock import owner
from scripts.collect_isaac_zhuhai_recycle import validate_recycle_boundary,RESOURCE_POLICY
from scripts.progress1_log_task2_first import log
ALLOCATOR_ENV={'MALLOC_ARENA_MAX':'2','MALLOC_TRIM_THRESHOLD_':'131072'}

def resume_command(command):
    result=list(command)
    position=result.index('CUDA_VISIBLE_DEVICES')+1
    for k,v in ALLOCATOR_ENV.items():
        if any(x.startswith(k+'=') for x in result):raise ValueError('Allocator setting already present')
        result.insert(position,k+'='+v)
    if '--resume' not in result:result.append('--resume')
    return result

def main(args):
    out=args.out.resolve();attempt=args.attempt.resolve()
    plan=json.loads((out/'plan.json').read_text());p.verify_plan_sources(plan)
    coordinator=identity(args.coordinator_pid);wrapper=identity(args.wrapper_pid)
    if str(ROOT/'scripts/run_zhuhai_task2_final.py') not in coordinator['command'] or Path(p._command_value(coordinator['command'],'--out'))!=out:
        raise ValueError('Wrong frozen coordinator')
    expected=p._command_value(coordinator['command'],'--gpu-lock-owner')
    if owner()!=expected:raise ValueError('GPU owner changed')
    fd=os.pidfd_open(coordinator['pid']);paused=False
    status=dict(status='preparing_allocator_resume',controller_pid=os.getpid(),active_workers=None,allocator_env=ALLOCATOR_ENV)
    def persist():p.atomic(attempt/'status.json',status)
    def release_old():
        if live(coordinator):
            signal.pidfd_send_signal(fd,signal.SIGTERM);signal.pidfd_send_signal(fd,signal.SIGCONT)
    try:
        signal.pidfd_send_signal(fd,signal.SIGSTOP);paused=True
        for _ in range(100):
            if identity(coordinator['pid'])['state']=='T':break
            time.sleep(.02)
        else:raise RuntimeError('Coordinator did not pause')
        state=json.loads((out/'status.json').read_text())
        if state['active_stage']!='randomized_collection':raise ValueError('Drain only during collection')
        workers=[]
        for name in state['active_stages']:
            entry=state['stages'][name];w=identity(entry['pid'])
            if w['state']=='Z':w['command']=entry['command']
            if str(ROOT/'scripts/collect_isaac_zhuhai_recycle.py') not in w['command']:raise ValueError('Unexpected worker')
            target=Path(p._command_value(w['command'],'--out'))
            if not target.is_relative_to(Path(plan['collection_import']['root'])/'zhuhai_task2/randomized/shards'):raise ValueError('Unexpected writer path')
            w.update(stage=name,manifest=str(target/'manifest.json'),shard_index=int(p._command_value(w['command'],'--shard-index')));workers.append(w)
        command=resume_command(json.loads((out/'launch.json').read_text())['command'])
        p.write_once(attempt/'intent.json',dict(old_coordinator=coordinator,old_wrapper=wrapper,workers=workers,resume_command=command,plan_sha256=p.sha(out/'plan.json'),allocator_env=ALLOCATOR_ENV,failure_streak_reset=False,resource_policy=RESOURCE_POLICY,started_unix=time.time()))
        status.update(status='draining_owned_workers',active_workers=len(workers));persist()
        log('回收仿真驻留内存：协调器暂停，worker完成当前有限小批次后正常退出；新进程限制glibc arena=2，再按同一冻结计划尝试四路，原90%/RAM底线及连续失败保护不变',actor='allocator_resume')
        last_log=time.monotonic()
        while True:
            if not live(coordinator) or identity(coordinator['pid'])['state']!='T':raise RuntimeError('Old coordinator no longer paused')
            active=[w for w in workers if live(w)];counts={}
            if active:
                resource=p.collection_resource_admission();mem=resource.get('memory',{})
                minimum=max(RESOURCE_POLICY['minimum_available_ram_bytes'],RESOURCE_POLICY['minimum_available_ram_fraction']*mem.get('total_bytes',32*p.GIB))
                if mem.get('available_bytes',0)<minimum or (resource.get('gpu_free_bytes') or 0)<RESOURCE_POLICY['minimum_free_gpu_bytes'] or resource.get('swap_fraction',1)>=.95:
                    raise RuntimeError('Resource reserve reached while draining; halt safely, no threshold bypass')
            else:resource=p.collection_resource_admission()
            for w in workers:
                m=json.loads(Path(w['manifest']).read_text());counts[w['stage']]=m['success_count']
                if live(w):continue
                if m['status']=='worker_recycle_boundary':validate_recycle_boundary(w['manifest'],task='zhuhai_task2')
                elif m['status']=='complete':p.collection_receipt(Path(w['manifest']),'zhuhai_task2','randomized',shard_index=w['shard_index'],shard_count=4)
                else:raise RuntimeError(f"Worker halted {w['stage']}: {m['status']}; no failure reset")
            status.update(active_workers=len(active),counts=counts,resources=resource,checked_unix=time.time());persist()
            if not active:break
            if time.monotonic()-last_log>=60:
                log('内存回收待当前小批次完成：'+str(counts)+'；正常示教不中断，随后恢复四路准入',actor='allocator_resume');last_log=time.monotonic()
            time.sleep(3)
        status['status']='reaping_old_coordinator';persist();release_old()
        deadline=time.monotonic()+180
        while live(coordinator) or live(wrapper):
            if time.monotonic()>deadline:raise RuntimeError('Old process groups not reaped; do not launch')
            time.sleep(1)
        p.verify_plan_sources(plan)
        status['before_resume']=p.collection_resource_admission();persist()
        with (attempt/'coordinator.log').open('x') as handle:proc=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        p.write_once(attempt/'activated.json',dict(pid=proc.pid,command=command,activated_unix=time.time(),plan_sha256=p.sha(out/'plan.json')))
        status.update(status='resumed_allocator_capped_pipeline',pipeline_wrapper_pid=proc.pid);persist()
        log('同一餐具冻结队列已恢复，新worker使用arena=2并最多四路；数据/评分/20k最终10测不变，失败历史保留',actor='allocator_resume')
        code=proc.wait();status.update(status='pipeline_exited',exit_code=code,finished_unix=time.time());persist();return code
    except BaseException as exc:
        status.update(status='halted_allocator_resume',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());persist()
        if paused:release_old()
        raise
    finally:os.close(fd)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--attempt',required=True,type=Path);ap.add_argument('--coordinator-pid',required=True,type=int);ap.add_argument('--wrapper-pid',required=True,type=int);args=ap.parse_args()
    with (args.out/'.allocator_resume_v2.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);raise SystemExit(main(args))
