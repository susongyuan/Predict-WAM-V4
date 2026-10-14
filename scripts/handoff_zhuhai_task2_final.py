#!/usr/bin/env python3
"""Drain current finite-lived task2 workers before changing training/eval schedule.

Pause only the coordinator, preserve physical attempts, enforce original reserve
thresholds during drain, then reap and hand off to the separately frozen run.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_zhuhai_task2_final as pipeline
from scripts.handoff_zhuhai_100_reuse import identity,live
from scripts.gpu_lock import owner
from scripts.collect_isaac_zhuhai_recycle import validate_recycle_boundary,RESOURCE_POLICY
from scripts.progress1_log_task2_first import log

def main(args):
    old=args.old_root.resolve();out=args.out.resolve()
    plan=json.loads((out/'plan.json').read_text());old_plan=json.loads((old/'plan.json').read_text())
    pipeline.verify_plan_sources(plan);pipeline.verify_plan_sources(old_plan)
    if plan['collection_import']!=dict(root=str(old),plan_sha256=pipeline.sha(old/'plan.json'),immutable_collector_and_existing_attempts=True):
        raise ValueError('Frozen collection import mismatch')
    coordinator=identity(args.coordinator_pid);wrapper=identity(args.wrapper_pid)
    if str(ROOT/'scripts/run_zhuhai_task2_first.py') not in coordinator['command'] or pipeline._command_value(coordinator['command'],'--out')!=str(old):
        raise ValueError('Wrong old coordinator identity')
    expected_owner=pipeline._command_value(coordinator['command'],'--gpu-lock-owner')
    if owner()!=expected_owner:raise ValueError('Old GPU owner mismatch')
    old_state=json.loads((old/'status.json').read_text())
    if old_state['active_stage']!='randomized_collection':raise ValueError('Drain only during collection')
    workers=[]
    for name in old_state['active_stages']:
        entry=old_state['stages'][name];worker=identity(entry['pid'])
        if str(ROOT/'scripts/collect_isaac_zhuhai_recycle.py') not in worker['command']:raise ValueError('Unexpected child')
        path=Path(pipeline._command_value(worker['command'],'--out'))
        if not path.is_relative_to(old/'zhuhai_task2/randomized/shards'):raise ValueError('Unexpected collection writer')
        worker.update(stage=name,manifest=str(path/'manifest.json'),shard_index=int(pipeline._command_value(worker['command'],'--shard-index')))
        workers.append(worker)
    pipeline.write_once(out/'handoff_intent.json',dict(old_coordinator=coordinator,old_wrapper=wrapper,workers=workers,new_plan_sha256=pipeline.sha(out/'plan.json'),reason='Latest user: continuous task2 20k then final 10 episodes; no intermediate evaluations',started_unix=time.time()))
    fd=os.pidfd_open(coordinator['pid']);stopped=False
    status=dict(status='draining_existing_task2_workers',active_stage='randomized_collection',handoff_pid=os.getpid(),resource_fallbacks=[])
    def persist():pipeline.atomic(out/'handoff_status.json',status)
    def release_old():
        if live(coordinator):
            signal.pidfd_send_signal(fd,signal.SIGTERM);signal.pidfd_send_signal(fd,signal.SIGCONT)
    try:
        signal.pidfd_send_signal(fd,signal.SIGSTOP);stopped=True
        for _ in range(100):
            if identity(coordinator['pid'])['state']=='T':break
            time.sleep(.02)
        else:raise RuntimeError('Coordinator did not pause')
        persist();log('用户改为task2连续20k后测10组：旧协调器暂停，四路采集完成当前小批次后正常退出，另版接续；不中断正常示教',actor='handoff_task2_final')
        requeued=set();last_log=time.monotonic()
        while True:
            if not live(coordinator) or identity(coordinator['pid'])['state']!='T':raise RuntimeError('Old coordinator no longer paused')
            active=[w for w in workers if live(w)]
            admission=pipeline.collection_resource_admission();memory=admission.get('memory',{})
            minimum=max(RESOURCE_POLICY['minimum_available_ram_bytes'],RESOURCE_POLICY['minimum_available_ram_fraction']*memory.get('total_bytes',32*pipeline.GIB))
            if active and (memory.get('available_bytes',0)<minimum or (admission.get('gpu_free_bytes') or 0)<RESOURCE_POLICY['minimum_free_gpu_bytes'] or admission.get('swap_fraction',1)>=.95):
                if len(active)<4:raise RuntimeError('Drain resource reserve reached; halt safely, no threshold bypass')
            if len(active)==4 and (admission.get('swap_fraction',1)>=.90 or memory.get('available_bytes',0)<minimum or (admission.get('gpu_free_bytes') or 0)<RESOURCE_POLICY['minimum_free_gpu_bytes']):
                newest=max(active,key=lambda w:w['start_ticks'])
                wfd=os.pidfd_open(newest['pid'])
                try:signal.pidfd_send_signal(wfd,signal.SIGTERM)
                finally:os.close(wfd)
                requeued.add(newest['stage']);status['resource_fallbacks'].append(dict(stage=newest['stage'],admission=admission,partial_preserved=True,checked_unix=time.time()))
            counts={}
            for w in workers:
                m=json.loads(Path(w['manifest']).read_text());counts[w['stage']]=m['success_count']
                if live(w):continue
                if w['stage'] in requeued:continue
                if m['status']=='worker_recycle_boundary':validate_recycle_boundary(w['manifest'],task='zhuhai_task2')
                elif m['status']=='complete':pipeline.collection_receipt(Path(w['manifest']),'zhuhai_task2','randomized',shard_index=w['shard_index'],shard_count=4)
                else:raise RuntimeError(f"Worker halted {w['stage']}: {m['status']}; preserve evidence and failure guard")
            status.update(counts=counts,active_workers=len(active),last_admission=admission,checked_unix=time.time());persist()
            if not active:break
            if time.monotonic()-last_log>=60:
                log('餐具最终10测交接：四路当前小批次继续，成功计数'+str(counts)+'；训练尚未开始',actor='handoff_task2_final');last_log=time.monotonic()
            time.sleep(5)
        status['status']='releasing_old_coordinator';persist();release_old()
        deadline=time.monotonic()+180
        while live(coordinator) or live(wrapper):
            if time.monotonic()>deadline:raise RuntimeError('Old writers not reaped; no new launch')
            time.sleep(1)
        pipeline.verify_plan_sources(plan)
        command=json.loads((out/'launch.json').read_text())['command']
        with (out/'coordinator.log').open('x') as handle:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        pipeline.write_once(out/'activated.json',dict(pid=proc.pid,command=command,activated_unix=time.time()))
        status.update(status='activated_final_only_pipeline',pipeline_wrapper_pid=proc.pid);persist()
        log('餐具20k后最终10测队列已接续；四路续采保留原合同与全部失败/成功记录，不清零失败计数',actor='handoff_task2_final')
        code=proc.wait();status.update(status='pipeline_exited',exit_code=code,finished_unix=time.time());persist();return code
    except BaseException as exc:
        status.update(status='halted_handoff',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());persist()
        if stopped:release_old()
        raise
    finally:os.close(fd)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--old-root',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);parser.add_argument('--coordinator-pid',required=True,type=int);parser.add_argument('--wrapper-pid',required=True,type=int)
    args=parser.parse_args()
    with (args.out/'.handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);raise SystemExit(main(args))
