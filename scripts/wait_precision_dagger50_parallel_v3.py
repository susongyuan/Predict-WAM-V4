#!/usr/bin/env python3
"""Drain the serial attempt, adopt evidence, then adaptive parallel DAgger50."""
from pathlib import Path
import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import wait_precision_dagger50_direct_v2 as serial
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts import run_zhuhai_task1_dr200_concurrent as admission
from scripts import gpu_lock
from harness.zhuhai_dagger import DaggerEpisode
from harness.zhuhai_dagger_parallel import snapshot,can_expand
from scripts.progress1_log import log


def unit_value(unit,key):
    return subprocess.run(['systemctl','--user','show',unit,'-p',key,'--value'],
        capture_output=True,text=True,check=True,timeout=5).stdout.strip()


def process_state(pid):
    try:
        text=(Path('/proc')/str(pid)/'stat').read_text()
        return text[text.rfind(')')+2:].split()[0]
    except FileNotFoundError:return None


def build(args):
    plan=serial.build(args)
    authorization=e.read_json(args.switch_authorization)
    e.require(authorization['reason']=='user_authorized_parallel_collection','Parallel user authorization required')
    previous=Path(authorization['previous_run']);old=e.read_json(previous/'plan.json')
    e.require(old['checkpoint']==plan['checkpoint'],'Serial/parallel frozen policy differs')
    # Fresh seeds, fixed TRAIN parents, all attempted serial seeds preserved.
    plan['candidates']=[dict(row,seed=1011000+i) for i,row in enumerate(plan['candidates'])]
    for rel in ('harness/zhuhai_dagger_parallel.py','scripts/wait_precision_dagger50_parallel_v3.py',
                'scripts/collect_precision_dagger_episode_parallel_v3.py'):
        p=ROOT/rel;plan['extension_sources'][str(p)]=e.sha256_file(p)
    plan.update(switch_authorization=str(Path(args.switch_authorization).resolve()),
        switch_authorization_sha256=e.sha256_file(args.switch_authorization),
        previous_run=str(previous),previous_plan_sha256=e.sha256_file(previous/'plan.json'),
        parallel=dict(maximum_workers=4,policy_workers_at_once=1,stagger_seconds=20,
            cold_start_ram_reserve_bytes=7*1024**3,additional_runtime_RAM_floor='max(2GiB,10% total RAM)',
            expansion_gpu_free_bytes=6*1024**3, fourth_worker_swap_exit_fraction=.90,
            absolute_swap_exit_fraction=.95,policy_slot_lock=str(Path(plan['out'])/'policy_slot.lock'),
            child_authorization='one coordinator exclusive GPU owner, explicit slot/seed/plan/PID participant receipts',
            no_training_concurrency=True), collection_variant='adaptive_parallel_DAgger50_v3')
    return plan


def verify(plan):
    serial.verify(plan)
    e.require(e.sha256_file(plan['switch_authorization'])==plan['switch_authorization_sha256'],'Parallel authorization changed')
    e.require(e.sha256_file(Path(plan['previous_run'])/'plan.json')==plan['previous_plan_sha256'],'Serial plan changed')


def drain_previous(plan,out):
    authorization=e.read_json(plan['switch_authorization']);unit=authorization['previous_unit']
    pid=authorization['stopped_parent_pid'];child=authorization['draining_child_pid']
    e.require(unit_value(unit,'MainPID')==str(pid) and process_state(pid)=='T','Expected stopped serial coordinator')
    deadline=time.monotonic()+3900
    while process_state(child) not in (None,'Z'):
        e.require(time.monotonic()<deadline,'Bounded serial drain timed out; preserve original attempt')
        time.sleep(2)
    # Pending SIGINT is delivered on resume; the already-finished child is not
    # interrupted. Original coordinator keeps its finalizer/GPU release logic.
    os.kill(pid,signal.SIGINT);os.kill(pid,signal.SIGCONT)
    deadline=time.monotonic()+90
    while unit_value(unit,'ActiveState') in ('active','activating','deactivating'):
        e.require(time.monotonic()<deadline,'Serial coordinator did not finish drain')
        time.sleep(1)
    e.require(gpu_lock.owner() is None and not admission.gpu_jobs(admission.processes()),'Serial GPU drain incomplete')
    e.write_json(out/'serial_drain_receipt.json',dict(previous_run=plan['previous_run'],
        serial_coordinator_pid=pid,finished_child_pid=child,all_original_evidence_preserved=True,
        original_manifest_sha256=e.sha256_file(Path(plan['previous_run'])/'manifest.json'),gpu_owner_released=True))


def admitted_row(raw,candidate):
    row=dict(path=raw['data_path'],sha256=raw['data_sha256'],seed=candidate['seed'],
        family_id=candidate['family_id'],source_parent=candidate['source_parent'],split='train',domain='dagger',
        policy_steps=raw['policy_steps'],expert_steps=raw['expert_steps'],takeover_step=raw['takeover_step'])
    ep=DaggerEpisode(row)
    e.require(ep.supervised_count==row['expert_steps'],'Expert-only supervised length mismatch')
    return row


def execute(plan):
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=False);e.write_json(out/'plan.json',plan)
    attempts=[];rows=[];running={};held=False;failures=0;guard=None;cursor=0;last_launch=-1e9
    def publish(status,error=None):
        value=dict(schema=plan['schema'],status=status,checkpoint=plan['checkpoint'],success_count=len(rows),
            requested_success_count=50,attempts=attempts,train_rows=rows,
            active_participants=[dict(slot=slot,seed=x['candidate']['seed'],pid=x['process'].pid,out=str(x['target']))
                for slot,x in sorted(running.items())],maximum_workers=4,policy_workers_at_once=1,
            consecutive_expert_failures=failures,initial_validation_passed=len(rows)>=10,
            record_video=False,third_view=False,heldout_used=False,training_started=False,
            previous_run=plan['previous_run'])
        if error:value['error']=error
        e.write_json(out/'manifest.json',value)
    publish('draining_serial_collection')
    def consume(raw,path,candidate,code=0,adopted=False):
        nonlocal failures
        status=raw['status']
        attempts.append(dict(seed=candidate['seed'],status=status,raw_result_path=str(path),
            raw_result_sha256=e.sha256_file(path),adopted=adopted))
        e.require(code==0 and status!='infrastructure_error','Collection infrastructure error; retain evidence, do not retry')
        if status=='success':rows.append(admitted_row(raw,candidate));failures=0
        elif status=='expert_failed':failures+=1
        elif status not in ('policy_succeeded_without_correction','resource_paused'):
            raise ValueError('Unknown terminal correction status: '+status)
        if failures>=3:
            e.write_json(out/'expert_failure_guard.json',dict(consecutive_failures=failures,attempts=attempts[-3:],
                order='durable coordinator completion-consumption order; inherited serial history'))
            raise ValueError('Three consecutive expert failures; drain other participants and retain guard')
        log(f'并行DAgger seed{candidate["seed"]} {status}，合格{len(rows)}/50，连续专家失败{failures}。','dagger_parallel')
    try:
        verify(plan);drain_previous(plan,out)
        previous=Path(plan['previous_run']);old=e.read_json(previous/'plan.json')
        prior_candidates={r['seed']:r for r in old['candidates']}
        # Durable attempt intents include the last drained child, even though
        # SIGSTOP prevented the old controller from publishing its result row.
        intents=sorted(previous.glob('attempt_seed_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
        for intent in intents:
            seed=e.read_json(intent)['seed'];path=previous/'episodes'/f'seed_{seed}'/'result.json'
            raw=e.read_json(path);consume(raw,path,prior_candidates[seed],adopted=True)
        e.write_json(out/'prior_admission.json',dict(adopted_attempts=len(attempts),valid_corrections=len(rows),
            original_results_unmodified=True,prior_attempts=copy.deepcopy(attempts)))
        gpu_lock.acquire(plan['gpu_lock_owner'],max_wait=0);held=True
        env=os.environ.copy();env.update(plan['child_env'])
        with Path(plan['checkpoint']['path']).open('rb') as checkpoint:
            fcntl.flock(checkpoint,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (out/'episodes').mkdir();publish('parallel_collecting')
            while len(rows)<50 or running:
                verify(plan)
                # Reap every owned process before resource admission. No
                # discovered or foreign process is signalled.
                for slot,item in list(sorted(running.items())):
                    code=item['process'].poll()
                    if code is None:
                        e.require(time.monotonic()-item['started']<3900,'Participant wall deadline')
                        continue
                    item['log'].close();del running[slot]
                    raw=e.read_json(item['target']/'result.json')
                    try:consume(raw,item['target']/'result.json',item['candidate'],code=code)
                    except BaseException as exc:guard=guard or f'{type(exc).__name__}: {exc}'
                    publish('draining_guard' if guard else 'parallel_collecting',guard)
                resource=snapshot()
                phases={slot:e.read_json(item['target']/'result.json').get('status')
                    if (item['target']/'result.json').exists() else 'starting' for slot,item in running.items()}
                with (out/'resource_snapshots.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(monotonic=time.monotonic(),active_count=len(running),phases=phases,**resource))+'\n')
                if guard:
                    if not running:raise ValueError(guard)
                elif (len(rows)+len(running)<50 and cursor<len(plan['candidates'])
                        and time.monotonic()-last_launch>=20 and can_expand(resource,len(running))
                        and not any(v=='starting' for v in phases.values())):
                    candidate=plan['candidates'][cursor];slot=next(s for s in range(4) if s not in running)
                    e.require(e.sha256_file(candidate['parent_path'])==candidate['parent_sha256'],'TRAIN parent changed')
                    target=out/'episodes'/f'seed_{candidate["seed"]}'
                    command=[str(e.PYTHON),'-B',str(ROOT/'scripts/collect_precision_dagger_episode_parallel_v3.py'),
                        '--plan',str(out/'plan.json'),'--index',str(cursor),'--worker-slot',str(slot),
                        '--gpu-lock-owner',plan['gpu_lock_owner']]
                    intent=dict(seed=candidate['seed'],slot=slot,command=command,resource=resource,out=str(target))
                    e.write_json(out/f'attempt_seed_{candidate["seed"]}.json',intent)
                    logfile=(out/f'seed_{candidate["seed"]}.log').open('xb')
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
                    running[slot]=dict(process=child,log=logfile,target=target,candidate=candidate,started=time.monotonic())
                    e.write_json(out/f'participant_seed_{candidate["seed"]}.json',dict(**intent,pid=child.pid,pgid=child.pid))
                    cursor+=1;last_launch=time.monotonic();publish('parallel_collecting')
                elif not running and cursor>=len(plan['candidates']) and len(rows)<50:
                    raise ValueError('Parallel candidate budget exhausted')
                elif not running and not can_expand(resource,0) and len(rows)<50:
                    publish('waiting_for_resources')
                if len(rows)==50 and not running:break
                time.sleep(5)
        e.require(len(rows)==50,'Exact 50 valid corrections required');verify(plan);publish('complete');return 0
    except BaseException as exc:
        publish('halted',f'{type(exc).__name__}: {exc}');raise
    finally:
        for slot,item in running.items():
            if item['process'].poll() is None:
                os.killpg(item['process'].pid,signal.SIGINT)
                try:item['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(item['process'].pid,signal.SIGTERM);item['process'].wait(timeout=15)
            item['log'].close()
        if held:
            if admission.gpu_jobs(admission.processes()):e.write_json(out/'owner_retained_guard.json',dict(owner=plan['gpu_lock_owner']))
            else:gpu_lock.release(plan['gpu_lock_owner'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--train-diagnostic',required=True);p.add_argument('--out',required=True)
    p.add_argument('--cancellation-receipt',required=True);p.add_argument('--switch-authorization',required=True)
    p.add_argument('--gpu-lock-owner',default='zhuhai-dagger50-parallel-20261010-v3');p.add_argument('--execute',action='store_true')
    args=p.parse_args();plan=build(args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False));return 0
    return execute(plan)


if __name__=='__main__':raise SystemExit(main())
