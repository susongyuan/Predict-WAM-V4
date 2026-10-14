#!/usr/bin/env python3
"""Retain twenty legacy episodes, collect fifty new segmented episodes with causal handbacks."""
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
from harness.zhuhai_dagger_segmented import SegmentedEpisode as DaggerEpisode, SCHEMA
from harness.zhuhai_dagger_parallel import snapshot,can_expand
from harness.zhuhai_dagger_segmented_admission import update_failure_count, actual_handbacks, pilot_eligible
from scripts.progress1_log import log
from harness.zhuhai_dagger_segmented_training_v7 import LEGACY_EPISODES, NEW_EPISODES, TOTAL_EPISODES, correction_counts, validate_quota


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
    e.require(authorization['reason']=='user_authorized_fifty_new_segmented_episodes','Parallel user authorization required')
    previous=Path(authorization['previous_run']);old=e.read_json(previous/'plan.json')
    e.require(old['checkpoint']==plan['checkpoint'],'Serial/parallel frozen policy differs')
    # Prefer not-yet-used TRAIN family bindings; scene geometry is generated
    # from fresh DR seeds, not by replaying the parent's original trajectory.
    from collections import Counter
    used=Counter(r['family_id'] for r in e.read_json(previous/'manifest.json')['train_rows'])
    for participant in e.read_json(previous/'manifest.json')['active_participants']:
        path=Path(participant['out'])/'result.json'
        raw=e.read_json(path) if path.exists() else {}
        if raw.get('status') in ('success','segmented_admitted'):
            used[raw['scenario']['family_id']]+=1
    parents=sorted(plan['candidates'][:75],key=lambda r:used[r['family_id']])
    plan['candidates']=[dict(parents[i%75],seed=1015000+i) for i in range(200)]
    prior_seeds={r['seed'] for r in old['candidates']} | {r['seed'] for r in e.read_json(previous/'manifest.json')['attempts']}
    e.require(not prior_seeds & {r['seed'] for r in plan['candidates']},'Fresh correction seeds overlap previous collection')
    for rel in ('harness/zhuhai_dagger_parallel.py', 'harness/zhuhai_dagger_segmented.py',
                'harness/isaac_zhuhai_segmented_expert.py', 'scripts/precision_segmented_policy_worker.py','harness/zhuhai_dagger_segmented_admission.py','scripts/wait_precision_dagger50_segmented_v7.py',
                'scripts/collect_precision_dagger_segmented_v5.py','harness/zhuhai_dagger_segmented_training_v7.py'):
        p=ROOT/rel;plan['extension_sources'][str(p)]=e.sha256_file(p)
    plan.update(switch_authorization=str(Path(args.switch_authorization).resolve()),
        switch_authorization_sha256=e.sha256_file(args.switch_authorization),
        previous_run=str(previous),previous_plan_sha256=e.sha256_file(previous/'plan.json'),
        parallel=dict(maximum_workers=3,policy_workers_at_once=1,stagger_seconds=20,
            cold_start_ram_reserve_bytes=7*1024**3,additional_runtime_RAM_floor='max(2GiB,10% total RAM)',
            expansion_gpu_free_bytes=6*1024**3, fourth_worker_swap_exit_fraction=.90,
            absolute_swap_exit_fraction=.95,policy_slot_lock=str(Path(plan['out'])/'policy_slot.lock'),
            child_authorization='one coordinator exclusive GPU owner, explicit slot/seed/plan/PID participant receipts',
            no_training_concurrency=True), collection_variant='fifty_new_segmented_plus_twenty_retained_legacy_v7', schema=SCHEMA,
        target_corrections=TOTAL_EPISODES, target_new_segmented_episodes=NEW_EPISODES,
        retained_legacy_episodes=LEGACY_EPISODES, quota_unit='accepted new segmented episodes',
        segmented=dict(maximum_interventions=6, resume_takeover_gate=dict(minimum_steps=60,stalled_steps=80,maximum_steps=180),
            local_goals=['verified_5s_grasp','verified_released_1s_placement'],
            require_real_policy_handback=True, full_task_success_required_for_local_labels=False,
            expert_failure_counter_unit='actual expert segments; a passed span resets, resource/budget interruptions do not increment',
            physical_pilot_scope='verified local spans and real handbacks; later failed spans remain masked and separately reported',
            physical_pilot_minimum_segments=2,physical_pilot_minimum_handbacks=2),
        supervision='verified local expert spans only; mixed real history is context; policy and failed-expert actions never positive labels')
    return plan


def verify(plan):
    serial.verify(plan)
    e.require(e.sha256_file(plan['switch_authorization'])==plan['switch_authorization_sha256'],'Parallel authorization changed')
    e.require(e.sha256_file(Path(plan['previous_run'])/'plan.json')==plan['previous_plan_sha256'],'Serial plan changed')


def drain_previous(plan,out):
    authorization=e.read_json(plan['switch_authorization']);unit=authorization['previous_unit']
    pid=authorization['stopped_parent_pid'];children=authorization['draining_child_pids']
    e.require(unit_value(unit,'MainPID')==str(pid) and process_state(pid)=='T','Expected stopped serial coordinator')
    deadline=time.monotonic()+3900
    while any(process_state(child) not in (None,'Z') for child in children):
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
        serial_coordinator_pid=pid,finished_child_pids=children,all_original_evidence_preserved=True,
        original_manifest_sha256=e.sha256_file(Path(plan['previous_run'])/'manifest.json'),gpu_owner_released=True))


def admitted_row(raw,candidate):
    row=dict(path=raw['data_path'],sha256=raw['data_sha256'],seed=candidate['seed'],
        family_id=candidate['family_id'],source_parent=candidate['source_parent'],split='train',domain='dagger',
        policy_steps=raw['policy_steps'],expert_steps=raw['expert_steps'],takeover_step=raw['takeover_step'])
    ep=DaggerEpisode(row)
    e.require(ep.supervised_count==row['expert_steps'] and ep.policy_count==row['policy_steps'],'Expert/policy archive count mismatch')
    row.update(correction_variant=ep.variant,verified_segment_count=len(ep.valid_segments),
        handback_count=int(actual_handbacks(ep)) if ep.variant=='segmented_local_interventions' else 0,expert_control_steps=raw.get('expert_control_steps',raw['expert_steps']),
        full_task_success=raw['final_rule_evidence']['score']['full_task_success'])
    if ep.variant=='segmented_local_interventions':
        e.require(row['handback_count']==raw.get('handback_count') and row['handback_count']>0,'Actual policy handbacks differ')
    return row


def execute(plan):
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=False);e.write_json(out/'plan.json',plan)
    attempts=[];rows=[];running={};held=False;failures=0;guard=None;cursor=0;last_launch=-1e9
    def new_count():
        return correction_counts(rows)['segmented']
    def publish(status,error=None):
        value=dict(schema=plan['schema'],status=status,checkpoint=plan['checkpoint'],success_count=new_count(),total_training_corrections=len(rows),
            requested_success_count=NEW_EPISODES,retained_legacy_required=LEGACY_EPISODES,attempts=attempts,train_rows=rows,
            active_participants=[dict(slot=slot,seed=x['candidate']['seed'],pid=x['process'].pid,out=str(x['target']))
                for slot,x in sorted(running.items())],maximum_workers=3,policy_workers_at_once=1,
            consecutive_expert_failures=failures,initial_validation_passed=new_count()>=10,
            record_video=False,third_view=False,heldout_used=False,training_started=False,
            previous_run=plan['previous_run'],
            legacy_corrections=sum(r['correction_variant']=='legacy_full_continuation' for r in rows),
            segmented_corrections=sum(r['correction_variant']=='segmented_local_interventions' for r in rows),
            verified_segment_count=sum(r['verified_segment_count'] for r in rows),
            policy_handback_count=sum(r['handback_count'] for r in rows),
            full_task_success_count=sum(r['full_task_success'] for r in rows),
            quota_unit='accepted new segmented episodes')
        if error:value['error']=error
        e.write_json(out/'manifest.json',value)
    publish('draining_serial_collection')
    def consume(raw,path,candidate,code=0,adopted=False):
        nonlocal failures
        status=raw['status']
        attempts.append(dict(seed=candidate['seed'],status=status,raw_result_path=str(path),
            raw_result_sha256=e.sha256_file(path),adopted=adopted))
        e.require(code==0 and status!='infrastructure_error','Collection infrastructure error; retain evidence, do not retry')
        if status in ('success','segmented_admitted'):
            e.require(raw['accepted_for_training'] is True,'Unaccepted correction cannot enter quota')
            row=admitted_row(raw,candidate);rows.append(row)
            failures=update_failure_count(failures,raw)
            if (row['correction_variant']=='segmented_local_interventions'
                    and row['verified_segment_count']>=2 and row['handback_count']>=2
                    and not (out/'segmented_pilot_admission.json').exists()):
                e.require(pilot_eligible(DaggerEpisode(row),raw),'Recorded physical handback pilot differs')
                e.write_json(out/'segmented_pilot_admission.json',dict(seed=row['seed'],
                    actual_verified_segments=row['verified_segment_count'],actual_model_handbacks=row['handback_count'],
                    archive_validated=True,raw_result_path=str(path),raw_result_sha256=e.sha256_file(path),
                    data_sha256=row['sha256'],local_vs_full_success_separate=True,
                    later_expert_failure=raw.get('expert_failure'),failed_spans_excluded=True,
                    admission_scope=plan['segmented']['physical_pilot_scope']))
        elif status=='expert_failed':failures=update_failure_count(failures,raw)
        elif status not in ('policy_succeeded_without_correction','resource_paused','bounded_no_correction'):
            raise ValueError('Unknown terminal correction status: '+status)
        else:failures=update_failure_count(failures,raw)
        if failures>=3:
            e.write_json(out/'expert_failure_guard.json',dict(consecutive_failures=failures,attempts=attempts[-3:],
                order='durable completion-consumption order, each actual local expert span; legacy single outcomes preserved'))
            raise ValueError('Three consecutive expert failures; drain other participants and retain guard')
        log(f'并行DAgger seed{candidate["seed"]} {status}，新分段合格{new_count()}/{NEW_EPISODES}，保留旧完整{len(rows)-new_count()}，连续专家失败{failures}。','dagger_parallel')
    try:
        verify(plan);drain_previous(plan,out)
        previous=Path(plan['previous_run']);old=e.read_json(previous/'manifest.json')
        e.require(not (previous/'expert_failure_guard.json').exists(),'Prior expert failure guard must remain blocking')
        seen_seeds=set()
        for item in old['attempts']:
            path=Path(item['raw_result_path']);raw=e.read_json(path)
            consume(raw,path,raw['scenario'],adopted=True);seen_seeds.add(raw['seed'])
        # Durable attempt intents include the last drained child, even though
        # SIGSTOP prevented the old controller from publishing its result row.
        intents=sorted(previous.glob('attempt_seed_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
        for intent in intents:
            seed=e.read_json(intent)['seed']
            if seed in seen_seeds:continue
            path=previous/'episodes'/f'seed_{seed}'/'result.json'
            raw=e.read_json(path);consume(raw,path,raw['scenario'],adopted=True)
        e.write_json(out/'prior_admission.json',dict(adopted_attempts=len(attempts),valid_corrections=len(rows),
            original_results_unmodified=True,prior_attempts=copy.deepcopy(attempts)))
        counts=correction_counts(rows)
        e.require(counts['legacy']==LEGACY_EPISODES and counts['segmented']<=NEW_EPISODES,'Prior retained/new quota differs')
        gpu_lock.acquire(plan['gpu_lock_owner'],max_wait=0);held=True
        env=os.environ.copy();env.update(plan['child_env'])
        with Path(plan['checkpoint']['path']).open('rb') as checkpoint:
            fcntl.flock(checkpoint,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (out/'episodes').mkdir();publish('parallel_collecting')
            while new_count()<NEW_EPISODES or running:
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
                elif (new_count()+len(running)<NEW_EPISODES and cursor<len(plan['candidates'])
                        and time.monotonic()-last_launch>=20 and can_expand(resource,len(running),maximum=3 if (out/'segmented_pilot_admission.json').exists() else 1)
                        and not any(v=='starting' for v in phases.values())):
                    candidate=plan['candidates'][cursor];slot=next(s for s in range(3) if s not in running)
                    e.require(e.sha256_file(candidate['parent_path'])==candidate['parent_sha256'],'TRAIN parent changed')
                    target=out/'episodes'/f'seed_{candidate["seed"]}'
                    command=[str(e.PYTHON),'-B',str(ROOT/'scripts/collect_precision_dagger_segmented_v5.py'),
                        '--plan',str(out/'plan.json'),'--index',str(cursor),'--worker-slot',str(slot),
                        '--gpu-lock-owner',plan['gpu_lock_owner']]
                    intent=dict(seed=candidate['seed'],slot=slot,command=command,resource=resource,out=str(target))
                    e.write_json(out/f'attempt_seed_{candidate["seed"]}.json',intent)
                    logfile=(out/f'seed_{candidate["seed"]}.log').open('xb')
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
                    running[slot]=dict(process=child,log=logfile,target=target,candidate=candidate,started=time.monotonic())
                    e.write_json(out/f'participant_seed_{candidate["seed"]}.json',dict(**intent,pid=child.pid,pgid=child.pid))
                    cursor+=1;last_launch=time.monotonic();publish('parallel_collecting')
                elif not running and cursor>=len(plan['candidates']) and new_count()<NEW_EPISODES:
                    raise ValueError('Parallel candidate budget exhausted')
                elif not running and not can_expand(resource,0,maximum=3) and new_count()<NEW_EPISODES:
                    publish('waiting_for_resources')
                if new_count()==NEW_EPISODES and not running:break
                time.sleep(5)
        e.require(new_count()==NEW_EPISODES and len(rows)==TOTAL_EPISODES,'Exact 50 new and 20 legacy episodes required')
        e.require((out/'segmented_pilot_admission.json').exists(),'Physical multi-handback pilot must pass before training')
        verify(plan);publish('complete');validate_quota(e.read_json(out/'manifest.json'));return 0
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
    p.add_argument('--gpu-lock-owner',default='zhuhai-dagger50-segmented-20261010-v7');p.add_argument('--execute',action='store_true')
    args=p.parse_args();plan=build(args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False));return 0
    return execute(plan)


if __name__=='__main__':raise SystemExit(main())
