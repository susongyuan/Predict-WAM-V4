#!/usr/bin/env python3
"""Explicit 20k continuation, original weights/trainer/rules/limiter unchanged.

Fresh controller namespace only. Adopt every already-attempted fixed seed;
never rerun physical failures. Distinguish zero-control bootstrap failure from
known control-safety aborts under the separately frozen attempt protocol.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import gpu_lock
from scripts import run_zhuhai_wam_precision_v2_pipeline as original
from scripts.progress1_log import log
h=original.precision
BASE=ROOT/'outputs/zhuhai2026_tasks12'
ORIGIN=BASE/'wam_precision_v2_task2_30k_20261010_v1'
OUT=BASE/'wam_precision_v2_task2_30k_continuation_20261010_r1'
AUDIT=BASE/'verification/precision_v2_overnight_20261009'
EVAL=ROOT/'scripts/eval_zhuhai_precision_fixed_attempts.py'
OWNER='zhuhai-precision-v2-continuation-20261010-r1'
SEEDS=list(range(932101,932106))
PROFILE=BASE/'verification/tasks12_collection_repair/task1_expert_repair_v2_20261009_c/launch.json'


def require(ok,message):
    if not ok:raise ValueError(message)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin',type=Path,default=ORIGIN)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--gpu-lock-owner',default=OWNER)
    p.add_argument('--execute',action='store_true')
    return p


def replace_flag(command,flag,value):
    command[command.index(flag)+1]=str(value)


def build_plan(args):
    original.configure()
    origin=h.no_symlink(args.origin.resolve());out=h.no_symlink(args.out.resolve())
    require(origin==ORIGIN and out==OUT,'This explicit recovery is bound to its original and fresh controller namespaces')
    require(args.gpu_lock_owner==OWNER,'Use the distinct continuation owner, never share the standalone evaluation owner')
    old=h.read_json(origin/'plan.json');state=h.read_json(origin/'status.json')
    require(state.get('status')=='failed' and state.get('active_stage')=='development_20000'
            and state.get('completed_stages')==['tiny_admission','train_to_20000'],'Only the recorded post20k interruption is admitted')
    h.verify_code_hashes(old['source_hashes'])
    status=h.read_json(origin/'train/status.json')
    require(status.get('step')==20000 and status.get('target_steps')==30000 and status.get('status')=='stopped_at_safe_boundary','Exact stopped20k optimizer boundary required')
    bad=origin/'development_20000/zhuhai_task2/seed_932101/result.json';error=h.read_json(bad)
    require(error.get('control_steps')==0 and error.get('full_task_success') is None and 'EULA not accepted' in error.get('error',''),'The preserved bootstrap attempt must have zero physical controls')
    audit=h.read_json(AUDIT/'checkpoint_020000_audit.json')
    require(audit.get('status')=='verified' and audit['full_state']['step']==20000 and audit['model_only']['step']==20000,'Independent full20k audit required')
    last=origin/'train/checkpoint_last.pt';last_receipt=h.read_json(last.with_name(last.name+'.sha256.json'))
    require(last_receipt['step']==20000 and last_receipt['sha256']==audit['full_state']['sha256'] and h.sha256_file(last)==last_receipt['sha256'],'Resume state changed since the verified boundary')
    policy=h.read_json(AUDIT/'fixed_attempts_protocol.json')
    require(policy['checkpoints']==[20000,25000,30000] and policy['seeds']==SEEDS
            and policy['scoring_source_unchanged'] and policy['postprocessing_source_unchanged'],'Fixed-attempt clarification must not alter scoring/control')
    profile=h.read_json(PROFILE)['plan']['child_env']
    require(profile.get('OMNI_KIT_ACCEPT_EULA')=='YES' and Path(profile['LD_PRELOAD']).is_file(),'Restore only the already-used Isaac bootstrap profile')
    adopted=[origin/'development_20000_bootstrap_recovery_20261010_r1/zhuhai_task2/seed_932101/result.json',
             origin/'development_20000_fixed_attempts_seed932102_r2/zhuhai_task2/seed_932102/result.json']
    stages=[]
    for old_stage in old['stages'][2:]:
        stage=copy.deepcopy(old_stage)
        if stage['kind']=='train':
            require(stage['step']in(25000,30000) and '--resume'in stage['command'],'No fresh training or budget restart')
            replace_flag(stage['command'],'--gpu-lock-owner',args.gpu_lock_owner)
            require(stage['command'][stage['command'].index('--steps')+1]=='30000','Full schedule remains30k')
        else:
            require(stage['seeds']==SEEDS,'Never replace or drop a fixed seed')
            stage['out']=str(out/stage['name'])
            cmd=[str(original.base.PYTHON),'-B',str(EVAL),'--checkpoint',str(origin/'train'/f"checkpoint_step_{stage['step']:06d}.pt"),
                 '--task',h.TASK,'--seeds',*map(str,SEEDS),'--out',stage['out'],'--gpu-lock-owner',args.gpu_lock_owner,'--execute']
            stage.update(command=cmd,evaluation_protocol='fixed_attempts_with_explicit_original_control_safety_aborts')
            if stage['step']==20000:
                stage['adopt_existing']=True;stage['adopted_results']=list(map(str,adopted))
                stage['adoption_semantics']='retain both already-attempted raw cases, then execute only the remaining three; never rerun a failure'
                for path in adopted:cmd.extend(['--adopt-result',str(path)])
        stages.append(stage)
    require([s['step']for s in stages]==[20000,25000,25000,30000,30000],'Continuation schedule changed')
    additions=[Path(__file__),EVAL,AUDIT/'fixed_attempts_protocol.json',AUDIT/'bootstrap_recovery_authorization.json',PROFILE,Path(profile['LD_PRELOAD']),origin/'plan.json',bad]
    require(all(p.is_file()for p in additions),'Required continuation source/receipt not ready')
    hashes=dict(old['source_hashes']);hashes.update({str(p.resolve()):h.sha256_file(p)for p in additions})
    return dict(schema=old['schema'],continuation_schema='precision-v2-explicit-20k-continuation-v1',out=str(out),
        training_origin=str(origin),origin_plan_sha256=h.sha256_file(origin/'plan.json'),gpu_lock_owner=args.gpu_lock_owner,
        task=h.TASK,stages=stages,source_hashes=hashes,training_updates=30000,batch_size=4,
        checkpoint_selection=old['checkpoint_selection'],development_seeds=SEEDS,final_seeds=[],
        record_video=False,third_view=False,collection_parallel=False,planned_physical_episodes=15,
        inherited_completed_stages=state['completed_stages'],inherited_zero_control_error=str(bad),
        frozen_attempt_protocol=str(AUDIT/'fixed_attempts_protocol.json'),evaluation_environment=profile,
        initial_resume_receipt=last_receipt,initial_resume_backup=str(AUDIT/'checkpoint_resume_020000_preserved.pt'),
        protocol_change='restore proven Isaac bootstrap environment; original limiter/rules frozen; retain known control-safety aborts as unsuccessful fixed attempts, no physical retries')


def verify_stage(stage):
    if stage['kind']=='train':return original.verify_result(stage)
    from scripts.eval_zhuhai_precision_fixed_attempts import verify_summary
    ck=Path(stage['command'][stage['command'].index('--checkpoint')+1])
    receipt=h.read_json(ck.with_name(ck.name+'.sha256.json'))
    return verify_summary(Path(stage['out'])/'summary.json',stage['step'],stage['seeds'],receipt['sha256'])


def run_stage(plan,stage):
    root=Path(plan['out']);env=os.environ.copy()
    if stage['kind']=='eval':env.update(plan['evaluation_environment'])
    if stage['kind']=='train':
        prior=20000 if stage['step']==25000 else 25000
        receipt=h.read_json(Path(stage['out'])/'checkpoint_last.pt.sha256.json')
        require(receipt.get('step')==prior and receipt.get('model_only') is False,'Wrong full-state resume boundary')
        if prior==20000:require(receipt==plan['initial_resume_receipt'],'Do not silently substitute another20k resume state')
    with(root/(stage['name']+'.log')).open('xb')as stream,(root/(stage['name']+'.resources.jsonl')).open('x')as metrics:
        child=subprocess.Popen(stage['command'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                row=original.resource_snapshot();row.update(stage=stage['name'],child_pid=child.pid)
                metrics.write(json.dumps(row)+'\n');metrics.flush()
                try:return child.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
        except BaseException:
            if child.poll()is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=180)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise


def summarize(plan):
    rows=[]
    for stage in plan['stages']:
        if stage['kind']!='eval':continue
        p=Path(stage['out'])/'summary.json'
        if not p.exists():rows.append(dict(step=stage['step'],status='not_completed',success_rate=None));continue
        value=h.read_json(p)
        row=dict(step=stage['step'],status=value.get('status'),evidence=str(p),success_rate=None)
        if value.get('status')=='completed':
            verify_stage(stage);episodes=value['episodes'];n=len(episodes);k=sum(x['full_task_success']is True for x in episodes)
            z=1.959963984540054;den=1+z*z/n;center=(k/n+z*z/(2*n))/den;half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
            row.update(successes=k,episodes=n,success_rate=k/n,safety_aborts=sum(x['status']=='control_safety_abort'for x in episodes),
                       wilson95=[max(0.,center-half),min(1.,center+half)],seeds=stage['seeds'])
        rows.append(row)
    h.write_json(Path(plan['out'])/'milestone_results.json',dict(schema='precision-v2-fixed-attempt-milestones-v1',results=rows,
        fixed_final_step=30000,paired_small_sample_not_independent_generalization=True,original_control_and_scoring_unchanged=True,physical_failures_not_retried=True))


def execute(plan,runner=run_stage):
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=True);owned=False;done=[]
    with(out/'.pipeline.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not any(p.name!='.pipeline.lock'for p in out.iterdir()),'Fresh continuation only; preserve prior controller/evidence')
        h.verify_code_hashes(plan['source_hashes']);h.write_json(out/'plan.json',plan,immutable=True)
        state=dict(status='waiting_for_exclusive_gpu',active_stage='development_20000',completed_stages=done,training_origin=plan['training_origin'])
        h.write_json(out/'status.json',state)
        try:
            gpu_lock.acquire(plan['gpu_lock_owner'],max_wait=4*3600);owned=True
            for stage in plan['stages']:
                h.verify_owner(plan['gpu_lock_owner']);h.verify_code_hashes(plan['source_hashes'])
                state.update(status='running',active_stage=stage['name'],completed_stages=done.copy());h.write_json(out/'status.json',state)
                log(f"V2显式续行启动{stage['name']}；原参数/评分/限幅，固定失败尝试保留不重试",actor='precision_continuation')
                code=runner(plan,stage)
                if code!=0:raise RuntimeError(f"{stage['name']} exit={code}; preserve and stop without retry")
                verify_stage(stage);done.append(stage['name']);summarize(plan)
                log(f"V2显式续行完成{stage['name']}；已保留全部固定尝试及安全中止",actor='precision_continuation')
            state.update(status='completed',active_stage=None,completed_stages=done);h.write_json(out/'status.json',state)
            return 0
        except BaseException as exc:
            state.update(status='failed',completed_stages=done,error=f'{type(exc).__name__}: {exc}');h.write_json(out/'status.json',state)
            raise
        finally:
            try:summarize(plan)
            finally:
                if owned:gpu_lock.release(plan['gpu_lock_owner'])


def main(argv=None):
    args=parser().parse_args(argv);plan=build_plan(args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    return execute(plan)


if __name__=='__main__':raise SystemExit(main())
