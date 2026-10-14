#!/usr/bin/env python3
"""Wait for actual zero50k, analyze, collect50 extra DR, train30k, test70k/80k five each."""
from pathlib import Path
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts.supervise_precision_dagger50k_v12 import read,unit_state,verify
from scripts.supervise_precision_dagger_milestones_v13 import conditional_recovery_required
from scripts.analyze_precision_dagger_milestones_v14 import analyze
from scripts import eval_precision_dagger_70k80k_fixed5_v14 as evaluator
from harness import zhuhai_dagger_training30k_v14 as quota
from scripts.progress1_log import log


def complete_five(row,step):
    rows=row.get('episodes',[])
    return (row.get('status')=='completed' and row.get('checkpoint',{}).get('step')==step
        and row.get('planned_episodes')==5 and row.get('seeds')==list(evaluator.SEEDS)
        and [r['seed'] for r in rows]==list(evaluator.SEEDS)
        and all(type(r.get('full_task_success')) is bool for r in rows)
        and row.get('full_successes')==sum(r['full_task_success'] for r in rows)
        and row.get('success_rate')==sum(r['full_task_success'] for r in rows)/5)


def after_trigger_stage(cm,tm,units):
    e.require(cm.get('status')!='halted' and tm.get('status') not in ('failed','halted'),
        'Conditional stage halted: '+str(cm.get('error') or tm.get('error')))
    if cm.get('status')!='complete':return 'collecting_extra50_DR'
    quota.validate_quota(cm)
    if not evaluator.training_finished(tm):return 'training_50k_to80k'
    if any(u['MainPID']!='0' for u in units):return 'waiting_for_training_exit'
    return 'evaluations_ready'


def report(value):
    if value['stage']=='waiting_for_zero50k_decision':
        log(f"条件监督等待两轮十测结果；主流程{value.get('primary_stage')}，正式{value.get('primary_step')}/50000。",'dagger_conditional_supervisor')
        return
    message=(f"追加分支阶段{value['stage']}；本轮新增DR恢复{value.get('newly_added_segmented',0)}/50；"
        f"正式{value.get('formal_step',50000)}/80000；"
        +'; '.join(f"{s}已测{len(r.get('episodes',[]))}/5、完整成功{r.get('full_successes',0)}" for s,r in value.get('evaluations',{}).items())
        +f"；异常{value.get('error') or '无'}。")
    log(message,'dagger_conditional_supervisor')
    p=ROOT/'docs/current_status.md'
    with p.open('r+',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);text=f.read()
        block='<!-- live-10min:start -->\n**实时监督 '+value['time']+'：** '+message+'\n<!-- live-10min:end -->'
        text=re.sub(r'<!-- live-10min:start -->.*?<!-- live-10min:end -->',lambda _:block,text,flags=re.S)
        e.require(len(text.splitlines())<200,'Existing current status must stay under 200 lines')
        f.seek(0);f.write(text);f.truncate()


def service(plan,out,name,command):
    unit=plan[name+'_unit'];env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/1000',
        DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    state=unit_state(unit);e.require(state['MainPID']=='0','Conditional unit already owns a process')
    cmd=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(ROOT),
        '--property=KillMode=mixed',*command]
    result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=30)
    e.write_json(out/(name+'_service.json'),dict(unit=unit,command=cmd,returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr))
    e.require(result.returncode==0,'Conditional service registration failed: '+result.stderr)


def wait_file(path,unit,seconds=600):
    deadline=time.monotonic()+seconds
    while not Path(path).exists():
        state=unit_state(unit)
        e.require(state['MainPID']!='0','Registered process disappeared before plan publication: '+unit)
        e.require(time.monotonic()<deadline,'Plan publication wait deadline; retain process for diagnosis')
        time.sleep(1)


def provision(plan,out,analysis_path):
    old=e.read_json(plan['prior_collection_authorization'])
    auth=dict(old,reason='user_conditional_zero50k_add50DR_train30k_test70k80k',
        previous_run=plan['retained_collection'],failure_analysis=str(analysis_path),
        failure_analysis_sha256=e.sha256_file(analysis_path),completed_pending_admission=[],
        additional_training_steps_after_quota=30000,total_corrections=120,target_new_segmented=100,
        newly_requested_segmented=50,maximum_workers=4,normal_target_workers=3,
        preferred_workers=4,retained_legacy=20,record_video=False,third_view=False,
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
    auth['inactive_previous_units']=list(dict.fromkeys(old['inactive_previous_units']+[
        plan['primary_collection_unit'],plan['primary_training_unit'],plan['primary_supervisor_unit']]))
    primary=unit_state(plan['primary_supervisor_unit']);pid=int(primary['MainPID'])
    state=read(Path(plan['primary_pipeline'])/'status.json')
    e.require(state.get('stage')=='conditional_recovery_pending_analysis' and pid>0,
        'Original paired supervisor must be at the completed conditional handoff')
    e.require('supervise_precision_dagger_milestones_v13.py' in Path(f'/proc/{pid}/cmdline').read_text(),
        'Conditional handoff PID identity differs')
    e.write_json(out/'primary_handoff_receipt.json',dict(pid=pid,previous_status=state,
        reason='completed paired tests triggered user-authorized extra50DR plus30k',
        signal='SIGINT',training_and_evaluation_children_already_exited=True))
    os.kill(pid,signal.SIGINT)
    deadline=time.monotonic()+60
    while unit_state(plan['primary_supervisor_unit'])['MainPID']!='0':
        e.require(time.monotonic()<deadline,'Prior supervisor handoff did not exit');time.sleep(1)
    auth_path=out/'collection_authorization.json';e.write_json(auth_path,auth)
    command=[str(ROOT/'.venvs/wam-fixed/bin/python'),'-B',str(ROOT/'scripts/wait_precision_dagger50_segmented_v14.py'),
        '--checkpoint',plan['parent_model'],'--train-diagnostic',old['train_diagnostic'] if 'train_diagnostic' in old
        else str(ROOT/'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_train_diagnostic10_20261010_v1'),
        '--out',plan['collection'],'--cancellation-receipt',plan['cancellation_receipt'],
        '--switch-authorization',str(auth_path),'--gpu-lock-owner',plan['collection_gpu_owner']]
    p=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=300)
    e.require(p.returncode==0,'ExtraDR collection preflight failed: '+p.stderr)
    e.write_json(out/'collection_preflight.json',json.loads(p.stdout))
    service(plan,out,'collection',command+['--execute'])
    wait_file(Path(plan['collection'])/'plan.json',plan['collection_unit'])
    e.require(read(Path(plan['collection'])/'plan.json')==read(out/'collection_preflight.json'),
        'Actual extraDR collection plan differs from preflight')
    command=[str(ROOT/'.venvs/wam-fixed/bin/python'),'-B',str(ROOT/'scripts/wait_precision_dagger_weighted80k_v16.py'),
        '--parent-full-state',plan['parent_full_state'],'--collection',plan['collection'],
        '--out',plan['training'],'--gpu-lock-owner',plan['training_gpu_owner']]
    p=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=300)
    e.require(p.returncode==0,'Additional30k preflight failed: '+p.stderr)
    e.write_json(out/'training_preflight.json',json.loads(p.stdout))
    service(plan,out,'training',command+['--execute'])
    wait_file(Path(plan['training'])/'plan.json',plan['training_unit'])
    e.require(read(Path(plan['training'])/'plan.json')==read(out/'training_preflight.json'),
        'Actual additional30k training plan differs from preflight')
    log('50k十测0/10已独立分析并触发：50新增DR恢复采集/30k等待续训服务已实际注册，原数据/失败均保留。','dagger_conditional_supervisor')


def execute(plan):
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=False);e.write_json(out/'plan.json',plan)
    lock=(out/'supervisor.lock').open('xb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();last_report=-1e9;previous_stage=None;triggered=False
    child=None;child_log=None;active_step=None;value={}
    try:
        while True:
            verify(plan);e.require(time.monotonic()-started<plan['maximum_seconds'],'Conditional monitoring deadline exceeded')
            if not triggered:
                primary=read(Path(plan['primary_pipeline'])/'status.json')
                e.require(primary.get('status')!='halted','Primary pipeline halted: '+str(primary.get('error')))
                state=unit_state(plan['primary_supervisor_unit'])
                e.require(state['MainPID']!='0' or primary.get('status')=='completed',
                    'Primary monitor disappeared without completion')
                cohorts=primary.get('evaluations',{})
                if primary.get('stage')=='completed':
                    e.require(not conditional_recovery_required(cohorts),'Zero50k cannot finish without authorized branch')
                    analysis_path=out/'failure_analysis_45k50k.json'
                    paths={s:Path(plan['primary_evaluation_outs'][str(s)])/'summary.json' for s in (45000,50000)}
                    e.write_json(analysis_path,analyze(paths))
                    e.write_json(out/'status.json',dict(status='completed',stage='not_needed',
                        conditional_triggered=False,failure_analysis=str(analysis_path),
                        reason='50k achieved at least one complete-task success in ten tests'))
                    e.write_json(out/'final_receipt.json',dict(status='completed',conditional_triggered=False,
                        original_training_step=50000,initial_evaluation_attempts=20,failure_analysis=str(analysis_path)))
                    return 0
                if primary.get('stage')=='conditional_recovery_pending_analysis':
                    e.require(conditional_recovery_required(cohorts),'Conditional trigger requires complete actual zero50k')
                    analysis_path=out/'failure_analysis_45k50k.json'
                    paths={s:Path(plan['primary_evaluation_outs'][str(s)])/'summary.json' for s in (45000,50000)}
                    e.write_json(analysis_path,analyze(paths));provision(plan,out,analysis_path);triggered=True
                else:
                    value=dict(schema=plan['schema'],status='running',stage='waiting_for_zero50k_decision',
                        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                        conditional_triggered=False,primary_step=primary.get('formal_step'),primary_stage=primary.get('stage'))
                    e.write_json(out/'status.json',value)
                    if time.monotonic()-last_report>=600:report(value);last_report=time.monotonic()
                    time.sleep(30);continue
            cm=read(Path(plan['collection'])/'manifest.json');tm=read(Path(plan['training'])/'status.json')
            train=read(Path(plan['training'])/'train/status.json')
            e.require(train.get('status')!='failed','Conditional training failed: '+str(train.get('error')))
            if train.get('status')=='training':
                e.require(time.time()-(Path(plan['training'])/'train/status.json').stat().st_mtime<600,
                    'Additional30k training heartbeat stale; retain process for diagnosis')
            units=[unit_state(plan[k]) for k in ('collection_unit','training_unit')]
            for state,row in zip(units,(cm,tm)):
                e.require(state['MainPID']!='0' or row.get('status') in ('complete','completed','halted','failed'),
                    'Conditional process vanished without terminal receipt')
            stage=after_trigger_stage(cm,tm,units)
            tests={str(s):read(Path(plan['evaluation_outs'][str(s)])/'summary.json') for s in (70000,80000)}
            if child is not None:
                stage=f'evaluating_{active_step}';code=child.poll()
                if code is not None:
                    e.require(code==0 and complete_five(tests[str(active_step)],active_step),
                        'Conditional cohort incomplete: '+str(tests[str(active_step)].get('error')))
                    child_log.close();child=None;child_log=None;active_step=None
            if child is None and (stage=='evaluations_ready' or stage.startswith('evaluating_')):
                remaining=[s for s in (70000,80000) if not complete_five(tests[str(s)],s)]
                if remaining:
                    step=remaining[0];intent=out/f'evaluation_command_{step}.json'
                    e.require(not intent.exists() and not Path(plan['evaluation_outs'][str(step)]).exists(),
                        'Prior conditional evaluation intent retained; no retry')
                    command=plan['evaluation_commands'][str(step)];e.write_json(intent,dict(command=command))
                    child_log=(out/f'evaluation_{step}.log').open('xb')
                    child=subprocess.Popen(command,cwd=ROOT,stdout=child_log,stderr=subprocess.STDOUT,start_new_session=True)
                    active_step=step;stage=f'evaluating_{step}'
                    e.write_json(out/f'evaluation_participant_{step}.json',dict(pid=child.pid,pgid=child.pid,command=command))
                else:
                    analysis=analyze({s:Path(plan['evaluation_outs'][str(s)])/'summary.json' for s in (70000,80000)},
                        steps=(70000,80000),count=5)
                    e.write_json(out/'failure_analysis_70k80k.json',analysis);stage='completed'
            value=dict(schema=plan['schema'],status='completed' if stage=='completed' else 'running',stage=stage,
                time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),conditional_triggered=True,
                newly_added_segmented=cm.get('newly_added_segmented_episodes',0),collection=cm,training=tm,
                training_child=train,formal_step=max(50000,train.get('step',50000),tm.get('step',50000)),
                runtime_units=units,evaluations=tests,record_video=False,third_view=False)
            e.write_json(out/'status.json',value)
            with (out/'supervision.jsonl').open('a') as f:f.write(json.dumps(dict(time=value['time'],stage=stage,
                new_extra50=value['newly_added_segmented'],formal_step=value['formal_step']),ensure_ascii=False)+'\n')
            if time.monotonic()-last_report>=600 or stage!=previous_stage:report(value);last_report=time.monotonic()
            previous_stage=stage
            if stage=='completed':
                e.write_json(out/'final_receipt.json',dict(status='completed',conditional_triggered=True,
                    new_extra_DR_episodes=50,total_corrections=120,additional_optimizer_updates=30000,final_step=80000,
                    tests_each=5,milestones={s:dict(successes=r['full_successes'],tests=5,success_rate=r['success_rate']) for s,r in tests.items()},
                    failure_analysis=str(out/'failure_analysis_70k80k.json'),record_video=False,third_view=False))
                return 0
            time.sleep(30)
    except BaseException as exc:
        value.update(status='halted',stage='halted',time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            error=f'{type(exc).__name__}: {exc}',all_original_evidence_preserved=True)
        e.write_json(out/'status.json',value);report(value);raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
        if child_log is not None:child_log.close()
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    args=p.parse_args();raise SystemExit(execute(e.read_json(args.plan)))
