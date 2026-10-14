#!/usr/bin/env python3
"""Keep current trainer frozen; test retained 45k/50k models, then conditional recovery."""
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
from scripts.supervise_precision_dagger50k_v12 import read,unit_state,verify,next_stage
from scripts.eval_precision_dagger_milestone_fixed10_v13 import SEEDS,metadata_milestone,training_finished
from scripts.progress1_log import log


def complete_cohort(row,step):
    episodes=row.get('episodes',[])
    return (row.get('status')=='completed' and row.get('checkpoint',{}).get('step')==step
        and row.get('planned_episodes')==10 and row.get('seeds')==list(SEEDS)
        and [r['seed'] for r in episodes]==list(SEEDS)
        and all(type(r.get('full_task_success')) is bool for r in episodes)
        and row.get('full_successes')==sum(r['full_task_success'] for r in episodes)
        and row.get('success_rate')==sum(r['full_task_success'] for r in episodes)/10)


def conditional_recovery_required(cohorts):
    e.require(all(complete_cohort(cohorts[str(step)],step) for step in (45000,50000)),
              'Both complete fixed ten-test cohorts required before the recovery decision')
    return cohorts['50000']['full_successes']==0


def report(value):
    tests=value['evaluations'];child=value['training_child']
    message=(f"持续监督阶段{value['stage']}；恢复50新＋20旧已齐；正式{value['formal_step']}/50000；"
        +'; '.join(f"{step}测试{len(tests[str(step)].get('episodes',[]))}/10、完整成功{tests[str(step)].get('full_successes',0)}" for step in (45000,50000))
        +f"；异常{value.get('error') or child.get('error') or '无'}。")
    log(message,'dagger_milestone_supervisor')
    path=ROOT/'docs/current_status.md'
    with path.open('r+',encoding='utf-8') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);text=stream.read()
        block='<!-- live-10min:start -->\n**实时监督 '+value['time']+'：** '+message+'\n<!-- live-10min:end -->'
        text=re.sub(r'<!-- live-10min:start -->.*?<!-- live-10min:end -->',lambda _:block,text,flags=re.S)
        e.require(len(text.splitlines())<200,'Existing current status must stay under 200 lines')
        stream.seek(0);stream.write(text);stream.truncate()


def execute(plan):
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=False);e.write_json(out/'plan.json',plan)
    lock=(out/'supervisor.lock').open('xb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None;child_log=None;active_step=None;started=time.monotonic();last_report=-1e9;value={};previous_stage=None
    try:
        while True:
            verify(plan);e.require(time.monotonic()-started<plan['maximum_seconds'],'Supervision deadline exceeded')
            cm=read(Path(plan['collection'])/'manifest.json');tm=read(Path(plan['training'])/'status.json')
            child_status=read(Path(plan['training'])/'train/status.json')
            units=[unit_state(plan[k]) for k in ('collection_unit','training_unit')]
            e.require(child_status.get('status')!='failed','Training child failed: '+str(child_status.get('error')))
            for row,state in zip(units,(cm,tm)):
                if row['MainPID']=='0':
                    e.require(state.get('status') in ('complete','completed','halted','failed'),
                        'Registered process disappeared without terminal receipt: '+row['unit'])
            if child_status.get('status')=='training':
                age=time.time()-(Path(plan['training'])/'train/status.json').stat().st_mtime
                e.require(age<600,'Training heartbeat stale for ten minutes; retain process for diagnosis')
            cohorts={str(step):read(Path(plan['evaluation_outs'][str(step)])/'summary.json') for step in (45000,50000)}
            stage=next_stage(cm,tm,units)
            if child is not None:
                stage=f'evaluating_{active_step}'
                code=child.poll()
                if code is not None:
                    e.require(code==0 and complete_cohort(cohorts[str(active_step)],active_step),
                        f'{active_step} cohort incomplete; retain every original attempt: '+str(cohorts[str(active_step)].get('error')))
                    child_log.close();child=None;child_log=None;active_step=None
            ready=stage=='final_evaluation_ready' or stage.startswith('evaluating_')
            if ready and child is None:
                remaining=[s for s in (45000,50000) if not complete_cohort(cohorts[str(s)],s)]
                if remaining:
                    step=remaining[0];intent=out/f'evaluation_command_{step}.json'
                    e.require(not intent.exists() and not Path(plan['evaluation_outs'][str(step)]).exists(),
                        'Repeated cohort launch prohibited; inspect prior failure')
                    command=plan['evaluation_commands'][str(step)];e.write_json(intent,dict(command=command))
                    child_log=(out/f'evaluation_{step}.log').open('xb')
                    child=subprocess.Popen(command,cwd=ROOT,stdout=child_log,stderr=subprocess.STDOUT,start_new_session=True)
                    active_step=step;stage=f'evaluating_{step}'
                    e.write_json(out/f'evaluation_participant_{step}.json',dict(pid=child.pid,pgid=child.pid,command=command))
                elif conditional_recovery_required(cohorts):
                    stage='conditional_recovery_pending_analysis'
                    trigger=out/'conditional_recovery_required.json'
                    if not trigger.exists():
                        e.write_json(trigger,dict(required=True,trigger='50000 complete-task success 0/10',
                            evaluation_summaries={s:str(Path(p)/'summary.json') for s,p in plan['evaluation_outs'].items()},
                            new_domain_randomized_segmented_episodes=50,additional_updates=30000,target_step=80000,
                            test_milestones=[70000,80000],tests_each=5,existing_data_retained=True,
                            status='failure analysis and targeted collection must be provisioned before launch'))
                        log('50k十测完整成功0/10，触发失败分析→50新域随机DAgger→追加30k→70k/80k各五测分支。','dagger_recovery_trigger')
                else:stage='completed'
            value=dict(schema=plan['schema'],status='completed' if stage=='completed' else 'running',
                time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),stage=stage,
                formal_step=max(30000,tm.get('step',30000),child_status.get('step',30000)),
                collection=cm,training=tm,training_child=child_status,runtime_units=units,evaluations=cohorts,
                conditional_policy=plan['conditional_policy'],record_video=False,third_view=False)
            milestone=Path(plan['training'])/'train/checkpoint_step_045000.pt'
            if milestone.exists() and milestone.with_name(milestone.name+'.sha256.json').exists() and not (out/'milestone_45000_saved.json').exists():
                meta,_=metadata_milestone(milestone,45000);e.write_json(out/'milestone_45000_saved.json',meta)
            e.write_json(out/'status.json',value)
            with (out/'supervision.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=value['time'],stage=stage,formal_step=value['formal_step'],
                    evaluations={s:len(r.get('episodes',[])) for s,r in cohorts.items()}),ensure_ascii=False)+'\n')
            if time.monotonic()-last_report>=600 or stage!=previous_stage:
                report(value);last_report=time.monotonic()
            previous_stage=stage
            if stage=='completed':
                e.write_json(out/'final_receipt.json',dict(status='completed',formal_step=50000,additional_updates=20000,
                    new_segmented=50,legacy=20,evaluation_attempts=20,
                    results={s:dict(full_successes=r['full_successes'],planned_episodes=10,success_rate=r['success_rate']) for s,r in cohorts.items()},
                    conditional_recovery_required=False,record_video=False,third_view=False))
                return 0
            time.sleep(30)
    except BaseException as exc:
        value.update(status='halted',stage='halted',time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            error=f'{type(exc).__name__}: {exc}',all_original_evidence_preserved=True)
        e.write_json(out/'status.json',value)
        if 'evaluations' in value:report(value)
        log('里程碑监督暂停：'+value['error']+'；保留所有权重/失败/partial，不重试既有测试。','dagger_milestone_supervisor')
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
        if child_log is not None:child_log.close()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    args=parser.parse_args();raise SystemExit(execute(e.read_json(args.plan)))
