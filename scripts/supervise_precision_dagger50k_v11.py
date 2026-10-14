#!/usr/bin/env python3
"""Durable supervision of collection, strict continuation, then final-only test."""
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
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts.eval_precision_dagger50k_fixed5_v9 import training_finished
from harness.zhuhai_dagger_segmented_training_v7 import validate_quota
from harness.zhuhai_dagger_parallel import snapshot
from scripts.progress1_log import log


def read(path):
    path=Path(path)
    return e.read_json(path) if path.exists() else {}


def unit_state(unit):
    result=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState','-p','SubState'],
                          text=True,capture_output=True,check=True,timeout=5)
    return dict(unit=unit,**dict(line.split('=',1) for line in result.stdout.splitlines()))


def next_stage(collection,training,units):
    if collection.get('status')=='halted' or training.get('status') in ('halted','failed'):
        raise ValueError('Pipeline stage halted; preserve evidence/guard: '+str(collection.get('error') or training.get('error')))
    if collection.get('status')!='complete':
        return 'collecting'
    validate_quota(collection)
    if not training_finished(training):
        return 'training'
    if any(row['MainPID']!='0' or row['ActiveState'] not in ('inactive','failed') for row in units):
        return 'waiting_for_training_process_exit'
    return 'final_evaluation_ready'


def verify(plan):
    for path,sha in plan['source_files'].items():
        e.require(e.sha256_file(path)==sha,'Supervisor-bound source drift: '+path)
    for path,sha in plan['bound_inputs'].items():
        e.require(e.sha256_file(path)==sha,'Supervisor-bound input drift: '+path)


def publish_documents(value):
    cm=value['collection'];tm=value['training'];final=value.get('evaluation',{})
    actual_workers=sum(Path('/proc',str(row['pid'])).exists() for row in cm.get('active_participants',[]))
    message=(f"无人值守阶段{value['stage']}，新分段{cm.get('success_count',0)}/50、旧完整{cm.get('legacy_corrections',0)}/20；"
             f"实际{actual_workers}路/目标最多4；正式步骤{value['formal_step']}/50000；"
             + (f"最终固定五测已测{len(final['episodes'])}/5、完整成功{final.get('full_successes',0)}；" if final.get('episodes') else '最终固定五测未开始；')
             + f"错误{value.get('error') or cm.get('error') or tm.get('error') or '无'}。")
    log(message,'dagger50k_supervisor')
    path=ROOT/'docs/current_status.md'
    with path.open('r+',encoding='utf-8') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);text=stream.read()
        block='<!-- live-10min:start -->\n**实时监督 '+value['time']+'：** '+message+'\n<!-- live-10min:end -->'
        text=re.sub(r'<!-- live-10min:start -->.*?<!-- live-10min:end -->',lambda _:block,text,flags=re.S)
        e.require(len(text.splitlines())<200,'Existing current_status must remain below 200 lines')
        stream.seek(0);stream.write(text);stream.truncate()
    with (ROOT/'docs/progress_archive/progress1_runtime_snapshots.jsonl').open('a') as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+'\n')


def execute(plan):
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=False);e.write_json(out/'plan.json',plan)
    started=time.monotonic();last_publish=-1e9;child=None;child_log=None;value={}
    lock=(out/'supervisor.lock').open('xb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        verify(plan)
        while True:
            e.require(time.monotonic()-started<plan['maximum_seconds'],'Bounded unattended supervision deadline exceeded')
            verify(plan)
            cm=read(Path(plan['collection'])/'manifest.json');tm=read(Path(plan['training'])/'status.json')
            units=[unit_state(plan[key]) for key in ('collection_unit','training_unit')]
            e.require(cm and tm,'Registered collection/training status must exist')
            child_train=read(Path(plan['training'])/'train/status.json')
            if child_train.get('status')=='failed':
                raise ValueError('Training child failed; preserve checkpoint: '+str(child_train.get('error')))
            for row,state in zip(units,(cm,tm)):
                if row['MainPID']=='0' and row['ActiveState'] in ('inactive','failed'):
                    e.require(state.get('status') in ('complete','completed','halted','failed'),
                              'Registered stage process disappeared before a terminal receipt: '+row['unit'])
            value=dict(schema=plan['schema'],collection=cm,training=tm,training_child=child_train,
                evaluation=read(Path(plan['evaluation_out'])/'summary.json'),runtime_units=units,
                formal_step=max(30000,tm.get('step',30000),child_train.get('step',30000)),resources=snapshot())
            stage=next_stage(cm,tm,units)
            final=read(Path(plan['evaluation_out'])/'summary.json')
            if child is not None:
                stage='evaluating'
                code=child.poll()
                if code is not None:
                    e.require(code==0 and final.get('status')=='completed' and len(final.get('episodes',[]))==5,
                              'Final evaluation did not finish the complete cohort; retain every attempt')
                    stage='completed'
            value=dict(schema=plan['schema'],status='completed' if stage=='completed' else 'running',
                time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),stage=stage,
                collection=cm,training=tm,training_child=child_train,evaluation=final,
                formal_step=max(30000,tm.get('step',30000),child_train.get('step',30000)),
                runtime_units=units,resources=snapshot(),maximum_workers=4,
                record_video=False,third_view=False,intermediate_physical_evaluations=False)
            e.write_json(out/'status.json',value)
            with (out/'supervision.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=value['time'],stage=stage,
                    new_segmented=cm['success_count'],active_workers=len(cm['active_participants']),
                    formal_step=value['formal_step'],resources=value['resources']),ensure_ascii=False)+'\n')
            if time.monotonic()-last_publish>=600 or stage=='completed':
                publish_documents(value);last_publish=time.monotonic()
            if stage=='completed':
                e.write_json(out/'final_receipt.json',dict(status='completed',new_segmented=50,legacy=20,
                    optimizer_updates=20000,final_step=50000,final_test_attempts=5,
                    full_successes=final['full_successes'],success_rate=final['success_rate'],
                    baseline=final['baseline'],final_summary=str(Path(plan['evaluation_out'])/'summary.json'),
                    no_intermediate_physical_tests=True,record_video=False,third_view=False))
                return 0
            if stage=='final_evaluation_ready' and child is None:
                e.require(not (out/'evaluation_command.json').exists(),'Final evaluation launch intent already exists; no retry')
                e.require(not Path(plan['evaluation_out']).exists(),'Final evaluation output already exists; no retry')
                e.write_json(out/'evaluation_command.json',dict(command=plan['evaluation_command']))
                child_log=(out/'final_evaluation.log').open('xb')
                child=subprocess.Popen(plan['evaluation_command'],cwd=ROOT,stdout=child_log,
                                       stderr=subprocess.STDOUT,start_new_session=True)
                e.write_json(out/'evaluation_participant.json',dict(pid=child.pid,pgid=child.pid,
                    command=plan['evaluation_command'],training_unit_exited=True,collection_unit_exited=True))
            time.sleep(30)
    except BaseException as exc:
        if 'collection' in value:
            value['collection']=read(Path(plan['collection'])/'manifest.json')
            value['training']=read(Path(plan['training'])/'status.json')
            value['runtime_units']=[unit_state(plan[key]) for key in ('collection_unit','training_unit')]
        value.update(status='halted',stage='halted',time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                     error=f'{type(exc).__name__}: {exc}',all_evidence_and_guards_preserved=True)
        e.write_json(out/'status.json',value)
        if 'collection' in value:publish_documents(value)
        log('无人值守监督暂停：'+value['error']+'；原证据/guard保留，不自动重试失败采集或评测。','dagger50k_supervisor')
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
