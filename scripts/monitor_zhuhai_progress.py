#!/usr/bin/env python3
"""Read-only runtime snapshots, published to existing progress docs every 10 minutes."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.progress1_log import log


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def unit_snapshot(text, marker):
    match=re.search(r'^<!-- '+re.escape(marker)+r': (.+) -->$',text,re.M)
    if not match:return None
    unit=match.group(1).strip()
    if not re.fullmatch(r'p2-[A-Za-z0-9@_.-]+',unit):raise ValueError('Invalid explicit user unit pointer')
    try:
        proc=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','Result'],
                            capture_output=True,text=True,timeout=5)
        if proc.returncode:return dict(unit=unit,error=proc.stderr.strip() or 'systemctl failed')
        return dict(unit=unit,**dict(line.split('=',1) for line in proc.stdout.splitlines() if '=' in line))
    except (OSError,subprocess.SubprocessError) as exc:return dict(unit=unit,error=str(exc))


def snapshot():
    text=(ROOT/'progress1.md').read_text()
    match=re.search(r'^<!-- active_training_run: (.+) -->$',text,re.M)
    if not match:raise ValueError('Explicit active_training_run pointer required')
    run=(ROOT/match.group(1)).resolve()
    if not run.is_relative_to(ROOT/'outputs'):raise ValueError('Only project outputs may be monitored')
    state=read(run/'status.json');plan=read(run/'plan.json');stage=state.get('active_stage')
    runtime_unit=unit_snapshot(text,'active_training_unit')
    if (state.get('status')=='running' and runtime_unit and not runtime_unit.get('error')
            and runtime_unit.get('ActiveState') not in ('active','activating','reloading')):
        state=dict(state,status='interrupted_unit_inactive',error='Preserved running snapshot but bound systemd unit is not active; inspect without automatic retry')

    # Original status bytes are evidence, not a liveness signal. User-authorized
    # external stops take precedence without rewriting the preserved run.
    external_stop=read(run/'external_stop_receipt.json')
    if external_stop:
        state=dict(state,status=external_stop.get('status','externally_stopped'),active_stage=None)
        stage=None
    stage_plan=next((s for s in plan.get('stages',[]) if s['name']==stage),None)
    detail=read(Path(stage_plan['out'])/'status.json') if stage_plan else {}
    if stage_plan and stage_plan.get('kind')=='eval':detail=read(Path(stage_plan['out'])/'summary.json')
    if plan.get('monitor_mode')=='final_evaluation' and detail.get('status')in('completed','halted_infrastructure_error'):
        finished=detail['status']=='completed'
        state=dict(status='completed' if finished else 'failed',active_stage=None if finished else stage,
                   completed_stages=[stage] if finished else [],error=detail.get('error'))
        if finished:stage=None

    artifact_run=Path(plan.get('training_origin',str(run))).resolve()
    if not artifact_run.is_relative_to(ROOT/'outputs'):raise ValueError('Training origin must stay below project outputs')
    train=read(artifact_run/'train/status.json');tiny=read(artifact_run/'tiny_admission/status.json')
    if plan.get('monitor_mode')=='direct_training':
        detail=train
        finished=train.get('status')=='completed' and train.get('step')==train.get('target_steps')==30000
        inactive=runtime_unit and not runtime_unit.get('error') and runtime_unit.get('ActiveState') not in ('active','activating','reloading')
        direct_status='completed' if finished else ('failed' if train.get('status')=='failed' else 'interrupted_unit_inactive' if inactive else 'running')
        stage=None if finished else 'train_to_30000'
        state=dict(status=direct_status,active_stage=stage,completed_stages=['train_to_30000'] if finished else [],error=train.get('error'))

    # A resume preparation may temporarily report step zero; a saved optimizer
    # boundary remains completed work and must not disappear from the handoff.
    checkpoint_receipt=read(artifact_run/'train/checkpoint_last.pt.sha256.json')
    checkpoint=artifact_run/'train/checkpoint_last.pt';saved_step=0
    if checkpoint.is_file() and checkpoint_receipt.get('size')==checkpoint.stat().st_size:
        saved_step=checkpoint_receipt.get('step',0)
        if type(saved_step) is not int or saved_step<0:saved_step=0
    formal_step=max(train.get('step',0),saved_step)
    metrics={};logfile=artifact_run/'train/log.jsonl'
    if logfile.is_file():
        with logfile.open('rb') as f:
            f.seek(max(0,logfile.stat().st_size-65536));lines=f.read().splitlines()
        if lines:
            try:metrics=json.loads(lines[-1])
            except (ValueError,UnicodeError):pass
    timing=dict(last_loss=metrics.get('loss'),train_elapsed_seconds=metrics.get('elapsed_seconds'),
        cuda_peak_reserved_bytes=metrics.get('cuda_peak_reserved_bytes'))
    # No ETA from demonstrations or preprocessing. Extrapolate only observed
    # optimizer throughput; excluding simulator time is stated explicitly.
    if metrics.get('step',0)>0 and metrics.get('elapsed_seconds',0)>0:
        parsed=[]
        for raw in lines:
            try:
                row=json.loads(raw)
                if 'step' in row and 'elapsed_seconds' in row:parsed.append(row)
            except (ValueError,UnicodeError):continue
        window=[]
        for row in reversed(parsed):
            if window and (row['step']>=window[-1]['step'] or row['elapsed_seconds']>=window[-1]['elapsed_seconds']):break
            window.append(row)
        if len(window)>=10:
            newest,oldest=window[0],window[-1]
            speed=(newest['elapsed_seconds']-oldest['elapsed_seconds'])/(newest['step']-oldest['step'])
            timing.update(recent_seconds_per_update=speed,recent_window_steps=[oldest['step'],newest['step']],
                estimated_remaining_optimizer_hours=max(0,30000-formal_step)*speed/3600,
                estimate_excludes='future physical eval, process/cache cold starts and future checkpoint costs')
    results=[]
    for planned in plan.get('stages',[]):
        if planned.get('kind')!='eval':continue
        report=read(Path(planned['out'])/'summary.json');rows=report.get('episodes',[])
        statuses=('completed','control_safety_abort') if report.get('schema')=='precision-v2-fixed-attempt-summary-v1' else ('completed',)
        if (report.get('status')=='completed' and report.get('checkpoint_step')==planned['step']
                and report.get('planned_episodes')==len(planned['seeds'])
                and [r.get('seed') for r in rows]==planned['seeds']
                and all(r.get('status')in statuses and type(r.get('full_task_success')) is bool and (r.get('status')!='control_safety_abort' or r.get('full_task_success')is False) for r in rows)):
            n=len(rows);k=sum(r['full_task_success'] for r in rows)
            results.append(dict(step=planned['step'],status='completed',successes=k,episodes=n,success_rate=k/n,safety_aborts=sum(r.get('status')=='control_safety_abort'for r in rows),
                                evidence=str(Path(planned['out'])/'summary.json')))

    first=next((s for s in plan.get('stages',[]) if s['kind']=='train'),None)
    frames=0;cache=ROOT/'data/zhuhai_hjepa_hybrid_features_20261009'
    if first:
        argv=first['command'];cache=Path(argv[argv.index('--cache-root')+1]).resolve()
        if not cache.is_relative_to(ROOT/'data'):raise ValueError('Cache path outside project data')
    if cache.exists():frames=sum(e.name.endswith('.payload') for e in os.scandir(cache))
    probe_match=re.search(r'^<!-- active_probe_run: (.+) -->$',text,re.M)
    probe_state=None
    if probe_match:
        probe_run=(ROOT/probe_match.group(1)).resolve()
        if not probe_run.is_relative_to(ROOT/'outputs'):raise ValueError('Only project probe outputs may be monitored')
        probe_summary=read(probe_run/'queue_summary.json')
        probe_rows=[read(path) for path in probe_run.glob('seed_*/manifest.json')]
        probe_state=dict(run=str(probe_run),status=probe_summary.get('status','running' if probe_run.exists() else 'not_started'),
            strict_successful_cases=sum(row.get('success_count',0) for row in probe_rows),
            started_cases=len(probe_rows),required_cases=4,production_quota_credit=0)
    collection=None
    collection_match=re.search(r'^<!-- active_collection_run: (.+) -->$',text,re.M)
    if collection_match:
        collection_run=(ROOT/collection_match.group(1)).resolve()
        if not collection_run.is_relative_to(ROOT/'outputs'):raise ValueError('Collection pointer must stay below outputs')
        queue=read(collection_run.with_name(collection_run.name+'.queue')/'status.json')
        production=read(collection_run/'status.json')
        collection_unit=unit_snapshot(text,'active_collection_unit')
        if (queue.get('status') in ('waiting_for_entire_pipeline','running_serial_collection')
                and collection_unit and not collection_unit.get('error')
                and collection_unit.get('ActiveState') not in ('active','activating','reloading')):
            queue=dict(queue,status='interrupted_unit_inactive',error='Preserved queue snapshot but bound unit is not active; no automatic retry')

        plan_for_collection=read(collection_run.with_name(collection_run.name+'.queue')/'plan.json')
        prior=plan_for_collection.get('collection',{}).get('preserved_dr',48)
        manifests=list((collection_run/'collection/zhuhai_task1/randomized/shards').glob('shard_*/manifest.json'))
        fresh=sum(read(p).get('success_count',0) for p in manifests)
        aggregate=read(collection_run/'collection/zhuhai_task1/randomized/manifest.json')
        collection=dict(run=str(collection_run),queue_status=queue.get('status','not_started'),
            production_status=production.get('status','not_started'),production_started=queue.get('production_started',False),
            preserved_dr=prior,new_dr=fresh,total_dr=aggregate.get('success_count',prior+fresh),target_dr=200,
            probe_quota_credit=0,mode='serial_after_full_training_and_evaluation',error=queue.get('error') or production.get('error'))
        pause=read(collection_run.with_name(collection_run.name+'.queue')/'user_test_priority_pause.json')
        if pause and collection_unit and not collection_unit.get('error') and collection_unit.get('ActiveState') not in ('active','activating','reloading'):
            collection.update(queue_status='paused_by_user_test_priority',mode='idle_only_after_tests',
                user_pause_receipt=pause,raw_queue_status=queue.get('status'),error=None)
    evaluation=None
    evaluation_match=re.search(r'^<!-- active_execution_evaluation_run: (.+) -->$',text,re.M)
    if evaluation_match:
        evaluation_run=(ROOT/evaluation_match.group(1)).resolve()
        if not evaluation_run.is_relative_to(ROOT/'outputs'):raise ValueError('Evaluation pointer must stay below outputs')
        report=read(evaluation_run/'summary.json')
        evaluation=dict(run=str(evaluation_run),status=report.get('status','not_started'),
            evaluation_version=report.get('evaluation_version'),finished_episodes=len(report.get('episodes',[])),
            planned_episodes=report.get('planned_episodes'),full_successes=report.get('full_successes'),
            success_rate=report.get('success_rate'),error=report.get('error'),
            runtime_unit=unit_snapshot(text,'active_execution_evaluation_unit'))
    new_work={}
    for key,marker,filename in [('train_diagnostic','active_train_diagnostic_run','summary.json'),
                                ('dagger','active_dagger_collection_run','manifest.json'),
                                ('dagger_training','active_dagger_training_run','status.json')]:
        pointer=re.search(r'^<!-- '+marker+r': (.+) -->$',text,re.M)
        if pointer:
            path=(ROOT/pointer.group(1)).resolve()
            if not path.is_relative_to(ROOT/'outputs'):raise ValueError('Task pointer must stay below outputs')
            report=read(path/filename)
            if key=='dagger_training' and (path/'train/status.json').is_file():
                report=read(path/'train/status.json')
            new_work[key]=dict(run=str(path),status=report.get('status','not_started'),
                finished=(len(report.get('episodes',[])) if key=='train_diagnostic' else max(0,report.get('step',30000)-30000) if key=='dagger_training' else report.get('success_count',0)),
                planned=10 if key=='train_diagnostic' else 20000 if key=='dagger_training' else 50,error=report.get('error'),
                runtime_unit=unit_snapshot(text,'active_train_diagnostic_unit' if key=='train_diagnostic' else 'active_dagger_training_unit' if key=='dagger_training' else 'active_dagger_collection_unit'))
    return dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),run=str(run),new_work=new_work,
        pipeline_status=state.get('status','not_started'),active_stage=stage,stage_status=detail.get('status','pending'),
        formal_step=formal_step,formal_target=30000,tiny_step=tiny.get('step',0),tiny_status=tiny.get('status','not_started'),derived_cache_frames=frames,
        completed_stages=state.get('completed_stages',[]),error=state.get('error') or detail.get('error'),
        probe=probe_state,collection=collection,execution_evaluation=evaluation,training_artifact_run=str(artifact_run),runtime_unit=runtime_unit,external_stop=external_stop or None,training_metrics=timing,milestone_results=results,
        source='external-stop receipt first, status/plan/checkpoint receipts, isolated probes and derived cache inventory; no GPU context')


def publish(value):
    message=(f"队列{value['pipeline_status']}/{value['active_stage']}；阶段{value['stage_status']}；"
        f"tiny {value['tiny_step']}/2（{value['tiny_status']}）；正式优化{value['formal_step']}/30000；派生特征缓存{value['derived_cache_frames']}帧（非新增示教）；"
        f"完成阶段{','.join(value['completed_stages']) or '无'}；错误{value['error'] or '无'}；"
        "餐具先训，垃圾200DR准入待完成，并发未准入不启动采集。")
    collection=value.get('collection')
    if collection:
        message=message.replace('餐具先训，垃圾200DR准入待完成，并发未准入不启动采集。',
            f"垃圾串行队列{collection['queue_status']}，DR {collection['total_dr']}/200（保留{collection['preserved_dr']}+新增{collection['new_dr']}）；测试优先、空闲补采，不并行。")
    evaluation=value.get('execution_evaluation')
    if evaluation:
        message+=f" 官方Panda限制+提前制动独立五测{evaluation['status']}，已核验{evaluation['finished_episodes']}/{evaluation['planned_episodes']}，完整成功{evaluation['full_successes']}（整组率{evaluation['success_rate']}）；原30k成绩另存不覆盖。"
    for key,value_new in value.get('new_work',{}).items():
        label='TRAIN布局闭环诊断（非泛化）' if key=='train_diagnostic' else 'DAgger追加optimizer更新' if key=='dagger_training' else '合格DAgger专家纠正'
        message+=f" {label}{value_new['status']} {value_new['finished']}/{value_new['planned']}；错误{value_new['error'] or '无'}。"
    results=[r for r in value.get('milestone_results',[]) if r.get('status')=='completed']
    if results:message+=' 固定种子物理五测：'+ '、'.join(f"{r['step']}步={r['successes']}/{r['episodes']}" for r in results)+'（非独立泛化估计）。'
    if value.get('external_stop'):message+=' 外部停止回执优先，旧running末快照不是活动训练。'
    probe=value.get('probe')
    if probe:message+=f" 垃圾专家probe {probe['status']}，严格通过{probe['strict_successful_cases']}/4，已启动{probe['started_cases']}组，生产配额计数0。"
    changed=log(message,actor='10min_monitor')
    if not changed:return False
    p=ROOT/'docs/current_status.md'
    with p.open('r+',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);text=f.read()
        block=f"<!-- live-10min:start -->\n**10分钟运行快照 {value['time']}：** {message}\n<!-- live-10min:end -->\n\n"
        if '<!-- live-10min:start -->' in text:
            text=re.sub(r'<!-- live-10min:start -->.*?<!-- live-10min:end -->\n*',lambda _:block,text,count=1,flags=re.S)
        else:text=block+text
        if len(text.splitlines())>=200:raise ValueError('current_status needs summary compression, not truncation')
        f.seek(0);f.write(text);f.truncate()
    with (ROOT/'docs/progress_archive/progress1_runtime_snapshots.jsonl').open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        value=snapshot();published=publish(value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
        collection_pending=value.get('collection',{}).get('queue_status') in ('waiting_for_entire_pipeline','running_serial_collection') if value.get('collection') else False
        evaluation_pending=value.get('execution_evaluation',{}).get('status') in ('waiting_for_exclusive_gpu','evaluating') if value.get('execution_evaluation') else False
        new_pending=any(v['status'] in ('waiting_for_train_diagnostic','waiting_for_exclusive_gpu','evaluating','validating_first10','collecting','waiting_for_collection','auditing_fifty_corrections','training_child_running','loading_30k_full_state','training','not_started','parallel_collecting','draining_serial_collection','waiting_for_resources','draining_guard') for v in value.get('new_work',{}).values())
        if not args.watch or (published and not collection_pending and not evaluation_pending and not new_pending and value['pipeline_status'] in ('completed','failed','stopped_by_user_requested_v2_replacement','stopped_by_authorized_model_upgrade','interrupted_unit_inactive')):return 0
        time.sleep(600)

if __name__=='__main__':raise SystemExit(main())
