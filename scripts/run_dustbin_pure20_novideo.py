#!/usr/bin/env python3
"""Uniform no-video version of the original fixed20 assignment; preserve prior scores."""
import argparse
import csv
import io
from pathlib import Path
import sys
import threading
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_dustbin_pure20 as b
from scripts import run_robotwin_checkpoint_queue as queue
from scripts.run_robotwin_posttrain_eval_pipeline import training_gate
from harness.robotwin_policy_worker import atomic_json,identity,sha256,training_contract

PRIOR=ROOT/'outputs/runs/put_bottles_dustbin/pure25k_fixed20_offscreen_repair_20261007'
OUT=ROOT/'outputs/runs/put_bottles_dustbin/pure25k_fixed20_novideo_20261007'


def build():
    previous=b.read(PRIOR/'manifest.json');state=b.read(PRIOR/'queue_status.json')
    if previous['seeds']!=b.SEEDS or state.get('status')!='stopped' or not (PRIOR/'completion.json').exists():
        raise ValueError('previous suite must be stopped and archived')
    ready,gate=training_gate(b.TRAIN,b.TRAIN_UNIT)
    if not ready:raise ValueError('training/GPU gate not ready: '+str(gate))
    contract,_,_,receipt=training_contract(b.TRAIN,b.CKPT)
    admitted=b.read(PRIOR/'training_admission.json')
    if receipt!=previous['checkpoint_receipt'] or sha256(b.CKPT)!=receipt['sha256']:
        raise ValueError('original frozen checkpoint differs')
    if contract['config']['dataset']['task_name']!=b.TASK:raise ValueError('wrong task')
    jobs=[dict(j,unit=f"p2-dustbin-pure20-novideo-{j['seed']}-20261007",
               run_dir=str(OUT/f"seed_{j['seed']}"),evaluator_entry='scripts/eval_robotwin_novideo.py')
          for j in previous['jobs']]
    sources=[Path(p) for p in previous['source_sha256']]
    sources += [Path(__file__),ROOT/'scripts/eval_robotwin_novideo.py']
    manifest=dict(previous,protocol='dustbin_pure25k_fixed20_novideo_v3',created_at=b.now(),
        jobs=jobs,require_video=False,record_video=False,
        source_sha256={str(p):sha256(p) for p in sources},
        protocol_deviation='User explicitly disabled video on 2026-10-07. No third camera/render/ffmpeg. All original twenty seeds run uniformly without video. Previous four scored attempts remain archived separately; never choose a better score or pool duplicate seeds across recording versions. Model/RGB/state/action/physics/native score and step/wall budgets retained.',
        previous_suite=identity(PRIOR/'manifest.json'),previous_results=identity(PRIOR/'summary.json'),
        attempt_selection='first valid terminal per seed in this no-video protocol; duplicates rejected, valid failures never replayed',
        scoring='Native benchmark predicate. Actions and onboard input records required, video explicitly disabled; strict physical predicates remain unverified.',
        baseline_freeze_scope='Prior runtime sources plus no-video entry and runner. All video-version results remain intact.')
    queue.validate_manifest(manifest)
    return manifest,admitted


def verify_episode(job,attempt,manifest):
    import json
    path=Path(attempt['run_dir']);result=b.read(path/'result.json');ep=result['episodes'][0]
    plan=b.read(path/'plan.json');directory=path/ep['episode']
    records=[json.loads(l) for l in (directory/'actions.jsonl').read_text().splitlines()]
    sent=[a for a in records if a['status']=='submitted'];returned=[a for a in records if a['status']=='native_returned']
    onboard=b.read(directory/'onboard/manifest.json');recording=b.read(directory/'recording.json')
    checks=dict(task=result['task']==b.TASK,seed=ep['seed']==job['seed'],
        checkpoint=plan['checkpoint_receipt']['sha256']==manifest['checkpoint_receipt']['sha256'],
        pure=plan['system']['mode']=='pure' and plan['system']['memory_writes_enabled'] is False,
        native8=plan['system']['execution_mode']=='native8',
        actions=len(sent)==len(returned)==ep['control_steps']==ep['native_take_action_count'] and len(records)==2*len(sent),
        sequential_ids=all(a['step']==i and a['episode']==ep['episode'] for rows in (sent,returned) for i,a in enumerate(rows)),
        ack_commands=all(a['bounded_action']==r['bounded_action'] and a['step']==r['step'] for a,r in zip(sent,returned)),
        zero_condition=all(a['policy']['task_condition']==dict(source='zero_single_task',sha256=b.ZERO_SHA) for a in records),
        no_recovery=all(a['policy']['execution_prefix']==8 and a['policy']['execution_reason']=='native_exec8' for a in records),
        onboard=onboard.get('complete') is True and onboard['frames']==ep['control_steps']+1,
        recording_off=plan.get('recording',{}).get('enabled') is False and recording.get('enabled') is False
          and recording.get('third_view_camera_created') is False and recording.get('video_files_created')==0
          and not list(directory.glob('*.mp4')) and not list(directory.glob('third_view*')))
    return dict(seed=job['seed'],passed=all(checks.values()),checks=checks,
                result=identity(path/'result.json'),actions=identity(directory/'actions.jsonl'))


def summarize(manifest):
    state=b.read(OUT/'queue_status.json');rows=[];verification=[]
    for job in manifest['jobs']:
        attempts=[a for a in state.get('jobs',[]) if a['seed']==job['seed']]
        valid=[a for a in attempts if a['status']=='valid_terminal']
        if len(valid)>1:raise ValueError('duplicate valid seed')
        attempt=valid[0] if valid else attempts[-1] if attempts else None
        result=b.read(Path(attempt['run_dir'])/'result.json') if attempt else {}
        ep=(result.get('episodes') or [{}])[0]
        row=dict(seed=job['seed'],status=attempt['status'] if attempt else 'not_started',
                 native_success=ep.get('evidence',{}).get('native_benchmark_success') if valid else None,
                 control_steps=ep.get('control_steps'),wall_seconds=ep.get('wall_seconds'),attempts=len(attempts))
        if valid:
            try:verification.append(verify_episode(job,attempt,manifest))
            except Exception as exc:verification.append(dict(seed=job['seed'],passed=False,error=str(exc)))
        rows.append(row)
    done=[r for r in rows if r['status']=='valid_terminal'];wins=sum(r['native_success'] is True for r in done)
    summary=dict(updated_at=b.now(),status=state.get('status','preregistered'),planned=20,scored=len(done),
        successes=wins,failures=len(done)-wins,valid_completed_20=len(done)==20,
        verified_complete=len(done)==20 and len(verification)==20 and all(v['passed'] for v in verification),
        native_success_rate=wins/20 if len(done)==20 else None,
        wilson_95=b.wilson(wins,20) if len(done)==20 else None,record_video=False,
        episodes=rows,error=state.get('error'),previous_suite_separate=str(PRIOR))
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'verification.json',dict(complete=summary['verified_complete'],episodes=verification))
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    b.atomic_text(OUT/'per_episode_results.csv',buffer.getvalue())
    lines=['# 25k纯WAM原定20种子：统一不录像版本','',f'更新时间：{summary["updated_at"]}；状态：{summary["status"]}。',
       f'有效完成{len(done)}/20，成功{wins}，失败{len(done)-wins}；最终成功率尚未完成。' if len(done)!=20 else f'原生成功率{wins}/20 = {wins/20:.0%}；逐动作核验全部通过：{summary["verified_complete"]}。',
       '', '用户要求默认不录像：不创建第三视角相机，不渲染录像，不启动ffmpeg；保留必要三RGB机载输入及动作日志。原视频版本已完成结果单独保留，不与本版合并或挑分数。固定原20种子、25k权重、native8、1700步与原生评分。',
       '', '|seed|状态|成功|动作数|墙钟秒|','|---|---|---|---:|---:|']
    lines += [f'|{r["seed"]}|{r["status"]}|{r["native_success"]}|{r["control_steps"]}|{r["wall_seconds"]}|' for r in rows]
    b.atomic_text(OUT/'report.md','\n'.join(lines)+'\n');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    b.bus_environment();manifest,admitted=build()
    if not args.execute:
        print(dict(protocol=manifest['protocol'],seeds=manifest['seeds'],video=False));return 0
    OUT.mkdir(parents=True,exist_ok=False);path=OUT/'manifest.json';atomic_json(path,manifest);atomic_json(OUT/'training_admission.json',admitted)
    stop=threading.Event();errors=[]
    def reporting():
        while not stop.wait(30):
            try:summarize(manifest)
            except Exception as exc:errors.append(str(exc))
    summarize(manifest);thread=threading.Thread(target=reporting,daemon=True);thread.start();code=1;error=None
    try:code=queue.run_manifest(manifest,path,before_job=lambda:b.verify_admission(manifest,admitted))
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        stop.set();thread.join(timeout=15);summary=summarize(manifest)
        changed=[p for p,h in manifest['source_sha256'].items() if sha256(p)!=h]
        if not summary['verified_complete'] or changed:code=1
        atomic_json(OUT/'completion.json',dict(exit_code=code,error=error,finished_at=b.now(),
            verified_complete=summary['verified_complete'] and not changed,source_changes=changed,reporting_errors=errors))
    return code


if __name__=='__main__':raise SystemExit(main())
