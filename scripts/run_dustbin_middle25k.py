#!/usr/bin/env python3
"""Existing measured middle/lower system vs completed pure WAM; frozen 25k, paired5.
No training. No VTT or full-system certification. Default plan-only has no writes.
"""
from pathlib import Path
import argparse,collections,datetime,fcntl,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness.robotwin_policy_worker import atomic_json,identity,sha256,training_contract
from scripts import run_robotwin_checkpoint_queue as queue
from scripts.gpu_lock import owner
TRAIN=ROOT/'outputs/runs/put_bottles_dustbin/wam_30k_20261007/train'
CKPT=TRAIN/'checkpoint_step_025000.pt'
BASE=ROOT/'outputs/runs/put_bottles_dustbin/pure_25k_30k_5each_20261007'
SMOKE=ROOT/'outputs/verification/dustbin_middle25k_20261007/smoke2_seed920010'
SEEDS=[920001,920002,920003,920004,920005]
TASK='put_bottles_dustbin'
def read(p):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return {}
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def check_admission():
    train=read(TRAIN/'status.json')
    if train.get('status')!='bounded_complete' or train.get('step')!=30000:raise ValueError('original training must be complete')
    contract,cache,stats,receipt=training_contract(TRAIN,CKPT)
    if (contract['config']['dataset']['task_name']!=TASK or receipt['step']!=25000
            or sha256(CKPT)!=receipt['sha256']):raise ValueError('frozen task/25k checkpoint mismatch')
    smoke=read(SMOKE/'result.json');eps=smoke.get('episodes',[])
    if smoke.get('status')!='completed' or len(eps)!=1 or eps[0].get('control_steps')!=2:raise ValueError('native 2-action middle smoke incomplete')
    middle=eps[0].get('middle_system',{})
    if (middle.get('configuration')!='middle_measured_feedback_v1' or middle.get('task')!=TASK
            or middle.get('with_vtt') is not False or middle.get('checkpoint_sha256')!=receipt['sha256']):raise ValueError('middle smoke did not use requested configuration')
    smoke_unit=queue.unit_state('p2-dustbin-middle25k-smoke-20261007')
    if smoke_unit.get('MainPID')!='0' or smoke_unit.get('Result')!='success':raise ValueError('smoke process must exit cleanly')
    from scripts.eval_robotwin_wam import dataset_seed_inventory
    original_seeds,_=dataset_seed_inventory(cache)
    if set(SEEDS)&(original_seeds|{920000,920010}):raise ValueError('test overlap with source/smoke seed')
    if owner() is not None:raise ValueError('GPU lock still held: '+str(owner()))
    return receipt

def build_manifest(out,receipt):
    base_manifest=read(BASE/'manifest.json');base_jobs={j['seed']:j for j in base_manifest['jobs'] if j['checkpoint_step']==25000}
    jobs=[];baseline=[]
    for seed in SEEDS:
        b=base_jobs[seed]
        if (b['system_mode']!='pure' or b['max_steps']!=1700 or b['execution_mode']!='native8'
                or b['camera_readback']!='serial' or b['settle_steps']!=0):raise ValueError('unexpected historical baseline protocol')
        original=read(Path(b['run_dir'])/'result.json')
        if original.get('status')!='completed' or original.get('scored_episodes')!=1:raise ValueError('missing baseline valid result')
        j=dict(b,unit=f'p2-dustbin-middle25k-{seed}-20261007',run_dir=str(out/f'seed_{seed}'),system_mode='middle')
        jobs.append(j);baseline.append(identity(Path(b['run_dir'])/'result.json'))
    sources=[Path(__file__),ROOT/'scripts/eval_robotwin_wam.py',ROOT/'scripts/run_robotwin_checkpoint_queue.py',ROOT/'scripts/gpu_lock.py']
    sources+=sorted((ROOT/'harness').glob('*.py'))+sorted((ROOT/'third_party/RoboTwin/envs').rglob('*.py'))+sorted((ROOT/'third_party/RoboTwin/env_cfg/task_config').glob('*.yml'))
    sources+=[ROOT/'configs/fixed_bimanual/robotwin_dustbin_50_wam.yaml']
    return {'protocol':'dustbin_middle25k_paired5_v1','task':TASK,'seeds':SEEDS,'jobs':jobs,'require_video':True,'max_invalid_retries':1,
            'training_run_dir':str(TRAIN),'checkpoint_receipt':receipt,'baseline_results':baseline,'baseline_manifest':identity(BASE/'manifest.json'),
            'smoke_evidence':identity(SMOKE/'result.json'),'created_at':now(),
            'source_sha256':{str(p.resolve()):sha256(p) for p in sources},
            'comparison':'Same task/config/25k weights/seeds1700step/renderer/serialRGB/native8/actuator limits; add existing measured controller middle path only.',
            'enabled':['WAM25k','transaction execution','actual robot encoder feedback','41-sample motion window','at most2 measured-stall HOLD/clear/reobserve','episode-scoped controller-outcome memory'],
            'disabled':['VLM','ASK','VTT adapter/bank (no admitted adapter for this task/checkpoint)','full strict TaskSupervisor (no admitted task-specific perception/postconditions)','physical target grasp/containment verification','cross-episode task-success memory'],
            'limits':'Existing development controller-stall thresholds unchanged. Nonstationary task failure is not detected as stall. Scores are official native predicate only. Prior five seeds are paired comparison, not fresh generalization.'}

def summarize(out,manifest,state):
    rows=[]
    for j in manifest['jobs']:
        attempts=[a for a in state.get('jobs',[]) if a['seed']==j['seed']]
        valid=[a for a in attempts if a['status']=='valid_terminal']
        attempt=valid[-1] if valid else attempts[-1] if attempts else None
        result=read(Path(attempt['run_dir'])/'result.json') if attempt else {}
        ep=(result.get('episodes') or [{}])[0];m=ep.get('middle_system',{})
        baseline=read(BASE/'step_025000'/f'seed_{j["seed"]}'/'result.json');be=baseline['episodes'][0]
        row={'seed':j['seed'],'baseline_native_success':be['evidence']['native_benchmark_success'],'baseline_control_steps':be['control_steps'],
             'middle_status':attempt['status'] if attempt else 'not_started','middle_native_success':ep.get('evidence',{}).get('native_benchmark_success') if valid else None,'middle_control_steps':ep.get('control_steps'),
             'supervisor_report':m,'attempts':len(attempts),'source_result':str(Path(attempt['run_dir'])/'result.json') if attempt else None}
        if result:
            directory=Path(attempt['run_dir'])/ep['episode'];events=[]
            if (directory/'middle_events.jsonl').is_file():events=[json.loads(l) for l in (directory/'middle_events.jsonl').read_text().splitlines()]
            row['event_counts']=dict(collections.Counter(e['kind'] for e in events))
            row['intervention_steps']=[e['step'] for e in events if e['kind']=='supervisor_hold_clear_reobserve']
            row['motion_windows']=sum(e.get('motion_window') is not None for e in events if e['kind']=='robot_feedback')
            def actions(p):return [x for x in (json.loads(l) for l in p.read_text().splitlines()) if x['status']=='native_returned']
            a=actions(directory/'actions.jsonl');b=actions(BASE/'step_025000'/f'seed_{j["seed"]}'/be['episode']/'actions.jsonl')
            row['raw_actions_exact_same']=len(a)==len(b) and all(x['raw_action']==y['raw_action'] for x,y in zip(a,b))
            row['bounded_actions_exact_same']=len(a)==len(b) and all(x['bounded_action']==y['bounded_action'] for x,y in zip(a,b))
            row['first_raw_action_difference']=next((i for i,(x,y) in enumerate(zip(a,b)) if x['raw_action']!=y['raw_action']),None)
            row['zero_condition_all_actions']=all(x['policy']['task_condition']['source']=='zero_single_task' for x in a)
            row['video']=str(directory/'third_view.mp4')
        rows.append(row)
    valid=[r for r in rows if r['middle_status']=='valid_terminal'];success=sum(r['middle_native_success'] is True for r in valid)
    summary={'status':state['status'],'created_at':now(),'checkpoint_step':25000,'valid_episodes':len(valid),'native_successes':success,'native_success_rate':success/5 if len(valid)==5 else None,
             'pure_baseline':'4/5 on same five seeds and25k','rows':rows,'enabled':manifest['enabled'],'disabled':manifest['disabled'],'limitations':manifest['limits'],'queue_error':state.get('error')}
    atomic_json(out/'comparison.json',summary)
    text=['# 25k WAM＋现有中下层：五种子配对测试','',f'更新时间：{now()}；队列状态：{state["status"]}。',
          '',f'中下层有效完成 {len(valid)}/5，原生成功 {success}；同权重纯WAM基线4/5。',
          '', '|seed|纯WAM|中下层|控制步数|HOLD/重观测次数|动作完全相同|','|---|---|---|---:|---:|---|']
    for r in rows:
        text.append(f"|{r['seed']}|{'成功' if r['baseline_native_success'] else '失败'}|{('成功' if r['middle_native_success'] else '失败') if r['middle_native_success'] is not None else '未评分'}|{r['middle_control_steps']}|{r['supervisor_report'].get('recovery_attempts','未知')}|{r.get('raw_actions_exact_same','未知')}|")
    text+=['','实际启用：'+ '、'.join(manifest['enabled'])+'。','未启用：'+ '、'.join(manifest['disabled'])+'。',
           '中下层采用原有控制运动停滞阈值，不是空抓/滑落/投放语义诊断；未将官方评估GT注入控制或经验。完整系统尚未接通，本轮不能冒称全功能系统验收。',
           '与基线相同：50示教来源、相同25k权重与归一化、same5种子、native1700步、5100/5400/5460秒限制、serial三RGB、none denoiser、native8、无settle、保存录像及onboard。初始化无效同seed最多重试一次；中断不当失败。',
           '原生成功只检查三个刚体瓶原点位于敞口垃圾桶范围，非开盖/柔性/严格完整容纳。若中下层未触发动作干预且与基线动作一致，不能声称它提高成功率。',
           f'详细证据：`{out.relative_to(ROOT)}/comparison.json`及逐seed录像。','']
    report='\n'.join(text);(out/'report.md').write_text(report)
    with (ROOT/'docs/archive/legacy_mobile_experiments_20261004.md').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## 2026-10-07垃圾桶25k中下层测试\n\n'+report);f.flush();fcntl.flock(f,fcntl.LOCK_UN)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='outputs/runs/put_bottles_dustbin/middle25k_5each_20261007');p.add_argument('--execute',action='store_true');a=p.parse_args();out=(ROOT/a.run_dir).resolve()
    if out.exists():raise FileExistsError('new output directory required')
    receipt=check_admission();manifest=build_manifest(out,receipt)
    queue.validate_manifest(manifest)
    if not a.execute:print(json.dumps({'status':'plan_only','manifest':manifest,'commands':[queue.command(j) for j in manifest['jobs']]},indent=2));return 0
    out.mkdir(parents=True);atomic_json(out/'manifest.json',manifest)
    try:code=queue.run_manifest(manifest,out/'manifest.json')
    finally:summarize(out,manifest,read(out/'queue_status.json'))
    return code
if __name__=='__main__':raise SystemExit(main())
