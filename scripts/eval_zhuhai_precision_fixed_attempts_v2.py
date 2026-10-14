#!/usr/bin/env python3
"""Finish exact fixed25k/30k attempts; preserve Isaac post-result exit semantics.

Original classifier/rules/limiter are immutable. A verified unsuccessful control
safety abort may return0 (SimulationApp.close exits) or2; record actual return
code rather than replacing the raw result or retrying the physical episode.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as old
BASE=ROOT/'outputs/zhuhai2026_tasks12'
ORIGIN=old.RUN
PRIOR=BASE/'wam_precision_v2_task2_30k_continuation_20261010_r2'
ADOPT25=PRIOR/'development_25000/zhuhai_task2/seed_932101/result.json'
SCHEMA='precision-v2-fixed-attempt-exit-aware-plan-v2'


def acceptable_exit(row,code):
    if row['status']=='control_safety_abort':return code in (0,2)
    return row['status']=='completed' and code==0


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--task',default=old.TASK);p.add_argument('--seeds',type=int,nargs='+',default=list(old.SEEDS))
    p.add_argument('--gpu-lock-owner');p.add_argument('--execute',action='store_true')
    return p


def plan(args):
    meta,contract=old.checkpoint_metadata(args.checkpoint)
    old.require(meta['step']in(25000,30000) and args.task==old.TASK and list(args.seeds)==list(old.SEEDS),'Only unchanged25k/30k cohorts')
    out=Path(args.out).absolute();old.require(not out.exists() and out.resolve()==out and BASE in out.parents,'Fresh canonical project output required')
    protocol=old.load_protocol();env=old.read_json(old.ENV_LAUNCH)['plan']['child_env']
    old.require(env.get('OMNI_KIT_ACCEPT_EULA')=='YES' and Path(env['LD_PRELOAD']).is_file(),'Restore known prior environment')
    adopted=[str(ADOPT25)]if meta['step']==25000 else[];commands=[]
    for seed in old.SEEDS:
        if adopted and seed==932101:continue
        target=out/old.TASK/f'seed_{seed}'
        cmd=[str(old.PYTHON),'-B',str(old.EVAL),'--checkpoint',meta['path'],'--out',str(target),'--task',old.TASK,
             '--gpu-lock-owner',args.gpu_lock_owner or'','--episode-seed',str(seed),'--max-steps','2400',
             '--wall-seconds','3600','--startup-timeout','180','--policy-timeout','60']
        commands.append(dict(seed=seed,out=str(target),command=cmd))
    return dict(schema=SCHEMA,out=str(out),checkpoint=meta,task=old.TASK,seeds=list(old.SEEDS),planned_episodes=5,
        adopt_results=adopted,commands=commands,gpu_lock_owner=args.gpu_lock_owner,child_env=env,record_video=False,third_view=False,
        frozen_sources=contract['source_files'],classifier_sha256=old.sha256_file(old.__file__),wrapper_sha256=old.sha256_file(__file__),
        protocol_sha256=old.sha256_file(old.PROTOCOL),
        exit_protocol='completed=>exit0; fully verified control_safety_abort=>exit0or2; actual code retained separately; other errors HALT')


def verify_sources(p):
    for name,sha in p['frozen_sources'].items():old.require(old.sha256_file(name)==sha,'Frozen source drift: '+name)
    old.require(old.sha256_file(old.__file__)==p['classifier_sha256'] and old.sha256_file(__file__)==p['wrapper_sha256'],'Wrapper/classifier drift')
    old.require(old.sha256_file(old.PROTOCOL)==p['protocol_sha256'],'Attempt scoring protocol drift')


def reject_repeated_attempts(p):
    allowed={str(Path(x).resolve())for x in p['adopt_results']}
    for run in BASE.glob('wam_precision_v2*'):
        for file in run.glob('*/plan.json'):
            prior=old.read_json(file)
            if prior.get('schema')not in ('precision-v2-fixed-attempt-plan-v1',SCHEMA):continue
            if prior.get('checkpoint',{}).get('sha256')!=p['checkpoint']['sha256']or file.parent==Path(p['out']):continue
            for intent in file.parent.glob('attempt_seed_*.json'):
                item=old.read_json(intent);result=str((Path(item['out'])/'result.json').resolve())
                old.require(result in allowed,'Already attempted seed outside exact adoption set; never retry '+str(intent))


def execute(p):
    old.require(p['gpu_lock_owner'],'Explicit owner required');old.verify_external_owner(p['gpu_lock_owner']);verify_sources(p);reject_repeated_attempts(p)
    out=Path(p['out']);meta=p['checkpoint'];rows=[]
    with Path(meta['path']).open('rb')as checkpoint_lock:
        fcntl.flock(checkpoint_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(parents=True,exist_ok=False)
        old.write_json(out/'plan.json',p)
        try:
            _,contract=old.checkpoint_metadata(meta['path'])
            for path in p['adopt_results']:
                row=old.classify_result(path,meta['step'],932101,meta['sha256'],contract=contract)
                old.require(row['status']=='control_safety_abort','Expected immutable25k failed first attempt');rows.append(row)
            old.write_json(out/'summary.json',old.build_summary(meta,rows,'evaluating'))
            env=os.environ.copy();env.update(p['child_env'])
            for item in p['commands']:
                old.verify_external_owner(p['gpu_lock_owner']);verify_sources(p)
                target=Path(item['out']);old.require(not target.exists(),'No repeated physical attempt');target.parent.mkdir(exist_ok=True)
                old.write_json(out/f'attempt_seed_{item["seed"]}.json',item)
                code=old.run_episode(item,env)
                old.write_json(out/f'exit_seed_{item["seed"]}.json',dict(seed=item['seed'],returncode=code,result=str(target/'result.json')))
                try:
                    row=old.classify_result(target/'result.json',meta['step'],item['seed'],meta['sha256'],contract=contract)
                    old.require(acceptable_exit(row,code),'Verified result and child exit disagree; preserve and HALT')
                except BaseException as exc:
                    rows.append(old.rejected_row(target/'result.json',item['seed'],exc));raise
                rows.append(row);old.write_json(out/'summary.json',old.build_summary(meta,rows,'evaluating'))
            old.write_json(out/'summary.json',old.build_summary(meta,rows,'completed'))
            old.verify_summary(out/'summary.json',meta['step'],p['seeds'],meta['sha256']);return 0
        except BaseException as exc:
            old.write_json(out/'summary.json',old.build_summary(meta,rows,'halted_infrastructure_error',error=f'{type(exc).__name__}: {exc}'))
            raise


def main(argv=None):
    args=parser().parse_args(argv);p=plan(args)
    if not args.execute:print(json.dumps(p,ensure_ascii=False,indent=2));return 0
    return execute(p)


if __name__=='__main__':raise SystemExit(main())
