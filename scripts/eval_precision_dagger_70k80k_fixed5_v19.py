#!/usr/bin/env python3
"""Paired 70k/80k immutable milestones, ten fixed source scenes each, after full training."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts import eval_zhuhai_precision_official_braking as evaluator
from scripts import gpu_lock, run_zhuhai_task1_dr200_concurrent as admission
from harness import zhuhai_precision_v2_training as model
from harness import zhuhai_dagger_training30k_v14 as quota
from harness.zhuhai_official_braking import protocol, VERSION
from scripts.progress1_log import log
from scripts import precision_physics_abort_evidence_v19 as contact
from scripts import eval_precision_dagger_70k80k_fixed5_v14 as prior
SCHEMA='precision-dagger-70k80k-official-braking-fixed5-v19'
SEEDS=tuple(e.SEEDS)
BASE=ROOT/'outputs/zhuhai2026_tasks12'
BASELINE=BASE/'wam_precision_v2_task2_official_braking30k_20261010_v1/summary.json'


import ast
from harness import zhuhai_official_braking as braking


def classify(path, metadata, frozen, source_hash):
    return contact.classify(path, metadata, frozen, source_hash, SEEDS, prior.classify)


def training_finished(status):
    return (status.get('status')=='completed' and status.get('step')==80000
            and status.get('additional_updates')==30000
            and status.get('new_segmented_episodes')==100 and status.get('legacy_episodes')==20
            and status.get('final_checkpoint_integrity_verified') is True)


def metadata_milestone(path, expected_step):
    e.require(type(expected_step) is int and expected_step in (70000,80000), "Only exact 70k/80k milestones admitted")
    path=Path(path).resolve(strict=True)
    receipt=e.read_json(path.with_name(path.name+'.sha256.json'))
    e.require(path.name==f'checkpoint_step_{expected_step:06d}.pt' and receipt.get('step')==expected_step
              and receipt.get('model_only') is True and receipt.get('size')==path.stat().st_size,
              'Exact immutable requested model-only milestone required')
    e.require(e.valid_sha(receipt.get('sha256')) and e.sha256_file(path)==receipt['sha256'],
              '50k checkpoint content differs from receipt')
    manifest_path=path.parent/'manifest.json';manifest=e.read_json(manifest_path);contract=manifest['contract']
    e.require(contract.get('format')==e.FORMAT and e.digest(contract)==manifest['contract_sha256'],
              '50k contract identity differs')
    model.validate_training_contract(contract);model.verify_code_hashes(contract['source_files'])
    d=contract['dagger_training']
    e.require((d['start_step'],d['target_step'],d['new_segmented_episodes'],
               d['legacy_correction_episodes'],d['correction_episodes'])==(50000,80000,100,20,120),
              '50k correction/training quota differs')
    return dict(path=str(path),step=expected_step,sha256=receipt['sha256'],size=receipt['size'],
                manifest_sha256=e.sha256_file(manifest_path),
                receipt_sha256=e.sha256_file(path.with_name(path.name+'.sha256.json'))),contract


def build_plan(args):
    training=Path(args.training).resolve();collection=Path(args.collection).resolve()
    e.require(training_finished(e.read_json(training/'status.json')), 'Training must finish all 30000 new updates first')
    cm=e.read_json(collection/'manifest.json');quota.validate_quota(cm)
    meta,contract=metadata_milestone(training/f'train/checkpoint_step_{args.milestone:06d}.pt',args.milestone)
    for name,model_only in [('checkpoint_last.pt',False),('checkpoint_step_080000.pt',True)]:
        file=training/'train'/name;receipt=e.read_json(file.with_name(file.name+'.sha256.json'))
        e.require(receipt['step']==80000 and receipt['model_only'] is model_only
                  and receipt['size']==file.stat().st_size and receipt['sha256']==e.sha256_file(file),
                  'Final model/full-state integrity must pass')
    e.require(contract['dagger_training']['collection']['path']==str(collection/'manifest.json')
              and contract['dagger_training']['collection']['sha256']==e.sha256_file(collection/'manifest.json'),
              'Evaluation collection differs from the final training contract')
    baseline=e.read_json(BASELINE)
    e.require(baseline['status']=='completed' and baseline['checkpoint']['step']==30000
              and baseline['seeds']==list(e.SEEDS) and baseline['evaluation_version']==VERSION,
              'Frozen official-braking 30k fixed cohort required for comparison')
    out=Path(args.out).absolute()
    e.require(out==out.resolve() and BASE in out.parents and not out.exists(), 'Fresh final evaluation output required')
    collection_plan=e.read_json(collection/'plan.json')
    extensions=dict(collection_plan['extension_sources'])
    for relative in ('scripts/eval_precision_dagger_70k80k_fixed5_v14.py', 'scripts/eval_precision_dagger_70k80k_fixed5_v19.py',
                     'scripts/precision_physics_abort_evidence_v19.py',
                     'scripts/eval_zhuhai_precision_official_braking.py',
                     'scripts/wait_precision_official_braking.py',
                     'scripts/eval_zhuhai_precision_fixed_attempts.py'):
        file=ROOT/relative;extensions[str(file)]=e.sha256_file(file)
    commands=[dict(seed=seed,out=str(out/e.TASK/f'seed_{seed}'),command=[str(e.PYTHON),'-B',
        str(Path(evaluator.__file__).resolve()),'--checkpoint',meta['path'],'--out',str(out/e.TASK/f'seed_{seed}'),
        '--task',e.TASK,'--gpu-lock-owner',args.gpu_lock_owner,'--episode-seed',str(seed),
        '--max-steps','2400','--wall-seconds','3600','--startup-timeout','180','--policy-timeout','60'])
        for seed in SEEDS]
    return dict(schema=SCHEMA,out=str(out),checkpoint=meta,frozen_sources=contract['source_files'],
        extension_sources=extensions,commands=commands,child_env=collection_plan['child_env'],
        gpu_lock_owner=args.gpu_lock_owner,action_postprocessing=protocol(),
        action_postprocessing_sha256=e.digest(protocol()),baseline=dict(path=str(BASELINE),sha256=e.sha256_file(BASELINE),
        success_rate=baseline['success_rate'],comparable_seeds=list(e.SEEDS),planned_episodes=5),collection=str(collection),training=str(training),
        record_video=False,third_view=False,randomized=False,seeds=list(SEEDS),
        interpretation='70k and80k each use identical five fixed source development scenes overlapping the30k baseline; no independent generalization claim',physical_evaluation_after_training=True)


def verify(plan):
    for file,sha in {**plan['frozen_sources'],**plan['extension_sources']}.items():
        e.require(e.sha256_file(file)==sha,'Frozen final-evaluation source drift: '+file)
    e.require(e.sha256_file(plan['baseline']['path'])==plan['baseline']['sha256'],'Original baseline changed')


def summary(plan,rows,status,error=None):
    done=status=='completed' and len(rows)==5
    if status=='completed':e.require([r['seed'] for r in rows]==list(SEEDS),'Exact five-seed order required')
    result=dict(schema=SCHEMA,status=status,checkpoint=plan['checkpoint'],checkpoint_step=plan['checkpoint']['step'],
        checkpoint_sha256=plan['checkpoint']['sha256'],evaluation_version=VERSION,
        action_postprocessing_sha256=plan['action_postprocessing_sha256'],seeds=list(SEEDS),
        planned_episodes=5,episodes=rows,full_successes=sum(r['full_task_success'] for r in rows),
        success_rate=sum(r['full_task_success'] for r in rows)/5 if done else None,
        baseline=plan['baseline'],record_video=False,third_view=False,randomized=False,
        interpretation=plan['interpretation'])
    if error:result['error']=error
    return result


def execute(plan):
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=False);e.write_json(out/'plan.json',plan)
    rows=[];held=False
    def publish(status,error=None):e.write_json(out/'summary.json',summary(plan,rows,status,error))
    publish('waiting_for_exclusive_gpu')
    try:
        verify(plan);deadline=time.monotonic()+21600
        while gpu_lock.owner() is not None or admission.gpu_jobs(admission.processes()):
            e.require(time.monotonic()<deadline,'Final evaluation GPU wait exceeded six hours');time.sleep(10)
        resource=admission.scheduler_module().collection_resource_admission()
        e.require(admission.scheduler_module().resource_ready(resource,expansion=True),'Final evaluation resources insufficient')
        gpu_lock.acquire(plan['gpu_lock_owner'],max_wait=0);held=True
        e.require(not admission.gpu_jobs(admission.processes()),'Foreign GPU job appeared during final admission')
        meta,contract=metadata_milestone(plan['checkpoint']['path'],plan['checkpoint']['step']);e.require(meta==plan['checkpoint'],'Final checkpoint changed')
        env=os.environ.copy();env.update(plan['child_env'])
        with Path(meta['path']).open('rb') as locked:
            fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB);publish('evaluating')
            log(f'完整追加30k完成，开始{meta["step"]}权重固定五测932101–932105；同官方制动/评分/源分布，无录像/额外视角。','dagger_milestone_test')
            for item in plan['commands']:
                verify(plan);e.verify_external_owner(plan['gpu_lock_owner'])
                target=Path(item['out']);target.parent.mkdir(exist_ok=True)
                e.require(not target.exists(),'No repeated physical final test attempt')
                e.write_json(out/f'attempt_seed_{item["seed"]}.json',item)
                code=e.run_episode(item,env);e.write_json(out/f'exit_seed_{item["seed"]}.json',dict(returncode=code))
                row=classify(target/'result.json',meta,contract,plan['extension_sources'][str(Path(evaluator.__file__).resolve())])
                e.require(code in (0,2) if row['status'] in ('control_safety_abort','physics_safety_abort') else code==0,'Final case/exit mismatch')
                rows.append(row);publish('evaluating')
                log(f'{meta["step"]}测试seed{item["seed"]}：{row["status"]}，完整成功{row["full_task_success"]}，末分{row["observed_base_score"]}。','dagger50k_final_test')
            verify(plan);publish('completed')
        return 0
    except BaseException as exc:
        publish('halted',f'{type(exc).__name__}: {exc}');raise
    finally:
        if held:
            jobs=admission.gpu_jobs(admission.processes())
            if jobs:e.write_json(out/'owner_retained_guard.json',dict(owner=plan['gpu_lock_owner'],jobs=jobs))
            else:gpu_lock.release(plan['gpu_lock_owner'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',required=True);parser.add_argument('--collection',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--gpu-lock-owner',required=True)
    parser.add_argument('--milestone',required=True,type=int,choices=(70000,80000));parser.add_argument('--execute',action='store_true');args=parser.parse_args();plan=build_plan(args)
    if not args.execute:print(json.dumps(plan));return 0
    return execute(plan)


if __name__=='__main__':raise SystemExit(main())
