#!/usr/bin/env python3
"""Final-only 50k fixed five-seed evaluation; frozen official-braking executor."""
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
from scripts.wait_precision_official_braking import classify
from scripts import gpu_lock, run_zhuhai_task1_dr200_concurrent as admission
from harness import zhuhai_precision_v2_training as model
from harness import zhuhai_dagger_segmented_training_v7 as quota
from harness.zhuhai_official_braking import protocol, VERSION
from scripts.progress1_log import log
SCHEMA='precision-dagger-final50k-official-braking-fixed5-v9'
BASE=ROOT/'outputs/zhuhai2026_tasks12'
BASELINE=BASE/'wam_precision_v2_task2_official_braking30k_20261010_v1/summary.json'


def training_finished(status):
    return (status.get('status')=='completed' and status.get('step')==50000
            and status.get('additional_updates')==20000
            and status.get('new_segmented_episodes')==50 and status.get('legacy_episodes')==20
            and status.get('final_checkpoint_integrity_verified') is True)


def metadata50(path):
    path=Path(path).resolve(strict=True)
    receipt=e.read_json(path.with_name(path.name+'.sha256.json'))
    e.require(path.name=='checkpoint_step_050000.pt' and receipt.get('step')==50000
              and receipt.get('model_only') is True and receipt.get('size')==path.stat().st_size,
              'Exact immutable 50000 model-only milestone required')
    e.require(e.valid_sha(receipt.get('sha256')) and e.sha256_file(path)==receipt['sha256'],
              '50k checkpoint content differs from receipt')
    manifest_path=path.parent/'manifest.json';manifest=e.read_json(manifest_path);contract=manifest['contract']
    e.require(contract.get('format')==e.FORMAT and e.digest(contract)==manifest['contract_sha256'],
              '50k contract identity differs')
    model.validate_training_contract(contract);model.verify_code_hashes(contract['source_files'])
    d=contract['dagger_training']
    e.require((d['start_step'],d['target_step'],d['new_segmented_episodes'],
               d['legacy_correction_episodes'],d['correction_episodes'])==(30000,50000,50,20,70),
              '50k correction/training quota differs')
    return dict(path=str(path),step=50000,sha256=receipt['sha256'],size=receipt['size'],
                manifest_sha256=e.sha256_file(manifest_path),
                receipt_sha256=e.sha256_file(path.with_name(path.name+'.sha256.json'))),contract


def build_plan(args):
    training=Path(args.training).resolve();collection=Path(args.collection).resolve()
    e.require(training_finished(e.read_json(training/'status.json')), 'Training must finish all 20000 new updates first')
    cm=e.read_json(collection/'manifest.json');quota.validate_quota(cm)
    meta,contract=metadata50(training/'train/checkpoint_step_050000.pt')
    for name,model_only in [('checkpoint_last.pt',False),('checkpoint_step_050000.pt',True)]:
        file=training/'train'/name;receipt=e.read_json(file.with_name(file.name+'.sha256.json'))
        e.require(receipt['step']==50000 and receipt['model_only'] is model_only
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
    for relative in ('scripts/eval_precision_dagger50k_fixed5_v9.py',
                     'scripts/eval_zhuhai_precision_official_braking.py',
                     'scripts/wait_precision_official_braking.py',
                     'scripts/eval_zhuhai_precision_fixed_attempts.py'):
        file=ROOT/relative;extensions[str(file)]=e.sha256_file(file)
    commands=[dict(seed=seed,out=str(out/e.TASK/f'seed_{seed}'),command=[str(e.PYTHON),'-B',
        str(Path(evaluator.__file__).resolve()),'--checkpoint',meta['path'],'--out',str(out/e.TASK/f'seed_{seed}'),
        '--task',e.TASK,'--gpu-lock-owner',args.gpu_lock_owner,'--episode-seed',str(seed),
        '--max-steps','2400','--wall-seconds','3600','--startup-timeout','180','--policy-timeout','60'])
        for seed in e.SEEDS]
    return dict(schema=SCHEMA,out=str(out),checkpoint=meta,frozen_sources=contract['source_files'],
        extension_sources=extensions,commands=commands,child_env=collection_plan['child_env'],
        gpu_lock_owner=args.gpu_lock_owner,action_postprocessing=protocol(),
        action_postprocessing_sha256=e.digest(protocol()),baseline=dict(path=str(BASELINE),sha256=e.sha256_file(BASELINE),
        success_rate=baseline['success_rate']),collection=str(collection),training=str(training),
        record_video=False,third_view=False,randomized=False,seeds=list(e.SEEDS),
        interpretation='same fixed development cohort and official braking as 30k; additional training plus correction data jointly changed; not an independent generalization estimate')


def verify(plan):
    for file,sha in {**plan['frozen_sources'],**plan['extension_sources']}.items():
        e.require(e.sha256_file(file)==sha,'Frozen final-evaluation source drift: '+file)
    e.require(e.sha256_file(plan['baseline']['path'])==plan['baseline']['sha256'],'Original baseline changed')


def summary(plan,rows,status,error=None):
    done=status=='completed' and len(rows)==5
    if status=='completed':e.require([r['seed'] for r in rows]==list(e.SEEDS),'Exact original five-seed order required')
    result=dict(schema=SCHEMA,status=status,checkpoint=plan['checkpoint'],checkpoint_step=50000,
        checkpoint_sha256=plan['checkpoint']['sha256'],evaluation_version=VERSION,
        action_postprocessing_sha256=plan['action_postprocessing_sha256'],seeds=list(e.SEEDS),
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
        meta,contract=metadata50(plan['checkpoint']['path']);e.require(meta==plan['checkpoint'],'Final checkpoint changed')
        env=os.environ.copy();env.update(plan['child_env'])
        with Path(meta['path']).open('rb') as locked:
            fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB);publish('evaluating')
            log('追加20k已完全完成、50k模型/full-state独立核验通过，开始原932101–932105五场源分布最终复测；同官方制动/评分、无中途评测/视频/第三视角。','dagger50k_final_test')
            for item in plan['commands']:
                verify(plan);e.verify_external_owner(plan['gpu_lock_owner'])
                target=Path(item['out']);target.parent.mkdir(exist_ok=True)
                e.require(not target.exists(),'No repeated physical final test attempt')
                e.write_json(out/f'attempt_seed_{item["seed"]}.json',item)
                code=e.run_episode(item,env);e.write_json(out/f'exit_seed_{item["seed"]}.json',dict(returncode=code))
                row=classify(target/'result.json',meta,contract,plan['extension_sources'][str(Path(evaluator.__file__).resolve())])
                e.require(code in (0,2) if row['status']=='control_safety_abort' else code==0,'Final case/exit mismatch')
                rows.append(row);publish('evaluating')
                log(f'50k最终seed{item["seed"]}：{row["status"]}，完整成功{row["full_task_success"]}，末分{row["observed_base_score"]}。','dagger50k_final_test')
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
    parser.add_argument('--execute',action='store_true');args=parser.parse_args();plan=build_plan(args)
    if not args.execute:print(json.dumps(plan));return 0
    return execute(plan)


if __name__=='__main__':raise SystemExit(main())
