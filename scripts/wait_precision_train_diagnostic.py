#!/usr/bin/env python3
"""Fixed ten TRAIN scene rollouts, immutable weights/protocol and no retries."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, sys, time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts import eval_zhuhai_precision_official_braking as evaluator
from scripts import wait_precision_official_braking as checks
from scripts import gpu_lock
from harness import zhuhai_official_braking as braking
from scripts.progress1_log import log


def select_scenarios(contract):
    rows=contract['splits']['zhuhai_task2']['train']; by_family={}
    for row in rows:by_family.setdefault(row['family_id'],[]).append(row)
    families=sorted(by_family,key=lambda f:hashlib.sha256(('train-diagnostic-20261010:'+f).encode()).hexdigest())[:5]
    scenarios=[]
    for family in families:
        for row in sorted(by_family[family],key=lambda r:r['domain']!='source'):
            import h5py
            with h5py.File(row['path'],'r') as f:meta=json.loads(f.attrs['metadata'])
            e.require(meta['family_id']==family and meta['seed']==row['seed'],'Source metadata mismatch')
            scenarios.append(dict(seed=row['seed'],family_id=family,split='train',domain=row['domain'],
                randomized=row['domain']=='randomized',source_parent=meta['parent_source_seed'],
                path=row['path'],sha256=row['sha256'],purpose='train_scene_diagnostic'))
    e.require(len(scenarios)==10 and len({s['seed'] for s in scenarios})==10,'Five paired families required')
    return scenarios


def build(args):
    meta,contract=e.checkpoint_metadata(args.checkpoint)
    e.require(meta['step']==30000,'Frozen30k required')
    out=Path(args.out).absolute();e.require(out==out.resolve() and not out.exists(),'Fresh output required')
    parent=e.read_json(checks.BASE/'wam_precision_v2_task2_official_braking30k_20261010_v1/plan.json')
    checks.verify_sources(parent)
    files={**parent['extension_sources'],str(Path(__file__).resolve()):e.sha256_file(__file__),
        str(ROOT/'scripts/eval_precision_train_scenarios.py'):e.sha256_file(ROOT/'scripts/eval_precision_train_scenarios.py')}
    scenarios=select_scenarios(contract)
    return dict(schema='precision-train-scene-diagnostic10-v1',checkpoint=meta,out=str(out),
        scenarios=scenarios,frozen_sources=contract['source_files'],extension_sources=files,
        action_postprocessing=braking.protocol(),action_postprocessing_sha256=e.digest(braking.protocol()),
        child_env=parent['child_env'],gpu_lock_owner=args.gpu_lock_owner,
        scope='TRAIN fit diagnostic only, not generalization; same scene seed/domain/parent, fresh autonomous rollout',
        selection='five TRAIN families sorted by SHA256 fixed salt before observing outcomes; source+paired DR',
        record_video=False,third_view=False,max_steps=2400)


def verify(plan):
    for name,sha in {**plan['frozen_sources'],**plan['extension_sources']}.items():
        e.require(e.sha256_file(name)==sha,'Frozen source drift: '+name)
    for row in plan['scenarios']:e.require(e.sha256_file(row['path'])==row['sha256'],'TRAIN source drift')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--gpu-lock-owner',default='zhuhai-train-diagnostic10-20261010-v1')
    ap.add_argument('--execute',action='store_true');args=ap.parse_args();plan=build(args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False));return 0
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=False);e.write_json(out/'plan.json',plan)
    rows=[];held=False
    def publish(status,error=None):
        result=dict(schema=plan['schema'],status=status,planned_episodes=10,episodes=rows,
            checkpoint=plan['checkpoint'],scope=plan['scope'],record_video=False,third_view=False,
            full_successes=sum(r.get('full_task_success') is True for r in rows),
            success_rate=sum(r.get('full_task_success') is True for r in rows)/10 if status=='completed' else None)
        if error:result['error']=error
        e.write_json(out/'summary.json',result)
    publish('waiting_for_exclusive_gpu')
    from scripts import run_zhuhai_task1_dr200_concurrent as admission
    try:
        verify(plan)
        deadline=time.monotonic()+3600
        while gpu_lock.owner() is not None or admission.gpu_jobs(admission.processes()):
            e.require(time.monotonic()<deadline,'GPU wait deadline');time.sleep(10)
        gpu_lock.acquire(plan['gpu_lock_owner'],max_wait=0);held=True
        resource=admission.scheduler_module().collection_resource_admission()
        e.require(admission.scheduler_module().resource_ready(resource,expansion=True),'Resource reserve denied')
        e.write_json(out/'resource_admission.json',resource)
        env=os.environ.copy();env.update(plan['child_env'])
        with Path(plan['checkpoint']['path']).open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            publish('evaluating')
            for i,scenario in enumerate(plan['scenarios']):
                verify(plan);target=out/'zhuhai_task2'/f'seed_{scenario["seed"]}';target.parent.mkdir(exist_ok=True)
                item=dict(seed=scenario['seed'],out=str(target),command=[str(e.PYTHON),'-B',
                    str(ROOT/'scripts/eval_precision_train_scenarios.py'),'--scenario-plan',str(out/'plan.json'),
                    '--scenario-index',str(i),'--checkpoint',plan['checkpoint']['path'],'--task','zhuhai_task2',
                    '--gpu-lock-owner',plan['gpu_lock_owner'],'--out',str(target),'--episode-seed',str(scenario['seed']),
                    '--max-steps','2400','--wall-seconds','3600','--startup-timeout','180','--policy-timeout','60'])
                e.write_json(out/f'attempt_seed_{scenario["seed"]}.json',item)
                code=e.run_episode(item,env);e.require(code==0,'TRAIN diagnostic failed; preserve, no retry')
                raw=e.read_json(target/'result.json');e.require(raw['status']=='completed','Diagnostic incomplete')
                runtime=e.validate_binding(raw,30000,scenario['seed'],plan['checkpoint']['sha256'],evaluator.overlay(e.checkpoint_metadata(args.checkpoint)[1]))
                score=e.validate_rule(raw['final_rule_evidence']);files=e.archive_evidence(target,raw,runtime,raw['final_rule_evidence'])
                e.require(raw['full_task_success']==score['full_task_success'],'Raw score mismatch')
                rows.append(dict(seed=scenario['seed'],family_id=scenario['family_id'],domain=scenario['domain'],
                    status='completed',full_task_success=raw['full_task_success'],base_score=raw['base_score'],
                    control_steps=raw['control_steps'],raw_result_path=str(target/'result.json'),
                    raw_result_sha256=e.sha256_file(target/'result.json'),evidence_files=files,attempt_count=1))
                publish('evaluating');log(f'TRAIN布局诊断 {i+1}/10 seed{scenario["seed"]} {scenario["domain"]}：完整成功{raw["full_task_success"]}，分数{raw["base_score"]}；非泛化成绩。','train_scene_diagnostic')
            verify(plan);publish('completed')
        return 0
    except BaseException as exc:publish('halted',f'{type(exc).__name__}: {exc}');raise
    finally:
        if held:
            if admission.gpu_jobs(admission.processes()):e.write_json(out/'owner_retained_guard.json',dict(owner=plan['gpu_lock_owner']))
            else:gpu_lock.release(plan['gpu_lock_owner'])


if __name__=='__main__':raise SystemExit(main())
