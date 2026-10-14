#!/usr/bin/env python3
"""Collect fifty verified on-policy expert corrections after TRAIN diagnostic."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,fcntl
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts import wait_precision_official_braking as checks
from scripts import gpu_lock
from harness.zhuhai_dagger import SCHEMA,DaggerEpisode
from scripts.progress1_log import log


def build(args):
    meta,contract=e.checkpoint_metadata(args.checkpoint)
    e.require(meta['step']==30000,'Frozen30k required')
    diagnostic=e.read_json(Path(args.train_diagnostic)/'plan.json')
    e.require(diagnostic['checkpoint']==meta,'TRAIN diagnostic checkpoint differs')
    excluded={r['family_id'] for r in diagnostic['scenarios']}
    parents=[r for r in contract['splits']['zhuhai_task2']['train'] if r['domain']=='source' and r['family_id'] not in excluded]
    parents.sort(key=lambda r:hashlib.sha256(('dagger50-20261010:'+r['family_id']).encode()).hexdigest())
    e.require(len(parents)==75,'Exactly75 remaining TRAIN families required')
    forbidden={r['seed'] for rows in contract['splits']['zhuhai_task2'].values() for r in rows}|set(e.SEEDS)
    candidates=[dict(seed=1010000+i,source_parent=parents[i%75]['seed'],family_id=parents[i%75]['family_id'],
        split='train',randomized=True,parent_path=parents[i%75]['path'],parent_sha256=parents[i%75]['sha256']) for i in range(200)]
    e.require(not forbidden & {r['seed'] for r in candidates},'Correction/test seed overlap')
    out=Path(args.out).absolute();e.require(out==out.resolve() and not out.exists(),'Fresh correction output required')
    binding=e.read_json(checks.BASE/'wam_precision_v2_task2_official_braking30k_20261010_v1/plan.json')
    sources={**binding['extension_sources']}
    for rel in ['harness/zhuhai_dagger.py','harness/isaac_zhuhai_dagger_expert.py',
                'harness/isaac_zhuhai_task2_expert_repair_v3.py','harness/isaac_zhuhai_expert.py',
                'harness/isaac_zhuhai_runtime.py','harness/isaac_zhuhai_task2.py','harness/zhuhai_task2_rules.py',
                'scripts/collect_precision_dagger_episode.py','scripts/wait_precision_dagger50.py']:
        sources[str(ROOT/rel)]=e.sha256_file(ROOT/rel)
    return dict(schema=SCHEMA,checkpoint=meta,out=str(out),candidates=candidates,
        train_diagnostic=str(Path(args.train_diagnostic).resolve()),train_diagnostic_plan_sha256=e.sha256_file(Path(args.train_diagnostic)/'plan.json'),
        frozen_sources=contract['source_files'],extension_sources=sources,child_env=binding['child_env'],
        gpu_lock_owner=args.gpu_lock_owner,target_corrections=50,initial_validation_corrections=10,
        maximum_attempts=200,maximum_consecutive_expert_failures=3,
        takeover_gate=dict(minimum_steps=300,stalled_steps=180,maximum_steps=800),
        record_video=False,third_view=False,heldout_used=False,
        supervision='verified expert action suffix only; full prefix is causal history; full strict task continuation required',
        policy='frozen30k; one collection iteration, no training in collection process',
        admitted_data_split='train; original VAL/heldout and fixed5 excluded')


def verify(plan):
    for path,sha in {**plan['frozen_sources'],**plan['extension_sources']}.items():
        e.require(e.sha256_file(path)==sha,'Frozen correction source drift: '+path)
    e.require(e.sha256_file(Path(plan['train_diagnostic'])/'plan.json')==plan['train_diagnostic_plan_sha256'],'Diagnostic plan drift')


def execute(plan):
    from scripts import run_zhuhai_task1_dr200_concurrent as admission
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=False);e.write_json(out/'plan.json',plan)
    attempts=[];rows=[];held=False;failures=0
    def publish(status,error=None):
        result=dict(schema=SCHEMA,status=status,checkpoint=plan['checkpoint'],success_count=len(rows),
            requested_success_count=50,attempts=attempts,train_rows=rows,
            consecutive_expert_failures=failures,initial_validation_passed=len(rows)>=10,
            record_video=False,third_view=False,heldout_used=False,training_started=False)
        if error:result['error']=error
        e.write_json(out/'manifest.json',result)
    publish('waiting_for_train_diagnostic')
    try:
        verify(plan);deadline=time.monotonic()+14400
        while True:
            d=e.read_json(Path(plan['train_diagnostic'])/'summary.json')
            if d['status']=='completed':break
            e.require(d['status']!='halted','TRAIN diagnostic halted; preserve and investigate')
            e.require(time.monotonic()<deadline,'Diagnostic wait deadline');time.sleep(10)
        publish('waiting_for_exclusive_gpu')
        while gpu_lock.owner() is not None or admission.gpu_jobs(admission.processes()):
            e.require(time.monotonic()<deadline,'GPU wait deadline');time.sleep(10)
        gpu_lock.acquire(plan['gpu_lock_owner'],max_wait=0);held=True
        env=os.environ.copy();env.update(plan['child_env'])
        with Path(plan['checkpoint']['path']).open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for index,candidate in enumerate(plan['candidates']):
                verify(plan)
                e.require(e.sha256_file(candidate['parent_path'])==candidate['parent_sha256'],'TRAIN parent source changed')
                resource=admission.scheduler_module().collection_resource_admission()
                e.require(admission.scheduler_module().resource_ready(resource,expansion=True),'Correction resource reserve denied')
                target=out/'episodes'/f'seed_{candidate["seed"]}';target.parent.mkdir(exist_ok=True)
                item=dict(seed=candidate['seed'],out=str(target),command=[str(e.PYTHON),'-B',
                    str(ROOT/'scripts/collect_precision_dagger_episode.py'),'--plan',str(out/'plan.json'),
                    '--index',str(index),'--gpu-lock-owner',plan['gpu_lock_owner']])
                e.write_json(out/f'attempt_seed_{candidate["seed"]}.json',dict(**item,resource=resource))
                publish('validating_first10' if len(rows)<10 else 'collecting')
                code=e.run_episode(item,env)
                raw=e.read_json(target/'result.json');attempts.append(dict(seed=candidate['seed'],status=raw['status'],
                    raw_result_path=str(target/'result.json'),raw_result_sha256=e.sha256_file(target/'result.json')))
                e.require(code==0 and raw['status']!='infrastructure_error','Correction infrastructure error; preserve, no retry')
                if raw['status']=='success':
                    admitted=dict(path=raw['data_path'],sha256=raw['data_sha256'],seed=candidate['seed'],
                        family_id=candidate['family_id'],source_parent=candidate['source_parent'],split='train',domain='dagger',
                        policy_steps=raw['policy_steps'],expert_steps=raw['expert_steps'],takeover_step=raw['takeover_step'])
                    episode=DaggerEpisode(admitted)
                    e.require(episode.supervised_count==raw['expert_steps'],'Expert supervision length mismatch')
                    rows.append(admitted);failures=0
                elif raw['status']=='expert_failed':failures+=1
                elif raw['status']!='policy_succeeded_without_correction':raise ValueError('Unknown correction terminal status')
                publish('validating_first10' if len(rows)<10 else 'collecting')
                log(f'DAgger接管seed{candidate["seed"]} {raw["status"]}；合格{len(rows)}/50，连续专家失败{failures}，失败policy仅上下文。','dagger_collection')
                if failures>=3:
                    e.write_json(out/'expert_failure_guard.json',dict(consecutive_failures=failures,attempts=attempts[-3:]))
                    raise ValueError('Three consecutive expert recovery failures; new version required')
                if len(rows)==50:verify(plan);publish('complete');return 0
            raise ValueError('Candidate budget exhausted before fifty valid corrections')
    except BaseException as exc:publish('halted',f'{type(exc).__name__}: {exc}');raise
    finally:
        if held:
            if admission.gpu_jobs(admission.processes()):e.write_json(out/'owner_retained_guard.json',dict(owner=plan['gpu_lock_owner']))
            else:gpu_lock.release(plan['gpu_lock_owner'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--train-diagnostic',required=True);p.add_argument('--out',required=True)
    p.add_argument('--gpu-lock-owner',default='zhuhai-dagger50-20261010-v1');p.add_argument('--execute',action='store_true')
    args=p.parse_args();plan=build(args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False));return 0
    return execute(plan)


if __name__=='__main__':raise SystemExit(main())
