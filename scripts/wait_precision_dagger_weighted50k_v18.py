#!/usr/bin/env python3
"""Wait for fifty new segmented plus twenty retained legacy episodes, then exclusively run the authorized +20k."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import train_precision_dagger_weighted50k_v18 as training
from scripts import gpu_lock
from scripts import run_zhuhai_task1_dr200_concurrent as admission
from scripts import eval_zhuhai_precision_fixed_attempts as e
from harness import zhuhai_dagger_segmented_training_v7 as d
from harness.zhuhai_dagger_segmented import SegmentedEpisode as DaggerEpisode
from scripts.progress1_log import log


def execute(args, plan):
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    e.write_json(root/'plan.json',plan)
    held=False
    def publish(status,error=None):
        row=dict(schema=d.SCHEMA,status=status,step=plan['resume_step'],target_steps=50000,additional_updates=plan['resume_step']-30000,action_loss_weighting=plan['action_loss_weighting'],
            required_corrections=d.TOTAL_EPISODES,required_new_segmented=d.NEW_EPISODES,retained_legacy=d.LEGACY_EPISODES,record_video=False,intermediate_physical_evaluations=False)
        if error:row['error']=error
        e.write_json(root/'status.json',row)
    publish('waiting_for_collection')
    try:
        training.verify(plan)
        while True:
            cm=e.read_json(Path(args.collection)/'manifest.json')
            if cm['status']=='complete':break
            if cm['status']=='halted':raise ValueError('Correction collection halted; do not train on incomplete quota')
            time.sleep(15)
        publish('auditing_fifty_new_plus_twenty_legacy')
        base=e.read_json(plan['parent_manifest'])['contract'];rows=d.validate_rows(base,cm)
        counts=[]
        for row in rows:
            ep=DaggerEpisode(row)
            if ep.supervised_count!=row['expert_steps']:raise ValueError('Expert-only label count mismatch')
            counts.append(ep.supervised_count)
        e.write_json(root/'collection_admission.json',dict(
            manifest_sha256=e.sha256_file(Path(args.collection)/'manifest.json'),valid_corrections=d.TOTAL_EPISODES,new_segmented_episodes=d.NEW_EPISODES,legacy_episodes=d.LEGACY_EPISODES,
            verified_intervention_segments=sum(r['verified_segment_count'] for r in rows),
            actual_model_handbacks=sum(r['handback_count'] for r in rows),
            expert_supervised_steps=sum(counts),episode_expert_counts=counts,all_original_splits_preserved=True))
        publish('waiting_for_exclusive_gpu');deadline=time.monotonic()+3600
        while gpu_lock.owner() is not None or admission.gpu_jobs(admission.processes()):
            if time.monotonic()>deadline:raise TimeoutError('Exclusive GPU admission deadline')
            time.sleep(10)
        training.verify(plan);gpu_lock.acquire(args.gpu_lock_owner,max_wait=0);held=True
        env=os.environ.copy();env.update(e.read_json(Path(args.collection)/'plan.json')['child_env'])
        # Launch the original model/optimizer ABI environment, not the Isaac ABI.
        command=[str(ROOT/'.venvs/wam-fixed/bin/python'),'-B',str(ROOT/'scripts/train_precision_dagger_weighted50k_v18.py'),
            '--parent-full-state',str(Path(args.parent_full_state).resolve()),'--collection',str(Path(args.collection).resolve()),
            '--out',str(root/'train'),'--cache-resume-receipt',args.cache_resume_receipt,'--gpu-lock-owner',args.gpu_lock_owner,'--execute']
        e.write_json(root/'training_command.json',dict(command=command))
        publish('training_child_running')
        log('50条新分段＋20条旧完整恢复独立准入完成，启动安全边界完整状态，纠正动作3倍权重＋160原TRAIN/70纠正混合追加20k，无中途物理测试。','dagger_training')
        item=dict(seed='training',out=str(root/'train'),command=command)
        # A training run can exceed the physical evaluator's 64-minute timeout.
        with (root/'training.log').open('xb') as output:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait()
            except BaseException:
                import signal
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGINT)
                    try:process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=30)
                raise
        final=e.read_json(root/'train/status.json')
        if code!=0 or final.get('status')!='completed' or final.get('step')!=50000:
            raise ValueError('Additional20k training not complete; preserve all state')
        training.verify(plan)
        for name in ('checkpoint_last.pt','checkpoint_step_050000.pt'):
            path=root/'train'/name;receipt=e.read_json(path.with_name(path.name+'.sha256.json'))
            if receipt['step']!=50000 or receipt['sha256']!=e.sha256_file(path) or receipt['size']!=path.stat().st_size:
                raise ValueError('50k output checkpoint identity differs')
        e.write_json(root/'status.json',dict(schema=d.SCHEMA,status='completed',step=50000,target_steps=50000,
            additional_updates=20000,valid_corrections=d.TOTAL_EPISODES,new_segmented_episodes=d.NEW_EPISODES,legacy_episodes=d.LEGACY_EPISODES,original_train_episodes=160,
            training_out=str(root/'train'),action_loss_weighting=plan['action_loss_weighting'],weighted_stage_start_step=plan['resume_step'],weighted_stage_updates=50000-plan['resume_step'],record_video=False,physical_success_rate=None,
            intermediate_physical_evaluations=False,final_checkpoint_integrity_verified=True))
        log('50新分段＋20旧完整＋追加20k完成：160原TRAIN/70纠正，正式50000步；最终模型/full-state哈希通过，未自动物理评测。','dagger_training')
        return 0
    except BaseException as exc:
        publish('halted',f'{type(exc).__name__}: {exc}');raise
    finally:
        if held:
            if admission.gpu_jobs(admission.processes()):e.write_json(root/'owner_retained_guard.json',dict(owner=args.gpu_lock_owner))
            else:gpu_lock.release(args.gpu_lock_owner)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-full-state',required=True)
    p.add_argument('--collection',required=True);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--gpu-lock-owner',default='zhuhai-dagger-weighted50k-20261011-v16');p.add_argument('--cache-resume-receipt',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    # Child plan binds all source files before the waiting service is published.
    child_args=argparse.Namespace(**vars(args));child_args.out=Path(args.out)/'train'
    plan=training.make_plan(child_args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False));return 0
    return execute(args,plan)


if __name__=='__main__':raise SystemExit(main())
