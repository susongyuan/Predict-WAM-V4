#!/usr/bin/env python3
"""Explicit task2-only 30k queue: admission, 20/25/30k development, final heldout."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.progress1_log_task2_repair import log
from harness import zhuhai_hjepa_training as h
TRAIN=ROOT/'scripts/train_zhuhai_wam_hjepa.py'
EVAL=ROOT/'scripts/eval_zhuhai_wam_hjepa.py'
PYTHON=ROOT/'.venvs/wam-fixed/bin/python'
DEV_SEEDS=list(range(932101,932106))
FINAL_SEEDS=list(range(932001,932011))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tasks',default=str(h.DEFAULT_TASKS))
    p.add_argument('--gpu-lock-owner')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--cache-root',default='data/zhuhai_hjepa_hybrid_features_20261009')
    p.add_argument('--preparation-root',default='data/zhuhai_hjepa_preparation_20261009')
    return p


def build_plan(args):
    out=h.no_symlink(h.resolve(args.out))
    if out==ROOT or out in ROOT.parents or (ROOT/'outputs') not in out.parents:
        raise ValueError('Independent pipeline must be below project outputs')
    owner=args.gpu_lock_owner or '<external-owner-required>'
    stages=[]
    def train(name,mode,step,resume=False):
        target=out/('tiny_admission' if mode=='tiny' else 'train')
        cmd=[str(PYTHON),'-B',str(TRAIN),'--tasks',str(h.resolve(args.tasks)),'--out',str(target),
            '--mode',mode,'--execute','--variant','combined','--steps','30000','--batch-size','4',
            '--save-every','5000','--source-count','100','--randomized-count','100',
            '--cache-root',str(h.resolve(args.cache_root)),'--preparation-root',str(h.resolve(args.preparation_root)),
            '--gpu-lock-owner',owner,'--max-updates',str(step)]
        if resume:cmd.append('--resume')
        stages.append(dict(name=name,kind='train',mode=mode,step=step,out=str(target),command=cmd))
    def evaluate(name,step,seeds):
        target=out/name
        cmd=[str(PYTHON),'-B',str(EVAL),'--checkpoint',str(out/'train'/f'checkpoint_step_{step:06d}.pt'),
             '--task',h.TASK,'--seeds',*map(str,seeds),'--out',str(target),'--max-steps','2400',
             '--gpu-lock-owner',owner,'--execute']
        stages.append(dict(name=name,kind='eval',step=step,seeds=seeds,out=str(target),command=cmd))
    train('tiny_admission','tiny',2)
    for step in (20000,25000,30000):
        train(f'train_to_{step}','train',step,resume=step>20000)
        evaluate(f'development_{step}',step,DEV_SEEDS)
    evaluate('final_30000',30000,FINAL_SEEDS)
    paths=[Path(__file__),TRAIN,EVAL,ROOT/'harness/zhuhai_hjepa_training.py',ROOT/'harness/wam_hjepa.py',ROOT/'harness/zhuhai_hjepa_dataset.py']
    return dict(schema='zhuhai-hjepa-task2-30k-pipeline-v1',out=str(out),gpu_lock_owner=owner,
        task=h.TASK,stages=stages,source_hashes={str(p):h.sha256_file(p) for p in paths},
        training_updates=30000,batch_size=4,checkpoint_selection='fixed final30k; report all paired development results',
        development_seeds=DEV_SEEDS,final_seeds=FINAL_SEEDS,record_video=False,third_view=False,
        collection_parallel=False,garbage_target=dict(source=100,randomized=200,admitted=False),
        protocol_change='user authorized task2-first training; task1 DR200 is a separate unexecuted collection contract')


def verify_result(stage):
    out=Path(stage['out'])
    if stage['kind']=='train':
        value=h.read_json(out/'status.json')
        if value.get('status') not in ('completed','stopped_at_safe_boundary') or value.get('step')!=stage['step']:
            raise ValueError('Training failed or stopped before required boundary')
        expected=2 if stage['mode']=='tiny' else 30000
        if value.get('target_steps')!=expected:raise ValueError('Optimizer schedule changed')
        if stage['mode']=='tiny' and value.get('auxiliary_gradient_admission_passed') is not True:
            raise ValueError('GPU auxiliary-gradient admission not passed')
        ck=out/f"checkpoint_step_{stage['step']:06d}.pt"
        receipt=h.read_json(ck.with_name(ck.name+'.sha256.json'))
        if receipt.get('step')!=stage['step'] or receipt.get('model_only') is not True or receipt.get('size')!=ck.stat().st_size:
            raise ValueError('Milestone receipt mismatch')
    else:
        value=h.read_json(out/'summary.json');rows=value.get('episodes',[])
        if value.get('status')!='completed' or value.get('checkpoint_step')!=stage['step'] or value.get('planned_episodes')!=len(stage['seeds']):
            raise ValueError('Evaluation incomplete or wrong checkpoint')
        if [r.get('seed') for r in rows]!=stage['seeds'] or any(r.get('task')!=h.TASK or r.get('status')!='completed' or type(r.get('full_task_success')) is not bool for r in rows):
            raise ValueError('Evaluation seed/task/evidence mismatch; keep failures, do not retry')
    return value


def run_stage(stage):
    out=Path(stage['out']).parent
    with (out/(stage['name']+'.log')).open('xb') as stream:
        process=subprocess.Popen(stage['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:return process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=180)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise


def execute(plan,runner=run_stage):
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=True)
    with (out/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another pipeline writer exists') from exc
        if any(p.name!='.pipeline.lock' for p in out.iterdir()):raise FileExistsError('Preserve existing pipeline; no automatic rerun')
        h.verify_owner(plan['gpu_lock_owner']);h.verify_code_hashes(plan['source_hashes'])
        h.write_json(out/'plan.json',plan,immutable=True);done=[]
        for stage in plan['stages']:
            h.verify_owner(plan['gpu_lock_owner']);h.verify_code_hashes(plan['source_hashes'])
            log(f"餐具30k队列启动{stage['name']}；原评分/固定种子、无录像、GPU串行",actor='hjepa_pipeline')
            h.write_json(out/'status.json',dict(status='running',active_stage=stage['name'],completed_stages=done))
            try:
                code=runner(stage)
                if code!=0:raise RuntimeError(f"{stage['name']} exit={code}; preserve evidence and stop")
                verify_result(stage)
            except BaseException as exc:
                h.write_json(out/'status.json',dict(status='failed',active_stage=stage['name'],completed_stages=done,error=f'{type(exc).__name__}: {exc}'))
                log(f"餐具30k队列停止于{stage['name']}：{type(exc).__name__}；不自动重跑失败",actor='hjepa_pipeline')
                raise
            done.append(stage['name'])
            log(f"餐具30k队列完成{stage['name']}，机器回执通过；策略失败同样保留",actor='hjepa_pipeline')
        h.write_json(out/'status.json',dict(status='completed',active_stage=None,completed_stages=done))
    return 0


def main(argv=None):
    args=parser().parse_args(argv);plan=build_plan(args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    if not args.gpu_lock_owner:raise ValueError('External GPU owner required')
    h.verify_owner(args.gpu_lock_owner)
    return execute(plan)

if __name__=='__main__':raise SystemExit(main())
