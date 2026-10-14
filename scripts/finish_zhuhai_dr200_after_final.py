#!/usr/bin/env python3
"""Mandatory DR200 after completed30k and its final fixed5 attempts.

Reuse the original one-worker queue/guards without changing any frozen source.
Only its read-only completion predicate is rebound in this private process;
all prior failed queues and partial25k evaluations remain untouched.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_zhuhai_task1_dr200_concurrent as q
from scripts import eval_zhuhai_precision_fixed_attempts as evaluation
ORIGIN=q.BASE/'wam_precision_v2_task2_30k_20261010_v1'
FINAL=q.BASE/'wam_precision_v2_task2_final30k_trainingfirst_20261010'
OUT=q.BASE/'production_tasks12_dr200_collection_only_20261010_serial_r3'
OWNER='zhuhai-task1-dr200-serial-20261010-r3'


def completed_training_and_final(run):
    q.require(Path(run).resolve()==ORIGIN.resolve(),'Exact original training artifacts required')
    train=q.read(ORIGIN/'train/status.json')
    if train.get('status')=='failed':raise RuntimeError('Actual training failed; no collection launch')
    if train.get('status')!='completed':return False
    q.require(train.get('step')==train.get('target_steps')==30000,'Full30k training required')
    summary=FINAL/'summary.json'
    if not summary.is_file():return False
    value=q.read(summary)
    if value.get('status')=='halted_infrastructure_error':raise RuntimeError('Final evaluation infrastructure error; retain pending collection without launching')
    if value.get('status')!='completed':return False
    receipt=q.read(ORIGIN/'train/checkpoint_step_030000.pt.sha256.json')
    checked=evaluation.verify_summary(summary,30000,list(evaluation.SEEDS),receipt['sha256'])
    q.require(checked['planned_episodes']==5,'All five attempts must be preserved; zero successes does not block collection')
    return True


def main(argv=None):
    parser=q.parser()
    parser.set_defaults(training_run=ORIGIN,out=OUT,gpu_lock_owner=OWNER)
    args=parser.parse_args(argv)
    q.require(args.training_run==ORIGIN and args.out==OUT and args.gpu_lock_owner==OWNER,'Use this fresh explicitly bound mandatory collection queue')
    q.pipeline_finished=completed_training_and_final
    if not args.execute:
        import json
        value=q.review(args);value.update(completion_predicate='true train30000 + verified final fixed5; ignores superseded failed controller snapshots',final_evaluation=str(FINAL),mandatory_before_optional50k=True)
        print(json.dumps(value,ensure_ascii=False,indent=2));return 0
    return q.execute_serial(args)


if __name__=='__main__':raise SystemExit(main())
