#!/usr/bin/env python3
"""Precision-WAM 30k queue, same original evaluation seeds and no parallel simulator."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_zhuhai_wam_hjepa_pipeline as base
from harness import zhuhai_precision_training as precision
from scripts.progress1_log import log
BASE_VERIFY_RESULT=base.verify_result


def configure():
    base.h=precision;base.log=log
    base.TRAIN=ROOT/'scripts/train_zhuhai_wam_precision.py'
    base.EVAL=ROOT/'scripts/eval_zhuhai_wam_precision.py'
    base.verify_result=verify_result


def verify_result(stage):
    value=BASE_VERIFY_RESULT(stage)
    if stage['kind']=='train':
        if stage['mode']=='tiny' and value.get('precision_gradient_admission_passed') is not True:
            raise ValueError('Precision clock/dense-refiner GPU gradient admission not passed')
        ck=Path(stage['out'])/f"checkpoint_step_{stage['step']:06d}.pt"
        receipt=precision.read_json(ck.with_name(ck.name+'.sha256.json'))
        if receipt.get('sha256')!=precision.sha256_file(ck):raise ValueError('Precision milestone content hash mismatch')
    return value


def build_plan(args):
    configure();plan=base.build_plan(args)
    plan['schema']='zhuhai-precision-wam-task2-30k-pipeline-v1'
    plan['precision']=precision.PRECISION_SPEC
    paths=[ROOT/p for p in precision.EXTENSION_SOURCES]
    plan['source_hashes'].update({str(p):precision.sha256_file(p) for p in paths})
    return plan


def main(argv=None):
    configure();args=base.parser().parse_args(argv);plan=build_plan(args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    if not args.gpu_lock_owner:raise ValueError('External GPU owner required')
    precision.verify_owner(args.gpu_lock_owner);return base.execute(plan)

if __name__=='__main__':raise SystemExit(main())
