#!/usr/bin/env python3
"""Reuse completed fixed20k cohort after controller-only interpreter failure.

This wrapper runs in wam-fixed Python (CPU NumPy available). Original training,
model, physical evaluator, safety limiter and every raw episode remain frozen.
No already-attempted episode is rerun; original full20k optimizer state resumes.
"""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_zhuhai_wam_precision_v2_continuation as prior
from scripts import eval_zhuhai_precision_fixed_attempts as evaluation

OUT=prior.BASE/'wam_precision_v2_task2_30k_continuation_20261010_r2'
OWNER='zhuhai-precision-v2-continuation-20261010-r2'
PRIOR=prior.BASE/'wam_precision_v2_task2_30k_continuation_20261010_r1'


def build_plan(args):
    # Capability check BEFORE publication, not after the expensive physical run.
    import numpy
    prior.require(numpy.__version__ is not None,'CPU NumPy is required for verified limiter evidence')
    prior.OUT=OUT;prior.OWNER=OWNER
    plan=prior.build_plan(args)
    path=PRIOR/'development_20000/summary.json'
    receipt=prior.h.read_json(prior.ORIGIN/'train/checkpoint_step_020000.pt.sha256.json')
    summary=evaluation.verify_summary(path,20000,prior.SEEDS,receipt['sha256'])
    prior.require(summary['status']=='completed','Whole original20k fixed cohort must complete before resuming25k')
    stage=plan['stages'][0]
    stage.update(out=str(path.parent),adopt_existing=True,precompleted_summary=True,
        completed_summary_path=str(path),completed_summary_sha256=prior.h.sha256_file(path),
        adoption_semantics='all five original physical attempts retained; no process is started for the already-complete20k cohort')
    prior.replace_flag(stage['command'],'--out',path.parent)
    plan.update(out=str(OUT),continuation_schema='precision-v2-explicit-post20k-controller-environment-continuation-v2',
        prior_controller=str(PRIOR),prior_controller_plan_sha256=prior.h.sha256_file(PRIOR/'plan.json'),
        controller_python=str(Path(sys.executable).resolve()),
        controller_environment_repair='use original wam-fixed Python for NumPy-backed read-only limiter proof; no model/eval/source changes',
        adopted_complete_20k_summary=str(path))
    for p in [Path(__file__),PRIOR/'plan.json',path]:plan['source_hashes'][str(p.resolve())]=prior.h.sha256_file(p)
    return plan


def run_stage(plan,stage):
    if stage.get('precompleted_summary'):
        prior.require(stage['kind']=='eval' and stage['step']==20000,'Only completed20k adoption is allowed')
        path=Path(stage['completed_summary_path'])
        prior.require(prior.h.sha256_file(path)==stage['completed_summary_sha256'],'Previously completed20k summary changed')
        prior.verify_stage(stage)
        prior.h.write_json(Path(plan['out'])/'development_20000_adoption_receipt.json',
            dict(status='verified_existing_five_attempts',summary=str(path),summary_sha256=stage['completed_summary_sha256'],
                 physical_episodes_rerun=0,control_and_scoring_changed=False),immutable=True)
        return 0
    return prior.run_stage(plan,stage)


def main(argv=None):
    prior.OUT=OUT;prior.OWNER=OWNER
    args=prior.parser().parse_args(argv)
    plan=build_plan(args)
    if not args.execute:print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    return prior.execute(plan,runner=run_stage)


if __name__=='__main__':raise SystemExit(main())
