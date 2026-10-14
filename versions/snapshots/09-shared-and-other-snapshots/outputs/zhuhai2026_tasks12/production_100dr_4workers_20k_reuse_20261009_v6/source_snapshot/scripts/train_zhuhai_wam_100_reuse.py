#!/usr/bin/env python3
"""Versioned 100+100/task trainer; reuse exact TRAIN preparation across milestones."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_zhuhai_wam as base
from scripts.train_zhuhai_wam_reuse import ReuseHooks,sha256,write_receipt


def validate_request(args):
    if args.allow_small_data or (args.source_count,args.randomized_count)!=(100,100):
        raise ValueError('Complete 100 source + 100 DR per task required')
    tasks=Path(args.tasks);tasks=tasks if tasks.is_absolute() else ROOT/tasks
    specs=json.loads(tasks.read_text()).get('tasks')
    if (not isinstance(specs,list) or len(specs)!=2 or
            {r.get('task') for r in specs}!={'zhuhai_task1','zhuhai_task2'} or
            any(not r.get('source_manifest') or not r.get('randomized_manifest') for r in specs)):
        raise ValueError('Both complete task inventories required; core trainer verifies every episode')
    if args.mode!='plan' and not args.execute_independent:
        raise ValueError('Versioned execution flag required')


def main():
    parser=base.parser()
    parser.add_argument('--preparation-root',type=Path,required=True)
    parser.add_argument('--execute-independent',action='store_true')
    args=parser.parse_args();validate_request(args)
    from harness.zhuhai_preparation_reuse import PreparationReuse
    paths=[Path(__file__).resolve(),ROOT/'scripts/train_zhuhai_wam_reuse.py',
        ROOT/'scripts/train_zhuhai_wam.py',ROOT/'harness/zhuhai_preparation_reuse.py',
        ROOT/'harness/zhuhai_training.py']
    metadata=dict(format='zhuhai-100dr-exact-preparation-reuse-v1',
        artifact_root=str(args.preparation_root.resolve()),sources_sha256={str(p):sha256(p) for p in paths},
        numerical_algorithm='original TRAIN-only fit_train_statistics/install_conditions unchanged',
        independent_execution=True,current_production_pipeline_hot_switched=False)
    cache=PreparationReuse(args.preparation_root)
    receipt=dict(requested_mode=args.mode,out=str(args.out.resolve()),execution_metadata=metadata,status='starting')
    try:
        with ReuseHooks(base.zt,cache,metadata=metadata,
                owner_check=lambda:base.verify_owner(args.gpu_lock_owner)):
            code=base.train(args)
        receipt.update(status='returned',exit_code=code)
        return code
    except BaseException as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['preparation']=cache.receipt()
        receipt['execution_sources_unchanged']=all(sha256(path)==digest for path,digest in metadata['sources_sha256'].items())
        target=write_receipt(args.preparation_root,receipt)
        print(json.dumps(dict(preparation_execution_receipt=str(target),status=receipt['status'])),flush=True)


if __name__=='__main__':raise SystemExit(main())
