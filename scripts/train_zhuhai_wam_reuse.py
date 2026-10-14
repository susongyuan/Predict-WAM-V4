#!/usr/bin/env python3
"""Independent, opt-in adapter for reusing identical native16 TRAIN preparation.

The running production pipeline still calls train_zhuhai_wam.py unchanged.
This entry point changes no model, loss, split, normalization algorithm, or
success criterion. Its numerical/GPU integration still needs validation after
complete collection. No automatic handoff or hot patch of a running job occurs.
"""
from __future__ import annotations
import argparse
from contextlib import AbstractContextManager
from copy import deepcopy
from datetime import datetime
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import uuid
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
VERSION='zhuhai-train-preparation-reuse-candidate-v1'
_PATCH_LOCK=threading.Lock()


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def execution_sources():
    paths=[Path(__file__),ROOT/'harness/zhuhai_preparation_reuse.py',
           ROOT/'scripts/train_zhuhai_wam.py',ROOT/'harness/zhuhai_training.py']
    return {str(p.resolve()):sha256(p) for p in paths}


def validate_recovery_source(old_out,new_out):
    """A prep-only interruption can be retried in a NEW directory, never erased."""
    old_out,new_out=Path(old_out).resolve(),Path(new_out).resolve()
    if old_out==new_out or old_out in new_out.parents:
        raise ValueError('Recovery requires a separate new attempt, not the original directory or its child')
    plan=old_out/'plan.json'
    if not plan.is_file() or (old_out/'manifest.json').exists():
        raise ValueError('Only a plan-only preparation interruption can use this recovery path')
    if list(old_out.glob('checkpoint*.pt')) or (old_out/'log.jsonl').exists():
        raise ValueError('Possible training state exists; use exact full-state resume, not prep recovery')
    value=json.loads(plan.read_text())
    contract=value.get('contract',{})
    if not contract.get('source_files') or not contract.get('sources') or not contract.get('config'):
        raise ValueError('Incomplete original preflight contract')
    for path,expected in contract['source_files'].items():
        if sha256(path)!=expected:raise ValueError(f'Recovery source code changed: {path}')
    sources=contract['sources']
    if not isinstance(sources,dict) or set(sources)!={'zhuhai_task1','zhuhai_task2'}:
        raise ValueError('Recovery requires both registered tasks with complete source/DR inventories')
    for task,source in sources.items():
        if (not isinstance(source,dict) or source.get('quotas')!={'source':100,'randomized':500}
                or source.get('families')!=100 or len(source.get('manifests',[]))!=2):
            raise ValueError(f'Incomplete source/DR recovery inventory for {task}')
        for item in source['manifests']:
            path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
            if sha256(path)!=item['sha256']:raise ValueError(f'Recovery data manifest changed: {path}')
    extra=contract['config'].get('training_preparation_execution',{}).get('sources_sha256',{})
    for path,expected in extra.items():
        if sha256(path)!=expected:raise ValueError(f'Recovery adapter source changed: {path}')
    if new_out.exists() and any(new_out.iterdir()):
        raise FileExistsError('New recovery output must be empty; original attempt is preserved')
    return {'source_output':str(old_out),'source_plan_sha256':sha256(plan),
            'new_output':str(new_out),'mode':'new_attempt_reusing_only_committed_preparation',
            'original_modified':False}


class ReuseHooks(AbstractContextManager):
    """Process-local, scoped hooks; original functions and source files survive."""
    def __init__(self,zt,cache,*,metadata,owner_check=lambda:None):
        self.zt,self.cache=zt,cache
        self.metadata=deepcopy(metadata);self.owner_check=owner_check;self.original={}

    def __enter__(self):
        if not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError('Nested/concurrent training preparation adapters are not allowed')
        try:
            originals={name:getattr(self.zt,name) for name in ('config_for','fit_train_statistics','install_conditions')}
            if not all(callable(value) for value in originals.values()):raise TypeError('Invalid base trainer hooks')
            self.original=originals
        except BaseException:
            _PATCH_LOCK.release();raise
        original_config=self.original['config_for'];original_fit=self.original['fit_train_statistics']
        original_bank=self.original['install_conditions']
        @wraps(original_config)
        def config(*args,**kwargs):
            value=original_config(*args,**kwargs)
            value['training_preparation_execution']=deepcopy(self.metadata)
            return value
        @wraps(original_fit)
        def statistics(datasets,cfg,*,feature_identity,check=lambda:None):
            return self.cache.statistics(datasets,cfg,feature_identity=feature_identity,
                                         builder=original_fit,check=check)
        @wraps(original_bank)
        def conditions(datasets,cfg,statistics):
            return self.cache.conditions(datasets,cfg,statistics,builder=original_bank,
                                         check=self.owner_check)
        self.zt.config_for=config;self.zt.fit_train_statistics=statistics;self.zt.install_conditions=conditions
        return self

    def __exit__(self,*exc):
        try:
            for name,value in self.original.items():setattr(self.zt,name,value)
        finally:
            self.original.clear();_PATCH_LOCK.release()
        return False


def write_receipt(root,receipt):
    directory=Path(root)/'dispatches';directory.mkdir(parents=True,exist_ok=True)
    target=directory/(uuid.uuid4().hex+'.json')
    with target.open('x') as handle:
        json.dump(receipt,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    return target


def build_parser():
    from scripts import train_zhuhai_wam as base
    parser=base.parser()
    parser.description=__doc__
    parser.add_argument('--preparation-root',type=Path,required=True,
        help='Independent persistent statistics/VTT bundle directory; not a replacement for the 1TB dense cache')
    parser.add_argument('--execute-independent',action='store_true',
        help='Explicitly opt into this separate trainer after complete collection and GPU validation; never hot-switches production')
    parser.add_argument('--recover-preparation-from',type=Path,
        help='Validate a preserved plan-only failed attempt, but execute in NEW --out; cannot resume optimizer state')
    return parser


def main(argv=None):
    from scripts import train_zhuhai_wam as base
    from scripts.progress1_log import log
    args=build_parser().parse_args(argv)
    if args.mode!='plan' and not args.execute_independent:
        raise ValueError('GPU training requires explicit --execute-independent; current frozen production is not changed')
    if args.allow_small_data or (args.source_count,args.randomized_count)!=(100,500):
        raise ValueError('This independent production entry point cannot bypass the complete 100+500 per-task quota')
    task_file=Path(args.tasks);task_file=task_file if task_file.is_absolute() else ROOT/task_file
    task_specs=json.loads(task_file.read_text()).get('tasks')
    if (not isinstance(task_specs,list) or len(task_specs)!=2
            or {item.get('task') for item in task_specs}!={'zhuhai_task1','zhuhai_task2'}
            or any(not item.get('source_manifest') or not item.get('randomized_manifest') for item in task_specs)):
        raise ValueError('Both complete task inventories are required; base trainer will validate all 1200 episodes')
    if args.recover_preparation_from and args.resume:
        raise ValueError('A fresh preparation retry and full-state --resume are different operations')
    recovery=(validate_recovery_source(args.recover_preparation_from,args.out)
              if args.recover_preparation_from else None)
    from harness.zhuhai_preparation_reuse import PreparationReuse
    sources=execution_sources();root=args.preparation_root.resolve()
    metadata={'format':VERSION,'artifact_root':str(root),'independent_execution':True,
        'sources_sha256':sources,'numerical_algorithm':'original TRAIN-only fit_train_statistics/install_conditions unchanged',
        'reuse_basis':'successful TRAIN content, robot/data, encoder/preprocess, views/history and algorithm hashes',
        'current_production_pipeline_hot_switched':False}
    cache=PreparationReuse(root)
    receipt={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'format':VERSION,'requested_mode':args.mode,'out':str(args.out.resolve()),
        'execution_metadata':metadata,'recovery':recovery,'status':'starting',
        'learned_policy_success_rate':None,'gpu_integration_validated_by_this_dispatch':False}
    log(f'独立统计/VTT复用候选：mode={args.mode}，out={args.out}；不修改当前冻结pipeline，仍先验1200条',
        actor='train-preparation-reuse')
    try:
        with ReuseHooks(base.zt,cache,metadata=metadata,
                       owner_check=lambda:base.verify_owner(args.gpu_lock_owner)):
            code=base.train(args)
        receipt.update(status='cpu_plan_returned' if args.mode=='plan' else 'independent_trainer_returned',return_code=code)
        return code
    except BaseException as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['preparation']=cache.receipt()
        receipt['execution_sources_unchanged']=all(sha256(path)==value for path,value in sources.items())
        path=write_receipt(root,receipt)
        print(json.dumps({'preparation_execution_receipt':str(path),'status':receipt['status']},ensure_ascii=False),flush=True)


if __name__=='__main__':raise SystemExit(main())
