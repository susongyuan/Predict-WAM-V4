#!/usr/bin/env python3
"""Independent task2 Precision-WAM. Plan by default; --execute needs the GPU lock."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from harness import zhuhai_precision_v2_training as h
from scripts.progress1_log import log


def learning_rate_for_update(training, completed_updates):
    """Use the fixed full schedule, including after a 20k/25k resume."""
    step=completed_updates;steps=training['max_steps'];warmup=training['warmup_steps']
    if type(step) is not int or not 0<=step<steps:raise ValueError('Invalid absolute optimizer step')
    factor=min(1.,(step+1)/max(1,warmup)) if step<warmup else .1+.9*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))/2
    return training['learning_rate']*factor


def restore_rng(saved,rng,torch_module):
    step=saved.get('step')
    if type(step) is not int or step<0:raise ValueError('Integer saved optimizer boundary required')
    rng.bit_generator.state=saved['rng']
    torch_module.set_rng_state(saved['torch_rng'])
    torch_module.cuda.set_rng_state_all(saved['cuda_rng'])
    return step


def gradient_evidence(wrapper):
    from harness.wam_hjepa import gradient_evidence as auxiliary_gradients
    import torch
    def norm(module):
        values=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
        return float(torch.stack(values).sum().sqrt()) if values else 0.
    result=auxiliary_gradients(wrapper);branch=wrapper.action_model.precision
    for name in ('clock','residual','patch_proj','attention'):
        result['precision_'+name+'_grad_norm']=norm(getattr(branch,name))
    return result


def validate_tiny_gradients(evidence,settings,completed_updates):
    required=['shared_visual_grad_norm','precision_clock_grad_norm','precision_residual_grad_norm']
    if settings['use_idm']:required.append('idm_head_grad_norm')
    if settings['multi_horizon']:required.append('future8_offset_grad_norm')
    # Zero-initialized output deliberately blocks dense attention on update 1.
    # Update 2 must open its path; admitting only the old auxiliary heads is insufficient.
    if completed_updates>=1:required+=['precision_patch_proj_grad_norm','precision_attention_grad_norm']
    if any(not math.isfinite(evidence[k]) or evidence[k]<=0 for k in required):
        raise ValueError('Precision/auxiliary gradient admission failed: '+','.join(required))


def completion_status(mode,step,target_steps):
    admitted=mode=='tiny' and step==target_steps==2
    return dict(status='completed' if step==target_steps else 'stopped_at_safe_boundary',step=step,target_steps=target_steps,
        physical_success_rate=None,tiny_run_cannot_establish_competition_rate=mode=='tiny',
        auxiliary_gradient_admission_passed=admitted,precision_gradient_admission_passed=admitted)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', default=str(h.DEFAULT_TASKS))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('plan','tiny','train'), default='plan')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--variant', choices=h.VARIANTS, default='combined')
    p.add_argument('--steps', type=int, default=h.DEFAULT_STEPS)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261009)
    p.add_argument('--cache-root', default='data/zhuhai_hjepa_hybrid_features_20261009')
    p.add_argument('--preparation-root', default='data/zhuhai_hjepa_preparation_20261009')
    p.add_argument('--cache-budget-bytes', type=int, default=10**12)
    p.add_argument('--cache-ram-budget-bytes', type=int, default=2*1024**3)
    p.add_argument('--minimum-free-bytes', type=int, default=100*1024**3)
    p.add_argument('--source-count', type=int, default=100)
    p.add_argument('--randomized-count', type=int, default=100)
    p.add_argument('--initialization', default=str(h.INITIALIZATION))
    p.add_argument('--initialization-sha256', default=h.INITIALIZATION_SHA)
    p.add_argument('--gpu-lock-owner')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-updates', type=int)
    return p


def reserve(args, *, startup=False):
    h.verify_owner(args.gpu_lock_owner)
    h.check_disk(args.out, minimum=args.minimum_free_bytes)
    memory = dict((k.rstrip(':'), int(v.split()[0])*1024) for k,v in
                  (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()))
    floor = 7*h.GIB if startup else max(2*h.GIB, .10*memory['MemTotal'])
    if memory['MemAvailable'] < floor:
        raise RuntimeError('RAM reserve reached; preserve evidence and stop this training process')
    if memory['SwapTotal'] and 1-memory['SwapFree']/memory['SwapTotal'] >= .95:
        raise RuntimeError('95% swap reserve reached')


def save_checkpoint(out, wrapper, contract, step, *, optimizer=None, rng=None):
    import torch
    from scripts.train_wam_v3 import cpu_tree
    h.validate_training_contract(contract)
    milestone = optimizer is None
    path = out/(f'checkpoint_step_{step:06d}.pt' if milestone else 'checkpoint_last.pt')
    if milestone and path.exists(): raise FileExistsError('Preserve existing immutable milestone')
    value = dict(format=h.FORMAT, step=step, model_only=milestone,
        model_state_dict=cpu_tree(wrapper.action_model.state_dict()), contract=contract, contract_sha256=h.digest(contract))
    if not milestone:
        value.update(optimizer_state_dict=cpu_tree(optimizer.state_dict()), rng=rng,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temp = path.with_name(path.name+'.tmp')
    torch.save(value,temp); temp.replace(path)
    h.write_json(path.with_name(path.name+'.sha256.json'), dict(sha256=h.sha256_file(path),
        size=path.stat().st_size,step=step,model_only=milestone))
    return path


def diagnostic(wrapper, datasets):
    import torch
    from torch.utils.data import default_collate
    from scripts.train_wam_v3 import isolated_rng
    result = {}; was_training=wrapper.training; wrapper.eval()
    try:
        with torch.no_grad(), isolated_rng(20261008,wrapper.device):
            for task, ds in datasets.items():
                loss, info, _ = wrapper(default_collate([ds[len(ds)//2]]))
                if not torch.isfinite(loss): raise ValueError('Nonfinite diagnostic')
                result[task] = dict(loss=float(loss),**info)
        return result
    finally:
        wrapper.train(was_training)


def execute(args, plan, initial):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from harness import zhuhai_training as zt
    from harness.wam_behavior import load_upstream
    from harness.zhuhai_dataset import FrozenDinoProducer, FrameFeatureStore, PREPROCESS_SHA256
    from harness.zhuhai_hybrid_cache import HybridFeatureCache
    base=plan['contract']; cfg=base['config']; args.out=h.resolve(args.out)
    steps=cfg['training']['max_steps']; save_every=cfg['training']['save_every']
    reserve(args,startup=True); h.verify_code_hashes(base['source_files'])
    previous = h.read_json(args.out/'manifest.json') if args.resume else None
    if args.resume:
        h.validate_training_contract(previous['contract'])
        if previous['contract_sha256'] != h.digest(previous['contract']): raise ValueError('Resume manifest hash mismatch')
    h.write_json(args.out/'plan.json', plan, immutable=True)
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(.90)
    h.write_json(args.out/'status.json',dict(status='preparing_TRAIN_statistics',step=0,target_steps=steps,physical_success_rate=None))
    encoder_path=h.resolve(cfg['model']['vision_encoder']['checkpoint_path'])
    encoder_sha=base['cache']['encoder_sha256']
    if h.sha256_file(encoder_path/'model.safetensors') != encoder_sha: raise ValueError('DINO changed')
    _,runner=load_upstream()
    encoder,hidden,registers,patch=runner.ModelFactory.create_vision_encoder(str(encoder_path),dtype=torch.bfloat16,device='cuda')
    if (hidden,registers,patch)!=(1024,4,16): raise ValueError('Unexpected DINO geometry')
    cache=HybridFeatureCache(args.cache_root,budget_bytes=args.cache_budget_bytes,min_free_bytes=args.minimum_free_bytes)
    binding=base['hjepa']['cache_binding']
    disk=FrameFeatureStore(h.SourceBoundFeatureCache(cache,binding),FrozenDinoProducer(zt.FrozenFeatureEncoder(encoder,device='cuda'),encoder),encoder_sha256=encoder_sha)
    store=zt.BoundedRamFeatures(disk,budget_bytes=args.cache_ram_budget_bytes)
    datasets=h.build_datasets(base['splits'],store,cfg,variant=args.variant)
    prep=h.preparation_cache(args.preparation_root,binding)
    feature_identity=dict(encoder_sha256=encoder_sha,preprocess_sha256=PREPROCESS_SHA256)
    check=lambda:reserve(args)
    statistics=prep.statistics(datasets['train'],cfg,feature_identity=feature_identity,builder=zt.fit_train_statistics,check=check)
    bank=prep.conditions(datasets,cfg,statistics,builder=zt.install_conditions,check=check)
    idm=prep.idm_statistics(datasets['train'],cfg,feature_identity=feature_identity,check=check)
    h.write_json(args.out/'preparation_receipt.json',prep.receipt())
    # Preparation can take hours; reject drift before lazily importing the model.
    h.verify_code_hashes(base['source_files'])
    wrapper,notes=h.build_wrapper(cfg,statistics,variant=args.variant,idm_statistics=idm,device='cuda',initialization=initial)
    del initial
    contract=dict(base,statistics=statistics,statistics_sha256=h.digest(statistics),conditions=bank['conditions'],
        condition_bank=bank,initialization_load=notes,idm_statistics=idm,idm_statistics_sha256=h.digest(idm),
        train_frame_counts={t:len(ds) for t,ds in datasets['train'].items()},
        condition_source='successful native16 TRAIN-only head CLS terminal-minus-start means; robot/statistics/features bound')
    h.validate_training_contract(contract)
    for name,value in [('statistics',statistics),('condition_bank',bank),('idm_statistics',idm)]:
        h.write_json(args.out/(name+'.json'),value,immutable=True)
    parameters=[p for p in wrapper.action_model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['training']['learning_rate'],betas=tuple(cfg['training']['betas']),weight_decay=cfg['training']['weight_decay'])
    rng=np.random.default_rng(args.seed);start=0
    if args.resume:
        if previous['contract']!=contract: raise ValueError('Resume source/config/model/statistics changed')
        last=args.out/'checkpoint_last.pt'; receipt=h.read_json(last.with_name(last.name+'.sha256.json'))
        if receipt['sha256']!=h.sha256_file(last) or receipt['size']!=last.stat().st_size or receipt.get('model_only') is not False: raise ValueError('Resume checkpoint changed')
        saved=torch.load(last,map_location='cpu',weights_only=False,mmap=True)
        if saved.get('model_only') is not False or saved.get('format')!=h.FORMAT or saved.get('contract')!=contract or saved.get('contract_sha256')!=h.digest(contract):
            raise ValueError('Exact full-state Precision-WAM checkpoint required for resume')
        if saved.get('step')!=receipt.get('step'):raise ValueError('Resume optimizer boundary differs from receipt')
        wrapper.action_model.load_state_dict(saved['model_state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer_state_dict'])
        start=restore_rng(saved,rng,torch);del saved
    else:
        h.write_json(args.out/'manifest.json',dict(contract=contract,contract_sha256=h.digest(contract)),immutable=True)
        save_checkpoint(args.out,wrapper,contract,0)
    stopped={'value':False}
    def stop(signum,frame): stopped['value']=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    end=min(steps,args.max_updates) if args.max_updates is not None else steps
    if start>end: raise ValueError('Resume boundary is before the saved optimizer step')
    tasks=list(datasets['train']);step=start;started=time.monotonic();wrapper.train()
    if not args.resume: h.write_json(args.out/'initial_diagnostic.json',diagnostic(wrapper,datasets['train']),immutable=True)
    try:
        while step<end and not stopped['value']:
            reserve(args);items=[]
            for _ in range(args.batch_size):
                task=tasks[int(rng.integers(len(tasks)))];ds=datasets['train'][task];items.append(ds[int(rng.integers(len(ds)))])
            batch=default_collate(items);del items  # Do not retain another B4 of dense labels through backward.
            for group in optimizer.param_groups:group['lr']=learning_rate_for_update(cfg['training'],step)
            optimizer.zero_grad(set_to_none=True);loss,info,outputs=wrapper(batch,cfg['training']['auxiliary_loss'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward(); evidence=gradient_evidence(wrapper)
            grad=torch.nn.utils.clip_grad_norm_(parameters,cfg['training']['grad_clip_norm'])
            projection=wrapper.action_model.task_cond_proj
            if (not torch.isfinite(grad) or float(grad)<=0 or any(p.grad is not None for p in projection.base.parameters())
                    or any(p.requires_grad or p.grad is not None for p in encoder.parameters())):
                raise ValueError('Invalid gradients or frozen base/DINO updated')
            if args.mode=='tiny':validate_tiny_gradients(evidence,wrapper.settings,step)
            optimizer.step();step+=1
            optimizer.zero_grad(set_to_none=True)  # No accumulation: free gradients before DINO reads/checkpoint/diagnostics.
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),**info,**evidence,
                elapsed_seconds=time.monotonic()-started,cache_stats=dict(cache.stats),ram_cache_stats=dict(store.stats),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            del batch,loss,outputs  # Release the training payload before checkpoint copies/diagnostics.
            with (args.out/'log.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            h.write_json(args.out/'status.json',dict(status='training',step=step,target_steps=steps,last_loss=row['loss'],elapsed_seconds=row['elapsed_seconds']))
            if step==1 or step%25==0:print(json.dumps(row),flush=True)
            if step%save_every==0 or step==end or stopped['value']:
                h.verify_code_hashes(contract['source_files']);log(f'餐具Precision-WAM保存step={step}，固定schedule={steps}，无本入口自动策略评测',actor='train_precision')
                save_checkpoint(args.out,wrapper,contract,step);save_checkpoint(args.out,wrapper,contract,step,optimizer=optimizer,rng=rng.bit_generator.state)
                h.write_json(args.out/f'diagnostic_{step:06d}.json',dict(train=diagnostic(wrapper,datasets['train']),val=diagnostic(wrapper,datasets['val'])),immutable=True)
        h.write_json(args.out/'status.json',completion_status(args.mode,step,steps))
        return 0
    except BaseException as exc:
        h.write_json(args.out/'status.json',dict(status='failed',step=step,target_steps=steps,error=f'{type(exc).__name__}: {exc}',evidence_preserved=True))
        raise


def main(argv=None):
    args=parser().parse_args(argv);h.validate_request(args)
    if args.execute: reserve(args,startup=True)
    plan,initial=h.make_plan(args)
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 0 if plan['status']=='cpu_preflight_completed' else 2
    if plan['status']!='cpu_preflight_completed':raise ValueError('Incomplete code/data preflight')
    args.out=h.resolve(args.out)
    with h.output_lock(args.out):
        if any(p.name!='.hjepa_training.lock' for p in args.out.iterdir()) and not args.resume:raise FileExistsError('Preserve nonempty output; choose new run')
        if args.resume and not (args.out/'manifest.json').is_file():raise FileNotFoundError('Resume needs committed manifest')
        args.out.mkdir(parents=True,exist_ok=True)
        try:return execute(args,plan,initial)
        except BaseException as exc:
            state=h.read_json(args.out/'status.json') if (args.out/'status.json').exists() else {}
            h.write_json(args.out/'status.json',dict(state,status='failed',error=f'{type(exc).__name__}: {exc}',evidence_preserved=True));raise

if __name__=='__main__':raise SystemExit(main())
