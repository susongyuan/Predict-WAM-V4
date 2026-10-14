#!/usr/bin/env python3
"""Train a separate 30k->50k full-state continuation after DAgger50 admission."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from harness import zhuhai_dagger_training as d
from harness import zhuhai_precision_v2_training as h
from scripts import train_zhuhai_wam_precision_v2 as trainer
from scripts.progress1_log import log


def identity(path):
    p = Path(path).resolve()
    receipt = h.read_json(p.with_name(p.name+'.sha256.json'))
    if receipt['size'] != p.stat().st_size or receipt['sha256'] != h.sha256_file(p):
        raise ValueError('Parent checkpoint receipt differs')
    if receipt.get('model_only') is not False or receipt.get('step') != d.START:
        raise ValueError('Exact 30k full-state checkpoint required')
    return dict(path=str(p), **receipt)


def verify(plan):
    h.verify_code_hashes(plan['source_files'])
    if h.sha256_file(plan['parent_manifest']) != plan['parent_manifest_sha256']:
        raise ValueError('Parent manifest changed')
    if h.sha256_file(plan['parent_full_state']['path']) != plan['parent_full_state']['sha256']:
        raise ValueError('30k full-state changed')
    if h.sha256_file(plan['collection_plan']) != plan['collection_plan_sha256']:
        raise ValueError('Correction collection plan changed')


def make_plan(args):
    parent = identity(args.parent_full_state)
    manifest_path = Path(parent['path']).parent/'manifest.json'
    manifest = h.read_json(manifest_path); base = manifest['contract']
    if manifest['contract_sha256'] != h.digest(base): raise ValueError('Parent manifest hash differs')
    h.validate_training_contract(base); h.verify_code_hashes(base['source_files'])
    collection = Path(args.collection).resolve()
    cp = h.read_json(collection/'plan.json')
    if cp['checkpoint']['step'] != d.START or not cp.get('train_diagnostic_skipped_by_user'):
        raise ValueError('Authorized direct DAgger collection required')
    sources = dict(base['source_files'])
    for rel in ('harness/zhuhai_dagger.py','harness/zhuhai_dagger_training.py',
                'scripts/train_precision_dagger20k.py','scripts/wait_precision_dagger20k.py'):
        path = ROOT/rel; sources[str(path)] = h.sha256_file(path)
    return dict(schema=d.SCHEMA, parent_full_state=parent,
        parent_manifest=str(manifest_path), parent_manifest_sha256=h.sha256_file(manifest_path),
        parent_model_only_sha256=cp['checkpoint']['sha256'],
        original_contract_sha256=h.digest(base), collection=str(collection),
        collection_plan=str(collection/'plan.json'), collection_plan_sha256=h.sha256_file(collection/'plan.json'),
        out=str(Path(args.out).resolve()), gpu_lock_owner=args.gpu_lock_owner,
        source_files=sources, start_step=d.START, target_step=d.END,
        sampler=d.SAMPLER, schedule=d.SCHEDULE, original_train_episodes=160,
        required_corrections=50, original_VAL_heldout_unchanged=True,
        record_video=False, intermediate_physical_evaluations=False)


def execute(plan, args):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from harness import zhuhai_training as zt
    from harness.wam_behavior import load_upstream
    from harness.zhuhai_dataset import FrozenDinoProducer, FrameFeatureStore
    from harness.zhuhai_hybrid_cache import HybridFeatureCache
    from harness.zhuhai_dagger import DaggerPrecisionDataset
    out = Path(plan['out']); out.mkdir(parents=True, exist_ok=False)
    verify(plan); trainer.reserve(args, startup=True)
    h.write_json(out/'plan.json', plan, immutable=True)
    base = h.read_json(plan['parent_manifest'])['contract']
    collection_path = Path(plan['collection'])/'manifest.json'
    cm = h.read_json(collection_path); rows = d.validate_rows(base, cm)
    collection = dict(path=str(collection_path), sha256=h.sha256_file(collection_path))
    new = d.contract(base, rows, parent=plan['parent_full_state'], collection=collection, sources=plan['source_files'])
    h.write_json(out/'manifest.json', dict(contract=new,contract_sha256=h.digest(new)), immutable=True)
    h.write_json(out/'status.json', dict(status='loading_30k_full_state',step=30000,target_steps=50000,physical_success_rate=None))
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.90)
    ck = torch.load(plan['parent_full_state']['path'], map_location='cpu', weights_only=False, mmap=True)
    if (ck.get('model_only') is not False or ck.get('step') != 30000
            or ck.get('contract') != base or ck.get('contract_sha256') != h.digest(base)):
        raise ValueError('Parent complete-state contract differs')
    wrapper, notes = h.build_wrapper(new['config'], base['statistics'], variant=base['hjepa']['variant'],
        idm_statistics=base['idm_statistics'], device='cuda', state_dict=ck['model_state_dict'])
    parameters = [p for p in wrapper.action_model.parameters() if p.requires_grad]
    training = new['config']['training']
    optimizer = torch.optim.AdamW(parameters, lr=training['learning_rate'], betas=tuple(training['betas']),weight_decay=training['weight_decay'])
    optimizer.load_state_dict(ck['optimizer_state_dict'])
    # Model-only milestone and full state must describe the identical 30k model.
    model_path = Path(plan['parent_full_state']['path']).parent/'checkpoint_step_030000.pt'
    if h.sha256_file(model_path) != plan['parent_model_only_sha256']: raise ValueError('30k model binding differs')
    milestone = torch.load(model_path,map_location='cpu',weights_only=False,mmap=True)
    if (set(milestone['model_state_dict']) != set(ck['model_state_dict']) or any(
            not torch.equal(v, milestone['model_state_dict'][k]) for k,v in ck['model_state_dict'].items())):
        raise ValueError('Full-state and collection policy weights differ')
    optimizer_steps = [int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
    if not optimizer_steps or any(x != 30000 for x in optimizer_steps):
        raise ValueError('Parent AdamW optimizer update count differs')
    rng = np.random.default_rng(); trainer.restore_rng(ck,rng,torch)
    saved_rng = dict(rng=ck['rng'],torch_rng=ck['torch_rng'],cuda_rng=ck['cuda_rng'],step=30000)
    del ck, milestone
    encoder_path = h.resolve(base['config']['model']['vision_encoder']['checkpoint_path'])
    if h.sha256_file(encoder_path/'model.safetensors') != base['cache']['encoder_sha256']:
        raise ValueError('Frozen DINO changed')
    _,runner=load_upstream()
    encoder,hidden,registers,patch=runner.ModelFactory.create_vision_encoder(str(encoder_path),dtype=torch.bfloat16,device='cuda')
    if (hidden,registers,patch) != (1024,4,16): raise ValueError('DINO geometry differs')
    cache = HybridFeatureCache(base['cache']['root'],budget_bytes=base['cache']['disk_budget_bytes'],min_free_bytes=base['cache']['minimum_free_bytes'])
    disk = FrameFeatureStore(h.SourceBoundFeatureCache(cache,base['hjepa']['cache_binding']),
        FrozenDinoProducer(zt.FrozenFeatureEncoder(encoder,device='cuda'),encoder),encoder_sha256=base['cache']['encoder_sha256'])
    store = zt.BoundedRamFeatures(disk,budget_bytes=base['cache']['ram_budget_bytes'])
    datasets = h.build_datasets(base['splits'],store,base['config'],variant=base['hjepa']['variant'])
    condition = np.asarray(base['conditions']['zhuhai_task2'],np.float32)
    for group in datasets.values():
        for ds in group.values(): ds.condition = condition.copy()
    original = datasets['train']['zhuhai_task2']
    corrections = DaggerPrecisionDataset('zhuhai_task2',rows,store,condition,
        proprio_lags=base['config']['model']['v3']['proprio_lags'],memory_lags=base['config']['model']['v3']['memory_lags'])
    if len(original.episodes)!=160 or len(corrections.episodes)!=50: raise ValueError('Training episode quota differs')
    h.write_json(out/'data_admission.json',dict(original_episodes=160,correction_episodes=50,
        original_anchors=len(original),expert_supervised_anchors=len(corrections),
        policy_context_anchors=sum(ep.expert_start for ep in corrections.episodes),
        correction_supervised_counts=[ep.supervised_count for ep in corrections.episodes],
        normalization_refitted=False,condition_refitted=False,sampler=d.SAMPLER),immutable=True)
    # Encoder construction must not consume the continuing WAM RNG stream.
    trainer.restore_rng(saved_rng,rng,torch);del saved_rng
    h.write_json(out/'restore_receipt.json',dict(step=30000,strict_model_load=True,optimizer_state_restored=True,
        optimizer_steps=sorted(set(optimizer_steps)),rng_restored_after_DINO_initialization=True,
        normalization_unchanged=True,condition_unchanged=True,precision_projections_reset=False),immutable=True)
    stopped={'value':False}
    signal.signal(signal.SIGINT,lambda signum,frame:stopped.update(value=True))
    signal.signal(signal.SIGTERM,lambda signum,frame:stopped.update(value=True))
    step=30000;started=time.monotonic();wrapper.train()
    try:
        while step<50000 and not stopped['value']:
            trainer.reserve(args)
            batch=default_collate(d.mixed_samples(original,corrections,rng))
            for group in optimizer.param_groups: group['lr']=d.rate(step)
            optimizer.zero_grad(set_to_none=True)
            loss,info,outputs=wrapper(batch,training['auxiliary_loss'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite mixed training loss')
            loss.backward(); evidence=trainer.gradient_evidence(wrapper)
            grad=torch.nn.utils.clip_grad_norm_(parameters,training['grad_clip_norm'])
            if (not torch.isfinite(grad) or float(grad)<=0
                    or any(p.grad is not None for p in wrapper.action_model.task_cond_proj.base.parameters())
                    or any(p.requires_grad or p.grad is not None for p in encoder.parameters())):
                raise ValueError('Invalid gradients or frozen model updated')
            optimizer.step();step+=1;optimizer.zero_grad(set_to_none=True)
            row=dict(step=step,additional_step=step-30000,loss=float(loss.detach()),gradient_norm=float(grad),
                learning_rate=optimizer.param_groups[0]['lr'],**info,**evidence,elapsed_seconds=time.monotonic()-started,
                original_samples=3,expert_samples=1,cache_stats=dict(cache.stats),ram_cache_stats=dict(store.stats),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            del batch,loss,outputs
            with (out/'log.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            h.write_json(out/'status.json',dict(status='training',step=step,additional_updates=step-30000,
                target_steps=50000,last_loss=row['loss'],elapsed_seconds=row['elapsed_seconds'],physical_success_rate=None))
            if step==30001 or step%25==0: print(json.dumps(row),flush=True)
            if step%1000==0 or stopped['value'] or step==50000:
                h.verify_code_hashes(new['source_files'])
                trainer.save_checkpoint(out,wrapper,new,step,optimizer=optimizer,rng=rng.bit_generator.state)
            if step%5000==0 or stopped['value'] or step==50000:
                trainer.save_checkpoint(out,wrapper,new,step)
                h.write_json(out/f'diagnostic_{step:06d}.json',dict(
                    train=trainer.diagnostic(wrapper,datasets['train']),val=trainer.diagnostic(wrapper,datasets['val']),
                    scope='fixed single-anchor offline diagnostic per split; not full VAL loss or physical success'),immutable=True)
                log(f'恢复混合续训保存{step}/50000，新增{step-30000}/20000，3原成功+1专家纠正，无插入物理测试。','dagger_training')
        h.write_json(out/'status.json',dict(status='completed' if step==50000 else 'stopped_at_safe_boundary',
            step=step,target_steps=50000,additional_updates=step-30000,physical_success_rate=None))
        return 0
    except BaseException as exc:
        h.write_json(out/'status.json',dict(status='failed',step=step,target_steps=50000,error=f'{type(exc).__name__}: {exc}',evidence_preserved=True))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-full-state',required=True)
    p.add_argument('--collection',required=True);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--gpu-lock-owner',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();args.minimum_free_bytes=100*1024**3
    plan=make_plan(args)
    if not args.execute: print(json.dumps(plan,ensure_ascii=False));return 0
    with Path(plan['parent_full_state']['path']).open('rb') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return execute(plan,args)


if __name__=='__main__':raise SystemExit(main())
