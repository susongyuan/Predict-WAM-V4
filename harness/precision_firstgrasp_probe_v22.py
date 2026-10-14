"""Small, paired inference probes using existing original TRAIN/VAL datasets."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from harness import zhuhai_precision_v2_training as h
from harness.zhuhai_action_smoothing import smooth_chunk
from scripts.train_wam_v3 import isolated_rng


def select_probes(datasets):
    probes = []
    for split in ('train', 'val'):
        ds = datasets[split]['zhuhai_task2']
        families = sorted({e.row['family_id'] for e in ds.episodes})
        selected = set(families if split == 'val' else
                       [families[i] for i in np.linspace(0, len(families)-1, 8, dtype=int)])
        for i, ep in enumerate(ds.episodes):
            if ep.row['family_id'] not in selected:
                continue
            closed = np.flatnonzero(ep.action[:, 7] < .2)
            if not len(closed):
                raise ValueError('Missing original first left closure')
            close = int(closed[0])
            for phase, t in [('before_close_4', close-4), ('after_close_4', close+4)]:
                if not 0 <= t <= len(ep)-32:
                    raise ValueError('Incomplete first-closure target')
                probes.append(dict(split=split, phase=phase, domain=ep.row['domain'],
                    family_id=ep.row['family_id'], path=str(ep.path), source_sha256=ep.sha256,
                    anchor=t, index=(ds.ends[i-1] if i else 0)+t))
    if len(probes) != 72:
        raise ValueError('Expected 16 TRAIN plus 20 VAL episodes, two probes each')
    return probes


def metrics(pred, target, action_range):
    arm = [*range(7), *range(8, 15)]
    error = pred-target
    scale = np.where(action_range < 1e-6, 1., action_range)
    result = dict(windows=len(pred), normalized_action_mse=float(np.mean((2*error/scale)**2)),
                  left_arm_mae_rad=float(np.abs(error[..., :7]).mean()),
                  right_arm_mae_rad=float(np.abs(error[..., 8:15]).mean()),
                  arm_mae_rad=float(np.abs(error[..., arm]).mean()))
    for name, dim in [('left',7),('right',15)]:
        p,t=pred[...,dim],target[...,dim]
        closed=t<.2
        result[name+'_gripper']=dict(mae_fraction=float(np.abs(p-t).mean()),
            target_closed_frames=int(closed.sum()),predicted_closed_frames=int((p<.2).sum()),
            close_recall=float(((p<.2)&closed).sum()/closed.sum()) if closed.any() else None,
            missed_close_stays_open=int(((p>.9)&closed).sum()),
            mean_prediction_when_target_closed=float(p[closed].mean()) if closed.any() else None)
    return result


def run_probe(wrapper, datasets, out, step, contract, *, checkpoint=None):
    """No optimizer, future inputs, source edits or changes to the caller's RNG."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    probes=select_probes(datasets)
    protocol=dict(schema='small-paired-firstclosure-probe-v22',probes=probes,
        integration_steps=10,batch_size=4,noise_seed_prefix='firstclosure-v22',
        train_episodes=16,val_episodes=20,heldout_unused=True,
        model_input_whitelist=['feats','state','pool','task_cond','event_state'],
        no_future_or_label_inputs=True,scope='offline teacher-state first closure only; not full task success',
        unchanged_action_postprocessing=True)
    h.write_json(out/'plan.json',protocol,immutable=True)
    was_training=wrapper.training;wrapper.eval()
    cpu_before=torch.get_rng_state().clone()
    cuda_before=[s.clone() for s in torch.cuda.get_rng_state_all()]
    predictions,labels,states,clocks=[],[],[],[]
    try:
        with torch.inference_mode(),isolated_rng(20261011,wrapper.device):
            for start in range(0,len(probes),4):
                items=probes[start:start+4]
                batch=default_collate([datasets[p['split']]['zhuhai_task2'][p['index']] for p in items])
                f=batch['feats']; layers=[f[:,:,i] for i in range(f.shape[2])]
                seed=int(hashlib.sha256(f'firstclosure-v22:{start}'.encode()).hexdigest()[:15],16)
                generator=torch.Generator(device=wrapper.device).manual_seed(seed)
                action,_=wrapper.sample_actions(layers,batch['state'],batch['pool'],batch['task_cond'],
                    num_steps=10,generator=generator,event_state=batch['event_state'])
                predictions.extend(action.float().cpu().numpy());labels.extend(batch['action_sequence'].numpy())
                states.extend(batch['state'].numpy());clocks.extend(batch['event_state'].numpy())
    finally:
        wrapper.train(was_training)
    if not torch.equal(cpu_before,torch.get_rng_state()) or any(
            not torch.equal(a,b) for a,b in zip(cuda_before,torch.cuda.get_rng_state_all())):
        raise ValueError('Diagnostic changed training RNG')
    pred,target=np.asarray(predictions),np.asarray(labels)
    if not np.isfinite(pred).all():raise ValueError('Nonfinite diagnostic predictions')
    # Compare existing spline targets; do not fabricate past limiter dynamics.
    smoothed=np.stack([smooth_chunk(p,contract['action_postprocessing']) for p in pred])
    values={}
    for split in ('train','val'):
        for domain in ('all','source','randomized'):
            for phase in ('all','before_close_4','after_close_4'):
                ix=[i for i,p in enumerate(probes) if p['split']==split and
                    (domain=='all' or p['domain']==domain) and (phase=='all' or p['phase']==phase)]
                if ix:
                    values[f'{split}/{domain}/{phase}']=dict(
                        raw_first8=metrics(pred[ix,:8],target[ix,:8],(wrapper.action_max-wrapper.action_min).cpu().numpy()),
                        spline_first8=metrics(smoothed[ix,:8],target[ix,:8],(wrapper.action_max-wrapper.action_min).cpu().numpy()))
    np.savez(out/f'actions_{step:06d}.npz',model_raw=pred,spline_targets=smoothed,
             expert_targets=target,measured_history=np.asarray(states),causal_clock=np.asarray(clocks))
    result=dict(step=step,checkpoint=checkpoint,rng_unchanged=True,model_training_mode_restored=True,
                optimizer_updates=0,metrics=values)
    h.write_json(out/f'diagnostic_{step:06d}.json',result,immutable=True)
    return result


def run_reference_probes(wrapper, datasets, out, contract, optimizer):
    """Compare archived milestones once, then put back the exact live weights."""
    from scripts.train_wam_v3 import cpu_tree
    before=cpu_tree(wrapper.action_model.state_dict())
    optimizer_steps=[int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
    root=h.ROOT/'outputs/zhuhai2026_tasks12'
    paths=[root/'wam_precision_v2_task2_30k_20261010_v1/train/checkpoint_step_030000.pt',
           root/'wam_precision_v2_task2_dagger_weighted50k_20261011_v18/train/checkpoint_step_045000.pt',
           root/'wam_precision_v2_task2_dagger_weighted50k_20261011_v18/train/checkpoint_step_050000.pt']
    try:
        for path in paths:
            receipt=h.read_json(path.with_name(path.name+'.sha256.json'))
            if receipt['sha256']!=h.sha256_file(path) or receipt['size']!=path.stat().st_size:
                raise ValueError('Reference model receipt differs')
            ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            base=h.validate_training_contract(ck['contract']);h.verify_code_hashes(base['source_files'])
            for k in ('statistics','conditions','precision','idm_statistics','splits'):
                if base[k]!=contract[k]:raise ValueError('Reference inputs differ: '+k)
            if not ck['model_only'] or ck['step']!=receipt['step']:
                raise ValueError('Immutable reference milestone required')
            wrapper.action_model.load_state_dict(ck['model_state_dict'],strict=True)
            run_probe(wrapper,datasets,out,ck['step'],contract,checkpoint=dict(path=str(path),**receipt))
            del ck
    finally:
        wrapper.action_model.load_state_dict(before,strict=True)
    if optimizer_steps!=[int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]:
        raise ValueError('Reference comparison changed optimizer steps')
    for key,value in wrapper.action_model.state_dict().items():
        if not torch.equal(value.cpu(),before[key]):raise ValueError('Live weights not exactly restored')
    h.write_json(Path(out)/'reference_restore_receipt.json',dict(
        live_model_exactly_restored=True,optimizer_steps=sorted(set(optimizer_steps)),
        reference_steps=[30000,45000,50000],optimizer_updated_by_comparison=False),immutable=True)
