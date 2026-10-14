#!/usr/bin/env python3
"""Meaningful synthetic VJP/time-alignment tests and real frozen-model GPU preflight."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from harness import zhuhai_overlap_guidance_v37 as g

def cpu():
    weights=g.mask();assert np.all(np.diff(weights)<=0) and np.all(weights[24:]==0)
    assert 0<weights[0]<1 and g.guidance_coefficient(0)==5.
    # Exact analytic check: F(x)=x+(1-t)(Mx+b), J=I+(1-t)M.
    matrix=torch.tensor([[.2,.1],[-.3,.4]],dtype=torch.float32)
    bias=torch.tensor([.03,-.02],dtype=torch.float32)
    target=torch.tensor([[.8,-.4]],dtype=torch.float32)
    w=torch.tensor([[.7,.2]],dtype=torch.float32);initial=torch.tensor([[.1,-.2]],dtype=torch.float32)
    actual,traces=g.integrate(lambda x,t:x@matrix.T+bias,initial,target,w)
    expected=initial.clone()
    for i in range(10):
        t=i/10;v=expected@matrix.T+bias;f=expected+(1-t)*v
        grad=((target-f)*w)@(torch.eye(2,dtype=torch.float32)+(1-t)*matrix)
        expected=expected+(v+g.guidance_coefficient(t)*grad)/10
    assert torch.allclose(actual,expected,rtol=0,atol=3e-7)
    plain,_=g.integrate(lambda x,t:x@matrix.T+bias,initial)
    zero,_=g.integrate(lambda x,t:x@matrix.T+bias,initial,target,torch.zeros_like(w))
    assert torch.equal(plain,zero)
    # A model that always postpones closure to index20 never executes it with
    # exec8. Soft overlap carries influence into the prefix but is NOT an event
    # commitment: this test prevents claiming that it guarantees closure.
    proposed=torch.ones(1,32,1);proposed[:,20:]=0
    old=proposed.clone();closures=[]
    for query in range(4):
        y=torch.zeros_like(old);y[:,:24]=old[:,8:]
        old,_=g.integrate(lambda x,t:torch.zeros_like(x),proposed.clone(),y,
            torch.from_numpy(weights)[None,:,None])
        closures.append(bool((old[:,:8]<=.2).any()))
    assert not bool((proposed[:,:8]<=.2).any()) and not any(closures)
    assert .2<float(old[:,:8].min())<.7
    return dict(status='passed',analytic_nonidentity_VJP=True,zero_guidance_exact=True,
        true_time_overlap_indices=[8,32,0,24],toy_postponed_closure_reaches_executed_prefix=closures,
        soft_guidance_not_a_discrete_event_guarantee=True,
        limitations='synthetic mechanism verification, not a robot success result')

def gpu():
    from scripts import gpu_lock
    from scripts import eval_zhuhai_wam_precision_v2 as original
    from harness import zhuhai_precision_v2_training as p
    from harness.robotwin_training_data import preprocess_rgb
    from harness.wam_behavior import load_upstream
    from harness.wam_v3 import V3Context
    from harness.wam_event_clock import EventClock
    from torch.nn.attention import sdpa_kernel,SDPBackend
    plan=json.loads((ROOT/'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_continuity10_20261012_v36_r2/plan.json').read_text())
    owner='zhuhai-rtc-model-replay-v38';assert gpu_lock.owner() is None
    gpu_lock.acquire(owner,max_wait=0)
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        wrapper,condition,contract=p.load_policy(plan['checkpoint']['path'],'zhuhai_task2','cuda')
        _,runner=load_upstream();cfg=contract['config']
        encoder,*_=runner.ModelFactory.create_vision_encoder(str(ROOT/cfg['model']['vision_encoder']['checkpoint_path']),dtype=torch.bfloat16,device='cuda')
        wrapper.base.vision_encoder=encoder.eval().requires_grad_(False)
        sampler=g.OverlapSampler(wrapper,condition,lambda: original.verify_owner(owner))
        v3=cfg['model']['v3'];context=V3Context(v3['proprio_lags'],v3['memory_lags'],original.base.MODEL_VIEWS,state_dim=16)
        context.reset();clock=EventClock(**p.CLOCK_CONFIG)
        directory=ROOT/'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_rtc_single_20261012_v37/zhuhai_task2/seed_932106'
        reports=[];first=None
        for step in range(17):
            with np.load(directory/f'onboard/observation_{step:06d}.npz') as z:
                measured=z['measured_joint_gripper_state'].copy();command=z['commanded_joint_gripper_state'].copy()
                pixels=np.stack([preprocess_rgb(z[v]) for v in original.base.VIEWS])
            event=clock.update(measured,command)
            with torch.inference_mode():feats,pool=wrapper.encode(torch.from_numpy(pixels)[None])
            context.push(measured,pool[0].float().cpu().numpy())
            if step%8:continue
            state,mem=context.inputs();seed_value=int(hashlib.sha256(f'932106:{step}'.encode()).hexdigest()[:15],16)
            with torch.inference_mode():
                # Explicit math-backend baseline for later comparisons.
                with sdpa_kernel(SDPBackend.MATH):
                    plain,_=wrapper.sample_actions(feats,torch.from_numpy(state)[None],torch.from_numpy(mem)[None],
                        torch.from_numpy(condition)[None],num_steps=10,generator=torch.Generator(device='cuda').manual_seed(seed_value),
                        event_state=torch.from_numpy(event.copy())[None])
            previous=sampler.previous
            guided=sampler(feats,context.inputs(),932106,step,event)
            plain=plain[0].float().cpu().numpy()
            if previous is not None:
                target=wrapper.normalize_action(torch.from_numpy(previous[8:]).to('cuda')).cpu().numpy()
                olderr=(wrapper.normalize_action(torch.from_numpy(plain[:24]).to('cuda')).cpu().numpy()-target)**2
                newerr=(wrapper.normalize_action(torch.from_numpy(guided[:24]).to('cuda')).cpu().numpy()-target)**2
                reports.append(dict(step=step,plain_overlap_mse=float(np.mean(olderr*g.mask()[:24,None])),
                    guided_overlap_mse=float(np.mean(newerr*g.mask()[:24,None])),
                    max_guidance_rms=max(t['guidance_rms'] for t in sampler.last_diagnostic['denoising_trace']),
                    inference_seconds=sampler.last_diagnostic['inference_seconds']))
            else:
                # Original v36 first chunk generated on exactly these inputs.
                recorded=json.loads((directory/'actions.jsonl').read_text().splitlines()[0])['policy']['raw_denormalized_chunk']
                assert np.array_equal(guided,np.asarray(recorded,dtype=np.float32)), 'first chunk changed'
                first=True
            archived=json.loads((directory/'actions.jsonl').read_text().splitlines()[step])['policy']['raw_denormalized_chunk']
            assert np.array_equal(guided,np.asarray(archived,dtype=np.float32)), 'actual archived guided chunk not reproduced'
            assert np.isfinite(guided).all() and all(p.grad is None for p in wrapper.parameters())
        assert all(r['guided_overlap_mse']<r['plain_overlap_mse'] for r in reports),'real guidance does not enforce overlap'
        return dict(status='passed',frozen_checkpoint_sha256=plan['checkpoint']['sha256'],
            actual_model_input_VJP=True,three_actual_recorded_guided_chunks_exact_replay=True,first_chunk_exact_original=True,parameter_gradients_absent=True,
            predetermined_queries=[0,8,16],queries=reports,
            scope='archived observations only; no physics rollout or score improvement claim')
    finally:gpu_lock.release(owner)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');parser.add_argument('--out',required=True)
    args=parser.parse_args();path=Path(args.out);assert not path.exists()
    result=dict(cpu=cpu(),gpu=gpu() if args.gpu else None,source_sha256=hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
