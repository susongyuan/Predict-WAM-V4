"""RTC Eq.2--5 soft overlap guidance for the unchanged native16 flow policy.

This is synchronous (simulation does not advance during inference), hence d=0.
The last chunk's unexecuted indices8:32 align with new indices0:24. There is
no hard commitment, phase oracle, action averaging, training, or progress critic.
"""
from pathlib import Path
import hashlib
import math
import time
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from harness.wam_v3 import bf16_from_bits

HORIZON, EXECUTION, DENOISE_STEPS, BETA = 32, 8, 10, 5.
VERSION='precision-v2-rtc-soft-overlap-c1-fine250-v37'
SOURCE='https://arxiv.org/html/2506.07339v1'

def mask(horizon=HORIZON,execution=EXECUTION,delay=0):
    if not 0<=delay<=horizon-execution or not 0<execution<horizon:
        raise ValueError('invalid physical overlap geometry')
    w=np.zeros(horizon,dtype=np.float32)
    overlap=horizon-execution
    w[:delay]=1.
    for i in range(delay,overlap):
        c=(overlap-i)/(overlap-delay+1)
        w[i]=c*math.expm1(c)/math.expm1(1.)
    return w

def guidance_coefficient(tau):
    if not 0<=tau<1:raise ValueError('flow tau outside [0,1)')
    if tau==0:return BETA
    variance=(1-tau)**2/(tau*tau+(1-tau)**2)
    return min(BETA,(1-tau)/(tau*variance))

def integrate(velocity,x,target=None,weights=None,num_steps=DENOISE_STEPS):
    """Exact VJP of denoised estimate, not identity-Jacobian approximation."""
    if num_steps!=DENOISE_STEPS:raise ValueError('bound ten-step integration required')
    traces=[]
    for i in range(num_steps):
        tau=i/num_steps
        if target is None:
            with torch.no_grad():v=velocity(x,tau).float();correction=torch.zeros_like(v)
        else:
            with torch.enable_grad():
                leaf=x.detach().requires_grad_(True)
                v=velocity(leaf,tau).float()
                estimate=leaf+(1-tau)*v
                cotangent=((target-estimate)*weights).detach()
                correction=torch.autograd.grad(estimate,leaf,grad_outputs=cotangent,
                    create_graph=False,retain_graph=False)[0]
            correction=correction.detach()*guidance_coefficient(tau)
        if not torch.isfinite(v).all() or not torch.isfinite(correction).all():
            raise ValueError('nonfinite flow or overlap guidance; never silently fall back')
        x=(x+v.detach()/num_steps+correction/num_steps).detach()
        traces.append(dict(tau=tau,coefficient=0. if target is None else guidance_coefficient(tau),
            flow_rms=float(v.square().mean().sqrt()),guidance_rms=float(correction.square().mean().sqrt())))
    return x,traces

class OverlapSampler:
    def __init__(self,wrapper,condition,verify_owner):
        self.wrapper=wrapper;self.condition=condition;self.verify_owner=verify_owner
        self.previous=None;self.previous_step=None;self.seed=None;self.last_diagnostic=None
        # Only differentiating noisy input, never any trained parameter.
        wrapper.eval().requires_grad_(False)

    def __call__(self,feats,inputs,seed,step,event_state):
        self.verify_owner();start=time.monotonic()
        if step==0:self.previous=None;self.previous_step=None;self.seed=seed
        if self.seed!=seed or (self.previous_step is not None and step!=self.previous_step+EXECUTION):
            raise ValueError('overlap history is not this episode or not exec8 aligned')
        wrapper=self.wrapper;device=wrapper.device
        state,pool=inputs
        seed_value=int(hashlib.sha256(f'{seed}:{step}'.encode()).hexdigest()[:15],16)
        generator=torch.Generator(device=device).manual_seed(seed_value)
        if self.previous is None:
            # First chunk is exactly the old sampler, including attention backend.
            with torch.inference_mode():
                actions,_=wrapper.sample_actions(feats,torch.from_numpy(state)[None],
                    torch.from_numpy(pool)[None],torch.from_numpy(self.condition)[None],
                    num_steps=DENOISE_STEPS,generator=generator,
                    event_state=torch.from_numpy(event_state.copy())[None])
            value=actions[0].float().cpu().numpy()
            target=None;traces=[];noise_sha=None
        else:
            # Clone inference-produced features into normal tensors for autograd.
            features=[bf16_from_bits(f.to(device)).detach().clone() for f in feats]
            if device.type=='cpu':features=[f.float() for f in features]
            qpos=wrapper.normalize_state(torch.from_numpy(state)[None].to(device,dtype=torch.float32)).detach()
            memory=wrapper.normalize_memory(torch.from_numpy(pool)[None].to(device,dtype=torch.float32)).detach()
            condition=torch.from_numpy(self.condition)[None].to(device,dtype=torch.float32)
            clock=torch.from_numpy(event_state.copy())[None].to(device,dtype=torch.float32)
            previous=torch.from_numpy(self.previous)[None].to(device,dtype=torch.float32)
            target=torch.zeros((1,HORIZON,16),device=device)
            target[:,:HORIZON-EXECUTION]=wrapper.normalize_action(previous[:,EXECUTION:])
            weights=torch.from_numpy(mask())[None,:,None].to(device)
            x=torch.randn((1,HORIZON,16),device=device,generator=generator)
            noise_sha=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
            def velocity(noisy,tau):
                t=torch.full((1,),tau,device=device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    return wrapper.action_model(t,noisy_actions=noisy,qpos_history=qpos,
                        dino_features_list=features,task_cond=condition,memory=memory,event_state=clock)['final_pred']
            # PyTorch flash SDPA lacks some backward paths; math SDPA supports VJP.
            with sdpa_kernel(SDPBackend.MATH):
                x,traces=integrate(velocity,x,target,weights)
            value=wrapper.denormalize_action(x)[0].float().cpu().numpy()
        if value.shape!=(HORIZON,16) or not np.isfinite(value).all():raise ValueError('invalid guided native16 chunk')
        self.last_diagnostic=dict(version=VERSION,source=SOURCE,previous_query_step=self.previous_step,
            execution_horizon=EXECUTION,overlap_actions=0 if self.previous is None else HORIZON-EXECUTION,
            physical_inference_delay_actions=0,soft_mask=mask().tolist(),guidance_clip=BETA,
            previous_unexecuted_native_chunk=None if self.previous is None else self.previous[EXECUTION:].tolist(),
            denoising_trace=traces,initial_noise_sha256=noise_sha,query_seed=seed_value,
            inference_seconds=time.monotonic()-start,model_weights_updated=False,
            same_time_overlap=True,stage_progress_or_grasp_oracle=False)
        self.previous=value.copy();self.previous_step=step
        return value
