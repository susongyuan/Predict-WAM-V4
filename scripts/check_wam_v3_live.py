#!/usr/bin/env python3
"""One GPU preflight: real V2 -> V3 migration, cached/live parity, same-noise K=1, gradients. No optimizer step."""
import os,sys,json,time,hashlib
from pathlib import Path
os.environ.setdefault('P2_TASK_ID','0')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from harness import task_profile
from harness.wam_behavior import load_upstream,model_views,normalize_image
from harness.frame_cache import FramePack
from harness.wam_v3 import (create_v3_action_model,load_v3_state,WamV3Wrapper,WamV3Dataset,bf16_from_bits,history_frames)

def main():
    torch.set_num_threads(4); torch.manual_seed(20260921);torch.cuda.set_per_process_memory_fraction(.7)
    cfg=OmegaConf.load(ROOT/'configs/lilawam_task000_multiview_manip_v3_train.yaml'); views=model_views(cfg)
    source=ROOT/'outputs/training/task0_manip_v2_20260928/checkpoint_es_best.pt'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg.training.init_from_sha256
    model=create_v3_action_model(cfg,dino_hidden_size=1024,num_dino_layers=3,task_cond_dim=1024,patch_size=16)
    ck=torch.load(source,map_location='cpu',weights_only=False)
    notes=load_v3_state(model,ck['model_state_dict']);del ck
    _,runner=load_upstream()
    enc,hidden,nreg,patch=runner.ModelFactory.create_vision_encoder(str(ROOT/cfg.model.vision_encoder.checkpoint_path),torch.bfloat16,torch.device('cuda'))
    wrapper=WamV3Wrapper(model,views=views,stats=json.loads((ROOT/cfg.model.v3.stats_path).read_text()),
        norm_stats_path=ROOT/cfg.dataset.stats_path,proprio_mask=cfg.model.proprio_mask,train_cfg=cfg.training,
        vision_encoder=enc,device='cuda').to('cuda').eval()
    ds=WamV3Dataset('train',feature_cache=ROOT/cfg.dataset.feature_cache,views=views,proprio_lags=cfg.model.v3.proprio_lags,
        memory_lags=cfg.model.v3.memory_lags,contract_targets=ROOT/cfg.dataset.contract_targets,contract_targets_sha256=cfg.dataset.contract_targets_sha256)
    indices=[0,len(ds)//2];batch=default_collate([ds[i] for i in indices])
    pixels=[]
    for i in indices:
        ep,t=ds.anchors[i]
        pixels.append(np.stack([normalize_image(FramePack(task_profile.FRAMES/v,ep).rgb(t)) for v in views]))
    live,pool=wrapper.encode(torch.from_numpy(np.stack(pixels)))
    bits=bf16_from_bits(batch['feats'].to('cuda'));cached=[bits[:,:,i] for i in range(3)]
    exact=[bool(torch.equal(l,c)) for l,c in zip(live,cached)]
    assert all(exact),exact
    pooldiff=float((pool.flatten(-2).cpu()-batch['pool'][:,:,0]).abs().max())
    assert pooldiff==0.,pooldiff
    g=lambda:torch.Generator(device='cuda').manual_seed(20260921)
    a,aux=wrapper.sample_actions(cached,batch['state'],batch['pool'],batch['task_cond'],num_steps=10,generator=g())
    b,_=wrapper.sample_actions(live,batch['state'],batch['pool'],batch['task_cond'],num_steps=10,generator=g())
    assert torch.equal(a,b)
    # Free the encoder before backward: training itself never loads it.
    wrapper.vision_encoder=None;del enc,live,cached,bits,pool;torch.cuda.empty_cache()
    wrapper.train();started=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,info,out=wrapper(batch,cfg.training.auxiliary_loss)
    assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize()
    grad={}
    for key in ['mem_proj.weight','phase_net.1.weight','gate_head.weight','phase_head.weight','dist_head.weight']:
        value=dict(model.named_parameters())[key].grad
        assert value is not None and torch.isfinite(value).all(),key
        grad[key]=float(value.norm());assert grad[key]>0,key
    report=dict(kind='GPU interface/gradient smoke; no optimizer step or capability claim',passed=True,initialization=str(source),
        migration=notes,cache_live_feature_bit_equal=exact,pool_max_abs_diff=pooldiff,cache_live_k1_actions_bit_equal=True,
        loss=float(loss),loss_info=info,new_module_grad_norms=grad,batch=2,forward_backward_seconds=time.monotonic()-started,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    path=ROOT/'outputs/validation/task0_v3_audit_20260928/live_smoke.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
