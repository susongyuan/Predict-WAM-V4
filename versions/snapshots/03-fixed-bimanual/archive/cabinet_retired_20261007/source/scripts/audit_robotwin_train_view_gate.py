#!/usr/bin/env python3
"""CPU-only trained view gate inspection; coefficients are not causal attribution."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,h5py
from omegaconf import OmegaConf
from harness.wam_v3 import create_v3_action_model,WamV3Wrapper,history_frames
from harness.robotwin_eval_evidence import ResourceGuard
class GateCaptured(Exception):pass

def main():
 out=ROOT/'outputs/validation/train455_view_gate_20261006';out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1);guard=ResourceGuard();guard.check();run=ROOT/'outputs/runs/put_object_cabinet/wam_30k_20261005';ck=torch.load(run/'checkpoint_step_030000.pt',map_location='cpu',weights_only=False,mmap=True);cfg=ck['contract']['config'];m=create_v3_action_model(OmegaConf.create(cfg),dino_hidden_size=1024,num_dino_layers=3,task_cond_dim=1024,patch_size=16);m.load_state_dict(ck['model_state_dict'],strict=True);del ck
 stats=json.loads((run/'temporal_stats.json').read_text());w=WamV3Wrapper(m,views=cfg['model']['views'],stats=stats,norm_stats_path=run/'normalization.json',normalization_key='robotwin',proprio_mask='native_all14',proprio_keep_override=np.ones(14),device='cpu').eval();captured=[]
 def hook(module,args,value):captured.append(value.detach().softmax(-1).tolist());raise GateCaptured()
 handle=m.gate_head.register_forward_hook(hook);cache=next((ROOT/'data/robotwin_features').glob('*/episode_0000455-*/feat.npy'));arr=np.load(cache,mmap_mode='r');pool=np.load(cache.parent/'pool.npy',mmap_mode='r');file=ROOT/'data/benchmarks/robotwin2/put_object_cabinet_500_v1/data/episode_0000455.hdf5'
 with h5py.File(file) as f:s=np.concatenate([np.asarray(f['state/'+k]) for k in ('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')],-1)
 records=[]
 for t in (0,8,16,24,56,69,139,209,278):
  guard.check();b=torch.from_numpy(np.array(arr[t],copy=True).view(np.int16)).view(torch.bfloat16).float()[None];feats=[b[:,:,i] for i in range(3)];pframes=[t]+history_frames(t,0,cfg['model']['v3']['proprio_lags']);mframes=[t]+history_frames(t,0,cfg['model']['v3']['memory_lags']);state=torch.from_numpy(s[pframes])[None];mem=torch.from_numpy(np.array(pool[mframes].transpose(1,0,2),copy=True))[None]
  with torch.inference_mode():
   try:m(torch.zeros(1),noisy_actions=torch.zeros(1,32,14),qpos_history=w.normalize_state(state),dino_features_list=feats,task_cond=torch.zeros(1,1024),memory=w.normalize_memory(mem))
   except GateCaptured:pass
  records.append({'observation_index':t,'view_coefficients_head_left_right':captured[-1][0]});print(records[-1],flush=True)
 handle.remove();r={'scope':'TRAIN455 only','checkpoint':'30k','records':records,'strict_load':True,'CUDA_initialized':torch.cuda.is_initialized(),'peak_resources':guard.peaks,'method':'local hook reads trained gate output then short-circuits before action expert','limitations':'Gate coefficients multiply attention bias; not causal contribution, not proof of useful head imagery or task success.'};(out/'result.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
