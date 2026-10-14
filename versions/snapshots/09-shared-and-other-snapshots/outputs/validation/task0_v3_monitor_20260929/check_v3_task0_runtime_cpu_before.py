#!/usr/bin/env python3
"""Real V3 action checkpoint + complete Task0 runtime init with a synthetic CPU encoder.
NOT a DINO/robot/physics check: catches config/constructor/orchestrator wiring before GPU simulator.
"""
import os,sys,json,time,argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
os.environ.setdefault('P2_TASK_ID','0')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from harness.wam_behavior import load_upstream
from harness.policy_server import WamPolicy
class Encoder(torch.nn.Module):
 def forward(self,pixel_values,**kwargs):
  n=len(pixel_values);dev=pixel_values.device
  x=torch.linspace(0,1,1024,device=dev)[None,None,:]+torch.arange(201,device=dev)[None,:,None]*.001
  x=x.expand(n,-1,-1)+pixel_values.float().mean((1,2,3))[:,None,None]*.001
  return SimpleNamespace(hidden_states=[x]*25,last_hidden_state=x)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='outputs/training/task0_v3_pilot_20260928_a3/checkpoint_best.pt');ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 cfg={'vlm':'none','program':'configs/task000_fixed_program.json','frozen':True,'train_targets':'data/skill_vtt_task000/contract_targets.json','navigation':'docking','postconditions':'task0_physical','precondition':False,'reapproach_on_retry':True,'max_attempts':3,'visible_objects':['radio','coffee_table'],'perceive_every':5,'min_budget_steps':{'move to':1000,'pick up from':1200,'press':600,'place on':700},'navigation_settle_steps':10,'manip_snapshot_every':30,'manip_executor':'wam','dock_target_at':'grasp','base_control':'hold_during_manipulation'}
 _,runner=load_upstream();start=time.monotonic()
 with patch.object(runner.ModelFactory,'create_vision_encoder',return_value=(Encoder(),1024,4,16)):
  p=WamPolicy(a.checkpoint,skill='move to',exec_horizon=8,num_steps=10,device='cpu',cuda_fraction=.3,seed=0,log_dir=out,orchestrate=cfg,samples=1)
  assert p.model.is_v3 and p.model.step==2000
  state=np.load(ROOT/'data/lowdim_cache_task000/episode_61_state.npy')[650].copy()
  obs={'robot_r1::proprio':state,'hidden_object_truth':{'radio':'not policy data'}}
  for link in ['zed_link','left_realsense_link','right_realsense_link']:obs[f'robot_r1::robot_r1:{link}:Camera:0::rgb']=np.zeros((224,224,4),np.uint8)
  for i in range(2):
   action=np.asarray(p.act(obs));assert action.shape==(23,) and np.isfinite(action).all()
  # Direct manipulation sampler with actual checkpoint, synthetic visual features: API/shape only.
  allowed=p._policy_observation(obs);_,pixels=p._preprocess(allowed)
  p._record_v3_frame(p.model,allowed,state,pixels,('cpu-only-sample','pick'))
  condition=p.prototypes[p.classes['pick up from']][None]
  chunk=p._sample(p._v3_frame[p.model.name][0],p._proprio_input(p.model,state),condition)
  assert chunk.shape==(32,23) and np.isfinite(chunk).all()
  p.trace.close()
 report={'passed':True,'kind':'CPU real-action-checkpoint / synthetic-encoder / missing-depth fail-closed runtime smoke','checkpoint':a.checkpoint,'seconds':time.monotonic()-start,'constructor':'actual Task0 full-runtime options','act_calls':2,'sample_shape':[32,23],'gpu_used':False,'physics_success_tested':False,'note':'synthetic zeros lack depth; no claim of navigation progress or pretrained vision correctness'}
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
