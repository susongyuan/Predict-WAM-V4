#!/usr/bin/env python3
"""Bounded native RoboTwin1 first RGB/target visibility probe; no policy or expert action."""
import os,sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R1=ROOT/'third_party/RoboTwin1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.chdir(R1);sys.path.insert(0,str(R1));sys.path.insert(0,str(ROOT))
import yaml,numpy as np,cv2
from envs.pick_apple_messy import pick_apple_messy
class Probe(pick_apple_messy):
 def setup_planner(self,**kw): pass # No planner constructed or motion command; original initial gripper opening remains.
def config():
 cfg=yaml.safe_load((R1/'task_config/pick_apple_messy.yml').read_text());cams=yaml.safe_load((R1/'task_config/_camera_config.yml').read_text())
 for name in ('head','wrist','front'):
  for k,v in cams[cfg[name+'_camera_type']].items():cfg[f'{name}_camera_{k}']=v
 cfg.update(render_freq=0,is_save=False,collect_data=False)
 cfg['data_type'].update(pointcloud=False,depth=False,rgb=True)
 return cfg
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=910000);p.add_argument('--gpu-lock-owner',required=True);a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
 from scripts.gpu_lock import owner
 assert owner()==a.gpu_lock_owner
 env=None;result={'status':'running','task':'pick_apple_messy','seed':a.seed,'policy_actions':0,'expert_actions':0,'initial_gripper_opening':'official_setup_pre_move','official_version':'RoboTwin-1.0','renderer':'rt32_depth8_none'}
 try:
  env=Probe();env.setup_demo(now_ep_num=0,seed=a.seed,is_test=True,**config());env._update_render()
  apple_id=env.apple.per_scene_id;result.update(apple_actor_id=apple_id,native_initial_success=bool(env.check_success()),apple_xyz=env.apple.get_pose().p.tolist(),views={})
  for v in ('head_camera','left_camera','right_camera','observer_camera'):
   cam=getattr(env,v);cam.take_picture();rgb=(cam.get_picture('Color')[:,:,:3]*255).clip(0,255).astype(np.uint8);seg=cam.get_picture('Segmentation');mask=seg[:,:,1]==apple_id
   cv2.imwrite(str(out/(v+'.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));np.save(out/(v+'_seg.npy'),seg)
   yy,xx=np.where(mask);result['views'][v]={'shape':list(rgb.shape),'target_pixels':int(mask.sum()),'bbox_xyxy':([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None)}
  result.update(status='captured_pending_visual_review',first_frame_target_visible=any(result['views'][v]['target_pixels']>0 for v in ('head_camera','left_camera','right_camera')))
 except BaseException as e:result.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if env is not None:
   try:env.close()
   except Exception as e:result['cleanup_error']=repr(e)
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
