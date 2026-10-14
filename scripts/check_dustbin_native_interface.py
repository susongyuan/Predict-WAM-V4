#!/usr/bin/env python3
"""Native RGB visibility and 14D hold-interface smoke; no policy/expert/grasp action.
Actor segmentation read only for this preflight report, never supplied to WAM.
"""
import os,sys,argparse,json,traceback,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.eval_robotwin_wam import load_task_config,make_env,close_env,ROBOTWIN
from harness.robotwin_qpos_adapter import NativeQposAdapter,policy_observation
from harness.robotwin_eval_evidence import joint_bounds
from harness.robotwin_policy_worker import atomic_json
from harness.robotwin_camera_readback import SerialColorReadback
from scripts.gpu_lock import owner
import numpy as np,cv2

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=920000);p.add_argument('--gpu-lock-owner',required=True);a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(ROBOTWIN)
 assert owner()==a.gpu_lock_owner
 env=None;r={'status':'running','task':'put_bottles_dustbin','seed':a.seed,'policy_constructed':False,'expert_called':False,'views':{}}
 try:
  cfg,_=load_task_config('demo_clean',r['task']);cfg['render_denoiser']='none'
  env=make_env(r['task']);env.setup_demo(now_ep_num=0,seed=a.seed,is_test=True,**cfg);serial=SerialColorReadback(env.cameras)
  obs=policy_observation(env.get_obs());r['initial_native_success']=bool(env.check_success());assert not r['initial_native_success']
  targets=[('bottle'+str(i),int(b.actor.per_scene_id)) for i,b in enumerate(env.bottles)]+[('dustbin',int(env.dustbin.actor.per_scene_id))]
  cameras={'left_camera':env.cameras.left_camera,'right_camera':env.cameras.right_camera,**dict(zip(env.cameras.static_camera_name,env.cameras.static_camera_list))}
  for v,image in obs['rgb'].items():
   cv2.imwrite(str(out/(v+'.png')),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));cam=cameras[v];seg=cam.get_picture('Segmentation')[:,:,1];r['views'][v]={}
   for name,actorid in targets:
    yy,xx=np.where(seg==actorid);r['views'][v][name]={'pixels':len(xx),'bbox_xyxy':([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None),'touches_border':bool(len(xx) and (xx.min()==0 or yy.min()==0 or xx.max()==image.shape[1]-1 or yy.max()==image.shape[0]-1))}
  r['all_three_targets_visible_first_frame']=all(any(r['views'][v]['bottle'+str(i)]['pixels']>=32 for v in r['views']) for i in range(3))
  assert r['all_three_targets_visible_first_frame'],'target absent from all policy views'
  r['initial_commanded_state']=obs['commanded_joint_gripper_state'].tolist();limits,_=joint_bounds(env.robot);adapter=NativeQposAdapter(env,limits)
  before=env.take_action_cnt;adapter.step(obs['commanded_joint_gripper_state']);after=env.take_action_cnt;new=policy_observation(env.get_obs());assert after==before+1 and np.isfinite(new['commanded_joint_gripper_state']).all()
  r.update(status='native_interface_smoke_complete',native_take_action_count_delta=after-before,adapter_topp_diagnostics=adapter.last_topp_diagnostics,native_step_limit=env.step_lim,after_hold_native_success=bool(env.check_success()),heldout_used=False,framing_note='visible means >=32 actor pixels in at least one actual WAM view; full-object framing is recorded separately, not required')
 except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  try:close_env(env)
  except Exception as e:r.update(status='failed',cleanup_error=repr(e))
  atomic_json(out/'result.json',r);print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
