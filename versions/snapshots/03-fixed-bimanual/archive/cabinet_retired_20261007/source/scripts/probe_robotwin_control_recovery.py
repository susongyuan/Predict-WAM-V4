#!/usr/bin/env python3
"""Real two-action recovery interface probe, never task-success evidence."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import eval_robotwin_wam as ev
from harness.robotwin_policy_worker import atomic_json,verify_gpu_owner
from harness.robotwin_qpos_adapter import policy_observation,NativeQposAdapter
from harness.robotwin_robot_feedback import hold_measured,arm_feedback
from harness.robotwin_eval_evidence import joint_bounds,bound_action

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--gpu-lock-owner',required=True);a=p.parse_args()
 out=Path(a.run_dir).resolve();out.mkdir(parents=True,exist_ok=False)
 args=ev.parse_args(['--run-dir',str(out),'--training-run-dir',str(ROOT/'outputs/runs/put_object_cabinet/wam_30k_20261005'),'--checkpoint',str(ROOT/'outputs/runs/put_object_cabinet/wam_30k_20261005/checkpoint_step_030000.pt'),'--gpu-lock-owner',a.gpu_lock_owner,'--seeds','10003','--render-denoiser','none','--max-steps','2'])
 atomic_json(out/'protocol.json',{'kind':'real_control_interface_not_task_validation','seed':10003,'sequence':['infer0','act0','clear_queue','hold_measured','observe1','infer1','act1'],'no_gt_policy':True,'task_success':None})
 env=client=None;r={'status':'running','task_success':None,'phase':'preflight'};atomic_json(out/'result.json',r)
 def phase(x):r['phase']=x;atomic_json(out/'result.json',r)
 try:
  verify_gpu_owner(a.gpu_lock_owner);plan=ev.plan_evaluation(args);atomic_json(out/'plan.json',plan)
  phase('worker_start');client=ev.make_client(args,out);r['ready']=client.ready
  phase('scene_start');os.chdir(ev.ROBOTWIN);env=ev.make_env();env.setup_demo(now_ep_num=0,seed=10003,is_test=True,**plan['task_kwargs'])
  limits,_=joint_bounds(env.robot);adapter=NativeQposAdapter(env,limits);client.reset('recovery_probe',10003)
  phase('observe0');o=policy_observation(env.get_obs());reply=client.action('recovery_probe',0,o);r['before_history']=reply['history_steps'];r['queue_before']=reply['queue_remaining']
  phase('act0');adapter.step(bound_action(reply['raw_action'],limits)['bounded_action']);r['native_count_before']=env.take_action_cnt
  phase('clear_actions');r['clear_ack']=client.clear_actions('recovery_probe',1,'interface_probe_after_consumed_action')
  phase('hold_measured');r['hold_receipt']=hold_measured(env.robot)
  phase('observe1');o=policy_observation(env.get_obs());reply2=client.action('recovery_probe',1,o)
  r['after_history']=reply2['history_steps'];r['replanned']=reply2['replanned']
  if r['after_history']!=2 or not r['replanned'] or r['clear_ack']['discarded_actions']!=7:raise RuntimeError('history/queue recovery invariant failed')
  phase('act1');adapter.step(bound_action(reply2['raw_action'],limits)['bounded_action']);r['native_count_after']=env.take_action_cnt
  if r['native_count_after']!=2:raise RuntimeError('wrong native consumption count')
  r.update(status='interface_pass_not_recovery_or_task_success',phase='complete')
 except BaseException as exc:r.update(status='error',error=f'{type(exc).__name__}: {exc}')
 finally:
  try:ev.close_env(env)
  except BaseException as exc:r['cleanup_error']=str(exc)
  if client:client.close()
  atomic_json(out/'result.json',r)
 return 0 if r['status'].startswith('interface_pass') else 1
if __name__=='__main__':raise SystemExit(main())
