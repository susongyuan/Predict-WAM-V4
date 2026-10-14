#!/usr/bin/env python3
"""Real Isaac physics, three onboard images, zero-score scene admission smoke."""
from pathlib import Path
import sys,json,time,traceback,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.progress1_log import log

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['zhuhai_task1','zhuhai_task2'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu-lock-owner',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--expert',action='store_true')
    p.add_argument('--randomized',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    result={'status':'starting','task':args.task,'engine':{'name':'Isaac Sim','version':'6.1.0.0'},'started_unix':time.time()}
    import hashlib
    sources=['harness/isaac_zhuhai_runtime.py','harness/isaac_zhuhai_expert.py','harness/isaac_zhuhai_task1.py',
        'harness/isaac_zhuhai_task2.py','harness/isaac_zhuhai_camera.py','harness/zhuhai_robot_spec.py',
        'harness/zhuhai_task1_rules.py','harness/zhuhai_task2_rules.py']
    result['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    result['protocol']={'record_video':False,'task2_geometry':'real visual+collision hull world extrema; unchanged physical tolerances',
        'observation_state':'actual measured native16','control_hz':10,'physics_dt':.004}
    def save():
        temp=args.out/'result.json.tmp';temp.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(args.out/'result.json')
    save();log(f'{args.task}真实Isaac物理/三相机烟测；保存机载PNG与接触评分，不录像',actor='isaac_smoke')
    env=None;expert=None
    def store_motion():
        if expert is None:return
        import numpy as np
        for filename,values in [('target_actions.npy',expert.motion_commands),('measured_states.npy',expert.motion_measured_states)]:
            np.save(args.out/filename,np.asarray(values,dtype=np.float32))
        score_file=args.out/'control_scores.jsonl'
        score_file.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in expert.motion_scores))
        result['motion_evidence']={name:{'sha256':hashlib.sha256((args.out/name).read_bytes()).hexdigest(),
            'bytes':(args.out/name).stat().st_size} for name in ('target_actions.npy','measured_states.npy','control_scores.jsonl')}
        result['motion_evidence']['control_steps']=len(expert.motion_commands)
    try:
        from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
        import numpy as np
        from PIL import Image
        env=IsaacZhuhaiEnv(args.gpu_lock_owner,args.out)
        env.debug_phase_images=True
        env.setup(args.task,args.seed,args.randomized,'body_text')
        images,evidence=env.capture()
        for name,rgb in images.items():Image.fromarray(rgb).save(args.out/f'{name}.png')
        result.update(initial_rule_evidence=env.rule_state(),camera_evidence=evidence,
                      actual_state=env.actual_state().tolist(),robot_contract=env.robot_contract,camera_contract=env.camera_contract)
        result['initial_body_states']=env.body_states()
        result['initial_actual_tool_poses']={arm:env.gripper_pose(arm).tolist() for arm in ('left','right')}
        from harness.isaac_zhuhai_expert import FrankaFK
        from pxr import UsdPhysics,UsdGeom
        fk=env.kinematics;actual=env.actual_state()
        result['mechanical_audit']={'fixed_support':env.support_contract,
            'articulation_root_paths':[str(p.GetPath()) for p in env.stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)],
            'camera_paths':[str(p.GetPath()) for p in env.stage.Traverse() if p.IsA(UsdGeom.Camera)],
            'arm_mount_joint_body0':{arm:[str(p) for p in UsdPhysics.Joint(env.stage.GetPrimAtPath(path)).GetBody0Rel().GetTargets()] for arm,path in env.base_joint_paths.items()},
            'actual_tool_FK_position_error_m':{arm:float(np.linalg.norm(env.gripper_pose(arm)[:3,3]-fk.fk(arm,actual[:7] if arm=='left' else actual[8:15])[:3,3])) for arm in ('left','right')}}
        env.stage.GetRootLayer().Export(str(args.out/'assembled_scene.usda'))
        save()
        for _ in range(10):env.step_action(env.command,.1)
        result['hold_rule_evidence']=env.rule_state()
        if args.expert:
            from harness.isaac_zhuhai_expert import SmoothExpert
            expert=SmoothExpert(env)
            result['expert_info']=expert.play_once()
            store_motion()
            result['final_rule_evidence']=env.rule_state()
            result['final_actual_state']=env.actual_state().tolist()
            result['final_body_states']=env.body_states()
            result['final_actual_tool_poses']={arm:env.gripper_pose(arm).tolist() for arm in ('left','right')}
            for name,rgb in env.capture()[0].items():Image.fromarray(rgb).save(args.out/f'final_{name}.png')
        result['status']='passed_scene_smoke' if (evidence['all_geometry_in_frame'] and evidence['all_named_entities_visible'] and result['initial_rule_evidence']['score']['base_score']==0) else 'failed_scene_admission'
        if args.expert and result['status']=='passed_scene_smoke':
            result['status']='passed_expert_smoke' if (result['expert_info']['plan_success'] and result['expert_info']['full_task_success']) else 'failed_expert_smoke'
        result['seconds']=time.time()-result['started_unix'];save()
    except BaseException as exc:
        store_motion()
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),seconds=time.time()-result['started_unix']);save();raise
    finally:
        if env is not None:env.close()
    print(json.dumps({'status':result['status'],'seconds':result['seconds']},ensure_ascii=False),flush=True)
    return 0 if result['status'] in {'passed_scene_smoke','passed_expert_smoke'} else 2

if __name__=='__main__':raise SystemExit(main())
