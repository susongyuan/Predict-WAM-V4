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
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    result={'status':'starting','task':args.task,'engine':{'name':'Isaac Sim','version':'6.1.0.0'},'started_unix':time.time()}
    def save():
        temp=args.out/'result.json.tmp';temp.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(args.out/'result.json')
    save();log(f'{args.task}真实Isaac物理/三相机烟测；保存机载PNG与接触评分，不录像',actor='isaac_smoke')
    env=None
    try:
        from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
        import numpy as np
        from PIL import Image
        env=IsaacZhuhaiEnv(args.gpu_lock_owner,args.out)
        env.setup(args.task,args.seed,False,'body_text')
        images,evidence=env.capture()
        for name,rgb in images.items():Image.fromarray(rgb).save(args.out/f'{name}.png')
        result.update(initial_rule_evidence=env.rule_state(),camera_evidence=evidence,
                      actual_native14=env.actual_native14().tolist(),camera_contract=env.camera_contract)
        result['initial_actual_tool_poses']={arm:env.gripper_pose(arm).tolist() for arm in ('left','right')}
        from harness.isaac_zhuhai_expert import URDFFK
        from pxr import UsdPhysics,UsdGeom
        fk=URDFFK();actual=env.actual_native14()
        result['mechanical_audit']={'fixed_support':env.support_contract,
            'articulation_root_paths':[str(p.GetPath()) for p in env.stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)],
            'camera_paths':[str(p.GetPath()) for p in env.stage.Traverse() if p.IsA(UsdGeom.Camera)],
            'arm_mount_joint_body0':{prefix:[str(p) for p in UsdPhysics.Joint(env.stage.GetPrimAtPath('/World/Robot/Physics/'+prefix+'_base_joint')).GetBody0Rel().GetTargets()] for prefix in ('fl','fr')},
            'actual_tool_FK_position_error_m':{arm:float(np.linalg.norm(env.gripper_pose(arm)[:3,3]-fk.fk(arm,actual[:6] if arm=='left' else actual[7:13])[:3,3])) for arm in ('left','right')}}
        env.stage.GetRootLayer().Export(str(args.out/'assembled_scene.usda'))
        save()
        for _ in range(10):env.step_native14(env.command,.1)
        result['hold_rule_evidence']=env.rule_state()
        if args.expert:
            from harness.isaac_zhuhai_expert import SmoothExpert
            result['expert_info']=SmoothExpert(env).play_once()
            result['final_rule_evidence']=env.rule_state()
            result['final_actual_native14']=env.actual_native14().tolist()
            result['final_actual_tool_poses']={arm:env.gripper_pose(arm).tolist() for arm in ('left','right')}
            for name,rgb in env.capture()[0].items():Image.fromarray(rgb).save(args.out/f'final_{name}.png')
        result['status']='passed_scene_smoke' if (evidence['all_geometry_in_frame'] and evidence['all_named_entities_visible'] and result['initial_rule_evidence']['score']['base_score']==0) else 'failed_scene_admission'
        result['seconds']=time.time()-result['started_unix'];save()
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),seconds=time.time()-result['started_unix']);save();raise
    finally:
        if env is not None:env.close()
    print(json.dumps({'status':result['status'],'seconds':result['seconds']},ensure_ascii=False),flush=True)
    return 0 if result['status']=='passed_scene_smoke' else 2

if __name__=='__main__':raise SystemExit(main())
