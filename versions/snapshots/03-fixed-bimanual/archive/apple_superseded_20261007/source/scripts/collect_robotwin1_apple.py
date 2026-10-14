#!/usr/bin/env python3
"""Collect one original RoboTwin1 apple expert rollout as three-RGB/native14 HDF.
Expert GT is used only by the upstream demonstration generator, never by WAM tests.
Each invocation owns one scene; errors and failed demonstrations are preserved.
"""
import os,sys,json,time,traceback,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R1=ROOT/'third_party/RoboTwin1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
os.chdir(R1);sys.path.insert(0,str(R1));sys.path.insert(0,str(ROOT))
import numpy as np,cv2,h5py,mplib
from scripts.probe_robotwin1_apple import config
from envs.pick_apple_messy import pick_apple_messy
VIEWS=('head_camera','left_camera','right_camera')
CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')
class LegacyPlanner:
    def __init__(self,group):
        self.inner=mplib.Planner(urdf=str(R1/'aloha_maniskill_sim/urdf/arx5_description_isaac.urdf'),srdf=str(R1/'aloha_maniskill_sim/srdf/arx5_description_isaac.srdf'),move_group=group)
        self.inner.set_base_pose(mplib.Pose([0,-.65,0],[1,0,0,1]))
        from harness.robotwin1_expert_planner import official_recipe_plan_screw
        import types
        self.inner.plan_screw=types.MethodType(official_recipe_plan_screw,self.inner)
    def plan_screw(self,target_pose,qpos,time_step,use_point_cloud=False,use_attach=False):
        if use_point_cloud or use_attach:raise ValueError('unimplemented legacy planner option')
        return self.inner.plan_screw(mplib.Pose(np.asarray(target_pose[:3]),np.asarray(target_pose[3:])),np.asarray(qpos),time_step=time_step)
    def __getattr__(self,name):return getattr(self.inner,name)
class Collector(pick_apple_messy):
    def __init__(self,out):super().__init__();self.out=out;self.record=False;self.states=[];self.images={v:[] for v in VIEWS};self.first_visibility={}
    def setup_planner(self,**kw):self.left_planner=LegacyPlanner('fl_link6');self.right_planner=LegacyPlanner('fr_link6')
    def _take_picture(self):
        if not self.record:return
        self._update_render()
        for v in VIEWS:
            cam=getattr(self,v);cam.take_picture();rgb=(cam.get_picture('Color')[:,:,:3]*255).clip(0,255).astype(np.uint8)
            if not self.states:
                cv2.imwrite(str(self.out/f'initial_{v}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));mask=cam.get_picture('Segmentation')[:,:,1]==self.apple.per_scene_id
                yy,xx=np.where(mask);self.first_visibility[v]={'target_pixels':int(mask.sum()),'bbox_xyxy':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None}
            ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
            self.images[v].append(jpeg.tobytes())
        state=np.asarray(self.get_left_arm_jointState()+self.get_right_arm_jointState(),np.float32)
        assert state.shape==(14,) and np.isfinite(state).all();self.states.append(state)

def write_hdf(env,path):
    states=np.stack(env.states);assert len(states)>32
    with h5py.File(path,'x') as f:
        f.create_dataset('data_format_version',data='v1.0',dtype=h5py.string_dtype('utf-8'))
        f.create_dataset('instructions',data=json.dumps(['Pick up the apple among the clutter.']),dtype=h5py.string_dtype('utf-8'))
        f.create_group('additional_info').create_dataset('frequency',data=250/15)
        f.attrs.update(source_format='RoboTwin1_native_local_expert_RGB',state_semantics='controller_joint_targets_and_gripper_command_not_measured_qpos',gripper_units='physical_meters_0_to_0.045',time_semantics='observation_index; nominal_frequency_not_measured_clock',alignment='state=command[t], action=command[t+1], RGB=observation[t]; final RGB omitted')
        for group,array in [('state',states[:-1]),('action',states[1:])]:
            g=f.create_group(group)
            for name,sl in [('left_arm_joint_states',slice(0,6)),('left_ee_joint_states',slice(6,7)),('right_arm_joint_states',slice(7,13)),('right_ee_joint_states',slice(13,14))]:g.create_dataset(name,data=array[:,sl])
        vision=f.create_group('vision')
        for v,c in zip(VIEWS,CAMERAS):
            g=vision.create_group(c);buffers=env.images[v][:-1];g.create_dataset('colors',data=np.asarray(buffers,dtype=f'S{max(map(len,buffers))}'));g.create_dataset('shape',data=[240,320,3])
    return {'frames':len(states)-1,'raw_observations':len(states),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--episode-id',required=True,type=int);p.add_argument('--gpu-lock-owner',required=True);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    from scripts.gpu_lock import owner
    assert owner()==a.gpu_lock_owner
    env=None;start=time.monotonic();result={'task':'pick_apple_messy','backend':'robotwin1','seed':a.seed,'episode_id':a.episode_id,'status':'running','source':'locally_collected_official_RoboTwin1_expert','expert_only_not_wam':True,'plan_screw':'MPLib0.2.1 Pose adaptation plus official RoboTwin1 INSTALLATION 3.2 collision-flag removal; local expert-only method'};code=2
    try:
        env=Collector(out);env.setup_demo(now_ep_num=a.episode_id,seed=a.seed,is_test=False,**config());assert not env.check_success(),'initial scene already success'
        env.record=True;env._take_picture();env.play_once();env._take_picture()
        result.update(plan_success=bool(env.plan_success),native_success=bool(env.check_success()),first_frame_visibility=env.first_visibility,observations=len(env.states),apple_z=float(env.apple.get_pose().p[2]))
        if env.plan_success and env.check_success() and env.first_visibility['head_camera']['target_pixels']>=32:
            result.update(status='accepted',**write_hdf(env,out/f'episode_{a.episode_id:07d}.hdf5'));code=0
        else:result['status']='rejected_expert_or_visibility'
    except BaseException as e:result.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        if env is not None:
            try:env.close()
            except Exception as e:result['cleanup_error']=repr(e);code=2
        result['elapsed_seconds']=time.monotonic()-start;(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
