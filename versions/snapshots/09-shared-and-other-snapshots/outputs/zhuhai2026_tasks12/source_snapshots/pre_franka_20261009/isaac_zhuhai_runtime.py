"""Isaac Sim 6.1 native dual-arm runtime; USD state/PhysX contacts are evidence.

No SAPIEN fallback, object teleporting, commanded-pose scoring, or video cameras.
The Kit application is retained across setup() calls and closed once per worker.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT/'data/isaac_zhuhai/aloha_native14_v4/usd/aloha_native14/aloha_native14.usda'


class IsaacZhuhaiEnv:
    engine = {'name': 'Isaac Sim', 'version': '6.1.0.0'}
    physics_dt = .004
    gripper_pose_is_tool = True

    def __init__(self, gpu_lock_owner, output_dir=None):
        from scripts.gpu_lock import owner
        if owner() != gpu_lock_owner:
            raise RuntimeError('Isaac worker requires the external shared GPU lock')
        self.gpu_lock_owner = gpu_lock_owner
        self.output_dir = Path(output_dir) if output_dir else None
        self.world = self.rig = self.stage = None
        self._contact_subscription = None
        self.physics_time = 0.
        self.command = np.r_[np.zeros(6),1.,np.zeros(6),1.]
        self.on_control_step = None
        import isaacsim
        deprecated = Path(isaacsim.__file__).parent/'extsDeprecated'
        from isaacsim import SimulationApp
        self.app = SimulationApp({'headless': True, 'width': 320, 'height': 240,
            'multi_gpu': False, 'renderer': 'RaytracedLighting', 'disable_viewport_updates': True,
            'extra_args': [f'--ext-folder={deprecated}', '--enable=isaacsim.core.api',
                           '--enable=isaacsim.core.prims']})
        import carb
        settings = carb.settings.get_settings()
        settings.set('/physics/updateToUsd', True)
        settings.set('/physics/updateVelocitiesToUsd', True)
        settings.set('/rtx/post/dlss/execMode', 0)
        settings.set('/rtx/rendermode', 'RaytracedLighting')
        self.joint_limits = np.tile([-10.,10.],(12,1))

    def _build_fixed_frame(self):
        from pxr import Gf,UsdGeom,UsdPhysics
        UsdGeom.Xform.Define(self.stage,'/World/FixedSupport')
        def part(name,center,size,color=(.18,.21,.24),collision=True):
            path='/World/FixedSupport/'+name
            cube=UsdGeom.Cube.Define(self.stage,path);cube.CreateSizeAttr(1.)
            xf=UsdGeom.Xformable(cube);xf.AddTranslateOp().Set(Gf.Vec3d(*center));xf.AddScaleOp().Set(Gf.Vec3d(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if collision:UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            return path
        parts=[]
        for label,x,y,z in [('left',-.297,-.4195,.782),('right',.3063,-.4185,.781)]:
            # Ground-bolted posts behind the table with short forward brackets.
            parts.append(part(label+'_foot',[x,-.52,.018],[.17,.22,.036]))
            parts.append(part(label+'_post',[x,-.52,.39],[.065,.065,.74]))
            parts.append(part(label+'_bracket',[x,(y-.52)/2,.752],[.075,y+.52+.08,.044]))
            parts.append(part(label+'_shoulder_plate',[x,y,z-.012],[.115,.10,.024],(.35,.38,.42)))
        parts.append(part('crossbeam',[.005,-.52,.63],[.76,.07,.085]))
        parts.append(part('mast_foot',[0,-.65,.018],[.20,.22,.036]))
        parts.append(part('mast',[0,-.65,.86],[.045,.045,1.66]))
        parts.append(part('mast_to_beam',[0,-.585,.63],[.055,.16,.075]))
        parts.append(part('head_camera_boom',[0,-.615,1.69],[.055,.13,.035]))
        parts.append(part('head_camera_mount',[0,-.59,1.724],[.050,.045,.045]))
        self.support_contract={'type':'ground-bolted fixed dual-arm frame',
            'static_collision_paths':parts,'base_world_position_m':[0,-.65,0],
            'shoulder_mounts_world_m':{'left':[-.297,-.4195,.782],'right':[.3063,-.4185,.781]},
            'head_mount_world_m':[0,-.58,1.72],
            'scope':'explicit ideal rigid fixed frame, not CAD structural-strength certification'}

    def _camera_housings(self):
        from pxr import Gf,UsdGeom,UsdPhysics
        for name,path in self.rig.paths.items():
            housing=UsdGeom.Cube.Define(self.stage,path+'/Housing')
            housing.CreateSizeAttr(1.)
            xf=UsdGeom.Xformable(housing)
            xf.AddTranslateOp().Set(Gf.Vec3d(0,0,.024))
            xf.AddScaleOp().Set(Gf.Vec3d(.038,.026,.018))
            housing.CreateDisplayColorAttr([Gf.Vec3f(.03,.035,.04)])
            UsdPhysics.CollisionAPI.Apply(housing.GetPrim())
            # Visual sensor cases are fixed children of the actual rigid link.
            # Lens points along camera -Z; the casing stays behind the lens.
    def _clear_scene(self):
        self._contact_subscription = None
        if self.rig is not None:
            self.rig.close(); self.rig = None
        if self.world is not None:
            self.world.stop()
            self.world.clear()
            self.world.clear_instance()
            self.world = None
        self.on_control_step = None

    def setup(self, task, seed, randomized=False, mapping='body_text', source_seed=None):
        if task not in {'zhuhai_task1','zhuhai_task2'}:
            raise ValueError(task)
        self._clear_scene()
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
        import omni.usd
        create_new_stage()
        self.stage = stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.Xform.Define(stage,'/World')
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        self.world = World(physics_dt=self.physics_dt,rendering_dt=.1,
                           stage_units_in_meters=1.,backend='numpy',device='cpu')
        physics = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/physicsScene'))
        physics.CreateEnableCCDAttr(True)
        physics.CreateEnableGPUDynamicsAttr(False)
        ground = UsdGeom.Cube.Define(stage,'/World/Ground')
        ground.CreateSizeAttr(1.)
        xf = UsdGeom.Xformable(ground)
        xf.AddTranslateOp().Set(Gf.Vec3d(0,0,-.06));xf.AddScaleOp().Set(Gf.Vec3d(5,5,.1))
        ground.CreateDisplayColorAttr([Gf.Vec3f(.20,.22,.24)])
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        self._build_fixed_frame()
        light = UsdLux.DomeLight.Define(stage,'/World/Light')
        light.CreateIntensityAttr(float(900+np.random.default_rng(seed).uniform(-100,100)) if randomized else 900.)
        robot = UsdGeom.Xform.Define(stage,'/World/Robot')
        robot.GetPrim().GetReferences().AddReference(str(ASSET))
        robot.GetPrim().GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
        xf = UsdGeom.Xformable(robot);xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0,-.65,0))
        xf.AddOrientOp().Set(Gf.Quatf(math.sqrt(.5),Gf.Vec3f(0,0,math.sqrt(.5))))
        for prefix,position in [('fl',[-.297,-.4195,.782]),('fr',[.3063,-.4185,.781])]:
            joint=UsdPhysics.Joint(stage.GetPrimAtPath('/World/Robot/Physics/'+prefix+'_base_joint'))
            local_rotation=joint.GetLocalRot0Attr().Get()
            joint.CreateBody0Rel().SetTargets([Sdf.Path('/World/FixedSupport')])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*position))
            joint.CreateLocalRot0Attr(Gf.Quatf(math.sqrt(.5),Gf.Vec3f(0,0,math.sqrt(.5)))*local_rotation)
        self.link_paths = {p.GetName():str(p.GetPath()) for p in stage.Traverse()
                           if str(p.GetPath()).startswith('/World/Robot/Geometry/')}
        self.arms = {}
        for arm,prefix in [('left','fl'),('right','fr')]:
            self.arms[arm] = self.world.scene.add(SingleArticulation(
                prim_path=self.link_paths[prefix+'_base_link'],name=arm+'_arm',reset_xform_properties=False))
        # The original asset has a mobile base. Keep it fixed; do not drive wheels.
        self.task,self.seed,self.randomized = task,int(seed),bool(randomized)
        if task=='zhuhai_task1':
            from harness.isaac_zhuhai_task1 import build, RuntimeJudge
            self.manifest = build(stage,seed=seed,randomized=randomized,source_seed=source_seed)
        else:
            from harness.isaac_zhuhai_task2 import build, RuntimeJudge
            self.manifest = build(stage,seed=seed,randomized=randomized,mapping=mapping,source_seed=source_seed)
        self.gripper_paths = [self.link_paths[f'{p}_link{i}'] for p in ('fl','fr') for i in (7,8)]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rb.CreateEnableCCDAttr(True)
                rb.CreateSolverPositionIterationCountAttr(16)
                rb.CreateSolverVelocityIterationCountAttr(4)
        from harness.isaac_zhuhai_camera import IsaacZhuhaiCameraRig
        self.rig = IsaacZhuhaiCameraRig(stage,seed=seed,randomized=randomized)
        self.camera_contract = self.rig.build(self.link_paths['fl_link6'],self.link_paths['fr_link6'])
        # Wrist cameras look along tool +X with local Z as image-up.
        camera_rotation = np.array([[0,0,-1],[-1,0,0],[0,1,0]],dtype=float)
        for name in ('left_camera','right_camera'):
            cam = UsdGeom.Xformable(stage.GetPrimAtPath(self.rig.paths[name]))
            matrix = np.eye(4);matrix[:3,:3]=camera_rotation;matrix[:3,3]=[.035,0,.045]
            cam.ClearXformOpOrder();cam.AddTransformOp().Set(Gf.Matrix4d(*matrix.T.reshape(-1).tolist()))
        self._camera_housings()
        self.camera_contract['physical_housings']=True
        self.camera_contract['support']=self.support_contract
        # Kit creates editor cameras even headless; they are never sensor inputs.
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdGeom.Camera) and str(prim.GetPath()).startswith('/OmniverseKit_'):
                with Usd.EditContext(stage,stage.GetSessionLayer()):
                    stage.OverridePrim(prim.GetPath()).SetActive(False)
        cameras = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
        if set(cameras) != set(self.rig.paths.values()):
            raise RuntimeError(f'Expected exactly three onboard cameras, found {cameras}')
        paths = {k:v.get('prim_path',v.get('body_path')) for k,v in self.manifest['objects'].items()}
        if task=='zhuhai_task1':
            paths.update({f'lid_{k}':v['lid_path'] for k,v in self.manifest['bins'].items()})
        self.body_names = list(paths)
        self.rigid = self.world.scene.add(RigidPrim(prim_paths_expr=list(paths.values()),
                    name='task_body_states',reset_xform_properties=False))
        self.wrists = RigidPrim(
            prim_paths_expr=[self.link_paths[p+'_link6'] for p in ('fl','fr')],
            name='actual_wrist_states',reset_xform_properties=False)
        self.judge = RuntimeJudge(self.manifest,robot_prim_paths=['/World/Robot'],
                                  gripper_prim_paths=self.gripper_paths)
        self._contact_pairs = {}
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools
        def contact_callback(headers,data):
            for header in headers:
                pair = tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                                     str(PhysicsSchemaTools.intToSdfPath(header.actor1)))))
                if header.type==ContactEventType.CONTACT_LOST:
                    self._contact_pairs.pop(pair,None)
                elif header.num_contact_data>0:
                    self._contact_pairs[pair]={'body0_path':pair[0],'body1_path':pair[1],
                                              'point_count':int(header.num_contact_data)}
        self._contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
        self.world.reset()
        self.wrists.initialize()
        self.command = np.array([2.,0,0,0,0,0,1.,-2.,0,0,0,0,0,1.])
        self.arm_indices = {}
        for arm,prefix in [('left','fl'),('right','fr')]:
            articulation = self.arms[arm]
            self.arm_indices[arm] = [articulation.get_dof_index(f'{prefix}_joint{i}') for i in range(1,9)]
            positions = np.zeros(articulation.num_dof)
            positions[self.arm_indices[arm][:6]] = self.command[:6] if arm=='left' else self.command[7:13]
            positions[self.arm_indices[arm][-2:]] = .04765
            articulation.set_joint_positions(positions)
        self._drive(self.command)
        # Initial settling is not scored and is not a demonstration frame.
        for _ in range(125): self.world.step(render=False,update_fabric=True)
        self.physics_time = 0.
        self.judge.observe(0.,self.body_states(),list(self._contact_pairs.values()))
        self.rig.attach_render_products()
        for _ in range(8): self.world.render()
        self.capture()
        return self.manifest

    def _drive(self,action):
        from isaacsim.core.utils.types import ArticulationAction
        for arm,offset in [('left',0),('right',7)]:
            values = np.r_[action[offset:offset+6],np.repeat(action[offset+6]*.04765,2)]
            self.arms[arm].apply_action(ArticulationAction(joint_positions=values,
                                      joint_indices=np.asarray(self.arm_indices[arm])))

    def step_native14(self,action,interval_seconds=.1):
        from harness.robotwin_qpos_adapter import validate_action
        action = validate_action(action,self.joint_limits)
        count = round(float(interval_seconds)/self.physics_dt)
        if count<=0 or abs(count*self.physics_dt-float(interval_seconds))>1e-8:
            raise ValueError('Control interval must be an exact positive multiple of physics_dt')
        self.command = action.copy();self._drive(action)
        for _ in range(count):
            self.world.step(render=False,update_fabric=True)
            self.physics_time += self.physics_dt
            self.judge.observe(self.physics_dt,self.body_states(),list(self._contact_pairs.values()))
        if self.on_control_step is not None:self.on_control_step(self)
        return self.rule_state()

    def body_states(self):
        positions,quaternions = self.rigid.get_world_poses()
        velocities = self.rigid.get_velocities()
        return {name:{'position_world_m':np.asarray(positions[i]).tolist(),
                      'quaternion_wxyz':np.asarray(quaternions[i]).tolist(),
                      'linear_velocity_m_s':np.asarray(velocities[i,:3]).tolist(),
                      'angular_velocity_rad_s':np.asarray(velocities[i,3:]).tolist()}
                for i,name in enumerate(self.body_names)}

    def actual_native14(self):
        result=[]
        for arm in ('left','right'):
            values=np.asarray(self.arms[arm].get_joint_positions())[self.arm_indices[arm]]
            result.extend(values[:6]);result.append(float(np.mean(values[6:])/.04765))
        return np.asarray(result)

    def get_native14(self):return self.command.copy()
    native14_state = get_native14

    def gripper_pose(self,arm):
        from harness.isaac_zhuhai_task2 import quaternion_matrix
        positions,quaternions=self.wrists.get_world_poses()
        i=0 if arm=='left' else 1
        matrix=np.eye(4);matrix[:3,:3]=quaternion_matrix(quaternions[i]);matrix[:3,3]=positions[i]
        matrix[:3,3]+=matrix[:3,:3]@np.array([.12,0,0])
        return matrix

    def capture(self):
        before=self.world.current_time
        # Drain the render/readback pipeline while physics remains frozen.
        for _ in range(3):self.world.render()
        images,evidence=self.rig.capture(self.judge.camera_geometry())
        if abs(self.world.current_time-before)>1e-8:
            raise RuntimeError('Rendering unexpectedly advanced physics')
        if self.physics_time==0.:self.judge.head_camera_coverage=evidence
        self.last_images,self.camera_evidence=images,evidence
        return images,evidence

    def policy_observation(self):
        images,_=self.capture()
        return {'rgb':images,'commanded_joint_gripper_state':self.command.copy()}

    def get_obs(self):
        observation=self.policy_observation()
        return {'observation':{k:{'rgb':v} for k,v in observation['rgb'].items()},
                'joint_action':{'left_arm':self.command[:6],'left_gripper':self.command[6],
                                'right_arm':self.command[7:13],'right_gripper':self.command[13]}}

    def rule_state(self):return self.judge.report()

    def close(self):
        self._clear_scene()
        if self.app is not None:self.app.close();self.app=None
