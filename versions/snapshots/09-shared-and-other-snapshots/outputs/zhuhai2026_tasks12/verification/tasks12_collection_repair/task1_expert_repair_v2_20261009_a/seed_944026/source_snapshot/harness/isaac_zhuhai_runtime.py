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
from harness.zhuhai_robot_spec import (ASSET,ROBOT_DOF,ARM_DOF,ARM_INDICES,GRIPPER_INDICES,
    DUAL_JOINT_LIMITS,VELOCITY_LIMITS,EFFORT_LIMITS,GRIPPER_FINGER_TRAVEL,BASE_POSITIONS,
    BASE_YAW,HOME_ACTION,TOOL_FROM_HAND,robot_contract,validate_action)



class IsaacZhuhaiEnv:
    engine = {'name': 'Isaac Sim', 'version': '6.1.0.0'}
    physics_dt = .004
    action_dim = ROBOT_DOF
    arm_dof = ARM_DOF
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
        self.command = np.asarray(HOME_ACTION,dtype=float)
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
        self.joint_limits = np.asarray(DUAL_JOINT_LIMITS)
        from harness.isaac_zhuhai_expert import FrankaFK
        self.kinematics = FrankaFK()
        from harness.isaac_zhuhai_collision import PandaSelfCollision
        self.kinematics.collision_checker = PandaSelfCollision(self.kinematics)
        self.robot_contract = robot_contract()
        self.safety_events = []

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
        for label,position in BASE_POSITIONS.items():
            x,y,z=position
            parts.append(part(label+'_foot',[x,y,.018],[.24,.24,.036]))
            parts.append(part(label+'_post',[x,y,(.036+z-.024)/2],[.10,.10,z-.024-.036]))
            parts.append(part(label+'_shoulder_plate',[x,y,z-.012],[.22,.21,.024],(.35,.38,.42)))
        parts.append(part('crossbeam',[0,-.55,.56],[.88,.075,.08]))
        parts.append(part('mast_foot',[0,-.68,.018],[.20,.22,.036]))
        parts.append(part('mast',[0,-.68,.95],[.045,.045,1.90]))
        parts.append(part('mast_to_beam',[0,-.615,.56],[.055,.16,.075]))
        parts.append(part('head_camera_boom',[0,-.635,1.872],[.055,.15,.035]))
        parts.append(part('head_camera_mount',[0,-.608,1.889],[.050,.045,.045]))
        # The original tabletop was a static slab. Give it actual grounded legs.
        for i,(x,y) in enumerate([(-.59,-.39),(.59,-.39),(-.59,.39),(.59,.39)]):
            parts.append(part('table_leg_'+str(i),[x,y,.3575],[.045,.045,.715],(.28,.29,.30)))
        self.support_contract={'type':'ground-bolted fixed dual-Panda frame and grounded table',
            'static_collision_paths':parts,
            'base_world_positions_m':{k:list(v) for k,v in BASE_POSITIONS.items()},
            'base_world_yaw_rad':float(BASE_YAW),'head_mount_world_m':[0,-.58,1.85],
            'scope':'ideal rigid anchored support; original Panda chain unscaled; not a structural-strength certification'}

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
        xf.AddTranslateOp().Set(Gf.Vec3d(0,0,-.05));xf.AddScaleOp().Set(Gf.Vec3d(5,5,.1))
        ground.CreateDisplayColorAttr([Gf.Vec3f(.20,.22,.24)])
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        self._build_fixed_frame()
        light = UsdLux.DomeLight.Define(stage,'/World/Light')
        light.CreateIntensityAttr(float(900+np.random.default_rng(seed).uniform(-100,100)) if randomized else 900.)
        UsdGeom.Xform.Define(stage,'/World/Robot')
        self.arms={};self.link_paths={};self.base_joint_paths={}
        for arm,position in BASE_POSITIONS.items():
            path='/World/Robot/'+arm
            robot=UsdGeom.Xform.Define(stage,path)
            robot.GetPrim().GetReferences().AddReference(str(ASSET))
            robot.GetPrim().GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
            robot.GetPrim().GetVariantSets().GetVariantSet('Colliders').SetVariantSelection('convex_hulls')
            xf=UsdGeom.Xformable(robot);xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*position))
            yaw=Gf.Quatf(math.cos(BASE_YAW/2),Gf.Vec3f(0,0,math.sin(BASE_YAW/2)))
            xf.AddOrientOp().Set(yaw)
            art=PhysxSchema.PhysxArticulationAPI.Apply(robot.GetPrim())
            art.CreateEnabledSelfCollisionsAttr(True)
            art.CreateSolverPositionIterationCountAttr(32)
            art.CreateSolverVelocityIterationCountAttr(4)
            joint=UsdPhysics.Joint(stage.GetPrimAtPath(path+'/PhysicsFixedJoint'))
            joint.CreateBody0Rel().SetTargets([])  # fixed to world exactly at the grounded mount
            joint.CreateLocalPos0Attr(Gf.Vec3f(*position));joint.CreateLocalRot0Attr(yaw)
            self.base_joint_paths[arm]=str(joint.GetPath())
            links={p.GetName():str(p.GetPath()) for p in stage.Traverse()
                   if str(p.GetPath()).startswith(path+'/Geometry/') and p.HasAPI(UsdPhysics.RigidBodyAPI)}
            self.link_paths[arm]=links
            self.arms[arm]=self.world.scene.add(SingleArticulation(prim_path=path,
                name=arm+'_arm',reset_xform_properties=False))
            for prim in stage.Traverse():
                if not str(prim.GetPath()).startswith(path+'/'):continue
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    i=int(prim.GetName().removeprefix('panda_joint'))-1
                    drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
                    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(800.)
                    drive.CreateDampingAttr(80.);drive.CreateMaxForceAttr(float(EFFORT_LIMITS[i]))
                    PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(float(np.rad2deg(VELOCITY_LIMITS[i])))
                elif prim.IsA(UsdPhysics.PrismaticJoint):
                    drive=UsdPhysics.DriveAPI.Apply(prim,'linear')
                    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(2000.)
                    drive.CreateDampingAttr(100.);drive.CreateMaxForceAttr(35.)
                    PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(.2)
        self.task,self.seed,self.randomized = task,int(seed),bool(randomized)
        if task=='zhuhai_task1':
            from harness.isaac_zhuhai_task1 import build, RuntimeJudge
            self.manifest = build(stage,seed=seed,randomized=randomized,source_seed=source_seed)
        else:
            from harness.isaac_zhuhai_task2 import build, RuntimeJudge
            self.manifest = build(stage,seed=seed,randomized=randomized,mapping=mapping,source_seed=source_seed)
        self.gripper_paths = [self.link_paths[arm][name] for arm in ('left','right') for name in ('panda_leftfinger','panda_rightfinger')]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rb.CreateEnableCCDAttr(True)
                rb.CreateDisableGravityAttr(False)
                rb.CreateSolverPositionIterationCountAttr(16)
                rb.CreateSolverVelocityIterationCountAttr(4)
        from harness.isaac_zhuhai_camera import IsaacZhuhaiCameraRig
        self.rig = IsaacZhuhaiCameraRig(stage,seed=seed,randomized=randomized)
        self.camera_contract = self.rig.build(self.link_paths['left']['panda_hand'],self.link_paths['right']['panda_hand'])
        # Physical wrist cases are beside the hand, looking past its fingertips.
        forward=np.array([-.07,0.,.18]);forward/=np.linalg.norm(forward)
        right=np.array([0.,-1.,0.]);up=np.cross(right,forward)
        camera_rotation=np.column_stack([right,up,-forward])
        for name in ('left_camera','right_camera'):
            cam=UsdGeom.Xformable(stage.GetPrimAtPath(self.rig.paths[name]))
            matrix=np.eye(4);matrix[:3,:3]=camera_rotation;matrix[:3,3]=[.07,0.,.025]
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
            prim_paths_expr=[self.link_paths[arm]['panda_hand'] for arm in ('left','right')],
            name='actual_wrist_states',reset_xform_properties=False)
        self.judge = RuntimeJudge(self.manifest,robot_prim_paths=['/World/Robot'],
                                  gripper_prim_paths=self.gripper_paths)
        self._contact_pairs = {}
        self.safety_events=[]
        self._safety_seen=set()
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
        self.command = np.asarray(HOME_ACTION,dtype=float).copy()
        self.arm_indices = {}
        for arm,offset in [('left',0),('right',8)]:
            articulation=self.arms[arm]
            self.arm_indices[arm]=[articulation.get_dof_index(f'panda_joint{i}') for i in range(1,8)]+[
                articulation.get_dof_index('panda_finger_joint1'),articulation.get_dof_index('panda_finger_joint2')]
            positions=np.zeros(articulation.num_dof)
            positions[self.arm_indices[arm][:7]]=self.command[offset:offset+7]
            positions[self.arm_indices[arm][-2:]]=GRIPPER_FINGER_TRAVEL
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
        for arm,offset in [('left',0),('right',8)]:
            values = np.r_[action[offset:offset+7],np.repeat(action[offset+7]*GRIPPER_FINGER_TRAVEL,2)]
            self.arms[arm].apply_action(ArticulationAction(joint_positions=values,
                                      joint_indices=np.asarray(self.arm_indices[arm])))

    def step_action(self,action,interval_seconds=.1):
        action = validate_action(action,self.joint_limits)
        count = round(float(interval_seconds)/self.physics_dt)
        if count<=0 or abs(count*self.physics_dt-float(interval_seconds))>1e-8:
            raise ValueError('Control interval must be an exact positive multiple of physics_dt')
        self.command = action.copy();self._drive(action)
        for _ in range(count):
            self.world.step(render=False,update_fabric=True)
            self.physics_time += self.physics_dt
            self._audit_physics_contacts()
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

    def actual_state(self):
        result=[]
        for arm in ('left','right'):
            values=np.asarray(self.arms[arm].get_joint_positions())[self.arm_indices[arm]]
            result.extend(values[:7]);result.append(float(np.mean(values[7:])/GRIPPER_FINGER_TRAVEL))
        return np.asarray(result)

    def get_action(self):return self.command.copy()

    def gripper_pose(self,arm):
        from harness.isaac_zhuhai_task2 import quaternion_matrix
        positions,quaternions=self.wrists.get_world_poses()
        i=0 if arm=='left' else 1
        matrix=np.eye(4);matrix[:3,:3]=quaternion_matrix(quaternions[i]);matrix[:3,3]=positions[i]
        return matrix@np.asarray(TOOL_FROM_HAND)

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
        return {'rgb':images,'measured_joint_gripper_state':self.actual_state(),
                'commanded_joint_gripper_state':self.command.copy()}

    def get_obs(self):
        observation=self.policy_observation()
        return {'observation':{k:{'rgb':v} for k,v in observation['rgb'].items()},
                'measured_joint_gripper_state':observation['measured_joint_gripper_state'],
                'joint_action':{'left_arm':self.command[:7],'left_gripper':self.command[7],
                                'right_arm':self.command[8:15],'right_gripper':self.command[15]}}

    def _audit_physics_contacts(self):
        """Conservative forbidden-contact monitor; intended finger contacts remain legal."""
        def robot_arm(path):
            for arm in ('left','right'):
                if path.startswith('/World/Robot/'+arm+'/'):return arm
            return None
        for pair in self._contact_pairs:
            a,b=pair;aa,bb=robot_arm(a),robot_arm(b);reason=None
            if not aa and not bb:continue
            if aa and bb:
                if aa!=bb:reason='inter_arm_collision'
                else:
                    # Report actors, not nested mesh paths. Adjacent joint pairs
                    # and closing fingers are expected contact-filter exclusions.
                    an,bn=a.rsplit('/',1)[-1],b.rsplit('/',1)[-1]
                    if an==bn:continue
                    if all(x in {'panda_hand','panda_leftfinger','panda_rightfinger'} for x in (an,bn)):continue
                    if 'panda_hand' in (an,bn) and 'panda_link7' in (an,bn):continue
                    if an.startswith('panda_link') and bn.startswith('panda_link'):
                        if abs(int(an[-1])-int(bn[-1]))<=1:continue
                    reason='non_adjacent_self_collision'
            else:
                robot,other=(a,b) if aa else (b,a)
                finger=any(x in robot for x in ('panda_leftfinger','panda_rightfinger'))
                base=robot.rsplit('/',1)[-1]=='panda_link0'
                if other.startswith('/World/FixedSupport/') and not base:reason='robot_support_collision'
                elif other.startswith('/World/Task/Table') and not finger:reason='non_finger_table_collision'
                elif other.startswith('/World/Ground'):reason='robot_ground_collision'
            if reason is not None and (pair,reason) not in self._safety_seen:
                self._safety_seen.add((pair,reason))
                self.safety_events.append({'time_seconds':round(self.physics_time,6),'bodies':list(pair),'reason':reason})

    def physics_safety(self):
        return {'passed':not self.safety_events,'events':list(self.safety_events),
            'checks':['standard manufacturer position/velocity/effort limits','full-arm convex colliders',
                'gravity enabled','non-adjacent self/inter-arm contacts','non-finger table/ground/support contacts'],
            'scope':'actual PhysX contact monitor; not continuous swept-volume or material-damage certification'}

    def rule_state(self):
        result=self.judge.report();result['physics_safety']=self.physics_safety();return result

    def close(self):
        self._clear_scene()
        if self.app is not None:self.app.close();self.app=None
