"""Smooth native14 physical expert from Aloha URDF FK and damped IK.

CPU-safe planner. Runtime commands real joint drives at 10Hz; no actor/root
poses, constraints attaching objects, or state-reset methods are used.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline

ROOT=Path(__file__).resolve().parents[1]
URDF=ROOT/'third_party/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'


def world_base_matrix(position=(0,-.65,0),quaternion_wxyz=(.707,0,0,.707)):
    q=np.asarray(quaternion_wxyz);q=q/np.linalg.norm(q)
    result=np.eye(4);result[:3,:3]=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix();result[:3,3]=position
    return result


@dataclass
class IKResult:
    success: bool
    joints: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int


class URDFFK:
    def __init__(self,urdf_path=URDF,base_world=None,tool_offset=(.12,0,0)):
        self.path=Path(urdf_path);self.base=world_base_matrix() if base_world is None else np.asarray(base_world)
        self.tool=np.eye(4);self.tool[:3,3]=tool_offset
        root=ET.parse(self.path).getroot();self.joints={};self.by_child={}
        for node in root.findall('joint'):
            origin=node.find('origin');xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();transform[:3,3]=xyz
            axis=node.find('axis');axis=np.fromstring(axis.get('xyz','0 0 1'),sep=' ') if axis is not None else np.array([0.,0.,1.])
            limit=node.find('limit');bounds=(float(limit.get('lower','-3.14159')),float(limit.get('upper','3.14159'))) if limit is not None else (-math.pi,math.pi)
            data={'name':node.get('name'),'parent':node.find('parent').get('link'),'child':node.find('child').get('link'),
                'kind':node.get('type'),'origin':transform,'axis':axis,'limits':bounds}
            self.joints[data['name']]=data;self.by_child[data['child']]=data
        self.chains={};self.limits={}
        for arm,prefix in [('left','fl'),('right','fr')]:
            chain=[];child=prefix+'_link6'
            while child in self.by_child:
                joint=self.by_child[child];chain.append(joint);child=joint['parent']
            self.chains[arm]=list(reversed(chain))
            self.limits[arm]=np.array([self.joints[f'{prefix}_joint{i}']['limits'] for i in range(1,7)])

    def fk(self,arm,q,tool=True,return_jacobian=False):
        q=np.asarray(q);prefix='fl' if arm=='left' else 'fr';matrix=self.base.copy();axes=[];origins=[]
        for joint in self.chains[arm]:
            matrix=matrix@joint['origin'];name=joint['name']
            if name in [f'{prefix}_joint{i}' for i in range(1,7)]:
                index=int(name.rsplit('joint',1)[1])-1
                origins.append(matrix[:3,3].copy());axes.append(matrix[:3,:3]@joint['axis'])
                rotation=np.eye(4);rotation[:3,:3]=Rotation.from_rotvec(joint['axis']*q[index]).as_matrix();matrix=matrix@rotation
        if tool:matrix=matrix@self.tool
        if not return_jacobian:return matrix
        jac=np.zeros((6,6))
        for i,(axis,origin) in enumerate(zip(axes,origins)):
            jac[:3,i]=np.cross(axis,matrix[:3,3]-origin);jac[3:,i]=axis
        return matrix,jac

    def ik(self,arm,target,seed=None,position_tolerance=.002,orientation_tolerance=.025,max_iterations=240):
        target=np.asarray(target);seed=np.zeros(6) if seed is None else np.asarray(seed,dtype=float)
        legal=self.limits[arm];lower=np.maximum(legal[:,0],-2*math.pi+.01);upper=np.minimum(legal[:,1],2*math.pi-.01)
        candidates=[seed,np.array([seed[0],-.7,1.4,-.7,0,seed[5]]),
                    np.array([seed[0],.7,-1.4,.7,0,seed[5]])]
        best=None
        for start in candidates:
            q=np.clip(start,lower,upper)
            for iteration in range(max_iterations):
                current,jac=self.fk(arm,q,return_jacobian=True)
                dp=target[:3,3]-current[:3,3]
                dr=Rotation.from_matrix(target[:3,:3]@current[:3,:3].T).as_rotvec()
                pe,oe=float(np.linalg.norm(dp)),float(np.linalg.norm(dr))
                result=IKResult(pe<position_tolerance and oe<orientation_tolerance,q.copy(),pe,oe,iteration+1)
                quality=pe+.15*oe+.0001*np.linalg.norm(q-seed)
                if best is None or quality<best[0]:best=(quality,result)
                if result.success:
                    # Keep revolute targets close to prior joints, avoiding full turns.
                    for i in range(6):
                        options=[q[i]+2*math.pi*k for k in [-1,0,1] if lower[i]<=q[i]+2*math.pi*k<=upper[i]]
                        q[i]=min(options,key=lambda value:abs(value-seed[i]))
                    result.joints=q.copy();return result
                error=np.r_[dp,dr*.15];weighted=jac.copy();weighted[3:]*=.15
                damping=.015
                delta=weighted.T@np.linalg.solve(weighted@weighted.T+damping*damping*np.eye(6),error)
                delta=np.clip(delta,-.13,.13)
                q=np.clip(q+delta,lower,upper)
        return best[1]


def minimum_jerk(start,target,dt=.1,max_joint_velocity=.75,max_joint_acceleration=1.5,max_gripper_velocity=.7):
    """Native14 minimum-jerk samples with analytic velocity/acceleration bounds."""
    start,target=np.asarray(start,dtype=float),np.asarray(target,dtype=float)
    delta=target-start;arm_indices=list(range(6))+list(range(7,13));g=[6,13]
    distance=float(np.max(np.abs(delta[arm_indices])))
    duration=max(dt,1.875*distance/max_joint_velocity,
        math.sqrt(5.7735027*distance/max_joint_acceleration),
        1.875*float(np.max(np.abs(delta[g])))/max_gripper_velocity)
    count=max(1,math.ceil(duration/dt));duration=count*dt
    t=np.arange(1,count+1)/count;s=10*t**3-15*t**4+6*t**5
    return start[None,:]+s[:,None]*delta[None,:],duration


def top_down_rotation(jaw_axis):
    x=np.array([0.,0.,-1.]);y=np.asarray(jaw_axis,dtype=float);y[2]=0;y/=np.linalg.norm(y)
    return np.column_stack([x,y,np.cross(x,y)])


def grasp_pose_candidates(key,spec,object_world_matrix,table_surface):
    """Common rim/handle grasps; tip extension leaves the tabletop clear."""
    matrix=np.asarray(object_world_matrix,dtype=float);scale=np.asarray(spec['scale'])
    if key=='bowl':local=np.array([0.,.95,-1.60])*scale;jaw=matrix[:3,2]
    elif key=='cup':local=np.array([0.,.95,.459])*scale;jaw=matrix[:3,2]
    elif key=='plate':local=np.array([0.,.62,4.15])*scale;jaw=matrix[:3,2]
    elif key=='spoon':local=np.array([-40.,0.,5.])*scale;jaw=matrix[:3,1]
    else:
        corners=np.asarray(spec.get('local_geometry_points_m',spec['local_corners_m']))
        local=(corners.min(0)+corners.max(0))/2;jaw=np.array([1.,0.,0.])
    point=matrix[:3,:3]@local+matrix[:3,3]
    if key in {'plate','spoon'}:point[2]=max(point[2]+.020,table_surface+.041)
    result=[]
    for sign in [1,-1]:
        pose=np.eye(4);pose[:3,:3]=top_down_rotation(jaw*sign);pose[:3,3]=point
        result.append(pose)
    return result


def object_destination_pose(key,spec,manifest):
    color=TaskDestinations(manifest)[key];bounds=manifest['bins'][color]
    center=(np.asarray(bounds['interior_min_world_m'])+np.asarray(bounds['interior_max_world_m']))/2
    pose=np.eye(4)
    if key!='spoon':pose[:3,:3]=np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=float)
    else:pose[:3,:3]=Rotation.from_euler('z',spec.get('yaw_rad',0.)).as_matrix()
    if color=='white':center[0]+=-.035 if key=='bowl' else .10
    if key=='spoon' and color=='white':
        # Preserve the spoon's yaw instead of rolling the thin shaft in the jaw.
        # The rear strip is clear of the bowl and contains the whole spoon hull.
        center[0]=-.285;center[1]=.045
    if key=='bowl' and color=='white':
        center[0]=-.42;center[1]=-.025
    if key=='cup':center[0]+=.07
    local=np.asarray(spec['local_geometry_points_m'])
    clearance=.010 if key in {'plate','spoon'} else .005
    center[2]=bounds['floor_world_m']-np.min(local@pose[:3,:3].T,axis=0)[2]+clearance
    pose[:3,3]=center
    return pose


def TaskDestinations(manifest):
    from harness.zhuhai_task2_rules import LIFE_DESTINATIONS
    return LIFE_DESTINATIONS[manifest.get('mapping','body_text')]


class SmoothExpert:
    def __init__(self,env,dt=.1,max_joint_velocity=.75,max_joint_acceleration=1.5,max_gripper_velocity=.7):
        self.env=env;self.dt=float(dt);self.manifest=env.manifest
        self.fk_model=URDFFK(base_world=getattr(env,'robot_base_world_matrix',None))
        self.max_velocity=max_joint_velocity;self.max_acceleration=max_joint_acceleration;self.max_gripper_velocity=max_gripper_velocity
        self.current=self._read_native14();self.trace=[];self.plan_success=True;self.last_failure=None
        self.motion_commands=[]

    def _read_native14(self):
        for name in ['get_native14','native14_state']:
            method=getattr(self.env,name,None)
            if callable(method):
                value=np.asarray(method(),dtype=float)
                if value.shape==(14,):return value.copy()
        for name in ['current_action','last_command','target14','commanded_state']:
            value=getattr(self.env,name,None)
            if value is not None and not callable(value):
                value=np.asarray(value,dtype=float)
                if value.shape==(14,):return value.copy()
        return np.zeros(14)

    def _arm_slice(self,arm):return slice(0,6) if arm=='left' else slice(7,13)
    def _gripper_index(self,arm):return 6 if arm=='left' else 13

    def _actual_tool_pose(self,arm):
        method=getattr(self.env,'gripper_pose',None)
        if callable(method):
            value=method(arm)
            if isinstance(value,dict):value=value.get('world_matrix')
            matrix=np.asarray(value,dtype=float)
            if matrix.shape==(4,4):
                return matrix if getattr(self.env,'gripper_pose_is_tool',False) else matrix@self.fk_model.tool
        return self.fk_model.fk(arm,self.current[self._arm_slice(arm)])

    def _drive(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(14,) or not np.isfinite(action).all():raise ValueError('Invalid native14 expert command')
        for arm in ['left','right']:
            q=action[self._arm_slice(arm)];limit=self.fk_model.limits[arm]
            if np.any(q<limit[:,0]) or np.any(q>limit[:,1]):raise ValueError('Expert command violates source URDF limits')
        if np.any(action[[6,13]]<0) or np.any(action[[6,13]]>1):raise ValueError('Expert gripper commands outside [0,1]')
        self.env.step_native14(action,self.dt);self.current=action.copy();self.motion_commands.append(action.tolist())

    def move_joints(self,target,label):
        samples,duration=minimum_jerk(self.current,target,self.dt,self.max_velocity,self.max_acceleration,self.max_gripper_velocity)
        for sample in samples:self._drive(sample)
        self.trace.append({'phase':label,'duration_seconds':duration,'samples':len(samples),'target_native14':np.asarray(target).tolist()})
        key=label.split(':',1)[0]
        if key in self.manifest['objects']:
            self.trace[-1]['physical_object_state']=self.env.rule_state()['objects'][key]

    def move_pose(self,arm,pose,label):
        result=self.fk_model.ik(arm,pose,seed=self.current[self._arm_slice(arm)])
        if not result.success:
            raise RuntimeError(f'{label}: {arm} IK failed position={result.position_error_m:.6f}m orientation={result.orientation_error_rad:.6f}rad')
        target=self.current.copy();target[self._arm_slice(arm)]=result.joints
        self.move_joints(target,label)
        self.trace[-1].update(ik_position_error_m=result.position_error_m,ik_orientation_error_rad=result.orientation_error_rad,
            target_tool_world_matrix=np.asarray(pose).tolist())

    def gripper(self,arm,value,label):
        target=self.current.copy();target[self._gripper_index(arm)]=value;self.move_joints(target,label)

    def hold(self,seconds,label):
        count=math.ceil(seconds/self.dt)
        for _ in range(count):self._drive(self.current)
        self.trace.append({'phase':label,'duration_seconds':count*self.dt,'samples':count})

    def move_pose_path(self,arm,poses,label,minimum_seconds=3.5):
        """Traverse contact arcs with a minimum-jerk time law and bounded joints."""
        joints=[self.current.copy()];seed=self.current[self._arm_slice(arm)].copy()
        for pose in poses:
            result=self.fk_model.ik(arm,pose,seed=seed)
            if not result.success:
                raise RuntimeError(f'{label}: arc IK failed position={result.position_error_m:.6f}m orientation={result.orientation_error_rad:.6f}rad')
            native=self.current.copy();native[self._arm_slice(arm)]=result.joints;joints.append(native);seed=result.joints
        spline=CubicSpline(np.linspace(0,1,len(joints)),np.asarray(joints),axis=0,bc_type='clamped')
        u=np.linspace(0,1,1001);s=10*u**3-15*u**4+6*u**5
        ds=30*u*u-60*u**3+30*u**4;dds=60*u-180*u*u+120*u**3
        dq=spline(s,1)*ds[:,None];ddq=spline(s,2)*ds[:,None]**2+spline(s,1)*dds[:,None]
        ids=list(range(6))+list(range(7,13))
        duration=max(minimum_seconds,float(np.max(np.abs(dq[:,ids])))/self.max_velocity,
            math.sqrt(float(np.max(np.abs(ddq[:,ids])))/self.max_acceleration))
        count=math.ceil(duration/self.dt);duration=count*self.dt
        u=np.arange(1,count+1)/count;s=10*u**3-15*u**4+6*u**5
        for native in spline(s):self._drive(native)
        self.trace.append({'phase':label,'duration_seconds':duration,'samples':count,
            'path_nodes':len(poses),'path_actuation':'real joint drives; no lid/object pose writes'})

    def _body_matrix(self,key):
        states=self.env.body_states();data=states[key]
        if 'world_matrix' in data:return np.asarray(data['world_matrix'],dtype=float)
        from harness.isaac_zhuhai_task2 import quaternion_matrix
        matrix=np.eye(4);matrix[:3,:3]=quaternion_matrix(data['quaternion_wxyz']);matrix[:3,3]=data['position_world_m'];return matrix

    def _choose_grasp(self,key,spec,arm):
        body=self._body_matrix(key);candidates=grasp_pose_candidates(key,spec,body,self.manifest['table_surface_world_m'])
        choices=[]
        for pose in candidates:
            pre=pose.copy();pre[2,3]+=.11
            result=self.fk_model.ik(arm,pre,seed=self.current[self._arm_slice(arm)])
            if result.success:
                grasp=self.fk_model.ik(arm,pose,seed=result.joints)
                if grasp.success:choices.append((np.linalg.norm(result.joints-self.current[self._arm_slice(arm)]),pose))
        if not choices:raise RuntimeError(f'{key}: no reachable collision-clear pregrasp candidate')
        return min(choices,key=lambda item:item[0])[1]

    def _task2_once(self):
        target=self.current.copy();target[[6,13]]=1.;self.move_joints(target,'open_both_grippers')
        # Same-side bowl/spoon and cup/plate; avoid cross-table transfers.
        for key in ['bowl','spoon','cup','plate']:
            spec=self.manifest['objects'][key];arm='left' if key in ['bowl','spoon'] else 'right'
            grasp=self._choose_grasp(key,spec,arm)
            pre=grasp.copy();pre[2,3]+=.11
            self.move_pose(arm,pre,f'{key}:pregrasp')
            self.move_pose(arm,grasp,f'{key}:approach')
            self.gripper(arm,0.,f'{key}:close')
            lift=self._actual_tool_pose(arm).copy();lift[2,3]+=.18
            self.move_pose(arm,lift,f'{key}:lift')
            self.hold(5.2,f'{key}:physical_continuous_hold')
            item=self.env.rule_state()['score']['items'][key]
            if not item['grasp_awarded']:
                raise RuntimeError(f'{key}: no actual gripper-supported 5s lifted hold; score={item}')
            body=self._body_matrix(key);tool=self._actual_tool_pose(arm)
            object_to_tool=np.linalg.inv(body)@tool
            destination=object_destination_pose(key,spec,self.manifest)
            if key=='spoon':
                # Thin handles can roll in a parallel jaw. Preserve the measured
                # grasp orientation and translate it; the spoon settles under
                # gravity after release instead of demanding a wrist flip.
                destination[:3,:3]=body[:3,:3]
                rotated=np.asarray(spec['local_geometry_points_m'])@body[:3,:3].T
                center=(rotated.min(0)+rotated.max(0))/2
                desired_xy=np.array([-.275,.055]) if TaskDestinations(self.manifest)[key]=='white' else np.array([.34,.02])
                destination[:2,3]=desired_xy-center[:2]
                destination[2,3]=self.manifest['bins'][TaskDestinations(self.manifest)[key]]['floor_world_m']-rotated[:,2].min()+.015
            place=destination@object_to_tool
            transit=place.copy();transit[2,3]+=.08
            self.move_pose(arm,transit,f'{key}:raised_transfer')
            if not self.env.rule_state()['objects'][key]['touching_gripper']:
                raise RuntimeError(f'{key}: physical grasp lost during raised transfer')
            self.move_pose(arm,place,f'{key}:placement')
            self.gripper(arm,1.,f'{key}:release')
            retreat=self._actual_tool_pose(arm).copy();retreat[2,3]+=.08
            self.move_pose(arm,retreat,f'{key}:retreat')
            self.hold(2.0,f'{key}:released_stability')
            item=self.env.rule_state()['score']['items'][key]
            if not item['placement_valid']:
                raise RuntimeError(f'{key}: released stable containment/upright/support criterion failed; score={item}')

    def _task1_once(self):
        target=self.current.copy();target[[6,13]]=1.;self.move_joints(target,'open_both_grippers')
        for color,arm in [('dry','left'),('wet','right')]:
            spec=self.manifest['bins'][color]
            grasp=np.asarray(spec['closed_grasp_transform_world'],dtype=float)
            pre=grasp.copy();pre[:3,3]-=grasp[:3,0]*.10
            self.move_pose(arm,pre,f'lid_{color}:pregrasp')
            self.move_pose(arm,grasp,f'lid_{color}:approach')
            self.gripper(arm,0.,f'lid_{color}:close_on_real_handle')
            arc=spec['expert_open_arc']
            self.move_pose_path(arm,arc['gripper_world_pose_nodes'],f'lid_{color}:passive_hinge_contact_arc',arc['seconds'])
            self.hold(.3,f'lid_{color}:past_vertical_settle')
            self.gripper(arm,1.,f'lid_{color}:release')
            retreat=self._actual_tool_pose(arm).copy();retreat[:3,3]-=retreat[:3,0]*.08
            self.move_pose(arm,retreat,f'lid_{color}:retreat')
            self.hold(.5,f'lid_{color}:released_lid_check')
            if not self.env.rule_state()['score']['lids'][color]['open_now']:
                raise RuntimeError(f'lid_{color}: real passive hinge did not remain open after contact release')
        for key,spec in self.manifest['objects'].items():
            arm=spec['expert_arm'];body=self._body_matrix(key)
            grasp=np.eye(4);grasp[:3,:3]=spec['grasp_tcp_world_rotation']
            grasp[:3,3]=body[:3,:3]@np.asarray(spec['grasp_point_local_m'])+body[:3,3]
            if key=='banana_peel':grasp[2,3]=max(grasp[2,3],self.manifest['table_surface_world_m']+.041)
            pre=grasp.copy();pre[2,3]+=.10
            self.move_pose(arm,pre,f'{key}:pregrasp');self.move_pose(arm,grasp,f'{key}:approach')
            self.gripper(arm,0.,f'{key}:close')
            lift=self._actual_tool_pose(arm).copy();lift[2,3]+=.20
            self.move_pose(arm,lift,f'{key}:lift');self.hold(5.2,f'{key}:physical_continuous_hold')
            if not self.env.rule_state()['score']['items'][key]['grasp_awarded']:
                raise RuntimeError(f'{key}: real physical 5s grasp hold failed')
            tilted=self._actual_tool_pose(arm).copy();tilted[:3,:3]=spec['drop_tcp_world_rotation']
            self.move_pose(arm,tilted,f'{key}:smooth_carry_tilt')
            drop=np.eye(4);drop[:3,:3]=spec['drop_tcp_world_rotation'];drop[:3,3]=spec['drop_tcp_position_world_m']
            # Preserve the actual grasp offset of thin banana cores as measured.
            if key=='banana_peel':
                actual=self._actual_tool_pose(arm);actual_body=self._body_matrix(key)
                offset=actual_body[:3,:3].T@(actual[:3,3]-actual_body[:3,3])
                from harness.isaac_zhuhai_task2 import quaternion_matrix
                desired_rotation=quaternion_matrix(spec['desired_object_orientation_wxyz'])
                center=np.asarray(spec['drop_position_world_m']).copy()
                center[2]-=.025  # Thin peel still clears the mouth; leaves IK height margin.
                drop[:3,3]=center+desired_rotation@offset
            predrop=drop.copy();predrop[2,3]+=.06
            self.move_pose(arm,predrop,f'{key}:above_open_mouth');self.move_pose(arm,drop,f'{key}:drop_pose')
            self.gripper(arm,1.,f'{key}:release_through_open_mouth')
            retreat=self._actual_tool_pose(arm).copy();retreat[:3,3]-=retreat[:3,0]*.06
            self.move_pose(arm,retreat,f'{key}:retreat');self.hold(2.,f'{key}:released_bin_stability')
            if not self.env.rule_state()['score']['items'][key]['placement_valid']:
                raise RuntimeError(f'{key}: real open-mouth passage/fully contained stable release failed')

    def play_once(self):
        try:
            if self.manifest.get('task')=='zhuhai_task2':self._task2_once()
            elif self.manifest.get('task')=='zhuhai_task1':self._task1_once()
            else:raise ValueError('Unsupported Zhuhai task')
        except Exception as exc:
            self.plan_success=False;self.last_failure=f'{type(exc).__name__}: {exc}'
            self.trace.append({'phase':'failure','reason':self.last_failure})
        if hasattr(self.env,'plan_success'):self.env.plan_success=self.plan_success
        evidence=self.env.rule_state()
        return {'expert':'URDF DLS IK with joint-drive minimum-jerk native14 commands','plan_success':self.plan_success,
            'full_task_success':bool(evidence['score']['full_task_success']),'failure':self.last_failure,
            'trace':self.trace,'command_count':len(self.motion_commands),'record_video':False,
            'limits':{'joint_velocity_rad_s':self.max_velocity,'joint_acceleration_rad_s2':self.max_acceleration,
                'gripper_normalized_velocity_s':self.max_gripper_velocity,'control_frequency_hz':1/self.dt}}
