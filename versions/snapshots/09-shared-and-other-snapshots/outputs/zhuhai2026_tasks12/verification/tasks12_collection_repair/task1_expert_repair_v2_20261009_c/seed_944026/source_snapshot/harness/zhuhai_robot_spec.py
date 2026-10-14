"""Single source of truth for the new fixed dual-Panda (native16) experiment.

No changes to frozen Aloha/native14 checkpoints or RoboTwin baseline adapters.
Units are metres, kilograms, seconds and joint radians; gripper 1 means open.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import math

ROOT=Path(__file__).resolve().parents[1]
ROBOT_DOF=16
ARM_DOF=7
GRIPPER_INDICES=(7,15)
ARM_INDICES=tuple(range(7))+tuple(range(8,15))
ARM_SLICES={'left':slice(0,7),'right':slice(8,15)}
ASSET=ROOT/'data/isaac_zhuhai/franka_official_v1/franka_panda.usda'
ASSET_MANIFEST=ASSET.parent/'asset_manifest.json'
KINEMATIC_CHAIN=ASSET.parent/'kinematic_chain.json'
ROBOT_PROFILE='fixed_dual_franka_panda_native16_v1'
ACTION_CONTRACT='native16_joint_radians_gripper_fraction_1_open'
STATE_CONTRACT='actual_measured_native16_joint_radians_gripper_fraction_1_open'
JOINT_LIMITS=[[-2.8973,2.8973],[-1.7628,1.7628],[-2.8973,2.8973],
    [-3.0718,-.0698],[-2.8973,2.8973],[-.0175,3.7525],[-2.8973,2.8973]]
DUAL_JOINT_LIMITS=JOINT_LIMITS+JOINT_LIMITS
VELOCITY_LIMITS=[2.175]*4+[2.61]*3
EFFORT_LIMITS=[87.]*4+[12.]*3
GRIPPER_FINGER_TRAVEL=.04
BASE_POSITIONS={'left':(-.42,-.55,.74),'right':(.42,-.55,.74)}
BASE_YAW=math.pi/2
HOME_ARM=[0.,-.569,0.,-2.81,0.,3.037,.741]
HOME_ACTION=HOME_ARM+[1.]+HOME_ARM+[1.]
HOME_ACTION[0]=1.10
HOME_ACTION[8]=-1.10
# Expert's X is the approach axis, Y is the pinch axis; real Panda hand Z
# is the finger extension and Y is the physical closing direction.
TOOL_FROM_HAND=[[0.,0.,-1.,0.],[0.,1.,0.,0.],[1.,0.,0.,.1034],[0.,0.,0.,1.]]

def base_matrix(arm):
    import numpy as np
    c,s=np.cos(BASE_YAW),np.sin(BASE_YAW)
    mat=np.eye(4);mat[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    mat[:3,3]=BASE_POSITIONS[arm]
    return mat

def validate_action(action,joint_limits=None):
    import numpy as np
    value=np.asarray(action,dtype=float)
    if value.shape!=(ROBOT_DOF,) or not np.isfinite(value).all():
        raise ValueError('Expected one finite native16 Panda action')
    bounds=np.asarray(DUAL_JOINT_LIMITS if joint_limits is None else joint_limits)
    if bounds.shape!=(14,2):raise ValueError('Panda limits must describe 14 physical arm joints')
    arm=value[list(ARM_INDICES)]
    if np.any(arm<bounds[:,0]-1e-7) or np.any(arm>bounds[:,1]+1e-7):
        raise ValueError('Panda target violates standard joint travel')
    if np.any(value[list(GRIPPER_INDICES)]<0) or np.any(value[list(GRIPPER_INDICES)]>1):
        raise ValueError('Panda gripper target must be in [0,1]')
    return value.copy()

def robot_contract():
    asset_sha=hashlib.sha256(ASSET.read_bytes()).hexdigest() if ASSET.is_file() else None
    receipt=json.loads(ASSET_MANIFEST.read_text()) if ASSET_MANIFEST.is_file() else {}
    return {'profile':ROBOT_PROFILE,'name':'two fixed Franka Emika Panda arms with standard Panda hands',
        'action_dim':ROBOT_DOF,'state_dim':ROBOT_DOF,'arm_dof':ARM_DOF,'gripper_indices':list(GRIPPER_INDICES),
        'action_contract':ACTION_CONTRACT,'state_contract':STATE_CONTRACT,
        'asset':str(ASSET),'asset_sha256':asset_sha,'asset_manifest':str(ASSET_MANIFEST),
        'asset_manifest_sha256':hashlib.sha256(ASSET_MANIFEST.read_bytes()).hexdigest() if ASSET_MANIFEST.is_file() else None,
        'asset_source':receipt.get('source_base_url'),'variants':{'Colliders':'convex_hulls','Physics':'physx'},
        'base_positions_m':{k:list(v) for k,v in BASE_POSITIONS.items()},'base_yaw_rad':float(BASE_YAW),
        'joint_limits_rad':[list(v) for v in JOINT_LIMITS],'joint_velocity_limits_rad_s':list(VELOCITY_LIMITS),
        'joint_effort_limits_nm':list(EFFORT_LIMITS),'finger_travel_m':GRIPPER_FINGER_TRAVEL,
        'video_recording':False,'camera_names':['head_camera','left_camera','right_camera'],
        'scope':'standard single-arm chains mounted as fixed bimanual rig; not manufacturer certification of support structure'}
