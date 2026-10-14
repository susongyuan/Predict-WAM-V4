"""Robot-only encoder/hold ingress. Does not accept env, target actors or GT.
Holding means issuing measured joint targets with zero target velocity; this is
not certified collision safety or proof that a grasp remains stable. No stepping.
"""
import numpy as np

def gripper_feedback(robot):
    """Measured finger encoders, separate from commanded normalized aperture.

    RoboTwin get_normal_real_gripper_val reads drive targets despite its name.
    Here qpos/qvel are mapped by actual single-DOF joint identities. Values are
    not clipped, and neither obstruction nor an empty grasp is inferred.
    """
    result={}
    for side in ('left','right'):
        entity=getattr(robot,side+'_entity');active=list(entity.get_active_joints())
        fingers=list(getattr(robot,side+'_gripper'))
        if not fingers or len(set(map(id,active)))!=len(active) or any(j.get_dof()!=1 for j in active):
            raise ValueError('unverified gripper single-DOF joint mapping')
        q=np.asarray(entity.get_qpos());v=np.asarray(entity.get_qvel())
        scale=np.asarray(getattr(robot,side+'_gripper_scale'),dtype=float)
        if q.shape!=(len(active),) or v.shape!=q.shape or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError('invalid gripper encoder measurement')
        if scale.shape!=(2,) or not np.isfinite(scale).all() or scale[1]<=scale[0]:
            raise ValueError('invalid normalized gripper scale')
        records=[]
        for joint,multiplier,offset in fingers:
            if not np.isfinite([multiplier,offset]).all() or multiplier==0:raise ValueError('invalid finger mimic mapping')
            idx=active.index(joint);target=np.asarray(joint.get_drive_target());limits=np.asarray(joint.get_limits())
            if target.shape!=(1,) or limits.shape!=(1,2) or not np.isfinite(target).all() or not np.isfinite(limits).all():
                raise ValueError('invalid finger target/limits')
            base=(float(q[idx])-offset)/multiplier
            records.append({'name':joint.get_name(),'actual_qpos':float(q[idx]),'actual_qvel':float(v[idx]),
                'drive_target':float(target[0]),'actual_minus_drive_target':float(q[idx]-target[0]),
                'base_equivalent_qpos':base,'normalized_actual':float((base-scale[0])/(scale[1]-scale[0])),
                'within_joint_limits':bool(limits[0,0]<=q[idx]<=limits[0,1])})
        target=(records[0]['drive_target']-scale[0])/(scale[1]-scale[0])
        result[side]={'fingers':records,'normalized_actual':records[0]['normalized_actual'],
            'normalized_drive_target':float(target),'mimic_disagreement':float(np.ptp([r['normalized_actual'] for r in records])),
            'commanded_aperture':float(getattr(robot,side+'_gripper_val')),
            'normalization_scale':scale.tolist(),'grasp_verified':None,'units':'native joint units; normalized values are unclipped'}
    return result

def robot_feedback(robot):
    """Robot-only measurements suitable for observation/recovery diagnostics."""
    result={'arms':arm_feedback(robot),'grippers':gripper_feedback(robot),'tcp_world':{}}
    for side in ('left','right'):
        pose=np.asarray(getattr(robot,'get_'+side+'_tcp_pose')(),dtype=float)
        if pose.shape!=(7,) or not np.isfinite(pose).all():raise ValueError('invalid measured robot TCP')
        result['tcp_world'][side]=pose.tolist()
    return result

def arm_feedback(robot):
    result={}
    for side in ('left','right'):
        entity=getattr(robot,side+'_entity');active=list(entity.get_active_joints());joints=list(getattr(robot,side+'_arm_joints'))
        if len(joints)!=6 or len(set(map(id,active)))!=len(active) or any(j.get_dof()!=1 for j in active):
            raise ValueError('unverified native6 single-DOF joint mapping')
        measured=np.asarray(entity.get_qpos());indices=[active.index(j) for j in joints]
        if measured.shape!=(len(active),) or not np.isfinite(measured).all():raise ValueError('invalid encoder measurement')
        targets=[];bounds=[]
        for j in joints:
            t=np.asarray(j.get_drive_target());b=np.asarray(j.get_limits())
            if t.shape!=(1,) or b.shape!=(1,2) or not np.isfinite(t).all() or not np.isfinite(b).all() or b[0,0]>b[0,1]:raise ValueError('invalid drive target/limit')
            targets.append(float(t[0]));bounds.append(b[0].tolist())
        q=measured[indices]
        if any(v<lo or v>hi for v,(lo,hi) in zip(q,bounds)):raise ValueError('encoder outside limits; refuse implicit clipping')
        result[side]={'joint_names':[j.get_name() for j in joints],'actual_qpos':q.tolist(),'drive_targets':targets,'limits':bounds}
    return result

def hold_measured(robot):
    before=arm_feedback(robot)  # Both arms validated before any setter.
    for side in ('left','right'):
        for joint,position in zip(getattr(robot,side+'_arm_joints'),before[side]['actual_qpos']):
            joint.set_drive_target(position);joint.set_drive_velocity_target(0.0)
    after=arm_feedback(robot)
    for side in ('left','right'):
        if not np.allclose(after[side]['drive_targets'],before[side]['actual_qpos'],atol=1e-6,rtol=0):raise RuntimeError('hold command readback mismatch')
    return {'status':'hold_targets_applied_not_physical_safety_proof','gripper_commands_changed':False,'physics_steps':0,'feedback_before':before,'feedback_after':after}
