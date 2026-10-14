import unittest
from types import SimpleNamespace
import numpy as np
from harness.robotwin_policy_worker import ChunkPolicy,InfrastructureError
from harness.robotwin_robot_feedback import hold_measured
class Joint:
    def __init__(self):self.target=.5;self.velocity=2
    def get_dof(self):return 1
    def get_name(self):return 'j'
    def get_drive_target(self):return [self.target]
    def get_limits(self):return [[-2,2]]
    def set_drive_target(self,x):self.target=x
    def set_drive_velocity_target(self,x):self.velocity=x
class RecoveryTests(unittest.TestCase):
    def test_queue_clear_preserves_context_and_step(self):
        calls=[];ctx=SimpleNamespace(reset=lambda:None,push=lambda*a:calls.append(a),inputs=lambda:None)
        p=ChunkPolicy(ctx,lambda x:(None,None),lambda*a:np.zeros((32,14)));p.reset('e',1)
        p.observe('e',0,{'rgb':{},'commanded_joint_gripper_state':np.zeros(14)})
        r=p.clear_actions('e',1,'visible_motion_stall');self.assertEqual(r['discarded_actions'],7);self.assertTrue(r['history_preserved']);self.assertEqual(len(calls),1)
        self.assertTrue(p.observe('e',1,{'rgb':{},'commanded_joint_gripper_state':np.zeros(14)})['replanned']);self.assertEqual(len(calls),2)
    def test_stale_clear_rejected_without_losing_queue(self):
        p=ChunkPolicy(SimpleNamespace(reset=lambda:None),None,None);p.reset('e',1);p.queue.append(1)
        with self.assertRaises(InfrastructureError):p.clear_actions('e',5,'x')
        self.assertEqual(len(p.queue),1)
    def robot(self):
        r=SimpleNamespace()
        for side in ('left','right'):
            j=[Joint() for _ in range(6)];setattr(r,side+'_arm_joints',j)
            setattr(r,side+'_entity',SimpleNamespace(get_active_joints=lambda j=j:j,get_qpos=lambda:np.arange(6)*.1))
        return r
    def test_hold_current_not_zero_and_no_gripper(self):
        r=self.robot();v=hold_measured(r);self.assertFalse(v['gripper_commands_changed']);self.assertEqual(v['physics_steps'],0)
        for s in ('left','right'):
            np.testing.assert_allclose([j.target for j in getattr(r,s+'_arm_joints')],np.arange(6)*.1)
            self.assertEqual([j.velocity for j in getattr(r,s+'_arm_joints')],[0]*6)
    def test_all_validated_before_mutation(self):
        r=self.robot();r.right_entity.get_qpos=lambda:np.full(6,np.nan)
        with self.assertRaises(ValueError):hold_measured(r)
        self.assertEqual([j.target for j in r.left_arm_joints],[.5]*6)
if __name__=='__main__':unittest.main()
