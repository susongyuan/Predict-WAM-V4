"""Synthetic math/unit checks; no physical or learned-capability claims."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.task0_prefix4_diagnostic import angle, errors, gripper
from harness.r1pro_kinematics import R1ProKinematics, axis_angle

class PrefixTests(unittest.TestCase):
    def test_identity(self):
        self.assertAlmostEqual(angle(np.eye(3),np.eye(3)),0)
    def test_pi(self):
        self.assertAlmostEqual(angle(np.eye(3),axis_angle([1,0,0],np.pi)),np.pi)
    def test_rotation_invariance(self):
        r=axis_angle([1,2,3],.8); a=axis_angle([1,0,0],.4); b=axis_angle([0,1,0],.2)
        self.assertAlmostEqual(angle(a,b),angle(r@a,r@b))
    def test_reject_reflection(self):
        with self.assertRaises(ValueError): angle(np.diag([-1,1,1]),np.eye(3))
    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError): angle(np.full((3,3),np.nan),np.eye(3))
    def test_fk_same(self):
        a=np.zeros((32,23)); e=errors(R1ProKinematics(),a,a)
        self.assertEqual(len(e),32)
        self.assertLess(max(max(row.values()) for row in e),1e-6)
    def test_prefix_event_exclusion_and_units(self):
        a=np.zeros((32,23)); a[:,14]=-1; a[4:,14]=1; a[:,22]=2
        self.assertEqual(gripper(a,4)['left']['internal_sign_transitions'],[])
        self.assertEqual(gripper(a,8)['left']['internal_sign_transitions'],[{'index':4,'from':-1,'to':1}])
        self.assertEqual(gripper(a,8)['left']['finger_target_m'][:5],[0,0,0,0,.05])
        self.assertEqual(gripper(a,4)['right']['commands'],[1]*4)
    def test_relative_pose_detects_single_arm_change(self):
        a=np.zeros((1,23)); b=a.copy(); b[0,15]=.3
        e=errors(R1ProKinematics(),a,b)[0]
        self.assertGreater(e['relative_rad'],.1)
        self.assertLess(e['left_m'],1e-9)

if __name__=='__main__': unittest.main(verbosity=2)
