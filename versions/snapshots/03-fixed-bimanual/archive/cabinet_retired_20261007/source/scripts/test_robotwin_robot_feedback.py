import unittest
from types import SimpleNamespace
import numpy as np
from harness.robotwin_robot_feedback import gripper_feedback

class Joint:
    def __init__(self,name,target):self.name=name;self.target=target
    def get_dof(self):return 1
    def get_name(self):return self.name
    def get_drive_target(self):return [self.target]
    def get_limits(self):return [[-.02,.06]]
class Entity:
    def __init__(self,joints,q):self.joints=joints;self.q=q
    def get_active_joints(self):return self.joints
    def get_qpos(self):return self.q
    def get_qvel(self):return np.zeros(len(self.q))
def robot():
    r=SimpleNamespace()
    for side in ('left','right'):
        base=Joint(side+'base',-.01);mimic=Joint(side+'mimic',-.01)
        # Active order differs from configured fingers to catch bad indexing.
        setattr(r,side+'_entity',Entity([mimic,base],[.012,.012]))
        setattr(r,side+'_gripper',[(base,1,0),(mimic,1,0)])
        setattr(r,side+'_gripper_scale',[-.01,.045]);setattr(r,side+'_gripper_val',0)
    return r
class Tests(unittest.TestCase):
    def test_actual_differs_from_closed_command(self):
        f=gripper_feedback(robot())
        self.assertAlmostEqual(f['left']['normalized_actual'],.4)
        self.assertEqual(f['left']['normalized_drive_target'],0)
        self.assertEqual(f['left']['mimic_disagreement'],0)
        self.assertIsNone(f['left']['grasp_verified'])
    def test_reordered_mimic_and_negative_multiplier(self):
        r=robot();r.left_gripper[1]=(r.left_gripper[1][0],-1,.002)
        r.left_entity.q=[-.01,.012]
        f=gripper_feedback(r);self.assertAlmostEqual(f['left']['mimic_disagreement'],0)
    def test_out_of_range_is_reported_unclipped(self):
        r=robot();r.left_entity.q=[.08,.08];f=gripper_feedback(r)
        self.assertGreater(f['left']['normalized_actual'],1)
        self.assertFalse(f['left']['fingers'][0]['within_joint_limits'])
    def test_bad_mapping_measurement_and_scale_rejected(self):
        for field,value in (('q',[np.nan,0]),('scale',[1,1]),('mimic',0)):
            r=robot()
            if field=='q':r.left_entity.q=value
            elif field=='scale':r.left_gripper_scale=value
            else:r.left_gripper[0]=(r.left_gripper[0][0],value,0)
            with self.assertRaises(ValueError):gripper_feedback(r)
if __name__=='__main__':unittest.main()
