#!/usr/bin/env python3
"""CPU-only FK, IK and smooth-command verification; no fake demonstrations."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pxr import Usd
from harness.isaac_zhuhai_expert import URDFFK,minimum_jerk,grasp_pose_candidates,object_destination_pose
from harness.isaac_zhuhai_task2 import build
from harness.isaac_zhuhai_task1 import scene_spec


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fk=URDFFK()

    def test_analytic_position_jacobian(self):
        q=np.array([.1,.5,-.3,.2,.15,-.1])
        for arm in ['left','right']:
            matrix,jac=self.fk.fk(arm,q,return_jacobian=True)
            for index in range(6):
                perturbed=q.copy();perturbed[index]+=1e-6
                finite=(self.fk.fk(arm,perturbed)[:3,3]-matrix[:3,3])/1e-6
                self.assertTrue(np.allclose(finite,jac[:3,index],atol=1e-6))

    def test_ik_roundtrip_and_unreachable(self):
        for arm in ['left','right']:
            q=np.array([.1,.5,-.3,.2,.15,-.1]);target=self.fk.fk(arm,q)
            result=self.fk.ik(arm,target,seed=q+.03)
            self.assertTrue(result.success)
        target=np.eye(4);target[:3,3]=[10,10,10]
        self.assertFalse(self.fk.ik('left',target,max_iterations=30).success)

    def test_minimum_jerk_bounds_and_endpoints(self):
        start=np.zeros(14);target=np.r_[np.ones(6)*2,1,np.ones(6)*-2,1]
        samples,duration=minimum_jerk(start,target)
        action=np.vstack([start,samples]);velocity=np.diff(action,axis=0)/.1;acc=np.diff(velocity,axis=0)/.1
        ids=list(range(6))+list(range(7,13))
        self.assertLessEqual(np.max(np.abs(velocity[:,ids])),.75+1e-8)
        self.assertLessEqual(np.max(np.abs(acc[:,ids])),1.5+1e-8)
        self.assertLessEqual(np.max(np.abs(velocity[:,[6,13]])),.7+1e-8)
        self.assertTrue(np.allclose(samples[-1],target))

    def test_task2_source_and_random_waypoints(self):
        for randomized in [False,True]:
            for seed in [101001,101002,101003]:
                manifest=build(Usd.Stage.CreateInMemory(),seed=seed,randomized=randomized)
                for key,spec in manifest['objects'].items():
                    arm='left' if key in ['bowl','spoon'] else 'right';body=np.asarray(spec['initial_world_matrix'])
                    grasp=grasp_pose_candidates(key,spec,body,manifest['table_surface_world_m'])[0]
                    place=object_destination_pose(key,spec,manifest)@np.linalg.inv(body)@grasp
                    q=np.zeros(6)
                    for phase,pose,dz in [('pregrasp',grasp,.11),('grasp',grasp,0),('lift',grasp,.18),
                                          ('transit',place,.08),('place',place,0),('retreat',place,.08)]:
                        pose=pose.copy();pose[2,3]+=dz;result=self.fk.ik(arm,pose,seed=q)
                        self.assertTrue(result.success,(randomized,seed,key,phase,result.position_error_m));q=result.joints

    def test_task1_passive_lid_arc(self):
        manifest=scene_spec(seed=101001)
        for color,arm in [('dry','left'),('wet','right')]:
            q=np.zeros(6)
            for node,pose in enumerate(manifest['bins'][color]['expert_open_arc']['gripper_world_pose_nodes']):
                result=self.fk.ik(arm,pose,seed=q)
                self.assertTrue(result.success,(color,node,result.position_error_m));q=result.joints


if __name__=='__main__':unittest.main()
