#!/usr/bin/env python3
"""CPU contract/regression tests for the new physical dual-Panda branch only."""
from pathlib import Path
import hashlib,json,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from harness.zhuhai_robot_spec import (ROBOT_DOF,ARM_INDICES,GRIPPER_INDICES,HOME_ACTION,
    ASSET,ASSET_MANIFEST,KINEMATIC_CHAIN,JOINT_LIMITS,TOOL_FROM_HAND,validate_action,robot_contract)
from harness.isaac_zhuhai_expert import FrankaFK,minimum_jerk
from harness.isaac_zhuhai_task2 import box_corners
from harness.zhuhai_task2_rules import wholly_contained

class DualPandaTests(unittest.TestCase):
    def test_native16_and_old14_rejection(self):
        self.assertEqual(ROBOT_DOF,16);self.assertEqual(GRIPPER_INDICES,(7,15))
        self.assertEqual(validate_action(HOME_ACTION).shape,(16,))
        for bad in [HOME_ACTION[:14],np.full(16,np.nan)]:
            with self.assertRaises(ValueError):validate_action(bad)
        bad=np.array(HOME_ACTION);bad[3]=0
        with self.assertRaises(ValueError):validate_action(bad)
        self.assertIn('actual_measured',robot_contract()['state_contract'])

    def test_all_official_dependency_hashes(self):
        r=json.loads(ASSET_MANIFEST.read_text());self.assertEqual(len(r['files']),12)
        for f in r['files'].values():
            self.assertEqual(hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest(),f['sha256'])
        self.assertEqual(r['variant_selection']['Colliders'],'convex_hulls')

    def test_exact_asset_kinematic_and_inertial_contract(self):
        r=json.loads(KINEMATIC_CHAIN.read_text());self.assertEqual(r['asset_sha256'],hashlib.sha256(ASSET.read_bytes()).hexdigest())
        self.assertEqual(len(r['joints']),7);self.assertEqual(len(r['bodies']),11)
        np.testing.assert_allclose([j['limits_rad'] for j in r['joints']],JOINT_LIMITS)
        for b in r['bodies']:
            i=np.array(b['principal_inertia_kg_m2']);self.assertGreater(b['mass_kg'],0)
            self.assertTrue((i>0).all());self.assertLessEqual(i.max(),i.sum()-i.max()+1e-8)
        r=np.array(TOOL_FROM_HAND)[:3,:3];np.testing.assert_allclose(r.T@r,np.eye(3));self.assertAlmostEqual(np.linalg.det(r),1)

    def test_fk_analytic_jacobian_and_ik_roundtrip(self):
        f=FrankaFK()
        for arm,q in [('left',np.array(HOME_ACTION[:7])),('right',np.array(HOME_ACTION[8:15]))]:
            p,j=f.fk(arm,q,return_jacobian=True);num=np.zeros_like(j);h=1e-6
            for k in range(7):
                q2=q.copy();q2[k]+=h;new=f.fk(arm,q2)
                num[:3,k]=(new[:3,3]-p[:3,3])/h
                num[3:,k]=Rotation.from_matrix(new[:3,:3]@p[:3,:3].T).as_rotvec()/h
            np.testing.assert_allclose(j,num,atol=1e-6)
            ik=f.ik(arm,p,q+.01);self.assertTrue(ik.success)
            self.assertLess(ik.position_error_m,.001)

    def test_minimum_jerk_all_14_arm_joint_limits(self):
        a=np.array(HOME_ACTION);b=a.copy();b[list(ARM_INDICES)]+=.02;b[list(GRIPPER_INDICES)]=0
        samples,d=minimum_jerk(a,b);seq=np.vstack([a,samples]);vel=np.diff(seq,axis=0)/.1
        self.assertLessEqual(np.max(np.abs(vel[:,list(ARM_INDICES)])),.75+1e-9)
        self.assertLessEqual(np.max(np.abs(vel[:,list(GRIPPER_INDICES)])),.7+1e-9)
        np.testing.assert_allclose(seq[-1],b)

    def test_world_hull_extrema_exact_but_rotated_local_box_not_exact(self):
        # Tetrahedron: one local AABB corner is not part of any real geometry.
        hull=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]])
        rot=Rotation.from_euler('z',45,degrees=True).as_matrix();world=hull@rot.T
        lo,hi=world.min(0),world.max(0)
        exact=box_corners((lo+hi)/2,(hi-lo)/2)
        self.assertTrue(wholly_contained(hull@rot.T,lo,hi,0))
        self.assertTrue(wholly_contained(exact,lo,hi,1e-15))
        false=box_corners([.5,.5,.5],[.5,.5,.5])@rot.T
        self.assertFalse(wholly_contained(false,lo,hi,1e-15))

    def test_actual_contact_monitor_distinguishes_mount_and_collision(self):
        from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
        env=IsaacZhuhaiEnv.__new__(IsaacZhuhaiEnv)
        env._contact_pairs={('/World/Robot/left/panda_link0','/World/FixedSupport/left_post'):{}}
        env.physics_time=.1;env._safety_seen=set();env.safety_events=[]
        env._audit_physics_contacts();self.assertTrue(env.physics_safety()['passed'])
        env._contact_pairs={('/World/Robot/left/panda_link4','/World/Robot/right/panda_link4'):{}}
        env._audit_physics_contacts();self.assertFalse(env.physics_safety()['passed'])
        self.assertEqual(env.safety_events[0]['reason'],'inter_arm_collision')

    def test_garbage_controlled_release_targets_use_actual_primitive_bounds(self):
        from harness.isaac_zhuhai_expert import garbage_place_poses,top_down_rotation
        from harness.isaac_zhuhai_task1 import scene_spec,primitive_world_bounds
        for random in (False,True):
            manifest=scene_spec(891001,random)
            for key,spec in manifest['objects'].items():
                body=np.eye(4);body[:3,3]=spec['initial_position_world_m']
                tool=np.eye(4);tool[:3,:3]=top_down_rotation([1,0,0]);tool[:3,3]=body[:3,3]+spec['grasp_point_local_m']
                above,place=garbage_place_poses(manifest,key,body,tool)
                for target,level in ((above,'above'),(place,'inside')):
                    moved=body.copy();moved[:3,3]+=target[:3,3]-tool[:3,3]
                    corners=primitive_world_bounds(spec['parts'],moved);dest=manifest['bins'][spec['destination']]
                    if level=='above':self.assertGreater(corners[:,2].min(),dest['mouth_height_m'])
                    else:self.assertTrue(wholly_contained(corners,*dest['destination_bounds_m']))

    def test_official_fcl_home_and_known_near_collision(self):
        from harness.isaac_zhuhai_collision import PandaSelfCollision
        f=FrankaFK();checker=PandaSelfCollision(f)
        self.assertTrue(checker.collision_free('left',HOME_ACTION[:7]))
        self.assertTrue(checker.collision_free('right',HOME_ACTION[8:15]))
        unsafe=[.74498454,-.9537331,-1.1042197,-3.0717899,-1.0619488,2.1551313,1.246735]
        self.assertFalse(checker.collision_free('left',unsafe))
        self.assertIn('panda_link5',checker.last_pair)

if __name__=='__main__':unittest.main()
