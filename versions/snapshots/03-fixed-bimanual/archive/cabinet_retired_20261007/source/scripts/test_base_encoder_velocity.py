#!/usr/bin/env python3
"""CPU checks for the interval encoder velocity convention; no simulator is imported."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.base_encoder_velocity import interval_base_velocity, replace_base_velocity
from harness.docking_controller import Odometry, wrap


class IntervalVelocityTests(unittest.TestCase):
    def test_stationary(self):
        q=np.array([10.,20.,0.,0.,0.,1.2])
        np.testing.assert_array_equal(interval_base_velocity(q,q,1/30),np.zeros(3))

    def test_translation_in_body_frame(self):
        a=np.array([10.,20.,0.,0.,0.,np.pi/2]);b=a.copy();b[1]+=.01
        np.testing.assert_allclose(interval_base_velocity(a,b,1/30),[.3,0,0],atol=1e-12)

    def test_encoder_position_offset_invariant(self):
        a=np.zeros(6);b=np.array([.01,-.02,0,0,0,.03]);offset=np.array([100,-300,7,0,0,0])
        np.testing.assert_allclose(interval_base_velocity(a,b,1/30),interval_base_velocity(a+offset,b+offset,1/30),atol=1e-10)

    def test_yaw_wrap(self):
        a=np.zeros(6);a[5]=np.pi-.01;b=a.copy();b[5]=-np.pi+.01
        self.assertAlmostEqual(interval_base_velocity(a,b,1/30)[2],.6)

    def test_encoder_path_reconstruction(self):
        a=np.array([4.,-3.,0,0,0,.7]);m=Odometry();initial=a.copy()
        R=np.array([[np.cos(a[5]),np.sin(a[5])],[-np.sin(a[5]),np.cos(a[5])]])
        for _ in range(100):
            b=a+np.array([.01,.006,0,0,0,.009])
            m.update(interval_base_velocity(a,b,1/30));a=b
        expected=np.r_[R@(a[:2]-initial[:2]),wrap(a[5]-initial[5])]
        np.testing.assert_allclose(m.pose,expected,atol=1e-10)

    def test_recorded_probe_not_a_fitted_gain(self):
        path=ROOT/'outputs/validation/task090_base_velocity_probe_20261001/run_v1/base_probe.json'
        rows=json.loads(path.read_text())['rows']; m=Odometry()
        for a,b in zip(rows,rows[1:]):
            dt=b['sim_time']-a['sim_time']
            m.update(interval_base_velocity(a['base_qpos_6dof'],b['base_qpos_6dof'],dt),dt)
        # Audit pose is used solely as an independent expected value, never passed to the helper.
        def yaw(p):
            x,y,z,w=p[3:7];return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        expected=wrap(yaw(rows[-1]['robot_pose_audit_only'])-yaw(rows[0]['robot_pose_audit_only']))
        self.assertLess(abs(wrap(m.pose[2]-expected)),1e-5)
        raw=sum(x['proprio_base_qvel'][2]*(b['sim_time']-a['sim_time']) for x,a,b in zip(rows[1:],rows,rows[1:]))
        self.assertGreater(abs(raw-expected),.1)

    def test_patch_copies_only_base_velocity(self):
        value=np.arange(61,dtype=np.float32);obs={'robot_r1':{'proprio':value,'camera':'unchanged'}}
        out=replace_base_velocity(obs,'robot_r1',[.1,.2,.3])
        np.testing.assert_array_equal(value,np.arange(61,dtype=np.float32))
        np.testing.assert_array_equal(out['robot_r1']['proprio'][3:],value[3:])
        self.assertEqual(out['robot_r1']['camera'],'unchanged')
        flat=replace_base_velocity({'robot_r1::proprio':value},'robot_r1',[0,0,0])
        np.testing.assert_array_equal(flat['robot_r1::proprio'][:3],[0,0,0])

    def test_invalid_contract(self):
        for dt in (0,-1,np.nan):
            with self.assertRaises(ValueError):interval_base_velocity(np.zeros(6),np.zeros(6),dt)
        with self.assertRaises(ValueError):interval_base_velocity(np.zeros(3),np.zeros(3),1/30)
        with self.assertRaises(ValueError):replace_base_velocity({'robot_r1::proprio':np.zeros(60)},'robot_r1',[0,0,0])


if __name__=='__main__':unittest.main(verbosity=2)
