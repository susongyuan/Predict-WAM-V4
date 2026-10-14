"""Geometry counterexamples for the unadmitted v2 RGB-D tracker."""
import unittest
import numpy as np
from harness.task090_rigid_observation_v2 import fit_rigid, robust_rigid, height_change_bound, commanded_finger_sum


class RigidGeometryTests(unittest.TestCase):
    def setUp(self):
        self.points=np.random.default_rng(9).uniform([.2,-.2,.4],[.6,.2,.7],size=(30,3))

    def test_translation_is_recovered(self):
        target=self.points+[0,0,.2]
        r,t,e=fit_rigid(self.points,target)
        np.testing.assert_allclose(r,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(t,[0,0,.2],atol=1e-12)
        b=height_change_bound(self.points,target,r,[0,0,0],[0,0,0],1.,float(e.max()),.01)
        self.assertGreater(b['conditional_min_height_gain_m'],.18)

    def test_tipping_is_not_certified_as_a_lift(self):
        theta=np.deg2rad(20)
        r=np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
        source=np.random.default_rng(1).uniform([.35,-.1,.6],[.4,.1,.67],size=(20,3))
        target=source@r.T  # rotates around a floor point, which has not lifted
        b=height_change_bound(source,target,r,[0,0,0],[0,0,0],1.,0)
        self.assertGreater(b['anchor_height_gain_m'],.03)
        self.assertLess(b['conditional_min_height_gain_m'],0)

    def test_base_frame_change_cancels_against_observed_floor(self):
        theta=.2
        r=np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        t=np.array([.1,-.2,.3]);target=self.points@r.T+t
        n=r@np.array([0.,0.,1.]);d=-float(n@t)
        floor=[-n[0]/n[2],-n[1]/n[2],-d/n[2]]
        b=height_change_bound(self.points,target,r,[0,0,0],floor,1.,0)
        self.assertAlmostEqual(b['anchor_height_gain_m'],0,places=12)
        self.assertAlmostEqual(b['rotation_allowance_m'],0,places=12)
        self.assertLessEqual(b['conditional_min_height_gain_m'],0)

    def test_ransac_rejects_wrong_correspondences(self):
        target=self.points+[0,0,.2]
        target[-5:]+=[.5,-.4,.3]
        r,t,error,keep=robust_rigid(self.points,target)
        self.assertEqual(int(keep.sum()),25)
        np.testing.assert_allclose(t,[0,0,.2],atol=1e-10)
        self.assertLess(error,1e-10)

    def test_collinear_points_are_unknown(self):
        source=np.c_[np.arange(8),np.zeros(8),np.ones(8)]
        with self.assertRaises(ValueError):fit_rigid(source,source)
        with self.assertRaises(ValueError):robust_rigid(source,source)

    def test_partial_closure_is_not_obstruction(self):
        self.assertAlmostEqual(commanded_finger_sum(-.4),.03)
        self.assertAlmostEqual(.03-commanded_finger_sum(-.4),0.)
        self.assertAlmostEqual(commanded_finger_sum(1),.1)
        self.assertAlmostEqual(commanded_finger_sum(-1),0.)
        self.assertGreater(.046-commanded_finger_sum(-1),.004)

    def test_no_inputs_or_global_rng_mutated(self):
        source=self.points.copy();target=source+[0,0,.2]
        before=source.copy();rng=np.random.get_state()
        robust_rigid(source,target)
        np.testing.assert_array_equal(source,before)
        after=np.random.get_state()
        np.testing.assert_array_equal(rng[1],after[1]);self.assertEqual(rng[2:],after[2:])


if __name__=='__main__':unittest.main()
