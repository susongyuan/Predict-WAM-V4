import unittest,time,numpy as np
from harness.robotwin_target_binding import parse_box,bind,unproject_depth
class Tests(unittest.TestCase):
 def test_resolution_and_coordinate_units(self):
  candidate=parse_box('[{"bbox_2d":[0,0,448,336],"label":"visible object"}]',[[1,24,32]])
  b=bind('ep','obs',time.monotonic(),np.zeros((240,320,3),np.uint8),candidate)
  self.assertEqual(b.bbox_native,(0,0,320,240));self.assertFalse(b.full_body_verified)
  with self.assertRaises(ValueError):parse_box('[{"bbox_2d":[0,0,448,336],"label":"visible object"}]',[[1,18,22]])
 def test_ambiguous_or_invalid_never_becomes_a_target(self):
  self.assertIsNone(parse_box('[]',[[1,24,32]]))
  for text in ('[{"bbox_2d":[0,0,449,10],"label":"x"}]','[{"bbox_2d":[0,0,10,10],"label":"x"},{"bbox_2d":[0,0,10,10],"label":"y"}]'):
   with self.assertRaises(ValueError):parse_box(text,[[1,24,32]])
 def test_metric_depth_and_cv_gl_axis_conversion(self):
  points,valid=unproject_depth(np.array([[1000.,2000.]]),np.eye(3),np.eye(4));np.testing.assert_allclose(points[0,0],[0,0,-1]);np.testing.assert_allclose(points[0,1],[2,0,-2]);self.assertTrue(valid.all())
 def test_missing_depth_and_invalid_calibration_unknown(self):
  points,valid=unproject_depth(np.array([[0.,np.nan]]),np.eye(3),np.eye(4));self.assertFalse(valid.any());self.assertTrue(np.isnan(points).all())
  T=np.eye(4);T[0,0]=2
  with self.assertRaises(ValueError):unproject_depth(np.ones((1,1)),np.eye(3),T)
if __name__=='__main__':unittest.main()
