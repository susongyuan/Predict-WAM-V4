import unittest,numpy as np
from harness.robotwin_observation_schedule import execution_prefix
from harness.robotwin_policy_worker import ChunkPolicy
class Context:
 def reset(self):self.n=0
 def push(self,*args):self.n+=1
 def inputs(self):return np.zeros((7,14)),np.zeros((3,4,2048))
class Tests(unittest.TestCase):
 def setUp(self):self.chunk=np.zeros((32,14));self.chunk[:,[6,13]]=1;self.state=np.zeros(14);self.state[[6,13]]=1
 def test_default_matches_old_queue(self):
  self.chunk[3,6]=.5;self.assertEqual(execution_prefix(self.chunk,self.state,'native8')[0],8)
 def test_refresh_before_close_and_through_transition(self):
  self.chunk[3,6]=.5;self.assertEqual(execution_prefix(self.chunk,self.state,'contact_refresh')[0],1)
  self.chunk[:,6]=1;self.state[13]=.5;self.assertEqual(execution_prefix(self.chunk,self.state,'contact_refresh')[0],1)
 def test_release_and_small_open_noise(self):
  self.state[13]=0;self.chunk[:,13]=0;self.chunk[3,13]=.3;self.assertEqual(execution_prefix(self.chunk,self.state,'contact_refresh')[0],1)
  self.state[13]=1;self.chunk[:,13]=.99;self.assertEqual(execution_prefix(self.chunk,self.state,'contact_refresh')[0],8)
 def test_every_observation_keeps_history_and_resampling_changes(self):
  self.chunk[3,6]=.5
  p=ChunkPolicy(Context(),lambda _: (None,None),lambda *args:self.chunk,'contact_refresh');p.reset('e',1)
  o={'rgb':{},'commanded_joint_gripper_state':self.state}
  for i in range(3):
   r=p.observe('e',i,o);self.assertTrue(r['replanned']);self.assertEqual(r['history_steps'],i+1);self.assertEqual(r['queue_remaining'],0)
  self.assertEqual(p.context.n,3)
 def test_invalid_inputs_rejected(self):
  self.chunk[0,0]=np.nan
  with self.assertRaises(ValueError):execution_prefix(self.chunk,self.state,'contact_refresh')
if __name__=='__main__':unittest.main()
