"""CPU-only task admission/memory isolation for dustbin measured middle25k."""
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import tempfile,unittest,time,json
from scripts.eval_robotwin_wam import validate_system_task
from harness.robotwin_middle_system import MiddleSystem
from harness.robotwin_policy_worker import InfrastructureError
from scripts.test_robotwin_middle_system import Client,feedback,observation
class Admission(unittest.TestCase):
 def args(self,**kw):return SimpleNamespace(**dict({'system_mode':'middle','execution_mode':'native8','vtt_adapter':None,'vtt_bank':None,'target_head':None},**kw))
 def test_dustbin_middle_is_explicitly_admitted(self):validate_system_task(self.args(),'put_bottles_dustbin')
 def test_disallow_unmatched_vtt_and_feedback_modes(self):
  for k,v in [('vtt_adapter','old_cabinet.pt'),('vtt_bank','bank'),('target_head','head'),('execution_mode','contact_refresh'),('system_mode','integrated')]:
   with self.subTest(key=k),self.assertRaises(InfrastructureError):validate_system_task(self.args(**{k:v}),'put_bottles_dustbin')
 def test_other_tasks_remain_rejected(self):
  with self.assertRaises(InfrastructureError):validate_system_task(self.args(),'open_laptop')
 def test_original_pure_and_cabinet_admission_remain(self):
  validate_system_task(self.args(system_mode='pure'),'open_laptop');validate_system_task(self.args(),'put_object_cabinet')
 def test_task_checkpoint_bound_and_measured_retry_works(self):
  with tempfile.TemporaryDirectory() as td:
   c=Client();s=MiddleSystem(c,lambda a:True,robot=lambda:object(),refresh=lambda:(observation(),time.monotonic()),out=td,episode='dustbin',with_vtt=False,task='put_bottles_dustbin',checkpoint_sha256='a'*64)
   s.reset('dustbin',920010)
   try:
    self.assertEqual(s.report['memory_scope']['task'],'put_bottles_dustbin');self.assertEqual(s.report['checkpoint_sha256'],'a'*64)
    self.assertNotIn('cabinet',str(s.scope))
    for step in range(202):
     with patch('harness.robotwin_middle_system.robot_feedback',return_value=feedback()),patch('harness.robotwin_middle_system.hold_measured',return_value={'test_only_hold':True}):
      s.prepare('dustbin',step,str(step),time.monotonic(),observation());s.execute('dustbin',step,str(step))
    self.assertGreater(s.report['recovery_attempts'],0);self.assertLessEqual(s.report['recovery_attempts'],2)
    self.assertEqual(c.selections,0);self.assertEqual(s.report['task_success_memory_records'],0);self.assertFalse(s.report['certified_full_middle_system']);self.assertGreater(s.report['controller_memory_records'],0)
   finally:s.close()
 def test_new_task_requires_checkpoint_scope(self):
  with tempfile.TemporaryDirectory() as td,self.assertRaises(ValueError):MiddleSystem(Client(),lambda a:True,robot=lambda:object(),refresh=lambda:None,out=td,episode='x',with_vtt=False,task='put_bottles_dustbin')
if __name__=='__main__':unittest.main()
