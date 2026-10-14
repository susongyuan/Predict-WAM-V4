import unittest
from scripts.run_robotwin_checkpoint_queue import evaluation_wall_budgets,command
class Budget(unittest.TestCase):
 def test_legacy_unchanged(self):self.assertEqual(evaluation_wall_budgets({'max_steps':700}),{'episode_wall_seconds':1200,'total_wall_seconds':1500,'external_wall_seconds':1560})
 def test_native_1700_actions(self):self.assertEqual(evaluation_wall_budgets({'max_steps':1700}),{'episode_wall_seconds':5100,'total_wall_seconds':5400,'external_wall_seconds':5460})
 def test_bad_bounds(self):
  for job in ({'max_steps':2001},{'max_steps':True},{'max_steps':1700,'episode_wall_seconds':-2},{'total_wall_seconds':100}):
   with self.subTest(job=job),self.assertRaises(ValueError):evaluation_wall_budgets(job)
 def test_actual_command_uses_budget(self):
  cmd=command({'unit':'test-dustbin','run_dir':'/new','training_run_dir':'/train','checkpoint':'/train/c.pt','seed':920001,'max_steps':1700})
  self.assertEqual(cmd[cmd.index('--episode-wall-seconds')+1],'5100');self.assertIn('--property=RuntimeMaxSec=5460',cmd)
if __name__=='__main__':unittest.main()
