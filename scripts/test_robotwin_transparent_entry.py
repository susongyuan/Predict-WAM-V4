import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts import eval_robotwin_wam as ev
from scripts.test_robotwin_eval_bridge import InjectedEnv,InjectedPolicy,InjectedVideo
from harness.robotwin_policy_worker import envelope
class Policy(InjectedPolicy):
    def reset(self,episode,seed):self.episode=episode;return envelope('reset_ok',episode,0)
    def action(self,episode,step,obs):return envelope('action',episode,step,raw_action=[.1]*14)
class TransparentEntry(unittest.TestCase):
    def run_mode(self,mode):
        with tempfile.TemporaryDirectory() as d:
            args=ev.parse_args(['--run-dir',d,'--checkpoint','/unused','--training-run-dir','/unused','--gpu-lock-owner','test','--seeds','10000','--max-steps','3','--settle-steps','0','--system-mode',mode])
            e=InjectedEnv();p=Policy();r=ev.run_evaluation(args,{'task_kwargs':{}},Path(d),client_factory=lambda*a:p,env_factory=lambda:e,video_factory=lambda*a:(InjectedVideo(),None),owner_check=lambda*a:None,guard_factory=lambda:SimpleNamespace(check=lambda:{},peaks={}))
            events=Path(d)/'episode_000_seed10000/system_events.jsonl'
            return r,e.actions,events.read_text() if events.exists() else ''
    def test_native_actions_identical_to_pure(self):
        a,aa,_=self.run_mode('pure');b,bb,events=self.run_mode('transparent')
        self.assertEqual(a['status'],'completed');self.assertEqual(b['status'],'completed')
        np.testing.assert_array_equal(aa,bb);self.assertEqual(len(bb),3)
        for l in events.splitlines():
            e=json.loads(l);self.assertTrue(e['driver_ack']);self.assertFalse(e['full_system']);self.assertIsNone(e['visible_outcome'])
if __name__=='__main__':unittest.main()
