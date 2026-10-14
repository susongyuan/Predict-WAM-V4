import tempfile,unittest,time
from pathlib import Path
from types import SimpleNamespace
from harness.robotwin_integrated_system import IntegratedSystem,IntegrationBlocked
from harness.robotwin_policy_worker import envelope
import numpy as np
class Visual:
    def __init__(self):self.state='unknown';self.desc='playing card box';self.ground='execute'
    def request(self,msg,kind):
        results={k:{'outcome':'ok','parsed':{'state':self.state,'object_description':self.desc}} for k in ('drawer_open','object_between_gripper_fingers','object_visibly_inside_open_drawer')}
        if msg['step']==0:results['ground']={'parsed':{'decision':self.ground,'capability':'put_object_cabinet'}}
        return envelope('visual',msg['episode'],msg['step'],replies=results)
class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.visual=Visual()
        self.obs={'rgb':{k:np.zeros((240,320,3),np.uint8) for k in ('head_camera','left_camera','right_camera')},'commanded_joint_gripper_state':np.zeros(14)}
        self.s=IntegratedSystem(None,None,visual=self.visual,robot=None,refresh=None,out=self.tmp.name,episode='ep')
    def tearDown(self):self.s.close();self.tmp.cleanup()
    def test_unknown_and_parse_failure_never_trigger_or_write(self):
        for i in range(4):self.assertFalse(self.s.inspect('ep',i*64,self.obs))
        self.assertEqual(self.s.memory.connection.execute('select count(*) from evidence').fetchone()[0],0)
        self.assertIsNone(self.s.report['system_verified_success'])
    def test_loss_requires_positive_subject_and_two_matching_negatives(self):
        self.visual.state='false';self.assertFalse(self.s.inspect('ep',0,self.obs))
        self.visual.state='true';self.assertFalse(self.s.inspect('ep',64,self.obs))
        self.visual.state='false';self.assertFalse(self.s.inspect('ep',128,self.obs));self.assertTrue(self.s.inspect('ep',192,self.obs))
    def test_unknown_or_other_subject_breaks_sequence(self):
        self.visual.state='true';self.s.inspect('ep',0,self.obs)
        self.visual.state='false';self.s.inspect('ep',64,self.obs)
        self.visual.desc='yellow background box';self.assertFalse(self.s.inspect('ep',128,self.obs));self.assertEqual(self.s.loss_votes,0)
    def test_grounding_cannot_be_forced(self):
        self.visual.ground='unsupported'
        with self.assertRaises(IntegrationBlocked):self.s.inspect('ep',0,self.obs)
    def test_gt_input_rejected_before_visual(self):
        self.obs['object_pose_world']=[0,0,0]
        with self.assertRaises(ValueError):self.s.inspect('ep',0,self.obs)
if __name__=='__main__':unittest.main()
