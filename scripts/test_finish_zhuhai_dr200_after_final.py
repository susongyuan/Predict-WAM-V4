import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import finish_zhuhai_dr200_after_final as f
class FinalQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name);self.origin=root/'origin';self.final=root/'final';(self.origin/'train').mkdir(parents=True);self.final.mkdir()
        for key,value in [('ORIGIN',self.origin),('FINAL',self.final)]:p=patch.object(f,key,value);p.start();self.addCleanup(p.stop)
        self.put(self.origin/'train/status.json',dict(status='completed',step=30000,target_steps=30000))
        self.put(self.origin/'status.json',dict(status='failed',active_stage='old_control_error'))
        self.put(self.origin/'train/checkpoint_step_030000.pt.sha256.json',dict(sha256='a'*64))
    def put(self,path,value):path.write_text(json.dumps(value))
    def test_no_final_yet_waits(self):self.assertFalse(f.completed_training_and_final(self.origin))
    def test_partial_five_waits(self):
        self.put(self.final/'summary.json',dict(status='evaluating'));self.assertFalse(f.completed_training_and_final(self.origin))
    def test_final_infra_fails_closed(self):
        self.put(self.final/'summary.json',dict(status='halted_infrastructure_error'))
        with self.assertRaises(RuntimeError):f.completed_training_and_final(self.origin)
    def test_actual_training_must_finish_not_old_controller(self):
        self.put(self.origin/'train/status.json',dict(status='training',step=29999,target_steps=30000))
        self.assertFalse(f.completed_training_and_final(self.origin))
    def test_verified_zero_successes_does_not_block_mandatory_collection(self):
        self.put(self.final/'summary.json',dict(status='completed'))
        with patch.object(f.evaluation,'verify_summary',return_value={'planned_episodes':5,'full_successes':0})as verify:
            self.assertTrue(f.completed_training_and_final(self.origin));verify.assert_called_once()
    def test_different_origin_rejected(self):
        with self.assertRaises(ValueError):f.completed_training_and_final(self.origin.parent/'other')
if __name__=='__main__':unittest.main(verbosity=2)
