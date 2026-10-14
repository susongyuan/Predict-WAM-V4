import tempfile,unittest,json
from pathlib import Path
import numpy as np
from harness.robotwin_onboard_recorder import OnboardRecorder
from harness.robotwin_qpos_adapter import RGB_VIEWS

class RecorderTests(unittest.TestCase):
    def observation(self):return {'rgb':{v:np.arange(18,dtype=np.uint8).reshape(2,3,3) for v in RGB_VIEWS},'commanded_joint_gripper_state':np.arange(14,dtype=np.float32)}
    def test_lossless_exact_inputs_and_final_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            r=OnboardRecorder(d,'e',1);o=self.observation();r.append(o,0);r.append(o,1,2);r.close(True)
            with np.load(Path(d)/'onboard/observation_000000.npz') as a:
                self.assertEqual(set(a.files),set(RGB_VIEWS)|{'commanded_joint_gripper_state'})
                for v in RGB_VIEWS:np.testing.assert_array_equal(a[v],o['rgb'][v])
                np.testing.assert_array_equal(a['commanded_joint_gripper_state'],o['commanded_joint_gripper_state'])
            m=json.loads((Path(d)/'onboard/manifest.json').read_text());self.assertTrue(m['complete']);self.assertEqual(m['frames'],2)
    def test_gt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=OnboardRecorder(d,'e',1);o=self.observation();o['ground_truth']={}
            with self.assertRaises(ValueError):r.append(o,0)
    def test_no_directory_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            OnboardRecorder(d,'e',1)
            with self.assertRaises(FileExistsError):OnboardRecorder(d,'e',1)
    def test_bad_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=OnboardRecorder(d,'e',1);o=self.observation();o['rgb'][RGB_VIEWS[0]]=np.zeros((2,3,3),float)
            with self.assertRaises(ValueError):r.append(o,0)
if __name__=='__main__':unittest.main()
