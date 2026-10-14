"""CPU checks: native RoboTwin1 timing is explicit, not silently labelled 15 Hz."""
import tempfile,unittest
from pathlib import Path
import h5py
from scripts.test_robotwin_dataset import episode_file,SHAPE
from harness.robotwin_dataset import RoboTwinEpisode
class NativeOneTiming(unittest.TestCase):
    def make(self,root,frequency=250/15,fmt='RoboTwin1_native_local_expert_RGB'):
        p=episode_file(Path(root))
        with h5py.File(p,'r+') as f:
            del f['additional_info/frequency']
            f['additional_info/frequency']=float(frequency)
            f.attrs['source_format']=fmt
        return p
    def test_native_one_accepts_own_nominal_frequency(self):
        with tempfile.TemporaryDirectory() as root:
            d=RoboTwinEpisode(self.make(root),image_shape=SHAPE)
            self.assertAlmostEqual(d.frequency,250/15)
    def test_non_native_format_does_not_bypass_old_frequency(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):RoboTwinEpisode(self.make(root,fmt='unknown'),image_shape=SHAPE)
    def test_explicit_requested_frequency_still_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):RoboTwinEpisode(self.make(root),image_shape=SHAPE,frequency=15)
    def test_native_format_not_arbitrary_frequency(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):RoboTwinEpisode(self.make(root,frequency=30),image_shape=SHAPE)
if __name__=='__main__':unittest.main()
