"""CPU schedule sealing tests; fake HDF content tests metadata only, not rendering."""
import unittest,tempfile,hashlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,yaml
from scripts import run_robotwin1_apple_schedule as s
class Episode:
    def __init__(self,p):self.state=np.zeros((3,14))
    def __len__(self):return 3
class Sealing(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.r=Path(self.temp.name)
        self.paths={'ROOT':self.r,'DATA':self.r/'data/new','COLLECT':self.r/'collection','CONFIG':self.r/'configs/new.yaml'}
        self.ctx=patch.multiple(s,**self.paths);self.ctx.start()
        self.paths['COLLECT'].mkdir();self.paths['CONFIG'].parent.mkdir()
        cfg=self.r/'configs/fixed_bimanual/robotwin_cabinet_500_wam.yaml';cfg.parent.mkdir()
        cfg.write_text(yaml.safe_dump({'dataset':{},'training':{'max_steps':50000,'schedule_steps':50000}}))
        self.collection={'status':'complete','accepted':[],'attempts':[]}
        for i in range(100):
            p=self.paths['COLLECT']/f'raw_{i}.hdf5';p.write_bytes(f'fake-hdf-{i}'.encode())
            self.collection['accepted'].append({'hdf5':str(p),'episode_id':i,'sha256':s.sha(p),'frames':3,'seed':i,'first_frame_visibility':{'head_camera':{'target_pixels':64}}})
            self.collection['attempts'].append({'seed':i})
        (self.paths['COLLECT']/'status.json').write_text(json.dumps(self.collection))
    def tearDown(self):self.ctx.stop();self.temp.cleanup()
    def test_seals_count_seed_exclusion_and_exact_80_10_10(self):
        with patch('harness.robotwin_dataset.RoboTwinEpisode',Episode):cfg=s.finalize_data(self.collection)
        self.assertEqual(cfg['dataset']['backend'],'robotwin1')
        self.assertEqual(cfg['training']['max_steps'],50000)
        m=json.loads((self.paths['DATA']/'split_manifest.json').read_text())
        self.assertEqual(m['counts'],{'train':80,'val':10,'heldout':10})
        self.assertEqual(len((self.paths['COLLECT']/'seed.txt').read_text().splitlines()),100)
        self.assertEqual(s.sha(self.paths['DATA']/'split_manifest.json'),cfg['dataset']['split_sha256'])
    def test_partial_collection_rejected_before_write(self):
        self.collection['status']='running'
        with self.assertRaises(AssertionError):s.finalize_data(self.collection)
        self.assertFalse(self.paths['DATA'].exists())
    def test_sha_mismatch_fails_closed(self):
        self.collection['accepted'][0]['sha256']='0'*64
        with self.assertRaises(AssertionError):s.finalize_data(self.collection)
        self.assertFalse(self.paths['CONFIG'].exists())
    def test_visibility_failure_fails_closed(self):
        self.collection['accepted'][0]['first_frame_visibility']['head_camera']['target_pixels']=0
        with self.assertRaises(AssertionError):s.finalize_data(self.collection)
        self.assertFalse(self.paths['CONFIG'].exists())
if __name__=='__main__':unittest.main()
