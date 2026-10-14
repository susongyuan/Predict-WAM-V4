"""CPU checks for expert masking, split integrity and continuation schedule."""
import copy
import unittest
import numpy as np
from harness import zhuhai_dagger_training as d


class ContinuationTests(unittest.TestCase):
    def data(self):
        base={'splits':{'zhuhai_task2':{'train':[dict(family_id='train')],
            'val':[dict(family_id='val')],'heldout':[dict(family_id='heldout')]}}}
        rows=[dict(path=f'/tmp/expert-{i}.hdf5',sha256=f'{i:064x}',seed=1010000+i,
            split='train',domain='dagger',family_id='train',takeover_step=300,policy_steps=300,expert_steps=100) for i in range(50)]
        manifest=dict(status='complete',success_count=50,record_video=False,heldout_used=False,train_rows=rows)
        return base,manifest

    def test_complete_quota_and_train_only(self):
        base,manifest=self.data();self.assertEqual(len(d.validate_rows(base,manifest)),50)
        for change in (dict(success_count=49),dict(status='collecting'),dict(heldout_used=True)):
            with self.assertRaises(ValueError):d.validate_rows(base,dict(manifest,**change))
        for family in ('val','heldout','unknown'):
            bad=copy.deepcopy(manifest);bad['train_rows'][0]['family_id']=family
            with self.assertRaises(ValueError):d.validate_rows(base,bad)

    def test_duplicate_or_misaligned_control_rejected(self):
        base,manifest=self.data()
        for field in ('path','sha256','seed'):
            bad=copy.deepcopy(manifest);bad['train_rows'][1][field]=bad['train_rows'][0][field]
            with self.assertRaises(ValueError):d.validate_rows(base,bad)
        bad=copy.deepcopy(manifest);bad['train_rows'][0]['takeover_step']=299
        with self.assertRaises(ValueError):d.validate_rows(base,bad)

    def test_three_original_one_expert_and_no_extra_batch_field(self):
        class DS:
            def __init__(self,expert=False):self.expert=expert
            def __len__(self):return 100
            def __getitem__(self,i):
                value=dict(action=i,authority='expert' if self.expert else 'original')
                if self.expert:value['supervision_source']='expert_correction'
                return value
        samples=d.mixed_samples(DS(),DS(True),np.random.default_rng(9))
        self.assertEqual([s['authority'] for s in samples],['original']*3+['expert'])
        self.assertTrue(all(set(s)=={'action','authority'} for s in samples))
        with self.assertRaises(ValueError):d.mixed_samples(DS(),DS(),np.random.default_rng(9))

    def test_new_schedule_has_exact_twenty_thousand_boundaries(self):
        values=[d.rate(s) for s in range(30000,50000)]
        self.assertEqual(len(values),20000)
        self.assertTrue(all(1e-6<=v<=3e-6 for v in values))
        self.assertAlmostEqual(values[99],3e-6)
        self.assertLess(values[-1],1.000001e-6)
        for s in (29999,50000):
            with self.assertRaises(ValueError):d.rate(s)


if __name__=='__main__':unittest.main()
