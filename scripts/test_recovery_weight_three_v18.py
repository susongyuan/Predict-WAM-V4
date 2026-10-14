"""CPU gradient, authority/RNG and verified-cache-resume checks."""
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from harness.wam_recovery_weighted_v18 import weighted_action_mse, mixed_weighted_samples
from harness.zhuhai_verified_cache_resume_v18 import clean_index_receipt, VerifiedResumeCache
from harness.zhuhai_hybrid_cache import HybridFeatureCache
from scripts.test_zhuhai_hybrid_cache import identity


class Tests(unittest.TestCase):
    def test_threefold_gradient_and_half_valid_mass(self):
        x = torch.ones(4, 32, 16, requires_grad=True)
        loss, info = weighted_action_mse(x, torch.zeros_like(x), None, torch.tensor([1.,1.,1.,3.]))
        self.assertEqual(float(loss), 1.); loss.backward()
        torch.testing.assert_close(x.grad[3], 3*x.grad[0])
        self.assertEqual(info['recovery_action_loss_mass_fraction'], .5)

    def test_same_sampler_rng_and_expert_only_labels(self):
        from harness.wam_recovery_weighted_v16 import mixed_weighted_samples as old
        class Original:
            def __len__(self): return 100
            def __getitem__(self,i): return dict(anchor=i)
        class Expert:
            def sample(self,rng): return dict(anchor=int(rng.integers(100)),supervision_source='expert_correction')
        a,b=np.random.default_rng(4),np.random.default_rng(4)
        expected=old(Original(),Expert(),a);actual=mixed_weighted_samples(Original(),Expert(),b)
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        self.assertEqual([x['anchor'] for x in actual],[x['anchor'] for x in expected])
        self.assertEqual([float(x['action_loss_weight']) for x in actual],[1.,1.,1.,3.])

    def test_clean_resume_skips_scan_and_corruption_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'cache';c=HybridFeatureCache(root,budget_bytes=1000000,min_free_bytes=0)
            c.read_bytes(identity(),lambda:b'original')
            proof=clean_index_receipt(root,budget_bytes=1000000,min_free_bytes=0,last_cache_stats=c.stats,exited_trainer_pid=99999999)
            path=Path(directory)/'proof.json';path.write_text(json.dumps(proof))
            resumed=VerifiedResumeCache(root,resume_receipt=path,budget_bytes=1000000,min_free_bytes=0)
            self.assertTrue(resumed.resume_index_reused)
            self.assertEqual(resumed.read_bytes(identity(),lambda:b'wrong'),b'original')
            self.assertEqual(resumed.stats['directory_scans'],0)
            next(root.glob('*.payload')).write_bytes(b'corrupt')
            self.assertEqual(resumed.read_bytes(identity(),lambda:b'rebuilt'),b'rebuilt')
            self.assertEqual(resumed.stats['corruptions'],1)

    def test_modified_index_falls_back_and_orphan_receipt_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'cache';c=HybridFeatureCache(root,budget_bytes=1000000,min_free_bytes=0)
            c.read_bytes(identity(),lambda:b'old')
            proof=clean_index_receipt(root,budget_bytes=1000000,min_free_bytes=0,last_cache_stats=c.stats,exited_trainer_pid=99999999)
            path=Path(directory)/'proof.json';path.write_text(json.dumps(proof))
            c.read_bytes(identity(1),lambda:b'new')
            resumed=VerifiedResumeCache(root,resume_receipt=path,budget_bytes=1000000,min_free_bytes=0)
            self.assertFalse(resumed.resume_index_reused)
            self.assertEqual(resumed.read_bytes(identity(),lambda:b'wrong'),b'old')
            self.assertEqual(resumed.stats['directory_scans'],1)
            (root/'orphan.txt').write_text('preserve')
            with self.assertRaises(ValueError):
                clean_index_receipt(root,budget_bytes=1000000,min_free_bytes=0,last_cache_stats=c.stats,exited_trainer_pid=99999999)


if __name__ == '__main__': unittest.main()
