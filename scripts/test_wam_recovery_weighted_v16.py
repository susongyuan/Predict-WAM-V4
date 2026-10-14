"""CPU loss/gradient/mask/RNG checks using the real small Precision-v2 model."""
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import unittest
import torch
from harness.wam_precision_v2 import upgrade_precision_v2
import numpy as np
from harness.wam_recovery_weighted_v16 import install_recovery_weighting, weighted_action_mse, mixed_weighted_samples
from scripts.test_wam_precision import fixture


class RecoveryWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_equal_errors_double_correction_gradient_and_keep_scale(self):
        p = torch.ones(4, 32, 16, requires_grad=True)
        loss, info = weighted_action_mse(p, torch.zeros_like(p), None, torch.tensor([1., 1., 1., 2.]))
        self.assertEqual(float(loss), 1.)
        loss.backward()
        torch.testing.assert_close(p.grad[3], 2*p.grad[0])
        self.assertAlmostEqual(info['recovery_action_loss_mass_fraction'], .4)

    def test_invalid_policy_targets_never_affect_loss_or_gradient(self):
        p = torch.ones(4, 32, 16, requires_grad=True)
        mask = torch.ones(4, 32, dtype=torch.bool); mask[3, 8:] = False
        target = torch.zeros_like(p); target[~mask] = float('nan')
        loss, info = weighted_action_mse(p, target, mask, torch.tensor([1., 1., 1., 2.]))
        self.assertEqual(float(loss), 1.)
        loss.backward()
        self.assertEqual(float(p.grad[~mask].abs().sum()), 0.)
        self.assertAlmostEqual(info['recovery_action_loss_mass_fraction'], 16/112)

    def test_real_wrapper_preserves_state_rng_predictions_and_auxiliaries(self):
        h, batch = fixture(); original = upgrade_precision_v2(h, queries_per_view=2)
        weighted = copy.deepcopy(original)
        before = copy.deepcopy(weighted.action_model.state_dict()); rng = torch.get_rng_state().clone()
        install_recovery_weighting(weighted)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(before), set(weighted.action_model.state_dict()))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, weighted.action_model.state_dict()[key]), key)
        torch.manual_seed(73); old_loss, old_info, old_out = original(batch); old_rng = torch.get_rng_state().clone()
        torch.manual_seed(73); loss, info, out = weighted(dict(batch, action_loss_weight=torch.ones(2)))
        torch.testing.assert_close(loss, old_loss, rtol=1e-7, atol=1e-7)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(out['final_pred'], old_out['final_pred']))
        for key in ('future', 'future8', 'idm'):
            self.assertEqual(info[key], old_info[key])
        torch.manual_seed(73); boosted, boosted_info, boosted_out = weighted(dict(batch, action_loss_weight=torch.tensor([1., 2.])))
        self.assertTrue(torch.isfinite(boosted))
        self.assertTrue(torch.equal(boosted_out['final_pred'], old_out['final_pred']))
        for key in ('future', 'future8', 'idm'):
            self.assertEqual(boosted_info[key], old_info[key])
        boosted.backward()
        self.assertTrue(any(p.grad is not None for p in weighted.action_model.precision.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in weighted.parameters()))

    def test_missing_weights_keep_offline_diagnostic_identical(self):
        h, batch = fixture(); a = upgrade_precision_v2(h, queries_per_view=2)
        b = install_recovery_weighting(copy.deepcopy(a))
        torch.manual_seed(29); expected = a(batch)[0]
        torch.manual_seed(29); actual = b(batch)[0]
        self.assertTrue(torch.equal(actual, expected))

    def test_bad_weights_and_empty_mask_fail_closed(self):
        p = torch.ones(4, 32, 16)
        for weights in (torch.tensor([1., 1., 1., 0.]), torch.tensor([1., 1., 1., float('nan')]), torch.ones(3)):
            with self.assertRaises(ValueError):
                weighted_action_mse(p, p, None, weights)
        with self.assertRaises(ValueError):
            weighted_action_mse(p, p, torch.zeros(4, 32, dtype=torch.bool), torch.ones(4))

    def test_sampler_preserves_authority_check_rng_and_three_plus_one(self):
        from harness.zhuhai_dagger_segmented_training_v7 import mixed_samples
        class Original:
            def __len__(self): return 100
            def __getitem__(self, i): return dict(anchor=i)
        class Corrections:
            def sample(self, rng): return dict(anchor=int(rng.integers(100)), supervision_source='expert_correction')
        a, b = np.random.default_rng(18), np.random.default_rng(18)
        old = mixed_samples(Original(), Corrections(), a)
        new = mixed_weighted_samples(Original(), Corrections(), b)
        self.assertEqual(old, [{k:v for k,v in x.items() if k != 'action_loss_weight'} for x in new])
        self.assertEqual(a.bit_generator.state, b.bit_generator.state)
        self.assertEqual([float(x['action_loss_weight']) for x in new], [1., 1., 1., 2.])
        class Policy(Corrections):
            def sample(self, rng): return dict(anchor=1, supervision_source='policy')
        with self.assertRaises(ValueError):
            mixed_weighted_samples(Original(), Policy(), b)


if __name__ == '__main__':
    unittest.main()
