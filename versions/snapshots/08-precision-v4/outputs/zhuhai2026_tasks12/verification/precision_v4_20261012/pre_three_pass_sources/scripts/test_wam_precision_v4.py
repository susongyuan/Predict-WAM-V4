#!/usr/bin/env python3
"""CPU tests of real small V3/V4 DiT, causal labels, gradients and ten Euler calls."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from harness.wam_precision_v3 import upgrade_precision_v3
from harness.wam_precision_v4 import ChunkVTTPredictor, chunk_vtt_loss, upgrade_precision_v4
from harness.zhuhai_precision_v4_training import ChunkVTTDataset
from scripts.test_wam_precision import fixture, run
from scripts.test_zhuhai_hjepa_dataset import Episode, MockStore, dataset


def make_pair():
    hjepa, batch = fixture()
    v3 = upgrade_precision_v3(hjepa, queries_per_view=2)
    with torch.no_grad():
        v3.action_model.precision.residual.weight.normal_(std=.02)
    v4 = upgrade_precision_v4(copy.deepcopy(v3), hidden=32)
    batch['chunk_vtt_target'] = torch.randn(2, 3, 8)
    batch['chunk_vtt_valid_mask'] = torch.ones(2, dtype=torch.bool)
    return v3, v4, batch


def sample(wrapper, batch):
    return wrapper.sample_actions([batch['feats'][:, :, i] for i in range(3)],
        batch['state'], batch['pool'], batch['task_cond'], num_steps=10,
        generator=torch.Generator().manual_seed(719), event_state=batch['event_state'])[0]


class PrecisionV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_added_parameter_count_and_dimensions(self):
        module = ChunkVTTPredictor(1024, 16, 1024, 768)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 7877377)
        self.assertEqual(module.predictor[0].in_features, 7184)
        self.assertEqual(module.predictor[-1].out_features, 3072)
        self.assertEqual(float(module.alpha), 0.)
        self.assertGreater(float(module.projection.weight.abs().sum()), 0.)

    def test_old_parameters_topology_and_rng_preserved(self):
        hjepa, _ = fixture()
        v3 = upgrade_precision_v3(hjepa, queries_per_view=2)
        old = {k: v.clone() for k, v in v3.action_model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with patch('torch.cuda.manual_seed_all') as cuda_seed:
            v4 = upgrade_precision_v4(v3, hidden=32)
        cuda_seed.assert_not_called()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        state = v4.action_model.state_dict()
        for key, value in old.items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0, msg=key)
        self.assertTrue(all(k.startswith('chunk_vtt16.') for k in set(state) - set(old)))
        self.assertEqual(v4.action_model.precision.precision_steps, 32)
        self.assertEqual(v4.action_model.dino_adapters[0].query_embed.shape, (1, 2, 32))
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            upgrade_precision_v4(v4)

    def test_zero_alpha_exact_full_ten_euler_v3_equivalence_predict_once(self):
        v3, v4, batch = make_pair()
        counts = {'predict': 0, 'task': 0}
        def prediction(*args):
            counts['predict'] += 1
        def task(*args):
            counts['task'] += 1
        handles = [v4.action_model.chunk_vtt16.register_forward_hook(prediction),
                   v4.action_model.task_cond_proj.register_forward_hook(task)]
        try:
            expected, actual = sample(v3, batch), sample(v4, batch)
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(counts, {'predict': 1, 'task': 10})
        self.assertFalse(v4.action_model.task_cond_proj._forward_hooks)
        self.assertFalse(v4.action_model._forward_pre_hooks)

    def test_open_alpha_changes_actions_and_zero_again_recovers_same_v3(self):
        v3, v4, batch = make_pair()
        expected = sample(v3, batch)
        with torch.no_grad():
            v4.action_model.chunk_vtt16.alpha.fill_(.2)
        self.assertGreater(float((sample(v4, batch) - expected).abs().max()), 1e-8)
        with torch.no_grad():
            v4.action_model.chunk_vtt16.alpha.zero_()
        torch.testing.assert_close(sample(v4, batch), expected, rtol=0, atol=0)

    def test_future_labels_never_affect_actions_or_prediction(self):
        _, v4, batch = make_pair()
        with torch.no_grad():
            v4.action_model.chunk_vtt16.alpha.fill_(.2)
        changed = copy.deepcopy(batch)
        for name in ('future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw', 'chunk_vtt_target'):
            changed[name] = changed[name] * -3 + 5
        _, _, before = run(v4, batch)
        _, _, after = run(v4, changed)
        for key in ('final_pred', 'chunk_vtt_prediction'):
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)

    def test_independent_supervision_learns_while_backbone_and_alpha_frozen(self):
        _, v4, batch = make_pair()
        v4.action_model.requires_grad_(False)
        v4.action_model.chunk_vtt16.predictor.requires_grad_(True)
        frozen = {name: p.clone() for name, p in v4.action_model.named_parameters() if not p.requires_grad}
        for name in ('pool', 'state', 'task_cond', 'chunk_vtt_target'):
            batch[name].requires_grad_()
        loss, info, _ = run(v4, batch)
        loss.backward()
        for p in v4.action_model.chunk_vtt16.predictor.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0.)
        for name, p in v4.action_model.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad, name)
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
        self.assertIsNone(batch['chunk_vtt_target'].grad)
        for key in ('chunk_vtt_loss', 'chunk_huber', 'chunk_cosine', 'alpha'):
            self.assertTrue(np.isfinite(info[key]))

    def test_zero_gate_has_gradient_and_labels_remain_detached(self):
        _, v4, batch = make_pair()
        batch['chunk_vtt_target'].requires_grad_()
        _, _, out = run(v4, batch)
        out['final_pred'].square().mean().backward()
        alpha = v4.action_model.chunk_vtt16.alpha
        self.assertIsNotNone(alpha.grad)
        self.assertTrue(torch.isfinite(alpha.grad))
        self.assertGreater(float(alpha.grad.abs()), 0.)
        self.assertIsNone(batch['chunk_vtt_target'].grad)

    def test_stationary_cosine_and_invalid_nan_labels(self):
        prediction = torch.randn(2, 3, 8, requires_grad=True)
        target = torch.zeros_like(prediction)
        _, _, cosine = chunk_vtt_loss(prediction, target, torch.ones(2, dtype=torch.bool))
        self.assertEqual(float(cosine), 0.)
        target[1].fill_(float('nan'))
        loss, _, _ = chunk_vtt_loss(prediction, target, torch.tensor([True, False]))
        self.assertTrue(torch.isfinite(loss))
        zero, _, _ = chunk_vtt_loss(prediction, target, torch.zeros(2, dtype=torch.bool))
        self.assertEqual(float(zero), 0.)
        zero.backward()
        with self.assertRaisesRegex(ValueError, 'Nonfinite valid'):
            chunk_vtt_loss(prediction, target, torch.ones(2, dtype=torch.bool))

    def test_input_checks_strict_identity_and_hook_cleanup(self):
        _, v4, batch = make_pair()
        module = v4.action_model.chunk_vtt16
        state = copy.deepcopy(module.state_dict())
        state['configuration'][1] = 32
        with self.assertRaisesRegex(RuntimeError, 'configuration'):
            module.load_state_dict(state, strict=True)
        broken = dict(batch, event_state=torch.zeros(2, 5))
        with self.assertRaisesRegex(ValueError, 'event_state'):
            run(v4, broken)
        self.assertFalse(v4.action_model.task_cond_proj._forward_hooks)
        self.assertFalse(v4.action_model._forward_pre_hooks)
        broken = copy.deepcopy(batch)
        broken['pool'][0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            run(v4, broken)

    def test_dataset_delta16_is_loss_only_and_terminal_mask_not_clamped(self):
        ep = Episode(n=20)
        store = MockStore()
        base = dataset([ep], store=store, multi_horizon=True, use_idm=True)
        candidate = ChunkVTTDataset(base)
        before = base[4]
        valid = candidate[4]
        self.assertTrue(valid['chunk_vtt_valid_mask'])
        np.testing.assert_array_equal(valid['chunk_vtt_target'], np.full((3, 1024), 16, np.float32))
        for key in before:
            if isinstance(before[key], (np.ndarray, np.generic)):
                np.testing.assert_array_equal(before[key], valid[key], err_msg=key)
        store.calls.clear()
        invalid = candidate[5]
        self.assertFalse(invalid['chunk_vtt_valid_mask'])
        self.assertFalse(invalid['chunk_vtt_target'].any())
        self.assertNotIn((str(ep.path), 21), store.calls)
        ep.has_terminal = False
        self.assertFalse(candidate[4]['chunk_vtt_valid_mask'])

    def test_predict_inputs_match_existing_online_normalization(self):
        _, v4, batch = make_pair()
        captured = []
        handle = v4.action_model.chunk_vtt16.register_forward_pre_hook(lambda m, args: captured.append(args))
        try:
            v4.predict_chunk(batch['pool'], batch['state'], batch['task_cond'])
        finally:
            handle.remove()
        current, history, q, task = captured[0]
        torch.testing.assert_close(current, batch['pool'][:, :, 0, 8:])
        torch.testing.assert_close(history, v4.normalize_memory(batch['pool'])[:, :, 1, 8:])
        torch.testing.assert_close(q, v4.normalize_state(batch['state'])[:, 0])
        torch.testing.assert_close(task, batch['task_cond'])


if __name__ == '__main__':
    unittest.main()
