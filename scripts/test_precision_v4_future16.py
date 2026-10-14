#!/usr/bin/env python3
"""Real CPU witnesses: label timing/masks, decoder gradients and unchanged sampling."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from harness import zhuhai_precision_v4_future16 as future
from scripts.test_zhuhai_hjepa_dataset import Episode, MockStore, dataset, features
from scripts.test_wam_precision_v4 import make_pair, sample
from scripts.test_eval_precision_v4 import fixture
from scripts import eval_precision_v4_final5 as legacy_queue
from scripts import eval_precision_v4_future16_final5 as new_queue


class Future16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_frame16_and_independent_terminal_masks(self):
        for terminal in (True, False):
            ep = Episode(n=40, terminal=terminal)
            store = MockStore()
            ds = future.Future16Dataset(dataset([ep], store=store, multi_horizon=False))
            first = ds[0]
            np.testing.assert_array_equal(first['future_feats_16'], features(ep, 16)[:, -1, 5:])
            np.testing.assert_array_equal(first['future_feats'], features(ep, 32)[:, -1, 5:])
            self.assertNotIn('future_feats_8', first)
            self.assertNotIn((str(ep.path), 8), store.calls)
            last_anchor = 24 if terminal else 23
            self.assertTrue(ds[last_anchor]['future_valid_mask_16'])
            invalid = ds[last_anchor + 1]
            self.assertFalse(invalid['future_valid_mask_16'])
            self.assertFalse(invalid['future_valid_mask'])
            self.assertFalse(invalid['future_feats_16'].any())

    def test_same_decoder_and_weights_use_only_frame16_labels(self):
        _, original, old_batch = make_pair()
        wrapper = future.Future16Wrapper(original.base, original.settings['variant'])
        batch = dict(old_batch)
        batch['future_feats_16'] = batch.pop('future_feats_8').clone().requires_grad_()
        batch['future_valid_mask_16'] = batch.pop('future_valid_mask_8')
        gold_batch = dict(batch, future_feats_8=batch['future_feats_16'],
                          future_valid_mask_8=batch['future_valid_mask_16'])
        rng = torch.get_rng_state().clone()
        gold, _, _ = original(gold_batch)
        torch.set_rng_state(rng)
        loss, info, output = wrapper(batch)
        torch.testing.assert_close(loss, gold, rtol=0, atol=0)
        self.assertEqual(info['future_prediction_horizons'], [16, 32])
        self.assertIn('future16', info)
        self.assertNotIn('future8', info)
        self.assertIn('future16', output['aux_losses'])
        loss.backward()
        grad = wrapper.action_model.hjepa_aux.future8_offset.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)
        self.assertIsNone(batch['future_feats_16'].grad)
        torch.testing.assert_close(sample(wrapper, batch), sample(original, old_batch), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'legacy'):
            wrapper(old_batch)
        with self.assertRaisesRegex(ValueError, 'Future16 loss'):
            wrapper(batch, {'future8': .25})

    def test_contract_changes_only_future_semantics_and_source_binding(self):
        old = json.loads((ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v4_task2_50k_20261012_r2/train/manifest.json').read_text())['contract']
        before = copy.deepcopy(old)
        new = future.new_contract(old)
        future.validate_training_contract(new)
        self.assertEqual(old, before)
        self.assertEqual(new['splits'], old['splits'])
        self.assertEqual(new['statistics'], old['statistics'])
        self.assertEqual(new['v4_training'], old['v4_training'])
        self.assertEqual(new['time'], old['time'])
        self.assertEqual(new['precision_v4']['execution_prefix'], 8)
        self.assertEqual(new['config']['training']['auxiliary_loss']['future16'], .25)
        self.assertNotIn('future8', new['config']['training']['auxiliary_loss'])
        for broken in (old, dict(new, future_prediction=dict(new['future_prediction'], horizons=[8, 32]))):
            with self.assertRaises(ValueError):
                future.validate_training_contract(broken)

    def test_production_dataset_wiring_does_not_generate_plus8(self):
        contract = json.loads((ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v4_task2_50k_20261012_r2/train/manifest.json').read_text())['contract']
        class Store:
            def __init__(self): self.calls = []
            def get(self, ep, frame):
                self.calls.append(frame)
                return (np.full((3, 3, 201, 1024), 0x3d00 + frame, np.uint16),
                        np.full((3, 2048), frame, np.float32))
        store = Store()
        splits = {future.TASK: {'train': contract['splits'][future.TASK]['train'][:1]}}
        ds = future.build_datasets(splits, store, contract['config'], variant='combined')['train'][future.TASK]
        ds.condition = np.asarray(contract['conditions'][future.TASK], np.float32)
        row = ds[0]
        self.assertEqual(int(row['future_feats_16'][0, 0, 0]), 0x3d00 + 16)
        self.assertEqual(int(row['future_feats'][0, 0, 0]), 0x3d00 + 32)
        self.assertNotIn(8, store.calls)
        self.assertNotIn('future_feats_8', row)

    def test_eval_binding_requires_actual_future16_and_exec8(self):
        raw, contract = fixture()
        rt = raw['policy_ready']['runtime']
        rt.update(chunk_vtt16=future.PRECISION_V4_SPEC,
                  future_prediction=future.FUTURE_PREDICTION,
                  future_prediction_horizons=[16, 32], execution_prefix=8)
        with patch.object(legacy_queue, 'v4', future):
            new_queue.validate_binding(raw, 130000, 932101, 'a' * 64, contract)
            rt['future_prediction_horizons'] = [8, 32]
            with self.assertRaises(legacy_queue.e.EvidenceError):
                new_queue.validate_binding(raw, 130000, 932101, 'a' * 64, contract)


if __name__ == '__main__':
    unittest.main()
