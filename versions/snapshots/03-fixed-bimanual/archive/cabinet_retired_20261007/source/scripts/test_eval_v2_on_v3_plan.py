#!/usr/bin/env python3
"""CPU-only tests for paired cached V2/V3 evaluation. No DINO/checkpoint downloads or CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch
from omegaconf import OmegaConf
import eval_v2_on_v3_plan as ev


class Action(torch.nn.Module):
    action_len = 32
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.states, self.noise = [], []
    def forward(self, t, noisy_actions, qpos_history, dino_features_list, task_cond):
        self.states.append(qpos_history.detach().clone())
        if float(t[0]) == 0:
            self.noise.append(noisy_actions.clone())
        return {'final_pred': torch.zeros_like(noisy_actions), 'cond_tokens': torch.zeros(1)}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / 'config.json').write_text(json.dumps({'hidden_size': 4, 'patch_size': 2, 'num_register_tokens': 2}))
        norm = {'behavior1k': {'state': {'min': [-1.] * 61, 'max': [1.] * 61},
                               'action': {'min': [-1.] * 23, 'max': [1.] * 23}}}
        (root / 'norm.json').write_text(json.dumps(norm))
        self.cfg = OmegaConf.create({'common': {'proprio_len': 1}, 'training': {},
            'dataset': {'stats_path': str(root / 'norm.json')},
            'model': {'views': ['observation.rgb.zed_link_camera_0'] * 3,
                      'proprio_mask': 'full', 'vision_encoder': {'checkpoint_path': str(root),
                                                                'feat_layers': [-12, -8, -4]}}})
        self.action = Action()
        with patch.object(ev, 'create_action_model', return_value=self.action):
            self.wrapper = ev.build_v2(self.cfg, {'model_state_dict': self.action.state_dict()}, 'cpu')

    def test_actual_v2_normalization_current_only_euler10_and_fixed_noise(self):
        state = torch.stack([torch.full((61,), .25), torch.full((61,), -99.)])[None]
        feats = [torch.zeros(1, 3, 9, 4, dtype=torch.bfloat16)] * 3
        gen = lambda: torch.Generator().manual_seed(123)
        pred = ev.sample_v2(self.wrapper, feats, state, torch.zeros(1, 4), gen())
        expected = torch.randn(1, 32, 23, generator=gen())
        torch.testing.assert_close(pred, self.wrapper.denormalize_action(expected), rtol=0, atol=0)
        self.assertEqual(len(self.action.states), 10)
        for x in self.action.states:
            self.assertEqual(x.shape, (1, 1, 61))
            torch.testing.assert_close(x, torch.full_like(x, .25), rtol=0, atol=0)
        self.assertIsNone(self.wrapper.vision_encoder)
        with self.assertRaisesRegex(ValueError, 'CURRENT'):
            ev.sample_v2(self.wrapper, feats, torch.zeros(2, 61), torch.zeros(1, 4), gen())

    def test_plan_validation_and_real_pilot_cli(self):
        fine = [{'index': i, 'episode': 1, 'anchor': i, 'skill': 'press', 'phase': 1,
                 'active_hand': 0, 'fine': True} for i in range(256)]
        p = {'version': 1, 'split': 'validation', 'seed': 0, 'samples_per_anchor': 1, 'terminal_step': 31, 'fine': fine}
        self.assertEqual(len(ev.validate_plan(p)), 256)
        for bad in ({**p, 'fine': fine[:-1]}, {**p, 'fine': [fine[0]] * 256},
                    {**p, 'split': 'train_diagnostic'}, {**p, 'terminal_step': 7}, {**p, 'samples_per_anchor': 8}):
            with self.assertRaises(ValueError):
                ev.validate_plan(bad)
        path = Path(self.tmp.name) / 'plan.json'
        path.write_text(json.dumps(p))
        with patch('builtins.print'), patch.object(ev, 'build_v2', side_effect=AssertionError('CLI loaded weights')):
            self.assertEqual(ev.main(['--plan', str(path), '--check-plan-only']), 0)

    def test_pair_uses_same_noise_terminal31_both_hands_and_hold(self):
        data, records = [], []
        for i, hand in enumerate((0, 1)):
            target = torch.zeros(32, 23)
            target[31, [3, 7, 15]] = torch.tensor([1., 2., 3.])
            state = torch.stack([torch.full((61,), .25), torch.full((61,), 99.)])
            data.append({'episode': 1, 'anchor': i * 2, 'phase': 1, 'active_hand': hand,
                'state': state, 'feats': torch.zeros(3, 3, 9, 4, dtype=torch.int16), 'pool': torch.zeros(3, 2, 8),
                'task_cond': torch.zeros(4), 'action_sequence': target})
            records.append({'index': i, 'episode': 1, 'anchor': i * 2, 'phase': 1, 'active_hand': hand,
                            'skill': ['pick up from', 'press'][i], 'fine': True})
        plan = {'seed': 9, 'fine': records}
        class Kin:
            def fk(self, arm, q):
                return np.array([q[0], q[4], 0.]), None
        noise3 = []
        class V3:
            device = torch.device('cpu')
            def sample_actions(self, feats, state, pool, cond, num_steps, generator):
                assert num_steps == 10
                noise3.append(torch.randn(1, 32, 23, generator=generator))
                result = torch.zeros(1, 32, 23)
                result[0, 31, [3, 7, 15]] = torch.tensor([1., 2., 3.])
                return result, {}
        r2 = ev.evaluate_fine(self.wrapper, data, plan, Kin(), 'v2')
        r3 = ev.evaluate_fine(V3(), data, plan, Kin(), 'v3')
        for i in range(2):
            torch.testing.assert_close(self.action.noise[i], noise3[i], rtol=0, atol=0)
            self.assertEqual(r2[i]['noise_seed'], ev.trainer.noise_seed(9, 1, i * 2))
            self.assertEqual(r3[i]['fine_eef_m'], 0.)  # only terminal step matches: detects index7/average error
            self.assertAlmostEqual(r3[i]['fine_hold_eef_m'], np.hypot(.25 - 1, .25 - (2 + i)))
        report = ev.compare(r2, r3)
        self.assertEqual(report['overall']['samples'], 2)
        self.assertEqual(set(report['by_skill']), {'pick up from', 'press'})
        self.assertLess(report['overall']['v3_minus_v2_m'], 0)
        wrong = [dict(x) for x in r3]
        wrong[0]['noise_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'noise_seed'):
            ev.compare(r2, wrong)

    def test_strict_loading_and_encoder_prohibition(self):
        with patch.object(ev, 'create_action_model', return_value=Action()):
            with self.assertRaises(RuntimeError):
                ev.build_v2(self.cfg, {'model_state_dict': {}}, 'cpu')
        with ev.forbid_encoder() as runner:
            with self.assertRaisesRegex(RuntimeError, 'DINO loading forbidden'):
                runner.ModelFactory.create_vision_encoder('fake')


if __name__ == '__main__':
    torch.set_num_threads(2)
    result = unittest.main(exit=False, verbosity=2).result
    assert not torch.cuda.is_initialized()
    raise SystemExit(0 if result.wasSuccessful() else 1)
