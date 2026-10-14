#!/usr/bin/env python3
"""Independent CPU wiring checks; no optimizer, GPU, training or robot rollout.

Set WAM_PRECISION_V2_REPORT to save a machine-readable test receipt. Small
fixtures exercise the real DiT/wrapper code; they are not task-success evidence.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch

from harness.wam_precision import upgrade_precision
from harness.wam_precision_v2 import (
    PRECISION_VERSION, REFINEMENT_VIEW_INDICES, PrecisionHeadsV2, upgrade_precision_v2)
from scripts.test_wam_precision import direct_inputs, fixture, run, sample


class WamPrecisionV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_v2(self):
        h, batch = fixture()
        return upgrade_precision_v2(h, queries_per_view=2), batch

    def assert_grad(self, module):
        grads = [p.grad for p in module.parameters()]
        self.assertTrue(grads)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_parameter_shapes_weights_and_rng_match_v1(self):
        h1, _ = fixture()
        h2, _ = fixture()
        v1 = upgrade_precision(h1, queries_per_view=2)
        rng = torch.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all') as seed_cuda:
            v2 = upgrade_precision_v2(h2, queries_per_view=2)
        seed_cuda.assert_not_called()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        p1, p2 = dict(v1.action_model.named_parameters()), dict(v2.action_model.named_parameters())
        self.assertEqual(set(p1), set(p2))
        self.assertEqual(sum(p.numel() for p in p1.values()), sum(p.numel() for p in p2.values()))
        for name in p1:
            self.assertEqual(p1[name].requires_grad, p2[name].requires_grad, name)
            torch.testing.assert_close(p1[name], p2[name], rtol=0, atol=0, msg=name)
        self.assertIsInstance(v2.action_model.precision, PrecisionHeadsV2)
        self.assertEqual(int(v2.action_model.precision.version), PRECISION_VERSION)
        self.assertEqual(v2.action_model.precision.configuration.tolist(), [2, 8, 0, 1, 2])

    def test_three_current_patch_grids_coordinates_and_action_queries_enter_kv(self):
        w, batch = self.make_v2()
        model = w.action_model
        with torch.no_grad():
            for i in REFINEMENT_VIEW_INDICES:
                model.view_emb[i].fill_(.1 * (i + 1))
        captured = {}
        handles = [
            model.concat_fusion.register_forward_hook(lambda m, a, out: captured.update(fused=out)),
            model.final_norm.register_forward_hook(lambda m, a, out: captured.update(hidden=out)),
            model.precision.patch_proj.register_forward_pre_hook(lambda m, a: captured.update(patch_input=a[0])),
            model.precision.patch_proj.register_forward_hook(lambda m, a, out: captured.update(projected=out)),
            model.precision.attention.register_forward_pre_hook(lambda m, a: captured.update(qkv=a)),
        ]
        try:
            model(**direct_inputs(w, batch))
        finally:
            for handle in handles:
                handle.remove()
        raw = captured['fused'].reshape(2, 3, 9, 8)[:, :, 5:]
        torch.testing.assert_close(captured['patch_input'][..., :8], raw, rtol=0, atol=0)
        xy = torch.tensor([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]])
        torch.testing.assert_close(captured['patch_input'][..., 8:], xy.expand(2, 3, 4, 2))
        q, k, v = captured['qkv']
        self.assertEqual(k.shape, (2, 12, 32))  # Three four-patch grids in the CPU fixture.
        torch.testing.assert_close(q, captured['hidden'][:, :8], rtol=0, atol=0)
        expected = captured['projected'] + model.view_emb.unsqueeze(0).squeeze(2)
        torch.testing.assert_close(k, expected.reshape(2, 12, 32), rtol=0, atol=0)
        self.assertIs(k, v)

    def test_zero_refinement_then_only_execution_prefix_changes(self):
        w, batch = self.make_v2()
        model = w.action_model
        direct = direct_inputs(w, batch)
        main = []
        handle = model.output_proj.register_forward_hook(lambda m, a, out: main.append(out))
        try:
            initial = model(**direct)['final_pred']
            torch.testing.assert_close(initial, main[-1], rtol=0, atol=0)
            self.assertFalse(model.precision.residual.weight.any())
            self.assertFalse(model.precision.residual.bias.any())
            with torch.no_grad():
                model.precision.residual.bias.fill_(.125)
            refined = model(**direct)['final_pred']
        finally:
            handle.remove()
        torch.testing.assert_close(refined[:, :8], initial[:, :8] + .125, rtol=0, atol=0)
        torch.testing.assert_close(refined[:, 8:], initial[:, 8:], rtol=0, atol=0)

    def test_zero_initialized_v1_v2_actions_and_global_tokens_are_identical(self):
        h1, batch = fixture()
        h2, _ = fixture()
        v1 = upgrade_precision(h1, queries_per_view=2)
        v2 = upgrade_precision_v2(h2, queries_per_view=2)
        first = v1.action_model(**direct_inputs(v1, batch))
        second = v2.action_model(**direct_inputs(v2, batch))
        for key in ('final_pred', 'cond_tokens', 'gate', 'phase_logits', 'log_dist'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_dense_branch_receives_gradients_from_each_camera(self):
        w, _ = self.make_v2()
        head = w.action_model.precision
        action_hidden = torch.randn(2, 32, 32)
        fused = torch.randn(2, 3, 9, 8, requires_grad=True)
        view_emb = w.action_model.view_emb
        # Open the zero-start path with test-only weights, without an optimizer.
        with torch.no_grad():
            head.residual.weight.copy_(torch.linspace(-.04, .03, 16 * 32).reshape(16, 32))
        head(action_hidden, fused, view_emb).square().mean().backward()
        for camera in REFINEMENT_VIEW_INDICES:
            self.assertGreater(float(fused.grad[:, camera, 5:].abs().sum()), 0.)
            self.assertGreater(float(view_emb.grad[camera].abs().sum()), 0.)
        self.assertFalse(fused.grad[:, :, :5].any())
        for module in (head.patch_proj, head.attention, head.residual):
            self.assert_grad(module)

    def test_action_loss_zero_start_then_shared_gradient_and_detached_inputs(self):
        w, batch = self.make_v2()
        model = w.action_model
        for key in ('feats', 'event_state', 'future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            batch[key].requires_grad_()
        run(w, batch)[0].backward()
        self.assert_grad(model.precision.residual)
        self.assert_grad(model.precision.clock)
        for module in (model.precision.patch_proj, model.precision.attention):
            self.assertTrue(all(p.grad is not None and not p.grad.any() for p in module.parameters()))
        w.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.precision.residual.weight.copy_(torch.linspace(-.04, .03, 16 * 32).reshape(16, 32))
        run(w, batch)[2]['final_pred'].square().mean().backward()
        for module in (model.precision.patch_proj, model.precision.attention,
                       model.concat_fusion, model.dino_adapters[0]):
            self.assert_grad(module)
        for key in ('feats', 'event_state', 'future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            self.assertIsNone(batch[key].grad, key)

    def test_future_perturbations_do_not_change_current_actions(self):
        w, batch = self.make_v2()
        changed = dict(batch)
        for key in ('future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            changed[key] = -batch[key] + 3.
        loss1, _, out1 = run(w, batch)
        loss2, _, out2 = run(w, changed)
        torch.testing.assert_close(out1['final_pred'], out2['final_pred'], rtol=0, atol=0)
        self.assertNotEqual(float(loss1), float(loss2))

    def test_idm_remains_before_action_and_clock_mixing(self):
        w, batch = self.make_v2()
        original = run(w, batch)[2]
        changed = dict(batch, event_state=batch['event_state'] + 10.)
        altered = run(w, changed, seed=77)[2]
        torch.testing.assert_close(original['idm_pred'], altered['idm_pred'], rtol=0, atol=0)
        original['aux_losses']['idm'].backward()
        self.assert_grad(w.action_model.concat_fusion)
        self.assert_grad(w.action_model.dino_adapters[0])
        self.assertTrue(all(p.grad is None for p in w.action_model.precision.parameters()))
        self.assertTrue(all(p.grad is None for p in w.action_model.action_proj.parameters()))

    def test_versions_reject_cross_loading_and_v2_roundtrips(self):
        h1, batch = fixture()
        h2, _ = fixture()
        v1 = upgrade_precision(h1, queries_per_view=2)
        v2 = upgrade_precision_v2(h2, queries_per_view=2)
        state1 = copy.deepcopy(v1.action_model.state_dict())
        state2 = copy.deepcopy(v2.action_model.state_dict())
        for destination, state in ((v1.action_model, state2), (v2.action_model, state1)):
            with self.subTest(version=int(state['precision.version'])):
                with self.assertRaisesRegex(RuntimeError, 'precision.version.*incompatible'):
                    destination.load_state_dict(state, strict=True)
        # Restore identities after intentionally rejected loads, whose PyTorch loader may copy valid keys.
        v1.action_model.precision.version.fill_(1)
        v2.action_model.precision.version.fill_(2)
        v1.action_model.load_state_dict(state1, strict=True)
        v2.action_model.load_state_dict(state2, strict=True)
        fresh, _ = self.make_v2()
        fresh.action_model.load_state_dict(state2, strict=True)
        torch.testing.assert_close(run(v2, batch)[2]['final_pred'], run(fresh, batch)[2]['final_pred'], rtol=0, atol=0)
        bad = copy.deepcopy(state2)
        bad['precision.configuration'][-1] = 1
        with self.assertRaisesRegex(RuntimeError, 'precision.configuration.*incompatible'):
            fresh.action_model.load_state_dict(bad, strict=True)

    def test_fresh_instance_only_and_no_online_auxiliary_calls(self):
        w, batch = self.make_v2()
        before = copy.deepcopy(w.action_model.state_dict())
        with self.assertRaisesRegex(ValueError, 'already installed'):
            upgrade_precision_v2(w, queries_per_view=2)
        for key, value in before.items():
            torch.testing.assert_close(w.action_model.state_dict()[key], value, rtol=0, atol=0)
        with mock.patch.object(w.vision_encoder, 'forward', side_effect=AssertionError('no encoder call')), \
                mock.patch.object(w.action_model.hjepa_aux, 'inverse_actions', side_effect=AssertionError('no IDM call')), \
                mock.patch.object(w.action_model.future_feat_decoder, 'forward', side_effect=AssertionError('no future decoder call')):
            actions, _ = sample(w, batch, event_state=batch['event_state'])
        self.assertEqual(actions.shape, (2, 32, 16))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertFalse(w.action_model._forward_pre_hooks)
        self.assertFalse(w.action_model.dino_adapters[0]._forward_hooks)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WamPrecisionV2Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report_path = os.environ.get('WAM_PRECISION_V2_REPORT')
    if report_path:
        root = Path(__file__).resolve().parents[1]
        sources = ('harness/wam_precision_v2.py', 'scripts/test_wam_precision_v2.py',
                   'harness/wam_precision.py', 'harness/wam_hjepa.py', 'scripts/test_wam_precision.py')
        report = dict(schema='precision-wam-v2-independent-cpu-checks-20261009',
            tests_run=result.testsRun, passed=result.wasSuccessful(),
            failures=[name for name, _ in result.failures], errors=[name for name, _ in result.errors],
            cuda_initialized=torch.cuda.is_initialized(), optimizer_created=False,
            training_updates=0, robot_policy_evaluation=False, task_success_evidence=False,
            fixture_scope='Small real DiT/wrapper CPU fixtures; no production data or encoder forward',
            sole_action_change='Near-8 refinement KV includes current head + both wrists',
            refinement_view_indices=list(REFINEMENT_VIEW_INDICES), production_kv_v1=392, production_kv_v2=588,
            added_learned_parameters=0, source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources})
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
