#!/usr/bin/env python3
"""Small real-DiT CPU contract tests: no files, encoder runs, optimizer steps or task-success claims."""
import copy
import io
import json
import os
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from torch import nn

from harness.wam_v3 import HEAD_CAMERA, LEFT_WRIST, RIGHT_WRIST, WamV3Wrapper, v3_model_class
from harness.wam_hjepa import upgrade_wrapper
from harness.wam_precision import GATE_MODULES, PRECISION_VERSION, PrecisionWrapper, upgrade_precision


def fixture(queries=4, variant='combined'):
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(20261009)
        model = v3_model_class()(num_views=3, num_memory=3, memory_dim=16, phase_hidden=16,
            action_dim=16, proprio_dim=16, hidden_dim=32, num_heads=4, depth=2, action_len=32,
            proprio_len=7, num_registers=2, dino_feat_dims=(8, 8, 8), vlm_num_queries=queries,
            adapter_depth=1, state_inject_start=1, task_cond_dim=8, fusion_mode='concat',
            concat_out_dim=8, use_future_feat=True, future_feat_num_queries=4,
            future_feat_out_dim=8, future_feat_depth=1, future_feat_heads=4)
        # A loaded checkpoint has nonzero DiT modulation. Initialize a small stand-in, without training.
        for block in model.blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=.02)
        stats = {'views': [HEAD_CAMERA, LEFT_WRIST, RIGHT_WRIST],
                 'proprio_diff_std': [[1.] * 16 for _ in range(6)],
                 'memory_diff_std': [[[1.] * 16 for _ in range(3)] for _ in range(3)]}
        norm = {'action_repr': 'absolute', 'native16': {
            'action': {'min': [-2.] * 16, 'max': [2.] * 16},
            'state': {'min': [-2.] * 16, 'max': [2.] * 16}}}
        # The production constructor reads JSON; supply it in memory, not a temporary workspace file.
        with mock.patch('harness.wam_v3.Path.read_text', return_value=json.dumps(norm)):
            base = WamV3Wrapper(model, views=stats['views'], stats=stats, norm_stats_path='unused.json',
                proprio_mask='native_all16', proprio_keep_override=[1.] * 16,
                normalization_key='native16', device='cpu', vision_encoder=nn.Linear(1, 1))
        b = 2
        batch = dict(feats=torch.randn(b, 3, 3, 9, 8), state=torch.randn(b, 7, 16) * .2,
            pool=torch.randn(b, 3, 4, 16), action_sequence=torch.randn(b, 32, 16) * .2,
            action_valid_mask=torch.ones(b, 32, dtype=torch.bool),
            future_valid_mask=torch.ones(b, dtype=torch.bool), future_feats=torch.randn(b, 3, 4, 8),
            task_cond=torch.randn(b, 8), phase=torch.full((b,), -1),
            future_feats_8=torch.randn(b, 3, 4, 8), future_valid_mask_8=torch.ones(b, dtype=torch.bool),
            idm_future_feats=torch.randn(b, 4, 3, 3, 9, 8), idm_target_raw=torch.randn(b, 4, 16) * .1,
            idm_valid_mask=torch.ones(b, dtype=torch.bool), event_state=torch.randn(b, 6))
        return upgrade_wrapper(base, variant=variant, idm_scale=[1.] * 16, idm_hidden=32), batch


def run(wrapper, batch, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return wrapper(batch)


def direct_inputs(wrapper, batch):
    return dict(t=torch.full((2,), .4), noisy_actions=batch['action_sequence'],
        qpos_history=wrapper.normalize_state(batch['state']),
        dino_features_list=[batch['feats'][:, :, i] for i in range(3)],
        task_cond=batch['task_cond'], memory=wrapper.normalize_memory(batch['pool']),
        event_state=batch['event_state'])


def sample(wrapper, batch, **kwargs):
    return wrapper.sample_actions([batch['feats'][:, :, i] for i in range(3)],
        batch['state'], batch['pool'], batch['task_cond'], num_steps=2,
        generator=torch.Generator(device='cpu').manual_seed(7), **kwargs)


class WamPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_grad(self, module):
        grads = [p.grad for p in module.parameters()]
        self.assertTrue(grads)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def assert_no_hooks(self, model):
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model.dino_adapters[0]._forward_hooks)

    def test_query_mean_weights_rng_and_instance_only_upgrade(self):
        h, b = fixture()
        m = h.action_model
        before = copy.deepcopy(m.state_dict())
        untouched = copy.deepcopy(m)
        cls_forward = type(m).forward
        rng = torch.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all') as cuda_seed, \
                mock.patch('torch.cuda.get_rng_state', side_effect=AssertionError('no CUDA RNG access')):
            w = upgrade_precision(h, queries_per_view=2)
        cuda_seed.assert_not_called()
        self.assertIsInstance(w, PrecisionWrapper)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        expected = before['dino_adapters.0.query_embed'].reshape(1, 2, 2, 32).mean(dim=2)
        torch.testing.assert_close(m.dino_adapters[0].query_embed, expected, rtol=0, atol=0)
        self.assertEqual(m.dino_adapters[0].num_queries, 2)
        after = m.state_dict()
        deleted = {key for key in before if key.split('.')[0] in GATE_MODULES}
        self.assertEqual(set(before) - set(after), deleted)
        self.assertTrue(all(key.startswith('precision.') for key in set(after) - set(before)))
        for key in set(before) & set(after) - {'dino_adapters.0.query_embed'}:
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
        self.assertEqual(int(m.precision.version), PRECISION_VERSION)
        self.assertIs(type(m).forward, cls_forward)
        self.assertIs(untouched.forward.__func__, cls_forward)
        self.assertIsNot(m.forward.__func__, cls_forward)
        args = direct_inputs(w, b)
        args.pop('event_state')
        self.assertEqual(untouched(**args)['final_pred'].shape, (2, 32, 16))
        with self.assertRaisesRegex(ValueError, 'event_state'):
            m(**args)
        self.assertTrue(all(p.requires_grad for p in m.concat_fusion.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.dino_adapters.parameters()))
        self.assertFalse(any(p.requires_grad for p in w.vision_encoder.parameters()))

    def test_default_token_budget_100_shapes_and_no_global_gate(self):
        h, b = fixture(queries=64)
        w = upgrade_precision(h)
        m = w.action_model
        seen = []
        handles = [block.attn1.register_forward_pre_hook(
            lambda mod, args, kw: seen.append((args[0].shape[1], kw.get('attn_mask'))), with_kwargs=True)
            for block in m.blocks]
        try:
            loss, info, out = run(w, b)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(seen, [(100, None), (100, None)])
        self.assertEqual(out['final_pred'].shape, (2, 32, 16))
        self.assertEqual(out['cond_tokens'].shape, (2, 68, 32))
        self.assertEqual(out['idm_pred'].shape, (2, 4, 16))
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), info['mse'] + .25 * info['future32']
                               + .25 * info['future8'] + .1 * info['idm'], places=5)
        torch.testing.assert_close(out['gate'], torch.full((2, 3), 1. / 3), rtol=0, atol=0)
        self.assertFalse(out['phase_logits'].any())
        self.assertFalse(out['log_dist'].any())
        self.assertTrue(all(not hasattr(m, name) for name in GATE_MODULES))
        self.assert_no_hooks(m)

    def test_zero_residual_and_only_execution_prefix_changes(self):
        h, b = fixture()
        w = upgrade_precision(h, queries_per_view=2)
        m = w.action_model
        base_predictions = []
        handle = m.output_proj.register_forward_hook(lambda mod, args, out: base_predictions.append(out))
        try:
            initial = m(**direct_inputs(w, b))['final_pred']
            torch.testing.assert_close(initial, base_predictions[-1], rtol=0, atol=0)
            self.assertFalse(m.precision.residual.weight.any())
            self.assertFalse(m.precision.residual.bias.any())
            with torch.no_grad():
                m.precision.residual.bias.fill_(.125)
            refined = m(**direct_inputs(w, b))['final_pred']
        finally:
            handle.remove()
        torch.testing.assert_close(refined[:, :8], initial[:, :8] + .125, rtol=0, atol=0)
        torch.testing.assert_close(refined[:, 8:], initial[:, 8:], rtol=0, atol=0)

    def test_zero_refiner_does_not_claim_lossless_legacy_policy_migration(self):
        h, b = fixture()
        legacy = copy.deepcopy(h.action_model)
        legacy_args = direct_inputs(h, b)
        legacy_args.pop('event_state')
        old = legacy(**legacy_args)
        w = upgrade_precision(h, queries_per_view=2)
        current = w.action_model(**direct_inputs(w, b))
        self.assertFalse(w.action_model.precision.residual.weight.any())
        self.assertFalse(w.action_model.precision.residual.bias.any())
        # This is a new observation bottleneck/clock-conditioned policy.
        # A zero refinement must never be reported as unchanged legacy output.
        self.assertNotEqual(old['cond_tokens'].shape, current['cond_tokens'].shape)
        self.assertFalse(torch.equal(old['final_pred'], current['final_pred']))

    def test_full_wrist_patch_coordinates_view_embeddings_and_query_hidden(self):
        h, b = fixture()
        w = upgrade_precision(h, queries_per_view=2)
        m = w.action_model
        with torch.no_grad():
            m.view_emb[1].fill_(.2)
            m.view_emb[2].fill_(-.3)
        captured = {}
        handles = [
            m.concat_fusion.register_forward_hook(lambda mod, args, out: captured.update(fused=out)),
            m.final_norm.register_forward_hook(lambda mod, args, out: captured.update(hidden=out)),
            m.precision.patch_proj.register_forward_pre_hook(lambda mod, args: captured.update(patch_input=args[0])),
            m.precision.patch_proj.register_forward_hook(lambda mod, args, out: captured.update(projected=out)),
            m.precision.attention.register_forward_pre_hook(lambda mod, args: captured.update(qkv=args)),
        ]
        try:
            m(**direct_inputs(w, b))
        finally:
            for handle in handles:
                handle.remove()
        raw = captured['fused'].reshape(2, 3, 9, 8)[:, 1:3, 5:]
        torch.testing.assert_close(captured['patch_input'][..., :8], raw, rtol=0, atol=0)
        xy = torch.tensor([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]])
        torch.testing.assert_close(captured['patch_input'][..., 8:], xy.expand(2, 2, 4, 2))
        q, k, v = captured['qkv']
        torch.testing.assert_close(q, captured['hidden'][:, :8], rtol=0, atol=0)
        expected_kv = captured['projected'] + m.view_emb[1:3].unsqueeze(0).squeeze(2)
        torch.testing.assert_close(k, expected_kv.reshape(2, 8, 32), rtol=0, atol=0)
        self.assertIs(k, v)

    def test_new_gradients_zero_start_then_open_path_without_optimizer(self):
        h, b = fixture()
        w = upgrade_precision(h, queries_per_view=2)
        b['feats'].requires_grad_()
        b['event_state'].requires_grad_()
        for key in ('future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            b[key].requires_grad_()
        m = w.action_model
        run(w, b)[0].backward()
        self.assert_grad(m.precision.residual)
        self.assert_grad(m.precision.clock)
        for mod in (m.precision.patch_proj, m.precision.attention):
            self.assertTrue(all(p.grad is not None and not p.grad.any() for p in mod.parameters()))
        w.zero_grad(set_to_none=True)
        with torch.no_grad():
            m.precision.residual.weight.copy_(torch.linspace(-.04, .03, 16 * 32).reshape(16, 32))
        # Nonzero test-only weights open the path; no optimizer is even constructed.
        out = run(w, b)[2]
        out['final_pred'].square().mean().backward()
        for mod in (m.precision.clock, m.precision.patch_proj, m.precision.attention,
                    m.precision.residual, m.concat_fusion, m.dino_adapters[0]):
            self.assert_grad(mod)
        for key in ('feats', 'event_state', 'future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            self.assertIsNone(b[key].grad, key)
        self.assertTrue(all(p.grad is None for p in w.vision_encoder.parameters()))

    def test_future_perturbations_cannot_change_current_actions(self):
        h, b = fixture()
        w = upgrade_precision(h, queries_per_view=2)
        changed = dict(b)
        for key in ('future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            changed[key] = -b[key] + 3.
        loss, _, original = run(w, b)
        altered_loss, _, altered = run(w, changed)
        torch.testing.assert_close(original['final_pred'], altered['final_pred'], rtol=0, atol=0)
        self.assertNotEqual(float(loss), float(altered_loss))
        self.assert_no_hooks(w.action_model)

    def test_idm_stays_pre_action_pre_clock_and_updates_shared_visuals(self):
        h, b = fixture()
        w = upgrade_precision(h, queries_per_view=2)
        original = run(w, b)[2]
        changed = dict(b, event_state=b['event_state'] + 10.)
        clock_changed = run(w, changed)[2]
        noise_changed = run(w, changed, seed=77)[2]
        torch.testing.assert_close(original['idm_pred'], clock_changed['idm_pred'], rtol=0, atol=0)
        torch.testing.assert_close(original['idm_pred'], noise_changed['idm_pred'], rtol=0, atol=0)
        self.assertFalse(torch.equal(original['final_pred'], clock_changed['final_pred']))
        original['aux_losses']['idm'].backward()
        self.assert_grad(w.action_model.concat_fusion)
        self.assert_grad(w.action_model.dino_adapters[0])
        self.assertTrue(all(p.grad is None for p in w.action_model.precision.parameters()))
        self.assertTrue(all(p.grad is None for p in w.action_model.action_proj.parameters()))
        self.assert_no_hooks(w.action_model)

    def test_missing_malformed_clock_and_phase_supervision_rejected(self):
        h, b = fixture()
        w = upgrade_precision(h, queries_per_view=2)
        missing = dict(b)
        missing.pop('event_state')
        with self.assertRaisesRegex(ValueError, 'event_state'):
            w(missing)
        with self.assertRaisesRegex(ValueError, 'event_state'):
            sample(w, b)
        for bad in (None, torch.zeros(2, 5), torch.zeros(1, 6), torch.zeros(2, 6, dtype=torch.int64),
                    torch.full((2, 6), float('nan')), torch.full((2, 6), float('inf'))):
            with self.subTest(clock=bad), self.assertRaisesRegex(ValueError, 'event_state'):
                w(dict(b, event_state=bad))
            self.assert_no_hooks(w.action_model)
        with self.assertRaisesRegex(ValueError, 'loss weights'):
            w(b, lambdas={'gate_kl': .1})
        self.assert_no_hooks(w.action_model)

    def test_strict_checkpoint_roundtrip_version_and_shape_contract(self):
        h, b = fixture()
        legacy = copy.deepcopy(h.action_model)
        w = upgrade_precision(h, queries_per_view=2)
        stream = io.BytesIO()
        torch.save(w.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
        new_h, _ = fixture()
        new = upgrade_precision(new_h, queries_per_view=2)
        new.load_state_dict(state, strict=True)
        torch.testing.assert_close(run(w, b)[2]['final_pred'], run(new, b)[2]['final_pred'], rtol=0, atol=0)
        model_state = copy.deepcopy(w.action_model.state_dict())
        with self.assertRaises(RuntimeError):
            legacy.load_state_dict(model_state, strict=True)
        wrong_version = copy.deepcopy(model_state)
        wrong_version['precision.version'] += 1
        with self.assertRaisesRegex(RuntimeError, 'precision.version'):
            new.action_model.load_state_dict(wrong_version, strict=True)
        steps_h, _ = fixture()
        fewer_steps = upgrade_precision(steps_h, queries_per_view=2, precision_steps=4)
        with self.assertRaisesRegex(RuntimeError, 'precision.configuration'):
            fewer_steps.action_model.load_state_dict(model_state, strict=True)
        missing = dict(model_state)
        missing.pop('precision.clock.weight')
        with self.assertRaises(RuntimeError):
            w.action_model.load_state_dict(missing, strict=True)

    def test_sampling_requires_only_current_inputs_never_encoder_or_future_aux(self):
        h, b = fixture()
        w = upgrade_precision(h, queries_per_view=2)
        m = w.action_model
        with mock.patch.object(w.vision_encoder, 'forward', side_effect=AssertionError('no DINO encoding')), \
                mock.patch.object(m.hjepa_aux, 'inverse_actions', side_effect=AssertionError('no IDM')), \
                mock.patch.object(m.future_feat_decoder, 'forward', side_effect=AssertionError('no future decoding')):
            actions, info = sample(w, b, event_state=b['event_state'])
        self.assertEqual(actions.shape, (2, 32, 16))
        self.assertTrue(torch.isfinite(actions).all())
        torch.testing.assert_close(info['gate'], torch.full((2, 3), 1. / 3))
        self.assertFalse(info['phase_logits'].any())
        self.assertFalse(info['log_dist'].any())
        self.assertFalse(hasattr(m, 'event_state'))
        self.assert_no_hooks(m)
        # No clock from that call may survive into a subsequent naked model call.
        args = direct_inputs(w, b)
        args.pop('event_state')
        with self.assertRaisesRegex(ValueError, 'event_state'):
            m(**args)
        again, _ = sample(w, b, event_state=b['event_state'])
        torch.testing.assert_close(actions, again, rtol=0, atol=0)

    def test_hook_cleanup_on_forward_aux_and_sampling_exceptions(self):
        h, b = fixture()
        w = upgrade_precision(h, queries_per_view=2)
        m = w.action_model
        with self.assertRaises(ValueError):
            w(dict(b, action_sequence=torch.zeros(2, 31, 16)))
        self.assert_no_hooks(m)
        for module in (m.precision.attention, m.future_feat_decoder, m.hjepa_aux.idm_head):
            with mock.patch.object(module, 'forward', side_effect=RuntimeError('injected failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    w(b)
            self.assert_no_hooks(m)
        with mock.patch.object(m.precision.attention, 'forward', side_effect=RuntimeError('sample failure')):
            with self.assertRaisesRegex(RuntimeError, 'sample failure'):
                sample(w, b, event_state=b['event_state'])
        self.assert_no_hooks(m)
        handle = m.register_forward_pre_hook(lambda mod, args, kw: (args, dict(kw, event_state=b['event_state'])),
                                             with_kwargs=True)
        try:
            with self.assertRaisesRegex(ValueError, 'supplied twice'):
                w(b)
            self.assertEqual(len(m._forward_pre_hooks), 1)  # Preserve someone else's hook.
        finally:
            handle.remove()
        self.assert_no_hooks(m)

    def test_repeat_upgrade_invalid_options_and_seed_reproducibility(self):
        h, _ = fixture()
        before = copy.deepcopy(h.action_model.state_dict())
        for options in ({'queries_per_view': 3}, {'queries_per_view': 0}, {'queries_per_view': True},
                        {'queries_per_view': 2.5}, {'precision_steps': 9}, {'precision_steps': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                upgrade_precision(h, **options)
            self.assertFalse(hasattr(h.action_model, 'precision'))
        for key, value in before.items():
            torch.testing.assert_close(h.action_model.state_dict()[key], value, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'H-JEPA'):
            upgrade_precision(h.base)
        w = upgrade_precision(h, queries_per_view=2, aux_seed=9)
        for target in (w, h):
            with self.assertRaisesRegex(ValueError, 'already installed'):
                upgrade_precision(target, queries_per_view=2)
        h2, _ = fixture()
        same = upgrade_precision(h2, queries_per_view=2, aux_seed=9)
        for key, value in w.action_model.precision.state_dict().items():
            torch.testing.assert_close(same.action_model.precision.state_dict()[key], value, rtol=0, atol=0)
        h3, _ = fixture()
        other = upgrade_precision(h3, queries_per_view=2, aux_seed=10)
        self.assertFalse(torch.equal(w.action_model.precision.clock.weight, other.action_model.precision.clock.weight))

    def test_all_hjepa_variants_remain_loss_compatible(self):
        for variant in ('baseline', 'multiscale', 'idm', 'combined'):
            with self.subTest(variant=variant):
                h, b = fixture(variant=variant)
                w = upgrade_precision(h, queries_per_view=2)
                loss, info, out = run(w, b)
                settings = w.settings
                expected = (info['mse'] + settings['future32_weight'] * info['future32']
                            + settings['future8_weight'] * info.get('future8', 0.)
                            + settings['idm_weight'] * info.get('idm', 0.))
                self.assertAlmostEqual(float(loss), expected, places=5)
                self.assertEqual(out['final_pred'].shape, (2, 32, 16))
                self.assert_no_hooks(w.action_model)

    def test_no_gpu_context(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
