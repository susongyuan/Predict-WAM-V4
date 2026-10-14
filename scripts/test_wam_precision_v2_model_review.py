#!/usr/bin/env python3
"""Adversarial CPU-only Precision-WAM V2 model/dataflow regressions.

Real small DiT fixtures; no optimizer, encoder forward, production dataset,
GPU, Isaac, or task-success claims. Weight edits below open zero-start paths
only in disposable fixtures. Run with OMP/MKL/OPENBLAS_NUM_THREADS=1.
"""
import copy
import os
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch

from harness.robotwin_vtt_adapter import ResidualVTTProjection
from harness.wam_event_clock import episode_event_features
from harness.wam_hjepa import adapt_independent_frames
from harness.wam_precision_v2 import PrecisionHeadsV2, upgrade_precision_v2
from scripts.test_wam_precision import direct_inputs, fixture, run


class PrecisionV2ModelReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, *, queries=4, reduced=2, open_residual=True):
        base, batch = fixture(queries=queries)
        wrapper = upgrade_precision_v2(base, queries_per_view=reduced).eval()
        if open_residual:
            self.open_residual(wrapper.action_model.precision)
        return wrapper, batch

    @staticmethod
    def open_residual(head):
        with torch.no_grad():
            head.residual.weight.copy_(torch.randn(
                head.residual.weight.shape,
                generator=torch.Generator(device='cpu').manual_seed(309)) * .025)

    def assert_finite_grads(self, wrapper):
        grads = [p.grad for p in wrapper.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        self.assertFalse(torch.cuda.is_initialized())

    def assert_no_hooks(self, wrapper):
        self.assertFalse(wrapper.action_model._forward_pre_hooks)
        self.assertFalse(wrapper.action_model.dino_adapters[0]._forward_hooks)

    def test_16_queries_100_tokens_positions_and_late_state_injection(self):
        base, batch = fixture(queries=64)
        original_queries = base.action_model.dino_adapters[0].query_embed.detach().clone()
        w = upgrade_precision_v2(base).eval()
        m = w.action_model
        inputs = direct_inputs(w, batch)
        seen = {}
        handles = [m.dino_adapters[0].register_forward_hook(
            lambda _m, _a, out: seen.update(adapted=out))]
        for i, block in enumerate(m.blocks):
            def capture(_m, args, kwargs, i=i):
                seen[i] = (args[0].detach().clone(), dict(kwargs))
            handles.append(block.register_forward_pre_hook(capture, with_kwargs=True))
        try:
            out = m(**inputs)
        finally:
            for h in handles:
                h.remove()
        torch.testing.assert_close(m.dino_adapters[0].query_embed,
                                   original_queries.reshape(1, 16, 4, 32).mean(2), rtol=0, atol=0)
        first, kw = seen[0]
        self.assertEqual(first.shape, (2, 100, 32))
        self.assertEqual(out['final_pred'].shape, (2, 32, 16))
        self.assertEqual(out['cond_tokens'].shape, (2, 68, 32))
        self.assertNotIn('attn_mask', kw)
        self.assertFalse(first[:, 32:39].any())
        torch.testing.assert_close(first[:, :32], m.action_proj(inputs['noisy_actions'])
                                   + m.action_pos_emb + m.type_emb_action, rtol=0, atol=0)
        torch.testing.assert_close(seen[1][0][:, 32:39], m.proprio_proj(inputs['qpos_history'])
                                   + m.proprio_pos_emb + m.type_emb_proprio, rtol=0, atol=0)
        visual = (seen['adapted'].reshape(2, 3, 16, 32)
                  + m.type_emb_dino_layers[0].unsqueeze(1)
                  + m.view_emb.unsqueeze(0).squeeze(2))
        torch.testing.assert_close(first[:, 42:90], visual.reshape(2, 48, 32), rtol=0, atol=0)
        memory = (m.mem_proj(inputs['memory']) + m.mem_slot_emb + m.type_emb_mem
                  + m.view_emb.unsqueeze(0).squeeze(2))
        torch.testing.assert_close(first[:, 90:99], memory.reshape(2, 9, 32), rtol=0, atol=0)
        torch.testing.assert_close(first[:, -1], m.precision.clock(batch['event_state']), rtol=0, atol=0)
        self.assertFalse(torch.equal(m.action_pos_emb[:, 0], m.action_pos_emb[:, 1]))
        self.assertFalse(torch.equal(m.proprio_pos_emb[:, 0], m.proprio_pos_emb[:, 1]))

    def test_every_current_view_and_other_online_condition_changes_native16(self):
        w, batch = self.make()
        m = w.action_model
        inputs = direct_inputs(w, batch)
        with torch.no_grad():
            expected = m(**inputs)['final_pred']
            for camera in range(3):
                changed = copy.deepcopy(inputs)
                for f in changed['dino_features_list']:
                    f[:, camera] = -f[:, camera].flip(-1)
                with self.subTest(camera=camera):
                    self.assertGreater(float((m(**changed)['final_pred'] - expected).abs().max()), 1e-8)
            for key in ('qpos_history', 'memory', 'task_cond', 'event_state', 't', 'noisy_actions'):
                changed = dict(inputs)
                changed[key] = inputs[key] * -.75 + .1
                with self.subTest(condition=key):
                    actual = m(**changed)['final_pred']
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertGreater(float((actual - expected).abs().max()), 1e-8)

    def test_future_targets_cannot_change_open_refiner_actions_or_action_gradients(self):
        w, batch = self.make()
        changed = dict(batch)
        for key in ('future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            changed[key] = -batch[key].flip(-1) + 2.
        loss1, _, out1 = run(w, batch)
        out1['final_pred'].square().mean().backward()
        grads1 = {n: p.grad.clone() for n, p in w.action_model.named_parameters() if p.grad is not None}
        w.zero_grad(set_to_none=True)
        loss2, _, out2 = run(w, changed)
        out2['final_pred'].square().mean().backward()
        torch.testing.assert_close(out1['final_pred'], out2['final_pred'], rtol=0, atol=0)
        torch.testing.assert_close(out1['cond_tokens'], out2['cond_tokens'], rtol=0, atol=0)
        self.assertNotEqual(float(loss1.detach()), float(loss2.detach()))
        for name, p in w.action_model.named_parameters():
            if name in grads1:
                torch.testing.assert_close(p.grad, grads1[name], rtol=0, atol=0, msg=name)
            else:
                self.assertIsNone(p.grad, name)

    def test_masked_nonfinite_action_padding_is_removed_before_noise(self):
        w, batch = self.make()
        mask = batch['action_valid_mask'].clone()
        mask[0, 24:] = False
        mask[1, 16:] = False
        batch = dict(batch, action_valid_mask=mask)
        reference_loss, _, reference_out = run(w, batch)
        for value in (float('nan'), float('inf'), -1e20):
            changed = dict(batch, action_sequence=batch['action_sequence'].masked_fill(~mask[..., None], value))
            with self.subTest(padding=value):
                loss, _, out = run(w, changed)
                torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
                torch.testing.assert_close(out['final_pred'], reference_out['final_pred'], rtol=0, atol=0)
        loss.backward()
        self.assert_finite_grads(w)

    def test_partial_auxiliary_masks_ignore_nan_teacher_rows(self):
        w, batch = self.make()
        batch = dict(batch)
        for key in ('future_valid_mask', 'future_valid_mask_8', 'idm_valid_mask'):
            batch[key] = torch.tensor([True, False])
        expected, _, out0 = run(w, batch)
        changed = dict(batch)
        for key in ('future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            changed[key] = batch[key].clone()
            changed[key][1] = float('nan')
        loss, info, out = run(w, changed)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        torch.testing.assert_close(out['final_pred'], out0['final_pred'], rtol=0, atol=0)
        self.assertEqual(info['future8_valid_samples'], 1)
        self.assertEqual(info['idm_valid_samples'], 1)
        loss.backward()
        self.assert_finite_grads(w)

    def test_all_auxiliary_masks_can_omit_unused_future_tensors(self):
        w, batch = self.make()
        batch = dict(batch)
        for key in ('future_valid_mask', 'future_valid_mask_8', 'idm_valid_mask'):
            batch[key] = torch.zeros(2, dtype=torch.bool)
        for key in ('future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            batch.pop(key)
        with mock.patch('harness.wam_hjepa.future_with_offset', side_effect=AssertionError('no valid future8')), \
                mock.patch.object(w.action_model.hjepa_aux, 'inverse_actions', side_effect=AssertionError('no valid IDM')):
            loss, info, out = run(w, batch)
        for key in ('future', 'future8', 'idm'):
            self.assertEqual(info[key], 0.)
        self.assertIsNone(out['idm_pred'])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assert_finite_grads(w)
        self.assert_no_hooks(w)

    def test_valid_nan_targets_malformed_masks_and_clock_fail_with_clean_hooks(self):
        w, batch = self.make()
        cases = []
        for key in ('future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw'):
            cases.append((key, torch.full_like(batch[key], float('nan'))))
        for key in ('action_valid_mask', 'future_valid_mask', 'future_valid_mask_8', 'idm_valid_mask'):
            cases.extend([(key, batch[key].float()), (key, batch[key][:1])])
        cases.extend([('event_state', None), ('event_state', torch.zeros(2, 5)),
                      ('event_state', torch.zeros(2, 6, dtype=torch.int64)),
                      ('event_state', torch.full((2, 6), float('inf')))])
        for key, value in cases:
            with self.subTest(field=key, shape=None if value is None else tuple(value.shape)):
                with self.assertRaises(ValueError):
                    run(w, dict(batch, **{key: value}))
                self.assert_no_hooks(w)
        loss, _, _ = run(w, batch)
        self.assertTrue(torch.isfinite(loss))

    def test_bf16_cache_bits_equal_dequantized_frozen_features(self):
        w, batch = self.make()
        quantized, decoded = dict(batch), dict(batch)
        for key in ('feats', 'future_feats', 'future_feats_8', 'idm_future_feats'):
            bf = batch[key].to(torch.bfloat16)
            quantized[key] = bf.view(torch.uint16)
            decoded[key] = bf.float()
        first, _, out1 = run(w, quantized)
        second, _, out2 = run(w, decoded)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(out1['final_pred'], out2['final_pred'], rtol=0, atol=0)
        torch.testing.assert_close(out1['idm_pred'], out2['idm_pred'], rtol=0, atol=0)

    def test_cpu_bfloat16_autocast_forward_backward_finite(self):
        w, batch = self.make()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss, _, out = run(w, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(out['final_pred']).all())
        loss.backward()
        self.assert_finite_grads(w)
        for module in (w.action_model.precision.patch_proj, w.action_model.precision.attention,
                       w.action_model.precision.clock, w.action_model.concat_fusion):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0.)

    def test_constant_velocity_euler_integrates_once_and_preserves_all_16_channels(self):
        w, batch = self.make()
        velocity = torch.linspace(-.15, .2, 16)[None, None].expand(2, 32, 16)
        times = []
        def constant_velocity(t, *, noisy_actions, event_state, **_):
            times.append(float(t[0]))
            torch.testing.assert_close(event_state, batch['event_state'], rtol=0, atol=0)
            return dict(final_pred=velocity, gate=torch.full((2, 3), 1/3),
                        phase_logits=torch.zeros(2, 4), log_dist=torch.zeros(2))
        generator = torch.Generator(device='cpu').manual_seed(41)
        initial = torch.randn(2, 32, 16, generator=torch.Generator(device='cpu').manual_seed(41))
        with mock.patch.object(w.action_model, 'forward', side_effect=constant_velocity):
            actual, _ = w.sample_actions([batch['feats'][:, :, i] for i in range(3)],
                batch['state'], batch['pool'], batch['task_cond'], num_steps=10,
                generator=generator, event_state=batch['event_state'])
        torch.testing.assert_close(actual, w.denormalize_action(initial + velocity), rtol=1e-6, atol=2e-6)
        torch.testing.assert_close(torch.tensor(times), torch.arange(10).float()/10, rtol=0, atol=0)
        self.assertEqual(actual.shape[-1], 16)  # Includes grippers at 7 and 15; no 14D remapping.
        self.assert_no_hooks(w)

    def test_real_v2_sampler_matches_manual_euler_and_holds_clock_fixed(self):
        w, batch = self.make()
        m = w.action_model
        clocks = []
        handle = m.precision.clock.register_forward_pre_hook(lambda _m, args: clocks.append(args[0].clone()))
        try:
            with mock.patch.object(w.vision_encoder, 'forward', side_effect=AssertionError('online encoder not needed')), \
                    mock.patch.object(m.hjepa_aux, 'inverse_actions', side_effect=AssertionError('online IDM')), \
                    mock.patch.object(m.future_feat_decoder, 'forward', side_effect=AssertionError('online teacher')):
                actual, _ = w.sample_actions([batch['feats'][:, :, i] for i in range(3)],
                    batch['state'], batch['pool'], batch['task_cond'], num_steps=10,
                    generator=torch.Generator(device='cpu').manual_seed(83), event_state=batch['event_state'])
        finally:
            handle.remove()
        self.assertEqual(len(clocks), 10)
        for clock in clocks:
            torch.testing.assert_close(clock, batch['event_state'], rtol=0, atol=0)
        inputs = direct_inputs(w, batch)
        x = torch.randn(2, 32, 16, generator=torch.Generator(device='cpu').manual_seed(83))
        with torch.no_grad():
            for i in range(10):
                out = m(**dict(inputs, t=torch.full((2,), i/10), noisy_actions=x))
                x = x + out['final_pred'].float()/10
        torch.testing.assert_close(actual, w.denormalize_action(x), rtol=0, atol=0)
        self.assert_no_hooks(w)

    def test_policy_is_observation_causal_not_autoregressive_chunk_causal(self):
        w, batch = self.make()
        inputs = direct_inputs(w, batch)
        changed = dict(inputs, noisy_actions=inputs['noisy_actions'].clone())
        changed['noisy_actions'][:, 8:] = -changed['noisy_actions'][:, 8:].flip(-1) + .3
        with torch.no_grad():
            before = w.action_model(**inputs)
            after = w.action_model(**changed)
        # Planned/noisy future actions are legitimate flow inputs, not future observations.
        self.assertGreater(float((before['final_pred'][:, :8]-after['final_pred'][:, :8]).abs().max()), 1e-8)
        self.assertGreater(float((before['cond_tokens']-after['cond_tokens']).abs().max()), 1e-8)

    def test_idm_ignores_history_clock_task_memory_and_flow_noise_but_reads_current_q(self):
        w, batch = self.make()
        reference = run(w, batch)[2]['idm_pred']
        state = batch['state'].clone()
        state[:, 1:] = -state[:, 1:] + .7
        changed = dict(batch, state=state, action_sequence=-batch['action_sequence'],
                       task_cond=-batch['task_cond'], pool=-batch['pool'], event_state=batch['event_state'] + 1.)
        out = run(w, changed, seed=77)[2]
        torch.testing.assert_close(out['idm_pred'], reference, rtol=0, atol=0)
        out['aux_losses']['idm'].backward()
        for module in (w.action_model.precision, w.action_model.action_proj,
                       w.action_model.proprio_proj, w.action_model.task_cond_proj, w.action_model.mem_proj):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        state[:, 0, :7] += .8
        with torch.no_grad():
            current_changed = run(w, dict(changed, state=state), seed=77)[2]['idm_pred']
        self.assertGreater(float((current_changed-reference).abs().max()), 1e-8)

    def test_auxiliary_visual_frames_are_independent_and_cache_values_detached(self):
        w, batch = self.make()
        features = torch.cat((batch['feats'][:, None], batch['idm_future_feats']), dim=1).requires_grad_()
        reference = adapt_independent_frames(w.action_model, features)
        changed = features.detach().clone()
        changed[1, 3, 2] = -changed[1, 3, 2].flip(-1)
        after = adapt_independent_frames(w.action_model, changed)
        unaffected = torch.ones(2, 5, 96, dtype=torch.bool)
        unaffected[1, 3, 64:] = False
        torch.testing.assert_close(reference[unaffected], after[unaffected], rtol=0, atol=0)
        self.assertGreater(float((reference[1, 3, 64:]-after[1, 3, 64:]).abs().max()), 1e-8)
        reference.square().mean().backward()
        self.assertIsNone(features.grad)
        self.assert_finite_grads(w)

    def test_full_14x14_three_view_dense_kv_excludes_cls_registers_and_preserves_xy(self):
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(615)
            head = PrecisionHeadsV2(8, 32, 4, 196, 16, 8)
            self.open_residual(head)
            hidden = torch.randn(2, 32, 32)
            fused = torch.randn(2, 3, 201, 8, requires_grad=True)
            views = torch.randn(3, 1, 1, 32) * .1
        seen = {}
        handle = head.attention.register_forward_pre_hook(lambda _m, args: seen.update(q=args[0], k=args[1]))
        try:
            reference = head(hidden, fused, views)
        finally:
            handle.remove()
        self.assertEqual(seen['k'].shape, (2, 588, 32))
        torch.testing.assert_close(seen['q'], hidden[:, :8], rtol=0, atol=0)
        torch.testing.assert_close(head.patch_xy[0, 0, 0], torch.tensor([-13/14, -13/14]))
        torch.testing.assert_close(head.patch_xy[0, 0, -1], torch.tensor([13/14, 13/14]))
        changed = fused.detach().clone()
        changed[:, :, :5] = 1e9
        torch.testing.assert_close(reference, head(hidden, changed, views), rtol=0, atol=0)
        with torch.no_grad():
            head.patch_xy.copy_(head.patch_xy.flip(2))
        self.assertGreater(float((reference-head(hidden, changed, views)).abs().max()), 1e-8)
        reference.square().mean().backward()
        self.assertFalse(fused.grad[:, :, :5].any())
        for camera in range(3):
            self.assertGreater(float(fused.grad[:, camera, 5:].abs().sum()), 0.)

    def test_checkpoint_rejects_missing_or_incompatible_spatial_and_prefix_identity(self):
        w, _ = self.make()
        good = copy.deepcopy(w.action_model.state_dict())
        for name in ('precision.version', 'precision.configuration', 'precision.patch_xy'):
            for missing in (False, True):
                bad = copy.deepcopy(good)
                if missing:
                    bad.pop(name)
                else:
                    bad[name] = bad[name] + 1
                target, _ = self.make()
                with self.subTest(key=name, missing=missing), self.assertRaises(RuntimeError):
                    target.action_model.load_state_dict(bad, strict=True)
                # A failed PyTorch strict load is not transactional: discard target.
        fresh, batch = self.make()
        fresh.action_model.load_state_dict(good, strict=True)
        torch.testing.assert_close(run(fresh, batch)[2]['final_pred'], run(w, batch)[2]['final_pred'], rtol=0, atol=0)

    def test_event_clock_past_only_4s_6s_distinction_reaches_policy(self):
        state = np.zeros((150, 16), dtype=np.float32)
        action = np.zeros_like(state)
        clock = episode_event_features(state, action, initial_target=action[0])
        changed_state, changed_action = state.copy(), action.copy()
        changed_state[61:, :7] = 2.
        changed_action[60:, [7, 15]] = 1.
        changed_clock = episode_event_features(changed_state, changed_action, initial_target=action[0])
        np.testing.assert_array_equal(clock[:61], changed_clock[:61])
        self.assertAlmostEqual(float(clock[40, 0]), .4, places=6)
        self.assertAlmostEqual(float(clock[60, 0]), .6, places=6)
        self.assertTrue((clock[-1, [0, 1, 3, 4]] == 1.).all())
        w, batch = self.make()
        inputs = direct_inputs(w, batch)
        with torch.no_grad():
            first = w.action_model(**dict(inputs, event_state=torch.from_numpy(clock[40]).expand(2, 6)))['final_pred']
            second = w.action_model(**dict(inputs, event_state=torch.from_numpy(clock[60]).expand(2, 6)))['final_pred']
        self.assertGreater(float((first-second).abs().max()), 1e-8)

    def test_production_residual_vtt_keeps_base_frozen_and_condition_residual_trainable(self):
        base, batch = fixture()
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(701)
            projection = torch.nn.Sequential(torch.nn.LayerNorm(1024), torch.nn.Linear(1024, 32),
                                             torch.nn.GELU(), torch.nn.Linear(32, 32))
            base.action_model.task_cond_proj = ResidualVTTProjection(projection, hidden=32)
            batch = dict(batch, task_cond=torch.randn(2, 1024))
        w = upgrade_precision_v2(base, queries_per_view=2).eval()
        inputs = direct_inputs(w, batch)
        vtt = w.action_model.task_cond_proj
        changed = dict(inputs, task_cond=-inputs['task_cond'].flip(-1))
        with torch.no_grad():
            zero_start = w.action_model(**inputs)['final_pred']
            torch.testing.assert_close(zero_start, w.action_model(**changed)['final_pred'], rtol=0, atol=0)
            vtt.residual.weight.copy_(torch.randn(vtt.residual.weight.shape,
                generator=torch.Generator(device='cpu').manual_seed(702)) * .002)
        out = w.action_model(**inputs)['final_pred']
        self.assertGreater(float((out-w.action_model(**changed)['final_pred']).abs().max()), 1e-8)
        out.square().mean().backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in vtt.base.parameters()))
        self.assertTrue(vtt.residual.weight.requires_grad)
        self.assertGreater(float(vtt.residual.weight.grad.abs().sum()), 0.)
        self.assert_finite_grads(w)

    def test_flow_endpoints_saturated_clock_and_history_limits_remain_finite(self):
        w, batch = self.make()
        inputs = direct_inputs(w, batch)
        for t in (0., 1.):
            for clock in (0., 1.):
                with self.subTest(flow_time=t, event_clock=clock):
                    out = w.action_model(**dict(inputs, t=torch.full((2,), t),
                        event_state=torch.full((2, 6), clock), memory=torch.full_like(inputs['memory'], 5.),
                        qpos_history=torch.full_like(inputs['qpos_history'], -5.)))
                    self.assertTrue(torch.isfinite(out['final_pred']).all())
                    w.zero_grad(set_to_none=True)
                    out['final_pred'].square().mean().backward()
                    self.assert_finite_grads(w)


if __name__ == '__main__':
    unittest.main(verbosity=2)
