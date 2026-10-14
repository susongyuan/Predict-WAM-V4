#!/usr/bin/env python3
"""CPU-only old-HandoffPolicy/V3 integration regression, no demo assets/checkpoints/GPU needed.

Run: CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .conda-envs/lilawam/bin/python scripts/test_wam_v3_handoff.py
Live launch (NOT executed by this test): P2_TASK_ID=0 ... run ... --wam-samples 1 --wam-exec-horizon 8
Optional --wam-fixed-noise-seed 0 means same initial noise at each replan, NOT exec_horizon=1.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch
from omegaconf import OmegaConf
import test_wam_v3_deployment as fixtures
import replay_demo_actions_feasibility as replay
from harness import policy_server as ps
from harness.wam_v3 import history_frames


class HandoffTest(unittest.TestCase):
    # Reuse the existing CPU fake networks and REAL WamPolicy/WamV3Wrapper loader fixture, not its tests.
    setUp = fixtures.DeploymentTest.setUp
    policy = fixtures.DeploymentTest.policy
    save_checkpoint = fixtures.DeploymentTest.save_checkpoint

    def handoff(self, p=None, N=5, seed=19):
        p = self.policy() if p is None else p
        p.base_control_mode = 'wam_whole_body'
        cond = np.arange(1024, dtype=np.float32) / 1024
        contract = 'pick up from(radio,coffee_table)'
        p.bank = NS(select=lambda *args: NS(vector=cond))
        targets = self.root / 'targets.json'
        targets.write_text(json.dumps({'contracts': {contract: {'vector': cond.tolist()}}}))
        out = self.root / f'handoff{len(self.policies)}_{len(list(self.root.glob("handoff*")))}'
        def fake_replay_init(obj, episode, out_dir, **kwargs):
            obj.out = Path(out_dir)
            obj.out.mkdir()
            obj.episode, obj.step, obj.episode_id = 0, 0, episode
            obj.proprio = np.stack([np.full(61, t / 100, np.float32) for t in range(128)])
            obj._demo_state = obj.proprio.copy()
            obj.actions = np.full((128, 23), -.75, np.float32)
            obj.action_path = obj.out / 'fake_demo_action.npy'
            np.save(obj.action_path, obj.actions)
            obj.records = []
        def factory(checkpoint, **kw):
            self.assertEqual(kw['device'], 'cpu')
            self.assertEqual(kw['samples'], p.samples)
            self.assertEqual(kw['exec_horizon'], 8)
            return p
        with patch.object(replay, 'ROOT', self.root), patch.object(replay, 'CONTRACT_TARGETS', targets), \
             patch.object(replay.ReplayPolicy, '__init__', fake_replay_init), \
             patch.object(replay.ReplayPolicy, 'emit', lambda obj, rec: obj.records.append(rec)):
            h = replay.HandoffPolicy(0, out, N, 'fake://checkpoint', wam_device='cpu', wam_samples=p.samples,
                                      wam_num_steps=2, exec_horizon=8, fixed_noise_seed=seed, wam_factory=factory,
                                      contract=contract, skill='pick up from', base_control='wam_whole_body')
        self.addCleanup(h.chunk_log.close)
        # Keep log-only base callbacks synthetic; all handoff/context/sampling code is real.
        h.emit = lambda rec: h.records.append(rec)
        return h

    def step(self, h, t, obs=None):
        obs = fixtures.observation(t) if obs is None else obs
        h.proprio[t] = obs['robot_r1::proprio']
        return h.next_action(t, obs)

    def test_first_handoff_reset_before_push_and_no_prefix_history(self):
        h = self.handoff()
        w, ctx = h.wam.wrapper, h.wam.model.context
        # Even a stale previous context must be discarded at the handoff boundary.
        ctx.push(np.full(61, -9, np.float32), np.full((3, 2, 4), -9, np.float32))
        for t in range(h.N):
            np.testing.assert_array_equal(self.step(h, t), h.actions[t])
        self.assertEqual(len(w.vision_encoder.calls), 0)  # prefix is replay, not a fake skill history
        self.assertEqual(ctx.steps, 1)  # injected stale context, not prefix frames
        first = self.step(h, h.N)
        self.assertEqual(first.shape, (23,))
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(ctx.steps, 1)
        state, pool = ctx.inputs()
        np.testing.assert_array_equal(state, np.full((7, 61), h.N / 100, np.float32))
        np.testing.assert_array_equal(pool, np.repeat(pool[:, :1], pool.shape[1], axis=1))
        self.assertEqual(len(w.vision_encoder.calls), 1)  # full-view pass only; no duplicate head consumer
        self.assertEqual(w.vision_encoder.calls[0].shape, (3, 3, 4, 4))
        rec = json.loads((h.out / 'handoff_record.json').read_text())
        self.assertEqual(rec['v3_context_steps'], 1)
        self.assertEqual(rec['v3_history_start_step'], h.N)
        self.assertFalse(rec['v3_prefix_history_used'])
        self.assertEqual(h.wam._proprio_frames, h.N + 1)  # old episode telemetry kept, NOT used by V3

    def test_queue_frames_reencode_and_training_history_matches(self):
        h = self.handoff()
        for t in range(h.N):
            self.step(h, t)
        states, pools = [], []
        w = h.wam.wrapper
        with patch.object(w, 'sample_actions', wraps=w.sample_actions) as sampler:
            for j in range(18):
                t = h.N + j
                self.step(h, t)
                ctx = h.wam.model.context
                state, pool = ctx.inputs()
                states.append(fixtures.observation(t)['robot_r1::proprio'])
                pools.append(pool[:, 0].copy())
                pi = [j] + history_frames(j, 0, [4, 8, 16, 32, 64])
                mi = [j] + history_frames(j, 0, [8, 32])
                np.testing.assert_array_equal(state, np.stack(states)[pi])
                np.testing.assert_array_equal(pool, np.stack(pools)[mi].transpose(1, 0, 2))
                self.assertEqual(ctx.steps, j + 1)
                self.assertEqual(len(w.vision_encoder.calls), j + 1)
            self.assertEqual(sampler.call_count, 3)  # exec8: replans at N, N+8, N+16; NOT 18 samples
        self.assertFalse(np.array_equal(pools[0], pools[1]), 'queued frame reused old pooled values')
        trace = [json.loads(x) for x in Path(h.wam.trace.name).read_text().splitlines()]
        rows = [x for x in trace if x['event'] == 'v3_context']
        self.assertEqual([x['step'] for x in rows], list(range(h.N, h.N + 18)))
        self.assertEqual(sum(x['reset_reason'] is not None for x in rows), 1)
        self.assertTrue(all(x['fresh_pooled'] and x['memory_layer'] == -4 for x in rows))
        chunks = [json.loads(x) for x in (h.out / 'chunks.jsonl').read_text().splitlines()]
        self.assertEqual([x['v3_context_steps'] for x in chunks], [1, 9, 17])
        self.assertTrue(all(x['samples'] == 1 and x['exec_horizon'] == 8 for x in chunks))

    def test_fixed_noise_seed_uses_same_generator_each_replan(self):
        h = self.handoff(seed=23)
        seen = []
        real = h.wam.wrapper.sample_actions
        def sample(*args, **kw):
            self.assertIs(kw['generator'], h.wam.gen)
            seen.append(kw['generator'].get_state().clone())
            return real(*args, **kw)
        with patch.object(h.wam.wrapper, 'sample_actions', side_effect=sample):
            for t in range(h.N + 10):
                self.step(h, t)
        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.equal(seen[0], seen[1]))
        self.assertTrue(torch.equal(seen[0], torch.Generator().manual_seed(23).get_state()))

    def test_reset_error_and_gap_do_not_reuse_context_or_queue(self):
        h = self.handoff()
        self.step(h, h.N)
        self.step(h, h.N + 1)
        bad = fixtures.observation(h.N + 2)
        del bad['robot_r1::robot_r1:right_realsense_link:Camera:0::rgb']
        with self.assertRaises(KeyError):
            self.step(h, h.N + 2, bad)
        self.assertEqual(h.wam.model.context.steps, 0)
        self.assertFalse(h.queue)
        self.step(h, h.N + 3)
        self.assertEqual(h.wam.model.context.steps, 1)
        self.step(h, h.N + 7)  # missing frames: fail-safe re-anchor, never invent a contiguous history
        self.assertEqual(h.wam.model.context.steps, 1)
        h.reset()  # real reset methods; synthetic base.step remains 0 so there is no file rotation
        self.assertEqual(h.wam.model.context.steps, 0)
        self.step(h, h.N, fixtures.observation(50))
        np.testing.assert_array_equal(h.wam.model.context.inputs()[0], np.full((7, 61), .5, np.float32))

    def test_v2_head_and_multiview_keep_prefix_history_and_queue_cadence(self):
        class LegacyAction(torch.nn.Module):
            action_len = 32
            def __init__(self):
                super().__init__()
                self.seen = []
            def forward(self, t, noisy_actions, qpos_history, dino_features_list, task_cond):
                self.seen.append(qpos_history.clone())
                return {'final_pred': noisy_actions * .01}
        class LegacyWrapper(torch.nn.Module):
            def __init__(self, encoder, views):
                super().__init__()
                self.vision_encoder, self.action_model = encoder, LegacyAction()
                self.views = views
                self.device, self.encoder_dtype = torch.device('cpu'), torch.bfloat16
                self.feat_layers, self.include_cls_register, self.num_register_tokens = [-12, -8, -4], True, 2
                self.action_min = torch.zeros(23)
                self.proprio_history = {'lags': [0, 2, 4]}
                self.action_repr_name, self.proprio_mask_name = 'absolute', 'positions_no_gripper'
            def normalize_state(self, s): return s
            def denormalize_action(self, x): return x
            def to_absolute_action(self, a, s): return a
            def get_vision_features(self, px):
                out = self.vision_encoder(pixel_values=px.reshape(-1, *px.shape[-3:]))
                return [out.hidden_states[l].reshape(1, len(self.views), 9, 4) for l in self.feat_layers]
        for views in ([ps.HEAD_CAMERA], fixtures.VIEWS):
            p = self.policy()
            cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
            del cfg.model.v3
            cfg.model.views, cfg.model.proprio_lags, cfg.common.proprio_len = views, [0, 2, 4], 3
            w = LegacyWrapper(p.wrapper.vision_encoder, views)
            p.model = ps.PolicyModel('main', 'fake://v2', 1, cfg, 'skill_proto', w)
            p.wrapper, p.cfg = w, cfg
            p.samples = 2  # V2 retains averaging support; do not impose the V3 K1 restriction
            h = self.handoff(p)
            with patch.object(p, '_record_v3_frame', side_effect=AssertionError('V2 entered V3 path')):
                for t in range(h.N + 10):
                    self.step(h, t)
            self.assertEqual(len(w.vision_encoder.calls), 2)  # V2 vision remains chunk-time only
            self.assertEqual(len(w.action_model.seen), 8)  # 2 chunks * 2 samples * 2 Euler steps
            np.testing.assert_array_equal(w.action_model.seen[0][0].numpy(),
                                          np.stack([np.full(61, t / 100, np.float32) for t in (5, 3, 1)]))
            self.assertNotIn('v3_context_steps', h.handoff_record)


if __name__ == '__main__':
    torch.set_num_threads(2)
    result = unittest.main(exit=False, verbosity=2).result
    assert not torch.cuda.is_initialized(), 'CPU handoff suite initialized CUDA'
    if not result.wasSuccessful():
        sys.exit(1)
    print('PASS: CPU fake V3 handoff + V2 compatibility; live parameters samples=1, exec_horizon=8; no GPU.')
