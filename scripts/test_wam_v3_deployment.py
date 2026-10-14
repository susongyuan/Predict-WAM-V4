#!/usr/bin/env python3
"""CPU-only deployment contract tests. No checkpoints/models downloaded, no CUDA initialization.

  CUDA_VISIBLE_DEVICES='' .conda-envs/lilawam/bin/python scripts/test_wam_v3_deployment.py

Uses the REAL WamV3Wrapper / V3Context / policy loader with tiny fake encoder/action networks.
The existing policy_server V2 synthetic regression is included (unless --skip-v2 is supplied).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from harness import policy_server as ps
from harness.wam_v3 import WamV3Wrapper, history_frames, pool_from_hidden

VIEWS = [ps.HEAD_CAMERA, 'observation.rgb.left_realsense_link_camera_0',
         'observation.rgb.right_realsense_link_camera_0']


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, pixel_values, **kwargs):
        self.calls.append(pixel_values.detach().clone())
        base = pixel_values.float().mean((1, 2, 3))[:, None, None]
        # CLS, 2 register tokens, 6 patch tokens. Registers deliberately unlike patches.
        h = base + torch.arange(9).float()[None, :, None] + torch.arange(4).float()[None, None, :]
        hs = [h + i * 10 for i in range(25)]
        return NS(hidden_states=hs, last_hidden_state=hs[-1])


class Action(torch.nn.Module):
    action_len = 32
    proprio_len = 7
    num_memory = 3

    def __init__(self):
        super().__init__()
        self.mem_proj = torch.nn.Linear(8, 4)
        self.proprio_proj = torch.nn.Linear(61, 4)
        self.action_proj = torch.nn.Linear(23, 4)
        self.output_proj = torch.nn.Linear(4, 23)
        self.gate_head = torch.nn.Linear(4, 3)
        self.seen = []

    def forward(self, t, noisy_actions, qpos_history, dino_features_list, task_cond, memory):
        self.seen.append((qpos_history.detach().clone(), memory.detach().clone(),
                          [f.detach().clone() for f in dino_features_list]))
        B = noisy_actions.shape[0]
        # Includes noise and temporal signals; K=1 equality would detect hidden extra sampling/averaging.
        pred = .03 * noisy_actions + .001 * qpos_history.mean((1, 2))[:, None, None]
        pred = pred + .001 * memory.mean((1, 2, 3))[:, None, None]
        return {'final_pred': pred, 'gate': torch.tensor([[.2, .3, .5]]).expand(B, -1),
                'phase_logits': torch.tensor([[0., 1., 2., 3.]]).expand(B, -1), 'log_dist': torch.zeros(B)}


def observation(t):
    obs = {'robot_r1::proprio': np.full(61, t / 100, np.float32)}
    for v, link in enumerate(('zed_link', 'left_realsense_link', 'right_realsense_link')):
        rgb = np.full((6 + v, 6 + v, 4), (t + 60 * v) % 240, np.uint8)
        rgb[..., 3] = 255  # alpha must not enter features
        obs[f'robot_r1::robot_r1:{link}:Camera:0::rgb'] = rgb
    return obs


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        stats = {'views': VIEWS, 'proprio_lags': [4, 8, 16, 32, 64], 'memory_lags': [8, 32],
                 'proprio_diff_std': np.ones((6, 61)).tolist(), 'memory_diff_std': np.ones((3, 3, 8)).tolist()}
        (self.root / 'stats.json').write_text(json.dumps(stats))
        norms = {'action_repr': 'absolute', 'behavior1k': {
            'action': {'min': [-1.] * 23, 'max': [1.] * 23}, 'state': {'min': [-1.] * 61, 'max': [1.] * 61}}}
        (self.root / 'norm.json').write_text(json.dumps(norms))
        self.cfg = OmegaConf.create({'common': {'proprio_len': 7},
            'dataset': {'image_size': [4, 4], 'stats_path': str(self.root / 'norm.json'), 'cond_mode': 'skill_proto'},
            'model': {'views': VIEWS, 'use_task_cond': True, 'proprio_mask': 'positions_no_gripper',
                      'vision_encoder': {'checkpoint_path': str(self.root / 'fake_encoder'),
                                         'feat_layers': [-12, -8, -4], 'include_cls_register': True},
                      'v3': {'proprio_lags': [4, 8, 16, 32, 64], 'memory_lags': [8, 32],
                             'stats_path': str(self.root / 'stats.json')}}})
        self.encoders = []
        def create(path, dtype, device):
            self.assertEqual(torch.device(device).type, 'cpu')
            e = Encoder()
            self.encoders.append(e)
            return e, 4, 2, 2
        class Factory:
            create_vision_encoder = staticmethod(create)
        self.runner = NS(ModelFactory=Factory)
        for target, replacement in [('harness.wam_behavior.load_upstream', lambda: (None, self.runner)),
                                     ('harness.wam_v3.create_v3_action_model', lambda *a, **kw: Action())]:
            p = patch(target, replacement)
            p.start()
            self.addCleanup(p.stop)
        self.ck = self.root / 'v3.pt'
        self.save_checkpoint(Action().state_dict())
        self.policies = []

    def save_checkpoint(self, sd):
        torch.save({'config': OmegaConf.to_container(self.cfg), 'model_state_dict': sd, 'step': 42}, self.ck)

    def policy(self, **kwargs):
        p = ps.WamPolicy(str(self.ck), skill='move to', exec_horizon=8, num_steps=2, device='cpu',
                         cuda_fraction=.3, seed=7, log_dir=self.root / f'policy{len(self.policies)}', **kwargs)
        p.action_low[:] = -100
        p.action_high[:] = 100
        self.policies.append(p)
        self.addCleanup(p.trace.close)
        return p

    def attach_orch(self, p):
        class Orch:
            def __init__(self):
                self.index = 0
                self.events = []
                self.closure = NS(state=None)
                self.contract = NS(index=0, attempts=0, started_step=0, key='pick up from(obj,floor)',
                                   arm=NS(arm_id='train'), arm_key='arm', is_navigation=False,
                                   step=NS(skill='pick up from', objects=['obj', 'floor']))
                self.mode = 'EXECUTE'
                self.new_chunk = False
            def tick(self, *args, **kwargs):
                return NS(mode=self.mode, new_chunk_required=self.new_chunk, skill=self.contract.step.skill,
                          condition=np.zeros(1024, np.float32))
        p.orch = Orch()
        p._grounded = True
        p.base_control_mode = 'wam_whole_body'
        p._active_contract = None
        p._orch_finish_episode = lambda: None
        return p.orch

    def test_checkpoint_strict_and_shared_frozen_encoder(self):
        cache = {}
        a = ps.load_policy_model('a', self.ck, torch.device('cpu'), cache)
        b = ps.load_policy_model('b', self.ck, torch.device('cpu'), cache)
        self.assertIsInstance(a.wrapper, WamV3Wrapper)
        self.assertIs(a.wrapper.vision_encoder, b.wrapper.vision_encoder)
        self.assertIsNot(a.wrapper.action_model, b.wrapper.action_model)
        self.assertEqual(len(self.encoders), 1)
        self.assertFalse(a.wrapper.vision_encoder.training)
        self.assertFalse(any(p.requires_grad for p in a.wrapper.vision_encoder.parameters()))
        good = Action().state_dict()
        bads = [{k: v for k, v in good.items() if k != 'gate_head.weight'},
                {**good, 'unexpected': torch.zeros(1)}, {**good, 'gate_head.weight': torch.zeros(1, 1)}]
        for bad in bads:
            self.save_checkpoint(bad)
            with self.assertRaises(RuntimeError):
                ps.load_policy_model('bad', self.ck, torch.device('cpu'), cache)
        self.assertEqual(len(self.encoders), 1)

    def test_every_frame_history_pool_layer_views_and_bounded_time(self):
        p = self.policy()
        w = p.wrapper
        pools, states = [], []
        with patch.object(w, 'sample_actions', wraps=w.sample_actions) as sampler:
            for t in range(74):
                obs = observation(t)
                p.act(obs)
                states.append(obs['robot_r1::proprio'].copy())
                pixels = w.vision_encoder.calls[-1]
                self.assertEqual(pixels.shape, (3, 3, 4, 4))
                for v, suffix in enumerate(p.model.view_suffixes):
                    raw = ps.WamPolicy._find(obs, suffix)
                    expected = torch.from_numpy(ps.prep_rgb(raw, 4)).to(torch.bfloat16)
                    torch.testing.assert_close(pixels[v], expected, rtol=0, atol=0)
                base = pixels.float().mean((1, 2, 3))[:, None, None]
                h = base + torch.arange(9)[None, :, None] + torch.arange(4)[None, None, :] + 210
                pooled = pool_from_hidden(h, 2).numpy().reshape(3, 8)
                pools.append(pooled)
                history, memory = p.model.context.inputs()
                pi = [t] + history_frames(t, 0, [4, 8, 16, 32, 64])
                mi = [t] + history_frames(t, 0, [8, 32])
                np.testing.assert_array_equal(history, np.stack(states)[pi])
                np.testing.assert_array_equal(memory, np.stack(pools)[mi].transpose(1, 0, 2))
            self.assertEqual(sampler.call_count, 10)  # ceil(74/8), not 74 and not 8 samples/chunk
        self.assertEqual(len(w.vision_encoder.calls), 148)  # head + REAL full-view pass at EVERY frame
        self.assertEqual(p.model.context.steps, 74)
        self.assertEqual(p.model.context.maxlen, 65)
        self.assertLessEqual(len(p.model.context.states), 65)
        trace = [json.loads(x) for x in Path(p.trace.name).read_text().splitlines()]
        frames = [x for x in trace if x['event'] == 'v3_context']
        self.assertEqual([x['context_steps'] for x in frames], list(range(1, 75)))
        self.assertTrue(all(x['fresh_pooled'] and x['memory_layer'] == -4 and x['encode_ms'] >= 0 for x in frames))
        chunks = [x for x in trace if x['event'] == 'chunk']
        self.assertEqual(list(chunks[-1]['view_gate']), VIEWS)
        self.assertEqual(chunks[-1]['samples'], 1)
        self.assertEqual(chunks[-1]['exec_horizon'], 8)
        self.assertEqual(chunks[-1]['v3_context_steps'], 73)
        self.assertEqual(len(w.action_model.seen), 20)

    def test_k1_exact_shared_sampler_and_no_mean(self):
        p = self.policy()
        obs = observation(0)
        w = p.wrapper
        with patch.object(w, 'sample_actions', wraps=w.sample_actions) as sample, patch.object(
                ps.np, 'mean', side_effect=AssertionError('K=1 must not average')):
            first = p.act(obs)
            args, kw = sample.call_args
            expected, _ = w.sample_actions(*args, num_steps=2, generator=torch.Generator().manual_seed(7))
            np.testing.assert_array_equal(first.numpy(), expected[0, 0].numpy())
            for t in range(1, 8):
                actual = p.act(observation(t))
                np.testing.assert_array_equal(actual.numpy(), expected[0, t].numpy())
            self.assertEqual(sample.call_count, 2)  # one deployment call + explicit reference above
        with self.assertRaisesRegex(ValueError, 'samples=1'):
            p._sample(None, None, None, samples=2)
        with self.assertRaisesRegex(ValueError, 'samples=1'):
            self.policy(samples=2)
        self.assertEqual(p.metadata()['exec_horizon'], 8)

    def test_contract_retry_hold_resume_and_episode_reset(self):
        p = self.policy()
        orch = self.attach_orch(p)
        for t in range(3):
            p.act(observation(t))
        self.assertEqual(p.model.context.steps, 3)
        self.assertEqual(p.stats['chunks'], 1)
        # Change contract even if the orchestrator did not set new_chunk_required.
        orch.contract.key = 'press(obj)'
        p.act(observation(10))
        self.assertEqual(p.model.context.steps, 1)
        self.assertEqual(p.stats['chunks'], 2)
        np.testing.assert_array_equal(p.model.context.inputs()[0], np.full((7, 61), .10, np.float32))
        p.act(observation(11))
        orch.contract.attempts += 1
        p.act(observation(20))
        self.assertEqual(p.model.context.steps, 1)
        orch.mode = 'HOLD'
        p.act(observation(21))
        p.act(observation(22))
        self.assertEqual(p.model.context.steps, 2)
        self.assertFalse(p.queue)
        orch.mode = 'EXECUTE'
        p.act(observation(30))
        self.assertEqual(p.model.context.steps, 1)
        self.assertEqual(p.stats['chunks'], 4)
        # Recovering from an encoder/input failure must not reuse pre-error start snapshots or queued actions.
        bad = observation(31)
        del bad['robot_r1::robot_r1:left_realsense_link:Camera:0::rgb']
        with self.assertRaises(KeyError):
            p.act(bad)
        self.assertEqual(p.model.context.steps, 0)
        self.assertFalse(p.queue)
        frames = p._proprio_frames
        p.hold(observation(31))
        self.assertEqual(p._proprio_frames, frames)
        self.assertEqual(p.model.context.steps, 0)
        p.act(observation(40))
        self.assertEqual(p.model.context.steps, 1)
        p.reset()
        self.assertEqual(p.model.context.steps, 0)
        self.assertFalse(p.queue)
        p.act(observation(50))
        np.testing.assert_array_equal(p.model.context.inputs()[0], np.full((7, 61), .5, np.float32))

    def test_privileged_keys_never_reach_policy_consumers(self):
        p = self.policy()
        obs = observation(1)
        obs['robot_r1::cam_rel_poses'] = np.eye(4)
        obs['robot_r1::robot_r1:zed_link:Camera:0::depth_linear'] = np.ones((4, 4))
        privileged = {'object_poses': object(), 'reward': object(), 'phase': object(), 'active_hand': object(),
                      'bd_dl': object(), 'privileged::proprio': object(),
                      'privileged::robot_r1:zed_link:Camera:0::rgb': object()}
        seen = []
        original = p._preprocess
        def inspect(obs):
            seen.append(set(obs))
            self.assertTrue(set(obs).isdisjoint(privileged))
            return original(obs)
        with patch.object(p, '_preprocess', side_effect=inspect):
            p.act({**privileged, **obs})
        self.assertEqual(seen, [set(obs)])
        clean = self.policy()
        # Same seed and sensor values -> exact same stochastic first action regardless of privileged values.
        contaminated = self.policy()
        np.testing.assert_array_equal(clean.act(obs).numpy(), contaminated.act({**obs, **privileged}).numpy())

    def test_residency_moves_only_action_models(self):
        p = self.policy()
        p.manip = ps.load_policy_model('manip', self.ck, torch.device('cpu'), p._encoder_cache)
        p.offload_inactive = True
        encoder = p.wrapper.vision_encoder
        with patch.object(encoder, 'to', side_effect=AssertionError('shared encoder must never be offloaded')), \
             patch.object(p.wrapper.action_model, 'to', wraps=p.wrapper.action_model.to) as main_to, \
             patch.object(p.manip.wrapper.action_model, 'to', wraps=p.manip.wrapper.action_model.to) as manip_to:
            p._ensure_resident(p.manip)
            p._ensure_resident(p.model)
            self.assertIs(p._resident, p.model)
            self.assertEqual(main_to.call_count, 2)
            self.assertEqual(manip_to.call_count, 2)
        self.assertIs(encoder, p.manip.wrapper.vision_encoder)
        self.assertEqual(len(self.encoders), 1)


    def test_clarification_resume_clears_history_without_control_frame(self):
        p = self.policy()
        p.act(observation(1))
        frames, encoder_calls = p._proprio_frames, len(self.encoders[0].calls)
        p.orch = NS(status='HOLD', question_id='q1', pending_question=['go'],
                    answer=lambda answer, question_id: answer == 'go' and question_id == 'q1')
        reply = ps.control_message(p, {'clarification': {'episode': p.episode, 'answer': 'go',
                                                        'question_id': 'q1', 'source': 'human'}})
        self.assertTrue(reply['accepted'])
        self.assertEqual(p.model.context.steps, 0)
        self.assertFalse(p.queue)
        self.assertEqual(p._proprio_frames, frames)
        self.assertEqual(len(self.encoders[0].calls), encoder_calls)
        p.orch = None
        p.act(observation(10))
        self.assertEqual(p.model.context.steps, 1)
        np.testing.assert_array_equal(p.model.context.inputs()[0], np.full((7, 61), .1, np.float32))

    def test_residency_budget_and_stats_fail_before_extra_encoder(self):
        cache = ps._EncoderCache(1)
        ps.load_policy_model('main', self.ck, torch.device('cpu'), cache)
        self.cfg.model.vision_encoder.checkpoint_path = str(self.root / 'different_encoder')
        self.save_checkpoint(Action().state_dict())
        factory = self.runner.ModelFactory.__dict__['create_vision_encoder']
        with self.assertRaisesRegex(ValueError, 'residency budget'):
            ps.load_policy_model('other', self.ck, torch.device('cpu'), cache)
        self.assertEqual(len(self.encoders), 1)
        self.assertIs(self.runner.ModelFactory.__dict__['create_vision_encoder'], factory)
        ck = torch.load(self.ck, weights_only=False)
        ck['v3_stats'] = {'changed': 'training snapshot'}
        torch.save(ck, self.ck)
        with self.assertRaisesRegex(ValueError, 'training snapshot'):
            ps.load_policy_model('bad', self.ck, torch.device('cpu'), cache)
        self.assertEqual(len(self.encoders), 1)

    def test_two_v3_models_route_reset_and_default_load_budget(self):
        built = []
        class TrackedAction(Action):
            def __init__(self):
                super().__init__()
                self.moves = []
            def to(self, *args, **kwargs):
                self.moves.append(str(args[0]))
                return super().to(*args, **kwargs)
        def build(*args, **kwargs):
            if built:
                self.assertEqual(built[0].moves, ['cpu'], 'main must offload BEFORE constructing manip')
            built.append(TrackedAction())
            return built[-1]
        with patch('harness.wam_v3.create_v3_action_model', side_effect=build):
            p = self.policy(orchestrate={'manip_checkpoint': str(self.ck), 'vlm': 'none',
                                         'answer': 'none', 'precondition': False, 'frozen': True})
        self.assertTrue(p.offload_inactive)
        self.assertIs(p._resident, p.model)
        self.assertEqual(p.metadata()['residency_budget']['action_models'], 1)
        orch = self.attach_orch(p)
        orch.contract.is_navigation = True
        p.act(observation(0))
        p.act(observation(1))
        self.assertEqual(p.model.context.steps, 2)
        self.assertEqual(p.manip.context.steps, 0)
        orch.contract.is_navigation = False
        p.act(observation(20))
        self.assertEqual(p.model.context.steps, 0)
        self.assertEqual(p.manip.context.steps, 1)
        self.assertEqual(p._queue_model, 'manip')
        self.assertIs(p._resident, p.manip)
        p.act(observation(21))
        orch.contract.is_navigation = True
        p.act(observation(40))
        self.assertEqual(p.manip.context.steps, 0)
        self.assertEqual(p.model.context.steps, 1)
        np.testing.assert_array_equal(p.model.context.inputs()[0], np.full((7, 61), .4, np.float32))
        self.assertEqual(len(self.encoders), 1)

    def test_mixed_v2_main_v3_manip_routing(self):
        class LegacyAction(torch.nn.Module):
            action_len = 32
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(.01))
                self.calls = 0
            def forward(self, t, noisy_actions, qpos_history, dino_features_list, task_cond):
                self.calls += 1
                return {'final_pred': noisy_actions * self.scale}
        class LegacyWrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.vision_encoder = encoder
                self.action_model = LegacyAction()
                self.views = [ps.HEAD_CAMERA]
                self.device = torch.device('cpu')
                self.encoder_dtype = torch.bfloat16
                self.feat_layers = [-12, -8, -4]
                self.include_cls_register = True
                self.num_register_tokens = 2
                self.action_min = torch.zeros(23)
                self.action_repr_name = 'absolute'
                self.proprio_mask_name = 'positions_no_gripper'
            def normalize_state(self, s): return s
            def denormalize_action(self, x): return x
            def to_absolute_action(self, a, s): return a
        def build_v2(cfg, device):
            encoder, *_ = self.runner.ModelFactory.create_vision_encoder(
                str(ROOT / cfg.model.vision_encoder.checkpoint_path), torch.bfloat16, device)
            return LegacyWrapper(encoder), None
        v2cfg = OmegaConf.to_container(self.cfg)
        del v2cfg['model']['v3']
        v2cfg['model']['views'] = [ps.HEAD_CAMERA]
        v2cfg['common']['proprio_len'] = 1
        v2ck = self.root / 'v2.pt'
        torch.save({'config': v2cfg, 'model_state_dict': LegacyAction().state_dict()}, v2ck)
        with patch.object(ps, 'build_models', side_effect=build_v2):
            p = ps.WamPolicy(str(v2ck), skill='move to', exec_horizon=8, num_steps=2, device='cpu',
                             cuda_fraction=.3, seed=0, log_dir=self.root / 'mixed',
                             orchestrate={'manip_checkpoint': str(self.ck), 'vlm': 'none',
                                          'answer': 'none', 'precondition': False})
        self.addCleanup(p.trace.close)
        orch = self.attach_orch(p)
        orch.contract.is_navigation = True
        p.act(observation(0))
        self.assertFalse(p.model.is_v3)
        self.assertEqual(len(self.encoders[0].calls), 1)  # V2 navigation remains head-only
        orch.contract.is_navigation = False
        for t in (10, 11):
            p.act(observation(t))
        self.assertEqual(p._queue_model, 'manip')
        self.assertEqual(p.manip.context.steps, 2)
        self.assertEqual(p.wrapper.action_model.calls, 2)
        self.assertEqual(len(self.encoders[0].calls), 5)  # V3 queued frame still encodes wrists
        orch.contract.is_navigation = True
        p.act(observation(20))
        self.assertEqual(p.manip.context.steps, 0)
        self.assertEqual(p._queue_model, 'main')
        self.assertEqual(p.wrapper.action_model.calls, 4)
        self.assertEqual(len(self.encoders), 1)


if __name__ == '__main__':
    skip_v2 = '--skip-v2' in sys.argv
    if skip_v2:
        sys.argv.remove('--skip-v2')
    torch.set_num_threads(2)
    result = unittest.main(exit=False, verbosity=2).result
    if not result.wasSuccessful():
        sys.exit(1)
    if not skip_v2:
        print('\nExisting V2 synthetic regression:')
        if ps.task_profile.TASK_ID == 13:
            print(json.dumps(ps.synthetic_self_test(), indent=2))
        else:
            # The historical regression uses Task-13 skill names; isolate its task profile from Task-0/1 tests.
            import subprocess
            subprocess.run([sys.executable, '-m', 'harness.policy_server', '--synthetic-self-test'],
                           cwd=ROOT, env={**os.environ, 'P2_TASK_ID': '13', 'PYTHONDONTWRITEBYTECODE': '1'},
                           check=True)
    assert not torch.cuda.is_initialized(), 'CPU suite initialized CUDA'
    print('\nPASS: fake CPU deployment contracts; no GPU, downloads or simulator. Latency is NOT a GPU benchmark.')
