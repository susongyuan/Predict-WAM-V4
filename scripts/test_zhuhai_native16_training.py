#!/usr/bin/env python3
"""CPU-only native16 integration/regression tests; never launch Isaac or CUDA.

Run: CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .venvs/wam-fixed/bin/python -B scripts/test_zhuhai_native16_training.py
Synthetic observations are explicitly test fixtures, not successful robot trials.
All temporary HDF/checkpoint/sensor files live under TemporaryDirectory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from harness import zhuhai_training as zt
from harness import zhuhai_robot_spec as spec
from harness.zhuhai_dataset import HybridTaskDataset, PREPROCESS_SHA256
from harness.zhuhai_demo_recording import StreamDemoRecorder
from harness.wam_v3 import create_v3_action_model, V3Context, history_frames
from harness.robotwin_multitask_training import install_joint_projection
from scripts import train_zhuhai_wam as trainer, eval_zhuhai_wam as evaluator


class SyntheticFeatures:
    """Cheap deterministic bf16 feature views; no encoder inference/downloads."""
    def __init__(self):
        self.calls = []

    def get(self, ep, frame):
        self.calls.append((ep.row['split'], str(ep.path), frame))
        scale = np.float32((frame + 1) / 40)
        pattern = np.sin(np.arange(1024, dtype=np.float32) / (17 + ep.row['seed'] % 5))
        bits = torch.from_numpy(pattern * scale).bfloat16().view(torch.int16).numpy().view(np.uint16)
        features = np.broadcast_to(bits, (3, 3, 201, 1024))
        pool = np.arange(3 * 2048, dtype=np.float32).reshape(3, 2048) / 6000 * scale
        return features, pool


def observation(t):
    command = np.asarray(spec.HOME_ACTION, np.float32)
    command[0] += .001 * t
    command[8] -= .001 * t
    command[[7, 15]] = [(t % 7) / 6, (t % 5) / 4]
    measured = command.copy()
    measured[0] -= .05
    measured[8] += .025
    measured[7] = 1.002  # Preserve real feedback overshoot; never use as target.
    rgb = {view: np.full((8, 12, 3), (t * 13 + i) % 256, np.uint8)
           for i, view in enumerate(evaluator.VIEWS)}
    return dict(rgb=rgb, measured_joint_gripper_state=measured, commanded_joint_gripper_state=command)


def synthetic_row(directory, name, split, seed, frames=35):
    path = directory / (name + '.hdf5')
    collection = dict(**zt.protocol_binding(), state_dim=16, action_dim=16,
        state_semantics='measured_joint_gripper', action_semantics='joint_position_target',
        state_contract=spec.STATE_CONTRACT, action_contract=spec.ACTION_CONTRACT,
        frequency_hz=10., camera_count=3, record_video=False, third_view=False)
    recorder = StreamDemoRecorder(path, metadata=dict(task='zhuhai_task1', synthetic_cpu_fixture=True, seed=seed, family_id=name,
                                   contract=collection), frequency=10.)
    try:
        for t in range(frames + 1):
            observed = observation(t)
            if split != 'train':
                observed['measured_joint_gripper_state'][0] += 10.  # Excluded VAL/heldout outlier.
            evidence = dict(synthetic_cpu_fixture=True, frame=t, physics_time_seconds=t/10.,
                physics_safety=dict(passed=True, events=[], checks=['CPU synthetic fixture only; no physical validation']),
                score=dict(base_score=int(t == frames), full_task_success=(t == frames)))
            recorder.capture(observed, t / 10., evidence)
        recorder.finish(final_evidence=evidence, successful=True)
    except BaseException:
        recorder.abort()
        raise
    return dict(path=str(path), sha256=zt.sha256_file(path), split=split,
                family_id=name, seed=seed, **zt.protocol_binding())


class Native16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261009)
        cls.tmp = tempfile.TemporaryDirectory(prefix='zhuhai-native16-cpu-tests-')
        cls.directory = Path(cls.tmp.name)
        cls.frozen, cls.initialization_receipt = zt.checkpoint_header()
        cls.full_cfg = zt.config_for(cls.frozen)
        cls.cfg = copy.deepcopy(cls.full_cfg)
        cls.cfg['model']['action_expert'].update(hidden_size=32, depth=1, num_heads=4,
                                                adapter_depth=1, vlm_adapter_num_queries=4)
        cls.cfg['model']['future_feat'].update(num_decoder_layers=1, num_heads=4)
        cls.cfg['model']['v3']['phase_hidden'] = 16
        cls.store = SyntheticFeatures()
        cls.rows = {split: [synthetic_row(cls.directory, split, split, 101 + i)]
                    for i, split in enumerate(('train', 'val', 'heldout'))}
        cls.splits = {'zhuhai_task1': cls.rows}
        cls.datasets = zt.build_datasets(cls.splits, cls.store, cls.cfg)
        cls.features = dict(encoder_sha256='a' * 64, preprocess_sha256=PREPROCESS_SHA256)
        cls.statistics = zt.fit_train_statistics(cls.datasets['train'], cls.cfg, feature_identity=cls.features)
        cls.bank = zt.install_conditions(cls.datasets, cls.cfg, cls.statistics)
        source_cfg = copy.deepcopy(cls.cfg)
        source_cfg['common'].update(action_dim=14, state_dim=14)
        with torch.device('cpu'):
            cls.source = create_v3_action_model(OmegaConf.create(source_cfg), dino_hidden_size=1024,
                num_dino_layers=3, task_cond_dim=1024, patch_size=16)
            install_joint_projection(cls.source)
        with torch.no_grad():
            cls.source.action_proj.bias.fill_(7.)
            cls.source.proprio_proj.bias.fill_(8.)
            cls.source.task_cond_proj.residual.weight.normal_(std=.01)
        cls.initialization = dict(format=zt.INITIALIZATION_FORMAT, step=50000, model_only=True,
                                  model_state_dict=cls.source.state_dict())
        cls.wrapper, cls.notes = zt.build_wrapper(cls.cfg, cls.statistics, initialization=cls.initialization)
        cls.contract = dict(format=zt.FORMAT, config=cls.cfg, statistics=cls.statistics,
            statistics_sha256=zt.digest(cls.statistics), condition_bank=cls.bank, conditions=cls.bank['conditions'],
            time=zt.time_contract(cls.cfg), splits=cls.splits, cache=cls.features, **zt.protocol_binding())
        cls.protected = {p: zt.sha256_file(ROOT / p) for p in (
            'harness/wam_v3.py', 'harness/robotwin_multitask_training.py', 'harness/robotwin_policy_worker.py',
            'harness/robotwin_vtt_adapter.py', 'harness/robotwin_training_data.py')}

    @classmethod
    def tearDownClass(cls):
        for name, expected in cls.protected.items():
            assert zt.sha256_file(ROOT / name) == expected, name
        assert not torch.cuda.is_initialized(), 'CUDA must remain uninitialized'
        cls.tmp.cleanup()

    def test_default_500_and_unique_native16_protocol(self):
        args = trainer.parser().parse_args(['--tasks', 'not-read.json', '--out', 'not-created'])
        self.assertEqual(args.steps, 500)
        self.assertEqual(self.full_cfg['training']['max_steps'], 500)
        self.assertIn('native16', args.cache_root)
        self.assertEqual(args.source_count, 100)
        self.assertEqual(args.randomized_count, 500)
        self.assertEqual(self.cfg['common']['state_dim'], 16)
        self.assertEqual(self.cfg['common']['action_dim'], 16)
        self.assertEqual(self.cfg['model']['proprio_mask'], 'native_all16')

    def test_hdf_measured_and_next_interval_target_alignment(self):
        ep = self.datasets['train']['zhuhai_task1'].episodes[0]
        np.testing.assert_array_equal(ep.state[0], observation(0)['measured_joint_gripper_state'])
        np.testing.assert_array_equal(ep.action[0], observation(1)['commanded_joint_gripper_state'])
        self.assertEqual(ep.observation_count, len(ep.state) + 1)
        self.assertNotEqual(ep.state[0, 0], ep.action[0, 0])
        self.assertGreater(ep.state[0, 7], 1.)
        self.assertTrue(np.all(ep.action[:, [7, 15]] <= 1.))

    def test_statistics_are_new_native16_train_only(self):
        p = self.statistics['normalization']['provenance']
        self.assertTrue(all(row['split'] == 'train' for row in p['records']))
        self.assertTrue(all(split == 'train' for split, _, _ in self.store.calls))
        ep = self.datasets['train']['zhuhai_task1'].episodes[0]
        norm = self.statistics['normalization'][zt.NORMALIZATION_KEY]
        np.testing.assert_allclose(norm['state']['mean'], ep.state.mean(0, dtype=np.float64))
        np.testing.assert_allclose(norm['action']['mean'], ep.action.mean(0, dtype=np.float64))
        self.assertNotIn('robotwin', self.statistics['normalization'])
        self.assertEqual(np.asarray(self.statistics['temporal_stats']['proprio_diff_std']).shape, (6, 16))
        self.assertEqual(np.asarray(self.statistics['temporal_stats']['memory_diff_std']).shape, (3, 3, 2048))
        self.assertEqual(self.statistics['proprio_keep_mask'], [1.] * 16)

    def test_val_or_heldout_stats_and_conditions_rejected(self):
        for split in ('val', 'heldout'):
            with self.assertRaises(ValueError):
                zt.fit_train_statistics(self.datasets[split], self.cfg, feature_identity=self.features)
            with self.assertRaises(ValueError):
                zt.train_provenance(self.datasets[split], self.cfg)
        expected = self.bank['conditions']['zhuhai_task1']
        for split in ('val', 'heldout'):
            np.testing.assert_array_equal(self.datasets[split]['zhuhai_task1'].condition, expected)

    def test_legacy_norm_mask_and_foreign_robot_rejected(self):
        for mutate in (
            lambda x: x['normalization'].__setitem__('robotwin', x['normalization'].pop(zt.NORMALIZATION_KEY)),
            lambda x: x.__setitem__('proprio_keep_mask', [1.] * 14),
            lambda x: x['robot_contract'].__setitem__('profile', 'aloha'),
            lambda x: x['normalization'][zt.NORMALIZATION_KEY]['state'].__setitem__('min', [0.] * 14),
            lambda x: x['normalization']['provenance'].__setitem__('fit_split', 'heldout'),
        ):
            value = copy.deepcopy(self.statistics)
            mutate(value)
            with self.assertRaises((ValueError, KeyError)):
                zt.validate_statistics(value, self.cfg)

    def test_condition_bank_binds_stats_robot_and_train_sources(self):
        for key, value in (('statistics_sha256', '0' * 64), ('source_split', 'heldout'),
                           ('robot_contract_sha256', 'f' * 64)):
            bank = copy.deepcopy(self.bank)
            bank[key] = value
            with self.assertRaises(ValueError):
                zt.validate_conditions(bank, self.statistics, self.cfg)
        bank = copy.deepcopy(self.bank)
        bank['conditions']['zhuhai_task1'] = [1.] * 1024
        with self.assertRaises(ValueError):
            zt.validate_conditions(bank, self.statistics, self.cfg)

    def test_exact_selective_load_only_six_projection_keys_reset(self):
        reset = {f'{name}.{part}' for name in zt.RESET_PROJECTIONS for part in ('weight', 'bias')}
        self.assertEqual(set(self.notes['reset']), reset)
        self.assertTrue(self.notes['residual_VTT_loaded'])
        self.assertFalse(self.notes['numerical_normalization_reused'])
        sd = self.wrapper.action_model.state_dict()
        for key, tensor in self.source.state_dict().items():
            if key not in reset:
                torch.testing.assert_close(sd[key], tensor, rtol=0, atol=0)
        self.assertEqual(tuple(sd['action_proj.weight'].shape), (32, 16))
        self.assertEqual(tuple(sd['proprio_proj.weight'].shape), (32, 16))
        self.assertEqual(tuple(sd['output_proj.weight'].shape), (16, 32))
        self.assertFalse(torch.all(sd['action_proj.bias'] == 7.))
        self.assertFalse(torch.all(sd['proprio_proj.bias'] == 8.))
        self.assertTrue(all(not p.requires_grad for p in self.wrapper.action_model.task_cond_proj.base.parameters()))
        self.assertTrue(self.wrapper.action_model.task_cond_proj.residual.weight.requires_grad)

    def test_non_projection_missing_or_wrong_shape_fails_closed(self):
        for mutation in ('missing', 'shape', 'extra'):
            ck = dict(self.initialization, model_state_dict=dict(self.initialization['model_state_dict']))
            if mutation == 'missing':
                del ck['model_state_dict']['task_cond_proj.residual.weight']
            elif mutation == 'shape':
                ck['model_state_dict']['gate_head.weight'] = torch.zeros(1)
            else:
                ck['model_state_dict']['unexpected_tensor'] = torch.zeros(1)
            with self.assertRaises(RuntimeError):
                zt.build_wrapper(self.cfg, self.statistics, initialization=ck, device='meta')

    def test_synthetic_native16_loss_backward_and_sampling(self):
        ds = self.datasets['train']['zhuhai_task1']
        samples = [ds[0], ds[len(ds) - 1]]
        for sample in samples:
            sample['feats'] = sample['feats'].copy()
            sample['future_valid_mask'] = bool(sample['future_valid_mask'])
        batch = default_collate(samples)
        self.assertEqual(tuple(batch['action_sequence'].shape), (2, 32, 16))
        self.wrapper.train(); self.wrapper.zero_grad(set_to_none=True)
        loss, info, _ = self.wrapper(batch, self.cfg['training']['auxiliary_loss'])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        projection = self.wrapper.action_model.task_cond_proj
        self.assertGreater(float(projection.residual.weight.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in projection.base.parameters()))
        self.assertTrue(all(info[k] == 0. for k in ('gate_kl', 'phase_ce', 'log_dist')))
        self.wrapper.eval()
        features = batch['feats'].view(torch.int16)
        actions, _ = self.wrapper.sample_actions([features[:, :, i] for i in range(3)],
            batch['state'], batch['pool'], batch['task_cond'], num_steps=2,
            generator=torch.Generator(device='cpu').manual_seed(17))
        self.assertEqual(tuple(actions.shape), (2, 32, 16))
        self.assertTrue(torch.isfinite(actions).all())

    def test_full_50k_checkpoint_meta_migration(self):
        wrapper, notes = zt.build_wrapper(self.full_cfg, self.statistics, device='meta', initialization=self.frozen)
        self.assertEqual(len(notes['loaded']), 319)
        self.assertEqual(sum(p.numel() for p in wrapper.action_model.parameters()), 222700056)
        self.assertEqual(sum(p.numel() for p in wrapper.action_model.parameters() if p.requires_grad), 221320216)
        self.assertEqual(tuple(wrapper.prop_std.shape), (6, 16))
        self.assertEqual(tuple(wrapper.proprio_keep.shape), (16,))
        self.assertEqual(self.initialization_receipt['sha256'], zt.INITIALIZATION_SHA)
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpoint_save_load_and_foreign_protocol_rejection(self):
        out = self.directory / 'checkpoint_roundtrip'
        out.mkdir(exist_ok=True)
        trainer.save_checkpoint(out, self.wrapper, self.contract, 1)
        wrapper, condition, contract = zt.load_policy(out / 'checkpoint_step_000001.pt', 'zhuhai_task1', device='cpu')
        self.assertEqual(contract, self.contract)
        self.assertEqual(wrapper.normalization_key, zt.NORMALIZATION_KEY)
        self.assertEqual(wrapper.state_dim, 16)
        np.testing.assert_array_equal(condition, self.bank['conditions']['zhuhai_task1'])
        old = dict(self.contract, format=zt.INITIALIZATION_FORMAT)
        with self.assertRaises(ValueError):
            zt.validate_training_contract(old)
        with self.assertRaises(ValueError):
            trainer.save_checkpoint(out, self.wrapper, old, 2)

    def test_observation_ipc_keeps_measured_and_command_distinct(self):
        obs = observation(3)
        wire = evaluator.encode_observation(obs)
        decoded = evaluator.decode_observation(wire)
        for key in evaluator.OBSERVATION_FIELDS[1:]:
            np.testing.assert_array_equal(decoded[key], obs[key])
        self.assertFalse(np.array_equal(decoded['measured_joint_gripper_state'], decoded['commanded_joint_gripper_state']))
        bad = dict(obs, phase=2)
        with self.assertRaises(evaluator.InfrastructureError):
            evaluator.encode_observation(bad)
        for key in evaluator.OBSERVATION_FIELDS[1:]:
            bad = dict(obs); bad[key] = np.zeros(14)
            with self.assertRaises(evaluator.InfrastructureError):
                evaluator.encode_observation(bad)
        del wire['measured_joint_gripper_state']
        with self.assertRaises(evaluator.InfrastructureError):
            evaluator.decode_observation(wire)

    def test_runtime_api_and_measured_only_context(self):
        obs = observation(4)
        env = SimpleNamespace(policy_observation=lambda: obs,
            actual_state=lambda: obs['measured_joint_gripper_state'], get_action=lambda: obs['commanded_joint_gripper_state'])
        actual = evaluator.policy_observation(env)
        np.testing.assert_array_equal(actual['measured_joint_gripper_state'], env.actual_state())
        ctx = V3Context([2,4,8,16,32], [4,16], evaluator.MODEL_VIEWS, state_dim=16)
        states, pools = [], []
        for t in range(70):
            states.append(observation(t)['measured_joint_gripper_state'])
            pools.append(np.full((3,2048), t, np.float32))
            ctx.push(states[-1], pools[-1])
            state, pool = ctx.inputs()
            np.testing.assert_array_equal(state, np.asarray(states)[[t] + history_frames(t,0,[2,4,8,16,32])])
        self.assertEqual(state.shape, (7,16))

    def test_spec_bounds_and_gripper_columns(self):
        bounded = evaluator.bound_action(spec.HOME_ACTION, spec.DUAL_JOINT_LIMITS)
        self.assertFalse(any(bounded['clip_mask']))
        raw = np.asarray(spec.HOME_ACTION).copy(); raw[7] = 1.1; raw[15] = -.1
        bounded = evaluator.bound_action(raw, spec.DUAL_JOINT_LIMITS)
        self.assertEqual(np.flatnonzero(bounded['clip_mask']).tolist(), [7,15])
        with self.assertRaises(evaluator.InfrastructureError):
            evaluator.bound_action(np.zeros(14), spec.DUAL_JOINT_LIMITS)

    def test_gate_fail_closed_and_robot_identity(self):
        passed = dict(passed=True, robot_contract_sha256=zt.protocol_binding()['robot_contract_sha256'])
        self.assertEqual(evaluator.validate_runtime_gates({'mechanical':passed, 'camera':passed})['camera'], passed)
        for invalid in (None, {'passed':False}, {'passed':True}, {'passed':True,'robot_contract_sha256':'0'*64}):
            with self.assertRaises(evaluator.InfrastructureError):
                evaluator.validate_runtime_gates({'mechanical':invalid, 'camera':passed})

    def test_onboard_archive_is_measured_native16_not_video(self):
        out = self.directory / 'onboard_roundtrip'; out.mkdir(exist_ok=True)
        recorder = evaluator.Native16OnboardRecorder(out, 'cpu_synthetic', 1)
        recorder.append(observation(0), 0, 0.)
        recorder.append(observation(1), 1, .1)
        recorder.close(complete=True)
        with np.load(out / 'onboard/observation_000000.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['measured_joint_gripper_state'], observation(0)['measured_joint_gripper_state'])
            self.assertEqual(archive['commanded_joint_gripper_state'].shape, (16,))
        self.assertFalse(recorder.manifest['record_video'])
        self.assertTrue(recorder.manifest['complete'])

    def test_immutable_output_and_nonempty_old_run_preserved(self):
        path = self.directory / 'keep.json'
        zt.immutable_json(path, {'preserved':True})
        with self.assertRaises(ValueError):
            zt.immutable_json(path, {'preserved':False})
        self.assertEqual(json.loads(path.read_text()), {'preserved':True})
        out = self.directory / 'old_run'; out.mkdir(exist_ok=True)
        marker = out / 'legacy.txt'; marker.write_text('original evidence')
        args = trainer.parser().parse_args(['--tasks','not-read.json','--out',str(out)])
        with self.assertRaises(FileExistsError):
            trainer.train(args)
        self.assertEqual(marker.read_text(), 'original evidence')
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_contract_rejects_cache_and_train_record_drift(self):
        changed = copy.deepcopy(self.contract)
        changed['cache']['encoder_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            zt.validate_training_contract(changed)
        changed = copy.deepcopy(self.contract)
        changed['condition_bank']['provenance']['records'][0]['sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            zt.validate_training_contract(changed)
        changed = copy.deepcopy(self.contract)
        changed['splits']['zhuhai_task1']['train'][0]['seed'] += 1
        with self.assertRaises(ValueError):
            zt.validate_training_contract(changed)

    def test_hdf_robot_contract_cannot_be_relabelled_by_manifest(self):
        import h5py
        import shutil
        path = self.directory / 'wrong_robot.hdf5'
        shutil.copyfile(self.rows['train'][0]['path'], path)
        with h5py.File(path, 'r+') as handle:
            metadata = json.loads(handle.attrs['metadata'])
            metadata['contract']['robot_contract']['profile'] = 'another_native16_robot'
            handle.attrs['metadata'] = json.dumps(metadata)
        row = dict(self.rows['train'][0], path=str(path), sha256=zt.sha256_file(path))
        with self.assertRaises(ValueError):
            zt.build_datasets({'zhuhai_task1': {'train':[row], 'val':[], 'heldout':[]}}, self.store, self.cfg)

    def test_actual_main_runtime_gate_fields_and_unsafe_boundary(self):
        safety = dict(passed=True, events=[], checks=['synthetic-only contact monitor fixture'])
        env = SimpleNamespace(robot_contract=zt.robot_contract(), physics_safety=lambda: safety,
            camera_evidence=dict(all_geometry_in_frame=True, all_named_entities_visible=True))
        gates = evaluator.runtime_gate_results(env, observation(0))
        self.assertTrue(evaluator.validate_runtime_gates(gates)['camera']['passed'])
        evaluator.validate_physics_safety({'physics_safety':safety})
        failed = {'physics_safety':dict(passed=False, events=[{'reason':'synthetic collision'}], checks=safety['checks'])}
        with self.assertRaises(evaluator.InfrastructureError):
            evaluator.validate_physics_safety(failed)
        env.camera_evidence['all_named_entities_visible'] = False
        with self.assertRaises(evaluator.InfrastructureError):
            evaluator.validate_runtime_gates(evaluator.runtime_gate_results(env, observation(0)))
        env.robot_contract = dict(zt.robot_contract(), profile='old_robot')
        with self.assertRaises(evaluator.InfrastructureError):
            evaluator.runtime_gate_results(env, observation(0))

    def test_off_cadence_sensor_evidence_is_retained(self):
        out = self.directory / 'bad_clock'; out.mkdir()
        recorder = evaluator.Native16OnboardRecorder(out, 'synthetic_bad_clock', 3)
        recorder.append(observation(0), 0, 0.)
        with self.assertRaises(ValueError):
            recorder.append(observation(1), 1, .2)
        recorder.close(complete=False)
        self.assertTrue((out / 'onboard/observation_000001.npz').is_file())
        self.assertEqual(recorder.manifest['frames'], 2)
        self.assertFalse(recorder.manifest['complete'])

    def test_real_smoothing_interface_uses_measured_history(self):
        from harness.zhuhai_action_smoothing import protocol, SmoothedChunkPolicy
        settings = zt.validate_postprocessing(protocol())
        context = V3Context([2,4,8,16,32], [4,16], evaluator.MODEL_VIEWS, state_dim=16)
        seen = []
        def sample(features, inputs, seed, step):
            seen.append(inputs[0].copy())
            return np.tile(np.asarray(spec.HOME_ACTION), (32,1))
        policy = SmoothedChunkPolicy(context, lambda rgb: (None,np.zeros((3,2048),np.float32)), sample, settings)
        policy.reset('synthetic_cpu', 1)
        reply = policy.observe('synthetic_cpu', 0, observation(0))
        np.testing.assert_array_equal(seen[0][0], observation(0)['measured_joint_gripper_state'])
        self.assertEqual(len(reply['limited_action']), 16)
        self.assertEqual(reply['execution_prefix'], 8)

    def test_source_evidence_and_full_cpu_plan(self):
        directory = self.directory / 'cpu_plan'; directory.mkdir()
        scene = directory / 'synthetic_scene.txt'; scene.write_text('CPU fixture, not a simulator scene')
        safety = dict(passed=True, events=[], checks=['synthetic CPU source-evidence fixture'])
        attempts = []
        for split, rows in self.rows.items():
            row = rows[0]
            attempts.append(dict(status='success', data_path=row['path'], data_sha256=row['sha256'],
                family_id=row['family_id'], seed=row['seed'], plan_success=True, strict_predicate=True,
                physics_safety_all_observations_passed=True, observation_count=36,
                accepted_physics_times=[t/10 for t in range(36)],
                initial_rule_evidence=dict(physics_time_seconds=0., physics_safety=safety,
                    score=dict(base_score=0, full_task_success=False)),
                final_rule_evidence=dict(physics_time_seconds=3.5, physics_safety=safety,
                    score=dict(base_score=1, full_task_success=True)),
                initial_head_camera_evidence=dict(all_geometry_in_frame=True, all_named_entities_visible=True),
                initial_camera_observations={v:dict(nonempty_rgb=True) for v in evaluator.VIEWS}))
        collection = dict(**zt.protocol_binding(), task='zhuhai_task1', track='life', mode='source',
            engine=dict(name='Isaac Sim', version='6.1.0.0'), frequency_hz=10., camera_count=3,
            record_video=False, third_view=False, physics_dt_seconds=.004,
            source_scene=dict(path=str(scene), sha256=zt.sha256_file(scene)))
        manifest = directory / 'source.json'
        manifest.write_text(json.dumps(dict(status='complete', contract=collection, attempts=attempts)))
        tasks = directory / 'tasks.json'
        tasks.write_text(json.dumps(dict(tasks=[dict(task='zhuhai_task1', source_manifest=str(manifest))])))
        splits, sources = zt.task_sources(tasks, allow_small=True)
        self.assertEqual(len(splits['zhuhai_task1']['train']), 1)
        self.assertEqual(sources['zhuhai_task1']['robot_contract'], zt.robot_contract())
        bad = dict(attempts[0], physics_safety_all_observations_passed=False)
        with self.assertRaises(ValueError):
            zt.validate_source_success(bad)
        args = trainer.parser().parse_args(['--tasks',str(tasks),'--out',str(directory/'out'),'--allow-small-data'])
        with mock.patch.object(zt, 'checkpoint_header', return_value=(self.frozen,self.initialization_receipt)), \
             mock.patch.object(trainer, 'verify_owner', side_effect=AssertionError('CPU plan must not request GPU')):
            self.assertEqual(trainer.train(args), 0)
        plan = json.loads((directory / 'out/plan.json').read_text())
        self.assertEqual(plan['contract']['config']['common']['action_dim'], 16)
        self.assertEqual(plan['contract']['config']['training']['max_steps'], 500)
        self.assertNotIn('statistics', plan['contract'])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
