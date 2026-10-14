#!/usr/bin/env python3
"""Independent CPU/mock rejection checks; creates no physical V3 outcome."""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_zhuhai_wam_precision_v3_v42 as runtime
from scripts import precision_v3_evidence_v42 as evidence
from harness import zhuhai_observer_video_v42 as observer
from harness import zhuhai_official_braking as braking

OLD_RAW = ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_80k_extra20_20261011_v24/zhuhai_task2/seed_932106/result.json'
MOCK_SHA = 'a' * 64
REAL_STEP0 = ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_v3fork80k_30k_20261012/train/checkpoint_step_000000.pt'


def binding_fixture(seed=932127):
    """Copy the old structure in memory only; these are explicitly mocked fields."""
    raw = json.loads(OLD_RAW.read_text())
    rt = raw['policy_ready']['runtime']
    frozen = {k: deepcopy(rt[k]) for k in ('robot_contract', 'robot_contract_sha256',
              'data_contract', 'data_contract_sha256', 'action_postprocessing', 'action_postprocessing_sha256')}
    changed = evidence.changed_contract(frozen)
    rt.update(format=evidence.FORMAT, checkpoint_sha256=MOCK_SHA, checkpoint_step=30000,
              precision_v3_actual=deepcopy(evidence.ACTUAL_PRECISION_BINDING),
              action_postprocessing=changed['action_postprocessing'],
              action_postprocessing_sha256=changed['action_postprocessing_sha256'],
              training_contract_sha256=evidence.digest(frozen),
              evaluation_version=braking.VERSION,
              evaluation_source_sha256=evidence.sha256_file(ROOT / 'scripts/eval_zhuhai_wam_precision_v3_v42.py'),
              system_mode='precision_wam_v3_official_braking', dino_frozen=True,
              precision_refiner_used_online=True, event_clock_used_online=True,
              event_clock_is_verified_grasp=False, hjepa_auxiliary_heads_used_online=False)
    raw.update(seed=seed, record_video=seed == 932126, third_view=seed == 932126,
               observer_policy_input=False, action_postprocessing_sha256=changed['action_postprocessing_sha256'])
    for key in ('final_rule_evidence', 'last_rule_evidence'):
        if key in raw:
            raw[key]['video_recording'] = seed == 932126
            raw[key]['observer_video_only'] = seed == 932126
    return raw, frozen, changed


def rule_fixture(time=0., unsafe=False):
    row = deepcopy(json.loads(OLD_RAW.read_text())['final_rule_evidence'])
    row['physics_time_seconds'] = time
    row['video_recording'] = row['observer_video_only'] = False
    row['physics_safety']['checks'] = deepcopy(evidence.CHECKS)
    row['physics_safety']['scope'] = 'mock_contact_monitor_scope'
    row['physics_safety']['passed'] = not unsafe
    row['physics_safety']['events'] = ([dict(time_seconds=time,
        bodies=['/World/Robot/left/panda_link7', '/World/Task/Table'],
        reason='non_finger_table_collision')] if unsafe else [])
    return row


class NativeV3BindingTests(unittest.TestCase):
    def test_real_step0_reader_is_bound_to_v3_and_does_not_construct_model(self):
        self.assertIs(runtime.FROZEN_READ_CONTRACT, runtime.precision.read_contract)
        self.assertTrue(REAL_STEP0.is_file())
        with patch.object(runtime.precision, 'build_wrapper', side_effect=AssertionError('CPU contract reader must not construct model')), \
             patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU reader must not initialize CUDA')):
            contract, receipt = runtime.read_contract(REAL_STEP0)
        self.assertEqual(contract['format'], evidence.FORMAT)
        self.assertEqual(contract['precision_v3']['refine_steps'], 32)
        self.assertEqual(receipt['step'], 0)
        self.assertTrue(receipt['model_only'])
        self.assertEqual(receipt['sha256'], evidence.sha256_file(REAL_STEP0))
        self.assertEqual(contract['action_postprocessing'], braking.protocol())
        self.assertFalse(contract['evaluation_overlay']['checkpoint_modified'])
        self.assertEqual(contract['evaluation_overlay']['training_contract_sha256'],
                         evidence.read_json(REAL_STEP0.parent / 'manifest.json')['contract_sha256'])

    def test_valid_mock_binding_then_reject_v2(self):
        raw, _, changed = binding_fixture()
        evidence.validate_binding(raw, 30000, 932127, MOCK_SHA, changed)
        raw['policy_ready']['runtime']['format'] = 'zhuhai-precision-wam-native16-measured-10hz-v2'
        with self.assertRaises(evidence.EvidenceError):
            evidence.validate_binding(raw, 30000, 932127, MOCK_SHA, changed)

    def test_wrong_actual32_and_wrong_sha_rejected(self):
        for change in ('steps', 'sha'):
            raw, _, changed = binding_fixture()
            if change == 'steps':
                raw['policy_ready']['runtime']['precision_v3_actual']['refined_flow_steps'] = 8
            else:
                raw['policy_ready']['runtime']['checkpoint_sha256'] = 'b' * 64
            with self.assertRaises(evidence.EvidenceError):
                evidence.validate_binding(raw, 30000, 932127, MOCK_SHA, changed)

    def test_non126_recording_rejected(self):
        raw, _, changed = binding_fixture()
        raw['record_video'] = raw['third_view'] = True
        with self.assertRaises(evidence.EvidenceError):
            evidence.validate_binding(raw, 30000, 932127, MOCK_SHA, changed)

    def test_observer_policy_input_and_future_input_rejected(self):
        for field in ('observer_policy_input', 'future_inputs_used_online'):
            raw, _, changed = binding_fixture()
            if field == 'observer_policy_input': raw[field] = True
            else: raw['policy_ready']['runtime'][field] = True
            with self.assertRaises(evidence.EvidenceError):
                evidence.validate_binding(raw, 30000, 932127, MOCK_SHA, changed)

    def test_strict_runtime_mechanisms_and_mode_cannot_be_relabelled(self):
        for field in ('dino_frozen', 'precision_refiner_used_online', 'event_clock_used_online',
                      'hjepa_auxiliary_heads_used_online', 'event_clock_is_verified_grasp', 'system_mode'):
            raw, _, changed = binding_fixture()
            rt = raw['policy_ready']['runtime']
            rt[field] = 'precision_wam_v2' if field == 'system_mode' else not rt[field]
            with self.assertRaises(evidence.EvidenceError):
                evidence.validate_binding(raw, 30000, 932127, MOCK_SHA, changed)

    def test_third_view_renamed_as_one_of_three_sensors_is_rejected(self):
        raw, _, _ = binding_fixture()
        rt = raw['policy_ready']['runtime']
        rt['data_contract']['camera_names'] = ['head_camera', 'left_camera', 'third_person']
        rt['data_contract_sha256'] = evidence.digest(rt['data_contract'])
        raw['data_contract'] = deepcopy(rt['data_contract'])
        raw['data_contract_sha256'] = rt['data_contract_sha256']
        with self.assertRaisesRegex(evidence.EvidenceError, 'extra camera'):
            evidence.validate_binding(raw, 30000, 932127, MOCK_SHA)

    def test_actual_head_not_stale_metadata(self):
        head = SimpleNamespace(version=torch.tensor(3), precision_steps=32,
            patch_proj=SimpleNamespace(in_features=1029, out_features=768),
            configuration=torch.tensor([16, 32, 0, 1, 2]))
        wrapper = SimpleNamespace(action_model=SimpleNamespace(precision=head))
        self.assertEqual(runtime.actual_precision_binding(wrapper), evidence.ACTUAL_PRECISION_BINDING)
        head.precision_steps = 8
        with self.assertRaises(runtime.base.InfrastructureError): runtime.actual_precision_binding(wrapper)
        head.precision_steps = 32; head.patch_proj.in_features = 1026
        with self.assertRaises(runtime.base.InfrastructureError): runtime.actual_precision_binding(wrapper)

    def test_postprocessing_overlay_preserves_frozen_input(self):
        raw, frozen, _ = binding_fixture()
        before = deepcopy(frozen); changed = runtime.overlay(frozen)
        self.assertEqual(frozen, before)
        self.assertEqual(changed['action_postprocessing'], braking.protocol())
        self.assertEqual(changed['evaluation_overlay']['training_contract_sha256'], runtime.base.digest(frozen))
        self.assertFalse(changed['evaluation_overlay']['checkpoint_modified'])
        self.assertEqual(changed['robot_contract'], frozen['robot_contract'])
        self.assertEqual(changed['data_contract'], frozen['data_contract'])


class PolicyIsolationTests(unittest.TestCase):
    def observation(self):
        return dict(rgb={v: np.zeros((8, 8, 3), np.uint8) for v in runtime.base.VIEWS},
                    measured_joint_gripper_state=np.zeros(16), commanded_joint_gripper_state=np.zeros(16))

    def test_original_observation_allowlist_rejects_gt_and_fourth_camera(self):
        runtime.base.validate_observation(self.observation())
        for extra in ('rule_state', 'object_poses', 'expert_actions', 'task_progress'):
            row = self.observation(); row[extra] = 0
            with self.assertRaises(runtime.base.InfrastructureError): runtime.base.validate_observation(row)
        row = self.observation(); row['rgb']['third_person'] = np.zeros((8, 8, 3), np.uint8)
        with self.assertRaises(runtime.base.InfrastructureError): runtime.base.validate_observation(row)

    def test_original_seed_sampler_ast_and_mock_execution(self):
        def sample_ast(path):
            tree = ast.parse(path.read_text())
            worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
            return next(n for n in worker.body if isinstance(n, ast.Try) for q in [n]
                        for n in q.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')
        actual = sample_ast(ROOT / 'scripts/eval_zhuhai_wam_precision_v3_v42.py')
        old = sample_ast(ROOT / 'scripts/eval_zhuhai_wam_precision_v2.py')
        self.assertEqual(ast.dump(actual, include_attributes=False), ast.dump(old, include_attributes=False))
        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            def __getitem__(self, index): return Tensor(self.value[index])
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
        class Generator:
            def __init__(self, device): self.device = device; self.seed = None
            def manual_seed(self, seed): self.seed = seed; return self
        class Wrapper:
            def sample_actions(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                return Tensor(np.zeros((1, 32, 16))), None
        wrapper = Wrapper()
        fake_torch = SimpleNamespace(Generator=Generator, inference_mode=nullcontext, from_numpy=Tensor)
        space = dict(torch=fake_torch, hashlib=hashlib, wrapper=wrapper, condition=np.zeros(16))
        module = ast.Module(body=[actual], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<mock V3 sample>', 'exec'), space)
        for step in (0, 8, 16):
            out = space['sample']('current_features', (np.zeros((1, 16)), np.zeros((3, 1024))), 932126, step, np.zeros(6))
            self.assertEqual(out.shape, (32, 16))
            self.assertEqual(wrapper.kwargs['generator'].seed, int(hashlib.sha256(f'932126:{step}'.encode()).hexdigest()[:15], 16))
            self.assertEqual(wrapper.kwargs['num_steps'], 10)
            self.assertEqual(set(wrapper.kwargs), {'num_steps', 'generator', 'event_state'})

    def test_no_rtc_c1_prefix_commit_in_worker(self):
        source = (ROOT / 'scripts/eval_zhuhai_wam_precision_v3_v42.py').read_text()
        self.assertIn('braking.install_braking(original.clocked_policy(', source)
        for forbidden in ('zhuhai_overlap_guidance', 'zhuhai_chunk_continuity', 'zhuhai_prefix_commit', 'stage_oracle'):
            self.assertNotIn(forbidden, source)

    def test_actual_original_spline_exec8_with_official_limiter_on_mock_observations(self):
        from harness.zhuhai_action_smoothing import smooth_chunk
        with np.load(OLD_RAW.parent / 'onboard/observation_000000.npz', allow_pickle=False) as z:
            initial = z['commanded_joint_gripper_state'].copy()
        class Context:
            def reset(self): self.values = []
            def push(self, measured, pooled): self.values.append((measured.copy(), pooled.copy()))
            def inputs(self): return self.values[-1]
        samples = []
        def sample(features, inputs, seed, step, event_state):
            self.assertEqual(seed, 932127)
            self.assertEqual(event_state.shape, (6,))
            chunk = np.repeat(initial[None], 32, axis=0)
            chunk[:, 0] += .025 * np.sin(np.arange(32) * 1.2)
            chunk[:, 7] = 1.; chunk[4:6, 7] = .1
            samples.append((step, chunk.copy()))
            return chunk
        settings = braking.protocol()
        policy = braking.install_braking(runtime.original.clocked_policy(
            Context(), lambda rgb: ('mock_current_features', np.zeros((3, 2, 1024))), sample, settings), settings)
        self.assertIsInstance(policy.limiter, braking.BrakingRateLimiter)
        policy.reset('zhuhai_task2_seed_932127', 932127)
        commanded = initial.copy()
        for step in range(9):
            observation = self.observation()
            observation['measured_joint_gripper_state'] = commanded.copy()
            observation['commanded_joint_gripper_state'] = commanded.copy()
            reply = policy.observe('zhuhai_task2_seed_932127', step, observation)
            self.assertEqual(reply['replanned'], step in (0, 8))
            self.assertEqual(reply['execution_prefix'], 8)
            self.assertEqual(reply['rate_limit_stats']['braking_version'], braking.VERSION)
            chunk = samples[-1][1]; index = step % 8
            self.assertEqual(reply['model_raw_action'], chunk[index].tolist())
            expected = smooth_chunk(chunk, settings)
            np.testing.assert_array_equal(reply['smoothed_action'], expected[index])
            np.testing.assert_array_equal(np.asarray(reply['limited_action'])[[7, 15]], chunk[index, [7, 15]])
            commanded = np.asarray(reply['limited_action'])
        self.assertEqual([step for step, _ in samples], [0, 8])
        self.assertEqual(len(policy.context.values), 9)


class RuleAndClassificationTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_rejected(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(evidence.EvidenceError): evidence.loads(text)

    def test_missing_physics_safety_cannot_be_policy_failure(self):
        for field in ('passed', 'events', 'checks'):
            row = rule_fixture(); row['physics_safety'].pop(field)
            with self.assertRaises(evidence.EvidenceError): evidence.validate_rule(row)

    def test_actual_contact_terminal_strict_reason_and_time(self):
        previous, terminal = rule_fixture(0.), rule_fixture(.1, unsafe=True)
        evidence.terminal_contact(terminal, previous)
        for change in ('reason', 'time', 'event', 'passed'):
            row = deepcopy(terminal)
            if change == 'reason': row['physics_safety']['events'][0]['reason'] = 'unknown_error'
            elif change == 'time': row['physics_safety']['events'][0]['time_seconds'] = 3.
            elif change == 'event': row['physics_safety']['events'] = []
            else: row['physics_safety']['passed'] = True
            with self.assertRaises(evidence.EvidenceError): evidence.terminal_contact(row, previous)

    def test_only_exact_physics_exception_takes_contact_classifier(self):
        with patch.object(evidence, 'read_json', return_value={'status': 'infrastructure_error', 'error': evidence.ERROR}), \
             patch.object(evidence, 'classify_contact', return_value='mock physical route') as contact:
            self.assertEqual(evidence.classify('unused', {}, {}, 'unused', evidence.SEEDS, lambda *a: 'mock prior'), 'mock physical route')
            contact.assert_called_once()
        with patch.object(evidence, 'read_json', return_value={'status': 'infrastructure_error', 'error': 'InfrastructureError: missing GPU'}):
            with self.assertRaises(evidence.EvidenceError):
                evidence.classify('unused', {}, {}, 'unused', evidence.SEEDS, evidence.classify_noncontact)

    def test_wrong_physical_score_rejected_before_archive_admission(self):
        raw, frozen, _ = binding_fixture(); raw['base_score'] = 30
        with patch.object(evidence, 'read_json', return_value=raw), patch.object(evidence, 'archive_evidence') as archive:
            with self.assertRaisesRegex(evidence.EvidenceError, 'Raw score differs'):
                evidence.classify_noncontact('unused', {'step': 30000, 'sha256': MOCK_SHA}, frozen,
                    raw['policy_ready']['runtime']['evaluation_source_sha256'])
            archive.assert_not_called()

    def test_wrong_evaluator_source_rejected_before_archive_admission(self):
        raw, frozen, _ = binding_fixture()
        with patch.object(evidence, 'read_json', return_value=raw), patch.object(evidence, 'archive_evidence') as archive:
            with self.assertRaisesRegex(evidence.EvidenceError, 'overlay identity'):
                evidence.classify_noncontact('unused', {'step': 30000, 'sha256': MOCK_SHA}, frozen, 'b' * 64)
            archive.assert_not_called()

    def test_archive_count_rejected_without_generating_fake_outcome(self):
        raw, _, _ = binding_fixture(); raw['control_steps'] = 1
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); (directory / 'onboard').mkdir()
            (directory / 'actions.jsonl').write_text('')
            (directory / 'onboard/frames.jsonl').write_text('{}\n{}\n')
            (directory / 'onboard/manifest.json').write_text('{}')
            (directory / 'rule_evidence.jsonl').write_text('{}\n{}\n')
            with self.assertRaisesRegex(evidence.EvidenceError, 'count mismatch'):
                evidence.archive_evidence(directory, raw, raw['policy_ready']['runtime'], {})


class ArchivePayloadTests(unittest.TestCase):
    def payloads(self):
        directory = OLD_RAW.parent / 'onboard'
        arrays = []
        for index in (0, 1):
            with np.load(directory / f'observation_{index:06d}.npz', allow_pickle=False) as z:
                arrays.append({key: z[key].copy() for key in z.files})
        return arrays

    def check(self, arrays, expected=None):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); (directory / 'onboard').mkdir()
            if expected is None: expected = arrays[1]['commanded_joint_gripper_state'].copy()
            (directory / 'actions.jsonl').write_text(json.dumps({'dispatched_action': expected.tolist()}) + '\n')
            for index, values in enumerate(arrays):
                np.savez(directory / 'onboard' / f'observation_{index:06d}.npz', **values)
            evidence.verify_native_payloads(directory, {'fixture_only': True}, 1)

    def test_valid_old_sensor_payload_and_dtype_exact_dispatch(self):
        self.check(self.payloads())

    def test_measured_overshoot_retained_but_nan_rejected(self):
        arrays = self.payloads(); arrays[1]['measured_joint_gripper_state'][7] = 1.03
        self.check(arrays)
        arrays[1]['measured_joint_gripper_state'][0] = np.nan
        with self.assertRaisesRegex(evidence.EvidenceError, 'nonfinite'):
            self.check(arrays)

    def test_one_native_command_sample_mismatch_rejected(self):
        arrays = self.payloads(); expected = arrays[1]['commanded_joint_gripper_state'].copy()
        arrays[1]['commanded_joint_gripper_state'][0] += .001
        with self.assertRaisesRegex(evidence.EvidenceError, 'actual driver target differs'):
            self.check(arrays, expected)

    def test_observer_payload_extra_view_and_nonuint8_rgb_rejected(self):
        arrays = self.payloads(); arrays[0]['third_person_camera'] = np.zeros((8, 8, 3), np.uint8)
        with self.assertRaisesRegex(evidence.EvidenceError, 'extra or missing'):
            self.check(arrays)
        arrays = self.payloads(); arrays[0]['head_camera'] = arrays[0]['head_camera'].astype(np.float32)
        with self.assertRaisesRegex(evidence.EvidenceError, 'invalid RGB'):
            self.check(arrays)


class ObserverTests(unittest.TestCase):
    def test_requested_first_only_and_receipt_attached_before_app_exit(self):
        import harness
        module = ModuleType('harness.isaac_zhuhai_runtime')
        created = []
        class Env:
            def __init__(self, directory): self.output_dir = directory
            def setup(self, *args, **kwargs): return 'original setup'
            def capture(self): return 'original onboard capture'
            def rule_state(self): return {'video_recording': False}
            def close(self): raise SystemExit('mock SimulationApp.close')
        class Recorder:
            def __init__(self, env): self.env = env; self.frames = 0; created.append(self)
            def append(self): self.frames += 1
            def close(self):
                (self.env.output_dir / 'third_person_video_receipt.json').write_text(json.dumps({'mock_only': True, 'frames': self.frames}))
        module.IsaacZhuhaiEnv = Env
        with patch.dict(sys.modules, {'harness.isaac_zhuhai_runtime': module}), \
             patch.object(harness, 'isaac_zhuhai_runtime', module, create=True), \
             patch.object(observer, 'ObserverRecorder', Recorder), \
             patch.dict(os.environ, {'P2_V3_THIRD_VIEW_SEED': '932126'}, clear=True), \
             tempfile.TemporaryDirectory() as temp:
            observer.install(); directory = Path(temp)
            (directory / 'result.json').write_text(json.dumps({'fixture_only': True}))
            first = module.IsaacZhuhaiEnv(directory); first.setup(seed=932126)
            self.assertEqual(first.capture(), 'original onboard capture')
            self.assertTrue(first.rule_state()['observer_video_only'])
            with self.assertRaises(SystemExit): first.close()
            saved = json.loads((directory / 'result.json').read_text())
            self.assertEqual(saved['observer_video_receipt'], {'mock_only': True, 'frames': 1})
            other = module.IsaacZhuhaiEnv(directory); other.setup(seed=932127)
            self.assertIsNone(other.observer)
            self.assertFalse(other.rule_state()['observer_video_only'])
            self.assertEqual(len(created), 1)

    def test_no_requested_recording_and_unknown_selection(self):
        for value in (None, '932127'):
            with patch.dict(os.environ, {} if value is None else {'P2_V3_THIRD_VIEW_SEED': value}, clear=True):
                self.assertFalse(runtime.requested_video(932126))
        with patch.dict(os.environ, {'P2_V3_THIRD_VIEW_SEED': '932126'}, clear=True):
            self.assertTrue(runtime.requested_video(932126))
            self.assertFalse(runtime.requested_video(932127))

    def test_non126_observer_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.assertEqual(evidence.video_evidence(directory, {'seed': 932127}), {})
            (directory / 'third_person_full.mp4').write_bytes(b'mock_not_a_video')
            with self.assertRaises(evidence.EvidenceError): evidence.video_evidence(directory, {'seed': 932127})

    def test_wrong_video_hash_and_policy_reads_flag_rejected_before_decode(self):
        receipt = dict(seed=932126, frames=2, fps=10, policy_reads_observer=False,
            physics_advanced_by_recording=False, success_selection=False,
            video_sha256='b' * 64, frame_index_sha256='a' * 64)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); (directory / 'actions.jsonl').write_text('{}\n')
            receipt.update(status='encoded', path=str((directory / 'third_person_full.mp4').resolve()),
                camera_path='/World/RequestedObserverCamera', resolution_wh=[960, 720],
                policy_inputs_unchanged=list(evidence.FIELDS[:3]), duration_seconds=.2)
            for mode in ('hash', 'observer_input', 'framecount'):
                current = deepcopy(receipt)
                if mode == 'observer_input': current['policy_reads_observer'] = True
                if mode == 'framecount': current['frames'] = 3
                raw = dict(seed=932126, observer_video_receipt=current)
                with patch.object(evidence, 'read_json', return_value=current), \
                     patch.object(evidence, 'sha256_file', return_value='a' * 64), \
                     patch.object(evidence.subprocess, 'run') as decode:
                    with self.assertRaises(evidence.EvidenceError): evidence.video_evidence(directory, raw)
                    decode.assert_not_called()

    def test_wrong_video_policy_input_identity_and_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); (directory / 'actions.jsonl').write_text('{}\n')
            receipt = dict(seed=932126, frames=2, fps=10, policy_reads_observer=False,
                physics_advanced_by_recording=False, success_selection=False,
                video_sha256='a' * 64, frame_index_sha256='a' * 64,
                status='encoded', path=str((directory / 'third_person_full.mp4').resolve()),
                camera_path='/World/RequestedObserverCamera', resolution_wh=[960, 720],
                policy_inputs_unchanged=list(evidence.FIELDS[:3]), duration_seconds=.2)
            for key, wrong in [('policy_inputs_unchanged', ['head_camera', 'third_person']),
                               ('resolution_wh', [320, 240]), ('duration_seconds', .1),
                               ('path', '/tmp/different_video.mp4'), ('status', 'starting'),
                               ('camera_path', '/World/PolicyCamera')]:
                current = deepcopy(receipt); current[key] = wrong
                raw = dict(seed=932126, observer_video_receipt=current)
                with patch.object(evidence, 'read_json', return_value=current), \
                     patch.object(evidence.subprocess, 'run') as decode:
                    with self.assertRaisesRegex(evidence.EvidenceError, 'observer identity'):
                        evidence.video_evidence(directory, raw)
                    decode.assert_not_called()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out and args.out.exists(): raise ValueError('Preserve previous test receipt')
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.out:
        files = [Path(__file__), OLD_RAW, REAL_STEP0.with_name(REAL_STEP0.name + '.sha256.json'),
                 REAL_STEP0.parent / 'manifest.json', *[ROOT / p for p in (
            'scripts/eval_zhuhai_wam_precision_v3_v42.py', 'scripts/precision_v3_evidence_v42.py',
            'harness/zhuhai_observer_video_v42.py', 'harness/zhuhai_precision_v3_evaluation_v42.py',
            'harness/zhuhai_official_braking.py', 'scripts/eval_zhuhai_wam_precision_v2.py')]]
        value = dict(schema='precision-v3-runtime-evidence-independent-cpu-tests-v42',
            status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
            failures=[dict(test=str(test), details=detail) for test, detail in [*result.failures, *result.errors]],
            time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            file_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            gpu_used=False, simulator_started=False, fake_physical_v3_outcomes_created=False,
            actual_immutable_step0_cpu_contract_read=dict(path=str(REAL_STEP0), step=0,
                receipt=evidence.read_json(REAL_STEP0.with_name(REAL_STEP0.name + '.sha256.json')),
                model_constructed=False, encoder_loaded=False, cuda_initialized=False,
                actual_reader_bound_to_v3=runtime.FROZEN_READ_CONTRACT is runtime.precision.read_contract),
            scope='CPU mocks/AST/old read-only raw structure; not model/GPU/physical-success evidence. Mock fixtures are temporary and not admitted cohort outcomes.')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as f: json.dump(value, f, indent=2, ensure_ascii=False); f.write('\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
