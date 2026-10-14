#!/usr/bin/env python3
"""CPU V3 loader rejection tests plus one actual milestone construction.

No DINO call, simulator, GPU, optimizer or training is executed.  The optional
same-noise synthetic-feature probe tests topology, not task success or quality.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from harness import zhuhai_precision_v3_evaluation_v42 as loader
from harness.wam_precision_v3 import camera_ray_grid

torch.set_num_threads(2)
TRAIN = ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_v3fork80k_30k_20261012/train'
DEFAULT_CHECKPOINT = TRAIN / 'checkpoint_step_000000.pt'
DEFAULT_OUT = ROOT / 'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24/precision_v3_loading_cpu_v42.json'


def signature_fixture():
    """Only the production signature tensors; successful full load is tested separately."""
    shapes = {
        'precision.patch_proj.weight': (768, 1029), 'precision.patch_proj.bias': (768,),
        'precision.clock.weight': (768, 6), 'precision.residual.weight': (16, 768),
        'precision.residual.bias': (16,), 'dino_adapters.0.query_embed': (1, 16, 768),
        'action_proj.weight': (768, 16), 'proprio_proj.weight': (768, 16),
        'output_proj.weight': (16, 768), 'action_pos_emb': (1, 32, 768),
    }
    state = {k: torch.zeros(shape) for k, shape in shapes.items()}
    axis = (torch.arange(14, dtype=torch.float32) + .5) * (2. / 14) - 1.
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    xy = torch.stack((xx, yy), dim=-1).reshape(1, 1, 196, 2)
    state.update({'precision.version': torch.tensor(3, dtype=torch.int64),
                  'precision.configuration': torch.tensor([16, 32, 0, 1, 2]),
                  'precision.patch_xy': xy, 'precision.patch_ray': camera_ray_grid(xy)})
    return state


class LoaderRejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((TRAIN / 'manifest.json').read_text())['contract']

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v3-loader-cpu-tamper-')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'checkpoint_step_000000.pt'
        self.checkpoint = dict(format=loader.FORMAT, step=0, model_only=True,
                               contract=copy.deepcopy(self.contract), model_state_dict=signature_fixture())

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertNotIn('isaacsim', sys.modules)

    def write(self):
        self.checkpoint['contract_sha256'] = loader.v2.digest(self.checkpoint['contract'])
        torch.save(self.checkpoint, self.path)
        receipt = dict(size=self.path.stat().st_size, sha256=loader.v2.sha256_file(self.path), step=0, model_only=True)
        self.receipt_path = self.path.with_name(self.path.name + '.sha256.json')
        self.receipt_path.write_text(json.dumps(receipt))
        return receipt

    def reject(self, regex):
        with patch.object(loader, 'build_wrapper', side_effect=AssertionError('invalid file must fail before model allocation')):
            with self.assertRaisesRegex(ValueError, regex):
                loader.load_policy(self.path, device='cpu')

    def test_actual_v3_identity_does_not_follow_stale_inherited_v2_fields(self):
        self.assertEqual(self.contract['precision_v3']['checkpoint_model_version'], 2)
        self.assertEqual(self.contract['precision_v3']['refined_flow_prefix'], 8)
        self.assertIs(loader.validate_training_contract(self.contract), self.contract)
        identity = loader.validate_model_identity(self.checkpoint['model_state_dict'])
        self.assertEqual((identity['version'], identity['refine_steps']), (3, 32))

    def test_v2_milestone_rejected(self):
        self.checkpoint['format'] = loader.v2.FORMAT
        self.write(); self.reject('Immutable Precision V3')

    def test_resume_checkpoint_rejected(self):
        self.checkpoint['model_only'] = False
        self.write(); self.reject('Immutable Precision V3')

    def test_actual_head_version_tamper_rejected(self):
        self.checkpoint['model_state_dict']['precision.version'] = torch.tensor(2)
        self.write(); self.reject('actual model version')

    def test_refine8_or_wrong_view_order_rejected(self):
        for config in ([16, 8, 0, 1, 2], [16, 32, 1, 0, 2]):
            with self.subTest(config=config):
                self.checkpoint['model_state_dict']['precision.configuration'] = torch.tensor(config)
                self.write(); self.reject('actual configuration')

    def test_v2_projection_shape_rejected(self):
        self.checkpoint['model_state_dict']['precision.patch_proj.weight'] = torch.zeros(768, 1026)
        self.write(); self.reject('tensor shape mismatch: precision.patch_proj.weight')

    def test_changed_or_missing_ray_rejected(self):
        self.checkpoint['model_state_dict']['precision.patch_ray'].zero_()
        self.write(); self.reject('training ray identity')
        self.checkpoint['model_state_dict'].pop('precision.patch_ray')
        self.write(); self.reject('tensor shape mismatch: precision.patch_ray')

    def test_changed_patch_xy_rejected(self):
        self.checkpoint['model_state_dict']['precision.patch_xy'][0, 0, 0, 0] += .01
        self.write(); self.reject('patch-centre XY identity')

    def test_nonfinite_model_state_rejected(self):
        self.checkpoint['model_state_dict']['precision.residual.weight'][0, 0] = float('nan')
        self.write(); self.reject('Nonfinite Precision V3')

    def test_full32_contract_extension_required(self):
        contract = self.checkpoint['contract']
        contract['precision_v3']['refine_steps'] = 8
        contract['config']['model']['precision_v3']['refine_steps'] = 8
        self.write(); self.reject('full-32 refinement contract')

    def test_contract_sha_tamper_rejected(self):
        self.write()
        self.checkpoint['contract_sha256'] = '0' * 64
        torch.save(self.checkpoint, self.path)
        receipt = json.loads(self.receipt_path.read_text())
        receipt.update(sha256=loader.v2.sha256_file(self.path), size=self.path.stat().st_size)
        self.receipt_path.write_text(json.dumps(receipt))
        self.reject('contract SHA mismatch')

    def test_receipt_tamper_rejected_before_deserialization(self):
        original = self.write()
        for key, value in (('sha256', '0' * 64), ('size', original['size'] + 1)):
            with self.subTest(key=key):
                self.receipt_path.write_text(json.dumps(dict(original, **{key: value})))
                with patch('torch.load', side_effect=AssertionError('bad receipt must not deserialize')):
                    self.reject('checkpoint receipt mismatch')

    def test_receipt_step_and_model_only_tamper_rejected(self):
        original = self.write()
        for key, value in (('step', 5000), ('model_only', False)):
            with self.subTest(key=key):
                self.receipt_path.write_text(json.dumps(dict(original, **{key: value})))
                self.reject('Immutable Precision V3')

    def test_wrong_task_rejected(self):
        self.write()
        with patch.object(loader, 'build_wrapper', side_effect=AssertionError('wrong task must not allocate model')):
            with self.assertRaisesRegex(ValueError, 'task/condition mismatch'):
                loader.load_policy(self.path, 'zhuhai_task1', device='cpu')

    def test_statistics_tamper_rejected(self):
        from harness.zhuhai_training import NORMALIZATION_KEY
        contract = self.checkpoint['contract']
        contract['statistics']['normalization'][NORMALIZATION_KEY]['state']['std'][0] = -1.
        contract['statistics_sha256'] = loader.v2.digest(contract['statistics'])
        self.write(); self.reject('finite native16 moments')

    def test_condition_alias_tamper_rejected(self):
        self.checkpoint['contract']['conditions'][loader.TASK][0] += .1
        self.write(); self.reject('condition alias differs')

    def test_source_hash_tamper_rejected(self):
        contract = self.checkpoint['contract']
        contract['source_files'][str(ROOT / 'harness/wam_precision_v3.py')] = '0' * 64
        contract['hjepa']['code_hashes'] = copy.deepcopy(contract['source_files'])
        self.write(); self.reject('Frozen source changed')

    def test_v3_source_binding_required(self):
        contract = self.checkpoint['contract']
        contract['source_files'].pop(str(ROOT / 'harness/wam_precision_v3.py'))
        contract['hjepa']['code_hashes'] = copy.deepcopy(contract['source_files'])
        self.write(); self.reject('V3 implementation source binding missing')


def tail_influence_probe(wrapper, condition):
    """Controlled in-memory comparison with synthetic inputs and identical noise.

    Tests whether later denoising rounds can feed refined tail information into
    executed prefix. Does not claim its OOD random features represent a robot
    scene, better actions, or a positive closed-loop outcome.
    """
    generator = torch.Generator(device='cpu').manual_seed(20261012)
    features = [torch.randn(1, 3, 201, 1024, generator=generator) for _ in range(3)]
    state = ((wrapper.state_min + wrapper.state_max) / 2)[None, None].repeat(1, 7, 1)
    pool = torch.zeros(1, 3, 4, 2048)
    task_condition = torch.from_numpy(condition)[None]
    clock = torch.zeros(1, 6)
    noise = torch.randn(1, 32, 16, generator=torch.Generator(device='cpu').manual_seed(731))
    args = dict(t=torch.zeros(1), noisy_actions=noise, qpos_history=wrapper.normalize_state(state),
                dino_features_list=features, task_cond=task_condition,
                memory=wrapper.normalize_memory(pool), event_state=clock)
    model = wrapper.action_model
    original_steps = model.precision.precision_steps
    before_configuration = model.precision.configuration.clone()
    try:
        with torch.inference_mode():
            model.precision.precision_steps = 32
            full_forward = model(**args)['final_pred']
            full_actions, _ = wrapper.sample_actions(
                features, state, pool, task_condition, num_steps=10,
                generator=torch.Generator(device='cpu').manual_seed(731), event_state=clock)
            model.precision.precision_steps = 8
            prefix_forward = model(**args)['final_pred']
            prefix_actions, _ = wrapper.sample_actions(
                features, state, pool, task_condition, num_steps=10,
                generator=torch.Generator(device='cpu').manual_seed(731), event_state=clock)
    finally:
        model.precision.precision_steps = original_steps
    single_prefix = full_forward[:, :8] - prefix_forward[:, :8]
    single_tail = full_forward[:, 8:] - prefix_forward[:, 8:]
    sampled_prefix = full_actions[:, :8] - prefix_actions[:, :8]
    if not torch.equal(model.precision.configuration, before_configuration):
        raise AssertionError('Synthetic probe changed checkpoint identity buffer')
    return dict(
        status='passed_topology_probe', inputs='synthetic fixed CPU features, midpoint measured-state history, zero memory/clock, actual TRAIN condition',
        same_flow_noise_seed=731, denoising_steps=10, original_refine_steps=original_steps,
        single_forward_prefix_max_abs=float(single_prefix.abs().max()),
        single_forward_tail_max_abs=float(single_tail.abs().max()),
        full_sampler_executed_prefix_max_abs=float(sampled_prefix.abs().max()),
        full_sampler_executed_prefix_rms=float(sampled_prefix.square().mean().sqrt()),
        full_sampler_executed_arm_prefix_rms_rad=float(sampled_prefix[:, :, [*range(7), *range(8, 15)]].square().mean().sqrt()),
        full_sampler_tail_max_abs=float((full_actions[:, 8:] - prefix_actions[:, 8:]).abs().max()),
        tail_changes_prefix_in_later_flow_rounds=bool(sampled_prefix.abs().max() > 1e-8),
        in_memory_property_restored=model.precision.precision_steps == original_steps,
        scoring_gt_used=False, encoder_used=False, model_weights_saved=False,
        scope='dependency evidence only; not task success, improvement, or true 3D calibration')


def actual_checkpoint_check(path, *, probe=False):
    started = time.monotonic()
    # One production model only; torch.load comparison uses mmap without a second model.
    contract, receipt = loader.read_contract(path)
    wrapper, condition, loaded_contract = loader.load_policy(path, device='cpu')
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    loaded_state = wrapper.action_model.state_dict()
    if set(loaded_state) != set(saved['model_state_dict']):
        raise AssertionError('Strict V3 state keys differ')
    for name, value in saved['model_state_dict'].items():
        if not torch.equal(value, loaded_state[name].cpu()):
            raise AssertionError(f'Actual V3 tensor did not load exactly: {name}')
    if loaded_contract != contract or wrapper.training or wrapper.base.vision_encoder is not None:
        raise AssertionError('Actual V3 contract/eval/encoder invariant failed')
    if torch.cuda.is_initialized() or 'isaacsim' in sys.modules:
        raise AssertionError('CPU loader initialized a forbidden runtime')
    result = dict(status='passed_actual_milestone_load', checkpoint=str(Path(path).resolve()),
                  checkpoint_sha256=receipt['sha256'], step=receipt['step'],
                  tensor_count=len(loaded_state), all_tensors_bit_exact=True,
                  condition_shape=list(condition.shape), condition_dtype=str(condition.dtype),
                  actual_architecture=wrapper.precision_v3_load_receipt['actual_architecture'],
                  self_contained_no_50k_or_80k_initializer=True, encoder_loaded=False,
                  cuda_initialized=False, torch_num_threads=torch.get_num_threads())
    result['parameter_inventory'] = [dict(name=name, shape=list(value.shape), numel=value.numel(),
                                         requires_grad=value.requires_grad)
                                    for name, value in wrapper.action_model.named_parameters()]
    result['parameter_count'] = sum(row['numel'] for row in result['parameter_inventory'])
    result['buffer_inventory'] = [dict(name=name, shape=list(value.shape))
                                 for name, value in wrapper.action_model.named_buffers()]
    if probe:
        result['tail_influence_probe'] = tail_influence_probe(wrapper, condition)
    result['elapsed_seconds'] = time.monotonic() - started
    del wrapper, saved, loaded_state
    gc.collect()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--skip-actual', action='store_true')
    parser.add_argument('--tail-influence-probe', action='store_true')
    args = parser.parse_args(argv)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LoaderRejectionTests))
    report = dict(status='failed' if not result.wasSuccessful() else 'passed_rejection_tests',
                  tests_run=result.testsRun, failure_count=len(result.failures), error_count=len(result.errors),
                  failures=[dict(test=str(test), trace=trace) for test, trace in result.failures + result.errors],
                  loader_source_sha256=loader.v2.sha256_file(Path(loader.__file__)),
                  test_source_sha256=loader.v2.sha256_file(Path(__file__)),
                  cuda_initialized=torch.cuda.is_initialized(), no_training_or_simulation=True)
    if result.wasSuccessful() and not args.skip_actual:
        try:
            report['actual_checkpoint'] = actual_checkpoint_check(args.checkpoint, probe=args.tail_influence_probe)
            report['status'] = 'passed_rejection_and_actual_loading'
        except BaseException as exc:
            report['status'] = 'failed_actual_loading'
            report['actual_error'] = f'{type(exc).__name__}: {exc}'
    report['elapsed_seconds'] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + '.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(args.out)
    print(json.dumps({k: report[k] for k in ('status', 'tests_run', 'failure_count', 'error_count', 'elapsed_seconds')}, ensure_ascii=False))
    return 0 if report['status'].startswith('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
