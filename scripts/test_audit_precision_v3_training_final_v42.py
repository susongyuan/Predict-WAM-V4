#!/usr/bin/env python3
"""CPU tamper/terminal-state tests for the V3 final-training admission gate."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import audit_precision_v3_training_final_v42 as a


def log_rows(total=8):
    rows = []
    for step in range(1, total + 1):
        factor = a.lr_factor(step - 1, total, min(200, max(1, total // 50)))
        rows.append(dict(step=step, loss=.08, grad=.4, mse=.002, future=.15, future32=.15,
                         future8=.14, idm=.01, elapsed=10. + step * .6,
                         lr_trunk=3e-6 * factor, lr_prec=1e-4 * factor,
                         phase_ce=0., gate_kl=0., log_dist=0., hjepa_variant='combined',
                         future8_valid_samples=4, idm_valid_samples=4,
                         future32_weight=.25, future8_weight=.25, idm_weight=.1))
    return rows


def terminal_runtime():
    return dict(unit=dict(LoadState='loaded', ActiveState='inactive', MainPID='0',
                          ExecMainStatus='0', Result='success'), trainer_processes=[], gpu_lock_owner=None)


def head():
    axis = (torch.arange(14, dtype=torch.float32) + .5) * (2. / 14) - 1.
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    xy = torch.stack((xx, yy), dim=-1).reshape(1, 1, 196, 2)
    t = math.tan(math.radians(65.) / 2)
    ray = torch.cat((xy[..., 0:1] * t, xy[..., 1:2] * t, -torch.ones_like(xy[..., 0:1])), dim=-1)
    ray = ray / ray.norm(dim=-1, keepdim=True)
    return {'precision.version': torch.tensor(3), 'precision.configuration': torch.tensor([16, 32, 0, 1, 2]),
            'precision.patch_xy': xy, 'precision.patch_ray': ray,
            'precision.patch_proj.weight': torch.zeros(768, 1029), 'precision.patch_proj.bias': torch.zeros(768)}


def optimizer():
    return dict(state={0: dict(step=torch.tensor(30000.), exp_avg=torch.zeros(2), exp_avg_sq=torch.ones(2)),
                       1: dict(step=torch.tensor(30000.), exp_avg=torch.zeros(3), exp_avg_sq=torch.ones(3))},
                param_groups=[dict(params=[0], lr=3e-7, betas=(.9, .99), weight_decay=.01),
                              dict(params=[1], lr=1e-5, betas=(.9, .99), weight_decay=.01)])


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory(prefix='v3-final-audit-cpu-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertNotIn('isaacsim', sys.modules)

    def write_log(self, rows):
        path = self.directory / 'log.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def test_contiguous_final_log_with_correct_warmup_cosine(self):
        result = a.validate_log(self.write_log(log_rows()), total=8, final=True)
        self.assertEqual(result['optimizer_updates'], 8)
        self.assertEqual(result['idm_parameter_updates'], 8)

    def test_idm_compact_digest_matches_its_producer_and_differs_from_contract_digest(self):
        value = [{'path': 'source.hdf5', 'split': 'train', 'frames': 921}]
        self.assertNotEqual(a.idm_digest(value), a.digest(value))
        self.assertEqual(a.idm_digest(value), __import__('hashlib').sha256(
            json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())

    def test_actual_contract_and_resigned_data_split_head_init_source_tampering(self):
        production = ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_v3fork80k_30k_20261012/train'
        original = json.loads((production / 'manifest.json').read_text())['contract']
        original_plan = json.loads((production / 'plan.json').read_text())

        def install(contract):
            sha = a.digest(contract)
            (self.directory / 'manifest.json').write_text(json.dumps(dict(contract=contract, contract_sha256=sha)))
            (self.directory / 'plan.json').write_text(json.dumps(dict(original_plan, contract_sha256=sha)))

        install(original)
        _, result = a.validate_manifest(self.directory)
        self.assertEqual(result['actual_train_episodes'], 160)
        mutations = [lambda c: c['splits'][a.TASK]['train'][0].update(domain='dagger'),
                     lambda c: c['splits'][a.TASK]['train'].pop(),
                     lambda c: c['splits'][a.TASK]['val'][0].update(family_id=c['splits'][a.TASK]['train'][0]['family_id']),
                     lambda c: c['condition_bank']['provenance'].update(heldout_used=True),
                     lambda c: c['warm_start'].update(checkpoint_sha256='0' * 64),
                     lambda c: c.update(statistics_sha256='0' * 64),
                     lambda c: c['precision_v3'].update(refine_steps=8),
                     lambda c: c.update(format='zhuhai-precision-wam-native16-measured-10hz-v2'),
                     lambda c: c['source_files'].update({str(ROOT / 'scripts/train_zhuhai_wam_precision_v3.py'): '0' * 64})]
        for mutation in mutations:
            value = copy.deepcopy(original); mutation(value); install(value)
            with self.assertRaises(ValueError):
                a.validate_manifest(self.directory)

        # Re-signing both the manifest and plan cannot hide on-disk source drift.
        fake = self.directory / 'frozen_helper.py'; fake.write_text('x = 1\n')
        value = copy.deepcopy(original); value['source_files'][str(fake)] = a.sha(fake)
        value['hjepa']['code_hashes'] = copy.deepcopy(value['source_files']); install(value)
        a.validate_manifest(self.directory)
        fake.write_text('x = 2\n')
        with self.assertRaisesRegex(ValueError, 'source drift'):
            a.validate_manifest(self.directory)

    def test_omitted_duplicate_reordered_nonfinite_wrong_lr_loss_contract_rejected(self):
        mutations = [lambda r: r.pop(3), lambda r: r.insert(3, copy.deepcopy(r[2])),
                     lambda r: r.reverse(), lambda r: r[2].update(loss=float('nan')),
                     lambda r: r[2].update(lr_prec=1e-4), lambda r: r[2].update(phase_ce=.1),
                     lambda r: r[2].update(future8_valid_samples=5),
                     lambda r: r[2].update(idm_valid_samples=-1),
                     lambda r: r[2].update(future32_weight=.5),
                     lambda r: r[2].update(elapsed=0.)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                rows = log_rows(); mutation(rows)
                with self.assertRaises(ValueError):
                    a.validate_log(self.write_log(rows), total=8, final=True)

    def test_valid_terminal_masked_samples_do_not_falsely_reject_log(self):
        rows = log_rows(); rows[2].update(future8_valid_samples=0, idm_valid_samples=0)
        rows[5].update(future8_valid_samples=3, idm_valid_samples=2)
        result = a.validate_log(self.write_log(rows), total=8, final=True)
        self.assertEqual(result['idm_parameter_updates'], 7)

    def test_live_partial_trailing_write_is_waiting_but_final_partial_write_rejected(self):
        path = self.write_log(log_rows()[:-1])
        with path.open('a') as stream:
            stream.write('{"step":8')
        self.assertEqual(a.validate_log(path, total=8)['optimizer_updates'], 7)
        with self.assertRaises(ValueError):
            a.validate_log(path, total=8, final=True)

    def test_version_configuration_projection_ray_and_finiteness_tampering(self):
        a.validate_head(head())
        mutations = [lambda s: s['precision.version'].fill_(2),
                     lambda s: s['precision.configuration'].__setitem__(1, 8),
                     lambda s: s.update({'precision.patch_proj.weight': torch.zeros(768, 1026)}),
                     lambda s: s['precision.patch_ray'].__setitem__((0, 0, 2, 1), .12),
                     lambda s: s['precision.patch_proj.weight'].__setitem__((0, 0), float('inf'))]
        for mutation in mutations:
            state = head(); mutation(state)
            with self.assertRaises(ValueError):
                a.validate_head(state)

    def test_optimizer_lr_duplicate_ids_step_moments_tamper(self):
        a.validate_optimizer(optimizer(), [3e-7, 1e-5])
        mutations = [lambda o: o['state'][0]['step'].fill_(29999),
                     lambda o: o['param_groups'][1].update(params=[0]),
                     lambda o: o['param_groups'][1].update(lr=3e-6),
                     lambda o: o['state'][1].pop('exp_avg'),
                     lambda o: o['state'][1]['exp_avg_sq'].fill_(-1),
                     lambda o: o['state'][0]['exp_avg'].fill_(float('nan'))]
        for mutation in mutations:
            value = optimizer(); mutation(value)
            with self.assertRaises(ValueError):
                a.validate_optimizer(value, [3e-7, 1e-5])

    def test_idm_skipped_update_only_allowed_for_corresponding_parameter_name(self):
        value = optimizer(); value['state'][0]['step'].fill_(29999)
        a.validate_optimizer(value, [3e-7, 1e-5], parameter_names=[['hjepa_aux.idm_head.0.weight'], ['precision.residual.weight']], idm_updates=29999)
        with self.assertRaises(ValueError):
            a.validate_optimizer(value, [3e-7, 1e-5], parameter_names=[['action_proj.weight'], ['precision.residual.weight']], idm_updates=29999)

    def test_rng_archive_restores_without_cuda_and_missing_fields_rejected(self):
        import numpy as np
        value = dict(rng=np.random.default_rng(1).bit_generator.state,
                     torch_rng=torch.Generator(device='cpu').get_state(), cuda_rng=[torch.zeros(16, dtype=torch.uint8)])
        a.validate_rng(value)
        for key in ('rng', 'torch_rng', 'cuda_rng'):
            bad = copy.deepcopy(value); bad.pop(key)
            with self.assertRaises(ValueError):
                a.validate_rng(bad)

    def test_final_runtime_requires_real_success_exit_and_no_live_owner_or_writer(self):
        self.assertTrue(a.runtime_ready(terminal_runtime()))
        for mutation in [lambda r: r.update(trainer_processes=[{'pid': 123}]),
                         lambda r: r.update(gpu_lock_owner=a.OWNER),
                         lambda r: r.update(gpu_lock_owner='other-owner'),
                         lambda r: r['unit'].update(ActiveState='active', MainPID='123'),
                         lambda r: r['unit'].update(ExecMainStatus='1'),
                         lambda r: r['unit'].update(LoadState='not-found')]:
            runtime = terminal_runtime(); mutation(runtime)
            self.assertFalse(a.runtime_ready(runtime))

    def test_training_and_completed_but_live_never_load_large_checkpoints(self):
        for completed, live in [(False, True), (True, True), (False, False)]:
            with self.subTest(completed=completed, live=live):
                status = dict(status='completed' if completed else 'training', step=30000 if completed else 100,
                              target_steps=30000)
                (self.directory / 'status.json').write_text(json.dumps(status))
                (self.directory / 'manifest.json').write_text('{}')
                (self.directory / 'log.jsonl').write_text('')
                runtime = terminal_runtime()
                if live:
                    runtime['trainer_processes'] = [{'pid': 123}]
                    runtime['unit'].update(ActiveState='active', MainPID='123')
                    runtime['gpu_lock_owner'] = a.OWNER
                with patch.object(a, 'validate_manifest', return_value=({}, {})), \
                     patch.object(a, 'validate_log', return_value={'optimizer_updates': status['step'], 'last': None}), \
                     patch.object(a, 'validate_final_artifacts') as final:
                    report = a.audit(self.directory, runtime=runtime)
                self.assertEqual(report['status'], 'waiting_for_training_completion')
                self.assertFalse(report['evaluation_ready'])
                final.assert_not_called()

    def test_step0_and_building_without_manifest_are_not_completed_or_failed(self):
        (self.directory / 'status.json').write_text(json.dumps(dict(status='building', step=0, target_steps=30000)))
        report = a.audit(self.directory, runtime=terminal_runtime())
        self.assertEqual(report['status'], 'waiting_for_training_completion')
        self.assertFalse(report['evaluation_ready'])

    def test_completed_gate_requires_final_artifact_proof_then_rechecks_runtime(self):
        (self.directory / 'status.json').write_text(json.dumps(dict(status='completed', step=30000, target_steps=30000)))
        (self.directory / 'manifest.json').write_text('{}')
        (self.directory / 'log.jsonl').write_text('')
        with patch.object(a, 'validate_manifest', return_value=({}, {})), \
             patch.object(a, 'validate_log', return_value={'optimizer_updates': 30000, 'last': {}}), \
             patch.object(a, 'validate_final_artifacts', return_value={'independent_fixture': True}) as final:
            report = a.audit(self.directory, runtime=terminal_runtime())
        self.assertTrue(report['evaluation_ready'])
        self.assertEqual(report['status'], 'passed_final_training_and_gpu_released')
        final.assert_called_once()
        with patch.object(a, 'validate_manifest', return_value=({}, {})), \
             patch.object(a, 'validate_log', return_value={'optimizer_updates': 30000, 'last': {}}), \
             patch.object(a, 'validate_final_artifacts', side_effect=ValueError('bad actual final weights')):
            rejected = a.audit(self.directory, runtime=terminal_runtime())
        self.assertEqual(rejected['status'], 'failed_audit_invariant')
        self.assertFalse(rejected['evaluation_ready'])

    def test_real_mmap_checkpoint_final_artifact_pipeline_and_resigned_weight_tamper(self):
        """Small tensor archives exercise the real parser, not a mocked green gate."""
        import numpy as np
        init_state = head()
        # Exactly 323 trainable tensors and six frozen task-condition tensors;
        # no policy/module is allocated. Production has the immutable real INIT.
        for index in range(321):
            init_state[f'fixture_parameter_{index}.weight'] = torch.tensor([.1])
        for index in range(6):
            init_state[f'task_cond_proj.base.fixture_{index}.weight'] = torch.tensor([.2])
        init_state['action_pos_emb'] = torch.zeros(1)
        init_state['proprio_pos_emb'] = torch.zeros(1)
        init_state['hjepa_aux.idm_scale'] = torch.ones(16)
        old_state = copy.deepcopy(init_state)
        old_state.pop('precision.patch_ray')
        old_state['precision.version'].fill_(2)
        old_state['precision.configuration'][1] = 8
        old_state['precision.patch_proj.weight'] = old_state['precision.patch_proj.weight'][:, :1026].clone()
        encoder = self.directory / 'fake_encoder'; encoder.mkdir()
        (encoder / 'model.safetensors').write_bytes(b'unchanged frozen encoder fixture')
        dataset = self.directory / 'source.hdf5'; dataset.write_bytes(b'unchanged dataset fixture')
        row = dict(path=str(dataset), sha256=a.sha(dataset))
        base = dict(format='zhuhai-precision-wam-native16-measured-10hz-v2',
                    precision=dict(schema='all_view_dense_near_term_precision_v2', model=dict(precision_steps=8),
                                   refined_flow_prefix=8, checkpoint_model_version=2),
                    config=dict(model=dict(vision_encoder=dict(checkpoint_path=str(encoder)))),
                    source_files={}, hjepa=dict(code_hashes={}), cache=dict(encoder_sha256=a.sha(encoder / 'model.safetensors')),
                    splits={a.TASK: dict(train=[row] * 160, val=[row] * 20, heldout=[row] * 20)})
        original = self.directory / 'old80k.pt'
        torch.save(dict(step=80000, contract=base, model_state_dict=old_state), original)
        init_sha = a.sha(original)
        contract = copy.deepcopy(base); contract['format'] = a.FORMAT
        spec = copy.deepcopy(base['precision']); spec.update(schema='full_chunk_camera_ray_precision_v3', refine_steps=32)
        contract['precision_v3'] = spec; contract['config']['model']['precision_v3'] = copy.deepcopy(spec)
        contract['source_files'] = {str(ROOT / rel): sha for rel, sha in a.TRUSTED_EXTENSION.items()}
        contract['hjepa']['code_hashes'] = copy.deepcopy(contract['source_files'])
        contract['warm_start'] = dict(fork='80k v2 Precision', checkpoint_sha256=init_sha,
                                     control_run='none (user directive: fork-80k only, highest success)', precision_head_lr=1e-4)
        final_state = copy.deepcopy(init_state); final_state['precision.patch_proj.weight'][0, -1] = .01
        buffers = {'action_pos_emb', 'proprio_pos_emb', 'hjepa_aux.idm_scale', 'precision.version',
                   'precision.configuration', 'precision.patch_xy', 'precision.patch_ray'}
        names = [name for name in init_state if name not in buffers and not name.startswith('task_cond_proj.base.')]
        by_group = [[name for name in names if not name.startswith('precision.')],
                    [name for name in names if name.startswith('precision.')]]
        ids = [list(range(len(by_group[0]))), list(range(len(by_group[0]), len(names)))]
        opt = dict(state={}, param_groups=[])
        for group_ids, group_names, lr in zip(ids, by_group, [3e-7, 1e-5]):
            opt['param_groups'].append(dict(params=group_ids, lr=lr, betas=(.9, .99), weight_decay=.01))
            for pid, name in zip(group_ids, group_names):
                opt['state'][pid] = dict(step=torch.tensor(30000.), exp_avg=torch.zeros_like(init_state[name]),
                                        exp_avg_sq=torch.ones_like(init_state[name]))

        def save(step, state, full=False):
            path = self.directory / ('checkpoint_last.pt' if full else f'checkpoint_step_{step:06d}.pt')
            ck = dict(format=a.FORMAT, step=step, model_only=not full, contract=contract,
                      contract_sha256=a.digest(contract), model_state_dict=state)
            if full:
                ck.update(optimizer_state_dict=opt, rng=np.random.default_rng(1).bit_generator.state,
                          torch_rng=torch.Generator(device='cpu').get_state(), cuda_rng=[torch.zeros(16, dtype=torch.uint8)])
            torch.save(ck, path)
            path.with_name(path.name + '.sha256.json').write_text(json.dumps(dict(
                sha256=a.sha(path), size=path.stat().st_size, step=step, model_only=not full)))

        save(0, init_state)
        for step in range(5000, 30001, 5000):
            save(step, final_state)
            (self.directory / f'diagnostic_{step:06d}.json').write_text(json.dumps(dict(
                train={a.TASK: dict(loss=.08)}, val={a.TASK: dict(loss=.081)})))
        save(30000, final_state, full=True)
        log = dict(last=dict(lr_trunk=3e-7, lr_prec=1e-5), idm_parameter_updates=30000)
        with patch.object(a, 'INITIALIZER', original), patch.object(a, 'INIT_SHA', init_sha):
            result = a.validate_final_artifacts(self.directory, contract, log)
            self.assertEqual(result['optimizer']['parameters_with_full_moments'], 323)
            self.assertEqual(result['head']['refinement_steps'], 32)
            self.assertEqual(result['heldout_archives_opened'], 0)
            # Re-signing a modified full-state file cannot hide disagreement
            # with the immutable final milestone.
            corrupt = copy.deepcopy(final_state); corrupt['precision.patch_proj.weight'][0, -1] = .02
            save(30000, corrupt, full=True)
            with self.assertRaisesRegex(ValueError, 'full-state and final milestone weights differ'):
                a.validate_final_artifacts(self.directory, contract, log)


if __name__ == '__main__':
    result = unittest.main(exit=False)
    require_no_gpu = not torch.cuda.is_initialized()
    print(json.dumps(dict(status='passed' if result.result.wasSuccessful() and require_no_gpu else 'failed',
                          tests_run=result.result.testsRun, model_instances_constructed=0,
                          cuda_initialized=torch.cuda.is_initialized())))
    raise SystemExit(0 if result.result.wasSuccessful() and require_no_gpu else 1)
