#!/usr/bin/env python3
"""Small CPU fixtures only. Never read production checkpoints or import training."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

spec = importlib.util.spec_from_file_location('checkpoint_verifier', Path(__file__).with_name('verify_zhuhai_precision_checkpoint.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class CheckpointVerifierTests(unittest.TestCase):
    def setUp(self):
        assert torch.cuda.is_initialized() is False
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'train'
        self.run.mkdir()
        self.sources = {}
        for i in range(33):
            p = self.root / f'frozen_{i}.py'
            p.write_text(f'# synthetic source {i}\n')
            with p.open('rb') as f:
                self.sources[str(p)] = v.file_sha(f)
        precision = dict(schema='all_view_dense_near_term_precision_v2', checkpoint_model_version=2)
        for key, value in [('SOURCE_DIGEST', v.digest(self.sources)), ('PRECISION_DIGEST', v.digest(precision))]:
            patch = mock.patch.object(v, key, value)
            patch.start()
            self.addCleanup(patch.stop)
        data = dict(schema='zhuhai-native16-measured-target-10hz-v1', robot_dof=16, state_dim=16,
                    action_dim=16, frequency_hz=10., gripper_indices=[7, 15],
                    state_semantics='measured_joint_gripper', action_semantics='joint_position_target')
        training = dict(max_steps=30000, schedule_steps=30000, batch_size=4, grad_accum_steps=1,
                        save_every=5000, warmup_steps=100, learning_rate=1e-5, lr_min=1e-6,
                        weight_decay=.01, betas=[.9, .99],
                        auxiliary_loss=dict(future=.25, future8=.25, idm=.1, gate_kl=0., phase_ce=0., log_dist=0.))
        self.c = dict(format=v.FORMAT, mode='train', precision=precision, source_files=self.sources,
                      config=dict(training=training, common=dict(action_dim=16, state_dim=16,
                                  action_chunk_size=32, action_execution_horizon=8),
                                  model=dict(precision=precision, proprio_mask='native_all16',
                                             action_expert=dict(hidden_size=4))),
                      hjepa=dict(variant='combined', task_scope=['zhuhai_task2'],
                                 settings=dict(multi_horizon=True, use_idm=True), code_hashes=self.sources),
                      protocol=dict(pure_wam=True, record_video=False, third_view_camera_created=False,
                                    vlm=False, high_level_planner=False, ask=False, lifelong_memory_writes=False),
                      data_contract=data, robot_contract={'fixture': 'robot'}, statistics={'fixture': 'train'},
                      idm_statistics={'fixture': 'idm'}, action_postprocessing={'fixture': 'native16'})
        for name in ('data_contract', 'robot_contract', 'statistics', 'idm_statistics', 'action_postprocessing'):
            self.c[name + '_sha256'] = v.digest(self.c[name])
        for name in ('data_contract', 'robot_contract'):
            self.c['config'][name] = self.c[name]
            self.c['config'][name + '_sha256'] = self.c[name + '_sha256']
        self.manifest()
        self.model_path = self.run / 'checkpoint_step_005000.pt'
        self.full_path = self.run / 'checkpoint_last.pt'
        self.model = self.payload(5000, True)
        self.full = self.payload(5000, False)
        self.publish(self.model_path, self.model)
        self.publish(self.full_path, self.full)

    def tearDown(self):
        assert torch.cuda.is_initialized() is False

    def manifest(self):
        (self.run / 'manifest.json').write_text(json.dumps(dict(contract=self.c, contract_sha256=v.digest(self.c))))

    def payload(self, step, model_only):
        state = {'precision.version': torch.tensor(2),
                 'precision.configuration': torch.tensor([16, 8, 0, 1, 2]),
                 'precision.clock.weight': torch.zeros(4, 6),
                 'precision.residual.weight': torch.zeros(16, 4),
                 'precision.patch_xy': torch.zeros(1, 1, 196, 2)}
        saved = dict(format=v.FORMAT, step=step, model_only=model_only,
                     model_state_dict=state, contract=copy.deepcopy(self.c), contract_sha256=v.digest(self.c))
        if not model_only:
            saved.update(optimizer_state_dict={'state': {0: dict(step=torch.tensor(float(step)),
                         exp_avg=torch.zeros(4, 6), exp_avg_sq=torch.zeros(4, 6))},
                         'param_groups': [dict(params=[0], lr=v.expected_lr(self.c['config']['training'], step),
                         betas=(.9, .99), weight_decay=.01, eps=1e-8, amsgrad=False, maximize=False)]},
                         torch_rng=torch.arange(64, dtype=torch.uint8), cuda_rng=[torch.arange(32, dtype=torch.uint8)],
                         rng=dict(bit_generator='PCG64', state=dict(state=123, inc=3), has_uint32=0, uinteger=0))
        return saved

    def receipt(self, path):
        return path.with_name(path.name + '.sha256.json')

    def publish(self, path, value):
        tmp = path.with_suffix('.tmp')
        torch.save(value, tmp)
        tmp.replace(path)
        with path.open('rb') as f:
            receipt = dict(sha256=v.file_sha(f), size=path.stat().st_size, step=value['step'], model_only=value['model_only'])
        self.receipt(path).write_text(json.dumps(receipt))
        # Deterministic publication ordering even on coarse timestamp filesystems.
        newer = path.stat().st_mtime_ns + 1_000_000
        os.utime(self.receipt(path), ns=(newer, newer))

    def check(self, expected='verified', reason=None):
        result = v.verify(self.run, 5000)
        self.assertEqual(result['status'], expected, result)
        self.assertIs(result['cuda_initialized'], False)
        if reason:
            self.assertIn(reason, result.get('reason', ''))
        return result

    def test_normal_checkpoint_and_read_only_inputs(self):
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        r = self.check()
        self.assertIs(r['model_only']['model_only_cannot_resume'], True)
        self.assertIs(r['full_state_is_requested_milestone'], True)
        self.assertEqual(r['full_state']['full_state']['last_update_index'], 4999)
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_run_root_is_accepted(self):
        self.assertEqual(v.verify(self.root, 5000)['status'], 'verified')

    def test_committed_bad_hash(self):
        p = self.receipt(self.model_path)
        r = json.loads(p.read_text()); r['sha256'] = '0' * 64
        p.write_text(json.dumps(r))
        newer = self.model_path.stat().st_mtime_ns + 1_000_000
        os.utime(p, ns=(newer, newer))
        self.check('invalid', 'SHA/size')

    def test_committed_bad_size(self):
        p = self.receipt(self.model_path)
        r = json.loads(p.read_text()); r['size'] += 1
        p.write_text(json.dumps(r))
        newer = self.model_path.stat().st_mtime_ns + 1_000_000
        os.utime(p, ns=(newer, newer))
        self.check('invalid', 'SHA/size')

    def test_wrong_milestone_step(self):
        self.model['step'] = 4999
        self.publish(self.model_path, self.model)
        self.check('invalid', 'milestone')

    def test_wrong_receipt_step(self):
        p = self.receipt(self.model_path)
        r = json.loads(p.read_text()); r['step'] = 5001
        p.write_text(json.dumps(r))
        self.check('invalid', 'step mismatch')

    def test_wrong_format(self):
        self.model['format'] = 'v1'
        self.publish(self.model_path, self.model)
        self.check('invalid', 'format')

    def test_wrong_model_only(self):
        self.model['model_only'] = False
        self.publish(self.model_path, self.model)
        self.check('invalid', 'model_only')

    def test_bool_step_rejected(self):
        self.model['step'] = True
        self.publish(self.model_path, self.model)
        self.check('invalid', 'integer')

    def test_manifest_bad_digest(self):
        p = self.run / 'manifest.json'; r = json.loads(p.read_text()); r['contract_sha256'] = '0' * 64
        p.write_text(json.dumps(r))
        self.check('invalid', 'digest')

    def test_checkpoint_bad_contract_digest(self):
        self.model['contract_sha256'] = '0' * 64
        self.publish(self.model_path, self.model)
        self.check('invalid', 'contract mismatch')

    def test_checkpoint_contract_differs(self):
        self.model['contract']['mode'] = 'tiny'
        self.model['contract_sha256'] = v.digest(self.model['contract'])
        self.publish(self.model_path, self.model)
        self.check('invalid', 'contract mismatch')

    def test_wrong_fixed_contract(self):
        for section, key, value in [('training', 'max_steps', 20000), ('training', 'batch_size', 8),
                                    ('training', 'schedule_steps', 25000), ('common', 'action_dim', 14)]:
            with self.subTest(key=key):
                old = self.c['config'][section][key]
                self.c['config'][section][key] = value; self.manifest()
                self.check('invalid')
                self.c['config'][section][key] = old
        self.manifest()

    def test_wrong_precision_schema(self):
        self.c['precision']['schema'] = 'v1'; self.manifest()
        self.check('invalid', 'V2 schema')

    def test_wrong_combined_variant(self):
        self.c['hjepa']['variant'] = 'idm'; self.manifest()
        self.check('invalid', 'variant')

    def test_missing_optimizer(self):
        del self.full['optimizer_state_dict']; self.publish(self.full_path, self.full)
        self.check('invalid', 'AdamW state')

    def test_missing_rng_fields(self):
        for name in ('rng', 'torch_rng', 'cuda_rng'):
            with self.subTest(name=name):
                f = copy.deepcopy(self.full); del f[name]; self.publish(self.full_path, f)
                self.check('invalid', 'RNG')

    def test_bad_rng_types(self):
        for name, value in [('rng', {}), ('torch_rng', torch.zeros(4)), ('cuda_rng', [])]:
            with self.subTest(name=name):
                f = copy.deepcopy(self.full); f[name] = value; self.publish(self.full_path, f)
                self.check('invalid', 'RNG')

    def test_wrong_last_lr(self):
        self.full['optimizer_state_dict']['param_groups'][0]['lr'] = 1e-5
        self.publish(self.full_path, self.full); self.check('invalid', 'LR')

    def test_lr_independent_absolute_reference(self):
        for step in (1, 100, 5000, 20000, 25000, 30000):
            i = step - 1
            import math
            expected = 1e-5 * ((i + 1) / 100 if i < 100 else .1 + .45 * (1 + math.cos(math.pi * (i - 100) / 29900)))
            self.assertAlmostEqual(v.expected_lr(self.c['config']['training'], step), expected, places=18)

    def test_optimizer_moment_missing(self):
        del self.full['optimizer_state_dict']['state'][0]['exp_avg_sq']
        self.publish(self.full_path, self.full); self.check('invalid', 'exp_avg_sq')

    def test_optimizer_wrong_step(self):
        self.full['optimizer_state_dict']['state'][0]['step'] = torch.tensor(4999.)
        self.publish(self.full_path, self.full); self.check('invalid', 'boundary')

    def test_source_drift(self):
        Path(next(iter(self.sources))).write_text('# drift\n')
        self.check('invalid', 'source drift')

    def test_source_hash_relabel_rejected(self):
        self.c['source_files'][next(iter(self.sources))] = '0' * 64; self.manifest()
        self.check('invalid', '33-source')

    def test_model_metadata_mismatch(self):
        self.full['model_state_dict']['extra.weight'] = torch.zeros(1)
        self.publish(self.full_path, self.full); self.check('invalid', 'keys/shapes/dtypes')

    def test_model_shape_mismatch(self):
        self.full['model_state_dict']['precision.clock.weight'] = torch.zeros(4, 7)
        self.publish(self.full_path, self.full); self.check('invalid', 'shape mismatch')

    def test_v1_identity_buffer_rejected(self):
        self.model['model_state_dict']['precision.version'] = torch.tensor(1)
        self.publish(self.model_path, self.model); self.check('invalid', 'version')

    def test_newer_full_state_reported_separately(self):
        self.publish(self.full_path, self.payload(20000, False))
        r = self.check()
        self.assertEqual(r['full_state_step_relation'], 'newer')
        self.assertEqual(r['model_only']['step'], 5000)
        self.assertEqual(r['full_state']['step'], 20000)
        self.assertIs(r['full_state_is_requested_milestone'], False)

    def test_older_full_state_pending(self):
        self.publish(self.full_path, self.payload(1000, False))
        r = self.check('pending', 'caught up')
        self.assertEqual(r['full_state_step_relation'], 'older')

    def test_missing_milestone_pending(self):
        self.model_path.unlink(); self.check('pending')

    def test_missing_receipt_pending(self):
        self.receipt(self.model_path).unlink(); self.check('pending')

    def test_new_weights_old_receipt_publication_gap(self):
        old = self.receipt(self.full_path)
        os.utime(old, ns=(1, 1))
        f = self.payload(20000, False)
        tmp = self.full_path.with_suffix('.tmp'); torch.save(f, tmp); tmp.replace(self.full_path)
        self.check('pending', 'publication gap')

    def test_replacement_during_load_pending_and_pinned_inode(self):
        original = torch.load
        def replacing(path, **kwargs):
            self.assertEqual(kwargs, dict(map_location='cpu', mmap=True, weights_only=True))
            self.assertTrue(str(path).startswith('/proc/self/fd/'))
            saved = original(path, **kwargs)
            self.publish(self.model_path, self.payload(5000, True))
            return saved
        with mock.patch.object(torch, 'load', side_effect=replacing):
            self.check('pending', 'changed')

    def test_replacement_during_hash_is_pending(self):
        original = v.file_sha
        def replacing(stream):
            result = original(stream)
            if str(stream.name) == str(self.model_path):
                replacement = self.model_path.with_suffix('.replacement')
                replacement.write_bytes(self.model_path.read_bytes())
                replacement.replace(self.model_path)
            return result
        with mock.patch.object(v, 'file_sha', side_effect=replacing):
            self.check('pending', 'changed')

    def test_missing_full_state_pending(self):
        self.full_path.unlink()
        self.check('pending')

    def test_wrong_checkpoint_model_only_with_correct_receipt(self):
        self.model['model_only'] = False
        self.publish(self.model_path, self.model)
        p = self.receipt(self.model_path)
        r = json.loads(p.read_text()); r['model_only'] = True
        p.write_text(json.dumps(r))
        self.check('invalid', 'model_only')

    def test_malformed_checkpoint_schema_with_matching_receipt(self):
        self.model_path.write_bytes(b'not a torch zip checkpoint')
        with self.model_path.open('rb') as stream:
            receipt = dict(sha256=v.file_sha(stream), size=self.model_path.stat().st_size,
                           step=5000, model_only=True)
        self.receipt(self.model_path).write_text(json.dumps(receipt))
        self.check('invalid', 'schema')

    def test_all_allowed_milestones(self):
        for step in v.STEPS:
            with self.subTest(step=step):
                self.publish(self.run / f'checkpoint_step_{step:06d}.pt', self.payload(step, True))
                self.publish(self.full_path, self.payload(step, False))
                result = v.verify(self.run, step)
                self.assertEqual(result['status'], 'verified', result)
                self.assertEqual(result['full_state']['full_state']['last_update_index'], step - 1)

    def test_every_parameter_group_lr_is_checked(self):
        group = copy.deepcopy(self.full['optimizer_state_dict']['param_groups'][0])
        group['params'] = [1]; group['lr'] = 1e-5
        self.full['optimizer_state_dict']['param_groups'].append(group)
        self.publish(self.full_path, self.full)
        self.check('invalid', 'LR')

    def test_no_training_logger_imports(self):
        self.check()
        for name in ('scripts.progress1_log', 'scripts.train_zhuhai_wam_precision_v2', 'harness.wam_precision_v2'):
            self.assertNotIn(name, sys.modules)

    def test_already_initialized_cuda_rejected(self):
        with mock.patch.object(torch.cuda, 'is_initialized', return_value=True):
            with self.assertRaises(AssertionError):
                v.verify(self.run, 5000)

    def test_out_cannot_overwrite_training_or_create_directory(self):
        before = self.model_path.read_bytes()
        with self.assertRaises(v.Invalid):
            v.write_report(self.model_path, {'status': 'verified'})
        self.assertEqual(before, self.model_path.read_bytes())
        with self.assertRaises(v.Invalid):
            v.write_report(self.root / 'new' / 'report.json', {})
        self.assertFalse((self.root / 'new').exists())

    def test_authorized_report_only(self):
        report_path = self.root / 'checkpoint_005000_audit.json'
        with mock.patch.object(v, 'AUDIT_DIR', self.root):
            v.write_report(report_path, self.check())
        self.assertEqual(json.loads(report_path.read_text())['status'], 'verified')

    def test_step_matched_audit_names_and_attempts_only(self):
        with mock.patch.object(v, 'AUDIT_DIR', self.root):
            for step in v.STEPS:
                for suffix in ('', '_attempt2'):
                    path = self.root / f'checkpoint_{step:06d}_audit{suffix}.json'
                    v.write_report(path, dict(requested_step=step, status='pending'))
                    self.assertEqual(json.loads(path.read_text())['requested_step'], step)
            for path in (self.root / 'checkpoint_020000_audit_attempt3.json',
                         self.root / 'checkpoint_005000_audit_retry.json',
                         self.root / 'checkpoint_005000_audit_attempt0.json',
                         self.root / 'checkpoint_verifier_review.json',
                         self.run / 'checkpoint_005000_audit.json',
                         self.root / 'new' / '..' / 'checkpoint_005000_audit_attempt3.json',
                         self.root.parent / 'checkpoint_005000_audit.json'):
                with self.subTest(path=path), self.assertRaises(v.Invalid):
                    v.write_report(path, dict(requested_step=5000))
            self.assertFalse((self.root / 'new').exists())

    def test_existing_review_audit_and_links_are_preserved(self):
        review = self.root / 'checkpoint_verifier_review.json'
        review.write_bytes(b'{"original_cpu_tests":44}\n')
        audit = self.root / 'checkpoint_005000_audit.json'
        audit.write_bytes(b'{"status":"pending"}\n')
        symlink = self.root / 'checkpoint_005000_audit_attempt2.json'
        symlink.symlink_to(self.model_path)
        hardlink = self.root / 'checkpoint_005000_audit_attempt3.json'
        os.link(self.model_path, hardlink)
        before = {p: p.read_bytes() for p in (review, audit, self.model_path)}
        with mock.patch.object(v, 'AUDIT_DIR', self.root):
            for path in (review, audit, symlink, hardlink):
                with self.subTest(path=path), self.assertRaises(v.Invalid):
                    v.write_report(path, dict(requested_step=5000))
            with mock.patch.object(v, 'verify') as verify, mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                v.main(['--run', str(self.run), '--step', '5000', '--out', str(audit)])
            verify.assert_not_called()
        alias = self.root / 'directory_link'
        alias.symlink_to(self.run, target_is_directory=True)
        with mock.patch.object(v, 'AUDIT_DIR', alias), self.assertRaises(v.Invalid):
            v.write_report(alias / 'checkpoint_005000_audit.json', dict(requested_step=5000))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertFalse((self.run / 'checkpoint_005000_audit.json').exists())

    def test_atomic_exclusive_output_creation_wins_over_race(self):
        path = self.root / 'checkpoint_005000_audit.json'
        original = v.validate_output_path
        def racing_validate(target, step):
            validated = original(target, step)
            validated.write_bytes(b'other writer evidence')
            return validated
        with mock.patch.object(v, 'AUDIT_DIR', self.root), \
                mock.patch.object(v, 'validate_output_path', side_effect=racing_validate), \
                self.assertRaises(FileExistsError):
            v.write_report(path, dict(requested_step=5000))
        self.assertEqual(path.read_bytes(), b'other writer evidence')

    def test_cli_requires_explicit_args_and_valid_step(self):
        for args in ([], ['--run', str(self.run), '--out', str(v.REVIEW_PATH)],
                     ['--run', str(self.run), '--step', '1000', '--out', str(v.REVIEW_PATH)]):
            with mock.patch('sys.stderr'), self.assertRaises(SystemExit) as exc:
                v.main(args)
            self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
