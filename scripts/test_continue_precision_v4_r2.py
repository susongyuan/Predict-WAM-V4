#!/usr/bin/env python3
"""CPU tests of continuation admission, ownership and no-retry boundaries."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import continue_precision_v4_r2 as s


SCHEDULE = dict(s.SCHEDULE, predictor_steps=2, gate_steps=4, steps=8)


def rows():
    result = []
    for update in range(1, 7):
        phase = 'predictor' if update <= 2 else 'gate' if update <= 4 else 'joint'
        active = phase != 'predictor'
        joint = phase == 'joint'
        row = dict(updates=update, step=80000 + update, phase=phase,
            loss=.15, mse=.01, future8=.1, future32=.1, idm=.02,
            chunk_vtt_loss=.3, chunk_huber=.2, chunk_cosine=1.,
            alpha_after_update=.01 if active else 0., gradient_norm=.1,
            predictor_grad_norm=.1, projection_grad_norm=.03 if active else 0.,
            alpha_grad_norm=.04 if active else 0., backbone_grad_norm=.02 if joint else 0.,
            seconds_per_step=1., cuda_allocated_bytes=100, cuda_reserved_bytes=200,
            cuda_peak_allocated_bytes=200, chunk_vtt_valid_samples=4,
            learning_rates=dict(backbone=1e-6 if joint else 0., predictor=1e-4,
                projection=1e-4 if active else 0., alpha=1e-4 if active else 0.))
        result.append(row)
    return result


def diagnostic(chunk=.8, mse=.01):
    data = {}
    for split in ('train', 'val'):
        values = dict(loss=.15, mse=mse, future8=.1, future32=.1, idm=.02,
                      chunk_vtt_loss=chunk, chunk_huber=chunk - .1, chunk_cosine=1.)
        data[split] = dict(count=8, means=values,
            rows=[dict(index=i * 100, episode=f'{split}_{i}.hdf5', anchor=i * 10, **values)
                  for i in range(8)])
    return data


def stopped_status():
    return dict(status='stopped_at_safe_boundary', updates=2250, step=82250,
                target_updates=50000, target_step=130000)


class SupervisorTests(unittest.TestCase):
    def test_full_phase_sequence_requires_learning_in_all_new_modules(self):
        summaries = s.audit_rows(rows(), SCHEDULE, 6)
        self.assertEqual(summaries['predictor']['updates'], 2)
        self.assertEqual(summaries['gate']['backbone_positive'], 0)
        self.assertEqual(summaries['joint']['alpha_nonzero'], 2)

    def test_gap_nonfinite_freeze_violation_and_missing_gradient_rejected(self):
        mutations = ((0, 'updates', 2), (2, 'loss', float('nan')),
            (1, 'backbone_grad_norm', .01), (0, 'alpha_after_update', .01),
            (3, 'predictor_grad_norm', 0), (5, 'phase', 'gate'))
        for index, key, value in mutations:
            with self.subTest(key=key):
                bad = rows()
                bad[index][key] = value
                with self.assertRaises(ValueError):
                    s.audit_rows(bad, SCHEDULE, 6)
        with self.assertRaises(ValueError):
            s.audit_rows(rows()[:-1], SCHEDULE, 6)

    def test_gate_cannot_appear_enabled_without_actual_alpha_learning(self):
        bad = rows()
        for row in bad:
            row['alpha_grad_norm'] = 0
        with self.assertRaisesRegex(ValueError, 'Fusion did not actually learn'):
            s.audit_rows(bad, SCHEDULE, 6)

    def test_fixed_anchor_validation_improves_chunk_and_allows_small_mse_drift(self):
        report = s.compare_diagnostics(diagnostic(), diagnostic(chunk=.6, mse=.011))
        self.assertAlmostEqual(report['splits']['val']['chunk_loss_ratio'], .75)
        self.assertAlmostEqual(report['splits']['val']['action_mse_ratio'], 1.1)
        self.assertIn('Offline stability', report['interpretation'])

    def test_chunk_nonimprovement_or_severe_action_regression_halts(self):
        for chunk, mse in ((.8, .01), (.9, .01), (.6, .0126)):
            with self.subTest(chunk=chunk, mse=mse), self.assertRaises(ValueError):
                s.compare_diagnostics(diagnostic(), diagnostic(chunk=chunk, mse=mse))

    def test_changed_anchor_or_fabricated_mean_halts(self):
        bad = diagnostic(chunk=.6)
        bad['val']['rows'][3]['anchor'] += 1
        with self.assertRaisesRegex(ValueError, 'anchors changed'):
            s.compare_diagnostics(diagnostic(), bad)
        bad = diagnostic(chunk=.6)
        bad['val']['means']['chunk_vtt_loss'] = .5
        with self.assertRaisesRegex(ValueError, 'mean does not match'):
            s.compare_diagnostics(diagnostic(), bad)

    def test_service_wait_is_cpu_only_at_30s_and_checks_actual_exit(self):
        stopped = dict(ActiveState='inactive', MainPID='0', ExecMainStatus='0', Result='success')
        active = dict(ActiveState='active', MainPID='1234', ExecMainStatus='0', Result='success')
        with patch.object(s, 'check_no_training_failure', side_effect=[dict(status='training', updates=100), stopped_status()]), \
             patch.object(s, 'short_service_state', side_effect=[active, stopped]), \
             patch.object(s.time, 'sleep') as sleep, patch.object(s, 'journal'):
            self.assertEqual(s.wait_short(Path('unused')), stopped)
            sleep.assert_called_once_with(30)
        with patch.object(s, 'check_no_training_failure', return_value=stopped_status()), \
             patch.object(s, 'short_service_state', return_value=dict(stopped, ExecMainStatus='1')), \
             self.assertRaisesRegex(ValueError, 'did not exit successfully'):
            s.wait_short(Path('unused'))

    def test_wrong_safe_boundary_and_existing_failure_guard_halt(self):
        for change in (dict(updates=2249), dict(step=82251), dict(target_updates=30000)):
            with self.assertRaises(ValueError):
                s.validate_short_status(dict(stopped_status(), **change))
        with tempfile.TemporaryDirectory() as directory:
            training = Path(directory)
            (training / 'failure_guard.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'failure guard exists'):
                s.check_no_training_failure(training)

    def test_single_attempt_child_restores_cuda_environment_and_receipts(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            run = Path(directory)
            command = [sys.executable, '-c', 'import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"']
            self.assertEqual(s.run_child(command, 'test', run, {}), 0)
            self.assertEqual(s.read_json(run / 'test_exit.json')['returncode'], 0)
            with self.assertRaises(FileExistsError):
                s.run_child(command, 'test', run, {})

    def test_nonzero_child_exit_is_preserved_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with self.assertRaisesRegex(ValueError, 'do not retry'):
                s.run_child([sys.executable, '-c', 'raise SystemExit(4)'], 'test', run, {})
            self.assertEqual(s.read_json(run / 'test_exit.json')['returncode'], 4)
            self.assertEqual(len(list(run.glob('*attempt.json'))), 1)

    def test_resume_command_keeps_schedule_owner_and_evaluation_has_no_wait_or_retry(self):
        train, evaluation = s.commands()
        self.assertNotIn('--max-updates', train)
        self.assertIn('--resume', train)
        self.assertEqual(train[train.index('--owner') + 1], s.TRAIN_OWNER)
        self.assertEqual(train[train.index('--steps') + 1], '50000')
        self.assertIn(str(s.PYTHON), evaluation)
        self.assertNotIn('--wait-training', evaluation)
        self.assertIn(str(s.EVALUATION), evaluation)

    def test_immutable_audit_and_source_hash_preserve_original_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.json'
            s.write_once(path, dict(status='passed'))
            sources = {str(path): s.digest_file(path)}
            s.verify_sources(sources)
            with self.assertRaises(FileExistsError):
                s.write_once(path, dict(status='failed'))
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                s.verify_sources(sources)

    def test_optimizer_update_counts_follow_actual_freeze_boundaries(self):
        import torch
        optimizer = dict(param_groups=[], state={})
        for index, (name, count) in enumerate((('backbone', 250), ('predictor', 2250),
                ('projection', 1250), ('alpha', 1250))):
            rate = 1e-6 if name == 'backbone' else 1e-4
            optimizer['param_groups'].append(dict(name=name, base_lr=rate, params=[index]))
            optimizer['state'][index] = dict(step=torch.tensor(float(count)),
                exp_avg=torch.zeros(2), exp_avg_sq=torch.ones(2))
        report = s.audit_optimizer(optimizer)
        self.assertEqual(report['projection']['maximum_step'], 1250)
        self.assertEqual(report['backbone']['maximum_step'], 250)
        bad = copy.deepcopy(optimizer)
        bad['state'][1]['step'] = torch.tensor(2000.)
        with self.assertRaisesRegex(ValueError, 'skipped enabled updates'):
            s.audit_optimizer(bad)
        bad = copy.deepcopy(optimizer)
        bad['state'][0]['step'] = torch.tensor(251.)
        with self.assertRaisesRegex(ValueError, 'enabled phase'):
            s.audit_optimizer(bad)
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_2000_boundary_proves_original_backbone_frozen_bitwise(self):
        import torch
        initial = dict(model_state_dict={'backbone.weight': torch.tensor([1., 2.]),
            'chunk_vtt16.alpha': torch.tensor(0.), 'chunk_vtt16.predictor.0.bias': torch.tensor(0.)})
        gate = copy.deepcopy(initial)
        gate['model_state_dict']['chunk_vtt16.alpha'] = torch.tensor(.001)
        gate['model_state_dict']['chunk_vtt16.predictor.0.bias'] = torch.tensor(.1)
        report = s.frozen_backbone_identity(initial, gate)
        self.assertEqual(report['backbone_tensors_bitwise_equal'], 1)
        gate['model_state_dict']['backbone.weight'][0] += .001
        with self.assertRaisesRegex(ValueError, 'changed while frozen'):
            s.frozen_backbone_identity(initial, gate)
        self.assertFalse(torch.cuda.is_initialized())

    def test_latest_heartbeat_replaces_status_but_preserves_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            s.write_once(run / 'audit.json', dict(status='passed'))
            s.latest(run, 'waiting_short')
            s.latest(run, 'waiting_short', dict(status='training', phase='gate', updates=1200))
            waiting = s.read_json(run / 'latest.json')
            self.assertEqual(waiting['phase'], 'waiting_short')
            self.assertEqual(waiting['details']['phase'], 'gate')
            s.latest(run, 'training', dict(updates=2250))
            self.assertEqual(s.read_json(run / 'latest.json')['phase'], 'training')
            self.assertEqual(s.read_json(run / 'audit.json'), dict(status='passed'))


if __name__ == '__main__':
    unittest.main()
