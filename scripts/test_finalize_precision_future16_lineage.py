#!/usr/bin/env python3
"""Regression checks for the CPU Future16 lineage finalizer; no physical trials."""
from contextlib import ExitStack
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import finalize_precision_future16_lineage_20261013 as f
from scripts.test_finalize_precision_v4_r2 import summary_fixture


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def plan_fixture(root):
    return dict(training=str(root / 'train'), out=str(root / 'eval'), seeds=list(f.analysis.SEEDS),
        cohorts=[dict(step=step, label='v4', sha256=str(step)) for step in f.analysis.STEPS],
        commands=[dict(checkpoint={'step': step}, seed=seed, command=['run', '--max-steps', '2400'])
            for step in f.analysis.STEPS for seed in f.analysis.SEEDS],
        record_video=False, third_view=False, randomized=False, max_steps=2400,
        export_success_videos=False, future_prediction=deepcopy(f.future.FUTURE_PREDICTION),
        frozen_sources={str(f.queue.RUNTIME): 'frozen-runtime'}, training_audit={})


class Future16FinalizerTests(unittest.TestCase):
    def test_exact_existing_ten_and_budget_required(self):
        root = Path('/test'); plan = plan_fixture(root)
        f.validate_plan(plan, root / 'train', root / 'eval')
        plan['commands'][0]['seed'] = 932106
        with self.assertRaisesRegex(ValueError, 'attempts'):
            f.validate_plan(plan, root / 'train', root / 'eval')

    def test_video_recording_export_and_extra_control_budget_rejected(self):
        root = Path('/test')
        for key, value in [('record_video', True), ('third_view', True), ('export_success_videos', True), ('max_steps', 2401)]:
            plan = plan_fixture(root); plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                f.validate_plan(plan, root / 'train', root / 'eval')

    def test_lineage_is_reused_without_changing_scored_loader(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            training = Path(name)
            stack.enter_context(patch.object(f.future, 'install'))
            for module, field in [(f.queue, 'v4'), (f.queue, 'RUNTIME'), (f.queue, 'validate_binding'),
                (f.queue.e, 'validate_binding'), (f.lineage, '_training'), (f.lineage, '_lineage'),
                (f.future, 'read_contract')]:
                stack.enter_context(patch.object(module, field, getattr(module, field)))
            original = f.lineage._read_contract
            f.configure_lineage(training)
            self.assertIs(f.future.read_contract, f.lineage.read_contract)
            self.assertIs(f.lineage._read_contract, original)
            self.assertEqual(f.lineage._training, training.resolve())
            self.assertIs(f.queue.e.validate_binding, f.lineage.original.validate_binding)

    def test_complete_training_and_lineage_audit_must_equal_admission(self):
        audit = {'status': 'passed', 'updates': 50000}
        witness = {'copied_checkpoint': {'sha256': 'same-bytes'}}
        cohorts = [{'step': 130000}, {'step': 127500}]
        plan = dict(cohorts=cohorts, training_audit=dict(audit, initial_checkpoint_lineage=witness))
        with patch.object(f, 'configure_lineage'), patch.object(f.lineage, '_lineage', witness), \
             patch.object(f.queue, 'audit_training', side_effect=lambda _: ({}, cohorts, dict(audit))):
            self.assertEqual(f.audited_training(Path('/training'), plan)[2], plan['training_audit'])
            plan['training_audit']['initial_checkpoint_lineage'] = {'copied_checkpoint': {'sha256': 'changed'}}
            with self.assertRaisesRegex(ValueError, 'audit differs'):
                f.audited_training(Path('/training'), plan)

    def test_missing_lineage_cannot_be_silently_accepted(self):
        with patch.object(f, 'configure_lineage'), patch.object(f.lineage, '_lineage', None), \
             patch.object(f.queue, 'audit_training', return_value=({}, [], {})):
            with self.assertRaisesRegex(ValueError, 'did not run'):
                f.audited_training(Path('/training'), {})

    def test_waits_for_queue_video_disabled_receipt_after_completed_summary(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); training = root / 'train'; evaluation = root / 'eval'
            save(evaluation / 'summary.json', summary_fixture())
            self.assertIsNone(f.poll_completed(training, evaluation))
            save(evaluation / 'success_videos.json', dict(status='disabled_by_default', videos=[], evaluation_unchanged=True))
            self.assertEqual(f.poll_completed(training, evaluation), summary_fixture())

    def test_technical_failure_or_retained_owner_stops_without_retries(self):
        for guard in ('failure_guard.json', 'owner_retained_guard.json', 'success_video_failure_guard.json'):
            with self.subTest(guard=guard), tempfile.TemporaryDirectory() as name:
                root = Path(name); save(root / 'eval' / guard, {'preserved': True})
                with self.assertRaisesRegex(ValueError, 'Preserved failure guard'):
                    f.poll_completed(root / 'train', root / 'eval')

    def test_halted_summary_is_never_treated_as_completed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); save(root / 'eval/summary.json', {'status': 'halted'})
            with self.assertRaisesRegex(ValueError, 'halted'):
                f.poll_completed(root / 'train', root / 'eval')

    def test_cpu_wait_is_thirty_seconds_and_never_launches_child(self):
        with patch.object(f, 'poll_completed', side_effect=[None, {'status': 'completed'}]), \
             patch.object(f.analysis, 'verify_sources') as verify, patch.object(f.time, 'sleep') as sleep, \
             patch.object(f.queue.subprocess, 'Popen') as child, patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(f.wait_completed(Path('/train'), Path('/eval'), {}), {'status': 'completed'})
        sleep.assert_called_once_with(30); self.assertEqual(verify.call_count, 2); child.assert_not_called()

    def test_disabled_receipt_cannot_hide_an_export(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            save(root / 'success_videos.json', dict(status='disabled_by_default', videos=[{'path': 'unexpected.mp4'}], evaluation_unchanged=True))
            with self.assertRaisesRegex(ValueError, 'disabled video'):
                f.disabled_video_receipt(root)

    def test_independent_classifier_rejects_changed_raw_score(self):
        plan = plan_fixture(Path('/test')); summary = summary_fixture()
        for row in summary['episodes']:
            row['raw_result_path'] = '/raw/result.json'
        with patch.object(f.queue.e, 'classify', return_value={'observed_base_score': 8}), \
             patch.object(f.queue, 'details', side_effect=lambda row: row):
            with self.assertRaisesRegex(ValueError, 'Independent raw admission differs'):
                f.reclassify(plan, summary, {})

    def test_actual_success_is_reported_without_video_or_evaluation_mutation(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            root = Path(name); training = root / 'train'; evaluation = root / 'eval'; out = root / 'out'; out.mkdir()
            plan = plan_fixture(root); summary = summary_fixture()
            summary['episodes'][0].update(observed_base_score=30, full_task_success=True)
            plan['v2_reference'] = {'path': str(root / 'baseline.json')}
            save(root / 'baseline.json', {'episodes': summary_fixture()['episodes'][:5]})
            save(evaluation / 'summary.json', summary)
            save(evaluation / 'success_videos.json', dict(status='disabled_by_default', videos=[], evaluation_unchanged=True))
            training.mkdir()
            metrics = ('loss', 'mse', 'future16', 'future32', 'idm', 'chunk_vtt_loss', 'alpha_after_update', 'seconds_per_step')
            (training / 'log.jsonl').write_text(json.dumps(dict.fromkeys(metrics, 0)) + '\n')
            before = {p.name: p.read_bytes() for p in evaluation.iterdir()}
            stack.enter_context(patch.object(f, 'ROOT', root))
            stack.enter_context(patch.object(f, 'reclassify', return_value=summary['episodes']))
            stack.enter_context(patch.object(f.analysis, 'analyze_motion', side_effect=lambda row: dict(row, failure_stage='before_grasp')))
            stack.enter_context(patch.object(f.baseline_analysis, 'episode'))
            stack.enter_context(patch.object(f.analysis, 'baseline_safe_score'))
            stack.enter_context(patch.object(f.queue, 'verify'))
            stack.enter_context(patch.object(f.analysis, 'verify_sources'))
            exporter = stack.enter_context(patch.object(f.queue, 'export_success'))
            child = stack.enter_context(patch.object(f.queue.subprocess, 'Popen'))
            stack.enter_context(patch('sys.stdout', new=io.StringIO()))
            delivery = f.finalize(training, evaluation, out, plan, summary, {}, plan['cohorts'], {}, {})
            self.assertEqual(delivery['comparison_table'][1]['success_fraction'], '1/5')
            self.assertEqual(delivery['success_videos']['videos'], [])
            self.assertFalse(delivery['protocol']['export_success_videos'])
            self.assertEqual({p.name: p.read_bytes() for p in evaluation.iterdir()}, before)
            self.assertTrue((out / 'completed.json').exists())
            exporter.assert_not_called(); child.assert_not_called()


if __name__ == '__main__':
    unittest.main()
