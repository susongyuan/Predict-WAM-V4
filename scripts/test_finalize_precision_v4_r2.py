#!/usr/bin/env python3
"""Focused CPU checks; no model, simulator, GPU or real documentation writes."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import finalize_precision_v4_r2 as f


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def summary_fixture():
    rows = [dict(checkpoint_step=step, seed=seed, observed_base_score=0,
                 full_task_success=False) for step in f.STEPS for seed in f.SEEDS]
    return dict(status='completed', planned_episodes=10, seeds=list(f.SEEDS),
                record_video=False, third_view=False, randomized=False,
                cohorts=[dict(checkpoint={'step': step}) for step in f.STEPS], episodes=rows)


class FinalizerTests(unittest.TestCase):
    def test_same_seed_twice_is_grouped_by_checkpoint_and_duplicate_rejected(self):
        s = summary_fixture()
        groups = f.cohort_rows(s)
        self.assertEqual(list(groups), [130000, 127500])
        self.assertEqual([r['seed'] for r in groups[127500]], list(f.SEEDS))
        s['episodes'][5]['checkpoint_step'] = 130000
        with self.assertRaisesRegex(ValueError, 'cohort order'):
            f.cohort_rows(s)

    def test_incomplete_and_wrong_checkpoint_cohorts_rejected(self):
        for field, value in [('status', 'evaluating'), ('planned_episodes', 5), ('record_video', True)]:
            s = summary_fixture(); s[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                f.cohort_rows(s)
        s = summary_fixture(); s['cohorts'][1]['checkpoint']['step'] = 125000
        with self.assertRaises(ValueError):
            f.cohort_rows(s)

    def test_path_distinguishes_still_reversal_and_straight_approach(self):
        straight = f.path_metrics([[0, 0], [1, 0], [2, 0]])
        self.assertEqual(straight['net_displacement'], 2)
        self.assertEqual(straight['cancellation_ratio'], 0)
        reverse = f.path_metrics([[0, 0], [1, 0], [0, 0]])
        self.assertEqual(reverse['path_length'], 2)
        self.assertEqual(reverse['cancellation_ratio'], 1)
        still = f.path_metrics([[0, 0], [0, 0]])
        self.assertTrue(still['stationary']); self.assertIsNone(still['cancellation_ratio'])

    def test_partial_exec_prefix_uses_pre_action_measurement(self):
        import numpy as np
        fk = SimpleNamespace(fk=lambda side, q: np.array([[1, 0, 0, q[0]], [0, 1, 0, 0],
                                                        [0, 0, 1, 0], [0, 0, 0, 1]]))
        q = np.zeros((4, 16)); q[:, 0] = [0, 1, 0, 1]
        q[:, 8] = [0, .5, 1, 1.5]
        actions = [dict(control_step=i, policy={'replanned': i == 0}) for i in range(3)]
        rules = [dict(objects={key: {'world_corners_m': [[2, 0, 0], [3, 1, 1]]}
                              for key in ('bowl', 'spoon', 'cup', 'plate')}) for i in range(4)]
        got = f.chunk_approach(actions, q, rules, fk)
        c = got['executed_chunks'][0]
        self.assertEqual(c['executed_steps'], 3)
        self.assertEqual(c['measured_tool_m']['left']['path_length'], 3)
        self.assertEqual(c['measured_tool_m']['left']['net_displacement'], 1)
        self.assertAlmostEqual(c['object_approach_distance_reduction_m']['right']['bowl'], 1.5)

    def test_guard_and_stale_heartbeat_halt_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); s = root / 'new'; t = root / 'train'; e = root / 'eval'
            save(s / 'latest.json', dict(phase='training', unix_seconds=1000))
            self.assertFalse(f.poll_state(s, t, e, now=1100))
            with self.assertRaisesRegex(ValueError, 'stale'):
                f.poll_state(s, t, e, now=2000)
            save(t / 'failure_guard.json', {'error': 'preserved'})
            with self.assertRaisesRegex(ValueError, 'failure guard'):
                f.poll_state(s, t, e, now=1100)

    def test_new_completion_required_old_guard_is_preserved_and_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); s = root / 'new'; old = root / 'old'; e = root / 'eval'; t = root / 'train'
            save(old / 'failure_guard.json', {'error': 'KeyboardInterrupt: reboot'})
            save(old / 'completed.json', {'status': 'completed'})
            save(s / 'latest.json', {'phase': 'training', 'unix_seconds': 1000})
            save(e / 'summary.json', summary_fixture())
            self.assertFalse(f.poll_state(s, t, e, now=1100))
            save(s / 'completed.json', dict(status='completed', physical_retries=0,
                evaluation_summary=str(e / 'summary.json'), evaluation_summary_sha256=f.sha(e / 'summary.json')))
            self.assertTrue(f.poll_state(s, t, e, now=1100))
            self.assertTrue((old / 'failure_guard.json').exists())
            save(e / 'summary.json', {'status': 'changed'})
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                f.poll_state(s, t, e, now=1100)

    def test_wait_poll_is_30seconds_and_never_launches_a_child(self):
        with patch.object(f, 'poll_state', side_effect=[False, True]), \
             patch.object(f, 'verify_sources') as verify, patch.object(f.time, 'sleep') as sleep:
            f.wait_completed(Path('/new'), Path('/train'), Path('/eval'), Path('/out'), {}, 900)
        sleep.assert_called_once_with(30); verify.assert_called_once_with({})

    def test_equal_success_with_more_progress_is_not_called_task_success(self):
        rows = summary_fixture()['episodes']
        baseline = f.table_row('V2', 80000, rows[:5])
        improved = deepcopy(rows[:5]); improved[0]['observed_base_score'] = 8
        final = f.table_row('V4', 130000, improved)
        got = f.verdict([baseline, final, baseline])
        self.assertIn('成功次数未增加', got['conclusion'])
        self.assertIn('不能称已解决整任务', got['conclusion'])
        self.assertEqual(final['success_fraction'], '0/5')
        self.assertAlmostEqual(final['mean_task_progress'], 8 / 150)

    def test_later_checkpoint_regression_recommends_analysis_not_autotrain(self):
        rows = summary_fixture()['episodes'][:5]
        b = f.table_row('V2', 80000, rows)
        final = f.table_row('V4', 130000, rows)
        older_rows = deepcopy(rows); older_rows[0].update(observed_base_score=30, full_task_success=True)
        older = f.table_row('V4', 127500, older_rows)
        self.assertIn('不自动延训', f.verdict([b, final, older])['further_training_recommendation'])

    def test_no_success_video_receipt_cannot_hide_partial_or_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / 'success_videos.json', dict(status='completed', evaluation_unchanged=True,
                                                  videos=[], no_full_task_success=True))
            rows = summary_fixture()['episodes']
            self.assertTrue(f.verify_videos(root, rows)['no_full_task_success'])
            rows[0]['full_task_success'] = True
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                f.verify_videos(root, rows)

    def test_success_video_hash_and_raw_source_binding_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); evaluation = root / 'eval'; path = root / 'success video' / 'success.mp4'
            path.parent.mkdir(); path.write_bytes(b'fake immutable exporter bytes, not a simulated trial')
            row = dict(checkpoint_step=130000, seed=f.SEEDS[0], full_task_success=True,
                       raw_result_path='/raw/result.json', raw_result_sha256='a' * 64, control_steps=8)
            video = dict(path=str(path), checkpoint_step=130000, seed=f.SEEDS[0], sha256=f.sha(path),
                         full_task_success=True, source_result=row['raw_result_path'],
                         source_result_sha256=row['raw_result_sha256'], frames=9, fps=10, duration_seconds=.9,
                         simulator_attempts_added=0, external_cameras_added=0, playback_speed=1)
            save(evaluation / 'success_videos.json', dict(status='completed', evaluation_unchanged=True,
                                                        videos=[video], no_full_task_success=False))
            save(evaluation / 'success_video_evidence' / (path.stem + '.json'), video)
            with patch.object(f, 'ROOT', root):
                self.assertTrue(f.verify_videos(evaluation, [row])['video_bytes_and_source_receipts_verified'])
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    f.verify_videos(evaluation, [row])

    def test_baseline_safe_score_rejects_copied_alias_and_unsafe_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'rule_evidence.jsonl'
            rules = [{'score': {'base_score': 0, 'full_task_success': False}},
                     {'score': {'base_score': 30, 'full_task_success': True}}]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rules))
            row = dict(status='physics_safety_abort', raw_result_path=str(root / 'result.json'),
                       observed_base_score=0, full_task_success=False,
                       evidence_files={'rule_evidence.jsonl': {'sha256': f.sha(path)}})
            self.assertFalse(f.baseline_safe_score(row)['full_task_success'])
            row.update(observed_base_score=30, full_task_success=True)
            with self.assertRaisesRegex(ValueError, 'safe terminal'):
                f.baseline_safe_score(row)

    def test_actual_bound_archive_byte_drift_is_rejected(self):
        from scripts.analyze_precision_v3_eval_v42 import binding_read
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'onboard/observation_000001.npz'; path.parent.mkdir()
            path.write_bytes(b'original')
            row = {'evidence_files': {'onboard/observation_000001.npz':
                    {'path': str(path), 'size': 8, 'sha256': f.sha(path)}}}
            self.assertEqual(binding_read(root, 'onboard/observation_000001.npz', row), b'original')
            path.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                binding_read(root, 'onboard/observation_000001.npz', row)

    def test_pinned_kinematic_dependency_drift_halts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kinematic_chain.json'; path.write_text('{"asset": "official"}')
            bindings = {str(path): f.sha(path)}; f.verify_sources(bindings)
            path.write_text('{"asset": "changed"}')
            with self.assertRaisesRegex(ValueError, 'Source drift'):
                f.verify_sources(bindings)

    def test_unsafe_terminal_is_excluded_before_closure_geometry_and_motion(self):
        import numpy as np
        from scripts import analyze_precision_v3_eval_v42 as a
        from harness import isaac_zhuhai_expert as expert
        rules = [dict(score={'base_score': 0, 'full_task_success': False}) for _ in range(4)]
        actions = [dict(control_step=i, policy={'replanned': i == 0}) for i in range(3)]
        q = np.zeros((4, 16)); sent = np.zeros((3, 16))
        row = dict(status='physics_safety_abort', checkpoint_step=130000, seed=f.SEEDS[0],
                   raw_result_sha256='a' * 64, observed_base_score=0, full_task_success=False)
        with ExitStack() as stack:
            stack.enter_context(patch.object(a, 'load_episode', return_value=({}, actions, rules, rules[:-1], q, q, sent)))
            stack.enter_context(patch.object(a, 'object_progress', return_value={'objects': {}, 'failure_stage': 'before_grasp'}))
            closure = stack.enter_context(patch.object(a, 'closure_analysis', return_value={}))
            geometry = stack.enter_context(patch.object(a, 'geometry_and_activity', return_value=({}, {}, {})))
            motion = stack.enter_context(patch.object(a, 'command_motion', return_value={}))
            chunks = stack.enter_context(patch.object(f, 'chunk_approach', return_value={}))
            stack.enter_context(patch.object(expert, 'FrankaFK', return_value='cpu_fk'))
            got = f.analyze_motion(row)
        self.assertEqual(got['safe_executed_steps'], 2)
        self.assertTrue(got['unsafe_terminal_excluded'])
        self.assertEqual(len(closure.call_args.args[0]), 2)
        self.assertEqual(len(geometry.call_args.args[0]), 3)
        self.assertEqual(len(motion.call_args.args[1]), 3)
        self.assertEqual(len(chunks.call_args.args[0]), 2)

    def test_missing_document_anchor_stops_and_prior_history_stays_intact(self):
        old = '# project2\n## 最新V4任务交接（old）\nold active\n\n## 历史\npreserved\n'
        new = f.documentation_content(old, 'progress1.md', 'actual 0/5', 'receipt.json')
        self.assertIn('actual 0/5', new); self.assertIn('## 历史\npreserved', new)
        with self.assertRaises(ValueError):
            f.documentation_content('no marker', 'progress1.md', 'actual', 'receipt')

    def test_document_publication_backs_up_all_existing_docs_before_edits(self):
        import zipfile
        from scripts import progress1_log_v41 as logger
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); verify = root / 'verify'; verify.mkdir(); out = verify / 'delivery'; out.mkdir()
            fixtures = {'progress1.md': '# project\n## 最新V4任务交接（old）\nold\n## 历史\nkeep\n',
                'docs/current_status.md': '# 当前状态：old\n## 最新V4授权与执行\nold\n## 历史\nkeep\n',
                'README.md': '# 最新执行：old\nold\n\n# VT-WAM\nkeep\n', 'progress.md': '# old\nkeep\n'}
            for rel, text in fixtures.items():
                path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
            archive = root / 'docs/archive/documentation_snapshot_20261007.zip'; archive.parent.mkdir()
            with zipfile.ZipFile(archive, 'w') as z: z.writestr('prior', 'preserved')
            rows = summary_fixture()['episodes'][:5]
            table = [f.table_row('V2', 80000, rows), f.table_row('V4', 130000, rows), f.table_row('V4', 127500, rows)]
            delivery = dict(comparison_table=table, verdict=f.verdict(table))
            with patch.object(f, 'ROOT', root), patch.object(f, 'VERIFY', verify), patch.object(logger, 'log', return_value=False) as log:
                got = f.update_documents(delivery, out)
            with zipfile.ZipFile(archive) as z:
                for rel, text in fixtures.items():
                    self.assertEqual(z.read(got['backup_prefix'] + '/' + rel).decode(), text)
                self.assertEqual(z.read('prior'), b'preserved')
            self.assertIn('0/5', (root / 'docs/current_status.md').read_text())
            self.assertLess(len((root / 'progress1.md').read_text().splitlines()), 200)
            log.assert_called_once(); self.assertFalse(got['logger_published_1800s_summary'])

    def test_independent_final_admission_reclassifies_all_ten_and_keeps_same_seed_rows(self):
        from scripts import eval_precision_v4_final5 as v
        from scripts import analyze_precision_dagger_milestones_v19 as old
        from harness.zhuhai_robot_spec import ASSET, ASSET_MANIFEST
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory); training = root / 'train'; evaluation = root / 'eval'; supervisor = root / 'new'
            verify = root / 'verify'; baseline_path = root / 'baseline.json'; original = root / 'original80k.pt'
            original.write_bytes(b'protected original')
            cohorts = [dict(path=str(training / f'checkpoint_step_{step:06d}.pt'), step=step,
                            sha256=str(i + 1) * 64, size=100, label='v4') for i, step in enumerate(f.STEPS)]
            rows = []; fresh = {}
            for ck in cohorts:
                for seed in f.SEEDS:
                    path = root / f"{ck['step']}_{seed}/result.json"
                    value = dict(seed=seed, observed_base_score=0, full_task_success=False, task_progress=0,
                                 raw_result_path=str(path), raw_result_sha256='c' * 64, status='completed')
                    fresh[(ck['step'], str(path))] = value
                    rows.append(dict(value, checkpoint_step=ck['step'], checkpoint_sha256=ck['sha256'], cohort='v4'))
            plan = dict(schema='precision-v4-final-paired5-v1', cohorts=cohorts,
                        v2_reference={'path': str(baseline_path)}, interpretation='fixed paired10',
                        frozen_sources={str(v.RUNTIME): 'd' * 64})
            saved = v.summary(plan, rows, 'completed')
            save(evaluation / 'plan.json', plan); save(evaluation / 'summary.json', saved)
            baseline_rows = []
            for seed in f.SEEDS:
                path = root / f'baseline_{seed}/result.json'
                save(path, {'policy_ready': {'runtime': {'checkpoint_step': 80000, 'checkpoint_sha256': 'a' * 64}}})
                baseline_rows.append(dict(seed=seed, raw_result_path=str(path), observed_base_score=0, full_task_success=False))
            save(baseline_path, dict(status='completed', seeds=list(f.SEEDS), checkpoint_step=80000,
                evaluation_version=saved['evaluation_version'], checkpoint={'path': str(original)},
                checkpoint_sha256='a' * 64, episodes=baseline_rows, full_successes=0))
            contract = dict(robot_contract={'asset_sha256': f.sha(ASSET), 'asset_manifest_sha256': f.sha(ASSET_MANIFEST)},
                precision_v4={}, v4_training={'initialization': {'path': str(original), 'sha256': f.sha(original)}})
            save(supervisor / 'plan.json', {'training_command': ['train'], 'evaluation_command': ['eval']})
            short_path = verify / 'r2_continuation_supervisor/short_admission_audit.json'
            save(short_path, {'status': 'passed', 'actual': 'original short audit'})
            save(supervisor / 'resume_admission.json', {'status': 'passed', 'short_admission_sha256': f.sha(short_path)})
            (supervisor / 'completed.json').write_text('{}')
            save(training / 'build_notes.json', {'added_parameters': 7877377})
            fields = ('mse', 'future8', 'future32', 'idm', 'chunk_vtt_loss', 'chunk_huber', 'chunk_cosine',
                      'alpha_after_update', 'predictor_grad_norm', 'backbone_grad_norm', 'seconds_per_step')
            (training / 'log.jsonl').write_text(json.dumps(dict.fromkeys(fields, 0)) + '\n')
            save(training / 'diagnostic_130000.json', {'actual': 'final fixed anchors'})
            stack.enter_context(patch.object(f, 'VERIFY', verify))
            stack.enter_context(patch.object(v, 'verify'))
            stack.enter_context(patch.object(v, 'audit_training', return_value=(contract, cohorts, {'status': 'passed'})))
            stack.enter_context(patch.object(v, 'receipt', return_value={'step': 80000, 'sha256': 'a' * 64}))
            classify = stack.enter_context(patch.object(v.e, 'classify', side_effect=lambda path, ck, *args: deepcopy(fresh[(ck['step'], str(path))])))
            stack.enter_context(patch.object(v, 'details', side_effect=lambda row: row))
            stack.enter_context(patch.object(old, 'episode', return_value={}))
            stack.enter_context(patch.object(f, 'baseline_safe_score', return_value={}))
            stack.enter_context(patch.object(f, 'verify_videos', return_value={'no_full_task_success': True}))
            stack.enter_context(patch.object(f, 'analyze_motion', side_effect=lambda row: dict(row,
                failure_stage='before_first_verified_grasp', safe_executed_steps=8, unsafe_terminal_excluded=False, objects={})))
            delivery, motion = f.independent_evidence(training, evaluation, supervisor)
            self.assertEqual(classify.call_count, 10)
            self.assertEqual([(r['checkpoint_step'], r['seed']) for r in motion['episodes']],
                             [(step, seed) for step in f.STEPS for seed in f.SEEDS])
            self.assertEqual(delivery['training']['short_audit']['actual'], 'original short audit')
            self.assertEqual([r['success_fraction'] for r in delivery['comparison_table']], ['0/5'] * 3)
            altered = deepcopy(saved); altered['episodes'][0]['observed_base_score'] = 8
            save(evaluation / 'summary.json', altered)
            with self.assertRaisesRegex(ValueError, 'Independent episode admission differs'):
                f.independent_evidence(training, evaluation, supervisor)

    def test_default_cli_is_plan_only_no_output_writes(self):
        with patch.object(f, 'execute') as execute, patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(f.main([]), 0)
        execute.assert_not_called()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinalizerTests))
    import torch
    assert not torch.cuda.is_initialized(), 'CPU tests initialized CUDA'
    if args.out:
        f.write_once(args.out, dict(status='passed' if result.wasSuccessful() else 'failed', time=f.stamp(),
            tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), cuda_initialized=False,
            source_sha256={str(Path(f.__file__).resolve()): f.sha(f.__file__), str(Path(__file__).resolve()): f.sha(__file__)},
            analysis_plan_sha256=f.sha(f.ANALYSIS_PLAN)))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
