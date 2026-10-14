#!/usr/bin/env python3
"""Small CPU-only regressions for preparation; no runtime/model imports or dispatch."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_task090_system_triplet as p

CLEAN = {'lock_present': False, 'active_runtime_processes': [], 'unreadable_pids': [],
         'lock_owner': None, 'resource_headroom': 'NOT_CHECKED'}


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read the real small source inventory once; freeze it for deterministic unit cases.
        cls.bundle = p.source_bundle(p.LAUNCH)
        cls.old = p.ROOT / 'outputs/validation/task090_remaining_stages_20261002/triplet_050000/state.json'
        cls.old_digest = hashlib.sha256(cls.old.read_bytes()).hexdigest()
        p.PLAN_ROOT.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='cpu_fixture_', dir=p.PLAN_ROOT)
        cls.root = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        unchanged = hashlib.sha256(cls.old.read_bytes()).hexdigest() == cls.old_digest
        cls.temp.cleanup()
        if not unchanged:
            raise AssertionError('old 50k triplet changed during CPU tests')

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(dir=self.root))
        self.addCleanup(patch.stopall)
        patch.object(p, 'PLAN_ROOT', self.work).start()
        patch.object(p, 'RUN_ROOT', self.work / 'unused_runs').start()
        patch.object(p, 'source_bundle', return_value=copy.deepcopy(self.bundle)).start()
        patch.object(p, 'runtime_snapshot', return_value=copy.deepcopy(CLEAN)).start()
        self.plan = p.build_plan('cpu_v01', self.work / 'planned')

    def inspect(self, plan=None, snapshot=None):
        return p.inspect_plan(self.plan if plan is None else plan, CLEAN if snapshot is None else snapshot)

    def reject(self):
        report = self.inspect()
        self.assertEqual(report['plan_validation'], 'REJECTED')
        self.assertTrue(report['errors'])
        self.assertEqual((report['status'], report['execution']), ('BLOCKED', 'NOT_STARTED'))

    def test_three_unique_predeclared_seeds_runs_units(self):
        rows = self.plan['rounds']
        self.assertEqual([r['seed'] for r in rows], [0, 1, 2])
        for key in ('unit', 'run_dir'):
            self.assertEqual(len({r[key] for r in rows}), 3)
        self.assertEqual(self.plan['completed'], [])
        self.assertTrue(all(r['result'] is None and r['status'] == 'NOT_STARTED' for r in rows))

    def test_same_instance_config_and_wam_parameters(self):
        versions, configs = set(), set()
        for row in self.plan['rounds']:
            argv = row['argv_for_main_thread_review_only']
            versions.add(row['config_version_sha256'])
            configs.add(p.value(argv, '--orchestrate'))
            for flag, expected in (('--profile', 'development'), ('--instances', '10'),
                                   ('--samples', '1'), ('--exec-horizon', '8'), ('--num-steps', '10')):
                self.assertEqual(p.value(argv, flag), expected)
            self.assertEqual(p.value(argv, '--seed'), str(row['seed']))
            self.assertEqual(row['expected_instance_id'], 311)
        self.assertEqual(len(versions), 1)
        self.assertEqual(len(configs), 1)

    def test_original_full_budget_and_both_inside_goals(self):
        protocol = self.plan['protocol']
        self.assertEqual(protocol['original_default_timeout_steps'], int(3638.62 * 1.5))
        self.assertIsNone(protocol['max_steps_override'])
        self.assertEqual(protocol['goals'], p.GOALS)
        self.assertEqual(len(protocol['goals']), 2)
        for row in self.plan['rounds']:
            self.assertNotIn('--max-steps', row['argv_for_main_thread_review_only'])

    def test_auxiliary_disclosure_and_sensor_only_completion(self):
        aux = self.plan['auxiliary']
        self.assertIs(aux['official_score_valid'], False)
        self.assertTrue(all(aux[k] for k in ('privbase', 'preposition', 'assisted_grasp')))
        self.assertIn('SENSOR_ONLY', self.plan['protocol']['online_manipulation_completion'])
        for row in self.plan['rounds']:
            self.assertIn('privbase_preposition_assisted', row['run_dir'])
            self.assertEqual(p.value(row['argv_for_main_thread_review_only'], '--env-wrapper'),
                             'harness.prepositioned_base_rgbd_wrapper.PrepositionedBaseRGBDWrapper')

    def test_no_observation_admission_always_blocks(self):
        report = self.inspect()
        self.assertEqual(report['plan_validation'], 'VALID_PREPARATION_ONLY')
        self.assertEqual((report['status'], report['execution']), ('BLOCKED', 'NOT_STARTED'))
        self.assertEqual(set(report['blocked_on']), set(p.REQUIRED))
        for group in self.plan['main_thread_evidence_required'].values():
            self.assertTrue(all(value is None for value in group['evidence'].values()))

    def test_arbitrary_true_config_never_makes_ready(self):
        altered = copy.deepcopy(self.bundle)
        altered[1].update(observer_admitted=True, completion_gate_admitted=True, ready=True)
        with patch.object(p, 'source_bundle', return_value=altered):
            plan = p.build_plan('cpu_v01', self.work / 'planned')
            report = self.inspect(plan)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertIn('completion_gate', report['blocked_on'])

    def test_fake_evidence_receipt_is_not_admission(self):
        self.plan['main_thread_evidence_required']['observer']['evidence']['producer_source_path'] = 'verified'
        self.reject()

    def test_duplicate_seed_rejected(self):
        self.plan['rounds'][2]['seed'] = 1
        self.reject()

    def test_changed_instance_rejected(self):
        self.plan['rounds'][1]['expected_instance_id'] = 312
        self.reject()

    def test_prefix_budgets_rejected(self):
        for prefix in ('600', '2323'):
            with self.subTest(prefix=prefix):
                altered = copy.deepcopy(self.plan)
                altered['rounds'][1]['argv_for_main_thread_review_only'] += ['--max-steps', prefix]
                self.assertEqual(self.inspect(altered)['plan_validation'], 'REJECTED')

    def test_checkpoint_path_and_digest_changes_rejected(self):
        for key, wrong in (('path', str(p.CHECKPOINT.parent / 'checkpoint_best.pt')), ('sha256', '0' * 64)):
            with self.subTest(key=key):
                altered = copy.deepcopy(self.plan)
                altered['checkpoint'][key] = wrong
                self.assertEqual(self.inspect(altered)['plan_validation'], 'REJECTED')

    def test_config_source_and_runtime_digest_drift_rejected(self):
        altered = copy.deepcopy(self.bundle)
        altered[2][str(p.LAUNCH)] = '0' * 64
        with patch.object(p, 'source_bundle', return_value=altered):
            self.reject()

    def test_unknown_config_source_rejected(self):
        self.plan['config_source'] = str(self.work / 'missing.json')
        self.reject()

    def test_missing_recording_rejected_and_old_framing_not_passed(self):
        self.assertIn('BLOCKED', self.plan['recording']['framing_review'])
        argv = self.plan['rounds'][0]['argv_for_main_thread_review_only']
        for flag, expected in (('--video-width', '1920'), ('--video-height', '1080'), ('--video-view', 'viewer')):
            self.assertEqual(p.value(argv, flag), expected)
        argv.remove('--write-video')
        self.reject()

    def test_serial_resource_and_existing_lock_contract(self):
        resources = self.plan['resource_contract']
        self.assertEqual(resources['cpu_percent_ceiling'], 85)
        self.assertEqual(resources['ram_percent_ceiling'], 85)
        self.assertEqual(resources['total_vram_abort_mib'], 15300)
        for row in self.plan['rounds']:
            argv = row['argv_for_main_thread_review_only']
            self.assertIn(f'--unit={row["unit"]}', argv)
            self.assertTrue(any('gpu_lock.py acquire --owner ' + row['unit'] in arg for arg in argv))
        self.assertIn('only that owner', self.plan['stop_and_release'])

    def test_lock_live_process_or_hidden_pid_blocks(self):
        for key, value in (('lock_present', True), ('active_runtime_processes', [{'pid': 123}]),
                           ('unreadable_pids', [123])):
            with self.subTest(key=key):
                snapshot = {**CLEAN, key: value}
                self.assertEqual(self.inspect(snapshot=snapshot)['plan_validation'], 'REJECTED')

    def test_existing_run_or_dangling_alias_rejected(self):
        run = Path(self.plan['rounds'][0]['run_dir'])
        run.parent.mkdir()
        run.symlink_to(self.work / 'missing_target')
        self.reject()
        run.unlink()
        run.mkdir()
        self.reject()

    def test_existing_output_preserved(self):
        out = Path(self.plan['out'])
        out.mkdir()
        sentinel = out / 'keep.json'
        sentinel.write_text('old result')
        with self.assertRaisesRegex(ValueError, 'existing output'):
            p.write_new_plan(self.plan)
        self.assertEqual(sentinel.read_text(), 'old result')

    def test_same_runs_reserved_by_other_plan_rejected(self):
        old = self.work / 'other'
        old.mkdir()
        (old / 'plan.json').write_text(json.dumps(self.plan))
        self.reject()

    def test_existing_unit_log_rejected(self):
        out = Path(self.plan['out'])
        out.mkdir()
        (out / 'seed1_unit.log').write_text('preserve')
        self.reject()

    def test_output_outside_plan_root_rejected(self):
        with self.assertRaises(ValueError):
            p.output_path(p.OLD_RUN)
        with self.assertRaises(ValueError):
            p.output_path(self.work)

    def test_check_only_is_read_only_and_returns_two(self):
        path = p.write_new_plan(self.plan)
        before = path.read_bytes()
        with patch.object(sys, 'argv', ['prepare', '--check-only', str(path)]), contextlib.redirect_stdout(io.StringIO()) as out:
            code = p.main()
        report = json.loads(out.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(report['plan_validation'], 'VALID_PREPARATION_ONLY')
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(any(Path(r['run_dir']).exists() for r in self.plan['rounds']))

    def test_old_three_test_evidence_not_overwritten(self):
        self.assertEqual(hashlib.sha256(self.old.read_bytes()).hexdigest(), self.old_digest)
        self.assertNotIn(str(self.old.parent), json.dumps([r['run_dir'] for r in self.plan['rounds']]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
