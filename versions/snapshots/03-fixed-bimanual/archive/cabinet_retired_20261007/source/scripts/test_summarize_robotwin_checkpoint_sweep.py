#!/usr/bin/env python3
"""Independent CPU fixtures; no policy, simulator, actual sweep or external data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location('sweep_summary', Path(__file__).with_name('summarize_robotwin_checkpoint_sweep.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_fixture(self, name, seed=10000, ck=30000, cap=700, status='native_success', steps=40,
                    native=True, strict=None, parent='completed'):
        run = self.root / name
        ep_name = f'episode_000_seed{seed}'
        ep = {'episode': ep_name, 'seed': seed, 'status': status, 'control_steps': steps,
              'wall_seconds': 20., 'native_step_limit': 700,
              'evidence': {'native_benchmark_success': native, **(strict or {})}}
        write(run / 'plan.json', {'task': 'put_object_cabinet', 'seeds': [seed],
                                 'checkpoint_receipt': {'step': ck, 'sha256': f'weight{ck}'},
                                 'limits': {'control_steps': cap, 'settle_physics_steps': 120},
                                 'renderer': {'denoiser': 'none'}})
        write(run / 'result.json', {'status': parent, 'episodes': [ep]})
        write(run / ep_name / 'episode.json', ep)
        return run, run / ep_name

    def test_complete_triplet_has_rate(self):
        runs = [self.run_fixture(str(s), seed=s)[0] for s in M.DEFAULT_SEEDS]
        r = M.summarize(runs)
        self.assertEqual(len(r['groups']), 1)
        self.assertEqual(r['groups'][0]['native_success_rate'], 1)

    def test_single_seed_default_rate_null(self):
        run, _ = self.run_fixture('one')
        r = M.summarize([run])
        self.assertIsNone(r['groups'][0]['native_success_rate'])
        self.assertEqual(r['groups'][0]['missing_valid_seeds'], [10001, 10002])

    def test_500_never_mixed_with_700(self):
        a, _ = self.run_fixture('a', cap=500, status='step_limit', steps=500, native=False)
        b, _ = self.run_fixture('b', cap=700, status='step_limit', steps=700, native=False)
        r = M.summarize([a, b], expected_seeds=[10000])
        self.assertEqual(len(r['groups']), 2)
        self.assertEqual({g['protocol']['control_step_limit'] for g in r['groups']}, {500, 700})
        self.assertTrue(all(g['native_success_rate'] == 0 for g in r['groups']))

    def test_retries_all_retained_not_best(self):
        a, _ = self.run_fixture('fail', status='step_limit', steps=700, native=False)
        b, _ = self.run_fixture('success')
        r = M.summarize([a, b], expected_seeds=[10000])
        self.assertEqual(len(r['attempts']), 2)
        self.assertEqual(r['groups'][0]['native_success_rate'], .5)
        self.assertEqual(len(r['groups'][0]['by_seed'][0]['attempt_ids']), 2)

    def test_infra_latch_not_success_and_blocks_rate(self):
        a, _ = self.run_fixture('infra', status='infrastructure_error', native=True)
        b, _ = self.run_fixture('success')
        r = M.summarize([a, b], expected_seeds=[10000])
        self.assertIsNone(r['attempts'][0]['native_success'])
        self.assertIsNone(r['groups'][0]['native_success_rate'])

    def test_parent_later_failure_does_not_erase_finished_episode(self):
        a, _ = self.run_fixture('a', status='step_limit', steps=500, cap=500, native=False, parent='infrastructure_error')
        self.assertEqual(M.summarize([a])['attempts'][0]['category'], 'valid_step_limit')

    def test_unknown_strict_evidence_remains_null(self):
        a, _ = self.run_fixture('a', strict={'stable_after_settle': True, 'release_verified': True})
        row = M.summarize([a])['attempts'][0]
        self.assertTrue(row['native_success'])
        self.assertIsNone(row['strict_evidence']['full_containment'])
        self.assertIsNone(row['stable_released_contained'])

    def test_unstable_native_success_is_not_strict(self):
        a, _ = self.run_fixture('a', strict={'stable_after_settle': False, 'release_verified': True})
        r = M.summarize([a])
        self.assertFalse(r['attempts'][0]['stable_released_contained'])
        self.assertIn('does not establish stable', M.markdown(r))

    def test_invalid_step_limit_short_of_cap(self):
        a, _ = self.run_fixture('a', status='step_limit', steps=500, cap=700, native=False)
        self.assertEqual(M.summarize([a])['attempts'][0]['category'], 'invalid_record')

    def test_missing_plan_cannot_validate_terminal(self):
        a, _ = self.run_fixture('a')
        (a / 'plan.json').unlink()
        self.assertFalse(M.summarize([a])['attempts'][0]['valid_native_outcome'])

    def test_queue_missing_jobs_and_references(self):
        ref, _ = self.run_fixture('ref', seed=10002)
        q = self.root / 'queue.json'
        write(q, {'jobs': [{'checkpoint_step': 15000, 'seed': 10000, 'run_dir': 'not-created', 'unit': 'u'}],
                  'reference_runs': [str(ref)]})
        r = M.summarize(queue_manifest=q)
        self.assertEqual(len(r['attempts']), 2)
        primary = next(a for a in r['attempts'] if a['role'] == 'primary')
        self.assertEqual(primary['category'], 'not_started')
        self.assertEqual(primary['checkpoint_step'], 15000)
        self.assertEqual(primary['unit'], 'u')
        self.assertTrue(all(g['native_success_rate'] is None for g in r['groups']))

    def test_same_physical_directory_is_not_extra_retry(self):
        a, _ = self.run_fixture('a')
        q = self.root / 'queue.json'
        write(q, {'jobs': [{'run_dir': str(a), 'seed': 10000, 'checkpoint_step': 30000}], 'reference_runs': [str(a)]})
        r = M.summarize([a], q)
        self.assertEqual(len(r['attempts']), 1)
        self.assertEqual(len(r['attempts'][0]['input_provenance']), 3)
        self.assertEqual(r['attempts'][0]['role'], 'primary')

    def test_plan_result_and_queue_contradiction(self):
        a, _ = self.run_fixture('a')
        q = self.root / 'queue.json'
        write(q, {'jobs': [{'run_dir': str(a), 'checkpoint_step': 5000, 'seed': 10000}]})
        self.assertEqual(M.summarize(queue_manifest=q)['attempts'][0]['category'], 'invalid_record')

    def test_action_logs_not_double_counted(self):
        a, ep = self.run_fixture('a')
        (ep / 'actions.jsonl').write_text('\n'.join(json.dumps(x) for x in [
            {'status': 'submitted', 'step': 0},
            {'status': 'native_returned', 'step': 0, 'policy': {'policy_seconds': .1},
             'control_seconds': .2, 'topp_fallback_observed': False, 'topp': [{'status': 'returned', 'steps': 30}]},
            {'status': 'native_returned', 'step': 1, 'policy': {'policy_seconds': .3},
             'topp_fallback_observed': True, 'topp': [{'status': 'empty', 'steps': 0}]}]))
        am = M.summarize([a])['attempts'][0]['actions']
        self.assertEqual(am['native_returned_rows'], 2)
        self.assertEqual(am['topp_fallback_actions'], 1)
        self.assertAlmostEqual(am['policy_wall_seconds']['mean'], .2)

    def test_partial_jsonl_fail_closed_but_report_survives(self):
        a, ep = self.run_fixture('a')
        (ep / 'actions.jsonl').write_text('{"status":"submitted"}\n{"status":')
        r = M.summarize([a], expected_seeds=[10000])
        self.assertEqual(r['attempts'][0]['category'], 'invalid_record')
        self.assertIsNone(r['groups'][0]['native_success_rate'])
        self.assertTrue(r['attempts'][0]['issues'])

    def test_gt_rollout_settle_and_actual_tracking(self):
        a, ep = self.run_fixture('a')
        diag = {'arm_joint_tracking': {'left': {'actual_qpos': [.1, -.2], 'drive_target': [0., 0.]}}}
        rows = [{'control_step': 0, 'object_pose_world': [0, 0, .7], 'native_diagnostics': diag},
                {'control_step': 1, 'object_pose_world': [0, 0, .8]},
                {'control_step': 1, 'settle_step': 1, 'object_pose_world': [0, 0, 1.]}]
        (ep / 'evaluation_gt.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
        gt = M.summarize([a])['attempts'][0]['gt']
        self.assertAlmostEqual(gt['rollout']['max_object_rise_m'], .1)
        self.assertAlmostEqual(gt['settle']['max_object_rise_m'], .3)
        self.assertAlmostEqual(gt['rollout']['tracking']['left']['max'], .2)
        self.assertIsNone(gt['rollout']['tracking']['right']['max'])

    def test_command_only_never_actual_tracking(self):
        a, ep = self.run_fixture('a')
        row = {'control_step': 0, 'native_diagnostics': {'arm_joint_tracking': {'left': {'drive_target': [1], 'max_abs_error': 1}}}}
        (ep / 'evaluation_gt.jsonl').write_text(json.dumps(row))
        self.assertIsNone(M.summarize([a])['attempts'][0]['gt']['rollout']['tracking']['left']['max'])

    def test_corrupt_queue_blocks_even_complete_rate(self):
        a, _ = self.run_fixture('a')
        q = self.root / 'queue.json'
        q.write_text('{')
        r = M.summarize([a], q, [10000])
        self.assertTrue(r['issues'])
        self.assertIsNone(r['groups'][0]['native_success_rate'])

    def test_corrupt_episode_not_hidden_by_summary(self):
        a, ep = self.run_fixture('a')
        (ep / 'episode.json').write_text('{')
        self.assertEqual(M.summarize([a])['attempts'][0]['category'], 'invalid_record')

    def test_explicit_reference_can_complete_identical_protocol(self):
        a, _ = self.run_fixture('a', seed=10000)
        b, _ = self.run_fixture('b', seed=10001)
        c, _ = self.run_fixture('ref', seed=10002)
        q = self.root / 'queue.json'
        write(q, {'jobs': [{'run_dir': str(a)}, {'run_dir': str(b)}], 'reference_runs': [str(c)]})
        r = M.summarize(queue_manifest=q)
        self.assertEqual(len(r['groups']), 1)
        self.assertEqual(r['groups'][0]['native_success_rate'], 1)
        self.assertEqual(r['groups'][0]['role_counts'], {'primary': 2, 'reference': 1})

    def test_pending_same_checkpoint_unverified_protocol_blocks_rate(self):
        a, _ = self.run_fixture('a')
        q = self.root / 'queue.json'
        write(q, {'protocol': {'max_steps': 700},
                  'jobs': [{'run_dir': str(a), 'checkpoint_step': 30000, 'seed': 10000},
                           {'run_dir': str(self.root / 'pending'), 'checkpoint_step': 30000, 'seed': 10000}]})
        r = M.summarize(queue_manifest=q, expected_seeds=[10000])
        self.assertTrue(all(g['native_success_rate'] is None for g in r['groups']))
        pending = next(x for x in r['attempts'] if x['category'] == 'not_started')
        self.assertEqual(pending['protocol']['control_step_limit'], 700)

    def test_checkpoint_identity_from_sidecar_not_run_directory(self):
        a, _ = self.run_fixture('dev_30000_misleading_folder', ck=15000)
        plan = json.loads((a / 'plan.json').read_text())
        del plan['checkpoint_receipt']
        cp = self.root / 'weights' / 'opaque_checkpoint.pt'
        plan['checkpoint'] = {'path': str(cp)}
        write(cp.with_name(cp.name + '.sha256.json'), {'step': 15000, 'sha256': 'weight15000'})
        write(a / 'plan.json', plan)
        row = M.summarize([a])['attempts'][0]
        self.assertEqual(row['checkpoint_step'], 15000)
        self.assertEqual(row['checkpoint_sha256'], 'weight15000')
        self.assertTrue(row['valid_native_outcome'])

    def test_no_checkpoint_identity_not_guessed_from_folder(self):
        a, _ = self.run_fixture('dev_30000')
        plan = json.loads((a / 'plan.json').read_text())
        del plan['checkpoint_receipt']
        write(a / 'plan.json', plan)
        row = M.summarize([a])['attempts'][0]
        self.assertIsNone(row['checkpoint_step'])
        self.assertEqual(row['category'], 'invalid_record')

    def test_output_cannot_overwrite_run_result(self):
        a, _ = self.run_fixture('a')
        before = (a / 'result.json').read_bytes()
        with self.assertRaises(SystemExit):
            M.main(['--run-dirs', str(a), '--output-prefix', str(a / 'result')])
        self.assertEqual((a / 'result.json').read_bytes(), before)

    def test_missing_queue_reported(self):
        a, _ = self.run_fixture('a')
        r = M.summarize([a], self.root / 'missing.json', [10000])
        self.assertTrue(r['issues'])
        self.assertIsNone(r['groups'][0]['native_success_rate'])

    def test_corrupt_video_metadata_not_silently_valid(self):
        a, ep = self.run_fixture('a')
        (ep / 'third_view_video.json').write_text('{')
        self.assertEqual(M.summarize([a])['attempts'][0]['category'], 'invalid_record')

    def test_manifest_jobs_wrong_type_is_reported(self):
        q = self.root / 'queue.json'
        write(q, {'jobs': 11})
        r = M.summarize(queue_manifest=q)
        self.assertTrue(r['issues'])
        self.assertEqual(r['attempts'], [])

    def test_cli_writes_json_markdown_and_preserves_inputs(self):
        a, ep = self.run_fixture('a')
        (ep / 'third_view.mp4').write_bytes(b'fixture-not-real-video')
        before = {str(p): p.read_bytes() for p in a.rglob('*') if p.is_file()}
        prefix = self.root / 'out' / 'summary'
        self.assertEqual(M.main(['--run-dirs', str(a), '--output-prefix', str(prefix)]), 0)
        self.assertTrue(prefix.with_suffix('.json').is_file())
        self.assertIn('third_view.mp4', prefix.with_suffix('.md').read_text())
        self.assertEqual(before, {str(p): p.read_bytes() for p in a.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main(verbosity=2)
