#!/usr/bin/env python3
"""CPU-only, stdlib, disposable fixtures; no real training/Isaac/GPU/torch calls.
Run: CUDA_VISIBLE_DEVICES='' python3 -B scripts/test_select_wam_v3_dev_best.py -v
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import select_wam_v3_dev_best as selector


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'outputs/training/audit_stages'
        self.out.mkdir(parents=True)
        (self.root / 'config.yaml').write_text('fixture: true\n')
        self.plan = {'version': 1, 'total_optimizer_steps': 30000,
                     'checkpoints': list(range(5000, 30001, 5000)),
                     'training': {'config': 'config.yaml', 'config_sha256': selector.digest(self.root / 'config.yaml'),
                                  'optimizer_resume': True, 'early_stopping': False},
                     'physical': {'profile': 'development', 'instance': 10, 'seed': 0, 'samples': 1,
                                  'exec_horizon': 8, 'task_id': 0, 'attempts_per_interval': 1,
                                  'video': [1280, 720], 'fixed_program': True},
                     'automatic_extension': False, 'no_test_feedback_to_training': True}
        self.put(self.out / 'plan.json', self.plan)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2))

    def fixture(self, step=5000, success=False, q=0.0):
        name = f'audit_s{step:06d}'
        run = self.root / 'outputs/training' / name
        run.mkdir(parents=True, exist_ok=True)
        ck = run / f'checkpoint_step_{step:06d}.pt'
        ck.write_bytes(f'CPU fixture at {step}; deliberately not a torch pickle'.encode())
        (run / 'checkpoint_last.pt').write_bytes(b'untouched last/optimizer')
        (run / 'optimizer.pt').write_bytes(b'untouched optimizer')
        self.put(ck.with_name(ck.name + '.sha256.json'),
                 {'step': step, 'size': ck.stat().st_size, 'sha256': selector.digest(ck), 'model_only': True})
        physical = f'{name}_dev10_k1'
        phys = self.root / 'outputs/behavior/closed_loop' / physical
        # Deliberately different: development index 10 resolves to instance ID 42.
        identity = {'task': selector.TASK, 'instance_id': 42, 'rollout_id': 0}
        raw = {**identity, 'success': success, 'q_score': {'final': q}, 'steps': 100,
               'effective_policy_seed': 0, 'effective_evaluator_seed': 0}
        ep = {'task': selector.TASK, 'mode': 'public_test', 'instance_indices': [10],
              'instance_ids': [42], 'seed': 0, 'num_rollouts': 1}
        man = {'run_name': physical, 'profile': 'development', 'task_id': 0, 'task_name': selector.TASK,
               'checkpoint': str(ck), 'checkpoint_sha256': selector.digest(ck),
               'instances_override': [10], 'samples_averaged': 1, 'exec_horizon': 8,
               'seed': 0, 'effective_policy_seed': 0, 'effective_evaluator_seed': 0,
               'evaluation_complete': True, 'evaluator_returncode': 0, 'launcher_returncode': 0,
               'official_score_valid': False, 'expected_episodes': 1, 'planned_episodes': [identity],
               'completed_episodes': [{**raw, 'q_score': q}]}
        row = {'phase': 'physical_complete', 'step_target': step, 'training_name': name,
               'unit': f'wam-train-{name}.service', 'physical_name': physical,
               'checkpoint': str(ck), 'checkpoint_sha256': selector.digest(ck), 'returncode': 0,
               'physical_manifest': man, 'physical_success_not_inferred': True, 'time': 123.0}
        paths = {'stage': self.out / f'stage_{step:06d}.json', 'manifest': phys / 'manifest.json',
                 'plan': phys / 'eval/evaluation_plan.json', 'raw': phys / 'eval/json/result.json', 'ck': ck}
        for key, data in [('stage', row), ('manifest', man), ('plan', ep), ('raw', raw)]:
            self.put(paths[key], data)
        return paths

    def update(self):
        return selector.update_best(self.out, root=self.root)

    def mutate(self, paths, key, change, sync_manifest=True):
        data = json.loads(paths[key].read_text())
        change(data)
        self.put(paths[key], data)
        if key == 'manifest' and sync_manifest:
            stage = json.loads(paths['stage'].read_text())
            stage['physical_manifest'] = data
            self.put(paths['stage'], stage)

    def assert_rejected(self, paths):
        result = self.update()
        self.assertIsNone(result['best'])
        self.assertEqual(result['candidate_count'], 0)
        self.assertEqual(result['rejected_count'], 1, result)
        self.assertEqual(result['rejected'][0]['stage_file'], str(paths['stage']))
        self.assertFalse((self.out / 'checkpoint_best_task.pt').is_symlink())
        return result

    def test_success_beats_higher_q_failure_and_preserves_isaac6_flag(self):
        self.fixture(5000, False, 1.0)
        winner = self.fixture(10000, True, 0.1)
        result = self.update()
        self.assertEqual(result['best']['step'], 10000)
        self.assertTrue(result['has_successful_checkpoint'])
        self.assertIs(result['best']['official_score_valid'], False)
        self.assertEqual(len(result['candidates']), 2)
        self.assertIs(result['candidates'][0]['success'], False)
        link = self.out / 'checkpoint_best_task.pt'
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), winner['ck'])

    def test_controller_video_evidence_path_preserved_without_running_checker(self):
        paths = self.fixture(5000, True, 0.8)
        evidence = str(paths['manifest'].parent / 'video_evidence.json')
        self.mutate(paths, 'stage', lambda r: r.update(video_evidence=evidence))
        self.assertEqual(self.update()['best']['video_evidence'], evidence)

    def test_q_breaks_same_success_tie(self):
        self.fixture(5000, True, 0.2)
        self.fixture(10000, True, 0.8)
        self.assertEqual(self.update()['best']['step'], 10000)

    def test_exact_tie_prefers_earlier_step(self):
        self.fixture(10000, True, 0.8)
        self.fixture(5000, True, 0.8)
        self.assertEqual(self.update()['best']['step'], 5000)

    def test_no_stage_results_no_fake_best_even_with_checkpoint_and_orphan_eval(self):
        result = self.update()
        self.assertEqual(result['status'], 'no_valid_results')
        self.assertIsNone(result['best'])
        self.assertEqual(len(result['pending']), 6)
        paths = self.fixture(5000, True, 1.0)
        paths['stage'].unlink()  # Completed evaluator alone is not a controller receipt.
        result = self.update()
        self.assertIsNone(result['best'])
        self.assertFalse((self.out / 'checkpoint_best_task.pt').is_symlink())

    def test_noncomplete_phase_not_eligible(self):
        paths = self.fixture(5000, True, 1.0)
        for phase in ('physical_test', 'stopped_infrastructure_failure', 'budget_complete'):
            with self.subTest(phase=phase):
                self.mutate(paths, 'stage', lambda r: r.update(phase=phase))
                result = self.update()
                self.assertIsNone(result['best'])
                self.assertEqual(result['pending'][0]['phase'], phase)

    def test_all_failures_explicit_no_success_but_keep_highest_q_candidate(self):
        self.fixture(5000, False, 0.2)
        self.fixture(10000, False, 0.7)
        self.fixture(15000, False, 0.7)
        result = self.update()
        self.assertEqual(result['status'], 'no_successful_checkpoint')
        self.assertEqual(result['message'], '尚无成功checkpoint')
        self.assertFalse(result['has_successful_checkpoint'])
        self.assertFalse(result['best']['success'])
        self.assertEqual(result['best']['step'], 10000)
        self.assertEqual(result['candidate_count'], 3)
        self.assertEqual((self.out / 'checkpoint_best_task.pt').resolve(), Path(result['best']['checkpoint']))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(selector.main(['--root', str(self.root), '--stages-dir', str(self.out)]), 0)

    def test_stage_binding_rejections(self):
        mutations = [dict(step_target=10000), dict(step_target=True), dict(training_name='other_s005000'),
                     dict(unit='other.service'), dict(physical_name='audit_s005000_dev11_k1'),
                     dict(checkpoint='outputs/training/audit_s005000/checkpoint_last.pt'),
                     dict(checkpoint_sha256='0' * 64), dict(returncode=False), dict(returncode=1),
                     dict(physical_success_not_inferred=False), dict(physical_manifest={})]
        for change in mutations:
            with self.subTest(change=change):
                paths = self.fixture()
                self.mutate(paths, 'stage', lambda r: r.update(change))
                self.assert_rejected(paths)

    def test_unplanned_and_duplicate_named_stage_rejected(self):
        paths = self.fixture(35000, True, 1.0)
        self.assert_rejected(paths)
        paths['stage'].unlink()
        paths = self.fixture()
        bad = self.out / 'stage_5000.json'
        paths['stage'].rename(bad)
        paths['stage'] = bad
        self.assert_rejected(paths)

    def test_checkpoint_sha_sidecar_and_exact_model_only_step(self):
        mutations = [dict(step=4999), dict(step=5000.0), dict(sha256='0' * 64),
                     dict(size=1), dict(model_only=False)]
        for change in mutations:
            with self.subTest(change=change):
                paths = self.fixture()
                sidecar = paths['ck'].with_name(paths['ck'].name + '.sha256.json')
                receipt = json.loads(sidecar.read_text())
                self.put(sidecar, {**receipt, **change})
                self.assert_rejected(paths)
        paths = self.fixture()
        paths['ck'].write_bytes(b'tampered exact checkpoint')
        self.assert_rejected(paths)
        paths = self.fixture()
        paths['ck'].with_name(paths['ck'].name + '.sha256.json').unlink()
        self.assert_rejected(paths)

    def test_exact_checkpoint_may_not_redirect_to_another_weight(self):
        paths = self.fixture()
        other = paths['ck'].with_name('another.pt')
        other.write_bytes(paths['ck'].read_bytes())
        paths['ck'].unlink()
        paths['ck'].symlink_to(other)
        self.assert_rejected(paths)
        self.assertTrue(other.exists())

    def test_physical_manifest_rejection_matrix(self):
        mutations = [dict(evaluation_complete=1), dict(evaluation_complete=False),
                     dict(evaluator_returncode=False), dict(launcher_returncode=1),
                     dict(run_name='foreign_run'), dict(profile='official'), dict(task_id=False),
                     dict(task_name='foreign_task'), dict(checkpoint='wrong.pt'),
                     dict(checkpoint_sha256='bad'), dict(instances_override=[11]),
                     dict(instances_override=[10.0]), dict(samples_averaged=True), dict(samples_averaged=2),
                     dict(exec_horizon=16), dict(seed=1), dict(effective_policy_seed=False),
                     dict(effective_evaluator_seed=1), dict(expected_episodes=2),
                     dict(planned_episodes=[]), dict(completed_episodes=[]), dict(official_score_valid='false')]
        for change in mutations:
            with self.subTest(change=change):
                paths = self.fixture()
                self.mutate(paths, 'manifest', lambda r: r.update(change))
                self.assert_rejected(paths)

    def test_eval_plan_rejection_matrix(self):
        mutations = [dict(task='foreign_task'), dict(mode='train'), dict(instance_indices=[11]),
                     dict(instance_indices=[10.0]), dict(instance_ids=[10]), dict(instance_ids=[42, 43]),
                     dict(instance_ids=[True]), dict(seed=False), dict(num_rollouts=2)]
        for change in mutations:
            with self.subTest(change=change):
                paths = self.fixture()
                self.mutate(paths, 'plan', lambda r: r.update(change))
                self.assert_rejected(paths)

    def test_eval_json_rejection_matrix(self):
        mutations = [dict(task='foreign_task'), dict(instance_id=10), dict(rollout_id=False),
                     dict(rollout_id=1), dict(effective_policy_seed=1), dict(effective_evaluator_seed=False),
                     dict(success='false'), dict(success=True), dict(q_score={'final': 0.5}),
                     dict(q_score={'final': True}), dict(q_score={'final': -0.1}), dict(q_score={'final': 1.1}),
                     dict(q_score={'final': float('nan')}), dict(q_score={'final': float('inf')}),
                     dict(q_score={}), dict(q_score=None), dict(steps=0), dict(steps=True), dict(steps=101)]
        for change in mutations:
            with self.subTest(change=change):
                paths = self.fixture()
                self.mutate(paths, 'raw', lambda r: r.update(change))
                self.assert_rejected(paths)

    def test_completed_episode_manifest_must_agree_with_eval(self):
        for field, value in [('success', True), ('q_score', 0.5), ('steps', 99),
                             ('effective_policy_seed', 1), ('effective_evaluator_seed', 1), ('instance_id', 10)]:
            with self.subTest(field=field):
                paths = self.fixture()
                self.mutate(paths, 'manifest', lambda m: m['completed_episodes'][0].update({field: value}))
                self.assert_rejected(paths)

    def test_missing_or_duplicate_eval_json(self):
        paths = self.fixture()
        paths['raw'].unlink()
        self.assert_rejected(paths)
        paths = self.fixture()
        paths['raw'].with_name('duplicate.json').write_bytes(paths['raw'].read_bytes())
        self.assert_rejected(paths)

    def test_malformed_and_duplicate_key_json_rejected(self):
        for key in ('stage', 'manifest', 'plan', 'raw'):
            for text in ('[]', '{broken', '{"seed": 0, "seed": 1}'):
                with self.subTest(key=key, text=text):
                    paths = self.fixture()
                    paths[key].write_text(text)
                    self.assert_rejected(paths)

    def test_bad_candidate_does_not_hide_valid_or_failed_candidates(self):
        self.fixture(5000, False, 0.5)
        self.fixture(10000, True, 0.5)
        bad = self.fixture(15000, True, 1.0)
        self.mutate(bad, 'stage', lambda r: r.update(checkpoint_sha256='bad'))
        result = self.update()
        self.assertEqual(result['best']['step'], 10000)
        self.assertEqual(result['candidate_count'], 2)
        self.assertEqual(result['rejected_count'], 1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(selector.main(['--root', str(self.root), '--stages-dir', str(self.out)]), 2)

    def test_relative_evidence_paths_resolve_against_root_not_cwd(self):
        paths = self.fixture(5000, True, 1.0)
        relative = str(paths['ck'].relative_to(self.root))
        self.mutate(paths, 'stage', lambda r: r.update(checkpoint=relative))
        self.mutate(paths, 'manifest', lambda m: m.update(checkpoint=relative))
        result = selector.update_best(self.out.relative_to(self.root), root=self.root)
        self.assertEqual(result['best']['checkpoint'], str(paths['ck']))

    def test_idempotent_bytes_mtimes_and_inputs_untouched(self):
        self.fixture(5000, True, 0.9)
        inputs = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        first = self.update()
        outputs = [self.out / name for name in ('best_task.json', 'selection_protocol.json', 'checkpoint_best_task.pt')]
        before = {p: (p.lstat().st_ino, p.lstat().st_mtime_ns, p.read_bytes()) for p in outputs}
        self.assertEqual(first, self.update())
        self.assertEqual(before, {p: (p.lstat().st_ino, p.lstat().st_mtime_ns, p.read_bytes()) for p in outputs})
        self.assertEqual(inputs, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in inputs})

    def test_link_replacement_repair_and_stale_best_removal_never_delete_weights(self):
        first = self.fixture(5000, True, 0.1)
        self.update()
        second = self.fixture(10000, True, 0.9)
        self.update()
        link = self.out / 'checkpoint_best_task.pt'
        self.assertEqual(link.resolve(), second['ck'])
        link.unlink()
        link.symlink_to(first['ck'])  # Simulated interrupted/incorrect index link.
        self.update()
        self.assertEqual(link.resolve(), second['ck'])
        for paths in (first, second):
            self.mutate(paths, 'stage', lambda r: r.update(checkpoint_sha256='bad'))
        self.assertIsNone(self.update()['best'])
        self.assertFalse(link.is_symlink())
        self.assertTrue(first['ck'].is_file())
        self.assertTrue(second['ck'].is_file())

    def test_refuse_regular_file_at_link_name(self):
        self.fixture(5000, True, 1.0)
        link = self.out / 'checkpoint_best_task.pt'
        link.write_bytes(b'must not overwrite a weight')
        with self.assertRaisesRegex(ValueError, 'non-symlink'):
            self.update()
        self.assertEqual(link.read_bytes(), b'must not overwrite a weight')

    def test_plan_and_protocol_immutable(self):
        original = (self.out / 'plan.json').read_bytes()
        self.update()
        protocol = (self.out / 'selection_protocol.json').read_bytes()
        # Even whitespace changes change the binding, not the plan content.
        (self.out / 'plan.json').write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'original plan changed'):
            self.update()
        (self.out / 'plan.json').write_bytes(original)
        self.assertEqual((self.out / 'selection_protocol.json').read_bytes(), protocol)
        self.mutate({'protocol': self.out / 'selection_protocol.json'}, 'protocol',
                    lambda p: p.update(ranking=['final Q descending']))
        with self.assertRaisesRegex(ValueError, 'protocol/original plan changed'):
            self.update()
        self.assertEqual((self.out / 'plan.json').read_bytes(), original)

    def test_wrong_plan_or_config_rejected_before_selection(self):
        for section, field, value in [('physical', 'instance', 11), ('physical', 'seed', False),
                                      ('physical', 'samples', 2), ('physical', 'exec_horizon', 16),
                                      ('training', 'early_stopping', True)]:
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                plan[section][field] = value
                self.put(self.out / 'plan.json', plan)
                with self.assertRaises(ValueError):
                    self.update()
                self.assertFalse((self.out / 'best_task.json').exists())
        self.put(self.out / 'plan.json', self.plan)
        (self.root / 'config.yaml').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'config SHA256'):
            self.update()

    def test_individually_atomic_replacements(self):
        paths = self.fixture(5000, True, 0.8)
        real_replace = os.replace
        replacements = []

        def inspect_replace(source, target):
            source, target = Path(source), Path(target)
            self.assertEqual(source.parent, target.parent)
            if target.name == 'checkpoint_best_task.pt':
                self.assertTrue(source.is_symlink())
                self.assertEqual(source.resolve(), paths['ck'])
            else:
                self.assertIsInstance(json.loads(source.read_text()), dict)
            replacements.append(target.name)
            return real_replace(source, target)

        with patch.object(selector.os, 'replace', side_effect=inspect_replace):
            self.update()
        self.assertEqual(replacements, ['selection_protocol.json', 'checkpoint_best_task.pt', 'best_task.json'])
        self.assertFalse(list(self.out.glob('.*.tmp')))

    def test_real_controller_physical_validator_accepts_same_fixture(self):
        # Read-only compatibility check; loading this stdlib controller starts nothing.
        source = Path(__file__).with_name('run_wam_v3_stages.py')
        spec = importlib.util.spec_from_file_location('controller_readonly', source)
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        paths = self.fixture(5000, True, 0.7)
        row = json.loads(paths['stage'].read_text())
        manifest = controller.validate_physical(paths['manifest'].parent, row['physical_name'],
                                                paths['ck'], selector.digest(paths['ck']), 10)
        self.assertEqual(manifest, row['physical_manifest'])
        self.assertEqual(self.update()['candidate_count'], 1)


if __name__ == '__main__':
    unittest.main()
