#!/usr/bin/env python3
"""CPU checks for paired V4 selection, strict evidence and no-retry execution."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_precision_v4_final5 as q
from scripts import eval_zhuhai_wam_precision_v4 as runtime


def fixture(step=130000):
    path = q.BASE / 'wam_precision_v2_task2_dagger_80k_fixed5_20261011_v23/zhuhai_task2/seed_932101/result.json'
    raw = json.loads(path.read_text())
    rt = raw['policy_ready']['runtime']
    rt.update(format=q.v4.FORMAT, checkpoint_step=step, checkpoint_sha256='a' * 64,
        precision_v3_actual=deepcopy(q.e.ACTUAL_PRECISION_BINDING),
        precision_v4_actual=dict(version=4, horizon=16, input_dim=7184, output_dim=3072, added_parameters=7877377),
        chunk_vtt16=deepcopy(q.v4.PRECISION_V4_SPEC), chunk_vtt16_alpha=0. if step == 80000 else .02,
        chunk_vtt16_used_online=True, observer_policy_input=False,
        system_mode='precision_wam_v4_official_braking', v3_warmstart_baseline=step == 80000)
    raw.update(observer_policy_input=False, record_video=False, third_view=False)
    contract = {k: deepcopy(rt[k]) for key in ('robot_contract', 'data_contract', 'action_postprocessing')
                for k in (key, key + '_sha256')}
    return raw, contract


class SelectionTests(unittest.TestCase):
    def test_old_driver_environment_cannot_override_verified_parent(self):
        prior = dict(child_env=dict(LD_LIBRARY_PATH='/old-driver', VK_DRIVER_FILES='/old-vulkan.json',
            __EGL_VENDOR_LIBRARY_FILENAMES='/old-egl.json', LD_PRELOAD='/isaac-libstdc++.so',
            OMNI_KIT_ACCEPT_EULA='YES', P2_V3_THIRD_VIEW_SEED='932126'),
            runtime_environment={'LD_LIBRARY_PATH': '/even-older-driver'})
        environment = q.simulator_environment(prior)
        parent = dict(LD_LIBRARY_PATH='/verified-driver', VK_DRIVER_FILES='/verified-vulkan.json')
        merged = dict(parent, **environment)
        self.assertEqual(merged['LD_LIBRARY_PATH'], '/verified-driver')
        self.assertEqual(merged['VK_DRIVER_FILES'], '/verified-vulkan.json')
        self.assertEqual(merged['LD_PRELOAD'], '/isaac-libstdc++.so')
        self.assertNotIn('P2_V3_THIRD_VIEW_SEED', environment)

    def test_final_and_nearest_actual_save_dynamic_50k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (80000, 122500, 125000, 127500, 130000):
                (root / f'checkpoint_step_{step:06d}.pt').touch()
            self.assertEqual([p.name for p in q.select_milestones(root, 130000)],
                ['checkpoint_step_130000.pt', 'checkpoint_step_127500.pt'])
            (root / 'checkpoint_step_127500.pt').unlink()
            self.assertEqual(q.select_milestones(root, 130000)[1].name, 'checkpoint_step_125000.pt')
            (root / 'checkpoint_step_125000.pt').unlink()
            with self.assertRaises(q.e.EvidenceError):
                q.select_milestones(root, 130000)

    def test_missing_final_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (125000, 127500):
                (root / f'checkpoint_step_{step:06d}.pt').touch()
            with self.assertRaises(q.e.EvidenceError):
                q.select_milestones(root, 130000)

    def test_initialization_is_not_a_second_trained_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (80000, 82500):
                (root / f'checkpoint_step_{step:06d}.pt').touch()
            with self.assertRaises(q.e.EvidenceError):
                q.select_milestones(root, 82500)

    def test_actual_checkpoint_step_must_match_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint_step_127500.pt'
            path.write_bytes(b'CPU mock receipt')
            path.with_name(path.name + '.sha256.json').write_text(json.dumps(
                dict(step=130000, sha256=q.e.sha256_file(path), size=path.stat().st_size)))
            with self.assertRaises(q.e.EvidenceError):
                q.receipt(path)


class BindingTests(unittest.TestCase):
    def test_runtime_reuses_sampler_and_never_installs_observer(self):
        from harness import zhuhai_observer_video_v42 as observer
        with ExitStack() as stack:
            for obj, names in ((runtime.previous, ('precision', 'FROZEN_READ_CONTRACT', 'verify_owner', '__file__')),
                (runtime.base, ('zt', 'worker', 'log', '__file__', 'verify_owner', 'read_contract',
                                'Native16PolicyClient', 'serve_native16', 'atomic_json'))):
                for name in names:
                    stack.enter_context(patch.object(obj, name, getattr(obj, name)))
            install = stack.enter_context(patch.object(observer, 'install', side_effect=AssertionError('No observer')))
            runtime.configure()
            self.assertIs(runtime.base.worker, runtime.previous.worker)
            self.assertIs(runtime.previous.precision.load_policy, runtime.load_policy)
            self.assertEqual(runtime.base.zt.FORMAT, q.v4.FORMAT)
            self.assertEqual(runtime.base.__file__, runtime.__file__)
            install.assert_not_called()

    def test_actual_v4_binding_and_isolated_evidence(self):
        raw, contract = fixture()
        q.validate_binding(raw, 130000, q.SEEDS[0], 'a' * 64, contract)
        from scripts import precision_v3_evidence_v42 as original
        self.assertNotEqual(original.FORMAT, q.e.FORMAT)
        self.assertEqual(original.SEEDS[0], 932126)

    def test_bad_identity_future_input_or_video_rejected(self):
        for key, value in [('format', 'zhuhai-precision-wam-native16-measured-10hz-v2'),
            ('future_inputs_used_online', True), ('observer_policy_input', True),
            ('chunk_vtt16_used_online', False), ('system_mode', 'precision_wam_v3_official_braking'),
            ('checkpoint_sha256', 'b' * 64)]:
            with self.subTest(key=key):
                raw, contract = fixture()
                raw['policy_ready']['runtime'][key] = value
                with self.assertRaises(q.e.EvidenceError):
                    q.validate_binding(raw, 130000, q.SEEDS[0], 'a' * 64, contract)
        raw, contract = fixture()
        raw['record_video'] = True
        with self.assertRaises(q.e.EvidenceError):
            q.validate_binding(raw, 130000, q.SEEDS[0], 'a' * 64, contract)

    def test_zero_fusion_baseline_identity_and_no_observer_artifacts(self):
        raw, contract = fixture(80000)
        q.validate_binding(raw, 80000, q.SEEDS[0], 'a' * 64, contract)
        raw['policy_ready']['runtime']['chunk_vtt16_alpha'] = .001
        with self.assertRaises(q.e.EvidenceError):
            q.validate_binding(raw, 80000, q.SEEDS[0], 'a' * 64, contract)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(q.reject_video(directory, raw), {})
            (Path(directory) / 'third_person_frames.jsonl').touch()
            with self.assertRaises(q.e.EvidenceError):
                q.reject_video(directory, raw)

    def test_unknown_infrastructure_failure_never_counts_as_task_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text(json.dumps(dict(status='infrastructure_error', error='CUDA startup error')))
            with self.assertRaises(q.e.EvidenceError):
                q.e.classify(path, {}, {}, 'a' * 64, q.SEEDS, q.e.classify_noncontact)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = self.root / 'train'
        self.train.mkdir()
        (self.train / 'manifest.json').write_text(json.dumps({'contract': {}}))
        self.out = self.root / 'evaluation'
        cohorts = [dict(path=str(self.train / f'checkpoint_step_{step:06d}.pt'), step=step,
                        sha256='a' * 64, size=0, label='v4') for step in (130000, 127500)]
        self.plan = dict(schema='CPU_MOCK_NOT_AN_EXPERIMENT', out=str(self.out), cohorts=cohorts,
            gpu_lock_owner='mock', child_env={}, frozen_sources={str(q.RUNTIME): 'a' * 64},
            training_audit={}, v2_reference={}, interpretation='CPU MOCK', commands=[
                dict(seed=seed, checkpoint=ck, out=str(self.out / str(ck['step']) / f'seed_{seed}'), command=['mock'])
                for ck in cohorts for seed in q.SEEDS])

    def mocks(self, *, unknown=False):
        from scripts import run_zhuhai_task1_dr200_concurrent as admission
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(q, 'verify'))
        stack.enter_context(patch.object(runtime, 'verify_owner'))
        stack.enter_context(patch.object(q.gpu_lock, 'owner', return_value=None))
        stack.enter_context(patch.object(q.gpu_lock, 'acquire'))
        self.release = stack.enter_context(patch.object(q.gpu_lock, 'release'))
        stack.enter_context(patch.object(admission, 'gpu_jobs', return_value=[]))
        stack.enter_context(patch.object(admission, 'processes', return_value=[]))
        scheduler = admission.scheduler_module()
        stack.enter_context(patch.object(scheduler, 'collection_resource_admission', return_value={}))
        stack.enter_context(patch.object(scheduler, 'resource_ready', return_value=True))
        self.run = stack.enter_context(patch.object(q, 'run_once', return_value=0))
        stack.enter_context(patch.object(q, 'log'))
        stack.enter_context(patch.object(q, 'details', side_effect=lambda r: dict(r, task_progress=0)))
        self.export = stack.enter_context(patch.object(q, 'export_success'))
        def classify(path, *args):
            if unknown:
                raise q.e.EvidenceError('Unknown infrastructure failure')
            seed = int(path.parent.name.removeprefix('seed_'))
            return dict(seed=seed, status='physics_safety_abort' if seed == q.SEEDS[0] else 'completed',
                        full_task_success=False, observed_base_score=0)
        stack.enter_context(patch.object(q.e, 'classify', side_effect=classify))

    def test_valid_failures_continue_all_ten_without_video_or_extra_baseline(self):
        self.mocks()
        self.assertEqual(q.execute(self.plan), 0)
        report = q.e.read_json(self.out / 'summary.json')
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['planned_episodes'], 10)
        self.assertEqual(self.run.call_count, 10)
        self.assertEqual([r['seed'] for r in report['episodes']], list(q.SEEDS) * 2)
        self.assertEqual([r['full_successes'] for r in report['cohorts']], [0, 0])
        self.export.assert_not_called()
        self.release.assert_called_once()

    def test_unknown_error_preserves_guard_and_never_retries(self):
        self.mocks(unknown=True)
        with self.assertRaises(q.e.EvidenceError):
            q.execute(self.plan)
        self.assertEqual(self.run.call_count, 1)
        self.assertTrue((self.out / 'failure_guard.json').is_file())
        self.assertEqual(q.e.read_json(self.out / 'summary.json')['status'], 'halted')
        self.release.assert_called_once()


class SuccessVideoTests(unittest.TestCase):
    def test_only_success_full_original_frames_and_decoder(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'onboard').mkdir()
            support = root / 'support'
            support.mkdir()
            result = root / 'result.json'
            result.write_text('{"fixture":"CPU synthetic video, not an experiment"}')
            rules = [dict(physics_time_seconds=index / 10, score={'base_score': index * 30}) for index in range(2)]
            rule_path = root / 'rule_evidence.jsonl'
            rule_path.write_text(''.join(json.dumps(rule) + '\n' for rule in rules))
            files = {'rule_evidence.jsonl': {'sha256': q.e.sha256_file(rule_path)}}
            for index in range(2):
                path = root / 'onboard' / f'observation_{index:06d}.npz'
                np.savez(path, **{name: np.full((240, 320, 3), 50 + index * 100, np.uint8) for name in q.e.FIELDS[:3]})
                files[f'onboard/{path.name}'] = {'sha256': q.e.sha256_file(path)}
            row = dict(full_task_success=True, raw_result_path=str(result), raw_result_sha256=q.e.sha256_file(result),
                evidence_files=files, checkpoint_step=130000, checkpoint_sha256='a' * 64, seed=q.SEEDS[0])
            with patch.object(q, 'VIDEO_DIRECTORY', root / 'success video'):
                video = q.export_success(row, support)
                self.assertEqual(video['frames'], 2)
                self.assertEqual(video['fps'], 10)
                self.assertEqual(video['simulator_attempts_added'], 0)
                self.assertTrue(Path(video['path']).is_file())
                with self.assertRaises(q.e.EvidenceError):
                    q.export_success(row, support)
                row['full_task_success'] = False
                with self.assertRaises(q.e.EvidenceError):
                    q.export_success(row, support)


if __name__ == '__main__':
    unittest.main()
