#!/usr/bin/env python3
"""CPU-only acceptance of the frozen-protocol serial fallback. No CUDA/Isaac."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_zhuhai_task1_dr200_concurrent as q
from scripts import gpu_lock


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SerialFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(q, 'BASE', self.base))
        self.stack.enter_context(patch.object(q, 'QUEUE_LOCK', self.base / 'queue.lock'))
        self.args = q.parser().parse_args(['--training-run', str(self.base / 'training'),
            '--out', str(self.base / 'production_tasks12_dr200_collection_only_test')])
        self.run = self.args.training_run
        self.scheduler = Mock()
        self.scheduler.accepted_task1_expert.return_value = dict(probe_success_count=4, seeds=[944025, 944026, 945024, 947000])
        self.scheduler.collection_resource_admission.return_value = {'CPU_FIXTURE_ONLY': True}
        self.scheduler.resource_ready.return_value = True
        self.scheduler.main.return_value = 0

    def complete_pipeline(self):
        stages = [dict(name='tiny_admission', kind='train', step=2),
                  dict(name='train30k', kind='train', step=30000), dict(name='eval30k', kind='eval', step=30000)]
        put(self.run / 'plan.json', dict(schema='zhuhai-precision-wam-task2-30k-pipeline-v2',
            stages=stages, gpu_lock_owner='distinct-training-owner'))
        put(self.run / 'status.json', dict(status='completed', active_stage=None,
            completed_stages=[s['name'] for s in stages]))
        (self.run / '.pipeline.lock').touch()

    def tiny_fixture(self):
        directory = self.run / 'tiny_admission'; directory.mkdir(parents=True, exist_ok=True)
        source = self.base / 'cpu_fixture.py'; source.write_text('# CPU fixture only\n')
        splits = {key: [dict(domain='source' if i % 2 else 'randomized') for i in range(n)]
                  for key, n in [('train', 160), ('val', 20), ('heldout', 20)]}
        c = dict(format='zhuhai-precision-wam-native16-measured-10hz-v2', mode='tiny', hjepa=dict(variant='combined'),
            config=dict(training=dict(batch_size=4, grad_accum_steps=1, max_steps=2)),
            splits=dict(zhuhai_task2=splits), source_files={str(source): q.sha(source)})
        put(directory / 'manifest.json', dict(contract=c,
            contract_sha256=hashlib.sha256(json.dumps(c, sort_keys=True, allow_nan=False).encode()).hexdigest()))
        put(directory / 'status.json', dict(status='completed', step=2, target_steps=2,
            auxiliary_gradient_admission_passed=True, precision_gradient_admission_passed=True))
        for name, model_only in [('checkpoint_step_000002.pt', True), ('checkpoint_last.pt', False)]:
            ck = directory / name; ck.write_bytes(b'CPU synthetic bytes; never a real model')
            put(ck.with_name(ck.name + '.sha256.json'), dict(step=2, model_only=model_only,
                size=ck.stat().st_size, sha256=q.sha(ck)))
        names = ['loss', 'gradient_norm', 'shared_visual_grad_norm', 'idm_head_grad_norm', 'future8_offset_grad_norm',
                 'precision_clock_grad_norm', 'precision_residual_grad_norm', 'precision_patch_proj_grad_norm',
                 'precision_attention_grad_norm', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes']
        rows = [dict(step=step, **{name: 1 for name in names}) for step in (1, 2)]
        (directory / 'log.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        return directory

    def execute(self, *, owner=None, jobs=None, tiny=True):
        with patch.object(q, 'scheduler_module', return_value=self.scheduler), \
             patch.object(q, 'tiny_review', return_value={'complete_b4_updates_verified': tiny}), \
             patch.object(q, 'processes', return_value=jobs or []), \
             patch.object(gpu_lock, 'owner', return_value=owner), \
             patch.object(gpu_lock, 'acquire') as acquire, patch.object(gpu_lock, 'release') as release:
            result = q.execute_serial(self.args)
            return result, acquire, release

    def test_plan_default_no_launch_no_output(self):
        with patch.object(q, 'scheduler_module', return_value=self.scheduler), \
             patch.object(q, 'processes', side_effect=AssertionError('plan must not inventory/launch')), \
             patch.object(q, 'sha', return_value='cpu-fixture'):
            before = list(self.base.iterdir()); result = q.review(self.args)
        self.assertFalse(self.args.execute)
        self.assertFalse(result['concurrent_admitted']); self.assertEqual(result['planned_workers'], 1)
        self.assertEqual(result['probes']['production_quota_credit'], 0)
        self.assertEqual(list(self.base.iterdir()), before); self.scheduler.main.assert_not_called()

    def test_concurrent_execute_always_refused_before_lock(self):
        self.args.schedule = 'concurrent'
        with self.assertRaisesRegex(ValueError, 'Concurrent execution prohibited'):
            q.execute_serial(self.args)
        self.assertFalse(q.QUEUE_LOCK.exists())

    def test_workers_cannot_expand(self):
        self.args.workers = 2
        with self.assertRaises(ValueError): q.validate(self.args)

    def test_dedicated_owner_required(self):
        self.args.gpu_lock_owner = 'training-owner'
        with self.assertRaises(ValueError): q.validate(self.args)

    def test_wait_must_be_bounded(self):
        for value in [0, -1, float('inf'), float('nan'), 172801]:
            self.args.wait_seconds = value
            with self.assertRaises(ValueError): q.validate(self.args)

    def test_training_eval_gap_is_not_complete(self):
        put(self.run / 'status.json', dict(status='running', active_stage='eval20k'))
        self.assertFalse(q.pipeline_finished(self.run))

    def test_missing_and_failed_pipeline_do_not_collect(self):
        self.assertFalse(q.pipeline_finished(self.run))
        put(self.run / 'status.json', dict(status='failed'))
        with self.assertRaises(RuntimeError): self.execute()
        self.scheduler.main.assert_not_called()
        self.assertTrue(self.args.out.with_name(self.args.out.name + '.queue').joinpath('failed_guard.json').exists())

    def test_partial_completion_or_no_final_eval_rejected(self):
        self.complete_pipeline(); state = q.read(self.run / 'status.json')
        state['completed_stages'].pop(); put(self.run / 'status.json', state)
        with self.assertRaises(ValueError): q.pipeline_finished(self.run)

    def test_complete_b4_checks_both_update_peaks_not_resource_admission(self):
        self.tiny_fixture(); result = q.tiny_review(self.run)
        self.assertTrue(result['complete_b4_updates_verified']); self.assertFalse(result['resource_admitted'])

    def test_preparation_and_one_update_cannot_admit(self):
        self.assertFalse(q.tiny_review(self.run)['complete_b4_updates_verified'])
        directory = self.tiny_fixture(); rows = (directory / 'log.jsonl').read_text().splitlines()
        (directory / 'log.jsonl').write_text(rows[0])
        self.assertFalse(q.tiny_review(self.run)['complete_b4_updates_verified'])

    def test_missing_peak_or_nan_blocks(self):
        directory = self.tiny_fixture(); path = directory / 'log.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        for value in [None, float('nan'), -1]:
            changed = deepcopy(rows); changed[1]['cuda_peak_reserved_bytes'] = value
            path.write_text('\n'.join(json.dumps(r) for r in changed))
            self.assertFalse(q.tiny_review(self.run)['complete_b4_updates_verified'])

    def test_checkpoint_tampering_blocks(self):
        directory = self.tiny_fixture(); (directory / 'checkpoint_last.pt').write_bytes(b'changed')
        self.assertFalse(q.tiny_review(self.run)['complete_b4_updates_verified'])

    def test_no_false_ancestor_match_and_no_blanket_ancestor_exemption(self):
        def row(pid, argv): return dict(pid=pid, ppid=1, state='S', argv=argv)
        rows = [row(1, ['bash', '-c', 'echo train_zhuhai_wam_precision_v2.py']),
                row(2, ['python3', 'scripts/gpu_lock.py', 'run', '--', 'python3', 'train_foo.py']),
                row(3, ['python3', '-B', 'train_zhuhai_wam_precision_v2.py']),
                dict(row(4, ['python3', 'collect_fake.py']), state='Z'),
                row(6, ['python3', '-c', "print('train_foo.py'); import subprocess"])]
        self.assertEqual([r['pid'] for r in q.gpu_jobs(rows, self_pid=5)], [3])

    def test_training_eval_collect_module_entries_block(self):
        modules = ('scripts.train_zhuhai_wam_precision_v2', 'scripts.eval_zhuhai_wam_precision_v2',
                   'scripts.collect_isaac_zhuhai_tasks12_repair_200',
                   'scripts.run_zhuhai_wam_precision_v2_pipeline',
                   'scripts.run_zhuhai_tasks12_collection_repair_200')
        for module in modules:
            for executable in ('python', 'python3', '/usr/bin/python3.10'):
                with self.subTest(module=module, executable=executable):
                    row = dict(pid=999999, ppid=1, state='S',
                        argv=[executable, '-B', '-X', 'utf8', '-W', 'ignore', '-m', module])
                    self.assertEqual(q.gpu_jobs([row], self_pid=5), [row])

    def test_cpu_only_module_entries_are_not_gpu_jobs(self):
        for module in ('unittest', 'http.server', 'json.tool',
                       'scripts.test_zhuhai_precision_v2_training', 'scripts.train_tools.cpu_report',
                       'scripts.gpu_lock'):
            with self.subTest(module=module):
                row = dict(pid=999999, ppid=1, state='S', argv=['python', '-m', module,
                    'scripts.train_zhuhai_wam_precision_v2', 'collect_fake.py', '--label', 'torch'])
                self.assertEqual(q.gpu_jobs([row], self_pid=5), [])

    def test_gpu_runtime_module_roots_still_block(self):
        for module in ('torch.distributed.run', 'isaacsim', 'omni.isaac', 'sapien', 'mujoco.viewer', 'cupy'):
            with self.subTest(module=module):
                row = dict(pid=999999, ppid=1, state='S', argv=['python3', '-m', module])
                self.assertEqual(q.gpu_jobs([row], self_pid=5), [row])

    def test_module_foreign_job_prevents_serial_acquire(self):
        self.complete_pipeline()
        row = dict(pid=999999, ppid=1, state='S',
            argv=['python', '-m', 'scripts.train_zhuhai_wam_precision_v2'])
        with patch.object(q, 'scheduler_module', return_value=self.scheduler), \
             patch.object(q, 'tiny_review', return_value={'complete_b4_updates_verified': True}), \
             patch.object(q, 'processes', return_value=[row]), \
             patch.object(gpu_lock, 'owner', return_value=None), \
             patch.object(gpu_lock, 'acquire') as acquire:
            with self.assertRaisesRegex(ValueError, 'Foreign/live'): q.execute_serial(self.args)
        acquire.assert_not_called(); self.scheduler.main.assert_not_called()

    def test_unrelated_cpu_module_does_not_block_serial(self):
        self.complete_pipeline()
        row = dict(pid=999999, ppid=1, state='S', argv=['python', '-m', 'unittest'])
        result, acquire, release = self.execute(jobs=[row])
        self.assertEqual(result, 0); acquire.assert_called_once(); release.assert_called_once()

    def test_held_pipeline_flock_refuses_second_lock(self):
        self.complete_pipeline()
        with q.file_lock(self.run / '.pipeline.lock'):
            with self.assertRaises(BlockingIOError):
                with q.file_lock(self.run / '.pipeline.lock', shared=True): pass

    def test_queue_mutex_prevents_second_writer(self):
        with q.file_lock(q.QUEUE_LOCK):
            with self.assertRaises(BlockingIOError): self.execute()
        self.scheduler.main.assert_not_called()

    def test_serial_happy_path_delegates_once_and_releases_distinct_owner(self):
        self.complete_pipeline(); result, acquire, release = self.execute()
        self.assertEqual(result, 0); acquire.assert_called_once_with(self.args.gpu_lock_owner, max_wait=0)
        release.assert_called_once_with(self.args.gpu_lock_owner); self.scheduler.main.assert_called_once()
        argv = self.scheduler.main.call_args.args[0]
        self.assertEqual(argv[argv.index('--collection-workers') + 1], '1')
        self.assertNotIn('--resume', argv); self.assertIn('--execute', argv)
        self.assertEqual(argv[argv.index('--randomized-count') + 1], '200')

    def test_old_output_and_failed_queue_never_retried(self):
        self.args.out.mkdir(); evidence = self.args.out / 'failed_guard.json'; evidence.write_text('preserve')
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(evidence.read_text(), 'preserve'); self.scheduler.main.assert_not_called()

    def test_foreign_job_refused_without_acquire_or_kill(self):
        self.complete_pipeline()
        with self.assertRaisesRegex(ValueError, 'Foreign/live'):
            self.execute(jobs=[dict(pid=999999, ppid=1, state='S', argv=['python3', 'collect_other.py'])])
        self.scheduler.main.assert_not_called()

    def test_resource_failure_preserves_guard_and_never_launches(self):
        self.complete_pipeline(); self.scheduler.resource_ready.return_value = False
        with self.assertRaisesRegex(ValueError, 'reserve denies'): self.execute()
        self.scheduler.main.assert_not_called()

    def test_scheduler_failure_not_retried(self):
        self.complete_pipeline(); self.scheduler.main.return_value = 2
        with self.assertRaisesRegex(ValueError, 'scheduler exit=2'): self.execute()
        self.scheduler.main.assert_called_once()

    def test_training_owner_held_during_idle_window_does_not_launch(self):
        self.complete_pipeline(); self.args.wait_seconds = 2
        with patch.object(q.time, 'monotonic', side_effect=[0, 1, 2]), patch.object(q.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'wait exhausted'):
                self.execute(owner='training-owner')
        self.scheduler.main.assert_not_called()

    def test_residual_worker_retains_lock_and_reports_failure(self):
        self.complete_pipeline()
        leftover = dict(pid=999999, ppid=1, state='S', argv=['python3', 'collect_leftover.py'])
        with patch.object(q, 'scheduler_module', return_value=self.scheduler), \
             patch.object(q, 'tiny_review', return_value={'complete_b4_updates_verified': True}), \
             patch.object(q, 'processes', side_effect=[[], [], [leftover]]), \
             patch.object(gpu_lock, 'owner', return_value=None), \
             patch.object(gpu_lock, 'acquire'), patch.object(gpu_lock, 'release') as release:
            with self.assertRaisesRegex(RuntimeError, 'owner retained'): q.execute_serial(self.args)
        release.assert_not_called()
        control = self.args.out.with_name(self.args.out.name + '.queue')
        self.assertEqual(q.read(control / 'status.json')['status'], 'halted_owner_retained')

    def test_actual_frozen_cli_still_parses_single_worker(self):
        parsed = q.scheduler_module().parser().parse_args(q.scheduler_args(self.args, execute=True))
        self.assertEqual(parsed.collection_workers, 1); self.assertEqual(parsed.randomized_count, 200)
        self.assertEqual(parsed.task1_acceptance, q.PROBE); self.assertFalse(parsed.resume)


if __name__ == '__main__':
    unittest.main(verbosity=2)
