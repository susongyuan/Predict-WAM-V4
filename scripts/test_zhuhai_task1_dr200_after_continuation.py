#!/usr/bin/env python3
"""CPU-only continuation queue tests. All production/GPU entrypoints are mocked.

Synthetic checkpoints are bytes, never Torch models. Temporary evidence is
outside the project; no live run/queue, scheduler, controller or unit is started.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_zhuhai_task1_dr200_after_continuation as w
from scripts import run_zhuhai_task1_dr200_concurrent as frozen
from scripts import gpu_lock


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.origin = self.base / 'wam_precision_v2_task2_30k_20261010_v1'
        self.run = self.base / 'wam_precision_v2_task2_30k_continuation_20261010_r2'
        self.out = self.base / 'production_tasks12_dr200_collection_only_test_r2'
        real_origin = w.ORIGIN
        original = json.loads((real_origin / 'plan.json').read_text())
        self.original = json.loads(json.dumps(original).replace(str(real_origin), str(self.origin)))
        put(self.origin / 'plan.json', self.original)
        put(self.origin / 'status.json', dict(status='failed', active_stage='development_20000'))
        self.stack.enter_context(patch.object(w, 'BASE', self.base))
        self.stack.enter_context(patch.object(w, 'ORIGIN', self.origin))
        self.stack.enter_context(patch.object(w, 'ORIGIN_PLAN_SHA256', w.sha(self.origin / 'plan.json')))
        self.q = w.private_queue()
        self.stack.enter_context(patch.object(self.q, 'BASE', self.base))
        self.stack.enter_context(patch.object(self.q, 'QUEUE_LOCK', self.base / 'queue.lock'))
        self.args = w.parser().parse_args(['--training-run', str(self.run), '--out', str(self.out)])
        self.plan = deepcopy(self.original)
        self.plan.update(out=str(self.run), training_origin=str(self.origin), origin_plan_sha256=w.ORIGIN_PLAN_SHA256,
                         gpu_lock_owner='continuation-training-owner', stages=deepcopy(self.original['stages'][2:]))
        self.plan.pop('precision', None)  # Real r2 omits redundant model metadata.
        self.plan['source_hashes'].update({str(w.ROOT / p): h for p, h in w.ATTEMPT_SHA256.items()})
        self.adopted = [self.origin / f'adopted_seed_{seed}' / 'result.json' for seed in w.SEEDS[:2]]
        self.api = Mock()
        self.api.known_adoptions.return_value = dict(zip(w.SEEDS[:2], self.adopted))
        def verify_summary(path, checkpoint_step, seeds, checkpoint_sha256=None):
            value = w.read(path); rows = value.get('episodes', [])
            w.require(value.get('checkpoint_step') == checkpoint_step and value.get('checkpoint_sha256') == checkpoint_sha256,
                      'Incomplete/wrong checkpoint evaluation')
            w.require(len(rows) == 5 and [r.get('seed') for r in rows] == seeds and all(
                r.get('status') in ('completed', 'control_safety_abort') and type(r.get('full_task_success')) is bool for r in rows),
                'Incomplete evaluation seed/task evidence')
            return value
        self.api.verify_summary.side_effect = verify_summary
        self.stack.enter_context(patch.object(w, 'fixed_api', return_value=self.api))
        for s in self.plan['stages']:
            if s['kind'] == 'eval':
                s['out'] = str(self.base / 'prior_controller/development_20000' if s['step'] == 20000 else self.run / s['name'])
                s['adopt_existing'] = s['step'] == 20000
                s['command'][2] = str(w.ROOT / 'scripts/eval_zhuhai_precision_fixed_attempts.py')
                index = s['command'].index('--max-steps'); del s['command'][index:index + 2]
                if s['step'] == 20000:
                    s['adopted_results'] = list(map(str, self.adopted))
                    for path in self.adopted: s['command'] += ['--adopt-result', str(path)]
            c = s['command']; c[c.index('--out') + 1] = s['out']
            c[c.index('--gpu-lock-owner') + 1] = self.plan['gpu_lock_owner']
        self.scheduler = Mock()
        self.scheduler.main.return_value = 0
        self.scheduler.resource_ready.return_value = True
        self.scheduler.collection_resource_admission.return_value = {'CPU_FIXTURE_ONLY': True}
        self.scheduler_factory = self.stack.enter_context(patch.object(self.q, 'scheduler_module', return_value=self.scheduler))
        self.stack.enter_context(patch.object(self.q, 'processes', return_value=[]))
        self.owner = self.stack.enter_context(patch.object(gpu_lock, 'owner', return_value=None))
        self.acquire = self.stack.enter_context(patch.object(gpu_lock, 'acquire'))
        self.release = self.stack.enter_context(patch.object(gpu_lock, 'release'))
        self.tiny = self.stack.enter_context(patch.object(self.q, 'tiny_review', return_value={'complete_b4_updates_verified': True}))

    def publish(self, completed=True):
        put(self.run / 'plan.json', self.plan)
        put(self.run / 'status.json', dict(status='completed' if completed else 'running',
            active_stage=None if completed else self.plan['stages'][0]['name'],
            completed_stages=[s['name'] for s in self.plan['stages']] if completed else []))
        (self.run / '.pipeline.lock').touch()
        for s in self.plan['stages']:
            if s['kind'] == 'eval':
                put(Path(s['out']) / 'summary.json', dict(schema='precision-v2-fixed-attempt-summary-v1', status='completed',
                    checkpoint_step=s['step'], checkpoint_sha256=hashlib.sha256(b'CPU fixture not a model').hexdigest(),
                    planned_episodes=5, normal_completed=4, safety_aborted=1, full_successes=0, success_rate=0.0,
                    episodes=[dict(seed=n, task='zhuhai_task2', status='control_safety_abort' if i == 0 else 'completed',
                                   full_task_success=False, raw_result_path=str(self.adopted[i]) if s['step'] == 20000 and i < 2
                                   else str(Path(s['out']) / f'seed_{n}/result.json')) for i, n in enumerate(w.SEEDS)]))
        put(self.origin / 'train/status.json', dict(status='completed', step=30000, target_steps=30000))
        for step in (20000, 25000, 30000):
            ck = self.origin / 'train' / f'checkpoint_step_{step:06d}.pt'; ck.write_bytes(b'CPU fixture not a model')
            put(ck.with_name(ck.name + '.sha256.json'), dict(step=step, model_only=True, size=ck.stat().st_size, sha256=w.sha(ck)))

    def bad_plan(self, change):
        change(self.plan); put(self.run / 'plan.json', self.plan)
        with self.assertRaises((ValueError, KeyError, TypeError, FileNotFoundError)):
            w.validate_binding(self.args)
        self.scheduler.main.assert_not_called(); self.acquire.assert_not_called()

    def run_execute(self):
        return w.execute(self.args, self.q)

    def test_defaults_and_owner_alias(self):
        a = w.parser().parse_args([])
        self.assertFalse(a.execute); self.assertEqual(a.training_run, w.DEFAULT_RUN)
        self.assertEqual(a.out, w.DEFAULT_OUT); self.assertEqual(a.gpu_lock_owner, w.DEFAULT_OWNER)
        self.assertEqual(a.wait_seconds, 172800); self.assertEqual(a.workers, 1)
        self.assertEqual(w.parser().parse_args(['--owner', 'foo']).gpu_lock_owner, 'foo')

    def test_plan_missing_readonly_without_scheduler(self):
        before = list(self.base.iterdir()); result = w.review(self.args, self.q)
        self.assertEqual(result['status'], 'awaiting_controller_plan')
        self.assertEqual(before, list(self.base.iterdir())); self.scheduler_factory.assert_not_called()
        self.assertFalse(result['collection_started']); self.owner.assert_not_called()

    def test_plan_valid_never_imports_scheduler(self):
        self.publish(False); self.assertTrue(w.review(self.args, self.q)['continuation_plan_present'])
        self.scheduler_factory.assert_not_called(); self.acquire.assert_not_called()

    def test_private_override_does_not_touch_original_globals(self):
        original = frozen.tiny_review; self.publish(); self.run_execute()
        self.assertIs(frozen.tiny_review, original); self.assertIs(self.q.tiny_review, self.tiny)
        self.assertEqual(self.q.execute_serial.__code__.co_code, frozen.execute_serial.__code__.co_code)

    def test_completed_continuation_ignores_failed_origin_status(self):
        self.publish(); self.assertEqual(self.run_execute(), 0)
        self.tiny.assert_called_once_with(self.origin); self.scheduler.main.assert_called_once()
        self.acquire.assert_called_once_with(self.args.gpu_lock_owner, max_wait=0)
        self.release.assert_called_once_with(self.args.gpu_lock_owner)
        argv = self.scheduler.main.call_args.args[0]
        self.assertIn('--execute', argv); self.assertNotIn('--resume', argv)
        self.assertEqual(argv[argv.index('--collection-workers') + 1], '1')
        self.assertEqual(argv[argv.index('--randomized-count') + 1], '200')

    def test_waits_for_plan_creation_no_scheduler_before(self):
        def publish(_):
            self.scheduler_factory.assert_not_called(); self.publish()
        with patch.object(w.time, 'sleep', side_effect=publish): self.assertEqual(self.run_execute(), 0)
        self.scheduler.main.assert_called_once()

    def test_missing_plan_timeout_no_scheduler_and_durable_new_guard(self):
        self.args.wait_seconds = 1
        with patch.object(w.time, 'monotonic', side_effect=[0, 0, 1, 2]), patch.object(w.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'wait exhausted'): self.run_execute()
        self.scheduler_factory.assert_not_called(); self.acquire.assert_not_called()
        state = w.read(Path(str(self.out) + '.queue') / 'failed_guard.json')
        self.assertFalse(state['production_started'])

    def test_failure_before_plan_appears(self):
        put(self.run / 'status.json', dict(status='failed'))
        with self.assertRaisesRegex(ValueError, 'failed/halted'): self.run_execute()
        self.scheduler_factory.assert_not_called()

    def test_running_pipeline_does_not_start(self):
        self.publish(False); self.args.wait_seconds = 1
        with patch.object(w.time, 'monotonic', side_effect=[0, 0, 1, 2]), patch.object(w.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'wait exhausted'): self.run_execute()
        self.scheduler_factory.assert_not_called()

    def test_failed_completed_and_halted_statuses_stop(self):
        self.publish()
        for status in ('failed', 'halted_stage_failure', 'halted_interrupted'):
            with self.subTest(status=status):
                self.args.out = self.base / ('production_tasks12_dr200_collection_only_' + status)
                put(self.run / 'status.json', dict(status=status))
                with self.assertRaisesRegex(ValueError, 'failed/halted'): self.run_execute()
                self.scheduler.main.assert_not_called()
                # Each refusal is durable; never run a real queue retry.

    def test_incomplete_completion_inventory_stops(self):
        self.publish(); state = w.read(self.run / 'status.json'); state['completed_stages'].pop()
        put(self.run / 'status.json', state)
        with self.assertRaisesRegex(ValueError, 'All planned stages'): self.run_execute()
        self.scheduler_factory.assert_not_called()

    def test_active_stage_must_be_none(self):
        self.publish(); state = w.read(self.run / 'status.json'); state['active_stage'] = 'development_30000'
        put(self.run / 'status.json', state)
        with self.assertRaisesRegex(ValueError, 'All planned stages'): self.run_execute()

    def test_wrong_origin(self): self.bad_plan(lambda p: p.update(training_origin=str(self.base / 'other')))
    def test_relative_origin(self): self.bad_plan(lambda p: p.update(training_origin='relative'))
    def test_origin_hash_mismatch(self): self.bad_plan(lambda p: p.update(origin_plan_sha256='0' * 64))
    def test_wrong_task(self): self.bad_plan(lambda p: p.update(task='zhuhai_task1'))
    def test_short_training_budget(self): self.bad_plan(lambda p: p.update(training_updates=25000))
    def test_wrong_batch(self): self.bad_plan(lambda p: p.update(batch_size=2))
    def test_changed_fixed_seeds(self): self.bad_plan(lambda p: p['stages'][0]['seeds'].reverse())
    def test_final_eval_missing(self): self.bad_plan(lambda p: p['stages'].pop())
    def test_extra_stage(self): self.bad_plan(lambda p: p['stages'].append(deepcopy(p['stages'][0])))
    def test_reordered_stages(self): self.bad_plan(lambda p: p['stages'].reverse())
    def test_stage20_not_adopted(self): self.bad_plan(lambda p: p['stages'][0].update(adopt_existing=False))
    def test_adopting_remaining_train(self): self.bad_plan(lambda p: p['stages'][1].update(adopt_existing=True))
    def test_adopting_later_eval(self): self.bad_plan(lambda p: p['stages'][2].update(adopt_existing=True))
    def test_train_out_not_origin(self): self.bad_plan(lambda p: p['stages'][1].update(out=str(self.run / 'train')))
    def test_eval_reuses_old_origin_evidence(self): self.bad_plan(lambda p: p['stages'][0].update(out=str(self.origin / 'development_20000')))
    def test_duplicate_eval_outputs(self): self.bad_plan(lambda p: p['stages'][2].update(out=p['stages'][0]['out']))
    def test_source_binding_missing(self): self.bad_plan(lambda p: p['source_hashes'].pop(next(iter(p['source_hashes']))))
    def test_source_binding_drift(self): self.bad_plan(lambda p: p['source_hashes'].update({next(iter(p['source_hashes'])): '0' * 64}))
    def test_recording_disabled(self): self.bad_plan(lambda p: p.update(record_video=True))
    def test_third_camera_disabled(self): self.bad_plan(lambda p: p.update(third_view=True))
    def test_shared_continuation_owner(self): self.bad_plan(lambda p: p.update(gpu_lock_owner=self.args.gpu_lock_owner))

    def test_wrong_command_task_and_duplicate_flag(self):
        for addition in (['--task', 'zhuhai_task1'], ['--record-video']):
            p = deepcopy(self.plan); p['stages'][0]['command'] += addition
            put(self.run / 'plan.json', p)
            with self.assertRaises(ValueError): w.validate_binding(self.args)

    def test_plan_pinned_against_later_mutation(self):
        self.publish(); binding = w.validate_binding(self.args)
        self.plan['unexpected_edit'] = True; put(self.run / 'plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'changed after first'): w.validate_binding(self.args, binding['plan_sha256'])

    def test_original_plan_tamper_even_if_claim_is_updated(self):
        self.original['batch_size'] = 2; put(self.origin / 'plan.json', self.original)
        self.plan['origin_plan_sha256'] = w.sha(self.origin / 'plan.json')
        put(self.run / 'plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'Origin plan hash'): w.validate_binding(self.args)

    def test_run_symlink_denied(self):
        real = self.base / 'other'; real.mkdir(); self.run.symlink_to(real, target_is_directory=True)
        with self.assertRaises(ValueError): w.review(self.args, self.q)

    def test_plan_symlink_denied(self):
        self.run.mkdir(); (self.run / 'plan.json').symlink_to(self.origin / 'plan.json')
        with self.assertRaises(ValueError): w.validate_binding(self.args)

    def test_origin_and_continuation_overlap_denied(self):
        self.args.training_run = self.origin / 'nested'
        with self.assertRaises(ValueError): w.review(self.args, self.q)

    def test_out_origin_overlap_denied(self):
        self.args.out = self.origin / 'production_tasks12_dr200_collection_only_nested'
        with self.assertRaises(ValueError): w.review(self.args, self.q)

    def test_old_failed_queue_and_partials_unchanged(self):
        old = self.base / 'production_tasks12_dr200_collection_only_20261010_serial_v1.queue'
        put(old / 'failed_guard.json', dict(status='halted', attempt=0))
        partial = self.base / 'old_partial'; partial.write_bytes(b'48 DR + all partial evidence sentinel')
        before = ((old / 'failed_guard.json').read_bytes(), partial.read_bytes())
        self.publish(); self.run_execute()
        self.assertEqual(before, ((old / 'failed_guard.json').read_bytes(), partial.read_bytes()))

    def test_existing_new_queue_never_reset(self):
        guard = Path(str(self.out) + '.queue') / 'failed_guard.json'; put(guard, {'keep': True})
        before = guard.read_bytes()
        with self.assertRaises(ValueError): self.run_execute()
        self.assertEqual(guard.read_bytes(), before); self.scheduler_factory.assert_not_called()

    def test_existing_output_never_reused(self):
        self.out.mkdir()
        with self.assertRaises(ValueError): self.run_execute()
        self.scheduler_factory.assert_not_called()

    def test_invalid_wait_bounds(self):
        for wait in (0, -1, 172801, float('nan'), float('inf')):
            self.args.wait_seconds = wait
            with self.assertRaises(ValueError): w.review(self.args, self.q)

    def test_multiple_workers_rejected(self):
        self.args.workers = 2
        with self.assertRaises(ValueError): w.review(self.args, self.q)

    def test_lock_held_pipeline_does_not_collect(self):
        self.publish(); self.args.wait_seconds = 1
        with self.q.file_lock(self.run / '.pipeline.lock'):
            with patch.object(self.q.time, 'sleep'), patch.object(w.time, 'monotonic', side_effect=itertools.count(0, .05)):
                with self.assertRaisesRegex(ValueError, 'wait exhausted'): self.run_execute()
        self.scheduler.main.assert_not_called(); self.acquire.assert_not_called(); self.tiny.assert_not_called()

    def test_foreign_job_blocks_unchanged_guard(self):
        self.publish()
        row = dict(pid=999999999, ppid=1, state='S', argv=['python3', '-m', 'scripts.train_foo'])
        with patch.object(self.q, 'processes', return_value=[row]):
            with self.assertRaisesRegex(ValueError, 'Foreign/live'): self.run_execute()
        self.acquire.assert_not_called(); self.scheduler.main.assert_not_called()

    def test_resource_denial_blocks(self):
        self.publish(); self.scheduler.resource_ready.return_value = False
        with self.assertRaisesRegex(ValueError, 'reserve denies'): self.run_execute()
        self.scheduler.main.assert_not_called(); self.release.assert_called_once()

    def test_tiny_incomplete_blocks(self):
        self.publish(); self.tiny.return_value = {'complete_b4_updates_verified': False}
        with self.assertRaisesRegex(ValueError, 'tiny update evidence'): self.run_execute()
        self.acquire.assert_not_called(); self.scheduler.main.assert_not_called()

    def test_tiny_route_uses_real_original_peak_validator(self):
        self.publish()
        with patch.object(self.q, 'tiny_review', frozen.tiny_review):
            with self.assertRaisesRegex(ValueError, 'tiny update evidence'): self.run_execute()
        self.acquire.assert_not_called()

    def test_partial_eval_evidence_stops_before_acquire(self):
        self.publish(); s = self.plan['stages'][0]; p = Path(s['out']) / 'summary.json'
        data = w.read(p); data['episodes'].pop(); put(p, data)
        with self.assertRaisesRegex(ValueError, 'Incomplete evaluation'): self.run_execute()
        self.acquire.assert_not_called(); self.scheduler.main.assert_not_called()

    def test_checkpoint_tamper_blocks_before_acquire(self):
        self.publish(); (self.origin / 'train/checkpoint_step_030000.pt').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checkpoint receipt'): self.run_execute()
        self.acquire.assert_not_called()

    def test_training_still_20k_blocks(self):
        self.publish(); put(self.origin / 'train/status.json', dict(status='stopped_at_safe_boundary', step=20000, target_steps=30000))
        with self.assertRaisesRegex(ValueError, 'full30k'): self.run_execute()
        self.acquire.assert_not_called()

    def test_scheduler_failure_no_retry_and_guard(self):
        self.publish(); self.scheduler.main.return_value = 2
        with self.assertRaisesRegex(ValueError, 'scheduler exit=2'): self.run_execute()
        self.scheduler.main.assert_called_once(); self.assertTrue((Path(str(self.out) + '.queue') / 'failed_guard.json').is_file())

    def test_json_duplicate_and_nonfinite_rejected(self):
        p = self.base / 'bad.json'
        for text in ('{"a":1,"a":2}', '{"a":NaN}'):
            p.write_text(text)
            with self.assertRaises(ValueError): w.read(p)

    def test_frozen_queue_source_hash_checks(self):
        with patch.object(w, 'FROZEN_SHA256', {'scripts/run_zhuhai_task1_dr200_concurrent.py': '0' * 64}):
            with self.assertRaisesRegex(ValueError, 'Frozen queue/guard/source drift'): w.private_queue()


    def test_plan_changed_during_serial_handoff_stops(self):
        self.publish(); original_execute = self.q.execute_serial
        def changed(args):
            self.plan['handoff_tamper'] = True; put(self.run / 'plan.json', self.plan)
            return original_execute(args)
        with patch.object(self.q, 'execute_serial', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'changed after first'): self.run_execute()
        self.acquire.assert_not_called(); self.scheduler.main.assert_not_called()

    def test_original_gpu_owner_must_release_before_collection(self):
        self.publish(); self.args.wait_seconds = 2; self.owner.return_value = 'training-owner'
        with patch.object(w.time, 'monotonic', side_effect=itertools.count(0, .1)), patch.object(w.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'wait exhausted'): self.run_execute()
        self.acquire.assert_not_called(); self.release.assert_not_called(); self.scheduler.main.assert_not_called()

    def test_residual_job_keeps_owner_without_signals(self):
        self.publish()
        row = dict(pid=999999999, ppid=1, state='S', argv=['python3', 'collect_leftover.py'])
        with patch.object(self.q, 'processes', side_effect=[[], [], [row]]):
            with self.assertRaisesRegex(RuntimeError, 'owner retained'): self.run_execute()
        self.release.assert_not_called(); self.scheduler.main.assert_called_once()
        self.assertTrue(w.read(Path(str(self.out) + '.queue') / 'owner_retained_guard.json')['owner_retained'])

    def test_symlink_eval_summary_denied(self):
        self.publish(); p = Path(self.plan['stages'][0]['out']) / 'summary.json'
        moved = self.base / 'elsewhere.json'; p.rename(moved); p.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, 'Noncanonical'): self.run_execute()
        self.acquire.assert_not_called()

    def test_main_default_only_plans(self):
        with patch.object(w, 'private_queue', return_value=self.q), patch.object(w, 'execute') as launch, patch('sys.stdout', new_callable=io.StringIO) as out:
            self.assertEqual(w.main(['--training-run', str(self.run), '--out', str(self.out)]), 0)
        launch.assert_not_called(); self.assertEqual(json.loads(out.getvalue())['mode'], 'plan')

    def test_shared_original_owner_denied(self):
        self.args.gpu_lock_owner = self.original['gpu_lock_owner']; self.publish()
        with self.assertRaisesRegex(ValueError, 'training owner'): w.validate_binding(self.args)

    def test_bad_json_plan_is_not_waited_or_launched(self):
        self.run.mkdir(); (self.run / 'plan.json').write_text('{"incomplete":')
        with self.assertRaises(ValueError): self.run_execute()
        self.scheduler_factory.assert_not_called(); self.acquire.assert_not_called()


    def precomplete(self):
        # The real source-bound summary lives below ROOT; CPU fixtures live in
        # a disposable /tmp project. Retain the original canonical/symlink checks.
        original_canonical = w.canonical
        def fixture_canonical(path, parent):
            if parent == w.ROOT and self.base in Path(path).parents:
                parent = self.base
            return original_canonical(path, parent)
        self.stack.enter_context(patch.object(w, 'canonical', side_effect=fixture_canonical))
        self.publish(False)
        stage = self.plan['stages'][0]; summary = Path(stage['out']) / 'summary.json'
        stage.update(precompleted_summary=True, completed_summary_path=str(summary), completed_summary_sha256=w.sha(summary))
        self.plan['source_hashes'][str(summary)] = w.sha(summary)
        put(self.run / 'plan.json', self.plan)

    def test_r2_precompleted_summary_default_review_accepts_zero_of_five(self):
        self.precomplete(); p = w.review(self.args, self.q)
        self.assertEqual(p['verified_precompleted_evaluations'][0]['full_successes'], 0)
        self.assertEqual(p['verified_precompleted_evaluations'][0]['safety_aborted'], 1)
        self.assertFalse(p['policy_success_required']); self.assertTrue(p['optional_50k_is_not_a_dependency'])
        self.scheduler_factory.assert_not_called(); self.acquire.assert_not_called()

    def test_explicit_active_run_not_hard_bound_to_r1_or_r2(self):
        self.run = self.base / 'another_active_continuation'; self.args.training_run = self.run
        self.plan['out'] = str(self.run); self.publish(False)
        self.assertTrue(w.review(self.args, self.q)['continuation_plan_present'])

    def test_precompleted_summary_hash_drift_stops(self):
        self.precomplete(); path = Path(self.plan['stages'][0]['out']) / 'summary.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaises(ValueError): w.review(self.args, self.q)
        self.scheduler_factory.assert_not_called()

    def test_final_eval_cannot_claim_precompleted(self):
        self.bad_plan(lambda p: p['stages'][-1].update(precompleted_summary=True))

    def test_adopted_paths_not_replaceable(self):
        self.bad_plan(lambda p: p['stages'][0]['adopted_results'].reverse())

    def test_later_stage_adoption_forbidden(self):
        self.bad_plan(lambda p: p['stages'][2].update(adopted_results=list(map(str, self.adopted))))

    def test_frozen_api_source_binding_required(self):
        self.bad_plan(lambda p: p['source_hashes'].pop(str(w.ROOT / 'scripts/eval_zhuhai_precision_fixed_attempts.py')))

    def test_numerical_or_infra_error_cannot_finish_fixed_cohort(self):
        self.publish(); path = Path(self.plan['stages'][-1]['out']) / 'summary.json'
        value = w.read(path); value['episodes'][0]['status'] = 'infrastructure_error'; put(path, value)
        with self.assertRaisesRegex(ValueError, 'Incomplete evaluation'): self.run_execute()
        self.acquire.assert_not_called(); self.scheduler.main.assert_not_called()

    def test_summary_verifier_receives_expected_checkpoint_sha(self):
        self.publish(); self.run_execute()
        self.assertEqual(self.api.verify_summary.call_count, 3)
        for call in self.api.verify_summary.call_args_list:
            self.assertEqual(call.kwargs['checkpoint_sha256'], hashlib.sha256(b'CPU fixture not a model').hexdigest())

    def test_0_of_5_safety_abort_cohorts_allow_dr_without_50k(self):
        self.publish(); self.assertEqual(self.run_execute(), 0)
        self.scheduler.main.assert_called_once()
        self.assertFalse(any(s.get('step', 0) > 30000 for s in self.plan['stages']))

    def test_changed_raw_adopted_path_stops_before_gpu(self):
        self.publish(); path = Path(self.plan['stages'][0]['out']) / 'summary.json'
        value = w.read(path); value['episodes'][0]['raw_result_path'] = '/replacement/result.json'; put(path, value)
        with self.assertRaisesRegex(ValueError, 'replaced or retried'): self.run_execute()
        self.acquire.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
