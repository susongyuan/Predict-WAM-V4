#!/usr/bin/env python3
"""CPU synthetic tests; all subprocesses, GPU locks and live units are mocked."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import run_task090_training_first as q


class SyntheticQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, value in dict(ROOT=self.root, OLD_OUT=self.root / 'old', TEMPLATE=self.root / 'old/train_045000_launch.json',
                                PYTHON=self.root / 'python', CONFIG=self.root / 'config.yaml').items():
            self.enterContext(patch.object(q, name, value))
        q.CONFIG.write_text('frozen synthetic configuration\n')
        (self.root / 'progress.md').write_text('# Progress\nUser notes\n' + q.TRIPLET_MARKER + 'old tests\n\n## 12. Old queue\nretain this\n')
        q.OLD_OUT.mkdir()
        self.out = self.root / 'new'
        self.enterContext(patch.object(q.subprocess, 'run', side_effect=AssertionError('unmocked subprocess forbidden')))
        self.enterContext(patch.object(q, 'gpu_owner', return_value=None))
        self.enterContext(patch.object(q.time, 'sleep', side_effect=AssertionError('unmocked sleep forbidden')))
        self.enterContext(patch.object(q, 'matching_processes', return_value=[]))
        self.enterContext(patch.object(q, 'unit_info', return_value=self.inactive()))
        self.enterContext(patch.object(q, 'file_lock', side_effect=lambda *a, **k: contextlib.nullcontext()))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.config = {'training': {'max_steps': 50000, 'schedule_steps': 50000}}
        self.make_segment(45000)
        self.template = self.make_template()
        q.write_json(q.TEMPLATE, dict(argv=self.template, source_sha256='4' * 64))
        self.original_checkpoint = q.checkpoint
        original = self.original_checkpoint
        self.original_checkpoint = original
        self.enterContext(patch.object(q, 'checkpoint', side_effect=lambda *a, **kw: original(*a, **kw, loader=q.read_json)))
        self.enterContext(patch.object(q, 'runtime_paths', return_value=[q.CONFIG, q.TEMPLATE]))
        self.enterContext(patch.object(q, 'runtime_reference', return_value={}))
        self.enterContext(patch.object(q, 'audit_sources', return_value={'synthetic': True}))
        self.enterContext(patch.dict(sys.modules, {'summarize_task090_checkpoints': types.SimpleNamespace(summarize_run=self.summarize)}))
        self.rows = {}

    def inactive(self):
        return dict(LoadState='not-found', ActiveState='inactive', MainPID=0)

    def make_template(self):
        unit = q.train_unit(45000)
        return ['systemd-run', '--user', '--collect', '--unit=' + unit, '--working-directory=' + str(self.root),
                '-p', 'CPUQuota=800%', '-p', 'MemoryMax=16G', '-p', 'MemorySwapMax=0',
                '-p', f'ExecStartPre={q.PYTHON} {self.root}/scripts/gpu_lock.py acquire --owner {unit} --max-wait 0',
                '-p', f'ExecStopPost={q.PYTHON} {self.root}/scripts/gpu_lock.py release --owner {unit}',
                '-p', 'StandardOutput=append:old.log', str(q.PYTHON), str(self.root / 'scripts/train_wam_task_v3.py'),
                '--config', str(q.CONFIG), '--run-name', q.run_name(45000), '--max-steps', '45000',
                '--resume-source', str(q.train_run(40000) / 'checkpoint_last.pt'), '--resume-source-sha256', '4' * 64,
                '--device', 'cuda', '--num-workers', '0']

    def make_segment(self, step, source_sha='4' * 64):
        run = q.train_run(step)
        run.mkdir(parents=True)
        m = dict(task_id=90, mode='task_adaptation_50k', run_name=q.run_name(step), requested_updates=step,
                 steps_start_at=step - 5000, schedule_steps=50000, fresh_optimizer=False, offline_proxy_only=True,
                 config=self.config, source_sha256={str(q.CONFIG): q.sha(q.CONFIG)}, feature_payload_stats={},
                 training_contract_sha256='c' * 64, initialization_sha256='i' * 64,
                 resume_source=str(q.train_run(step - 5000) / 'checkpoint_last.pt'), resume_source_sha256=source_sha)
        q.write_json(run / 'manifest.json', m)
        q.write_json(run / 'status.json', dict(status='segment_complete', step=step))
        q.write_json(run / 'summary.json', dict(completed_updates=step, resumed_from_step=step - 5000))
        for tag in ('last', f'step_{step:06d}', 'final'):
            ck = dict(step=step, model_only=tag != 'last', mode=m['mode'], config=self.config,
                      training_contract_sha256=m['training_contract_sha256'], manifest_sha256=q.sha(run / 'manifest.json'),
                      initialization_sha256=m['initialization_sha256'])
            if tag == 'last':
                ck.update(resumable=True, resume_format='wam-v3-last-v2', optimizer_state_saved=True,
                          resume_state=dict(optimizer_state_dict={}, scheduler_state_dict={'last_epoch': step}, rng={},
                                            cursor={}, best={}, step=step, safe_optimizer_boundary=True, device_type='cuda'))
            self.write_ck(run / f'checkpoint_{tag}.pt', ck)
        return run

    def write_ck(self, path, ck):
        q.write_json(path, ck)
        q.write_json(str(path) + '.sha256.json', dict(step=ck['step'], size=path.stat().st_size,
                                                    model_only=ck['model_only'], sha256=q.sha(path)))

    def summarize(self, run):
        return [copy.deepcopy(self.rows[str(run)])]

    def make_triplet(self, step):
        path = self.out / f'triplet_{step:06d}'
        path.mkdir()
        digest = q.sha(q.train_run(step) / f'checkpoint_step_{step:06d}.pt')
        rows = []
        for seed, run in enumerate(q.triplet_runs(step)):
            run.mkdir(parents=True)
            row = dict(run=run.name, status='scored', seed=seed, instance_id=311, checkpoint_step=step,
                       checkpoint_sha256=digest, policy_step_alignment=True, interface_sha256='same',
                       first_grasp_success=False, task_success=False, evaluation_complete=seed == 0,
                       evaluator_returncode=0 if seed == 0 else 1)
            rows.append(row)
            self.rows[str(run)] = row
        q.write_json(path / 'state.json', dict(status='triplet_complete', checkpoint_step=step, seeds=[0, 1, 2], completed=rows))
        return path

    def fake_launch(self, cmd, **kwargs):
        if '--run-name' in cmd:
            self.make_segment(50000, cmd[cmd.index('--resume-source-sha256') + 1])
        elif '--triplet-entry' in cmd:
            self.make_triplet(int(cmd[cmd.index('--step') + 1]))
        else:
            self.fail('unexpected external command')
        return subprocess.CompletedProcess(cmd, 0)

    def test_training_command_preserves_budget_lock_and_full_last(self):
        cmd = q.training_command(self.out, 'a' * 64)
        self.assertIn('MemoryMax=16G', cmd)
        self.assertIn('MemorySwapMax=0', cmd)
        self.assertIn('CPUQuota=800%', cmd)
        self.assertEqual(cmd[cmd.index('--max-steps') + 1], '50000')
        self.assertEqual(cmd[cmd.index('--resume-source') + 1], str(q.train_run(45000) / 'checkpoint_last.pt'))
        self.assertIn('--owner ' + q.train_unit(50000), ' '.join(cmd))
        self.assertEqual(cmd[cmd.index('--resume-source-sha256') + 1], 'a' * 64)

    def test_resource_budget_mutation_stops(self):
        cmd = self.template.copy()
        cmd[cmd.index('MemoryMax=16G')] = 'MemoryMax=32G'
        q.write_json(q.TEMPLATE, {'argv': cmd})
        with self.assertRaisesRegex(RuntimeError, 'resource budget'):
            q.training_command(self.out, 'a' * 64)

    def test_missing_lock_hook_stops(self):
        cmd = [v for v in self.template if not v.startswith('ExecStopPost=')]
        q.write_json(q.TEMPLATE, {'argv': cmd})
        with self.assertRaisesRegex(RuntimeError, 'lock ownership'):
            q.training_command(self.out, 'a' * 64)

    def test_50k_final_is_model_only_and_last_resumable(self):
        self.make_segment(50000)
        result = q.complete_segment(50000, source_sha='4' * 64)
        self.assertTrue(result['final']['model_only'])
        self.assertFalse(result['last']['model_only'])
        self.assertEqual(result['final']['step'], 50000)

    def test_50k_requires_final_and_fixed(self):
        run = self.make_segment(50000)
        (run / 'checkpoint_final.pt').unlink()
        with self.assertRaises(FileNotFoundError):
            q.complete_segment(50000)
        (q.train_run(45000) / 'checkpoint_step_045000.pt').unlink()
        with self.assertRaises(FileNotFoundError):
            q.complete_segment(45000)

    def test_terminal_step_field_not_global_step(self):
        q.write_json(q.train_run(45000) / 'status.json', dict(status='segment_complete', global_step=45000))
        with self.assertRaisesRegex(RuntimeError, 'not a completed'):
            q.complete_segment(45000)

    def test_nonterminal_status_cannot_pass(self):
        q.write_json(q.train_run(45000) / 'status.json', dict(status='preflight', step=45000))
        with self.assertRaisesRegex(RuntimeError, 'not a completed'):
            q.complete_segment(45000)

    def test_wrong_source_sha_stops(self):
        self.make_segment(50000)
        with self.assertRaisesRegex(RuntimeError, 'different last'):
            q.complete_segment(50000, source_sha='wrong')

    def test_unsafe_last_or_scheduler_rejected(self):
        path = q.train_run(45000) / 'checkpoint_last.pt'
        good = q.read_json(path)
        for bad in ('safe', 'scheduler', 'resumable', 'rng'):
            ck = copy.deepcopy(good)
            if bad == 'safe':
                ck['resume_state']['safe_optimizer_boundary'] = False
            elif bad == 'scheduler':
                ck['resume_state']['scheduler_state_dict']['last_epoch'] = 40000
            elif bad == 'resumable':
                ck['resumable'] = False
            else:
                del ck['resume_state']['rng']
            self.write_ck(path, ck)
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, 'unsafe resume'):
                q.complete_segment(45000)

    def test_hash_failure_precedes_deserialization(self):
        path = q.train_run(45000) / 'checkpoint_last.pt'
        text = path.read_text()
        path.write_text(text.replace('true', 'null', 1))  # Same length, invalid digest.
        loader = Mock()
        with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
            self.original_checkpoint(path, 45000, False, q.manifest_identity(45000), loader=loader)
        loader.assert_not_called()

    def test_metadata_mismatch_with_valid_hash_rejected(self):
        path = q.train_run(45000) / 'checkpoint_step_045000.pt'
        ck = q.read_json(path)
        ck['training_contract_sha256'] = 'wrong'
        self.write_ck(path, ck)
        with self.assertRaisesRegex(RuntimeError, 'metadata identity'):
            q.complete_segment(45000)

    def test_old_queue_refusal_precedes_any_progress_or_output_write(self):
        before = (self.root / 'progress.md').read_bytes()
        q.unit_info.return_value = dict(LoadState='loaded', ActiveState='active', MainPID=123)
        with self.assertRaisesRegex(RuntimeError, 'old queue still running'):
            q.Queue(self.out).prepare()
        self.assertFalse(self.out.exists())
        self.assertEqual((self.root / 'progress.md').read_bytes(), before)

    def test_old_queue_orphan_refused(self):
        q.matching_processes.return_value = [123]
        with self.assertRaisesRegex(RuntimeError, 'old queue process'):
            q.old_queue_absent()

    def test_live_unit_requires_pid_argv_and_independent_cgroup(self):
        unit, expected = q.train_unit(45000), q.command_tail(self.template)
        info = dict(LoadState='loaded', ActiveState='active', MainPID=q.SOURCE_PID, ControlGroup='/app.slice/' + unit + '.service')
        q.unit_info.return_value = info
        with patch.object(q, 'process_argv', return_value=expected):
            self.assertEqual(q.validate_live_unit(unit, expected, expected_pid=q.SOURCE_PID), q.SOURCE_PID)
            with self.assertRaisesRegex(RuntimeError, 'wrong trainer PID'):
                q.validate_live_unit(unit, expected, expected_pid=999)
            info['ControlGroup'] = '/old-queue.service'
            with self.assertRaisesRegex(RuntimeError, 'not independent'):
                q.validate_live_unit(unit, expected)
        with patch.object(q, 'process_argv', return_value=['wrong']):
            with self.assertRaisesRegex(RuntimeError, 'argv identity'):
                q.validate_live_unit(unit, expected)

    def test_failed_unit_is_not_training_success(self):
        q.unit_info.return_value = dict(LoadState='loaded', ActiveState='failed', MainPID=0)
        with self.assertRaisesRegex(RuntimeError, 'unit failed'):
            q.validate_live_unit(q.train_unit(45000), q.command_tail(self.template))

    def test_waiting_training_records_step_without_launch(self):
        queue = q.Queue(self.out)
        queue.prepare()
        (q.train_run(45000) / 'log.jsonl').write_text(json.dumps(dict(event='train', step=44320, ram_percent=40)) + '\n')
        with patch.object(queue, 'guard'), patch.object(q, 'validate_live_unit', side_effect=[q.SOURCE_PID, None]), \
                patch.object(q, 'gpu_owner', return_value=q.train_unit(45000)), patch.object(q.time, 'sleep') as sleep:
            queue.wait_training(45000, self.template)
        sleep.assert_called_once_with(60)
        records = [json.loads(x) for x in (self.out / 'status.jsonl').read_text().splitlines()]
        self.assertTrue(any(r.get('observed_step') == 44320 for r in records))
        q.subprocess.run.assert_not_called()

    def test_gpu_occupied_prevents_dispatch_and_never_releases(self):
        queue = q.Queue(self.out)
        queue.prepare()
        with patch.object(q, 'gpu_owner', return_value='other-job'), patch.object(q.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'GPU lock still occupied'):
                queue.dispatch(self.out / 'launch.json', 'u', ['fake'])
        self.assertFalse((self.out / 'launch.json').exists())
        q.subprocess.run.assert_not_called()

    def test_durable_intent_before_side_effect_and_no_second_dispatch(self):
        queue = q.Queue(self.out)
        queue.prepare()
        launch = self.out / 'launch.json'
        def invoked(cmd, **kw):
            self.assertTrue(launch.exists())
            self.assertEqual(q.read_json(self.out / 'state.json')['status'], 'launch_intent')
            self.assertIn('launch_intent', (self.root / 'progress.md').read_text())
            return subprocess.CompletedProcess(cmd, 0)
        q.subprocess.run.side_effect = invoked
        queue.dispatch(launch, 'test', ['fake'])
        with self.assertRaisesRegex(RuntimeError, 'intent exists'):
            queue.dispatch(launch, 'test', ['fake'])
        self.assertEqual(q.subprocess.run.call_count, 1)

    def test_launch_exception_retains_intent_and_error(self):
        q.subprocess.run.side_effect = RuntimeError('systemd submission failed')
        with self.assertRaisesRegex(RuntimeError, 'submission failed'):
            q.Queue(self.out).run()
        self.assertTrue((self.out / 'train_050000_launch.json').exists())
        self.assertEqual(q.read_json(self.out / 'state.json')['status'], 'stopped_on_error')
        with self.assertRaisesRegex(RuntimeError, 'stopped on error'):
            q.Queue(self.out).run()
        self.assertEqual(q.subprocess.run.call_count, 1)

    def test_ambiguous_intent_is_never_repeated(self):
        queue = q.Queue(self.out)
        queue.prepare()
        digest = q.sha(q.train_run(45000) / 'checkpoint_last.pt')
        q.write_json(self.out / 'train_050000_launch.json', dict(argv=q.training_command(self.out, digest), source_sha256=digest))
        with self.assertRaisesRegex(RuntimeError, 'ambiguous, not repeated'):
            q.Queue(self.out).run()
        q.subprocess.run.assert_not_called()

    def test_end_to_end_order_and_restart_no_repetition(self):
        q.subprocess.run.side_effect = self.fake_launch
        q.Queue(self.out).run()
        self.assertEqual(q.subprocess.run.call_count, 3)
        calls = [c.args[0] for c in q.subprocess.run.call_args_list]
        self.assertIn('--max-steps', calls[0])
        self.assertEqual([cmd[cmd.index('--step') + 1] for cmd in calls[1:]], ['45000', '50000'])
        q.Queue(self.out).run()
        self.assertEqual(q.subprocess.run.call_count, 3)
        state = q.read_json(self.out / 'state.json')
        self.assertEqual(state['status'], 'authorized_budget_and_deferred_checks_complete')
        self.assertEqual([row['trials'] for row in state['completed']], [3, 3])
        self.assertEqual([row['task_successes'] for row in state['completed']], [0, 0])
        self.assertEqual([row['clean_exits'] for row in state['completed']], [1, 1])

    def test_resume_while_later_stage_owns_gpu_does_not_block_old_verification(self):
        q.subprocess.run.side_effect = self.fake_launch
        q.Queue(self.out).run()
        with patch.object(q, 'gpu_owner', return_value='already-owned-later-stage'):
            q.Queue(self.out).run()
        self.assertEqual(q.subprocess.run.call_count, 3)

    def test_live_50k_is_adopted_without_second_trainer(self):
        queue = q.Queue(self.out)
        queue.prepare()
        digest = q.sha(q.train_run(45000) / 'checkpoint_last.pt')
        self.make_segment(50000, digest)
        cmd = q.training_command(self.out, digest)
        q.write_json(self.out / 'train_050000_launch.json', dict(argv=cmd, source_sha256=digest))
        alive = [True]
        def unit_info(unit):
            if unit == q.train_unit(50000) and alive[0]:
                return dict(LoadState='loaded', ActiveState='active', MainPID=777,
                            ControlGroup='/app.slice/' + unit + '.service')
            return self.inactive()
        def tick(seconds):
            self.assertEqual(seconds, 60)
            alive[0] = False
        q.subprocess.run.side_effect = self.fake_launch
        with patch.object(q, 'unit_info', side_effect=unit_info), \
                patch.object(q, 'process_argv', return_value=q.command_tail(cmd)), \
                patch.object(q, 'gpu_owner', side_effect=lambda: q.train_unit(50000) if alive[0] else None), \
                patch.object(q.time, 'sleep', side_effect=tick):
            q.Queue(self.out).run()
        self.assertEqual(q.subprocess.run.call_count, 2)
        self.assertTrue(all('--triplet-entry' in call.args[0] for call in q.subprocess.run.call_args_list))

    def test_live_triplet_is_adopted_without_duplicate_tests(self):
        queue = q.Queue(self.out)
        queue.prepare()
        digest = q.sha(q.train_run(45000) / 'checkpoint_last.pt')
        self.make_segment(50000, digest)
        q.write_json(self.out / 'train_050000_launch.json', dict(argv=q.training_command(self.out, digest), source_sha256=digest))
        unit, cmd = q.triplet_command(self.out, 45000)
        q.write_json(self.out / 'triplet_045000_launch.json', dict(argv=cmd))
        alive = [True]
        def unit_info(name):
            if name == unit and alive[0]:
                return dict(LoadState='loaded', ActiveState='active', MainPID=888,
                            ControlGroup='/app.slice/' + name + '.service')
            return self.inactive()
        def tick(seconds):
            self.assertEqual(seconds, 60)
            self.make_triplet(45000)
            alive[0] = False
        q.subprocess.run.side_effect = self.fake_launch
        with patch.object(q, 'unit_info', side_effect=unit_info), \
                patch.object(q, 'process_argv', return_value=q.command_tail(cmd)), \
                patch.object(q.time, 'sleep', side_effect=tick):
            q.Queue(self.out).run()
        self.assertEqual(q.subprocess.run.call_count, 1)
        final = q.subprocess.run.call_args.args[0]
        self.assertEqual(final[final.index('--step') + 1], '50000')

    def test_internal_triplet_cannot_bypass_queue_launch(self):
        with self.assertRaises(FileNotFoundError):
            q.main(['--triplet-entry', '--step', '45000', '--out', str(self.out / 'triplet_045000')])
        q.subprocess.run.assert_not_called()

    def test_existing_unowned_50k_not_overwritten(self):
        self.make_segment(50000)
        with self.assertRaisesRegex(RuntimeError, 'unowned 50k'):
            q.Queue(self.out).run()
        q.subprocess.run.assert_not_called()

    def test_existing_output_without_manifest_refused(self):
        self.out.mkdir()
        (self.out / 'evidence.txt').write_text('keep')
        with self.assertRaisesRegex(RuntimeError, 'lacks durable'):
            q.Queue(self.out).prepare()
        self.assertEqual((self.out / 'evidence.txt').read_text(), 'keep')

    def test_partial_triplet_refused_without_repeat(self):
        q.subprocess.run.side_effect = self.fake_launch
        q.Queue(self.out).run()
        path = self.out / 'triplet_045000/state.json'
        state = q.read_json(path)
        state['status'] = 'running'
        state['completed'] = state['completed'][:1]
        q.write_json(path, state)
        with self.assertRaisesRegex(RuntimeError, 'partial triplet'):
            q.Queue(self.out).run()
        self.assertEqual(q.subprocess.run.call_count, 3)

    def test_trial_identity_or_raw_evidence_mismatch_stops(self):
        self.out.mkdir()
        path = self.make_triplet(45000)
        fixed = q.sha(q.train_run(45000) / 'checkpoint_step_045000.pt')
        original = q.read_json(path / 'state.json')
        for key, value in [('seed', 8), ('instance_id', 0), ('checkpoint_sha256', 'bad'), ('policy_step_alignment', False),
                           ('interface_sha256', 'changed'), ('task_success', True)]:
            state = copy.deepcopy(original)
            state['completed'][1][key] = value
            q.write_json(path / 'state.json', state)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                q.verify_triplet(path, 45000, fixed)

    def test_progress_updates_preserve_other_sections(self):
        before = (self.root / 'progress.md').read_text()
        updated = q.merge_queue_progress(before, '- waiting\n')
        updated = q.replace_section(updated, q.TRIPLET_MARKER, '- seed0\n')
        updated = q.merge_queue_progress(updated, '- done\n')
        self.assertIn('retain this', updated)
        self.assertIn('User notes', updated)
        self.assertEqual(updated.count(q.BEGIN), 1)
        self.assertLess(updated.index(q.BEGIN), updated.index(q.TRIPLET_MARKER))
        self.assertIn('- seed0', updated)

    def test_duplicate_progress_markers_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'broken queue'):
            q.merge_queue_progress(q.BEGIN, 'text')

    def test_legacy_triplet_entry_progress_adapter_only(self):
        def legacy_main():
            path = self.root / 'progress.md'
            text = path.read_text().split(q.TRIPLET_MARKER)[0]
            path.write_text(text + q.TRIPLET_MARKER + 'new triplet\n')
            (self.root / 'unchanged_writer.json').write_text('{}')
        with patch.dict(sys.modules, {'run_task090_checkpoint_triplet': types.SimpleNamespace(main=legacy_main)}):
            q.triplet_entry(45000, self.out)
        self.assertIn('retain this', (self.root / 'progress.md').read_text())
        self.assertIn('new triplet', (self.root / 'progress.md').read_text())
        self.assertEqual((self.root / 'unchanged_writer.json').read_text(), '{}')

    def test_check_only_leaves_progress_and_output_untouched(self):
        before = (self.root / 'progress.md').read_bytes()
        with patch.object(q, 'old_queue_absent', side_effect=RuntimeError('old queue still running')):
            self.assertEqual(q.main(['--check-only', '--out', str(self.out)]), 2)
        self.assertFalse(self.out.exists())
        self.assertEqual((self.root / 'progress.md').read_bytes(), before)
        q.subprocess.run.assert_not_called()


class IndependentGuards(unittest.TestCase):
    def test_source_ledger_and_payload_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, payload = root / 'source.py', root / 'feature.npy'
            source.write_bytes(b'a')
            payload.write_bytes(b'b')
            m = dict(source_sha256={str(source): q.sha(source)}, feature_payload_stats={str(payload): q.stamp(payload)})
            with patch.object(q, 'ROOT', root):
                result = q.audit_sources(m)
                self.assertEqual(result['source_hashes'], 1)
                source.write_bytes(b'c')
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    q.audit_sources(m)
                source.write_bytes(b'a')
                payload.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'payload identity'):
                    q.audit_sources(m)

    def test_new_scheduler_cannot_be_in_training_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / 'scripts/run_task090_training_first.py'
            p.parent.mkdir()
            p.write_text('synthetic')
            with patch.object(q, 'ROOT', root), self.assertRaisesRegex(RuntimeError, 'frozen training contract'):
                q.audit_sources(dict(source_sha256={str(p): q.sha(p)}, feature_payload_stats={}))

    def test_global_queue_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queue.lock'
            with q.file_lock(path, nonblocking=True):
                with self.assertRaises(BlockingIOError):
                    with q.file_lock(path, nonblocking=True):
                        self.fail('duplicate queue admitted')
            with q.file_lock(path, nonblocking=True):
                pass

    def test_systemctl_read_error_is_not_inactive(self):
        with patch.object(q.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'bus unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'cannot inspect unit'):
                q.unit_info('test')

    def test_missing_unit_is_accepted_only_when_explicit(self):
        result = subprocess.CompletedProcess([], 4, 'LoadState=not-found\nActiveState=inactive\nMainPID=0\n', '')
        with patch.object(q.subprocess, 'run', return_value=result):
            self.assertEqual(q.unit_info('test')['MainPID'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
