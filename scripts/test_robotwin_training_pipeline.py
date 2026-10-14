#!/usr/bin/env python3
"""Independent stdlib/CPU tests; no torch, simulator, systemd jobs or GPU queries.

    python3 -B -m unittest scripts.test_robotwin_training_pipeline -v
"""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_robotwin_training_pipeline as p


class FakeLock:
    def __init__(self, events, foreign=None, fail=False, interrupted=False):
        self.events, self.owner = events, foreign
        self.fail, self.interrupted = fail, interrupted

    def acquire(self, name, max_wait):
        self.events.append(('acquire', name))
        if self.fail:
            raise TimeoutError('foreign session busy')
        if self.owner is not None:
            raise AssertionError('lock held already')
        self.owner = name
        if self.interrupted:
            raise KeyboardInterrupt('interrupted immediately after lock creation')

    def release(self, name):
        self.events.append(('release', name))
        if self.owner == name:
            self.owner = None
            return True
        return False


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.source = self.data / 'SOURCE.json'
        self.put(self.source, {'task': 'fruit_task_selected_by_operator'})
        self.split = self.root / 'split.json'
        self.put(self.split, {'data_root': str(self.data),
                             'splits': {'train': ['episode_0000000.hdf5'], 'val': ['episode_0000001.hdf5']}})
        self.cfg = {'dataset': {'task_name': 'fruit_task_selected_by_operator',
                               'split_manifest': str(self.split), 'split_sha256': p.sha256(self.split),
                               'feature_cache': str(self.root / 'cache')},
                    'training': {'max_steps': 50000, 'schedule_steps': 50000, 'learning_rate': 5e-5}}
        self.config = self.root / 'config.json'
        self.put(self.config, self.cfg)
        self.rgb = self.root / 'first_frame.png'
        # Evidence is only hash-bound operator attestation, never image decoding.
        self.rgb.write_bytes(b'CPU fixture, not real visual approval')
        self.native = self.root / 'native.json'
        self.put(self.native, {'status': 'native_first_frame_complete'})
        self.audit = {'schema_version': 1, 'task': self.cfg['dataset']['task_name'],
                      'source_split_sha256': p.sha256(self.split), **{k: True for k in p.FLAGS},
                      'operator': 'CPU test operator', 'confirmed_at': '2026-10-07T10:00:00+08:00',
                      'note': 'Test-only attestation; no actual robot or GPU is invoked.',
                      'evidence': {'first_frame_rgb': p.identity(self.rgb), 'native_preflight': p.identity(self.native)}}
        self.admission = self.root / 'admission.json'
        self.put(self.admission, self.audit)
        self.out = self.root / 'pipeline'
        self.argv = ['--config', str(self.config), '--admission-json', str(self.admission),
                     '--out', str(self.out), '--unit', 'rw-cpu-test', '--python', sys.executable]
        self.events = []
        self.lock = FakeLock(self.events)

    def put(self, path, value):
        Path(path).write_text(json.dumps(value))

    def plan(self, *args):
        return p.build_plan(p.parse_args(self.argv + list(args)))

    def prepare(self):
        plan = self.plan()
        self.out.mkdir()
        (self.out / 'logs').mkdir()
        return plan

    def runner(self, command, log, *, gpu):
        name = Path(log).stem.split('_', 1)[1]
        self.events.append(('run', name, gpu, self.lock.owner))
        self.assertEqual(gpu, self.lock.owner is not None)
        Path(log).write_text('fake child\n')
        if name == 'feature_cache':
            return 0
        out = Path(command[command.index('--out') + 1])
        out.mkdir(exist_ok=False)
        if name == 'cpu_preflight':
            self.put(out / 'plan.json', {'status': 'cache_preflight_passed_not_a_smoke', 'mode': 'plan',
                                        'model_constructed': False, 'schedule_budget': 50000})
        else:
            mode, step = ('smoke', 2) if name == 'cuda_smoke' else ('train', 30000)
            self.put(out / 'status.json', {'status': 'bounded_complete', 'mode': mode,
                                         'step': step, 'target': step, 'schedule_budget': 50000})
            if name == 'train':
                for mark in (25000, 30000):
                    ck = out / f'checkpoint_step_{mark:06d}.pt'
                    ck.write_bytes(b'CPU test checkpoint placeholder')
                    self.put(ck.with_name(ck.name + '.sha256.json'),
                             {'sha256': p.sha256(ck), 'step': mark, 'model_only': True, 'size': ck.stat().st_size})
        return 0

    def run_worker(self, plan, runner=None, lock=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return p.run_pipeline(plan, runner=runner or self.runner, lock=lock or self.lock)

    def state(self):
        return p.read_json(self.out / 'pipeline_status.json')

    def test_default_plan_no_writes_no_subprocess_no_gpu_import(self):
        before = {str(x): x.read_bytes() for x in self.root.rglob('*') if x.is_file()}
        output = io.StringIO()
        with patch.object(p.subprocess, 'run', side_effect=AssertionError('subprocess forbidden')), \
             patch.object(p.subprocess, 'Popen', side_effect=AssertionError('child forbidden')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(p.main(self.argv), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value['status'], 'plan_only_no_jobs_started')
        self.assertFalse(self.out.exists())
        self.assertEqual(before, {str(x): x.read_bytes() for x in self.root.rglob('*') if x.is_file()})

    def test_independent_import_does_not_load_gpu_stack(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys; from scripts import run_robotwin_training_pipeline; '
            'assert not any(x in sys.modules for x in ("torch", "cv2", "numpy", "scripts.gpu_lock"))'],
            cwd=p.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_command_chain_numerics_paths_and_posttrain_handoff(self):
        original = self.config.read_bytes()
        plan = self.plan()
        stages = plan['stages']
        self.assertEqual([x['name'] for x in stages], ['feature_cache', 'cpu_preflight', 'cuda_smoke', 'train'])
        self.assertEqual([x['gpu'] for x in stages], [True, False, True, True])
        cache, cpu, smoke, train = [x['command'] for x in stages]
        self.assertEqual(cache[cache.index('--splits') + 1:cache.index('--batch')], ['train', 'val'])
        self.assertEqual(cpu[cpu.index('--device') + 1], 'cpu')
        self.assertEqual(cpu[cpu.index('--mode') + 1], 'plan')
        self.assertNotIn('--gpu-lock-owner', cpu)
        self.assertEqual(smoke[smoke.index('--smoke-steps') + 1], '2')
        self.assertEqual(train[train.index('--max-steps') + 1], '30000')
        self.assertEqual(train[train.index('--gpu-lock-owner') + 1], plan['lock_owner'])
        for command in (cpu, smoke, train):
            self.assertEqual(command[command.index('--config') + 1], str(self.config))
            self.assertNotIn('--resume', command)
            self.assertNotIn('--cuda-memory-fraction', command)
        self.assertEqual(original, self.config.read_bytes())
        self.assertEqual(plan['schedule_budget'], 50000)
        self.assertEqual(plan['posttrain'], {'training_run_dir': str(self.out / 'train'), 'training_unit': 'rw-cpu-test'})
        self.assertNotEqual(plan['lock_owner'], self.plan()['lock_owner'])

    def test_venv_python_symlink_not_resolved(self):
        venv = self.root / 'venv' / 'bin' / 'python'
        venv.parent.mkdir(parents=True)
        venv.symlink_to(sys.executable)
        plan = self.plan('--python', str(venv))
        self.assertEqual(plan['python'], str(venv))
        self.assertEqual(plan['stages'][0]['command'][0], str(venv))

    def test_template_is_fail_closed_and_cli_only_prints(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(p.subprocess, 'run') as run:
            self.assertEqual(p.main(['--print-admission-template']), 0)
        run.assert_not_called()
        template = json.loads(output.getvalue())
        self.assertEqual(set(template), p.ADMISSION_KEYS)
        self.assertTrue(all(template[k] is False for k in p.FLAGS))

    def test_strict_admission_rejects_unapproved_wrong_task_hash_types_and_keys(self):
        mutations = [('task', 'wrong_task'), ('source_split_sha256', '0' * 64), ('schema_version', True),
                     ('operator', ''), ('note', '<placeholder>'), ('confirmed_at', '2026-10-07T10:00:00'),
                     ('unexpected', True)]
        mutations += [(key, value) for key in p.FLAGS for value in (False, 1, 'true', None)]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                audit = copy.deepcopy(self.audit)
                audit[key] = value
                self.put(self.admission, audit)
                with self.assertRaises(ValueError):
                    self.plan()
        self.put(self.admission, {k: v for k, v in self.audit.items() if k != 'interface_accepted'})
        with self.assertRaises(ValueError):
            self.plan()
        self.assertFalse(self.out.exists())

    def test_duplicate_json_keys_and_nonfinite_values_rejected(self):
        for text in ('{"task":"a","task":"b"}', '{"a":NaN}'):
            self.admission.write_text(text)
            with self.assertRaises(ValueError):
                p.read_json(self.admission)

    def test_evidence_missing_or_changed_rejected(self):
        self.rgb.write_bytes(b'changed RGB')
        with self.assertRaisesRegex(ValueError, 'evidence SHA mismatch'):
            self.plan()
        self.rgb.unlink()
        with self.assertRaisesRegex(ValueError, 'required nonempty file'):
            self.plan()

    def test_missing_config_wrong_split_source_and_schedule_rejected(self):
        cases = ['split', 'source', 'max_steps', 'schedule_steps', 'config']
        for case in cases:
            with self.subTest(case=case):
                old_config, old_split, old_source = (x.read_bytes() for x in (self.config, self.split, self.source))
                if case == 'split':
                    self.split.write_text(self.split.read_text() + '\n')
                elif case == 'source':
                    self.put(self.source, {'task': 'wrong'})
                elif case == 'config':
                    self.config.unlink()
                else:
                    cfg = copy.deepcopy(self.cfg)
                    cfg['training'][case] = 30000
                    self.put(self.config, cfg)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.plan()
                for path, content in zip((self.config, self.split, self.source), (old_config, old_split, old_source)):
                    path.write_bytes(content)

    def test_existing_directory_file_and_dangling_symlink_never_overwritten(self):
        for kind in ('directory', 'file', 'symlink'):
            with self.subTest(kind=kind):
                if kind == 'directory':
                    self.out.mkdir()
                elif kind == 'file':
                    self.out.write_text('untouched')
                else:
                    self.out.symlink_to(self.root / 'missing')
                with self.assertRaises(FileExistsError):
                    self.plan()
                if kind == 'directory':
                    self.out.rmdir()
                else:
                    self.out.unlink()

    def test_invalid_options_and_cache_overlap_rejected(self):
        for flags in (('--unit', 'bad.service'), ('--unit', 'x;y'), ('--cache-batch', '0'),
                      ('--gpu-lock-wait-seconds', '-1'), ('--gpu-lock-wait-seconds', 'nan'),
                      ('--gpu-lock-wait-seconds', 'inf'), ('--out', str(self.root / 'cache' / 'new'))):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.plan(*flags)

    def test_systemd_contract_retained_no_restart_no_collection_or_shell(self):
        plan = self.plan()
        command = p.systemd_command(plan, 'abc')
        for arg in ('--property=RemainAfterExit=yes', '--property=Restart=no',
                    '--property=KillMode=control-group', '--property=TimeoutStopSec=120', '--service-type=exec'):
            self.assertIn(arg, command)
        self.assertIn('--setenv=HF_HUB_OFFLINE=1', command)
        self.assertIn('--setenv=TRANSFORMERS_OFFLINE=1', command)
        self.assertTrue(any(x.startswith('--property=ExecStopPost=') and
                            x.endswith(' release --owner ' + plan['lock_owner']) for x in command))
        for unsafe in ('--collect', '--scope', '--wait', '--pipe', '--pty', '/bin/bash', '--resume'):
            self.assertNotIn(unsafe, command)
        self.assertEqual(command[command.index('--_manifest-sha') + 1], 'abc')

    def test_serial_success_locks_only_gpu_stages_finally_released_and_status_logged(self):
        plan = self.prepare()
        self.assertEqual(self.run_worker(plan), 0)
        self.assertIsNone(self.lock.owner)
        self.assertEqual([e[0] for e in self.events],
                         ['acquire', 'run', 'release', 'run', 'acquire', 'run', 'release', 'acquire', 'run', 'release'])
        self.assertEqual([e[1] for e in self.events if e[0] == 'run'],
                         ['feature_cache', 'cpu_preflight', 'cuda_smoke', 'train'])
        self.assertEqual(self.state()['status'], 'complete')
        self.assertTrue(all(s['status'] == 'complete' for s in self.state()['stages'].values()))
        self.assertEqual(len(list((self.out / 'logs').glob('*.log'))), 4)
        history = [json.loads(x) for x in (self.out / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(history[-1]['status'], 'complete')
        self.assertIn('waiting_gpu', [x['status'] for x in history])
        self.assertEqual(p.read_json(self.out / 'train' / 'status.json')['step'], 30000)

    def test_every_stage_failure_stops_no_retry(self):
        for fail_stage in ('feature_cache', 'cpu_preflight', 'cuda_smoke', 'train'):
            with self.subTest(stage=fail_stage):
                self.out = self.root / fail_stage
                plan = self.plan('--out', str(self.out))
                self.out.mkdir()
                (self.out / 'logs').mkdir()
                calls = []
                def runner(command, log, *, gpu):
                    name = Path(log).stem.split('_', 1)[1]
                    calls.append(name)
                    if name == fail_stage:
                        return 7
                    return self.runner(command, log, gpu=gpu)
                self.assertEqual(self.run_worker(plan, runner), 1)
                self.assertEqual(self.state()['status'], 'failed')
                self.assertEqual(self.state()['stage'], fail_stage)
                self.assertEqual(calls.count(fail_stage), 1)
                self.assertEqual(calls[-1], fail_stage)
                self.assertIsNone(self.lock.owner)
                self.assertEqual(self.state()['stages'][fail_stage]['returncode'], 7)

    def test_zero_exit_without_cpu_certificate_stops_before_smoke(self):
        plan = self.prepare()
        def runner(command, log, *, gpu):
            code = self.runner(command, log, gpu=gpu)
            if not gpu:
                self.put(self.out / 'cpu_preflight' / 'plan.json', {'status': 'blocked_preflight'})
            return code
        self.assertEqual(self.run_worker(plan, runner), 1)
        self.assertEqual(self.state()['stage'], 'cpu_preflight')
        self.assertFalse((self.out / 'cuda_smoke').exists())

    def test_partial_smoke_status_not_promoted_to_train(self):
        plan = self.prepare()
        def runner(command, log, *, gpu):
            code = self.runner(command, log, gpu=gpu)
            if '03_cuda_smoke' in log:
                status = self.out / 'cuda_smoke' / 'status.json'
                value = p.read_json(status)
                value['step'] = 1
                self.put(status, value)
            return code
        self.assertEqual(self.run_worker(plan, runner), 1)
        self.assertEqual(self.state()['stage'], 'cuda_smoke')
        self.assertFalse((self.out / 'train').exists())
        self.assertIsNone(self.lock.owner)

    def test_missing_posttrain_checkpoint_fails_without_editing_trainer_status(self):
        plan = self.prepare()
        def runner(command, log, *, gpu):
            code = self.runner(command, log, gpu=gpu)
            if '04_train' in log:
                (self.out / 'train' / 'checkpoint_step_025000.pt').unlink()
            return code
        self.assertEqual(self.run_worker(plan, runner), 1)
        self.assertIn('missing posttrain checkpoint', self.state()['error'])
        self.assertEqual(p.read_json(self.out / 'train' / 'status.json')['status'], 'bounded_complete')

    def test_existing_child_output_refused_no_overwrite(self):
        plan = self.prepare()
        existing = self.out / 'cpu_preflight'
        existing.mkdir()
        sentinel = existing / 'sentinel'
        sentinel.write_text('unchanged')
        self.assertEqual(self.run_worker(plan), 1)
        self.assertEqual(sentinel.read_text(), 'unchanged')
        self.assertEqual([e[1] for e in self.events if e[0] == 'run'], ['feature_cache'])

    def test_claim_forbids_second_execution_even_after_success_or_failure(self):
        plan = self.prepare()
        self.assertEqual(self.run_worker(plan), 0)
        before = (self.out / 'pipeline_status.json').read_bytes()
        count = len(self.events)
        with self.assertRaises(FileExistsError):
            self.run_worker(plan)
        self.assertEqual(before, (self.out / 'pipeline_status.json').read_bytes())
        self.assertEqual(count, len(self.events))

    def test_changed_input_before_worker_never_acquires_lock(self):
        plan = self.prepare()
        self.config.write_text(self.config.read_text() + '\n')
        self.assertEqual(self.run_worker(plan), 1)
        self.assertEqual(self.events, [])
        self.assertIn('input changed', self.state()['error'])

    def test_changed_input_during_lock_wait_releases_without_child(self):
        plan = self.prepare()
        acquire = self.lock.acquire
        def change(name, max_wait):
            acquire(name, max_wait)
            self.rgb.write_bytes(b'new evidence')
        self.lock.acquire = change
        self.assertEqual(self.run_worker(plan), 1)
        self.assertEqual([x[0] for x in self.events], ['acquire', 'release'])
        self.assertIsNone(self.lock.owner)

    def test_busy_foreign_lock_is_never_stolen_or_retried(self):
        plan = self.prepare()
        self.lock = FakeLock(self.events, foreign='other-session', fail=True)
        self.assertEqual(self.run_worker(plan), 1)
        self.assertEqual(self.lock.owner, 'other-session')
        self.assertEqual([x[0] for x in self.events], ['acquire', 'release'])

    def test_interrupt_after_acquire_and_child_exception_release_lock(self):
        plan = self.prepare()
        self.lock = FakeLock(self.events, interrupted=True)
        self.assertEqual(self.run_worker(plan), 130)
        self.assertIsNone(self.lock.owner)
        self.assertEqual(self.state()['status'], 'stopped')
        with self.assertRaises(FileExistsError):
            self.run_worker(plan)

    def test_signal_exception_and_launch_exception_stop_and_release(self):
        for error in (p.StopPipeline('signal 15'), FileNotFoundError('interpreter missing')):
            with self.subTest(error=type(error).__name__):
                self.out = self.root / type(error).__name__
                plan = self.plan('--out', str(self.out))
                self.out.mkdir()
                (self.out / 'logs').mkdir()
                def fail(*args, **kwargs):
                    raise error
                self.assertIn(self.run_worker(plan, fail), (1, 130))
                self.assertIsNone(self.lock.owner)
                self.assertEqual(len([s for s in self.state()['stages'].values() if s['status'] == 'pending']), 3)

    def test_real_cpu_child_logs_and_offline_environment(self):
        log = self.root / 'child.log'
        code = ('import os; print("cpu-child-ok"); '
                'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""; '
                'assert os.environ["HF_HUB_OFFLINE"] == "1"; '
                'assert os.environ["TRANSFORMERS_OFFLINE"] == "1"')
        self.assertEqual(p.run_child([sys.executable, '-B', '-c', code], log, gpu=False), 0)
        self.assertIn('cpu-child-ok', log.read_text())
        with self.assertRaises(FileExistsError):
            p.run_child([sys.executable, '-c', 'pass'], log, gpu=False)

    def test_child_interruption_reaps_only_owned_group_before_propagating(self):
        child = SimpleNamespace(pid=234567, poll=lambda: None)
        events = []
        def wait(timeout=None):
            events.append(('wait', timeout))
            if timeout is None and len(events) == 1:
                raise p.StopPipeline('signal 15')
            return -15
        child.wait = wait
        with patch.object(p.subprocess, 'Popen', return_value=child), \
             patch.object(p.os, 'killpg', side_effect=lambda pid, sig: events.append(('killpg', pid, sig))):
            with self.assertRaises(p.StopPipeline):
                p.run_child(['fake-cpu-child'], self.root / 'stop.log', gpu=False)
        self.assertEqual(events, [('wait', None), ('killpg', 234567, signal.SIGTERM), ('wait', 60)])

    def test_unresponsive_child_is_killed_and_reaped_in_own_group(self):
        events = []
        child = SimpleNamespace(pid=234568, poll=lambda: None)
        def wait(timeout=None):
            events.append(('wait', timeout))
            if len(events) == 1:
                raise KeyboardInterrupt()
            if timeout == 60:
                raise subprocess.TimeoutExpired('fake', 60)
            return -9
        child.wait = wait
        with patch.object(p.subprocess, 'Popen', return_value=child), \
             patch.object(p.os, 'killpg', side_effect=lambda pid, sig: events.append(('killpg', pid, sig))):
            with self.assertRaises(KeyboardInterrupt):
                p.run_child(['fake-cpu-child'], self.root / 'kill.log', gpu=True)
        self.assertEqual(events, [('wait', None), ('killpg', 234568, signal.SIGTERM),
                                  ('wait', 60), ('killpg', 234568, signal.SIGKILL), ('wait', None)])

    def test_lock_ownership_lost_is_failure_without_deleting_foreign_lock(self):
        plan = self.prepare()
        def runner(*args, **kwargs):
            self.lock.owner = 'foreign-owner'
            return 0
        self.assertEqual(self.run_worker(plan, runner), 1)
        self.assertEqual(self.lock.owner, 'foreign-owner')
        self.assertIn('ownership lost', self.state()['error'])
        self.assertFalse((self.out / 'cpu_preflight').exists())

    def test_exact_train_status_rejects_partial_smoke_and_wrong_schedule(self):
        plan = self.plan()
        stage = plan['stages'][-1]
        self.out.mkdir()
        out = self.out / 'train'
        out.mkdir()
        good = {'status': 'bounded_complete', 'mode': 'train', 'step': 30000,
                'target': 30000, 'schedule_budget': 50000}
        for key, value in (('step', 25000), ('target', 50000), ('mode', 'smoke'),
                           ('schedule_budget', 30000), ('status', 'running'), ('step', 30000.0)):
            with self.subTest(key=key, value=value):
                self.put(out / 'status.json', {**good, key: value})
                with self.assertRaisesRegex(ValueError, 'exact target'):
                    p.stage_success(stage)

    def test_invalid_checkpoint_receipt_fails_after_training_without_retry(self):
        plan = self.prepare()
        def runner(command, log, *, gpu):
            code = self.runner(command, log, gpu=gpu)
            if '04_train' in log:
                receipt = self.out / 'train' / 'checkpoint_step_030000.pt.sha256.json'
                record = p.read_json(receipt)
                record['model_only'] = False
                self.put(receipt, record)
            return code
        self.assertEqual(self.run_worker(plan, runner), 1)
        self.assertIn('invalid posttrain checkpoint receipt', self.state()['error'])
        self.assertEqual(sum(x[:2] == ('run', 'train') for x in self.events), 1)
        self.assertIsNone(self.lock.owner)

    def test_input_change_during_final_training_prevents_pipeline_success(self):
        plan = self.prepare()
        def runner(command, log, *, gpu):
            code = self.runner(command, log, gpu=gpu)
            if '04_train' in log:
                self.admission.write_text(self.admission.read_text() + '\n')
            return code
        self.assertEqual(self.run_worker(plan, runner), 1)
        self.assertEqual(self.state()['stage'], 'train')
        self.assertIn('input changed', self.state()['error'])
        self.assertIsNone(self.lock.owner)

    def test_configuration_change_while_planning_cannot_be_admitted(self):
        load = p.load_config
        def change(path):
            value = load(path)
            self.config.write_text(self.config.read_text() + '\n')
            return value
        with patch.object(p, 'load_config', side_effect=change):
            with self.assertRaisesRegex(ValueError, 'input changed'):
                self.plan()

    def test_signals_restored_after_pipeline_failure(self):
        plan = self.prepare()
        before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        def fail(*args, **kwargs):
            raise KeyboardInterrupt()
        self.assertEqual(self.run_worker(plan, fail), 130)
        self.assertEqual(before, {s: signal.getsignal(s) for s in before})

    def system_reply(self, command, **kwargs):
        if command[0] == 'loginctl':
            return SimpleNamespace(stdout='yes\n', stderr='')
        if command[0] == 'systemctl':
            return SimpleNamespace(stdout='not-found\n', stderr='')
        if command[0] == 'systemd-run':
            return SimpleNamespace(stdout='', stderr='')
        raise AssertionError(command)

    def test_execute_submits_one_service_and_writes_manifest_once(self):
        plan = self.plan()
        with patch.object(p.subprocess, 'run', side_effect=self.system_reply) as run:
            result = p.submit(plan)
        self.assertEqual(result['status'], 'submitted_not_training_complete')
        calls = [c.args[0] for c in run.call_args_list]
        self.assertEqual([c[0] for c in calls], ['loginctl', 'systemctl', 'systemd-run'])
        self.assertEqual(self.state()['status'], 'submitted')
        command = calls[-1]
        self.assertEqual(command[command.index('--_manifest-sha') + 1], p.sha256(self.out / 'pipeline_manifest.json'))
        self.assertFalse((self.out / 'train').exists())
        self.assertEqual(p.read_json(self.out / 'submission.json')['status'], 'accepted')

    def test_submission_timeout_never_retries_or_overwrites_worker_status(self):
        plan = self.plan()
        def reply(command, **kwargs):
            if command[0] == 'systemd-run':
                self.put(self.out / 'pipeline_status.json', {'status': 'running', 'stage': 'feature_cache'})
                raise subprocess.TimeoutExpired(command, 30)
            return self.system_reply(command, **kwargs)
        with patch.object(p.subprocess, 'run', side_effect=reply) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                p.submit(plan)
        self.assertEqual(sum(c.args[0][0] == 'systemd-run' for c in run.call_args_list), 1)
        self.assertEqual(self.state()['status'], 'running')
        self.assertEqual(p.read_json(self.out / 'submission.json')['status'], 'submission_unconfirmed_no_retry')
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_no_linger_or_existing_unit_refused_without_changes(self):
        for blocked in ('loginctl', 'systemctl'):
            with self.subTest(blocked=blocked):
                plan = self.plan()
                def reply(command, **kwargs):
                    if command[0] == blocked:
                        return SimpleNamespace(stdout='no\n' if blocked == 'loginctl' else 'loaded\n')
                    return self.system_reply(command, **kwargs)
                with patch.object(p.subprocess, 'run', side_effect=reply) as run:
                    with self.assertRaises(RuntimeError):
                        p.submit(plan)
                self.assertFalse(self.out.exists())
                self.assertNotIn('systemd-run', [c.args[0][0] for c in run.call_args_list])

    def test_internal_worker_cannot_bypass_execute_or_manifest_binding(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p.main(['--_worker-manifest', 'x'])
        manifest = self.root / 'pipeline_manifest.json'
        self.put(manifest, {})
        with patch.dict(os.environ, {'INVOCATION_ID': 'CPU test, not real systemd'}):
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                p.main(['--execute', '--_worker-manifest', str(manifest), '--_manifest-sha', '0' * 64])

    def test_actual_posttrain_gate_accepts_only_clean_retained_success_and_no_lock(self):
        # Extract the real gate AST, avoiding the evaluation module's robot deps.
        file = p.ROOT / 'scripts/run_robotwin_posttrain_eval_pipeline.py'
        tree = ast.parse(file.read_text(), filename=str(file))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in ('TrainingFailed', 'training_gate')]
        state = {'LoadState': 'loaded', 'ActiveState': 'active', 'SubState': 'exited',
                 'Result': 'success', 'ExecMainCode': '1', 'ExecMainStatus': '0', 'MainPID': '0'}
        namespace = {'Path': Path, 'time': time, 'gpu_owner': lambda: None,
                     'queue': SimpleNamespace(read_result=p.read_json, unit_state=lambda unit: state)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(file), 'exec'), namespace)
        plan = self.prepare()
        self.assertEqual(self.run_worker(plan), 0)
        gate = namespace['training_gate']
        args = (plan['posttrain']['training_run_dir'], plan['posttrain']['training_unit'])
        self.assertTrue(gate(*args)[0])
        self.assertFalse(gate(*args, owner_reader=lambda: 'other-session')[0])
        state['MainPID'] = '1234'
        self.assertFalse(gate(*args)[0])
        state.update(MainPID='0', Result='exit-code', ActiveState='failed')
        with self.assertRaises(namespace['TrainingFailed']):
            gate(*args)


if __name__ == '__main__':
    unittest.main()
