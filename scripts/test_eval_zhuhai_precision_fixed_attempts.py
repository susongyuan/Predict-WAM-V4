#!/usr/bin/env python3
"""CPU-only fixtures. Never launch the frozen evaluator or mutate original evidence."""
import copy
import contextlib
import io
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts import eval_zhuhai_precision_fixed_attempts as w


CHECKPOINT_BYTES = b'CPU fixture bytes, never torch loaded'
SHA = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def npy(shape, descr):
    header = repr(dict(descr=descr, fortran_order=False, shape=shape)).encode()
    header += b' ' * ((-len(header) - 11) % 64) + b'\n'
    size = 1
    for n in shape:
        size *= n
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + bytes(size * (4 if descr == '<f4' else 1))


def fixture(directory, seed=932101, status='control_safety_abort', step=25000):
    directory.mkdir(parents=True, exist_ok=True)
    robot = {'profile': 'CPU fixture, not a real admission'}
    dc = dict(record_video=False, third_view=False, camera_count=3)
    post = {'fixture': 'unchanged simulated binding'}
    runtime = dict(format=w.FORMAT, task=w.TASK, checkpoint_step=step, checkpoint_sha256=SHA,
                   strict_load=True, projections_reset=False, vlm=False, high_level_planner=False,
                   ask=False, memory_writes=False, future_inputs_used_online=False)
    for k, v in [('robot_contract', robot), ('data_contract', dc), ('action_postprocessing', post)]:
        runtime[k] = v
        runtime[k + '_sha256'] = w.digest(v)
    gates = {k: dict(passed=True, events=[], robot_contract=robot, robot_contract_sha256=w.digest(robot)) for k in ('mechanical', 'camera')}
    abort = status == 'control_safety_abort'
    rule0 = dict(physics_time_seconds=0.0, score=dict(base_score=0, full_task_success=False, rule_source='CPU fixture'),
                 objects={'fixture': {}}, physics_safety=dict(passed=True, events=[], checks=['fixture']), video_recording=False)
    rule1 = copy.deepcopy(rule0)
    rule1['physics_time_seconds'] = .1
    if not abort:
        rule1['score'].update(base_score=30, full_task_success=True)
    name = f'{w.TASK}_seed_{seed}'
    raw = dict(task=w.TASK, seed=seed, status='infrastructure_error' if abort else 'completed', control_steps=1,
               full_task_success=None if abort else True, base_score=None if abort else 30,
               max_control_steps=2400, maximum_physics_seconds=240., record_video=False,
               policy_ready=dict(kind='ready', version=1, episode='startup', runtime=runtime), admission_gates=gates,
               last_rule_evidence=rule1)
    for k in ('robot_contract', 'data_contract', 'robot_contract_sha256', 'data_contract_sha256', 'action_postprocessing_sha256'):
        raw[k] = runtime[k]
    if abort:
        reply = dict(version=1, kind='infrastructure_error', episode='startup', step=0, error='ValueError: ' + w.SIGNATURE)
        raw['error'] = f'InfrastructureError: wrong worker reply/ID, expected action/{name}/1: {reply!r}'
    else:
        raw.update(final_rule_evidence=rule1, termination='full_rule_success')
    dump(directory / 'result.json', raw)
    (directory / 'policy_worker.log').write_text('Traceback (most recent call last):\n  File "/fixture/harness/zhuhai_action_smoothing.py", line 132, in step\nValueError: ' + w.SIGNATURE + '\n')
    action = dict(control_step=0, raw_action=[0.] * 16, smoothed_action=[0.] * 16, limited_action=[0.] * 16,
                  dispatched_action=[0.] * 16, clip_mask=[False] * 16, action_postprocessing_sha256=w.digest(post),
                  physics_time_before=0., policy=dict(kind='action', episode=name, step=0, limited_action=[0.] * 16))
    (directory / 'actions.jsonl').write_text(json.dumps(action) + '\n')
    (directory / 'rule_evidence.jsonl').write_text('\n'.join(json.dumps(dict(control_step=i, **r)) for i, r in enumerate((rule0, rule1))) + '\n')
    manifest = dict(schema=w.FORMAT + '/onboard', frames=2, seed=seed, episode=name, lossless=True,
                    fields=list(w.FIELDS), record_video=False, third_view=False, complete=not abort)
    for k in ('robot_contract', 'robot_contract_sha256', 'data_contract', 'data_contract_sha256'):
        manifest[k] = runtime[k]
    dump(directory / 'onboard/manifest.json', manifest)
    frames = []
    for i in range(2):
        filename = f'observation_{i:06d}.npz'
        frames.append(dict(index=i, control_step=i, file=filename, physics_time_seconds=i * .1))
        with zipfile.ZipFile(directory / 'onboard' / filename, 'w') as z:
            for field in w.FIELDS:
                z.writestr(field + '.npy', npy((1, 1, 3) if field.endswith('_camera') else (16,), '|u1' if field.endswith('_camera') else '<f4'))
    (directory / 'onboard/frames.jsonl').write_text('\n'.join(json.dumps(f) for f in frames) + '\n')
    return directory / 'result.json'


class FixedAttemptsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='precision-fixed-cpu-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.no_process = mock.patch.object(w.subprocess, 'Popen', side_effect=AssertionError('CPU tests cannot launch processes'))
        self.no_process.start()
        self.addCleanup(self.no_process.stop)

    def classify(self, p, seed=932101, step=25000):
        return w.classify_result(p, step, seed, SHA)

    def test_abort_preserves_raw_null_score_and_hash_read_only(self):
        p = fixture(self.root / 'case')
        before = {str(f): (w.sha256_file(f), f.stat().st_mtime_ns) for f in p.parent.rglob('*') if f.is_file()}
        row = self.classify(p)
        self.assertEqual(row['status'], 'control_safety_abort')
        self.assertIsNone(row['base_score'])
        self.assertEqual(row['observed_base_score'], 0)
        self.assertFalse(row['full_task_success'])
        self.assertEqual(row['raw_status'], 'infrastructure_error')
        self.assertEqual(before, {str(f): (w.sha256_file(f), f.stat().st_mtime_ns) for f in p.parent.rglob('*') if f.is_file()})

    def test_completed_uses_original_final_rule_and_rejects_mismatch(self):
        p = fixture(self.root / 'case', status='completed')
        self.assertTrue(self.classify(p)['full_task_success'])
        raw = w.read_json(p)
        raw['base_score'] = 29
        dump(p, raw)
        with self.assertRaisesRegex(ValueError, 'score differs'):
            self.classify(p)

    def test_zero_step_eula_and_other_errors_never_become_failure(self):
        for i, (status, steps, error) in enumerate([
            ('infrastructure_error', 0, 'SystemExit: EULA not accepted'),
            ('infrastructure_error', 1, 'ValueError: nonfinite policy action'),
            ('infrastructure_error', 1, 'RuntimeError: CUDA out of memory'),
            ('wall_timeout', 1, 'TimeoutError: fixed wall-clock budget'),
            ('running', 1, None),
        ]):
            with self.subTest(error=error):
                p = fixture(self.root / str(i))
                raw = w.read_json(p)
                raw.update(status=status, control_steps=steps, error=error)
                dump(p, raw)
                with self.assertRaises(ValueError):
                    self.classify(p)

    def test_wrong_seed_sha_step_or_gates_rejected(self):
        for i, field in enumerate(('seed', 'sha', 'step', 'gate', 'video', 'success')):
            with self.subTest(field=field):
                p = fixture(self.root / str(i))
                raw = w.read_json(p)
                if field == 'seed': raw['seed'] += 1
                if field == 'sha': raw['policy_ready']['runtime']['checkpoint_sha256'] = 'b' * 64
                if field == 'step': raw['policy_ready']['runtime']['checkpoint_step'] = 20000
                if field == 'gate': raw['admission_gates']['mechanical']['passed'] = False
                if field == 'video': raw['record_video'] = True
                if field == 'success': raw['last_rule_evidence']['score']['full_task_success'] = True
                dump(p, raw)
                with self.assertRaises(ValueError): self.classify(p)

    def test_missing_traceback_or_archive_count_rejected(self):
        p = fixture(self.root / 'case')
        (p.parent / 'policy_worker.log').write_text('Traceback (most recent call last):\n')
        with self.assertRaisesRegex(ValueError, 'traceback'): self.classify(p)
        p = fixture(self.root / 'case2')
        (p.parent / 'actions.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'count mismatch'): self.classify(p)
        p = fixture(self.root / 'case3')
        (p.parent / 'onboard/observation_000001.npz').unlink()
        with self.assertRaises(ValueError): self.classify(p)

    def cohort(self, mixed=False):
        rows = []
        for i, seed in enumerate(w.SEEDS):
            p = fixture(self.root / str(seed), seed, 'completed' if mixed and i < 2 else 'control_safety_abort')
            rows.append(self.classify(p, seed))
        cp = self.root / 'train/checkpoint_step_025000.pt'
        cp.parent.mkdir()
        cp.write_bytes(CHECKPOINT_BYTES)
        dump(cp.with_name(cp.name + '.sha256.json'), dict(step=25000, sha256=SHA, size=cp.stat().st_size, model_only=True))
        runtime = w.read_json(rows[0]['raw_result_path'])['policy_ready']['runtime']
        contract = {k: runtime[k] for k in ('format', 'robot_contract', 'robot_contract_sha256', 'data_contract', 'data_contract_sha256', 'action_postprocessing', 'action_postprocessing_sha256')}
        contract['source_files'] = {str(w.EVAL): w.sha256_file(w.EVAL)}
        dump(cp.parent / 'manifest.json', dict(contract=contract, contract_sha256=w.digest(contract)))
        metadata, _ = w.checkpoint_metadata(cp)
        return w.build_summary(metadata, rows, 'completed')

    def test_all_safety_aborts_complete_zero_of_five_and_verify_no_process(self):
        value = self.cohort()
        p = self.root / 'summary.json'
        dump(p, value)
        self.assertEqual(w.verify_summary(p, 25000, w.SEEDS, SHA), value)
        self.assertEqual((value['normal_completed'], value['safety_aborted'], value['success_rate']), (0, 5, 0.0))

    def test_mixed_cohort_and_tampered_summary_rejected(self):
        value = self.cohort(mixed=True)
        p = self.root / 'summary.json'
        dump(p, value)
        self.assertEqual(w.verify_summary(p, 25000, w.SEEDS)['success_rate'], .4)
        for field in ('success_rate', 'checkpoint_sha256', 'duplicate', 'raw_status'):
            bad = copy.deepcopy(value)
            if field == 'success_rate': bad[field] = 1.
            if field == 'checkpoint_sha256': bad[field] = 'b' * 64
            if field == 'duplicate': bad['episodes'][1] = bad['episodes'][0]
            if field == 'raw_status': bad['episodes'][0]['raw_status'] = 'control_safety_abort'
            dump(p, bad)
            with self.subTest(field=field), self.assertRaises(ValueError): w.verify_summary(p, 25000, w.SEEDS, SHA)

    def test_incomplete_cohort_has_no_rate(self):
        value = self.cohort()
        metadata = value['checkpoint']
        with self.assertRaises(ValueError): w.build_summary(metadata, value['episodes'][:4], 'completed')
        partial = w.build_summary(metadata, value['episodes'][:4], 'halted_infrastructure_error')
        self.assertIsNone(partial['success_rate'])
        p = self.root / 'summary.json'
        dump(p, partial)
        with self.assertRaises(ValueError): w.verify_summary(p, 25000, w.SEEDS)

    @contextlib.contextmanager
    def plan_environment(self, step=25000):
        fake_eval = self.root / 'old_eval.py'
        fake_eval.write_text('# frozen fixture\n')
        cp = self.root / 'train' / f'checkpoint_step_{step:06d}.pt'
        cp.parent.mkdir(exist_ok=True)
        cp.write_bytes(b'CPU fixture bytes, never torch loaded')
        dump(cp.with_name(cp.name + '.sha256.json'), dict(step=step, sha256=w.sha256_file(cp), size=cp.stat().st_size, model_only=True))
        contract = dict(format=w.FORMAT, source_files={str(fake_eval): w.sha256_file(fake_eval)})
        dump(cp.parent / 'manifest.json', dict(contract=contract, contract_sha256=w.digest(contract)))
        protocol = dict(schema='precision-v2-fixed-attempt-evaluation-protocol-v1', checkpoints=list(w.STEPS), seeds=list(w.SEEDS),
                        planned_attempts_each=5, safety_abort_signature=w.SIGNATURE, scoring_source_unchanged=True,
                        postprocessing_source_unchanged=True, do_not_rerun_failed_seed=True, raw_result_files_must_not_be_modified=True,
                        record_video=False, third_view=False, original_failed_physical_attempt={'result_sha256': SHA})
        protocol_path = self.root / 'protocol.json'
        dump(protocol_path, protocol)
        lib = self.root / '.conda-envs/behavior6/lib/libstdc++.so.6.0.36'
        lib.parent.mkdir(parents=True, exist_ok=True)
        lib.write_bytes(b'fixture')
        env_path = self.root / 'launch.json'
        dump(env_path, {'plan': {'child_env': dict(OMNI_KIT_ACCEPT_EULA='YES', LD_PRELOAD=str(lib), OMP_NUM_THREADS='2',
                 MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')}})
        with mock.patch.multiple(w, ROOT=self.root, RUN=self.root / 'run', EVAL=fake_eval, PROTOCOL=protocol_path, ENV_LAUNCH=env_path):
            yield cp

    def test_default_plan_no_process_no_out_and_fixed_budgets(self):
        with self.plan_environment() as cp:
            out = self.root / 'newout'
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(w.main(['--checkpoint', str(cp), '--out', str(out)]), 0)
            plan = json.loads(stream.getvalue())
            self.assertFalse(out.exists())
            self.assertEqual(len(plan['commands']), 5)
            self.assertEqual(plan['child_env']['OMNI_KIT_ACCEPT_EULA'], 'YES')
            for item in plan['commands']:
                argv = item['command']
                self.assertIn('--episode-seed', argv)
                self.assertEqual(argv[argv.index('--max-steps') + 1], '2400')
                self.assertEqual(argv[argv.index('--wall-seconds') + 1], '3600')
                self.assertEqual(argv[argv.index('--startup-timeout') + 1], '180')
                self.assertEqual(argv[argv.index('--policy-timeout') + 1], '60')

    def test_adopt_both_exact_original_paths_and_duplicate_rejected(self):
        with self.plan_environment(20000) as cp:
            argv = ['--checkpoint', str(cp), '--out', str(self.root / 'out')]
            adopted = list(w.known_adoptions().values())
            for path in adopted: argv += ['--adopt-result', str(path)]
            plan = w.build_plan(w.parser().parse_args(argv))
            self.assertEqual([r['seed'] for r in plan['commands']], [932103, 932104, 932105])
            self.assertTrue(plan['adopt_existing'])
            self.assertIn('development_20000_fixed_attempts_seed932102_r2/', str(adopted[1]))
            legacy_argv = [arg.replace('development_20000_fixed_attempts_seed932102_r2/', 'development_20000_fixed_attempts_seed932102/') for arg in argv]
            with self.assertRaises(ValueError): w.build_plan(w.parser().parse_args(legacy_argv))
            with self.assertRaisesRegex(ValueError, 'duplicate'): w.build_plan(w.parser().parse_args(argv + ['--adopt-result', str(adopted[0])]))
            with self.assertRaises(ValueError): w.build_plan(w.parser().parse_args(argv[:-2]))

    def test_metadata_fresh_out_and_frozen_source_protection(self):
        with self.plan_environment() as cp:
            args = w.parser().parse_args(['--checkpoint', str(cp), '--out', str(self.root / 'out')])
            w.build_plan(args)
            Path(args.out).mkdir()
            with self.assertRaisesRegex(ValueError, 'fresh'): w.build_plan(args)
            Path(args.out).rmdir()
            cp.write_bytes(b'changed')
            with self.assertRaises(ValueError): w.build_plan(args)
        with self.plan_environment() as cp:
            w.EVAL.write_text('changed frozen source')
            with self.assertRaisesRegex(ValueError, 'source changed'): w.checkpoint_metadata(cp)

    def test_serial_executor_all_abort_once_and_infra_halts_without_retry(self):
        value = self.cohort()
        metadata = value['checkpoint']
        for mode in ('abort', 'timeout'):
            out = self.root / ('execute_' + mode) / 'out'
            plan = dict(schema='precision-v2-fixed-attempt-plan-v1', out=str(out), checkpoint=metadata,
                        gpu_lock_owner='CPU-fixture-only', child_env={}, adopt_results=[],
                        protocol_sha256=w.sha256_file(w.PROTOCOL), env_launch_sha256=w.sha256_file(w.ENV_LAUNCH),
                        wrapper_sha256=w.sha256_file(w.__file__), frozen_sources={str(w.EVAL): w.sha256_file(w.EVAL)},
                        commands=[dict(seed=seed, out=str(out / w.TASK / f'seed_{seed}'), command=['NEVER_EXECUTE']) for seed in w.SEEDS])
            calls = []
            def fake_runner(item, env):
                calls.append(item['seed'])
                if mode == 'timeout': raise TimeoutError('CPU simulated infrastructure timeout')
                fixture(Path(item['out']), item['seed'])
                return 2
            with mock.patch.object(w, 'run_episode', side_effect=fake_runner), mock.patch.object(w, 'verify_external_owner'), contextlib.redirect_stderr(io.StringIO()):
                code = w.execute(plan)
            summary = w.read_json(out / 'summary.json')
            if mode == 'abort':
                self.assertEqual(code, 0)
                self.assertEqual(calls, list(w.SEEDS))
                self.assertEqual(summary['status'], 'completed')
                self.assertEqual(summary['success_rate'], 0.)
            else:
                self.assertEqual(code, 2)
                self.assertEqual(calls, [w.SEEDS[0]])
                self.assertEqual(summary['status'], 'halted_infrastructure_error')
                self.assertIsNone(summary['success_rate'])
            with self.assertRaises(FileExistsError): w.execute(plan)

    def test_read_only_real_abort_and_original_eula(self):
        # Explicitly authorized deterministic original-limiter replay substitutes
        # for a truncated traceback; neither logs nor unseen actions are invented.
        p = w.known_adoptions()[932101]
        if not p.is_file(): self.skipTest('original evidence not mounted')
        raw = w.read_json(p)
        before = (w.sha256_file(p), p.stat().st_mtime_ns)
        log = p.parent / 'policy_worker.log'
        log_before = (w.sha256_file(log), log.stat().st_mtime_ns)
        sha = raw['policy_ready']['runtime']['checkpoint_sha256']
        first = w.classify_result(p, 20000, 932101, sha)
        self.assertEqual(first['status'], 'control_safety_abort')
        self.assertIsNone(first['base_score'])
        self.assertFalse(first['full_task_success'])
        proof = first['safety_abort_evidence']
        self.assertEqual(proof['replayed_control_steps'], 1524)
        self.assertTrue(proof['all_limited_actions_exactly_equal'])
        self.assertFalse(proof['unknown_next_action_reconstructed'])
        self.assertGreater(proof['infeasible_intervals'][0]['feasible_v_lower'], proof['infeasible_intervals'][0]['feasible_v_upper'])
        # One bounded negative replay on a COPY, no original evidence is edited.
        import shutil
        tampered = self.root / 'tampered_replay'
        (tampered / 'onboard').mkdir(parents=True)
        shutil.copyfile(p.parent / 'onboard/observation_000000.npz', tampered / 'onboard/observation_000000.npz')
        action_lines = (p.parent / 'actions.jsonl').read_text().splitlines()
        bad = json.loads(action_lines[0])
        bad['limited_action'][0] += .01
        action_lines[0] = json.dumps(bad)
        (tampered / 'actions.jsonl').write_text('\n'.join(action_lines) + '\n')
        with self.assertRaisesRegex(ValueError, 'trajectory mismatch'):
            w.replay_limiter_evidence(tampered, raw, raw['policy_ready']['runtime'])
        self.assertEqual(before, (w.sha256_file(p), p.stat().st_mtime_ns))
        self.assertEqual(log_before, (w.sha256_file(log), log.stat().st_mtime_ns))
        eula = w.RUN / 'development_20000/zhuhai_task2/seed_932101/result.json'
        with self.assertRaisesRegex(ValueError, 'zero-control'): w.classify_result(eula, 20000, 932101, sha)
        second = w.known_adoptions()[932102]
        before_second = (w.sha256_file(second), second.stat().st_mtime_ns)
        row = w.classify_result(second, 20000, 932102, sha)
        self.assertEqual((row['status'], row['control_steps'], row['base_score'], row['full_task_success']), ('completed', 2400, 0, False))
        self.assertEqual(before_second, (w.sha256_file(second), second.stat().st_mtime_ns))
        legacy = w.RUN / 'development_20000_fixed_attempts_seed932102/zhuhai_task2/seed_932102/result.json'
        with self.assertRaisesRegex(ValueError, 'zero-control'): w.classify_result(legacy, 20000, 932102, sha)


if __name__ == '__main__':
    unittest.main(verbosity=2)
