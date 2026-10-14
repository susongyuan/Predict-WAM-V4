#!/usr/bin/env python3
"""Persistent, immutable staged coordinator; may run inside systemd-run.

The outer scripts/gpu_lock.py run or human coordinator owns the lock. Each
stage writes intent/status/logs before acting. Collector failure halts expansion
and never turns an incomplete dataset into a training run. Resuming unchanged
data stages is explicit; scene/config changes require a different run directory.
Incomplete core training/evaluation outputs are protected until the main thread
provides transactional resume; this coordinator never overwrites their evidence.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.progress1_log_200 import log
from scripts.collect_isaac_zhuhai_200 import (PRODUCTION_QUOTAS, TOTAL_PER_TASK, TOTAL_ALL_TASKS,
    MINIMUM_FREE_BYTES, ONBOARD_CAMERAS, robot_binding, validate_manifest, write_once,
    sha256, check_disk, runtime_interface, data_contract, data_contract_hash, robot_spec, json_digest,
    STATE_SEMANTICS, ACTION_SEMANTICS)

GIB = 1024**3
SECOND_START_DELAY_SECONDS = 15.
COLLECTION_POLL_SECONDS = .2
SNAPSHOT_FILE_LIMIT = 8 * 1024**2
SNAPSHOT_TOTAL_LIMIT = 64 * 1024**2
SNAPSHOT_SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.sh', '.md', '.txt'}


def sha(path):
    return sha256(path)


def atomic(path, value):
    """Mutable coordinator index; one writer, unique temporary publication."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix='.' + path.name + '-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_owner(expected):
    from scripts.gpu_lock import owner
    if not expected or owner() != expected:
        raise RuntimeError('coordinator must hold the matching external GPU lock')


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--source-root', required=True, type=Path, help='Original frozen run; import completed 100-source manifests without relabelling')
    ap.add_argument('--life-mapping', choices=['body_text', 'scoring_table'], required=True)
    ap.add_argument('--collection-workers', type=int, choices=(1, 2), default=1,
                    help='Only task1/task2 collectors may overlap; low/unknown RAM or GPU headroom falls back to serial')
    ap.add_argument('--source-count', type=int, default=PRODUCTION_QUOTAS['source'])
    ap.add_argument('--randomized-count', type=int, default=PRODUCTION_QUOTAS['randomized'])
    ap.add_argument('--train-steps', type=int, default=5000)
    ap.add_argument('--train-batch-size', type=int, default=4)
    ap.add_argument('--evaluation-count', type=int, default=20)
    ap.add_argument('--evaluation-max-steps', type=int, default=12000)
    ap.add_argument('--cache-root', default='data/zhuhai_native16_hybrid_features_20261009')
    ap.add_argument('--gpu-lock-owner')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--resume', action='store_true')
    return ap


def build_plan(args):
    scenes = [ROOT / 'harness' / f'isaac_zhuhai_task{i}.py' for i in (1, 2)]
    files = [Path(__file__).resolve(), ROOT / 'scripts/collect_isaac_zhuhai.py',
        ROOT / 'scripts/progress1_log.py', ROOT/'scripts/progress1_log_200.py',
        ROOT/'scripts/collect_isaac_zhuhai_200.py', ROOT/'scripts/handoff_zhuhai_200.py',
        ROOT / 'harness/isaac_zhuhai_runtime.py', ROOT / 'scripts/collect_zhuhai_data.py',
        ROOT / 'harness/zhuhai_robot_spec.py', ROOT / 'harness/robotwin_dataset.py',
        ROOT / 'harness/isaac_zhuhai_expert.py', ROOT / 'harness/isaac_zhuhai_collision.py', ROOT / 'scripts/zhuhai_resource_watch.py',
        ROOT / 'scripts/train_zhuhai_wam.py', ROOT / 'scripts/eval_zhuhai_wam.py',
        ROOT / 'harness/zhuhai_training.py', ROOT / 'harness/zhuhai_dataset.py',
        ROOT / 'harness/zhuhai_hybrid_cache.py', ROOT / 'harness/zhuhai_demo_recording.py',
        ROOT / 'harness/zhuhai_task1_rules.py', ROOT / 'harness/zhuhai_task2_rules.py',
        ROOT / 'harness/isaac_zhuhai_camera.py', ROOT / 'harness/zhuhai_action_smoothing.py',
        ROOT / 'third_party/LiLa-WAM/robotwin_infer.py', robot_spec.ASSET_MANIFEST,
        robot_spec.KINEMATIC_CHAIN, *scenes]
    files.extend([ROOT/'harness/wam_v3.py',ROOT/'harness/wam_behavior.py',
        ROOT/'harness/robotwin_multitask_training.py',ROOT/'harness/robotwin_vtt_adapter.py'])
    files.extend(sorted((ROOT/'third_party/LiLa-WAM').rglob('*.py')))
    files=list(dict.fromkeys(files))
    # Keep the persistent coordinator stdlib-only. Numerical imports occur in
    # the already pinned WAM environment without constructing models or CUDA.
    smoothing = json.loads(subprocess.check_output([
        str(ROOT / '.venvs/wam-fixed/bin/python'), '-B', '-c',
        'import json; from harness.zhuhai_action_smoothing import protocol; print(json.dumps(protocol()))'],
        cwd=ROOT, text=True))
    smoothing_hash = hashlib.sha256(json.dumps(smoothing, sort_keys=True, allow_nan=False).encode()).hexdigest()
    binding, interface = robot_binding(), runtime_interface()
    ready = all(path.is_file() for path in files) and binding['assets_ready'] and interface['ready']
    return dict(schema='zhuhai-isaac-native16-200dr-pipeline-v1', engine=dict(name='Isaac Sim', version='6.1.0.0'),
        execution_status='Isaac-only adapter; SAPIEN execution disabled', source_count=args.source_count,
        collection_workers=args.collection_workers,
        collection_scheduling=dict(max_workers=args.collection_workers, source_barrier_before_randomization=True,
            same_external_gpu_lock_owner=True, training_and_evaluation_workers=1,
            second_start_after='new first-worker initial_rule_evidence OR 15s after its spawn',
            second_start_delay_seconds=SECOND_START_DELAY_SECONDS,
            minimum_available_ram_bytes=12*GIB, minimum_free_gpu_bytes=6*GIB,
            low_or_unknown_resources='record serial fallback before any second spawn',
            failure='halt later stages; SIGTERM only live child process groups spawned by this group; wait/reap without SIGKILL'),
        source_snapshot_policy=dict(directory='source_snapshot', receipt='source_snapshot/receipt.json',
            source='plan.source_hashes code and small configurations', max_file_bytes=SNAPSHOT_FILE_LIMIT,
            max_total_bytes=SNAPSHOT_TOTAL_LIMIT, weights_and_HDF_excluded=True, overwrite=False),
        randomized_count=args.randomized_count, total_per_task=TOTAL_PER_TASK, total_all_tasks=TOTAL_ALL_TASKS,
        quota_interpretation='100 original successful source + 200 additional successful randomized physical episodes per task',
        source_import=dict(root=str(args.source_root.resolve()), original_plan_sha256=sha(args.source_root/'plan.json'), source_evidence_modified=False),
        family_split=dict(source_families=dict(train=80, val=10, heldout=10),
            successful_randomized_per_family=2, episodes_per_task=dict(train=240, val=30, heldout=30),
            VTT_source='TRAIN only'), robot_binding=binding, runtime_interface=interface,
        robot_contract=binding['robot_contract'], robot_contract_sha256=binding['robot_contract_sha256'],
        data_contract=data_contract(), data_contract_sha256=data_contract_hash(),
        state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS,
        train_steps=args.train_steps, train_batch_size=args.train_batch_size,
        initialization=dict(path=str(ROOT / 'outputs/runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/train/checkpoint_step_050000.pt'),
            sha256='8960be6ceb4d5293d2c5455ac4308421f64f1e31bf64f0646b7208735f63a5f4',
            strategy='reuse compatible frozen native14 backbone/VTT; native16 projections and optimizer initialized by training core',
            source_dof=14, target_dof=16, strict_all_layer_load=False,
            projections_reset=['action_proj', 'proprio_proj', 'output_proj'], bias_reset=True, residual_VTT_loaded=True,
            numerical_normalization='new successful TRAIN-only native16 measured-state/target-action statistics; old normalization forbidden',
            old_checkpoint_and_scores_modified=False,
            LiLa_initialization_comparison='future ablation, not an additional current run'),
        life_mapping=args.life_mapping, evaluation_count=args.evaluation_count,
        evaluation_max_steps=args.evaluation_max_steps,
        evaluation_seeds=list(range(932001, 932001 + args.evaluation_count)),
        cache=dict(root=str((ROOT / args.cache_root).resolve()), disk_budget_bytes=10**12,
                   ram_budget_bytes=2 * 1024**3, minimum_free_bytes=100 * 1024**3,
                   raw_source_evicted=False),
        recording=dict(video=False, third_view=False, camera_count=3,
            camera_names=list(ONBOARD_CAMERAS), onboard_sensor_archive=True),
        time=dict(frequency_hz=10, action_interval_seconds=0.1, chunk_seconds=3.2, execute_prefix_seconds=0.8),
        action_postprocessing=smoothing, action_postprocessing_sha256=smoothing_hash,
        stages=['source_task1', 'source_task2', 'randomized_task1', 'randomized_task2',
                'tiny_training', 'training', 'fresh_seed_physical_evaluation'],
        failure_policy='3 consecutive collector failures halt; no skipped quotas or silent seed replacement',
        resume_policy='completed stages are hash-revalidated; incomplete core train/eval output is protected pending transactional resume',
        scene_ready=ready, readiness_scope='static native16 interface/assets only; not physical expert/mechanical acceptance', source_hashes={str(path): sha(path) for path in files if path.is_file()})


def collection_receipt(path, task, mode):
    rows = validate_manifest(path, task=task, mode=mode)
    return dict(manifest=str(path), manifest_sha256=sha(path), successful_episodes=len(rows),
                source_families=len({r['family_id'] for r in rows}), state_dim=16, action_dim=16,
                state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS, data_contract_sha256=data_contract_hash())


def checkpoint_receipt(directory, step):
    contract = json.loads((directory / 'manifest.json').read_text())['contract']
    common = contract.get('config', {}).get('common', {})
    if (contract.get('format') != 'zhuhai-isaac-native16-measured-10hz-v1' or
            common.get('state_dim') != 16 or common.get('action_dim') != 16 or
            contract.get('data_contract') != data_contract() or
            contract.get('robot_contract') != robot_spec.robot_contract()):
        raise ValueError('Completed training stage must bind native16 measured-state data, never legacy14')
    path = directory / f'checkpoint_step_{step:06d}.pt'
    receipt = json.loads(path.with_name(path.name + '.sha256.json').read_text())
    if receipt.get('step') != step or receipt.get('sha256') != sha(path) or receipt.get('size') != path.stat().st_size:
        raise ValueError('Stage checkpoint/receipt missing, changed or at the wrong step')
    return dict(path=str(path), state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS,
                data_contract_sha256=data_contract_hash(), robot_contract_sha256=json_digest(contract['robot_contract']), **receipt)


def evaluation_receipt(out, plan):
    path = out / 'summary.json'
    summary = json.loads(path.read_text())
    rows = summary.get('episodes', [])
    expected = {(task, seed) for task in ('zhuhai_task1', 'zhuhai_task2') for seed in plan['evaluation_seeds']}
    actual = [(row.get('task'), row.get('seed')) for row in rows]
    if summary.get('status') != 'completed' or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Evaluation must retain every fixed task/seed exactly once, including policy failures')
    evidence = {}
    for row in rows:
        if row.get('status') != 'completed' or type(row.get('full_task_success')) is not bool:
            raise ValueError('Infrastructure/partial evaluation is not a completed physical trial')
        result = out / row['task'] / f"seed_{row['seed']}" / 'result.json'
        if json.loads(result.read_text()) != row:
            raise ValueError('Evaluation summary differs from original seed evidence')
        evidence[str(result)] = sha(result)
    return dict(summary_sha256=sha(path), completed_episodes=len(rows),
                full_successes=sum(r['full_task_success'] for r in rows), evidence_sha256=evidence)


def stage_needs_run(out, name, command, args, state, validate=None):
    existing = state['stages'].get(name)
    if existing and [s for s in existing['command'] if s != '--resume'] != [s for s in command if s != '--resume']:
        raise ValueError('Frozen stage command changed; use a new run directory')
    if existing and existing['status'] == 'completed':
        receipt = validate() if validate else None
        if receipt != existing.get('validation'):
            raise ValueError(f'{name}: completed-stage evidence changed; never silently reuse a status flag')
        return False
    if existing and not args.resume:
        raise RuntimeError('existing incomplete stage requires explicit --resume after diagnosis')
    if existing and existing.get('pid') and (existing['status'] == 'running' or existing.get('process_exited') is False):
        try:
            os.kill(existing['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Previous stage PID is still alive; do not start a second writer')
    if existing and name in ('tiny_training', 'training', 'fresh_seed_physical_evaluation'):
        target = out / ('evaluation' if name == 'fresh_seed_physical_evaluation' else name)
        if target.exists() and any(target.iterdir()):
            raise RuntimeError('Incomplete core-stage output retained unchanged; main-thread transactional resume or a new output is required')
    return True


def run_stage(out, name, command, args, state, validate=None):
    if not stage_needs_run(out, name, command, args, state, validate):
        return
    existing = state['stages'].get(name)
    verify_owner(args.gpu_lock_owner)
    check_disk(out, 1024**2)
    log(f'阶段操作前：{name}；独立attempt目录保留旧日志与结果，失败立即停止后续阶段', actor='run_zhuhai_pipeline')
    number = existing.get('attempt_number', 0) + 1 if existing else 1
    attempt_dir = out / 'stage_attempts' / name / f'attempt_{number:04d}'
    attempt_dir.mkdir(parents=True, exist_ok=False)
    entry = dict(status='running', command=command, started_unix=time.time(),
                 previous_attempt=existing, attempt_number=number, evidence_directory=str(attempt_dir))
    state['status'], state['active_stage'] = 'running', name
    state['stages'][name] = entry
    write_once(attempt_dir / 'intent.json', entry)
    atomic(out / 'status.json', state)
    proc = None
    old_handlers = {}
    try:
        with (attempt_dir / 'process.log').open('x') as handle:
            proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT,
                                    start_new_session=True)
            entry['pid'] = proc.pid
            atomic(out / 'status.json', state)
            def stop(signum, frame):
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                raise KeyboardInterrupt(f'pipeline interrupted by signal {signum}')
            old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
            code = proc.wait()
        entry['exit_code'] = code
        if code:
            raise RuntimeError(f'{name} exited {code}; retained evidence, no automatic bypass')
        entry['validation'] = validate() if validate else None
        entry.update(status='completed', finished_unix=time.time())
        state['status'] = 'running'
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                except ProcessLookupError:
                    pass
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        entry.update(status='interrupted' if interrupted else 'failed',
                     error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        state['status'] = 'halted_interrupted' if interrupted else 'halted_stage_failure'
        raise
    finally:
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
        write_once(attempt_dir / 'result.json', entry)
        atomic(out / 'status.json', state)


def snapshot_sources(out, plan):
    """Freeze hash-verified code/small configs, never weights/HDF or old copies.

    The whole inventory is verified before publishing any file. An interrupted
    copy can be resumed only when each existing file is byte-identical. This is
    evidence for reproducing a run, not permission to resume changed live code.
    """
    destination = Path(out) / 'source_snapshot'
    if destination.is_symlink():
        raise ValueError('Snapshot destination symlink forbidden')
    records, skipped, payloads = {}, {}, {}
    total = 0
    for name, expected in sorted(plan['source_hashes'].items()):
        source = (ROOT / name).absolute()
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(ROOT.resolve())
        except ValueError:
            raise ValueError(f'Snapshot source outside workspace: {name}')
        if source != resolved or not source.is_file():
            raise ValueError(f'Snapshot source must be a regular non-symlink workspace file: {name}')
        key = relative.as_posix()
        if key == 'receipt.json':
            raise ValueError('Snapshot source conflicts with reserved receipt path')
        size = source.stat().st_size
        if source.suffix.lower() not in SNAPSHOT_SUFFIXES or size > SNAPSHOT_FILE_LIMIT:
            skipped[key] = dict(sha256=expected, size_bytes=size, reason='not code/small config; never copy large artifacts, weights or source HDF')
            continue
        data = source.read_bytes()
        if len(data) > SNAPSHOT_FILE_LIMIT or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Frozen source changed before snapshot: {source}')
        total += len(data)
        if total > SNAPSHOT_TOTAL_LIMIT:
            raise ValueError('Source snapshot exceeds small-code/config total limit; no large artifact copy')
        target = destination / relative
        for component in (destination, *target.parents):
            if component.is_symlink():
                raise ValueError(f'Snapshot destination symlink forbidden: {component}')
            if component == Path(out):
                break
        if target.is_symlink() or (target.exists() and (not target.is_file() or sha(target) != expected)):
            raise ValueError(f'Existing source snapshot differs; never overwrite: {target}')
        records[key] = dict(source=str(source), sha256=expected, size_bytes=len(data))
        payloads[target] = data
    receipt = dict(schema='zhuhai-source-snapshot-v1', plan_sha256=json_digest(plan),
                   files=records, skipped=skipped, copied_bytes=total, overwrite=False)
    receipt_path = destination / 'receipt.json'
    if receipt_path.is_symlink():
        raise ValueError('Snapshot receipt symlink forbidden')
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError('Immutable source snapshot receipt differs; new run required')
    check_disk(out, total + 1024**2)
    destination.mkdir(parents=True, exist_ok=True)
    for target, data in payloads.items():
        if target.exists():
            continue  # Already hash-verified; do not change bytes or mtime.
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.snapshot-', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            os.link(temporary, target)  # Atomic and exclusive: never replace a prior snapshot.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    write_once(receipt_path, receipt, identical_ok=True)
    return dict(path=str(receipt_path), sha256=sha(receipt_path), files=len(records), copied_bytes=total)


def collection_resource_admission():
    """Only /proc/meminfo + nvidia-smi inventory; no CUDA context or allocation."""
    from scripts.zhuhai_resource_watch import memory_info, gpu_info
    answer = dict(minimum_available_ram_bytes=12*GIB, minimum_free_gpu_bytes=6*GIB,
                  checked_unix=time.time(), read_only=True, cuda_context_created=False)
    try:
        memory, gpu = memory_info(), gpu_info()
        available = memory['available_bytes']
        devices = gpu.get('devices', []) if gpu.get('status') == 'read_only_inventory' else []
        # A single physical GPU is expected. Taking the minimum across every
        # reported device is conservative if visibility/selection is ambiguous.
        free = min((d['total_bytes'] - d['used_bytes'] for d in devices), default=None)
        answer.update(memory=memory, gpu=gpu, available_ram_bytes=available, gpu_free_bytes=free,
            admitted=available >= 12*GIB and free is not None and free >= 6*GIB)
        answer['reason'] = 'headroom_sufficient' if answer['admitted'] else 'low_or_unknown_RAM_GPU_headroom'
    except Exception as exc:
        answer.update(admitted=False, reason='resource_inventory_unavailable', error=f'{type(exc).__name__}: {exc}')
    return answer


def _command_value(command, flag):
    if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
        raise ValueError(f'Exactly one {flag} required for an isolated collector')
    return command[command.index(flag) + 1]


def _collection_jobs(out, mode, jobs, args):
    if mode not in PRODUCTION_QUOTAS or [job['name'] for job in jobs] != [f'{mode}_task1', f'{mode}_task2']:
        raise ValueError('Only the two independent task1/task2 collectors of the same mode may be scheduled')
    seeds = []
    for i, job in enumerate(jobs, 1):
        task = f'zhuhai_task{i}'
        expected = Path(out) / task / mode
        command = job['command']
        if (Path(_command_value(command, '--out')).resolve() != expected or expected.resolve() != expected
                or Path(job['manifest']) != expected / 'manifest.json'
                or _command_value(command, '--task') != task or _command_value(command, '--mode') != mode
                or _command_value(command, '--count') != str(PRODUCTION_QUOTAS[mode])
                or _command_value(command, '--gpu-lock-owner') != args.gpu_lock_owner):
            raise ValueError('Collector task/output/manifest/quota/owner isolation mismatch')
        if mode == 'randomized' and Path(_command_value(command, '--source-manifest')) != args.source_root.resolve()/task/'source/manifest.json':
            raise ValueError('DR source-family parent manifest must belong to the same task')
        seeds.append(_command_value(command, '--seed-start'))
    if len(set(seeds)) != 2:
        raise ValueError('Collector task seed streams must remain distinct')


def _new_initial_evidence(child):
    try:
        rows = json.loads(child['job']['manifest'].read_text()).get('attempts', [])
        return any(isinstance(row.get('initial_rule_evidence'), dict) for row in rows[child['initial_attempts']:])
    except (OSError, ValueError, TypeError):
        return False


def collection_barrier(state, mode):
    """Only per-child validated receipts satisfy the source/DR quota barrier."""
    for i in (1, 2):
        stage = state['stages'].get(f'{mode}_task{i}', {})
        validation = stage.get('validation') or {}
        if (stage.get('status') != 'completed' or validation.get('successful_episodes') != PRODUCTION_QUOTAS[mode]
                or validation.get('source_families') != PRODUCTION_QUOTAS['source']):
            raise RuntimeError(f'{mode} task{i} quota/validation barrier incomplete; no following stages')


def run_collection_group(out, mode, jobs, args, state, plan):
    """Main-thread scheduler: no run_stage threads, shared .tmp or nested locks."""
    _collection_jobs(out, mode, jobs, args)
    if args.collection_workers == 1:
        for job in jobs:
            verify_plan_sources(plan)
            run_stage(out, job['name'], job['command'], args, state, validate=job['validate'])
        return
    # Check every prior stage before launching anything, including live old PIDs.
    pending = [j for j in jobs if stage_needs_run(out, j['name'], j['command'], args, state, j['validate'])]
    if not pending:
        return
    previous = state.setdefault('collection_groups', {}).get(mode)
    number = previous.get('attempt_number', 0) + 1 if previous else 1
    group_dir = out/'collection_groups'/mode/f'attempt_{number:04d}'
    group_dir.mkdir(parents=True, exist_ok=False)
    group = dict(status='running', mode=mode, requested_workers=2, peak_active_workers=0,
                 started_unix=time.time(), attempt_number=number, previous_attempt=previous,
                 pending=[j['name'] for j in pending], decisions=[], external_gpu_lock_owner=args.gpu_lock_owner)
    state['collection_groups'][mode] = group
    state.update(status='running', active_stage=mode+'_collection', active_stages=[])
    write_once(group_dir/'intent.json', group)
    atomic(out/'status.json', state)
    active = {}
    allow_overlap = True
    handlers = {}
    stop_requested = {'signal': None}

    def persist(child=None):
        group['pending'] = [j['name'] for j in pending]
        state['active_stages'] = list(active)
        if child is not None:
            atomic(child['directory']/'status.json', child['entry'])
        atomic(out/'status.json', state)

    def start(job):
        if stop_requested['signal'] is not None:
            raise KeyboardInterrupt(f'collection group interrupted by signal {stop_requested["signal"]}')
        verify_plan_sources(plan); verify_owner(args.gpu_lock_owner); check_disk(out, 1024**2)
        existing = state['stages'].get(job['name'])
        attempt = existing.get('attempt_number', 0)+1 if existing else 1
        directory = out/'stage_attempts'/job['name']/f'attempt_{attempt:04d}'
        directory.mkdir(parents=True, exist_ok=False)
        entry = dict(status='running', command=job['command'], started_unix=time.time(), attempt_number=attempt,
            previous_attempt=existing, evidence_directory=str(directory), exit_code=None, validation=None,
            process_exited=False, group=mode, external_gpu_lock_owner=args.gpu_lock_owner)
        state['stages'][job['name']] = entry
        child = dict(job=job, entry=entry, directory=directory, proc=None, handle=None,
                     started_monotonic=time.monotonic(), initial_attempts=0)
        try:
            child['initial_attempts'] = len(json.loads(job['manifest'].read_text()).get('attempts', []))
        except (OSError, ValueError, TypeError):
            pass
        write_once(directory/'intent.json', entry); persist(child)
        log(f'采集子阶段启动前：{job["name"]} attempt={attempt}，独立目录，共享外部owner；不启动训练/评测', actor='run_zhuhai_pipeline')
        try:
            child['handle'] = (directory/'process.log').open('x')
            proc = subprocess.Popen(job['command'], stdout=child['handle'], stderr=subprocess.STDOUT,
                                    cwd=ROOT, start_new_session=True)
            child['proc'] = proc
            entry.update(pid=proc.pid, pgid=proc.pid)
            active[job['name']] = child
            group['peak_active_workers'] = max(group['peak_active_workers'], len(active))
            persist(child)
            if stop_requested['signal'] is not None:
                raise KeyboardInterrupt(f'collection group interrupted by signal {stop_requested["signal"]}')
        except BaseException as exc:
            # If spawn succeeded but status publication failed, keep the known
            # Popen in active so the outer cleanup still terminates/reaps it.
            if child['proc'] is None:
                entry.update(status='failed', error=f'{type(exc).__name__}: {exc}', process_exited=True, finished_unix=time.time())
                if child['handle'] is not None:
                    child['handle'].close()
                write_once(directory/'result.json', entry); persist(child)
            raise

    def finish(child, code, cancellation=None):
        entry = child['entry']
        entry.update(exit_code=code, process_exited=True, finished_unix=time.time())
        error = None
        if cancellation:
            entry.update(status='interrupted', cancellation_reason=cancellation)
        elif code != 0:
            entry.update(status='failed', error=f'collector exited {code}')
            error = RuntimeError(f'{child["job"]["name"]} exited {code}; no quota bypass')
        else:
            try:
                entry['validation'] = child['job']['validate']()
                entry['status'] = 'completed'
            except Exception as exc:
                entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                error = exc
        child['handle'].close()
        active.pop(child['job']['name'], None)
        write_once(child['directory']/'result.json', entry); persist(child)
        return error

    def decision(value):
        group['decisions'].append(value)
        write_once(group_dir/f'decision_{len(group["decisions"]):04d}.json', value)
        persist()

    def signal_stop(signum, frame):
        # Cleanup below is the only place sending a signal, using Popen objects
        # spawned here, NEVER PIDs loaded from a manifest or process-name scans.
        # Defer the exception until Popen ownership/status is registered. A
        # signal arriving during spawn must not orphan an untracked child.
        stop_requested['signal'] = signum

    try:
        handlers = {sig: signal.signal(sig, signal_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        while pending or active:
            if stop_requested['signal'] is not None:
                raise KeyboardInterrupt(f'collection group interrupted by signal {stop_requested["signal"]}')
            failure = None
            for child in list(active.values()):
                code = child['proc'].poll()
                if code is not None:
                    error = finish(child, code)
                    failure = failure or error
            if failure:
                raise failure
            if pending and not active:
                if group['peak_active_workers'] and not group['decisions']:
                    decision(dict(decision='serial_first_finished_before_overlap', checked_unix=time.time()))
                start(pending.pop(0))
            elif pending and allow_overlap and len(active) == 1:
                first = next(iter(active.values()))
                initial_ready = _new_initial_evidence(first)
                elapsed = time.monotonic()-first['started_monotonic']
                if initial_ready or elapsed >= SECOND_START_DELAY_SECONDS:
                    try:
                        admission = collection_resource_admission()
                    except Exception as exc:
                        admission = dict(admitted=False, error=f'{type(exc).__name__}: {exc}', reason='resource_inventory_unavailable')
                    admitted = admission.get('admitted') is True
                    decision(dict(decision='parallel_admitted' if admitted else 'serial_resource_fallback',
                        first_stage=first['job']['name'], first_elapsed_seconds=elapsed,
                        warmup_trigger='new_initial_rule_evidence' if initial_ready else '15s_startup_delay', admission=admission))
                    if not admitted:
                        allow_overlap = False
                    elif first['proc'].poll() is None:
                        start(pending.pop(0))
            if pending or active:
                time.sleep(COLLECTION_POLL_SECONDS)
        group.update(status='completed', finished_unix=time.time())
        state.update(status='running', active_stage=None)
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'
        group.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed', error=reason)
        state['status'] = 'halted_interrupted' if group['status'] == 'interrupted' else 'halted_stage_failure'
        # Signal only currently live sibling/owned child process groups. Reap
        # them before releasing the coordinator's external lock; no SIGKILL.
        for child in list(active.values()):
            if child['proc'].poll() is None:
                try:
                    os.killpg(child['proc'].pid, signal.SIGTERM)
                    child['entry']['termination_signal'] = 'SIGTERM'
                    child['entry']['cancellation_reason'] = reason
                except ProcessLookupError:
                    pass
                persist(child)
        for child in list(active.values()):
            code = child['proc'].wait()
            finish(child, code, cancellation=reason if child['entry'].get('termination_signal') else None)
        group['finished_unix'] = time.time()
        state['active_stage'] = None
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        group['pending'] = [j['name'] for j in pending]
        write_once(group_dir/'result.json', group)
        persist()


def verify_plan_sources(plan):
    for path, expected in {**plan['source_hashes'], **plan['robot_binding']['asset_sha256']}.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen code/asset changed: {path}; use a new run')


def pipeline(args):
    if args.collection_workers not in (1, 2):
        raise ValueError('collection-workers must be 1 or 2')
    if (args.source_count, args.randomized_count) != (PRODUCTION_QUOTAS['source'], PRODUCTION_QUOTAS['randomized']):
        raise ValueError('production quota is 100 source + another 200 randomized = 300 per task, 600 total')
    if not 1 <= args.evaluation_count <= 50 or not 1 <= args.evaluation_max_steps <= 12000:
        raise ValueError('evaluation count 1..50 and maximum 1200 simulation seconds required')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    log('流水线预检：冻结生活task1/2场景与相机代码、数据配额和缓存预算，供后台断点恢复', actor='run_zhuhai_pipeline')
    plan = build_plan(args)
    path = args.out / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('immutable scene/code/config plan changed; use a new run directory')
    write_once(path, plan, identical_ok=True)
    if not args.execute:
        print(json.dumps(dict(status='cpu_plan_ready' if plan['scene_ready'] else 'waiting_Isaac_runtime_and_collector', plan=plan),
                         ensure_ascii=False, indent=2))
        return 0
    if not plan['scene_ready']:
        raise RuntimeError('both scenes/rules/helpers must be complete before collection')
    verify_owner(args.gpu_lock_owner)
    state_path = args.out / 'status.json'
    if state_path.exists() and not args.resume:
        raise RuntimeError('existing pipeline requires explicit --resume')
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        status='ready', active_stage=None, stages={}, plan_sha256=sha(path))
    if state['plan_sha256'] != sha(path):
        raise RuntimeError('pipeline state and immutable plan differ')
    verify_plan_sources(plan)
    state['source_snapshot'] = snapshot_sources(args.out, plan)
    atomic(state_path, state)
    sim = str(ROOT / '.conda-envs/behavior6/bin/python')
    wam = str(ROOT / '.venvs/wam-fixed/bin/python')
    collector = str(ROOT / 'scripts/collect_isaac_zhuhai_200.py')
    trainer = str(ROOT / 'scripts/train_zhuhai_wam.py')
    evaluator = str(ROOT / 'scripts/eval_zhuhai_wam.py')
    rows = []
    for i in (1, 2):
        task = f'zhuhai_task{i}'
        rows.append(dict(task=task, source_manifest=str(args.source_root.resolve() / task / 'source/manifest.json'),
                         randomized_manifest=str(args.out / task / 'randomized/manifest.json')))
    tasks_path = args.out / 'tasks.json'
    supplied = dict(tasks=rows)
    if tasks_path.exists() and json.loads(tasks_path.read_text()) != supplied:
        raise ValueError('task manifest changed')
    write_once(tasks_path, supplied, identical_ok=True)
    # Admission uses the original validator and immutable source contracts.
    for i, spec in enumerate(rows, 1):
        name=f'source_task{i}'
        receipt=collection_receipt(Path(spec['source_manifest']), spec['task'], 'source')
        previous=state['stages'].get(name)
        if previous and previous.get('validation') != receipt:
            raise ValueError('Imported original source changed')
        state['stages'][name]=dict(status='completed', imported=True, validation=receipt, source_relabelled=False)
    atomic(state_path, state)
    collection_barrier(state, 'source')
    for mode, count in [('randomized', PRODUCTION_QUOTAS['randomized'])]:
        jobs = []
        for i, spec in enumerate(rows, 1):
            command = [sim, '-B', collector, '--task', spec['task'], '--track', 'life',
                '--mode', mode, '--count', str(count), '--seed-start', str(933000 + i * 10000 + (1000 if mode == 'randomized' else 0)),
                '--out', str(args.out / spec['task'] / mode), '--gpu-lock-owner', args.gpu_lock_owner,
                '--max-consecutive-failures', '3']
            if i == 2:
                command.extend(['--life-mapping', args.life_mapping])
            if mode == 'randomized':
                command.extend(['--source-manifest', spec['source_manifest']])
            if args.resume and (args.out / spec['task'] / mode / 'manifest.json').exists():
                command.append('--resume')
            jobs.append(dict(name=f'{mode}_task{i}', command=command,
                manifest=Path(spec[f'{mode}_manifest']),
                validate=lambda spec=spec, mode=mode: collection_receipt(
                    Path(spec[f'{mode}_manifest']), spec['task'], mode)))
        run_collection_group(args.out, mode, jobs, args, state, plan)
        collection_barrier(state, mode)
    # Both validated source groups (200) and additional DR groups (400)
    # must exist before even the tiny trainer is launched.
    collection_barrier(state, 'source')
    collection_barrier(state, 'randomized')
    common = [wam, '-B', trainer, '--tasks', str(tasks_path), '--gpu-lock-owner', args.gpu_lock_owner,
              '--cache-root', plan['cache']['root'],
              '--source-count', str(PRODUCTION_QUOTAS['source']),
              '--randomized-count', str(PRODUCTION_QUOTAS['randomized']),
              '--cache-budget-bytes', str(plan['cache']['disk_budget_bytes']),
              '--cache-ram-budget-bytes', str(plan['cache']['ram_budget_bytes']),
              '--minimum-free-bytes', str(MINIMUM_FREE_BYTES)]
    tiny = [*common, '--out', str(args.out / 'tiny_training'), '--mode', 'tiny', '--steps', '2',
            '--batch-size', '1', '--save-every', '2', '--overfit-anchors', '2']
    if args.resume and (args.out / 'tiny_training/manifest.json').exists() and state['stages'].get('tiny_training', {}).get('status') not in (None, 'completed'):
        tiny.append('--resume')
    verify_plan_sources(plan)
    run_stage(args.out, 'tiny_training', tiny, args, state,
              validate=lambda: checkpoint_receipt(args.out / 'tiny_training', 2))
    command = [*common, '--out', str(args.out / 'training'), '--mode', 'train',
               '--steps', str(args.train_steps), '--batch-size', str(args.train_batch_size)]
    if args.resume and (args.out / 'training/manifest.json').exists() and state['stages'].get('training', {}).get('status') not in (None, 'completed'):
        command.append('--resume')
    verify_plan_sources(plan)
    run_stage(args.out, 'training', command, args, state,
              validate=lambda: checkpoint_receipt(args.out / 'training', args.train_steps))
    command = [sim, '-B', evaluator, '--checkpoint', str(args.out / 'training' / f'checkpoint_step_{args.train_steps:06d}.pt'),
        '--out', str(args.out / 'evaluation'), '--execute', '--gpu-lock-owner', args.gpu_lock_owner,
        '--max-steps', str(args.evaluation_max_steps), '--seeds', *map(str, plan['evaluation_seeds'])]
    verify_plan_sources(plan)
    run_stage(args.out, 'fresh_seed_physical_evaluation', command, args, state,
              validate=lambda: evaluation_receipt(args.out / 'evaluation', plan))
    state.update(status='completed', active_stage=None, completed_unix=time.time())
    atomic(state_path, state)
    summary = json.loads((args.out / 'evaluation/summary.json').read_text())
    write_once(args.out / 'completion.json', dict(status='completed', evaluation=summary,
        source_quota_per_task=PRODUCTION_QUOTAS['source'], randomized_quota_per_task=PRODUCTION_QUOTAS['randomized'],
        total_per_task=TOTAL_PER_TASK, total_all_tasks=TOTAL_ALL_TASKS,
        state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS,
        data_contract_sha256=data_contract_hash(), robot_contract_sha256=plan['robot_contract_sha256'],
        scoring_scope='strict rigid simulation proxy', recording_video=False,
        third_view=False, camera_names=list(ONBOARD_CAMERAS)), identical_ok=True)
    log('竞赛生活任务流水线完成；保存训练里程碑和固定未见种子独立规则评分，原五任务baseline保留', actor='run_zhuhai_pipeline')
    return 0


if __name__ == '__main__':
    args = parser().parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            raise SystemExit(pipeline(args))
        except Exception as exc:
            print(f'pipeline halted: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
            raise SystemExit(2)
