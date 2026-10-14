#!/usr/bin/env python3
"""Collection only: preserve 13 task2 DR and collect the remaining 87.

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
from scripts.progress1_log_task2_repair import log
from scripts.collect_isaac_zhuhai_task2_repair import (PRODUCTION_QUOTAS, TOTAL_PER_TASK, TOTAL_ALL_TASKS,
    MINIMUM_FREE_BYTES, ONBOARD_CAMERAS, robot_binding, validate_manifest, write_once,
    sha256, check_disk, runtime_interface, data_contract, data_contract_hash, robot_spec, json_digest,
    STATE_SEMANTICS, ACTION_SEMANTICS, RESOURCE_POLICY, RECYCLE_EXIT_CODE, WORKER_RECYCLING, validate_recycle_boundary,
    prior_rows, residual_families, SHARED_CONTRACT_FIELDS)

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
    ap.add_argument('--collection-root',required=True,type=Path,help='Independent corrected collection output root')
    ap.add_argument('--prior-inventory',required=True,type=Path)
    ap.add_argument('--deferred-root',required=True,type=Path,help='Preserved garbage collection evidence; never resume it in this run')
    ap.add_argument('--source-root', required=True, type=Path, help='Original frozen run; import completed 100-source manifests without relabelling')
    ap.add_argument('--life-mapping', choices=['body_text', 'scoring_table'], required=True)
    ap.add_argument('--collection-workers', type=int, choices=(1, 2, 3, 4), default=3,
                    help='Only task2 collection shards may overlap; low/unknown RAM or GPU headroom falls back to serial')
    ap.add_argument('--source-count', type=int, default=PRODUCTION_QUOTAS['source'])
    ap.add_argument('--randomized-count', type=int, default=PRODUCTION_QUOTAS['randomized'])
    ap.add_argument('--gpu-lock-owner')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--resume', action='store_true')
    # Compatibility values used only while constructing the inherited static
    # interface inventory; all training/evaluation fields are removed below.
    ap.set_defaults(train_steps=20000,train_batch_size=4,evaluation_count=10,evaluation_max_steps=2400,
        cache_root='data/zhuhai_native16_hybrid_features_20261009')
    return ap


def validate_prior_import(args):
    """Bind the 13 preserved trials and their exact remaining source families."""
    source = args.source_root.resolve()/'zhuhai_task2/source/manifest.json'
    original = validate_manifest(source, task='zhuhai_task2', mode='source')
    previous = prior_rows(args.prior_inventory, verify_files=True)
    if len(original) != 100 or len(previous) != 13:
        raise ValueError('Repair handoff requires 100 original sources and 13 preserved successful DR')
    source_contract = json.loads(source.read_text())['contract']
    binding = {key: source_contract.get(key) for key in SHARED_CONTRACT_FIELDS}
    binding.update(source_manifest=dict(path=str(source), sha256=sha(source), engine=source_contract['engine']),
        prior_inventory=dict(path=str(args.prior_inventory.resolve()), sha256=sha(args.prior_inventory)),
        collection_sharding=dict(shard_count=args.collection_workers,
            family_assignment='source seeds ascending; index modulo shard_count',
            global_randomized_count=100, randomized_per_family=1,
            disjoint_seed_stride=2000 if args.collection_workers==1 else 1000))
    remaining = [residual_families(binding, index, args.collection_workers, verify_files=True)
                 for index in range(args.collection_workers)]
    if sum(map(len, remaining)) != 87:
        raise ValueError('Exactly 87 disjoint remaining source families required')
    if args.collection_root.resolve() == args.source_root.resolve():
        raise ValueError('Corrected collection must not modify original source evidence')
    prior_directories = [Path(ref['path']).resolve().parent
        for ref in json.loads(args.prior_inventory.read_text())['components']]
    for index in range(args.collection_workers):
        target = args.collection_root.resolve()/'zhuhai_task2/randomized/shards'/f'shard_{index:02d}'
        if target in prior_directories:
            raise ValueError('Corrected collection must not resume or overwrite prior stopped shards')
    return dict(path=str(args.prior_inventory.resolve()), sha256=sha(args.prior_inventory),
        successful_episodes=len(previous), original_success_evidence_unchanged=True,
        new_successful_episode_target=87, residual_shard_targets=list(map(len, remaining)),
        source_manifest_sha256=sha(source))


def build_plan(args):
    prior_import = validate_prior_import(args)
    scenes = [ROOT / 'harness' / f'isaac_zhuhai_task{i}.py' for i in (1, 2)]
    files = [Path(__file__).resolve(), ROOT/'harness/isaac_zhuhai_task2_expert_repair_v3.py',ROOT/'scripts/probe_zhuhai_task2_expert_repair_v3.py', ROOT/'scripts/run_zhuhai_task2_first.py', ROOT / 'scripts/collect_isaac_zhuhai.py',
        ROOT / 'scripts/progress1_log.py', ROOT/'scripts/progress1_log_200.py',
        ROOT/'scripts/collect_isaac_zhuhai_200.py', ROOT/'scripts/handoff_zhuhai_200.py', ROOT/'scripts/handoff_zhuhai_100_parallel.py', ROOT/'scripts/handoff_zhuhai_100_reuse.py',
        ROOT/'scripts/train_zhuhai_task2_reuse.py', ROOT/'scripts/train_zhuhai_wam_reuse.py', ROOT/'harness/zhuhai_preparation_reuse.py',
        ROOT/'scripts/collect_isaac_zhuhai_task2_repair.py', ROOT/'scripts/benchmark_zhuhai_worker_scaling.py', ROOT/'scripts/progress1_log_task2_repair.py',
        ROOT / 'harness/isaac_zhuhai_runtime.py', ROOT / 'scripts/collect_zhuhai_data.py',
        ROOT / 'harness/zhuhai_robot_spec.py', ROOT / 'harness/robotwin_dataset.py',
        ROOT / 'harness/isaac_zhuhai_expert.py', ROOT / 'harness/isaac_zhuhai_collision.py', ROOT / 'scripts/zhuhai_resource_watch.py',
        ROOT / 'scripts/train_zhuhai_wam.py', ROOT / 'scripts/eval_zhuhai_wam.py',
        ROOT / 'harness/zhuhai_training.py', ROOT / 'harness/zhuhai_dataset.py',
        ROOT / 'harness/zhuhai_hybrid_cache.py', ROOT / 'harness/zhuhai_demo_recording.py',
        ROOT / 'harness/zhuhai_task1_rules.py', ROOT / 'harness/zhuhai_task2_rules.py',
        ROOT / 'harness/isaac_zhuhai_camera.py', ROOT / 'harness/zhuhai_action_smoothing.py',
        ROOT / 'third_party/LiLa-WAM/robotwin_infer.py',
        ROOT / '.conda-envs/behavior6/lib/python3.12/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py', robot_spec.ASSET_MANIFEST,
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
    plan = dict(schema='zhuhai-isaac-native16-task2-repaired-collection-only-v1', engine=dict(name='Isaac Sim', version='6.1.0.0'),
        execution_status='Isaac-only adapter; SAPIEN execution disabled', source_count=args.source_count,
        training_tasks=['zhuhai_task2'], current_run_successful_episode_target=200,
        prior_DR_import=prior_import,collection_root=str(args.collection_root.resolve()),repair_reason='Three preserved expert failures; independently changed grasp/transfer planner, old failure streaks untouched',allocator_environment=dict(MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072'),
        deferred_tasks=[dict(task='zhuhai_task1',source_manifest=str(args.source_root.resolve()/'zhuhai_task1/source/manifest.json'),
            randomized_manifest=str(args.deferred_root.resolve()/'zhuhai_task1/randomized/manifest.json'))],
        deferred_plan_sha256=sha(args.deferred_root/'plan.json'),
        later_training='append garbage in a separate native16 continuation with task2 replay; budget decided after task2 tests',
        worker_recycling=dict(WORKER_RECYCLING),
        collection_workers=args.collection_workers,
        randomized_shards_per_task=args.collection_workers,
        shard_policy='Each worker job owns source seeds modulo shard count; same family never split; one DR relative/family; immutable aggregate index preserves all original attempts',
        collection_scheduling=dict(max_workers=args.collection_workers, source_barrier_before_randomization=True,
            same_external_gpu_lock_owner=True, training_and_evaluation_workers=1,
            second_start_after='new first-worker initial_rule_evidence OR 15s after its spawn',
            second_start_delay_seconds=SECOND_START_DELAY_SECONDS,
            minimum_available_ram_bytes=7*GIB, minimum_free_gpu_bytes=3*GIB, maximum_swap_fraction=.90,
            resource_policy=dict(RESOURCE_POLICY),
            runtime_fallback='At four workers and >=90% swap, or real RAM/GPU reserve reached, stop only newest owned child, preserve partial, requeue with --resume and reduce concurrency',
            low_or_unknown_resources='record serial fallback before any second spawn',
            failure='halt later stages; SIGTERM only live child process groups spawned by this group; wait/reap without SIGKILL'),
        source_snapshot_policy=dict(directory='source_snapshot', receipt='source_snapshot/receipt.json',
            source='plan.source_hashes code and small configurations', max_file_bytes=SNAPSHOT_FILE_LIMIT,
            max_total_bytes=SNAPSHOT_TOTAL_LIMIT, weights_and_HDF_excluded=True, overwrite=False),
        randomized_count=args.randomized_count, total_per_task=TOTAL_PER_TASK, total_all_tasks=TOTAL_PER_TASK,
        quota_interpretation='task2 only: 100 original successful source + 100 successful DR; garbage evidence deferred',
        source_import=dict(root=str(args.source_root.resolve()), original_plan_sha256=sha(args.source_root/'plan.json'), source_evidence_modified=False),
        family_split=dict(source_families=dict(train=80, val=10, heldout=10),
            successful_randomized_per_family=1, episodes_per_task=dict(train=160, val=20, heldout=20),
            VTT_source='TRAIN only'), robot_binding=binding, runtime_interface=interface,
        robot_contract=binding['robot_contract'], robot_contract_sha256=binding['robot_contract_sha256'],
        data_contract=data_contract(), data_contract_sha256=data_contract_hash(),
        state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS,
        train_steps=args.train_steps, train_batch_size=args.train_batch_size,
        training_milestones=[args.train_steps], checkpoint_save_interval=5000,
        preparation_reuse=dict(root=str(args.out.resolve()/'preparation'), algorithm_unchanged=True,
            content_encoder_history_and_code_hash_bound=True, first_fit_required=True, repeated_full_fit_avoided=True,
            numerical_GPU_gate='full admitted data tiny2 before first 5k; halt on mismatch'),
        milestone_protocol=dict(continuous_optimizer_and_rng=True, fixed_20k_lr_schedule=True,
            evaluation_between_training_segments=False, evaluation_after_all_20000_updates=True, same_seeds_at_every_milestone=True,
            evaluation_failures_retained=True, checkpoint_selection=False),
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
        stages=['source_task2', 'randomized_task2', 'tiny_training',
                'training_020000', 'evaluation_020000'],
        failure_policy='3 consecutive collector failures halt; no skipped quotas or silent seed replacement',
        resume_policy='completed stages are hash-revalidated; incomplete core train/eval output is protected pending transactional resume',
        scene_ready=ready, readiness_scope='static native16 interface/assets only; not physical expert/mechanical acceptance', source_hashes={str(path): sha(path) for path in files if path.is_file()})
    for key in ('train_steps','train_batch_size','training_milestones','checkpoint_save_interval','preparation_reuse',
                'milestone_protocol','initialization','evaluation_count','evaluation_max_steps','evaluation_seeds','cache',
                'later_training','family_split'):
        plan.pop(key,None)
    plan.update(training_authorized=False,automatic_training=False,automatic_evaluation=False,
        execution_scope='task2 collection only; task1 collection repair follows separately; never train automatically',
        stages=['source_task2','randomized_task2'],
        next_action='task1 remaining DR collection after independent expert repair acceptance; training requires later user instruction')
    return plan


def collection_receipt(path, task, mode, **shard):
    rows = validate_manifest(path, task=task, mode=mode, **shard)
    return dict(manifest=str(path), manifest_sha256=sha(path), successful_episodes=len(rows),
                source_families=len({r['family_id'] for r in rows}), state_dim=16, action_dim=16,
                state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS, data_contract_sha256=data_contract_hash())






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
    if existing and (name in ('tiny_training', 'training', 'fresh_seed_physical_evaluation') or name.startswith(('training_', 'evaluation_'))):
        target = (out/'training' if name.startswith('training_') else out/'evaluations'/('step_'+name.split('_')[-1]) if name.startswith('evaluation_') else out / ('evaluation' if name == 'fresh_seed_physical_evaluation' else name))
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
    from scripts.zhuhai_resource_watch import memory_info, gpu_info
    answer=dict(minimum_available_ram_bytes=7*GIB, minimum_free_gpu_bytes=3*GIB,
                maximum_swap_fraction=.90, checked_unix=time.time(), read_only=True, cuda_context_created=False)
    try:
        memory,gpu=memory_info(),gpu_info()
        free=min((d['total_bytes']-d['used_bytes'] for d in gpu.get('devices',[])),default=None)
        fraction=memory['swap_used_bytes']/memory['swap_total_bytes'] if memory['swap_total_bytes'] else 0
        answer.update(memory=memory,gpu=gpu,available_ram_bytes=memory['available_bytes'],gpu_free_bytes=free,swap_fraction=fraction,
            admitted=memory['available_bytes']>=7*GIB and free is not None and free>=3*GIB and fraction<.95,
            fourth_admitted=memory['available_bytes']>=7*GIB and free is not None and free>=3*GIB and fraction<.90)
        answer['reason']='headroom_sufficient' if answer['admitted'] else 'low_or_unknown_RAM_GPU_swap_headroom'
    except Exception as exc:answer.update(admitted=False,reason='resource_inventory_unavailable',error=str(exc))
    return answer


def _command_value(command, flag):
    if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
        raise ValueError(f'Exactly one {flag} required for an isolated collector')
    return command[command.index(flag) + 1]


def _collection_jobs(out, mode, jobs, args):
    expected={(f'zhuhai_task{i}',j) for i in (2,) for j in range(args.collection_workers)}
    if mode!='randomized' or {(j['task'],j['shard_index']) for j in jobs}!=expected or len(jobs)!=len(expected):
        raise ValueError('Exactly one disjoint family job per task/shard required')
    streams=[]
    for job in jobs:
        task,index=job['task'],job['shard_index']; command=job['command']
        target=Path(args.collection_root).resolve()/task/'randomized/shards'/f'shard_{index:02d}'
        if (Path(_command_value(command,'--out')).resolve()!=target or Path(job['manifest'])!=target/'manifest.json'
                or _command_value(command,'--task')!=task or _command_value(command,'--mode')!=mode
                or _command_value(command,'--count')!='100' or _command_value(command,'--shard-index')!=str(index)
                or _command_value(command,'--shard-count')!=str(args.collection_workers)
                or _command_value(command,'--gpu-lock-owner')!=args.gpu_lock_owner
                or Path(_command_value(command,'--prior-inventory')).resolve()!=args.prior_inventory.resolve()
                or Path(_command_value(command,'--source-manifest'))!=args.source_root.resolve()/task/'source/manifest.json'):
            raise ValueError('Shard output/quota/parent/owner isolation mismatch')
        start=int(_command_value(command,'--seed-start'))+index*(2000 if args.collection_workers==1 else 1000)
        streams.append(start)
    if len(set(streams))!=len(streams):raise ValueError('Effective shard seed reservations collide')


def _new_initial_evidence(child):
    try:
        rows = json.loads(child['job']['manifest'].read_text()).get('attempts', [])
        return any(isinstance(row.get('initial_rule_evidence'), dict) for row in rows[child['initial_attempts']:])
    except (OSError, ValueError, TypeError):
        return False


def collection_barrier(state, mode):
    """Only per-child validated receipts satisfy the source/DR quota barrier."""
    for i in (2,):
        stage = state['stages'].get(f'{mode}_task{i}', {})
        validation = stage.get('validation') or {}
        if (stage.get('status') != 'completed' or validation.get('successful_episodes') != PRODUCTION_QUOTAS[mode]
                or validation.get('source_families') != PRODUCTION_QUOTAS['source']):
            raise RuntimeError(f'{mode} task{i} quota/validation barrier incomplete; no following stages')


def run_collection_group(out, mode, jobs, args, state, plan):
    """Main-thread scheduler: no run_stage threads, shared .tmp or nested locks."""
    _collection_jobs(out, mode, jobs, args)
    # Check every prior stage before launching anything, including live old PIDs.
    pending = [j for j in jobs if stage_needs_run(out, j['name'], j['command'], args, state, j['validate'])]
    if not pending:
        return
    previous = state.setdefault('collection_groups', {}).get(mode)
    number = previous.get('attempt_number', 0) + 1 if previous else 1
    group_dir = out/'collection_groups'/mode/f'attempt_{number:04d}'
    group_dir.mkdir(parents=True, exist_ok=False)
    group = dict(status='running', mode=mode, requested_workers=args.collection_workers, peak_active_workers=0,
                 started_unix=time.time(), attempt_number=number, previous_attempt=previous,
                 pending=[j['name'] for j in pending], decisions=[], external_gpu_lock_owner=args.gpu_lock_owner)
    state['collection_groups'][mode] = group
    state.update(status='running', active_stage=mode+'_collection', active_stages=[])
    write_once(group_dir/'intent.json', group)
    atomic(out/'status.json', state)
    active = {}
    allow_overlap = True
    runtime_cap=args.collection_workers
    resource_checked=0.
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
            if len(active)>=3 and time.monotonic()-resource_checked >= (1 if len(active)>=4 else 5):
                admission=collection_resource_admission();resource_checked=time.monotonic()
                memory=admission.get('memory',{});free=admission.get('gpu_free_bytes')
                minimum_ram=max(RESOURCE_POLICY['minimum_available_ram_bytes'],
                                RESOURCE_POLICY['minimum_available_ram_fraction']*memory.get('total_bytes',32*GIB))
                violated=(memory.get('available_bytes',0)<minimum_ram
                    or (len(active)>=4 and admission.get('swap_fraction',1)>=.90)
                    or free is None or free<RESOURCE_POLICY['minimum_free_gpu_bytes'])
                if violated:
                    child=max(active.values(),key=lambda c:c['started_monotonic'])
                    decision(dict(decision='runtime_resource_fallback', stage=child['job']['name'], admission=admission, next_worker_cap=len(active)-1))
                    if child['proc'].poll() is None:
                        try:os.killpg(child['proc'].pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                    code=child['proc'].wait()
                    finish(child,code,cancellation='resource fallback; partial retained, disjoint shard requeued')
                    job=dict(child['job']);job['command']=list(job['command'])
                    if '--resume' not in job['command']:job['command'].append('--resume')
                    pending.insert(0,job);runtime_cap=max(1,len(active));allow_overlap=True
            if stop_requested['signal'] is not None:
                raise KeyboardInterrupt(f'collection group interrupted by signal {stop_requested["signal"]}')
            failure = None
            for child in list(active.values()):
                code = child['proc'].poll()
                if code is not None:
                    if code == RECYCLE_EXIT_CODE:
                        boundary=validate_recycle_boundary(child['job']['manifest'],task=child['job']['task'])
                        decision(dict(decision='planned_worker_recycle',stage=child['job']['name'],boundary=boundary))
                        finish(child,code,cancellation='planned memory recycle at completed physical attempt boundary')
                        job=dict(child['job']);job['command']=list(job['command'])
                        if '--resume' not in job['command']:job['command'].append('--resume')
                        pending.append(job)
                        # Fresh workers may restore four when existing admission gates allow it.
                        runtime_cap=args.collection_workers
                    else:
                        error = finish(child, code)
                        failure = failure or error
            if failure:
                raise failure
            if pending and not active:
                if group['peak_active_workers'] and not group['decisions']:
                    decision(dict(decision='serial_first_finished_before_overlap', checked_unix=time.time()))
                start(pending.pop(0))
            elif pending and allow_overlap and len(active) < runtime_cap:
                first = max(active.values(),key=lambda c:c['started_monotonic'])
                initial_ready = _new_initial_evidence(first)
                elapsed = time.monotonic()-first['started_monotonic']
                if initial_ready or elapsed >= SECOND_START_DELAY_SECONDS:
                    try:
                        admission = collection_resource_admission()
                    except Exception as exc:
                        admission = dict(admitted=False, error=f'{type(exc).__name__}: {exc}', reason='resource_inventory_unavailable')
                    admitted = admission.get('admitted') is True and (len(active)<3 or admission.get('swap_fraction',1)<.90)
                    decision(dict(decision='parallel_admitted' if admitted else 'serial_resource_fallback',
                        first_stage=first['job']['name'], first_elapsed_seconds=elapsed,
                        warmup_trigger='new_initial_rule_evidence' if initial_ready else '15s_startup_delay', admission=admission))
                    if not admitted:
                        runtime_cap=max(1,len(active))
                        allow_overlap = True
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
    if args.collection_workers not in (1, 2, 3, 4):
        raise ValueError('collection-workers must be 1..4')
    if (args.source_count, args.randomized_count) != (PRODUCTION_QUOTAS['source'], PRODUCTION_QUOTAS['randomized']):
        raise ValueError('production quota is 100 source + another 100 randomized = 200 total for task2')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    log('仅采集餐具预检：保留13条DR并补87，禁用自动训练和评测；垃圾采集修复另行追加', actor='run_zhuhai_collection_only')
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
    collector = str(ROOT / 'scripts/collect_isaac_zhuhai_task2_repair.py')
    rows = []
    for i in (2,):
        task = f'zhuhai_task{i}'
        rows.append(dict(task=task, source_manifest=str(args.source_root.resolve() / task / 'source/manifest.json'),
                         randomized_manifest=str(args.collection_root.resolve() / task / 'randomized/manifest.json')))
    tasks_path = args.out / 'tasks.json'
    supplied = dict(tasks=rows)
    if tasks_path.exists() and json.loads(tasks_path.read_text()) != supplied:
        raise ValueError('task manifest changed')
    write_once(tasks_path, supplied, identical_ok=True)
    # Admission uses the original validator and immutable source contracts.
    for spec in rows:
        i=int(spec['task'][-1])
        name=f'source_task{i}'
        receipt=collection_receipt(Path(spec['source_manifest']), spec['task'], 'source')
        previous=state['stages'].get(name)
        if previous and previous.get('validation') != receipt:
            raise ValueError('Imported original source changed')
        state['stages'][name]=dict(status='completed', imported=True, validation=receipt, source_relabelled=False)
    atomic(state_path, state)
    collection_barrier(state, 'source')
    mode='randomized';jobs=[]
    for spec in rows:
        i=int(spec['task'][-1])
        for index in range(args.collection_workers):
            target=args.collection_root.resolve()/spec['task']/'randomized/shards'/f'shard_{index:02d}'
            command=[sim,'-B',collector,'--task',spec['task'],'--track','life','--mode',mode,'--count','100',
                '--seed-start',str(943000+i*10000+1000),'--out',str(target),'--gpu-lock-owner',args.gpu_lock_owner,
                '--max-consecutive-failures','3','--shard-index',str(index),'--shard-count',str(args.collection_workers),
                '--source-manifest',spec['source_manifest'],'--prior-inventory',str(args.prior_inventory.resolve())]
            if i==2:command.extend(['--life-mapping',args.life_mapping])
            if (target/'manifest.json').exists():command.append('--resume')
            jobs.append(dict(name=f'randomized_task{i}_shard{index}',task=spec['task'],shard_index=index,command=command,
                manifest=target/'manifest.json',validate=lambda target=target,spec=spec,index=index: collection_receipt(
                    target/'manifest.json',spec['task'],mode,shard_index=index,shard_count=args.collection_workers)))
    run_collection_group(args.out,mode,jobs,args,state,plan)
    for spec in rows:
        i=int(spec['task'][-1])
        paths=[args.collection_root.resolve()/spec['task']/'randomized/shards'/f'shard_{j:02d}'/'manifest.json' for j in range(args.collection_workers)]
        from scripts.collect_isaac_zhuhai_task2_repair import aggregate_shards
        aggregate_shards(Path(spec['randomized_manifest']), paths, task=spec['task'])
        state['stages'][f'randomized_task{i}']=dict(status='completed',aggregated=True,
            validation=collection_receipt(Path(spec['randomized_manifest']),spec['task'],'randomized'))
    atomic(state_path,state);collection_barrier(state,'randomized')
    state.update(status='completed_collection_only',active_stage=None,completed_unix=time.time(),
        training_authorized=False,automatic_training=False,automatic_evaluation=False,
        next_action='task1 remaining DR collection after independent expert repair acceptance')
    atomic(state_path,state)
    write_once(args.out/'completion.json',dict(status='completed_collection_only',
        task='zhuhai_task2',source_successful_episodes=100,randomized_successful_episodes=100,
        randomized_validation=state['stages']['randomized_task2']['validation'],
        training_authorized=False,automatic_training=False,automatic_evaluation=False,
        source_and_prior_evidence_modified=False,recording_video=False,third_view=False),identical_ok=True)
    log('餐具采集完成100源+100DR；按用户最新要求停止于数据屏障，不执行tiny、训练或评测',actor='run_zhuhai_collection_only')
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
