#!/usr/bin/env python3
"""CPU plan by default; concurrent execution is DENIED by the frozen protocol.

--execute explicitly queues ONE original DR200 worker AFTER the entire successful
V2 pipeline, including evaluation, exits. This is serial, never an idle-GPU or
same-owner shortcut. No frozen module is patched. No discovered PID is signalled.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/zhuhai2026_tasks12'
VERIFY = BASE / 'verification/tasks12_collection_repair'
SCHEMA = 'task1_dr200_frozen_serial_fallback_v1'
QUEUE_LOCK = Path('/tmp/project2_task1_dr200_after_training.lock')
PROBE = VERIFY / 'task1_expert_repair_v2_20261009_c/production_admission.json'
PRIOR = VERIFY / 'task1_prior_success_inventory.json'
SOURCE = BASE / 'production_20261009_v1'
DEFERRED = BASE / 'production_100dr_4workers_20k_reuse_20261009_v6'
TASK2 = BASE / 'production_task2_collection_only_20261009_v11'
TASK2_PRIOR = BASE / 'production_task2_final10_20k_20261009_v8/verification/expert_repair_v1/prior_success_inventory.json'
MINIMUM_CHANGES = {
    'scripts/gpu_lock.py': 'Real distinct participant leases/resource reservations; retain legacy exclusive mode.',
    'harness/zhuhai_precision_v2_training.py': 'Role-aware owner verification instead of inherited exclusive string comparison.',
    'scripts/train_zhuhai_wam_precision_v2.py': 'Measured full B4/DINO/AdamW/checkpoint/diagnostic RAM+VRAM envelope and cooperative admission.',
    'scripts/run_zhuhai_wam_precision_v2_pipeline.py': 'Phase handshake: stop collection admission, drain/reap workers, then exclusive eval; idle training child is not idle pipeline.',
    'scripts/run_zhuhai_tasks12_collection_repair_200.py': 'Lease-aware scheduler and durable drain acknowledgement; retain failure/recycle/partial rules.',
    'scripts/collect_isaac_zhuhai_tasks12_repair_200.py': 'Versioned collector lease verification instead of inherited collect_zhuhai_data.verify_owner.',
    'harness/isaac_zhuhai_runtime.py': 'Simulator currently compares exclusive owner literally; must validate a scoped lease.',
    'scripts/zhuhai_resource_watch.py': 'Training/collector process-tree attribution, cold-start peaks and fail-closed aggregate budget.',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run', type=Path, default=BASE / 'wam_precision_v2_task2_30k_20261010_v1')
    p.add_argument('--out', type=Path, default=BASE / 'production_tasks12_dr200_collection_only_20261010_serial_v1')
    p.add_argument('--gpu-lock-owner', default='zhuhai-task1-dr200-serial-20261010-v1')
    p.add_argument('--workers', type=int, choices=(1,), default=1)
    p.add_argument('--schedule', choices=('after-training', 'concurrent'), default='after-training')
    p.add_argument('--wait-seconds', type=float, default=172800)
    p.add_argument('--execute', action='store_true', help='Serial after-training only; never runs concurrently')
    return p


def scheduler_module():
    from scripts import run_zhuhai_tasks12_collection_repair_200
    return run_zhuhai_tasks12_collection_repair_200


def scheduler_args(args, execute=False):
    values = dict(out=args.out, collection_root=args.out / 'collection', source_root=SOURCE,
        deferred_root=DEFERRED, task2_collection_run=TASK2, task1_prior_inventory=PRIOR,
        task2_prior_inventory=TASK2_PRIOR, task1_acceptance=PROBE, life_mapping='body_text',
        collection_workers=1, source_count=100, randomized_count=200, gpu_lock_owner=args.gpu_lock_owner)
    return [x for k, v in values.items() for x in ('--' + k.replace('_', '-'), str(v))] + (['--execute'] if execute else [])


def validate(args):
    for path in (args.out, args.training_run):
        require(path.absolute() == path.resolve() and BASE.resolve() in path.resolve().parents,
                'Run paths must be nonsymlink descendants of the task output root')
    require(args.out != args.training_run and args.training_run not in args.out.parents
            and args.out not in args.training_run.parents, 'Training/collection outputs must be disjoint')
    require(re.fullmatch(r'production_tasks12_dr200_collection_only_[A-Za-z0-9_.-]+', args.out.name),
            'Use a fresh DR200 production namespace, not an old evidence directory')
    require(re.fullmatch(r'zhuhai-task1-dr200-serial-[A-Za-z0-9_.-]+', args.gpu_lock_owner),
            'A dedicated serial collection owner is required')
    require(args.workers == 1 and math.isfinite(args.wait_seconds) and 0 < args.wait_seconds <= 172800,
            'One worker and a bounded positive wait of at most 48h are required')


def tiny_review(run):
    """Hash/read only. CUDA allocator peaks are NOT whole-machine admission."""
    directory = run / 'tiny_admission'
    result = dict(path=str(directory), complete_b4_updates_verified=False, resource_admitted=False)
    try:
        status = read(directory / 'status.json')
        manifest = read(directory / 'manifest.json'); c = manifest['contract']
        require(manifest['contract_sha256'] == hashlib.sha256(json.dumps(c, sort_keys=True, allow_nan=False).encode()).hexdigest(), 'Tiny contract hash mismatch')
        require(c['format'] == 'zhuhai-precision-wam-native16-measured-10hz-v2'
                and c['mode'] == 'tiny' and c['hjepa']['variant'] == 'combined', 'Full V2 combined tiny required')
        cfg = c['config']['training']
        require(cfg['batch_size'] == 4 and cfg['grad_accum_steps'] == 1 and cfg['max_steps'] == 2,
                'Actual B4 two optimizer updates required, not preprocessing or microbatch accumulation')
        splits = c['splits']['zhuhai_task2']
        require({k: len(v) for k, v in splits.items()} == dict(train=160, val=20, heldout=20), 'Full 100+100 split required')
        require(Counter(r['domain'] for rows in splits.values() for r in rows) == Counter(source=100, randomized=100), 'Tiny cannot reduce the data quota')
        require(status['status'] == 'completed' and status['step'] == status['target_steps'] == 2
                and status['auxiliary_gradient_admission_passed'] is True
                and status['precision_gradient_admission_passed'] is True, 'Tiny numerical/gradient admission incomplete')
        require(c.get('source_files'), 'Missing frozen source bindings')
        for path, digest in c['source_files'].items():
            require(sha(path) == digest, 'Tiny frozen source drift: ' + path)
        for name, model_only in (('checkpoint_step_000002.pt', True), ('checkpoint_last.pt', False)):
            ck = directory / name; receipt = read(ck.with_name(ck.name + '.sha256.json'))
            require(receipt['step'] == 2 and receipt['model_only'] is model_only
                    and receipt['size'] == ck.stat().st_size and receipt['sha256'] == sha(ck), 'Tiny checkpoint receipt mismatch')
        rows = [json.loads(line) for line in (directory / 'log.jsonl').read_text().splitlines() if line.strip()]
        require([r['step'] for r in rows] == [1, 2], 'Exactly two completed optimizer log rows required')
        for row in rows:
            names = ['loss', 'gradient_norm', 'shared_visual_grad_norm', 'idm_head_grad_norm',
                     'future8_offset_grad_norm', 'precision_clock_grad_norm', 'precision_residual_grad_norm',
                     'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes']
            if row['step'] == 2:
                names += ['precision_patch_proj_grad_norm', 'precision_attention_grad_norm']
            require(all(type(row[n]) in (int, float) and math.isfinite(row[n]) and row[n] > 0 for n in names),
                    'Missing/nonfinite gradient or CUDA peak evidence')
            require(row['cuda_peak_reserved_bytes'] >= row['cuda_peak_allocated_bytes'], 'Invalid CUDA peak ordering')
        result.update(complete_b4_updates_verified=True,
            cuda_peak_reserved_bytes=max(r['cuda_peak_reserved_bytes'] for r in rows), log_sha256=sha(directory / 'log.jsonl'))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result['reason'] = f'{type(exc).__name__}: {exc}'
    result['resource_limit'] = ('30s availableRAM/GPUused samples and torch allocator peaks do not bound whole-process '
        'DINO/cold-start/checkpoint/diagnostic transients or a concurrent Isaac worker. No shared admission issued.')
    result['stage_resource_archive'] = str(run / 'tiny_admission.resources.jsonl')
    return result


def pipeline_finished(run):
    """Successful WHOLE pipeline only; no train/eval boundary is a release."""
    path = run / 'status.json'
    if not path.exists():
        return False
    state = read(path)
    if state.get('status') in ('failed', 'halted_stage_failure', 'halted_interrupted'):
        raise RuntimeError('Training pipeline failed; stop queue without consuming a production attempt')
    if state.get('status') != 'completed':
        return False
    plan = read(run / 'plan.json')
    require(plan['schema'] == 'zhuhai-precision-wam-task2-30k-pipeline-v2', 'Wrong training pipeline version')
    names = [stage['name'] for stage in plan['stages']]
    require(names and len(names) == len(set(names)) and state.get('active_stage') is None
            and state.get('completed_stages') == names, 'All planned stages, including eval, must complete')
    require(any(s['kind'] == 'eval' and s['step'] == 30000 for s in plan['stages']), 'Final 30k evaluation missing')
    return True


def processes():
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / 'stat').read_text(); fields = text[text.rfind(')') + 2:].split()
            rows.append(dict(pid=int(entry.name), ppid=int(fields[1]), state=fields[0],
                argv=(entry / 'cmdline').read_text().rstrip('\0').split('\0')))
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Permission/inventory errors propagate: no unsafe admission on missing telemetry.
    return rows


def gpu_jobs(rows, self_pid=None):
    """Inspect actual interpreter entry points, not arbitrary ancestor argv text.

    A real training ancestor still blocks. We NEVER signal it or any foreign PID.
    Shell -c text mentioning train_*.py and a gpu_lock wrapper are not GPU children.
    """
    jobs = []
    for row in rows:
        argv = row['argv']
        if row['state'] == 'Z' or row['pid'] == (self_pid or os.getpid()) or not argv:
            continue
        exe = Path(argv[0]).name.lower(); name = exe
        if exe.startswith('python'):
            rest = iter(argv[1:]); name = ''
            for item in rest:
                if item in ('-W', '-X'):
                    next(rest, None)
                elif item in ('-c', '-m'):
                    code = next(rest, '')
                    modules = [code] if item == '-m' else []
                    if item == '-m':
                        name = code.rsplit('.', 1)[-1] + '.py'  # Same leaf-name guard as a script entrypoint.
                    if item == '-c':
                        try:
                            for node in ast.walk(ast.parse(code)):
                                if isinstance(node, ast.Import): modules.extend(a.name for a in node.names)
                                elif isinstance(node, ast.ImportFrom): modules.append(node.module or '')
                        except SyntaxError:
                            modules = ['torch']  # Uninspectable interpreter payload: fail closed.
                    if any(m.split('.')[0] in ('torch', 'isaacsim', 'omni', 'sapien', 'mujoco', 'cupy') for m in modules):
                        name = 'unclassified-python-workload'
                    break
                elif not item.startswith('-'):
                    name = Path(item).name; break
        if (name == 'unclassified-python-workload' or name in ('kit', 'isaacsim', 'isaac-sim', 'isaac-sim.sh')
                or (name.endswith('.py') and name.startswith(('train_', 'eval_', 'collect_', 'probe_', 'run_zhuhai_', 'cache_dino_')))):
            jobs.append(row)
    return jobs


@contextmanager
def file_lock(path, shared=False):
    fd = os.open(path, os.O_RDONLY if shared else os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def queue_record(directory, value, name='status.json'):
    path = directory / name
    temp = directory / (name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def review(args):
    validate(args)
    try:
        proof = scheduler_module().accepted_task1_expert(PROBE)
        probes = dict(strict_passes=proof['probe_success_count'], seeds=proof['seeds'], production_quota_credit=0,
                      acceptance_path=str(PROBE), acceptance_sha256=sha(PROBE))
    except Exception as exc:
        probes = dict(error=f'{type(exc).__name__}: {exc}', production_quota_credit=0)
    argv = [str(ROOT / '.conda-envs/behavior6/bin/python'), '-B', str(Path(__file__).resolve()),
            '--training-run', str(args.training_run), '--out', str(args.out), '--gpu-lock-owner', args.gpu_lock_owner,
            '--workers', '1', '--schedule', 'after-training', '--execute']
    return dict(schema=SCHEMA, reviewed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), mode='plan',
        concurrent_admitted=False, collection_started=False, active_collection_workers=0, planned_workers=1,
        training_preparation_must_not_wait_for_this_review=True, tiny=tiny_review(args.training_run), probes=probes,
        blockers=['FROZEN_EXCLUSIVE_GPU_OWNER_PROTOCOL', 'NO_EVAL_DRAIN_OR_SHARED_LEASE_PROTOCOL',
                  'NO_FULL_TRAIN_PLUS_ISAAC_PEAK_RESOURCE_ADMISSION'],
        minimum_shared_protocol_changes=MINIMUM_CHANGES, frozen_files_modified=False,
        serial_fallback=dict(argv=argv, command=shlex.join(argv), automatically_started=False,
            output_fresh=not args.out.exists(), requires_successful_whole_pipeline_exit=True,
            requires_pipeline_flock_released=True, requires_exclusive_new_owner=True,
            on_training_failure='stop without collection', max_wait_seconds=args.wait_seconds),
        collection=dict(source_target=100, randomized_target=200, preserved_dr=48, new_dr_target=152,
            probe_quota=0, original_scheduler='scripts/run_zhuhai_tasks12_collection_repair_200.py',
            no_resume_or_guard_reset=True, all_attempts_and_partial_preserved=True,
            record_video=False, extra_camera=False, onboard_sensor_archives_preserved=True),
        foreign_process_policy='No kill/signals. Inspect executable entrypoints; real GPU ancestors block; no blanket ancestor exemption.')


def execute_serial(args):
    validate(args)
    require(args.schedule == 'after-training', 'Concurrent execution prohibited by frozen exclusive protocol')
    from scripts import gpu_lock
    scheduler = scheduler_module()
    control = args.out.with_name(args.out.name + '.queue')
    with file_lock(QUEUE_LOCK):
        require(not args.out.exists() and not control.exists(), 'Never duplicate/retry a production writer or failed queue; retain all evidence')
        control.mkdir(parents=True)
        state = dict(schema=SCHEMA, status='waiting_for_entire_pipeline', training_run=str(args.training_run),
                     owner=args.gpu_lock_owner, production_started=False, record_video=False, extra_camera=False)
        queue_record(control, state)
        held = False
        try:
            deadline = time.monotonic() + args.wait_seconds
            while True:
                require(time.monotonic() < deadline, 'Bounded queue wait exhausted; no collection started')
                if not pipeline_finished(args.training_run):
                    time.sleep(10); continue
                try:
                    with file_lock(args.training_run / '.pipeline.lock', shared=True):
                        require(pipeline_finished(args.training_run), 'Pipeline changed during exit verification')
                        tiny = tiny_review(args.training_run)
                        require(tiny['complete_b4_updates_verified'], 'Full B4 tiny update evidence not validated: ' + str(tiny.get('reason')))
                        require(read(args.training_run / 'plan.json')['gpu_lock_owner'] != args.gpu_lock_owner, 'Never reuse training owner')
                        if gpu_lock.owner() is not None:
                            time.sleep(10); continue
                        require(not gpu_jobs(processes()), 'Foreign/live GPU jobs remain; refuse without signalling them')
                        try:
                            gpu_lock.acquire(args.gpu_lock_owner, max_wait=0)
                        except TimeoutError:
                            time.sleep(10); continue
                        held = True
                        require(not gpu_jobs(processes()), 'A foreign GPU job appeared during exclusive admission')
                        admission = scheduler.collection_resource_admission()
                        require(scheduler.resource_ready(admission, expansion=True), 'Original RAM/GPU/swap reserve denies single-worker admission')
                        scheduler.accepted_task1_expert(PROBE)  # All four strict probes, credit zero.
                        require(not args.out.exists(), 'Production output appeared while waiting; do not repeat writer')
                        state.update(status='running_serial_collection', production_started=True, admission=admission)
                        queue_record(control, state)
                        code = scheduler.main(scheduler_args(args, execute=True))
                        require(code == 0, f'Original scheduler exit={code}; retain all guards/attempts, no retry')
                        state['status'] = 'completed_serial_collection'; queue_record(control, state)
                        return 0
                except BlockingIOError:
                    time.sleep(10)  # Whole pipeline still owns its flock, even if GPU is idle.
        except BaseException as exc:
            state.update(status='halted', error=f'{type(exc).__name__}: {exc}')
            queue_record(control, state, 'failed_guard.json'); queue_record(control, state)
            raise
        finally:
            if held:
                try:
                    require(not gpu_jobs(processes()), 'Residual GPU workload: keep owner for explicit audit, never kill foreign jobs')
                except BaseException as exc:
                    queue_record(control, dict(error=str(exc), owner_retained=True), 'owner_retained_guard.json')
                    state.update(status='halted_owner_retained', error=str(exc)); queue_record(control, state)
                    raise RuntimeError('GPU owner retained for audit; residual workload was not signalled') from exc
                else:
                    gpu_lock.release(args.gpu_lock_owner)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.execute:
            return execute_serial(args)
        print(json.dumps(review(args), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'No unsafe launch: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
