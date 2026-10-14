#!/usr/bin/env python3
"""Collection-only repair coordinator (CPU preflight; execution is opt-in).

Import an independently completed task2 100-source/100-DR run, then aggregate
48 preserved task1 DR with 52 newly admitted DR. No tiny/train/eval entry point.
A versioned task1 collector and >=4 full physical probes must be hash-bound;
never hot-patch a collector already frozen in a probe or production run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.progress1_log_tasks12_collection_repair import log
from scripts import collect_isaac_zhuhai_tasks12_repair as default_collector
from scripts.collect_isaac_zhuhai_tasks12_repair import (
    PRODUCTION_QUOTAS, MINIMUM_FREE_BYTES, ONBOARD_CAMERAS, robot_binding,
    validate_manifest, write_once, sha256, check_disk, runtime_interface,
    data_contract, data_contract_hash, json_digest, STATE_SEMANTICS,
    ACTION_SEMANTICS, RESOURCE_POLICY, RECYCLE_EXIT_CODE, WORKER_RECYCLING,
    SHARED_CONTRACT_FIELDS)

GIB = 1024**3
SECOND_START_DELAY_SECONDS = 15.
COLLECTION_POLL_SECONDS = .2
TASK1_SEED_START = 974000
SNAPSHOT_FILE_LIMIT = 8 * 1024**2
SNAPSHOT_TOTAL_LIMIT = 64 * 1024**2
SNAPSHOT_SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.sh', '.md', '.txt'}
DISABLED = dict(training_authorized=False, automatic_training=False, automatic_evaluation=False)
COLLECTOR_PATTERN = r'collect_isaac_zhuhai_tasks12_repair(?:_v[1-9][0-9]*)?'
RECEIPT_FIELDS = ('manifest', 'manifest_sha256', 'successful_episodes', 'source_families',
                  'state_dim', 'action_dim', 'state_semantics', 'action_semantics', 'data_contract_sha256')


def sha(path):
    return sha256(path)


def read_json(path):
    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON evidence: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


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
    ap.add_argument('--collection-root', required=True, type=Path)
    ap.add_argument('--task2-prior-inventory', required=True, type=Path)
    ap.add_argument('--task1-prior-inventory', required=True, type=Path)
    ap.add_argument('--task2-collection-run', required=True, type=Path)
    ap.add_argument('--task1-acceptance', '--task1-admission', dest='task1_acceptance', required=True, type=Path)
    ap.add_argument('--task1-collector', type=Path, default=Path(default_collector.__file__).resolve(),
                    help='Versioned import-safe repair collector; must match the admission hash')
    ap.add_argument('--deferred-root', required=True, type=Path, help='Preserved old task1 run; never written')
    ap.add_argument('--source-root', required=True, type=Path)
    ap.add_argument('--life-mapping', choices=['body_text', 'scoring_table'], required=True)
    ap.add_argument('--collection-workers', type=int, choices=(1, 2, 3, 4), default=3)
    ap.add_argument('--source-count', type=int, choices=(100,), default=100)
    ap.add_argument('--randomized-count', type=int, choices=(100,), default=100)
    ap.add_argument('--gpu-lock-owner')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--resume', action='store_true')
    return ap


def validate_args(args):
    if (type(args.collection_workers) is not int or args.collection_workers not in (1, 2, 3, 4)):
        raise ValueError('collection-workers must be 1..4')
    if any(type(getattr(args, key)) is not int or getattr(args, key) != 100
           for key in ('source_count', 'randomized_count')):
        raise ValueError('Fixed quota: each task requires 100 source + 100 randomized, never 48 or 52 total')
    if args.life_mapping not in ('body_text', 'scoring_table'):
        raise ValueError('Unsupported frozen life mapping')


def regular_path(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError(f'Evidence/code must be a regular non-symlink file: {path}')
    return path.resolve()


def reference(path):
    path = regular_path(path)
    return dict(path=str(path), sha256=sha(path))


def verify_reference(ref):
    if not isinstance(ref, dict) or not isinstance(ref.get('sha256'), str):
        raise ValueError('Missing hash-bound evidence reference')
    actual = reference(ref['path'])
    if actual['sha256'] != ref['sha256']:
        raise ValueError(f'Hash-bound evidence changed: {actual["path"]}')
    return Path(actual['path'])


def selected_collector(args):
    path = regular_path(args.task1_collector)
    if path.parent != ROOT/'scripts' or re.fullmatch(COLLECTOR_PATTERN, path.stem) is None:
        raise ValueError('Only an explicit versioned tasks12 repair collector in scripts/ is allowed')
    module = importlib.import_module('scripts.' + path.stem)
    if Path(module.__file__).resolve() != path:
        raise ValueError('Imported collector differs from the selected file')
    for key, expected in (('PRODUCTION_QUOTAS', dict(source=100, randomized=100)),
                          ('RESOURCE_POLICY', RESOURCE_POLICY), ('WORKER_RECYCLING', WORKER_RECYCLING),
                          ('RECYCLE_EXIT_CODE', RECYCLE_EXIT_CODE), ('MINIMUM_FREE_BYTES', MINIMUM_FREE_BYTES)):
        if getattr(module, key, None) != expected:
            raise ValueError(f'Selected collector changed frozen quota/resource policy: {key}')
    for key in ('validate_manifest', '_success_rows', 'prior_rows', 'residual_families',
                'aggregate_shards', 'validate_header', 'validate_recycle_boundary', 'consecutive_failures'):
        if not callable(getattr(module, key, None)):
            raise ValueError(f'Selected collector lacks the CPU verification API: {key}')
    regular_path(module.TASK_EXPERT_PATHS['zhuhai_task1'])
    return module


def _overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def protect_output_paths(args):
    """Check before creating locks/plans, including aliases and ancestors."""
    protected = [args.source_root, args.deferred_root, args.task2_collection_run]
    for inventory in (args.task1_prior_inventory, args.task2_prior_inventory):
        protected.append(inventory)
        if inventory.is_file():
            protected.extend(Path(ref['path']).parent for ref in read_json(inventory).get('components', []))
    protected.append(args.task1_acceptance)
    if args.task1_acceptance.is_file():
        protected.extend(Path(ref['path']).parent for ref in read_json(args.task1_acceptance).get('probe_manifests', []))
    tasks = args.task2_collection_run/'tasks.json'
    if tasks.is_file():
        for spec in read_json(tasks).get('tasks', []):
            protected.extend(Path(spec[key]).parent for key in ('source_manifest', 'randomized_manifest'))
    outputs = [args.out, args.collection_root, args.collection_root/'zhuhai_task1/randomized',
               *(args.collection_root/'zhuhai_task1/randomized/shards'/f'shard_{i:02d}'
                 for i in range(args.collection_workers)),
               *(args.out/name for name in ('stage_attempts', 'collection_groups', 'source_snapshot',
                                           'plan.json', 'status.json', 'tasks.json', 'readiness.json', 'completion.json'))]
    for path in outputs:
        if any(part.is_symlink() for part in (path, *path.absolute().parents)):
            raise ValueError('Output/collection symlink aliases are forbidden')
        if any(_overlap(path.resolve(), Path(old).resolve()) for old in protected):
            raise ValueError('Output/collection overlaps preserved source, prior, probe or task2 evidence')


@contextmanager
def exclusive_lock(path, *, readonly=False):
    """Never truncate locks; borrowed v11 locks are opened read-only, not created."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Lock symlinks are forbidden')
    if readonly and not path.exists():
        yield
        return
    flags = os.O_RDONLY if readonly else os.O_RDWR | os.O_CREAT
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Live writer lock; never start a second writer: {path}') from exc
        yield
    finally:
        os.close(fd)


def _unique_rows(rows):
    for key, values in (
        ('seed', [row['seed'] for row in rows]),
        ('path', [str(Path(row['data_path']).resolve()) for row in rows]),
        ('content', [row['data_sha256'] for row in rows])):
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate physical episode {key}')


def family_integrity(source_rows, randomized_rows, *, complete=True):
    """Source is the split authority: DR cannot rename families or set new labels.

    This collection queue does not fit statistics or materialize a TRAIN split.
    Its auditable family index is the indivisible split unit for later training.
    Existing split labels, if present, must agree at both row/metadata levels.
    """
    parents = {row['family_id']: row for row in source_rows}
    if len(source_rows) != 100 or len(parents) != 100:
        raise ValueError('Exactly 100 distinct source families required')
    if len({row['family_id'] for row in randomized_rows}) != len(randomized_rows):
        raise ValueError('Duplicate randomized family')
    if complete and {row['family_id'] for row in randomized_rows} != set(parents):
        raise ValueError('Exactly one DR for every registered source family required')
    _unique_rows([*source_rows, *randomized_rows])
    def split(row):
        labels = [container['split'] for container in (row, row.get('metadata', {})) if 'split' in container]
        if any(label not in ('train', 'val', 'heldout') for label in labels) or len(set(labels)) > 1:
            raise ValueError('Conflicting/invalid family split label')
        return labels[0] if labels else None
    labels = {family: split(row) for family, row in parents.items()}
    for row in randomized_rows:
        parent = parents.get(row['family_id'])
        if (parent is None or row.get('source_parent') != parent['seed'] or
                row.get('metadata', {}).get('parent_source_seed') != parent['seed']):
            raise ValueError('Unregistered randomized source family/parent')
        label = split(row)
        if label is not None and label != labels[row['family_id']]:
            raise ValueError('Same source family may not cross split or introduce a DR-only split')
    index = [dict(family_id=family, source_seed=parents[family]['seed'], split=labels[family])
             for family in sorted(parents)]
    return dict(family_index=index, family_index_sha256=json_digest(index),
                split_unit='source family; all relatives inherit the source split',
                training_split_materialized=False)


def collection_receipt(path, task, mode, *, collector=None, **shard):
    validator = collector or default_collector
    path = regular_path(path)
    before = sha(path)
    rows = validator.validate_manifest(path, task=task, mode=mode, verify_files=True, **shard)
    _unique_rows(rows)
    if sha(path) != before:
        raise ValueError('Manifest changed during validation')
    return dict(manifest=str(path), manifest_sha256=before, successful_episodes=len(rows),
                source_families=len({r['family_id'] for r in rows}), state_dim=16, action_dim=16,
                state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS,
                data_contract_sha256=data_contract_hash())


def _check_recorded_receipt(stored, fresh):
    if not isinstance(stored, dict) or any(stored.get(key) != fresh[key] for key in RECEIPT_FIELDS):
        raise ValueError('Recorded completion receipt differs from freshly verified physical manifests')


def stage_needs_run(out, name, command, args, state, validate=None):
    if re.fullmatch(r'randomized_task1_shard[0-3]', name) is None or validate is None:
        raise ValueError('Only validated task1 randomized collection shards can run; no tiny/train/eval')
    existing = state['stages'].get(name)
    if existing and [s for s in existing['command'] if s != '--resume'] != [s for s in command if s != '--resume']:
        raise ValueError('Frozen stage command changed; use a new run directory')
    if existing and existing.get('pid') and (existing['status'] == 'running' or existing.get('process_exited') is False):
        try:
            os.kill(existing['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Previous stage PID is still alive; do not start a second writer')
    if existing and existing['status'] == 'completed':
        if validate() != existing.get('validation'):
            raise ValueError(f'{name}: completed-stage evidence changed; never reuse a status flag')
        return False
    if existing and not args.resume:
        raise RuntimeError('Existing incomplete stage requires explicit --resume after diagnosis')
    return True


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
    answer = dict(minimum_available_ram_bytes=7*GIB, minimum_free_gpu_bytes=3*GIB,
                  fourth_worker_maximum_swap_fraction=.90, maximum_swap_fraction=.95,
                  checked_unix=time.time(), read_only=True, cuda_context_created=False)
    try:
        memory, gpu = memory_info(), gpu_info()
        free = min((d['total_bytes']-d['used_bytes'] for d in gpu.get('devices', [])), default=None)
        fraction = memory['swap_used_bytes']/memory['swap_total_bytes'] if memory['swap_total_bytes'] else 0
        answer.update(memory=memory, gpu=gpu, gpu_free_bytes=free, swap_fraction=fraction)
        answer['admitted'] = resource_ready(answer, expansion=True)
        answer['fourth_admitted'] = resource_ready(answer, expansion=True, fourth=True)
    except Exception as exc:
        answer.update(admitted=False, fourth_admitted=False, error=str(exc))
    return answer


def resource_ready(admission, *, expansion=False, fourth=False):
    """Check measured reserves, not an untrusted admitted=True summary."""
    memory = admission.get('memory', {})
    values = [memory.get('available_bytes'), memory.get('total_bytes'),
              admission.get('gpu_free_bytes'), admission.get('swap_fraction')]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values):
        return False
    available, total, gpu, swap = values
    if total <= 0 or available > total or swap > 1:
        return False
    ram_floor = max(RESOURCE_POLICY['minimum_available_ram_bytes'],
                    RESOURCE_POLICY['minimum_available_ram_fraction']*total, 7*GIB if expansion else 0)
    gpu_floor = 3*GIB if expansion else RESOURCE_POLICY['minimum_free_gpu_bytes']
    return available >= ram_floor and gpu >= gpu_floor and swap < (.90 if fourth else .95)


def _command_value(command, flag):
    if command.count(flag) != 1 or command.index(flag)+1 >= len(command):
        raise ValueError(f'Exactly one {flag} required for an isolated collector')
    return command[command.index(flag)+1]


def collector_command(args, index):
    target = args.collection_root.resolve()/'zhuhai_task1/randomized/shards'/f'shard_{index:02d}'
    return [str(ROOT/'.conda-envs/behavior6/bin/python'), '-B', str(args.task1_collector.resolve()),
            '--task', 'zhuhai_task1', '--track', 'life', '--mode', 'randomized', '--count', '100',
            '--seed-start', str(TASK1_SEED_START), '--out', str(target),
            '--gpu-lock-owner', args.gpu_lock_owner, '--max-consecutive-failures', '3',
            '--shard-index', str(index), '--shard-count', str(args.collection_workers),
            '--source-manifest', str(args.source_root.resolve()/'zhuhai_task1/source/manifest.json'),
            '--prior-inventory', str(args.task1_prior_inventory.resolve()), '--life-mapping', args.life_mapping]


def _collection_jobs(out, mode, jobs, args):
    validate_args(args); protect_output_paths(args)
    expected = {('zhuhai_task1', j) for j in range(args.collection_workers)}
    if mode != 'randomized' or {(j['task'], j['shard_index']) for j in jobs} != expected or len(jobs) != len(expected):
        raise ValueError('Exactly one disjoint task1 collection job per shard required; no task2 writer')
    for job in jobs:
        index, command = job['shard_index'], job['command']
        expected_command = collector_command(args, index)
        if (type(index) is not int or job['name'] != f'randomized_task1_shard{index}' or
                command.count('--resume') > 1 or
                [x for x in command if x != '--resume'] != expected_command or
                Path(job['manifest']) != Path(_command_value(expected_command, '--out'))/'manifest.json'):
            raise ValueError('Collection-only command/output/quota/parent/owner isolation mismatch')


def resume_target_guard(job, args, collector):
    path = job['manifest']
    with exclusive_lock(path.parent/'.collection.lock', readonly=True):
        if not path.exists():
            if path.parent.exists() and any(p.name != '.collection.lock' for p in path.parent.iterdir()):
                raise ValueError('Unindexed partial output retained; diagnose before creating a writer')
            return
        if not args.resume:
            raise RuntimeError('Existing shard requires explicit --resume; preserve failure/partial evidence')
        manifest = read_json(path)
        collector.validate_header(manifest['contract'], 'zhuhai_task1', verify_files=True)
        collector._success_rows(manifest, verify_files=True)
        if (manifest.get('status') in ('stopped_consecutive_failures', 'stopped_attempt_limit') or
                collector.consecutive_failures(manifest['attempts']) >= 3 or
                (manifest.get('status') != 'complete' and
                 len(manifest['attempts']) >= manifest['requested_success_count']*10)):
            raise RuntimeError('Failure guard/attempt limit retained; no automatic reset or retry bypass')


def collection_barrier(state, mode, validators=None):
    if mode not in ('source', 'randomized') or validators is None or set(validators) != {'zhuhai_task1', 'zhuhai_task2'}:
        raise RuntimeError('Physical evidence validators required; counts/status flags cannot satisfy a barrier')
    for task, validate in validators.items():
        stage = state['stages'].get(f'{mode}_task{task[-1]}', {})
        fresh = validate()
        if (stage.get('status') != 'completed' or fresh.get('successful_episodes') != 100 or
                fresh.get('source_families') != 100):
            raise RuntimeError(f'{mode} {task} quota/validation barrier incomplete')
        _check_recorded_receipt(stage.get('validation'), fresh)


def run_collection_group(out, mode, jobs, args, state, plan):
    """Main-thread scheduler: no run_stage threads, shared .tmp or nested locks."""
    _collection_jobs(out, mode, jobs, args)
    collector = selected_collector(args)
    for job in jobs:
        resume_target_guard(job, args, collector)
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
            if active and time.monotonic()-resource_checked >= (1 if len(active)>=4 else 5):
                admission = collection_resource_admission()
                resource_checked = time.monotonic()
                if not resource_ready(admission, fourth=len(active)>=4):
                    child = max(active.values(), key=lambda c:c['started_monotonic'])
                    decision(dict(decision='runtime_resource_fallback', stage=child['job']['name'],
                                  admission=admission, next_worker_cap=max(0, len(active)-1)))
                    if child['proc'].poll() is None:
                        try:
                            os.killpg(child['proc'].pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    code = child['proc'].wait()
                    finish(child, code, cancellation='resource fallback; partial retained, disjoint shard requeued')
                    job = dict(child['job']); job['command'] = list(job['command'])
                    if '--resume' not in job['command']:
                        job['command'].append('--resume')
                    pending.insert(0, job)
                    runtime_cap = max(1, len(active))
                    if not active:
                        raise RuntimeError('Even one worker lacks known hard RAM/GPU/swap reserves; partial retained')
            if stop_requested['signal'] is not None:
                raise KeyboardInterrupt(f'collection group interrupted by signal {stop_requested["signal"]}')
            failure = None
            for child in list(active.values()):
                code = child['proc'].poll()
                if code is not None:
                    if code == RECYCLE_EXIT_CODE:
                        boundary=collector.validate_recycle_boundary(child['job']['manifest'],task=child['job']['task'])
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
                admission = collection_resource_admission()
                if not resource_ready(admission):
                    decision(dict(decision='unsafe_first_worker', admission=admission))
                    raise RuntimeError('First worker lacks known hard RAM/GPU/swap reserves')
                start(pending.pop(0))
            elif pending and allow_overlap and len(active) < runtime_cap:
                first = max(active.values(),key=lambda c:c['started_monotonic'])
                elapsed = time.monotonic()-first['started_monotonic']
                if elapsed >= SECOND_START_DELAY_SECONDS:
                    try:
                        admission = collection_resource_admission()
                    except Exception as exc:
                        admission = dict(admitted=False, error=f'{type(exc).__name__}: {exc}', reason='resource_inventory_unavailable')
                    admitted = resource_ready(admission, expansion=True, fourth=len(active)>=3)
                    decision(dict(decision='parallel_admitted' if admitted else 'serial_resource_fallback',
                        first_stage=first['job']['name'], first_elapsed_seconds=elapsed,
                        warmup_trigger='15s_startup_delay', admission=admission))
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
    for path, expected in {**plan['source_hashes'], **plan.get('evidence_hashes', {}),
                           **plan['robot_binding']['asset_sha256']}.items():
        if sha(regular_path(path)) != expected:
            raise ValueError(f'Frozen code/asset/evidence changed: {path}; use a new run')


def _strict_task1_probe(row, contract, expert):
    """Inspect full scoring and the hashed trace, not just success booleans."""
    from harness.zhuhai_task1_rules import Task1Judge, DESTINATIONS
    evidence = row['final_rule_evidence']; score = evidence['score']
    if evidence['physics_time_seconds']-row['initial_rule_evidence']['physics_time_seconds'] < 6.:
        raise ValueError('Full physical probe cannot finish before five-second grasp and stable placement')
    expected = Task1Judge().report()
    if (score.get('base_score') != 30 or score.get('base_score_max') != 30 or
            score.get('full_task_success') is not True or score.get('bins_fixed_throughout') is not True or
            score.get('simulation_tolerances') != expected['simulation_tolerances'] or
            set(score.get('items', {})) != set(DESTINATIONS) or set(score.get('lids', {})) != {'dry', 'wet'} or
            set(evidence.get('objects', {})) != set(DESTINATIONS)):
        raise ValueError('Full unchanged task1 physical score/objects/tolerances required')
    for name, destination in DESTINATIONS.items():
        item, obj = score['items'][name], evidence['objects'][name]
        if (item.get('destination') != destination or item.get('base_points') != 8 or
                any(item.get(key) is not True for key in ('grasp_awarded', 'placement_valid', 'passed_open_mouth')) or
                type(item.get('stable_seconds')) not in (int, float) or item['stable_seconds'] < 1 or
                obj.get('released') is not True or obj.get('touching_robot') is not False):
            raise ValueError('All three physical grasps and released stable placements must succeed')
    if any(lid.get('opened_awarded') is not True or lid.get('open_now') is not True or lid.get('base_points') != 3
           for lid in score['lids'].values()):
        raise ValueError('Both physical lids must stay open')
    receipt = read_json(row['expert_receipt_path'])
    info = receipt.get('expert_info', {})
    trace, commands = receipt.get('trace'), receipt.get('motion_commands')
    expert_class = 'harness.' + Path(expert['path']).stem + '.SmoothExpert'
    if (receipt.get('expert_class') != expert_class or receipt.get('plan_success') is not True or
            info.get('plan_success') is not True or info.get('full_task_success') is not True or
            info.get('failure') is not None or not isinstance(trace, list) or not trace or
            info.get('trace') != trace or any(not isinstance(t, dict) or not t.get('phase') for t in trace) or
            not isinstance(commands, list) or len(commands) != row['observation_count']-1 or
            info.get('command_count') != len(commands)):
        raise ValueError('Complete hash-bound expert trace/actions required; boolean receipt is insufficient')
    for key, value in dict(state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS,
                           data_contract_sha256=data_contract_hash(),
                           robot_contract_sha256=contract['robot_contract_sha256']).items():
        if receipt.get(key) != value:
            raise ValueError('Expert receipt native16/contract binding mismatch: ' + key)
    if any(not isinstance(action, list) or len(action) != 16 or
           any(type(v) not in (int, float) or not math.isfinite(v) for v in action) for action in commands):
        raise ValueError('Expert motion trace must contain finite native16 actions')


V2_PROBE_SCOPE = 'task1_expert_repair_v2_physical_probe_not_production'
V2_PROBE_CASES = {944025: 943064, 944026: 943064, 945024: 943061, 947000: 943003}


def _v2_probe_suite(probe, contract, plan, expert):
    """Read the new v2 probe's real suite format without importing its runtime.

    A verify_suite() boolean is not an admission. Re-read and hash-check the
    suite, exact original cases, inherited v1 dependencies and source parents.
    """
    suite_path = regular_path(probe.parent.parent/'suite_plan.json')
    suite = read_json(suite_path); bindings = suite.get('bindings', {})
    fingerprint = json_digest(bindings)
    entry = reference(ROOT/'scripts/probe_zhuhai_task1_expert_repair_v2.py')
    candidate = reference(ROOT/'harness/isaac_zhuhai_task1_expert_repair_v2.py')
    inherited = reference(ROOT/'harness/isaac_zhuhai_task1_expert_repair_v1.py')
    if (expert != candidate or bindings.get(inherited['path']) != inherited['sha256'] or
            suite.get('schema') != V2_PROBE_SCOPE or
            suite.get('required_cases') != [dict(seed=seed, parent=parent) for seed, parent in V2_PROBE_CASES.items()] or
            suite.get('minimum_strict_physical_passes') != 4 or
            suite.get('candidate_sha256') != expert['sha256'] or
            bindings.get(expert['path']) != expert['sha256'] or
            suite.get('bindings_sha256') != fingerprint or
            contract.get('probe_bindings_sha256') != fingerprint or
            plan.get('suite_bindings_sha256') != fingerprint or
            contract.get('probe_implementation') != entry or
            contract.get('allow_unvalidated_probe') is not True or
            contract.get('requested_success_count') != 1 or contract.get('max_attempts') != 1 or
            contract.get('max_consecutive_failures') != 3 or
            any(value.get('admitted_to_production') is not False or value.get('production_quota_credit') != 0
                for value in (suite, contract, plan)) or
            suite.get('record_video') is not False or suite.get('third_view') is not False):
        raise ValueError('V2 probe requires the unchanged hash-bound four-case non-production suite')
    required = {**contract['scene_dependency_sha256'], **contract['robot_binding']['asset_sha256'],
                contract['source_manifest']['path']: contract['source_manifest']['sha256'],
                contract['collection_implementation']['path']: contract['collection_implementation']['sha256'],
                entry['path']: entry['sha256']}
    if any(bindings.get(name) != digest for name, digest in required.items()):
        raise ValueError('V2 suite omitted or changed a probe/source/asset binding')
    for name, digest in bindings.items():
        verify_reference(dict(path=name, sha256=digest))
    return {**bindings, str(suite_path): sha(suite_path)}, (str(suite_path), fingerprint)


def accepted_task1_expert(path, collector=None):
    collector = collector or default_collector
    admission_ref = reference(path); value = read_json(path)
    expert = reference(collector.TASK_EXPERT_PATHS['zhuhai_task1'])
    implementation = reference(collector.__file__)
    if (value.get('status') != 'passed' or value.get('task') != 'zhuhai_task1' or
            value.get('expert') != expert or value.get('collector') != implementation):
        raise ValueError('Task1 admission must bind the selected collector and expert hashes')
    references = value.get('probe_manifests', [])
    if not isinstance(references, list) or len(references) < 4:
        raise ValueError('Task1 acceptance requires at least four completed physical probes')
    seeds, paths, rows = set(), set(), []
    hashes = {admission_ref['path']: admission_ref['sha256']}
    source_ref = None
    suite_identity = None; v2_seeds = set()
    parents = {}
    for ref in references:
        probe = verify_reference(ref)
        if str(probe) in paths:
            raise ValueError('Repeated task1 probe evidence')
        paths.add(str(probe)); manifest = read_json(probe); c = manifest['contract']
        probe_plan = probe.parent/'plan.json'
        plan = read_json(probe_plan)
        is_v2 = c.get('collection_scope') == V2_PROBE_SCOPE
        if (manifest.get('status') != 'complete' or type(manifest.get('success_count')) is not int or
                manifest['success_count'] != 1 or type(manifest.get('requested_success_count')) is not int or manifest['requested_success_count'] != 1 or
                len(manifest.get('attempts', [])) != 1 or c.get('task') != 'zhuhai_task1' or
                c.get('mode') != 'randomized' or c.get('source_probe') is not True or
                c.get('collection_scope') not in ('expert_repair_probe_not_production', V2_PROBE_SCOPE) or
                c.get('record_video') is not False or c.get('third_view') is not False or
                c.get('expert_variant') != expert or plan.get('contract') != c or
                plan.get('admitted_to_production') is not False):
            raise ValueError('Task1 probe must be complete, isolated, non-production and bound to its immutable plan')
        if is_v2:
            suite_hashes, identity = _v2_probe_suite(probe, c, plan, expert)
            if suite_identity is not None and suite_identity != identity:
                raise ValueError('V2 probes must share one frozen suite binding')
            suite_identity = identity; hashes.update(suite_hashes)
        for bound in (c['probe_implementation'], c['collection_implementation']):
            verify_reference(bound); hashes[bound['path']] = bound['sha256']
        for name, digest in {**c['scene_dependency_sha256'], **c['robot_binding']['asset_sha256']}.items():
            verify_reference(dict(path=name, sha256=digest)); hashes[name] = digest
        if source_ref is None:
            source_ref = c['source_manifest']; source = verify_reference(source_ref)
            source_rows = collector.validate_manifest(source, task='zhuhai_task1', mode='source', verify_files=True)
            parents = {r['seed']: r for r in source_rows}
            if len(source_rows) != 100 or len(parents) != 100:
                raise ValueError('Probe source must be the original completed 100-source pool')
        if c['source_manifest'] != source_ref:
            raise ValueError('Task1 probes must use the same original source pool')
        original_contract = read_json(source_ref['path'])['contract']
        for key in SHARED_CONTRACT_FIELDS:
            if is_v2 and key == 'scene_dependency_sha256':
                if any(c[key].get(name) != digest for name, digest in original_contract[key].items()):
                    raise ValueError('V2 extra dependencies must not change original scene/rules hashes')
                continue
            if key not in ('source_probe', 'allow_unvalidated_probe', 'collection_scope') and c.get(key) != original_contract.get(key):
                raise ValueError('Probe changed original scene/rules contract: ' + key)
        successes = collector._success_rows(manifest, verify_files=True)
        if len(successes) != 1 or successes[0]['seed'] in seeds:
            raise ValueError('Four independent strictly successful physical probe seeds required')
        row = successes[0]; parent = parents.get(row.get('source_parent'))
        if (parent is None or row['family_id'] != parent['family_id'] or
                row['metadata'].get('parent_source_seed') != parent['seed'] or
                plan.get('source_parent') != parent['seed'] or plan.get('seed' if is_v2 else 'failed_seed') != row['seed']):
            raise ValueError('Probe seed/source parent provenance mismatch')
        if is_v2:
            if V2_PROBE_CASES.get(row['seed']) != parent['seed'] or hashes.get(parent['data_path']) != parent['data_sha256']:
                raise ValueError('V2 probe original seed/parent or parent data binding changed')
            v2_seeds.add(row['seed'])
        _strict_task1_probe(row, c, expert)
        seeds.add(row['seed']); rows.append(row)
        for evidence_path in (probe, probe_plan, Path(row['data_path']), Path(row['expert_receipt_path'])):
            hashes[str(regular_path(evidence_path))] = sha(evidence_path)
    if suite_identity is not None and (v2_seeds != set(V2_PROBE_CASES) or v2_seeds != seeds):
        raise ValueError('All four unchanged v2 suite cases required; no mixing legacy probes')
    _unique_rows(rows)
    _unique_rows([*parents.values(), *rows])
    hashes[source_ref['path']] = source_ref['sha256']
    return dict(**admission_ref, expert=expert, collector=implementation, probe_manifests=references,
                seeds=sorted(seeds), probe_success_count=len(seeds), source_manifest=source_ref, evidence_hashes=hashes,
                probe_episodes=[{key: row[key] for key in ('seed', 'data_path', 'data_sha256')} for row in rows])


def _task2_validator(implementation):
    from scripts import collect_isaac_zhuhai_task2_repair as previous
    allowed = {str(Path(m.__file__).resolve()): m for m in (default_collector, previous)}
    validator = allowed.get(implementation.get('path'))
    if validator is None or reference(validator.__file__) != implementation:
        raise ValueError('Task2 imported collector changed or is not allowed')
    return validator


def completed_task2_import(run):
    """Read-only import: all hashes, full manifests and the completed transaction."""
    run = Path(run).resolve()
    with exclusive_lock(run/'.pipeline.lock', readonly=True):
        plan_path, state_path = run/'plan.json', run/'status.json'
        plan, state = read_json(plan_path), read_json(state_path)
        completion = read_json(run/'completion.json')
        if (state.get('status') != 'completed_collection_only' or
                state.get('plan_sha256') != sha(plan_path) or state.get('active_stage') is not None or
                state.get('active_stages') or
                any(value.get(key) is not False for value in (plan, state, completion) for key in DISABLED) or
                plan.get('source_count') != 100 or plan.get('randomized_count') != 100 or
                plan.get('stages') != ['source_task2', 'randomized_task2'] or
                completion.get('status') != 'completed_collection_only' or completion.get('task') != 'zhuhai_task2'):
            raise ValueError('Task2 collection-only run must genuinely finish before task1 workers start')
        for name, stage in state.get('stages', {}).items():
            if (re.fullmatch(r'(?:source_task2|randomized_task2(?:_shard[0-3])?)', name) is None or
                    stage.get('status') != 'completed' or stage.get('process_exited') is False):
                raise ValueError('Imported task2 run contains incomplete or non-collection stages')
        spec = read_json(run/'tasks.json').get('tasks', [])
        if len(spec) != 1 or spec[0].get('task') != 'zhuhai_task2':
            raise ValueError('Task2 import must contain exactly task2')
        source, randomized = (regular_path(spec[0][key]) for key in ('source_manifest', 'randomized_manifest'))
        contract = read_json(randomized)['contract']
        validator = _task2_validator(contract['collection_implementation'])
        if contract['source_manifest']['path'] != str(source) or contract['source_manifest']['sha256'] != sha(source):
            raise ValueError('Task2 tasks.json source differs from the randomized parent')
        source_receipt = collection_receipt(source, 'zhuhai_task2', 'source', collector=validator)
        dr_receipt = collection_receipt(randomized, 'zhuhai_task2', 'randomized', collector=validator)
        for name, receipt in (('source_task2', source_receipt), ('randomized_task2', dr_receipt)):
            _check_recorded_receipt(state.get('stages', {}).get(name, {}).get('validation'), receipt)
            if receipt['successful_episodes'] != 100 or receipt['source_families'] != 100:
                raise ValueError('Task2 import requires 100 source plus 100 DR')
        _check_recorded_receipt(completion.get('randomized_validation'), dr_receipt)
        if completion.get('source_successful_episodes') != 100 or completion.get('randomized_successful_episodes') != 100:
            raise ValueError('Task2 completion quota mismatch')
        family = family_integrity(validator.validate_manifest(source, task='zhuhai_task2', mode='source'),
                                  validator.validate_manifest(randomized, task='zhuhai_task2', mode='randomized'))
        hashes = {str(p): sha(p) for p in (plan_path, state_path, run/'tasks.json', run/'completion.json', source, randomized)}
        return dict(run=str(run), source_manifest=str(source), randomized_manifest=str(randomized),
                    source_manifest_sha256=sha(source), randomized_manifest_sha256=sha(randomized),
                    source_validation=source_receipt, randomized_validation=dr_receipt,
                    successful_source_episodes=100, successful_randomized_episodes=100,
                    family_integrity=family, evidence_hashes=hashes, original_evidence_modified=False)


def task1_prior_import(args, collector=None):
    collector = collector or selected_collector(args)
    source = args.source_root.resolve()/'zhuhai_task1/source/manifest.json'
    original = collector.validate_manifest(source, task='zhuhai_task1', mode='source', verify_files=True)
    previous = collector.prior_rows(args.task1_prior_inventory, verify_files=True)
    if len(original) != 100 or len(previous) != 48:
        raise ValueError('Task1 requires 100 original sources and exactly 48 preserved DR')
    family = family_integrity(original, previous, complete=False)
    source_contract = read_json(source)['contract']
    if source_contract.get('life_mapping') != args.life_mapping:
        raise ValueError('Task1 original life mapping may not change')
    binding = {key: source_contract.get(key) for key in SHARED_CONTRACT_FIELDS}
    binding.update(source_manifest=dict(path=str(source), sha256=sha(source), engine=source_contract['engine']),
        prior_inventory=reference(args.task1_prior_inventory),
        collection_sharding=dict(shard_count=args.collection_workers,
            family_assignment='source seeds ascending; index modulo shard_count', global_randomized_count=100,
            randomized_per_family=1, disjoint_seed_stride=2000 if args.collection_workers==1 else 1000))
    remaining = [collector.residual_families(binding, i, args.collection_workers, verify_files=True)
                 for i in range(args.collection_workers)]
    admitted = {r['family_id'] for r in previous}
    expected = [r for r in sorted(original, key=lambda r:r['seed']) if r['family_id'] not in admitted]
    flat = [r for rows in remaining for r in rows]
    if len(flat) != 52 or sorted(flat, key=lambda r:r['seed']) != expected:
        raise ValueError('Task1 must retain exactly 52 disjoint remaining families')
    for i, rows in enumerate(remaining):
        assigned = [r for j, r in enumerate(sorted(original, key=lambda r:r['seed']))
                    if j % args.collection_workers == i and r['family_id'] not in admitted]
        if rows != assigned:
            raise ValueError('Task1 residual family shard assignment changed')
    return dict(**reference(args.task1_prior_inventory), source_manifest=binding['source_manifest'],
                successful_episodes=48, new_successful_episode_target=52,
                residual_shard_targets=list(map(len, remaining)), family_integrity=family,
                original_success_evidence_unchanged=True)


def expert_dependency_hashes(collector):
    """Bind expert-only inheritance without changing original source scene fields."""
    expert = reference(collector.TASK_EXPERT_PATHS['zhuhai_task1'])
    expected = {expert['path']: expert['sha256']}
    if Path(expert['path']).name == 'isaac_zhuhai_task1_expert_repair_v2.py':
        inherited = reference(ROOT/'harness/isaac_zhuhai_task1_expert_repair_v1.py')
        expected[inherited['path']] = inherited['sha256']
    bind = getattr(collector, 'expert_dependency_binding', None)
    if bind is not None and bind('zhuhai_task1') != expected:
        raise ValueError('Selected collector expert-only dependency binding differs from frozen v1/v2 files')
    return expected


def build_plan(args):
    validate_args(args); protect_output_paths(args)
    # Deliberately validate task2 BEFORE selecting/loading any future collector.
    task2 = completed_task2_import(args.task2_collection_run)
    collector = selected_collector(args)
    acceptance = accepted_task1_expert(args.task1_acceptance, collector)
    task1 = task1_prior_import(args, collector)
    if acceptance['source_manifest'] != task1['source_manifest']:
        raise ValueError('Task1 admission uses a different source pool')
    task2_contract = read_json(task2['randomized_manifest'])['contract']
    if (task2_contract.get('life_mapping') != args.life_mapping or
            task2_contract.get('prior_inventory') != reference(args.task2_prior_inventory)):
        raise ValueError('Task2 imported life mapping/prior inventory differs from the selected handoff')
    binding, interface = robot_binding(), runtime_interface()
    files = [Path(__file__).resolve(), Path(collector.__file__).resolve(), Path(default_collector.__file__).resolve(),
             Path(acceptance['expert']['path']), ROOT/'scripts/collect_isaac_zhuhai.py',
             ROOT/'scripts/collect_zhuhai_data.py', ROOT/'scripts/collect_isaac_zhuhai_100_parallel.py',
             ROOT/'scripts/collect_isaac_zhuhai_recycle.py', ROOT/'scripts/zhuhai_resource_watch.py',
             ROOT/'scripts/gpu_lock.py', ROOT/'scripts/progress1_log_tasks12_collection_repair.py']
    source_contract = read_json(task1['source_manifest']['path'])['contract']
    source_hashes = {str(path): sha(path) for path in files}
    source_hashes.update(source_contract['scene_dependency_sha256'])
    expert_dependencies = expert_dependency_hashes(collector)
    source_hashes.update(expert_dependencies)  # Included in immutable source_snapshot, not just admission JSON.
    evidence = {**task2['evidence_hashes'], **acceptance['evidence_hashes'],
                task1['path']: task1['sha256'], task1['source_manifest']['path']: task1['source_manifest']['sha256'],
                str(args.task2_prior_inventory.resolve()): sha(args.task2_prior_inventory)}
    plan = dict(schema='zhuhai-native16-tasks12-repaired-collection-only-v2', **DISABLED,
        source_count=100, randomized_count=100, total_per_task=200, total_all_tasks=400,
        current_run_successful_episode_target=400, training_tasks=[], collection_tasks=['zhuhai_task1', 'zhuhai_task2'],
        collection_root=str(args.collection_root.resolve()), task1_collector=reference(collector.__file__),
        expert_dependency_sha256=expert_dependencies,
        collection_workers=args.collection_workers, randomized_shards_per_task=args.collection_workers,
        task2_collection_import=task2, task1_expert_acceptance=acceptance,
        prior_DR_import={'zhuhai_task1': task1, 'zhuhai_task2': reference(args.task2_prior_inventory)},
        stages=['source_task1', 'source_task2', 'randomized_task2', 'randomized_task1'],
        life_mapping=args.life_mapping, worker_recycling=dict(WORKER_RECYCLING),
        collection_scheduling=dict(max_workers=args.collection_workers, same_external_gpu_lock_owner=True,
            second_start_delay_seconds=SECOND_START_DELAY_SECONDS, minimum_available_ram_bytes=7*GIB,
            minimum_free_gpu_bytes=3*GIB, resource_policy=dict(RESOURCE_POLICY),
            failure='halt; SIGTERM/reap only owned children; preserve failed attempts, guards and partials'),
        recording=dict(video=False, third_view=False, camera_names=list(ONBOARD_CAMERAS), onboard_sensor_archive=True),
        robot_binding=binding, runtime_interface=interface, source_hashes=source_hashes, evidence_hashes=evidence,
        scene_ready=binding['assets_ready'] and interface['ready'],
        family_split=dict(unit='source family', existing_labels_preserved=True, materialized=False,
                          preparation_requires_separate_training_authorization=True),
        next_action='Both datasets complete; wait for a separate user training instruction')
    verify_plan_sources(plan)
    return plan


def _pipeline(args):
    disabled = dict(DISABLED)
    task2_state = args.task2_collection_run/'status.json'
    waiting = []
    if not task2_state.is_file() or read_json(task2_state).get('status') != 'completed_collection_only':
        waiting.append('task2_collection')
    if not args.task1_acceptance.is_file():
        waiting.append('task1_physical_expert_acceptance')
    if not args.task1_collector.is_file():
        waiting.append('task1_versioned_collector')
    if waiting:
        readiness = dict(status='waiting_collection_prerequisites', waiting_for=waiting, plan_frozen=False, **disabled)
        atomic(args.out/'readiness.json', readiness); print(json.dumps(readiness))
        return 3 if args.execute else 0
    plan = build_plan(args); plan_path = args.out/'plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Frozen collection plan changed; use a new run')
    write_once(plan_path, plan, identical_ok=True)
    if not args.execute:
        print(json.dumps(dict(status='cpu_collection_plan_ready', plan=plan), ensure_ascii=False))
        return 0
    if not plan['scene_ready']:
        raise RuntimeError('Static assets/interface not ready; never start physical collection')
    verify_owner(args.gpu_lock_owner); verify_plan_sources(plan)
    args.collection_root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.collection_root/'.collection_repair.lock'):
        return execute_collection(args, plan, plan_path)


def execute_collection(args, plan, plan_path):
    """Reached only with both local writer locks and matching external owner."""
    state_path = args.out/'status.json'
    if state_path.exists() and not args.resume:
        raise RuntimeError('Existing collection requires explicit --resume')
    state = read_json(state_path) if state_path.exists() else dict(
        status='ready', active_stage=None, stages={}, plan_sha256=sha(plan_path), **DISABLED)
    if state.get('plan_sha256') != sha(plan_path) or any(state.get(key) is not False for key in DISABLED):
        raise ValueError('Collection status does not match frozen collection-only plan')
    if any(re.fullmatch(r'(?:source_task[12]|randomized_task[12]|randomized_task1_shard[0-3])', name) is None
           for name in state['stages']):
        raise ValueError('Non-collection stages forbidden; never tiny/train/eval')
    collector = selected_collector(args)
    borrowed = completed_task2_import(args.task2_collection_run)
    if borrowed != plan['task2_collection_import']:
        raise ValueError('Imported task2 evidence changed since preflight')
    state['source_snapshot'] = snapshot_sources(args.out, plan)
    specs = [dict(task='zhuhai_task1', source_manifest=str(args.source_root.resolve()/'zhuhai_task1/source/manifest.json'),
                  randomized_manifest=str(args.collection_root/'zhuhai_task1/randomized/manifest.json')),
             dict(task='zhuhai_task2', source_manifest=borrowed['source_manifest'], randomized_manifest=borrowed['randomized_manifest'])]
    write_once(args.out/'tasks.json', dict(tasks=specs), identical_ok=True)
    source_validators = {s['task']: (lambda s=s: collection_receipt(Path(s['source_manifest']), s['task'], 'source')) for s in specs}
    for spec in specs:
        task = spec['task']; receipt = source_validators[task]()
        name = 'source_task'+task[-1]; previous = state['stages'].get(name)
        if previous and previous.get('validation') != receipt:
            raise ValueError('Original source evidence changed')
        state['stages'][name] = dict(status='completed', imported=True, validation=receipt)
    state['stages']['randomized_task2'] = dict(status='completed', imported=True, validation=borrowed['randomized_validation'])
    atomic(state_path, state); collection_barrier(state, 'source', source_validators)
    jobs = []
    for index in range(args.collection_workers):
        command = collector_command(args, index); target = Path(_command_value(command, '--out'))
        if (target/'manifest.json').exists():
            if not args.resume:
                raise RuntimeError('Existing task1 shard requires explicit --resume')
            command.append('--resume')
        jobs.append(dict(name=f'randomized_task1_shard{index}', task='zhuhai_task1', shard_index=index,
            command=command, manifest=target/'manifest.json', validate=lambda target=target, index=index:
                collection_receipt(target/'manifest.json', 'zhuhai_task1', 'randomized', collector=collector,
                                   shard_index=index, shard_count=args.collection_workers)))
    try:
        run_collection_group(args.out, 'randomized', jobs, args, state, plan)
        components = [j['manifest'] for j in jobs]
        target = Path(specs[0]['randomized_manifest'])
        # Validate every shard and the combined family/data set BEFORE publishing.
        new_rows = []
        for index, path in enumerate(components):
            new_rows.extend(collector.validate_manifest(path, task='zhuhai_task1', mode='randomized',
                                                        shard_index=index, shard_count=args.collection_workers))
        old_rows = collector.prior_rows(args.task1_prior_inventory, verify_files=True)
        if len(old_rows) != 48 or len(new_rows) != 52:
            raise ValueError('Final task1 quota requires exactly old48 + new52, not shard counts alone')
        source_rows = collector.validate_manifest(specs[0]['source_manifest'], task='zhuhai_task1', mode='source')
        family = family_integrity(source_rows, [*old_rows, *new_rows])
        _unique_rows([*source_rows, *old_rows, *new_rows, *plan['task1_expert_acceptance']['probe_episodes']])
        collector.aggregate_shards(target, components, task='zhuhai_task1')
        dr_validators = {
            'zhuhai_task1': lambda: collection_receipt(target, 'zhuhai_task1', 'randomized', collector=collector),
            'zhuhai_task2': lambda: completed_task2_import(args.task2_collection_run)['randomized_validation']}
        state['stages']['randomized_task1'] = dict(status='completed', validation=dr_validators['zhuhai_task1'](),
                                                  family_integrity=family)
        collection_barrier(state, 'randomized', dr_validators)
        verify_plan_sources(plan)
        state.update(status='completed_collection_only', active_stage=None, active_stages=[], completed_unix=time.time(), **DISABLED)
        atomic(state_path, state)
        write_once(args.out/'completion.json', dict(status='completed_collection_only', tasks=specs,
            source_count_per_task=100, randomized_count_per_task=100, total_successful_episodes=400, **DISABLED), identical_ok=True)
        log('两任务各100源+100DR严格验收；无tiny、训练或评测', actor='tasks12_collection_only')
        return 0
    except BaseException as exc:
        state.update(status='halted_interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'halted_stage_failure',
                     error=f'{type(exc).__name__}: {exc}', active_stage=None, **DISABLED)
        atomic(state_path, state)
        raise


def pipeline(args):
    validate_args(args); protect_output_paths(args)
    args.out = args.out.resolve(); args.collection_root = args.collection_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.out/'.pipeline.lock'):
        return _pipeline(args)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return pipeline(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'pipeline halted: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
