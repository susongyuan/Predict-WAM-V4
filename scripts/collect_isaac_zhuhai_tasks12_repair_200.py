#!/usr/bin/env python3
"""Independent task1 100-source + 200-DR contract over frozen v2 physics.

Every source family has exactly two DR relatives. The original 48 success
archives stay at their original paths; 152 new slots use a fresh seed range.
Only the quota-sensitive logic is replaced in a private function namespace.
The imported frozen module is never mutated. No source collection or video.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import FunctionType, ModuleType
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_isaac_zhuhai_tasks12_repair_v2 as frozen


def isolate_protocol(module, name, file):
    """Clone module-owned functions into private globals, preserving frozen code."""
    result = ModuleType(name)
    result.__dict__.update(module.__dict__)
    result.__name__, result.__file__ = name, str(Path(file).resolve())
    for key, value in module.__dict__.items():
        if isinstance(value, FunctionType) and value.__globals__ is module.__dict__:
            clone = FunctionType(value.__code__, result.__dict__, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__, clone.__annotations__ = value.__kwdefaults__, value.__annotations__
            result.__dict__[key] = clone
    return result


_p = isolate_protocol(frozen, __name__ + '._physics', __file__)
PRODUCTION_QUOTAS = dict(source=100, randomized=200)
TOTAL_PER_TASK, TOTAL_ALL_TASKS = 300, 500  # task2 remains its completed 100+100.
SEED_STRIDE = 4000  # enough for the one-worker 152 * 10 bounded attempts.
CONTRACT_VERSION = 'task1-two-dr-per-source-family-v1'
for _key in ('RESOURCE_POLICY', 'WORKER_RECYCLING', 'RECYCLE_EXIT_CODE', 'MINIMUM_FREE_BYTES',
             'TASK_EXPERT_PATHS', 'SHARED_CONTRACT_FIELDS', 'ONBOARD_CAMERAS', 'ENGINE',
             'source_protocol', 'STATE_SEMANTICS', 'ACTION_SEMANTICS'):
    globals()[_key] = getattr(frozen, _key)


def inherited_binding():
    return dict(path=str(Path(frozen.__file__).resolve()), sha256=frozen.sha256(frozen.__file__))


def sharding(count):
    if type(count) is not int or count not in (1, 2, 3, 4):
        raise ValueError('Collection workers must be 1..4')
    return dict(shard_count=count, family_assignment='source seeds ascending; index modulo shard_count',
                global_randomized_count=200, randomized_per_family=2, disjoint_seed_stride=SEED_STRIDE)


def contract(args):
    if args.task != 'zhuhai_task1' or args.mode != 'randomized' or args.count != 200 or args.source_probe:
        raise ValueError('Independent task1 DR200 contract imports unchanged 100 source; source/probe collection forbidden')
    result = _original_contract(args)
    result.update(collection_contract_version=CONTRACT_VERSION, inherited_collection_implementation=inherited_binding(),
        production_quota=dict(PRODUCTION_QUOTAS, total_per_task=300, total_all_tasks=500),
        family_relation='independent PhysX rollouts; exactly two successful DR relatives per registered source family',
        collection_sharding=sharding(args.shard_count))
    source = json.loads(args.source_manifest.read_text())
    inventory = json.loads(args.prior_inventory.read_text())
    reserved_stop = args.seed_start + args.shard_count * SEED_STRIDE
    attempted = [r['seed'] for r in source['attempts']]
    for ref in inventory['components']:
        attempted.extend(r['seed'] for r in json.loads(Path(ref['path']).read_text())['attempts'])
    if any(args.seed_start <= seed < reserved_stop for seed in attempted):
        raise ValueError('New DR seed reservation overlaps preserved source/prior attempts')
    return result


def validate_header(c, task, verify_files=True):
    if (task != 'zhuhai_task1' or c.get('collection_contract_version') != CONTRACT_VERSION or
            c.get('inherited_collection_implementation') != inherited_binding() or
            c.get('collection_implementation') != dict(path=str(Path(__file__).resolve()), sha256=frozen.sha256(__file__)) or
            c.get('production_quota') != dict(PRODUCTION_QUOTAS, total_per_task=300, total_all_tasks=500) or
            c.get('requested_success_count') != 200 or c.get('max_attempts') != 2000 or
            c.get('source_quota_transition') != dict(from_quota=source_protocol.PRODUCTION_QUOTAS,
                                                  to_quota=PRODUCTION_QUOTAS, source_relabelled=False)):
        raise ValueError('DR200 version/quota/inherited implementation changed')
    count = c.get('collection_sharding', {}).get('shard_count')
    if c.get('collection_sharding') != sharding(count):
        raise ValueError('DR200 deterministic family sharding changed')
    # Reuse every original physical/resource/safety/version predicate. Only the
    # five explicit quota-sensitive values get a private legacy validation view.
    legacy = deepcopy(c)
    legacy.update(collection_implementation=inherited_binding(), requested_success_count=100, max_attempts=1000,
        production_quota=dict(source=100, randomized=100, total_per_task=200, total_all_tasks=400),
        source_quota_transition=dict(from_quota=source_protocol.PRODUCTION_QUOTAS,
                                     to_quota=frozen.PRODUCTION_QUOTAS, source_relabelled=False),
        collection_sharding=dict(shard_count=count,
            family_assignment='source seeds ascending; index modulo shard_count', global_randomized_count=100,
            randomized_per_family=1, disjoint_seed_stride=2000 if count == 1 else 1000))
    frozen.validate_header(legacy, task, verify_files)


def residual_families(c, shard_index, shard_count, verify_files=True):
    if (type(shard_index) is not int or not 0 <= shard_index < shard_count or
            c['collection_sharding'] != sharding(shard_count)):
        raise ValueError('Invalid DR200 residual shard')
    ref = c['source_manifest']
    if frozen.sha256(ref['path']) != ref['sha256']:
        raise ValueError('Original source changed')
    parents = source_protocol.validate_manifest(ref['path'], task=c['task'], mode='source', verify_files=verify_files)
    if len(parents) != 100 or len({r['family_id'] for r in parents}) != 100:
        raise ValueError('Exactly 100 immutable source families required')
    source = json.loads(Path(ref['path']).read_text())['contract']
    if any(c.get(k) != source.get(k) for k in SHARED_CONTRACT_FIELDS):
        raise ValueError('Original source scene/rules/data contract changed')
    prior = c['prior_inventory']
    if frozen.sha256(prior['path']) != prior['sha256']:
        raise ValueError('Original prior inventory changed')
    admitted = frozen.prior_rows(prior['path'], verify_files)
    registered = {r['family_id']: r for r in parents}
    counts = Counter()
    for row in admitted:
        parent = registered.get(row['family_id'])
        if (parent is None or row.get('source_parent') != parent['seed'] or
                row['metadata'].get('parent_source_seed') != parent['seed'] or
                row['metadata']['contract'].get('source_manifest') != ref or
                any(row['metadata']['contract'].get(k) != c.get(k) for k in SHARED_CONTRACT_FIELDS)):
            raise ValueError('Preserved DR source family/protocol binding changed')
        counts[row['family_id']] += 1
    if any(n > 2 for n in counts.values()):
        raise ValueError('Prior family exceeds the fixed two-DR quota')
    return [r for j, r in enumerate(sorted(parents, key=lambda r: r['seed']))
            if j % shard_count == shard_index for _ in range(2 - counts[r['family_id']])]


def next_parent(families, attempts):
    limits, completed = Counter(r['family_id'] for r in families), Counter()
    for row in attempts:
        if row.get('status') == 'success':
            if row.get('family_id') not in limits:
                raise ValueError('Unregistered successful residual family')
            completed[row['family_id']] += 1
    if any(completed[f] > n for f, n in limits.items()):
        raise ValueError('Residual family success quota exceeded; no extra relatives')
    available = [r for r in families if completed[r['family_id']] < limits[r['family_id']]]
    if not available:
        raise ValueError('No remaining family slot; global count must already be complete')
    return min(available, key=lambda r: completed[r['family_id']])


def _family_rows(rows, parents, *, complete):
    limits = Counter(r['family_id'] for r in parents)
    counts = Counter(r['family_id'] for r in rows)
    registered = {r['family_id']: r['seed'] for r in parents}
    if (set(counts) - set(limits) or any(n > limits[f] for f, n in counts.items()) or
            (complete and counts != limits) or any(r.get('source_parent') != registered[r['family_id']]
                                                  for r in rows)):
        raise ValueError('DR200 per-family residual coverage/parent mismatch')


def validate_manifest(path, *, task, mode, count=None, families=None, verify_files=True,
                      shard_index=None, shard_count=None):
    if mode == 'source':
        return source_protocol.validate_manifest(path, task=task, mode=mode, count=count,
                                                 families=families, verify_files=verify_files)
    m = json.loads(Path(path).read_text()); c = m['contract']
    validate_header(c, task, verify_files)
    if m.get('status') != 'complete':
        raise ValueError('Complete physical residual quota required')
    n, stride = c['collection_sharding']['shard_count'], SEED_STRIDE
    if shard_index is not None:
        if shard_count != n or type(shard_index) is not int or not 0 <= shard_index < n:
            raise ValueError('DR200 shard mismatch')
        parents = residual_families(c, shard_index, n, verify_files)
        rows = frozen._success_rows(m, verify_files=verify_files)
        if m['requested_success_count'] != len(parents) or m['success_count'] != len(rows) or len(rows) != len(parents):
            raise ValueError('Incomplete residual slot quota')
        _family_rows(rows, parents, complete=True)
        begin = c['seed_start'] + shard_index * stride
        if any(not begin <= r['seed'] < begin + stride for r in m['attempts']):
            raise ValueError('Attempt seed escaped its disjoint reservation')
        return rows
    if count not in (None, 200):
        raise ValueError('Final DR quota is exactly 200')
    refs = m.get('shard_manifests', [])
    if len(refs) != n or {r['index'] for r in refs} != set(range(n)):
        raise ValueError('DR200 component inventory incomplete')
    combined = list(frozen.prior_rows(c['prior_inventory']['path'], verify_files))
    for ref in refs:
        if (frozen.sha256(ref['path']) != ref['sha256'] or
                json.loads(Path(ref['path']).read_text())['contract'] != c):
            raise ValueError('Residual component changed')
        combined.extend(validate_manifest(ref['path'], task=task, mode=mode, verify_files=verify_files,
                                          shard_index=ref['index'], shard_count=n))
    rows = sorted(combined, key=lambda r: r['seed'])
    if (rows != [r for r in m['attempts'] if r['status'] == 'success'] or
            m['success_count'] != 200 or m['requested_success_count'] != 200 or len(rows) != 200):
        raise ValueError('Final quota requires preserved48 + new152 = 200')
    parents = source_protocol.validate_manifest(c['source_manifest']['path'], task=task, mode='source', verify_files=verify_files)
    _family_rows(rows, [r for r in parents for _ in range(2)], complete=True)
    for key in ('seed', 'data_path', 'data_sha256'):
        if len({r[key] for r in rows}) != 200:
            raise ValueError('Repeated DR physical evidence')
    source = json.loads(Path(c['source_manifest']['path']).read_text())
    if ({r['seed'] for r in rows} & {r['seed'] for r in source['attempts']} or
            {r['data_sha256'] for r in rows} & {r['data_sha256'] for r in parents}):
        raise ValueError('DR reused original source')
    return rows


def aggregate_shards(path, components, *, task):
    refs, attempts, c = [], [], None
    for index, component in enumerate(components):
        validate_manifest(component, task=task, mode='randomized', shard_index=index, shard_count=len(components))
        m = json.loads(Path(component).read_text())
        c = m['contract'] if c is None else c
        if c != m['contract']:
            raise ValueError('Residual component contracts differ')
        refs.append(dict(index=index, path=str(Path(component).resolve()), sha256=frozen.sha256(component)))
        attempts.extend(m['attempts'])
    for ref in json.loads(Path(c['prior_inventory']['path']).read_text())['components']:
        attempts.extend(json.loads(Path(ref['path']).read_text())['attempts'])
    value = dict(contract=c, status='complete', requested_success_count=200, success_count=200,
        attempts=sorted(attempts, key=lambda r: r['seed']), shard_manifests=refs,
        prior_inventory=c['prior_inventory'], mixed_expert_versions_explicit=True)
    frozen.write_once(path, value, identical_ok=True)
    validate_manifest(path, task=task, mode='randomized')
    return Path(path)


def validate_recycle_boundary(path, *, task):
    m = json.loads(Path(path).read_text()); c = m['contract']; validate_header(c, task)
    if (m.get('status') != 'worker_recycle_boundary' or frozen.consecutive_failures(m['attempts']) >= 3 or
            not 0 <= m['success_count'] < m['requested_success_count'] or
            any(r['status'] == 'running' for r in m['attempts'])):
        raise ValueError('Unsafe recycle boundary; failure/partial evidence retained')
    rows = frozen._success_rows(m, verify_files=True)
    n, begin = c['collection_sharding']['shard_count'], c['seed_start']
    slots = {(r['seed'] - begin) // SEED_STRIDE for r in m['attempts']}
    if len(slots) != 1 or not slots <= set(range(n)):
        raise ValueError('Recycle attempts cross reserved seed ranges')
    parents = residual_families(c, slots.pop(), n)
    if (m['requested_success_count'] != len(parents) or m['success_count'] != len(rows) or
            len(m['attempts']) >= len(parents) * 10):
        raise ValueError('Recycle quota/attempt limit changed')
    _family_rows(rows, parents, complete=False)
    registered = {r['family_id']: r for r in parents}
    for row in m['attempts']:
        parent = registered.get(row.get('family_id')); md = row.get('metadata', {})
        if (parent is None or row.get('source_parent') != parent['seed'] or
                row.get('status') not in ('success', 'failed', 'interrupted') or
                md.get('seed') != row['seed'] or md.get('family_id') != row['family_id'] or
                md.get('parent_source_seed') != parent['seed'] or md.get('contract') != c):
            raise ValueError('Recycle family/provenance mismatch')
    return dict(successful_episodes=len(rows), requested_local_quota=m['requested_success_count'],
                quota_complete=False, manifest_sha256=frozen.sha256(path), completed_attempts=len(m['attempts']))


_original_contract = _p.contract
_original_owner = _p.verify_owner

def _guarded_owner(expected):
    if frozen.sha256(frozen.__file__) != _p.inherited_hash:
        raise RuntimeError('Frozen inherited collector changed during execution')
    return _original_owner(expected)


# Private globals ensure inherited collect/main invoke new quota methods only.
_p.__dict__.update(PRODUCTION_QUOTAS=PRODUCTION_QUOTAS, TOTAL_PER_TASK=TOTAL_PER_TASK,
    TOTAL_ALL_TASKS=TOTAL_ALL_TASKS, inherited_hash=inherited_binding()['sha256'],
    contract=contract, residual_families=residual_families, next_parent=next_parent,
    validate_header=validate_header, validate_manifest=validate_manifest,
    aggregate_shards=aggregate_shards, validate_recycle_boundary=validate_recycle_boundary,
    verify_owner=_guarded_owner)
for _key in ('_success_rows', 'prior_rows', 'consecutive_failures', 'check_disk',
             'write_once', 'sha256', 'runtime_interface', 'robot_binding', 'expert_dependency_binding'):
    globals()[_key] = getattr(_p, _key)


def parser():
    result = frozen.parser()
    result.description = __doc__
    result.add_argument('--execute', action='store_true', help='Explicit physical collection; default is CPU plan only')
    result.add_argument('--task1-acceptance', type=Path, help='Required strict four-v2-probe receipt for physical execution')
    return result


def main():
    # Main's two literal legacy strides are replaced with explicit DR200 logic.
    args = parser().parse_args()
    if (args.task != 'zhuhai_task1' or args.mode != 'randomized' or args.count != 200 or args.source_probe or
            args.seed_start < 0 or args.max_consecutive_failures != 3 or args.max_attempts is not None or
            not 0 <= args.shard_index < args.shard_count):
        raise ValueError('Fixed task1 100-source/200-DR production only; original failure guard retained')
    args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.collection.lock').open('a') as lock:
        _p.fcntl.flock(lock, _p.fcntl.LOCK_EX | _p.fcntl.LOCK_NB)
        _p.source_families(args); c = contract(args)
        if (args.out/'manifest.json').exists() and not args.resume:
            raise FileExistsError('Existing writer requires explicit resume; retain old evidence')
        families = residual_families(c, args.shard_index, args.shard_count)
        write_once(args.out/'plan.json', dict(contract=c, global_requested_count=200,
            local_requested_count=len(families), local_seed_start=args.seed_start + args.shard_index * SEED_STRIDE,
            local_max_attempts=len(families) * 10, shard_index=args.shard_index,
            physical_collection_started=False), identical_ok=True)
        if args.plan_only or not args.execute:
            return 0
        if not c['robot_binding']['assets_ready'] or not c['runtime_interface']['ready']:
            raise RuntimeError('Assets/interface not ready; no physical launch')
        if args.task1_acceptance is None:
            raise ValueError('Physical collection requires strict four-v2-probe acceptance')
        from scripts.run_zhuhai_tasks12_collection_repair_200 import accepted_task1_expert
        proof = accepted_task1_expert(args.task1_acceptance)
        if proof['source_manifest'] != c['source_manifest']:
            raise ValueError('Probe acceptance bound to a different original source pool')
        _guarded_owner(args.gpu_lock_owner)
        args.count = len(families); args.seed_start += args.shard_index * SEED_STRIDE
        args.max_attempts = args.count * 10
        return _p.collect(args, c, families)


if __name__ == '__main__':
    raise SystemExit(main())
