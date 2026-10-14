#!/usr/bin/env python3
"""Resumable genuine Isaac demonstrations: 100 source + 200 DR/task; no video.

The new branch is native16 measured state / target action on the official dual
Panda specification. Frozen native14 data/weights remain separate. CPU plans
can report missing assets/interfaces without importing or starting Isaac.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.progress1_log_200 import log
from scripts import collect_isaac_zhuhai as source_protocol
from scripts.collect_zhuhai_data import atomic_json, sha256, verify_owner, validate_evidence
from harness import zhuhai_robot_spec as robot_spec
from harness.zhuhai_demo_recording import (data_contract, data_contract_hash, STATE_DIM, ACTION_DIM,
    STATE_SEMANTICS, ACTION_SEMANTICS)

PRODUCTION_QUOTAS = {'source': 100, 'randomized': 200}
TOTAL_PER_TASK = sum(PRODUCTION_QUOTAS.values())
TOTAL_ALL_TASKS = 2 * TOTAL_PER_TASK
MINIMUM_FREE_BYTES = 100 * 1024**3
ONBOARD_CAMERAS = ('head_camera', 'left_camera', 'right_camera')
ENGINE = {'name': 'Isaac Sim', 'version': '6.1.0.0'}
SHARED_CONTRACT_FIELDS = ('engine', 'task', 'track', 'life_mapping', 'robot',
    'robot_binding', 'frequency_hz', 'physics_dt_seconds', 'record_video',
    'third_view', 'camera_count', 'camera_names', 'source_scene',
    'scene_dependency_sha256', 'robot_contract', 'robot_contract_sha256',
    'data_contract', 'data_contract_sha256', 'state_dim', 'action_dim', 'state_semantics',
    'action_semantics', 'state_contract', 'action_contract', 'runtime_interface',
    'source_probe', 'allow_unvalidated_probe', 'collection_scope')


def write_once(path, value, *, identical_ok=False):
    """Immutable plans/evidence: never truncate an existing file, even on resume."""
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if identical_ok and path.is_file() and not path.is_symlink():
        if json.loads(path.read_text()) == value:
            return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix='.evidence-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication AND exclusive destination.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_disk(path, required_bytes=0):
    free = shutil.disk_usage(path).free
    if free < MINIMUM_FREE_BYTES + required_bytes:
        raise RuntimeError(f'Disk reserve: {free} free; require 100GiB + {required_bytes} pending bytes')
    return free


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=['zhuhai_task1', 'zhuhai_task2'])
    p.add_argument('--track', choices=['life'], default='life')
    p.add_argument('--mode', required=True, choices=list(PRODUCTION_QUOTAS))
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--seed-start', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu-lock-owner')
    p.add_argument('--life-mapping', choices=['body_text', 'scoring_table'], default='body_text')
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--source-probe', action='store_true',
        help='Exactly one physical source attempt for HDF/interface verification, never a production quota or DR parent pool')
    p.add_argument('--max-attempts', type=int)
    p.add_argument('--max-consecutive-failures', type=int, default=3)
    return p


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def runtime_interface():
    path = ROOT/'harness/isaac_zhuhai_runtime.py'
    required = {'step_action', 'actual_state', 'get_action', 'policy_observation', 'get_obs'}
    tree = ast.parse(path.read_text())
    methods = {node.name for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == 'IsaacZhuhaiEnv'
               for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return dict(required=sorted(required), missing=sorted(required-methods), ready=required <= methods,
                observation_state_key='measured_joint_gripper_state', action_dim=16, state_dim=16)


def robot_binding():
    """Read only the stdlib-safe spec/receipts, never import runtime or Isaac."""
    if (robot_spec.ROBOT_DOF != 16 or robot_spec.ARM_DOF != 7 or
            tuple(robot_spec.GRIPPER_INDICES) != (7, 15) or
            tuple(robot_spec.ARM_INDICES) != (*range(7), *range(8, 15))):
        raise ValueError('Official dual-Panda spec must declare native16 left7/gripper/right7/gripper')
    robot = robot_spec.robot_contract()
    if robot.get('action_dim') != 16 or robot.get('state_dim') != 16:
        raise ValueError('Robot contract must explicitly bind 16D measured state and target action')
    assets, missing = {}, []
    def bind(path, expected=None):
        path = Path(path).resolve()
        if not path.is_file():
            missing.append(str(path)); return
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Official asset differs from immutable receipt: {path}')
        assets[str(path)] = actual
    bind(robot_spec.ASSET, robot.get('asset_sha256'))
    bind(robot_spec.ASSET_MANIFEST, robot.get('asset_manifest_sha256'))
    if robot_spec.ASSET_MANIFEST.is_file():
        receipt = json.loads(robot_spec.ASSET_MANIFEST.read_text())
        for item in receipt.get('files', {}).values():
            bind(ROOT/item['path'], item['sha256'])
    if getattr(robot_spec, 'KINEMATIC_CHAIN', None) is not None:
        bind(robot_spec.KINEMATIC_CHAIN)
    return dict(robot=robot.get('profile', robot.get('name')), robot_contract=robot,
        robot_contract_sha256=json_digest(robot), asset_sha256=assets,
        assets_ready=not missing, missing_assets=sorted(set(missing)),
        action_dimensions_changed=True, previous_action_dim=14, action_dim=16, state_dim=16,
        state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS,
        mechanical_acceptance='not established by a data contract or asset download',
        asset_hash_scope='official spec plus every cached receipt file and declared kinematic chain')


def validate_native16_contract(value):
    expected = dict(state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS,
        action_semantics=ACTION_SEMANTICS, state_contract=robot_spec.STATE_CONTRACT,
        action_contract=robot_spec.ACTION_CONTRACT, data_contract=data_contract(),
        data_contract_sha256=data_contract_hash())
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Native16 measured-state / target-action data contract required; legacy14 is not admitted')
    robot = value.get('robot_contract', {})
    if robot.get('state_dim') != 16 or robot.get('action_dim') != 16 or robot.get('gripper_indices') != [7, 15]:
        raise ValueError('Native16 robot contract/layout missing')
    if value.get('robot_contract_sha256') != json_digest(robot):
        raise ValueError('Native16 robot contract digest mismatch')


def expert_receipt(expert, info, directory, run_contract, jsonable):
    path = directory/'expert_receipt.json'
    receipt = dict(expert_class=f'{type(expert).__module__}.{type(expert).__name__}',
        expert_info=jsonable(info), trace=jsonable(getattr(expert, 'trace', [])),
        motion_commands=jsonable(getattr(expert, 'motion_commands', [])),
        plan_success=getattr(expert, 'plan_success', None),
        state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS,
        data_contract_sha256=run_contract['data_contract_sha256'],
        robot_contract_sha256=run_contract['robot_contract_sha256'])
    # Invalid nonfinite trace values remain explicit failed evidence, not usable
    # numerical actions; this cannot turn an exception into a success.
    try:
        json.dumps(receipt, allow_nan=False)
    except (ValueError, TypeError) as exc:
        receipt = dict(serialization_error=str(exc), raw_receipt_repr=repr(receipt),
                       invalid_expert_receipt=True, plan_success=False, action_dim=16, state_dim=16)
    write_once(path, receipt)
    return dict(expert_receipt_path=str(path.resolve()), expert_receipt_sha256=sha256(path),
                expert_receipt_valid=not receipt.get('invalid_expert_receipt', False))


def contract(args):
    scene = ROOT / 'harness' / f'isaac_{args.task}.py'
    dependencies = [scene, ROOT/'harness/isaac_zhuhai_runtime.py', ROOT/'harness/isaac_zhuhai_camera.py',
        ROOT/'harness/isaac_zhuhai_expert.py', ROOT/'harness/isaac_zhuhai_collision.py', ROOT/'harness'/f'{args.task}_rules.py',
        ROOT/'harness/zhuhai_demo_recording.py', ROOT/'harness/zhuhai_robot_spec.py',
        ROOT/'harness/robotwin_dataset.py', ROOT/'scripts/collect_isaac_zhuhai.py',
        ROOT/'scripts/collect_zhuhai_data.py', ROOT/'scripts/zhuhai_resource_watch.py']
    binding = robot_binding()
    result = dict(schema='zhuhai_isaac_native16_physical_source_v3', engine=ENGINE,
        task=args.task, track=args.track, mode=args.mode, seed_start=args.seed_start,
        source_probe=bool(getattr(args, 'source_probe', False)),
        allow_unvalidated_probe=bool(getattr(args, 'source_probe', False)),
        collection_scope='single_source_interface_probe_not_production' if getattr(args, 'source_probe', False) else 'production',
        requested_success_count=args.count, production_quota=dict(PRODUCTION_QUOTAS,
            total_per_task=TOTAL_PER_TASK, total_all_tasks=TOTAL_ALL_TASKS),
        life_mapping=args.life_mapping, robot=binding['robot'], robot_binding=binding,
        robot_contract=binding['robot_contract'], robot_contract_sha256=binding['robot_contract_sha256'],
        data_contract=data_contract(), data_contract_sha256=data_contract_hash(),
        state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS,
        state_contract=robot_spec.STATE_CONTRACT, action_contract=robot_spec.ACTION_CONTRACT,
        runtime_interface=runtime_interface(),
        frequency_hz=10., physics_dt_seconds=.004, record_video=False, third_view=False,
        camera_count=3, camera_names=list(ONBOARD_CAMERAS), minimum_free_bytes=MINIMUM_FREE_BYTES,
        expert='native16 SmoothExpert.step_action; measured actual_state and separate target get_action',
        family_relation='independent PhysX rollouts; exactly two successful DR relatives per registered source family',
        source_scene={'path': str(scene), 'sha256': sha256(scene)},
        scene_dependency_sha256={str(p): sha256(p) for p in dependencies},
        max_consecutive_failures=args.max_consecutive_failures,
        max_attempts=args.max_attempts or args.count * 10,
        success_scope='strict declared simulation rules; physical evidence, no competition certification')
    result['collection_implementation'] = dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__))
    result['source_quota_transition'] = dict(from_quota=source_protocol.PRODUCTION_QUOTAS, to_quota=PRODUCTION_QUOTAS, source_relabelled=False)
    if args.source_manifest:
        source = json.loads(args.source_manifest.read_text())
        result['source_manifest'] = dict(path=str(args.source_manifest.resolve()),
            sha256=sha256(args.source_manifest), engine=source['contract']['engine'])
        stop = args.seed_start + (args.max_attempts or args.count * 10)
        if any(args.seed_start <= row['seed'] < stop for row in source['attempts']):
            raise ValueError('DR seed range overlaps an attempted source seed; no source-seed replay')
        for field in SHARED_CONTRACT_FIELDS:
            if source['contract'].get(field) != result.get(field):
                raise ValueError(f'Source/DR frozen contract differs: {field}; use a new run, never relabel evidence')
    return result


def require_physics_safety(evidence):
    safety = evidence.get('physics_safety') if isinstance(evidence, dict) else None
    if (not isinstance(safety, dict) or safety.get('passed') is not True or
            not isinstance(safety.get('events'), list) or 'checks' not in safety):
        raise ValueError('physics_safety must explicitly pass with retained events/checks; unsafe or missing is not a demonstration')
    return safety


def _success_rows(manifest, *, verify_files=True):
    """A status string alone cannot admit a failed/interrupted/duplicate episode."""
    validate_native16_contract(manifest['contract'])
    rows, seeds, paths, digests = [], set(), set(), set()
    for item in manifest['attempts']:
        seed = item.get('seed')
        if type(seed) is not int or seed < 0 or seed in seeds:
            raise ValueError('All attempted physical episode seeds must be unique nonnegative integers')
        seeds.add(seed)
        if item.get('status') != 'success':
            continue
        if item.get('plan_success') is not True or item.get('strict_predicate') is not True:
            raise ValueError('Success requires explicit expert plan_success and strict physical predicate')
        if type(item.get('observation_count')) is not int or item['observation_count'] < 2:
            raise ValueError('Success requires at least two actual observations')
        initial_time, initial = validate_evidence(item.get('initial_rule_evidence'))
        final_time, final = validate_evidence(item.get('final_rule_evidence'))
        require_physics_safety(item['initial_rule_evidence'])
        require_physics_safety(item['final_rule_evidence'])
        if item.get('physics_safety_all_observations_passed') is not True:
            raise ValueError('Success requires every recorded physics_safety check to pass')
        cameras = item.get('initial_camera_observations', {})
        if set(cameras) != set(ONBOARD_CAMERAS) or any(v.get('nonempty_rgb') is not True for v in cameras.values()):
            raise ValueError('Successful source requires actual nonempty RGB evidence from all three onboard cameras')
        coverage = item.get('initial_head_camera_evidence', {})
        if coverage.get('all_geometry_in_frame') is not True or coverage.get('all_named_entities_visible') is not True:
            raise ValueError('Successful source requires actual initial head coverage/visibility evidence')
        if initial['base_score'] != 0 or initial['full_task_success'] or not final['full_task_success'] or final_time <= initial_time:
            raise ValueError('Success requires zero initial score, forward physical time and final full score')
        times = item.get('accepted_physics_times', [])
        if len(times) != item['observation_count'] or any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
            raise ValueError('Missing/invalid actual observation timestamps')
        if abs(times[0] - initial_time) > 1e-7 or abs(times[-1] - final_time) > 1e-7 or any(abs(b-a-.1) > 1e-7 for a, b in zip(times, times[1:])):
            raise ValueError('Successful demonstration must retain uniform 10Hz timestamps including terminal state')
        for field, expected in dict(state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS,
                action_semantics=ACTION_SEMANTICS, data_contract_sha256=data_contract_hash(),
                robot_contract_sha256=manifest['contract']['robot_contract_sha256']).items():
            if item.get(field) != expected:
                raise ValueError(f'Success attempt must explicitly bind native16 measured data: {field}')
        stats = item.get('episode_statistics', {})
        for kind, semantics in (('state', STATE_SEMANTICS), ('action', ACTION_SEMANTICS)):
            values = stats.get(kind, {})
            if values.get('dimension') != 16 or values.get('semantics') != semantics or values.get('count') != item['observation_count']-1:
                raise ValueError('Success requires 16D measured-state / target-action episode statistics')
            for field in ('mean', 'std', 'minimum', 'maximum'):
                array = values.get(field, [])
                if len(array) != 16 or any(type(v) not in (int, float) or not math.isfinite(v) for v in array):
                    raise ValueError('Episode statistics must retain all 16 finite dimensions')
        if item.get('expert_receipt_valid') is not True:
            raise ValueError('Success requires a valid auditable expert receipt')
        receipt = Path(item.get('expert_receipt_path', ''))
        if verify_files and (not receipt.is_file() or sha256(receipt) != item.get('expert_receipt_sha256')):
            raise ValueError('Expert trace/receipt is missing or changed')
        metadata = item.get('metadata', {})
        if metadata.get('seed') != seed or metadata.get('family_id') != item.get('family_id') or metadata.get('contract') != manifest['contract']:
            raise ValueError('Attempt family/seed/contract provenance mismatch')
        path = Path(item['data_path'])
        if path.is_symlink():
            raise ValueError('Source data must not be a symlink')
        resolved, digest = str(path.resolve()), item['data_sha256']
        if resolved in paths or digest in digests:
            raise ValueError('Duplicate physical demonstration path/content cannot fill a quota')
        if verify_files and (not path.is_file() or sha256(path) != digest):
            raise ValueError(f'Previously successful demonstration is missing or changed: {path}')
        paths.add(resolved); digests.add(digest); rows.append(item)
    if manifest.get('success_count', len(rows)) != len(rows):
        raise ValueError('Manifest success_count disagrees with strictly successful attempts')
    return rows


def validate_manifest(path, *, task, mode, count=None, families=None, verify_files=True):
    """CPU admission used on both fresh stages and completed-stage resume."""
    if mode == 'source':
        return source_protocol.validate_manifest(path, task=task, mode=mode, count=count, families=families, verify_files=verify_files)
    path = Path(path)
    manifest = json.loads(path.read_text())
    expected = PRODUCTION_QUOTAS[mode] if count is None else count
    c = manifest['contract']
    if count is None and (c.get('source_probe') or c.get('allow_unvalidated_probe')):
        raise ValueError('Single-source interface probe is not production data or a DR source pool')
    if manifest.get('status') != 'complete' or manifest.get('requested_success_count') != expected:
        raise ValueError('Completed immutable production quota required; old 100+400 runs are not upgraded in place')
    if (c.get('engine'), c.get('task'), c.get('track'), c.get('mode')) != (ENGINE, task, 'life', mode):
        raise ValueError('Source manifest engine/task/track/mode mismatch')
    if c.get('record_video') is not False or c.get('third_view') is not False or c.get('camera_count') != 3:
        raise ValueError('Only three onboard cameras, with video/third-view OFF, are admitted')
    if c.get('camera_names') != list(ONBOARD_CAMERAS) or c.get('frequency_hz') != 10. or c.get('physics_dt_seconds') != .004:
        raise ValueError('Camera/10Hz/physics protocol mismatch')
    if c.get('production_quota') != dict(PRODUCTION_QUOTAS, total_per_task=TOTAL_PER_TASK, total_all_tasks=TOTAL_ALL_TASKS):
        raise ValueError('Production quota contract must be 100+200=300/task, 600 total')
    implementation = c.get('collection_implementation', {})
    if implementation != dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)):
        raise ValueError('Versioned DR collector implementation changed or absent')
    transition = dict(from_quota=source_protocol.PRODUCTION_QUOTAS, to_quota=PRODUCTION_QUOTAS, source_relabelled=False)
    if c.get('source_quota_transition') != transition:
        raise ValueError('Explicit original-source quota transition required')
    if c.get('requested_success_count') != expected or c.get('minimum_free_bytes', 0) < MINIMUM_FREE_BYTES:
        raise ValueError('Frozen requested quota / 100GiB disk reserve mismatch')
    if verify_files:
        for name, digest in {**c.get('scene_dependency_sha256', {}), **c.get('robot_binding', {}).get('asset_sha256', {})}.items():
            if sha256(name) != digest:
                raise ValueError(f'Frozen source/asset changed: {name}')
    rows = _success_rows(manifest, verify_files=verify_files)
    if len(rows) != expected:
        raise ValueError(f'{mode} requires {expected} strictly successful demonstrations, got {len(rows)}')
    if mode == 'source':
        family_ids = {r['family_id'] for r in rows}
        if len(family_ids) != len(rows) or any(r['family_id'] != f'life:{task}:source:{r["seed"]}' for r in rows):
            raise ValueError('Every successful source must register one distinct canonical source family')
    else:
        source_ref = c.get('source_manifest', {})
        source_path = Path(source_ref['path'])
        if sha256(source_path) != source_ref.get('sha256'):
            raise ValueError('Registered source manifest changed')
        source = json.loads(source_path.read_text())
        for field in SHARED_CONTRACT_FIELDS:
            if source['contract'].get(field) != c.get(field):
                raise ValueError(f'Source/DR contract differs: {field}')
        if families is None:
            families = validate_manifest(source_path, task=task, mode='source', verify_files=verify_files)
        parents = {r['family_id']: r for r in families}
        source_seeds = {r['seed'] for r in source['attempts']}
        source_paths = {str(Path(r['data_path']).resolve()) for r in families}
        source_hashes = {r['data_sha256'] for r in families}
        family_counts = dict.fromkeys(parents, 0)
        for r in rows:
            parent = parents.get(r['family_id'])
            if parent is None or r.get('source_parent') != parent['seed'] or r['metadata'].get('parent_source_seed') != parent['seed']:
                raise ValueError('DR source-family/parent-seed mismatch')
            if r['seed'] in source_seeds or str(Path(r['data_path']).resolve()) in source_paths or r['data_sha256'] in source_hashes:
                raise ValueError('DR must be additional physical rollouts, not reused source seeds/files/content')
            family_counts[r['family_id']] += 1
        if expected == PRODUCTION_QUOTAS['randomized'] and (len(parents) != 100 or set(family_counts.values()) != {2}):
            raise ValueError('Production DR requires two successful relatives for each of the 100 source families')
    return rows


def source_families(args):
    if args.mode == 'source':
        if args.source_manifest:
            raise ValueError('Source collection cannot itself have a source parent manifest')
        return []
    if not args.source_manifest:
        raise ValueError('DR requires a completed, hash-verified source manifest')
    return validate_manifest(args.source_manifest, task=args.task, mode='source')


def initial_manifest(path, run_contract, count):
    if path.exists():
        result = json.loads(path.read_text())
        if result['contract'] != run_contract or result['requested_success_count'] != count:
            raise ValueError('Resume contract/quota changed: use a new output directory; no in-place upgrade')
        _success_rows(result)
        for item in result['attempts']:
            if item.get('status') == 'running':
                item.update(status='interrupted', failure_reason='Previous process stopped; retain evidence and do not replay seed')
    else:
        result = dict(contract=run_contract, requested_success_count=count, attempts=[], status='ready', success_count=0)
    atomic_json(path, result)  # Mutable index only; immutable per-attempt evidence is never rewritten.
    return result


def consecutive_failures(attempts):
    count = 0
    for row in reversed(attempts):
        if row.get('status') == 'success':
            break
        count += 1
    return count


def next_parent(families, attempts):
    if not families:
        return None
    counts = dict.fromkeys((r['family_id'] for r in families), 0)
    for row in attempts:
        if row.get('status') == 'success':
            if row['family_id'] not in counts:
                raise ValueError('Unregistered DR family on resume')
            counts[row['family_id']] += 1
    return min(families, key=lambda r: counts[r['family_id']])


def collect(args, run_contract, families):
    manifest_path = args.out/'manifest.json'
    manifest = initial_manifest(manifest_path, run_contract, args.count)
    successes = sum(row.get('status') == 'success' for row in manifest['attempts'])  # Already strictly validated above.
    seed = max((a['seed'] for a in manifest['attempts']), default=args.seed_start-1) + 1
    maximum = args.max_attempts or args.count*10
    source_seeds = ({r['seed'] for r in json.loads(args.source_manifest.read_text())['attempts']}
                    if getattr(args, 'source_manifest', None) else set())
    failures = consecutive_failures(manifest['attempts'])
    if failures >= args.max_consecutive_failures:
        manifest['status'] = 'stopped_consecutive_failures'
        atomic_json(manifest_path, manifest)
        return 2  # A restart is not permission to reset the failure policy.
    from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
    from harness.isaac_zhuhai_expert import SmoothExpert
    from harness.zhuhai_demo_recording import StreamDemoRecorder, jsonable, validate_native16
    from scripts.zhuhai_resource_watch import snapshot
    env = None
    try:
        while successes < args.count and len(manifest['attempts']) < maximum:
            log(f'{args.task}/{args.mode}采集前：{successes}/{args.count}；核对资源、源码hash、GPU锁和头部视野', actor='isaac_collect')
            verify_owner(args.gpu_lock_owner); check_disk(args.out, 1024**2)
            implementation = run_contract['collection_implementation']
            if sha256(implementation['path']) != implementation['sha256']:
                raise RuntimeError('Versioned collector changed during collection')
            for path, expected in {**run_contract['scene_dependency_sha256'], **run_contract['robot_binding']['asset_sha256']}.items():
                if sha256(path) != expected:
                    raise RuntimeError(f'Frozen source/asset changed during collection: {path}')
            if args.source_manifest and sha256(args.source_manifest) != run_contract['source_manifest']['sha256']:
                raise RuntimeError('Source manifest changed during DR collection')
            parent = next_parent(families, manifest['attempts'])
            family_id = parent['family_id'] if parent else f'life:{args.task}:source:{seed}'
            if seed in source_seeds:
                raise ValueError('DR seed overlaps an attempted source seed')
            if parent and sha256(parent['data_path']) != parent['data_sha256']:
                raise ValueError('Registered parent source changed during DR collection')
            metadata = dict(task=args.task, track='life', mode=args.mode, seed=seed,
                family_id=family_id, parent_source_seed=parent['seed'] if parent else None,
                record_video=False, third_view=False, camera_names=list(ONBOARD_CAMERAS), contract=run_contract)
            name = f'episode_seed_{seed:010d}'
            path = args.out/'data'/f'{name}.hdf5'
            evidence_dir = args.out/'attempt_evidence'/name
            if path.exists() or path.with_suffix('.hdf5.partial').exists() or (args.out/'attempt_evidence'/f'{name}.json').exists():
                raise FileExistsError(f'Orphan/existing evidence preserved; inspect before retry: {name}')
            resources = snapshot()
            resource_dir = args.out/'resource_checks'
            resource_dir.mkdir(exist_ok=True)
            resource_path = resource_dir/f'{time.time_ns()}_{uuid.uuid4().hex}.json'
            write_once(resource_path, resources)
            if resources['advice'] != 'resources_within_declared_reserves':
                manifest.update(status='paused_resource_reserve', resource_advice=resources)
                atomic_json(manifest_path, manifest)
                return 3
            evidence_dir.mkdir(parents=True, exist_ok=False)
            attempt = dict(seed=seed, family_id=family_id, source_parent=parent['seed'] if parent else None,
                state_dim=16, action_dim=16, state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS,
                data_contract_sha256=data_contract_hash(), robot_contract_sha256=run_contract['robot_contract_sha256'],
                metadata=metadata, domain=args.mode, status='running', started_unix=time.time(),
                evidence_directory=str(evidence_dir.resolve()), candidate_data_path=str(path.resolve()),
                resource_snapshot_path=str(resource_path.resolve()))
            manifest['attempts'].append(attempt); manifest['status'] = 'collecting'
            write_once(evidence_dir/'intent.json', attempt)
            atomic_json(manifest_path, manifest)
            recorder = None; expert = None; info = None
            try:
                if env is None:
                    env = IsaacZhuhaiEnv(args.gpu_lock_owner, evidence_dir)
                env.output_dir = evidence_dir  # Reused simulator, unique artifact destination for every seed.
                env.setup(args.task, seed, args.mode == 'randomized', args.life_mapping,
                          source_seed=parent['seed'] if parent else seed)
                initial = jsonable(env.rule_state()); initial_time, initial_score = validate_evidence(initial)
                attempt['initial_rule_evidence'] = initial
                require_physics_safety(initial)
                if getattr(env, 'action_dim', None) != 16 or getattr(env, 'arm_dof', None) != 7 or getattr(env, 'robot_contract', None) != run_contract['robot_contract']:
                    raise ValueError('Runtime instance must match the frozen native16 robot contract')
                if initial_score['base_score'] != 0 or initial_score['full_task_success']:
                    raise ValueError('Initial scene must have zero score')
                coverage = jsonable(env.camera_evidence)
                attempt['initial_head_camera_evidence'] = coverage
                recorder = StreamDemoRecorder(path, metadata=metadata, frequency=10.)
                accepted = []
                def capture(environment):
                    target = robot_spec.validate_action(environment.get_action(), environment.joint_limits)
                    measured = validate_native16(environment.actual_state(), name='actual_state measured native16')
                    now = float(environment.physics_time)
                    if accepted and abs(now-accepted[-1]-.1) > 1e-7:
                        raise ValueError('Expert control timestamps must remain uniform at 10 Hz')
                    observation = environment.get_obs()
                    if set(observation['observation']) != set(ONBOARD_CAMERAS):
                        raise ValueError('Exactly head + left/right wrist onboard observations are required')
                    pending = 1024**2 + sum(v['rgb'].nbytes * 2 for v in observation['observation'].values())
                    check_disk(args.out, pending)
                    boundary = jsonable(environment.rule_state())
                    recorder.capture(observation, now, boundary,
                        measured_state=measured, target_action=target); accepted.append(now)
                    if len(accepted) == 1:
                        cameras = {}
                        for name, value in observation['observation'].items():
                            rgb = value['rgb']
                            cameras[name] = dict(shape=list(rgb.shape), dtype=str(rgb.dtype),
                                nonempty_rgb=bool(rgb.size and rgb.any()), minimum=int(rgb.min()), maximum=int(rgb.max()),
                                std=float(rgb.std()), sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                                sha256_scope='raw_uint8_rgb_before_source_codec')
                        attempt['initial_camera_observations'] = cameras
                        if not all(v['nonempty_rgb'] for v in cameras.values()):
                            raise ValueError('Initial actual RGB is empty/black for an onboard camera')
                        if coverage.get('all_geometry_in_frame') is not True or coverage.get('all_named_entities_visible') is not True:
                            raise ValueError('Head camera lacks geometric coverage or actual rendered object visibility')
                    try:
                        require_physics_safety(boundary)
                    except ValueError:
                        attempt['physics_safety_failure_evidence'] = boundary
                        raise
                capture(env); env.on_control_step = capture
                expert = SmoothExpert(env)
                info = expert.play_once()
                env.on_control_step = None
                final = jsonable(env.rule_state()); final_time, score = validate_evidence(final)
                attempt['final_rule_evidence'] = final
                require_physics_safety(final)
                plan_success = isinstance(info, dict) and info.get('plan_success') is True
                successful = bool(score['full_task_success'] and plan_success and final_time > initial_time and recorder.length >= 2
                    and abs(accepted[-1] - final_time) <= 1e-7)
                attempt.update(expert_info=jsonable(info), final_rule_evidence=final,
                    strict_predicate=successful, plan_success=plan_success, observation_count=recorder.length,
                    sampling_method='exact_10hz_actual_control_step', accepted_physics_times=accepted)
                check_disk(args.out, 1024**2 + recorder.length * 1024)
                attempt.update(expert_receipt(expert, info, evidence_dir, run_contract, jsonable))
                if not attempt['expert_receipt_valid']:
                    raise ValueError('Expert trace/receipt contains invalid values; never count as success')
                finished = recorder.finish(final_evidence=final, successful=successful)
                attempt['episode_statistics'] = recorder.statistics
                attempt['physics_safety_all_observations_passed'] = bool(recorder.physics_safety_passed)
                if successful:
                    attempt.update(status='success', path=str(finished.resolve()), data_path=str(finished.resolve()), data_sha256=sha256(finished))
                    successes += 1; failures = 0
                else:
                    # Do not move/unlink finished evidence. Failed HDF is flagged
                    # unsuccessful and never appears in the admitted success list.
                    attempt.update(status='failed', failure_reason='Strict physical expert criteria not achieved',
                        evidence_path=str(finished.resolve()), evidence_sha256=sha256(finished))
            except BaseException as exc:
                attempt.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                    failure_reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                if recorder is not None and (not recorder.closed or recorder.temporary.exists()):
                    partial = recorder.abort(); attempt['partial_evidence_path'] = str(partial.resolve())
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                if expert is not None and 'expert_receipt_path' not in attempt:
                    attempt.update(expert_receipt(expert, info, evidence_dir, run_contract, jsonable))
                if env is not None:
                    env.on_control_step = None
                attempt['finished_unix'] = time.time(); manifest['success_count'] = successes
                write_once(evidence_dir/'result.json', attempt)
                atomic_json(manifest_path, manifest)
                gc.collect()
            print(json.dumps(dict(task=args.task, seed=seed, status=attempt['status'], successes=successes, target=args.count)), flush=True)
            if attempt['status'] != 'success':
                failures += 1
            if failures >= args.max_consecutive_failures:
                manifest['status'] = 'stopped_consecutive_failures'; atomic_json(manifest_path, manifest)
                return 2
            seed += 1
        manifest['status'] = 'complete' if successes == args.count else 'stopped_attempt_limit'
        atomic_json(manifest_path, manifest)
        return 0 if successes == args.count else 2
    finally:
        if env is not None:
            env.close()


def main():
    args = parser().parse_args()
    if args.mode != 'randomized' or args.source_probe:
        raise ValueError('200-DR version imports original source; collect source with the frozen source collector')
    if args.source_probe:
        if args.mode != 'source' or args.count != 1 or args.max_attempts not in (None, 1):
            raise ValueError('--source-probe requires source/count=1 and at most one physical attempt')
        args.max_attempts = 1
    elif args.count != PRODUCTION_QUOTAS[args.mode]:
        raise ValueError('Production requires 100 original source + another 200 randomized = 300/task (600 total); use --source-probe for one source interface check')
    if args.seed_start < 0:
        raise ValueError('seed-start must be nonnegative')
    if not 1 <= args.max_consecutive_failures <= 3:
        raise ValueError('Consecutive failure limit must be 1..3, never weakened on restart')
    if args.max_attempts is not None and not 0 < args.max_attempts <= args.count*10:
        raise ValueError('max-attempts must be positive and no greater than count*10')
    args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        families = source_families(args); run_contract = contract(args)
        if (args.out/'manifest.json').exists() and not args.resume:
            raise FileExistsError('Existing collection requires explicit --resume; never overwrite its data/evidence')
        write_once(args.out/'plan.json', dict(contract=run_contract, requested_count=args.count,
            physical_collection_started=False), identical_ok=True)
        if args.plan_only:
            return 0
        if not run_contract['robot_binding']['assets_ready'] or not run_contract['runtime_interface']['ready']:
            raise RuntimeError('Native16 robot assets/runtime interface incomplete; CPU plan only, no Isaac launch')
        verify_owner(args.gpu_lock_owner)
        return collect(args, run_contract, families)


if __name__ == '__main__':
    raise SystemExit(main())
