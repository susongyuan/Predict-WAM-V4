#!/usr/bin/env python3
"""Additive SOURCE.members/seed.txt compatibility for frozen multitask evaluator.

Does not edit source/cache, train, acquire locks, load weights, change scoring,
change seeds/limits, or record video. Normal CLI delegates to frozen eval.main.
--audit-training-manifest PATH is a read-only, checkpoint-free CPU audit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zlib

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_robotwin_multitask as evaluation
from harness.robotwin_multitask_policy_worker import data_digest, task_specs
from harness.robotwin_policy_worker import InfrastructureError, digest, identity, sha256

ENTRY = Path(__file__).resolve()
ENTRY_SHA256 = sha256(ENTRY)
FORMAT = 'robotwin-multitask-source-members-compat-v1'
FIX = ('Frozen seed inventory expects SOURCE.episodes. Official clean50 SOURCE.members '
       'instead binds source/<task>/aloha_agilex/seed.txt. Validate current SOURCE identity '
       'against cache, archived receipt identity, seed size/CRC32 and complete uint32 '
       'seed list; retain native episodes-format lookup unchanged. No seed, model, '
       'native predicate, limit, video, retry or evaluation outcome changes.')
_ORIGINAL_SEEDS = evaluation.native.dataset_seed_inventory
_ORIGINAL_TASK_PLAN = evaluation.task_plan
_ORIGINAL_QUEUE_PLAN = evaluation.queue_plan
_ORIGINAL_CHILD_COMMAND = evaluation.child_command


def require(value, reason):
    if not value:
        raise InfrastructureError('SOURCE compatibility: ' + reason)


def bound_source(cache):
    root = Path(cache['data_root']).resolve()
    source = next((p / 'SOURCE.json' for p in (root, *root.parents) if (p / 'SOURCE.json').is_file()), None)
    require(source is not None, 'SOURCE.json missing above cached data_root')
    require(sha256(source) == cache['source_sha256'], 'SOURCE bytes differ from frozen cache identity')
    value = json.loads(source.read_text())
    require(value == cache['source'], 'SOURCE JSON differs from embedded cache source')
    require(value.get('status') == 'complete', 'SOURCE receipt incomplete')
    require(isinstance(value.get('task'), str) and re.fullmatch(r'[a-z][a-z0-9_]*', value['task']), 'invalid source task')
    return source, value


def verified_archive_seed(receipt_path, receipt, task):
    """Only the exact task/aloha member; no ancestor guess or glob fallback."""
    require(receipt.get('status') == 'complete' and receipt.get('task') == task, 'archive task/status mismatch')
    require(receipt.get('archive') in ('demo_clean.zip', 'demo_randomized.zip'), 'unsupported demo archive')
    expected_hash = receipt.get('expected_sha256')
    require(isinstance(expected_hash, str) and re.fullmatch(r'[0-9a-f]{64}', expected_hash)
            and receipt.get('sha256') == expected_hash and receipt.get('zip_crc_verified') is True,
            'archive has no verified SHA256/CRC receipt')
    require(type(receipt.get('expected_bytes')) is int and receipt['expected_bytes'] > 0
            and receipt.get('received_bytes') == receipt['expected_bytes'], 'archive byte receipt incomplete')
    members = receipt.get('members')
    require(isinstance(members, list), 'missing archive members')
    wanted = f'source/{task}/aloha_agilex/seed.txt'
    # Reject unsafe inventory paths rather than silently accepting traversal.
    for member in members:
        path = member.get('path')
        require(isinstance(path, str) and not PurePosixPath(path).is_absolute()
            and '..' not in PurePosixPath(path).parts and '\\' not in path, 'unsafe archive member path')
    found = [row for row in members if row['path'] == wanted]
    require(len(found) == 1, 'need exactly one recorded task/aloha seed.txt member')
    member = found[0]
    base = Path(receipt_path).resolve().parent
    path = (base / wanted).resolve()
    require(path.is_relative_to(base), 'seed member escapes source directory')
    size, crc = member.get('bytes'), member.get('zip_crc32')
    require(type(size) is int and 0 < size <= 2 * 1024 * 1024, 'invalid seed member size')
    require(isinstance(crc, str) and re.fullmatch(r'[0-9a-fA-F]{8}', crc), 'invalid seed CRC32 receipt')
    require(path.is_file() and path.stat().st_size == size, 'seed.txt missing/size mismatch')
    payload = path.read_bytes()
    require(len(payload) == size and f'{zlib.crc32(payload) & 0xffffffff:08x}' == crc.lower(), 'seed.txt CRC32 mismatch')
    try:
        tokens = payload.decode('ascii').split()
    except UnicodeError as exc:
        raise InfrastructureError('SOURCE compatibility: seed.txt is not ASCII') from exc
    require(tokens and all(re.fullmatch(r'[0-9]+', token) for token in tokens), 'seed.txt must contain decimal uint32 integers')
    sequence = [int(token) for token in tokens]
    require(all(0 <= seed < 2**32 for seed in sequence), 'seed outside uint32 range')
    require(len(sequence) == len(set(sequence)), 'duplicate source seeds')
    prefix = f'source/{task}/aloha_agilex/data/'
    episodes = [row['path'] for row in members if row['path'].startswith(prefix) and row['path'].endswith('.hdf5')]
    require(episodes and len(episodes) == len(set(episodes)) and len(sequence) == len(episodes),
            'seed count does not equal unique recorded task HDF5 count')
    return set(sequence), path, {'seed_file': identity(path), 'bytes_verified': size,
        'crc32_verified': crc.lower(), 'seed_count': len(sequence), 'unique_seed_count': len(set(sequence)),
        'hdf5_member_count': len(episodes), 'seed_min': min(sequence), 'seed_max': max(sequence),
        'archive_receipt': identity(receipt_path), 'archive_sha256': expected_hash,
        'archive_assurance': 'prior verified archive SHA receipt; live seed CRC/size, not re-reading archive/HDF payloads'}


def seed_inventory_with_evidence(cache):
    source_path, source = bound_source(cache)
    evidence = {'task': source['task'], 'source': identity(source_path), 'source_matches_frozen_cache': True}
    if 'episodes' in source:
        # Preserve legacy task behavior and the exact seed files it resolves.
        seeds, files = _ORIGINAL_SEEDS(cache)
        evidence['lookup'] = 'unchanged_native_SOURCE.episodes'
        if 'raw_source_receipt' in source:
            raw_path = Path(source['raw_source_receipt']).resolve()
            require(sha256(raw_path) == source['raw_source_sha256'], 'legacy raw SOURCE hash mismatch')
            raw = json.loads(raw_path.read_text())
            require(raw.get('sha256') == source['raw_archive_sha256']
                    and raw.get('repo') == source.get('repo') and raw.get('revision') == source.get('revision'),
                    'legacy raw archive/revision mismatch')
            checked, path, record = verified_archive_seed(raw_path, raw, source['task'])
            require(seeds == checked and files == {path}, 'legacy/native seed lookup differs from CRC-verified archive member')
            evidence['seed_integrity'] = record
        else:
            evidence['seed_integrity'] = {'seed_files': [identity(p) for p in sorted(files)],
                'seed_count': len(seeds), 'crc32_verified': None, 'assurance': 'legacy native lookup only; no raw archive member receipt'}
    else:
        seeds, path, record = verified_archive_seed(source_path, source, source['task'])
        require(Path(cache['data_root']).resolve() == path.parent / 'data', 'cached data_root differs from task/aloha archive root')
        files = {path}
        evidence.update(lookup='verified_SOURCE.members_task_aloha_seed', seed_integrity=record)
    require(seeds, 'empty source seed inventory')
    evidence.update(seed_count=len(seeds), evaluation_seeds=list(evaluation.SEEDS),
                    overlap=sorted(seeds & set(evaluation.SEEDS)))
    require(not evidence['overlap'], 'registered evaluation seeds overlap source dataset: ' + str(evidence['overlap']))
    return seeds, files, evidence


def dataset_seed_inventory(cache):
    seeds, files, _ = seed_inventory_with_evidence(cache)
    return seeds, files


def compat_metadata():
    require(sha256(ENTRY) == ENTRY_SHA256, 'compatibility entry changed while process running')
    return {'format': FORMAT, 'entrypoint': identity(ENTRY), 'repair': FIX,
        'frozen_evaluator': identity(Path(evaluation.__file__)), 'process_local_wrapper_only': True,
        'original_sources_modified': False, 'seed_replacement': False, 'policy_or_predicate_changed': False,
        'compatibility_changes_scoring': False}


def task_plan(args, bundle, task, seed=None, assets=None):
    plan = _ORIGINAL_TASK_PLAN(args, bundle, task, seed=seed, assets=assets)
    _, _, audit = seed_inventory_with_evidence(bundle['caches'][task]['value'])
    metadata = compat_metadata()
    plan['source_compat'] = {**metadata, 'seed_inventory_audit': audit}
    plan['source_files'] = [*plan['source_files'], metadata['entrypoint']]
    return plan


def queue_plan(args, bundle):
    plan = _ORIGINAL_QUEUE_PLAN(args, bundle)
    plan['source_compat'] = compat_metadata()
    return plan


def child_command(args, task, seed, out):
    command = _ORIGINAL_CHILD_COMMAND(args, task, seed, out)
    require(len(command) > 2 and Path(command[2]).resolve() == Path(evaluation.__file__).resolve(),
            'frozen child command entrypoint contract changed')
    command[2] = str(ENTRY)
    return [*command, '--compat-sha256', ENTRY_SHA256]


@contextmanager
def installed():
    """Only this process; restore all globals even if eval/main raises."""
    targets = [(evaluation.native, 'dataset_seed_inventory', dataset_seed_inventory),
        (evaluation, 'task_plan', task_plan), (evaluation, 'queue_plan', queue_plan),
        (evaluation, 'child_command', child_command)]
    previous = [(module, key, getattr(module, key)) for module, key, value in targets]
    try:
        for module, key, value in targets:
            setattr(module, key, value)
        yield
    finally:
        for module, key, value in previous:
            setattr(module, key, value)


def audit_training_manifest(path):
    """Real immutable training metadata only: does not require any checkpoint."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    contract = manifest['contract']
    require(manifest['contract_sha256'] == digest(contract), 'training contract digest mismatch')
    require(contract['format'] == 'robotwin-multitask-native14-vtt-v1', 'wrong training format')
    caches = {}
    for item in contract['cache_files']:
        if Path(item['path']).name != 'cache_config.json':
            continue
        cache_path = Path(item['path']).resolve()
        require(cache_path.stat().st_size == item['size'] and sha256(cache_path) == item['sha256'], 'cache_config identity changed')
        cache = json.loads(cache_path.read_text())
        task = cache['source']['task']
        require(task not in caches or caches[task]['value'] == cache, 'multiple different caches for task')
        caches[task] = {'path': str(cache_path), 'value': cache}
    specs = task_specs(contract)
    require(set(caches) == {spec['task'] for spec in specs}, 'cache/task membership mismatch')
    tasks = {}
    for spec in specs:
        task = spec['task']
        cache = caches[task]['value']
        split = json.loads(Path(spec['split_manifest']).read_text())
        provenance = contract['condition_bank']['provenance']['tasks'][task]
        require(data_digest(split) == provenance['split_manifest_sha256'], 'task split digest changed')
        require(data_digest(cache) == provenance['cache_config_id'], 'task cache config ID mismatch')
        require(Path(split['data_root']).resolve() == Path(cache['data_root']).resolve(), 'split/cache data_root mismatch')
        _, _, record = seed_inventory_with_evidence(cache)
        tasks[task] = {**record, 'cache_config': identity(caches[task]['path']),
                      'split_manifest': identity(spec['split_manifest'])}
    return {'status': 'cpu_seed_inventory_verified', 'training_manifest': identity(path),
        'contract_sha256': manifest['contract_sha256'], 'source_compat': compat_metadata(), 'tasks': tasks,
        'all_zero_overlap': all(not row['overlap'] for row in tasks.values()),
        'all_seed_crc_size_verified': all(row['seed_integrity'].get('crc32_verified') is not None for row in tasks.values()),
        'model_constructed': False, 'simulator_started': False, 'cuda_initialized': False,
        'weights_loaded': False, 'files_modified': False, 'planned_evaluation_seeds': list(evaluation.SEEDS)}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--audit-training-manifest')
    parser.add_argument('--compat-sha256')
    own, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if own.compat_sha256:
        require(own.compat_sha256 == ENTRY_SHA256, 'child compatibility source SHA differs from queue')
    if own.audit_training_manifest:
        require(not remaining, 'audit mode cannot be mixed with evaluation arguments')
        print(json.dumps(audit_training_manifest(own.audit_training_manifest), sort_keys=True, indent=2))
        return 0
    with installed():
        return evaluation.main(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
