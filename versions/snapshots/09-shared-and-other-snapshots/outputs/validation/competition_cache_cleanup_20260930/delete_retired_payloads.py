#!/usr/bin/env python3
"""Delete only the user-authorized, inventoried retired Task0 DINO payloads."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

ROOT = Path('/home/user/Robotic/project2')
BASE = ROOT / 'outputs/validation/competition_cache_cleanup_20260930'
CACHE = ROOT / 'data/dino_cache_task000'
PLAN = BASE / 'proposed_actions.json'
LOCK = Path('/tmp/project2_gpu.lock')
OWNER = f'competition-cache-maintenance-{os.getpid()}'


def workers():
    targets = {'train_wam_v3.py', 'train_lilawam_behavior.py', 'cache_dino_features.py',
               'behavior_closed_loop.py', 'replay_demo_actions_feasibility.py', 'policy_server.py'}
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = [a.decode(errors='replace') for a in path.read_bytes().split(b'\0') if a]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if argv and re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', Path(argv[0]).name):
            if any(Path(a).name in targets for a in argv[1:]):
                found.append({'pid': int(path.parent.name), 'argv': argv})
    return found


def check(row):
    path = Path(row['path'])
    if path.resolve() != path or not path.is_relative_to(CACHE) or path.parent.parent != CACHE:
        raise RuntimeError(f'Candidate escape: {path}')
    if not re.fullmatch(r'episode_(?:0|[1-9]\d*)\.(?:feat|pool)\.npy', path.name):
        raise RuntimeError(f'Unexpected payload: {path}')
    current = path.lstat()
    expected = (row['device'], row['inode'], row['logical_bytes'], row['mtime_ns'], row['ctime_ns'], row['nlink'])
    observed = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns, current.st_nlink)
    if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1 or observed != expected:
        raise RuntimeError(f'Candidate identity changed: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    rows = plan['safe_first_batch']['exact_files']
    if len(rows) != 1200 or len({r['path'] for r in rows}) != 1200:
        raise RuntimeError('Expected exactly 1200 unique inventoried payloads')
    if workers():
        raise RuntimeError('Training/simulation/cache worker is active')
    if LOCK.exists() or LOCK.is_symlink():
        raise RuntimeError('GPU/maintenance lock occupied; no stealing')
    for row in rows:
        check(row)
    for native in plan['safe_first_batch']['rebuild_sources']['native_RGB_paths']:
        if not Path(native).is_file():
            raise RuntimeError('Missing regeneration source: '+native)
    protected = {}
    for path in CACHE.rglob('*'):
        if path.is_file() and not (path.name.endswith('.feat.npy') or path.name.endswith('.pool.npy')):
            if path.is_symlink() or path.stat().st_size > 16*1024**2:
                raise RuntimeError('Unexpected protected cache file: '+str(path))
            protected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in plan['checkpoint_retention']:
        if not Path(row['path']).exists():
            raise RuntimeError('Missing retained checkpoint: '+row['path'])
    before = shutil.disk_usage(ROOT)._asdict()
    summary = {'planned_files': len(rows), 'planned_allocated_bytes': sum(r['allocated_bytes'] for r in rows),
               'protected_small_files': len(protected), 'workers': [], 'execute': args.execute,
               'plan_sha256': hashlib.sha256(plan_bytes).hexdigest()}
    if not args.execute:
        print(json.dumps(summary, indent=2)); return
    pending = BASE/'deletion_started.json'
    with pending.open('x') as f:
        json.dump({**summary, 'started_at': datetime.datetime.now().astimezone().isoformat(),
                   'before_disk': before, 'protected_hashes': protected}, f, indent=2)
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(fd, (OWNER+'\n').encode()); os.close(fd)
    deleted = 0
    try:
        if workers():
            raise RuntimeError('Worker appeared after maintenance lock')
        for row in rows:
            check(row)
        with (BASE/'deleted_payloads.jsonl').open('x') as log:
            for row in rows:
                path = check(row)
                path.unlink()
                log.write(json.dumps({'path': str(path), 'allocated_bytes': row['allocated_bytes'], 'deleted': True})+'\n')
                log.flush()
                deleted += 1
        for path, digest in protected.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Protected metadata/index changed: '+path)
        if any(Path(r['path']).exists() for r in rows):
            raise RuntimeError('Inventoried payload remains')
        report = {**summary, 'deleted_files': deleted, 'deleted_allocated_bytes': sum(r['allocated_bytes'] for r in rows),
                  'after_disk': shutil.disk_usage(ROOT)._asdict(), 'protected_files_unchanged': True,
                  'finished_at': datetime.datetime.now().astimezone().isoformat(),
                  'original_data_and_weights_deleted': False,
                  'rebuild_warning': 'Old Task0 training cache is unavailable. Retained per-episode complete=true metadata is historical only; cache_dino_features skips it. Rebuild into a fresh --out directory with a new source/stat ledger, never reuse this retired directory or claim its old ledger is current.'}
        with (CACHE/'PAYLOADS_RETIRED_20260930.json').open('x') as f:
            json.dump({'cache_payloads_present': False, 'receipt': str(BASE/'deletion_result.json'),
                       'rebuild_warning': report['rebuild_warning']}, f, indent=2)
        with (BASE/'deletion_result.json').open('x') as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
    finally:
        if LOCK.is_file() and LOCK.read_text().strip() == OWNER:
            LOCK.unlink()


if __name__ == '__main__':
    main()
