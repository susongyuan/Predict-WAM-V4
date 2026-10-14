#!/usr/bin/env python3
"""CPU-default, fail-closed single-case v2 probe; NEVER production data.

Main-agent integration: plan each of REQUIRED_CASES with this entry, freeze
candidate_sha256 AND bindings_sha256, then (only in a separately authorized
GPU window) pass --execute and both --expected-* hashes. An external owner
zhuhai-task1-repair-v2-<run> must already hold /tmp/project2_gpu.lock. Use
slot 0 sequentially, or distinct --worker-slot 0..3 with >=15s launch stagger.
This entry neither acquires/releases that lock nor starts another collector.
Use verify_suite() after all four; even a pass does not auto-promote v2.

Production integration is intentionally NOT implemented here: a new versioned
collector must import this SmoothExpert and bind v2 + inherited v1 + original
scene/score assets, require four strict same-binding probe passes, and preserve
the original production failure ledger. Do not edit the running generic file.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VERIFICATION = ROOT / 'outputs/zhuhai2026_tasks12/verification/tasks12_collection_repair'
SOURCE = ROOT / 'outputs/zhuhai2026_tasks12/production_20261009_v1/zhuhai_task1/source/manifest.json'
CANDIDATE = ROOT / 'harness/isaac_zhuhai_task1_expert_repair_v2.py'
ENTRY = Path(__file__).resolve()
GPU_LOCK = Path('/tmp/project2_gpu.lock')
REQUIRED_CASES = {944025: 943064, 944026: 943064, 945024: 943061, 947000: 943003}
OWNER_PREFIX = 'zhuhai-task1-repair-v2-'
SCOPE = 'task1_expert_repair_v2_physical_probe_not_production'
GIB = 1024**3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True, choices=tuple(REQUIRED_CASES))
    p.add_argument('--source-parent', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu-lock-owner')
    p.add_argument('--worker-slot', type=int, choices=range(4), default=0)
    p.add_argument('--execute', action='store_true', help='Requires separate GPU-window authorization; OFF by default')
    p.add_argument('--expected-candidate-sha256')
    p.add_argument('--expected-bindings-sha256')
    return p


def validate_request(args):
    if REQUIRED_CASES.get(args.seed) != args.source_parent:
        raise ValueError('Use the original four seed/parent pairs; no seed selection')
    raw = args.out.absolute()
    out = raw.resolve()
    if raw.is_symlink() or out.exists():
        raise FileExistsError('Fresh case directory required; existing failures/partial are preserved')
    if (out.parent.parent != VERIFICATION.resolve()
            or not re.fullmatch(r'task1_expert_repair_v2(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?', out.parent.name)
            or out.name != f'seed_{args.seed}' or raw != out):
        raise ValueError('Use verification/tasks12_collection_repair/task1_expert_repair_v2_<run>/seed_<seed>; no v1, production or symlink reuse')
    if args.gpu_lock_owner is not None and not re.fullmatch(OWNER_PREFIX + r'[A-Za-z0-9_.-]+', args.gpu_lock_owner):
        raise ValueError('A dedicated v2 owner is required; never borrow task2/v11 owner')
    if args.execute:
        if not args.gpu_lock_owner:
            raise ValueError('--execute requires an external shared --gpu-lock-owner')
        for name in ('expected_candidate_sha256', 'expected_bindings_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', getattr(args, name) or ''):
                raise ValueError(f'--execute requires frozen {name}, never auto-accept current code')
    return out


def verify_owner(owner):
    if not owner or not GPU_LOCK.is_file() or GPU_LOCK.is_symlink() or GPU_LOCK.read_text().strip() != owner:
        raise RuntimeError('External shared GPU owner mismatch; no acquisition/release/stealing allowed')


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if sha256(path) != expected:
            raise RuntimeError(f'Frozen probe source/asset/parent changed: {path}')


def make_plan(args, collector=None):
    """CPU-only and read-only, including the default CLI path. No GPU query."""
    out = validate_request(args)
    if collector is None:
        from scripts import collect_isaac_zhuhai_recycle as collector
    source = json.loads(SOURCE.read_text())
    matches = [row for row in source['attempts']
               if row['seed'] == args.source_parent and row['status'] == 'success']
    if len(matches) != 1:
        raise ValueError('Exactly one registered successful original source parent required')
    row = matches[0]
    if row.get('strict_predicate') is not True or row.get('plan_success') is not True:
        raise ValueError('Parent must already pass the original strict source criteria')
    parent = {key: row[key] for key in ('seed', 'family_id', 'data_path', 'data_sha256')}
    if Path(parent['data_path']).is_symlink() or sha256(parent['data_path']) != parent['data_sha256']:
        raise ValueError('Registered source parent data changed')
    # Construct the compatible original source contract FIRST; only the new
    # probe contract is then marked nonproduction. The source is never relabelled.
    argv = ['--task', 'zhuhai_task1', '--mode', 'randomized', '--count', '100',
        '--seed-start', str(args.seed), '--source-manifest', str(SOURCE), '--out', str(out),
        '--max-attempts', '1', '--max-consecutive-failures', '3']
    if args.gpu_lock_owner:
        argv += ['--gpu-lock-owner', args.gpu_lock_owner]
    config = collector.parser().parse_args(argv)
    contract = collector.contract(config)
    extra = [CANDIDATE, ENTRY, ROOT/'harness/isaac_zhuhai_task1_expert_repair_v1.py',
             ROOT/'harness/isaac_zhuhai_task2.py', ROOT/'harness/zhuhai_task2_rules.py',
             ROOT/'scripts/test_zhuhai_task1_expert_repair_v2.py']
    contract['scene_dependency_sha256'].update({str(p): sha256(p) for p in extra})
    bindings = {**contract['scene_dependency_sha256'], **contract['robot_binding']['asset_sha256'],
        contract['collection_implementation']['path']: contract['collection_implementation']['sha256'],
        str(SOURCE): sha256(SOURCE), parent['data_path']: parent['data_sha256']}
    # Include ALL required parents in the suite binding, so two different
    # parents do not produce incompatible per-case frozen fingerprints.
    for seed in set(REQUIRED_CASES.values()):
        registered = [r for r in source['attempts'] if r['seed'] == seed and r['status'] == 'success']
        if len(registered) != 1:
            raise ValueError('Missing required original source parent')
        bindings[registered[0]['data_path']] = registered[0]['data_sha256']
    fingerprint, candidate_hash = digest_json(bindings), bindings[str(CANDIDATE)]
    for actual, expected in ((candidate_hash, args.expected_candidate_sha256),
                             (fingerprint, args.expected_bindings_sha256)):
        if expected is not None and actual != expected:
            raise RuntimeError('Frozen candidate/bindings hash mismatch; create a NEW versioned probe')
    contract.update(source_probe=True, allow_unvalidated_probe=True, collection_scope=SCOPE,
        admitted_to_production=False, production_quota_credit=0, requested_success_count=1,
        record_video=False, third_view=False, expert_variant=dict(path=str(CANDIDATE), sha256=candidate_hash),
        probe_implementation=dict(path=str(ENTRY), sha256=bindings[str(ENTRY)]),
        probe_bindings_sha256=fingerprint, max_attempts=1, max_consecutive_failures=3)
    config.count = config.max_attempts = 1
    suite = dict(schema=SCOPE, required_cases=[dict(seed=s, parent=p) for s, p in REQUIRED_CASES.items()],
        minimum_strict_physical_passes=4, candidate_sha256=candidate_hash,
        bindings_sha256=fingerprint, bindings=bindings, admitted_to_production=False,
        production_quota_credit=0, record_video=False, third_view=False,
        failure_policy='one attempt/case; no resume/replacement/guard reset; all four required')
    return config, contract, parent, suite


def require_resources(report, slot, *, launch):
    """Pure predicate; actual read-only GPU inventory is execute-only."""
    ram, disk, devices = report['ram'], report['disk'], report['gpu'].get('devices', [])
    minimum = max(2 * GIB, .10 * ram['total_bytes'], 7 * GIB if launch else 0)
    limit = .90 if slot == 3 else .95
    swap = ram['swap_used_bytes'] / ram['swap_total_bytes'] if ram['swap_total_bytes'] else 0.
    if (ram['available_bytes'] < minimum or swap >= limit or disk['free_bytes'] < 100 * GIB
            or not devices or any(d['total_bytes'] - d['used_bytes'] < (3 * GIB if launch else GIB // 2) for d in devices)):
        raise RuntimeError('Probe resource reserve reached; stop and preserve all partial evidence')


@contextmanager
def worker_slot(owner, slot):
    verify_owner(owner)
    name = hashlib.sha256(owner.encode()).hexdigest()[:20]
    path = Path('/tmp') / f'project2_task1_repair_v2_{name}_slot{slot}.lock'
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Shared-owner worker slot already occupied; maximum four') from exc
        yield
    finally:
        os.close(fd)  # Never unlink GPU lock, slot files, failed HDFs or partials.


def freeze_plan(config, contract, suite, collector):
    suite_dir = config.out.parent
    suite_dir.mkdir(parents=True, exist_ok=True)
    collector.write_once(suite_dir/'suite_plan.json', suite, identical_ok=True)
    config.out.mkdir(exist_ok=False)
    snapshots = config.out/'source_snapshot'; snapshots.mkdir()
    for path, expected in suite['bindings'].items():
        p = Path(path)
        if p.suffix == '.py' and p.is_relative_to(ROOT):
            content = p.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f'Source changed while snapshotting: {p}')
            target = snapshots/p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(content)
    collector.write_once(config.out/'plan.json', dict(contract=contract, seed=config.seed_start,
        source_parent=REQUIRED_CASES[config.seed_start], suite_bindings_sha256=suite['bindings_sha256'],
        admitted_to_production=False, production_quota_credit=0))


def execute_probe(args, config, contract, parent, suite):
    from scripts import collect_isaac_zhuhai_recycle as c
    with worker_slot(args.gpu_lock_owner, args.worker_slot):
        verify_bindings(suite['bindings'])
        require_resources(c.resource_snapshot(), args.worker_slot, launch=True)
        if os.environ.get('OMNI_KIT_ACCEPT_EULA') != 'YES':
            raise RuntimeError('Explicit OMNI_KIT_ACCEPT_EULA=YES required in future launch environment')
        freeze_plan(config, contract, suite, c)
        verify_owner(args.gpu_lock_owner)
        verify_bindings(suite['bindings'])
        # Imports/patching occur ONLY in this new probe process after all gates.
        from harness.isaac_zhuhai_task1_expert_repair_v2 import SmoothExpert
        from harness import isaac_zhuhai_expert as original
        from harness import isaac_zhuhai_runtime as runtime
        old_expert, old_env, old_log = original.SmoothExpert, runtime.IsaacZhuhaiEnv, c.log

        class ProbeEnv(old_env):
            def step_action(self, *values, **kwargs):
                verify_owner(args.gpu_lock_owner)
                now = time.monotonic()
                if now - getattr(self, '_v2_resource_check', 0.) >= 15.:
                    require_resources(c.resource_snapshot(), args.worker_slot, launch=False)
                    self._v2_resource_check = now
                return super().step_action(*values, **kwargs)

            def close(self):
                code = 2
                try:
                    manifest = json.loads((config.out/'manifest.json').read_text())
                    if manifest['status'] == 'complete' and manifest['success_count'] == 1:
                        verify_bindings(suite['bindings'])
                        code = 0
                    self._clear_scene()
                finally:
                    if self.app is not None:
                        self.app.close(exit_code=code); self.app = None

        original.SmoothExpert, runtime.IsaacZhuhaiEnv = SmoothExpert, ProbeEnv
        # Do not let a standalone probe edit progress1.md/current running docs.
        c.log = lambda message, **kw: print(f'[v2-probe] {message}', flush=True)
        try:
            return c.collect(config, contract, [parent])
        finally:
            original.SmoothExpert, runtime.IsaacZhuhaiEnv, c.log = old_expert, old_env, old_log


def verify_suite(directory):
    """Read-only strict four-case gate for the main agent; no promotion or writes."""
    from scripts import collect_isaac_zhuhai_recycle as c
    directory = Path(directory).resolve()
    if (directory.parent != VERIFICATION.resolve()
            or not re.fullmatch(r'task1_expert_repair_v2(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?', directory.name)):
        raise ValueError('Only a new v2 verification suite can be admitted for review')
    suite = json.loads((directory/'suite_plan.json').read_text())
    expected_cases = [dict(seed=s, parent=p) for s, p in REQUIRED_CASES.items()]
    if (suite.get('schema') != SCOPE or suite.get('required_cases') != expected_cases
            or suite.get('admitted_to_production') is not False
            or suite.get('production_quota_credit') != 0
            or suite.get('minimum_strict_physical_passes') != 4
            or suite.get('record_video') is not False or suite.get('third_view') is not False
            or suite.get('candidate_sha256') != suite['bindings'].get(str(CANDIDATE))
            or digest_json(suite['bindings']) != suite['bindings_sha256']):
        raise ValueError('Not an immutable complete v2 suite protocol')
    verify_bindings(suite['bindings'])
    paths = []
    for seed, parent in REQUIRED_CASES.items():
        path = directory/f'seed_{seed}'/'manifest.json'
        manifest = json.loads(path.read_text()); contract = manifest['contract']
        if (manifest['status'] != 'complete' or len(manifest['attempts']) != 1
                or manifest.get('requested_success_count') != 1
                or contract.get('collection_scope') != SCOPE
                or contract.get('source_probe') is not True
                or contract.get('admitted_to_production') is not False
                or contract.get('production_quota_credit') != 0
                or contract.get('probe_bindings_sha256') != suite['bindings_sha256']
                or contract.get('expert_variant', {}).get('sha256') != suite['candidate_sha256']
                or contract.get('record_video') is not False or contract.get('third_view') is not False):
            raise ValueError(f'Incomplete/nonmatching strict v2 case: {seed}')
        rows = c._success_rows(manifest, verify_files=True)  # ORIGINAL strict judge/evidence validator.
        if len(rows) != 1 or rows[0]['seed'] != seed or rows[0]['source_parent'] != parent:
            raise ValueError(f'Physical probe failed or original seed/parent was changed: {seed}')
        receipt = json.loads(Path(rows[0]['expert_receipt_path']).read_text())
        if receipt['expert_class'] != 'harness.isaac_zhuhai_task1_expert_repair_v2.SmoothExpert':
            raise ValueError('Probe did not execute the versioned v2 expert')
        paths.append(str(path))
    return dict(all_four_strict_passed=True, manifests=paths, candidate_sha256=suite['candidate_sha256'],
                bindings_sha256=suite['bindings_sha256'], admitted_to_production=False, production_quota_credit=0)


def main(argv=None):
    args = parser().parse_args(argv)
    # Fail before even loading a collector/parent if execution owner is wrong.
    validate_request(args)
    if args.execute:
        verify_owner(args.gpu_lock_owner)
    config, contract, parent, suite = make_plan(args)
    if not args.execute:
        print(json.dumps(dict(mode='CPU_preflight_only_NO_GPU_NO_WRITES', out=str(config.out),
            seed=args.seed, source_parent=args.source_parent, candidate_sha256=suite['candidate_sha256'],
            bindings_sha256=suite['bindings_sha256'], suite=suite), ensure_ascii=False, indent=2))
        return 0
    return execute_probe(args, config, contract, parent, suite)


if __name__ == '__main__':
    raise SystemExit(main())
