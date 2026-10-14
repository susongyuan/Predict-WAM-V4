#!/usr/bin/env python3
"""Fixed four-case task1 v2 probe queue. Default: read-only CPU plan, NO GPU.

Execution needs an already-held dedicated external v2 owner and both frozen
probe fingerprints. This queue never acquires/releases that GPU lock. It only
launches probe_zhuhai_task1_expert_repair_v2.py, never production/train/tiny/eval.
Ordinary physical failures do not select/retry seeds: finish all four once and
verify_suite(). Infrastructure/source/owner/foreign-process failures stop the
queue and only its own new-session child groups. All evidence is retained.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import probe_zhuhai_task1_expert_repair_v2 as probe

ENTRY = Path(__file__).resolve()
TEST = ROOT/'scripts/test_zhuhai_task1_repair_v2_probe_queue.py'
PYTHON = ROOT/'.conda-envs/behavior6/bin/python'
LIBSTDCPP = ROOT/'.conda-envs/behavior6/lib/libstdc++.so.6'
SCHEMA = 'task1_repair_v2_fixed_four_probe_queue_v1'
STAGGER_SECONDS = 15.
POLL_SECONDS = 1.
SHUTDOWN_GRACE_SECONDS = 30.


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=probe.VERIFICATION/'task1_expert_repair_v2_queue')
    p.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    p.add_argument('--gpu-lock-owner')
    p.add_argument('--expected-candidate-sha256')
    p.add_argument('--expected-bindings-sha256')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resource-wait-seconds', type=float, default=1800.)
    p.add_argument('--worker-timeout-seconds', type=float, default=3600.)
    return p


def validate_args(args):
    raw = args.out.absolute(); out = raw.resolve()
    if raw != out or raw.is_symlink() or out.exists():
        raise FileExistsError('New nonsymlink suite required; never reuse failures/partial/old directories')
    if (out.parent != probe.VERIFICATION.resolve()
            or not re.fullmatch(r'task1_expert_repair_v2(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?', out.name)):
        raise ValueError('Only a fresh task1_expert_repair_v2_<run> verification directory is allowed')
    if args.workers not in range(1, 5):
        raise ValueError('Maximum four workers')
    if not (0 < args.resource_wait_seconds < float('inf') and 0 < args.worker_timeout_seconds < float('inf')):
        raise ValueError('Resource/worker timeouts must be positive and finite')
    for seed, parent in probe.REQUIRED_CASES.items():
        probe.validate_request(SimpleNamespace(seed=seed, source_parent=parent, out=out/f'seed_{seed}',
            execute=args.execute, gpu_lock_owner=args.gpu_lock_owner,
            expected_candidate_sha256=args.expected_candidate_sha256,
            expected_bindings_sha256=args.expected_bindings_sha256))
    return out


def file_stamp(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def build_plan(args):
    """Read/hash only. No mkdir, subprocess, GPU inventory, lock, or Isaac import."""
    out = validate_args(args)
    suite = None
    for seed, parent in probe.REQUIRED_CASES.items():
        request = SimpleNamespace(seed=seed, source_parent=parent, out=out/f'seed_{seed}',
            execute=args.execute, gpu_lock_owner=args.gpu_lock_owner, worker_slot=0,
            expected_candidate_sha256=args.expected_candidate_sha256,
            expected_bindings_sha256=args.expected_bindings_sha256)
        _, _, _, candidate = probe.make_plan(request)
        if suite is not None and candidate != suite:
            raise RuntimeError('Four cases must share exactly the same frozen probe suite')
        suite = candidate
    own_bindings = {str(p): probe.sha256(p) for p in (ENTRY, TEST, PYTHON.resolve(), LIBSTDCPP.resolve())}
    bindings = {**suite['bindings'], **own_bindings}
    owner = args.gpu_lock_owner
    cases = []
    for seed, parent in probe.REQUIRED_CASES.items():
        commands = {}
        for slot in range(args.workers):
            commands[str(slot)] = [str(PYTHON.resolve()), '-B', str(probe.ENTRY),
                '--seed', str(seed), '--source-parent', str(parent), '--out', str(out/f'seed_{seed}'),
                '--gpu-lock-owner', owner or '<external-v2-owner-required>', '--worker-slot', str(slot),
                '--execute', '--expected-candidate-sha256', suite['candidate_sha256'],
                '--expected-bindings-sha256', suite['bindings_sha256']]
        cases.append(dict(seed=seed, parent=parent, out=str(out/f'seed_{seed}'), commands_by_slot=commands))
    return dict(schema=SCHEMA, out=str(out), gpu_lock_owner=owner, workers=args.workers,
        stagger_seconds=STAGGER_SECONDS, resource_wait_seconds=args.resource_wait_seconds,
        worker_timeout_seconds=args.worker_timeout_seconds, cases=cases, suite=suite,
        candidate_sha256=suite['candidate_sha256'], bindings_sha256=suite['bindings_sha256'],
        queue_bindings=own_bindings, file_stamps={path: file_stamp(path) for path in bindings},
        child_env=dict(OMNI_KIT_ACCEPT_EULA='YES', LD_PRELOAD=str(LIBSTDCPP.resolve()),
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1'),
        record_video=False, third_view=False, admitted_to_production=False, production_quota_credit=0,
        policy='exactly four original cases, one launch each; no resume/retry/replacement; no GPU lock release')


class FrozenGuard:
    def __init__(self, plan):
        self.plan = plan
        self.digest = probe.digest_json(plan)
        self.bindings = {**plan['suite']['bindings'], **plan['queue_bindings']}

    def check(self, *, full=False):
        if probe.digest_json(self.plan) != self.digest:
            raise RuntimeError('Frozen command/launch plan changed')
        for path, expected in self.bindings.items():
            if file_stamp(path) != self.plan['file_stamps'][path]:
                raise RuntimeError(f'Frozen source/asset/parent changed: {path}')
            # Hash code every poll, even if size/mtime was restored. Large
            # unchanged parent HDFs/assets are rehashed only at admission.
            if (full or Path(path).suffix == '.py') and probe.sha256(path) != expected:
                raise RuntimeError(f'Frozen source hash drift: {path}')


def process_snapshot():
    """Read /proc; never import a simulator or signal a discovered process."""
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = entry.joinpath('stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            argv = [x for x in entry.joinpath('cmdline').read_text().split('\0') if x]
            rows.append(dict(pid=int(entry.name), ppid=int(fields[1]), pgid=int(fields[2]),
                state=fields[0], start_ticks=int(fields[19]), argv=argv,
                comm=stat[stat.find('(') + 1:stat.rfind(')')]))
        except (FileNotFoundError, ProcessLookupError):
            continue  # Normal process exit during inventory.
        except PermissionError:
            if entry.exists() and entry.stat().st_uid == os.getuid():
                raise RuntimeError(f'Cannot inventory same-user process {entry.name}; fail closed')
    return rows


def is_gpu_job(row):
    argv = row['argv']
    names = [Path(x).name.lower() for x in argv if not x.startswith('-')]
    native = (Path(argv[0]).name.lower() if argv else row.get('comm', '').lower())
    if native in {'kit', 'isaac-sim', 'isaacsim', 'isaac-sim.sh'} or native.startswith('kit_'):
        return True
    for name in names:
        if name == ENTRY.name.lower() or name.startswith('run_zhuhai_pipeline'):
            return True
        if name.endswith('.py') and (
                name.startswith(('collect_', 'probe_', 'train_', 'eval_', 'tiny_'))
                or name.startswith(('inspect_isaac_', 'prepare_isaac_', 'replay_demo_actions'))):
            return True
    if '-m' in argv and argv.index('-m') + 1 < len(argv):
        if argv[argv.index('-m') + 1].startswith(('isaacsim', 'omni.', 'robomimic.scripts.train')):
            return True
    if '-c' in argv and argv.index('-c') + 1 < len(argv):
        code = argv[argv.index('-c') + 1]
        if re.search(r'\b(import|from)\s+(isaacsim|omni|sapien|mujoco)\b|\bSimulationApp\s*\(', code):
            return True
    return False


def reject_foreign(rows, own_groups=(), self_pid=None):
    self_pid = os.getpid() if self_pid is None else self_pid
    foreign = [r for r in rows if r['state'] != 'Z' and r['pid'] != self_pid
               and r['pgid'] not in own_groups and is_gpu_job(r)]
    if foreign:
        raise RuntimeError('Foreign collector/simulator/GPU job (possibly orphan); refuse without killing it: '
                           + json.dumps([dict(pid=r['pid'], argv=r['argv']) for r in foreign]))


def resource_snapshot():
    # Execute-only; existing function does read-only nvidia-smi inventory.
    from scripts.collect_isaac_zhuhai_recycle import resource_snapshot as snapshot
    return snapshot()


def launch_allowed(report, slot):
    try:
        probe.require_resources(report, slot, launch=True)
        ram = report['ram']
        if slot == 3 and ram['available_bytes'] <= max(2*probe.GIB, .10*ram['total_bytes']):
            return False
        return True
    except RuntimeError:
        return False


@dataclass
class Services:
    popen: object = subprocess.Popen
    now: object = time.monotonic
    sleep: object = time.sleep
    resources: object = resource_snapshot
    processes: object = process_snapshot
    owner: object = probe.verify_owner
    verify: object = probe.verify_suite
    signal_group: object = os.killpg


@dataclass
class Child:
    process: object
    seed: int
    slot: int
    started: float
    log: object
    exit_seen: float | None = None


class FileEvidence:
    def __init__(self, plan):
        self.out = Path(plan['out'])
        self.out.mkdir(parents=True, exist_ok=False)
        self.write('launch.json', dict(plan=plan, plan_sha256=probe.digest_json(plan)))
        self.write('suite_plan.json', plan['suite'])  # Probe write_once accepts only an identical suite.
        target = self.out/'queue_sources'; target.mkdir()
        for path, digest in plan['queue_bindings'].items():
            if Path(path).suffix != '.py':
                continue
            content = Path(path).read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise RuntimeError('Queue source drift while freezing launch')
            with (target/Path(path).name).open('xb') as handle:
                handle.write(content)

    def write(self, name, value):
        with (self.out/name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())

    def event(self, value):
        with (self.out/'queue_events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()

    def log(self, seed):
        return (self.out/f'worker_{seed}.log').open('xb', buffering=0)


def _group_rows(rows, child):
    return [r for r in rows if r['pgid'] == child.process.pid and r['pid'] != child.process.pid and r['state'] != 'Z']


def stop_owned(children, services, evidence):
    """Only Popen-created new-session PGIDs; NEVER kill a process found by name."""
    def live():
        try:
            rows = services.processes()
        except Exception as exc:
            # Inventory failure must not strand known children before SIGINT.
            # If descendants cannot be audited, keep the result unadmitted and
            # never release the external GPU owner, even after escalation.
            evidence.event(dict(event='cleanup_inventory_failed', error=str(exc)))
            return list(children)
        return [c for c in children if c.process.poll() is None or _group_rows(rows, c)]
    # Graceful SIGINT lets probe/recorder preserve partials; bounded escalation
    # handles an unresponsive child without releasing the external GPU lock.
    for sig, grace in ((signal.SIGINT, SHUTDOWN_GRACE_SECONDS), (signal.SIGTERM, 5.), (signal.SIGKILL, 5.)):
        remaining = live()
        if not remaining:
            return
        for child in remaining:
            try:
                services.signal_group(child.process.pid, sig)
                evidence.event(dict(event='signal_owned_group', seed=child.seed, pgid=child.process.pid, signal=int(sig)))
            except ProcessLookupError:
                pass
        deadline = services.now() + grace
        while services.now() < deadline:
            if not live():
                return
            services.sleep(POLL_SECONDS)
    if live():
        raise RuntimeError('Owned child group did not exit; external GPU lock MUST remain held')


def run_queue(plan, services, evidence, guard):
    """Scheduler is dependency-injected for CPU Popen tests. No retries/selecting."""
    expected_plan = probe.digest_json(plan)
    states = {c['seed']: dict(seed=c['seed'], parent=c['parent'], out=c['out'], status='not_started', exit_code=None)
              for c in plan['cases']}
    children, active, pending = [], {}, list(plan['cases'])
    next_launch, starved_since = services.now(), None
    fatal, verification, cleanup_error = None, None, None
    try:
        while pending or active:
            guard.check()
            if probe.digest_json(plan) != expected_plan:
                raise RuntimeError('Frozen worker commands changed')
            services.owner(plan['gpu_lock_owner'])
            rows = services.processes()
            reject_foreign(rows, [c.process.pid for c in children])
            now = services.now()
            for slot, child in list(active.items()):
                code = child.process.poll()
                if code is not None:
                    if child.exit_seen is None:
                        child.exit_seen = now
                    if _group_rows(rows, child):
                        if now - child.exit_seen >= SHUTDOWN_GRACE_SECONDS:
                            raise RuntimeError(f'Probe {child.seed} left live descendants after exit')
                        continue  # Keep its slot occupied until its group drains.
                    states[child.seed].update(status='finished', exit_code=code)
                    evidence.event(dict(event='worker_exit', seed=child.seed, exit_code=code, time=now))
                    child.log.close(); del active[slot]
                elif now - child.started >= plan['worker_timeout_seconds']:
                    raise RuntimeError(f'Probe worker timeout: {child.seed}')
            free = [slot for slot in range(plan['workers']) if slot not in active]
            if pending and free and now >= next_launch:
                slot = free[0]
                report = services.resources()
                if launch_allowed(report, slot):
                    # Recheck every gate immediately before publishing/launching
                    # the selected command, including after a resource wait.
                    guard.check()
                    services.owner(plan['gpu_lock_owner'])
                    reject_foreign(services.processes(), [c.process.pid for c in children])
                    case = pending[0]
                    if Path(case['out']).exists() or Path(case['out']).is_symlink():
                        raise FileExistsError('Case/partial directory appeared; never overwrite or replay')
                    command = list(case['commands_by_slot'][str(slot)])
                    evidence.write(f'command_{case["seed"]}.json', dict(seed=case['seed'], slot=slot,
                        argv=command, command_sha256=probe.digest_json(command),
                        candidate_sha256=plan['candidate_sha256'], bindings_sha256=plan['bindings_sha256']))
                    log = evidence.log(case['seed'])
                    try:
                        guard.check()  # Also catch drift during command/log publication.
                        services.owner(plan['gpu_lock_owner'])
                        process = services.popen(command, cwd=str(ROOT), env={**os.environ, **plan['child_env']},
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close(); states[case['seed']]['status'] = 'launch_failed'
                        raise
                    child = Child(process, case['seed'], slot, services.now(), log)
                    children.append(child); active[slot] = child; pending.pop(0)
                    states[child.seed].update(status='running', pid=process.pid, slot=slot, command=command)
                    evidence.event(dict(event='worker_start', seed=child.seed, pid=process.pid, slot=slot, time=child.started))
                    next_launch = child.started + STAGGER_SECONDS
                    starved_since = None
                else:
                    if starved_since is None:
                        starved_since = now
                        evidence.event(dict(event='resource_wait_lower_concurrency', time=now, active=len(active), slot=slot, snapshot=report))
                    if now - starved_since >= plan['resource_wait_seconds']:
                        raise RuntimeError('Timed out waiting for launch resource reserves; no seed replay')
            if pending or active:
                services.sleep(POLL_SECONDS)
        guard.check(full=True)
        services.owner(plan['gpu_lock_owner'])
        reject_foreign(services.processes(), [c.process.pid for c in children])
        # Run the ORIGINAL strict suite validator even after a physical failure;
        # retain its actual failure, never substitute a process-exit success.
        try:
            verification = services.verify(Path(plan['out']))
        except Exception as exc:
            verification = dict(all_four_strict_passed=False, error=f'{type(exc).__name__}: {exc}')
    except BaseException as exc:
        fatal = f'{type(exc).__name__}: {exc}'
        try:
            stop_owned(children, services, evidence)
        except BaseException as cleanup:
            cleanup_error = f'{type(cleanup).__name__}: {cleanup}'
    finally:
        for child in children:
            if not child.log.closed:
                child.log.close()
            if states[child.seed]['status'] == 'running':
                states[child.seed].update(status='interrupted', exit_code=child.process.poll())
    all_zero = all(s['status'] == 'finished' and s['exit_code'] == 0 for s in states.values())
    verified = (fatal is None and cleanup_error is None and all_zero and isinstance(verification, dict)
        and verification.get('all_four_strict_passed') is True
        and verification.get('candidate_sha256') == plan['candidate_sha256']
        and verification.get('bindings_sha256') == plan['bindings_sha256']
        and verification.get('admitted_to_production') is False and verification.get('production_quota_credit') == 0)
    summary = dict(schema=SCHEMA, status='complete_strict_probe_suite' if verified else 'failed_unadmitted',
        all_four_strict_passed=bool(verified), attempts=list(states.values()), verification=verification,
        fatal_error=fatal, cleanup_error=cleanup_error, candidate_sha256=plan['candidate_sha256'],
        bindings_sha256=plan['bindings_sha256'], launch_plan_sha256=expected_plan,
        admitted_to_production=False, production_quota_credit=0, record_video=False, third_view=False,
        external_gpu_lock_released=False)
    evidence.write('queue_summary.json', summary)
    return (0 if verified else 2), summary


@contextmanager
def queue_mutex(owner):
    probe.verify_owner(owner)
    name = hashlib.sha256(owner.encode()).hexdigest()[:20]
    fd = os.open(f'/tmp/project2_task1_repair_v2_queue_{name}.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another v2 probe queue already owns this external owner window') from exc
        yield
    finally:
        os.close(fd)  # Never unlink or release /tmp/project2_gpu.lock.


@contextmanager
def interrupt_cleanup():
    def interrupted(signum, frame):
        raise InterruptedError(f'Queue received signal {signum}')
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def execute_plan(plan):
    services = Services()
    with queue_mutex(plan['gpu_lock_owner']), interrupt_cleanup():
        guard = FrozenGuard(plan); guard.check(full=True)
        services.owner(plan['gpu_lock_owner'])
        reject_foreign(services.processes())  # BEFORE creating any output or Popen.
        evidence = FileEvidence(plan)  # Freeze commands/hashes/suite before first worker.
        return run_queue(plan, services, evidence, guard)


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    if args.execute:
        probe.verify_owner(args.gpu_lock_owner)
        reject_foreign(process_snapshot())  # Refuse a stale lock with live orphan jobs.
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(dict(mode='CPU_PLAN_NO_GPU_NO_WRITES', plan_sha256=probe.digest_json(plan), plan=plan),
                         ensure_ascii=False, indent=2))
        return 0
    code, summary = execute_plan(plan)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
