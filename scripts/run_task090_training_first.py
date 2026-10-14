#!/usr/bin/env python3
"""Task90 handoff: finish the existing 45k trainer, train 50k, then check 45k/50k.

No trainer/policy changes. --check-only is read-only. Reuse the SAME --out on restart:
verified live units are adopted, completed work is checked, ambiguous/partial work stops.
Stopping this queue does NOT stop independent trainer/triplet/GPU child units or release locks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/validation/task090_takeover_20261002_1319/training_first'
OLD_OUT = ROOT / 'outputs/validation/task090_remaining_stages_20261002'
TEMPLATE = OLD_OUT / 'train_045000_launch.json'
PYTHON = ROOT / '.conda-envs/lilawam/bin/python'
CONFIG = ROOT / 'configs/lilawam_task090_multiview_manip_v3_50k_train.yaml'
OLD_UNIT = 'task090-remaining-stages.service'
QUEUE_UNIT = 'task090-training-first.service'
SOURCE_PID = 2433544
BEGIN = '<!-- task090-training-first:begin -->'
END = '<!-- task090-training-first:end -->'
TRIPLET_MARKER = '\n## 10. 当前检查点三次测试\n'
ACTIVE = {'active', 'activating', 'deactivating', 'reloading'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def stamp(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                inode=s.st_ino, device=s.st_dev)


def sha(path):
    before = stamp(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    require(before == stamp(path), f'file changed while hashing: {path}')
    return h.hexdigest()


def write_json(path, data, *, exclusive=False):
    path = Path(path)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    else:
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


@contextmanager
def file_lock(path, *, nonblocking=False):
    with Path(path).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def replace_section(text, marker, section):
    """Replace exactly one level-two section, not every later progress section."""
    require(text.count(marker) <= 1, 'duplicate progress section')
    if marker not in text:
        return text.rstrip() + marker + section
    prefix, rest = text.split(marker, 1)
    at = rest.find('\n## ')
    return prefix + marker + section.rstrip() + '\n' + (rest[at:] if at >= 0 else '')


def merge_queue_progress(text, section):
    require(text.count(BEGIN) == text.count(END) <= 1, 'broken queue progress markers')
    block = BEGIN + '\n### 训练优先接管队列\n' + section.rstrip() + '\n' + END + '\n'
    if BEGIN in text:
        left, rest = text.split(BEGIN, 1)
        _, right = rest.split(END, 1)
        return left + block + right.lstrip('\n')
    # The legacy triplet truncates from section 10. Keep our marker before it as well.
    at = text.find(TRIPLET_MARKER)
    if at < 0:
        return text.rstrip() + '\n\n' + block
    return text[:at] + '\n\n' + block + text[at:]


def update_progress(transform):
    path = ROOT / 'progress.md'
    lock = Path('/tmp/project2_task090_training_first_progress.lock')
    with file_lock(lock):
        text = path.read_text()
        result = transform(text)
        require(path.read_text() == text, 'progress changed concurrently; refusing stale write')
        tmp = path.with_name(f'.progress.training-first.{os.getpid()}.tmp')
        try:
            # Use open, not Path.write_text: the legacy triplet adapter intercepts the latter.
            with tmp.open('x') as f:
                f.write(result)
                f.flush()
                os.fsync(f.fileno())
            require(path.read_text() == text, 'progress changed concurrently before replace')
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def triplet_entry(step, out):
    """Run the unchanged triplet entry; adapt ONLY its destructive progress writer."""
    from unittest.mock import patch
    import run_task090_checkpoint_triplet as triplet
    original = Path.write_text

    def preserving_write(path, text, *args, **kwargs):
        if path.resolve() != (ROOT / 'progress.md').resolve():
            return original(path, text, *args, **kwargs)
        require(text.count(TRIPLET_MARKER) == 1, 'unexpected legacy triplet progress format')
        section = text.split(TRIPLET_MARKER, 1)[1]
        update_progress(lambda live: replace_section(live, TRIPLET_MARKER, section))
        return len(text)

    with patch.object(sys, 'argv', [str(ROOT / 'scripts/run_task090_checkpoint_triplet.py'),
                                   '--step', str(step), '--out', str(out)]), \
            patch.object(Path, 'write_text', preserving_write):
        triplet.main()


def run_name(step):
    return f'task090_v3_50k_20260930_s{step:06d}'


def train_run(step):
    return ROOT / 'outputs/training' / run_name(step)


def train_unit(step):
    return 'wam-train-' + run_name(step)


def unit_info(unit):
    p = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'LoadState', '-p', 'ActiveState',
                        '-p', 'MainPID', '-p', 'ControlGroup', '-p', 'Result', '-p', 'ExecMainStatus'],
                       capture_output=True, text=True, timeout=15)
    info = dict(line.split('=', 1) for line in p.stdout.splitlines() if '=' in line)
    require(p.returncode == 0 or info.get('LoadState') == 'not-found',
            f'cannot inspect unit {unit}: {p.stderr.strip()}')
    require(info.get('LoadState') in {'loaded', 'not-found'}, f'unknown unit state: {unit} {info}')
    info['MainPID'] = int(info.get('MainPID', 0))
    return info


def process_argv(pid):
    try:
        status = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        if status == 'Z':
            return None
        return [s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s]
    except FileNotFoundError:
        return None


def matching_processes(script, token=None):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        argv = process_argv(int(path.name))
        if argv and str(script) in argv and (token is None or token in argv):
            result.append(int(path.name))
    return result


def gpu_owner():
    from gpu_lock import owner
    return owner()


def old_queue_absent():
    info = unit_info(OLD_UNIT)
    require(info.get('ActiveState') not in ACTIVE and info['MainPID'] == 0,
            'old queue still running; stop ONLY task090-remaining-stages.service before handoff')
    require(not matching_processes(ROOT / 'scripts/run_task090_remaining_stages.py'),
            'old queue process still exists outside/inside its unit')


def command_tail(argv):
    return argv[argv.index(str(PYTHON)):]


def validate_live_unit(unit, expected, *, expected_pid=None):
    info = unit_info(unit)
    if info.get('ActiveState') not in ACTIVE:
        require(info['MainPID'] == 0, f'inactive unit still has PID: {unit}')
        require(info.get('ActiveState') in {'inactive', 'failed'}, f'unknown active state: {info}')
        require(info.get('ActiveState') != 'failed', f'unit failed; inspect retained evidence: {unit}')
        require(not matching_processes(Path(expected[1]), expected[-1] if '--triplet-entry' in expected else
                                       expected[expected.index('--run-name') + 1]),
                f'orphan process for inactive unit: {unit}')
        return None
    pid = info['MainPID']
    require(pid > 0, f'unit transition has no verifiable PID; inspect before restart: {unit}')
    require(expected_pid is None or pid == expected_pid, f'wrong trainer PID: {pid}')
    require(process_argv(pid) == expected, f'unit process argv identity mismatch: {unit}')
    require(info.get('ControlGroup', '').endswith('/' + unit.removesuffix('.service') + '.service'),
            f'unit is not independent: {unit}')
    return pid


def manifest_identity(step):
    m = read_json(train_run(step) / 'manifest.json')
    require(m.get('task_id') == 90 and m.get('mode') == 'task_adaptation_50k'
            and m.get('run_name') == run_name(step) and m.get('requested_updates') == step
            and m.get('steps_start_at') == step - 5000 and m.get('schedule_steps') == 50000
            and m.get('fresh_optimizer') is False and m.get('offline_proxy_only') is True,
            f'training manifest identity mismatch at {step}')
    require(m['config']['training']['max_steps'] == m['config']['training']['schedule_steps'] == 50000,
            'training budget/config changed')
    require(m.get('resume_source') == str(train_run(step - 5000) / 'checkpoint_last.pt'),
            'resume source is not the previous complete last')
    require(sha(CONFIG) == m['source_sha256'].get(str(CONFIG)), 'frozen training config changed')
    return m


def audit_sources(m):
    files = m['source_sha256']
    for path, digest in files.items():
        require(sha(path) == digest, f'frozen training source changed: {path}')
    for path, identity in m['feature_payload_stats'].items():
        require(stamp(path) == identity, f'feature payload identity changed: {path}')
    added = [ROOT / 'scripts/run_task090_training_first.py', ROOT / 'scripts/test_task090_training_first.py']
    require(not any(str(p) in files for p in added), 'new scheduler unexpectedly belongs to frozen training contract')
    return dict(source_hashes=len(files), payload_stat_checks=len(m['feature_payload_stats']),
                full_feature_payload_hashes=False, new_files_in_training_contract=[])


def runtime_reference():
    """Historical fingerprints are provenance, NOT a ban on reviewed pre-launch evaluator fixes."""
    frozen = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/rigid_v2_runtime_frozen.json'
    reference = triplet_runs(40000)[0] / 'manifest.json'
    prior = {str(ROOT / key): value for key, value in read_json(reference)['artifact_sha256'].items()
             if not key.endswith('.pt')}
    prior.update({str(ROOT / key): value for key, value in read_json(frozen)['sha256'].items()})
    return prior


def runtime_paths():
    paths = [ROOT / 'scripts' / s for s in ('run_task090_training_first.py', 'gpu_lock.py',
             'run_task090_checkpoint_triplet.py', 'run_task090_preposition_matrix.py',
             'summarize_task090_checkpoints.py')]
    paths += [TEMPLATE, ROOT / 'outputs/validation/task090_takeover_20261001/prepositionv4_15k_seed0_launch.json',
              train_run(45000) / 'manifest.json', ROOT / 'harness/task090_predicates.py',
              ROOT / 'outputs/validation/task090_onboard_grasp_20261002/rigid_v2_runtime_frozen.json',
              triplet_runs(40000)[0] / 'manifest.json']
    paths += [Path(path) for path in runtime_reference()]
    return list(dict.fromkeys(paths))


def checkpoint(path, step, model_only, manifest, *, loader=None):
    """Verify local full SHA BEFORE CPU mmap unpickling; never construct a model or CUDA context."""
    receipt = read_json(str(path) + '.sha256.json')
    require(receipt.get('step') == step and receipt.get('model_only') is model_only
            and receipt.get('size') == path.stat().st_size, f'checkpoint receipt mismatch: {path}')
    digest = sha(path)
    require(digest == receipt.get('sha256'), f'checkpoint SHA mismatch: {path}')
    before = stamp(path)
    if loader is None:
        import torch
        require(not torch.cuda.is_initialized(), 'queue must remain CPU-only')
        loader = lambda p: torch.load(p, map_location='cpu', weights_only=False, mmap=True)
    ck = loader(path)
    require(ck.get('step') == step and ck.get('model_only') is model_only
            and ck.get('mode') == manifest['mode'] and ck.get('config') == manifest['config']
            and ck.get('training_contract_sha256') == manifest['training_contract_sha256']
            and ck.get('manifest_sha256') == sha(path.parent / 'manifest.json')
            and ck.get('initialization_sha256') == manifest['initialization_sha256'],
            f'checkpoint metadata identity mismatch: {path}')
    if not model_only:
        s = ck.get('resume_state') or {}
        required = {'optimizer_state_dict', 'scheduler_state_dict', 'rng', 'cursor', 'best', 'step'}
        require(ck.get('resumable') is True and ck.get('resume_format') == 'wam-v3-last-v2'
                and ck.get('optimizer_state_saved') is True and required.issubset(s)
                and s.get('safe_optimizer_boundary') is True and s.get('device_type') == 'cuda'
                and s.get('step') == step and s['scheduler_state_dict'].get('last_epoch') == step,
                f'incomplete/unsafe resume state: {path}')
    require(stamp(path) == before, f'checkpoint changed during load: {path}')
    return dict(path=str(path), sha256=digest, step=step, model_only=model_only, stat=before)


def complete_segment(step, *, source_sha=None):
    run = train_run(step)
    status, summary = read_json(run / 'status.json'), read_json(run / 'summary.json')
    require(status.get('status') == 'segment_complete' and status.get('step') == step
            and summary.get('completed_updates') == step and summary.get('resumed_from_step') == step - 5000,
            f'not a completed {step} segment; program exit/last existence is insufficient')
    m = manifest_identity(step)
    if source_sha is not None:
        require(m.get('resume_source_sha256') == source_sha, '50k resumed a different last checkpoint')
    result = {'last': checkpoint(run / 'checkpoint_last.pt', step, False, m),
              'fixed': checkpoint(run / f'checkpoint_step_{step:06d}.pt', step, True, m)}
    if step == 50000:
        result['final'] = checkpoint(run / 'checkpoint_final.pt', step, True, m)
    return result


def training_command(out, source_sha):
    original = read_json(TEMPLATE)['argv']
    cmd = [x.replace(train_unit(45000), train_unit(50000)) for x in original]
    for key, value in (('--run-name', run_name(50000)), ('--max-steps', '50000'),
                       ('--resume-source', str(train_run(45000) / 'checkpoint_last.pt')),
                       ('--resume-source-sha256', source_sha)):
        require(cmd.count(key) == 1, f'ambiguous training template flag {key}')
        cmd[cmd.index(key) + 1] = value
    cmd = ['StandardOutput=append:' + str(out / 'train_050000.log')
           if x.startswith('StandardOutput=append:') else x for x in cmd]
    require(all(x in cmd for x in ('MemoryMax=16G', 'CPUQuota=800%', 'MemorySwapMax=0')),
            'training resource budget differs')
    require(cmd[cmd.index('--config') + 1] == str(CONFIG)
            and cmd[cmd.index('--device') + 1] == 'cuda' and cmd[cmd.index('--num-workers') + 1] == '0',
            'training template configuration differs')
    require(f'ExecStartPre={PYTHON} {ROOT}/scripts/gpu_lock.py acquire --owner {train_unit(50000)} --max-wait 0' in cmd
            and f'ExecStopPost={PYTHON} {ROOT}/scripts/gpu_lock.py release --owner {train_unit(50000)}' in cmd,
            'missing exact GPU lock ownership hooks')
    return cmd


def triplet_command(out, step):
    unit = f'task090-training-first-triplet-{step:06d}'
    cmd = ['systemd-run', '--user', '--collect', '--unit=' + unit, '--working-directory=' + str(ROOT),
           '-p', 'CPUQuota=100%', '-p', 'MemoryMax=1G', '-p', 'MemorySwapMax=0',
           '-p', 'StandardOutput=append:' + str(out / f'triplet_{step:06d}.log'), '-p', 'StandardError=inherit',
           '--setenv=P2_TASK_ID=90', '--setenv=PYTHONDONTWRITEBYTECODE=1',
           str(PYTHON), str(ROOT / 'scripts/run_task090_training_first.py'), '--triplet-entry',
           '--step', str(step), '--out', str(out / f'triplet_{step:06d}')]
    return unit, cmd


def triplet_runs(step):
    return [ROOT / 'outputs/behavior/closed_loop' /
            f'task090_wam_s{step:06d}_dev10_seed{seed}_privbase_preposition_v4_rigidlog_v2_firstgrasp'
            for seed in (0, 1, 2)]


def verify_triplet(path, step, fixed_sha):
    state = read_json(path / 'state.json')
    rows = state.get('completed', [])
    require(state.get('status') == 'triplet_complete' and state.get('checkpoint_step') == step
            and state.get('seeds') == [0, 1, 2] and len(rows) == 3, 'partial triplet; never relaunch over it')
    from summarize_task090_checkpoints import summarize_run
    for seed, (row, run) in enumerate(zip(rows, triplet_runs(step))):
        require(row.get('status') == 'scored' and row.get('seed') == seed and row.get('instance_id') == 311
                and row.get('checkpoint_step') == step and row.get('checkpoint_sha256') == fixed_sha
                and row.get('policy_step_alignment') is True and row.get('interface_sha256')
                and row['interface_sha256'] == rows[0]['interface_sha256'], 'triplet protocol/identity mismatch')
        require(summarize_run(run) == [row], 'triplet saved row differs from original run evidence')
    return dict(step=step, trials=3, first_grasp_successes=sum(r.get('first_grasp_success') is True for r in rows),
                task_successes=sum(r.get('task_success') is True for r in rows),
                clean_exits=sum(r.get('evaluation_complete') is True for r in rows),
                evaluator_returncodes=[r.get('evaluator_returncode') for r in rows], state_path=str(path / 'state.json'))


class Queue:
    def __init__(self, out):
        self.out = out
        self.state = {}
        self.verified = {}
        self.manifest = None
        self.runtime_stamps = {}

    def save(self, **changes):
        self.state.update(changes, updated_local=time.strftime('%Y-%m-%d %H:%M:%S %z'), queue_pid=os.getpid())
        write_json(self.out / 'state.json', self.state)
        with (self.out / 'status.jsonl').open('a') as f:
            f.write(json.dumps(self.state, ensure_ascii=False) + '\n')
        update_progress(lambda text: merge_queue_progress(text,
            f'- 状态：{self.state["status"]}；目标：{self.state.get("target_step")}；已确认步数：{self.state.get("observed_step")}。\n'
            f'- 独立unit：{self.state.get("unit")}；PID：{self.state.get("pid")}；状态与launch：{self.out}。\n'
            '- 顺序：45k完整last→50k→45k三测→50k三测；三测记录齐不等于成功，原预定位/无录像协议保留。\n'
            f'- 停止先停{QUEUE_UNIT}，独立训练/测试unit不会随之停止；训练主PID用SIGTERM保存last；不要删锁或重跑部分三测。\n'))
        print(json.dumps(self.state, ensure_ascii=False), flush=True)

    def guard(self):
        old_queue_absent()
        for path, digest in self.manifest['runtime_sha256'].items():
            current = stamp(path)
            if self.runtime_stamps.get(path) != current:
                require(sha(path) == digest, f'queue/entry/source identity changed: {path}')
                self.runtime_stamps[path] = current

    def prepare(self):
        old_queue_absent()  # No output/progress mutation if the previous scheduler is alive.
        m = manifest_identity(45000)
        source_launch = read_json(TEMPLATE)
        require(m['resume_source_sha256'] == source_launch['source_sha256'], '45k launch/manifest source differs')
        audit = audit_sources(m)
        manifest_path = self.out / 'manifest.json'
        if self.out.exists():
            require(manifest_path.is_file() and (self.out / 'state.json').is_file(),
                    'existing output lacks durable manifest/state; inspect, do not overwrite')
            self.manifest = read_json(manifest_path)
            require(self.manifest.get('schema') == 'task090-training-first-v1'
                    and self.manifest.get('out') == str(self.out), 'queue output identity differs')
            self.state = read_json(self.out / 'state.json')
            require(self.state.get('status') != 'stopped_on_error', 'queue stopped on error; inspect evidence, no automatic retry')
            self.guard()
        else:
            require(not train_run(50000).exists(), 'unowned 50k output already exists')
            require(unit_info(train_unit(50000)).get('ActiveState') not in ACTIVE, 'unowned 50k unit is live')
            for step in (45000, 50000):
                require(not (OLD_OUT / f'triplet_{step:06d}').exists() and not any(p.exists() for p in triplet_runs(step)),
                        'old/partial tests already exist; inspect rather than duplicate')
            self.out.mkdir(parents=True, exist_ok=False)
            current = {str(p): sha(p) for p in runtime_paths()}
            prior = runtime_reference()
            self.manifest = dict(schema='task090-training-first-v1', out=str(self.out),
                                 training_budget=50000, order=['finish_45000', 'train_50000', 'triplet_45000', 'triplet_50000'],
                                 training_contract_sha256=m['training_contract_sha256'], source_audit=audit,
                                 runtime_sha256=current, runtime_prior_sha256=prior,
                                 runtime_changes_since_40k={p: {'before': digest, 'now': current.get(p)}
                                                           for p, digest in prior.items() if current.get(p) != digest},
                                 protocol='existing dev10/instance311 seeds0,1,2; privileged base; no video; no test feedback')
            write_json(manifest_path, self.manifest, exclusive=True)
            self.save(status='prepared', completed=[])

    def idle_gpu(self):
        for _ in range(30):
            self.guard()
            if gpu_owner() is None:
                return
            time.sleep(2)
        raise RuntimeError('GPU lock still occupied; no launch and no forced release')

    def wait_training(self, step, cmd, *, source_sha=None):
        unit = train_unit(step)
        expected = command_tail(cmd)
        while True:
            self.guard()
            pid = validate_live_unit(unit, expected, expected_pid=SOURCE_PID if step == 45000 else None)
            if pid is None:
                break
            require(gpu_owner() == unit, 'live trainer does not own its GPU lock')
            row = {}
            log = train_run(step) / 'log.jsonl'
            if log.exists():
                for line in reversed(log.read_text().splitlines()):
                    try:
                        candidate = json.loads(line)
                    except ValueError:
                        continue
                    if candidate.get('event') == 'train':
                        row = candidate
                        break
            self.save(status='waiting_45k' if step == 45000 else 'training', target_step=step, unit=unit, pid=pid,
                      observed_step=row.get('step', step - 5000),
                      resources={k: row.get(k) for k in ('ram_percent', 'rss_kib', 'vram_global_mib', 'gpu_util')})
            time.sleep(60)
        # Read-only verification may run on recovery while an OWNED later stage is live.
        # The dispatch boundary, not historical checkpoint inspection, waits for the GPU lock.
        result = complete_segment(step, source_sha=source_sha)
        require(manifest_identity(step)['training_contract_sha256'] == self.manifest['training_contract_sha256'],
                'training continuation contract changed')
        self.verified[step] = result
        self.save(status='segment_verified', target_step=step, observed_step=step, unit=None, pid=None,
                  **{f'checkpoint_{step}': result})

    def dispatch(self, path, unit, cmd, **extra):
        """Durable intent before side effects. Existing intent NEVER causes a second dispatch."""
        require(not path.exists(), 'launch intent exists; ambiguous work must not be relaunched')
        self.guard()
        self.idle_gpu()
        write_json(path, dict(unit=unit, argv=cmd, **extra), exclusive=True)
        self.save(status='launch_intent', unit=unit, pid=None, launch_path=str(path))
        self.guard()
        subprocess.run(cmd, check=True, timeout=45)
        # Record that systemd accepted the submission, even if the queue dies before the next poll.
        write_json(path.with_name(path.stem + '_accepted.json'), {'unit': unit, 'accepted_at': time.time()}, exclusive=True)
        for _ in range(100):
            info = unit_info(unit)
            if info['MainPID'] > 0 or info.get('ActiveState') not in ACTIVE:
                return
            time.sleep(.2)
        raise RuntimeError('submitted unit has no identifiable process; inspect launch record')

    def run(self):
        self.prepare()
        try:
            self.wait_training(45000, read_json(TEMPLATE)['argv'])
            source_sha = self.verified[45000]['last']['sha256']
            cmd = training_command(self.out, source_sha)
            launch = self.out / 'train_050000_launch.json'
            if launch.exists():
                record = read_json(launch)
                require(record.get('argv') == cmd and record.get('source_sha256') == source_sha,
                        'existing 50k launch differs from verified continuation')
                require(train_run(50000).exists() or unit_info(train_unit(50000)).get('ActiveState') in ACTIVE,
                        '50k launch intent without run/live unit; ambiguous, not repeated')
            else:
                require(not train_run(50000).exists() and unit_info(train_unit(50000)).get('LoadState') == 'not-found',
                        '50k output/unit not owned by this queue')
                audit_sources(manifest_identity(45000))
                self.save(status='ready_to_train_50k', target_step=50000)
                self.dispatch(launch, train_unit(50000), cmd, source_sha256=source_sha)
            self.wait_training(50000, cmd, source_sha=source_sha)
            for step in (45000, 50000):
                path = self.out / f'triplet_{step:06d}'
                unit, cmd = triplet_command(self.out, step)
                launch = self.out / f'triplet_{step:06d}_launch.json'
                if launch.exists():
                    require(read_json(launch).get('argv') == cmd, 'triplet launch identity changed')
                    require(path.exists() or unit_info(unit).get('ActiveState') in ACTIVE,
                            'triplet launch intent without output/live unit; not repeated')
                else:
                    require(not path.exists() and not any(p.exists() for p in triplet_runs(step))
                            and unit_info(unit).get('LoadState') == 'not-found', 'existing unowned triplet evidence/unit')
                    require(stamp(self.verified[step]['fixed']['path']) == self.verified[step]['fixed']['stat'],
                            'fixed checkpoint changed before deferred test')
                    self.save(status='ready_for_triplet', target_step=step, test_state_path=str(path / 'state.json'))
                    self.dispatch(launch, unit, cmd, checkpoint_sha256=self.verified[step]['fixed']['sha256'])
                while True:
                    self.guard()
                    pid = validate_live_unit(unit, command_tail(cmd))
                    if pid is None:
                        break
                    self.save(status='three_physical_checks', target_step=step, unit=unit, pid=pid,
                              test_state_path=str(path / 'state.json'))
                    time.sleep(60)
                result = verify_triplet(path, step, self.verified[step]['fixed']['sha256'])
                self.state['completed'] = [r for r in self.state.get('completed', []) if r['step'] != step] + [result]
                self.save(status='triplet_verified', target_step=step, unit=None, pid=None)
            self.save(status='authorized_budget_and_deferred_checks_complete', unit=None, pid=None)
        except Exception as exc:
            self.save(status='stopped_on_error', error=f'{type(exc).__name__}: {exc}')
            raise


def check_only(out):
    """No directory creation, progress writes, lock acquisition, model loading or GPU queries."""
    m = manifest_identity(45000)
    blockers = []
    try:
        old_queue_absent()
    except RuntimeError as exc:
        blockers.append(str(exc))
    cmd = read_json(TEMPLATE)['argv']
    pid = validate_live_unit(train_unit(45000), command_tail(cmd), expected_pid=SOURCE_PID)
    audit = audit_sources(m)
    runtime = runtime_paths()  # Includes the frozen diagnostic and 40k interface fingerprints.
    training_command(out, '0' * 64)
    print(json.dumps(dict(status='blocked' if blockers else 'preflight_read_only', blockers=blockers,
                          source_pid=pid, source_status=read_json(train_run(45000) / 'status.json'),
                          gpu_lock_owner=gpu_owner(), out=str(out), existing_output=out.exists(), source_audit=audit,
                          runtime_files=len(runtime),
                          note='not launch authorization; fixed/last/full resume state checked only after trainer exit'),
                     ensure_ascii=False, indent=2))
    return 2 if blockers else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--triplet-entry', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--step', type=int, choices=(45000, 50000), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    out = args.out.resolve()
    if args.triplet_entry:
        require(args.step is not None and not args.check_only, 'invalid internal triplet invocation')
        old_queue_absent()
        parent = out.parent
        require(out == parent / f'triplet_{args.step:06d}', 'unexpected internal triplet output')
        unit, expected = triplet_command(parent, args.step)
        launch = read_json(parent / f'triplet_{args.step:06d}_launch.json')
        require(launch.get('unit') == unit and launch.get('argv') == expected,
                'internal triplet entry requires a matching durable queue launch')
        queue = Queue(parent)
        queue.manifest = read_json(parent / 'manifest.json')
        queue.guard()
        state = read_json(parent / 'state.json')
        require(state.get('checkpoint_50000', {}).get('last', {}).get('step') == 50000,
                'deferred tests require verified 50k training first')
        for step in (45000, 50000):
            fixed = state[f'checkpoint_{step}']['fixed']
            require(stamp(fixed['path']) == fixed['stat'], 'fixed checkpoint changed before internal triplet')
        triplet_entry(args.step, out)
        return 0
    require(args.step is None, '--step is internal to the unchanged triplet entry')
    if args.check_only:
        return check_only(out)
    old_queue_absent()
    with file_lock('/tmp/project2_task090_training_first.queue.lock', nonblocking=True):
        Queue(out).run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
