#!/usr/bin/env python3
"""CPU-only observer; writes only monitor_status.json / monitor_events.jsonl.

python3 -B scripts/monitor_wam_v3_stages.py --prefix task0_v3_30k_20260929 --interval 60
Use --once for one sample. Exit: 0 once/valid budget completion, 2 attention,
3 stopped_*. No checkpoint reads, GPU calls, resource enforcement or unit mutation.
Nonstandard unit names can be specified with --controller-unit.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TAIL_BYTES = 65536
JSON_BYTES = 2 * 1024 * 1024


def read_json(path, errors, required=False):
    try:
        with path.open('rb') as f:
            data = f.read(JSON_BYTES + 1)
        if len(data) > JSON_BYTES:
            raise ValueError('JSON exceeds size limit')
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError('expected JSON object')
        return value
    except FileNotFoundError:
        if required:
            errors.append(f'missing: {path}')
    except (OSError, ValueError) as exc:
        errors.append(f'{path}: {exc}')
    return None


def tail_json(path, limit=TAIL_BYTES):
    """Read at most limit bytes; skip cut first line and unfinished final line."""
    try:
        with path.open('rb') as f:
            size = f.seek(0, os.SEEK_END)
            start = max(0, size - limit)
            f.seek(start)
            data = f.read(limit)
        if start:
            data = data.partition(b'\n')[2]
        lines = data.split(b'\n')[:-1]
        for line in reversed(lines):
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    return row
            except (ValueError, UnicodeError):
                pass
    except FileNotFoundError:
        pass
    return None


def controller_state(prefix, unit=None):
    """systemctl show only; discovery binds ExecStart --prefix to this run."""
    props = ['Id', 'LoadState', 'ActiveState', 'SubState', 'MainPID',
             'Result', 'ExecMainStatus', 'ExecStart']
    units = [unit] if unit else ['*staged*.service', 'wam-stages-*.service', '*controller*.service']
    args = ['systemctl', '--user', 'show', *units]
    for prop in props:
        args.extend(['-p', prop])
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        if result.returncode:
            return {'available': False, 'error': 'systemctl query failed: ' + result.stderr.strip()[:1000]}
        candidates = []
        for block in result.stdout.strip().split('\n\n'):
            fields = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
            command = fields.get('ExecStart', '')
            if unit or ('run_wam_v3_stages.py' in command and
                        re.search(r'--prefix(?:=|\s+)' + re.escape(prefix) + r'(?=\s|;|$)', command)):
                candidates.append(fields)
        live = [f for f in candidates if f.get('LoadState') == 'loaded'
                and f.get('ActiveState') in {'active', 'activating', 'reloading'}
                and f.get('MainPID', '0').isdigit() and int(f['MainPID']) > 0]
        if len(live) == 1:
            return {'available': True, **live[0]}
        return {'available': False, 'error': 'controller missing/inactive or ambiguous',
                'units': candidates}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'available': False, 'error': str(exc)}


def sample(root, prefix, controller=None):
    out = root / 'outputs/training' / (prefix + '_stages')
    errors = []
    plan = read_json(out / 'plan.json', errors, True) or {}
    status = read_json(out / 'status.json', errors, True) or {}
    phase = status.get('phase', '')
    phase = phase if isinstance(phase, str) else ''
    if not phase:
        errors.append('missing/invalid stage phase')
    step = status.get('step_target')
    checkpoints = plan.get('checkpoints', [])
    total = plan.get('total_optimizer_steps')
    valid_plan = (type(total) is int and total > 0 and isinstance(checkpoints, list)
                  and bool(checkpoints) and all(type(x) is int and x > 0 for x in checkpoints)
                  and checkpoints == sorted(set(checkpoints)) and checkpoints[-1] == total)
    if not valid_plan:
        errors.append('invalid plan budget/checkpoints')
    train_name = status.get('training_name')
    if not train_name and type(step) is int:
        train_name = f'{prefix}_s{step:06d}'
    if not train_name and phase == 'budget_complete' and valid_plan:
        train_name = f'{prefix}_s{total:06d}'
    physical_name = status.get('physical_name')
    physical_plan = plan.get('physical', {})
    if not physical_name and train_name and isinstance(physical_plan, dict):
        instance = physical_plan.get('instance')
        if type(instance) is int:
            physical_name = f'{train_name}_dev{instance}_k1'
    training, physical = {}, {}
    # Only named small JSON artifacts, never walk a run or open a checkpoint.
    for name, base, target, files in [
        (train_name, root / 'outputs/training', training, {'status': 'status.json', 'summary': 'summary.json'}),
        (physical_name, root / 'outputs/behavior/closed_loop', physical,
         {'manifest': 'manifest.json', 'metadata': 'eval/policy_metadata.json',
          'resource_summary': 'resource_summary.json', 'summary': 'summary.json',
          'eval_summary': 'eval/summary.json'})]:
        if not name:
            continue
        if not isinstance(name, str) or Path(name).name != name or name in {'.', '..'}:
            errors.append('invalid artifact directory name')
            continue
        directory = base / name
        target['name'] = name
        target.update({key: read_json(directory / file, errors) for key, file in files.items()})
        log = directory / ('log.jsonl' if target is training else 'resources.jsonl')
        try:
            target['last_record' if target is training else 'resources_last'] = tail_json(log)
            target['log_mtime'] = log.stat().st_mtime if log.exists() else None
        except OSError as exc:
            errors.append(f'{log}: {exc}')
    last = training.get('last_record') or {}
    training['observed_step'] = last.get('step')
    training['total_optimizer_steps'] = total
    training['step_target'] = step
    # No success or Q is synthesized; raw evidence is explicitly unvalidated.
    physical['evidence_verified_by_monitor'] = False
    physical['evaluation_complete_reported'] = (physical.get('manifest') or {}).get('evaluation_complete') is True
    best = read_json(out / 'best_task.json', errors)
    control = controller if controller is not None else controller_state(prefix)
    stopped = phase.startswith('stopped_')
    complete = (phase == 'budget_complete' and valid_plan
                and type(status.get('steps')) is int and status['steps'] == total)
    if phase == 'budget_complete' and not complete:
        errors.append('budget_complete status disagrees with plan')
    if status.get('error'):
        errors.append(str(status['error']))
    if not stopped and not complete and not control.get('available'):
        errors.append(control.get('error', 'controller unavailable'))
    attention = bool(errors) and not stopped
    observed = 'attention_required' if attention else phase
    # Invalid completion is attention, never a successful terminal claim.
    complete = complete and not attention
    return {'schema_version': 1, 'observed_at': time.time(), 'prefix': prefix,
            'phase': observed, 'source_phase': phase, 'terminal': stopped or complete or attention,
            'budget_complete': complete, 'exit_code': 3 if stopped else 2 if attention else 0,
            'errors': errors, 'plan': plan, 'status': status, 'training': training,
            'physical': physical, 'best_task': best, 'controller': control}


def event_key(row):
    return [row.get('phase'), row.get('source_phase'), (row.get('status') or {}).get('step_target'),
            row.get('errors'), row.get('budget_complete'), row.get('terminal'),
            (row.get('physical') or {}).get('evaluation_complete_reported'),
            ((row.get('training') or {}).get('status') or {}).get('status'),
            ((row.get('training') or {}).get('last_record') or {}).get('event') in {'complete', 'error'},
            ((row.get('status') or {}).get('observed_status') or {})]


def publish(out, row):
    """Atomic snapshot; append-only transitions deduplicated across restarts."""
    event_path = out / 'monitor_events.jsonl'
    previous = tail_json(event_path)
    if previous is None or previous.get('change') != event_key(row):
        event = {'observed_at': row['observed_at'], 'phase': row['phase'],
                 'change': event_key(row), 'errors': row['errors']}
        # Preserve even an interrupted prior append, without concatenating JSON.
        with event_path.open('a+b') as f:
            if f.seek(0, os.SEEK_END):
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b'\n':
                    f.write(b'\n')
            f.write((json.dumps(event, ensure_ascii=False) + '\n').encode())
            f.flush()
            os.fsync(f.fileno())
    fd, tmp = tempfile.mkstemp(prefix='.monitor_status.', dir=out)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(row, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out / 'monitor_status.json')
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--controller-unit', help='Override controller discovery for unusual unit names')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.prefix):
        parser.error('prefix must be a simple run name')
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('interval must be finite and positive')
    out = ROOT / 'outputs/training' / (args.prefix + '_stages')
    if not out.is_dir():
        parser.error(f'existing stages directory required: {out}')
    try:
        while True:
            start = time.monotonic()
            row = sample(ROOT, args.prefix, controller_state(args.prefix, args.controller_unit))
            publish(out, row)
            print(json.dumps({k: row[k] for k in ['phase', 'budget_complete', 'errors', 'exit_code']}), flush=True)
            if args.once or row['terminal']:
                return row['exit_code']
            time.sleep(max(0, args.interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f'monitor I/O failure: {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
