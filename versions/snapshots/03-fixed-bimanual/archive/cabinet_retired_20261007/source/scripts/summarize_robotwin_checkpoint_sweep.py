#!/usr/bin/env python3
"""Read-only CPU sweep summarizer. No model/simulator imports or run discovery.

python3 -B scripts/summarize_robotwin_checkpoint_sweep.py --run-dirs RUN ... \
  --queue-manifest QUEUE.json --expected-seeds 10000 10001 10002 \
  --output-prefix outputs/validation/resume_20261005_1913/bottleneck_report

Relative queue paths: existing cwd-relative path, otherwise manifest-relative.
Distinct directories are distinct attempts; repeated references to the same path
are one physical run with all input provenance retained. Explicit manifest references can fill dev coverage only under the same protocol;
their roles remain visible (no post-hoc best-reference selection). Rates require every planned attempt in a protocol group
to be valid and all expected seeds covered. Repeated attempts remain in denominator.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
DEFAULT_SEEDS = [10000, 10001, 10002]
TERMINAL = {'native_success', 'step_limit'}
INFRA = {'infrastructure_error', 'wall_timeout', 'cancelled', 'external_stop_native_block', 'preflight_error'}
STRICT_KEYS = ('stable_after_settle', 'release_verified', 'full_containment',
               'five_second_hold', 'competition_success', 'native_terminal_predicate',
               'native_predicate_after_settle')


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return type(value) is int and value >= 0


def boolean(value):
    return value if type(value) is bool else None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def stats(values):
    values = sorted(v for v in values if number(v))
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'p95': None, 'max': None}
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': values[max(0, math.ceil(.95 * len(values)) - 1)], 'max': values[-1]}


class Reader:
    """A best-effort snapshot; partial/changed files are visible, never silently trusted."""
    def __init__(self):
        self.issues = []
        self.files = []

    def issue(self, path, message):
        self.issues.append(f'{path}: {message}')

    def read_json(self, path):
        path = Path(path)
        if not path.exists():
            return {}
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            self.files.append({'path': str(path), 'bytes': len(raw),
                               'sha256': hashlib.sha256(raw).hexdigest()})
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                self.issue(path, 'changed during read; retry summary after writer completes')
            obj = json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            json.dumps(obj, allow_nan=False)  # also reject overflow such as 1e999
            if not isinstance(obj, dict):
                raise ValueError('expected a JSON object')
            return obj
        except (OSError, ValueError) as exc:
            self.issue(path, f'unreadable JSON: {exc}')
            return {}

    def jsonl(self, path):
        path = Path(path)
        if not path.exists():
            return
        try:
            before = path.stat()
            h = hashlib.sha256()
            with path.open('rb') as f:
                for line_no, raw in enumerate(f, 1):
                    h.update(raw)
                    if not raw.strip():
                        continue
                    try:
                        obj = json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
                        if not isinstance(obj, dict):
                            raise ValueError('expected object')
                        yield obj
                    except (ValueError, UnicodeError) as exc:
                        self.issue(path, f'line {line_no}: {exc}; metrics are partial')
            after = path.stat()
            self.files.append({'path': str(path), 'bytes': after.st_size, 'sha256': h.hexdigest()})
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                self.issue(path, 'changed during read; metrics may be partial')
        except OSError as exc:
            self.issue(path, str(exc))


def action_metrics(reader, directory):
    path = directory / 'actions.jsonl'
    returned, submitted, seen = [], 0, set()
    for row in reader.jsonl(path):
        if row.get('status') == 'submitted':
            submitted += 1
        if row.get('status') != 'native_returned':
            continue
        key = (row.get('episode'), row.get('step'))
        if key in seen:
            reader.issue(path, f'duplicate native_returned for {key}; retained, no best-attempt selection')
        seen.add(key)
        returned.append(row)
    fallback_known = [r['topp_fallback_observed'] for r in returned
                      if type(r.get('topp_fallback_observed')) is bool]
    topp = [t for r in returned for t in r.get('topp', []) if isinstance(t, dict)]
    return {'log_available': path.exists(), 'submitted_rows': submitted,
            'native_returned_rows': len(returned),
            'policy_wall_seconds': stats([r.get('policy', {}).get('policy_seconds') for r in returned]),
            'control_wall_seconds': stats([r.get('control_seconds') for r in returned]),
            'topp_fallback_actions': sum(fallback_known) if fallback_known else None,
            'topp_fallback_known_actions': len(fallback_known),
            'topp_calls_observed': len(topp),
            'topp_exception_or_empty_calls': sum(t.get('status') in ('exception', 'empty') for t in topp),
            'topp_points_per_arm': stats([t.get('steps') for t in topp]),
            'clipped_actions': sum(any(r.get('clip_mask', [])) for r in returned) if returned else None}


def gt_metrics(reader, directory):
    # Separate rollout from settle to avoid duplicate control_step/settle samples
    # dominating tracking statistics or presenting settle movement as policy lifting.
    buckets = {k: {'z': [], 'origin_delta': [], 'tracking': {'left': [], 'right': []},
                   'tracking_rows': {'left': 0, 'right': 0}, 'rows': 0} for k in ('rollout', 'settle')}
    initial_z = None
    path = directory / 'evaluation_gt.jsonl'
    for row in reader.jsonl(path):
        b = buckets['settle' if 'settle_step' in row else 'rollout']
        b['rows'] += 1
        pose = row.get('object_pose_world')
        if isinstance(pose, list) and len(pose) >= 3 and number(pose[2]):
            b['z'].append(pose[2])
            if row.get('control_step') == 0 and 'settle_step' not in row and initial_z is None:
                initial_z = pose[2]
        diag = row.get('native_diagnostics') or {}
        height = (diag.get('native_predicate_evidence') or {}).get('height_delta_m')
        if number(height):
            b['origin_delta'].append(height)
        tracking = diag.get('arm_joint_tracking') or {}
        for side in ('left', 'right'):
            t = tracking.get(side) or {}
            actual, target = t.get('actual_qpos'), t.get('drive_target')
            # Require actual qpos AND drive target, never command deltas alone.
            if (isinstance(actual, list) and isinstance(target, list) and len(actual) == len(target)
                    and len(actual) > 0 and all(number(x) for x in actual + target)):
                b['tracking'][side].extend(abs(a - d) for a, d in zip(actual, target))
                b['tracking_rows'][side] += 1
    out = {'log_available': path.exists(), 'initial_object_z_m': initial_z,
           'height_reference': 'actor origin z minus control_step=0 z; not grasp/hold proof',
           'tracking_reference': 'abs(actual_qpos - drive_target), native joint units, no wrapping; grippers excluded'}
    for name, b in buckets.items():
        out[name] = {'rows': b['rows'], 'object_pose_samples': len(b['z']),
                     'max_object_rise_m': max(b['z']) - initial_z if b['z'] and initial_z is not None else None,
                     'last_object_rise_m': b['z'][-1] - initial_z if b['z'] and initial_z is not None else None,
                     'max_native_origin_height_delta_m': max(b['origin_delta']) if b['origin_delta'] else None,
                     'tracking': {side: {**stats(b['tracking'][side]), 'snapshot_rows': b['tracking_rows'][side]}
                                  for side in ('left', 'right')}}
    return out


def protocol(plan, entry):
    limits = plan.get('limits') or entry.get('limits') or {}
    return {'task': plan.get('task'), 'task_config': plan.get('task_config'),
            'control_step_limit': limits.get('control_steps', entry.get('control_steps')),
            'episode_wall_seconds': limits.get('episode_wall_seconds'),
            'settle_physics_steps': limits.get('settle_physics_steps'),
            'renderer': plan.get('renderer'), 'policy': plan.get('policy'),
            'contract_sha256': plan.get('contract_sha256'),
            'task_kwargs': plan.get('task_kwargs'),
            'control_source_hashes': {Path(x['path']).name: x.get('sha256')
                                     for x in plan.get('source_files', [])
                                     if isinstance(x, dict) and x.get('path') and Path(x['path']).name in
                                     ('_base_task.py', 'robot.py', 'planner.py', 'robotwin_qpos_adapter.py')}}


def classify(ep, limit, issues):
    status = ep.get('status', 'not_started')
    native = boolean((ep.get('evidence') or {}).get('native_benchmark_success'))
    steps = ep.get('control_steps')
    if issues:
        return 'invalid_record', False
    if status in INFRA or ep.get('error') or ep.get('cleanup_errors'):
        return 'infrastructure_or_cancelled', False
    if status in TERMINAL:
        if not integer(limit) or limit == 0 or not integer(steps) or steps == 0 or steps > limit:
            return 'invalid_record', False
        if status == 'native_success' and native is True:
            return 'valid_native_success', True
        if status == 'step_limit' and native is False and steps == limit:
            return 'valid_step_limit', True
        return 'invalid_record', False
    if status in ('not_started', 'queued', 'pending'):
        return 'not_started', False
    if status in ('running', 'starting'):
        return 'in_progress_or_unfinalized', False
    return 'invalid_or_unscored', False


def scan_run(entry):
    root = Path(entry['run_dir'])
    reader = Reader()
    plan = reader.read_json(root / 'plan.json')
    result = reader.read_json(root / 'result.json')
    run_issues = list(reader.issues)
    for key in ('limits', 'checkpoint_receipt', 'renderer', 'policy', 'task_kwargs'):
        if key in plan and plan[key] is not None and not isinstance(plan[key], dict):
            run_issues.append(f'plan.{key} must be an object')
            plan[key] = {}
    if 'episodes' in result and not isinstance(result['episodes'], list):
        run_issues.append('result.episodes must be a list')
        result['episodes'] = []
    receipt_issue_start = len(reader.issues)
    receipt = plan.get('checkpoint_receipt') or {}
    checkpoint = plan.get('checkpoint') or entry.get('checkpoint')
    checkpoint_path = checkpoint.get('path') if isinstance(checkpoint, dict) else checkpoint
    if checkpoint_path:
        cp = Path(checkpoint_path)
        if not cp.is_absolute() and not cp.exists():
            cp = root / cp
        sidecar = reader.read_json(cp.with_name(cp.name + '.sha256.json'))
        if sidecar:
            for key in ('step', 'sha256'):
                if receipt.get(key) is not None and sidecar.get(key) != receipt[key]:
                    run_issues.append(f'checkpoint sidecar disagrees with plan {key}')
            receipt = receipt or sidecar
    run_issues.extend(reader.issues[receipt_issue_start:])
    ck = receipt.get('step')
    if ck is None:
        ck = entry.get('checkpoint_step')
    elif entry.get('checkpoint_step') is not None and ck != entry['checkpoint_step']:
        run_issues.append('queue checkpoint_step disagrees with plan receipt')
    proto = protocol(plan, entry)
    summaries = {}
    for i, ep in enumerate(result.get('episodes') or []):
        if not isinstance(ep, dict):
            run_issues.append('non-object episode in result')
            continue
        name = ep.get('episode', f'episode_{i:03d}_seed{ep.get("seed", "unknown")}')
        if name in summaries:
            run_issues.append(f'duplicate episode name in result: {name}')
        summaries[name] = ep
    # Explicit run directory only; no global discovery or checkpoint selection.
    names = set(summaries) | {p.name for p in root.glob('episode_*') if p.is_dir()}
    planned_seeds = plan.get('seeds') or ([entry['seed']] if entry.get('seed') is not None else [])
    if entry.get('seed') is not None and plan.get('seeds') and entry['seed'] not in plan['seeds']:
        run_issues.append('queue seed absent from plan seeds')
    known_seeds = {e.get('seed') for e in summaries.values()}
    for i, seed in enumerate(planned_seeds):
        name = f'episode_{i:03d}_seed{seed}'
        if seed not in known_seeds and not any(n.endswith(f'_seed{seed}') for n in names):
            names.add(name)
    if not names:
        names.add('unresolved_episode')
    attempts = []
    for name in sorted(names):
        start = len(reader.issues)
        summary = summaries.get(name, {})
        directory = root / name
        saved = reader.read_json(directory / 'episode.json')
        ep = saved or summary
        issues = list(run_issues)
        if saved and summary:
            for key in ('seed', 'status', 'control_steps'):
                if key in saved and key in summary and saved[key] != summary[key]:
                    # Stale starting/running summary may lag a durable terminal episode.
                    if key == 'status' and summary[key] in ('starting', 'running') and saved[key] in TERMINAL:
                        continue
                    issues.append(f'episode.json/result disagree on {key}: {saved[key]} / {summary[key]}')
        seed = ep.get('seed', entry.get('seed'))
        if seed is None and '_seed' in name:
            try:
                seed = int(name.rsplit('_seed', 1)[1])
            except ValueError:
                pass
        if not ep:
            ep = {'status': 'not_started', 'seed': seed}
            if result.get('status') in INFRA:
                # Distinguish failure before episode creation from simply queued jobs.
                ep = {**ep, 'status': result['status'], 'error': result.get('error')}
        if ep.get('status') in TERMINAL and (not plan or not integer(ck)):
            issues.append('terminal outcome requires readable plan and checkpoint identity')
        am = action_metrics(reader, directory)
        gm = gt_metrics(reader, directory)
        if not saved and ep.get('status') == 'not_started' and am['submitted_rows'] + am['native_returned_rows'] > 0:
            ep = {**ep, 'status': 'running'}
        video_metadata = reader.read_json(directory / 'third_view_video.json')
        issues.extend(reader.issues[start:])
        category, valid = classify(ep, proto['control_step_limit'], issues)
        if ep.get('status') in TERMINAL and category == 'invalid_record' and not issues:
            issues.append('terminal status/evidence/steps/configured cap inconsistent or unknown')
        evidence = ep.get('evidence') or {}
        strict = {k: boolean(evidence.get(k)) for k in STRICT_KEYS}
        conditions = [strict[k] for k in ('stable_after_settle', 'release_verified', 'full_containment')]
        composite = (False if False in conditions else True if all(v is True for v in conditions) else None) if valid else None
        wall, steps = ep.get('wall_seconds'), ep.get('control_steps')
        attempts.append({'attempt_id': str(directory), 'run_dir': str(root), 'episode': name,
                         'role': entry['role'], 'input_provenance': entry['provenance'],
                         'checkpoint_step': ck, 'checkpoint_sha256': receipt.get('sha256'),
                         'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
                         'protocol_verified': bool(plan) and integer(proto['control_step_limit']),
                         'seed': seed, 'unit': entry.get('unit'), 'protocol_id': digest(proto), 'protocol': proto,
                         'run_status': result.get('status'), 'status': ep.get('status'), 'category': category,
                         'valid_native_outcome': valid,
                         'native_success': category == 'valid_native_success' if valid else None,
                         'observed_native_flag': boolean(evidence.get('native_benchmark_success')),
                         'native_observed_before_error': ep.get('native_observed_before_error'),
                         'strict_evidence': strict, 'stable_released_contained': composite,
                         'strict_composite_scope': 'three physical checks only, not competition success; unknown stays null',
                         'control_steps': steps, 'native_step_limit': ep.get('native_step_limit'),
                         'native_take_action_count': ep.get('native_take_action_count'),
                         'wall_seconds': wall if number(wall) else None,
                         'wall_seconds_per_control_step': wall / steps if number(wall) and integer(steps) and steps > 0 else None,
                         'actions': am, 'gt': gm,
                         'videos': [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(directory.glob('*.mp4'))],
                         'video_metadata': video_metadata,
                         'error': ep.get('error'), 'failure_phase': ep.get('failure_phase'),
                         'run_error': result.get('error'), 'issues': issues})
    return attempts, reader.files


def resolve_queue_path(value, manifest):
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    # Project-relative 'outputs/...' commonly does not exist yet for queued runs.
    if path.parts and path.parts[0] == 'outputs':
        return path.resolve()
    return (manifest.parent / path).resolve()


def collect_entries(run_dirs, queue_manifest=None):
    entries = {}
    queue_reader = Reader()
    queue = queue_reader.read_json(queue_manifest) if queue_manifest else {}
    if queue_manifest and not Path(queue_manifest).exists():
        queue_reader.issue(queue_manifest, 'explicit queue manifest does not exist')
    for key in ('jobs', 'reference_runs'):
        if key in queue and not isinstance(queue[key], list):
            queue_reader.issue(queue_manifest, f'{key} must be a list')
            queue[key] = []
    def add(path, role, meta, provenance):
        key = str(Path(path).resolve())
        if key in entries:
            old = entries[key]
            old['provenance'].append(provenance)
            for field in ('checkpoint_step', 'seed'):
                if old.get(field) is not None and meta.get(field) is not None and old[field] != meta[field]:
                    queue_reader.issue(path, f'conflicting queue metadata for {field}')
            if role == 'primary':
                old['role'] = 'primary'
            old.update({k: v for k, v in meta.items() if v is not None and k != 'run_dir'})
        else:
            entries[key] = {**meta, 'run_dir': key, 'role': role, 'provenance': [provenance]}
    for path in run_dirs:
        add(path, 'primary', {}, 'explicit CLI run')
    for i, job in enumerate(queue.get('jobs') or []):
        if not isinstance(job, dict) or not job.get('run_dir'):
            queue_reader.issue(queue_manifest, f'jobs[{i}] missing run_dir')
            continue
        declared = queue.get('protocol') or {}
        limits = {'control_steps': declared.get('max_steps'), 'settle_physics_steps': declared.get('settle_steps'),
                  'episode_wall_seconds': declared.get('episode_seconds')}
        add(resolve_queue_path(job['run_dir'], Path(queue_manifest)), 'primary',
            {'limits': limits, **job}, f'jobs[{i}]')
    for i, ref in enumerate(queue.get('reference_runs') or []):
        ref = {'run_dir': ref} if isinstance(ref, str) else ref
        if not isinstance(ref, dict) or not ref.get('run_dir'):
            queue_reader.issue(queue_manifest, f'reference_runs[{i}] missing run_dir')
            continue
        add(resolve_queue_path(ref['run_dir'], Path(queue_manifest)), 'reference', ref, f'reference_runs[{i}]')
    return list(entries.values()), queue, queue_reader


def aggregate(attempts, expected_seeds, global_issues=()):
    grouped = {}
    for a in attempts:
        key = (a['checkpoint_step'], a['checkpoint_sha256'], a['protocol_id'])
        grouped.setdefault(key, []).append(a)
    groups = []
    for (ck, sha, pid), rows in grouped.items():
        roles = sorted({a['role'] for a in rows})
        role = '+'.join(roles)
        unknown_protocol_pending = any(a['checkpoint_step'] in (ck, None)
                                       and not a['valid_native_outcome']
                                       and not a['protocol_verified'] for a in attempts)
        seeds = {a['seed'] for a in rows if integer(a['seed'])}
        valid = [a for a in rows if a['valid_native_outcome']]
        valid_seeds = {a['seed'] for a in valid}
        complete = (not global_issues and not unknown_protocol_pending and bool(expected_seeds) and set(expected_seeds) <= valid_seeds
                    and len(valid) == len(rows) and seeds <= set(expected_seeds))
        successes = sum(a['native_success'] is True for a in valid)
        groups.append({'role': role, 'checkpoint_step': ck, 'checkpoint_sha256': sha, 'protocol_id': pid,
                       'protocol': rows[0]['protocol'], 'expected_seeds': expected_seeds,
                       'observed_seeds': sorted(seeds), 'valid_seeds': sorted(valid_seeds),
                       'missing_valid_seeds': sorted(set(expected_seeds) - valid_seeds),
                       'attempts': len(rows), 'valid_attempts': len(valid), 'native_successes': successes,
                       'complete_coverage': complete, 'unknown_protocol_pending_blocks_rate': unknown_protocol_pending,
                       'native_success_rate': successes / len(rows) if complete else None,
                       'rate_denominator': 'all compatible attempts including explicit references and repeats; never best-per-seed',
                       'role_counts': {r: sum(a['role'] == r for a in rows) for r in roles},
                       'category_counts': {c: sum(a['category'] == c for a in rows) for c in sorted({a['category'] for a in rows})},
                       'by_seed': [{'seed': seed, 'attempt_ids': [a['attempt_id'] for a in rows if a['seed'] == seed],
                                    'outcomes': [a['category'] for a in rows if a['seed'] == seed]}
                                   for seed in sorted(seeds)]})
    return groups


def summarize(run_dirs=(), queue_manifest=None, expected_seeds=None):
    entries, queue, qr = collect_entries(run_dirs, queue_manifest)
    expected = expected_seeds if expected_seeds is not None else queue.get('expected_seeds', DEFAULT_SEEDS)
    if not isinstance(expected, list) or not expected or not all(integer(s) for s in expected) or len(set(expected)) != len(expected):
        raise ValueError('expected_seeds must be a nonempty unique list of nonnegative integers')
    attempts, files = [], list(qr.files)
    for entry in entries:
        rows, read_files = scan_run(entry)
        attempts.extend(rows)
        files.extend(read_files)
    return {'schema': 'robotwin-checkpoint-sweep-summary-v1',
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'development comparison only; not generalization; native != strict or competition success',
            'read_only_cpu': True, 'expected_seeds': expected,
            'rate_policy': 'null for incomplete coverage, any unresolved attempt or input issue; no best retry selection',
            'issues': qr.issues, 'source_files': files,
            'groups': aggregate(attempts, expected, qr.issues), 'attempts': attempts}


def display(value):
    if value is None:
        return 'null'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def markdown(report):
    lines = ['# RoboTwin checkpoint sweep / bottleneck report', '', report['scope'], '',
             f"Expected dev seeds: {report['expected_seeds']}. Generated UTC: {report['generated_at_utc']}",
             '', '**No GPU/model/simulator run. No best-attempt filtering.**',
             'Rates are null unless all expected seeds and every attempt in the same checkpoint/protocol group are valid.',
             '500-step and 700-step protocols are separate; explicit references are labeled and only fill coverage within an identical protocol.',
             'Unknown strict evidence stays null. Stable/released/contained is not a competition-success certificate.', '',
             '## Protocol groups', '',
             '| Role | Checkpoint | Protocol | Renderer | Step cap | Valid/all attempts | Native successes | Coverage | Rate | Missing valid seeds |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for g in report['groups']:
        vals = [g['role'], g['checkpoint_step'], g['protocol_id'],
                (g['protocol']['renderer'] or {}).get('denoiser'), g['protocol']['control_step_limit'],
                f"{g['valid_attempts']}/{g['attempts']}", g['native_successes'], g['complete_coverage'],
                g['native_success_rate'], g['missing_valid_seeds']]
        lines.append('| ' + ' | '.join(map(display, vals)) + ' |')
    lines += ['', '## Every attempt (including failures and not-started jobs)', '',
              '| Checkpoint | Seed | Cap | Category | Reported steps | Wall s | s/step | Native | Stable | Release | Containment | TOPP fallback | Lift max m | Track L/R max |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for a in report['attempts']:
        ev, gt = a['strict_evidence'], a['gt']['rollout']
        vals = [a['checkpoint_step'], a['seed'], a['protocol']['control_step_limit'], a['category'],
                a['control_steps'], a['wall_seconds'], a['wall_seconds_per_control_step'], a['native_success'],
                ev['stable_after_settle'], ev['release_verified'], ev['full_containment'],
                a['actions']['topp_fallback_actions'], gt['max_object_rise_m'],
                f"{display(gt['tracking']['left']['max'])}/{display(gt['tracking']['right']['max'])}"]
        lines.append('| ' + ' | '.join(map(display, vals)) + ' |')
    lines += ['', '## Evidence and bottleneck observations', '',
              'Latency is wall time, not scene time. Object rise is actor-origin displacement, not verified grasp/hold.',
              'Tracking uses measured actual_qpos minus simultaneous drive target (native units, no wrapping); missing actual data is null.',
              'TOPP fallback is a diagnostic, not proof of collision safety. Settling statistics are separate from rollout.', '']
    for i, a in enumerate(report['attempts'], 1):
        ev = a['strict_evidence']
        lines += [f"### {i}. ckpt={a['checkpoint_step']} seed={a['seed']} ({a['role']})", '',
                  f"- Run: `{a['run_dir']}`; episode: `{a['episode']}`; unit: `{a['unit']}`.",
                  f"- Status: {a['status']} → **{a['category']}**; parent run={a['run_status']}.",
                  f"- Error: {display(a['error'] or a['run_error'])}; phase={display(a['failure_phase'])}.",
                  f"- Native flag observed={display(a['observed_native_flag'])}; scored native={display(a['native_success'])}.",
                  f"- Strict evidence: `{json.dumps(ev, sort_keys=True)}`.",
                  f"- Action latency policy/control: `{json.dumps(a['actions']['policy_wall_seconds'])}` / `{json.dumps(a['actions']['control_wall_seconds'])}` seconds.",
                  f"- Actions returned/submitted={a['actions']['native_returned_rows']}/{a['actions']['submitted_rows']}; fallback known on {a['actions']['topp_fallback_known_actions']} actions.",
                  f"- GT: `{json.dumps(a['gt'], sort_keys=True)}`."]
        if a['category'] == 'valid_native_success' and a['stable_released_contained'] is not True:
            lines.append('- Bottleneck: native success does not establish stable, released, fully contained success.')
        elif a['category'] == 'valid_step_limit':
            lines.append('- Bottleneck: valid failure at this configured cap, not proof of failure at a larger cap.')
        elif not a['valid_native_outcome']:
            lines.append('- Bottleneck: unscored infrastructure/pending/invalid evidence; do not count as policy failure or success.')
        lines += [f"- Video: `{v['path']}` ({v['bytes']} bytes; presence alone does not certify complete encoding)." for v in a['videos']]
        if not a['videos']:
            lines.append('- Video: unavailable.')
        lines += [f'- Input warning: {display(w)}' for w in a['issues']]
        lines.append('')
    lines += ['## Input issues', ''] + ([f'- {display(i)}' for i in report['issues']] or ['None.'])
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', nargs='+', default=[])
    parser.add_argument('--queue-manifest', type=Path)
    parser.add_argument('--expected-seeds', nargs='+', type=int)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.run_dirs and not args.queue_manifest:
        parser.error('explicit --run-dirs or --queue-manifest required')
    report = summarize(args.run_dirs, args.queue_manifest, args.expected_seeds)
    prefix = args.output_prefix
    targets = [Path(str(prefix) + suffix).resolve() for suffix in ('.json', '.md')]
    inputs = {Path(f['path']).resolve() for f in report['source_files']}
    roots = {Path(a['run_dir']).resolve() for a in report['attempts']}
    if any(t in inputs or any(t.is_relative_to(root) for root in roots) for t in targets):
        parser.error('output must be outside all input run directories and must not overwrite any input file')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, text in [('.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                         ('.md', markdown(report))]:
        target = Path(str(prefix) + suffix)
        # Output only; never touch run inputs. Atomic replacement for readers.
        temp = target.with_name(target.name + '.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(target)
    print(json.dumps({'output_prefix': str(prefix), 'attempts': len(report['attempts']),
                      'groups': len(report['groups']), 'issues': len(report['issues'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
