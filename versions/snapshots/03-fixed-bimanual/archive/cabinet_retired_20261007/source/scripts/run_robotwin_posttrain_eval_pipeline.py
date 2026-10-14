#!/usr/bin/env python3
"""CPU supervisor: wait for successful full 30k, then pure-WAM 25k/30k paired five.

No training is launched here. Without --execute this prints a CPU plan only.
Selection audit JSON: {"complete": true, "selection_seeds": [all prior physical
smoke/development/selection seeds], "note": "scope and provenance of the list"}.
This is a user attestation, not a claim that prior selection can be inferred.
Dataset seed.txt files (all splits), optimizer/validation RNG seeds and the audit
are excluded. Exactly the same five seeds are used at both milestones. An invalid
initialization may retry the SAME seed in a new directory, never an easier seed.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.robotwin_policy_worker import atomic_json, digest, identity, sha256, training_contract
from scripts import run_robotwin_checkpoint_queue as queue
from scripts.eval_robotwin_wam import CONFIG_DIR, ROBOTWIN, dataset_seed_inventory, validate_task
from scripts.gpu_lock import owner as gpu_owner


class TrainingFailed(RuntimeError):
    pass


def training_gate(run, unit, *, state_reader=queue.unit_state, owner_reader=gpu_owner):
    """Presence of 30k checkpoint or an inactive unit is NEVER sufficient."""
    status = queue.read_result(Path(run)/'status.json')
    state = state_reader(unit)
    if status.get('status') in ('stopped', 'failed'):
        raise TrainingFailed('training did not finish successfully: ' + str(status))
    if state.get('ActiveState') == 'failed' or state.get('Result') in ('timeout', 'exit-code', 'signal', 'oom-kill', 'core-dump'):
        raise TrainingFailed('training unit failed: ' + str(state))
    terminal = status.get('status') in ('complete', 'bounded_complete')
    if terminal and (status.get('step') != 30000 or status.get('target') != 30000):
        raise TrainingFailed('requires completed target=step=30000, not a smoke/partial/larger run')
    if not terminal:
        return False, {'reason': 'waiting_for_training_status', 'training_status': status, 'unit': state}
    exited = (state.get('ActiveState'), state.get('SubState')) in (('inactive', 'dead'), ('active', 'exited'))
    clean = (state.get('LoadState') == 'loaded' and exited and state.get('Result') == 'success'
             and state.get('ExecMainCode') == '1' and state.get('ExecMainStatus') == '0'
             and state.get('MainPID') == '0')
    if not clean:
        return False, {'reason': 'waiting_for_confirmed_clean_unit_exit', 'training_status': status, 'unit': state}
    lock = owner_reader()
    if lock is not None:
        return False, {'reason': 'waiting_for_gpu_lock_release', 'lock_owner': lock, 'unit': state}
    return True, {'training_status': status, 'unit': state, 'lock_owner': None, 'checked_at': time.time()}


def seed_audit(config, seeds, path):
    audit = json.loads(Path(path).read_text())
    if audit.get('complete') is not True or not isinstance(audit.get('note'), str) or not audit['note'].strip():
        raise ValueError('selection seed audit requires complete=true and a provenance note')
    selected = audit.get('selection_seeds')
    if not isinstance(selected, list) or any(type(s) is not int or not 0 <= s < 2**32 for s in selected):
        raise ValueError('selection_seeds must list uint32 prior smoke/development/selection seeds')
    excluded = set(selected)
    for key in ('seed', 'validation_seed'):
        if key in config.get('training', {}):
            excluded.add(config['training'][key])
    if excluded.intersection(seeds):
        raise ValueError('evaluation seeds overlap prior selection/training RNG seeds')
    return {'audit_file': identity(path), 'attestation': audit, 'excluded_seeds': sorted(excluded)}


def build_manifest(args, config):
    import yaml
    task = validate_task(config['dataset']['task_name'])
    run, out = Path(args.training_run_dir).resolve(), Path(args.run_dir).resolve()
    steps = yaml.safe_load((CONFIG_DIR/'_eval_step_limit.yml').read_text())
    max_steps = steps.get(task)
    if type(max_steps) is not int or not 1 <= max_steps <= 2000:
        raise ValueError('missing/unsupported native step limit for ' + task)
    prefix = 'rw-post-' + digest(str(out))[:12]
    jobs = [dict(unit=f'{prefix}-{step}-s{seed}', run_dir=str(out/f'step_{step:06d}'/f'seed_{seed}'),
                 training_run_dir=str(run), checkpoint=str(run/f'checkpoint_step_{step:06d}.pt'),
                 checkpoint_step=step, seed=seed, task=task, system_mode='pure', execution_mode='native8',
                 task_config=args.task_config or config['dataset'].get('task_config'),
                 record_onboard=True, camera_readback=args.camera_readback, max_steps=max_steps, settle_steps=0)
            for step in (25000, 30000) for seed in args.seeds]
    sources = [Path(__file__), ROOT/'scripts/eval_robotwin_wam.py', ROOT/'scripts/run_robotwin_checkpoint_queue.py', ROOT/'scripts/gpu_lock.py']
    sources += sorted((ROOT/'harness').glob('*.py')) + sorted((ROBOTWIN/'envs').rglob('*.py'))
    sources += sorted(CONFIG_DIR.glob('*.yml'))
    manifest = dict(protocol='pure_wam_30k_paired5_v1', task=task, training_run_dir=str(run),
        training_unit=args.training_unit, training_unit_retention='RemainAfterExit=yes',
        training_config=identity(args.training_config),
        seeds=args.seeds, jobs=jobs, max_invalid_retries=args.max_invalid_retries, require_video=True,
        created_at=time.time(), source_sha256={str(p.resolve()): sha256(p) for p in sources},
        selection_audit=seed_audit(config, args.seeds, args.selection_seeds_file),
        sampling='fixed paired seeds; same-seed initialization retries only; no score-based selection',
        scoring='native task latch; interruption/invalid initialization unscored; incomplete rate is null')
    return manifest


def verify_frozen(manifest):
    items = dict(manifest['source_sha256'])
    for entry in (manifest['training_config'], manifest['selection_audit']['audit_file']):
        items[entry['path']] = entry['sha256']
    for path, expected in items.items():
        if sha256(path) != expected:
            raise ValueError('preregistered source/config/audit changed: ' + path)


def checkpoint_admission(manifest):
    receipts, excluded, seed_files = [], set(manifest['selection_audit']['excluded_seeds']), set()
    for step in (25000, 30000):
        ck = Path(manifest['training_run_dir'])/f'checkpoint_step_{step:06d}.pt'
        contract, cache, _, receipt = training_contract(manifest['training_run_dir'], ck)
        if contract['config']['dataset']['task_name'] != manifest['task']:
            raise ValueError('training manifest task differs from preregistered task')
        if contract.get('config_sha256') != manifest['training_config']['sha256']:
            raise ValueError('training manifest configuration differs from preregistration')
        if receipt.get('step') != step or receipt.get('model_only') is not True or sha256(ck) != receipt.get('sha256'):
            raise ValueError('milestone receipt step/model-only/SHA256 mismatch')
        seeds, files = dataset_seed_inventory(cache)
        excluded.update(seeds)
        seed_files.update(files)
        receipts.append({'checkpoint': identity(ck, content=False), 'receipt': receipt})
    overlap = excluded.intersection(manifest['seeds'])
    if overlap:
        raise ValueError('evaluation seeds overlap dataset/selection/training provenance: ' + str(sorted(overlap)))
    return {'checkpoints': receipts, 'excluded_seeds': sorted(excluded),
            'seed_files': [identity(p) for p in sorted(seed_files)], 'seed_overlap': []}


def wait_ready(args, manifest, state, out):
    deadline = time.monotonic() + args.wait_seconds
    while True:
        verify_frozen(manifest)
        ready, evidence = training_gate(manifest['training_run_dir'], manifest['training_unit'])
        state.update(status='waiting_for_training_and_gpu', gate=evidence)
        atomic_json(out/'pipeline_status.json', state)
        if ready:
            return evidence
        if time.monotonic() >= deadline:
            raise TimeoutError('training/GPU admission wait expired; no policy failures scored')
        time.sleep(args.poll_seconds)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--training-config', required=True)
    ap.add_argument('--training-run-dir', required=True)
    ap.add_argument('--training-unit', required=True, help='training unit with RemainAfterExit=yes so its successful exit stays auditable')
    ap.add_argument('--run-dir', required=True, help='new archive directory; never resumed or overwritten')
    ap.add_argument('--seeds', nargs=5, type=int, required=True)
    ap.add_argument('--selection-seeds-file', required=True)
    ap.add_argument('--task-config')
    ap.add_argument('--camera-readback', choices=('native', 'serial'), default='native')
    ap.add_argument('--max-invalid-retries', type=int, choices=range(4), default=1)
    ap.add_argument('--poll-seconds', type=float, default=30)
    ap.add_argument('--wait-seconds', type=float, default=172800)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    if len(set(args.seeds)) != 5 or any(not 0 <= s < 2**32 for s in args.seeds):
        ap.error('five distinct uint32 seeds required')
    if not 0 < args.poll_seconds <= 300 or not 0 < args.wait_seconds <= 604800:
        ap.error('bounded positive poll/wait required (max wait seven days)')
    return args


def main(argv=None):
    import yaml
    args = parse_args(argv)
    config = yaml.safe_load(Path(args.training_config).read_text())
    manifest = build_manifest(args, config)
    if not args.execute:
        print(json.dumps({'status': 'cpu_plan_only_no_jobs_started', 'manifest': manifest,
                          'commands_after_training_success': [queue.command(j) for j in manifest['jobs']]}, indent=2))
        return 0
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out/'manifest.json'
    atomic_json(manifest_path, manifest)
    state = {'status': 'preregistered', 'pid': os.getpid(), 'counts': queue.counts(manifest['jobs'], [])}
    previous = {sig: signal.signal(sig, queue.stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    code = 1
    try:
        gate = wait_ready(args, manifest, state, out)
        admission = checkpoint_admission(manifest)
        atomic_json(out/'training_admission.json', {'gate': gate, **admission})
        def before_job():
            wait_ready(args, manifest, state, out)
            # Receipts were fully hashed at admission; the worker rehashes tensors.
            for record in admission['checkpoints']:
                ck = Path(record['checkpoint']['path'])
                if (queue.read_result(ck.with_name(ck.name + '.sha256.json')) != record['receipt']
                        or ck.stat().st_size != record['checkpoint']['size']):
                    raise ValueError('admitted checkpoint/receipt changed')
            # Dataset seed provenance cannot change between the paired checkpoints.
            for record in admission['seed_files']:
                if sha256(record['path']) != record['sha256']:
                    raise ValueError('dataset seed provenance changed after admission')
        code = queue.run_manifest(manifest, manifest_path, before_job=before_job)
        result = queue.read_result(out/'queue_status.json')
        state.update(status=result.get('status', 'incomplete'), counts=result.get('counts', state['counts']),
                     error=result.get('error'))
    except BaseException as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
    finally:
        state['finished_at'] = time.time()
        atomic_json(out/'pipeline_status.json', state)
        # Index keeps videos/episode evidence in their original per-attempt paths.
        archive = [str(p.relative_to(out)) for p in sorted(out.rglob('*')) if p.is_file()]
        atomic_json(out/'archive_index.json', {'status': state['status'], 'counts': state['counts'],
                    'files': archive, 'no_best_checkpoint_selection': True})
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
