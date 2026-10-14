#!/usr/bin/env python3
"""Prepare a BLOCKED Task90 development triplet; never launch a process or admit a gate.

Prepare: --version v01 --out <new-directory-under-system_triplet_plan>
Inspect: --check-only <plan.json>  (read-only JSON; exit 2 means BLOCKED)
Optional --config-source is a real JSON orchestrator source, not an admission receipt.
The old checkpoint triplet's seed/identity rules and recorded_dev02 argv are reused;
its main() is deliberately not called (it starts GPU work and edits progress.md).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from gpu_lock import LOCK, owner
from wam_v3_resource_authorization import RECEIPT, validate_authorization

ROOT = Path(__file__).resolve().parents[1]
PLAN_ROOT = ROOT / 'outputs/validation/task090_completion_20261003_0156/system_triplet_plan'
RUN_ROOT = ROOT / 'outputs/behavior/closed_loop'
RECORDED = ROOT / 'outputs/validation/task090_completion_20261002_1840/recorded_dev02'
LAUNCH = RECORDED / 'launch.json'
LAUNCH_SHA = 'f79cce8d45f987b6ff8b93ec27d678824f7493652a6d6aced9f8a7c7bfa8adc4'
OLD_RUN = RUN_ROOT / 'task090_s035k_dev10_seed1_privbase_preposition_onboard_hd_20261002'
CHECKPOINT = ROOT / 'outputs/training/task090_v3_50k_20260930_s035000/checkpoint_step_035000.pt'
CHECKPOINT_SHA = '902a2a6fa7d97bf2a0f89530b6d24994725d68c2265a523bef1ad6248a958cb9'
SIDECAR_SHA = '4f03349c855ce402fedbe1a03891f4ccdd8bef3938c0d27d61b3eb0c894c9e89'
EVAL_ROOT = ROOT / 'third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/eval'
TASK_STATS = ROOT / 'data/omnigibson/2026-challenge-task-instances/metadata/task.jsonl'
GOALS = ['(inside ?half__banana.n.01_1 ?ashcan.n.01_1)',
         '(inside ?half__pomegranate.n.01_1 ?ashcan.n.01_1)']
BDDL = ROOT / 'third_party/BEHAVIOR-1K/bddl3/bddl/activity_definitions/composting_waste/problem0.bddl'
SEEDS = (0, 1, 2)  # run_task090_checkpoint_triplet.py's fixed ordered triplet
REQUIRED = {
    'observer': {
        'reason': 'CPU lifecycle tests only; real sensor relation producer not admitted.',
        'fields': ['producer_source_path', 'producer_source_sha256', 'policy_source_sha256',
                   'navigation_manipulation_hold_runtime_trace_path', 'raw_rgbd_proprio_action_feedback_index',
                   'episode_target_hand_timestamp_binding_evidence', 'identity_support_grasp_transport_release_evidence',
                   'independent_labels_and_error_bounds_path', 'heldout_validation_path', 'applicability_scope_path']},
    'dense': {
        'reason': 'Dense holdout repair is unfinished; CPU positives are not physical admission.',
        'fields': ['frozen_source_sha256', 'cpu_positive_negative_and_no_leakage_results_path',
                   'registered_21_frame_validation_path', 'independent_review_path']},
    'completion_gate': {
        'reason': 'No independently validated sensor-only producer-to-consumer admission.',
        'fields': ['consumer_source_path', 'consumer_source_sha256', 'integration_config_path',
                   'integration_config_sha256', 'independent_admission_receipt_path',
                   'calibration_and_validation_sources', 'expired_wrong_target_unknown_hold_tests_path',
                   'gt_vtt_timeout_exclusion_audit_path', 'both_inside_and_release_stability_audit_protocol_path']},
    'recording': {
        'reason': 'Old 1080p video failed framing; dimensions/auto setting are not a framing pass.',
        'fields': ['viewer_framing_fix_source_sha256', 'pre_run_framing_review_path',
                   'all_phase_sensor_capture_contract_path', 'disk_space_and_full_budget_storage_estimate_path',
                   'post_run_per_seed_episode_bound_decode_sidecar_and_visual_review_paths']},
    'resources': {
        'reason': 'Lock absence is only a snapshot; live resource/headroom/unit checks remain mandatory.',
        'fields': ['fresh_cpu_ram_gpu_measurements_path', 'combined_policy_simulator_headroom_check_path',
                   'single_gpu_lock_owner_and_atomic_acquisition_receipt', 'active_process_and_systemd_unit_audit_path',
                   'effective_85pct_cpu_ram_15300mib_vram_monitor_contract_path',
                   'per_seed_parent_child_exit_and_owner_only_release_receipts']},
    'main_thread': {
        'reason': 'Preparation only. Main thread must review, freeze, register and dispatch; no auto-ready route.',
        'fields': ['progress_launch_registration_path', 'frozen_config_and_runtime_manifest_path',
                   'config_diff_review_path', 'checkpoint_cached_digest_provenance_review_path',
                   'original_default_budget_and_wall_timeout_review_path', 'public10_instance311_evaluator_plan_paths',
                   'per_seed_effective_seeds_interface_hashes_and_all_results_paths']},
}


def read_json(path):
    return json.loads(Path(path).read_text())


def small_sha(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 2 * 1024**2:
        raise ValueError(f'missing/oversize small provenance source: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def value(argv, flag):
    require(argv.count(flag) == 1, f'expected exactly one {flag}')
    return argv[argv.index(flag) + 1]


def replace(argv, flag, replacement):
    value(argv, flag)
    argv[argv.index(flag) + 1] = str(replacement)


def output_path(path):
    path = Path(path).absolute()
    require(path == path.resolve(), 'output aliases/symlinks are not allowed')
    require(path != PLAN_ROOT and path.is_relative_to(PLAN_ROOT), 'output must be a new child of system_triplet_plan')
    return path


def source_bundle(config_source):
    require(small_sha(LAUNCH) == LAUNCH_SHA, 'recorded_dev02 launch source mismatch')
    launch = read_json(LAUNCH)
    baseline = json.loads(value(launch['argv'], '--orchestrate'))
    require(baseline == launch['orchestrate'], 'recorded config/argv mismatch')
    config = baseline if config_source == LAUNCH else read_json(config_source)
    require(isinstance(config, dict), 'config source must be an orchestrator JSON object')
    # Preserve the registered instance-independent program, budgets and auxiliary boundary.
    for key in ('program', 'train_targets', 'navigation', 'manip_executor', 'candidate_outcome_policy',
                'frozen', 'vlm', 'answer', 'ask_budget', 'max_attempts', 'min_budget_steps',
                '_navigation_budget_recipe', 'skip_failed_objects', 'privileged_base_navigation',
                'onboard_recording', 'manip_snapshot_sensors', 'dock_target_at', 'base_control'):
        require(config.get(key) == baseline[key], f'config source changes protected field: {key}')
    sidecar = Path(str(CHECKPOINT) + '.sha256.json')
    require(small_sha(sidecar) == SIDECAR_SHA, 'cached checkpoint digest source mismatch')
    cached, manifest = read_json(sidecar), read_json(OLD_RUN / 'manifest.json')
    require(not CHECKPOINT.is_symlink() and CHECKPOINT.is_file(), 'fixed checkpoint missing or aliased')
    stat = CHECKPOINT.stat()  # Deliberately never read/load/hash the 888 MB weight.
    require(cached['step'] == 35000 and cached['sha256'] == CHECKPOINT_SHA and cached['size'] == stat.st_size,
            'fixed checkpoint metadata mismatch')
    require(manifest['checkpoint'] == str(CHECKPOINT) and manifest['checkpoint_sha256'] == CHECKPOINT_SHA,
            'recorded checkpoint provenance mismatch')
    require(value(launch['argv'], '--checkpoint') == str(CHECKPOINT), 'launch checkpoint mismatch')
    episode = read_json(OLD_RUN / 'eval/evaluation_plan.json')
    require(episode['instance_indices'] == [10] and episode['instance_ids'] == [311], 'public dev10 mapping mismatch')
    stats = [json.loads(line) for line in TASK_STATS.read_text().splitlines() if line.strip()]
    task = next(row for row in stats if row['task_index'] == 90)
    utils = (EVAL_ROOT / 'utils/eval_utils.py').read_text()
    require(re.search(r'^EVAL_TIMEOUT_MULTIPLIER = 1\.5\s*$', utils, re.M), 'default budget multiplier changed')
    require(int(task['length'] * 1.5) == 5457, 'full task budget source mismatch')
    require(all(goal in BDDL.read_text() for goal in GOALS), 'original two inside goals changed')
    paths = {LAUNCH, config_source, sidecar, OLD_RUN / 'manifest.json', OLD_RUN / 'eval/evaluation_plan.json',
             TASK_STATS, BDDL, RECEIPT, Path(__file__).resolve(),
             ROOT / 'scripts/run_task090_checkpoint_triplet.py', ROOT / 'scripts/gpu_lock.py',
             ROOT / 'scripts/wam_v3_resource_authorization.py', ROOT / 'scripts/behavior_resources.py',
             EVAL_ROOT / 'utils/eval_utils.py', EVAL_ROOT / 'utils/score_utils.py',
             EVAL_ROOT / 'utils/recording_utils.py', ROOT / 'harness/dense_rgbd_registration.py',
             ROOT / 'harness/manipulation_evidence.py'}
    # Reuse the launcher's existing manifest inventory, only hashing small source/config files.
    paths.update(Path(p) for p in manifest['artifact_sha256'] if Path(p).suffix in ('.py', '.json', '.yaml'))
    for p in (config['program'], config['train_targets']):
        require(str(Path(p).resolve()) == p, 'config file source must use its canonical absolute path')
    snapshots = {str(p): small_sha(p) for p in sorted(paths)}
    return launch, config, snapshots, {
        'path': str(CHECKPOINT), 'step': 35000, 'sha256': CHECKPOINT_SHA,
        'digest_source': str(sidecar), 'corroborating_manifest': str(OLD_RUN / 'manifest.json'),
        'size': stat.st_size, 'mtime_ns_at_preparation': stat.st_mtime_ns,
        'verification': 'REUSED_CACHED_DIGEST_PLUS_STAT; NOT_REHASHED_OR_MODEL_LOADED',
    }


def build_plan(version, out, config_source=LAUNCH):
    require(re.fullmatch(r'[a-z][a-z0-9_]{0,31}', version) is not None, 'version must be a short lowercase tag')
    out, config_source = output_path(out), Path(config_source).resolve()
    require(config_source.is_relative_to(ROOT), 'config source must be a local project JSON file')
    launch, config, sources, checkpoint = source_bundle(config_source)
    resource = validate_authorization(str(RECEIPT))
    config_version = digest({'config': config, 'sources': sources})
    rows = []
    for seed in SEEDS:
        name = f'task090_s035k_dev10_seed{seed}_privbase_preposition_assisted_hd_full_{version}'
        unit = f'task090-system-{version}-seed{seed}'
        cmd = [x.replace(launch['unit'], unit) for x in launch['argv']]
        for flag, new in (('--run-name', name), ('--seed', seed), ('--checkpoint', CHECKPOINT),
                          ('--orchestrate', json.dumps(config))):
            replace(cmd, flag, new)
        value(cmd, '--max-steps')
        index = cmd.index('--max-steps')
        del cmd[index:index + 2]  # Retain the evaluator's original full-task default, not a prefix.
        cmd = [('StandardOutput=append:' + str(out / f'seed{seed}_unit.log'))
               if item.startswith('StandardOutput=append:') else item for item in cmd]
        rows.append({'seed': seed, 'public_index': 10, 'expected_instance_id': 311,
                     'unit': unit, 'run_dir': str(RUN_ROOT / name), 'status': 'NOT_STARTED',
                     'config_version_sha256': config_version, 'result': None,
                     'argv_for_main_thread_review_only': cmd})
    plan = {
        'schema': 'task090-system-triplet-preparation-v1', 'status': 'BLOCKED', 'execution': 'NOT_STARTED',
        'version': version, 'out': str(out), 'config_source': str(config_source),
        'config_source_sha256': sources[str(config_source)], 'source_sha256': sources,
        'config_version_sha256': config_version, 'orchestrate': config, 'checkpoint': checkpoint,
        'selection': 'PRESELECTED_35K_DEVELOPMENT_CANDIDATE; not random-best/report/generalization',
        'protocol': {'seeds': list(SEEDS), 'profile': 'development', 'task': 'composting_waste',
                     'public_index': 10, 'expected_instance_id': 311, 'samples': 1, 'exec_horizon': 8,
                     'num_inference_steps': 10, 'max_steps_override': None,
                     'original_default_timeout_steps': 5457, 'wall_timeout_seconds': 1800,
                     'budget_source': str(TASK_STATS), 'budget_formula': 'int(3638.62 * 1.5) = 5457',
                     'budget_note': 'No max-steps override. Docs 5458 is consistent with actual timeout+1 steps; preserve actual counts. Neither 600 nor 2323 is full budget. Wall timeout is an abort, never completion.',
                     'goals': GOALS, 'online_manipulation_completion': 'SENSOR_ONLY; NOT_ADMITTED',
                     'evaluation_gt': 'Evaluation-only success/Q/object-pose audit, never policy/gate input.',
                     'retention': 'Keep all three successes/failures/not-reached/aborts. Report x/3 full-task results; do not require 3/3 or select best.',
                     'serial': 'One GPU run at a time; no queue, no auto-dispatch or retry in this tool.'},
        'auxiliary': {'privbase': True, 'preposition': True, 'assisted_grasp': True,
                      'official_score_valid': False, 'isaac_port': '6.1',
                      'base_odometry': 'IntervalBase simulated encoder; not verified hardware odometry',
                      'claim_boundary': 'Assisted development full-budget attempts only; not autonomous navigation, official score, or real friction grasp.'},
        'recording': {'write_video': True, 'view': 'viewer', 'width': 1920, 'height': 1080,
                      'framing': 'auto', 'framing_review': 'BLOCKED; old recorded_dev02 framing failed',
                      'onboard': 'all policy-received phases; do not invent unsent terminal observation',
                      'reuse': str(ROOT / 'scripts/check_closed_loop_video.py'),
                      'post_run': 'Per seed full decode, episode binding, frame count=actual steps+1, sidecar and visual framing review; no extra showcase rerun.'},
        'resource_contract': resource,
        'reuse': {'triplet_reference': str(ROOT / 'scripts/run_task090_checkpoint_triplet.py'),
                  'launch_template': str(LAUNCH), 'launcher': str(ROOT / 'scripts/behavior_closed_loop.py'),
                  'note': 'Reused fixed triplet identity and argv; old main forbids video and writes progress, so is never invoked.'},
        'stop_and_release': 'Main thread stops parent unit, confirms its wam-policy-PID and behavior-run-PID children exited, then releases only that owner via gpu_lock.py; never unlink a foreign lock.',
        'main_thread_evidence_required': {k: {'reason': v['reason'], 'evidence': dict.fromkeys(v['fields'])}
                                          for k, v in REQUIRED.items()},
        'rounds': rows, 'completed': [],
    }
    plan['plan_sha256'] = digest(plan)
    return plan


def runtime_snapshot():
    """Read /proc and existing lock only; no telemetry probe, GPU import or lock mutation."""
    names = {'behavior_closed_loop.py', 'policy_server.py', 'omnigibson.eval.eval',
             'train_wam_task_v3.py', 'train_wam_v3.py', 'train_lilawam_behavior.py',
             'run_task090_checkpoint_triplet.py', 'run_task090_remaining_stages.py',
             'run_task090_training_first.py', 'run_wam_v3_stages.py'}
    active, unreadable = [], []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        except FileNotFoundError:
            continue
        except PermissionError:
            unreadable.append(int(proc.name))
            continue
        if int(proc.name) != os.getpid() and any(Path(arg).name in names for arg in argv):
            active.append({'pid': int(proc.name), 'argv': argv})
    return {'gpu_lock_path': str(LOCK), 'lock_present': LOCK.exists() or LOCK.is_symlink(),
            'lock_owner': owner(), 'active_runtime_processes': active, 'unreadable_pids': unreadable,
            'resource_headroom': 'NOT_CHECKED; main thread must collect live measurements'}


def inspect_plan(plan, snapshot=None):
    errors = []
    try:
        expected = build_plan(plan['version'], plan['out'], Path(plan['config_source']))
        for key in sorted(set(plan) | set(expected)):
            if plan.get(key) != expected.get(key):
                errors.append(f'plan/source mismatch: {key}')
        for saved in PLAN_ROOT.rglob('plan.json'):
            if saved == Path(expected['out']) / 'plan.json':
                continue
            other = read_json(saved)
            if other.get('schema') != expected['schema']:
                continue
            reserved = {r['run_dir'] for r in other['rounds']} | {r['unit'] for r in other['rounds']}
            if any(r['run_dir'] in reserved or r['unit'] in reserved for r in expected['rounds']):
                errors.append(f'run/unit already reserved by another plan: {saved}')
        for row in expected['rounds']:
            run = Path(row['run_dir'])
            if run.exists() or run.is_symlink():
                errors.append(f'existing run output: {run}')
            log = Path(expected['out']) / f"seed{row['seed']}_unit.log"
            if log.exists() or log.is_symlink():
                errors.append(f'existing run log: {log}')
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        errors.append(f'invalid provenance/plan: {exc}')
    snapshot = runtime_snapshot() if snapshot is None else snapshot
    if snapshot['lock_present'] or snapshot['active_runtime_processes'] or snapshot['unreadable_pids']:
        errors.append('existing GPU lock/runtime or incomplete process visibility; refuse preparation/dispatch')
    return {'status': 'BLOCKED', 'execution': 'NOT_STARTED',
            'plan_validation': 'REJECTED' if errors else 'VALID_PREPARATION_ONLY',
            'errors': errors, 'blocked_on': {k: v['reason'] for k, v in REQUIRED.items()},
            'runtime_snapshot': snapshot, 'plan_sha256': plan.get('plan_sha256'),
            'note': 'No supplied config booleans or evidence strings can make this tool READY. CPU checks do not admit manipulation.'}


def write_new_plan(plan):
    out = output_path(plan['out'])
    require(not out.exists() and not out.is_symlink(), 'existing output directory; never overwrite or resume')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'plan.json').open('x') as file:
        file.write(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
    return out / 'plan.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--config-source', type=Path, default=LAUNCH)
    parser.add_argument('--check-only', type=Path, metavar='PLAN_JSON')
    args = parser.parse_args()
    try:
        if args.check_only:
            require(args.out is None and args.version is None and args.config_source == LAUNCH,
                    'check-only accepts only the saved plan path')
            path = args.check_only.resolve()
            require(path.is_relative_to(PLAN_ROOT), 'check-only plan must be under system_triplet_plan')
            plan = read_json(path)
            require(path == Path(plan['out']) / 'plan.json', 'plan/output source mismatch')
            report = inspect_plan(plan)
        else:
            require(args.version is not None and args.out is not None, '--version and --out required to prepare')
            out = output_path(args.out)
            require(not out.exists() and not out.is_symlink(), 'existing output directory; never overwrite or resume')
            plan = build_plan(args.version, out, args.config_source)
            report = inspect_plan(plan)
            if not report['errors']:
                report['plan_path'] = str(write_new_plan(plan))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(json.dumps({'status': 'BLOCKED', 'execution': 'NOT_STARTED', 'plan_validation': 'REJECTED',
                          'errors': [str(exc)]}, ensure_ascii=False))
    return 2  # Intentionally nonzero even for a structurally valid, saved plan.


if __name__ == '__main__':
    sys.exit(main())
