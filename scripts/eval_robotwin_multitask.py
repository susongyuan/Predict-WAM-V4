#!/usr/bin/env python3
"""Fixed 5-task x 10-seed joint WAM V3+VTT evaluation, never video recording.

Default is CPU preflight; --execute starts a serial isolated episode queue.
The pipeline must already hold --gpu-lock-owner. Only native predicates score.
No expert seed filtering, model self-scoring, failure replacement, or retries.
--multitask-manifest accepts trainer input tasks JSON or output manifest.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.robotwin_policy_worker import InfrastructureError, PolicyClient, atomic_json, digest, identity
from harness.robotwin_multitask_policy_worker import load_contract, select_condition, verify_gpu_owner
from scripts import eval_robotwin_wam as native
from scripts.eval_robotwin_novideo import disabled_factory

SEEDS = tuple(range(922001, 922011))
STEP_LIMITS = native.CONFIG_DIR / '_eval_step_limit.yml'
WORKER = ROOT / 'harness/robotwin_multitask_policy_worker.py'


class QueueCancelled(BaseException):
    pass


def on_signal(signum, frame):
    raise QueueCancelled(f'signal {signum}')


def official_step_limit(task):
    import yaml
    value = yaml.safe_load(STEP_LIMITS.read_text()).get(task)
    if type(value) is not int or not 1 <= value <= 2000:
        raise InfrastructureError('missing/invalid official native step limit: ' + task)
    return value


def wall_budgets(steps):
    episode = max(1200, steps * 3)
    return dict(episode_wall_seconds=episode, total_wall_seconds=episode + 300,
                external_wall_seconds=episode + 360)


def native_args(args, task, seed=None):
    steps = official_step_limit(task)
    budgets = wall_budgets(steps)
    return argparse.Namespace(checkpoint=args.checkpoint, training_run_dir=str(Path(args.checkpoint).parent),
        multitask_manifest=args.multitask_manifest, task=task,
        gpu_lock_owner=args.gpu_lock_owner, seeds=[seed] if seed is not None else list(SEEDS),
        task_config=None, system_mode='pure', execution_mode='native8', target_head=None,
        vtt_adapter=None, vtt_bank=None, record_onboard=True, max_steps=steps, settle_steps=0,
        startup_timeout=args.startup_timeout, policy_timeout=args.policy_timeout,
        episode_wall_seconds=budgets['episode_wall_seconds'], total_wall_seconds=budgets['total_wall_seconds'],
        camera_readback=args.camera_readback, render_denoiser=args.render_denoiser,
        video_fps=30, ffmpeg=None)


def asset_evidence():
    assets = json.loads(native.ASSET_RECEIPT.read_text())
    if assets.get('status') != 'complete':
        raise InfrastructureError('downloaded native asset receipt incomplete')
    links = {str(p): str(p.resolve()) for p in (native.ROBOTWIN / 'assets').iterdir() if p.is_symlink()}
    if any(not Path(p).is_dir() for p in links.values()):
        raise InfrastructureError('broken native asset link')
    inventories = []
    for archive in assets['archives'].values():
        if not archive.get('file_inventory'):
            continue
        path = Path(archive['file_inventory'])
        inventories.append(identity(path))
        for line in path.read_text().splitlines():
            row = json.loads(line)
            target = Path(assets['asset_root']) / row['path']
            if not target.is_file() or target.stat().st_size != row['bytes']:
                raise InfrastructureError('missing/size-changed asset: ' + str(target))
    return {'asset_receipt': identity(native.ASSET_RECEIPT), 'asset_revision': assets['revision'],
            'asset_inventories': inventories, 'asset_links': links,
            'asset_assurance': 'existing receipt + live file/size inventory, not fresh full asset hash'}


def source_task_config(source):
    """Derive only from explicit native config or official archive basename.

    RoboTwin1 converted SOURCE receipts omit task_config but pin demo_clean.zip.
    Do not mutate SOURCE/cache to add a field: this derivation is eval metadata.
    """
    explicit = source.get('task_config')
    archive = source.get('archive')
    inferred = None
    if archive is not None:
        if not isinstance(archive, str):
            raise InfrastructureError('invalid source archive')
        name = Path(archive).name
        if name in ('demo_clean.zip', 'demo_randomized.zip'):
            inferred = name[:-4]
        elif explicit is None:
            raise InfrastructureError('cannot derive task config from source archive: ' + name)
    if explicit is not None and explicit not in ('demo_clean', 'demo_randomized'):
        raise InfrastructureError('unsupported source task config')
    if explicit and inferred and explicit != inferred:
        raise InfrastructureError('source task_config conflicts with archive provenance')
    config = explicit or inferred
    if config is None:
        raise InfrastructureError('source needs task_config or recognized official demo archive')
    return config


def task_plan(args, bundle, task, seed=None, assets=None):
    native.validate_task(task)
    if task not in bundle['caches']:
        raise InfrastructureError('task not trained by this checkpoint: ' + task)
    cache_info = bundle['caches'][task]
    cache = cache_info['value']
    config_name = source_task_config(cache['source'])
    kwargs, config_files = native.load_task_config(config_name, task)
    kwargs['render_denoiser'] = args.render_denoiser
    original, seed_files = native.dataset_seed_inventory(cache)
    # Always validate all 10 planned seeds, including in a single-episode child.
    overlap = set(SEEDS) & original
    if overlap:
        raise InfrastructureError(f'{task} evaluation seeds overlap source dataset: {sorted(overlap)}')
    contract = bundle['contract']
    settings = native_args(args, task, seed)
    policy = {'views': ['head_camera', 'left_camera', 'right_camera'], 'state_dim': 14,
        'state_semantics': 'commanded drive targets + stored gripper commands, not measured qpos',
        'K': 1, 'euler_steps': 10, 'chunk': 32, 'execute_prefix': 8,
        'condition': 'embedded_TRAIN_task_prototype', 'normalization': 'shared_all_TRAIN_frames',
        'projection': 'jointly_trained_task_cond_proj.residual.weight', 'dino_frozen': True,
        'middle_enabled': False, 'vlm_enabled': False, 'ask_enabled': False,
        'memory_updates_enabled': False, 'recovery_enabled': False, 'gt_control': False,
        'history': 'reset every episode; encode/push all observations, including queued actions'}
    files = [Path(__file__), WORKER, ROOT / 'scripts/eval_robotwin_wam.py',
        ROOT / 'scripts/eval_robotwin_novideo.py', ROOT / 'harness/robotwin_policy_worker.py',
        ROOT / 'harness/robotwin_eval_evidence.py', ROOT / 'harness/robotwin_onboard_recorder.py',
        ROOT / 'harness/robotwin_qpos_adapter.py', ROOT / 'harness/robotwin_camera_readback.py']
    return {'status': 'cpu_plan_ready_not_physical_validation', 'task': task, 'task_config': config_name,
        'task_config_derivation': {'explicit': cache['source'].get('task_config'),
            'archive': cache['source'].get('archive'), 'resolved': config_name, 'source_modified': False},
        'task_kwargs': kwargs, 'seeds': settings.seeds, 'seed_files': [identity(p) for p in sorted(seed_files)],
        'seed_overlap': [], 'task_specs': bundle['tasks'], 'checkpoint': identity(args.checkpoint, content=False),
        'checkpoint_receipt': bundle['receipt'], 'training_manifest': identity(bundle['manifest']),
        'contract_sha256': digest(contract), 'condition_bank_sha256': digest(contract['condition_bank']),
        'statistics_sha256': digest(contract['statistics']),
        'task_condition_sha256': hashlib.sha256(select_condition(contract, task).tobytes()).hexdigest(),
        'cache_config': identity(cache_info['path']),
        'initialization_disclosure': contract.get('initialization_disclosure'),
        'model_constructed': False, 'simulator_started': False, 'cuda_initialized': False,
        'recording': {'enabled': False, 'third_view_camera_created': False, 'video_files_created': 0},
        'onboard_archive': {'enabled': True, 'lossless_native_policy_inputs': True, 'not_video_recording': True},
        'native_success_predicate': f'envs.{task}.{task}.check_success',
        'score_source': 'native env.eval_success latch plus terminal native check_success evidence; never worker self-score',
        'system': {'mode': 'pure', 'execution_mode': 'native8', 'memory_writes_enabled': False},
        'camera_readback': args.camera_readback,
        'renderer': {'denoiser': args.render_denoiser, 'native_default': 'oidn',
                     'protocol_deviation': args.render_denoiser != 'oidn'},
        'limits': {'control_steps': settings.max_steps, **wall_budgets(settings.max_steps),
                   'startup_timeout_seconds': settings.startup_timeout, 'policy_timeout_seconds': settings.policy_timeout},
        'official_step_limit_source': identity(STEP_LIMITS), 'policy': policy,
        'source_files': [identity(p) for p in files] + [identity(p) for p in sorted((native.ROBOTWIN / 'envs').rglob('*.py'))],
        'config_files': [identity(p) for p in config_files], **(assets if assets is not None else asset_evidence()),
        'protocol_deviations': ['fixed fresh seeds; no expert filtering or replacement',
            'single joint warm-start checkpoint, not a baseline comparison',
            'K1/10Euler/chunk32/native8; physical bounds retained and raw actions archived',
            'serial onboard camera readback and denoiser none are explicitly selected',
            'native task success is not an independent containment/release/hold verifier',
            'one isolated process per seed; infra/timeout remain unscored, never counted as model failures']}


def make_client(args, out):
    command = [str(ROOT / '.venvs/wam-fixed/bin/python'), '-B', str(WORKER),
        '--checkpoint', args.checkpoint, '--multitask-manifest', args.multitask_manifest,
        '--task', args.task, '--gpu-lock-owner', args.gpu_lock_owner]
    return PolicyClient(command, Path(out) / 'policy_worker.log', args.startup_timeout, args.policy_timeout)


def run_episode(args):
    """Internal isolated child; no first-frame branch, legacy adapter, or camera."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    journal = native.PhaseJournal(out)
    try:
        bundle = journal.call('preflight_joint_contract', load_contract, args.checkpoint, args.multitask_manifest)
        plan = journal.call('preflight_plan', task_plan, args, bundle, args.episode_task, args.episode_seed)
        atomic_json(out / 'plan.json', plan)
        settings = native_args(args, args.episode_task, args.episode_seed)
        verify_gpu_owner(args.gpu_lock_owner)
        os.chdir(native.ROBOTWIN)
        result = native.run_evaluation(settings, plan, out, client_factory=make_client,
            video_factory=disabled_factory, owner_check=verify_gpu_owner, journal=journal)
        return 0 if result['status'] == 'completed' else 2
    except BaseException as exc:
        if journal.state['status'] != 'error':
            journal.failed(exc)
        atomic_json(out / 'result.json', {'status': 'infrastructure_error', 'task': args.episode_task,
            'error': f'{type(exc).__name__}: {exc}', 'episodes': [{'seed': args.episode_seed,
                'status': 'infrastructure_error', 'evidence': {'native_benchmark_success': None}}]})
        return 2


def child_command(args, task, seed, out):
    return [str(ROOT / '.venvs/robotwin-sim/bin/python'), '-B', str(Path(__file__).resolve()),
        '--checkpoint', args.checkpoint, '--multitask-manifest', args.multitask_manifest,
        '--out', str(out), '--gpu-lock-owner', args.gpu_lock_owner,
        '--camera-readback', args.camera_readback, '--render-denoiser', args.render_denoiser,
        '--startup-timeout', str(args.startup_timeout), '--policy-timeout', str(args.policy_timeout),
        '--execute', '--episode-task', task, '--episode-seed', str(seed)]


def stop_child(process):
    """Only our child's group. The dedicated worker has parent-death SIGKILL."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def launch_episode(command, log_path, timeout):
    started = time.monotonic()
    env = os.environ.copy()
    for key in ('DISPLAY', 'WAYLAND_DISPLAY'):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    process = None
    with Path(log_path).open('xb') as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                start_new_session=True, cwd=ROOT, env=env)
            try:
                rc = process.wait(timeout=timeout)
                return {'returncode': rc, 'external_timeout': False, 'wall_seconds': time.monotonic() - started}
            except subprocess.TimeoutExpired:
                stop_child(process)
                return {'returncode': process.returncode, 'external_timeout': True,
                        'wall_seconds': time.monotonic() - started}
        finally:
            if process is not None and process.poll() is None:
                stop_child(process)


def verify_native_evidence(native_out, row):
    """CPU postcheck of original evidence, never a substitute scoring predicate."""
    import numpy as np
    native_out = Path(native_out)
    plan = json.loads((native_out / 'plan.json').read_text())
    result = json.loads((native_out / 'result.json').read_text())
    episode = row['episode']
    if Path(episode).name != episode:
        raise InfrastructureError('unsafe episode evidence path')
    directory = native_out / episode
    def require(value, message):
        if not value:
            raise InfrastructureError('evidence verification: ' + message)
    require(json.loads((directory / 'episode.json').read_text()) == row, 'episode/result row mismatch')
    require(plan['task'] == result['task'] and plan['seeds'] == [row['seed']], 'plan task/seed mismatch')
    count = row['control_steps']
    require(type(count) is int and 0 < count <= plan['limits']['control_steps'], 'invalid action count')
    require(row['native_take_action_count'] == count, 'native action acknowledgment count mismatch')
    require(row['native_step_limit'] == plan['limits']['control_steps'], 'native official step limit mismatch')
    require(type(row['evidence']['native_terminal_predicate']) is bool, 'missing native terminal predicate')
    require(type(row['evidence']['native_benchmark_success']) is bool, 'missing native benchmark latch')
    require(row['evidence']['native_predicate'] == plan['native_success_predicate'], 'wrong native predicate')
    if row['status'] == 'step_limit':
        require(count == plan['limits']['control_steps'], 'failure before registered step limit')
    runtime = result['policy_ready']['runtime']
    require(runtime['task'] == plan['task'], 'worker task mismatch')
    for key in ('checkpoint_sha256', 'contract_sha256', 'statistics_sha256', 'condition_bank_sha256', 'task_condition_sha256'):
        expected = plan['checkpoint_receipt']['sha256'] if key == 'checkpoint_sha256' else plan[key]
        require(runtime.get(key) == expected, 'worker identity mismatch: ' + key)
    require(runtime.get('dino_frozen') is True and runtime.get('strict_load') is True, 'strict/frozen policy missing')
    require(plan['recording']['enabled'] is False and plan['onboard_archive']['enabled'] is True, 'recording protocol')
    recording = json.loads((directory / 'recording.json').read_text())
    require(recording.get('enabled') is False and recording.get('video_files_created') == 0
        and recording.get('third_view_camera_created') is False and recording.get('episode') == episode
        and recording.get('seed') == row['seed'], 'no-video receipt invalid')
    require(not any(p.suffix.lower() in ('.mp4', '.avi', '.mov', '.webm', '.mkv') or 'third_view' in p.name.lower()
        for p in native_out.rglob('*') if p.is_file()), 'unexpected recording/third-view artifact')
    actions_path = directory / 'actions.jsonl'
    actions = [json.loads(line) for line in actions_path.read_text().splitlines() if line.strip()]
    require(len(actions) == count * 2, 'incomplete submitted/returned action pairs')
    for step in range(count):
        submitted, returned = actions[step * 2:step * 2 + 2]
        for action, status in ((submitted, 'submitted'), (returned, 'native_returned')):
            require(action['step'] == step and action['episode'] == episode and action['status'] == status,
                    'action ordering/status mismatch')
            for name in ('raw_action', 'bounded_action'):
                x = np.asarray(action[name])
                require(x.shape == (14,) and x.dtype.kind in 'fiu' and np.isfinite(x).all(), 'invalid ' + name)
            condition = action['policy']['task_condition']
            require(condition.get('task') == plan['task'] and condition.get('sha256') == plan['task_condition_sha256'],
                    'per-action task condition identity mismatch')
        require(returned['native_take_action_count'] == step + 1, 'action not acknowledged once')
        require(submitted['raw_action'] == returned['raw_action'] and submitted['bounded_action'] == returned['bounded_action']
                and submitted['policy'] == returned['policy'], 'action pair changed between submission and acknowledgment')
    onboard = directory / 'onboard'
    manifest = json.loads((onboard / 'manifest.json').read_text())
    require(manifest.get('schema') == 'robotwin-onboard-observations-v1' and manifest.get('complete') is True
        and manifest.get('lossless') is True and manifest.get('policy_inputs_unchanged') is True
        and manifest.get('episode') == episode and manifest.get('seed') == row['seed']
        and manifest.get('frames') == count + 1, 'onboard completion/count mismatch')
    frames = [json.loads(line) for line in (onboard / 'frames.jsonl').read_text().splitlines() if line.strip()]
    require(len(frames) == count + 1, 'onboard frame index count mismatch')
    archive = []
    fields = set(plan['policy']['views']) | {'commanded_joint_gripper_state'}
    require(set(manifest['fields']) == fields, 'onboard field allowlist mismatch')
    require(len(list(onboard.glob('observation_*.npz'))) == count + 1, 'onboard archive file count mismatch')
    for index, frame in enumerate(frames):
        require(frame['index'] == index and frame['control_step'] == index and frame['settle_step'] == 0
            and frame['file'] == f'observation_{index:06d}.npz', 'onboard frame order mismatch')
        path = onboard / frame['file']
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files) == fields, 'onboard archive fields mismatch')
            for view in plan['policy']['views']:
                rgb = data[view]
                require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.size > 0, 'invalid onboard RGB')
            state = data['commanded_joint_gripper_state']
            require(state.shape == (14,) and state.dtype.kind in 'fiu' and np.isfinite(state).all(), 'invalid onboard state')
        archive.append(identity(path))
    return {'verified_complete': True, 'scoring_source': plan['native_success_predicate'],
        'actions_acknowledged': count, 'onboard_frames_verified': count + 1,
        'onboard_files_sha256': digest(archive), 'video_files_created': 0, 'third_view_camera_created': False,
        'evidence_files': [identity(p) for p in (native_out / 'plan.json', native_out / 'result.json',
            directory / 'episode.json', actions_path, directory / 'recording.json', onboard / 'manifest.json', onboard / 'frames.jsonl')]}


def collect_episode(task, seed, native_out, execution):
    row = {'task': task, 'seed': seed, 'status': 'infrastructure_error', 'native_success': None,
           'native_output': str(native_out), 'execution': execution}
    result_path = Path(native_out) / 'result.json'
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
            rows = result.get('episodes', [])
            if result.get('task') != task or len(rows) != 1 or rows[0].get('seed') != seed:
                raise InfrastructureError('child task/seed/result cardinality mismatch')
            evidence = rows[0].get('evidence', {}).get('native_benchmark_success')
            status = rows[0]['status']
            row.update(status=status, native_result=identity(result_path), episode=rows[0],
                       child_result_status=result.get('status'))
            if (result.get('status') == 'completed' and execution.get('returncode') == 0
                    and not execution.get('external_timeout')
                    and ((status == 'native_success' and evidence is True)
                         or (status == 'step_limit' and evidence is False))):
                row['verification'] = verify_native_evidence(native_out, rows[0])
                row['native_success'] = evidence
            elif status in ('native_success', 'step_limit'):
                row.update(status='infrastructure_error', error='untrusted/incomplete native terminal result',
                           native_observed_before_error=evidence)
        except Exception as exc:
            row.update(status='infrastructure_error', error=f'{type(exc).__name__}: {exc}')
    else:
        row['error'] = 'child produced no result.json; evidence directory/log retained'
    if execution.get('external_timeout'):
        row.update(status='external_timeout', native_success=None)
    return row


def wilson95(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total))/denom
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return [max(0.0, center-radius), min(1.0, center+radius)]


def score_rows(rows):
    success = sum(row.get('native_success') is True for row in rows)
    failure = sum(row.get('native_success') is False for row in rows)
    scored, total = success + failure, len(rows)
    return {'planned_episodes': total, 'scored_episodes': scored, 'native_successes': success,
        'native_failures': failure, 'unscored_episodes': total - scored,
        'verified_complete': bool(total) and scored == total and all(r.get('verification', {}).get('verified_complete') is True for r in rows),
        'status_counts': dict(Counter(row['status'] for row in rows)),
        'native_success_rate': success / total if total and scored == total else None,
        'wilson_95': wilson95(success, total) if total and scored == total else None,
        'observed_scored_success_rate': success / scored if scored else None,
        'successful_completion_fraction': success / total if total else None,
        'interpretation': 'successful_completion_fraction is a fixed-denominator execution outcome, NOT model success rate; infra/timeout are unscored'}


def summarize(rows, task_names, state):
    return {'status': state, **score_rows(rows), 'tasks': {
        task: score_rows([r for r in rows if r['task'] == task]) for task in task_names},
        'episodes': rows, 'record_video': False, 'third_view_camera_created': False,
        'baseline_comparison': False, 'task_count': len(task_names),
        'seed_replacement': False, 'retry_policy': False, 'onboard_archives_required': True}


def queue_plan(args, bundle):
    names = [row['task'] for row in bundle['tasks']]
    if set(names) != {'put_bottles_dustbin', 'place_cans_plasticbox', 'open_laptop', 'click_bell', 'press_stapler'} or len(names) != 5:
        raise InfrastructureError('this registered evaluation requires the five fixed checkpoint tasks')
    assets = asset_evidence()
    plans = {task: task_plan(args, bundle, task, assets=assets) for task in names}
    return {'status': 'cpu_plan_ready_not_physical_validation', 'tasks': plans,
        'seeds_per_task': list(SEEDS), 'planned_episodes': len(names) * len(SEEDS),
        'checkpoint_sha256': bundle['receipt']['sha256'], 'contract_sha256': digest(bundle['contract']),
        'no_retries_or_seed_replacement': True, 'stop_after_consecutive_unscored': 3, 'record_video': False, 'onboard_archives_required': True,
        'queue': [{'task': task, 'seed': seed} for task in names for seed in SEEDS]}


def run_queue(args, plan, *, launcher=launch_episode):
    """launcher injection is for CPU tests only, not exposed as CLI fallback."""
    out = Path(args.out)
    names = list(plan['tasks'])
    rows = [{**job, 'status': 'not_started', 'native_success': None} for job in plan['queue']]
    state = 'running'
    consecutive_unscored = 0
    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        atomic_json(out / 'summary.json', summarize(rows, names, state))
        for i, row in enumerate(rows):
            task, seed = row['task'], row['seed']
            directory = out / task / f'seed_{seed}'
            directory.mkdir(parents=True, exist_ok=False)
            native_out = directory / 'native'
            command = child_command(args, task, seed, native_out)
            row.update(status='starting', native_output=str(native_out))
            atomic_json(directory / 'attempt.json', {'task': task, 'seed': seed, 'attempt': 1,
                'command': command, 'registered_before_start': True,
                'checkpoint_sha256': plan['checkpoint_sha256'], 'contract_sha256': plan['contract_sha256']})
            atomic_json(out / 'summary.json', summarize(rows, names, state))
            try:
                execution = launcher(command, directory / 'episode.log',
                    plan['tasks'][task]['limits']['external_wall_seconds'])
                rows[i] = collect_episode(task, seed, native_out, execution)
            except Exception as exc:
                rows[i] = {**row, 'status': 'infrastructure_error', 'native_success': None,
                           'error': f'{type(exc).__name__}: {exc}'}
            atomic_json(directory / 'outcome.json', rows[i])
            atomic_json(out / 'summary.json', summarize(rows, names, state))
            consecutive_unscored = consecutive_unscored + 1 if rows[i]['native_success'] is None else 0
            if consecutive_unscored >= 3:
                state = 'stopped_after_three_unscored'
                break
        if state == 'running':
            state = 'completed' if all(r['native_success'] is not None and r.get('verification', {}).get('verified_complete') is True for r in rows) else 'completed_with_unscored'
    except QueueCancelled as exc:
        state = 'cancelled'
        for row in rows:
            if row['status'] == 'starting':
                row.update(status='cancelled', native_success=None, error=str(exc))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        summary = summarize(rows, names, state)
        atomic_json(out / 'summary.json', summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--multitask-manifest', required=True)
    parser.add_argument('--out', required=True, help='new output directory, never overwrite evidence')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu-lock-owner')
    parser.add_argument('--camera-readback', choices=('serial',), default='serial')
    parser.add_argument('--render-denoiser', choices=('none',), default='none')
    parser.add_argument('--startup-timeout', type=float, default=180)
    parser.add_argument('--policy-timeout', type=float, default=30)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--episode-task', help=argparse.SUPPRESS)
    parser.add_argument('--episode-seed', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if tuple(args.seeds) != SEEDS:
        parser.error('registered seeds are exactly 922001..922010, in order')
    if args.execute and not args.gpu_lock_owner:
        parser.error('--execute requires externally held --gpu-lock-owner')
    if (args.episode_task is None) != (args.episode_seed is None):
        parser.error('internal episode task and seed must be paired')
    if args.episode_task is not None and (not args.execute or args.episode_seed not in SEEDS):
        parser.error('internal child requires execute and a registered seed')
    if not (0 < args.startup_timeout <= 300 and 0 < args.policy_timeout <= 300):
        parser.error('worker timeouts must be in (0,300] seconds')
    for name in ('checkpoint', 'multitask_manifest', 'out'):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    return args


def main(argv=None):
    args = parse_args(argv)
    for key in ('DISPLAY', 'WAYLAND_DISPLAY'):
        os.environ.pop(key, None)
    if args.episode_task is not None:
        return run_episode(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        bundle = load_contract(args.checkpoint, args.multitask_manifest)
        plan = queue_plan(args, bundle)
        atomic_json(out / 'plan.json', plan)
        if not args.execute:
            print(json.dumps({'status': plan['status'], 'planned_episodes': 50, 'out': str(out)}))
            return 0
        verify_gpu_owner(args.gpu_lock_owner)
        summary = run_queue(args, plan)
        print(json.dumps({k: summary[k] for k in ('status', 'native_success_rate', 'scored_episodes', 'unscored_episodes')}))
        return 0 if summary['status'] == 'completed' else 2
    except Exception as exc:
        failure = {'status': 'infrastructure_error', 'phase': 'preflight', 'error': f'{type(exc).__name__}: {exc}',
                   'model_constructed': False, 'simulator_started': False, 'scored_episodes': 0}
        atomic_json(out / 'preflight_error.json', failure)
        print(json.dumps(failure))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
