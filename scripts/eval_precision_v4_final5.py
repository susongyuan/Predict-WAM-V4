#!/usr/bin/env python3
"""Two final V4 milestones: five paired trials each, with the archived V2 baseline.

No video or observer during scoring. After all trials, export successful trials
from their original onboard frames without another simulator attempt.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness import zhuhai_precision_v4_training as v4
from harness import zhuhai_precision_v2_training as v2
from scripts import gpu_lock
from scripts.progress1_log_v41 import log
from scripts.eval_zhuhai_wam_precision_v4 import RAY_SCOPE

BASE = ROOT / 'outputs/zhuhai2026_tasks12'
RUNTIME = ROOT / 'scripts/eval_zhuhai_wam_precision_v4.py'
PYTHON = ROOT / '.conda-envs/behavior6/bin/python'
SEEDS = (932101, 932102, 932103, 932104, 932105)
TASK = 'zhuhai_task2'
V2_BASELINE = BASE / 'wam_precision_v2_task2_dagger_80k_fixed5_20261011_v23/summary.json'
VIDEO_DIRECTORY = ROOT / 'success video'

# An isolated module preserves all archive, physics-abort and braking replay checks.
spec = importlib.util.spec_from_file_location('_precision_v4_archive_evidence',
    ROOT / 'scripts/precision_v3_evidence_v42.py')
e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e
spec.loader.exec_module(e)
e.FORMAT, e.SEEDS = v4.FORMAT, SEEDS
BACKBONE_BINDING = dict(e.ACTUAL_PRECISION_BINDING, ray_scope=RAY_SCOPE)


def validate_binding(raw, step, seed, sha, contract=None):
    e.require(raw.get('task') == TASK and type(raw.get('seed')) is int and raw['seed'] == seed,
              'Task/seed mismatch')
    ready = raw.get('policy_ready', {})
    runtime = ready.get('runtime', {})
    e.require(ready.get('kind') == 'ready' and ready.get('version') == 1
        and ready.get('episode') == 'startup', 'Policy startup identity missing')
    e.require(runtime.get('checkpoint_sha256') == sha and runtime.get('checkpoint_step') == step
        and runtime.get('format') == v4.FORMAT and runtime.get('task') == TASK
        and runtime.get('strict_load') is True and runtime.get('projections_reset') is False,
        'Strict V4 checkpoint binding required')
    e.require(raw.get('record_video') is False and raw.get('third_view') is False
        and raw.get('observer_policy_input') is False and raw.get('max_control_steps') == 2400
        and raw.get('maximum_physics_seconds') == 240., 'Recording or control budget changed')
    for field in ('vlm', 'high_level_planner', 'ask', 'memory_writes', 'future_inputs_used_online',
                  'observer_policy_input', 'hjepa_auxiliary_heads_used_online', 'event_clock_is_verified_grasp'):
        e.require(runtime.get(field) is False, 'Unexpected online input/mechanism: ' + field)
    for field in ('dino_frozen', 'precision_refiner_used_online', 'event_clock_used_online', 'chunk_vtt16_used_online'):
        e.require(runtime.get(field) is True, 'Required online mechanism missing: ' + field)
    for key in ('robot_contract', 'data_contract', 'action_postprocessing'):
        binding, digest = runtime.get(key), runtime.get(key + '_sha256')
        e.require(isinstance(binding, dict) and e.valid_sha(digest) and e.digest(binding) == digest,
                  'Invalid runtime binding: ' + key)
        e.require(raw.get(key + '_sha256') == digest, 'Result/runtime digest mismatch: ' + key)
        if key != 'action_postprocessing':
            e.require(raw.get(key) == binding, 'Result/runtime contract mismatch: ' + key)
        if contract is not None:
            e.require(contract.get(key) == binding and contract.get(key + '_sha256') == digest,
                      'Checkpoint/runtime mismatch: ' + key)
    e.require(runtime.get('precision_v3_actual') == BACKBONE_BINDING
        and runtime.get('system_mode') == 'precision_wam_v4_official_braking', 'Wrong actual backbone')
    expected = dict(version=4, horizon=16, input_dim=7184, output_dim=3072, added_parameters=7877377,
                    camera_ray=v4.PRECISION_V4_SPEC['camera_ray'])
    e.require(runtime.get('precision_v4_actual') == expected
        and runtime.get('chunk_vtt16') == v4.PRECISION_V4_SPEC
        and e.finite(runtime.get('chunk_vtt16_alpha')), 'Wrong actual Chunk-VTT16')
    baseline = step == 80000
    e.require(runtime.get('v3_warmstart_baseline') is baseline
        and (not baseline or runtime['chunk_vtt16_alpha'] == 0.), 'V3 warm-start baseline must have zero fusion')
    dc = runtime['data_contract']
    e.require(dc.get('record_video') is False and dc.get('third_view') is False
        and dc.get('camera_count') == 3 and dc.get('camera_names') == list(e.FIELDS[:3]),
        'Only the original three policy cameras are admitted')
    for name in ('mechanical', 'camera'):
        gate = raw.get('admission_gates', {}).get(name, {})
        keys = [key for key in ('robot_contract', 'robot_contract_sha256') if key in gate]
        e.require(gate.get('passed') is True and keys and all(gate[k] == runtime[k] for k in keys),
                  'Actual admission gate failed: ' + name)
        if name == 'mechanical':
            e.require(gate.get('events') == [], 'Mechanical admission contains safety events')
    return runtime


def reject_video(directory, raw):
    e.require(raw['record_video'] is False and raw['third_view'] is False
        and not list(Path(directory).glob('third_person*')) and not list(Path(directory).glob('*.mp4')),
        'Video/observer artifacts are forbidden during scored trials')
    return {}


e.validate_binding, e.video_evidence = validate_binding, reject_video


def select_milestones(training, final_step):
    candidates = []
    for path in Path(training).glob('checkpoint_step_*.pt'):
        suffix = path.stem.removeprefix('checkpoint_step_')
        if suffix.isdigit():
            step = int(suffix)
            if 80000 < step and final_step - 5000 <= step <= final_step:
                candidates.append((step, path))
    candidates.sort(reverse=True)
    e.require(len(candidates) >= 2 and candidates[0][0] == final_step
        and candidates[0][0] != candidates[1][0], 'Need final and nearest distinct saved milestone in last 5000 updates')
    return [path for _, path in candidates[:2]]


def receipt(path):
    row = e.read_json(path.with_name(path.name + '.sha256.json'))
    e.require(row.get('size') == path.stat().st_size and row.get('sha256') == e.sha256_file(path),
              'Checkpoint bytes differ from receipt: ' + str(path))
    if path.name.startswith('checkpoint_step_'):
        e.require(type(row.get('step')) is int and path.name == f"checkpoint_step_{row['step']:06d}.pt",
                  'Checkpoint filename and actual saved step disagree')
    return dict(path=str(path), step=row['step'], sha256=row['sha256'], size=row['size'])


def audit_training(training):
    import torch
    status = e.read_json(training / 'status.json')
    manifest = e.read_json(training / 'manifest.json')
    contract = v4.validate_training_contract(manifest['contract'])
    schedule = contract['v4_training']['schedule']
    updates, final_step = schedule['steps'], 80000 + schedule['steps']
    e.require(status.get('status') == 'completed' and status.get('step') == final_step
        and status.get('updates') == updates and status.get('target_updates') == updates
        and status.get('target_step') == final_step, 'Formal training has not completed')
    e.require(manifest['contract_sha256'] == e.digest(contract), 'Manifest contract digest mismatch')
    v2.verify_code_hashes(contract['source_files'])
    rows = [e.loads(line) for line in (training / 'log.jsonl').read_text().splitlines()]
    e.require(len(rows) == updates, 'Training log does not contain all optimizer updates')
    short_metric = 'future16' if contract.get('future_prediction', {}).get('horizons') == [16, 32] else 'future8'
    for index, row in enumerate(rows, 1):
        e.require(row.get('updates') == index and row.get('step') == 80000 + index,
                  'Training update gap or duplicate')
        for key in ('loss', 'mse', short_metric, 'future32', 'idm', 'chunk_vtt_loss',
                    'alpha_after_update', 'seconds_per_step', 'predictor_grad_norm'):
            e.require(e.finite(row.get(key)), 'Missing/nonfinite training metric: ' + key)
        if row.get('chunk_vtt_valid_samples', 0):
            e.require(row['predictor_grad_norm'] > 0, 'Predictor supervision gradient missing')
    paths = select_milestones(training, final_step)
    baseline = training / 'checkpoint_step_080000.pt'
    cohorts = []
    for path in paths:
        frozen, saved = v4.read_contract(path)
        e.require(frozen == contract and saved.get('model_only') is True
            and saved.get('verification_only') is not True, 'Require an actual trained checkpoint matching its manifest')
        cohorts.append(dict(**receipt(path), label='v4'))
    v4.read_contract(baseline)
    initial = torch.load(baseline, map_location='cpu', weights_only=False, mmap=True)
    e.require(initial.get('updates') == 0 and initial.get('origin_step') == 80000
        and float(initial['model_state_dict']['chunk_vtt16.alpha']) == 0., 'Baseline is not the untrained V3 warm start')
    del initial
    last_path = training / 'checkpoint_last.pt'
    full_receipt = receipt(last_path)
    full = torch.load(last_path, map_location='cpu', weights_only=False, mmap=True)
    final = torch.load(paths[0], map_location='cpu', weights_only=False, mmap=True)
    e.require(full.get('model_only') is False and full.get('step') == final_step
        and full.get('updates') == updates and full.get('contract') == contract
        and full.get('rng') is not None and full.get('torch_rng') is not None and full.get('cuda_rng'),
        'Final resumable state is incomplete')
    e.require(set(full['model_state_dict']) == set(final['model_state_dict'])
        and all(torch.equal(tensor, final['model_state_dict'][key]) for key, tensor in full['model_state_dict'].items()),
        'Full state differs from final milestone')
    states = full.get('optimizer_state_dict', {}).get('state', {})
    e.require(bool(states), 'Final optimizer state missing')
    for state in states.values():
        e.require(all(not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
                      for value in state.values()), 'Nonfinite optimizer state')
    return contract, cohorts, dict(status='passed', updates=updates, final_step=final_step,
        full_state=full_receipt, optimizer_states=len(states), log_sha256=e.sha256_file(training / 'log.jsonl'),
        manifest_sha256=e.sha256_file(training / 'manifest.json'))


def simulator_environment(prior):
    # Keep Isaac's ABI settings; the checked parent owns NVIDIA/Vulkan/EGL selection.
    keys = ('LD_PRELOAD', 'OMNI_KIT_ACCEPT_EULA', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')
    environment = {key: prior['child_env'][key] for key in keys if key in prior['child_env']}
    environment['CUDA_VISIBLE_DEVICES'] = '0'
    return environment


def make_plan(args):
    training, out = args.training.resolve(strict=True), args.out.absolute()
    e.require(not out.exists() and out == out.resolve() and BASE in out.parents, 'Use a fresh project evaluation directory')
    contract, cohorts, audit = audit_training(training)
    old = e.read_json(V2_BASELINE)
    e.require(old['status'] == 'completed' and old['seeds'] == list(SEEDS)
        and old['evaluation_version'] == e.braking.VERSION, 'Original V2 comparison protocol differs')
    prior = e.read_json(BASE / 'wam_precision_v2_task2_80k_extra20_20261011_v24/plan.json')
    environment = simulator_environment(prior)
    sources = dict(contract['source_files'])
    for relative in ('scripts/eval_precision_v4_final5.py', 'scripts/eval_zhuhai_wam_precision_v4.py',
        'scripts/eval_zhuhai_wam_precision_v3_v42.py', 'scripts/precision_v3_evidence_v42.py',
        'harness/zhuhai_official_braking.py', 'scripts/with_project_gpu.py',
        'harness/isaac_zhuhai_runtime.py', 'harness/zhuhai_task2_rules.py',
        'harness/robotwin_multitask_policy_worker.py', 'scripts/progress1_log_v41.py',
        'scripts/run_zhuhai_task1_dr200_concurrent.py', 'scripts/test_eval_precision_v4.py'):
        sources[str(ROOT / relative)] = e.sha256_file(ROOT / relative)
    commands = []
    for ck in cohorts:
        for seed in SEEDS:
            target = out / f"{ck['label']}_{ck['step']:06d}" / TASK / f'seed_{seed}'
            command = [sys.executable, str(ROOT / 'scripts/with_project_gpu.py'), '--', str(PYTHON), '-B',
                str(RUNTIME), '--checkpoint', ck['path'], '--out', str(target), '--task', TASK,
                '--gpu-lock-owner', args.gpu_lock_owner, '--episode-seed', str(seed), '--max-steps', '2400',
                '--wall-seconds', '3600', '--startup-timeout', '180', '--policy-timeout', '60']
            commands.append(dict(seed=seed, checkpoint=ck, out=str(target), command=command))
    return dict(schema='precision-v4-final-paired5-v1', out=str(out), training=str(training),
        cohorts=cohorts, seeds=list(SEEDS), commands=commands, child_env=environment,
        gpu_lock_owner=args.gpu_lock_owner, frozen_sources=sources, training_audit=audit,
        record_video=False, third_view=False, randomized=False, max_steps=2400,
        v2_reference=dict(path=str(V2_BASELINE), sha256=e.sha256_file(V2_BASELINE), label='original_V2_80000',
            full_successes=old['full_successes'], scores=[r['observed_base_score'] for r in old['episodes']],
            mean_task_progress=sum(r['observed_base_score'] for r in old['episodes']) / 150),
        interpretation='Two final V4 checkpoints versus the original V2 80000 checkpoint on the same five seeds. '
            'The archived V2 trials are reused unchanged; no additional baseline/ablation simulations. '
            'Five paired development scenes, not a generalization estimate.')


def verify(plan):
    v2.verify_code_hashes(plan['frozen_sources'])
    for ck in plan['cohorts']:
        e.require(receipt(Path(ck['path'])) == {k: ck[k] for k in ('path', 'step', 'sha256', 'size')},
                  'Selected checkpoint changed')
    e.require(e.sha256_file(plan['v2_reference']['path']) == plan['v2_reference']['sha256'], 'Original V2 result changed')


def details(row):
    raw = e.read_json(row['raw_result_path'])
    terminal = raw.get('final_rule_evidence') or raw['last_rule_evidence']
    rules = [e.loads(line) for line in (Path(row['raw_result_path']).parent / 'rule_evidence.jsonl').read_text().splitlines()]
    objects = {}
    for name in terminal['objects']:
        seen = [(rule['physics_time_seconds'], rule['score']['items'][name]) for rule in rules]
        objects[name] = dict(terminal_state=terminal['objects'][name], terminal_score=terminal['score']['items'][name],
            first_grasp_seconds=next((t for t, item in seen if item['grasp_awarded']), None),
            first_placement_seconds=next((t for t, item in seen if item['placement_valid']), None))
    unfinished = [name for name in ('bowl', 'spoon', 'cup', 'plate')
                  if not (terminal['score']['items'][name]['placement_valid']
                          and terminal['score']['items'][name]['grasp_awarded'])]
    first = unfinished[0] if unfinished else None
    if row['full_task_success']:
        failure = None
    elif row['status'] != 'completed':
        failure = row['status']
    elif first:
        failure = first + ('_placement' if objects[first]['first_grasp_seconds'] is not None else '_before_verified_grasp')
    else:
        failure = 'strict_rule_incomplete'
    return dict(row, task_progress=row['observed_base_score'] / 30., failure_stage=failure,
                failure_stage_basis='first unfinished item in expert order; diagnostic only, never policy input',
                key_objects=objects, terminal_physics_seconds=terminal['physics_time_seconds'])


def summary(plan, rows, status, error=None):
    if status == 'completed':
        e.require(len(plan['cohorts']) == 2 and len(rows) == 10, 'Completed evaluation requires both five-trial cohorts')
    cohorts = []
    for ck in plan['cohorts']:
        selected = [row for row in rows if row['checkpoint_step'] == ck['step']]
        done = len(selected) == 5
        if done:
            e.require([r['seed'] for r in selected] == list(SEEDS), 'Paired cohort seed/order mismatch')
        cohorts.append(dict(checkpoint=ck, completed_episodes=len(selected), planned_episodes=5,
            full_successes=sum(r['full_task_success'] for r in selected),
            success_rate=sum(r['full_task_success'] for r in selected) / 5 if done else None,
            mean_task_progress=sum(r['task_progress'] for r in selected) / 5 if done else None,
            scores=[r['observed_base_score'] for r in selected]))
    return dict(schema=plan['schema'], status=status, cohorts=cohorts, episodes=rows, planned_episodes=10,
        seeds=list(SEEDS), record_video=False, third_view=False, randomized=False,
        evaluation_version=e.braking.VERSION, v2_reference=plan['v2_reference'],
        interpretation=plan['interpretation'], error=error)


def run_once(item, environment):
    target = Path(item['out'])
    target.parent.mkdir(parents=True, exist_ok=True)
    e.require(not target.exists(), 'Never repeat an attempted physical episode')
    with target.with_suffix('.log').open('xb') as stream:
        child = subprocess.Popen(item['command'], cwd=ROOT, env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            started = time.monotonic()
            while child.poll() is None:
                e.require(time.monotonic() - started < 3840, 'Physical episode wall deadline')
                path = target / 'result.json'
                if path.exists():
                    row = e.read_json(path)
                    age = time.time() - path.stat().st_mtime
                    e.require(not (row.get('status') == 'running' and age > 600), 'Action heartbeat stale')
                    e.require(not (row.get('status') == 'starting' and time.monotonic() - started > 300),
                              'Simulator startup exceeded 300 seconds')
                else:
                    e.require(time.monotonic() - started < 300, 'No simulator startup result after 300 seconds')
                time.sleep(30)
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def export_success(row, support):
    import cv2
    import numpy as np
    e.require(row['full_task_success'] is True, 'Only actual full-task successes may be exported')
    directory = Path(row['raw_result_path']).parent
    e.require(e.sha256_file(row['raw_result_path']) == row['raw_result_sha256'], 'Success result changed')
    files = row['evidence_files']
    e.require(e.sha256_file(directory / 'rule_evidence.jsonl') == files['rule_evidence.jsonl']['sha256'],
              'Success rule evidence changed')
    rules = [e.loads(line) for line in (directory / 'rule_evidence.jsonl').read_text().splitlines()]
    VIDEO_DIRECTORY.mkdir(exist_ok=True)
    path = VIDEO_DIRECTORY / f"precision_v4_{row['checkpoint_step']:06d}_seed{row['seed']}_full_{row['checkpoint_sha256'][:8]}.mp4"
    e.require(not path.exists(), 'Preserve existing success video')
    ffmpeg = next((ROOT / '.conda-envs/behavior6/lib/python3.12/site-packages/imageio_ffmpeg/binaries').glob('ffmpeg-linux-*'))
    command = [str(ffmpeg), '-nostdin', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '960x560', '-r', '10', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
        '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    with (support / (path.stem + '.encoder.log')).open('xb') as stream:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stream)
        try:
            for index, rule in enumerate(rules):
                name = f'onboard/observation_{index:06d}.npz'
                e.require(e.sha256_file(directory / name) == files[name]['sha256'], 'Success onboard frame changed')
                with np.load(directory / name, allow_pickle=False) as archive:
                    frames = [archive[name] for name in e.FIELDS[:3]]
                canvas = np.zeros((560, 960, 3), np.uint8)
                canvas[60:540, :640] = cv2.resize(frames[0], (640, 480))
                canvas[60:300, 640:] = frames[1]
                canvas[300:540, 640:] = frames[2]
                label = f"step {row['checkpoint_step']} | seed {row['seed']} | {rule['physics_time_seconds']:.1f}s | score {rule['score']['base_score']}/30"
                cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                cv2.putText(canvas, 'Full successful trial | original onboard frames | 1x playback',
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
                child.stdin.write(canvas.tobytes())
        finally:
            child.stdin.close()
            code = child.wait(timeout=60)
    e.require(code == 0, 'Success video encoder failed; preserve partial video and log')
    result = subprocess.run([str(ffmpeg), '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
        '-map', '0:v:0', '-progress', 'pipe:1', '-f', 'null', '-'], capture_output=True, text=True, timeout=180)
    frames = [int(line.split('=', 1)[1]) for line in result.stdout.splitlines() if line.startswith('frame=')]
    e.require(result.returncode == 0 and frames and frames[-1] == len(rules), 'Success video full decode/count failed')
    record = dict(path=str(path), sha256=e.sha256_file(path), frames=len(rules), fps=10,
        duration_seconds=len(rules) / 10, full_task_success=True, source_result=row['raw_result_path'],
        source_result_sha256=row['raw_result_sha256'], checkpoint_step=row['checkpoint_step'],
        seed=row['seed'], simulator_attempts_added=0, external_cameras_added=0, playback_speed=1)
    e.write_json(support / (path.stem + '.json'), record)
    return record


def execute(plan):
    from scripts import run_zhuhai_task1_dr200_concurrent as admission
    from scripts.eval_zhuhai_wam_precision_v4 import verify_owner
    out = Path(plan['out'])
    out.mkdir(parents=True, exist_ok=False)
    e.write_json(out / 'plan.json', plan)
    e.write_json(out / 'training_final_audit.json', plan['training_audit'])
    rows, held = [], False
    with (out / '.writer.lock').open('a') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify(plan)
            os.environ.update(plan['child_env'])
            e.require(gpu_lock.owner() is None and not admission.gpu_jobs(admission.processes()), 'GPU is occupied')
            resource = admission.scheduler_module().collection_resource_admission()
            e.require(admission.scheduler_module().resource_ready(resource, expansion=True), 'GPU/RAM resources insufficient')
            gpu_lock.acquire(plan['gpu_lock_owner'], max_wait=0)
            held = True
            e.require(not admission.gpu_jobs(admission.processes()), 'Foreign GPU participant appeared during admission')
            verify_owner(plan['gpu_lock_owner'])
            for item in plan['commands']:
                verify(plan)
                verify_owner(plan['gpu_lock_owner'])
                ck = item['checkpoint']
                name = f"step_{ck['step']:06d}_seed_{item['seed']}"
                e.write_json(out / ('attempt_' + name + '.json'), item)
                code = run_once(item, dict(os.environ, **plan['child_env']))
                e.write_json(out / ('exit_' + name + '.json'), dict(returncode=code))
                contract = e.read_json(Path(ck['path']).parent / 'manifest.json')['contract']
                row = e.classify(Path(item['out']) / 'result.json', ck, contract,
                    plan['frozen_sources'][str(RUNTIME)], SEEDS, e.classify_noncontact)
                e.require(code in (0, 2) if row['status'] in ('control_safety_abort', 'physics_safety_abort') else code == 0,
                          'Physical evidence and actual child exit disagree')
                row = details(row)
                row.update(checkpoint_step=ck['step'], checkpoint_sha256=ck['sha256'], cohort=ck['label'])
                rows.append(row)
                e.write_json(out / 'summary.json', summary(plan, rows, 'evaluating'))
                log(f"V4固定评估{len(rows)}/10：{ck['label']} step{ck['step']} seed{item['seed']}，"
                    f"整任务成功={row['full_task_success']}，分数={row['observed_base_score']}/30。", 'v4_fixed5')
            verify(plan)
            e.write_json(out / 'summary.json', summary(plan, rows, 'completed'))
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            e.write_json(out / 'summary.json', summary(plan, rows, 'halted', error))
            e.write_json(out / 'failure_guard.json', dict(error=error, preserved=True, physical_retries=0))
            raise
        finally:
            if held:
                jobs = admission.gpu_jobs(admission.processes())
                if jobs:
                    e.write_json(out / 'owner_retained_guard.json', dict(owner=plan['gpu_lock_owner'], jobs=jobs))
                else:
                    gpu_lock.release(plan['gpu_lock_owner'])
    if plan.get('export_success_videos') is False:
        e.write_json(out / 'success_videos.json', dict(status='disabled_by_default', videos=[],
            evaluation_unchanged=True))
        return 0
    videos = []
    support = out / 'success_video_evidence'
    support.mkdir()
    try:
        for row in rows:
            if row['full_task_success']:
                videos.append(export_success(row, support))
        e.write_json(out / 'success_videos.json', dict(status='completed', videos=videos,
            no_full_task_success=not videos, evaluation_unchanged=True))
    except BaseException as exc:
        e.write_json(out / 'success_video_failure_guard.json', dict(error=f'{type(exc).__name__}: {exc}',
            evaluation_unchanged=True, preserved=True))
        raise
    return 0


def live_v4_trainers():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(Path(arg).name in ('train_zhuhai_wam_precision_v4.py', 'train_zhuhai_wam_precision_v4_future16.py')
               or arg in ('scripts.train_zhuhai_wam_precision_v4', 'scripts.train_zhuhai_wam_precision_v4_future16') for arg in argv):
            found.append(int(proc.name))
    return found


def wait_for_training(training):
    training = Path(training)
    last_report = -1800.
    while True:
        guard = training / 'failure_guard.json'
        e.require(not guard.exists(), 'Training failure guard exists; preserve it and do not evaluate')
        status = e.read_json(training / 'status.json') if (training / 'status.json').exists() else {}
        manifest = e.read_json(training / 'manifest.json') if (training / 'manifest.json').exists() else {}
        phase = status.get('status', 'not_started')
        e.require(not str(phase).startswith(('failed', 'halted')), 'Training failed; do not evaluate or restart')
        if phase == 'completed':
            steps = manifest.get('contract', {}).get('v4_training', {}).get('schedule', {}).get('steps')
            e.require(type(steps) is int and status.get('updates') == steps
                and status.get('step') == 80000 + steps and status.get('target_updates') == steps
                and status.get('target_step') == 80000 + steps, 'Completed training status/manifest disagree')
            if not live_v4_trainers() and gpu_lock.owner() is None:
                return
        now = time.monotonic()
        if now - last_report >= 1800:
            log(f"V4评估CPU等待训练：status={phase}，新增更新{status.get('updates')}/"
                f"{status.get('target_updates')}；未占GPU、未启动物理测试。", 'v4_fixed5_queue')
            last_report = now
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path, help='Directory containing V4 manifest/checkpoints')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gpu-lock-owner', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wait-training', action='store_true', help='Wait on CPU for completed training and released GPU')
    args = parser.parse_args()
    if args.wait_training and not args.execute:
        parser.error('--wait-training requires --execute')
    if args.wait_training:
        wait_for_training(args.training)
    plan = make_plan(args)
    if args.execute:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}; preserve the current attempt')
        signal.signal(signal.SIGTERM, interrupted)
        return execute(plan)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
