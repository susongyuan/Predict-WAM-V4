#!/usr/bin/env python3
"""Explicit exec16 deployment: two final V4 checkpoints plus original V2, 5 seeds each.

The frozen TRAIN contracts still declare exec8. This is a separate deployment
protocol, never a checkpoint rewrite or a reuse of archived exec8 scores.
Scoring uses the original simulator, physical rules, onboard archives and
official braking replay. Only actual full successes are exported after scoring.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_precision_v4_final5 as old

v2, v4, gpu_lock, log = old.v2, old.v4, old.gpu_lock, old.log
BASE, TASK, SEEDS, PYTHON = old.BASE, old.TASK, old.SEEDS, old.PYTHON
RUNTIME = ROOT / 'scripts/eval_zhuhai_wam_precision_exec16.py'
VIDEO_DIRECTORY = old.VIDEO_DIRECTORY
V2_BASELINE = old.V2_BASELINE
STEPS = (130000, 127500, 80000)
SCHEMA = 'precision-exec16-final-paired5-v1'


def runtime_module():
    from scripts import eval_zhuhai_wam_precision_exec16
    return eval_zhuhai_wam_precision_exec16


def isolated_evidence(format_name):
    name = '_precision_exec16_evidence_' + ('v4' if format_name == v4.FORMAT else 'v2')
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/precision_v3_evidence_v42.py')
    evidence = importlib.util.module_from_spec(spec)
    sys.modules[name] = evidence
    spec.loader.exec_module(evidence)
    evidence.FORMAT, evidence.SEEDS = format_name, SEEDS
    evidence.validate_binding = validate_binding
    evidence.video_evidence = old.reject_video
    evidence.changed_contract = lambda frozen: runtime_module().overlay(frozen)
    return evidence


# Separate modules keep V2/V4 onboard schemas isolated and retain original checks.
e = old.e
EVIDENCE = {}


def validate_binding(raw, step, seed, sha, contract=None):
    e.require(raw.get('task') == TASK and type(raw.get('seed')) is int and raw['seed'] == seed,
              'Task/seed mismatch')
    ready = raw.get('policy_ready', {})
    rt = ready.get('runtime', {})
    format_name = v2.FORMAT if step == 80000 else v4.FORMAT
    e.require(ready.get('kind') == 'ready' and ready.get('version') == 1
              and ready.get('episode') == 'startup', 'Policy startup identity missing')
    e.require(rt.get('checkpoint_sha256') == sha and rt.get('checkpoint_step') == step
              and rt.get('format') == format_name and rt.get('task') == TASK
              and rt.get('strict_load') is True and rt.get('projections_reset') is False,
              'Original checkpoint must retain its exact format and strict load')
    e.require(step in STEPS, 'Only the requested two final V4 milestones and original V2 are admitted')
    deployment = runtime_module().deployment_protocol()
    e.require(deployment.get('execution_prefix') == 16 and deployment.get('training_execution_prefix') == 8
              and deployment.get('action_chunk') == 32 and deployment.get('euler_steps') == 10
              and deployment.get('checkpoint_modified') is False, 'Invalid exec16 deployment protocol')
    e.require(rt.get('deployment') == deployment and rt.get('deployment_sha256') == e.digest(deployment),
              'Missing or altered exec16 deployment provenance')
    e.require(raw.get('record_video') is False and raw.get('third_view') is False
              and raw.get('observer_policy_input') is False and raw.get('max_control_steps') == 2400
              and raw.get('maximum_physics_seconds') == 240., 'Recording or fixed control budget changed')
    for field in ('vlm', 'high_level_planner', 'ask', 'memory_writes', 'future_inputs_used_online',
                  'observer_policy_input', 'hjepa_auxiliary_heads_used_online', 'event_clock_is_verified_grasp'):
        e.require(rt.get(field) is False, 'Unexpected online mechanism: ' + field)
    for field in ('dino_frozen', 'precision_refiner_used_online', 'event_clock_used_online'):
        e.require(rt.get(field) is True, 'Required online mechanism missing: ' + field)
    for key in ('robot_contract', 'data_contract', 'action_postprocessing'):
        binding, digest = rt.get(key), rt.get(key + '_sha256')
        e.require(isinstance(binding, dict) and e.valid_sha(digest) and e.digest(binding) == digest,
                  'Invalid runtime binding: ' + key)
        e.require(raw.get(key + '_sha256') == digest, 'Result/runtime digest mismatch: ' + key)
        if key != 'action_postprocessing':
            e.require(raw.get(key) == binding, 'Result/runtime contract mismatch: ' + key)
        if contract is not None:
            e.require(contract.get(key) == binding and contract.get(key + '_sha256') == digest,
                      'Checkpoint/deployment contract mismatch: ' + key)
    e.require(rt.get('action_postprocessing') == runtime_module().protocol(),
              'Exact original braking plus exec16 selection provenance is required')
    e.require(rt.get('time', {}).get('execution_frames') == 8
              and rt.get('precision', {}).get('execution_prefix') == 8,
              'Frozen training execution prefix must remain eight')
    if format_name == v4.FORMAT:
        expected = dict(old.BACKBONE_BINDING, executed_prefix=16)
        e.require(rt.get('precision_v3_actual') == expected
                  and rt.get('system_mode') == 'precision_wam_v4_official_braking_exec16',
                  'Wrong actual V4 backbone/deployment identity')
        identity = dict(version=4, horizon=16, input_dim=7184, output_dim=3072, added_parameters=7877377,
                        camera_ray=v4.PRECISION_V4_SPEC['camera_ray'])
        e.require(rt.get('precision_v4_actual') == identity
                  and rt.get('chunk_vtt16') == v4.PRECISION_V4_SPEC
                  and e.finite(rt.get('chunk_vtt16_alpha'))
                  and rt.get('chunk_vtt16_used_online') is True
                  and rt.get('v3_warmstart_baseline') is False, 'Wrong trained Chunk-VTT16 identity')
    else:
        expected = dict(version=2, refined_flow_steps=8, dense_kv_tokens=588,
                        queries_per_view=16, global_visual_tokens=48, patch_projection_input_dim=1026,
                        executed_prefix=16)
        e.require(rt.get('precision_v2_actual') == expected
                  and rt.get('system_mode') == 'precision_wam_v2_official_braking_exec16'
                  and rt.get('chunk_vtt16_used_online') is False,
                  'Baseline must be original V2, never the V4 alpha-zero warm start')
        e.require(not rt.get('precision_v4_actual') and not rt.get('chunk_vtt16'),
                  'Original V2 baseline cannot acquire V4 modules')
    dc = rt['data_contract']
    e.require(dc.get('record_video') is False and dc.get('third_view') is False
              and dc.get('camera_count') == 3 and dc.get('camera_names') == list(e.FIELDS[:3]),
              'Only the original three policy cameras are admitted')
    for name in ('mechanical', 'camera'):
        gate = raw.get('admission_gates', {}).get(name, {})
        keys = [key for key in ('robot_contract', 'robot_contract_sha256') if key in gate]
        e.require(gate.get('passed') is True and keys and all(gate[k] == rt[k] for k in keys),
                  'Actual admission gate failed: ' + name)
        if name == 'mechanical':
            e.require(gate.get('events') == [], 'Mechanical admission contains safety events')
    return rt


def verify_exec16_actions(directory, settings):
    """Prove dispatched prefix/cadence from original chunks, not a runtime label."""
    import numpy as np
    from harness.zhuhai_action_smoothing import smooth_chunk
    rows = [e.loads(line) for line in (Path(directory) / 'actions.jsonl').read_text().splitlines()]
    raw_chunk = smoothed_chunk = None
    for index, row in enumerate(rows):
        policy = row['policy']
        offset = index % 16
        e.require(policy.get('execution_prefix') == 16 and policy.get('replanned') is (offset == 0)
                  and policy.get('queue_remaining') == 15 - offset
                  and policy.get('history_steps') == index + 1, 'Actual action queue is not exec16')
        if offset == 0:
            raw_chunk = np.asarray(policy.get('raw_denormalized_chunk'), dtype=np.float64)
            smoothed_chunk = np.asarray(policy.get('smoothed_denormalized_chunk'), dtype=np.float64)
            e.require(raw_chunk.shape == smoothed_chunk.shape == (32, 16)
                      and np.isfinite(raw_chunk).all() and np.isfinite(smoothed_chunk).all(),
                      'Actual original 32-step chunk missing')
            e.require(np.array_equal(smooth_chunk(raw_chunk, settings), smoothed_chunk),
                      'Original full-chunk spline replay differs')
        else:
            e.require('raw_denormalized_chunk' not in policy and 'smoothed_denormalized_chunk' not in policy,
                      'Unexpected extra replan inside exec16 prefix')
        e.require(np.array_equal(np.asarray(policy.get('model_raw_action')), raw_chunk[offset])
                  and np.array_equal(np.asarray(row['smoothed_action']), smoothed_chunk[offset])
                  and policy.get('smoothed_action') == row['smoothed_action'],
                  'Dispatched sample differs from the original exec16 prefix')
    return dict(execution_prefix=16, dispatched_steps=len(rows),
                replans=sum(row['policy']['replanned'] for row in rows),
                full_chunk_spline_replay_verified=True, chunk_prefix_replay_verified=True)


def classify(item, sources):
    ck = item['checkpoint']
    format_name = v2.FORMAT if ck['label'] == 'original_v2' else v4.FORMAT
    evidence = EVIDENCE.setdefault(format_name, isolated_evidence(format_name))
    contract = e.read_json(Path(ck['path']).parent / 'manifest.json')['contract']
    evidence.require(contract.get('format') == format_name, 'Cohort and frozen model format differ')
    row = evidence.classify(Path(item['out']) / 'result.json', ck, contract,
                            sources[str(RUNTIME)], SEEDS, evidence.classify_noncontact)
    row = old.details(row)
    rt = e.read_json(row['raw_result_path'])['policy_ready']['runtime']
    row.update(checkpoint_step=ck['step'], checkpoint_sha256=ck['sha256'], cohort=ck['label'],
               deployment=rt['deployment'], deployment_sha256=rt['deployment_sha256'],
               exec16_evidence=verify_exec16_actions(item['out'], rt['action_postprocessing']))
    return row


def baseline_checkpoint():
    import torch
    archived = e.read_json(V2_BASELINE)
    e.require(archived.get('status') == 'completed' and archived.get('seeds') == list(SEEDS),
              'Original V2 reference identity differs')
    ck = old.receipt(Path(archived['checkpoint']['path']))
    e.require(ck['step'] == 80000 and ck['sha256'] == archived['checkpoint']['sha256'],
              'Original V2 checkpoint differs from archived binding')
    frozen = torch.load(ck['path'], map_location='cpu', weights_only=False, mmap=True)
    contract = v2.validate_training_contract(frozen['contract'])
    e.require(frozen.get('format') == v2.FORMAT and frozen.get('model_only') is True
              and frozen.get('step') == 80000 and frozen.get('contract_sha256') == e.digest(contract),
              'Original immutable V2 model-only milestone required')
    e.require(e.read_json(Path(ck['path']).parent / 'manifest.json')['contract'] == contract,
              'Original V2 manifest/actual checkpoint contract differ')
    v2.verify_code_hashes(contract['source_files'])
    return dict(ck, label='original_v2', format=v2.FORMAT), contract


def make_plan(args):
    training, out = args.training.resolve(strict=True), args.out.absolute()
    e.require(not out.exists() and out == out.resolve() and BASE in out.parents,
              'Use a fresh project evaluation directory')
    contract, cohorts, audit = old.audit_training(training)
    e.require([ck['step'] for ck in cohorts] == list(STEPS[:2]),
              'This deployment is bound to final 130000 and nearest saved 127500')
    cohorts = [dict(ck, format=v4.FORMAT) for ck in cohorts]
    baseline, baseline_contract = baseline_checkpoint()
    cohorts.append(baseline)
    sources = dict(contract['source_files'])
    for path, digest in baseline_contract['source_files'].items():
        e.require(path not in sources or sources[path] == digest, 'Frozen baseline/training source conflict')
        sources[path] = digest
    for relative in ('scripts/eval_precision_exec16_final5.py', 'scripts/test_eval_precision_exec16.py',
                     'scripts/eval_zhuhai_wam_precision_exec16.py', 'harness/zhuhai_exec16_deployment.py',
                     'scripts/eval_precision_v4_final5.py', 'scripts/precision_v3_evidence_v42.py',
                     'scripts/eval_zhuhai_wam_precision_v4.py', 'scripts/eval_zhuhai_wam_precision_v3_v42.py',
                     'harness/zhuhai_official_braking.py', 'scripts/with_project_gpu.py',
                     'harness/isaac_zhuhai_runtime.py', 'harness/zhuhai_task2_rules.py',
                     'harness/robotwin_multitask_policy_worker.py', 'scripts/progress1_log_v41.py',
                     'scripts/run_zhuhai_task1_dr200_concurrent.py'):
        path = ROOT / relative
        sources[str(path)] = e.sha256_file(path)
    prior = e.read_json(BASE / 'wam_precision_v2_task2_80k_extra20_20261011_v24/plan.json')
    commands = []
    for ck in cohorts:
        for seed in SEEDS:
            target = out / f"{ck['label']}_{ck['step']:06d}" / TASK / f'seed_{seed}'
            command = [sys.executable, str(ROOT / 'scripts/with_project_gpu.py'), '--', str(PYTHON), '-B',
                       str(RUNTIME), '--checkpoint', ck['path'], '--out', str(target), '--task', TASK,
                       '--gpu-lock-owner', args.gpu_lock_owner, '--episode-seed', str(seed), '--max-steps', '2400',
                       '--wall-seconds', '3600', '--startup-timeout', '180', '--policy-timeout', '60']
            commands.append(dict(seed=seed, checkpoint=ck, out=str(target), command=command))
    deployment = runtime_module().deployment_protocol()
    return dict(schema=SCHEMA, out=str(out), training=str(training), cohorts=cohorts, seeds=list(SEEDS),
                commands=commands, child_env=old.simulator_environment(prior), gpu_lock_owner=args.gpu_lock_owner,
                frozen_sources=sources, training_audit=audit, planned_episodes=15,
                record_video=False, third_view=False, randomized=False, max_steps=2400,
                deployment=deployment, deployment_sha256=e.digest(deployment),
                v2_reference=dict(path=str(V2_BASELINE), sha256=e.sha256_file(V2_BASELINE),
                                  protocol='archived_exec8_identity_only_not_comparison_scores', scores_reused=False),
                interpretation='Three actual exec16 cohorts on the same five fixed current-task seeds. '
                'Original V2 weights are re-evaluated under exec16; archived exec8 scores are not reused. '
                'TRAIN remains exec8; deployment changed explicitly. Five development scenes, not a generalization estimate.')


def verify(plan):
    v2.verify_code_hashes(plan['frozen_sources'])
    e.require(plan.get('schema') == SCHEMA and plan.get('planned_episodes') == 15
              and [ck['step'] for ck in plan['cohorts']] == list(STEPS)
              and [ck['label'] for ck in plan['cohorts']] == ['v4', 'v4', 'original_v2']
              and plan.get('seeds') == list(SEEDS), 'Exec16 plan must contain exactly three paired cohorts')
    expected = [(ck['step'], seed) for ck in plan['cohorts'] for seed in SEEDS]
    e.require([(item['checkpoint']['step'], item['seed']) for item in plan['commands']] == expected,
              'Actual fixed attempt list differs from paired plan')
    for item in plan['commands']:
        ck, command = item['checkpoint'], item['command']
        e.require(ck in plan['cohorts'] and str(RUNTIME) in command
                  and str(ROOT / 'scripts/with_project_gpu.py') in command
                  and command[command.index('--checkpoint') + 1] == ck['path']
                  and command[command.index('--task') + 1] == TASK
                  and command[command.index('--episode-seed') + 1] == str(item['seed'])
                  and command[command.index('--max-steps') + 1] == '2400'
                  and command[command.index('--out') + 1] == item['out']
                  and command[command.index('--gpu-lock-owner') + 1] == plan['gpu_lock_owner'],
                  'Physical command changed or points to legacy exec8 runtime')
    for ck in plan['cohorts']:
        e.require(old.receipt(Path(ck['path'])) == {k: ck[k] for k in ('path', 'step', 'sha256', 'size')},
                  'Selected checkpoint changed')
    e.require(e.sha256_file(plan['v2_reference']['path']) == plan['v2_reference']['sha256'],
              'Archived original V2 identity changed')
    deployment = runtime_module().deployment_protocol()
    e.require(plan.get('deployment') == deployment and plan.get('deployment_sha256') == e.digest(deployment),
              'Deployment provenance changed after plan creation')


def summary(plan, rows, status, error=None):
    if status == 'completed':
        e.require(len(plan['cohorts']) == 3 and len(rows) == 15,
                  'Completed evaluation requires all three five-trial cohorts')
    cohorts = []
    for ck in plan['cohorts']:
        selected = [row for row in rows if row['checkpoint_step'] == ck['step'] and row['cohort'] == ck['label']]
        done = len(selected) == 5
        if done:
            e.require([row['seed'] for row in selected] == list(SEEDS), 'Paired cohort seed/order mismatch')
        e.require(len(selected) <= 5, 'Duplicate physical attempt in cohort')
        cohorts.append(dict(checkpoint=ck, completed_episodes=len(selected), planned_episodes=5,
                            full_successes=sum(row['full_task_success'] for row in selected),
                            success_rate=sum(row['full_task_success'] for row in selected) / 5 if done else None,
                            mean_task_progress=sum(row['task_progress'] for row in selected) / 5 if done else None,
                            scores=[row['observed_base_score'] for row in selected]))
    if status == 'completed':
        e.require(all(cohort['completed_episodes'] == 5 for cohort in cohorts), 'Incomplete or unknown cohort')
    return dict(schema=SCHEMA, status=status, cohorts=cohorts, episodes=rows, planned_episodes=15,
                seeds=list(SEEDS), record_video=False, third_view=False, randomized=False,
                evaluation_version=e.braking.VERSION, deployment=plan['deployment'],
                deployment_sha256=plan['deployment_sha256'], v2_reference=plan['v2_reference'],
                interpretation=plan['interpretation'], error=error)


def export_success(row, support):
    """Export one proven success from its unchanged original onboard archive."""
    import cv2
    import numpy as np
    e.require(row['full_task_success'] is True and row['deployment']['execution_prefix'] == 16,
              'Only actual full-task exec16 successes may be exported')
    directory = Path(row['raw_result_path']).parent
    e.require(e.sha256_file(row['raw_result_path']) == row['raw_result_sha256'], 'Success result changed')
    files = row['evidence_files']
    e.require(e.sha256_file(directory / 'rule_evidence.jsonl') == files['rule_evidence.jsonl']['sha256'],
              'Success rule evidence changed')
    rules = [e.loads(line) for line in (directory / 'rule_evidence.jsonl').read_text().splitlines()]
    VIDEO_DIRECTORY.mkdir(exist_ok=True)
    path = VIDEO_DIRECTORY / f"precision_{row['cohort']}_exec16_{row['checkpoint_step']:06d}_seed{row['seed']}_full_{row['checkpoint_sha256'][:8]}.mp4"
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
                    frames = [archive[field] for field in e.FIELDS[:3]]
                canvas = np.zeros((560, 960, 3), np.uint8)
                canvas[60:540, :640] = cv2.resize(frames[0], (640, 480))
                canvas[60:300, 640:] = frames[1]
                canvas[300:540, 640:] = frames[2]
                label = f"{row['cohort']} exec16 | step {row['checkpoint_step']} | seed {row['seed']} | {rule['physics_time_seconds']:.1f}s | {rule['score']['base_score']}/30"
                cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                cv2.putText(canvas, 'Full successful trial | original onboard frames | 1x playback',
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
                child.stdin.write(canvas.tobytes())
        finally:
            child.stdin.close()
            code = child.wait(timeout=60)
    e.require(code == 0, 'Success video encoder failed; preserve partial video and log')
    result = subprocess.run([str(ffmpeg), '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
                             '-map', '0:v:0', '-progress', 'pipe:1', '-f', 'null', '-'],
                            capture_output=True, text=True, timeout=180)
    frames = [int(line.split('=', 1)[1]) for line in result.stdout.splitlines() if line.startswith('frame=')]
    e.require(result.returncode == 0 and frames and frames[-1] == len(rules), 'Success video full decode/count failed')
    record = dict(path=str(path), sha256=e.sha256_file(path), frames=len(rules), fps=10,
                  duration_seconds=len(rules) / 10, full_task_success=True,
                  source_result=row['raw_result_path'], source_result_sha256=row['raw_result_sha256'],
                  checkpoint_step=row['checkpoint_step'], checkpoint_sha256=row['checkpoint_sha256'],
                  cohort=row['cohort'], deployment=row['deployment'], deployment_sha256=row['deployment_sha256'],
                  seed=row['seed'], simulator_attempts_added=0, external_cameras_added=0, playback_speed=1)
    e.write_json(support / (path.stem + '.json'), record)
    return record


def execute(plan):
    from scripts import run_zhuhai_task1_dr200_concurrent as admission
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
            scheduler = admission.scheduler_module()
            e.require(scheduler.resource_ready(scheduler.collection_resource_admission(), expansion=True),
                      'GPU/RAM resources insufficient')
            gpu_lock.acquire(plan['gpu_lock_owner'], max_wait=0)
            held = True
            e.require(not admission.gpu_jobs(admission.processes()), 'Foreign GPU participant appeared during admission')
            runtime_module().verify_owner(plan['gpu_lock_owner'])
            for item in plan['commands']:
                verify(plan)
                runtime_module().verify_owner(plan['gpu_lock_owner'])
                ck = item['checkpoint']
                name = f"{ck['label']}_step_{ck['step']:06d}_seed_{item['seed']}"
                e.write_json(out / ('attempt_' + name + '.json'), item)
                code = old.run_once(item, dict(os.environ, **plan['child_env']))
                e.write_json(out / ('exit_' + name + '.json'), dict(returncode=code))
                row = classify(item, plan['frozen_sources'])
                e.require(code in (0, 2) if row['status'] in ('control_safety_abort', 'physics_safety_abort') else code == 0,
                          'Physical evidence and child exit disagree')
                rows.append(row)
                e.write_json(out / 'summary.json', summary(plan, rows, 'evaluating'))
                log(f"exec16固定评估{len(rows)}/15：{ck['label']} step{ck['step']} seed{item['seed']}，"
                    f"整任务成功={row['full_task_success']}，分数={row['observed_base_score']}/30。", 'v4_exec16_fixed5')
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
    e.write_json(out / 'success_videos.json', dict(status='disabled_by_default', videos=[],
                 evaluation_unchanged=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gpu-lock-owner', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wait-training', action='store_true')
    args = parser.parse_args()
    if args.wait_training and not args.execute:
        parser.error('--wait-training requires --execute')
    if args.wait_training:
        old.wait_for_training(args.training)
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
