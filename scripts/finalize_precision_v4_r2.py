#!/usr/bin/env python3
"""CPU-only final delivery from the already scheduled V4 paired ten trials.

No model inference, simulation, GPU access, physical retries or new training.
Default invocation prints a plan. Execution requires a fresh output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/zhuhai2026_tasks12'
VERIFY = BASE / 'verification/precision_v4_20261012'
TRAINING = BASE / 'wam_precision_v4_task2_50k_20261012_r2/train'
EVALUATION = BASE / 'wam_precision_v4_task2_final_paired5_20261012_r2'
SUPERVISOR = VERIFY / 'r2_post_reboot_supervisor_20261012'
ANALYSIS_PLAN = VERIFY / 'final_motion_analysis_plan.json'
SEEDS = (932101, 932102, 932103, 932104, 932105)
STEPS = (130000, 127500)
DOCS = ('progress1.md', 'docs/current_status.md', 'README.md', 'progress.md')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(data)
        after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'Evidence changed during read: ' + str(path))
    return h.hexdigest()


def write_once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def stamp():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def verify_sources(bindings):
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Source drift: ' + path)


def startup_bindings(validation):
    analysis = read(ANALYSIS_PLAN)
    require(analysis['status'] == 'prepared_not_executed' and analysis['inputs']['episodes_exact'] == 10,
            'Saved final10 analysis plan required')
    bindings = dict(analysis['dependency_sha256'])
    launch = read(VERIFY / 'r2_launch_and_admission.json')
    bindings.update(launch['source_hashes'])
    receipt = read(validation)
    require(receipt.get('status') == 'passed' and receipt.get('tests', 0) >= 8
            and receipt.get('failures') == 0 and receipt.get('errors') == 0,
            'Successful focused CPU validation receipt required')
    own = str(Path(__file__).resolve())
    test = str(ROOT / 'scripts/test_finalize_precision_v4_r2.py')
    require(receipt.get('source_sha256', {}).get(own) == sha(own)
            and receipt.get('source_sha256', {}).get(test) == sha(test)
            and receipt.get('analysis_plan_sha256') == sha(ANALYSIS_PLAN), 'Finalizer validation/source binding changed')
    bindings.update({own: sha(own), test: sha(test), str(ANALYSIS_PLAN): sha(ANALYSIS_PLAN),
                     str(validation.resolve()): sha(validation)})
    from harness.zhuhai_robot_spec import KINEMATIC_CHAIN, ASSET, ASSET_MANIFEST
    for path in (KINEMATIC_CHAIN, ASSET, ASSET_MANIFEST, ROOT / 'scripts/resume_precision_v4_after_reboot.py'):
        bindings[str(path)] = sha(path)
    verify_sources(bindings)
    return bindings


def cohort_rows(summary):
    require(summary.get('status') == 'completed' and summary.get('planned_episodes') == 10
            and summary.get('seeds') == list(SEEDS), 'Exactly completed final paired10 required')
    require(summary.get('record_video') is False and summary.get('third_view') is False
            and summary.get('randomized') is False, 'Scored final protocol changed')
    rows = summary.get('episodes', [])
    require(len(rows) == 10, 'Final10 row count differs')
    expected = [(s, seed) for s in STEPS for seed in SEEDS]
    require([(r.get('checkpoint_step'), r.get('seed')) for r in rows] == expected,
            'Checkpoint/seed cohort order, identity or duplicate differs')
    require([c.get('checkpoint', {}).get('step') for c in summary.get('cohorts', [])] == list(STEPS),
            'Final and nearest saved checkpoint cohorts differ')
    return {step: [r for r in rows if r['checkpoint_step'] == step] for step in STEPS}


def poll_state(supervisor, training, evaluation, *, now=None, stale_seconds=900):
    """Return ready only after the NEW post-reboot supervisor completed normally."""
    now = time.time() if now is None else now
    for directory in (supervisor, training, evaluation):
        for name in ('failure_guard.json', 'owner_retained_guard.json', 'success_video_failure_guard.json'):
            require(not (directory / name).exists(), 'Preserved failure guard; no automatic retry: ' + str(directory / name))
    completed = supervisor / 'completed.json'
    if completed.exists():
        done = read(completed)
        require(done.get('status') == 'completed' and done.get('physical_retries') == 0,
                'Supervisor did not complete without physical retry')
        path = evaluation / 'summary.json'
        require(Path(done.get('evaluation_summary', '')).resolve() == path.resolve()
                and done.get('evaluation_summary_sha256') == sha(path), 'Supervisor summary binding differs')
        cohort_rows(read(path))
        return True
    require(supervisor.is_dir(), 'New continuation supervisor has not been launched')
    latest = supervisor / 'latest.json'
    require(latest.exists(), 'New supervisor heartbeat missing')
    state = read(latest)
    require(state.get('phase') not in ('failed', 'halted'), 'New supervisor reported failure')
    tick = state.get('unix_seconds')
    require(type(tick) in (int, float) and math.isfinite(tick) and -60 <= now - tick <= stale_seconds,
            'New supervisor heartbeat stale or invalid; preserve evidence and halt')
    for directory, filename in ((training, 'status.json'), (evaluation, 'summary.json')):
        path = directory / filename
        if path.exists():
            require(read(path).get('status') not in ('failed', 'halted'), 'Child reported failure: ' + str(path))
    return False


def wait_completed(supervisor, training, evaluation, run, bindings, stale_seconds):
    from scripts.progress1_log_v41 import log
    previous_report = time.monotonic()
    while not poll_state(supervisor, training, evaluation, stale_seconds=stale_seconds):
        verify_sources(bindings)
        if time.monotonic() - previous_report >= 1800:
            log('V4最终CPU交付等待新的重启后监督完成训练及同10场实测；没有新增仿真、GPU命令或自动重试。', 'v4_finalizer')
            previous_report = time.monotonic()
        # Only thirty-second CPU polling; no long blocking sleep.
        time.sleep(30)


def path_metrics(points):
    import numpy as np
    points = np.asarray(points, dtype=float)
    require(points.ndim == 2 and len(points) >= 2 and np.isfinite(points).all(), 'Invalid trajectory points')
    travel = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    net = float(np.linalg.norm(points[-1] - points[0]))
    return dict(path_length=travel, net_displacement=net,
                cancellation_ratio=max(0., min(1., 1. - net / travel)) if travel > 1e-9 else None,
                stationary=travel <= 1e-9)


def chunk_approach(actions, q, rules, fk):
    import numpy as np
    from scripts.analyze_garbage_time_v41 import SIDES, OBJECTS
    n = len(actions)
    require(len(q) == len(rules) == n + 1, 'Safe physical trace alignment differs')
    points = {side: np.stack([fk.fk(side, state[sl])[:3, 3] for state in q])
              for side, (sl, _) in SIDES.items()}
    distances, overview = {}, {}
    for side in SIDES:
        distances[side], overview[side] = {}, {}
        for key in OBJECTS:
            corners = np.asarray([r['objects'][key]['world_corners_m'] for r in rules])
            lo, hi = corners.min(1), corners.max(1)
            d = np.linalg.norm(np.maximum(np.maximum(lo - points[side], points[side] - hi), 0.), axis=1)
            distances[side][key] = d
            nearest = int(np.argmin(d))
            overview[side][key] = dict(initial_distance_m=float(d[0]), final_distance_m=float(d[-1]),
                nearest_distance_m=float(d[nearest]), nearest_observation_step=nearest,
                near_lt_5cm_observations=int((d < .05).sum()))
    chunks = []
    arm = [*range(7), *range(8, 15)]
    for a in actions:
        if not a['policy']['replanned']:
            continue
        s = a['control_step']; m = min(8, n - s)
        if m <= 0:
            continue
        chunks.append(dict(first_action_step=s, executed_steps=m,
            measured_tool_m={side: path_metrics(pt[s:s + m + 1]) for side, pt in points.items()},
            measured_fourteen_joint_rad=path_metrics(q[s:s + m + 1, arm]),
            object_approach_distance_reduction_m={side: {key: float(d[s] - d[s + m])
                for key, d in values.items()} for side, values in distances.items()}))
    return dict(object_distance_overview=overview, executed_chunks=chunks,
        definition='Measured q[s..s+m], m=min(8, remaining safe executed steps); cancellation=1-net/path, stationary=null. Positive object distance reduction means closer to its dynamic AABB.',
        limits='Tool reference/AABB lower bound; object motion may change distance. Cancellation can represent valid maneuvering; interpret with scored grasp/placement.')


def analyze_motion(row):
    from harness.isaac_zhuhai_expert import FrankaFK
    from scripts import analyze_precision_v3_eval_v42 as a
    raw, actions, rules, safe_rules, q, commanded, sent = a.load_episode(row)
    safe_n = len(safe_rules) - 1
    require(safe_n >= 0 and safe_n <= len(actions), 'Invalid safe physical prefix')
    actions, sent, q = actions[:safe_n], sent[:safe_n], q[:safe_n + 1]
    objects = a.object_progress(safe_rules)
    closure = a.closure_analysis(actions, sent, safe_rules)
    fk = FrankaFK()
    geometry, activity, transitions = a.geometry_and_activity(q, safe_rules, objects['objects'], fk, closure)
    require(safe_rules[-1]['score']['base_score'] == row['observed_base_score']
            and safe_rules[-1]['score']['full_task_success'] == row['full_task_success'], 'Safe terminal score changed')
    return dict(checkpoint_step=row['checkpoint_step'], seed=row['seed'],
        source_result_sha256=row['raw_result_sha256'], status=row['status'],
        score=row['observed_base_score'], full_task_success=row['full_task_success'],
        safe_executed_steps=safe_n, unsafe_terminal_excluded=row['status'] == 'physics_safety_abort',
        failure_stage=objects['failure_stage'], objects=objects['objects'], closure=closure,
        geometry=geometry, activity=activity, transitions=transitions,
        motion=a.command_motion(actions, q, sent) if safe_n else {'available': False, 'reason': 'no safe actions'},
        approach_and_cancellation=chunk_approach(actions, q, safe_rules, fk),
        all_archives_rehashed=True, actual_command_alignment_verified=True)


def table_row(label, step, rows):
    require(len(rows) == 5 and [r['seed'] for r in rows] == list(SEEDS), 'Comparison requires each exact same five seeds')
    scores = [r['observed_base_score'] for r in rows]
    successes = sum(r['full_task_success'] is True for r in rows)
    require(all(type(s) in (int, float) and math.isfinite(s) and 0 <= s <= 30 for s in scores), 'Invalid task score')
    return dict(model=label, checkpoint_step=step, successes=successes, tests=5,
                success_fraction=f'{successes}/5', success_rate=successes / 5.,
                mean_score=sum(scores) / 5., mean_task_progress=sum(scores) / 150., scores=scores)


def verdict(table):
    baseline, final, preceding = table
    delta_success = final['successes'] - baseline['successes']
    delta_progress = final['mean_task_progress'] - baseline['mean_task_progress']
    if delta_success > 0:
        conclusion = '最终V4在这五组固定种子上完成成功次数增加；样本有限，不声称统计显著或组件因果收益。'
    elif delta_success < 0:
        conclusion = '最终V4在这五组固定种子上完成成功次数下降。'
    elif delta_progress > 0:
        conclusion = '整任务成功次数未增加，但最终V4的平均任务进度提高；不能称已解决整任务。'
    elif delta_progress < 0:
        conclusion = '整任务成功次数未增加，最终V4平均任务进度下降。'
    else:
        conclusion = '最终V4在这五组固定种子上没有观察到整任务成功或平均进度提升。'
    if final['successes'] == 5:
        recommendation = '本固定五场全部成功，先保留权重和成功证据；当前没有必须追加训练的实证依据。'
    elif preceding['successes'] > final['successes'] or (preceding['successes'] == final['successes']
            and preceding['mean_task_progress'] > final['mean_task_progress']):
        recommendation = '更早末期权重实测更好，先分析联合训练退化及失败阶段；不自动延训或从零重训。'
    else:
        recommendation = '整任务仍有失败，先根据实际接近、闭合、持物和阶段证据定位瓶颈，再决定是否追加训练；本次不自动增加训练预算。'
    return dict(conclusion=conclusion, final_vs_v2_success_count_delta=delta_success,
        final_vs_v2_mean_progress_delta=delta_progress, further_training_recommendation=recommendation,
        scope='Same five fixed development seeds; no statistical significance, generalization or isolated architecture attribution.')


def verify_videos(evaluation, rows):
    manifest = read(evaluation / 'success_videos.json')
    require(manifest.get('status') == 'completed' and manifest.get('evaluation_unchanged') is True,
            'Success export did not finish')
    success = {(r['checkpoint_step'], r['seed']): r for r in rows if r['full_task_success']}
    videos = manifest.get('videos', [])
    require(manifest.get('no_full_task_success') is (not success) and len(videos) == len(success),
            'Success/no-success video evidence disagrees with real trials')
    seen = set()
    for video in videos:
        key = (video.get('checkpoint_step'), video.get('seed'))
        require(key in success and key not in seen, 'Video is not a distinct actual full-task success')
        seen.add(key); row = success[key]; path = Path(video['path'])
        require(path.parent.resolve() == (ROOT / 'success video').resolve() and not path.is_symlink()
                and sha(path) == video['sha256'], 'Success video path/bytes changed')
        require(video.get('full_task_success') is True and video.get('source_result') == row['raw_result_path']
                and video.get('source_result_sha256') == row['raw_result_sha256']
                and video.get('frames') == row['control_steps'] + 1 and video.get('fps') == 10
                and video.get('duration_seconds') == video['frames'] / 10.
                and video.get('simulator_attempts_added') == 0 and video.get('external_cameras_added') == 0
                and video.get('playback_speed') == 1, 'Success video source, duration or replay identity differs')
        require(read(evaluation / 'success_video_evidence' / (path.stem + '.json')) == video,
                'Per-video export receipt changed')
    return dict(**manifest, manifest_path=str(evaluation / 'success_videos.json'),
                manifest_sha256=sha(evaluation / 'success_videos.json'), video_bytes_and_source_receipts_verified=True,
                decode_evidence='Immutable successful exporter receipt is published only after full CPU ffmpeg decode/count succeeds; finalizer independently rehashes its video bytes and validates source/count.')


def baseline_safe_score(row):
    """Compare bound safe terminal rules with the baseline row, never copied aliases."""
    from scripts.analyze_precision_v3_eval_v42 import binding_read
    rules = [json.loads(line) for line in binding_read(Path(row['raw_result_path']).parent,
              'rule_evidence.jsonl', row).splitlines()]
    safe = rules[:-1] if row['status'] == 'physics_safety_abort' else rules
    require(bool(safe), 'Baseline has no safe scored observations')
    terminal = safe[-1]['score']
    require(terminal['base_score'] == row['observed_base_score']
            and terminal['full_task_success'] == row['full_task_success'],
            'Baseline safe terminal score/success differs')
    return terminal


def independent_evidence(training, evaluation, supervisor):
    from scripts import eval_precision_v4_final5 as v
    from scripts import analyze_precision_dagger_milestones_v19 as old
    plan = v.e.read_json(evaluation / 'plan.json')
    saved = v.e.read_json(evaluation / 'summary.json')
    groups = cohort_rows(saved)
    v.verify(plan)
    contract, cohorts, audit = v.audit_training(training)
    from harness.zhuhai_robot_spec import KINEMATIC_CHAIN, ASSET, ASSET_MANIFEST, TOOL_FROM_HAND
    robot = contract['robot_contract']; chain = read(KINEMATIC_CHAIN)
    require(sha(ASSET) == robot['asset_sha256'] and sha(ASSET_MANIFEST) == robot['asset_manifest_sha256']
            and chain['asset_sha256'] == robot['asset_sha256'] and chain['tool_from_hand'] == TOOL_FROM_HAND,
            'Offline FK chain/asset/tool does not match frozen robot contract')
    require([c['step'] for c in cohorts] == list(STEPS), 'Final checkpoints differ from fixed finalizer contract')
    require([{k: c[k] for k in ('path', 'step', 'sha256', 'size')} for c in plan['cohorts']] ==
            [{k: c[k] for k in ('path', 'step', 'sha256', 'size')} for c in cohorts], 'Evaluated checkpoints differ from completed training')
    fresh_rows = []
    for ck in plan['cohorts']:
        for row in groups[ck['step']]:
            require(row['checkpoint_sha256'] == ck['sha256'] and row.get('cohort') == ck['label'], 'Row checkpoint binding changed')
            fresh = v.e.classify(Path(row['raw_result_path']), ck, contract,
                plan['frozen_sources'][str(v.RUNTIME)], SEEDS, v.e.classify_noncontact)
            fresh = v.details(fresh)
            for key, value in fresh.items():
                require(row.get(key) == value, 'Independent episode admission differs: ' + key)
            fresh.update(checkpoint_step=ck['step'], checkpoint_sha256=ck['sha256'], cohort=ck['label'])
            fresh_rows.append(fresh)
    require(v.summary(plan, fresh_rows, 'completed') == saved, 'Independently reconstructed summary differs')
    baseline_path = Path(plan['v2_reference']['path'])
    baseline = v.e.read_json(baseline_path)
    require(baseline['status'] == 'completed' and baseline['seeds'] == list(SEEDS)
            and baseline['checkpoint_step'] == 80000 and baseline['evaluation_version'] == saved['evaluation_version'],
            'V2 baseline protocol/checkpoint differs')
    baseline_receipt = v.receipt(Path(baseline['checkpoint']['path']))
    require(baseline_receipt['sha256'] == baseline['checkpoint_sha256']
            and baseline_receipt['step'] == 80000, 'Baseline summary/actual checkpoint receipt differs')
    for row in baseline['episodes']:
        old.episode(row)  # Raw result and physical-rule provenance verification.
        baseline_safe_score(row)
        raw = v.e.read_json(row['raw_result_path']); rt = raw['policy_ready']['runtime']
        require(rt['checkpoint_step'] == 80000 and rt['checkpoint_sha256'] == baseline['checkpoint_sha256'],
                'Baseline raw checkpoint identity differs')
    require(baseline['full_successes'] == sum(r['full_task_success'] for r in baseline['episodes']), 'Baseline success count differs')
    table = [table_row('V2原80k', 80000, baseline['episodes']),
             *[table_row('V4', step, groups[step]) for step in STEPS]]
    motion = [analyze_motion(row) for row in fresh_rows]
    videos = verify_videos(evaluation, fresh_rows)
    training_plan = read(supervisor / 'plan.json')
    reboot_admission = read(supervisor / 'resume_admission.json')
    short_path = VERIFY / 'r2_continuation_supervisor/short_admission_audit.json'
    require(reboot_admission.get('status') == 'passed'
            and reboot_admission.get('short_admission_sha256') == sha(short_path),
            'Post-reboot continuation/short-run admission binding differs')
    short_audit = read(short_path)
    require(short_audit.get('status') == 'passed', 'Original short-run admission was not passed')
    log_rows = [read_row for line in (training / 'log.jsonl').read_text().splitlines()
                if (read_row := json.loads(line))]
    loss_keys = ('mse', 'future8', 'future32', 'idm', 'chunk_vtt_loss', 'chunk_huber', 'chunk_cosine',
                 'alpha_after_update', 'predictor_grad_norm', 'backbone_grad_norm', 'seconds_per_step')
    loss_windows = {label: {key: sum(row[key] for row in selected) / len(selected) for key in loss_keys}
                    for label, selected in (('first200', log_rows[:200]), ('last200', log_rows[-200:]))}
    notes = read(training / 'build_notes.json')
    require(notes['added_parameters'] == 7877377, 'Actual added parameter count differs')
    delivery = dict(schema='precision-v4-r2-final-delivery-v1', status='verified_actual_experiments', time=stamp(),
        comparison_table=table, verdict=verdict(table), actual_final_episodes=fresh_rows,
        raw_summary_diagnostic_limit='These independently re-admitted original summary rows remain unchanged. On physics_safety_abort, key_objects/terminal timestamps may include the unsafe terminal frame and are NOT conclusion evidence; motion_analysis contains safe-only object/closure/geometry diagnostics.',
        safe_failure_evidence=[dict(checkpoint_step=r['checkpoint_step'], seed=r['seed'],
            failure_stage=r['failure_stage'], safe_executed_steps=r['safe_executed_steps'],
            unsafe_terminal_excluded=r['unsafe_terminal_excluded'], objects=r['objects']) for r in motion],
        failure_stage_counts={str(step): dict(Counter(r['failure_stage'] for r in motion
            if r['checkpoint_step'] == step and not r['full_task_success'])) for step in STEPS},
        model=dict(new_parameters=notes['added_parameters'], structure='7184->512->512->3072; projection3072->768; scalar alpha initialized0',
            chunk_vtt16=contract['precision_v4'], changed_files=['harness/wam_precision_v4.py',
                'harness/zhuhai_precision_v4_training.py', 'scripts/train_zhuhai_wam_precision_v4.py',
                'scripts/eval_zhuhai_wam_precision_v4.py', 'scripts/eval_precision_v4_final5.py'],
            limitations='Pooled short-range visual change is not an explicit object goal; ray geometry also corrected; alpha0 removes module only from current same backbone.'),
        training=dict(strategy=contract['v4_training'], completed_audit=audit,
            initialize_vs_scratch='Strict verified V2 80k warm start through existing V3 full32; short-run gradient/freeze/validation admission supported continuation. No controlled persistent negative-transfer evidence warrants scratch training.',
            short_audit=short_audit, reboot_admission=reboot_admission,
            losses_descriptive_windows=loss_windows, final_diagnostic_values=read(training / 'diagnostic_130000.json'),
            training_command=training_plan.get('training_command'), evaluation_command=training_plan.get('evaluation_command'),
            original80k_sha256=sha(Path(contract['v4_training']['initialization']['path'])),
            log=str(training / 'log.jsonl'), final_diagnostic=str(training / 'diagnostic_130000.json')),
        videos=videos, source_bindings=dict(summary_path=str(evaluation / 'summary.json'),
            summary_sha256=sha(evaluation / 'summary.json'), evaluation_plan_sha256=sha(evaluation / 'plan.json'),
            supervisor_completed_sha256=sha(supervisor / 'completed.json'), baseline_path=str(baseline_path), baseline_sha256=sha(baseline_path)),
        interpretation=['Task successes use original physical rules; loss is not task success.',
            'Only the prescribed ten V4 trials are analyzed; no retries, new simulator calls or ablations.',
            'Deleted trained V3 checkpoint is unavailable; baseline is explicitly original V2 80k.',
            'Video absence for zero full successes is a real outcome, never replaced by partial success clips.'])
    require(delivery['training']['original80k_sha256'] == contract['v4_training']['initialization']['sha256'],
            'Protected original80k checkpoint changed')
    return delivery, dict(schema='precision-v4-r2-final10-motion-analysis-v1', time=stamp(),
        source_summary_sha256=delivery['source_bindings']['summary_sha256'], episodes=motion, extra_trials=0,
        analysis_plan_path=str(ANALYSIS_PLAN), analysis_plan_sha256=sha(ANALYSIS_PLAN))


def documentation_content(old, relative, message, result_path):
    entry = f'V4最终独立核验完成：{message}。证据：{result_path}；全部原始失败和成功证据保留，未增加试次或自动延训。'
    if relative == 'progress1.md':
        pattern = r'(?ms)^## 最新V4任务交接[^\n]*\n.*?(?=^## )'
        require(re.search(pattern, old), 'Latest progress1 overview anchor missing')
        new = re.sub(pattern, '## 最新V4任务交接（' + stamp() + '，最终实测交付）\n' + entry + '\n\n', old, count=1)
        require(len(new.splitlines()) < 200, 'progress1 exceeds199 lines; preserve prior docs and stop')
        return new
    if relative == 'docs/current_status.md':
        require('## 最新V4授权与执行' in old, 'Current status V4 anchor missing')
        new = re.sub(r'(?ms)^## 最新V4授权与执行\n.*?(?=^## )', '## 最新V4授权与执行\n' + entry + '\n\n', old, count=1)
        new = re.sub(r'(?m)^# 当前状态[^\n]*', '# 当前状态：V4训练及固定10次真实仿真已完成、独立交付核验通过', new, count=1)
        new = re.sub(r'(?s)(<!-- live-10min:start -->).*?(<!-- live-10min:end -->)', r'\1\n' + entry + r'\n\2', new, count=1)
        return new
    if relative == 'README.md':
        require(old.startswith('# 最新执行：'), 'README short entry anchor missing')
        new = re.sub(r'(?s)^# 最新执行：.*?(?=\n# VT-WAM)', '# 最新执行：V4固定双臂任务最终结果\n' + entry + '\n', old, count=1)
        return re.sub(r'(?m)^> \*\*当前（[^\n]*', '> **当前：** V4训练与固定10次真实测试已独立核验；最终结果见上方和[当前状态](docs/current_status.md)。', new, count=1)
    require(relative == 'progress.md', 'Unknown documentation target')
    return old.rstrip() + '\n\n## ' + stamp() + ' V4最终真实测试与独立交付\n\n' + entry + '\n'


def update_documents(delivery, run):
    from scripts.progress1_log_v41 import log
    table = delivery['comparison_table']
    message = '；'.join(f"{r['model']} {r['checkpoint_step']}：{r['success_fraction']}，平均进度{r['mean_task_progress']:.2%}" for r in table)
    message += '；' + delivery['verdict']['conclusion']
    relative_result = str((run / 'final_delivery.json').relative_to(ROOT))
    archive = ROOT / 'docs/archive/documentation_snapshot_20261007.zip'
    require(archive.is_file(), 'Existing documentation backup zip required')
    with (VERIFY / '.final_delivery_docs.lock').open('a') as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX)
        handles = {rel: stack.enter_context((ROOT / rel).open('r+', encoding='utf-8')) for rel in DOCS}
        for handle in handles.values():
            fcntl.flock(handle, fcntl.LOCK_EX)
        old = {rel: handle.read() for rel, handle in handles.items()}
        new = {rel: documentation_content(text, rel, message, relative_result) for rel, text in old.items()}
        backup_root = 'v4_final_delivery/' + datetime.now().strftime('%Y%m%dT%H%M%S%f')
        with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as zipped:
            for rel, text in old.items():
                zipped.writestr(backup_root + '/' + rel, text)
        for rel, handle in handles.items():
            handle.seek(0); handle.write(new[rel]); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
    published = log(message + '；CPU最终交付核验完成，实际视频/无整成功回执已验证，不自动加训。', 'v4_finalizer')
    return dict(time=stamp(), files={str(ROOT / rel): sha(ROOT / rel) for rel in DOCS},
                backup_zip=str(archive), backup_prefix=backup_root, logger_published_1800s_summary=published)


def execute(args):
    bindings = startup_bindings(args.validation)
    args.out.mkdir(parents=False, exist_ok=False)
    with (args.out / '.writer.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(args.out / 'plan.json', dict(training=str(args.training), evaluation=str(args.evaluation),
            supervisor=str(args.supervisor), source_bindings=bindings, wait_poll_seconds=30,
            stale_seconds=args.stale_seconds, physical_retries=0, gpu_calls=0))
        try:
            if args.wait:
                wait_completed(args.supervisor, args.training, args.evaluation, args.out, bindings, args.stale_seconds)
            require(poll_state(args.supervisor, args.training, args.evaluation, stale_seconds=args.stale_seconds),
                    'Final experiments not complete; use --wait to observe without retries')
            verify_sources(bindings)
            delivery, motion = independent_evidence(args.training, args.evaluation, args.supervisor)
            verify_sources(bindings)
            import torch
            require(not torch.cuda.is_initialized(), 'CPU finalizer initialized CUDA unexpectedly')
            write_once(args.out / 'motion_analysis.json', motion)
            delivery['motion_analysis_path'] = str(args.out / 'motion_analysis.json')
            delivery['motion_analysis_sha256'] = sha(args.out / 'motion_analysis.json')
            write_once(args.out / 'final_delivery.json', delivery)
            docs = update_documents(delivery, args.out)
            write_once(args.out / 'documentation_receipt.json', docs)
            write_once(args.out / 'completed.json', dict(status='completed', time=stamp(),
                final_delivery_sha256=sha(args.out / 'final_delivery.json'), motion_sha256=sha(args.out / 'motion_analysis.json'),
                documentation_receipt_sha256=sha(args.out / 'documentation_receipt.json'), physical_retries=0, gpu_calls=0))
            print(json.dumps(dict(status='completed', out=str(args.out), table=delivery['comparison_table']), ensure_ascii=False))
            return 0
        except BaseException as exc:
            write_once(args.out / 'failure_guard.json', dict(time=stamp(), error=f'{type(exc).__name__}: {exc}',
                preserved=True, automatic_retry=False, physical_retries=0, gpu_calls=0))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=VERIFY / 'r2_final_delivery')
    parser.add_argument('--training', type=Path, default=TRAINING)
    parser.add_argument('--evaluation', type=Path, default=EVALUATION)
    parser.add_argument('--supervisor', type=Path, default=SUPERVISOR)
    parser.add_argument('--validation', type=Path, default=VERIFY / 'finalizer_cpu_tests.json')
    parser.add_argument('--stale-seconds', type=int, default=900)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require(args.stale_seconds >= 300, 'Heartbeat timeout must allow normal supervisor work')
    require(args.out.absolute() == args.out.resolve() and args.out.parent == VERIFY,
            'Use a fresh direct verification child directory; never symlink outputs')
    if not args.execute:
        print(json.dumps(dict(status='plan_only', supervisor=str(args.supervisor), out=str(args.out),
            final_checkpoints=list(STEPS), seeds=list(SEEDS), polls_seconds=30,
            analysis_plan=str(ANALYSIS_PLAN), gpu_calls=0, new_simulator_trials=0), indent=2))
        return 0
    def stopped(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum) + '; preserve finalizer state, no automatic retry')
    signal.signal(signal.SIGTERM, stopped)
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
