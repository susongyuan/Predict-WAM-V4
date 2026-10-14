#!/usr/bin/env python3
"""CPU-only independent finalization of the existing Future16 paired ten trials.

Reuse the admitted copied-initialization lineage resolver without changing any
frozen evaluation source. Never launch trials, train, or export/record videos.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import finalize_precision_v4_r2 as analysis
from scripts import eval_precision_v4_future16_lineage as lineage
from scripts import analyze_precision_v3_eval_v42 as motion_analysis
from scripts import analyze_precision_dagger_milestones_v19 as baseline_analysis

queue = lineage.queue
future = lineage.future
PROTOCOL = dict(record_evaluation_video=False, third_view=False,
                export_success_videos=False, video_from_original_success_archives=False,
                extra_physical_trials=0, further_training_launched=False,
                reason='Current authorization is the existing ten evaluations; video recording and export remain off by default. Onboard observation archives are retained for verification.')


def configure_lineage(training):
    """Install the same resolver used by the running queue, only in this process."""
    future.install()
    queue.v4 = future
    queue.RUNTIME = ROOT / 'scripts/eval_zhuhai_wam_precision_v4_future16.py'
    queue.validate_binding = lineage.original.validate_binding
    queue.e.validate_binding = lineage.original.validate_binding
    lineage._training = training.resolve(strict=True)
    lineage._lineage = None
    future.read_contract = lineage.read_contract


def validate_plan(plan, training, evaluation):
    analysis.require(Path(plan['training']).resolve() == training.resolve()
        and Path(plan['out']).resolve() == evaluation.resolve(), 'Evaluation directory identity differs')
    analysis.require(plan['seeds'] == list(analysis.SEEDS)
        and [ck['step'] for ck in plan['cohorts']] == list(analysis.STEPS), 'Only the original paired ten are allowed')
    analysis.require(plan['record_video'] is False and plan['third_view'] is False
        and plan['randomized'] is False and plan['max_steps'] == 2400
        and plan.get('export_success_videos') is False
        and plan.get('future_prediction') == future.FUTURE_PREDICTION, 'Frozen Future16 non-video protocol differs')
    expected = [(step, seed) for step in analysis.STEPS for seed in analysis.SEEDS]
    analysis.require([(item['checkpoint']['step'], item['seed']) for item in plan['commands']] == expected,
                     'Planned attempts, checkpoint identity or order differ')
    for item in plan['commands']:
        command = item['command']
        analysis.require(command[command.index('--max-steps') + 1] == '2400'
            and '--record-video' not in command and '--third-view' not in command,
            'Attempt control budget/video protocol differs')


def audited_training(training, plan):
    configure_lineage(training)
    contract, cohorts, audit = queue.audit_training(training)
    analysis.require(lineage._lineage is not None, 'Copied-initialization lineage audit did not run')
    audit['initial_checkpoint_lineage'] = lineage._lineage
    analysis.require(cohorts == plan['cohorts'], 'Independently selected milestones differ')
    analysis.require(audit == plan['training_audit'], 'Independent training/lineage audit differs from admitted plan')
    return contract, cohorts, audit


def source_bindings(plan, evaluation, validation):
    bindings = dict(plan['frozen_sources'])
    # Pin the existing CPU analysis dependency closure as well as the new wrapper.
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and name.startswith(('scripts.', 'harness.')):
            path = Path(path).resolve()
            if ROOT in path.parents and path.suffix == '.py':
                bindings.setdefault(str(path), analysis.sha(path))
    from harness.zhuhai_robot_spec import KINEMATIC_CHAIN, ASSET, ASSET_MANIFEST
    for path in (Path(__file__).resolve(), ROOT / 'scripts/test_finalize_precision_future16_lineage.py',
                 evaluation / 'plan.json', validation, KINEMATIC_CHAIN, ASSET, ASSET_MANIFEST):
        bindings.setdefault(str(path), analysis.sha(path))
    receipt = analysis.read(validation)
    analysis.require(receipt.get('status') == 'passed' and receipt.get('failures') == 0
        and receipt.get('errors') == 0 and receipt.get('tests', 0) >= 12,
        'Successful CPU regression receipt required')
    for path in (Path(__file__).resolve(), ROOT / 'scripts/test_finalize_precision_future16_lineage.py'):
        analysis.require(receipt['source_sha256'].get(str(path)) == bindings[str(path)],
                         'Regression source binding differs')
    analysis.verify_sources(bindings)
    return bindings


def disabled_video_receipt(evaluation):
    path = evaluation / 'success_videos.json'
    value = analysis.read(path)
    analysis.require(value.get('status') == 'disabled_by_default' and value.get('videos') == []
        and value.get('evaluation_unchanged') is True, 'Expected unchanged disabled video export receipt')
    return dict(**value, manifest_path=str(path), manifest_sha256=analysis.sha(path))


def poll_completed(training, evaluation):
    for directory in (training, evaluation):
        for name in ('failure_guard.json', 'owner_retained_guard.json', 'success_video_failure_guard.json'):
            analysis.require(not (directory / name).exists(), 'Preserved failure guard; no retry: ' + str(directory / name))
    path = evaluation / 'summary.json'
    if not path.exists():
        return None
    summary = analysis.read(path)
    analysis.require(summary.get('status') not in ('failed', 'halted'), 'Evaluation halted; preserve all attempts')
    if summary.get('status') != 'completed':
        return None
    analysis.cohort_rows(summary)
    # The queue publishes this after closing its writer and releasing its GPU lock.
    if not (evaluation / 'success_videos.json').exists():
        return None
    disabled_video_receipt(evaluation)
    return summary


def wait_completed(training, evaluation, bindings):
    last_report = 0.
    while True:
        analysis.verify_sources(bindings)
        summary = poll_completed(training, evaluation)
        if summary is not None:
            return summary
        if time.monotonic() - last_report >= 1800:
            print('Future16独立CPU核算等待既定10测完成；视频关闭、无新增物理尝试。', flush=True)
            last_report = time.monotonic()
        time.sleep(30)


def reclassify(plan, summary, contract):
    groups = analysis.cohort_rows(summary)
    fresh_rows = []
    for ck in plan['cohorts']:
        for row in groups[ck['step']]:
            fresh = queue.e.classify(Path(row['raw_result_path']), ck, contract,
                plan['frozen_sources'][str(queue.RUNTIME)], queue.SEEDS, queue.e.classify_noncontact)
            fresh = queue.details(fresh)
            analysis.require(all(row.get(key) == value for key, value in fresh.items()),
                             'Independent raw admission differs')
            fresh.update(checkpoint_step=ck['step'], checkpoint_sha256=ck['sha256'], cohort=ck['label'])
            fresh_rows.append(fresh)
    analysis.require(queue.summary(plan, fresh_rows, 'completed') == summary, 'Reconstructed summary differs')
    return fresh_rows


def finalize(training, evaluation, out, plan, summary, contract, cohorts, audit, bindings):
    summary_hash = analysis.sha(evaluation / 'summary.json')
    rows = reclassify(plan, summary, contract)
    motion = [analysis.analyze_motion(row) for row in rows]
    analysis.write_once(out / 'motion_analysis.json', dict(episodes=motion, extra_trials=0,
        source_summary_sha256=summary_hash, unsafe_terminal_frames_excluded=True, measured_commands_rehashed=True))
    videos = disabled_video_receipt(evaluation)
    baseline = analysis.read(Path(plan['v2_reference']['path']))
    for row in baseline['episodes']:
        baseline_analysis.episode(row)
        analysis.baseline_safe_score(row)
    groups = analysis.cohort_rows(summary)
    table = [analysis.table_row('V2原80k', 80000, baseline['episodes']),
        *[analysis.table_row('Future16 V4', ck['step'], groups[ck['step']]) for ck in cohorts]]
    train_rows = [queue.e.loads(line) for line in (training / 'log.jsonl').read_text().splitlines()]
    metrics = ('loss', 'mse', 'future16', 'future32', 'idm', 'chunk_vtt_loss', 'alpha_after_update', 'seconds_per_step')
    windows = {label: {key: sum(row[key] for row in selected)/len(selected) for key in metrics}
        for label, selected in [('first200', train_rows[:200]), ('last200', train_rows[-200:])]}
    delivery = dict(status='verified_actual_experiments', time=analysis.stamp(), comparison_table=table,
        verdict=analysis.verdict(table), training_audit=audit, loss_windows=windows,
        runtime_performance_profile=contract.get('runtime_performance_profile'),
        failure_stage_counts={str(ck['step']): dict(Counter(row['failure_stage'] for row in motion
            if row['checkpoint_step'] == ck['step'] and not row['full_task_success'])) for ck in cohorts},
        safe_failure_evidence=motion, success_videos=videos, protocol=PROTOCOL,
        source_summary_sha256=summary_hash, source_bindings=bindings,
        limitations=['Fixed five scenes per checkpoint; no generalization or isolated causal attribution.',
            'Synthetic batch throughput excludes data IO; actual training timings are separate.',
            'Batch changed from 4 to 23 after 5000 updates; sample budget changed explicitly.',
            'Swap exhaustion confirmed; historical CPU memory growth root cause not established.'],
        extra_physical_trials=0, further_training_launched=False)
    queue.verify(plan)
    analysis.verify_sources(bindings)
    analysis.require(analysis.sha(evaluation / 'summary.json') == summary_hash, 'Summary changed during independent analysis')
    analysis.write_once(out / 'final_delivery.json', delivery)
    analysis.write_once(out / 'completed.json', dict(status='completed', time=delivery['time'],
        final_delivery_sha256=analysis.sha(out / 'final_delivery.json'), extra_trials=0, video_exports=0))
    message = (f"Future16正式10测已完成并独立核验；最终{table[1]['success_fraction']}，前一权重{table[2]['success_fraction']}；"
        f"视频录制和导出均关闭。{delivery['verdict']['conclusion']} 分析见{out}/final_delivery.json。")
    with (ROOT / 'progress.md').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write('\n' + delivery['time'] + ' ' + message + '\n')
    print(message, flush=True)
    return delivery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    args = parser.parse_args()
    training, evaluation, validation = (path.resolve(strict=True) for path in (args.training, args.evaluation, args.validation))
    out = args.out.absolute()
    analysis.require(queue.BASE in out.parents and not out.exists() and out == out.resolve(), 'Use a fresh project analysis directory')
    plan = analysis.read(evaluation / 'plan.json')
    validate_plan(plan, training, evaluation)
    bindings = source_bindings(plan, evaluation, validation)
    out.mkdir()
    analysis.write_once(out / 'plan.json', dict(training=str(training), evaluation=str(evaluation),
        source_bindings=bindings, protocol=PROTOCOL, poll_seconds=30, saved_plan_sha256=analysis.sha(evaluation / 'plan.json')))
    try:
        queue.verify(plan)
        _, _, audit = audited_training(training, plan)
        analysis.write_once(out / 'startup_training_audit.json', audit)
        print('Future16独立核算初始化谱系及完整训练状态校验通过。', flush=True)
        summary = wait_completed(training, evaluation, bindings)
        contract, cohorts, audit = audited_training(training, plan)
        finalize(training, evaluation, out, plan, summary, contract, cohorts, audit, bindings)
    except BaseException as exc:
        analysis.write_once(out / 'failure_guard.json', dict(error=f'{type(exc).__name__}: {exc}', preserved=True,
            extra_physical_trials=0, video_exports=0))
        raise


if __name__ == '__main__':
    main()
