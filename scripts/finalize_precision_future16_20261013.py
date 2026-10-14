#!/usr/bin/env python3
"""Wait for scored Future16 trials; independently analyze and export actual successes."""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from scripts import finalize_precision_v4_r2 as analysis
    from scripts import eval_precision_v4_final5 as queue
    from scripts import eval_precision_v4_future16_final5 as future_queue
    from harness import zhuhai_precision_v4_future16 as future
    from scripts.progress1_log_v41 import log
    future.install()
    queue.v4 = future
    queue.RUNTIME = ROOT / 'scripts/eval_zhuhai_wam_precision_v4_future16.py'
    queue.validate_binding = future_queue.validate_binding
    queue.e.validate_binding = future_queue.validate_binding
    read, write = analysis.read, analysis.write_once
    bindings = {str(Path(__file__).resolve()): analysis.sha(Path(__file__).resolve()),
                str(ROOT / 'scripts/finalize_precision_v4_r2.py'): analysis.sha(ROOT / 'scripts/finalize_precision_v4_r2.py')}
    write(args.out / 'plan.json', dict(training=str(args.training), evaluation=str(args.evaluation),
        source_bindings=bindings, video_from_original_success_archives=True, extra_trials=0,
        record_evaluation_video=False, poll_seconds=30))
    try:
        last_report = -1800.
        while True:
            analysis.verify_sources(bindings)
            for directory in (args.training, args.evaluation):
                analysis.require(not (directory / 'failure_guard.json').exists(), 'Failure guard: ' + str(directory))
            if (args.evaluation / 'summary.json').exists():
                summary = read(args.evaluation / 'summary.json')
                analysis.require(summary.get('status') != 'halted', 'Evaluation halted; preserve attempts')
                if summary.get('status') == 'completed':
                    break
            if time.monotonic() - last_report >= 1800:
                log('Future16最终分析/成功录像CPU等待正式10测完成；不录像、不新增仿真。', 'future16_finalizer')
                last_report = time.monotonic()
            time.sleep(30)
        plan = read(args.evaluation / 'plan.json')
        queue.verify(plan)
        contract, cohorts, training_audit = queue.audit_training(args.training)
        rows = summary['episodes']
        analysis.require(len(rows) == 10 and summary['record_video'] is False and summary['third_view'] is False,
                         'Exactly original ten non-video trials required')
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
        motion = [analysis.analyze_motion(row) for row in fresh_rows]
        write(args.out / 'motion_analysis.json', dict(episodes=motion, extra_trials=0,
            source_summary_sha256=analysis.sha(args.evaluation / 'summary.json'),
            unsafe_terminal_frames_excluded=True, measured_commands_rehashed=True))
        support = args.evaluation / 'success_video_evidence'
        support.mkdir(exist_ok=False)
        videos = [queue.export_success(row, support) for row in fresh_rows if row['full_task_success']]
        disabled = args.evaluation / 'success_videos.json'
        if disabled.exists():
            disabled.rename(args.evaluation / 'success_videos_disabled_original.json')
        write(disabled, dict(status='completed', videos=videos, no_full_task_success=not videos,
            evaluation_unchanged=True, user_requested_after_test=True))
        verified_videos = analysis.verify_videos(args.evaluation, fresh_rows)
        baseline = read(Path(plan['v2_reference']['path']))
        from scripts import analyze_precision_dagger_milestones_v19 as original_analysis
        for row in baseline['episodes']:
            original_analysis.episode(row)
            analysis.baseline_safe_score(row)
        table = [analysis.table_row('V2原80k', 80000, baseline['episodes']),
                 *[analysis.table_row('Future16 V4', ck['step'], groups[ck['step']]) for ck in cohorts]]
        train_rows = [json.loads(line) for line in (args.training / 'log.jsonl').read_text().splitlines()]
        metrics = ('loss', 'mse', 'future16', 'future32', 'idm', 'chunk_vtt_loss',
                   'alpha_after_update', 'seconds_per_step')
        windows = {label: {key: sum(row[key] for row in selected)/len(selected) for key in metrics}
            for label, selected in [('first200', train_rows[:200]), ('last200', train_rows[-200:])]}
        delivery = dict(status='verified_actual_experiments', time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            comparison_table=table, verdict=analysis.verdict(table), training_audit=training_audit,
            loss_windows=windows, runtime_performance_profile=contract.get('runtime_performance_profile'),
            failure_stage_counts={str(ck['step']): dict(Counter(row['failure_stage'] for row in motion
                if row['checkpoint_step']==ck['step'] and not row['full_task_success'])) for ck in cohorts},
            safe_failure_evidence=motion, success_videos=verified_videos,
            source_summary_sha256=analysis.sha(args.evaluation / 'summary.json'),
            limitations=['Fixed five scenes per checkpoint; no generalization or isolated causal attribution.',
                        'Synthetic batch throughput excludes data IO; actual training timings are separate.',
                        'Batch changed from 4 to 23 after 5000 updates; sample budget changed explicitly.',
                        'Swap exhaustion confirmed; historical CPU memory growth root cause not established.'],
            extra_physical_trials=0, further_training_launched=False)
        analysis.verify_sources(bindings)
        write(args.out / 'final_delivery.json', delivery)
        message = f"Future16正式10测已完成并独立核验；最终{table[1]['success_fraction']}，前一权重{table[2]['success_fraction']}；成功原速录像{len(videos)}份。{delivery['verdict']['conclusion']} 分析见{args.out}/final_delivery.json。"
        for relative in ('docs/current_status.md', 'docs/full_middle_execution_prompt_20261007.md'):
            path = ROOT / relative
            path.write_text('<!-- future16-final-delivery:start -->\n' + message +
                '\n<!-- future16-final-delivery:end -->\n\n' + path.read_text())
        with (ROOT / 'progress.md').open('a') as stream:
            stream.write('\n' + delivery['time'] + ' ' + message + '\n')
        log(message, 'future16_finalizer')
        write(args.out / 'completed.json', dict(status='completed', time=delivery['time'],
            final_delivery_sha256=analysis.sha(args.out / 'final_delivery.json'), extra_trials=0))
        print(message, flush=True)
    except BaseException as exc:
        write(args.out / 'failure_guard.json', dict(error=f'{type(exc).__name__}: {exc}', preserved=True))
        raise

if __name__ == '__main__':
    main()
