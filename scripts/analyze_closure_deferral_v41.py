#!/usr/bin/env python3
"""Closure-event deferral diagnosis over archived replans (read-only, CPU).

For every replan of the 80k extra20 cohort plus the v36 (C1) and v37 (RTC)
single cases, take the generated 32-step chunk before dispatch and locate the
first left/right gripper close (<=0.2). Report whether the close target falls
inside the executed 8-step prefix or only in the discarded tail, and the index
trajectory over replan number k (stable offset vs receding vs oscillating vs
approaching). Raw chunks are kept distinct from the spline/C1-smoothed chunk
and from the actually dispatched stream; gripper samples are asserted to be
preserved exactly by smoothing, so only raw + dispatched are counted twice.

No model, simulator, evaluator or GPU is touched; no archived file is modified.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import spearmanr

CLOSE = 0.2
EXEC = 8
HORIZON = 32
GRIPPERS = {'left': 7, 'right': 15}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def require(flag, message):
    if not flag:
        raise RuntimeError(message)


def runs(flags):
    result, start = [], None
    for i, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            result.append([start, i - 1])
            start = None
    return result


def first_close(values):
    for i, v in enumerate(values):
        if v <= CLOSE:
            return i
    return None


def side_stats(recs, dispatched_runs, dispatched_first):
    n = len(recs)
    none = sum(r['i1'] is None for r in recs)
    exec_count = sum(r['i1'] is not None and r['i1'] <= EXEC for r in recs)
    tail_count = sum(r['i1'] is not None and r['i1'] > EXEC for r in recs)
    histogram = Counter(str(r['i1']) for r in recs if r['i1'] is not None)
    with_closure = [r for r in recs if r['i1'] is not None]
    values = [r['i1'] for r in with_closure]
    ks = [r['k'] for r in with_closure]
    conditional_p_exec = exec_count / len(with_closure) if with_closure else None
    trend = None
    if len(values) >= 2:
        x = np.asarray(ks, dtype=float)
        y = np.asarray(values, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0] * 100.0)
        diffs = [b - a for a, b in zip(values, values[1:])]
        increasing = sum(d > 0 for d in diffs)
        decreasing = sum(d < 0 for d in diffs)
        nonzero = [d for d in diffs if d != 0]
        sign_changes = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a * b < 0)
        trend = dict(
            slope_per_100_replans=slope,
            net_drift=values[-1] - values[0],
            first_i1=values[0], last_i1=values[-1],
            frac_increasing=increasing / len(diffs) if diffs else None,
            frac_decreasing=decreasing / len(diffs) if diffs else None,
            sign_changes=sign_changes,
            sign_change_fraction=(sign_changes / (len(nonzero) - 1)) if len(nonzero) > 1 else None,
            spearman_rho=None, spearman_p=None)
        if len(values) >= 4 and len(set(values)) > 1:
            rho, p = spearmanr(ks, values)
            trend['spearman_rho'] = float(rho)
            trend['spearman_p'] = float(p)
    stats = dict(
        replans=n, no_closure_in_32=none, first_close_in_executed_prefix=exec_count,
        first_close_only_in_tail=tail_count, conditional_p_first_close_le_8=conditional_p_exec,
        first_close_index_1based_histogram=dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
        min_i1=min(values) if values else None, max_i1=max(values) if values else None,
        median_i1=float(np.median(values)) if values else None,
        iqr_i1=[float(np.percentile(values, 25)), float(np.percentile(values, 75))] if values else None,
        median_lookahead_seconds=float(np.median(values) / 10.0) if values else None,
        all_closures_in_tail_only=(exec_count == 0 and tail_count > 0),
        no_closure_anywhere=(none == n),
        trend=trend)
    stats['dispatched_close_run_count'] = len(dispatched_runs)
    stats['dispatched_close_runs'] = dispatched_runs
    stats['dispatched_first_close_step'] = dispatched_first
    return stats


def analyze_episode(label, row, out_records):
    directory = Path(row['raw_result_path']).parent
    action_bytes = (directory / 'actions.jsonl').read_bytes()
    require(sha_bytes(action_bytes) == row['evidence_files']['actions.jsonl']['sha256'],
            f'{label} seed {row["seed"]}: actions hash differs')
    actions = [json.loads(line) for line in action_bytes.splitlines()]
    steps = [a['control_step'] for a in actions]
    require(steps == list(range(len(actions))), 'nonconsecutive action archive')
    sent = np.asarray([a['dispatched_action'] for a in actions], dtype=float)
    replans = [a for a in actions if a['policy']['replanned']]
    replan_steps = [a['control_step'] for a in replans]
    deltas = [b - a for a, b in zip(replan_steps, replan_steps[1:])]
    records = {'left': [], 'right': []}
    variant_identity = True
    for k, a in enumerate(replans):
        policy = a['policy']
        require(policy.get('execution_prefix') == EXEC, 'different execution horizon')
        raw = np.asarray(policy['raw_denormalized_chunk'], dtype=float)
        smoothed = np.asarray(policy['smoothed_denormalized_chunk'], dtype=float)
        require(raw.shape == (HORIZON, 16) and smoothed.shape == (HORIZON, 16), 'unexpected chunk shape')
        bspline = policy.get('bspline_denormalized_chunk')
        if bspline is not None:
            bspline = np.asarray(bspline, dtype=float)
        for g in (7, 15):
            if not np.array_equal(raw[:, g], smoothed[:, g]):
                variant_identity = False
            if bspline is not None and not np.array_equal(raw[:, g], bspline[:, g]):
                variant_identity = False
        for side, g in GRIPPERS.items():
            i = first_close(raw[:, g])
            region = 'none' if i is None else ('exec' if i < EXEC else 'tail')
            records[side].append(dict(k=k, control_step=a['control_step'], i0=i,
                                      i1=None if i is None else i + 1, region=region))
    result = dict(cohort=label, seed=row['seed'], status=row['status'],
                  control_steps=len(actions), replans=len(replans),
                  replan_step_deltas={str(k): v for k, v in sorted(Counter(deltas).items())},
                  chunk_variant_gripper_identity=variant_identity,
                  trajectories={'left': [[r['k'], r['i1']] for r in records['left']],
                                'right': [[r['k'], r['i1']] for r in records['right']]})
    for side, g in GRIPPERS.items():
        close_runs = runs(sent[:, g] <= CLOSE)
        first_dispatched = close_runs[0][0] if close_runs else None
        result[side] = side_stats(records[side], close_runs, first_dispatched)
        for r in records[side]:
            out_records.append(dict(cohort=label, seed=row['seed'], side=side,
                                    k=r['k'], control_step=r['control_step'], i0=r['i0'],
                                    i1=r['i1'], region=r['region'],
                                    dispatched_closed_at_step=bool(sent[r['control_step'], g] <= CLOSE)))
    return result


def write_svg(rows, path):
    panel_w, panel_h, pad = 240, 120, 8
    cols = 4
    rows_n = (len(rows) + cols - 1) // cols
    width = cols * panel_w
    height = rows_n * panel_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>']
    for index, (label, left, right) in enumerate(rows):
        cx = (index % cols) * panel_w
        cy = (index // cols) * panel_h
        x0, y0 = cx + pad + 28, cy + pad + 14
        w, h = panel_w - 2 * pad - 34, panel_h - 2 * pad - 26
        def px(k):
            return x0 + k / 299.0 * w
        def py(i):
            return y0 + (32 - i) / 31.0 * h
        parts.append(f'<text x="{x0-26}" y="{y0+6}" font-size="10">32</text>')
        parts.append(f'<text x="{x0-26}" y="{y0+h+4}" font-size="10">1</text>')
        parts.append(f'<line x1="{x0}" y1="{py(8)}" x2="{x0+w}" y2="{py(8)}" stroke="#cc2222" stroke-width="1"/>')
        parts.append(f'<text x="{x0+w+2}" y="{py(8)+3}" font-size="9" fill="#cc2222">8</text>')
        for side, traj, color, width_ in (('R', right, '#bbbbbb', 0.8), ('L', left, '#1a4fbb', 1.2)):
            segment = []
            for k, i1 in traj:
                if i1 is None:
                    if len(segment) > 1:
                        parts.append('<polyline fill="none" stroke="%s" stroke-width="%s" points="%s"/>'
                                     % (color, width_, ' '.join(f'{px(a):.1f},{py(b):.1f}' for a, b in segment)))
                    segment = []
                else:
                    segment.append((k, i1))
            if len(segment) > 1:
                parts.append('<polyline fill="none" stroke="%s" stroke-width="%s" points="%s"/>'
                             % (color, width_, ' '.join(f'{px(a):.1f},{py(b):.1f}' for a, b in segment)))
        parts.append(f'<text x="{x0}" y="{cy+pad+9}" font-size="10">{label}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n')


def aggregate(rows):
    result = {}
    for side in GRIPPERS:
        replans = sum(r['replans'] for r in rows)
        none = sum(r[side]['no_closure_in_32'] for r in rows)
        exec_count = sum(r[side]['first_close_in_executed_prefix'] for r in rows)
        tail = sum(r[side]['first_close_only_in_tail'] for r in rows)
        with_closure = replans - none
        no_close_seeds = sum(r[side]['no_closure_anywhere'] for r in rows)
        tail_only_seeds = sum(r[side]['all_closures_in_tail_only'] for r in rows)
        exec_seeds = sum(r[side]['first_close_in_executed_prefix'] > 0 for r in rows)
        histogram = Counter()
        for r in rows:
            histogram.update(r[side]['first_close_index_1based_histogram'])
        closure_seeds = [r for r in rows if r[side]['no_closure_in_32'] < r['replans']]
        slopes = [r[side]['trend']['slope_per_100_replans'] for r in closure_seeds if r[side]['trend']]
        rhos = [r[side]['trend']['spearman_rho'] for r in closure_seeds
                if r[side]['trend'] and r[side]['trend']['spearman_rho'] is not None]
        widths = [(r[side]['iqr_i1'][1] - r[side]['iqr_i1'][0]) for r in closure_seeds
                  if r[side]['iqr_i1'] is not None]
        medians = [r[side]['median_i1'] for r in closure_seeds if r[side]['median_i1'] is not None]
        result[side] = dict(
            replans=replans, no_closure_in_32=none, first_close_in_executed_prefix=exec_count,
            first_close_only_in_tail=tail,
            conditional_p_first_close_le_8=(exec_count / with_closure) if with_closure else None,
            first_close_index_1based_histogram=dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
            seeds_no_closure_anywhere=no_close_seeds,
            seeds_all_closures_tail_only=tail_only_seeds,
            seeds_with_any_executed_prefix_close=exec_seeds,
            closure_bearing_seeds=len(closure_seeds),
            drift=dict(
                median_slope_per_100_replans=float(np.median(slopes)) if slopes else None,
                share_rho_gt_0_3=sum(x > 0.3 for x in rhos) / len(rhos) if rhos else None,
                share_rho_lt_minus_0_3=sum(x < -0.3 for x in rhos) / len(rhos) if rhos else None,
                share_abs_rho_le_0_3=sum(abs(x) <= 0.3 for x in rhos) / len(rhos) if rhos else None,
                median_abs_rho=float(np.median(np.abs(rhos))) if rhos else None,
                median_iqr_width=float(np.median(widths)) if widths else None,
                median_of_seed_median_i1=float(np.median(medians)) if medians else None),
            dispatched=dict(
                seeds_with_any_dispatched_close=sum(r[side]['dispatched_close_run_count'] > 0 for r in rows),
                total_dispatched_close_runs=sum(r[side]['dispatched_close_run_count'] for r in rows)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-extra20', required=True)
    parser.add_argument('--v36-cohort', required=True)
    parser.add_argument('--v37-summary', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--svg', required=True)
    args = parser.parse_args()
    for name in (args.out, args.csv, args.svg):
        require(not Path(name).exists(), f'preserve existing artifact: {name}')

    sources = {}
    snap = Path(args.summary_extra20).read_bytes()
    extra = json.loads(snap)
    require(extra['status'] == 'completed', 'extra20 cohort incomplete')
    sources['extra20'] = dict(path=str(Path(args.summary_extra20).resolve()),
                              sha256=sha_bytes(snap), rows=extra['episodes'])
    snap = Path(args.v36_cohort).read_bytes()
    v36 = json.loads(snap)
    rows36 = [r for r in v36['episodes'] if r['seed'] == 932106]
    require(len(rows36) == 1, 'v36 first case missing')
    sources['v36_C1'] = dict(path=str(Path(args.v36_cohort).resolve()),
                             sha256=sha_bytes(snap), rows=rows36)
    snap = Path(args.v37_summary).read_bytes()
    v37 = json.loads(snap)
    rows37 = [r for r in v37['episodes'] if r['seed'] == 932106]
    require(len(rows37) == 1, 'v37 first case missing')
    sources['v37_RTC'] = dict(path=str(Path(args.v37_summary).resolve()),
                              sha256=sha_bytes(snap), rows=rows37)

    csv_rows, per_seed = [], []
    for label in ('extra20', 'v36_C1', 'v37_RTC'):
        for row in sources[label]['rows']:
            value = analyze_episode(label, row, csv_rows)
            per_seed.append(value)
            print(json.dumps({k: value[k] for k in ('cohort', 'seed', 'replans',
                  'chunk_variant_gripper_identity')}
                  | {'left': {k: value['left'][k] for k in ('no_closure_in_32',
                     'first_close_in_executed_prefix', 'first_close_only_in_tail',
                     'conditional_p_first_close_le_8', 'median_i1')}}, ensure_ascii=False), flush=True)

    extra_rows = [r for r in per_seed if r['cohort'] == 'extra20']
    value = dict(
        schema='closure-event-deferral-diagnosis-v41',
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        thresholds=dict(closed_le=CLOSE, executed_prefix=EXEC, chunk_horizon=HORIZON),
        sources={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} | {'episode_count': len(v['rows'])}
                 for k, v in sources.items()},
        per_seed=per_seed,
        aggregate_extra20=aggregate(extra_rows),
        interpretation=[
            'i_k is the first index (1-based) in the generated 32-step chunk whose gripper value is <=0.2; regions: 1..8 executed prefix, 9..32 discarded tail, none.',
            'Gripper samples are asserted identical across raw/spline/C1-smoothed chunks, so raw, bspline and smoothed share the same i_k by construction; only the dispatched stream is a separate stream.',
            'Trajectories are (replan index k, i_k) pairs; null i_k means no close anywhere in that chunk.',
            'Trend statistics use only replans whose chunk contains a close; spearman needs >=4 points and variation.',
            'Dispatched close runs are runs of dispatched_action gripper <=0.2 in the executed stream.',
        ],
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with Path(args.csv).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cohort', 'seed', 'side', 'k', 'control_step',
                                                    'i0', 'i1', 'region', 'dispatched_closed_at_step'])
        writer.writeheader()
        writer.writerows(csv_rows)
    svg_rows = []
    for r in per_seed:
        left = [tuple(p) for p in r['trajectories']['left']]
        right = [tuple(p) for p in r['trajectories']['right']]
        svg_rows.append((f'{r["cohort"]} seed {r["seed"]}', left, right))
    write_svg(svg_rows, Path(args.svg))
    print(json.dumps(dict(out=str(out.resolve()), csv=str(Path(args.csv).resolve()),
                          svg=str(Path(args.svg).resolve()),
                          aggregate=value['aggregate_extra20']['left']), ensure_ascii=False))


if __name__ == '__main__':
    main()
