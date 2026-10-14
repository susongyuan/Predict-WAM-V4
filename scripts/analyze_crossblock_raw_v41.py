#!/usr/bin/env python3
"""Cross-block generation discontinuity at the exec-8 horizon (read-only, CPU).

For consecutive replans k, k+1 with control steps s_k and s_{k+1}=s_k+8,
chunk index j of replan k is dispatched at control step s_k+j and refers to
physical time (s_k+j+1)*0.1 s, so chunk_k[8..31] and chunk_{k+1}[0..23] refer
to the same physical instants. This script measures the aligned mismatch
|position| and |velocity| (10 Hz finite difference) at j=0 against within-chunk
adjacent-step statistics, plus the mismatch profile over j and the executed
(dispatched) boundary step. Raw model chunks, per-chunk B-spline chunks,
C1-smoothed chunks and the dispatched stream are kept separate.

No model/simulator/GPU call; archived files are only read.
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

ARM = [*range(7), *range(8, 15)]
CONTROL_DT = 0.1
EXEC = 8
HORIZON = 32


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def require(flag, message):
    if not flag:
        raise RuntimeError(message)


def dist_summary(values):
    array = np.asarray(values, dtype=float)
    if not array.size:
        return None
    return dict(mean=float(array.mean()), median=float(np.median(array)),
                p95=float(np.percentile(array, 95)), maximum=float(array.max()))


def load_sources(args):
    sources = {}
    snap = Path(args.summary_extra20).read_bytes()
    extra = json.loads(snap)
    require(extra['status'] == 'completed', 'extra20 cohort incomplete')
    sources['extra20'] = dict(path=str(Path(args.summary_extra20).resolve()),
                              sha256=sha_bytes(snap), rows=extra['episodes'])
    snap = Path(args.v36_cohort).read_bytes()
    v36 = json.loads(snap)
    rows = [r for r in v36['episodes'] if r['seed'] == 932106]
    require(len(rows) == 1, 'v36 first case missing')
    sources['v36_C1'] = dict(path=str(Path(args.v36_cohort).resolve()),
                             sha256=sha_bytes(snap), rows=rows)
    snap = Path(args.v37_summary).read_bytes()
    v37 = json.loads(snap)
    rows = [r for r in v37['episodes'] if r['seed'] == 932106]
    require(len(rows) == 1, 'v37 first case missing')
    sources['v37_RTC'] = dict(path=str(Path(args.v37_summary).resolve()),
                              sha256=sha_bytes(snap), rows=rows)
    return sources


def variant_summary(jump_pos, ratio_pos, jump_vel, ratio_vel, exceed_fraction,
                    profiles_pos, profiles_vel, gap_start):
    ratio_pos_a = np.asarray([r for r in ratio_pos if r is not None], dtype=float)
    ratio_vel_a = np.asarray([r for r in ratio_vel if r is not None], dtype=float)
    return dict(
        pairs=len(jump_pos),
        jump_pos0_rad=dist_summary(jump_pos),
        ratio_pos=dist_summary(ratio_pos_a),
        jump_vel0_rad_s=dist_summary(jump_vel),
        ratio_vel=dist_summary(ratio_vel_a),
        fraction_ratio_pos_gt_1=float((ratio_pos_a > 1).mean()) if ratio_pos_a.size else None,
        fraction_ratio_pos_gt_2=float((ratio_pos_a > 2).mean()) if ratio_pos_a.size else None,
        fraction_ratio_pos_gt_5=float((ratio_pos_a > 5).mean()) if ratio_pos_a.size else None,
        fraction_jump_exceeds_within_max=exceed_fraction,
        chunk_start_to_last_dispatched_rad=dist_summary(gap_start),
        profile_pos_j0_to_j23=[float(x) for x in np.mean(np.asarray(profiles_pos), axis=0)],
        profile_vel_j0_to_j22=[float(x) for x in np.mean(np.asarray(profiles_vel), axis=0)])


def analyze_episode(label, row, csv_rows):
    directory = Path(row['raw_result_path']).parent
    action_bytes = (directory / 'actions.jsonl').read_bytes()
    require(sha_bytes(action_bytes) == row['evidence_files']['actions.jsonl']['sha256'],
            f'{label} seed {row["seed"]}: actions hash differs')
    actions = [json.loads(line) for line in action_bytes.splitlines()]
    require([a['control_step'] for a in actions] == list(range(len(actions))), 'nonconsecutive actions')
    sent = np.asarray([a['dispatched_action'] for a in actions], dtype=float)
    replans = [a for a in actions if a['policy']['replanned']]
    steps = [a['control_step'] for a in replans]
    deltas = [b - a for a, b in zip(steps, steps[1:])]
    require(all(d == EXEC for d in deltas), f'replan spacing differs: {Counter(deltas)}')
    fields = {'raw': 'raw_denormalized_chunk', 'smoothed': 'smoothed_denormalized_chunk'}
    if 'bspline_denormalized_chunk' in replans[0]['policy']:
        fields['bspline'] = 'bspline_denormalized_chunk'
    result = dict(cohort=label, seed=row['seed'], status=row['status'],
                  control_steps=len(actions), replans=len(replans), variants={})
    for name, field in fields.items():
        chunks = np.asarray([a['policy'][field] for a in replans], dtype=float)
        require(chunks.shape == (len(replans), HORIZON, 16), 'unexpected chunk shape')
        arm = chunks[:, :, ARM]
        velocity = np.diff(arm, axis=1) / CONTROL_DT
        jump_pos, ratio_pos, jump_vel, ratio_vel = [], [], [], []
        profiles_pos, profiles_vel, gap_start = [], [], []
        exceed = 0
        for k in range(len(replans) - 1):
            d = steps[k + 1] - steps[k]
            align = HORIZON - d
            delta = arm[k + 1, :align] - arm[k, d:d + align]
            vdelta = velocity[k + 1, :align - 1] - velocity[k, d:d + align - 1]
            npos = np.linalg.norm(delta, axis=1)
            nvel = np.linalg.norm(vdelta, axis=1)
            within = np.concatenate([np.linalg.norm(np.diff(arm[k], axis=0), axis=1),
                                     np.linalg.norm(np.diff(arm[k + 1], axis=0), axis=1)])
            vwithin = np.concatenate([np.linalg.norm(np.diff(velocity[k], axis=0), axis=1),
                                      np.linalg.norm(np.diff(velocity[k + 1], axis=0), axis=1)])
            med = float(np.median(within))
            medv = float(np.median(vwithin))
            jump_pos.append(float(npos[0]))
            ratio_pos.append(float(npos[0] / med) if med > 0 else None)
            jump_vel.append(float(nvel[0]))
            ratio_vel.append(float(nvel[0] / medv) if medv > 0 else None)
            exceed += int(npos[0] > float(within.max()))
            profiles_pos.append(npos)
            profiles_vel.append(nvel)
            gap = float(np.linalg.norm(chunks[k + 1, 0, ARM] - sent[steps[k + 1] - 1, ARM]))
            gap_start.append(gap)
            csv_rows.append(dict(cohort=label, seed=row['seed'], variant=name, k=k,
                                 control_step_next=steps[k + 1], jump_pos0=npos[0],
                                 within_pos_median=med, ratio_pos=(npos[0] / med) if med > 0 else '',
                                 jump_vel0=nvel[0], within_vel_median=medv,
                                 ratio_vel=(nvel[0] / medv) if medv > 0 else '',
                                 chunk_start_to_last_dispatched_rad=gap))
        summary = variant_summary(jump_pos, ratio_pos, jump_vel, ratio_vel,
                                  exceed / max(1, len(jump_pos)), profiles_pos, profiles_vel, gap_start)
        result['variants'][name] = summary
    boundary = set(steps[1:])
    boundary_norm = [float(np.linalg.norm(sent[t, ARM] - sent[t - 1, ARM])) for t in steps[1:]]
    within_norm = [float(np.linalg.norm(sent[t, ARM] - sent[t - 1, ARM]))
                   for t in range(1, len(actions)) if t not in boundary]
    within_p95 = float(np.percentile(within_norm, 95)) if within_norm else None
    result['executed_stream'] = dict(
        boundary_step_norm_rad=dist_summary(boundary_norm),
        within_step_norm_rad=dist_summary(within_norm),
        boundary_over_within_mean=(float(np.mean(boundary_norm) / np.mean(within_norm))
                                   if within_norm and np.mean(within_norm) > 0 else None),
        fraction_boundary_above_within_p95=(float(np.mean([x > within_p95 for x in boundary_norm]))
                                            if within_p95 else None))
    return result


def aggregate(rows):
    out = {}
    names = sorted({name for r in rows for name in r['variants']})
    for name in names:
        selected = [r['variants'][name] for r in rows if name in r['variants']]
        med_ratio = [s['ratio_pos']['median'] for s in selected]
        med_vel = [s['ratio_vel']['median'] for s in selected]
        med_jump = [s['jump_pos0_rad']['median'] for s in selected]
        frac_gt2 = [s['fraction_ratio_pos_gt_2'] for s in selected]
        frac_exceed = [s['fraction_jump_exceeds_within_max'] for s in selected]
        out[name] = dict(
            seeds=len(selected),
            median_of_seed_median_ratio_pos=float(np.median(med_ratio)),
            median_of_seed_median_ratio_vel=float(np.median(med_vel)),
            median_of_seed_median_jump_pos0_rad=float(np.median(med_jump)),
            mean_of_seed_fraction_ratio_gt_2=float(np.mean(frac_gt2)),
            mean_of_seed_fraction_jump_exceeds_within_max=float(np.mean(frac_exceed)))
    return out


def v30_reference(path):
    doc = json.loads(Path(path).read_text())
    rows = []
    for episode in doc['cohorts']['80000extra20']['episodes']:
        both = episode['motion']['measured_after_action']['both_arms']
        rows.append(dict(seed=episode['seed'],
                         measured_boundary_step_norm_mean_rad=both['boundary_step_norm_mean_rad'],
                         measured_within_step_norm_mean_rad=both['within_step_norm_mean_rad'],
                         measured_boundary_over_within=both['boundary_to_within_step_norm_ratio'],
                         measured_acceleration_energy_fraction_replan_frames=both['replan_or_next_acceleration_energy_fraction']))
    energies = [r['measured_acceleration_energy_fraction_replan_frames'] for r in rows
                if r['measured_acceleration_energy_fraction_replan_frames'] is not None]
    ratios = [r['measured_boundary_over_within'] for r in rows if r['measured_boundary_over_within'] is not None]
    return dict(source=str(Path(path).resolve()), per_seed=rows,
                mean_acceleration_energy_fraction=float(np.mean(energies)) if energies else None,
                min_acceleration_energy_fraction=float(np.min(energies)) if energies else None,
                max_acceleration_energy_fraction=float(np.max(energies)) if energies else None,
                mean_measured_boundary_over_within=float(np.mean(ratios)) if ratios else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-extra20', required=True)
    parser.add_argument('--v36-cohort', required=True)
    parser.add_argument('--v37-summary', required=True)
    parser.add_argument('--v30', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()
    for name in (args.out, args.csv):
        require(not Path(name).exists(), f'preserve existing artifact: {name}')
    sources = load_sources(args)
    csv_rows, per_seed = [], []
    for label in ('extra20', 'v36_C1', 'v37_RTC'):
        for row in sources[label]['rows']:
            value = analyze_episode(label, row, csv_rows)
            per_seed.append(value)
            print(json.dumps(dict(cohort=label, seed=value['seed'],
                                  raw_ratio_pos=value['variants']['raw']['ratio_pos']['median'],
                                  raw_jump_pos0=value['variants']['raw']['jump_pos0_rad']['median'],
                                  smoothed_ratio_pos=value['variants']['smoothed']['ratio_pos']['median'],
                                  executed_ratio=value['executed_stream']['boundary_over_within_mean'])),
                  flush=True)
    extra_rows = [r for r in per_seed if r['cohort'] == 'extra20']
    value = dict(
        schema='crossblock-raw-generation-discontinuity-v41',
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        settings=dict(align_offset_steps=EXEC, horizon=HORIZON, control_dt_seconds=CONTROL_DT,
                      arm_joints=ARM,
                      alignment='chunk_k[8..31] vs chunk_{k+1}[0..23] (same physical instants)',
                      velocity='10 Hz finite difference within each chunk, compared at aligned indices',
                      within_statistic='median over adjacent-step norms of both aligned chunks',
                      ratio='aligned j=0 mismatch norm / within-chunk median adjacent-step norm'),
        sources={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} | {'episode_count': len(v['rows'])}
                 for k, v in sources.items()},
        per_seed=per_seed,
        aggregate_extra20=aggregate(extra_rows),
        v30_measured_cross_reference=v30_reference(args.v30),
        interpretation=[
            'j=0 of the alignment is the exec-8 horizon: chunk_{k+1}[0] and chunk_k[8] are two generations of the same physical instant; their difference is the prediction disagreement that becomes an executed jump unless rate limited.',
            'Ratio >1 means the seam disagreement exceeds a typical within-chunk 0.1 s step of the same chunks.',
            'The C1-smoothed chunk is position/velocity anchored to the last dispatched target, so its j=0 ratio is expected to collapse; the raw ratio is the generation-side quantity.',
            'The executed stream boundary step is the rate-limited consequence; measured acceleration-energy concentration is cross-referenced from the archived v30 diagnostic, not recomputed here.',
        ],
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with Path(args.csv).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cohort', 'seed', 'variant', 'k', 'control_step_next',
                                                    'jump_pos0', 'within_pos_median', 'ratio_pos',
                                                    'jump_vel0', 'within_vel_median', 'ratio_vel',
                                                    'chunk_start_to_last_dispatched_rad'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(dict(out=str(out.resolve()), csv=str(Path(args.csv).resolve()),
                          aggregate=value['aggregate_extra20'],
                          v30=dict(mean_acceleration_energy_fraction=value['v30_measured_cross_reference']['mean_acceleration_energy_fraction'],
                                   mean_measured_boundary_over_within=value['v30_measured_cross_reference']['mean_measured_boundary_over_within'])),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
