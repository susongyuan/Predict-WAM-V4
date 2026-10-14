#!/usr/bin/env python3
"""Garbage-time quantification on archived episodes (read-only, CPU).

Definitions follow arm_boundary_and_box_visits_v25.json /
final20_boundary_boxes_v30.json where applicable: tool reference point from
CPU FK on measured joints (tool offset included); white/green/blue interior
AABB expanded +2 cm in XY and from interior bottom to top +25 cm in Z; a
retained visit is >=3 consecutive archived observations inside the region.
(a) empty-hand white-box round trips: side tool inside the expanded white
    region with that side commanded gripper open (>=0.8) and zero task-object
    gripper contact (either gripper) throughout the visit.
(b) approach-without-close: side tool outside-AABB distance to a task object
    < threshold for >=1 s while that side commanded gripper is never <=0.2 in
    the window (5 cm primary, 2/10 cm sensitivity).
(c) bowl-completed stall: for seeds with a valid bowl placement, remaining
    steps, min tool-to-spoon distance, spoon contact frames, next closure
    attempts (planned in raw chunks and dispatched).
(d) sustained shaking: tool-point direction reversals (>120 deg turn, both
    displacement segments >1 cm) in windows with no contact and no approach,
    reported per minute, with a variant that also excludes box interiors.
Effective task time (contact / approach / inside any box) is reported
separately. No model/simulator/GPU; archived files only read.
"""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.isaac_zhuhai_expert import FrankaFK

OBJECTS = ('bowl', 'spoon', 'cup', 'plate')
SIDES = {'left': (slice(0, 7), 7), 'right': (slice(8, 15), 15)}
APPROACH_PRIMARY = 0.05
APPROACH_SENSITIVITY = (0.02, 0.05, 0.10)
MIN_VISIT = 3
MIN_APPROACH_OBS = 10
REVERSAL_ANGLE_DEG = 120.0
REVERSAL_AMPLITUDE_M = 0.01


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
            result.append((start, i - 1))
            start = None
    return result


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


def analyze_episode(label, row, fk, csv_rows, v30_lookup):
    directory = Path(row['raw_result_path']).parent
    evidence = row['evidence_files']
    action_bytes = (directory / 'actions.jsonl').read_bytes()
    require(sha_bytes(action_bytes) == evidence['actions.jsonl']['sha256'],
            f'{label} seed {row["seed"]}: actions hash differs')
    rule_bytes = (directory / 'rule_evidence.jsonl').read_bytes()
    require(sha_bytes(rule_bytes) == evidence['rule_evidence.jsonl']['sha256'],
            f'{label} seed {row["seed"]}: rule hash differs')
    frame_bytes = (directory / 'onboard/frames.jsonl').read_bytes()
    require(sha_bytes(frame_bytes) == evidence['onboard/frames.jsonl']['sha256'],
            f'{label} seed {row["seed"]}: frames hash differs')
    actions = [json.loads(line) for line in action_bytes.splitlines()]
    rules = [json.loads(line) for line in rule_bytes.splitlines()]
    frames = [json.loads(line) for line in frame_bytes.splitlines()]
    n = len(actions)
    require([a['control_step'] for a in actions] == list(range(n)), 'actions nonconsecutive')
    require(len(rules) == n + 1 and len(frames) == n + 1, 'observation counts differ')
    require([r['control_step'] for r in rules] == list(range(n + 1)), 'rules nonconsecutive')
    require([f['control_step'] for f in frames] == list(range(n + 1)), 'frames nonconsecutive')
    q = np.empty((n + 1, 16))
    commanded = np.empty((n + 1, 16))
    for frame in frames:
        name = 'onboard/' + frame['file']
        blob = (directory / 'onboard' / frame['file']).read_bytes()
        require(sha_bytes(blob) == evidence[name]['sha256'],
                f'{label} seed {row["seed"]}: npz hash differs')
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            t = frame['control_step']
            q[t] = archive['measured_joint_gripper_state'].astype(float)
            commanded[t] = archive['commanded_joint_gripper_state'].astype(float)
    sent = np.asarray([a['dispatched_action'] for a in actions], dtype=float)
    tools = {}
    for side, (sl, g) in SIDES.items():
        points = np.empty((n + 1, 3))
        for t in range(n + 1):
            points[t] = fk.fk(side, q[t, sl])[:3, 3]
        tools[side] = points
    obj_lo = {k: np.empty((n + 1, 3)) for k in OBJECTS}
    obj_hi = {k: np.empty((n + 1, 3)) for k in OBJECTS}
    contact_any = np.zeros(n + 1, dtype=bool)
    for t, rule in enumerate(rules):
        for k in OBJECTS:
            corners = np.asarray(rule['objects'][k]['world_corners_m'], dtype=float)
            obj_lo[k][t] = corners.min(axis=0)
            obj_hi[k][t] = corners.max(axis=0)
            contact_any[t] |= bool(rule['objects'][k]['touching_gripper'])
    dist = {}
    for side in SIDES:
        dist[side] = {}
        for k in OBJECTS:
            outside = np.maximum(np.maximum(obj_lo[k] - tools[side], tools[side] - obj_hi[k]), 0.0)
            dist[side][k] = np.linalg.norm(outside, axis=1)
    min_dist_side = {side: np.min(np.stack([dist[side][k] for k in OBJECTS]), axis=0)
                     for side in SIDES}
    min_dist_any = np.minimum(min_dist_side['left'], min_dist_side['right'])
    box_regions = {}
    for color, (lo, hi) in rules[0]['destination_interiors_world_m'].items():
        box_regions[color] = (np.asarray(lo, dtype=float) + np.array([-0.02, -0.02, 0.0]),
                              np.asarray(hi, dtype=float) + np.array([0.02, 0.02, 0.25]))
    inside = {}
    for side in SIDES:
        inside[side] = {color: np.all((tools[side] >= lo) & (tools[side] <= hi), axis=1)
                        for color, (lo, hi) in box_regions.items()}
    inside_any = {side: np.logical_or.reduce([inside[side][color] for color in box_regions])
                  for side in SIDES}
    result = dict(cohort=label, seed=row['seed'], status=row['status'], control_steps=n,
                  observation_count=n + 1, episode_seconds=(n + 1) / 10.0,
                  box_regions={}, approach_no_close={}, shaking={}, effective_time={},
                  bowl_completion_stall=None, v30_cross_check={})
    for side, (sl, g) in SIDES.items():
        commanded_open = commanded[:, g] >= 0.8
        commanded_closed = commanded[:, g] <= 0.2
        measured_open = q[:, g] >= 0.8
        box_info = {}
        for color in box_regions:
            region = inside[side][color]
            region_runs = runs(region)
            retained = [(s, e) for s, e in region_runs if e - s + 1 >= MIN_VISIT]
            retained_no_contact = sum(1 for s, e in retained if not contact_any[s:e + 1].any())
            empty_hand = []
            for s, e in region_runs:
                if e - s + 1 < MIN_VISIT:
                    continue
                if not (commanded_open[s:e + 1].all() and not contact_any[s:e + 1].any()):
                    continue
                detail = dict(start_observation=s, end_observation=e, seconds=(e - s + 1) / 10.0,
                              measured_gripper_open_throughout=bool(measured_open[s:e + 1].all()),
                              min_distance_any_task_object_m=float(min_dist_any[s:e + 1].min()))
                empty_hand.append(detail)
                csv_rows.append(dict(category='a_empty_hand_box', cohort=label, seed=row['seed'],
                                     arm=side, color=color, start_obs=s, end_obs=e,
                                     seconds=(e - s + 1) / 10.0,
                                     detail=f"min_dist_any={detail['min_distance_any_task_object_m']:.4f}"))
            box_info[color] = dict(
                all_runs=len(region_runs), retained_visits_min3=len(retained),
                retained_without_any_contact=retained_no_contact,
                empty_hand_visits=len(empty_hand),
                empty_hand_seconds=sum(v['seconds'] for v in empty_hand),
                empty_hand_measured_open_seconds=sum(v['seconds'] for v in empty_hand
                                                     if v['measured_gripper_open_throughout']),
                empty_hand_visits_detail=empty_hand if color == 'white' else [])
        result['box_regions'][side] = box_info
        approach = {}
        for threshold in APPROACH_SENSITIVITY:
            windows = []
            for s, e in runs(min_dist_side[side] < threshold):
                if e - s + 1 < MIN_APPROACH_OBS:
                    continue
                if commanded_closed[s:e + 1].any():
                    continue
                nearest = min(OBJECTS, key=lambda k: float(dist[side][k][s:e + 1].min()))
                window = dict(start_observation=s, end_observation=e, seconds=(e - s + 1) / 10.0,
                              nearest_object=nearest,
                              min_distance_m=float(min_dist_side[side][s:e + 1].min()),
                              any_task_object_contact_in_window=bool(contact_any[s:e + 1].any()))
                windows.append(window)
                csv_rows.append(dict(category='b_approach_no_close', cohort=label, seed=row['seed'],
                                     arm=side, color=nearest, start_obs=s, end_obs=e,
                                     seconds=window['seconds'],
                                     detail=f"thr={threshold},min_dist={window['min_distance_m']:.4f}"))
            approach[f'{threshold:.2f}'] = dict(
                windows=len(windows), seconds=sum(w['seconds'] for w in windows),
                windows_with_contact=sum(w['any_task_object_contact_in_window'] for w in windows),
                window_details=windows if threshold == APPROACH_PRIMARY else [])
        result['approach_no_close'][side] = approach
        displacement = tools[side][1:] - tools[side][:-1]
        norms = np.linalg.norm(displacement, axis=1)
        shaking = {}
        for tag, window_flags in (
                ('exclude_box_interiors',
                 (~contact_any) & (min_dist_side[side] >= APPROACH_PRIMARY) & (~inside_any[side])),
                ('any_free_space',
                 (~contact_any) & (min_dist_side[side] >= APPROACH_PRIMARY))):
            count = 0
            for i in range(1, len(displacement)):
                if not (window_flags[i - 1] and window_flags[i] and window_flags[i + 1]):
                    continue
                if norms[i - 1] <= REVERSAL_AMPLITUDE_M or norms[i] <= REVERSAL_AMPLITUDE_M:
                    continue
                cosine = float(np.dot(displacement[i - 1], displacement[i]) / (norms[i - 1] * norms[i]))
                if np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) > REVERSAL_ANGLE_DEG:
                    count += 1
            seconds = float(window_flags.sum()) / 10.0
            shaking[tag] = dict(reversals=count, window_seconds=seconds,
                                reversals_per_minute=(count / (seconds / 60.0)) if seconds > 0 else None)
            csv_rows.append(dict(category='d_shaking', cohort=label, seed=row['seed'], arm=side,
                                 color='-', start_obs=0, end_obs=n, seconds=seconds,
                                 detail=f"{tag}: reversals={count}, per_min="
                                        f"{shaking[tag]['reversals_per_minute']}"))
        result['shaking'][side] = shaking
    effective = contact_any | (min_dist_any < APPROACH_PRIMARY) | inside_any['left'] | inside_any['right']
    result['effective_time'] = dict(
        effective_seconds=float(effective.sum()) / 10.0,
        garbage_seconds=float((~effective).sum()) / 10.0,
        effective_fraction=float(effective.mean()))
    placement = next((t for t, rule in enumerate(rules)
                      if rule['score']['items']['bowl']['placement_valid']), None)
    if placement is not None:
        after = slice(placement + 1, n + 1)
        replans_after = [a for a in actions if a['control_step'] >= placement and a['policy']['replanned']]
        close_plan_any = close_plan_exec = right_close_plan = 0
        for a in replans_after:
            raw = np.asarray(a['policy']['raw_denormalized_chunk'], dtype=float)
            left = raw[:, 7] <= 0.2
            right = raw[:, 15] <= 0.2
            close_plan_any += int(left.any() or right.any())
            close_plan_exec += int(left[:8].any() or right[:8].any())
            right_close_plan += int(right.any())
        result['bowl_completion_stall'] = dict(
            first_bowl_placement_observation=placement,
            physics_seconds_at_placement=rules[placement]['physics_time_seconds'],
            remaining_control_steps=n - placement,
            remaining_seconds=rules[-1]['physics_time_seconds'] - rules[placement]['physics_time_seconds'],
            min_tool_to_spoon_AABB_m={side: float(dist[side]['spoon'][after].min()) for side in SIDES},
            spoon_contact_observations_after=int(sum(
                rules[t]['objects']['spoon']['touching_gripper'] for t in range(placement + 1, n + 1))),
            object_contact_observations_after={k: int(sum(
                rules[t]['objects'][k]['touching_gripper'] for t in range(placement + 1, n + 1)))
                for k in OBJECTS},
            replans_after=len(replans_after),
            replans_planning_any_gripper_close=close_plan_any,
            replans_planning_close_in_executed_prefix=close_plan_exec,
            replans_planning_right_gripper_close=right_close_plan,
            dispatched_close_runs_after={'left': [[s, e] for s, e in runs(sent[:, 7] <= 0.2) if s >= placement],
                                         'right': [[s, e] for s, e in runs(sent[:, 15] <= 0.2) if s >= placement]},
            dispatched_closed_observations_after={'left': int((sent[placement:, 7] <= 0.2).sum()),
                                                  'right': int((sent[placement:, 15] <= 0.2).sum())})
        csv_rows.append(dict(category='c_bowl_stall', cohort=label, seed=row['seed'], arm='both',
                             color='-', start_obs=placement, end_obs=n,
                             seconds=result['bowl_completion_stall']['remaining_seconds'],
                             detail=f"spoon_contact={result['bowl_completion_stall']['spoon_contact_observations_after']}"))
    if v30_lookup is not None and row['seed'] in v30_lookup:
        reference = v30_lookup[row['seed']]
        result['v30_cross_check'] = {
            side: dict(
                v30_retained_white_visits=reference[side]['white']['retained_visits_minimum3_observations'],
                v30_retained_white_visits_without_contact=reference[side]['white']['retained_visits_without_any_task_object_gripper_contact'],
                v41_retained_white_visits=result['box_regions'][side]['white']['retained_visits_min3'],
                v41_retained_white_visits_without_contact=result['box_regions'][side]['white']['retained_without_any_contact'],
                retained_visits_match=(result['box_regions'][side]['white']['retained_visits_min3']
                                       == reference[side]['white']['retained_visits_minimum3_observations']),
                retained_without_contact_match=(result['box_regions'][side]['white']['retained_without_any_contact']
                                                == reference[side]['white']['retained_visits_without_any_task_object_gripper_contact']))
            for side in SIDES}
    return result


def aggregate(rows):
    agg = dict(seeds=len(rows), seconds_total=float(sum(r['episode_seconds'] for r in rows)))
    white = {}
    for side in SIDES:
        visits = [r['box_regions'][side]['white']['empty_hand_visits'] for r in rows]
        seconds = [r['box_regions'][side]['white']['empty_hand_seconds'] for r in rows]
        white[side] = dict(
            seeds_with_empty_hand_visits=sum(v > 0 for v in visits),
            total_visits=sum(visits), total_seconds=sum(seconds),
            mean_visits_per_episode=float(np.mean(visits)),
            mean_visits_per_episode_with_visits=(float(np.mean([v for v in visits if v > 0]))
                                                 if any(v > 0 for v in visits) else None))
    agg['a_empty_hand_white_box'] = white
    approach = {}
    for threshold in (f'{x:.2f}' for x in APPROACH_SENSITIVITY):
        approach[threshold] = dict(
            seeds_with_any_window=sum(
                r['approach_no_close']['left'][threshold]['windows']
                + r['approach_no_close']['right'][threshold]['windows'] > 0 for r in rows),
            total_windows=sum(r['approach_no_close'][side][threshold]['windows']
                              for r in rows for side in SIDES),
            total_seconds=sum(r['approach_no_close'][side][threshold]['seconds']
                              for r in rows for side in SIDES),
            windows_with_any_contact=sum(r['approach_no_close'][side][threshold]['windows_with_contact']
                                         for r in rows for side in SIDES))
    agg['b_approach_no_close'] = approach
    shaking = {}
    for tag in ('exclude_box_interiors', 'any_free_space'):
        total = sum(r['shaking'][side][tag]['reversals'] for r in rows for side in SIDES)
        minutes = sum(r['shaking'][side][tag]['window_seconds'] for r in rows for side in SIDES) / 60.0
        shaking[tag] = dict(
            total_reversals=total, total_window_minutes=minutes,
            reversals_per_minute=(total / minutes) if minutes > 0 else None,
            per_side={side: dict(
                total_reversals=sum(r['shaking'][side][tag]['reversals'] for r in rows),
                total_window_minutes=sum(r['shaking'][side][tag]['window_seconds'] for r in rows) / 60.0)
                for side in SIDES})
    agg['d_shaking'] = shaking
    fractions = [r['effective_time']['effective_fraction'] for r in rows]
    agg['effective_time'] = dict(
        mean_effective_fraction=float(np.mean(fractions)),
        min_effective_fraction=float(np.min(fractions)),
        max_effective_fraction=float(np.max(fractions)),
        total_effective_seconds=sum(r['effective_time']['effective_seconds'] for r in rows),
        total_garbage_seconds=sum(r['effective_time']['garbage_seconds'] for r in rows))
    agg['c_bowl_completion_stall_seeds'] = [r['seed'] for r in rows
                                            if r['bowl_completion_stall'] is not None]
    agg['v30_cross_check_retained_visits_match_all'] = all(
        all(r['v30_cross_check'][side]['retained_visits_match'] for side in SIDES)
        for r in rows if r['v30_cross_check'])
    return agg


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
    v30_lookup = {}
    v30_doc = json.loads(Path(args.v30).read_text())
    for episode in v30_doc['cohorts']['80000extra20']['episodes']:
        v30_lookup[episode['seed']] = episode['box_regions']
    fk = FrankaFK()
    csv_rows, per_seed = [], []
    for label in ('extra20', 'v36_C1', 'v37_RTC'):
        for row in sources[label]['rows']:
            value = analyze_episode(label, row, fk, csv_rows,
                                    v30_lookup if label == 'extra20' else None)
            per_seed.append(value)
            print(json.dumps(dict(cohort=label, seed=value['seed'],
                                  effective_fraction=value['effective_time']['effective_fraction'],
                                  white_empty_left=value['box_regions']['left']['white']['empty_hand_visits'],
                                  approach05_left=value['approach_no_close']['left']['0.05']['windows'],
                                  shaking_left=value['shaking']['left']['exclude_box_interiors']['reversals'])),
                  flush=True)
    extra_rows = [r for r in per_seed if r['cohort'] == 'extra20']
    value = dict(
        schema='garbage-time-metrics-v41',
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        definitions=dict(
            tool_point='CPU FK (harness.isaac_zhuhai_expert.FrankaFK) on measured joints, tool offset included',
            box_region='v30 definition: interior AABB XY +/-0.02 m, Z interior bottom to top +0.25 m',
            retained_visit='>=3 consecutive archived observations inside the region',
            empty_hand='commanded gripper of that side >=0.8 and zero task-object gripper contact (either gripper) throughout the retained visit',
            approach='side tool outside-AABB distance < threshold for >=1 s; that side commanded gripper never <=0.2 in the window',
            shaking='tool-point direction reversal: consecutive 10 Hz displacement segments turn >120 deg with both segment lengths >1 cm, inside no-contact no-approach windows',
            effective_task_time='contact with any task object, or any tool within 0.05 m of a task-object AABB, or a tool inside any destination box region',
            contact_flag='rule_evidence objects[*].touching_gripper aggregates either gripper (v30 limitation)'),
        sources={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} | {'episode_count': len(v['rows'])}
                 for k, v in sources.items()},
        per_seed=per_seed,
        aggregate_extra20=aggregate(extra_rows),
        differences_from_v25_v30=[
            'v30 counted all retained box visits per side; v41 adds the empty-hand filter (commanded open, zero contact) and reproduces v30 retained-visit counts as an exact cross-check.',
            'v25 used a 2 cm box-interior threshold for close entries; v41 approach uses object AABBs at 5 cm primary with 2/10 cm sensitivity.',
            'v41 separates effective task time from the four garbage categories; the categories may overlap each other.'],
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with Path(args.csv).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['category', 'cohort', 'seed', 'arm', 'color',
                                                    'start_obs', 'end_obs', 'seconds', 'detail'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(dict(out=str(out.resolve()), csv=str(Path(args.csv).resolve()),
                          aggregate=value['aggregate_extra20']), ensure_ascii=False))


if __name__ == '__main__':
    main()
