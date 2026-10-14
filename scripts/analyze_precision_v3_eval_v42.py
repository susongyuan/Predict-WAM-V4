#!/usr/bin/env python3
"""Read-only, CPU analysis of a preregistered Precision-V3 five-case cohort.

Privileged rule/AABB evidence is used only offline. This is neither a simulator
nor a policy, and compatibility mode never presents old episodes as V3 results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.isaac_zhuhai_expert import FrankaFK
from scripts.analyze_precision_dagger_milestones_v19 import progress
from scripts.compare_precision_continuity10_v36 import metrics
from scripts.analyze_garbage_time_v41 import runs, APPROACH_PRIMARY, SIDES, OBJECTS

ARM = [*range(7), *range(8, 15)]
DT, EXEC, HORIZON, CLOSE = .1, 8, 32, .2
VERIFY = ROOT / 'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24'
OLD_EFFECTIVE = VERIFY / 'garbage_time_metrics_v41.json'
DEPENDENCIES = ('scripts/analyze_precision_dagger_milestones_v19.py',
                'scripts/compare_precision_continuity10_v36.py',
                'scripts/analyze_garbage_time_v41.py',
                'harness/isaac_zhuhai_expert.py')


def require(flag, message):
    if not flag:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def definitions(seeds):
    old = OLD_EFFECTIVE.read_bytes()
    previous = json.loads(old)
    fraction = previous['aggregate_extra20']['effective_time']['mean_effective_fraction']
    return dict(
        schema='precision-v3-fivecase-analysis-definitions-v42', expected_seeds=seeds,
        control_dt_seconds=DT, execution_prefix=EXEC, horizon=HORIZON, arm_indices=ARM,
        close_threshold=CLOSE,
        grasp='Original Task2Judge grasp_awarded, >=5s strict lifted/contact/not-released hold; never inferred from closure.',
        placement='Original current placement_valid; ever placement and final placement reported separately.',
        object_order=['bowl', 'spoon', 'cup', 'plate'],
        object_order_is_diagnostic_only=True, original_scoring_order_not_changed=True,
        contact_lift_hold='contact=touching_gripper(either gripper); lifted=AABB bottom>table+.01; held=contact AND lifted AND not released.',
        empty_close='A dispatched gripper<=.2 interval whose following observations contain zero ANY-object gripper contact; aggregate contacts cannot identify one hand.',
        effective_time_v41='Mean over ALL N+1 archived observations, dt=.1: any-object gripper contact OR any tool-AABB distance<.05m OR any tool inside destination interior expanded XY+/-.02m and upperZ+.25m.',
        effective_time_v41_denominator='(N+1)*.1s is an observation-weight convention, not N*.1 physics duration; both are reported.',
        baseline_effective_fraction=float(fraction),
        baseline_effective_fraction_source=dict(path=str(OLD_EFFECTIVE), sha256=sha(old)),
        baseline_15_5_percent='Rounded descriptive mean of original20 (0.1548313202832153); not a task-success threshold or calibrated progress value. Box-only wandering can increase it.',
        corrected_activity_proxy='Contact OR approach OR inside box while strict holding ANY object; still a geometry proxy, never proof of progress.',
        above_rim='Signed measured-FK tool_z minus current object AABB top; report global minimum plus approach(<5cm), contact and first verified-grasp observations. Negative alone does not establish grasp.',
        motion='Reuse v36 metrics on q[1..N] (same original after-action convention), fourteen arms only; derivatives at10Hz, replan-or-next acceleration energy; also velocity and travel.',
        overlap='Same physical instants: previous_raw[8..31] vs next_raw[0..23]; RMS over14arms and24steps, not adjacent chunk endpoints.',
        limits=['No 100ms archive metric proves 250Hz actual dynamics or all transient contacts.',
                'Tool reference point is not finger surface or pose error; AABB distance is a lower bound.',
                'Offline GT geometry is never an online policy input.',
                'Less speed/travel can lower jitter metrics without improving progress.',
                'Five cases give coarse outcome evidence, not component causal attribution.'],
        dependency_sha256={rel: sha((ROOT / rel).read_bytes()) for rel in DEPENDENCIES},
        analyzer_sha256=sha(Path(__file__).read_bytes()))


def binding_read(directory, relative, row):
    binding = row['evidence_files'].get(relative)
    require(isinstance(binding, dict), f'Missing evidence binding: {relative}')
    path = directory / relative
    if binding.get('path'):
        require(path.resolve() == Path(binding['path']).resolve(), f'Evidence path differs: {relative}')
    data = path.read_bytes()
    require(sha(data) == binding['sha256'], f'Evidence SHA differs: {relative}')
    if binding.get('size') is not None:
        require(len(data) == binding['size'], f'Evidence size differs: {relative}')
    return data


def load_episode(row):
    raw = Path(row['raw_result_path'])
    data = raw.read_bytes()
    require(sha(data) == row['raw_result_sha256'], 'Result SHA differs')
    result = json.loads(data)
    require(result['seed'] == row['seed'], 'Raw seed differs')
    directory = raw.parent
    archives = {}
    for name in ('actions.jsonl', 'rule_evidence.jsonl', 'onboard/frames.jsonl'):
        archives[name] = [json.loads(line) for line in binding_read(directory, name, row).splitlines()]
    actions, rules, frames = [archives[k] for k in ('actions.jsonl', 'rule_evidence.jsonl', 'onboard/frames.jsonl')]
    n = len(actions)
    require(n == row['control_steps'], 'Action/summary step count differs')
    require([a['control_step'] for a in actions] == list(range(n)), 'Nonconsecutive actions')
    require(len(rules) == len(frames) == n + 1, 'Observation count differs')
    for values in (rules, frames):
        require([r['control_step'] for r in values] == list(range(n + 1)), 'Nonconsecutive observations')
    times = np.asarray([r['physics_time_seconds'] for r in rules])
    require(np.allclose(np.diff(times), DT, rtol=0, atol=1e-8), 'Control time differs')
    require(not any(r['domain_parameters']['randomized'] for r in rules), 'Source distribution must remain randomized=False')
    q, commanded = [], []
    sent = np.asarray([a['dispatched_action'] for a in actions], float)
    require(sent.shape == (n, 16) and np.isfinite(sent).all(), 'Invalid sent commands')
    for frame in frames:
        name = 'onboard/' + frame['file']
        blob = binding_read(directory, name, row)
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            measured = z['measured_joint_gripper_state']
            target = z['commanded_joint_gripper_state']
            require(measured.shape == target.shape == (16,), 'Invalid native16 state')
            require(np.isfinite(measured).all() and np.isfinite(target).all(), 'Nonfinite state')
            t = frame['control_step']
            if t:
                require(np.array_equal(target, sent[t - 1].astype(target.dtype)), 'Actual endpoint differs from dispatched command')
            q.append(measured.astype(float)); commanded.append(target.astype(float))
    # Terminal unsafe observation remains archived, but never counts toward scored success.
    safe_rules = rules[:-1] if row['status'] == 'physics_safety_abort' else rules
    return result, actions, rules, safe_rules, np.stack(q), np.stack(commanded), sent


def closure_analysis(actions, sent, rules):
    replans = [a for a in actions if a['policy']['replanned']]
    out = dict(replans=len(replans), gripper_events_preserved_by_smoothing=True, sides={})
    for side, (_, g) in SIDES.items():
        indices = []
        for a in replans:
            p = a['policy']; raw = np.asarray(p['raw_denormalized_chunk'])
            smooth = np.asarray(p['smoothed_denormalized_chunk'])
            require(raw.shape == smooth.shape == (HORIZON, 16), 'Wrong chunk shape')
            require(p['execution_prefix'] == EXEC, 'Different execution prefix')
            close = np.flatnonzero(raw[:, g] <= CLOSE)
            indices.append(None if not len(close) else int(close[0]) + 1)
            for field in ('smoothed_denormalized_chunk', 'bspline_denormalized_chunk'):
                if field in p:
                    out['gripper_events_preserved_by_smoothing'] &= bool(np.array_equal(raw[:, g], np.asarray(p[field])[:, g]))
        intervals = []
        for start, end in runs(sent[:, g] <= CLOSE):
            observed = rules[start + 1:end + 2]
            contacts = {k: sum(bool(r['objects'][k]['touching_gripper']) for r in observed) for k in OBJECTS}
            intervals.append(dict(first_action_step=start, last_action_step=end,
                                  following_observation_range=[start + 1, end + 1],
                                  any_gripper_contact_observations_by_object=contacts,
                                  no_contact_with_any_task_object=not any(contacts.values())))
        values = [i for i in indices if i is not None]
        out['sides'][side] = dict(no_close_in_32=sum(i is None for i in indices),
            first_close_in_executed_prefix=sum(i is not None and i <= EXEC for i in indices),
            first_close_only_in_tail=sum(i is not None and i > EXEC for i in indices),
            first_close_1based_histogram=dict(Counter(str(i) for i in values)),
            no_close_anywhere=not values,
            only_tail_close=bool(values) and all(i > EXEC for i in values),
            dispatched_closed_steps=int((sent[:, g] <= CLOSE).sum()),
            dispatched_close_intervals=intervals,
            first_dispatched_close_action_step=intervals[0]['first_action_step'] if intervals else None)
    return out


def object_progress(rules):
    out = progress(rules)
    for key, obj in out['objects'].items():
        contact = np.asarray([r['objects'][key]['touching_gripper'] for r in rules], bool)
        bottom = np.asarray([min(p[2] for p in r['objects'][key]['world_corners_m']) for r in rules])
        lifted = bottom > np.asarray([r['table_surface_world_m'] for r in rules]) + .01
        held = contact & lifted & ~np.asarray([r['objects'][key]['released'] for r in rules], bool)
        obj.update(first_contact_step=next((i for i, x in enumerate(contact) if x), None),
            contact_without_lift_observations=int((contact & ~lifted).sum()),
            lifted_contact_observations=int((contact & lifted).sum()),
            strict_held_observations=int(held.sum()), strict_held_seconds_observation_weight=float(held.sum()) * DT,
            first_strict_held_step=next((i for i, x in enumerate(held) if x), None),
            held_intervals=runs(held), final_valid_placement=bool(rules[-1]['score']['items'][key]['placement_valid']))
        for name in ('first_contact_step', 'first_strict_held_step', 'first_verified_grasp_step', 'first_valid_placement_step'):
            t = obj[name]
            obj[name.replace('_step', '_physics_seconds')] = None if t is None else rules[t]['physics_time_seconds']
    return out


def geometry_and_activity(q, rules, objects, fk, closure=None):
    count = len(rules); q = q[:count]
    points = {side: np.stack([fk.fk(side, state[sl])[:3, 3] for state in q]) for side, (sl, _) in SIDES.items()}
    contacts = {k: np.asarray([r['objects'][k]['touching_gripper'] for r in rules], bool) for k in OBJECTS}
    contact_any = np.logical_or.reduce(list(contacts.values()))
    held_any = np.zeros(count, bool)
    dist, geometry = {}, {}
    for k in OBJECTS:
        corners = np.asarray([r['objects'][k]['world_corners_m'] for r in rules])
        lo, hi = corners.min(axis=1), corners.max(axis=1)
        held_any |= contacts[k] & (lo[:, 2] > np.asarray([r['table_surface_world_m'] for r in rules]) + .01) & ~np.asarray([r['objects'][k]['released'] for r in rules], bool)
        geometry[k] = {}; dist[k] = {}
        for side in SIDES:
            pt = points[side]; outside = np.maximum(np.maximum(lo - pt, pt - hi), 0)
            distance = np.linalg.norm(outside, axis=1); dist[k][side] = distance
            above = pt[:, 2] - hi[:, 2]; near = distance < APPROACH_PRIMARY
            first_below = next((i for i in range(count) if near[i] and above[i] < 0), None)
            first_grasp = objects[k]['first_verified_grasp_step']
            first_contact = objects[k]['first_contact_step']
            def at(t):
                return None if t is None else dict(observation_step=t, physics_seconds=rules[t]['physics_time_seconds'],
                    tool_world_m=pt[t].tolist(), signed_above_rim_m=float(above[t]),
                    tool_to_object_AABB_m=float(distance[t]))
            geometry[k][side] = dict(min_signed_above_rim_m=float(above.min()),
                min_signed_above_rim_near_object_m=float(above[near].min()) if near.any() else None,
                min_signed_above_rim_during_contact_m=float(above[contacts[k]].min()) if contacts[k].any() else None,
                nearest=at(int(np.argmin(distance))), first_below_rim_near=at(first_below),
                first_contact=at(first_contact), first_verified_grasp=at(first_grasp))
            if closure is not None:
                close_step = closure['sides'][side]['first_dispatched_close_action_step']
                observation = None if close_step is None or close_step + 1 >= count else close_step + 1
                geometry[k][side]['after_first_dispatched_close'] = at(observation)
    minimum = np.min(np.stack([dist[k][s] for k in OBJECTS for s in SIDES]), axis=0)
    inside_box = np.zeros(count, bool)
    for side in SIDES:
        for lower, upper in rules[0]['destination_interiors_world_m'].values():
            lo = np.asarray(lower) + [-.02, -.02, 0.]
            hi = np.asarray(upper) + [.02, .02, .25]
            inside_box |= np.all((points[side] >= lo) & (points[side] <= hi), axis=1)
    approach = minimum < APPROACH_PRIMARY
    effective = contact_any | approach | inside_box
    corrected = contact_any | approach | (inside_box & held_any)
    activity = dict(observation_count=count, physical_seconds=rules[-1]['physics_time_seconds'] - rules[0]['physics_time_seconds'],
        observation_weight_seconds=count * DT, effective_fraction_v41=float(effective.mean()),
        effective_seconds_v41=float(effective.sum()) * DT,
        non_effective_seconds_v41=float((~effective).sum()) * DT,
        corrected_activity_proxy_fraction=float(corrected.mean()),
        box_only_without_held_object_observations=int((inside_box & ~held_any & ~contact_any & ~approach).sum()),
        strict_held_any_object_observations=int(held_any.sum()))
    # Stage progression is evidence, not an online rule; later-object-before-bowl remains visible.
    transition = {}
    for previous, following in zip(OBJECTS, OBJECTS[1:]):
        t = objects[previous]['first_valid_placement_step']
        if t is None:
            continue
        next_grasp = objects[following]['first_verified_grasp_step']
        after = slice(t + 1, count)
        transition[f'{previous}_to_{following}'] = dict(first_previous_placement_step=t,
            remaining_physics_seconds=rules[-1]['physics_time_seconds'] - rules[t]['physics_time_seconds'],
            following_contact_observations_after=int(contacts[following][after].sum()),
            min_tool_to_following_AABB_m_after={s: float(dist[following][s][after].min()) if t + 1 < count else None for s in SIDES},
            following_grasp_first_step=next_grasp,
            following_grasp_occurs_after_previous_placement=next_grasp is not None and next_grasp > t,
            following_grasp_already_awarded_at_previous_placement=next_grasp is not None and next_grasp <= t,
            no_following_verified_grasp_after_previous_placement=not any(
                r['score']['items'][following]['grasp_awarded'] for r in rules[t + 1:]))
    return geometry, activity, transition


def command_motion(actions, q, sent):
    replanned = [a['policy']['replanned'] for a in actions]
    require(all('model_raw_action' in a['policy'] for a in actions), 'Missing pre-spline model action')
    raw = np.asarray([a['policy']['model_raw_action'] for a in actions], float)
    smooth = np.asarray([a['smoothed_action'] for a in actions], float)
    streams = dict(model_raw=raw, smoothed_reference=smooth, dispatched=sent, measured_after_action=q[1:])
    out = dict(streams={k: metrics(v, replanned) if len(v) >= 4 else {'available': False, 'reason': 'fewer than four archive samples'} for k, v in streams.items()})
    if all('bspline_action' in a['policy'] for a in actions):
        spline = np.asarray([a['policy']['bspline_action'] for a in actions], float)
        out['streams']['bspline'] = metrics(spline, replanned) if len(spline) >= 4 else {'available': False}
    for a, b in (('model_raw', 'smoothed_reference'), ('smoothed_reference', 'dispatched'), ('dispatched', 'measured_after_action')):
        delta = streams[b][:, ARM] - streams[a][:, ARM]
        out[f'{a}_to_{b}'] = dict(arm_rms_rad=float(np.sqrt(np.mean(delta ** 2))), arm_max_abs_rad=float(abs(delta).max()))
    replans = [a for a in actions if a['policy']['replanned']]
    overlaps = []
    for previous, following in zip(replans, replans[1:]):
        require(following['control_step'] - previous['control_step'] == EXEC, 'Unexpected replan spacing')
        before = np.asarray(previous['policy']['raw_denormalized_chunk'])
        after = np.asarray(following['policy']['raw_denormalized_chunk'])
        overlaps.append(float(np.sqrt(np.mean((after[:HORIZON - EXEC, ARM] - before[EXEC:, ARM]) ** 2))))
    out['same_time_raw_overlap'] = dict(pairs=len(overlaps), mean_rms_rad=float(np.mean(overlaps)) if overlaps else None,
        median_rms_rad=float(np.median(overlaps)) if overlaps else None)
    return out


def analyze_episode(row, fk):
    result, actions, rules, safe_rules, q, commanded, sent = load_episode(row)
    physical = object_progress(safe_rules)
    closure = closure_analysis(actions, sent, safe_rules)
    geometry, activity, transitions = geometry_and_activity(q, safe_rules, physical['objects'], fk, closure)
    safe_terminal = safe_rules[-1]['score']
    score = row.get('observed_base_score', row.get('base_score'))
    require(score == safe_terminal['base_score'], 'Admitted score differs from safe terminal rules')
    require(row['full_task_success'] == safe_terminal['full_task_success'], 'Full success differs from original rules')
    tags = []
    for side, info in closure['sides'].items():
        if not info['dispatched_closed_steps']:
            tags.append(f'{side}_never_closed_le_0_2')
        if info['no_close_anywhere']:
            tags.append(f'{side}_no_close_in_any_32step_plan')
        elif info['only_tail_close']:
            tags.append(f'{side}_close_only_unexecuted_tail')
        if any(i['no_contact_with_any_task_object'] for i in info['dispatched_close_intervals']):
            tags.append(f'{side}_empty_close_interval_any_gripper_contact_proxy')
    for key, obj in physical['objects'].items():
        if obj['gripper_contact_frames'] and not obj['strict_held_observations']:
            tags.append(f'{key}_contact_without_lifted_hold')
        if obj['strict_held_observations'] and not obj['ever_verified_grasp']:
            tags.append(f'{key}_lifted_contact_hold_without_verified_5s_grasp')
        if obj['ever_verified_grasp'] and not obj['ever_valid_placement']:
            tags.append(f'{key}_verified_grasp_without_placement')
        if obj['ever_valid_placement'] and not obj['final_valid_placement']:
            tags.append(f'{key}_placement_later_lost')
    for stage, value in transitions.items():
        if value['no_following_verified_grasp_after_previous_placement']:
            tags.append(f'{stage}_no_next_verified_grasp')
    return dict(seed=row['seed'], status=row['status'], control_steps=len(actions), score=score,
        full_task_success=row['full_task_success'], failure_stage=physical['failure_stage'], objects=physical['objects'],
        closure=closure, geometry=geometry, activity=activity, transitions=transitions,
        overlapping_evidence_tags=tags, motion=command_motion(actions, q, sent),
        source_result_sha256=row['raw_result_sha256'], all_onboard_archives_rehashed=True,
        actual_endpoint_exact_archive_dtype_verified=True,
        unsafe_terminal_excluded_from_progress=row['status'] == 'physics_safety_abort',
        note='Descriptive independent analysis complements protocol/driver/model replay audit; does not replace it.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expected-seeds', type=int, nargs='+', required=True)
    p.add_argument('--definitions-only', action='store_true')
    p.add_argument('--compatibility-only', action='store_true')
    args = p.parse_args(argv)
    require(not args.out.exists(), 'Preserve prior analysis receipt; choose fresh output')
    require(len(set(args.expected_seeds)) == len(args.expected_seeds), 'Duplicate preregistered seeds')
    declared = definitions(args.expected_seeds)
    if args.definitions_only:
        require(not args.compatibility_only, 'Definitions and compatibility are separate modes')
        value = dict(declared, status='preregistered_definitions_only', time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'), no_v3_results=True)
    else:
        require(args.summary is not None, '--summary required for analysis')
        data = args.summary.read_bytes(); summary = json.loads(data)
        require(summary['status'] == 'completed', 'Cohort must finish before final analysis')
        by_seed = {r['seed']: r for r in summary['episodes']}
        require(len(by_seed) == len(summary['episodes']), 'Duplicate episode rows')
        if args.compatibility_only:
            require(set(args.expected_seeds) <= set(by_seed), 'Compatibility seeds not in old cohort')
        else:
            require(len(args.expected_seeds) == 5, 'Exactly five preregistered cases required')
            require(summary['planned_episodes'] == 5, 'Not a five-case cohort')
            require(summary['seeds'] == args.expected_seeds, 'Summary seed order differs from preregistration')
            require([r['seed'] for r in summary['episodes']] == args.expected_seeds, 'Episode seed order differs')
            identity = ' '.join(str(summary.get(k, '')) for k in ('schema', 'evaluation_version', 'system_mode'))
            require('v3' in identity.lower(), 'Production mode requires explicit V3 evaluation identity')
        rows = [analyze_episode(by_seed[s], FrankaFK()) for s in args.expected_seeds]
        value = dict(schema='precision-v3-fivecase-independent-analysis-v42',
            status='passed_old_archive_compatibility_not_v3_results' if args.compatibility_only else 'completed_independent_descriptive_analysis',
            time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            no_v3_results=args.compatibility_only, definitions=declared,
            summary_binding=dict(path=str(args.summary.resolve()), sha256=sha(data)),
            checkpoint=summary.get('checkpoint'), checkpoint_sha256=summary.get('checkpoint_sha256'),
            tests=len(rows), full_successes=sum(r['full_task_success'] for r in rows),
            scores=[r['score'] for r in rows],
            stage_counts=dict(Counter(r['failure_stage'] for r in rows)),
            overlapping_evidence_tag_counts=dict(Counter(t for r in rows for t in r['overlapping_evidence_tags'])),
            per_object={k: dict(verified_grasp_cases=sum(r['objects'][k]['ever_verified_grasp'] for r in rows),
                ever_valid_placement_cases=sum(r['objects'][k]['ever_valid_placement'] for r in rows),
                final_valid_placement_cases=sum(r['objects'][k]['final_valid_placement'] for r in rows)) for k in OBJECTS},
            effective_fraction_v41_mean=float(np.mean([r['activity']['effective_fraction_v41'] for r in rows])),
            episodes=rows)
        if not args.compatibility_only:
            require(summary['full_successes'] == value['full_successes'], 'Summary success total differs')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k: value[k] for k in ('status', 'tests', 'full_successes', 'scores') if k in value}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
