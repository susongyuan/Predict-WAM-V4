#!/usr/bin/env python3
"""Compare retained original and new execution cohorts without GPU or reruns."""
from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import eval_zhuhai_precision_fixed_attempts as evidence
from harness import zhuhai_robot_spec as spec


def episode(row):
    directory = Path(row['raw_result_path']).parent
    raw = evidence.read_json(directory/'result.json')
    actions = [json.loads(x) for x in (directory/'actions.jsonl').read_text().splitlines()]
    items = {}; peak = 0; unsafe = 0
    for line in (directory/'rule_evidence.jsonl').read_text().splitlines():
        rule = json.loads(line); peak = max(peak, rule['score']['base_score'])
        unsafe += int(not rule['physics_safety']['passed'])
        for name, score in rule['score']['items'].items():
            detail = items.setdefault(name, dict(first_grasp_award_step=None, first_valid_placement_step=None,
                maximum_hold_seconds=0., maximum_stable_seconds=0., gripper_contact_frames=0))
            for field, signal in [('first_grasp_award_step','grasp_awarded'), ('first_valid_placement_step','placement_valid')]:
                if score[signal] and detail[field] is None: detail[field] = rule['control_step']
            detail['maximum_hold_seconds'] = max(detail['maximum_hold_seconds'], score['hold_seconds'])
            detail['maximum_stable_seconds'] = max(detail['maximum_stable_seconds'], score['stable_seconds'])
            detail['gripper_contact_frames'] += int(rule['objects'][name]['touching_gripper'])
    bounds = np.asarray(spec.DUAL_JOINT_LIMITS); arm = list(spec.ARM_INDICES)
    measured_excess = dispatched_excess = raw_excess = 0.; measured_frames = 0
    for action in actions:
        for key in ('raw_action','dispatched_action'):
            q = np.asarray(action[key])[arm]
            excess = float(np.maximum(bounds[:,0]-q, q-bounds[:,1]).max(initial=0.))
            if key == 'raw_action': raw_excess = max(raw_excess, excess)
            else: dispatched_excess = max(dispatched_excess, excess)
    for path in sorted((directory/'onboard').glob('observation_*.npz')):
        with np.load(path, allow_pickle=False) as z: q = z['measured_joint_gripper_state'][arm]
        excess = float(np.maximum(bounds[:,0]-q, q-bounds[:,1]).max(initial=0.))
        measured_excess = max(measured_excess, excess); measured_frames += int(excess > 1e-6)
    stats = [a['policy']['rate_limit_stats'] for a in actions]
    return dict(seed=row['seed'], status=row['status'], full_task_success=row['full_task_success'],
        control_steps=raw['control_steps'], terminal_score=row['observed_base_score'], peak_score=peak,
        items=items, physics_unsafe_frames=unsafe,
        adjusted_steps=sum(s['limited_arm_dimensions']>0 for s in stats),
        braking_steps=sum(s.get('braking_limited_dimensions',0)>0 for s in stats),
        maximum_adjustment_rad=max(s['arm_limit_delta_max_rad'] for s in stats),
        maximum_raw_prediction_angle_excess_rad=raw_excess,
        maximum_dispatched_angle_excess_rad=dispatched_excess,
        maximum_measured_angle_excess_rad=measured_excess, measured_frames_excess_above_1e_minus_6rad=measured_frames,
        angle_scope='10Hz archives; not continuous plant certification', record_video=raw['record_video'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-summary',required=True); parser.add_argument('--baseline-summary',required=True)
    parser.add_argument('--out',required=True); args = parser.parse_args()
    out = Path(args.out).resolve(); evidence.require(not out.exists(), 'Immutable new audit output required')
    new = evidence.read_json(args.new_summary); old = evidence.read_json(args.baseline_summary)
    evidence.require(new['status'] == old['status'] == 'completed', 'Both full cohorts must be complete')
    evidence.require(new['seeds'] == old['seeds'] == list(evidence.SEEDS), 'Same fixed seeds required')
    evidence.require(new['checkpoint']['sha256'] == old['checkpoint_sha256'], 'Same model SHA required')
    rows = []
    for before, after in zip(old['episodes'], new['episodes']):
        a, b = Path(before['raw_result_path']).parent, Path(after['raw_result_path']).parent
        initial = {}
        with np.load(a/'onboard/observation_000000.npz',allow_pickle=False) as x, np.load(b/'onboard/observation_000000.npz',allow_pickle=False) as y:
            for key in x.files:
                initial[key] = dict(equal=bool(np.array_equal(x[key],y[key])),
                    mean_absolute_difference=float(np.abs(x[key].astype(float)-y[key].astype(float)).mean()))
        rows.append(dict(seed=after['seed'],baseline=episode(before),new=episode(after),initial_observation_comparison=initial))
    result = dict(status='completed', baseline_sha256=evidence.sha256_file(args.baseline_summary),
        new_summary_sha256=evidence.sha256_file(args.new_summary), checkpoint_sha256=new['checkpoint']['sha256'],
        baseline_full_success_rate=old['success_rate'],new_full_success_rate=new['success_rate'],
        baseline_safety_aborts=old['safety_aborted'],new_safety_aborts=new['safety_aborted'],
        episodes=rows,gpu_used=False,physical_reruns=0,
        interpretation='Same model/seeds/scoring, combined official-limits+braking change. Initial render differences retained; not an isolated component ablation.')
    out.parent.mkdir(parents=True,exist_ok=True);evidence.write_json(out,result)
    print(json.dumps({k:result[k]for k in ('status','baseline_full_success_rate','new_full_success_rate','baseline_safety_aborts','new_safety_aborts')},ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
