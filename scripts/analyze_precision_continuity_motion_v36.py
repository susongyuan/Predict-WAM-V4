#!/usr/bin/env python3
"""Read-only CPU analysis of archived arm motion, plan boundaries and closure FK.

This does not query a model, run a simulator, change smoothing or alter scores.
10 Hz finite differences are descriptive and are not physics-substep dynamics.
"""
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
sys.path.insert(0, str(ROOT))
from harness.isaac_zhuhai_expert import FrankaFK
from harness.zhuhai_action_smoothing import smooth_chunk
from harness.zhuhai_chunk_continuity_v36 import reference
from scripts.analyze_precision_execution_commands_v23 import command_episode, runs

ARM = np.array([*range(7), *range(8, 15)])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(flag, message):
    if not flag:
        raise RuntimeError(message)


def distribution(a):
    a = np.asarray(a, dtype=float)
    if not a.size:
        return None
    return dict(mean=float(np.mean(a)), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), maximum=float(np.max(a)))


def motion(q, replanned):
    dq = np.diff(q[:, ARM], axis=0)
    velocity = dq / .1
    acceleration = np.diff(velocity, axis=0) / .1
    # Only reversals with >= .02 rad/s on BOTH sides are counted.
    rev = ((velocity[1:] * velocity[:-1] < 0)
           & (np.abs(velocity[1:]) >= .02) & (np.abs(velocity[:-1]) >= .02))
    boundaries = np.asarray(replanned[1:], bool)
    return dict(velocity_rms_rad_s=float(np.sqrt(np.mean(velocity**2))),
                acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration**2))),
                max_abs_step_rad=float(np.max(np.abs(dq))),
                boundary_step_norm_rad=distribution(np.linalg.norm(dq[boundaries], axis=1)),
                within_chunk_step_norm_rad=distribution(np.linalg.norm(dq[~boundaries], axis=1)),
                meaningful_direction_reversals_per_joint=rev.sum(0).tolist(),
                reversal_threshold_rad_s=.02,
                total_joint_travel_rad=np.abs(dq).sum(0).tolist(),
                joint_span_rad=np.ptp(q[:, ARM], axis=0).tolist())


def distance(point, bounds):
    lo, hi = np.asarray(bounds)
    outside = np.maximum(np.maximum(lo - point, point - hi), 0.)
    return float(np.linalg.norm(outside))


def analyze_episode(row, settings, fk, plot_dir=None, label=None):
    cmd = command_episode(row)
    directory = Path(row['raw_result_path']).parent
    actions = [json.loads(line) for line in (directory/'actions.jsonl').read_text().splitlines()]
    rules = [json.loads(line) for line in (directory/'rule_evidence.jsonl').read_text().splitlines()]
    rule_by_step = {r['control_step']: r for r in rules}
    frames = [json.loads(line) for line in (directory/'onboard/frames.jsonl').read_text().splitlines()]
    require([f['control_step'] for f in frames] == list(range(len(actions)+1)), 'Action/observation alignment differs')
    measured, commanded = [], []
    for frame in frames:
        path = directory/'onboard'/frame['file']
        b = path.read_bytes()
        require(digest(b) == row['evidence_files']['onboard/'+frame['file']]['sha256'], 'Measured archive hash differs')
        with np.load(io.BytesIO(b), allow_pickle=False) as z:
            measured.append(z['measured_joint_gripper_state'].astype(float))
            commanded.append(z['commanded_joint_gripper_state'].astype(float))
    q = np.stack(measured)
    previous_targets = np.stack(commanded)
    raw = np.array([a['policy']['model_raw_action'] for a in actions])
    smoothed = np.array([a['policy']['smoothed_action'] for a in actions])
    sent = np.array([a['dispatched_action'] for a in actions])
    require(np.allclose(previous_targets[1:], sent, rtol=0, atol=3e-7), 'Observation t+1 target does not match action t')
    replanned = [a['policy']['replanned'] for a in actions]
    replans = [a for a in actions if a['policy']['replanned']]
    overlap, first_delta, examples = [], [], []
    for i, a in enumerate(replans):
        chunk = np.array(a['policy']['raw_denormalized_chunk'])
        smooth = np.array(a['policy']['smoothed_denormalized_chunk'])
        bspline=smooth_chunk(chunk,settings)
        require(np.array_equal(bspline,a['policy']['bspline_denormalized_chunk']),'Original spline differs')
        _,expected=reference(bspline,np.asarray(a['policy']['continuity_anchor_q_rad']),
                             np.asarray(a['policy']['continuity_anchor_v_rad_s']))
        require(np.array_equal(expected,smooth),'Archived C1 reference differs')
        if i:
            prev = np.array(replans[i-1]['policy']['smoothed_denormalized_chunk'])
            delta = smooth[0, ARM]-prev[8, ARM]
            overlap.append(np.sqrt(np.mean((smooth[:24, ARM]-prev[8:, ARM])**2)))
            first_delta.append(np.linalg.norm(delta))
            examples.append(dict(control_step=a['control_step'],
                                 next_action_plan_disagreement_max_rad=float(np.max(np.abs(delta))),
                                 next_action_plan_disagreement_norm_rad=float(np.linalg.norm(delta)),
                                 smoothed_to_sent_max_rad=float(np.max(np.abs(sent[a['control_step'], ARM]-smooth[0, ARM])))))
    point_cache = {}
    def point(step, side='left'):
        key = (step, side)
        if key not in point_cache:
            index = slice(0, 7) if side == 'left' else slice(8, 15)
            point_cache[key] = fk.fk(side, q[step, index])[:3, 3]
        return point_cache[key]
    def sample(step):
        rule = rule_by_step[step]
        p = point(step)
        bowl = np.asarray(rule['objects']['bowl']['world_corners_m'])
        box = rule['destination_interiors_world_m']['white']
        return dict(observation_control_step=step,
                    physics_time_seconds=rule['physics_time_seconds'],
                    left_tool_world_m=p.tolist(), left_joint_rad=q[step, :7].tolist(),
                    left_measured_gripper=float(q[step, 7]), left_commanded_gripper=float(previous_targets[step, 7]),
                    bowl_AABB_world_m=[bowl.min(0).tolist(), bowl.max(0).tolist()],
                    tool_to_bowl_AABB_lower_bound_m=distance(p, [bowl.min(0), bowl.max(0)]),
                    tool_to_white_interior_AABB_lower_bound_m=distance(p, box),
                    bowl_touching_gripper=rule['objects']['bowl']['touching_gripper'],
                    bowl_released=rule['objects']['bowl']['released'],
                    bowl_score=rule['score']['items']['bowl'],
                    physics_safety_passed=rule['physics_safety']['passed'])
    close_runs = runs(sent[:, 7] <= .2)
    close_events = []
    for start, end in close_runs:
        obs_step = start+1
        event = dict(first_close_action_step=start, last_close_action_step=end,
                     after_action_observation=sample(obs_step),
                     bowl_contact_frames_during_close=sum(rule_by_step[t]['objects']['bowl']['touching_gripper']
                         for t in range(start+1, end+2) if t in rule_by_step))
        close_events.append(event)
    first_measured = next((i for i in range(1, len(q)) if q[i,7] <= .2), None)
    first_command = close_runs[0][0] if close_runs else None
    first_grasp = cmd['physical_progress']['objects']['bowl']['first_verified_grasp_step']
    first_place = cmd['physical_progress']['objects']['bowl']['first_valid_placement_step']
    important = {0, len(actions)}
    for t in (first_command, None if first_command is None else first_command+1, first_measured, first_grasp, first_place):
        if t is not None and t in rule_by_step:
            important.add(t)
    # Spatial evidence is offline GT diagnostics, never fed to policy.
    box_close = [ev for ev in close_events if ev['after_action_observation']['tool_to_white_interior_AABB_lower_bound_m'] <= .02]
    before_grasp_box = [ev for ev in box_close if first_grasp is None or ev['first_close_action_step'] < first_grasp]
    obj_contacts = {k:cmd['physical_progress']['objects'][k]['gripper_contact_frames'] for k in ('bowl','spoon','cup','plate')}
    if first_grasp is not None:
        subtype = 'bowl_placed_then_next_stage' if first_place is not None else 'verified_bowl_grasp_without_valid_placement'
    elif first_command is None:
        subtype = 'no_executed_left_close_le_0_2'
    elif obj_contacts['bowl']:
        subtype = 'bowl_contact_without_verified_hold'
    else:
        subtype = 'executed_left_close_without_bowl_contact'
    motion_stats = {name:motion(v, replanned) for name,v in [('model_raw',raw),('c1_reference_target',smoothed),('original_bspline_target',np.array([a['policy']['bspline_action'] for a in actions])),('dispatched_target',sent),('measured_after_action',q[1:])]}
    result = dict(seed=row['seed'], command_diagnostic=cmd, measured_frames_verified=len(q),
                  failure_subtype=subtype, motion=motion_stats,
                  actual_tracking_error_after_action_arm_rad=distribution(np.abs(q[1:, ARM]-sent[:, ARM])),
                  plan_overlap_rms_rad=distribution(overlap),
                  replan_next_action_disagreement_norm_rad=distribution(first_delta),
                  largest_replan_disagreements=sorted(examples,key=lambda x:x['next_action_plan_disagreement_norm_rad'],reverse=True)[:8],
                  first_close_command_action_step=first_command,
                  first_measured_closed_observation_step=first_measured,
                  geometry_samples=[sample(t) for t in sorted(important)],
                  close_events=close_events, close_entries_near_white_box_count=len(box_close),
                  close_entries_near_white_box_before_verified_bowl_grasp=len(before_grasp_box),
                  close_entries_near_white_box_with_no_bowl_contact=sum(not ev['bowl_contact_frames_during_close'] for ev in box_close))
    if plot_dir is not None and row['seed'] in (932101,932102,932105):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plot_dir.mkdir(parents=True, exist_ok=True)
        end = min(len(sent), 400 if label=='30000' else 1100)
        time = np.arange(end)*.1
        fig, ax = plt.subplots(3,1,figsize=(13,8),sharex=True,constrained_layout=True)
        for name, data in [('model',raw),('C1 reference',smoothed),('sent',sent),('measured',q[1:])]:
            ax[0].plot(time,np.rad2deg(data[:end,6]),label=name,lw=1,alpha=.85)
        ax[0].set_ylabel('Left joint 7 (deg)');ax[0].legend(ncol=4)
        ax[1].plot(time,np.rad2deg(np.r_[0,np.diff(q[1:end+1,6])]/.1),label='measured J7 velocity',lw=1)
        ax[1].set_ylabel('Velocity (deg/s)');ax[1].legend()
        ax[2].plot(time,sent[:end,7],label='sent left gripper',lw=1)
        ax[2].plot(time,q[1:end+1,7],label='measured left gripper',lw=1)
        ax[2].set_ylabel('Gripper (0=closed)');ax[2].set_xlabel('Action index / 10 (s)');ax[2].legend()
        for t in np.flatnonzero(replanned)[:end//8]:
            if t < end:
                for axis in ax:
                    axis.axvline(t*.1,color='gray',alpha=.12,lw=.5)
        fig.suptitle(f'{label} seed {row["seed"]}: archived motion; gray lines = replans')
        name=plot_dir/f'{label}_seed{row["seed"]}_arm_motion.png'
        require(not name.exists(),'Preserve existing motion figure')
        fig.savefig(name,dpi=160);plt.close(fig)
        result['motion_figure']=str(name.resolve())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',action='append',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--plot-dir')
    args=parser.parse_args()
    out=Path(args.out);require(not out.exists(),'Preserve existing diagnostic')
    fk=FrankaFK();cohorts={};bindings={}
    for item in args.cohort:
        label,name=item.split('=',1);path=Path(name);snapshot=path.read_bytes();s=json.loads(snapshot)
        require(s['status']=='completed' or args.allow_partial,'Cohort incomplete')
        plan=json.loads((path.parent/'plan.json').read_text())
        settings=plan['action_postprocessing']
        rows=[]
        for row in s['episodes']:
            value=analyze_episode(row,settings,fk,Path(args.plot_dir) if args.plot_dir else None,label)
            rows.append(value)
            print(json.dumps(dict(cohort=label,seed=row['seed'],subtype=value['failure_subtype'],
                  first_close=value['first_close_command_action_step'],
                  box_close_entries=value['close_entries_near_white_box_count'],
                  measured_accel_rms=value['motion']['measured_after_action']['acceleration_rms_rad_s2'],
                  overlap=value['plan_overlap_rms_rad']),ensure_ascii=False),flush=True)
        cohorts[label]=dict(status_at_snapshot=s['status'],episodes=rows,
                            failure_subtypes=dict(Counter(r['failure_subtype'] for r in rows)))
        bindings[label]=dict(path=str(path.resolve()),snapshot_sha256=digest(snapshot))
    result=dict(schema='precision-continuity-arm-motion-and-first-close-geometry-v36',
                time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                cohorts=cohorts,summary_bindings=bindings,analyzer_sha256=digest(Path(__file__).read_bytes()),
                interpretation=['Action t is paired with observation t+1; each measured NPZ hash checked.',
                    'AABB distances use the tool reference point, not finger-surface clearance or collision.',
                    '10Hz measured finite differences do not resolve intra-step physics oscillations.',
                    'Replan disagreement and direction reversals describe inconsistent trajectories, not causal effects of RNG, architecture or safety limits.',
                    'Box proximity uses destination interior AABB <=2cm; not proof of grasping the box.',
                    'No isolated experiments: no underfitting/overfitting or data-quantity diagnosis established.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(str(out.resolve()),flush=True)


if __name__=='__main__':
    main()
