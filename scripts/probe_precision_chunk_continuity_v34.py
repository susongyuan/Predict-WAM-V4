#!/usr/bin/env python3
"""Offline continuity candidates on archived plans; no policy or plant rollout.

These are postprocessing controls, NOT RTC implementation or proof of success.
All alternate trajectories use frozen observations from the original rollout.
"""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.zhuhai_action_smoothing import ARM, GRIPPER, smooth_chunk
from harness.zhuhai_official_braking import BrakingRateLimiter


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def bridge(chunk, position, velocity, duration=.4, dt=.1):
    """C1 reference at t=0 and at the end of a short Hermite correction.

The first returned target is t=dt, never a duplicate t=0 target. Anchors
describe the previous dispatched command, not an unobserved plant state.
Official braking remains a separate downstream filter; C1 is pre-filter only.
"""
    require(chunk.shape == (32, 16) and np.isfinite(chunk).all(), 'finite native16 required')
    require(duration >= 2*dt and duration <= 8*dt, 'bridge must fit execution prefix')
    time = (np.arange(32)+1)*dt
    curve = CubicSpline(time, chunk[:, ARM], axis=0)
    delta_q = np.asarray(position)-curve(0.)
    delta_v = np.asarray(velocity)-curve(0., 1)
    s = np.minimum(time/duration, 1.)
    h00 = 2*s**3-3*s**2+1
    h10 = s**3-2*s**2+s
    correction = h00[:, None]*delta_q + duration*h10[:, None]*delta_v
    out = chunk.copy()
    out[:, ARM] += correction
    # Exact analytic endpoint residuals; no finite-difference C1 claim.
    start_q_error = np.max(np.abs(curve(0.)+delta_q-position))
    start_v_error = np.max(np.abs(curve(0., 1)+delta_v-velocity))
    require(start_q_error < 1e-12 and start_v_error < 1e-12, 'anchor mismatch')
    require(np.array_equal(out[:, GRIPPER], chunk[:, GRIPPER]), 'gripper event changed')
    return out


def aligned_blend(previous, current, duration_steps=4):
    """Previous plan [8:32] and current [0:24] denote identical action times."""
    out = current.copy()
    u = np.minimum((np.arange(24)+1)/duration_steps, 1.)
    old_weight = 1-(6*u**5-15*u**4+10*u**3)
    out[:24, ARM] = old_weight[:, None]*previous[8:, ARM]+(1-old_weight[:, None])*current[:24, ARM]
    require(np.array_equal(out[:, GRIPPER], current[:, GRIPPER]), 'gripper event changed')
    return out


def motion(values):
    delta = np.diff(values[:, ARM], axis=0)
    acceleration = np.diff(delta, axis=0)/.01
    jerk = np.diff(acceleration, axis=0)/.1
    return dict(acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration**2))),
                finite_difference_jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk**2))))


def episode(row, settings):
    directory = Path(row['raw_result_path']).parent
    data = (directory/'actions.jsonl').read_bytes()
    require(digest(data) == row['evidence_files']['actions.jsonl']['sha256'], 'actions hash differs')
    actions = [json.loads(line) for line in data.splitlines()]
    frames = [json.loads(line) for line in (directory/'onboard/frames.jsonl').read_bytes().splitlines()]
    initial_name = 'onboard/'+frames[0]['file']
    initial_data = (directory/initial_name).read_bytes()
    require(digest(initial_data) == row['evidence_files'][initial_name]['sha256'], 'initial target hash differs')
    with np.load(io.BytesIO(initial_data), allow_pickle=False) as z:
        initial = z['commanded_joint_gripper_state'].astype(float)
    recorded = np.array([a['dispatched_action'] for a in actions])
    result = {}
    for method in ('baseline_replay', 'aligned_overlap', 'c1_bridge', 'aligned_overlap_c1_bridge'):
        limiter = BrakingRateLimiter(settings['rate_limit'])
        limiter.reset(initial)
        previous = None
        sent, requested = [], []
        for step, action in enumerate(actions):
            policy = action['policy']
            require(bool(policy['replanned']) == (step % 8 == 0), 'unexpected queue timeline')
            if policy['replanned']:
                plan = np.asarray(policy['smoothed_denormalized_chunk'], dtype=float)
                require(np.array_equal(smooth_chunk(np.array(policy['raw_denormalized_chunk']), settings), plan),
                        'base spline replay mismatch')
                selected = plan.copy()
                if previous is not None and 'aligned_overlap' in method:
                    selected = aligned_blend(previous, selected)
                if 'c1_bridge' in method:
                    selected = bridge(selected, limiter.previous, limiter.velocity)
                # Use the preceding original plan; no recursive old-plan averaging.
                previous = plan
            target = selected[step % 8]
            command, stats = limiter.step(target)
            requested.append(target)
            sent.append(command)
        sent, requested = np.array(sent), np.array(requested)
        require(np.array_equal(sent[:, GRIPPER], recorded[:, GRIPPER]), 'dispatched gripper event changed')
        if method == 'baseline_replay':
            require(np.array_equal(sent, recorded), 'official baseline dispatch replay mismatch')
        result[method] = dict(command_motion=motion(sent), prefilter_motion=motion(requested),
            commanded_arm_deviation_from_recorded_rms_rad=float(np.sqrt(np.mean((sent[:, ARM]-recorded[:, ARM])**2))),
            commanded_arm_deviation_from_recorded_max_rad=float(np.max(np.abs(sent[:, ARM]-recorded[:, ARM]))),
            official_braking_preserved=True, gripper_samples_exactly_preserved=True,
            simulated_measured_motion_available=False, grasp_success_available=False)
    return dict(seed=row['seed'],control_steps=len(actions),methods=result,
                bindings=dict(actions_sha256=digest(data),initial_target_npz_sha256=digest(initial_data),
                              original_result_sha256=row['raw_result_sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path, out = Path(args.summary), Path(args.out)
    require(not out.exists(), 'preserve prior evidence')
    data = path.read_bytes()
    summary = json.loads(data)
    require(summary['status'] == 'completed', 'completed cohort required')
    settings = json.loads((path.parent/'plan.json').read_text())['action_postprocessing']
    rows = []
    for row in summary['episodes']:
        rows.append(episode(row, settings))
        print(f'{row["seed"]}: exact baseline replay; all candidates preserve braking/gripper PASS', flush=True)
    methods = {method:dict(mean_command_acceleration_rms_rad_s2=float(np.mean([
        r['methods'][method]['command_motion']['acceleration_rms_rad_s2'] for r in rows])),
        mean_command_finite_difference_jerk_rms_rad_s3=float(np.mean([
        r['methods'][method]['command_motion']['finite_difference_jerk_rms_rad_s3'] for r in rows])),
        maximum_command_deviation_rad=max(r['methods'][method]['commanded_arm_deviation_from_recorded_max_rad'] for r in rows))
        for method in rows[0]['methods']}
    value = dict(schema='offline-frozen-observation-continuity-controls-v34',
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        status='passed_offline_reference_only',summary_sha256=digest(data),analyzer_sha256=digest(Path(__file__).read_bytes()),
        fixed_bridge_seconds=.4,fixed_overlap_transition_steps=4,episodes=rows,aggregates=methods,
        limitations=['No policy calls, no simulator rollout, no new evaluation attempts or training.',
            'Changed targets would change observations; frozen-plan comparisons cannot predict autonomous success.',
            'C1 holds for the prefilter analytic reference; 10Hz dispatch and braking do not establish continuous plant jerk.',
            'Candidates are not imported by evaluation and have not been promoted.',
            'Post-hoc controls are not RTC or learned prefix conditioning; no hyperparameter selection on scores.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(methods,ensure_ascii=False))


if __name__ == '__main__':
    main()
