#!/usr/bin/env python3
"""Replay a failed case's safe prefix, then test the repaired local teacher.

Verification only: no production quota credit, no policy inference, no videos,
no teleport to a failed state, and no change to the physical scoring/scene.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import gpu_lock
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts import eval_zhuhai_precision_official_braking as frozen
from harness.zhuhai_dagger_parallel import check_runtime_resources, snapshot, can_expand
from harness.zhuhai_official_braking import BrakingRateLimiter, protocol
from harness.zhuhai_robot_spec import ARM_INDICES


def execute(plan):
    for path, expected in plan['sources'].items():
        e.require(e.sha256_file(path) == expected, 'Probe source/input changed: '+path)
    e.require(can_expand(snapshot(), 0, maximum=1), 'Cold probe resource admission failed')
    out = Path(plan['out']); out.mkdir(parents=True, exist_ok=False)
    e.write_json(out/'plan.json', plan)
    frozen.configure(); base = frozen.base
    from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
    from harness.isaac_zhuhai_segmented_expert_v8 import SegmentedExpert
    from harness.zhuhai_demo_recording import StreamDemoRecorder, jsonable
    from harness.robotwin_eval_evidence import jsonl
    row = dict(status='starting', seed=plan['scenario']['seed'], production=False,
        quota_credit=0, record_video=False, third_view=False, control_steps=0)
    held = False; env = recorder = onboard = expert = None; step = 0
    started = time.monotonic()
    def capture(authority):
        observation = base.policy_observation(env)
        rule = jsonable(env.rule_state())
        recorder.capture(observation, rule['physics_time_seconds'], rule)
        onboard.append(observation, step, rule['physics_time_seconds'])
        jsonl(out/'rule_evidence.jsonl', dict(control_step=step, authority=authority, **rule))
        row.update(control_steps=step, elapsed_seconds=time.monotonic()-started,
                   final_physics_safety=rule['physics_safety'], score=rule['score'])
        if step % 50 == 0:
            e.write_json(out/'result.json', row)
            check_runtime_resources(0)
        base.validate_rule(rule)
        if time.monotonic()-started > 3600:
            raise TimeoutError('Bounded physical support regression deadline')
    try:
        gpu_lock.acquire(plan['gpu_lock_owner'], max_wait=0); held = True
        env = IsaacZhuhaiEnv(plan['gpu_lock_owner'], output_dir=out)
        scene = plan['scenario']
        env.setup('zhuhai_task2', scene['seed'], randomized=scene['randomized'],
                  mapping='body_text', source_seed=scene['source_parent'])
        initial = base.policy_observation(env)
        gates = base.runtime_gate_results(env, initial); base.validate_runtime_gates(gates)
        e.write_json(out/'scene_and_gates.json', dict(scene=env.manifest, gates=gates))
        recorder = StreamDemoRecorder(out/'probe.hdf5', metadata=dict(task='zhuhai_task2',
            mode='verification_safe_prefix_replay_support_fix', split='verification',
            seed=scene['seed'], production=False, quota_credit=0), frequency=10.)
        onboard = base.Native16OnboardRecorder(out, 'support-regression-'+str(scene['seed']), scene['seed'])
        capture('initial')
        commands = [json.loads(line) for line in Path(plan['old_actions']).read_text().splitlines()]
        prefix = commands[:plan['replay_until_control_step']]
        row['status'] = 'replaying_safe_prefix'
        for index, record in enumerate(prefix):
            e.require(record['control_step'] == index, 'Original dispatched command stream is not continuous')
            action = np.asarray(record['dispatched_action'], dtype=float)
            jsonl(out/'actions.jsonl', dict(control_step=step, authority='archived_safe_prefix',
                                           dispatched_action=action.tolist()))
            step += 1; env.step_action(action, .1); capture('archived_safe_prefix')
        import h5py
        with h5py.File(plan['old_partial'], 'r') as source:
            measured = source['measured_state'][step]
        row['prefix_replay_measured_state_max_difference'] = float(np.max(np.abs(env.actual_state()-measured)))
        row['at_local_intervention'] = jsonable(env.rule_state())
        e.write_json(out/'prefix_replay_receipt.json', row)
        # Reconstruct the exact inherited 10Hz commanded velocity/acceleration
        # from the original full-precision dispatched targets, not measured q.
        last = np.asarray([r['dispatched_action'] for r in prefix[-3:]])
        limiter = BrakingRateLimiter(protocol()['rate_limit']); limiter.reset(last[-1])
        arm = list(ARM_INDICES)
        limiter.velocity = (last[-1, arm]-last[-2, arm])/.1
        previous_velocity = (last[-2, arm]-last[-3, arm])/.1
        limiter.acceleration = (limiter.velocity-previous_velocity)/.1
        limiter.feasible_interval()
        expert = SegmentedExpert(env, limiter)
        row['status'] = 'testing_repaired_local_expert'
        e.write_json(out/'result.json', row)
        def after_expert(environment):
            nonlocal step
            jsonl(out/'actions.jsonl', dict(control_step=step, authority='repaired_expert',
                phase=getattr(expert, 'active_phase', None), dispatched_action=environment.get_action().tolist(),
                rate_limit_stats=environment.expert_limit_stats))
            step += 1; capture('repaired_expert')
        env.on_control_step = after_expert
        info = expert.run_segment(plan['goal'], plan['key']); env.on_control_step = None
        e.write_json(out/'expert_result.json', jsonable(info))
        final = jsonable(env.rule_state()); base.validate_rule(final)
        data = recorder.finish(final_evidence=final, successful=False)
        onboard.close(complete=True)
        passed = bool(info['passed'] and info['expert_path_screening']['guarded_commands'] > 0
                      and recorder.physics_safety_passed and final['physics_safety']['passed'])
        row.update(status='passed' if passed else 'safe_rejection_or_local_goal_failure',
            passed=passed, expert_result=info, final_rule_evidence=final,
            data_path=str(data), data_sha256=e.sha256_file(data),
            scene_reset_after_initial_setup=False, policy_inference=False,
            independent_of_production_quota=True)
        for path, expected in plan['sources'].items():
            e.require(e.sha256_file(path) == expected, 'Probe source/input drift: '+path)
        e.write_json(out/'result.json', jsonable(row))
        return 0 if passed else 1
    except BaseException as exc:
        row.update(status='failed', passed=False, error=f'{type(exc).__name__}: {exc}',
                   elapsed_seconds=time.monotonic()-started)
        if env is not None:
            row['actual_rule_evidence_at_failure'] = jsonable(env.rule_state())
        e.write_json(out/'result.json', jsonable(row)); raise
    finally:
        if env is not None:
            env.on_control_step = None
        if recorder is not None and not recorder.closed:
            recorder.abort()
        if onboard is not None and not onboard.manifest['complete']:
            onboard.close(complete=False)
        if env is not None:
            env.close()
        if held:
            gpu_lock.release(plan['gpu_lock_owner'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    raise SystemExit(execute(e.read_json(args.plan)))
