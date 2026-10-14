#!/usr/bin/env python3
"""Alternate real WAM rollout and locally verified expert interventions."""
from pathlib import Path
import argparse
import fcntl
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_zhuhai_precision_official_braking as frozen
from harness.zhuhai_dagger import StallTrigger
from harness.zhuhai_dagger_parallel import ResourcePressure, check_runtime_resources
from harness.zhuhai_dagger_resource_v10 import outer_failure_fields
from harness.zhuhai_dagger_segmented import (POLICY_PROTOCOL, choose_goal, finish_segments,
                                             required_visual_frames, limiter_state)


class EpisodeBudgetReached(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--worker-slot', type=int, required=True)
    p.add_argument('--gpu-lock-owner', required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    scenario = plan['candidates'][args.index]
    assert args.gpu_lock_owner == plan['gpu_lock_owner'] and scenario['split'] == 'train'
    assert 0 <= args.worker_slot < plan['parallel']['maximum_workers']
    frozen.configure()
    base = frozen.base
    import numpy as np
    from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
    from harness.isaac_zhuhai_segmented_expert_v8 import SegmentedExpert
    from harness.isaac_zhuhai_support_collision_v8 import VERSION as SUPPORT_SCREEN_VERSION
    from harness.zhuhai_demo_recording import StreamDemoRecorder, data_contract, jsonable
    from harness.zhuhai_robot_spec import ARM_INDICES
    from harness import zhuhai_official_braking as braking
    from harness.robotwin_eval_evidence import jsonl
    out = Path(plan['out'])/'episodes'/f'seed_{scenario["seed"]}'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    row = dict(schema=plan['schema'], scenario=scenario, seed=scenario['seed'], status='starting',
        policy_full_task_success=False, expert_full_task_success=False, accepted_for_training=False,
        record_video=False, third_view=False, worker_slot=args.worker_slot,
        checkpoint_sha256=plan['checkpoint']['sha256'], protocol='segmented_local_interventions',
        expert_path_screening_protocol='panda-expert-fixed-support-screen-v1')
    base.atomic_json(out/'result.json', row)
    env = client = onboard = rec = slot_lock = expert = None
    authorities, phases, segments, resumptions = [], [], [], []
    states, commands = [], []
    kind = step = 0
    allowed = None
    expert_failure = resource_failure = None
    termination = 'unknown'

    def capture(environment):
        nonlocal allowed
        allowed = base.policy_observation(environment)
        rule = jsonable(environment.rule_state())
        if step and len(authorities) < step:
            authorities.append(kind)
            phases.append(getattr(expert, 'active_phase', 'expert') if kind else 'policy')
        rec.capture(allowed, rule['physics_time_seconds'], rule)
        onboard.append(allowed, step, rule['physics_time_seconds'])
        states.append(np.asarray(allowed['measured_joint_gripper_state'], np.float32).tolist())
        commands.append(np.asarray(allowed['commanded_joint_gripper_state'], np.float32).tolist())
        jsonl(out/'rule_evidence.jsonl', dict(control_step=step, authority='expert' if kind else 'policy', **rule))
        row.update(control_steps=step, policy_steps=authorities.count(0), expert_control_steps=authorities.count(1),
                   elapsed_wall_seconds=time.monotonic()-started, partial_base_score=rule['score']['base_score'])
        if rule.get('physics_safety',{}).get('passed') is not True:
            row['physics_safety_at_failure']=rule.get('physics_safety')
        base.validate_rule(rule)
        if step % 10 == 0:
            base.atomic_json(out/'result.json', row)
        if time.monotonic()-started > 3600:
            raise EpisodeBudgetReached('One-hour collection wall budget')
        if step >= 2400 and not rule['score']['full_task_success']:
            raise EpisodeBudgetReached('2400 real control-step budget')
        if step % 50 == 0:
            check_runtime_resources(args.worker_slot)
            from scripts.collect_isaac_zhuhai_recycle import check_disk
            check_disk(out)

    def close_policy():
        nonlocal client, slot_lock
        if client is not None:
            client.close()
            client = None
        if slot_lock is not None:
            slot_lock.close()
            slot_lock = None

    def open_policy(round_index):
        nonlocal client, slot_lock
        slot_lock = Path(plan['parallel']['policy_slot_lock']).open('a+')
        row['status'] = 'waiting_for_policy_slot'
        base.atomic_json(out/'result.json', row)
        while True:
            try:
                fcntl.flock(slot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                check_runtime_resources(args.worker_slot)
                if time.monotonic()-started > 3600:
                    raise EpisodeBudgetReached('Policy-slot wait wall budget')
                time.sleep(.5)
        command = [str(ROOT/'.venvs/wam-fixed/bin/python'), '-B',
                   str(ROOT/'scripts/precision_segmented_policy_worker.py'), '--worker',
                   '--checkpoint', plan['checkpoint']['path'], '--task', 'zhuhai_task2',
                   '--gpu-lock-owner', args.gpu_lock_owner]
        client = base.Native16PolicyClient(command, out/f'policy_worker_round_{round_index}.log', 180, 120)
        runtime = client.ready['runtime']
        if (runtime.get('segmented_policy_support') != POLICY_PROTOCOL
                or runtime.get('segmented_worker_source_sha256') != base.identity(command[2])['sha256']):
            raise ValueError('Segmented policy worker identity mismatch')
        client.reset(name, scenario['seed'])
        row['policy_ready'] = client.ready
        if step:
            frames = {}
            lags = contract['config']['model']['v3']['memory_lags']
            for index in required_visual_frames(step, lags):
                with np.load(out/'onboard'/f'observation_{index:06d}.npz', allow_pickle=False) as obs:
                    observation = dict(rgb={v: obs[v].copy() for v in base.VIEWS},
                        measured_joint_gripper_state=obs['measured_joint_gripper_state'].copy(),
                        commanded_joint_gripper_state=obs['commanded_joint_gripper_state'].copy())
                frames[str(index)] = base.encode_observation(observation)
            if not np.allclose(limiter.previous, np.asarray(allowed['commanded_joint_gripper_state'])[list(ARM_INDICES)], rtol=0, atol=1e-6):
                raise ValueError('Inherited command does not match actual issued target')
            receipt = client.request(base.envelope('restore_history', name, step,
                states=states[:step], commands=commands[:step], visuals=frames, control=limiter_state(limiter)), 'history_restored')
            resumptions.append(dict(start=step, model_control_steps=0, restored_history=receipt,
                                    after_segment=len(segments)-1))
            base.atomic_json(out/f'history_resume_round_{round_index}.json', resumptions[-1])
        row['status'] = 'policy_rollout'
        base.atomic_json(out/'result.json', row)

    def rollout(round_index):
        nonlocal step, kind, termination
        kind = 0
        try:
            open_policy(round_index)
            settings = plan['takeover_gate'] if round_index == 0 else plan['segmented']['resume_takeover_gate']
            trigger = StallTrigger(**settings)
            initial = step
            for local_index in range(settings['maximum_steps']):
                reply = client.action(name, step, allowed)
                action, stats = limiter.step(reply['smoothed_action'])
                if not np.array_equal(action, reply['limited_action']) or stats != reply['rate_limit_stats']:
                    raise ValueError('Resumed worker and inherited parent limiter differ')
                jsonl(out/'actions.jsonl', dict(control_step=step, authority='policy', round=round_index,
                    raw_action=reply['model_raw_action'], dispatched_action=action.tolist(), policy=reply))
                step += 1
                env.step_action(action, .1)
                if round_index:
                    resumptions[-1]['model_control_steps'] += 1
                capture(env)
                evidence = jsonable(env.rule_state())
                if evidence['score']['full_task_success']:
                    row['policy_full_task_success'] = True
                    termination = 'full_task_finished_by_policy'
                    return None
                reason = trigger.observe(step-initial, evidence)
                if reason:
                    return reason
            return 'bounded_resumed_policy_without_full_success'
        finally:
            close_policy()

    try:
        base.verify_owner(args.gpu_lock_owner)
        contract, receipt = base.read_contract(plan['checkpoint']['path'])
        if receipt['sha256'] != plan['checkpoint']['sha256']:
            raise ValueError('Frozen collection checkpoint mismatch')
        env = IsaacZhuhaiEnv(args.gpu_lock_owner, output_dir=out)
        env.setup('zhuhai_task2', scenario['seed'], randomized=scenario['randomized'],
                  mapping='body_text', source_seed=scenario['source_parent'])
        allowed = base.policy_observation(env)
        gates = base.runtime_gate_results(env, allowed)
        base.validate_runtime_gates(gates)
        row.update(admission_gates=gates, scene_manifest=env.manifest)
        metadata = dict(task='zhuhai_task2', track='life', mode='segmented_dagger', split='train',
            seed=scenario['seed'], family_id=scenario['family_id'], parent_source_seed=scenario['source_parent'],
            data_contract=data_contract(), record_video=False, third_view=False, checkpoint_sha256=receipt['sha256'],
            plan_sha256=base.identity(args.plan)['sha256'], policy_fields=list(base.OBSERVATION_FIELDS),
            collection_gate='privileged local-goal teacher; not deployed monitor',
            expert_path_screening_protocol=SUPPORT_SCREEN_VERSION,
            supervision='verified local expert spans only; all other real frames causal context; whole-task outcome independent')
        rec = StreamDemoRecorder(out/'correction.hdf5', metadata=metadata, frequency=10.)
        name = f'zhuhai_task2_seed_{scenario["seed"]}'
        onboard = base.Native16OnboardRecorder(out, name, scenario['seed'])
        capture(env)
        limiter = braking.BrakingRateLimiter(contract['action_postprocessing']['rate_limit'])
        limiter.reset(allowed['commanded_joint_gripper_state'])
        try:
            for round_index in range(plan['segmented']['maximum_interventions']+1):
                reason = rollout(round_index)
                if reason is None:
                    break
                if round_index == plan['segmented']['maximum_interventions']:
                    termination = 'intervention_limit_after_real_policy_handback'
                    break
                goal, key = choose_goal(env.rule_state(), segments)
                kind = 1
                expert = SegmentedExpert(env, limiter)
                start = step
                row.update(status='expert_recovery', local_goal=goal, local_object=key,
                           takeover_step=segments[0]['start'] if segments else step, takeover_reason=reason,
                           takeover_evidence=jsonable(env.rule_state()))
                base.atomic_json(out/'result.json', row)
                def after_expert(environment):
                    nonlocal step
                    step += 1
                    jsonl(out/'actions.jsonl', dict(control_step=step-1, authority='expert',
                        segment=len(segments), phase=getattr(expert, 'active_phase', None),
                        dispatched_action=environment.get_action().tolist(), rate_limit_stats=environment.expert_limit_stats))
                    capture(environment)
                env.on_control_step = after_expert
                info = expert.run_segment(goal, key)
                env.on_control_step = None
                segment = dict(start=start, end=step, goal=goal, key=key, passed=info['passed'], failure=info['failure'])
                if step <= start:
                    raise ValueError('Local intervention produced no real action')
                segments.append(segment)
                base.atomic_json(out/f'expert_segment_{len(segments)-1:02d}.json', jsonable(info))
                row.update(segments=segments, verified_segment_count=sum(s['passed'] for s in segments),
                           policy_resumptions=resumptions,expert_path_screening=info['expert_path_screening'])
                base.atomic_json(out/'result.json', row)
                if info['failure']:
                    if info['failure'].startswith('ResourcePressure:'):
                        resource_failure = info['failure']
                    elif info['failure'].startswith('EpisodeBudgetReached:'):
                        termination = 'control_or_wall_budget'
                    else:
                        expert_failure = info['failure']
                    termination = termination if termination != 'unknown' else 'expert_segment_failed'
                    break
                if info['full_task_success']:
                    row['expert_full_task_success'] = True
                    termination = 'full_task_finished_by_local_expert'
                    break
        except EpisodeBudgetReached:
            termination = 'control_or_wall_budget'
        except ResourcePressure as exc:
            resource_failure = f'{type(exc).__name__}: {exc}'
            termination = 'resource_pause'
        final = jsonable(env.rule_state())
        base.validate_rule(final)
        valid_steps = finish_segments(rec, authorities, phases, segments)
        handbacks = sum(r['model_control_steps'] > 0 for r in resumptions)
        accepted = bool(valid_steps > 0 and handbacks > 0 and rec.physics_safety_passed and resource_failure is None)
        data = rec.finish(final_evidence=final, successful=accepted)
        onboard.close(complete=True)
        row.update(status='resource_paused' if resource_failure else 'segmented_admitted' if accepted
                   else 'expert_failed' if expert_failure else 'policy_succeeded_without_correction' if row['policy_full_task_success']
                   else 'bounded_no_correction', accepted_for_training=accepted, termination=termination,
            expert_failure=expert_failure, resource_failure=resource_failure,
            expert_steps=valid_steps, expert_control_steps=authorities.count(1), policy_steps=authorities.count(0),
            takeover_step=segments[0]['start'] if segments else None, segments=segments,
            verified_segment_count=sum(s['passed'] for s in segments), policy_resumptions=resumptions,
            handback_count=handbacks, full_task_success=bool(final['score']['full_task_success']),
            final_rule_evidence=final, data_path=str(data.resolve()), data_sha256=base.identity(data)['sha256'],
            all_observations_safe=rec.physics_safety_passed)
        base.atomic_json(out/'result.json', row)
        return 0
    except BaseException as exc:
        fields, code = outer_failure_fields(exc, step)
        row.update(**fields, elapsed_wall_seconds=time.monotonic()-started)
        base.atomic_json(out/'result.json', row)
        return code
    finally:
        if env is not None:
            env.on_control_step = None
        close_policy()
        if rec is not None and not rec.closed:
            rec.abort()
        if onboard is not None and not onboard.manifest['complete']:
            onboard.close(complete=False)
        if env is not None:
            env.close()


if __name__ == '__main__':
    raise SystemExit(main())
