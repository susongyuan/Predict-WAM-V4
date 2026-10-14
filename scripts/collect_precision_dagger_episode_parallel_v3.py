#!/usr/bin/env python3
"""One actual policy rollout followed by one privileged expert correction."""
from pathlib import Path
import argparse, json, math, sys, time, fcntl
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_precision_official_braking as frozen
from harness.zhuhai_dagger import StallTrigger,finish_mask
from harness.zhuhai_dagger_parallel import ResourcePressure,check_runtime_resources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--index',required=True,type=int);p.add_argument('--worker-slot',required=True,type=int);p.add_argument('--gpu-lock-owner',required=True)
    args=p.parse_args();plan=json.loads(Path(args.plan).read_text());scenario=plan['candidates'][args.index]
    assert args.gpu_lock_owner==plan['gpu_lock_owner'] and scenario['split']=='train'
    assert 0<=args.worker_slot<plan['parallel']['maximum_workers']
    frozen.configure();base=frozen.base
    import numpy as np
    from harness.isaac_zhuhai_runtime import IsaacZhuhaiEnv
    from harness.isaac_zhuhai_dagger_expert import RecoveryExpert
    from harness.zhuhai_demo_recording import StreamDemoRecorder,data_contract,jsonable
    from harness.zhuhai_robot_spec import ARM_INDICES
    from harness import zhuhai_official_braking as braking
    from harness.robotwin_eval_evidence import jsonl
    out=Path(plan['out'])/'episodes'/f'seed_{scenario["seed"]}'
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    row=dict(schema=plan['schema'],scenario=scenario,seed=scenario['seed'],status='starting',
        policy_full_task_success=False,expert_full_task_success=False,accepted_for_training=False,
        record_video=False,third_view=False,worker_slot=args.worker_slot,checkpoint_sha256=plan['checkpoint']['sha256'])
    base.atomic_json(out/'result.json',row)
    env=client=onboard=rec=slot_lock=None;authorities=[];phases=[];kind=0;expert=None;allowed=None;step=0
    def capture(environment):
        nonlocal allowed,step
        if time.monotonic()-started>3600:raise TimeoutError('Bounded correction wall budget')
        allowed=base.policy_observation(environment);rule=jsonable(environment.rule_state())
        base.validate_rule(rule)
        if step and len(authorities)<step:
            authorities.append(kind)
            phases.append(getattr(expert,'active_phase','expert') if kind else 'policy')
        rec.capture(allowed,rule['physics_time_seconds'],rule)
        onboard.append(allowed,step,rule['physics_time_seconds'])
        jsonl(out/'rule_evidence.jsonl',dict(control_step=step,authority='expert' if kind else 'policy',**rule))
        row.update(control_steps=step,policy_steps=sum(x==0 for x in authorities),expert_steps=sum(x==1 for x in authorities),
            elapsed_wall_seconds=time.monotonic()-started,partial_base_score=rule['score']['base_score'])
        if step%10==0:base.atomic_json(out/'result.json',row)
        if step>=2400 and not rule['score']['full_task_success']:raise TimeoutError('2400-step correction budget')
        if step%50==0:
            check_runtime_resources(args.worker_slot)
            from scripts.collect_isaac_zhuhai_recycle import check_disk
            check_disk(out)
    try:
        base.verify_owner(args.gpu_lock_owner)
        contract,receipt=base.read_contract(plan['checkpoint']['path'])
        if receipt['sha256']!=plan['checkpoint']['sha256']:raise ValueError('Correction checkpoint mismatch')
        env=IsaacZhuhaiEnv(args.gpu_lock_owner,output_dir=out)
        env.setup('zhuhai_task2',scenario['seed'],randomized=scenario['randomized'],
                  mapping='body_text',source_seed=scenario['source_parent'])
        allowed=base.policy_observation(env);gates=base.runtime_gate_results(env,allowed);base.validate_runtime_gates(gates)
        row['admission_gates']=gates;row['scene_manifest']=env.manifest
        metadata=dict(task='zhuhai_task2',track='life',mode='dagger',split='train',seed=scenario['seed'],
            family_id=scenario['family_id'],parent_source_seed=scenario['source_parent'],data_contract=data_contract(),
            record_video=False,third_view=False,checkpoint_sha256=receipt['sha256'],
            plan_sha256=base.identity(args.plan)['sha256'],
            policy_fields=list(base.OBSERVATION_FIELDS),collection_gate='privileged rule teacher; not deployed monitor',
            supervision='only actions after actual control transfer; entire policy prefix retained as causal context')
        rec=StreamDemoRecorder(out/'correction.hdf5',metadata=metadata,frequency=10.)
        name=f'zhuhai_task2_seed_{scenario["seed"]}'
        onboard=base.Native16OnboardRecorder(out,name,scenario['seed']);capture(env)
        command=[str(ROOT/'.venvs/wam-fixed/bin/python'),'-B',str(Path(frozen.__file__).resolve()),
            '--worker','--checkpoint',plan['checkpoint']['path'],'--task','zhuhai_task2','--gpu-lock-owner',args.gpu_lock_owner]
        slot_lock=Path(plan['parallel']['policy_slot_lock']).open('a+')
        row['status']='waiting_for_policy_slot';base.atomic_json(out/'result.json',row)
        while True:
            try:fcntl.flock(slot_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                check_runtime_resources(args.worker_slot)
                if time.monotonic()-started>3600:raise TimeoutError('Policy slot wait budget')
                time.sleep(.5)
        client=base.Native16PolicyClient(command,out/'policy_worker.log',180,60);client.reset(name,scenario['seed'])
        row['policy_ready']=client.ready;row['status']='policy_rollout'
        limiter=braking.BrakingRateLimiter(contract['action_postprocessing']['rate_limit'])
        limiter.reset(allowed['commanded_joint_gripper_state'])
        trigger=StallTrigger(**plan['takeover_gate']);reason=None
        for index in range(plan['takeover_gate']['maximum_steps']):
            reply=client.action(name,index,allowed)
            action,stats=limiter.step(reply['smoothed_action'])
            if not np.array_equal(action,reply['limited_action']) or stats!=reply['rate_limit_stats']:
                raise ValueError('Policy controller replay mismatch before expert handoff')
            jsonl(out/'actions.jsonl',dict(control_step=index,authority='policy',raw_action=reply['model_raw_action'],
                dispatched_action=action.tolist(),policy=reply))
            step+=1;env.step_action(action,.1);capture(env)
            final=jsonable(env.rule_state())
            if final['score']['full_task_success']:
                row.update(status='policy_succeeded_without_correction',policy_full_task_success=True)
                rec.finish(final_evidence=final,successful=False);onboard.close(complete=True)
                base.atomic_json(out/'result.json',row);return 0
            reason=trigger.observe(step,final)
            if reason:break
        if not reason:raise ValueError('No valid teacher trigger')
        row.update(status='expert_recovery',takeover_step=step,takeover_reason=reason,
                   takeover_evidence=jsonable(env.rule_state()))
        base.atomic_json(out/'result.json',row)
        # Close the WAM without resetting simulation. Controller state passes to
        # the expert, including previous velocity/acceleration and object poses.
        client.close();client=None;slot_lock.close();slot_lock=None;kind=1
        expert=RecoveryExpert(env,limiter)
        def after_expert(environment):
            nonlocal step
            step+=1;capture(environment)
            jsonl(out/'actions.jsonl',dict(control_step=step-1,authority='expert',phase=getattr(expert,'active_phase',None),
                dispatched_action=environment.get_action().tolist(),rate_limit_stats=environment.expert_limit_stats))
        env.on_control_step=after_expert
        info=expert.play_once();env.on_control_step=None;final=jsonable(env.rule_state())
        base.validate_rule(final)
        good=info['plan_success'] is True and final['score']['full_task_success'] is True
        finish_mask(rec,authorities,phases,row['takeover_step'])
        data=rec.finish(final_evidence=final,successful=bool(good));onboard.close(complete=True)
        base.atomic_json(out/'expert_receipt.json',jsonable(info))
        row.update(status='success' if good else 'resource_paused' if str(info['failure']).startswith('ResourcePressure:') else 'expert_failed',expert_full_task_success=bool(good),
            accepted_for_training=bool(good),expert_failure=info['failure'],final_rule_evidence=final,
            data_path=str(data.resolve()),data_sha256=base.identity(data)['sha256'],
            expert_receipt_sha256=base.identity(out/'expert_receipt.json')['sha256'],
            all_observations_safe=rec.physics_safety_passed)
        base.atomic_json(out/'result.json',row);return 0
    except ResourcePressure as exc:
        row.update(status='resource_paused',error=f'{type(exc).__name__}: {exc}',accepted_for_training=False)
        base.atomic_json(out/'result.json',row);return 0
    except BaseException as exc:
        row.update(status='infrastructure_error',error=f'{type(exc).__name__}: {exc}',
            accepted_for_training=False,elapsed_wall_seconds=time.monotonic()-started)
        base.atomic_json(out/'result.json',row);return 2
    finally:
        if env is not None:env.on_control_step=None
        if rec is not None and not rec.closed:rec.abort()
        if onboard is not None and not onboard.manifest['complete']:onboard.close(complete=False)
        if client is not None:client.close()
        if slot_lock is not None:slot_lock.close()
        if env is not None:env.close()


if __name__=='__main__':raise SystemExit(main())
