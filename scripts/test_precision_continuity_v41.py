#!/usr/bin/env python3
"""CPU witnesses for new references, official limits, events and dispatch."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import os
import tempfile
import numpy as np
from scipy.interpolate import CubicSpline
from harness import zhuhai_chunk_continuity_v41 as c
from harness import zhuhai_official_braking as old
from harness.zhuhai_robot_spec import HOME_ACTION
from scripts import eval_zhuhai_wam_precision_v2 as original


def main():
    settings=c.protocol(); home=np.asarray(HOME_ACTION,dtype=float)
    fast=c.FineBrakingLimiter(settings['fine_rate_limit'])
    gold=old.BrakingRateLimiter(settings['rate_limit']);gold.dt=c.DT
    braking_gold=old.BrakingRateLimiter(settings['rate_limit']);braking_gold.dt=c.DT
    braking_gold.amax-=c.ACCELERATION_INNER_MARGIN
    gold.safe_speed=braking_gold.safe_speed;gold.future_stop_distance=braking_gold.future_stop_distance
    def gold_interval():
        decrement=(gold.amax-c.ACCELERATION_INNER_MARGIN)*gold.dt
        return (np.maximum(np.maximum(-gold.vmax,gold.velocity-decrement),-gold.safe_speed(gold.previous-gold.lower)),
                np.minimum(np.minimum(gold.vmax,gold.velocity+decrement),gold.safe_speed(gold.upper-gold.previous)))
    gold.feasible_interval=gold_interval
    fast.reset(home);gold.reset(home)
    rng=np.random.default_rng(20261012)
    for _ in range(100):
        distance=rng.uniform(0,6,14)
        assert np.array_equal(fast.safe_speed(distance),gold.safe_speed(distance))
    for i in range(1200):
        requested=home.copy()
        requested[c.ARM]=np.where((i//200)%2,fast.upper,fast.lower)
        requested[c.GRIPPER]=[i%2,1-i%2]
        actual,stats=fast.step(requested);expected,witness=gold.step(requested)
        assert np.array_equal(actual,expected) and stats==witness
        assert stats['minimum_braking_surplus_rad']>=-1e-10
    raw=np.tile(home,(32,1));raw[:,c.ARM]+=np.linspace(.1,.4,32)[:,None]
    raw[:,c.GRIPPER]=rng.uniform(-.1,1.1,(32,2))
    sm=c.spline.smooth_chunk(raw,settings)
    fine,coarse=c.reference(sm,home[c.ARM],np.zeros(14))
    assert np.array_equal(fine[:,c.GRIPPER],np.repeat(raw[:,c.GRIPPER],25,axis=0))
    curve=CubicSpline((np.arange(32)+1)*.1,sm[:,c.ARM],axis=0)
    # The correction and derivative vanish at .4s; no tail time shift.
    assert np.allclose(fine[99:,c.ARM],curve((np.arange(99,800)+1)*c.DT),atol=2e-14)
    assert np.array_equal(coarse,fine[24::25])
    for bad in (np.zeros((32,14)),np.full((32,16),np.nan)):
        try:c.reference(bad,home[c.ARM],np.zeros(14))
        except ValueError:pass
        else:raise AssertionError('malformed reference admitted')
    class Context:
        def reset(self):self.states=[]
        def push(self,q,pool):self.states.append(q.copy())
        def inputs(self):return self.states[-1],np.zeros(1)
    calls=[]
    def sample(features,inputs,seed,step,event):
        previous=raw if step else None
        fresh=raw-np.r_[np.ones(7)*.1,0,np.ones(7)*.1,0]
        chunk=raw if step==0 else np.vstack([previous[8:],fresh[24:]])
        policy.commit_sampler.last_diagnostic=dict(commitment='hard_replacement_inpainting_normalized_space',
            physical_inference_delay_actions=0,execution_horizon=8,model_weights_updated=False,stage_progress_or_grasp_oracle=False,
            previous_query_step=None if step==0 else step-8,committed_actions=0 if step==0 else 24,
            fresh_rows=[0,32] if step==0 else [24,32],
            previous_unexecuted_native_chunk=None if step==0 else previous[8:].tolist(),
            denoising_trace=[] if step==0 else [dict(tau=i/10,flow_rms=0.,suffix_flow_rms=0.,pinned_prefix_flow_rms=0.) for i in range(10)],
            suffix_boundary=None if step==0 else dict(position_jump_rad=0.,within_prefix_step_rad=0.,within_suffix_step_rad=0.))
        calls.append(step);return chunk
    policy=c.install_braking(original.clocked_policy(Context(),lambda rgb:(None,np.zeros(1)),sample,settings),settings)
    from types import SimpleNamespace
    policy.commit_sampler=SimpleNamespace(last_diagnostic=dict(cpu_mock=True))
    policy.reset('test',932106)
    state=home.astype(np.float32);reply=None;archive_actions=[];archive_states=[state.copy()]
    for step in range(16):
        observation=dict(rgb={name:np.zeros((2,2,3),np.uint8) for name in original.base.VIEWS},
            measured_joint_gripper_state=state,commanded_joint_gripper_state=state)
        reply=policy.observe('test',step,observation)
        targets=np.asarray(reply['substep_targets']);state=targets[-1].copy()
        assert targets.shape==(25,16) and reply['replanned']==(step%8==0)
        assert np.array_equal(targets[:,c.GRIPPER],np.repeat(np.clip((raw if step<8 else raw[8:])[step%8,c.GRIPPER],0,1)[None],25,axis=0))
        archive_states.append(state.copy())
        archive_actions.append(dict(control_step=step,policy=reply,smoothed_action=reply['smoothed_action'],
            dispatched_action=reply['limited_action']))
    assert calls==[0,8]
    from scripts.precision_continuity_replay_v41 import replay
    with tempfile.TemporaryDirectory(prefix='p2-continuity-cpu-') as temporary:
        directory=Path(temporary);(directory/'onboard').mkdir()
        from scripts import precision_continuity_evidence_v37 as evidence
        evidence.write_json(directory/'interface_check.json',dict(value=True))
        assert evidence.read_json(directory/'interface_check.json')==dict(value=True)
        assert callable(evidence.verify_external_owner)
        for index,state in enumerate(archive_states):
            np.savez(directory/f'onboard/observation_{index:06d}.npz',commanded_joint_gripper_state=state.astype(np.float32))
        (directory/'actions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in archive_actions))
        witness=replay(directory,settings)
        assert witness['fine_commands_verified']==400
        assert witness['independent_finite_difference_limits_verified']
        archive_actions[0]['policy']['substep_targets'][0][0]+=.0001
        (directory/'actions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in archive_actions))
        try:replay(directory,settings)
        except ValueError:pass
        else:raise AssertionError('altered fine command archive admitted')
        archive_actions[0]['policy']['substep_targets'][0][0]-=.0001
    # Exercise the actual driver override with fake physics; no simulator retry.
    from harness import isaac_zhuhai_runtime as runtime
    saved=runtime.IsaacZhuhaiEnv
    class FakeWorld:
        def __init__(self):self.steps=0
        def step(self,**kw):self.steps+=1
    class Judge:
        def __init__(self):self.calls=0
        def observe(self,*args):self.calls+=1
    class Env:
        physics_dt=c.DT
        def __init__(self):
            from harness.zhuhai_robot_spec import DUAL_JOINT_LIMITS
            self.world=FakeWorld();self.judge=Judge();self.physics_time=0.
            self._contact_pairs={};self.joint_limits=DUAL_JOINT_LIMITS;self.on_control_step=None;self.sent=[]
        def _drive(self,q):self.sent.append(q.copy())
        def _audit_physics_contacts(self):pass
        def body_states(self):return {}
        def rule_state(self):return {}
        def setup(self,*args,**kwargs):pass
        def capture(self):return {'head_camera':None,'left_camera':None,'right_camera':None}
        def close(self):pass
    runtime.IsaacZhuhaiEnv=Env;c.install_environment()
    env=runtime.IsaacZhuhaiEnv();c.set_pending(reply)
    try:c.set_pending(reply)
    except ValueError:pass
    else:raise AssertionError('packet overwrite admitted')
    env.step_action(np.asarray(reply['limited_action']))
    assert len(env.sent)==env.world.steps==env.judge.calls==25
    assert abs(env.physics_time-.1)<1e-15 and c.PENDING is None
    assert np.array_equal(np.asarray(env.sent),reply['substep_targets'])
    from harness import zhuhai_observer_video_v41 as video
    saved_recorder=video.ObserverRecorder
    class Recorder:
        opened=[]
        def __init__(self,env):self.opened.append(env);self.frames=0
        def append(self):self.frames+=1
        def close(self):pass
    video.ObserverRecorder=Recorder
    runtime.IsaacZhuhaiEnv=Env;video.install()
    saved_selection=os.environ.get('P2_CONTINUITY_THIRD_VIEW_SEED')
    os.environ['P2_CONTINUITY_THIRD_VIEW_SEED']='932106'
    first=runtime.IsaacZhuhaiEnv();first.setup(task='zhuhai_task2',seed=932106)
    other=runtime.IsaacZhuhaiEnv();other.setup(task='zhuhai_task2',seed=932107)
    assert len(Recorder.opened)==1 and first.rule_state()['video_recording'] is True
    assert other.observer is None and other.rule_state()['video_recording'] is False
    assert set(first.capture())==set(original.base.VIEWS) and first.observer.frames==1
    first.output_dir=ROOT/'nonexistent_cpu_observer_directory';first.close();other.close();video.ObserverRecorder=saved_recorder
    if saved_selection is None:os.environ.pop('P2_CONTINUITY_THIRD_VIEW_SEED',None)
    else:os.environ['P2_CONTINUITY_THIRD_VIEW_SEED']=saved_selection
    runtime.IsaacZhuhaiEnv=saved
    policy.reset('next',932107)
    assert not policy.queue and policy.limiter.previous is None and not policy.context.states
    result=dict(status='passed',official_gold_steps=1200,safe_speed_vectors=100,
        native16=True,exact_unchanged_gripper_timing=True,fine_dispatch_steps=25,
        physics_judge_all_substeps=True,exec8_alignment_and_reset=True,
        full_stream_replay_and_tamper_rejection=True,only_first_case_observer_created=True,
        frozen_sources_edited=False,success_rate_claim=False,hard_prefix_commitment_contract=True)
    out=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24/continuity_cpu_checks_v41.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
