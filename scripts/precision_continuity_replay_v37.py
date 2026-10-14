"""Independent v37 audit of v36 targets; accepts only the archived exact float32 cast."""
from pathlib import Path
import json
import subprocess
import math
import numpy as np
from harness import zhuhai_chunk_continuity_v37 as c
from scripts import precision_continuity_evidence_v37 as e


def replay(directory,settings):
    directory=Path(directory)
    with np.load(directory/'onboard/observation_000000.npz',allow_pickle=False) as z:
        initial=z['commanded_joint_gripper_state'].copy()
    limiter=c.FineBrakingLimiter(settings['fine_rate_limit']);limiter.reset(initial)
    previous=initial[c.ARM].copy();previous_v=np.zeros(14);previous_a=np.zeros(14)
    actions=[e.loads(line) for line in (directory/'actions.jsonl').read_text().splitlines()]
    groups=[];brake_steps=0;max_delta=0.;previous_chunk=None
    for i,action in enumerate(actions):
        reply=action['policy'];j=i%8
        e.require(reply['replanned'] is (j==0) and reply['execution_prefix']==8,'changed execution prefix/cadence')
        if j==0:
            raw=np.asarray(reply['raw_denormalized_chunk'],dtype=float)
            diagnostic=reply['overlap_guidance']
            expected_mask=[((24-k)/25)*math.expm1((24-k)/25)/math.expm1(1) if k<24 else 0. for k in range(32)]
            e.require(np.allclose(diagnostic['soft_mask'],expected_mask,rtol=0,atol=3e-8)
                and diagnostic['physical_inference_delay_actions']==0
                and diagnostic['guidance_clip']==5. and diagnostic['execution_horizon']==8
                and diagnostic['model_weights_updated'] is False
                and diagnostic['stage_progress_or_grasp_oracle'] is False,'changed guidance contract')
            e.require(diagnostic['previous_query_step']==(None if i==0 else i-8)
                and diagnostic['overlap_actions']==(0 if i==0 else 24),'wrong overlap physical time alignment')
            if i==0:
                e.require(diagnostic['previous_unexecuted_native_chunk'] is None
                    and diagnostic['denoising_trace']==[],'first query received a fictitious prefix')
            else:
                e.require(np.array_equal(diagnostic['previous_unexecuted_native_chunk'],previous_chunk[8:]),
                    'prior overlap shifted, replayed, or substituted')
                e.require(len(diagnostic['denoising_trace'])==10 and all(t['tau']==k/10
                    and e.finite(t['guidance_rms']) and e.finite(t['flow_rms'])
                    for k,t in enumerate(diagnostic['denoising_trace'])),'guidance denoising evidence missing')
            previous_chunk=raw.copy()
            bspline=c.spline.smooth_chunk(raw,settings)
            e.require(np.array_equal(bspline,reply['bspline_denormalized_chunk']),'original spline changed')
            e.require(np.array_equal(limiter.previous,reply['continuity_anchor_q_rad'])
                and np.array_equal(limiter.velocity,reply['continuity_anchor_v_rad_s']),'C1 anchor differs')
            fine,coarse=c.reference(bspline,limiter.previous,limiter.velocity)
            e.require(np.array_equal(coarse,reply['smoothed_denormalized_chunk']),'C1 reference replay mismatch')
            groups=[fine[k*25:(k+1)*25] for k in range(8)]
        e.require(np.array_equal(raw[j],reply['model_raw_action'])
            and np.array_equal(bspline[j],reply['bspline_action'])
            and np.array_equal(coarse[j],action['smoothed_action']),'chunk indices changed')
        requested=np.asarray(reply['substep_requested_targets'],dtype=float)
        e.require(np.array_equal(requested,groups[j]) and reply['substep_dt_seconds']==c.DT
            and reply['substep_count']==25,'fine target time alignment differs')
        stats=[];targets=[]
        for target in requested:
            actual,stat=limiter.step(target);stats.append(stat);targets.append(actual)
            # Independent checks from dispatched positions, not recorded stats.
            velocity=(actual[c.ARM]-previous)/c.DT
            acceleration=(velocity-previous_v)/c.DT
            jerk=(acceleration-previous_a)/c.DT
            e.require(np.all(abs(velocity)<=limiter.vmax+1e-9)
                and np.all(abs(acceleration)<=limiter.amax+1e-8)
                and np.all(abs(jerk)<=limiter.jmax+1e-7),'independent fine velocity/acceleration/jerk violation')
            e.require(np.all(actual[c.ARM]>=limiter.lower) and np.all(actual[c.ARM]<=limiter.upper),
                'independent fine position violation')
            dec=(limiter.amax-c.ACCELERATION_INNER_MARGIN)*c.DT
            n=np.maximum(np.ceil(abs(velocity)/dec)-1,0)
            stopping=c.DT*np.maximum(n*abs(velocity)-dec*n*(n+1)/2,0)
            available=np.where(velocity>=0,limiter.upper-actual[c.ARM],actual[c.ARM]-limiter.lower)
            e.require(np.all(stopping<=available+1e-10),'independent fine stopping-distance violation')
            e.require(np.array_equal(actual[c.GRIPPER],np.clip(raw[j,c.GRIPPER],0,1)),
                'gripper event time/value differs')
            previous,previous_v,previous_a=actual[c.ARM].copy(),velocity,acceleration
        targets=np.stack(targets)
        e.require(np.array_equal(targets,reply['substep_targets'])
            and np.array_equal(targets[-1],action['dispatched_action']),'fine dispatch replay mismatch')
        aggregate=dict(stats[-1])
        for key in ('max_abs_velocity_rad_s','max_abs_acceleration_rad_s2','max_abs_jerk_rad_s3'):
            aggregate[key]=max(s[key] for s in stats)
        aggregate['minimum_braking_surplus_rad']=min(s['minimum_braking_surplus_rad'] for s in stats)
        e.require(aggregate==reply['rate_limit_stats'],'fine statistics replay mismatch')
        with np.load(directory/f'onboard/observation_{i+1:06d}.npz',allow_pickle=False) as z:
            archived=z['commanded_joint_gripper_state']
            e.require(archived.dtype==np.float32 and np.array_equal(archived,targets[-1].astype(np.float32)),
                'actual endpoint not the exact documented float32 archive representation')
        brake_steps+=int(any(s['braking_limited_dimensions']>0 for s in stats))
        max_delta=max(max_delta,max(s['arm_limit_delta_max_rad'] for s in stats))
    lo,hi=limiter.feasible_interval()
    return dict(braking_steps=brake_steps,max_command_adjustment_rad=max_delta,
        terminal_braking_interval_infeasible=bool(np.any(lo>hi+1e-10)),
        deterministic_command_replay_verified=True,fine_commands_verified=len(actions)*25,
        independent_finite_difference_limits_verified=True,actual_endpoint_archives_verified=True)


def video_evidence(directory,raw):
    directory=Path(directory)
    if raw['seed']!=932106:
        e.require(not list(directory.glob('third_person*')),'unexpected observer artifacts on unrecorded case')
        return {}
    receipt=e.read_json(directory/'third_person_video_receipt.json')
    e.require((raw.get('observer_video_receipt',receipt)==receipt) and receipt['seed']==932106
        and receipt['frames']==len((directory/'actions.jsonl').read_text().splitlines())+1
        and receipt['fps']==10 and receipt['policy_reads_observer'] is False
        and receipt['physics_advanced_by_recording'] is False and receipt['success_selection'] is False,
        'requested observer binding/count differs')
    video=directory/'third_person_full.mp4'
    e.require(receipt['video_sha256']==e.sha256_file(video),'observer video SHA differs')
    frames=directory/'third_person_frames.jsonl'
    e.require(receipt['frame_index_sha256']==e.sha256_file(frames),'observer frame index SHA differs')
    entries=[e.loads(line) for line in frames.read_text().splitlines()]
    e.require(len(entries)==receipt['frames'] and all(r['index']==i and abs(r['physics_time_seconds']-i*.1)<1e-6
        for i,r in enumerate(entries)),'observer physical frame clock differs')
    ffmpeg=next((e.ROOT/'.conda-envs/behavior6/lib/python3.12/site-packages/imageio_ffmpeg/binaries').glob('ffmpeg-linux-*'))
    result=subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-i',str(video),'-map','0:v:0',
        '-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True,timeout=180)
    decoded=[int(line.split('=',1)[1]) for line in result.stdout.splitlines() if line.startswith('frame=')]
    e.require(result.returncode==0 and decoded and max(decoded)==receipt['frames'],'full observer video decode failed')
    evidence={}
    for filename in ('third_person_full.mp4','third_person_frames.jsonl','third_person_video_receipt.json','third_person_encoder.log'):
        p=directory/filename;evidence[filename]=dict(path=str(p.resolve()),sha256=e.sha256_file(p),size=p.stat().st_size)
    return evidence
