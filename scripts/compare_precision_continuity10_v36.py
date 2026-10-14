#!/usr/bin/env python3
"""Preregistered paired10 comparison; common-prefix motion and strict outcomes."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import io
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_precision_dagger_milestones_v19 import episode,ORDER
from harness.zhuhai_action_smoothing import ARM


def sha(data):return hashlib.sha256(data).hexdigest()


def metrics(q,replanned):
    q=q[:,ARM];velocity=np.diff(q,axis=0)/.1
    acceleration=np.diff(velocity,axis=0)/.1;jerk=np.diff(acceleration,axis=0)/.1
    v=float(np.sqrt(np.mean(velocity**2)));a=float(np.sqrt(np.mean(acceleration**2)))
    mask=np.asarray(replanned[2:],bool)|np.asarray(replanned[1:-1],bool)
    energy=np.sum(acceleration**2,axis=1)
    return dict(velocity_rms_rad_s=v,acceleration_rms_rad_s2=a,
        jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk**2))),
        acceleration_to_velocity_ratio_per_s=a/v if v else None,
        total14_joint_travel_rad=float(abs(np.diff(q,axis=0)).sum()),
        boundary_acceleration_energy_fraction=float(energy[mask].sum()/energy.sum()) if energy.sum() else None)


def trace(row,count):
    directory=Path(row['raw_result_path']).parent
    data=(directory/'actions.jsonl').read_bytes()
    assert sha(data)==row['evidence_files']['actions.jsonl']['sha256']
    actions=[json.loads(line) for line in data.splitlines()][:count]
    q=[]
    for i in range(1,count+1):
        name=f'onboard/observation_{i:06d}.npz';data=(directory/name).read_bytes()
        assert sha(data)==row['evidence_files'][name]['sha256']
        with np.load(io.BytesIO(data),allow_pickle=False) as z:q.append(z['measured_joint_gripper_state'].copy())
    measured=np.stack(q);sent=np.asarray([a['dispatched_action'] for a in actions]);replan=[a['policy']['replanned'] for a in actions]
    return dict(measured=metrics(measured,replan),command_endpoints10hz=metrics(sent,replan),
        mean_absolute_measured_target_error_rad=float(abs(measured[:,ARM]-sent[:,ARM]).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True);parser.add_argument('--new',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--plot')
    args=parser.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('preserve previous comparison')
    before=Path(args.baseline).read_bytes();after=Path(args.new).read_bytes()
    b=json.loads(before);n=json.loads(after);seeds=list(range(932106,932116))
    assert b['status']==n['status']=='completed' and n['seeds']==seeds and len(n['episodes'])==10
    assert b['checkpoint']==n['checkpoint']
    baseline={r['seed']:r for r in b['episodes']};new={r['seed']:r for r in n['episodes']};rows=[]
    for seed in seeds:
        old,now=baseline[seed],new[seed];count=min(old['control_steps'],now['control_steps'])
        assert count>=4,'too few physics actions for paired jitter metrics'
        progress_old,progress_new=episode(old),episode(now)
        rows.append(dict(seed=seed,common_dispatched_steps=count,common_physics_seconds=count*.1,
            baseline=trace(old,count),new=trace(now,count),
            baseline_score=old['observed_base_score'],new_score=now['observed_base_score'],
            baseline_full=old['full_task_success'],new_full=now['full_task_success'],
            baseline_stage=progress_old['failure_stage'],new_stage=progress_new['failure_stage'],
            baseline_status=old['status'],new_status=now['status'],
            baseline_objects=progress_old['objects'],new_objects=progress_new['objects'],
            observer_recorded=seed==932106))
        print(seed,'common-prefix metrics verified',flush=True)
    def aggregate(selected):
        value={}
        for side in ('baseline','new'):
            value[side]=dict(full_successes=sum(r[side+'_full'] for r in selected),
                scores=[r[side+'_score'] for r in selected],failure_stages=dict(Counter(r[side+'_stage'] for r in selected)),
                placements={key:sum(r[side+'_objects'][key]['ever_valid_placement'] for r in selected) for key in ORDER},
                measured_motion_mean={key:float(np.mean([r[side]['measured'][key] for r in selected]))
                    for key in selected[0][side]['measured'] if all(r[side]['measured'][key] is not None for r in selected)})
        value['tests']=len(selected)
        value['relative_measured_acceleration_reduction']=1-value['new']['measured_motion_mean']['acceleration_rms_rad_s2']/value['baseline']['measured_motion_mean']['acceleration_rms_rad_s2']
        value['relative_measured_jerk_reduction']=1-value['new']['measured_motion_mean']['jerk_rms_rad_s3']/value['baseline']['measured_motion_mean']['jerk_rms_rad_s3']
        return value
    primary=aggregate(rows);secondary=aggregate(rows[1:])
    result=dict(schema='fixed80k-paired-continuity10-v36',status='verified_comparison',
        baseline_summary_sha256=sha(before),new_summary_sha256=sha(after),checkpoint=n['checkpoint'],
        primary_all10=primary,secondary_preregistered_nonrecorded9=secondary,episodes=rows,
        interpretation=['All original outcomes preserved; exact preregistered first10 seeds, no outcome selection.',
            'Motion uses paired common physical prefixes and14 arm joints, excludes grippers;10Hz descriptive derivatives are not full measured250Hz dynamics.',
            'C1 reference and250Hz dispatch changed together; first case additionally has requested observer rendering.',
            'Less motion or slower motion may reduce derivative RMS; travel and velocity must be considered alongside grasp/placement progress.',
            'This development comparison cannot establish under/overfitting, data sufficiency or single-component causality.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        keys=['acceleration_rms_rad_s2','jerk_rms_rad_s3','total14_joint_travel_rad']
        titles=['Measured acceleration RMS (rad/s2)','Measured jerk RMS (rad/s3)','Total14-joint travel (rad)']
        x=np.arange(10)
        for ax,key,title in zip(axes,keys,titles):
            ax.bar(x-.18,[r['baseline']['measured'][key] for r in rows],.36,label='Original80k')
            ax.bar(x+.18,[r['new']['measured'][key] for r in rows],.36,label='C1/fine250')
            ax.set_title(title);ax.set_xticks(x,[str(s)[-3:] for s in seeds],rotation=45);ax.set_xlabel('Seed suffix');ax.grid(axis='y',alpha=.2)
        axes[0].legend();fig.suptitle('Fixed weights, paired common physical prefixes |10Hz measured state, arms only')
        fig.savefig(args.plot,dpi=160);plt.close(fig)
    print(json.dumps(dict(primary=primary,secondary9=secondary),ensure_ascii=False))


if __name__=='__main__':main()
