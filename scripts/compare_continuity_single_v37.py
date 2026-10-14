#!/usr/bin/env python3
"""Three same-seed development trajectories: measured motion AND actual progress."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.compare_precision_continuity10_v36 import trace,metrics
from scripts.analyze_precision_dagger_milestones_v19 import episode
from scripts.analyze_precision_continuity_motion_v36 import analyze_episode
from harness.isaac_zhuhai_expert import FrankaFK
from harness.zhuhai_action_smoothing import ARM

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--new',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--plot');args=parser.parse_args();dest=Path(args.out);assert not dest.exists()
    v=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24'
    original_path=ROOT/'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_80k_extra20_20261011_v24/summary.json'
    original=json.loads(original_path.read_text());c1=json.loads((v/'continuity_firstcase_independent_audit_v37.json').read_text())
    new_path=Path(args.new);new=json.loads(new_path.read_text());assert new['status']=='completed' and new['seeds']==[932106]
    assert original['checkpoint']==c1['checkpoint']==new['checkpoint']
    rows=dict(original=original['episodes'][0],C1_fine250=c1['episodes'][0],RTC_C1_fine250=new['episodes'][0])
    assert all(r['seed']==932106 for r in rows.values())
    count=min(r['control_steps'] for r in rows.values());assert count>=4
    results={};arrays={}
    for label,row in rows.items():
        progress=episode(row);directory=Path(row['raw_result_path']).parent
        data=(directory/'actions.jsonl').read_bytes();actions=[json.loads(line) for line in data.splitlines()]
        q=[]
        for i in range(count+1):
            relative=f'onboard/observation_{i:06d}.npz';data=(directory/relative).read_bytes()
            assert hashlib.sha256(data).hexdigest()==row['evidence_files'][relative]['sha256']
            with np.load(io.BytesIO(data)) as z:q.append(z['measured_joint_gripper_state'].astype(float))
        q=np.stack(q);arrays[label]=q
        overlaps=[];tail_only=0;any_close=0;queries=0
        previous=None
        for a in actions:
            if not a['policy']['replanned']:continue
            chunk=np.asarray(a['policy']['raw_denormalized_chunk']);queries+=1
            if previous is not None:overlaps.append(float(np.sqrt(np.mean((chunk[:24,ARM]-previous[8:,ARM])**2))))
            has=bool((chunk[:,7]<=.2).any());prefix=bool((chunk[:8,7]<=.2).any())
            tail_only+=int(has and not prefix);any_close+=int(has);previous=chunk
        results[label]=dict(score=row['observed_base_score'],full_task_success=row['full_task_success'],
            status=row['status'],control_steps=row['control_steps'],progress=progress,
            common_prefix_motion=trace(row,count),raw_generation_overlap_rms_rad=float(np.mean(overlaps)),
            model_queries=queries,queries_with_left_close_anywhere=any_close,queries_with_left_close_only_unexecuted_tail=tail_only,
            executed_left_closed_steps=int(sum(a['dispatched_action'][7]<=.2 for a in actions)),
            any_task_object_contact_frames=sum(int(any(r['objects'][k]['touching_gripper'] for k in ('bowl','spoon','cup','plate')))
                for r in map(json.loads,(directory/'rule_evidence.jsonl').read_text().splitlines())),
            no_verified_grasp_seconds=row['control_steps']*.1 if not any(o['ever_verified_grasp'] for o in progress['objects'].values()) else None)
    settings=json.loads((new_path.parent/'plan.json').read_text())['action_postprocessing']
    detailed=analyze_episode(rows['RTC_C1_fine250'],settings,FrankaFK())
    initial=results['original']['common_prefix_motion']['measured'];final=results['RTC_C1_fine250']['common_prefix_motion']['measured']
    result=dict(schema='same-seed-three-protocol-motion-and-progress-v37',status='passed',seed=932106,
        checkpoint=new['checkpoint'],common_physical_steps=count,trajectories=results,detailed_new_motion_geometry=detailed,
        measured_acceleration_rms_reduction_vs_original=1-final['acceleration_rms_rad_s2']/initial['acceleration_rms_rad_s2'],
        measured_jerk_rms_reduction_vs_original=1-final['jerk_rms_rad_s3']/initial['jerk_rms_rad_s3'],
        summary_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (original_path,v/'continuity_firstcase_independent_audit_v37.json',new_path)},
        scope=['One predeclared development seed, not success-rate evidence or an isolated causal study.',
            'Original has no observer; both new cases have the same requested observer camera and frozen physical stepping.',
            '14 arm joints exclude grippers; derivatives are measured10Hz descriptive quantities, not250Hz actual-joint guarantees.',
            'Reduced speed/travel can reduce derivative RMS without improving task progress.',
            'RTC generated output is archived as model_raw_action BEFORE B-spline/C1/official braking; it is not an unguided counterfactual.'])
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if args.plot:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        labels=list(results);colors=['#64748b','#e58b32','#2563eb']
        for ax,key,title in zip(axes[0],['acceleration_rms_rad_s2','jerk_rms_rad_s3','total14_joint_travel_rad'],
            ['Measured arm acceleration RMS (rad/s²)','Measured arm jerk RMS (rad/s³)','Total14-joint travel (rad)']):
            values=[results[k]['common_prefix_motion']['measured'][key] for k in labels]
            ax.bar(labels,values,color=colors);ax.set_title(title);ax.tick_params(axis='x',labelrotation=12)
            for i,x in enumerate(values):ax.text(i,x,f'{x:.3f}',ha='center',va='bottom')
        for label,color in zip(labels,colors):
            q=arrays[label];t=np.arange(len(q))*.1
            axes[1,0].plot(t,q[:,6],label=label,color=color,lw=.8)
            axes[1,1].plot(t,q[:,7],label=label,color=color,lw=.8)
            axes[1,2].bar(label,results[label]['score'],color=color)
        axes[1,0].set(title='Actual left arm joint7 (rad)',xlabel='Physics time (s)');axes[1,0].legend(fontsize=8)
        axes[1,1].set(title='Actual left gripper state (0 closed /1 open)',xlabel='Physics time (s)')
        axes[1,2].set(title='Strict whole-trial score (maximum30)',ylim=(0,30));axes[1,2].tick_params(axis='x',labelrotation=12)
        fig.suptitle('Fixed80k, seed932106 | Continuity and task progress must be assessed separately')
        fig.savefig(args.plot,dpi=170);fig.savefig(Path(args.plot).with_suffix('.pdf'));plt.close(fig)
    print(json.dumps({k:dict(score=r['score'],closed_steps=r['executed_left_closed_steps'],overlap=r['raw_generation_overlap_rms_rad'],motion=r['common_prefix_motion']['measured']) for k,r in results.items()}))

if __name__=='__main__':main()
