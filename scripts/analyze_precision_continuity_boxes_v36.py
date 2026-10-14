#!/usr/bin/env python3
"""Offline arm-only plan-boundary and box-region diagnostics from archives.

No evaluator/model calls. Box proximity uses FK tool reference points, not
finger geometry. Contact and score GT are offline diagnostics, never control.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.isaac_zhuhai_expert import FrankaFK

ARM=np.array([*range(7),*range(8,15)])
SIDES={'left':(np.arange(7),7),'right':(np.arange(8,15),15)}


def sha(data):return hashlib.sha256(data).hexdigest()


def require(flag,message):
    if not flag:raise ValueError(message)


def runs(flags):
    result=[];start=None
    for i,flag in enumerate([*flags,False]):
        if flag and start is None:start=i
        if not flag and start is not None:result.append((start,i-1));start=None
    return result


def stats(q,indices,replanned):
    delta=np.diff(q[:,indices],axis=0)
    boundary=replanned[1:];acceleration=np.diff(delta,axis=0)/.01
    energy=np.sum(acceleration**2,axis=1)
    accmask=replanned[2:]|replanned[1:-1]
    norms=np.linalg.norm(delta,axis=1)
    within=float(norms[~boundary].mean());edge=float(norms[boundary].mean())
    return dict(acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration**2))),
        replan_or_next_acceleration_energy_fraction=float(energy[accmask].sum()/energy.sum())
            if energy.sum()>0 else None,
        replan_or_next_acceleration_samples=int(accmask.sum()),
        total_acceleration_samples=len(acceleration),
        boundary_step_norm_mean_rad=edge,within_step_norm_mean_rad=within,
        boundary_to_within_step_norm_ratio=edge/within if within>0 else None)


def analyze(row,fk):
    directory=Path(row['raw_result_path']).parent
    bound={}
    def read(name):
        data=(directory/name).read_bytes()
        require(sha(data)==row['evidence_files'][name]['sha256'],'Archive hash differs: '+name)
        bound[name]=sha(data)
        return [json.loads(line) for line in data.splitlines()]
    actions=read('actions.jsonl');rules=read('rule_evidence.jsonl')
    frames=read('onboard/frames.jsonl');q=[];targets=[]
    require(len(frames)==len(actions)+1==len(rules),'Archive alignment differs')
    require([r['control_step'] for r in rules]==list(range(len(rules))), 'Rule steps differ')
    require([r['control_step'] for r in frames]==list(range(len(frames))), 'Frame steps differ')
    for f in frames:
        b=(directory/'onboard'/f['file']).read_bytes()
        require(sha(b)==row['evidence_files']['onboard/'+f['file']]['sha256'],'Measured hash differs')
        with np.load(io.BytesIO(b),allow_pickle=False) as z:
            q.append(z['measured_joint_gripper_state'].astype(float))
            targets.append(z['commanded_joint_gripper_state'].astype(float))
    q=np.stack(q);targets=np.stack(targets)
    replanned=np.array([a['policy']['replanned'] for a in actions],dtype=bool)
    arrays={'model_raw':np.array([a['policy']['model_raw_action'] for a in actions]),
        'c1_reference_target':np.array([a['policy']['smoothed_action'] for a in actions]),
        'original_bspline_target':np.array([a['policy']['bspline_action'] for a in actions]),
        'dispatched_target':np.array([a['dispatched_action'] for a in actions]),
        'measured_after_action':q[1:]}
    require(np.allclose(targets[1:],arrays['dispatched_target'],atol=3e-7,rtol=0),
            'Measured archive target does not match action t')
    motion={name:{side:stats(value,indices,replanned)
        for side,indices in [('both_arms',ARM),*[(s,ix) for s,(ix,g) in SIDES.items()]]}
        for name,value in arrays.items()}
    boxes={}
    for side,(indices,gripper_index) in SIDES.items():
        points=np.stack([fk.fk(side,state[indices])[:3,3] for state in q])
        sideboxes={}
        for color in rules[0]['destination_interiors_world_m']:
            lows=[];highs=[]
            for rule in rules:
                lo,hi=rule['destination_interiors_world_m'][color]
                lows.append(np.array(lo)+[-.02,-.02,0.])
                highs.append(np.array(hi)+[.02,.02,.25])
            lows=np.stack(lows);highs=np.stack(highs)
            flags=np.all((points>=lows)&(points<=highs),axis=1)
            visits=[]
            for start,end in runs(flags):
                if end-start+1<3:continue
                segment=rules[start:end+1]
                contacts={key:sum(r['objects'][key]['touching_gripper'] for r in segment)
                    for key in ('bowl','spoon','cup','plate')}
                held={key:sum(r['objects'][key]['touching_gripper']
                    and not r['objects'][key]['released']
                    and min(p[2] for p in r['objects'][key]['world_corners_m'])>
                        r['table_surface_world_m']+.01 for r in segment)
                    for key in contacts}
                visits.append(dict(first_observation_step=start,last_observation_step=end,
                    start_physics_time_seconds=rules[start]['physics_time_seconds'],
                    end_physics_time_seconds=rules[end]['physics_time_seconds'],
                    archived_observations=end-start+1,
                    contact_observations_with_any_gripper_by_object=contacts,
                    strict_lifted_hold_observations_with_any_gripper_by_object=held,
                    no_task_object_gripper_contact_in_all_archived_visit_observations=not any(contacts.values()),
                    no_strict_lifted_hold_in_all_archived_visit_observations=not any(held.values()),
                    verified_grasp_objects_at_entry=[key for key,item in rules[start]['score']['items'].items()
                        if item['grasp_awarded']],
                    validly_placed_objects_at_entry=[key for key,item in rules[start]['score']['items'].items()
                        if item['placement_valid']],
                    measured_this_gripper_closed_observations=int((q[start:end+1,gripper_index]<=.2).sum()),
                    commanded_this_gripper_closed_observations=int((targets[start:end+1,gripper_index]<=.2).sum()),
                    unsafe_observations=sum(not r['physics_safety']['passed'] for r in segment)))
            sideboxes[color]=dict(all_region_observations=int(flags.sum()),
                retained_visits_minimum3_observations=len(visits),
                retained_visits_without_any_task_object_gripper_contact=sum(
                    r['no_task_object_gripper_contact_in_all_archived_visit_observations'] for r in visits),
                retained_visits_without_strict_lifted_hold=sum(
                    r['no_strict_lifted_hold_in_all_archived_visit_observations'] for r in visits),visits=visits)
        boxes[side]=sideboxes
    return dict(seed=row['seed'],control_steps=row['control_steps'],status=row['status'],
        score=row['observed_base_score'],full_task_success=row['full_task_success'],
        archive_files_verified=len(frames)+len(bound),archive_manifest_bindings=bound,
        source_result_sha256=row['raw_result_sha256'],motion=motion,box_regions=boxes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',action='append',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();out=Path(args.out);require(not out.exists(),'Preserve previous evidence')
    fk=FrankaFK();cohorts={};bindings={}
    for item in args.cohort:
        label,path=item.split('=',1);p=Path(path);b=p.read_bytes();s=json.loads(b)
        require(s['status']=='completed' or args.allow_partial,'Evaluation incomplete')
        rows=[]
        for row in s['episodes']:
            value=analyze(row,fk);rows.append(value)
            print(json.dumps(dict(label=label,seed=row['seed'],
                measured_boundary=value['motion']['measured_after_action']['both_arms'],
                left_box_visits={c:v['retained_visits_minimum3_observations']
                    for c,v in value['box_regions']['left'].items()})),flush=True)
        cohorts[label]=dict(status_at_snapshot=s['status'],tests=len(rows),episodes=rows)
        bindings[label]=dict(path=str(p.resolve()),sha256=sha(b))
    result=dict(schema='continuity-whole-arm-boundary-and-bilateral-box-regions-v36',
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        cohorts=cohorts,summary_bindings=bindings,analyzer_sha256=sha(Path(__file__).read_bytes()),
        region_definition=dict(tool_reference_point=True,XY_expansion_m=.02,
            Z_from_interior_bottom_to_top_plus_m=.25,minimum_consecutive_archive_observations=3),
        limitations=['Arm metrics exclude both gripper dimensions.',
            'Action t matches measured observation t+1. Acceleration uses10Hz finite differences.',
            'Box-region visits do not prove grasping a box or end-effector collision.',
            'Object contact/hold predicates aggregate either gripper; cannot identify which hand by these flags alone.',
            'No inference about unarchived250Hz dynamics, PD/RNG causality, under/overfitting or model capacity.',
            'Unsafe terminal archive poses, if present, are diagnostic only and flagged in each visit.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
