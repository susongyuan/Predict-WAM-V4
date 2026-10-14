#!/usr/bin/env python3
"""Analyze actual scored trajectories; never infer a grasp from gripper commands."""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts.supervise_precision_dagger_milestones_v13 import complete_cohort
ORDER=('bowl','spoon','cup','plate')


def progress(rows):
    objects={k:dict(ever_verified_grasp=False,ever_valid_placement=False,
        max_hold_seconds=0.,max_stable_seconds=0.,gripper_contact_frames=0,
        first_verified_grasp_step=None,first_valid_placement_step=None) for k in ORDER}
    total=0;last_step=-1;terminal=None
    for row in rows:
        step=row['control_step'];e.require(step>last_step,'Rule observations must be chronological')
        last_step=step;total+=1;terminal=row
        for key in ORDER:
            item=row['score']['items'][key];value=objects[key]
            value['max_hold_seconds']=max(value['max_hold_seconds'],item['hold_seconds'])
            value['max_stable_seconds']=max(value['max_stable_seconds'],item['stable_seconds'])
            value['gripper_contact_frames']+=int(row['objects'][key]['touching_gripper'])
            for flag,ever,first in [('grasp_awarded','ever_verified_grasp','first_verified_grasp_step'),
                ('placement_valid','ever_valid_placement','first_valid_placement_step')]:
                if item[flag] and not value[ever]:value[ever]=True;value[first]=step
    e.require(terminal is not None,'No scored rule observations')
    if terminal['score']['full_task_success']:stage='full_task_success'
    elif not any(v['ever_verified_grasp'] for v in objects.values()):stage='before_first_verified_grasp'
    else:
        missing=next((k for k in ORDER if not terminal['score']['items'][k]['placement_valid']),None)
        if missing is None:stage='terminal_full_task_rule_not_met'
        elif objects[missing]['ever_valid_placement']:stage=missing+'_lost_valid_placement'
        elif objects[missing]['ever_verified_grasp']:stage=missing+'_placement'
        else:stage=missing+'_grasp'
    return dict(objects=objects,failure_stage=stage,rule_frames=total,last_control_step=last_step)


def episode(row):
    raw_path=Path(row['raw_result_path']);e.require(e.sha256_file(raw_path)==row['raw_result_sha256'],
        'Original episode result changed')
    rules=raw_path.parent/'rule_evidence.jsonl'
    binding=row['evidence_files']['rule_evidence.jsonl']
    e.require(str(rules.resolve())==binding['path'] and e.sha256_file(rules)==binding['sha256'],
        'Scored trajectory differs from admitted evidence')
    with rules.open() as f:value=progress(e.loads(line) for line in f)
    value.update(seed=row['seed'],status=row['status'],full_task_success=row['full_task_success'],
        observed_base_score=row['observed_base_score'],control_steps=row['control_steps'],
        raw_result_path=str(raw_path),raw_result_sha256=row['raw_result_sha256'],
        rule_evidence_sha256=binding['sha256'],braking_steps=row['braking_steps'],
        maximum_command_adjustment_rad=row['max_command_adjustment_rad'])
    if row['status']=='control_safety_abort':value['failure_stage']='controller_safety_abort'
    return value


def analyze(paths,steps=(45000,50000),count=10):
    cohorts={str(step):e.read_json(path) for step,path in paths.items()}
    if count==10:
        valid=all(complete_cohort(cohorts[str(s)],s) for s in steps)
    else:
        seeds=list(e.SEEDS)
        valid=all(cohorts[str(s)].get('status')=='completed'
            and cohorts[str(s)].get('checkpoint',{}).get('step')==s
            and cohorts[str(s)].get('planned_episodes')==5
            and [r['seed'] for r in cohorts[str(s)].get('episodes',[])]==seeds
            and cohorts[str(s)].get('full_successes')==sum(r['full_task_success'] for r in cohorts[str(s)]['episodes'])
            for s in steps) and count==5 and steps==(70000,80000)
    e.require(valid,'Both exact paired test cohorts must finish before analysis')
    result={}
    for step in steps:
        rows=[episode(row) for row in cohorts[str(step)]['episodes']]
        result[str(step)]=dict(full_successes=sum(r['full_task_success'] for r in rows),tests=count,
            scores=[r['observed_base_score'] for r in rows],
            failure_stage_counts=dict(Counter(r['failure_stage'] for r in rows if not r['full_task_success'])),
            per_object={k:dict(verified_grasp_cases=sum(r['objects'][k]['ever_verified_grasp'] for r in rows),
                valid_placement_cases=sum(r['objects'][k]['ever_valid_placement'] for r in rows)) for k in ORDER},episodes=rows)
    paired=[dict(seed=a['seed'],scores={str(steps[0]):a['observed_base_score'],str(steps[1]):b['observed_base_score']},
        successes={str(steps[0]):a['full_task_success'],str(steps[1]):b['full_task_success']},
        stages={str(steps[0]):a['failure_stage'],str(steps[1]):b['failure_stage']})
        for a,b in zip(result[str(steps[0])]['episodes'],result[str(steps[1])]['episodes'])]
    return dict(schema='precision-dagger-paired-milestone-failure-analysis-v14',
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),milestones=result,paired_cases=paired,
        input_summaries={str(s):dict(path=str(Path(p).resolve()),sha256=e.sha256_file(p)) for s,p in paths.items()},
        recovery_required=steps==(45000,50000) and result[str(steps[-1])]['full_successes']==0,
        collection_failure_priorities=result[str(steps[-1])]['failure_stage_counts'],
        interpretation='Privileged physical rules diagnose the first unmet subgoal; these are offline diagnostics, not policy inputs. Training MSE or one-anchor VAL alone cannot establish underfitting or the neural cause.',
        conditional_next_steps=dict(new_domain_randomized_segmented_episodes=50,additional_updates=30000,
            target_step=80000,test_milestones=[70000,80000],tests_each=5,retain_all_existing_data=True)
            if steps==(45000,50000) else None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluation45',required=True)
    parser.add_argument('--evaluation50',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out);e.require(not out.exists(),'Fresh analysis receipt required')
    e.write_json(out,analyze({45000:args.evaluation45,50000:args.evaluation50}))
