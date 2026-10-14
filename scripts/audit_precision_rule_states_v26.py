#!/usr/bin/env python3
"""Recompute score arithmetic and instantaneous predicates from archived rules.

The archive has 10Hz observations of a 250Hz judge. This verifies timer-state
consistency at observations, not an exact replay of unarchived substeps.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.zhuhai_task2_rules import LIFE_DESTINATIONS, LIFE_POINTS


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode(row):
    path=Path(row['raw_result_path']).parent/'rule_evidence.jsonl'
    require(sha(path)==row['evidence_files']['rule_evidence.jsonl']['sha256'],'Rule archive changed')
    states=[json.loads(line) for line in path.read_text().splitlines()]
    if row['status']=='physics_safety_abort':states=states[:-1]
    previous=None;transitions={key:dict(grasp_award_steps=[],valid_placement_steps=[],
        timer_threshold_between_archived_observations=[]) for key in LIFE_POINTS}
    for state in states:
        score=state['score'];mapping=LIFE_DESTINATIONS[score['mapping']]
        require(score['track']=='life','Unexpected scoring track')
        expected_total=0;complete=True
        for key,(grasp_points,place_points) in LIFE_POINTS.items():
            obj=state['objects'][key];item=score['items'][key];corners=obj['world_corners_m']
            require(item['destination']==mapping[key],'Destination mapping differs')
            hold=item['hold_seconds'];stable=item['stable_seconds']
            require(math.isfinite(hold) and hold>=0 and math.isfinite(stable) and stable>=0,'Invalid timer')
            lifted=min(v[2] for v in corners)>state['table_surface_world_m']+.01
            held=lifted and not obj['released'] and obj['touching_gripper']
            lower,upper=state['destination_interiors_world_m'][mapping[key]]
            contained=all(all(lo-.001<=x<=hi+.001 for x,lo,hi in zip(corner,lower,upper)) for corner in corners)
            linear=math.sqrt(sum(x*x for x in obj['linear_velocity_m_s']))
            angular=math.sqrt(sum(x*x for x in obj['angular_velocity_rad_s']))
            support=obj['support_destinations']
            valid=(contained and obj['released'] and not obj['touching_robot'] and linear<.015 and angular<.15
                and (support is None or mapping[key] in support)
                and (key!='cup' or obj['upright_cosine']>=math.cos(math.radians(10))))
            require((hold>0)==bool(held and state['physics_time_seconds']>0),'Hold timer/current physical predicate differs')
            require((stable>0)==bool(valid and state['physics_time_seconds']>0),'Stable timer/current physical predicate differs')
            require(item['placement_valid']==bool(valid and stable>=1.),'Placement predicate/timer differs')
            require(hold<5.-1e-9 or item['grasp_awarded'],'>=5s hold has no grasp award')
            expected=int(item['grasp_awarded'])*grasp_points+int(item['placement_valid'])*place_points
            require(item['base_points']==expected,'Object score arithmetic differs')
            expected_total+=expected;complete&=item['grasp_awarded'] and item['placement_valid']
            if previous:
                past=previous['score']['items'][key]
                delta=state['physics_time_seconds']-previous['physics_time_seconds']
                require(hold<=past['hold_seconds']+delta+1e-6 and stable<=past['stable_seconds']+delta+1e-6,
                        'Timer grew faster than actual physics time')
                require(not past['grasp_awarded'] or item['grasp_awarded'],'Grasp latch regressed')
                if not past['grasp_awarded'] and item['grasp_awarded']:
                    transitions[key]['grasp_award_steps'].append(state['control_step'])
                    if hold<5.-1e-9:
                        # A threshold can be crossed then the object released during one .1s interval.
                        require(past['hold_seconds']+delta>=5.-1e-6,'Unexplained grasp award')
                        transitions[key]['timer_threshold_between_archived_observations'].append(state['control_step'])
                if not past['placement_valid'] and item['placement_valid']:
                    transitions[key]['valid_placement_steps'].append(state['control_step'])
            else:
                require(hold==stable==0 and not item['grasp_awarded'] and not item['placement_valid'],'Initial score state differs')
        require(score['base_score']==expected_total and score['base_score_max']==30,'Total score differs')
        require(score['full_task_success']==bool(complete),'Full success does not match strict object predicates')
        previous=state
    require(previous['score']['base_score']==row['observed_base_score'],'Admitted final score differs')
    require(previous['score']['full_task_success']==row['full_task_success'],'Admitted final success differs')
    return dict(seed=row['seed'],frames=len(states),score=row['observed_base_score'],
                full_task_success=row['full_task_success'],object_transitions=transitions,
                score_arithmetic_all_frames_verified=True,
                instantaneous_grasp_and_placement_predicates_verified=True,
                archived_timer_consistency_verified=True,rule_sha256=sha(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',action='append',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();out=Path(args.out);require(not out.exists(),'Preserve existing rule audit')
    cohorts={}
    for value in args.cohort:
        name,p=value.split('=',1);p=Path(p);snapshot=p.read_bytes();s=json.loads(snapshot)
        require(s['status']=='completed' or args.allow_partial,'Cohort incomplete')
        rows=[]
        for row in s['episodes']:
            rows.append(episode(row));print(name,row['seed'],'score/predicates/timer consistency PASS',flush=True)
        cohorts[name]=dict(status_at_snapshot=s['status'],summary_sha256=hashlib.sha256(snapshot).hexdigest(),episodes=rows)
    result=dict(schema='independent-archived-rule-state-consistency-v26',
                time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),cohorts=cohorts,
                scorer_sha256=sha(ROOT/'harness/zhuhai_task2_rules.py'),auditor_sha256=sha(__file__),
                limitation='The frozen judge updates every4ms, while archives record every100ms; unarchived intermediate contacts and exact timer evolution cannot be replayed. Hash/binding audits plus current-state predicates, timer consistency and score arithmetic are independently verified, not substituted for a full substep trace.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
