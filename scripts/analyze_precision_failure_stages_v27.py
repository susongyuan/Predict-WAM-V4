#!/usr/bin/env python3
"""Separate archived no-close, empty-close, contact/no-lift and short-hold evidence.

Offline privileged observations diagnose failures, never enter online control.
All contact/lift counts are observations at10Hz, not unarchived substep counts.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def sha(data):return hashlib.sha256(data).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def intervals(values):
    result=[];start=None
    for i,flag in enumerate([*values,False]):
        if flag and start is None:start=i
        if not flag and start is not None:result.append([start,i-1]);start=None
    return result


def episode(row,motion):
    directory=Path(row['raw_result_path']).parent
    rulebytes=(directory/'rule_evidence.jsonl').read_bytes()
    actionbytes=(directory/'actions.jsonl').read_bytes()
    require(sha(rulebytes)==row['evidence_files']['rule_evidence.jsonl']['sha256'],'Rule hash differs')
    require(sha(actionbytes)==row['evidence_files']['actions.jsonl']['sha256'],'Action hash differs')
    rules=[json.loads(l) for l in rulebytes.splitlines()]
    if row['status']=='physics_safety_abort':rules=rules[:-1]
    actions=[json.loads(l) for l in actionbytes.splitlines()]
    require(motion['seed']==row['seed'],'Motion case differs')
    require(motion['command_diagnostic']['physical_progress']['raw_result_sha256']==row['raw_result_sha256'],
            'Motion analysis references a different result')
    objects={}
    for key in ('bowl','spoon','cup','plate'):
        contact=[r['objects'][key]['touching_gripper'] for r in rules]
        bottom=[min(p[2] for p in r['objects'][key]['world_corners_m']) for r in rules]
        lifted=[z>r['table_surface_world_m']+.01 for z,r in zip(bottom,rules)]
        held=[c and l and not r['objects'][key]['released'] for c,l,r in zip(contact,lifted,rules)]
        scores=[r['score']['items'][key] for r in rules]
        evergrasp=any(s['grasp_awarded'] for s in scores);everplace=any(s['placement_valid'] for s in scores)
        objects[key]=dict(gripper_contact_observations=sum(contact),contact_observation_intervals=intervals(contact),
            contact_without_lift_observations=sum(c and not l for c,l in zip(contact,lifted)),
            lifted_contact_observations=sum(c and l for c,l in zip(contact,lifted)),
            strict_held_observations=sum(held),strict_held_observation_intervals=intervals(held),
            first_contact_observation=next((i for i,c in enumerate(contact) if c),None),
            first_strict_held_observation=next((i for i,c in enumerate(held) if c),None),
            contact_bottom_z_m_range=[min(z for z,c in zip(bottom,contact) if c),max(z for z,c in zip(bottom,contact) if c)] if any(contact) else None,
            table_height_m=rules[0]['table_surface_world_m'],max_bottom_height_m=max(bottom),
            max_hold_seconds=max(s['hold_seconds'] for s in scores),
            max_stable_seconds=max(s['stable_seconds'] for s in scores),
            first_verified_grasp_observation=next((i for i,s in enumerate(scores) if s['grasp_awarded']),None),
            first_valid_placement_observation=next((i for i,s in enumerate(scores) if s['placement_valid']),None),
            ever_verified_grasp=evergrasp,ever_valid_placement=everplace,
            final_valid_placement=scores[-1]['placement_valid'])
    rule_by_step={r['control_step']:r for r in rules}
    close_contacts=[]
    for event in motion['close_events']:
        start,end=event['first_close_action_step'],event['last_close_action_step']
        contacts={key:sum(rule_by_step[t]['objects'][key]['touching_gripper']
            for t in range(start+1,end+2) if t in rule_by_step) for key in objects}
        close_contacts.append(dict(first_close_action_step=start,last_close_action_step=end,
            any_gripper_contact_observations_by_object=contacts,
            no_contact_with_any_task_object=not any(contacts.values())))
    verified_objects=[key for key,v in objects.items() if v['ever_verified_grasp']]
    placed_objects=[key for key,v in objects.items() if v['final_valid_placement']]
    tags=[];bowl=objects['bowl'];gripper=motion['command_diagnostic']['grippers']['left']
    closed=gripper['dispatched']['closed_frames']
    if row['status']=='physics_safety_abort':tags.append('actual_physics_collision_stop')
    if row['status']=='control_safety_abort':tags.append('command_safety_stop')
    if not bowl['ever_verified_grasp']:
        if not closed:
            tags.append('left_command_never_reached_closed_le_0_2')
            if gripper['planned_close_timing']['no_close_in_32']==gripper['planned_close_timing']['replans']:
                tags.append('no_closed_target_anywhere_in_all_32step_plans')
            else:tags.append('closed_targets_planned_only_in_unexecuted_tail')
        if motion['close_events'] and not any(e['bowl_contact_frames_during_close'] for e in motion['close_events']):
            tags.append('executed_close_with_no_bowl_contact_in_closed_intervals')
        if close_contacts and all(e['no_contact_with_any_task_object'] for e in close_contacts):
            tags.append('all_left_closed_intervals_have_no_contact_with_any_task_object')
        if any(key!='bowl' for key in verified_objects):
            tags.append('other_object_verified_grasp_while_bowl_uncompleted')
        if any(key!='bowl' for key in placed_objects):
            tags.append('other_object_validly_placed_while_bowl_uncompleted')
        if bowl['gripper_contact_observations'] and not bowl['strict_held_observations']:
            tags.append('contact_without_establishing_lifted_hold')
        if bowl['strict_held_observations'] and bowl['max_hold_seconds']<5.-1e-9:
            tags.append('lifted_hold_broken_before_strict_5seconds')
    elif not bowl['ever_valid_placement']:tags.append('verified_bowl_hold_without_valid_placement')
    elif not bowl['final_valid_placement']:tags.append('bowl_valid_placement_later_lost')
    elif not objects['spoon']['ever_verified_grasp']:tags.append('bowl_complete_next_spoon_grasp_not_established')
    physical=motion['command_diagnostic']['physical_progress']
    remainder=None;place=bowl['first_valid_placement_observation']
    if place is not None:
        following=actions[place:];following_rules=rules[place:]
        remainder=dict(first_bowl_placement_observation=place,
            first_bowl_placement_physics_seconds=rules[place]['physics_time_seconds'],
            remaining_dispatched_control_steps=len(following),
            remaining_physics_seconds=rules[-1]['physics_time_seconds']-rules[place]['physics_time_seconds'],
            replans_after_bowl_placement=sum(a['policy']['replanned'] for a in following),
            left_closed_dispatched_commands_after_bowl_placement=sum(a['dispatched_action'][7]<=.2 for a in following),
            right_closed_dispatched_commands_after_bowl_placement=sum(a['dispatched_action'][15]<=.2 for a in following),
            object_gripper_contact_observations_after_bowl_placement={key:sum(r['objects'][key]['touching_gripper']
                for r in following_rules) for key in objects},
            validly_placed_objects_at_first_bowl_placement=[key for key,s in rules[place]['score']['items'].items()
                if s['placement_valid']],
            object_first_verified_grasp_observation_at_or_after_bowl_placement={key:
                obj['first_verified_grasp_observation'] if obj['first_verified_grasp_observation'] is not None
                    and obj['first_verified_grasp_observation']>=place else None
                for key,obj in objects.items()})
    first=motion['close_events'][0] if motion['close_events'] else None
    closure=None
    if first:
        geo=first['after_action_observation'];point=geo['left_tool_world_m'];lo,hi=geo['bowl_AABB_world_m']
        axis=[max(lo[i]-point[i],point[i]-hi[i],0) for i in range(3)]
        object_geometry={}
        for key,obj in rule_by_step[geo['observation_control_step']]['objects'].items():
            corners=obj['world_corners_m'];lower=[min(p[i] for p in corners) for i in range(3)]
            upper=[max(p[i] for p in corners) for i in range(3)]
            outside=[max(lower[i]-point[i],point[i]-upper[i],0) for i in range(3)]
            object_geometry[key]=dict(tool_to_AABB_lower_bound_m=sum(v*v for v in outside)**.5,
                outside_AABB_axis_distance_m=outside,
                any_gripper_contact_at_first_close_observation=obj['touching_gripper'])
        closure=dict(action_step=first['first_close_action_step'],observation_step=geo['observation_control_step'],
            tool_to_bowl_AABB_m=geo['tool_to_bowl_AABB_lower_bound_m'],outside_AABB_axis_distance_m=axis,
            tool_above_bowl_AABB_top_m=max(point[2]-hi[2],0),
            bowl_contact_observations_in_this_close_interval=first['bowl_contact_frames_during_close'],
            task_object_geometry=object_geometry,
            limitation='Tool reference point, not finger surface or grasp-pose error.')
    return dict(seed=row['seed'],control_steps=row['control_steps'],status=row['status'],
        score=row['observed_base_score'],full_task_success=row['full_task_success'],
        first_uncompleted_stage=physical.get('first_unmet_subgoal',physical['failure_stage']),
        failure_stage=physical['failure_stage'],evidence_tags=tags,objects=objects,
        verified_grasp_objects=verified_objects,final_valid_placement_objects=placed_objects,
        bowl_completed_remainder=remainder,
        left_closed_interval_contacts=close_contacts,
        left_closed_command_observations=closed,first_left_close_geometry=closure,
        gripper_smoothing_and_dispatch_preserve_closed_events=gripper['close_threshold_preserved_every_step'],
        planning_closed_target_statistics=gripper['planned_close_timing'],
        arm_motion=motion['motion'],safety_events=row.get('safety_events',[]),
        source_result_sha256=row['raw_result_sha256'],rule_sha256=sha(rulebytes),actions_sha256=sha(actionbytes))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True)
    parser.add_argument('--motion',required=True)
    parser.add_argument('--motion-cohort',default='80000extra20')
    parser.add_argument('--out',required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();out=Path(args.out);require(not out.exists(),'Preserve existing failure receipt')
    snapshot=Path(args.summary).read_bytes();summary=json.loads(snapshot)
    require(summary['status']=='completed' or args.allow_partial,'Cohort incomplete')
    motion=json.loads(Path(args.motion).read_text())['cohorts'][args.motion_cohort]
    motionrows={r['seed']:r for r in motion['episodes']}
    summaryseeds={r['seed'] for r in summary['episodes']}
    require(set(motionrows)<=summaryseeds if args.allow_partial else set(motionrows)==summaryseeds,
            'Motion snapshot and summary cases differ')
    selected=[r for r in summary['episodes'] if r['seed'] in motionrows]
    rows=[episode(r,motionrows[r['seed']]) for r in selected]
    value=dict(schema='precision-physical-failure-substages-v27',
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        status_at_snapshot=summary['status'],summary_completed_count_at_snapshot=len(summary['episodes']),
        selected_seeds=[r['seed'] for r in rows],tests=len(rows),full_successes=sum(r['full_task_success'] for r in rows),
        scores=[r['score'] for r in rows],stage_counts=dict(Counter(r['failure_stage'] for r in rows)),
        overlapping_evidence_tag_counts=dict(Counter(t for r in rows for t in r['evidence_tags'])),episodes=rows,
        summary_snapshot_sha256=sha(snapshot),motion_analysis_sha256=sha(Path(args.motion).read_bytes()),
        analyzer_sha256=sha(Path(__file__).read_bytes()),
        interpretation='Evidence tags overlap: a case can touch an object without lifting, then close in empty space later. Contact and tool-AABB proximity never establish a verified5s grasp. Counts are100ms observations; no causal under/overfitting, model-capacity or architecture claim.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:value[k] for k in ('tests','scores','stage_counts','overlapping_evidence_tag_counts')},ensure_ascii=False))


if __name__=='__main__':main()
