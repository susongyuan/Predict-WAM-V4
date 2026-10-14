#!/usr/bin/env python3
"""Read-only integrated experiment diagnostics; GT stays in reporting only."""
import argparse,json,math
from pathlib import Path

def lines(p):
    if not p.exists():return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def diagnose(run):
    run=Path(run);result=run/'result.json'
    if not result.exists():return {'run':str(run),'status':'not_started'}
    r=json.loads(result.read_text());row=r['episodes'][0];d=run/row['episode'];gt=lines(d/'evaluation_gt.jsonl');actions=[a for a in lines(d/'actions.jsonl') if a['status']=='native_returned'];events=lines(d/'integrated_events.jsonl')
    rollout=[g for g in gt if 'settle_step' not in g]
    answer={'run':str(run),'seed':row['seed'],'status':r['status'],'episode_status':row['status'],'steps':row['control_steps'],'native_success':row['evidence']['native_benchmark_success'],'stable_after_settle':row['evidence']['stable_after_settle'],'system':row.get('integrated_system'),'error':row.get('error',r.get('error')),'visual_states':[]}
    if rollout:
        origin=rollout[0]['object_pose_world'][2];delta=[g['object_pose_world'][2]-origin for g in rollout];lift=next((g['control_step'] for g,z in zip(rollout,delta) if z>.007),None);peak=max(delta);fall=next((g['control_step'] for g,z in zip(rollout,delta) if lift is not None and g['control_step']>lift and z<-.05),None)
        answer.update(object_model=rollout[0].get('object_model'),max_lift_m=peak,first_lift_step=lift,fall_below_start_step=fall,max_drawer_open_m=max(max(g['cabinet_joint_qpos']) for g in rollout),contact_steps=sum(g.get('contact_evidence',{}).get('target_robot_contact_count',0)>0 for g in rollout))
        if answer['native_success'] is True:stage='native_success'
        elif answer['native_success'] is None:stage='unscored_or_in_progress'
        elif peak<=.007:stage='no_effective_lift'
        elif fall is not None:stage='lift_then_drop'
        else:stage='lifted_but_no_native_placement'
        answer['failure_stage']=stage
        tracking=[a['max_abs_error'] for g in rollout for a in g.get('native_diagnostics',{}).get('arm_joint_tracking',{}).values() if 'max_abs_error' in a]
        answer['joint_tracking_max_rad']=max(tracking) if tracking else None
        answer['predicate_mismatch_count']=sum(g.get('native_diagnostics',{}).get('native_predicate_evidence',{}).get('recomputed_matches_native') is False for g in rollout)
        answer['fallback_actions']=sum(a.get('topp_fallback_observed',False) for a in actions)
    for event in events:
        if event['kind']=='visual_observation':
            states={k:{'outcome':v.get('outcome'),'parsed':v.get('parsed'),'seconds':v.get('cost',v.get('timing',{}))} for k,v in event['reply']['replies'].items()}
            answer['visual_states'].append({'step':event['step'],'states':states,'strict_supervisor':event.get('strict_supervisor'),'recovery_requested':event['experimental_recovery_requested']})
    return answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();m=json.loads(Path(a.manifest).read_text());rows=[diagnose(j['run_dir']) for j in m['jobs']];valid=[r for r in rows if r.get('status')=='completed' and r.get('native_success') is not None];success=sum(r['native_success'] is True for r in valid)
    result={'planned':len(rows),'finished_valid':len(valid),'native_successes':success,'native_rate_valid':success/len(valid) if valid else None,'all_10_completed':len(valid)==len(rows),'system_verified_successes':None,'results':rows}
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False))
if __name__=='__main__':main()
