#!/usr/bin/env python3
"""Independent CPU evidence audit of the completed middle25k paired5 run.
No simulator, model load, result mutation, or inference. Report only.
"""
from pathlib import Path
import json,hashlib,sqlite3,datetime,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/runs/put_bottles_dustbin/middle25k_5each_20261007'
BASE=ROOT/'outputs/runs/put_bottles_dustbin/pure_25k_30k_5each_20261007'
TRAIN=ROOT/'outputs/runs/put_bottles_dustbin/wam_30k_20261007/train'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def require(ok,note):
 if not ok:raise ValueError(note)
def jsonl(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def native_actions(p):return [r for r in jsonl(p) if r['status']=='native_returned']
def same_bits(a,b):
 x=np.asarray(a,np.float64);y=np.asarray(b,np.float64)
 return x.shape==y.shape and x.tobytes()==y.tobytes()
def main():
 q=read(OUT/'queue_status.json');require(q['status']=='completed','five-test queue incomplete')
 manifest=read(OUT/'manifest.json');require(manifest['seeds']==list(range(920001,920006)),'wrong seeds');require(len(manifest['jobs'])==5,'wrong job count')
 receipt=read(TRAIN/'checkpoint_step_025000.pt.sha256.json');require(sha(TRAIN/'checkpoint_step_025000.pt')==receipt['sha256'],'checkpoint digest')
 baseline=read(BASE/'queue_status.json');baseline_counts=baseline['counts']['25000'];require(baseline_counts['scored_episodes']==5,'baseline incomplete')
 rows=[]
 for job in manifest['jobs']:
  seed=job['seed'];require(job['checkpoint_step']==25000 and job['system_mode']=='middle','wrong weight/mode')
  require((job['max_steps'],job['episode_wall_seconds'],job['total_wall_seconds'],job['external_wall_seconds'],job['settle_steps'])==(1700,5100,5400,5460,0),'different evaluation budget')
  require(job['camera_readback']=='serial' and job['execution_mode']=='native8','different observation/execution mode')
  attempts=[r for r in q['jobs'] if r['seed']==seed];valid=[r for r in attempts if r['status']=='valid_terminal'];require(len(valid)==1,'exactly one valid episode per seed')
  d=Path(valid[0]['run_dir']);res=read(d/'result.json');ep=res['episodes'][0];runtime=res['policy_ready']['runtime'];s=ep['middle_system'];p=d/ep['episode']
  require(res['status']=='completed' and res['task']=='put_bottles_dustbin' and res['scored_episodes']==1,'invalid terminal/task')
  require(runtime['checkpoint_step']==25000 and runtime['checkpoint_sha256']==receipt['sha256'],'wrong runtime checkpoint')
  require(runtime['vtt_adapter'] is None and runtime['vtt_monitor'] is None,'unadmitted VTT')
  require(s['task']=='put_bottles_dustbin' and s['checkpoint_sha256']==receipt['sha256'],'incorrect memory scope')
  require(s['VLM'] is False and s['ASK'] is False and s['with_vtt'] is False,'different middleware config')
  require(s['certified_full_middle_system'] is False and s['task_success_memory_records']==0,'unsupported full-system/task-memory claims')
  require(s['thresholds_changed_for_this_task'] is False,'thresholds were tuned on these tests')
  actions=native_actions(p/'actions.jsonl');steps=ep['control_steps'];require(len(actions)==steps,'action count')
  require(all(a['policy']['task_condition']['source']=='zero_single_task' for a in actions),'nonzero task condition')
  video=read(p/'third_view_video.json');onboard=read(p/'onboard/manifest.json')
  require(video['complete'] is True and video['synthetic'] is False and video['frames']==steps+1 and (p/'third_view.mp4').stat().st_size>0,'video incomplete')
  require(onboard['complete'] and onboard['frames']==steps+1,'onboard incomplete')
  events=jsonl(p/'middle_events.jsonl');feedback=[e for e in events if e['kind']=='robot_feedback'];require(len(feedback)==steps,'actual feedback not present for every action')
  interventions=[e for e in events if e['kind']=='supervisor_hold_clear_reobserve'];require(len(interventions)==s['recovery_attempts']<=2,'bad intervention count')
  require(all(e['clear']['history_preserved'] is True for e in interventions),'recovery reset history')
  c=sqlite3.connect('file:'+str(p/'controller_experience.sqlite')+'?mode=ro',uri=True)
  records=c.execute('SELECT scope,predicates,verifier_kind,outcome FROM evidence').fetchall();c.close()
  require(len(records)==s['controller_memory_records'],'memory record count mismatch')
  for scope,preds,kind,outcome in records:
   require('put_bottles_dustbin' in scope and 'put_object_cabinet' not in scope,'foreign task memory')
   require(set(json.loads(preds))=={'controller_motion_resumed'} and kind=='controller_postconditions','GT/task-success memory used')
  b=BASE/'step_025000'/f'seed_{seed}';be=read(b/'result.json')['episodes'][0];bp=b/be['episode'];ba=native_actions(bp/'actions.jsonl')
  with np.load(p/'onboard/observation_000000.npz',allow_pickle=False) as x,np.load(bp/'onboard/observation_000000.npz',allow_pickle=False) as y:
   initial_equal=set(x.files)==set(y.files) and all(x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes() for k in x.files)
  raw_equal=same_bits([a['raw_action'] for a in actions],[a['raw_action'] for a in ba]);bounded_equal=same_bits([a['bounded_action'] for a in actions],[a['bounded_action'] for a in ba])
  eligible_windows=[e['motion_window'] for e in feedback if e['step']>=160 and e.get('motion_window') is not None]
  stall_analysis={'windows_after_step160':len(eligible_windows),
                  'controller_stalled_windows':sum(w['controller_stalled'] for w in eligible_windows),
                  'minimum_actual_joint_path_rad':min((w['actual_joint_path_rad'] for w in eligible_windows),default=None),
                  'minimum_command_joint_path_rad':min((w['command_joint_path_rad'] for w in eligible_windows),default=None),
                  'threshold_actual_joint_path_rad':0.05,'threshold_command_joint_path_rad':0.1,
                  'scope':'mechanical motion over40 transitions, not object/task progress'}
  rows.append({'stall_analysis':stall_analysis,'seed':seed,'pure_native_success':be['evidence']['native_benchmark_success'],'middle_native_success':ep['evidence']['native_benchmark_success'],'pure_steps':be['control_steps'],'middle_steps':steps,'robot_feedback_samples':len(feedback),'evaluated_motion_windows':sum(e['motion_window'] is not None for e in feedback),'supervisor_interventions':len(interventions),'intervention_steps':[e['step'] for e in interventions],'controller_memory_records':len(records),'initial_observations_bitwise_equal':initial_equal,'raw_actions_bitwise_equal':raw_equal,'bounded_actions_bitwise_equal':bounded_equal,'result_sha256':sha(d/'result.json'),'video':str((p/'third_view.mp4').relative_to(ROOT))})
 counts=q['counts']['25000'];require(counts['scored_episodes']==5,'not five scored episodes');require(sum(r['middle_native_success'] for r in rows)==counts['native_successes'],'aggregate mismatch')
 summary={'status':'verified','created_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'task':'put_bottles_dustbin','checkpoint_step':25000,'checkpoint_sha256':receipt['sha256'],'baseline_successes':baseline_counts['native_successes'],'middle_counts':counts,'rows':rows,'total_measured_feedback':sum(r['robot_feedback_samples'] for r in rows),'total_interventions':sum(r['supervisor_interventions'] for r in rows),'total_controller_memory_records':sum(r['controller_memory_records'] for r in rows),'all_actions_bitwise_equal_to_pure':all(r['raw_actions_bitwise_equal'] and r['bounded_actions_bitwise_equal'] for r in rows),'certified_full_system':False,'notes':['Existing mechanical-motion stall middleware only; no task-semantic grasp/release verifier or matched VTT.','Task score is original native predicate, not strict containment/release/5-second hold.','Zero interventions with equal actions cannot support a middleware performance-gain claim.']}
 (OUT/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
