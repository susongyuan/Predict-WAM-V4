#!/usr/bin/env python3
"""CPU-only, fail-closed final evidence reconciliation; never changes policy/tests."""
from pathlib import Path
import csv,hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'outputs/runs/put_bottles_dustbin/pure_25k_30k_5each_20261007'
T=ROOT/'outputs/runs/put_bottles_dustbin/wam_30k_20261007/train'
O=ROOT/'outputs/pipelines/dustbin_20261007'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 q=read(R/'queue_status.json');assert q['status']=='completed';train=read(T/'status.json');assert train['step']==train['target']==30000 and train['status']=='bounded_complete'
 receipts={}
 for step in (25000,30000):
  p=T/f'checkpoint_step_{step:06d}.pt';rec=read(str(p)+'.sha256.json');assert rec['step']==step and rec['model_only'] and sha(p)==rec['sha256'];receipts[str(step)]=rec
 rows=[]
 for step in (25000,30000):
  c=q['counts'][str(step)];assert c['scored_episodes']==5 and c['attempted_episodes']==5 and not c['incomplete_attempts'] and not c['invalid_initializations']
  for seed in range(920001,920006):
   d=R/f'step_{step:06d}'/f'seed_{seed}';res=read(d/'result.json');assert res['status']=='completed' and res['scored_episodes']==1 and res['task']=='put_bottles_dustbin';ep=res['episodes'][0];success=ep['evidence']['native_benchmark_success'];assert type(success)is bool and res['native_successes']==int(success);assert ep['seed']==seed and ep['status']==('native_success' if success else 'step_limit');assert ep['evidence']['native_terminal_predicate']==success
   runtime=res['policy_ready']['runtime'];assert runtime['checkpoint_step']==step and runtime['checkpoint_sha256']==receipts[str(step)]['sha256'] and runtime['vtt_adapter'] is None and runtime['vtt_monitor'] is None
   sub=d/ep['episode'];vid=read(sub/'third_view_video.json');assert vid['complete'] and vid['synthetic'] is False and vid['frames']==ep['control_steps']+1 and (sub/'third_view.mp4').stat().st_size>0
   actions=[json.loads(x) for x in (sub/'actions.jsonl').read_text().splitlines()];actions=[x for x in actions if x['status']=='native_returned'];assert len(actions)==ep['control_steps'];assert all(x['policy']['task_condition']['source']=='zero_single_task' for x in actions);assert all(x['policy']['execution_prefix']==8 for x in actions)
   rows.append({'checkpoint_step':step,'seed':seed,'native_success':success,'control_steps':ep['control_steps'],'wall_seconds':ep['wall_seconds'],'video_frames':vid['frames'],'policy_condition':'zero_single_task','actuator_clipped_actions':sum(any(x['clip_mask']) for x in actions),'native_topp_fallback_actions':sum(x.get('topp_fallback_observed',False) for x in actions),'video':str((sub/'third_view.mp4').relative_to(ROOT)),'result_sha256':sha(d/'result.json')})
  assert sum(x['native_success'] for x in rows if x['checkpoint_step']==step)==c['native_successes']
 report={'status':'verified','verified_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'task':'put_bottles_dustbin','training':train,'checkpoints':receipts,'counts':q['counts'],'rows':rows,'claims':'Native task success only, rigid bottles and open trash bin. No lid-opening/deformable/strict containment or broad generalization claim. No score-selected retries. Actuator clipping and original TOPP hold fallback are not VLM/expert/recovery.'}
 (O/'final_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 with (O/'per_episode_results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'status':report['status'],'counts':q['counts'],'episodes':len(rows)},indent=2))
if __name__=='__main__':main()
