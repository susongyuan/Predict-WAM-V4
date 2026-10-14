#!/usr/bin/env python3
"""CPU-only paired failure audit. Original results/weights are read-only.
Motion is commanded state, not measured qpos; images cannot establish contact,
release or containment. Thresholds label command events, not physical grasps.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
from pathlib import Path
import argparse,hashlib,json,datetime
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/runs/put_bottles_dustbin/pure_25k_30k_5each_20261007'
VIEWS=('head_camera','left_camera','right_camera')
ARM_INDICES=[0,1,2,3,4,5,7,8,9,10,11,12]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def command_events(values):
    state='open' if values[0]>=.8 else 'closed' if values[0]<=.2 else 'intermediate'
    events=[]
    for i,v in enumerate(values):
        new='open' if v>=.8 else 'closed' if v<=.2 else state
        if new!=state:
            events.append({'action_index':i,'command_state':new,'value':float(v)})
            state=new
    return events

def audit_run(step,seed,out):
    d=RUN/f'step_{step:06d}'/f'seed_{seed}';result=read(d/'result.json');ep=result['episodes'][0];p=d/ep['episode']
    assert result['status']=='completed' and ep['status'] in ('native_success','step_limit')
    actions=[json.loads(line) for line in (p/'actions.jsonl').read_text().splitlines()]
    actions=[x for x in actions if x['status']=='native_returned']
    assert len(actions)==ep['control_steps'] and [a['step'] for a in actions]==list(range(len(actions)))
    raw=np.array([x['raw_action'] for x in actions]);bounded=np.array([x['bounded_action'] for x in actions]);mask=np.array([x['clip_mask'] for x in actions]);assert mask.shape==bounded.shape==(len(actions),14)
    frames=[json.loads(l) for l in (p/'onboard/frames.jsonl').read_text().splitlines()];assert len(frames)==len(actions)+1
    state=[]
    for frame in frames:
        with np.load(p/'onboard'/frame['file'],allow_pickle=False) as f:state.append(f['commanded_joint_gripper_state'])
    state=np.stack(state);delta=np.abs(np.diff(state,axis=0))
    indices=sorted(set(min(t,len(actions)) for t in [0,100,200,300,400,500,650,800,1000,1200,1400,1700]))
    tiles=[]
    for idx in indices:
        with np.load(p/'onboard'/frames[idx]['file'],allow_pickle=False) as f:
            rgb=f['head_camera'];bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            tile=cv2.copyMakeBorder(bgr,28,0,0,0,cv2.BORDER_CONSTANT,value=(248,248,248))
            cv2.putText(tile,f'obs {idx}: grip L={state[idx,6]:.2f} R={state[idx,13]:.2f}',(4,19),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
            tiles.append(tile)
    while len(tiles)%4:tiles.append(np.full_like(tiles[0],248))
    canvas=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
    image=out/f'ckpt{step}_seed{seed}_head.jpg';cv2.imwrite(str(image),canvas)
    image_final=out/f'ckpt{step}_seed{seed}_final_three.jpg'
    with np.load(p/'onboard'/frames[-1]['file'],allow_pickle=False) as f:
        cv2.imwrite(str(image_final),np.hstack([cv2.cvtColor(f[v],cv2.COLOR_RGB2BGR) for v in VIEWS]))
    groups={}
    for name,ix in [('arm_joints',ARM_INDICES),('grippers',[6,13])]:
        error=np.abs(raw[:,ix]-bounded[:,ix]);nonzero=error[error>0]
        groups[name]={'actions_clipped':int(np.any(mask[:,ix],axis=1).sum()),'fraction_actions':float(np.any(mask[:,ix],axis=1).mean()),'max_clip':float(error.max()),'p95_nonzero_clip':float(np.percentile(nonzero,95)) if len(nonzero) else 0.0}
    temporal=[]
    for lo,hi in [(0,min(500,len(actions))),(max(0,len(actions)-400),len(actions))]:
        temporal.append({'action_range':[lo,hi],'mean_abs_commanded_joint_delta_rad':float(delta[lo:hi,ARM_INDICES].mean()),'p95_max_joint_delta_rad':float(np.percentile(delta[lo:hi,ARM_INDICES].max(axis=1),95)), 'commanded_joint_path_L1_rad':float(delta[lo:hi,ARM_INDICES].sum()),'commanded_gripper_min':state[lo:hi+1,[6,13]].min(axis=0).tolist(),'commanded_gripper_max':state[lo:hi+1,[6,13]].max(axis=0).tolist()})
    return {'checkpoint_step':step,'seed':seed,'native_success':ep['evidence']['native_benchmark_success'],'control_steps':len(actions),'status':ep['status'],'source_result':str((d/'result.json').relative_to(ROOT)),'source_result_sha256':sha(d/'result.json'),'action_log_sha256':sha(p/'actions.jsonl'),'initial_observation_sha256':sha(p/'onboard'/frames[0]['file']),'clip':groups,'command_events':{'left':command_events(bounded[:,6]),'right':command_events(bounded[:,13])},'motion_windows':temporal,'topp_fallback_action_indices':[a['step'] for a in actions if a.get('topp_fallback_observed')], 'zero_condition_verified':all(a['policy']['task_condition']['source']=='zero_single_task' for a in actions),'head_contact_sheet':str(image.relative_to(ROOT)),'final_three_views':str(image_final.relative_to(ROOT))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/verification/dustbin_failure_review_20261007');a=p.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    rows=[audit_run(step,seed,out) for seed in (920003,920004,920005) for step in (25000,30000)]
    paired=[]
    for seed in (920003,920004,920005):
        a,b=[r for r in rows if r['seed']==seed]
        paired.append({'seed':seed,'initial_npz_byte_identical':a['initial_observation_sha256']==b['initial_observation_sha256']})
    result={'status':'quantitative_complete_visual_review_pending','created_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'posthoc failure diagnostic of three discordant seed pairs; not new evaluation or checkpoint selection','rows':rows,'paired':paired,'limitations':['No per-bottle GT/contact/grasp traces were recorded for this generic task.','Command events and joint paths are commanded, not measured qpos or physical grasp.','Identical initial files test visual/state starting equivalence only, not universal reproducibility.','Original success rates and weights are not modified; no GPU/simulator/model is run.']}
    (out/'quantitative_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'out':str(out),'pairs':paired,'run_count':len(rows)},indent=2))
if __name__=='__main__':main()
