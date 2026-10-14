#!/usr/bin/env python3
"""CPU posthoc diagnostic of completed frozen tests; never changes policy/protocol."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from harness.robotwin_policy_worker import atomic_json,identity


def main():
    base=ROOT/'outputs/runs/put_bottles_dustbin/pure25k_fixed20_offscreen_repair_20261007'
    out=ROOT/'outputs/verification/dustbin_fixed20_failures_20261007';out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in (921001,921002,921003):
        directory=base/f'seed_{seed}';result=json.loads((directory/'result.json').read_text());ep=result['episodes'][0]
        assert result['status']=='completed' and ep['control_steps']==1700
        path=directory/ep['episode'];actions=[json.loads(l) for l in (path/'actions.jsonl').read_text().splitlines()]
        actions=[r for r in actions if r['status']=='native_returned'];assert len(actions)==1700
        frames=[json.loads(l) for l in (path/'onboard/frames.jsonl').read_text().splitlines()]
        indices=[0,100,200,300,400,500,600,700,800,1000,1200,1400,1500,1600,1690,1700]
        sheet=Image.new('RGB',(1280,4*268),'white');draw=ImageDraw.Draw(sheet)
        for j,index in enumerate(indices):
            with np.load(path/'onboard'/frames[index]['file'],allow_pickle=False) as f:rgb=f['head_camera']
            x=(j%4)*320;y=(j//4)*268;sheet.paste(Image.fromarray(rgb),(x,y+28));draw.text((x+4,y+7),f'seed {seed}, observation {index}',fill='black')
        sheet.save(out/f'seed_{seed}_head.jpg')
        with np.load(path/'onboard'/frames[-1]['file'],allow_pickle=False) as f:
            final=Image.fromarray(np.hstack([f[v] for v in ('head_camera','left_camera','right_camera')]))
        final.save(out/f'seed_{seed}_final_three.jpg')
        policy=np.array([a['policy']['policy_seconds'] for a in actions]);control=np.array([a['control_seconds'] for a in actions])
        interval=frames[-1]['wall_monotonic']-frames[0]['wall_monotonic']
        rows.append(dict(seed=seed,task=result['task'],checkpoint=result['policy_ready']['runtime']['checkpoint_sha256'],
            system_mode='pure',control_steps=len(actions),native_success=ep['evidence']['native_benchmark_success'],
            wall_seconds=ep['wall_seconds'],observed_rollout_wall_seconds=interval,
            policy_seconds_sum=float(policy.sum()),policy_seconds_mean=float(policy.mean()),
            control_seconds_sum=float(control.sum()),native_execution_and_control_overhead_seconds=float((control-policy).sum()),
            remaining_rollout_RGB_recording_video_and_loop_seconds=float(interval-control.sum()),
            topp_fallback_steps=[a['step'] for a in actions if a.get('topp_fallback_observed')],
            joint_limit_clipped_actions=sum(any(a['clip_mask'][i] for i in (0,1,2,3,4,5,7,8,9,10,11,12)) for a in actions),
            evidence=identity(directory/'result.json'),actions_evidence=identity(path/'actions.jsonl'),
            timing_limit='control_seconds covers parent-to-worker action request/log and native execution; policy_seconds is worker inference. Residual includes get_obs, archive compression, video, guards and loop overhead; not a separately instrumented renderer profile. No physical-time claim.'))
    atomic_json(out/'audit.json',dict(scope='Posthoc diagnosis only; no tuning or new evaluation',rows=rows,
        object_contact_truth_available=False,physical_failure_cause_verified=False))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
