#!/usr/bin/env python3
"""Recorded from the 2026-09-22 17:40 diagnostic (outputs/validation/openlid_noise_bias_*); offline, not task success."""
import json, sys, numpy as np, torch
from pathlib import Path
ROOT=Path("/home/user/Robotic/project2"); sys.path.insert(0,str(ROOT))
from harness.policy_server import WamPolicy
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, SkillTable, hold_baseline
from harness.frame_cache import FramePack
import argparse
ap=argparse.ArgumentParser(description="Sampling-noise vs bias of WAM open-lid predictions on validation demos (offline diagnostic, not task success)")
ap.add_argument("--checkpoint", default="outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt"); ap.add_argument("--out", required=True)
args=ap.parse_args(); out=ROOT/args.out; out.mkdir(exist_ok=False)
targets=json.loads((ROOT/"data/skill_vtt_task013/contract_targets.json").read_text())
vec=torch.tensor(targets["contracts"]["open lid(car)"]["vector"],dtype=torch.float32)[None]
table=SkillTable(); index=json.loads((LOWDIM/"index.json").read_text())
eps=[ep for ep in sorted(table.episodes) if index["episodes"][str(ep)]["split"]=="validation"][:10]
pol=WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device="cuda", cuda_fraction=0.4, seed=0, log_dir=out, orchestrate={"vlm":"none","frozen":True,"train_targets":"data/skill_vtt_task013/contract_targets.json"})
groups={"trunk":(3,7),"left_arm":(7,14),"right_arm":(15,22)}
res={g:{"bias_mae":[],"noise_mae":[],"hold_mae":[],"demo_intent":[],"mean_intent":[]} for g in groups}
K=8
for ep in eps:
    seg=[s for s in table.episodes[ep]["segments"] if s["skill"]=="open lid"][0]
    act=np.load(LOWDIM/f"episode_{ep}_action.npy",mmap_mode="r"); st=np.load(LOWDIM/f"episode_{ep}_state.npy",mmap_mode="r"); pack=FramePack(FRAMES/HEAD_CAMERA,ep)
    lo,hi=seg["start"],seg["end"]
    for frac in [0.02,0.2,0.4,0.6,0.8]:
        a=int(lo+frac*(hi-lo-33)); a=(a+1)//2*2
        state=np.array(st[a],np.float32); demo=np.array(act[a:a+32],np.float32)
        _,px=pol._preprocess({"r::proprio":state,"r::zed_link::rgb":pack.rgb(a)}); feats,_=pol._encode(px)
        samples=[]
        for k in range(K):
            pol.gen.manual_seed(7919*k+a); samples.append(pol._sample(feats,state,vec))
        S=np.stack(samples); M=S.mean(0); hold=hold_baseline(torch.from_numpy(state)[None])[0].numpy()
        for g,(l,h) in groups.items():
            res[g]["bias_mae"].append(float(np.abs(M[:,l:h]-demo[:,l:h]).mean()))
            res[g]["noise_mae"].append(float(np.abs(S[:,:,l:h]-M[None,:,l:h]).mean()))
            res[g]["hold_mae"].append(float(np.abs(hold[:,l:h]-demo[:,l:h]).mean()))
            res[g]["demo_intent"].append(float(np.linalg.norm(demo[-1,l:h]-demo[0,l:h])))
            res[g]["mean_intent"].append(float(np.linalg.norm(M[-1,l:h]-M[0,l:h])))
summary={g:{k:float(np.mean(v)) for k,v in d.items()} for g,d in res.items()}
summary["note"]="K=8 flow samples per anchor, 10 validation episodes x 5 fractions; bias = |mean sample - demo|, noise = |sample - mean sample|; offline, not task success"
(out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
pol.trace.close()
