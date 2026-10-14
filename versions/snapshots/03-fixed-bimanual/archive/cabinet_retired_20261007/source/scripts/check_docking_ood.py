#!/usr/bin/env python3
"""Recorded from the 2026-09-22 17:45 diagnostic (outputs/validation/docking_ood_dev10_20260922); offline, not task success."""
import json, sys, numpy as np, torch
from pathlib import Path
from PIL import Image
ROOT=Path("/home/user/Robotic/project2"); sys.path.insert(0,str(ROOT))
from harness.policy_server import WamPolicy
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, SkillTable
from harness.frame_cache import FramePack
import argparse
ap=argparse.ArgumentParser(description="Nearest-neighbour cosine of closed-loop docking frames to train open-lid start-frame CLS (offline; no simulator state)")
ap.add_argument("--out", required=True); args=ap.parse_args(); out=ROOT/args.out; out.mkdir(exist_ok=False)
pol=WamPolicy("outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt", skill="move to", exec_horizon=8, num_steps=10, device="cuda", cuda_fraction=0.4, seed=0, log_dir=out, orchestrate={"vlm":"none","frozen":True,"train_targets":"data/skill_vtt_task013/contract_targets.json"})
z=np.load(ROOT/"data/skill_vtt_task013/precond_cls_cache.npz"); keys=z["keys"]; cls=z["cls"].astype(np.float64)
# which rows are open lid(car) start frames from TRAIN? cache has train+val? check by recomputing val ones
sel=[i for i,k in enumerate(keys) if str(k)=="open lid(car)"]
print("cached open lid(car) start CLS rows", len(sel))
table=SkillTable(); index=json.loads((LOWDIM/"index.json").read_text())
def enc(rgb):
    _,px=pol._preprocess({"r::proprio":np.zeros(61,np.float32),"r::zed_link::rgb":rgb}); _,c=pol._encode(px); return c.astype(np.float64)
train_cls=[]; val_cls=[]
for ep in sorted(table.episodes):
    seg=[s for s in table.episodes[ep]["segments"] if s["skill"]=="open lid"][0]
    f=seg["start"]; f-=f%2
    c=enc(FramePack(FRAMES/HEAD_CAMERA,ep).rgb(f))
    (train_cls if index["episodes"][str(ep)]["split"]=="train" else val_cls).append(c)
T=np.stack(train_cls); V=np.stack(val_cls)
Tu=T/np.linalg.norm(T,axis=1,keepdims=True)
def nn(c):
    u=c/np.linalg.norm(c); s=Tu@u; return float(s.max()), float(np.sort(s)[-5:].mean())
val_nn=np.array([nn(c) for c in V]); 
# leave-one-out for train
loo=[]
for i,c in enumerate(T):
    u=c/np.linalg.norm(c); s=Tu@u; s[i]=-1; loo.append([s.max(), np.sort(s)[-5:].mean()])
loo=np.array(loo)
runs={"basehold_reapproach":("outputs/behavior/closed_loop/s1_dev10_posonly_best5500_basehold_reapproach_exec32_20260922/observations",["000608","001759"]),
      "720p_wholebody":("outputs/behavior/closed_loop/s1_dev10_posonly_best5500_exec32_720p_20260922/observations",None),
      "contractadapt_exec32":("outputs/behavior/closed_loop/s1_dev10_contractadapt_exec32_20260922/observations",None)}
res={"train_loo_nn_cos":{"mean":float(loo[:,0].mean()),"p05":float(np.percentile(loo[:,0],5)),"min":float(loo[:,0].min())},
     "val_nn_cos":{"mean":float(val_nn[:,0].mean()),"p05":float(np.percentile(val_nn[:,0],5)),"min":float(val_nn[:,0].min()), "n":len(V)},
     "closed_loop":{}}
import glob, os
for name,(d,steps) in runs.items():
    files=sorted(glob.glob(d+"/ep2_step*.png"))
    if steps: files=[f for f in files if any(s in f for s in steps)]
    else: files=files[:12]
    res["closed_loop"][name]={}
    for f in files:
        rgb=np.asarray(Image.open(f).convert("RGB"))
        m,m5=nn(enc(rgb)); res["closed_loop"][name][os.path.basename(f)]={"nn_cos":m,"top5_cos":m5}
print(json.dumps(res,indent=1))
(out/"summary.json").write_text(json.dumps(res,indent=2)); pol.trace.close()
