#!/usr/bin/env python3
"""Small TRAIN-only frozen-DINO foreground localization pilot; CPU only.
Eight independently annotated first frames fit a linear patch head; next eight
are a fixed check. No WAM/encoder changes. Never a physical success verifier.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,hashlib,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from scripts.probe_robotwin_target_binding import iou
from harness.robotwin_eval_evidence import ResourceGuard

def feature(path):
    a=np.load(path,mmap_mode='r');bits=np.array(a[0,0,-1,5:],copy=True);f=torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).float()
    if f.shape!=(196,1024):raise ValueError('DINO_patch_shape_mismatch')
    f=torch.nn.functional.normalize(f,dim=-1)
    yy,xx=torch.meshgrid(torch.linspace(1/28,27/28,14),torch.linspace(1/28,27/28,14),indexing='ij');x=xx.flatten();y=yy.flatten();pos=torch.stack([x,y,x*x,y*y,x*y],-1)
    return torch.cat([f,pos],-1)

def labels(box):
    b=[v/1000*14 for v in box];out=[]
    for y in range(14):
        for x in range(14):out.append(float(max(0,min(x+1,b[2])-max(x,b[0]))*max(0,min(y+1,b[3])-max(y,b[1]))>=.15))
    if sum(out)==0:raise ValueError('no_visible_positive_patch')
    return torch.tensor(out)[:,None]

def detect(prob):
    mask=(prob.reshape(14,14)>=.7).astype(np.uint8);n,ids,stats,cent=cv2.connectedComponentsWithStats(mask,8);candidates=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if not 1<=area<=15:continue
        score=float(prob.reshape(14,14)[ids==i].mean());candidates.append((score,[float(x)/14*1000,float(y)/14*1000,float(x+w)/14*1000,float(y+h)/14*1000]))
    candidates.sort(reverse=True,key=lambda x:x[0])
    if not candidates:return None,'no_candidate'
    if len(candidates)>1 and candidates[1][0]>=.9*candidates[0][0]:return None,'ambiguous_candidates'
    return candidates[0][1],'single_visible_candidate'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(2);torch.manual_seed(20261006);guard=ResourceGuard();guard.check()
    base=ROOT/'outputs/validation/target_binding_train_20261006';gold=json.loads((base/'annotations.json').read_text());xs=[];ys=[];inputs=[]
    for g in gold['annotations']:
        paths=list((ROOT/'data/robotwin_features').glob('*/'+g['episode'].removesuffix('.hdf5')+'-*/feat.npy'))
        if len(paths)!=1:raise ValueError('ambiguous_or_missing_frozen_feature_cache')
        meta=json.loads((paths[0].parent/'meta.json').read_text());xs.append(feature(paths[0]));ys.append(labels(g['head_bbox_1000']));inputs.append({'episode':g['episode'],'feature_file':str(paths[0]),'cache_meta':meta})
    protocol={'registered_before_fit':True,'model':'1029_to_1_linear_patch_head_on_frozen_DINO_layer_minus4','fit_cases':8,'fixed_check_cases':8,'steps':1000,'seed':20261006,'learning_rate':.02,'weight_decay':.01,'threshold':.7,'connected_component_pixels':[1,15],'ambiguity_second_to_first_ratio':.9,'task_scope':'initial_foreground_visible_target_only','no_WAM_or_encoder_update':True,'TRAIN_only':True,'admission_criterion':'pilot only; at least 7 of 8 fixed check IoU >=0.3 required before further validation; not sufficient for physical feedback','annotation_sha256':hashlib.sha256((base/'annotations.json').read_bytes()).hexdigest(),'inputs':inputs}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');X=torch.cat(xs[:8]);Y=torch.cat(ys[:8]);pos=(len(Y)-Y.sum())/Y.sum();model=torch.nn.Linear(1029,1);opt=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.01);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=pos);start=time.monotonic()
    for step in range(1000):
        guard.check();opt.zero_grad();loss=lossfn(model(X),Y);loss.backward();opt.step()
        if step%100==0:print('UPDATE',step,'LOSS',float(loss),flush=True)
    torch.save({'format':'initial_visible_target_patch_head_v1','model_state_dict':model.state_dict(),'protocol':protocol,'fit_updates':1000},out/'model.pt')
    records=[]
    with torch.inference_mode():
        for i,(x,g) in enumerate(zip(xs,gold['annotations'])):
            prob=model(x).sigmoid().numpy().flatten();box,reason=detect(prob);score=iou(box,g['head_bbox_1000']) if box else 0;records.append({'episode':g['episode'],'partition':'fit' if i<8 else 'fixed_check','bbox':box,'iou':score,'passed':score>=.3,'reason':reason})
            img=Image.open(base/g['episode'].removesuffix('.hdf5')/'head.png').convert('RGB');draw=ImageDraw.Draw(img);draw.rectangle([v/1000*(320 if j%2==0 else 240) for j,v in enumerate(g['head_bbox_1000'])],outline='lime',width=2)
            if box:draw.rectangle([v/1000*(320 if j%2==0 else 240) for j,v in enumerate(box)],outline='red',width=2)
            img.save(out/(g['episode'].removesuffix('.hdf5')+'.png'))
    result={'completed':True,'optimizer_updates':1000,'fit_passes':sum(r['passed'] for r in records[:8]),'fixed_check_passes':sum(r['passed'] for r in records[8:]),'records':records,'elapsed_seconds':time.monotonic()-start,'peak_resources':guard.peaks,'CUDA_initialized':torch.cuda.is_initialized(),'admitted':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',result['fit_passes'],result['fixed_check_passes'],flush=True)
if __name__=='__main__':main()
