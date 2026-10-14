#!/usr/bin/env python3
"""Advisory initial-RGB three-bottle pilot; never a physical verifier or admitted tracker."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from harness.robotwin_policy_worker import atomic_json,sha256
from harness.robotwin_training_data import load_manifest,resolve_cache,episode_identity,episode_directory,validate_payload

# Draft pixel boxes from direct inspection of sixteen TRAIN head RGB frames.
# These are approximate assistant annotations, not independent reviewed ground truth.
BOXES={
0:[[47,18,94,122],[184,0,220,64],[242,0,295,94]],
1:[[236,0,276,68],[108,0,149,83],[258,8,319,119]],
2:[[103,0,142,95],[205,7,254,106],[282,0,320,75]],
3:[[200,0,239,84],[76,23,121,119],[270,0,320,57]],
4:[[50,0,96,75],[221,0,275,89],[95,10,137,119]],
6:[[208,0,254,84],[252,0,299,61],[288,36,320,110]],
7:[[196,0,237,75],[55,0,105,85],[268,0,320,75]],
9:[[247,6,302,106],[109,0,145,83],[34,11,88,123]],
10:[[214,0,253,64],[24,19,83,116],[276,0,320,88]],
11:[[68,0,110,73],[247,0,299,73],[184,1,231,111]],
12:[[190,0,223,60],[253,0,308,69],[46,0,94,100]],
15:[[118,0,157,62],[79,14,127,124],[187,0,229,87]],
16:[[122,0,156,96],[204,0,247,66],[51,0,98,75]],
17:[[92,17,134,120],[292,37,320,108],[242,0,289,58]],
18:[[93,0,132,60],[228,0,260,61],[220,10,271,120]],
19:[[226,7,275,121],[84,0,123,60],[66,0,116,104]],
}


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261007)
    base=ROOT/'outputs/construction/dustbin_full_perception_train_20261007'
    out=ROOT/'outputs/construction/dustbin_initial_patch_pilot_20261007';out.mkdir(parents=True,exist_ok=False)
    source=json.loads((base/'protocol.json').read_text())
    split_path=ROOT/'data/benchmarks/robotwin2/put_bottles_dustbin_50_v1/split_manifest.json'
    split=load_manifest(split_path);cache,cid=resolve_cache(split,ROOT/'data/dustbin_dino_features')
    names=source['fit_episodes'];X=[];Y=[];annotations=[]
    yy,xx=torch.meshgrid(torch.linspace(1/28,27/28,14),torch.linspace(1/28,27/28,14),indexing='ij')
    pos=torch.stack([xx.flatten(),yy.flatten(),xx.flatten()**2,yy.flatten()**2,(xx*yy).flatten()],-1)
    for name in names:
        identity=episode_identity(Path(split['data_root'])/name,cid);validate_payload(cache,identity,load_features=False)
        features=np.load(episode_directory(cache,identity)/'feat.npy',mmap_mode='r',allow_pickle=False)
        bits=np.array(features[0,0,-1,5:],copy=True).view(np.int16)
        feat=torch.from_numpy(bits).view(torch.bfloat16).float();feat=torch.nn.functional.normalize(feat,dim=-1)
        X.append(torch.cat([feat,pos],-1));boxes=BOXES[int(Path(name).stem.split('_')[1])]
        target=np.zeros((14,14,4),np.float32)
        for y in range(14):
            for x in range(14):
                covers=[]
                for b in boxes:
                    a=[b[0]/320*14,b[1]/240*14,b[2]/320*14,b[3]/240*14]
                    covers.append(max(0,min(x+1,a[2])-max(x,a[0]))*max(0,min(y+1,a[3])-max(y,a[1])))
                row=[max(0,1-sum(covers)),*covers];target[y,x]=np.asarray(row)/max(sum(row),1e-8)
        Y.append(torch.from_numpy(target.reshape(196,4)))
        image_path=base/Path(name).stem/'cam_head_frame000000.png'
        annotations.append(dict(episode=name,frame=0,view='cam_head',bbox_xyxy_pixels=boxes,
            image_sha256=sha256(image_path),annotation_source='offline assistant direct pixel inspection',
            quality='draft approximate boxes; require independent review before admission'))
        img=Image.open(image_path);draw=ImageDraw.Draw(img)
        for color,box in zip(['blue','orange','red'],boxes):draw.rectangle(box,outline=color,width=2)
        img.save(out/(Path(name).stem+'_draft_labels.png'))
    atomic_json(out/'draft_fit_annotations.json',dict(classes=source['semantic_classes'][:3],annotations=annotations,reviewed=False))
    protocol=dict(format='dustbin-initial-multiclass-patch-pilot-v1',task='put_bottles_dustbin',
        model='Linear 1029->4 soft patch classification; frozen DINO[-4] head patches plus xy polynomial',
        fit_episodes=names,validation_episodes=[],reserved_check_not_used=True,heldout_used=False,
        source_split='train',split_sha256=sha256(split_path),cache_id=cid,
        steps=1000,seed=20261007,lr=.02,weight_decay=.01,
        annotation_sha256=sha256(out/'draft_fit_annotations.json'),
        scope='frame0 visible head-RGB bottle proposals only; not temporal tracking/hand binding',
        physical_verifier=False,provider_admitted=False,reviewed_annotations=False)
    atomic_json(out/'protocol.json',protocol)
    x=torch.cat(X);y=torch.cat(Y);model=torch.nn.Linear(1029,4);opt=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.01)
    weights=1/torch.clamp(y.sum(0),min=1);weights=weights/weights.mean();start=time.monotonic()
    for step in range(1000):
        opt.zero_grad();logp=model(x).log_softmax(-1);loss=-(y*logp*weights).sum(-1).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite patch pilot')
        loss.backward();opt.step()
    torch.save(dict(format=protocol['format'],state_dict=model.state_dict(),protocol=protocol),out/'model.pt')
    with torch.inference_mode():
        probability=model(x).softmax(-1);accuracy=float((probability.argmax(-1)==y.argmax(-1)).float().mean())
    atomic_json(out/'result.json',dict(status='completed_advisory_pilot',fit_soft_cross_entropy=float(loss.detach()),
        fit_patch_accuracy=accuracy,elapsed_seconds=time.monotonic()-start,
        independent_validation=False,provider_admitted=False,physical_success_verified=False,
        CUDA_initialized=torch.cuda.is_initialized(),next_gate='review draft labels and evaluate unseen reserved TRAIN frames before temporal validation'))
    print(json.dumps(dict(status='completed_advisory_pilot',path=str(out),fit_patch_accuracy=accuracy,provider_admitted=False)))


if __name__=='__main__':main()
