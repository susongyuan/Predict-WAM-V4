#!/usr/bin/env python3
"""One bounded TRAIN-only localization fit, fixed pre-registered 16-case check."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image,ImageDraw
from harness.robotwin_wrist_target import model,patch_features,detect
from scripts.probe_robotwin_target_binding import iou
from harness.robotwin_eval_evidence import ResourceGuard
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--background-negatives',action='store_true');p.add_argument('--annotations-dir',default='outputs/validation/wrist_target_train32_20261006');p.add_argument('--fit-cases',type=int,default=16);p.add_argument('--negative-image-weight',type=float,default=1.);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False)
    if not 1 <= a.negative_image_weight <= 4 or a.negative_image_weight != 1 and not a.background_negatives:raise ValueError('bounded negative weighting needs negative images')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261006);guard=ResourceGuard();guard.check();base=ROOT/a.annotations_dir;gold=json.loads((base/'annotations.json').read_text());xs=[];ys=[];masks=[];n=a.fit_cases
    if n<=0 or n>=len(gold['cases']):raise ValueError('fit/check partition required')
    if any(c['partition']!=('fit' if i<n else 'reserved_check') for i,c in enumerate(gold['cases'])):raise ValueError('partition order mismatch')
    for c in gold['cases']:
        cache=next((ROOT/'data/robotwin_features').glob('*/'+c['episode'].removesuffix('.hdf5')+'-*/feat.npy'));arr=np.load(cache,mmap_mode='r');view=1 if c['view']=='cam_left_wrist' else 2
        bits=np.array(arr[c['frame'],view,-1,5:],copy=True);x=patch_features(torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).float());xs.append(x)
        b=[v/(320 if j%2==0 else 240)*14 for j,v in enumerate(c['bbox_native'])];overlap=[]
        for y in range(14):
            for xx in range(14):overlap.append(max(0,min(xx+1,b[2])-max(xx,b[0]))*max(0,min(y+1,b[3])-max(y,b[1])))
        o=torch.tensor(overlap)[:,None];ys.append((o>=.5).float());masks.append((o==0)|(o>=.5))
    nx=[]
    if a.background_negatives:
        for c in gold['cases']:
            cache=next((ROOT/'data/robotwin_features').glob('*/'+c['episode'].removesuffix('.hdf5')+'-*/feat.npy'));arr=np.load(cache,mmap_mode='r');view=1 if c['view']=='cam_left_wrist' else 2
            bits=np.array(arr[0,view,-1,5:],copy=True);nx.append(patch_features(torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).float()))
    protocol={'registered_before_fit':True,'fit_episodes':16,'fixed_check_episodes':16,'steps':1500,'seed':20261006,'learning_rate':.003,'weight_decay':.1,'dropout':.2,'architecture':'1029_64_1_GELU','threshold':.6,'component_area':[6,160],'ambiguity_ratio':.9,'admission':'at least14/16 fixedcheck IoU>=.3 to proceed to wider testing; never physical verification','scope':'TRAIN wrist pregrasp partial visible boxes only','annotation_sha256':hashlib.sha256((base/'annotations.json').read_bytes()).hexdigest(),'encoder_WAM_updates':False}
    protocol['fit_episodes']=n;protocol['fixed_check_episodes']=len(xs)-n;protocol['fit_episode_names']=[c['episode'] for c in gold['cases'][:n]];protocol['check_episode_names']=[c['episode'] for c in gold['cases'][n:]]
    protocol['background_negative_fit_cases']=n if nx else 0;protocol['background_negative_check_cases']=len(nx)-n if nx else 0
    protocol['check_previously_seen_in_v1']=bool(nx) and n==16;protocol['admission']='fresh32: >=30/32 positive IoU>=.3 and zero background false candidates; observation aid only, not physical verifier' if n==32 else 'development only after repeated check; fresh episode validation needed, never physical verification'
    protocol['negative_image_weight']=a.negative_image_weight;protocol['check_previously_used_for_development']=a.negative_image_weight!=1
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');X=torch.stack(xs[:n]+nx[:n]);Y=torch.stack(ys[:n]+[torch.zeros(196,1) for _ in nx[:n]]);M=torch.stack(masks[:n]+[torch.ones(196,1,dtype=torch.bool) for _ in nx[:n]]);net=model();opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.1);pos=(M.sum()-(Y*M).sum())/(Y*M).sum();sample_weight=torch.ones(len(X),1,1);sample_weight[n:]=a.negative_image_weight;start=time.monotonic()
    for step in range(1500):
        guard.check();opt.zero_grad();raw=torch.nn.functional.binary_cross_entropy_with_logits(net(X),Y,pos_weight=pos,reduction='none');loss=(raw*M*sample_weight).sum()/(M*sample_weight).sum();loss.backward();opt.step()
        if step%250==0:print('UPDATE',step,'LOSS',float(loss),flush=True)
    net.eval();torch.save({'format':'wrist_target_patch_mlp_v1','state_dict':net.state_dict(),'protocol':protocol,'trained_views':['left_camera','right_camera']},out/'model.pt');records=[]
    with torch.inference_mode():
        for c,x in zip(gold['cases'],xs):
            prob=net(x).sigmoid().numpy().flatten();cand,reason=detect(prob);box=cand['bbox_1000'] if cand else None;expected=[v/(320 if j%2==0 else 240)*1000 for j,v in enumerate(c['bbox_native'])];score=iou(box,expected) if box else 0
            records.append({**c,'prediction':cand,'iou':score,'passed':score>=.3,'reason':reason})
            im=Image.open(base/f'{c["index"]:02d}.png').convert('RGB');d=ImageDraw.Draw(im);d.rectangle(c['bbox_native'],outline='lime',width=2)
            if box:d.rectangle([v/1000*(320 if j%2==0 else 240) for j,v in enumerate(box)],outline='red',width=2)
            im.save(out/f'{c["index"]:02d}.png')
    negatives=[]
    with torch.inference_mode():
        for c,x in zip(gold['cases'],nx):
            cand,reason=detect(net(x).sigmoid().numpy());negatives.append({'index':c['index'],'partition':c['partition'],'candidate':cand,'reason':reason})
    result={'completed':True,'fit_passes':sum(c['passed'] for c in records[:n]),'fixed_check_passes':sum(c['passed'] for c in records[n:]),'physical_verifier_admitted':False,'localization_admitted':False,'records':records,'elapsed_seconds':time.monotonic()-start,'resources':guard.peaks,'CUDA_initialized':torch.cuda.is_initialized()}
    result['background_false_positives_fit']=sum(c['candidate'] is not None for c in negatives[:n]);result['background_false_positives_check']=sum(c['candidate'] is not None for c in negatives[n:]);result['background_records']=negatives
    result['eligible_for_wider_testing']=result['fixed_check_passes']>=(30 if n==32 else 14) and bool(nx) and result['background_false_positives_check']==0
    result['independent_check']=not protocol['check_previously_used_for_development']
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',result['fit_passes'],result['fixed_check_passes'],flush=True)
if __name__=='__main__':main()
