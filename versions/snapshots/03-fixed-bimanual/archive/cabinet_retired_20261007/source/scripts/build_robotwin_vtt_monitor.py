#!/usr/bin/env python3
"""Whole-task VTT monitoring bank from TRAIN frozen DINO CLS transitions.
This is not an action condition and never proves physical task completion.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
def floats(a):return torch.from_numpy(np.array(a,copy=True).view(np.int16)).view(torch.bfloat16).float().numpy()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);torch.set_num_threads(2)
    split_path=ROOT/'data/benchmarks/robotwin2/put_object_cabinet_500_v1/split_manifest.json';split=json.loads(split_path.read_text());names=split['splits']['train'];paths={}
    for path in (ROOT/'data/robotwin_features').glob('*/episode_*-*/meta.json'):
        meta=json.loads(path.read_text());name=meta['identity']['episode']
        if name in names and meta.get('complete'):
            if name in paths:raise ValueError('ambiguous episode cache')
            paths[name]=path.parent/'feat.npy'
    records=[];deltas=[];starts=[]
    for name in names:
        arr=np.load(paths[name],mmap_mode='r');idx=[int(round((len(arr)-1)*f)) for f in (0,.25,.5,.75,1)]
        cls=floats(arr[idx,0,-1,0]);delta=cls-cls[0];deltas.append(delta);starts.append(cls[0]);records.append({'episode':name,'frames':idx,'length':len(arr)})
    ds=np.stack(deltas);proto=ds[:320,-1].mean(0);norm=float(np.linalg.norm(proto));unit=proto/(norm+1e-8)
    progress=ds@unit/(norm+1e-8);alignment=(ds@unit)/np.maximum(np.linalg.norm(ds,axis=-1),1e-8)
    # TRAIN inner-fit fixed 90% endpoint nomination coverage. No physical or
    # native GT success label, and no val/heldout used for these thresholds.
    pt=float(np.quantile(progress[:320,-1],.1));at=float(np.quantile(alignment[:320,-1],.1))
    nominations=(progress>=pt)&(alignment>=at)
    for i,r in enumerate(records):r.update(partition='inner_fit' if i<320 else 'inner_check',progress=progress[i].tolist(),alignment=alignment[i].tolist(),candidate=nominations[i].tolist())
    np.save(out/'prototypes.npy',proto[None].astype(np.float32));(out/'meta.json').write_text(json.dumps({'classes':{'put_object_cabinet':0},'representation':'DINOv3 layer-4 head CLS endpoint minus start, raw, 1024D','scope':'whole-task monitoring only; not WAM task condition; TRAIN inner split320/80'},indent=2)+'\n')
    np.save(out/'action_vtt_prototype.npy',ds[:,-1].mean(0).astype(np.float32))
    np.save(out/'vtt_instances.npy',ds[:,-1].astype(np.float32));np.save(out/'context_cls.npy',np.stack(starts).astype(np.float32))
    result={'completed':True,'TRAIN_episodes':400,'inner_fit':320,'inner_check':80,'prototype_norm':norm,'done_progress':pt,'done_alignment':at,'min_steps':30,'whole_task_monitor_admitted':False,'action_condition_compatible':False,'physical_success_verifier':False,'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'inner_check_candidate_counts_by_fraction':nominations[320:].sum(0).tolist(),'records':records}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='records'},flush=True)
if __name__=='__main__':main()
