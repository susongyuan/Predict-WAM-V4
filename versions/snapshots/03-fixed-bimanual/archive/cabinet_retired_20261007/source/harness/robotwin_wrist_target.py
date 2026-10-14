"""Frozen DINO patch localization on wrists; appearance-only, no grasp claims."""
import numpy as np
def patch_features(tokens):
    import torch
    if tokens.shape!=(196,1024):raise ValueError('expected14x14 frozen DINO patches')
    f=torch.nn.functional.normalize(tokens.float(),dim=-1)
    y,x=torch.meshgrid(torch.linspace(1/28,27/28,14,device=f.device),torch.linspace(1/28,27/28,14,device=f.device),indexing='ij')
    x=x.flatten();y=y.flatten();return torch.cat((f,torch.stack((x,y,x*x,y*y,x*y),-1)),-1)
def model():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(1029,64),torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.Linear(64,1))
def detect(prob):
    import cv2
    a=np.asarray(prob).reshape(14,14)
    if not np.isfinite(a).all():raise ValueError('invalid localization probabilities')
    n,ids,stats,_=cv2.connectedComponentsWithStats((a>=.6).astype(np.uint8),8);c=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if not 6<=area<=160:continue
        score=float(a[ids==i].mean());c.append({'bbox_1000':[x/14*1000,y/14*1000,(x+w)/14*1000,(y+h)/14*1000],'score':score,'patch_area':area})
    c.sort(key=lambda x:x['score'],reverse=True)
    if not c:return None,'no_candidate'
    if len(c)>1 and c[1]['score']>=.9*c[0]['score']:return None,'ambiguous_candidates'
    return c[0],'visible_patch_candidate_not_grasp_verified'
