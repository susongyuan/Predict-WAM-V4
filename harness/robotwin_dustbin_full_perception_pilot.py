"""Unadmitted bottle proposals from sensor DINO patches, initial frame only.

This model supplies neither a persistent track ID nor grasp/completion truth.
It must not be passed as a trusted grounder or verifier in StrictMiddleKernel.
"""
from pathlib import Path
import numpy as np


class InitialBottleProposer:
    def __init__(self, artifact):
        import torch
        from harness.robotwin_policy_worker import sha256
        ck=torch.load(Path(artifact),map_location='cpu',weights_only=False)
        protocol=ck.get('protocol',{})
        if (ck.get('format')!='dustbin-initial-multiclass-patch-pilot-v1'
                or protocol.get('task')!='put_bottles_dustbin'
                or protocol.get('provider_admitted') is not False):
            raise ValueError('explicit unadmitted bottle-pilot artifact required')
        self.model=torch.nn.Linear(1029,4);self.model.load_state_dict(ck['state_dict'],strict=True);self.model.eval()
        self.classes=('clear_blue_label_bottle','yellow_bottle','dark_red_label_bottle')
        self.metadata=dict(artifact_sha256=sha256(artifact),status='advisory_unadmitted',
                           physical_success_verified=None,persistent_tracking=False)

    def propose(self, head_patch_features, *, frame_index, threshold=.5):
        import torch
        import cv2
        if frame_index!=0:
            raise ValueError('pilot validated scope is initial frame only; temporal provider not available')
        features=np.asarray(head_patch_features,np.float32)
        if features.shape!=(196,1024) or not np.isfinite(features).all() or not 0<threshold<1:
            raise ValueError('current sensor DINO[-4] 196x1024 patches required')
        f=torch.nn.functional.normalize(torch.from_numpy(features.copy()),dim=-1)
        yy,xx=torch.meshgrid(torch.linspace(1/28,27/28,14),torch.linspace(1/28,27/28,14),indexing='ij')
        pos=torch.stack([xx.flatten(),yy.flatten(),xx.flatten()**2,yy.flatten()**2,(xx*yy).flatten()],-1)
        with torch.inference_mode():prob=self.model(torch.cat([f,pos],-1)).softmax(-1).numpy().reshape(14,14,4)
        candidates=[]
        for class_index,name in enumerate(self.classes,1):
            mask=((prob.argmax(-1)==class_index)&(prob[:,:,class_index]>=threshold)).astype(np.uint8)
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
            for i in range(1,count):
                x,y,w,h,area=stats[i]
                if area<1:continue
                candidates.append(dict(semantic_class=name,bbox_xyxy_normalized=[x/14,y/14,(x+w)/14,(y+h)/14],
                    mean_probability=float(prob[:,:,class_index][labels==i].mean()),patch_area=int(area),
                    source='current_head_RGB_DINO_patches',status='advisory_unadmitted',
                    track_id=None,grasp_verified=None,task_success_verified=None))
        return candidates
