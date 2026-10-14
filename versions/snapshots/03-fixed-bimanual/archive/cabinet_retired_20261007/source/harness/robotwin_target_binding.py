"""Pixel-coordinate target ingress; no hidden object IDs or simulator state.
Qwen2.5-VL uses absolute pixels. Images are explicitly resized to multiples of
28 so processor resolution is known and verified before boxes are accepted.
"""
import hashlib,json,math
from dataclasses import dataclass
import numpy as np
from harness.robotwin_system_adapter import reject_gt
SIZE=(448,336)
PROMPT='''Locate the one small movable task object on the foreground table IN FRONT OF the cabinet, near either robot gripper. Exclude robot parts, the cabinet, and clutter farther back beside/behind the cabinet. The supplied HEAD image is exactly 448 pixels wide and 336 pixels high. Return ONLY a JSON list with ONE entry {"bbox_2d":[xmin,ymin,xmax,ymax],"label":"short visible appearance"}. Coordinates are ABSOLUTE PIXELS in this 448x336 image, not normalized. Bound only the visible portion. If the target is not visible or not uniquely identifiable, return an empty list. No actions or task-success claims.'''

@dataclass(frozen=True)
class TargetBinding:
    episode_id:str
    observation_id:str
    captured_at:float
    target_id:str
    bbox_native:tuple
    description:str
    source_rgb_sha256:str
    full_body_verified:bool=False

def parse_box(raw,grid,patch_size=14):
    if grid!=[[1,SIZE[1]//patch_size,SIZE[0]//patch_size]]:
        raise ValueError('actual_processor_resolution_mismatch')
    text=raw.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4].strip()
    result=json.loads(text)
    if not isinstance(result,list):raise ValueError('expected_candidate_list')
    if len(result)==0:return None
    if len(result)!=1:raise ValueError('ambiguous_multiple_candidates')
    obj=result[0]
    if not isinstance(obj,dict) or set(obj)!={'bbox_2d','label'}:raise ValueError('invalid_candidate_schema')
    b=obj['bbox_2d']
    if not isinstance(b,list) or len(b)!=4 or any(type(x)not in(int,float) or not math.isfinite(x) for x in b):raise ValueError('invalid_pixel_box')
    if not 0<=b[0]<b[2]<=SIZE[0] or not 0<=b[1]<b[3]<=SIZE[1]:raise ValueError('pixel_box_out_of_bounds')
    if not isinstance(obj['label'],str) or not obj['label'].strip():raise ValueError('missing_appearance')
    return {'bbox_2d':b,'label':obj['label']}

def bind(episode,oid,captured_at,rgb,candidate):
    reject_gt(candidate)
    if candidate is None:return None
    h,w=rgb.shape[:2]
    if rgb.dtype!=np.uint8 or rgb.shape!=(h,w,3) or not math.isfinite(captured_at):raise ValueError('invalid_sensor_frame')
    box=candidate['bbox_2d'];native=tuple(v*w/SIZE[0] if i%2==0 else v*h/SIZE[1] for i,v in enumerate(box))
    digest=hashlib.sha256(rgb.tobytes()).hexdigest()
    tid=hashlib.sha256((episode+oid+digest+json.dumps(box)).encode()).hexdigest()[:24]
    return TargetBinding(episode,oid,captured_at,tid,native,candidate['label'],digest)

def unproject_depth(depth_mm,intrinsic_cv,cam2world_gl):
    """Use metric optical depth + calibrated camera pose, never renderer IDs."""
    d=np.asarray(depth_mm,dtype=float)/1000;K=np.asarray(intrinsic_cv,dtype=float);T=np.asarray(cam2world_gl,dtype=float)
    if d.ndim!=2 or K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all() or abs(np.linalg.det(K))<1e-10:raise ValueError('invalid_camera_calibration')
    if not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4):raise ValueError('nonrigid_camera_calibration')
    h,w=d.shape;u,v=np.meshgrid(np.arange(w),np.arange(h));rays=np.stack([u,v,np.ones_like(u)],-1)@np.linalg.inv(K).T
    cv=rays*d[...,None];gl=cv*np.array([1,-1,-1]);world=gl@T[:3,:3].T+T[:3,3]
    valid=np.isfinite(d)&(d>0)&(d<10);world[~valid]=np.nan
    return world,valid
