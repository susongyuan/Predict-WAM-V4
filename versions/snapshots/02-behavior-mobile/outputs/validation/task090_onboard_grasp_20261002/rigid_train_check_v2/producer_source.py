"""Versioned RGB-D rigid-motion diagnostics, never an admitted manipulation verdict.

A height-change bound is conditional on correct material-point correspondences,
rigidity, the supplied diameter bound and the stated measurement margins. It is
not an absolute support-clearance estimate or a class-identity certificate.
"""
from __future__ import annotations

import cv2
import numpy as np

from harness.object_perception import Camera, camera_from_pose, head_pose
from harness.task090_perception import observed_floor_plane, load_calibration
from harness.task090_predicates import EEF, FINGERS
from harness.task090_observation import Task090ObservationProducer
from harness.task1_perception_adapter import find, HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX


def fit_rigid(source, target):
    a,b=np.asarray(source,float),np.asarray(target,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid_correspondences')
    ac,bc=a-a.mean(0),b-b.mean(0)
    u,s,vh=np.linalg.svd(ac.T@bc)
    if s[1] <= max(1e-10,s[0]*1e-4):
        raise ValueError('collinear_correspondences')
    r=vh.T@u.T
    if np.linalg.det(r)<0:
        vh[-1]*=-1
        r=vh.T@u.T
    t=b.mean(0)-r@a.mean(0)
    residual=np.linalg.norm(a@r.T+t-b,axis=1)
    return r,t,residual


def robust_rigid(source,target,threshold=.01):
    a,b=np.asarray(source,float),np.asarray(target,float)
    if len(a)<6:
        raise ValueError('too_few_correspondences')
    generator=np.random.default_rng(0)  # local RNG only; cannot alter WAM or simulator RNG
    best=None
    for _ in range(48):
        ids=generator.choice(len(a),3,replace=False)
        try:r,t,_=fit_rigid(a[ids],b[ids])
        except ValueError:continue
        keep=np.linalg.norm(a@r.T+t-b,axis=1)<=threshold
        if best is None or keep.sum()>best.sum():best=keep
    if best is None or best.sum()<6:
        raise ValueError('no_rigid_consensus')
    r,t,residual=fit_rigid(a[best],b[best])
    return r,t,float(np.sqrt(np.mean(residual**2))),best


def floor_normal(coefficients):
    a,b,c=np.asarray(coefficients,float)
    scale=np.sqrt(1+a*a+b*b)
    return np.array([-a,-b,1.])/scale,-c/scale


def height_change_bound(source,target,rotation,origin_floor,current_floor,diameter,fit_rms,plane_margin=0.):
    """Conditional bound for any point in the same rigid body, relative to the floor."""
    if diameter<=0 or fit_rms<0 or plane_margin<0:
        raise ValueError('invalid_geometry_margin')
    n0,d0=floor_normal(origin_floor);n1,d1=floor_normal(current_floor)
    a,b=np.asarray(source,float),np.asarray(target,float)
    gain=float(n1@b.mean(0)+d1-n0@a.mean(0)-d0)
    rotation_allowance=float(diameter*np.linalg.norm(np.asarray(rotation).T@n1-n0))
    spread=float(np.linalg.norm(np.ptp(a,axis=0)))
    # Conservative engineering allowance, not a statistical or certified sensor-error guarantee.
    measurement_allowance=float(plane_margin+.005+3*fit_rms+diameter*min(2.,3*fit_rms/max(spread,.05)))
    return dict(anchor_height_gain_m=gain,rotation_allowance_m=rotation_allowance,
                assumed_measurement_allowance_m=measurement_allowance,
                conditional_min_height_gain_m=gain-rotation_allowance-measurement_allowance,
                tracked_point_spread_m=spread)


class Task090RigidObservation:
    requires_applied_action=True

    def __init__(self,calibration_path,dino_artifact,max_frame_gap=6):
        self.measure=Task090ObservationProducer(calibration_path,max_frame_gap=max_frame_gap,dino_artifact=dino_artifact)
        self.calibration_path=str(calibration_path)
        self.max_frame_gap=max_frame_gap
        # Supplied object-model prior, not a runtime simulator state query.
        prior=load_calibration(self.calibration_path)['classes']['trash_can']['geometry_prior']
        self.diameter=float(np.linalg.norm(prior['native_bbox_m']))
        self.reset()

    def reset(self):
        self.episode=None
        self._clear()

    def _clear(self):
        self.uv=self.reference=self.previous_points=self.previous_gray=None
        self.origin_floor=self.origin_margin=None
        self.previous_eef=self.previous_width=self.previous_pose=None
        self.candidate_start=None
        self.candidate_count=0
        self.last_step=None

    @staticmethod
    def points(camera,depth,uv):
        u=np.rint(uv[:,0]).astype(int);v=np.rint(uv[:,1]).astype(int)
        return camera.backproject(uv[:,0],uv[:,1],depth[v,u])

    def update(self,observation,state,*,episode,step,head_features,action=None,object_id='trash_can'):
        report=self.measure.update(observation,state,episode=episode,step=step,
                                   object_id=object_id,head_features=head_features)
        diagnostic={'status':'unknown','candidate_grasp':False,'verified':False,
                    'assumptions':['correct material-point identity','rigid tracked body','valid diameter and sensor margins'],
                    'absolute_support_clearance':'unknown'}
        report['rigid_tracking']=diagnostic
        if episode!=self.episode:
            self._clear();self.episode=episode
        if report['status']!='measurements_available':
            self._clear();diagnostic['reason']=report['reason'];return report
        try:
            if self.last_step is not None and step-self.last_step>self.max_frame_gap:
                self._clear()
            rgb=np.asarray(find(observation,HEAD_RGB_SUFFIX))[...,:3].astype(np.uint8)
            depth=np.asarray(find(observation,HEAD_DEPTH_SUFFIX)).squeeze()
            cam=Camera(*camera_from_pose(head_pose(find(observation,CAM_POSE_SUFFIX))),rgb.shape[1],rgb.shape[0])
            floor=observed_floor_plane(depth,cam)
            if floor is None:raise ValueError('floor_not_observed')
            state=np.asarray(state,float);eef=state[EEF['right']];width=float(state[FINGERS['right']].sum())
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            box=report['detections'][0]['bbox_xyxy']
            mask=np.zeros(gray.shape,np.uint8);mask[box[1]:box[3],box[0]:box[2]]=255
            hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
            appearance=np.zeros(gray.shape,np.uint8)
            calibration=load_calibration(self.calibration_path)
            for lower,upper in calibration['classes']['trash_can']['hsv_boxes']:
                appearance|=cv2.inRange(hsv,np.asarray(lower,np.uint8),np.asarray(upper,np.uint8))
            mask[(appearance==0)|~np.isfinite(depth)|(depth<=.05)|(depth>=calibration['depth_max_m'])]=0
            if self.uv is None:
                if width<.095:raise ValueError('cannot_rebind_target_after_gripper_closes')
                found=cv2.goodFeaturesToTrack(gray,maxCorners=80,qualityLevel=.005,minDistance=6,mask=mask,blockSize=5)
                if found is None or len(found)<6:raise ValueError('not_enough_object_features')
                uv=found[:,0,:];points=self.points(cam,depth,uv)
                # Initialize away from the still-open hand, not on its moving fingers.
                keep=np.linalg.norm(points-eef,axis=1)>.2
                uv,points=uv[keep],points[keep]
                if len(points)<6:raise ValueError('features_not_separate_from_hand')
                self.uv=uv;self.reference=points.copy();self.previous_points=points.copy()
                self.origin_floor=floor['coefficients'].copy();self.origin_margin=float(floor['margin_m'])
                self.previous_pose=(np.eye(3),np.zeros(3))
                diagnostic.update(status='tracking_initialized',points=len(points))
            else:
                current,ok,_=cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,self.uv[:,None].astype(np.float32),None,
                    winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
                if current is None:raise ValueError('optical_flow_missing')
                backward,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.previous_gray,current,None,winSize=(31,31),maxLevel=3)
                if backward is None:raise ValueError('backward_flow_missing')
                uv=current[:,0];h,w=gray.shape
                valid=ok[:,0].astype(bool)&back_ok[:,0].astype(bool)&(np.linalg.norm(backward[:,0]-self.uv,axis=1)<=1.5)
                valid&=(uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1)
                u=np.clip(np.rint(uv[:,0]).astype(int),0,w-1);v=np.clip(np.rint(uv[:,1]).astype(int),0,h-1)
                valid&=mask[v,u]>0
                uv=uv[valid];reference=self.reference[valid]
                points=self.points(cam,depth,uv)
                r,t,rms,keep=robust_rigid(reference,points)
                uv,reference,points=uv[keep],reference[keep],points[keep]
                bound=height_change_bound(reference,points,r,self.origin_floor,floor['coefficients'],
                    self.diameter,rms,self.origin_margin+float(floor['margin_m']))
                old_r,old_t=self.previous_pose
                delta_r=r@old_r.T;delta_t=t-delta_r@old_t
                hand_error=float(np.linalg.norm(delta_r@self.previous_eef+delta_t-eef))
                near=float(np.min(np.linalg.norm(points-eef,axis=1)))
                applied=np.asarray(action,float) if action is not None else None
                closing=applied is not None and applied.shape==(23,) and np.isfinite(applied).all() and applied[22]<0
                stable=.01<=width<=.09 and abs(width-self.previous_width)<=.001
                other_open=float(state[FINGERS['left']].sum())>=.095
                sample_gap=None if self.last_step is None else step-self.last_step
                candidate=bool(closing and stable and other_open and near<=.25 and hand_error<=.02
                               and bound['conditional_min_height_gain_m']>=.03 and sample_gap==1)
                if candidate:
                    self.candidate_start=step if self.candidate_start is None else self.candidate_start
                    self.candidate_count+=1
                else:self.candidate_start=None;self.candidate_count=0
                held=bool(candidate and self.candidate_count>=3 and step-self.candidate_start>=15)
                diagnostic.update(status='rigid_motion_measured',points=len(points),fit_rms_m=rms,
                    hand_motion_residual_m=hand_error,nearest_tracked_point_to_eef_m=near,
                    gripper_contact_candidate=bool(closing and stable),candidate_grasp=held,
                    sample_gap_steps=sample_gap,sparse_sampling=sample_gap!=1,
                    assumed_rigid_body_diameter_m=self.diameter,
                    candidate_observations=self.candidate_count,**bound)
                self.uv,self.reference,self.previous_points=uv,reference,points
                self.previous_pose=(r,t)
            self.previous_gray=gray;self.previous_eef=eef.copy();self.previous_width=width;self.last_step=step
        except (ValueError,TypeError,IndexError,cv2.error,np.linalg.LinAlgError) as error:
            self._clear();diagnostic.update(status='unknown',reason=type(error).__name__+':'+str(error))
        return report
