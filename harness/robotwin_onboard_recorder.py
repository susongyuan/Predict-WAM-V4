"""Lossless native policy observation archive; evaluator-side only, no GT."""
from pathlib import Path
import os
import time
import numpy as np
from harness.robotwin_qpos_adapter import RGB_VIEWS
from harness.robotwin_policy_worker import atomic_json

class OnboardRecorder:
    def __init__(self, directory, episode, seed):
        self.path=Path(directory)/'onboard';self.path.mkdir(exist_ok=False)
        self.episode,self.seed=episode,seed;self.count=0
        self.manifest={'schema':'robotwin-onboard-observations-v1','episode':episode,'seed':seed,
                       'frames':0,'complete':False,'lossless':True,'policy_inputs_unchanged':True,
                       'fields':list(RGB_VIEWS)+['commanded_joint_gripper_state'],
                       'time_semantics':'wall monotonic capture time, not simulation time',
                       'source':'same native RGB + commanded state allowlist as WAM, never GT or third view'}
        atomic_json(self.path/'manifest.json',self.manifest)
    def append(self, observation, control_step, settle_step=0):
        if set(observation)!={'rgb','commanded_joint_gripper_state'} or set(observation['rgb'])!=set(RGB_VIEWS):
            raise ValueError('onboard archive accepts exact policy allowlist only')
        data={}
        for v in RGB_VIEWS:
            im=np.asarray(observation['rgb'][v])
            if im.dtype!=np.uint8 or im.ndim!=3 or im.shape[-1]!=3:raise ValueError('native RGB required')
            data[v]=im
        state=np.asarray(observation['commanded_joint_gripper_state'])
        if state.shape!=(14,) or not np.isfinite(state).all():raise ValueError('finite native14 required')
        data['commanded_joint_gripper_state']=state
        final=self.path/f'observation_{self.count:06d}.npz';tmp=final.with_suffix('.tmp')
        with tmp.open('xb') as f:np.savez_compressed(f,**data)
        os.replace(tmp,final)
        from harness.robotwin_eval_evidence import jsonl
        jsonl(self.path/'frames.jsonl',{'index':self.count,'file':final.name,'control_step':control_step,
                                     'settle_step':settle_step,'wall_monotonic':time.monotonic()})
        self.count+=1
    def close(self, complete=False):
        self.manifest.update(frames=self.count,complete=complete)
        atomic_json(self.path/'manifest.json',self.manifest)
