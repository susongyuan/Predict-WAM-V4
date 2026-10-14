"""DAgger admission, causal history and expert-only native16 action supervision."""
from __future__ import annotations
from pathlib import Path
from bisect import bisect_right
import hashlib, json, math, types
import numpy as np

FORMAT = 'zhuhai-native16-dagger-mixed-expert-mask-v1'
SCHEMA = 'zhuhai-task2-dagger50-v1'


class StallTrigger:
    """Privileged collection-time teacher gate, NOT a deployed visual monitor."""
    def __init__(self, minimum_steps=300, stalled_steps=180, maximum_steps=800):
        self.minimum_steps=minimum_steps;self.stalled_steps=stalled_steps;self.maximum_steps=maximum_steps
        self.achievements=set();self.last_progress=0

    def observe(self, step, evidence):
        if evidence['physics_safety']['passed'] is not True:
            raise ValueError('Unsafe policy prefix cannot enter correction collection')
        score=evidence['score']
        if score['full_task_success']:return None
        achieved={(key,field) for key,item in score['items'].items()
                  for field in ('grasp_awarded','placement_valid') if item[field]}
        if achieved-self.achievements:self.last_progress=step
        self.achievements |= achieved
        if step>=self.maximum_steps:return 'bounded_policy_prefix_without_full_success'
        if step>=self.minimum_steps and step-self.last_progress>=self.stalled_steps:
            return 'no_new_verified_grasp_or_placement'
        return None


def finish_mask(recorder, authorities, phases, takeover):
    """Distinct format makes the original unmasked SourceEpisode reject it."""
    n=recorder.length-1
    values=np.asarray(authorities,dtype=np.uint8)
    if values.shape!=(n,) or len(phases)!=n or not 0<takeover<n:
        raise ValueError('Control authority must align with every real action')
    if np.any(values[:takeover]!=0) or np.any(values[takeover:]!=1):
        raise ValueError('One actual policy-to-expert transfer required')
    recorder.file.attrs['source_format']=FORMAT
    recorder.file.attrs['expert_action_start']=takeover
    recorder.file.attrs['successful_expert_scope']='verified expert continuation; policy prefix is context only'
    recorder.file.attrs['training_requires_expert_mask']=True
    recorder.file.create_dataset('control_authority',data=values)
    recorder.file.create_dataset('expert_action_mask',data=values.astype(bool))
    import h5py
    recorder.file.create_dataset('control_phase',data=np.asarray(phases,dtype=object),dtype=h5py.string_dtype('utf-8'))


class DaggerEpisode:
    """Validate the original schema using an isolated format binding, not mutation.

    Full causal prefix remains available to history/clock; only expert anchors
    can be sampled. The original loader rejects this different source format.
    """
    def __init__(self,row):
        from harness.zhuhai_dataset import SourceEpisode
        import h5py
        self.row=dict(row)
        if row.get('split')!='train':raise ValueError('Correction families must stay TRAIN')
        # Reuse all existing measured-target/time/safety/RGB validation without
        # modifying a frozen module or relabeling the on-disk source.
        original=SourceEpisode.__init__
        validate=types.FunctionType(original.__code__,{**original.__globals__,'SOURCE_FORMAT':FORMAT},
                                    original.__name__,original.__defaults__,original.__closure__)
        self.native=SourceEpisode.__new__(SourceEpisode);validate(self.native,row)
        with h5py.File(self.native.path,'r') as f:
            self.expert_start=int(f.attrs['expert_action_start'])
            self.authority=f['control_authority'][:];mask=f['expert_action_mask'][:]
            self.phases=f['control_phase'].asstr()[:]
            self.metadata=json.loads(f.attrs['metadata'])
            final=json.loads(f.attrs['final_rule_evidence'])
            if f.attrs.get('training_requires_expert_mask')!=True or final['score']['full_task_success'] is not True:
                raise ValueError('Unverified correction continuation')
        n=len(self.native)
        if (not 0<self.expert_start<n or self.authority.shape!=(n,) or mask.shape!=(n,)
                or len(self.phases)!=n or np.any(self.authority[:self.expert_start]!=0)
                or np.any(self.authority[self.expert_start:]!=1)
                or not np.array_equal(mask,self.authority==1)):
            raise ValueError('Invalid expert action boundaries/mask')
        if self.metadata.get('family_id')!=row['family_id'] or self.metadata.get('split')!='train':
            raise ValueError('Correction family/split binding changed')

    def __getattr__(self,name):return getattr(self.native,name)
    def __len__(self):return len(self.native)
    def rgb(self,frame):return self.native.rgb(frame)
    @property
    def supervised_count(self):return len(self)-self.expert_start


class DaggerPrecisionDataset:
    """V2-compatible samples, full past context, no policy action labels/aux."""
    def __init__(self,task,rows,store,condition,chunk=32,future_offset=32,
                 proprio_lags=(2,4,8,16,32),memory_lags=(4,16)):
        self.task=task;self.episodes=[DaggerEpisode(r) for r in rows];self.store=store
        self.condition=np.asarray(condition,dtype=np.float32).copy();self.chunk=chunk
        self.future_offset=future_offset;self.proprio_lags=list(proprio_lags);self.memory_lags=list(memory_lags)
        self.ends=np.cumsum([ep.supervised_count for ep in self.episodes]).tolist()

    def __len__(self):return self.ends[-1] if self.ends else 0

    def __getitem__(self,index):
        from harness.wam_v3 import history_frames
        from harness.robotwin_training_data import FEAT_SHAPE
        from harness.wam_event_clock import episode_event_features
        from harness.zhuhai_precision_v2_training import CLOCK_CONFIG
        if not 0<=index<len(self):raise IndexError(index)
        i=bisect_right(self.ends,index);ep=self.episodes[i]
        t=ep.expert_start+index-(self.ends[i-1] if i else 0)
        feat,pool=self.store.get(ep,t);pools=[pool]
        for old in history_frames(t,0,self.memory_lags):pools.append(pool if old==t else self.store.get(ep,old)[1])
        valid=min(self.chunk,len(ep)-t);actions=np.zeros((self.chunk,16),np.float32)
        actions[:valid]=ep.action[t:t+valid]
        if not ep.authority[t:t+valid].all():raise ValueError('Policy actions cannot be positive labels')
        future=t+self.future_offset;fv=future<ep.observation_count
        short=t+8;sv=short<ep.observation_count;iv=t+4<ep.observation_count
        sample=dict(state=ep.state[[t]+history_frames(t,0,self.proprio_lags)].copy(),
            action_sequence=actions,action_valid_mask=np.arange(self.chunk)<valid,
            feats=feat,pool=np.stack(pools,axis=1),future_feats=self.store.get(ep,future)[0][:,-1,5:].copy() if fv else np.zeros((3,196,1024),np.uint16),
            future_valid_mask=np.bool_(fv),future_feats_8=self.store.get(ep,short)[0][:,-1,5:].copy() if sv else np.zeros((3,196,1024),np.uint16),
            future_valid_mask_8=np.bool_(sv),idm_valid_mask=np.bool_(iv),
            idm_future_feats=np.stack([self.store.get(ep,t+k)[0] for k in range(1,5)]) if iv else np.zeros((4,*FEAT_SHAPE),np.uint16),
            idm_target_raw=(ep.action[t:t+4]-ep.state[t]).copy() if iv else np.zeros((4,16),np.float32),
            task_cond=self.condition.copy(),phase=np.int64(-1),active_hand=np.int64(-1),dist=np.float32(0),
            task=self.task,episode=ep.path.name,anchor=t,supervision_source='expert_correction')
        start=max(0,t-math.ceil(CLOCK_CONFIG['max_seconds']/CLOCK_CONFIG['dt'])-2)
        import h5py
        if start==0:
            with h5py.File(ep.path,'r') as f:initial=f['target_command'][0].copy()
        else:initial=ep.action[start-1]
        sample['event_state']=episode_event_features(ep.state[start:t+1],ep.action[start:t+1],initial_target=initial,**CLOCK_CONFIG)[-1].copy()
        return sample
