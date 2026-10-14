#!/usr/bin/env python3
"""Read actual10Hz TRAIN joint arrays; separate expert spans from policy history."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
ARM=np.r_[0:7,8:15]

def motion(q,mask,segments):
    v=np.diff(q[:,ARM],axis=0)/.1;a=np.diff(v,axis=0)/.1;j=np.diff(a,axis=0)/.1
    validv=mask
    valida=mask[1:]&mask[:-1]&(segments[1:]==segments[:-1])
    validj=valida[1:]&valida[:-1]
    result={}
    for label,value,valid in (('velocity',v,validv),('acceleration',a,valida),('jerk',j,validj)):
        x=value[valid];assert x.size and np.isfinite(x).all()
        result[label]=dict(samples=len(x),sum_squared=float((x*x).sum()),elements=x.size,
            rms=float(np.sqrt(np.mean(x*x))),max_abs=float(abs(x).max()))
    moving=(np.max(abs(v),axis=1)>.02)&mask
    result['moving_frames']=int(moving.sum());result['selected_frames']=int(mask.sum())
    return result

def main():
    base=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24'
    out=base/'actual_demonstration_and_recovery_motion_v38.json';assert not out.exists()
    manifest=ROOT/'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger30k_diagnostic_20261011_v23/train/manifest.json'
    data=manifest.read_bytes();c=json.loads(data)['contract'];groups={}
    for label,rows in [('normal_train160',c['splits']['zhuhai_task2']['train']),('recovery_train120',c['dagger_training']['train_rows'])]:
        values=[]
        for row in rows:
            path=Path(row['path']);assert path.is_file() and not path.is_symlink()
            with h5py.File(path,'r') as f:
                q=f['measured_state'][:].astype(float);u=f['target_command'][:].astype(float);n=len(q)-1
                assert q.shape==u.shape==(n+1,16) and f.attrs['state_semantics']=='measured_joint_gripper'
                assert np.allclose(np.diff(f['physics_time'][:]),.1,rtol=0,atol=1e-6)
                mask=f['expert_action_mask'][:].astype(bool) if 'expert_action_mask' in f else np.ones(n,bool)
                seg=f['expert_segment_id'][:] if 'expert_segment_id' in f else np.zeros(n,int)
                assert len(mask)==len(seg)==n
                measured=motion(q,mask,seg);target=motion(u,mask,seg)
                boundaries=np.flatnonzero(mask[1:]&~mask[:-1])+1
                seam_max=float(abs(u[boundaries+1][:,ARM]-u[boundaries][:,ARM]).max()) if len(boundaries) else None
                arrays=hashlib.sha256(q.tobytes()+u.tobytes()+mask.tobytes()+seg.tobytes()).hexdigest()
                values.append(dict(path=str(path),seed=row['seed'],bound_full_hdf_sha256=row['sha256'],
                    read_motion_arrays_sha256=arrays,selected_expert_steps=int(mask.sum()),excluded_policy_steps=int((~mask).sum()),
                    measured=measured,target=target,expert_takeover_count=len(boundaries),max_first_expert_target_step_rad=seam_max))
        aggregate={}
        for kind in ('measured','target'):
            aggregate[kind]={}
            for quantity in ('velocity','acceleration','jerk'):
                samples=[r[kind][quantity] for r in values]
                aggregate[kind][quantity]=dict(rms=float(np.sqrt(sum(x['sum_squared'] for x in samples)/sum(x['elements'] for x in samples))),
                    max_abs=max(x['max_abs'] for x in samples),episode_rms_p50=float(np.median([x['rms'] for x in samples])),
                    episode_rms_p95=float(np.percentile([x['rms'] for x in samples],95)))
        groups[label]=dict(episodes=len(values),expert_steps=sum(x['selected_expert_steps'] for x in values),
            excluded_policy_steps=sum(x['excluded_policy_steps'] for x in values),aggregate=aggregate,episodes_detail=values)
    result=dict(schema='actual10Hz-demonstration-expert-span-motion-v38',status='passed',groups=groups,
        original_manifest_sha256=hashlib.sha256(data).hexdigest(),joint_indices=ARM.tolist(),
        frequency_hz=10,grippers_excluded=True,segments_never_concatenated=True,
        caveats=['Original full-HDF hashes were validated by previous training admission; this extra check hashes the freshly read motion arrays, not every RGB byte again.',
            'Only valid expert masks define action labels; policy histories are excluded from these smoothness statistics.',
            'Acceleration/jerk require genuinely adjacent expert intervals in the same segment, not concatenated takeovers.',
            'Measured10Hz derivatives are descriptive; contacts and100ms position-hold dispatch can add short physical transients.',
            'Reasonable expert trajectories do not prove recovery coverage or learned closed-loop success.'])
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('episodes','expert_steps','excluded_policy_steps','aggregate')} for k,v in groups.items()}))

if __name__=='__main__':main()
