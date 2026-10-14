#!/usr/bin/env python3
"""Measure v2 rigid tracking on denser existing TRAIN sensors, without admission."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.task090_rigid_observation_v2 import Task090RigidObservation
from scripts.build_task090_dino_localizer import cached_frame,CACHE


def main():
    source=ROOT/'outputs/validation/task090_onboard_grasp_20261002/dense_train_v1/frames.jsonl'
    rows=[json.loads(line) for line in source.open()]
    output=ROOT/'outputs/validation/task090_onboard_grasp_20261002/rigid_train_check_v2b'
    output.mkdir(parents=True,exist_ok=False)
    lookup={}
    for path in sorted({row['lowdim_source'] for row in rows}):
        table=pq.read_table(path,columns=['episode_index','frame_index','action'],
                            filters=[('episode_index','in',[18000,18001])])
        for row in table.to_pylist():lookup[(row['episode_index'],row['frame_index'])]=row['action']
    cached={ep:set(np.load(CACHE/f'episode_{ep}.frames.npy',allow_pickle=False).tolist()) for ep in (18000,18001)}
    producer=Task090RigidObservation(ROOT/'configs/task090_perception_geometry_v2.json',
        ROOT/'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz',max_frame_gap=6)
    counts=Counter();bounds=[];started=time.monotonic();processed=0;candidates=0
    with (output/'measurements.jsonl').open('x') as result:
        for row in rows:
            ep,step=row['episode'],row['frame_index']
            assert ep in (18000,18001) and row['split']=='train' and row['physical_success_label'] is None
            if step not in cached[ep]:
                producer.reset();producer.measure.reset()
                report={'status':'unknown','reason':'no_frozen_feature_for_exact_frame','episode':ep,'observation_id':step,
                        'verified':False,'admitted':False}
                counts[report['reason']]+=1
            else:
                obs={'robot_r1::robot_r1:zed_link:Camera:0::rgb':np.asarray(Image.open(row['rgb'])),
                     'robot_r1::robot_r1:zed_link:Camera:0::depth_linear':np.load(row['depth_m'],allow_pickle=False),
                     'robot_r1::cam_rel_poses':np.asarray(row['camera_pose7'])}
                report=producer.update(obs,np.asarray(row['state61']),episode=ep,step=step,
                    head_features=cached_frame(ep,step),action=lookup.get((ep,step-1)))
                rigid=report['rigid_tracking'];counts[rigid.get('reason',rigid['status'])]+=1
                if 'conditional_min_height_gain_m' in rigid:bounds.append(rigid['conditional_min_height_gain_m'])
                candidates+=bool(rigid.get('candidate_grasp'))
                processed+=1
            report['source_frame']={'episode':ep,'frame':step,'rgb':row['rgb'],'depth':row['depth_m'],
                                    'physical_success_label':None,'action_semantics':'recorded action at frame-1; never executed here'}
            result.write(json.dumps(report,ensure_ascii=False)+'\n')
    summary=dict(status='cpu_diagnostic_not_admission',samples=len(rows),exact_cached_frames_processed=processed,
        reasons=dict(counts),conditional_height_bound_range=None if not bounds else [min(bounds),max(bounds)],
        candidate_grasp_frames=candidates,physical_labels_available=0,admitted=False,
        sampling='5fps real samples, 6 control steps apart; not dense 30Hz success evidence',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256((ROOT/'harness/task090_rigid_observation_v2.py').read_bytes()).hexdigest(),
        uses_dev=False,uses_test=False,simulator_executed=False,policy_executed=False,
        wall_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
