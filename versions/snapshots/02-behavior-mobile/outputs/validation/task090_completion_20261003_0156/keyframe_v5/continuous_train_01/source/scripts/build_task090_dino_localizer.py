#!/usr/bin/env python3
"""Build a TRAIN-only diagnostic localizer from existing frozen DINO caches."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / 'data/dino_cache_task090/observation.rgb.zed_link_camera_0'


def cached_frame(episode, frame):
    meta = json.loads((CACHE / f'episode_{episode}.meta.json').read_text())
    indices = np.load(CACHE / f'episode_{episode}.frames.npy', mmap_mode='r', allow_pickle=False)
    index = int(np.searchsorted(indices, frame))
    if index >= len(indices) or int(indices[index]) != frame:
        raise ValueError('requested frame is not in the frozen cache')
    source_meta = ROOT / f'data/frame_cache/observation.rgb.zed_link_camera_0/episode_{episode}.meta.json'
    if hashlib.sha256(source_meta.read_bytes()).hexdigest() != meta['source_frame_meta_sha256']:
        raise ValueError('source frame metadata mismatch')
    array = np.load(CACHE / f'episode_{episode}.feat.npy', mmap_mode='r', allow_pickle=False)
    if array.dtype != np.uint16 or tuple(array.shape[1:]) != (3,201,1024):
        raise ValueError('unexpected bfloat16 spatial cache')
    result = (array[index].astype(np.uint32) << 16).view(np.float32).copy()
    array._mmap.close(); indices._mmap.close()
    return result


def main():
    out = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1'
    out.mkdir(parents=True, exist_ok=False)
    rgb_path = ROOT / 'outputs/validation/task090_first_grasp_evidence_20261001/extracted/episode_18000/f00480_head_rgb.png'
    width, height = Image.open(rgb_path).size
    assert (width,height)==(720,720)
    annotations_path = ROOT / 'configs/task090_perception_annotations.json'
    annotations = json.loads(annotations_path.read_text())
    positive = next(x['bbox_xyxy'] for x in annotations['annotations']
                    if x['episode']==18000 and x['frame']==480 and x['label']=='trash_can')
    negative = next(x['bbox_xyxy'] for x in annotations['negative_rois']
                    if x['episode']==18000 and x['frame']==480)
    features = cached_frame(18000,480)[:,5:]
    features /= np.maximum(np.linalg.norm(features,axis=-1,keepdims=True),1e-8)
    row,col=np.divmod(np.arange(196),14)
    x,y=(col+.5)*width/14,(row+.5)*height/14
    def prototype(box):
        keep=(x>=box[0])&(x<box[2])&(y>=box[1])&(y<box[3])
        p=features[:,keep].mean(1)
        return p/np.linalg.norm(p,axis=-1,keepdims=True)
    artifact=out/'prototypes.npz'
    np.savez(artifact,positive=prototype(positive),negative=prototype(negative))
    metadata=dict(calibration_id='task090-dino-appearance-18000-480-v1',source_split='train',
        fit_episode=18000,fit_frame=480,positive_roi=positive,negative_roi=negative,
        source_annotation_sha256=hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        feat_layers=[-12,-8,-4],prefix_tokens=5,spatial_grid=[14,14],
        image_size_verified=[width,height],physical_success_verifier=False,presence_admitted=False,
        uses_dev=False,uses_test=False,uses_report=False,
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        limitation='single-frame appearance ranking; no absence threshold, metric/identity or physical admission')
    artifact.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    from harness.task090_dino_localizer import Task090DinoLocalizer
    from harness.object_perception import Camera, camera_from_pose
    source=ROOT/'outputs/validation/task090_first_grasp_evidence_20261001/extracted/frames.jsonl'
    sample=next(json.loads(line) for line in source.open() if json.loads(line)['episode']==18001 and json.loads(line)['frame_index']==870)
    image=np.asarray(Image.open(sample['rgb']));depth=np.load(sample['depth_m'],allow_pickle=False)
    cam=Camera(*camera_from_pose(sample['camera_pose7']),image.shape[1],image.shape[0])
    localizer=Task090DinoLocalizer(artifact)
    proposed=localizer.surface(image,depth,cam,cached_frame(18001,870),ROOT/'configs/task090_perception_geometry_v2.json')
    result=dict(status='diagnostic_candidate_only',held_out_episode=18001,frame=870,
        detection=None if proposed is None else proposed['detection'],
        surface_points=0 if proposed is None else len(proposed['cloud']),
        physical_success_verified=False,presence_verified=False)
    (out/'heldout_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
