#!/usr/bin/env python3
"""Pair existing TRAIN RGB observations with officially decoded head depth, in metres."""
import hashlib
import json
import math
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.depth_utils import dequantize_depth
import lerobot.datasets.depth_utils as codec

ROOT=Path('/home/user/Robotic/project2')
DATA=ROOT/'data/behavior-1k/2026-challenge-demos'
SOURCE=ROOT/'outputs/validation/task090_preparation_20260930/data_runtime/train_observations/manifest.json'
OUT=ROOT/'outputs/validation/task090_preparation_20260930/train_depth_samples'
DEPTH='observation.depth_linear.zed_link_camera_0'
RGB='observation.rgb.zed_link_camera_0'


def decode(path,timestamp,tolerance):
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        stream.codec_context.thread_count=1
        container.seek(int(max(0.,timestamp-.5)/stream.time_base),stream=stream,backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:continue
            pts=float(frame.pts*stream.time_base)
            if pts < timestamp-tolerance:continue
            if abs(pts-timestamp)>tolerance:raise ValueError(f'Depth PTS mismatch: {timestamp} vs {pts}')
            f=frame.reformat(format='gray12le')
            plane=f.planes[0]
            q=np.frombuffer(plane,dtype='<u2',count=f.height*(plane.line_size//2)).reshape(f.height,plane.line_size//2)[:,:f.width].copy()
            if q.max()>4095:raise ValueError('Depth sample is not an unscaled 12-bit code')
            return q,pts,frame.format.name
    raise ValueError('Depth timestamp missing: '+str(path))


def main():
    if OUT.exists():raise SystemExit('Refusing to overwrite depth calibration samples')
    source=json.loads(SOURCE.read_text());info=json.loads((DATA/'meta/info.json').read_text())
    train=set(json.loads((DATA/'split_task_090.json').read_text())['train_episode_indices'])
    meta={r['episode_index']:r for p in (DATA/'meta/episodes/chunk-090').glob('*.parquet') for r in pq.read_table(p).to_pylist()}
    enc=info['features'][DEPTH]['info'];fps=float(enc['video.fps'])
    if enc['video.pix_fmt']!='gray12le' or not enc['video.use_log']:raise ValueError('Unexpected depth encoding')
    OUT.mkdir(parents=True,exist_ok=False)
    records=[]
    for sample in source['samples']:
        if sample['camera']!=RGB:continue
        ep,frame=sample['episode'],sample['frame']
        if ep not in train or sample['split']!='train':raise ValueError('Non-TRAIN sample refused')
        row=meta[ep];prefix='videos/'+DEPTH
        video=DATA/info['video_path'].format(video_key=DEPTH,chunk_index=row[prefix+'/chunk_index'],file_index=row[prefix+'/file_index'])
        wanted=float(row[prefix+'/from_timestamp'])+frame/fps
        q,pts,pix=decode(video,wanted,.5/fps)
        params=dict(depth_min=enc['video.depth_min'],depth_max=enc['video.depth_max'],shift=enc['video.shift'],use_log=True,pix_fmt='gray12le')
        depth=dequantize_depth(q,**params,output_unit='m').squeeze(-1)
        mm=dequantize_depth(q,**params,output_unit='mm').squeeze(-1)
        reference=np.exp(q.astype(np.float64)/4095*(math.log(params['depth_max']+params['shift'])-math.log(params['depth_min']+params['shift']))+math.log(params['depth_min']+params['shift']))-params['shift']
        reference=np.clip(reference,params['depth_min'],params['depth_max'])
        assert depth.dtype==np.float32 and depth.shape==tuple(sample['image_size'])
        assert np.isfinite(depth).all() and np.max(np.abs(depth-reference))<1e-5
        assert np.max(np.abs(mm.astype(float)/1000-depth))<=.000501
        path=OUT/f'ep{ep}_f{frame:05d}_head_depth_m.npy';np.save(path,depth,allow_pickle=False)
        records.append({**sample,'depth_m':str(path),'depth_unit':'m','depth_shape':list(depth.shape),
                        'depth_source':str(video),'depth_requested_pts':wanted,'depth_decoded_pts':pts,'depth_original_pixel_format':pix,
                        'gray12_code_range':[int(q.min()),int(q.max())],'depth_range_m':[float(depth.min()),float(depth.max())],
                        'max_reference_error_m':float(np.max(np.abs(depth-reference))),
                        'frame_pairing':'same episode and original frame_index; per-stream timestamps are resolved independently',
                        'physical_success_label':None,'annotation_hint_is_not_outcome':True})
    report={'task_id':90,'split':'TRAIN only','sample_count':len(records),'source_rgb_manifest':str(SOURCE),
            'source_rgb_manifest_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'codec_source':codec.__file__,'codec_sha256':hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest(),
            'codec_parameters':enc,'stored_output_unit':'m','metadata_default_api_output_unit':enc['video.output_unit'],
            'cuda_initialized':torch.cuda.is_initialized(),'samples':records}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'samples':len(records),'manifest':str(OUT/'manifest.json'),'output_unit':'m','cuda_initialized':torch.cuda.is_initialized()},indent=2))


if __name__=='__main__':main()
