#!/usr/bin/env python3
"""Fetch only pinned Task90 head-depth videos; never overwrite shared RGB metadata."""
import datetime
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT=Path('/home/user/Robotic/project2')
DATA=ROOT/'data/behavior-1k/2026-challenge-demos'
REPO='behavior-1k/2026-challenge-demos'
REVISION='4f50b44796641a4d526a19d9aeadc8aa51e2f2c2'
PREFIX='videos/observation.depth_linear.zed_link_camera_0/chunk-090'


def main():
    receipt=DATA/'DOWNLOAD_PROVENANCE_task0090_head_depth.json'
    if receipt.exists():
        raise SystemExit('Head-depth receipt already exists; inspect rather than repeat this job')
    rows=[x for x in HfApi().list_repo_tree(REPO,path_in_repo=PREFIX,repo_type='dataset',revision=REVISION,recursive=True) if hasattr(x,'size')]
    if not rows or any(not x.path.startswith(PREFIX+'/') or not x.path.endswith('.mp4') for x in rows):
        raise RuntimeError('Unexpected head-depth inventory')
    planned={'repo_id':REPO,'revision':REVISION,'task_id':90,'stream':'head_depth_only',
             'files':[{'path':x.path,'bytes':x.size} for x in rows], 'bytes':sum(x.size for x in rows),
             'decode_status':'not decoded; must use official gray12le/log-depth conversion, output units mm',
             'calibration_split_rule':'only TRAIN episode indices; never use dev/report frames to fit'}
    print(json.dumps({'planned_files':len(rows),'bytes':planned['bytes']},indent=2),flush=True)
    snapshot_download(REPO,repo_type='dataset',revision=REVISION,allow_patterns=[PREFIX+'/**'],local_dir=DATA,max_workers=2)
    for x in rows:
        if (DATA/x.path).stat().st_size!=x.size:raise RuntimeError('Missing/incomplete depth file: '+x.path)
    planned['completed_at']=datetime.datetime.now().astimezone().isoformat()
    with receipt.open('x') as f:json.dump(planned,f,indent=2)
    print(json.dumps({'completed':True,'receipt':str(receipt),'files':len(rows)},indent=2),flush=True)


if __name__=='__main__':main()
