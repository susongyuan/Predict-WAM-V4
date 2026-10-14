#!/usr/bin/env python3
"""Export the predetermined first case after it exits; never select by score."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import precision_continuity_evidence_v36 as e
from scripts.precision_continuity_replay_v36 import video_evidence
from scripts.progress1_log import log


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True);parser.add_argument('--wait-seconds',type=float,default=1800)
    args=parser.parse_args();root=Path(args.directory);deadline=time.monotonic()+args.wait_seconds
    while not (root/'exit_seed_932106.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('first-case exit receipt not ready')
        if (root/'failure_guard.json').exists():raise ValueError('guard before first-case receipt; preserve')
        time.sleep(10)
    directory=root/e.TASK/'seed_932106';raw=e.read_json(directory/'result.json')
    sources=video_evidence(directory,raw)
    out=ROOT/'success video/80k_continuity_seed932106_third_person_full.mp4'
    if out.exists():raise ValueError('preserve previous video export')
    log('第一例预注册第三视角完整录像完成；独立全解码/源hash/物理时钟验证后导出success video，不按得分选片。','root')
    shutil.copy2(directory/'third_person_full.mp4',out)
    e.require(e.sha256_file(out)==sources['third_person_full.mp4']['sha256'],'export differs')
    preview=out.with_suffix('.preview.png')
    ffmpeg=next((ROOT/'.conda-envs/behavior6/lib/python3.12/site-packages/imageio_ffmpeg/binaries').glob('ffmpeg-linux-*'))
    subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-i',str(out),'-frames:v','1',str(preview)],check=True)
    value=dict(schema='verified-requested-third-person-export-v36',status='passed',
        path=str(out),video_sha256=e.sha256_file(out),source_result_sha256=e.sha256_file(directory/'result.json'),
        full_decode_verified=True,selection='preregistered first seed932106 regardless of outcome',
        sources=sources,recording_receipt=raw['observer_video_receipt'],raw_status=raw['status'],
        full_task_success=raw['full_task_success'],base_score=raw['base_score'],preview=str(preview),
        original_evaluation_evidence_preserved=True,policy_input=False)
    out.with_suffix('.provenance.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
