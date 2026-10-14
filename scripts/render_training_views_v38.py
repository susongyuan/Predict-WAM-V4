#!/usr/bin/env python3
"""Requested matched recovery/source onboard replays; MP4 only in user folder."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import h5py
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

VERIFY=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24'
SUPPORT=VERIFY/'user_video_support_artifacts';SUPPORT.mkdir(exist_ok=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def relocate():
    receipt=VERIFY/'mp4_only_folder_relocation_v38.json'
    assert not receipt.exists();rows=[]
    for p in (ROOT/'success video').iterdir():
        if p.is_file() and p.suffix.lower()!='.mp4':
            target=SUPPORT/p.name;assert not target.exists()
            before=sha(p);shutil.move(str(p),str(target));assert sha(target)==before
            rows.append(dict(old_path=str(p),new_path=str(target),sha256=before))
    receipt.write_text(json.dumps(dict(user_request='only MP4 in success video',files=rows,original_bytes_preserved=True),ensure_ascii=False,indent=2)+'\n')

def render(row,label,outname,recovery):
    source=Path(row['path']);assert sha(source)==row['sha256'],'source HDF changed'
    output=ROOT/'success video'/outname;assert not output.exists()
    ffmpeg=next((ROOT/'.conda-envs/behavior6/lib/python3.12/site-packages/imageio_ffmpeg/binaries').glob('ffmpeg-linux-*'))
    fontfile='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title=ImageFont.truetype(fontfile,28);font=ImageFont.truetype(fontfile,23);small=ImageFont.truetype(fontfile,21)
    logfile=SUPPORT/(output.stem+'.encoder.log')
    command=[str(ffmpeg),'-nostdin','-n','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24',
        '-video_size','1440x900','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','2',
        '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with h5py.File(source,'r') as f,logfile.open('xb') as log:
        q=f['measured_state'][:];u=f['target_command'][:];time=f['physics_time'][:]
        rules=[json.loads(x) for x in f['rule_evidence'][:]];n=len(q)-1
        mask=f['expert_action_mask'][:].astype(bool) if recovery else np.ones(n,bool)
        control=f['control_authority'][:].astype(int) if recovery else np.ones(n,int)
        assert len(mask)==n and len(rules)==len(q) and np.allclose(np.diff(time),.1,rtol=0,atol=1e-6)
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for i in range(n+1):
                canvas=Image.new('RGB',(1440,900),'#101820');d=ImageDraw.Draw(canvas)
                authority=('终止观测（没有后续动作标签）' if i==n else
                    ('有效专家纠正：本步用于动作训练' if mask[i] else
                     ('专家未通过核验：本步不作动作训练标签' if control[i]==1 else 'WAM自主动作：本步不作动作训练标签')))
                d.text((20,8),f'{label}｜seed {row["seed"]}｜三路机载图像原速回放',font=title,fill='white')
                d.text((20,48),f'{time[i]:.1f}秒｜{authority}',font=font,fill='#9cddbb' if i<n and mask[i] else '#ffca82')
                for camera,size,xy,name in zip(('cam_head','cam_left_wrist','cam_right_wrist'),
                    ((960,720),(480,360),(480,360)),((0,100),(960,100),(960,460)),('头部相机','左腕相机','右腕相机')):
                    key=('terminal_observation/vision/' if i==n else 'vision/')+camera+'/colors'
                    encoded=np.asarray(f[key][:] if i==n else f[key][i],dtype=np.uint8)
                    decoded=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                    assert decoded is not None and decoded.shape==(240,320,3),'missing or altered onboard frame'
                    rgb=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
                    canvas.paste(Image.fromarray(rgb).resize(size,Image.Resampling.LANCZOS),xy)
                    d.rectangle((xy[0]+5,xy[1]+5,xy[0]+170,xy[1]+38),fill='#101820')
                    d.text((xy[0]+10,xy[1]+8),name,font=small,fill='white')
                score=rules[i]['score'];items=score['items']
                names=dict(bowl='碗',spoon='勺',cup='杯',plate='盘')
                flags='  '.join(names[k]+('：已放好' if items[k]['placement_valid'] else ('：抓持达标' if items[k]['grasp_awarded'] else '：未完成')) for k in names)
                d.text((20,832),f'原规则得分 {score["base_score"]}/30｜{flags}',font=font,fill='#bdd6ee')
                d.text((20,868),f'左夹爪：当前目标 {u[i,7]:.2f} / 实测 {q[i,7]:.2f}（0闭合、1打开）｜原始320×240图像放大，未补帧或重跑',font=small,fill='white')
                child.stdin.write(canvas.tobytes())
        finally:
            child.stdin.close();code=child.wait(timeout=60)
        assert code==0,'encoder failed; preserve partial'
    decoded=subprocess.run([str(ffmpeg),'-nostdin','-v','error','-xerror','-i',str(output),'-map','0:v:0',
        '-progress','pipe:1','-f','null','-'],capture_output=True,text=True,timeout=180)
    counts=[int(x.split('=',1)[1]) for x in decoded.stdout.splitlines() if x.startswith('frame=')]
    assert decoded.returncode==0 and counts[-1]==n+1,'full video decode failed'
    assert sha(source)==row['sha256'],'source changed during export'
    result=dict(schema='requested-training-camera-replay-v38',source=row,video=str(output),sha256=sha(output),
        frames=n+1,duration_seconds=(n+1)/10,expert_label_frames=int(mask.sum()),
        excluded_nonlabel_frames=int((~mask).sum()),excluded_policy_frames=int(((~mask)&(control==0)).sum()),
        excluded_failed_expert_frames=int(((~mask)&(control==1)).sum()),whole_task_success=score['full_task_success'],score=score['base_score'],
        original_rgb_resolution=[320,240],policy_preprocessing_resolution=[224,224],playback_speed=1,
        source_hash_reverified=True,full_video_decode_verified=True,added_simulation_attempts=0,new_cameras=0,
        sample_selection='earliest TRAIN recovery with complete30-point expert continuation; source demonstration from same parent family',
        caveat='high-quality example means verified complete expert correction, not a claim that all camera frames/data are optimal')
    (SUPPORT/(output.stem+'.provenance.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('video','frames','duration_seconds','score','expert_label_frames','excluded_policy_frames','excluded_failed_expert_frames')},ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    relocate()
    manifest=ROOT/'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger30k_diagnostic_20261011_v23/train/manifest.json'
    contract=json.loads(manifest.read_text())['contract']
    recovery=next(r for r in contract['dagger_training']['train_rows'] if r.get('correction_variant')=='legacy_full_continuation' and r['full_task_success'])
    normal=next(r for r in contract['splits']['zhuhai_task2']['train'] if r['seed']==recovery['source_parent'] and r['domain']=='source')
    videos=[render(recovery,'完整成功恢复示教','recovery_high_quality_seed1010000_full.mp4',True),
        render(normal,'同来源场景正常示教','normal_source_seed953048_full.mp4',False)]
    assert all(p.suffix.lower()=='.mp4' for p in (ROOT/'success video').iterdir() if p.is_file())
    (VERIFY/'requested_training_camera_videos_v38.json').write_text(json.dumps(dict(status='passed',videos=videos,
        directory_contains_only_mp4=True,original_manifest_sha256=sha(manifest)),ensure_ascii=False,indent=2)+'\n')
