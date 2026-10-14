#!/usr/bin/env python3
"""User-requested videos derived from original onboard archives, never a simulator replay.

Select first verified native-success row per fixed task before rendering. Every
archived RGB observation is used once, followed by an explicitly labeled final
still. Fixed control-step playback rate is NOT simulator/real-time recording.
Original evaluation evidence and no-video receipts remain untouched.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import av

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness.robotwin_eval_evidence import resolve_ffmpeg
from harness.robotwin_policy_worker import identity,digest

TASKS=[('put_bottles_dustbin','投瓶入垃圾桶'),('place_cans_plasticbox','易拉罐放入塑料盒'),
       ('open_laptop','打开笔记本'),('click_bell','按铃'),('press_stapler','按订书机')]
W,H=1280,720
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def metadata(path):
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        frames=0;first=None;last=None
        for frame in container.decode(stream):
            if first is None:first=frame.to_image()
            last=frame.to_image();frames+=1
        return dict(codec=stream.codec_context.name,width=stream.width,height=stream.height,
                    frames=frames,fps=float(stream.average_rate),duration=float(stream.duration*stream.time_base)),first,last


def render(row,stats,label,out,fps,hold,ffmpeg):
    native=Path(row['native_output']);ep=row['episode'];onboard=native/ep['episode']/'onboard'
    source_manifest=json.loads((onboard/'manifest.json').read_text())
    index=[json.loads(line) for line in (onboard/'frames.jsonl').read_text().splitlines() if line.strip()]
    if not (row['native_success'] is True and row['verification']['verified_complete'] is True
            and source_manifest['complete'] is True and len(index)==ep['control_steps']+1):
        raise ValueError('only complete verified original success evidence is exportable')
    name=f"{row['task']}_seed_{row['seed']}";target=out/(name+'.mp4')
    font=ImageFont.truetype(FONT,26,index=2);small=ImageFont.truetype(FONT,18,index=2)
    dark=(15,22,34);white=(234,241,249);muted=(173,189,207);green=(92,222,162)
    base=Image.new('RGB',(W,H),dark);draw=ImageDraw.Draw(base)
    draw.text((16,7),label+' · 成功回合回放',font=font,fill=white)
    draw.text((714,13),f"WAM V3+VTT 50k  |  seed {row['seed']}  |  本任务 {stats['native_successes']}/10",font=small,fill=green)
    draw.text((16,39),'头部相机（原机载图像）',font=small,fill=muted)
    draw.text((880,39),'左腕相机',font=small,fill=muted)
    draw.text((880,352),'右腕相机',font=small,fill=muted)
    command=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo','-pixel_format','rgb24',
             '-video_size',f'{W}x{H}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
             '-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    archive=[];first_preview=None;last_pixels=None;start_wall=index[0]['wall_monotonic']
    def compose(images,frame,holding=False):
        canvas=base.copy()
        for key,box,size in [('head_camera',(16,64),(800,600)),('left_camera',(880,64),(384,288)),('right_camera',(880,376),(384,288))]:
            rgb=images[key]
            if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('invalid original RGB')
            canvas.paste(Image.fromarray(rgb).resize(size,Image.Resampling.BICUBIC),box)
        d=ImageDraw.Draw(canvas)
        if holding:
            text='末帧定格展示（没有继续仿真，不代表额外稳定性验证）'
        else:
            text=f"控制步 {frame['control_step']:04d}/{ep['control_steps']:04d}  |  采集墙钟 +{frame['wall_monotonic']-start_wall:.1f}s  |  {fps} fps 按控制步等间隔播放，非实时速度"
        d.text((16,675),text,font=small,fill=muted if not holding else green)
        d.rectangle((16,710,1264,715),fill=(47,65,84))
        end=16+int(1248*frame['control_step']/max(1,ep['control_steps']))
        d.rectangle((16,710,end,715),fill=green)
        return canvas
    with (out/(name+'_encode.log')).open('xb') as err:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=err)
        try:
            for i,frame in enumerate(index):
                if frame['index']!=i or frame['control_step']!=i or frame['file']!=f'observation_{i:06d}.npz':
                    raise ValueError('archive ordering mismatch')
                path=onboard/frame['file'];archive.append(identity(path))
                with np.load(path,allow_pickle=False) as data:
                    pixels={k:np.array(data[k],copy=True) for k in ('head_camera','left_camera','right_camera')}
                image=compose(pixels,frame)
                if i==0:first_preview=image.copy()
                if i==len(index)-1:last_pixels=pixels
                proc.stdin.write(image.tobytes())
            if digest(archive)!=row['verification']['onboard_files_sha256']:
                raise ValueError('original archived observation hashes differ from evaluation verification')
            still=compose(last_pixels,index[-1],True)
            for _ in range(hold*fps):proc.stdin.write(still.tobytes())
            proc.stdin.close()
            if proc.wait(timeout=180)!=0:raise RuntimeError('ffmpeg encode failed')
        except BaseException:
            if proc.poll() is None:proc.kill();proc.wait()
            raise
    probe,decoded_first,decoded_last=metadata(target)
    expected=len(index)+hold*fps
    if probe['frames']!=expected or (probe['width'],probe['height'])!=(W,H) or probe['fps']!=fps or probe['codec']!='h264':
        raise ValueError('encoded video did not pass full decode/frame-count verification')
    preview=Image.new('RGB',(W,H*2),dark);preview.paste(decoded_first,(0,0));preview.paste(decoded_last,(0,H));preview.save(out/(name+'_qa.jpg'),quality=90)
    decoded_first.save(out/(name+'_poster.jpg'),quality=92)
    record=dict(task=row['task'],label=label,seed=row['seed'],selected_by='first verified success in the fixed evaluation order',
        trial_successes=stats['native_successes'],trial_count=10,source_native_result=identity(native/'result.json'),
        source_manifest=identity(onboard/'manifest.json'),source_frame_index=identity(onboard/'frames.jsonl'),
        source_observation_aggregate_sha256=row['verification']['onboard_files_sha256'],source_observations=len(index),
        all_source_frames_used=True,source_control_steps=ep['control_steps'],source_episode_wall_seconds=ep['wall_seconds'],
        video=str(target.resolve()),video_sha256=hash_file(target),video_bytes=target.stat().st_size,**probe,
        final_still_seconds=hold,playback='uniform control-step samples; NOT realtime or continuous physics-rate capture',
        original_evaluation_record_video=False,user_requested_posthoc_export=True,simulation_rerun=False,
        new_camera_created=False,source_evidence_modified=False)
    write(out/(name+'_provenance.json'),record)
    print(json.dumps({k:record[k] for k in ('task','seed','frames','duration','video_bytes')},ensure_ascii=False),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fps',type=int,default=10);p.add_argument('--final-still-seconds',type=int,default=2);a=p.parse_args()
    if not 1<=a.fps<=30 or not 0<=a.final_still_seconds<=5:raise ValueError('invalid playback settings')
    run=Path(a.run).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((run/'eval/summary.json').read_text())
    if summary['verified_complete'] is not True or summary['scored_episodes']!=50:raise ValueError('final verified50 results required')
    selected=[]
    for task,label in TASKS:
        row=next(e for e in summary['episodes'] if e['task']==task and e['native_success'] is True)
        selected.append((row,summary['tasks'][task],label))
    ffmpeg=resolve_ffmpeg()
    selection=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        explicit_user_request='give successful videos for the five tasks',source_summary=identity(run/'eval/summary.json'),
        source_checkpoint_sha256=json.loads((run/'eval/plan.json').read_text())['checkpoint_sha256'],
        selections=[dict(task=row['task'],seed=row['seed'],successes=stats['native_successes'],trials=10) for row,stats,_ in selected],
        policy='first verified success per task, selected before rendering; never rerun or alter a result',
        fps=a.fps,final_still_seconds=a.final_still_seconds,ffmpeg=identity(ffmpeg),no_third_person_source_available=True)
    write(out/'selection.json',selection)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(render,row,stats,label,out,a.fps,a.final_still_seconds,ffmpeg) for row,stats,label in selected]
        videos=[f.result() for f in futures]
    playlist=out/'concat.txt';playlist.write_text(''.join("file '"+v['video'].replace("'","'\\''")+"'\n" for v in videos))
    combined=out/'five_tasks_success_replays.mp4'
    with (out/'combined_encode.log').open('xb') as log:
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-f','concat','-safe','0','-i',str(playlist),
                        '-c','copy','-movflags','+faststart',str(combined)],check=True,stdout=subprocess.DEVNULL,stderr=log,timeout=180)
    combined_meta,_,_=metadata(combined)
    if combined_meta['frames']!=sum(v['frames'] for v in videos):raise ValueError('combined video frame count mismatch')
    write(out/'manifest.json',dict(status='completed',created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        selection=selection,videos=videos,combined=dict(path=str(combined),sha256=hash_file(combined),bytes=combined.stat().st_size,**combined_meta),
        original_evidence_unchanged=True,simulation_rerun=False,recording_default_remains_off=True))

if __name__=='__main__':main()
