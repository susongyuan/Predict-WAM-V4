"""CPU-only, complete 10Hz replay of the first accepted SOURCE per task."""
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]
ROOT = RUN.parents[2]
FFMPEG = ROOT / 'reproductions/homer_20261007/env/runtime/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS = {n: ImageFont.truetype(FONT, n) for n in (17, 20, 24)}
OBJECTS = {'lid_dry': '开启干垃圾桶盖', 'lid_wet': '开启湿垃圾桶盖',
           'milk_carton': '奶盒→干桶', 'squeeze_bottle': '挤酱瓶→干桶',
           'banana_peel': '刚性香蕉皮代理→湿桶', 'bowl': '碗→白盒',
           'spoon': '勺→白盒', 'cup': '杯→绿盒', 'plate': '盘→蓝盒'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

report = dict(as_of=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
              selection='first status=success in each production SOURCE manifest; no outcome substitution',
              authorization='user requested separate demonstrations as videos',
              protocol=dict(isaac_rerun=False, gpu_used=False, source_modified=False,
                            evaluation_protocol_changed=False, extra_camera_created=False,
                            frame_rate=10, playback_speed=1, frame_drops=0,
                            appended_freeze_frames=0, default_recording_remains_off=True),
              tasks={})
for task in ('zhuhai_task1', 'zhuhai_task2'):
    manifest_path = RUN / task / 'source/manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    successes = [row for row in manifest['attempts'] if row['status'] == 'success']
    row = successes[0]
    source = Path(row['data_path'])
    before = sha(source)
    assert before == row['data_sha256']
    receipt_path = RUN / task / 'source/attempt_evidence' / source.stem / 'expert_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    segments = []
    frame_end = 0
    for step in receipt['trace']:
        count = step.get('samples', 0)
        if count:
            segments.append(dict(start_frame=frame_end + 1, end_frame=frame_end + count,
                                 phase=step['phase']))
            frame_end += count
    n = row['observation_count']
    assert len(receipt['motion_commands']) == n - 1
    if frame_end < n - 1:
        segments.append(dict(start_frame=frame_end + 1, end_frame=n - 1,
                             phase=receipt['trace'][-1]['active_phase']))
    output = OUT / f'{task}_source_seed_{row["seed"]}_1x.mp4'
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix('.partial.mp4')
    log_path = output.with_suffix('.ffmpeg.log')
    with log_path.open('w') as log, h5py.File(source, 'r') as hdf:
        assert hdf.attrs['successful_expert'] and hdf.attrs['physics_safety_all_observations_passed']
        assert json.loads(hdf.attrs['metadata'])['task'] == task
        times = hdf['physics_time'][:]
        assert len(times) == n and np.allclose(np.diff(times), .1, atol=1e-7, rtol=0)
        proc = subprocess.Popen([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-n',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '960x620', '-r', '10', '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-threads', '2',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE, stderr=log)
        segment_index = 0
        for frame in range(n):
            if frame:
                while frame > segments[segment_index]['end_frame']:
                    segment_index += 1
                phase = segments[segment_index]['phase']
            else:
                phase = 'initial_observation'
            canvas = Image.new('RGB', (960, 620), (18, 24, 34))
            for name, xy, size in [('cam_head', (0, 100), (640, 480)),
                                    ('cam_left_wrist', (640, 100), (320, 240)),
                                    ('cam_right_wrist', (640, 340), (320, 240))]:
                # finish() trims vision to N action-aligned observations;
                # the final observation is preserved separately, without loss.
                encoded = (hdf[f'vision/{name}/colors'][frame] if frame < n-1
                           else hdf[f'terminal_observation/vision/{name}/colors'][:])
                rgb = Image.open(io.BytesIO(encoded.tobytes())).convert('RGB')
                assert rgb.size == (320, 240)
                canvas.paste(rgb.resize(size, Image.Resampling.NEAREST), xy)
            draw = ImageDraw.Draw(canvas)
            title = '任务1：垃圾分类与投放' if task.endswith('1') else '任务2：餐具分类收纳'
            draw.text((12, 5), f'{title} | seed {row["seed"]} | 正式专家示教', font=FONTS[24], fill='white')
            draw.text((12, 39), f'原速1×  物理时间 {times[frame]:.1f}/{times[-1]:.1f}秒  帧 {frame+1}/{n}', font=FONTS[20], fill='#90d8ff')
            object_label = OBJECTS.get(phase.split(':')[0], '初始状态 / 打开夹爪')
            draw.text((12, 69), f'{object_label} | {phase}', font=FONTS[17], fill='white')
            for xy, label in [((8, 103), '头部相机'), ((648, 103), '左腕相机'), ((648, 343), '右腕相机')]:
                draw.rectangle((xy[0]-3, xy[1]-1, xy[0]+100, xy[1]+27), fill=(18, 24, 34))
                draw.text(xy, label, font=FONTS[20], fill='white')
            draw.text((12, 588), '原始机载RGB回放；未重跑、未删帧、未加定格；专家示教≠学习策略评测', font=FONTS[20], fill='white')
            proc.stdin.write(canvas.tobytes())
            if frame in {0, n//2, n-1}:
                canvas.save(OUT / f'{task}_frame_{frame:04d}.jpg', quality=90)
        proc.stdin.close()
        assert proc.wait() == 0, log_path.read_text()
    check = subprocess.run([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-i', str(temporary),
                            '-map', '0:v:0', '-threads', '2', '-f', 'null', '-', '-progress', 'pipe:1'],
                           text=True, capture_output=True, check=True)
    decoded_frames = int([line[6:] for line in check.stdout.splitlines() if line.startswith('frame=')][-1])
    assert decoded_frames == n
    assert sha(source) == before
    os.rename(temporary, output)
    sizes = [Path(x['data_path']).stat().st_size for x in successes]
    lengths = [(x['observation_count']-1)/10 for x in successes]
    groups = collections.defaultdict(float)
    for segment in segments:
        groups[segment['phase'].split(':')[0]] += (segment['end_frame']-segment['start_frame']+1)/10
    report['tasks'][task] = dict(seed=row['seed'], source=str(source), source_sha256=before,
        manifest_snapshot_sha256=hashlib.sha256(manifest_bytes).hexdigest(), accepted_count=len(successes),
        mean_hdf_bytes=statistics.mean(sizes), total_success_hdf_bytes=sum(sizes),
        duration_range_seconds=[min(lengths), max(lengths)], mean_duration_seconds=statistics.mean(lengths),
        observation_frames=n, action_frames=n-1, physics_duration_seconds=float(times[-1]-times[0]),
        video_duration_seconds=n/10, video=str(output), video_bytes=output.stat().st_size,
        video_sha256=sha(output), decoded_video_frames=decoded_frames,
        source_unchanged=True, phase_segments=segments, phase_group_seconds=dict(groups))
    print(task, n, output.stat().st_size, dict(groups), flush=True)
disk = shutil.disk_usage(ROOT)
raw_estimate = sum(x['mean_hdf_bytes']*600 for x in report['tasks'].values())
current_raw = sum(x['total_success_hdf_bytes'] for x in report['tasks'].values())
report['storage'] = dict(total_bytes=disk.total, used_bytes=disk.used, free_bytes=disk.free,
    estimated_1200_successful_hdf_bytes=raw_estimate, remaining_successful_hdf_bytes=raw_estimate-current_raw,
    estimate_basis='per-task successful SOURCE mean x600; randomized lengths not yet measured; failures/evidence/checkpoints extra',
    dense_train_cache_previous_audit_bytes=4123825795100, cache_cap_bytes=10**12,
    mandatory_free_reserve_bytes=100*1024**3,
    free_after_remaining_successful_hdf_and_full_cache_and_reserve_bytes=disk.free-(raw_estimate-current_raw)-10**12-100*1024**3,
    raw_data_fits=True, full_dense_train_residency_fits=False)
(OUT/'review_receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print('storage', report['storage'], flush=True)
