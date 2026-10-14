#!/usr/bin/env python3
"""Single CPU sparse decode of one existing TRAIN head RGB video frame."""
import hashlib
import json
import os
import resource
import time
from fractions import Fraction
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import av
from PIL import Image

ROOT = Path('/home/user/Robotic/project2')
OUT = ROOT / 'outputs/validation/task090_completion_20261003_0156/support_coverage'
SOURCE = ROOT / 'data/behavior-1k/2026-challenge-demos/videos/observation.rgb.zed_link_camera_0/chunk-090/file-000.mp4'

def main():
    # Chosen only after cache showed a full 2D bin outline and neighboring floor.
    # Narrow purpose: check the visible lower boundary at original recording resolution.
    ep, frame_idx = 18001, 724
    exact_time = Fraction(3425 + frame_idx, 30)
    t0, cpu0 = time.monotonic(), time.process_time()
    decoded = []
    with av.open(str(SOURCE)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'NONE'
        stream.codec_context.thread_count = 1
        target_pts = exact_time / stream.time_base
        assert target_pts.denominator == 1
        container.seek(int(target_pts), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            actual = frame.pts * frame.time_base
            decoded.append({'pts': frame.pts, 'time_base': str(frame.time_base), 'pts_s': float(actual)})
            if actual < exact_time:
                continue
            assert actual == exact_time, (actual, exact_time)
            path = OUT / f'ep{ep}_f{frame_idx:05d}_head_native.png'
            image = frame.to_image()
            image.save(path)
            row = {'episode': ep, 'split': 'train', 'frame': frame_idx, 'camera': 'observation.rgb.zed_link_camera_0', 'source': str(SOURCE), 'source_revision': '4f50b44796641a4d526a19d9aeadc8aa51e2f2c2', 'source_bytes': SOURCE.stat().st_size, 'image': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size': list(image.size), 'episode_time_s': frame_idx/30, 'episode_offset_s': 3425/30, 'requested_video_pts_s': float(exact_time), 'actual_video_pts_s': float(actual), 'actual_pts': frame.pts, 'time_base': str(frame.time_base), 'pts_error_s': float(actual - exact_time), 'state_source': str(ROOT / f'data/lowdim_cache_task090/episode_{ep}_state.npy'), 'state_row': frame_idx, 'phase': 'original_navigation_window_[0,870)_not_outcome', 'depth': None, 'physical_success_label': None}
            break
        else:
            raise RuntimeError('Requested exact frame not found')
    log = {'samples': [row], 'scope': 'one native-resolution source RGB frame; not a new recording, depth extraction, simulation or physical label', 'selection_basis': 'cache coverage inspection only, no dev/report/GT outcomes', 'video_open_calls': 1, 'seek_calls': 1, 'cpu_codec_threads': 1, 'decoded_frames_including_keyframe_preroll': decoded, 'saved_native_frame_count': 1, 'gpu_used': False, 'download': False, 'elapsed_s': time.monotonic()-t0, 'cpu_s': time.process_time()-cpu0, 'process_max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (OUT / 'native_manifest.json').write_text(json.dumps(log, indent=2) + '\n')
    print(json.dumps(log))

if __name__ == '__main__':
    main()
