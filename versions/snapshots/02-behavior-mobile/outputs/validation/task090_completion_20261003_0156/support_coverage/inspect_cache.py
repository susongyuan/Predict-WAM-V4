#!/usr/bin/env python3
"""Read only pre-existing TRAIN JPEG packs; write small visibility-only previews."""
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/home/user/Robotic/project2')
OUT = ROOT / 'outputs/validation/task090_completion_20261003_0156/support_coverage'
DATA = ROOT / 'data/behavior-1k/2026-challenge-demos'
CAMERAS = {'head': 'observation.rgb.zed_link_camera_0', 'left': 'observation.rgb.left_realsense_link_camera_0', 'right': 'observation.rgb.right_realsense_link_camera_0'}
FRAMES = {18000: [220, 240, 260, 280, 300, 320, 420, 480], 18001: [0, 144, 290, 434, 580, 724, 840, 870]}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    states = []
    for ep, frames in FRAMES.items():
        state_path = ROOT / f'data/lowdim_cache_task090/episode_{ep}_state.npy'
        state = np.load(state_path, mmap_mode='r')
        for frame in frames:
            states.append({'episode': ep, 'frame': frame, 'source': str(state_path), 'row_index': frame, 'source_sha256': sha(state_path), 'state61': state[frame].tolist(), 'semantics': 'recorded TRAIN observation.state; not physical outcome or contact evidence'})
        for label, camera in CAMERAS.items():
            base = ROOT / 'data/frame_cache' / camera / f'episode_{ep}'
            meta_path = base.with_suffix('.meta.json')
            meta = json.loads(meta_path.read_text())
            idx = np.load(base.with_suffix('.idx.npy'), mmap_mode='r')
            with base.with_suffix('.jpgpack').open('rb') as pack:
                for frame in frames:
                    assert frame % meta['stride'] == 0
                    i = frame // meta['stride']
                    start, end = int(idx[i]), int(idx[i + 1])
                    pack.seek(start)
                    blob = pack.read(end - start)
                    path = OUT / f'ep{ep}_f{frame:05d}_{label}_cached.jpg'
                    path.write_bytes(blob)
                    assert Image.open(io.BytesIO(blob)).size == (224, 224)
                    rows.append({'episode': ep, 'split': 'train', 'frame': frame, 'camera': camera, 'camera_label': label, 'image': str(path), 'sha256': sha(path), 'image_size': [224, 224], 'provenance': 'existing packed JPEG bytes copied, no video decoding or upsampling', 'rgb_video_source': str(DATA / meta['video_path']), 'cache_metadata': str(meta_path), 'cache_metadata_sha256': sha(meta_path), 'cache_pack': str(base.with_suffix('.jpgpack')), 'cache_pack_sha256_recorded_not_rehashed': meta['sha256'], 'cache_index': str(base.with_suffix('.idx.npy')), 'cache_index_sha256': sha(base.with_suffix('.idx.npy')), 'cache_index_row': i, 'cache_byte_interval': [start, end], 'episode_offset_s': meta['from_timestamp'], 'episode_time_s': frame / 30, 'expected_video_pts_s': meta['from_timestamp'] + frame / 30, 'pts_provenance': 'derived from episode offset and frame/30; existing cache metadata reports maximum PTS error, not a new decode', 'cache_recorded_max_pts_error_s': meta['max_pts_error_s'], 'state_source': str(state_path), 'state_row': frame, 'phase_from_original_skill_window_not_outcome': 'navigation' if frame < (480 if ep == 18000 else 870) else 'first_pick_window_start', 'depth': None, 'physical_success_label': None})
        for part in range(2):
            chosen = frames[part * 4:(part + 1) * 4]
            sheet = Image.new('RGB', (3 * 224, 4 * 250), 'white')
            draw = ImageDraw.Draw(sheet)
            for j, frame in enumerate(chosen):
                for i, label in enumerate(CAMERAS):
                    image = Image.open(OUT / f'ep{ep}_f{frame:05d}_{label}_cached.jpg')
                    sheet.paste(image, (i * 224, j * 250 + 26))
                    draw.text((i * 224 + 4, j * 250 + 6), f'ep{ep} f{frame:05d} {label} cache224', fill='black')
            sheet.save(OUT / f'ep{ep}_cache_contact_sheet_{part + 1}.jpg', quality=95)
    (OUT / 'cache_manifest.json').write_text(json.dumps({'source': 'local_existing_train_cache', 'selection_basis': 'Original first navigation windows [0,480), [0,870), plus pick-window starts. ep18000 samples concentrate around target entering view in previously inspected 270 frame; ep18001 uses navigation coverage fractions. No outcomes, dev/report records, object GT or fitted thresholds.', 'frames_by_episode': FRAMES, 'new_video_frames_decoded': 0, 'samples': rows}, indent=2) + '\n')
    (OUT / 'states.json').write_text(json.dumps(states, indent=2) + '\n')
    print(json.dumps({'cache_images': len(rows), 'state_rows': len(states), 'new_video_frames_decoded': 0, 'output': str(OUT)}))

if __name__ == '__main__':
    main()
