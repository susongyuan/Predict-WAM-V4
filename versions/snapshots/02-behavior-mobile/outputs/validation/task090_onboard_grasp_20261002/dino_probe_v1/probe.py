#!/usr/bin/env python3
"""Two-frame, read-only DINO prototype probe; no model imports or image decoding."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import resource
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path('/home/user/Robotic/project2')
OUT = Path(__file__).resolve().parent
VIEW = 'observation.rgb.zed_link_camera_0'
CACHE = ROOT / 'data/dino_cache_task090'
start = time.monotonic()

def small_json(path):
    assert path.stat().st_size < 128 * 1024, f'oversized metadata: {path}'
    return json.loads(path.read_text())

def header(path):
    with path.open('rb') as f:
        version = np.lib.format.read_magic(f)
        assert version in ((1, 0), (2, 0)), version
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(f)
        assert not fortran and dtype == np.dtype('<u2')
        return shape, dtype, f.tell()

cfg = small_json(CACHE / 'cache_config.json')
enc = small_json(ROOT / cfg['encoder'] / 'config.json')
assert cfg['feat_layers'] == [-12, -8, -4] and cfg['stride'] == 2
assert (enc['patch_size'], enc['num_register_tokens'], enc['hidden_size']) == (16, 4, 1024)
assert cfg['tokens_per_image'] == 201
records = []

def read_frame(ep, frame):
    base = CACHE / VIEW
    stem = f'episode_{ep}'
    meta = small_json(base / f'{stem}.meta.json')
    srcpath = ROOT / 'data/frame_cache' / VIEW / f'{stem}.meta.json'
    src = small_json(srcpath)
    assert hashlib.sha256(srcpath.read_bytes()).hexdigest() == meta['source_frame_meta_sha256']
    assert meta['complete'] and meta['episode'] == ep and meta['view'] == VIEW
    assert meta['stride'] == 2 and meta['feat_layers'] == cfg['feat_layers']
    assert src['complete'] and src['size'] == 224 and src['stride'] == 2
    assert src['episode_index'] == ep and src['camera'] == VIEW and frame < src['length']
    indices = np.load(base / f'{stem}.frames.npy', mmap_mode='r', allow_pickle=False)
    assert indices.dtype == np.int64 and indices.ndim == 1
    assert len(indices) == meta['rows'] and np.all(np.diff(indices) > 0) and np.all(indices % 2 == 0)
    row = int(np.searchsorted(indices, frame))
    assert row < len(indices) and int(indices[row]) == frame, f'uncached frame {ep}/{frame}'
    path = base / f'{stem}.feat.npy'
    shape, dtype, offset = header(path)
    assert list(shape) == meta['feat_shape'] == [len(indices), 3, 201, 1024]
    assert path.stat().st_size == offset + int(np.prod(shape)) * dtype.itemsize
    layers = []
    for layer in range(3):
        byte = offset + ((row * 3 + layer) * 201 + 5) * 1024 * 2
        bits = np.memmap(path, mode='r', dtype=dtype, offset=byte, shape=(196, 1024))
        values = (bits.astype(np.uint32) << 16).view(np.float32)
        assert np.isfinite(values).all()
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        assert np.all(norms > 0)
        layers.append(values / norms)
        bits._mmap.close()
    records.append({'episode': ep, 'frame': frame, 'cache_row_zero_based': row,
                    'neighbor_frame_indices': indices[max(0, row-1):row+2].tolist(),
                    'shape': list(shape), 'source_frame_meta_hash_verified': True,
                    'feature_payload_bytes_read': 3 * 196 * 1024 * 2})
    indices._mmap.close()
    return np.stack(layers)

# Fit only the requested full bounding boxes, never the held-out boxes or tiny color-fit sub-ROIs.
annotations = small_json(ROOT / 'configs/task090_perception_annotations.json')
pos = next(a['bbox_xyxy'] for a in annotations['annotations']
           if (a['episode'], a['frame'], a['camera'], a['label']) == (18000, 480, VIEW, 'trash_can'))
neg = next(a['bbox_xyxy'] for a in annotations['negative_rois']
           if (a['episode'], a['frame'], a['camera']) == (18000, 480, VIEW))
assert pos == [220, 540, 419, 720] and neg == [57, 342, 147, 425]
# Native 720x720 coordinate domain is supplied by the task. Cache writer uses full-frame AREA resize.
rows, cols = np.divmod(np.arange(196), 14)
x, y = (cols + .5) * (720 / 14), (rows + .5) * (720 / 14)
def mask(box):
    a, b, c, d = box
    return (x >= a) & (x < c) & (y >= b) & (y < d)

def proto(features, box):
    selected = mask(box)
    assert selected.any()
    p = features[:, selected].mean(axis=1)
    return p / np.linalg.norm(p, axis=-1, keepdims=True)

train = read_frame(18000, 480)
positive, negative = proto(train, pos), proto(train, neg)
# All method choices are fixed before reading the held-out frame; no layer selection or threshold fitting.
held = read_frame(18001, 870)
cos_pos = np.einsum('lpd,ld->lp', held, positive).mean(axis=0)
cos_neg = np.einsum('lpd,ld->lp', held, negative).mean(axis=0)
difference = cos_pos - cos_neg
candidates = {'assistant_observed_bin': [485, 548, 720, 720], 'incorrect_tall_cabinet': [628, 10, 720, 137]}

def patch(i):
    i = int(i)
    return {'patch_index': i, 'row_col': [int(rows[i]), int(cols[i])],
            'native_center_xy': [round(float(x[i]), 3), round(float(y[i]), 3)],
            'positive_cosine': round(float(cos_pos[i]), 6),
            'negative_cosine': round(float(cos_neg[i]), 6),
            'difference': round(float(difference[i]), 6),
            'in_candidates': [name for name, box in candidates.items() if mask(box)[i]]}

result = {'status': 'ok', 'method': 'L2-normalize patches, mean ROI then L2-normalize prototypes per layer; equal mean across all three cached layers; difference = positive cosine - negative cosine',
          'layers': cfg['feat_layers'], 'spatial_layout': {'grid': [14,14], 'patch_size_cached_pixels': 16,
          'dropped_prefix_tokens': 5, 'native_size_wh': [720,720], 'native_patch_pitch': 720/14,
          'native_size_provenance': 'task-supplied coordinate domain; no independent original-image header verification',
          'coordinate_mapping': 'row-major patch centers; full-frame AREA resize to 224x224, no crop; xyxy upper bounds exclusive'},
          'fit': {'episode': 18000, 'frame': 480, 'positive_roi': pos, 'negative_roi': neg,
                  'positive_patch_count': int(mask(pos).sum()), 'negative_patch_count': int(mask(neg).sum())},
          'cache_reads': records,
          'top5_positive': [patch(i) for i in np.argsort(-cos_pos, kind='stable')[:5]],
          'top5_difference': [patch(i) for i in np.argsort(-difference, kind='stable')[:5]],
          'candidates': {}}
for name, box in candidates.items():
    indices = np.flatnonzero(mask(box))
    best = int(indices[np.argmax(difference[indices])])
    result['candidates'][name] = {'bbox_xyxy': box, 'patch_count': len(indices),
        'positive_mean': round(float(cos_pos[indices].mean()), 6),
        'positive_max': round(float(cos_pos[indices].max()), 6),
        'difference_mean': round(float(difference[indices].mean()), 6),
        'difference_max': round(float(difference[indices].max()), 6),
        'best_difference_global_rank': int(np.count_nonzero(difference > difference[best])) + 1,
        'best_difference_patch': patch(best)}
result['global_difference_peak_in_bin'] = bool(mask(candidates['assistant_observed_bin'])[np.argmax(difference)])
result['global_positive_peak_in_bin'] = bool(mask(candidates['assistant_observed_bin'])[np.argmax(cos_pos)])
result['resources'] = {'cpu_affinity_count': len(os.sched_getaffinity(0)),
    'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 3),
    'wall_seconds': round(time.monotonic()-start, 3), 'virtual_memory_limit_gib': 3}
result['limitations'] = ['Single training frame and single held-out assistant-observed box, not human or physical-success labels.',
    'TRAIN18001 is held out from this prototype only, not claimed to be a dataset validation split.',
    'Coarse patch-center membership is not segmentation, identity, metric geometry, graspability, or physical success.',
    'No thresholds fitted; no held-out prototype edits; no model load, GPU, image re-encoding, or cache payload hashing.']
path = OUT / 'result.json'
with path.open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
print('RESULT_PATH=' + str(path))
