#!/usr/bin/env python3
"""Fit bounded TRAIN appearance proposals and TRAIN programs, using CPU only.

Assistant fit ROIs are explicit annotations, NOT human ground truth. Rechecks are
resubstitution checks, not detector accuracy/admission. Never modifies targets,
stats, training code or data. No simulator, torch, downloads or GPU needed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/validation/task090_preparation_20260930/system_runtime_v2"
ANNOTATIONS = ROOT / "configs/task090_perception_annotations.json"
SPLIT = ROOT / "data/behavior-1k/2026-challenge-demos/split_task_090.json"
SEGMENTS = ROOT / "data/behavior-1k/2026-challenge-demos/skill_segments_task090.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def read_annotations():
    a = json.loads(ANNOTATIONS.read_text())
    assert a['task_id'] == 90 and a['split'] == 'train'
    assert a['annotation_author'] == 'assistant; not a human annotator'
    for field in ('uses_validation', 'uses_dev', 'uses_test', 'uses_hidden_truth'):
        assert a[field] is False
    manifest_path = ROOT / a['source_manifest']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['task_id'] == 90 and manifest['split'] == 'train'
    train = set(json.loads(SPLIT.read_text())['train_episode_indices'])
    lookup = {(s['episode'], s['frame'], s['camera']): s for s in manifest['samples']}
    rows = []
    for ann in a['annotations'] + a['negative_rois']:
        s = lookup[ann['episode'], ann['frame'], ann['camera']]
        assert s['episode'] in train and s['split'] == 'train'
        assert s['camera'] == 'observation.rgb.zed_link_camera_0'
        path = Path(s['rgb'])
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        for box in [ann['bbox_xyxy']] + ann.get('fit_rois_xyxy', []):
            x0, y0, x1, y1 = box
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        for x0, y0, x1, y1 in ann.get('fit_rois_xyxy', []):
            bx0, by0, bx1, by1 = ann['bbox_xyxy']
            assert bx0 <= x0 < x1 <= bx1 and by0 <= y0 < y1 <= by1
        record = {**ann, 'rgb': str(path), 'rgb_sha256': sha(path),
                  'image_size': [w, h], 'source': s['source'], 'decoded_pts': s['decoded_pts'],
                  'split': 'train', 'annotation_author': a['annotation_author']}
        rows.append((record, image))
    provenance = {'schema': a['schema'], 'annotation_author': a['annotation_author'],
                  'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
                  'annotations': str(ANNOTATIONS), 'annotations_sha256': sha(ANNOTATIONS),
                  'split': str(SPLIT), 'split_sha256': sha(SPLIT),
                  'rows': [r for r, _ in rows], 'uses_dev_test_hidden_truth': False}
    return rows, provenance


def hue_boxes(pixels, label):
    """Quantiles of explicitly selected object pixels; red uses circular hue."""
    p = pixels.astype(float)
    if label == 'half_pomegranate':
        p[:, 0] = (p[:, 0] + 90) % 180 - 90
    lo = np.floor(np.quantile(p, .02, axis=0) - [3, 18, 18]).astype(int)
    hi = np.ceil(np.quantile(p, .98, axis=0) + [3, 18, 18]).astype(int)
    lo[1:] = np.clip(lo[1:], 0, 255)
    hi[1:] = np.clip(hi[1:], 0, 255)
    if label == 'trash_can':
        # Black has unstable hue/saturation; only fit the observed brightness bound.
        return [[[0, 0, 0], [179, 255, int(hi[2])]]]
    spans = [(max(0, lo[0]), min(179, hi[0]))]
    if lo[0] < 0:
        spans.append((180 + lo[0], 179))
    if hi[0] > 179:
        spans.append((0, hi[0] - 180))
    return [[[int(l), int(lo[1]), int(lo[2])], [int(h), int(hi[1]), int(hi[2])]]
            for l, h in spans if l <= h]


def color_mask(image, boxes):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = np.zeros(image.shape[:2], np.uint8)
    for lo, hi in boxes:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def fit_appearance(rows, provenance):
    cfg = {'task_id': 90, 'split': 'train', 'annotation_author': provenance['annotation_author'],
           'calibration_id': 'task090-assistant-appearance-' + provenance['annotations_sha256'][:12],
           'source_manifest_sha256': provenance['manifest_sha256'],
           'annotations_sha256': provenance['annotations_sha256'],
           'classes': {}, 'max_candidates_per_class': 8,
           'depth_max_m': 6., 'min_depth_pixels': 6, 'min_depth_fraction': .7,
           'metric_threshold_note': 'Depth gates are conservative sensor hygiene, not fitted metric object calibration.',
           'fit_method': '2/98 percentile HSV on assistant pixel ROIs plus [3,18,18] margin; black V only; finite component gates from labelled boxes.',
           'point_semantics': 'observed_surface_sample', 'metric_calibration_admitted': False,
           'grasp_reference_valid': False, 'relations_admitted': False,
           'unsupported': ['furniture_sink', 'floors', 'countertop', 'object_hand_tracks', 'relations'],
           'runtime_fixed_roi': False, 'accuracy_validated': False}
    for label in ('half', 'half_pomegranate', 'trash_can'):
        selected = [(a, im) for a, im in rows if a.get('label') == label]
        pixels = []
        for ann, im in selected:
            hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            pixels.extend(hsv[y0:y1, x0:x1].reshape(-1, 3) for x0,y0,x1,y1 in ann['fit_rois_xyxy'])
        boxes = hue_boxes(np.concatenate(pixels), label)
        areas, aspects, fills = [], [], []
        for ann, im in selected:
            x0, y0, x1, y1 = ann['bbox_xyxy']
            mask = color_mask(im, boxes)[y0:y1, x0:x1]
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            assert n > 1, f'No fitted component in {label} annotation'
            x, y, w, h, area = stats[1:][np.argmax(stats[1:, cv2.CC_STAT_AREA])]
            areas.append(float(area) * (720 / im.shape[1]) ** 2)
            aspects.append(float(h / w)); fills.append(float(area / (w*h)))
        cfg['classes'][label] = {'physical_category': selected[0][0]['physical_category'],
            'hsv_boxes': boxes, 'area_px_720': [max(12, int(min(areas)*.2)), int(max(areas)*2.5)],
            'aspect_h_over_w': [min(aspects)*.4, max(aspects)*2.5],
            'fill_min': min(fills)*.5, 'annotation_count': len(selected),
            'fit_pixel_count': int(sum(len(p) for p in pixels))}
    save(ROOT / 'configs/task090_perception.json', cfg)
    save(OUT / 'annotation_sources.json', provenance)
    return cfg


def build_programs():
    from harness.vtt_monitor import canonical_object
    data = json.loads(SEGMENTS.read_text())
    train = set(json.loads(SPLIT.read_text())['train_episode_indices'])
    groups, programs = defaultdict(list), defaultdict(list)
    for ep in sorted(train):
        record = data['episodes'][str(ep)]
        assert record['split'] == 'train'
        program = []
        for position, s in enumerate(record['segments']):
            assert len(s['objects']) == 1, 'Ambiguous TRAIN objects require review'
            objects = tuple(canonical_object(x) for x in s['objects'][0])
            if s['skill'] == 'place on next to':
                assert objects == ('trash_can', 'floors', 'furniture_sink')
            key = (s['skill'], objects)
            groups[key].append((s['end']-s['start'], position, s['skill_type']))
            program.append(key)
        programs[tuple(program)].append(ep)
    top = max(programs, key=lambda k: len(programs[k]))
    skills = []
    for (skill, objects), rs in sorted(groups.items()):
        skills.append({'skill': skill, 'objects': list(objects), 'count': len(rs),
                       'median_frames': int(np.median([r[0] for r in rs])),
                       'median_position': float(np.median([r[1] for r in rs])),
                       'skill_type': Counter(r[2] for r in rs).most_common(1)[0][0]})
    source = {'segments': str(SEGMENTS), 'segments_sha256': sha(SEGMENTS),
              'split': str(SPLIT), 'split_sha256': sha(SPLIT), 'train_episodes': sorted(train),
              'dataset_revision': data['dataset_revision'], 'uses_validation': False,
              'top_program_episodes': programs[top]}
    top_rows = [{'skill': s, 'objects': list(o)} for s, o in top]
    skill_program = {'task_id': 90, 'task': 'composting_waste', 'source': source,
                     'skill_classes': data['classes'],
                     'skills': skills, 'top_program': top_rows, 'top_program_count': len(programs[top]),
                     'alias_note': 'half is the existing TRAIN banana token; physical_category=half_banana'}
    medians = {(s['skill'], tuple(s['objects'])): s['median_frames'] for s in skills}
    fixed = {'task_id': 90, 'task': 'composting_waste', 'source': source,
             'note': 'Modal TRAIN sequence, no steps removed. Budgets are timeouts, never completion evidence.',
             'program': [{**s, 'budget_frames': medians[s['skill'], tuple(s['objects'])]} for s in top_rows]}
    save(ROOT / 'configs/task090_skill_program.json', skill_program)
    save(ROOT / 'configs/task090_fixed_program.json', fixed)
    return {'train_episodes': len(train), 'top_program_count': len(programs[top]), 'steps': len(top)}


def main():
    cv2.setNumThreads(1)
    rows, provenance = read_annotations()
    cfg = fit_appearance(rows, provenance)
    programs = build_programs()
    save(OUT / 'build_result.json', {'status': 'cpu_generated_not_admitted',
        'calibration_id': cfg['calibration_id'], 'annotation_count': len(rows), **programs,
        'gpu_started': False, 'postconditions_admitted': False, 'complete_task_validated': False})
    print(json.dumps({'output': str(OUT), 'classes': list(cfg['classes']), **programs}))


if __name__ == '__main__':
    main()
