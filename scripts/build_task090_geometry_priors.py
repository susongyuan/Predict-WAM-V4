#!/usr/bin/env python3
"""Add explicit static-model size priors, not live simulator state, to Task90 proposals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPECS = {'half': ('half_banana', 'xytkre'), 'half_pomegranate': ('half_pomegranate', 'vqybii'),
         'trash_can': ('trash_can', 'ifzxzj')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--floor-prior', action='store_true', help='require observed floor and a compatible upright-bin height')
    args = parser.parse_args()
    source = ROOT/'configs/task090_perception.json'
    out = ROOT/('configs/task090_perception_floor_v3.json' if args.floor_prior else 'configs/task090_perception_geometry_v2.json')
    if out.exists():
        raise SystemExit(f'Refusing overwrite: {out}')
    cfg = json.loads(source.read_text())
    cfg['calibration_id'] = cfg['calibration_id'] + (':static_floor_v3' if args.floor_prior else ':static_geometry_v2')
    cfg['base_appearance_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    cfg['geometry_scope'] = {
        'source': 'static Task90 asset metadata, no live object state or pose',
        'scale_assumption': [1.0, 1.0, 1.0],
        'extent_rule': '98th-2nd percentile observed spans; upper bound 1.2 times native size/diagonal',
        'vertical_rule': 'require observed vertical relief >=0.2 of minimum fruit extent or bin body height',
        'boundary': 'Wide proposal sanity only; assumed upright uncarried bin; not identity, center, hand or relation admission. No dev pixels used to fit thresholds.'}
    for label, (category, model) in SPECS.items():
        path = ROOT/f'data/omnigibson/behavior-1k-assets/objects/{category}/{model}/misc/metadata.json'
        meta = json.loads(path.read_text())
        size = np.asarray(meta['bbox_size'], float)
        if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError('Invalid static bbox: '+str(path))
        body = np.asarray(meta['link_bounding_boxes']['base_link']['visual']['axis_aligned']['extent'], float)
        prior = {'asset': str(path.relative_to(ROOT)), 'asset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'native_bbox_m': size.tolist(), 'max_axis_span_m': float(1.2*np.linalg.norm(size)),
                 'min_vertical_span_m': float(.2*(body[2] if label == 'trash_can' else size.min()))}
        if label == 'trash_can':
            prior['max_horizontal_axis_span_m'] = float(1.2*np.linalg.norm(body[:2]))
            if args.floor_prior:
                prior['max_height_above_floor_m'] = float(size[2])
                cfg['geometry_scope']['bin_floor_rule'] = 'Navigation-only upright floor-bin prior; observed RGB-D ground plane near base footprint, plus 3*floor-IQR noise allowance. Not a carried-object or outcome detector.'
        cfg['classes'][label]['geometry_prior'] = prior
    with out.open('x') as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps({'written': str(out), 'geometry_scope': cfg['geometry_scope'],
                      'priors': {k:v['geometry_prior'] for k,v in cfg['classes'].items()}}, indent=2))


if __name__ == '__main__':
    main()
