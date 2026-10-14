#!/usr/bin/env python3
"""Two frozen TRAIN RGB-D examples; appearance candidates, never physical outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.instance_surface import SurfaceThresholds
from harness.object_perception import Camera, camera_from_pose
from harness.robot_self_geometry import RobotSelfGeometry
from harness.task090_dino_localizer import Task090DinoLocalizer
from scripts.build_task090_dino_localizer import cached_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    artifact = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz'
    localizer = Task090DinoLocalizer(artifact)
    source = ROOT / 'outputs/validation/task090_first_grasp_evidence_20261001/extracted/frames.jsonl'
    samples = {(r['episode'], r['frame_index']): r for r in map(json.loads, source.open())}
    fit = cached_frame(18000, 480)
    pos, neg = localizer.scores(fit)
    row, col = np.divmod(np.arange(196), 14)
    x, y = (col + .5) * 720 / 14, (row + .5) * 720 / 14
    def inside(box):
        return (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])
    positive = inside(localizer.meta['positive_roi'])
    negative = inside(localizer.meta['negative_roi'])
    # Fixed source-only separator, deliberately not claimed to be calibrated presence.
    positive_min = float((pos[positive].min() + pos[negative].max()) / 2)
    margin_min = float((pos - neg)[positive].min() / 2)
    thresholds = SurfaceThresholds(positive_min=positive_min, margin_min=margin_min,
        depth_min_m=.05, depth_max_m=3., neighbor_distance_m=.02,
        neighbor_color_linf=24., min_component_pixels=64)
    settings = dict(thresholds=thresholds.__dict__, provenance=dict(
        score_source=str(artifact.relative_to(ROOT)) + '#' + localizer.meta['artifact_sha256'],
        threshold_source='TRAIN18000/f480 positive/negative ROI midpoint and half-positive margin; '
            '2cm/24RGB/64pixel local surface heuristics are NOT calibrated physical error bounds',
        alignment_source='frozen runtime full-frame 224 INTER_AREA to 14x14 spatial tokens; native RGB-D'),
        use_robot_self_geometry=True)
    (args.output / 'surface_refinement.json').write_text(json.dumps(settings, indent=2) + '\n')
    geometry = RobotSelfGeometry()
    provenance = {**settings['provenance'], 'robot_self_mask_source': geometry.version}
    rows, started = [], time.monotonic()
    for episode, frame in ((18000, 480), (18001, 870)):
        r = samples[(episode, frame)]
        rgb = np.asarray(Image.open(r['rgb']))
        depth = np.load(r['depth_m'], allow_pickle=False)
        camera = Camera(*camera_from_pose(r['camera_pose7']), rgb.shape[1], rgb.shape[0])
        features = fit if (episode, frame) == (18000, 480) else cached_frame(episode, frame)
        result = localizer.surfaces(rgb, depth, camera, features, thresholds=thresholds,
                                   provenance=provenance, self_geometry=geometry, state=r['state61'])
        Image.fromarray(result['labels'].astype(np.uint16)).save(args.output / f'{episode}_{frame}_components.png')
        candidates = [{k: v for k, v in c.items() if k not in ('cloud', 'pixel_uv')}
                      for c in result['candidates']]
        rows.append(dict(episode=episode, frame=frame, split='train', role='fit' if episode == 18000 else 'heldout',
            candidates=candidates, diagnostics=result['diagnostics'], status=result['status'],
            single_surface=len(candidates) == 1, instance_identity_verified=False, physical_success_verified=False,
            rgb_sha256=hashlib.sha256(Path(r['rgb']).read_bytes()).hexdigest(),
            depth_sha256=hashlib.sha256(Path(r['depth_m']).read_bytes()).hexdigest()))
    sources = ['harness/instance_surface.py', 'harness/task090_dino_localizer.py',
               'harness/task090_observation.py', 'scripts/check_task090_surface_refinement.py']
    result = dict(status='offline_train_candidates_only', rows=rows, elapsed_s=time.monotonic() - started,
        thresholds=settings, geometry_sources=geometry.source_sha256,
        code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        gpu_runs=0, simulation_runs=0, physical_success_labels_used=0, admitted=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('geometry_sources', 'code_sha256', 'thresholds')}, indent=2))


if __name__ == '__main__':
    main()
