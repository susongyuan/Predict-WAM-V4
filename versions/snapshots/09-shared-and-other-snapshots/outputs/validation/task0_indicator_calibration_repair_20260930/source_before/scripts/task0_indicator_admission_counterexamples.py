#!/usr/bin/env python3
"""CPU-only counterexamples for the unadmitted Task0 indicator; no policy changes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from harness.task0_indicator import classify, _synthetic


def main():
    cv2.setNumThreads(1)
    out = ROOT / 'outputs/validation/task0_indicator_admission_20260930'
    out.mkdir(parents=True, exist_ok=True)
    dest = out / 'synthetic_counterexamples.json'
    if dest.exists():
        raise SystemExit(f'Refusing overwrite: {dest}')

    # No radio or bound target is present: just an unrelated green disc on black.
    unrelated = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(unrelated, (35, 35), 7, (30, 210, 40), -1)
    # An OFF-looking target plus a second, unrelated green component elsewhere.
    mixed = _synthetic((215, 25, 20))
    cv2.rectangle(mixed, (5, 5), (45, 45), (15, 15, 15), -1)
    cv2.circle(mixed, (25, 25), 7, (30, 210, 40), -1)

    cases = []
    for name, rgb, reason in [
        ('unrelated_green_disc', unrelated, 'No radio or target binding exists in the synthetic input.'),
        ('off_target_plus_unrelated_green', mixed, 'Conflicting candidate colours; green is not target-bound.'),
        ('missing_image', None, 'Missing image must remain unknown.'),
    ]:
        verdict, info = classify(rgb)
        cases.append({'name': name, 'verdict': verdict, 'info': info,
                      'expected_admission': 'unknown', 'reason': reason})

    fit_path = ROOT / 'scripts/fit_task0_indicator.py'
    spec = importlib.util.spec_from_file_location('fit_task0_indicator_audit', fit_path)
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    params = {'min_area': 12, 'ring_green': 0.35, 'ring_red': 0.6}
    sparse = [{'ep': 0, 'frame': f, 'truth': 'off', 'g': [(50, 1.0)], 'r': []}
              for f in (0, 100, 200)]
    sparse_result = fit.episode_level(sparse, params)
    assert cases[0]['verdict'] == 'on'
    assert cases[1]['verdict'] == 'on'
    assert cases[2]['verdict'] == 'unknown'
    assert sparse_result['on_run_before'] == 1

    report = {
        'scope': 'CPU synthetic counterexamples only; no real frames, no dataset accuracy estimate',
        'admitted': False,
        'policy_modified': False,
        'gpu_used': False,
        'cases': cases,
        'sparse_frame_counterexample': {
            'frame_indices': [0, 100, 200], 'assumed_stride': 2,
            'result': sparse_result,
            'finding': 'Existing episode_level counts separated observations as a consecutive run; it does not check frame gaps.',
        },
        'claimed_fit_exists': (ROOT / 'outputs/validation/task0_indicator_20260928/fit.json').exists(),
        'label_boundary': 'Existing fit script uses first positive next.reward as an irreversible state boundary. This needs independent state/time-alignment validation and does not establish later ON persistence.',
        'next_requirements': [
            'Bind the detected indicator to an observed target object, reject conflicting/unbound blobs.',
            'Reject stale/repeated frames and enforce actual frame/timestamp continuity in any temporal evidence gate.',
            'Establish independent ON/OFF labels and alignment before reporting real-data accuracy.',
            'Fit only on TRAIN and freeze parameters before VALIDATION; do not use development failures to tune thresholds.',
            'Do not wire this unadmitted detector into candidate-HOLD completion or success memory.',
        ],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (ROOT / 'harness/task0_indicator.py', fit_path, Path(__file__))},
    }
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
