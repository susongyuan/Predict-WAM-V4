#!/usr/bin/env python3
"""Replay existing TRAIN sensor samples through the unadmitted measurement producer."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.task090_observation import Task090ObservationProducer


def main():
    source = ROOT / 'outputs/validation/task090_first_grasp_evidence_20261001/extracted/frames.jsonl'
    output = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/main_probe_v1'
    output.mkdir(parents=True, exist_ok=False)
    calibration = ROOT / 'configs/task090_perception_geometry_v2.json'
    producer = Task090ObservationProducer(calibration)
    counts, measurements, floor, sparse, n = Counter(), 0, 0, 0, 0
    started = time.monotonic()
    with (output / 'measurements.jsonl').open('x') as result:
        for line in source.open():
            row = json.loads(line)
            assert row['split'] == 'train' and row['episode'] in (18000,18001)
            assert row['physical_success_label'] is None
            obs = {'robot_r1::robot_r1:zed_link:Camera:0::rgb': np.asarray(Image.open(row['rgb'])),
                   'robot_r1::robot_r1:zed_link:Camera:0::depth_linear': np.load(row['depth_m'],allow_pickle=False),
                   'robot_r1::cam_rel_poses': np.asarray(row['camera_pose7'])}
            measured = producer.update(obs,np.asarray(row['state61']),episode=row['episode'],step=row['frame_index'])
            measured['source_frame'] = dict(episode=row['episode'],frame=row['frame_index'],
                rgb=row['rgb'], depth=row['depth_m'], physical_success_label=None)
            result.write(json.dumps(measured,ensure_ascii=False)+'\n')
            n += 1; counts[measured['reason']] += 1
            measurements += measured['status']=='measurements_available'
            floor += bool(measured.get('detail',{}).get('floor_observed'))
            sparse += bool(measured.get('detail',{}).get('sparse_gap'))
    summary = dict(status='cpu_sensor_measurements_not_admission', source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256((ROOT/'harness/task090_observation.py').read_bytes()).hexdigest(),
        calibration=str(calibration), calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        samples=n, episodes=[18000,18001], measurements_available=measurements, floor_observed=floor,
        sparse_measurement_frames=sparse, reasons=dict(counts), elapsed_seconds=time.monotonic()-started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        uses_dev=False, uses_test=False, physical_labels_available=0, admitted=False,
        limitations=['2fps samples are not continuous 30Hz evidence',
                      'appearance candidate identity not certified',
                      'visible surface clearance is not whole-object support clearance'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
