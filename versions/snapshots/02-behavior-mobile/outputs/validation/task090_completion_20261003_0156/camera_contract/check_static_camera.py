#!/usr/bin/env python3
"""Read one static robot camera on CPU; no simulator or scene state."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PXR_WORK_THREAD_LIMIT'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import json
from pathlib import Path
import resource
import sys
import time
from pxr import Usd, UsdGeom

root = Path('/home/user/Robotic/project2')
out = Path(__file__).resolve().parent
asset = root / 'data/omnigibson/omnigibson-robot-assets/models/r1pro/usd/r1pro.usda'
prior_path = out.parent / 'self_geometry_usd/extraction.json'
prior = json.loads(prior_path.read_text())
start = time.monotonic()
assert asset.stat().st_size == prior['source_files'][str(asset)]['bytes']
stage = Usd.Stage.OpenMasked(str(asset), Usd.StagePopulationMask(['/r1pro']), Usd.Stage.LoadNone)
rows = []
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Camera) or 'zed_link' not in str(prim.GetPath()):
        continue
    camera = UsdGeom.Camera(prim)
    attrs = {key: prim.GetAttribute(key).Get() for key in (
        'focalLength', 'horizontalAperture', 'verticalAperture',
        'horizontalApertureOffset', 'verticalApertureOffset', 'projection')}
    f, ha, va = (attrs[k] for k in ('focalLength', 'horizontalAperture', 'verticalAperture'))
    rows.append({'prim': str(prim.GetPath()), **attrs,
                 'clippingRange': list(camera.GetClippingRangeAttr().Get()),
                 'raw_fx_720': 720 * f / ha, 'raw_fy_720': 720 * f / va,
                 'centered_cx_720': 360.0, 'centered_cy_720': 360.0,
                 'centered_principal_point_applicable': attrs['horizontalApertureOffset'] == attrs['verticalApertureOffset'] == 0})
assert len(rows) == 1, f'Expected one head camera, found {len(rows)}'
result = {'status': 'static_camera_read_not_runtime_admission', 'source': str(asset),
          'source_sha256_from_prior': prior['source_files'][str(asset)]['sha256'],
          'source_rehashed': False, 'meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage),
          'up_axis': str(UsdGeom.GetStageUpAxis(stage)), 'cameras': rows,
          'harness_K_720': {'fx': 306.0, 'fy': 306.0, 'cx': 360.0, 'cy': 360.0},
          'runtime_aperture_resolution_overrides_checked': False,
          'gpu_or_simulator_used': False, 'admitted': False,
          'resources': {'python': sys.executable, 'pid': os.getpid(),
                        'seconds': time.monotonic()-start,
                        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}
with (out/'static_camera.json').open('x') as file:
    json.dump(result, file, indent=2)
    file.write('\n')
print(json.dumps(result, indent=2))
