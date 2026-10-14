#!/usr/bin/env python3
"""Offline DINO localizer replay over the 601-frame onboard record of the 35k/seed1 dev run.

Diagnostic only: no runtime change, no admission, no GPU training. Uses the frozen TRAIN prototype
artifact and the runtime preprocessing recipe (cv2 INTER_AREA 224 + ImageNet CHW).
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, cv2, torch
from transformers import AutoModel

ROOT = Path('/home/user/Robotic/project2')
sys.path.insert(0, str(ROOT))
from harness.task090_dino_localizer import Task090DinoLocalizer
from harness.object_perception import Camera, camera_from_pose, head_pose
from harness.task090_perception import observed_floor_plane
from harness.wam_behavior import normalize_image

RUN = ROOT / 'outputs/behavior/closed_loop/task090_s035k_dev10_seed1_privbase_preposition_onboard_hd_20261002/onboard_received'
ART = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz'
GEOM = ROOT / 'configs/task090_perception_geometry_v2.json'
OUT = ROOT / 'outputs/validation/task090_relation_producer_20261004/dino_offline_601'

def main():
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(str(ROOT / 'models/dinov3-vitl16-pretrain-lvd1689m'),
                                      torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
    loc = Task090DinoLocalizer(ART)
    files = sorted(RUN.glob('received_*.npz'))
    rows, t0 = [], time.monotonic()
    for i, f in enumerate(files):
        a = np.load(f, allow_pickle=True)
        rgb = a['head_rgb'][..., :3]
        depth = a['head_depth_m']
        camp = a['camera_pose']
        prop = a['proprio']
        try: md = json.loads(str(a['metadata_json']))
        except Exception: md = {}
        step = md.get('observation_step', i)
        px = cv2.resize(np.ascontiguousarray(rgb), (224, 224), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(normalize_image(px))[None].to('cuda', torch.bfloat16)
        hs = model(pixel_values=t, output_hidden_states=True, return_dict=True).hidden_states
        feats = [hs[-12][0], hs[-8][0], hs[-4][0]]
        ext = camera_from_pose(head_pose(camp))
        cam = None if ext is None else Camera(*ext, rgb.shape[1], rgb.shape[0])
        floor = None if cam is None else observed_floor_plane(np.asarray(depth, float).squeeze(), cam)
        prop_out, err = None, None
        if cam is None:
            err = 'no_plausible_camera_pose'
        else:
            try:
                prop_out = loc.surface(rgb, np.asarray(depth, float).squeeze(), cam, feats, GEOM)
            except Exception as e:
                err = type(e).__name__ + ':' + str(e)
        rec = {'i': i, 'step': step, 'floor': floor is not None,
               'finger_sum_r': float(prop[49:51].sum()), 'finger_sum_l': float(prop[24:26].sum()),
               'eef_r_z': float(prop[44]), 'eef_l_z': float(prop[19])}
        if prop_out is None:
            rec['det'] = None
            rec['err'] = err
        else:
            d = prop_out['detection']; cl = prop_out['cloud']
            rec['det'] = {k: d[k] for k in ('bbox_xyxy', 'xyz_base', 'surface_pixel_uv', 'surface_depth_m')}
            rec['det']['top5'] = [(p['index'], round(p['positive'], 3)) for p in d['dino_top_patches']]
            rec['cloud_n'] = int(len(cl))
            if floor is not None:
                coef = floor['coefficients']
                clear = cl[:, 2] - np.c_[cl[:, :2], np.ones(len(cl))] @ coef
                rec['clear_p02'] = float(np.percentile(clear, 2)); rec['clear_p50'] = float(np.percentile(clear, 50))
                rec['clear_p98'] = float(np.percentile(clear, 98))
        rows.append(rec)
        if i % 100 == 0:
            print(f'  {i}/{len(files)} step={step} det={"y" if rec["det"] else "n"}', flush=True)
    el = time.monotonic() - t0
    (OUT / 'dino_offline_601.json').write_text(json.dumps(
        {'run': str(RUN), 'artifact': str(ART), 'n': len(rows), 'elapsed_s': el, 'rows': rows}, indent=1))
    print(f'done {len(rows)} frames in {el:.1f}s')

if __name__ == '__main__':
    main()
