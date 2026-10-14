"""TRAIN-prototype DINO surface proposals; never object-presence or grasp certification."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from harness.task090_perception import load_calibration


class Task090DinoLocalizer:
    def __init__(self, artifact):
        path = Path(artifact)
        self.meta = json.loads(path.with_suffix('.json').read_text())
        if (self.meta.get('source_split') != 'train' or self.meta.get('physical_success_verifier') is not False
                or any(self.meta.get(k,False) is not False for k in ('uses_dev','uses_test','uses_report'))):
            raise ValueError('DINO localizer requires TRAIN-only appearance provenance')
        if path.stat().st_size > 1024**2 or hashlib.sha256(path.read_bytes()).hexdigest() != self.meta.get('artifact_sha256'):
            raise ValueError('DINO appearance prototype integrity mismatch')
        with np.load(path, allow_pickle=False) as data:
            self.positive = data['positive'].copy()
            self.negative = data['negative'].copy()
        if self.positive.shape != (3, 1024) or self.negative.shape != (3, 1024):
            raise ValueError('unexpected prototype shape')
        if not np.isfinite(self.positive).all() or not np.isfinite(self.negative).all():
            raise ValueError('nonfinite prototypes')

    def scores(self, features):
        layers = []
        if len(features) != 3:
            raise ValueError('three current DINO layers required')
        for value in features:
            if hasattr(value, 'detach'):
                value = value.detach().float().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.shape == (1, 201, 1024):
                value = value[0]
            if value.shape != (201, 1024) or not np.isfinite(value).all():
                raise ValueError('current head features must include 5 prefix + 196 spatial tokens')
            value = value[5:]
            layers.append(value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8))
        layers = np.stack(layers)
        pos = np.einsum('lpd,ld->lp', layers, self.positive).mean(0)
        neg = np.einsum('lpd,ld->lp', layers, self.negative).mean(0)
        return pos, neg

    def surface(self, rgb, depth, camera, features, appearance_calibration):
        pos, neg = self.scores(features)
        h, w = rgb.shape[:2]
        # Ranking is diagnostic only: absent/occluded objects may still have an argmax.
        top = np.argsort(-pos, kind='stable')[:5]
        grid = np.zeros((14, 14), dtype=np.uint8)
        grid.flat[top] = 1
        region = cv2.resize(grid, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        cfg = load_calibration(str(appearance_calibration))
        hsv = cv2.cvtColor(np.asarray(rgb[..., :3], np.uint8), cv2.COLOR_RGB2HSV)
        mask = np.zeros((h, w), np.uint8)
        for lower, upper in cfg['classes']['trash_can']['hsv_boxes']:
            mask |= cv2.inRange(hsv, np.asarray(lower,np.uint8), np.asarray(upper,np.uint8))
        valid = region & (mask > 0) & np.isfinite(depth) & (depth > .05) & (depth < cfg['depth_max_m'])
        v, u = np.nonzero(valid)
        if len(u) < cfg['min_depth_pixels']:
            return None
        cloud = camera.backproject(u, v, depth[v, u])
        median = np.median(cloud, axis=0)
        k = int(np.argmin(np.sum((cloud-median)**2, axis=1)))
        return {'detection': {'label':'trash_can','physical_category':'trash_can',
            'bbox_xyxy':[int(u.min()),int(v.min()),int(u.max()+1),int(v.max()+1)],
            'xyz_base':cloud[k].tolist(), 'surface_pixel_uv':[int(u[k]),int(v[k])],
            'surface_depth_m':float(depth[v[k],u[k]]), 'source':'rgbd',
            'calibration_id':self.meta['calibration_id'], 'point_semantics':'observed_surface_sample',
            'semantic_status':'unadmitted_dino_appearance_candidate',
            'object_presence_verified':False, 'center_valid':False, 'grasp_reference_valid':False,
            'dino_top_patches':[{'index':int(i), 'positive':float(pos[i]), 'negative':float(neg[i]),
                                 'native_center_xy':[(int(i)%14+.5)*w/14,(int(i)//14+.5)*h/14]} for i in top]},
            'cloud':cloud}
