"""Precision-WAM v3 training wiring: reuse the entire v2 build, swap in the full-chunk
geometric refinement head, and warm-start it from the trained v2 (80k) refinement head.

Only the refinement head differs from v2. The DiT trunk, resampler/adapter, clock, and
every v2 weight are built and strict-loaded by v2.build_wrapper and kept. The v3 head
copies the v2 head's clock/attention/residual and the feature+xy columns of patch_proj,
leaving the new camera-ray columns at zero -> the exec-8 prefix reproduces 80k at step 0,
and only the ray usage and the extra tail steps (9..32) have to be learned.
"""
from __future__ import annotations

import copy

from harness import zhuhai_precision_v2_training as v2
from harness.wam_precision_v3 import PrecisionHeadsV3

FORMAT = 'zhuhai-precision-wam-native16-measured-10hz-v3'
REFINE_STEPS = 32
PRECISION_V3_SPEC = dict(copy.deepcopy(v2.PRECISION_SPEC), schema='full_chunk_camera_ray_precision_v3',
                         refine_steps=REFINE_STEPS)
# Buffers that define the v3 architecture identity; never overwrite from the v2 head.
_PROTECT = ('version', 'configuration', 'patch_ray')


def warm_start_v3_head(v3_head, v2_head):
    """Copy shape-matching tensors from a (trained) v2 refinement head into the v3 head;
    widen patch_proj by placing the v2 feature+xy columns first and zeroing the ray columns."""
    v2_sd, v3_sd = v2_head.state_dict(), v3_head.state_dict()
    report = dict(copied=[], widened=None, kept_v3=[])
    for k in list(v3_sd.keys()):
        if k in _PROTECT:
            report['kept_v3'].append(k); continue
        if k in v2_sd and tuple(v2_sd[k].shape) == tuple(v3_sd[k].shape):
            v3_sd[k] = v2_sd[k]; report['copied'].append(k)
        elif k == 'patch_proj.weight' and k in v2_sd and v3_sd[k].shape[0] == v2_sd[k].shape[0] \
                and v3_sd[k].shape[1] == v2_sd[k].shape[1] + 3:
            w = v3_sd[k].clone(); w.zero_(); w[:, :v2_sd[k].shape[1]] = v2_sd[k]
            v3_sd[k] = w; report['widened'] = 'patch_proj.weight: v2 feature+xy columns copied, 3 ray columns zero'
        else:
            report['kept_v3'].append(k)
    v3_head.load_state_dict(v3_sd, strict=True)
    return report


def build_wrapper(cfg, statistics, *, variant, idm_statistics, device='cpu',
                  initialization=None, state_dict=None, refine_steps=REFINE_STEPS):
    """Build via v2 (identical construction + checkpoint load), then replace the refinement head with v3."""
    wrapper, notes = v2.build_wrapper(cfg, statistics, variant=variant, idm_statistics=idm_statistics,
                                      device=device, initialization=initialization, state_dict=state_dict)
    model = wrapper.action_model
    v2_head = model.precision
    v3_head = PrecisionHeadsV3(v2_head.patch_proj.in_features - 2, model.hidden_dim,
                               v2_head.attention.num_heads, v2_head.patch_xy.shape[2],
                               v2_head.queries_per_view, refine_steps)
    v3_head.to(device=v2_head.patch_xy.device, dtype=v2_head.patch_proj.weight.dtype).train(model.training)
    report = warm_start_v3_head(v3_head, v2_head)
    model.precision = v3_head
    notes.update(precision_v3=copy.deepcopy(PRECISION_V3_SPEC), v3_warm_start=report,
                 action_model_parameters=sum(p.numel() for p in model.parameters()))
    return wrapper, notes
