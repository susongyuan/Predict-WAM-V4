"""Precision-WAM v3: full-chunk dense-patch refinement with a camera-ray geometric PE.

Diagnosis-driven (2026-10-12). The global visual bottleneck compresses 196 patches
per view into 16 queries (48 tokens) feeding the DiT trunk; only the first 8 action
steps get a dense bypass to all 588 raw patches (PrecisionHeadsV2). The tail steps
(indices 25-32) that decide descent/closure therefore ride the compressed 48-token
representation and structurally cannot resolve the last-centimetre alignment.

V3 changes ONLY the refinement head (upstream resampler/adapter and DiT trunk are
untouched, so the policy stays a general WAM and warm-starts from the 80k weights):
 1. extend the dense-patch refinement from the exec-8 prefix to ALL `refine_steps`
    chunk steps (default 32 = full horizon H);
 2. append a per-patch camera-ray 3D direction (PRoPE-style geometric encoding) to the
    2D grid coordinate, so attention reasons about viewing direction / depth, not a flat
    index. The ray columns of patch_proj are zero at construction; a warm-start loader
    copies the 80k feature+xy columns and leaves ray zero, so step-0 behaviour == v2.

No encoder call, no future frame, no task-specific prior; residual is zero-initialised.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from harness.wam_precision import PATCH_START, upgrade_precision
from harness.wam_precision_v2 import PrecisionHeadsV2, REFINEMENT_VIEW_INDICES

PRECISION_VERSION = 3
CAMERA_FOVY_DEGREES = 65.0  # head and both wrists share fovy (harness/isaac_zhuhai_camera.py)


def camera_ray_grid(patch_xy, fovy_degrees=CAMERA_FOVY_DEGREES):
    """(1,1,P,2) normalised patch-centre xy in [-1,1] -> (1,1,P,3) unit camera-frame rays.

    Pinhole with a square 224x224 DINO input and vertical fov=fovy: a normalised image
    coordinate maps to the camera-frame direction normalize([x*t, y*t, -1]) with
    t=tan(fovy/2) and the camera looking down -z. This gives every patch a 3D viewing
    direction instead of a flat 2D index.
    """
    t = math.tan(math.radians(fovy_degrees) / 2.0)
    x = patch_xy[..., 0:1] * t
    y = patch_xy[..., 1:2] * t
    z = -torch.ones_like(x)
    ray = torch.cat((x, y, z), dim=-1)
    return ray / ray.norm(dim=-1, keepdim=True)


class PrecisionHeadsV3(PrecisionHeadsV2):
    """All-step dense-patch refinement over the three current views + camera-ray geometric PE."""

    def __init__(self, feature_dim, hidden_dim, heads, patches, queries, steps):
        super().__init__(feature_dim, hidden_dim, heads, patches, queries, steps)
        self.version.fill_(PRECISION_VERSION)
        self.configuration = self.configuration.new_tensor([queries, steps, *REFINEMENT_VIEW_INDICES])
        self.register_buffer('patch_ray', camera_ray_grid(self.patch_xy))  # (1,1,P,3), camera frame
        # Widen the KV projection to feature + 2D grid + 3D ray; zero so a warm-start loader fills it.
        self.patch_proj = nn.Linear(feature_dim + 2 + 3, hidden_dim)
        nn.init.zeros_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)

    def forward(self, action_hidden, fused, view_emb):
        """Refine EVERY chunk step's flow velocity from the full three-view patch KV.

        Same dense KV path as v2 (each action step attends over all current patches of
        the three views in one softmax) but across the whole chunk and with the 3D
        camera ray appended to each patch's 2D grid coordinate.
        """
        b, views, tokens, _ = fused.shape
        patches = self.patch_xy.shape[2]
        if views != len(REFINEMENT_VIEW_INDICES) or tokens != PATCH_START + patches:
            raise ValueError('precision v3 features must be three views of CLS + 4 registers + patches')
        current_patches = fused[:, :, PATCH_START:, :]
        xy = self.patch_xy.to(current_patches).expand(b, views, -1, -1)
        ray = self.patch_ray.to(current_patches).expand(b, views, -1, -1)
        kv = self.patch_proj(torch.cat((current_patches, xy, ray), dim=-1))
        kv = kv + view_emb.unsqueeze(0).squeeze(2)
        kv = kv.reshape(b, views * patches, -1)
        hidden, _ = self.attention(action_hidden[:, :self.precision_steps], kv, kv, need_weights=False)
        return self.residual(hidden)


def upgrade_precision_v3(hjepa_wrapper, *, queries_per_view=16, refine_steps=32, aux_seed=20261009):
    """Create v3 on a fresh instance: unchanged v2 global compression + full-chunk geometric refinement.

    The global resampler/adapter and DiT trunk stay the v1/v2 ones (queries_per_view
    unchanged, default 16). The only structural change is the refinement head, now
    covering all `refine_steps` chunk steps (default 32) with a camera-ray PE. Build the
    v1 head first (steps clipped to the exec8 guard, reusing adapter reduction / gate
    removal / forward swap), then swap in the v3 head.
    """
    if not (isinstance(refine_steps, int) and 0 < refine_steps):
        raise ValueError('refine_steps must be a positive int (full chunk = 32)')
    wrapper = upgrade_precision(hjepa_wrapper, queries_per_view=queries_per_view,
                                precision_steps=min(refine_steps, 8), aux_seed=aux_seed)
    model = wrapper.action_model
    src = model.precision  # freshly built v1 head
    feature_dim = src.patch_proj.in_features - 2
    head = PrecisionHeadsV3(feature_dim, model.hidden_dim, model.blocks[0].attn1.num_heads,
                            src.patch_xy.shape[2], src.queries_per_view, refine_steps)
    head.to(device=src.patch_xy.device, dtype=src.patch_proj.weight.dtype).train(model.training)
    model.precision = head
    return wrapper
