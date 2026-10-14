"""Causal +16 pooled-visual prediction and resize-aware rays around Precision-WAM V3.

The projected prediction is computed before action/noise mixing and reused by
every Euler iteration. Future values are loss-only labels, never model inputs.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn
import torch.nn.functional as F

from harness.wam_precision import PrecisionWrapper
from harness.isaac_zhuhai_camera import RESOLUTION, HEAD_FOVY_DEGREES

HORIZON = 16
CHUNK_WEIGHT = .1
COSINE_WEIGHT = .1
COSINE_EPS = 1e-6
CAMERA_RAY_SPEC = dict(schema='resize_aware_camera_ray_v4',
    native_resolution_wh=list(RESOLUTION), dino_resize_wh=[224, 224],
    fovy_degrees=HEAD_FOVY_DEGREES,
    formula='normalize([x*t*(320/240), -y*t, -1]); t=tan(fovy/2)',
    image_axes='x_right_y_down', camera_axes='x_right_y_up_minus_z_forward',
    depth=False, extrinsics=False)


def corrected_camera_ray_grid(patch_xy):
    # Compute buffers on CPU so checkpoint identity is independent of the build device.
    xy = patch_xy.detach().to(device='cpu', dtype=torch.float32)
    if xy.shape[-1:] != (2,) or not torch.isfinite(xy).all():
        raise ValueError('Camera rays need finite image patch XY coordinates')
    t = math.tan(math.radians(HEAD_FOVY_DEGREES) / 2.)
    ray = torch.cat((xy[..., :1] * t * (RESOLUTION[0] / RESOLUTION[1]),
                     -xy[..., 1:] * t, -torch.ones_like(xy[..., :1])), dim=-1)
    return (ray / ray.norm(dim=-1, keepdim=True)).to(patch_xy)


class ChunkVTTPredictor(nn.Module):
    def __init__(self, feature_dim, state_dim, task_dim, condition_dim, hidden=512):
        super().__init__()
        self.feature_dim, self.task_dim = feature_dim, task_dim
        input_dim, output_dim = 6 * feature_dim + state_dim + task_dim, 3 * feature_dim
        self.register_buffer('configuration', torch.tensor(
            [4, HORIZON, feature_dim, state_dim, task_dim, condition_dim, hidden]))
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
        # A nonzero projection lets a zero alpha receive its first action gradient.
        self.projection = nn.Linear(output_dim, condition_dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, current, history_change, state, task):
        b, d = state.shape[0], self.feature_dim
        if (current.shape != (b, 3, d) or history_change.shape != (b, 3, d)
                or task.shape != (b, self.task_dim)
                or state.shape != (b, int(self.configuration[3]))):
            raise ValueError('Chunk-VTT current/history/state/task shape mismatch')
        values = (current, history_change, state, task)
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError('Chunk-VTT inputs must be finite')
        x = torch.cat((F.layer_norm(current, (d,)).flatten(1),
                       history_change.flatten(1), state,
                       F.layer_norm(task, (self.task_dim,))), dim=-1)
        prediction = self.predictor(x)
        return prediction.reshape(b, 3, d), self.alpha * self.projection(prediction)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + 'configuration')
        if saved is not None and (not isinstance(saved, torch.Tensor) or saved.dtype != torch.int64
                                  or not torch.equal(saved.cpu(), self.configuration.cpu())):
            error_msgs.append(prefix + 'configuration differs from the Chunk-VTT architecture')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def chunk_vtt_loss(prediction, target, valid):
    """Raw frozen feature differences; stationary views have no cosine direction."""
    if target.shape != prediction.shape or valid.shape != prediction.shape[:1] or valid.dtype != torch.bool:
        raise ValueError('Chunk-VTT label/mask shape or dtype mismatch')
    prediction, target = prediction[valid].float(), target.detach()[valid].float()
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite valid Chunk-VTT prediction or target')
    if not valid.any():
        zero = prediction.sum() * 0.
        return zero, zero, zero
    huber = F.smooth_l1_loss(prediction, target)
    directional = target.norm(dim=-1) > COSINE_EPS
    cosine = ((1. - F.cosine_similarity(prediction[directional], target[directional], dim=-1,
                                       eps=COSINE_EPS)).mean()
              if directional.any() else prediction.sum() * 0.)
    return huber + COSINE_WEIGHT * cosine, huber, cosine


@contextmanager
def _fused_task(model, shift):
    # Scope matches the existing clock hook; sampling retains no episode state.
    handle = model.task_cond_proj.register_forward_hook(lambda module, args, output: output + shift.to(output))
    try:
        yield
    finally:
        handle.remove()


class PrecisionV4Wrapper(PrecisionWrapper):
    def predict_chunk(self, pool, state, task_cond):
        pool = pool.detach().to(self.device, torch.float32)
        state = state.detach().to(self.device, torch.float32)
        task = task_cond.detach().to(self.device, torch.float32)
        memory = self.base.normalize_memory(pool)
        d = self.action_model.chunk_vtt16.feature_dim
        current = pool[:, :, 0, d:]
        history_change = memory[:, :, 1, d:]  # Existing memory lags [4,16,start].
        q = self.base.normalize_state(state)[:, 0]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            return self.action_model.chunk_vtt16(current, history_change, q, task)

    def forward(self, batch, lambdas=None, action_valid_mask=None, future_valid_mask=None):
        prediction, shift = self.predict_chunk(batch['pool'], batch['state'], batch['task_cond'])
        with _fused_task(self.action_model, shift):
            total, info, out = super().forward(batch, lambdas, action_valid_mask, future_valid_mask)
        valid = self._mask(batch, 'chunk_vtt_valid_mask', prediction.shape[0])
        target = batch['chunk_vtt_target'].to(self.device, dtype=torch.float32)
        loss, huber, cosine = chunk_vtt_loss(prediction, target, valid)
        total = total + CHUNK_WEIGHT * loss
        info = dict(info, chunk_vtt_loss=float(loss.detach()), chunk_huber=float(huber.detach()),
                    chunk_cosine=float(cosine.detach()), alpha=float(self.action_model.chunk_vtt16.alpha.detach()),
                    chunk_vtt_weight=CHUNK_WEIGHT, chunk_vtt_valid_samples=int(valid.sum()))
        out = dict(out, chunk_vtt_prediction=prediction)
        return total, info, out

    @torch.no_grad()
    def sample_actions(self, feats, state, pool, task_cond, num_steps=10, generator=None, *, event_state=None):
        _, shift = self.predict_chunk(pool, state, task_cond)
        with _fused_task(self.action_model, shift):
            return super().sample_actions(feats, state, pool, task_cond, num_steps=num_steps,
                                          generator=generator, event_state=event_state)


def upgrade_precision_v4(wrapper, *, hidden=512, seed=20261012, memory_lags=(4, 16)):
    if not isinstance(wrapper, PrecisionWrapper) or hasattr(wrapper.action_model, 'chunk_vtt16'):
        raise ValueError('Install Chunk-VTT exactly once on a fresh Precision V3 wrapper')
    model = wrapper.action_model
    if (int(model.precision.version) != 3 or model.precision.precision_steps != 32
            or model.num_views != 3 or tuple(memory_lags) != (4, 16) or model.num_memory != 3):
        raise ValueError('Chunk-VTT needs V3 full32, three cameras and causal memory lags [4,16]')
    d = model.mem_proj.in_features // 2
    if d <= 0 or model.mem_proj.in_features != 2 * d:
        raise ValueError('Expected pooled CLS plus mean-patch features')
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(seed)
        module = ChunkVTTPredictor(d, wrapper.state_dim, d, model.hidden_dim, hidden)
    model.precision.patch_ray.copy_(corrected_camera_ray_grid(model.precision.patch_xy))
    model.add_module('chunk_vtt16', module.to(device=wrapper.device).train(model.training))
    result = PrecisionV4Wrapper(wrapper.base, wrapper.settings['variant'])
    result.training = wrapper.training
    return result
