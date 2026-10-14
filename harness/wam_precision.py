"""Instance-only precision upgrade AFTER native16 50k loading and H-JEPA installation.

Call before constructing an optimizer/DDP wrapper. For a precision checkpoint,
recreate that architecture, upgrade it, then strict-load the full action-model
state; it is not a legacy v3 checkpoint. New state lives under ``precision.*``.

Views retain the existing head/left-wrist/right-wrist order. The cache contains
CLS + four registers + a square, row-major patch grid (normally 14 x 14).
Patch-center XY coordinates are normalized to [-1, 1]. No encoder is called.
Clock values are explicit, current B x 6 observations, never persistent state.

The query average is a lossy checkpoint warm start, not output-preserving
compression. The clock token and removed gate bias also change the backbone.
Only the added wrist refinement starts at zero relative to that NEW backbone:
its output corrects normalized flow velocity, not joint positions or IK.
"""
from __future__ import annotations

from contextlib import contextmanager
from math import isqrt
from operator import index
from types import MethodType

import torch
from torch import nn

from harness.wam_hjepa import HjepaWrapper

PRECISION_VERSION = 1
EXECUTION_STEPS = 8
PATCH_START = 5
GATE_MODULES = ('phase_net', 'gate_head', 'phase_head', 'dist_head')


def _positive_int(value, name):
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if isinstance(value, bool) or result <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return result


def _check_clock(event_state, batch_size):
    if (not isinstance(event_state, torch.Tensor)
            or event_state.shape != (batch_size, 6)
            or not event_state.is_floating_point()
            or not torch.isfinite(event_state).all()):
        raise ValueError('explicit event_state must be a finite floating B x 6 tensor')
    return event_state.detach()


class PrecisionHeads(nn.Module):
    def __init__(self, feature_dim, hidden_dim, heads, patches, queries, steps):
        super().__init__()
        side = isqrt(patches)
        if side <= 0 or side * side != patches:
            raise ValueError('precision requires a square DINO patch grid')
        self.queries_per_view, self.precision_steps = queries, steps
        self.register_buffer('version', torch.tensor(PRECISION_VERSION, dtype=torch.int64))
        self.register_buffer('configuration', torch.tensor([queries, steps], dtype=torch.int64))
        axis = (torch.arange(side, dtype=torch.float32) + .5) * (2. / side) - 1.
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('patch_xy', torch.stack((xx, yy), dim=-1).reshape(1, 1, patches, 2))
        self.clock = nn.Linear(6, hidden_dim)
        self.patch_proj = nn.Linear(feature_dim + 2, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=0., batch_first=True)
        self.residual = nn.Linear(hidden_dim, 16)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, action_hidden, fused, view_emb):
        """Correct near-term flow velocity from current wrist KV and final action hidden.

        All current wrist patches remain available even though the global
        backbone compresses each view to sixteen tokens. There are no future
        images or proprio values on this action path.
        """
        b, views, tokens, feature_dim = fused.shape
        patches = self.patch_xy.shape[2]
        if views != 3 or tokens != PATCH_START + patches:
            raise ValueError('precision features must be three views of CLS + 4 registers + patches')
        wrist = fused[:, 1:3, PATCH_START:, :]
        xy = self.patch_xy.to(wrist).expand(b, 2, -1, -1)
        kv = self.patch_proj(torch.cat((wrist, xy), dim=-1))
        kv = kv + view_emb[1:3].unsqueeze(0).squeeze(2)
        kv = kv.reshape(b, 2 * patches, -1)
        hidden, _ = self.attention(action_hidden[:, :self.precision_steps], kv, kv, need_weights=False)
        return self.residual(hidden)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Steps have no parameter-shape signature; strict loading must not silently change their meaning.
        for name in ('version', 'configuration', 'patch_xy'):
            saved = state_dict.get(prefix + name)
            expected = getattr(self, name)
            if saved is not None and (not isinstance(saved, torch.Tensor)
                                      or not torch.equal(saved.cpu(), expected.cpu())):
                error_msgs.append(f'{prefix}{name} is incompatible with this precision architecture')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def _precision_forward(self, t, noisy_actions, qpos_history=None, dino_features_list=None,
                       task_cond=None, memory=None, *, event_state=None):
    """Faithful v3 token/state-injection structure, minus global gate bias, plus clock/refinement."""
    b = noisy_actions.shape[0]
    clock = _check_clock(event_state, b)
    if noisy_actions.shape != (b, self.action_len, 16):
        raise ValueError('precision expects native16 actions with chunk length 32')
    if qpos_history is None or qpos_history.shape != (b, self.proprio_len, 16):
        raise ValueError('precision proprio history shape differs from the loaded model')
    if not isinstance(dino_features_list, (list, tuple)) or len(dino_features_list) != self.num_dino_layers:
        raise ValueError('precision requires the three current DINO feature layers')
    n = PATCH_START + self.precision.patch_xy.shape[2]
    if any(f.ndim != 4 or f.shape[:3] != (b, self.num_views, n) for f in dino_features_list):
        raise ValueError('precision current DINO features must have shape B x 3 x (5 + patches) x D')
    if memory is None or memory.shape != (b, self.num_views, self.num_memory, self.mem_proj.in_features):
        raise ValueError('precision memory shape differs from the loaded model')

    t_emb = self.time_mlp(t)
    x_action = self.action_proj(noisy_actions) + self.action_pos_emb + self.type_emb_action
    x_proprio_real = self.proprio_proj(qpos_history) + self.proprio_pos_emb + self.type_emb_proprio
    regs = self.register_tokens.expand(b, -1, -1)
    tc = self.task_cond_proj(task_cond).unsqueeze(1) + self.type_emb_task_cond
    # Detach encoder values only: both shared fusion and adapter must retain gradients.
    fused = self.concat_fusion([f.detach().reshape(b * self.num_views, n, f.shape[-1])
                                for f in dino_features_list])
    vis = self.dino_adapters[0](fused)  # H-JEPA's hook still sees only pre-action, pre-clock visual tokens.
    q = vis.shape[1]
    vis = (vis.reshape(b, self.num_views, q, -1) + self.type_emb_dino_layers[0].unsqueeze(1)
           + self.view_emb.unsqueeze(0).squeeze(2))
    mem = (self.mem_proj(memory.to(vis.dtype)) + self.mem_slot_emb + self.type_emb_mem
           + self.view_emb.unsqueeze(0).squeeze(2))
    clock_token = self.precision.clock(clock.to(device=x_action.device,
                                               dtype=self.precision.clock.weight.dtype)).unsqueeze(1)
    x = torch.cat((x_action, torch.zeros_like(x_proprio_real), regs, tc,
                   vis.reshape(b, self.num_views * q, -1),
                   mem.reshape(b, self.num_views * self.num_memory, -1), clock_token), dim=1)
    for i, block in enumerate(self.blocks):
        if i == self.state_inject_start:
            # Avoid overwriting a view saved for backward by an earlier block.
            x = torch.cat((x[:, :self.action_len], x_proprio_real,
                           x[:, self.action_len + self.proprio_len:]), dim=1)
        x = block(x, t_emb)  # No global view-gate attention mask.
    x = self.final_norm(x)
    pred = self.output_proj(x[:, :self.action_len])
    residual = self.precision(x[:, :self.action_len],
                              fused.reshape(b, self.num_views, n, -1), self.view_emb)
    steps = self.precision.precision_steps
    pred = torch.cat((pred[:, :steps] + residual, pred[:, steps:]), dim=1)
    return {'final_pred': pred, 'cond_tokens': x[:, self.action_len:],
            'gate': torch.full((b, self.num_views), 1. / self.num_views, device=x.device),
            'phase_logits': torch.zeros(b, 4, device=x.device),
            'log_dist': torch.zeros(b, device=x.device)}


@contextmanager
def _clock_input(model, event_state):
    def inject(_model, args, kwargs):
        if 'event_state' in kwargs:
            raise ValueError('event_state supplied twice; precision calls must not overlap')
        return args, dict(kwargs, event_state=event_state)

    handle = model.register_forward_pre_hook(inject, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


class PrecisionWrapper(HjepaWrapper):
    """Same H-JEPA losses; batch/sample clock is required and scoped to a single call."""
    def forward(self, batch, lambdas=None, action_valid_mask=None, future_valid_mask=None):
        clock = _check_clock(batch.get('event_state'), batch['state'].shape[0])
        with _clock_input(self.action_model, clock):
            return super().forward(batch, lambdas, action_valid_mask, future_valid_mask)

    @torch.no_grad()
    def sample_actions(self, feats, state, pool, task_cond, num_steps=10, generator=None, *, event_state=None):
        clock = _check_clock(event_state, state.shape[0])
        with _clock_input(self.action_model, clock):
            return self.base.sample_actions(feats, state, pool, task_cond, num_steps=num_steps, generator=generator)


def upgrade_precision(hjepa_wrapper, *, queries_per_view=16, precision_steps=8, aux_seed=20261009):
    """Upgrade this instance before its optimizer; retain shared weights and caller RNG.

    Averaging query groups intentionally changes the visual bottleneck, so
    retained weights do not imply preservation of the previous policy output.
    """
    if not isinstance(hjepa_wrapper, HjepaWrapper):
        raise ValueError('install H-JEPA auxiliaries before upgrading precision')
    model = hjepa_wrapper.action_model
    if hasattr(model, 'precision'):
        raise ValueError('precision already installed; repeated upgrade is forbidden')
    queries = _positive_int(queries_per_view, 'queries_per_view')
    steps = _positive_int(precision_steps, 'precision_steps')
    if steps > EXECUTION_STEPS:
        raise ValueError('precision_steps cannot exceed the unchanged exec8 prefix')
    if (hjepa_wrapper.action_dim != 16 or hjepa_wrapper.state_dim != 16 or model.action_len != 32
            or model.num_views != 3 or model.num_dino_layers != 3
            or hjepa_wrapper.num_register_tokens != 4 or not model.use_task_cond
            or not model.use_future_feat or not hasattr(model, 'hjepa_aux')
            or model.fusion_mode != 'concat' or len(model.dino_adapters) != 1):
        raise ValueError('precision requires native16/chunk32/three-view, three-layer shared-adapter H-JEPA WAM')
    if not all(hasattr(model, name) for name in GATE_MODULES):
        raise ValueError('expected the original v3 gate modules before precision upgrade')
    if not model.blocks or not 0 <= model.state_inject_start < len(model.blocks):
        raise ValueError('precision requires an active DiT proprio state-injection layer')
    adapter = model.dino_adapters[0]
    old_query = adapter.query_embed
    if (old_query.ndim != 3 or old_query.shape[0] != 1 or old_query.shape[2] != model.hidden_dim
            or old_query.shape[1] % queries or adapter.num_queries != old_query.shape[1]):
        raise ValueError('queries_per_view must evenly divide the original adapter query count')
    with torch.random.fork_rng(devices=[], enabled=True), torch.device('cpu'):
        # Seed ONLY the CPU generator, never torch.manual_seed (which also touches CUDA/MPS).
        torch.random.default_generator.manual_seed(index(aux_seed))
        precision = PrecisionHeads(adapter.feature_proj.in_features, model.hidden_dim,
                                   model.blocks[0].attn1.num_heads, model.future_patches, queries, steps)
    precision.to(device=old_query.device, dtype=old_query.dtype).train(model.training)
    reduced = old_query.detach().reshape(1, queries, old_query.shape[1] // queries, model.hidden_dim).mean(dim=2)
    adapter.query_embed = nn.Parameter(reduced)
    adapter.num_queries = queries
    model.concat_fusion.requires_grad_(True)
    adapter.requires_grad_(True)
    for name in GATE_MODULES:
        delattr(model, name)
    model.add_module('precision', precision)
    model.forward = MethodType(_precision_forward, model)
    wrapped = PrecisionWrapper(hjepa_wrapper.base, hjepa_wrapper.settings['variant'])
    wrapped.training = hjepa_wrapper.training
    return wrapped
