"""Independent Precision-WAM v2: current head + both wrists for near-term refinement.

The sole action-structure change from v1 is the refinement KV view set:
three current patch grids instead of the two current wrist grids. Global
compression, event clock, flow integration, execution prefix and H-JEPA
auxiliaries retain the v1 implementation. No encoder is called here, and no
future frame is an online input. More KV tokens increase computation, not
parameter count. Attention is learned focus, not calibrated camera reliability.

Install on a fresh H-JEPA instance before constructing an optimizer or DDP.
This installer rejects an already installed v1/v2 instance; it does not migrate
or mutate a running v1. Recreate v2 before strict-loading a v2 checkpoint.
"""
from __future__ import annotations

import torch
from torch import nn

from harness.wam_precision import PATCH_START, PrecisionHeads, upgrade_precision

PRECISION_VERSION = 2
REFINEMENT_VIEW_INDICES = (0, 1, 2)  # Existing head, left wrist, right wrist order.


class PrecisionHeadsV2(PrecisionHeads):
    """Same learned parameters as v1; refinement attends to all current views."""

    def __init__(self, feature_dim, hidden_dim, heads, patches, queries, steps):
        super().__init__(feature_dim, hidden_dim, heads, patches, queries, steps)
        self.version.fill_(PRECISION_VERSION)
        self.configuration = self.configuration.new_tensor(
            [queries, steps, *REFINEMENT_VIEW_INDICES])

    @classmethod
    def _reuse_initialized_head(cls, source):
        """Transfer a freshly created v1 head without new weights or RNG use.

        Called only inside the fresh-instance installer below. Existing modules
        and the coordinate buffer are reused, while checkpoint identity buffers
        are independent. The discarded temporary head never belonged to a
        running v1 or an optimizer.
        """
        head = cls.__new__(cls)
        nn.Module.__init__(head)
        head.queries_per_view = source.queries_per_view
        head.precision_steps = source.precision_steps
        head.register_buffer('version', source.version.new_tensor(PRECISION_VERSION))
        head.register_buffer('configuration', source.configuration.new_tensor(
            [source.queries_per_view, source.precision_steps, *REFINEMENT_VIEW_INDICES]))
        head.register_buffer('patch_xy', source.patch_xy)
        for name in ('clock', 'patch_proj', 'attention', 'residual'):
            head.add_module(name, getattr(source, name))
        head.train(source.training)
        return head

    def forward(self, action_hidden, fused, view_emb):
        """Add a normalized flow-velocity correction for the execution prefix.

        Each action query chooses patches across the three current cameras in
        one softmax. There is no separate camera gate or a second multiplication
        by camera weights. CLS/register tokens do not enter this dense KV path.
        """
        b, views, tokens, _ = fused.shape
        patches = self.patch_xy.shape[2]
        if views != len(REFINEMENT_VIEW_INDICES) or tokens != PATCH_START + patches:
            raise ValueError('precision v2 features must be three views of CLS + 4 registers + patches')
        current_patches = fused[:, :, PATCH_START:, :]
        xy = self.patch_xy.to(current_patches).expand(b, views, -1, -1)
        kv = self.patch_proj(torch.cat((current_patches, xy), dim=-1))
        kv = kv + view_emb.unsqueeze(0).squeeze(2)
        kv = kv.reshape(b, views * patches, -1)
        hidden, _ = self.attention(
            action_hidden[:, :self.precision_steps], kv, kv, need_weights=False)
        return self.residual(hidden)


def upgrade_precision_v2(hjepa_wrapper, *, queries_per_view=16, precision_steps=8,
                         aux_seed=20261009):
    """Create v2 on a fresh instance, retaining all v1 learned-weight shapes.

    V1 validation, query averaging and instance-scoped model forward are reused.
    The inherited strict loader checks v2 identity/configuration buffers, so v1
    and v2 checkpoints are incompatible even though their parameter shapes match.
    """
    wrapper = upgrade_precision(hjepa_wrapper, queries_per_view=queries_per_view,
                                precision_steps=precision_steps, aux_seed=aux_seed)
    model = wrapper.action_model
    model.precision = PrecisionHeadsV2._reuse_initialized_head(model.precision)
    return wrapper
