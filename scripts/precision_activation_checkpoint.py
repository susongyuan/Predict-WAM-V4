"""Opt-in training recomputation; tensor names and inference behavior stay unchanged."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

def enable(model):
    selected = []
    for name, module in model.named_modules():
        if not (isinstance(module, (nn.TransformerDecoderLayer, nn.TransformerEncoderLayer))
                or type(module).__name__ in ('DiTBlock', 'MultiLayerConcatFusion', 'PrecisionHeadsV3')):
            continue
        if getattr(module, '_project_activation_checkpoint', False):
            raise ValueError('Activation checkpoint already installed: '+name)
        original = module.forward
        def forward(*args, _module=module, _original=original, **kwargs):
            if _module.training and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
            return _original(*args, **kwargs)
        module.forward = forward
        module._project_activation_checkpoint = True
        selected.append(name)
    if not selected:
        raise ValueError('No supported production layers found')
    return selected
