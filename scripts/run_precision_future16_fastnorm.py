#!/usr/bin/env python3
"""Batched gradient telemetry and 16GiB RAM cache around the frozen trainer."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_precision_future16_memory as runtime
from scripts import run_precision_future16_prefetch as prefetch

class LargeRamFeatures(prefetch.ParallelRamFeatures):
    def __init__(self, store, budget_bytes=16*1024**3):
        if not 0 <= budget_bytes <= 16*1024**3:
            raise ValueError('16GiB feature RAM cap')
        super().__init__(store, min(budget_bytes, 8*1024**3))
        self.budget_bytes = budget_bytes

def gradient_norm(parameters):
    import torch
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    if not gradients:
        return 0.
    if any(g.dtype != torch.float32 for g in gradients):
        gradients = [g.float() for g in gradients]
    # This affects telemetry only. The trainer's actual clipping is unchanged.
    return float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients, 2))))

def install(profile):
    bootstrap = copy.deepcopy(profile)
    bootstrap['ram_feature_cache_bytes'] = min(profile['ram_feature_cache_bytes'], 8*1024**3)
    selected = prefetch.install(bootstrap)
    from harness import zhuhai_training as training
    from scripts import train_zhuhai_wam_precision_v4 as trainer
    training.BoundedRamFeatures = LargeRamFeatures
    trainer.gradient_norm = gradient_norm
    original_binding = runtime.bind_contract
    def bind(contract, unused):
        result = original_binding(contract, profile)
        path = Path(__file__).resolve()
        result['source_files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['hjepa']['code_hashes'] = copy.deepcopy(result['source_files'])
        return result
    runtime.bind_contract = bind
    return selected

def main():
    profile = json.loads(Path(sys.argv[2]).read_text())
    assert profile['gradient_norm_telemetry'] == 'foreach_l2; clipping unchanged'
    assert profile['ram_feature_cache_bytes'] == 16*1024**3
    selected = install(profile)
    runtime.install_profile = lambda unused: None
    try:
        runtime.main()
    finally:
        if selected['prefetch'] is not None:
            selected['prefetch'].close()

if __name__ == '__main__':
    main()
