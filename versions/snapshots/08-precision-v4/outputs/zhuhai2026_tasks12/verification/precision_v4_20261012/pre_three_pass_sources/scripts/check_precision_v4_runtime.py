#!/usr/bin/env python3
"""Real 80k weights and full-size synthetic tensors; no encoder, training, or physics."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch


def verify_migration(original, migrated, extended):
    """Only the documented V3 ray widening and V4 module may change the 80k state."""
    protected = {'precision.version', 'precision.configuration'}
    assert set(migrated) - set(original) == {'precision.patch_ray'}
    assert not set(original) - set(migrated)
    copied = 0
    for key, value in original.items():
        actual = migrated[key].detach().cpu()
        if key in protected:
            continue
        if key == 'precision.patch_proj.weight':
            assert actual.shape == (value.shape[0], value.shape[1] + 3)
            torch.testing.assert_close(actual[:, :value.shape[1]], value, rtol=0, atol=0)
            assert not actual[:, value.shape[1]:].count_nonzero()
        else:
            torch.testing.assert_close(actual, value, rtol=0, atol=0, msg=key)
            copied += 1
    assert int(migrated['precision.version']) == 3
    assert set(migrated) <= set(extended)
    assert all(key.startswith('chunk_vtt16.') for key in set(extended) - set(migrated))
    for key, value in migrated.items():
        torch.testing.assert_close(extended[key], value, rtol=0, atol=0, msg=key)
    return dict(original_tensors=len(original), exact_copied_tensors=copied,
                widened_ray_columns_zero=True, v3_to_v4_existing_tensors_exact=len(migrated),
                added_state_tensors=len(extended) - len(migrated))


def synthetic_batch(wrapper, task, seed=720):
    """Full production shapes with artificial values, not observations or teacher labels."""
    m = wrapper.action_model
    generator = torch.Generator().manual_seed(seed)
    def normal(*shape):
        return torch.randn(*shape, generator=generator) * .2
    b, v, d = 1, m.num_views, m.mem_proj.in_features // 2
    patches = m.precision.patch_xy.shape[2]
    feats = normal(b, v, m.num_dino_layers, patches + 5, d)
    state = ((wrapper.state_min + wrapper.state_max) / 2).detach().cpu()
    state = state[None, None].repeat(b, m.proprio_len, 1)
    pool = normal(b, v, m.num_memory + 1, 2 * d)
    pool[:, :, 0] = torch.cat((feats[:, :, -1, 0], feats[:, :, -1, 5:].mean(2)), -1)
    actions = ((wrapper.action_min + wrapper.action_max) / 2).detach().cpu()
    return dict(feats=feats, state=state, pool=pool,
        action_sequence=actions[None, None].repeat(b, m.action_len, 1),
        action_valid_mask=torch.ones(b, m.action_len, dtype=torch.bool),
        future_valid_mask=torch.ones(b, dtype=torch.bool), future_feats=normal(b, v, patches, d),
        task_cond=torch.as_tensor(task, dtype=torch.float32).reshape(b, d),
        phase=torch.full((b,), -1), future_feats_8=normal(b, v, patches, d),
        future_valid_mask_8=torch.ones(b, dtype=torch.bool),
        idm_future_feats=normal(b, 4, v, m.num_dino_layers, patches + 5, d),
        idm_target_raw=normal(b, 4, wrapper.action_dim), idm_valid_mask=torch.ones(b, dtype=torch.bool),
        event_state=torch.zeros(b, 6), chunk_vtt_target=normal(b, v, d),
        chunk_vtt_valid_mask=torch.ones(b, dtype=torch.bool))


def verify_runtime(v3, v4, batch):
    v3.eval(); v4.eval()
    model, chunk = v4.action_model, v4.action_model.chunk_vtt16
    assert float(chunk.alpha) == 0.
    def sample(wrapper):
        return wrapper.sample_actions([batch['feats'][:, :, i] for i in range(3)],
            batch['state'], batch['pool'], batch['task_cond'], num_steps=10,
            generator=torch.Generator(device=wrapper.device).manual_seed(719),
            event_state=batch['event_state'])[0]
    counts = dict(predictor=0, task_projection=0)
    def count(key):
        def increment(*args):
            counts[key] += 1
        return increment
    expected = sample(v3)
    handles = [chunk.register_forward_hook(count('predictor')),
               model.task_cond_proj.register_forward_hook(count('task_projection'))]
    try:
        actual = sample(v4)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert counts == dict(predictor=1, task_projection=10), counts
    with torch.no_grad():
        chunk.alpha.fill_(.2)
    opened_difference = float((sample(v4) - expected).abs().max())
    assert opened_difference > 0.
    devices = [v4.device.index or 0] if v4.device.type == 'cuda' else []
    def forward(value):
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(730)
            if devices:
                torch.cuda.manual_seed_all(730)
            return v4(value)
    labels = ('future_feats', 'future_feats_8', 'idm_future_feats', 'idm_target_raw', 'chunk_vtt_target')
    changed = dict(batch, **{key: batch[key] * -3 + 5 for key in labels})
    with torch.no_grad():
        _, _, before = forward(batch)
        _, _, after = forward(changed)
        for key in ('final_pred', 'chunk_vtt_prediction'):
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0, msg=key)
        chunk.alpha.zero_()
    del before, after, changed
    torch.testing.assert_close(sample(v4), expected, rtol=0, atol=0)
    frozen = {name: p.requires_grad for name, p in model.named_parameters()}
    try:
        model.requires_grad_(False)
        chunk.predictor.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        supervised = dict(batch, chunk_vtt_target=batch['chunk_vtt_target'].clone().requires_grad_())
        loss, info, out = forward(supervised)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in chunk.predictor.parameters()]
        assert all(g is not None and torch.isfinite(g).all() and g.count_nonzero() for g in grads)
        norm = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
        assert supervised['chunk_vtt_target'].grad is None
        assert all(p.grad is None for name, p in model.named_parameters()
                   if not name.startswith('chunk_vtt16.predictor.'))
        del loss, out, grads, supervised
        model.zero_grad(set_to_none=True)
        chunk.predictor.requires_grad_(False)
        chunk.alpha.requires_grad_(True)
        _, _, out = forward(batch)
        out['final_pred'].float().square().mean().backward()
        assert chunk.alpha.grad is not None and torch.isfinite(chunk.alpha.grad)
        gate_gradient = float(chunk.alpha.grad)
        assert gate_gradient != 0.
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(frozen[name])
        with torch.no_grad():
            chunk.alpha.zero_()
    assert not model.task_cond_proj._forward_hooks and not model._forward_pre_hooks
    return dict(alpha_zero_ten_euler_exact=True, calls_per_replan=counts,
        open_alpha_action_max_difference=opened_difference, zero_again_exact=True,
        future_label_perturbation_action_and_prediction_exact=True,
        frozen_backbone_no_grad=True, predictor_gradient_norm=norm,
        alpha_zero_action_gradient=gate_gradient, losses=info, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing runtime evidence')
    from harness import zhuhai_precision_v2_training as v2
    from harness import zhuhai_precision_v3_training as v3
    from harness import zhuhai_precision_v4_training as v4
    from scripts.train_zhuhai_wam_precision_v3 import CK80K, CK80K_SHA
    torch.set_num_threads(4)
    started = time.monotonic()
    assert v2.sha256_file(CK80K) == CK80K_SHA
    saved = torch.load(CK80K, map_location='cpu', weights_only=False, mmap=True)
    assert saved['format'] == v2.FORMAT and saved['step'] == 80000
    contract = saved['contract']
    v2.verify_code_hashes(contract['source_files'])
    kwargs = dict(variant=contract['hjepa']['variant'], idm_statistics=contract['idm_statistics'],
                  device=args.device, state_dict=saved['model_state_dict'])
    old, _ = v3.build_wrapper(contract['config'], contract['statistics'], **kwargs)
    new, notes = v4.build_wrapper(contract['config'], contract['statistics'], **kwargs)
    report = dict(schema='precision-v4-runtime-check-v1', timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        checkpoint=str(CK80K), checkpoint_sha256=CK80K_SHA, device=args.device,
        inputs='synthetic full-size tensors; no recorded observations or DINO encoder execution',
        physical_success_tested=False, task_success_rate=None, build=notes)
    report['migration'] = verify_migration(saved['model_state_dict'], old.action_model.state_dict(), new.action_model.state_dict())
    batch = synthetic_batch(new, contract['conditions'][v2.TASK])
    report['input_shapes'] = {key: list(value.shape) for key, value in batch.items()}
    report['runtime'] = verify_runtime(old, new, batch)
    report.update(status='passed', elapsed_seconds=time.monotonic() - started,
                  added_parameters=sum(p.numel() for p in new.action_model.chunk_vtt16.parameters()))
    if args.device == 'cuda':
        report.update(gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    else:
        assert not torch.cuda.is_initialized()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
