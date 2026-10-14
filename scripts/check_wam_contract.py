#!/usr/bin/env python3
"""CPU-only check of upstream LiLa-WAM's configurable R1Pro tensor contract.

Uses random weights and tiny dimensions. This is NOT a checkpoint evaluation.
Run with .conda-envs/lilawam/bin/python scripts/check_wam_contract.py.
"""
import importlib.util
import json
from pathlib import Path

import torch


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    root = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(root))
    from harness.wam_behavior import BehaviorVLAWrapper, hold_baseline, load_upstream
    state = torch.zeros(3, 61)
    for i, aperture in enumerate((0.0, 0.025, 0.05)):
        state[i, 24:26] = state[i, 49:51] = aperture
    held = hold_baseline(state)
    assert torch.allclose(held[:, 0, 14], torch.tensor([-1., 0., 1.]))
    assert torch.equal(held[:, :, 14], held[:, :, 22])
    assert torch.all(held[:, :, :3] == 0)
    # Reconstruct controller position targets: smooth mode uses mean finger qpos.
    assert torch.allclose((held[:, 0, 14] + 1) * 0.025, state[:, 24])
    spec = importlib.util.spec_from_file_location(
        "upstream_wam", root / "third_party/LiLa-WAM/models/vla_model_fm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.VLAModel(
        action_dim=23, proprio_dim=61, action_len=32, proprio_len=1,
        hidden_dim=32, depth=1, num_heads=4, dino_feat_dims=(16,),
        vlm_num_queries=4, adapter_depth=1, num_registers=2,
        task_cond_dim=16, fusion_mode="concat", concat_out_dim=16,
        use_future_feat=True, future_feat_num_queries=4,
        future_feat_out_dim=16, future_feat_depth=1, future_feat_heads=4)
    action = torch.randn(1, 32, 23)
    features = [torch.randn(1, 4, 16)]
    state, condition = torch.randn(1, 1, 61), torch.randn(1, 16)
    result = model(torch.ones(1), action, state, features, condition)
    future = model.future_feat_decoder(result['cond_tokens'])
    assert result['final_pred'].shape == (1, 32, 23)
    assert future.shape == (1, 4, 16)
    loss, _ = module.calc_flow_matching_loss(
        model, action, features, state, task_cond=condition,
        future_feat_target=torch.randn_like(future), use_future_feat=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.output_proj.weight.grad is not None
    assert torch.isfinite(model.output_proj.weight.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.future_feat_decoder.parameters())

    # The real frozen encoder's positional module randomizes coordinates in train mode.
    # No pretrained weights: one RoPE module plus a tiny frozen parameter for the gradient check.
    from transformers import AutoConfig
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding
    cfg = AutoConfig.from_pretrained(root / "models/dinov3-vitl16-pretrain-lvd1689m", local_files_only=True)
    rope = DINOv3ViTRopePositionEmbedding(cfg)
    encoder = torch.nn.Sequential(rope, torch.nn.Linear(1, 1)).requires_grad_(False).eval()
    _, runner = load_upstream()
    wrapper = BehaviorVLAWrapper(
        runner.VLAWrapper, encoder_dtype=torch.float32, vision_encoder=encoder, action_model=model,
        time_sampler="logit_normal", feat_layers=[-1], include_cls_register=True, num_register_tokens=4,
        device=torch.device("cpu"), dtype=torch.float32, norm_stats_key="behavior1k",
        norm_stats_path=root / "data/behavior-1k/2026-challenge-demos/train_stats_task013.json")
    pixels = torch.zeros(1, 3, 224, 224)
    expected = rope(pixels)
    flags = [p.requires_grad for p in wrapper.parameters()]
    grad = model.output_proj.weight.grad.clone()
    for mode in (True, False, True):  # Initial training, validation, and return to training.
        assert wrapper.train(mode) is wrapper
        assert wrapper.training == model.training == mode
        assert not any(m.training for m in encoder.modules())
        actual = rope(pixels)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        assert [p.requires_grad for p in wrapper.parameters()] == flags
        assert torch.equal(model.output_proj.weight.grad, grad)
    print(json.dumps({
        'status': 'passed', 'device': 'cpu', 'trained_weights': False,
        'check': 'upstream configurable 23D action / 61D state; future head; backward',
        'action_shape': list(result['final_pred'].shape),
        'future_shape': list(future.shape),
        'frozen_encoder_modes': 'train/eval/train preserve encoder eval, deterministic RoPE, and gradients',
        'limitation': 'Tiny random action model and real DINO RoPE only; no pretrained weights or simulator.'
    }, indent=2))


if __name__ == '__main__':
    main()
