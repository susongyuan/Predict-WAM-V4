#!/usr/bin/env python3
"""CPU check of the multi-view (head + wrists) WAM path; no DINO, no GPU, no training.

1. stage3 best loads into MultiViewVLAModel with ONLY `view_emb` missing (zero init).
2. Head-only input (wrist tokens dropped) through the multi-view model == upstream VLAModel with the stage3
   checkpoint, for identical features / noise / t / state / condition (final_pred, cond_tokens, future decoder).
3. Three-view forward: shapes, cond_tokens = 4 + 3*64 tokens, future decoder still (B,196,1024), finite loss,
   gradients reach `view_emb` and the shared adapter.
4. Dataset: multi-view item is (V,3,224,224) in config order, view 0 / future frame / actions / state /
   condition are byte-identical to the legacy single-view item; legacy item identical to the pre-change code
   (optional, when --orig-module points at a saved copy of harness/wam_behavior.py).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import (BehaviorChunkDataset, HEAD_CAMERA, collate, create_action_model, load_action_state,  # noqa: E402
                                  load_upstream, model_views)

STAGE3 = ROOT / "outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt"
MV_CONFIG = ROOT / "configs/lilawam_behavior_multiview_manip_train.yaml"
DINO_HIDDEN, PATCH, TOKENS = 1024, 16, 1 + 4 + 196   # DINOv3 ViT-L/16 @224 with CLS + 4 registers


def dataset_check(cfg, orig_module):
    kw = dict(stride=int(cfg.dataset.frame_stride), chunk=int(cfg.common.action_chunk_size),
              future_offset=int(cfg.dataset.future_frame_offset), image_size=224, cond_mode=cfg.dataset.cond_mode,
              instance_prob=0.0, seed=int(cfg.training.seed), contract_targets=cfg.dataset.contract_targets,
              contract_targets_sha256=cfg.dataset.contract_targets_sha256,
              skip_unsupported_contracts=bool(cfg.dataset.skip_unsupported_contracts),
              scope=cfg.dataset.scope, in_segment=cfg.dataset.in_segment)
    split = json.loads((ROOT / cfg.dataset.root / "split_task_013.json").read_text())
    eps = sorted(split["validation_episode_indices"])[:2]
    views = model_views(cfg)
    single = BehaviorChunkDataset("validation", episodes=eps, **kw)
    multi = BehaviorChunkDataset("validation", episodes=eps, views=views, **kw)
    assert single.anchors == multi.anchors and multi.views == views
    idx = np.linspace(0, len(single) - 1, 6).astype(int).tolist()
    items = [multi[i] for i in idx]
    for i, m in zip(idx, items):
        s = single[i]
        assert s["pixel_values"].shape == (3, 224, 224) and m["pixel_values"].shape == (len(views), 3, 224, 224)
        assert torch.equal(m["pixel_values"][0], s["pixel_values"])
        for key in ("future_pixel_values", "state", "action_sequence", "task_cond"):
            assert torch.equal(m[key], s[key]), key
        for v in range(1, len(views)):   # wrist frames are distinct images from the head frame
            assert not torch.equal(m["pixel_values"][v], m["pixel_values"][0])
    batch = collate(items)
    assert batch["pixel_values"].shape == (len(items), len(views), 3, 224, 224)
    assert batch["future_pixel_values"].shape == (len(items), 3, 224, 224)
    result = {"episodes": eps, "anchors": len(single), "checked_items": idx,
              "multi_pixel_shape": list(batch["pixel_values"].shape), "head_equal_legacy": True}
    if orig_module:
        import types
        source = Path(orig_module).read_text()
        anchor = "ROOT = Path(__file__).resolve().parents[1]"
        assert source.count(anchor) == 1
        orig = types.ModuleType("wam_behavior_orig")   # saved copy may live outside the repo: pin its ROOT
        exec(compile(source.replace(anchor, f"ROOT = Path({str(ROOT)!r})"), str(orig_module), "exec"), orig.__dict__)
        legacy = orig.BehaviorChunkDataset("validation", episodes=eps, **kw)
        assert legacy.anchors == single.anchors and legacy.anchor_filter == single.anchor_filter
        for i in idx:
            a, b = legacy[i], single[i]
            assert a.keys() == b.keys() and all(
                torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k] for k in a)
        result["legacy_single_view_identical_to_pre_change_code"] = True
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-module", help="saved pre-change harness/wam_behavior.py for the legacy dataset identity check")
    ap.add_argument("--out")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    load_upstream()
    ck = torch.load(STAGE3, map_location="cpu", weights_only=False, mmap=True)
    src_cfg = OmegaConf.create(ck["config"])
    mv_cfg = OmegaConf.load(MV_CONFIG)
    assert "views" not in src_cfg.model and model_views(src_cfg) == [HEAD_CAMERA]
    src_model_cfg = OmegaConf.to_container(src_cfg.model); mv_model_cfg = OmegaConf.to_container(mv_cfg.model)
    assert mv_model_cfg.pop("views") == model_views(mv_cfg) and mv_model_cfg == src_model_cfg, "only model.views may differ"
    fk = dict(dino_hidden_size=DINO_HIDDEN, num_dino_layers=len(src_cfg.model.vision_encoder.feat_layers),
              task_cond_dim=DINO_HIDDEN, patch_size=PATCH)
    ref = create_action_model(src_cfg, **fk).eval()
    ref.load_state_dict(ck["model_state_dict"], strict=True)
    mv = create_action_model(mv_cfg, **fk).eval()
    missing = load_action_state(mv, ck["model_state_dict"])
    assert missing == ["view_emb"] and mv.num_views == 3 and float(mv.view_emb.abs().max()) == 0.0
    assert type(ref).__name__ == "VLAModel" and type(mv).__name__ == "MultiViewVLAModel"
    shared = {k: v for k, v in mv.state_dict().items() if k != "view_emb"}
    assert shared.keys() == ref.state_dict().keys() and all(torch.equal(v, ref.state_dict()[k]) for k, v in shared.items())

    B, V = 2, 3
    g = torch.Generator().manual_seed(1)
    feats = [torch.randn(B, V, TOKENS, DINO_HIDDEN, generator=g) for _ in range(fk["num_dino_layers"])]
    noisy = torch.randn(B, 32, 23, generator=g); qpos = torch.rand(B, 1, 61, generator=g) * 2 - 1
    cond = torch.randn(B, DINO_HIDDEN, generator=g); t = torch.rand(B, generator=g)
    report = {"stage3_checkpoint": str(STAGE3.relative_to(ROOT)), "stage3_step": ck.get("step"), "missing_keys": missing}
    with torch.no_grad():
        head3d = [f[:, 0] for f in feats]
        r = ref(t, noisy_actions=noisy, qpos_history=qpos, dino_features_list=head3d, task_cond=cond)
        m1 = mv(t, noisy_actions=noisy, qpos_history=qpos, dino_features_list=[f[:, :1] for f in feats], task_cond=cond)
        m0 = mv(t, noisy_actions=noisy, qpos_history=qpos, dino_features_list=head3d, task_cond=cond)
        diffs = {}
        for name, out in (("head_only_4d", m1), ("legacy_3d", m0)):
            for key in ("final_pred", "cond_tokens"):
                diffs[f"{name}.{key}"] = float((out[key] - r[key]).abs().max())
            diffs[f"{name}.future_feat"] = float((mv.future_feat_decoder(out["cond_tokens"]) - ref.future_feat_decoder(r["cond_tokens"])).abs().max())
        report["head_only_vs_stage3_max_abs_diff"] = diffs
        report["head_only_vs_stage3_final_pred_scale"] = float(r["final_pred"].abs().max())
        assert all(v == 0.0 for v in diffs.values()), diffs
        m3 = mv(t, noisy_actions=noisy, qpos_history=qpos, dino_features_list=feats, task_cond=cond)
        assert m3["final_pred"].shape == (B, 32, 23)
        n_obs = 1 + src_cfg.model.num_registers + 1   # proprio + registers + task token
        assert m3["cond_tokens"].shape == (B, n_obs + V * 64, 768) and r["cond_tokens"].shape == (B, n_obs + 64, 768)
        ff = mv.future_feat_decoder(m3["cond_tokens"])
        assert ff.shape == (B, 196, DINO_HIDDEN)
        report["three_view"] = {"final_pred": list(m3["final_pred"].shape), "cond_tokens": list(m3["cond_tokens"].shape),
                                "future_feat": list(ff.shape),
                                "final_pred_change_vs_head_only_max_abs": float((m3["final_pred"] - m1["final_pred"]).abs().max())}
    # Gradient reaches the new view embedding and the shared adapter through the upstream flow + future loss.
    vla, _ = load_upstream()
    mv.train()
    loss, info = vla.calc_flow_matching_loss(mv, x1=noisy.clamp(-1, 1), dino_features_list=feats, qpos_history=qpos, task_cond=cond,
                                             time_sampler="logit_normal", future_feat_target=torch.randn(B, 196, DINO_HIDDEN),
                                             use_future_feat=True, lambda_future_feat=0.5)
    loss.backward()
    grad_view = mv.view_emb.grad.abs().sum(dim=(1, 2, 3))
    assert torch.isfinite(loss) and (grad_view > 0).all() and mv.dino_adapters[0].query_embed.grad.abs().sum() > 0
    report["three_view"].update(loss=float(loss), loss_mse=info["loss_mse"], loss_future_feat=info["loss_future_feat"],
                                view_emb_grad_abs_sum=grad_view.tolist())
    report["params"] = {"stage3": sum(p.numel() for p in ref.parameters()), "multiview": sum(p.numel() for p in mv.parameters())}
    report["dataset"] = dataset_check(mv_cfg, args.orig_module)
    report["status"] = "PASS"
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
