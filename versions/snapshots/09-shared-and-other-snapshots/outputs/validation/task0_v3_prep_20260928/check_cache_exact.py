"""Exactness of the DINO feature cache vs the training path: dataset-style (B, V, 3, H, W) batch ->
wrapper.get_vision_features / get_future_target_features (exactly what training/deployment call)."""
import json, sys, numpy as np, torch
sys.path.insert(0, '/home/user/Robotic/project2')
from pathlib import Path
from omegaconf import OmegaConf
from harness import task_profile
from harness.wam_behavior import build_models, normalize_image
from harness.frame_cache import FramePack
cfg = OmegaConf.load('/home/user/Robotic/project2/outputs/training/task0_manip_v2_20260928/config.yaml')
w, _ = build_models(cfg, torch.device('cuda'))
w.eval()
out = Path(sys.argv[1]); ep = int(sys.argv[2])
rng = np.random.default_rng(0)
frames = np.load(out / w.views[0] / f'episode_{ep}.frames.npy')
for v in w.views:
    assert np.array_equal(np.load(out / v / f'episode_{ep}.frames.npy'), frames)
pick = np.sort(rng.choice(len(frames), 16, replace=False))
px = torch.from_numpy(np.stack([np.stack([normalize_image(FramePack(task_profile.FRAMES / v, ep).rgb(int(frames[i])))
                                          for v in w.views]) for i in pick]))       # (B, V, 3, H, W) training CPU path
with torch.no_grad():
    f = w.get_vision_features(px)                    # per layer (B, V, T, D) bf16
    ref_all = torch.stack(f, dim=2)                  # (B, V, L, T, D)
res = {}
for vi, view in enumerate(w.views):
    feat = np.load(out / view / f'episode_{ep}.feat.npy', mmap_mode='r')
    pool = np.load(out / view / f'episode_{ep}.pool.npy')
    ref = ref_all[:, vi]
    with torch.no_grad():
        fut = w.get_future_target_features(px[:, vi])   # (B, P, D) layer -4 patches (head-only in v1/v2 training)
    cached = torch.from_numpy(np.ascontiguousarray(feat[pick]).view(np.int16)).view(torch.bfloat16).to(ref.device)
    d = (cached.float() - ref.float()).abs()
    dfut = (cached[:, 2, 5:].float() - fut.float()).abs()
    last = ref[:, -1].float(); pooled_ref = torch.stack([last[:, 0], last[:, 5:].mean(1)], 1).cpu().numpy()
    rel = (d.mean() / ref.float().abs().mean()).item()
    res[view] = {"bit_equal_fraction": float((cached.contiguous().view(torch.int16) == ref.contiguous().view(torch.int16)).float().mean()),
                 "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "relative_mean_abs_diff": rel,
                 "future_target_bit_equal_fraction": float((cached[:, 2, 5:].contiguous().view(torch.int16) == fut.contiguous().view(torch.int16)).float().mean()),
                 "future_target_max_abs_diff": float(dfut.max()),
                 "pool_max_abs_diff": float(np.abs(pool[pick] - pooled_ref).max())}
print(json.dumps(res, indent=1))
