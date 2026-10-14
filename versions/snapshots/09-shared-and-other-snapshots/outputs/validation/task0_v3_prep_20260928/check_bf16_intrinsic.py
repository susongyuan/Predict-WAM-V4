"""Intrinsic variability of the frozen bf16 DINO pass across batch composition (training B*V batch vs deployment
single-sample 3-view batch), to judge whether the cache's ~1% deviation is noise of the same kind."""
import json, sys, numpy as np, torch
sys.path.insert(0, '/home/user/Robotic/project2')
from pathlib import Path
from omegaconf import OmegaConf
from harness import task_profile
from harness.wam_behavior import build_models, normalize_image
from harness.frame_cache import FramePack
cfg = OmegaConf.load('/home/user/Robotic/project2/outputs/training/task0_manip_v2_20260928/config.yaml')
w, _ = build_models(cfg, torch.device('cuda')); w.eval()
out = Path(sys.argv[1]); ep = int(sys.argv[2])
rng = np.random.default_rng(0)
frames = np.load(out / w.views[0] / f'episode_{ep}.frames.npy')
pick = np.sort(rng.choice(len(frames), 16, replace=False))
px = torch.from_numpy(np.stack([np.stack([normalize_image(FramePack(task_profile.FRAMES / v, ep).rgb(int(frames[i])))
                                          for v in w.views]) for i in pick]))
def feats(p):
    with torch.no_grad():
        return torch.stack(w.get_vision_features(p), dim=2).float()   # (B, V, L, T, D)
a = feats(px)                                            # training-like: 16 samples x 3 views in one batch
b = torch.cat([feats(px[i:i + 1]) for i in range(len(px))])   # deployment-like: one sample (3 views) per call
c = torch.cat([feats(px[i:i + 4]) for i in range(0, len(px), 4)])
cached = torch.stack([torch.from_numpy(np.ascontiguousarray(np.load(out / v / f'episode_{ep}.feat.npy', mmap_mode='r')[pick]).view(np.int16)).view(torch.bfloat16)
                      for v in w.views], dim=1).float().to(a.device)
def cmp(x, y):
    d = (x - y).abs()
    return {"rel_mean_abs": round((d.mean() / y.abs().mean()).item(), 5), "max_abs": float(d.max()),
            "bit_equal": round(float((x == y).float().mean()), 4),
            "cls_cos_min": round(float(torch.nn.functional.cosine_similarity(x[:, :, -1, 0], y[:, :, -1, 0], dim=-1).min()), 5)}
print(json.dumps({"train_B16_vs_deploy_B1": cmp(b, a), "train_B16_vs_B4": cmp(c, a),
                  "cache_vs_train_B16": cmp(cached, a), "cache_vs_deploy_B1": cmp(cached, b)}, indent=1))
