#!/usr/bin/env python3
"""Frozen DINOv3 feature cache for long WAM training (task via P2_TASK_ID, like every data script).

The vision encoder is frozen and runs in eval mode, so its hidden states for a cached 224 JPEG frame are a fixed
function of that frame. Training re-computed them every step (~60% of the step time, 2026-09-28 benchmark). This
script encodes, once, every frame a manipulation-scope training/validation sample can touch (current, future and
history frames all lie inside one annotated non-'move to' segment: even frames in [ceil_even(start), end)) with the
same pass as training and deployment (harness.wam_behavior.normalize_image -> bf16 -> hidden_states at the model's
feat_layers, CLS + registers kept) and stores, per camera view and episode:

  <out>/<view>/episode_<ep>.feat.npy    uint16 = bf16 bit pattern, (N, L, T, D): frames x feat layers x (1+R+P) x 1024
  <out>/<view>/episode_<ep>.pool.npy    float32 (N, 2, D): [CLS, mean patch] of the LAST feat layer (memory tokens)
  <out>/<view>/episode_<ep>.frames.npy  int64 (N,) sorted frame indices of the rows
  <out>/<view>/episode_<ep>.meta.json   complete flag, shapes, source frame-cache meta SHA-256

The stored bits are the bf16 activations the model consumes (no re-quantization). Resumable (complete episodes are
skipped). A duty cycle keeps the average GPU busy fraction <= --duty; VRAM/RAM guards abort on the project limits.
Training-time image augmentation is not possible on cached features (recorded in the training manifest).

  P2_TASK_ID=0 .conda-envs/lilawam/bin/python scripts/cache_dino_features.py --config <train.yaml> [--dry-run]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import task_profile  # noqa: E402
from harness.wam_behavior import SkillTable, model_views, normalize_image  # noqa: E402


def manipulation_frames(segments, length, stride=2):
    """Sorted even frame indices inside every non-'move to' annotated segment: [ceil_even(start), min(end, length))."""
    keep = set()
    for seg in segments:
        if seg["skill"] == "move to":
            continue
        lo = int(seg["start"]) + (-int(seg["start"]) % stride)
        keep.update(range(lo, min(int(seg["end"]), length), stride))
    return np.array(sorted(keep), dtype=np.int64)


class FrameList(torch.utils.data.Dataset):
    def __init__(self, view_dir, episode, frames):
        self.view_dir, self.episode, self.frames = view_dir, episode, frames
        self.pack = None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        if self.pack is None:
            from harness.frame_cache import FramePack
            self.pack = FramePack(self.view_dir, self.episode)
        rgb = self.pack.rgb(int(self.frames[i]))
        if rgb.shape[:2] != (224, 224):
            raise ValueError(f"frame cache is {rgb.shape}, expected 224x224 (training never resized it)")
        # normalize_image on the CPU exactly as BehaviorChunkDataset.frame / the policy server: GPU division by a
        # scalar is computed as multiplication by the reciprocal and changes low bits (2026-09-28 exactness check)
        return torch.from_numpy(normalize_image(rgb))


def gpu_state():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10).stdout.strip().split(",")
    return int(q[0]), float(q[1])


def ram_percent():
    info = dict(line.split(":")[0:2] for line in open("/proc/meminfo").read().splitlines()[:3])
    total, avail = int(info["MemTotal"].split()[0]), int(info["MemAvailable"].split()[0])
    return 100.0 * (total - avail) / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training config (views, feat_layers, encoder)")
    ap.add_argument("--out", default=None, help="default data/dino_cache_task{id:03d}")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--duty", type=float, default=0.8)
    ap.add_argument("--vram-abort-mib", type=int, default=15300)
    ap.add_argument("--ram-abort-percent", type=float, default=85.0)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / args.config)
    if int(cfg.dataset.get("task_id", 13)) != task_profile.TASK_ID:
        raise SystemExit(f"config task {cfg.dataset.get('task_id')} != P2_TASK_ID {task_profile.TASK_ID}")
    views = model_views(cfg)
    feat_layers = [int(x) for x in cfg.model.vision_encoder.feat_layers]
    if not cfg.model.vision_encoder.include_cls_register:
        raise SystemExit("cache stores CLS+register tokens; config drops them")
    stride = int(cfg.dataset.frame_stride)
    out = Path(args.out) if args.out else ROOT / f"data/dino_cache_task{task_profile.TASK_ID:03d}"
    index = json.loads((task_profile.LOWDIM / "index.json").read_text())
    skills = SkillTable()
    episodes = sorted(int(k) for k in index["episodes"] if args.episodes is None or int(k) in args.episodes)
    plan = {ep: manipulation_frames(skills.episodes[ep]["segments"], index["episodes"][str(ep)]["length"], stride)
            for ep in episodes}
    n_frames = sum(len(f) for f in plan.values())
    tokens = 1 + 4 + (224 // 16) ** 2
    bytes_per = len(feat_layers) * tokens * 1024 * 2
    summary = {"task_id": task_profile.TASK_ID, "views": views, "feat_layers": feat_layers, "episodes": len(episodes),
               "frames_per_view": n_frames, "images": n_frames * len(views), "tokens_per_image": tokens,
               "bytes_feat": n_frames * len(views) * bytes_per, "gib_feat": round(n_frames * len(views) * bytes_per / 2**30, 1)}
    print(json.dumps(summary), flush=True)
    if args.dry_run:
        return 0

    from harness.wam_behavior import load_upstream
    _, runner = load_upstream()
    device = torch.device("cuda")
    enc, hidden, n_reg, patch = runner.ModelFactory.create_vision_encoder(
        str(ROOT / cfg.model.vision_encoder.checkpoint_path), torch.bfloat16, device)
    if (hidden, n_reg, patch) != (1024, 4, 16):
        raise SystemExit(f"unexpected encoder geometry {(hidden, n_reg, patch)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache_config.json").write_text(json.dumps({**summary, "config": str(args.config), "stride": stride,
                                                         "encoder": str(cfg.model.vision_encoder.checkpoint_path),
                                                         "encoder_revision": str(cfg.model.vision_encoder.get("revision")),
                                                         "dtype": "bf16 bit pattern stored as uint16",
                                                         "pool": "float32 [CLS, mean over patch tokens] of feat layer -1 of the list",
                                                         "rule": "even frames in [ceil_even(start), end) of every non-'move to' segment"}, indent=2))
    t_start, done_images, busy_total = time.time(), 0, 0.0
    for view in views:
        vdir = out / view
        vdir.mkdir(exist_ok=True)
        for ep in episodes:
            frames = plan[ep]
            meta_path = vdir / f"episode_{ep}.meta.json"
            if meta_path.exists() and json.loads(meta_path.read_text()).get("complete"):
                continue
            if not len(frames):
                continue
            src_meta = task_profile.FRAMES / view / f"episode_{ep}.meta.json"
            feat_tmp = vdir / f"episode_{ep}.feat.tmp.npy"
            feat = np.lib.format.open_memmap(feat_tmp, mode="w+", dtype=np.uint16,
                                             shape=(len(frames), len(feat_layers), tokens, 1024))
            pool = np.zeros((len(frames), 2, 1024), np.float32)
            loader = torch.utils.data.DataLoader(FrameList(task_profile.FRAMES / view, ep, frames), batch_size=args.batch,
                                                 shuffle=False, num_workers=args.workers, pin_memory=True)
            row = 0
            for img in loader:
                t0 = time.time()
                with torch.no_grad():
                    px = img.to(device, torch.bfloat16)   # the wrapper's cast: pixel_values.to(device, encoder_dtype)
                    hs = enc(pixel_values=px, output_hidden_states=True, return_dict=True).hidden_states
                    h = torch.stack([hs[l] for l in feat_layers], dim=1)          # (B, L, T, D) bf16
                    last = h[:, -1].float()
                    pooled = torch.stack([last[:, 0], last[:, 1 + n_reg:].mean(dim=1)], dim=1)
                    bits = h.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
                torch.cuda.synchronize()
                busy = time.time() - t0
                n = bits.shape[0]
                feat[row:row + n] = bits
                pool[row:row + n] = pooled.cpu().numpy()
                row += n
                done_images += n
                busy_total += busy
                time.sleep(busy * (1 - args.duty) / args.duty)
                vram, util = gpu_state()
                ram = ram_percent()
                if vram >= args.vram_abort_mib or ram >= args.ram_abort_percent:
                    raise SystemExit(f"resource guard: VRAM {vram} MiB, RAM {ram:.1f}%")
            if row != len(frames):
                raise SystemExit(f"episode {ep} {view}: wrote {row} of {len(frames)} rows")
            feat.flush()
            del feat
            os.replace(feat_tmp, vdir / f"episode_{ep}.feat.npy")
            np.save(vdir / f"episode_{ep}.pool.npy", pool)
            np.save(vdir / f"episode_{ep}.frames.npy", frames)
            meta_path.write_text(json.dumps({"complete": True, "episode": ep, "view": view, "rows": int(len(frames)),
                                             "feat_shape": [int(len(frames)), len(feat_layers), tokens, 1024],
                                             "feat_layers": feat_layers, "stride": stride,
                                             "source_frame_meta_sha256": hashlib.sha256(src_meta.read_bytes()).hexdigest()}))
            el = time.time() - t_start
            print(json.dumps({"event": "episode", "view": view, "episode": ep, "rows": int(len(frames)),
                              "images_done": done_images, "elapsed_min": round(el / 60, 1),
                              "img_per_s": round(done_images / max(el, 1e-6), 1),
                              "busy_fraction": round(busy_total / max(el, 1e-6), 2)}), flush=True)
    print(json.dumps({"event": "done", "images": done_images, "elapsed_min": round((time.time() - t_start) / 60, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
