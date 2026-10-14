#!/usr/bin/env python3
"""Generic (task-agnostic, colour-free) arrival prototypes for every manipulation key: DINOv3 last-layer PATCH tokens.

Motivation (progress.md 2026-09-23 11:20, outputs/validation/dock_patch_gate_20260923/): the CLS-cosine arrival gate at
its fitted tau accepts oblique / too-far docks, and the hand-made HSV red-car feature is instance specific.  A spatially
aligned patch-token prototype (per grid position mean of L2-normalised last-layer patches over TRAIN demo frames at the
start of the manipulation segment) separates the demonstrated arrival pose from frames 3-10 s before arrival and from
every recorded mis-dock, using only features the policy already computes.  This script builds that prototype for EVERY
manipulation key in precond_meta.json (same key rule as the CLS prototypes), so any navigation->manipulation boundary of
any task gets the same gate without task-specific code.

Threshold rule (per key): the score at `--recall` (default 0.85) recall on VALIDATION positives; reported alongside the
false-accept rate on validation hard negatives (frames `--neg-seconds` before the manipulation start, when a navigation
segment precedes it).  Training-split frames only enter the prototype; validation frames only set/report thresholds.
No simulator ground truth is read.

Outputs: data/skill_vtt_task013/precond_patch_prototypes.npz (keys, proto[K,P,D] float16 unit vectors, thr[K])
         data/skill_vtt_task013/precond_patch_meta.json (protocol, counts, per-key recall / FAR, provenance)
"""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, VTT_DIR, SkillTable, normalize_image, load_upstream  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402
from harness.vtt_monitor import precondition_key  # noqa: E402

FPS = 30
NAV = "move to"


def align(frame, stride, length):
    return int(min(max((frame // stride) * stride, 0), ((length - 1) // stride) * stride))


class PatchEncoder:
    def __init__(self, dino, device, dtype, batch):
        import torch
        self.torch = torch
        _, runner = load_upstream()
        self.enc, self.hidden, self.n_reg, self.patch = runner.ModelFactory.create_vision_encoder(str(ROOT / dino), dtype, device)
        self.enc.eval()
        self.device, self.dtype, self.batch = device, dtype, batch

    def __call__(self, images):
        """list of normalised CHW float32 -> unit patch tokens (N, P, D) float32 (last hidden layer, registers dropped)."""
        torch = self.torch
        out_all = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch):
                px = torch.from_numpy(np.stack(images[i:i + self.batch])).to(self.device, self.dtype)
                out = self.enc(pixel_values=px, output_hidden_states=False, return_dict=True)
                p = out.last_hidden_state[:, 1 + self.n_reg:].float()
                p = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
                out_all.append(p.cpu().numpy())
        return np.concatenate(out_all)


def l2n(x, axis=-1):
    x = np.asarray(x, np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def spatial_cos(unit_patches, proto_unit):
    """(N,P,D) unit patches vs (P,D) unit prototype -> (N,) mean over positions of per-position cosine."""
    return (unit_patches * proto_unit[None]).sum(-1).mean(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dino", default="models/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--anchors", default="0.0,0.05,0.1", help="fractions into the manipulation segment used as positives")
    ap.add_argument("--neg-seconds", default="3,6,10", help="seconds before the manipulation start used as hard negatives")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--cuda-fraction", type=float, default=0.3)
    ap.add_argument("--out-dir", type=Path, default=VTT_DIR)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_fraction)
    t0 = time.time()
    anchors = [float(x) for x in args.anchors.split(",")]
    neg_seconds = [float(x) for x in args.neg_seconds.split(",")]
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())

    # ---- jobs --------------------------------------------------------------------------------------------------
    pos = []   # (ep, split, frame, key, frac)
    neg = []   # (ep, split, frame, key, offset_s)  validation only (thresholds are validation-side)
    for ep in sorted(table.episodes):
        info = table.episodes[ep]
        split = index["episodes"][str(ep)]["split"]
        try:
            pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        except Exception:
            continue
        segs, L, stride = info["segments"], pack.length, pack.stride
        for i, s in enumerate(segs):
            if s["skill"] == NAV:
                continue
            key = precondition_key(s["skill"], s["objects"])
            lo, hi = s["start"], s["end"]
            for fr in anchors:
                pos.append((ep, split, align(int(lo + fr * (hi - lo)), stride, L), key, fr))
            prev = segs[i - 1] if i > 0 and segs[i - 1]["skill"] == NAV else None
            if prev is not None and split == "validation":
                for sec in neg_seconds:
                    a = align(lo - int(sec * FPS), stride, L)
                    if a >= prev["start"]:
                        neg.append((ep, split, a, key, sec))
    train_counts = defaultdict(int)
    for ep, split, fr, key, _ in pos:
        if split == "train" and _ == anchors[0]:
            train_counts[key] += 1
    keys = sorted(k for k, c in train_counts.items() if c >= args.min_train)
    kidx = {k: i for i, k in enumerate(keys)}
    print(f"keys {len(keys)}; positives {len(pos)} (train {sum(p[1] == 'train' for p in pos)}), validation hard negatives {len(neg)}", flush=True)

    enc = PatchEncoder(args.dino, args.device, torch.bfloat16 if args.device == "cuda" else torch.float32, args.batch)

    def encode_jobs(jobs, log_every=400):
        """jobs sorted by episode; yields (job, unit patches) in batches without holding everything in RAM."""
        buf, ids, pack_ep, pack, done = [], [], None, None, 0
        for job in jobs:
            ep, fr = job[0], job[2]
            if pack_ep != ep:
                pack, pack_ep = FramePack(FRAMES / HEAD_CAMERA, ep), ep
            buf.append(normalize_image(pack.rgb(fr)))
            ids.append(job)
            if len(buf) == args.batch:
                yield from zip(ids, enc(buf))
                done += len(buf)
                buf, ids = [], []
                if done % log_every < args.batch:
                    el = time.time() - t0
                    print(f"[embed] {done}/{len(jobs)} {el:.0f}s", flush=True)
        if buf:
            yield from zip(ids, enc(buf))

    # ---- pass 1: prototypes from TRAIN positives (running per-key sums, no frame storage) -------------------------
    P, D = None, None
    sums, counts = {}, defaultdict(int)
    train_jobs = sorted([p for p in pos if p[1] == "train" and p[3] in kidx], key=lambda j: (j[0], j[2]))
    for job, up in encode_jobs(train_jobs):
        if P is None:
            P, D = up.shape
        k = job[3]
        sums[k] = sums.get(k, 0.0) + up.astype(np.float64)
        counts[k] += 1
    proto = np.stack([l2n(sums[k] / counts[k]) for k in keys]).astype(np.float32)  # (K, P, D) unit per position

    # ---- pass 2: validation positives / hard negatives -> thresholds and reporting ----------------------------------
    val_jobs = sorted([p for p in pos if p[1] == "validation" and p[3] in kidx] + [(n[0], n[1], n[2], n[3], -n[4]) for n in neg if n[3] in kidx],
                      key=lambda j: (j[0], j[2]))
    val_scores = defaultdict(list)   # key -> [(frac_or_negoffset, score)]
    for job, up in encode_jobs(val_jobs):
        val_scores[job[3]].append((job[4], float(spatial_cos(up[None], proto[kidx[job[3]]])[0])))

    thr = np.full(len(keys), np.nan, np.float32)
    per_key = {}
    for k in keys:
        rows = val_scores.get(k, [])
        p = np.array([s for f, s in rows if f >= 0], np.float32)
        n_by = {sec: np.array([s for f, s in rows if f == -sec], np.float32) for sec in neg_seconds}
        entry = {"train_segments": int(counts[k]), "val_positives": int(len(p)), "val_hardneg": {str(s): int(len(v)) for s, v in n_by.items()}}
        if len(p) >= 5:
            t = float(np.percentile(p, 100.0 * (1.0 - args.recall)))
            thr[kidx[k]] = t
            entry.update({"threshold": t, "recall_val": float((p >= t).mean()),
                          "val_pos_mean": float(p.mean()), "val_pos_std": float(p.std()), "val_pos_min": float(p.min()),
                          "far_hardneg": {str(s): (float((v >= t).mean()) if len(v) else None) for s, v in n_by.items()},
                          "hardneg_mean": {str(s): (float(v.mean()) if len(v) else None) for s, v in n_by.items()}})
        else:
            entry["threshold"] = None
        per_key[k] = entry
        print(k, json.dumps({kk: vv for kk, vv in entry.items() if kk in ("threshold", "recall_val", "far_hardneg", "val_pos_mean", "hardneg_mean")}), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "precond_patch_prototypes.npz", keys=np.array(keys, dtype=str), proto=proto.astype(np.float16), thr=thr)
    dino_dir = ROOT / args.dino
    weights = sorted(dino_dir.glob("*.safetensors")) + sorted(dino_dir.glob("*.bin"))
    meta = {"format": "precond-patch-arrival-v1", "dino": args.dino,
            "dino_weights_sha256": {w.name: hashlib.sha256(w.read_bytes()).hexdigest() for w in weights[:1]},
            "feature": "DINOv3 last_hidden_state patch tokens (registers dropped), per-token L2 normalised, 14x14 grid at 224",
            "score": "mean over grid positions of cosine(current unit patch, per-position unit prototype)",
            "prototype": "per-position mean of unit patches over TRAIN start frames of the key's manipulation segment, re-normalised",
            "anchors": anchors, "hardneg_seconds": neg_seconds, "threshold_rule": f"score at {args.recall:.2f} recall on VALIDATION positives",
            "grid": [int(P), int(D)], "keys": keys, "min_train": args.min_train, "per_key": per_key,
            "prototypes_use_validation": False, "ground_truth_used": False,
            "sources": ["data/behavior-1k/2026-challenge-demos/skill_segments_task013.json", "data/frame_cache", "data/lowdim_cache/index.json"],
            "seconds": time.time() - t0}
    (args.out_dir / "precond_patch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"keys": len(keys), "grid": [P, D], "seconds": round(meta["seconds"], 1),
                      "thresholds": {k: (None if np.isnan(thr[kidx[k]]) else round(float(thr[kidx[k]]), 3)) for k in keys}}, indent=1))


if __name__ == "__main__":
    main()
