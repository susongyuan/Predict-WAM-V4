#!/usr/bin/env python3
"""Generic policy-visible POSTCONDITION prototypes for every manipulation key (DINOv3 last-layer PATCH tokens of the
demonstrated END frames), the completion counterpart of scripts/build_patch_arrival_prototypes.py.

Why: the orchestrator now requires an independent, policy-visible postcondition before a manipulation skill may advance
(VTT alignment only nominates a closure candidate; progress.md 2026-09-23).  The same spatially aligned patch prototype
that separates docking poses also separates "skill finished" appearance (lid open, door open, object placed) from the
"still working" appearance, using only features the policy already computes.  One prototype per manipulation key;
no task-specific code; no simulator ground truth.

Positives  : TRAIN segment frames at --anchors (fractions near the segment end, default 0.9,0.95,1.0).
Score      : --score contrast (default): per grid position, the direction d[p] = proto_end[p] - proto_start[p]
             (start prototype from the same TRAIN segments at --start-anchors); score(frame) =
             sum_p <unit_patch[p], d[p]> / sum_p |d[p]|.  Shared scene content (same object seen from the same
             docking pose) cancels and positions that change when the skill completes dominate, so an unfinished
             frame scores near/below zero.  --score mean_cos is the plain arrival-style similarity to proto_end.
Threshold  : VALIDATION positives give the score at --recall recall; VALIDATION frames at --neg-fractions of the SAME
             segment (default 0.0,0.2,0.4 = clearly unfinished) give the score at --max-far false-accept rate; the
             deployed threshold is the max of the two (fails closed).  Fraction 0.6 frames are reported separately
             as an ambiguous band (skill partly done).  Training frames never set thresholds.

Outputs: data/skill_vtt_task013/postcond_patch_prototypes.npz (keys, proto[K,P,D] float16, thr[K])
         data/skill_vtt_task013/postcond_patch_meta.json (protocol, per-key recall / FAR, provenance)
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
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, VTT_DIR, SkillTable, normalize_image  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402
from harness.vtt_monitor import precondition_key  # noqa: E402
from scripts.build_patch_arrival_prototypes import PatchEncoder, align, l2n, spatial_cos  # noqa: E402

NAV = "move to"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dino", default="models/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--anchors", default="0.9,0.95,1.0", help="fractions into the manipulation segment used as positives")
    ap.add_argument("--start-anchors", default="0.0,0.05,0.1", help="fractions used for the start prototype (contrast score)")
    ap.add_argument("--score", default="contrast", choices=("contrast", "mean_cos"))
    ap.add_argument("--neg-fractions", default="0.0,0.2,0.4", help="same-segment fractions used as unfinished negatives")
    ap.add_argument("--ambiguous-fractions", default="0.6", help="reported only (partly done)")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--max-far", type=float, default=0.05)
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
    start_anchors = [float(x) for x in args.start_anchors.split(",")] if args.score == "contrast" else []
    neg_fr = [float(x) for x in args.neg_fractions.split(",")]
    amb_fr = [float(x) for x in args.ambiguous_fractions.split(",")] if args.ambiguous_fractions else []
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())

    pos, neg = [], []   # (ep, split, frame, key, tag, frac)
    for ep in sorted(table.episodes):
        info = table.episodes[ep]
        split = index["episodes"][str(ep)]["split"]
        try:
            pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        except Exception:
            continue
        L, stride = pack.length, pack.stride
        for s in info["segments"]:
            if s["skill"] == NAV:
                continue
            key = precondition_key(s["skill"], s["objects"])
            lo, hi = s["start"], s["end"]
            for fr in anchors:
                pos.append((ep, split, align(int(lo + fr * (hi - lo - 1)), stride, L), key, "pos", fr))
            if split == "train":
                for fr in start_anchors:
                    pos.append((ep, split, align(int(lo + fr * (hi - lo - 1)), stride, L), key, "start", fr))
            if split == "validation":
                for fr in neg_fr + amb_fr:
                    neg.append((ep, split, align(int(lo + fr * (hi - lo - 1)), stride, L), key, "neg", fr))
    train_counts = defaultdict(int)
    for job in pos:
        if job[1] == "train" and job[4] == "pos" and job[5] == anchors[0]:
            train_counts[job[3]] += 1
    keys = sorted(k for k, c in train_counts.items() if c >= args.min_train)
    kidx = {k: i for i, k in enumerate(keys)}
    print(f"keys {len(keys)}; positives {len(pos)} (train {sum(p[1] == 'train' for p in pos)}), validation negatives {len(neg)}", flush=True)

    enc = PatchEncoder(args.dino, args.device, torch.bfloat16 if args.device == "cuda" else torch.float32, args.batch)

    def encode_jobs(jobs, log_every=400):
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
                    print(f"[embed] {done}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        if buf:
            yield from zip(ids, enc(buf))

    P = D = None
    sums, counts = {}, defaultdict(int)
    sums0, counts0 = {}, defaultdict(int)
    train_jobs = sorted([p for p in pos if p[1] == "train" and p[3] in kidx], key=lambda j: (j[0], j[2]))
    for job, up in encode_jobs(train_jobs):
        if P is None:
            P, D = up.shape
        if job[4] == "pos":
            sums[job[3]] = sums.get(job[3], 0.0) + up.astype(np.float64)
            counts[job[3]] += 1
        else:
            sums0[job[3]] = sums0.get(job[3], 0.0) + up.astype(np.float64)
            counts0[job[3]] += 1
    proto = np.stack([l2n(sums[k] / counts[k]) for k in keys]).astype(np.float32)
    if args.score == "contrast":
        proto0 = np.stack([l2n(sums0[k] / counts0[k]) for k in keys]).astype(np.float32)
        direction = proto - proto0                                   # (K, P, D), not re-normalised
        dir_norm_sum = np.linalg.norm(direction, axis=-1).sum(-1)    # (K,)

        def score_fn(up, i):
            return float((up[0] * direction[i]).sum() / (dir_norm_sum[i] + 1e-8))
    else:
        direction, dir_norm_sum = None, None

        def score_fn(up, i):
            return float(spatial_cos(up, proto[i])[0])

    val_jobs = sorted([p for p in pos if p[1] == "validation" and p[3] in kidx] + [n for n in neg if n[3] in kidx],
                      key=lambda j: (j[0], j[2]))
    val_scores = defaultdict(list)   # key -> [(tag, frac, score)]
    for job, up in encode_jobs(val_jobs):
        val_scores[job[3]].append((job[4], job[5], score_fn(up[None], kidx[job[3]])))

    thr = np.full(len(keys), np.nan, np.float32)
    per_key = {}
    for k in keys:
        rows = val_scores.get(k, [])
        p = np.array([s for tag, f, s in rows if tag == "pos"], np.float32)
        n_by = {f: np.array([s for tag, ff, s in rows if tag == "neg" and ff == f], np.float32) for f in neg_fr + amb_fr}
        hard = np.concatenate([n_by[f] for f in neg_fr]) if neg_fr else np.zeros(0, np.float32)
        entry = {"train_segments": int(counts[k]), "val_positives": int(len(p)),
                 "val_negatives": {str(f): int(len(v)) for f, v in n_by.items()}}
        if len(p) >= 5:
            t_recall = float(np.percentile(p, 100.0 * (1.0 - args.recall)))
            t_far = float(np.percentile(hard, 100.0 * (1.0 - args.max_far))) if len(hard) else -np.inf
            t = max(t_recall, t_far)
            thr[kidx[k]] = t
            entry.update({"threshold": t, "threshold_recall_rule": t_recall, "threshold_far_rule": (None if not np.isfinite(t_far) else t_far),
                          "recall_val": float((p >= t).mean()), "val_pos_mean": float(p.mean()), "val_pos_std": float(p.std()),
                          "val_pos_min": float(p.min()),
                          "far_by_fraction": {str(f): (float((v >= t).mean()) if len(v) else None) for f, v in n_by.items()},
                          "neg_mean_by_fraction": {str(f): (float(v.mean()) if len(v) else None) for f, v in n_by.items()}})
        else:
            entry["threshold"] = None
        per_key[k] = entry
        print(k, json.dumps({kk: vv for kk, vv in entry.items() if kk in ("threshold", "recall_val", "far_by_fraction", "val_pos_mean")}), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extra = {} if direction is None else {"direction": direction.astype(np.float16), "dir_norm_sum": dir_norm_sum.astype(np.float32)}
    np.savez(args.out_dir / "postcond_patch_prototypes.npz", keys=np.array(keys, dtype=str), proto=proto.astype(np.float16), thr=thr,
             mode=np.array(args.score), **extra)
    dino_dir = ROOT / args.dino
    weights = sorted(dino_dir.glob("*.safetensors")) + sorted(dino_dir.glob("*.bin"))
    meta = {"format": "postcond-patch-v1", "dino": args.dino,
            "dino_weights_sha256": {w.name: hashlib.sha256(w.read_bytes()).hexdigest() for w in weights[:1]},
            "feature": "DINOv3 last_hidden_state patch tokens (registers dropped), per-token L2 normalised, 14x14 grid at 224",
            "score": ("sum_p <unit patch[p], proto_end[p]-proto_start[p]> / sum_p |proto_end[p]-proto_start[p]|" if args.score == "contrast"
                      else "mean over grid positions of cosine(current unit patch, per-position unit prototype)"),
            "score_mode": args.score, "start_anchors": start_anchors,
            "prototype": "per-position mean of unit patches over TRAIN END frames (anchors) of the key's manipulation segment, re-normalised",
            "anchors": anchors, "neg_fractions": neg_fr, "ambiguous_fractions": amb_fr,
            "threshold_rule": f"max(score at {args.recall:.2f} recall on VALIDATION positives, score at {args.max_far:.2f} FAR on VALIDATION unfinished frames)",
            "grid": [int(P), int(D)], "keys": keys, "min_train": args.min_train, "per_key": per_key,
            "prototypes_use_validation": False, "ground_truth_used": False, "policy_visible": True,
            "sources": ["data/behavior-1k/2026-challenge-demos/skill_segments_task013.json", "data/frame_cache", "data/lowdim_cache/index.json"],
            "seconds": time.time() - t0}
    (args.out_dir / "postcond_patch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"keys": len(keys), "grid": [P, D], "seconds": round(meta["seconds"], 1),
                      "thresholds": {k: (None if np.isnan(thr[kidx[k]]) else round(float(thr[kidx[k]]), 3)) for k in keys}}, indent=1))


if __name__ == "__main__":
    main()
