#!/usr/bin/env python3
"""Compute Visual Transition Tokens (VTT) per skill segment and per skill class from cached head frames.

VTT(segment) = DINOv3 last-layer CLS(last cached frame of segment) - CLS(first cached frame).
Prototypes are the mean over TRAIN segments of each skill class; validation instance VTTs are
stored but never averaged in. Also writes a task-level vector (episode first/last, train only)
for the upstream-style single-condition baseline, and separation diagnostics.

Outputs in harness.task_profile.VTT_DIR (P2_TASK_ID; default 13 = data/skill_vtt_task013/, Task 1 =
data/skill_vtt_task001/): instances.npy, instances_index.json, prototypes.npy, task_level.npy, meta.json,
diagnostics.json. Segments/splits come from task_profile.SEGMENTS; an existing output directory is never
overwritten unless --overwrite is given (protects the Task 13 artifacts pinned by trained checkpoints).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import task_profile  # noqa
from harness.wam_behavior import DATA, FRAMES, LOWDIM, VTT_DIR, HEAD_CAMERA, SkillTable, normalize_image, load_upstream  # noqa
from harness.frame_cache import FramePack  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", default="models/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true", help="replace existing VTT artifacts in the task's VTT_DIR")
    args = ap.parse_args()
    if (VTT_DIR / "instances.npy").exists() and not args.overwrite:
        raise SystemExit(f"{VTT_DIR} already holds VTT artifacts (P2_TASK_ID={task_profile.TASK_ID}); pass --overwrite to replace")
    torch.set_num_threads(4)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.5)
    _, runner = load_upstream()
    enc, hidden, _, _ = runner.ModelFactory.create_vision_encoder(str(ROOT / args.dino), torch.bfloat16, device)
    skills = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    for ep, info in skills.episodes.items():  # prototypes/task vector are TRAIN-only: splits must agree with the cache
        if index["episodes"][str(ep)]["split"] != info["split"]:
            raise ValueError(f"episode {ep}: split differs between {LOWDIM / 'index.json'} and the skill segments")
    jobs = []  # (episode, seg_i, first_frame, last_frame, class, split)
    ep_jobs = []
    for ep, info in sorted(skills.episodes.items()):
        L = info["length"]; sp = info["split"]
        for i, s in enumerate(info["segments"]):
            first = ((s["start"] + args.stride - 1) // args.stride) * args.stride
            last = ((s["end"] - 1) // args.stride) * args.stride
            if last <= first:
                last = first
            jobs.append((ep, i, min(first, L - 1), min(last, L - 1), s["class"], sp))
        ep_jobs.append((ep, 0, ((L - 1) // args.stride) * args.stride, sp))

    @torch.no_grad()
    def cls_batch(frames):
        px = torch.from_numpy(np.stack(frames)).to(device, torch.bfloat16)
        return enc(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :].float().cpu().numpy()

    def embed(pairs):
        """pairs: list of (ep, frame). Returns (N, D) CLS in order, batching decodes per episode."""
        out = np.zeros((len(pairs), hidden), dtype=np.float32)
        packs = {}
        buf, ids = [], []
        for k, (ep, fr) in enumerate(pairs):
            if ep not in packs:
                packs = {ep: FramePack(FRAMES / HEAD_CAMERA, ep)}
            buf.append(normalize_image(packs[ep].rgb(fr))); ids.append(k)
            if len(buf) == args.batch:
                out[ids] = cls_batch(buf); buf, ids = [], []
        if buf:
            out[ids] = cls_batch(buf)
        return out

    t0 = time.time()
    firsts = embed([(j[0], j[2]) for j in jobs]); lasts = embed([(j[0], j[3]) for j in jobs])
    inst = lasts - firsts
    ep_first = embed([(e[0], e[1]) for e in ep_jobs]); ep_last = embed([(e[0], e[2]) for e in ep_jobs])
    ep_diff = ep_last - ep_first
    n_cls = skills.num_classes
    protos = np.zeros((n_cls, hidden), dtype=np.float32); counts = np.zeros(n_cls, dtype=int)
    for j, v in zip(jobs, inst):
        if j[5] == "train":
            protos[j[4]] += v; counts[j[4]] += 1
    protos /= np.maximum(counts, 1)[:, None]
    task_level = ep_diff[[k for k, e in enumerate(ep_jobs) if e[3] == "train"]].mean(0)

    # Diagnostics: does VTT separate skills? cosine of each train instance to all prototypes -> nearest-prototype accuracy
    def cos(a, b):
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8); b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        return a @ b.T
    names = {v: k for k, v in skills.classes.items()}
    diag = {}
    for sp in ("train", "validation"):
        sel = [k for k, j in enumerate(jobs) if j[5] == sp]
        sims = cos(inst[sel], protos)
        pred = sims.argmax(1); truth = np.array([jobs[k][4] for k in sel])
        conf = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(pred, truth):
            conf[t, p] += 1
        diag[sp] = {"nearest_prototype_accuracy": float((pred == truth).mean()), "n": len(sel),
                    "per_class_recall": {names[c]: float(conf[c, c] / max(conf[c].sum(), 1)) for c in range(n_cls)},
                    "confusion_rows_true_cols_pred": conf.tolist()}
    diag["prototype_cosine_matrix"] = cos(protos, protos).round(3).tolist()
    diag["prototype_norms"] = np.linalg.norm(protos, axis=1).round(3).tolist()
    diag["class_names"] = [names[c] for c in range(n_cls)]
    diag["train_counts"] = counts.tolist()
    diag["note"] = ("Nearest-prototype accuracy on validation segments measures whether DINOv3 CLS transitions "
                    f"separate the {n_cls} annotated skills; chance = 1/{n_cls}. Not a policy result.")

    VTT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VTT_DIR / "instances.npy", inst); np.save(VTT_DIR / "prototypes.npy", protos); np.save(VTT_DIR / "task_level.npy", task_level)
    (VTT_DIR / "instances_index.json").write_text(json.dumps(
        [{"episode": j[0], "segment": j[1], "first_frame": j[2], "last_frame": j[3], "class": j[4], "split": j[5]} for j in jobs]))
    (VTT_DIR / "meta.json").write_text(json.dumps({
        "dino": args.dino, "dino_sha256": json.loads((ROOT / args.dino / "DOWNLOAD_PROVENANCE.json").read_text())["files"]["model.safetensors"]["sha256"],
        "frame_stride": args.stride, "image_size": 224, "camera": HEAD_CAMERA, "feature": "last_hidden_state CLS (upstream precompute_task_cond recipe)",
        "classes": skills.classes, "prototype_train_counts": counts.tolist(), "prototypes_use_validation": False,
        "skill_segments_sha256": hashlib.sha256(task_profile.SEGMENTS.read_bytes()).hexdigest(),
        **({} if task_profile.LEGACY else {"task_id": task_profile.TASK_ID, "task_name": task_profile.TASK_NAME,
                                          "skill_segments": str(task_profile.SEGMENTS.relative_to(ROOT))}),
        "seconds": round(time.time() - t0, 1)}, indent=1))
    (VTT_DIR / "diagnostics.json").write_text(json.dumps(diag, indent=1))
    print(json.dumps({"segments": len(jobs), "classes": n_cls, "train_counts": counts.tolist(),
                      "nearest_proto_acc_val": diag["validation"]["nearest_prototype_accuracy"],
                      "nearest_proto_acc_train": diag["train"]["nearest_prototype_accuracy"],
                      "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
