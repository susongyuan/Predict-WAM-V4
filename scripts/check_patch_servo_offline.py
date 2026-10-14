#!/usr/bin/env python3
"""Offline feasibility of a GENERIC prototype-referenced patch visual servo for navigation docking.

For a head frame taken before a manipulation segment starts, search the shift (dx columns, dy rows) and scale s of
the current 14x14 DINOv3 patch grid that best matches the key's ARRIVAL prototype (precond_patch_prototypes.npz,
built from train start frames).  Hypothesis: dx* predicts the yaw / lateral correction the demonstrator still
performed before arriving, s*<1 predicts remaining forward travel.  Ground truth for the check is the demo's own
cumulative base action (0:3 = vx, vy, vyaw, local frame) from the frame to the segment start.  No simulator, no
privileged state; demo actions are only used as the offline label.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, SkillTable  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402
from harness.vtt_monitor import precondition_key  # noqa: E402
from validate_dock_patch_offline import Encoder, align, load_png  # noqa: E402

FPS = 30
SCALES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25)


def servo_search(patch, proto, torch, device, dxs=range(-5, 6), dys=range(-3, 4), scales=SCALES):
    """patch (P,D) f16, proto (P,D) unit. Returns dict(dx, dy, s, best, base) over the overlap-mean cosine."""
    g = int(round(len(patch) ** 0.5))
    x = torch.from_numpy(np.asarray(patch, np.float32)).to(device).reshape(g, g, -1).permute(2, 0, 1)[None]
    q = torch.from_numpy(np.asarray(proto, np.float32)).to(device).reshape(g, g, -1)
    best = (-9.0, 0, 0, 1.0)
    base = None
    for s in scales:
        if s == 1.0:
            xs = x
        else:  # zoom about the centre: s>1 enlarges content (as if closer)
            n = max(2, int(round(g * s)))
            z = torch.nn.functional.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
            if n >= g:
                o = (n - g) // 2
                xs = z[:, :, o:o + g, o:o + g]
            else:
                xs = torch.full_like(x, float("nan"))
                o = (g - n) // 2
                xs[:, :, o:o + n, o:o + n] = z
        u = xs[0].permute(1, 2, 0)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
        for dy in dys:
            for dx in dxs:
                # current content shifted by (dy,dx) compared with prototype position
                ys = slice(max(0, dy), g + min(0, dy)); yq = slice(max(0, -dy), g + min(0, -dy))
                xs_ = slice(max(0, dx), g + min(0, dx)); xq = slice(max(0, -dx), g + min(0, -dx))
                c = (u[ys, xs_] * q[yq, xq]).sum(-1)
                c = c[torch.isfinite(c)]
                if c.numel() < 0.4 * g * g:
                    continue
                v = float(c.mean())
                if s == 1.0 and dx == 0 and dy == 0:
                    base = v
                if v > best[0]:
                    best = (v, dx, dy, s)
    return {"best": best[0], "dx": best[1], "dy": best[2], "s": best[3], "base": base}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/validation/patch_servo_offline_20260923")
    ap.add_argument("--dino", default="models/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--pre-seconds", default="1,3,6,10")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--eps-yaw", type=float, default=0.5, help="|sum vyaw action| below this = no decisive turn")
    ap.add_argument("--eps-lat", type=float, default=0.5)
    ap.add_argument("--pngs", nargs="*", default=[
        f"outputs/behavior/closed_loop/s1_dev10_stage3_postgate_motionstall_hold_s8_20260923/observations/ep2_step{s:06d}.png"
        for s in (1125, 1500, 1800, 2100, 2400)])
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.35)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    pz = np.load(ROOT / "data/skill_vtt_task013/precond_patch_prototypes.npz")
    keys = [str(k) for k in pz["keys"]]
    protos = {k: pz["proto"][i] for i, k in enumerate(keys)}
    thr = {k: float(pz["thr"][i]) for i, k in enumerate(keys)}
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    pre = [float(x) for x in args.pre_seconds.split(",")]

    jobs, n_ep = [], 0
    for ep in sorted(table.episodes):
        segs = table.episodes[ep]["segments"]
        try:
            pack = None
            for i in range(1, len(segs)):
                if segs[i - 1]["skill"] != "move to" or segs[i]["skill"] == "move to":
                    continue
                key = precondition_key(segs[i]["skill"], segs[i].get("objects", []))
                if key not in protos:
                    continue
                if pack is None:
                    pack = FramePack(FRAMES / HEAD_CAMERA, ep)
                lo, st = segs[i]["start"], segs[i - 1]["start"]
                for sec in pre:
                    a = align(lo - int(sec * FPS), pack.stride, pack.length)
                    if a >= st:
                        jobs.append({"episode": ep, "split": index["episodes"][str(ep)]["split"], "key": key,
                                     "frame": a, "seg_start": lo, "pre_s": sec})
        except Exception as e:  # noqa: BLE001
            print(f"[skip] ep{ep}: {e}")
            continue
        n_ep += 1
        if args.max_episodes and n_ep >= args.max_episodes:
            break
    print(f"[jobs] {len(jobs)} demo frames from {n_ep} episodes", flush=True)

    enc = Encoder(args.dino, device, torch.bfloat16 if device == "cuda" else torch.float32, -4, 16)
    from harness.wam_behavior import normalize_image
    rows, pack_ep, pack = [], None, None
    buf, meta = [], []

    def flush():
        if not buf:
            return
        _, last, _ = enc(buf)
        for m, p in zip(meta, last):
            r = servo_search(p, protos[m["key"]], torch, device)
            rows.append({**m, **r})
        buf.clear(); meta.clear()

    for j in jobs:
        if j["episode"] != pack_ep:
            pack_ep, pack = j["episode"], FramePack(FRAMES / HEAD_CAMERA, j["episode"])
        act = np.load(LOWDIM / f"episode_{j['episode']}_action.npy", mmap_mode="r")
        rem = np.asarray(act[j["frame"]:j["seg_start"], :3], np.float64).sum(0) / FPS
        buf.append(normalize_image(pack.rgb(j["frame"])))
        meta.append({**j, "rem_vx": float(rem[0]), "rem_vy": float(rem[1]), "rem_vyaw": float(rem[2])})
        if len(buf) >= 16:
            flush()
    flush()
    cl = []
    for f in args.pngs:
        rgb = load_png(ROOT / f)
        if rgb is None:
            continue
        _, last, _ = enc([normalize_image(rgb)])
        cl.append({"file": f, **servo_search(last[0], protos["open lid(car)"], torch, device)})

    # sign agreement: image dx*>0 means content sits right of where the prototype has it -> robot must turn right
    # (negative yaw in a right-handed z-up base frame) and/or move right (negative vy).  The sign convention is
    # fitted on TRAIN and evaluated on VALIDATION to avoid assuming it.
    def agree(sub, field, eps):
        s = [r for r in sub if abs(r[field]) >= eps and r["dx"] != 0]
        if not s:
            return None, 0
        return float(np.mean([np.sign(r["dx"]) == np.sign(r[field]) for r in s])), len(s)

    rep = {"runtime_s": round(time.time() - t0, 1), "n_frames": len(rows), "closed_loop": cl, "groups": {}}
    for split in ("train", "validation"):
        for key in sorted({r["key"] for r in rows}) + ["ALL"]:
            sub = [r for r in rows if r["split"] == split and (key == "ALL" or r["key"] == key)]
            if not sub:
                continue
            g = {"n": len(sub),
                 "frac_dx_nonzero": float(np.mean([r["dx"] != 0 for r in sub])),
                 "mean_gain_best_minus_base": float(np.mean([r["best"] - r["base"] for r in sub])),
                 "frac_scale_lt1": float(np.mean([r["s"] < 1 for r in sub])),
                 "corr_dx_vyaw": float(np.corrcoef([r["dx"] for r in sub], [r["rem_vyaw"] for r in sub])[0, 1]) if len(sub) > 2 else None,
                 "corr_dx_vy": float(np.corrcoef([r["dx"] for r in sub], [r["rem_vy"] for r in sub])[0, 1]) if len(sub) > 2 else None,
                 "corr_s_vx": float(np.corrcoef([r["s"] for r in sub], [r["rem_vx"] for r in sub])[0, 1]) if len(sub) > 2 else None}
            for fld, eps in (("rem_vyaw", args.eps_yaw), ("rem_vy", args.eps_lat)):
                a, n = agree(sub, fld, eps)
                g[f"sign_agree_dx_{fld}"] = a; g[f"n_decisive_{fld}"] = n
            for sec in pre:
                ss = [r for r in sub if r["pre_s"] == sec]
                if ss:
                    g[f"pre{int(sec)}s"] = {"n": len(ss), "corr_dx_vyaw": float(np.corrcoef([r["dx"] for r in ss], [r["rem_vyaw"] for r in ss])[0, 1]) if len(ss) > 2 and np.std([r["dx"] for r in ss]) > 0 else None,
                                           "median_s": float(np.median([r["s"] for r in ss])), "frac_above_thr_base": float(np.mean([r["base"] >= thr[r["key"]] for r in ss]))}
            rep["groups"][f"{split}:{key}"] = g
    (args.out / "report.json").write_text(json.dumps(rep, indent=1))
    with open(args.out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "groups"}, indent=1))
    for k, g in rep["groups"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in g.items() if not a.startswith("pre")})


if __name__ == "__main__":
    main()
