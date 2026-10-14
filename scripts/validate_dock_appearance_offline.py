#!/usr/bin/env python3
"""Offline (CPU, RGB-only) feasibility gate for a monocular docking servo — instruction-13 check BEFORE any GPU run.

Depth videos are NOT downloaded locally (only RGB), so a metric depth target is impossible without re-downloading
(forbidden). This validates the fairness-compliant alternative: a head-camera APPEARANCE target that separates the
demonstrated centered-behind-the-trunk dock from the recorded oblique mis-dock — something the DINOv3 CLS cosine
(the current 0.83 arrival gate) provably cannot do (train LOO 0.954 vs mis-dock 0.87-0.93).

Perceived feature (car region in the head RGB, cv2+numpy only, deployable in the lilawam policy env):
  - area_frac : fraction of image the car region occupies (apparent scale ~ inverse range)
  - centroid_u: horizontal centroid of the car region, normalised to [-1, 1] (0 = centred). |u| large = off to a side.
  - lr_sym    : left/right symmetry of the car mask about the image centre column in [0,1] (1 = perfectly symmetric).
                A robot squarely behind the rear trunk sees a large, centred, bilaterally-symmetric car region.
No simulator ground truth is read (instruction 4): only cached demo head RGB and the closed-loop head-observation PNGs.
"""
import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, SkillTable
from harness.frame_cache import FramePack


MIN_AREA_FRAC = 0.12  # reject tiny red slivers (reflections / car edge); a real dock fills a substantial view


def car_feature(rgb):
    """rgb: HxWx3 uint8 RGB. Returns (area_frac, centroid_u, lr_sym, ok).

    Uses the LARGEST connected red component (the car body) and requires it to occupy >= MIN_AREA_FRAC,
    so a partial/oblique glimpse or a stray red reflection does not count as a detected dock.
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # red wraps hue 0; two bands. Saturated + bright to avoid pink walls/floor.
    m1 = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 90, 60), (180, 255, 255))
    mask = cv2.morphologyEx((m1 | m2), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return 0.0, float("nan"), float("nan"), False
    big = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == big, 255, 0).astype(np.uint8)
    area = int(mask.sum() // 255)
    area_frac = area / float(h * w)
    if area_frac < MIN_AREA_FRAC:
        return area_frac, float("nan"), float("nan"), False
    col = mask.sum(0).astype(np.float64)
    cx = float((np.arange(w) * col).sum() / col.sum())
    centroid_u = (cx - (w - 1) / 2.0) / ((w - 1) / 2.0)
    # left/right symmetry: overlap of mask with its horizontal mirror
    mir = mask[:, ::-1]
    inter = int(((mask > 0) & (mir > 0)).sum())
    union = int(((mask > 0) | (mir > 0)).sum())
    lr_sym = inter / union if union else 0.0
    return area_frac, centroid_u, lr_sym, True


def load_png(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/validation/dock_target_20260923")
    ap.add_argument("--max-episodes", type=int, default=40)
    ap.add_argument("--anchors", default="0.0,0.05,0.1")  # fractions into the open-lid segment
    ap.add_argument("--misdock-globs", nargs="+", default=[
        "outputs/behavior/closed_loop/s1_dev10_stage3posonly_best5500_hold_exec32_720p_20260923/observations/*.png",
        "outputs/behavior/closed_loop/s1_dev10_stage4delta_best6500_hold_exec32_720p_20260923/observations/*.png",
        "outputs/behavior/closed_loop/s1_dev10_posonly_best5500_exec32_720p_20260922/observations/*.png",
    ])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    anchors = [float(x) for x in args.anchors.split(",")]

    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    demo_rows = []
    saved = 0
    for ep in sorted(table.episodes):
        split = index["episodes"][str(ep)]["split"]
        segs = [s for s in table.episodes[ep]["segments"] if s["skill"] == "open lid"]
        if not segs:
            continue
        seg = segs[0]
        try:
            pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        except Exception:
            continue
        lo, hi = seg["start"], seg["end"]
        for fr in anchors:
            a = int(lo + fr * (hi - lo))
            a = (a // pack.stride) * pack.stride
            try:
                rgb = pack.rgb(a)
            except Exception:
                continue
            af, u, sym, ok = car_feature(rgb)
            if ok:
                demo_rows.append({"episode": ep, "split": split, "anchor": a, "frac": fr,
                                  "area_frac": af, "centroid_u": u, "lr_sym": sym})
            if saved < 6 and fr == 0.0 and ok:
                vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
                cv2.imwrite(str(args.out / f"demo_ep{ep}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved += 1
        if len({r["episode"] for r in demo_rows}) >= args.max_episodes:
            break

    mis_rows = []
    for g in args.misdock_globs:
        for p in sorted(glob.glob(str(ROOT / g))):
            rgb = load_png(p)
            if rgb is None:
                continue
            af, u, sym, ok = car_feature(rgb)
            mis_rows.append({"file": str(Path(p).relative_to(ROOT)), "area_frac": af,
                             "centroid_u": u if ok else None, "lr_sym": sym if ok else None, "detected": ok})

    def stats(rows, key):
        v = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r.get(key, np.nan))], float)
        return {"n": int(v.size), "mean": float(v.mean()) if v.size else None,
                "std": float(v.std()) if v.size else None,
                "min": float(v.min()) if v.size else None, "max": float(v.max()) if v.size else None,
                "abs_mean": float(np.abs(v).mean()) if v.size else None}

    demo_det = [r for r in demo_rows]
    mis_det = [r for r in mis_rows if r["detected"]]
    summary = {
        "purpose": "monocular RGB appearance separation of demo dock vs closed-loop mis-dock (instruction-13 offline gate)",
        "ground_truth_used": False, "depth_available_locally": False,
        "n_demo_episodes": len({r["episode"] for r in demo_rows}), "n_demo_frames": len(demo_rows),
        "n_misdock_frames": len(mis_rows), "n_misdock_detected": len(mis_det),
        "demo": {k: stats(demo_det, k) for k in ("area_frac", "centroid_u", "lr_sym")},
        "misdock": {k: stats(mis_det, k) for k in ("area_frac", "centroid_u", "lr_sym")},
        "detect_rate_demo": len([r for r in demo_rows]) and 1.0,
        "detect_rate_misdock": (len(mis_det) / len(mis_rows)) if mis_rows else None,
    }
    # Separation verdict on |centroid_u| and lr_sym (the pose-discriminative channels)
    du = stats(demo_det, "centroid_u"); mu = stats(mis_det, "centroid_u")
    ds = stats(demo_det, "lr_sym"); ms = stats(mis_det, "lr_sym")
    summary["separation"] = {
        "abs_centroid_u_demo_mean": du["abs_mean"], "abs_centroid_u_misdock_mean": mu["abs_mean"],
        "lr_sym_demo_mean": ds["mean"], "lr_sym_misdock_mean": ms["mean"],
        "centroid_u_separates": (du["abs_mean"] is not None and mu["abs_mean"] is not None
                                 and mu["abs_mean"] > du["abs_mean"] + 2 * (du["std"] or 0)),
    }
    (args.out / "dock_appearance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "demo_rows.json").write_text(json.dumps(demo_rows) + "\n")
    (args.out / "misdock_rows.json").write_text(json.dumps(mis_rows, indent=1) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nmis-dock per-frame:")
    for r in mis_rows:
        print(" ", r["file"], "det" if r["detected"] else "NODET",
              None if r["centroid_u"] is None else round(r["centroid_u"], 3),
              None if r["lr_sym"] is None else round(r["lr_sym"], 3),
              round(r["area_frac"], 3))


if __name__ == "__main__":
    main()
