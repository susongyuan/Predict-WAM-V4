"""Monocular head-camera docking feature — shared by the offline validator and the live policy server.

Offline validation (scripts/validate_dock_appearance_offline.py, outputs/validation/dock_target_20260923/) showed this
feature separates the demonstrated trunk-facing dock from the recorded oblique mis-dock at AUC 0.891, whereas the
DINOv3 CLS prototype-cosine arrival gate cannot (train 0.954 vs mis-dock 0.87-0.93 overlap).  It reads ONLY the head
RGB the policy already receives (no depth, no simulator ground truth), so it is legal as a live arrival sub-gate.

The robot squarely behind the car's rear trunk sees a large, horizontally-centred, bilaterally-symmetric car region;
an oblique / side / front mis-dock sees the car off to one side and asymmetric.
NOTE: the red HSV segmentation is calibrated to the current (red-car) instances; a colour-agnostic detector is needed
before cross-instance generalisation (dev11-19).  dock_score = lr_sym - |centroid_u| (higher = better dock).
"""
import numpy as np
import cv2

MIN_AREA_FRAC = 0.12       # reject tiny red slivers (reflections / partial glimpses)
DOCK_SCORE_ARRIVED = 0.30  # offline: demo docks ~0.55, mis-docks ~-0.05; 0.30 gives 83% pass / 86% reject


def car_dock_feature(rgb):
    """rgb: HxWx3 uint8 RGB (any size). Returns dict {area_frac, centroid_u, lr_sym, ok, score}.

    centroid_u in [-1,1] (0 = centred), lr_sym in [0,1] (1 = symmetric), score = lr_sym - |centroid_u|.
    ok is False (score NaN) when no car region of at least MIN_AREA_FRAC is found.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 90, 60), (180, 255, 255))
    mask = cv2.morphologyEx((m1 | m2), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return {"area_frac": 0.0, "centroid_u": float("nan"), "lr_sym": float("nan"), "ok": False, "score": float("nan")}
    big = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == big, 255, 0).astype(np.uint8)
    area_frac = float((mask > 0).sum()) / float(h * w)
    if area_frac < MIN_AREA_FRAC:
        return {"area_frac": area_frac, "centroid_u": float("nan"), "lr_sym": float("nan"), "ok": False, "score": float("nan")}
    col = mask.sum(0).astype(np.float64)
    cx = float((np.arange(w) * col).sum() / col.sum())
    centroid_u = (cx - (w - 1) / 2.0) / ((w - 1) / 2.0)
    mir = mask[:, ::-1]
    inter = int(((mask > 0) & (mir > 0)).sum())
    union = int(((mask > 0) | (mir > 0)).sum())
    lr_sym = inter / union if union else 0.0
    return {"area_frac": area_frac, "centroid_u": float(centroid_u), "lr_sym": float(lr_sym),
            "ok": True, "score": float(lr_sym - abs(centroid_u))}
