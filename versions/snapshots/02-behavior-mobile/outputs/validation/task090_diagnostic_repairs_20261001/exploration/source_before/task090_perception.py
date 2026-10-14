"""Task90 TRAIN-only appearance proposals and RGB-D observed surface samples.

This is NOT a center/grasp estimator or an admitted relation/hand tracker. Runtime
uses the full head image, never annotation ROIs. Missing aligned depth/pose gives
no metric detections. Task1 geometry is reused, not its copper-bin parameters.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np

from harness.object_perception import Camera, camera_from_pose, head_pose, obstacle_points

DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "configs/task090_perception.json"
UNSUPPORTED = ("furniture_sink", "floors", "countertop", "object_hand_tracks", "relations")


def capability_status():
    """Explicit limits; absence of a proposal must not be treated as observed absence."""
    return {"appearance_labels": ["half", "half_pomegranate", "trash_can"],
            "unsupported": list(UNSUPPORTED), "visibility": "unknown",
            "point_semantics": "observed_surface_sample", "metric_calibration_admitted": False,
            "grasp_reference_valid": False, "relations_admitted": False}


@lru_cache(maxsize=4)
def load_calibration(path=str(DEFAULT_CALIBRATION)):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("task_id") != 90 or cfg.get("split") != "train":
        raise ValueError("Task90 appearance calibration must be TRAIN-only")
    if cfg.get("annotation_author") != "assistant; not a human annotator":
        raise ValueError("Explicit assistant annotation provenance required")
    return cfg


def _rgb(rgb):
    if rgb is None:
        return None
    a = np.asarray(rgb)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.ndim != 3 or a.shape[-1] not in (3, 4) or min(a.shape[:2]) < 8:
        return None
    a = a[..., :3]
    if not np.isfinite(a).all():
        return None
    if a.dtype != np.uint8:
        a = np.clip(a * (255 if a.max() <= 1 else 1), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def appearance_candidates(rgb, calibration_path=str(DEFAULT_CALIBRATION)):
    """Unverified 2D appearance proposals; no object presence/absence certification."""
    image = _rgb(rgb)
    if image is None:
        return []
    cfg = load_calibration(str(calibration_path))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    scale = (image.shape[1] / 720.0) ** 2
    out = []
    for label, gate in cfg["classes"].items():
        mask = np.zeros(image.shape[:2], np.uint8)
        for lower, upper in gate["hsv_boxes"]:
            mask |= cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8))
        n, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in np.argsort(-stats[:, cv2.CC_STAT_AREA]):
            if i == 0:
                continue
            x, y, w, h, area = map(int, stats[i])
            if not gate["area_px_720"][0] * scale <= area <= gate["area_px_720"][1] * scale:
                continue
            if not gate["aspect_h_over_w"][0] <= h / w <= gate["aspect_h_over_w"][1]:
                continue
            if area / (w * h) < gate["fill_min"]:
                continue
            yy, xx = np.nonzero(components[y:y+h, x:x+w] == i)
            out.append({"label": label, "physical_category": gate["physical_category"],
                        "bbox_xyxy": [x, y, x+w, y+h], "pixels": int(area),
                        "pixel_x": xx+x, "pixel_y": yy+y})
            if sum(d["label"] == label for d in out) >= cfg["max_candidates_per_class"]:
                break
    return out


def geometry_sanity(points, prior):
    """Reject impossible size / flat horizontal texture proposals; not object identity proof."""
    if prior is None:
        return True
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3 or not np.isfinite(pts).all():
        return False
    span = np.percentile(pts, 98, axis=0) - np.percentile(pts, 2, axis=0)
    if np.max(span) > prior["max_axis_span_m"] or span[2] < prior["min_vertical_span_m"]:
        return False
    horizontal = prior.get("max_horizontal_axis_span_m")
    return horizontal is None or bool(np.max(span[:2]) <= horizontal)


def detect_from_observation(rgb, depth, cam_rel_pose, state, *, calibration_path=str(DEFAULT_CALIBRATION)):
    """Adapter API -> (surface detections, None). None means visibility is unknown.

    xyz_base is one actual observed masked pixel backprojection nearest the median
    of valid surface samples, NOT a median 3D center or a TRAIN EEF reference.
    No plane intersection, metric offsets, fixed ROI, or temporal identity fallback.
    """
    image = _rgb(rgb)
    if image is None or depth is None or cam_rel_pose is None:
        return [], None
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    elif d.ndim == 3 and d.shape[0] == 1:
        d = d[0]
    if d.shape != image.shape[:2] or image.shape[0] != image.shape[1]:
        return [], None  # uncalibrated depth alignment/intrinsics
    try:
        extrinsics = camera_from_pose(head_pose(cam_rel_pose))
    except (TypeError, ValueError):
        return [], None
    if extrinsics is None:
        return [], None
    cam = Camera(*extrinsics, image.shape[1], image.shape[0])
    cfg = load_calibration(str(calibration_path))
    detections = []
    for candidate in appearance_candidates(image, str(calibration_path)):
        u, v = candidate.pop("pixel_x"), candidate.pop("pixel_y")
        z = d[v, u]
        valid = np.isfinite(z) & (z > 0.05) & (z < cfg["depth_max_m"])
        if valid.sum() < cfg["min_depth_pixels"] or valid.mean() < cfg["min_depth_fraction"]:
            continue
        u, v, z = u[valid], v[valid], z[valid]
        pts = cam.backproject(u, v, z)
        prior = cfg["classes"][candidate["label"]].get("geometry_prior")
        if not geometry_sanity(pts, prior):
            continue
        median = np.median(pts, axis=0)
        k = int(np.argmin(np.sum((pts-median)**2, axis=1)))
        detections.append({**candidate, "xyz_base": pts[k].tolist(),
                           "surface_pixel_uv": [int(u[k]), int(v[k])], "surface_depth_m": float(z[k]),
                           "source": "rgbd", "point_semantics": "observed_surface_sample",
                           "geometry_sanity": "static_model_v2" if prior is not None else "not_applied",
                           "center_valid": False, "grasp_reference_valid": False,
                           "metric_calibration_admitted": False, "relations_admitted": False,
                           "identity_status": "appearance_candidate", "held_status": "unknown",
                           "calibration_id": cfg["calibration_id"]})
    return detections, None
