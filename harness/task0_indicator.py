"""Onboard press-outcome indicator for Task 0 (turning_on_radio): the ToggledOn marker seen by the LEFT WRIST camera.

OmniGibson's ToggledOn state (object_states/toggle.py) renders a small sphere at the radio's toggle button (the centre of
the black speaker grille, visible to the left wrist camera during a press): RED while off, GREEN once toggled on
(visual_marker.visible = True, colour set in _set_value). It is part of the rendered RGB observation the policy is allowed
to use; on a real radio it stands for an indicator light. It is a SIMULATOR-RENDERED indicator: record it as such.

classify(rgb) -> (verdict, info), verdict in {"on", "off", "unknown"}:
  "on"      a green blob (sphere-sized) inside a mostly dark ring (the grille) is visible;
  "off"     no green, and a red sphere-sized blob inside a mostly dark ring is visible (the red radio body never has a
            dark ring around it);
  "unknown" neither (marker occluded by the fist, out of view, image missing).
Thresholds come from TRAIN demo frames only (scripts/fit_task0_indicator.py -> outputs/validation/task0_indicator_20260928/
fit.json); VALIDATION frames are only reported. Pure numpy + OpenCV; CPU self-test: `python3 -m harness.task0_indicator`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

LEFT_WRIST_SUFFIX = ":left_realsense_link:Camera:0::rgb"
SOURCE = "left_wrist_toggle_marker:v1 (simulator-rendered ToggledOn marker; indicator-light proxy)"


@dataclass(frozen=True)
class IndicatorConfig:
    size: int = 224                 # images are resized to this (the WAM wrist input / TRAIN frame-cache size)
    green_h: tuple = (45, 80)       # OpenCV hue 0..180
    green_s: int = 150
    green_v: int = 60
    red_h: tuple = (8, 170)         # red = hue <= red_h[0] or >= red_h[1]
    red_s: int = 150
    red_v: int = 60
    min_area: int = 12              # px at 224: the marker sphere is ~15-600 px, the radio body is >> max_area
    max_area: int = 900
    ring_px: int = 3
    dark_v: int = 70                # grille / gripper pixels
    ring_dark_frac_green: float = 0.35
    ring_dark_frac_red: float = 0.6


DEFAULT = IndicatorConfig()


def _prep(rgb, size):
    img = np.asarray(rgb)[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(np.ascontiguousarray(img), (size, size), interpolation=cv2.INTER_AREA)
    return img


def _blobs(mask, dark, cfg, frac):
    """Sphere-sized components of `mask` whose surrounding ring is mostly dark: [(area, ring_dark_frac, cx, cy)]."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    k = np.ones((2 * cfg.ring_px + 1, 2 * cfg.ring_px + 1), np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not cfg.min_area <= area <= cfg.max_area:
            continue
        x, y, w, h = (int(v) for v in stats[i, :4])
        x0, y0 = max(0, x - cfg.ring_px - 1), max(0, y - cfg.ring_px - 1)
        x1, y1 = min(mask.shape[1], x + w + cfg.ring_px + 1), min(mask.shape[0], y + h + cfg.ring_px + 1)
        comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(comp, k) > 0) & (comp == 0)
        if not ring.any():
            continue
        f = float(dark[y0:y1, x0:x1][ring].mean())
        if f >= frac:
            out.append((area, round(f, 3), float(cent[i][0]), float(cent[i][1])))
    return out


def classify(rgb, cfg: IndicatorConfig = DEFAULT, **overrides):
    if overrides:
        cfg = replace(cfg, **overrides)
    if rgb is None:
        return "unknown", {"reason": "no_image"}
    img = _prep(rgb, cfg.size)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dark = V < cfg.dark_v
    green = (H >= cfg.green_h[0]) & (H <= cfg.green_h[1]) & (S >= cfg.green_s) & (V >= cfg.green_v)
    red = ((H <= cfg.red_h[0]) | (H >= cfg.red_h[1])) & (S >= cfg.red_s) & (V >= cfg.red_v)
    g = _blobs(green, dark, cfg, cfg.ring_dark_frac_green)
    if g:
        best = max(g)
        return "on", {"green_area": best[0], "ring_dark": best[1], "uv": [round(best[2], 1), round(best[3], 1)]}
    r = _blobs(red, dark, cfg, cfg.ring_dark_frac_red)
    if r:
        best = max(r)
        return "off", {"red_area": best[0], "ring_dark": best[1], "uv": [round(best[2], 1), round(best[3], 1)]}
    return "unknown", {"green_px": int(green.sum())}


def find_left_wrist(obs):
    for k in obs:
        if isinstance(k, str) and k.endswith(LEFT_WRIST_SUFFIX):
            return obs[k]
    return None


def _synthetic(marker):
    """224 image: red radio face, black grille disc, a marker sphere of the given colour (or none) at its centre."""
    img = np.full((224, 224, 3), (200, 20, 15), np.uint8)             # red body
    cv2.rectangle(img, (0, 170), (224, 224), (170, 190, 215), -1)      # light-blue sofa
    cv2.circle(img, (112, 100), 60, (235, 235, 235), -1)              # white ring
    cv2.circle(img, (112, 100), 48, (20, 20, 20), -1)                 # black grille
    if marker is not None:
        cv2.circle(img, (112, 100), 7, marker, -1)
    return img


def _self_test():
    assert classify(_synthetic((30, 210, 40)))[0] == "on"
    assert classify(_synthetic((215, 25, 20)))[0] == "off"
    assert classify(_synthetic(None))[0] == "unknown"                 # marker occluded: never 'off' from the body
    plain = np.full((224, 224, 3), (200, 20, 15), np.uint8)
    assert classify(plain)[0] == "unknown" and classify(None)[0] == "unknown"
    big = _synthetic(None)
    cv2.circle(big, (112, 100), 40, (30, 210, 40), -1)                # a large green object is not the marker
    assert classify(big)[0] == "unknown", classify(big)
    rgba = np.concatenate([_synthetic((30, 210, 40)), np.full((224, 224, 1), 255, np.uint8)], -1)
    assert classify(cv2.resize(rgba, (480, 480), interpolation=cv2.INTER_NEAREST))[0] == "on"   # RGBA, other size
    print("PASS task0_indicator self-test (synthetic images; accuracy on demo frames: outputs/validation/task0_indicator_20260928)")


if __name__ == "__main__":
    _self_test()
