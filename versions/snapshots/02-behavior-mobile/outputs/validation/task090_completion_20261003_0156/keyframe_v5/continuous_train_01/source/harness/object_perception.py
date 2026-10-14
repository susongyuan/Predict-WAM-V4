"""Onboard Task 1 object detector: head RGB(-D) + head camera pose -> base-frame object positions (docking inputs).

Called by harness/task1_perception_adapter.py; the output feeds harness/docking_controller.ObjectMap.update.

Inputs are policy-visible only:
  * head camera (zed_link) RGB, native 720x720 (HxWx3 or HxWx4 uint8; other square sizes are rescaled intrinsics),
  * head depth_linear (z-depth in metres; optional - without it a ground-plane / rim-plane geometry fallback is used),
  * `cam_rel_poses` from proprio (flat 21 = [left_wrist, right_wrist, head] x [x, y, z, qx, qy, qz, qw] in the robot base
    frame, USD camera axes: -Z forward, +Y up; a single 7-vector or a (3, 7) array is also accepted),
  * the 61-D proprio vector (EEF positions and gripper finger widths, to reject held / in-bin objects).

Method (no learning; thresholds calibrated on the tuning frames listed in scripts/eval_object_perception.py):
  * soda cans: full-resolution HSV colour blobs (orange H 3-18 S>=190; dark blue H 104-128 S>=120), accepted only if
    their metric size (pixel extent x depth / f) is can-sized and their 3-D points are at floor height. Without depth
    the blob centroid ray is intersected with the plane z = can centre height. Far cans are only a few pixels wide,
    hence full resolution for the colour masks.
  * trash can: copper interior blob (H 5-19, saturated, darker than the wood floor). The visible interior is bounded
    by the rim circle, so the filled blob hull back-projected onto the rim plane (z = 0.279 m on the floor) is a disk
    of the rim radius whose mean is the bin centre (needs the camera pose only). Depth is used to reject floor/furniture
    look-alikes (interior points must lie inside that disk and below the rim) and lifted (carried) bins.
  * blobs that lie inside a bin's image hull (cans in the bin, bin interior speckle of can colour) and blobs next to an
    EEF whose gripper is closed (held can / carried bin) are rejected: they are no navigation targets.

Point convention (what docking_controller's calibration expects, see load_calibration and
outputs/validation/task1_docking_calibration_20260926/grasp_relative_poses_train.json):
  * can_of_soda: xyz_base = can centre (the left-hand grasp point ~ can centre),
  * trash_can:   xyz_base = RIGHT-HAND RIM GRASP POINT = bin centre - RIM_OFFSET (base frame, the median offset measured
    on train demos in scripts/build_task1_object_gt.py), z = rim height. The bin is rotationally symmetric, so the
    offset is applied in the current base frame; at the docking pose this is exactly the calibrated grasp point.
    The bin centre is returned as well under "center_base" (ObjectMap ignores extra keys).

Also: `obstacle_points` (base-frame xy of furniture-height depth points, own arms/held objects removed) for the docking
controller's corridor check. numpy + cv2 only; ~10-20 ms per call on one CPU core at 720x720.
CPU self-test on synthetic data: `python3 -m harness.object_perception`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

# ---- geometry constants -------------------------------------------------------------------------------------------
HEAD_F_720 = 306.0                     # head intrinsics at 720x720 (official R1Pro eval utils; cx = cy = W / 2)
CV_FROM_USD = np.diag([1.0, -1.0, -1.0])
CAN_RADIUS, CAN_CENTRE_Z = 0.037, 0.066          # itolcg / lugwcz assets (height 0.127-0.135 m)
BIN_RADIUS, BIN_RIM_Z = 0.115, 0.279             # wkxtxh asset: outer radius, rim height when standing on the floor
RIM_OFFSET = np.array([0.016, 0.088])  # median (bin centre - right-hand rim grasp point) in the base frame, train demos
EEF_POS = {"left": slice(17, 20), "right": slice(42, 45)}          # proprio slices (base frame)
GRIPPER = {"left": slice(24, 26), "right": slice(49, 51)}          # finger positions; sum ~0.10 open, ~0.016 on the bin


@dataclass
class DetectorConfig:
    # colour gates (OpenCV HSV, H in [0, 180))
    orange_h: tuple = (3, 18)
    orange_s_min: int = 190
    orange_v_min: int = 110            # bin interior pixels this saturated are dark (V < ~100): excluded
    blue_h: tuple = (104, 128)
    blue_s_min: int = 120
    blue_v: tuple = (18, 200)
    orange_blob_sv: tuple = (205, 130) # blob mean S, V minimum (orange cans: S >= 218, V >= 156 on replay tuning
    #                                    frames, 5th pct S 214 / V 126 in the HEVC demo videos; door-frame wood 193/121)
    blue_blob_sv: tuple = (125, 0)     # (blue cans: mean S >= 162 replay, 5th pct 131 demo videos)
    blue_blob_v_max: float = 120.0     # dark-blue cans: blob mean V 50-61 replay, median 71 demos; shaded blue sofa is
    #                                    brighter
    copper_h: tuple = (5, 19)
    copper_s_min: int = 90
    copper_s_minus_v: int = -40        # S >= V - 40: interior is more saturated / darker than the sunlit wood floor
    copper_v_min: int = 40
    # can blob acceptance
    can_min_px: int = 4                # at 720x720 (scaled by (W/720)^2)
    can_width: tuple = (0.035, 0.20)   # m, apparent width (standing 0.066-0.075; lying on its side up to ~0.17)
    can_height: tuple = (0.022, 0.24)  # m, apparent height (side + top ellipse)
    can_top_z_max: float = 0.22        # m above the floor
    can_mid_z: tuple = (-0.03, 0.15)
    can_aspect: tuple = (0.6, 3.2)     # apparent height / width (standing can: ~1.2-2.6 from any view)
    can_continue_frac: float = 0.6     # max fraction of the pixel band above the blob at the blob's own depth
    # bin acceptance
    bin_min_px: int = 40               # copper hull pixels at 720x720
    bin_radius: tuple = (0.05, 0.19)   # m, disk radius of the hull back-projected onto the rim plane (2 sigma)
    bin_minor_min: float = 0.04        # m (thin wooden chair rails / frames are narrower)
    bin_solidity_min: float = 0.45     # copper pixels / filled hull pixels (bin: 0.54-0.84)
    bin_depth_disk: tuple = (0.06, 0.15)  # m, median horizontal distance of interior depth points from the fitted
    #                                       centre (inner wall / bottom of a real bin: ~0.10; flat look-alikes ~0.04)
    bin_depth_disk80_max: float = 0.20 # m, 80th percentile of the same
    bin_z90: tuple = (0.12, 0.32)      # m, 90th percentile height of interior depth points (standing bin ~0.22-0.26;
    #                                    floor patch ~0, carried bin / raised hearth > 0.34)
    # geometric bin detector (depth)
    bin_geo_zmax: float = 0.34         # m, "low object" band 0.04..zmax above the floor
    bin_geo_range: float = 6.0         # m
    bin_geo_cell: float = 0.02         # m, top-down grid resolution
    bin_geo_min_pts: int = 20          # low points (half-res pixels) in the blob (a bin at 5 m gives ~50)
    bin_geo_top: tuple = (0.24, 0.31)  # m, 98th pct height of the blob (rim 0.279; tuning frames 0.24-0.29)
    bin_geo_tall_cells: int = 4        # max grid cells with points 0.36-1.2 m over the (dilated) footprint
    bin_geo_radius: tuple = (0.08, 0.13)   # m, fitted circle radius (asset: 0.115 outer)
    bin_geo_resid: float = 0.02        # m, rms circle residual
    # held / in-bin rejection
    held_radius: float = 0.22          # m, horizontal distance to an EEF whose gripper is closed
    held_ray_dist: float = 0.30        # m, no depth: EEF (closed) this close to the camera->blob ray segment
    #                                    (held can, cans inside the carried bin)
    max_range: float = 6.0             # m, detections beyond this horizontal range are dropped
    bin_without_depth: bool = False    # copper-colour bin detector without depth: many false positives on wood
    #                                    (floor, furniture) in the demo videos -> off by default
    gripper_closed_sum: float = 0.085  # m, finger sum below this = closed (open 0.10; on a can ~0.066, bin ~0.016)
    # visibility predicate (ObjectMap ageing)
    vis_range: tuple = (0.45, 4.5)     # m, horizontal range in which a present object is reliably detected
    vis_margin_px: int = 24            # at 720
    vis_occlusion_tol: float = 0.25    # m, depth in front of the expected point by more than this => occluded
    max_candidates: int = 150          # colour blobs examined per colour (largest first)


DEFAULT = DetectorConfig()


# ---- camera model ---------------------------------------------------------------------------------------------------
def quat_to_mat(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def head_pose(cam_rel_pose):
    """Head camera pose (7,) from the proprio `cam_rel_poses` entry (21 flat, (3, 7) or already (7,))."""
    if cam_rel_pose is None:
        return None
    a = np.asarray(cam_rel_pose, np.float64).reshape(-1)
    if a.size == 21:
        return a[14:21]
    if a.size == 7:
        return a
    return None


def camera_from_pose(pose7):
    """(R, t): OpenCV-convention camera (x right, y down, z forward) to robot base frame, or None if implausible.

    Plausibility (ranges seen over all 199 train/val demos and the replay run, with margin): position x in [-0.1, 0.9],
    |y| < 0.15, z in [0.6, 1.8]; optical axis pointing forward (x > 0.1) and not upward. Right after an env reset the
    proprio `cam_rel_poses` can hold a garbage (world-like) pose; detections are suppressed then."""
    if pose7 is None or not np.all(np.isfinite(pose7)):
        return None
    t, q = pose7[:3], pose7[3:7]
    if not (0.9 < np.linalg.norm(q) < 1.1) or not (-0.1 <= t[0] <= 0.9 and abs(t[1]) < 0.15 and 0.6 <= t[2] <= 1.8):
        return None
    R = quat_to_mat(q / np.linalg.norm(q)) @ CV_FROM_USD
    ax = R[:, 2]
    if ax[0] < 0.1 or abs(ax[1]) > 0.3 or ax[2] > 0.0:
        return None
    return R, t.copy()


def intrinsics(width):
    f = HEAD_F_720 * width / 720.0
    return f, width / 2.0, width / 2.0


class Camera:
    def __init__(self, R, t, width, height):
        self.R, self.t, self.W, self.H = R, t, int(width), int(height)
        self.f, self.cx, self.cy = HEAD_F_720 * width / 720.0, width / 2.0, height / 2.0

    def rays(self, u, v):
        """Base-frame direction of pixel rays, scaled so that the camera z component is 1 (z-depth multiplies)."""
        d = np.stack([(np.asarray(u, np.float64) - self.cx) / self.f, (np.asarray(v, np.float64) - self.cy) / self.f,
                      np.ones(np.shape(u))], -1)
        return d @ self.R.T

    def backproject(self, u, v, z):
        return self.t + self.rays(u, v) * np.asarray(z, np.float64)[..., None]

    def plane_hit(self, u, v, z_plane):
        """Intersection of pixel rays with the horizontal plane z = z_plane; (points, z-depth, valid)."""
        d = self.rays(u, v)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (z_plane - self.t[2]) / d[..., 2]
        ok = np.isfinite(lam) & (lam > 0.05) & (d[..., 2] < -1e-3)
        lam = np.where(ok, lam, np.nan)
        return self.t + d * lam[..., None], lam, ok

    def project(self, p):
        """Base-frame points (N, 3) -> pixel (u, v) and z-depth."""
        pc = (np.atleast_2d(p) - self.t) @ self.R
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.f * pc[:, 0] / z + self.cx
            v = self.f * pc[:, 1] / z + self.cy
        return u, v, z


# ---- input normalisation ----------------------------------------------------------------------------------------------
def _rgb(img):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):    # CHW -> HWC
        a = np.moveaxis(a, 0, -1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _depth(depth, shape):
    if depth is None:
        return None
    d = np.asarray(depth, np.float32)
    if d.ndim == 3:
        d = d[..., 0] if d.shape[-1] == 1 else d[0]
    if d.shape != shape:
        d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    d = np.where(np.isfinite(d) & (d > 0.05) & (d < 30.0), d, 0.0).astype(np.float32)   # 0 = invalid
    return d


def _held_eefs(state, cfg):
    """Base-frame xyz of EEFs whose gripper is closed (holding the bin or a can)."""
    out = []
    s = np.asarray(state, np.float64).reshape(-1) if state is not None else np.zeros(0)
    if s.size >= 61:
        for hand in ("left", "right"):
            if float(np.sum(s[GRIPPER[hand]])) < cfg.gripper_closed_sum:
                out.append(s[EEF_POS[hand]].copy())
    return out


# ---- detectors ------------------------------------------------------------------------------------------------------
def _colour_masks(hsv, cfg):
    orange = cv2.inRange(hsv, (cfg.orange_h[0], cfg.orange_s_min, cfg.orange_v_min), (cfg.orange_h[1], 255, 255))
    blue = cv2.inRange(hsv, (cfg.blue_h[0], cfg.blue_s_min, cfg.blue_v[0]), (cfg.blue_h[1], 255, cfg.blue_v[1]))
    return orange, blue


def _copper_mask(hsv_small, cfg):
    h, s, v = hsv_small[..., 0], hsv_small[..., 1].astype(np.int16), hsv_small[..., 2].astype(np.int16)
    can_orange = (s >= cfg.orange_s_min) & (v >= cfg.orange_v_min)          # bright saturated = can, not interior
    m = ((h >= cfg.copper_h[0]) & (h <= cfg.copper_h[1]) & (s >= cfg.copper_s_min) & (v >= cfg.copper_v_min)
         & (s >= v + cfg.copper_s_minus_v) & ~can_orange)
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))       # wood-grain speckle
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


TRACE = None      # set to a list to record every candidate with its features and rejection reason (offline tuning)


def _trace(kind, reason, **info):
    if TRACE is not None:
        TRACE.append({"kind": kind, "reject": reason, **info})
    return reason


def _detect_bins(hsv, depth, cam, cfg, held):
    """Returns [(det, hull_in_full_res_pixels)] and all large copper hulls (for in-bin rejection of can blobs)."""
    H, W = hsv.shape[:2]
    k = 2                                                        # copper at half resolution (bins are >= ~14 px wide)
    small = hsv[::k, ::k]
    mask = _copper_mask(small, cfg)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_px = max(4, int(cfg.bin_min_px * (W / 720.0) ** 2 / (k * k)))
    order = [i for i in np.argsort(-stats[:, cv2.CC_STAT_AREA]) if i > 0 and stats[i, cv2.CC_STAT_AREA] >= min_px]
    dets, hulls = [], []
    for i in order[:8]:
        x0, y0, w, h = stats[i, :4]
        ys, xs = np.nonzero(lab[y0:y0 + h, x0:x0 + w] == i)
        pts = np.stack([xs + x0, ys + y0], 1).astype(np.int32)
        hull = cv2.convexHull(pts)
        fill = np.zeros((h + 2, w + 2), np.uint8)
        cv2.fillConvexPoly(fill, hull - np.array([x0 - 1, y0 - 1], np.int32), 1)
        hy, hx = np.nonzero(fill)
        solidity = len(pts) / max(len(hx), 1)
        step = max(1, int(np.sqrt(len(hx) / 1500.0)))
        hx, hy = hx[::step * step], hy[::step * step]
        u = (hx + x0 - 1 + 0.5) * k - 0.5
        v = (hy + y0 - 1 + 0.5) * k - 0.5
        P, lam, ok = cam.plane_hit(u, v, BIN_RIM_Z)
        hull_full = (hull.reshape(-1, 2).astype(np.float64) + 0.5) * k - 0.5
        info = {"px": int(len(pts) * k * k), "solidity": float(solidity), "uv": (float(u.mean()), float(v.mean()))}
        if ok.mean() < 0.9:
            _trace("bin", "above_horizon", **info)
            continue
        P = P[ok][:, :2]
        c = P.mean(0)
        ev = np.linalg.eigvalsh(np.cov((P - c).T) + 1e-9 * np.eye(2))
        r_minor, r_major = 2.0 * np.sqrt(np.maximum(ev, 0))
        info.update(r_major=float(r_major), r_minor=float(r_minor), center=c.copy())
        if r_major >= cfg.bin_radius[0]:
            hulls.append(hull_full)
        if not (cfg.bin_radius[0] <= r_major <= cfg.bin_radius[1] and r_minor >= cfg.bin_minor_min):
            _trace("bin", "disk_size", **info)
            continue
        if solidity < cfg.bin_solidity_min:
            _trace("bin", "solidity", **info)
            continue
        if depth is not None:
            # interior depth points of the copper pixels (not the filled hull) must lie inside the rim disk, below the rim
            uu = np.clip(((xs + x0 + 0.5) * k - 0.5).astype(int), 0, W - 1)
            vv = np.clip(((ys + y0 + 0.5) * k - 0.5).astype(int), 0, H - 1)
            z = depth[vv, uu]
            good = z > 0
            if good.sum() < 5:
                _trace("bin", "no_depth", **info)
                continue
            Q = cam.backproject(uu[good], vv[good], z[good])
            rad = np.linalg.norm(Q[:, :2] - c, axis=1)
            info.update(disk=float(np.median(rad)), disk80=float(np.percentile(rad, 80)),
                        z90=float(np.percentile(Q[:, 2], 90)), z05=float(np.percentile(Q[:, 2], 5)))
            # the far inner wall reaches up to the rim (z90 well above the floor; a floor patch has z ~ 0), a carried
            # bin is higher than the rim of a standing one
            if not (cfg.bin_depth_disk[0] <= info["disk"] <= cfg.bin_depth_disk[1]) or info["disk80"] > cfg.bin_depth_disk80_max:
                _trace("bin", "depth_off_disk", **info)
                continue
            if not (cfg.bin_z90[0] <= info["z90"] <= cfg.bin_z90[1]) or info["z05"] < -0.08:
                _trace("bin", "depth_height", **info)
                continue
        if any(np.linalg.norm(c - e[:2]) < cfg.held_radius for e in held):
            _trace("bin", "held", **info)
            continue                                               # carried bin (closed gripper on the rim)
        clipped = bool(hull_full[:, 0].min() <= 2 or hull_full[:, 1].min() <= 2 or hull_full[:, 0].max() >= W - 3
                       or hull_full[:, 1].max() >= H - 3)
        score = float(min(1.0, info["px"] / 400.0) * (0.6 if clipped else 1.0))
        rim = c - RIM_OFFSET
        _trace("bin", None, **info)
        info.pop("center")
        dets.append(({"label": "trash_can", "xyz_base": np.array([rim[0], rim[1], BIN_RIM_Z]), "score": score,
                      "center_base": np.array([c[0], c[1], BIN_RIM_Z / 2]), "clipped": clipped, **info}, hull_full))
    return dets, hulls


def _circle_fit(q):
    """Algebraic (Kasa) circle fit to (N, 2) points -> centre, radius, rms residual."""
    A = np.c_[q, np.ones(len(q))]
    sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
    c = sol[:2] / 2.0
    r = float(np.sqrt(max(sol[2] + c @ c, 0.0)))
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(q - c, axis=1) - r) ** 2)))


def _detect_bins_depth(depth, cam, cfg, held, state=None):
    """Geometric bin detector (needs depth): an isolated, hollow-topped cylinder of rim height standing on the floor.

    Depth (half resolution) -> base-frame points -> top-down grid (2 cm). Cells with points 0.04-0.34 m above the
    floor form blobs (after a 1-cell dilation that closes the gap between the visible outer front wall and the inner far
    wall). A bin blob: top (98th pct height) at the rim (0.24-0.31 m), footprint <= ~0.4 m, no taller structure
    (0.36-1.2 m) over its footprint (chair / table legs, furniture plinths), and its wall points lie on a circle of
    radius 0.08-0.13 m (robust algebraic circle fit -> centre). Appearance-free: works for the dark-copper and the lit
    tan interior alike; cans are too low."""
    H, W = depth.shape
    k = 2
    d = depth[k // 2::k, k // 2::k]
    vv, uu = np.mgrid[k // 2:H:k, k // 2:W:k]
    valid = d > 0
    P = cam.backproject(uu[valid], vv[valid], d[valid])
    pix = np.stack([uu[valid], vv[valid]], 1)
    keep = (P[:, 2] > 0.04) & (P[:, 2] < 1.2) & (np.hypot(P[:, 0], P[:, 1]) < cfg.bin_geo_range)
    P, pix = P[keep], pix[keep]
    res = cfg.bin_geo_cell
    x_lo, y_lo = -1.0, -cfg.bin_geo_range
    nx, ny = int((cfg.bin_geo_range + 1.0) / res) + 1, int(2 * cfg.bin_geo_range / res) + 1
    ix = np.clip(((P[:, 0] - x_lo) / res).astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - y_lo) / res).astype(int), 0, ny - 1)
    cell = ix * ny + iy
    low = P[:, 2] < cfg.bin_geo_zmax
    tall_pts = P[:, 2] > cfg.bin_geo_zmax + 0.02
    if state is not None:                                             # own arms / grippers above a bin (<1.2 m)
        near = tall_pts & (np.hypot(P[:, 0], P[:, 1]) < 1.2)
        tall_pts[near] = ~_self_mask(P[near], state, cam.t)
    low_grid = np.bincount(cell[low], minlength=nx * ny).reshape(nx, ny) > 0
    tall_grid = (np.bincount(cell[tall_pts], minlength=nx * ny).reshape(nx, ny) > 0).astype(np.uint8)
    blob = cv2.dilate(low_grid.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    max_cells = int((0.46 / res) ** 2)
    dets, hulls = [], []
    cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] * res <= 0.46 and stats[i, cv2.CC_STAT_HEIGHT] * res <= 0.46
            and 4 <= stats[i, cv2.CC_STAT_AREA] <= max_cells]
    if not cand:
        return dets, hulls
    pl = lab.reshape(-1)[cell]
    for i in cand:
        sel = low & (pl == i)
        pts = P[sel]
        if len(pts) < cfg.bin_geo_min_pts:
            continue
        info = {"px": int(len(pts) * k * k), "uv": tuple(float(v) for v in pix[sel].mean(0)), "geo": True}
        ztop = float(np.percentile(pts[:, 2], 98))
        info["ztop"] = ztop
        if not (cfg.bin_geo_top[0] <= ztop <= cfg.bin_geo_top[1]):
            _trace("bin", "geo_height", **info)
            continue
        x0, y0, bw, bh = stats[i, :4]           # note: grid x axis = columns of `lab` = iy
        fp = (lab[max(y0 - 3, 0):y0 + bh + 3, max(x0 - 3, 0):x0 + bw + 3] == i).astype(np.uint8)
        fp = cv2.dilate(fp, np.ones((5, 5), np.uint8))
        n_tall = int((fp & tall_grid[max(y0 - 3, 0):y0 + bh + 3, max(x0 - 3, 0):x0 + bw + 3]).sum())
        info["tall_cells"] = n_tall
        if n_tall > cfg.bin_geo_tall_cells:
            _trace("bin", "geo_continues_up", **info)
            continue
        q = pts[(pts[:, 2] > 0.05) & (pts[:, 2] < 0.30)][:, :2]
        if len(q) < 8:
            _trace("bin", "geo_few", **info)
            continue
        c, r, resid = _circle_fit(q)
        inl = np.abs(np.linalg.norm(q - c, axis=1) - r) < 0.03          # drop a can / debris touching the bin
        if 8 <= inl.sum() < len(q):
            c, r, resid = _circle_fit(q[inl])
        info.update(center=c.copy(), r_fit=r, resid=resid)
        if not (cfg.bin_geo_radius[0] <= r <= cfg.bin_geo_radius[1]) or resid > cfg.bin_geo_resid:
            _trace("bin", "geo_circle", **info)
            continue
        hull_full = cv2.convexHull(pix[sel].astype(np.int32)).reshape(-1, 2).astype(np.float64)
        hulls.append(hull_full)
        if any(np.linalg.norm(c - e[:2]) < cfg.held_radius for e in held):
            _trace("bin", "held", **info)
            continue
        clipped = bool(hull_full[:, 0].min() <= 3 or hull_full[:, 1].min() <= 3 or hull_full[:, 0].max() >= W - 4
                       or hull_full[:, 1].max() >= H - 4)
        score = float(min(1.0, info["px"] / 300.0) * (0.6 if clipped else 1.0))
        _trace("bin", None, **info)
        info.pop("center")
        rim = c - RIM_OFFSET
        dets.append(({"label": "trash_can", "xyz_base": np.array([rim[0], rim[1], BIN_RIM_Z]), "score": score,
                      "center_base": np.array([c[0], c[1], BIN_RIM_Z / 2]), "clipped": clipped, **info}, hull_full))
    return dets, hulls


def _inside_any(hulls, u, v, pad=3.0):
    for h in hulls:
        if cv2.pointPolygonTest(h.astype(np.float32).reshape(-1, 1, 2), (float(u), float(v)), True) >= -pad:
            return True
    return False


def _seg_dist(e, a, b):
    """Distance from point e to the 3-D segment a-b."""
    ab = b - a
    t = float(np.clip((e - a) @ ab / max(ab @ ab, 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(e - (a + t * ab)))


def _detect_cans(masks, depth, cam, cfg, hulls, held, hsv=None):
    H, W = masks[0].shape
    min_px = max(2, int(round(cfg.can_min_px * (W / 720.0) ** 2)))
    dets = []
    for colour, mask in zip(("orange", "blue"), masks):
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        order = [i for i in np.argsort(-stats[:, cv2.CC_STAT_AREA]) if i > 0 and stats[i, cv2.CC_STAT_AREA] >= min_px]
        for i in order[:cfg.max_candidates]:
            x0, y0, w, h, area = (int(v) for v in stats[i])
            uc, vc = (float(v) for v in cents[i])
            info = {"colour": colour, "px": area, "uv": (uc, vc), "bbox": (x0, y0, w, h)}
            if hsv is not None:
                blob = lab[y0:y0 + h, x0:x0 + w] == i
                info["mean_sv"] = tuple(float(v) for v in hsv[y0:y0 + h, x0:x0 + w][blob][:, 1:].mean(0))
                lo_s, lo_v = cfg.orange_blob_sv if colour == "orange" else cfg.blue_blob_sv
                if (info["mean_sv"][0] < lo_s or info["mean_sv"][1] < lo_v
                        or (colour == "blue" and info["mean_sv"][1] > cfg.blue_blob_v_max)):
                    _trace("can", "blob_colour", **info)
                    continue                                          # dull wood (door frame / furniture), not a can
            if depth is not None:
                # cheap pre-check (brick walls / furniture give many large blobs): metric bbox at the centroid depth
                win = depth[max(int(vc) - 1, 0):int(vc) + 2, max(int(uc) - 1, 0):int(uc) + 2]
                win = win[win > 0]
                if win.size and min(w, h) * float(win.min()) / cam.f > 1.5 * cfg.can_width[1]:
                    _trace("can", "size", **info)
                    continue
            if _inside_any(hulls, uc, vc):
                _trace("can", "in_bin_hull", **info)
                continue                                              # can in the bin / bin-interior speckle
            if depth is not None:
                ys, xs = np.nonzero(lab[y0:y0 + h, x0:x0 + w] == i)
                z = depth[ys + y0, xs + x0]
                good = z > 0
                if good.sum() < max(2, 0.3 * area):
                    _trace("can", "no_depth", **info)
                    continue
                zs = z[good]
                zref = float(np.percentile(zs, 30))                  # can surface, not the floor behind edge pixels
                keep = zs <= zref + 0.08
                Q = cam.backproject(xs[good][keep] + x0, ys[good][keep] + y0, zs[keep])
                ztop, zmid = float(np.percentile(Q[:, 2], 90)), float(np.median(Q[:, 2]))
                p = np.median(Q, 0)
                # the visible half-cylinder's centroid is ~0.64 r in front of the axis: push away from the camera
                d = p[:2] - cam.t[:2]
                p[:2] += 0.64 * CAN_RADIUS * d / max(np.linalg.norm(d), 1e-6)
            else:
                P, lam, ok = cam.plane_hit(np.array([uc]), np.array([vc]), CAN_CENTRE_Z)
                if not ok[0]:
                    _trace("can", "above_horizon", **info)
                    continue
                zref, p = float(lam[0]), P[0]
                ztop = zmid = CAN_CENTRE_Z                           # no height evidence without depth
            wm, hm = w * zref / cam.f, h * zref / cam.f
            info.update(p=p.copy(), zref=zref, size_m=(wm, hm), ztop=ztop, zmid=zmid)
            if np.hypot(p[0], p[1]) > cfg.max_range:
                _trace("can", "range", **info)
                continue
            if not (cfg.can_width[0] <= wm <= cfg.can_width[1] and cfg.can_height[0] <= hm <= cfg.can_height[1]):
                _trace("can", "size", **info)
                continue
            if min(w, h) >= 3 and not (cfg.can_aspect[0] * wm <= hm <= cfg.can_aspect[1] * wm):
                _trace("can", "aspect", **info)
                continue                                              # flat strip / thin post, not a can
            gap = int(np.ceil(0.07 * cam.f / zref)) + 1                # skip the (non-coloured) lid / upper rim
            if depth is not None and y0 - gap >= 3:
                # an isolated can has floor (farther) or a gripper (nearer) above it; a structure that continues
                # upward at the same depth (door frame, furniture edge) is not a can
                band = depth[max(y0 - gap - 4, 0):y0 - gap, x0:x0 + w]
                band = band[band > 0]
                same = float(np.mean(np.abs(band - zref) < 0.04 + 0.01 * zref)) if band.size >= 2 else 0.0
                info["continue_frac"] = same
                if same > cfg.can_continue_frac:
                    _trace("can", "continues_up", **info)
                    continue
            if ztop > cfg.can_top_z_max or not (cfg.can_mid_z[0] <= zmid <= cfg.can_mid_z[1]):
                _trace("can", "height", **info)
                continue
            if any(np.linalg.norm(p[:2] - e[:2]) < cfg.held_radius for e in held) or (
                    depth is None and any(_seg_dist(e, cam.t, p) < cfg.held_ray_dist for e in held)):
                _trace("can", "held", **info)
                continue                                              # held can / can in the carried bin
            p[2] = CAN_CENTRE_Z
            score = float(1.0 - np.exp(-area * (720.0 / W) ** 2 / 12.0))
            _trace("can", None, **info)
            dets.append({"label": "can_of_soda", "xyz_base": p, "score": score, "colour": colour, "px": area,
                         "uv": (uc, vc), "size_m": (float(wm), float(hm))})
    return dets


def _nms(dets, radius):
    """Merge duplicates of the same label closer than `radius` (keep the higher score)."""
    out = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(o["label"] != d["label"] or np.linalg.norm(o["xyz_base"][:2] - d["xyz_base"][:2]) > radius for o in out):
            out.append(d)
    return out


def make_visible_fn(cam, depth, cfg=DEFAULT):
    """visible_fn(p_base) -> bool: the point projects inside the image (with margin), lies in the reliable detection
    range and is not occluded (depth at its pixel not much in front of it). 2-D points get z = 0.1 m (between the can
    centre and the rim point)."""
    margin = cfg.vis_margin_px * cam.W / 720.0

    def visible(p_base):
        p = np.asarray(p_base, np.float64).reshape(-1)
        p = np.array([p[0], p[1], p[2] if p.size >= 3 else 0.1])
        rng = float(np.hypot(p[0], p[1]))
        if not (cfg.vis_range[0] <= rng <= cfg.vis_range[1]):
            return False
        u, v, z = cam.project(p[None])
        u, v, z = float(u[0]), float(v[0]), float(z[0])
        if not (z > 0.1 and margin <= u < cam.W - margin and margin <= v < cam.H - margin):
            return False
        if depth is not None:
            ui, vi = int(u), int(v)
            win = depth[max(vi - 3, 0):vi + 4, max(ui - 3, 0):ui + 4]
            win = win[win > 0]
            if win.size and float(np.median(win)) < z - cfg.vis_occlusion_tol:
                return False
        return True

    return visible


def detect_from_observation(rgb, depth=None, cam_rel_pose=None, state=None, **overrides):
    """Head RGB(-D) + head camera pose (+ proprio) -> (detections, visible_fn).

    detections: [{"label": "trash_can" | "can_of_soda", "xyz_base": (3,), "score": float, ...extra diagnostics}]
    (trash_can xyz_base = right-hand rim grasp point, see module docstring). visible_fn is None when the camera pose
    is missing or implausible (then no detections either)."""
    cfg = replace(DEFAULT, **overrides) if overrides else DEFAULT
    img = _rgb(rgb)
    H, W = img.shape[:2]
    cp = camera_from_pose(head_pose(cam_rel_pose))
    if cp is None:
        return [], None
    cam = Camera(cp[0], cp[1], W, H)
    dep = _depth(depth, (H, W))
    held = _held_eefs(state, cfg)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if dep is not None:
        # geometric detector; its hulls (incl. held bins standing on the floor) reject cans inside a bin. Cans in a
        # carried bin are rejected by the closed-gripper rule.
        bins, hulls = _detect_bins_depth(dep, cam, cfg, held, state)
    elif cfg.bin_without_depth:
        bins, hulls = _detect_bins(hsv, dep, cam, cfg, held)
        hulls = [h for (_, h) in bins]
    else:
        bins, hulls = [], []
    cans = _detect_cans(_colour_masks(hsv, cfg), dep, cam, cfg, hulls, held, hsv)
    dets = _nms([b for b, _ in bins], 0.3) + _nms(cans, 0.08)
    return dets, make_visible_fn(cam, dep, cfg)


# ---- obstacles --------------------------------------------------------------------------------------------------------
def _self_mask(P, state, cam_t):
    """True for points on the robot itself: forearm capsules (EEF -> approximate shoulder below/beside the head camera),
    a gripper sphere, a held-object cylinder around a *closed* gripper (covers a carried bin: 0.3 m radius, 0.4 m
    below the EEF), and the base footprint box.  The closed-gripper qualification matters: dev10 v7 drove an open
    left hand into a wooden chair, but the old unconditional 0.3 m cylinder erased the chair from the obstacle cloud
    exactly when it became hazardous. Approximation without arm FK; tuned on the replay frames (see eval script)."""
    bad = (P[:, 0] > -0.45) & (P[:, 0] < 0.33) & (np.abs(P[:, 1]) < 0.33)              # body / base footprint
    s = np.asarray(state, np.float64).reshape(-1) if state is not None else np.zeros(0)
    if s.size < 61:
        return bad | ((P[:, 0] < 0.75) & (np.abs(P[:, 1]) < 0.6) & (P[:, 2] > 0.25))      # no proprio: arm-reach box
    for hand, sign in (("left", 1.0), ("right", -1.0)):
        e = s[EEF_POS[hand]]
        sh = np.array([cam_t[0] - 0.15, sign * 0.23, cam_t[2] - 0.30])                   # approximate shoulder
        seg = sh - e
        L2 = float(seg @ seg) + 1e-9
        tt = np.clip((P - e) @ seg / L2, 0.0, 1.0)
        dist = np.linalg.norm(P - (e + tt[:, None] * seg), axis=1)
        bad |= dist < 0.11                 # physical arm capsule, not a large collision-free halo
        if float(np.sum(s[GRIPPER[hand]])) < DetectorConfig.gripper_closed_sum:
            bad |= ((np.hypot(P[:, 0] - e[0], P[:, 1] - e[1]) < 0.30)
                    & (P[:, 2] > e[2] - 0.42) & (P[:, 2] < e[2] + 0.35))
        else:
            bad |= np.linalg.norm(P - e, axis=1) < 0.12
    return bad


def obstacle_points(depth, cam_rel_pose, state=None, z_band=(0.05, 1.5), max_range=6.0, stride=6, max_points=2000):
    """Base-frame xy (N, 2) of head depth points in the furniture height band, excluding the robot's own arms, grippers
    and a carried object (see _self_mask), subsampled to <= max_points (depth decimated by `stride` first)."""
    cp = camera_from_pose(head_pose(cam_rel_pose))
    if depth is None or cp is None:
        return np.zeros((0, 2))
    d = np.asarray(depth, np.float32)
    if d.ndim == 3:
        d = d[..., 0] if d.shape[-1] == 1 else d[0]
    H, W = d.shape
    cam = Camera(cp[0], cp[1], W, H)
    vv, uu = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
    z = d[vv, uu]
    ok = np.isfinite(z) & (z > 0.1) & (z < max_range + 2)
    P = cam.backproject(uu[ok], vv[ok], z[ok])
    keep = (P[:, 2] > z_band[0]) & (P[:, 2] < z_band[1]) & (np.hypot(P[:, 0], P[:, 1]) < max_range)
    P = P[keep]
    P = P[~_self_mask(P, state, cam.t)]
    if len(P) > max_points:
        P = P[np.linspace(0, len(P) - 1, max_points).astype(int)]
    return P[:, :2].copy()


# ---- CPU self-test on synthetic data ------------------------------------------------------------------------------------
def _synthetic_scene(W=720):
    """Wood-coloured floor, a 0.9 m / 1.39 m high head camera pitched down 30 deg, one orange can at (2.0, 0.3), one blue
    can at (3.0, -0.5), a bin (grey wall + copper interior) centred at (1.6, -0.6). Rendered by ray casting."""
    pitch = np.radians(30.0)
    # USD camera axes in the base frame: -Z (forward) = (cos p, 0, -sin p), +Y (up) = (sin p, 0, cos p), +X = (0, -1, 0)
    fwd, up = np.array([np.cos(pitch), 0, -np.sin(pitch)]), np.array([np.sin(pitch), 0, np.cos(pitch)])
    Rusd = np.stack([np.cross(up, -fwd), up, -fwd], 1)
    tr = np.trace(Rusd)
    qw = np.sqrt(max(1 + tr, 1e-12)) / 2
    q = np.array([(Rusd[2, 1] - Rusd[1, 2]) / (4 * qw), (Rusd[0, 2] - Rusd[2, 0]) / (4 * qw),
                  (Rusd[1, 0] - Rusd[0, 1]) / (4 * qw), qw])
    pose = np.r_[0.24, 0.0, 1.39, q]
    R, t = camera_from_pose(pose)
    cam = Camera(R, t, W, W)
    vv, uu = np.mgrid[0:W, 0:W]
    d = cam.rays(uu + 0.0, vv + 0.0)
    depth = np.full((W, W), np.inf, np.float32)
    rgb = np.zeros((W, W, 3), np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(d[..., 2] < 0, -t[2] / d[..., 2], np.inf)
    depth[:] = lam
    rgb[:] = (205, 170, 135)                                               # wood floor
    rgb[np.isinf(lam)] = (235, 235, 235)                                   # wall / sky

    def cylinder(cx, cy, r, h, colour_side, colour_top=None, hollow=False):
        ox, oy = t[0] - cx, t[1] - cy
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = 2 * (ox * d[..., 0] + oy * d[..., 1])
        c = ox * ox + oy * oy - r * r
        disc = b * b - 4 * a * c
        with np.errstate(invalid="ignore", divide="ignore"):
            l1 = (-b - np.sqrt(disc)) / (2 * a)
        zh = t[2] + l1 * d[..., 2]
        side = (disc > 0) & (l1 > 0) & (zh >= 0) & (zh <= h) & (l1 < depth)
        depth[side] = l1[side]
        rgb[side] = colour_side
        with np.errstate(divide="ignore", invalid="ignore"):
            lt = (h - t[2]) / d[..., 2]
        xt, yt = t[0] + lt * d[..., 0], t[1] + lt * d[..., 1]
        top = (lt > 0) & ((xt - cx) ** 2 + (yt - cy) ** 2 <= r * r) & (lt < depth + 1e-6)
        if hollow:                                         # interior seen through the rim: far inner wall or bottom
            with np.errstate(divide="ignore", invalid="ignore"):
                l2 = (-b + np.sqrt(disc)) / (2 * a)
                lb = (0.02 - t[2]) / d[..., 2]
            depth[top] = np.where(t[2] + l2[top] * d[..., 2][top] >= 0.02, l2[top], lb[top])
            rgb[top] = colour_top
        elif colour_top is not None:
            depth[top] = lt[top]
            rgb[top] = colour_top

    cylinder(1.6, -0.6, BIN_RADIUS, BIN_RIM_Z, (170, 170, 172), (150, 95, 55), hollow=True)
    cylinder(2.0, 0.3, CAN_RADIUS, 0.13, (235, 90, 20), (190, 190, 190))
    cylinder(3.0, -0.5, CAN_RADIUS, 0.13, (20, 30, 110), (190, 190, 190))
    cam21 = np.r_[np.zeros(14), pose]
    return rgb, depth, cam21


def _self_test():
    import time
    rgb, depth, cam21 = _synthetic_scene()
    state = np.zeros(61)
    state[24:26] = state[49:51] = 0.05                                     # grippers open
    state[17:20], state[42:45] = (0.3, 0.25, 0.5), (0.3, -0.25, 0.5)
    truth = {"orange": np.array([2.0, 0.3]), "blue": np.array([3.0, -0.5])}
    bin_centre = np.array([1.6, -0.6])
    for with_depth in (True, False):
        dets, vis = detect_from_observation(rgb, depth if with_depth else None, cam21, state, bin_without_depth=True)
        cans = [d for d in dets if d["label"] == "can_of_soda"]
        bins = [d for d in dets if d["label"] == "trash_can"]
        assert len(cans) == 2 and len(bins) == 1, (with_depth, [(d["label"], d["xyz_base"].round(2)) for d in dets])
        for d in cans:
            err = np.linalg.norm(d["xyz_base"][:2] - truth[d["colour"]])
            assert err < 0.06, (d["colour"], err)
        e_bin = np.linalg.norm(bins[0]["center_base"][:2] - bin_centre)
        assert e_bin < 0.05, e_bin
        assert np.allclose(bins[0]["xyz_base"][:2], bins[0]["center_base"][:2] - RIM_OFFSET)
        assert vis(truth["orange"]) and not vis(np.array([-1.0, 0.0])) and not vis(np.array([0.2, 3.0]))
        print(f"depth={with_depth}: cans " + ", ".join(f"{d['colour']} {d['xyz_base'][:2].round(3)}" for d in cans)
              + f"; bin centre {bins[0]['center_base'][:2].round(3)} (err {e_bin:.3f} m)")
    # occlusion: a wall 1 m in front of the camera hides everything behind it
    wall = np.full_like(depth, 1.0)
    _, vis = detect_from_observation(rgb, wall, cam21, state)
    assert not vis(truth["orange"])
    # implausible (post-reset) camera pose -> no detections, no visibility predicate
    bad = cam21.copy()
    bad[14:17] = (2.75, -3.83, 1.39)
    assert detect_from_observation(rgb, depth, bad, state) == ([], None)
    # held rejection: closed left gripper right next to the orange can
    s2 = state.copy()
    s2[24:26] = 0.005
    s2[17:20] = (2.0, 0.3, 0.3)
    assert not any(d.get("colour") == "orange" for d in detect_from_observation(rgb, depth, cam21, s2)[0])
    # An open hand must not erase nearby furniture. The same point is masked only after the gripper closes, when it
    # can plausibly belong to a carried object (regression for the dev10-v7 dragged-chair failure).
    cam_t = camera_from_pose(head_pose(cam21))[1]
    probe = np.array([[0.30, 0.48, 0.50]])
    assert not bool(_self_mask(probe, state, cam_t)[0])
    s3 = state.copy()
    s3[24:26] = 0.005
    assert bool(_self_mask(probe, s3, cam_t)[0])
    obs = obstacle_points(depth, cam21, state)
    assert obs.ndim == 2 and obs.shape[1] == 2 and len(obs) <= 2000
    assert np.any(np.linalg.norm(obs - bin_centre, axis=1) < 0.2)          # the bin wall is an obstacle point
    t0 = time.perf_counter()
    for _ in range(20):
        detect_from_observation(rgb, depth, cam21, state)
        obstacle_points(depth, cam21, state)
    print(f"latency detect+obstacles: {(time.perf_counter() - t0) / 20 * 1e3:.1f} ms per call (720x720, CPU)")
    print("PASS object_perception self-test")


if __name__ == "__main__":
    _self_test()
