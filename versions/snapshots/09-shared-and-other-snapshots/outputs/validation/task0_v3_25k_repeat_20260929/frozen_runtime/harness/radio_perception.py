"""Onboard Task 0 radio detector: head RGB(-D) + head camera pose -> base-frame radio GRASP POINT (docking input).

Same interface as harness/object_perception.py (Task 1), so a Task 0 perception adapter can call it the same way:
    detect_from_observation(rgb, depth=None, cam_rel_pose=None, state=None, **overrides) -> (detections, visible_fn)
and `obstacle_points` (re-exported from object_perception: furniture-height depth points, own arms / a held object
removed) for the docking controller's corridor check.

Inputs are policy-visible only (no simulator state is ever read here):
  * head camera (zed_link) RGB, native 720x720 (HxWx3/4 uint8; other square sizes rescale the intrinsics),
  * head depth_linear (z-depth, metres) or None (then a table-plane fallback localises the blob),
  * the proprio `cam_rel_poses` entry (flat 21 / (3, 7) / head 7-vector; parsed by object_perception.head_pose),
  * the 61-D proprio (EEF positions + finger widths: a radio next to a CLOSED gripper is the robot's own held radio).

Radio (BEHAVIOR asset radio/wxnicr): bbox 0.138 (x, thickness) x 0.321 (y, length) x 0.235 (z) m; large RED front and
back faces, white rounded band around ends / top, a flat RED handle bar on top. It stands on the coffee table
(table top ~0.42 m, bbox centre ~0.534 m above the floor in all TRAIN instances).

Point convention (what docking_controller's Task 0 calibration expects):
  outputs/validation/task0_docking_calibration_20260927/grasp_relative_poses_train.json (built by
  scripts/build_task0_docking_calibration.py) stores the RIGHT-HAND GRASP POINT = right EEF position at gripper
  closure, in the base frame at pick start; docking_controller.load_calibration uses its median as the docking target
  and ObjectMap consumes detections' xyz_base[:2]. The grasp point sits on the handle: TRAIN median
  (EEF - bbox centre) in the RADIO frame = (+0.014, +0.018, +0.101) m (scripts/build_task0_object_gt.py calibration,
  gt_summary.json). So  xyz_base = estimated bbox centre + (0, 0, GRASP_DZ). The +1.4/+1.8 cm in-plane part is not
  applied: the radio's front/back faces look alike (both red), so its yaw is only known modulo 180 deg and the sign of
  that offset is unobservable (<= 2.3 cm, below the detector error). The bbox centre is returned as "center_base".

Method (no learning; thresholds from TRAIN data only, see scripts/eval_radio_perception.py):
  1. Full-resolution HSV red mask (|H| <= 7, S >= 150, V >= 55; wood / skin-tone orange has H 8-10). Pieces of one
     radio (face split by the white speaker ring, handle bar) are joined by a small dilation before connected
     components; each component keeps only its own red pixels.
  2. With depth: red pixels -> base-frame points (nearest depth layer only). Accepted if the points are radio-sized
     (horizontal extent <= 0.45 m, vertical <= 0.34 m) and at radio height on a table: 5th pct height 0.33-0.62 m, 95th pct
     0.48-0.80 m (paintings, wall switches, kitchen lights, outdoor play equipment are >= 0.9 m; floor rugs ~0 m).
     Centre: PCA of the points' xy gives the long axis a / normal n; the centre is the middle of the point extent
     along a, and along n the middle shifted away from the camera by (T - extent_n) / 2 (T = 0.138 m: a single visible
     red face lies 0.069 m in front of the centre; both faces seen edge-on -> no shift). Few points (far radio): median
     pushed away from the camera by FAR_PUSH.
     Without depth: the red pixel centroid ray is intersected with the plane z = RADIO_CENTRE_Z (+ FAR_PUSH along the
     viewing direction), size-checked with that range.
  3. Held: a component within `held_radius` (3-D, depth) / `held_ray_dist` of the viewing ray (no depth) of an EEF whose
     gripper is closed is the radio in the robot's own hand: suppressed by default (the docking controller must never
     navigate to it; ObjectMap would otherwise make a track of it), returned with "held": True if return_held=True.
     The robot's own arms (white/black) are not red; red points that fall inside object_perception._self_mask
     (arm capsules / closed-gripper envelope) are dropped before the geometry tests.
visible_fn: object_perception.make_visible_fn semantics; 2-D points (ObjectMap tracks) are lifted to the radio centre
height (object_perception lifts to 0.1 m, which would test the floor under the coffee table).
numpy + cv2 only. CPU self-test on a synthetic scene: `python3 -m harness.radio_perception`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from harness.object_perception import (  # noqa: F401  (re-exported: same camera math as Task 1)
    EEF_POS, GRIPPER, HEAD_F_720, Camera, _depth, _held_eefs, _rgb, _self_mask, _seg_dist, camera_from_pose, head_pose,
    intrinsics, make_visible_fn, obstacle_points, quat_to_mat)

# ---- radio geometry (asset wxnicr + TRAIN demos) -------------------------------------------------------------------
RADIO_SIZE = np.array([0.138, 0.321, 0.235])  # m: x thickness (face normal), y length (handle axis), z height
RADIO_CENTRE_Z = 0.534                        # m: bbox centre height standing on the coffee table (TRAIN instances)
TABLE_TOP_Z = 0.42                            # m
GRASP_OFFSET_RADIO = np.array([0.0136, 0.0178, 0.1010])  # m: TRAIN median (right EEF at closure - bbox centre), radio frame
GRASP_DZ = float(GRASP_OFFSET_RADIO[2])       # applied offset (base z; the radio stands upright, tilt ~3 deg)


@dataclass
class RadioConfig:
    # colour gate (OpenCV HSV, H in [0, 180), red wraps around 0)
    red_h: int = 7                    # |H| <= red_h  (radio: 98 % of pixels H in [-1, 6]; wood / floor 8-10)
    red_s_min: int = 150
    red_v_min: int = 55               # shaded faces are dark but stay saturated
    blob_s_min: float = 185.0         # component mean S (radio >= 220 in sim renders)
    min_px: int = 10                  # red pixels per component at 720x720 (radio at 6 m: ~60 px)
    join_px: int = 2                  # dilation radius (px at 720) before connected components
    piece_min_px: int = 3             # red pixels of a component to be kept as a piece
    max_candidates: int = 60          # components examined (largest first)
    group_dist: float = 0.26          # m: pieces whose representative 3-D points are closer are one object (depth)
    group_dist_nodepth: float = 0.35  # m: same on the z = RADIO_CENTRE_Z plane (no depth)
    # depth geometry
    near_layer: float = 0.45          # m: keep red points within this z-depth behind the 15th percentile (one object)
    min_pts: int = 6                  # valid depth points
    xy_extent_max: float = 0.45       # m: p5-p95 horizontal extent (radio diagonal 0.35)
    z_extent_max: float = 0.34        # m: p5-p95 vertical extent (radio 0.235)
    z_bottom: tuple = (0.33, 0.62)    # m: 5th pct height of the red points (standing radio 0.43-0.51)
    z_top: tuple = (0.48, 0.80)       # m: 95th pct height (radio top 0.65)
    face_min_pts: int = 40            # points for the PCA face model; fewer -> FAR_PUSH fallback
    far_push: float = 0.06            # m: visible-surface -> centre shift along the viewing direction (fallback)
    # depth-free fallback
    size_min: float = 0.035           # m: apparent max(width, height)
    size_max: tuple = (0.55, 0.45)    # m: apparent (width, height)
    # held / own-body rejection
    held_radius: float = 0.22         # m: closed-gripper EEF to the nearest red point (3-D)
    held_centre_radius: float = 0.30  # m: closed-gripper EEF to the estimated centre (3-D)
    held_ray_dist: float = 0.25       # m: no depth: closed EEF this close to the camera->blob segment
    gripper_closed_sum: float = 0.085 # m: finger sum below this = closed (open ~0.10; on the handle 0.02-0.085)
    return_held: bool = False         # include held detections (flagged "held": True) instead of suppressing them
    side_clip_min_ext: float = 0.22   # m: a blob touching the left/right image border (radio entering the view while
    #                                   the robot turns) needs this long-axis extent (depth; always rejected without
    #                                   depth): truncated partial views bias the centre by 5-10 cm (replay tuning)
    max_range: float = 9.0            # m: horizontal range limit
    nms_radius: float = 0.35          # m
    # visibility predicate (ObjectMap ageing): same keys as object_perception.DetectorConfig
    vis_range: tuple = (0.45, 5.0)    # m
    vis_margin_px: int = 24
    vis_occlusion_tol: float = 0.25
    vis_z: float = RADIO_CENTRE_Z     # height assigned to 2-D query points


DEFAULT = RadioConfig()
TRACE = None       # set to a list to record every candidate with its features and rejection reason (offline tuning)


def _trace(reason, **info):
    if TRACE is not None:
        TRACE.append({"reject": reason, **info})
    return reason


def red_mask(hsv, cfg=DEFAULT):
    h = hsv[..., 0]
    hue = (h <= cfg.red_h) | (h >= 180 - cfg.red_h)
    return (hue & (hsv[..., 1] >= cfg.red_s_min) & (hsv[..., 2] >= cfg.red_v_min)).astype(np.uint8)


def _centre_from_points(Q, cam_t, cfg):
    """Radio bbox centre (xy) from the visible red points (base frame); returns (xy, info)."""
    xy = Q[:, :2]
    view = np.median(xy, 0) - cam_t[:2]
    view /= max(np.linalg.norm(view), 1e-6)
    if len(Q) < cfg.face_min_pts:
        return np.median(xy, 0) + cfg.far_push * view, {"method": "median_push"}
    # face points: below the handle bar (top 3 cm)
    ztop = np.percentile(Q[:, 2], 97)
    face = xy[Q[:, 2] < ztop - 0.03]
    if len(face) < cfg.face_min_pts // 2:
        face = xy
    c0 = face.mean(0)
    ev, evec = np.linalg.eigh(np.cov((face - c0).T) + 1e-9 * np.eye(2))
    a, n = evec[:, 1], evec[:, 0]                                    # long axis, normal
    if n @ view < 0:
        n = -n                                                       # normal points away from the camera
    pa, pn = xy @ a, xy @ n
    a_lo, a_hi = np.percentile(pa, [3, 97])
    n_lo, n_hi = np.percentile(pn, [3, 97])
    ext_a, ext_n = a_hi - a_lo, n_hi - n_lo
    # unseen thickness lies behind the visible face; seen end-on (normal across the view) the side is ambiguous
    mid_n = 0.5 * (n_lo + n_hi) + 0.5 * max(0.0, RADIO_SIZE[0] - ext_n) * float(n @ view)
    mid_a = 0.5 * (a_lo + a_hi)
    if ext_a < 0.8 * RADIO_SIZE[1] and abs(a @ view) > 0.7:          # long axis along the view: far end unseen
        mid_a += 0.5 * max(0.0, RADIO_SIZE[1] - ext_a) * np.sign(a @ view)
    centre = mid_a * a + mid_n * n
    yaw = float(np.arctan2(n[1], n[0]))                              # radio x axis (face normal), modulo pi
    return centre, {"method": "face_pca", "ext_a": float(ext_a), "ext_n": float(ext_n), "yaw_mod_pi": yaw}


def _pieces(mask, cfg, W):
    """Red connected components (after a small dilation), largest first: [(u, v, bbox)] of their own red pixels."""
    r = max(1, int(round(cfg.join_px * W / 720.0)))
    joined = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    out = []
    for i in [i for i in np.argsort(-stats[:, cv2.CC_STAT_AREA]) if i > 0][:cfg.max_candidates]:
        x0, y0, w, h = (int(v) for v in stats[i, :4])
        ys, xs = np.nonzero((lab[y0:y0 + h, x0:x0 + w] == i) & (mask[y0:y0 + h, x0:x0 + w] > 0))
        if len(xs) >= cfg.piece_min_px:
            out.append((xs + x0, ys + y0, (x0, y0, w, h)))
    return out


def _near_layer_points(us, vs, dep, cam, cfg):
    z = dep[vs, us]
    good = z > 0
    if good.sum() < 2:
        return None, None
    zs = z[good]
    zref = float(np.percentile(zs, 15))
    keep = zs <= zref + cfg.near_layer
    return cam.backproject(us[good][keep], vs[good][keep], zs[keep]), zref


def _group(pieces, dep, cam, cfg):
    """Merge pieces of one radio (a face split by the white speaker ring, the handle bar above the white top band):
    single linkage on each piece's representative 3-D point (depth: median of its near-layer points; no depth: its
    centroid ray on the plane z = RADIO_CENTRE_Z)."""
    reps = []
    for us, vs, _ in pieces:
        if dep is not None:
            Q, _ = _near_layer_points(us, vs, dep, cam, cfg)
            reps.append(None if Q is None or not len(Q) else np.median(Q, 0))
        else:
            P, _, ok = cam.plane_hit(np.array([us.mean()]), np.array([vs.mean()]), RADIO_CENTRE_Z)
            reps.append(P[0] if ok[0] else None)
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    link = cfg.group_dist if dep is not None else cfg.group_dist_nodepth
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if reps[i] is not None and reps[j] is not None and np.linalg.norm(reps[i] - reps[j]) < link:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(pieces)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        us = np.concatenate([pieces[i][0] for i in idx])
        vs = np.concatenate([pieces[i][1] for i in idx])
        x0 = min(pieces[i][2][0] for i in idx)
        y0 = min(pieces[i][2][1] for i in idx)
        x1 = max(pieces[i][2][0] + pieces[i][2][2] for i in idx)
        y1 = max(pieces[i][2][1] + pieces[i][2][3] for i in idx)
        out.append((us, vs, (x0, y0, x1 - x0, y1 - y0), len(idx)))
    return sorted(out, key=lambda g: -len(g[0]))


def _detect(img, dep, cam, cfg, held, state):
    H, W = img.shape[:2]
    s = W / 720.0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = red_mask(hsv, cfg)
    min_px = max(3, int(round(cfg.min_px * s * s)))
    dets = []
    for us, vs, (x0, y0, w, h), n_pieces in _group(_pieces(mask, cfg, W), dep, cam, cfg):
        area = len(us)
        if area < min_px:
            continue
        uc, vc = float(us.mean()), float(vs.mean())
        mean_s = float(hsv[vs, us, 1].mean())
        clipped = bool(x0 <= 1 or y0 <= 1 or x0 + w >= W - 2 or y0 + h >= H - 2)
        info = {"px": int(area), "uv": (uc, vc), "bbox": (x0, y0, w, h), "mean_s": mean_s, "clipped": clipped,
                "pieces": int(n_pieces)}
        if mean_s < cfg.blob_s_min:
            _trace("blob_colour", **info)
            continue
        side_clipped = bool(x0 <= 1 or x0 + w >= W - 2)
        if dep is not None:
            Q, zref = _near_layer_points(us, vs, dep, cam, cfg)
            if Q is None or len(Q) < cfg.min_pts:
                _trace("no_depth", **info)
                continue
            z05, z95 = (float(v) for v in np.percentile(Q[:, 2], [5, 95]))
            lo, hi = np.percentile(Q[:, :2], [5, 95], axis=0)
            ext_xy = float(np.hypot(*(hi - lo)))
            info.update(zref=zref, z05=z05, z95=z95, ext_xy=ext_xy, n_pts=int(len(Q)))
            centre_xy, cinfo = _centre_from_points(Q, cam.t, cfg)
            info.update(cinfo)
            centre = np.array([centre_xy[0], centre_xy[1], float(np.clip(z95 - RADIO_SIZE[2] / 2 + 0.005, 0.0, 2.0))])
            # held test before the table-height gates (a lifted radio in the hand is still reported as held)
            d_near = min((float(np.min(np.linalg.norm(Q - e, axis=1))) for e in held), default=np.inf)
            d_centre = min((float(np.linalg.norm(centre - e)) for e in held), default=np.inf)
            is_held = d_near < cfg.held_radius or d_centre < cfg.held_centre_radius
            info["held_dist"] = min(d_near, 99.0)
            if ext_xy > cfg.xy_extent_max or (z95 - z05) > cfg.z_extent_max:
                _trace("size", **info)
                continue
            if side_clipped and info.get("ext_a", 0.0) < cfg.side_clip_min_ext:
                _trace("side_clipped", **info)                         # truncated partial view while turning
                continue
            if not is_held and not (cfg.z_bottom[0] <= z05 <= cfg.z_bottom[1] and cfg.z_top[0] <= z95 <= cfg.z_top[1]):
                _trace("height", **info)
                continue
            if is_held and z95 > 1.6:
                _trace("height", **info)
                continue
        else:
            P, lam, ok = cam.plane_hit(np.array([uc]), np.array([vc]), RADIO_CENTRE_Z)
            if not ok[0]:
                _trace("above_horizon", **info)
                continue
            zref = float(lam[0])
            view = P[0, :2] - cam.t[:2]
            view /= max(np.linalg.norm(view), 1e-6)
            centre = np.array([P[0, 0] + cfg.far_push * view[0], P[0, 1] + cfg.far_push * view[1], RADIO_CENTRE_Z])
            wm, hm = w * zref / cam.f, h * zref / cam.f
            info.update(zref=zref, size_m=(float(wm), float(hm)), method="plane")
            if max(wm, hm) < cfg.size_min or wm > cfg.size_max[0] or hm > cfg.size_max[1]:
                _trace("size", **info)
                continue
            is_held = any(_seg_dist(e, cam.t, centre) < cfg.held_ray_dist for e in held)
            if side_clipped and not is_held:
                _trace("side_clipped", **info)                         # no extent evidence without depth
                continue
        rng = float(np.hypot(centre[0], centre[1]))
        info["range"] = rng
        if rng > cfg.max_range:
            _trace("range", **info)
            continue
        if is_held and not cfg.return_held:
            _trace("held", **info)
            continue
        score = float((1.0 - np.exp(-area * (720.0 / W) ** 2 / 40.0)) * (0.6 if clipped else 1.0))
        _trace(None, **info)
        grasp = centre + np.array([0.0, 0.0, GRASP_DZ])
        dets.append({"label": "radio", "xyz_base": grasp, "score": score, "center_base": centre, "held": bool(is_held),
                     **{k: v for k, v in info.items() if k not in ("bbox",)}})
    return dets


def _nms(dets, radius):
    out = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(np.linalg.norm(o["xyz_base"][:2] - d["xyz_base"][:2]) > radius for o in out):
            out.append(d)
    return out


def make_radio_visible_fn(cam, depth, cfg=DEFAULT):
    """object_perception.make_visible_fn with 2-D query points (ObjectMap tracks) lifted to the radio centre height."""
    base = make_visible_fn(cam, depth, cfg)

    def visible(p_base):
        p = np.asarray(p_base, np.float64).reshape(-1)
        if p.size < 3:
            p = np.array([p[0], p[1], cfg.vis_z])
        return base(p)

    return visible


def detect_from_observation(rgb, depth=None, cam_rel_pose=None, state=None, **overrides):
    """Head RGB(-D) + head camera pose (+ proprio) -> (detections, visible_fn).

    detections: [{"label": "radio", "xyz_base": (3,) grasp point (base frame), "score": float, "center_base": (3,),
    "held": bool, ...diagnostics}]; held radios are suppressed unless return_held=True. visible_fn is None when the
    camera pose is missing or implausible (then no detections either)."""
    cfg = replace(DEFAULT, **overrides) if overrides else DEFAULT
    img = _rgb(rgb)
    H, W = img.shape[:2]
    cp = camera_from_pose(head_pose(cam_rel_pose))
    if cp is None:
        return [], None
    cam = Camera(cp[0], cp[1], W, H)
    dep = _depth(depth, (H, W))
    held = _held_eefs(state, cfg)
    st = None if state is None else np.asarray(state, np.float64).reshape(-1)
    dets = _detect(img, dep, cam, cfg, held, st if st is not None and st.size >= 61 else None)
    return _nms(dets, cfg.nms_radius), make_radio_visible_fn(cam, dep, cfg)


# ---- CPU self-test on synthetic data ------------------------------------------------------------------------------------
def _synthetic_scene(W=720, radio=(2.2, 0.4, 0.5), extra_red=("picture", "rug")):
    """Wood floor, a grey coffee table (top 0.42 m) with the radio (red faces, white band, red handle bar) standing on it
    at base xy radio[:2] with yaw radio[2]; distractors: a red picture on the wall at 1.4 m height, a red rug patch on the
    floor. Head camera at 1.39 m pitched down 30 deg. Rendered by ray casting (numpy)."""
    pitch = np.radians(30.0)
    fwd, up = np.array([np.cos(pitch), 0, -np.sin(pitch)]), np.array([np.sin(pitch), 0, np.cos(pitch)])
    Rusd = np.stack([np.cross(up, -fwd), up, -fwd], 1)
    qw = np.sqrt(max(1 + np.trace(Rusd), 1e-12)) / 2
    q = np.array([(Rusd[2, 1] - Rusd[1, 2]) / (4 * qw), (Rusd[0, 2] - Rusd[2, 0]) / (4 * qw),
                  (Rusd[1, 0] - Rusd[0, 1]) / (4 * qw), qw])
    pose = np.r_[0.24, 0.0, 1.39, q]
    R, t = camera_from_pose(pose)
    cam = Camera(R, t, W, W)
    vv, uu = np.mgrid[0:W, 0:W]
    d = cam.rays(uu + 0.0, vv + 0.0)
    depth = np.full((W, W), np.inf)
    rgb = np.zeros((W, W, 3), np.uint8)
    rgb[:] = (235, 235, 235)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(d[..., 2] < 0, -t[2] / d[..., 2], np.inf)
    floor = np.isfinite(lam)
    depth[floor] = lam[floor]
    rgb[floor] = (205, 170, 135)

    def box(c, half, yaw, colour_fn):
        """Axis-aligned-in-yaw box (slab method); colour_fn(local hit point, face axis) -> RGB."""
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        o = (t - c) @ Rz
        dl = d @ Rz
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / dl, (half - o) / dl
        tn = np.minimum(t1, t2)
        tf = np.maximum(t1, t2)
        tnear, tfar = tn.max(-1), tf.min(-1)
        hit = (tnear < tfar) & (tnear > 0.05) & (tnear < depth)
        axis = tn.argmax(-1)
        pl = o + dl * tnear[..., None]
        depth[hit] = tnear[hit]
        rgb[hit] = colour_fn(pl[hit], axis[hit])

    RED, WHITE = np.array([225, 25, 20]), np.array([240, 240, 240])

    def radio_colour(pl, axis):
        out = np.tile(WHITE, (len(pl), 1))
        out[axis == 0] = RED                                            # front / back faces (x = +-0.069)
        bar = (axis == 2) & (np.abs(pl[:, 0]) < 0.02) & (np.abs(pl[:, 1]) < 0.11)
        out[bar] = RED                                                  # handle bar on top
        ring = (axis == 0) & (np.hypot(pl[:, 1] - 0.06, pl[:, 2] + 0.01) < 0.05)
        out[ring] = WHITE                                               # speaker ring on the faces
        return out

    box(np.array([2.3, 0.2, 0.21]), np.array([0.40, 0.80, 0.21]), 0.0, lambda p, a: np.tile([120, 118, 110], (len(p), 1)))
    box(np.array([radio[0], radio[1], TABLE_TOP_Z + RADIO_SIZE[2] / 2]), RADIO_SIZE / 2, radio[2], radio_colour)
    if "picture" in extra_red:
        box(np.array([4.5, -1.2, 1.4]), np.array([0.02, 0.35, 0.25]), 0.0, lambda p, a: np.tile(RED, (len(p), 1)))
    if "rug" in extra_red:
        box(np.array([1.6, -1.0, 0.002]), np.array([0.3, 0.2, 0.002]), 0.3, lambda p, a: np.tile(RED, (len(p), 1)))
    cam21 = np.r_[np.zeros(14), pose]
    depth = np.where(np.isfinite(depth), depth, np.inf).astype(np.float32)
    return rgb, depth, cam21


def _self_test():
    import time
    state = np.zeros(61)
    state[24:26] = state[49:51] = 0.05                                      # grippers open (sum 0.10)
    state[17:20], state[42:45] = (0.3, 0.25, 0.5), (0.3, -0.25, 0.5)
    worst = 0.0
    for radio in ((2.2, 0.4, 0.5), (1.3, -0.3, 1.4), (3.5, 0.9, 0.0), (1.1, 0.1, 2.6)):
        truth = np.array([radio[0], radio[1], TABLE_TOP_Z + RADIO_SIZE[2] / 2])
        for with_depth in (True, False):
            # the depth-free fallback cannot tell a red floor rug from a radio (no height evidence): rug only with depth
            rgb, depth, cam21 = _synthetic_scene(radio=radio, extra_red=("picture", "rug") if with_depth else ("picture",))
            dets, vis = detect_from_observation(rgb, depth if with_depth else None, cam21, state)
            assert len(dets) == 1, (radio, with_depth, [(d["xyz_base"].round(2), d.get("method")) for d in dets])
            d = dets[0]
            err = float(np.linalg.norm(d["center_base"][:2] - truth[:2]))
            worst = max(worst, err) if with_depth else worst
            # depth-free: the ray through the red centroid meets z = RADIO_CENTRE_Z; seen end-on only the handle bar on top
            # is red, which overshoots by ~0.2 m at 1.3 m range (fallback accuracy, the deployment has depth)
            assert err < (0.05 if with_depth else 0.25), (radio, with_depth, err, d.get("method"))
            assert np.allclose(d["xyz_base"], d["center_base"] + [0, 0, GRASP_DZ])
            assert vis(truth[:2]) and vis(d["xyz_base"]) and not vis(np.array([-1.0, 0.0]))
            print(f"radio at {radio}: depth={with_depth} centre {d['center_base'].round(3)} err {err:.3f} m "
                  f"({d.get('method')}, {d['px']} px)")
    # occlusion: a wall 1 m in front of the camera hides the radio
    rgb, depth, cam21 = _synthetic_scene()
    _, vis = detect_from_observation(rgb, np.full_like(depth, 1.0), cam21, state)
    assert not vis(np.array([2.2, 0.4]))
    # implausible (post-reset) camera pose -> no detections, no visibility predicate
    bad = cam21.copy()
    bad[14:17] = (2.75, -3.83, 1.39)
    assert detect_from_observation(rgb, depth, bad, state) == ([], None)
    # held: closed right gripper at the handle -> suppressed, or flagged with return_held
    rgb, depth, cam21 = _synthetic_scene(extra_red=("picture",))
    s2 = state.copy()
    s2[49:51] = 0.02
    s2[42:45] = (2.2, 0.4, TABLE_TOP_Z + RADIO_SIZE[2] + 0.01)
    for with_depth in (True, False):
        assert detect_from_observation(rgb, depth if with_depth else None, cam21, s2)[0] == [], with_depth
        dh, _ = detect_from_observation(rgb, depth if with_depth else None, cam21, s2, return_held=True)
        assert len(dh) == 1 and dh[0]["held"], with_depth
    # an open hand at the same place does not suppress it
    s3 = s2.copy()
    s3[49:51] = 0.05
    assert len(detect_from_observation(rgb, depth, cam21, s3)[0]) == 1
    # without the radio: the red wall picture and the red floor rug are rejected with depth
    rgb0, depth0, _ = _synthetic_scene(radio=(30.0, 0.0, 0.0))
    assert detect_from_observation(rgb0, depth0, cam21, state)[0] == []
    obs = obstacle_points(depth, cam21, state)
    assert obs.ndim == 2 and obs.shape[1] == 2 and np.any(np.linalg.norm(obs - [2.2, 0.4], axis=1) < 0.25)
    t0 = time.perf_counter()
    for _ in range(20):
        detect_from_observation(rgb, depth, cam21, state)
    print(f"latency detect: {(time.perf_counter() - t0) / 20 * 1e3:.1f} ms per call (720x720, CPU); "
          f"worst depth-mode centre error {worst:.3f} m")
    print("PASS radio_perception self-test")


if __name__ == "__main__":
    _self_test()
