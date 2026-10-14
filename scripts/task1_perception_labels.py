"""Offline label geometry for Task 1 perception development (NOT used by the policy at run time).

Projects the ground truth written by scripts/build_task1_object_gt.py into a camera image and decides whether each
floor object is usable as a positive label: inside the frustum, low odometry-label uncertainty, not behind a wall
(2-D ray cast on the scene's floor_trav_no_obj map), and no furniture footprint under the low part of the viewing
ray (floor_trav map). Cans that fail the furniture test but show their colour inside the projected box are
re-admitted ("colour verified"). Everything uncertain becomes an IGNORE region for training and evaluation.
"""
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "outputs/validation/task1_perception_20260926/gt"
LAYOUT = ROOT / "data/omnigibson/behavior-1k-assets/scenes/house_double_floor_lower/layout"
HALF = {"wkxtxh": (0.115, 0.139), "itolcg": (0.0376, 0.0673), "lugwcz": (0.0381, 0.0635)}
CLASS_OF_MODEL = {"wkxtxh": 0, "itolcg": 1, "lugwcz": 2}   # 0 trash_can, 1 orange can, 2 blue can
FLIP = np.diag([1.0, -1.0, -1.0])
MAP_RES = 0.01


def quat2mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cam_T(pose7):
    T = np.eye(4)
    T[:3, :3] = quat2mat(pose7[3:7]) @ FLIP
    T[:3, 3] = pose7[:3]
    return T


def intrinsics(width):
    f = 306.0 * width / 720.0
    return np.array([[f, 0, width / 2.0], [0, f, width / 2.0], [0, 0, 1.0]])


class SceneMaps:
    def __init__(self):
        self.walls = cv2.imread(str(LAYOUT / "floor_trav_no_obj_0.png"), 0) == 0
        trav = cv2.imread(str(LAYOUT / "floor_trav_0.png"), 0) == 0
        self.furniture = trav & ~self.walls
        self.size = self.walls.shape[0]

    def rc(self, x, y):
        return np.round(y / MAP_RES + self.size / 2).astype(int), np.round(x / MAP_RES + self.size / 2).astype(int)

    def ray(self, a, b, grid, skip_start=0.0, skip_end=0.0, frac_range=(0.0, 1.0)):
        """True if any cell of `grid` lies on the world segment a->b (xy), within the fraction range."""
        d = np.hypot(b[0] - a[0], b[1] - a[1])
        n = max(int(d / 0.02), 2)
        f = np.linspace(0, 1, n)
        keep = (f * d >= skip_start) & ((1 - f) * d >= skip_end) & (f >= frac_range[0]) & (f <= frac_range[1])
        if not keep.any():
            return False
        f = f[keep]
        r, c = self.rc(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        ok = (r >= 0) & (r < self.size) & (c >= 0) & (c < self.size)
        return bool(grid[r[ok], c[ok]].any())


def project_points(P_base, pose7, K):
    Tinv = np.linalg.inv(cam_T(pose7))
    pc = np.c_[P_base, np.ones(len(P_base))] @ Tinv.T
    z = pc[:, 2]
    uv = pc[:, :3] @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return uv[:, :2] / uv[:, 2:3], z


def cylinder_box(center, half, pose7, K):
    r, h = half
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts = np.concatenate([np.c_[center[0] + r * np.cos(a), center[1] + r * np.sin(a), np.full(16, center[2] + dz)]
                          for dz in (-h, h)])
    uv, z = project_points(pts, pose7, K)
    if (z < 0.05).any():
        return None
    return float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())


ORANGE = ((4, 200, 90), (20, 255, 255))
BLUE = ((104, 110, 25), (130, 255, 210))


def colour_count(rgb, box, cls):
    if cls == 0:
        return 0
    h, w = rgb.shape[:2]
    u0, v0, u1, v1 = [int(round(x)) for x in box]
    u0, v0, u1, v1 = max(u0 - 2, 0), max(v0 - 2, 0), min(u1 + 3, w), min(v1 + 3, h)
    if u1 <= u0 or v1 <= v0:
        return 0
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[v0:v1, u0:u1]), cv2.COLOR_RGB2HSV)
    lo, hi = ORANGE if cls == 1 else BLUE
    return int(cv2.inRange(hsv, lo, hi).astype(bool).sum())


class EpisodeLabels:
    """Per-frame object labels for one demo episode (head camera)."""

    def __init__(self, episode, maps=None):
        d = np.load(GT_DIR / f"episode_{episode}.npz")
        self.pos = d["pos_base"].astype(np.float64)
        self.state = d["state"]
        self.rot_ref = d["rot_to_ref"]
        self.trav_ref = d["trav_to_ref"]
        self.cam = d["cam_zed"].astype(np.float64)
        self.xyth = d["robot_xyth"].astype(np.float64)
        self.models = [str(m) for m in d["models"]]
        self.world = d["world"].astype(np.float64)
        self.eef = {"left": d["eef_left"].astype(np.float64), "right": d["eef_right"].astype(np.float64)}
        self.maps = maps or SceneMaps()
        self.length = len(self.state)

    def objects(self, t, width=720, rgb=None, max_rot=1.2, margin=0):
        """List of dicts for floor objects at frame t (all, with flags)."""
        K = intrinsics(width)
        out = []
        x, y, th = self.xyth[t]
        cpos = self.cam[t, :3]
        cam_w = np.array([x + math.cos(th) * cpos[0] - math.sin(th) * cpos[1],
                          y + math.sin(th) * cpos[0] + math.cos(th) * cpos[1]])
        for j, m in enumerate(self.models):
            if self.state[t, j] != 0:
                continue
            c = self.pos[t, j]
            cls = CLASS_OF_MODEL[m]
            box = cylinder_box(c, HALF[m], self.cam[t], K)
            item = {"j": j, "cls": cls, "model": m, "xyz": c.copy(), "box": box,
                    "rot_ref": float(self.rot_ref[t, j]), "trav_ref": float(self.trav_ref[t, j])}
            uv, z = project_points(c[None], self.cam[t], K)
            item["uv"] = uv[0] if z[0] > 0.05 else None
            item["cam_dist"] = float(np.linalg.norm((np.linalg.inv(cam_T(self.cam[t])) @ np.append(c, 1))[:3]))
            item["dist"] = float(np.hypot(c[0], c[1]))
            if box is None:
                item["in_frustum"] = False
            else:
                u0, v0, u1, v1 = box
                cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
                item["in_frustum"] = bool(margin <= cu < width - margin and margin <= cv < width - margin)
                item["box_clipped_frac"] = float(max(0, min(u1, width) - max(u0, 0)) * max(0, min(v1, width) - max(v0, 0))
                                                 / max((u1 - u0) * (v1 - v0), 1e-6))
            ow = self.world[j][:2]
            item["wall"] = self.maps.ray(cam_w, ow, self.maps.walls, skip_start=0.25, skip_end=0.18)
            r = HALF[m][0]
            # furniture footprint under the part of the viewing ray lower than ~0.9 m
            h_cam = cpos[2]
            top = c[2] + HALF[m][1]
            f_hi = min(1.0, max(0.0, (0.9 - top) / max(h_cam - top, 1e-3)))
            item["furniture"] = self.maps.ray(ow, cam_w, self.maps.furniture, skip_start=r + 0.03, frac_range=(0.0, f_hi))
            item["good_label"] = item["rot_ref"] < max_rot
            item["colour_px"] = colour_count(rgb, box, cls) if (rgb is not None and box is not None and item["in_frustum"]) else 0
            area = 0 if box is None else max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
            item["colour_verified"] = bool(cls > 0 and item["colour_px"] >= max(3, 0.08 * area * (width / 720.0) ** 0))
            clear = item["in_frustum"] and not item["wall"] and not item["furniture"]
            item["visible"] = bool(item["good_label"] and item["in_frustum"] and not item["wall"]
                                   and (not item["furniture"] or item["colour_verified"]))
            item["clear_los"] = bool(clear)
            out.append(item)
        return out

    def held_objects(self, t):
        """Base-frame positions of held / in-bin objects (to test rejection)."""
        return [(j, int(self.state[t, j]), self.pos[t, j]) for j in range(len(self.models)) if self.state[t, j] in (1, 2)]
