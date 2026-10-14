"""Recording-only camera geometry; no observations from this module go to a policy."""
import numpy as np


def validate_viewer_config(width, height, offset, target):
    if type(width) is not int or type(height) is not int or width < 1280 or height < 720 or width % 2 or height % 2:
        raise ValueError("Viewer video requires even native dimensions >= 1280 x 720")
    offset, target = np.asarray(offset, dtype=float), np.asarray(target, dtype=float)
    if offset.shape != (3,) or target.shape != (3,) or not np.isfinite([offset, target]).all():
        raise ValueError("Video camera offset/target must each contain three finite metres")
    if np.linalg.norm((target - offset)[:2]) < 1e-6:
        raise ValueError("Video camera must look horizontally as well as vertically")


def viewer_pose(base_position, base_rotation, offset, target):
    """Robot-yaw-relative camera; USD cameras look along local -Z with +Y up."""
    yaw = np.arctan2(base_rotation[1, 0], base_rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    position = np.asarray(base_position) + rotation @ np.asarray(offset)
    aim = np.asarray(base_position) + rotation @ np.asarray(target)
    backward = position - aim
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    return position, np.column_stack((right, up, backward))


def native_rgb_frame(rgb, width, height):
    """Fail on stale/low-resolution render buffers; never upscale to claim HD."""
    frame = np.asarray(rgb)
    if frame.dtype != np.uint8 or frame.shape not in ((height, width, 3), (height, width, 4)):
        raise ValueError(f"Expected native {width} x {height} uint8 RGB(A), got {frame.shape} {frame.dtype}")
    return np.ascontiguousarray(frame[..., :3])


def look_at_rotation(position, aim):
    """USD camera rotation (columns right, up, backward) looking from @position at @aim."""
    backward = np.asarray(position, dtype=float) - np.asarray(aim, dtype=float)
    backward /= np.linalg.norm(backward)
    right = np.cross([0.0, 0.0, 1.0], backward)
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(backward, right), backward))


def in_frustum(points, position, rotation, tan_half_h, tan_half_v, margin=0.92, near=0.3):
    """True where world @points project inside the image (square pixels, NDC |x|,|y| <= margin)."""
    cam = (np.atleast_2d(points) - np.asarray(position)) @ np.asarray(rotation)  # rows: (x_right, y_up, z_backward)
    depth = -cam[:, 2]
    safe = np.maximum(depth, 1e-6)
    return (depth > near) & (np.abs(cam[:, 0] / safe) <= margin * tan_half_h) & (np.abs(cam[:, 1] / safe) <= margin * tan_half_v)


def _yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ObserverFraming:
    """Evaluator-only occlusion-aware observer camera for recordings; nothing here is a policy observation.

    Candidates are offsets on a bearing / radius / height grid around the configured offset, ranked by a fixed cost
    (deviation from the configured bearing, shrink, lowering), so selection is deterministic. A pose is valid when every
    required robot point has an unobstructed line of sight (``blocked(start, end, ignore)`` is an injected ray test; each
    ray starts ``clearance`` metres behind the camera so a camera inside/behind a wall or above the ceiling is rejected)
    and every frustum point projects into the image. Optional points (end effectors, nearby task objects such as the car
    rear) only add cost when hidden.

    The chosen offset is anchored to the robot heading at selection time (it translates with the robot but does not
    orbit when the robot turns in place); the aim point follows the live heading. The current pose is kept while valid;
    a re-selection happens when it becomes invalid, or at periodic checks after a hold time if a candidate beats the
    current pose (re-scored at the robot's present heading) by a margin. Pose changes are exponentially smoothed with a
    per-frame speed cap (a glide); the camera only cuts when the gliding pose itself would be occluded or cropped.
    """

    BEARING_STEP_DEG = 22.5
    RADIUS_SCALES = (1.0, 0.8, 0.62, 0.48)
    HEIGHT_SCALES = (1.0, 0.8)

    def __init__(self, offset, target, width, height, horizontal_aperture, focal_length, blocked,
                 smoothing=0.2, max_step=0.12, upgrade_period=45, hold_after_switch=90, switch_margin=0.5,
                 clearance=0.3, optional_weight=1.0):
        offset, self.target = np.asarray(offset, dtype=float), np.asarray(target, dtype=float)
        self.blocked, self.smoothing, self.max_step = blocked, float(smoothing), float(max_step)
        self.clearance, self.optional_weight = float(clearance), float(optional_weight)
        self.upgrade_period, self.hold_after_switch = int(upgrade_period), int(hold_after_switch)
        self.switch_margin = float(switch_margin)
        self.tan_half_h = 0.5 * horizontal_aperture / focal_length
        self.tan_half_v = self.tan_half_h * height / width
        self.radius0, self.bearing0, self.height0 = np.linalg.norm(offset[:2]), np.arctan2(offset[1], offset[0]), offset[2]
        step = np.deg2rad(self.BEARING_STEP_DEG)
        candidates = []
        for k in range(int(round(360 / self.BEARING_STEP_DEG))):
            dev = ((k + 1) // 2) * (1 if k % 2 else -1) * step  # 0, +, -, ..., +180: fixed tie order
            for rs in self.RADIUS_SCALES:
                for hs in self.HEIGHT_SCALES:
                    b = self.bearing0 + dev
                    o = np.array([self.radius0 * rs * np.cos(b), self.radius0 * rs * np.sin(b), self.height0 * hs])
                    candidates.append((self.cost(o), len(candidates), o))
        candidates.sort(key=lambda c: (c[0], c[1]))
        self.candidates = [(c, o) for c, _, o in candidates]  # index 0 == configured offset
        self.reset()

    def cost(self, offset_yaw_frame):
        """Preference of a robot-yaw-frame offset: bearing deviation (1 per 45 deg), shrink, lowering."""
        o = np.asarray(offset_yaw_frame, dtype=float)
        dev = abs((np.arctan2(o[1], o[0]) - self.bearing0 + np.pi) % (2 * np.pi) - np.pi)
        shrink = max(0.0, 1.0 - np.linalg.norm(o[:2]) / self.radius0)
        return float(dev / np.deg2rad(45.0) + 1.5 * shrink + (0.25 if o[2] < self.height0 - 1e-6 else 0.0))

    def reset(self):
        self.current, self.anchor_yaw, self.frame, self.last_switch = None, 0.0, 0, -10**9
        self._pos = self._aim = None
        self.stats = {"frames": 0, "selections": 0, "cuts": 0, "glides": 0, "frames_robot_unobstructed_in_frame": 0,
                      "frames_fallback_best_effort": 0, "frames_by_candidate": {}}

    def _ray_clear(self, position, point, ignore):
        away = position - point
        away /= max(np.linalg.norm(away), 1e-9)
        return not self.blocked(position + self.clearance * away, point, ignore)

    def _evaluate(self, position, aim, subject):
        rotation = look_at_rotation(position, aim)
        n_required = sum(self._ray_clear(position, p, subject["robot_ignore"]) for p in subject["required"])
        framed = bool(in_frustum(subject["frustum"], position, rotation, self.tan_half_h, self.tan_half_v).all())
        missing = 0
        for point, ignore in subject.get("optional", ()):
            point = np.asarray(point, dtype=float)
            if not (in_frustum(point, position, rotation, self.tan_half_h, self.tan_half_v)[0]
                    and self._ray_clear(position, point, ignore)):
                missing += 1
        return {"valid": framed and n_required == len(subject["required"]), "required_clear": int(n_required),
                "framed": framed, "optional_missing": int(missing)}

    def _search(self, base, aim, yaw_rot, subject):
        best, fallback = None, None
        for index, (cost, offset) in enumerate(self.candidates):
            if best is not None and cost >= best[0]:
                break  # sorted costs and non-negative penalties: nothing later can win
            ev = self._evaluate(base + yaw_rot @ offset, aim, subject)
            if ev["valid"]:
                total = cost + self.optional_weight * ev["optional_missing"]
                if best is None or total < best[0]:
                    best = (total, index, ev)
            elif best is None:
                key = (-ev["required_clear"], not ev["framed"], cost)
                if fallback is None or key < fallback[0]:
                    fallback = (key, index, ev)
        return (best[1], best[2], best[0]) if best is not None else (fallback[1], fallback[2], np.inf)

    @staticmethod
    def _approach(current, goal, alpha, max_step):
        delta = alpha * (goal - current)
        norm = np.linalg.norm(delta)
        return current + (delta * (max_step / norm) if norm > max_step else delta)

    def update(self, base_position, base_rotation, subject):
        """subject: required (M,3) ray points, frustum (N,3) points, robot_ignore, optional [(point, ignore)]."""
        base = np.asarray(base_position, dtype=float)
        yaw = float(np.arctan2(base_rotation[1][0], base_rotation[0][0]))
        yaw_rot = _yaw_matrix(yaw)
        raw_aim = base + yaw_rot @ self.target
        selected = False
        if self.current is None:
            self.current, ev, _ = self._search(base, raw_aim, yaw_rot, subject)
            self.anchor_yaw, selected = yaw, True
        else:
            anchor_offset = _yaw_matrix(self.anchor_yaw) @ self.candidates[self.current][1]
            ev = self._evaluate(base + anchor_offset, raw_aim, subject)
            periodic = (self.frame % self.upgrade_period == 0 and self.frame - self.last_switch >= self.hold_after_switch)
            if not ev["valid"] or periodic:
                current_total = (self.cost(yaw_rot.T @ anchor_offset) + self.optional_weight * ev["optional_missing"]
                                 if ev["valid"] else np.inf)
                index, new_ev, total = self._search(base, raw_aim, yaw_rot, subject)
                if (new_ev["valid"] and total < current_total - (self.switch_margin if ev["valid"] else 0.0)) or (
                        not ev["valid"] and not new_ev["valid"] and new_ev["required_clear"] > ev["required_clear"]):
                    self.current, self.anchor_yaw, ev, selected = index, yaw, new_ev, True
        if selected:
            self.last_switch = self.frame
        raw_pos = base + _yaw_matrix(self.anchor_yaw) @ self.candidates[self.current][1]
        cut = self._pos is None
        if not cut:
            pos = self._approach(self._pos, raw_pos, self.smoothing, self.max_step)
            aim = self._approach(self._aim, raw_aim, self.smoothing, self.max_step)
            smooth_ev = self._evaluate(pos, aim, subject)
            if smooth_ev["valid"] or not ev["valid"]:
                ev = smooth_ev
            else:
                cut = True  # the gliding pose is occluded/cropped although the target pose is fine
        if cut:
            pos, aim = raw_pos, raw_aim
        gliding = not cut and np.linalg.norm(raw_pos - pos) > 0.25
        self._pos, self._aim = pos, aim
        s = self.stats
        s["frames"] += 1
        s["selections"] += int(selected and self.frame > 0)
        s["cuts"] += int(cut and self.frame > 0)
        s["glides"] += int(selected and self.frame > 0 and not cut)
        s["frames_robot_unobstructed_in_frame"] += int(ev["valid"])
        s["frames_fallback_best_effort"] += int(not ev["valid"])
        s["frames_by_candidate"][str(self.current)] = s["frames_by_candidate"].get(str(self.current), 0) + 1
        record = {"frame": self.frame, "candidate": self.current,
                  "offset_yaw_frame": self.candidates[self.current][1].round(4).tolist(),
                  "anchor_yaw": round(self.anchor_yaw, 4), "selected": bool(selected), "cut": bool(cut),
                  "gliding": bool(gliding), "position": pos.round(4).tolist(), "aim": aim.round(4).tolist(), **ev}
        self.frame += 1
        return pos, look_at_rotation(pos, aim), record
