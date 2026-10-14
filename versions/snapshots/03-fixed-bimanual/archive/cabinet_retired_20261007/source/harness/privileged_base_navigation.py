"""Opt-in Task90 FIRST-bin privbase diagnostic; never an autonomous score.

The only live truth is base/root-link pose + the named ashcan pose/scale. It is
kept in a separate docking controller/map, never the WAM or predicate map.
The near-side rim is a STATIC MODEL approximation, not a validated grasp point.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from harness.docking_controller import BASE_SCALE, DockingController, ObjectMap, Odometry, Track, rot, wrap

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = "harness.privileged_base_rgbd_wrapper.PrivilegedBaseRGBDWrapper"
PREPOSITION_WRAPPER = "harness.prepositioned_base_rgbd_wrapper.PrepositionedBaseRGBDWrapper"
WRAPPERS = (WRAPPER, PREPOSITION_WRAPPER)
OPTION = "privileged_base_navigation"
GOAL = "ashcan.n.01_1"
MODEL = "ifzxzj"
KEY = f"privbase_nav::{GOAL}::{MODEL}"
ASSET = ROOT / f"data/omnigibson/behavior-1k-assets/objects/trash_can/{MODEL}/misc/metadata.json"
PROTOCOL = {
    "name": "privbase-first-bin-v1", "wrapper": WRAPPER, "key": KEY,
    "packet": "float64[22]: version, task_id, reset_id, step, sim_time, base_xyz_xyzw[7], named_root_xyz_xyzw[7], scale[3]",
    "scope": "first move to(trash_can) only; later navigation fails closed",
    "control": "physical base velocities, existing depth avoidance and safety envelope; no teleport/IK",
    "upper_body": "fixed TRAIN pick-start joint posture during navigation, independent of live GT; WAM after handoff",
    "reference": "TRAIN EEF closure transported to pick start; NOT object center",
    "offset": "nearest-side midpoint of static base_link visual bbox rim, selected once at reset and latched in object frame",
    "limitations": "rim-to-EEF offset is an unvalidated geometric approximation; no contact/orientation fit or dev coordinates",
    "wam_gt_input": False, "predicate_gt_input": False, "autonomous": False, "official_score_valid": False,
}


def validate_config(cfg, task_id):
    selected = (cfg or {}).get(OPTION)
    if selected is None:
        return False
    if selected not in WRAPPERS or task_id != 90:
        raise ValueError("privbase requires Task90 and the exact privileged wrapper opt-in")
    if (cfg.get("navigation") != "docking" or cfg.get("manip_executor", "wam") != "wam"
            or cfg.get("postconditions") != "task090_observed" or cfg.get("dock_target_at", "start") != "start"
            or cfg.get("candidate_outcome_policy", "hold") != "hold" or cfg.get("vlm", "none") != "none"):
        raise ValueError("privbase requires docking/start + WAM + task090_observed/hold + fixed program")
    return True


def validate_launch(cfg, wrapper, run_name, profile, task_id):
    enabled = validate_config(cfg, task_id)
    if enabled != (wrapper in WRAPPERS) or (enabled and cfg.get(OPTION) != wrapper):
        raise ValueError("privbase wrapper and orchestrator opt-in must both be selected")
    if enabled and ("privbase" not in run_name or profile == "report"):
        raise ValueError("privbase requires a diagnostic run name containing privbase, not report")
    if enabled and wrapper == PREPOSITION_WRAPPER and "preposition" not in run_name:
        raise ValueError("prepositioned setup requires an explicit preposition run label")
    return enabled


def _vector(value, n):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"privbase requires finite shape ({n},)")
    return a


def pose_matrix(pose):
    pose = _vector(pose, 7)
    q = pose[3:]
    if abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError("privbase quaternion must be unit xyzw")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class _FixedPostureDocking(DockingController):
    def _posture_target(self, dist):
        # No GT distance-dependent manipulation targets or IK, even during docking.
        return self.calib["posture"]["pick_start_trash_can"]


class PrivbaseNavigator:
    def __init__(self, calibration, config):
        if not config.avoid_obstacles or not config.safety_filter or config.height_bands:
            raise ValueError("privbase v1 requires the existing 2-D depth avoidance and safety filter")
        self.controller = _FixedPostureDocking(calibration, config)
        box = json.loads(ASSET.read_text())["link_bounding_boxes"]["base_link"]["visual"]["axis_aligned"]
        self.center = np.asarray(box["transform"], float)[:3, 3]
        self.extent = np.asarray(box["extent"], float)
        self.odom, self.omap = Odometry(), ObjectMap(min_hits=3, max_age_steps=1)
        self._previous_epoch = None
        self.reset()

    def reset(self):
        self._previous_epoch = getattr(self, "epoch", self._previous_epoch)
        self.epoch = self.last_step = self.last_time = self.offset = self.scale = None
        self.binding = None
        self.blocked = self.finished = False
        self.odom.reset()
        self.omap.reset()
        self.controller.reset()

    def invalidate(self):
        self.blocked = True
        self.omap.reset()
        self.controller.last_status = {}

    def prepare(self, packet, step):
        try:
            if self.blocked or self.finished:
                raise ValueError("privbase navigation is closed until episode reset")
            a = _vector(packet, 22)
            if (a[0] != 1 or a[1] != 90 or a[2] < 0 or a[2] != int(a[2])
                    or a[3] != step or a[4] < 0):
                raise ValueError("privbase missing/wrong protocol, task, reset or observation step")
            if self.epoch is None:
                if step != 0 or (self._previous_epoch is not None and a[2] <= self._previous_epoch):
                    raise ValueError("privbase must begin at a fresh normal reset")
            elif a[2] != self.epoch or step != self.last_step + 1 or a[4] <= self.last_time:
                raise ValueError("privbase stale/cross-episode packet")
            base, target, scale = a[5:12], a[12:19], a[19:22]
            rb, rt = pose_matrix(base), pose_matrix(target)
            if rb[2, 2] < .98 or rt[2, 2] < .98 or (scale <= 0).any() or (scale > 10).any():
                raise ValueError("privbase requires upright base/bin and positive model scale")
            if self.scale is not None and not np.allclose(self.scale, scale, atol=1e-6, rtol=0):
                raise ValueError("privbase target geometry changed")
            if self.offset is None:
                # Object-root frame, not whole-object center (lid included). Never a world-coordinate constant.
                center, half = self.center * scale, self.extent * scale / 2
                toward = rt.T @ (base[:3] - target[:3]) - center
                axis = int(np.argmax(np.abs(toward[:2]) / half[:2]))
                self.offset = center.copy()
                self.offset[axis] += math.copysign(half[axis], toward[axis])
                self.offset[2] += half[2]
            rim = target[:3] + rt @ self.offset
            self.odom.pose = np.array([base[0], base[1], math.atan2(rb[1, 0], rb[0, 0])])
            self.omap.advance(step)
            self.omap.tracks = [Track("trash_can", rim[:2], hits=3, last_seen=step, track_id=1, z=float(rim[2]-base[2]))]
            self.epoch, self.last_step, self.last_time, self.scale = int(a[2]), step, a[4], scale.copy()
        except (ValueError, TypeError, OverflowError):
            self.invalidate()
            raise

    def _bind(self, contract, index):
        if (self.blocked or self.finished or index != 0 or contract is None or not contract.is_navigation
                or tuple(contract.step.objects) != ("trash_can",) or contract.key != "move to(trash_can)"):
            self.invalidate()
            raise ValueError("privbase supports ONLY first move to(trash_can), bound to ashcan.n.01_1")
        binding = (index, contract.key, contract.started_step)
        if self.binding is not None and self.binding != binding:
            self.invalidate()
            raise ValueError("privbase goal/contract rebinding refused")
        return binding

    def start(self, contract, index):
        self.binding = self._bind(contract, index)
        self.controller.start("trash_can", "before_bin", self.odom)

    def evidence(self, contract, index):
        self._bind(contract, index)
        st = self.controller.last_status
        p = self.odom.to_base(self.omap.tracks[0].pos)
        tgt = self.controller.calib["dock"]["trash_can"]["target"]
        error = float(np.linalg.norm(p - tgt))
        bearing = abs(wrap(math.atan2(p[1], p[0]) - math.atan2(tgt[1], tgt[0])))
        return {"docked": bool(st.get("docked") and error <= self.controller.cfg.tol_xy
                                and math.degrees(bearing) <= self.controller.cfg.tol_bearing_deg),
                "key": contract.key, "observation_id": self.last_step, "error_norm": error,
                "dock_target": np.asarray(tgt).tolist(), "mode": st.get("mode"), "source": PROTOCOL["name"]}

    def action(self, state, obs, perception, contract, index):
        self._bind(contract, index)
        # Missing RGB-D must not silently turn obstacle avoidance into an empty cloud.
        depth = next((v for k, v in obs.items() if k.endswith(":zed_link:Camera:0::depth_linear")), None)
        cam = obs.get("robot_r1::cam_rel_poses")
        if depth is None or cam is None or not np.isfinite(cam).all() or not np.any(np.isfinite(depth) & (np.asarray(depth) > 0)):
            self.invalidate()
            raise ValueError("privbase requires live depth and camera-relative pose for safety")
        obstacles = perception.obstacles(obs, state)
        if not np.isfinite(obstacles).all():
            self.invalidate()
            raise ValueError("privbase invalid depth obstacles")
        action, status = self.controller.step(state, self.omap, self.odom, obstacles=obstacles)
        if status.get("recovery") in ("backoff", "stuck_detected"):
            # Legacy recovery skips the shield; privileged diagnosis must not bypass it.
            safe, info = self.controller._safe(action[:3] * BASE_SCALE, self.omap, self.odom, 1,
                                               obstacles=obstacles, target_p=self.odom.to_base(self.omap.tracks[0].pos))
            action[:3] = safe / BASE_SCALE
            self.controller._last_sent = safe.copy()
            status["privbase_recovery_safety"] = info
        status.update(source=PROTOCOL["name"], goal=GOAL, grasp_reference="static_near_rim_approximation",
                      root_to_rim_local_m=self.offset.tolist())
        return action, status

    def finish_navigation(self):
        self.finished = True
        self.omap.reset()
        self.odom.reset()
        self.offset = self.scale = None
        self.controller.reset()
