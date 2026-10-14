"""Perception-driven navigation/docking for manipulation handoff (Task 1 `move to` contracts).

Why this exists (progress.md 2026-09-26 00:20): distant targets are a few pixels in the WAM's 224 head input, and the
Task 13 WAM navigation never docked reliably. `move to(X)` is therefore executed by this controller, and the WAM
takes over only at a docking pose calibrated on TRAIN demonstrations (object position in the robot base frame at
the start of the following pick, recovered from grasp kinematics; outputs/validation/task1_docking_calibration_20260926/).

Inputs are policy-visible only: object detections in the robot base frame (harness/object_perception.py: head RGB-D +
camera pose from proprioception), the measured local base velocity and joint positions (61-D proprio). The episode-
local odometry frame starts at the robot's reset pose; no simulator pose is ever read.

Pure numpy. CPU self-test: `python3 -m harness.docking_controller`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = ROOT / "outputs/validation/task1_docking_calibration_20260926"
DT = 1.0 / 30.0
BASE_SCALE = np.array([0.75, 0.75, 1.0])     # controller multiplies the base action by this (m/s, m/s, rad/s)
TRUNK, LARM, RARM = slice(53, 57), slice(3, 10), slice(28, 35)          # state slices (qpos)
A_TRUNK, A_LARM, A_LGRIP, A_RARM, A_RGRIP = slice(3, 7), slice(7, 14), 14, slice(15, 22), 22   # action slices
EEF_POS = {"left": slice(17, 20), "right": slice(42, 45)}
GRIPPER = {"left": slice(24, 26), "right": slice(49, 51)}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class Odometry:
    """Episode-local SE(2) pose from the measured local base velocity (proprio base_qvel), reset at episode start."""

    def __init__(self):
        self.pose = np.zeros(3)

    def reset(self):
        self.pose = np.zeros(3)

    def update(self, base_vel_local, dt=DT):
        vx, vy, wz = (float(v) for v in base_vel_local)
        x, y, th = self.pose
        # midpoint heading for the translation (better than Euler for turning motion)
        thm = th + 0.5 * wz * dt
        c, s = math.cos(thm), math.sin(thm)
        self.pose = np.array([x + (vx * c - vy * s) * dt, y + (vx * s + vy * c) * dt, wrap(th + wz * dt)])

    def to_odom(self, p_base):
        return self.pose[:2] + rot(self.pose[2]) @ np.asarray(p_base[:2], float)

    def to_base(self, p_odom):
        return rot(-self.pose[2]) @ (np.asarray(p_odom[:2], float) - self.pose[:2])


@dataclass
class Track:
    label: str
    pos: np.ndarray                  # odometry frame (x, y)
    hits: int = 1
    misses: int = 0
    last_seen: int = 0
    consumed: bool = False           # picked up / in the bin: never a navigation target again
    track_id: int = 0
    z: float | None = None           # running-mean base-frame height of the detection reference point (if given)

    def summary(self):
        return {"id": self.track_id, "label": self.label, "pos": [round(float(v), 3) for v in self.pos], "hits": self.hits,
                "misses": self.misses, "last_seen": self.last_seen, "consumed": self.consumed}


class ObjectMap:
    """Odometry-frame object tracks built from base-frame detections (nearest-neighbour association, running mean)."""

    GATE = {"can_of_soda": 0.35, "trash_can": 0.6}

    def __init__(self, min_hits=3):
        self.tracks: list[Track] = []
        self.min_hits = int(min_hits)
        self._next_id = 1

    def reset(self):
        self.tracks, self._next_id = [], 1

    def update(self, detections, odom: Odometry, step: int, visible_fn=None):
        """detections: [{label, xyz_base, score}]; visible_fn(p_base)->bool says whether an existing track should have
        been seen in this frame (inside the image and the reliable range); unseen-but-visible tracks accumulate misses."""
        seen = set()
        for det in detections:
            label = det["label"]
            p = odom.to_odom(det["xyz_base"])
            cands = [t for t in self.tracks if t.label == label and not t.consumed and id(t) not in seen]
            best = min(cands, key=lambda t: np.linalg.norm(t.pos - p), default=None)
            gate = self.GATE.get(label, 0.4)
            if best is not None and np.linalg.norm(best.pos - p) <= gate:
                w = 1.0 / min(best.hits + 1, 10)            # running mean, then slow exponential tracking
                best.pos = (1 - w) * best.pos + w * p
                zd = _det_z(det)
                if zd is not None:
                    best.z = zd if best.z is None else (1 - w) * best.z + w * zd
                best.hits += 1
                best.misses = 0
                best.last_seen = step
                seen.add(id(best))
            else:
                t = Track(label, p, last_seen=step, track_id=self._next_id, z=_det_z(det))
                self._next_id += 1
                self.tracks.append(t)
                seen.add(id(t))
        if visible_fn is not None:
            for t in self.tracks:
                if id(t) in seen or t.consumed:
                    continue
                if visible_fn(odom.to_base(t.pos)):
                    t.misses += 1
        # drop tentative tracks that failed to confirm and confirmed tracks repeatedly missed while in view
        self.tracks = [t for t in self.tracks if t.consumed or not ((t.hits < self.min_hits and t.misses >= 3)
                                                                   or (t.misses >= 12))]

    def confirmed(self, label):
        return [t for t in self.tracks if t.label == label and not t.consumed and t.hits >= self.min_hits]

    def consume_nearest(self, label, p_odom, radius=0.6):
        cands = [t for t in self.tracks if t.label == label and not t.consumed]
        best = min(cands, key=lambda t: np.linalg.norm(t.pos - p_odom), default=None)
        if best is not None and np.linalg.norm(best.pos - p_odom) <= radius:
            best.consumed = True
            return best
        return None


def _det_z(det):
    xyz = np.asarray(det.get("xyz_base", ()), float).reshape(-1)
    return float(xyz[2]) if xyz.size >= 3 and np.isfinite(xyz[2]) else None


def grasp_time_rel(r):
    """Object position in the base frame AT GRASP: the start-frame position minus the base odometry pose between pick
    start and gripper closure (`base_shift` = [x, y, yaw] in the start frame, re-verified from lowdim base_qvel)."""
    sx, sy, syaw = r["base_shift"]
    return rot(-syaw) @ (np.array([r["rel_x"], r["rel_y"]]) - np.array([sx, sy]))


def load_calibration(path=None, target_at="start"):
    """Docking targets and postures from TRAIN demonstrations (standard hands: right = bin, left = cans).

    target_at="start": object pose at the start of the demo's pick segment (the demo then still moves the base forward,
    median 6-7 cm, p90 21 cm). target_at="grasp" (2026-09-27): object pose at gripper closure, for deployments where
    the base is held during manipulation because the WAM does not reproduce those sparse base pulses
    (outputs/validation/task1_handoff_20260926/handoff_b_pickcan113_ep389_N2341_v2final_headrgbd)."""
    if target_at not in ("start", "grasp"):
        raise ValueError("target_at must be start|grasp")
    if path is None:              # per-task calibration (harness/task_profile.DOCK_CALIB; Task 1 = CALIB_DIR)
        from harness import task_profile
        path = task_profile.DOCK_CALIB or CALIB_DIR
    grasps = json.loads((Path(path) / "grasp_relative_poses_train.json").read_text())
    stats = json.loads((Path(path) / "posture_and_gripper_stats_train.json").read_text())
    dock = {}
    # Labels and their grasping hand come from the TRAIN rows (majority hand per category): Task 1 gives exactly the
    # former fixed (trash_can, right), (can_of_soda, left); Task 0 gives (radio, right).
    hands = {}
    for r in grasps:
        hands.setdefault(r["cat"], []).append(r["hand"])
    for label, hs in hands.items():
        hand = max(set(hs), key=hs.count)
        sel = [r for r in grasps if r["cat"] == label and r["hand"] == hand]
        rows = np.array([grasp_time_rel(r) if target_at == "grasp" else (r["rel_x"], r["rel_y"]) for r in sel])
        dock[label] = {"hand": hand, "target": np.median(rows, 0), "p10": np.percentile(rows, 10, 0),
                       "p90": np.percentile(rows, 90, 0), "n": int(len(rows))}
    post = stats["postures"]

    def pose(key):
        return {g: np.array(post[key][g]["median"], float) for g in ("trunk", "larm", "rarm")}
    posture = {"nav_before_bin": pose("nav_mid|before_bin")}
    if "nav_mid|carrying_bin" in post:
        posture["nav_carrying_bin"] = pose("nav_mid|carrying_bin")
    for label in dock:
        posture[f"pick_start_{label}"] = pose(f"pick up from_start|{label}")
    return {"dock": dock, "posture": posture,
            "gripper": stats["gripper_finger_sum_m"], "source": str(Path(path).resolve().relative_to(ROOT)), "target_at": target_at}


@dataclass
class DockConfig:
    v_max: float = 0.45              # m/s approach speed
    v_align: float = 0.15            # m/s during fine alignment
    w_max: float = 0.6               # rad/s
    w_search: float = 0.45           # rad/s in-place scan
    k_v: float = 1.2
    k_w: float = 1.5
    align_radius: float = 0.45       # m: switch from heading-first approach to holonomic alignment
    tol_xy: float = 0.06             # m: docking position tolerance (demo p10-p90 half-width is ~0.15 m)
    tol_bearing_deg: float = 6.0
    posture_tol: float = 0.05        # rad: trunk / free-arm targets must have reached the pick-start posture
    settle_steps: int = 10           # consecutive in-tolerance steps before reporting docked
    require_posture: bool = True     # docked also needs the pick-start posture (WAM start distribution); the IK
    #                                  primitives plan from any posture (dev10 run 2026-09-27: xy error 2 mm for
    #                                  4000 steps, never "docked" because a joint stayed > posture_tol)
    joint_rate: float = 0.8          # rad/s upper-body interpolation toward posture targets
    posture_switch_dist: float = 1.2 # m: start moving to the pick-start posture inside this range
    lost_steps: int = 90             # steps without a confirmed target before returning to search
    avoid_obstacles: bool = True     # local depth-point corridor check during approach (no global map)
    corridor_half_width: float = 0.38  # m: robot half width + margin
    corridor_start: float = 0.25     # m: points closer than this along the motion are the robot's own body/arms
    corridor_length: float = 0.7     # m: look-ahead along the motion direction
    target_mask_radius: float = 0.35 # m: depth points around the target object are not obstacles
    target_mask_by_label: dict = field(default_factory=lambda: {"can_of_soda": 0.12, "trash_can": 0.25, "radio": 0.20})
    # radio (Task 0): 0.32 x 0.14 m body, half-diagonal 0.175 m around the handle grasp point + detection slack; the
    # coffee table around it stays an obstacle (outputs/validation/task0_perception_20260927/gt_summary.json radio_asset)
    # Docking pose selection (2026-09-27, dev10 v2: approaching one can of a cluster knocked all three over with the
    # base front): choose the final heading so that no other known object lies inside the inflated base footprint at
    # the docking pose or along the straight final approach, then drive to a waypoint behind that pose first.
    plan_dock_pose: bool = True
    plan_within: float = 3.0         # m: plan once the target is this close (neighbours are mapped by then)
    footprint: tuple = (-0.40, 0.40, -0.36, 0.36)   # base footprint box x_min, x_max, y_min, y_max (m)
    footprint_margin: float = 0.08   # m: object radius + clearance
    free_hand_radius: float = 0.18   # m: 2-D arm/hand clearance around measured EEF during navigation
    held_object_radius: float = 0.25 # m: hand + carried-bin envelope around a closed gripper
    forearm_radius: float = 0.14     # m: physical arm radius plus clearance for shoulder-to-EEF capsules
    gripper_closed_sum: float = 0.085
    obstacle_voxel: float = 0.08     # m: depth cloud compression before sampled-command collision checks
    safety_max_depth_points: int = 256
    final_approach: float = 0.6      # m: straight-in distance along the chosen heading
    plan_headings: int = 24
    safety_filter: bool = True       # footprint-aware check of every base command against known object tracks
    safety_horizon: float = 1.0      # s: constant-velocity forward prediction
    stuck_steps: int = 90            # commanded >= 0.15 m/s but measured < 0.03 m/s for this long -> back off
    backoff_steps: int = 45          # reverse at 0.15 m/s (v3 run: pinned against furniture the head depth missed)
    sidestep_speed: float = 0.2      # m/s lateral slide toward the freer side when the corridor is blocked
    explore_after_turns: float = 1.1 # full in-place scans without a confirmed target before exploring
    explore_distance: float = 1.5    # m driven toward the most open heading seen during the scan
    explore_speed: float = 0.35      # m/s
    max_explorations: int = 4
    # Height-aware shield (opt-in, default off = the 2-D behaviour above; Task 0 profile: task_dock_config(0)).
    # Depth obstacles arrive as height bands [(z_lo, z_hi, is_support, xy_base)] (object_perception.obstacle_points
    # z_band calls). The base box, the approach corridor, dock-pose planning and free-space bookkeeping only see bands
    # that reach below base_top_z (what the base can hit); a torso box and the arm envelope see bands above it, each arm
    # part only for bands overlapping its own height span (EEF z from proprio, shoulder z from URDF FK). The target's
    # support top (s = target track z - support_offset_by_label[label]) defines a thin support band
    # [s + support_band[0], s + support_band[1]] that is exempt from the ARM envelope only in the slow ALIGN phase (the
    # demonstrated pick-start hands hover ~3 cm above the table top); the base box and torso box never get an exemption.
    height_bands: bool = False
    base_top_z: float = 0.37          # R1Pro base_link top (curobo collision spheres 0.365 m)
    torso_box: tuple = (-0.25, 0.12, -0.20, 0.20)   # torso column + shoulders (curobo torso spheres +- 0.1 m) + slack
    hand_dz: float = 0.10             # m: vertical half-extent of a hand / held object around the measured EEF z
    arm_dz: float = 0.08              # m: vertical half-extent of the shoulder-to-EEF capsule
    support_offset_by_label: dict = field(default_factory=dict)
    support_band: tuple = (-0.05, 0.02)
    # A new dock plan whose pose P is already within align_radius (heading within 30 deg) goes straight to ALIGN instead
    # of first backing off to the waypoint W (a re-approach after a missed grasp starts AT the dock pose, with the hands
    # over the table; turning/backing away there is exactly what the shield must refuse). Task 0 profile only.
    skip_waypoint_near_dock: bool = False

    def __post_init__(self):
        x0, x1, y0, y1 = self.footprint
        if not (x0 < x1 and y0 < y1):
            raise ValueError("footprint must be (x_min, x_max, y_min, y_max)")
        if not self.support_band[0] < self.support_band[1]:
            raise ValueError("support_band must be (below, above) with below < above")


def task_dock_config(task_id=None):
    """Per-task docking configuration. Every task except 0 gets DockConfig() unchanged (Task 1 byte-identical).

    Task 0 (turning_on_radio; progress.md 2026-09-28): the radio stands on a 0.42 m coffee table and the TRAIN demos dock
    with the base front at the table edge (train instance 65 replay, evaluator pose log: edge 0.25 m ahead of the base
    centre at pick start). The symmetric (+-0.40) footprint puts the base front 0.15 m too far forward (R1Pro base_link
    curobo spheres x -0.41..0.25, y +-0.355, top 0.365 m), and a 2-D shield treats the table top under the hands as a
    wall. Hence the URDF footprint and the height-aware shield; support offset 0.22 m = TRAIN grasp-point height median
    0.641 m minus the table top 0.423 m."""
    if task_id is None:
        from harness import task_profile
        task_id = task_profile.TASK_ID
    if int(task_id) != 0:
        return DockConfig()
    return DockConfig(footprint=(-0.41, 0.25, -0.355, 0.355), height_bands=True, support_offset_by_label={"radio": 0.22},
                      skip_waypoint_near_dock=True)


class DockingController:
    """SEARCH (scan in place) -> APPROACH (face target, drive) -> ALIGN (holonomic) -> DOCKED.

    `phase` selects the navigation posture: 'before_bin' (both hands free) or 'carrying_bin' (right hand holds the bin).
    Output: a 23-D action (base velocity command in controller units, upper-body position targets, gripper commands).
    """

    def __init__(self, calib=None, config: DockConfig | None = None):
        self.calib = calib or load_calibration()
        self.cfg = config or DockConfig()
        self.reset()

    def reset(self):
        self.label = self.phase = None
        self.mode = "idle"
        self.track_id = None
        self.in_tol = 0
        self.search_yaw = 0.0
        self.no_target_steps = 0
        self.search_turns = 0.0
        self.last_status = {}
        self._latched = {}
        self._free = {}          # odometry yaw bin (10 deg) -> max free range seen along that heading (m)
        self._explore = None     # (start xy, heading) while exploring
        self.explorations = 0
        self._visited = []       # odometry xy where scans happened (exploration prefers new ground)
        self._active_footprint = tuple(self.cfg.footprint)
        self._active_geometry = {"box": tuple(self.cfg.footprint), "circles": [], "capsules": []}
        self._plan_age = 0

    def start(self, label, phase, odom: Odometry):
        if label not in self.calib["dock"]:
            raise ValueError(f"no docking calibration for {label!r}")
        self.label, self.phase = label, phase
        self.mode, self.track_id, self.in_tol = "search", None, 0
        self._plan = None
        self._banned_psi = []
        self._stuck, self._backoff = 0, 0
        self._latched = {}
        self._free, self._explore, self.explorations = {}, None, 0
        self._visited = []
        self._active_footprint = tuple(self.cfg.footprint)
        self._active_geometry = {"box": tuple(self.cfg.footprint), "circles": [], "capsules": []}
        self._plan_age = 0
        self.search_yaw, self.search_turns, self.no_target_steps = float(odom.pose[2]), 0.0, 0

    # ---- target selection ------------------------------------------------------------------------------
    def _target(self, omap: ObjectMap, odom: Odometry):
        tracks = omap.confirmed(self.label)
        if self.track_id is not None:
            cur = next((t for t in tracks if t.track_id == self.track_id), None)
            if cur is not None:
                return cur
        if not tracks:
            return None
        best = min(tracks, key=lambda t: np.linalg.norm(t.pos - odom.pose[:2]))
        self.track_id = best.track_id
        return best

    # ---- upper body ------------------------------------------------------------------------------------
    def _posture_target(self, dist):
        post = self.calib["posture"]
        nav = post["nav_before_bin"] if self.phase == "before_bin" else post["nav_carrying_bin"]
        pick = post[f"pick_start_{self.label}"]
        if dist is None or dist > self.cfg.posture_switch_dist:
            tgt = nav
        else:
            tgt = pick
        if self.phase == "carrying_bin":
            # The arm holding the bin keeps the bin rim at the demos' carrying pose (TRAIN median right eef
            # (0.616, -0.304, 0.522) + orientation) for the current trunk target, by IK. Latching its joints let the
            # bin sink with the trunk crouch until it dragged on the floor and blocked the crouch (dev10 v5).
            tgt = {**tgt, "rarm": self._carry_rarm(tgt["trunk"])}
        return tgt

    def _carry_rarm(self, trunk):
        key = tuple(np.round(np.asarray(trunk, float), 3))
        cache = self.__dict__.setdefault("_carry_cache", {})
        if key not in cache:
            try:
                from harness.r1pro_kinematics import R1ProKinematics
                geo = json.loads((CALIB_DIR / "grasp_geometry_train.json").read_text())["carry"]
                kin = self.__dict__.setdefault("_kin", R1ProKinematics())
                q0 = np.r_[np.asarray(trunk, float), self.calib["posture"]["nav_carrying_bin"]["rarm"]]
                q, info = kin.ik("right", q0, np.array(geo["right_eef_pos_median"]), np.array(geo["right_eef_R_median"]),
                                 q_bias=q0, iters=400)
                cache[key] = q[4:] if info["pos_err"] < 0.02 else None
            except (FileNotFoundError, KeyError):
                cache[key] = None
        return cache[key]

    def _posture_reached(self, state, dist):
        """Trunk and free-arm joints within posture_tol of the pick-start posture (the WAM's demonstrated start)."""
        tgt = self._posture_target(dist)
        ok, err = True, {}
        for g, s in (("trunk", TRUNK), ("larm", LARM), ("rarm", RARM)):
            if tgt.get(g) is not None:
                d = np.abs(np.asarray(state[s], float) - tgt[g])
                err[g] = [round(float(d.max()), 3), int(d.argmax())]
                ok = ok and float(d.max()) <= self.cfg.posture_tol
        self.posture_err = err
        return ok or not self.cfg.require_posture

    def _upper_body(self, state, target):
        """Rate-limited position targets. A group without a target (the arm holding the bin) keeps the joint targets
        latched at contract start: re-reading the measured pose every step would let gravity sag accumulate."""
        out = {}
        step = self.cfg.joint_rate * DT
        for g, s in (("trunk", TRUNK), ("larm", LARM), ("rarm", RARM)):
            q = np.asarray(state[s], float)
            t = target.get(g)
            if t is None:
                out[g] = self._latched.setdefault(g, q.copy())
            else:
                prev = self._latched.get(g, q)
                out[g] = prev + np.clip(t - prev, -step, step)
                self._latched[g] = out[g]
        return out

    # ---- height-aware shield (cfg.height_bands; Task 0) -------------------------------------------------
    def support_top(self, omap: ObjectMap):
        """Support-top height under the current target (base frame, m) from its tracked reference-point height, or None."""
        off = self.cfg.support_offset_by_label.get(self.label)
        if off is None:
            return None
        tracks = omap.confirmed(self.label)
        tr = next((t for t in tracks if t.track_id == self.track_id), None) or (tracks[0] if len(tracks) == 1 else None)
        if tr is None or tr.z is None:
            return None
        return float(np.clip(tr.z - off, 0.10, 1.20))

    def band_edges(self, omap: ObjectMap, z_min=0.05, z_max=1.5):
        """Disjoint height bands [(z_lo, z_hi, is_support)] partitioning [z_min, z_max] at the base top and the support band."""
        cfg = self.cfg
        cuts = {z_min, z_max, float(cfg.base_top_z)}
        st = self.support_top(omap)
        sb = None
        if st is not None:
            sb = (st + cfg.support_band[0], st + cfg.support_band[1])
            cuts |= {float(np.clip(v, z_min, z_max)) for v in sb}
        zs = sorted(c for c in cuts if z_min <= c <= z_max)
        out = []
        for lo, hi in zip(zs[:-1], zs[1:]):
            if hi - lo > 1e-6:
                out.append((lo, hi, sb is not None and lo >= sb[0] - 1e-6 and hi <= sb[1] + 1e-6))
        return out

    def _shoulder_z(self, state, arm):
        """Height of the arm's first joint (base frame) from the measured trunk joints by URDF FK (cached)."""
        trunk = np.asarray(state[TRUNK], float)
        key = (arm, tuple(np.round(trunk, 2)))
        cache = self.__dict__.setdefault("_shoulder_cache", {})
        if key not in cache:
            try:
                from harness.r1pro_kinematics import R1ProKinematics, axis_angle
                kin = self.__dict__.setdefault("_kin", R1ProKinematics())
                T, k = np.eye(4), 0
                for n in kin.chain[arm]:
                    j = kin.joints[n]
                    T = T @ j["T"]
                    if j["type"] != "fixed":
                        if k == 4:                      # first arm joint (after the 4 trunk joints)
                            break
                        T = T @ kin._T(axis_angle(j["axis"], trunk[k]), np.zeros(3))
                        k += 1
                cache[key] = float(T[2, 3] + kin.root_offset[2])
            except (FileNotFoundError, KeyError, IndexError):
                cache[key] = 1.0
        return cache[key]

    def _band_groups(self, state, bands, target_p):
        """Shield groups [(points, geometry)] for height-band mode (see DockConfig.height_bands)."""
        cfg = self.cfg
        geom = self._geometry_for_state(state)
        s = np.asarray(state, float).reshape(-1)
        spans = []                                   # per circle/capsule: (z_lo, z_hi) from proprio / FK
        for hand in ("left", "right"):
            ez = float(s[EEF_POS[hand]][2]) if s.size >= 61 and np.isfinite(s[EEF_POS[hand]][2]) else None
            if ez is None:
                continue
            spans.append(("circle", (ez - cfg.hand_dz, ez + cfg.hand_dz)))
            zs = self._shoulder_z(s, hand)
            spans.append(("capsule", (min(zs, ez) - cfg.arm_dz, max(zs, ez) + cfg.arm_dz)))
        circles = [c for c in geom["circles"]]
        capsules = [c for c in geom["capsules"]]
        cspan = [sp for kind, sp in spans if kind == "circle"]
        kspan = [sp for kind, sp in spans if kind == "capsule"]
        align = self.mode == "align"
        groups, n_pts = [], 0
        pts_all = []
        for lo, hi, support, pts in bands:
            P = self._depth_for_safety(pts, target_p)
            if not len(P):
                continue
            g = {"box": tuple(float(v) for v in cfg.footprint) if lo < cfg.base_top_z else None,
                 "torso": tuple(float(v) for v in cfg.torso_box) if hi > cfg.base_top_z else None,
                 "circles": [] if (support and align) else [c for c, sp in zip(circles, cspan) if lo < sp[1] and hi > sp[0]],
                 "capsules": [] if (support and align) else [c for c, sp in zip(capsules, kspan) if lo < sp[1] and hi > sp[0]]}
            if g["box"] is None and g["torso"] is None and not g["circles"] and not g["capsules"]:
                continue
            groups.append((P, g))
            pts_all.append(P)
            n_pts += len(P)
        if n_pts > cfg.safety_max_depth_points:          # keep the nearest points overall (same budget as 2-D mode)
            d = np.sort(np.concatenate([np.linalg.norm(P, axis=1) for P in pts_all]))
            cut = d[cfg.safety_max_depth_points - 1]
            groups = [(P[np.linalg.norm(P, axis=1) <= cut], g) for P, g in groups]
            groups = [(P, g) for P, g in groups if len(P)]
        return groups

    def _unsafe_groups(self, cmd, groups):
        """Height-band version of _unsafe: each point group has its own geometry (box / torso / arm parts)."""
        cfg = self.cfg
        m = cfg.footprint_margin

        def depth_in(o, g):
            depths = []
            for box in (g["box"], g["torso"]):
                if box is not None:
                    x0, x1, y0, y1 = box
                    depths.append(min(o[0] - (x0 - m), (x1 + m) - o[0], o[1] - (y0 - m), (y1 + m) - o[1]))
            depths.extend(r - float(np.linalg.norm(o - c)) for c, r in g["circles"])
            for a, b, r in g["capsules"]:
                ab = b - a
                u = float(np.clip((o - a) @ ab / (float(ab @ ab) + 1e-9), 0.0, 1.0))
                depths.append(r - float(np.linalg.norm(o - (a + u * ab))))
            return max(depths) if depths else -np.inf
        n = 10
        for P, g in groups:
            for o in P:
                d0 = depth_in(o, g)
                pose = np.zeros(3)
                d = d0
                for k in range(1, n + 1):
                    dt = cfg.safety_horizon / n
                    th = pose[2] + 0.5 * cmd[2] * dt
                    pose = pose + np.array([(cmd[0] * math.cos(th) - cmd[1] * math.sin(th)) * dt,
                                            (cmd[0] * math.sin(th) + cmd[1] * math.cos(th)) * dt, cmd[2] * dt])
                    d = depth_in(rot(-pose[2]) @ (np.asarray(o) - pose[:2]), g)
                    if d > 0 and d > d0 + 1e-3:
                        return True
                if d0 > 0 and float(np.linalg.norm(cmd)) > 1e-4 and d > d0 - 3e-3:
                    return True
        return False

    # ---- main step -------------------------------------------------------------------------------------
    def _avoid(self, base, obstacles, target_p):
        """Scale the planar command down and add a lateral slide when depth points block the motion corridor."""
        cfg = self.cfg
        v = np.asarray(base[:2], float)
        speed = float(np.linalg.norm(v))
        if not cfg.avoid_obstacles or obstacles is None or len(obstacles) == 0 or speed < 1e-3:
            return base, None
        P = np.asarray(obstacles, float).reshape(-1, 2)
        if target_p is not None:
            r = cfg.target_mask_by_label.get(getattr(self, "label", None), cfg.target_mask_radius)
            P = P[np.linalg.norm(P - np.asarray(target_p[:2], float), axis=1) > r]
        if not len(P):
            return base, None
        d = v / speed
        n = np.array([-d[1], d[0]])
        along, lateral = P @ d, P @ n
        ahead = (along > cfg.corridor_start) & (along < cfg.corridor_start + cfg.corridor_length) & (np.abs(lateral) < cfg.corridor_half_width)
        if not ahead.any():
            return base, {"blocked": False}
        near = float(along[ahead].min())
        scale = float(np.clip((near - cfg.corridor_start) / cfg.corridor_length, 0.0, 1.0))
        band = (along > 0.0) & (along < cfg.corridor_start + 2 * cfg.corridor_length) & (np.abs(lateral) < 1.2)
        side = 1.0 if np.count_nonzero(band & (lateral > 0)) < np.count_nonzero(band & (lateral < 0)) else -1.0
        out = np.array(base, float)
        out[:2] = v * scale + side * cfg.sidestep_speed * n
        return out, {"blocked": True, "nearest_along": round(near, 3), "scale": round(scale, 2), "side": side}

    def _record_free(self, odom, obstacles, fov_deg=40.0, max_range=4.0):
        """Free range straight ahead (from depth points in a narrow cone), stored per odometry heading bin."""
        if obstacles is None:
            return
        P = np.asarray(obstacles, float).reshape(-1, 2)
        ang = np.degrees(np.arctan2(P[:, 1], P[:, 0])) if len(P) else np.zeros(0)
        cone = P[np.abs(ang) < fov_deg / 2] if len(P) else P
        free = float(min(max_range, np.linalg.norm(cone, axis=1).min())) if len(cone) else max_range
        key = int(round(math.degrees(odom.pose[2]) / 10.0)) % 36
        self._free[key] = max(self._free.get(key, 0.0), free)

    def _explore_cmd(self, odom, obstacles):
        """Drive toward the most open heading of the last scan; returns None when the leg is finished."""
        cfg = self.cfg
        if self._explore is None:
            if not self._free:
                return None
            here = odom.pose[:2].copy()
            self._visited.append(here)

            def score(key):
                # open space, discounted when the heading points back toward ground already scanned
                h = math.radians(key * 10.0)
                d = np.array([math.cos(h), math.sin(h)])
                back = max((float(d @ (v - here)) / (np.linalg.norm(v - here) + 1e-6) for v in self._visited[:-1]
                            if np.linalg.norm(v - here) > 0.3), default=0.0)
                return min(self._free[key], cfg.explore_distance + 1.0) * (1.0 - 0.6 * max(0.0, back))
            key = max(self._free, key=score)
            self._explore = (here, math.radians(key * 10.0))
            self.explorations += 1
        start, heading = self._explore
        if float(np.linalg.norm(odom.pose[:2] - start)) >= cfg.explore_distance:
            self._explore, self._free, self.search_turns = None, {}, 0.0
            return None
        err = wrap(heading - odom.pose[2])
        base = np.array([cfg.explore_speed * max(0.0, math.cos(err)) ** 4, 0.0, np.clip(cfg.k_w * err, -cfg.w_max, cfg.w_max)])
        base, _ = self._avoid(base, obstacles, None)
        return base

    def _footprint_for_state(self, state):
        """Axis-aligned base-frame collision envelope of base, measured EEFs and a possible carried object.

        A fixed mobile-base footprint was insufficient in dev10 v7: the open left arm caught a lounge chair while
        the base itself remained clear, then the robot dragged the chair for the rest of the episode.  This envelope
        is deliberately policy-visible (proprio only) and conservative; it is a safety shield, not a task planner.
        """
        x0, x1, y0, y1 = (float(v) for v in self.cfg.footprint)
        s = np.asarray(state, float).reshape(-1)
        if s.size < 61:
            return x0, x1, y0, y1
        for hand in ("left", "right"):
            e = s[EEF_POS[hand]]
            if not np.all(np.isfinite(e)):
                continue
            closed = float(np.sum(s[GRIPPER[hand]])) < self.cfg.gripper_closed_sum
            r = self.cfg.held_object_radius if closed else self.cfg.free_hand_radius
            x0, x1 = min(x0, float(e[0] - r)), max(x1, float(e[0] + r))
            y0, y1 = min(y0, float(e[1] - r)), max(y1, float(e[1] + r))
        return x0, x1, y0, y1

    def _geometry_for_state(self, state):
        """Union geometry used by the online shield: base box + arm capsules + hand/held-object circles.

        Unlike the bounding rectangle used for coarse dock-pose planning, this union does not mark the empty corners
        between an extended hand and the base as occupied. That distinction is necessary for safe in-place scans.
        """
        geom = {"box": tuple(float(v) for v in self.cfg.footprint), "circles": [], "capsules": []}
        s = np.asarray(state, float).reshape(-1)
        if s.size < 61:
            return geom
        for hand, sign in (("left", 1.0), ("right", -1.0)):
            e = np.asarray(s[EEF_POS[hand]][:2], float)
            if not np.all(np.isfinite(e)):
                continue
            closed = float(np.sum(s[GRIPPER[hand]])) < self.cfg.gripper_closed_sum
            r = self.cfg.held_object_radius if closed else self.cfg.free_hand_radius
            geom["circles"].append((e.copy(), float(r)))
            shoulder = np.array([0.05, sign * 0.24])
            geom["capsules"].append((shoulder, e.copy(), float(self.cfg.forearm_radius)))
        return geom

    def _depth_for_safety(self, obstacles, target_p):
        """Mask the current manipulation target, voxelise depth, and keep the nearest collision evidence."""
        if obstacles is None or not len(obstacles):
            return np.zeros((0, 2), float)
        P = np.asarray(obstacles, float).reshape(-1, 2)
        P = P[np.all(np.isfinite(P), axis=1) & (np.linalg.norm(P, axis=1) < 2.5)]
        if target_p is not None and len(P):
            r = self.cfg.target_mask_by_label.get(getattr(self, "label", None), self.cfg.target_mask_radius)
            P = P[np.linalg.norm(P - np.asarray(target_p[:2], float), axis=1) > r]
        if not len(P):
            return P
        cell = max(float(self.cfg.obstacle_voxel), 1e-3)
        _, idx = np.unique(np.floor(P / cell).astype(np.int32), axis=0, return_index=True)
        P = P[np.sort(idx)]
        if len(P) > self.cfg.safety_max_depth_points:
            P = P[np.argsort(np.linalg.norm(P, axis=1))[:self.cfg.safety_max_depth_points]]
        return P

    def _plan_pose(self, tr, omap: ObjectMap, odom: Odometry, obstacles_odom=None):
        """Choose the docking heading psi (odometry frame). Robot position for psi: P = T - R(psi) @ dock_target. A psi
        is rejected when another known object (confirmed, not consumed) lies inside the inflated footprint at P or in
        the box swept along the last `final_approach` metres; among the rest the one closest to the natural heading
        (facing the target from here at the docking bearing) wins."""
        cfg = self.cfg
        T = np.asarray(tr.pos, float)
        tgt = np.asarray(self.calib["dock"][self.label]["target"], float)
        others = [t.pos for t in omap.tracks if t.track_id != tr.track_id and not t.consumed
                  and t.hits >= 3 and float(np.linalg.norm(t.pos - T)) < 2.0]
        d = T - odom.pose[:2]
        psi_nat = math.atan2(d[1], d[0]) - math.atan2(tgt[1], tgt[0])
        x0, x1, y0, y1 = self._active_footprint
        m = cfg.footprint_margin
        best, n_blocked = None, 0
        banned = getattr(self, "_banned_psi", [])
        walls = np.zeros((0, 2)) if obstacles_odom is None else np.asarray(obstacles_odom, float).reshape(-1, 2)
        if len(walls):
            walls = walls[np.linalg.norm(walls - T, axis=1) > cfg.target_mask_by_label.get(self.label, 0.35)]
        for k in range(cfg.plan_headings):
            psi = wrap(psi_nat + 2 * math.pi * k / cfg.plan_headings)
            if any(abs(wrap(psi - b)) < math.radians(30) for b in banned):
                continue
            P = T - rot(psi) @ tgt
            hits = 0
            if len(walls):             # depth points (walls, furniture) inside the footprint or the final corridor
                wb = (walls - P) @ rot(psi)           # rows: rot(-psi) @ (w - P)
                inside = ((x0 - cfg.final_approach) < wb[:, 0]) & (wb[:, 0] < x1) & (y0 < wb[:, 1]) & (wb[:, 1] < y1)
                hits += int(np.count_nonzero(inside) >= 3)
            for o in others:
                ob = rot(-psi) @ (np.asarray(o) - P)
                if (x0 - m - cfg.final_approach) < ob[0] < (x1 + m) and (y0 - m) < ob[1] < (y1 + m):
                    hits += 1
            cost = hits * 10.0 + abs(wrap(psi - psi_nat))
            if best is None or cost < best[0]:
                best = (cost, psi, P, hits)
            n_blocked += hits > 0
        if best is None:
            return None
        _, psi, P, hits = best
        W = P - cfg.final_approach * np.array([math.cos(psi), math.sin(psi)])
        return {"track": tr.track_id, "psi": psi, "P": P, "W": W, "clear": hits == 0, "n_others": len(others),
                "n_depth": int(len(walls)), "has_obstacles": bool(len(others) or len(walls)),
                "headings_blocked": n_blocked, "psi_nat": wrap(psi_nat), "reached_W": False,
                "best_to_W": float("inf"), "stall_steps": 0}

    def _unsafe(self, cmd, others):
        """True if moving with cmd (vx, vy, wz; base frame) for safety_horizon brings any object (base-frame xy) inside
        the inflated footprint that it is not already inside (objects already inside may only be moved away from)."""
        cfg = self.cfg
        geom = getattr(self, "_active_geometry", {"box": tuple(cfg.footprint), "circles": [], "capsules": []})
        x0, x1, y0, y1 = geom["box"]
        m = cfg.footprint_margin

        def depth_in(o):              # penetration depth into the inflated box (> 0 inside)
            depths = [min(o[0] - (x0 - m), (x1 + m) - o[0], o[1] - (y0 - m), (y1 + m) - o[1])]
            depths.extend(r - float(np.linalg.norm(o - c)) for c, r in geom["circles"])
            for a, b, r in geom["capsules"]:
                ab = b - a
                u = float(np.clip((o - a) @ ab / (float(ab @ ab) + 1e-9), 0.0, 1.0))
                depths.append(r - float(np.linalg.norm(o - (a + u * ab))))
            return max(depths)
        n = 10
        for o in others:
            d0 = depth_in(o)
            pose = np.zeros(3)
            for k in range(1, n + 1):
                dt = cfg.safety_horizon / n
                th = pose[2] + 0.5 * cmd[2] * dt
                pose = pose + np.array([(cmd[0] * math.cos(th) - cmd[1] * math.sin(th)) * dt,
                                        (cmd[0] * math.sin(th) + cmd[1] * math.cos(th)) * dt, cmd[2] * dt])
                ob = rot(-pose[2]) @ (np.asarray(o) - pose[:2])
                d = depth_in(ob)
                if d > 0 and d > d0 + 1e-3:
                    return True
            # Once an obstacle has entered the robot/arm envelope, any non-zero escape command must actually reduce
            # penetration.  The previous "not deeper" rule permitted tangential motion and therefore chair dragging.
            if d0 > 0 and float(np.linalg.norm(cmd)) > 1e-4 and d > d0 - 3e-3:
                return True
        return False

    def _safe(self, base, omap: ObjectMap, odom: Odometry, target_id, obstacles=None, target_p=None, state=None, bands=None):
        """Closest safe command to `base` against semantic objects *and* raw furniture-depth points."""
        if not self.cfg.safety_filter:
            return base, None
        semantic = [odom.to_base(t.pos) for t in omap.tracks if t.track_id != target_id and not t.consumed and t.hits >= 3
                    and float(np.linalg.norm(odom.to_base(t.pos))) < 2.0]
        if bands is not None and self.cfg.height_bands and state is not None:
            groups = self._band_groups(state, bands, target_p)
            n_depth = int(sum(len(P) for P, _ in groups))
            if semantic:          # other task objects: full base + arm geometry, as in the 2-D mode
                g = self._active_geometry
                groups = [(np.asarray(semantic, float).reshape(-1, 2),
                           {"box": g["box"], "torso": None, "circles": g["circles"], "capsules": g["capsules"]})] + groups
            if not groups or not self._unsafe_groups(base, groups):
                return base, None
            ok = [c for c in self._safe_candidates(base) if not self._unsafe_groups(c, groups)]
            info = {"n_semantic": len(semantic), "n_depth": n_depth, "height_bands": True,
                    "support_exempt": bool(self.mode == "align" and any(b[2] for b in bands))}
            if not ok:
                return np.zeros(3), {"safety": "stop", **info}
            return min(ok, key=lambda c: float(np.linalg.norm(c - base))), {"safety": "modified", **info}
        depth = self._depth_for_safety(obstacles, target_p)
        others = semantic + [p for p in depth]
        if not others or not self._unsafe(base, others):
            return base, None
        cands = self._safe_candidates(base)
        ok = [c for c in cands if not self._unsafe(c, others)]
        if not ok:
            return np.zeros(3), {"safety": "stop", "n_semantic": len(semantic), "n_depth": int(len(depth))}
        best = min(ok, key=lambda c: float(np.linalg.norm(c - base)))
        return best, {"safety": "modified", "n_semantic": len(semantic), "n_depth": int(len(depth))}

    def _safe_candidates(self, base):
        cands = []
        for sc in (0.7, 0.4, 0.2):
            cands.append(np.array([base[0] * sc, base[1] * sc, base[2]]))
        for sc in (1.0, 0.5):
            for side in (-1, 1):
                cands.append(np.array([base[0] * 0.3, side * 0.15 * sc, base[2]]))
                cands.append(np.array([-0.1 * sc, side * 0.1 * sc, 0.0]))
        for a in np.linspace(-math.pi, math.pi, 8, endpoint=False):
            cands.append(np.array([0.12 * math.cos(a), 0.12 * math.sin(a), 0.0]))
        turn = max(0.18, min(abs(float(base[2])), self.cfg.w_max))
        cands += [np.array([0.0, 0.0, base[2]]), np.array([0.0, 0.0, turn]), np.array([0.0, 0.0, -turn]),
                  np.array([-0.1, 0.0, 0.0]), np.zeros(3)]
        return cands

    def step(self, state, omap: ObjectMap, odom: Odometry, obstacles=None, bands=None):
        """bands (cfg.height_bands only): [(z_lo, z_hi, is_support, xy_base (N, 2))] from band_edges(); then `obstacles`
        is ignored: base-height bands drive the corridor/planning and the full band set drives the shield."""
        cfg, cal = self.cfg, self.calib["dock"][self.label]
        self._active_footprint = self._footprint_for_state(state)
        self._active_geometry = self._geometry_for_state(state)
        band_mode = bool(cfg.height_bands and bands is not None)
        if band_mode:
            low = [np.asarray(pts, float).reshape(-1, 2) for lo, hi, _, pts in bands if lo < cfg.base_top_z]
            obstacles = np.concatenate(low) if low else np.zeros((0, 2))
            self._active_footprint = tuple(float(v) for v in cfg.footprint)   # arms are checked per height in _safe
        tr = self._target(omap, odom)
        base = np.zeros(3)
        dist = None
        target_p = None
        status = {"mode": self.mode, "label": self.label, "phase": self.phase,
                  "collision_footprint": [round(float(v), 3) for v in self._active_footprint]}
        if tr is None:
            self.no_target_steps += 1
            if self.mode in ("approach", "align") and self.no_target_steps < cfg.lost_steps:
                base[:] = 0.0                      # brief loss: hold still and wait for re-detection
            else:
                explore = None
                if self.search_turns >= cfg.explore_after_turns and self.explorations < cfg.max_explorations or self._explore:
                    explore = self._explore_cmd(odom, obstacles)
                if explore is not None:
                    self.mode = "explore"
                    base[:] = explore
                else:
                    self.mode = "search"
                    base[2] = cfg.w_search
                    self.search_turns += cfg.w_search * DT / (2 * math.pi)
                    self._record_free(odom, obstacles)
            status.update(target=None, search_turns=round(self.search_turns, 2), explorations=self.explorations)
        else:
            self.no_target_steps = 0
            p = odom.to_base(tr.pos)
            target_p = p
            dist = float(np.linalg.norm(p))
            tgt = np.asarray(cal["target"], float)
            e = p - tgt
            bearing = math.atan2(p[1], p[0])
            bearing_err = wrap(bearing - math.atan2(tgt[1], tgt[0]))
            if self.mode in ("search", "idle"):
                self.mode = "approach"
            plan = getattr(self, "_plan", None)
            if cfg.plan_dock_pose and dist < cfg.plan_within and self.mode == "approach":
                n_now = sum(1 for t in omap.tracks if t.track_id != tr.track_id and not t.consumed and t.hits >= 3)
                self._plan_age = getattr(self, "_plan_age", 0) + 1
                if (plan is None or plan["track"] != tr.track_id or plan.get("n_seen") != n_now
                        or (not plan.get("reached_W") and self._plan_age >= 60)):
                    obst_odom = None
                    if obstacles is not None and len(obstacles):
                        obst_odom = odom.pose[:2] + np.asarray(obstacles, float).reshape(-1, 2) @ rot(odom.pose[2]).T
                    new = self._plan_pose(tr, omap, odom, obst_odom)
                    plan = self._plan = None if new is None else {**new, "n_seen": n_now}
                    self._plan_age = 0
                    if (plan is not None and cfg.skip_waypoint_near_dock
                            and float(np.linalg.norm(plan["P"] - odom.pose[:2])) <= cfg.align_radius
                            and abs(wrap(plan["psi"] - odom.pose[2])) <= math.radians(30)):
                        plan["reached_W"] = True          # already at the dock pose: no back-off to W
            if plan is not None and plan["track"] == tr.track_id and plan.get("has_obstacles"):
                # the chosen heading replaces the target-bearing heading; the final approach is straight in
                bearing_err = wrap(plan["psi"] - odom.pose[2])      # > 0: turn left (same sign as the bearing error)
                if self.mode == "approach" and not plan["reached_W"]:
                    wb = odom.to_base(plan["W"])
                    dW = float(np.linalg.norm(wb))
                    if dW < plan["best_to_W"] - 0.05:
                        plan["best_to_W"], plan["stall_steps"] = dW, 0
                    else:
                        plan["stall_steps"] += 1
                    if plan["stall_steps"] > 150:  # no progress toward W: ban this heading and re-plan next step
                        self._banned_psi = getattr(self, "_banned_psi", []) + [plan["psi"]]
                        status["dock_plan_banned_deg"] = [round(math.degrees(b), 1) for b in self._banned_psi]
                        self._plan = None
                        plan = None
                if plan is not None and self.mode == "approach" and not plan["reached_W"]:
                    wb = odom.to_base(plan["W"])
                    if float(np.linalg.norm(wb)) < 0.12:
                        plan["reached_W"] = True
                        self.mode = "align"
                    else:
                        h = math.atan2(wb[1], wb[0])
                        near = float(np.linalg.norm(wb)) < 0.5      # close to W: holonomic, already turning to psi
                        w = np.clip(cfg.k_w * (bearing_err if near else h), -cfg.w_max, cfg.w_max)
                        if near:
                            v = np.clip(cfg.k_v * wb, -cfg.v_align, cfg.v_align)
                            base[:] = (v[0], v[1], w)
                        else:
                            fwd = np.clip(cfg.k_v * float(np.linalg.norm(wb)), 0.0, cfg.v_max) * max(0.0, math.cos(h)) ** 4
                            base[:] = (fwd, 0.0, w)
                        obst = obstacles
                        others = [odom.to_base(t.pos) for t in omap.tracks if t.track_id != tr.track_id and not t.consumed and t.hits >= 3]
                        if others:
                            ring = np.concatenate([o + 0.05 * np.stack([np.cos(np.linspace(0, 6.28, 8)), np.sin(np.linspace(0, 6.28, 8))], 1) for o in others])
                            obst = ring if obst is None or not len(obst) else np.concatenate([np.asarray(obst, float).reshape(-1, 2), ring])
                        base[:], avoid = self._avoid(base, obst, p)
                        status.update(dock_plan={"psi_deg": round(math.degrees(plan["psi"]), 1), "clear": plan["clear"],
                                                 "n_others": plan["n_others"], "n_depth": plan.get("n_depth", 0),
                                                 "to_W": round(float(np.linalg.norm(wb)), 3)},
                                      **({"avoid": avoid} if avoid is not None else {}))
                        self.mode = "approach_W"
            if self.mode == "approach_W":
                self.mode = "approach"
                status["mode_detail"] = "approach_W"
            elif self.mode == "approach" and float(np.linalg.norm(e)) <= cfg.align_radius and (plan is None or plan["track"] != tr.track_id or plan["n_others"] == 0 or plan["reached_W"]):
                self.mode = "align"
            if self.mode == "align" and float(np.linalg.norm(e)) > cfg.align_radius + 0.25:
                self.mode = "approach"
            if status.get("mode_detail") == "approach_W":
                pass
            elif self.mode == "approach":
                # face the target at the docking bearing first, then drive; forward speed fades with heading error
                w = np.clip(cfg.k_w * bearing_err, -cfg.w_max, cfg.w_max)
                fwd = np.clip(cfg.k_v * (dist - float(np.linalg.norm(tgt))), -cfg.v_align, cfg.v_max)
                fwd *= max(0.0, math.cos(bearing_err)) ** 4
                base[:] = (fwd, 0.0, w)
                base[:], avoid = self._avoid(base, obstacles, p)
                if avoid is not None:
                    status["avoid"] = avoid
            else:
                v = np.clip(cfg.k_v * e, -cfg.v_align, cfg.v_align)
                w = np.clip(cfg.k_w * bearing_err, -cfg.w_max * 0.5, cfg.w_max * 0.5)
                base[:] = (v[0], v[1], w)
                ok = (float(np.linalg.norm(e)) <= cfg.tol_xy and abs(math.degrees(bearing_err)) <= cfg.tol_bearing_deg
                      and self._posture_reached(state, dist))
                self.in_tol = self.in_tol + 1 if ok else 0
                if self.in_tol >= cfg.settle_steps:
                    self.mode = "docked"
            if self.mode == "docked":
                base[:] = 0.0
                if float(np.linalg.norm(e)) > 2 * cfg.tol_xy:     # drifted: re-align
                    self.mode, self.in_tol = "align", 0
            status.update(target=tr.summary(), p_base=[round(float(v), 3) for v in p], dock_target=tgt.round(3).tolist(),
                          error_xy=[round(float(v), 3) for v in e], error_norm=round(float(np.linalg.norm(e)), 3),
                          bearing_err_deg=round(math.degrees(bearing_err), 1), in_tol=self.in_tol,
                          posture_err=getattr(self, "posture_err", None))
        # stuck recovery (physical contact the depth corridor does not see): back off, ban the planned heading
        meas = float(np.linalg.norm(np.asarray(state[:2], float)))
        # compare with the command actually SENT last step (after the safety filter): a filter-reduced speed is not
        # a contact (dev10 v5: safety-scaled align motion was taken for "stuck" and triggered back-off loops)
        cmd_v = float(np.linalg.norm(getattr(self, "_last_sent", np.zeros(3))[:2]))
        if getattr(self, "_backoff", 0) > 0:
            self._backoff -= 1
            base[:] = (-0.15, 0.0, 0.0)
            status["recovery"] = "backoff"
        elif cmd_v >= 0.15 and meas < 0.03:
            self._stuck = getattr(self, "_stuck", 0) + 1
            if self._stuck >= cfg.stuck_steps:
                self._stuck, self._backoff = 0, cfg.backoff_steps
                self.stuck_events = getattr(self, "stuck_events", 0) + 1
                if getattr(self, "_plan", None) is not None:
                    self._banned_psi = getattr(self, "_banned_psi", []) + [self._plan["psi"]]
                    self._plan = None
                status["recovery"] = "stuck_detected"
        else:
            self._stuck = 0
        status["mode"] = self.mode
        if self.mode not in ("docked",) and float(np.abs(base).max()) > 1e-6 and not getattr(self, "_backoff", 0):
            if band_mode:
                base[:], safety = self._safe(base, omap, odom, self.track_id, obstacles=obstacles, target_p=target_p,
                                             state=state, bands=bands)
            else:
                base[:], safety = self._safe(base, omap, odom, self.track_id, obstacles=obstacles, target_p=target_p)
            if safety is not None:
                status["safety"] = safety
        self._last_sent = base.copy()
        ub = self._upper_body(state, self._posture_target(dist))
        action = np.zeros(23, np.float32)
        action[0:3] = np.clip(base / BASE_SCALE, -1.0, 1.0)
        action[A_TRUNK], action[A_LARM], action[A_RARM] = ub["trunk"], ub["larm"], ub["rarm"]
        action[A_LGRIP] = 1.0                                   # left hand free while navigating (opens)
        action[A_RGRIP] = 1.0 if self.phase == "before_bin" else -1.0   # right hand keeps the bin closed
        status["docked"] = self.mode == "docked"
        self.last_status = status
        return action, status


# ---- CPU self-test: holonomic robot + static objects, detections from the true relative pose -----------------
def _self_test():
    rng = np.random.default_rng(0)
    cal = load_calibration()
    assert cal["dock"]["trash_can"]["hand"] == "right" and cal["dock"]["can_of_soda"]["hand"] == "left"
    for label, obj, start_yaw in (("trash_can", np.array([3.0, 1.5]), 2.5), ("can_of_soda", np.array([-2.0, 2.5]), 0.0),
                                  ("can_of_soda", np.array([0.4, -0.3]), 1.0)):
        odom, omap, ctl = Odometry(), ObjectMap(), DockingController(cal)
        true = np.array([0.0, 0.0, start_yaw])                      # world pose; odometry starts at zero
        state = np.zeros(61, np.float32)
        ctl.start(label, "before_bin" if label == "trash_can" else "carrying_bin", odom)
        docked_at = None
        for t in range(3000):
            rel = rot(-true[2]) @ (obj - true[:2])
            fov = abs(math.atan2(rel[1], rel[0])) < math.radians(45) and 0.3 < np.linalg.norm(rel) < 6.0
            dets = ([{"label": label, "xyz_base": np.r_[rel + rng.normal(0, 0.02, 2), 0.0], "score": 1.0}]
                    if fov and t % 5 == 0 else [])
            if t % 5 == 0:
                omap.update(dets, odom, t, visible_fn=lambda pb: abs(math.atan2(pb[1], pb[0])) < math.radians(40)
                            and 0.4 < np.linalg.norm(pb) < 5.0)
            action, st = ctl.step(state, omap, odom)
            vel = action[:3] * BASE_SCALE                                # ideal velocity tracking
            state[0:3] = vel
            c, s = math.cos(true[2]), math.sin(true[2])
            true = np.array([true[0] + (vel[0] * c - vel[1] * s) * DT, true[1] + (vel[0] * s + vel[1] * c) * DT,
                             wrap(true[2] + vel[2] * DT)])
            odom.update(vel)
            state[TRUNK], state[LARM], state[RARM] = action[A_TRUNK], action[A_LARM], action[A_RARM]
            if st["docked"]:
                docked_at = t
                break
        rel = rot(-true[2]) @ (obj - true[:2])
        err = float(np.linalg.norm(rel - cal["dock"][label]["target"]))
        print(f"{label}: docked_at={docked_at} steps, true object in base frame {rel.round(3)} "
              f"vs target {np.round(cal['dock'][label]['target'], 3)}, error {err:.3f} m")
        assert docked_at is not None and err < 0.08, (label, docked_at, err)
    # Cluster: the target can has two neighbours 0.3 m away on the natural approach side (dev10 v2 geometry). The
    # docking pose selection must end with both neighbours outside the base footprint (no base contact).
    cal_g = load_calibration(target_at="grasp")
    cans = {"t": np.array([2.5, 0.0]), "n1": np.array([2.2, 0.25]), "n2": np.array([2.25, -0.2])}
    odom, omap, ctl = Odometry(), ObjectMap(), DockingController(cal_g)
    true = np.array([0.0, 0.0, 0.0])
    state = np.zeros(61, np.float32)
    ctl.start("can_of_soda", "carrying_bin", odom)
    docked_at, min_clear = None, 9.0
    x0, x1, y0, y1 = ctl.cfg.footprint
    for t in range(4000):
        dets = []
        for k, o in cans.items():
            rel = rot(-true[2]) @ (o - true[:2])
            if abs(math.atan2(rel[1], rel[0])) < math.radians(45) and 0.3 < np.linalg.norm(rel) < 6.0 and t % 5 == 0:
                dets.append({"label": "can_of_soda", "xyz_base": np.r_[rel + rng.normal(0, 0.01, 2), 0.0], "score": 1.0})
        if t % 5 == 0:
            omap.update(dets, odom, t)
        if ctl.track_id is None and omap.confirmed("can_of_soda"):
            # the orchestrator's target is the far can; force the controller onto it for this test
            ctl.track_id = min(omap.confirmed("can_of_soda"), key=lambda tr_: -float(tr_.pos[0])).track_id
        action, st = ctl.step(state, omap, odom)
        vel = action[:3] * BASE_SCALE
        state[0:3] = vel
        c, s_ = math.cos(true[2]), math.sin(true[2])
        true = np.array([true[0] + (vel[0] * c - vel[1] * s_) * DT, true[1] + (vel[0] * s_ + vel[1] * c) * DT,
                         wrap(true[2] + vel[2] * DT)])
        odom.update(vel)
        state[TRUNK], state[LARM], state[RARM] = action[A_TRUNK], action[A_LARM], action[A_RARM]
        for k in ("n1", "n2"):             # clearance of the neighbours to the (un-inflated) base footprint
            ob = rot(-true[2]) @ (cans[k] - true[:2])
            dx = max(x0 - ob[0], 0.0, ob[0] - x1)
            dy = max(y0 - ob[1], 0.0, ob[1] - y1)
            min_clear = min(min_clear, math.hypot(dx, dy))
        if st["docked"]:
            docked_at = t
            break
    rel = rot(-true[2]) @ (cans["t"] - true[:2])
    err = float(np.linalg.norm(rel - cal_g["dock"]["can_of_soda"]["target"]))
    print(f"cluster: docked_at={docked_at}, target error {err:.3f} m, min neighbour clearance to the footprint "
          f"{min_clear:.3f} m, plan {ctl._plan and {k: (round(float(v), 2) if np.isscalar(v) else v) for k, v in ctl._plan.items() if k in ('psi', 'clear', 'n_others', 'headings_blocked')}}")
    assert docked_at is not None and err < 0.08 and min_clear > 0.0, (docked_at, err, min_clear)
    # Furniture has no semantic ObjectMap track. Raw depth alone must still modify a command that would sweep the
    # measured left arm into it (regression for dev10 v7, where `_safe` only saw cans / trash_can tracks).
    odom, omap, ctl = Odometry(), ObjectMap(), DockingController(cal_g)
    ctl.start("can_of_soda", "carrying_bin", odom)
    state = np.zeros(61, np.float32)
    state[17:20], state[42:45] = (0.62, 0.28, 0.43), (0.61, -0.29, 0.52)
    state[24:26], state[49:51] = 0.05, 0.007
    ctl._active_footprint = ctl._footprint_for_state(state)
    desired = np.array([0.30, 0.0, 0.0])
    ctl._active_geometry = ctl._geometry_for_state(state)
    shielded, info = ctl._safe(desired, omap, odom, None, obstacles=np.array([[0.76, 0.36]]))
    assert info is not None and info["n_depth"] == 1 and not np.allclose(shielded, desired), (shielded, info)
    # A depth-only obstacle must also activate dock-pose planning; previously its plan was silently ignored unless a
    # second semantic task object happened to be present.
    target = Track("can_of_soda", np.array([1.5, 0.0]), hits=3, track_id=1)
    omap.tracks = [target]
    plan = ctl._plan_pose(target, omap, odom, obstacles_odom=np.array([[0.8, 0.45], [0.82, 0.45], [0.84, 0.45]]))
    assert plan is not None and plan["has_obstacles"] and plan["n_others"] == 0 and plan["n_depth"] == 3
    print("PASS docking_controller self-test")


if __name__ == "__main__":
    _self_test()
