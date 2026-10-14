"""Explicitly authorized base-only reset prepositioning for WAM manipulation tests.

No task object or upper-body joint is set. This is not navigation success.
"""
import json
import math

import numpy as np
import omnigibson as og
import torch

from harness.docking_controller import load_calibration, rot
from harness.privileged_base_navigation import ASSET, GOAL, ROOT, pose_matrix
from harness.privileged_base_rgbd_wrapper import PrivilegedBaseRGBDWrapper


def candidate_stances(base_pose, target_pose, scale, center, extent, dock_target, obstacles):
    """Object-relative ring, ranked from original approach; conservative base footprint."""
    rb = pose_matrix(target_pose)
    center, half = np.asarray(center) * scale, np.asarray(extent) * scale / 2
    target = np.asarray(target_pose[:3])
    delta = target[:2] - np.asarray(base_pose[:2])
    initial_yaw = math.atan2(delta[1], delta[0]) - math.atan2(dock_target[1], dock_target[0])
    offsets = [0] + [sign * i for i in range(1, 12) for sign in (1, -1)] + [12]
    records = []
    for offset in offsets:
        yaw = initial_yaw + offset * math.pi / 12
        toward = rb.T @ np.r_[-rot(yaw) @ dock_target, 0.]
        axis = int(np.argmax(np.abs(toward[:2]) / half[:2]))
        local = center.copy()
        local[axis] += math.copysign(half[axis], toward[axis])
        local[2] += half[2]
        rim = target + rb @ local
        xy = rim[:2] - rot(yaw) @ dock_target
        # Base-only envelope; upper-body motion still goes through the existing shield.
        corners = (rot(yaw) @ np.array([[-.42, -.44], [-.42, .44], [.42, -.44], [.42, .44]]).T).T + xy
        lo, hi = corners.min(0), corners.max(0)
        toward_actual = rb.T @ (np.r_[xy, base_pose[2]] - target) - center
        actual_axis = int(np.argmax(np.abs(toward_actual[:2]) / half[:2]))
        consistent = actual_axis == axis and math.copysign(1., toward_actual[axis]) == math.copysign(1., toward[axis])
        blockers = ([] if consistent else ["inconsistent_rim_side"]) + [name for name, lower, upper in obstacles
                    if np.all(hi > np.asarray(lower)[:2]) and np.all(np.asarray(upper)[:2] > lo)]
        records.append({"yaw": yaw, "xy": xy.tolist(), "rim_local": local.tolist(), "blocked_by": blockers})
    return records


class PrepositionedBaseRGBDWrapper(PrivilegedBaseRGBDWrapper):
    def set_preposition_enabled(self, enabled):
        # The evaluator arms this only after loading the requested instance.
        self._preposition_enabled = bool(enabled)

    def reset(self):
        obs, info = super().reset()
        if not getattr(self, "_preposition_enabled", False):
            return obs, info
        robot = self._velocity_robot
        if any(obj is not None for obj in getattr(robot, "_ag_obj_in_hand", {}).values()):
            raise ValueError("base prepositioning requires empty hands at normal reset")
        entity = self.env.task.object_scope[GOAL]
        before_base = self._pose(robot)
        before_target = self._pose(entity)
        scale = np.asarray(entity.scale.detach().cpu() if hasattr(entity.scale, "detach") else entity.scale, float)
        box = json.loads(ASSET.read_text())["link_bounding_boxes"]["base_link"]["visual"]["axis_aligned"]
        center = np.asarray(box["transform"], float)[:3, 3]
        calibration = ROOT / "outputs/validation/task090_docking_calibration_train_20260930"
        target = load_calibration(path=calibration, target_at="start")["dock"]["trash_can"]["target"]
        obstacles = []
        for obj in self.env.scene.objects:
            if obj is robot or getattr(obj, "category", "") in ("floors", "floor"):
                continue
            lower, upper = obj.aabb
            lower = np.asarray(lower.detach().cpu() if hasattr(lower, "detach") else lower, float)
            upper = np.asarray(upper.detach().cpu() if hasattr(upper, "detach") else upper, float)
            if not np.isfinite(np.r_[lower, upper]).all():
                raise ValueError("nonfinite preposition obstacle AABB")
            if lower[2] < before_base[2] + .42 and upper[2] > before_base[2] + .08:
                obstacles.append((obj.name, lower.tolist(), upper.tolist()))
        candidates = candidate_stances(before_base, before_target, scale, center,
                                       np.asarray(box["extent"]), target, obstacles)
        ignored = [robot.prim_path] + [obj.prim_path for obj in self.env.scene.objects
                                      if getattr(obj, "category", "") in ("floors", "floor")]
        for candidate in candidates:
            candidate["aabb_blocked_by"] = candidate["blocked_by"]
            hits = []
            def report_hit(hit):
                path = str(getattr(hit, "rigid_body", "unknown_collider"))
                if not any(path == prefix or path.startswith(prefix + "/") for prefix in ignored):
                    hits.append(path)
                return True
            yaw = candidate["yaw"]
            og.sim.psqi.overlap_box(halfExtent=[.42, .44, .17],
                                   pos=[*candidate["xy"], float(before_base[2] + .25)],
                                   rot=[0., 0., math.sin(yaw/2), math.cos(yaw/2)], reportFn=report_hit)
            candidate["blocked_by"] = sorted(set(hits))
            if "inconsistent_rim_side" in candidate["aabb_blocked_by"]:
                candidate["blocked_by"].append("inconsistent_rim_side")
        chosen = next((candidate for candidate in candidates if not candidate["blocked_by"]), None)
        receipt = {"protocol": "prepositioned-base-only-v3", "goal": GOAL,
                   "collision_query": "PhysX oriented overlap_box against actual collision shapes; base only",
                   "before_base_pose": before_base.tolist(), "target_pose_unchanged": before_target.tolist(),
                   "train_dock_target_xy": target.tolist(), "candidates": candidates,
                   "selected": chosen, "physics_setup_steps": 0,
                   "scope": "base reset placement only, not navigation or grasp success"}
        if chosen is None:
            print("P2_PREPOSITION " + json.dumps(receipt), flush=True)
            raise ValueError("no collision-clear conservative base preposition candidate")
        yaw = chosen["yaw"]
        position = torch.tensor([*chosen["xy"], before_base[2]], dtype=torch.float32)
        quaternion = torch.tensor([0., 0., math.sin(yaw/2), math.cos(yaw/2)], dtype=torch.float32)
        robot.set_position_orientation(position=position, orientation=quaternion)
        velocities = robot.get_joint_velocities().clone()
        velocities[robot.base_idx] = 0
        robot.set_joint_velocities(velocities)
        self._interval_velocity = self._interval_end = None
        # Refresh camera transforms without a physics step or fabricated elapsed time.
        for _ in range(3):
            og.sim.render()
        after_target = self._pose(entity)
        if not np.allclose(before_target, after_target, atol=1e-6, rtol=0):
            raise ValueError("target moved during base-only prepositioning")
        receipt["after_base_pose"] = self._pose(robot).tolist()
        self._preposition_receipt = receipt
        print("P2_PREPOSITION " + json.dumps(receipt), flush=True)
        self._privbase_step = 0
        return self.get_obs()
