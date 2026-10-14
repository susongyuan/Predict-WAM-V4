"""Policy-side adapter between evaluator observations and a task's onboard detector (docking inputs).

Detector module: `cfg["module"]` (orchestrate.perception.module) if given, else by task (harness.task_profile
P2_TASK_ID): Task 0 -> harness.radio_perception, every other task -> harness.object_perception (Task 1, unchanged).
Both expose detect_from_observation(rgb, depth, cam_rel_pose, state, **overrides) -> (detections, visible_fn) and
obstacle_points(depth, cam_rel_pose, state); the module is imported lazily (it may be missing at import time).
Detections flagged `held` (object in a closed hand) are never navigation targets and are dropped here.

Only onboard observations are used: head RGB (+ depth_linear when the RGB-D wrapper is active), and the head camera
pose relative to the robot base from the proprioceptive `cam_rel_poses` entry. Returns detections in the robot base
frame and a visibility predicate used by the object map to age out tracks that should have been seen.

`fake` (CPU tests only) replaces the detector with a callable(obs, state) -> detections; `fake_obstacles` (CPU tests
only, used together with `fake`) replaces obstacle_points with a callable(obs, state) -> base-frame xy points.
"""
from __future__ import annotations

import numpy as np

HEAD_RGB_SUFFIX = ":zed_link:Camera:0::rgb"
HEAD_DEPTH_SUFFIX = ":zed_link:Camera:0::depth_linear"
CAM_POSE_SUFFIX = "::cam_rel_poses"


def find(obs, suffix):
    for k in obs:
        if isinstance(k, str) and k.endswith(suffix):
            return obs[k]
    return None


DETECTOR_BY_TASK = {0: "harness.radio_perception"}
DEFAULT_DETECTOR = "harness.object_perception"


def detector_module_name(cfg=None, task_id=None):
    """Detector module for the task: explicit cfg['module'] wins, else the per-task default."""
    if cfg and cfg.get("module"):
        return str(cfg["module"])
    if task_id is None:
        from harness import task_profile
        task_id = task_profile.TASK_ID
    return DETECTOR_BY_TASK.get(int(task_id), DEFAULT_DETECTOR)


def _not_held(dets):
    return [d for d in dets if not d.get("held")]


class TaskPerception:
    def __init__(self, cfg=None, device=None, encoder=None, fake=None, fake_obstacles=None, fake_bands=None):
        self.cfg = dict(cfg or {})
        self.device, self.encoder = device, encoder
        self.fake = fake
        self.fake_obstacles = fake_obstacles     # CPU tests only: callable(obs, state) -> base-frame xy (N, 2)
        self.fake_bands = fake_bands             # CPU tests only: callable(obs, state, edges) -> obstacle_bands() rows
        self._op = None
        self.last = {}
        self.module_name = detector_module_name(self.cfg)

    def _module(self):
        if self._op is None:
            import importlib
            self._op = importlib.import_module(self.module_name)
        return self._op

    def detect(self, obs, state):
        if self.fake is not None:
            dets = list(self.fake(obs, state))
            return _not_held(dets), None
        op = self._module()
        rgb = find(obs, HEAD_RGB_SUFFIX)
        if rgb is None:
            return [], None
        depth = find(obs, HEAD_DEPTH_SUFFIX)
        cam = find(obs, CAM_POSE_SUFFIX)
        dets, visible_fn = op.detect_from_observation(np.asarray(rgb), None if depth is None else np.asarray(depth),
                                                      None if cam is None else np.asarray(cam, np.float64),
                                                      np.asarray(state, np.float64), **self.cfg.get("detector", {}))
        kept = _not_held(dets)
        self.last = {"n": len(dets)} if len(kept) == len(dets) else {"n": len(dets), "held": len(dets) - len(kept)}
        return kept, visible_fn

    def obstacle_bands(self, obs, state, edges):
        """Height-banded depth obstacles [(z_lo, z_hi, is_support, xy_base (N, 2))] for DockConfig.height_bands (Task 0):
        one detector obstacle_points(..., z_band=(z_lo, z_hi)) call per band. A detector without z_band support yields
        one full-height band (conservative: every body part then checks every point)."""
        empty = [(lo, hi, sup, np.zeros((0, 2))) for lo, hi, sup in edges]
        if self.fake is not None:
            return self.fake_bands(obs, state, edges) if getattr(self, "fake_bands", None) is not None else empty
        op = self._module()
        depth, cam = find(obs, HEAD_DEPTH_SUFFIX), find(obs, CAM_POSE_SUFFIX)
        if depth is None or cam is None or not hasattr(op, "obstacle_points"):
            return empty
        import inspect
        args = (np.asarray(depth), np.asarray(cam, np.float64), np.asarray(state, np.float64))
        if "z_band" not in inspect.signature(op.obstacle_points).parameters:
            pts = np.asarray(op.obstacle_points(*args), np.float64).reshape(-1, 2)
            self.last["bands"] = "full_height_fallback"
            return [(min(e[0] for e in edges), max(e[1] for e in edges), False, pts)]
        return [(lo, hi, sup, np.asarray(op.obstacle_points(*args, z_band=(lo, hi)), np.float64).reshape(-1, 2))
                for lo, hi, sup in edges]

    def obstacles(self, obs, state):
        """Base-frame xy of depth points in the furniture height band (empty without depth or detector support)."""
        if self.fake is not None:
            if self.fake_obstacles is not None:
                return np.asarray(self.fake_obstacles(obs, state), np.float64).reshape(-1, 2)
            return np.zeros((0, 2))
        op = self._module()
        depth, cam = find(obs, HEAD_DEPTH_SUFFIX), find(obs, CAM_POSE_SUFFIX)
        if depth is None or cam is None or not hasattr(op, "obstacle_points"):
            return np.zeros((0, 2))
        return np.asarray(op.obstacle_points(np.asarray(depth), np.asarray(cam, np.float64), np.asarray(state, np.float64)),
                          np.float64).reshape(-1, 2)
