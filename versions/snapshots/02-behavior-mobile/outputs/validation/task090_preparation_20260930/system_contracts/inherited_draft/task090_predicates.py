"""Task90 object-bound observation gate, NOT an admitted physical-success verifier.

Entry: postconditions=task090_observed. Reuses the current RGB-D detections; never reads
scene/object_scope, evaluator goals, VTT, reward or demonstration phase labels. Task1's
finger/EEF proxy is deliberately NOT a fallback. Missing inputs return unknown.

The current fruit/bin detector supplies docking points, not observed support/container
relations or persistent object/hand binding. Therefore ADMISSION_BLOCKERS cannot be
turned off by a config flag. Even a complete diagnostic sequence remains unverified.
A later admission requires a separately checked TRAIN/geometry calibration AND a real
RGB-D relation producer (including floors/countertop/sink), not new test coordinates.

Optional diagnostic protocol (fields on a current detection, never on evaluator obs):
  object_id: exact program token; track_id: stable nonempty str; hand: left|right;
  visible=True, metric_valid=True, source='rgbd', calibration_id, xyz_base[3];
  relation: {reference: exact second token, reference_track_id, reference_visible=True,
    reference_xyz_base[3], kind: support|inside|on_next_to, support_clearance_m (pick),
    placement_margin_m (place)}.
placement_margin_m is a conservative signed margin for the WHOLE object in an observed
container interior, or on observed support AND next to observed sink. It cannot be an
EEF-height or centroid-only surrogate. Positive = inside all required boundaries.

Calibration is optional and diagnostic only: source_split=train|geometry, source (a
provenance string), calibration_id, and thresholds below. No Task0/1 thresholds or
silently guessed defaults are used. No calibration is shipped by this module.
"""
from __future__ import annotations

import hashlib
import numpy as np

from harness.grounding import category
from harness.task1_perception_adapter import find, HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX

SOURCE = "task090_observed:v1"
# Majority active hands from 535 TRAIN grasp references, not observed grasp labels.
HAND = {"trash_can": "right", "half_banana": "left", "half_pomegranate": "left"}
EEF = {"left": slice(17, 20), "right": slice(42, 45)}
FINGERS = {"left": slice(24, 26), "right": slice(49, 51)}
GRIP_ACTION = {"left": 14, "right": 22}
RELATION = {"pick up from": "support", "place in": "inside", "place on next to": "on_next_to"}
ADMISSION_BLOCKERS = (
    "rgbd_object_hand_tracks_not_admitted",
    "observed_support_container_sink_relations_not_admitted",
    "train_geometry_relation_calibration_not_admitted",
)
METRIC_THRESHOLDS = ("hold_finger_min", "hold_finger_max", "open_finger_min", "eef_near_max",
                     "lift_min", "motion_min", "co_motion_max", "support_start_max",
                     "detach_min", "settle_max")
COUNT_THRESHOLDS = ("hold_frames", "settle_frames", "max_frame_gap")


def validate_mode(task_id, cfg):
    """Task90-only config guard; legacy task configurations are not rewritten."""
    mode = cfg.get("postconditions", "patch_candidate")
    if mode == "task090_observed" and task_id != 90:
        raise ValueError("postconditions task090_observed requires P2_TASK_ID=90")
    if task_id == 90:
        if mode != "task090_observed":
            raise ValueError("Task90 requires task090_observed; Task1/patch evidence is not admitted")
        if cfg.get("manip_executor", "wam") != "wam":
            raise ValueError("Task90 manipulation must use WAM, not Task1 IK primitives")
        if cfg.get("skip_failed_objects"):
            raise ValueError("Task90 may not skip required objects after an unknown outcome")
        if cfg.get("candidate_outcome_policy", "hold") != "hold":
            raise ValueError("Task90 requires candidate_outcome_policy=hold")


def _vector(value):
    x = np.asarray(value, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("invalid_metric_point")
    return x


class Task090Postconditions:
    """Same start/update lifecycle as existing postconditions, with explicit sensor inputs.

    update(..., episode=..., observation=..., detections=...) receives ONLY current
    sensor-derived detections. A stale ObjectMap or a held-object EEF estimate is not
    a detection. All evidence is unknown until the above admission work is completed.
    """

    def __init__(self, calibration=None):
        self.calibration = dict(calibration or {})
        self.calibration_error = self._check_calibration()
        self.reset_episode()

    def _check_calibration(self):
        c = self.calibration
        if not c:
            return "missing_relation_calibration"
        if (c.get("source_split") not in ("train", "geometry") or not c.get("source")
                or not isinstance(c.get("calibration_id"), str) or not c["calibration_id"]
                or c.get("uses_validation", False) is not False):
            return "invalid_calibration_provenance"
        for k in METRIC_THRESHOLDS:
            if type(c.get(k)) not in (int, float) or not np.isfinite(c[k]) or c[k] <= 0:
                return "invalid_calibration_threshold:" + k
        for k in COUNT_THRESHOLDS:
            if type(c.get(k)) is not int or c[k] < (1 if k == "max_frame_gap" else 2):
                return "invalid_calibration_threshold:" + k
        if not (c["hold_finger_min"] < c["hold_finger_max"] < c["open_finger_min"]
                and c["detach_min"] > c["eef_near_max"] and c["motion_min"] > c["co_motion_max"]):
            return "inconsistent_calibration_thresholds"
        return None

    def reset_episode(self):
        self.key = self.episode = None
        self.last_step = None
        self.seen_frames = set()
        self._clear_sequence()

    def _clear_sequence(self):
        self.baseline = self.previous = self.release = None
        self.binding = None
        self.hold_n = self.settle_n = 0
        self.held_motion = False

    def start(self, key, skill, objects, state, step, *, episode=None):
        if episode != self.episode:
            self.reset_episode()
        self.episode, self.key, self.skill = episode, key, skill
        self.objects = tuple(objects)
        self.hand = HAND.get(category(objects[0])) if objects else None
        self.started = int(step)
        self._clear_sequence()  # release/grasp evidence must be observed anew, even on retry

    def _evidence(self, step, reason, *, detail=None, candidate=False):
        return {"verified": False, "satisfied": False, "failed": False, "status": "unknown",
                "verifier_kind": "perception_predicates", "source": SOURCE, "key": self.key,
                "observation_id": int(step), "episode": self.episode, "hand": self.hand,
                "objects": list(self.objects), "reason": reason,
                "admitted": False, "admission_blockers": list(ADMISSION_BLOCKERS),
                "detail": {**(detail or {}), "candidate_satisfied": bool(candidate)}}

    def update(self, state, action, step, *, episode=None, observation=None, detections=None):
        if self.key is None:
            return None
        detail = {"detections_received": len(detections) if isinstance(detections, (list, tuple)) else 0,
                  "calibration_blocker": self.calibration_error}
        try:
            if episode is None or episode != self.episode:
                raise ValueError("episode_mismatch")
            if self.skill not in RELATION or len(self.objects) != 2 or self.hand is None:
                raise ValueError("unsupported_task090_binding")
            if type(step) is not int or step < self.started or (self.last_step is not None and step <= self.last_step):
                raise ValueError("nonmonotonic_frame")
            gap = None if self.last_step is None else step - self.last_step
            self.last_step = step
            if not isinstance(observation, dict):
                raise ValueError("missing_rgbd_observation")
            rgb, depth, cam = (find(observation, s) for s in (HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX))
            if rgb is None or depth is None or cam is None:
                raise ValueError("missing_rgb_depth_or_camera_pose")
            rgb, depth, cam = np.asarray(rgb), np.asarray(depth), np.asarray(cam)
            if (rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or depth.squeeze().shape != rgb.shape[:2]
                    or not np.isfinite(cam).all() or cam.size < 7 or not np.isfinite(depth).any()
                    or not np.any(depth > 0)):
                raise ValueError("invalid_rgbd_or_camera_pose")
            # A new control-step number / camera pose alone cannot make an old image fresh.
            h = hashlib.sha256()
            for x in (rgb[..., :3], depth):
                h.update(str((x.shape, x.dtype.str)).encode())
                h.update(np.ascontiguousarray(x).tobytes())
            digest = h.hexdigest()
            if digest in self.seen_frames:
                raise ValueError("repeated_sensor_frame")
            self.seen_frames.add(digest)
            if not isinstance(detections, (list, tuple)):
                raise ValueError("missing_current_detections")
            matches = [d for d in detections if isinstance(d, dict)
                       and (d.get("object_id") == self.objects[0] or d.get("label") == category(self.objects[0]))]
            if len(matches) != 1:
                raise ValueError("target_missing_or_ambiguous")
            d = matches[0]
            if d.get("object_id") != self.objects[0] or not isinstance(d.get("track_id"), str) or not d["track_id"]:
                raise ValueError("missing_exact_target_track")
            if d.get("hand") != self.hand:
                raise ValueError("hand_binding_mismatch")
            if d.get("visible") is not True or d.get("metric_valid") is not True or d.get("source") != "rgbd":
                raise ValueError("target_not_visibly_metric_rgbd")
            if self.calibration_error:
                raise ValueError(self.calibration_error)
            c = self.calibration
            if d.get("calibration_id") != c["calibration_id"]:
                raise ValueError("calibration_mismatch")
            if gap is not None and gap > c["max_frame_gap"]:
                raise ValueError("frame_gap")
            r = d.get("relation")
            if not isinstance(r, dict) or r.get("reference") != self.objects[1] or r.get("kind") != RELATION[self.skill]:
                raise ValueError("missing_target_reference_relation")
            if (r.get("reference_visible") is not True or not isinstance(r.get("reference_track_id"), str)
                    or not r["reference_track_id"]):
                raise ValueError("reference_not_observed")
            binding = (d["track_id"], r["reference_track_id"])
            if self.binding is not None and self.binding != binding:
                raise ValueError("track_binding_changed")
            self.binding = binding
            s, a = np.asarray(state, float), np.asarray(action, float)
            if s.shape != (61,) or a.shape != (23,) or not np.isfinite(s).all() or not np.isfinite(a).all():
                raise ValueError("invalid_proprio_or_applied_action")
            obj, ref, eef = _vector(d.get("xyz_base")), _vector(r.get("reference_xyz_base")), s[EEF[self.hand]]
            row = {"obj_ref": obj - ref, "eef_ref": eef - ref, "obj_eef": obj - eef}
            fs = float(s[FINGERS[self.hand]].sum())
            holding = (a[GRIP_ACTION[self.hand]] < 0 and c["hold_finger_min"] <= fs <= c["hold_finger_max"]
                       and np.linalg.norm(row["obj_eef"]) <= c["eef_near_max"])
            opened = a[GRIP_ACTION[self.hand]] > 0 and fs >= c["open_finger_min"]
            detail.update(target=d["object_id"], track_id=d["track_id"], reference=r["reference"],
                          finger_sum=fs, holding_joint_evidence=bool(holding))
            if self.skill == "pick up from":
                clearance = float(r["support_clearance_m"])
                if not np.isfinite(clearance):
                    raise ValueError("invalid_support_clearance")
                if self.baseline is None:
                    if abs(clearance) > c["support_start_max"]:
                        raise ValueError("no_observed_supported_start")
                    self.baseline = {**row, "clearance": clearance}
                candidate = self._held_motion(row, holding)
                lift = clearance - self.baseline["clearance"]
                candidate = candidate and lift >= c["lift_min"] and row["obj_ref"][2] - self.baseline["obj_ref"][2] >= c["lift_min"]
                detail.update(observed_lift_m=lift, holding_frames=self.hold_n)
            else:
                margin = float(r["placement_margin_m"])
                if not np.isfinite(margin):
                    raise ValueError("invalid_placement_margin")
                if self.release is None:
                    if holding:
                        self._held_motion(row, holding)
                    if opened and self.held_motion and self.hold_n >= c["hold_frames"]:
                        self.release = {"row": row, "origin": None}
                    elif not holding:
                        self.hold_n = 0
                        self.held_motion = False
                        self.baseline = None
                    candidate = False
                else:
                    detached = np.linalg.norm(row["obj_eef"]) >= c["detach_min"]
                    if not opened or not detached or margin < 0:
                        self.settle_n = 0
                        self.release["origin"] = None
                    else:
                        origin = self.release["origin"]
                        if origin is None or np.linalg.norm(row["obj_ref"] - origin) > c["settle_max"]:
                            self.release["origin"] = row["obj_ref"].copy()
                            self.settle_n = 1
                        else:
                            self.settle_n += 1
                    candidate = self.settle_n >= c["settle_frames"]
                detail.update(placement_margin_m=margin, released=self.release is not None, stable_frames=self.settle_n)
            self.previous = row
            return self._evidence(step, "relation_sequence_not_admitted" if candidate else "observing_object_relation",
                                  detail=detail, candidate=candidate)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self._clear_sequence()
            return self._evidence(step, str(error), detail=detail)

    def _held_motion(self, row, holding):
        c = self.calibration
        if not holding:
            self.hold_n = 0
            self.held_motion = False
            return False
        if self.baseline is None:
            self.baseline = row
        if self.previous is not None and np.linalg.norm(row["obj_eef"] - self.previous["obj_eef"]) > c["co_motion_max"]:
            self.hold_n = 0
            self.held_motion = False
            return False
        self.hold_n += 1
        moved = (np.linalg.norm(row["obj_ref"] - self.baseline["obj_ref"]) >= c["motion_min"]
                 and np.linalg.norm(row["eef_ref"] - self.baseline["eef_ref"]) >= c["motion_min"]
                 and np.linalg.norm(row["obj_eef"] - self.baseline["obj_eef"]) <= c["co_motion_max"])
        self.held_motion = self.held_motion or bool(moved)
        return self.held_motion and self.hold_n >= c["hold_frames"]
