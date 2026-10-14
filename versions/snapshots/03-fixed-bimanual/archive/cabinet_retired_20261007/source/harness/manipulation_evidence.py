"""Fail-closed onboard manipulation evidence; no production admission is shipped.

Reuse Task090's calibration checks and held-motion primitive, not Task0/1 EEF-only
success proxies. Operations are pick / transport / place. A Task090 adapter below
checks its exact TRAIN contract vocabulary without rewriting checkpoint keys.

Trust boundary: detections passed by a caller are DIAGNOSTIC ONLY. Verification
requires this gate to call an independently reviewed producer.measure(packet),
whose source/version, calibration and applicability are pinned by a reviewed
manifest digest in TRUSTED_REVIEW_SHA256. The shipped set is EMPTY. A configuration
boolean, producer output, synthetic test or arbitrary manifest cannot admit it.
No evaluator object/scene/reward fields enter the producer packet. This is an
in-process integrity gate, not a sandbox against malicious Python code.

A trusted sensor ingress must supply captured_at/now from one monotonic clock.
RGB-D content identity is recomputed here. Repeated pixels never become fresh just
because an observation ID, camera pose or timestamp changed. Keep updating the
same gate during navigation/HOLD to preserve carry continuity; start() changes
only the contract sequence, never the episode's target/hand identity map.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import hashlib
import inspect
import json
from pathlib import Path
import secrets

import numpy as np

from harness.core import _number, _text
from harness.task090_predicates import (
    Task090Postconditions, sensor_frame_sha256, _vector, EEF, FINGERS, GRIP_ACTION,
    HAND, BINDINGS, RELATION,
)
from harness.grounding import category
from harness.task1_perception_adapter import HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX

SOURCE = "manipulation_evidence:v1"
# Change only after independent review of REAL onboard labels, error bounds and
# applicability. No CLI/config/env switch populates this trust root.
TRUSTED_REVIEW_SHA256 = frozenset()
TIME_THRESHOLDS = ("max_age_s", "min_period_s", "max_period_s", "hold_duration_s", "settle_duration_s")
DETECTION_FIELDS = frozenset((
    "object_id", "track_id", "hand", "episode", "observation_id", "sensor_frame_sha256",
    "contract_id", "producer_version", "visible", "metric_valid", "source", "calibration_id",
    "xyz_base", "position_error_m", "eef_error_m", "relation",
))
RELATION_FIELDS = frozenset((
    "reference", "reference_track_id", "reference_visible", "reference_xyz_base", "reference_error_m",
    "kind", "support_clearance_m", "clearance_error_m", "placement_margin_m", "margin_error_m",
    "support", "support_track_id", "support_visible", "support_xyz_base", "support_error_m",
))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _metric(mapping, key, *, nonnegative=False):
    x = mapping.get(key)
    if not _number(x) or (nonnegative and x < 0):
        raise ValueError("missing_or_invalid_metric:" + key)
    return float(x)


@dataclass(frozen=True)
class Binding:
    """Concrete contract attempt and observed identities, not category-only names."""
    episode: int
    contract_id: str
    key: str
    started_step: int
    operation: str
    target: str
    track_id: str
    hand: str
    reference: str
    reference_track_id: str
    relation_kind: str
    support: str | None = None
    support_track_id: str | None = None

    def validate(self):
        if (type(self.episode) is not int or type(self.started_step) is not int or self.started_step < 0
                or self.hand not in EEF or self.operation not in ("pick", "transport", "place")
                or any(not _text(getattr(self, k)) for k in (
                    "contract_id", "key", "target", "track_id", "reference", "reference_track_id"))):
            raise ValueError("invalid_contract_binding")
        allowed = {"pick": ("support",), "transport": ("destination",),
                   "place": ("inside", "support", "on_next_to")}
        if self.relation_kind not in allowed[self.operation]:
            raise ValueError("operation_relation_mismatch")
        if self.relation_kind == "on_next_to":
            if not _text(self.support) or not _text(self.support_track_id):
                raise ValueError("missing_placement_support_binding")
        elif self.support is not None or self.support_track_id is not None:
            raise ValueError("unexpected_support_binding")
        ids = [self.track_id, self.reference_track_id] + ([self.support_track_id] if self.support else [])
        names = [self.target, self.reference] + ([self.support] if self.support else [])
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("aliased_object_reference_tracks")


class ManipulationEvidence(Task090Postconditions):
    """Minimal verifier + receipt consumer. No motion or evaluator access.

    producer.measure receives only {rgb, depth, camera_pose, state, applied_action,
    binding, observation_id, captured_at, sensor_frame_sha256}; returns exactly one
    detection with DETECTION_FIELDS. Error bounds must cover total propagated
    tracking/calibration error; placement margin covers the WHOLE object against
    observed geometry (support AND adjacency for on_next_to), never just centroid.
    A producer's claimed satisfied/verified/candidate fields are rejected.

    calibration extends Task090 thresholds with TIME_THRESHOLDS; no defaults.
    scope is an exact independently reviewed robot/sensor/workspace/geometry scope.
    consume() accepts only the latest unmodified receipt from THIS instance and
    rechecks expected binding, observation, pixels, freshness and admission.
    """

    def __init__(self, calibration=None, *, producer=None, review_path=None, scope=None):
        self.producer = producer
        self.review_path = None if review_path is None else Path(review_path)
        self.scope = copy.deepcopy(scope or {})
        self._review = None
        self._review_error = "missing_independent_admission"
        super().__init__(copy.deepcopy(calibration))
        if self.calibration_error is None:
            c = self.calibration
            if any(not _number(c.get(k)) or c[k] <= 0 for k in TIME_THRESHOLDS):
                self.calibration_error = "missing_or_invalid_temporal_calibration"
            elif c["min_period_s"] > c["max_period_s"] or c["max_frame_gap"] != 1:
                self.calibration_error = "continuous_observation_calibration_required"
        self._load_review()

    def reset_episode(self):
        super().reset_episode()
        self.contract = None
        self._identities = {}
        self._known_tracks = {}
        self._carry = None
        self._last_captured = None
        self._issued = None
        self._start_error = None
        self._carry_required = False
        self._hold_since = self._settle_since = None
        self._blocked_identity = False

    def _load_review(self):
        """Pinned real artifacts only. Parsing arbitrary JSON is never admission."""
        if self.review_path is None:
            return
        try:
            if _digest_file(self.review_path) not in TRUSTED_REVIEW_SHA256:
                raise ValueError("untrusted_admission_review")
            if self.review_path.stat().st_size > 1024 * 1024:
                raise ValueError("oversized_admission_review")
            r = json.loads(self.review_path.read_text())
            if r.get("schema") != "onboard_manipulation_review_v1" or r.get("label_source") != "independent_onboard_annotation":
                raise ValueError("invalid_independent_review")
            if r.get("split") not in ("train", "geometry") or r.get("synthetic") is not False:
                raise ValueError("inadmissible_review_split_or_synthetic_data")
            kinds = set()
            for item in r["artifacts"]:
                if _digest_file(self.review_path.parent / item["path"]) != item["sha256"]:
                    raise ValueError("review_artifact_hash_mismatch")
                kinds.add(item["kind"])
            if not {"independent_physical_labels", "calibration_validation", "applicability"} <= kinds:
                raise ValueError("missing_independent_review_artifacts")
            self._review = r
            self._review_error = None
        except (OSError, ValueError, KeyError, TypeError):
            self._review_error = "invalid_or_untrusted_admission_review"

    def _admission_blocker(self):
        if self.calibration_error:
            return self.calibration_error
        if self._review is None:
            return self._review_error
        r = self._review
        try:
            if self.producer is None or not callable(getattr(self.producer, "measure", None)):
                return "missing_reviewed_relation_producer"
            path = inspect.getsourcefile(type(self.producer))
            if not path or _digest_file(path) != r["producer_sha256"]:
                return "producer_source_mismatch"
            if getattr(self.producer, "version", None) != r["producer_version"]:
                return "producer_version_mismatch"
            if _digest_file(inspect.getsourcefile(Task090Postconditions)) != r["predicate_sha256"]:
                return "predicate_source_mismatch"
            if hashlib.sha256(_canonical(self.calibration).encode()).hexdigest() != r["calibration_sha256"]:
                return "calibration_review_mismatch"
            if (set(self.scope) != {"robot_version", "sensor_setup", "workspace_regime", "object_geometry"}
                    or any(not _text(v) for v in self.scope.values()) or self.scope != r["scope"]):
                return "outside_reviewed_scope"
            if self.contract and (self.contract.operation not in r["operations"]
                                  or self.contract.relation_kind not in r["relations"]):
                return "operation_or_relation_not_admitted"
        except (OSError, ValueError, KeyError, TypeError):
            return "incomplete_independent_admission"
        return None

    def admission_status(self):
        blocker = self._admission_blocker()
        return {"source": SOURCE, "admitted": blocker is None,
                "admission_blockers": [] if blocker is None else [blocker]}

    def start(self, binding: Binding):
        self._issued = None
        binding.validate()
        if binding.episode != self.episode:
            self.reset_episode()
        self.episode, self.key, self.hand = binding.episode, binding.key, binding.hand
        self.contract = binding
        self.objects = (binding.target, binding.reference)
        self.started = binding.started_step
        self._issued = None
        self._clear_sequence()  # resets sequence only; NEVER the sticky identity map
        self._hold_since = self._settle_since = None
        self._start_error = None
        named_tracks = [(binding.target, binding.track_id), (binding.reference, binding.reference_track_id)]
        if binding.support is not None:
            named_tracks.append((binding.support, binding.support_track_id))
        for name, track in named_tracks:
            if ((name in self._known_tracks and self._known_tracks[name] != track)
                    or any(n != name and t == track for n, t in self._known_tracks.items())):
                self._start_error = "object_or_reference_track_rebound"
            else:
                self._known_tracks[name] = track
        identity = (binding.track_id, binding.hand)
        if binding.target in self._identities and self._identities[binding.target] != identity:
            self._start_error = "target_or_hand_rebound_across_contracts"
        elif any(track == binding.track_id and name != binding.target
                 for name, (track, _) in self._identities.items()):
            self._start_error = "track_reused_for_another_target"
        else:
            self._identities[binding.target] = identity
        self._carry_required = binding.operation in ("transport", "place")
        if self._carry_required and (self._carry is None or self._carry[:3] != (binding.target, *identity)
                                     or self._carry[4] != "held"):
            self._start_error = self._start_error or "missing_verified_pick_continuity"
        # Once identity is lost while carrying, a new contract cannot silently reacquire.
        if self._blocked_identity:
            self._start_error = self._start_error or "carry_identity_lost_requires_recovery"

    def _emit_result(self, step, captured_at, digest, candidate, reason, detail, *, trusted=False):
        blocker = self._admission_blocker()
        value = candidate if trusted and blocker is None else None
        result = {"source": SOURCE, "verifier_kind": "perception_predicates",
                  "verified": value is not None, "satisfied": value is True, "failed": False,
                  "status": "unknown" if value is None else ("true" if value else "false"),
                  "value": value, "candidate_satisfied": candidate is True,
                  "reason": blocker if candidate is not None and blocker else reason,
                  "admitted": blocker is None, "admission_blockers": [] if blocker is None else [blocker],
                  "key": self.key, "episode": self.episode, "hand": self.hand,
                  "binding": asdict(self.contract), "observation_id": step,
                  "captured_at": captured_at, "sensor_frame_sha256": digest,
                  "producer_version": getattr(self.producer, "version", None),
                  "detail": detail, "receipt": secrets.token_hex(16)}
        self._issued = _canonical(result)
        return result

    def _invalidate(self):
        if self._carry is not None or self._carry_required:
            self._blocked_identity = True
        self._carry = None
        self._clear_sequence()
        self._hold_since = self._settle_since = None

    def update(self, state, action, step, *, episode, observation, captured_at, now, detections=None):
        """Caller detections can exercise candidates, NEVER verification or a carry lease."""
        if self.contract is None:
            return None
        b, c = self.contract, self.calibration
        digest = None
        detail = {}
        try:
            if self._start_error:
                raise ValueError(self._start_error)
            if type(episode) is not int or episode != b.episode:
                raise ValueError("episode_mismatch")
            if type(step) is not int or step <= b.started_step:
                raise ValueError("not_a_post_action_observation")
            if self.last_step is not None and step != self.last_step + 1:
                raise ValueError("nonconsecutive_observation")
            if self.calibration_error:
                raise ValueError(self.calibration_error)
            if not _number(captured_at) or not _number(now) or not 0 <= now - captured_at <= c["max_age_s"]:
                raise ValueError("stale_or_future_observation")
            if self._last_captured is not None and not c["min_period_s"] <= captured_at - self._last_captured <= c["max_period_s"]:
                raise ValueError("observation_frequency_out_of_scope")
            if not isinstance(observation, dict):
                raise ValueError("missing_onboard_observation")
            # Exact sensor selection: reject ambiguous duplicate cameras; exclude every other field.
            arrays = []
            for suffix in (HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX):
                matches = [v for k, v in observation.items() if isinstance(k, str) and k.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError("missing_or_ambiguous_onboard_sensor")
                arrays.append(np.asarray(matches[0]))
            rgb, depth, cam = arrays
            s, a = np.asarray(state, float), np.asarray(action, float)
            if (rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or not np.isfinite(rgb).all()
                    or depth.squeeze().shape != rgb.shape[:2] or not np.any(np.isfinite(depth) & (depth > 0))
                    or cam.shape not in ((7,), (21,)) or not np.isfinite(cam).all() or np.linalg.norm(cam[3:7]) == 0
                    or s.shape != (61,) or a.shape != (23,) or not np.isfinite(s).all() or not np.isfinite(a).all()):
                raise ValueError("invalid_onboard_sensor_shapes_or_values")
            digest = sensor_frame_sha256(rgb, depth)
            if digest in self.seen_frames:
                raise ValueError("repeated_sensor_frame")
            # Count sensor receipt even if the producer cannot track; later gaps cannot be hidden.
            self.last_step, self._last_captured = step, captured_at
            self.seen_frames.add(digest)
            direct = detections is not None
            if not direct:
                if self.producer is None:
                    raise ValueError("missing_relation_producer")
                packet = dict(rgb=rgb.copy(), depth=depth.copy(), camera_pose=cam.copy(), state=s.copy(),
                              applied_action=a.copy(), binding=asdict(b), observation_id=step,
                              captured_at=captured_at, sensor_frame_sha256=digest)
                detections = self.producer.measure(packet)
            if not isinstance(detections, (list, tuple)) or len(detections) != 1:
                raise ValueError("target_missing_or_ambiguous")
            d = detections[0]
            if not isinstance(d, dict) or set(d) != DETECTION_FIELDS:
                raise ValueError("unexpected_or_missing_measurement_fields")
            expected = dict(object_id=b.target, track_id=b.track_id, hand=b.hand, episode=b.episode,
                            observation_id=step, contract_id=b.contract_id, sensor_frame_sha256=digest)
            if (any(d[k] != v for k, v in expected.items()) or type(d["episode"]) is not int
                    or type(d["observation_id"]) is not int):
                raise ValueError("measurement_identity_or_frame_mismatch")
            if d["visible"] is not True or d["metric_valid"] is not True or d["source"] != "rgbd":
                raise ValueError("target_not_observed_rgbd")
            if not _text(d["producer_version"]) or (not direct and d["producer_version"] != getattr(self.producer, "version", None)):
                raise ValueError("measurement_producer_version_mismatch")
            if d["calibration_id"] != c["calibration_id"]:
                raise ValueError("measurement_calibration_mismatch")
            r = d["relation"]
            if not isinstance(r, dict) or set(r) - RELATION_FIELDS:
                raise ValueError("unexpected_relation_fields")
            if (r.get("reference") != b.reference or r.get("reference_track_id") != b.reference_track_id
                    or r.get("kind") != b.relation_kind or r.get("reference_visible") is not True):
                raise ValueError("missing_bound_reference_relation")
            obj, ref, eef = _vector(d["xyz_base"]), _vector(r.get("reference_xyz_base")), s[EEF[b.hand]]
            pe = _metric(d, "position_error_m", nonnegative=True)
            ee = _metric(d, "eef_error_m", nonnegative=True)
            re = _metric(r, "reference_error_m", nonnegative=True)
            row = {"obj_ref": obj-ref, "eef_ref": eef-ref, "obj_eef": obj-eef,
                   "co_error": pe+ee, "rel_error": pe+re, "settle_relative": obj-ref,
                   "settle_error": pe+re}
            if b.relation_kind == "on_next_to":
                if (r.get("support") != b.support or r.get("support_track_id") != b.support_track_id
                        or r.get("support_visible") is not True):
                    raise ValueError("missing_observed_placement_support")
                sup = _vector(r.get("support_xyz_base"))
                se = _metric(r, "support_error_m", nonnegative=True)
                row["settle_relative"] = np.concatenate([obj-ref, obj-sup])
                row["settle_error"] += pe+se
            fs = float(s[FINGERS[b.hand]].sum())
            holding = (a[GRIP_ACTION[b.hand]] < 0 and c["hold_finger_min"] <= fs <= c["hold_finger_max"]
                       and np.linalg.norm(row["obj_eef"]) + row["co_error"] <= c["eef_near_max"])
            opened = a[GRIP_ACTION[b.hand]] > 0 and fs >= c["open_finger_min"]
            if self._carry_required:
                if (self._carry is None or self._carry[:3] != (b.target, b.track_id, b.hand)
                        or self._carry[3] != step-1):
                    raise ValueError("missing_continuous_verified_carry")
            detail.update(finger_sum=fs, holding=bool(holding), direct_diagnostic=direct)
            if b.operation == "pick":
                clear = _metric(r, "support_clearance_m")
                err = _metric(r, "clearance_error_m", nonnegative=True)
                if self.baseline is None:
                    if abs(clear) + err > c["support_start_max"]:
                        raise ValueError("no_observed_supported_start")
                    self.baseline = {**row, "clearance": clear, "clearance_error": err}
                held = self._held(row, holding, captured_at)
                lift = clear-err-self.baseline["clearance"]-self.baseline["clearance_error"]
                dz = row["obj_ref"][2]-self.baseline["obj_ref"][2]-row["rel_error"]-self.baseline["rel_error"]
                candidate = held and lift >= c["lift_min"] and dz >= c["lift_min"]
                detail.update(lift_lower_m=lift, holding_frames=self.hold_n)
            elif b.operation == "transport":
                margin = _metric(r, "placement_margin_m") - _metric(r, "margin_error_m", nonnegative=True)
                candidate = self._held(row, holding, captured_at) and margin >= 0
                detail.update(destination_margin_lower_m=margin, holding_frames=self.hold_n)
            else:
                margin = _metric(r, "placement_margin_m") - _metric(r, "margin_error_m", nonnegative=True)
                if self.release is None:
                    if not holding and not opened:
                        raise ValueError("unobserved_release_transition")
                    if opened:
                        self.release = {"origin": None}
                    candidate = False
                else:
                    if not opened:
                        raise ValueError("reclosed_after_release")
                    detached = np.linalg.norm(row["obj_eef"]) - row["co_error"] >= c["detach_min"]
                    if detached and margin >= 0:
                        origin = self.release["origin"]
                        if origin is None or np.linalg.norm(row["settle_relative"]-origin["settle_relative"]) + row["settle_error"] + origin["settle_error"] > c["settle_max"]:
                            self.release["origin"], self.settle_n, self._settle_since = row, 1, captured_at
                        else:
                            self.settle_n += 1
                    else:
                        self.release["origin"], self.settle_n, self._settle_since = None, 0, None
                    candidate = (self.settle_n >= c["settle_frames"] and self._settle_since is not None
                                 and captured_at-self._settle_since >= c["settle_duration_s"])
                detail.update(placement_margin_lower_m=margin, released=self.release is not None,
                              stable_frames=self.settle_n)
            self.previous = row
            trusted = not direct and self._admission_blocker() is None
            # A place sequence maintains provenance of its release, NOT a new holding claim.
            if trusted and ((holding and (candidate or self._carry is not None))
                            or (b.operation == "place" and self.release is not None)):
                self._carry = (b.target, b.track_id, b.hand, step,
                               "released" if b.operation == "place" and self.release is not None else "held")
            elif self._carry is not None:
                self._invalidate()
                raise ValueError("carry_measurement_lost")
            return self._emit_result(step, captured_at, digest, bool(candidate),
                                     "all_postconditions_observed" if candidate else "postconditions_not_yet_met",
                                     detail, trusted=trusted)
        except Exception as error:
            # Producer exceptions/malformed sensor objects must never leave a reusable success receipt.
            self._invalidate()
            return self._emit_result(step if type(step) is int else None,
                                     captured_at if _number(captured_at) else None, digest, None,
                                     str(error) if isinstance(error, (ValueError, TypeError, KeyError)) else "producer_or_sensor_error:"+type(error).__name__, detail)

    def _held(self, row, holding, captured_at):
        # Conservative total error bounds supplement the reused motion primitive.
        for origin in (self.previous, self.hold_origin):
            if origin is not None and np.linalg.norm(row["obj_eef"]-origin["obj_eef"]) + row["co_error"] + origin["co_error"] > self.calibration["co_motion_max"]:
                holding = False
        if not holding:
            self._hold_since = None
        elif self._hold_since is None:
            self._hold_since = captured_at
        candidate = self._held_motion(row, holding)
        if self.hold_origin is not None:
            error = row["rel_error"] + self.hold_origin["rel_error"]
            candidate = candidate and min(np.linalg.norm(row[k]-self.hold_origin[k]) for k in ("obj_ref", "eef_ref")) - error >= self.calibration["motion_min"]
        return bool(candidate and self._hold_since is not None
                    and captured_at-self._hold_since >= self.calibration["hold_duration_s"])

    def consume(self, evidence, *, binding, observation_id, sensor_frame_sha256, now):
        """Return an orchestrator-compatible verified record, else None; never trust flags."""
        try:
            if (not isinstance(evidence, dict) or _canonical(evidence) != self._issued
                    or self.contract != binding or evidence["binding"] != asdict(binding)
                    or evidence["verified"] is not True or evidence["value"] not in (True, False)
                    or type(observation_id) is not int or evidence["observation_id"] != observation_id
                    or evidence["sensor_frame_sha256"] != sensor_frame_sha256
                    or not _number(now) or not 0 <= now-evidence["captured_at"] <= self.calibration["max_age_s"]
                    or self._admission_blocker() is not None):
                return None
            return copy.deepcopy(evidence)
        except (TypeError, ValueError, KeyError):
            return None


class Task090EvidencePostconditions(ManipulationEvidence):
    """Drop-in lifecycle ADAPTER once policy supplies a persistent observed Binding.

    Extra binding is mandatory; it is not guessed from a class label/docking map.
    No existing policy/config is switched by importing this class.
    """
    def start(self, key, skill, objects, state, step, *, episode=None, binding=None):
        if not isinstance(binding, Binding):
            raise ValueError("current_observed_binding_required")
        if (tuple(category(o) for o in objects) not in BINDINGS.get(skill, set())
                or binding.key != key or binding.episode != episode or binding.started_step != step
                or binding.target != objects[0] or binding.reference != objects[-1]
                or binding.hand != HAND.get(category(objects[0]))
                or binding.operation != ("pick" if skill == "pick up from" else "place")
                or binding.relation_kind != RELATION[skill]
                or (skill == "place on next to" and binding.support != objects[1])):
            raise ValueError("task090_contract_binding_mismatch")
        super().start(binding)
