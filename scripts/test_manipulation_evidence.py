#!/usr/bin/env python3
"""CPU contract tests ONLY. Synthetic measurements and mocked admission are NOT
calibration, real physical labels or runtime admission. No manifest is generated.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

from harness.manipulation_evidence import (
    ManipulationEvidence, Task090EvidencePostconditions, Binding, TRUSTED_REVIEW_SHA256,
)
from harness.task090_predicates import EEF, FINGERS, GRIP_ACTION, sensor_frame_sha256
from harness.task1_perception_adapter import HEAD_RGB_SUFFIX, HEAD_DEPTH_SUFFIX, CAM_POSE_SUFFIX
from harness.orchestrator import Orchestrator

# Explicitly synthetic unit-test scale, NEVER written as a calibration manifest.
CAL = dict(source_split="geometry", source="synthetic CPU fixture NOT ADMITTED",
           calibration_id="synthetic-not-admitted", hold_finger_min=.01, hold_finger_max=.06,
           open_finger_min=.09, eef_near_max=.02, lift_min=.03, motion_min=.03,
           co_motion_max=.01, support_start_max=.01, detach_min=.08, settle_max=.01,
           hold_frames=3, settle_frames=3, max_frame_gap=1,
           max_age_s=.15, min_period_s=.08, max_period_s=.12,
           hold_duration_s=.15, settle_duration_s=.15)


class SyntheticProducer:
    version = "synthetic-test-double-not-admitted"

    def __init__(self):
        self.change = None
        self.object_z = None
        self.margin = .02
        self.packet = None

    def measure(self, packet):
        self.packet = packet
        b, step = packet["binding"], packet["observation_id"]
        z = (step-1)*.06 if self.object_z is None else self.object_z
        r = dict(reference=b["reference"], reference_track_id=b["reference_track_id"],
                 reference_visible=True, reference_xyz_base=[0., 0., 0.], reference_error_m=.0001,
                 kind=b["relation_kind"], support_clearance_m=z, clearance_error_m=.0001,
                 placement_margin_m=self.margin, margin_error_m=.0001)
        if b["support"]:
            r.update(support=b["support"], support_track_id=b["support_track_id"], support_visible=True,
                     support_xyz_base=[0., 0., 0.], support_error_m=.0001)
        d = dict(object_id=b["target"], track_id=b["track_id"], hand=b["hand"], episode=b["episode"],
                 contract_id=b["contract_id"], observation_id=step,
                 sensor_frame_sha256=packet["sensor_frame_sha256"], producer_version=self.version,
                 visible=True, metric_valid=True, source="rgbd", calibration_id=CAL["calibration_id"],
                 xyz_base=[0., 0., z], position_error_m=.0001, eef_error_m=.0001, relation=r)
        if self.change:
            self.change(d)
        return [d]


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.producer = SyntheticProducer()
        self.gate = ManipulationEvidence(CAL, producer=self.producer)
        self.binding = Binding(7, "attempt-pick-1", "pick up from(half,countertop)", 0,
                               "pick", "half", "track-fruit", "left", "countertop", "track-counter", "support")
        self.gate.start(self.binding)

    def mocked_admission(self):
        # Exercise success code in memory; never alter trust roots or write a review.
        stub = patch.object(self.gate, "_admission_blocker", return_value=None)
        stub.start()
        self.addCleanup(stub.stop)

    def inputs(self, step, *, opened=False, detached=False, timestamp=None, pixel=None):
        state = np.zeros(61)
        state[EEF[self.binding.hand]] = [0., 0., (step-1)*.06]
        if self.producer.object_z is not None:
            state[EEF[self.binding.hand]][2] = self.producer.object_z
        if detached:
            state[EEF[self.binding.hand]][0] = .2
        state[FINGERS[self.binding.hand]] = .05 if opened else .02
        action = np.zeros(23)
        action[GRIP_ACTION[self.binding.hand]] = 1 if opened else -1
        obs = {"robot"+HEAD_RGB_SUFFIX: np.full((2, 2, 3), step if pixel is None else pixel, dtype=np.uint8),
               "robot"+HEAD_DEPTH_SUFFIX: np.ones((2, 2)),
               "robot"+CAM_POSE_SUFFIX: np.array([0., 0., 0., 0., 0., 0., 1.])}
        captured = step*.1 if timestamp is None else timestamp
        return state, action, obs, captured

    def update(self, step, **kwargs):
        s, a, obs, captured = self.inputs(step, **kwargs)
        return self.gate.update(s, a, step, episode=self.binding.episode,
                                observation=obs, captured_at=captured, now=captured+.01)

    def pick(self):
        ev = None
        for step in range(1, 4):
            ev = self.update(step)
        return ev

    def place(self, kind="inside"):
        self.binding = replace(self.binding, contract_id="attempt-place-2", started_step=3,
                               key="place in(half,trash_can)", operation="place", reference="trash_can",
                               reference_track_id="track-bin", relation_kind=kind,
                               support="floors" if kind == "on_next_to" else None,
                               support_track_id="track-floor" if kind == "on_next_to" else None)
        self.gate.start(self.binding)
        self.producer.object_z = .12

    def consume(self, ev, **overrides):
        args = dict(binding=self.binding, observation_id=ev["observation_id"],
                    sensor_frame_sha256=ev["sensor_frame_sha256"], now=ev["captured_at"]+.01)
        args.update(overrides)
        return self.gate.consume(ev, **args)

    def assertUnknown(self, ev):
        self.assertEqual(ev["status"], "unknown", ev)
        self.assertIsNone(ev["value"])
        self.assertFalse(ev["verified"])
        self.assertFalse(ev["satisfied"])
        self.assertFalse(ev["failed"])
        self.assertIsNone(self.consume(ev))

    def test_shipped_trust_root_empty(self):
        self.assertEqual(TRUSTED_REVIEW_SHA256, frozenset())
        self.assertFalse(self.gate.admission_status()["admitted"])

    def test_positive_candidate_without_independent_admission_is_unknown(self):
        ev = self.pick()
        self.assertTrue(ev["candidate_satisfied"], ev)
        self.assertUnknown(ev)
        self.assertEqual(ev["reason"], "missing_independent_admission")

    def test_config_flags_do_not_admit(self):
        self.gate.calibration.update(admitted=True, verified=True, satisfied=True)
        self.assertUnknown(self.pick())

    def test_mocked_admission_exercises_true_and_false_not_physical_validation(self):
        self.mocked_admission()
        first = self.update(1)
        self.assertIs(first["value"], False)
        self.assertFalse(first["failed"])
        self.update(2)
        ev = self.update(3)
        self.assertIs(ev["value"], True)
        self.assertIsNotNone(self.consume(ev))
        self.assertIsNotNone(Orchestrator._postcondition_evidence(self.consume(ev), key=self.binding.key, observation_id=3))

    def test_direct_detections_cannot_be_verified_even_with_mocked_admission(self):
        self.mocked_admission()
        for t in range(1, 4):
            s, a, obs, ts = self.inputs(t)
            packet = dict(binding=self.binding.__dict__, observation_id=t,
                          sensor_frame_sha256=sensor_frame_sha256(obs["robot"+HEAD_RGB_SUFFIX], obs["robot"+HEAD_DEPTH_SUFFIX]))
            ev = self.gate.update(s, a, t, episode=7, observation=obs, captured_at=ts, now=ts,
                                  detections=self.producer.measure(packet))
        self.assertTrue(ev["candidate_satisfied"])
        self.assertUnknown(ev)
        self.assertIsNone(self.gate._carry)

    def test_fake_evidence_fields_rejected(self):
        for key in ("verified", "satisfied", "admitted", "candidate_satisfied", "gt", "evaluator", "privbase", "vtt"):
            with self.subTest(field=key):
                self.setUp()
                self.producer.change = lambda d, key=key: d.update({key: True})
                ev = self.update(1)
                self.assertUnknown(ev)
                self.assertEqual(ev["reason"], "unexpected_or_missing_measurement_fields")

    def test_wrong_target_track_hand_episode_contract_and_frame_rejected(self):
        for key, value in (("object_id", "other-fruit"), ("track_id", "other-track"), ("hand", "right"),
                           ("episode", 8), ("episode", True), ("contract_id", "other-attempt"),
                           ("observation_id", 2), ("sensor_frame_sha256", "f"*64), ("producer_version", "other:v1")):
            with self.subTest(field=key):
                self.setUp()
                self.producer.change = lambda d, k=key, v=value: d.update({k: v})
                self.assertUnknown(self.update(1))

    def test_wrong_update_episode(self):
        s, a, obs, ts = self.inputs(1)
        ev = self.gate.update(s, a, 1, episode=8, observation=obs, captured_at=ts, now=ts)
        self.assertEqual(ev["reason"], "episode_mismatch")
        self.assertUnknown(ev)

    def test_stale_and_future_timestamps(self):
        for now in (.4, .05, float("nan")):
            with self.subTest(now=now):
                self.setUp()
                s, a, obs, ts = self.inputs(1)
                ev = self.gate.update(s, a, 1, episode=7, observation=obs, captured_at=ts, now=now)
                self.assertEqual(ev["reason"], "stale_or_future_observation")
                self.assertUnknown(ev)

    def test_repeat_pixels_new_step_cannot_count(self):
        self.update(1)
        ev = self.update(2, pixel=1)
        self.assertEqual(ev["reason"], "repeated_sensor_frame")
        self.assertUnknown(ev)

    def test_replayed_observation_id_and_gap(self):
        for step in (1, 3):
            with self.subTest(step=step):
                self.setUp()
                self.update(1)
                ev = self.update(step)
                self.assertEqual(ev["reason"], "nonconsecutive_observation")
                self.assertUnknown(ev)

    def test_sampling_too_slow_or_fast_cannot_count_as_stability(self):
        for timestamp in (.101, .4):
            with self.subTest(timestamp=timestamp):
                self.setUp()
                self.update(1)
                ev = self.update(2, timestamp=timestamp)
                self.assertEqual(ev["reason"], "observation_frequency_out_of_scope")
                self.assertUnknown(ev)

    def test_pre_action_start_observation_not_counted(self):
        self.assertUnknown(self.update(0))

    def test_missing_temporal_calibration_and_sparse_frames_rejected(self):
        for change in ({"max_frame_gap": 2}, {"max_age_s": None}):
            with self.subTest(change=change):
                self.gate = ManipulationEvidence({**CAL, **change}, producer=self.producer)
                self.gate.start(self.binding)
                self.assertUnknown(self.update(1))

    def test_only_eef_motion_only_object_motion_only_fingers_not_enough(self):
        for mode in ("eef_only", "object_only", "stationary", "open"):
            with self.subTest(mode=mode):
                self.setUp()
                for t in range(1, 4):
                    s, a, obs, ts = self.inputs(t, opened=mode == "open")
                    if mode in ("eef_only", "stationary"):
                        self.producer.object_z = 0.
                    if mode in ("object_only", "stationary"):
                        s[EEF["left"]] = 0.
                    ev = self.gate.update(s, a, t, episode=7, observation=obs, captured_at=ts, now=ts)
                self.assertFalse(ev["candidate_satisfied"], ev)

    def test_missing_relation_or_invalid_bounds_cannot_pass(self):
        changes = [lambda d: d.update(relation={}),
                   lambda d: d["relation"].update(reference_visible=False),
                   lambda d: d["relation"].update(reference_track_id="other"),
                   lambda d: d.update(position_error_m=float("nan")),
                   lambda d: d.update(position_error_m=-1),
                   lambda d: d["relation"].pop("clearance_error_m"),
                   lambda d: d["relation"].update(verified=True)]
        for change in changes:
            with self.subTest(change=change):
                self.setUp()
                self.producer.change = change
                self.assertUnknown(self.update(1))

    def test_uncertainty_is_not_ignored(self):
        self.producer.change = lambda d: d.update(position_error_m=.1)
        self.assertFalse(self.pick()["candidate_satisfied"])

    def test_place_needs_verified_pick_not_static_program(self):
        self.pick()  # candidate only, not a lease
        self.place()
        self.assertUnknown(self.update(4, opened=True, detached=True))

    def test_cross_skill_target_or_hand_rebinding_rejected(self):
        for change in ({"track_id": "new-track"}, {"hand": "right"}):
            with self.subTest(change=change):
                self.setUp()
                with patch.object(self.gate, "_admission_blocker", return_value=None):
                    self.assertTrue(self.pick()["satisfied"])
                    self.place()
                    self.binding = replace(self.binding, **change)
                    self.gate.start(self.binding)
                    self.assertUnknown(self.update(4))

    def test_container_reference_cannot_change_track_across_contracts(self):
        self.gate._known_tracks["trash_can"] = "previously-observed-bin"
        self.place()
        self.assertUnknown(self.update(4))

    def test_track_aliasing_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.binding, reference_track_id=self.binding.track_id).validate()

    def test_place_release_geometry_detachment_and_stability_all_required(self):
        self.mocked_admission()
        self.assertTrue(self.pick()["satisfied"])
        self.place()
        self.assertFalse(self.update(4, opened=True, detached=True)["satisfied"])
        self.assertFalse(self.update(5, opened=True, detached=True)["satisfied"])
        self.assertFalse(self.update(6, opened=True, detached=True)["satisfied"])
        ev = self.update(7, opened=True, detached=True)
        self.assertTrue(ev["satisfied"], ev)
        self.assertIsNotNone(self.consume(ev))

    def test_place_cannot_succeed_by_release_or_wait_alone(self):
        for mode in ("no_release", "not_detached", "outside", "uncertain_margin", "moving"):
            with self.subTest(mode=mode):
                self.setUp()
                with patch.object(self.gate, "_admission_blocker", return_value=None):
                    self.pick()
                    self.place()
                    if mode == "outside":
                        self.producer.margin = -.02
                    if mode == "uncertain_margin":
                        self.producer.change = lambda d: d["relation"].update(margin_error_m=.04)
                    for t in range(4, 10):
                        if mode == "moving":
                            self.producer.object_z += .02
                        ev = self.update(t, opened=mode != "no_release", detached=mode != "not_detached")
                    self.assertFalse(ev["satisfied"], ev)

    def test_sink_adjacency_requires_floor_and_sink_not_docking_point(self):
        self.mocked_admission()
        self.pick()
        self.place("on_next_to")
        self.producer.change = lambda d: d["relation"].pop("support_visible")
        ev = self.update(4, opened=True, detached=True)
        self.assertEqual(ev["reason"], "missing_observed_placement_support")
        self.assertUnknown(ev)

    def test_released_object_cannot_be_reused_as_a_held_pick_lease(self):
        self.mocked_admission()
        self.pick()
        self.place()
        for t in range(4, 8):
            ev = self.update(t, opened=True, detached=True)
        self.assertTrue(ev["satisfied"])
        self.binding = replace(self.binding, contract_id="another-place", started_step=7)
        self.gate.start(self.binding)
        self.assertUnknown(self.update(8, opened=True, detached=True))

    def test_carry_loss_cannot_be_hidden_by_starting_new_skill(self):
        self.mocked_admission()
        self.pick()
        self.producer.change = lambda d: d.update(track_id="wrong")
        self.assertUnknown(self.update(4))
        self.producer.change = None
        self.place()
        self.assertUnknown(self.update(5))

    def test_navigation_gap_cannot_preserve_carry(self):
        self.mocked_admission()
        self.pick()
        self.place()
        self.assertUnknown(self.update(7))

    def test_transport_requires_carry_motion_and_observed_destination(self):
        self.mocked_admission()
        self.pick()
        self.binding = replace(self.binding, operation="transport", relation_kind="destination",
                               key="transport(half,workzone)", contract_id="transport-2", started_step=3,
                               reference="workzone", reference_track_id="track-workzone")
        self.gate.start(self.binding)
        self.producer.margin = -.01
        for t in range(4, 7):
            ev = self.update(t)
        self.assertFalse(ev["satisfied"])
        self.producer.margin = .02
        ev = self.update(7)
        self.assertTrue(ev["satisfied"], ev)

    def test_forged_or_mutated_receipts_rejected(self):
        self.mocked_admission()
        ev = self.pick()
        self.assertIsNotNone(self.consume(ev))
        for key, value in (("satisfied", False), ("episode", 8), ("receipt", "fake"),
                           ("source", "other"), ("verified", 1), ("extra", True)):
            with self.subTest(key=key):
                changed = copy.deepcopy(ev)
                changed[key] = value
                self.assertIsNone(self.consume(changed))
        unknown_gate = ManipulationEvidence(CAL, producer=self.producer)
        unknown_gate.start(self.binding)
        self.assertIsNone(unknown_gate.consume(ev, binding=self.binding, observation_id=3,
                         sensor_frame_sha256=ev["sensor_frame_sha256"], now=.31))

    def test_receipt_wrong_context_stale_or_revoked_rejected(self):
        self.mocked_admission()
        ev = self.pick()
        for changes in ({"now": 1.}, {"observation_id": 4}, {"sensor_frame_sha256": "wrong"},
                        {"binding": replace(self.binding, contract_id="retry-2")}):
            with self.subTest(changes=changes):
                self.assertIsNone(self.consume(ev, **changes))
        with patch.object(self.gate, "_admission_blocker", return_value="revoked"):
            self.assertIsNone(self.consume(ev))
        self.update(4)
        self.assertIsNone(self.consume(ev))

    def test_episode_reset_clears_receipts_and_held_identity(self):
        self.mocked_admission()
        ev = self.pick()
        self.binding = replace(self.binding, episode=8, contract_id="episode-8")
        self.gate.start(self.binding)
        self.assertIsNone(self.consume(ev))
        self.assertIsNone(self.gate._carry)
        self.assertIsNone(self.gate.last_step)

    def test_evaluator_and_privbase_payloads_never_reach_producer(self):
        s, a, obs, ts = self.inputs(1)
        obs.update(evaluator={"task_success": True}, privbase={"object_pose": [9, 9, 9]}, reward=1)
        self.gate.update(s, a, 1, episode=7, observation=obs, captured_at=ts, now=ts)
        self.assertEqual(set(self.producer.packet), {"rgb", "depth", "camera_pose", "state", "applied_action",
                         "binding", "observation_id", "captured_at", "sensor_frame_sha256"})

    def test_ambiguous_sensor_and_producer_exception_fail_closed(self):
        s, a, obs, ts = self.inputs(1)
        obs["other"+HEAD_RGB_SUFFIX] = obs["robot"+HEAD_RGB_SUFFIX]
        self.assertUnknown(self.gate.update(s, a, 1, episode=7, observation=obs, captured_at=ts, now=ts))
        self.setUp()
        with patch.object(self.producer, "measure", side_effect=RuntimeError("test error")):
            self.assertUnknown(self.update(1))

    def test_task090_adapter_requires_exact_binding_and_preserves_key(self):
        gate = Task090EvidencePostconditions(CAL, producer=self.producer)
        gate.start(self.binding.key, "pick up from", ("half", "countertop"), np.zeros(61), 0,
                   episode=7, binding=self.binding)
        self.assertEqual(gate.key, self.binding.key)
        with self.assertRaises(ValueError):
            gate.start(self.binding.key, "pick up from", ("half_banana", "countertop"), np.zeros(61), 0,
                       episode=7, binding=self.binding)
        with self.assertRaises(ValueError):
            gate.start(self.binding.key, "pick up from", ("half", "countertop"), np.zeros(61), 0, episode=7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(kind="synthetic_cpu_contract_tests_not_admission", tests=result.testsRun,
                  failures=[{"test": str(t), "traceback": trace} for t, trace in result.failures],
                  errors=[{"test": str(t), "traceback": trace} for t, trace in result.errors],
                  passed=result.wasSuccessful(), seconds=round(time.monotonic()-started, 4),
                  production_admitted=False, physical_validation_performed=False,
                  gpu_used=False, trusted_review_count=len(TRUSTED_REVIEW_SHA256),
                  positive_branches="unittest mock only; no review/calibration manifest generated",
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__).resolve(), Path(__file__).resolve().parents[1]/"harness/manipulation_evidence.py")})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
