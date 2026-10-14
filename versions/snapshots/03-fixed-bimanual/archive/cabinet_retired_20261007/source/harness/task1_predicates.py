"""Policy-visible, calibrated postconditions for Task 1 manipulation contracts (proprioception only).

Independent of the WAM's visual features (the patch/VTT gates are appearance candidates only): each predicate reads
finger joint positions and end-effector positions (robot base frame) from the 61-D proprio, with thresholds taken
from TRAIN demonstrations (outputs/validation/task1_docking_calibration_20260926/*_train.json):

  finger-sum (two fingers, m): open 0.100; left holding a can 0.063-0.078 (p5-p95); right holding the bin rim
  0.006-0.026; commanded-closed-on-nothing ~0.000.
  lift after grasp (eef z, base frame): bin p5 0.124 m, can p5 0.076 m.
  release over the bin: |left eef - right eef| horizontal p95 0.235 m (max 0.28) at the release command; while
  carrying without releasing the same distance has p5 0.43 m.

Evidence dicts follow harness.orchestrator.Orchestrator._postcondition_evidence (verified, verifier_kind,
source, key, observation_id, satisfied) plus `failed` (a verified negative outcome that triggers recovery).
CPU self-test: `python3 -m harness.task1_predicates`.
"""
from __future__ import annotations

import numpy as np

from harness.grounding import category

HAND = {"trash_can": "right", "can_of_soda": "left"}
FINGERS = {"left": slice(24, 26), "right": slice(49, 51)}
EEF = {"left": slice(17, 20), "right": slice(42, 45)}
GRIP_ACTION = {"left": 14, "right": 22}
HOLD_RANGE = {"can_of_soda": (0.03, 0.092), "trash_can": (0.004, 0.06)}
LIFT_MIN = {"can_of_soda": 0.06, "trash_can": 0.10}
EMPTY_MAX = 0.003
OPEN_MIN = 0.095
RELEASE_HORIZ_MAX = 0.30
RELEASE_DZ = (-0.06, 0.26)
SOURCE = "task1_physical_predicates:v1"


def finger_sum(state, hand):
    return float(np.sum(np.asarray(state, float)[FINGERS[hand]]))


class Task1Postconditions:
    """start(key, state, step) at contract start; update(state, action, step) every control step -> evidence."""

    def __init__(self, hold_streak=20, empty_streak=60, settle_after_release=10):
        self.hold_streak, self.empty_streak, self.settle = int(hold_streak), int(empty_streak), int(settle_after_release)
        self.key = None

    def start(self, key, skill, objects, state, step):
        self.key, self.skill = key, skill
        self.obj = category(objects[0])
        self.hand = HAND.get(self.obj)
        self.started = int(step)
        self.hold_n = self.empty_n = 0
        self.z_hold0 = None
        self.released_at = None
        self.release_delta = None
        self.detail = {}
        if skill == "place in":
            # the can must be in the left hand when the release skill starts (precondition, observed not assumed)
            fs = finger_sum(state, "left")
            self.detail["start_left_finger_sum"] = round(fs, 4)
            self.was_holding = HOLD_RANGE["can_of_soda"][0] <= fs <= HOLD_RANGE["can_of_soda"][1]

    def _evidence(self, step, satisfied, failed, reason):
        return {"verified": True, "verifier_kind": "controller_postconditions", "source": SOURCE, "key": self.key,
                "observation_id": int(step), "satisfied": bool(satisfied), "failed": bool(failed), "reason": reason,
                "detail": dict(self.detail)}

    def update(self, state, action, step):
        if self.key is None or self.hand is None and self.skill != "place in":
            return None
        state = np.asarray(state, float)
        if self.skill == "pick up from":
            lo, hi = HOLD_RANGE[self.obj]
            fs = finger_sum(state, self.hand)
            z = float(state[EEF[self.hand]][2])
            cmd_closed = action is not None and float(action[GRIP_ACTION[self.hand]]) < 0.0
            holding = cmd_closed and lo <= fs <= hi
            if holding:
                if self.z_hold0 is None:
                    self.z_hold0 = z
                self.hold_n += 1
            else:
                self.hold_n, self.z_hold0 = 0, None
            self.empty_n = self.empty_n + 1 if (cmd_closed and fs < EMPTY_MAX) else 0
            lift = None if self.z_hold0 is None else z - self.z_hold0
            self.detail = {"hand": self.hand, "finger_sum": round(fs, 4), "holding_streak": self.hold_n,
                           "lift": None if lift is None else round(lift, 3), "empty_closed_streak": self.empty_n}
            if holding and self.hold_n >= self.hold_streak and lift is not None and lift >= LIFT_MIN[self.obj]:
                return self._evidence(step, True, False, "held_and_lifted")
            if self.empty_n >= self.empty_streak:
                return self._evidence(step, False, True, "closed_on_nothing")
            return self._evidence(step, False, False, "in_progress")
        if self.skill == "place in":
            fl, fr = finger_sum(state, "left"), finger_sum(state, "right")
            self.detail.update(left_finger_sum=round(fl, 4), right_finger_sum=round(fr, 4))
            if not self.was_holding:
                return self._evidence(step, False, True, "precondition_object_not_in_hand")
            if fr < EMPTY_MAX:
                return self._evidence(step, False, True, "container_dropped")
            if self.released_at is None:
                if fl >= OPEN_MIN:
                    self.released_at = int(step)
                    d = state[EEF["left"]] - state[EEF["right"]]
                    self.release_delta = d
                    self.detail.update(release_delta=[round(float(v), 3) for v in d],
                                       release_horiz=round(float(np.linalg.norm(d[:2])), 3))
                else:
                    # keep the last pre-release geometry (the fingers open over a few steps)
                    self._pre_release = state[EEF["left"]] - state[EEF["right"]]
                return self._evidence(step, False, False, "holding")
            d = getattr(self, "_pre_release", self.release_delta)
            over = float(np.linalg.norm(d[:2])) <= RELEASE_HORIZ_MAX and RELEASE_DZ[0] <= float(d[2]) <= RELEASE_DZ[1]
            self.detail.update(pre_release_delta=[round(float(v), 3) for v in d], over_container=bool(over))
            if not over:
                return self._evidence(step, False, True, "released_outside_container")
            if step - self.released_at >= self.settle:
                return self._evidence(step, True, False, "released_over_container")
            return self._evidence(step, False, False, "released_settling")
        return None


def _self_test():
    p = Task1Postconditions()
    s = np.zeros(61)
    s[24:26], s[49:51] = 0.05, 0.05
    s[19], s[44] = 0.12, 0.30
    a = np.zeros(23)
    # pick can with the left hand: close, hold, lift
    p.start("pick up from(can_of_soda,floors)", "pick up from", ("can_of_soda@1", "floors"), s, 0)
    a[14] = -1.0
    s[24:26] = 0.037
    ev = None
    for t in range(1, 60):
        s[19] = 0.12 + 0.004 * t
        ev = p.update(s, a, t)
        if ev["satisfied"]:
            break
    assert ev["satisfied"] and not ev["failed"], ev
    # pick that closes on nothing fails
    p.start("pick up from(can_of_soda,floors)", "pick up from", ("can_of_soda@2", "floors"), s, 0)
    s[24:26] = 0.0
    for t in range(1, 80):
        ev = p.update(s, a, t)
        if ev["failed"]:
            break
    assert ev["failed"] and ev["reason"] == "closed_on_nothing", ev
    # place over the bin: left eef 0.16 m from the right eef, then open
    s[24:26], s[49:51] = 0.037, 0.005
    s[17:20], s[42:45] = (0.40, 0.10, 0.45), (0.45, -0.05, 0.40)
    p.start("place in(can_of_soda,trash_can)", "place in", ("can_of_soda@1", "trash_can"), s, 0)
    for t in range(1, 40):
        if t == 5:
            s[24:26] = 0.05
        ev = p.update(s, a, t)
        if ev["satisfied"] or ev["failed"]:
            break
    assert ev["satisfied"] and ev["reason"] == "released_over_container", ev
    # release far from the bin -> failure
    s[24:26] = 0.037
    s[17:20] = (0.40, 0.55, 0.45)
    p.start("place in(can_of_soda,trash_can)", "place in", ("can_of_soda@2", "trash_can"), s, 0)
    for t in range(1, 40):
        if t == 5:
            s[24:26] = 0.05
        ev = p.update(s, a, t)
        if ev["satisfied"] or ev["failed"]:
            break
    assert ev["failed"] and ev["reason"] == "released_outside_container", ev
    print("PASS task1_predicates self-test")


if __name__ == "__main__":
    _self_test()
