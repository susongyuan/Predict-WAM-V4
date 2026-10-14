"""Policy-visible postconditions for Task 0 turning_on_radio contracts (proprioception only; CPU self-test:
`python3 -m harness.task0_predicates`).

Thresholds are TRAIN-demo statistics (165 standard-hand demos: right hand holds the radio, left fist presses;
outputs/validation/task0_demo_geometry_20260927/, scripts/analyze_task0_demo_geometry.py, progress.md 2026-09-27 17:30):
  right finger sum while holding the radio p1-p99 0.024-0.078 m (open 0.100, closed on nothing ~0.000);
  right eef lift from closure to pick end p5 0.079 m (median 0.170);
  at the evaluator's toggle frame: left finger sum 0.000 (closed fist), |left eef - right eef| p1-p99 0.134-0.213 m;
  place on release: right eef z - grasp z p1-p99 -0.005..+0.075 m.

`press` is special. OmniGibson ToggledOn flips when a finger overlaps the radio's toggle-button marker for 5
consecutive steps and changes no texture by default, so no onboard sensor here observes "radio on". The press
predicate therefore returns a CONTACT CANDIDATE (verifier_kind "contact_candidate", outcome_observable False): the
program may move on, but the orchestrator records it as unverified (no memory promotion, no reliability update), and
the candidate is not re-pressed because a second press would toggle the radio off. Only the evaluator decides success.
Every press evidence record also carries `retry_safe`: True only while the left hand has stayed farther than
CONTACT_POSSIBLE_DIST from the right (radio-holding) hand during the whole attempt, i.e. the fist cannot have touched
the held radio (radio 0.32 m wide, handle held at its centre). The orchestrator retries a failed press only when
retry_safe is True (harness/orchestrator.py irreversible_skills); otherwise it holds instead of pressing again.
"""
from __future__ import annotations

import numpy as np

FINGERS = {"left": slice(24, 26), "right": slice(49, 51)}
EEF = {"left": slice(17, 20), "right": slice(42, 45)}
GRIP_ACTION = {"left": 14, "right": 22}
HOLD_RANGE = (0.020, 0.085)
EMPTY_MAX = 0.003
OPEN_MIN = 0.095
LIFT_MIN = 0.06
PRESS_DIST = (0.10, 0.23)
PRESS_STREAK = 10          # > the 5 overlap steps ToggledOn needs; the demo fist dwells (median speed 0.11 m/s)
CONTACT_POSSIBLE_DIST = 0.30   # m, |left eef - right eef| below this: the left hand may touch the held radio
RELEASE_DZ = (-0.05, 0.10)
RELEASE_Z_ABS = (0.58, 0.75)   # fallback when no grasp height was observed in this episode
SOURCE = "task0_physical_predicates:v1"


def finger_sum(state, hand):
    return float(np.sum(np.asarray(state, float)[FINGERS[hand]]))


class Task0Postconditions:
    """Same interface as harness.task1_predicates.Task1Postconditions: start(...) at contract start, update(...) each
    control step -> evidence dict or None."""

    def __init__(self, hold_streak=20, empty_streak=60, settle_after_release=10, press_streak=PRESS_STREAK):
        self.hold_streak, self.empty_streak = int(hold_streak), int(empty_streak)
        self.settle, self.press_streak = int(settle_after_release), int(press_streak)
        self.key = None
        self.grasp_z = None          # right eef z at the verified pick (kept across contracts within an episode)

    def reset_episode(self):
        self.key, self.grasp_z = None, None

    def start(self, key, skill, objects, state, step):
        self.key, self.skill, self.started = key, skill, int(step)
        self.hold_n = self.empty_n = self.press_n = 0
        self.min_lr = float("inf")   # closest left-right eef distance in this attempt (press contact possibility)
        self.z_hold0 = None
        self.released_at = None
        self.detail = {}
        fs = finger_sum(state, "right")
        self.was_holding = HOLD_RANGE[0] <= fs <= HOLD_RANGE[1]
        self.detail["start_right_finger_sum"] = round(fs, 4)

    def _evidence(self, step, satisfied, failed, reason, kind="controller_postconditions", observable=True, retry_safe=None):
        ev = {"verified": kind == "controller_postconditions", "verifier_kind": kind, "outcome_observable": observable,
              "source": SOURCE, "key": self.key, "observation_id": int(step), "satisfied": bool(satisfied),
              "failed": bool(failed), "reason": reason, "detail": dict(self.detail)}
        if retry_safe is not None:        # press only: may this attempt be repeated without undoing its effect?
            ev["retry_safe"] = bool(retry_safe)
        return ev

    def update(self, state, action, step):
        if self.key is None:
            return None
        state = np.asarray(state, float)
        fr, fl = finger_sum(state, "right"), finger_sum(state, "left")
        zr = float(state[EEF["right"]][2])
        if self.skill == "pick up from":
            cmd_closed = action is not None and float(action[GRIP_ACTION["right"]]) < 0.0
            holding = cmd_closed and HOLD_RANGE[0] <= fr <= HOLD_RANGE[1]
            if holding:
                self.z_hold0 = zr if self.z_hold0 is None else self.z_hold0
                self.hold_n += 1
            else:
                self.hold_n, self.z_hold0 = 0, None
            self.empty_n = self.empty_n + 1 if (cmd_closed and fr < EMPTY_MAX) else 0
            lift = None if self.z_hold0 is None else zr - self.z_hold0
            self.detail = {"hand": "right", "finger_sum": round(fr, 4), "holding_streak": self.hold_n,
                           "lift": None if lift is None else round(lift, 3), "empty_closed_streak": self.empty_n}
            if holding and self.hold_n >= self.hold_streak and lift is not None and lift >= LIFT_MIN:
                self.grasp_z = self.z_hold0
                return self._evidence(step, True, False, "held_and_lifted")
            if self.empty_n >= self.empty_streak:
                return self._evidence(step, False, True, "closed_on_nothing")
            return self._evidence(step, False, False, "in_progress")
        if self.skill == "press":
            d = float(np.linalg.norm(state[EEF["left"]] - state[EEF["right"]]))
            self.min_lr = min(self.min_lr, d)
            safe = self.min_lr > CONTACT_POSSIBLE_DIST
            if not self.was_holding:
                return self._evidence(step, False, True, "precondition_radio_not_in_hand", retry_safe=safe)
            if fr < EMPTY_MAX:
                return self._evidence(step, False, True, "radio_dropped", retry_safe=safe)
            fist = fl < EMPTY_MAX and action is not None and float(action[GRIP_ACTION["left"]]) < 0.0
            near = PRESS_DIST[0] <= d <= PRESS_DIST[1]
            self.press_n = self.press_n + 1 if (fist and near) else 0
            self.detail = {"left_finger_sum": round(fl, 4), "right_finger_sum": round(fr, 4), "lr_dist": round(d, 3),
                           "fist": bool(fist), "press_streak": self.press_n, "min_lr_dist": round(self.min_lr, 3)}
            if self.press_n >= self.press_streak:
                return self._evidence(step, True, False, "fist_dwell_at_held_radio", kind="contact_candidate", observable=False,
                                      retry_safe=safe)
            return self._evidence(step, False, False, "in_progress", kind="contact_candidate", observable=False, retry_safe=safe)
        if self.skill == "place on":
            if not self.was_holding:
                return self._evidence(step, False, True, "precondition_radio_not_in_hand")
            self.detail.update(right_finger_sum=round(fr, 4), right_eef_z=round(zr, 3))
            if self.released_at is None:
                if fr >= OPEN_MIN:
                    self.released_at = int(step)
                    z = getattr(self, "_pre_z", zr)
                    ok = (RELEASE_DZ[0] <= z - self.grasp_z <= RELEASE_DZ[1]) if self.grasp_z is not None else \
                        (RELEASE_Z_ABS[0] <= z <= RELEASE_Z_ABS[1])
                    self.detail.update(release_z=round(z, 3), grasp_z=None if self.grasp_z is None else round(self.grasp_z, 3),
                                       at_surface_height=bool(ok))
                    self._release_ok = ok
                else:
                    self._pre_z = zr
                return self._evidence(step, False, False, "holding")
            if not self._release_ok:
                return self._evidence(step, False, True, "released_away_from_surface")
            if step - self.released_at >= self.settle:
                return self._evidence(step, True, False, "released_at_surface_height")
            return self._evidence(step, False, False, "released_settling")
        return None


def _self_test():
    p = Task0Postconditions()
    s = np.zeros(61); a = np.zeros(23)
    s[24:26], s[49:51] = 0.05, 0.05
    s[42:45] = (0.70, -0.15, 0.64)
    p.start("pick up from(radio,coffee_table)", "pick up from", ("radio", "coffee_table"), s, 0)
    a[22] = -1.0; s[49:51] = 0.019
    for t in range(1, 80):
        s[44] = 0.64 + 0.004 * t
        ev = p.update(s, a, t)
        if ev["satisfied"]:
            break
    assert ev["satisfied"] and ev["verified"] and abs(p.grasp_z - 0.644) < 1e-6, ev
    p.start("pick up from(radio,coffee_table)", "pick up from", ("radio", "coffee_table"), s, 0)
    s2 = s.copy(); s2[49:51] = 0.0
    for t in range(1, 80):
        ev = p.update(s2, a, t)
        if ev["failed"]:
            break
    assert ev["failed"] and ev["reason"] == "closed_on_nothing", ev
    # press: left fist next to the held radio for PRESS_STREAK steps -> unverified contact candidate
    s[42:45] = (0.65, -0.11, 0.81); s[17:20] = (0.60, 0.05, 0.78); s[24:26] = 0.0; a[14] = -1.0
    p.start("press(radio)", "press", ("radio",), s, 0)
    for t in range(1, 30):
        ev = p.update(s, a, t)
        if ev["satisfied"]:
            break
    assert ev["satisfied"] and not ev["verified"] and ev["verifier_kind"] == "contact_candidate" and t == PRESS_STREAK, (ev, t)
    assert ev["retry_safe"] is False and ev["outcome_observable"] is False   # the fist was at the radio: never re-press
    # a press attempt whose left hand never came near the held radio stays retry-safe (it cannot have toggled it)
    s_far = s.copy(); s_far[17:20] = (0.30, 0.45, 0.60)
    p.start("press(radio)", "press", ("radio",), s_far, 0)
    ev = p.update(s_far, a, 1)
    assert ev["retry_safe"] is True and not ev["satisfied"] and ev["detail"]["min_lr_dist"] > CONTACT_POSSIBLE_DIST, ev
    ev = p.update(s, a, 2)          # one step near the radio is enough to forbid a blind retry of this attempt
    s_far2 = s_far.copy()
    assert p.update(s_far2, a, 3)["retry_safe"] is False
    # press with the radio dropped -> failure
    s3 = s.copy(); s3[49:51] = 0.0
    p.start("press(radio)", "press", ("radio",), s, 0)
    assert p.update(s3, a, 1)["reason"] == "radio_dropped"
    # place on: release near the grasp height
    s[44] = 0.66
    p.start("place on(radio,coffee_table)", "place on", ("radio", "coffee_table"), s, 0)
    for t in range(1, 40):
        if t == 5:
            s[49:51] = 0.05
        ev = p.update(s, a, t)
        if ev["satisfied"] or ev["failed"]:
            break
    assert ev["satisfied"] and ev["reason"] == "released_at_surface_height", ev
    # place on: released 0.3 m above the table -> failure
    s[49:51] = 0.019; s[44] = 0.95
    p.start("place on(radio,coffee_table)", "place on", ("radio", "coffee_table"), s, 0)
    for t in range(1, 40):
        if t == 5:
            s[49:51] = 0.05
        ev = p.update(s, a, t)
        if ev["satisfied"] or ev["failed"]:
            break
    assert ev["failed"] and ev["reason"] == "released_away_from_surface", ev
    print("PASS task0_predicates self-test")


if __name__ == "__main__":
    _self_test()
