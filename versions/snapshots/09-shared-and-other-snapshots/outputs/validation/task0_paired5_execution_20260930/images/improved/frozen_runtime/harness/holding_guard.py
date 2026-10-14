"""Episode-scoped gripper closure commitments, not an object detector or a success verifier.

post30k_holdguard_v1 is adapted ONLY to Task0's calibrated right-hand radio predicates.
The hand/object interface is explicit so a future adapter need not lock an unrelated hand.
No compiled carried state, demonstration pose, hidden world state or action history is evidence.
"""
from __future__ import annotations

import numpy as np

from harness.task0_predicates import FINGERS, GRIP_ACTION, GUARDED_SOURCE, HOLD_RANGE, OPEN_MIN

VERSION = "post30k_holdguard_v1"


class HoldingGuard:
    def __init__(self, episode=0):
        self.episode = None
        self.reset_episode(episode)

    def reset_episode(self, episode):
        if type(episode) is not int or (self.episode is not None and episode <= self.episode):
            raise ValueError("holding guard requires a new episode id on reset")
        self.episode = episode
        self.holds = {}                 # hand -> verified provenance + sticky loss status
        self.events = []
        self._accepted = set()
        self._release_commanded = set()

    def _check_episode(self, episode):
        if episode != self.episode:
            raise ValueError("holding guard episode mismatch")

    def _event(self, kind, step, **detail):
        self.events.append({"event": "holding_guard_" + kind, "version": VERSION,
                            "episode": self.episode, "step": int(step), **detail})

    def accept_pick(self, evidence, *, episode, step, key, objects, hand, started_step):
        """Consume only the live predicate output, NEVER an old contract_end or carried declaration.

        Matching objects is exact (instance tags included), unlike the category-level contract key.
        The trusted predicate has already checked the consecutive closed-grasp and EEF lift thresholds.
        """
        self._check_episode(episode)
        e = evidence
        if (hand not in GRIP_ACTION or not objects or not isinstance(e, dict)
                or e.get("source") != GUARDED_SOURCE
                or e.get("verified") is not True or e.get("verifier_kind") != "controller_postconditions"
                or e.get("outcome_observable") is not True
                or e.get("satisfied") is not True or e.get("failed") is not False
                or e.get("reason") != "held_and_lifted"
                or type(e.get("episode")) is not int or e["episode"] != episode
                or type(e.get("observation_id")) is not int or e["observation_id"] != step
                or type(e.get("started_step")) is not int or e["started_step"] != started_step
                or started_step >= step or e.get("key") != key
                or e.get("objects") != list(objects) or e.get("hand") != hand):
            return False
        token = (episode, key, started_step, step)
        if token in self._accepted:
            return False
        # A second object cannot silently replace a still-held object on this hand.
        if hand in self.holds and not self.holds[hand]["lost"]:
            return False
        self._accepted.add(token)
        self.holds[hand] = {"object": objects[0], "key": key, "verified_step": int(step),
                            "started_step": started_step, "lost": False}
        self._event("latched", step, hand=hand, **self.holds[hand])
        return True

    def is_holding(self, obj, hand):
        h = self.holds.get(hand)
        return bool(h is not None and h["object"] == obj and not h["lost"])

    def observe(self, state, *, episode, step):
        """Loss invalidates the evidence, but keeps a conservative close command until explicit recovery.

        Finger opening after an authorized place command retires the commitment, NOT a placement success.
        Returning to half-closed fingers cannot heal loss; only another verified pick can do that.
        """
        self._check_episode(episode)
        s = np.asarray(state)
        finite = s.shape == (61,) and bool(np.isfinite(s).all())
        for hand, h in list(self.holds.items()):
            fs = float(s[FINGERS[hand]].sum()) if s.shape == (61,) else float("nan")
            if finite and hand in self._release_commanded and fs >= OPEN_MIN:
                del self.holds[hand]
                self._event("release_observed", step, hand=hand, object=h["object"], placement_verified=False)
                continue
            # A commanded place may temporarily leave HOLD_RANGE while the fingers open. Do not turn
            # that into a held claim either: loss remains sticky, and only this exact place may continue.
            if not finite or not HOLD_RANGE[0] <= fs <= HOLD_RANGE[1]:
                if not h["lost"]:
                    h["lost"] = True
                    self._event("lost", step, hand=hand, object=h["object"],
                                reason="nonfinite_state" if not finite else "outside_hold_range",
                                finger_sum=fs if np.isfinite(fs) else None)

    def apply(self, action, state, *, episode, step, skill=None, objects=(), hand=None,
              started_step=None, executing=False):
        """Apply on EVERY action return/dequeue, after routing/clipping; never change a queued model chunk."""
        self.observe(state, episode=episode, step=step)
        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (23,) or not np.isfinite(raw).all():
            raise ValueError("holding guard cannot return a nonfinite or malformed action")
        applied = raw.copy()
        self._release_commanded.clear()
        for held_hand, h in list(self.holds.items()):
            matching = executing and hand == held_hand and bool(objects) and objects[0] == h["object"]
            idx = GRIP_ACTION[held_hand]
            if matching and skill in ("place on", "place in") and len(objects) >= 2:
                if raw[idx] > 0:
                    self._release_commanded.add(held_hand)
                    self._event("release_command", step, hand=held_hand, object=h["object"],
                                skill=skill, gripper=float(raw[idx]), placement_verified=False)
                continue
            if (matching and skill == "pick up from" and h["lost"] and started_step is not None
                    and started_step > h["verified_step"]):
                del self.holds[held_hand]
                self._event("regrasp", step, hand=held_hand, object=h["object"], started_step=started_step)
                continue
            applied[idx] = -1.0
        changed = np.flatnonzero(applied != raw)
        if len(changed):
            self._event("rewrite", step, skill=skill, executing=bool(executing),
                        changed_indices=changed.tolist(), raw_action=raw.tolist(), applied_action=applied.tolist(),
                        commitments={k: dict(v) for k, v in self.holds.items()})
        return applied
