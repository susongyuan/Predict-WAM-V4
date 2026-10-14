#!/usr/bin/env python3
"""CPU test for the DIAGNOSTIC-ONLY advance mode (orchestrate.diagnostic_advance).

It must let the program advance past a skill whose outcome is unverifiable, while never producing
success=True, never satisfying a postcondition, and never updating reliability or memory.
This is a control-flow test, not a physical capability claim.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
from pathlib import Path
import unittest
import numpy as np

from harness.grounding import Catalog, CandidateProgram, Step
from harness.orchestrator import ConditionBank, Orchestrator, _FakeClosure

ROOT = Path(__file__).resolve().parents[1]
PICK = ("pick up from", ("trash_can", "floors"))
PLACE = ("place on next to", ("trash_can", "floors", "furniture_sink"))


def catalog():
    data = json.loads((ROOT / "data/skill_vtt_task090/contract_targets.json").read_text())
    cat = Catalog.__new__(Catalog); cat.patterns = {}; cat.classes = {}
    for key in list(data["contracts"]) + list(data["navigation"]):
        name, objects = key[:-1].split("(", 1)
        cat.patterns.setdefault(name, set()).add(tuple(objects.split(",")))
    cat.categories = sorted({o for rows in cat.patterns.values() for row in rows for o in row})
    return cat


def build(diagnostic_advance):
    cat = catalog()
    protos = {name: np.ones(1024, dtype=np.float32) for name in cat.patterns}
    bank = ConditionBank(protos)
    o = Orchestrator(catalog=cat, bank=bank, closure_monitor=_FakeClosure(1), precondition_monitor=None,
                     budgets={"move to(trash_can)": 2, "move to(furniture_sink)": 2,
                              "pick up from(trash_can,floors)": 3,
                              "place on next to(trash_can,floors,furniture_sink)": 3},
                     max_attempts=1, budget_factor=1, navigation_completion="docking", navigation_settle_steps=1,
                     manipulation_completion="postcondition", diagnostic_advance=diagnostic_advance)
    o.reset([CandidateProgram("diag", [Step(*PICK), Step(*PLACE)])])
    return o


def unknown(key, step):
    return {"verified": False, "satisfied": False, "failed": False, "status": "unknown",
            "verifier_kind": "perception_predicates", "source": "task090_observed:v1",
            "key": key, "observation_id": step, "detail": {"calibration_blocker": "missing_relation_calibration"}}


class DiagnosticAdvanceTest(unittest.TestCase):
    def drive(self, diagnostic_advance, steps=40):
        o = build(diagnostic_advance)
        events, directive, seen_place_start = [], None, False
        for t in range(steps):
            c = o.contract
            key = None if c is None else c.key
            nav = None
            if c is not None and c.is_navigation:
                nav = dict(docked=True, key=c.key, observation_id=c.started_step + (t - c.started_step))
            directive = o.tick(np.ones(1024), t, base_velocity=np.zeros(3),
                               postcondition=None if key is None else unknown(key, t),
                               navigation_evidence=nav)
            events.extend(o.events); o.events.clear()
            if any(e["event"] == "contract_start" and e["key"].startswith(PLACE[0]) for e in events):
                seen_place_start = True
            if directive.mode == "HOLD":
                break
        return o, events, directive, seen_place_start

    def test_default_still_holds_at_budget_with_success_false(self):
        o, events, directive, _ = self.drive(None)
        ends = [e for e in events if e["event"] == "contract_end" and e["key"].startswith(PICK[0])]
        self.assertTrue(ends, "the pick contract never ended")
        last = ends[-1]
        self.assertEqual(last["predicate"], "budget_without_verified_postcondition")
        self.assertIs(last["success"], False)
        self.assertNotIn("diagnostic_advance", last)
        self.assertFalse(any(e["event"] == "diagnostic_advance" for e in events))
        self.assertEqual(directive.mode, "HOLD")

    def test_diagnostic_mode_advances_without_any_success_claim(self):
        o, events, directive, place_started = self.drive("on_contract_budget")
        ends = [e for e in events if e["event"] == "contract_end" and e["key"].startswith(PICK[0])]
        self.assertTrue(ends, "the pick contract never ended")
        last = ends[-1]
        self.assertEqual(last["predicate"], "diagnostic_unverified_advance")
        self.assertIs(last["success"], None)          # never True, never a declared failure
        self.assertIs(last["verified"], False)
        adv = [e for e in events if e["event"] == "diagnostic_advance"]
        self.assertTrue(adv)
        self.assertIs(adv[0]["success"], None)
        self.assertIs(adv[0]["verified"], False)
        # the program moved on: a contract for the NEXT key was started
        self.assertTrue(place_started, "the place contract never started")
        # nothing was promoted: no reliability, no memory arm
        self.assertFalse(any(a.successes or a.failures or a.support_count for arms in o.bank.arms.values() for a in arms))
        # navigation contracts legitimately succeeded; no MANIPULATION outcome may ever be True here
        self.assertFalse(any(r["success"] is True for r in o.realized if r["skill"] != "move to"))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            build("advance_everything")


if __name__ == "__main__":
    unittest.main(verbosity=2)
