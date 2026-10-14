#!/usr/bin/env python3
"""CPU judge tests; these do not establish physical expert/policy success."""
import itertools
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.zhuhai_task1_rules import DESTINATIONS, PhysicalState, Task1Judge


def state(center, released=True, contact=False, extent=(.04, .04, .04), velocity=(0, 0, 0)):
    corners = [[center[i]+sign[i]*extent[i]/2 for i in range(3)]
               for sign in itertools.product((-1, 1), repeat=3)]
    return PhysicalState(corners, velocity, (0, 0, 0), released, touching_robot=contact)


class Task1RulesTest(unittest.TestCase):
    def setUp(self):
        self.judge = Task1Judge()
        self.bounds = {"dry": ((-.4, -.1, .74), (-.1, .2, .95)),
                       "wet": ((.1, -.1, .74), (.4, .2, .95))}
        self.states = {key: state((0, -.2, .78)) for key in DESTINATIONS}
        self.lids = {"dry": -1.65, "wet": -1.65}

    def observe(self, seconds, states=None, lids=None, fixed=True):
        for _ in range(round(seconds/.01)):
            self.judge.observe(.01, states or self.states, self.bounds, .74,
                               lids or self.lids, bin_fixed=fixed)

    def test_five_second_hold_requires_lift_contact_and_continuity(self):
        self.states["milk_carton"] = state((0, 0, .90), released=False, contact=True)
        self.observe(4.99)
        self.assertFalse(self.judge.grasp_awarded["milk_carton"])
        self.observe(.01)
        self.assertTrue(self.judge.grasp_awarded["milk_carton"])
        self.states["squeeze_bottle"] = state((0, 0, .90), released=False, contact=True)
        self.observe(3)
        self.states["squeeze_bottle"] = state((0, 0, .90), released=True)
        self.observe(.01)
        self.states["squeeze_bottle"] = state((0, 0, .90), released=False, contact=True)
        self.observe(3)
        self.assertFalse(self.judge.grasp_awarded["squeeze_bottle"])

    def test_requires_open_mouth_passage_not_final_teleport(self):
        self.states["milk_carton"] = state((-.25, .05, .80))
        self.observe(2)
        self.assertFalse(self.judge.placement_valid["milk_carton"])
        self.states["milk_carton"] = state((-.25, .05, 1.02), released=False, contact=True)
        self.observe(.01)
        self.states["milk_carton"] = state((-.25, .05, .80))
        self.observe(1.01)
        self.assertTrue(self.judge.placement_valid["milk_carton"])
        self.states["milk_carton"] = state((-.10, .05, .80))
        self.observe(.01)
        self.assertFalse(self.judge.placement_valid["milk_carton"])

    def test_wrong_bin_held_or_moving_never_valid(self):
        self.states["banana_peel"] = state((-.25, .05, 1.02))
        self.observe(.01)
        self.states["banana_peel"] = state((-.25, .05, .80))
        self.observe(2)
        self.assertFalse(self.judge.placement_valid["banana_peel"])
        self.states["banana_peel"] = state((.25, .05, 1.02))
        self.observe(.01)
        self.states["banana_peel"] = state((.25, .05, .80), released=False, contact=True)
        self.observe(2)
        self.assertFalse(self.judge.placement_valid["banana_peel"])
        self.states["banana_peel"] = state((.25, .05, .80), velocity=(.04, 0, 0))
        self.observe(2)
        self.assertFalse(self.judge.placement_valid["banana_peel"])

    def test_full_thirty_points_and_current_lid_requirement(self):
        self.states = {key: state((0, 0, .90), released=False, contact=True) for key in DESTINATIONS}
        self.observe(5.01)
        self.states = {key: state((-.25 if color == "dry" else .25, .05, 1.02),
                                 released=False, contact=True) for key, color in DESTINATIONS.items()}
        self.observe(.01)
        self.states = {key: state((-.25 if color == "dry" else .25, .05, .80))
                       for key, color in DESTINATIONS.items()}
        self.observe(1.01)
        self.assertEqual(self.judge.report()["base_score"], 30)
        self.assertTrue(self.judge.report()["full_task_success"])
        self.assertFalse(self.judge.report()["final_lid_closed_required"])
        self.observe(.01, lids={"dry": 0, "wet": 0})
        self.assertFalse(self.judge.report()["full_task_success"])
        self.assertTrue(self.judge.lid_open_awarded["dry"])


if __name__ == "__main__":
    unittest.main()
