#!/usr/bin/env python3
"""Adversarial regression checks for task-2 physical score admission."""
import json
from pathlib import Path
import sys
import unittest
from dataclasses import replace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.zhuhai_task2_rules import PhysicalState,Task2Judge,wholly_contained


def state(x=.1,z=.8,**kwargs):
    return PhysicalState([[x-.01,.09,z],[x+.01,.11,z+.03]],[0,0,0],[0,0,0],
        released=True,support_destinations=["white","green","blue"],**kwargs)


class Rules(unittest.TestCase):
    def setUp(self):
        self.bounds={color:([0,0,.74],[.2,.2,1.]) for color in ["white","green","blue"]}
        self.states={key:state() for key in ["bowl","cup","plate","spoon"]}
        self.judge=Task2Judge("life","body_text")

    def test_body_mapping_and_thirty_points(self):
        self.assertEqual(self.judge.destinations,{"bowl":"white","spoon":"white","cup":"green","plate":"blue"})
        self.assertEqual(self.judge.report()["base_score_max"],30)

    def test_initial_zero_snapshot(self):
        self.judge.observe(0,self.states,self.bounds,.74)
        self.assertEqual(self.judge.report()["base_score"],0)

    def test_life_requires_continuous_five_seconds_real_gripper(self):
        held={key:replace(value,released=False,touching_robot=True,touching_gripper=True) for key,value in self.states.items()}
        for _ in range(1249):self.judge.observe(.004,held,self.bounds,.74)
        self.assertFalse(any(self.judge.grasp_awarded.values()))
        self.judge.observe(.004,held,self.bounds,.74)
        self.assertTrue(all(self.judge.grasp_awarded.values()))

    def test_interrupted_hold_resets(self):
        held={key:replace(value,released=False,touching_robot=True,touching_gripper=True) for key,value in self.states.items()}
        self.judge.observe(4,held,self.bounds,.74)
        self.judge.observe(.1,self.states,self.bounds,.74)
        self.judge.observe(1.1,held,self.bounds,.74)
        self.assertFalse(any(self.judge.grasp_awarded.values()))

    def test_arm_contact_cannot_award_grasp(self):
        arm={key:replace(value,released=False,touching_robot=True,touching_gripper=False) for key,value in self.states.items()}
        self.judge.observe(5.1,arm,self.bounds,.74)
        self.assertFalse(any(self.judge.grasp_awarded.values()))

    def test_placement_needs_release_stability_full_containment_support(self):
        self.judge.observe(1.1,self.states,self.bounds,.74)
        self.assertTrue(all(self.judge.placement_valid.values()))
        for invalid in [replace(state(),released=False),replace(state(),touching_robot=True),
                        replace(state(),linear_velocity=[.02,0,0]),replace(state(),support_destinations=[]),state(x=.205)]:
            self.states["plate"]=invalid
            self.judge.observe(.004,self.states,self.bounds,.74)
            self.assertFalse(self.judge.placement_valid["plate"])

    def test_tipped_cup_rejected_grasp_retained(self):
        held={key:replace(value,released=False,touching_robot=True,touching_gripper=True) for key,value in self.states.items()}
        self.judge.observe(5.1,held,self.bounds,.74)
        self.states["cup"]=replace(self.states["cup"],upright_cosine=.95)
        self.judge.observe(1.1,self.states,self.bounds,.74)
        self.assertFalse(self.judge.placement_valid["cup"])
        self.assertTrue(self.judge.grasp_awarded["cup"])

    def test_nan_and_malformed_bounds_rejected(self):
        self.assertFalse(wholly_contained([[float('nan'),.1,.8]],[0,0,.74],[.2,.2,1]))
        self.assertFalse(wholly_contained([[.1,.1]],[0,0,.74],[.2,.2,1]))
        self.assertFalse(wholly_contained([],[0,0,.74],[.2,.2,1]))
        with self.assertRaises(ValueError):self.judge.observe(-.004,self.states,self.bounds,.74)

    def test_final_success_and_reversal(self):
        held={key:replace(value,released=False,touching_robot=True,touching_gripper=True) for key,value in self.states.items()}
        self.judge.observe(5.1,held,self.bounds,.74)
        self.judge.observe(1.1,self.states,self.bounds,.74)
        self.assertTrue(self.judge.report()["full_task_success"])
        self.assertEqual(self.judge.report()["base_score"],30)
        self.states["spoon"]=state(x=.25)
        self.judge.observe(.004,self.states,self.bounds,.74)
        self.assertFalse(self.judge.report()["full_task_success"])
        self.assertTrue(self.judge.grasp_awarded["spoon"])


if __name__=='__main__':unittest.main()
