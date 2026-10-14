#!/usr/bin/env python3
"""CPU counterexamples for evaluator-only grasp evidence and result aggregation."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.summarize_task090_checkpoints import grasp_result

ROOT = Path(__file__).resolve().parents[1]


class GraspResults(unittest.TestCase):
    def setUp(self):
        self.score = dict(task="composting_waste", instance_id=311, rollout_id=0,
                          policy_episode=2, policy_step_alignment=True, steps=30)
        self.trace = [dict(event="orch_contract_start", episode=2, step=10, index=1,
                           key="pick up from(trash_can,floors)"),
                      dict(event="chunk", episode=2, step=10),
                      dict(event="orch_contract_end", episode=2, step=30, index=1, success=False)]
        self.poses = [dict(type="header", episode_identity={k: self.score[k] for k in (
            "task", "instance_id", "rollout_id", "policy_episode")},
            entities={"ashcan.n.01_1": dict(category="trash_can", aabb_at_start_world=[[0, 0, 0], [1, 1, 1]])})]
        for step in range(10, 31):
            self.poses.append(dict(step=step, manipulation=dict(status="available", grasping_mode="assisted",
                assisted_grasp={"right": dict(object_instance="ashcan.n.01_1", constraint_active=True,
                                             release_pending=False, aabb_min_z=0.04)})))

    def result(self):
        return grasp_result(self.score, self.trace, self.poses)

    def test_valid_assisted_lift_is_not_task_success(self):
        self.assertTrue(self.result()["success"])
        self.assertEqual(self.result()["first_grasp_step"], 10)
        self.assertNotIn("task_success", self.result())

    def test_navigation_only_not_grasp_failure(self):
        self.trace = []
        self.assertEqual(self.result()["status"], "not_reached_manipulation")
        self.assertIsNone(self.result()["success"])

    def test_pose_rise_without_constraint_rejected(self):
        for row in self.poses[1:]:
            row["manipulation"]["assisted_grasp"]["right"]["constraint_active"] = False
        self.assertFalse(self.result()["success"])

    def test_wrong_object_rejected(self):
        for row in self.poses[1:]:
            row["manipulation"]["assisted_grasp"]["right"]["object_instance"] = "fruit"
        self.assertFalse(self.result()["success"])

    def test_pending_release_rejected(self):
        for row in self.poses[1:]:
            row["manipulation"]["assisted_grasp"]["right"]["release_pending"] = True
        self.assertFalse(self.result()["success"])

    def test_no_lift_rejected(self):
        for row in self.poses[1:]:
            row["manipulation"]["assisted_grasp"]["right"]["aabb_min_z"] = 0.02
        self.assertFalse(self.result()["success"])

    def test_gap_breaks_consecutive_support(self):
        self.poses.pop(11)
        self.assertIsNone(self.result()["success"])
        self.assertLess(self.result()["longest_held_lift_steps"], 15)

    def test_wrong_episode_unknown(self):
        self.poses[0]["episode_identity"]["policy_episode"] = 99
        self.assertIsNone(self.result()["success"])

    def test_missing_evidence_unknown(self):
        for row in self.poses[1:]:
            row.pop("manipulation")
        self.assertIsNone(self.result()["success"])

    def test_invalid_step_alignment_unknown(self):
        self.score["policy_step_alignment"] = False
        self.assertIsNone(self.result()["success"])

    def test_no_mutation_of_inputs(self):
        old = copy.deepcopy((self.score, self.trace, self.poses))
        self.result()
        self.assertEqual(old, (self.score, self.trace, self.poses))


class EvaluatorGetter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/eval/evaluator.py"
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_manipulation_pose_evidence")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        cls.get_evidence = staticmethod(namespace[method.name])

    def make(self, pending=None):
        obj = SimpleNamespace(name="test_bin", aabb=([0, 0, 0.1], [1, 1, 0.8]))
        robot = SimpleNamespace(grasping_mode="assisted", eef_links={"right": object()},
                                _ag_obj_in_hand={"right": obj}, _ag_release_counter={"right": pending})
        return SimpleNamespace(robot=robot, _pose_log_entities={"ashcan.n.01_1": obj}, _pose7=lambda obj: [0]*7)

    def test_active_target_is_read_only(self):
        fake = self.make()
        evidence = self.get_evidence(fake)
        self.assertEqual(evidence["assisted_grasp"]["right"]["object_instance"], "ashcan.n.01_1")
        self.assertTrue(evidence["assisted_grasp"]["right"]["constraint_active"])
        self.assertEqual(evidence["assisted_grasp"]["right"]["aabb_min_z"], 0.1)
        self.assertIsNone(fake.robot._ag_release_counter["right"])

    def test_zero_release_counter_not_active(self):
        evidence = self.get_evidence(self.make(pending=0))
        self.assertFalse(evidence["assisted_grasp"]["right"]["constraint_active"])
        self.assertTrue(evidence["assisted_grasp"]["right"]["release_pending"])

    def test_missing_fields_unknown_not_false_grasp(self):
        self.assertEqual(self.get_evidence(SimpleNamespace())["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
