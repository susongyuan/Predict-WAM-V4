#!/usr/bin/env python3
"""CPU stub tests only. AST-extract the real predicate, never import RoboTwin/SAPIEN.

Run: .venvs/robotwin-sim/bin/python -B scripts/test_robotwin_native_predicate_evidence.py
No scene creation, model loading, GPU, files written or integration into snapshot.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace as NS
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.robotwin_native_diagnostics import native_diagnostics


def source_method(path, class_name, method_name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[method_name]


native_check = source_method("third_party/RoboTwin/envs/put_object_cabinet.py", "put_object_cabinet", "check_success")
left_open = source_method("third_party/RoboTwin/envs/robot/robot.py", "Robot", "is_left_gripper_open")
right_open = source_method("third_party/RoboTwin/envs/robot/robot.py", "Robot", "is_right_gripper_open")


class Joint:
    def __init__(self, name, target, dof=1):
        self.name, self.target, self.dof = name, target, dof

    def get_name(self):
        return self.name

    def get_dof(self):
        return self.dof

    def get_drive_target(self):
        return np.array([self.target])


class Robot:
    is_left_gripper_open = left_open
    is_right_gripper_open = right_open

    def __init__(self):
        self.left_gripper_val, self.right_gripper_val = 1.0, 0.0
        for side in ("left", "right"):
            joints = [Joint("a", 0.25), Joint("b", 0.75), Joint("jaw", 0.0)]
            # Deliberately permuted arm order; jaw is not returned as measured arm qpos.
            setattr(self, f"{side}_arm_joints", [joints[1], joints[0]])
            setattr(self, f"{side}_entity", NS(get_active_joints=lambda js=joints: js,
                    get_qpos=lambda: np.array([0.5, 0.5, 0.1])))


class Cabinet:
    def __init__(self):
        self.target = [0., 0., 42., 1., 0., 0., 0.]
        self.indices = []

    def get_functional_point(self, index):
        self.indices.append(index)
        if index != 0:
            raise AssertionError("wrong native functional point index")
        return self.target

    def get_pose(self):
        return NS(p=np.array([3., 4., 5.]), q=np.array([1., 0., 0., 0.]))

    def get_qpos(self):
        return np.array([0.12, 0., 0.])


class Env:
    task_name = "put_object_cabinet"
    check_success = native_check

    def __init__(self):
        self.xyz = np.array([0., 0., 0.05])
        self.object = NS(get_pose=lambda: NS(p=self.xyz, q=np.array([1., 0., 0., 0.])),
                         get_functional_point=lambda *a: self.forbidden())
        self.cabinet, self.robot = Cabinet(), Robot()
        self.origin_z, self.arm_tag, self.eval_success = np.float64(0.), "left", False
        self.scene = NS(get_timestep=lambda: 0.004, step=self.forbidden)
        self.take_action = self.forbidden

    def forbidden(self, *args):
        raise AssertionError("diagnostic attempted an unused point, action or physics step")


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()

    def evidence(self):
        d = native_diagnostics(self.env)
        json.dumps(d, allow_nan=False)
        return d, d["native_predicate_evidence"]

    def test_native_inputs_and_reference_index(self):
        d, p = self.evidence()
        self.assertEqual(d["errors"], {})
        self.assertTrue(p["native_predicate"])
        self.assertTrue(p["recomputed_matches_native"])
        self.assertFalse(p["native_success_latch"])
        self.assertEqual(self.env.cabinet.indices, [0, 0])
        self.assertIsNone(p["object_functional_point_index"])
        self.assertEqual(p["object_functional_point_status"], "not_used_by_native_predicate")
        self.assertEqual(p["cabinet_functional_point_world"], [0., 0., 42.])
        self.assertEqual(p["cabinet_pose_world"][:3], [3., 4., 5.])
        self.assertEqual(p["origin_z"], 0.)

    def test_strict_threshold_boundaries_against_original_method(self):
        for field, value, expected in [("x", .05, False), ("x", -.05, False),
                ("y", .05, False), ("y", -.05, False), ("x", .049999, True),
                ("z", .007, False), ("z", .120, False), ("z", .007001, True),
                ("z", .119999, True), ("grip", .8, False), ("grip", .800001, True)]:
            with self.subTest(field=field, value=value):
                self.env = Env()
                if field == "grip":
                    self.env.robot.left_gripper_val = value
                else:
                    self.env.xyz["xyz".index(field)] = value
                _, p = self.evidence()
                self.assertEqual(p["native_predicate"], expected)
                self.assertEqual(p["recomputed_predicate"], expected)

    def test_origin_z_not_current_z_and_target_z_not_height_reference(self):
        self.env.origin_z = np.float64(0.045)
        _, p = self.evidence()
        self.assertFalse(p["native_predicate"])
        self.assertFalse(p["recomputed_predicate"])
        self.assertAlmostEqual(p["height_delta_m"], .005)

    def test_right_arm_command_not_left_and_not_measured_opening(self):
        self.env.arm_tag = "right"
        _, p = self.evidence()
        self.assertFalse(p["native_predicate"])
        self.env.robot.right_gripper_val = .9
        _, p = self.evidence()
        self.assertTrue(p["native_predicate"])
        self.assertEqual(p["selected_gripper_command"], .9)
        self.assertEqual(p["selected_gripper_arm"], "right")

    def test_armtag_like_native_equality(self):
        class Tag:
            def __eq__(self, value):
                return value == "left"
        self.env.arm_tag = Tag()
        _, p = self.evidence()
        self.assertEqual(p["arm_tag"], "left")
        self.assertTrue(p["recomputed_matches_native"])

    def test_unknown_arm_not_silently_right(self):
        self.env.arm_tag = "invalid"
        d, p = self.evidence()
        self.assertIsNone(p["recomputed_predicate"])
        self.assertIn("predicate.arm_tag", d["errors"])

    def test_saved_operands_can_recompute_after_json_roundtrip(self):
        for dtype in (np.float32, np.float64):
            for dz in (.007, .007001, .119999, .12):
                with self.subTest(dtype=dtype, dz=dz):
                    self.env.xyz = np.array([.049999, 0., dz], dtype=dtype)
                    self.env.origin_z = dtype(0.)
                    d = json.loads(json.dumps(native_diagnostics(self.env), allow_nan=False))
                    p = d["native_predicate_evidence"]
                    types = p["operand_dtypes"]
                    obj = np.asarray(p["object_position_world"], dtype=types["object_position_world"])
                    target = np.asarray(p["cabinet_functional_point_world"], dtype=types["cabinet_functional_point_world"])
                    origin = (p["origin_z"] if p["operand_scalar_kinds"]["origin_z"] == "python"
                              else np.asarray(p["origin_z"], dtype=types["origin_z"])[()])
                    grip = (p["selected_gripper_command"] if p["operand_scalar_kinds"]["left_gripper_command"] == "python"
                            else np.asarray(p["selected_gripper_command"], dtype=types["left_gripper_command"])[()])
                    t = p["thresholds"]
                    recomputed = bool(np.all(abs(obj[:2]-target[:2]) < np.array(t["abs_xy_lt_m"]))
                        and obj[2]-origin > t["height_delta_gt_m"]
                        and obj[2]-origin < t["height_delta_lt_m"] and grip > t["gripper_command_gt"])
                    self.assertEqual(recomputed, p["native_predicate"])
                    self.assertTrue(p["recomputed_matches_native"])

    def test_python_and_numpy_scalar_promotion_matches_native(self):
        for scalar in (float, np.float32, np.float64):
            for value in (.007, .12, .807):
                with self.subTest(scalar=scalar, value=value):
                    self.env.xyz = np.array([0., 0., value], dtype=np.float32)
                    self.env.origin_z = scalar(.8 if value == .807 else 0.)
                    self.env.robot.left_gripper_val = scalar(.80000001)
                    _, p = self.evidence()
                    self.assertTrue(p["recomputed_matches_native"])

    def test_latch_is_separate_no_score_mutation(self):
        self.env.eval_success = True
        self.env.xyz[0] = 1.
        before = self.env.xyz.copy()
        _, p = self.evidence()
        self.assertFalse(p["native_predicate"])
        self.assertTrue(p["native_success_latch"])
        self.assertTrue(self.env.eval_success)
        np.testing.assert_array_equal(self.env.xyz, before)

    def test_missing_origin_and_native_exception_are_unknown(self):
        del self.env.origin_z
        d, p = self.evidence()
        self.assertIsNone(p["origin_z"])
        self.assertIsNone(p["native_predicate"])
        self.assertIsNone(p["recomputed_predicate"])
        self.assertIn("predicate.origin_z", d["errors"])
        self.assertIn("predicate.native_predicate", d["errors"])
        self.assertIsNotNone(d["arm_joint_tracking"]["left"])

    def test_bad_target_shape_missing_and_nonfinite(self):
        for target in (None, [0., 0.], [0., float("nan"), 0.], [[0., 0., 0.]]):
            with self.subTest(target=target):
                self.env.cabinet.target = target
                d, p = self.evidence()
                self.assertIsNone(p["cabinet_functional_point_world"])
                self.assertIsNone(p["recomputed_predicate"])
                self.assertIn("predicate.cabinet_functional_point_world", d["errors"])

    def test_nonfinite_command_is_not_success_evidence(self):
        self.env.robot.left_gripper_val = float("inf")
        d, p = self.evidence()
        self.assertIsNone(p["selected_gripper_command"])
        self.assertIsNone(p["recomputed_predicate"])
        self.assertIn("predicate.gripper_commands.left", d["errors"])

    def test_auxiliary_cabinet_pose_failure_does_not_block_predicate(self):
        self.env.cabinet.get_pose = lambda: 1 / 0
        d, p = self.evidence()
        self.assertIsNone(p["cabinet_pose_world"])
        self.assertTrue(p["recomputed_predicate"])
        self.assertIn("predicate.cabinet_pose_world", d["errors"])

    def test_native_disagreement_and_nonboolean_are_not_coerced(self):
        self.env.check_success = lambda: False
        _, p = self.evidence()
        self.assertFalse(p["recomputed_matches_native"])
        self.assertTrue(p["recomputed_predicate"])
        self.env.check_success = lambda: "success"
        d, p = self.evidence()
        self.assertIsNone(p["native_predicate"])
        self.assertIn("predicate.native_predicate", d["errors"])

    def test_unsupported_task_does_not_call_other_predicate(self):
        self.env.task_name = "another_task"
        self.env.check_success = self.env.forbidden
        d = native_diagnostics(self.env)
        self.assertIsNone(d["native_predicate_evidence"])
        self.assertIn("native_predicate_evidence", d["errors"])
        self.assertEqual(d["scene_dt_seconds"], .004)

    def test_arm_qpos_is_measured_mapped_and_signed(self):
        d, _ = self.evidence()
        arm = d["arm_joint_tracking"]["left"]
        self.assertEqual(arm["active_joint_indices"], [1, 0])
        self.assertEqual(arm["actual_qpos"], [.5, .5])
        self.assertEqual(arm["drive_target"], [.75, .25])
        self.assertEqual(arm["actual_minus_drive_target"], [-.25, .25])
        self.assertFalse(arm["grippers_included"])

    def test_ambiguous_joint_mapping_is_null(self):
        for fault in ("multi_dof", "foreign_joint", "duplicate", "bad_qpos", "nonfinite_target"):
            with self.subTest(fault=fault):
                self.env = Env()
                joints = self.env.robot.left_arm_joints
                if fault == "multi_dof":
                    joints[0].dof = 2
                elif fault == "foreign_joint":
                    joints[0] = Joint("a", .1)
                elif fault == "duplicate":
                    joints[0] = joints[1]
                elif fault == "bad_qpos":
                    self.env.robot.left_entity.get_qpos = lambda: np.array([0.])
                else:
                    joints[0].target = float("nan")
                d, p = self.evidence()
                self.assertIsNone(d["arm_joint_tracking"]["left"])
                self.assertIn("arm_joint_tracking.left", d["errors"])
                self.assertIsNotNone(d["arm_joint_tracking"]["right"])
                self.assertTrue(p["native_predicate"])

    def test_scene_dt_not_elapsed_or_control_time(self):
        d, _ = self.evidence()
        self.assertEqual(d["scene_dt_seconds"], .004)
        self.assertIsNone(d["scene_elapsed_seconds"])
        for value in (0., -.1, float("inf"), float("nan"), [0.004]):
            self.env.scene.get_timestep = lambda value=value: value
            d, _ = self.evidence()
            self.assertIsNone(d["scene_dt_seconds"])
            self.assertIn("scene_dt_seconds", d["errors"])
        del self.env.scene.get_timestep
        d, _ = self.evidence()
        self.assertIsNone(d["scene_dt_seconds"])
        self.assertIn("scene_dt_seconds", d["errors"])

    def test_cancellation_is_not_swallowed(self):
        def cancel():
            raise KeyboardInterrupt()
        self.env.object.get_pose = cancel
        with self.assertRaises(KeyboardInterrupt):
            native_diagnostics(self.env)

    def test_no_imports_of_gpu_or_simulator(self):
        # Isolate from other CPU suites which may have imported optional bindings.
        code = """import sys
from types import SimpleNamespace
from harness.robotwin_native_diagnostics import native_diagnostics
native_diagnostics(SimpleNamespace())
assert not any(k == 'torch' or k.startswith('torch.') or k == 'sapien'
               or k.startswith('sapien.') or k == 'harness.robotwin_policy_worker'
               for k in sys.modules)
"""
        completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
