#!/usr/bin/env python3
"""Native USD CPU checks; no simulator launch, rendering, or success-rate claim."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdSemantics
from harness.isaac_zhuhai_camera import head_pose, projection_report
from harness.isaac_zhuhai_task1 import RuntimeJudge, build, lid_arc_world_poses, scene_spec


class IsaacTask1Test(unittest.TestCase):
    def test_native_usd_hinges_mass_colliders_and_three_camera_contract(self):
        stage = Usd.Stage.CreateInMemory()
        manifest = build(stage)
        self.assertEqual(set(manifest["objects"]), {"milk_carton", "squeeze_bottle", "banana_peel"})
        self.assertEqual(manifest["cameras"]["count"], 3)
        self.assertFalse(any(prim.IsA(UsdGeom.Camera) for prim in stage.Traverse()))
        for color, spec in manifest["bins"].items():
            lid = stage.GetPrimAtPath(spec["lid_path"])
            self.assertTrue(lid.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(UsdPhysics.RigidBodyAPI(lid).GetKinematicEnabledAttr().Get())
            self.assertGreater(UsdPhysics.MassAPI(lid).GetMassAttr().Get(), 0)
            joint = UsdPhysics.RevoluteJoint.Get(stage, spec["joint_path"])
            self.assertLess(joint.GetLowerLimitAttr().Get(), -90)
            self.assertEqual(joint.GetAxisAttr().Get(), "X")
            self.assertFalse(UsdPhysics.DriveAPI.Get(joint.GetPrim(), "angular"))
            self.assertEqual(list(joint.GetBody0Rel().GetTargets()), [])
            self.assertEqual(str(joint.GetBody1Rel().GetTargets()[0]), spec["lid_path"])
        for name, obj in manifest["objects"].items():
            labels = UsdSemantics.LabelsAPI.Get(stage.GetPrimAtPath(obj["prim_path"]), "class")
            self.assertEqual(list(labels.GetLabelsAttr().Get()), [name])
            for part in obj["parts"]:
                self.assertTrue(stage.GetPrimAtPath(obj["prim_path"]+"/"+part["name"]).HasAPI(UsdPhysics.CollisionAPI))
        with self.assertRaises(ValueError):
            build(stage)
        json.dumps(manifest, allow_nan=False)

    def test_seeded_domains_and_head_full_geometry_projection(self):
        first = scene_spec(seed=32, randomized=True)
        self.assertEqual(first, scene_spec(seed=32, randomized=True))
        self.assertNotEqual(first["objects"], scene_spec(seed=33, randomized=True)["objects"])
        for seed in (0, 1, 32, 99):
            manifest = scene_spec(seed=seed, randomized=True)
            projection = projection_report(manifest["geometry_world_m"], head_pose())
            self.assertTrue(projection["all_geometry_in_frame"])
            for color, spec in manifest["bins"].items():
                # The head must retain full open lid geometry as well as the
                # initial closed scene; rendering verifies occlusion separately.
                angle = -1.68
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                anchor = np.array(spec["hinge_world_m"])
                local = np.array(spec["lid_local_corners_m"])-spec["hinge_in_lid_m"]
                opened = local@rotation.T+anchor
                self.assertTrue(projection_report({f"lid_{color}": opened}, head_pose())["all_geometry_in_frame"])

    def test_minimum_jerk_lid_arc_preserves_rigid_offset(self):
        spec = scene_spec()["bins"]["dry"]
        start = np.array(spec["closed_handle_transform_world"])
        poses = lid_arc_world_poses(spec, start)
        np.testing.assert_allclose(poses[0], start)
        anchor = np.array(spec["hinge_world_m"])
        norms = [np.linalg.norm(pose[:3, 3]-anchor) for pose in poses]
        self.assertLess(np.ptp(norms), 1e-12)
        displacement = np.linalg.norm(np.diff([pose[:3, 3] for pose in poses], axis=0), axis=1)
        self.assertLess(displacement[0], displacement[len(displacement)//2]/100)
        self.assertLess(displacement[-1], displacement[len(displacement)//2]/100)

    def test_judge_requires_measured_lids_and_actual_contacts(self):
        manifest = scene_spec()
        judge = RuntimeJudge(manifest, robot_prim_paths=["/World/Robot"],
                              gripper_prim_paths=["/World/Robot/finger"])
        states = {name: {"position_world_m": obj["initial_position_world_m"],
                         "quaternion_wxyz": [1, 0, 0, 0], "linear_velocity_m_s": [0, 0, 0],
                         "angular_velocity_rad_s": [0, 0, 0]} for name, obj in manifest["objects"].items()}
        with self.assertRaises(ValueError):
            judge.observe(.004, states, [])
        result = judge.observe(.004, states, [], lid_angles_rad={"dry": 0., "wet": 0.})
        self.assertEqual(result["score"]["base_score"], 0)
        self.assertFalse(result["score"]["full_task_success"])


if __name__ == "__main__":
    unittest.main()
