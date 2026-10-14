#!/usr/bin/env python3
"""Tiny synthetic CPU fixtures. These are NOT observations from a robot."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
from check_robot_io_contract import compare_model, load_json, validate

OUT = Path(__file__).resolve().parents[1] / "outputs/validation/task090_completion_20261003_0156/robot_io"
SCRIPT = Path(__file__).with_name("check_robot_io_contract.py")


def synthetic_contract():
    stamp = {"value": 100.0, "unit": "s", "clock_domain": "synthetic_boot_1"}
    arm = {"name": "arm.q", "quantity": "angular_position", "unit": "rad", "joint": "arm", "frame": "joint"}
    finger = {"name": "finger.q", "quantity": "linear_position", "unit": "m", "joint": "finger", "frame": "joint"}
    command = dict(arm, limits=[-1.0, 1.0])
    gripper = {"name": "grip", "quantity": "gripper_aperture", "unit": "normalized", "joint": None, "frame": "joint",
               "limits": [-1.0, 1.0], "affine_to_si": {"unit": "m", "scale": 0.025, "offset": 0.025}}
    camera = {"name": "test_camera", "device": "camera1",
              "rgb": {"key": "cam/rgb", "shape": [4, 4, 3], "layout": "HWC", "encoding": "rgb8", "stamp": copy.deepcopy(stamp)},
              "intrinsics": {"model": "pinhole_rectified", "unit": "pixel", "resolution_hw": [4, 4],
                             "K": [[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]], "calibration_id": "synthetic_intrinsics"},
              "extrinsics": {"from_frame": "cam_optical", "to_frame": "base", "camera_axes": "opencv_x_right_y_down_z_forward",
                             "matrix": [[1., 0., 0., .1], [0., 1., 0., .2], [0., 0., 1., .3], [0., 0., 0., 1.]],
                             "translation_unit": "m", "kind": "dynamic", "stamp": copy.deepcopy(stamp), "calibration_id": "synthetic_pose"},
              "depth": {"enabled": True, "key": "cam/depth_m", "shape": [4, 4], "unit": "m", "quantity": "z_depth", "aligned_to_rgb": True,
                        "valid_range_m": [.1, 4.0], "stamp": copy.deepcopy(stamp),
                        "valid_mask": {"key": "cam/depth_valid", "shape": [4, 4], "dtype": "bool",
                                       "valid_when": "finite_and_in_range", "invalid_value": "null"},
                        "probe": {"values_m": [.5, None, 0.], "valid_mask": [True, False, False]}}}
    return {"schema_version": 1, "robot": {"id": "SYNTHETIC_NOT_HARDWARE", "evidence": "declared"},
            "frames": {"base": "base", "base_axes": "x_forward_y_left_z_up", "handedness": "right", "quaternion_order": "xyzw",
                       "transform_convention": "p_base=T_base_camera@p_camera"},
            "timing": {"clock_domain": "synthetic_boot_1", "unit": "s", "action_hz": 20.0, "observation_hz": 20.0,
                       "max_age_s": .2, "max_command_age_s": .2, "max_skew_s": .03},
            "devices": [{"id": "camera1", "kind": "camera", "available": True, "evidence": "declared"},
                        {"id": "controller1", "kind": "controller", "available": True, "evidence": "declared"},
                        {"id": "estop1", "kind": "estop", "available": True, "evidence": "declared"}],
            "joints": [{"name": "arm", "kind": "revolute", "unit": "rad", "limits": [-1., 1.], "max_velocity": .5, "velocity_unit": "rad/s"},
                       {"name": "finger", "kind": "prismatic", "unit": "m", "limits": [0., .05], "max_velocity": .02, "velocity_unit": "m/s"}],
            "proprio": {"dim": 2, "layout": [arm, finger], "stamp": copy.deepcopy(stamp)},
            "action": {"dim": 2, "layout": [command, gripper], "stamp": copy.deepcopy(stamp), "device": "controller1", "queue_flushed_on_stop": True, "position_command_mode": "absolute"},
            "camera_allowlist": ["cam/rgb", "cam/depth_m", "cam/depth_valid"], "cameras": [camera],
            "grippers": [{"action_name": "grip", "open_command": 1., "closed_command": -1., "larger_command": "open",
                          "feedback_joints": ["finger"], "feedback_aggregation": "mean", "hold_semantics": "retain measured mean finger position; no force guarantee"}],
            "safety": {"watchdog": {"enabled": True, "independent_of_policy": True, "location": "device", "timeout_s": .1, "on_expiry": "stop"},
                       "hold": {"supported": True, "interface": "declaration_only/hold", "semantics": "copy fresh joint position; stop base",
                                "on_stale_proprio": "stop"},
                       "stop": {"supported": True, "interface": "declaration_only/stop", "semantics": "device controlled stop with queue flush"},
                       "estop": {"supported": True, "interface": "declaration_only/physical_estop", "semantics": "latched device emergency stop",
                                 "physical": True, "latched": True, "manual_reset": True, "device": "estop1"}},
            "model_input": {"image_size_hw": [224, 224], "proprio_lags": [0], "action_chunk_size": 1}}


def synthetic_profile(contract):
    return {"profile_version": 1, "name": "synthetic_only", "expects": {
        "frames": copy.deepcopy(contract["frames"]),
        "action": {"dim": contract["action"]["dim"], "layout": copy.deepcopy(contract["action"]["layout"])},
        "proprio": {"dim": contract["proprio"]["dim"], "layout": copy.deepcopy(contract["proprio"]["layout"])},
        "cameras": {"names": [c["name"] for c in contract["cameras"]]},
        "timing": {"action_hz": contract["timing"]["action_hz"], "observation_hz": contract["timing"]["observation_hz"]},
        "model_input": copy.deepcopy(contract["model_input"])}}


class RobotIOContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = synthetic_contract()
        self.profile = synthetic_profile(self.contract)

    def check(self, path=None):
        result = validate(self.contract, now=100.05, clock_domain="synthetic_boot_1", model_profile=self.profile)
        self.assertEqual(result["hardware_execution"]["status"], "NOT_TESTED")
        self.assertIs(result["real_robot_admission"], False)
        if path:
            self.assertEqual(result["readiness_status"], "BLOCKED")
            self.assertTrue(any(path in i["path"] for i in result["issues"]), result["issues"])
        return result

    def test_complete_declaration_is_not_hardware_admission(self):
        report = self.check()
        self.assertEqual(report["readiness_status"], "PASS", report["issues"])
        self.assertEqual(report["schema_status"], "PASS")
        self.assertEqual(report["model_compatibility"]["status"], "COMPATIBLE_DECLARED")
        self.assertEqual(report["evidence"], "declared_only")

    def test_depth_mm_rejected(self):
        self.contract["cameras"][0]["depth"]["unit"] = "mm"
        self.check("depth.unit")

    def test_action_degrees_rejected(self):
        self.contract["action"]["layout"][0]["unit"] = "deg"
        self.check("action.layout[0].unit")

    def test_joint_degree_limits_rejected(self):
        self.contract["joints"][0]["unit"] = "deg"
        self.check("joints[0].unit")

    def test_nonorthogonal_rotation_rejected(self):
        self.contract["cameras"][0]["extrinsics"]["matrix"][0][0] = 2
        self.check("extrinsics.matrix")

    def test_reflection_rejected(self):
        self.contract["cameras"][0]["extrinsics"]["matrix"][0][0] = -1
        self.check("extrinsics.matrix")

    def test_bad_homogeneous_row_rejected(self):
        self.contract["cameras"][0]["extrinsics"]["matrix"][3][2] = .1
        self.check("extrinsics.matrix")

    def test_camera_direction_rejected(self):
        self.contract["cameras"][0]["extrinsics"]["to_frame"] = "world"
        self.check("extrinsics")

    def test_bad_camera_focal_rejected(self):
        self.contract["cameras"][0]["intrinsics"]["K"][0][0] = -1
        self.check("intrinsics.K")

    def test_wrong_intrinsics_resolution_rejected(self):
        self.contract["cameras"][0]["intrinsics"]["resolution_hw"] = [480, 480]
        self.check("intrinsics.resolution_hw")

    def test_allowlist_extra_ground_truth_rejected(self):
        self.contract["camera_allowlist"].append("object/ground_truth")
        self.check("camera_allowlist")

    def test_duplicate_camera_rejected(self):
        self.contract["cameras"].append(copy.deepcopy(self.contract["cameras"][0]))
        self.check("cameras")

    def test_mask_missing_rejected(self):
        del self.contract["cameras"][0]["depth"]["valid_mask"]
        self.check("valid_mask")

    def test_zero_depth_marked_valid_rejected(self):
        self.contract["cameras"][0]["depth"]["probe"]["valid_mask"][2] = True
        self.check("depth.probe")

    def test_nonfinite_depth_probe_rejected(self):
        self.contract["cameras"][0]["depth"]["probe"]["values_m"][0] = float("nan")
        self.check("depth.probe")

    def test_stale_proprio_rejected(self):
        self.contract["proprio"]["stamp"]["value"] = 99.0
        self.check("proprio.stamp")

    def test_future_command_rejected(self):
        self.contract["action"]["stamp"]["value"] = 101.0
        self.check("action.stamp")

    def test_stale_command_rejected(self):
        self.contract["action"]["stamp"]["value"] = 99.0
        self.check("action.stamp")

    def test_time_domain_mismatch_rejected(self):
        self.contract["proprio"]["stamp"]["clock_domain"] = "different_boot"
        self.check("clock_domain")

    def test_millisecond_timestamps_rejected(self):
        self.contract["proprio"]["stamp"]["unit"] = "ms"
        self.check("proprio.stamp.unit")

    def test_stale_dynamic_extrinsic_rejected(self):
        self.contract["cameras"][0]["extrinsics"]["stamp"]["value"] = 99.0
        self.check("extrinsics.stamp")

    def test_unsynchronized_sensors_rejected(self):
        self.contract["cameras"][0]["rgb"]["stamp"]["value"] = 99.9
        self.check("max_skew_s")

    def test_missing_watchdog_rejected(self):
        del self.contract["safety"]["watchdog"]
        self.check("watchdog")

    def test_policy_watchdog_not_device_watchdog(self):
        self.contract["safety"]["watchdog"]["location"] = "policy"
        self.check("watchdog.location")

    def test_watchdog_over_age_budget_rejected(self):
        self.contract["safety"]["watchdog"]["timeout_s"] = 10.0
        self.check("watchdog.timeout_s")

    def test_stop_queue_not_flushed_rejected(self):
        self.contract["action"]["queue_flushed_on_stop"] = False
        self.check("queue_flushed_on_stop")

    def test_no_physical_estop_rejected(self):
        self.contract["safety"]["estop"]["physical"] = False
        self.check("estop.physical")

    def test_declared_device_unavailable_rejected(self):
        self.contract["devices"][0]["available"] = False
        self.check("available")

    def test_measured_claim_not_accepted_as_proof(self):
        self.contract["devices"][0]["evidence"] = "measured"
        self.check("evidence")

    def test_gripper_sign_rejected(self):
        self.contract["grippers"][0]["larger_command"] = "close"
        self.check("larger_command")

    def test_gripper_negative_mapping_rejected(self):
        self.contract["action"]["layout"][1]["affine_to_si"]["scale"] *= -1
        self.check("grippers")

    def test_joint_command_limit_rejected(self):
        self.contract["action"]["layout"][0]["limits"] = [-5, 5]
        self.check("action.layout[0].limits")

    def test_joint_layout_incompatible_separately(self):
        self.contract["proprio"]["layout"].reverse()
        report = self.check()
        self.assertEqual(report["readiness_status"], "PASS")
        self.assertEqual(report["model_compatibility"]["status"], "INCOMPATIBLE_OR_UNKNOWN")
        self.assertTrue(any("proprio.layout" in x for x in report["model_compatibility"]["issues"]))

    def test_same_dimensions_unknown_joint_names_not_assumed_compatible(self):
        self.contract["proprio"]["layout"][0]["name"] = "different_robot.q"
        report = self.check()
        self.assertEqual(report["readiness_status"], "PASS")
        self.assertEqual(report["model_compatibility"]["status"], "INCOMPATIBLE_OR_UNKNOWN")

    def test_delta_commands_need_explicit_adapter(self):
        self.contract["action"]["position_command_mode"] = "delta"
        self.check("position_command_mode")

    def test_gripper_metric_mapping_exceeds_finger_limits(self):
        self.contract["action"]["layout"][1]["affine_to_si"]["scale"] = .05
        self.contract["action"]["layout"][1]["affine_to_si"]["offset"] = .05
        self.check("feedback_joints")

    def test_missing_required_depth_blocks(self):
        self.contract["cameras"][0]["depth"] = {"enabled": False}
        self.contract["camera_allowlist"] = ["cam/rgb"]
        self.check("cameras.depth")

    def test_additional_rgb_only_camera_supported(self):
        camera = copy.deepcopy(self.contract["cameras"][0])
        camera["name"] = "rgb_only"
        camera["rgb"]["key"] = "cam2/rgb"
        camera["depth"] = {"enabled": False}
        self.contract["cameras"].append(camera)
        self.contract["camera_allowlist"].append("cam2/rgb")
        self.assertEqual(self.check()["readiness_status"], "PASS")

    def test_malformed_joint_kind_is_reported_not_crashed(self):
        self.contract["joints"][0]["kind"] = []
        self.check("joints[0].kind")

    def test_extreme_number_is_rejected(self):
        self.contract["action"]["layout"][0]["limits"] = [-10**400, 10**400]
        self.check("limits")

    def test_missing_dimension_rejected(self):
        self.contract["action"]["dim"] = 23
        self.check("action.dim")

    def test_unknown_required_field_blocks(self):
        self.contract["cameras"][0]["extrinsics"]["matrix"] = None
        self.check("extrinsics.matrix")

    def test_null_sections_fail_closed(self):
        for section in self.contract:
            with self.subTest(section=section):
                contract = synthetic_contract()
                contract[section] = None
                result = validate(contract, now=100.05, clock_domain="synthetic_boot_1")
                if section != "model_input":  # optional unless a model profile is requested
                    self.assertEqual(result["readiness_status"], "BLOCKED")

    def test_empty_contract(self):
        self.contract = {}
        self.check("schema_version")

    def test_no_profile_is_not_checked(self):
        result = compare_model(self.contract, None)
        self.assertEqual(result["status"], "NOT_CHECKED")

    def test_cli_exit_codes_and_json(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cli_test_", dir=OUT) as directory:
            contract = Path(directory) / "synthetic.json"
            contract.write_text(json.dumps(self.contract))
            args = [sys.executable, "-B", str(SCRIPT), str(contract), "--clock-domain", "synthetic_boot_1", "--now", "100.05"]
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["real_robot_admission"])
            self.contract["safety"]["watchdog"] = None
            contract.write_text(json.dumps(self.contract))
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(result.stdout)["readiness_status"], "BLOCKED")
            contract.write_text('{"schema_version": 1, "schema_version": 2}')
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 3)
            self.assertIn("duplicate", json.loads(result.stdout)["input_error"])

    def test_non_unix_clock_needs_now(self):
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "/nonexistent", "--clock-domain", "robot_boot_1"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 3)
        self.assertIn("requires explicit --now", json.loads(result.stdout)["input_error"])

    def test_nonstandard_json_nan_rejected(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="json_test_", dir=OUT) as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"value": NaN}')
            with self.assertRaises(ValueError):
                load_json(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
