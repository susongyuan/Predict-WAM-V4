#!/usr/bin/env python3
"""CPU-only authorization regressions: tiny temporary attestations; no weights/cache/GPU work.

Synthetic check files are test fixtures, never smoke evidence or production approval.
Run with PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' and the lilawam Python.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("trainer_gpu_authorization_under_test", HERE / "train_wam_v3.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)


class ResumeBoundaryReached(Exception):
    """Stop the tested main before checkpoint reading, even on an admitted smoke path."""


class TrainerGPUAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.assertFalse(T.torch.cuda.is_initialized())
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.cuda = self.stack.enter_context(patch.object(
            T.torch.cuda, "_lazy_init", side_effect=AssertionError("test must not initialize CUDA")))
        self.cuda_available = self.stack.enter_context(patch.object(
            T.torch.cuda, "is_available", side_effect=AssertionError("test must not query CUDA")))
        self.torch_load = self.stack.enter_context(patch.object(
            T.torch, "load", side_effect=AssertionError("test must not read checkpoints")))
        self.new_run = self.stack.enter_context(patch.object(
            T, "new_run", side_effect=AssertionError("test must not create a run")))
        self.guard = self.stack.enter_context(patch.object(
            T, "ResourceGuard", side_effect=AssertionError("test must not start resource monitoring")))
        self.resume = self.stack.enter_context(patch.object(
            T, "load_resume_source", side_effect=ResumeBoundaryReached("no checkpoint opened")))
        self.auth_path = T.resolve(T.GPU_AUTHORIZATION)
        self.auth_payload = self.auth_path.read_bytes()  # 1.9 KiB, not model/cache data.
        self.args = SimpleNamespace(continuation_plan="synthetic-plan.json", device="cuda",
                                    gpu_authorization=str(self.auth_path),
                                    gpu_authorization_sha256=T.GPU_AUTHORIZATION_SHA256)
        self.config = T.resolve("configs/lilawam_task000_multiview_manip_v3_micro16_30k_train.yaml")
        self.context = {
            "sha256": "a" * 64,
            "plan": {"parent": {"sha256": T.CONTINUATION_PARENT_SHA,
                                "training_contract_sha256": "b" * 64},
                     "schedule": dict(T.CONTINUATION_SCHEDULE)},
        }

    def tearDown(self):
        self.cuda.assert_not_called()
        self.cuda_available.assert_not_called()
        self.torch_load.assert_not_called()
        self.new_run.assert_not_called()
        self.guard.assert_not_called()
        self.assertFalse(T.torch.cuda.is_initialized())

    def test_explicit_path_and_sha_required(self):
        for path, sha in [(None, None), (str(self.auth_path), None),
                          (None, T.GPU_AUTHORIZATION_SHA256), (str(self.auth_path), "0" * 64)]:
            with self.subTest(path=path, sha=sha):
                args = SimpleNamespace(**{**vars(self.args), "gpu_authorization": path,
                                          "gpu_authorization_sha256": sha})
                with self.assertRaisesRegex(ValueError, "requires explicit"):
                    T.verify_gpu_authorization(args)

    def test_missing_pinned_receipt_rejected(self):
        with patch.object(Path, "read_bytes", side_effect=FileNotFoundError("fixture missing")):
            with self.assertRaisesRegex(ValueError, "missing or unreadable"):
                T.verify_gpu_authorization(self.args)

    def test_same_bytes_at_substitute_path_rejected(self):
        substitute = self.tmp / "authorization-copy.json"
        substitute.write_bytes(self.auth_payload)
        self.args.gpu_authorization = str(substitute)
        with self.assertRaisesRegex(ValueError, "not a substitute"):
            T.verify_gpu_authorization(self.args)

    def test_symlink_substitute_rejected(self):
        substitute = self.tmp / "authorization-link.json"
        substitute.symlink_to(self.auth_path)
        self.args.gpu_authorization = str(substitute)
        with self.assertRaisesRegex(ValueError, "not a substitute"):
            T.verify_gpu_authorization(self.args)

    def test_rewritten_receipt_with_original_hash_rejected(self):
        mutations = [b'{"verified":true}', b'{"approved":true}', b'not JSON', self.auth_payload + b'\n']
        receipt = json.loads(self.auth_payload)
        for key, value in [("user_text", "synthetic permission"),
                           ("gpu_peak_above_85_percent_allowed", False),
                           ("training_simulation_gpu_concurrency", "allowed"),
                           ("cpu_percent_ceiling_unchanged", 100)]:
            mutations.append(json.dumps({**receipt, key: value}).encode())
        for payload in mutations:
            with self.subTest(payload=payload[:60]), patch.object(Path, "read_bytes", return_value=payload):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    T.verify_gpu_authorization(self.args)

    def test_forged_verified_true_with_its_own_hash_rejected(self):
        payload = b'{"verified":true,"approved":true,"gpu_peak_above_85_percent_allowed":true}'
        self.args.gpu_authorization_sha256 = hashlib.sha256(payload).hexdigest()
        with patch.object(Path, "read_bytes", return_value=payload) as read:
            with self.assertRaisesRegex(ValueError, "requires explicit"):
                T.verify_gpu_authorization(self.args)
        read.assert_not_called()

    def test_environment_is_not_authorization(self):
        self.args.gpu_authorization = self.args.gpu_authorization_sha256 = None
        env = {"GPU_AUTHORIZATION": str(self.auth_path),
               "P2_GPU_AUTHORIZATION_SHA256": T.GPU_AUTHORIZATION_SHA256,
               "P2_GPU_UTILIZATION_ALLOWED": "100", "GPU_AUTHORIZED": "true"}
        with patch.dict(os.environ, env):
            with self.assertRaisesRegex(ValueError, "requires explicit"):
                T.verify_gpu_authorization(self.args)

    def test_exact_user_receipt_passes_resource_gate(self):
        result = T.verify_gpu_authorization(self.args)
        self.assertEqual(result["path"], str(self.auth_path))
        self.assertEqual(result["sha256"], hashlib.sha256(self.auth_payload).hexdigest())
        self.assertEqual(result["attestation"], json.loads(self.auth_payload))
        self.assertEqual(result["attestation"]["gpu_utilization_percent_allowed"], 100)
        self.assertNotIn("approved", result)

    def test_cpu_skips_file_and_cuda_access(self):
        self.args.device = "cpu"
        self.args.gpu_authorization = "/must-not-be-read/authorization.json"
        self.args.gpu_authorization_sha256 = "invalid"
        with patch.object(Path, "read_bytes", side_effect=AssertionError("CPU must skip receipt reads")):
            self.assertIsNone(T.verify_gpu_authorization(self.args))

    def test_legacy_noncontinuation_skips_new_gate(self):
        self.args.continuation_plan = None
        self.args.gpu_authorization = self.args.gpu_authorization_sha256 = None
        with patch.object(Path, "read_bytes", side_effect=AssertionError("legacy must skip receipt reads")):
            self.assertIsNone(T.verify_gpu_authorization(self.args))

    def argv(self, *, auth=True, device="cuda", scope="four_5k_segments"):
        args = ["--config", str(self.config), "--mode", "long", "--run-name",
                T.CONTINUATION_PREFIX + "authorization_unit_fixture", "--resume-source", T.CONTINUATION_PARENT,
                "--continuation-plan", "synthetic-plan.json", "--long-run-approval",
                str(self.tmp / "synthetic-approval.json"), "--max-steps",
                "30001" if scope == "gpu_smoke_only" else "35000", "--device", device]
        if auth:
            args += ["--gpu-authorization", str(self.auth_path), "--gpu-authorization-sha256",
                     T.GPU_AUTHORIZATION_SHA256]
        return args

    def mock_preflight_inputs(self, *, scope="four_5k_segments", smoke=None):
        # Import stubs stop main from importing model/simulator dependencies. Config and approval
        # parsing/check hashing remain real; every check payload below is a tiny synthetic fixture.
        stubs = {"harness": SimpleNamespace(task_profile=SimpleNamespace(TASK_ID=0)),
                 "harness.wam_v3": SimpleNamespace(WamV3Dataset=None, reward_frames=None),
                 "harness.wam_behavior": SimpleNamespace(hold_baseline=None),
                 "harness.r1pro_kinematics": SimpleNamespace(R1ProKinematics=None),
                 "scripts.train_lilawam_behavior": SimpleNamespace(motion_sampling_weights=None)}
        self.stack.enter_context(patch.dict(sys.modules, stubs))
        self.stack.enter_context(patch.dict(os.environ, {"P2_TASK_ID": "0"}))
        self.plan = self.stack.enter_context(patch.object(T, "read_continuation_plan", return_value=self.context))
        check_path = self.tmp / "synthetic-cpu-check.json"
        check_path.write_text('{"synthetic_test_fixture":true,"physical_evidence":false}\n')
        check = {"passed": True, "path": str(check_path), "sha256": T.sha256(check_path)}
        names = ("cache_audit", "cpu_checks", "tiny_overfit", "pilot_2k", "independent_review",
                 "continuation_cpu", "continuation_review")
        approval = {"approved": True, "config_sha256": T.sha256(self.config),
                    "initialization_sha256": T.V2_SHA256, "scope": scope,
                    "continuation": T.continuation_identity(self.context),
                    "additional_steps": 20000, "global_budget": 50000,
                    "checks": {name: dict(check) for name in names}}
        if smoke is not None:
            approval["checks"]["continuation_gpu_smoke"] = smoke
        (self.tmp / "synthetic-approval.json").write_text(json.dumps(approval))

    def test_main_missing_authorization_rejects_before_plan_run_or_load(self):
        with patch.object(T, "read_continuation_plan") as plan, patch.object(T, "verify_long_approval") as approval:
            with self.assertRaisesRegex(ValueError, "requires explicit"):
                T.main(self.argv(auth=False))
            plan.assert_not_called()
            approval.assert_not_called()
        self.resume.assert_not_called()

    def test_main_wrong_hash_rejects_before_plan_run_or_load(self):
        argv = self.argv()
        argv[-1] = "0" * 64
        with patch.object(T, "read_continuation_plan") as plan:
            with self.assertRaisesRegex(ValueError, "requires explicit"):
                T.main(argv)
            plan.assert_not_called()
        self.resume.assert_not_called()

    def test_main_valid_authorization_missing_smoke_blocks_production(self):
        self.mock_preflight_inputs()
        with self.assertRaisesRegex(ValueError, "continuation prerequisite not approved: continuation_gpu_smoke"):
            T.main(self.argv())
        self.plan.assert_called_once()
        self.resume.assert_not_called()

    def test_main_verified_label_does_not_replace_smoke_approval(self):
        self.mock_preflight_inputs(smoke={"verified": True})
        with self.assertRaisesRegex(ValueError, "continuation prerequisite not approved: continuation_gpu_smoke"):
            T.main(self.argv())
        self.resume.assert_not_called()

    def test_main_passed_label_without_smoke_file_rejected(self):
        self.mock_preflight_inputs(smoke={"passed": True, "path": str(self.tmp / "absent-smoke.json"),
                                          "sha256": "f" * 64})
        with self.assertRaises(FileNotFoundError):
            T.main(self.argv())
        self.resume.assert_not_called()

    def test_main_wrong_smoke_hash_rejected(self):
        check_path = self.tmp / "not-smoke-evidence.json"
        check_path.write_text('{"synthetic":true}\n')
        self.mock_preflight_inputs(smoke={"passed": True, "path": str(check_path), "sha256": "f" * 64})
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            T.main(self.argv())
        self.resume.assert_not_called()

    def test_main_explicit_smoke_only_scope_reaches_mocked_resume_boundary(self):
        self.mock_preflight_inputs(scope="gpu_smoke_only")
        with self.assertRaises(ResumeBoundaryReached):
            T.main(self.argv(scope="gpu_smoke_only"))
        self.resume.assert_called_once()
        self.assertEqual(self.resume.call_args.args[3], "cuda")

    def test_main_smoke_scope_cannot_admit_production_target(self):
        self.mock_preflight_inputs(scope="gpu_smoke_only")
        with self.assertRaisesRegex(ValueError, "at most ten additional updates"):
            T.main(self.argv())
        self.resume.assert_not_called()

    def test_main_cpu_reaches_mocked_resume_without_gpu_authorization(self):
        self.mock_preflight_inputs(scope="gpu_smoke_only")
        with patch.object(Path, "read_bytes", side_effect=AssertionError("CPU must not read authorization")):
            with self.assertRaises(ResumeBoundaryReached):
                T.main(self.argv(auth=False, device="cpu", scope="gpu_smoke_only"))
        self.resume.assert_called_once()
        self.assertEqual(self.resume.call_args.args[3], "cpu")

    def test_main_gate_and_approval_precede_loading_and_run_creation(self):
        tree = ast.parse(Path(T.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = {n.func.id: n.lineno for n in ast.walk(main)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in {"verify_gpu_authorization", "read_continuation_plan", "verify_long_approval",
                                   "load_resume_source", "new_run", "build_cached_wrapper"}}
        self.assertLess(calls["verify_gpu_authorization"], calls["read_continuation_plan"])
        self.assertLess(calls["verify_gpu_authorization"], calls["verify_long_approval"])
        self.assertLess(calls["verify_long_approval"], calls["load_resume_source"])
        self.assertLess(calls["load_resume_source"], calls["new_run"])
        self.assertLess(calls["new_run"], calls["build_cached_wrapper"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
