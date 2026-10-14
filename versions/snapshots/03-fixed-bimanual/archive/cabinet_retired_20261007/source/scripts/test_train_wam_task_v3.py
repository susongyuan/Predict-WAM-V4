#!/usr/bin/env python3
"""Focused CPU checks for new task-isolated V3 trainer; no GPU, simulator or real training.
Run --checkpoint-forward separately for one real-size synthetic CPU forward after strict local initialization.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_wam_task_v3 as T


def config():
    return OmegaConf.to_container(OmegaConf.load(ROOT / T.DEFAULT_CONFIG), resolve=True)


def for_task(task, name):
    cfg = config()
    cfg["dataset"] = {k: (v.replace("090", f"{task:03d}") if isinstance(v, str) and "090" in v else v)
                      for k, v in cfg["dataset"].items()}
    cfg["dataset"].update(task_id=task, task_name=name)
    if task == 13:
        cfg["dataset"]["lowdim_cache"] = "data/lowdim_cache"
    cfg["model"]["v3"]["stats_path"] = f"data/behavior-1k/2026-challenge-demos/wam_v3_stats_task{task:03d}.json"
    cfg["manifest"]["task_id"] = task
    return cfg


class TaskSchemaTests(unittest.TestCase):
    def test_90_config_and_legacy_task_paths(self):
        for task, name in ((0, "turning_on_radio"), (1, "picking_up_trash"), (13, "loading_the_car"), (90, "composting_waste")):
            self.assertEqual(T.validate_config(for_task(task, name), T.parse_args(["--check-only"])), task)

    def test_arbitrary_metadata_registered_id_not_hardcoded_to_90(self):
        rows = [json.loads(x) for x in (ROOT / "data/behavior-1k/2026-challenge-demos/meta/tasks.jsonl").read_text().splitlines()]
        row = next(x for x in rows if x["task_index"] == 2)
        self.assertEqual(T.validate_config(for_task(2, row["task_name"]), T.parse_args(["--check-only"])), 2)

    def test_cross_task_sources_fail(self):
        for key, path in (("stats_path", "data/behavior-1k/2026-challenge-demos/train_stats_task000.json"),
                          ("feature_cache", "data/dino_cache_task000"), ("split_path", "data/behavior-1k/2026-challenge-demos/split_task_001.json"),
                          ("contract_targets", "data/skill_vtt_task013/contract_targets.json")):
            cfg = config(); cfg["dataset"][key] = path
            with self.assertRaisesRegex(ValueError, "task-isolated"):
                T.validate_config(cfg, T.parse_args(["--check-only"]))

    def test_wrong_task_name_fails(self):
        cfg = config(); cfg["dataset"]["task_name"] = "picking_up_trash"
        with self.assertRaisesRegex(ValueError, "official metadata"):
            T.validate_config(cfg, T.parse_args(["--check-only"]))

    def test_environment_conflict_before_import(self):
        with patch.dict(os.environ, {"P2_TASK_ID": "0"}):
            with self.assertRaisesRegex(ValueError, "P2_TASK_ID differs"):
                T.select_task(90)

    def test_bool_and_unregistered_id_fail(self):
        for task in (True, -1, 1000):
            cfg = config(); cfg["dataset"]["task_id"] = task
            with self.assertRaises(ValueError):
                T.validate_config(cfg, T.parse_args(["--check-only"]))

    def test_budget_is_new_50k_not_old_continuation(self):
        cfg = config()
        cfg["training"]["max_steps"] = 30000
        with self.assertRaises(ValueError):
            T.validate_config(cfg, T.parse_args(["--check-only"]))
        with self.assertRaises(ValueError):
            T.validate_config(config(), T.parse_args(["--check-only", "--max-steps", "5001"]))
        self.assertEqual(T.validate_config(config(), T.parse_args(["--run-name", "task090_v3_50k_s005000"])), 90)

    def test_no_legacy_approval_or_gpu_resource_gate_invoked(self):
        with patch.object(T.core, "verify_long_approval", side_effect=AssertionError("legacy approval")), \
             patch.object(T.core, "verify_gpu_authorization", side_effect=AssertionError("legacy resource chain")), \
             patch.object(torch.cuda, "init", side_effect=AssertionError("GPU used")):
            T.validate_config(config(), T.parse_args(["--check-only"]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_missing_data_not_ready(self):
        import contextlib
        import io
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"P2_TASK_ID": "90"}), \
             patch.object(T, "required_paths", return_value=[Path(tmp) / "task090_missing.json"]), contextlib.redirect_stdout(stream):
            result = T.main(["--check-only"])
        self.assertEqual(result, 2)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("task090" in p for p in report["missing"]))


class PayloadChecks(unittest.TestCase):
    def test_header_edges_finite_and_stat_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "array.npy"
            np.save(path, np.zeros((3, 2), np.float32))
            ledger = T.InputLedger()
            T.checked_array(path, (3, 2), np.float32, ledger)
            self.assertFalse(ledger.files)  # never claimed a payload hash
            ledger.unchanged()
            np.save(path, np.ones((3, 2), np.float32))
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
            with self.assertRaises(RuntimeError):
                ledger.unchanged()
            np.save(path, np.full((3, 2), np.nan, np.float32))
            with self.assertRaises(FloatingPointError):
                T.checked_array(path, (3, 2), np.float32, T.InputLedger())

    def test_truncated_and_wrong_shapes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "array.npy"
            np.save(path, np.zeros((3, 2), np.float32))
            with self.assertRaises(ValueError):
                T.checked_array(path, (2, 3), np.float32, T.InputLedger())
            with path.open("ab") as f:
                f.write(b"extra")
            with self.assertRaises(ValueError):
                T.checked_array(path, (3, 2), np.float32, T.InputLedger())

    def test_bfloat16_bits_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "array.npy"
            a = torch.ones((3, 2), dtype=torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            np.save(path, a)
            T.checked_array(path, (3, 2), np.uint16, T.InputLedger(), bits=True)


class RestoreChecks(unittest.TestCase):
    def test_complete_adaptation_restore_and_bad_ancestry_rejection(self):
        from scripts.test_train_wam_v3 import FakeWrapper
        cfg = config()
        torch.manual_seed(123)
        wrapper = FakeWrapper()
        opt, sch = T.core.make_optimizer(wrapper.action_model, cfg["training"])
        states = {}
        def save(tag, step, extra):
            if tag == "last":
                states.update(step=step, extra=copy.deepcopy(extra), model=copy.deepcopy(wrapper.action_model.state_dict()))
        T.core.train_loop(wrapper, opt, sch, [{"train_target": torch.tensor(1.)}] * 4,
                          max_steps=2, accum=2, grad_clip=1, lambdas={}, validate_fn=lambda s: {T.core.METRIC: 1.0},
                          save_fn=save, emit=lambda x: None, every=2)
        with tempfile.TemporaryDirectory() as tmp:
            meta = {"mode": T.MODE, "config": cfg, "initialization_sha256": cfg["training"]["init_from_sha256"], **states["extra"]}
            path = T.core.save_checkpoint(tmp, "last", wrapper.action_model, 2, meta)
            ck, digest = T.load_resume(path, cfg, "cpu")
            restored = FakeWrapper()
            restored.action_model.load_state_dict(ck["model_state_dict"], strict=True)
            ropt, rsch = T.core.make_optimizer(restored.action_model, cfg["training"])
            T.core.restore_training_state(restored, ropt, rsch, ck["resume_state"])
            self.assertEqual(rsch.last_epoch, 2)
            self.assertEqual(ropt.param_groups[0]["lr"], opt.param_groups[0]["lr"])
            self.assertEqual(ck["resume_state"]["cursor"]["batch_skip"], 4)
            for a, b in zip(wrapper.action_model.parameters(), restored.action_model.parameters()):
                self.assertTrue(torch.equal(a, b))
            result = T.core.train_loop(restored, ropt, rsch, [{"train_target": torch.tensor(1.)}] * 4,
                        max_steps=4, accum=2, grad_clip=1, lambdas={}, validate_fn=lambda s: {T.core.METRIC: 1.0},
                        save_fn=lambda *a: None, emit=lambda x: None, every=2, resume_state=ck["resume_state"])
            self.assertEqual(result["completed_updates"], 4)
            changed = copy.deepcopy(cfg); changed["dataset"]["task_id"] = 1
            with self.assertRaises(ValueError):
                T.load_resume(path, changed, "cpu")
            with self.assertRaises(ValueError):
                T.load_resume(path, cfg, "cuda")
            path = T.core.save_checkpoint(tmp, "final", wrapper.action_model, 2, {"mode": T.MODE, "config": cfg})
            with self.assertRaises(ValueError):
                T.load_resume(path, cfg, "cpu")


def checkpoint_forward():
    cfg = config()
    T.select_task(90)
    torch.set_num_threads(2)
    model, enc = T.build_action_model(cfg)
    notes = T.initialize_model(cfg, model)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([.5]), noisy_actions=torch.zeros(1, 32, 23), qpos_history=torch.zeros(1, 7, 61),
                    dino_features_list=[torch.zeros(1, 3, 201, 1024) for _ in range(3)], task_cond=torch.zeros(1, 1024),
                    memory=torch.zeros(1, 3, 3, 2048))
        future = model.future_feat_decoder(out["cond_tokens"])
        for key, value in out.items():
            T.core.finite(value, key)
        T.core.finite(future, "future")
    assert out["final_pred"].shape == (1, 32, 23)
    assert future.shape == (1, 3 * 196, 1024)
    print(json.dumps({"status": "passed", "cpu_only": True, "synthetic_inputs": True, "actual_local_checkpoint": notes,
                      "action_shape": list(out["final_pred"].shape), "future_shape": list(future.shape),
                      "cuda_initialized": torch.cuda.is_initialized(), "task90_capability_not_evaluated": True}))


if __name__ == "__main__":
    if "--checkpoint-forward" in sys.argv:
        checkpoint_forward()
    else:
        torch.set_num_threads(2)
        unittest.main()
