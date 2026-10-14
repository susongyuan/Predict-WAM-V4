#!/usr/bin/env python3
"""CPU synthetic contract tests; not evidence of real-data/model/GPU smoke."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_robotwin_wam as trainer


class TinyWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_model = torch.nn.Linear(14, 14)
        self.device = torch.device("cpu")

    def normalize_action(self, x):
        return x * 2


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = OmegaConf.to_container(OmegaConf.load(ROOT / trainer.DEFAULT_CONFIG), resolve=True)

    def test_recipe_is_native_fixed_and_rejects_legacy_or_auxiliary_labels(self):
        trainer.validate_recipe(self.cfg)
        for path, bad in (("common.action_dim", 23), ("common.state_dim", 61),
                          ("training.auxiliary_loss.phase_ce", 0.1), ("training.max_steps", 2),
                          ("model.action_expert.depth", 1), ("dataset.cond_mode", "goal"),
                          ("model.v3.proprio_lags", [1, 2, 3, 4, 5])):
            c = copy.deepcopy(self.cfg)
            d = c
            pieces = path.split(".")
            for k in pieces[:-1]:
                d = d[k]
            d[pieces[-1]] = bad
            with self.subTest(path=path), self.assertRaises(ValueError):
                trainer.validate_recipe(c)

    def test_resume_cadence_preserves_fixed_milestones_and_target(self):
        for interval in (1000, 1500):
            self.assertTrue(trainer.checkpoint_due(5000, 30000, interval))
            self.assertTrue(trainer.checkpoint_due(interval, 30000, interval))
            self.assertFalse(trainer.checkpoint_due(999, 30000, interval))
            self.assertTrue(trainer.checkpoint_due(1234, 1234, interval))
        for interval in (0, -1, True, 5001):
            cfg = copy.deepcopy(self.cfg)
            cfg["training"]["resume_save_every"] = interval
            with self.assertRaises(ValueError):
                trainer.validate_recipe(cfg)

    def test_defaults_plan_and_explicit_train(self):
        self.assertEqual(trainer.parse_args([]).mode, "plan")
        self.assertEqual(trainer.parse_args([]).device, "cpu")
        args = trainer.parse_args(["--mode", "smoke", "--out", "/tmp/not-created"])
        self.assertEqual((args.smoke_steps, args.smoke_batch_size), (2, 1))
        for args in (["--mode", "train"], ["--max-steps", "2"],
                     ["--mode", "train", "--out", "/tmp/x", "--max-steps", "50001"],
                     ["--mode", "smoke", "--out", "/tmp/x", "--device", "cuda"],
                     ["--cuda-memory-fraction", "0"], ["--resume"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                trainer.parse_args(args)

    def test_fresh_optimizer_full_schedule_even_for_smoke(self):
        wrapper = TinyWrapper()
        optimizer, scheduler = trainer.make_optimizer(wrapper, self.cfg["training"])
        self.assertEqual(optimizer.state_dict()["state"], {})
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-7)
        for _ in range(2):
            optimizer.zero_grad()
            wrapper.action_model(torch.ones(1, 14)).square().mean().backward()
            optimizer.step(); scheduler.step()
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-5 * (0.01 + 0.99 * 2 / 2000))

    def test_metrics_native_arms_and_grippers_and_validity(self):
        wrapper = TinyWrapper()
        target = torch.zeros(1, 2, 14)
        pred = torch.zeros_like(target)
        pred[0, 0, :6] = 1
        pred[0, 0, 6] = 2
        pred[0, 0, 7:13] = 3
        pred[0, 0, 13] = 4
        pred[0, 1] = float("nan")
        m = trainer.action_metrics(pred, target, torch.tensor([[True, False]]), wrapper)
        self.assertEqual(m["left_arm_rad_mae"], 1)
        self.assertEqual(m["right_arm_rad_mae"], 3)
        self.assertEqual(m["left_gripper_units_mae"], 2)
        self.assertEqual(m["right_gripper_units_mae"], 4)
        self.assertAlmostEqual(m["normalized_mse"], (6 + 4 + 54 + 16) / 14 * 4, places=5)
        self.assertEqual(m["valid_action_rows"], 1)
        with self.assertRaises(ValueError):
            trainer.action_metrics(pred, target, torch.zeros(1, 2, dtype=torch.bool), wrapper)
        with self.assertRaises(ValueError):
            trainer.action_metrics(pred, target, torch.ones(1, 2, dtype=torch.bool), wrapper)

    def test_fixed_validation_rng_does_not_touch_training_numpy(self):
        np.random.seed(19)
        before = np.random.get_state()
        one = trainer.validation_indices(1000, 20261004, 512)
        two = trainer.validation_indices(1000, 20261004, 512)
        self.assertEqual(one, two)
        self.assertEqual(len(set(one)), 512)
        np.testing.assert_array_equal(before[1], np.random.get_state()[1])
        self.assertEqual(trainer.validation_indices(2, 5, 512).__len__(), 2)

    def test_evaluation_rejects_heldout_before_sampling(self):
        class Heldout:
            split = "heldout"
        with self.assertRaisesRegex(ValueError, "only use val"):
            trainer.evaluate(TinyWrapper(), Heldout(), [0], 19)

    def test_checkpoint_roundtrip_and_resume_are_strict(self):
        with tempfile.TemporaryDirectory() as d:
            out, wrapper = Path(d), TinyWrapper()
            optimizer, scheduler = trainer.make_optimizer(wrapper, self.cfg["training"])
            contract = {"mode": "synthetic_test", "config": self.cfg}
            cursor = {"epoch": 1, "batch_skip": 0}
            path = trainer.save_checkpoint(out, "last", wrapper, 0, contract, optimizer, scheduler, cursor)
            ck = trainer.load_checkpoint(path, contract)
            self.assertFalse(ck["model_only"])
            self.assertEqual(ck["resume_state"]["cursor"], cursor)
            self.assertEqual(ck["checkpoint_metric"], "diagnostic_only_no_selection")
            for k, v in wrapper.action_model.state_dict().items():
                self.assertTrue(torch.equal(ck["model_state_dict"][k], v))
            with self.assertRaisesRegex(ValueError, "immutable"):
                trainer.load_checkpoint(path, {**contract, "changed": True})
            model = trainer.save_checkpoint(out, "step_000000", wrapper, 0, contract)
            with self.assertRaisesRegex(ValueError, "checkpoint_last"):
                trainer.load_checkpoint(model, contract)
            self.assertNotIn("resume_state", trainer.load_checkpoint(model, contract, resume=False))
            with self.assertRaises(ValueError):
                trainer.save_checkpoint(out, "last", wrapper, 0, contract)
            # Different bytes must be rejected BEFORE unpickling.
            del ck
            with path.open("ab") as f:
                f.write(b"bad")
            with patch.object(torch, "load", side_effect=AssertionError("must not unpickle")):
                with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                    trainer.load_checkpoint(path, contract)

    def test_exact_stochastic_optimizer_resume(self):
        def make_loader(epoch):
            return DataLoader(torch.arange(63, dtype=torch.float32).reshape(-1, 1).repeat(1, 14),
                batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(77 + epoch))
        def update(wrapper, optimizer, scheduler, stream):
            optimizer.zero_grad()
            for _ in range(2):
                batch = stream.next()
                noise = torch.randn_like(batch) + random.random() + np.random.random()
                ((wrapper.action_model(batch + noise) ** 2).mean() / 2).backward()
            optimizer.step(); scheduler.step()
        random.seed(17); np.random.seed(17); torch.manual_seed(17)
        w = TinyWrapper()
        opt, sched = trainer.make_optimizer(w, self.cfg["training"])
        stream = trainer.BatchStream(make_loader, "cpu")
        update(w, opt, sched, stream)
        with tempfile.TemporaryDirectory() as d:
            path = trainer.save_checkpoint(Path(d), "last", w, 1, {}, opt, sched, stream.state())
            update(w, opt, sched, stream)
            expected = copy.deepcopy(w.state_dict())
            expected_rng = trainer.capture_rng("cpu")
            ck = trainer.load_checkpoint(path, {})
            restored = TinyWrapper()
            restored.action_model.load_state_dict(ck["model_state_dict"], strict=True)
            ropt, rsched = trainer.make_optimizer(restored, self.cfg["training"])
            ropt.load_state_dict(ck["resume_state"]["optimizer"])
            rsched.load_state_dict(ck["resume_state"]["scheduler"])
            rstream = trainer.BatchStream(make_loader, "cpu", cursor=ck["resume_state"]["cursor"])
            trainer.restore_rng(ck["resume_state"]["rng"], "cpu")
            update(restored, ropt, rsched, rstream)
            for k, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[k]), k)
            self.assertEqual(rsched.last_epoch, 2)
            self.assertTrue(torch.equal(expected_rng["torch_cpu"], torch.get_rng_state()))
            self.assertEqual(expected_rng["python"], random.getstate())
            np.testing.assert_array_equal(expected_rng["numpy"][1], np.random.get_state()[1])

    def test_cache_identity_detects_changed_payload(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "payload"
            p.write_bytes(b"one")
            entry = trainer.file_identity(p)
            contract = {"cache_config": entry, "cache_files": [entry]}
            trainer.check_cache_unchanged(contract)
            p.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "immutable cache"):
                trainer.check_cache_unchanged(contract)

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d, patch.object(trainer, "preflight") as preflight:
            with self.assertRaises(FileExistsError):
                trainer.main(["--out", d])
            preflight.assert_not_called()
            with self.assertRaisesRegex(ValueError, "checkpoint_last"):
                trainer.main(["--mode", "train", "--out", d, "--resume"])
            preflight.assert_not_called()

    def test_plan_missing_cache_is_blocked_without_model_or_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "plan"
            with patch.object(trainer, "preflight", side_effect=ValueError("missing cache config")), \
                 patch.object(trainer, "build_wrapper", side_effect=AssertionError("must not build")):
                result = trainer.main(["--mode", "plan", "--out", str(out)])
            self.assertEqual(result, 2)
            plan = json.loads((out / "plan.json").read_text())
            self.assertEqual(plan["status"], "blocked_preflight")
            self.assertFalse(plan["random_feature_fallback"])
            self.assertFalse(plan["encoder_loaded"])

    def test_synthetic_two_step_loop_saves_reloads_samples_once(self):
        class Dataset:
            def __init__(self, split):
                self.split = split
                self.episodes = ["synthetic-only"]
                self.ends = [4]
            def __len__(self):
                return 4
            def __getitem__(self, i):
                return {"episode": "synthetic-only", "anchor": i, "state": np.ones(14, np.float32)}
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.device = torch.device("cpu")
                self.action_model = torch.nn.Module()
                self.action_model.action_proj = torch.nn.Linear(14, 14)
            def forward(self, batch, lambdas):
                loss = self.action_model.action_proj(batch["state"] + torch.randn_like(batch["state"])).square().mean()
                return loss, {"mse": float(loss.detach())}, {}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            args = trainer.parse_args(["--mode", "smoke", "--out", d])
            contract = {"execution": {"batch_size": 1, "accum": 1}, "synthetic_test_only": True}
            w = Wrapper()
            with patch.object(trainer, "stats_contract", return_value=({}, {})), \
                 patch.object(trainer, "immutable_contract", return_value=contract), \
                 patch.object(trainer, "build_wrapper", return_value=(w, {})), \
                 patch.object(trainer, "check_cache_unchanged"), \
                 patch.object(trainer.Guard, "check"), \
                 patch.object(trainer.signal, "signal"), \
                 patch.object(trainer, "evaluate", return_value={"synthetic_test_only": True}) as evaluate:
                result = trainer.execute(self.cfg, args, out, {k: Dataset(k) for k in ("train", "val")}, {})
            self.assertEqual(result["step"], 2)
            evaluate.assert_called_once()
            self.assertEqual(len(evaluate.call_args.args[2]), 1)
            self.assertTrue((out / "validation_000002.json").is_file())
            self.assertTrue((out / "peak_resources.json").is_file())
            last = trainer.load_checkpoint(out / "checkpoint_last.pt", contract)
            self.assertEqual(last["step"], 2)
            self.assertEqual(last["resume_state"]["scheduler"]["last_epoch"], 2)
            events = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(sum(x["event"] == "optimizer_step" for x in events), 2)
            self.assertTrue(any(x["event"] == "real_cached_smoke_save_reload" for x in events))

    def test_validation_anchor_plan_does_not_load_features(self):
        class Dataset:
            ends = [3, 7]
            episodes = ["first", "second"]
            def __getitem__(self, i):
                raise AssertionError("planning must not map features")
        self.assertEqual(trainer.validation_anchors(Dataset(), [0, 2, 3, 6]),
            [{"episode": "first", "anchor": 0}, {"episode": "first", "anchor": 2},
             {"episode": "second", "anchor": 0}, {"episode": "second", "anchor": 3}])

    def test_cpu_guard_enforces_85_and_records_peaks(self):
        def base_check(self, force=False):
            self.last += 1
            self.latest = {"ram_percent": 10, "rss_kib": 20}
        with patch.object(trainer.ResourceGuard, "check", base_check), \
             patch.object(trainer.Guard, "cpu_ticks", side_effect=[(100, 50), (200, 60)]):
            guard = trainer.Guard(self.cfg["system"], "cpu", lambda _: None)
            with self.assertRaisesRegex(trainer.StopRequested, "CPU limit"):
                guard.check()
            self.assertEqual(guard.peaks["cpu_percent"], 90)

    def test_cuda_guard_never_acquires_external_lock(self):
        def base_check(self, force=False):
            self.last += 1
            self.latest = {"ram_percent": 10}
        with patch.object(trainer.ResourceGuard, "check", base_check), \
             patch.object(trainer.Guard, "cpu_ticks", side_effect=[(100, 50), (200, 120)]), \
             patch("scripts.gpu_lock.owner", return_value="somebody-else"), \
             patch("scripts.gpu_lock.acquire", side_effect=AssertionError("must not acquire")):
            guard = trainer.Guard(self.cfg["system"], "cuda", lambda _: None, "main-scheduled")
            with self.assertRaisesRegex(trainer.StopRequested, "external GPU lock"):
                guard.check()


if __name__ == "__main__":
    unittest.main(verbosity=2)
