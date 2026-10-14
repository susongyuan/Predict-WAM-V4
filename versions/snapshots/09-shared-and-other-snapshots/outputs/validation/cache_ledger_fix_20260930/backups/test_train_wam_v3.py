#!/usr/bin/env python3
"""Trainer-only CPU tests. Synthetic tensors, temporary files; no real dataset, encoder, CUDA or simulation."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("train_wam_v3_under_test", HERE / "train_wam_v3.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)


class FakeDataset:
    def __init__(self, n=700, episode=41, near_from=400):
        self.episodes = [episode]
        self.anchors = [(episode, a) for a in range(n)]
        phase = np.where(np.arange(n) < near_from, 0, 1).astype(np.int8)
        phase[-5:] = -1
        hand = (np.arange(n) % 2).astype(np.int8)
        hand[-5:] = -1
        self.labels = {episode: (phase, np.full(n, 0.05, np.float32), hand)}
        self.skills = SimpleNamespace(segment=lambda ep, a: {"skill": "press" if a % 2 else "pick up from", "start": 0})
        self.label_report = {episode: [{"skill": "place on", "hand": None, "reason": "no event"}]}
        self.fetches = 0

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        self.fetches += 1
        ep, a = self.anchors[i]
        ph, di, ha = self.labels[ep]
        action = torch.zeros(32, 23)
        action[31, 7:10] = 1
        action[31, 15:18] = 2
        # Deliberately contradictory observation EEF positions; evaluation must ignore those columns.
        state = torch.zeros(2, 61)
        state[:, 17:20] = 900
        state[:, 42:45] = -900
        return {"episode": ep, "anchor": a, "contract_start": 0, "phase": int(ph[a]), "active_hand": int(ha[a]),
                "dist": float(di[a]), "state": state, "pool": torch.full((3, 2, 2), a / 100.0),
                "feats": torch.zeros((3, 1, 2, 2), dtype=torch.int16), "task_cond": torch.zeros(2),
                "future_feats": torch.zeros((3, 1, 2), dtype=torch.int16), "action_sequence": action, "skill_class": a % 2}


class FakeKin:
    def __init__(self):
        self.calls = []

    def fk(self, hand, q):
        self.calls.append((hand, np.array(q)))
        assert len(q) == 11
        return q[:3] + q[4:7], np.eye(3)


def fake_hold(state, chunk_len=32):
    out = torch.zeros(len(state), chunk_len, 23)
    out[..., 3:7] = state[:, :1, 53:57]
    out[..., 7:14] = state[:, :1, 3:10]
    out[..., 15:22] = state[:, :1, 28:35]
    return out


class FakeActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.mem_proj = torch.nn.Linear(2, 3, bias=False)
        torch.nn.init.constant_(self.mem_proj.weight, 1)


class FakeWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.action_model = FakeActionModel()
        self.samples = []
        self.forward_calls = 0
        self.nan_at = None

    def forward(self, batch, lambdas):
        self.forward_calls += 1
        target = batch.get("train_target", torch.tensor(1.0))
        loss = (self.action_model.weight - target).square().mean()
        if self.nan_at == self.forward_calls:
            loss = loss * float("nan")
        # Explicit random consumption to test eval isolation, without making the fake's loss stochastic.
        torch.randn(5); random.random(); np.random.rand()
        return loss, {"mse": float(loss.detach())}, {}

    def sample_actions(self, feats, state, pool, task_cond, num_steps, generator):
        assert len(state) == 1
        assert generator is not None
        projection = self.action_model.mem_proj(pool.float())
        x = torch.randn((1, 32, 23), generator=generator) * 0.01 + self.action_model.weight
        self.samples.append({"seed": generator.initial_seed(), "pred": x.detach().clone(), "num_steps": num_steps})
        return x, {"gate": torch.ones(1, 3) / 3, "phase_logits": torch.tensor([[0., 2., 0., 0.]]),
                   "log_dist": projection.mean((1, 2, 3))}


def small_plan():
    ds = FakeDataset(20, near_from=0)
    return ds, T.fixed_plan(ds, 12, general_n=5, fine_n=6)


def evaluate(wrapper, ds, plan):
    return T.evaluate(wrapper, ds, plan, kin=FakeKin(), hold_fn=fake_hold, lambdas={}, num_steps=10)


def loop(wrapper=None, *, steps=5, accum=3, every=2, check=lambda: None, validation=None, loader=None):
    wrapper = wrapper or FakeWrapper()
    updates = []
    class CountedSGD(torch.optim.SGD):
        def step(self, closure=None):
            updates.append(1)
            return super().step(closure)
    opt = CountedSGD(wrapper.parameters(), lr=0.1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1)
    saved, logs, validations = [], [], []
    def val(step):
        validations.append(step)
        return validation(step) if validation else {T.METRIC: 1 + step, "flow": {"mse": 1 / (step + 1)}}
    def save(tag, step, meta):
        saved.append((tag, step, copy.deepcopy(meta)))
    try:
        result = T.train_loop(wrapper, opt, sch, loader if loader is not None else [{"train_target": torch.tensor(1.)}] * 2,
                              max_steps=steps, accum=accum, grad_clip=1, lambdas={}, validate_fn=val, save_fn=save,
                              emit=logs.append, check=check, every=every)
    except BaseException as exc:
        exc.test_artifacts = (saved, logs, validations, updates, wrapper)
        raise
    return result, saved, logs, validations, updates, wrapper, sch


class FixedValidationTests(unittest.TestCase):
    def test_filter_full_dataset_before_fixed_256_without_loading_features(self):
        ds = FakeDataset()
        plan = T.fixed_plan(ds, 0, general_n=2, fine_n=256)
        self.assertEqual(ds.fetches, 0)
        self.assertEqual(len(plan["fine"]), 256)
        self.assertEqual(plan["full_split_fine_candidates"], 295)
        self.assertTrue(all(r["anchor"] >= 400 for r in plan["fine"]))
        self.assertGreater(len({r["anchor"] for r in plan["fine"]} - {r["anchor"] for r in plan["general"]}), 250)
        self.assertEqual(plan, T.fixed_plan(ds, 0, 2, 256))
        self.assertEqual(sum(v["missing_labels"] for v in plan["coverage_by_skill"].values()), 5)
        self.assertIn("no event", json.dumps(plan))

    def test_insufficient_fine_population_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "require exactly 256"):
            T.fixed_plan(FakeDataset(100, near_from=80), 0)

    def test_k1_noise_repeatable_order_independent_and_rng_restored(self):
        ds, plan = small_plan()
        wrapper = FakeWrapper().train()
        torch.manual_seed(333); random.seed(55); np.random.seed(77)
        tr, pr, nr = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        first, rows1 = evaluate(wrapper, ds, plan)
        self.assertTrue(torch.equal(tr, torch.get_rng_state()))
        self.assertEqual(pr, random.getstate())
        np.testing.assert_array_equal(nr[1], np.random.get_state()[1])
        self.assertTrue(wrapper.training)
        n = len({(r["episode"], r["anchor"]) for r in plan["general"] + plan["fine"]})
        self.assertEqual(len(wrapper.samples), n)  # exactly one, NOT a mean over K samples
        self.assertEqual(len({r["seed"] for r in wrapper.samples}), n)
        plan["general"].reverse(); plan["fine"].reverse()
        second, rows2 = evaluate(wrapper, ds, plan)
        self.assertEqual(rows1, rows2)
        for a, b in zip(wrapper.samples[:n], wrapper.samples[n:]):
            self.assertTrue(torch.equal(a["pred"], b["pred"]))
            self.assertEqual(a["pred"].shape, (1, 32, 23))
        self.assertEqual(first[T.METRIC], second[T.METRIC])
        self.assertEqual(first["samples_per_anchor"], 1)
        self.assertEqual(first["terminal_step"], 31)

    def test_action31_fk_active_hand_and_hold_not_observation_eef(self):
        ds, plan = small_plan()
        wrapper = FakeWrapper()
        report, rows = evaluate(wrapper, ds, plan)
        for row in rows:
            if not row["is_fine"]:
                continue
            target_distance = math.sqrt(3) * (1 if row["active_hand"] == 0 else 2)
            self.assertAlmostEqual(row["fine_hold_eef_m"], target_distance)
            self.assertLess(row[T.METRIC], 10)  # observed EEF is +/-900 and must not enter this metric
        kin = FakeKin()
        action = np.arange(23)
        T.action_eef(kin, action, 1)
        np.testing.assert_array_equal(kin.calls[0][1], np.r_[action[3:7], action[15:22]])
        self.assertEqual(kin.calls[0][0], "right")
        self.assertAlmostEqual(report["fine_pred_minus_hold_m"], report[T.METRIC] - report["fine_hold_eef_m"])

    def test_gate_phase_and_projection_by_skill(self):
        ds, plan = small_plan()
        report, rows = evaluate(FakeWrapper(), ds, plan)
        self.assertAlmostEqual(report["gate_entropy_nats"], math.log(3), places=6)
        self.assertGreater(report["memory_projection"]["variance_mean"], 0)
        self.assertEqual(report["memory_projection"]["anchors"], len(rows))
        for skill in ("pick up from", "press"):
            value = report["by_skill"][skill]
            self.assertEqual(value["phase_accuracy"], 1)
            self.assertIn("phase_confusion", value)
            self.assertIn("memory_projection", value)
            self.assertIn("missing_labels", value["full_split_coverage"])

    def test_slot_embeddings_do_not_mask_projection_collapse(self):
        m = T.ProjectionMoments()
        x = torch.arange(18).reshape(3, 2, 3).float()
        for _ in range(6):
            m.add(x)
        self.assertEqual(m.result()["variance_mean"], 0)
        self.assertEqual(m.result()["fraction_dims_variance_below_1e-8"], 1)

    def test_eval_failure_restores_training_mode_and_hooks(self):
        ds, plan = small_plan()
        wrapper = FakeWrapper().train()
        wrapper.nan_at = 1
        with self.assertRaises(FloatingPointError):
            evaluate(wrapper, ds, plan)
        self.assertTrue(wrapper.training)
        self.assertFalse(wrapper.action_model.mem_proj._forward_hooks)

    def test_label_drift_rejected(self):
        ds, plan = small_plan()
        ep = ds.episodes[0]
        a = plan["fine"][0]["anchor"]
        ds.labels[ep][0][a] = 3
        with self.assertRaisesRegex(ValueError, "labels drifted"):
            evaluate(FakeWrapper(), ds, plan)


class LoopTests(unittest.TestCase):
    def test_best_uses_only_fk_strict_improvement_not_flow(self):
        best = T.Best()
        self.assertTrue(best.update({T.METRIC: 0.1, "flow": 99}, 0))
        self.assertFalse(best.update({T.METRIC: 0.2, "flow": 0}, 5000))
        self.assertFalse(best.update({T.METRIC: 0.1, "flow": -1}, 10000))
        self.assertTrue(best.update({T.METRIC: 0.099999, "flow": 999}, 15000))
        self.assertEqual(best.step, 15000)
        with self.assertRaises(FloatingPointError):
            best.update({T.METRIC: float("nan")}, 20000)

    def test_optimizer_step_count_accum_across_epochs_no_early_stop(self):
        r, saved, logs, vals, updates, wrapper, scheduler = loop()
        self.assertEqual(r["completed_updates"], 5)
        self.assertEqual(len(updates), 5)
        self.assertEqual(wrapper.forward_calls, 15)
        self.assertEqual(scheduler.last_epoch, 5)
        self.assertEqual(vals, [0, 2, 4, 5])
        self.assertEqual(r["best_step"], 0)  # worsening FK must not terminate training
        self.assertFalse(r["early_stopping"])
        self.assertIn(("baseline", 0), [(a, b) for a, b, _ in saved])
        self.assertIn(("step_000002", 2), [(a, b) for a, b, _ in saved])
        self.assertEqual(saved[-1][0:2], ("last", 5))
        self.assertGreater(r["epoch"], 1)

    def test_periodic_final_is_not_double_validated(self):
        r, saved, logs, vals, *_ = loop(steps=4)
        self.assertEqual(vals, [0, 2, 4])
        self.assertEqual(r["completed_updates"], 4)

    def test_baseline_best_and_later_best_selection(self):
        values = {0: 0.4, 2: 0.3, 4: 0.5, 5: 0.2}
        r, saved, *_ = loop(validation=lambda s: {T.METRIC: values[s], "flow": 10 * s})
        self.assertEqual([step for tag, step, _ in saved if tag == "best"], [0, 2, 5])
        self.assertEqual(r["best_step"], 5)

    def test_nan_loss_saves_last_without_counting_partial_update(self):
        wrapper = FakeWrapper()
        wrapper.nan_at = 4
        with self.assertRaises(FloatingPointError) as ctx:
            loop(wrapper)
        saved, logs, vals, updates, _ = ctx.exception.test_artifacts
        self.assertEqual(len(updates), 1)
        self.assertEqual(saved[-1][0:2], ("last", 1))
        self.assertTrue(saved[-1][2]["aborted"])

    def test_infinite_grad_saves_last(self):
        wrapper = FakeWrapper()
        wrapper.action_model.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
        with self.assertRaises(RuntimeError) as ctx:
            loop(wrapper, accum=1)
        saved, _, _, updates, _ = ctx.exception.test_artifacts
        self.assertEqual(updates, [])
        self.assertEqual(saved[-1][0:2], ("last", 0))

    def test_signal_or_resource_interrupt_saves_completed_step(self):
        wrapper = FakeWrapper()
        def check():
            if wrapper.forward_calls >= 4:
                raise T.StopRequested("SIGTERM or resource limit")
        with self.assertRaises(T.StopRequested) as ctx:
            loop(wrapper, check=check)
        saved, _, _, updates, _ = ctx.exception.test_artifacts
        self.assertEqual(len(updates), 1)
        self.assertEqual(saved[-1][0:2], ("last", 1))
        self.assertEqual(saved[-1][2]["discarded_microbatches"], 1)

    def test_validation_exception_also_saves_last(self):
        def val(step):
            if step:
                raise RuntimeError("validation failure")
            return {T.METRIC: 0.1}
        with self.assertRaisesRegex(RuntimeError, "validation failure") as ctx:
            loop(validation=val)
        saved, *_ = ctx.exception.test_artifacts
        self.assertEqual(saved[-1][0:2], ("last", 2))
        self.assertIn(("step_000002", 2), [(a, b) for a, b, _ in saved])

    def test_empty_loader_saves_last(self):
        with self.assertRaisesRegex(ValueError, "no batches") as ctx:
            loop(loader=[])
        self.assertEqual(ctx.exception.test_artifacts[0][-1][0:2], ("last", 0))


class SafetyTests(unittest.TestCase):
    def config(self):
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(T.ROOT / T.DEFAULT_CONFIG), resolve=True)

    def test_pilot_is_default_and_cannot_promote(self):
        cfg = self.config()
        args = T.parse_args(["--run-name", "fake"])
        self.assertEqual(args.mode, "pilot")
        self.assertEqual(T.validate_config(cfg, args), 2000)
        args.max_steps = 2001
        with self.assertRaisesRegex(ValueError, "no pilot auto-promotion"):
            T.validate_config(cfg, args)
        args.max_steps = 2
        self.assertEqual(T.validate_config(cfg, args), 2)
        self.assertEqual(cfg["training"]["schedule_steps"], 100000)

    def test_horizon_independent_from_k(self):
        cfg = self.config()
        args = T.parse_args(["--run-name", "fake"])
        for horizon in (1, 8, 16):
            cfg["common"]["action_execution_horizon"] = horizon
            self.assertEqual(T.validate_config(cfg, args), 2000)
        cfg["inference"]["samples"] = 8
        with self.assertRaisesRegex(ValueError, "NOT execution horizon"):
            T.validate_config(cfg, args)

    def test_long_and_early_stop_rejected_without_authorization(self):
        cfg = self.config()
        args = T.parse_args(["--run-name", "fake", "--mode", "long"])
        with self.assertRaisesRegex(ValueError, "approval"):
            T.validate_config(cfg, args)
        args.mode = "pilot"
        cfg["training"]["early_stopping"] = {"patience": 10}
        with self.assertRaisesRegex(ValueError, "forbidden"):
            T.validate_config(cfg, args)

    def test_source_pin_cannot_be_overridden_by_config(self):
        cfg = self.config()
        cfg["training"]["init_from_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "pinned V2"):
            T.validate_config(cfg, T.parse_args(["--run-name", "fake"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.pt"
            path.write_bytes(b"not a trusted pickle")
            with patch.object(torch, "load", side_effect=AssertionError("must not deserialize")) as load:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    T.require_hash(path, T.V2_SHA256)
                load.assert_not_called()

    def test_exclusive_run_no_traversal_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = T.new_run("pilot_cpu_test", tmp)
            (out / "evidence").write_text("preserve")
            with self.assertRaises(FileExistsError):
                T.new_run("pilot_cpu_test", tmp)
            with self.assertRaises(ValueError):
                T.new_run("../shared", tmp)
            self.assertEqual((out / "evidence").read_text(), "preserve")

    def test_ledger_hash_and_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source"
            path.write_bytes(b"source")
            ledger = T.SourceLedger()
            ledger.add(path, T.sha256(path)); ledger.unchanged()
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "source drift"):
                ledger.unchanged()

    def test_model_only_checkpoint_keeps_mode_step_and_no_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeActionModel()
            path = T.save_checkpoint(tmp, "best", model, 7, {"mode": "pilot", "initialization_sha256": T.V2_SHA256})
            state = torch.load(path, weights_only=True)
            self.assertEqual(state["step"], 7)
            self.assertEqual(state["mode"], "pilot")
            self.assertTrue(state["model_only"])
            self.assertNotIn("optimizer_state_dict", state)
            self.assertNotIn("scheduler_state_dict", state)
            self.assertEqual(state["checkpoint_metric"], T.METRIC)
            self.assertFalse(Path(path + ".tmp").exists())

    def test_optimizer_and_schedule_fresh_and_not_shortened_for_pilot(self):
        cfg = self.config()["training"]
        model = FakeActionModel()
        opt, sch = T.make_optimizer(model, cfg)
        self.assertEqual(len(opt.state), 0)
        self.assertEqual(sch.last_epoch, 0)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], cfg["learning_rate"] * 0.01)
        self.assertAlmostEqual(sch.lr_lambdas[0](2000), 1)
        self.assertGreater(sch.lr_lambdas[0](5000), cfg["lr_min"] / cfg["learning_rate"])
        self.assertAlmostEqual(sch.lr_lambdas[0](100000), cfg["lr_min"] / cfg["learning_rate"])

    def test_split_leakage_and_wrong_split_rejected(self):
        train = FakeDataset(10, episode=1, near_from=0)
        val = FakeDataset(10, episode=2, near_from=0)
        split = {"train_episode_indices": [1], "validation_episode_indices": [2]}
        T.assert_splits(train, val, split)
        val.episodes = [1]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            T.assert_splits(train, val, split)
        val.episodes = [3]
        with self.assertRaisesRegex(ValueError, "pinned"):
            T.assert_splits(train, val, split)

    def test_long_approval_requires_all_hashed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval.json"
            evidence = Path(tmp) / "review.json"
            evidence.write_text('{"passed":true}')
            data = {"approved": True, "config_sha256": "c" * 64, "initialization_sha256": T.V2_SHA256,
                    "checks": {k: {"passed": True, "path": str(evidence), "sha256": T.sha256(evidence)}
                               for k in ("cache_audit", "cpu_checks", "tiny_overfit", "pilot_2k", "independent_review")}}
            T.atomic_json(path, data)
            args = SimpleNamespace(mode="long", long_run_approval=str(path))
            self.assertTrue(T.verify_long_approval(args, "c" * 64)["attestation"]["approved"])
            evidence.write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                T.verify_long_approval(args, "c" * 64)

    def test_cpu_resource_guard_no_nvidia_calls(self):
        cfg = self.config()["system"]
        cfg["ram_abort_percent"] = 101
        guard = T.ResourceGuard(cfg, "cpu", lambda r: None)
        with patch.object(T.subprocess, "run", side_effect=AssertionError("no GPU telemetry in CPU test")):
            guard.check(force=True)
        self.assertIn("ram_percent", guard.latest)
        guard.signal = 15
        with self.assertRaisesRegex(T.StopRequested, "signal 15"):
            guard.check()

    def test_resource_telemetry_fails_closed_after_three_errors(self):
        guard = T.ResourceGuard(self.config()["system"], "cpu", lambda r: None)
        with patch.object(Path, "read_text", side_effect=OSError("telemetry unavailable")):
            guard.check(force=True)
            guard.check(force=True)
            with self.assertRaisesRegex(T.StopRequested, "three times"):
                guard.check(force=True)

    def test_tiny_mode_budget_and_config_not_mutated(self):
        cfg = self.config()
        original = copy.deepcopy(cfg)
        args = T.parse_args(["--mode", "tiny", "--run-name", "tiny_cpu_fake"])
        self.assertEqual(T.validate_config(cfg, args), 200)
        self.assertEqual(cfg, original)
        args.max_steps = 2001
        with self.assertRaises(ValueError):
            T.validate_config(cfg, args)
        ds = FakeDataset(20, episode=1, near_from=0)
        plan = T.fixed_plan(ds, 0, 4, 4, tiny=True)
        self.assertEqual(plan["split"], "train_diagnostic")
        self.assertTrue(all(r["episode"] == 1 for r in plan["fine"]))

    def test_source_contract_no_encoder_resume_or_early_stop_calls(self):
        source = (HERE / "train_wam_v3.py").read_text()
        tree = ast.parse(source)
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertNotIn("create_vision_encoder", calls)
        self.assertNotIn("build_models", calls)
        self.assertNotIn("early_stopping_update", calls)
        self.assertIn("restore_training_state", calls)
        self.assertIn('optimizer.load_state_dict(state["optimizer_state_dict"])', source)
        self.assertIn('scheduler.load_state_dict(state["scheduler_state_dict"])', source)
        self.assertIn('model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)', source)
        self.assertIn("motion_sampling_weights", calls)
        self.assertIn("R1ProKinematics", calls)
        self.assertIn('target[0, 31]', source)
        self.assertIn('pred[0, 31]', source)
        self.assertIn('vision_encoder=None', source)
        self.assertNotIn("omnigibson.launch", source)
        # Only CLI main may start work; importing module did not initialize CUDA.
        self.assertFalse(torch.cuda.is_initialized())


class ResumeAndCacheTests(unittest.TestCase):
    def config(self):
        return SafetyTests().config()

    @staticmethod
    def seeded(seed=71):
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    def run_stochastic(self, target_step, *, checkpoint=None, stop_after_calls=None, disk=None):
        class StochasticWrapper(FakeWrapper):
            def forward(self, batch, lambdas):
                self.forward_calls += 1
                noise = torch.randn(()) + random.random() + float(np.random.rand())
                target = batch["train_target"].float().mean() + noise
                loss = (self.action_model.weight - target).square()
                return loss, {"mse": float(loss.detach())}, {}
        wrapper = StochasticWrapper()
        if checkpoint:
            wrapper.action_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        opt, scheduler = T.make_optimizer(wrapper.action_model, self.config()["training"])
        def loader(epoch):
            gen = torch.Generator().manual_seed(T.noise_seed(9, epoch, 0, "train-sampler-epoch"))
            sampler = torch.utils.data.WeightedRandomSampler(torch.tensor([1., 4., 2.]), 6, replacement=True, generator=gen)
            return torch.utils.data.DataLoader([{"train_target": float(i)} for i in range(3)], batch_size=2, sampler=sampler,
                                              generator=torch.Generator().manual_seed(epoch + 99), num_workers=0)
        saved = []
        def save(tag, step, extra):
            meta = {"config": self.config(), "mode": "pilot", "initialization_sha256": T.V2_SHA256, **extra}
            # Freeze fake artifacts just as torch.save would; optimizer.state_dict() references mutable tensors.
            frozen = copy.deepcopy({"step": step, "model_state_dict": wrapper.action_model.state_dict(), **meta})
            saved.append((tag, frozen))
            if disk is not None:
                T.save_checkpoint(disk, tag, wrapper.action_model, step, meta)
        def check():
            if stop_after_calls is not None and wrapper.forward_calls >= stop_after_calls:
                raise T.StopRequested("interrupt midway through accumulation")
        try:
            result = T.train_loop(wrapper, opt, scheduler, loader, max_steps=target_step, accum=2, grad_clip=1,
                                  lambdas={}, validate_fn=lambda step: {T.METRIC: 1 + step}, save_fn=save,
                                  emit=lambda r: None, every=2, check=check,
                                  resume_state=checkpoint["resume_state"] if checkpoint else None)
        except T.StopRequested:
            result = None
        return wrapper, opt, scheduler, saved, result

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a.cpu(), b.cpu()))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_exact_resume_matches_uninterrupted_adamw_rng_and_epoch_cursor(self):
        self.seeded()
        whole, opt, sch, saved, _ = self.run_stochastic(6)
        whole_rng = T.capture_rng("cpu")
        self.seeded()
        _, _, _, interrupted, result = self.run_stochastic(6, stop_after_calls=5)
        self.assertIsNone(result)
        source = interrupted[-1][1]
        self.assertEqual(source["step"], 2)
        self.assertEqual(source["resume_state"]["cursor"]["epoch"], 2)
        self.assertEqual(source["resume_state"]["cursor"]["batch_skip"], 1)
        self.assertEqual(source["resume_state"]["discarded_microbatches_replayed"], 1)
        self.seeded(999)  # construction/prefetch/random work before restoring must have no effect
        resumed, opt2, sch2, saved2, report = self.run_stochastic(6, checkpoint=source)
        self.assertEqual(report["completed_updates"], 6)
        self.assertEqual(report["resumed_from_step"], 2)
        self.assertEqual(report["best_step"], 0)
        self.assert_tree_equal(whole.action_model.state_dict(), resumed.action_model.state_dict())
        self.assert_tree_equal(opt.state_dict(), opt2.state_dict())
        self.assert_tree_equal(sch.state_dict(), sch2.state_dict())
        self.assert_tree_equal(whole_rng, T.capture_rng("cpu"))
        self.assertIn("resume_baseline", [tag for tag, _ in saved2])
        self.assertNotIn("baseline", [tag for tag, _ in saved2])

    def test_last_serializes_restorable_state_while_best_is_model_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.seeded()
            self.run_stochastic(2, disk=tmp)
            path = Path(tmp) / "checkpoint_last.pt"
            ck, digest = T.load_resume_source(path, self.config(), "pilot", "cpu")
            self.assertEqual(ck["step"], 2)
            self.assertFalse(ck["model_only"])
            self.assertTrue(ck["optimizer_state_saved"])
            self.assertTrue(ck["resume_state"]["optimizer_state_dict"]["state"])
            self.assertEqual(ck["resume_state"]["scheduler_state_dict"]["last_epoch"], 2)
            self.assertEqual(ck["resume_state"]["best"]["step"], 0)
            self.assertEqual(digest, T.sha256(path))
            best = torch.load(Path(tmp) / "checkpoint_best.pt", weights_only=False)
            self.assertTrue(best["model_only"])
            self.assertNotIn("resume_state", best)
            with self.assertRaisesRegex(ValueError, "safe optimizer-boundary"):
                T.load_resume_source(Path(tmp) / "checkpoint_best.pt", self.config(), "pilot", "cpu")
            # Verify before unpickling; tampering with last must never deserialize.
            with path.open("ab") as f:
                f.write(b"changed")
            with patch.object(torch, "load", side_effect=AssertionError("unsafe pickle")) as load:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    T.load_resume_source(path, self.config(), "pilot", "cpu")
                load.assert_not_called()

    def test_resume_inherits_prior_best_into_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old"; old.mkdir()
            new = Path(tmp) / "new"; new.mkdir()
            self.run_stochastic(2, disk=old)
            ck, _ = T.load_resume_source(old / "checkpoint_last.pt", self.config(), "pilot", "cpu")
            inherited = T.inherit_best_checkpoint(new, ck, old / "checkpoint_last.pt")
            self.assertEqual(inherited["step"], 0)
            self.assertEqual(T.sha256(old / "checkpoint_best.pt"), T.sha256(new / "checkpoint_best.pt"))
            self.assertEqual(Path(inherited["path"]).parent, new)
            with self.assertRaises(FileExistsError):
                T.inherit_best_checkpoint(new, ck, old / "checkpoint_last.pt")

    def test_resume_refuses_config_mode_and_schedule_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_stochastic(2, disk=tmp)
            path = Path(tmp) / "checkpoint_last.pt"
            cfg = self.config()
            cfg["training"]["schedule_steps"] = 2000
            with self.assertRaisesRegex(ValueError, "config/mode"):
                T.load_resume_source(path, cfg, "pilot", "cpu")
            with self.assertRaisesRegex(ValueError, "config/mode"):
                T.load_resume_source(path, self.config(), "long", "cpu")
            ck = torch.load(path, weights_only=False)
            ck["resume_state"]["scheduler_state_dict"]["last_epoch"] = 0
            torch.save(ck, path)
            with self.assertRaisesRegex(ValueError, "scheduler/update"):
                T.load_resume_source(path, self.config(), "pilot", "cpu", T.sha256(path))

    def test_unsafe_optimizer_update_cannot_auto_resume(self):
        wrapper = FakeWrapper()
        opt, sch = T.make_optimizer(wrapper.action_model, self.config()["training"])
        saved = []
        def broken_step():
            with torch.no_grad():
                wrapper.action_model.weight.add_(999)
            raise RuntimeError("partially mutated optimizer")
        with patch.object(opt, "step", broken_step):
            with self.assertRaisesRegex(RuntimeError, "partially mutated"):
                T.train_loop(wrapper, opt, sch, [{"train_target": torch.tensor(1.)}], max_steps=3, accum=1,
                             grad_clip=1, lambdas={}, validate_fn=lambda s: {T.METRIC: 1}, emit=lambda r: None,
                             save_fn=lambda tag, step, meta: saved.append((tag, step, copy.deepcopy(meta))))
        self.assertEqual(saved[-1][0], "last")
        self.assertFalse(saved[-1][2]["resume_state"]["safe_optimizer_boundary"])

    def test_cache_receipt_reuses_without_reading_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = Path(tmp) / "feat.npy"
            payload.write_bytes(b"frozen-feature-bits")
            receipt = Path(tmp) / "cache-ledger.json"
            bindings = {"config": "a" * 64, "meta": "b" * 64}
            first = T.CacheLedger(receipt, T.SourceLedger()).verify_or_create([payload], bindings)
            self.assertEqual(first["payloads_rehashed_this_run"], 1)
            self.assertFalse(first["not_fresh_payload_hash"])
            contents = json.loads(receipt.read_text())
            row = contents["files"][0]
            self.assertTrue({"path", "size", "mtime_ns", "ctime_ns", "inode", "sha256"}.issubset(row))
            real_hash = T.sha256
            def no_payload_read(path, tick=None):
                self.assertNotEqual(Path(path), payload)
                return real_hash(path, tick)
            with patch.object(T, "sha256", side_effect=no_payload_read):
                sources = T.SourceLedger()
                reused = T.CacheLedger(receipt, sources).verify_or_create([payload], bindings)
            self.assertEqual(reused["payloads_rehashed_this_run"], 0)
            self.assertTrue(reused["not_fresh_payload_hash"])
            self.assertIn("historical", reused["assurance"])
            self.assertEqual(sources.files[str(payload)], real_hash(payload))

    def test_cache_receipt_invalidates_changed_file_even_with_restored_mtime(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            payload = Path(tmp) / "feat.npy"
            payload.write_bytes(b"aaaa")
            receipt = Path(tmp) / "cache-ledger.json"
            bindings = {"meta": "b" * 64}
            T.CacheLedger(receipt, T.SourceLedger()).verify_or_create([payload], bindings)
            old = payload.stat()
            payload.write_bytes(b"bbbb")  # same size
            os.utime(payload, ns=(old.st_atime_ns, old.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "invalidated"):
                T.CacheLedger(receipt, T.SourceLedger()).verify_or_create([payload], bindings)

    def test_cache_receipt_invalidates_config_meta_inventory_or_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = Path(tmp) / "feat.npy"
            payload.write_bytes(b"cache")
            receipt = Path(tmp) / "cache-ledger.json"
            bindings = {"meta": "b" * 64}
            T.CacheLedger(receipt, T.SourceLedger()).verify_or_create([payload], bindings)
            with self.assertRaisesRegex(ValueError, "config/meta"):
                T.CacheLedger(receipt, T.SourceLedger()).verify_or_create([payload], {"meta": "c" * 64})
            with self.assertRaisesRegex(ValueError, "inventory"):
                T.CacheLedger(receipt, T.SourceLedger()).verify_or_create([], bindings)
            data = json.loads(receipt.read_text()); data["files"][0]["sha256"] = "0" * 64
            T.atomic_json(receipt, data)
            with self.assertRaisesRegex(ValueError, "record digest"):
                T.CacheLedger(receipt, T.SourceLedger()).verify_or_create([payload], bindings)

    def test_resume_cli_separate_source_and_absolute_target(self):
        args = T.parse_args(["--run-name", "new_resume_run", "--resume-source", "/tmp/old/checkpoint_last.pt",
                             "--max-steps", "2000", "--cache-ledger", "/tmp/cache-receipt.json"])
        self.assertEqual(args.resume_source, "/tmp/old/checkpoint_last.pt")
        self.assertEqual(T.validate_config(self.config(), args), 2000)
        self.assertEqual(args.cache_ledger, "/tmp/cache-receipt.json")
        with patch.object(sys, "stderr"), self.assertRaises(SystemExit):
            T.parse_args(["--run-name", "x", "--init-from", "v2.pt", "--resume-source", "last.pt"])


class SegmentedBudgetTests(unittest.TestCase):
    def config(self, budget=30000):
        cfg = SafetyTests().config()
        cfg["training"].update(max_steps=budget, schedule_steps=budget)
        return cfg

    def args(self, *extra):
        return T.parse_args(["--run-name", "segmented_cpu_fake", "--mode", "long",
                             "--long-run-approval", "reviewed-approval.json", *extra])

    def test_30k_50k_100k_default_and_absolute_segments(self):
        for budget in (30000, 50000, 100000):
            with self.subTest(budget=budget):
                cfg = self.config(budget)
                self.assertEqual(T.validate_config(cfg, self.args()), budget)
                for target in range(5000, budget + 1, 5000):
                    args = self.args("--max-steps", str(target), "--resume-source", "previous/checkpoint_last.pt")
                    self.assertEqual(T.validate_config(cfg, args), target)
                    self.assertEqual(cfg["training"]["schedule_steps"], budget)
                with self.assertRaises(ValueError):
                    T.validate_config(cfg, self.args("--max-steps", str(budget + 1)))

    def test_invalid_budget_or_schedule_mismatch_still_fails_closed(self):
        for budget, schedule in ((2000, 2000), (40000, 40000), (30000, 100000), (50000, 30000)):
            cfg = self.config(budget)
            cfg["training"]["schedule_steps"] = schedule
            with self.assertRaisesRegex(ValueError, "schedule_steps must equal"):
                T.validate_config(cfg, self.args())

    def test_new_budget_does_not_bypass_approval_or_promote_pilot(self):
        cfg = self.config()
        args = self.args("--max-steps", "5000")
        args.long_run_approval = None
        with self.assertRaisesRegex(ValueError, "approval"):
            T.validate_config(cfg, args)
        args.mode = "pilot"
        with self.assertRaisesRegex(ValueError, "no pilot auto-promotion"):
            T.validate_config(cfg, args)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old-approval.json"
            T.atomic_json(path, {"approved": True, "config_sha256": "a" * 64, "initialization_sha256": T.V2_SHA256})
            args = self.args(); args.long_run_approval = str(path)
            with self.assertRaisesRegex(ValueError, "different config/source"):
                T.verify_long_approval(args, "b" * 64)

    def test_30k_config_diff_is_exactly_two_budget_values_and_keeps_v2_source(self):
        from omegaconf import OmegaConf
        base = OmegaConf.to_container(OmegaConf.load(T.ROOT / "configs/lilawam_task000_multiview_manip_v3_micro16_train.yaml"), resolve=True)
        new = OmegaConf.to_container(OmegaConf.load(T.ROOT / "configs/lilawam_task000_multiview_manip_v3_micro16_30k_train.yaml"), resolve=True)
        expected = copy.deepcopy(base)
        expected["training"].update(max_steps=30000, schedule_steps=30000)
        self.assertEqual(new, expected)
        self.assertEqual(new["training"]["init_from_sha256"], T.V2_SHA256)
        self.assertEqual(new["training"]["batch_size"], 16)
        self.assertEqual(new["training"]["grad_accum_steps"], 2)
        args = self.args("--max-steps", "5000")
        self.assertEqual(args.init_from, T.DEFAULT_INIT)
        self.assertIsNone(args.resume_source)
        self.assertEqual(T.validate_config(new, args), 5000)

    def test_schedule_uses_total_budget_not_first_segment_target(self):
        for budget in (30000, 50000, 100000):
            training = self.config(budget)["training"]
            opt, scheduler = T.make_optimizer(FakeActionModel(), training)
            floor = training["lr_min"] / training["learning_rate"]
            self.assertGreater(scheduler.lr_lambdas[0](5000), floor)
            self.assertAlmostEqual(scheduler.lr_lambdas[0](budget), floor)
            self.assertAlmostEqual(opt.param_groups[0]["lr"], training["learning_rate"] * 0.01)

    def test_segment_boundary_last_and_resume_keep_absolute_counter_and_schedule(self):
        # Synthetic pre-positioned boundaries: exercise just the boundary updates, NOT 30k training steps.
        training = self.config()["training"]
        wrapper = FakeWrapper()
        opt, scheduler = T.make_optimizer(wrapper.action_model, training)
        opt.param_groups[0]["lr"] = training["learning_rate"] * scheduler.lr_lambdas[0](4999)
        schedule_state = scheduler.state_dict()
        schedule_state.update(last_epoch=4999, _step_count=5000, _last_lr=[opt.param_groups[0]["lr"]])
        resume = {"step": 4999, "optimizer_state_dict": copy.deepcopy(opt.state_dict()),
                  "scheduler_state_dict": schedule_state, "rng": T.capture_rng("cpu"),
                  "cursor": {"epoch": 1, "batch_skip": 0}, "best": {"value": 0.1, "step": 0},
                  "initial_validation": {T.METRIC: 0.1}, "final_validation": {T.METRIC: 0.2}}
        saved = []
        def segment(target, state):
            return T.train_loop(wrapper, opt, scheduler, [{"train_target": torch.tensor(1.)}],
                                max_steps=target, accum=1, grad_clip=1, lambdas={},
                                validate_fn=lambda step: {T.METRIC: 1 + step}, emit=lambda row: None,
                                save_fn=lambda tag, step, meta: saved.append((tag, step, copy.deepcopy(meta))),
                                every=5000, resume_state=state)
        report = segment(5000, resume)
        self.assertEqual(report["completed_updates"], 5000)
        self.assertFalse(report["early_stopping"])
        self.assertEqual(wrapper.forward_calls, 1)
        self.assertEqual(saved[-1][:2], ("last", 5000))
        self.assertIn(("step_005000", 5000), [(tag, step) for tag, step, _ in saved])
        self.assertEqual(scheduler.last_epoch, 5000)
        next_state = saved[-1][2]["resume_state"]
        self.assertTrue(next_state["optimizer_state_dict"]["state"])
        report = segment(5001, next_state)
        self.assertEqual(report["resumed_from_step"], 5000)
        self.assertEqual(report["completed_updates"], 5001)
        self.assertEqual(scheduler.last_epoch, 5001)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], training["learning_rate"] * scheduler.lr_lambdas[0](5001))
        # Check the final 30k boundary with the same total-budget LR function.
        final_state = saved[-1][2]["resume_state"]
        final_state["step"] = 29999
        final_state["scheduler_state_dict"]["last_epoch"] = 29999
        final_state["optimizer_state_dict"]["param_groups"][0]["lr"] = training["learning_rate"] * scheduler.lr_lambdas[0](29999)
        report = segment(30000, final_state)
        self.assertEqual(report["completed_updates"], 30000)
        self.assertEqual(saved[-1][:2], ("last", 30000))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], training["lr_min"])

    def test_new_plan_label_report_is_json_roundtrip_stable_without_mutating_dataset(self):
        ds, _ = small_plan()
        # Numeric order 2,10 differs from JSON string-key order 10,2: this reproduces the old defect.
        ds.label_report = {2: [{"skill": "pick"}], 10: [{"skill": "press"}]}
        original = copy.deepcopy(ds.label_report)
        plan = T.fixed_plan(ds, 12, general_n=5, fine_n=6)
        self.assertEqual(set(plan["label_report"]), {"2", "10"})
        self.assertEqual(ds.label_report, original)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            T.atomic_json(path, plan)
            reloaded = json.loads(path.read_text())
        self.assertEqual(T.digest_json(plan), T.digest_json(reloaded))
        self.assertEqual(plan["general"], reloaded["general"])
        self.assertEqual(plan["fine"], reloaded["fine"])

    def test_global_legacy_digest_contract_is_unchanged(self):
        legacy = {"label_report": {2: [], 10: []}}
        expected = T.hashlib.sha256(b'{"label_report":{"2":[],"10":[]}}').hexdigest()
        self.assertEqual(T.digest_json(legacy), expected)
        # Intentionally do NOT silently recanonicalize old signatures; old-plan validation is separate.
        self.assertNotEqual(T.digest_json(legacy), T.digest_json(json.loads(json.dumps(legacy))))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
