#!/usr/bin/env python3
"""CPU-only extension tests; temporary synthetic caches and compact real WAM.

No GPU, services, simulator, actual training artifacts or frozen files touched.
The original synthetic fixture is reused, not copied. Run with wam-fixed -B.
"""
from __future__ import annotations
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import test_robotwin_multitask_training as fixtures
from scripts import train_robotwin_multitask_extended as ext
from scripts import train_robotwin_multitask_wam as old


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(nested_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(nested_equal(v, w) for v, w in zip(a, b))
    return a == b


class ExtendedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.MultiTaskTests.setUpClass()
        f = fixtures.MultiTaskTests
        cls.root, cls.specs = f.root, f.specs
        cls.cfg = copy.deepcopy(f.cfg)
        cls.cfg["training"].update(max_steps=20000, schedule_steps=20000)
        # Tiny CPU telemetry windows can round to 100%; do not make this
        # synthetic correctness test depend on concurrent system utilization.
        cls.cfg["system"]["cpu_abort_percent"] = 101
        # Two real CPU updates instead of 5000; same original 20k schema. All
        # continuation tests still have TOTAL50000, not a shortened new budget.
        cls.parent_dir = cls.root / "extension_parent"
        cls.parent_dir.mkdir()
        args = old.parse_args(["--config", "unused", "--tasks", "unused", "--mode", "train",
                               "--out", str(cls.parent_dir), "--max-steps", "2"])
        datasets, source = old.preflight(cls.cfg, cls.specs)
        old.execute(cls.cfg, args, cls.parent_dir, cls.specs, datasets, source)
        cls.parent = cls.parent_dir / "checkpoint_last.pt"
        cls.meta = ext.inspect_parent(cls.parent)
        cls.ck = old.load_checkpoint(cls.parent)
        cls.parent_bytes = {str(p): (p.stat().st_size, old.sha256(p)) for p in cls.parent_dir.iterdir() if p.is_file()}
        cls.frozen_bytes = {p: old.sha256(p) for p in cls.ck["contract"]["source_hashes"]}
        cls.original_factories = (old.make_optimizer, old.immutable_contract, old.execute)

    @classmethod
    def tearDownClass(cls):
        try:
            for name, before in cls.parent_bytes.items():
                p = Path(name)
                assert (p.stat().st_size, old.sha256(p)) == before, f"parent changed: {p}"
            for name, before in cls.frozen_bytes.items():
                assert old.sha256(name) == before, f"frozen source changed: {name}"
            assert (old.make_optimizer, old.immutable_contract, old.execute) == cls.original_factories
            assert not torch.cuda.is_initialized()
        finally:
            fixtures.MultiTaskTests.tearDownClass()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual((old.make_optimizer, old.immutable_contract, old.execute), self.original_factories)

    def args(self, name, step=4, resume=False, extra=()):
        argv = ["--parent-checkpoint", str(self.parent), "--out", str(self.root / name),
                "--total-steps", "50000", "--mode", "train"]
        if step is not None:
            argv += ["--stop-at-step", str(step)]
        if resume:
            argv += ["--resume"]
        return ext.parse_args([*argv, *extra])

    def test_metadata_plan_never_loads_tensors_or_constructs_encoder(self):
        absent = self.root / "plan_must_not_create"
        with patch.object(torch, "load", side_effect=AssertionError("tensor load")), \
             patch.object(old.mt, "build_datasets", side_effect=AssertionError("dataset")), \
             patch.object(old.mt, "build_wrapper", side_effect=AssertionError("model")), \
             patch.object(torch.cuda, "init", side_effect=AssertionError("CUDA")), \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(ext.main(["--parent-checkpoint", str(self.parent), "--out", str(absent)]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["start_step"], 2)
        self.assertEqual(result["total_steps"], 50000)
        self.assertTrue(result["handoff_val_ready_metadata"])
        self.assertFalse(result["weights_loaded"])
        self.assertFalse(result["writes_performed"])
        self.assertFalse(absent.exists())
        self.assertEqual(result["evaluation_checkpoint"], "checkpoint_step_050000.pt")

    def test_cli_refuses_unauthorized_budgets_and_bad_bounds(self):
        base = ["--parent-checkpoint", str(self.parent)]
        for extra in (["--total-steps", "150000"], ["--total-steps", "20000"], ["--mode", "train"],
                      ["--resume"], ["--stop-at-step", "50001"], ["--stop-at-step", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ext.parse_args([*base, *extra])
        self.assertEqual(ext.parse_args([*base, "--stop-at-step", "5002"]).stop_at_step, 5002)

    def test_budget_whitelist_preserves_every_other_inherited_field(self):
        contract = ext.make_contract(self.meta, ext.validate_resume_state(self.ck))
        self.assertEqual(contract["format"], old.FORMAT)
        self.assertEqual(contract["config"]["training"]["max_steps"], 50000)
        self.assertEqual(contract["config"]["training"]["schedule_steps"], 50000)
        self.assertEqual(contract["condition_bank"], self.ck["contract"]["condition_bank"])
        self.assertEqual(contract["statistics"], self.ck["contract"]["statistics"])
        self.assertEqual(contract["budget_extension"]["parent_checkpoint"]["sha256"], self.meta["receipt"]["sha256"])
        self.assertEqual(contract["budget_extension"]["parent_contract"], self.ck["contract"])
        for path, value in ((["config", "training", "batch_size"], 16),
                            (["config", "training", "learning_rate"], 9e-4),
                            (["config", "common", "action_dim"], 23),
                            (["inference", "record_video"], True),
                            (["condition_bank", "conditions", "synthetic_0"], [0.] * 1024)):
            bad = copy.deepcopy(contract)
            node = bad
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside"):
                ext.assert_whitelist(bad, self.ck["contract"])

    def test_continuous_lr_at_5000_first_next_update_and_final50000(self):
        t = copy.deepcopy(self.cfg["training"])
        t.update(learning_rate=5e-5, lr_min=5e-6, warmup_steps=1000)
        m = torch.nn.Linear(1, 1)
        opt, sched = old.make_optimizer(m, t)
        # Advance scheduler state cheaply; no GPU/model training here.
        with patch.object(opt, "step"):
            for _ in range(5000):
                opt.step(); sched.step()
        saved_opt, saved_sched = copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict())
        start_lr = opt.param_groups[0]["lr"]
        schedule = {"start_step": 5000, "start_lrs": [start_lr], "end_step": 50000, "lr_min": 5e-6}
        self.assertEqual(ext.continuation_lrs(5000, schedule), [start_lr])
        self.assertEqual(ext.continuation_lrs(50000, schedule), [5e-6])
        values = [ext.continuation_lrs(s, schedule)[0] for s in range(5000, 50001)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        t.update(max_steps=50000, schedule_steps=50000)
        contract = {"config": {"training": t}, "budget_extension": {"schedule": schedule}}
        continued_opt, continued_sched = ext.optimizer_factory(contract)(m, t)
        continued_opt.load_state_dict(saved_opt); continued_sched.load_state_dict(saved_sched)
        self.assertEqual(continued_opt.param_groups[0]["lr"], start_lr)
        self.assertEqual(continued_sched.last_epoch, 5000)
        with patch.object(continued_opt, "step"):
            continued_opt.step(); continued_sched.step()
        self.assertEqual(continued_sched.last_epoch, 5001)
        self.assertAlmostEqual(continued_opt.param_groups[0]["lr"], ext.continuation_lrs(5001, schedule)[0], places=20)
        _, fresh_sched = old.make_optimizer(m, t)
        fresh_lr_at5000 = t["learning_rate"] * fresh_sched.lr_lambdas[0](5000)
        self.assertNotAlmostEqual(fresh_lr_at5000, start_lr, places=10)
        with self.assertRaises(ValueError):
            ext.continuation_lrs(50001, schedule)

    def test_reject_missing_or_corrupt_full_continuation_states(self):
        def remove_moment(ck):
            next(iter(ck["resume_state"]["optimizer"]["state"].values())).pop("exp_avg")
        def wrong_moment_step(ck):
            next(iter(ck["resume_state"]["optimizer"]["state"].values()))["step"].fill_(0)
        mutations = [lambda ck: ck.update(model_only=True),
            lambda ck: ck["resume_state"].update(safe_optimizer_boundary=1),
            lambda ck: ck["resume_state"].pop("rng"),
            lambda ck: ck["resume_state"]["rng"].pop("python"),
            lambda ck: ck["resume_state"]["cursor"].update(batch_skip=1),
            lambda ck: ck["resume_state"]["scheduler"].update(last_epoch=0),
            lambda ck: ck["resume_state"]["scheduler"].update(_last_lr=[0.]),
            lambda ck: ck["task_sample_counts"].update(synthetic_0=9999),
            remove_moment, wrong_moment_step]
        for i, mutate in enumerate(mutations):
            bad = copy.deepcopy(self.ck); mutate(bad)
            with self.subTest(mutation=i), self.assertRaises((ValueError, KeyError)):
                ext.validate_resume_state(bad)

    def test_fresh_process_meta_geometry_without_prior_optimizer_import(self):
        import subprocess
        code = """from scripts import train_robotwin_multitask_extended as e
import sys, torch
ck=e.old.load_checkpoint(sys.argv[1])
e.verify_optimizer_geometry(ck)
assert not torch.cuda.is_initialized()
print('cold_meta_geometry_passed')
"""
        result=subprocess.run([sys.executable,'-B','-c',code,str(self.parent)],cwd=ROOT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},
            text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('cold_meta_geometry_passed',result.stdout)

    def test_meta_geometry_rejects_wrong_model_or_adam_shape(self):
        ext.verify_optimizer_geometry(self.ck)
        bad = copy.deepcopy(self.ck)
        bad["model_state_dict"]["action_proj.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "topology"):
            ext.verify_optimizer_geometry(bad)
        bad = copy.deepcopy(self.ck)
        moments = next(iter(bad["resume_state"]["optimizer"]["state"].values()))
        moments["exp_avg"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "shape"):
            ext.verify_optimizer_geometry(bad)

    def test_seed_checkpoint_preserves_weights_adam_rng_cursor_and_counts_bitwise(self):
        out = self.root / "seed_only"
        out.mkdir()
        contract = ext.make_contract(self.meta, ext.validate_resume_state(self.ck))
        ext.write_seed_checkpoint(out / "checkpoint_last.pt", self.ck, contract)
        ck = old.load_checkpoint(out / "checkpoint_last.pt", contract)
        for key in ("model_state_dict", "resume_state", "task_sample_counts", "gradient_evidence"):
            self.assertTrue(nested_equal(ck[key], self.ck[key]), key)
        self.assertEqual(ck["step"], 2)
        self.assertEqual(ck["format"], old.FORMAT)

    def test_cpu_continuation_split_resume_is_bit_exact_and_old_loader_compatible(self):
        # The actual frozen execute, real compact model, loss, sampler, VAL and
        # checkpoint functions run here. No fitted artifacts may be recomputed.
        with patch.object(old.mt, "fit_shared_statistics", side_effect=AssertionError("refit stats")), \
             patch.object(old.mt, "build_train_bank", side_effect=AssertionError("refit bank")):
            full = ext.run_train(self.args("ext_full"), self.meta)
            partial = ext.run_train(self.args("ext_split", 3), self.meta)
            resumed = ext.run_train(self.args("ext_split", 4, resume=True), self.meta)
        self.assertEqual(full["step"], 4)
        self.assertEqual(partial["step"], 3)
        self.assertEqual(resumed["step"], 4)
        self.assertEqual(full["schedule_budget"], 50000)
        self.assertIsNone(full["evaluation_checkpoint"])
        a = old.load_checkpoint(self.root / "ext_full/checkpoint_last.pt")
        b = old.load_checkpoint(self.root / "ext_split/checkpoint_last.pt")
        for key in ("model_state_dict", "resume_state", "task_sample_counts", "gradient_evidence", "contract"):
            self.assertTrue(nested_equal(a[key], b[key]), key)
        self.assertEqual(a["contract"]["condition_bank"], self.ck["contract"]["condition_bank"])
        self.assertEqual(a["contract"]["statistics"], self.ck["contract"]["statistics"])
        self.assertEqual(ext.read_json(self.root / "ext_full/validation_plan.json"), self.meta["artifacts"]["validation_plan.json"])
        for name in ("action_proj.weight", "task_cond_proj.residual.weight"):
            self.assertFalse(torch.equal(a["model_state_dict"][name], self.ck["model_state_dict"][name]))
        wrapper, condition, contract = old.mt.load_multitask_policy(
            self.root / "ext_full/checkpoint_step_000004.pt", "synthetic_0", "cpu")
        self.assertFalse(wrapper.training)
        self.assertIsNone(wrapper.vision_encoder)
        self.assertEqual(contract["config"]["training"]["max_steps"], 50000)
        np.testing.assert_array_equal(condition, self.ck["contract"]["condition_bank"]["conditions"]["synthetic_0"])
        self.assertTrue(nested_equal(wrapper.action_model.state_dict(), a["model_state_dict"]))
        metrics = [json.loads(s) for s in (self.root / "ext_full/metrics.jsonl").read_text().splitlines()]
        self.assertEqual([r["step"] for r in metrics if r["event"] == "optimizer_step"], [3, 4])
        init = next(r for r in metrics if r["event"] == "initialization")
        self.assertFalse(init["fresh_optimizer"])
        self.assertFalse(init["encoder_loaded"])
        with self.assertRaises(FileExistsError):
            ext.run_train(self.args("ext_full"), self.meta)
        with self.assertRaisesRegex(ValueError, "exceed"):
            ext.run_train(self.args("ext_split", 4, resume=True), self.meta)

    def test_independent_smoke_cannot_resume_or_become_a_parent(self):
        status = ext.run_train(self.args("cuda_smoke"), self.meta)
        self.assertEqual(status["step"], 4)
        self.assertEqual(status["status"], "smoke_passed")
        self.assertTrue(status["smoke_verification"]["frozen_base_unchanged"])
        self.assertFalse(status["smoke_verification"]["resume_admissible"])
        manifest = ext.read_json(self.root / "cuda_smoke/manifest.json")
        extension = manifest["contract"]["budget_extension"]
        self.assertEqual(extension["run_role"], "smoke")
        self.assertFalse(extension["evaluation_admissible_at_final_budget_only"])
        with self.assertRaisesRegex(ValueError, "smoke"):
            ext.run_train(self.args("cuda_smoke", 5, resume=True), self.meta)
        with self.assertRaisesRegex(ValueError, "original"):
            ext.inspect_parent(self.root / "cuda_smoke/checkpoint_last.pt")
        with self.assertRaisesRegex(ValueError, "smoke"):
            ext.run_train(self.args("separate_smoke", 3, extra=["--run-role", "smoke"]), self.meta)

    def test_missing_validation_and_changed_source_fail_before_output(self):
        bad = copy.deepcopy(self.meta); bad["ready"] = False
        with self.assertRaisesRegex(ValueError, "VAL"):
            ext.run_train(self.args("missing_val"), bad)
        self.assertFalse((self.root / "missing_val").exists())
        bad = copy.deepcopy(self.meta)
        key = next(iter(bad["parent_contract"]["source_hashes"]))
        bad["parent_contract"]["source_hashes"][key] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source"):
            ext.run_train(self.args("bad_source"), bad)
        self.assertFalse((self.root / "bad_source").exists())

    def test_device_and_external_gpu_lock_requirement_without_touching_gpu(self):
        contract = copy.deepcopy(self.ck["contract"])
        contract["execution"]["device"] = "cuda"
        with self.assertRaisesRegex(ValueError, "gpu-lock-owner"):
            ext.runtime_args(self.args("unused"), contract)
        with self.assertRaisesRegex(ValueError, "device"):
            ext.runtime_args(self.args("unused", extra=["--device", "cpu"]), contract)
        rt = ext.runtime_args(self.args("unused", extra=["--gpu-lock-owner", "external-owner"]), contract)
        self.assertEqual(rt.gpu_lock_owner, "external-owner")
        self.assertEqual(rt.device, "cuda")

    def test_parent_writer_race_is_rejected_before_creating_new_run(self):
        original = old.file_identity
        calls = 0
        def identity(path, *args, **kwargs):
            nonlocal calls
            result = original(path, *args, **kwargs)
            if str(Path(path).resolve()) == str(self.parent.resolve()):
                calls += 1
                if calls > 1:
                    result["inode"] += 1
            return result
        with patch.object(old, "file_identity", side_effect=identity), self.assertRaisesRegex(ValueError, "parent changed"):
            ext.run_train(self.args("writer_race"), self.meta)
        self.assertFalse((self.root / "writer_race").exists())

    def test_never_reuse_parent_directory_and_reject_resume_contract_tamper(self):
        args = self.args("unused")
        args.out = str(self.parent_dir)
        with self.assertRaisesRegex(ValueError, "parent run"):
            ext.run_train(args, self.meta)
        ext.run_train(self.args("tamper_child", 3), self.meta)
        path = self.root / "tamper_child/manifest.json"
        manifest = ext.read_json(path)
        manifest["contract"]["config"]["training"]["batch_size"] += 1
        manifest["contract_sha256"] = old.digest(manifest["contract"])
        old.atomic_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "contract/source/parent"):
            ext.run_train(self.args("tamper_child", 4, resume=True), self.meta)

    def test_fixed_validation_plan_cannot_change_on_extension_resume(self):
        ext.run_train(self.args("val_tamper_child", 3), self.meta)
        path = self.root / "val_tamper_child/validation_plan.json"
        plan = ext.read_json(path)
        plan["tasks"]["synthetic_0"]["seed"] += 1
        old.atomic_json(path, plan)
        with self.assertRaisesRegex(ValueError, "fixed validation plan"):
            ext.run_train(self.args("val_tamper_child", 4, resume=True), self.meta)

    def test_delegation_does_not_mutate_frozen_module_even_on_failure(self):
        contract = ext.make_contract(self.meta, ext.validate_resume_state(self.ck))
        rt = ext.runtime_args(self.args("delegation_failure"), contract)
        datasets, source = old.preflight(contract["config"], contract["tasks"])
        out = self.root / "delegation_failure"; out.mkdir()
        for name, value in self.meta["artifacts"].items():
            old.atomic_json(out / name, value)
        bad = copy.deepcopy(contract); bad["runtime"]["torch"] = "wrong"
        with self.assertRaisesRegex(ValueError, "live data/source"):
            ext.delegated_execute(bad, rt, out, datasets, source)
        self.assertEqual((old.make_optimizer, old.immutable_contract, old.execute), self.original_factories)


if __name__ == "__main__":
    unittest.main(verbosity=2)
