#!/usr/bin/env python3
"""CPU-only synthetic tests: 5 task caches, real compact V3, two-step updates.

Run: .venvs/wam-fixed/bin/python -B scripts/test_robotwin_multitask_training.py
Temporary lowdim HDF and sparse canonical DINO-shaped arrays, never an encoder,
download, simulator, real training update or modification of baseline artifacts.
"""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate
from harness import robotwin_multitask_training as mt
from harness import robotwin_training_data as data
from harness.robotwin_dataset import JOINT_FIELDS, make_split
from harness.wam_v3 import create_v3_action_model
from scripts import train_robotwin_multitask_wam as trainer
from scripts.train_wam_v3 import BatchStream


def compact_config():
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/fixed_bimanual/robotwin_place_cans_wam.yaml"), resolve=True)
    cfg["dataset"]["cond_mode"] = "train_task_vtt"
    cfg["model"]["action_expert"].update(hidden_size=32, depth=1, num_heads=4, vlm_adapter_num_queries=4, adapter_depth=1)
    cfg["model"]["vision_encoder"]["concat"]["out_dim"] = 32
    cfg["model"]["v3"]["phase_hidden"] = 16
    cfg["model"]["future_feat"].update(num_decoder_layers=1)
    cfg["training"].update(max_steps=2, schedule_steps=2, warmup_steps=0, batch_size=2, grad_accum_steps=1,
        learning_rate=1e-3, lr_min=1e-4, save_every=1, validate_every=2, resume_save_every=1,
        validation_samples=1, sampler_epoch_batches=2)
    cfg["system"].update(cpu_threads=2, cpu_abort_percent=100, ram_abort_percent=100)
    return cfg


class MultiTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(197)
        cls.tmp = tempfile.TemporaryDirectory(prefix="test-robotwin-multitask-")
        cls.root = Path(cls.tmp.name)
        cls.cfg = compact_config()
        cls.specs = []
        for task_index in range(5):
            task = f"synthetic_{task_index}"
            task_root = cls.root / task
            hdf_root = task_root / "data"
            hdf_root.mkdir(parents=True)
            data.atomic_json(task_root / "SOURCE.json", {"task": task, "revision": "synthetic-only"})
            for episode in range(4):
                with h5py.File(hdf_root / f"episode_{episode:07d}.hdf5", "w") as f:
                    offset = 0
                    for field, width in JOINT_FIELDS:
                        value = (task_index * 10 + episode + np.arange(3)[:, None] * .1
                                 + np.arange(offset, offset + width)[None] * .01).astype(np.float32)
                        f[f"state/{field}"] = value
                        f[f"action/{field}"] = value + .5
                        offset += width
            manifest = make_split(hdf_root, seed=19, counts=(2, 1, 1))
            # VAL is intentionally out-of-distribution; fitting must not see it.
            with h5py.File(hdf_root / manifest["splits"]["val"][0], "r+") as f:
                for group in ("state", "action"):
                    for field, _ in JOINT_FIELDS:
                        f[f"{group}/{field}"][:] = 1e5
            split_path = task_root / "split.json"
            data.atomic_json(split_path, manifest)
            identity = data.cache_identity(manifest)
            cache = task_root / "cache" / data.digest(identity)
            data.atomic_json(cache / "cache_config.json", identity)
            for split in ("train", "val"):
                for name in manifest["splits"][split]:
                    ep = data.episode_identity(hdf_root / name, cache.name)
                    directory = data.episode_directory(cache, ep)
                    directory.mkdir()
                    for key, (shape, dtype) in data.payload_specs(ep).items():
                        arr = np.lib.format.open_memmap(directory / f"{key}.npy", mode="w+", dtype=dtype,
                                                       shape=shape, version=(1, 0))
                        if key == "feat":
                            pattern = np.random.default_rng(task_index).normal(size=1024).astype(np.float32)
                            for frame in range(3):
                                bits = torch.from_numpy(pattern * frame).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
                                arr[frame, 0, -1, 0] = bits
                        else:
                            for frame in range(3):
                                arr[frame] = frame * (task_index + 1) if split == "train" else 1e5
                        arr.flush()
                        arr._mmap.close()
                    data.atomic_json(directory / "meta.json", data.completed_meta(ep))
            cls.specs.append({"task": task, "split_manifest": str(split_path), "feature_cache": str(cache)})
        cls.datasets = mt.build_datasets(cls.specs, cls.cfg)
        cls.statistics = mt.fit_shared_statistics(cls.datasets["train"])
        cls.bank = mt.build_train_bank(cls.datasets["train"])
        model = create_v3_action_model(OmegaConf.create(cls.cfg), dino_hidden_size=1024, num_dino_layers=3,
                                      task_cond_dim=1024, patch_size=16)
        # Simulate a nonzero pretrained DiT mixing path rather than a fresh zero
        # AdaLN gate; production initialization is the native14 trained 30k model.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "adaLN_modulation.1.bias" in name:
                    parameter.fill_(.1)
        cls.initialization = {"model_state_dict": model.state_dict(), "step": 30000, "model_only": True}
        init_path = cls.root / "synthetic_native14.pt"
        torch.save(cls.initialization, init_path)
        cls.cfg["training"].update(init_from=str(init_path), init_from_sha256=trainer.sha256(init_path))
        cls.cfg = trainer.effective_config(cls.cfg)
        del model

    @classmethod
    def tearDownClass(cls):
        for rows in cls.datasets.values():
            for ds in rows.values():
                ds.close()
        cls.tmp.cleanup()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), "CPU test initialized CUDA")
        for rows in self.datasets.values():
            for ds in rows.values():
                ds.close()

    def test_task_schema_and_alias_fail_closed(self):
        self.assertEqual(mt.load_task_specs({"tasks": self.specs}), self.specs)
        for bad in (self.specs[:1], [self.specs[0], self.specs[0]], [{**self.specs[0], "task": "../x"}, self.specs[1]]):
            with self.assertRaises(ValueError):
                mt.load_task_specs(bad)
        with self.assertRaisesRegex(ValueError, "SOURCE mismatch"):
            mt.build_datasets([{**self.specs[0], "task": "wrong"}, self.specs[1]], self.cfg)

    def test_history_and_masks_do_not_cross_episode_or_task(self):
        mixed = mt.MultitaskDataset(self.datasets["train"], self.bank)
        for task_index, task in enumerate(mixed.tasks):
            ds = self.datasets["train"][task]
            offset = mixed.ends[task_index - 1] if task_index else 0
            for start in (0, 3):
                sample = mixed[offset + start]
                self.assertEqual(sample["task"], task)
                self.assertEqual(sample["anchor"], 0)
                self.assertEqual(sample["contract_start"], 0)
                self.assertTrue(sample["episode"].startswith(task + ":"))
                state = ds.lowdim[start // 3][0][0]
                np.testing.assert_array_equal(sample["state"], np.repeat(state[None], 7, 0))
                self.assertFalse(sample["pool"].any())
                np.testing.assert_array_equal(sample["task_cond"], self.bank["conditions"][task])
            tail = mixed[offset + 5]
            self.assertEqual(tail["action_valid_mask"].sum(), 1)
            self.assertFalse(tail["future_valid_mask"])
            self.assertFalse(tail["future_feats"].any())
            self.assertFalse(tail["action_sequence"][1:].any())
        with self.assertRaises(IndexError):
            mixed[-1]

    def test_lru_closes_maps_and_copied_samples_survive_eviction(self):
        ds = next(iter(self.datasets["train"].values()))
        sample = ds[0]
        mapping = ds._mapped[0]["feat"]._mmap
        copy_before = sample["feats"].copy()
        ds[3]
        self.assertTrue(mapping.closed)
        self.assertEqual(list(ds._mapped), [1])
        self.assertEqual(len(ds._mapped), 1)
        np.testing.assert_array_equal(sample["feats"], copy_before)
        ds[0]
        ds.close()
        self.assertFalse(ds._mapped)
        self.assertFalse(ds.__getstate__()["_mapped"])

    def test_train_only_statistics_bank_and_global_normalization(self):
        allowed = {(str(ds.cache_root), ep["hdf_sha256"]) for ds in self.datasets["train"].values() for ep in ds.identities}
        original = data.validate_payload
        def train_only(root, identity, **kwargs):
            self.assertIn((str(root), identity["hdf_sha256"]), allowed)
            return original(root, identity, **kwargs)
        with patch.object(data, "validate_payload", side_effect=train_only):
            statistics = mt.fit_shared_statistics(self.datasets["train"])
            bank = mt.build_train_bank(self.datasets["train"])
        self.assertEqual(statistics, self.statistics)
        self.assertEqual(bank, self.bank)
        states = np.concatenate([s for ds in self.datasets["train"].values() for s, _ in ds.lowdim])
        got = statistics["normalization"]["robotwin"]["state"]
        np.testing.assert_allclose(got["mean"], states.astype(np.float64).mean(0))
        np.testing.assert_allclose(got["std"], states.astype(np.float64).std(0))
        self.assertLess(max(got["max"]), 100)
        self.assertEqual(statistics["normalization"]["provenance"]["frame_count"], 30)
        self.assertEqual(bank["provenance"]["fit_split"], "train")
        with self.assertRaises(ValueError):
            mt.fit_shared_statistics(self.datasets["val"])
        with self.assertRaises(ValueError):
            mt.build_train_bank(self.datasets["val"])
        for row in self.specs:
            manifest = data.load_manifest(row["split_manifest"])
            self.assertEqual([r["episode"] for r in bank["records"][row["task"]]], manifest["splits"]["train"])

    def test_bank_rejects_zero_nonfinite_constant_and_identical_conditions(self):
        first, second = list(self.bank["conditions"])[:2]
        for value in ([0.] * 1024, [1.] * 1024, [float("nan")] * 1024, self.bank["conditions"][second]):
            bank = copy.deepcopy(self.bank)
            bank["conditions"][first] = value
            with self.assertRaises(ValueError):
                mt.validate_bank(bank)
        bank = copy.deepcopy(self.bank)
        bank["source_split"] = "val"
        with self.assertRaises(ValueError):
            mt.validate_bank(bank)

    def test_uniform_tasks_not_uniform_concat_frames_and_deterministic_replay(self):
        lengths = [2, 10, 100, 1000, 10000]
        sampler = mt.UniformTaskSampler(lengths, 25000, 91)
        indices = list(sampler)
        self.assertEqual(indices, list(sampler))
        self.assertNotEqual(indices[:100], list(mt.UniformTaskSampler(lengths, 100, 92)))
        counts = np.bincount(np.searchsorted(np.cumsum(lengths), indices, side="right"), minlength=5)
        self.assertTrue(np.all(np.abs(counts / len(indices) - .2) < .01), counts)
        mixed = mt.MultitaskDataset(self.datasets["train"], self.bank)
        def loader(epoch):
            return DataLoader(mixed, batch_size=2, sampler=mt.UniformTaskSampler([6] * 5, 6, 71 + epoch),
                              generator=torch.Generator().manual_seed(epoch), num_workers=0)
        stream = BatchStream(loader, "cpu")
        stream.next(); stream.next()
        cursor = stream.state()
        expected = stream.next()
        restored = BatchStream(loader, "cpu", cursor=cursor)
        actual = restored.next()
        for key in ("episode", "task"):
            self.assertEqual(expected[key], actual[key])
        for key in ("state", "pool", "task_cond", "action_sequence"):
            self.assertTrue(torch.equal(expected[key], actual[key]))

    def test_real_compact_v3_joint_gradient_frozen_base_and_condition_influence(self):
        wrapper, notes = mt.build_wrapper(self.cfg, self.statistics, initialization=self.initialization)
        self.assertEqual(notes["initialization"], "strict_native14_WAM_V3")
        self.assertTrue(torch.equal(wrapper.action_model.action_proj.weight, self.initialization["model_state_dict"]["action_proj.weight"]))
        self.assertIsNone(wrapper.vision_encoder)
        mixed = mt.MultitaskDataset(self.datasets["train"], self.bank)
        batch = default_collate([mixed[0], mixed[6]])
        projection = wrapper.action_model.task_cond_proj
        base = {k: v.clone() for k, v in projection.base.state_dict().items()}
        before = wrapper.action_model.action_proj.weight.detach().clone()
        optimizer, _ = trainer.make_optimizer(wrapper.action_model, self.cfg["training"])
        for _ in range(2):
            optimizer.zero_grad()
            loss, _, _ = wrapper(batch, self.cfg["training"]["auxiliary_loss"])
            loss.backward()
            evidence = mt.gradient_evidence(wrapper)
            self.assertGreater(evidence["vtt_projection_grad_norm"], 0)
            self.assertGreater(evidence["shared_action_grad_norm"], 0)
            optimizer.step()
        self.assertFalse(torch.equal(before, wrapper.action_model.action_proj.weight))
        self.assertGreater(float(projection.residual.weight.norm()), 0)
        for key, value in projection.base.state_dict().items():
            self.assertTrue(torch.equal(base[key], value))
        cond = batch["task_cond"]
        self.assertFalse(torch.allclose(projection(cond[:1]), projection(cond[1:])))
        feats = [batch["feats"][:1, :, layer] for layer in range(3)]
        predictions = []
        wrapper.eval()
        with torch.no_grad():
            for condition in (cond[:1], cond[1:]):
                pred, _ = wrapper.sample_actions(feats, batch["state"][:1], batch["pool"][:1], condition,
                    num_steps=2, generator=torch.Generator().manual_seed(37))
                predictions.append(pred)
        self.assertGreater(float((predictions[0] - predictions[1]).abs().max()), 0)

    def test_cli_schedule_safety_and_mode_isolation(self):
        args = trainer.parse_args(["--config", "a", "--tasks", "b"])
        self.assertEqual((args.mode, args.device), ("plan", "cpu"))
        for extra in (["--mode", "train"], ["--device", "cuda", "--mode", "smoke", "--out", "x"],
                      ["--max-mapped-episodes", "0"], ["--max-steps", "1"], ["--resume"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(["--config", "a", "--tasks", "b", *extra])
        for path, value in (("dataset.cond_mode", "zero_single_task"), ("training.schedule_steps", 3),
                            ("common.action_dim", 23), ("training.batch_size", 0)):
            cfg = copy.deepcopy(self.cfg)
            block, key = path.split(".")
            cfg[block][key] = value
            with self.assertRaises(ValueError):
                trainer.effective_config(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["training"].update(max_steps=20000, schedule_steps=20000, warmup_steps=1000)
        self.assertEqual(trainer.effective_config(cfg)["training"]["schedule_steps"], 20000)

    def test_cpu_end_to_end_resume_checkpoint_contract_and_inference(self):
        def run(name, max_steps=None, resume=False, mode="train"):
            out = self.root / name
            out.mkdir(exist_ok=resume)
            argv = ["--config", "unused", "--tasks", "unused", "--out", str(out), "--mode", mode]
            if max_steps:
                argv += ["--max-steps", str(max_steps)]
            if resume:
                argv += ["--resume"]
            args = trainer.parse_args(argv)
            source = trainer.file_identity(self.cfg["training"]["init_from"], True)
            return trainer.execute(self.cfg, args, out, self.specs, self.datasets, source)
        full = run("full")
        self.assertEqual(full["status"], "completed_fixed_budget")
        partial = run("resume", max_steps=1)
        self.assertEqual(partial["status"], "stopped_at_requested_boundary")
        resumed = run("resume", resume=True)
        self.assertEqual(resumed["status"], "completed_fixed_budget")
        a = trainer.load_checkpoint(self.root / "full/checkpoint_last.pt")
        b = trainer.load_checkpoint(self.root / "resume/checkpoint_last.pt")
        for key, value in a["model_state_dict"].items():
            self.assertTrue(torch.equal(value, b["model_state_dict"][key]), key)
        self.assertEqual(a["task_sample_counts"], b["task_sample_counts"])
        self.assertEqual(a["resume_state"]["cursor"], b["resume_state"]["cursor"])
        self.assertEqual(a["resume_state"]["scheduler"], b["resume_state"]["scheduler"])
        self.assertEqual(a["contract"]["condition_bank"], self.bank)
        self.assertEqual(a["contract"]["statistics"], self.statistics)
        self.assertFalse(a["contract"]["inference"]["record_video"])
        self.assertGreater(a["gradient_evidence"]["vtt_projection_grad_norm_max"], 0)
        path = self.root / "full/checkpoint_step_000002.pt"
        with self.assertRaises(ValueError):
            trainer.load_checkpoint(path, resume=True)
        wrapper, condition, contract = mt.load_multitask_policy(path, "synthetic_2", "cpu")
        self.assertFalse(wrapper.training)
        self.assertEqual(condition.dtype, np.float32)
        np.testing.assert_array_equal(condition, self.bank["conditions"]["synthetic_2"])
        self.assertEqual(contract, a["contract"])
        for key, value in wrapper.action_model.state_dict().items():
            self.assertTrue(torch.equal(value, a["model_state_dict"][key]))
        with self.assertRaisesRegex(ValueError, "unknown"):
            mt.load_multitask_policy(path, "not_in_training")
        wrong = copy.deepcopy(a["contract"])
        wrong["execution"]["mode"] = "smoke"
        with self.assertRaises(ValueError):
            trainer.load_checkpoint(self.root / "full/checkpoint_last.pt", wrong)
        result = run("smoke", mode="smoke")
        self.assertEqual(result["status"], "smoke_passed")
        self.assertIsNone(result["evaluation_checkpoint"])
        result_file = self.root / "full/validation_step_000002.json"
        metrics = json.loads(result_file.read_text())
        self.assertEqual(len(metrics["per_task"]), 5)
        self.assertEqual(metrics["selection"], "none")
        self.assertFalse(metrics["physical_evaluation"])
        self.assertTrue(all(not ds._mapped for rows in self.datasets.values() for ds in rows.values()))
        # Byte-level mutation must be caught before torch.load.
        damaged = self.root / "damaged.pt"
        damaged.write_bytes(b"not a checkpoint")
        trainer.atomic_json(damaged.with_name(damaged.name + ".sha256.json"), {"size": 16, "sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "receipt"):
            trainer.load_checkpoint(damaged)


if __name__ == "__main__":
    with patch.object(torch.cuda, "init", side_effect=AssertionError("CUDA is forbidden in CPU tests")):
        unittest.main(verbosity=2)
