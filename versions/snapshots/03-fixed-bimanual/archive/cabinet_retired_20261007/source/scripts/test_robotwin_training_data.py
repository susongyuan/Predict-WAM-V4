#!/usr/bin/env python3
"""Small synthetic CPU tests; no real encoder/GPU/download/simulator.

.venvs/wam-fixed/bin/python -B scripts/test_robotwin_training_data.py
Evidence: outputs/validation/robotwin_training_data_20261004/tests.json.
Synthetic HDF/cache files are temporary children of that output directory.
"""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import h5py
import numpy as np
import torch
from harness import robotwin_training_data as data
from harness.robotwin_dataset import make_split, RoboTwinEpisode
from harness.wam_behavior import normalize_image
from harness.wam_v3 import pool_from_hidden
from scripts.test_robotwin_dataset import episode_file, SHAPE
from scripts import cache_robotwin_features as cli

OUT = ROOT / "outputs/validation/robotwin_training_data_20261004"


class CachedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=OUT, prefix="synthetic-")
        cls.root = Path(cls.tmp.name)
        cls.hdf = cls.root / "data"
        cls.hdf.mkdir()
        (cls.root / "SOURCE.json").write_text(json.dumps({"task": "synthetic", "revision": "test-only"}))
        for i in range(5):
            episode_file(cls.hdf, i, frames=4)
        cls.manifest = make_split(cls.hdf, seed=20261004, counts=(3, 1, 1))
        cls.manifest_path = cls.root / "split.json"
        data.atomic_json(cls.manifest_path, cls.manifest)
        cls.identity, plan = cli.build_plan(cls.manifest, cls.root / "cache")
        cls.cache = Path(plan["cache_root"])
        cls.config_id = plan["config_id"]
        data.atomic_json(cls.cache / "cache_config.json", cls.identity)
        cls.identities = {}
        # Only TRAIN has payloads. Val/heldout must not be needed by fitting.
        for name in cls.manifest["splits"]["train"]:
            ep = data.episode_identity(cls.hdf / name, cls.config_id)
            cls.identities[name] = ep
            directory = data.episode_directory(cls.cache, ep)
            directory.mkdir()
            for key, (shape, dtype) in data.payload_specs(ep).items():
                arr = np.lib.format.open_memmap(directory / f"{key}.npy", "w+", dtype=dtype,
                                                 shape=shape, version=(1, 0))
                if key == "feat":
                    for frame in range(4):
                        bits = torch.tensor(float(frame + 1)).to(torch.bfloat16).view(torch.int16).item() & 0xffff
                        arr[frame] = bits
                else:
                    for frame in range(4):
                        for view in range(3):
                            arr[frame, view] = frame * (view + 1)
                arr.flush()
                del arr
            data.atomic_json(directory / "meta.json", data.completed_meta(ep))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def dataset(self, **kwargs):
        return data.RobotwinCachedDataset(self.manifest_path, "train", self.cache.parent,
                                         [1, 3], [1, 3], **kwargs)

    def test_history_masks_no_action_shift_or_episode_crossing(self):
        ds = self.dataset(chunk_size=3, future_offset=1)
        self.assertEqual(len(ds), 12)
        sample = ds[1]
        state, actions = ds.lowdim[0]
        np.testing.assert_array_equal(sample["state"], state[[1, 0, 0, 0]])
        np.testing.assert_array_equal(sample["action_sequence"], actions[1:4])
        np.testing.assert_array_equal(sample["pool"][:, :, 0], [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]])
        end, start = ds[3], ds[4]
        np.testing.assert_array_equal(end["action_valid_mask"], [True, False, False])
        self.assertFalse(end["future_valid_mask"])
        self.assertFalse(end["future_feats"].any())
        self.assertFalse(end["action_sequence"][1:].any())
        np.testing.assert_array_equal(start["state"], np.repeat(ds.lowdim[1][0][:1], 4, axis=0))
        self.assertEqual(start["anchor"], 0)
        np.testing.assert_array_equal(ds[3]["state"], state[[3, 2, 0, 0]])

    def test_shapes_types_future_and_collation(self):
        ds = self.dataset(future_offset=1)
        sample = ds[1]
        self.assertEqual(sample["feats"].shape, (3, 3, 201, 1024))
        self.assertEqual(sample["feats"].dtype, np.uint16)
        self.assertEqual(sample["future_feats"].shape, (3, 196, 1024))
        self.assertEqual(sample["pool"].shape, (3, 4, 2048))
        self.assertEqual(sample["pool"].dtype, np.float32)
        self.assertTrue(sample["future_valid_mask"])
        np.testing.assert_array_equal(sample["future_feats"], ds[2]["feats"][:, -1, 5:])
        self.assertEqual((sample["phase"], sample["active_hand"], sample["dist"]), (-1, -1, 0))
        self.assertFalse(sample["task_cond"].any())
        batch = torch.utils.data.default_collate([ds[0], ds[1]])
        self.assertEqual(batch["feats"].dtype, torch.uint16)
        self.assertEqual(batch["future_valid_mask"].dtype, torch.bool)

    def test_missing_payload_wrong_dtype_shape_and_identity_rejected(self):
        ep = next(iter(self.identities.values()))
        directory = data.episode_directory(self.cache, ep)
        file = directory / "pool.npy"
        backup = file.with_suffix(".bak")
        file.rename(backup)
        try:
            with self.assertRaises(ValueError):
                self.dataset()
            for array in (np.zeros((4, 3, 2048), np.float64), np.zeros((4, 3, 2047), np.float32)):
                np.save(file, array)
                with self.assertRaises(ValueError):
                    self.dataset()
        finally:
            file.unlink(missing_ok=True)
            backup.rename(file)
        meta = directory / "meta.json"
        try:
            data.atomic_json(meta, {"complete": True, "identity": {**ep, "hdf_sha256": "bad"}})
            with self.assertRaises(ValueError):
                self.dataset()
        finally:
            data.atomic_json(meta, data.completed_meta(ep))

    def test_train_only_statistics_float64_and_temporal_shape(self):
        seen = []
        original = data.read_lowdim
        def tracked(path):
            seen.append(Path(path).name)
            self.assertIn(Path(path).name, self.manifest["splits"]["train"])
            return original(path)
        with patch.object(data, "read_lowdim", side_effect=tracked):
            paths = data.fit_train_statistics(self.manifest, self.cache, [1, 3], [1, 3], self.root / "stats")
        self.assertEqual(seen, self.manifest["splits"]["train"])
        stats = json.loads(Path(paths["normalization"]).read_text())
        temporal = json.loads(Path(paths["temporal_stats"]).read_text())
        values = np.concatenate([original(self.hdf / name)[0] for name in seen]).astype(np.float64)
        np.testing.assert_allclose(stats["robotwin"]["state"]["std"], values.std(0), rtol=1e-12)
        self.assertEqual(stats["action_repr"], "absolute")
        self.assertEqual(temporal["views"], data.MODEL_VIEWS)
        self.assertEqual(np.shape(temporal["proprio_diff_std"]), (3, 14))
        self.assertEqual(np.shape(temporal["memory_diff_std"]), (3, 3, 2048))
        self.assertEqual(temporal["provenance"]["fit_split"], "train")
        self.assertEqual(temporal["provenance"]["episode_count"], 3)
        self.assertEqual(temporal["provenance"]["frame_count"], 12)
        expected = np.array([0, 1, 1, 1], np.float64).std()
        np.testing.assert_allclose(np.array(temporal["memory_diff_std"])[:, 0, 0], expected * np.arange(1, 4))
        # Fitting does not require any val/heldout payloads and never calls image decoder.
        with patch.object(data._codec(), "decode_image_bit", side_effect=AssertionError("image decoded")):
            data.fit_train_statistics(self.manifest, self.cache, [1, 3], [1, 3], self.root / "stats")

    def test_finite_and_std_floor(self):
        m = data.RunningMoments()
        m.update(np.ones((2, 14)))
        m.update(np.ones((3, 14)))
        self.assertTrue(np.all(np.array(m.result()["std"]) == 1e-3))
        with self.assertRaises(ValueError):
            m.update([[np.nan] * 14])
        ep = next(iter(self.identities.values()))
        pool_path = data.episode_directory(self.cache, ep) / "pool.npy"
        pool = np.load(pool_path, mmap_mode="r+")
        old = float(pool[0, 0, 0])
        try:
            pool[0, 0, 0] = np.nan
            pool.flush()
            with self.assertRaises(ValueError):
                self.dataset()[0]
            with self.assertRaises(ValueError):
                data.fit_train_statistics(self.manifest, self.cache, [1], [1], self.root / "bad-stats")
        finally:
            pool[0, 0, 0] = old
            pool.flush()
        del pool

    def test_exact_disk_plan_reuse_and_explicit_heldout(self):
        _, plan = cli.build_plan(self.manifest, self.cache.parent, splits=["train"])
        files = [self.cache / "cache_config.json"]
        for ep in self.identities.values():
            files.extend(data.episode_directory(self.cache, ep).glob("*"))
        self.assertEqual(plan["final_logical_bytes"], sum(f.stat().st_size for f in files))
        self.assertEqual(plan["reused_episodes"], 3)
        self.assertEqual(plan["rebuild_logical_bytes"], 0)
        _, default = cli.build_plan(self.manifest, self.cache.parent)
        self.assertEqual(default["episodes"], 4)
        heldout = self.manifest["splits"]["heldout"][0]
        self.assertNotIn(heldout, [e["identity"]["episode"] for e in default["entries"]])
        with self.assertRaises(ValueError):
            cli.build_plan(self.manifest, self.cache.parent, episodes=[heldout])
        _, explicit = cli.build_plan(self.manifest, self.cache.parent, splits=["heldout"])
        self.assertEqual(explicit["episodes"], 1)
        with patch.object(cli, "load_upstream", side_effect=AssertionError("encoder loaded")), \
             patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("GPU queried")), \
             patch.object(sys, "argv", ["cache", "--split-manifest", str(self.manifest_path), "--out", str(self.cache.parent), "--plan-only"]), \
             patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(cli.main(), 0)

    def test_preprocess_and_pool_exclude_registers(self):
        rgb = np.full((8, 10, 3), (230, 50, 10), np.uint8)
        expected = normalize_image(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA))
        np.testing.assert_array_equal(data.preprocess_rgb(rgb), expected)
        hidden = torch.zeros(2, 201, 1024, dtype=torch.bfloat16)
        hidden[:, 0] = 2
        hidden[:, 1:5] = 100
        hidden[:, 5:] = 3
        pool = pool_from_hidden(hidden)
        self.assertEqual(tuple(pool.shape), (2, 2, 1024))
        self.assertEqual(pool.dtype, torch.float32)
        self.assertTrue(torch.all(pool[:, 0] == 2) and torch.all(pool[:, 1] == 3))
        self.assertEqual(list(data.MODEL_TO_CAMERA.values()), list(data.POLICY_VIEWS))

    def test_streaming_encoder_pipeline_on_cpu_stub(self):
        class Encoder:
            def __call__(self, pixel_values, **kwargs):
                self_check = tuple(pixel_values.shape[1:]) == (3, 224, 224)
                if not self_check or pixel_values.dtype != torch.bfloat16:
                    raise AssertionError("preprocess contract")
                h = torch.ones(len(pixel_values), 201, 1024, dtype=torch.bfloat16)
                return type("Output", (), {"hidden_states": [h] * 25})()
        name = self.manifest["splits"]["val"][0]
        ep = data.episode_identity(self.hdf / name, self.config_id)
        with patch.object(cli, "RoboTwinEpisode", side_effect=lambda p: RoboTwinEpisode(p, image_shape=SHAPE)):
            cli.encode_episode(Encoder(), self.hdf / name, self.root / "stub", ep, batch=2, device="cpu")
        arrays = data.validate_payload(self.root / "stub", ep)
        self.assertTrue(np.all(arrays["pool"] == 1))
        self.assertTrue(np.all(arrays["feat"] == 0x3f80))

    def test_bad_lags_indices_and_default_future(self):
        ds = self.dataset()
        self.assertFalse(ds[0]["future_valid_mask"])
        for item in (-1, len(ds)):
            with self.assertRaises(IndexError):
                ds[item]
        for lags in ([0], [1, 1], [1.5]):
            with self.assertRaises((ValueError, TypeError)):
                data.RobotwinCachedDataset(self.manifest, "train", self.cache, lags, [1])


if __name__ == "__main__":
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CachedDataTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    data.atomic_json(OUT / "tests.json", {"tests_run": result.testsRun, "failures": len(result.failures),
                                         "errors": len(result.errors), "success": result.wasSuccessful(),
                                         "synthetic_only": True, "gpu_jobs": 0})
    raise SystemExit(0 if result.wasSuccessful() else 1)
