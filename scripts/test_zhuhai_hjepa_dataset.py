#!/usr/bin/env python3
"""CPU-only mocked contract tests; no GPU, model loading, or archive writes."""
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from harness.robotwin_training_data import FEAT_SHAPE
from harness.zhuhai_dataset import FrameFeatureStore, HybridTaskDataset
from harness.zhuhai_hjepa_dataset import (
    HybridHjepaTaskDataset, IDM_STATISTICS_SCHEMA, fit_idm_statistics,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class Episode:
    def __init__(self, name="one", n=10, *, marker=1, terminal=True, split="train",
                 state=None, action=None):
        self.n, self.marker = n, marker
        grid = np.arange(n, dtype=np.float32)[:, None] + np.arange(16, dtype=np.float32)[None] / 20
        self.state = (grid * np.float32(.25) + marker) if state is None else state
        self.action = (grid * np.float32(2) - marker) if action is None else action
        self.path = Path("/cpu-only-hjepa-fixtures") / (name + ".hdf5")
        self.has_terminal = terminal
        self.observation_count = n + int(terminal)
        self.sha256 = hashlib.sha256(name.encode() + self.state.tobytes() + self.action.tobytes()).hexdigest()
        self.row = dict(path=str(self.path), sha256=self.sha256, split=split,
                        family_id="family:" + name, seed=marker, domain="source")

    def __len__(self):
        return self.n


def features(episode, frame):
    # Distinct, finite bf16 bit patterns per frame/view/layer/token, no model.
    view = np.arange(3, dtype=np.uint16)[:, None, None, None] * 256
    layer = np.arange(3, dtype=np.uint16)[None, :, None, None] * 64
    token = np.arange(201, dtype=np.uint16)[None, None, :, None]
    bits = np.uint16(0x3d00 + episode.marker * 7 + frame * 17) + view + layer + token
    return np.broadcast_to(bits, FEAT_SHAPE)


class MockStore:
    def __init__(self):
        self.calls = []

    def get(self, episode, frame):
        if not 0 <= frame < episode.observation_count:
            raise AssertionError("Read outside this episode")
        if frame == len(episode) and not episode.has_terminal:
            raise AssertionError("Read invented terminal")
        self.calls.append((str(episode.path), int(frame)))
        return features(episode, frame).copy(), np.full((3, 2048), frame, np.float32)


class ForbiddenStore:
    def get(self, *args):
        raise AssertionError("Statistics must never read features")


def dataset(episodes, cls=HybridHjepaTaskDataset, *, task="task1", store=None, **kwargs):
    kwargs.setdefault("condition", np.ones(1024, np.float32))
    with patch("harness.zhuhai_dataset.SourceEpisode", side_effect=lambda row: row["fixture"]):
        return cls(task, [{"fixture": ep} for ep in episodes],
                   MockStore() if store is None else store, **kwargs)


def stats_dataset(episodes, task="task1"):
    return dataset(episodes, task=task, store=ForbiddenStore(), condition=None)


def expected_labels(episode):
    last = min(episode.observation_count - 1,
               len(episode) if episode.has_terminal else len(episode) - 1)
    windows = max(0, min(len(episode), last) - 3)
    return [episode.action[t:t + 4].astype(np.float32)
            - episode.state[t].astype(np.float32) for t in range(windows)]


class DatasetTests(unittest.TestCase):
    def assert_base_equal(self, actual, base):
        for key, value in base.items():
            if isinstance(value, (np.ndarray, np.generic)):
                np.testing.assert_array_equal(actual[key], value, err_msg=key)
                self.assertEqual(actual[key].dtype, value.dtype, key)
            else:
                self.assertEqual(actual[key], value, key)

    def test_default_base_compatibility_and_original_kwargs(self):
        episodes = [Episode("first", 5), Episode("second", 6, marker=2)]
        options = dict(chunk=5, future_offset=2, proprio_lags=(1, 3), memory_lags=(1,))
        base = dataset(episodes, HybridTaskDataset, **options)
        candidate = dataset(episodes, **options)
        self.assertEqual(len(candidate), len(base))
        self.assertEqual(candidate.ends, base.ends)
        for index in range(len(base)):
            old, new = base[index], candidate[index]
            self.assertEqual(set(old), set(new))
            self.assert_base_equal(new, old)
        self.assertEqual(candidate.store.calls, base.store.calls)

    def test_each_switch_and_unchanged_32_step_terminal_target(self):
        ep = Episode(n=40)
        base = dataset([ep], HybridTaskDataset)
        original = base[8]  # t+32 == T is a real terminal frame.
        self.assertTrue(original["future_valid_mask"])
        for multi, idm in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(multi=multi, idm=idm):
                ds = dataset([ep], multi_horizon=multi, use_idm=idm)
                sample = ds[8]
                extra = ({"future_feats_8", "future_valid_mask_8"} if multi else set())
                extra |= ({"idm_future_feats", "idm_target_raw", "idm_valid_mask"} if idm else set())
                self.assertEqual(set(sample), set(original) | extra)
                self.assert_base_equal(sample, original)
                extra_frames = ([16] if multi else []) + ([9, 10, 11, 12] if idm else [])
                self.assertEqual(ds.store.calls, base.store.calls +
                                 [(str(ep.path), frame) for frame in extra_frames])
                if idm:
                    self.assertEqual(sample["idm_future_feats"].shape, (4, 3, 3, 201, 1024))
                    self.assertEqual(sample["idm_future_feats"].dtype, np.uint16)
                    self.assertEqual(sample["idm_target_raw"].dtype, np.float32)
                    self.assertEqual(np.shape(sample["idm_valid_mask"]), ())

    def test_horizon32_real_terminal_tail_masks_and_no_episode_wrap(self):
        first, second = Episode("long-first", 40), Episode("long-second", 40, marker=2)
        ds = dataset([first, second], multi_horizon=True, use_idm=True)
        terminal = ds[8]
        self.assertTrue(terminal["future_valid_mask"])
        np.testing.assert_array_equal(terminal["future_feats"], features(first, 40)[:, -1, 5:])
        self.assertFalse(ds[9]["future_valid_mask"])
        ds.store.calls.clear()
        tail = ds[39]
        for mask, target in (("future_valid_mask", "future_feats"),
                             ("future_valid_mask_8", "future_feats_8"),
                             ("idm_valid_mask", "idm_future_feats")):
            self.assertFalse(tail[mask])
            self.assertFalse(tail[target].any())
        self.assertEqual(tail["action_valid_mask"].sum(), 1)
        self.assertFalse(tail["action_sequence"][1:].any())
        self.assertTrue(all(path == str(first.path) and frame <= 39 for path, frame in ds.store.calls))
        self.assertEqual((ds[40]["episode"], ds[40]["anchor"]), (second.path.name, 0))

    def test_horizon8_terminal_is_valid_but_next_anchor_is_not_clamped(self):
        ep = Episode(n=10)
        ds = dataset([ep], multi_horizon=True)
        sample = ds[2]
        self.assertTrue(sample["future_valid_mask_8"])
        self.assertEqual(sample["future_feats_8"].shape, (3, 196, 1024))
        self.assertEqual(sample["future_feats_8"].dtype, np.uint16)
        np.testing.assert_array_equal(sample["future_feats_8"], features(ep, 10)[:, -1, 5:])
        ds.store.calls.clear()
        invalid = ds[3]
        self.assertFalse(invalid["future_valid_mask_8"])
        self.assertFalse(invalid["future_feats_8"].any())
        self.assertTrue(all(frame <= 3 for _, frame in ds.store.calls))
        self.assertEqual(np.shape(invalid["future_valid_mask_8"]), ())

    def test_terminal_flag_is_required_not_just_observation_count(self):
        for n, flag in ((8, "multi_horizon"), (4, "use_idm")):
            with self.subTest(n=n):
                ep = Episode(n=n, terminal=False)
                ep.observation_count = n + 1  # Deliberately inconsistent mock.
                ds = dataset([ep], **{flag: True})
                sample = ds[0]
                key = "future_valid_mask_8" if flag == "multi_horizon" else "idm_valid_mask"
                self.assertFalse(sample[key])
                self.assertEqual(ds.store.calls, [(str(ep.path), 0)])

    def test_idm_terminal_four_steps_fixed_q_and_feature_order(self):
        ep = Episode(n=7)
        ds = dataset([ep], use_idm=True)
        sample = ds[3]
        self.assertTrue(sample["idm_valid_mask"])
        self.assertEqual(sample["idm_target_raw"].shape, (4, 16))
        expected = ep.action[3:7] - ep.state[3]
        np.testing.assert_array_equal(sample["idm_target_raw"], expected)
        self.assertFalse(np.array_equal(expected, ep.action[3:7] - ep.state[3:7]))
        for k, frame in enumerate(range(4, 8)):
            np.testing.assert_array_equal(sample["idm_future_feats"][k], features(ep, frame))
        self.assertEqual(ds.store.calls[-4:], [(str(ep.path), f) for f in range(4, 8)])

    def test_incomplete_idm_reads_no_extra_frames_and_returns_all_zeros(self):
        ep = Episode(n=7)
        for anchor in (4, 5, 6):
            with self.subTest(anchor=anchor):
                base = dataset([ep], HybridTaskDataset)
                old = base[anchor]
                ds = dataset([ep], multi_horizon=True, use_idm=True)
                sample = ds[anchor]
                self.assert_base_equal(sample, old)
                self.assertEqual(ds.store.calls, base.store.calls)
                self.assertFalse(sample["idm_valid_mask"])
                self.assertFalse(sample["idm_future_feats"].any())
                self.assertFalse(sample["idm_target_raw"].any())

    def test_no_episode_boundary_crossing_or_sampler_reindexing(self):
        first, second = Episode("train-a", 4), Episode("train-b", 5, marker=2)
        ds = dataset([first, second], multi_horizon=True, use_idm=True)
        self.assertEqual(len(ds), 9)
        self.assertTrue(ds[0]["idm_valid_mask"])
        ds.store.calls.clear()
        self.assertFalse(ds[1]["idm_valid_mask"])
        self.assertTrue(all(path == str(first.path) for path, _ in ds.store.calls))
        sample = ds[4]
        self.assertEqual((sample["episode"], sample["anchor"]), (second.path.name, 0))
        np.testing.assert_array_equal(sample["idm_target_raw"], second.action[:4] - second.state[0])

    def test_future_aux_never_reads_other_split_neighbor(self):
        for split in ("val", "heldout"):
            with self.subTest(split=split):
                first = Episode("train-short", 3)
                other = Episode("other-split", 10, split=split, marker=2)
                ds = dataset([first, other], multi_horizon=True, use_idm=True)
                sample = ds[0]
                self.assertFalse(sample["future_valid_mask_8"])
                self.assertFalse(sample["idm_valid_mask"])
                self.assertEqual(ds.store.calls, [(str(first.path), 0)])

    def test_no_future_proprio_read_or_returned_key(self):
        ep = Episode(n=10)
        original_state, reads, anchor = ep.state, [], 2

        class PastOnlyState:
            def __getitem__(self, index):
                frames = np.atleast_1d(index)
                if (frames > anchor).any():
                    raise AssertionError("Future proprio read")
                reads.extend(frames.tolist())
                return original_state[index]

        ep.state = PastOnlyState()
        ds = dataset([ep], use_idm=True, multi_horizon=True)
        sample = ds[anchor]
        self.assertTrue(sample["idm_valid_mask"])
        self.assertTrue(reads and max(reads) == anchor)
        self.assertEqual({key for key in sample if key.startswith("idm_")},
                         {"idm_future_feats", "idm_target_raw", "idm_valid_mask"})
        np.testing.assert_array_equal(sample["state"][0], original_state[anchor])

    def test_cast_before_subtraction_and_no_centering_or_clipping(self):
        state = np.full((4, 16), 2**24 + 1, np.float64)
        action = np.repeat((2**24 + np.arange(1, 5, dtype=np.float64))[:, None], 16, axis=1)
        ep = Episode(n=4, state=state, action=action)
        ds = dataset([ep], use_idm=True)
        expected = action.astype(np.float32) - state[0].astype(np.float32)
        sample = ds[0]
        np.testing.assert_array_equal(sample["idm_target_raw"], expected)
        self.assertFalse(np.array_equal(expected, (action - state[0]).astype(np.float32)))
        result = fit_idm_statistics([ds])
        np.testing.assert_allclose(result["std16"], expected.astype(np.float64).std(axis=0))
        np.testing.assert_array_equal(ds[0]["idm_target_raw"], expected)
        self.assertFalse(result["center"])
        self.assertFalse(result["clip"])
        ep2 = Episode("negative", 4, state=np.full((4, 16), 30, np.float32),
                      action=np.full((4, 16), -9, np.float32))
        np.testing.assert_array_equal(dataset([ep2], use_idm=True)[0]["idm_target_raw"],
                                      np.full((4, 16), -39, np.float32))

    def test_real_frame_feature_store_bf16_bit_roundtrip(self):
        calls = []

        class MemoryOnlyCache:
            def read_bytes(self, identity, producer):
                calls.append(dict(identity))
                return producer()

        def producer(ep, frame):
            buffer = BytesIO()
            np.savez(buffer, feat=features(ep, frame), pool=np.zeros((3, 2048), np.float32))
            return buffer.getvalue()

        ep = Episode(n=4)
        store = FrameFeatureStore(MemoryOnlyCache(), producer, encoder_sha256="e" * 64)
        sample = dataset([ep], store=store, use_idm=True)[0]
        bits = sample["idm_future_feats"]
        tensor = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
        np.testing.assert_array_equal(tensor.view(torch.int16).numpy().view(np.uint16), bits)
        self.assertEqual([entry["frame_index"] for entry in calls], [0, 1, 2, 3, 4])
        self.assertTrue(all(entry["source_sha256"] == ep.sha256 for entry in calls))
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_collation_scalar_masks_and_bits(self):
        ds = dataset([Episode(n=5)], use_idm=True, multi_horizon=True)
        batch = torch.utils.data.default_collate([ds[0], ds[2]])
        self.assertEqual(batch["idm_future_feats"].shape, (2, 4, 3, 3, 201, 1024))
        self.assertEqual(batch["idm_future_feats"].dtype, torch.uint16)
        self.assertEqual(batch["idm_target_raw"].dtype, torch.float32)
        self.assertEqual(batch["idm_valid_mask"].shape, (2,))
        self.assertEqual(batch["idm_valid_mask"].tolist(), [True, False])
        self.assertEqual(batch["state"][:, 0].shape, (2, 16))
        self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_index_and_missing_condition_keep_base_errors(self):
        ds = dataset([Episode(n=4)], use_idm=True)
        for index in (-1, 4):
            with self.assertRaises(IndexError):
                ds[index]
        self.assertFalse(ds.store.calls)
        ds.condition = None
        with self.assertRaisesRegex(ValueError, "prototype"):
            ds[0]
        self.assertFalse(ds.store.calls)

    def test_switches_require_booleans(self):
        for name in ("multi_horizon", "use_idm"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                dataset([Episode(n=4)], **{name: "False"})
        self.assertTrue(dataset([Episode(n=4)], use_idm=np.bool_(True)).use_idm)


class StatisticsTests(unittest.TestCase):
    def test_all_valid_windows_fp64_population_moments_and_hashes(self):
        eps = [Episode("alpha", 6), Episode("beta", 5, marker=2)]
        datasets = [stats_dataset([eps[0]], "task1"), stats_dataset([eps[1]], "task2")]
        result = fit_idm_statistics(datasets)
        labels = np.concatenate([block for ep in eps for block in expected_labels(ep)]).astype(np.float64)
        actions = np.concatenate([ep.action.astype(np.float32) for ep in eps]).astype(np.float64)
        action_range, std = np.ptp(actions, axis=0), labels.std(axis=0)
        np.testing.assert_allclose(result["std16"], std, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(result["action_range16"], action_range)
        np.testing.assert_allclose(result["scale16"], np.maximum(np.maximum(std, .01 * action_range), 1e-6))
        self.assertEqual(result["schema"], IDM_STATISTICS_SCHEMA)
        self.assertEqual(result["fit_split"], "train")
        self.assertFalse(result["heldout_used"])
        self.assertEqual(result["valid_windows"], 5)
        self.assertEqual(result["valid_label_vectors"], 20)
        self.assertEqual(result["train_sources_sha256"], digest(result["train_sources"]))
        self.assertEqual(result["statistics_sha256"], digest({k: v for k, v in result.items()
                                                              if k != "statistics_sha256"}))
        self.assertTrue(all(ds.condition is None for ds in datasets))
        json.dumps(result, allow_nan=False)

    def test_sequence_mapping_and_order_independence(self):
        first, second = Episode("a", 5), Episode("b", 6, marker=2)
        ds = stats_dataset([first, second])
        reference = fit_idm_statistics([ds])
        self.assertEqual(reference, fit_idm_statistics({"task1": ds}))
        self.assertEqual(reference, fit_idm_statistics([stats_dataset([second, first])]))
        a, b = stats_dataset([first], "a"), stats_dataset([second], "b")
        self.assertEqual(fit_idm_statistics([a, b]), fit_idm_statistics([b, a]))

    def test_preflight_rejects_nontrain_before_any_numeric_read(self):
        class UnreadableEpisode:
            def __init__(self, split):
                self.row = {"split": split}

            @property
            def state(self):
                raise AssertionError("Numeric data read before complete split preflight")

            @property
            def action(self):
                raise AssertionError("Numeric data read before complete split preflight")

            def __len__(self):
                raise AssertionError("Even len must wait for split preflight")

        for split in ("val", "heldout", None, "TRAIN"):
            with self.subTest(split=split):
                ds = SimpleNamespace(task="task", episodes=[UnreadableEpisode("train"), UnreadableEpisode(split)])
                with self.assertRaisesRegex(ValueError, "TRAIN"):
                    fit_idm_statistics([ds])

    def test_no_cross_episode_windows_and_no_valid_data_failure(self):
        short = [Episode("s1", 3), Episode("s2", 3, marker=2)]
        with self.assertRaisesRegex(ValueError, "No valid"):
            fit_idm_statistics([stats_dataset(short)])
        complete = Episode("complete", 4)
        result = fit_idm_statistics([stats_dataset(short + [complete])])
        self.assertEqual(result["valid_windows"], 1)
        self.assertEqual(result["valid_label_vectors"], 4)

    def test_real_terminal_controls_last_valid_window(self):
        for terminal, expected in ((True, 3), (False, 2)):
            with self.subTest(terminal=terminal):
                result = fit_idm_statistics([stats_dataset([Episode(n=6, terminal=terminal)])])
                self.assertEqual(result["valid_windows"], expected)
                self.assertEqual(result["valid_label_vectors"], 4 * expected)
        with self.assertRaisesRegex(ValueError, "No valid"):
            fit_idm_statistics([stats_dataset([Episode(n=4, terminal=False)])])

    def test_std_is_not_rms_and_absolute_floor_does_not_center_labels(self):
        ep = Episode(n=4, state=np.full((4, 16), 4, np.float32),
                     action=np.full((4, 16), 5, np.float32))
        result = fit_idm_statistics([stats_dataset([ep])])
        np.testing.assert_array_equal(result["std16"], np.zeros(16))
        np.testing.assert_array_equal(result["scale16"], np.full(16, 1e-6))
        np.testing.assert_array_equal(dataset([ep], use_idm=True)[0]["idm_target_raw"], np.ones((4, 16)))

    def test_range_floor_uses_all_train_commands_even_short_episodes(self):
        valid = Episode("valid", 4, state=np.full((4, 16), 4, np.float32),
                        action=np.full((4, 16), 5, np.float32))
        short = Episode("short", 2, marker=2, action=np.repeat(np.array([[0], [200]], np.float32), 16, axis=1))
        result = fit_idm_statistics([stats_dataset([valid, short])])
        np.testing.assert_array_equal(result["std16"], np.zeros(16))
        np.testing.assert_array_equal(result["action_range16"], np.full(16, 200))
        np.testing.assert_array_equal(result["scale16"], np.full(16, 2))
        self.assertEqual(result["valid_windows"], 1)

    def test_moments_stable_across_processing_blocks(self):
        n = 4120
        actions = np.repeat((np.float32(1e8) + (np.arange(n) % 2) * np.float32(16))[:, None], 16, axis=1)
        ep = Episode(n=n, state=np.zeros((n, 16), np.float32), action=actions)
        result = fit_idm_statistics([stats_dataset([ep])])
        labels = np.concatenate(expected_labels(ep)).astype(np.float64)
        np.testing.assert_allclose(result["std16"], labels.std(axis=0), rtol=1e-10, atol=1e-10)
        self.assertEqual(result["valid_windows"], n - 3)

    def test_rejects_duplicate_content_paths_and_admission_mismatch(self):
        ep = Episode(n=4)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            fit_idm_statistics([stats_dataset([ep, ep])])
        for key in ("sha256", "data_sha256"):
            with self.subTest(key=key):
                altered = Episode(n=4)
                altered.row[key] = "0" * 64
                with self.assertRaisesRegex(ValueError, "hash"):
                    fit_idm_statistics([stats_dataset([altered])])
        altered = Episode(n=4)
        altered.row["path"] = "/different/source.hdf5"
        with self.assertRaisesRegex(ValueError, "path"):
            fit_idm_statistics([stats_dataset([altered])])

    def test_provenance_binds_actual_numeric_arrays_and_source_metadata(self):
        ep = Episode(n=5)
        before = fit_idm_statistics([stats_dataset([ep])])
        ep.state = ep.state.copy()
        ep.state[0, 0] += 1
        after = fit_idm_statistics([stats_dataset([ep])])
        self.assertNotEqual(before["train_sources_sha256"], after["train_sources_sha256"])
        self.assertNotEqual(before["statistics_sha256"], after["statistics_sha256"])
        self.assertNotEqual(before["train_sources"][0]["state_sha256"],
                            after["train_sources"][0]["state_sha256"])
        ep.row["family_id"] = "different-family"
        metadata = fit_idm_statistics([stats_dataset([ep])])
        self.assertNotEqual(after["train_sources_sha256"], metadata["train_sources_sha256"])

    def test_finite_aligned_native16_required(self):
        cases = [np.zeros((4, 14), np.float32), np.full((4, 16), np.nan),
                 np.full((4, 16), 1e100), np.zeros((3, 16), np.float32)]
        for state in cases:
            with self.subTest(shape=state.shape):
                ep = Episode(n=4, state=state)
                with self.assertRaises(ValueError):
                    fit_idm_statistics([stats_dataset([ep])])

    def test_empty_and_ambiguous_interfaces_rejected(self):
        for value in ([], {}):
            with self.assertRaisesRegex(ValueError, "No TRAIN"):
                fit_idm_statistics(value)
        ds = stats_dataset([Episode(n=4)])
        for value in (ds, "train", [{"split": "train"}], {"train": {"task": ds}}):
            with self.subTest(kind=type(value)), self.assertRaises(TypeError):
                fit_idm_statistics(value)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
