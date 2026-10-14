"""CPU contract tests; optionally freeze the real 40/5/5 split and write JSON evidence.

Run with behavior6 Python, -B, and --data-root <episode directory> --output <directory>.
No training, simulator, GPU, feature cache, or normalization is involved.
"""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import h5py
import numpy as np
from harness.robotwin_dataset import (
    ACTION_DIM, JOINT_FIELDS, POLICY_VIEWS, RoboTwinDataset, RoboTwinEpisode,
    _codec, make_split, validate_split,
)

SHAPE = (8, 10, 3)
REAL_ROOT = None
REAL_AUDIT = {}


def episode_file(root, number=0, mode="decoded", frames=3):
    path = Path(root) / f"episode_{number:07d}.hdf5"
    rgb = np.full((frames, *SHAPE), (230, 50, 10), dtype=np.uint8)
    with h5py.File(path, "w") as file:
        file["data_format_version"] = "v1.0"
        file["additional_info/frequency"] = 15
        file["instructions"] = json.dumps([f"instruction {i}" for i in range(100)])
        offset = 0
        for name, width in JOINT_FIELDS:
            values = (np.arange(frames)[:, None] * 100 +
                      np.arange(offset, offset + width)[None] + number * 1000).astype(np.float32)
            file[f"state/{name}"] = values
            file[f"action/{name}"] = values + 10
            offset += width
        for view in POLICY_VIEWS:
            camera = file.require_group(f"vision/{view}")
            camera["shape"] = SHAPE
            if mode == "decoded":
                camera["colors"] = rgb
            else:
                buffers = [_codec().encode_image_bit(image) if mode.startswith("standard")
                           else cv2.imencode(".jpg", image)[1].tobytes() for image in rgb]
                if mode.endswith("vlen"):
                    ds = camera.create_dataset("colors", (frames,), dtype=h5py.vlen_dtype(np.uint8))
                    for i, buffer in enumerate(buffers):
                        ds[i] = np.frombuffer(buffer, np.uint8)
                elif mode.endswith("padded"):
                    data = np.zeros((frames, max(map(len, buffers)) + 16), np.uint8)
                    for i, buffer in enumerate(buffers):
                        data[i, :len(buffer)] = np.frombuffer(buffer, np.uint8)
                    camera["colors"] = data
                else:
                    camera.create_dataset("colors", data=buffers, dtype=f"S{max(map(len, buffers)) + 16}")
        # Deliberately malformed evaluator camera: policy must never read it.
        file["vision/cam_third_view/colors"] = "evaluation only"
    return path


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="robotwin-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_decoded_rgb_joint_order_no_shift_and_instruction_unit(self):
        episode = RoboTwinEpisode(episode_file(self.root), image_shape=SHAPE)
        sample = episode.chunk(1, 2)
        self.assertEqual(len(episode), 3)
        self.assertEqual(len(sample["instructions"]), 100)
        self.assertEqual(tuple(sample["images"]), POLICY_VIEWS)
        np.testing.assert_array_equal(sample["state"], np.arange(14) + 100)
        np.testing.assert_array_equal(sample["actions"][0], np.arange(14) + 110)
        np.testing.assert_array_equal(sample["actions"][1], np.arange(14) + 210)
        np.testing.assert_array_equal(sample["images"]["cam_head"][0, 0], [230, 50, 10])
        sample["state"][0] = -999
        self.assertEqual(episode.state[1, 0], 100)

    def test_legacy_and_standard_marked_rgb_all_buffer_layouts(self):
        for mode in ("legacy", "standard", "legacy_padded", "standard_padded", "legacy_vlen", "standard_vlen"):
            with self.subTest(mode=mode):
                sample = RoboTwinEpisode(episode_file(self.root, mode=mode), image_shape=SHAPE).chunk(1, 1)
                for image in sample["images"].values():
                    np.testing.assert_allclose(image[0, 0], [230, 50, 10], atol=3)

    def test_padding_uses_only_stored_labels(self):
        episode = RoboTwinEpisode(episode_file(self.root), image_shape=SHAPE)
        sample = episode.chunk(2, 4)
        np.testing.assert_array_equal(sample["action_valid_mask"], [True, False, False, False])
        np.testing.assert_array_equal(sample["actions"][0], np.arange(14) + 210)
        np.testing.assert_array_equal(sample["actions"][1:], np.zeros((3, 14)))
        self.assertEqual(sample["action_valid_mask"].dtype, np.bool_)
        self.assertEqual(sample["actions"].dtype, np.float32)
        self.assertTrue(episode.chunk(0, 3)["action_valid_mask"].all())

    def test_invalid_indices_and_chunk_size(self):
        episode = RoboTwinEpisode(episode_file(self.root), image_shape=SHAPE)
        for frame in (-1, 3):
            with self.assertRaises(IndexError):
                episode.chunk(frame, 1)
        for size in (0, -1):
            with self.assertRaises(ValueError):
                episode.chunk(0, size)
        with self.assertRaises(TypeError):
            episode.chunk(0.5, 1)

    def test_lowdim_wrong_width_nonfinite_missing_and_frame_count(self):
        mutations = [
            ("action/left_arm_joint_states", np.zeros((3, 7))),
            ("state/left_ee_joint_states", np.zeros(3)),
            ("action/right_ee_joint_states", np.full((3, 1), np.nan)),
            ("state/right_arm_joint_states", np.full((3, 6), np.inf)),
            ("action/right_arm_joint_states", np.full((3, 6), 1e100)),
            ("state/right_arm_joint_states", np.zeros((2, 6))),
            ("action/left_ee_joint_states", None),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value_shape=getattr(value, "shape", None)):
                path = episode_file(self.root)
                with h5py.File(path, "a") as file:
                    del file[key]
                    if value is not None:
                        file[key] = value
                with self.assertRaises(ValueError):
                    RoboTwinEpisode(path, image_shape=SHAPE)

    def test_missing_view_bad_rgb_shape_and_frequency(self):
        mutations = [
            ("vision/cam_left_wrist", None),
            ("vision/cam_head/colors", np.zeros((3, 8, 10, 4), np.uint8)),
            ("vision/cam_head/colors", np.zeros((3, *SHAPE), np.float32)),
            ("vision/cam_head/colors", np.zeros((2, *SHAPE), np.uint8)),
            ("vision/cam_head/shape", [8, 11, 3]),
            ("additional_info/frequency", 30),
            ("data_format_version", "v2.0"),
            ("instructions", '["valid", 123]'),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                path = episode_file(self.root)
                with h5py.File(path, "a") as file:
                    del file[key]
                    if value is not None:
                        file[key] = value
                with self.assertRaises(ValueError):
                    RoboTwinEpisode(path, image_shape=SHAPE)

    def test_corrupt_later_image_rejected_at_sampling(self):
        path = episode_file(self.root, mode="standard")
        with h5py.File(path, "a") as file:
            file["vision/cam_right_wrist/colors"][1] = b"not an image"
        episode = RoboTwinEpisode(path, image_shape=SHAPE)
        with self.assertRaises(ValueError):
            episode.chunk(1, 1)

    def test_split_deterministic_disjoint_and_tamper_rejected(self):
        for number in reversed(range(50)):
            (self.root / f"episode_{number:07d}.hdf5").touch()
        manifest = make_split(self.root)
        self.assertEqual(manifest, make_split(self.root))
        self.assertEqual(manifest["counts"], {"train": 40, "val": 5, "heldout": 5})
        names = [name for split in manifest["splits"].values() for name in split]
        self.assertEqual(len(names), len(set(names)))
        self.assertNotEqual(manifest["splits"], make_split(self.root, seed=1)["splits"])
        validate_split(manifest)
        for kind in ("overlap", "missing", "path_escape", "wrong_count", "reassign"):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(manifest)
                if kind == "overlap":
                    bad["splits"]["val"][0] = bad["splits"]["train"][0]
                elif kind == "missing":
                    bad["splits"]["heldout"].pop()
                elif kind == "path_escape":
                    bad["splits"]["train"][0] = "../episode_0000000.hdf5"
                elif kind == "wrong_count":
                    bad["counts"]["train"] = 39
                else:
                    bad["splits"]["val"], bad["splits"]["heldout"] = bad["splits"]["heldout"], bad["splits"]["val"]
                with self.assertRaises(ValueError):
                    validate_split(bad)

    def test_split_aliases_rejected(self):
        first = self.root / "episode_0000000.hdf5"
        first.touch()
        second = self.root / "episode_0000001.hdf5"
        second.symlink_to(first)
        (self.root / "episode_0000002.hdf5").touch()
        with self.assertRaises(ValueError):
            make_split(self.root, counts=(1, 1, 1))
        second.unlink()
        os.link(first, second)
        with self.assertRaises(ValueError):
            make_split(self.root, counts=(1, 1, 1))

    def test_dataset_episode_boundary_and_no_instruction_expansion(self):
        for number in range(4):
            episode_file(self.root, number)
        manifest = make_split(self.root, counts=(2, 1, 1))
        dataset = RoboTwinDataset(manifest, "train", 4, image_shape=SHAPE)
        self.assertEqual(len(dataset), 6)  # Not 600 instruction-expanded frames.
        last, following = dataset[2], dataset[3]
        self.assertNotEqual(last["episode_id"], following["episode_id"])
        np.testing.assert_array_equal(last["action_valid_mask"], [True, False, False, False])
        np.testing.assert_array_equal(last["actions"][1:], 0)
        self.assertEqual(following["frame_index"], 0)
        self.assertEqual(following["episode_id"], Path(manifest["splits"]["train"][1]).stem)
        with self.assertRaises(IndexError):
            dataset[len(dataset)]
        with self.assertRaises(ValueError):
            RoboTwinDataset(manifest, "report", 4, image_shape=SHAPE)


class RealTests(unittest.TestCase):
    def setUp(self):
        if REAL_ROOT is None:
            self.skipTest("Pass --data-root to test the real corpus")

    def test_all_50_lowdim_first_policy_images_and_frozen_split(self):
        manifest = make_split(REAL_ROOT)
        validate_split(manifest)
        records = []
        for name in sorted(name for group in manifest["splits"].values() for name in group):
            episode = RoboTwinEpisode(REAL_ROOT / name)
            self.assertEqual(len(episode.instructions), 100)
            self.assertEqual(episode.actions.shape[1], ACTION_DIM)
            records.append({"episode_id": episode.episode_id, "frames": len(episode),
                            "instruction_variants": len(episode.instructions)})
        REAL_AUDIT["episodes"] = records
        REAL_AUDIT["total_frames"] = sum(record["frames"] for record in records)
        REAL_AUDIT["scope"] = "All lowdim values and first policy RGB per episode; not every image payload"

    def test_first_two_episodes_two_frames_match_official_decoder_and_actions(self):
        checked = []
        for path in sorted(REAL_ROOT.glob("episode_*.hdf5"))[:2]:
            episode = RoboTwinEpisode(path)
            with h5py.File(path, "r") as file:
                for frame in (0, 1):
                    sample = episode.chunk(frame, 3)
                    expected = np.concatenate([file[f"action/{name}"][frame:frame + 3]
                                               for name, _ in JOINT_FIELDS], axis=1)
                    np.testing.assert_array_equal(sample["actions"], expected)
                    for view in POLICY_VIEWS:
                        official = _codec().decode_image_bit(file[f"vision/{view}/colors"][frame])
                        np.testing.assert_array_equal(sample["images"][view], official)
                        self.assertEqual(official.shape, (240, 320, 3))
                    self.assertEqual(tuple(sample["images"]), POLICY_VIEWS)
                final = episode.chunk(len(episode) - 1, 8)
                np.testing.assert_array_equal(final["actions"][0], episode.actions[-1])
                np.testing.assert_array_equal(final["actions"][1:], 0)
                self.assertEqual(int(final["action_valid_mask"].sum()), 1)
            delta = float(np.abs(episode.actions - episode.state).mean())
            self.assertGreater(delta, 0)
            checked.append({"episode_id": episode.episode_id, "frames_checked": [0, 1],
                            "mean_abs_action_state_difference": delta, "final_chunk_valid": 1})
        self.assertEqual(len(checked), 2)
        REAL_AUDIT["official_comparisons"] = checked


def main():
    global REAL_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.data_root) != bool(args.output):
        parser.error("--data-root and --output must be provided together")
    REAL_ROOT = args.data_root.resolve() if args.data_root else None
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), end="")
    summary = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "device": "cpu", "python": sys.executable,
               "numpy_version": np.__version__, "h5py_version": h5py.__version__,
               "policy_views": list(POLICY_VIEWS), "action_dim": ACTION_DIM,
               "training_run": False, "normalization_fitted": False,
               "physical_task_success_claimed": False, "real_audit": REAL_AUDIT}
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if result.wasSuccessful():
            manifest = make_split(REAL_ROOT)
            destination = args.output / "split_manifest.json"
            if destination.exists():
                if json.loads(destination.read_text()) != manifest:
                    raise ValueError("Refusing to overwrite a different frozen split")
            else:
                with destination.open("x") as file:
                    json.dump(manifest, file, indent=2)
                    file.write("\n")
        (args.output / "test_results.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "test_output.txt").write_text(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
