import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from harness.robotwin_dataset import _codec, RoboTwinEpisode
from scripts.convert_robotwin_legacy import CAMERAS, FIELDS, convert_episode


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "episode0.hdf5"
        self.output = self.root / "episode_0000000.hdf5"
        with h5py.File(self.source, "x") as file:
            joints = file.create_group("joint_action")
            for old, _, width in FIELDS:
                values = np.arange(3 * width, dtype=np.float64).reshape(3, width) / 10
                joints.create_dataset(old, data=values[:, 0] if width == 1 else values)
            observations = file.create_group("observation")
            image = np.zeros((240, 320, 3), np.uint8)
            image[:, :, 0] = 230
            encoded = _codec().encode_image_bit(image)
            for old, _ in CAMERAS:
                observations.create_group(old).create_dataset("rgb", data=np.array([encoded] * 3))

    def test_exact_temporal_pair_and_rgb_bytes(self):
        result = convert_episode(self.source, self.output, ["Put the object in the drawer"], 15)
        self.assertEqual(result["frames"], 2)
        with h5py.File(self.source) as src, h5py.File(self.output) as dst:
            for old, new, width in FIELDS:
                values = src[f"joint_action/{old}"][:].reshape(3, width).astype(np.float32)
                np.testing.assert_array_equal(dst[f"state/{new}"][:], values[:-1])
                np.testing.assert_array_equal(dst[f"action/{new}"][:], values[1:])
            for old, new in CAMERAS:
                np.testing.assert_array_equal(src[f"observation/{old}/rgb"][:-1], dst[f"vision/{new}/colors"][:])
            self.assertIn("not_measured_qpos", dst.attrs["state_semantics"])
        episode = RoboTwinEpisode(self.output)
        np.testing.assert_array_equal(episode.chunk(1, 32)["action_valid_mask"], np.arange(32) == 0)
        self.assertGreater(episode.chunk(0, 32)["images"]["cam_head"][0, 0, 0], 200)

    def test_refuses_overwrite(self):
        self.output.write_bytes(b"keep")
        with self.assertRaises(ValueError):
            convert_episode(self.source, self.output, ["task"], 15)
        self.assertEqual(self.output.read_bytes(), b"keep")

    def test_refuses_nonfinite_command(self):
        with h5py.File(self.source, "a") as file:
            file["joint_action/left_arm"][1, 0] = np.nan
        with self.assertRaises(ValueError):
            convert_episode(self.source, self.output, ["task"], 15)
        self.assertFalse(self.output.exists())

    def test_refuses_image_length_mismatch(self):
        with h5py.File(self.source, "a") as file:
            del file["observation/head_camera/rgb"]
            file["observation/head_camera"].create_dataset("rgb", data=np.array([b"bad"]))
        with self.assertRaises(ValueError):
            convert_episode(self.source, self.output, ["task"], 15)

    def test_actual_probe(self):
        source = Path(__file__).resolve().parents[1] / "outputs/validation/robotwin_continuous_task_20261004/sample_episode.hdf5"
        self.assertTrue(source.is_file())
        result = convert_episode(source, self.output, ["Open the drawer and put the object inside"], 15)
        episode = RoboTwinEpisode(self.output)
        self.assertEqual(len(episode), result["source_frames"] - 1)
        self.assertTrue(result["jpeg_buffers_identical"])
        with h5py.File(source) as raw:
            np.testing.assert_array_equal(episode.actions[0, :6], raw["joint_action/left_arm"][1].astype(np.float32))
        self.assertEqual(episode.chunk(0, 32)["images"]["cam_head"].shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()
