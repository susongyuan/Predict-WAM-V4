#!/usr/bin/env python3
"""Losslessly carry legacy RoboTwin JPEG buffers into the v1 training layout.

Joint commands at t are state; commands at t+1 are the target. No measured
joint state or timestamps are invented. Frequency is explicitly nominal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.robotwin_dataset import decode_rgb, make_split

FIELDS = (("left_arm", "left_arm_joint_states", 6),
          ("left_gripper", "left_ee_joint_states", 1),
          ("right_arm", "right_arm_joint_states", 6),
          ("right_gripper", "right_ee_joint_states", 1))
CAMERAS = (("head_camera", "cam_head"), ("left_camera", "cam_left_wrist"),
           ("right_camera", "cam_right_wrist"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def convert_episode(source, destination, instructions, nominal_frequency):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or destination.exists():
        raise ValueError("Source must be direct; destination must be new")
    if not instructions or not all(isinstance(x, str) and x.strip() for x in instructions):
        raise ValueError("Missing source instructions")
    if nominal_frequency <= 0 or not np.isfinite(nominal_frequency):
        raise ValueError("Nominal frequency must be finite and positive")
    source_sha = sha(source)
    temporary = destination.with_suffix(".partial.hdf5")
    if temporary.exists():
        raise ValueError("Incomplete previous conversion exists")
    with h5py.File(source, "r") as src:
        arrays = {}
        for old, new, width in FIELDS:
            values = src[f"joint_action/{old}"][:]
            if width == 1 and values.ndim == 1:
                values = values[:, None]
            if (values.ndim != 2 or values.shape[1] != width or values.dtype.kind not in "fiu"
                    or not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any()):
                raise ValueError(f"Invalid joint command {old}")
            arrays[new] = values.astype(np.float32)
        lengths = {len(a) for a in arrays.values()}
        if len(lengths) != 1 or min(lengths) < 2:
            raise ValueError("Joint command lengths differ or fewer than two frames")
        length = lengths.pop()
        for old, _ in CAMERAS:
            values = src[f"observation/{old}/rgb"]
            if values.ndim != 1 or values.dtype.kind != "S" or len(values) != length:
                raise ValueError(f"Expected fixed-width encoded camera buffers: {old}")
            decode_rgb(values[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(temporary, "x") as dst:
            text = h5py.string_dtype("utf-8")
            dst.attrs.update(source_format="RoboTwin_legacy", source_path=str(source.resolve()),
                             source_sha256=source_sha,
                             state_semantics="controller_joint_targets_and_gripper_command_not_measured_qpos",
                             time_semantics="observation_index; nominal_hz_not_measured_clock")
            dst.create_dataset("data_format_version", data="v1.0", dtype=text)
            dst.create_dataset("instructions", data=json.dumps(instructions), dtype=text)
            dst.create_group("additional_info").create_dataset("frequency", data=nominal_frequency)
            states, actions, vision = dst.create_group("state"), dst.create_group("action"), dst.create_group("vision")
            for name, values in arrays.items():
                states.create_dataset(name, data=values[:-1])
                actions.create_dataset(name, data=values[1:])
            for old, new in CAMERAS:
                camera = vision.create_group(new)
                camera.create_dataset("colors", data=src[f"observation/{old}/rgb"][:-1])
                camera.create_dataset("shape", data=np.array([240, 320, 3], np.int32))
                # Matrix fields are not consumed by this WAM; preserve their source convention explicitly.
                camera.attrs["extrinsics_not_converted"] = True
    with h5py.File(source, "r") as src, h5py.File(temporary, "r") as dst:
        for old, new in CAMERAS:
            if not np.array_equal(src[f"observation/{old}/rgb"][:-1], dst[f"vision/{new}/colors"][:]):
                raise ValueError("JPEG buffers changed")
        for name, values in arrays.items():
            if not (np.array_equal(dst[f"state/{name}"][:], values[:-1])
                    and np.array_equal(dst[f"action/{name}"][:], values[1:])):
                raise ValueError("State/action alignment changed")
    if sha(source) != source_sha:
        raise ValueError("Source changed during conversion")
    temporary.rename(destination)
    return {"source": str(source.resolve()), "source_sha256": source_sha,
            "path": str(destination.resolve()), "sha256": sha(destination),
            "source_frames": length, "frames": length - 1, "jpeg_buffers_identical": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--nominal-frequency", required=True, type=int)
    args = parser.parse_args()
    raw = json.loads(args.source_receipt.read_text())
    if raw.get("status") != "complete" or raw.get("hdf5_files") != 500 or raw.get("task") != args.task:
        raise ValueError("Expected verified single-task 500-episode source receipt")
    files = list((args.source_root / "data").glob("episode*.hdf5"))
    ids = [int(re.fullmatch(r"episode(\d+)\.hdf5", p.name)[1]) for p in files]
    if len(files) != 500 or set(ids) != set(range(500)):
        raise ValueError("Expected exactly 500 unique source episode IDs")
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "converting", "task": args.task, "repo": raw["repo"],
                "revision": raw["revision"], "raw_source_receipt": str(args.source_receipt.resolve()),
                "raw_source_receipt_sha256": sha(args.source_receipt), "raw_archive_sha256": raw["sha256"],
                "converter_sha256": sha(__file__), "official_alignment_source_sha256": sha(ROOT / "third_party/RoboTwin/envs/utils/pkl2hdf5.py"),
                "nominal_frequency": args.nominal_frequency, "measured_clock_available": False,
                "frequency_basis": "upstream v1 converter nominal default; legacy files contain no measured clock",
                "state_semantics": "controller_joint_targets_and_gripper_command_not_measured_qpos",
                "alignment": "state=legacy[t], action=legacy[t+1], RGB=legacy[t]; omit last observation",
                "images": "original JPEG buffers; no channel swap or re-encoding",
                "omitted": ["external camera", "endpose", "camera matrices", "expert trajectories"],
                "instruction_policy": "source seen instructions only; alternatives remain in their episode",
                "episodes": [], "started_unix": time.time()}
    receipt = args.out / "SOURCE.json"
    try:
        for episode_id, source in sorted(zip(ids, files)):
            instruction_path = args.source_root / "instructions" / f"episode{episode_id}.json"
            instructions = json.loads(instruction_path.read_text())["seen"]
            destination = args.out / "data" / f"episode_{episode_id:07d}.hdf5"
            row = convert_episode(source, destination, instructions, args.nominal_frequency)
            row.update(source_episode_id=episode_id, instruction_sha256=sha(instruction_path))
            metadata["episodes"].append(row)
            receipt.write_text(json.dumps(metadata, indent=2) + "\n")
        split = make_split(args.out / "data", seed=20261004, counts=(400, 50, 50))
        (args.out / "split_manifest.json").write_text(json.dumps(split, indent=2) + "\n")
        metadata.update(status="complete", hdf5_files=500, split_sha256=sha(args.out / "split_manifest.json"),
                        elapsed_seconds=time.time() - metadata["started_unix"])
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
