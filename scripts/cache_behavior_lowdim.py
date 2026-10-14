#!/usr/bin/env python3
"""Cache per-episode 23D action / 61D state arrays as float32 .npy for fast training access.

Reads the fixed parquet files through the validated BehaviorData metadata, verifies frame
indices are contiguous per episode, writes data/lowdim_cache/episode_<id>_{action,state}.npy
plus an index JSON with lengths, split and SHA-256 of each array.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.behavior_data import BehaviorData, _matrix  # noqa: E402

from harness import task_profile  # noqa: E402
OUT = task_profile.LOWDIM


def main():
    data = BehaviorData(split="train")
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"task_id": data.task_id, "revision": data.provenance["revision"], "split_sha256": data.split_sha256, "episodes": {}}
    for path, episodes in sorted(data.files.items()):
        table = pq.ParquetFile(path).read(columns=["action", "observation.state", "episode_index", "frame_index"], use_threads=False)
        ep_col = table["episode_index"].to_numpy()
        for row in episodes:
            ep = int(row["episode_index"])
            mask = ep_col == ep
            sub = table.filter(mask).combine_chunks()
            frames = sub["frame_index"].to_numpy()
            assert np.array_equal(frames, np.arange(row["length"])), f"episode {ep} rows not contiguous"
            action = _matrix(sub["action"].chunk(0), 23).astype(np.float32)
            state = _matrix(sub["observation.state"].chunk(0), 61).astype(np.float32)
            a_path, s_path = OUT / f"episode_{ep}_action.npy", OUT / f"episode_{ep}_state.npy"
            np.save(a_path, action); np.save(s_path, state)
            index["episodes"][str(ep)] = {
                "length": int(row["length"]),
                "split": "train" if ep in data.splits["train"] else "validation",
                "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
                "state_sha256": hashlib.sha256(state.tobytes()).hexdigest()}
            print(ep, row["length"], flush=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print("DONE", len(index["episodes"]), "episodes")


if __name__ == "__main__":
    main()
