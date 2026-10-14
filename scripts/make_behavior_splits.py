#!/usr/bin/env python3
"""Create a deterministic train/validation split for one BEHAVIOR task.

The official simulator evaluation instances stay outside this split and are
the only source of final closed-loop test scores.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=13)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/behavior-1k/2026-challenge-demos"),
    )
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk = f"chunk-{args.task_id:03d}"
    episode_meta = args.dataset_root / "meta" / "episodes" / chunk / "file-000.parquet"
    if not episode_meta.exists():
        raise SystemExit(f"missing episode metadata: {episode_meta}")

    table = pq.read_table(episode_meta)
    if "episode_index" not in table.column_names:
        raise SystemExit(f"episode_index not present; columns={table.column_names}")
    episode_ids = sorted(set(table["episode_index"].to_pylist()))
    if args.val_count <= 0 or args.val_count >= len(episode_ids):
        raise SystemExit("--val-count must leave at least one training episode")

    shuffled = episode_ids[:]
    random.Random(args.seed).shuffle(shuffled)
    val = sorted(shuffled[: args.val_count])
    train = sorted(shuffled[args.val_count :])

    split = {
        "task_id": args.task_id,
        "chunk": chunk,
        "seed": args.seed,
        "train_episode_indices": train,
        "validation_episode_indices": val,
        "closed_loop_test": {
            "benchmark": "BEHAVIOR-1K 2026 challenge",
            "public_report_instance_indices": list(range(10)),
            "development_instance_indices": list(range(10, 20)),
            "note": "Do not use simulator evaluation instances for gradient updates.",
        },
    }
    output = args.dataset_root / f"split_task_{args.task_id:03d}.json"
    output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}: train={len(train)}, validation={len(val)}")


if __name__ == "__main__":
    main()

