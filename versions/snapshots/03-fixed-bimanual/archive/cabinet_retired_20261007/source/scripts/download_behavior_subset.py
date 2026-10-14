#!/usr/bin/env python3
"""Download a task-complete BEHAVIOR-1K 2026 subset from Hugging Face.

The official release is 3.27 TB.  This downloader keeps one task internally
consistent by fetching all low-dimensional parquet files, episode metadata,
skill annotations, and the selected camera streams for that task.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "behavior-1k/2026-challenge-demos"
# Pin the release used by this project.  Do not silently train and evaluate on
# a moving dataset revision.
REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"

RGB_KEYS = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)
DEPTH_KEYS = (
    "observation.depth_linear.zed_link_camera_0",
    "observation.depth_linear.left_realsense_link_camera_0",
    "observation.depth_linear.right_realsense_link_camera_0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=13)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/behavior-1k/2026-challenge-demos"),
    )
    parser.add_argument(
        "--cameras",
        choices=("none", "head", "rgb", "rgbd"),
        default="rgb",
        help="none=no videos (annotations/low-dim only); head=head RGB only; rgb=all three RGB cameras; rgbd=RGB+depth",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.task_id <= 99:
        raise SystemExit("--task-id must be in [0, 99]")

    chunk = f"chunk-{args.task_id:03d}"
    annotation_task = f"task-{args.task_id:04d}"
    patterns = [
        "LICENSE",
        "README.md",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.jsonl",
        "meta/tasks.parquet",
        f"meta/episodes/{chunk}/**",
        f"data/{chunk}/**",
        f"annotations/{annotation_task}/**",
    ]

    camera_keys = () if args.cameras == "none" else (RGB_KEYS[0],) if args.cameras == "head" else RGB_KEYS
    if args.cameras == "rgbd":
        camera_keys += DEPTH_KEYS
    patterns.extend(f"videos/{key}/{chunk}/**" for key in camera_keys)

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=patterns,
        local_dir=args.output,
        max_workers=args.max_workers,
    )

    provenance = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "task_id": args.task_id,
        "chunk": chunk,
        "camera_selection": args.cameras,
        "allow_patterns": patterns,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": str(Path(snapshot_path).resolve()),
    }
    # One provenance file per task so a second task never overwrites the first
    # task's record (Task 13 keeps the original DOWNLOAD_PROVENANCE.json).
    name = "DOWNLOAD_PROVENANCE.json" if args.task_id == 13 else f"DOWNLOAD_PROVENANCE_task{args.task_id:04d}_{args.cameras}.json"
    (args.output / name).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

