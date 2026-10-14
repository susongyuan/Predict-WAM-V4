"""Task-dependent data paths, selected by the environment variable P2_TASK_ID (default 13).

Task 13 keeps its original (legacy, unsuffixed) paths byte for byte, so every existing checkpoint, cache and
calibration still resolves exactly as before. Any other task writes to separate `*_task{id:03d}` locations and
never touches Task 13 artifacts. Episode indices of different tasks do not collide (Task 0: 0-199, Task 1: 200-399,
Task 13: 2600-2799), so the per-camera frame cache directory is shared.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/behavior-1k/2026-challenge-demos"
TASK_NAMES = {0: "turning_on_radio", 1: "picking_up_trash", 13: "loading_the_car", 90: "composting_waste"}

TASK_ID = int(os.environ.get("P2_TASK_ID", "13"))
if TASK_ID not in TASK_NAMES:
    raise ValueError(f"P2_TASK_ID={TASK_ID} has no registered task name; known: {sorted(TASK_NAMES)}")
TASK_NAME = TASK_NAMES[TASK_ID]
SCENE_NAME = "Rs_int" if TASK_ID == 90 else "house_double_floor_lower"
CHUNK = f"chunk-{TASK_ID:03d}"
LEGACY = TASK_ID == 13

PROVENANCE = DATA / ("DOWNLOAD_PROVENANCE.json" if LEGACY else f"DOWNLOAD_PROVENANCE_task{TASK_ID:04d}_rgb.json")
SPLIT = DATA / f"split_task_{TASK_ID:03d}.json"
SEGMENTS = DATA / f"skill_segments_task{TASK_ID:03d}.json"
TRAIN_STATS = DATA / f"train_stats_task{TASK_ID:03d}.json"
PROPHIST_STATS = DATA / f"train_stats_task{TASK_ID:03d}_prophist.json"
LOWDIM = ROOT / ("data/lowdim_cache" if LEGACY else f"data/lowdim_cache_task{TASK_ID:03d}")
FRAMES = ROOT / "data/frame_cache"
VTT_DIR = ROOT / f"data/skill_vtt_task{TASK_ID:03d}"
EPISODES_META = DATA / "meta/episodes" / CHUNK
# TRAIN-demo docking calibration read by harness/docking_controller.load_calibration (None: no docking for the task).
DOCK_CALIB = {0: ROOT / "outputs/validation/task0_docking_calibration_20260927",
              1: ROOT / "outputs/validation/task1_docking_calibration_20260926",
              90: ROOT / "outputs/validation/task090_docking_calibration_train_20260930"}.get(TASK_ID)
