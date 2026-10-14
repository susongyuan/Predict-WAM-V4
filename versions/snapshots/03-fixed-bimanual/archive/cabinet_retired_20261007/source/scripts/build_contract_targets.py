#!/usr/bin/env python3
"""Build train-only skill/object and navigation-stage VTT targets from existing CPU caches.

No encoder or torch is loaded. Run --self-test for split/stage isolation, --check to
recompute and verify the saved artifact against its source hashes and train means.
Paths follow harness.task_profile (P2_TASK_ID; default 13 = data/skill_vtt_task013 + skill_segments_task013.json,
unchanged); an existing, different contract_targets.json is never overwritten unless --overwrite is given.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import task_profile  # noqa: E402
from harness.vtt_monitor import precondition_key  # noqa: E402

VTT_DIR = task_profile.VTT_DIR          # Task 13: ROOT / "data/skill_vtt_task013"
ANNOTATIONS = task_profile.SEGMENTS     # Task 13: ROOT / "data/behavior-1k/2026-challenge-demos/skill_segments_task013.json"
OUTPUT = VTT_DIR / "contract_targets.json"


def build_targets(annotations, index, vectors, min_train=5):
    vectors = np.asarray(vectors, dtype=np.float64)
    if (type(min_train) is not int or min_train < 1 or vectors.shape != (len(index), 1024)
            or not np.isfinite(vectors).all()):
        raise ValueError("Expected finite N x 1024 cached VTTs and a positive support threshold")
    groups, seen, train_episodes = defaultdict(list), set(), set()
    for i, row in enumerate(index):
        identity = row["episode"], row["segment"]
        info = annotations["episodes"][str(row["episode"])]
        segments = info["segments"]
        segment = segments[row["segment"]]
        if (identity in seen or row["split"] != info["split"] or row["class"] != segment["class"]
                or row["split"] not in ("train", "validation")):
            raise ValueError(f"Duplicate or mismatched cache/annotation segment: {identity}")
        seen.add(identity)
        if row["split"] != "train":
            continue
        train_episodes.add(row["episode"])
        duration = segment["end"] - segment["start"]
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid training segment duration: {identity}")
        key = precondition_key(segment["skill"], segment["objects"])
        groups[key, None].append((i, duration))
        if segment["skill"] == "move to" and row["segment"] + 1 < len(segments):
            following = segments[row["segment"] + 1]
            if following["skill"] != "move to":
                nxt = precondition_key(following["skill"], following["objects"])
                groups[key, nxt].append((i, duration))

    result = {"format": "train-contract-vtt-v1", "dimension": 1024, "source_split": "train",
              "uses_validation_vectors": False, "uses_validation_scalars": False,
              "training_recipe": "Raw mean of cached DINOv3 CLS(end)-CLS(start), training segments only; no normalization or scaling.",
              "duration_recipe": "Median annotation end-start in original control frames, training segments only (not gap-filled bounds).",
              "navigation_key_recipe": "navigation[current_step.key()][immediate_next_manipulation_step.key()]",
              "min_train_segments": min_train, "train_episodes": sorted(train_episodes),
              "contracts": {}, "navigation": {}, "excluded_insufficient_training": [],
              "validation_requirement": "Training-derived targets, not verified episodic memory or evidence of task success."}
    for (key, nxt), records in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        count = len(records)
        if count < min_train:
            excluded = {"kind": "navigation" if nxt else "contract", "key": key, "train_count": count}
            if nxt:
                excluded["next_key"] = nxt
            result["excluded_insufficient_training"].append(excluded)
            continue
        indices, durations = zip(*records)
        vector = vectors[list(indices)].mean(0).astype(np.float32)
        if np.linalg.norm(vector) < 1e-3:
            raise ValueError(f"Degenerate mean transition: {key} -> {nxt}")
        target = {"train_count": count, "count": count, "median_frames": float(np.median(durations)),
                  "vector": vector.tolist()}
        if nxt:
            result["navigation"].setdefault(key, {})[nxt] = target
        else:
            result["contracts"][key] = target
    return result


def self_test():
    annotations, index, vectors = {"episodes": {}}, [], []
    for ep, split in ((1, "train"), (2, "train"), (3, "validation")):
        specifications = [("move to", ["car"], 10, 1), ("open lid", ["car"], 30, 2),
                          ("move to", ["car"], 20, 3), ("place in", ["toy_box", "car"], 40, 4)]
        if ep == 1:
            specifications.append(("push to", ["door", "robot"], 50, 5))  # insufficient train support
        if split == "validation":
            specifications.append(("close lid", ["car"], 9000, 900))  # no train support
        segments = []
        for i, (skill, objects, duration, value) in enumerate(specifications):
            segments.append({"skill": skill, "objects": [objects], "class": i,
                             "start": 0, "end": duration if split == "train" else 9000})
            index.append({"episode": ep, "segment": i, "class": i, "split": split})
            vectors.append(np.full(1024, value if split == "train" else 900, np.float32))
        annotations["episodes"][str(ep)] = {"split": split, "segments": segments}
    result = build_targets(annotations, index, vectors, min_train=2)
    assert result["train_episodes"] == [1, 2]
    assert "push to(door,robot)" not in result["contracts"] and "close lid(car)" not in result["contracts"]
    stages = result["navigation"]["move to(car)"]
    for key, duration, value in (("open lid(car)", 10, 1), ("place in(toy_box,car)", 20, 3)):
        assert stages[key]["count"] == stages[key]["train_count"] == 2
        assert stages[key]["median_frames"] == duration
        np.testing.assert_array_equal(stages[key]["vector"], np.full(1024, value))
    assert result["contracts"]["move to(car)"]["median_frames"] == 15
    # A forged cache split must fail instead of moving a validation vector into the training mean.
    index[-1]["split"] = "train"
    try:
        build_targets(annotations, index, vectors, min_train=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched validation split was accepted")
    print("PASS: train-only vectors/medians, rare/validation-only exclusions, navigation-stage isolation, split integrity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify saved targets by recomputing, without writing")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing, different contract_targets.json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    paths = [ANNOTATIONS, VTT_DIR / "instances_index.json", VTT_DIR / "instances.npy", VTT_DIR / "meta.json"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    annotations, index, meta = (json.loads(path.read_text()) for path in (paths[0], paths[1], paths[3]))
    if (meta["skill_segments_sha256"] != hashes[str(ANNOTATIONS.relative_to(ROOT))]
            or meta["classes"] != annotations["classes"]):
        raise ValueError("Cached VTT provenance does not match the source annotations/classes")
    result = build_targets(annotations, index, np.load(paths[2], allow_pickle=False))
    result["source_sha256"] = hashes
    if args.check:
        if json.loads(OUTPUT.read_text()) != result:
            raise ValueError("Saved targets differ from current source hashes, train means, counts, or medians")
    else:
        text = json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
        if OUTPUT.exists() and OUTPUT.read_text() != text and not args.overwrite:
            raise SystemExit(f"{OUTPUT} exists with different content (checkpoints pin its SHA256); pass --overwrite to replace")
        OUTPUT.write_text(text)
    print(json.dumps({"checked" if args.check else "written": str(OUTPUT.relative_to(ROOT)),
                      "contracts": len(result["contracts"]), "navigation_stages": sum(map(len, result["navigation"].values()))}))


if __name__ == "__main__":
    main()
