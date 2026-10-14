#!/usr/bin/env python3
"""Build the Task 13 skill-segment table from official BEHAVIOR annotations.

Output: data/behavior-1k/2026-challenge-demos/skill_segments_task013.json
  - vocabulary: skill classes (skill_description string -> class index), counts split by train/val
  - episodes: per episode a list of segments {start, end, skill, skill_type, objects, class}
Frames not covered by any segment are assigned to the nearest segment (gap fill), so that
every frame has exactly one skill class. Only train episodes may later contribute to
skill prototypes; validation segments are marked split="validation".
"""
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.behavior_data import BehaviorData  # noqa: E402


def main():
    data = BehaviorData(split="train")
    split = {"train": data.splits["train"], "validation": data.splits["validation"]}
    ann_dir = data.root / "annotations" / f"task-{data.task_id:04d}"
    vocab = Counter(); vocab_split = {"train": Counter(), "validation": Counter()}
    episodes = {}
    for ep, row in sorted(data.episodes.items()):
        path = data.root / row["annotation_path"] if (data.root / row["annotation_path"]).exists() else ann_dir / Path(row["annotation_path"]).name
        ann = json.loads(path.read_text())
        length = int(row["length"])
        sp = "train" if ep in split["train"] else "validation"
        expanded = []
        for s in ann["skill_annotation"]:
            fd = s["frame_duration"]
            intervals = fd if fd and isinstance(fd[0], list) else [fd]  # one segment may list several intervals
            for iv in intervals:
                expanded.append({**s, "frame_duration": [int(iv[0]), int(iv[1])]})
        raw = sorted(expanded, key=lambda s: s["frame_duration"][0])
        segs = []
        for s in raw:
            a, b = int(s["frame_duration"][0]), int(s["frame_duration"][1])
            a, b = max(0, a), min(length, b)
            if b <= a:
                continue
            desc = " | ".join(s["skill_description"])
            stype = " | ".join(s["skill_type"])
            segs.append({"start": a, "end": b, "skill": desc, "skill_type": stype,
                         "objects": s["object_id"], "manipulating": s["manipulating_object_id"],
                         "memory_prefix": s.get("memory_prefix", []), "skill_idx": s["skill_idx"]})
            vocab[desc] += 1; vocab_split[sp][desc] += 1
        # gap fill: extend each segment end to next start; first starts at 0, last ends at length
        for i, s in enumerate(segs):
            s["fill_start"] = 0 if i == 0 else segs[i - 1]["fill_end"]
            s["fill_end"] = length if i == len(segs) - 1 else (s["end"] + segs[i + 1]["start"]) // 2 if segs[i + 1]["start"] > s["end"] else s["end"]
            if s["fill_end"] < s["end"]:
                s["fill_end"] = s["end"]
        # ensure monotone, contiguous
        for i in range(1, len(segs)):
            segs[i]["fill_start"] = segs[i - 1]["fill_end"]
            if segs[i]["fill_end"] < segs[i]["fill_start"]:
                segs[i]["fill_end"] = segs[i]["fill_start"]
        assert segs[0]["fill_start"] == 0 and segs[-1]["fill_end"] == length
        episodes[str(ep)] = {"split": sp, "length": length, "annotation_duration": ann["meta_data"]["task_duration"],
                             "segments": segs}
    classes = {name: i for i, (name, _) in enumerate(sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0])))}
    for ep in episodes.values():
        for s in ep["segments"]:
            s["class"] = classes[s["skill"]]
    out = {"task_id": data.task_id, "dataset_revision": data.provenance["revision"], "split_sha256": data.split_sha256,
           "num_classes": len(classes), "classes": classes,
           "class_counts": {"all": dict(vocab), "train": dict(vocab_split["train"]), "validation": dict(vocab_split["validation"])},
           "note": "skill_description is the class label; objects retained for VLM grounding; gap frames assigned to neighbouring segments",
           "episodes": episodes}
    target = data.root / f"skill_segments_task{data.task_id:03d}.json"
    target.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"output": str(target), "episodes": len(episodes), "segments": sum(vocab.values()),
                      "num_classes": len(classes), "classes": classes,
                      "train_counts": dict(vocab_split["train"]), "val_counts": dict(vocab_split["validation"])}, indent=1))


if __name__ == "__main__":
    main()
