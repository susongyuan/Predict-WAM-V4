#!/usr/bin/env python3
"""Render one representative V4 future16 closed-loop failure case (head|left|right, 10 fps).

Case: final checkpoint step 130000, seed 932101. Independent rule evidence shows
score 0/30, zero grasps and zero holds over the full 240 s budget: the clearest
representative of the V4 future16 regression. Video is additive evidence only;
the frozen evaluation wrote no video.
"""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'outputs/zhuhai2026_tasks12/wam_precision_v4_task2_future16_fixed5_20261013_b23_fast/v4_130000/zhuhai_task2/seed_932101/onboard'
OUT = ROOT / 'success vieo' / 'v4_future16_130000_seed932101_full_failure_0s_240s.mp4'
FPS, VIEWS = 10, ('head_camera', 'left_camera', 'right_camera')
CAPTION = 'V4 future16 step130000 seed932101 | head|left|right | 0-240s | 0 grasp, score 0/30'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rows = [json.loads(l) for l in (SRC / 'frames.jsonl').read_text().splitlines()]
    assert rows, 'no onboard frames'
    probe = np.load(SRC / rows[0]['file'])
    h, w = probe['head_camera'].shape[:2]
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w * 3, h))
    assert writer.isOpened(), 'VideoWriter failed to open'
    for row in rows:
        obs = np.load(SRC / row['file'])
        frame = np.concatenate([obs[v][:, :, ::-1] for v in VIEWS], axis=1)
        cv2.putText(frame, f"t={row['physics_time_seconds']:.1f}s  step {row['control_step']}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    assert OUT.is_file() and OUT.stat().st_size > 0, 'empty video'
    decoded = cv2.VideoCapture(str(OUT))
    n = int(decoded.get(cv2.CAP_PROP_FRAME_COUNT)); fps = decoded.get(cv2.CAP_PROP_FPS)
    provenance = dict(schema='ogm-wam-v4-failure-case-video-v1', selection=CAPTION,
                      milestone=130000, seed=932101, task='zhuhai_task2', full_task_success=False,
                      score=0, grasps=0, holds=0, fps=FPS, decoded_frames=n, dimensions=[w * 3, h],
                      video=str(OUT), video_sha256=sha(OUT), video_size=OUT.stat().st_size,
                      source=str(SRC), source_frames=len(rows), renderer_sha256=sha(__file__),
                      note='additive evidence; frozen eval recorded no video')
    OUT.with_suffix('.provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(video=str(OUT), frames=n, fps=fps, size=OUT.stat().st_size), ensure_ascii=False))


if __name__ == '__main__':
    main()
