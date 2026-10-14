#!/usr/bin/env python3
"""Fit (TRAIN) and report (VALIDATION) the Task 0 left-wrist toggle-marker indicator (harness/task0_indicator.py).

Labels come from the demonstrations only: the evaluator's first reward frame (parquet next.reward > 0) is the
ToggledOn flip; left-wrist frames of the press segment +- margins before it are "off", from TOGGLE_GUARD frames after it
"on" (the marker colour changes with the state; frames in the guard window are skipped). The marker is often occluded
by the fist, so "unknown" is an expected, safe verdict; a false "on" (claims success) or a false "off" after the toggle
(would permit a re-press that turns the radio off) are the dangerous errors. Thresholds (component area / dark-ring
fraction) are chosen on TRAIN to keep both dangerous error rates at 0 while maximising recall; VALIDATION is only
reported. The marker is SIMULATOR-RENDERED (OmniGibson ToggledOn visual marker; an indicator-light proxy).

    .conda-envs/lilawam/bin/python scripts/fit_task0_indicator.py      (CPU, ~minutes)
"""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import task0_indicator as ti  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402

DEMO = ROOT / "data/behavior-1k/2026-challenge-demos"
CACHE = ROOT / "data/frame_cache/observation.rgb.left_realsense_link_camera_0"
OUT = ROOT / "outputs/validation/task0_indicator_20260928"
MARGIN_BEFORE, MARGIN_AFTER, TOGGLE_GUARD = 60, 120, 4
N_LIVE = 5                     # predicate: consecutive live (30 Hz) frames; the cache has stride 2 -> 3 cached frames


def toggle_frames():
    out = {}
    for f in sorted((DEMO / "data/chunk-000").glob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "next.reward"])
        ep, fr, r = (np.asarray(t.column(c)) for c in ("episode_index", "frame_index", "next.reward"))
        for e in np.unique(ep):
            m = (ep == e) & (r > 0)
            if m.any():
                out[int(e)] = int(fr[m].min())
    return out


def press_segments():
    segs = json.loads((DEMO / "skill_segments_task000.json").read_text())
    rows = segs["segments"] if isinstance(segs, dict) and "segments" in segs else segs
    out = {}
    for s in rows:
        if s.get("skill") == "press":
            out.setdefault(int(s["episode"]), (int(s["start"]), int(s["end"])))
    return out


def comps(img, cfg):
    """Loose per-frame components: green / red sphere-sized blobs with their dark-ring fraction (thresholds applied later)."""
    loose = replace(cfg, min_area=5, ring_dark_frac_green=0.0, ring_dark_frac_red=0.0)
    im = ti._prep(img, cfg.size)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dark = V < cfg.dark_v
    green = (H >= cfg.green_h[0]) & (H <= cfg.green_h[1]) & (S >= cfg.green_s) & (V >= cfg.green_v)
    red = ((H <= cfg.red_h[0]) | (H >= cfg.red_h[1])) & (S >= cfg.red_s) & (V >= cfg.red_v)
    return [(a, f) for a, f, _, _ in ti._blobs(green, dark, loose, 0.0)], [(a, f) for a, f, _, _ in ti._blobs(red, dark, loose, 0.0)]


def verdict(g, r, p):
    if any(a >= p["min_area"] and f >= p["ring_green"] for a, f in g):
        return "on"
    if any(a >= p["min_area"] and f >= p["ring_red"] for a, f in r):
        return "off"
    return "unknown"


def collect(episodes, toggles, press, cfg):
    rows = []
    for ep in episodes:
        if ep not in toggles or ep not in press or not (CACHE / f"episode_{ep}.jpgpack").exists():
            continue
        fp = FramePack(CACHE, ep)
        t0, t1 = press[ep]
        tg = toggles[ep]
        for f in range(max(0, t0 - MARGIN_BEFORE), min(fp.length, max(t1, tg) + MARGIN_AFTER)):
            if not fp.has(f) or tg <= f < tg + TOGGLE_GUARD:
                continue
            g, r = comps(fp.rgb(f), cfg)
            rows.append({"ep": ep, "frame": f, "truth": "on" if f >= tg else "off", "g": g, "r": r})
    return rows


def confusion(rows, p):
    c = {t: {v: 0 for v in ("on", "off", "unknown")} for t in ("off", "on")}
    for x in rows:
        c[x["truth"]][verdict(x["g"], x["r"], p)] += 1
    return c


def episode_level(rows, p, stride=2):
    """Per episode: does the predicate's N_LIVE-frame run of 'on' occur after the toggle (and never before)? Does an
    N_LIVE run of 'off' occur before it (re-press permitted) or ever after it (dangerous)?"""
    need = int(np.ceil(N_LIVE / stride))
    out = {"episodes": 0, "on_run_after": 0, "on_run_before": 0, "off_run_before": 0, "off_run_after": 0}
    by = {}
    for x in rows:
        by.setdefault(x["ep"], []).append(x)
    for ep, xs in by.items():
        xs.sort(key=lambda x: x["frame"])
        out["episodes"] += 1
        for truth in ("off", "on"):
            for want in ("on", "off"):
                run = best = 0
                for x in xs:
                    run = run + 1 if (x["truth"] == truth and verdict(x["g"], x["r"], p) == want) else 0
                    best = max(best, run)
                key = f"{want}_run_{'before' if truth == 'off' else 'after'}"
                out[key] += int(best >= need)
    return out


def main():
    split = json.loads((DEMO / "split_task_000.json").read_text())
    train, val = split["train_episode_indices"], split["validation_episode_indices"]
    toggles, press = toggle_frames(), press_segments()
    cfg = ti.DEFAULT
    tr = collect(train, toggles, press, cfg)
    va = collect(val, toggles, press, cfg)
    grid = [{"min_area": a, "ring_green": g, "ring_red": r} for a in (8, 12, 20, 30, 45) for g in (0.2, 0.35, 0.5, 0.65)
            for r in (0.5, 0.6, 0.75, 0.9)]
    scored = []
    for p in grid:
        c = confusion(tr, p)
        n_off, n_on = sum(c["off"].values()), sum(c["on"].values())
        false_on, false_off_after = c["off"]["on"] / max(n_off, 1), c["on"]["off"] / max(n_on, 1)
        scored.append((false_on == 0 and false_off_after == 0, c["on"]["on"] / max(n_on, 1), c["off"]["off"] / max(n_off, 1),
                       -false_on - false_off_after, p, c))
    ok = [s for s in scored if s[0]]
    best = max(ok or scored, key=lambda s: (s[0], s[1] + 0.5 * s[2], s[3]))
    p = best[4]
    fit = {"statement": "TRAIN demo frames only; labels = evaluator first reward frame (ToggledOn flip); simulator-rendered "
                        "ToggledOn marker (indicator-light proxy), not a real-robot signal",
           "config": asdict(cfg), "chosen": p, "zero_dangerous_errors_on_train": bool(best[0]),
           "train_frames": len(tr), "train_episodes": len({x['ep'] for x in tr}), "train_confusion": best[5],
           "train_episode_level": episode_level(tr, p),
           "grid_size": len(grid), "margins": {"before": MARGIN_BEFORE, "after": MARGIN_AFTER, "toggle_guard": TOGGLE_GUARD}}
    rep = {"statement": "VALIDATION demo frames, thresholds fixed from TRAIN (fit.json); not a closed-loop result",
           "chosen": p, "validation_frames": len(va), "validation_episodes": len({x['ep'] for x in va}),
           "validation_confusion": confusion(va, p), "validation_episode_level": episode_level(va, p),
           "rows_legend": "confusion[truth][verdict]; truth 'on' = frames >= toggle + guard"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit.json").write_text(json.dumps(fit, indent=1) + "\n")
    (OUT / "validation.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"chosen": p, "train": fit["train_confusion"], "train_ep": fit["train_episode_level"],
                      "val": rep["validation_confusion"], "val_ep": rep["validation_episode_level"]}, indent=1))


if __name__ == "__main__":
    main()
