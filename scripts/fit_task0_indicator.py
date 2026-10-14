#!/usr/bin/env python3
"""CPU diagnostics for the unadmitted Task 0 toggle-marker calibration.

Fitting is BLOCKED: first-positive next.reward does not establish an irreversible
ToggledOn state, the initial state, or reward-to-RGB alignment. Independent physical
labels and reviewed export semantics are missing. No label-admission interface is
provided; a JSON `verified: true` is not evidence. This tool never writes a fit.

    python scripts/fit_task0_indicator.py --check-segments  # metadata only
    python scripts/fit_task0_indicator.py                   # refusal, exit 2

Component/confusion helpers are diagnostic only, not physical verification.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "data/behavior-1k/2026-challenge-demos"
CACHE = ROOT / "data/frame_cache/observation.rgb.left_realsense_link_camera_0"
OUT = ROOT / "outputs/validation/task0_indicator_20260928"
MARGIN_BEFORE, MARGIN_AFTER, TOGGLE_GUARD = 60, 120, 4
N_LIVE = 5

BLOCK_REASON = (
    "Calibration refused: independent physical ON/OFF labels and reviewed export "
    "semantics (initial state and reward/frame/RGB alignment) are missing. "
    "First-positive next.reward, including traces with later negative rewards, "
    "must not generate trusted labels. No fitting or output writes are permitted."
)


class CalibrationRefused(ValueError):
    """The legacy reward-derived label path has no admissible evidence."""


def toggle_frames():
    """Retained name, but never scan rewards or return an alleged state boundary."""
    raise CalibrationRefused(BLOCK_REASON)


def _index(value, where):
    if type(value) is not int or value < 0:
        raise ValueError(f"{where}: expected a nonnegative integer, not {value!r}")
    return value


def _episode_id(value, where):
    # JSON object keys are strings; reject aliases such as '00', floats and bools.
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        if str(int(value)) == value:
            return int(value)
    return _index(value, where)


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def parse_press_segments(document):
    """Validate nested Task0 metadata, or legacy flat rows; return ep -> (start, end).

    Ranges are half-open. Exactly one press is required per listed episode: the
    old return type cannot represent multiple presses without silently losing one.
    This is structural validation only, NOT label/split-provenance admission.
    """
    groups = {}
    if isinstance(document, dict):
        if "task_id" in document:
            if _index(document["task_id"], "task_id") != 0:
                raise ValueError("task_id: expected Task0")
        if ("episodes" in document) == ("segments" in document):
            raise ValueError("expected exactly one of episodes or segments")
        if "episodes" in document:
            if "task_id" not in document:
                raise ValueError("nested metadata requires task_id=0")
            episodes = document["episodes"]
            if not isinstance(episodes, dict) or not episodes:
                raise ValueError("episodes: expected a nonempty object")
            for raw_ep, record in episodes.items():
                ep = _episode_id(raw_ep, "episode id")
                if ep in groups:
                    raise ValueError(f"duplicate episode: {ep}")
                if not isinstance(record, dict):
                    raise ValueError(f"episode {ep}: expected an object")
                length = _index(record.get("length"), f"episode {ep} length")
                if length == 0:
                    raise ValueError(f"episode {ep}: length must be positive")
                if record.get("split") not in ("train", "validation"):
                    raise ValueError(f"episode {ep}: expected train or validation split")
                groups[ep] = (record.get("segments"), length)
        else:
            document = document["segments"]
    if not groups:
        if not isinstance(document, list) or not document:
            raise ValueError("segments: expected a nonempty list")
        for row in document:
            if not isinstance(row, dict):
                raise ValueError("segment: expected an object")
            ep = _episode_id(row.get("episode"), "segment episode")
            groups.setdefault(ep, ([], None))[0].append(row)

    out = {}
    for ep, (segments, length) in groups.items():
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"episode {ep}: segments must be a nonempty list")
        previous_end = None
        for i, segment in enumerate(segments):
            where = f"episode {ep} segment {i}"
            if not isinstance(segment, dict):
                raise ValueError(f"{where}: expected an object")
            if "episode" in segment and _episode_id(segment["episode"], where) != ep:
                raise ValueError(f"{where}: conflicting episode scope")
            skill = segment.get("skill")
            if not isinstance(skill, str) or not skill.strip():
                raise ValueError(f"{where}: skill must be a nonempty string")
            start = _index(segment.get("start"), f"{where} start")
            end = _index(segment.get("end"), f"{where} end")
            if end <= start or (length is not None and end > length):
                raise ValueError(f"{where}: invalid or out-of-bounds range [{start}, {end})")
            if previous_end is not None and start < previous_end:
                raise ValueError(f"{where}: duplicate, overlapping or unordered segment")
            previous_end = end
            if skill == "press":
                if ep in out:
                    raise ValueError(f"episode {ep}: multiple press segments are unsupported")
                out[ep] = (start, end)
        if ep not in out:
            raise ValueError(f"episode {ep}: missing press segment")
    return out


def press_segments(path=None):
    """Read metadata without discarding duplicate JSON keys or coercing bad types."""
    path = Path(path) if path is not None else DEMO / "skill_segments_task000.json"
    document = json.loads(path.read_text(), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    return parse_press_segments(document)


def comps(img, cfg):
    """Loose diagnostic blobs, unchanged; no component is trusted outcome evidence."""
    # Keep metadata/refusal CLI and the small unit tests free of image dependencies.
    import cv2
    from harness import task0_indicator as ti

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
    """Reject legacy auto-label collection before accessing any frames or cache."""
    raise CalibrationRefused(BLOCK_REASON)


def confusion(rows, p):
    """Diagnostic counts for caller-supplied rows, not label admission."""
    c = {t: {v: 0 for v in ("on", "off", "unknown")} for t in ("off", "on")}
    for x in rows:
        c[x["truth"]][verdict(x["g"], x["r"], p)] += 1
    return c


def episode_level(rows, p, stride=2):
    """Count runs at the declared sampling stride, in original per-episode order.

    `frame` is the original frame_index, not row position. Gaps reset a run;
    duplicates/reversals reject the input, rather than being sorted into evidence.
    Legacy before/after keys mean truth=off/on, NOT time relative to first reward.
    At stride 2 this is only a cached-sample proxy; unseen live frames are unproven.
    Neither these counts nor caller-provided truth labels establish admission.
    """
    if _index(stride, "stride") == 0:
        raise ValueError("stride must be positive")
    need = (N_LIVE + stride - 1) // stride
    out = {"episodes": 0, "on_run_after": 0, "on_run_before": 0, "off_run_before": 0, "off_run_after": 0}
    by = {}
    for x in rows:
        if not isinstance(x, dict):
            raise ValueError("episode row must be an object")
        ep = _index(x.get("ep"), "row ep")
        frame = _index(x.get("frame"), "row frame_index")
        if x.get("truth") not in ("off", "on") or "g" not in x or "r" not in x:
            raise ValueError("episode row requires truth=off/on and g/r components")
        xs = by.setdefault(ep, [])
        if xs and frame <= xs[-1]["frame"]:
            raise ValueError(f"episode {ep}: duplicate or out-of-order frame_index {frame}")
        xs.append(x)
    for xs in by.values():
        out["episodes"] += 1
        for truth in ("off", "on"):
            for want in ("on", "off"):
                run = best = 0
                previous = None
                for x in xs:
                    if previous is None or x["frame"] - previous != stride:
                        run = 0
                    run = run + 1 if (x["truth"] == truth and verdict(x["g"], x["r"], p) == want) else 0
                    previous = x["frame"]
                    best = max(best, run)
                key = f"{want}_run_{'before' if truth == 'off' else 'after'}"
                out[key] += int(best >= need)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-segments", action="store_true", help="validate metadata only; never fit")
    parser.add_argument("--segments", type=Path, help="metadata path for --check-segments (not a label source)")
    args = parser.parse_args(argv)
    if args.segments is not None and not args.check_segments:
        parser.error("--segments requires --check-segments")
    if args.check_segments:
        path = args.segments if args.segments is not None else DEMO / "skill_segments_task000.json"
        try:
            press = press_segments(path)
        except (OSError, ValueError) as exc:
            print(json.dumps({"status": "invalid_segments", "admitted": False,
                              "source": str(path), "reason": str(exc)}), file=sys.stderr)
            return 2
        print(json.dumps({"status": "segments_valid", "admitted": False,
                          "source": str(path), "episodes": len(press),
                          "fit_status": "blocked", "reason": BLOCK_REASON}))
        return 0
    print(json.dumps({"status": "blocked", "admitted": False, "reason": BLOCK_REASON}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
