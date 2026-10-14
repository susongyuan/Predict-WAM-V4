#!/usr/bin/env python3
"""False-accept rate of the PATCH POSTCONDITION appearance candidate (default key `open lid(car)`) on REAL physical
negatives from dev closed-loop runs.

Measurement only: calibration/validation of an UNVERIFIED gate, not a robot capability or task-success result.  The
gate stays `appearance_candidate`; nothing here promotes it and the deployed threshold is NOT retuned (a post-hoc
"FAR<=5% on these negatives" threshold is reported separately, labelled optimistic, never written back).

Negatives : every closed-loop HEAD snapshot <run>/observations/ep<E>_step<k>.png (wrist renders live in a subdir and are
            ignored) whose (episode, step) binds to an evaluator-side object-joint log row (eval/object_joints/*.jsonl,
            --log-object-joints).  Row k is written after the k-th env.step (row 0 at reset), i.e. it is the physical
            state that rendered observation k, which the policy saw at step_in_episode k and snapshotted
            (OmniGibson eval/evaluator.py step(): policy.forward(obs) -> env.step -> _write_joint_log).  Episode
            binding: header episode_identity.policy_episode when logged, else exactly one joint log + one snapshot
            episode in the run.  Label NEGATIVE iff j_trunk < lower + 0.05 (upper - lower) (OmniGibson Open-state
            revolute threshold = 0.1265 rad for this car).  Unbound frames are skipped and counted.
Skill tag : orchestrator contract windows [start, end] (inclusive) from the run's policy trace.
Positives : VALIDATION demo end frames of the key's segments at the calibration anchors, selected exactly as
            scripts/build_patch_postcondition_prototypes.py does (segment-fraction labels, NOT physical labels).
Encoding  : WamPolicy._preprocess (prep_rgb: uint8 -> INTER_AREA to 224 -> ImageNet CHW) and WamPolicy._encode (one
            DINOv3 pass; last_hidden_state minus CLS/registers) called unbound on a CPU namespace, then
            harness.vtt_monitor.PatchPostconditionGate.score.  Only deviation from deployment: fp32 on CPU instead of
            bf16 on CUDA (GPU reserved); quantified against the runtime post_score logged in traces where available.
"""
import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: the GPU is reserved for other jobs (set before torch import)

from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.frame_cache import FramePack  # noqa: E402
from harness.policy_server import WamPolicy  # noqa: E402
from harness.vtt_monitor import PatchPostconditionGate, precondition_key  # noqa: E402
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, VTT_DIR, SkillTable, load_upstream  # noqa: E402
from scripts.build_patch_arrival_prototypes import align  # noqa: E402

SNAP_RE = re.compile(r"^ep(\d+)_step(\d+)\.png$")
OPEN_FRACTION = 0.05     # omnigibson/object_states/open_state.py JOINT_THRESHOLD_BY_TYPE[REVOLUTE]
JOINT = "j_trunk"


def ram_used_frac():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
    return 1.0 - info["MemAvailable"] / info["MemTotal"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RuntimeEncoder:
    """WamPolicy._preprocess + WamPolicy._encode (the deployed code objects, called unbound) on a CPU fp32 namespace."""

    def __init__(self, dino_path, image_size=224, feat_layers=(-12, -8, -4), include_cls_register=True):
        _, runner = load_upstream()
        enc, _, n_reg, _ = runner.ModelFactory.create_vision_encoder(str(dino_path), torch.float32, "cpu")
        wrapper = SimpleNamespace(device=torch.device("cpu"), encoder_dtype=torch.float32, vision_encoder=enc,
                                  feat_layers=list(feat_layers), include_cls_register=include_cls_register,
                                  num_register_tokens=n_reg)
        cfg = SimpleNamespace(dataset=SimpleNamespace(image_size=[image_size, image_size]))
        self.ns = SimpleNamespace(cfg=cfg, _find=WamPolicy._find, wrapper=wrapper, _last_patch=None)
        self.state = np.zeros(61, np.float32)   # _preprocess requires proprio; the image path never reads it

    def patches(self, rgb):
        _, px = WamPolicy._preprocess(self.ns, {"r::proprio": self.state, "r::zed_link::rgb": rgb})
        WamPolicy._encode(self.ns, px)
        return self.ns._last_patch


def jpeg90_roundtrip(rgb):
    """Demo frame-cache compression stage (PIL JPEG q90 encode, cv2 decode as FramePack.rgb) for a sensitivity check."""
    import cv2
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    arr = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------------------------- closed-loop data
def load_joint_log(path):
    header, end, rows = None, None, {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "header":
                header = r
            elif r.get("type") == "end":
                end = r
            elif "step" in r:
                rows[int(r["step"])] = r["joints"]
    if header is None:
        raise ValueError(f"{path}: no header")
    for name, e in header["entities"].items():
        if JOINT in e["joints"]:
            j = e["joints"].index(JOINT)
            lo, hi = float(e["lower"][j]), float(e["upper"][j])
            values = {k: float(v[name][j]) for k, v in rows.items() if name in v}
            return {"entity": name, "lower": lo, "upper": hi, "open_threshold": lo + OPEN_FRACTION * (hi - lo),
                    "values": values, "identity": header.get("episode_identity"),
                    "environment_steps": None if end is None else end.get("environment_steps"),
                    "peak": max(values.values()) if values else float("nan")}
    raise ValueError(f"{path}: no {JOINT} in header entities")


def load_trace(run, episode, key):
    windows, runtime_post, chunks, holds = [], {}, [], []
    for tf in sorted(run.glob("policy_trace_*.jsonl")):
        with open(tf) as f:
            for line in f:
                r = json.loads(line)
                if r.get("episode") != episode:
                    continue
                ev = r.get("event")
                if ev == "orch_contract_start":
                    windows.append({"key": r["key"], "index": r.get("index"), "attempt": r.get("attempt"),
                                    "start": int(r["step"]), "end": None, "predicate": None})
                elif ev == "orch_contract_end":
                    for w in reversed(windows):
                        if w["end"] is None and w["key"] == r["key"] and w["index"] == r.get("index"):
                            w["end"], w["predicate"] = int(r["step"]), r.get("predicate")
                            break
                elif ev == "progress" and r.get("post_key") == key and r.get("post_score") is not None:
                    runtime_post[int(r["step"])] = float(r["post_score"])
                elif ev == "chunk":
                    chunks.append((int(r["step"]), r.get("skill")))
                elif ev == "orch_hold":
                    holds.append((int(r["step"]), r.get("reason")))
    return windows, runtime_post, chunks, holds


def tag_step(step, windows, chunks, holds, key):
    inside = [w for w in windows if w["start"] <= step and (w["end"] is None or step <= w["end"])]
    ol = [w for w in inside if w["key"] == key]
    if inside:
        active = max(inside, key=lambda w: w["start"])
        name, attempt = active["key"], active["attempt"]
    elif windows:
        prev = [r for st, r in holds if st <= step]      # program halted after its last contract (robot holding)
        name, attempt = (f"orch_hold:{prev[-1]}" if prev else "no_active_contract"), None
    else:
        prev = [s for st, s in chunks if st <= step]
        name, attempt = (f"chunk_skill:{prev[-1]}" if prev else "unknown"), None
    return {"active_contract": name, "contract_attempt": attempt, "in_key_attempt": bool(ol),
            "key_attempt": (max(ol, key=lambda w: w["start"])["attempt"] if ol else None)}


def closed_loop_frames(root, key):
    runs, frames, skipped = [], [], {"no_joint_log": 0, "episode_binding": 0, "step_missing_in_joint_log": 0}
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        snaps = []
        for p in sorted(run.glob("observations/ep*_step*.png")):
            m = SNAP_RE.match(p.name)
            if m:
                snaps.append((int(m.group(1)), int(m.group(2)), p))
        logs = sorted(run.glob("eval/object_joints/*.jsonl"))
        info = {"run": run.name, "snapshots": len(snaps), "joint_logs": len(logs)}
        runs.append(info)
        if not snaps:
            continue
        if not logs:
            skipped["no_joint_log"] += len(snaps)
            info["skipped"] = {"no_joint_log": len(snaps)}
            continue
        eps = sorted({e for e, _, _ in snaps})
        jl = [load_joint_log(p) for p in logs]
        bound = {}
        for p, j in zip(logs, jl):
            ident = j["identity"] or {}
            if ident.get("policy_episode") is not None:
                bound[int(ident["policy_episode"])] = (p, j, "header episode_identity.policy_episode")
        if not bound and len(logs) == 1 and len(eps) == 1:
            bound[eps[0]] = (logs[0], jl[0], "single joint log + single snapshot episode in run")
        eval_json = sorted(run.glob("eval/json/*.json"))
        align_flag = None
        if eval_json:
            ej = json.loads(eval_json[0].read_text())
            align_flag = ej.get("policy_step_alignment")
            pe = ej.get("policy_episode")
            if pe is not None and pe not in bound:
                bound = {}
        info.update({"episodes_in_snapshots": eps, "bound_episodes": sorted(bound),
                     "binding_rule": sorted({v[2] for v in bound.values()}), "policy_step_alignment": align_flag,
                     "joint_peak": {str(e): round(v[1]["peak"], 5) for e, v in bound.items()},
                     "open_threshold": {str(e): round(v[1]["open_threshold"], 5) for e, v in bound.items()}})
        sk = {"episode_binding": 0, "step_missing_in_joint_log": 0}
        traces = {}
        for ep, step, p in snaps:
            if ep not in bound:
                sk["episode_binding"] += 1
                continue
            _, j, _ = bound[ep]
            if step not in j["values"]:
                sk["step_missing_in_joint_log"] += 1
                continue
            if ep not in traces:
                traces[ep] = load_trace(run, ep, key)
                w = traces[ep][0]
                info.setdefault("contracts", {})[str(ep)] = [
                    {"key": x["key"], "attempt": x["attempt"], "start": x["start"], "end": x["end"], "predicate": x["predicate"]}
                    for x in w]
            windows, runtime_post, chunks, holds = traces[ep]
            jt = j["values"][step]
            frames.append({"run": run.name, "episode": ep, "step": step, "file": str(p.relative_to(ROOT)),
                           "j_trunk": jt, "open_threshold": j["open_threshold"],
                           "label": "negative" if jt < j["open_threshold"] else "physical_positive",
                           **tag_step(step, windows, chunks, holds, key), "runtime_post_score": runtime_post.get(step)})
        cov = {}
        snap_steps = {s for e, s, _ in snaps}
        for e, (_, j, _) in bound.items():
            ep_steps = {s for ee, s, _ in snaps if ee == e}
            cov[str(e)] = {f"logged_steps_j_ge_{t}": sum(v >= t for v in j["values"].values()) for t in (0.01, 0.05)}
            cov[str(e)].update({f"snapshotted_steps_j_ge_{t}": sum(j["values"].get(s, -1.0) >= t for s in ep_steps) for t in (0.01, 0.05)})
        info["lid_motion_snapshot_coverage"] = cov
        # runtime-logged post_score (deployed bf16 CUDA encoder, every 30 steps while the key contract ran) joined with
        # the joint log at the same step: image-free supplementary negatives from the runtime itself
        rt = []
        for e, (windows, runtime_post, _, _) in traces.items():
            _, j, _ = bound[e]
            rt += [{"run": run.name, "episode": e, "step": st, "score": sc, "j_trunk": j["values"][st]}
                   for st, sc in sorted(runtime_post.items()) if st in j["values"]]
        info["runtime_post_rows"] = rt
        del snap_steps
        for k, v in sk.items():
            skipped[k] += v
        info["skipped"] = sk
        info["scored_frames"] = sum(1 for f in frames if f["run"] == run.name)
    return runs, frames, skipped


def demo_jobs(key, anchors, ref_fractions):
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    jobs = []
    for ep in sorted(table.episodes):
        if index["episodes"][str(ep)]["split"] != "validation":
            continue
        try:
            pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        except Exception:
            continue
        L, stride = pack.length, pack.stride
        for s in table.episodes[ep]["segments"]:
            if s["skill"] == "move to" or precondition_key(s["skill"], s["objects"]) != key:
                continue
            lo, hi = s["start"], s["end"]
            for fr in anchors:
                jobs.append({"episode": ep, "frame": align(int(lo + fr * (hi - lo - 1)), stride, L), "kind": "demo_positive", "fraction": fr})
            for fr in ref_fractions:
                jobs.append({"episode": ep, "frame": align(int(lo + fr * (hi - lo - 1)), stride, L), "kind": "demo_unfinished_ref", "fraction": fr})
    return jobs


# ------------------------------------------------------------------------------------------------------- statistics
def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                 + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1)))


def clopper_pearson(x, n, alpha=0.05):
    if n == 0:
        return [None, None]

    def bisect(f, lo, hi):
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return 0.5 * (lo + hi)
    lo = 0.0 if x == 0 else bisect(lambda p: (1.0 - binom_cdf(x - 1, n, p)) - alpha / 2, 0.0, x / n)
    hi = 1.0 if x == n else bisect(lambda p: alpha / 2 - binom_cdf(x, n, p), x / n, 1.0)
    return [round(lo, 4), round(hi, 4)]


def dist(v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    return {"n": int(v.size), "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
            **{k: round(float(x), 4) for k, x in zip(("min", "p5", "p25", "median", "p75", "p95", "max"), q)}}


def far_block(scores, thr):
    s = np.asarray(scores, np.float64)
    fa = int((s >= thr).sum())
    return {"n": int(s.size), "false_accepts": fa, "far": (round(fa / s.size, 4) if s.size else None),
            "far_95ci_clopper_pearson": clopper_pearson(fa, int(s.size)),
            "max_score": (round(float(s.max()), 4) if s.size else None),
            "margin_max_minus_threshold": (round(float(s.max() - thr), 4) if s.size else None), "distribution": dist(s)}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, np.float64)[:, None], np.asarray(neg, np.float64)[None, :]
    if pos.size == 0 or neg.size == 0:
        return None
    return round(float((pos > neg).mean() + 0.5 * (pos == neg).mean()), 4)


def ranks(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    for val in np.unique(x):                      # average ranks for ties
        m = x == val
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return {"n": int(a.size), "pearson": None, "spearman": None}
    return {"n": int(a.size), "pearson": round(float(np.corrcoef(a, b)[0, 1]), 4),
            "spearman": round(float(np.corrcoef(ranks(a), ranks(b))[0, 1]), 4)}


def threshold_for_far(neg, far):
    """Smallest threshold strictly above the (floor(far*N)+1)-th largest negative -> at most floor(far*N) accepts."""
    s = np.sort(np.asarray(neg, np.float64))[::-1]
    k = int(math.floor(far * len(s)))
    return float("-inf") if k >= len(s) else float(np.nextafter(s[k], np.inf))


# ---------------------------------------------------------------------------------------------------- contact sheet
def contact_sheet(tiles, path, title, cols=4, size=224, cap=46):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
        tfont = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except OSError:
        font = tfont = ImageFont.load_default()
    rows = math.ceil(len(tiles) / cols)
    head = 26
    sheet = Image.new("RGB", (cols * size, head + rows * (size + cap)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((6, 5), title, fill=(0, 0, 0), font=tfont)
    for i, (rgb, lines, colour) in enumerate(tiles):
        x, y = (i % cols) * size, head + (i // cols) * (size + cap)
        sheet.paste(Image.fromarray(rgb), (x, y))
        d.rectangle([x, y, x + size - 1, y + size - 1], outline=colour, width=3)
        for j, line in enumerate(lines[:3]):
            d.text((x + 4, y + size + 2 + 14 * j), line, fill=(0, 0, 0), font=font)
    sheet.save(path, optimize=True)


def short(run):
    return re.sub(r"_2026\d{4}$", "", run.replace("s1_dev10_", ""))


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default="open lid(car)")
    ap.add_argument("--closed-loop-root", type=Path, default=ROOT / "outputs/behavior/closed_loop")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/validation/postcond_real_negatives_20260924")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-ram", type=float, default=0.85, help="abort if system RAM use exceeds this fraction")
    ap.add_argument("--far-target", type=float, default=0.05, help="post-hoc (NOT deployed) FAR target on these negatives")
    ap.add_argument("--per-run", type=int, default=3, help="max top-scoring negatives per run on the contact sheet")
    ap.add_argument("--no-jpeg-check", action="store_true", help="skip the JPEG-q90 compression sensitivity pass")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(args.threads)
    assert not torch.cuda.is_available(), "CUDA must stay unused (CUDA_VISIBLE_DEVICES is cleared)"
    if ram_used_frac() > args.max_ram:
        raise SystemExit(f"RAM use {ram_used_frac():.2f} > {args.max_ram}; not starting")
    args.out.mkdir(parents=True, exist_ok=True)

    gate = PatchPostconditionGate(VTT_DIR)
    key = args.key
    thr = gate.threshold(key)
    if thr is None or gate.mode != "contrast":
        raise SystemExit(f"key {key!r}: threshold {thr}, mode {gate.mode}; nothing to measure")
    meta = gate.meta
    cal = meta["per_key"][key]

    runs, frames, skipped = closed_loop_frames(args.closed_loop_root, key)
    jobs = demo_jobs(key, [float(x) for x in meta["anchors"]],
                     [float(x) for x in meta["neg_fractions"]] + [float(x) for x in meta.get("ambiguous_fractions", [])])
    print(f"closed-loop frames bound {len(frames)}, skipped {skipped}; demo jobs {len(jobs)}", flush=True)

    enc = RuntimeEncoder(ROOT / meta["dino"])
    print(f"encoder loaded {time.time() - t0:.0f}s, RAM {ram_used_frac():.2f}", flush=True)

    def score(rgb):
        return float(gate.score(enc.patches(rgb), key))

    from PIL import Image
    for i, f in enumerate(frames):
        rgb = np.asarray(Image.open(ROOT / f["file"]).convert("RGB"))
        f["score"] = score(rgb)
        f["score_jpeg90"] = None if args.no_jpeg_check else score(jpeg90_roundtrip(rgb))
        if i % 25 == 0:
            if ram_used_frac() > args.max_ram:
                raise SystemExit(f"RAM use {ram_used_frac():.2f} > {args.max_ram}; aborting")
            print(f"[closed-loop] {i + 1}/{len(frames)} {time.time() - t0:.0f}s", flush=True)
    packs = {}
    for i, jb in enumerate(jobs):
        if jb["episode"] not in packs:
            packs = {jb["episode"]: FramePack(FRAMES / HEAD_CAMERA, jb["episode"])}
        jb["score"] = score(packs[jb["episode"]].rgb(jb["frame"]))
        if i % 25 == 0:
            print(f"[demo] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)

    neg = [f for f in frames if f["label"] == "negative"]
    phys_pos = [f for f in frames if f["label"] == "physical_positive"]
    neg_ol = [f for f in neg if f["in_key_attempt"]]
    neg_nav = [f for f in neg if not f["in_key_attempt"] and f["active_contract"].startswith("move to")]
    neg_idle = [f for f in neg if f["active_contract"].startswith(("orch_hold", "no_active_contract"))]
    pos = [j for j in jobs if j["kind"] == "demo_positive"]
    ref = [j for j in jobs if j["kind"] == "demo_unfinished_ref"]
    S = lambda rows, k="score": [r[k] for r in rows]  # noqa: E731

    # post-hoc thresholds (NOT deployed): computed on the evaluation negatives themselves -> optimistic
    posthoc = {}
    for name, rows in (("all_negatives", neg), ("key_attempt_negatives", neg_ol)):
        if not rows:
            continue
        t5 = threshold_for_far(S(rows), args.far_target)
        t0f = float(np.nextafter(max(S(rows)), np.inf))
        posthoc[name] = {
            f"threshold_far_le_{args.far_target:.2f}": round(t5, 4),
            "far_achieved": round(float(np.mean(np.asarray(S(rows)) >= t5)), 4),
            "demo_recall_at_it": round(float(np.mean(np.asarray(S(pos)) >= t5)), 4),
            "threshold_far_0": round(t0f, 4), "demo_recall_at_far_0": round(float(np.mean(np.asarray(S(pos)) >= t0f)), 4)}

    # fidelity: offline CPU-fp32 score vs runtime (CUDA-bf16) post_score logged at the same step and frame
    fid = [f for f in frames if f["runtime_post_score"] is not None]
    d = np.array([f["score"] - f["runtime_post_score"] for f in fid], np.float64)
    fidelity = {"n_matched_frames": len(fid), "runs": sorted({f["run"] for f in fid}),
                "mean_abs_diff": (round(float(np.abs(d).mean()), 4) if d.size else None),
                "max_abs_diff": (round(float(np.abs(d).max()), 4) if d.size else None),
                "mean_signed_diff_offline_minus_runtime": (round(float(d.mean()), 4) if d.size else None),
                "decision_agreement": (round(float(np.mean([(f["score"] >= thr) == (f["runtime_post_score"] >= thr) for f in fid])), 4) if fid else None),
                "note": "runtime post_score is rounded to 3 decimals in the trace; runtime encoder is bf16 on CUDA, offline fp32 on CPU"}

    rt_rows = [x for r in runs for x in r.pop("runtime_post_rows", [])]
    open_thr = frames[0]["open_threshold"] if frames else float("-inf")
    rt_neg = [x for x in rt_rows if x["j_trunk"] < open_thr]
    runtime_logged = {"what": "post_score emitted by the deployed runtime (bf16 CUDA) every 30 steps during key contracts, "
                              "joined with j_trunk at the same step; independent of the offline re-encoding",
                      "runs": sorted({x["run"] for x in rt_rows}), "n_rows": len(rt_rows), "negatives": far_block([x["score"] for x in rt_neg], thr),
                      "max_j_trunk": (round(max(x["j_trunk"] for x in rt_rows), 5) if rt_rows else None)}
    jpeg = None
    if not args.no_jpeg_check and neg:
        dj = np.array([f["score_jpeg90"] - f["score"] for f in neg], np.float64)
        jpeg = {"what": "negatives re-encoded with the demo frame-cache JPEG stage (PIL q90) before the same encoder; "
                        "sensitivity of the score to the compression-domain gap between lossless closed-loop renders "
                        "and JPEG demo positives (the mp4 stage of the demo cache is not reproduced)",
                "mean_diff_jpeg_minus_png": round(float(dj.mean()), 4), "max_abs_diff": round(float(np.abs(dj).max()), 4),
                "all": far_block(S(neg, "score_jpeg90"), thr),
                "key_attempt": far_block(S(neg_ol, "score_jpeg90"), thr) if neg_ol else None}

    hard = sorted(neg, key=lambda f: -f["score"])
    top_j = sorted(neg, key=lambda f: -f["j_trunk"])[:5]
    per_run = []
    for r in runs:
        rows = [f for f in neg if f["run"] == r["run"]]
        rr = dict(r)
        if rows:
            rr.update({"negatives": len(rows), "key_attempt_negatives": sum(f["in_key_attempt"] for f in rows),
                       "false_accepts": sum(f["score"] >= thr for f in rows), "max_score": round(max(S(rows)), 4),
                       "max_score_key_attempt": (round(max(f["score"] for f in rows if f["in_key_attempt"]), 4)
                                                 if any(f["in_key_attempt"] for f in rows) else None)})
        per_run.append(rr)

    def brief(f):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()
                if k in ("run", "episode", "step", "file", "score", "score_jpeg90", "j_trunk", "active_contract",
                         "contract_attempt", "in_key_attempt", "runtime_post_score")}

    report = {
        "title": f"Patch postcondition candidate `{key}` on real physical closed-loop negatives",
        "measurement_kind": "offline calibration/validation measurement of an UNVERIFIED appearance candidate; "
                            "NOT a robot capability or task-success result",
        "gate": {"class": "harness.vtt_monitor.PatchPostconditionGate", "mode": gate.mode, "key": key,
                 "deployed_threshold": round(thr, 6), "status": "appearance_candidate (unchanged by this report)",
                 "prototypes": "data/skill_vtt_task013/postcond_patch_prototypes.npz",
                 "prototypes_sha256": sha256(VTT_DIR / "postcond_patch_prototypes.npz"),
                 "calibration": {k: cal.get(k) for k in ("threshold_recall_rule", "threshold_far_rule", "recall_val",
                                                          "val_pos_mean", "far_by_fraction", "neg_mean_by_fraction")},
                 "calibration_label_source": "demo segment fractions (not physical)"},
        "encoding": {"path": "WamPolicy._preprocess (prep_rgb 224 INTER_AREA + ImageNet norm) -> WamPolicy._encode "
                             "(DINOv3 last_hidden_state[0, 1+num_register_tokens:]) -> PatchPostconditionGate.score; "
                             "deployed functions called unbound",
                     "dino": meta["dino"], "device": "cpu", "dtype": "float32 (deployment: bfloat16 on CUDA)",
                     "closed_loop_input": "observations/ep*_step*.png = lossless snapshot of the exact head obs array "
                                          "the policy received at that step (224x224 uint8)",
                     "demo_input": "data/frame_cache zed_link JPEG q90 frames via FramePack.rgb (as the build script)"},
        "physical_label": {"joint": JOINT, "rule": f"negative iff {JOINT} < lower + {OPEN_FRACTION} * (upper - lower) "
                                                   "(OmniGibson Open-state revolute threshold)",
                           "open_threshold": round(frames[0]["open_threshold"], 5) if frames else None,
                           "binding": "joint row k = state after k env steps = state that rendered observation k = policy "
                                      "step_in_episode k snapshot (evaluator.step: policy.forward(obs) -> env.step -> "
                                      "_write_joint_log; row 0 written at reset)",
                           "peak_j_trunk_over_scored_frames": round(max(f["j_trunk"] for f in frames), 5) if frames else None,
                           "peak_j_trunk_over_all_logged_steps": max((max(r["joint_peak"].values()) for r in runs if r.get("joint_peak")), default=None)},
        "counts": {"runs": len(runs), "runs_with_snapshots": sum(r["snapshots"] > 0 for r in runs),
                   "runs_with_joint_log_and_snapshots": sum(r["snapshots"] > 0 and r["joint_logs"] > 0 for r in runs),
                   "head_snapshots_total": sum(r["snapshots"] for r in runs), "scored_bound_frames": len(frames),
                   "skipped": skipped, "negatives": len(neg), "physical_positives": len(phys_pos),
                   "negatives_during_key_attempts": len(neg_ol), "negatives_navigation": len(neg_nav),
                   "negatives_halted_holding": len(neg_idle),
                   "demo_validation_positives": len(pos), "demo_unfinished_reference": len(ref)},
        "negatives": {"all": far_block(S(neg), thr), "during_key_attempts": far_block(S(neg_ol), thr),
                      "navigation_move_to": far_block(S(neg_nav), thr), "halted_holding": far_block(S(neg_idle), thr),
                      "runs_with_any_false_accept": sorted({f["run"] for f in neg if f["score"] >= thr}),
                      "operational_note": "the runtime gate is only queried while a non-navigation contract is active, so "
                                          "during_key_attempts is the in-context FAR; the other strata test specificity "
                                          "(what an off-nominal view scores if it occurred during an attempt)"},
        "demo_validation_positives": {"recall_at_deployed_threshold": round(float(np.mean(np.asarray(S(pos)) >= thr)), 4),
                                      "n": len(pos), "distribution": dist(S(pos)),
                                      "calibration_recall_val_bf16_cuda": cal.get("recall_val"),
                                      "label_source": "demo segment fraction (anchors %s), not physical" % meta["anchors"]},
        "demo_unfinished_reference": {str(fr): dist([j["score"] for j in ref if j["fraction"] == fr])
                                      for fr in sorted({j["fraction"] for j in ref})},
        "separation_auroc_demo_positives_vs": {"all_negatives": auroc(S(pos), S(neg)),
                                               "key_attempt_negatives": auroc(S(pos), S(neg_ol))},
        "posthoc_threshold_NOT_deployed": {**posthoc, "note": "computed on the same negatives it is evaluated on "
                                                              "(optimistic); reported for planning only, never written back"},
        "jtrunk_vs_score_over_negatives": {"all_raw": corr(S(neg, "j_trunk"), S(neg)),
                                           "during_key_attempts_raw": corr(S(neg_ol, "j_trunk"), S(neg_ol)),
                                           "all_quantised_1e-4rad": corr(np.round(S(neg, "j_trunk"), 4), S(neg)),
                                           "during_key_attempts_quantised_1e-4rad": corr(np.round(S(neg_ol, "j_trunk"), 4), S(neg_ol)),
                                           "n_negatives_j_ge_1e-3": sum(f["j_trunk"] >= 1e-3 for f in neg),
                                           "lid_motion_snapshot_coverage_total": {
                                               k: sum(c[k] for r in runs for c in r.get("lid_motion_snapshot_coverage", {}).values())
                                               for k in ("logged_steps_j_ge_0.01", "snapshotted_steps_j_ge_0.01",
                                                         "logged_steps_j_ge_0.05", "snapshotted_steps_j_ge_0.05")},
                                           "note": "j_trunk at snapshot steps is ~1e-12 solver noise except a handful of frames; "
                                                   "raw rank correlation is driven by per-run noise sign (run identity), not "
                                                   "lid motion; lid-lifted steps are almost never snapshotted (coverage)",
                                           "largest_j_trunk_frames": [brief(f) for f in top_j]},
        "hardest_negatives": [brief(f) for f in hard[:15]],
        "hardest_negatives_during_key_attempts": [brief(f) for f in sorted(neg_ol, key=lambda f: -f["score"])[:10]],
        "runtime_fidelity": fidelity,
        "runtime_logged_scores_supplementary": runtime_logged,
        "jpeg90_sensitivity": jpeg,
        "per_run": per_run,
        "caveats": [
            "All negatives come from one scene instance (loading_the_car 311) and a handful of dev checkpoints; frames "
            "within a run are correlated, so the binomial CI is optimistic (see per_run).",
            "Closed-loop docking reached front-left/side of the car, not behind the trunk; the hardest physically "
            "possible negatives (arm on the lid at the demo docking pose, lid shut) are not represented.",
            "No physically labelled positives exist yet (trunk never opened in any run); recall is measured on demo "
            "segment-fraction positives from the JPEG frame cache, a different compression domain from runtime renders.",
            "Offline encoder is fp32 CPU; deployment is bf16 CUDA (runtime_fidelity quantifies the gap where traces logged post_score)."],
        "seconds": round(time.time() - t0, 1),
    }
    txt = json.dumps(report, indent=1)
    assert '"verified": true' not in txt.lower()
    (args.out / "report.json").write_text(txt + "\n")
    with open(args.out / "frame_scores.jsonl", "w") as fh:
        for f in frames:
            fh.write(json.dumps(brief(f) | {"label": f["label"]}) + "\n")
        for j in jobs:
            fh.write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in j.items()}) + "\n")

    # contact sheet (4 columns): row 1-2 top-scoring negatives (<= --per-run per run, the rest are near-duplicates of a
    # held pose), row 3 top negatives DURING key attempts (the in-context case), row 4 validation demo positives
    tiles = []

    def neg_tile(f):
        rgb = np.asarray(Image.open(ROOT / f["file"]).convert("RGB"))
        ctx = f"{key.split('(')[0]} att{f['key_attempt']}" if f["in_key_attempt"] else f["active_contract"].split(":")[0][:18]
        return (rgb, [f"NEG {short(f['run'])[:30]}", f"step {f['step']}  {ctx}", f"score {f['score']:.3f}  j_trunk {f['j_trunk']:.3f}"],
                (220, 40, 40) if f["score"] >= thr else (120, 120, 120))
    per_run_n, top_div = {}, []
    for f in hard:
        if per_run_n.get(f["run"], 0) < args.per_run:
            per_run_n[f["run"]] = per_run_n.get(f["run"], 0) + 1
            top_div.append(f)
        if len(top_div) == 8:
            break
    tiles += [neg_tile(f) for f in top_div]
    tiles += [neg_tile(f) for f in sorted(neg_ol, key=lambda f: -f["score"])[:4]]
    ps = sorted(pos, key=lambda j: -j["score"])
    acc = [j for j in ps if j["score"] >= thr]
    picks, seen = [], set()
    for j in [ps[0], ps[len(ps) // 2]] + ([acc[-1]] if acc else []) + [ps[-1]]:
        if (j["episode"], j["frame"]) not in seen:
            seen.add((j["episode"], j["frame"]))
            picks.append(j)
    for j in picks:
        rgb = FramePack(FRAMES / HEAD_CAMERA, j["episode"]).rgb(j["frame"])
        tiles.append((rgb, [f"DEMO POS (val) ep{j['episode']} f{j['frame']}", f"segment fraction {j['fraction']}",
                            f"score {j['score']:.3f}"], (40, 160, 40) if j["score"] >= thr else (200, 140, 0)))
    report_sheet = {"rows": ["top negatives overall (<=%d per run)" % args.per_run, "top negatives during key attempts",
                             "validation demo positives: max, median, lowest accepted, min"],
                    "frames": [f["file"] for f in top_div] + [f["file"] for f in sorted(neg_ol, key=lambda f: -f["score"])[:4]]
                    + [f"demo ep{j['episode']} frame {j['frame']}" for j in picks]}
    (args.out / "contact_sheet.json").write_text(json.dumps(report_sheet, indent=1) + "\n")
    contact_sheet(tiles, args.out / "contact_sheet.png",
                  f"{key} thr={thr:.3f} | red=false accept, grey=rejected | rows: top negatives, top in-attempt negatives, demo positives")
    print(json.dumps({"counts": report["counts"], "far_all": report["negatives"]["all"]["far"],
                      "far_key_attempts": report["negatives"]["during_key_attempts"]["far"],
                      "max_neg": report["negatives"]["all"]["max_score"],
                      "recall": report["demo_validation_positives"]["recall_at_deployed_threshold"],
                      "posthoc": posthoc, "fidelity": fidelity, "seconds": report["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
