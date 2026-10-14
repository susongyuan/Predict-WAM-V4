#!/usr/bin/env python3
"""Is the closed-loop base under-command an IMAGE domain-gap effect?

Training frames: 720x720 HEVC decode -> swscale AREA downsample to 224 -> JPEG q90 (scripts/cache_behavior_frames.py),
decoded with cv2 (harness/frame_cache.py).  Closed loop: a NATIVE 224x224 simulator render, no JPEG, no area filter
(outputs/behavior/closed_loop/*/observations/*.png are saved straight from the observation).  This offline diagnostic
re-runs the frozen WAM on the saved closed-loop PNGs under image variants that emulate the training pipeline
(JPEG q90; mild blur + JPEG q90; JPEG q75 sensitivity probe) and compares predicted base-command amplitudes with the
in-distribution reference on cached validation demo frames (where the same variants act as a double-JPEG control).

Proprio state is NOT saved with the PNGs, so closed-loop frames use PROXY 61D states taken from cached validation
demos (mid first `move to` segment for navigation frames, open-lid segment start for open-lid frames).  The
checkpoint's positions_only mask zeroes every velocity, so only joint positions matter and those are near the
demo tucked pose during navigation.  No simulator is launched; no ground truth is read; task success is not evaluated.
"""
import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.policy_server import WamPolicy  # noqa: E402
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, STATE_SLICES, SkillTable  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402

# action slice -> proprio position slice (absolute position targets in the demos), as in check_openlid_replay.py
POSITION_PAIRS = {"trunk": ((3, 7), STATE_SLICES["trunk_qpos"]),
                  "left_arm": ((7, 14), STATE_SLICES["arm_left_qpos"]),
                  "right_arm": ((15, 22), STATE_SLICES["arm_right_qpos"])}
DEFAULT_RUNS = ["s1_dev10_stage3_gain442_dockgate_20260923",
                "s1_dev10_stage3_gain442_dockgate_s8_20260923",
                "s1_dev10_stage3posonly_best5500_hold_exec32_720p_20260923"]
NAV_KEY, MANIP_KEY = "move to(car)", "open lid(car)"


# ---- image variants -----------------------------------------------------------------------------------
def jpeg_roundtrip(rgb, quality):
    """PIL encode (same call as cache_behavior_frames.py: default 4:2:0 subsampling) -> cv2 decode like FramePack.rgb."""
    import cv2
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=int(quality))
    arr = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("JPEG decode failed")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def blur(rgb, sigma):
    import cv2
    return cv2.GaussianBlur(np.ascontiguousarray(rgb), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def make_variants(sigma, q_main, q_probe):
    return {"raw": lambda x: x,
            f"jpeg{q_main}": lambda x: jpeg_roundtrip(x, q_main),
            "blur": lambda x: blur(x, sigma),
            f"blur+jpeg{q_main}": lambda x: jpeg_roundtrip(blur(x, sigma), q_main),
            f"jpeg{q_probe}": lambda x: jpeg_roundtrip(x, q_probe)}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else float("nan")


# ---- closed-loop frame collection ---------------------------------------------------------------------
def contract_intervals(run_dir):
    """[(episode, key, start_step, end_step)] from orch_contract_start/end pairs in the policy trace."""
    out, open_ = [], {}
    for trace in sorted(run_dir.glob("policy_trace_*.jsonl")):
        for line in trace.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "orch_contract_start":
                open_[(r["episode"], r["key"])] = r
            elif r.get("event") == "orch_contract_end":
                s = open_.pop((r["episode"], r["key"]), None)
                start = s["step"] if s else None
                out.append((r["episode"], r["key"], start, r["step"]))
    for (ep, key), s in open_.items():  # contract still open at episode end (e.g. budget cut by max-steps)
        out.append((ep, key, s["step"], 10**9))
    return out


def trace_raw_base(run_dir):
    """Logged raw WAM base commands (before any deploy gain) per (episode, step) and per contract key."""
    per_step, per_key = {}, {}
    for trace in sorted(run_dir.glob("policy_trace_*.jsonl")):
        for line in trace.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "chunk" and "base_cmd_raw_wam" in r:
                key = str(r.get("arm_key", "")).split(" -> ")[0]
                per_step[(r["episode"], r["step"])] = r["base_cmd_raw_wam"]
                per_key.setdefault(key, []).append(r["base_cmd_raw_wam"])
    return per_step, per_key


def collect_closed_loop_frames(run_dirs):
    from PIL import Image
    frames = []
    for run_dir in run_dirs:
        intervals = contract_intervals(run_dir)
        per_step, _ = trace_raw_base(run_dir)
        for png in sorted((run_dir / "observations").glob("ep*_step*.png")):
            m = re.match(r"ep(\d+)_step(\d+)\.png", png.name)
            ep, step = int(m.group(1)), int(m.group(2))
            key = None
            for e, k, s, t in intervals:
                if e == ep and s is not None and s <= step < t:
                    key = k
                    break
            if key is None:
                continue
            rgb = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
            if rgb.shape != (224, 224, 3):
                raise ValueError(f"{png}: expected native 224x224x3, got {rgb.shape}")
            frames.append({"run": run_dir.name, "file": png.name, "episode": ep, "step": step, "contract": key,
                           "rgb": rgb, "trace_raw_base_at_step": per_step.get((ep, step))})
    return frames


# ---- main ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt")
    ap.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="closed-loop run names under outputs/behavior/closed_loop")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cuda-fraction", type=float, default=0.35)
    ap.add_argument("--samples", type=int, default=4, help="flow samples averaged per frame/variant (fixed seeds)")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proxy-episodes", default="2604,2605", help="validation episodes supplying proxy proprio states")
    ap.add_argument("--demo-fractions", default="0.2,0.5,0.8", help="anchor fractions inside the first move-to segment")
    ap.add_argument("--max-demo-episodes", type=int, default=20)
    ap.add_argument("--blur-sigma", type=float, default=0.8)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--probe-quality", type=int, default=75)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    if args.cuda_fraction > 0.4:
        ap.error("--cuda-fraction must be <= 0.4 (shared GPU)")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    targets = json.loads(Path(args.targets).read_text())
    nav_vec = torch.tensor(targets["navigation"][NAV_KEY][MANIP_KEY]["vector"], dtype=torch.float32)[None]
    manip_vec = torch.tensor(targets["contracts"][MANIP_KEY]["vector"], dtype=torch.float32)[None]
    # secondary navigation contract present in one run (move to(door) -> open door(door)); evaluated under its own vector
    door_vec = None
    if "move to(door)" in targets["navigation"] and "open door(door)" in targets["navigation"]["move to(door)"]:
        door_vec = torch.tensor(targets["navigation"]["move to(door)"]["open door(door)"]["vector"], dtype=torch.float32)[None]
    cond_for = {NAV_KEY: nav_vec, MANIP_KEY: manip_vec, "move to(door)": door_vec}

    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=32, num_steps=args.num_steps, device=args.device,
                       cuda_fraction=args.cuda_fraction, seed=args.seed, log_dir=args.out,
                       orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})
    torch.set_num_threads(int(args.threads))
    assert policy.cfg.dataset.image_size[0] == 224

    # ---- proxy states from validation demos --------------------------------------------------------------
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    val_eps = [ep for ep in sorted(table.episodes) if index["episodes"][str(ep)]["split"] == "validation"]
    proxy_eps = [int(x) for x in args.proxy_episodes.split(",")]
    proxies = {"nav": [], "open_lid": []}
    for ep in proxy_eps:
        if ep not in val_eps:
            raise ValueError(f"proxy episode {ep} is not a validation episode")
        segs = table.episodes[ep]["segments"]
        state_all = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        nav = segs[0]
        assert nav["skill"] == "move to", nav
        mid = (nav["fill_start"] + nav["fill_end"]) // 2
        proxies["nav"].append({"episode": ep, "frame": int(mid), "state": np.array(state_all[mid], np.float32)})
        lid = next(s for s in segs if s["skill"] == "open lid")
        f0 = lid["fill_start"] + 2
        proxies["open_lid"].append({"episode": ep, "frame": int(f0), "state": np.array(state_all[f0], np.float32)})

    variants = make_variants(args.blur_sigma, args.jpeg_quality, args.probe_quality)
    K = int(args.samples)

    def predict(rgb, state, cond, seed_base):
        """Deploy path: _preprocess -> _encode -> K averaged flow samples (fixed seeds) -> actuator clip. Returns chunk, CLS."""
        _, pixels = policy._preprocess({"r::proprio": state, "r::zed_link::rgb": rgb})
        feats, cls = policy._encode(pixels)
        chunks = []
        for k in range(K):
            policy.gen.manual_seed(int(seed_base + 7919 * k))
            chunks.append(policy._sample_once(feats, state, cond))
        chunk = policy._bounded_chunk(np.mean(chunks, axis=0))
        return chunk, cls

    def base_stats(chunk):
        b = np.abs(chunk[:, :3]).mean(0)
        return {"abs_vx": float(b[0]), "abs_vy": float(b[1]), "abs_vyaw": float(b[2]),
                "abs_lin": float(np.linalg.norm(chunk[:, :2], axis=1).mean()),
                "mean_vx": float(chunk[:, 0].mean()), "mean_vyaw": float(chunk[:, 2].mean())}

    def intent_stats(chunk, state):
        out = {}
        for name, ((alo, ahi), (slo, shi)) in POSITION_PAIRS.items():
            out["abs_delta_" + name] = float(np.abs(chunk[:, alo:ahi] - state[None, slo:shi]).mean())
        out["abs_delta_upper"] = float(np.mean([out["abs_delta_" + n] for n in POSITION_PAIRS]))
        return out

    rows = []
    # ---- closed-loop frames ------------------------------------------------------------------------------
    run_dirs = [ROOT / "outputs/behavior/closed_loop" / r for r in args.runs]
    cl_frames = collect_closed_loop_frames(run_dirs)
    for i, fr in enumerate(cl_frames):
        key = fr["contract"]
        cond = cond_for.get(key)
        if cond is None:
            continue
        group = {NAV_KEY: "closed_loop_nav_car", MANIP_KEY: "closed_loop_open_lid", "move to(door)": "closed_loop_nav_door"}[key]
        plist = proxies["open_lid" if key == MANIP_KEY else "nav"]
        row = {"group": group, "run": fr["run"], "file": fr["file"], "episode": fr["episode"], "step": fr["step"], "contract": key,
               "trace_raw_base_at_step": fr["trace_raw_base_at_step"], "proxy": [(p["episode"], p["frame"]) for p in plist],
               "variants": {}}
        raw_cls = None
        for vname, fn in variants.items():
            img = fn(fr["rgb"])
            per_proxy = []
            for pi, p in enumerate(plist):
                chunk, cls = predict(img, p["state"], cond, seed_base=args.seed + 100003 * i + 1009 * pi)
                d = base_stats(chunk)
                if key == MANIP_KEY:
                    d.update(intent_stats(chunk, p["state"]))
                d["first_base_cmd"] = [round(float(x), 4) for x in chunk[0, :3]]
                per_proxy.append(d)
            if vname == "raw":
                raw_cls = cls
            agg = {k: float(np.mean([d[k] for d in per_proxy])) for k in per_proxy[0] if k != "first_base_cmd"}
            agg["cls_cos_to_raw"] = cosine(cls, raw_cls)
            agg["per_proxy"] = per_proxy
            row["variants"][vname] = agg
        rows.append(row)
        print(f"[{i + 1}/{len(cl_frames)}] {fr['run']}/{fr['file']} {key}: raw |vx|={row['variants']['raw']['abs_vx']:.3f} "
              f"jpeg |vx|={row['variants'][f'jpeg{args.jpeg_quality}']['abs_vx']:.3f}", flush=True)

    # ---- demo validation navigation frames (in-distribution reference; variants = double JPEG control) ------
    fractions = [float(x) for x in args.demo_fractions.split(",")]
    for j, ep in enumerate(val_eps[:args.max_demo_episodes]):
        seg = table.episodes[ep]["segments"][0]
        if seg["skill"] != "move to":
            continue
        action = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
        state_all = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        lo, hi = seg["fill_start"], seg["fill_end"]
        for fi, frac in enumerate(fractions):
            a = int(lo + frac * (hi - lo - 33))
            a = (a + 1) // 2 * 2
            state = np.array(state_all[a], np.float32)
            demo = np.array(action[a:a + 32], np.float32)
            rgb = pack.rgb(a)
            row = {"group": "demo_nav_car", "episode": ep, "anchor": a, "fraction": frac, "contract": NAV_KEY,
                   "demo_abs_base": np.abs(demo[:, :3]).mean(0).tolist(), "demo_mean_vx": float(demo[:, 0].mean()),
                   "variants": {}}
            raw_cls = None
            for vname, fn in variants.items():
                chunk, cls = predict(fn(rgb), state, nav_vec, seed_base=args.seed + 500009 + 1000 * ep + 31 * fi)
                if vname == "raw":
                    raw_cls = cls
                d = base_stats(chunk)
                d["cls_cos_to_raw"] = cosine(cls, raw_cls)
                d["first_base_cmd"] = [round(float(x), 4) for x in chunk[0, :3]]
                row["variants"][vname] = d
            rows.append(row)
        print(f"demo ep {ep}: raw |vx|={rows[-1]['variants']['raw']['abs_vx']:.3f} demo |vx|={rows[-1]['demo_abs_base'][0]:.3f}", flush=True)

    # ---- summary -------------------------------------------------------------------------------------------
    def group_table(gname, keys):
        rs = [r for r in rows if r["group"] == gname]
        tab = {"n_frames": len(rs)}
        for vname in variants:
            if not rs:
                continue
            tab[vname] = {k: float(np.mean([r["variants"][vname][k] for r in rs])) for k in keys if k in rs[0]["variants"][vname]}
            tab[vname]["cls_cos_to_raw"] = float(np.mean([r["variants"][vname]["cls_cos_to_raw"] for r in rs]))
            tab[vname]["cls_cos_to_raw_min"] = float(np.min([r["variants"][vname]["cls_cos_to_raw"] for r in rs]))
        return tab

    base_keys = ["abs_vx", "abs_vy", "abs_vyaw", "abs_lin", "mean_vx", "mean_vyaw"]
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": policy.checkpoint_step, "proprio_mask": policy.wrapper.proprio_mask_name,
               "action_repr": policy.wrapper.action_repr_name, "runs": args.runs, "samples_per_prediction": K,
               "num_inference_steps": args.num_steps, "seed": args.seed, "blur_sigma_px": args.blur_sigma,
               "jpeg_quality": args.jpeg_quality, "probe_quality": args.probe_quality,
               "proxy_states": {k: [(p["episode"], p["frame"]) for p in v] for k, v in proxies.items()},
               "proxy_note": ("closed-loop PNGs carry no proprio; predictions use demo proxy states (validation episodes), "
                              "averaged over the listed proxies. positions_only mask zeroes all velocities."),
               "closed_loop_nav_car": group_table("closed_loop_nav_car", base_keys),
               "closed_loop_nav_door": group_table("closed_loop_nav_door", base_keys),
               "closed_loop_open_lid": group_table("closed_loop_open_lid", base_keys + ["abs_delta_trunk", "abs_delta_left_arm", "abs_delta_right_arm", "abs_delta_upper"]),
               "demo_nav_car": group_table("demo_nav_car", base_keys),
               "task_success_evaluated": False, "simulator_launched": False}
    demo_rows = [r for r in rows if r["group"] == "demo_nav_car"]
    if demo_rows:
        summary["demo_nav_car"]["demo_actual_abs_base"] = np.mean([r["demo_abs_base"] for r in demo_rows], axis=0).tolist()
    # logged closed-loop raw WAM base amplitude during move to(car) (the quantity the hypothesis is about)
    logged = {}
    for rd in run_dirs:
        _, per_key = trace_raw_base(rd)
        for key, cmds in per_key.items():
            logged.setdefault(key, []).extend(cmds)
    summary["trace_raw_wam_abs_base_by_contract"] = {k: {"n_chunks": len(v), "abs_mean": np.abs(np.array(v)).mean(0).tolist()}
                                                     for k, v in logged.items()}
    # paired check: our raw-variant prediction at PNG steps that coincide with a logged chunk (proxy-state sanity)
    paired = [(r["variants"]["raw"]["per_proxy"][0]["first_base_cmd"], r["trace_raw_base_at_step"])
              for r in rows if r["group"].startswith("closed_loop") and r.get("trace_raw_base_at_step") is not None]
    if paired:
        p = np.array([a for a, _ in paired]); t = np.array([b for _, b in paired])
        summary["proxy_state_sanity"] = {"n_paired_steps": len(paired), "pred_first_cmd_abs_mean": np.abs(p).mean(0).tolist(),
                                         "trace_first_cmd_abs_mean": np.abs(t).mean(0).tolist(),
                                         "note": "first-step base cmd, our raw variant (proxy state, K samples) vs logged raw WAM (true state, 1 sample)"}

    # verdict
    jn, bn = f"jpeg{args.jpeg_quality}", f"blur+jpeg{args.jpeg_quality}"
    nav = summary["closed_loop_nav_car"]
    verdict = {"threshold_strong_signal": 1.5}
    if nav["n_frames"] > 0:
        ref = summary["demo_nav_car"]["raw"]["abs_vx"] if demo_rows else None
        for v in (jn, bn, f"jpeg{args.probe_quality}"):
            verdict[v] = {"abs_vx_ratio_to_raw": nav[v]["abs_vx"] / max(nav["raw"]["abs_vx"], 1e-6),
                          "abs_lin_ratio_to_raw": nav[v]["abs_lin"] / max(nav["raw"]["abs_lin"], 1e-6),
                          "abs_vx_fraction_of_demo_pred": (nav[v]["abs_vx"] / ref) if ref else None}
        r_j, r_b = verdict[jn]["abs_vx_ratio_to_raw"], verdict[bn]["abs_vx_ratio_to_raw"]
        if max(r_j, r_b) >= 1.5:
            verdict["conclusion"] = ("STRONG SIGNAL: emulating the training image pipeline raises closed-loop nav |vx| by "
                                     f">=1.5x (jpeg {r_j:.2f}x, blur+jpeg {r_b:.2f}x); image domain gap is a plausible amplitude cause")
        elif max(r_j, r_b) >= 1.2:
            verdict["conclusion"] = (f"WEAK SIGNAL: modest increase (jpeg {r_j:.2f}x, blur+jpeg {r_b:.2f}x); the image domain gap "
                                     "contributes at most a minor part of the ~4x amplitude deficit")
        else:
            verdict["conclusion"] = (f"RULED OUT as the amplitude cause: JPEG/blur emulation does not move closed-loop nav |vx| "
                                     f"(jpeg {r_j:.2f}x, blur+jpeg {r_b:.2f}x vs raw); the encoder is insensitive to this compression/filter gap")
    summary["verdict"] = verdict
    summary["seconds"] = time.monotonic() - started

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "rgb"}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (args.out / "rows.json").write_text(json.dumps(clean(rows)) + "\n")
    (args.out / "summary.json").write_text(json.dumps(clean(summary), indent=2) + "\n")
    (args.out / "README.md").write_text(readme(summary, variants, args))
    policy.trace.close()
    for path in args.out.glob("policy_trace_*.jsonl"):
        if path.stat().st_size == 0:
            path.unlink()
    print(json.dumps(clean(summary["verdict"]), indent=2))
    print((args.out / "README.md").read_text())


def readme(summary, variants, args):
    lines = ["# Image domain gap diagnostic (offline, no simulator)", "",
             f"Checkpoint `{summary['checkpoint']}` (step {summary['checkpoint_step']}, proprio_mask={summary['proprio_mask']}, "
             f"action_repr={summary['action_repr']}); K={summary['samples_per_prediction']} flow samples averaged per prediction, "
             f"{summary['num_inference_steps']} Euler steps, fixed seeds shared across variants (paired comparison).", "",
             "Variants: `raw` = deploy path as is (native 224 render); `jpegQ` = PIL JPEG quality Q (default 4:2:0 subsampling, "
             "as in cache_behavior_frames.py) then cv2 decode (as FramePack.rgb); `blur` = Gaussian sigma "
             f"{args.blur_sigma} px (emulates 720->224 AREA downsampling); `blur+jpegQ` = both.", "",
             f"Proxy proprio states for closed-loop frames (PNGs carry no proprio): {summary['proxy_states']}. "
             "Demo nav frames use their own true state and are already JPEG-cached, so their variants are a double-JPEG control.", ""]
    for g, title in (("closed_loop_nav_car", "Closed-loop `move to(car)` frames (navigation contract)"),
                     ("closed_loop_nav_door", "Closed-loop `move to(door)` frames (move to(door)->open door(door) contract)"),
                     ("closed_loop_open_lid", "Closed-loop `open lid(car)` frames (manipulation contract)"),
                     ("demo_nav_car", "Demo validation `move to(car)` frames (in-distribution reference)")):
        t = summary[g]
        lines += [f"## {title}", "", f"n frames = {t['n_frames']}", ""]
        if t["n_frames"] == 0:
            continue
        extra = ["abs_delta_trunk", "abs_delta_left_arm", "abs_delta_right_arm"] if g == "closed_loop_open_lid" else []
        hdr = ["variant", "CLS cos to raw (mean/min)", "|vx|", "|vy|", "|vyaw|", "|lin|", "mean vx"] + [e.replace("abs_delta_", "|d ") + "|" for e in extra]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for v in variants:
            d = t[v]
            cells = [v, f"{d['cls_cos_to_raw']:.4f} / {d['cls_cos_to_raw_min']:.4f}", f"{d['abs_vx']:.4f}", f"{d['abs_vy']:.4f}",
                     f"{d['abs_vyaw']:.4f}", f"{d['abs_lin']:.4f}", f"{d['mean_vx']:+.4f}"] + [f"{d[e]:.4f}" for e in extra]
            lines.append("| " + " | ".join(cells) + " |")
        if g == "demo_nav_car":
            a = t["demo_actual_abs_base"]
            lines.append(f"| demo actual action[0:3] | - | {a[0]:.4f} | {a[1]:.4f} | {a[2]:.4f} | - | - |")
        lines.append("")
    tr = summary.get("trace_raw_wam_abs_base_by_contract", {})
    if tr:
        lines += ["## Logged closed-loop raw WAM base amplitude (true state, from policy traces)", ""]
        for k, v in tr.items():
            a = v["abs_mean"]
            lines.append(f"- `{k}`: n_chunks={v['n_chunks']} |vx|={a[0]:.4f} |vy|={a[1]:.4f} |vyaw|={a[2]:.4f}")
        lines.append("")
    ps = summary.get("proxy_state_sanity")
    if ps:
        lines += ["## Proxy-state sanity (first-step base cmd at PNG steps that coincide with a logged chunk)", "",
                  f"n={ps['n_paired_steps']}: ours (proxy state) |cmd|={[round(x, 4) for x in ps['pred_first_cmd_abs_mean']]} vs "
                  f"logged (true state) |cmd|={[round(x, 4) for x in ps['trace_first_cmd_abs_mean']]}", ""]
    v = summary["verdict"]
    lines += ["## Verdict", ""]
    for name, d in v.items():
        if isinstance(d, dict):
            lines.append(f"- `{name}`: |vx| ratio to raw = {d['abs_vx_ratio_to_raw']:.2f}x, |lin| ratio = {d['abs_lin_ratio_to_raw']:.2f}x"
                         + (f", |vx| as fraction of demo-frame prediction = {d['abs_vx_fraction_of_demo_pred']:.2f}" if d.get('abs_vx_fraction_of_demo_pred') else ""))
    lines += ["", f"**{v.get('conclusion', 'no closed-loop navigation frames found')}**", "",
              "Caveats: proxy proprio states (not the true closed-loop state); few `move to(car)` PNGs exist because snapshots are taken "
              "every 300 steps or at chunk boundaries; offline chunk amplitude, not task success; no simulator or ground truth used.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
