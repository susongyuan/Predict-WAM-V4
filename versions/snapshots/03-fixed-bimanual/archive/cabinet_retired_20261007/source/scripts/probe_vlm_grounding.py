#!/usr/bin/env python3
"""Probe: can Qwen2.5-VL-3B-Instruct on CPU (bf16) serve as the instruction-grounding VLM of harness/grounding.py?

Runs ground() for four instructions (with a real head-camera frame and text-only), records latency, raw text,
parsed candidates, rejections and divergence, then writes outputs/grounding/qwen3b_cpu_probe.{json,md}.

    env CUDA_VISIBLE_DEVICES="" taskset -c 8-19 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
        .conda-envs/lilawam/bin/python scripts/probe_vlm_grounding.py [--only a,b] [--max-new-tokens 700]
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.grounding import Catalog, QwenVLBackend, Step, divergence, ground  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
FRAME_DIR = ROOT / "data/frame_cache/observation.rgb.zed_link_camera_0"
OUT_DIR = ROOT / "outputs/grounding"
VISIBLE = ["car", "door", "toy_box", "floors", "digital", "tennis", "coffee_table"]
INSTRUCTIONS = {
    "a": "Load the car: put the toy box into the car, put the digital camera into the toy box, put the tennis racket into the car, then close the car lid.",
    "b": "Pack the small items from the coffee table into the car.",
    "c": "Put the camera in the car.",
    "d": "Close the trunk.",
}
EXPECT = {
    "a": "exactly one candidate whose program equals configs/task013_skill_program.json top_program",
    "b": "two candidates that differ only in which small item is picked first; shared prefix move to(car), open lid(car)",
    "c": "unsupported literal place in(digital,car) must produce no executable candidate; silently changing the destination fails",
    "d": "exactly one candidate: move to(car), close lid(car)",
}


def reference_program():
    prog = json.loads((ROOT / "configs/task013_skill_program.json").read_text())
    steps = prog.get("program") or prog["top_program"]
    return [Step(s["skill"], tuple(s["objects"])).key() for s in steps]


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def judge(tag, cands, rejected, ref):
    keys = [[s.key() for s in c.steps] for c in cands]
    prefix, choices = divergence(cands, 0)
    if tag == "a":
        ok = len(cands) == 1 and keys[0] == ref
        why = f"{len(cands)} candidate(s); program {'==' if ok else '!='} reference"
    elif tag == "b":
        picks = [[k for k in ks if k.startswith("pick up from")] for ks in keys]
        first = {p[0] for p in picks if p}
        ok = len(cands) == 2 and prefix == 2 and len(choices) == 2 and len(first) == 2 \
            and all(ks[:2] == ["move to(car)", "open lid(car)"] for ks in keys)
        why = f"{len(cands)} candidate(s); shared prefix {prefix}; first picks {sorted(first)}"
    elif tag == "c":
        lit = any(r == "untrained_skill:place in(digital,car)" or "place in(digital,car)" in r for r in rejected)
        changed_destination = any("place in(digital,toy_box)" in ks for ks in keys)
        ok = not cands and (lit or not rejected)
        why = f"literal rejected={lit}; changed destination={changed_destination}; {len(cands)} executable candidate(s)"
    else:
        ok = len(cands) == 1 and keys[0] == ["move to(car)", "close lid(car)"]
        why = f"{len(cands)} candidate(s); {keys[0] if keys else 'none'}"
    return bool(ok), why, prefix, list(choices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="a,b,c,d", help="comma-separated instruction tags")
    ap.add_argument("--modes", default="image,text", help="comma-separated: image,text")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--episode", type=int, default=2600)
    ap.add_argument("--out", default=str(OUT_DIR / "qwen3b_cpu_probe.json"))
    ap.add_argument("--merge", action="store_true", help="update existing JSON in place (re-run of a subset)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from harness.frame_cache import FramePack
    torch.set_num_threads(12)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if args.merge and out_path.exists() else {}
    results.setdefault("model_id", MODEL_ID)
    results.setdefault("runs", {})
    results["settings"] = {"quant": None, "dtype": "bfloat16", "device_map": "cpu", "torch_threads": torch.get_num_threads(),
                           "max_new_tokens": args.max_new_tokens, "visible_objects": VISIBLE,
                           "frame": f"episode_{args.episode} frame {args.frame} ({FRAME_DIR.name})",
                           "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__}
    if args.note:
        results["note"] = args.note

    cat = Catalog()
    ref = reference_program()
    results["reference_program"] = ref
    frame = FramePack(FRAME_DIR, args.episode).rgb(args.frame)
    image = Image.fromarray(frame)
    results["settings"]["image_size"] = list(image.size)

    backend = QwenVLBackend(model_id=MODEL_ID, quant=None, device_map="cpu", max_new_tokens=args.max_new_tokens)
    t = time.perf_counter()
    backend._load()
    load_s = time.perf_counter() - t
    results["load_seconds"] = round(load_s, 1)
    results["rss_after_load_gib"] = round(rss_gib(), 2)
    results["model_device"] = str(backend._model.device)
    results["model_dtype"] = str(next(backend._model.parameters()).dtype)
    print(f"[load] {load_s:.1f}s  rss={rss_gib():.2f} GiB  device={backend._model.device}", flush=True)

    tok = backend._proc.tokenizer
    first = True
    for tag in args.only.split(","):
        for mode in args.modes.split(","):
            instr = INSTRUCTIONS[tag]
            t = time.perf_counter()
            cands, rejected, prompt = ground(backend, instr, VISIBLE, cat, image=image if mode == "image" else None)
            dt = time.perf_counter() - t
            ok, why, prefix, choices = judge(tag, cands, rejected, ref)
            # ground() returns parsed results only; regenerate nothing—recover the raw text via the last decode
            raw = backend.last_text if hasattr(backend, "last_text") else None
            n_tok = len(tok(raw)["input_ids"]) if raw else None
            row = {"instruction": instr, "mode": mode, "expectation": EXPECT[tag], "latency_s": round(dt, 1),
                   "includes_load": False, "first_call_incl_load_s": round(dt + load_s, 1) if first else None,
                   "new_tokens_est": n_tok, "tok_per_s": round(n_tok / dt, 2) if n_tok else None,
                   "prompt_chars": len(prompt), "raw_text": raw,
                   "candidates": [{"interpretation": c.interpretation, "program": [s.key() for s in c.steps],
                                   "preconditions": c.preconditions, "postconditions": c.postconditions} for c in cands],
                   "rejections": rejected, "divergence": {"common_prefix": prefix, "choices": choices},
                   "pass": ok, "verdict": why, "rss_gib": round(rss_gib(), 2)}
            results["runs"][f"{tag}:{mode}"] = row
            first = False
            print(f"[{tag}:{mode}] {dt:.1f}s ({n_tok} tok, {row['tok_per_s']} tok/s) pass={ok} {why} rej={rejected} rss={rss_gib():.2f} GiB", flush=True)
            out_path.write_text(json.dumps(results, indent=1))
    results["peak_rss_gib"] = round(rss_gib(), 2)
    out_path.write_text(json.dumps(results, indent=1))
    write_md(results, out_path.with_suffix(".md"))
    print(f"[done] peak rss {rss_gib():.2f} GiB -> {out_path}", flush=True)


def write_md(results, path):
    runs = results["runs"]
    lines = [f"# Qwen2.5-VL-3B-Instruct on CPU as grounding VLM — probe", "",
             f"Model `{results['model_id']}`, bf16, device_map=cpu, torch threads {results['settings']['torch_threads']}, "
             f"max_new_tokens {results['settings']['max_new_tokens']}, affinity {results['settings']['cpu_affinity'][0]}-{results['settings']['cpu_affinity'][-1]}.",
             f"Image: {results['settings']['frame']}, {results['settings']['image_size']} px. Load {results.get('load_seconds')} s, "
             f"RSS after load {results.get('rss_after_load_gib')} GiB, peak RSS {results.get('peak_rss_gib')} GiB.", ""]
    if results.get("note"):
        lines += [f"Note: {results['note']}", ""]
    lines += ["## Latency and verdict", "", "| id | mode | latency s | new tok | tok/s | #cand | pass | verdict | rejections |", "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(runs):
        r = runs[k]
        lines.append(f"| {k.split(':')[0]} | {r['mode']} | {r['latency_s']} | {r['new_tokens_est']} | {r['tok_per_s']} | {len(r['candidates'])} | "
                     f"{'PASS' if r['pass'] else 'FAIL'} | {r['verdict']} | {'; '.join(r['rejections']) or '-'} |")
    lines += ["", f"First call including model load: {next((r['first_call_incl_load_s'] for r in runs.values() if r.get('first_call_incl_load_s')), None)} s.", ""]
    lines += ["## Image vs text-only", ""]
    for tag in "abcd":
        a, b = runs.get(f"{tag}:image"), runs.get(f"{tag}:text")
        if a and b:
            same = [c["program"] for c in a["candidates"]] == [c["program"] for c in b["candidates"]]
            lines.append(f"- {tag}: programs {'identical' if same else 'DIFFER'}; latency {a['latency_s']} s (image) vs {b['latency_s']} s (text); "
                         f"raw text {'identical' if a['raw_text'] == b['raw_text'] else 'differs'}")
    lines += ["", "## Per-instruction detail", ""]
    for k in sorted(runs):
        r = runs[k]
        lines += [f"### {k} — {'PASS' if r['pass'] else 'FAIL'}", "", f"Instruction: {r['instruction']}", "", f"Expected: {r['expectation']}", "",
                  f"Divergence(candidates, 0): prefix {r['divergence']['common_prefix']}, choices {r['divergence']['choices']}", ""]
        for i, c in enumerate(r["candidates"]):
            lines.append(f"- cand {i}: *{c['interpretation']}* -> {' ; '.join(c['program'])}")
        if r["rejections"]:
            lines.append(f"- rejections: {r['rejections']}")
        raw = (r["raw_text"] or "")
        lines += ["", "Raw output (trimmed to 600 chars):", "", "```", raw[:600] + ("…" if len(raw) > 600 else ""), "```", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
