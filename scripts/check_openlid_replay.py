#!/usr/bin/env python3
"""Can the frozen WAM reproduce the demonstrated `open lid(car)` upper-body motion IN distribution?

Offline replay on cached held-out (validation) demonstrations: at several points inside every validation
open-lid segment the real demo observation (head RGB + 61D proprio) and the training contract condition are
fed to the checkpoint and the 32-step chunk is compared with the demonstrated actions.  Motion intent is
measured as (action target - current joint position) so a copy-state hold scores zero.

Purpose: separate "the executor cannot produce the lid-lifting motion at all" from "the closed loop docks
outside the demo distribution".  It is an offline prediction diagnostic; it is not closed-loop task success.

Multi-view checkpoints (config `model.views`, e.g. head zed + left/right wrist realsense) get the SAME anchor frame
from every view cache, stacked as (1, V, 3, H, W) in config order, and are sampled through
`wrapper.sample_actions`.  Anchors, per-anchor seed and metrics are unchanged; every flow sample is checked to
consume exactly one (1, 32, 23) normal draw from the seeded generator, i.e. the same noise the single-view
path uses, so rows stay paired with the single-view source baseline.  Checkpoints without `model.views` run the
original single-view code path unchanged.  `--admit-against SOURCE_DIR` then writes `paired_admission.json`
(pre-registered gate, see scripts/compare_manipulation_predictions.py) into `--out`.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.policy_server import WamPolicy
from harness.wam_behavior import (ACTION_GROUPS, FRAMES, HEAD_CAMERA, LOWDIM, STATE_SLICES, SkillTable, hold_baseline,
                                  model_proprio_lags, normalize_image, stack_proprio_history)
from harness.frame_cache import FramePack
from harness.vtt_monitor import precondition_key

sys.path.insert(0, str(ROOT / "scripts"))
from compare_manipulation_predictions import admit  # noqa: E402

# action slice -> proprio position slice (absolute position targets in the demos)
POSITION_PAIRS = {"trunk": ((3, 7), STATE_SLICES["trunk_qpos"]),
                  "left_arm": ((7, 14), STATE_SLICES["arm_left_qpos"]),
                  "right_arm": ((15, 22), STATE_SLICES["arm_right_qpos"])}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else float("nan")


def file_sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_views(cfg, wrapper):
    """Camera keys of a multi-view checkpoint in model order; [] means the original single-view head path."""
    views = [str(v) for v in (cfg.model.get("views", None) or [])]
    declared = getattr(wrapper, "views", None)
    if views and declared is not None and [str(v) for v in declared] != views:
        raise ValueError(f"wrapper.views {list(declared)} != config model.views {views}")
    for v in views:
        if not (FRAMES / v).is_dir():
            raise FileNotFoundError(f"No frame cache for view {v}: {FRAMES / v}")
    return views


def view_pixels(packs, frame_index, size):
    """Same anchor frame from every view -> (V, 3, H, W), with the policy_server head preprocessing per view."""
    import cv2
    out = []
    for pack in packs:
        rgb = np.asarray(pack.rgb(frame_index))[..., :3]
        if rgb.shape[0] != size or rgb.shape[1] != size:
            rgb = cv2.resize(np.ascontiguousarray(rgb), (size, size), interpolation=cv2.INTER_AREA)
        out.append(normalize_image(rgb))
    return np.stack(out)


def sample_multiview(wrapper, gen, pixels, state, cond, samples, num_steps):
    """K-sample mean through `wrapper.sample_actions`, asserting each sample consumes exactly the single-view
    noise draw (one torch.randn(1, action_len, action_dim)) so the paired comparison sees identical noise."""
    px = torch.from_numpy(pixels)[None]
    st = torch.from_numpy(state)[None]
    shape = (1, int(wrapper.action_model.action_len), int(wrapper.action_min.numel()))
    preds = []
    for _ in range(samples):
        before = gen.get_state()
        out = wrapper.sample_actions(px, st, task_cond=cond, num_steps=num_steps, generator=gen)
        ref = torch.Generator(device=gen.device)
        ref.set_state(before)
        torch.randn(*shape, device=gen.device, generator=ref)
        if not torch.equal(ref.get_state(), gen.get_state()):
            raise RuntimeError("Multi-view sampler consumed the generator differently from the single-view path "
                               "(expected exactly one randn%s per sample); paired noise would differ" % (shape,))
        act = out[0] if isinstance(out, (tuple, list)) else out
        preds.append(act[0].float().cpu().numpy())
    return np.mean(preds, axis=0)


def self_test():
    """CPU-only: real validation frames from all three caches, stub sampler (no checkpoint, no GPU)."""
    import types
    from omegaconf import OmegaConf
    views = [HEAD_CAMERA, "observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0"]
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    ep = [e for e in sorted(table.episodes) if index["episodes"][str(e)]["split"] == "validation"][0]
    seg = [s for s in table.episodes[ep]["segments"] if precondition_key(s["skill"], s["objects"]) == "open lid(car)"][0]
    a = (int(seg["start"] + 0.5 * (seg["end"] - seg["start"] - 33)) + 1) // 2 * 2
    state = np.array(np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")[a], np.float32)
    cfg = OmegaConf.create({"model": {"views": views}, "dataset": {"image_size": [224, 224]}})

    class Stub:
        def __init__(self, draws=1):
            self.views, self.draws, self.seen = list(views), draws, []
            self.action_model = types.SimpleNamespace(action_len=32)
            self.action_min = torch.zeros(23)

        def sample_actions(self, pixel_values, state, task_cond=None, num_steps=10, generator=None):
            self.seen.append(pixel_values.shape)
            for _ in range(self.draws):
                x = torch.randn(pixel_values.shape[0], 32, 23, generator=generator)
            return x + pixel_values.mean(), None, None

    stub = Stub()
    assert checkpoint_views(cfg, stub) == views
    packs = [FramePack(FRAMES / v, ep) for v in views]
    pixels = view_pixels(packs, a, 224)
    assert pixels.shape == (3, 3, 224, 224) and pixels.dtype == np.float32
    # view 0 must be byte-identical to the single-view policy_server head preprocessing of the same anchor
    fake = types.SimpleNamespace(cfg=cfg, _find=WamPolicy._find)
    _, head = WamPolicy._preprocess(fake, {"r::proprio": state, "r::zed_link::rgb": packs[0].rgb(a)})
    assert np.array_equal(pixels[0], head), "head view differs from the single-view preprocessing"
    assert not np.array_equal(pixels[1], pixels[0]) and not np.array_equal(pixels[2], pixels[1]), "wrist views not distinct"
    # generator stream: K multi-view samples use exactly the noise of K single-view `_sample_once` draws
    k, seed = 8, 1000 * ep + a
    gen = torch.Generator().manual_seed(seed)
    pred = sample_multiview(stub, gen, pixels, state, None, k, 10)
    ref = torch.Generator().manual_seed(seed)
    expect = np.mean([(torch.randn(1, 32, 23, generator=ref) + torch.from_numpy(pixels)[None].mean())[0].numpy()
                      for _ in range(k)], axis=0)
    assert np.array_equal(pred, expect) and torch.equal(gen.get_state(), ref.get_state())
    assert stub.seen == [torch.Size([1, 3, 3, 224, 224])] * k
    try:
        sample_multiview(Stub(draws=2), torch.Generator().manual_seed(seed), pixels, state, None, 1, 10)
    except RuntimeError:
        pass
    else:
        raise AssertionError("extra generator consumption was not detected")
    bad = OmegaConf.create({"model": {"views": views[::-1]}})
    try:
        checkpoint_views(bad, stub)
    except ValueError:
        pass
    else:
        raise AssertionError("view-order mismatch was not detected")
    print(f"PASS (CPU stub): episode {ep} anchor {a}: (1,3,3,224,224) from 3 caches, head view == single-view "
          f"preprocessing, K={k} noise stream identical to single-view draws, extra draws / view order rejected")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint")
    ap.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    ap.add_argument("--key", default="open lid(car)")
    ap.add_argument("--split", default="validation", choices=("train", "validation"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fractions", default="0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--samples", type=int, default=1, help="average this many flow samples per anchor (deployment option)")
    ap.add_argument("--admit-against", type=Path, help="source replay dir; writes <out>/paired_admission.json")
    ap.add_argument("--self-test", action="store_true", help="CPU stub test of the multi-view loading path")
    ap.add_argument("--skip-missing", action="store_true",
                    help="skip (and record) episodes without the key instead of failing; same skip set for paired runs")
    ap.add_argument("--static-history", action="store_true",
                    help="copycat diagnostic for proprio-history checkpoints: every history frame = the current frame "
                         "(a robot that has not moved), same anchors/seeds")
    if "--self-test" in sys.argv[1:]:
        return self_test()
    args = ap.parse_args()
    if not args.checkpoint:
        ap.error("--checkpoint is required")
    if args.admit_against is not None and not (args.admit_against / "rows.json").is_file():
        ap.error(f"--admit-against has no rows.json: {args.admit_against}")
    if args.samples < 1:
        ap.error("--samples must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    targets = json.loads(Path(args.targets).read_text())
    vector = torch.tensor(targets["contracts"][args.key]["vector"], dtype=torch.float32)[None]
    skill = args.key.partition("(")[0]
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    episodes = [ep for ep, row in sorted(table.episodes.items()) if index["episodes"][str(ep)]["split"] == args.split]
    episodes = episodes[:args.max_episodes]
    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device=args.device,
                       cuda_fraction=0.4, seed=0, log_dir=args.out,
                       orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})
    views = checkpoint_views(policy.cfg, policy.wrapper)
    lags = model_proprio_lags(policy.cfg)   # [0] for single-frame checkpoints (legacy path, bit-identical)
    if args.static_history and lags == [0]:
        ap.error("--static-history needs a proprio-history checkpoint (model.proprio_lags)")
    image_size = int(policy.cfg.dataset.image_size[0])
    rng_checks = 0
    fractions = [float(x) for x in args.fractions.split(",")]
    if not fractions or not all(np.isfinite(f) and 0 <= f <= 1 for f in fractions):
        ap.error("fractions must be finite values in [0,1]")
    rows, skipped = [], []
    started = time.monotonic()
    for ep in episodes:
        segs = [s for s in table.episodes[ep]["segments"] if precondition_key(s["skill"], s["objects"]) == args.key]
        if not segs and args.skip_missing:
            skipped.append(ep)
            continue
        if len(segs) != 1:
            raise ValueError(f"Expected one exact {args.key} segment in episode {ep}, found {len(segs)}")
        seg = segs[0]
        action = np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r")
        state_all = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        if views:
            packs = [FramePack(FRAMES / v, ep) for v in views]
        else:
            pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        lo, hi = seg["start"], seg["end"]
        for frac in fractions:
            a = int(lo + frac * (hi - lo - 33))
            a = (a + 1) // 2 * 2
            if not lo <= a or a + 32 >= hi:
                raise ValueError(f"Anchor/future crosses the evaluated segment: episode={ep}, anchor={a}, segment={lo, hi}")
            state = np.array(state_all[a], np.float32)   # current frame: hold baseline, intents, preprocessing
            if lags == [0]:
                state_in = state
            elif args.static_history:
                state_in = np.repeat(state[None], len(lags), axis=0)
            else:   # the training / policy-server rule: frames a-lag clamped to the episode's frame 0
                state_in = stack_proprio_history(state_all, a, lags)
            demo = np.array(action[a:a + 32], np.float32)
            if views:
                pixels = view_pixels(packs, a, image_size)
                policy.gen.manual_seed(1000 * ep + a)
                pred = sample_multiview(policy.wrapper, policy.gen, pixels, state_in, vector, args.samples, policy.num_steps)
                rng_checks += args.samples
            else:
                rgb = pack.rgb(a)
                _, pixels = policy._preprocess({"r::proprio": state, "r::zed_link::rgb": rgb})
                feats, _ = policy._encode(pixels)
                policy.gen.manual_seed(1000 * ep + a)
                pred = np.mean([policy._sample(feats, state_in, vector) for _ in range(args.samples)], axis=0)
            hold = hold_baseline(torch.from_numpy(state)[None])[0].numpy()
            row = {"episode": ep, "anchor": a, "fraction": frac, "segment": [lo, hi], "groups": {}}
            for name, (alo, ahi) in ACTION_GROUPS.items():
                row["groups"][name] = {"mae_pred": float(np.abs(pred[:, alo:ahi] - demo[:, alo:ahi]).mean()),
                                       "mae_hold": float(np.abs(hold[:, alo:ahi] - demo[:, alo:ahi]).mean())}
            for name, ((alo, ahi), (slo, shi)) in POSITION_PAIRS.items():
                q = state[slo:shi]
                d_demo = demo[-1, alo:ahi] - q
                d_pred = pred[-1, alo:ahi] - q
                row["groups"][name].update({"intent_norm_demo": float(np.linalg.norm(d_demo)),
                                            "intent_norm_pred": float(np.linalg.norm(d_pred)),
                                            "intent_cos": cosine(d_demo, d_pred)})
            row["trunk_state"] = state[STATE_SLICES["trunk_qpos"][0]:STATE_SLICES["trunk_qpos"][1]].tolist()
            row["trunk_demo_end"] = demo[-1, 3:7].tolist()
            row["trunk_pred_end"] = pred[-1, 3:7].tolist()
            row["base_pred_mean"] = pred[:, :3].mean(0).tolist()
            row["base_demo_mean"] = demo[:, :3].mean(0).tolist()
            rows.append(row)
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": policy.checkpoint_step, "key": args.key, "split": args.split,
               "samples_averaged": args.samples, "action_repr": policy.wrapper.action_repr_name,
               "n_anchors": len(rows), "episodes": episodes, "seconds": time.monotonic() - started,
               "condition": "training contract vector (train-only)", "closed_loop": False, "task_success_evaluated": False,
               "groups": {}, "by_fraction": {},
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), Path(args.targets), LOWDIM / "index.json")},
               "sampler_seed": "1000 * episode + anchor, reset per anchor", "fractions": fractions,
               "checkpoint_sha256": checkpoint_sha256, "views": views or [HEAD_CAMERA], "multiview": bool(views),
               "num_flow_steps": policy.num_steps, "skipped_episodes": skipped, "proprio_lags": lags, "static_history": bool(args.static_history),
               "multiview_rng_contract": ({"samples_checked": rng_checks, "rule": "each sample consumes exactly one "
                                           "randn(1, action_len, action_dim), identical to the single-view path"}
                                          if views else None)}
    for name in ACTION_GROUPS:
        g = summary["groups"][name] = {"mae_pred": float(np.mean([r["groups"][name]["mae_pred"] for r in rows])),
                                       "mae_hold": float(np.mean([r["groups"][name]["mae_hold"] for r in rows]))}
        if name in POSITION_PAIRS:
            cos = np.array([r["groups"][name]["intent_cos"] for r in rows])
            g.update({"intent_norm_demo": float(np.mean([r["groups"][name]["intent_norm_demo"] for r in rows])),
                      "intent_norm_pred": float(np.mean([r["groups"][name]["intent_norm_pred"] for r in rows])),
                      "intent_cos_mean": float(np.nanmean(cos)), "intent_cos_frac_positive": float(np.nanmean(cos > 0))})
    for frac in fractions:
        rs = [r for r in rows if r["fraction"] == frac]
        summary["by_fraction"][str(frac)] = {name: {"mae_pred": float(np.mean([r["groups"][name]["mae_pred"] for r in rs])),
                                                    "mae_hold": float(np.mean([r["groups"][name]["mae_hold"] for r in rs])),
                                                    "intent_cos": float(np.nanmean([r["groups"][name]["intent_cos"] for r in rs]))}
                                             for name in POSITION_PAIRS}
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    policy.trace.close()
    for path in args.out.glob("policy_trace_*.jsonl"):
        if path.stat().st_size == 0:
            path.unlink()
    print(json.dumps({k: v for k, v in summary.items() if k != "by_fraction"}, indent=2))
    for frac, g in summary["by_fraction"].items():
        print(frac, {n: (round(v["mae_pred"], 4), round(v["mae_hold"], 4), round(v["intent_cos"], 2)) for n, v in g.items()})
    if args.admit_against is not None:
        report = admit(args.admit_against, args.out, args.out / "paired_admission.json")
        print("paired admission:", "PASS" if report["gate"]["passed"] else "FAIL", args.out / "paired_admission.json")


if __name__ == "__main__":
    main()
