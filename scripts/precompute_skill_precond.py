#!/usr/bin/env python3
"""Next-skill precondition prototypes from the frozen DINOv3 encoder (no training), and their evaluation as
an "arrived" predicate for navigation segments ("move to X"), whose completion VTT closure cannot judge.

Definition (docs/system_design.md §3.6): for every program step with key "skill(obj1,obj2)" (same short object
names as configs/task013_skill_program.json), take the head-camera CLS of the segment START frame;
q_key = mean over TRAIN episodes. Navigation "move to X" is finished when CLS(t) is nearest to q of the NEXT
manipulation step and cos >= tau[key]; tau is fitted on VALIDATION navigation segments (<=5% false arrival in
the first half of the segment, >=80% caught by the end) and additionally checked 2-fold by episode.

Encoder recipe is identical to scripts/precompute_skill_vtt.py (last_hidden_state CLS, 224 px cached frames,
ImageNet normalisation) but runs on CPU in float32 (the GPU is owned by training).

Outputs: data/skill_vtt_task013/precond_prototypes.npy (K x 1024 float32), precond_meta.json,
         precond_cls_cache.npz (CLS of every frame embedded here, so re-runs are free),
         outputs/validation/nav_precondition_eval.json + .md
"""
import argparse, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import DATA, FRAMES, VTT_DIR, HEAD_CAMERA, SkillTable, normalize_image, load_upstream  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402
from harness.vtt_monitor import precondition_key  # noqa: E402

NAV = "move to"
OUT_DIR = ROOT / "outputs/validation"


def grid_first(frame, stride, L):
    return min(((frame + stride - 1) // stride) * stride, L - 1)


def grid_last(end, stride, L):
    return min(((end - 1) // stride) * stride, L - 1)


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def spearman(y):
    """Spearman rho of y against its index."""
    n = len(y)
    r = np.argsort(np.argsort(y))
    x = np.arange(n)
    return float(np.corrcoef(x, r)[0, 1]) if n > 2 and np.ptp(r) > 0 else 0.0


class Embedder:
    def __init__(self, dino, batch, cache_path):
        import torch
        self.torch = torch
        self.batch = batch
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            z = np.load(cache_path)
            self.cache = {str(k): v for k, v in zip(z["keys"], z["cls"])}
        self.enc = None
        self.dino = dino

    def _load(self):
        if self.enc is None:
            _, runner = load_upstream()
            self.enc, self.hidden, _, _ = runner.ModelFactory.create_vision_encoder(str(ROOT / self.dino), self.torch.float32, "cpu")

    def __call__(self, pairs, log_every=100):
        """pairs: list of (episode, frame). Returns (N, D) float32 CLS in order; embeds only uncached frames."""
        todo = sorted({p for p in pairs if f"{p[0]}:{p[1]}" not in self.cache})
        if todo:
            self._load()
            print(f"[embed] {len(todo)} frames to embed ({len(pairs) - len(todo)} cached)", flush=True)
            t0 = time.time(); done = 0; pack = None; pack_ep = None; buf = []; ids = []

            def flush():
                px = self.torch.from_numpy(np.stack(buf))
                with self.torch.no_grad():
                    out = self.enc(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :].float().numpy()
                for (ep, fr), v in zip(ids, out):
                    self.cache[f"{ep}:{fr}"] = v.astype(np.float32)

            for ep, fr in todo:
                if pack_ep != ep:
                    pack, pack_ep = FramePack(FRAMES / HEAD_CAMERA, ep), ep
                buf.append(normalize_image(pack.rgb(fr))); ids.append((ep, fr))
                if len(buf) == self.batch:
                    flush(); done += len(buf); buf, ids = [], []
                    if done % log_every < self.batch:
                        el = time.time() - t0
                        print(f"[embed] {done}/{len(todo)} {el:.0f}s ({el / done:.2f} s/img, eta {el / done * (len(todo) - done) / 60:.1f} min)", flush=True)
            if buf:
                flush(); done += len(buf)
            print(f"[embed] done {done} frames in {time.time() - t0:.0f}s", flush=True)
            keys = np.array(sorted(self.cache), dtype=str)
            np.savez(self.cache_path, keys=keys, cls=np.stack([self.cache[k] for k in keys]))
        return np.stack([self.cache[f"{ep}:{fr}"] for ep, fr in pairs]).astype(np.float32)


def fit_tau(curves, top1, fracs, fa_max, catch_min, grid):
    """curves: (n, T) cos(CLS_t, q_next); top1: (n, T) bool q_next is nearest. Predicate(t) = top1 & cos>=tau.
    Returns dict with the smallest tau whose early false-arrival rate <= fa_max, its catch rate, and usability."""
    early = fracs <= 0.5
    late = ~early
    rows = []
    for tau in grid:
        pred = top1 & (curves >= tau)
        fa = float(pred[:, early].any(1).mean())
        catch = float(pred[:, late].any(1).mean())
        catch_end = float(pred[:, -1].mean())
        rows.append((float(tau), fa, catch, catch_end))
    ok = [r for r in rows if r[1] <= fa_max]
    best = min(ok, key=lambda r: r[0]) if ok else None  # smallest tau => largest catch under the FA constraint
    if best is None:
        return {"tau": None, "usable": False, "false_arrival_first_half": None, "catch_by_end": None, "catch_at_end": None, "sweep": rows}
    return {"tau": best[0], "usable": bool(best[2] >= catch_min), "false_arrival_first_half": best[1],
            "catch_by_end": best[2], "catch_at_end": best[3], "sweep": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dino", default="models/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--nav-points", type=int, default=10)
    ap.add_argument("--max-nav", type=int, default=600)
    ap.add_argument("--min-train", type=int, default=5, help="keys with fewer train segments get no prototype")
    ap.add_argument("--fa-max", type=float, default=0.05)
    ap.add_argument("--catch-min", type=float, default=0.80)
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    t_start = time.time()

    skills = SkillTable()
    # ---- jobs -------------------------------------------------------------------------------------------------
    manip = []   # (ep, seg_i, frame, key, split)
    nav = []     # (ep, seg_i, [frames], next_key, split, n_frames_in_segment)
    fracs = np.linspace(0.0, 1.0, args.nav_points)
    for ep, info in sorted(skills.episodes.items()):
        L, sp, segs = info["length"], info["split"], info["segments"]
        for i, s in enumerate(segs):
            key = precondition_key(s["skill"], s["objects"])
            if s["skill"] != NAV:
                manip.append((ep, i, grid_first(s["start"], args.stride, L), key, sp))
            elif sp == "validation":
                nxt = next((segs[j] for j in range(i + 1, len(segs)) if segs[j]["skill"] != NAV), None)
                if nxt is None:
                    continue
                a, b = grid_first(s["start"], args.stride, L), grid_last(s["end"], args.stride, L)
                frames = [int(round(a + f * (b - a)) // args.stride * args.stride) for f in fracs]  # keep T fixed
                nav.append((ep, i, frames, precondition_key(nxt["skill"], nxt["objects"]), sp, s["end"] - s["start"]))
    if len(nav) > args.max_nav:
        sel = np.linspace(0, len(nav) - 1, args.max_nav).round().astype(int)
        nav = [nav[k] for k in sel]
    print(f"manipulation starts: {len(manip)} (train {sum(m[4] == 'train' for m in manip)}), validation navigation segments: {len(nav)} x {args.nav_points} frames", flush=True)

    embed = Embedder(args.dino, args.batch, VTT_DIR / "precond_cls_cache.npz")
    pairs = [(m[0], m[2]) for m in manip] + [(n[0], f) for n in nav for f in n[2]]
    cls = embed(pairs)
    cls_manip = cls[:len(manip)]
    cls_nav = cls[len(manip):].reshape(len(nav), args.nav_points, -1)
    hidden = cls.shape[1]

    # ---- prototypes (train only) ------------------------------------------------------------------------------
    train_counts = defaultdict(int); val_counts = defaultdict(int)
    for m in manip:
        (train_counts if m[4] == "train" else val_counts)[m[3]] += 1
    keys = sorted(k for k, c in train_counts.items() if c >= args.min_train)
    kidx = {k: i for i, k in enumerate(keys)}
    proto = np.zeros((len(keys), hidden), dtype=np.float64)
    for m, v in zip(manip, cls_manip):
        if m[4] == "train" and m[3] in kidx:
            proto[kidx[m[3]]] += v
    proto /= np.array([train_counts[k] for k in keys])[:, None]
    train_mask = np.array([m[4] == "train" for m in manip])
    gmean = cls_manip[train_mask].astype(np.float64).mean(0)
    excluded = {k: c for k, c in train_counts.items() if c < args.min_train}

    def sims(x, centred):
        mu = gmean if centred else 0.0
        return unit(np.asarray(x, dtype=np.float64) - mu) @ unit(proto - mu).T

    # ---- eval 1: nearest prototype accuracy of start-CLS ------------------------------------------------------
    eval1 = {}
    for centred in (False, True):
        res = {}
        for sp in ("train", "validation"):
            sel = [k for k, m in enumerate(manip) if m[4] == sp]
            S = sims(cls_manip[sel], centred)
            pred = S.argmax(1)
            truth = np.array([kidx.get(manip[k][3], -1) for k in sel])
            correct = pred == truth
            per_key = {}
            for k in keys:
                mk = truth == kidx[k]
                if mk.any():
                    per_key[k] = {"n": int(mk.sum()), "recall": float(correct[mk].mean()), "mean_cos_own": float(S[mk, kidx[k]].mean())}
            conf = np.zeros((len(keys), len(keys)), dtype=int)
            for p, t in zip(pred, truth):
                if t >= 0:
                    conf[t, p] += 1
            res[sp] = {"n": len(sel), "n_without_prototype": int((truth < 0).sum()), "accuracy": float(correct.mean()),
                       "per_key": per_key, "confusion_rows_true_cols_pred": conf.tolist()}
        res["prototype_cosine_matrix"] = sims(proto, centred).round(3).tolist()
        eval1["centred" if centred else "raw"] = res

    # ---- eval 2: navigation end vs next-skill prototype --------------------------------------------------------
    nav_eps = sorted({n[0] for n in nav})
    fold = {ep: i % 2 for i, ep in enumerate(nav_eps)}
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    eval2 = {}
    for centred in (False, True):
        S = sims(cls_nav.reshape(-1, hidden), centred).reshape(len(nav), args.nav_points, len(keys))
        rows = [(k, n) for k, n in enumerate(nav) if n[3] in kidx]
        skipped = [n[3] for n in nav if n[3] not in kidx]
        curves = np.stack([S[k, :, kidx[n[3]]] for k, n in rows])            # (n, T) cos to q_next
        top1 = np.stack([S[k].argmax(1) == kidx[n[3]] for k, n in rows])      # (n, T)
        best_other = np.stack([np.delete(S[k], kidx[n[3]], axis=1).max(1) for k, n in rows])
        c_start, c_end = curves[:, 0], curves[:, -1]
        d = c_end - c_start
        per_key = {}
        for key in keys:
            idx = [i for i, (k, n) in enumerate(rows) if n[3] == key]
            if not idx:
                continue
            idx = np.array(idx)
            fit_all = fit_tau(curves[idx], top1[idx], fracs, args.fa_max, args.catch_min, grid)
            # 2-fold by episode: fit tau on one half, apply to the other
            held = {"n": 0, "false_arrival_first_half": 0, "catch_by_end": 0}
            for f in (0, 1):
                fit_i = np.array([i for i in idx if fold[rows[i][1][0]] != f]); test_i = np.array([i for i in idx if fold[rows[i][1][0]] == f])
                if len(fit_i) == 0 or len(test_i) == 0:
                    continue
                ft = fit_tau(curves[fit_i], top1[fit_i], fracs, args.fa_max, args.catch_min, grid)
                if ft["tau"] is None:
                    continue
                pred = top1[test_i] & (curves[test_i] >= ft["tau"])
                held["n"] += len(test_i); held["false_arrival_first_half"] += int(pred[:, fracs <= 0.5].any(1).sum()); held["catch_by_end"] += int(pred[:, fracs > 0.5].any(1).sum())
            if held["n"]:
                held = {"n": held["n"], "false_arrival_first_half": held["false_arrival_first_half"] / held["n"], "catch_by_end": held["catch_by_end"] / held["n"]}
            crossing = []
            if fit_all["tau"] is not None:
                for i in idx:
                    hit = np.flatnonzero(top1[i] & (curves[i] >= fit_all["tau"]))
                    crossing.append(float(fracs[hit[0]]) if hit.size else None)
            rho = [spearman(curves[i]) for i in idx]
            per_key[key] = {
                "n_nav_segments": int(len(idx)),
                "cos_end_next_mean": float(c_end[idx].mean()), "cos_start_next_mean": float(c_start[idx].mean()),
                "frac_end_gt_start": float((d[idx] > 0).mean()), "frac_end_gt_start_by_margin": float((d[idx] > args.margin).mean()),
                "top1_at_end": float(top1[idx, -1].mean()), "top1_at_start": float(top1[idx, 0].mean()),
                "margin_over_best_other_at_end_mean": float((c_end[idx] - best_other[idx, -1]).mean()),
                "cos_end_best_other_mean": float(best_other[idx, -1].mean()),
                "time_course_mean_cos": curves[idx].mean(0).round(4).tolist(),
                "time_course_top1_rate": top1[idx].mean(0).round(3).tolist(),
                "spearman_rho_mean": float(np.mean(rho)), "frac_rho_gt_0.5": float(np.mean(np.array(rho) > 0.5)),
                "frac_argmax_in_last_two": float(np.mean([curves[i].argmax() >= args.nav_points - 2 for i in idx])),
                "tau": fit_all["tau"], "tau_usable": fit_all["usable"],
                "false_arrival_first_half": fit_all["false_arrival_first_half"], "catch_by_end": fit_all["catch_by_end"], "catch_at_end": fit_all["catch_at_end"],
                "held_out_2fold": held if held.get("n") else None,
                "crossing_fraction_median": (float(np.median([c for c in crossing if c is not None])) if any(c is not None for c in crossing) else None),
                "crossing_fractions": crossing,
            }
        usable = [k for k, v in per_key.items() if v["tau_usable"]]
        n_us = sum(per_key[k]["n_nav_segments"] for k in usable)
        eval2["centred" if centred else "raw"] = {
            "n_nav_segments": int(len(rows)), "n_skipped_no_prototype": len(skipped), "skipped_next_keys": sorted(set(skipped)),
            "cos_end_next_mean": float(c_end.mean()), "cos_start_next_mean": float(c_start.mean()),
            "delta_mean": float(d.mean()), "frac_end_gt_start": float((d > 0).mean()), "frac_end_gt_start_by_margin": float((d > args.margin).mean()),
            "cos_end_best_other_mean": float(best_other[:, -1].mean()), "top1_at_end": float(top1[:, -1].mean()), "top1_at_start": float(top1[:, 0].mean()),
            "time_course_mean_cos": curves.mean(0).round(4).tolist(), "time_course_top1_rate": top1.mean(0).round(3).tolist(),
            "spearman_rho_mean": float(np.mean([spearman(c) for c in curves])),
            "keys_usable": usable, "keys_total": len(per_key),
            "nav_segments_covered_by_usable_keys": n_us, "coverage_of_nav_segments": float(n_us / max(len(rows), 1)),
            "per_key": per_key,
        }
    # pick the scoring mode by the use case: most usable keys, then coverage
    mode = max(("raw", "centred"), key=lambda m: (len(eval2[m]["keys_usable"]), eval2[m]["coverage_of_nav_segments"], eval2[m]["top1_at_end"]))
    centred = mode == "centred"
    tau = {k: (eval2[mode]["per_key"].get(k, {}).get("tau") if eval2[mode]["per_key"].get(k, {}).get("tau_usable") else None) for k in keys}

    # ---- save ----------------------------------------------------------------------------------------------------
    VTT_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VTT_DIR / "precond_prototypes.npy", proto.astype(np.float32))
    seconds = round(time.time() - t_start, 1)
    meta = {
        "dino": args.dino,
        "dino_sha256": json.loads((ROOT / args.dino / "DOWNLOAD_PROVENANCE.json").read_text())["files"]["model.safetensors"]["sha256"],
        "encoder_dtype_device": "float32 cpu (prototypes.npy of the VTTs were bfloat16 cuda)",
        "frame_stride": args.stride, "image_size": 224, "camera": HEAD_CAMERA,
        "feature": "last_hidden_state CLS of the segment START frame (upstream precompute_task_cond recipe)",
        "key_rule": "skill(canonical objects); canonical_object strips '_<n>' then '_<6 letters>' (harness.vtt_monitor.precondition_key)",
        "keys": keys, "train_counts": [train_counts[k] for k in keys], "validation_counts": [val_counts.get(k, 0) for k in keys],
        "excluded_keys_train_count_below_min": excluded, "min_train": args.min_train,
        "prototypes_use_validation": False,
        "centred": centred, "global_mean": gmean.astype(np.float32).tolist(),
        "global_mean_note": "mean start-CLS over all TRAIN manipulation segments; subtracted before cosine iff centred",
        "tau": tau, "tau_rule": f"smallest tau with <= {args.fa_max:.0%} false arrival in first 50% of validation navigation segments; "
                                 f"usable only if >= {args.catch_min:.0%} of segments are caught by 100%; predicate also requires q_next to be top-1",
        "tau_fitted_on": "validation navigation segments (not held out from the numbers in nav_precondition_eval.json; see held_out_2fold there)",
        "skill_segments_sha256": hashlib.sha256((DATA / "skill_segments_task013.json").read_bytes()).hexdigest(),
        "seconds": seconds,
    }
    (VTT_DIR / "precond_meta.json").write_text(json.dumps(meta, indent=1))
    report = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": seconds,
              "settings": vars(args), "n_manip_train": int(train_mask.sum()), "n_manip_validation": int((~train_mask).sum()),
              "n_nav_validation": len(nav), "nav_points_fractions": fracs.round(3).tolist(), "chosen_mode": mode,
              "keys": keys, "train_counts": meta["train_counts"], "eval1_nearest_prototype": eval1, "eval2_navigation": eval2,
              "note": "Offline validation-demo evidence for the next-skill precondition predicate; not a policy or robot result."}
    (OUT_DIR / "nav_precondition_eval.json").write_text(json.dumps(report, indent=1))
    (OUT_DIR / "nav_precondition_eval.md").write_text(render_md(report))
    print(json.dumps({"keys": len(keys), "mode": mode, "eval1_val_acc_raw": eval1["raw"]["validation"]["accuracy"],
                      "eval1_val_acc_centred": eval1["centred"]["validation"]["accuracy"],
                      "nav_top1_end": eval2[mode]["top1_at_end"], "nav_top1_start": eval2[mode]["top1_at_start"],
                      "cos_end": eval2[mode]["cos_end_next_mean"], "cos_start": eval2[mode]["cos_start_next_mean"],
                      "usable_keys": eval2[mode]["keys_usable"], "coverage": eval2[mode]["coverage_of_nav_segments"], "seconds": seconds}, indent=1), flush=True)


def render_md(r):
    L = [f"# Navigation completion via next-skill precondition (Task 13, validation demos)", "",
         f"Recorded {r['recorded_at']}, {r['seconds']} s CPU. Prototypes q_key = mean start-frame CLS over TRAIN segments "
         f"({r['n_manip_train']} manipulation segments, {len(r['keys'])} keys); evaluated on {r['n_manip_validation']} validation "
         f"manipulation starts and {r['n_nav_validation']} validation navigation segments x {len(r['nav_points_fractions'])} frames. "
         f"Chosen scoring mode: **{r['chosen_mode']}**.", ""]
    L += ["## 1. Nearest-prototype accuracy of start-CLS (all manipulation keys)", "", "| mode | split | n | accuracy | n without prototype |", "|---|---|---|---|---|"]
    for mode in ("raw", "centred"):
        for sp in ("train", "validation"):
            e = r["eval1_nearest_prototype"][mode][sp]
            L.append(f"| {mode} | {sp} | {e['n']} | {e['accuracy']:.3f} | {e['n_without_prototype']} |")
    L += ["", "Per-key validation recall (raw / centred):", "", "| key | n | recall raw | recall centred | mean cos own raw |", "|---|---|---|---|---|"]
    pk_r = r["eval1_nearest_prototype"]["raw"]["validation"]["per_key"]; pk_c = r["eval1_nearest_prototype"]["centred"]["validation"]["per_key"]
    for k in r["keys"]:
        if k in pk_r:
            L.append(f"| {k} | {pk_r[k]['n']} | {pk_r[k]['recall']:.2f} | {pk_c[k]['recall']:.2f} | {pk_r[k]['mean_cos_own']:.3f} |")
    L += ["", "## 2. Navigation segments: is CLS(end) close to q_next and far from CLS(start)?", ""]
    for mode in ("raw", "centred"):
        e = r["eval2_navigation"][mode]
        L += [f"### mode = {mode}", "",
              f"- n = {e['n_nav_segments']} navigation segments (skipped {e['n_skipped_no_prototype']} whose next key has no prototype: {e['skipped_next_keys']})",
              f"- cos(CLS_end, q_next) = {e['cos_end_next_mean']:.3f} vs cos(CLS_start, q_next) = {e['cos_start_next_mean']:.3f}; "
              f"end > start in {e['frac_end_gt_start']:.1%}, by margin {r['settings']['margin']} in {e['frac_end_gt_start_by_margin']:.1%}",
              f"- cos(CLS_end, best other q) = {e['cos_end_best_other_mean']:.3f}; q_next is top-1 at end in {e['top1_at_end']:.1%} (at start: {e['top1_at_start']:.1%})",
              f"- time course of mean cos(CLS_t, q_next) at fractions {r['nav_points_fractions']}: {e['time_course_mean_cos']}",
              f"- time course of top-1 rate: {e['time_course_top1_rate']}; mean Spearman rho(cos, t) = {e['spearman_rho_mean']:.2f}",
              f"- usable keys ({len(e['keys_usable'])}/{e['keys_total']}): {e['keys_usable']}; they cover {e['coverage_of_nav_segments']:.1%} of navigation segments", "",
              "| next key | n | cos end | cos start | end>start+m | top1 end | top1 start | rho | tau | FA first half | catch by end | held-out FA / catch | median crossing |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in e["per_key"].items():
            ho = v["held_out_2fold"]; ho_s = f"{ho['false_arrival_first_half']:.2f} / {ho['catch_by_end']:.2f}" if ho else "-"
            tau_s = f"{v['tau']:.2f}{'' if v['tau_usable'] else ' (unusable)'}" if v["tau"] is not None else "-"
            fa = f"{v['false_arrival_first_half']:.2f}" if v["false_arrival_first_half"] is not None else "-"
            cb = f"{v['catch_by_end']:.2f}" if v["catch_by_end"] is not None else "-"
            cr = f"{v['crossing_fraction_median']:.2f}" if v["crossing_fraction_median"] is not None else "-"
            L.append(f"| {k} | {v['n_nav_segments']} | {v['cos_end_next_mean']:.3f} | {v['cos_start_next_mean']:.3f} | {v['frac_end_gt_start_by_margin']:.2f} | "
                     f"{v['top1_at_end']:.2f} | {v['top1_at_start']:.2f} | {v['spearman_rho_mean']:.2f} | {tau_s} | {fa} | {cb} | {ho_s} | {cr} |")
        L.append("")
    L += ["Definitions: FA first half = fraction of segments where the predicate (top-1 and cos >= tau) fires at any sampled frame with t <= 50%; "
          "catch by end = fires at any frame with t > 50%; held-out = tau fitted on half the validation episodes, rates on the other half. "
          "tau is fitted on validation, so the non-held-out rates are optimistic.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
