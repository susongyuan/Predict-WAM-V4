#!/usr/bin/env python3
"""Offline evaluation of GENERIC (colour-free, demo-only) arrival scores for the navigation -> manipulation
hand-off, benchmarked on the `move to(car) -> open lid(car)` boundary against
  (i)  the deployed CLS prototype-cosine gate (harness/vtt_monitor.py PreconditionMonitor.arrived), and
  (ii) the instance-specific HSV red-car dock_score (harness/docking.py, AUC 0.891 in dock_target_20260923).

All candidate scores use only the frozen DINOv3 features the policy already computes (harness/policy_server.py
WamPolicy._encode: last_hidden_state CLS + hidden-state patch tokens) and prototypes built from TRAIN demo frames.
No simulator is launched and no ground truth is read (only cached demo head RGB + closed-loop head PNGs).

Data protocol (same as scripts/validate_dock_appearance_offline.py, extended):
  positives        : head frames at fractions 0, 0.05, 0.1 into the first `open lid` segment of each demo episode
                     (train split -> prototypes / classifier fit, validation split -> held-out positives)
  demo hard negs   : frames of the SAME episode's preceding `move to(car)` segment 3 s, 6 s, 10 s (90/180/300 frames)
                     before the open-lid start: approaching but not yet docked.  Needs no closed-loop data at all,
                     which is what makes the gate transferable to every skill boundary of every task.
  demo pre-1s      : frame 30 before the open-lid start (reported as its own diagnostic group, not a positive)
  recorded mis-dock: closed-loop observation PNGs of the three exec32 runs that docked obliquely (--misdock-globs)
  closed-loop good : observation PNGs of the two dockgate runs at steps inside the open-lid phase (from the
                     policy_trace orch_contract_start/end events); these should score as arrived.

Candidate scores:
  A  cls_cos              cosine(last-layer CLS, mean train-positive CLS)                       [deployed gate]
  A' cls_cos_deployed     cosine against the shipped precond_prototypes.npy row (tau = 0.83)      [as deployed]
  B  patch_spatial_cos    mean over the 14x14 grid of cosine(patch_i, per-position mean prototype_i), last layer
  C  patch_spatial_cos_mid same as B on hidden_states[-4] (a layer the policy already feeds the action model)
  D  patch_nn_cos         max over train positives of the spatial patch-cosine to that frame (nearest neighbour)
  E  patch_logreg         torch logistic regression on the 196-dim per-position cosine map of B,
                          fitted on train positives vs train demo hard negatives (E_mid: same on C's map)
  H  hsv_dock_score       harness.docking.car_dock_feature score (NaN/undetected -> -1, i.e. "not arrived")
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, VTT_DIR, SkillTable, normalize_image, load_upstream  # noqa: E402
from harness.frame_cache import FramePack  # noqa: E402
from harness.docking import car_dock_feature, DOCK_SCORE_ARRIVED  # noqa: E402

FPS = 30
OPEN_LID_KEY = "open lid(car)"
DEPLOYED_TAU = 0.83


# ----------------------------------------------------------------------------------------------------------------------
# frame selection
# ----------------------------------------------------------------------------------------------------------------------
def align(frame, stride, length):
    return int(min(max((frame // stride) * stride, 0), ((length - 1) // stride) * stride))


def collect_demo_jobs(table, index, anchors, neg_seconds, pre_seconds, max_episodes):
    """Returns list of dict(episode, split, frame, group, offset_s)."""
    jobs, n_ep = [], 0
    for ep in sorted(table.episodes):
        info = table.episodes[ep]
        split = index["episodes"][str(ep)]["split"]
        segs = info["segments"]
        oi = next((i for i, s in enumerate(segs) if s["skill"] == "open lid"), None)
        if oi is None:
            continue
        try:
            pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        except Exception:
            continue
        seg = segs[oi]
        lo, hi, L = seg["start"], seg["end"], pack.length
        prev = segs[oi - 1] if oi > 0 and segs[oi - 1]["skill"] == "move to" else None
        for fr in anchors:
            a = align(int(lo + fr * (hi - lo)), pack.stride, L)
            jobs.append({"episode": ep, "split": split, "frame": a, "group": "demo_pos", "frac": fr, "offset_s": 0.0})
        if prev is not None:
            for sec in pre_seconds:
                a = align(lo - int(sec * FPS), pack.stride, L)
                if a >= prev["start"]:
                    jobs.append({"episode": ep, "split": split, "frame": a, "group": "demo_pre1s", "offset_s": -sec})
            for sec in neg_seconds:
                a = align(lo - int(sec * FPS), pack.stride, L)
                if a >= prev["start"]:
                    jobs.append({"episode": ep, "split": split, "frame": a, "group": "demo_hardneg", "offset_s": -sec})
        n_ep += 1
        if max_episodes and n_ep >= max_episodes:
            break
    return jobs


def openlid_step_ranges(run_dir):
    """[(start, end)] step ranges of `open lid(car)` contracts from the policy trace (base is held inside them)."""
    ranges, open_start = [], None
    for tr in sorted(Path(run_dir).glob("policy_trace_*.jsonl")):
        for line in tr.read_text().splitlines():
            if '"orch_contract' not in line:
                continue
            r = json.loads(line)
            if r.get("key") != OPEN_LID_KEY:
                continue
            if r["event"] == "orch_contract_start":
                open_start = int(r["step"])
            elif r["event"] == "orch_contract_end" and open_start is not None:
                ranges.append((open_start, int(r["step"])))
                open_start = None
    if open_start is not None:
        ranges.append((open_start, 10 ** 9))
    return ranges


def collect_png_jobs(misdock_globs, gooddock_runs):
    jobs = []
    for g in misdock_globs:
        for p in sorted(glob.glob(str(ROOT / g))):
            jobs.append({"file": str(Path(p).relative_to(ROOT)), "group": "misdock"})
    for run in gooddock_runs:
        rd = ROOT / run
        ranges = openlid_step_ranges(rd)
        for p in sorted((rd / "observations").glob("*.png")):
            step = int(p.stem.split("step")[-1])
            inside = any(a <= step <= b for a, b in ranges)
            jobs.append({"file": str(p.relative_to(ROOT)), "group": "cl_gooddock" if inside else "cl_approach",
                         "step": step, "openlid_ranges": ranges})
    return jobs


def load_png(path):
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != 224 or rgb.shape[1] != 224:
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return rgb


# ----------------------------------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------------------------------
class Encoder:
    def __init__(self, dino, device, dtype, mid_layer, batch):
        import torch
        self.torch = torch
        _, runner = load_upstream()
        self.enc, self.hidden, self.n_reg, self.patch = runner.ModelFactory.create_vision_encoder(str(ROOT / dino), dtype, device)
        self.device, self.dtype, self.mid_layer, self.batch = device, dtype, mid_layer, batch

    def __call__(self, images):
        """images: list of normalized CHW float32. Returns cls (N,D) f32, patch_last (N,P,D) f16, patch_mid (N,P,D) f16."""
        torch = self.torch
        cls, last, mid = [], [], []
        with torch.no_grad():
            for i in range(0, len(images), self.batch):
                px = torch.from_numpy(np.stack(images[i:i + self.batch])).to(self.device, self.dtype)
                out = self.enc(pixel_values=px, output_hidden_states=True, return_dict=True)
                k = 1 + self.n_reg
                cls.append(out.last_hidden_state[:, 0].float().cpu().numpy())
                last.append(out.last_hidden_state[:, k:].to(torch.float16).cpu().numpy())
                mid.append(out.hidden_states[self.mid_layer][:, k:].to(torch.float16).cpu().numpy())
        return np.concatenate(cls), np.concatenate(last), np.concatenate(mid)


def l2n(x, axis=-1):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def spatial_cos_map(patches_f16, proto_unit, torch, device, batch=64):
    """patches (N,P,D) -> per-position cosine map (N,P) against a per-position unit prototype (P,D)."""
    q = torch.from_numpy(proto_unit).to(device, torch.float32)
    out = []
    for i in range(0, len(patches_f16), batch):
        p = torch.from_numpy(patches_f16[i:i + batch]).to(device, torch.float32)
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
        out.append((p * q[None]).sum(-1).cpu().numpy())
    return np.concatenate(out)


def nn_spatial_cos(patches_f16, bank_f16, torch, device, batch=16):
    """max over bank frames of mean-over-positions cosine; excludes exact self-matches by index mask (None = none)."""
    bank = torch.from_numpy(bank_f16).to(device, torch.float16)
    bank = bank / (bank.norm(dim=-1, keepdim=True) + 1e-4)
    out = []
    for i in range(0, len(patches_f16), batch):
        p = torch.from_numpy(patches_f16[i:i + batch]).to(device, torch.float16)
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-4)
        s = torch.einsum("bpd,npd->bnp", p, bank).float().mean(-1)  # (b, n)
        out.append(s.cpu().numpy())
    return np.concatenate(out)  # (N, n_bank); caller takes max (with LOO for train positives)


def fit_logreg(X, y, torch, l2=1e-2, iters=300, seed=0):
    """Small full-batch L-BFGS logistic regression; returns (w, b, mu, sd) with standardized inputs."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = torch.from_numpy((X - mu) / sd).float()
    yt = torch.from_numpy(y.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    w = (0.01 * torch.randn(X.shape[1], generator=g)).requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(Xs @ w + b, yt) + l2 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach()), mu, sd


def logreg_score(X, w, b, mu, sd):
    return ((X - mu) / sd) @ w + b


# ----------------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------------
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, float)
    sv = allv[order]
    i = 0
    while i < sv.size:  # average ranks for ties
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def dist(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
            "min": round(float(v.min()), 4), "p05": round(float(np.percentile(v, 5)), 4),
            "p50": round(float(np.median(v)), 4), "p95": round(float(np.percentile(v, 95)), 4),
            "max": round(float(v.max()), 4)}


def threshold_at_recall(pos, recall):
    s = np.sort(np.asarray(pos, float))
    k = int(np.floor((1.0 - recall) * s.size))  # values >= s[k] are >= recall fraction of positives
    return float(s[min(k, s.size - 1)])


def rate_ge(v, thr):
    v = np.asarray(v, float)
    return None if v.size == 0 else float((v >= thr).mean())


# ----------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/validation/dock_patch_gate_20260923")
    ap.add_argument("--dino", default="models/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.3)  # <= ~5 GB of the 16 GB card
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mid-layer", type=int, default=-4)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all 200 demo episodes")
    ap.add_argument("--anchors", default="0.0,0.05,0.1")
    ap.add_argument("--neg-seconds", default="3,6,10")
    ap.add_argument("--pre-seconds", default="1")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--misdock-globs", nargs="+", default=[
        "outputs/behavior/closed_loop/s1_dev10_stage3posonly_best5500_hold_exec32_720p_20260923/observations/*.png",
        "outputs/behavior/closed_loop/s1_dev10_stage4delta_best6500_hold_exec32_720p_20260923/observations/*.png",
        "outputs/behavior/closed_loop/s1_dev10_posonly_best5500_exec32_720p_20260922/observations/*.png",
    ])
    ap.add_argument("--gooddock-runs", nargs="+", default=[
        "outputs/behavior/closed_loop/s1_dev10_stage3_gain442_dockgate_20260923",
        "outputs/behavior/closed_loop/s1_dev10_stage3_gain442_dockgate_s8_20260923",
    ])
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- jobs -----------------------------------------------------------------------------------------------------
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    demo_jobs = collect_demo_jobs(table, index, [float(x) for x in args.anchors.split(",")],
                                  [float(x) for x in args.neg_seconds.split(",")],
                                  [float(x) for x in args.pre_seconds.split(",")], args.max_episodes)
    png_jobs = collect_png_jobs(args.misdock_globs, args.gooddock_runs)
    jobs = demo_jobs + png_jobs
    print(f"[jobs] demo={len(demo_jobs)} png={len(png_jobs)} device={device} dtype={dtype}", flush=True)

    # ---- decode + HSV baseline + encode ------------------------------------------------------------------------------
    enc = Encoder(args.dino, device, dtype, args.mid_layer, args.batch)
    grid = 224 // enc.patch
    cls_all, last_all, mid_all, hsv_all, keep = [], [], [], [], []
    packs, buf = {}, []

    def flush():
        if not buf:
            return
        c, l, m = enc(buf)
        cls_all.append(c); last_all.append(l); mid_all.append(m); buf.clear()

    for k, j in enumerate(jobs):
        if "episode" in j:
            ep = j["episode"]
            if ep not in packs:
                packs = {ep: FramePack(FRAMES / HEAD_CAMERA, ep)}
            try:
                rgb = packs[ep].rgb(j["frame"])
            except Exception as e:  # noqa: BLE001
                print(f"[skip] ep{ep} frame {j['frame']}: {e}")
                continue
        else:
            rgb = load_png(ROOT / j["file"])
            if rgb is None:
                continue
        f = car_dock_feature(rgb)
        hsv_all.append(-1.0 if not f["ok"] else f["score"])
        j["hsv_detected"] = bool(f["ok"])
        buf.append(normalize_image(rgb)); keep.append(k)
        if len(buf) >= 4 * args.batch:
            flush()
        if (k + 1) % 200 == 0:
            print(f"[encode] {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    flush()
    jobs = [jobs[k] for k in keep]
    cls = np.concatenate(cls_all); last = np.concatenate(last_all); mid = np.concatenate(mid_all)
    hsv = np.array(hsv_all, float)
    assert last.shape[1] == grid * grid, last.shape
    print(f"[encode] done N={len(jobs)} patches={last.shape[1]} D={last.shape[2]} {time.time() - t0:.0f}s", flush=True)
    if device == "cuda":
        print(f"[gpu] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    group = np.array([j["group"] for j in jobs])
    split = np.array([j.get("split", "closed_loop") for j in jobs])
    off = np.array([j.get("offset_s", np.nan) for j in jobs], float)
    tr_pos = (group == "demo_pos") & (split == "train")
    va_pos = (group == "demo_pos") & (split == "validation")
    tr_neg = (group == "demo_hardneg") & (split == "train")
    va_neg = (group == "demo_hardneg") & (split == "validation")
    mis = group == "misdock"; clg = group == "cl_gooddock"; cla = group == "cl_approach"

    # ---- scores -----------------------------------------------------------------------------------------------------
    scores = {}
    # A: CLS cosine to mean train-positive CLS (recomputed prototype), and A': shipped prototype
    proto_cls = cls[tr_pos].mean(0)
    scores["A_cls_cos"] = l2n(cls) @ l2n(proto_cls)
    meta = json.loads((VTT_DIR / "precond_meta.json").read_text())
    shipped = np.load(VTT_DIR / "precond_prototypes.npy")[meta["keys"].index(OPEN_LID_KEY)]
    scores["A_cls_cos_deployed"] = l2n(cls) @ l2n(shipped)

    # B / C: spatially aligned per-position prototypes
    maps = {}
    for name, P in (("B_patch_spatial_cos", last), ("C_patch_spatial_cos_mid", mid)):
        unit_tr = l2n(P[tr_pos])                         # (n, P, D)
        proto = l2n(unit_tr.mean(0))                      # (P, D) per-position unit prototype
        maps[name] = spatial_cos_map(P, proto, torch, device)   # (N, P)
        scores[name] = maps[name].mean(1)
    # D: nearest-neighbour spatial cosine over train positives (LOO for train positives themselves)
    bank_idx = np.flatnonzero(tr_pos)
    S = nn_spatial_cos(last, last[tr_pos], torch, device)      # (N, n_bank)
    for r, k in enumerate(bank_idx):
        S[k, r] = -np.inf
    scores["D_patch_nn_cos"] = S.max(1)
    # E: logistic regression on the 196-dim cosine map (train pos vs train hard negs)
    for name, src in (("E_patch_logreg", "B_patch_spatial_cos"), ("E_patch_logreg_mid", "C_patch_spatial_cos_mid")):
        X = maps[src]
        fit = tr_pos | tr_neg
        w, b, mu, sd = fit_logreg(X[fit], tr_pos[fit].astype(np.float32), torch)
        scores[name] = logreg_score(X, w, b, mu, sd)
        np.save(args.out / f"{name}_weights.npy", np.concatenate([w, [b]]).astype(np.float32))
    scores["H_hsv_dock_score"] = hsv
    del S

    # ---- metrics ----------------------------------------------------------------------------------------------------
    fixed_thr = {"A_cls_cos_deployed": DEPLOYED_TAU, "H_hsv_dock_score": DOCK_SCORE_ARRIVED}
    hsv_det = np.array([j.get("hsv_detected", False) for j in jobs])
    mis_hard = mis & hsv_det  # car fills >= 12% of the view: the truly oblique/near mis-docks (not far-away navigation)

    def dprime(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size < 2 or b.size < 2:
            return None
        return float((a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2.0) + 1e-8))
    trained_on_negs = {"E_patch_logreg", "E_patch_logreg_mid"}
    results = {}
    for name, s in scores.items():
        thr = threshold_at_recall(s[va_pos], args.recall)
        thr95 = threshold_at_recall(s[va_pos], 0.95)
        res = {
            "auc_valpos_vs_val_hardneg": auc(s[va_pos], s[va_neg]),
            "auc_valpos_vs_train_hardneg": None if name in trained_on_negs else auc(s[va_pos], s[tr_neg]),
            "auc_valpos_vs_val_hardneg_by_offset": {f"{-o:g}s": auc(s[va_pos], s[va_neg & (off == o)]) for o in sorted(set(off[va_neg]))},
            "auc_valpos_vs_misdock": auc(s[va_pos], s[mis]),
            "auc_valpos_vs_misdock_hsv_detected_only": auc(s[va_pos], s[mis_hard]),
            "dprime_valpos_vs_misdock_hard": dprime(s[va_pos], s[mis_hard]),
            "dprime_valpos_vs_val_hardneg_3s": dprime(s[va_pos], s[va_neg & (off == -3)]),
            "margin_thr85_over_misdock_hard_max_in_valpos_std": float((thr - s[mis_hard].max()) / (s[va_pos].std() + 1e-8)) if mis_hard.any() else None,
            "auc_trainpos_vs_train_hardneg": None if name in trained_on_negs else auc(s[tr_pos], s[tr_neg]),
            "dist": {
                "train_pos": dist(s[tr_pos]), "val_pos": dist(s[va_pos]),
                "val_hardneg_3s": dist(s[va_neg & (off == -3)]), "val_hardneg_6s": dist(s[va_neg & (off == -6)]),
                "val_hardneg_10s": dist(s[va_neg & (off == -10)]),
                "train_hardneg_3s": dist(s[tr_neg & (off == -3)]), "train_hardneg_6s": dist(s[tr_neg & (off == -6)]),
                "train_hardneg_10s": dist(s[tr_neg & (off == -10)]),
                "demo_pre1s_val": dist(s[(group == "demo_pre1s") & (split == "validation")]),
                "demo_pre1s_train": dist(s[(group == "demo_pre1s") & (split == "train")]),
                "misdock": dist(s[mis]), "cl_gooddock": dist(s[clg]), "cl_approach": dist(s[cla]),
            },
            "threshold_recall85": {
                "thr": thr, "recall_val_pos": rate_ge(s[va_pos], thr),
                "far_val_hardneg": rate_ge(s[va_neg], thr), "far_train_hardneg": rate_ge(s[tr_neg], thr),
                "far_val_hardneg_3s": rate_ge(s[va_neg & (off == -3)], thr), "far_val_hardneg_6s": rate_ge(s[va_neg & (off == -6)], thr),
                "far_val_hardneg_10s": rate_ge(s[va_neg & (off == -10)], thr),
                "far_train_hardneg_3s": rate_ge(s[tr_neg & (off == -3)], thr), "far_train_hardneg_6s": rate_ge(s[tr_neg & (off == -6)], thr),
                "far_misdock": rate_ge(s[mis], thr), "far_misdock_hard": rate_ge(s[mis_hard], thr),
                "accept_demo_pre1s": rate_ge(s[group == "demo_pre1s"], thr),
                "accept_cl_gooddock": rate_ge(s[clg], thr), "accept_cl_approach": rate_ge(s[cla], thr),
            },
            "threshold_recall95": {
                "thr": thr95, "far_val_hardneg": rate_ge(s[va_neg], thr95), "far_train_hardneg": rate_ge(s[tr_neg], thr95),
                "far_misdock": rate_ge(s[mis], thr95), "accept_cl_gooddock": rate_ge(s[clg], thr95),
            },
        }
        if name in fixed_thr:
            t = fixed_thr[name]
            res["threshold_fixed"] = {"thr": t, "recall_val_pos": rate_ge(s[va_pos], t), "recall_train_pos": rate_ge(s[tr_pos], t),
                                     "far_val_hardneg": rate_ge(s[va_neg], t), "far_train_hardneg": rate_ge(s[tr_neg], t),
                                     "far_misdock": rate_ge(s[mis], t), "far_misdock_hard": rate_ge(s[mis_hard], t),
                                     "accept_cl_gooddock": rate_ge(s[clg], t)}
        results[name] = res

    # ---- verdict ----------------------------------------------------------------------------------------------------
    hsv_auc_mis = results["H_hsv_dock_score"]["auc_valpos_vs_misdock_hsv_detected_only"]
    generic = [n for n in scores if n not in ("H_hsv_dock_score", "A_cls_cos_deployed")]  # A' is the same feature as A

    def passes(n):
        r = results[n]
        return (r["auc_valpos_vs_misdock_hsv_detected_only"] is not None and r["auc_valpos_vs_misdock_hsv_detected_only"] >= 0.85
                and r["auc_valpos_vs_misdock_hsv_detected_only"] >= hsv_auc_mis - 1e-9
                and r["auc_valpos_vs_val_hardneg_by_offset"].get("3s", 0) >= 0.85
                and r["auc_valpos_vs_val_hardneg_by_offset"].get("6s", 0) >= 0.85)
    passing = [n for n in generic if passes(n)]

    def rank_key(n):
        r = results[n]
        return (round(min(r["auc_valpos_vs_misdock_hsv_detected_only"], r["auc_valpos_vs_val_hardneg_by_offset"].get("3s", 0)), 2),
                r["dprime_valpos_vs_misdock_hard"] or 0.0)
    ranked = sorted(generic, key=rank_key, reverse=True)
    best = ranked[0]
    verdict = {
        "criterion": "generic score with AUC(val pos vs HSV-detected mis-dock) >= 0.85 and >= HSV AUC on the same frames, and AUC(val pos vs demo hard neg) >= 0.85 at both 3 s and 6 s before arrival; ranked by the weaker of the two AUCs (2 dp), then d' on the hard mis-docks",
        "hsv_auc_valpos_vs_misdock_hard": hsv_auc_mis, "hsv_auc_valpos_vs_misdock_all": results["H_hsv_dock_score"]["auc_valpos_vs_misdock"],
        "passing_scores": passing, "ranking": ranked, "best_generic_score": best,
        "recommended": {"score": best, "threshold_recall85": results[best]["threshold_recall85"],
                        "auc_misdock_hard": results[best]["auc_valpos_vs_misdock_hsv_detected_only"], "auc_misdock_all": results[best]["auc_valpos_vs_misdock"],
                        "auc_val_hardneg": results[best]["auc_valpos_vs_val_hardneg"], "auc_by_offset": results[best]["auc_valpos_vs_val_hardneg_by_offset"],
                        "dprime_misdock_hard": results[best]["dprime_valpos_vs_misdock_hard"]},
        "exists_generic_score_at_least_as_good_as_hsv": bool(passing),
        "deployed_gate_note": (f"A_cls_cos_deployed at tau={DEPLOYED_TAU}: FAR {results['A_cls_cos_deployed']['threshold_fixed']['far_misdock_hard']:.2f} on hard mis-docks, "
                               f"{results['A_cls_cos_deployed']['threshold_fixed']['far_val_hardneg']:.2f} on demo hard negatives; the same feature at its 85%-recall "
                               f"threshold {results['A_cls_cos_deployed']['threshold_recall85']['thr']:.3f} gives FAR {results['A_cls_cos_deployed']['threshold_recall85']['far_misdock_hard']:.2f} / "
                               f"{results['A_cls_cos_deployed']['threshold_recall85']['far_val_hardneg']:.2f}: the CLS gate's failure is its threshold, not (only) its feature."),
        "cl_gooddock_note": {n: {"accept_at_thr85": results[n]["threshold_recall85"]["accept_cl_gooddock"], "mean": results[n]["dist"]["cl_gooddock"].get("mean"),
                                 "val_hardneg_3s_mean": results[n]["dist"]["val_hardneg_3s"].get("mean"), "val_pos_mean": results[n]["dist"]["val_pos"].get("mean")}
                             for n in scores},
    }

    counts = {g: int((group == g).sum()) for g in sorted(set(group))}
    counts.update({"train_pos": int(tr_pos.sum()), "val_pos": int(va_pos.sum()), "train_hardneg": int(tr_neg.sum()),
                   "val_hardneg": int(va_neg.sum()), "hsv_detected_misdock": int(sum(j.get("hsv_detected", False) for j in jobs if j["group"] == "misdock")),
                   "hsv_detected_val_pos": int(sum(j.get("hsv_detected", False) for j, m in zip(jobs, va_pos) if m)),
                   "hsv_detected_cl_gooddock": int(sum(j.get("hsv_detected", False) for j, m in zip(jobs, clg) if m)),
                   "demo_episodes": len({j["episode"] for j in jobs if "episode" in j})})
    summary = {
        "purpose": "generic (colour-free, demo-only) DINOv3 patch-based arrival scores vs CLS gate and HSV dock_score at move to(car)->open lid(car)",
        "ground_truth_used": False, "simulator_launched": False,
        "encoder": {"dino": args.dino, "device": device, "dtype": str(dtype), "image_size": 224, "grid": grid,
                    "mid_layer": args.mid_layer, "cls": "last_hidden_state[:,0]", "patch_last": "last_hidden_state[:,1+reg:]",
                    "patch_mid": f"hidden_states[{args.mid_layer}][:,1+reg:]", "preprocess": "harness.wam_behavior.normalize_image"},
        "protocol": {"anchors": args.anchors, "hardneg_seconds_before_openlid": args.neg_seconds, "pre_seconds": args.pre_seconds,
                     "fps": FPS, "prototype_split": "train", "eval_split": "validation", "recall_target": args.recall,
                     "hsv_undetected_score": -1.0, "misdock_globs": args.misdock_globs, "gooddock_runs": args.gooddock_runs,
                     "gooddock_openlid_ranges": {r: openlid_step_ranges(ROOT / r) for r in args.gooddock_runs}},
        "counts": counts, "results": results, "verdict": verdict, "seconds": round(time.time() - t0, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    rows = []
    for k, j in enumerate(jobs):
        r = {kk: v for kk, v in j.items() if kk != "openlid_ranges"}
        r.update({n: round(float(scores[n][k]), 5) for n in scores})
        rows.append(r)
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n")
    np.save(args.out / "proto_patch_last.npy", l2n(l2n(last[tr_pos]).mean(0)).astype(np.float32))
    np.save(args.out / "proto_cls.npy", proto_cls.astype(np.float32))
    write_contact_sheet(args.out, jobs, scores, va_pos, va_neg & (off == -3), clg, mis_hard)
    write_readme(args, summary)
    print(json.dumps({"counts": counts, "verdict": verdict}, indent=1))
    print(f"[done] {time.time() - t0:.0f}s -> {args.out}")


def write_contact_sheet(out, jobs, scores, va_pos, neg3, clg, mis_hard, per_row=4):
    """4 rows x 4: validation positives, 3 s hard negatives, closed-loop 'good dock', hard mis-docks (annotated A/B/H)."""
    import cv2
    tiles = []
    for label, mask in (("val_pos", va_pos), ("hardneg_3s", neg3), ("cl_gooddock", clg), ("misdock_hard", mis_hard)):
        idx = np.flatnonzero(mask)
        idx = idx[np.linspace(0, len(idx) - 1, per_row).round().astype(int)] if len(idx) else idx
        row = []
        for k in idx:
            j = jobs[k]
            im = FramePack(FRAMES / HEAD_CAMERA, j["episode"]).rgb(j["frame"]) if "episode" in j else load_png(ROOT / j["file"])
            im = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2BGR)
            txt = f"{label} A{scores['A_cls_cos'][k]:.2f} B{scores['B_patch_spatial_cos'][k]:.2f} H{scores['H_hsv_dock_score'][k]:.2f}"
            cv2.putText(im, txt, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
            row.append(im)
        while len(row) < per_row:
            row.append(np.zeros((224, 224, 3), np.uint8))
        tiles.append(np.concatenate(row, 1))
    cv2.imwrite(str(out / "contact_sheet.jpg"), np.concatenate(tiles, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])


def write_readme(args, summary):
    R = summary["results"]; c = summary["counts"]

    def f(x, d=3):
        return "-" if x is None else f"{x:.{d}f}"
    lines = ["# Generic patch-based arrival gate vs CLS gate vs HSV dock_score (offline, no sim, no GT)", "",
             f"Boundary `move to(car) -> open lid(car)`, {c['demo_episodes']} demo episodes; encoder {summary['encoder']['dino']} "
             f"({summary['encoder']['device']} {summary['encoder']['dtype']}), 224 px, {summary['encoder']['grid']}x{summary['encoder']['grid']} patch grid. "
             f"Prototypes / classifier from the TRAIN split only; every number below is on held-out frames unless marked.", "",
             "Groups: train_pos {train_pos}, val_pos {val_pos} (open-lid start, anchors {a}), val_hardneg {val_hardneg} / train_hardneg {train_hardneg} "
             "(same episodes' move-to segment {n} s before open-lid start), misdock {misdock} recorded oblique closed-loop frames "
             "(HSV detects {hd}), cl_gooddock {cl_gooddock} closed-loop dockgate frames inside the open-lid phase, cl_approach {cl_approach}, "
             "demo_pre1s {demo_pre1s} (1 s before start, diagnostic only).".format(a=args.anchors, n=args.neg_seconds, hd=c["hsv_detected_misdock"], **c), "",
             "| score | AUC val-pos vs val hard-neg | AUC @3s | AUC @6s | AUC @10s | AUC vs mis-dock (all 71) | AUC vs mis-dock (hard 44) | d' hard mis-dock | thr (85% recall) | FAR val hard-neg | FAR hard-neg 3s | FAR hard-neg 6s | FAR mis-dock all | FAR mis-dock hard | margin thr-to-worst-misdock (val-pos std) | accept cl-good-dock | cl-good-dock mean | val-pos mean+-std | mis-dock mean+-std |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in R.items():
        t = r["threshold_recall85"]; d = r["dist"]; bo = r["auc_valpos_vs_val_hardneg_by_offset"]
        lines.append(f"| {n} | {f(r['auc_valpos_vs_val_hardneg'])} | {f(bo.get('3s'))} | {f(bo.get('6s'))} | {f(bo.get('10s'))} | {f(r['auc_valpos_vs_misdock'])} | "
                     f"{f(r['auc_valpos_vs_misdock_hsv_detected_only'])} | {f(r['dprime_valpos_vs_misdock_hard'],2)} | "
                     f"{t['thr']:.3f} | {f(t['far_val_hardneg'],2)} | {f(t['far_val_hardneg_3s'],2)} | {f(t['far_val_hardneg_6s'],2)} | {f(t['far_misdock'],2)} | {f(t['far_misdock_hard'],2)} | "
                     f"{f(r['margin_thr85_over_misdock_hard_max_in_valpos_std'],2)} | "
                     f"{f(t['accept_cl_gooddock'],2)} | {f(d['cl_gooddock'].get('mean'))} | {f(d['val_pos'].get('mean'))}+-{f(d['val_pos'].get('std'))} | "
                     f"{f(d['misdock'].get('mean'))}+-{f(d['misdock'].get('std'))} |")
    lines += ["", "Fixed deployed thresholds:"]
    for n in ("A_cls_cos_deployed", "H_hsv_dock_score"):
        t = R[n].get("threshold_fixed")
        if t:
            lines.append(f"- `{n}` @ {t['thr']}: recall val-pos {f(t['recall_val_pos'],2)}, FAR val hard-neg {f(t['far_val_hardneg'],2)}, "
                         f"FAR mis-dock all/hard {f(t['far_misdock'],2)}/{f(t['far_misdock_hard'],2)}, accept cl-good-dock {f(t['accept_cl_gooddock'],2)}")
    v = summary["verdict"]
    lines += ["", "## Verdict", "",
              f"Criterion: {v['criterion']}.", "",
              f"HSV dock_score AUC (val-pos vs mis-dock) = {f(v['hsv_auc_valpos_vs_misdock_hard'])} on the 44 hard (car-filling) mis-dock frames, "
              f"{f(v['hsv_auc_valpos_vs_misdock_all'])} on all 71. Generic scores passing: {v['passing_scores'] or 'none'}; ranking {v['ranking']}.", "",
              f"Best generic score: `{v['best_generic_score']}` — AUC hard mis-dock {f(v['recommended']['auc_misdock_hard'])} (all {f(v['recommended']['auc_misdock_all'])}, "
              f"d' {f(v['recommended']['dprime_misdock_hard'],2)}), AUC demo hard-neg {f(v['recommended']['auc_val_hardneg'])} "
              f"(3 s {f(v['recommended']['auc_by_offset'].get('3s'))}, 6 s {f(v['recommended']['auc_by_offset'].get('6s'))}); "
              f"threshold {v['recommended']['threshold_recall85']['thr']:.3f} at 85% val recall gives FAR {f(v['recommended']['threshold_recall85']['far_val_hardneg'],2)} on demo hard-negs, "
              f"{f(v['recommended']['threshold_recall85']['far_misdock'],2)} on recorded mis-docks, and accepts {f(v['recommended']['threshold_recall85']['accept_cl_gooddock'],2)} of closed-loop good-dock frames.", "",
              v["deployed_gate_note"], "",
              "Closed-loop 'good dock' column (contact_sheet.jpg, row 3): the dockgate runs' open-lid-phase frames show the car centred but at roughly twice the demo "
              "docking range (car spans about half the image height vs filling it in demo positives, row 1), i.e. they resemble the demo frames 3 s BEFORE arrival "
              "(row 2) more than arrived frames; the lid was never opened in those runs. The scale-invariant HSV dock_score accepts them (a false accept of its own), "
              "the DINO scores reject them, which is the intended behaviour; their per-score means are in summary.json verdict.cl_gooddock_note. "
              "The closed-loop renderer also differs slightly from the demo videos (flatter lighting), so part of the closed-loop score drop may be domain gap; "
              "this cannot be separated from range offline without ground truth.", "",
              "Caveats: 20 validation episodes (60 positives / 60 hard negatives); mis-dock frames come from 3 closed-loop runs of one instance, and include "
              "pre-arrival navigation frames (all are 'not arrived'); closed-loop good-dock frames include arm-occluded views during the open-lid attempt; "
              "E is a 196-weight classifier fitted on train demo frames only (no closed-loop data). Files: summary.json (all metrics), rows.json (per-frame scores), "
              "proto_patch_last.npy / proto_cls.npy (train prototypes), E_*_weights.npy."]
    (args.out / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
