#!/usr/bin/env python3
"""Evaluate deployable single-sample selection rules for open-lid WAM action chunks.

Offline validation diagnostic on held-out demonstration observations (same anchors, seeds and K as
`check_openlid_sample_modes.py`).  K stochastic flow-matching samples are drawn per anchor and each
selection rule commits to ONE sample using only information available at deployment time: the K
samples themselves, their predicted future DINO patch features (the WAM future-feature decoder applied
to the last integration step's cond tokens) and the robot's own realized motion over the previous 32
steps.  The `oracle*` rules look at the demonstrated future and are reported only as non-deployable
capacity bounds.  Nothing here is closed-loop task success.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.frame_cache import FramePack  # noqa: E402
from harness.policy_server import WamPolicy  # noqa: E402
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, STATE_SLICES, SkillTable  # noqa: E402


POSITION_PAIRS = {
    "trunk": ((3, 7), STATE_SLICES["trunk_qpos"]),
    "left_arm": ((7, 14), STATE_SLICES["arm_left_qpos"]),
    "right_arm": ((15, 22), STATE_SLICES["arm_right_qpos"]),
}
GROUPS = ("trunk", "left_arm", "right_arm", "upper_body")
UB_STATE_INDEX = np.concatenate([np.arange(s0, s1) for (_, (s0, s1)) in POSITION_PAIRS.values()])
RULES = {
    "single": (True, "expected value of the current K=1 deployment (metrics averaged over the K samples)"),
    "mean": (True, "K-sample average chunk (current --samples K deployment)"),
    "medoid_ub": (True, "sample whose 18-D upper-body intent has the highest mean cosine to the other K-1 intents"),
    "medoid_chunk": (True, "medoid of the K-mean-centred normalized 32x23 chunks (cosine, model action space)"),
    "momentum": (True, "sample whose upper-body intent best aligns with the realized motion over the previous 32 steps"),
    "medoid_momentum": (True, "argmax of medoid_ub score + 0.5 * cosine to the previous realized motion"),
    "future_medoid": (True, "sample whose predicted future DINO patch features are most typical of the other K-1 samples"),
    "oracle": (False, "best-of-K by cosine of the 18-D upper-body intent to the demonstrated intent"),
    "oracle_group": (False, "per-group best-of-K by cosine to the demonstrated intent (check_openlid_sample_modes oracle)"),
}
INDEX_RULES = ("medoid_ub", "medoid_chunk", "momentum", "medoid_momentum", "future_medoid", "oracle")


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-8 and nb > 1e-8 else None


def safe_cos_rows(X, v, eps=1e-8):
    """Cosine of each row of X to v; 0 where undefined so that argmax stays well defined."""
    nv = np.linalg.norm(v)
    nx = np.linalg.norm(X, axis=1)
    if nv < eps:
        return np.zeros(X.shape[0])
    return np.where(nx > eps, X @ v / (np.maximum(nx, eps) * nv), 0.0)


def medoid_scores(X, eps=1e-8):
    """Mean cosine of each row to the other rows, and the mean pairwise cosine."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    C = Xn @ Xn.T
    K = X.shape[0]
    return (C.sum(axis=1) - np.diag(C)) / (K - 1), float((C.sum() - np.trace(C)) / (K * (K - 1)))


def mean_finite(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def vector_metrics(selected, demo_vec):
    c = cosine(selected, demo_vec)
    dn = np.linalg.norm(demo_vec)
    return {"cos": c, "norm_ratio": float(np.linalg.norm(selected) / dn) if dn > 1e-8 else None,
            "positive": None if c is None else float(c > 0)}


@torch.no_grad()
def sample_with_future(policy, feats, state, cond):
    """Replica of WamPolicy._sample_once that also returns the future-feature prediction.

    The cond tokens of the LAST Euler step feed `future_feat_decoder`, exactly as `_Wrapper.sample_actions`
    exposes them; the generator draw order is identical to `_sample_once`, so samples match the K=16 audit.
    """
    w = policy.wrapper
    state_t = torch.from_numpy(state)[None].to(w.device, torch.float32)
    qpos = w.normalize_state(state_t).unsqueeze(1)
    x = torch.randn(1, w.action_model.action_len, w.action_min.numel(), device=w.device, generator=policy.gen)
    tc = None if cond is None else cond.to(w.device, torch.float32)
    cuda = w.device.type == "cuda"
    if not cuda:
        feats = [f.float() for f in feats]
    cond_tokens = None
    for i in range(policy.num_steps):
        t = torch.full((1,), i / policy.num_steps, device=w.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            out = w.action_model(t, noisy_actions=x, qpos_history=qpos, dino_features_list=feats, task_cond=tc)
        x = x + out["final_pred"].float() / policy.num_steps
        cond_tokens = out["cond_tokens"]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        future = w.action_model.future_feat_decoder(cond_tokens)  # (1, 196, 1024)
    absolute = w.to_absolute_action(w.denormalize_action(x), state_t)
    return absolute[0].float().cpu().numpy(), future[0].float()


@torch.no_grad()
def evaluate_anchor(policy, condition, state, rgb, demo, seed, K, prev_motion):
    """Draw K samples at one anchor and apply every selection rule."""
    _, pixels = policy._preprocess({"r::proprio": state, "r::zed_link::rgb": rgb})
    feats, _ = policy._encode(pixels)
    policy.gen.manual_seed(seed)
    chunks, futures = [], []
    for _ in range(K):
        chunk, future = sample_with_future(policy, feats, state, condition)
        chunks.append(chunk)
        futures.append(future)
    samples = np.stack(chunks)                                   # (K, 32, 23) absolute controller units
    future = torch.stack(futures)                                # (K, 196, 1024)
    w = policy.wrapper

    intents, demo_intents = {}, {}
    for group, ((a0, a1), (s0, s1)) in POSITION_PAIRS.items():
        q = state[s0:s1]
        intents[group] = samples[:, -1, a0:a1] - q[None]
        demo_intents[group] = demo[-1, a0:a1] - q
    intents["upper_body"] = np.concatenate([intents[g] for g in POSITION_PAIRS], axis=1)
    demo_intents["upper_body"] = np.concatenate([demo_intents[g] for g in POSITION_PAIRS])
    base_vx = samples[:, :, 0].mean(axis=1)
    demo_vx = float(demo[:, 0].mean())

    ub = intents["upper_body"]
    medoid_ub, ub_pairwise = medoid_scores(ub)
    state_rep = torch.from_numpy(state)[None].expand(K, -1).contiguous()
    normalized = w.normalize_action(w.to_model_action(torch.from_numpy(samples).to(torch.float32), state_rep))
    normalized = normalized.reshape(K, -1).cpu().numpy()
    centred = normalized - normalized.mean(axis=0, keepdims=True)
    medoid_chunk, chunk_pairwise = medoid_scores(centred)
    Fn = torch.nn.functional.normalize(future, dim=-1)
    C = torch.einsum("ktd,jtd->kj", Fn, Fn) / Fn.shape[1]     # per-token cosine averaged over the 196 tokens
    future_scores = ((C.sum(1) - C.diagonal()) / (K - 1)).cpu().numpy()
    future_pairwise = float((C.sum() - C.diagonal().sum()) / (K * (K - 1)))
    oracle_cos = safe_cos_rows(ub, demo_intents["upper_body"])

    selection = {"medoid_ub": int(np.argmax(medoid_ub)), "medoid_chunk": int(np.argmax(medoid_chunk)),
                 "future_medoid": int(np.argmax(future_scores)), "oracle": int(np.argmax(oracle_cos)),
                 "momentum": None, "medoid_momentum": None}
    momentum_cos = None
    if prev_motion is not None:
        momentum_cos = safe_cos_rows(ub, prev_motion)
        selection["momentum"] = int(np.argmax(momentum_cos))
        selection["medoid_momentum"] = int(np.argmax(medoid_ub + 0.5 * momentum_cos))
    oracle_group_index = {g: int(np.argmax(safe_cos_rows(intents[g], demo_intents[g]))) for g in POSITION_PAIRS}
    order = np.argsort(-oracle_cos)
    rank = np.empty(K, dtype=int)
    rank[order] = np.arange(1, K + 1)
    return {"intents": intents, "demo_intents": demo_intents, "base_vx": base_vx, "demo_vx": demo_vx,
            "selection": selection, "oracle_group_index": oracle_group_index, "oracle_rank": rank,
            "sample_ub_demo_cos": [float(v) for v in oracle_cos],
            "scores": {"medoid_ub": medoid_ub, "medoid_chunk": medoid_chunk, "future": future_scores, "momentum": momentum_cos},
            "diagnostics": {"ub_pairwise_cos": ub_pairwise, "chunk_pairwise_cos": chunk_pairwise,
                            "future_pairwise_cos": future_pairwise}}


def rule_metrics(result, rule):
    """Per-group cos / norm ratio / positive indicator and base vx of the chunk a rule commits to."""
    intents, demo = result["intents"], result["demo_intents"]
    out = {"index": None, "groups": {}, "base_vx": None}
    if rule == "single":
        for g in GROUPS:
            per_sample = [vector_metrics(v, demo[g]) for v in intents[g]]
            out["groups"][g] = {key: mean_finite([m[key] for m in per_sample]) for key in ("cos", "norm_ratio", "positive")}
        out["base_vx"] = float(result["base_vx"].mean())
    elif rule == "mean":
        for g in GROUPS:
            out["groups"][g] = vector_metrics(intents[g].mean(axis=0), demo[g])
        out["base_vx"] = float(result["base_vx"].mean())
    elif rule == "oracle_group":
        for g in POSITION_PAIRS:
            out["groups"][g] = vector_metrics(intents[g][result["oracle_group_index"][g]], demo[g])
        out["groups"]["upper_body"] = {"cos": None, "norm_ratio": None, "positive": None}   # mixes samples
    else:
        k = result["selection"][rule]
        if k is None:
            return None
        out["index"] = k
        for g in GROUPS:
            out["groups"][g] = vector_metrics(intents[g][k], demo[g])
        out["base_vx"] = float(result["base_vx"][k])
    return out


def aggregate(rows, rules):
    table = {}
    for rule in rules:
        table[rule] = {}
        for g in GROUPS:
            entries = [row["rules"][rule]["groups"][g] for row in rows if row["rules"][rule] is not None]
            table[rule][g] = {"cos": mean_finite([e["cos"] for e in entries]),
                              "norm_ratio": mean_finite([e["norm_ratio"] for e in entries]),
                              "frac_pos": mean_finite([e["positive"] for e in entries]),
                              "n": len([e for e in entries if e["cos"] is not None])}
    return table


def aggregate_base(rows, rules):
    out = {"demo_vx": mean_finite([row["demo_vx"] for row in rows]), "rules": {}}
    for rule in rules:
        pairs = [(row["rules"][rule]["base_vx"], row["demo_vx"]) for row in rows
                 if row["rules"][rule] is not None and row["rules"][rule]["base_vx"] is not None]
        out["rules"][rule] = {"vx": mean_finite([p[0] for p in pairs]),
                              "abs_err": mean_finite([abs(p[0] - p[1]) for p in pairs]), "n": len(pairs)}
    return out


def rule_ub_choice(result, rule):
    """Upper-body intent(s) a rule commits to at one anchor: (18,) vector, (K, 18) set for `single`, or None."""
    if rule == "single":
        return result["intents"]["upper_body"]
    if rule == "mean":
        return result["intents"]["upper_body"].mean(axis=0)
    k = result["selection"][rule]
    return None if k is None else result["intents"]["upper_body"][k]


def consecutive_cos(a, b):
    """Cosine between the choices at two consecutive anchors; expected over all K x K pairs for `single`."""
    if a is None or b is None:
        return None
    if a.ndim == 2:
        An = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        Bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        return float((An @ Bn.T).mean())
    return cosine(a, b)


def fmt(value, nd=3):
    return "n/a" if value is None else f"{value:.{nd}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt")
    parser.add_argument("--targets", default="data/skill_vtt_task013/contract_targets.json")
    parser.add_argument("--out", type=Path, default=Path("outputs/validation/sample_selection_openlid_20260923"))
    parser.add_argument("--fractions", default="0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--commit-episodes", type=int, default=8)
    parser.add_argument("--commit-steps", type=int, default=6)
    parser.add_argument("--guard-seconds", type=float, default=720.0, help="projected budget above which the run drops to 10 episodes")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    args.out.mkdir(parents=True)
    log_path = args.out / "run.log"

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a") as handle:
            handle.write(line + "\n")

    targets = json.loads((ROOT / args.targets).read_text())
    condition = torch.tensor(targets["contracts"]["open lid(car)"]["vector"], dtype=torch.float32)[None]
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    episodes = [episode for episode in sorted(table.episodes)
                if index["episodes"][str(episode)]["split"] == "validation"][:args.max_episodes]
    fractions = [float(value) for value in args.fractions.split(",")]
    policy = WamPolicy(args.checkpoint, skill="move to", exec_horizon=8, num_steps=10, device=args.device,
                       cuda_fraction=0.4, seed=0, log_dir=args.out,
                       orchestrate={"vlm": "none", "frozen": True, "train_targets": args.targets})
    torch.set_num_threads(args.threads)
    K = args.samples
    if not hasattr(policy.wrapper.action_model, "future_feat_decoder"):
        raise SystemExit("checkpoint has no future_feat_decoder; future_medoid needs it")

    def episode_data(episode):
        segment = next(segment for segment in table.episodes[episode]["segments"] if segment["skill"] == "open lid")
        action = np.load(LOWDIM / f"episode_{episode}_action.npy", mmap_mode="r")
        state_all = np.load(LOWDIM / f"episode_{episode}_state.npy", mmap_mode="r")
        return segment["start"], segment["end"], action, state_all, FramePack(FRAMES / HEAD_CAMERA, episode)

    def run_anchor(episode, anchor, lo, action, state_all, pack):
        state = np.array(state_all[anchor], dtype=np.float32)          # copy: the mmap'd cache is read-only
        demo = np.array(action[anchor:anchor + 32], dtype=np.float32)
        prev_motion, prev_norm = None, None
        if anchor - 32 >= lo:
            prev_motion = state[UB_STATE_INDEX] - np.asarray(state_all[anchor - 32], dtype=np.float32)[UB_STATE_INDEX]
            prev_norm = float(np.linalg.norm(prev_motion))
        result = evaluate_anchor(policy, condition, state, pack.rgb(anchor), demo, 1000 * episode + anchor, K, prev_motion)
        result["prev_motion_norm"] = prev_norm
        return result

    def make_row(episode, anchor, fraction, result):
        row = {"episode": episode, "anchor": anchor, "fraction": fraction,
               "momentum_available": result["selection"]["momentum"] is not None,
               "prev_motion_norm": result["prev_motion_norm"], "demo_vx": result["demo_vx"],
               "selection": result["selection"], "oracle_group_index": result["oracle_group_index"],
               "oracle_rank": {rule: (None if result["selection"][rule] is None else int(result["oracle_rank"][result["selection"][rule]]))
                               for rule in INDEX_RULES},
               "sample_ub_demo_cos": result["sample_ub_demo_cos"], "diagnostics": result["diagnostics"],
               "rules": {rule: rule_metrics(result, rule) for rule in RULES}}
        return row

    # ---- main pass: 10 fractions x episodes, with a runtime guard timed on the first 10 anchors ----
    started = time.monotonic()
    rows, reduced = [], False
    plan = [(episode, fraction) for episode in episodes for fraction in fractions]
    n_commit = min(args.commit_episodes, len(episodes)) * args.commit_steps
    cache = {}
    i = 0
    while i < len(plan):
        episode, fraction = plan[i]
        if episode not in cache:
            cache = {episode: episode_data(episode)}
        lo, hi, action, state_all, pack = cache[episode]
        anchor = int(lo + fraction * (hi - lo - 33))
        anchor = (anchor + 1) // 2 * 2
        rows.append(make_row(episode, anchor, fraction, run_anchor(episode, anchor, lo, action, state_all, pack)))
        i += 1
        if i == 10:
            per_anchor = (time.monotonic() - started) / 10
            projected = per_anchor * (len(plan) + n_commit)
            log(f"guard: {per_anchor:.2f} s/anchor, projected {projected:.0f} s for {len(plan) + n_commit} anchors")
            if projected > args.guard_seconds:
                episodes = episodes[:10]
                plan = [(episode, fraction) for episode in episodes for fraction in fractions]
                reduced = True
                log(f"guard: projected > {args.guard_seconds:.0f} s, reducing to {len(episodes)} episodes")
        if i % 20 == 0 or i == len(plan):
            log(f"main pass {i}/{len(plan)} anchors, {time.monotonic() - started:.0f} s")
    main_seconds = time.monotonic() - started

    # ---- commitment pass: consecutive anchors in the first episodes ----
    commit_rows = []
    for episode in episodes[:args.commit_episodes]:
        lo, hi, action, state_all, pack = episode_data(episode)
        t0 = int(lo + 0.1 * (hi - lo))
        anchors = [(t0 + 32 * s + 1) // 2 * 2 for s in range(args.commit_steps)]
        anchors = [a for a in anchors if a + 32 <= hi]
        choices, demo_ub, group_choices, demo_groups = [], [], [], []
        for anchor in anchors:
            result = run_anchor(episode, anchor, lo, action, state_all, pack)
            choices.append({rule: rule_ub_choice(result, rule) for rule in RULES if rule != "oracle_group"})
            demo_ub.append(result["demo_intents"]["upper_body"])
            group_choices.append({rule: (None if result["selection"][rule] is None
                                         else {g: result["intents"][g][result["selection"][rule]] for g in POSITION_PAIRS})
                                  for rule in INDEX_RULES})
            group_choices[-1]["mean"] = {g: result["intents"][g].mean(axis=0) for g in POSITION_PAIRS}
            group_choices[-1]["single"] = {g: result["intents"][g] for g in POSITION_PAIRS}
            demo_groups.append({g: result["demo_intents"][g] for g in POSITION_PAIRS})
        for s in range(len(anchors) - 1):
            pair = {"episode": episode, "anchor_a": anchors[s], "anchor_b": anchors[s + 1],
                    "demo": {"upper_body": cosine(demo_ub[s], demo_ub[s + 1]),
                             **{g: cosine(demo_groups[s][g], demo_groups[s + 1][g]) for g in POSITION_PAIRS}},
                    "rules": {}}
            for rule in choices[s]:
                entry = {"upper_body": consecutive_cos(choices[s][rule], choices[s + 1][rule])}
                ga, gb = group_choices[s].get(rule), group_choices[s + 1].get(rule)
                for g in POSITION_PAIRS:
                    entry[g] = None if ga is None or gb is None else consecutive_cos(ga[g], gb[g])
                pair["rules"][rule] = entry
            commit_rows.append(pair)
        log(f"commitment pass episode {episode}: {len(anchors)} anchors, {time.monotonic() - started:.0f} s")
    commit_rules = [rule for rule in RULES if rule != "oracle_group"]
    commitment = {"episodes": episodes[:args.commit_episodes], "steps_per_episode": args.commit_steps,
                  "n_pairs": len(commit_rows),
                  "demo": {g: mean_finite([p["demo"][g] for p in commit_rows]) for g in GROUPS},
                  "rules": {rule: {g: mean_finite([p["rules"][rule][g] for p in commit_rows]) for g in GROUPS}
                            for rule in commit_rules},
                  "n_pairs_by_rule": {rule: len([p for p in commit_rows if p["rules"][rule]["upper_body"] is not None])
                                      for rule in commit_rules},
                  "rows": commit_rows}

    # ---- aggregation ----
    rules = list(RULES)
    momentum_rows = [row for row in rows if row["momentum_available"]]
    overall = aggregate(rows, rules)
    momentum_subset = aggregate(momentum_rows, rules)
    by_fraction = {str(f): aggregate([row for row in rows if row["fraction"] == f], rules) for f in fractions}
    base = {"overall": aggregate_base(rows, rules),
            "by_fraction": {str(f): aggregate_base([row for row in rows if row["fraction"] == f], rules) for f in fractions}}
    oracle_rank = {rule: {"mean_rank": mean_finite([row["oracle_rank"][rule] for row in rows]),
                          "frac_equal_oracle": mean_finite([float(row["selection"][rule] == row["selection"]["oracle"])
                                                            for row in rows if row["selection"][rule] is not None])}
                   for rule in INDEX_RULES if rule != "oracle"}
    diagnostics = {key: mean_finite([row["diagnostics"][key] for row in rows])
                   for key in ("ub_pairwise_cos", "chunk_pairwise_cos", "future_pairwise_cos")}
    report = {
        "checkpoint": args.checkpoint, "checkpoint_step": policy.checkpoint_step, "key": "open lid(car)",
        "split": "validation", "samples": K, "num_steps": policy.num_steps, "episodes": episodes, "fractions": fractions,
        "n_anchors": len(rows), "seconds": time.monotonic() - started, "main_pass_seconds": main_seconds,
        "runtime_guard": {"budget_seconds": args.guard_seconds, "reduced_to_10_episodes": reduced},
        "policy_inference": True, "closed_loop": False, "task_success_evaluated": False,
        "oracle_warning": "oracle/oracle_group use the demonstrated future actions; diagnostic only, never deploy",
        "rules": {rule: {"deployable": dep, "description": desc} for rule, (dep, desc) in RULES.items()},
        "intent_definition": "a[-1, group] - q_now[group] for trunk/left_arm/right_arm; upper_body = concatenation (18-D); base = mean over 32 steps of a[:, 0:3]",
        "momentum": {"available_anchors": len(momentum_rows), "skipped_anchors": len(rows) - len(momentum_rows),
                     "degenerate_prev_motion": len([row for row in momentum_rows if row["prev_motion_norm"] is not None and row["prev_motion_norm"] < 1e-6])},
        "overall": overall, "momentum_subset": momentum_subset, "by_fraction": by_fraction, "base": base,
        "oracle_rank": oracle_rank, "sample_diagnostics": diagnostics, "commitment": commitment, "rows": rows,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    # ---- README ----
    deployable = [rule for rule in rules if RULES[rule][0]]
    full = [rule for rule in deployable if rule not in ("momentum", "medoid_momentum")]
    best = max(full, key=lambda rule: overall[rule]["upper_body"]["cos"])
    best_mom = max(("momentum", "medoid_momentum"), key=lambda rule: momentum_subset[rule]["upper_body"]["cos"])
    ub = lambda tab, rule: tab[rule]["upper_body"]["cos"]  # noqa: E731
    best_commit = max([rule for rule in full if rule not in ("mean", "single")], key=lambda rule: ub(overall, rule))
    d_mean, d_single = ub(overall, best_commit) - ub(overall, "mean"), ub(overall, best_commit) - ub(overall, "single")
    d_mom = ub(momentum_subset, best_mom) - max(ub(momentum_subset, "mean"), ub(momentum_subset, best))
    material = 0.05

    def verdict(delta):
        return "materially beats" if delta >= material else ("matches" if delta > -material else "is materially worse than")

    def cell(m):
        return f"{fmt(m['cos'])} / {fmt(m['norm_ratio'], 2)} / {fmt(m['frac_pos'], 2)}"

    def main_table(tab, title):
        out = [f"### {title}", "", "cos to demo intent / norm ratio (selected / demo) / fraction of anchors with cos > 0", "",
               "| Rule | deployable | trunk | left_arm | right_arm | upper_body (18-D) | n |", "|---|---|---:|---:|---:|---:|---:|"]
        for rule in rules:
            out.append(f"| {rule} | {'yes' if RULES[rule][0] else 'NO (diagnostic)'} | " +
                       " | ".join(cell(tab[rule][g]) for g in GROUPS) + f" | {tab[rule]['trunk']['n']} |")
        return out + [""]

    lines = ["# Open-lid sample selection rules (K=16, stage3 contract posonly, step %d)" % policy.checkpoint_step, "",
             "Offline held-out prediction diagnostic on the 20 validation open-lid segments; not closed-loop task success. "
             "Each rule commits to ONE of K=16 stochastic chunks per anchor. `oracle` and `oracle_group` read the demonstrated "
             "future and are NOT deployable.", "",
             f"Anchors: {len(rows)} ({len(episodes)} episodes x {len(fractions)} fractions); momentum rules need 32 steps of "
             f"segment history and cover {len(momentum_rows)} anchors ({len(rows) - len(momentum_rows)} skipped). "
             f"Runtime {report['seconds']:.0f} s" + (" (guard reduced the run to 10 episodes)" if reduced else "") + ".", ""]
    lines += main_table(overall, "Main table (all anchors; momentum rules on their subset)")
    lines += main_table(momentum_subset, f"Fair comparison on the {len(momentum_rows)} anchors where momentum is available")
    for g in ("upper_body", "trunk", "left_arm", "right_arm"):
        lines += [f"### Cosine to demo intent by fraction: {g}", "",
                  "| Rule | " + " | ".join(str(f) for f in fractions) + " |", "|---|" + "---:|" * len(fractions)]
        for rule in rules:
            lines.append(f"| {rule} | " + " | ".join(fmt(by_fraction[str(f)][rule][g]["cos"]) for f in fractions) + " |")
        lines.append("")
    lines += ["### Base group: mean signed vx of the committed chunk vs the demo (m/s, mean over the 32 steps)", "",
              "| Rule | " + " | ".join(str(f) for f in fractions) + " | overall | mean abs err |", "|---|" + "---:|" * (len(fractions) + 2),
              "| demo | " + " | ".join(fmt(base["by_fraction"][str(f)]["demo_vx"]) for f in fractions) +
              f" | {fmt(base['overall']['demo_vx'])} | - |"]
    for rule in rules:
        if rule == "oracle_group":
            continue
        lines.append(f"| {rule} | " + " | ".join(fmt(base["by_fraction"][str(f)]["rules"][rule]["vx"]) for f in fractions) +
                     f" | {fmt(base['overall']['rules'][rule]['vx'])} | {fmt(base['overall']['rules'][rule]['abs_err'])} |")
    lines += ["", "### Commitment: cosine between the intents chosen at consecutive anchors (t0 = start + 0.1 len, step 32)", "",
              f"First {len(commitment['episodes'])} validation episodes, {commitment['n_pairs']} consecutive pairs. "
              "`single` = expected value over all K x K sample pairs.", "",
              "| Rule | trunk | left_arm | right_arm | upper_body (18-D) | pairs |", "|---|---:|---:|---:|---:|---:|",
              "| demo (reference) | " + " | ".join(fmt(commitment["demo"][g]) for g in GROUPS) + f" | {commitment['n_pairs']} |"]
    for rule in commit_rules:
        lines.append(f"| {rule} | " + " | ".join(fmt(commitment["rules"][rule][g]) for g in GROUPS) +
                     f" | {commitment['n_pairs_by_rule'][rule]} |")
    lines += ["", "### Agreement with the oracle pick (rank 1 = the oracle's sample)", "",
              "| Rule | mean oracle rank of the pick | fraction equal to oracle |", "|---|---:|---:|"]
    for rule, values in oracle_rank.items():
        lines.append(f"| {rule} | {fmt(values['mean_rank'], 2)} | {fmt(values['frac_equal_oracle'], 2)} |")
    lines += ["", f"Sample agreement diagnostics (mean pairwise cosine among the K samples): upper-body intents "
              f"{fmt(diagnostics['ub_pairwise_cos'])}, centred normalized chunks {fmt(diagnostics['chunk_pairwise_cos'])}, "
              f"predicted future features {fmt(diagnostics['future_pairwise_cos'])}.", "",
              "## Conclusion", "",
              f"1. Best deployable rule on all {len(rows)} anchors by upper-body cosine: `{best}` "
              f"({fmt(ub(overall, best))}; mean {fmt(ub(overall, 'mean'))}, single {fmt(ub(overall, 'single'))}); best single-sample "
              f"committing rule: `{best_commit}` ({fmt(ub(overall, best_commit))}; per group "
              + ", ".join(f"{g} {fmt(overall[best_commit][g]['cos'])}" for g in POSITION_PAIRS) + ").",
              f"2. `{best_commit}` {verdict(d_mean)} `mean` ({d_mean:+.3f} upper-body cos) and {verdict(d_single)} `single` "
              f"({d_single:+.3f}); material = |delta| >= {material}.",
              f"3. Best momentum-based rule on the {len(momentum_rows)}-anchor subset: `{best_mom}` "
              f"({fmt(ub(momentum_subset, best_mom))}) vs mean {fmt(ub(momentum_subset, 'mean'))} and `{best}` "
              f"{fmt(ub(momentum_subset, best))} on the same anchors ({d_mom:+.3f} vs the better of the two).",
              f"4. Oracle gap: best-of-16 reaches {fmt(ub(overall, 'oracle'))} upper-body cos "
              f"(per group " + ", ".join(f"{fmt(overall['oracle_group'][g]['cos'])}" for g in POSITION_PAIRS) +
              f"), so the deployable rules recover {fmt((ub(overall, best) - ub(overall, 'mean')) / max(ub(overall, 'oracle') - ub(overall, 'mean'), 1e-6), 2)} "
              "of the mean-to-oracle gap.",
              f"5. Commitment (consecutive-anchor cosine, upper body): demo {fmt(commitment['demo']['upper_body'])}, "
              + ", ".join(f"{rule} {fmt(commitment['rules'][rule]['upper_body'])}"
                          for rule in dict.fromkeys(("single", "mean", best, "future_medoid", "medoid_momentum")) if rule in commitment["rules"])
              + "; offline diagnostic only, closed-loop docking/manipulation success is not measured here.", ""]
    (args.out / "README.md").write_text("\n".join(lines))
    log(f"done: {len(rows)} anchors, {report['seconds']:.0f} s, best deployable {best} ub cos {ub(overall, best):.3f} "
        f"(mean {ub(overall, 'mean'):.3f}, single {ub(overall, 'single'):.3f}, oracle {ub(overall, 'oracle'):.3f})")
    print(json.dumps({"out": str(args.out), "n_anchors": len(rows), "seconds": report["seconds"],
                      "upper_body_cos": {rule: overall[rule]["upper_body"]["cos"] for rule in rules}}, indent=2))


if __name__ == "__main__":
    main()
