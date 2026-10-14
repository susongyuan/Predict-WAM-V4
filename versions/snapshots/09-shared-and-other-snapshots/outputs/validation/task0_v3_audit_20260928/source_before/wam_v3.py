"""WAM v3 (2026-09-28): multi-view WAM + temporal memory + phase-aware view gating, trained on cached DINOv3 features.

Why (progress.md 6.3/6.4): the Task-0 v1/v2 WAM sees ONE frame (proprio_len 1, no visual history), weighs the three
cameras equally, and under-fits the final centimetres even on TRAIN anchors (task0_fit_gap_20260928). v3 adds, on top
of harness.wam_behavior.MultiViewVLAModel (everything else unchanged):

1. Proprio history tokens (the Task-13 stage7b design that first beat the hold baseline): token 0 = min-max current
   state; token j = clip((s_t - s_{t_j}) / std_j, +-5) for lags 4/8/16/32/64 frames and for the CONTRACT START frame.
   The checkpoint's proprio mask (velocities + finger openings) applies to every token (copycat, twice confirmed).
2. Visual memory tokens: the frozen DINOv3 last feature layer pooled to [CLS, mean patch] (2048-D) per view; tokens =
   Linear(clip((pool_t - pool_{t_j}) / std_j, +-5)) for lags 8/32 frames and the contract start. The encoder is frozen,
   so the latent cannot collapse; no second vision model is loaded next to the simulator.
3. Phase-aware view gate: w = softmax(g(per-view visual summary, proprio tokens, memory tokens, contract token)) over
   (head, left wrist, right wrist), applied as an additive attention bias log(V * w_v) on every key of view v (visual +
   memory tokens). Scaling the tokens instead would be undone by the blocks' pre-LayerNorm. Uniform w => bias 0 =>
   bit-identical to the ungated computation (2026-09-28 check), and the gate head is zero-initialised.
   Train-time only, privileged demo labels shape it: phase class (far / near <0.12 m / contact / post) and the
   active-hand distance to its contact point; a weak KL prior (far: head-led; near/contact: active-wrist-led). At
   deployment the gate sees RGB features + proprio only.
4. Future-feature (world-model) targets for all views, not only the head.

History rule shared by training and deployment (`history_frames`): frames t - lag are clamped to the contract start
(the annotated segment start in training, the orchestrator's contract start / handoff step in deployment). Visual
memory frames in training are the even (cached) frames, the start rounded up to even (<= 1 frame from deployment).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from harness import task_profile
from harness.wam_behavior import (ACTION_GROUPS, BehaviorChunkDataset, load_upstream, multiview_model_class,
                                  proprio_keep_mask, HEAD_CAMERA)

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("far", "near", "contact", "post")
NEAR_M = 0.12            # active-hand distance to its contact point below which the approach counts as "near"
CONTACT_FRAMES = 30      # frames after the contact event that still count as "contact" (1 s at 30 Hz)
# Gate prior per phase over (head, active wrist, other wrist); mapped to the model's view order per sample.
GATE_PRIOR = {"far": (0.60, 0.25, 0.15), "near": (0.25, 0.60, 0.15), "contact": (0.25, 0.60, 0.15),
              "post": (0.45, 0.40, 0.15)}
GRIP_COL = {"left": 14, "right": 22}                  # action columns, +1 open / -1 closed
EEF_POS = {"left": slice(17, 20), "right": slice(42, 45)}   # proprio eef position (robot base frame)
ARM_ACT = {"left": slice(7, 14), "right": slice(15, 22)}
ARM_STATE = {"left": slice(3, 10), "right": slice(28, 35)}
TRUNK_ACT, TRUNK_STATE = slice(3, 7), slice(53, 57)
LEFT_WRIST, RIGHT_WRIST = "observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0"


# ----------------------------------------------------------------------------------------------------------- history
def history_frames(t, start, lags):
    """[t - lag clamped to start for each lag] + [start]. THE rule for proprio and memory history (train + deploy)."""
    t, start = int(t), int(start)
    if start > t:
        raise ValueError(f"contract start {start} after current frame {t}")
    return [max(t - int(L), start) for L in lags] + [start]


def pool_from_hidden(h, num_register_tokens=4):
    """(..., 1+R+P, D) hidden states of the memory layer -> (..., 2, D) float32 [CLS, mean patch]."""
    h = h.float()
    return torch.stack([h[..., 0, :], h[..., 1 + num_register_tokens:, :].mean(dim=-2)], dim=-2)


def ceil_even(x):
    return int(x) + (int(x) % 2)


# ------------------------------------------------------------------------------------------------------ phase labels
def reward_frames(task_id=None):
    """{episode: first frame with next.reward > 0 or None} from the demo parquet files (train-time labels only)."""
    import pyarrow.parquet as pq
    out = {}
    for f in sorted((task_profile.DATA / "data").glob("chunk-*/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "next.reward"])
        ep = np.asarray(t.column("episode_index")); fr = np.asarray(t.column("frame_index")); r = np.asarray(t.column("next.reward"))
        for e in np.unique(ep):
            m = (ep == e) & (r > 0)
            out[int(e)] = int(fr[m].min()) if m.any() else None
    return out


def segment_event(segment, action, state, hold_hand=None, reward_frame=None):
    """(active hand, event frame, reason) for one manipulation segment, or (None, None, reason).

    pick-like ('pick up from' / 'grasp'): the one hand with an open->close command transition that is still closed at
    the segment end; event = its last such transition. 'press': the hand NOT holding the object (hold_hand = the pick's
    hand); event = the first rewarded frame inside the segment. release-like ('place on' / 'place in' / 'put'): the
    holding hand's first close->open transition."""
    skill, s, e = segment["skill"], int(segment["start"]), min(int(segment["end"]), len(action)) - 1
    if skill == "press":
        if hold_hand is None:
            return None, None, "press: holding hand unknown"
        if reward_frame is None or not s <= reward_frame <= e:
            return None, None, "press: no reward frame inside the segment"
        return ("left" if hold_hand == "right" else "right"), int(reward_frame), "press"
    if skill in ("pick up from", "grasp"):
        cands = []
        for hand, col in GRIP_COL.items():
            g = action[:, col]
            tr = [t for t in range(max(s, 1), e + 1) if g[t - 1] > 0 >= g[t]]
            if tr and g[e] <= 0:
                cands.append((hand, tr[-1]))
        if len(cands) != 1:
            return None, None, f"pick: {len(cands)} closing hands"
        return cands[0][0], int(cands[0][1]), "pick"
    if skill.startswith("place") or skill == "put":
        hands = [hold_hand] if hold_hand else list(GRIP_COL)
        cands = []
        for hand in hands:
            g = action[:, GRIP_COL[hand]]
            tr = [t for t in range(max(s, 1), e + 1) if g[t - 1] <= 0 < g[t]]
            if tr:
                cands.append((hand, tr[0]))
        if len(cands) != 1:
            return None, None, f"place: {len(cands)} releasing hands"
        return cands[0][0], int(cands[0][1]), "place"
    return None, None, f"unsupported skill {skill!r}"


def episode_phase_labels(segments, action, state, reward_frame):
    """Per-frame arrays over the episode: phase (int8, -1 = no label), dist (float32, m; active eef to its event
    position, robot base frame), hand (int8: 0 left, 1 right, -1 none), plus a per-segment report."""
    n = len(action)
    phase = np.full(n, -1, np.int8); dist = np.full(n, np.nan, np.float32); hand = np.full(n, -1, np.int8)
    report, hold_hand = [], None
    for seg in segments:
        if seg["skill"] == "move to":
            continue
        h, ev, why = segment_event(seg, action, state, hold_hand, reward_frame)
        report.append({"skill": seg["skill"], "start": int(seg["start"]), "end": int(seg["end"]), "hand": h, "event": ev,
                       "reason": why})
        if seg["skill"] in ("pick up from", "grasp") and h is not None:
            hold_hand = h
        if h is None:
            continue
        lo, hi = int(seg["start"]), min(int(seg["end"]), n)
        t = np.arange(lo, hi)
        d = np.linalg.norm(state[lo:hi, EEF_POS[h]].astype(np.float64) - state[ev, EEF_POS[h]].astype(np.float64), axis=1)
        cls = np.where(t < ev, np.where(d >= NEAR_M, 0, 1), np.where(t < ev + CONTACT_FRAMES, 2, 3))
        phase[lo:hi] = cls; dist[lo:hi] = d; hand[lo:hi] = 0 if h == "left" else 1
    return phase, dist, hand, report


def gate_prior(phase, hand, views):
    """(B, V) prior over the model's views for phase classes (B,) and active hands (B,) (0 left, 1 right)."""
    idx = {v: i for i, v in enumerate(views)}
    out = np.zeros((len(phase), len(views)), np.float32)
    for b, (p, h) in enumerate(zip(phase, hand)):
        if p < 0:
            continue
        head_w, act_w, oth_w = GATE_PRIOR[PHASES[p]]
        act, oth = (LEFT_WRIST, RIGHT_WRIST) if h == 0 else (RIGHT_WRIST, LEFT_WRIST)
        out[b, idx[HEAD_CAMERA]], out[b, idx[act]], out[b, idx[oth]] = head_w, act_w, oth_w
    return out


# ---------------------------------------------------------------------------------------------------------- dataset
class FeatureCache:
    """Read-only access to scripts/cache_dino_features.py output: rows by frame, bf16 bits as int16 torch tensors."""

    def __init__(self, root, views):
        self.root, self.views = Path(root), list(views)
        self._rows, self._feat, self._pool = {}, {}, {}

    def _open(self, view, ep):
        key = (view, ep)
        if key not in self._rows:
            d = self.root / view
            meta = json.loads((d / f"episode_{ep}.meta.json").read_text())
            if not meta.get("complete"):
                raise ValueError(f"incomplete feature cache {view} episode {ep}")
            frames = np.load(d / f"episode_{ep}.frames.npy")
            self._rows[key] = {int(f): i for i, f in enumerate(frames)}
            self._feat[key] = np.load(d / f"episode_{ep}.feat.npy", mmap_mode="r")
            self._pool[key] = np.load(d / f"episode_{ep}.pool.npy", mmap_mode="r")
        return key

    def row(self, view, ep, frame):
        key = self._open(view, ep)
        try:
            return self._rows[key][int(frame)]
        except KeyError:
            raise KeyError(f"frame {frame} of episode {ep} not cached for {view}") from None

    def feat(self, view, ep, frame):
        key = self._open(view, ep)
        return np.asarray(self._feat[key][self.row(view, ep, frame)])          # (L, T, D) uint16

    def pool(self, view, ep, frames):
        key = self._open(view, ep)
        return np.asarray(self._pool[key][[self.row(view, ep, f) for f in frames]])   # (n, 2, D) float32


def bf16_from_bits(x):
    """uint16/int16 bf16 bit pattern (numpy or torch) -> torch.bfloat16 tensor (no copy for torch int16)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x).view(np.int16))
    return x.view(torch.bfloat16)


class WamV3Dataset(BehaviorChunkDataset):
    """Manipulation-scope anchors of BehaviorChunkDataset (same anchors/conditions as v1/v2) with cached features,
    proprio + visual-memory history, all-view future targets and train-time phase labels."""

    def __init__(self, split, *, feature_cache, views, proprio_lags, memory_lags, contract_targets,
                 contract_targets_sha256, seed=0, rewards=None):
        super().__init__(split, stride=2, chunk=32, future_offset=32, image_size=224, cond_mode="contract_proto",
                         instance_prob=0.0, seed=seed, contract_targets=contract_targets,
                         contract_targets_sha256=contract_targets_sha256, skip_unsupported_contracts=True,
                         scope="manipulation", in_segment=True, views=views)
        self.cache = FeatureCache(feature_cache, views)
        self.proprio_lags, self.memory_lags = [int(v) for v in proprio_lags], [int(v) for v in memory_lags]
        if any(L % self.stride for L in self.memory_lags):
            raise ValueError("memory lags must be multiples of the frame stride (cached frames)")
        rewards = reward_frames() if rewards is None else rewards
        self.labels, self.label_report = {}, {}
        for ep in self.episodes:
            action, state = self.lowdim(ep)
            ph, di, ha, rep = episode_phase_labels(self.skills.episodes[ep]["segments"], np.asarray(action),
                                                   np.asarray(state), rewards.get(ep))
            self.labels[ep] = (ph, di, ha)
            self.label_report[ep] = rep

    def label_summary(self):
        counts = np.zeros(len(PHASES) + 1, np.int64)
        for ep, a in self.anchors:
            counts[int(self.labels[ep][0][a]) + 1] += 1
        bad = [(ep, r) for ep, reps in self.label_report.items() for r in reps if r["hand"] is None]
        return {"anchors_by_phase": {("unlabelled" if i == 0 else PHASES[i - 1]): int(c) for i, c in enumerate(counts)},
                "unlabelled_segments": len(bad), "unlabelled_examples": bad[:10]}

    def __getitem__(self, i):
        ep, a = self.anchors[i]
        action, state = self.lowdim(ep)
        seg = self.skills.segment(ep, a)
        start = int(seg["start"])
        pframes = [a] + history_frames(a, start, self.proprio_lags)
        mframes = [a] + history_frames(a, ceil_even(start), self.memory_lags)
        fut = a + self.future_offset
        feats = np.stack([self.cache.feat(v, ep, a) for v in self.views])                  # (V, L, T, D) uint16
        future = np.stack([self.cache.feat(v, ep, fut)[-1, 5:] for v in self.views])       # (V, P, D) last layer patches
        pool = np.stack([self.cache.pool(v, ep, mframes).reshape(len(mframes), -1) for v in self.views])  # (V, M+1, 2D)
        ph, di, ha = self.labels[ep]
        cond, cls = self.condition(ep, a)
        return {"feats": torch.from_numpy(feats.view(np.int16)), "future_feats": torch.from_numpy(future.view(np.int16)),
                "state": torch.from_numpy(np.asarray(state[pframes], dtype=np.float32)),
                "pool": torch.from_numpy(pool.astype(np.float32)),
                "action_sequence": torch.from_numpy(np.array(action[a:a + self.chunk], dtype=np.float32)),
                "task_cond": cond, "skill_class": cls, "episode": ep, "anchor": a, "contract_start": start,
                "phase": int(ph[a]), "dist": float(di[a]) if ph[a] >= 0 else float("nan"), "active_hand": int(ha[a])}


# ------------------------------------------------------------------------------------------------------------ model
_V3_CLASS = None


def v3_model_class():
    """WamV3Model(num_views, num_memory, **VLAModel kwargs): MultiViewVLAModel + memory tokens + view gate + phase
    heads + all-view future-feature decoder. Blocks run exactly the upstream DiTBlock with an additive attention bias."""
    global _V3_CLASS
    if _V3_CLASS is not None:
        return _V3_CLASS
    vla, _ = load_upstream()
    MV = multiview_model_class()

    class WamV3Model(MV):
        def __init__(self, num_views, num_memory, memory_dim=2048, phase_hidden=512, **kwargs):
            super().__init__(num_views, **kwargs)
            H = self.hidden_dim
            self.num_memory = int(num_memory)
            self.mem_proj = torch.nn.Linear(memory_dim, H)
            self.mem_slot_emb = torch.nn.Parameter(torch.zeros(1, 1, self.num_memory, H))
            self.type_emb_mem = torch.nn.Parameter(torch.zeros(1, 1, 1, H))
            torch.nn.init.normal_(self.mem_slot_emb, std=0.02)
            torch.nn.init.normal_(self.type_emb_mem, std=0.02)
            gate_in = (num_views + self.proprio_len + num_views * self.num_memory + 1) * H
            self.phase_net = torch.nn.Sequential(torch.nn.LayerNorm(gate_in), torch.nn.Linear(gate_in, phase_hidden),
                                                 torch.nn.GELU(), torch.nn.Linear(phase_hidden, phase_hidden), torch.nn.GELU())
            self.gate_head = torch.nn.Linear(phase_hidden, num_views)
            torch.nn.init.zeros_(self.gate_head.weight)
            torch.nn.init.zeros_(self.gate_head.bias)
            self.phase_head = torch.nn.Linear(phase_hidden, len(PHASES))
            self.dist_head = torch.nn.Linear(phase_hidden, 1)
            if self.use_future_feat:  # one query set per view (view-major), same decoder recipe as upstream
                dec = self.future_feat_decoder
                self.future_patches = dec.num_queries
                layer = dec.decoder.layers[0]
                self.future_feat_decoder = vla.FutureFeatureDecoder(
                    num_queries=num_views * dec.num_queries, hidden_dim=H, out_dim=dec.out_dim,
                    num_heads=layer.self_attn.num_heads, num_layers=len(dec.decoder.layers))

        def forward(self, t, noisy_actions, qpos_history=None, dino_features_list=None, task_cond=None, memory=None):
            """memory: (B, V, M, memory_dim) normalized memory inputs. Returns final_pred, cond_tokens, gate (B, V),
            phase_logits (B, 4), log_dist (B,)."""
            B = noisy_actions.shape[0]
            t_emb = self.time_mlp(t)
            x_action = self.action_proj(noisy_actions) + self.action_pos_emb[:, :noisy_actions.shape[1], :] + self.type_emb_action
            x_proprio_real = self.proprio_proj(qpos_history) + self.proprio_pos_emb[:, :qpos_history.shape[1], :] + self.type_emb_proprio
            x_proprio = torch.zeros_like(x_proprio_real)
            regs = self.register_tokens.expand(B, -1, -1)
            tc = self.task_cond_proj(task_cond).unsqueeze(1) + self.type_emb_task_cond
            V, N = dino_features_list[0].shape[1:3]
            if V != self.num_views or any(f.shape[:3] != (B, V, N) for f in dino_features_list):
                raise ValueError(f"expected per-layer (B, {self.num_views}, N, D) features")
            fused = self.concat_fusion([f.reshape(B * V, N, f.shape[-1]) for f in dino_features_list])
            vis = self.dino_adapters[0](fused)
            Q = vis.shape[1]
            vis = vis.reshape(B, V, Q, -1) + self.type_emb_dino_layers[0].unsqueeze(1) + self.view_emb[:V].unsqueeze(0).squeeze(2)
            if memory.shape[:3] != (B, V, self.num_memory):
                raise ValueError(f"memory {tuple(memory.shape)} != (B, {V}, {self.num_memory}, D)")
            mem = (self.mem_proj(memory.to(vis.dtype)) + self.mem_slot_emb + self.type_emb_mem
                   + self.view_emb[:V].unsqueeze(0).squeeze(2))                      # (B, V, M, H)
            g_in = torch.cat([vis.mean(dim=2).reshape(B, -1), x_proprio_real.reshape(B, -1), mem.reshape(B, -1),
                              tc.reshape(B, -1)], dim=-1)
            hph = self.phase_net(g_in.float())
            gate = torch.softmax(self.gate_head(hph).float(), dim=-1)                          # (B, V)
            phase_logits, log_dist = self.phase_head(hph).float(), self.dist_head(hph).float().squeeze(-1)
            tokens = [x_action, x_proprio, regs, tc, vis.reshape(B, V * Q, -1), mem.reshape(B, V * self.num_memory, -1)]
            x = torch.cat(tokens, dim=1)
            proprio_start = x_action.shape[1]
            proprio_end = proprio_start + x_proprio.shape[1]
            # additive attention bias log(V * w_v) on every key of view v (visual and memory tokens)
            S = x.shape[1]
            log_w = torch.log(gate * V)
            n_front = S - V * Q - V * self.num_memory
            bias = torch.cat([torch.zeros(B, n_front, device=x.device),
                              log_w.repeat_interleave(Q, dim=1), log_w.repeat_interleave(self.num_memory, dim=1)], dim=1)
            heads = self.blocks[0].attn1.num_heads
            mask = bias[:, None, None, :].expand(B, heads, S, S).reshape(B * heads, S, S)
            state_injected = False
            for i, block in enumerate(self.blocks):
                if i == self.state_inject_start and not state_injected:
                    x[:, proprio_start:proprio_end, :] = x_proprio_real
                    state_injected = True
                x = block(x, t_emb, attn_mask=mask)
            x = self.final_norm(x)
            return {"final_pred": self.output_proj(x[:, :self.action_len, :]), "cond_tokens": x[:, self.action_len:, :],
                    "gate": gate, "phase_logits": phase_logits, "log_dist": log_dist}

    _V3_CLASS = WamV3Model
    return WamV3Model


def v3_settings(config):
    v3 = config.model.get("v3", None)
    if v3 is None:
        raise ValueError("config has no model.v3 block")
    out = {"proprio_lags": [int(v) for v in v3.proprio_lags], "memory_lags": [int(v) for v in v3.memory_lags],
           "stats_path": str(v3.stats_path), "phase_hidden": int(v3.get("phase_hidden", 512))}
    if int(config.common.proprio_len) != len(out["proprio_lags"]) + 2:
        raise ValueError(f"common.proprio_len must be 1 + len(proprio_lags) + 1 (start) = {len(out['proprio_lags']) + 2}")
    return out


def create_v3_action_model(config, **factory_kwargs):
    """Upstream ModelFactory mapping (config -> VLAModel kwargs), constructing WamV3Model instead."""
    from harness.wam_behavior import model_views
    _, runner = load_upstream()
    views = model_views(config)
    s = v3_settings(config)
    cls = v3_model_class()
    upstream_cls = runner.VLAModel
    runner.VLAModel = lambda **kw: cls(num_views=len(views), num_memory=len(s["memory_lags"]) + 1,
                                       phase_hidden=s["phase_hidden"], **kw)
    try:
        model = runner.ModelFactory.create_action_model(config, **factory_kwargs)
    finally:
        runner.VLAModel = upstream_cls
    assert isinstance(model, cls)
    return model


V3_NEW_PREFIXES = ("mem_proj.", "mem_slot_emb", "type_emb_mem", "phase_net.", "gate_head.", "phase_head.", "dist_head.")


def load_v3_state(model, state_dict):
    """Load a v1/v2 multi-view checkpoint (or a v3 one) into WamV3Model. Allowed differences, all reported: the new v3
    modules (kept at init), the longer sin-cos proprio_pos_emb (prefix must match), and the future-feature decoder
    queries (source head queries tiled to every view). Anything else must load strictly."""
    own = model.state_dict()
    sd = dict(state_dict)
    notes = {}
    if "proprio_pos_emb" in sd and sd["proprio_pos_emb"].shape != own["proprio_pos_emb"].shape:
        src, dst = sd["proprio_pos_emb"], own["proprio_pos_emb"]
        n = min(src.shape[1], dst.shape[1])
        if not torch.allclose(src[:, :n].float(), dst[:, :n].float().cpu(), atol=1e-6):
            raise RuntimeError("proprio_pos_emb prefix mismatch")
        sd["proprio_pos_emb"] = dst.detach().clone()
        notes["proprio_pos_emb"] = f"{tuple(src.shape)} -> {tuple(dst.shape)} (sin-cos prefix identical)"
    key = "future_feat_decoder.query_embed"
    if key in sd and sd[key].shape != own[key].shape:
        reps = own[key].shape[1] // sd[key].shape[1]
        if reps * sd[key].shape[1] != own[key].shape[1]:
            raise RuntimeError(f"cannot tile {key} {tuple(sd[key].shape)} -> {tuple(own[key].shape)}")
        sd[key] = sd[key].repeat(1, reps, 1)
        notes[key] = f"head queries tiled x{reps}"
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(V3_NEW_PREFIXES)]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"v3 load: unexpected missing={missing} unexpected={list(result.unexpected_keys)}")
    notes["new_v3_modules_at_init"] = sorted({k.split(".")[0] for k in result.missing_keys})
    return notes


# ---------------------------------------------------------------------------------------------------------- wrapper
class WamV3Wrapper(torch.nn.Module):
    """Normalization + losses + sampler around WamV3Model. Inputs are ALWAYS raw: state history (B, K, 61) = [current,
    lag frames..., contract start]; pool history (B, V, M+1, 2048) = [current, lag frames..., contract start]; DINO
    features per layer (B, V, T, D) bf16. The same methods serve training, offline evaluation and the policy server."""

    def __init__(self, action_model, *, views, stats, norm_stats_path, proprio_mask, train_cfg=None, vision_encoder=None,
                 feat_layers=(-12, -8, -4), num_register_tokens=4, device="cuda"):
        super().__init__()
        self.action_model, self.vision_encoder = action_model, vision_encoder
        self.views, self.feat_layers, self.num_register_tokens = list(views), list(feat_layers), int(num_register_tokens)
        self.device = torch.device(device)
        ns = json.loads(Path(norm_stats_path).read_text())
        if ns.get("action_repr", "absolute") != "absolute":
            raise ValueError("v3 uses absolute actions")
        st = ns["behavior1k"]
        for name, arr in (("action_min", st["action"]["min"]), ("action_max", st["action"]["max"]),
                          ("state_min", st["state"]["min"]), ("state_max", st["state"]["max"])):
            self.register_buffer(name, torch.tensor(arr, dtype=torch.float32))
        self.register_buffer("proprio_keep", torch.from_numpy(proprio_keep_mask(proprio_mask).astype(np.float32)))
        self.register_buffer("prop_std", torch.tensor(stats["proprio_diff_std"], dtype=torch.float32))   # (K-1, 61)
        self.register_buffer("mem_std", torch.tensor(stats["memory_diff_std"], dtype=torch.float32))     # (V, M, 2048)
        if list(stats["views"]) != self.views:
            raise ValueError(f"stats views {stats['views']} != model views {self.views}")
        self.proprio_mask_name = proprio_mask
        self.train_cfg = dict(train_cfg or {})

    # ---- normalization (shared) ----
    @staticmethod
    def _minmax(x, lo, hi):
        den = (hi - lo).clone()
        den[den < 1e-6] = 1.0
        return 2 * (x - lo) / den - 1

    def normalize_action(self, a):
        return self._minmax(a, self.action_min, self.action_max)

    def denormalize_action(self, x):
        den = (self.action_max - self.action_min).clone()
        den[den < 1e-6] = 1.0
        return (x + 1) / 2 * den + self.action_min

    def normalize_state(self, state):
        """(B, K, 61) raw [current, history..., start] -> (B, K, 61): minmax current, clipped standardized diffs."""
        if state.dim() != 3 or state.shape[1] != self.prop_std.shape[0] + 1:
            raise ValueError(f"state history {tuple(state.shape)} != (B, {self.prop_std.shape[0] + 1}, 61)")
        cur = self._minmax(state[:, :1], self.state_min, self.state_max)
        diff = ((state[:, :1] - state[:, 1:]) / self.prop_std[None]).clamp(-5.0, 5.0)
        return torch.cat([cur, diff], dim=1) * self.proprio_keep

    def normalize_memory(self, pool):
        """(B, V, M+1, 2048) raw pooled [current, history..., start] -> (B, V, M, 2048) clipped standardized diffs."""
        if pool.dim() != 4 or pool.shape[1:3] != (self.mem_std.shape[0], self.mem_std.shape[1] + 1):
            raise ValueError(f"pool history {tuple(pool.shape)} != (B, {self.mem_std.shape[0]}, {self.mem_std.shape[1] + 1}, 2048)")
        return ((pool[:, :, :1] - pool[:, :, 1:]) / self.mem_std[None]).clamp(-5.0, 5.0)

    # ---- vision (deployment/offline; training uses cached features) ----
    @torch.no_grad()
    def encode(self, pixels):
        """(B, V, 3, H, W) ImageNet-normalized -> (per-layer (B, V, T, D) bf16, pooled (B, V, 2, D) float32)."""
        B, V = pixels.shape[:2]
        px = pixels.reshape(B * V, *pixels.shape[2:]).to(self.device, torch.bfloat16)
        hs = self.vision_encoder(pixel_values=px, output_hidden_states=True, return_dict=True).hidden_states
        feats = [hs[l].reshape(B, V, *hs[l].shape[1:]) for l in self.feat_layers]
        return feats, pool_from_hidden(feats[-1], self.num_register_tokens)

    # ---- training loss ----
    def forward(self, batch, lambdas):
        dev = self.device
        feats_all = bf16_from_bits(batch["feats"].to(dev, non_blocking=True))          # (B, V, L, T, D)
        feats = [feats_all[:, :, i] for i in range(feats_all.shape[2])]
        x1 = self.normalize_action(batch["action_sequence"].to(dev, torch.float32))
        qpos = self.normalize_state(batch["state"].to(dev, torch.float32))
        memory = self.normalize_memory(batch["pool"].to(dev, torch.float32))
        tc = batch["task_cond"].to(dev, torch.float32)
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.sigmoid(torch.randn(B, device=dev) * float(self.train_cfg.get("time_sigma", 1.0)) + float(self.train_cfg.get("time_mu", 0.0)))
        x_t = (1 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
        out = self.action_model(t, noisy_actions=x_t, qpos_history=qpos, dino_features_list=feats, task_cond=tc, memory=memory)
        loss_mse = F.mse_loss(out["final_pred"].float(), (x1 - x0).float())
        losses = {"mse": loss_mse}
        if self.action_model.use_future_feat:
            fut = bf16_from_bits(batch["future_feats"].to(dev, non_blocking=True))   # (B, V, P, D)
            pred = self.action_model.future_feat_decoder(out["cond_tokens"])
            vla, _ = load_upstream()
            losses["future"] = vla.future_feature_cosine_loss(pred, fut.reshape(B, -1, fut.shape[-1]))
        phase = batch["phase"].to(dev)
        valid = phase >= 0
        gate = out["gate"]
        if valid.any():
            prior = torch.from_numpy(gate_prior(batch["phase"].numpy(), batch["active_hand"].numpy(), self.views)).to(dev)
            kl = (prior * (torch.log(prior.clamp_min(1e-8)) - torch.log(gate.clamp_min(1e-8)))).sum(-1)
            losses["gate_kl"] = kl[valid].mean()
            losses["phase_ce"] = F.cross_entropy(out["phase_logits"][valid], phase[valid].long())
            dist = batch["dist"].to(dev, torch.float32)
            losses["log_dist"] = F.smooth_l1_loss(out["log_dist"][valid], torch.log(dist[valid] + 0.02))
        else:
            zero = loss_mse * 0
            losses.update(gate_kl=zero, phase_ce=zero, log_dist=zero)
        total = loss_mse + sum(float(lambdas.get(k, 0.0)) * v for k, v in losses.items() if k != "mse")
        info = {k: float(v.detach()) for k, v in losses.items()}
        info["gate_mean"] = gate.detach().mean(0).tolist()
        return total, info, out

    # ---- sampling (validation + deployment) ----
    @torch.no_grad()
    def sample_actions(self, feats, state, pool, task_cond, num_steps=10, generator=None):
        """feats: per-layer (B, V, T, D); state (B, K, 61) raw; pool (B, V, M+1, 2048) raw; -> absolute (B, 32, 23)
        actions (float32) + the last forward's gate/phase outputs."""
        qpos = self.normalize_state(state.to(self.device, torch.float32))
        memory = self.normalize_memory(pool.to(self.device, torch.float32))
        tc = task_cond.to(self.device, torch.float32)
        B = qpos.shape[0]
        x = torch.randn(B, self.action_model.action_len, self.action_min.numel(), device=self.device, generator=generator)
        out = None
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.action_model(t, noisy_actions=x, qpos_history=qpos, dino_features_list=feats, task_cond=tc, memory=memory)
            x = x + out["final_pred"].float() / num_steps
        return self.denormalize_action(x), {"gate": out["gate"].float(), "phase_logits": out["phase_logits"].float(),
                                            "log_dist": out["log_dist"].float()}


# ------------------------------------------------------------------------------------------------------- deployment
class V3Context:
    """Per-contract raw buffers for deployment: one raw state and one pooled vector per view for EVERY control step
    since the contract start (reset at each contract start). `inputs()` applies `history_frames` exactly as training."""

    def __init__(self, proprio_lags, memory_lags, views):
        self.proprio_lags, self.memory_lags, self.views = list(proprio_lags), list(memory_lags), list(views)
        self.reset()

    def reset(self):
        self.states, self.pools = [], []

    def push(self, state, pooled):
        """state (61,) raw; pooled (V, 2, D) float32 of THIS control step."""
        if np.shape(pooled)[0] != len(self.views):
            raise ValueError("pooled must hold one [CLS, mean patch] pair per view")
        self.states.append(np.asarray(state, np.float32).copy())
        self.pools.append(np.asarray(pooled, np.float32).reshape(len(self.views), -1).copy())

    def inputs(self):
        """(state history (K, 61), pool history (V, M+1, 2D)) for the current (last pushed) step."""
        if not self.states:
            raise RuntimeError("no step pushed since the contract start")
        t = len(self.states) - 1
        pf = [t] + history_frames(t, 0, self.proprio_lags)
        mf = [t] + history_frames(t, 0, self.memory_lags)
        return np.stack([self.states[i] for i in pf]), np.stack([self.pools[i] for i in mf], axis=1)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
