"""BEHAVIOR Task 13 training glue for LiLa-WAM: dataset, skill-VTT conditioning, model loading.

Upstream code in third_party/LiLa-WAM is imported unmodified. Deviations from the RoboTwin
recipe are explicit here: 61D state / 23D action, BEHAVIOR train statistics, square 224 head
camera, fp32 master weights with bf16 autocast, and skill-level Visual Transition Token (VTT)
conditioning built from official skill annotations instead of one task-level constant.
"""
import importlib
import hashlib
import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/behavior-1k/2026-challenge-demos"
LOWDIM = ROOT / "data/lowdim_cache"
FRAMES = ROOT / "data/frame_cache"
VTT_DIR = ROOT / "data/skill_vtt_task013"
HEAD_CAMERA = "observation.rgb.zed_link_camera_0"
LEFT_WRIST_CAMERA = "observation.rgb.left_realsense_link_camera_0"
RIGHT_WRIST_CAMERA = "observation.rgb.right_realsense_link_camera_0"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ACTION_GROUPS = {"base": (0, 3), "trunk": (3, 7), "left_arm": (7, 14), "left_gripper": (14, 15),
                 "right_arm": (15, 22), "right_gripper": (22, 23)}
STATE_SLICES = {"base_qvel": (0, 3), "arm_left_qpos": (3, 10), "arm_left_qvel": (10, 17), "eef_left_pos": (17, 20),
                "eef_left_quat": (20, 24), "gripper_left_qpos": (24, 26), "gripper_left_qvel": (26, 28),
                "arm_right_qpos": (28, 35), "arm_right_qvel": (35, 42), "eef_right_pos": (42, 45), "eef_right_quat": (45, 49),
                "gripper_right_qpos": (49, 51), "gripper_right_qvel": (51, 53), "trunk_qpos": (53, 57), "trunk_qvel": (57, 61)}
PROPRIO_MASKS = {
    # Every velocity component is hidden from the policy: measured velocity otherwise acts as a copy of the
    # previous command ("copycat"), so a stopped robot stays stopped (outputs/validation/velocity_copycat_*).
    "positions_only": ("base_qvel", "arm_left_qvel", "gripper_left_qvel", "arm_right_qvel", "gripper_right_qvel", "trunk_qvel"),
    "no_base_velocity": ("base_qvel",),
    "full": (),
}


# Action representation contracts. `upper_body_delta`: the position-controlled trunk/arm targets are learned
# as (target - current joint position) so a copy-state hold is exactly zero and the flow-matching loss is no
# longer dominated by the base velocity dims (progress.md 2026-09-22 17:55). Base and grippers stay absolute.
ACTION_REPRS = {"absolute": (),
                "upper_body_delta": (("trunk", "trunk_qpos"), ("left_arm", "arm_left_qpos"), ("right_arm", "arm_right_qpos"))}


def action_repr_pairs(name):
    """[(action slice, state slice), ...] of the groups expressed as deltas for a named representation."""
    if name is None:
        name = "absolute"
    if name not in ACTION_REPRS:
        raise ValueError(f"Unknown action representation {name!r}; known: {sorted(ACTION_REPRS)}")
    return [(ACTION_GROUPS[a], STATE_SLICES[s]) for a, s in ACTION_REPRS[name]]


def proprio_keep_mask(name):
    """Boolean keep-mask over the 61D proprio state for a named input contract (None/'full' keeps everything)."""
    if name is None:
        name = "full"
    if name not in PROPRIO_MASKS:
        raise ValueError(f"Unknown proprio mask {name!r}; known: {sorted(PROPRIO_MASKS)}")
    keep = np.ones(61, dtype=bool)
    for group in PROPRIO_MASKS[name]:
        lo, hi = STATE_SLICES[group]
        keep[lo:hi] = False
    return keep


def load_upstream():
    """Import LiLa-WAM's `models` package under a private name (avoids clashing with ./models weights dir)."""
    if "lilawam_models" not in sys.modules:
        pkg = types.ModuleType("lilawam_models")
        pkg.__path__ = [str(ROOT / "third_party/LiLa-WAM/models")]
        sys.modules["lilawam_models"] = pkg
    vla = importlib.import_module("lilawam_models.vla_model_fm")
    runner = importlib.import_module("lilawam_models.model_runner")
    return vla, runner


def normalize_image(rgb_uint8):
    img = rgb_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))


class SkillTable:
    """Per-frame skill class lookup from skill_segments_task013.json (gap-filled, contiguous)."""

    def __init__(self, path=DATA / "skill_segments_task013.json"):
        self.raw = json.loads(Path(path).read_text())
        self.classes = self.raw["classes"]
        self.num_classes = self.raw["num_classes"]
        self.episodes = {int(k): v for k, v in self.raw["episodes"].items()}
        self._bounds = {}
        for ep, info in self.episodes.items():
            segs = info["segments"]
            self._bounds[ep] = (np.array([s["fill_start"] for s in segs]), np.array([s["fill_end"] for s in segs]))

    def segment_index(self, episode, frame):
        starts, ends = self._bounds[episode]
        i = int(np.searchsorted(ends, frame, side="right"))
        if i >= len(starts) or not starts[i] <= frame < ends[i]:
            raise ValueError(f"frame {frame} outside segments of episode {episode}")
        return i

    def segment(self, episode, frame):
        return self.episodes[episode]["segments"][self.segment_index(episode, frame)]


class BehaviorChunkDataset(Dataset):
    """32-step 23D action chunks with head-camera current/future frames and skill-VTT condition.

    cond_mode: 'skill_proto' (class prototype, with `instance_prob` chance of the segment's own
    instance VTT on train), 'contract_proto' (train-only object/navigation-stage means),
    'task_level' (upstream-style single task vector), 'none'. Unsupported contracts fail
    unless skipping is explicitly enabled; every skipped segment/anchor is reported.
    scope='all' keeps the legacy anchor grid. 'manipulation' excludes 'move to' and
    always enforces real annotation [start, end) bounds for every action/future frame.
    in_segment=True also applies those bounds to 'all'; no action dimensions are masked.
    views: optional camera keys (head first). None or one view keeps the legacy (3,H,W) `pixel_values`;
    more views return (V,3,H,W) in `views` order. The future frame is always the head camera (views[0]).
    """

    def __init__(self, split, *, stride=2, chunk=32, future_offset=32, image_size=224,
                 camera=HEAD_CAMERA, cond_mode="skill_proto", instance_prob=0.2, seed=0,
                 episodes=None, max_anchors_per_episode=None, contract_targets=VTT_DIR / "contract_targets.json",
                 contract_targets_sha256=None, skip_unsupported_contracts=False,
                 scope="all", in_segment=False, views=None):
        views = [camera] if views is None else list(views)
        if not views or views[0] != camera or len(set(views)) != len(views):
            raise ValueError(f"views must be unique and start with the head/future camera {camera!r}: {views}")
        if scope not in ("all", "manipulation"):
            raise ValueError(f"Unknown dataset scope {scope!r}; expected 'all' or 'manipulation'")
        if stride <= 0 or chunk <= 0 or future_offset < 0:
            raise ValueError("Positive stride/chunk and nonnegative future_offset required")
        assert split in ("train", "validation")
        assert future_offset % stride == 0, "future frame must be cached"
        if cond_mode not in ("skill_proto", "contract_proto", "task_level", "none") or not 0 <= instance_prob <= 1:
            raise ValueError("Unknown conditioning mode or invalid instance probability")
        from harness.vtt_monitor import precondition_key
        self.scope, self.in_segment = scope, bool(in_segment or scope == "manipulation")
        self.split, self.stride, self.chunk, self.future_offset = split, stride, chunk, future_offset
        self.image_size, self.camera, self.cond_mode, self.instance_prob = image_size, camera, cond_mode, instance_prob
        self.views = views
        self.index = json.loads((LOWDIM / "index.json").read_text())
        self.skills = SkillTable()
        self.rng = np.random.default_rng(seed)
        self.episodes = sorted(int(k) for k, v in self.index["episodes"].items()
                               if v["split"] == split and (episodes is None or int(k) in episodes))
        self.contract_targets_sha256 = None
        self.contract_filter = {"skip_enabled": bool(skip_unsupported_contracts), "original_anchors": 0,
                                "skipped_anchors": 0, "unsupported_segments": []}
        if cond_mode != "none":
            self.prototypes = np.load(VTT_DIR / "prototypes.npy").astype(np.float32)
            self.task_level = np.load(VTT_DIR / "task_level.npy").astype(np.float32)
            inst_index = json.loads((VTT_DIR / "instances_index.json").read_text())
            self.instances = np.load(VTT_DIR / "instances.npy").astype(np.float32)
            self._inst_lookup = {(r["episode"], r["segment"]): i for i, r in enumerate(inst_index)}
        if cond_mode == "contract_proto":
            from harness.orchestrator import ConditionBank
            target_path = ROOT / contract_targets
            target_bytes = target_path.read_bytes()
            self.contract_targets_sha256 = hashlib.sha256(target_bytes).hexdigest()
            if contract_targets_sha256 and self.contract_targets_sha256 != contract_targets_sha256:
                raise ValueError("Contract target SHA256 differs from the pinned training configuration")
            targets = json.loads(target_bytes)
            for source in (DATA / "skill_segments_task013.json", VTT_DIR / "instances_index.json",
                           VTT_DIR / "instances.npy", VTT_DIR / "meta.json"):
                if targets.get("source_sha256", {}).get(str(source.relative_to(ROOT))) != hashlib.sha256(source.read_bytes()).hexdigest():
                    raise ValueError(f"Stale or unverified contract target source: {source}")
            bank = ConditionBank({name: self.prototypes[cid] for name, cid in self.skills.classes.items()}, train_targets=targets)
            self._contract_keys, self._contract_vectors = {}, {}
            for ep in self.episodes:
                segments = self.skills.episodes[ep]["segments"]
                for i, seg in enumerate(segments):
                    key = precondition_key(seg["skill"], seg["objects"])
                    nxt = segments[i + 1] if i + 1 < len(segments) else None
                    next_key = precondition_key(nxt["skill"], nxt["objects"]) if nxt and nxt["skill"] != "move to" else None
                    key = bank.storage_key(key, seg["skill"], next_key)
                    self._contract_keys[ep, i] = key
                    # Navigation needs its immediate manipulation goal; a generic object mean is not a fallback.
                    if key in bank.targets and (seg["skill"] != "move to" or next_key is not None):
                        self._contract_vectors[ep, i] = bank.targets[key]
        self._lowdim, self._packs = {}, {}
        # Exclusions are disjoint, in this order; unsupported data still fails unless explicitly skipped.
        self.anchor_filter = {"scope": scope, "in_segment": self.in_segment,
                              "excluded_by_reason": dict.fromkeys(("unsupported_contract", "navigation", "gap_padding",
                                                                    "segment_boundary", "anchor_limit"), 0),
                              "by_skill": {}, "by_contract": {}, "empty_episodes": []}
        excluded = self.anchor_filter["excluded_by_reason"]
        horizon = max(chunk - 1, future_offset)
        self.anchors = []  # (episode, anchor)
        for ep in self.episodes:
            length = self.index["episodes"][str(ep)]["length"]
            for view in views:
                meta = json.loads((FRAMES / view / f"episode_{ep}.meta.json").read_text())
                if not (meta["complete"] and meta["stride"] == stride and meta["length"] == length):
                    raise ValueError(f"Incomplete/incompatible {view} frame cache for episode {ep}: {meta}")
            info = self.skills.episodes[ep]
            if info["split"] != split or info["length"] != length:
                raise ValueError(f"Cache/annotation split or length mismatch for episode {ep}")
            last = length - 1 - horizon
            a = np.arange(0, last + 1, stride)
            original = a
            self.contract_filter["original_anchors"] += len(a)
            if cond_mode == "contract_proto":
                for i, seg in enumerate(self.skills.episodes[ep]["segments"]):
                    if (ep, i) in self._contract_vectors:
                        continue
                    skipped = (a >= seg["fill_start"]) & (a < seg["fill_end"])
                    row = {"episode": ep, "segment": i, "key": self._contract_keys[ep, i], "anchors": int(skipped.sum())}
                    if not skip_unsupported_contracts:
                        raise ValueError(f"Unsupported {split} contract (explicit skip required): {row}")
                    self.contract_filter["unsupported_segments"].append(row)
                    self.contract_filter["skipped_anchors"] += row["anchors"]
                    a = a[~skipped]
            if self.in_segment:
                keep = np.zeros(len(a), dtype=bool)
                for seg in info["segments"]:
                    assigned = (a >= seg["fill_start"]) & (a < seg["fill_end"])
                    if scope == "manipulation" and seg["skill"] == "move to":
                        excluded["navigation"] += int(assigned.sum())
                        continue
                    real = assigned & (a >= seg["start"]) & (a < seg["end"])
                    safe = real & (a + horizon < min(seg["end"], seg["fill_end"]))
                    excluded["gap_padding"] += int(assigned.sum() - real.sum())
                    excluded["segment_boundary"] += int(real.sum() - safe.sum())
                    keep |= safe
                a = a[keep]
            if max_anchors_per_episode and len(a) > max_anchors_per_episode:
                excluded["anchor_limit"] += len(a) - max_anchors_per_episode
                a = np.sort(self.rng.choice(a, max_anchors_per_episode, replace=False))
            for i, seg in enumerate(info["segments"]):
                bounds = [seg["fill_start"], seg["fill_end"]]
                lo, hi = np.searchsorted(original, bounds)
                kept_lo, kept_hi = np.searchsorted(a, bounds)
                counts = {"original_anchors": int(hi - lo), "retained_anchors": int(kept_hi - kept_lo),
                          "excluded_anchors": int(hi - lo - kept_hi + kept_lo)}
                key = self._contract_keys[ep, i] if cond_mode == "contract_proto" else precondition_key(seg["skill"], seg["objects"])
                for group, label in (("by_skill", seg["skill"]), ("by_contract", key)):
                    row = self.anchor_filter[group].setdefault(label, dict.fromkeys(counts, 0))
                    for name, count in counts.items():
                        row[name] += count
            if not len(a):
                self.anchor_filter["empty_episodes"].append(ep)
            self.anchors.extend((ep, int(x)) for x in a)
        self.contract_filter["retained_anchors"] = len(self.anchors)
        excluded["unsupported_contract"] = self.contract_filter["skipped_anchors"]
        self.anchor_filter.update(original_anchors=self.contract_filter["original_anchors"], retained_anchors=len(self.anchors),
                                  excluded_anchors=self.contract_filter["original_anchors"] - len(self.anchors))
        if (sum(excluded.values()) != self.anchor_filter["excluded_anchors"]
                or sum(r["original_anchors"] for r in self.anchor_filter["by_skill"].values()) != self.anchor_filter["original_anchors"]):
            raise ValueError(f"Incomplete/overlapping segment coverage in {split} anchor accounting")
        if self.contract_filter["unsupported_segments"]:
            warnings.warn(f"{split} contract_proto explicitly skipped {self.contract_filter['skipped_anchors']} anchors: "
                          f"{self.contract_filter['unsupported_segments']}", stacklevel=2)
        if not self.anchors:
            raise ValueError(f"No anchors for split={split!r}, scope={scope!r}: {self.anchor_filter}")

    def __len__(self):
        return len(self.anchors)

    def lowdim(self, ep):
        if ep not in self._lowdim:
            self._lowdim[ep] = (np.load(LOWDIM / f"episode_{ep}_action.npy", mmap_mode="r"),
                                np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r"))
        return self._lowdim[ep]

    def pack(self, ep, camera=None):
        key = ep if camera in (None, self.camera) else (ep, camera)
        if key not in self._packs:
            from harness.frame_cache import FramePack
            self._packs[key] = FramePack(FRAMES / (camera or self.camera), ep)
        return self._packs[key]

    def frame(self, ep, idx, camera=None):
        rgb = self.pack(ep, camera).rgb(idx)
        if rgb.shape[0] != self.image_size:
            import cv2
            rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return normalize_image(rgb)

    def condition(self, ep, anchor):
        seg_i = self.skills.segment_index(ep, anchor)
        seg = self.skills.episodes[ep]["segments"][seg_i]
        cls = seg["class"]
        if self.cond_mode == "task_level":
            vec = self.task_level
        else:
            if self.cond_mode == "contract_proto":
                if (ep, seg_i) not in self._contract_vectors:
                    raise ValueError(f"Unsupported contract has no training target: {self._contract_keys[ep, seg_i]}")
                vec = self._contract_vectors[ep, seg_i]
            else:
                vec = self.prototypes[cls]
            if self.split == "train" and self.instance_prob > 0 and self.rng.random() < self.instance_prob:
                j = self._inst_lookup.get((ep, seg_i))
                if j is not None:
                    vec = self.instances[j]
        return torch.from_numpy(np.ascontiguousarray(vec)), cls

    def __getitem__(self, i):
        ep, a = self.anchors[i]
        action, state = self.lowdim(ep)
        item = {"state": torch.from_numpy(np.array(state[a:a + 1], dtype=np.float32)),
                "action_sequence": torch.from_numpy(np.array(action[a:a + self.chunk], dtype=np.float32)),
                "pixel_values": (torch.from_numpy(self.frame(ep, a)) if len(self.views) == 1 else
                                 torch.from_numpy(np.stack([self.frame(ep, a, view) for view in self.views]))),
                "future_pixel_values": torch.from_numpy(self.frame(ep, a + self.future_offset)),
                "episode": ep, "anchor": a}
        if self.cond_mode != "none":
            item["task_cond"], item["skill_class"] = self.condition(ep, a)
        else:
            item["skill_class"] = self.skills.segment(ep, a)["class"]
        return item


def collate(batch):
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        out[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else torch.tensor(vals)
    return out


def model_views(config):
    """Camera keys fed to the model, head first; absent `model.views` means the legacy head-only model."""
    views = config.model.get("views", None)
    views = [HEAD_CAMERA] if views is None else [str(v) for v in views]
    if not views or views[0] != HEAD_CAMERA or len(set(views)) != len(views):
        raise ValueError(f"model.views must be unique and start with {HEAD_CAMERA!r}: {views}")
    return views


_MULTIVIEW_CLASS = None


def multiview_model_class():
    """`MultiViewVLAModel(num_views, **VLAModel kwargs)`: upstream VLAModel + per-view visual tokens.

    Every view's frozen DINO layers go through the SHARED concat fusion and SHARED query adapter
    (`dino_adapters[0]`), then get `type_emb_dino_layers[0] + view_emb[v]`; the V*Q visual tokens are appended
    in view order (head = view 0). `view_emb` is zero-initialised, so the head-only path equals the source
    single-view checkpoint. Everything else is a faithful copy of upstream `VLAModel.forward`; cond_tokens
    (all tokens after the action tokens, now including every view) still feed the future-feature decoder,
    whose cross-attention accepts any number of memory tokens.
    """
    global _MULTIVIEW_CLASS
    if _MULTIVIEW_CLASS is not None:
        return _MULTIVIEW_CLASS
    vla, _ = load_upstream()

    class MultiViewVLAModel(vla.VLAModel):
        def __init__(self, num_views, **kwargs):
            super().__init__(**kwargs)
            if self.fusion_mode != "concat":
                raise ValueError("MultiViewVLAModel requires fusion_mode='concat' (one shared adapter)")
            if num_views < 2:
                raise ValueError("Use the upstream VLAModel for a single view")
            self.num_views = int(num_views)
            self.view_emb = torch.nn.Parameter(torch.zeros(self.num_views, 1, 1, self.hidden_dim))

        def forward(self, t, noisy_actions, qpos_history=None, dino_features_list=None, task_cond=None):
            """dino_features_list: per-layer (B, V, N, D) (multi-view) or (B, N, D) (legacy head-only, no view_emb)."""
            t_emb = self.time_mlp(t)

            # 1. Action / Proprio tokens
            x_action = self.action_proj(noisy_actions) + \
                self.action_pos_emb[:, :noisy_actions.shape[1], :] + \
                self.type_emb_action
            x_proprio_real = self.proprio_proj(qpos_history) + \
                self.proprio_pos_emb[:, :qpos_history.shape[1], :] + \
                self.type_emb_proprio
            x_proprio = torch.zeros_like(x_proprio_real)
            tokens_list = [x_action, x_proprio]
            proprio_start = x_action.shape[1]
            proprio_end = proprio_start + x_proprio.shape[1]

            # 2. Register Tokens
            if self.num_registers > 0:
                tokens_list.append(self.register_tokens.expand(noisy_actions.shape[0], -1, -1))

            # 2b. Task Condition Token
            if self.use_task_cond:
                assert task_cond is not None, "use_task_cond=True but task_cond was not passed to forward"
                tokens_list.append(self.task_cond_proj(task_cond).unsqueeze(1) + self.type_emb_task_cond)

            # 3. Visual tokens: shared fusion + shared adapter per view, plus a per-view type embedding
            if dino_features_list is not None:
                assert len(dino_features_list) == self.num_dino_layers, \
                    f"Expected {self.num_dino_layers} DINO feature layers, got {len(dino_features_list)}"
                if dino_features_list[0].dim() == 3:   # legacy head-only input: exactly upstream concat path
                    fused = self.concat_fusion(dino_features_list)
                    tokens_list.append(self.dino_adapters[0](fused) + self.type_emb_dino_layers[0])
                else:
                    B, V, N = dino_features_list[0].shape[:3]
                    if V > self.num_views or any(f.shape[:3] != (B, V, N) for f in dino_features_list):
                        raise ValueError(f"Expected per-layer (B, V<={self.num_views}, N, D) features")
                    fused = self.concat_fusion([f.reshape(B * V, N, f.shape[-1]) for f in dino_features_list])
                    cond = self.dino_adapters[0](fused)                        # (B*V, Q, hidden)
                    cond = cond.reshape(B, V, cond.shape[1], cond.shape[2]) + self.type_emb_dino_layers[0].unsqueeze(1) \
                        + self.view_emb[:V].unsqueeze(0).squeeze(2)            # (B, V, Q, hidden)
                    tokens_list.append(cond.reshape(B, V * cond.shape[2], cond.shape[3]))

            # 4. Concat all tokens
            x = torch.cat(tokens_list, dim=1)

            # 5. Transformer Blocks
            state_injected = False
            for i, block in enumerate(self.blocks):
                if i == self.state_inject_start and not state_injected:
                    x[:, proprio_start:proprio_end, :] = x_proprio_real
                    state_injected = True
                x = block(x, t_emb)

            # 6. Output Head
            x = self.final_norm(x)
            final_pred = self.output_proj(x[:, :self.action_len, :])
            cond_tokens = x[:, self.action_len:, :]
            return {"final_pred": final_pred, "cond_tokens": cond_tokens}

    _MULTIVIEW_CLASS = MultiViewVLAModel
    return MultiViewVLAModel


def load_action_state(action_model, state_dict):
    """Load a checkpoint into the action model. A single-view source loaded into a multi-view model may lack
    ONLY `view_emb` (kept at its zero init); every other mismatch raises. Returns the missing key list."""
    if getattr(action_model, "num_views", 1) > 1 and "view_emb" not in state_dict:
        result = action_model.load_state_dict(state_dict, strict=False)
        if list(result.missing_keys) != ["view_emb"] or list(result.unexpected_keys):
            raise RuntimeError(f"Single-view -> multi-view load must miss only view_emb: missing={result.missing_keys} "
                               f"unexpected={result.unexpected_keys}")
        with torch.no_grad():
            action_model.view_emb.zero_()
        return ["view_emb"]
    action_model.load_state_dict(state_dict, strict=True)
    return []


def create_action_model(config, **factory_kwargs):
    """Upstream `ModelFactory.create_action_model`; with >1 `model.views` the same config->kwargs mapping builds
    `MultiViewVLAModel` instead (only the constructed class differs; nothing else is re-implemented)."""
    _, runner = load_upstream()
    views = model_views(config)
    if len(views) == 1:
        return runner.ModelFactory.create_action_model(config, **factory_kwargs)
    mv_cls = multiview_model_class()
    upstream_cls = runner.VLAModel
    runner.VLAModel = lambda **kw: mv_cls(num_views=len(views), **kw)
    try:
        action_model = runner.ModelFactory.create_action_model(config, **factory_kwargs)
    finally:
        runner.VLAModel = upstream_cls
    assert isinstance(action_model, mv_cls) and action_model.num_views == len(views)
    return action_model


def build_models(config, device, dino_dtype=torch.bfloat16):
    """Frozen DINOv3 (bf16) + fp32 action model + wrapper with BEHAVIOR train statistics.

    `model.views` with more than one camera builds `MultiViewVLAModel`; otherwise the upstream VLAModel is
    built exactly as before (single-view configs/checkpoints are unaffected)."""
    vla, runner = load_upstream()
    enc, hidden, n_reg, patch = runner.ModelFactory.create_vision_encoder(
        str(ROOT / config.model.vision_encoder.checkpoint_path), dino_dtype, device)
    feat_layers = list(config.model.vision_encoder.feat_layers)
    use_task_cond = bool(config.model.get("use_task_cond", False))
    views = model_views(config)
    action_model = create_action_model(config, dino_hidden_size=hidden, num_dino_layers=len(feat_layers),
                                       task_cond_dim=hidden if use_task_cond else None, patch_size=patch)
    action_model.to(device)  # fp32 master weights; bf16 autocast in the train step
    ff = config.model.get("future_feat", {})
    t = config.training
    train_cfg = {"time_mu": t.time_mu, "time_sigma": t.time_sigma, "use_vel_weight": t.use_vel_weight,
                 "vel_weight_alpha": t.vel_weight_alpha, "vel_weight_sigma": t.vel_weight_sigma,
                 "use_future_feat": bool(ff.get("enabled", False)), "lambda_future_feat": t.get("lambda_future_feat", 0.0)}
    wrapper = BehaviorVLAWrapper(
        runner.VLAWrapper, vision_encoder=enc, action_model=action_model, time_sampler=t.time_sampler,
        feat_layers=feat_layers, include_cls_register=config.model.vision_encoder.include_cls_register,
        num_register_tokens=n_reg, device=device, dtype=torch.float32,
        norm_stats_path=str(ROOT / config.dataset.stats_path), train_config=train_cfg,
        future_feat_target_layer=ff.get("target_layer", -1), norm_stats_key="behavior1k", encoder_dtype=dino_dtype,
        proprio_mask=config.model.get("proprio_mask", None), action_repr=config.model.get("action_repr", None),
        views=views)
    return wrapper, vla


def BehaviorVLAWrapper(base_cls, *, encoder_dtype, proprio_mask=None, action_repr=None, views=None, **kwargs):
    """VLAWrapper whose image casts follow the encoder dtype while actions/states stay fp32.

    `proprio_mask` names a PROPRIO_MASKS entry; masked state dimensions are zeroed AFTER normalization in
    the one method both training (`forward`) and deployment (`sample_actions`, policy server) call, so the
    checkpoint's config fixes the input contract and the 61D layout/architecture stay unchanged.

    `action_repr` names an ACTION_REPRS entry. `to_model_action`/`to_absolute_action` are the single pair of
    transforms used by training (`forward`) and deployment (`sample_actions`, policy server); the model
    always sees normalized model-space actions, callers always see absolute controller-unit actions. The
    normalization statistics file must declare the same representation (`action_repr` key; absent = absolute).

    `views` (camera keys, head first; default head only) is exposed as `wrapper.views`. With more than one
    view, `get_vision_features` / `sample_actions` take (B, V, 3, H, W) pixels in that order; future-feature
    targets stay head-only (B, 3, H, W).
    """
    views = [HEAD_CAMERA] if views is None else list(views)
    keep = proprio_keep_mask(proprio_mask)
    delta_pairs = action_repr_pairs(action_repr)
    stats_path = kwargs.get("norm_stats_path")
    declared = json.loads(Path(stats_path).read_text()).get("action_repr", "absolute") if stats_path else "absolute"
    if declared != (action_repr or "absolute"):
        raise ValueError(f"Normalization stats declare action_repr={declared!r} but the model uses {action_repr or 'absolute'!r}")

    class _Wrapper(base_cls):
        def __init__(self, **kw):
            self.encoder_dtype = encoder_dtype
            self.proprio_mask_name = proprio_mask or "full"
            self.action_repr_name = action_repr or "absolute"
            self.views = list(views)
            super().__init__(**kw)
            self.register_buffer("proprio_keep", torch.from_numpy(keep.astype(np.float32)), persistent=False)

        @staticmethod
        def _state2d(state):
            state = state.reshape(state.shape[0], -1, 61)[:, 0]   # (B, 61): the current proprio frame
            return state

        def to_model_action(self, action_raw, state_raw):
            """Absolute controller-unit chunk (B, T, 23) + raw state (B, [1,] 61) -> model-space chunk."""
            if not delta_pairs:
                return action_raw
            out = action_raw.clone()
            q = self._state2d(state_raw.to(action_raw.device, action_raw.dtype))
            for (alo, ahi), (slo, shi) in delta_pairs:
                out[:, :, alo:ahi] = action_raw[:, :, alo:ahi] - q[:, None, slo:shi]
            return out

        def to_absolute_action(self, action_model, state_raw):
            """Inverse of `to_model_action` (after denormalization)."""
            if not delta_pairs:
                return action_model
            out = action_model.clone()
            q = self._state2d(state_raw.to(action_model.device, action_model.dtype))
            for (alo, ahi), (slo, shi) in delta_pairs:
                out[:, :, alo:ahi] = action_model[:, :, alo:ahi] + q[:, None, slo:shi]
            return out

        def forward(self, batch):
            if delta_pairs:
                batch = dict(batch)
                batch["action_sequence"] = self.to_model_action(batch["action_sequence"].to(self.device, self.dtype),
                                                                batch["state"].to(self.device, self.dtype))
            return super().forward(batch)

        def train(self, mode=True):
            super().train(mode)
            self.vision_encoder.eval()  # Frozen DINOv3 must not randomize RoPE coordinates during training.
            return self

        def normalize_state(self, state):
            norm = super().normalize_state(state)
            if self.proprio_mask_name != "full":
                norm = norm * self.proprio_keep.to(norm.device, norm.dtype)
            return norm

        @torch.no_grad()
        def get_vision_features(self, pixel_values, return_outputs=False):
            """(B,3,H,W) -> per-layer (B,N,D) (legacy); (B,V,3,H,W) -> per-layer (B,V,N,D), one DINO pass over
            B*V images. return_outputs=True also returns the raw DINO output whose rows are b*V + v."""
            multi = pixel_values.dim() == 5
            if multi and pixel_values.shape[1] != len(self.views):
                raise ValueError(f"Expected {len(self.views)} views {self.views}, got {pixel_values.shape[1]}")
            if not multi and len(self.views) > 1:
                raise ValueError(f"Multi-view model expects (B,{len(self.views)},3,H,W) pixels")
            px = pixel_values.to(self.device, self.encoder_dtype)
            if multi:
                B, V = px.shape[:2]
                px = px.reshape(B * V, *px.shape[2:])
            out = self.vision_encoder(pixel_values=px, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            feats = []
            for layer in self.feat_layers:
                h = hs[layer]
                if not self.include_cls_register:
                    h = h[:, 1 + self.num_register_tokens:, :]
                feats.append(h.reshape(B, V, *h.shape[1:]) if multi else h)
            return (feats, out) if return_outputs else feats

        @torch.no_grad()
        def get_future_target_features(self, future_pixel_values):
            px = future_pixel_values.to(self.device, self.encoder_dtype)
            hs = self.vision_encoder(pixel_values=px, output_hidden_states=True, return_dict=True).hidden_states
            return hs[self.future_feat_target_layer][:, 1 + self.num_register_tokens:, :]

        @torch.no_grad()
        def sample_actions(self, pixel_values, state, task_cond=None, num_steps=10, generator=None):
            """Euler flow-matching sampler; returns denormalized (B, 32, 23) actions and cond tokens."""
            feats = self.get_vision_features(pixel_values)
            qpos = self.normalize_state(state.to(self.device, torch.float32))
            if qpos.dim() == 2:
                qpos = qpos.unsqueeze(1)
            B = qpos.shape[0]
            x = torch.randn(B, self.action_model.action_len, self.action_min.numel(), device=self.device, generator=generator)
            cond_tokens = None
            for i in range(num_steps):
                t = torch.full((B,), i / num_steps, device=self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.action_model(t, noisy_actions=x, qpos_history=qpos, dino_features_list=feats,
                                            task_cond=None if task_cond is None else task_cond.to(self.device, torch.float32))
                x = x + out["final_pred"].float() / num_steps
                cond_tokens = out["cond_tokens"]
            return self.to_absolute_action(self.denormalize_action(x), state), cond_tokens, feats

    return _Wrapper(**kwargs)


def group_errors(pred, target):
    """Mean absolute error per action group in native units. pred/target: (B, 32, 23) tensors."""
    err = (pred - target).abs()
    return {name: float(err[..., lo:hi].mean()) for name, (lo, hi) in ACTION_GROUPS.items()}


def hold_baseline(state, chunk_len=32, gripper_limits=(0.0, 0.05)):
    """Copy-state targets, stopped base, and inverse-scaled smooth grippers.

    R1Pro's four finger joints are 0..0.05 m in its official URDF; smooth position control maps
    input -1..1 to that aperture. A zero command means half open, not hold. Keep limits tunable
    for a changed robot/controller; this is a positional no-op, not a grasp-force guarantee.
    """
    low, high = gripper_limits
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("finite increasing gripper limits required")
    B = state.shape[0]
    s = state.reshape(B, -1)
    if s.shape[1] != 61 or not torch.isfinite(s).all():
        raise ValueError("hold requires a finite 61D observation")
    out = torch.zeros(B, chunk_len, 23, dtype=state.dtype, device=state.device)
    out[..., 3:7] = s[:, None, 53:57]     # trunk qpos
    out[..., 7:14] = s[:, None, 3:10]     # left arm qpos
    out[..., 15:22] = s[:, None, 28:35]   # right arm qpos
    out[..., 14] = (2 * (s[:, 24:26].mean(-1, keepdim=True) - low) / (high - low) - 1).clamp(-1, 1)
    out[..., 22] = (2 * (s[:, 49:51].mean(-1, keepdim=True) - low) / (high - low) - 1).clamp(-1, 1)
    return out
