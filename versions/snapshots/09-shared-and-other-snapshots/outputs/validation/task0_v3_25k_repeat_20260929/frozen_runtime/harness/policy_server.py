#!/usr/bin/env python3
"""LiLa-WAM (BEHAVIOR Task 13) policy server for the official BEHAVIOR-1K evaluator.

Speaks the evaluator's websocket protocol (openpi-style msgpack with numpy extension, `/healthz`
HTTP probe, `{"reset": True}` messages) so the simulator process (Isaac Sim 6.1 environment) and the
model process (lilawam environment) never share Python dependencies.

Observation keys seen by the policy (flattened by the evaluator, official r1pro.yaml):
    robot_r1::proprio                                    61D float32
    robot_r1::robot_r1:zed_link:Camera:0::rgb            224x224x(3|4) uint8 head camera
    robot_r1::robot_r1:left_realsense_link:Camera:0::rgb / right_realsense_link ...
Action returned: 23D float32 in controller units (base velocity command, absolute joint targets,
smooth gripper commands), exactly the layout the training data used.

The optional orchestrator supplies skill programs, asking and condition-bank selection. The independent
physical evidence gate is not integrated; this server is a feasibility adapter, not a validated safety layer.
"""
import argparse
import asyncio
import functools
import hashlib
import http
import json
import logging
import re
import sys
import time
from collections import deque
from pathlib import Path

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as ws_server

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import (ACTION_GROUPS, HEAD_CAMERA, VTT_DIR, build_models, hold_baseline,  # noqa: E402
                                  model_proprio_lags, normalize_image, stack_proprio_history)
from harness.vtt_monitor import VTTClosureMonitor, default_configs, precondition_key  # noqa: E402
from harness.orchestrator import ClosureAdapter, ConditionBank, Orchestrator, budgets_from_program, fixed_program_candidates  # noqa: E402
from harness import task_profile  # noqa: E402  P2_TASK_ID (default 13: every Task 13 path unchanged)
from harness.grounding import CandidateProgram, category  # noqa: E402

SKILL_PROGRAM = ROOT / f"configs/task{task_profile.TASK_ID:03d}_skill_program.json"

log = logging.getLogger("policy_server")


# ---- msgpack numpy extension (must match omnigibson.eval.utils.network_utils exactly) -------------
def pack_data(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data, strict_map_key=False)


# ---- model inputs ---------------------------------------------------------------------------------
_VIEW_KEY = re.compile(r"^observation\.rgb\.(?P<link>[A-Za-z0-9_]+)_camera_(?P<index>\d+)$")
# Evaluator-side wrist cameras (robot_r1:<link>:Camera:0 in eval/r1pro.yaml; DefaultWrapper renders every robot
# camera at 224x224 RGBA). Snapshots only: evidence of what the wrists saw, never a policy input by themselves.
WRIST_SNAPSHOT_SUFFIXES = (("lwrist", ":left_realsense_link:Camera:0::rgb"), ("rwrist", ":right_realsense_link:Camera:0::rgb"))


def live_rgb_suffix(view):
    """Dataset camera key -> suffix of the flattened evaluator observation key.

    observation.rgb.left_realsense_link_camera_0 -> ':left_realsense_link:Camera:0::rgb', which the live key
    robot_r1::robot_r1:left_realsense_link:Camera:0::rgb ends with (the same naming holds for zed_link)."""
    m = _VIEW_KEY.match(str(view))
    if m is None:
        raise ValueError(f"Cannot map camera key {view!r} to a live observation key")
    return f":{m['link']}:Camera:{m['index']}::rgb"


def prep_rgb(rgb, size):
    """The single image chain for every camera: RGB(A) -> uint8 -> cv2 INTER_AREA to size -> ImageNet CHW."""
    rgb = np.asarray(rgb)[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        import cv2
        rgb = cv2.resize(np.ascontiguousarray(rgb), (size, size), interpolation=cv2.INTER_AREA)
    return normalize_image(rgb)


class PolicyModel:
    """One loaded action model: checkpoint identity, input contract (views, image size) and wrapper."""

    def __init__(self, name, checkpoint, step, cfg, cond_mode, wrapper):
        self.name, self.checkpoint, self.step = str(name), str(checkpoint), int(step)
        self.cfg, self.cond_mode, self.wrapper = cfg, cond_mode, wrapper
        self.views = [str(v) for v in (getattr(wrapper, "views", None) or [HEAD_CAMERA])]
        declared = cfg.model.get("views", None)
        if declared is not None and [str(v) for v in declared] != self.views:
            raise ValueError(f"{name}: wrapper.views {self.views} != config model.views {list(declared)}")
        if self.views[0] != HEAD_CAMERA or len(set(self.views)) != len(self.views):
            raise ValueError(f"{name}: views must be unique and start with the head camera {HEAD_CAMERA!r}: {self.views}")
        self.view_suffixes = [live_rgb_suffix(v) for v in self.views]
        self.image_size = int(cfg.dataset.image_size[0])
        # Proprio input contract, read once: lags in 30 Hz frames, lag 0 first ([0] = legacy single-frame checkpoint).
        self.is_v3 = "v3" in cfg.model
        self.context = None
        self.memory_lags = []
        self.last_gate = {}
        if self.is_v3:
            from harness.wam_v3 import V3Context, v3_settings
            settings = v3_settings(cfg)
            self.proprio_lags = [0, *settings["proprio_lags"]]
            self.memory_lags = settings["memory_lags"]
            self.context = V3Context(settings["proprio_lags"], self.memory_lags, self.views)
        else:
            self.proprio_lags = model_proprio_lags(cfg)
        hist = getattr(wrapper, "proprio_history", None)
        if hist is not None and [int(v) for v in hist["lags"]] != self.proprio_lags:
            raise ValueError(f"{name}: proprio history stats lags {list(hist['lags'])} != model.proprio_lags {self.proprio_lags}")

    @property
    def multiview(self):
        return len(self.views) > 1

    def describe(self):
        return {"model": self.name, "checkpoint": self.checkpoint, "checkpoint_step": self.step,
                "architecture": "v3" if self.is_v3 else "v2", "memory_lags": self.memory_lags,
                "history_contract_start_token": self.is_v3,
                "cond_mode": self.cond_mode, "views": list(self.views), "proprio_lags": list(self.proprio_lags),
                "vision_encoder": str(self.cfg.model.vision_encoder.checkpoint_path)}


def _build_v3_wrapper(cfg, device):
    """Construct the deployment architecture, never the V2->V3 training migration path."""
    from harness.wam_behavior import load_upstream, model_views
    from harness.wam_v3 import WamV3Wrapper, create_v3_action_model, v3_settings
    settings = v3_settings(cfg)
    views = model_views(cfg)
    stats = json.loads((ROOT / settings["stats_path"]).read_text())
    for key, expected in (("views", views), ("proprio_lags", settings["proprio_lags"]),
                          ("memory_lags", settings["memory_lags"])):
        if list(stats[key]) != list(expected):
            raise ValueError(f"V3 stats {key} do not match checkpoint config")
    if not cfg.model.get("use_task_cond", False) or not cfg.model.vision_encoder.get("include_cls_register", True):
        raise ValueError("V3 requires task conditioning and CLS/register features")
    _, runner = load_upstream()
    enc, hidden, n_reg, patch = runner.ModelFactory.create_vision_encoder(
        str(ROOT / cfg.model.vision_encoder.checkpoint_path), torch.bfloat16, device)
    enc.eval().requires_grad_(False)
    layers = list(cfg.model.vision_encoder.feat_layers)
    for key, shape in (("proprio_diff_std", (len(settings["proprio_lags"]) + 1, 61)),
                       ("memory_diff_std", (len(views), len(settings["memory_lags"]) + 1, 2 * hidden))):
        arr = np.asarray(stats[key])
        if arr.shape != shape or not np.isfinite(arr).all() or (arr <= 0).any():
            raise ValueError(f"V3 stats {key} must be finite positive with shape {shape}")
    action = create_v3_action_model(cfg, dino_hidden_size=hidden, num_dino_layers=len(layers),
                                    task_cond_dim=hidden, patch_size=patch)
    return WamV3Wrapper(action, views=views, stats=stats, norm_stats_path=str(ROOT / cfg.dataset.stats_path),
                        proprio_mask=cfg.model.get("proprio_mask", None), vision_encoder=enc,
                        feat_layers=layers, num_register_tokens=n_reg, device=device).to(device)


class _EncoderCache(dict):
    """Per-server allocation budget. Plain dict callers retain the legacy unrestricted helper behavior."""
    def __init__(self, max_encoders=1):
        super().__init__()
        self.max_encoders = int(max_encoders)
        if self.max_encoders < 1:
            raise ValueError("max_vision_encoders must be positive")


def _build_sharing_encoder(cfg, device, encoder_cache):
    """build_models() with the frozen DINOv3 memoized by (resolved path, dtype, device): a second checkpoint
    that names the same encoder directory reuses the already-loaded module instead of a second copy."""
    from harness.wam_behavior import load_upstream
    _, runner = load_upstream()
    factory = runner.ModelFactory
    original = factory.__dict__["create_vision_encoder"]
    create = original.__func__ if isinstance(original, staticmethod) else original

    def cached(checkpoint_path, dtype=torch.bfloat16, device="cuda"):
        key = (str(Path(checkpoint_path).resolve()), str(dtype), str(device))
        if key not in encoder_cache:
            limit = getattr(encoder_cache, "max_encoders", None)
            if limit is not None and len(encoder_cache) >= limit:
                raise ValueError("vision encoder residency budget exceeded; use the same DINO path/dtype/device "
                                 "or explicitly budget orchestrate.max_vision_encoders")
            encoder_cache[key] = create(checkpoint_path, dtype, device)
        else:
            log.info("reusing loaded vision encoder %s", key[0])
        return encoder_cache[key]

    factory.create_vision_encoder = staticmethod(cached)
    try:
        if "v3" in cfg.model:
            wrapper = _build_v3_wrapper(cfg, device)
        else:
            wrapper, _ = build_models(cfg, device)
    finally:
        factory.create_vision_encoder = original
    return wrapper


def load_policy_model(name, checkpoint, device, encoder_cache):
    """Checkpoint -> PolicyModel (fp32 action weights, strict load, eval mode; encoder shared via encoder_cache)."""
    from omegaconf import OmegaConf
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = OmegaConf.create(ck["config"])
    cond_mode = ck.get("cond_mode", cfg.dataset.get("cond_mode", "skill_proto"))
    if "v3" in cfg.model and "v3_stats" in ck:
        live_stats = json.loads((ROOT / str(cfg.model.v3.stats_path)).read_text())
        if ck["v3_stats"] != live_stats:
            raise ValueError("V3 normalization stats differ from the checkpoint's training snapshot")
    wrapper = _build_sharing_encoder(cfg, device, encoder_cache)
    # Deployment is NOT initialization: no missing V3 heads, resized embeddings or tiled queries allowed.
    wrapper.action_model.load_state_dict(ck["model_state_dict"], strict=True)
    wrapper.eval()
    return PolicyModel(name, checkpoint, int(ck.get("step", -1)), cfg, cond_mode, wrapper)


# ---- policy ---------------------------------------------------------------------------------------
class WamPolicy:
    def __init__(self, checkpoint, *, skill, exec_horizon, num_steps, device, cuda_fraction, seed, log_dir,
                 skill_schedule=None, program=None, budget_factor=1.5, memory_vtt=None, orchestrate=None, samples=1,
                 model_loader=None):
        if memory_vtt and orchestrate is None:
            raise ValueError("Memory requires orchestrator contract keys and an explicit memory_context")
        self.device = torch.device(device)
        if self.device.type == "cuda":
            policy = json.loads((ROOT / "configs/behavior_environment.json").read_text())["resource_policy"]
            if not policy["gpu_execution_enabled"]:
                raise RuntimeError(policy["gpu_execution_block_reason"])
        if not 1 <= int(exec_horizon) <= 32 or int(num_steps) < 1 or int(samples) < 1:
            raise ValueError("exec_horizon must be 1..32; num_steps and samples must be positive")
        self.samples = int(samples)   # flow samples averaged per chunk (reduces sampler noise; K x inference cost)
        # Checkpoint-level decoupling: orchestrate.manip_checkpoint (optional) serves every NON-navigation contract;
        # navigation ("move to") contracts and all non-orchestrated modes keep the main checkpoint.
        manip_checkpoint = (orchestrate or {}).get("manip_checkpoint")
        if manip_checkpoint is not None and (not isinstance(manip_checkpoint, str) or not manip_checkpoint):
            raise ValueError("orchestrate.manip_checkpoint must be a checkpoint path")
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(float(cuda_fraction))
        torch.set_num_threads(2)
        loader = model_loader or load_policy_model
        self._encoder_cache = _EncoderCache((orchestrate or {}).get("max_vision_encoders", 1))
        self.cuda_fraction = float(cuda_fraction)
        # Budget defaults: one action model on-device and one shared encoder. Two resident action models
        # require explicit offload_inactive=false; the process CUDA memory fraction remains the hard cap.
        # offload_inactive: with two checkpoints, keep only the action model in use on the GPU (the shared frozen
        # DINOv3 never moves). Weights move losslessly, so samples are identical; VRAM matches a single-model run.
        self.offload_inactive = bool((orchestrate or {}).get("offload_inactive", True)) and bool(manip_checkpoint)
        self._resident = None
        self.model = loader("main", checkpoint, self.device, self._encoder_cache)
        self.cfg, self.cond_mode, self.wrapper = self.model.cfg, self.model.cond_mode, self.model.wrapper
        if self.cond_mode == "contract_proto" and (orchestrate is None or not orchestrate.get("train_targets")):
            raise ValueError("Contract-conditioned weights require orchestrator train_targets")
        self.checkpoint = str(checkpoint)
        self.checkpoint_step = self.model.step
        self.manip = None
        self.encoder_shared = None
        if manip_checkpoint:
            if self.offload_inactive:
                self._offload(self.model)   # never hold two action models on the GPU, not even while loading
            self.manip = loader("manip", manip_checkpoint, self.device, self._encoder_cache)
            if self.offload_inactive:
                self._offload(self.manip)
                self._ensure_resident(self.model)
            if self.manip.cond_mode != self.cond_mode:
                raise ValueError(f"manip checkpoint cond_mode {self.manip.cond_mode!r} != main {self.cond_mode!r}")
            self.encoder_shared = self.manip.wrapper.vision_encoder is self.wrapper.vision_encoder
            log.info("manip checkpoint %s (step %d), views=%s, proprio_lags=%s, vision encoder shared with main: %s",
                     manip_checkpoint, self.manip.step, self.manip.views, self.manip.proprio_lags, self.encoder_shared)
        if any(m is not None and m.is_v3 for m in (self.model, self.manip)):
            if self.samples != 1 or (self.model.is_v3 and int((orchestrate or {}).get("samples_navigation") or 1) != 1):
                raise ValueError("V3 deployment requires samples=1; exec_horizon is independent")
            log.warning("V3 encodes fresh pooled features every control frame, including queued actions; "
                        "30 Hz is not guaranteed. Inspect v3_context.encode_ms and server_timing.infer_ms.")
        if self.manip is not None and self.manip.image_size != self.model.image_size:
            raise ValueError("main/manip checkpoints must use the same live preprocessing image size")
        self.exec_horizon = int(exec_horizon)
        self.budget_factor = float(budget_factor)
        self.num_steps = int(num_steps)
        self.seed = int(seed)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.seed)
        meta = json.loads((VTT_DIR / "meta.json").read_text())
        self.classes = meta["classes"]
        self.prototypes = torch.from_numpy(np.load(VTT_DIR / "prototypes.npy").astype(np.float32))
        self.task_level = torch.from_numpy(np.load(VTT_DIR / "task_level.npy").astype(np.float32))
        self.skill = self._skill_id(skill)
        self.skill_schedule = [(int(s), self._skill_id(k)) for s, k in (skill_schedule or [])]
        # Program mode: ordered skills advanced by VTT closure (True) or per-skill budget (False); navigation
        # has no closure threshold and advances on budget only until a perception precondition verifier exists.
        self.program = None
        if program is not None:
            prog = json.loads(Path(program).read_text())
            from harness.grounding import Catalog, compile_programs
            candidates, budgets = fixed_program_candidates(program)
            compiled, rejected = compile_programs(candidates, Catalog())
            if rejected or not compiled:
                raise ValueError(f"Invalid program: {rejected}")
            fallback_budgets = budgets_from_program(SKILL_PROGRAM)
            self.program = [(self._skill_id(step.skill), list(step.objects),
                             int(budgets.get(step.key(), fallback_budgets.get(step.key(), 900))))
                            for step in compiled[0].steps]
            self.monitor = VTTClosureMonitor(VTT_DIR)
            self.closure_cfgs = default_configs(program_path=SKILL_PROGRAM, exec_horizon=exec_horizon, budget_factor=budget_factor)
            self.program_index = 0
            self.closure_events = []
        # Orchestrator mode: grounding -> skill contracts -> closure/precondition predicates -> ask -> memory arms.
        self.orch = None
        if orchestrate is not None:
            self.orch_cfg = orchestrate
            self.base_control_mode = str(orchestrate.get("base_control", "wam_whole_body"))
            if self.base_control_mode not in ("wam_whole_body", "hold_during_manipulation"):
                raise ValueError("base_control must be wam_whole_body or hold_during_manipulation")
            # Diagnostic: closed-loop WAM under-commands the base ~4x vs demos (measured |vx| 0.038 vs demo 0.168),
            # collapsing navigation to a slow wander that never reaches the trunk-facing dock. This optional gain
            # scales ONLY the navigation ("move to") base velocity back toward the demonstrated magnitude; the
            # command is re-clipped to the actuator range and the arrival/settle gate is unchanged, so it cannot
            # manufacture a false arrival. It does not touch upper-body actions or any contact skill.
            # Per-contract flow-sample count: navigation commits to one base mode (K averaging collapses the base
            # command and lets the upper body drift before arrival), contact skills keep the K-sample mean.
            # Evidence only: extra lossless head/wrist snapshots every N steps while a manipulation contract runs, so
            # evaluator-side joint logs can label them physically offline. Never a policy input.
            ms = orchestrate.get("manip_snapshot_every")
            self.manip_snapshot_every = None if ms is None else int(ms)
            if self.manip_snapshot_every is not None and self.manip_snapshot_every < 1:
                raise ValueError("manip_snapshot_every must be positive")
            ns = orchestrate.get("samples_navigation")
            self.samples_navigation = None if ns is None else int(ns)
            if self.samples_navigation is not None and self.samples_navigation < 1:
                raise ValueError("samples_navigation must be positive")
            g = orchestrate.get("move_to_base_gain", 1.0)
            g = [g, g, g] if np.isscalar(g) else list(g)
            self.move_to_base_gain = np.asarray(g, np.float32)
            if (self.move_to_base_gain.shape != (3,) or not np.isfinite(self.move_to_base_gain).all()
                    or bool((self.move_to_base_gain <= 0).any())):
                raise ValueError("move_to_base_gain must be a positive scalar or length-3 [vx,vy,vyaw] list")
            # Displacement gain on the contact-skill (open lid) arm push, analogous to move_to_base_gain.
            # The frozen WAM under-reaches: two open-lid runs nudged the car j_trunk to only ~0.07 rad
            # (open threshold 0.1265) then it fell back.  Scaling the commanded arm/trunk displacement from
            # the MEASURED qpos pushes the arm further along its OWN intended trajectory; re-clipping to the
            # actuator limits keeps it legal.  It reads no ground truth and does not change the checkpoint.
            self.open_lid_arm_gain = float(orchestrate.get("open_lid_arm_gain", 1.0))
            if not np.isfinite(self.open_lid_arm_gain) or self.open_lid_arm_gain <= 0:
                raise ValueError("open_lid_arm_gain must be positive")
            self.monitor = VTTClosureMonitor(VTT_DIR)
            self.closure_cfgs = default_configs(program_path=SKILL_PROGRAM, exec_horizon=exec_horizon, budget_factor=budget_factor)
            # Task 1 (progress.md 2026-09-26): `navigation: docking` executes move-to contracts with the perception-
            # driven docking controller (WAM never sees them); `postconditions: task1_physical` completes manipulation
            # contracts on calibrated proprio predicates (harness/task1_predicates.py) instead of appearance candidates.
            # Task 0 (turning_on_radio): `postconditions: task0_physical` = harness/task0_predicates.py (right hand picks
            # and holds the radio, left closed fist presses). Its press evidence is an unobservable-outcome CONTACT
            # CANDIDATE: it advances the program unverified and press is never re-executed blindly (orchestrator
            # candidate_advance_kinds / irreversible_skills); only the evaluator decides success.
            self.navigation_mode = str(orchestrate.get("navigation", "wam"))
            self.postcondition_mode = str(orchestrate.get("postconditions", "patch_candidate"))
            if self.navigation_mode not in ("wam", "docking") or self.postcondition_mode not in ("patch_candidate", "task1_physical",
                                                                                                "task0_physical"):
                raise ValueError("navigation must be wam|docking and postconditions patch_candidate|task1_physical|task0_physical")
            if self.postcondition_mode == "task0_physical" and task_profile.TASK_ID != 0:
                raise ValueError("postconditions task0_physical requires P2_TASK_ID=0")
            self.docking = self.odom = self.omap = self.perception = self.post_task1 = None
            self._active_contract, self._last_action = None, None
            if self.navigation_mode == "docking":
                from harness.docking_controller import DockingController, ObjectMap, Odometry, load_calibration, task_dock_config
                from harness.task1_perception_adapter import TaskPerception
                # dock_target_at: 'start' (pick-segment start pose) or 'grasp' (pose at gripper closure; use with
                # base_control=hold_during_manipulation, progress.md 2026-09-27 01:45). task_dock_config: DockConfig()
                # for every task but 0; Task 0 = URDF base footprint + height-aware shield (coffee-table support).
                self.docking = DockingController(load_calibration(target_at=str(orchestrate.get("dock_target_at", "start"))),
                                                 config=task_dock_config())
                self._bands_odom = None
                if orchestrate.get("manip_executor") == "primitive":
                    self.docking.cfg.require_posture = False     # the IK primitives plan from any posture
                self.odom, self.omap = Odometry(), ObjectMap()
                self.perception = TaskPerception(orchestrate.get("perception", {}), device=self.device,
                                                 encoder=self.wrapper.vision_encoder)
                self.perceive_every = int(orchestrate.get("perceive_every", 5))
            if self.postcondition_mode == "task1_physical":
                from harness.task1_predicates import Task1Postconditions
                self.post_task1 = Task1Postconditions()
            elif self.postcondition_mode == "task0_physical":
                # same start/update interface; the attribute keeps its historical name (physical postconditions)
                from harness.task0_predicates import Task0Postconditions
                self.post_task1 = Task0Postconditions()
            # manip_executor: 'wam' (default) or 'primitive' = harness/grasp_primitive.py for Task 1 pick/place contracts
            # (the WAM's arm drift misses the can even with the demo base; progress.md 2026-09-27 02:10). Targets come
            # from the docking target track (onboard perception + odometry), the held bin from proprio.
            self.manip_executor = str(orchestrate.get("manip_executor", "wam"))
            if self.manip_executor not in ("wam", "primitive") or (self.manip_executor == "primitive" and self.docking is None):
                raise ValueError("manip_executor must be wam|primitive (primitive requires navigation: docking)")
            if self.manip_executor == "primitive" and task_profile.TASK_ID == 0:
                # the IK primitives are Task-1 diagnostics (bin/can geometry); Task 0 manipulation is the WAM's
                raise ValueError("Task 0 manipulation must be executed by the WAM (manip_executor: wam)")
            self._prim, self._prim_cid = None, None
            from harness.grounding import Catalog
            self.catalog = Catalog()
            protos = {name: self.prototypes[cid].numpy() for name, cid in self.classes.items()}
            targets_path = orchestrate.get("train_targets")
            train_targets = json.loads(Path(targets_path).read_text()) if targets_path else None
            self.train_targets_sha256 = hashlib.sha256(Path(targets_path).read_bytes()).hexdigest() if targets_path else None
            for m in (self.model, self.manip):
                if (m is not None and m.cond_mode == "contract_proto"
                        and self.train_targets_sha256 != m.cfg.dataset.get("contract_targets_sha256")):
                    raise ValueError(f"Runtime target vectors differ from the {m.name} checkpoint's training targets")
            self.bank = ConditionBank(protos, frozen=bool(orchestrate.get("frozen")), seed=int(seed), train_targets=train_targets)
            memory_path = memory_vtt or orchestrate.get("bank_in")
            if memory_vtt and orchestrate.get("bank_in"):
                raise ValueError("Choose memory-vtt or bank_in, not both")
            if memory_path:
                from harness.memory_vtt import validate_context
                saved = json.loads(Path(memory_path).read_text())
                if saved.get("format") != "verified-vtt-v1":
                    raise ValueError("Only independently verified, scoped VTT exports may enter the policy")
                validate_context(saved.get("context"), orchestrate.get("memory_context"))
                self.bank.load(saved["arms"])
                log.info("condition bank loaded: %s", {k: len(v) for k, v in self.bank.arms.items()})
            precond = None
            if orchestrate.get("precondition", True) and self.navigation_mode != "docking":
                try:
                    from harness.vtt_monitor import PreconditionMonitor
                    precond = PreconditionMonitor(VTT_DIR)
                    for key, value in orchestrate.get("precondition_tau", {}).items():
                        if key not in precond.index or type(value) not in (int, float) or not -1 <= value <= 1:
                            raise ValueError("Precondition thresholds require known keys and finite cosines")
                        precond.tau[key] = float(value)
                    log.info("navigation precondition monitor loaded")
                except Exception as e:
                    log.warning("no precondition monitor (%s); navigation will time out and HOLD", e)
            budgets = budgets_from_program(SKILL_PROGRAM)
            self.orch = Orchestrator(catalog=self.catalog, bank=self.bank, closure_monitor=ClosureAdapter(self.monitor, self.closure_cfgs),
                                     precondition_monitor=precond, budgets=budgets, ask_budget=int(orchestrate.get("ask_budget", 2)),
                                     max_attempts=int(orchestrate.get("max_attempts", 2)), budget_factor=budget_factor,
                                     responder=self._responder,
                                     navigation_settle_steps=int(orchestrate.get("navigation_settle_steps", 15)),
                                     navigation_linear_speed_max=float(orchestrate.get("navigation_linear_speed_max", 0.05)),
                                     navigation_yaw_speed_max=float(orchestrate.get("navigation_yaw_speed_max", 0.05)),
                                     reapproach_on_retry=bool(orchestrate.get("reapproach_on_retry", False)),
                                     dock_score_min=orchestrate.get("dock_score_min", None),
                                     # Until a calibrated policy-visible postcondition monitor is connected,
                                     # manipulation VTT closure is only a candidate and cannot advance the program.
                                     require_manipulation_postcondition=True,
                                     navigation_completion="docking" if self.navigation_mode == "docking" else "precondition",
                                     manipulation_completion=("postcondition" if self.postcondition_mode in ("task1_physical",
                                                                                                            "task0_physical")
                                                              else "vtt_and_postcondition"),
                                     rollback_on_release_failure=bool(orchestrate.get("rollback_on_release_failure", False)),
                                     replanner=self._skip_failed_object if orchestrate.get("skip_failed_objects") else None,
                                     min_budget_steps=orchestrate.get("min_budget_steps"),
                                     **({"candidate_advance_kinds": ("contact_candidate",), "irreversible_skills": ("press",),
                                         "reapproach_fallback_in_place": True}
                                        if self.postcondition_mode == "task0_physical" else {}))
            self._vlm = None
            self._grounded = False
            # Generic arrival sub-gate (patch-token prototypes per manipulation key); replaces the HSV dock feature.
            self.patch_gate = None
            gate = orchestrate.get("arrival_gate")
            if gate not in (None, "none", "patch"):
                raise ValueError("arrival_gate must be 'patch' or omitted")
            if gate == "patch":
                if orchestrate.get("dock_score_min") is not None:
                    raise ValueError("arrival_gate=patch and dock_score_min (HSV) are mutually exclusive")
                from harness.vtt_monitor import PatchArrivalGate
                self.patch_gate = PatchArrivalGate(VTT_DIR)
                log.info("patch arrival gate loaded: %s", {k: v for k, v in self.patch_gate.thr.items() if v is not None})
            # Generic policy-visible postcondition gate (patch-token prototypes of the demonstrated END frames per
            # manipulation key).  Without it every manipulation contract fails closed at its budget.
            self.post_gate = None
            pgate = orchestrate.get("postcondition_gate")
            if pgate not in (None, "none", "patch"):
                raise ValueError("postcondition_gate must be 'patch' or omitted")
            if pgate == "patch":
                from harness.vtt_monitor import PatchPostconditionGate
                self.post_gate = PatchPostconditionGate(VTT_DIR)
                log.info("patch postcondition gate loaded: %s", {k: v for k, v in self.post_gate.thr.items() if v is not None})
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace = (self.log_dir / f"policy_trace_{time.strftime('%Y%m%d_%H%M%S')}.jsonl").open("a")
        bounds = json.loads((ROOT / "configs/r1pro_action_bounds.json").read_text())
        self.action_low = np.asarray(bounds["action_low"], np.float32)
        self.action_high = np.asarray(bounds["action_high"], np.float32)
        if (self.action_low.shape != (23,) or self.action_high.shape != (23,)
                or not np.isfinite([self.action_low, self.action_high]).all()
                or not (self.action_low < self.action_high).all()):
            raise ValueError("invalid physical action bounds; rerun scene smoke")
        self.queue = []
        self._queue_model = "main"
        # Per-episode raw proprio buffer: one 61-D state per act() call, i.e. per 30 Hz evaluator control step
        # (steps that only pop the queue or HOLD included). A window of max(lag)+1 frames gives exactly
        # stack_proprio_history over the whole episode: while the episode is shorter than the window, the window IS
        # the episode (clamping pads with the episode's first frame); afterwards no lag reaches past the window.
        self._proprio_buf = deque(maxlen=1 + max(max(m.proprio_lags) for m in (self.model, self.manip) if m is not None))
        self._v3_token = None
        self._v3_frame = {}
        self._proprio_frames = 0      # states recorded this episode (current step t = frames - 1)
        self.step_in_episode = 0
        self.episode = 0
        self.chunk_id = 0
        self._last_cls = None
        self._last_pixels = None
        self.stats = {"chunks": 0, "steps": 0, "holds": 0, "infer_ms": []}
        log.info("checkpoint %s (step %d), cond_mode=%s, views=%s, proprio_lags=%s, skill=%s, exec_horizon=%d, num_steps=%d",
                 checkpoint, self.checkpoint_step, self.cond_mode, self.model.views, self.model.proprio_lags, self.skill,
                 self.exec_horizon, self.num_steps)

    # ---- orchestrator mode helpers ----
    def _responder(self, choices):
        """Scripted answerer for simulation experiments (the human is replaced by a fixed rule, logged as such)."""
        policy = str(self.orch_cfg.get("answer", "none"))
        self._emit({"event": "question", "episode": self.episode, "step": self.step_in_episode, "choices": list(choices), "answer_policy": policy})
        if policy == "none":
            return None
        if policy.isdigit():
            return int(policy)
        hits = [i for i, c in enumerate(choices) if policy.lower() in c.lower()]
        return hits[0] if hits else None

    def _ground(self, pixels_rgb_uint8):
        """Instruction -> candidate skill programs. vlm=none uses the fixed training program (one candidate)."""
        cfg = self.orch_cfg
        if cfg.get("vlm", "none") == "none":
            cands, _ = (fixed_program_candidates(ROOT / cfg["program"]) if cfg.get("program") else fixed_program_candidates())
            rejected, prompt = [], None
        else:
            from harness.grounding import QwenVLBackend, ground
            from PIL import Image
            if self._vlm is None:
                dev = "cpu" if cfg["vlm"].endswith("cpu") else "auto"
                if dev != "cpu" and not json.loads((ROOT / "configs/behavior_environment.json").read_text())["resource_policy"]["gpu_execution_enabled"]:
                    raise RuntimeError("GPU grounding disabled by resource policy; use qwen-cpu")
                self._vlm = QwenVLBackend(model_id=cfg.get("vlm_model", "Qwen/Qwen2.5-VL-3B-Instruct"), quant=None, device_map=dev)
            image = Image.fromarray(pixels_rgb_uint8) if cfg.get("vlm_image", True) else None
            t0 = time.monotonic()
            cands, rejected, prompt = ground(self._vlm, cfg["instruction"], cfg["visible_objects"], self.catalog, image=image)
            log.info("grounding took %.1f s: %d candidates, rejected %s", time.monotonic() - t0, len(cands), rejected)
        self.orch.reset(cands, visible_objects=cfg.get("visible_objects"))
        self._emit({"event": "ground", "episode": self.episode, "instruction": cfg.get("instruction"), "vlm": cfg.get("vlm", "none"),
                    "candidates": [{"interpretation": c.interpretation, "program": [s.key() for s in c.steps]} for c in self.orch.candidates],
                    "rejected": rejected + self.orch.program_rejections,
                    "train_targets_sha256": self.train_targets_sha256})
        self._grounded = True

    def _skip_failed_object(self, failed_key, candidates, failed_step=None):
        """Replanner for independent sub-goals (Task 1 cans): after bounded retries of one object failed, drop that
        object's remaining steps and continue with the others. The trash can itself is never skipped."""
        if failed_step is None or not candidates:
            return None
        obj = failed_step.objects[0]
        if category(obj) == "trash_can":
            return None
        done = list(self.orch.completed_steps)
        out = []
        for c in candidates:
            rest = [st for st in c.steps[len(done):] if obj not in st.objects]
            if any(st.skill != "move to" for st in rest):
                out.append(CandidateProgram(f"{c.interpretation} [skipped {obj}]", done + rest,
                                            list(c.preconditions), list(c.postconditions)))
        if self.docking is not None and self.docking.track_id is not None:
            for t in self.omap.tracks:
                if t.track_id == self.docking.track_id:
                    t.consumed = True     # abandoned: never the nearest target again
        self._emit({"event": "skip_failed_object", "episode": self.episode, "step": self.step_in_episode, "object": obj,
                    "failed_key": failed_key, "remaining": [st.key() for st in (out[0].steps[len(done):] if out else [])]})
        return out or None

    def _primitive_action(self, state):
        """Task 1 perception+IK primitive for the live manipulation contract (built once per contract attempt)."""
        from harness.grasp_primitive import Primitive
        c = self.orch.contract
        cid = self._contract_id(c)
        if self._prim_cid != cid:
            self._prim_cid, self._prim = cid, None
            obj = category(c.step.objects[0])
            kind = {("pick up from", "can_of_soda"): "pick_can", ("pick up from", "trash_can"): "pick_bin",
                    ("place in", "can_of_soda"): "place_in"}.get((c.step.skill, obj))
            if kind is None:
                self._emit({"event": "primitive_unsupported", "episode": self.episode, "step": self.step_in_episode - 1, "key": c.key})
                return None
            target, align_to = None, None
            if kind != "place_in":
                track = next((t for t in self.omap.tracks if t.track_id == self.docking.track_id), None)
                if track is None or track.label != obj:
                    self._emit({"event": "primitive_no_target", "episode": self.episode, "step": self.step_in_episode - 1,
                                "key": c.key, "dock_track": self.docking.track_id})
                    self._prim = "no_target"
                else:
                    target = np.r_[self.odom.to_base(track.pos), 0.06]
                    if obj == "trash_can":      # tracks hold the detector's rim grasp point; the primitive wants the centre
                        from harness.object_perception import RIM_OFFSET
                        target[:2] = target[:2] + RIM_OFFSET
                    if self.docking.calib.get("target_at") == "grasp":
                        align_to = self.docking.calib["dock"][obj]["target"]
            if self._prim is None:
                self._prim = Primitive(kind, state, target, carry_posture=self.docking.calib["posture"]["nav_carrying_bin"],
                                       align_to=align_to)
                self._emit({"event": "primitive_start", "episode": self.episode, "step": self.step_in_episode - 1, "key": c.key,
                            "kind": kind, "attempt": c.attempts, "plan": self._prim.plan_info, "feasible": self._prim.feasible})
        if self._prim == "no_target":
            held = hold_baseline(torch.from_numpy(state)[None])[0, 0].numpy()
            return held
        phase = self._prim.phase
        a = self._prim.step(state)
        if phase != getattr(self, "_prim_phase", None) or self.step_in_episode % 30 == 0:
            self._prim_phase = phase
            self._emit({"event": "primitive", "episode": self.episode, "step": self.step_in_episode - 1, "key": c.key,
                        "phase": phase, "feasible": self._prim.feasible, "plan": self._prim.plan_info,
                        "grip_cmd": [float(a[14]), float(a[22])],
                        "finger_sum": [round(float(state[24:26].sum()), 4), round(float(state[49:51].sum()), 4)],
                        "eef_left": [round(float(v), 3) for v in state[17:20]], "eef_right": [round(float(v), 3) for v in state[42:45]],
                        # trunk(4)+left(7)+right(7): measured vs commanded (tracking / blocked-joint diagnosis)
                        "q_meas": [round(float(v), 3) for v in np.r_[state[53:57], state[3:10], state[28:35]]],
                        "q_cmd": [round(float(v), 3) for v in np.r_[a[3:7], a[7:14], a[15:22]]]})
        return a

    @staticmethod
    def _contract_id(c):
        return None if c is None else (c.index, c.attempts, c.started_step)

    def _carrying_bin(self):
        return any(s.skill == "pick up from" and category(s.objects[0]) == "trash_can" for s in self.orch.completed_steps)

    def _orch_finish_episode(self):
        if self.orch is None or not self._grounded:
            return
        if self._last_cls is not None:
            self.orch.flush_open_contract(self._last_cls, self.step_in_episode)
        with (self.log_dir / "realized_vtt.jsonl").open("a") as f:
            for r in self.orch.realized:
                f.write(json.dumps({"episode": self.episode, **r}) + "\n")
        for e in self.orch.events:
            self._emit({"event": "orch_" + e.pop("event"), "episode": self.episode, **e})
        self.orch.events.clear()
        (self.log_dir / "condition_bank.json").write_text(json.dumps({"frozen": self.bank.frozen, "arms": self.bank.export()}) + "\n")
        self._grounded = False

    def _skill_id(self, skill):
        if skill is None:
            return None
        if isinstance(skill, int) or str(skill).isdigit():
            return int(skill)
        return int(self.classes[str(skill)])

    def metadata(self):
        return {"policy": f"lilawam_behavior_task{task_profile.TASK_ID}", "checkpoint": self.checkpoint, "checkpoint_step": self.checkpoint_step,
                "cond_mode": self.cond_mode, "skill_id": self.skill, "exec_horizon": self.exec_horizon,
                "num_inference_steps": self.num_steps, "samples_averaged": self.samples,
                "seed": self.seed, "rng_reset_behavior": "seed once per process; RNG continues across episode resets",
                "action_repr": getattr(self.wrapper, "action_repr_name", "absolute"), "proprio_mask": self.wrapper.proprio_mask_name, "action_dim": 23, "state_dim": 61,
                "program": None if self.program is None else [(self.monitor.names[c], o, b) for c, o, b in self.program],
                "memory_contracts": sorted(self.bank.arms) if self.orch is not None else [],
                "orchestrator": None if self.orch is None else {k: v for k, v in self.orch_cfg.items() if k != "visible_objects"},
                "models": [m.describe() for m in (self.model, self.manip) if m is not None],
                "v3_history_rule": "V3Context per control frame; reset on contract/retry/HOLD/resume/model change",
                "v3_vision_cadence": "fresh frozen DINO every control frame, including queued actions; not amortized by exec_horizon",
                "views": list(self.model.views),
                "proprio_lags": list(self.model.proprio_lags),
                "manip_checkpoint": None if self.manip is None else self.manip.checkpoint,
                "manip_checkpoint_step": None if self.manip is None else self.manip.step,
                "manip_views": None if self.manip is None else list(self.manip.views),
                "manip_proprio_lags": None if self.manip is None else list(self.manip.proprio_lags),
                "offload_inactive": bool(getattr(self, "offload_inactive", False)),
                "residency_budget": {"action_models": 1 if self.offload_inactive or self.manip is None else 2,
                                     "max_vision_encoders": self._encoder_cache.max_encoders,
                                     "loaded_vision_encoders": len(self._encoder_cache),
                                     "cuda_memory_fraction": self.cuda_fraction},
                "proprio_history_rule": ("V3Context: current + lags clamped to contract start + immutable start; one push per control frame"
                                          if self.model.is_v3 else "stack_proprio_history over one raw state per act() call, clamped to the episode's first frame"),
                "vision_encoder_shared": self.encoder_shared,
                "model_routing": ("single checkpoint" if self.manip is None else
                                  "navigation contracts -> main checkpoint; all other contracts -> manip checkpoint")}

    def reset(self):
        self._emit({"event": "reset", "episode": self.episode, "steps": self.step_in_episode})
        self.finish_episode()
        self.queue = []
        self._queue_model = "main"
        self._reset_v3_contexts("episode_reset")
        self._proprio_buf.clear()     # the next episode's history pads with its OWN first frame
        self._proprio_frames = 0
        if getattr(self, "docking", None) is not None:
            self.docking.reset()
            self.odom.reset()
            self.omap.reset()
            self._obst_odom = None
            self._bands_odom = None
        self._active_contract, self._last_action = None, None
        if hasattr(getattr(self, "post_task1", None), "reset_episode"):
            self.post_task1.reset_episode()     # Task 0: the verified grasp height belongs to one episode
        self.step_in_episode = 0
        self.episode += 1
        self._prev_ub = None
        self._last_post = None
        if self.program is not None:
            self.program_index = 0

    def finish_episode(self):
        """Persist a final incomplete contract as unknown, including on evaluator disconnect."""
        self._reset_v3_contexts("finish_episode")
        self.queue = []
        self._orch_finish_episode()
        if self.program is not None:
            if self.monitor.state is not None and self.monitor.state.steps > 0 and self._last_cls is not None:
                self._record_realized(self._last_cls, None)
            self.monitor.state = None

    def _emit(self, rec):
        rec["time"] = time.time()
        self.trace.write(json.dumps(rec) + "\n")
        self.trace.flush()

    def _current_skill(self):
        if self.program is not None:
            return self.program[min(self.program_index, len(self.program) - 1)][0]
        skill = self.skill
        for start, sid in self.skill_schedule:
            if self.step_in_episode >= start:
                skill = sid
        return skill

    def _advance_program(self, cls_now):
        """VTT closure bookkeeping; returns True when the active skill changed."""
        if self.program is None:
            return False
        skill, objects, budget = self.program[min(self.program_index, len(self.program) - 1)]
        if self.monitor.state is None or self.monitor.state.skill != skill:
            self.monitor.start(skill, cls_now)
        cfg = self.closure_cfgs[skill]
        cfg.max_steps = int(budget * self.budget_factor)
        verdict, progress, alignment = self.monitor.update(cls_now, cfg)
        if verdict is None or self.program_index >= len(self.program) - 1:
            return False
        self.closure_events.append({"episode": self.episode, "step": self.step_in_episode, "index": self.program_index,
                                    "skill": self.monitor.names[skill], "objects": objects, "verdict": verdict,
                                    "progress": round(progress, 3), "alignment": round(alignment, 3),
                                    "skill_steps": self.monitor.state.steps})
        self._emit({"event": "closure", **self.closure_events[-1]})
        self._record_realized(cls_now, verdict)
        self.program_index += 1
        self.monitor.start(self.program[self.program_index][0], cls_now)
        return True

    def _record_realized(self, cls_now, verdict):
        """Store the realized transition CLS(now) - CLS(skill start) for the skill that just ended, so a
        memory writer can promote it (same recipe as data/skill_vtt_task013 instance VTTs)."""
        st = self.monitor.state
        vtt = np.asarray(cls_now, dtype=np.float32) - st.start_cls.astype(np.float32)
        skill, objects, _ = self.program[min(self.program_index, len(self.program) - 1)]
        rec = {"episode": self.episode, "index": self.program_index, "skill": self.monitor.names[skill], "objects": objects,
               "closure": verdict, "skill_steps": st.steps, "end_step": self.step_in_episode, "vtt": vtt.tolist()}
        with (self.log_dir / "realized_vtt.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    @staticmethod
    def _policy_observation(obs):
        """Sensor allowlist: evaluator object poses, rewards, phase/hand labels and BDDL never enter policy."""
        allowed = {}
        for key, value in obs.items():
            if not isinstance(key, str):
                continue
            if re.fullmatch(r"robot_r1::(?:proprio|cam_rel_poses)", key) or re.fullmatch(
                    r"robot_r1::robot_r1:[A-Za-z0-9_]+:Camera:\d+::(?:rgb|depth|depth_linear)", key):
                allowed[key] = value
        return allowed

    @staticmethod
    def _find(obs, suffix, contains=None):
        for k in obs:
            if k.endswith(suffix) and (contains is None or contains in k):
                return obs[k]
        raise KeyError(f"no observation key ending with {suffix!r} containing {contains!r}; keys={sorted(obs)}")

    def _preprocess(self, obs):
        state = np.array(self._find(obs, "::proprio"), dtype=np.float32, copy=True).reshape(-1)
        assert state.shape == (61,), state.shape
        # Head camera only; must stay usable unbound with a (cfg, _find) namespace (scripts/check_openlid_replay.py).
        return state, prep_rgb(self._find(obs, "::rgb", "zed_link"), int(self.cfg.dataset.image_size[0]))

    def _view_pixels(self, obs, model, head_pixels):
        """(V, 3, S, S) in model.views order. View 0 IS the head tensor from `_preprocess` (byte-identical);
        every other view goes through the same prep_rgb chain from its live observation key."""
        if head_pixels.shape[-1] != model.image_size:
            raise ValueError(f"{model.name}: image_size {model.image_size} differs from the head preprocessing")
        out = [head_pixels]
        for view, suffix in zip(model.views[1:], model.view_suffixes[1:]):
            out.append(prep_rgb(self._find(obs, suffix), model.image_size))
        return np.stack(out)

    @torch.no_grad()
    def _model_feats(self, model, obs, head_pixels, head_feats):
        """Action-model features for `model`. Single-view models on the shared encoder reuse the head pass;
        multi-view models get ONE extra batched DINO pass over (1, V, 3, S, S) at chunk time only. The head-only
        consumers (CLS for VTT/precondition, patch tokens for arrival/postcondition gates) never see it."""
        w = model.wrapper
        if getattr(model, "is_v3", False):
            if model.name not in self._v3_frame:
                raise RuntimeError("V3 features not recorded for this control frame")
            return self._v3_frame[model.name][0]
        if model.multiview:
            px = torch.from_numpy(self._view_pixels(obs, model, head_pixels))[None]
            return w.get_vision_features(px)
        main = self.wrapper
        if w is main or (w.vision_encoder is main.vision_encoder and list(w.feat_layers) == list(main.feat_layers)
                         and w.include_cls_register == getattr(main, "include_cls_register", True)
                         and w.num_register_tokens == main.num_register_tokens):
            return head_feats
        return w.get_vision_features(torch.from_numpy(head_pixels)[None])

    def _reset_v3_contexts(self, reason):
        for m in (getattr(self, "model", None), getattr(self, "manip", None)):
            if m is not None and getattr(m, "is_v3", False):
                m.context.reset()
                m.last_gate = {}
        self._v3_token = None
        self._v3_frame = {}
        self._v3_reset_reason = reason

    @torch.no_grad()
    def _record_v3_frame(self, model, obs, state, head_pixels, token):
        """Called once AFTER contract transitions, BEFORE queue/HOLD handling. No cross-frame feature cache.

        Only the routed model has a live contract; inactive contexts are cleared on every routing boundary.
        The additional full-view DINO pass is deliberate: correctness first, latency reported for EACH frame.
        """
        has_v3 = any(getattr(m, "is_v3", False) for m in (getattr(self, "model", None), getattr(self, "manip", None)))
        if not has_v3:
            return
        token = (getattr(model, "name", None), token)
        if token != self._v3_token:
            self._reset_v3_contexts("contract_or_mode_change")
            self._v3_token = token
            self.queue = []
        if not getattr(model, "is_v3", False):
            return
        t0 = time.monotonic()
        pixels = torch.from_numpy(self._view_pixels(obs, model, head_pixels))[None]
        try:
            feats, pooled = model.wrapper.encode(pixels)
        except Exception:
            self._reset_v3_contexts("encoder_error")
            self.queue = []
            raise
        # encode() uses feat_layers[-1] (-4), NOT the last_hidden_state used by the VTT monitor.
        # CPU copy synchronizes CUDA too, so this timer includes actual completion of frozen DINO.
        model.context.push(state, pooled[0].float().cpu().numpy())
        self._v3_frame[model.name] = (feats, model.context.inputs())
        elapsed = (time.monotonic() - t0) * 1000
        total_vision_ms = elapsed + getattr(self, "_head_encode_ms", 0.)
        self._emit({"event": "v3_context", "episode": self.episode, "step": self.step_in_episode,
                    "model": model.name, "context_steps": model.context.steps, "context_maxlen": model.context.maxlen,
                    "memory_layer": int(model.wrapper.feat_layers[-1]), "views": model.views,
                    "reset_reason": getattr(self, "_v3_reset_reason", None) if model.context.steps == 1 else None,
                    "encode_ms": round(elapsed, 3), "total_vision_ms": round(total_vision_ms, 3),
                    "control_budget_ms": round(1000 / 30, 3),
                    "over_30hz_budget": total_vision_ms > 1000 / 30, "fresh_pooled": True})

    def _v3_chunk_log(self, model):
        if not getattr(model, "is_v3", False):
            return {}
        return {"architecture": "v3", "v3_context_steps": model.context.steps,
                "v3_context_maxlen": model.context.maxlen, "memory_lags": model.memory_lags,
                "memory_layer": int(model.wrapper.feat_layers[-1]), "exec_horizon": self.exec_horizon,
                **model.last_gate}

    def _offload(self, model):
        model.wrapper.action_model.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _ensure_resident(self, model):
        """offload_inactive: move `model`'s action model onto the device (the other one to CPU) before sampling."""
        if not getattr(self, "offload_inactive", False) or model is None or getattr(self, "_resident", None) is model:
            return
        t0 = time.monotonic()
        for other in (getattr(self, "model", None), getattr(self, "manip", None)):
            if other is not None and other is not model:
                self._offload(other)
        model.wrapper.action_model.to(self.device)
        previous, self._resident = getattr(self, "_resident", None), model
        if previous is not None and hasattr(self, "_emit"):
            self._emit({"event": "model_residency", "episode": getattr(self, "episode", None),
                        "step": getattr(self, "step_in_episode", None), "from_model": previous.name,
                        "to_model": model.name, "ms": round(1000 * (time.monotonic() - t0), 1)})

    def _route_model(self):
        """Model for the current orchestrator contract: manip for every non-navigation contract when loaded."""
        manip = getattr(self, "manip", None)
        contract = getattr(getattr(self, "orch", None), "contract", None)
        if manip is None or contract is None or contract.is_navigation:
            return getattr(self, "model", None)
        return manip

    def _record_proprio(self, state):
        """Append this control step's raw 61-D state: exactly once per act(), right after preprocessing, so a step
        that only pops the queue, HOLDs, or is answered by the server's hold-on-error fallback is still recorded."""
        self._proprio_buf.append(np.array(state, dtype=np.float32, copy=True))
        self._proprio_frames += 1

    def _proprio_input(self, model, state):
        """Proprio input for `model` (None = main) at the current step. lags [0]: the (61,) current state itself (the
        legacy call, same array). Otherwise the training rule verbatim, stack_proprio_history(episode states, t, lags)
        -> (len(lags), 61), lag 0 first, computed on the per-step buffer whose last frame must be this state."""
        m = getattr(self, "model", None) if model is None else model
        if getattr(m, "is_v3", False):
            if m.name not in self._v3_frame:
                raise RuntimeError("V3 requires a fresh control-frame context before sampling")
            return self._v3_frame[m.name][1][0]
        lags = list(getattr(m, "proprio_lags", [0]))
        if lags == [0]:
            return state
        buf = self._proprio_buf
        if not buf or buf.maxlen < max(lags) + 1 or not np.array_equal(buf[-1], state, equal_nan=True):
            raise RuntimeError(f"{m.name}: proprio history buffer is not in sync with the current control step")
        return stack_proprio_history(np.stack(buf), len(buf) - 1, lags)

    def _proprio_log(self, model):
        """Chunk-event fields: the model's lags, the raw states recorded this episode (current step = frames - 1) and
        how many lags reached before the episode start (clamped to its first frame, as in training)."""
        m = getattr(self, "model", None) if model is None else model
        lags = list(getattr(m, "proprio_lags", [0]))
        frames = m.context.steps if getattr(m, "is_v3", False) else getattr(self, "_proprio_frames", None)
        return {"proprio_lags": lags, "proprio_frames": frames,
                "proprio_padded_lags": None if frames is None else sum(l > frames - 1 for l in lags)}

    @torch.no_grad()
    def _encode(self, pixels):
        """One DINOv3 pass: multi-layer features for the action model + last-layer CLS for VTT closure."""
        w = self.wrapper
        px = torch.from_numpy(pixels)[None].to(w.device, getattr(w, "encoder_dtype", torch.bfloat16))
        out = w.vision_encoder(pixel_values=px, output_hidden_states=True, return_dict=True)
        feats = []
        for layer in w.feat_layers:
            h = out.hidden_states[layer]
            if not getattr(w, "include_cls_register", True):
                h = h[:, 1 + w.num_register_tokens:, :]
            feats.append(h)
        cls = out.last_hidden_state[0, 0].float().cpu().numpy()  # same recipe as precompute_skill_vtt.py
        # Last-layer patch tokens (registers dropped) for the generic PatchArrivalGate; kept on device, no extra pass.
        self._last_patch = out.last_hidden_state[0, 1 + w.num_register_tokens:]
        return feats, cls

    @torch.no_grad()
    def _sample(self, feats, state, cond, samples=None, model=None):
        k = self.samples if samples is None else int(samples)
        m = getattr(self, "model", None) if model is None else model
        if getattr(m, "is_v3", False) and k != 1:
            raise ValueError("V3 deployment requires samples=1 (independent of exec_horizon)")
        if k > 1:
            return np.mean([self._sample_once(feats, state, cond, model=model) for _ in range(k)], axis=0)
        return self._sample_once(feats, state, cond, model=model)

    @torch.no_grad()
    def _sample_once(self, feats, state, cond, model=None):
        w = self.wrapper if model is None else model.wrapper
        # state: (61,) current frame, or (L, 61) proprio history (lag 0 first) for checkpoints with model.proprio_lags
        m = getattr(self, "model", None) if model is None else model
        if getattr(m, "is_v3", False):
            self._ensure_resident(m)
            current_feats, (history, pool) = self._v3_frame[m.name]
            if not np.array_equal(history, state):
                raise RuntimeError("V3 state does not match this control frame")
            if cond is None:
                raise ValueError("V3 requires a task/contract condition")
            actions, aux = w.sample_actions(current_feats, torch.from_numpy(history)[None],
                                             torch.from_numpy(pool)[None], cond,
                                             num_steps=self.num_steps, generator=self.gen)
            gate = aux["gate"][0].float().cpu().numpy()
            phase = torch.softmax(aux["phase_logits"][0].float(), -1).cpu().tolist()
            if gate.shape != (len(m.views),) or not np.isfinite(gate).all():
                raise ValueError("invalid V3 view gate")
            from harness.wam_v3 import PHASES
            m.last_gate = {"view_gate": dict(zip(m.views, gate.tolist())),
                           "phase_probabilities": dict(zip(PHASES, phase)),
                           "predicted_log_dist": aux["log_dist"][0].float().cpu().reshape(-1).tolist()}
            return actions[0].float().cpu().numpy()
        lags = list(getattr(m, "proprio_lags", [0]))
        want = (61,) if lags == [0] else (len(lags), 61)
        if np.shape(state) != want:  # never feed a single frame to a history checkpoint (or vice versa) silently
            raise ValueError(f"{getattr(m, 'name', 'model')}: proprio input {np.shape(state)} != {want} for lags {lags}")
        qpos = w.normalize_state(torch.from_numpy(state)[None].to(w.device, torch.float32))
        if qpos.dim() == 2:
            qpos = qpos.unsqueeze(1)
        x = torch.randn(1, w.action_model.action_len, w.action_min.numel(), device=w.device, generator=self.gen)
        tc = None if cond is None else cond.to(w.device, torch.float32)
        cuda = w.device.type == "cuda"
        if not cuda:  # CPU self-test path: no autocast, so features must match the fp32 weights
            feats = [f.float() for f in feats]
        for i in range(self.num_steps):
            t = torch.full((1,), i / self.num_steps, device=w.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                out = w.action_model(t, noisy_actions=x, qpos_history=qpos, dino_features_list=feats, task_cond=tc)
            x = x + out["final_pred"].float() / self.num_steps
        # denormalize in model space, then back to absolute controller units (identity for absolute checkpoints)
        absolute = w.to_absolute_action(w.denormalize_action(x), torch.from_numpy(state)[None].to(w.device, torch.float32))
        return absolute[0].float().cpu().numpy()

    @torch.no_grad()
    def act(self, obs):
        try:
            return self._act(obs)
        except Exception:
            # Fail closed even when the caller elects NOT to use hold-on-error or reconnects later.
            self.queue = []
            self._reset_v3_contexts("act_error")
            raise

    @torch.no_grad()
    def _act(self, obs):
        t0 = time.monotonic()
        obs = self._policy_observation(obs)
        self._v3_frame = {}  # feature reuse is allowed within this call ONLY
        state, pixels = self._preprocess(obs)
        self._record_proprio(state)         # one raw state per control step, before anything else can fail
        head_started = time.monotonic()
        feats, cls = self._encode(pixels)   # head-only pass: CLS/patch tokens for every head-feature consumer
        self._head_encode_ms = (time.monotonic() - head_started) * 1000
        self._last_cls = cls
        self._last_pixels = pixels
        if self.orch is not None:
            return self._act_orchestrated(obs, state, feats, cls, t0)
        switched = self._advance_program(cls)
        self._record_v3_frame(self.model, obs, state, pixels,
                              (self.episode, "executor", self.program_index if self.program is not None else None,
                               self._current_skill()))
        if switched:
            self.queue = []  # a new skill condition invalidates the old chunk
        if self.program is not None and self.step_in_episode % 30 == 0:
            p, a = self.monitor.signals(cls)
            self._emit({"event": "progress", "episode": self.episode, "step": self.step_in_episode, "index": self.program_index,
                        "skill": self.monitor.names[self._current_skill()], "progress": round(p, 3), "alignment": round(a, 3)})
        if not self.queue:
            skill = self._current_skill()
            cond = None
            if self.cond_mode == "skill_proto":
                cond = self.prototypes[skill][None]
            elif self.cond_mode == "task_level":
                cond = self.task_level[None]
            state_in = self._proprio_input(self.model, state)   # (61,) legacy, (L, 61) for a history checkpoint
            if self.model.multiview or self.model.is_v3:
                chunk = self._bounded_chunk(self._sample(self._model_feats(self.model, obs, pixels, feats), state_in, cond))
            else:
                chunk = self._bounded_chunk(self._sample(feats, state_in, cond))
            self.queue = [chunk[i] for i in range(self.exec_horizon)]
            self.chunk_id += 1
            self.stats["chunks"] += 1
            infer_ms = (time.monotonic() - t0) * 1000
            self.stats["infer_ms"].append(infer_ms)
            self._emit({"event": "chunk", "episode": self.episode, "step": self.step_in_episode, "chunk": self.chunk_id,
                        "model": "main", "model_checkpoint": self.checkpoint, "model_views": len(self.model.views),
                        **self._proprio_log(self.model), **self._v3_chunk_log(self.model), "samples": self.samples,
                        "skill": skill, "infer_ms": round(infer_ms, 1),
                        "base_cmd": [round(float(x), 4) for x in chunk[0, :3]],
                        "grip": [round(float(chunk[0, 14]), 3), round(float(chunk[0, 22]), 3)]})
        action = self.queue.pop(0)
        self.step_in_episode += 1
        self.stats["steps"] += 1
        return torch.from_numpy(np.asarray(action, dtype=np.float32))

    @torch.no_grad()
    def _act_orchestrated(self, obs, state, feats, cls, t0):
        head_rgb = np.asarray(self._find(obs, "::rgb", "zed_link"))[..., :3].astype(np.uint8)
        if not self._grounded:
            self._ground(head_rgb)
        # Arrival sub-gate signal (no ground truth). Generic: DINOv3 patch-token prototype of the NEXT manipulation
        # key (arrival_gate="patch", any task); legacy: the instance-specific HSV car feature (dock_score_min).
        dock, dock_thr = None, None
        contract = getattr(self.orch, "contract", None)
        if getattr(self, "patch_gate", None) is not None:
            if contract is not None and contract.is_navigation and contract.precondition_key:
                dock_thr = self.patch_gate.threshold(contract.precondition_key)
                if dock_thr is not None:
                    dock = self.patch_gate.score(self._last_patch, contract.precondition_key)
                    self._last_dock = {"gate": "patch", "key": contract.precondition_key, "score": dock, "threshold": dock_thr,
                                       "centroid_u": float("nan"), "lr_sym": float("nan")}
        elif getattr(self.orch, "dock_score_min", None) is not None:
            from harness.docking import car_dock_feature
            self._last_dock = car_dock_feature(head_rgb)
            self._last_dock["gate"] = "hsv"
            dock = self._last_dock.get("score")
        # Proprioception stores measured local-base velocity in the first three entries.  A visual
        # precondition is not an arrival until the platform has also settled; no simulator pose is read.
        # End-frame appearance is a candidate, NOT independent outcome evidence. Its calibration labels are
        # demonstration segment fractions, not physical predicates. Log it, but never let it self-certify success
        # or promote memory; the orchestrator waits for a separately validated observable postcondition.
        post = None
        if getattr(self, "post_gate", None) is not None and contract is not None and not contract.is_navigation:
            pkey = precondition_key(contract.step.skill, contract.step.objects)
            pthr = self.post_gate.threshold(pkey)
            if pthr is not None:
                pscore = self.post_gate.score(self._last_patch, pkey)
                post = {"verified": False, "verifier_kind": "appearance_candidate", "source": "patch_postcondition_gate:v1",
                        "observation_id": int(self.step_in_episode), "satisfied": bool(np.isfinite(pscore) and pscore >= pthr),
                        "score": float(pscore), "threshold": float(pthr), "key": pkey}
        self._last_post = post
        # Policy-visible upper-body motion (L1 joint change since the previous control step) for the generic
        # motion-stall guard: a VTT plateau only counts as a stall when the robot is also idle.
        ub = np.concatenate([state[53:57], state[3:10], state[28:35]]).astype(np.float64)
        motion = float(np.abs(ub - self._prev_ub).sum()) if getattr(self, "_prev_ub", None) is not None else None
        self._prev_ub = ub
        nav_ev = None
        if getattr(self, "docking", None) is not None:
            self.odom.update(state[:3])          # measured local base velocity -> episode-local odometry
            if self.step_in_episode % self.perceive_every == 0:
                dets, visible_fn = self.perception.detect(obs, state)
                self.omap.update(dets, self.odom, self.step_in_episode, visible_fn=visible_fn)
                to_odom = np.array([[np.cos(self.odom.pose[2]), np.sin(self.odom.pose[2])],
                                    [-np.sin(self.odom.pose[2]), np.cos(self.odom.pose[2])]])
                if getattr(self.docking.cfg, "height_bands", False):
                    # height-banded depth points (base frame now -> odometry frame), see DockConfig.height_bands
                    bands = self.perception.obstacle_bands(obs, state, self.docking.band_edges(self.omap))
                    self._bands_odom = [(lo, hi, sup, self.odom.pose[:2] + P @ to_odom if len(P) else np.zeros((0, 2)))
                                        for lo, hi, sup, P in bands]
                    self._obst_odom = None
                else:
                    pts = self.perception.obstacles(obs, state)       # base frame now -> odometry frame (moves with us)
                    self._obst_odom = (self.odom.pose[:2] + pts @ to_odom if len(pts) else None)
                if dets and self.step_in_episode % (self.perceive_every * 6) == 0:
                    self._emit({"event": "perception", "episode": self.episode, "step": self.step_in_episode,
                                "detections": [{"label": d_["label"], "xyz_base": [round(float(v), 3) for v in d_["xyz_base"]],
                                                "score": round(float(d_.get("score", 0)), 3)} for d_ in dets],
                                "tracks": [t.summary() for t in self.omap.tracks]})
            if (contract is not None and contract.is_navigation
                    and self._active_contract == self._contract_id(contract)):
                st = self.docking.last_status
                nav_ev = {"docked": bool(st.get("docked")), "key": contract.key, "observation_id": int(self.step_in_episode),
                          "error_norm": st.get("error_norm"), "dock_target": st.get("dock_target"), "mode": st.get("mode")}
        if (getattr(self, "post_task1", None) is not None and contract is not None and not contract.is_navigation
                and self._active_contract == self._contract_id(contract)):
            post = self.post_task1.update(state, self._last_action, self.step_in_episode)
            self._last_post_t1 = post
        d = self.orch.tick(cls, self.step_in_episode, base_velocity=state[:3], dock_score=dock, dock_threshold=dock_thr,
                           postcondition=post, motion=motion, navigation_evidence=nav_ev)
        if self.orch.events:
            for e in self.orch.events:
                if (e.get("event") == "contract_end" and e.get("success") and str(e.get("key", "")).startswith("pick up from")
                        and getattr(self, "docking", None) is not None and self.docking.track_id is not None):
                    for t in self.omap.tracks:          # the grasped object leaves the floor map
                        if t.track_id == self.docking.track_id:
                            t.consumed = True
                self._emit({"event": "orch_" + e.pop("event"), "episode": self.episode, **e})
            self.orch.events.clear()
        live_c = getattr(self.orch, "contract", None)
        arm = getattr(live_c, "arm", None)
        self._record_v3_frame(self._route_model(), obs, state, self._last_pixels,
                              (self.episode, self._contract_id(live_c), getattr(live_c, "key", None),
                               getattr(arm, "arm_id", None), d.mode))
        if live_c is not None and self._contract_id(live_c) != getattr(self, "_active_contract", None):
            self._active_contract = self._contract_id(live_c)
            if live_c.is_navigation and getattr(self, "docking", None) is not None:
                self.docking.start(category(live_c.step.objects[0]), "carrying_bin" if self._carrying_bin() else "before_bin", self.odom)
            elif not live_c.is_navigation and getattr(self, "post_task1", None) is not None:
                self.post_task1.start(live_c.key, live_c.step.skill, live_c.step.objects, state, self.step_in_episode)
        if self.step_in_episode % 30 == 0 and d.mode == "EXECUTE" and self.orch.closure.state is not None:
            p, a = self.orch.closure.signals(cls)
            nav = {}
            contract = self.orch.contract
            if contract.is_navigation and self.orch.precond is not None and contract.precondition_key:
                arrived, score, margin = self.orch.precond.arrived(cls, contract.precondition_key)
                nav = {"next_key": contract.precondition_key, "arrived": arrived,
                       "precondition_score": float(score) if np.isfinite(score) else None,
                       "precondition_margin": float(margin) if np.isfinite(margin) else None}
            dock_dbg = {}
            if getattr(self, "_last_dock", None) is not None:
                dock_dbg = {"dock_gate": self._last_dock.get("gate"),
                            "dock_score": round(self._last_dock["score"], 3) if np.isfinite(self._last_dock["score"]) else None,
                            "dock_threshold": self._last_dock.get("threshold"),
                            "dock_centroid_u": round(self._last_dock["centroid_u"], 3) if np.isfinite(self._last_dock["centroid_u"]) else None,
                            "dock_lr_sym": round(self._last_dock["lr_sym"], 3) if np.isfinite(self._last_dock["lr_sym"]) else None}
            post_dbg = {"motion_window_path": getattr(self.orch.closure, "last_motion_path", None)}
            t1 = getattr(self, "_last_post_t1", None)
            if t1 is not None and t1.get("key") == self.orch.contract.key:
                post_dbg["t1_post"] = {k: t1.get(k) for k in ("satisfied", "failed", "reason", "detail")}
            if getattr(self, "_last_post", None) is not None:
                post_dbg = {"post_key": self._last_post["key"], "post_score": round(self._last_post["score"], 3),
                            "post_threshold": round(self._last_post["threshold"], 3), "post_satisfied": self._last_post["satisfied"]}
            self._emit({"event": "progress", "episode": self.episode, "step": self.step_in_episode, "index": self.orch.index,
                        "skill": d.skill, "progress": round(p, 3), "alignment": round(a, 3),
                        "base_velocity_observed": state[:3].tolist(), **nav, **dock_dbg, **post_dbg})
        every = getattr(self, "manip_snapshot_every", None)
        live = getattr(self.orch, "contract", None)   # None once the program has ended / is holding
        dense = every is not None and live is not None and not live.is_navigation and self.step_in_episode % every == 0
        if d.new_chunk_required or self.step_in_episode % 300 == 0 or dense:
            from PIL import Image
            snapshots = self.log_dir / "observations"
            snapshots.mkdir(exist_ok=True)
            rgb = np.asarray(self._find(obs, "::rgb", "zed_link"))[..., :3].astype(np.uint8)
            Image.fromarray(rgb).save(snapshots / f"ep{self.episode}_step{self.step_in_episode:06d}.png")
            # Raw wrist renders at the same steps (evidence). Separate subdirectory: offline tools glob
            # observations/*.png and observations/ep*_step*.png as HEAD frames.
            for tag, suffix in WRIST_SNAPSHOT_SUFFIXES:
                wrist = next((obs[k] for k in obs if k.endswith(suffix)), None)
                if wrist is not None:
                    (snapshots / "wrist").mkdir(exist_ok=True)
                    Image.fromarray(np.asarray(wrist)[..., :3].astype(np.uint8)).save(
                        snapshots / "wrist" / f"ep{self.episode}_step{self.step_in_episode:06d}_{tag}.png")
        self.step_in_episode += 1
        self.stats["steps"] += 1
        if d.mode != "EXECUTE":
            self.queue = []
            self.stats["holds"] += 1
            held = hold_baseline(torch.from_numpy(state)[None])[0, 0]
            self._last_action = held.numpy().copy()
            return held
        if d.new_chunk_required:
            self.queue = []
        if getattr(self, "docking", None) is not None and self.orch.contract is not None and self.orch.contract.is_navigation:
            self.queue = []
            obst = getattr(self, "_obst_odom", None)
            if obst is not None:
                c_, s_ = np.cos(self.odom.pose[2]), np.sin(self.odom.pose[2])
                obst = (obst - self.odom.pose[:2]) @ np.array([[c_, -s_], [s_, c_]])
            bands_odom = getattr(self, "_bands_odom", None)
            if bands_odom is not None:
                c_, s_ = np.cos(self.odom.pose[2]), np.sin(self.odom.pose[2])
                to_base = np.array([[c_, -s_], [s_, c_]])
                bands = [(lo, hi, sup, (P - self.odom.pose[:2]) @ to_base if len(P) else P) for lo, hi, sup, P in bands_odom]
                action, st = self.docking.step(state, self.omap, self.odom, obstacles=obst, bands=bands)
            else:
                action, st = self.docking.step(state, self.omap, self.odom, obstacles=obst)
            action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
            if self.step_in_episode % 15 == 1 or st.get("docked"):
                self._emit({"event": "dock", "episode": self.episode, "step": self.step_in_episode - 1, "index": self.orch.index,
                            "key": self.orch.contract.key, "odom": [round(float(v), 3) for v in self.odom.pose],
                            "base_cmd": [round(float(v), 3) for v in action[:3]], **st})
            self._last_action = action
            return torch.from_numpy(action)
        if getattr(self, "manip_executor", "wam") == "primitive" and self.orch.contract is not None and not self.orch.contract.is_navigation:
            self.queue = []
            action = self._primitive_action(state)
            if action is not None:
                action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
                self._last_action = action
                return torch.from_numpy(action)
        # Checkpoint routing. Contract changes already clear the queue (new_chunk_required / HOLD above); the
        # owner check additionally guarantees a chunk from one checkpoint is never executed under the other.
        model = self._route_model()
        model_name = "main" if model is None else model.name
        if self.queue and getattr(self, "_queue_model", "main") != model_name:
            self._emit({"event": "queue_cleared_model_switch", "episode": self.episode, "step": self.step_in_episode - 1,
                        "from_model": getattr(self, "_queue_model", "main"), "to_model": model_name, "dropped": len(self.queue)})
            self.queue = []
        if not self.queue:
            cond = torch.from_numpy(np.asarray(d.condition, np.float32))[None]
            k = getattr(self, "samples_navigation", None) if self.orch.contract.is_navigation else None
            # proprio input for the model actually used: nav/legacy (61,), history checkpoint (len(lags), 61)
            state_in = self._proprio_input(model, state)
            self._ensure_resident(model)
            if model is None or (model is getattr(self, "model", None) and not model.multiview and not model.is_v3):
                chunk = self._bounded_chunk(self._sample(feats, state_in, cond) if k is None else self._sample(feats, state_in, cond, samples=k))
            else:
                mfeats = self._model_feats(model, obs, self._last_pixels, feats)
                chunk = self._bounded_chunk(self._sample(mfeats, state_in, cond, samples=k, model=model))
            self._queue_model = model_name
            raw_base_cmd = chunk[0, :3].copy()
            chunk = self._apply_base_control(chunk, d.skill)
            chunk = self._apply_arm_gain(chunk, state, d.skill)
            self.queue = [chunk[i] for i in range(self.exec_horizon)]
            self.chunk_id += 1
            self.stats["chunks"] += 1
            infer_ms = (time.monotonic() - t0) * 1000
            self.stats["infer_ms"].append(infer_ms)
            self._emit({"event": "chunk", "episode": self.episode, "step": self.step_in_episode - 1, "chunk": self.chunk_id,
                        "skill": d.skill, "index": self.orch.index, "arm": self.orch.contract.arm.arm_id, "infer_ms": round(infer_ms, 1),
                        "arm_key": self.orch.contract.arm_key,
                        # which checkpoint produced this chunk (navigation -> main; other contracts -> manip if loaded)
                        "model": model_name,
                        "model_checkpoint": getattr(self, "checkpoint", None) if model is None else model.checkpoint,
                        "model_checkpoint_step": getattr(self, "checkpoint_step", None) if model is None else model.step,
                        "model_views": 1 if model is None else len(model.views),
                        **self._proprio_log(model), **self._v3_chunk_log(model),
                        "base_control": self.base_control_mode, "samples": int(k or getattr(self, "samples", 1)),
                        "base_cmd_raw_wam": [round(float(x), 4) for x in raw_base_cmd],
                        "base_cmd": [round(float(x), 4) for x in chunk[0, :3]],
                        "base_velocity_command_local": (chunk[0, :3] * [0.75, 0.75, 1.0]).tolist(),
                        "grip": [round(float(chunk[0, 14]), 3), round(float(chunk[0, 22]), 3)],
                        # Generic execution telemetry (any skill/task): measured upper-body joint positions when the
                        # chunk was planned and the chunk's final commanded targets, trunk(4)+left arm(7)+right arm(7).
                        # Lets a failed contact skill be attributed to "commanded hold" vs "moved but missed" offline.
                        "q_meas": [round(float(x), 3) for x in np.concatenate([state[53:57], state[3:10], state[28:35]])],
                        "q_cmd_end": [round(float(x), 3) for x in np.concatenate([chunk[-1, 3:7], chunk[-1, 7:14], chunk[-1, 15:22]])],
                        # full raw proprio at chunk start: lets an offline check compare the closed-loop input
                        # distribution (eef poses, grippers, trunk) against the demonstrations (covariate shift audit)
                        "state_raw": [round(float(x), 4) for x in state]})
        out = np.asarray(self.queue.pop(0), dtype=np.float32)
        self._last_action = out
        return torch.from_numpy(out)

    def _apply_base_control(self, chunk, skill):
        """Assign one owner to the base without changing the checkpoint or upper-body actions.

        `hold_during_manipulation` is the first, no-retraining hybrid diagnostic: WAM still drives
        navigation, but its base residual is ignored for contact skills.  It is not a SLAM claim.
        """
        if self.base_control_mode == "hold_during_manipulation" and skill != "move to":
            chunk = np.array(chunk, dtype=np.float32, copy=True)
            chunk[:, :3] = 0.0
        elif skill == "move to" and getattr(self, "move_to_base_gain", None) is not None and bool((self.move_to_base_gain != 1.0).any()):
            chunk = np.array(chunk, dtype=np.float32, copy=True)
            chunk[:, :3] = np.clip(chunk[:, :3] * self.move_to_base_gain[None, :],
                                   self.action_low[:3], self.action_high[:3])
        return chunk

    # (action_slice, measured-qpos state_slice) for trunk, left arm, right arm.  Grippers (14:15, 22:23)
    # and base (0:3) are NOT amplified: the open-lid demo keeps the gripper open (a push/lift, not a grasp).
    ARM_GAIN_PAIRS = (((3, 7), (53, 57)), ((7, 14), (3, 10)), ((15, 22), (28, 35)))

    def _apply_arm_gain(self, chunk, state, skill):
        """Scale the contact-skill arm/trunk displacement from the measured pose: q0 + g*(target - q0).

        Only joints the WAM actually commands to move are affected (a holding joint has target~=q0, so the
        product is ~0).  Re-clipped to the actuator range; no checkpoint change, no ground truth read.
        """
        g = getattr(self, "open_lid_arm_gain", 1.0)
        if skill != "open lid" or g == 1.0:
            return chunk
        chunk = np.array(chunk, dtype=np.float32, copy=True)
        for (alo, ahi), (slo, shi) in self.ARM_GAIN_PAIRS:
            q0 = np.asarray(state[slo:shi], np.float32)[None, :]
            chunk[:, alo:ahi] = q0 + g * (chunk[:, alo:ahi] - q0)
        chunk[:, 3:22] = np.clip(chunk[:, 3:22], self.action_low[3:22], self.action_high[3:22])
        return chunk

    def _bounded_chunk(self, chunk):
        """Match actuator clipping using scene-exported limits; record raw violations for diagnosis."""
        if chunk.shape != (32, 23) or not np.isfinite(chunk).all():
            raise ValueError("WAM must produce a finite 32x23 chunk")
        clipped = np.clip(chunk, self.action_low, self.action_high)
        count = int(np.count_nonzero(chunk != clipped))
        if count:
            self._emit({"event": "action_bounds", "episode": self.episode, "step": self.step_in_episode,
                        "clipped_components": count, "max_correction": float(np.max(np.abs(chunk - clipped))),
                        "clipped_by_group": {name: int(np.count_nonzero(chunk[:, low:high] != clipped[:, low:high]))
                                             for name, (low, high) in ACTION_GROUPS.items()}})
        return clipped

    def hold(self, obs):
        """Copy-state hold in controller units (NOT a zero vector): arms/trunk keep current joints, base stops.
        Does not touch the proprio buffer: the server only calls it after act() has already recorded this step."""
        self.queue = []
        self._reset_v3_contexts("hold_on_error")
        state, _ = self._preprocess(self._policy_observation(obs))
        self.stats["holds"] += 1
        return hold_baseline(torch.from_numpy(state)[None])[0, 0]


# ---- synthetic CPU self-test (stub models; no checkpoint, weights, GPU or simulator) ----------------
def synthetic_self_test():
    """Routing, multi-view preprocessing, head-only gates, queue clearing, logging, encoder sharing and per-step proprio
    history (training rule, per model actually used) with stubs."""
    import tempfile
    import types
    from omegaconf import OmegaConf
    targets = "data/skill_vtt_task013/contract_targets.json"
    targets_sha = hashlib.sha256((ROOT / targets).read_bytes()).hexdigest()
    wrists = ["observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0"]
    MARK = {"main": 0.01, "manip": 0.02}
    calls = []  # (encoder id, batch) per DINO pass

    class Encoder(torch.nn.Module):
        def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
            px = pixel_values.float()
            calls.append((id(self), px.shape[0], px.clone()))
            B = px.shape[0]
            base = px.mean(dim=(1, 2, 3))[:, None, None]
            h = base + torch.linspace(0, 1, 1024)[None, None, :] * torch.arange(1, 202).float()[None, :, None] / 201
            return types.SimpleNamespace(hidden_states=[h + i for i in range(25)], last_hidden_state=h)

    class Action(torch.nn.Module):
        action_len = 32

        def __init__(self, name):
            super().__init__()
            self.name, self.seen = name, []
            self.gain = 0.0   # > 0: chunk also depends on proprio input and noise (bit-identity checks)

        def forward(self, t, noisy_actions, qpos_history, dino_features_list, task_cond):
            self.seen.append([tuple(f.shape) for f in dino_features_list])
            i = float(t[0]) * 10  # Euler step index for num_steps=10: lands exactly on the marker
            out = (MARK[self.name] - noisy_actions) * 10 / (10 - i)
            if self.gain:
                out = out + self.gain * (qpos_history.float().mean(dim=(1, 2))[:, None, None] + 0.1 * noisy_actions)
            return {"final_pred": out}

    class Wrapper(torch.nn.Module):
        def __init__(self, name, views, encoder, lags=(0,)):
            super().__init__()
            self.vision_encoder, self.action_model, self.views = encoder, Action(name), views
            self.device, self.encoder_dtype = torch.device("cpu"), torch.float32
            self.feat_layers, self.include_cls_register, self.num_register_tokens = [-12, -8, -4], True, 4
            self.action_min = torch.zeros(23)
            self.action_repr_name, self.proprio_mask_name = "absolute", "positions_only"
            self.proprio_history = None if list(lags) == [0] else {"lags": list(lags)}
            self.proprio_in = []   # raw proprio tensor handed to normalize_state, one per flow sample

        def normalize_state(self, s):
            self.proprio_in.append(s.detach().clone())
            return s
        def denormalize_action(self, x): return x
        def to_absolute_action(self, a, s): return a

        def get_vision_features(self, px):  # contract of harness.wam_behavior (B,V,3,H,W) -> per-layer (B,V,N,D)
            multi = px.dim() == 5
            if multi != (len(self.views) > 1) or (multi and px.shape[1] != len(self.views)):
                raise ValueError("view count mismatch")
            B = px.shape[0]
            out = self.vision_encoder(pixel_values=px.reshape(-1, *px.shape[-3:]))
            return [out.hidden_states[l].reshape(B, -1, 201, 1024) if multi else out.hidden_states[l] for l in self.feat_layers]

    def cfg_for(views, sha=targets_sha, cond="contract_proto", encoder="models/dinov3-vitl16-pretrain-lvd1689m", lags=None):
        model = {"vision_encoder": {"checkpoint_path": encoder}}
        if views is not None:
            model["views"] = views
        if lags is not None:
            model["proprio_lags"] = list(lags)
        return OmegaConf.create({"dataset": {"image_size": [224, 224], "cond_mode": cond, "contract_targets_sha256": sha},
                                 "model": model, "common": {"proprio_len": 1 if lags is None else len(lags)}})

    def loader_for(specs):
        def load(name, checkpoint, device, cache):
            cfg = specs[checkpoint]
            key = str(cfg.model.vision_encoder.checkpoint_path)
            enc = cache.setdefault(key, Encoder())
            views = [HEAD_CAMERA] + [str(v) for v in (cfg.model.get("views") or [HEAD_CAMERA])[1:]]
            return PolicyModel(name, checkpoint, 7 if name == "main" else 9, cfg, cfg.dataset.cond_mode,
                               Wrapper(name, views, enc, lags=cfg.model.get("proprio_lags") or [0]))
        return load

    colors = {"zed_link": 10, "left_realsense_link": 100, "right_realsense_link": 200}
    obs = {"robot_r1::proprio": np.zeros(61, np.float32),
           **{f"robot_r1::robot_r1:{link}:Camera:0::rgb": np.full((224, 224, 4), c, np.uint8) for link, c in colors.items()}}
    orch = {"vlm": "none", "answer": "none", "frozen": True, "precondition": True, "train_targets": targets}
    specs = {"stub://main": cfg_for(None), "stub://manip": cfg_for([HEAD_CAMERA, *wrists])}
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        made = []

        def make(orchestrate, specs=specs, main="stub://main", exec_horizon=4):  # one log dir per policy: 1 s trace names
            made.append(Path(tmp) / f"policy{len(made)}")
            return WamPolicy(main, skill="move to", exec_horizon=exec_horizon, num_steps=10, device="cpu", cuda_fraction=0.3,
                             seed=0, log_dir=made[-1], orchestrate=orchestrate, model_loader=loader_for(specs))

        # 1) Single-view main only: head preprocessing identical to the pre-change chain; one head DINO pass per step.
        p = make(dict(orch))
        import cv2
        ref = np.asarray(obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"])[..., :3]
        ref = normalize_image(cv2.resize(ref, (224, 224), interpolation=cv2.INTER_AREA) if ref.shape[0] != 224 else ref)
        assert np.array_equal(p._preprocess(obs)[1], ref)
        fake = types.SimpleNamespace(cfg=p.cfg, _find=WamPolicy._find)  # unbound use by check_openlid_replay.py
        assert np.array_equal(WamPolicy._preprocess(fake, {"r::proprio": obs["robot_r1::proprio"], "r::zed_link::rgb": obs[
            "robot_r1::robot_r1:zed_link:Camera:0::rgb"]})[1], ref)
        calls.clear()
        a = p.act(obs)
        assert np.allclose(a.numpy(), MARK["main"]) and [c[1] for c in calls] == [1]
        assert p.wrapper.action_model.seen[0] == [(1, 201, 1024)] * 3 and p.manip is None
        assert p.metadata()["manip_checkpoint"] is None and p.metadata()["views"] == [HEAD_CAMERA]
        results["single_view"] = "head-only pass, legacy sampler call, 3D features"

        # 2) nav -> main, manip -> multi-view manip checkpoint, back to nav -> main.
        p = make({**orch, "manip_checkpoint": "stub://manip"})
        assert p.encoder_shared and p.manip.wrapper.vision_encoder is p.wrapper.vision_encoder and p.manip.multiview
        calls.clear()
        a = p.act(obs)
        assert p.orch.contract.is_navigation and np.allclose(a.numpy(), MARK["main"]) and [c[1] for c in calls] == [1]
        assert len(p.queue) == 3 and p._queue_model == "main"
        p.orch.index, p.orch.contract = 1, None     # contract boundary: next program step is manipulation
        calls.clear()
        a = p.act(obs)
        assert not p.orch.contract.is_navigation and np.allclose(a.numpy(), MARK["manip"]), "queued main chunk leaked"
        assert [c[1] for c in calls] == [1, 3], calls  # head-only pass, then ONE batched 3-view pass at chunk time
        head_px = torch.from_numpy(p._preprocess(obs)[1])
        assert torch.equal(calls[0][2][0], head_px) and torch.equal(calls[1][2][0], head_px)
        for v, link in ((1, "left_realsense_link"), (2, "right_realsense_link")):
            want = prep_rgb(obs[f"robot_r1::robot_r1:{link}:Camera:0::rgb"], 224)
            assert torch.equal(calls[1][2][v], torch.from_numpy(want)), f"view {v} order/preprocessing"
        assert p.manip.wrapper.action_model.seen[-1] == [(1, 3, 201, 1024)] * 3
        # head-only consumers: patch tokens/CLS come from the solo head pass, not the 3-view batch
        solo = p.wrapper.vision_encoder(pixel_values=head_px[None]).last_hidden_state
        assert torch.equal(p._last_patch, solo[0, 5:]) and np.array_equal(p._last_cls, solo[0, 0].numpy())
        # owner check: a queue owned by the other checkpoint is dropped even without a contract change
        p._queue_model = "main"
        a = p.act(obs)
        assert np.allclose(a.numpy(), MARK["manip"]) and p._queue_model == "manip"
        p.orch.index, p.orch.contract = 2, None     # back to navigation
        a = p.act(obs)
        assert p.orch.contract.is_navigation and np.allclose(a.numpy(), MARK["main"])
        trace = [json.loads(l) for l in open(p.trace.name)]
        chunks = [r for r in trace if r["event"] == "chunk"]
        assert [r["model"] for r in chunks] == ["main", "manip", "manip", "main"], [r["model"] for r in chunks]
        assert [r["model_checkpoint"] for r in chunks] == ["stub://main", "stub://manip", "stub://manip", "stub://main"]
        assert [r["model_views"] for r in chunks] == [1, 3, 3, 1]
        assert sum(r["event"] == "queue_cleared_model_switch" for r in trace) == 1
        heads = sorted(x.name for x in (p.log_dir / "observations").glob("ep*_step*.png"))
        snaps = sorted(x.name for x in (p.log_dir / "observations/wrist").glob("*.png"))
        assert heads and all("wrist" not in n for n in heads), heads  # head-frame globs stay head-only
        assert {n.replace("_lwrist", "").replace("_rwrist", "") for n in snaps} == set(heads), (heads, snaps)
        meta = p.metadata()
        assert meta["manip_checkpoint"] == "stub://manip" and meta["manip_views"] == [HEAD_CAMERA, *wrists]
        results["routing"] = [r["model"] for r in chunks]
        results["snapshots"] = {"head": heads[:2], "wrist": snaps[:4]}

        # 3) Missing wrist observation fails loudly (server answers hold with --hold-on-error), never a silent head-only chunk.
        p.orch.index, p.orch.contract = 1, None
        frames = p._proprio_frames
        try:
            p.act({k: v for k, v in obs.items() if "left_realsense" not in k})
        except KeyError:
            pass
        else:
            raise AssertionError("missing wrist camera accepted")
        p.hold(obs)   # the server's hold-on-error answer for that same control step
        assert p._proprio_frames == frames + 1, "a failed act() + hold must record the step's state exactly once"

        # 4) Invalid configurations are rejected.
        for bad, why in (({"stub://main": specs["stub://main"], "stub://manip": cfg_for([HEAD_CAMERA, *wrists], sha="0" * 64)}, "targets"),
                         ({"stub://main": specs["stub://main"], "stub://manip": cfg_for([HEAD_CAMERA, *wrists], cond="skill_proto")}, "cond"),
                         ({"stub://main": specs["stub://main"], "stub://manip": cfg_for([HEAD_CAMERA, "observation.depth.x"])}, "view")):
            try:
                make({**orch, "manip_checkpoint": "stub://manip"}, specs=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid manip checkpoint accepted: {why}")
        assert live_rgb_suffix(HEAD_CAMERA) == ":zed_link:Camera:0::rgb"
        assert "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb".endswith(live_rgb_suffix(wrists[1]))

        # 5) Real build path shares the encoder: build_models called twice through the memoizing factory.
        from harness.wam_behavior import load_upstream
        _, runner = load_upstream()
        saved = runner.ModelFactory.__dict__["create_vision_encoder"]
        created = []
        runner.ModelFactory.create_vision_encoder = staticmethod(lambda path, dtype, device: created.append(path) or (object(), 1024, 4, 16))
        stub_factory = runner.ModelFactory.__dict__["create_vision_encoder"]
        global build_models
        real_build = build_models
        build_models = lambda cfg, device: ((runner.ModelFactory.create_vision_encoder(
            str(ROOT / cfg.model.vision_encoder.checkpoint_path), torch.bfloat16, device)), None)
        try:
            cache = {}
            e1 = _build_sharing_encoder(cfg_for(None), torch.device("cpu"), cache)
            e2 = _build_sharing_encoder(cfg_for([HEAD_CAMERA, *wrists]), torch.device("cpu"), cache)
            e3 = _build_sharing_encoder(cfg_for(None, encoder="models/other_encoder"), torch.device("cpu"), cache)
            assert e1 is e2 and e3 is not e1 and len(created) == 2
            assert runner.ModelFactory.__dict__["create_vision_encoder"] is stub_factory
        finally:
            build_models = real_build
            runner.ModelFactory.create_vision_encoder = saved
        results["encoder_sharing"] = "same path -> one encoder object; different path -> separate load; factory restored"

        # 6) Proprio history: one raw state per act(); a history checkpoint gets stack_proprio_history (the training rule).
        LAGS = [0, 8, 16, 32, 64]
        hspecs = {**specs, "stub://legacy": cfg_for(None, cond="skill_proto"),
                  "stub://hist": cfg_for(None, cond="skill_proto", lags=LAGS), "stub://histm": cfg_for(None, lags=LAGS)}
        rng = np.random.default_rng(0)
        at = lambda s: {**obs, "robot_r1::proprio": s}   # noqa: E731
        rule = lambda seq, t: torch.from_numpy(stack_proprio_history(seq, t, LAGS))   # noqa: E731

        # (a) lags [0]: the raw (61,) state object reaches the sampler, chunks bit-identical to the pre-history call
        #     _sample(head feats, raw state, cond) on an identically seeded policy (stub output depends on state + noise).
        seq = rng.standard_normal((12, 61)).astype(np.float32)
        p, ref = make(None, hspecs, "stub://legacy"), make(None, hspecs, "stub://legacy")
        p.model.wrapper.action_model.gain = ref.model.wrapper.action_model.gain = 1e-3
        chunks = []
        for t in range(len(seq)):
            a = p.act(at(seq[t])).numpy()
            if t % 4 == 0:
                state, px = ref._preprocess(at(seq[t]))
                chunks.append(ref._bounded_chunk(ref._sample(ref._encode(px)[0], state, ref.prototypes[ref.skill][None])))
                assert torch.equal(p.model.wrapper.proprio_in[-1], torch.from_numpy(seq[t])[None])   # (1, 61)
            assert np.array_equal(a, chunks[-1][t % 4]), f"legacy chunk differs at step {t}"
        last = seq[-1]   # (each seq[-1] is a new view object)
        assert len({c.tobytes() for c in chunks}) == 3 and p._proprio_input(p.model, last) is last
        assert p.metadata()["proprio_lags"] == [0] and p.metadata()["manip_proprio_lags"] is None

        # (b) every step sampled: history == stack_proprio_history(seq, t, lags) with padding (t < 64) and without.
        seqA = rng.standard_normal((100, 61)).astype(np.float32)
        p = make(None, hspecs, "stub://hist", exec_horizon=1)
        seen = p.model.wrapper.proprio_in
        assert p.model.proprio_lags == LAGS and p._proprio_buf.maxlen == 65
        for t in range(len(seqA)):
            p.act(at(seqA[t]))
            assert len(seen) == t + 1 and seen[-1].shape == (1, 5, 61) and torch.equal(seen[-1][0], rule(seqA, t)), t
        assert torch.equal(seen[0][0], torch.from_numpy(seqA[[0] * 5]))            # episode start: all padded
        assert torch.equal(seen[20][0], torch.from_numpy(seqA[[20, 12, 4, 0, 0]]))
        assert torch.equal(seen[99][0], torch.from_numpy(seqA[[99, 91, 83, 67, 35]]))

        # (c) reset() clears: the next episode pads with its OWN first frame.
        p.reset()
        assert len(p._proprio_buf) == 0 and p._proprio_frames == 0
        seqB = rng.standard_normal((20, 61)).astype(np.float32)
        n0 = len(seen)
        for t in range(len(seqB)):
            p.act(at(seqB[t]))
            assert torch.equal(seen[-1][0], rule(seqB, t)), t
        assert torch.equal(seen[n0][0], torch.from_numpy(seqB[[0] * 5]))

        # (e) steps served from the queue are recorded too (exec_horizon 8: sampled only every 8th step).
        p = make(None, hspecs, "stub://hist", exec_horizon=8)
        seen = p.model.wrapper.proprio_in
        for t in range(len(seqA)):
            p.act(at(seqA[t]))
            if t % 8 == 0:
                assert torch.equal(seen[-1][0], rule(seqA, t)), t
        assert len(seen) == 13 and p._proprio_frames == 100 and np.array_equal(np.stack(p._proprio_buf), seqA[-65:])
        ch = [r for r in (json.loads(l) for l in open(p.trace.name)) if r["event"] == "chunk"]
        assert [r["step"] for r in ch] == list(range(0, 100, 8))
        assert all(r["proprio_lags"] == LAGS and r["proprio_frames"] == r["step"] + 1
                   and r["proprio_padded_lags"] == sum(l > r["step"] for l in LAGS) for r in ch)

        # (d) orchestrated nav (main, lags [0]) <-> manip (history checkpoint): (61,) vs (5, 61) per model actually used.
        p = make({**orch, "manip_checkpoint": "stub://histm"}, hspecs)
        nav_in, man_in = p.model.wrapper.proprio_in, p.manip.wrapper.proprio_in
        assert p.model.proprio_lags == [0] and p.manip.proprio_lags == LAGS and p._proprio_buf.maxlen == 65
        seqC = rng.standard_normal((24, 61)).astype(np.float32)
        for t in range(len(seqC)):
            if t in (12, 20):   # contract boundary: 12 -> manipulation (history model), 20 -> navigation (main)
                p.orch.index, p.orch.contract = (1 if t == 12 else 2), None
            n_nav, n_man = len(nav_in), len(man_in)
            p.act(at(seqC[t]))
            assert all(s.shape == (1, 61) and torch.equal(s[0], torch.from_numpy(seqC[t])) for s in nav_in[n_nav:]), t
            assert all(s.shape == (1, 5, 61) and torch.equal(s[0], rule(seqC, t)) for s in man_in[n_man:]), t
            if t in (0, 12, 20):
                assert (len(man_in) > n_man) if t == 12 else (len(nav_in) > n_nav), f"no chunk at contract start {t}"
        assert p._proprio_frames == len(seqC)
        ch = [r for r in (json.loads(l) for l in open(p.trace.name)) if r["event"] == "chunk"]
        assert {r["model"]: r["proprio_lags"] for r in ch} == {"main": [0], "manip": LAGS}
        assert all(r["proprio_frames"] == r["step"] + 1 for r in ch)
        assert p.metadata()["proprio_lags"] == [0] and p.metadata()["manip_proprio_lags"] == LAGS
        # guards: a single frame never reaches a history model; the buffer must hold this step's state
        for call, err in ((lambda: p._sample_once(None, seqC[-1], None, model=p.manip), ValueError),
                          (lambda: p._proprio_input(p.manip, seqC[0]), RuntimeError),
                          (lambda: PolicyModel("x", "stub://x", 0, hspecs["stub://hist"], "skill_proto",
                                               Wrapper("main", [HEAD_CAMERA], Encoder(), lags=[0, 4, 8, 16, 32])), ValueError)):
            try:
                call()
            except err:
                pass
            else:
                raise AssertionError("proprio contract violation accepted")
        results["proprio_history"] = {"lags": LAGS, "legacy_bit_identical_chunks": len(chunks),
                                      "orchestrated_chunks": [f"{r['model']}@step{r['step']}:padded_lags={r['proprio_padded_lags']}"
                                                              for r in ch]}
    return results


# ---- server ---------------------------------------------------------------------------------------
def control_message(policy, msg):
    """Local status/clarification channel. Replies cannot execute a cached action."""
    if not isinstance(msg, dict) or not ("status" in msg or "clarification" in msg):
        return None
    orch = policy.orch
    result = {"episode": policy.episode, "step": policy.step_in_episode,
              "status": orch.status if orch else "executor_only",
              "question_id": orch.question_id if orch else None,
              "choices": list(orch.pending_question) if orch else []}
    if "clarification" in msg:
        reply = msg["clarification"]
        source = reply.get("source", "human") if isinstance(reply, dict) else None
        accepted = (isinstance(reply, dict) and source in ("human", "simulated") and orch is not None
                    and type(reply.get("episode")) is int and reply["episode"] == policy.episode
                    and orch.answer(reply.get("answer"), question_id=reply.get("question_id")))
        if accepted:
            policy.queue = []
            if hasattr(policy, "_reset_v3_contexts"):
                policy._reset_v3_contexts("clarification_resume")
        result["accepted"] = bool(accepted)
        result["source"] = source
        policy._emit({"event": "clarification_response", "episode": policy.episode,
                      "question_id": reply.get("question_id") if isinstance(reply, dict) else None,
                      "answer": reply.get("answer") if isinstance(reply, dict) else None,
                      "source": source, "accepted": bool(accepted)})
        result["status"] = orch.status if orch else "executor_only"
        if orch:
            result.update(question_id=orch.question_id, choices=list(orch.pending_question))
    return result


def _health(connection, request):
    if getattr(request, "path", None) == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def serve(policy, host, port, hold_on_error):
    async def handler(websocket):
        log.info("connection from %s", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(policy.metadata()))
        controls_episode = False
        while True:
            try:
                msg = unpackb(await websocket.recv())
            except websockets.ConnectionClosed:
                if controls_episode:
                    policy.finish_episode()
                log.info("connection closed; stats=%s", {k: (round(float(np.mean(v)), 1) if k == "infer_ms" and v else v)
                                                          for k, v in policy.stats.items()})
                return
            control = control_message(policy, msg)
            if control is not None:
                await websocket.send(packer.pack(control))
                continue
            if "reset" in msg:
                controls_episode = True
                policy.reset()
                continue
            controls_episode = True
            t0 = time.monotonic()
            try:
                action = policy.act(msg)
            except Exception:
                log.exception("policy.act failed")
                if not hold_on_error:
                    raise
                action = policy.hold(msg)
            await websocket.send(packer.pack({"action": action.cpu().numpy(),
                                              "server_timing": {"infer_ms": (time.monotonic() - t0) * 1000}}))

    async with ws_server.serve(handler, host, port, compression=None, max_size=None, process_request=_health) as server:
        log.info("serving on %s:%d", host, port)
        await server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="main checkpoint (navigation + non-orchestrated modes); "
                    "orchestrate.manip_checkpoint optionally serves all non-navigation contracts")
    ap.add_argument("--skill", default="move to", help="VTT prototype class name or id used when no schedule entry applies")
    ap.add_argument("--skill-schedule", default=None,
                    help='JSON list of [start_step, skill] pairs, e.g. \'[[0,"move to"],[600,"pick up from"]]\'')
    ap.add_argument("--program", default=None, help="configs/task013_fixed_schedule.json: ordered skills advanced by VTT closure / budget")
    ap.add_argument("--budget-factor", type=float, default=1.5, help="per-skill step budget = median demo frames x this")
    ap.add_argument("--memory-vtt", default=None, help="verified-vtt-v1 export; requires orchestrate.memory_context")
    ap.add_argument("--orchestrate", default=None,
                    help='JSON: {"instruction": str, "visible_objects": [..], "vlm": "none|qwen-cpu|qwen-gpu", "answer": "none|<idx>|<substring>", '
                         '"ask_budget": 2, "max_attempts": 2, "frozen": false, "bank_in": path, "precondition": true}')
    ap.add_argument("--exec-horizon", type=int, default=8, help="steps executed from each 32-step chunk before resampling")
    ap.add_argument("--num-steps", type=int, default=10, help="flow-matching Euler steps")
    ap.add_argument("--samples", type=int, default=1, help="flow samples averaged per chunk (K x inference cost)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cuda-fraction", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log-dir", default=str(ROOT / "outputs/behavior/policy_server"))
    ap.add_argument("--hold-on-error", action="store_true", help="answer copy-state hold instead of closing on errors")
    ap.add_argument("--self-test", action="store_true", help="load model, run one synthetic act(), exit")
    ap.add_argument("--synthetic-self-test", action="store_true",
                    help="CPU stub-model test of routing/multi-view/queue/logging; needs no checkpoint")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.synthetic_self_test:
        print(json.dumps({"synthetic_self_test": "PASS", **synthetic_self_test(), "note": "stub models; not a robot result"}, indent=2))
        return
    if not args.checkpoint:
        ap.error("--checkpoint is required")
    schedule = json.loads(args.skill_schedule) if args.skill_schedule else None
    policy = WamPolicy(args.checkpoint, skill=args.skill, exec_horizon=args.exec_horizon, num_steps=args.num_steps,
                       device=args.device, cuda_fraction=args.cuda_fraction, seed=args.seed, log_dir=args.log_dir,
                       skill_schedule=schedule, program=args.program, budget_factor=args.budget_factor, memory_vtt=args.memory_vtt,
                       orchestrate=json.loads(args.orchestrate) if args.orchestrate else None, samples=args.samples)
    if args.self_test:
        obs = {"robot_r1::proprio": np.zeros(61, np.float32),
               **{f"robot_r1::robot_r1:{link}:Camera:0::rgb": np.zeros((224, 224, 4), np.uint8)
                  for link in ("zed_link", "left_realsense_link", "right_realsense_link")}}
        t0 = time.monotonic()
        a = policy.act(obs)
        report = {"self_test": "ok", "action_shape": list(a.shape), "first_call_ms": round((time.monotonic() - t0) * 1000, 1)}
        if policy.manip is not None:  # the orchestrator starts with navigation; also exercise the manip model once
            t0 = time.monotonic()
            state, px = policy._preprocess(obs)
            feats, _ = policy._encode(px)
            cond = policy.prototypes[policy._skill_id("open lid")][None]
            if policy.manip.is_v3:
                policy._record_v3_frame(policy.manip, obs, state, px, ("self_test_manip",))
            policy._ensure_resident(policy.manip)
            state_in = policy._proprio_input(policy.manip, state)   # act() above recorded this same state
            chunk = policy._sample(policy._model_feats(policy.manip, obs, px, feats), state_in, cond, samples=1, model=policy.manip)
            assert chunk.shape == (32, 23) and np.isfinite(chunk).all()
            report.update(manip_chunk_shape=list(chunk.shape), manip_call_ms=round((time.monotonic() - t0) * 1000, 1),
                          manip_proprio_input_shape=list(np.shape(state_in)), vision_encoder_shared=policy.encoder_shared)
        print(json.dumps({**report, "metadata": policy.metadata(), "note": "synthetic zeros; not a robot result"}, indent=2))
        return
    asyncio.run(serve(policy, args.host, args.port, args.hold_on_error))


if __name__ == "__main__":
    main()
