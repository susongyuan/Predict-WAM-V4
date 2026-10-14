"""VTT closure monitor: the skill condition vector doubles as the skill's own progress/completion signal.

LiLa-WAM's Visual Transition Token for a demonstration segment is VTT = CLS(end) - CLS(start) in DINOv3
space, and the policy is conditioned on the class prototype (mean over training segments). At run time
the same frozen encoder gives CLS(t); the realized transition r(t) = CLS(t) - CLS(t_skill_start) can be
projected onto the prototype direction p̂:

    progress(t) = <r(t), p̂> / |p|        (1.0 == a prototypical complete transition)
    alignment(t) = cos(r(t), p̂)

This module only computes these signals and applies a three-valued rule; thresholds must be fitted on
the development split (offline: validation demonstrations; online: dev instances 10-19). Offline
evidence (docs/system_design.md §3.4): alignment at segment end is strong for articulated skills
(open lid 0.85, close lid 0.76, open door 0.64), moderate for pick/place (0.37) and weak for
navigation (0.17), so navigation completion must come from another predicate (next-skill precondition).

Nothing here touches simulator ground truth.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from harness import task_profile

ROOT = Path(__file__).resolve().parents[1]
VTT_DIR = task_profile.VTT_DIR  # P2_TASK_ID; default 13 = ROOT / "data/skill_vtt_task013" (unchanged)


@dataclass
class ClosureConfig:
    done_progress: float        # progress >= this -> closure True (per class, fitted on dev)
    done_alignment: float       # and cos >= this
    stall_steps: int            # no progress improvement for this many policy steps -> stalled
    stall_delta: float = 0.02   # minimum improvement that counts as progress
    min_steps: int = 30         # never declare closure before this many steps
    max_steps: int | None = None  # budget from median demo duration * factor; None = no budget
    # Generic motion-stall guard (configs/motion_stall_calibration.json, demo-calibrated): a VTT plateau counts as a
    # stall only if the policy-visible upper-body joint path over the last `motion_window` control steps is also
    # below `motion_eps` (the robot is idle at a hold fixed point).  None = legacy VTT-only stall rule.  The demos'
    # visual transition is back-loaded (open lid: lift after ~40% of the segment), so a VTT plateau alone would fail
    # a robot that is still working, exactly as it failed every closed-loop open-lid attempt at 942-1530 steps.
    motion_window: int = 300
    motion_eps: float | None = None


@dataclass
class ClosureState:
    skill: int
    start_cls: np.ndarray
    steps: int = 0
    best_progress: float = -1e9
    best_step: int = 0
    history: list = field(default_factory=list)


class VTTClosureMonitor:
    def __init__(self, vtt_dir=VTT_DIR):
        meta = json.loads((Path(vtt_dir) / "meta.json").read_text())
        self.classes = meta["classes"]
        self.names = {v: k for k, v in self.classes.items()}
        self.proto = np.load(Path(vtt_dir) / "prototypes.npy").astype(np.float64)
        self.norm = np.linalg.norm(self.proto, axis=1)
        self.unit = self.proto / (self.norm[:, None] + 1e-8)
        self.state = None

    def start(self, skill, cls_now):
        self.state = ClosureState(int(skill), np.asarray(cls_now, dtype=np.float64).copy())
        return self.state

    def signals(self, cls_now):
        """Return (progress, alignment) of the realized transition against the active skill prototype."""
        s = self.state
        r = np.asarray(cls_now, dtype=np.float64) - s.start_cls
        rn = np.linalg.norm(r)
        progress = float(r @ self.unit[s.skill] / (self.norm[s.skill] + 1e-8))
        alignment = float(r @ self.unit[s.skill] / (rn + 1e-8)) if rn > 1e-6 else 0.0
        return progress, alignment

    def update(self, cls_now, cfg: ClosureConfig):
        """Three-valued closure predicate: True (done), False (stalled/over budget), None (unknown)."""
        s = self.state
        s.steps += 1
        progress, alignment = self.signals(cls_now)
        if progress > s.best_progress + cfg.stall_delta:
            s.best_progress, s.best_step = progress, s.steps
        s.history.append((s.steps, round(progress, 4), round(alignment, 4)))
        verdict = None
        if s.steps >= cfg.min_steps and progress >= cfg.done_progress and alignment >= cfg.done_alignment:
            verdict = True
        elif cfg.max_steps is not None and s.steps >= cfg.max_steps:
            verdict = False
        elif s.steps - s.best_step >= cfg.stall_steps and s.steps >= cfg.min_steps:
            verdict = False
        return verdict, progress, alignment


def default_configs(program_path=ROOT / "configs/task013_skill_program.json", exec_horizon=8, budget_factor=2.0,
                    motion_calibration=ROOT / "configs/motion_stall_calibration.json"):
    """Provisional per-class configs from training-demo durations; done thresholds are placeholders until
    fitted on the development split (see scripts/fit_vtt_closure.py).  Motion-stall thresholds come from the
    demo-calibrated file when present (per skill class), else the legacy VTT-only stall applies."""
    prog = json.loads(Path(program_path).read_text())
    motion = {}
    motion_window = 300
    stall_cal = {}
    stall_file = ROOT / "configs/vtt_stall_calibration.json"
    if stall_file.exists():   # demo-calibrated plateau clock (outputs/validation/vtt_closure_demo_audit_20260923)
        stall_cal = {k: int(v) for k, v in json.loads(stall_file.read_text()).get("per_class_stall_steps", {}).items()}
    if motion_calibration is not None and Path(motion_calibration).exists():
        cal = json.loads(Path(motion_calibration).read_text())
        motion_window = int(cal.get("window_steps", 300))
        motion = {k: float(v["eps"]) for k, v in cal.get("per_skill", {}).items()}
    median = {}
    for row in prog["skills"]:
        median.setdefault(row["skill"], []).append(row["median_frames"])
    classes = prog["skill_classes"]
    provisional = {  # (done_progress, done_alignment): articulated skills close strongly, navigation does not
        "open lid": (0.7, 0.6), "close lid": (0.6, 0.5), "open door": (0.5, 0.45),
        "pick up from": (0.35, 0.25), "place in": (0.35, 0.25), "push to": (0.3, 0.2), "move to": (1e9, 1e9),
        "place on": (0.35, 0.25),   # Task 1 (picking_up_trash) class; same provisional values as place in
        "place on next to": (0.35, 0.25),  # Task90; provisional signal only, not observed placement success
        "press": (0.35, 0.25)}      # Task 0 (turning_on_radio); placeholder like pick up from (completion = physical predicates)
    out = {}
    for name, cid in classes.items():
        frames = int(np.median(median.get(name, [900])))
        dp, da = provisional[name]
        out[cid] = ClosureConfig(done_progress=dp, done_alignment=da,
                                 stall_steps=min(stall_cal.get(name, max(60, frames // 3)), int(frames * budget_factor)),
                                 min_steps=max(30, frames // 6), max_steps=int(frames * budget_factor),
                                 motion_window=motion_window, motion_eps=motion.get(name))
    return out


# --------------------------------------------------------------------------------------------------
# Next-skill precondition predicate (navigation completion). Built by scripts/precompute_skill_precond.py
# from the same frozen encoder: q_key = mean over TRAIN segments of CLS(start frame) of every program step
# with key "skill(obj1,obj2)". "move to X" is finished when the current CLS is nearest to q of the NEXT
# manipulation step and its cosine exceeds that key's tau (fitted on validation navigation segments).
# --------------------------------------------------------------------------------------------------
import re

_OBJ_INSTANCE = re.compile(r"_\d+$")
_OBJ_MODEL = re.compile(r"_[a-z]{6}$")


def canonical_object(name):
    """'car_ssxsje_0' -> 'car', 'toy_box_81' -> 'toy_box', 'digital_camera_79' -> 'digital' (same short names
    as configs/task013_skill_program.json); Task 1: 'can_of_soda_114' -> 'can_of_soda', 'trash_can_116' ->
    'trash_can', 'floors_ulujpr_0' -> 'floors'."""
    return _OBJ_MODEL.sub("", _OBJ_INSTANCE.sub("", str(name)))


def precondition_key(skill, objects):
    """Key of a program step: 'pick up from(toy_box,floors)'. `objects` may be nested lists (annotation form)."""
    flat = []
    for o in objects:
        flat.extend(o if isinstance(o, (list, tuple)) else [o])
    return f"{skill}({','.join(canonical_object(o) for o in flat)})"


class PreconditionMonitor:
    """arrived(cls_now, next_key) -> (verdict, cos_next, margin_over_best_other).

    verdict True  : cos(cls_now, q_next) >= tau[next_key] and q_next is the top-1 prototype;
            False : not (yet) arrived;
            None  : next_key has no prototype or no usable tau (predicate not applicable).
    Scores are cosines after subtracting the global mean start-CLS if the meta says so (`centred`).
    """

    def __init__(self, vtt_dir=VTT_DIR, proto_file="precond_prototypes.npy", meta_file="precond_meta.json"):
        vtt_dir = Path(vtt_dir)
        meta = json.loads((vtt_dir / meta_file).read_text())
        self.meta = meta
        self.keys = list(meta["keys"])
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.tau = {k: (None if v is None else float(v)) for k, v in meta["tau"].items()}
        proto = np.load(vtt_dir / proto_file).astype(np.float64)
        if proto.shape[0] != len(self.keys):
            raise ValueError(f"{proto_file} has {proto.shape[0]} rows, meta lists {len(self.keys)} keys")
        self.mean = np.asarray(meta["global_mean"], dtype=np.float64) if meta.get("centred", False) else np.zeros(proto.shape[1])
        centred = proto - self.mean
        self.unit = centred / (np.linalg.norm(centred, axis=1, keepdims=True) + 1e-8)

    def scores(self, cls_now):
        x = np.asarray(cls_now, dtype=np.float64) - self.mean
        return self.unit @ x / (np.linalg.norm(x) + 1e-8)

    def arrived(self, cls_now, next_key):
        if next_key not in self.index:
            return None, float("nan"), float("nan")
        sims = self.scores(cls_now)
        i = self.index[next_key]
        cos_next = float(sims[i])
        others = np.delete(sims, i)
        margin = float(cos_next - others.max()) if others.size else float("inf")
        tau = self.tau.get(next_key)
        if tau is None:
            return None, cos_next, margin
        return bool(cos_next >= tau and margin > 0.0), cos_next, margin


class PatchArrivalGate:
    """Generic arrival sub-gate from DINOv3 last-layer PATCH tokens (built by scripts/build_patch_arrival_prototypes.py).

    score(patch_tokens, key) = mean over the 14x14 grid of cosine(current unit patch, per-position unit prototype of the
    key's TRAIN demo start frames); spatially aligned, so it is sensitive to range and heading, which the CLS cosine is
    not (outputs/validation/dock_patch_gate_20260923/).  threshold(key) is the validation-recall threshold, None when the
    key has no usable prototype (gate not applicable).  Colour-free and task-agnostic: one prototype per manipulation key.
    """

    def __init__(self, vtt_dir=VTT_DIR, proto_file="precond_patch_prototypes.npz", meta_file="precond_patch_meta.json"):
        vtt_dir = Path(vtt_dir)
        z = np.load(vtt_dir / proto_file)
        self.meta = json.loads((vtt_dir / meta_file).read_text())
        self.keys = [str(k) for k in z["keys"]]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.proto = z["proto"].astype(np.float32)          # (K, P, D) unit vectors per position
        thr = np.asarray(z["thr"], np.float64)
        self.thr = {k: (None if not np.isfinite(thr[i]) else float(thr[i])) for i, k in enumerate(self.keys)}
        self._torch_proto = {}
        # score mode: "mean_cos" (arrival prototypes) or "contrast" (postcondition: end-minus-start direction per
        # grid position, weighted projection; see scripts/build_patch_postcondition_prototypes.py)
        self.mode = str(z["mode"]) if "mode" in z.files else "mean_cos"
        if self.mode == "contrast":
            self.direction = z["direction"].astype(np.float32)          # (K, P, D)
            self.dir_norm_sum = np.asarray(z["dir_norm_sum"], np.float64)
        elif self.mode != "mean_cos":
            raise ValueError(f"unknown patch gate score mode {self.mode!r}")

    def threshold(self, key):
        return self.thr.get(key)

    def score(self, patch_tokens, key):
        """patch_tokens: (P, D) numpy or torch tensor (any dtype/device). Returns float score or nan if key unknown."""
        if key not in self.index:
            return float("nan")
        i = self.index[key]
        ref = self.direction[i] if self.mode == "contrast" else self.proto[i]
        try:
            import torch
            if isinstance(patch_tokens, torch.Tensor):
                dev = patch_tokens.device
                if (i, dev) not in self._torch_proto:
                    self._torch_proto[(i, dev)] = torch.from_numpy(ref).to(dev, torch.float32)
                p = patch_tokens.float()
                p = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
                if self.mode == "contrast":
                    return float((p * self._torch_proto[(i, dev)]).sum().item() / (self.dir_norm_sum[i] + 1e-8))
                return float((p * self._torch_proto[(i, dev)]).sum(-1).mean().item())
        except ImportError:
            pass
        p = np.asarray(patch_tokens, np.float32)
        p = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
        if self.mode == "contrast":
            return float((p * ref).sum() / (self.dir_norm_sum[i] + 1e-8))
        return float((p * ref).sum(-1).mean())

    def arrived(self, patch_tokens, key):
        """(verdict True/False/None, score); None when the key has no usable threshold."""
        t = self.threshold(key)
        s = self.score(patch_tokens, key)
        if t is None or not np.isfinite(s):
            return None, s
        return bool(s >= t), s


class PatchPostconditionGate(PatchArrivalGate):
    """Generic policy-visible POSTCONDITION gate (built by scripts/build_patch_postcondition_prototypes.py): the same
    spatially aligned patch-token score, against the per-key prototype of the demonstrated END frames of the
    manipulation segment.  threshold(key) = max(validation-recall threshold, validation false-accept threshold on
    unfinished frames), so a missing/low score never advances a skill (fails closed).  It is independent of the VTT
    closure signal (end-appearance prototype vs transition direction) but still policy-visible; evaluator-side object
    joint logs remain the only physical evidence and are used for post-hoc audit, never as policy input.
    """

    def __init__(self, vtt_dir=VTT_DIR, proto_file="postcond_patch_prototypes.npz", meta_file="postcond_patch_meta.json"):
        super().__init__(vtt_dir, proto_file=proto_file, meta_file=meta_file)

    def satisfied(self, patch_tokens, key):
        """(verdict True/False/None, score); None when the key has no calibrated threshold."""
        return self.arrived(patch_tokens, key)


def _selftest_precondition():
    """CPU self-test on synthetic vectors written to a temporary directory (no data files needed)."""
    import tempfile
    rng = np.random.default_rng(0)
    d, keys = 32, ["open lid(car)", "pick up from(toy_box,floors)", "push to(door,robot)"]
    mean = rng.normal(size=d)
    proto = mean + np.eye(d)[:3] * 3.0 + rng.normal(scale=0.05, size=(3, d))
    with tempfile.TemporaryDirectory() as tmp:
        np.save(Path(tmp) / "precond_prototypes.npy", proto.astype(np.float32))
        (Path(tmp) / "precond_meta.json").write_text(json.dumps({
            "keys": keys, "tau": {keys[0]: 0.5, keys[1]: 0.5, keys[2]: None}, "centred": True, "global_mean": mean.tolist()}))
        mon = PreconditionMonitor(tmp)
        v, c, m = mon.arrived(proto[0], keys[0])
        assert v is True and c > 0.9 and m > 0.5, (v, c, m)
        v, c, m = mon.arrived(proto[1], keys[0])                       # wrong target: not top-1
        assert v is False and m < 0, (v, c, m)
        v, c, m = mon.arrived(mean + 0.3 * (proto[0] - mean) + rng.normal(scale=0.4, size=d), keys[0])
        assert v in (True, False) and np.isfinite(c) and np.isfinite(m)
        v, c, m = mon.arrived(proto[2], keys[2])                       # no tau -> not applicable
        assert v is None and c > 0.9
        v, c, m = mon.arrived(proto[0], "place in(unknown,key)")      # unknown key -> not applicable
        assert v is None and np.isnan(c)
        assert precondition_key("pick up from", [["toy_box_81", "floors_ulujpr_0"]]) == keys[1]
        assert precondition_key("place in", ["digital_camera_79", "car_ssxsje_0"]) == "place in(digital,car)"
        # Task 1 (picking_up_trash) contract keys
        assert precondition_key("pick up from", [["can_of_soda_114", "floors_ulujpr_0"]]) == "pick up from(can_of_soda,floors)"
        assert precondition_key("pick up from", [["trash_can_116", "floors_zqjkvm_0"]]) == "pick up from(trash_can,floors)"
        assert precondition_key("place in", [["can_of_soda_115", "trash_can_116"]]) == "place in(can_of_soda,trash_can)"
        assert precondition_key("place on", [["trash_can_116", "floors_ulujpr_0"]]) == "place on(trash_can,floors)"
    # existing closure monitor must be unaffected
    m = VTTClosureMonitor.__new__(VTTClosureMonitor)
    m.proto = np.eye(3, dtype=np.float64) * 2; m.norm = np.linalg.norm(m.proto, axis=1); m.unit = m.proto / m.norm[:, None]
    m.start(0, np.zeros(3))
    p, a = m.signals(np.array([2.0, 0.0, 0.0]))
    assert abs(p - 1.0) < 1e-6 and abs(a - 1.0) < 1e-6
    return True


if __name__ == "__main__":
    _selftest_precondition()
    real = VTT_DIR / "precond_meta.json"
    if real.exists():
        mon = PreconditionMonitor()
        usable = sorted(k for k, t in mon.tau.items() if t is not None)
        print(json.dumps({"precond_keys": len(mon.keys), "usable_tau": usable, "centred": bool(mon.meta.get("centred"))}))
    print("vtt_monitor self-test passed")
