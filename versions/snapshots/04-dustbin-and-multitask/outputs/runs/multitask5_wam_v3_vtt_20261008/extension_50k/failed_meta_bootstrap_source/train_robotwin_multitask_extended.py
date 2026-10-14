#!/usr/bin/env python3
"""Safe TOTAL-50k extension of a frozen 20k shared multi-task run.

Only config.training.{max_steps,schedule_steps} change in the inherited
contract. A budget_extension section records the parent and the *piecewise*
schedule: retain its completed-step LR, then cosine to its original lr_min at
global 50000. This is NOT a fresh 50k cosine, restart, or per-task budget.

plan reads JSON/stat metadata only (not torch.load, cache tensors or CUDA).
train requires a receipt-verified checkpoint_last.pt at a completed VAL/save
boundary. --stop-at-step is an absolute global step, e.g. 5002 for a separate
two-update smoke with the SAME batch/accumulation/loss. --resume resumes only
this new output directory. No old source/config/output is written.

The frozen trainer's execute function is bound to a private globals dictionary
with two explicit hooks (contract and LR factory). Its original function and
module globals remain untouched; loss/loop/VAL/checkpoint/guard stay identical.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import types

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import train_robotwin_multitask_wam as old

FORMAT = old.FORMAT
EXTENSION_FORMAT = "robotwin-multitask-total-budget-extension-v1"
TOTAL_STEPS = 50000
OLD_TARGET = 20000
BUDGET_KEYS = ("max_steps", "schedule_steps")


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def extended_config(parent, total_steps=TOTAL_STEPS):
    if type(total_steps) is not int or total_steps != TOTAL_STEPS:
        raise ValueError("only TOTAL 50000 (not per-task) is authorized")
    if parent.get("format") != FORMAT or "budget_extension" in parent:
        raise ValueError("parent must be an original multi-task native14 contract")
    cfg = old.effective_config(parent["config"])
    if cfg != parent["config"]:
        raise ValueError("parent config must already be effective; no implicit new defaults")
    if any(cfg["training"][key] != OLD_TARGET for key in BUDGET_KEYS):
        raise ValueError("parent must have the original TOTAL 20000 budget")
    if parent["execution"]["mode"] != "train" or len(parent["tasks"]) != 5:
        raise ValueError("parent must be the shared five-task TRAIN run, not a smoke")
    t, e = cfg["training"], parent["execution"]
    if e["batch_size"] != t["batch_size"] or e["accum"] != t["grad_accum_steps"] or e["num_workers"] != 0:
        raise ValueError("parent sampler/batch contract inconsistent")
    if parent.get("heldout_used") is not False or parent.get("encoder_loaded_for_training") is not False:
        raise ValueError("TRAIN-only cached/frozen DINO contract required")
    inf = parent["inference"]
    if inf.get("record_video") is not False or inf.get("evaluation_only_third_view_camera") is not False:
        raise ValueError("recording and evaluation-only third camera must remain disabled")
    for key in BUDGET_KEYS:
        cfg["training"][key] = total_steps
    return old.effective_config(cfg)


def inspect_parent(path, total_steps=TOTAL_STEPS):
    """Small-file inspection only; a positive result is NOT tensor verification."""
    path = old.mt.resolve(path)
    if path.name != "checkpoint_last.pt":
        raise ValueError("parent must be safe-resume checkpoint_last.pt, not model-only")
    manifest = read_json(path.parent / "manifest.json")
    contract = manifest["contract"]
    if manifest.get("contract_sha256") != old.digest(contract):
        raise ValueError("parent manifest contract digest mismatch")
    cfg = extended_config(contract, total_steps)
    receipt = read_json(path.with_name(path.name + ".sha256.json"))
    if (receipt.get("model_only") is not False or type(receipt.get("step")) is not int
            or not contract["config"]["training"]["warmup_steps"] <= receipt["step"] < OLD_TARGET
            or receipt["step"] <= 0 or receipt.get("size") != path.stat().st_size
            or len(receipt.get("sha256", "")) != 64):
        raise ValueError("invalid parent checkpoint metadata/safe boundary")
    artifacts = {name: read_json(path.parent / name) for name in
                 ("statistics.json", "condition_bank.json", "validation_plan.json")}
    if artifacts["statistics.json"] != contract["statistics"] or artifacts["condition_bank.json"] != contract["condition_bank"]:
        raise ValueError("parent TRAIN bank/statistics differ from embedded contract")
    plan = artifacts["validation_plan.json"]
    tasks = {r["task"] for r in contract["tasks"]}
    if (plan.get("selection") != "none" or plan.get("heldout_used") is not False
            or plan.get("split") != "val" or set(plan.get("tasks", {})) != tasks):
        raise ValueError("parent fixed validation plan is invalid")
    validation_path = path.parent / f"validation_step_{receipt['step']:06d}.json"
    validation = read_json(validation_path) if validation_path.is_file() else None
    if validation is not None and (validation.get("selection") != "none"
            or validation.get("physical_evaluation") is not False
            or set(validation.get("per_task", {})) != tasks
            or not math.isfinite(validation.get("macro_normalized_mse", float("nan")))):
        raise ValueError("invalid parent diagnostic VAL result")
    on_boundary = receipt["step"] % contract["config"]["training"]["validate_every"] == 0
    return {"path": str(path), "receipt": receipt, "parent_contract": contract,
            "config": cfg, "artifacts": artifacts, "validation": validation,
            "validation_path": str(validation_path), "ready": validation is not None and on_boundary}


def verify_sources(contract):
    for name, expected in contract["source_hashes"].items():
        if old.sha256(name) != expected:
            raise ValueError(f"frozen parent source changed: {name}")


def validate_resume_state(ck):
    """Validate state geometry/counts without loading an encoder or initializing CUDA."""
    contract, step = ck["contract"], ck["step"]
    state = ck.get("resume_state", {})
    if (ck.get("format") != FORMAT or ck.get("model_only") is not False
            or type(step) is not int or step < 1
            or ck.get("contract_sha256") != old.digest(contract)
            or state.get("safe_optimizer_boundary") is not True
            or not {"optimizer", "scheduler", "rng", "cursor"} <= state.keys()):
        raise ValueError("complete safe-boundary model/Adam/RNG/sampler state required")
    sd = ck.get("model_state_dict")
    if not isinstance(sd, dict) or not sd or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        raise ValueError("missing model tensor state")
    opt, sched, rng, cursor = (state[k] for k in ("optimizer", "scheduler", "rng", "cursor"))
    groups = opt.get("param_groups", [])
    # The frozen factory makes exactly one AdamW group, including frozen params.
    if len(groups) != 1 or not opt.get("state"):
        raise ValueError("complete original AdamW group/moments required")
    group = groups[0]
    ids = group["params"]
    if len(set(ids)) != len(ids) or not set(opt["state"]) <= set(ids):
        raise ValueError("invalid Adam parameter IDs")
    t, execution = contract["config"]["training"], contract["execution"]
    if (tuple(group["betas"]) != tuple(t["betas"]) or group["weight_decay"] != t["weight_decay"]
            or group.get("initial_lr") != t["learning_rate"] or group.get("amsgrad") is not False):
        raise ValueError("Adam recipe drift")
    lrs = [g["lr"] for g in groups]
    if (sched.get("last_epoch") != step or sched.get("_step_count") != step + 1
            or sched.get("_last_lr") != lrs or sched.get("base_lrs") != [t["learning_rate"]]
            or not all(math.isfinite(lr) and t["lr_min"] <= lr <= t["learning_rate"] for lr in lrs)):
        raise ValueError("scheduler/global-step/LR state mismatch")
    for moments in opt["state"].values():
        if not {"step", "exp_avg", "exp_avg_sq"} <= moments.keys():
            raise ValueError("Adam moments missing")
        a, b, n = (moments[k] for k in ("exp_avg", "exp_avg_sq", "step"))
        if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                or a.shape != b.shape or a.dtype != b.dtype
                or float(n) != step):
            raise ValueError("Adam moment geometry/step mismatch")
    if (type(cursor.get("epoch")) is not int or type(cursor.get("batch_skip")) is not int
            or cursor["epoch"] < 1 or not 0 <= cursor["batch_skip"] <= t["sampler_epoch_batches"]
            or (cursor["epoch"] - 1) * t["sampler_epoch_batches"] + cursor["batch_skip"] != step * execution["accum"]):
        raise ValueError("sampler cursor does not match consumed optimizer microbatches")
    counts = ck.get("task_sample_counts", {})
    tasks = {r["task"] for r in contract["tasks"]}
    if (not set(counts) <= tasks or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values()) != step * execution["batch_size"] * execution["accum"]):
        raise ValueError("task sample counts do not match global step")
    if not {"python", "numpy", "torch_cpu", "torch_cuda"} <= rng.keys():
        raise ValueError("incomplete RNG state")
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    cpu_rng = rng["torch_cpu"]
    if not isinstance(cpu_rng, torch.Tensor) or cpu_rng.dtype != torch.uint8 or cpu_rng.ndim != 1:
        raise ValueError("invalid torch CPU RNG")
    torch.Generator(device="cpu").set_state(cpu_rng.cpu())
    cuda_rng = rng["torch_cuda"]
    if execution["device"] == "cuda":
        if (not isinstance(cuda_rng, list) or not cuda_rng or any(not isinstance(v, torch.Tensor)
                or v.dtype != torch.uint8 or v.ndim != 1 or not v.numel() for v in cuda_rng)):
            raise ValueError("CUDA parent missing exact CUDA RNG state")
    elif cuda_rng is not None:
        raise ValueError("CPU parent unexpectedly has CUDA RNG")
    return lrs


def make_contract(meta, start_lrs, run_role="train"):
    if run_role not in ("train", "smoke"):
        raise ValueError("invalid extension run role")
    parent = meta["parent_contract"]
    contract = copy.deepcopy(parent)
    contract["config"] = copy.deepcopy(meta["config"])
    r = meta["receipt"]
    contract["budget_extension"] = {
        "format": EXTENSION_FORMAT, "run_role": run_role,
        "evaluation_admissible_at_final_budget_only": run_role == "train",
        "parent_checkpoint": {"path": meta["path"], **r},
        "parent_contract_sha256": old.digest(parent),
        "parent_contract": copy.deepcopy(parent),
        "old_total_steps": OLD_TARGET, "new_total_steps": TOTAL_STEPS,
        "changed_inherited_fields": [f"config.training.{key}" for key in BUDGET_KEYS],
        "schedule": {"kind": "continuous_cosine_from_saved_lr_no_rewarmup",
                     "start_step": r["step"], "start_lrs": list(start_lrs),
                     "end_step": TOTAL_STEPS, "lr_min": parent["config"]["training"]["lr_min"],
                     "step_meaning": "completed global optimizer updates; saved LR is used by the NEXT update",
                     "not_a_fresh_50k_cosine": True},
        "validation_plan": copy.deepcopy(meta["artifacts"]["validation_plan.json"]),
        "parent_validation": {"path": meta["validation_path"], "sha256": old.digest(meta["validation"])},
        "source_hashes": {str(Path(__file__).resolve()): old.sha256(__file__)},
        "inherited_without_refit": ["weights", "Adam_moments", "RNG", "sampler_cursor",
                                   "task_sample_counts", "TRAIN_condition_bank", "TRAIN_statistics"],
        "checkpoint_selection": "fixed_final_global_50000_only_not_VAL_or_test_selected"}
    assert_whitelist(contract, parent)
    return contract


def assert_whitelist(contract, parent):
    stripped = copy.deepcopy(contract)
    stripped.pop("budget_extension", None)
    for key in BUDGET_KEYS:
        if stripped["config"]["training"][key] != TOTAL_STEPS:
            raise ValueError("extended budget must be TOTAL 50000")
        stripped["config"]["training"][key] = parent["config"]["training"][key]
    if stripped != parent:
        raise ValueError("inherited contract changed outside the two budget fields")


def continuation_lrs(step, schedule):
    if type(step) is not int or not 0 <= step <= schedule["end_step"]:
        raise ValueError("LR step outside global budget")
    if step <= schedule["start_step"]:
        # LambdaLR construction runs at zero BEFORE the saved state is restored.
        return list(schedule["start_lrs"])
    progress = (step - schedule["start_step"]) / (schedule["end_step"] - schedule["start_step"])
    return [schedule["lr_min"] + (lr - schedule["lr_min"]) * .5 * (1 + math.cos(math.pi * progress))
            for lr in schedule["start_lrs"]]


def optimizer_factory(contract):
    schedule = contract["budget_extension"]["schedule"]
    def factory(model, cfg):
        if cfg != contract["config"]["training"]:
            raise ValueError("training config changed at optimizer construction")
        optimizer, scheduler = old.make_optimizer(model, cfg)
        # Keep the exact original AdamW factory and scheduler state schema. The
        # saved last_epoch/_step_count/LR are loaded by frozen execute afterward.
        scheduler.lr_lambdas = [lambda step: continuation_lrs(step, schedule)[0] / cfg["learning_rate"]]
        return optimizer, scheduler
    return factory


def verify_optimizer_geometry(ck):
    """Meta-device construction checks parameter order/shapes without real weights."""
    from omegaconf import OmegaConf
    cfg = ck["contract"]["config"]
    with torch.device("meta"):
        model = old.mt.create_v3_action_model(OmegaConf.create(cfg), dino_hidden_size=1024,
            num_dino_layers=3, task_cond_dim=1024, patch_size=16)
        old.mt.install_joint_projection(model)
    expected = model.state_dict()
    actual = ck["model_state_dict"]
    if expected.keys() != actual.keys() or any(
            expected[k].shape != actual[k].shape or expected[k].dtype != actual[k].dtype for k in expected):
        raise ValueError("model tensor topology differs from inherited architecture")
    groups, states = ck["resume_state"]["optimizer"]["param_groups"], ck["resume_state"]["optimizer"]["state"]
    params = list(model.named_parameters())
    if groups[0]["params"] != list(range(len(params))):
        raise ValueError("Adam parameter topology mismatch")
    required = {"action_proj.weight", "task_cond_proj.residual.weight"}
    for (name, parameter), pid in zip(params, groups[0]["params"]):
        moments = states.get(pid)
        if name in required and moments is None:
            raise ValueError(f"missing trained joint Adam moments: {name}")
        if moments is not None and (not parameter.requires_grad or any(
                moments[k].shape != parameter.shape or moments[k].dtype != parameter.dtype
                for k in ("exp_avg", "exp_avg_sq"))):
            raise ValueError(f"Adam moment/parameter shape or frozen state mismatch: {name}")


def write_seed_checkpoint(path, ck, contract):
    """One new checkpoint, with state objects unchanged; never write parent path."""
    payload = {**ck, "contract": contract, "contract_sha256": old.digest(contract)}
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    old.atomic_json(path.with_name(path.name + ".sha256.json"),
                    {"sha256": old.sha256(path), "size": path.stat().st_size,
                     "step": ck["step"], "model_only": False})


def delegated_execute(contract, args, out, datasets, source):
    def immutable(cfg, runtime_args, specs, live_datasets, init, statistics, bank):
        live = old.immutable_contract(cfg, runtime_args, specs, live_datasets, init, statistics, bank)
        inherited = {k: v for k, v in contract.items() if k != "budget_extension"}
        if live != inherited:
            raise ValueError("live data/source/config/runtime/recipe differs from inherited contract")
        return copy.deepcopy(contract)
    namespace = dict(old.execute.__globals__)
    namespace.update(immutable_contract=immutable, make_optimizer=optimizer_factory(contract))
    execute = types.FunctionType(old.execute.__code__, namespace, old.execute.__name__,
                                 old.execute.__defaults__, old.execute.__closure__)
    return execute(contract["config"], args, out, contract["tasks"], datasets, source)


def runtime_args(args, contract):
    execution = contract["execution"]
    if args.device is not None and args.device != execution["device"]:
        raise ValueError("device change would break exact RNG continuation")
    if execution["device"] == "cuda" and not args.gpu_lock_owner:
        raise ValueError("CUDA requires externally held --gpu-lock-owner (no service/lock takeover)")
    return argparse.Namespace(mode="train", resume=True, device=execution["device"],
        max_steps=args.stop_at_step, gpu_lock_owner=args.gpu_lock_owner,
        max_mapped_episodes=execution["max_mapped_episodes_per_task_split"],
        cuda_memory_fraction=execution["cuda_memory_fraction"], smoke_steps=2, smoke_batch_size=1)


def run_train(args, meta):
    if not meta["ready"]:
        raise ValueError("wait for parent fixed-VAL boundary and validation_step file before handoff")
    out = old.mt.resolve(args.out)
    if out == Path(meta["path"]).parent or out in Path(meta["path"]).parents:
        raise ValueError("new output cannot be the parent run or an ancestor")
    if out.exists() and not args.resume:
        raise FileExistsError(f"fresh extension/smoke requires a NEW directory: {out}")
    role = args.run_role or ("smoke" if out.name == "cuda_smoke" else "train")
    if role == "smoke" and (args.resume or args.stop_at_step != meta["receipt"]["step"] + 2):
        raise ValueError("smoke must be an independent two-update branch; never resume/admit it")
    verify_sources(meta["parent_contract"])
    # Hash/load only under explicit train. Metadata plan never accesses tensors.
    parent_before = old.file_identity(meta["path"])
    if args.resume:
        manifest = read_json(out / "manifest.json")
        contract = manifest["contract"]
        extension = contract.get("budget_extension", {})
        if extension.get("run_role") == "smoke":
            raise ValueError("smoke artifacts are never admissible for continuation")
        expected = make_contract(meta, extension["schedule"]["start_lrs"], role)
        if contract != expected or manifest.get("contract_sha256") != old.digest(contract):
            raise ValueError("extension contract/source/parent identity changed on resume")
        if old.sha256(meta["path"]) != meta["receipt"]["sha256"]:
            raise ValueError("parent identity changed on extension resume")
        checkpoint_path = out / "checkpoint_last.pt"
        ck = old.load_checkpoint(checkpoint_path, contract)
    else:
        checkpoint_path = Path(meta["path"])
        ck = old.load_checkpoint(checkpoint_path, meta["parent_contract"])
        contract = make_contract(meta, validate_resume_state(ck), role)
    if (parent_before != old.file_identity(meta["path"])
            or read_json(Path(meta["path"] + ".sha256.json")) != meta["receipt"]):
        raise ValueError("parent changed during handoff; stop its writer and retry into a new directory")
    lrs = validate_resume_state(ck)
    if ck["step"] < meta["receipt"]["step"] or (not args.resume and ck["step"] != meta["receipt"]["step"]):
        raise ValueError("checkpoint global step inconsistent with parent")
    expected_lrs = continuation_lrs(ck["step"], contract["budget_extension"]["schedule"])
    if any(not math.isclose(a, b, rel_tol=1e-14, abs_tol=0.) for a, b in zip(lrs, expected_lrs)):
        raise ValueError("resume LR differs from its registered continuation schedule")
    target = args.stop_at_step or TOTAL_STEPS
    if not ck["step"] < target <= TOTAL_STEPS:
        raise ValueError("stop target must exceed saved global step and be <= TOTAL 50000")
    rt = runtime_args(args, contract)
    verify_optimizer_geometry(ck)
    datasets = None
    try:
        datasets, source = old.preflight(contract["config"], contract["tasks"], rt.max_mapped_episodes)
        if not args.resume:
            out.mkdir(parents=True, exist_ok=False)
            for name, value in meta["artifacts"].items():
                old.atomic_json(out / name, value)
            old.atomic_json(out / "manifest.json", {"contract": contract, "contract_sha256": old.digest(contract)})
            write_seed_checkpoint(out / "checkpoint_last.pt", ck, contract)
            old.atomic_json(out / "handoff_receipt.json", {
                "format": EXTENSION_FORMAT, "parent_checkpoint": contract["budget_extension"]["parent_checkpoint"],
                "parent_contract_sha256": old.digest(meta["parent_contract"]),
                "contract_sha256": old.digest(contract), "global_step": ck["step"],
                "saved_next_update_lrs": lrs, "sampler_cursor": ck["resume_state"]["cursor"],
                "task_sample_counts": ck["task_sample_counts"], "fresh_optimizer": False,
                "refitted_bank_or_statistics": False, "encoder_loaded": False,
                "state_transfer": "same tensor/state objects serialized; only contract and its digest changed",
                "physical_evaluation": False, "not_a_fresh_50k_cosine": True})
        smoke_before = None
        if role == "smoke":
            smoke_before = {name: value.detach().cpu().clone()
                            for name, value in ck["model_state_dict"].items()
                            if name in ("action_proj.weight", "task_cond_proj.residual.weight")
                            or name.startswith("task_cond_proj.base.")}
        del ck
        result = delegated_execute(contract, rt, out, datasets, source)
        if role == "smoke":
            finished = old.load_checkpoint(out / "checkpoint_last.pt", contract)
            validate_resume_state(finished)
            changed = {name: not torch.equal(smoke_before[name], finished["model_state_dict"][name])
                       for name in ("action_proj.weight", "task_cond_proj.residual.weight")}
            frozen = all(torch.equal(value, finished["model_state_dict"][name])
                         for name, value in smoke_before.items() if name.startswith("task_cond_proj.base."))
            if finished["step"] != meta["receipt"]["step"] + 2 or not all(changed.values()) or not frozen:
                raise ValueError(f"two-update smoke failed actual joint changes/frozen base: {changed}, {frozen}")
            evidence = {"joint_weights_changed": changed, "frozen_base_unchanged": frozen,
                        "start_step": meta["receipt"]["step"], "end_step": finished["step"],
                        "evaluation_admissible": False, "resume_admissible": False,
                        "formal_run_must_start_again_from_original_parent": True}
            old.atomic_json(out / "smoke_receipt.json", evidence)
            result = {**result, "status": "smoke_passed", "smoke_verification": evidence}
            old.atomic_json(out / "status.json", result)
        return result
    finally:
        if datasets:
            for rows in datasets.values():
                for ds in rows.values():
                    ds.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--out", help="NEW run directory; --resume uses that same new run")
    p.add_argument("--total-steps", type=int, choices=(TOTAL_STEPS,), default=TOTAL_STEPS)
    p.add_argument("--mode", choices=("plan", "train"), default="plan")
    p.add_argument("--stop-at-step", type=int, help="absolute global stop, e.g. 5002; schedule stays TOTAL50k")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-role", choices=("train", "smoke"),
                   help="default train, except an output named cuda_smoke; smoke is permanently non-admissible")
    p.add_argument("--device", choices=("cpu", "cuda"), help="optional assertion; cannot change parent device")
    p.add_argument("--gpu-lock-owner", help="external pipeline lock; never acquired or released here")
    args = p.parse_args(argv)
    if args.mode == "train" and not args.out:
        p.error("--out required for train")
    if args.resume and (args.mode != "train" or not args.out):
        p.error("--resume requires --mode train --out NEW_RUN")
    if args.stop_at_step is not None and not 0 < args.stop_at_step <= args.total_steps:
        p.error("--stop-at-step must be in [1, TOTAL50000]")
    return args


def main(argv=None):
    args = parse_args(argv)
    meta = inspect_parent(args.parent_checkpoint, args.total_steps)
    if args.mode == "plan":
        t, e = meta["parent_contract"]["config"]["training"], meta["parent_contract"]["execution"]
        result = {"status": "metadata_plan_only_not_tensor_verified", "format": FORMAT,
            "parent_checkpoint": {"path": meta["path"], **meta["receipt"]},
            "parent_contract_sha256": old.digest(meta["parent_contract"]),
            "handoff_val_ready_metadata": meta["ready"],
            "old_total_steps": OLD_TARGET, "total_steps": TOTAL_STEPS,
            "start_step": meta["receipt"]["step"], "remaining_updates": TOTAL_STEPS - meta["receipt"]["step"],
            "tasks": [r["task"] for r in meta["parent_contract"]["tasks"]],
            "batch_size": e["batch_size"], "grad_accum_steps": e["accum"],
            "validation_every": t["validate_every"], "validation_samples_per_task": t["validation_samples"],
            "schedule": "saved parent LR -> continuous cosine to original lr_min at global50000; no rewarmup",
            "first_lr": "read from optimizer only during explicit train", "lr_min": t["lr_min"],
            "weights_loaded": False, "encoder_loaded": False, "cuda_initialized_by_plan": False,
            "record_video": False, "evaluation_only_third_view_camera": False,
            "evaluation_checkpoint": "checkpoint_step_050000.pt",
            "writes_performed": False}
    else:
        result = run_train(args, meta)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
