#!/usr/bin/env python3
"""CPU-only commanded-state hold reference on the existing, fixed VAL plan.

Run with .venvs/wam-fixed/bin/python -B scripts/check_robotwin_hold_baseline.py.
--self-test-only runs analytic tests without opening any real episode. All writes
are confined to the assigned baseline directory. No training entrypoint, model,
feature cache, image payload, simulator, TRAIN HDF, or heldout HDF is loaded.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

# Set before importing numerical libraries; never initialize a CUDA context.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "2"
sys.dont_write_bytecode = True

import h5py
import numpy as np
import psutil
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/runs/put_object_cabinet/wam_30k_20261005"
OUT = ROOT / "outputs/validation/robotwin_cabinet_baseline_20261005"
SPLIT = ROOT / "data/benchmarks/robotwin2/put_object_cabinet_500_v1/split_manifest.json"
CONFIG = ROOT / "configs/fixed_bimanual/robotwin_cabinet_500_wam.yaml"
TRAINER = ROOT / "scripts/train_robotwin_wam.py"
WRAPPER = ROOT / "harness/wam_v3.py"
READER = ROOT / "harness/robotwin_dataset.py"
SEMANTICS = "controller_joint_targets_and_gripper_command_not_measured_qpos"
GROUPS = {"left_arm_rad": (0, 6), "left_gripper_units": (6, 7),
          "right_arm_rad": (7, 13), "right_gripper_units": (13, 14)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    # Only called for small code/config/JSON files, never complete HDF files.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def digest(value, *, pretty=False):
    text = json_text(value) if pretty else json.dumps(value, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(name, value):
    require(Path(name).name == name, "output must be an assigned-directory filename")
    temporary = OUT / (name + ".tmp")
    temporary.write_text(json_text(value))
    temporary.replace(OUT / name)


def primitive(path, name, *, class_name=None, namespace=None):
    """Compile an inspected pure function, not its module or model constructors."""
    tree = ast.parse(path.read_text(), filename=str(path))
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    scope = {"torch": torch, "np": np, **(namespace or {})}
    unit = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(unit, str(path), "exec"), scope)
    return scope[name]


# Reuse exact current float32 metric and normalization functions without any
# training-module import side effects or WamV3Wrapper/model instantiation.
action_metrics = primitive(TRAINER, "action_metrics")
validation_indices = primitive(TRAINER, "validation_indices")
_reader_tree = ast.parse(READER.read_text())
JOINT_FIELDS = ast.literal_eval(next(n.value for n in _reader_tree.body
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "JOINT_FIELDS"
                                       for t in n.targets)))
lowdim = primitive(READER, "_lowdim", namespace={"JOINT_FIELDS": JOINT_FIELDS})


class TrainMinMax:
    _minmax = staticmethod(primitive(WRAPPER, "_minmax", class_name="WamV3Wrapper"))
    normalize_action = primitive(WRAPPER, "normalize_action", class_name="WamV3Wrapper")

    def __init__(self, minimum, maximum):
        self.action_dim = 14
        self.action_min = torch.tensor(minimum, dtype=torch.float32, device="cpu")
        self.action_max = torch.tensor(maximum, dtype=torch.float32, device="cpu")
        span = self.action_max - self.action_min
        require(self.action_min.shape == self.action_max.shape == (14,)
                and bool(torch.isfinite(self.action_min).all())
                and bool(torch.isfinite(self.action_max).all())
                and bool(torch.isfinite(span).all()) and bool((span >= 0).all()),
                "invalid TRAIN action min/max")


class ResourceGuard:
    def __init__(self):
        self.peaks = {"system_cpu_percent": 0.0, "system_ram_percent": 0.0,
                      "process_rss_bytes": 0}
        self.process = psutil.Process()
        psutil.cpu_percent(interval=None)
        self.check(interval=0.1)

    def check(self, interval=None):
        values = {"system_cpu_percent": psutil.cpu_percent(interval=interval),
                  "system_ram_percent": psutil.virtual_memory().percent,
                  "process_rss_bytes": self.process.memory_info().rss}
        for key, value in values.items():
            self.peaks[key] = max(value, self.peaks[key])
        require(values["system_cpu_percent"] < 85 and values["system_ram_percent"] < 85,
                f"CPU/RAM85 guard: {values}")


def hold_chunk(state, action, anchor):
    require(state.shape == action.shape and state.ndim == 2 and state.shape[1] == 14,
            "state/action must be matching (T,14) arrays")
    require(type(anchor) is int and 0 <= anchor < len(state), "invalid fixed anchor")
    count = min(32, len(action) - anchor)
    target = np.zeros((1, 32, 14), dtype=np.float32)
    target[0, :count] = action[anchor:anchor + count]  # already aligned; NO extra shift
    prediction = np.repeat(state[anchor][None, None, :], 32, axis=1).copy()
    valid = (np.arange(32) < count)[None, :]
    return torch.from_numpy(prediction), torch.from_numpy(target), torch.from_numpy(valid)


def summarize(metrics, horizon):
    rows = sum(m["valid_action_rows"] for m in metrics)
    require(rows > 0, "empty metric aggregate")
    result = {key: sum(m[key] * m["valid_action_rows"] for m in metrics) / rows
              for key in metrics[0] if key != "valid_action_rows"}
    for group in GROUPS:
        result[group + "_rmse"] = math.sqrt(result.pop(group + "_mse"))
    return {"samples": len(metrics), "valid_action_rows": rows,
            "masked_padding_rows": len(metrics) * horizon - rows,
            "normalized_scalar_denominator": rows * 14,
            "group_scalar_denominators": {name: rows * (hi - lo)
                                           for name, (lo, hi) in GROUPS.items()}, **result}


def self_test():
    normalizer = TrainMinMax([0] * 14, [2] * 14)
    error = torch.tensor([2] * 6 + [4] + [-3] * 6 + [-5], dtype=torch.float32)
    prediction = torch.full((2, 3, 14), float("inf"))
    target = torch.full_like(prediction, float("nan"))
    valid = torch.tensor([[True, False, False], [True, False, False]])
    prediction[0, 0], prediction[1, 0] = error, 0
    target[valid] = 0
    actual = action_metrics(prediction, target, valid, normalizer)
    # One error row plus one zero row: squared sum = 6*4+16+6*9+25 = 119.
    expected = {"normalized_mse": 119 / (2 * 14), "valid_action_rows": 2,
                "left_arm_rad_mae": 1, "left_arm_rad_mse": 2,
                "right_arm_rad_mae": 1.5, "right_arm_rad_mse": 4.5,
                "left_gripper_units_mae": 2, "left_gripper_units_mse": 8,
                "right_gripper_units_mae": 2.5, "right_gripper_units_mse": 12.5}
    for key, value in expected.items():
        require(math.isclose(actual[key], value, rel_tol=1e-6, abs_tol=1e-7), f"analytic metric {key}")
    one = action_metrics(error.reshape(1, 1, 14), torch.zeros(1, 1, 14),
                         torch.ones(1, 1, dtype=torch.bool), normalizer)
    two = action_metrics(torch.zeros(1, 2, 14), torch.zeros(1, 2, 14),
                         torch.ones(1, 2, dtype=torch.bool), normalizer)
    aggregate = summarize([one, two], 3)
    require(math.isclose(aggregate["normalized_mse"], 119 / 42), "row-weighted MSE")
    require(math.isclose(aggregate["left_arm_rad_rmse"], math.sqrt(4 / 3)), "RMSE after aggregation")
    near_constant = TrainMinMax([0] * 14, [0, 5e-7] + [2] * 12)
    normalized = near_constant.normalize_action(torch.full((1, 14), 2.0))
    require(torch.equal(normalized[0, :2], torch.tensor([3., 3.])), "range floor, no clipping")
    require(torch.equal(normalized[0, 2:], torch.ones(12)), "minmax nonconstant range")
    state = np.arange(42, dtype=np.float32).reshape(3, 14)
    action = state + 100
    for anchor in (0, 2):
        p, t, mask = hold_chunk(state, action, anchor)
        n = 3 - anchor
        require(np.array_equal(p[0].numpy(), np.repeat(state[anchor][None], 32, axis=0)), "hold current state")
        require(np.array_equal(t[0, :n].numpy(), action[anchor:]), "no second action shift")
        require(int(mask.sum()) == n and not bool(t[0, n:].any()), "episode-end padding")
        require(action_metrics(p[:, :8], t[:, :8], mask[:, :8], normalizer)["valid_action_rows"] == n,
                "exec8 prefix mask")
    fields, start = {}, 0
    for name, width in JOINT_FIELDS:
        fields["state/" + name] = state[:, start:start + width]
        start += width
    require(np.array_equal(lowdim(fields, "state"), state), "official HDF field concatenation")
    for bad_mask in (torch.zeros_like(valid), valid.float()):
        try:
            action_metrics(prediction, target, bad_mask, normalizer)
        except ValueError:
            pass
        else:
            raise ValueError("invalid masks must be rejected")
    return {"passed": True, "device": "cpu", "threads": 2,
            "checks": ["masked_nan_inf_ignored", "normalized_denominator_valid_rows_times_14",
                       "left_right_arm_gripper_slices", "valid_row_weighted_aggregation",
                       "rmse_after_mse_aggregation", "minmax_small_range_floor_no_clip",
                       "constant_current_state_prediction", "aligned_action_no_extra_shift",
                       "episode_end_padding_and_exec8", "official_lowdim_field_order",
                       "empty_and_nonboolean_masks_rejected"],
            "analytic_expected": expected, "analytic_actual": actual,
            "weighted_aggregation_actual": aggregate}


def evaluate(guard, tests):
    started = time.monotonic()
    manifest_path, norm_path, plan_path = [RUN / n for n in
                                          ("manifest.json", "normalization.json", "validation_plan.json")]
    split = json.loads(SPLIT.read_text())
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["contract"]
    norm = json.loads(norm_path.read_text())
    plan = json.loads(plan_path.read_text())
    cfg = contract["config"]
    source_paths = [Path(__file__).resolve(), TRAINER, WRAPPER, READER,
                    ROOT / "harness/robotwin_training_data.py", ROOT / "scripts/convert_robotwin_legacy.py",
                    CONFIG, SPLIT, manifest_path, norm_path, plan_path]
    hashes = {str(p): sha256(p) for p in source_paths}
    require(digest(contract) == manifest["contract_sha256"], "run contract digest differs")
    require(hashes[str(CONFIG)] == contract["config_sha256"], "frozen config bytes differ")
    require(hashes[str(SPLIT)] == contract["split_sha256"] == cfg["dataset"]["split_sha256"], "split bytes differ")
    require((ROOT / cfg["dataset"]["split_manifest"]).resolve() == SPLIT, "wrong configured split")
    require(hashes[str(norm_path)] == contract["stats"]["normalization"]["sha256"], "TRAIN stats bytes differ")
    for path in (TRAINER, WRAPPER, READER, ROOT / "harness/robotwin_training_data.py"):
        require(hashes[str(path)] == contract["source_hashes"][str(path)], f"frozen metric/data source differs: {path}")
    provenance = norm["provenance"]
    require(norm["action_repr"] == "absolute" and provenance["fit_split"] == "train", "TRAIN absolute stats required")
    require(provenance["episodes"] == contract["episodes"]["train"], "TRAIN identity receipts differ")
    require(provenance["episode_count"] == len(split["splits"]["train"])
            and [e["episode"] for e in provenance["episodes"]] == split["splits"]["train"], "wrong fit members")
    require(provenance["split_manifest_sha256"] == digest(split, pretty=True), "normalization split digest differs")
    require(provenance["cache_config_id"] == contract["cache_config_id"], "normalization cache provenance differs")
    require(cfg["common"]["action_dim"] == cfg["common"]["state_dim"] == 14
            and cfg["common"]["action_chunk_size"] == 32
            and cfg["common"]["action_execution_horizon"] == 8, "wrong native14/chunk32/exec8 contract")
    require(plan["split"] == "val" and plan["seed"] == cfg["training"]["validation_seed"], "wrong validation plan")
    anchors, indices = plan["anchors"], plan["indices"]
    require(len(anchors) == len(indices) == cfg["training"]["validation_samples"] == 512, "need all 512 anchors")
    require(len(set(indices)) == 512 and len({(a["episode"], a["anchor"]) for a in anchors}) == 512,
            "duplicate fixed anchors")
    val_members = split["splits"]["val"]
    require(not (set(val_members) & set(split["splits"]["train"]))
            and not (set(val_members) & set(split["splits"]["heldout"])), "split overlap")
    require(all(a["episode"] in val_members for a in anchors), "non-VAL anchor")
    identities = contract["episodes"]["val"]
    require([e["episode"] for e in identities] == val_members, "run VAL membership/order differs")
    counts = Counter(a["episode"] for a in anchors)
    require(set(counts) == set(val_members), "this plan must cover all 50 VAL episodes")
    normalizer = TrainMinMax(norm["robotwin"]["action"]["min"], norm["robotwin"]["action"]["max"])
    data_root = Path(split["data_root"]).resolve()
    episodes, evidence, offsets, offset = {}, [], {}, 0
    for identity in identities:
        guard.check()
        name = identity["episode"]
        require(Path(name).name == name, "unsafe episode filename")
        path = data_root / name
        require(not path.is_symlink() and path.resolve().parent == data_root, "episode must be a direct VAL file")
        before = path.stat()
        with h5py.File(path, "r") as file:
            semantics = str(file.attrs.get("state_semantics", ""))
            require(semantics == SEMANTICS, "expected commanded targets, not measured qpos")
            # Only these eight low-dimensional datasets and root metadata are read.
            state, action = lowdim(file, "state"), lowdim(file, "action")
        after = path.stat()
        require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns), "HDF changed while reading")
        require(state.shape == action.shape == (identity["length"], 14) and len(state) > 0,
                "run episode length differs")
        array_hashes = {key: hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()
                        for key, value in (("state", state), ("action", action))}
        episodes[name], offsets[name] = (state, action), offset
        offset += len(state)
        evidence.append({"episode": name, "path": str(path), "length": len(state),
                         "anchor_count": counts[name], "state_semantics": semantics,
                         "run_receipt_hdf_sha256_not_rehashed": identity["hdf_sha256"],
                         "fresh_lowdim_float32_sha256": array_hashes,
                         "lowdim_hash_encoding": "C-order little-endian float32 (T,14), state and action separately",
                         "file_stat": {"size": after.st_size, "mtime_ns": after.st_mtime_ns, "inode": after.st_ino}})
    require(indices == validation_indices(offset, plan["seed"], 512), "stored indices differ from official fixed sampling")
    for index, anchor in zip(indices, anchors):
        require(type(anchor["anchor"]) is int and 0 <= anchor["anchor"] < len(episodes[anchor["episode"]][0]),
                "anchor outside episode")
        require(index == offsets[anchor["episode"]] + anchor["anchor"], "fixed anchor/index mapping differs")
    rows = []
    for ordinal, (index, anchor) in enumerate(zip(indices, anchors)):
        if ordinal % 64 == 0:
            guard.check()
        state, action = episodes[anchor["episode"]]
        prediction, target, valid = hold_chunk(state, action, anchor["anchor"])
        row = {"plan_ordinal": ordinal, "dataset_index": index, **anchor}
        for label, horizon in (("chunk32", 32), ("exec8", 8)):
            row[label] = action_metrics(prediction[:, :horizon], target[:, :horizon],
                                        valid[:, :horizon], normalizer)
        rows.append(row)
    metrics = {label: summarize([r[label] for r in rows], horizon)
               for label, horizon in (("chunk32", 32), ("exec8", 8))}
    per_episode = [{"episode": name, "anchor_count": counts[name], **{
        label: summarize([r[label] for r in rows if r["episode"] == name], horizon)
        for label, horizon in (("chunk32", 32), ("exec8", 8))}} for name in val_members]
    # Check all input code/config/JSON bytes and lowdim HDF identities stayed stable.
    require(all(sha256(p) == hashes[str(p)] for p in source_paths), "input source bytes changed during evaluation")
    for item in evidence:
        s = Path(item["path"]).stat()
        require(item["file_stat"] == {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino},
                "VAL HDF identity changed during evaluation")
    guard.check()
    require(not torch.cuda.is_initialized(), "CUDA must never initialize")
    return {"schema_version": 1, "baseline": "hold_current_commanded_state", "split": "val",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_state": SEMANTICS,
            "prediction": "prediction[t:t+32] = repeat(state[t],32); stored action[t:t+32]; no additional shift",
            "metrics": metrics,
            "metric_semantics": {
                "implementation": "exact AST-extracted train_robotwin_wam.action_metrics and WamV3Wrapper.normalize_action/_minmax",
                "precision": "CPU float32 per-anchor metrics, Python float64 row-weighted accumulation as trainer.evaluate",
                "normalization": "TRAIN action min/max: 2*(x-min)/den-1; den=max-min, den<1e-6 replaced by 1; no clipping",
                "normalized_mse": "sum of squared normalized errors / (valid_action_rows*14)",
                "group_slices_zero_based_end_exclusive": GROUPS,
                "rmse": "sqrt(valid-row-weighted mean squared error), not mean of anchor RMSEs",
                "padding": "zero target padding; excluded by action_valid_mask from all metrics and denominators",
                "overlap": "overlapping anchor windows retain their original multiplicity, exactly as trainer.evaluate",
                "exec8": "same fixed anchors, first min(8,T-t) valid targets only; configured execution prefix, not an executed rollout",
                "selection": "none; all original anchors in original order, no outcome-based sampling"},
            "counts": {"fixed_anchors": len(rows), "unique_val_episodes": len(evidence),
                       "validation_dataset_frames": offset, "train_hdf_opened": 0, "heldout_hdf_opened": 0,
                       "val_hdf_opened": len(evidence), "image_datasets_read": 0, "cache_files_read": 0},
            "provenance": {"source_file_sha256": hashes, "run_contract_sha256": manifest["contract_sha256"],
                           "fixed_anchor_list_sha256": digest(anchors), "validation_seed": plan["seed"],
                           "normalization_fit_split": "train", "normalization_fit_episode_count": provenance["episode_count"],
                           "normalization_fit_frame_count": provenance["frame_count"],
                           "run_config": cfg, "baseline_config": {"horizon": 32, "execution_prefix": 8,
                                                                     "state_dim": 14, "action_dim": 14, "threads": 2},
                           "hdf_hash_assurance": "whole-HDF SHA256 copied from immutable run receipt, not reverified; fresh hashes cover exactly loaded float32 lowdim arrays; no image payload read",
                           "val_sources": evidence},
            "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                        "torch": str(torch.__version__), "h5py": h5py.__version__, "device": "cpu",
                        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                        "cuda_initialized": torch.cuda.is_initialized(), "elapsed_seconds": time.monotonic() - started,
                        "sampled_resource_peaks": guard.peaks, "cpu_ram_abort_percent": 85},
            "self_test_passed": tests["passed"],
            "claims": {"training_performed": False, "model_loaded": False, "gpu_used": False,
                       "simulation_performed": False, "physical_evaluation": False,
                       "physical_success_claim": False, "model_comparison_performed": False,
                       "checkpoint_selection": "none", "exec8_model_comparison": "not performed; model exec8 not reported"}}, per_episode, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    guard = ResourceGuard()
    OUT.mkdir(parents=True, exist_ok=True)
    tests = self_test()
    write_json("self_test.json", tests)
    if args.self_test_only:
        print(json_text(tests), end="")
        return
    result, episodes, anchors = evaluate(guard, tests)
    write_json("per_episode.json", episodes)
    temporary = OUT / "per_anchor.jsonl.tmp"
    with temporary.open("w") as file:
        for row in anchors:
            file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(OUT / "per_anchor.jsonl")
    result["artifacts"] = {name: {"path": str(OUT / name), "sha256": sha256(OUT / name)}
                           for name in ("self_test.json", "per_episode.json", "per_anchor.jsonl")}
    write_json("result.json", result)
    print(json_text({"result": str(OUT / "result.json"), "metrics": result["metrics"],
                     "counts": result["counts"], "self_test_passed": True}), end="")


if __name__ == "__main__":
    main()
