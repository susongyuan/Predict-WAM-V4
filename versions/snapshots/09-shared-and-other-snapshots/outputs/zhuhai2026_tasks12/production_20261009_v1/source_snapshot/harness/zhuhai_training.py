"""Native16 Zhuhai contracts and selective initialization, isolated from RoboTwin.

State is measured joint/gripper feedback; action is the next interval's target.
No native14 numerical normalization crosses this boundary. All moments and VTTs
are fitted on successful TRAIN episodes and bound to the robot/data/feature
contracts. Importing this module does not import Torch or start a simulator, so
manifest validation also works in the separate Isaac ABI environment.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "zhuhai-isaac-native16-measured-10hz-v1"
INITIALIZATION_FORMAT = "robotwin-multitask-native14-vtt-v1"
INITIALIZATION = ROOT / "outputs/runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/train/checkpoint_step_050000.pt"
INITIALIZATION_SHA = "8960be6ceb4d5293d2c5455ac4308421f64f1e31bf64f0646b7208735f63a5f4"
FREQUENCY = 10.0
ROBOT_DOF = 16
NORMALIZATION_KEY = "zhuhai_native16"
MODEL_VIEWS = ("observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
               "observation.rgb.right_realsense_link_camera_0")
RESET_PROJECTIONS = ("action_proj", "proprio_proj", "output_proj")


class BoundedRamFeatures:
    """LRU for frozen derived arrays only; model/minibatch memory is separate.

    Array object sizes, keys, tuple values and the OrderedDict index count
    toward this resident cap. Oversize/zero-budget entries are served online.
    No raw image/episode archive is owned or evicted by this layer.
    """
    def __init__(self, store, budget_bytes=2 * 1024**3):
        if not 0 <= budget_bytes <= 2 * 1024**3:
            raise ValueError('resident derived-feature RAM cap must be at most 2GiB')
        self.store, self.budget_bytes = store, int(budget_bytes)
        self.entries = OrderedDict()
        self.used = 0
        self.stats = dict(hits=0, misses=0, evictions=0, resident_bytes=0, peak_resident_bytes=0)

    def get(self, episode, frame):
        key = (episode.sha256, int(frame))
        if key in self.entries:
            self.stats['hits'] += 1
            value, size = self.entries.pop(key)
            self.entries[key] = (value, size)
            return value
        self.stats['misses'] += 1
        value = self.store.get(episode, frame)
        size = (sum(sys.getsizeof(x) for x in value) + sys.getsizeof(value)
                + sys.getsizeof(key) + sum(sys.getsizeof(x) for x in key) + 128)
        if size + sys.getsizeof(self.entries) <= self.budget_bytes:
            while self.entries and self.used + size + sys.getsizeof(self.entries) > self.budget_bytes:
                _, (_, old_size) = self.entries.popitem(last=False)
                self.used -= old_size
                self.stats['evictions'] += 1
            self.entries[key] = (value, size)
            self.used += size
            # Account for an index resize caused by insertion as well.
            while self.entries and self.used + sys.getsizeof(self.entries) > self.budget_bytes:
                _, (_, old_size) = self.entries.popitem(last=False)
                self.used -= old_size
                self.stats['evictions'] += 1
        resident = self.used + (sys.getsizeof(self.entries) if self.entries else 0)
        self.stats['resident_bytes'] = resident
        self.stats['peak_resident_bytes'] = max(self.stats['peak_resident_bytes'], resident)
        return value

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"refusing to replace immutable evidence: {path}")
        return
    atomic_json(path, value)


def resolve(path):
    return (ROOT / path).resolve()


def robot_contract():
    from harness import zhuhai_robot_spec as spec
    if (spec.ROBOT_DOF != ROBOT_DOF or spec.ARM_DOF != 7
            or tuple(spec.GRIPPER_INDICES) != (7, 15)
            or tuple(spec.ARM_INDICES) != tuple(range(7)) + tuple(range(8, 15))):
        raise ValueError("native16 requires left7/gripper/right7/gripper, 1=open")
    result = copy.deepcopy(spec.robot_contract())
    if not isinstance(result, dict) or not result:
        raise ValueError("official robot contract must be a nonempty dictionary")
    if (result.get("action_contract") != spec.ACTION_CONTRACT
            or result.get("state_contract") != spec.STATE_CONTRACT
            or not result.get("asset_sha256") or not result.get("asset_manifest_sha256")):
        raise ValueError("cached official asset and measured/command robot contracts required")
    digest(result)
    return result


def data_contract():
    from harness import zhuhai_demo_recording as recording
    result = copy.deepcopy(recording.data_contract())
    if (recording.STATE_DIM != ROBOT_DOF or recording.ACTION_DIM != ROBOT_DOF
            or result.get("state_semantics") != "measured_joint_gripper"
            or result.get("action_semantics") != "joint_position_target"):
        raise ValueError("native16 measured-state / target-action data contract required")
    digest(result)
    return result


def protocol_binding():
    robot, data = robot_contract(), data_contract()
    return dict(robot_contract=robot, robot_contract_sha256=digest(robot),
                data_contract=data, data_contract_sha256=digest(data))


def validate_binding(value):
    expected = protocol_binding()
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("robot/data contract or SHA differs from current native16 protocol")
    return expected


def validate_collection_contract(contract):
    # Exact contracts, not a robot name or dimension-only compatibility claim.
    expected = protocol_binding()
    if any(contract.get(k) != expected[k] for k in ("robot_contract", "data_contract")):
        raise ValueError("collection robot/data contract is not current measured native16")
    for k in ("robot_contract_sha256", "data_contract_sha256"):
        if k in contract and contract[k] != expected[k]:
            raise ValueError("collection contract digest mismatch")


def validate_config(cfg):
    validate_binding(cfg)
    c, model = cfg["common"], cfg["model"]
    v3 = model["v3"]
    if (c.get("action_dim") != ROBOT_DOF or c.get("state_dim") != ROBOT_DOF
            or model.get("proprio_mask") != "native_all16"
            or list(model["views"]) != list(MODEL_VIEWS)
            or cfg["dataset"].get("frequency_hz") != FREQUENCY
            or c.get("action_chunk_size") != 32 or c.get("action_execution_horizon") != 8
            or cfg["dataset"].get("future_frame_offset") != 32
            or c.get("num_inference_steps") != 10
            or list(cfg["dataset"].get("image_size", [])) != [224, 224]
            or list(model["vision_encoder"].get("feat_layers", [])) != [-12, -8, -4]
            or c.get("proprio_len") != len(v3["proprio_lags"]) + 2):
        raise ValueError("incompatible native16 model/history/10Hz configuration")
    if any(type(x) is not int or x <= 0 for x in v3["proprio_lags"] + v3["memory_lags"]):
        raise ValueError("history lags must be positive integer control frames")


def validate_source_success(row):
    if (row.get("plan_success") is not True or row.get("strict_predicate") is not True
            or row.get("physics_safety_all_observations_passed") is not True):
        raise ValueError("source success requires plan/rule/all-observation physics safety evidence")
    initial, final = row.get("initial_rule_evidence", {}), row.get("final_rule_evidence", {})
    for evidence in (initial, final):
        safety = evidence.get("physics_safety", {})
        if safety.get("passed") is not True or safety.get("events") != [] or not safety.get("checks"):
            raise ValueError("unsafe/unverified source episode cannot enter TRAIN or VTT")
    if (initial.get("score", {}).get("base_score") != 0
            or initial.get("score", {}).get("full_task_success") is not False
            or final.get("score", {}).get("full_task_success") is not True):
        raise ValueError("source requires zero initial score and actual final full-task success")
    times = np.asarray(row.get("accepted_physics_times", []), np.float64)
    count = row.get("observation_count")
    clocks = [evidence.get("physics_time_seconds") for evidence in (initial, final)]
    if any(type(t) not in (int, float) or not np.isfinite(t) for t in clocks):
        raise ValueError("source rule evidence must retain finite actual physical times")
    if (type(count) is not int or count < 2 or times.shape != (count,)
            or not np.isfinite(times).all() or times[0] < 0
            or not np.allclose(np.diff(times), .1, rtol=0, atol=1e-7)
            or not np.isclose(times[0], clocks[0], rtol=0, atol=1e-7)
            or not np.isclose(times[-1], clocks[1], rtol=0, atol=1e-7)):
        raise ValueError("source observations require an aligned, actual 10Hz clock")
    coverage, cameras = row.get("initial_head_camera_evidence", {}), row.get("initial_camera_observations", {})
    if (coverage.get("all_geometry_in_frame") is not True or coverage.get("all_named_entities_visible") is not True
            or set(cameras) != set(data_contract()["camera_names"])
            or any(item.get("nonempty_rgb") is not True for item in cameras.values())):
        raise ValueError("source requires actual head coverage and three RGB readbacks")


def task_sources(path, *, allow_small=False, source_count=100, randomized_count=500):
    """Freeze family splits from admitted manifests, never expert-filter eval seeds."""
    from harness.zhuhai_dataset import grouped_split
    supplied = json.loads(resolve(path).read_text())
    specs = supplied.get("tasks")
    if not isinstance(specs, list) or not 1 <= len(specs) <= 2:
        raise ValueError("tasks must contain one or both registered life tasks")
    tasks, sources = {}, {}
    for spec in specs:
        task = spec["task"]
        if task not in ("zhuhai_task1", "zhuhai_task2") or task in tasks:
            raise ValueError("task must be unique zhuhai_task1/zhuhai_task2")
        rows, inventory = [], []
        counts = {}
        reference = None
        for mode, field in (("source", "source_manifest"), ("randomized", "randomized_manifest")):
            name = spec.get(field)
            if not name:
                if mode == "randomized" and allow_small:
                    counts[mode] = 0
                    continue
                raise ValueError(f"Missing {field} for {task}")
            file = resolve(name)
            manifest = json.loads(file.read_text())
            contract = manifest["contract"]
            validate_collection_contract(contract)
            engine = contract.get('engine')
            if not isinstance(engine, dict) or engine != {'name': 'Isaac Sim', 'version': '6.1.0.0'}:
                raise ValueError('User-selected Isaac Sim engine provenance required; SAPIEN demonstrations are not admitted')
            if (contract.get("task"), contract.get("track"), contract.get("mode"),
                    contract.get("frequency_hz")) != (
                    task, "life", mode, FREQUENCY):
                raise ValueError("physical collection task/track/robot/10Hz mismatch")
            if contract.get("record_video") is not False or contract.get("third_view") is not False:
                raise ValueError("video/third-view protocol mismatch")
            if contract.get('camera_count') != 3 or not 0 < contract.get('physics_dt_seconds', 0) <= 1 / FREQUENCY:
                raise ValueError('Isaac head+dual-wrist and a declared physics period required')
            if not allow_small and (contract.get("allow_unvalidated_probe") or manifest.get("status") != "complete"):
                raise ValueError("production requires a completed validated collection")
            scene = contract["source_scene"]
            if sha256_file(resolve(scene['path'])) != scene['sha256']:
                raise ValueError('live scene code differs from collection contract; use new data/run')
            for name, expected in contract.get('config_hashes', {}).items():
                if sha256_file(resolve(name)) != expected:
                    raise ValueError('camera/task configuration changed after collection')
            dependencies = contract.get('scene_dependency_sha256', {})
            for name, expected in dependencies.items():
                if sha256_file(resolve(name)) != expected:
                    raise ValueError('camera helper or rule implementation changed after collection')
            scene_key = (scene["path"], scene["sha256"], contract.get("life_mapping"), digest(dependencies))
            if reference is not None and reference != scene_key:
                raise ValueError("source/randomized scene/rule mapping changed; use a new run")
            reference = scene_key
            admitted = [r for r in manifest["attempts"] if r.get("status") == "success"]
            counts[mode] = len(admitted)
            for row in admitted:
                validate_source_success(row)
                rows.append(dict(path=str(resolve(row["data_path"])), sha256=row["data_sha256"],
                                 family_id=row["family_id"], seed=row["seed"], domain=mode,
                                 robot_contract=contract["robot_contract"],
                                 robot_contract_sha256=digest(contract["robot_contract"]),
                                 data_contract=contract["data_contract"],
                                 data_contract_sha256=digest(contract["data_contract"])))
            inventory.append(dict(path=str(file), sha256=sha256_file(file), contract=contract,
                                  admitted_episodes=len(admitted)))
        if not allow_small and counts != {"source": source_count, "randomized": randomized_count}:
            raise ValueError(f"Complete source/DR quota required before training: {task} {counts}")
        if not rows:
            raise ValueError("No successful physical demonstrations")
        if len({r["path"] for r in rows}) != len(rows):
            raise ValueError("duplicate physical demonstration")
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError('source/randomized episode seeds must be unique')
        families = {r["family_id"] for r in rows if r["domain"] == "source"}
        if len(families) != counts['source']:
            raise ValueError('each source demonstration must register a distinct family')
        if any(r["family_id"] not in families for r in rows):
            raise ValueError("randomized demonstration family absent from source")
        n = len(families)
        split_counts = (80, 10, 10) if n == 100 else ((n - 2, 1, 1) if n >= 3 else (n, 0, 0))
        tasks[task] = grouped_split(rows, counts=split_counts)
        sources[task] = dict(manifests=inventory, quotas=counts, families=n,
                             split_family_counts=dict(zip(("train", "val", "heldout"), split_counts)),
                             scene=scene, engine=engine, life_mapping=contract.get("life_mapping"),
                             robot_contract=contract["robot_contract"], data_contract=contract["data_contract"])
    return tasks, sources

def checkpoint_header(path=INITIALIZATION, *, expected_sha=INITIALIZATION_SHA):
    import torch
    path = resolve(path)
    receipt = json.loads(path.with_name(path.name + ".sha256.json").read_text())
    if (expected_sha != INITIALIZATION_SHA or receipt.get("sha256") != expected_sha
            or receipt.get("size") != path.stat().st_size or sha256_file(path) != expected_sha):
        raise ValueError("frozen 50k initialization digest/receipt mismatch")
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if ck.get("format") != INITIALIZATION_FORMAT or ck.get("model_only") is not True or ck.get("step") != 50000:
        raise ValueError("immutable native14 residual-VTT 50k initializer required")
    sd = ck["model_state_dict"]
    expected = {"action_proj.weight": (768, 14), "proprio_proj.weight": (768, 14),
                "output_proj.weight": (14, 768), "task_cond_proj.residual.weight": (768, 1024)}
    if any(k not in sd or tuple(sd[k].shape) != shape for k, shape in expected.items()):
        raise ValueError("initializer embodiment/residual-VTT schema differs")
    return ck, dict(path=str(path), sha256=expected_sha, size=path.stat().st_size, step=ck["step"],
                   source_dof=14, target_dof=ROBOT_DOF, projections_reset=list(RESET_PROJECTIONS),
                   bias_reset=True, residual_VTT_loaded=True, optimizer_state_reused=False,
                   numerical_normalization="new native16 successful TRAIN-only statistics; old norm forbidden")


def config_for(checkpoint, *, steps=500, seed=20261009, batch=4):
    if any(type(x) is not int or x <= 0 for x in (steps, batch)):
        raise ValueError("positive integer steps/batch required")
    cfg = copy.deepcopy(checkpoint["contract"]["config"])
    cfg.update(protocol_binding())
    cfg["common"].update(action_dim=ROBOT_DOF, state_dim=ROBOT_DOF)
    cfg["model"]["proprio_mask"] = "native_all16"
    cfg["model"]["v3"]["stats_path"] = "statistics.json"
    cfg["training"].update(max_steps=steps, schedule_steps=steps, warmup_steps=min(100, steps // 10),
                           seed=seed, batch_size=batch, grad_accum_steps=1, save_every=min(500, steps),
                           validate_every=min(500, steps), learning_rate=1e-5, lr_min=1e-6,
                           init_from=str(INITIALIZATION), init_from_sha256=INITIALIZATION_SHA)
    cfg["dataset"].update(frequency_hz=FREQUENCY, state_semantics="measured_joint_gripper",
                          action_semantics="joint_position_target", cond_mode="train_task_vtt")
    validate_config(cfg)
    return cfg


def validate_episode_binding(episode):
    """A manifest cannot relabel another robot's native16 HDF as this robot."""
    import h5py
    validate_binding(episode.row)
    with h5py.File(episode.path, "r") as handle:
        metadata = json.loads(handle.attrs["metadata"])
    collection = metadata.get("contract")
    if not isinstance(collection, dict):
        raise ValueError("source HDF must embed its collection robot/data contract")
    validate_collection_contract(collection)


def build_datasets(splits, store, cfg):
    from harness.zhuhai_dataset import HybridTaskDataset
    validate_config(cfg)
    v3 = cfg["model"]["v3"]
    result = {split: {} for split in ("train", "val", "heldout")}
    seen_paths, seen_hashes = set(), set()
    for task, task_splits in splits.items():
        family_splits = {}
        for split, rows in task_splits.items():
            if split not in result:
                raise ValueError("unknown split")
            for row in rows:
                validate_binding(row)
                if row.get("split") != split:
                    raise ValueError("episode split mismatch")
                path = str(resolve(row["path"]))
                if path in seen_paths or row["sha256"] in seen_hashes:
                    raise ValueError("duplicate/leaked episode across task splits")
                seen_paths.add(path); seen_hashes.add(row["sha256"])
                family = row["family_id"]
                if family in family_splits and family_splits[family] != split:
                    raise ValueError("source family leaked across splits")
                family_splits[family] = split
            if rows:
                result[split][task] = HybridTaskDataset(task, rows, store,
                    proprio_lags=v3["proprio_lags"], memory_lags=v3["memory_lags"],
                    chunk=cfg["common"]["action_chunk_size"], future_offset=cfg["dataset"]["future_frame_offset"])
                for ep in result[split][task].episodes:
                    validate_episode_binding(ep)
    return result


def train_provenance(datasets, cfg):
    validate_config(cfg)
    records = []
    seen = set()
    for task, ds in sorted(datasets.items()):
        if (list(ds.proprio_lags) != cfg["model"]["v3"]["proprio_lags"]
                or list(ds.memory_lags) != cfg["model"]["v3"]["memory_lags"]):
            raise ValueError("TRAIN history contract differs")
        for ep in ds.episodes:
            validate_episode_binding(ep)
            if ep.row.get("split") != "train" or ep.sha256 != ep.row.get("sha256"):
                raise ValueError("statistics/VTT require hash-verified TRAIN episodes only")
            state, action = np.asarray(ep.state), np.asarray(ep.action)
            if (state.ndim != 2 or state.shape != action.shape or state.shape[1] != ROBOT_DOF
                    or not len(state) or not np.isfinite(state).all() or not np.isfinite(action).all()):
                raise ValueError("finite aligned (N,16) measured state / target action required")
            # Preserve measured overshoot/undertravel; only commands must stay in [0,1].
            if ((action[:, [7, 15]] < 0) | (action[:, [7, 15]] > 1)).any():
                raise ValueError("gripper target must use [0,1], 1=open")
            if ep.sha256 in seen:
                raise ValueError("duplicate TRAIN content")
            seen.add(ep.sha256)
            records.append(dict(task=task, path=str(ep.path), sha256=ep.sha256, split="train",
                                family_id=ep.row["family_id"], seed=ep.row["seed"], frames=len(state)))
    if not records:
        raise ValueError("no successful TRAIN episodes")
    return dict(fit_split="train", heldout_used=False, records=records, train_data_sha256=digest(records),
                episode_count=len(records), frame_count=sum(r["frames"] for r in records), **protocol_binding())


def fit_train_statistics(datasets, cfg, *, feature_identity, check=lambda: None):
    """Bounded float64 moments; never use initializer, VAL or heldout moments."""
    from harness.robotwin_training_data import RunningMoments
    from harness.wam_v3 import history_frames
    provenance = train_provenance(datasets, cfg)
    if (set(feature_identity) != {"encoder_sha256", "preprocess_sha256"}
            or any(not isinstance(x, str) or len(x) != 64 for x in feature_identity.values())):
        raise ValueError("DINO/preprocessing hashes required for TRAIN feature moments")
    provenance.update(feature_identity=copy.deepcopy(feature_identity),
        variance="population_float64_parallel_welford", std_floor=1e-3, history_start="episode_frame_0")
    st, act, prop, mem = (RunningMoments() for _ in range(4))
    for _, ds in sorted(datasets.items()):
        for ep in ds.episodes:
            for lo in range(0, len(ep.state), 64):
                check()
                hi = min(lo + 64, len(ep.state))
                frames = list(range(lo, hi))
                state = np.asarray(ep.state, np.float64)
                st.update(state[lo:hi]); act.update(ep.action[lo:hi])
                pf = np.array([history_frames(t, 0, ds.proprio_lags) for t in frames])
                prop.update(state[lo:hi, None] - state[pf])
                differences = []
                for t in frames:
                    current = np.asarray(ds.store.get(ep, t)[1], np.float64)
                    old = np.stack([ds.store.get(ep, i)[1] for i in history_frames(t, 0, ds.memory_lags)], axis=1)
                    if current.shape != (3, 2048) or old.shape != (3, len(ds.memory_lags) + 1, 2048):
                        raise ValueError("DINO history pool geometry mismatch")
                    differences.append(current[:, None] - old.astype(np.float64))
                mem.update(np.stack(differences))
    statistics = dict(normalization={"schema_version": 2, "action_repr": "absolute",
        NORMALIZATION_KEY: {"state": st.result(), "action": act.result()}, "provenance": provenance},
        temporal_stats=dict(schema_version=2, views=list(MODEL_VIEWS),
            proprio_lags=cfg["model"]["v3"]["proprio_lags"], memory_lags=cfg["model"]["v3"]["memory_lags"],
            proprio_diff_std=prop.result()["std"], memory_diff_std=mem.result()["std"], provenance=provenance),
        proprio_keep_mask=[1.0] * ROBOT_DOF, **protocol_binding())
    validate_statistics(statistics, cfg)
    return statistics


def validate_statistics(statistics, cfg):
    validate_config(cfg); validate_binding(statistics)
    norm, temporal = statistics["normalization"], statistics["temporal_stats"]
    if "robotwin" in norm or norm.get("action_repr") != "absolute" or NORMALIZATION_KEY not in norm:
        raise ValueError("legacy/native14 or non-absolute normalization is forbidden")
    provenance = norm["provenance"]
    validate_binding(provenance)
    if (provenance.get("fit_split") != "train" or provenance.get("heldout_used") is not False
            or not provenance.get("records") or temporal.get("provenance") != provenance
            or any(r.get("split") != "train" for r in provenance["records"])
            or provenance.get("train_data_sha256") != digest(provenance["records"])):
        raise ValueError("unbound/non-TRAIN statistics provenance")
    if (provenance.get("episode_count") != len(provenance["records"])
            or any(type(r.get("frames")) is not int or r["frames"] <= 0 for r in provenance["records"])
            or provenance.get("frame_count") != sum(r["frames"] for r in provenance["records"])
            or set(provenance.get("feature_identity", {})) != {"encoder_sha256", "preprocess_sha256"}):
        raise ValueError("TRAIN frame counts/feature identity are not bound")
    for key in ("state", "action"):
        record = norm[NORMALIZATION_KEY][key]
        for k in ("min", "max", "mean", "std"):
            x = np.asarray(record[k], np.float64)
            if x.shape != (ROBOT_DOF,) or not np.isfinite(x).all() or (k == "std" and (x <= 0).any()):
                raise ValueError("statistics must contain finite native16 moments")
        if (np.asarray(record["max"]) < record["min"]).any():
            raise ValueError("normalization max below min")
    v3 = cfg["model"]["v3"]
    if (temporal.get("views") != list(MODEL_VIEWS) or temporal.get("proprio_lags") != v3["proprio_lags"]
            or temporal.get("memory_lags") != v3["memory_lags"]
            or statistics.get("proprio_keep_mask") != [1.0] * ROBOT_DOF):
        raise ValueError("normalization view/history/native16 mask differs")
    for name, shape in (("proprio_diff_std", (len(v3["proprio_lags"]) + 1, ROBOT_DOF)),
                        ("memory_diff_std", (3, len(v3["memory_lags"]) + 1, 2048))):
        value = np.asarray(temporal[name], np.float64)
        if value.shape != shape or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError("invalid native16 temporal normalization")
    return provenance


def install_conditions(datasets, cfg, statistics):
    provenance = train_provenance(datasets["train"], cfg)
    fitted = validate_statistics(statistics, cfg)
    if provenance["train_data_sha256"] != fitted["train_data_sha256"]:
        raise ValueError("VTT TRAIN records differ from normalization records")
    conditions = {task: ds.train_prototype().tolist() for task, ds in sorted(datasets["train"].items())}
    bank = dict(format=FORMAT + "/vtt", source_split="train", conditions=conditions,
        provenance=provenance, statistics_sha256=digest(statistics),
        feature_identity=fitted["feature_identity"], **protocol_binding())
    validate_conditions(bank, statistics, cfg)
    for split in ("train", "val", "heldout"):
        for task, ds in datasets[split].items():
            if task not in conditions:
                raise ValueError("evaluation task has no TRAIN VTT")
            ds.condition = np.asarray(conditions[task], np.float32).copy()
    return bank


def validate_conditions(bank, statistics, cfg):
    validate_binding(bank)
    provenance = validate_statistics(statistics, cfg)
    if (bank.get("format") != FORMAT + "/vtt" or bank.get("source_split") != "train"
            or bank.get("statistics_sha256") != digest(statistics)
            or bank.get("feature_identity") != provenance["feature_identity"]
            or bank.get("provenance", {}).get("train_data_sha256") != provenance["train_data_sha256"]):
        raise ValueError("VTT bank must bind the same TRAIN data, robot, features and native16 statistics")
    source = bank.get("provenance", {})
    validate_binding(source)
    if (source.get("fit_split") != "train" or source.get("heldout_used") is not False
            or source.get("records") != provenance["records"]
            or set(bank["conditions"]) != {row["task"] for row in provenance["records"]}):
        raise ValueError("condition source records/tasks differ from the successful TRAIN statistics")
    vectors = []
    for task, value in bank["conditions"].items():
        x = np.asarray(value, np.float64)
        if x.shape != (1024,) or not np.isfinite(x).all() or np.linalg.norm(x - x.mean()) < 1e-6:
            raise ValueError("invalid/LayerNorm-degenerate TRAIN VTT")
        z = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
        if any(np.allclose(z, other, rtol=1e-5, atol=1e-6) for other in vectors):
            raise ValueError("indistinguishable task VTTs after LayerNorm")
        vectors.append(z)
    if not vectors:
        raise ValueError("empty VTT bank")


def validate_training_contract(contract):
    if contract.get("format") != FORMAT:
        raise ValueError("native16 measured-state checkpoint format required; native14 cannot resume here")
    validate_binding(contract)
    statistics, cfg = contract["statistics"], contract["config"]
    validate_statistics(statistics, cfg)
    if contract.get("statistics_sha256") != digest(statistics):
        raise ValueError("native16 statistics digest mismatch")
    validate_conditions(contract["condition_bank"], statistics, cfg)
    if contract.get("conditions") != contract["condition_bank"]["conditions"]:
        raise ValueError("checkpoint condition alias differs from bound TRAIN bank")
    if contract.get("time") != time_contract(cfg):
        raise ValueError("checkpoint timing contract changed")
    provenance = statistics["normalization"]["provenance"]
    feature = {k: contract["cache"].get(k) for k in ("encoder_sha256", "preprocess_sha256")}
    if feature != provenance["feature_identity"]:
        raise ValueError("statistics/conditions use different DINO or preprocessing from this checkpoint")
    actual_records = set()
    for task, splits in contract["splits"].items():
        families = {}
        for split, rows in splits.items():
            if split not in ("train", "val", "heldout"):
                raise ValueError("unknown checkpoint data split")
            for row in rows:
                validate_binding(row)
                if row.get("split") != split:
                    raise ValueError("checkpoint row split mismatch")
                family = str(row["family_id"])
                if family in families and families[family] != split:
                    raise ValueError("checkpoint source family leaks across splits")
                families[family] = split
                if split == "train":
                    actual_records.add((task, str(resolve(row["path"])), row["sha256"], family, row["seed"]))
    fitted_records = {(r["task"], str(resolve(r["path"])), r["sha256"], str(r["family_id"]), r["seed"])
                      for r in provenance["records"]}
    if fitted_records != actual_records or len(fitted_records) != len(provenance["records"]):
        raise ValueError("statistics/condition provenance does not match this contract's TRAIN split")
    return contract


def validate_postprocessing(settings):
    from harness import zhuhai_robot_spec as spec
    validate_binding(settings)
    if (settings.get("enabled") is not True
            or settings.get("smoothed_dimensions") != list(spec.ARM_INDICES)
            or settings.get("preserved_gripper_dimensions") != list(spec.GRIPPER_INDICES)):
        raise ValueError("native16 arm-only smoothing and untouched gripper columns 7/15 required")
    rate = settings["rate_limit"]
    bounds = np.asarray(spec.DUAL_JOINT_LIMITS, np.float64)
    if rate.get("frequency_hz") != FREQUENCY:
        raise ValueError("postprocessing must use the same 10Hz clock")
    for key in ("position_lower_rad", "position_upper_rad", "arm_velocity_rad_s", "arm_acceleration_rad_s2"):
        value = np.asarray(rate[key], np.float64)
        if value.shape != (14,) or not np.isfinite(value).all():
            raise ValueError("postprocessing must cover fourteen physical arm joints")
        if key.startswith("arm_") and (value <= 0).any():
            raise ValueError("command rate/acceleration bounds must be positive")
    if (not np.allclose(rate["position_lower_rad"], bounds[:, 0], rtol=0, atol=1e-7)
            or not np.allclose(rate["position_upper_rad"], bounds[:, 1], rtol=0, atol=1e-7)):
        raise ValueError("postprocessing bounds must match official Panda joint travel")
    return settings


class FrozenFeatureEncoder:
    """Encoding bridge before new statistics exist; no action model/old norm."""
    def __init__(self, encoder, *, device):
        import torch
        self.vision_encoder = encoder.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.views, self.feat_layers, self.num_register_tokens = list(MODEL_VIEWS), [-12, -8, -4], 4

    def encode(self, pixels):
        from harness.wam_v3 import WamV3Wrapper
        return WamV3Wrapper.encode(self, pixels)


def create_action_model(cfg, *, device="cpu"):
    import torch
    from omegaconf import OmegaConf
    from harness.wam_v3 import create_v3_action_model
    from harness.robotwin_multitask_training import install_joint_projection
    validate_config(cfg)
    with torch.device(device):
        model = create_v3_action_model(OmegaConf.create(cfg), dino_hidden_size=1024,
                                      num_dino_layers=3, task_cond_dim=1024, patch_size=16)
        # BEFORE any initializer load: its keys already contain base.* and residual.*.
        install_joint_projection(model)
    return model.to(device)


def build_wrapper(cfg, statistics, *, device="cpu", state_dict=None, initialization=None):
    import torch
    from harness.wam_v3 import WamV3Wrapper, load_retargeted_v3_state
    validate_statistics(statistics, cfg)
    if (state_dict is None) == (initialization is None):
        raise ValueError("supply either a native16 state_dict or the frozen initializer, never both/neither")
    model = create_action_model(cfg, device=device)
    sd = initialization["model_state_dict"] if initialization is not None else state_dict
    if torch.device(device).type == "meta":
        sd = {k: value.to("meta") for k, value in sd.items()}
    if initialization is not None:
        if (initialization.get("format") != INITIALIZATION_FORMAT
                or initialization.get("step") != 50000 or initialization.get("model_only") is not True
                or sd["action_proj.weight"].shape[1] != 14):
            raise ValueError("selective migration requires the native14 residual-VTT 50k initializer")
        notes = dict(initialization="native14_50k_to_native16_selective", **load_retargeted_v3_state(model, sd))
        notes.update(residual_VTT_loaded=True, numerical_normalization_reused=False,
                     robot_contract_sha256=cfg["robot_contract_sha256"])
    else:
        model.load_state_dict(sd, strict=True)
        notes = dict(initialization="strict_native16_checkpoint", projections_reset=False)
    with tempfile.TemporaryDirectory(prefix="zhuhai-native16-stats-") as directory:
        path = Path(directory) / "normalization.json"
        path.write_text(json.dumps(statistics["normalization"], allow_nan=False))
        wrapper = WamV3Wrapper(model, views=cfg["model"]["views"], stats=statistics["temporal_stats"],
            norm_stats_path=path, normalization_key=NORMALIZATION_KEY, proprio_mask="native_all16",
            proprio_keep_override=statistics["proprio_keep_mask"], train_cfg=cfg["training"],
            vision_encoder=None, feat_layers=(-12, -8, -4), num_register_tokens=4, device=device).to(device)
    return wrapper, notes


def time_contract(cfg):
    v3 = cfg["model"]["v3"]
    return dict(frequency_hz=FREQUENCY, seconds_per_action=.1,
                proprio_lags_frames=v3["proprio_lags"], proprio_lags_seconds=[x / FREQUENCY for x in v3["proprio_lags"]],
                memory_lags_frames=v3["memory_lags"], memory_lags_seconds=[x / FREQUENCY for x in v3["memory_lags"]],
                chunk_frames=32, chunk_seconds=3.2, execution_frames=8, execution_seconds=.8,
                future_offset_frames=32, future_offset_seconds=3.2,
                protocol_change="native14 variable TOPP/command-state -> native16 fixed 10Hz/measured-state; new TRAIN normalization")


def load_policy(path, task, device="cuda"):
    import torch
    path = resolve(path)
    receipt = json.loads(path.with_name(path.name + ".sha256.json").read_text())
    if receipt.get("size") != path.stat().st_size or receipt.get("sha256") != sha256_file(path):
        raise ValueError("native16 checkpoint content/receipt mismatch")
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if ck.get("format") != FORMAT or ck.get("model_only") is not True:
        raise ValueError("evaluation requires an immutable native16 Zhuhai milestone")
    if receipt.get("model_only") is not True or receipt.get("step") != ck.get("step"):
        raise ValueError("milestone step/model-only receipt differs from checkpoint")
    contract = validate_training_contract(ck["contract"])
    if ck["contract_sha256"] != digest(contract) or task not in contract["conditions"]:
        raise ValueError("checkpoint task/contract mismatch")
    wrapper, _ = build_wrapper(contract["config"], contract["statistics"], device=device, state_dict=ck["model_state_dict"])
    wrapper.eval()
    return wrapper, np.asarray(contract["conditions"][task], np.float32), contract
