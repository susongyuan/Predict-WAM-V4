#!/usr/bin/env python3
"""Read-only BEHAVIOR readiness report. Never imports Isaac Sim, torch, or OmniGibson."""
import argparse
import ast
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/behavior_environment.json").read_text())
SOURCE = ROOT / CONFIG.get("source_dir", "third_party/BEHAVIOR-1K")
ENV = ROOT / CONFIG.get("environment_dir", ".conda-envs/behavior")


def child_environment(extra=None):
    """Environment variables for guarded simulator subprocesses (data path, headless, EULA, thread caps, libstdc++)."""
    env = {"OMNIGIBSON_DATA_PATH": str(ROOT / CONFIG["data_path"]), "OMNIGIBSON_HEADLESS": "1",
           "OMNIGIBSON_APPDATA_PATH": str(ROOT / "outputs/behavior/appdata"), "OMNI_KIT_ACCEPT_EULA": "YES",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    if CONFIG.get("ld_preload"):
        env["LD_PRELOAD"] = str(ENV / CONFIG["ld_preload"])  # conda libstdc++ (CXXABI_1.3.15+) for Kit python extensions
    env.update(extra or {})
    return env


def source_commit():
    if (SOURCE / ".git").exists():
        return subprocess.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    prov = json.loads((SOURCE / CONFIG.get("source_provenance_file", "PORT_PROVENANCE.json")).read_text())
    return prov["base_commit"]


def source_slices(name):
    tree = ast.parse((SOURCE / "OmniGibson/omnigibson/eval/utils/eval_utils.py").read_text())
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    robot = value.values[[ast.literal_eval(k) for k in value.keys].index("R1Pro")]
    mapping = robot.args[0]  # OrderedDict({...}); parse syntax instead of importing the simulator.
    return {ast.literal_eval(k): [ast.literal_eval(v.slice.lower), ast.literal_eval(v.slice.upper)]
            for k, v in zip(mapping.keys, mapping.values)}


def inspect_environment():
    python = ENV / "bin/python"
    if not python.exists():
        return {"python": None, "packages": {}, "robot_config": None}
    script = """
import importlib.metadata as m, json, sys
versions = {}
for name in json.loads(sys.argv[1]):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
try:
    import yaml
    with open(sys.argv[2]) as f: cfg = yaml.safe_load(f)
except ImportError:
    cfg = None
print(json.dumps({'python': '.'.join(map(str,sys.version_info[:3])), 'packages': versions, 'robot_config': cfg}))
"""
    result = subprocess.run([str(python), "-I", "-c", script, json.dumps(list(CONFIG["packages"])),
                             str(SOURCE / "OmniGibson/omnigibson/eval/r1pro.yaml")],
                            check=True, text=True, capture_output=True, timeout=15)
    return json.loads(result.stdout)


def verify_interfaces(environment):
    action = source_slices("ACTION_QPOS_INDICES")
    state = source_slices("PROPRIOCEPTION_INDICES")
    assert action == CONFIG["action_slices"], "Official 23D action layout changed"
    assert state == CONFIG["state_slices"], "Official 61D state layout changed"
    for mapping, length in ((action, 23), (state, 61)):
        assert [i for start, end in mapping.values() for i in range(start, end)] == list(range(length))
    cfg = environment["robot_config"]
    assert cfg is not None, "Install the base environment (PyYAML required for controller verification)"
    assert cfg["model"] == "r1pro" and cfg["action_normalize"] is False
    assert cfg["proprio_obs"] == list(state), "Official state concatenation order changed"
    controllers = cfg["controller_config"]
    assert controllers["base"]["motor_type"] == "velocity"
    for part in ("arm_left", "arm_right", "trunk"):
        assert controllers[part]["motor_type"] == "position"
        assert controllers[part]["use_delta_commands"] is False
        assert controllers[part]["use_impedances"] is False
    for part in ("gripper_left", "gripper_right"):
        assert controllers[part]["mode"] == "smooth"
    assert cfg["grasping_mode"] == "assisted"
    info = json.loads((ROOT / "data/behavior-1k/2026-challenge-demos/meta/info.json").read_text())
    assert info["features"]["action"]["shape"] == [23]
    assert info["features"]["observation.state"]["shape"] == [61]
    assert info["fps"] == 30
    return {"verified": True, "action_slices": action, "state_slices": state,
            "controllers": controllers, "grasping_mode": cfg["grasping_mode"],
            "zero_action_is_safe_hold": False, "frequency_hz": 30}


def hardware_snapshot():
    memory = {k: int(v.split()[0]) * 1024 for k, v in
              (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    def cpu_ticks():
        # Exclude guest ticks, which Linux already includes in user/nice.
        values = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
        return sum(values), values[3] + values[4]
    total1, idle1 = cpu_ticks()
    time.sleep(0.2)
    total2, idle2 = cpu_ticks()
    result = {"cpu_logical": os.cpu_count(), "cpu_used_percent": round(100 * (1 - (idle2-idle1) / max(1, total2-total1)), 2),
              "memory_total_bytes": memory["MemTotal"], "memory_available_bytes": memory["MemAvailable"],
              "memory_used_percent": round(100 * (1 - memory["MemAvailable"] / memory["MemTotal"]), 2),
              "disk_free_bytes": shutil.disk_usage(ROOT).free, "glibc": platform.libc_ver(), "gpu": []}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5)
        for row in gpu.stdout.splitlines():
            name, driver, total, used, utilization = [x.strip() for x in row.split(",")]
            result["gpu"].append({"name": name, "driver": driver, "vram_total_mib": int(total),
                                  "vram_used_mib": int(used), "utilization_percent": int(utilization)})
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["gpu_probe_error"] = str(error)
    return result


def collect():
    blockers = []
    environment = inspect_environment()
    commit = source_commit()
    if commit != CONFIG["source_commit"]:
        blockers.append("source_commit_mismatch")
    if not str(environment["python"]).startswith(CONFIG["python"] + "."):
        blockers.append(f"python_{CONFIG['python'].replace('.', '_')}_missing")
    for package, expected in CONFIG["packages"].items():
        actual = environment["packages"].get(package)
        if actual is None or actual.split("+", 1)[0] != expected:
            blockers.append(f"package:{package}:requires:{expected}:found:{actual}")
    try:
        interfaces = verify_interfaces(environment)
    except (AssertionError, OSError, ValueError, StopIteration) as error:
        interfaces = {"verified": False, "error": str(error)}
        blockers.append("interface_check_failed")
    data_path = Path(os.environ.get("OMNIGIBSON_DATA_PATH", ROOT / CONFIG["data_path"])).expanduser().resolve()
    assets = []
    for asset in CONFIG["assets"]:
        directory = data_path / asset["directory"]
        version_file = directory / "VERSION"
        version = version_file.read_text().strip() if version_file.exists() else None
        present = directory.is_dir() and any(directory.iterdir())
        valid = present and (asset["version"] is None or version == asset["version"])
        assets.append({"directory": str(directory), "present": present, "version": version, "valid": valid})
        if not valid:
            blockers.append(f"asset_missing_or_wrong_version:{asset['directory']}")
    for name in ("B100_task_misc.csv", "task.jsonl"):
        if not (data_path / "2026-challenge-task-instances/metadata" / name).is_file():
            blockers.append(f"metadata_missing:{name}")
    asset_key_present = (data_path / "omnigibson.key").is_file()
    if not asset_key_present:
        blockers.append("official_asset_key_not_installed")
    if not CONFIG["resource_policy"]["gpu_execution_enabled"]:
        blockers.append("gpu_execution_disabled")
    snapshot = hardware_snapshot()
    if max(snapshot["cpu_used_percent"], snapshot["memory_used_percent"]) >= CONFIG["resource_policy"]["admission_total_percent"]:
        blockers.append("host_at_or_above_admission_budget")
    smoke_path = ROOT / "outputs/behavior/scene_smoke/scene_smoke.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    return {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "prepared_not_sim_ready" if blockers else "ready_for_guarded_scene_smoke",
            "scene_smoke": smoke,
            "source_commit": commit, "environment": str(ENV), "python": environment["python"],
            "packages": environment["packages"], "interfaces": interfaces, "assets": assets,
            "asset_key_present": asset_key_present,
            "hardware_snapshot": snapshot, "resource_policy": CONFIG["resource_policy"],
            "license_acceptance": CONFIG.get("authorization", {}),
            "blockers": blockers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save this JSON snapshot")
    parser.add_argument("--check", action="store_true", help="Exit 2 when simulation is blocked")
    parser.add_argument("--self-test", action="store_true", help="Check official action/state/controller contracts only")
    args = parser.parse_args()
    if args.self_test:
        verify_interfaces(inspect_environment())
        assert set(CONFIG["profiles"]["development"]).isdisjoint(CONFIG["profiles"]["report"])
        print("PASS: official 23D action, 61D state, controllers, metadata, and disjoint eval splits")
        return
    result = collect()
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if args.check and result["blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
