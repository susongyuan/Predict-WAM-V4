#!/usr/bin/env python3
"""Run one process tree in a cgroup while monitoring global RAM and GPU use.

This is a conservative admission/abort monitor, not a Vulkan VRAM hard cap.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from behavior_preflight import CONFIG, ROOT, hardware_snapshot


def resource_policy(environment):
    """Resolve separate CPU/RAM and GPU budgets from the pinned user receipt."""
    policy = dict(CONFIG["resource_policy"])
    numeric_keys = ("P2_GPU_AUTHORIZATION_SHA256", "P2_GPU_UTILIZATION_ALLOWED",
                    "P2_TOTAL_VRAM_ABORT_MIB", "P2_CPU_PERCENT_CEILING",
                    "P2_RAM_PERCENT_CEILING", "P2_GPU_CONCURRENCY")
    authorization = environment.get("GPU_AUTHORIZATION")
    if not authorization:
        if any(key in environment for key in numeric_keys):
            raise ValueError("resource overrides require the pinned GPU authorization receipt")
        return policy, None
    from wam_v3_resource_authorization import validate_authorization, resource_environment
    contract = validate_authorization(authorization)
    expected = resource_environment(contract)
    for key in numeric_keys:
        if key in environment and environment[key] != expected[key]:
            raise ValueError(f"resource environment disagrees with authorization: {key}")
    policy.update(monitor_exit_percent=contract["cpu_percent_ceiling"],
                  ram_exit_percent=contract["ram_percent_ceiling"],
                  admission_total_percent=min(contract["cpu_percent_ceiling"], contract["ram_percent_ceiling"]),
                  gpu_total_vram_limit_mib=contract["total_vram_abort_mib"],
                  gpu_vram_exit_percent=100, gpu_admission_percent=100,
                  gpu_utilization_exit_percent=contract["gpu_utilization_percent_allowed"],
                  authorized_gpu_peaks=True)
    return policy, contract


def violation(snapshot, policy=None):
    """Abort at the first over-limit sample. Polling cannot cap an instantaneous GPU spike."""
    policy = CONFIG["resource_policy"] if policy is None else policy
    if snapshot["memory_used_percent"] >= policy.get("ram_exit_percent", policy["monitor_exit_percent"]):
        return "global RAM threshold reached"
    if snapshot["cpu_used_percent"] >= policy["monitor_exit_percent"]:
        return "global CPU threshold reached"
    if not snapshot["gpu"]:
        return "GPU monitoring unavailable"
    gpu = snapshot["gpu"][0]
    limit = min(policy["gpu_total_vram_limit_mib"],
                gpu["vram_total_mib"] * policy.get("gpu_vram_exit_percent", policy["monitor_exit_percent"]) / 100)
    if gpu["vram_used_mib"] >= limit:
        return "global VRAM threshold reached"
    peaks_allowed = policy.get("authorized_gpu_peaks", CONFIG.get("authorization", {}).get("short_gpu_utilization_spikes_allowed", False))
    if gpu["utilization_percent"] >= policy["gpu_utilization_exit_percent"] and not peaks_allowed:
        return "global GPU utilization threshold reached"
    return None


def run_guarded(argv, environment, output_dir, timeout=1200, memory_max_bytes=None, abort=None):
    policy, resource_contract = resource_policy({**os.environ, **environment})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = hardware_snapshot()
    if not policy["gpu_execution_enabled"]:
        raise RuntimeError("GPU execution disabled in resource policy")
    reason = violation(snapshot, policy)
    if reason:
        raise RuntimeError("Admission denied: " + reason)
    if not snapshot["gpu"]:
        raise RuntimeError("Cannot admit a GPU process without GPU telemetry")
    gpu = snapshot["gpu"][0]
    memory_used = snapshot["memory_total_bytes"] - snapshot["memory_available_bytes"]
    admission_fraction = policy["admission_total_percent"] / 100
    memory_budget = int(snapshot["memory_total_bytes"] * admission_fraction - memory_used)
    vram_budget = min(policy["gpu_total_vram_limit_mib"],
                      gpu["vram_total_mib"] * policy.get("gpu_admission_percent", policy["admission_total_percent"]) / 100)
    if memory_budget < 8 * 1024**3 or gpu["vram_used_mib"] + 8192 > vram_budget:
        raise RuntimeError("Admission denied: an estimated 8 GiB RAM / 8 GiB VRAM exceeds the configured total budget")
    if snapshot["cpu_used_percent"] + 100 * policy.get("simulation_cpu_threads", 4) / snapshot["cpu_logical"] >= policy["admission_total_percent"]:
        raise RuntimeError("Admission denied: CPU headroom is insufficient")
    if memory_max_bytes is not None:
        if memory_max_bytes <= 0:
            raise ValueError("memory_max_bytes must be positive")
        memory_budget = min(memory_budget, int(memory_max_bytes))
    cpu_affinity = sorted(os.sched_getaffinity(0))[:policy.get("simulation_cpu_threads", 4)]
    unit = f"behavior-run-{os.getpid()}"
    command = ["systemd-run", "--user", "--quiet", "--wait", "--pipe", "--collect", f"--unit={unit}",
               f"--working-directory={ROOT}", "-p", "CPUAffinity=" + " ".join(map(str, cpu_affinity)),
               "-p", f"MemoryMax={memory_budget}",
               "-p", "MemorySwapMax=0", "env", *[f"{k}={v}" for k, v in environment.items()], *argv]
    result = {"unit": unit, "argv": argv, "ram_cgroup_max_bytes": memory_budget,
              "cpu_affinity": cpu_affinity,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "guarantees_global_vram_cap": False, "guarantees_gpu_utilization_cap": False,
              "monitor_policy": policy, "resource_authorization": resource_contract,
              "peak_cpu_percent": 0, "peak_ram_percent": 0, "peak_gpu_percent": 0, "peak_vram_mib": 0}
    started = time.monotonic()
    reason = None
    with (output_dir / "process.log").open("w") as log, (output_dir / "resources.jsonl").open("w") as metrics:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                snapshot = hardware_snapshot()
                metrics.write(json.dumps({"elapsed_seconds": time.monotonic() - started, **snapshot}) + "\n")
                metrics.flush()
                result["peak_cpu_percent"] = max(result["peak_cpu_percent"], snapshot["cpu_used_percent"])
                result["peak_ram_percent"] = max(result["peak_ram_percent"], snapshot["memory_used_percent"])
                if snapshot["gpu"]:
                    result["peak_gpu_percent"] = max(result["peak_gpu_percent"], snapshot["gpu"][0]["utilization_percent"])
                    result["peak_vram_mib"] = max(result["peak_vram_mib"], snapshot["gpu"][0]["vram_used_mib"])
                reason = violation(snapshot, policy)
                if timeout is not None and time.monotonic() - started > timeout:
                    reason = "run time limit reached"
                if abort is not None and abort.is_set():   # e.g. the launcher refused to start the policy
                    reason = "aborted by launcher: " + str(getattr(abort, "reason", "abort requested"))
                if reason:
                    break
                time.sleep(policy["poll_seconds"])
        finally:
            if child.poll() is None:
                subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", unit], capture_output=True)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", unit], capture_output=True)
                    child.wait(timeout=10)
        result.update(returncode=child.returncode, abort_reason=reason, elapsed_seconds=time.monotonic() - started)
    (output_dir / "resource_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return child.returncode if child.returncode else (2 if reason else 0)


def self_test():
    snapshot = {"memory_used_percent": 40, "cpu_used_percent": 10,
                "gpu": [{"vram_total_mib": 16311, "vram_used_mib": 2500, "utilization_percent": 10}]}
    allowed = CONFIG["authorization"]["short_gpu_utilization_spikes_allowed"]
    CONFIG["authorization"]["short_gpu_utilization_spikes_allowed"] = False
    assert violation(snapshot) is None
    snapshot["gpu"][0]["utilization_percent"] = 85
    assert violation(snapshot) == "global GPU utilization threshold reached"
    snapshot["gpu"][0]["utilization_percent"] = 10
    assert violation(snapshot) is None
    snapshot["gpu"][0]["utilization_percent"] = 100
    assert violation(snapshot) == "global GPU utilization threshold reached"
    CONFIG["authorization"]["short_gpu_utilization_spikes_allowed"] = True
    assert violation(snapshot) is None
    snapshot["gpu"][0]["utilization_percent"] = 10
    snapshot["gpu"][0]["vram_used_mib"] = CONFIG["resource_policy"]["gpu_total_vram_limit_mib"]
    assert violation(snapshot) == "global VRAM threshold reached"
    snapshot["memory_used_percent"] = CONFIG["resource_policy"]["monitor_exit_percent"]
    assert violation(snapshot) == "global RAM threshold reached"
    snapshot["memory_used_percent"] = 40
    snapshot["cpu_used_percent"] = CONFIG["resource_policy"]["monitor_exit_percent"]
    assert violation(snapshot) == "global CPU threshold reached"
    snapshot["cpu_used_percent"] = 10
    snapshot["gpu"] = []
    assert violation(snapshot) == "GPU monitoring unavailable"
    CONFIG["authorization"]["short_gpu_utilization_spikes_allowed"] = allowed
    print("PASS: CUDA utilization exception requires explicit authorization; CPU/RAM/VRAM and missing telemetry still abort")


if __name__ == "__main__":
    self_test()
