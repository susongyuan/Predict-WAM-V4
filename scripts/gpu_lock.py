#!/usr/bin/env python3
"""Single-GPU mutual exclusion for simulator runs, DINOv3 precomputes, WAM training and GPU admission checks.

The lock is the file /tmp/project2_gpu.lock holding one owner name (same file and semantics as
scripts/replay_demo_actions_feasibility.py acquire_gpu_lock/release_gpu_lock). Every GPU job takes it; the
simulator and training never share the GPU.

  python3 scripts/gpu_lock.py status
  python3 scripts/gpu_lock.py acquire --owner NAME [--max-wait S]   # blocks until free, then writes NAME
  python3 scripts/gpu_lock.py release --owner NAME                  # removes only if NAME owns it
  python3 scripts/gpu_lock.py hold-for-unit --owner NAME --unit U   # detached releaser: frees when unit U stops
  python3 scripts/gpu_lock.py run --owner NAME -- CMD ...           # acquire, run CMD, always release
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path("/tmp/project2_gpu.lock")


def owner():
    try:
        return LOCK.read_text().strip()
    except FileNotFoundError:
        return None


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.split()
    return int(out[0])


def acquire(name, max_wait=None, poll=10.0, max_used_mib=4000):
    started, last = time.monotonic(), None
    while True:
        reason = None
        if LOCK.exists():
            reason = f"lock held by {owner()!r}"
        elif gpu_used_mib() >= max_used_mib:
            reason = f"GPU memory used {gpu_used_mib()} MiB >= {max_used_mib} without a lock"
        else:
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                reason = "lock appeared during acquisition"
            else:
                os.write(fd, f"{name}\n".encode())
                os.close(fd)
                return time.monotonic() - started
        if reason != last:
            print(f"[gpu_lock] waiting: {reason}", flush=True)
            last = reason
        if max_wait is not None and time.monotonic() - started > max_wait:
            raise TimeoutError(f"GPU not free after {max_wait} s: {reason}")
        time.sleep(poll)


def release(name):
    if owner() == name:
        LOCK.unlink()
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("status", "acquire", "release", "hold-for-unit", "run"))
    ap.add_argument("--owner")
    ap.add_argument("--unit")
    ap.add_argument("--max-wait", type=float)
    ap.add_argument("cmd", nargs="*")
    # Options may come before or after the action (a REMAINDER positional used to swallow `--owner` after the
    # action); everything after the first `--` is the command for `run`, passed through untouched.
    argv = sys.argv[1:]
    tail = []
    if "--" in argv:
        split = argv.index("--")
        argv, tail = argv[:split], argv[split + 1:]
    a = ap.parse_args(argv)
    a.cmd = list(a.cmd) + tail
    if a.action == "status":
        print({"lock_owner": owner(), "gpu_used_mib": gpu_used_mib()})
        return 0
    if not a.owner:
        ap.error("--owner is required")
    if a.action == "acquire":
        print(f"[gpu_lock] acquired by {a.owner} after {acquire(a.owner, a.max_wait):.0f} s", flush=True)
        return 0
    if a.action == "release":
        print(f"[gpu_lock] released={release(a.owner)}", flush=True)
        return 0
    if a.action == "hold-for-unit":
        if not a.unit or owner() != a.owner:
            ap.error("hold-for-unit needs --unit and the lock must already be owned by --owner")
        script = (f"sleep 20; while systemctl --user is-active --quiet {a.unit}; do sleep 30; done; "
                  f"[ \"$(cat {LOCK} 2>/dev/null)\" = \"{a.owner}\" ] && rm -f {LOCK}")
        subprocess.run(["systemd-run", "--user", "--quiet", "--collect", f"--unit=p2-gpulock-{a.owner}"[:120],
                        "/bin/bash", "-c", script], check=True)
        print(f"[gpu_lock] releaser started for unit {a.unit}", flush=True)
        return 0
    cmd = a.cmd
    if not cmd:
        ap.error("run needs a command after --")
    acquire(a.owner, a.max_wait)
    try:
        return subprocess.run(cmd).returncode
    finally:
        release(a.owner)


if __name__ == "__main__":
    sys.exit(main())
