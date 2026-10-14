#!/usr/bin/env python3
"""Run a command with NVIDIA libraries matching the loaded kernel driver."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/zhuhai2026_tasks12/verification/dagger_driver_repair_20261011_v20/local_driver_bundle.json'


def kernel_version():
    match = re.search(r'\b\d+\.\d+\.\d+\b', Path('/proc/driver/nvidia/version').read_text())
    if match is None:
        raise RuntimeError('Cannot identify the loaded NVIDIA kernel driver')
    return match.group()


def healthy(environment, version):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                env=environment, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    versions = result.stdout.split()
    return result.returncode == 0 and bool(versions) and all(v == version for v in versions)


def choose_environment(environment):
    environment = dict(environment)
    version = kernel_version()
    if healthy(environment, version):
        return environment
    bundle = json.loads(BUNDLE.read_text())
    if bundle['kernel_version'] != version or bundle['userspace_version'] != version:
        raise RuntimeError(f'No verified local NVIDIA libraries for loaded driver {version}')
    for filename, expected in bundle['file_hashes'].items():
        digest = hashlib.sha256()
        with Path(filename).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError('Local NVIDIA library changed: ' + filename)
    previous = environment.get('LD_LIBRARY_PATH', '')
    environment.update(bundle['environment'])
    if previous:
        environment['LD_LIBRARY_PATH'] += ':' + previous
    if not healthy(environment, version):
        raise RuntimeError('Matched NVIDIA libraries still fail; inspect GPU/kernel errors before launching')
    return environment


def main():
    command = sys.argv[1:]
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        raise SystemExit('Usage: python3 scripts/with_project_gpu.py -- COMMAND [ARGS...]')
    try:
        environment = choose_environment(os.environ)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit('GPU environment rejected: ' + str(exc)) from exc
    os.execvpe(command[0], command, environment)


if __name__ == '__main__':
    main()
