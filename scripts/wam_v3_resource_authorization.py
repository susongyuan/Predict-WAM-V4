#!/usr/bin/env python3
"""CPU-only validation of the actual 2026-09-30 GPU-peak receipt.

A caller supplies a receipt path, never an approval boolean or a numeric override.
The pinned bytes are the actual user-message receipt reviewed for this migration;
this is not a general-purpose mechanism for self-issuing GPU authorization.
Resource permission is separate from training approval and GPU-lock ownership.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'outputs/validation/task0_gpu_peak_authorization_20260930/authorization.json'
RECEIPT_SHA256 = '9bd06b2fbdda711378c8e31b2b7605da3fd3ed0453e6ebf2c797925ec5c69cad'
GPU_LOCK = Path('/tmp/project2_gpu.lock')
USER_TEXT = '允许gpu超过85%的限制，允许达到峰值，我觉得测试和训练应该可以并行吧'


def validate_authorization(path):
    if not path:
        raise ValueError('continuation GPU execution blocked: resource-window authorization unresolved; explicit GPU authorization receipt required')
    candidate = ROOT / path
    if candidate.is_symlink() or candidate.resolve() != RECEIPT.resolve():
        raise ValueError('GPU authorization must name the pinned actual-user receipt, not a substitute')
    payload = candidate.read_bytes()
    if hashlib.sha256(payload).hexdigest() != RECEIPT_SHA256:
        raise ValueError('GPU authorization receipt SHA256 mismatch')
    receipt = json.loads(payload)
    expected = {
        'format': 'project2-gpu-peak-authorization-v1',
        'source': 'actual_user_message_in_current_session',
        'user_text': USER_TEXT,
        'gpu_peak_above_85_percent_allowed': True,
        'gpu_utilization_percent_allowed': 100,
        'operational_total_vram_abort_mib': 15300,
        'cpu_percent_ceiling_unchanged': 85,
        'ram_percent_ceiling_unchanged': 85,
        'training_simulation_gpu_concurrency': 'not_admitted_without_measured_combined_headroom',
        'initial_execution_policy': 'exclusive GPU lock; serialize simulation and training; CPU analysis may run concurrently',
    }
    for key, value in expected.items():
        if type(receipt.get(key)) is not type(value) or receipt[key] != value:
            raise ValueError(f'GPU authorization receipt field differs: {key}')
    return {
        'format': 'wam-v3-gpu-resource-contract-v1',
        'authorization': str(RECEIPT),
        'authorization_sha256': RECEIPT_SHA256,
        'gpu_utilization_percent_allowed': 100,
        'total_vram_abort_mib': 15300,
        'cpu_percent_ceiling': 85,
        'ram_percent_ceiling': 85,
        'gpu_lock': str(GPU_LOCK),
        'training_simulation_concurrency': False,
        'monitoring_is_hard_quota': False,
        'training_approval_implied': False,
        'training_throttle_unchanged': True,
    }


def validate_resource_contract(contract):
    if not isinstance(contract, dict):
        return validate_authorization(None)  # Historical CPU-only plans stay non-executable.
    expected = validate_authorization(contract.get('authorization'))
    if contract != expected:
        raise ValueError('GPU resource plan differs from the pinned authorization budget')
    return expected


def require_unlocked():
    if GPU_LOCK.exists() or GPU_LOCK.is_symlink():
        raise ValueError('external GPU lock is already held; no waiting, stealing or auto-queuing')


def resource_environment(contract):
    validated = validate_resource_contract(contract)
    # These values are provenance for downstream launchers, not environment-based permission.
    # Consumers must validate GPU_AUTHORIZATION independently before admitting execution.
    return {
        'GPU_AUTHORIZATION': validated['authorization'],
        'P2_GPU_AUTHORIZATION_SHA256': validated['authorization_sha256'],
        'P2_GPU_UTILIZATION_ALLOWED': '100',
        'P2_TOTAL_VRAM_ABORT_MIB': '15300',
        'P2_CPU_PERCENT_CEILING': '85',
        'P2_RAM_PERCENT_CEILING': '85',
        'P2_GPU_CONCURRENCY': 'exclusive',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--check-lock', action='store_true')
    args = parser.parse_args()
    try:
        contract = validate_authorization(args.authorization)
        if args.check_lock:
            require_unlocked()
        print(json.dumps(contract, sort_keys=True))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f'GPU resource authorization: {error}\n')


if __name__ == '__main__':
    main()
