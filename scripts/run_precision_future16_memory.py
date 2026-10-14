#!/usr/bin/env python3
"""Runtime-only swap reserve adjustment; preserve all frozen training sources."""
from pathlib import Path
import copy
import hashlib
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_zhuhai_wam_precision_v2 as legacy

ORIGINAL_RESERVE = legacy.reserve
SWAP_STOP_FRACTION = .98

def memory():
    return dict((key, int(value.split()[0])) for key, value in
        (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()))

def reserve(args, *, startup=False):
    try:
        return ORIGINAL_RESERVE(args, startup=startup)
    except RuntimeError as exc:
        if str(exc) != '95% swap reserve reached':
            raise
        mem = memory()
        # The original owner, disk and RAM checks have already passed.
        if not mem['SwapTotal'] or 1 - mem['SwapFree'] / mem['SwapTotal'] < SWAP_STOP_FRACTION:
            return
        raise RuntimeError('98% swap reserve reached') from exc

def main():
    if len(sys.argv) < 5 or sys.argv[1] != '--profile' or sys.argv[3] != '--entry':
        raise ValueError('Use --profile PROFILE --entry FROZEN_FUTURE16_TRAINER followed by original arguments')
    profile_path = Path(sys.argv[2]).resolve()
    profile = json.loads(profile_path.read_text())
    entry = Path(sys.argv[4]).resolve()
    if entry != ROOT / 'scripts/train_zhuhai_wam_precision_v4_future16.py':
        raise ValueError('Only the current frozen Future16 trainer is supported')
    install_profile(profile)
    sys.argv = [str(entry)] + sys.argv[5:]
    legacy.reserve = reserve
    runpy.run_path(str(entry), run_name='__main__')

def bind_contract(contract, profile):
    contract = copy.deepcopy(contract)
    contract['runtime_performance_profile'] = copy.deepcopy(profile)
    path = Path(__file__).resolve()
    contract['source_files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    contract['hjepa']['code_hashes'] = copy.deepcopy(contract['source_files'])
    return contract

def install_profile(profile):
    import torch
    from harness import zhuhai_precision_v4_future16 as future
    from harness import zhuhai_precision_v4_training as v4
    from harness import zhuhai_training as training
    from harness import zhuhai_hybrid_cache as cache
    assert profile['swap_stop_fraction'] == .98
    assert profile['torch_cpu_threads'] == 8
    assert profile['cuda_memory_fraction'] == .95
    assert 0 <= profile['ram_feature_cache_bytes'] <= 8 * 1024**3
    original_threads = torch.set_num_threads
    original_fraction = torch.cuda.set_per_process_memory_fraction
    torch.set_num_threads = lambda requested: original_threads(profile['torch_cpu_threads'])
    torch.cuda.set_per_process_memory_fraction = lambda requested: original_fraction(profile['cuda_memory_fraction'])
    original_ram = training.BoundedRamFeatures
    class RuntimeRamFeatures(original_ram):
        def __init__(self, store, budget_bytes=2*1024**3):
            assert budget_bytes == profile['ram_feature_cache_bytes']
            super().__init__(store, min(budget_bytes, 2*1024**3))
            self.budget_bytes = budget_bytes
    training.BoundedRamFeatures = RuntimeRamFeatures
    class VerifiedReadCache(cache.HybridFeatureCache):
        def read_bytes(self, identity, generator):
            # Immutable, SHA-verified hits need no SQLite LRU commit or directory scan.
            # A miss still uses the original bounded writer and its recovery checks.
            key, normalized = self._identity(identity)
            payload, metadata = self._paths(key)
            try:
                self._allocated(payload); self._allocated(metadata)
                info = json.loads(metadata.read_text())
                content = payload.read_bytes()
                if (info['identity'] != normalized or info['key'] != key or info['size'] != len(content)
                        or info['payload_sha256'] != hashlib.sha256(content).hexdigest()):
                    raise ValueError('Provenance/payload integrity mismatch')
            except (OSError, ValueError, KeyError, TypeError):
                return super().read_bytes(identity, generator)
            self.stats['hits'] += 1
            return content
    cache.HybridFeatureCache = VerifiedReadCache
    original_install = future.install
    original_contract = future.new_contract
    def install():
        original_install()
        v4.new_contract = lambda base: bind_contract(original_contract(base), profile)
    future.install = install

if __name__ == '__main__':
    main()
