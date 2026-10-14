#!/usr/bin/env python3
"""Bounded CPU prefetch with logical RNG state independent of speculative reads."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_precision_future16_memory as runtime

class ParallelRamFeatures:
    def __init__(self, store, budget_bytes=8*1024**3):
        if not 0 <= budget_bytes <= 8*1024**3:
            raise ValueError('8GiB feature RAM cap')
        self.store, self.budget_bytes = store, budget_bytes
        self.entries, self.used = OrderedDict(), 0
        self.lock = threading.RLock()
        self.stats = dict(hits=0, misses=0, evictions=0, resident_bytes=0, peak_resident_bytes=0)

    def get(self, episode, frame):
        key = (episode.sha256, int(frame))
        with self.lock:
            if key in self.entries:
                self.stats['hits'] += 1
                value, size = self.entries.pop(key)
                self.entries[key] = (value, size)
                return value
            self.stats['misses'] += 1
        value = self.store.get(episode, frame)
        size = (sum(sys.getsizeof(x) for x in value) + sys.getsizeof(value) +
                sys.getsizeof(key) + sum(sys.getsizeof(x) for x in key) + 128)
        with self.lock:
            if key in self.entries:
                return self.entries[key][0]
            if size + sys.getsizeof(self.entries) <= self.budget_bytes:
                while self.entries and self.used + size + sys.getsizeof(self.entries) > self.budget_bytes:
                    _, (_, previous) = self.entries.popitem(last=False)
                    self.used -= previous
                    self.stats['evictions'] += 1
                self.entries[key] = (value, size)
                self.used += size
                while self.entries and self.used + sys.getsizeof(self.entries) > self.budget_bytes:
                    _, (_, previous) = self.entries.popitem(last=False)
                    self.used -= previous
                    self.stats['evictions'] += 1
            resident = self.used + (sys.getsizeof(self.entries) if self.entries else 0)
            self.stats['resident_bytes'] = resident
            self.stats['peak_resident_bytes'] = max(self.stats['peak_resident_bytes'], resident)
        return value

class BatchPrefetch:
    def __init__(self, dataset, rng, batch_size, workers, factory):
        self.dataset, self.rng, self.batch_size = dataset, rng, batch_size
        self.factory = factory
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='future16-data')
        self.pending = None
        self.current_indices, self.current_futures, self.cursor = [], [], 0
        self.awaiting_sample = None

    @property
    def bit_generator(self):
        return self.rng.bit_generator

    def prefetch_next(self):
        if self.pending is not None:
            return
        shadow = self.factory(0)
        shadow.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        indices = [int(shadow.integers(len(self.dataset))) for _ in range(self.batch_size)]
        self.pending = (indices, [self.pool.submit(self.dataset.__getitem__, i) for i in indices])

    def integers(self, high):
        if int(high) != len(self.dataset) or self.awaiting_sample is not None:
            raise ValueError('Unexpected sampling call while prefetching')
        if self.cursor == len(self.current_indices):
            self.prefetch_next()
            indices, futures = self.pending
            actual = [int(self.rng.integers(high)) for _ in range(self.batch_size)]
            if actual != indices:
                raise ValueError('Prefetch indices diverged from logical training RNG')
            self.pending = None
            self.current_indices, self.current_futures, self.cursor = indices, futures, 0
        index = self.current_indices[self.cursor]
        self.awaiting_sample = index
        return index

    def get(self, index):
        if self.awaiting_sample is None:
            return self.dataset[index]  # Deterministic diagnostics do not consume sampling RNG.
        if index != self.awaiting_sample:
            raise ValueError('Prefetched dataset read out of order')
        future = self.current_futures[self.cursor]
        value = future.result()
        self.current_futures[self.cursor] = None
        self.cursor += 1
        self.awaiting_sample = None
        return value

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending = None

class DatasetProxy:
    def __init__(self, dataset):
        self.dataset, self.prefetch = dataset, None
    def __len__(self): return len(self.dataset)
    def __getattr__(self, name): return getattr(self.dataset, name)
    @property
    def condition(self): return self.dataset.condition
    @condition.setter
    def condition(self, value): self.dataset.condition = value
    def __getitem__(self, index):
        return self.prefetch.get(index) if self.prefetch else self.dataset[index]

def install(profile):
    import numpy as np
    from harness import zhuhai_training as training
    from harness import zhuhai_hybrid_cache as cache
    from harness import zhuhai_precision_v4_future16 as future
    from harness import zhuhai_precision_v4_training as v4
    import torch.utils.data
    runtime.install_profile(profile)
    original_binding = runtime.bind_contract
    def bind(contract, settings):
        result = original_binding(contract, settings)
        path = Path(__file__).resolve()
        result['source_files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['hjepa']['code_hashes'] = copy.deepcopy(result['source_files'])
        return result
    runtime.bind_contract = bind
    training.BoundedRamFeatures = ParallelRamFeatures
    verified_cache = cache.HybridFeatureCache
    class ThreadSafeReadCache(verified_cache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.writer_lock = threading.RLock()
        def _locked(self):
            # The original fallback writer shares a connection field. Serialize misses.
            from contextlib import contextmanager
            @contextmanager
            def locked():
                with self.writer_lock:
                    with super(ThreadSafeReadCache, self)._locked():
                        yield
            return locked()
    cache.HybridFeatureCache = ThreadSafeReadCache
    original_datasets = future.build_datasets
    selected = {'dataset': None, 'prefetch': None}
    def datasets(*args, **kwargs):
        result = original_datasets(*args, **kwargs)
        result['train'][future.TASK] = DatasetProxy(result['train'][future.TASK])
        selected['dataset'] = result['train'][future.TASK]
        return result
    future.build_datasets = datasets
    prior_install = future.install
    def future_install():
        prior_install()
        v4.build_datasets = datasets
    future.install = future_install
    factory = np.random.default_rng
    def new_rng(seed=None):
        rng = factory(seed)
        ds = selected['dataset']
        if seed == profile['training_seed'] and ds is not None and selected['prefetch'] is None:
            prefetch = BatchPrefetch(ds.dataset, rng, profile['batch_transition']['new_batch_size'],
                                     profile['data_workers'], factory)
            ds.prefetch = selected['prefetch'] = prefetch
            return prefetch
        return rng
    np.random.default_rng = new_rng
    collate = torch.utils.data.default_collate
    def ready(batch):
        result = collate(batch)
        prefetch = selected['prefetch']
        if prefetch is not None and len(batch) == prefetch.batch_size:
            prefetch.prefetch_next()
        return result
    torch.utils.data.default_collate = ready
    return selected

def main():
    profile = json.loads(Path(sys.argv[2]).read_text())
    assert profile['data_workers'] == 16
    # Install CPU prefetch first; runtime.main would otherwise install the profile twice.
    selected = install(profile)
    runtime.install_profile = lambda unused: None
    try:
        runtime.main()
    finally:
        if selected['prefetch'] is not None:
            selected['prefetch'].close()

if __name__ == '__main__':
    main()
