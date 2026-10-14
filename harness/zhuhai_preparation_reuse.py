"""Immutable TRAIN preparation reuse; does not change the production pipeline.

Builders retain the upstream signatures. conditions() requires statistics()
to have committed the matching value first. Every new process verifies TRAIN
HDF SHA + lowdim content once; unchanged file-stat tokens reuse that verification
within the instance (no RGB decoding). Orphan attempts are preserved, never
adopted without the final commit pointer. No model imports or feature reads here.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback
import uuid

import numpy as np
from harness import zhuhai_training as zt

_SCHEMA = 'zhuhai-train-preparation-reuse-v1'
_LIMIT = 16 * 1024**2
_FIELDS = ('left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states')
_SOURCES = ('harness/zhuhai_training.py', 'harness/zhuhai_dataset.py', 'harness/wam_v3.py',
            'harness/wam_behavior.py', 'harness/robotwin_training_data.py', 'harness/zhuhai_demo_recording.py')


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def _directory(path):
    if path.is_symlink():
        raise ValueError(f'cache directory symlink forbidden: {path}')
    path.mkdir(parents=True, exist_ok=True)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(path, value):
    raw = _bytes(value)
    if len(raw) > _LIMIT:
        raise ValueError('preparation JSON exceeds bounded metadata limit')
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    os.link(temporary, path)  # fails if destination already exists; never replaces old evidence
    temporary.unlink()  # only this successful publication's own temporary link
    _sync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f'untrusted preparation file: {path}')
    if path.stat().st_size > _LIMIT:
        raise ValueError('preparation JSON exceeds bounded metadata limit')
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(_bytes([array.dtype.str, list(array.shape)]))
    h.update(memoryview(array).cast('B'))
    return h.hexdigest()


def _token(path):
    s = path.stat()
    return (str(path), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _source_hashes():
    # Conservative file-level dependencies; NOT the trainer/optimizer/run config.
    return {name: zt.sha256_file(zt.ROOT / name) for name in _SOURCES}


class PreparationReuse:
    def __init__(self, root, *, lock_timeout=30):
        if type(lock_timeout) not in (int, float) or not math.isfinite(lock_timeout) or lock_timeout < 0:
            raise ValueError('lock_timeout must be finite and nonnegative')
        self.root = Path(root).expanduser().absolute()
        _directory(self.root)
        self.lock_timeout = lock_timeout
        self._operations, self._verified = [], {}

    @contextmanager
    def _locked(self, kind, check):
        operation = dict(kind=kind, outcome='checking', started_ns=time.time_ns())
        self._operations.append(operation)
        try:
            fd = os.open(self.root / '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'a+b') as handle:
                deadline = time.monotonic() + self.lock_timeout
                try:
                    while True:
                        check()
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError('preparation lock timeout; existing builder/evidence untouched')
                            time.sleep(min(.05, max(0., deadline - time.monotonic())))
                    yield operation
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        except BaseException as exc:
            operation.update(outcome='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            operation['finished_ns'] = time.time_ns()

    def _identity(self, datasets, cfg, feature_identity, check):
        import h5py
        provenance = zt.train_provenance(datasets, cfg)
        if (set(feature_identity) != {'encoder_sha256', 'preprocess_sha256'}
                or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in feature_identity.values())):
            raise ValueError('explicit encoder/preprocessing SHA256 required')
        for ds in datasets.values():
            for ep in ds.episodes:
                check()
                path = Path(ep.path)
                if path.is_symlink():
                    raise ValueError('TRAIN source symlink forbidden')
                token = _token(path)
                if token not in self._verified:
                    source_sha = zt.sha256_file(path)
                    with h5py.File(path, 'r') as handle:
                        lowdim = {kind: _array_hash(np.concatenate([handle[f'{kind}/{f}'][:] for f in _FIELDS], axis=1))
                                  for kind in ('state', 'action')}
                    if token != _token(path):
                        raise ValueError('TRAIN source changed during verification')
                    self._verified[token] = (source_sha, lowdim)
                source_sha, lowdim = self._verified[token]
                if source_sha != ep.sha256 or any(lowdim[k] != _array_hash(getattr(ep, k)) for k in lowdim):
                    raise ValueError('TRAIN file or in-memory state/action differs from admitted content')
        return json.loads(_bytes(dict(schema=_SCHEMA, train_provenance=provenance,
            feature_identity=feature_identity, views=cfg['model']['views'],
            proprio_lags=cfg['model']['v3']['proprio_lags'], memory_lags=cfg['model']['v3']['memory_lags'],
            algorithm_source_sha256=_source_hashes())))

    def _statistics_valid(self, value, identity, cfg):
        provenance = zt.validate_statistics(value, cfg)
        if (any(provenance.get(k) != v for k, v in identity['train_provenance'].items())
                or provenance.get('feature_identity') != identity['feature_identity']):
            raise ValueError('statistics provenance differs from exact current TRAIN content/features')

    def _stage(self, kind, identity):
        key = zt.digest(identity)
        home = self.root / key
        _directory(home); _directory(home / kind)
        return key, home / kind

    def _load(self, kind, identity, validate):
        key, home = self._stage(kind, identity)
        pointer = home / 'COMMITTED.json'
        if not pointer.exists() and not pointer.is_symlink():
            return None
        marker, _ = _read(pointer)
        if (not isinstance(marker, dict) or marker.get('schema') != _SCHEMA or marker.get('key') != key or marker.get('kind') != kind
                or not isinstance(marker.get('attempt'), str) or not re.fullmatch(r'attempt-[0-9a-f]{32}', marker.get('attempt', ''))):
            raise ValueError('invalid preparation commit pointer')
        attempt = home / marker['attempt']
        if attempt.is_symlink() or not attempt.is_dir():
            raise ValueError('invalid committed attempt directory')
        receipt, receipt_sha = _read(attempt / 'receipt.json')
        value, payload_sha = _read(attempt / 'payload.json')
        if (receipt_sha != marker.get('receipt_sha256') or receipt.get('schema') != _SCHEMA
                or receipt.get('identity') != identity or receipt.get('kind') != kind or receipt.get('key') != key
                or receipt.get('payload_sha256') != payload_sha):
            raise ValueError('preparation receipt/payload hash or input contract mismatch')
        validate(value)
        return value, dict(key=key, attempt=str(attempt), payload_sha256=payload_sha, receipt_sha256=receipt_sha)

    def _obtain(self, kind, identity, validate, build, recheck, check, operation):
        key, home = self._stage(kind, identity)
        operation['key'] = key
        cached = self._load(kind, identity, validate)
        if cached is not None:
            value, info = cached
            check(); recheck()
            operation.update(outcome='hit', **info)
            return value
        # No final pointer: keep all previous attempts, including orphan payloads/receipts.
        operation['ignored_uncommitted_attempts'] = sorted(p.name for p in home.iterdir())
        attempt = home / ('attempt-' + uuid.uuid4().hex)
        attempt.mkdir()
        operation.update(attempt=str(attempt), outcome='building')
        _publish(attempt / 'started.json', dict(schema=_SCHEMA, key=key, kind=kind,
            pid=os.getpid(), state='building', started_ns=time.time_ns(), identity=identity))
        try:
            check(); value = build(); check(); recheck(); validate(value)
            payload_sha = _publish(attempt / 'payload.json', value)
            receipt_sha = _publish(attempt / 'receipt.json', dict(schema=_SCHEMA, key=key,
                kind=kind, identity=identity, payload_sha256=payload_sha))
            _publish(home / 'COMMITTED.json', dict(schema=_SCHEMA, key=key, kind=kind,
                attempt=attempt.name, receipt_sha256=receipt_sha))
            operation.update(outcome='built', payload_sha256=payload_sha, receipt_sha256=receipt_sha)
            return json.loads(_bytes(value))
        except BaseException as exc:
            try:
                _publish(attempt / 'failed.json', dict(error=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc()[-65536:], state='failed', failed_ns=time.time_ns()))
            except Exception:
                pass  # started.json remains durable; the OS releases flock even on process death.
            raise

    def statistics(self, train_datasets, cfg, *, feature_identity, builder, check=lambda: None):
        with self._locked('statistics', check) as operation:
            identity = self._identity(train_datasets, cfg, feature_identity, check)
            def recheck():
                if self._identity(train_datasets, cfg, feature_identity, check) != identity:
                    raise ValueError('TRAIN/algorithm identity changed during preparation')
            return self._obtain('statistics', identity, lambda x: self._statistics_valid(x, identity, cfg),
                lambda: builder(train_datasets, cfg, feature_identity=feature_identity, check=check),
                recheck, check, operation)

    def conditions(self, all_datasets, cfg, statistics, *, builder, check=lambda: None):
        with self._locked('conditions', check) as operation:
            if set(all_datasets) != {'train', 'val', 'heldout'}:
                raise ValueError('explicit train/val/heldout split containers required')
            feature = statistics['normalization']['provenance']['feature_identity']
            base = self._identity(all_datasets['train'], cfg, feature, check)
            self._statistics_valid(statistics, base, cfg)
            saved = self._load('statistics', base, lambda x: self._statistics_valid(x, base, cfg))
            if saved is None or zt.digest(saved[0]) != zt.digest(statistics):
                raise ValueError('conditions require the matching committed statistics; call statistics() first')
            targets = [(task, ds) for split in all_datasets.values() for task, ds in split.items()]
            if any(task not in all_datasets['train'] for task, _ in targets):
                raise ValueError('every split task must have a TRAIN condition')
            identity = dict(base, statistics_sha256=zt.digest(statistics))
            def validate(bank):
                zt.validate_conditions(bank, statistics, cfg)
                if bank.get('provenance') != base['train_provenance']:
                    raise ValueError('VTT provenance differs from exact current TRAIN content')
            def recheck():
                if self._identity(all_datasets['train'], cfg, feature, check) != base or zt.digest(statistics) != identity['statistics_sha256']:
                    raise ValueError('TRAIN/statistics/algorithm changed during VTT preparation')
            bank = self._obtain('conditions', identity, validate,
                lambda: builder(all_datasets, cfg, statistics), recheck, check, operation)
            for task, ds in targets:
                ds.condition = np.asarray(bank['conditions'][task], np.float32).copy()
            return bank

    def receipt(self):
        return json.loads(_bytes(dict(schema=_SCHEMA, root=str(self.root), lock_timeout=self.lock_timeout,
            operations=self._operations, source_files_verified_in_this_instance=len(self._verified))))
