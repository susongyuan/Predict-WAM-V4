"""Reuse a clean, owned, hash-bound cache index after a verified trainer exit.

Payload identity/hash checks, LRU, disk cap and free-space checks remain in the
frozen cache. Any changed index/directory falls back to its full startup scan.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from harness.zhuhai_hybrid_cache import HybridFeatureCache, _ENTRY, _OWNER


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def clean_index_receipt(root, *, budget_bytes, min_free_bytes, last_cache_stats, exited_trainer_pid):
    root = Path(root).resolve()
    if Path('/proc', str(exited_trainer_pid)).exists():
        raise ValueError('Prior feature writer must have exited')
    if any(last_cache_stats.get(k) != 0 for k in ('corruptions', 'preserved_unowned', 'online')):
        raise ValueError('Clean, fully owned cache evidence required')
    with (root/'.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if HybridFeatureCache._allocated(root/_OWNER) != 0:
            raise ValueError('Invalid ownership marker')
        index = root/'index.sqlite3'; HybridFeatureCache._allocated(index)
        with sqlite3.connect(index.as_uri()+'?mode=ro', uri=True) as db:
            rows = db.execute('SELECT key,allocated FROM entries').fetchall()
            used = db.execute('SELECT used FROM state WHERE id=1').fetchone()[0]
        if used != sum(x[1] for x in rows):
            raise ValueError('Index cumulative size differs')
        names = set(); counts = {}
        with os.scandir(root) as inventory:
            for entry in inventory:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError('Unowned directory/link in feature cache')
                if entry.name in ('.lock', _OWNER, 'index.sqlite3'): continue
                if not _ENTRY.fullmatch(entry.name):
                    raise ValueError('Unknown/orphan cache files require normal recovery')
                names.add(entry.name); key, suffix = entry.name.rsplit('.', 1)
                counts.setdefault(key, set()).add(suffix)
        if set(counts) != {x[0] for x in rows} or any(x != {'json', 'payload'} for x in counts.values()):
            raise ValueError('Index and complete owned payload/metadata pairs differ')
        return dict(schema='clean-owned-feature-index-resume-v18', root=str(root),
            index_sha256=digest_file(index), index_size=index.stat().st_size,
            directory_mtime_ns=root.stat().st_mtime_ns, directory_inode=root.stat().st_ino,
            entry_count=len(rows), inventory_sha256=hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest(),
            used_payload_bytes=used, budget_bytes=budget_bytes, min_free_bytes=min_free_bytes,
            last_cache_stats=last_cache_stats, exited_trainer_pid=exited_trainer_pid,
            per_access_payload_and_source_integrity_checks_unchanged=True)


class VerifiedResumeCache(HybridFeatureCache):
    def __init__(self, root, *, resume_receipt, **kwargs):
        super().__init__(root, **kwargs)
        proof = json.loads(Path(resume_receipt).read_text())
        self.resume_index_reused = False
        with self._lock_path.open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (proof.get('schema') == 'clean-owned-feature-index-resume-v18'
                    and proof['root'] == str(self.root)
                    and proof['budget_bytes'] == self.budget_bytes and proof['min_free_bytes'] == self.min_free_bytes
                    and proof['directory_mtime_ns'] == self.root.stat().st_mtime_ns
                    and proof['directory_inode'] == self.root.stat().st_ino
                    and proof['index_size'] == self._index_path.stat().st_size
                    and proof['index_sha256'] == digest_file(self._index_path)):
                self._ownership()
                self._startup_done = True
                self.resume_index_reused = True
