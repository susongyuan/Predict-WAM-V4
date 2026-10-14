"""Bounded, indexed derived-feature cache; raw demonstrations are never evicted.

``read_bytes(identity, generator)`` returns encoded npz bytes. Alternatively use
``with get_or_compute(...) as path``: the path is pinned only within that lease.
Keys bind source, frozen DINO encoder/checkpoint, preprocessing, frame and schema;
the checkpoint identity must never refer to updating WAM weights. Normal access
uses SQLite key/LRU indices and cumulative usage, without directory scans. A
startup recovery or explicit audit scans owned files. SQLite, metadata and write
files count toward the cap; in-memory SQLite journals avoid disk journal peaks.
Zero/oversize requests use read_bytes online in memory without dropping frames.
Path leases require disk admission: there is no budget-exempt temporary spill.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from typing import Callable, Iterator

_REQUIRED = ("source_sha256", "encoder_sha256", "preprocess_sha256",
             "checkpoint_sha256", "frame_index", "feature_schema_version")
_ENTRY = re.compile(r"^[0-9a-f]{64}\.(payload|json)$")
_TMP = re.compile(r"^\.write-[0-9a-f]{64}-[0-9a-z_]+$")
_INDEX = "index.sqlite3"
_OWNER = ".zhuhai-derived-feature-cache-v1"
HARD_DISK_BUDGET_BYTES = 10**12
MINIMUM_FREE_BYTES = 100 * 1024**3


class HybridFeatureCache:
    def __init__(self, root, budget_bytes=HARD_DISK_BUDGET_BYTES, min_free_bytes=MINIMUM_FREE_BYTES,
                 online_temp_root=None):
        if type(budget_bytes) is not int or not 0 <= budget_bytes <= HARD_DISK_BUDGET_BYTES:
            raise ValueError("Disk cache hard budget is 1TB = 10^12 bytes, not 1TiB")
        if type(min_free_bytes) is not int or min_free_bytes < 0:
            raise ValueError("Free-space reserve must be a nonnegative integer")
        # A smaller reserve is only useful for isolated CPU fixtures. Production
        # entry points enforce >=100GiB, and cannot override the 1TB hard cap.
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.budget_bytes = int(budget_bytes); self.min_free_bytes = int(min_free_bytes)
        self.online_temp_root = Path(online_temp_root).resolve() if online_temp_root else self.root.parent
        self.stats = dict(hits=0, misses=0, evictions=0, corruptions=0, online=0,
                          directory_scans=0, preserved_unowned=0)
        self._lock_path = self.root / ".lock"
        self._index_path = self.root / _INDEX
        self._startup_done = False; self._pending_bytes = 0; self._db = None
        self._preserved_bytes = 0; self._ownership_checked = False
        self._owner_path = self.root / _OWNER
        if any(p.is_symlink() for p in (self._lock_path, self._index_path, self._owner_path)):
            raise ValueError("Cache lock/index must not be symlinks")

    @staticmethod
    def _identity(identity):
        missing = [x for x in _REQUIRED if x not in identity]
        if missing: raise ValueError(f"Missing cache identities: {missing}")
        for name in _REQUIRED[:4]:
            if not isinstance(identity[name], str) or not re.fullmatch(r"[0-9a-f]{64}", identity[name]):
                raise ValueError(f"{name} must be a lowercase SHA256 digest")
        if identity['frame_index'] is None or not identity['feature_schema_version']:
            raise ValueError("Frame/chunk index and schema required")
        encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest(),json.loads(encoded)

    def _paths(self,key): return self.root/(key+'.payload'),self.root/(key+'.json')

    @staticmethod
    def _allocated(path):
        try:
            if path.is_symlink(): raise ValueError(f"Cache symlink forbidden: {path}")
            s=path.stat()
            if not path.is_file() or s.st_nlink != 1:
                raise ValueError(f"Managed cache file must be regular and not hardlinked: {path}")
            return max(s.st_size,s.st_blocks*512)
        except FileNotFoundError:return 0

    def _proven_metadata(self, key, meta):
        info = json.loads(meta.read_text())
        actual, _ = self._identity(info['identity'])
        if info.get('key') != key or actual != key:
            raise ValueError('Not owned derived-feature provenance')
        return info

    def _ownership(self):
        if self._ownership_checked:
            return
        if self._owner_path.exists():
            if self._allocated(self._owner_path) != 0:
                raise ValueError('Unknown cache ownership marker: preserve and choose another root')
        else:
            if self._index_path.exists():
                # Migrate only a recognized legacy index, never overwrite an
                # unrelated SQLite database/weight/evidence named index.sqlite3.
                conn = None
                recognized = False
                try:
                    conn = sqlite3.connect(self._index_path.as_uri() + '?mode=ro', uri=True)
                    columns = [r[1] for r in conn.execute('PRAGMA table_info(entries)')]
                    state_columns = [r[1] for r in conn.execute('PRAGMA table_info(state)')]
                    recognized = columns == ['key', 'allocated', 'accessed'] and state_columns == ['id', 'used']
                except sqlite3.DatabaseError:
                    # Corrupt legacy indices may be rebuilt only when valid
                    # feature provenance independently establishes ownership.
                    for meta in self.root.glob('*.json'):
                        if _ENTRY.fullmatch(meta.name) and not meta.is_symlink():
                            try:
                                self._proven_metadata(meta.stem, meta)
                                recognized = True
                                break
                            except (ValueError, KeyError, TypeError, OSError):
                                pass
                finally:
                    if conn is not None:
                        conn.close()
                if not recognized:
                    raise ValueError('Unowned cache index preserved; use a separate derived-cache directory')
            self._owner_path.touch(exist_ok=False)  # Empty marker: no allocated payload bytes.
        self._ownership_checked = True

    def _connect(self, memory=False):
        conn=sqlite3.connect(':memory:' if memory else str(self._index_path))
        conn.execute('PRAGMA journal_mode=MEMORY');conn.execute('PRAGMA synchronous=FULL')
        if not memory:
            page=conn.execute('PRAGMA page_size').fetchone()[0]
            conn.execute(f'PRAGMA max_page_count={max(8,self.budget_bytes//page)}')
        conn.execute('CREATE TABLE IF NOT EXISTS entries(key TEXT PRIMARY KEY, allocated INTEGER NOT NULL, accessed INTEGER NOT NULL)')
        conn.execute('CREATE INDEX IF NOT EXISTS entries_lru ON entries(accessed)')
        conn.execute('CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY, used INTEGER NOT NULL)')
        conn.execute('INSERT OR IGNORE INTO state VALUES(1,0)');conn.commit()
        return conn

    def _used_payloads(self):
        return self._db.execute('SELECT used FROM state WHERE id=1').fetchone()[0]

    def _usage(self):
        return self._used_payloads()+self._allocated(self._index_path)+self._allocated(self._lock_path)+self._pending_bytes+self._preserved_bytes+self._allocated(self._owner_path)

    def _delete(self,key):
        if not re.fullmatch(r'[0-9a-f]{64}', key):
            raise ValueError('Invalid indexed cache key; refuse path traversal')
        for p in self._paths(key):
            self._allocated(p)  # Reject symlink/hardlink replacement, never follow it.
        row=self._db.execute('SELECT allocated FROM entries WHERE key=?',(key,)).fetchone()
        for p in self._paths(key):p.unlink(missing_ok=True)
        if row:
            self._db.execute('DELETE FROM entries WHERE key=?',(key,))
            self._db.execute('UPDATE state SET used=used-? WHERE id=1',(row[0],));self._db.commit()

    def _register(self,key,allocated,accessed):
        old=self._db.execute('SELECT allocated FROM entries WHERE key=?',(key,)).fetchone()
        self._db.execute('INSERT OR REPLACE INTO entries VALUES(?,?,?)',(key,allocated,accessed))
        self._db.execute('UPDATE state SET used=used+? WHERE id=1',(allocated-(old[0] if old else 0),))
        self._db.commit()

    def _recovery(self):
        self.stats['directory_scans'] += 1
        owned = {}
        inventory = list(self.root.iterdir())
        for p in inventory:
            if p.is_symlink():
                raise ValueError(f'Cache symlink forbidden: {p}')
            if p.is_dir():
                raise ValueError(f'Derived cache must be flat: {p}')
        indexed = {r[0] for r in self._db.execute('SELECT key FROM entries')}
        self._preserved_bytes = 0
        for p in inventory:
            if _TMP.fullmatch(p.name):
                # A name alone is not proof of ownership after a crash. Count
                # but preserve unknown/orphan writes; never delete source/evidence.
                self._preserved_bytes += self._allocated(p)
                self.stats['preserved_unowned'] += 1
            elif _ENTRY.fullmatch(p.name):
                owned.setdefault(p.stem, {})[p.suffix] = p
        self._db.execute('DELETE FROM entries'); self._db.execute('UPDATE state SET used=0 WHERE id=1'); self._db.commit()
        for key, pair in owned.items():
            proven = False
            try:
                meta = pair['.json']
                self._allocated(meta)
                info = self._proven_metadata(key, meta)
                proven = True
                payload = pair['.payload']
                self._allocated(payload)
                if info['size'] != payload.stat().st_size:
                    raise ValueError('Invalid payload length')
            except (OSError, ValueError, KeyError, TypeError):
                if proven or key in indexed:
                    for p in pair.values():
                        self._allocated(p)
                        p.unlink(missing_ok=True)
                    self.stats['corruptions'] += 1
                else:
                    self._preserved_bytes += sum(self._allocated(p) for p in pair.values())
                    self.stats['preserved_unowned'] += len(pair)
                continue
            allocated = self._allocated(payload) + self._allocated(meta)
            if not self._admit(allocated):
                payload.unlink(missing_ok=True); meta.unlink(missing_ok=True)
                self.stats['evictions'] += 1
                continue
            self._register(key, allocated, payload.stat().st_mtime_ns)

    def _reconcile(self):
        while self._usage()>self.budget_bytes:
            oldest=self._db.execute('SELECT key FROM entries ORDER BY accessed LIMIT 1').fetchone()
            if oldest:
                self._delete(oldest[0]);self.stats['evictions']+=1
            else:
                # A shrunken cap may not even fit the empty index. Remove that
                # owned index and retain only an in-memory index/online mode.
                self._db.close();self._index_path.unlink(missing_ok=True)
                self._db=self._connect(memory=True)
                if self._usage()>self.budget_bytes:
                    raise OSError('Preserved unowned files exceed cache cap; choose a clean cache root')
                break

    @contextmanager
    def _locked(self):
        with self._lock_path.open('a+b') as h:
            fcntl.flock(h,fcntl.LOCK_EX)
            try:
                self._ownership()
                existing=self._index_path.exists()
                # Empty SQLite requires four pages. Do not create a disk index
                # that could by itself violate a tiny/zero budget.
                memory=not existing and (self.budget_bytes<32768 or not self._space(32768))
                try:self._db=self._connect(memory=memory)
                except sqlite3.DatabaseError:
                    self._index_path.unlink(missing_ok=True)
                    memory=self.budget_bytes<32768 or not self._space(32768)
                    self._db=self._connect(memory=memory)
                    self._startup_done=False
                if not self._startup_done or (not existing and not memory):
                    self._recovery();self._startup_done=True
                self._reconcile()
                page=self._db.execute('PRAGMA page_size').fetchone()[0]
                max_pages=max(1,(self.budget_bytes-self._used_payloads()-self._preserved_bytes
                    -self._allocated(self._lock_path)-self._allocated(self._owner_path))//page)
                self._db.execute(f'PRAGMA max_page_count={max_pages}')
                yield
            finally:
                if self._db is not None:self._db.close();self._db=None
                fcntl.flock(h,fcntl.LOCK_UN)

    def usage_bytes(self):
        with self._locked():return self._usage()

    def audit(self):
        """Explicit directory reconciliation; payload digests also checked on hit."""
        with self._locked():
            self._recovery();self._reconcile()
            return dict(usage_bytes=self._usage(),entries=self._db.execute('SELECT COUNT(*) FROM entries').fetchone()[0])

    def _hit(self,key,identity):
        if not self._db.execute('SELECT 1 FROM entries WHERE key=?',(key,)).fetchone():return None
        payload,meta=self._paths(key)
        for p in (payload,meta):self._allocated(p)
        try:
            info=json.loads(meta.read_text());data=payload.read_bytes()
            if info['identity']!=identity or info['key']!=key or info['size']!=len(data) or info['payload_sha256']!=hashlib.sha256(data).hexdigest():
                raise ValueError('Provenance/payload integrity mismatch')
            stamp=time.time_ns();os.utime(payload,ns=(stamp,stamp))
            if not self._space(32768):
                return data  # No LRU B-tree growth when the free-space floor is threatened.
            try:
                self._db.execute('UPDATE entries SET accessed=? WHERE key=?',(stamp,key));self._db.commit()
            except sqlite3.OperationalError as error:
                self._db.rollback()
                if 'full' not in str(error).lower():raise
                # The feature remains usable if the budget leaves no room for
                # an LRU B-tree split; keep its previous indexed access time.
            return data
        except (OSError,ValueError,KeyError,TypeError):
            self._delete(key);self.stats['corruptions']+=1;return None

    def _space(self,required):return shutil.disk_usage(self.root).free>=self.min_free_bytes+required

    def _admit(self,required):
        # Reserve bounded B-tree page growth before writing payload or metadata;
        # SQLite max_page_count below supplies an additional hard allocation cap.
        reserve=32768
        if not self._index_path.exists() or required+reserve+self._allocated(self._index_path)>self.budget_bytes:return False
        while self._usage()+required+reserve>self.budget_bytes or not self._space(required+reserve):
            oldest=self._db.execute('SELECT key FROM entries ORDER BY accessed LIMIT 1').fetchone()
            if not oldest:return False
            self._delete(oldest[0]);self.stats['evictions']+=1
        page=self._db.execute('PRAGMA page_size').fetchone()[0]
        max_pages=(self.budget_bytes-self._used_payloads()-self._preserved_bytes-required
            -self._allocated(self._lock_path)-self._allocated(self._owner_path))//page
        self._db.execute(f'PRAGMA max_page_count={max_pages}')
        return True

    def _compute_locked(self,identity,generator):
        key,identity=self._identity(identity);data=self._hit(key,identity)
        if data is not None:self.stats['hits']+=1;return key,data,self._paths(key)[0]
        self.stats['misses']+=1;data=generator()
        if not isinstance(data,(bytes,bytearray,memoryview)):raise TypeError('Generator must return bytes')
        data=bytes(data)
        if any(p.exists() or p.is_symlink() for p in self._paths(key)):
            # Unregistered names are not disposable cache entries.
            self.stats['online']+=1;return key,data,None
        metadata=json.dumps(dict(key=key,identity=identity,size=len(data),payload_sha256=hashlib.sha256(data).hexdigest()),sort_keys=True,separators=(',',':')).encode()
        block=os.statvfs(self.root).f_frsize or 4096
        rounded=lambda n:((n+block-1)//block)*block
        if not self._admit(rounded(len(data))+rounded(len(metadata))):
            self.stats['online']+=1;return key,data,None
        payload,meta=self._paths(key);temporary=[];written=[];created=[]
        try:
            for target,content in ((payload,data),(meta,metadata)):
                with tempfile.NamedTemporaryFile(prefix=f'.write-{key}-',dir=self.root,delete=False) as h:
                    p=Path(h.name);temporary.append(p);h.write(content);h.flush();os.fsync(h.fileno())
                written.append(p);self._pending_bytes=sum(self._allocated(p) for p in written)
                if self._usage()>self.budget_bytes:raise RuntimeError('Filesystem allocation exceeded budget')
                os.link(p,target)  # Exclusive publication; never replace an unowned collision.
                created.append(target);p.unlink();written[-1]=target
            stamp=time.time_ns();os.utime(payload,ns=(stamp,stamp))
            self._register(key,self._pending_bytes,stamp);self._pending_bytes=0
            if self._usage()>self.budget_bytes:raise RuntimeError('Index exceeded reserved budget')
            return key,data,payload
        except BaseException:
            self._db.rollback()
            for p in created:p.unlink(missing_ok=True)
            if self._db.execute('SELECT 1 FROM entries WHERE key=?',(key,)).fetchone():self._delete(key)
            raise
        finally:
            self._pending_bytes=0
            for p in temporary:p.unlink(missing_ok=True)

    def read_bytes(self,identity:dict,generator:Callable[[],bytes])->bytes:
        with self._locked():return self._compute_locked(identity,generator)[1]

    @contextmanager
    def get_or_compute(self,identity:dict,generator:Callable[[],bytes])->Iterator[Path]:
        """Pin an admitted path; use read_bytes for zero/oversize/low-disk cases.

        No temporary directory outside the cache can bypass its hard disk cap.
        The generated bytes remain available through read_bytes without CUDA,
        frame dropping, or resampling; training already uses that byte API.
        """
        with self._locked():
            _,data,path=self._compute_locked(identity,generator)
            if path is None:
                raise OSError('No bounded disk lease; use read_bytes for in-memory online features')
            yield path
