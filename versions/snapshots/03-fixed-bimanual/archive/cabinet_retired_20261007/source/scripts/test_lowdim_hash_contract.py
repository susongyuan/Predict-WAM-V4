#!/usr/bin/env python3
"""Standalone CPU regression for lowdim index ARRAY hashes versus whole .npy FILE hashes.

Default: small synthetic cases + real Task-0 episode 0 (action and state).
Optional --all-episodes: additionally check all 200 Task-0 episodes / 400 arrays.

  PYTHONDONTWRITEBYTECODE=1 .conda-envs/lilawam/bin/python scripts/test_lowdim_hash_contract.py
  PYTHONDONTWRITEBYTECODE=1 .conda-envs/lilawam/bin/python scripts/test_lowdim_hash_contract.py --all-episodes

Dataset and trainer are READ ONLY. Synthetic files live in TemporaryDirectory only. No torch/GPU imports,
training, cache rebuild, ledger writes, or edits to existing trainer tests. To exercise the actual trainer
fix without importing its runtime, compile only audit_sources' small `for kind in (action,state)` block
from the current source AST. A refactor that removes that block fails explicitly rather than silently
replacing the production behavior with a test-only implementation.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/lowdim_cache_task000"
TRAINER = ROOT / "scripts/train_wam_v3.py"
ALL_EPISODES = False


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    # cache_behavior_lowdim.py writes SHA256(action.tobytes()) on float32 C-order arrays.
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def identity(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def trainer_check():
    tree = ast.parse(TRAINER.read_text(), filename=str(TRAINER))
    audit = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "audit_sources"), None)
    if audit is None:
        raise AssertionError("audit_sources missing; update this isolated adapter after reviewing the refactor")
    matches = []
    for node in ast.walk(audit):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "kind":
            try:
                kinds = tuple(ast.literal_eval(node.iter))
            except (ValueError, TypeError):
                continue
            if kinds == ("action", "state"):
                matches.append(node)
    if len(matches) != 1:
        raise AssertionError("expected one real lowdim hash loop; do not silently bypass production checks")
    module = ast.parse("def check(ep, info, tp, ledger):\n    pass\n")
    module.body[0].body = [copy.deepcopy(matches[0])]
    namespace = {"np": np, "hashlib": hashlib}
    exec(compile(ast.fix_missing_locations(module), str(TRAINER), "exec"), namespace)
    return namespace["check"]


class ReadOnlyFileLedger:
    """In-memory stand-in for SourceLedger.add's FILE-hash contract. Never writes a ledger."""
    def __init__(self):
        self.files, self.expected_arguments = {}, []

    def add(self, path, expected=None):
        actual = file_sha(path)
        self.expected_arguments.append(expected)
        if expected is not None and actual != expected:
            raise ValueError(f"FILE hash mismatch: {path}")
        self.files[str(path)] = actual
        return actual


class SyntheticHashContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production_check = staticmethod(trainer_check())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lowdim-hash-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.arrays = {"action": np.arange(6 * 23, dtype=np.float32).reshape(6, 23),
                       "state": np.arange(6 * 61, dtype=np.float32).reshape(6, 61)}
        self.info = {"length": 6, **{f"{kind}_sha256": array_sha(a) for kind, a in self.arrays.items()}}
        for kind, array in self.arrays.items():
            np.save(self.path(kind), array, allow_pickle=False)

    def path(self, kind):
        return self.root / f"episode_0_{kind}.npy"

    def check(self, info=None):
        ledger = ReadOnlyFileLedger()
        self.production_check(0, self.info if info is None else info, SimpleNamespace(LOWDIM=self.root), ledger)
        return ledger

    def test_real_trainer_accepts_payload_index_and_records_separate_file_hash(self):
        ledger = self.check()
        self.assertEqual(ledger.expected_arguments, [None, None])
        for kind in self.arrays:
            self.assertNotEqual(self.info[f"{kind}_sha256"], file_sha(self.path(kind)))
            self.assertEqual(ledger.files[str(self.path(kind))], file_sha(self.path(kind)))

    def test_regression_whole_file_hash_must_not_be_accepted_as_payload_index(self):
        info = dict(self.info)
        info["action_sha256"] = file_sha(self.path("action"))
        with self.assertRaises(ValueError):
            self.check(info)

    def test_payload_mutation_is_rejected_for_both_action_and_state(self):
        for kind in ("action", "state"):
            with self.subTest(kind=kind):
                changed = self.arrays[kind].copy()
                changed[0, 0] += 1
                np.save(self.path(kind), changed, allow_pickle=False)
                with self.assertRaises(ValueError):
                    self.check()
                np.save(self.path(kind), self.arrays[kind], allow_pickle=False)

    def test_header_version_change_preserves_array_hash_not_file_hash(self):
        before = file_sha(self.path("action"))
        with self.path("action").open("wb") as handle:
            np.lib.format.write_array(handle, self.arrays["action"], version=(2, 0), allow_pickle=False)
        self.assertNotEqual(file_sha(self.path("action")), before)
        self.assertEqual(array_sha(np.load(self.path("action"), allow_pickle=False)), self.info["action_sha256"])
        self.check()  # valid new header, unchanged array; provenance must retain the new FILE hash

    def test_fortran_storage_still_uses_logical_c_order_payload(self):
        for kind, array in self.arrays.items():
            np.save(self.path(kind), np.asfortranarray(array), allow_pickle=False)
        self.check()

    def test_original_bug_is_reproduced_by_comparing_file_to_index(self):
        with self.assertRaisesRegex(ValueError, "FILE hash mismatch"):
            ReadOnlyFileLedger().add(self.path("action"), self.info["action_sha256"])


class RealDatasetHashContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production_check = staticmethod(trainer_check())
        cls.index_path = DATA / "index.json"
        cls.index = json.loads(cls.index_path.read_text())
        if cls.index.get("task_id") != 0:
            raise AssertionError("this regression test requires the Task-0 lowdim index")

    def verify_episode(self, ep):
        info = self.index["episodes"][str(ep)]
        paths = [DATA / f"episode_{ep}_{kind}.npy" for kind in ("action", "state")]
        before = {p: identity(p) for p in paths + [self.index_path]}
        expected = {}
        for kind, path, width in zip(("action", "state"), paths, (23, 61)):
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            self.assertEqual(value.shape, (info["length"], width))
            self.assertEqual(value.dtype, np.dtype("float32"))
            payload_digest = array_sha(value)
            self.assertEqual(payload_digest, info[f"{kind}_sha256"], f"episode {ep} {kind} payload")
            expected[str(path)] = file_sha(path)
            self.assertNotEqual(expected[str(path)], payload_digest, "file and array are distinct hash domains")
            del value
        ledger = ReadOnlyFileLedger()
        self.production_check(ep, info, SimpleNamespace(LOWDIM=DATA), ledger)
        self.assertEqual(ledger.files, expected)
        self.assertEqual(ledger.expected_arguments, [None, None], "do not compare array hash in FILE ledger.add")
        for path, stamp in before.items():
            self.assertEqual(identity(path), stamp, f"dataset changed during read-only test: {path}")

    def test_real_episode_zero(self):
        self.verify_episode(0)

    def test_optional_all_200_episodes(self):
        if not ALL_EPISODES:
            self.skipTest("opt in with --all-episodes (reads all 400 lowdim .npy files; no DINO features)")
        episodes = sorted(int(ep) for ep in self.index["episodes"])
        self.assertEqual(episodes, list(range(200)), "expected the full Task-0 200-episode inventory")
        for ep in episodes:
            with self.subTest(episode=ep):
                self.verify_episode(ep)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-episodes", action="store_true", help="include all 200 Task-0 episodes, not episode 200")
    args = parser.parse_args()
    global ALL_EPISODES
    ALL_EPISODES = args.all_episodes
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__, fromlist=["*"]))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
