#!/usr/bin/env python3
"""CPU-only AST-isolated check: never imports simulator/torch or uses GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
SOURCE = Path(__file__).resolve().parents[1] / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/eval/evaluator.py"
class CacheTests(unittest.TestCase):
    def function(self, initialized):
        tree = ast.parse(SOURCE.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "release_unused_cuda_cache")
        self.cuda = Mock()
        self.cuda.is_initialized.return_value = initialized
        self.cuda.memory_allocated.side_effect = [123, 123]
        self.cuda.memory_reserved.side_effect = [4000, 128]
        self.gc = Mock()
        ns = {"th": SimpleNamespace(cuda=self.cuda), "gc": self.gc}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), ns)
        return ns[function.name]
    def test_no_context_creation(self):
        self.assertEqual(self.function(False)(), {"cuda_initialized": False})
        self.cuda.empty_cache.assert_not_called()
        self.cuda.memory_allocated.assert_not_called()
        self.gc.collect.assert_not_called()
    def test_no_live_tensor_or_rng_mutation(self):
        row = self.function(True)()
        self.assertEqual(row["before_bytes"], {"allocated":123,"reserved":4000})
        self.assertEqual(row["after_bytes"], {"allocated":123,"reserved":128})
        self.assertEqual([c[0] for c in self.cuda.mock_calls], ["is_initialized", "memory_allocated", "memory_reserved", "empty_cache", "memory_allocated", "memory_reserved"])
        self.gc.collect.assert_called_once_with()
    def test_boundary_before_policy_after_sensor_settings(self):
        src = SOURCE.read_text()
        self.assertLess(src.index("self._apply_robot_eval_settings()"), src.index("cache_receipt = release_unused_cuda_cache()"))
        self.assertLess(src.index("cache_receipt = release_unused_cuda_cache()"), src.index("self.policy = self.load_policy()"))
if __name__ == "__main__": unittest.main()
