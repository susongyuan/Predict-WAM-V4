#!/usr/bin/env python3
"""Exercise runtime-check assertions with small real V3/V4 CPU models."""
import unittest

from scripts.test_wam_precision_v4 import make_pair
from scripts.check_precision_v4_runtime import verify_runtime
import torch


class RuntimeCheckTests(unittest.TestCase):
    def test_runtime_assertions_on_real_small_models(self):
        torch.set_num_threads(1)
        v3, v4, batch = make_pair()
        result = verify_runtime(v3, v4, batch)
        self.assertTrue(result['alpha_zero_ten_euler_exact'])
        self.assertGreater(result['predictor_gradient_norm'], 0.)
        self.assertEqual(result['calls_per_replan']['predictor'], 1)
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
