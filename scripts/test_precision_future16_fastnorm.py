import math
import unittest
import torch
from scripts.run_precision_future16_fastnorm import LargeRamFeatures, gradient_norm

class Tests(unittest.TestCase):
    def test_same_gradient_metric_for_real_parameter_shapes(self):
        generator = torch.Generator().manual_seed(20261012)
        parameters = [torch.nn.Parameter(torch.zeros(shape)) for shape in ((32,64),(64,),(),(4,7))]
        for p in parameters: p.grad = torch.randn(p.shape, generator=generator)
        expected = float(torch.stack([p.grad.detach().float().square().sum() for p in parameters]).sum().sqrt())
        self.assertAlmostEqual(gradient_norm(parameters), expected, places=4)
        self.assertTrue(all(p.grad is not None for p in parameters))

    def test_nonfinite_and_missing_gradients_remain_detectable(self):
        p = torch.nn.Parameter(torch.ones(3))
        self.assertEqual(gradient_norm([p]), 0.)
        for value in (float('nan'), float('inf')):
            p.grad = torch.full_like(p, value)
            self.assertFalse(math.isfinite(gradient_norm([p])))

    def test_explicit_ram_cap(self):
        self.assertEqual(LargeRamFeatures(object()).budget_bytes,16*1024**3)
        with self.assertRaises(ValueError): LargeRamFeatures(object(),17*1024**3)

if __name__ == '__main__': unittest.main()
