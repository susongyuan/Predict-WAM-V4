"""CPU-only resource protection sampling contract; no GPU/scene."""
import unittest
from unittest.mock import patch
from harness.robotwin_eval_evidence import ResourceGuard
from harness.robotwin_policy_worker import InfrastructureError

class WindowTests(unittest.TestCase):
    def scenario(self, times, ticks, memory='MemTotal: 1000 kB\nMemAvailable: 700 kB\n'):
        stack = __import__('contextlib').ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('harness.robotwin_eval_evidence.time.monotonic', side_effect=times))
        stack.enter_context(patch.object(ResourceGuard, 'ticks', side_effect=ticks))
        stack.enter_context(patch('harness.robotwin_eval_evidence.Path.read_text', return_value=memory))
        return ResourceGuard()
    def test_short_spike_accumulates_not_overload(self):
        g=self.scenario([0,.01,1.01],[(0,0),(10,0),(3200,3000)])
        self.assertIsNone(g.check()['system_cpu_percent'])
        self.assertEqual(g.previous,(0,0))
        self.assertAlmostEqual(g.check()['system_cpu_percent'],6.25)
    def test_sustained_overload_still_fails(self):
        g=self.scenario([0,1.01],[(0,0),(3200,100)])
        with self.assertRaises(InfrastructureError): g.check()
    def test_memory_immediate_even_short_window(self):
        g=self.scenario([0,.001],[(0,0),(0,0)],'MemTotal: 1000 kB\nMemAvailable: 100 kB\n')
        with self.assertRaises(InfrastructureError): g.check()
    def test_no_tick_window_does_not_claim_idle(self):
        g=self.scenario([0,1.1],[(0,0),(0,0)])
        self.assertIsNone(g.check()['system_cpu_percent'])
        self.assertEqual(g.previous_time,0)
    def test_baseline_resets_only_after_valid_window(self):
        g=self.scenario([0,1.1,1.11,2.2],[(0,0),(3200,3000),(3210,3000),(6400,6000)])
        self.assertAlmostEqual(g.check()['system_cpu_percent'],6.25)
        self.assertIsNone(g.check()['system_cpu_percent'])
        self.assertAlmostEqual(g.check()['system_cpu_percent'],6.25)

if __name__=='__main__': unittest.main()
