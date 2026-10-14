#!/usr/bin/env python3
"""Reproduce measured single-worker stall, refill capacity, preserve hard guards."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.zhuhai_dagger_parallel import GIB
from harness.zhuhai_dagger_throughput_v11 import ResourceAdmission as Old
from harness.zhuhai_dagger_resource_v12 import ResourceAdmission


class Regression(unittest.TestCase):
    def row(self,available=17,swap=.8818):
        return dict(available=available*GIB,total=32*GIB,gpu_free_bytes=7*GIB,swap_fraction=swap)

    def test_observed_single_worker_stall_can_expand(self):
        row=self.row()
        self.assertFalse(Old(lambda:0.).allowed(row,1))
        admission=ResourceAdmission(lambda:0.)
        self.assertTrue(admission.allowed(row,1))
        self.assertTrue(admission.allowed(self.row(12,.90),2))
        self.assertEqual(admission.block_reasons(row,1),[])

    def test_RAM_GPU_swap_and_fourth_guards_preserved_and_reported(self):
        admission=ResourceAdmission(lambda:0.)
        for available in (3,10,11.69):
            self.assertFalse(admission.allowed(self.row(available),2))
        self.assertTrue(admission.allowed(self.row(11.71),2))
        row=self.row(16,.84)
        self.assertTrue(admission.allowed(row,3))
        row['swap_fraction']=.85
        self.assertFalse(admission.allowed(row,3))
        self.assertIn('fourth_worker_swap_85_percent',admission.block_reasons(row,3))
        for count in range(4):
            self.assertFalse(admission.allowed(self.row(24,.95),count))
        row=self.row();row['gpu_free_bytes']=5*GIB
        self.assertFalse(admission.allowed(row,2))
        self.assertIn('GPU_free_below_6GiB',admission.block_reasons(row,2))
        self.assertFalse(admission.allowed(self.row(24,.7),4))

    def test_pressure_refills_without_half_hour_serial_fallback(self):
        now=[10.];admission=ResourceAdmission(lambda:now[0])
        admission.pressure()
        self.assertEqual(admission.serial_until,0.)
        self.assertFalse(admission.allowed(self.row(),1))
        self.assertIn('resource_cooldown_60_seconds',admission.block_reasons(self.row(),1))
        now[0]=70.
        self.assertTrue(admission.allowed(self.row(),1))
        self.assertTrue(admission.allowed(self.row(12,.90),2))
        self.assertFalse(admission.allowed(self.row(10,.90),2))


if __name__=='__main__':unittest.main()
