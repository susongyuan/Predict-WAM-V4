import unittest
from harness.zhuhai_dagger_parallel import can_expand,GIB


class ResourceTests(unittest.TestCase):
    def test_cold_start_has_an_additional_runtime_ram_reserve(self):
        row=dict(available=11*GIB,total=32*GIB,gpu_free_bytes=7*GIB,swap_fraction=.6)
        self.assertTrue(can_expand(row,1))
        for change in (dict(available=9*GIB),dict(gpu_free_bytes=5*GIB),dict(swap_fraction=.95)):
            self.assertFalse(can_expand(dict(row,**change),1))

    def test_fourth_worker_and_maximum_are_distinct(self):
        row=dict(available=20*GIB,total=32*GIB,gpu_free_bytes=9*GIB,swap_fraction=.91)
        self.assertTrue(can_expand(row,2))
        self.assertFalse(can_expand(row,3))
        self.assertFalse(can_expand(dict(row,swap_fraction=.1),4))


if __name__=='__main__':unittest.main()
