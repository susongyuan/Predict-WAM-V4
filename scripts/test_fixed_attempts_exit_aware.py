import unittest
from scripts import eval_zhuhai_precision_fixed_attempts_v2 as v
class ExitTests(unittest.TestCase):
    def test_real_abort_exit_zero_or_two_are_not_retried(self):
        for code in(0,2):self.assertTrue(v.acceptable_exit({'status':'control_safety_abort'},code))
    def test_normal_completed_requires_zero(self):
        self.assertTrue(v.acceptable_exit({'status':'completed'},0));self.assertFalse(v.acceptable_exit({'status':'completed'},2))
    def test_crash_or_signal_not_admitted_by_valid_abort(self):
        for code in(-9,-11,1,3,137):self.assertFalse(v.acceptable_exit({'status':'control_safety_abort'},code))
    def test_unknown_or_infrastructure_not_relabelled(self):
        for status in('infrastructure_error','wall_timeout','starting','evidence_rejected'):
            for code in(0,2):self.assertFalse(v.acceptable_exit({'status':status},code))
    def test_adopt_exact_failed25k_path_only(self):
        self.assertIn('continuation_20261010_r2/development_25000',str(v.ADOPT25))
        self.assertEqual(v.ADOPT25.parent.name,'seed_932101')
if __name__=='__main__':unittest.main(verbosity=2)
