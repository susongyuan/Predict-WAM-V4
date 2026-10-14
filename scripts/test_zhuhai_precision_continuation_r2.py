"""CPU-only r2 adoption capability guards; never spawn training or simulation."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_zhuhai_wam_precision_v2_continuation_r2 as r


class AdoptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
        self.summary=self.out/'summary.json';self.summary.write_text('{}')
        self.plan=dict(out=str(self.out))
        self.stage=dict(kind='eval',step=20000,precompleted_summary=True,completed_summary_path=str(self.summary),completed_summary_sha256=r.prior.h.sha256_file(self.summary))
    def test_adopt_completed20k_without_spawning_any_process(self):
        with patch.object(r.prior,'verify_stage',return_value={'status':'completed'})as verify,patch.object(r.prior,'run_stage',side_effect=AssertionError('must never rerun')):
            self.assertEqual(r.run_stage(self.plan,self.stage),0);verify.assert_called_once_with(self.stage)
        receipt=r.prior.h.read_json(self.out/'development_20000_adoption_receipt.json')
        self.assertEqual(receipt['physical_episodes_rerun'],0)
    def test_changed_summary_rejected_no_write(self):
        self.summary.write_text('{"modified":1}')
        with patch.object(r.prior,'verify_stage')as verify:
            with self.assertRaises(ValueError):r.run_stage(self.plan,self.stage)
            verify.assert_not_called()
    def test_no_adoption_of_other_steps_or_training(self):
        for changes in [dict(step=25000),dict(kind='train')]:
            stage=dict(self.stage,**changes)
            with self.assertRaises(ValueError):r.run_stage(self.plan,stage)
    def test_future_stages_use_original_frozen_commands_runner(self):
        stage=dict(kind='train',step=25000)
        with patch.object(r.prior,'run_stage',return_value=0)as run:
            self.assertEqual(r.run_stage(self.plan,stage),0);run.assert_called_once_with(self.plan,stage)
    def test_duplicate_adoption_receipt_not_silently_overwritten(self):
        with patch.object(r.prior,'verify_stage',return_value={}):
            r.run_stage(self.plan,self.stage)
            p=self.out/'development_20000_adoption_receipt.json';p.write_text('{"different":1}')
            with self.assertRaises(ValueError):r.run_stage(self.plan,self.stage)


if __name__=='__main__':unittest.main(verbosity=2)
