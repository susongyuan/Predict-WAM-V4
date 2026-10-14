"""CPU fixtures only: quota transition keeps physical evidence admission strict."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import collect_isaac_zhuhai_200 as new
from scripts import collect_isaac_zhuhai as old
from scripts.test_zhuhai_pipeline import fixture_contract, fixture_row
from scripts import handoff_zhuhai_200 as handoff


class QuotaTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source_contract=fixture_contract()
        self.source_rows=[fixture_row(self.root, self.source_contract, 1000+i) for i in range(100)]
        self.source=self.root/'source.json'
        old.write_once(self.source, dict(status='complete', requested_success_count=100, success_count=100,
            contract=self.source_contract, attempts=self.source_rows))
        self.original_bytes=self.source.read_bytes()
        self.contract=deepcopy(self.source_contract)
        self.contract.update(mode='randomized', requested_success_count=200,
            production_quota=dict(new.PRODUCTION_QUOTAS, total_per_task=300, total_all_tasks=600),
            source_manifest=dict(path=str(self.source), sha256=old.sha256(self.source)),
            collection_implementation=dict(path=str(Path(new.__file__).resolve()), sha256=old.sha256(new.__file__)),
            source_quota_transition=dict(from_quota=old.PRODUCTION_QUOTAS, to_quota=new.PRODUCTION_QUOTAS, source_relabelled=False))
        self.rows=[fixture_row(self.root, self.contract, 10000+i, parent=1000+i%100) for i in range(200)]
        self.dr=self.root/'dr.json'; self.write()

    def write(self):
        self.dr.write_text(json.dumps(dict(status='complete', requested_success_count=200,
            success_count=200, contract=self.contract, attempts=self.rows)))

    def test_original_source_keeps_old_quota_and_hash(self):
        self.assertEqual(len(new.validate_manifest(self.source, task='zhuhai_task1', mode='source')),100)
        self.assertEqual(len(new.validate_manifest(self.dr, task='zhuhai_task1', mode='randomized')),200)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        self.assertEqual(old.PRODUCTION_QUOTAS['randomized'],500)

    def test_wrong_family_balance_rejected(self):
        row=self.rows[-1]; row['family_id']=self.rows[0]['family_id']; row['source_parent']=1000
        row['metadata'].update(family_id=row['family_id'],parent_source_seed=1000)
        self.write()
        with self.assertRaisesRegex(ValueError,'two successful relatives'):
            new.validate_manifest(self.dr, task='zhuhai_task1', mode='randomized')

    def test_unsafe_success_rejected(self):
        self.rows[0]['initial_rule_evidence']['physics_safety']['passed']=False; self.write()
        with self.assertRaises(ValueError):new.validate_manifest(self.dr, task='zhuhai_task1', mode='randomized')

    def test_changed_collector_rejected(self):
        self.contract['collection_implementation']['sha256']='0'*64; self.write()
        with self.assertRaisesRegex(ValueError,'implementation'):
            new.validate_manifest(self.dr, task='zhuhai_task1', mode='randomized')

    def test_changed_source_manifest_rejected(self):
        self.source.write_bytes(self.original_bytes+b' ')
        with self.assertRaisesRegex(ValueError,'source manifest changed'):
            new.validate_manifest(self.dr, task='zhuhai_task1', mode='randomized')

    def test_reused_pid_is_not_live(self):
        with patch.object(handoff,'identity',return_value=dict(start_ticks=20,state='S')):
            self.assertFalse(handoff.live(dict(pid=123,start_ticks=10)))

if __name__ == '__main__':unittest.main()
