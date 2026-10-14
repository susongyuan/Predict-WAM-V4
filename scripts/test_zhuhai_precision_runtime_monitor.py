"""CPU-only runtime handoff regressions; every write is inside a temporary root."""
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo
from scripts import monitor_zhuhai_progress as m
from scripts import progress1_log as journal


class MonitorTests(unittest.TestCase):
    def fixture(self, root):
        run=root/'outputs/run';run.mkdir(parents=True)
        (root/'progress1.md').write_text('<!-- active_training_run: outputs/run -->\n## 关键操作记录\n')
        (run/'status.json').write_text(json.dumps(dict(status='running',active_stage='tiny_admission')))
        tiny=run/'tiny_admission';tiny.mkdir()
        (tiny/'status.json').write_text(json.dumps(dict(status='preparing_TRAIN_statistics',step=0)))
        (run/'plan.json').write_text(json.dumps(dict(stages=[dict(name='tiny_admission',kind='train',out=str(tiny),command=['p','--cache-root',str(root/'data/cache')])])))
        return run

    def test_external_stop_overrides_preserved_running_status_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root);original=(run/'status.json').read_bytes()
            (run/'external_stop_receipt.json').write_text(json.dumps(dict(status='stopped_by_user_requested_v2_replacement',formal_optimizer_updates=0)))
            with patch.object(m,'ROOT',root):result=m.snapshot()
            self.assertEqual(result['pipeline_status'],'stopped_by_user_requested_v2_replacement')
            self.assertIsNone(result['active_stage']);self.assertEqual(result['formal_step'],0)
            self.assertEqual((run/'status.json').read_bytes(),original)

    def test_inactive_bound_unit_overrides_stale_running_without_rewriting_evidence(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root);original=(run/'status.json').read_bytes()
            f=root/'progress1.md';f.write_text(f.read_text()+'<!-- active_training_unit: p2-test-v2.service -->\n')
            reply=SimpleNamespace(returncode=0,stdout='ActiveState=inactive\nSubState=dead\nMainPID=0\nResult=signal\n',stderr='')
            with patch.object(m,'ROOT',root),patch.object(m.subprocess,'run',return_value=reply):result=m.snapshot()
            self.assertEqual(result['pipeline_status'],'interrupted_unit_inactive')
            self.assertEqual(result['runtime_unit']['MainPID'],'0')
            self.assertEqual((run/'status.json').read_bytes(),original)

    def test_resume_preparation_cannot_erase_saved_optimizer_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root);train=run/'train';train.mkdir()
            (train/'status.json').write_text(json.dumps(dict(status='preparing_TRAIN_statistics',step=0)))
            (train/'checkpoint_last.pt').write_bytes(b'fake')
            (train/'checkpoint_last.pt.sha256.json').write_text(json.dumps(dict(step=20000,size=4)))
            with patch.object(m,'ROOT',root):result=m.snapshot()
            self.assertEqual(result['formal_step'],20000)
            (train/'checkpoint_last.pt').write_bytes(b'wrong_size')
            with patch.object(m,'ROOT',root):result=m.snapshot()
            self.assertEqual(result['formal_step'],0)

    def test_continuation_reads_original_train_and_tiny_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root);origin=root/'outputs/origin';(origin/'train').mkdir(parents=True)
            (origin/'train/status.json').write_text(json.dumps(dict(status='stopped_at_safe_boundary',step=20000)))
            (origin/'tiny_admission').mkdir();(origin/'tiny_admission/status.json').write_text(json.dumps(dict(status='completed',step=2)))
            plan=json.loads((run/'plan.json').read_text());plan['training_origin']=str(origin);(run/'plan.json').write_text(json.dumps(plan))
            with patch.object(m,'ROOT',root):result=m.snapshot()
            self.assertEqual(result['formal_step'],20000);self.assertEqual(result['tiny_step'],2)
            self.assertEqual(result['training_artifact_run'],str(origin))

    def test_direct_training_monitor_finishes_on_true30000_not_stale_controller_state(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root);origin=root/'outputs/origin';(origin/'train').mkdir(parents=True)
            plan=json.loads((run/'plan.json').read_text());plan.update(training_origin=str(origin),monitor_mode='direct_training');(run/'plan.json').write_text(json.dumps(plan))
            (origin/'train/status.json').write_text(json.dumps(dict(status='training',step=25100,target_steps=30000)))
            with patch.object(m,'ROOT',root):value=m.snapshot()
            self.assertEqual(value['pipeline_status'],'running');self.assertEqual(value['active_stage'],'train_to_30000')
            (origin/'train/status.json').write_text(json.dumps(dict(status='completed',step=30000,target_steps=30000)))
            with patch.object(m,'ROOT',root):value=m.snapshot()
            self.assertEqual(value['pipeline_status'],'completed');self.assertIsNone(value['active_stage'])

    def test_tiny_progress_is_not_formal_training_or_physical_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root)
            (run/'tiny_admission/status.json').write_text(json.dumps(dict(status='completed',step=2)))
            with patch.object(m,'ROOT',root):result=m.snapshot()
            self.assertEqual(result['tiny_step'],2);self.assertEqual(result['formal_step'],0)
            self.assertEqual(result['milestone_results'],[])


class BoundedJournalTests(unittest.TestCase):
    def test_48_windows_preserve_full_journal_and_archive_while_markdown_stays_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'progress1.md').write_text('# Test\n## 累计进展摘要\n- Essential history.\n\n## 关键操作记录\n')
            start=datetime(2026,10,10,tzinfo=ZoneInfo('Asia/Shanghai'))
            with patch.object(journal,'ROOT',root):
                for k in range(48):
                    with patch.object(journal,'datetime') as clock:
                        clock.now.return_value=start+timedelta(seconds=601*k)
                        clock.fromisoformat.side_effect=datetime.fromisoformat
                        journal.log(f'window {k}',actor='cpu_fixture')
                    self.assertLess(len((root/'progress1.md').read_text().splitlines()),200)
                with patch.object(journal,'datetime') as clock:
                    clock.now.return_value=start+timedelta(seconds=601*47+1)
                    clock.fromisoformat.side_effect=datetime.fromisoformat
                    self.assertFalse(journal.log('same-window detail',actor='cpu_fixture'))
            text=(root/'progress1.md').read_text()
            self.assertIn('Essential history.',text)
            self.assertEqual(text.count('- 滚动记录压缩摘要：'),1)
            self.assertEqual(text.count(' — 10分钟汇总：'),26)  # 24 live plus first/last in compressed entry
            events=(root/'docs/progress_archive/progress1_operations.jsonl').read_text().splitlines()
            self.assertEqual(len(events),49)
            import zipfile
            with zipfile.ZipFile(root/'docs/archive/documentation_snapshot_20261007.zip') as z:
                self.assertEqual(len(z.namelist()),24)


if __name__=='__main__':unittest.main(verbosity=2)
