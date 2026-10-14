import json
from datetime import datetime,timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo
from scripts import progress1_log as p

class LogTests(unittest.TestCase):
    def test_throttle_archives_every_event_and_preserves_rules(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);doc=root/'progress1.md';doc.write_text('# 规则与累计摘要\n旧失败保留\n\n## 关键操作记录\n')
            now=datetime(2026,10,9,20,0,tzinfo=ZoneInfo('Asia/Shanghai'))
            with patch.object(p,'ROOT',root),patch.object(p,'datetime') as clock:
                clock.fromisoformat=datetime.fromisoformat;clock.now.return_value=now
                self.assertTrue(p.log('开始预处理','test'));before=doc.read_text()
                clock.now.return_value=now+timedelta(seconds=599)
                self.assertFalse(p.log('资源失败保留','worker'));self.assertEqual(doc.read_text(),before)
                clock.now.return_value=now+timedelta(seconds=600)
                self.assertTrue(p.log('停止后续','test'));self.assertIn('资源失败保留',doc.read_text())
                for i in range(1,29):
                    clock.now.return_value=now+timedelta(seconds=600*(i+1));p.log(f'阶段{i}','test')
            self.assertLess(len(doc.read_text().splitlines()),200)
            self.assertIn('旧失败保留',doc.read_text());self.assertIn('滚动记录压缩摘要',doc.read_text())
            rows=[json.loads(x) for x in (root/'docs/progress_archive/progress1_operations.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),31);self.assertEqual(rows[1]['message'],'资源失败保留')
    def test_missing_section_fails_without_destroying_document(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'progress1.md').write_text('preserved')
            with patch.object(p,'ROOT',root),self.assertRaises(ValueError):p.log('event')
            self.assertEqual((root/'progress1.md').read_text(),'preserved')

if __name__=='__main__':unittest.main()
