"""Regression against the actual v8 sidecar/schema failure, not mocked metadata."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from harness.zhuhai_dagger_screened_pilot_v9 import screened_pilot
from scripts.wait_precision_dagger50_segmented_v8 import admitted_row

SOURCE=Path('outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger50_segmented_20261010_v8/episodes/seed_1017000')


class Tests(unittest.TestCase):
    def test_real_minimal_spans_and_separate_screening_sidecars_admit(self):
        raw=json.loads((SOURCE/'result.json').read_text())
        self.assertNotIn('expert_path_screening',raw['segments'][0])
        pilot=screened_pilot(raw,SOURCE/'result.json',admitted_row(raw,raw['scenario']))
        self.assertEqual((pilot['actual_verified_segments'],pilot['actual_model_handbacks']),(5,5))
        self.assertEqual(sum(r['passed'] for r in pilot['expert_segment_receipts']),5)
        self.assertFalse(pilot['expert_segment_receipts'][-1]['passed'])

    def test_changed_or_missing_teacher_outcome_is_rejected(self):
        raw=json.loads((SOURCE/'result.json').read_text());row=admitted_row(raw,raw['scenario'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'episodes'/'seed_1017000';out.mkdir(parents=True)
            shutil.copyfile(SOURCE.parents[1]/'plan.json',root/'plan.json')
            for file in SOURCE.glob('expert_segment_*.json'):shutil.copyfile(file,out/file.name)
            bad=out/'expert_segment_00.json';content=json.loads(bad.read_text());content['passed']=False;bad.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError,'outcome differs'):screened_pilot(raw,out/'result.json',row)
            bad.unlink()
            with self.assertRaises(FileNotFoundError):screened_pilot(raw,out/'result.json',row)

    def test_legacy_without_support_screening_cannot_unlock_four_workers(self):
        self.assertIsNone(screened_pilot({},SOURCE/'result.json',{}))


if __name__=='__main__':unittest.main()
