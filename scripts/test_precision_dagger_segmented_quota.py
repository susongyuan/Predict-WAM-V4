"""New episode quota and immutable mixed archive admission, CPU only."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
import h5py
from harness import zhuhai_dagger_segmented_training_v7 as training
from harness.zhuhai_dataset import sha256_file
from scripts.test_precision_dagger_segmented import fixture, old_fixture


def manifest(rows):
    counts = training.correction_counts(rows)
    return dict(status='complete', success_count=counts['segmented'], requested_success_count=50,
        legacy_corrections=counts['legacy'], segmented_corrections=counts['segmented'],
        total_training_corrections=counts['total'], quota_unit='accepted new segmented episodes',
        record_video=False, heldout_used=False, train_rows=rows)


class Tests(unittest.TestCase):
    def test_old_episodes_do_not_consume_new_quota(self):
        old = [dict(correction_variant='legacy_full_continuation') for _ in range(20)]
        new = [dict(correction_variant='segmented_local_interventions') for _ in range(50)]
        self.assertEqual(training.validate_quota(manifest(old+new)), dict(legacy=20, segmented=50, total=70))
        for rows in (old+new[:30], old+new[:49], new, old+new+[new[0]]):
            with self.assertRaisesRegex(ValueError, '50 new segmented'):
                training.validate_quota(manifest(rows))

    def test_many_recoveries_cannot_replace_a_missing_episode(self):
        rows = [dict(correction_variant='legacy_full_continuation') for _ in range(20)]
        rows += [dict(correction_variant='segmented_local_interventions', verified_segment_count=6, handback_count=6) for _ in range(49)]
        m = manifest(rows)
        m.update(success_count=50, segmented_corrections=50, total_training_corrections=70)
        with self.assertRaisesRegex(ValueError, 'segments are not episodes'):
            training.validate_quota(m)

    def test_seventy_real_fixture_archives_admit_and_reject_a_false_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            segmented, legacy = fixture(folder), old_fixture(folder)
            rows = []
            for i in range(70):
                old = i < 20
                source = legacy if old else segmented
                path = folder/f'episode_{i}.hdf5'
                shutil.copyfile(source['path'], path)
                with h5py.File(path, 'r+') as archive:
                    archive.attrs['cpu_fixture_unique_seed'] = i
                rows.append(dict(source, path=str(path), sha256=sha256_file(path), seed=1020000+i,
                    domain='dagger', correction_variant='legacy_full_continuation' if old else 'segmented_local_interventions',
                    policy_steps=3 if old else 6, expert_steps=5 if old else 15, takeover_step=3 if old else 2,
                    verified_segment_count=1 if old else 2, handback_count=0 if old else 2))
            base = dict(splits=dict(zhuhai_task2=dict(train=[dict(family_id='TRAIN-fixture')], val=[], heldout=[])))
            m = manifest(rows)
            self.assertEqual(len(training.validate_rows(base, m)), 70)
            bad = deepcopy(m)
            # Keep metadata totals intact while lying about the archive formats.
            bad['train_rows'][0]['correction_variant'], bad['train_rows'][20]['correction_variant'] = (
                bad['train_rows'][20]['correction_variant'], bad['train_rows'][0]['correction_variant'])
            with self.assertRaisesRegex(ValueError, 'variant differs'):
                training.validate_rows(base, bad)
            bad = deepcopy(m)
            bad['train_rows'][20]['handback_count'] = 3
            with self.assertRaisesRegex(ValueError, 'handback'):
                training.validate_rows(base, bad)

    def test_training_contract_records_seventy_corrections_and_keeps_schedule(self):
        from harness import zhuhai_precision_v2_training as h
        path = Path('outputs/zhuhai2026_tasks12/wam_precision_v2_task2_30k_20261010_v1/train/manifest.json')
        base = h.read_json(path)['contract']
        rows = [dict(correction_variant='legacy_full_continuation') for _ in range(20)]
        rows += [dict(correction_variant='segmented_local_interventions') for _ in range(50)]
        result = training.contract(base, rows, parent={}, collection={}, sources={})
        d = result['dagger_training']
        self.assertEqual((d['correction_episodes'], d['new_segmented_episodes'], d['legacy_correction_episodes']), (70, 50, 20))
        self.assertEqual(d['total_train_episodes'], 230)
        self.assertEqual((d['start_step'], d['target_step']), (30000, 50000))
        self.assertEqual(result['statistics'], base['statistics'])
        self.assertEqual(result['conditions'], base['conditions'])
        self.assertEqual(result['idm_statistics'], base['idm_statistics'])


if __name__ == '__main__':
    unittest.main()
