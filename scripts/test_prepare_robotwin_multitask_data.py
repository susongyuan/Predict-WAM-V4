#!/usr/bin/env python3
"""CPU synthetic tests for the pilot data preparer; no network, GPU, or simulator."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from scripts import prepare_robotwin_multitask_data as prep


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='robotwin-pilot-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def episode(self, frames=2, bad_rgb=False):
        path = self.root / 'episode_0000000.hdf5'
        with h5py.File(path, 'w') as f:
            f.attrs['source_format'] = 'RoboTwin'
            f['data_format_version'] = 'v1.0'
            f['additional_info/frequency'] = 15
            f['instructions'] = json.dumps(['synthetic contract fixture'])
            for group in ('state', 'action'):
                for field, width in prep.JOINT_FIELDS:
                    f[f'{group}/{field}'] = np.ones((frames, width), dtype=np.float32)
            for view in prep.VIEWS:
                f[f'vision/{view}/shape'] = [240, 320, 3]
                if bad_rgb:
                    f[f'vision/{view}/colors'] = np.zeros((frames, 3), dtype=np.uint8)
                else:
                    f[f'vision/{view}/colors'] = np.zeros((frames, 240, 320, 3), dtype=np.uint8)
            # This malformed non-policy camera must never be read or decoded.
            f['vision/cam_third_view/colors'] = 'ignored original dataset evidence'
        return path

    def member(self, path):
        return {'path': 'source/click_bell/aloha_agilex/data/' + path.name,
                'bytes': path.stat().st_size, 'zip_crc32': prep.file_digest(path)[1]}

    def test_budget_reserve_and_inclusive_boundary(self):
        peak = 9 * prep.GIB
        prep.budget_guard(300 * prep.GIB + peak, 0, prep.GIB)
        with self.assertRaisesRegex(ValueError, 'reserve'):
            prep.budget_guard(300 * prep.GIB + peak - 1, 0, prep.GIB)
        with self.assertRaisesRegex(ValueError, 'budget'):
            prep.budget_guard(1000 * prep.GIB, 42 * prep.GIB, prep.GIB)
        with self.assertRaisesRegex(ValueError, 'size'):
            prep.budget_guard(1000 * prep.GIB, 0, 4 * prep.GIB + 1)

    def test_write_once_idempotent_refuses_changed_artifact(self):
        path = self.root / 'new/split.json'
        prep.write_once(path, {'seed': 1})
        before = path.stat().st_mtime_ns
        prep.write_once(path, {'seed': 1})
        self.assertEqual(before, path.stat().st_mtime_ns)
        with self.assertRaisesRegex(ValueError, 'Refusing'):
            prep.write_once(path, {'seed': 2})
        self.assertEqual(json.loads(path.read_text()), {'seed': 1})

    def test_frozen_split_preserved_and_cross_split_mutation_rejected(self):
        folder = self.root / 'data'
        folder.mkdir()
        for i in range(50):
            (folder / f'episode_{i:07d}.hdf5').write_text(str(i))
        original = prep.make_task_split(folder)
        frozen = self.root / 'frozen.json'
        frozen.write_text(json.dumps(original))
        content = frozen.read_bytes()
        self.assertEqual(prep.make_task_split(folder, frozen), original)
        self.assertEqual(frozen.read_bytes(), content)
        self.assertEqual(original['counts'], {'train': 40, 'val': 5, 'heldout': 5})
        changed = copy.deepcopy(original)
        a, b = changed['splits']['train'], changed['splits']['heldout']
        a[0], b[0] = b[0], a[0]
        frozen.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            prep.make_task_split(folder, frozen)

    def test_split_rejects_aliases_and_missing_episodes(self):
        for i in range(49):
            (self.root / f'episode_{i:07d}.hdf5').write_text(str(i))
        with self.assertRaises(ValueError):
            prep.make_task_split(self.root)
        (self.root / 'episode_0000049.hdf5').symlink_to(self.root / 'episode_0000000.hdf5')
        with self.assertRaises(ValueError):
            prep.make_task_split(self.root)

    def test_native14_first_onboard_rgb_check_excludes_third_view(self):
        p = self.episode()
        row = prep.audit_episode(p, 'train', self.member(p))
        self.assertEqual(row['frames'], 2)
        self.assertTrue(row['lowdim_finite_and_first_onboard_rgb_validated'])
        self.assertEqual(row['ignored_nonpolicy_views'], ['cam_third_view'])
        self.assertFalse(row['heldout_integrity_metadata_only'])

    def test_heldout_never_constructs_semantic_reader(self):
        p = self.episode(bad_rgb=True)
        with patch('harness.robotwin_dataset.RoboTwinEpisode', side_effect=AssertionError('heldout leaked')):
            row = prep.audit_episode(p, 'heldout', self.member(p))
        self.assertTrue(row['heldout_integrity_metadata_only'])
        self.assertFalse(row['lowdim_finite_and_first_onboard_rgb_validated'])

    def test_nonfinite_train_rejected_but_heldout_payload_is_sealed(self):
        p = self.episode()
        with h5py.File(p, 'r+') as f:
            f['state/left_arm_joint_states'][0, 0] = np.nan
        member = self.member(p)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            prep.audit_episode(p, 'train', member)
        self.assertTrue(prep.audit_episode(p, 'heldout', member)['heldout_integrity_metadata_only'])

    def test_official_crc_and_member_size_checked(self):
        p = self.episode()
        member = self.member(p)
        member['zip_crc32'] = '00000000'
        with self.assertRaisesRegex(ValueError, 'CRC32'):
            prep.audit_episode(p, 'train', member)
        member['bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'size'):
            prep.audit_episode(p, 'train', member)

    def test_wrong_embodiment_dimension_rejected_even_heldout(self):
        p = self.episode()
        with h5py.File(p, 'r+') as f:
            del f['action/left_arm_joint_states']
            f['action/left_arm_joint_states'] = np.zeros((2, 7), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'native14'):
            prep.audit_episode(p, 'heldout', self.member(p))

    def test_nonintegral_frequency_and_unknown_split_rejected(self):
        p = self.episode()
        with self.assertRaisesRegex(ValueError, 'Unknown split'):
            prep.audit_episode(p, 'test', self.member(p))
        with h5py.File(p, 'r+') as f:
            del f['additional_info/frequency']
            f['additional_info/frequency'] = 15.5
        with self.assertRaisesRegex(ValueError, 'frequency'):
            prep.audit_episode(p, 'heldout', self.member(p))

    def test_urdf_missing_mesh_fail_closed(self):
        p = self.root / 'robot.urdf'
        p.write_text('<robot><link><visual><geometry><mesh filename="mesh.obj"/></geometry></visual></link></robot>')
        with self.assertRaisesRegex(ValueError, 'Missing URDF mesh'):
            prep.check_urdf(p)
        (self.root / 'mesh.obj').write_text('v 0 0 0\n')
        checked = prep.check_urdf(p)
        self.assertEqual(len(checked['mesh_references_checked']), 1)

    def test_download_whitelist_fail_before_network(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('network called')):
            for task in ('put_object_cabinet', '../escape', 'open_laptop'):
                with self.assertRaisesRegex(ValueError, 'selected official'):
                    prep.fetch_metadata(task)

    def test_partial_existing_source_never_overwritten(self):
        (self.root / 'click_bell').mkdir()
        with patch('subprocess.run', side_effect=AssertionError('download called')):
            with self.assertRaisesRegex(ValueError, 'partial source'):
                prep.ensure_source(self.root, 'click_bell', True, 0)

    def test_source_receipt_requires_native_embodiment(self):
        folder = self.root / 'click_bell'
        folder.mkdir()
        source = dict(status='complete', task='click_bell', repo=prep.REPO,
                      revision=prep.REVISION, archive='demo_clean.zip',
                      expected_sha256='a' * 64, sha256='a' * 64, zip_crc_verified=True,
                      members=[{'path': f'source/click_bell/franka/data/episode_{i:07d}.hdf5'}
                               for i in range(50)])
        (folder / 'SOURCE.json').write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, 'aloha_agilex'):
            prep.read_source(folder, 'click_bell')


if __name__ == '__main__':
    unittest.main(verbosity=2)
