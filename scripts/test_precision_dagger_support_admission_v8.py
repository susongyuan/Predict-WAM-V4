"""Real archived rejection must not turn into recovery or training credit."""
import json
from pathlib import Path
import tempfile
import unittest
import h5py
from harness.zhuhai_dagger_support_admission_v8 import digest, probe_evidence, verify_pair

SOURCE = Path('outputs/zhuhai2026_tasks12/verification/dagger50_segmented_20261010_v8/physical_support_probe')


class Tests(unittest.TestCase):
    def test_real_safe_rejection_is_auditable_but_not_a_reachable_goal(self):
        result, _ = probe_evidence(SOURCE/'result.json')
        self.assertFalse(result['passed'])
        with self.assertRaisesRegex(ValueError, 'reachable local goal'):
            verify_pair(SOURCE/'result.json', SOURCE/'result.json')

    def test_production_or_quota_credit_cannot_be_admitted_as_verification(self):
        for field, value in [('production', True), ('quota_credit', 1)]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = json.loads((SOURCE/'result.json').read_text())
                result[field] = value
                (root/'result.json').write_text(json.dumps(result))
                (root/'plan.json').write_bytes((SOURCE/'plan.json').read_bytes())
                with self.assertRaisesRegex(ValueError, 'independently archived'):
                    probe_evidence(root/'result.json')

    def test_a_single_unsafe_observation_invalidates_an_otherwise_safe_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'probe.hdf5'
            with h5py.File(archive, 'w') as stream:
                stream.attrs['physics_safety_all_observations_passed'] = True
                stream.attrs['source_observation_count'] = 2
                stream.create_dataset('rule_evidence', data=[
                    json.dumps(dict(physics_safety=dict(passed=True))),
                    json.dumps(dict(physics_safety=dict(passed=False)))],
                    dtype=h5py.string_dtype())
            result = json.loads((SOURCE/'result.json').read_text())
            result.update(data_path=str(archive), data_sha256=digest(archive), control_steps=1)
            (root/'result.json').write_text(json.dumps(result))
            (root/'plan.json').write_bytes((SOURCE/'plan.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Unsafe observed frame'):
                probe_evidence(root/'result.json')


if __name__ == '__main__':
    unittest.main()
