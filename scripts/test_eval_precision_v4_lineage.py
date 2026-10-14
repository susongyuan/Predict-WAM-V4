#!/usr/bin/env python3
"""Admission regression: a performance fork cannot weaken model/source checks."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_precision_v4_future16_lineage as lineage


class LineageTests(unittest.TestCase):
    def fixture(self):
        initial = dict(v4_training=dict(schedule=dict(batch_size=4, steps=50000)),
                       source_files={'model.py': 'frozen'}, hjepa=dict(code_hashes={'model.py': 'frozen'}),
                       model=dict(horizons=[16, 32], execution_prefix=8))
        current = copy.deepcopy(initial)
        current['v4_training']['schedule']['batch_size'] = 23
        current['runtime_performance_profile'] = dict(batch_transition=dict(old_batch_size=4,
                new_batch_size=23, at_updates=5000))
        for mapping in (current['source_files'], current['hjepa']['code_hashes']):
            mapping.update({key: 'new-performance-wrapper' for key in lineage.PERFORMANCE_SOURCES})
        return initial, current

    def test_only_recorded_performance_transition_is_admitted(self):
        lineage.validate_performance_fork(*self.fixture())

    def test_model_change_is_rejected(self):
        initial, current = self.fixture()
        current['model']['execution_prefix'] = 16
        with self.assertRaises(lineage.queue.e.EvidenceError):
            lineage.validate_performance_fork(initial, current)

    def test_changed_frozen_code_is_rejected(self):
        initial, current = self.fixture()
        current['source_files']['model.py'] = 'changed'
        with self.assertRaises(lineage.queue.e.EvidenceError):
            lineage.validate_performance_fork(initial, current)

    def test_unknown_source_addition_is_rejected(self):
        initial, current = self.fixture()
        current['source_files']['different_policy.py'] = 'new'
        with self.assertRaises(lineage.queue.e.EvidenceError):
            lineage.validate_performance_fork(initial, current)

    def test_wrong_transition_point_is_rejected(self):
        initial, current = self.fixture()
        current['runtime_performance_profile']['batch_transition']['at_updates'] = 6000
        with self.assertRaises(lineage.queue.e.EvidenceError):
            lineage.validate_performance_fork(initial, current)


if __name__ == '__main__':
    unittest.main()
