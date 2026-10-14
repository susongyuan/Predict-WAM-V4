#!/usr/bin/env python3
"""CPU-only tests of proposed labels, family isolation and budget arithmetic, not model validation."""
import unittest
import numpy as np
from scripts.audit_zhuhai_wam_design import (FRAME_CACHE_BYTES, coverage, family_split,
                                           label_residuals, tail_eval_steps, valid_endpoint_anchors)


class DesignAuditTests(unittest.TestCase):
    def sources(self):
        return [dict(family_id=f'f{i:03d}', domain='source', path=f's{i}', data_sha256=f'h{i}', action_count=100)
                for i in range(100)]

    def test_terminal_observation_is_valid(self):
        self.assertEqual(valid_endpoint_anchors(32, 32), 1)
        self.assertEqual(valid_endpoint_anchors(32, 8), 25)
        self.assertEqual(valid_endpoint_anchors(32, 4), 29)

    def test_no_clamped_future_or_cross_episode_label(self):
        self.assertEqual(valid_endpoint_anchors(3, 4), 0)
        self.assertEqual(label_residuals(np.zeros((3,16)),np.zeros((3,16))).shape, (0,4,16))

    def test_fixed_measured_anchor_not_future_state_or_target(self):
        q = np.arange(6.)[:,None] * np.ones((1,16))
        u = q + 10
        y = label_residuals(q,u)
        np.testing.assert_allclose(y[0,:,0], [10,11,12,13])
        q[1:] += 100
        np.testing.assert_allclose(label_residuals(q,u)[0], y[0])

    def test_reject_legacy_and_nonfinite(self):
        with self.assertRaises(ValueError): label_residuals(np.zeros((5,14)),np.zeros((5,14)))
        q = np.zeros((5,16)); q[2,0]=np.nan
        with self.assertRaises(ValueError): label_residuals(q,np.zeros((5,16)))

    def test_split_has_exact_source_counts_and_DR_same_family(self):
        source = self.sources()
        dr = [dict(r,domain='randomized',path='d'+r['path'],data_sha256='d'+r['data_sha256']) for r in source]
        splits = family_split(source+dr)
        self.assertEqual([len(splits[s]) for s in ('train','val','heldout')],[160,20,20])
        for rows in splits.values():
            self.assertTrue(all(sum(x['family_id']==r['family_id'] for x in rows)==2 for r in rows))

    def test_duplicate_family_rejected(self):
        rows=self.sources()
        with self.assertRaises(ValueError): family_split(rows+[dict(rows[0],path='other',data_sha256='other')])

    def test_duplicate_hash_and_path_rejected(self):
        for field in ('path','data_sha256'):
            rows=self.sources(); rows[1][field]=rows[0][field]
            with self.assertRaises(ValueError): family_split(rows)

    def test_unknown_family_rejected(self):
        rows=self.sources()
        with self.assertRaises(ValueError): family_split(rows+[dict(rows[0],family_id='unknown',domain='randomized')])

    def test_exact_coverage_equal_tasks(self):
        result=coverage({'a':100,'b':100},50,4)
        self.assertAlmostEqual(result['expected_unique_anchor_fraction'],1-(199/200)**200)
        self.assertEqual(result['equivalent_global_passes_not_epochs'],1)
        self.assertLess(result['expected_unique_anchor_fraction'],.64)

    def test_task_balancing_not_global_uniformity(self):
        result=coverage({'a':10,'b':100},10,4)
        self.assertGreater(result['expected_unique_anchor_fraction_by_task']['a'],result['expected_unique_anchor_fraction_by_task']['b'])

    def test_single_anchor_edge(self):
        self.assertEqual(coverage({'a':1},1,1)['expected_unique_anchor_fraction'],1)

    def test_last_tenk_protocol(self):
        self.assertEqual(tail_eval_steps(100000),[90000,95000,100000])
        self.assertEqual(tail_eval_steps(20000),[10000,15000,20000])
        self.assertEqual(tail_eval_steps(10000),[5000,10000])
        with self.assertRaises(ValueError): tail_eval_steps(9999)

    def test_cache_unit(self):
        self.assertEqual(FRAME_CACHE_BYTES,3729408)

    def test_invalid_endpoint(self):
        for n,h in ((-1,4),(5,0),(3.5,4),(5,True)):
            with self.assertRaises(ValueError): valid_endpoint_anchors(n,h)


if __name__ == '__main__':
    unittest.main()
