#!/usr/bin/env python3
"""CPU synthetic/tamper checks of the offline V3 analyzer; not robot outcomes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_precision_v3_eval_v42 as a


def rule(step=0):
    objects = {k: dict(world_corners_m=[[5., 5., .74], [5.1, 5.1, .79]],
                       touching_gripper=False, released=True) for k in a.OBJECTS}
    items = {k: dict(grasp_awarded=False, placement_valid=False,
                     hold_seconds=0., stable_seconds=0.) for k in a.OBJECTS}
    return dict(control_step=step, physics_time_seconds=step * .1,
                objects=objects, table_surface_world_m=.74,
                destination_interiors_world_m={'white': [[-.1, -.1, 0.], [.1, .1, .1]]},
                score=dict(items=items, full_task_success=False, base_score=0))


class FixedPointFK:
    def fk(self, side, q):
        out = np.eye(4); out[:3, 3] = 0.
        return out


def action(step, chunk, sent=None):
    values = chunk[min(step, 31)] if sent is None else sent
    return dict(control_step=step, dispatched_action=values.tolist(), smoothed_action=values.tolist(),
                policy=dict(replanned=step % 8 == 0, execution_prefix=8,
                            model_raw_action=values.tolist(), raw_denormalized_chunk=chunk.tolist(),
                            smoothed_denormalized_chunk=chunk.tolist()))


class EvidenceSemantics(unittest.TestCase):
    def test_contact_and_closure_do_not_award_lifted_grasp(self):
        rows = [rule(i) for i in range(4)]
        for r in rows:
            r['objects']['bowl']['touching_gripper'] = True
            r['objects']['bowl']['released'] = False
        found = a.object_progress(rows)
        self.assertEqual(found['failure_stage'], 'before_first_verified_grasp')
        self.assertFalse(found['objects']['bowl']['ever_verified_grasp'])
        self.assertEqual(found['objects']['bowl']['contact_without_lift_observations'], 4)
        self.assertEqual(found['objects']['bowl']['strict_held_observations'], 0)

    def test_tail_closure_is_not_executed_prefix_closure(self):
        chunk = np.ones((32, 16)); chunk[25:, 7] = .1
        acts = [action(i, chunk) for i in range(8)]
        sent = np.asarray([x['dispatched_action'] for x in acts])
        got = a.closure_analysis(acts, sent, [rule(i) for i in range(9)])['sides']['left']
        self.assertTrue(got['only_tail_close'])
        self.assertFalse(got['no_close_anywhere'])
        self.assertEqual(got['dispatched_closed_steps'], 0)
        self.assertEqual(got['first_close_1based_histogram'], {'26': 1})

    def test_empty_close_uses_following_observations_and_any_object_contact(self):
        chunk = np.ones((32, 16)); chunk[:, 7] = .1
        acts = [action(i, chunk) for i in range(8)]
        sent = np.asarray([x['dispatched_action'] for x in acts])
        rows = [rule(i) for i in range(9)]
        # Contact in another object's aggregated either-hand flag must prevent a definitive empty label.
        rows[4]['objects']['spoon']['touching_gripper'] = True
        got = a.closure_analysis(acts, sent, rows)['sides']['left']['dispatched_close_intervals'][0]
        self.assertEqual(got['following_observation_range'], [1, 8])
        self.assertFalse(got['no_contact_with_any_task_object'])
        rows[4]['objects']['spoon']['touching_gripper'] = False
        self.assertTrue(a.closure_analysis(acts, sent, rows)['sides']['left']['dispatched_close_intervals'][0]['no_contact_with_any_task_object'])

    def test_empty_box_only_counts_in_old_effective_proxy(self):
        rows = [rule(i) for i in range(4)]
        objects = a.object_progress(rows)['objects']
        _, activity, _ = a.geometry_and_activity(np.zeros((4, 16)), rows, objects, FixedPointFK())
        self.assertEqual(activity['effective_fraction_v41'], 1.)
        self.assertEqual(activity['corrected_activity_proxy_fraction'], 0.)
        self.assertEqual(activity['box_only_without_held_object_observations'], 4)
        self.assertAlmostEqual(activity['physical_seconds'], .3)
        self.assertAlmostEqual(activity['observation_weight_seconds'], .4)

    def test_previously_awarded_next_grasp_is_not_falsely_called_missing(self):
        rows = [rule(i) for i in range(4)]
        for r in rows:
            r['score']['items']['spoon']['grasp_awarded'] = True
        for r in rows[1:]:
            r['score']['items']['bowl']['placement_valid'] = True
        objects = a.object_progress(rows)['objects']
        _, _, stages = a.geometry_and_activity(np.zeros((4, 16)), rows, objects, FixedPointFK())
        stage = stages['bowl_to_spoon']
        self.assertTrue(stage['following_grasp_already_awarded_at_previous_placement'])
        self.assertFalse(stage['following_grasp_occurs_after_previous_placement'])
        self.assertFalse(stage['no_following_verified_grasp_after_previous_placement'])

    def test_gripper_motion_cannot_inflate_arm_jitter(self):
        q = np.zeros((9, 16)); q[:, 7] = np.arange(9) % 2; q[:, 15] = 1 - q[:, 7]
        chunk = np.ones((32, 16)); acts = [action(i, chunk, sent=q[i + 1]) for i in range(8)]
        got = a.command_motion(acts, q, q[1:])
        for stream in got['streams'].values():
            self.assertEqual(stream['acceleration_rms_rad_s2'], 0.)
            self.assertEqual(stream['jerk_rms_rad_s3'], 0.)

    def test_sha_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); path = directory / 'evidence.json'
            path.write_bytes(b'original')
            row = {'evidence_files': {'evidence.json': {'sha256': a.sha(b'original'), 'path': str(path)}}}
            self.assertEqual(a.binding_read(directory, 'evidence.json', row), b'original')
            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                a.binding_read(directory, 'evidence.json', row)


if __name__ == '__main__':
    unittest.main()
