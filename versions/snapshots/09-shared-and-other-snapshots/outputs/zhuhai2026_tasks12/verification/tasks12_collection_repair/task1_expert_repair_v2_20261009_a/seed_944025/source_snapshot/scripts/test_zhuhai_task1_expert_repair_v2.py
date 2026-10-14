#!/usr/bin/env python3
"""Read-only CPU tests: no Isaac/USD/FCL/torch import, GPU query or output writes.

Run: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .conda-envs/behavior6/bin/python -B scripts/test_zhuhai_task1_expert_repair_v2.py
Mock feedback below tests control flow, NEVER represents a physical probe pass.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.abc
class NoSimulationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'isaacsim', 'omni', 'carb', 'pxr', 'torch', 'fcl', 'trimesh'}:
            raise AssertionError(f'CPU tests must not import {fullname}')
sys.meta_path.insert(0, NoSimulationImports())

import copy
from contextlib import nullcontext, redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from harness import isaac_zhuhai_expert as original
from harness import isaac_zhuhai_task1_expert_repair_v2 as h
from harness.isaac_zhuhai_task1_expert_repair_v1 import SmoothExpert as V1Expert
from harness.zhuhai_robot_spec import HOME_ACTION
from scripts import probe_zhuhai_task1_expert_repair_v2 as p

V1 = p.VERIFICATION/'task1_expert_repair_v1'
MANIFESTS = {s: json.loads((V1/f'seed_{s}'/'manifest.json').read_text()) for s in p.REQUIRED_CASES}


def geometry(seed=944025):
    initial = MANIFESTS[seed]['attempts'][0]['initial_rule_evidence']
    spec = copy.deepcopy(initial['domain_parameters']['objects']['squeeze_bottle'])
    body = np.eye(4)
    # Initial recorded bodies are upright; derive the settled translation from
    # recorded world bounds rather than using commanded/initial spawn height.
    body[:3, 3] = (np.asarray(initial['objects']['squeeze_bottle']['world_corners_m']).mean(0)
                     - np.asarray(spec['local_corners_m']).mean(0))
    return spec, body


def args(seed=944025, **overrides):
    values = dict(seed=seed, source_parent=p.REQUIRED_CASES[seed],
        out=p.VERIFICATION/f'task1_expert_repair_v2_CPU_TEST_DO_NOT_EXECUTE/seed_{seed}',
        gpu_lock_owner=None, worker_slot=0, execute=False,
        expected_candidate_sha256=None, expected_bindings_sha256=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class EvidenceTests(unittest.TestCase):
    def test_four_original_outcomes_and_guard(self):
        for seed, manifest in MANIFESTS.items():
            row = manifest['attempts'][0]
            passed = seed in (944026, 947000)
            self.assertEqual(row['strict_predicate'], passed)
            self.assertEqual(manifest['status'], 'complete' if passed else 'stopped_attempt_limit')
            self.assertTrue(row['physics_safety_all_observations_passed'])
            self.assertEqual(row['final_rule_evidence']['score']['simulation_tolerances']['hold_seconds'], 5.)

    def test_failed_before_close_not_a_short_hold(self):
        for seed, error in ((944025, .019516077556906843), (945024, .020070794280811658)):
            trace = MANIFESTS[seed]['attempts'][0]['expert_info']['trace']
            phases = [t for t in trace if t['phase'].startswith('squeeze_bottle:')]
            retry = [t for t in phases if t['phase'].endswith('measured_approach_retry')]
            self.assertEqual(len(retry), 1)
            self.assertAlmostEqual(retry[0]['position_error_m'], error)
            self.assertFalse(any(t['phase'] in ('squeeze_bottle:close', 'squeeze_bottle:physical_continuous_hold') for t in phases))
            approach = next(t for t in phases if t['phase'] == 'squeeze_bottle:approach')
            self.assertLess(approach['ik_position_error_m'], 1e-6)
            self.assertTrue(approach['physical_object_state']['touching_robot'])
            self.assertFalse(approach['physical_object_state']['touching_gripper'])

    def test_recorded_contact_pairs_and_real_hold(self):
        import h5py  # Only small existing rule_evidence slices, never RGB or writes.
        for seed, manifest in MANIFESTS.items():
            offset, slices = 0, {}
            for t in manifest['attempts'][0]['expert_info']['trace']:
                count = t.get('samples', 0)
                if count:
                    slices[t['phase']] = (offset + 1, offset + count + 1)
                offset += count
            with h5py.File(V1/f'seed_{seed}'/'data'/f'episode_seed_{seed:010d}.hdf5', 'r') as data:
                start, end = slices['squeeze_bottle:approach_settle']
                records = [json.loads(x) for x in data['rule_evidence'][start:end]]
                contacts = [h.contact_feedback(x, '/World/Task/Objects/squeeze_bottle', 'left') for x in records]
                self.assertTrue(any('panda_hand' in c['nonfinger'] for c in contacts))
                if seed in (944025, 945024):
                    self.assertTrue(all(not c['fingers'] for c in contacts))
                else:
                    start, end = slices['squeeze_bottle:physical_continuous_hold']
                    records = [json.loads(x) for x in data['rule_evidence'][start:end]]
                    self.assertGreater(max(x['score']['items']['squeeze_bottle']['hold_seconds'] for x in records), 6.)
                    self.assertTrue(all(h.contact_feedback(x, '/World/Task/Objects/squeeze_bottle', 'left')['bilateral'] for x in records))


class GeometryTests(unittest.TestCase):
    def test_geometry_explains_large_bottle_collision(self):
        for seed in p.REQUIRED_CASES:
            spec, body = geometry(seed)
            old = np.eye(4); old[:3, :3] = spec['grasp_tcp_world_rotation']
            old[:3, 3] = body[:3, 3] + np.asarray(spec['grasp_point_local_m'])
            self.assertEqual(h.palm_clearance(spec, body, old) < 0, seed in (944025, 945024))

    def test_all_four_new_cap_poses_clear_palm_and_shoulder(self):
        for seed in p.REQUIRED_CASES:
            spec, body = geometry(seed); untouched = copy.deepcopy(spec)
            choices = h.bottle_candidates(spec, body, .74)
            self.assertGreaterEqual(len(choices), 6)
            self.assertEqual(len({c.candidate_id for c in choices}), len(choices))
            for c in choices:
                self.assertGreaterEqual(c.palm_clearance_m, h.PALM_CLEARANCE_M)
                self.assertGreaterEqual(c.pad_overlap_m, h.MIN_CAP_OVERLAP_M)
                self.assertGreater(h.gap_from_fraction(c.open_fraction), spec['parts'][0]['dimensions_m'][0])
                self.assertLess(c.open_fraction, 1.)
                self.assertGreater(c.pose[2, 3], body[2, 3] + spec['grasp_point_local_m'][2] + .01)
            self.assertEqual(spec, untouched)

    def test_reobserved_translation_is_used(self):
        spec, body = geometry()
        first = h.bottle_candidates(spec, body, .74)[0]
        body[:3, 3] += [.013, -.009, .002]
        second = h.bottle_candidates(spec, body, .74)[0]
        np.testing.assert_allclose(second.pose[:3, 3] - first.pose[:3, 3], [.013, -.009, .002])
        self.assertEqual(first.candidate_id, second.candidate_id)

    def test_aperture_round_trip_and_no_clipping(self):
        for gap in (.023, .051, h.MAX_GAP_M):
            self.assertAlmostEqual(h.gap_from_fraction(h.fraction_for_gap(gap)), gap)
        for gap in (0., .081, float('nan')):
            with self.assertRaises(ValueError): h.fraction_for_gap(gap)

    def test_tipped_nonfinite_or_oversize_fail_closed(self):
        spec, body = geometry()
        tilted = body.copy(); tilted[:3, :3] = h.Rotation.from_euler('x', 30, degrees=True).as_matrix()
        with self.assertRaises(RuntimeError): h.bottle_candidates(spec, tilted, .74)
        body[0, 0] = float('nan')
        with self.assertRaises(ValueError): h.bottle_candidates(spec, body, .74)
        spec, body = geometry(); spec['parts'][0]['dimensions_m'][:2] = [.10, .10]
        with self.assertRaises(RuntimeError): h.bottle_candidates(spec, body, .74)

    def test_contacts_require_both_correct_arm_fingers(self):
        def report(leaves, arm='left'):
            return {'contacts': [dict(body0_path='/World/Task/Objects/squeeze_bottle',
                body1_path=f'/World/Robot/{arm}/panda_hand/{leaf}') for leaf in leaves]}
        self.assertFalse(h.contact_feedback(report(['panda_hand']), '/World/Task/Objects/squeeze_bottle', 'left')['bilateral'])
        self.assertFalse(h.contact_feedback(report(['panda_leftfinger']), '/World/Task/Objects/squeeze_bottle', 'left')['bilateral'])
        self.assertFalse(h.contact_feedback(report(['panda_leftfinger', 'panda_rightfinger'], 'right'), '/World/Task/Objects/squeeze_bottle', 'left')['bilateral'])
        self.assertTrue(h.contact_feedback(report(['panda_leftfinger', 'panda_rightfinger']), '/World/Task/Objects/squeeze_bottle', 'left')['bilateral'])

    def test_real_cpu_ik_four_cap_targets(self):
        fk = original.FrankaFK()  # Cached JSON only; deliberately no collision/Isaac import.
        for seed in p.REQUIRED_CASES:
            spec, body = geometry(seed); candidate = h.bottle_candidates(spec, body, .74)[0]
            q = np.asarray(HOME_ACTION[:7])
            for height in (.10, 0., .20):
                target = candidate.pose.copy(); target[2, 3] += height
                solved = fk.ik('left', target, seed=q)
                self.assertTrue(solved.success, (seed, height, solved))
                self.assertLess(solved.position_error_m, .001)
                self.assertLess(solved.orientation_error_rad, .012)
                q = solved.joints


class FeedbackExpert(h.SmoothExpert):
    """In-memory control-flow fixture, not a physics simulator."""
    def __init__(self, *, bad_approach=False, missing_contact=False, award=True, guard=False):
        self.spec, self.body = geometry()
        self.manifest = {'table_surface_world_m': .74}
        self.current = np.asarray(HOME_ACTION).copy(); self.dof = 16; self.arm_dof = 7
        self.trace, self.calls = [], []
        self.pose = h.bottle_candidates(self.spec, self.body, .74)[0].pose.copy()
        self.bad_approach, self.missing_contact, self.award, self.guard = bad_approach, missing_contact, award, guard
        self.held = self.awarded = False; self.opening = 1.
        self.env = SimpleNamespace(actual_state=self.actual_state, rule_state=self.rule_state)

    def _body_matrix(self, key): return self.body.copy()
    def _actual_tool_pose(self, arm): return self.pose.copy()
    def actual_state(self):
        result = self.current.copy(); result[7] = self.opening; return result
    def rule_state(self):
        contacts = []
        if self.held:
            contacts = [dict(body0_path=self.spec['body_path'], body1_path=f'/World/Robot/left/panda_hand/{leaf}')
                        for leaf in ('panda_leftfinger', 'panda_rightfinger')]
        return dict(contacts=contacts, objects={'squeeze_bottle': {'touching_gripper': self.held}},
                    score={'items': {'squeeze_bottle': {'grasp_awarded': self.awarded}}})
    def _cap_choice(self, spec, arm, tried):
        self.choice = next(c for c in h.bottle_candidates(spec, self.body, .74) if c.candidate_id not in tried)
        return self.choice
    def move_pose(self, arm, pose, label): self.pose = pose.copy(); self.calls.append(label)
    def _cap_line(self, arm, target, phase, seconds=2.5):
        if self.guard and phase == 'v2_cap_approach':
            raise RuntimeError('PhysX contact safety guard stopped expert before further commands')
        self.calls.append(phase)
        if self.held: self.body[:3, 3] += target[:3, 3] - self.pose[:3, 3]
        self.pose = target.copy()
        if self.bad_approach and phase == 'v2_cap_approach': self.pose[0, 3] += .020
    def gripper(self, arm, value, label):
        self.calls.append(label)
        self.held = value == 0. and not self.missing_contact
        self.opening = h.fraction_for_gap(self.choice.cap_span_m) if self.held else value
    def hold(self, seconds, label):
        self.calls.append((label, seconds))
        if label.endswith('physical_continuous_hold'): self.awarded = self.award


class MotionTests(unittest.TestCase):
    def test_other_objects_and_all_base_guards_are_inherited(self):
        for name in ('_drive', 'move_pose_path', '_open_lid', '_place_garbage_native16', '_task1_once'):
            self.assertIs(getattr(h.SmoothExpert, name), getattr(V1Expert, name))
        ex = FeedbackExpert()
        with patch.object(V1Expert, '_acquire_garbage') as base:
            ex._acquire_garbage('milk_carton', {}, 'left')
            base.assert_called_once_with('milk_carton', {}, 'left')
        self.assertEqual(h.APPROACH_ERROR_M, .006)
        self.assertIsNot(original.SmoothExpert, h.SmoothExpert)

    def test_order_bilateral_low_lift_then_unchanged_hold(self):
        ex = FeedbackExpert(); ex._acquire_garbage('squeeze_bottle', ex.spec, 'left')
        self.assertLess(ex.calls.index('squeeze_bottle:v2_cap_close'), ex.calls.index('v2_low_test_lift'))
        self.assertLess(ex.calls.index('v2_low_test_lift'), ex.calls.index('v2_full_lift'))
        self.assertIn(('squeeze_bottle:physical_continuous_hold', 5.2), ex.calls)
        self.assertFalse(any('retry_release' in str(x) for x in ex.calls))

    def test_bad_approach_never_closes_and_tries_three_distinct_poses(self):
        ex = FeedbackExpert(bad_approach=True)
        with self.assertRaisesRegex(RuntimeError, 'BEFORE closing'):
            ex._acquire_garbage('squeeze_bottle', ex.spec, 'left')
        self.assertNotIn('squeeze_bottle:v2_cap_close', ex.calls)
        ids = [t['candidate_id'] for t in ex.trace if t['phase'].endswith('v2_cap_candidate')]
        self.assertEqual(len(set(ids)), h.MAX_RETRIES)

    def test_missing_contact_never_lifts(self):
        ex = FeedbackExpert(missing_contact=True)
        with self.assertRaisesRegex(RuntimeError, 'bilateral cap contact'):
            ex._acquire_garbage('squeeze_bottle', ex.spec, 'left')
        self.assertNotIn('v2_low_test_lift', ex.calls)
        self.assertNotIn('v2_full_lift', ex.calls)

    def test_no_award_no_success_and_lower_before_release(self):
        ex = FeedbackExpert(award=False)
        with self.assertRaisesRegex(RuntimeError, 'physical5s'):
            ex._acquire_garbage('squeeze_bottle', ex.spec, 'left')
        lower = ex.calls.index('v2_retry_lower_before_release')
        release = ex.calls.index('squeeze_bottle:v2_retry_release')
        self.assertLess(lower, release)
        self.assertEqual(ex.calls.count(('squeeze_bottle:physical_continuous_hold', 5.2)), 3)

    def test_physics_guard_propagates_no_retry_or_reset(self):
        ex = FeedbackExpert(guard=True)
        with self.assertRaisesRegex(RuntimeError, 'PhysX contact safety guard'):
            ex._acquire_garbage('squeeze_bottle', ex.spec, 'left')
        self.assertFalse(any('retry' in str(x) for x in ex.calls))

    def test_planner_rejects_detached_ik_detour(self):
        ex = FeedbackExpert()
        ex.fk_model = SimpleNamespace(ik=lambda *a, **kw: SimpleNamespace(success=True,
            joints=np.zeros(7), joint_waypoints=[np.zeros(7)]))
        with self.assertRaisesRegex(RuntimeError, 'strict cap descent/lift path'):
            h.SmoothExpert._cap_choice(ex, ex.spec, 'left', set())

    def test_vertical_path_has_nine_nodes_and_uses_original_executor(self):
        ex = FeedbackExpert(); ex.move_pose_path = Mock()
        target = ex.pose.copy(); target[2, 3] += .035
        h.SmoothExpert._cap_line(ex, 'left', target, 'test')
        poses = ex.move_pose_path.call_args.args[1]
        self.assertEqual(len(poses), 9)
        np.testing.assert_allclose([x[:2, 3] for x in poses], np.tile(target[:2, 3], (9, 1)))


class ProbeTests(unittest.TestCase):
    def test_cpu_default_no_launch_or_writes(self):
        fake = (SimpleNamespace(out=args().out), {}, {}, {'candidate_sha256': 'a'*64, 'bindings_sha256': 'b'*64})
        with patch.object(p, 'make_plan', return_value=fake), patch.object(p, 'execute_probe') as execute, \
                patch.object(p, 'verify_owner') as owner, patch.object(Path, 'mkdir') as mkdir, redirect_stdout(io.StringIO()):
            self.assertEqual(p.main(['--seed', '944025', '--source-parent', '943064', '--out', str(args().out)]), 0)
        execute.assert_not_called(); owner.assert_not_called(); mkdir.assert_not_called()

    def test_refuses_v1_production_wrong_parent_symlink_and_existing(self):
        for out in (V1/'seed_944025', ROOT/'outputs/production_fake/seed_944025'):
            with self.assertRaises((ValueError, FileExistsError)): p.validate_request(args(out=out))
        with self.assertRaises(ValueError): p.validate_request(args(source_parent=943003))
        with patch.object(Path, 'exists', return_value=True):
            with self.assertRaises(FileExistsError): p.validate_request(args())
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaises(FileExistsError): p.validate_request(args())

    def test_execute_requires_both_hashes_and_dedicated_owner(self):
        for values in (dict(execute=True), dict(execute=True, gpu_lock_owner='zhuhai-task1-repair-v2-test'),
                       dict(gpu_lock_owner='zhuhai-task2-collection-only-v2-20261009')):
            with self.assertRaises(ValueError): p.validate_request(args(**values))
        p.validate_request(args(execute=True, gpu_lock_owner='zhuhai-task1-repair-v2-test',
            expected_candidate_sha256='a'*64, expected_bindings_sha256='b'*64))

    def test_owner_failure_precedes_collector_or_runtime(self):
        with patch.object(p, 'verify_owner', side_effect=RuntimeError('owner mismatch')), patch.object(p, 'make_plan') as plan:
            with self.assertRaisesRegex(RuntimeError, 'owner mismatch'):
                p.main(['--seed', '944025', '--source-parent', '943064', '--out', str(args().out),
                    '--execute', '--gpu-lock-owner', 'zhuhai-task1-repair-v2-test',
                    '--expected-candidate-sha256', 'a'*64, '--expected-bindings-sha256', 'b'*64])
            plan.assert_not_called()

    def test_bindings_drift_rejected(self):
        with patch.object(p, 'sha256', return_value='b'*64):
            with self.assertRaisesRegex(RuntimeError, 'changed'): p.verify_bindings({'/fake/candidate.py': 'a'*64})

    def test_maximum_four_explicit_slots(self):
        with redirect_stdout(io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit): p.parser().parse_args(['--seed', '944025', '--source-parent', '943064',
                '--out', str(args().out), '--worker-slot', '4'])

    def test_resource_predicates_launch_and_fourth_slot(self):
        report = dict(ram=dict(total_bytes=32*p.GIB, available_bytes=8*p.GIB,
            swap_used_bytes=0, swap_total_bytes=10*p.GIB), disk={'free_bytes': 150*p.GIB},
            gpu={'devices': [dict(total_bytes=16*p.GIB, used_bytes=4*p.GIB)]})
        p.require_resources(report, 0, launch=True)
        report['ram']['available_bytes'] = 6*p.GIB
        with self.assertRaises(RuntimeError): p.require_resources(report, 0, launch=True)
        p.require_resources(report, 0, launch=False)
        report['ram']['swap_used_bytes'] = 9*p.GIB
        with self.assertRaises(RuntimeError): p.require_resources(report, 3, launch=False)
        p.require_resources(report, 2, launch=False)
        report['gpu']['devices'] = []
        with self.assertRaises(RuntimeError): p.require_resources(report, 0, launch=False)

    def test_plan_binding_is_common_to_four_cases_and_never_quota(self):
        from scripts import collect_isaac_zhuhai_recycle as c
        source = {'attempts': [dict(seed=s, status='success', strict_predicate=True, plan_success=True,
            family_id=f'source:{s}', data_path=f'/fake/{s}.hdf5', data_sha256='a'*64) for s in set(p.REQUIRED_CASES.values())]}
        template = copy.deepcopy(MANIFESTS[944025]['contract'])
        fake = SimpleNamespace(parser=c.parser, contract=lambda _: copy.deepcopy(template))
        original_read = Path.read_text
        def read(path, *a, **kw): return json.dumps(source) if path == p.SOURCE else original_read(path, *a, **kw)
        fingerprints = []
        with patch.object(Path, 'read_text', new=read), patch.object(p, 'sha256', return_value='a'*64):
            for seed in p.REQUIRED_CASES:
                config, contract, _, suite = p.make_plan(args(seed), fake)
                self.assertEqual((config.count, config.max_attempts, config.max_consecutive_failures), (1, 1, 3))
                self.assertEqual(contract['production_quota_credit'], 0)
                self.assertTrue(contract['source_probe']); self.assertFalse(contract['record_video'])
                self.assertFalse(contract['third_view']); self.assertFalse(suite['admitted_to_production'])
                self.assertIn(str(p.CANDIDATE), contract['scene_dependency_sha256'])
                self.assertEqual(len(suite['required_cases']), 4)
                fingerprints.append(suite['bindings_sha256'])
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                p.make_plan(args(expected_candidate_sha256='c'*64), fake)
        self.assertEqual(len(set(fingerprints)), 1)

    def test_resource_failure_cannot_import_isaac_or_make_case(self):
        from scripts import collect_isaac_zhuhai_recycle as c
        with patch.object(p, 'worker_slot', return_value=nullcontext()), patch.object(p, 'verify_bindings'), \
                patch.object(c, 'resource_snapshot', side_effect=RuntimeError('no resources')), patch.object(p, 'freeze_plan') as freeze:
            with self.assertRaisesRegex(RuntimeError, 'no resources'):
                p.execute_probe(args(), SimpleNamespace(), {}, {}, {'bindings': {}})
            freeze.assert_not_called()

    def test_no_heavy_runtime_loaded(self):
        for name in sys.modules:
            self.assertNotEqual(name, 'harness.isaac_zhuhai_runtime')
            self.assertNotIn(name.split('.')[0], {'isaacsim', 'omni', 'carb', 'pxr', 'torch', 'fcl', 'trimesh'})


class AdditionalSafetyTests(unittest.TestCase):
    def test_jammed_preshape_cannot_descend(self):
        ex = FeedbackExpert()
        original_gripper = ex.gripper
        def jammed(arm, value, label):
            original_gripper(arm, value, label)
            if label.endswith('v2_geometry_preshape'): ex.opening = .1
        ex.gripper = jammed
        with self.assertRaisesRegex(RuntimeError, 'BEFORE descent'):
            ex._acquire_garbage('squeeze_bottle', ex.spec, 'left')
        self.assertNotIn('v2_cap_approach', ex.calls)

    def test_moving_object_does_not_bypass_original_target_tracking_gate(self):
        ex = FeedbackExpert()
        original_line = ex._cap_line
        def moved_together(arm, target, phase, seconds=2.5):
            original_line(arm, target, phase, seconds)
            if phase == 'v2_cap_approach':
                ex.pose[0, 3] += .020; ex.body[0, 3] += .020
        ex._cap_line = moved_together
        with self.assertRaisesRegex(RuntimeError, 'BEFORE closing'):
            ex._acquire_garbage('squeeze_bottle', ex.spec, 'left')
        self.assertNotIn('squeeze_bottle:v2_cap_close', ex.calls)
        measured = [t for t in ex.trace if t['phase'].endswith('v2_measured_approach')]
        self.assertTrue(all(t['position_error_m'] > .006 and t['live_cap_error_m'] < 1e-6 for t in measured))

    def test_original_strict_validator_confirms_v1_outcomes(self):
        from scripts import collect_isaac_zhuhai_recycle as c
        for seed, manifest in MANIFESTS.items():
            self.assertEqual(len(c._success_rows(manifest, verify_files=False)), int(seed in (944026, 947000)))

    def test_busy_slot_fails_without_unlinking_or_running(self):
        with patch.object(p, 'verify_owner'), patch.object(p.os, 'open', return_value=123), \
                patch.object(p.fcntl, 'flock', side_effect=BlockingIOError()), patch.object(p.os, 'close') as close, \
                patch.object(Path, 'unlink') as unlink:
            with self.assertRaisesRegex(RuntimeError, 'slot already occupied'):
                with p.worker_slot('zhuhai-task1-repair-v2-test', 0):
                    self.fail('busy slot must not start a worker')
            close.assert_called_once_with(123); unlink.assert_not_called()

    def test_immutable_suite_binding_mismatch_stops_before_case_creation(self):
        config = SimpleNamespace(out=args().out)
        collector = SimpleNamespace(write_once=Mock(side_effect=FileExistsError('suite plan differs')))
        with patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaises(FileExistsError): p.freeze_plan(config, {}, {}, collector)
            self.assertEqual(mkdir.call_count, 1)  # Only suite mkdir; no case/snapshot.


class SuiteGateTests(unittest.TestCase):
    def fixture(self):
        # Synthetic protocol objects in memory ONLY. The strict physical
        # validator is tested on real v1 evidence separately above.
        directory = args().out.parent
        bindings = {str(p.CANDIDATE): 'a'*64}
        fingerprint = p.digest_json(bindings)
        suite = dict(schema=p.SCOPE, required_cases=[dict(seed=s, parent=x) for s, x in p.REQUIRED_CASES.items()],
            candidate_sha256='a'*64, bindings=bindings, bindings_sha256=fingerprint,
            minimum_strict_physical_passes=4, admitted_to_production=False,
            production_quota_credit=0, record_video=False, third_view=False)
        files = {str(directory/'suite_plan.json'): suite}
        for seed, parent in p.REQUIRED_CASES.items():
            receipt = directory/f'seed_{seed}'/'test_receipt.json'
            contract = dict(collection_scope=p.SCOPE, source_probe=True, admitted_to_production=False,
                production_quota_credit=0, probe_bindings_sha256=fingerprint,
                expert_variant={'sha256': 'a'*64}, record_video=False, third_view=False)
            files[str(directory/f'seed_{seed}'/'manifest.json')] = dict(status='complete', requested_success_count=1,
                contract=contract, attempts=[dict(seed=seed, source_parent=parent, status='success', expert_receipt_path=str(receipt))])
            files[str(receipt)] = {'expert_class': 'harness.isaac_zhuhai_task1_expert_repair_v2.SmoothExpert'}
        return directory, files

    def verify_virtual(self, directory, files):
        from scripts import collect_isaac_zhuhai_recycle as c
        def read(path, *a, **kw):
            if str(path) not in files: raise FileNotFoundError(str(path))
            return json.dumps(files[str(path)])
        with patch.object(Path, 'read_text', new=read), patch.object(p, 'verify_bindings'), \
                patch.object(c, '_success_rows', side_effect=lambda m, **kw: [r for r in m['attempts'] if r['status'] == 'success']) as strict:
            result = p.verify_suite(directory)
            self.assertEqual(strict.call_count, 4)
            self.assertTrue(all(call.kwargs == {'verify_files': True} for call in strict.call_args_list))
            return result

    def test_requires_four_strict_cases_but_never_promotes(self):
        directory, files = self.fixture()
        result = self.verify_virtual(directory, files)
        self.assertTrue(result['all_four_strict_passed'])
        self.assertFalse(result['admitted_to_production'])
        self.assertEqual(result['production_quota_credit'], 0)

    def test_rejects_one_failed_case(self):
        directory, files = self.fixture()
        files[str(directory/'seed_945024'/'manifest.json')]['attempts'][0]['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'Physical probe failed'): self.verify_virtual(directory, files)

    def test_rejects_missing_fourth_case(self):
        directory, files = self.fixture()
        del files[str(directory/'seed_947000'/'manifest.json')]
        with self.assertRaises(FileNotFoundError): self.verify_virtual(directory, files)

    def test_rejects_different_candidate_or_v1_receipt(self):
        for change in ('hash', 'class'):
            directory, files = self.fixture()
            if change == 'hash':
                files[str(directory/'seed_944026'/'manifest.json')]['contract']['expert_variant']['sha256'] = 'b'*64
            else:
                files[str(directory/'seed_944026'/'test_receipt.json')]['expert_class'] = 'harness.isaac_zhuhai_task1_expert_repair_v1.SmoothExpert'
            with self.assertRaises(ValueError): self.verify_virtual(directory, files)

    def test_rejects_v1_directory_and_tampered_suite_protocol(self):
        with self.assertRaisesRegex(ValueError, 'new v2'): p.verify_suite(V1)
        directory, files = self.fixture()
        files[str(directory/'suite_plan.json')]['minimum_strict_physical_passes'] = 2
        with self.assertRaises(ValueError): self.verify_virtual(directory, files)


if __name__ == '__main__':
    unittest.main(verbosity=2)
