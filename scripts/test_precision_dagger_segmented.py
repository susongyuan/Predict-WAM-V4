"""Boundary, local-goal and causal-restoration tests; CPU fixtures only."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from harness.zhuhai_dagger_segmented import (SegmentedEpisode, SegmentedPrecisionDataset,
    finish_segments, goal_met, choose_goal, restore_policy, limiter_state, required_visual_frames)
from harness.zhuhai_demo_recording import StreamDemoRecorder
from harness.zhuhai_dataset import SourceEpisode
from harness.zhuhai_robot_spec import HOME_ACTION
from harness.zhuhai_official_braking import BrakingRateLimiter, protocol
from harness.wam_v3 import V3Context
from harness.wam_event_clock import EventClock, episode_event_features
from scripts.test_zhuhai_dagger import Store, fixture as old_fixture


def evidence(t, *, held=False, placed=False, historical=False):
    keys = ('bowl', 'spoon', 'cup', 'plate')
    d = dict(physics_time_seconds=t*.1, physics_safety=dict(passed=True, events=[], checks=['CPU fixture only']),
        score=dict(base_score=8 if placed else 4 if held else 0, full_task_success=False, items={}), objects={})
    for k in keys:
        active = k == 'bowl'
        d['score']['items'][k] = dict(grasp_awarded=active and (held or placed or historical),
            placement_valid=active and placed, hold_seconds=5.2 if active and held else 0.,
            stable_seconds=1.2 if active and placed else 0.)
        d['objects'][k] = dict(touching_gripper=active and held, touching_robot=active and held,
                                released=not (active and held))
    return d


def fixture(directory):
    p = Path(directory)/'segmented.hdf5'
    rec = StreamDemoRecorder(p, metadata=dict(task='zhuhai_task2', family_id='TRAIN-fixture', split='train'), frequency=10.)
    for i in range(24):
        q = np.asarray(HOME_ACTION, np.float32)
        q[0] += .001*i
        command = np.asarray(HOME_ACTION, np.float32)
        command[0] += .01*i
        rgb = np.full((16, 16, 3), i, np.uint8)
        rule = evidence(i, held=i == 9, placed=i >= 20)
        rec.capture(dict(rgb={k: rgb for k in ('head_camera', 'left_camera', 'right_camera')},
            measured_joint_gripper_state=q, commanded_joint_gripper_state=command), i*.1, rule)
    spans = [dict(start=2, end=9, passed=True, goal='grasp', key='bowl'),
             dict(start=12, end=20, passed=True, goal='placement', key='bowl'),
             dict(start=21, end=23, passed=False, goal='grasp', key='spoon')]
    authority = np.zeros(23, np.uint8)
    for s in spans:
        authority[s['start']:s['end']] = 1
    finish_segments(rec, authority, ['fixture']*23, spans)
    rec.finish(final_evidence=evidence(23, placed=True), successful=True)
    return dict(path=str(p), split='train', family_id='TRAIN-fixture')


class Tests(unittest.TestCase):
    def test_local_success_survives_later_model_or_expert_failure(self):
        with tempfile.TemporaryDirectory() as d:
            row = fixture(d)
            ep = SegmentedEpisode(row)
            self.assertEqual(ep.supervised_count, 15)
            self.assertEqual(ep.policy_count, 6)
            self.assertEqual(len(ep.valid_segments), 2)
            self.assertFalse(ep.mask[21:].any())
            with self.assertRaisesRegex(ValueError, 'source format'):
                SourceEpisode(row)

    def test_action_chunk_stops_at_handoff_and_never_skips_policy_gap(self):
        with tempfile.TemporaryDirectory() as d:
            ds = SegmentedPrecisionDataset('zhuhai_task2', [fixture(d)], Store(), np.ones(1024))
            ep = ds.episodes[0]
            sample = ds[0]
            self.assertEqual(sample['anchor'], 2)
            self.assertEqual(int(sample['action_valid_mask'].sum()), 7)
            np.testing.assert_array_equal(sample['action_sequence'][:7], ep.action[2:9])
            self.assertTrue((sample['action_sequence'][7:] == 0).all())
            self.assertEqual(ds[7]['anchor'], 12)
            self.assertEqual(ds[len(ds)-1]['anchor'], 19)

    def test_auxiliary_windows_stop_at_boundary_but_end_observation_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            ds = SegmentedPrecisionDataset('zhuhai_task2', [fixture(d)], Store(), np.ones(1024))
            sample = ds.at(0, 5)  # 4-step future == first expert endpoint.
            self.assertTrue(sample['idm_valid_mask'])
            self.assertFalse(ds.at(0, 6)['idm_valid_mask'])
            self.assertTrue(ds.at(0, 12)['future_valid_mask_8'])
            self.assertFalse(ds.at(0, 13)['future_valid_mask_8'])
            self.assertFalse(sample['future_valid_mask'])

    def test_real_mixed_history_and_event_clock_remain_causal(self):
        with tempfile.TemporaryDirectory() as d:
            ds = SegmentedPrecisionDataset('zhuhai_task2', [fixture(d)], Store(), np.ones(1024))
            ep, sample = ds.episodes[0], ds.at(0, 12)
            np.testing.assert_array_equal(sample['state'][1], ep.state[10])  # Model-controlled gap.
            import h5py
            with h5py.File(ep.path, 'r') as f:
                initial = f['target_command'][0].copy()
            expected = episode_event_features(ep.state[:13], ep.action[:13], initial_target=initial)[-1]
            np.testing.assert_array_equal(sample['event_state'], expected)

    def test_failed_expert_labels_or_fake_success_rejected(self):
        import h5py
        with tempfile.TemporaryDirectory() as d:
            row = fixture(d)
            with h5py.File(row['path'], 'r+') as f:
                f['expert_action_mask'][21] = True
            with self.assertRaisesRegex(ValueError, 'leak'):
                SegmentedEpisode(row)
        with tempfile.TemporaryDirectory() as d:
            row = fixture(d)
            with h5py.File(row['path'], 'r+') as f:
                rule = json.loads(f['rule_evidence'][9])
                rule['objects']['bowl']['touching_gripper'] = False
                f['rule_evidence'][9] = json.dumps(rule)
            with self.assertRaisesRegex(ValueError, 'Unverified'):
                SegmentedEpisode(row)

    def test_historical_grasp_credit_is_not_current_hold(self):
        stale = evidence(3, historical=True)
        self.assertFalse(goal_met(stale, 'grasp', 'bowl'))
        self.assertEqual(choose_goal(stale, []), ('grasp', 'bowl'))
        self.assertEqual(choose_goal(stale, [dict(key='bowl', goal='grasp', passed=True)]), ('placement', 'bowl'))
        self.assertEqual(choose_goal(evidence(3, held=True), []), ('placement', 'bowl'))

    def test_legacy_archives_remain_readable_without_rewriting(self):
        with tempfile.TemporaryDirectory() as d:
            row = old_fixture(d)
            before = Path(row['path']).read_bytes()
            ep = SegmentedEpisode(row)
            self.assertEqual(ep.supervised_count, 5)
            self.assertEqual(ep.variant, 'legacy_full_continuation')
            self.assertEqual(Path(row['path']).read_bytes(), before)

    def test_uniform_segment_sampler_does_not_weight_by_segment_length(self):
        with tempfile.TemporaryDirectory() as d:
            ds = SegmentedPrecisionDataset('zhuhai_task2', [fixture(d)], Store(), np.ones(1024))
            # Sample spans directly with a deterministic stub; covers both endpoints.
            class RNG:
                def __init__(self, i): self.i = i
                def integers(self, low, high=None): return self.i if high is None else high-1
            self.assertEqual(ds.sample(RNG(0))['anchor'], 8)
            self.assertEqual(ds.sample(RNG(1))['anchor'], 19)

    def policies(self):
        from scripts.eval_zhuhai_wam_precision_v2 import clocked_policy
        from harness.zhuhai_official_braking import install_braking
        def encode(rgb):
            pool = np.full((3, 2048), float(rgb['head_camera'][0, 0, 0]), np.float32)
            return None, pool
        def sample(*args):
            return np.tile(np.asarray(HOME_ACTION), (32, 1))
        def make():
            return install_braking(clocked_policy(V3Context([2, 4, 8, 16, 32], [4, 16], ['a', 'b', 'c'], state_dim=16),
                encode, sample, protocol()), protocol())
        return make

    def test_sparse_visual_restore_matches_continuous_history_and_braking(self):
        make = self.policies()
        continuous, resumed = make(), make()
        for policy in (continuous, resumed):
            policy.reset('episode', 42)
        observations = []
        controller = BrakingRateLimiter(protocol()['rate_limit'])
        controller.reset(np.asarray(HOME_ACTION))
        for i in range(75):
            q = np.asarray(HOME_ACTION, np.float32)
            q[0] += .001*i
            q[7] = 0.
            observation = dict(rgb={v: np.full((2, 2, 3), i, np.uint8)
                for v in ('head_camera', 'left_camera', 'right_camera')},
                measured_joint_gripper_state=q, commanded_joint_gripper_state=np.asarray(HOME_ACTION, np.float32))
            observations.append(observation)
            _, pool = continuous.encode(observation['rgb'])
            continuous.context.push(q, pool)
            continuous.event_state = continuous.clock.update(q, observation['commanded_joint_gripper_state'])
        command = np.asarray(HOME_ACTION).copy()
        command[0] += .1
        controller.step(command)
        visuals = {i: observations[i] for i in required_visual_frames(75, [4, 16])}
        receipt = restore_policy(resumed, 'episode', 75,
            [o['measured_joint_gripper_state'] for o in observations],
            [o['commanded_joint_gripper_state'] for o in observations], visuals, limiter_state(controller))
        for actual, expected in zip(resumed.context.inputs(), continuous.context.inputs()):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(resumed.event_state, continuous.event_state)
        for key in ('previous', 'velocity', 'acceleration'):
            np.testing.assert_array_equal(getattr(resumed.limiter, key), getattr(controller, key))
        self.assertEqual(receipt['next_step'], 75)
        self.assertEqual(len(receipt['real_visual_frames_encoded']), 18)
        self.assertEqual(resumed.limiter.step(command)[1], controller.step(command)[1])
        self.assertFalse(resumed.queue)

    def test_missing_visual_history_and_wrong_episode_rejected(self):
        policy = self.policies()()
        policy.reset('episode', 42)
        with self.assertRaisesRegex(ValueError, 'visual frames'):
            restore_policy(policy, 'episode', 2, [HOME_ACTION]*2, [HOME_ACTION]*2, {}, {})
        with self.assertRaisesRegex(ValueError, 'same episode'):
            restore_policy(policy, 'other', 2, [HOME_ACTION]*2, [HOME_ACTION]*2, {}, {})

    def test_mixed_batch_preserves_original_schema_and_only_expert_labels(self):
        from torch.utils.data import default_collate
        from harness import zhuhai_dagger_segmented_training as training
        with tempfile.TemporaryDirectory() as d:
            ds = SegmentedPrecisionDataset('zhuhai_task2', [fixture(d)], Store(), np.ones(1024))
            class Original:
                def __len__(self): return 1
                def __getitem__(self, index):
                    s = ds[0].copy()
                    s.pop('supervision_source')
                    return s
            batch = default_collate(training.mixed_samples(Original(), ds, np.random.default_rng(1)))
            self.assertEqual(tuple(batch['action_sequence'].shape), (4, 32, 16))
            self.assertEqual(tuple(batch['state'].shape), (4, 7, 16))
            self.assertTrue(batch['action_valid_mask'].any(dim=1).all())
            self.assertNotIn('supervision_source', batch)

    def test_fifty_mixed_episode_admission_counts_segments_separately(self):
        import shutil
        import h5py
        from harness import zhuhai_dagger_segmented_training as training
        with tempfile.TemporaryDirectory() as d:
            row = fixture(d)
            rows = []
            from harness.zhuhai_dataset import sha256_file
            for i in range(50):
                path = Path(d)/f'copy_{i}.hdf5'
                shutil.copyfile(row['path'], path)
                with h5py.File(path, 'r+') as f:
                    f.attrs['cpu_fixture_seed'] = i
                rows.append(dict(row, path=str(path), seed=1016000+i, domain='dagger', policy_steps=6,
                    sha256=sha256_file(path), expert_steps=15, takeover_step=2, verified_segment_count=2, handback_count=2))
            base = dict(splits=dict(zhuhai_task2=dict(train=[dict(family_id='TRAIN-fixture')],
                val=[dict(family_id='VAL-fixture')], heldout=[])))
            manifest = dict(status='complete', success_count=50, record_video=False, heldout_used=False, train_rows=rows)
            self.assertEqual(len(training.validate_rows(base, manifest)), 50)
            bad = deepcopy(manifest)
            bad['train_rows'][0]['handback_count'] = 0
            with self.assertRaisesRegex(ValueError, 'handback'):
                training.validate_rows(base, bad)
            bad = deepcopy(manifest)
            bad['train_rows'][0]['family_id'] = 'VAL-fixture'
            with self.assertRaisesRegex(ValueError, 'split'):
                training.validate_rows(base, bad)

    def test_failure_guard_counts_actual_interventions_not_whole_episodes(self):
        from harness.zhuhai_dagger_segmented_admission import update_failure_count
        raw = dict(segments=[dict(passed=True), dict(passed=True), dict(passed=False, failure='RuntimeError: failed')])
        self.assertEqual(update_failure_count(2, raw), 1)
        self.assertEqual(update_failure_count(2, dict(segments=[dict(passed=False, failure='RuntimeError: failed')])), 3)
        self.assertEqual(update_failure_count(2, dict(segments=[dict(passed=False, failure='ResourcePressure: reserve')])), 2)
        self.assertEqual(update_failure_count(2, dict(segments=[dict(passed=False, failure='EpisodeBudgetReached: wall')])), 2)
        self.assertEqual(update_failure_count(2, dict(status='success')), 0)
        self.assertEqual(update_failure_count(2, dict(status='expert_failed')), 3)

    def test_real_handback_pilot_is_independent_of_later_failed_span(self):
        from harness.zhuhai_dagger_segmented_admission import actual_handbacks, pilot_eligible
        with tempfile.TemporaryDirectory() as d:
            ep = SegmentedEpisode(fixture(d))
            self.assertEqual(actual_handbacks(ep), 2)
            raw = dict(handback_count=2, all_observations_safe=True, expert_failure='later failure')
            self.assertTrue(pilot_eligible(ep, raw))
            with self.assertRaisesRegex(ValueError, 'real archived'):
                pilot_eligible(ep, dict(raw, handback_count=3))


if __name__ == '__main__':
    unittest.main()
