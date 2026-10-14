#!/usr/bin/env python3
"""CPU-only event-clock contract tests; no training, simulator, or object-hold claims."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.wam_event_clock import EventClock, episode_event_features


def sample(left=1., right=1.):
    q = np.zeros(16)
    q[[7, 15]] = [left, right]
    return q


class EventClockTests(unittest.TestCase):
    def test_t0_and_first_still_entry_are_zero(self):
        clock, q = EventClock(), sample(0, 1)
        first = clock.update(q)
        self.assertEqual(first.shape, (6,))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, [0, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(clock.update(q, q), [.01, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(clock.update(q, q), [.02, .01, 1, 0, .01, 0])

    def test_last_command_required_and_explicit_t0_allowed(self):
        clock, q, target = EventClock(), sample(), sample(0, 0)
        np.testing.assert_array_equal(clock.update(q, target), [0, 0, 1, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, 'last issued command'):
            clock.update(q)
        np.testing.assert_allclose(clock.update(q, target), [.01, 0, 1, .01, 0, 1])

    def test_closed_transitions_and_left_right_independence(self):
        clock, q = EventClock(dt=.25, max_seconds=1), sample()
        clock.update(q)
        targets = [sample(0, 1), sample(0, 0), sample(1, 0), sample(0, 0)]
        expected = [[0, 1, 0, 0], [.25, 1, 0, 1], [0, 0, .25, 1], [0, 1, .5, 1]]
        for target, row in zip(targets, expected):
            np.testing.assert_array_equal(clock.update(q, target)[[0, 2, 3, 5]], row)

    def test_still_velocity_boundary_all_seven_joints_and_reentry(self):
        clock = EventClock(dt=.5, still_velocity_rad_s=.125, max_seconds=2)
        q, target = sample(), sample()
        clock.update(q)
        q[:7] += .0625  # Every joint is exactly at the inclusive velocity threshold.
        np.testing.assert_array_equal(clock.update(q, target)[[1, 4]], [0, 0])
        q[:7] += .0625
        np.testing.assert_array_equal(clock.update(q, target)[[1, 4]], [.25, .25])
        q[6] += .125
        np.testing.assert_array_equal(clock.update(q, target)[[1, 4]], [0, .5])
        q[14] += .125
        np.testing.assert_array_equal(clock.update(q, target)[[1, 4]], [0, 0])
        np.testing.assert_array_equal(clock.update(q, target)[[1, 4]], [.25, 0])
        np.testing.assert_array_equal(clock.update(q, target)[[1, 4]], [.5, .25])

    def test_measured_overshoot_is_not_clipped_or_used_as_later_target(self):
        q, target = sample(-.001, 1.001), sample(0, 1)
        q[:7], q[8:15] = -2., 2.
        original, clock = q.copy(), EventClock()
        np.testing.assert_array_equal(clock.update(q), [0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(q, original)
        q[[7, 15]] = [1.002, -.002]
        clock.update(q, target)
        q[[7, 15]] = [-.003, 1.003]
        np.testing.assert_allclose(clock.update(q, target), [.02, .01, 1, 0, .01, 0])
        outside = EventClock(close_threshold=0)
        self.assertEqual(outside.update(sample(-.001, 1.001))[2], 1)

    def test_close_threshold_is_strict_and_zero_velocity_is_valid(self):
        q = sample(.5, np.nextafter(.5, 0.))
        clock = EventClock(still_velocity_rad_s=0)
        np.testing.assert_array_equal(clock.update(q)[[2, 5]], [0, 1])
        clock.update(q, q)
        q[0] += 1e-8
        np.testing.assert_allclose(clock.update(q, q)[[1, 4]], [0, .01])
        self.assertEqual(EventClock(close_threshold=1).update(sample(1, 0))[2], 0)

    def test_four_and_six_seconds_are_distinguishable(self):
        state = np.tile(sample(0, 0), (61, 1))
        features = episode_event_features(state, state)
        # Stillness enters at t1 with zero time, not an invented interval at t0.
        np.testing.assert_allclose(features[40, [0, 1, 3, 4]], [.4, .39, .4, .39])
        np.testing.assert_allclose(features[60, [0, 1, 3, 4]], [.6, .59, .6, .59])

    def test_saturation_and_dt_larger_than_cap(self):
        state = np.tile(sample(0, 0), (100, 1))
        for dt, cap in [(.1, .3), (.5, .2)]:
            features = episode_event_features(state, state, dt=dt, max_seconds=cap)
            np.testing.assert_array_equal(features[-1], np.ones(6))
            self.assertTrue(np.isfinite(features).all())
            self.assertTrue(((0 <= features) & (features <= 1)).all())

    def test_future_actions_cannot_change_any_prefix(self):
        rng = np.random.default_rng(7)
        state = np.tile(sample(0, 1), (20, 1))
        action = rng.normal(size=state.shape)
        action[:, [7, 15]] = rng.integers(0, 2, size=(len(state), 2))
        for initial, first in [(None, [0, 0, 1, 0, 0, 0]),
                               (sample(1, 0), [0, 0, 0, 0, 0, 1])]:
            expected = episode_event_features(state, action, initial_target=initial)
            for t in range(len(state)):
                changed = action.copy()
                changed[t:] += 100
                changed[t:, [7, 15]] = 1 - action[t:, [7, 15]]
                actual = episode_event_features(state, changed, initial_target=initial)
                np.testing.assert_array_equal(actual[:t + 1], expected[:t + 1])
            np.testing.assert_array_equal(expected[0], first)

    def test_initial_target_overrides_only_t0_and_matches_online(self):
        state = np.tile(sample(-.001, 1.001), (4, 1))
        action = np.array([sample(0, 0), sample(0, 1), sample(0, 1), sample(1, 0)])
        initial = sample(1, 0)
        original = initial.copy()
        initial.setflags(write=False)
        kwargs = dict(dt=.5, max_seconds=2)
        actual = episode_event_features(state, action, initial_target=initial, **kwargs)
        np.testing.assert_array_equal(actual, [[0, 0, 0, 0, 0, 1],
                                                [0, 0, 1, .25, 0, 1],
                                                [.25, .25, 1, 0, .25, 0],
                                                [.5, .5, 1, 0, .5, 0]])
        clock = EventClock(**kwargs)
        online = np.array([clock.update(q, action[t - 1] if t else initial)
                           for t, q in enumerate(state)])
        np.testing.assert_array_equal(actual, online)
        np.testing.assert_array_equal(initial, original)

    def test_n_plus_one_commands_and_slice_initialization(self):
        target_command = np.array([sample(1, 0), sample(0, 0), sample(0, 1),
                                   sample(0, 1), sample(1, 1), sample(1, 0)])
        action = target_command[1:]
        state = np.tile(sample(0, 1), (len(action), 1))
        for start in range(len(state)):
            initial = action[start - 1] if start else target_command[0]
            actual = episode_event_features(state[start:], action[start:],
                                            initial_target=initial)
            clock = EventClock()
            online = np.array([clock.update(state[t], target_command[t])
                               for t in range(start, len(state))])
            np.testing.assert_array_equal(actual, online)
            np.testing.assert_array_equal(actual[0, [0, 1, 3, 4]], np.zeros(4))

    def test_initial_target_validation_including_empty_episode(self):
        bad = [np.zeros(15), np.zeros((2, 8)), np.zeros(16, dtype=complex), ['x'] * 16]
        for value in (np.nan, np.inf, -np.inf):
            for index in (0, 7, 14, 15):
                target = sample()
                target[index] = value
                bad.append(target)
        for n in (0, 1):
            state = np.zeros((n, 16))
            for initial in bad:
                with self.subTest(n=n, initial=str(initial)), self.assertRaises(ValueError):
                    episode_event_features(state, state, initial_target=initial)
        empty = episode_event_features(np.empty((0, 16)), np.empty((0, 16)),
                                       initial_target=sample())
        self.assertEqual(empty.shape, (0, 6))
        self.assertEqual(empty.dtype, np.float32)

    def test_offline_online_match_without_mutating_sources(self):
        rng = np.random.default_rng(8)
        state = rng.normal(0, .001, (30, 16)).astype(np.float32)
        action = rng.normal(size=state.shape).astype(np.float32)
        originals = state.copy(), action.copy()
        state.setflags(write=False)
        action.setflags(write=False)
        kwargs = dict(dt=.2, still_velocity_rad_s=.01, max_seconds=.9, close_threshold=.7)
        offline, clock = episode_event_features(state, action, **kwargs), EventClock(**kwargs)
        online = np.array([clock.update(q, action[t - 1] if t else None)
                           for t, q in enumerate(state)])
        np.testing.assert_array_equal(offline, online)
        np.testing.assert_array_equal(state, originals[0])
        np.testing.assert_array_equal(action, originals[1])

    def test_reset_starts_a_new_episode(self):
        q, clock = sample(0, 0), EventClock()
        clock.update(q)
        for _ in range(10):
            clock.update(q, q)
        clock.reset()
        np.testing.assert_array_equal(clock.update(q), [0, 0, 1, 0, 0, 1])
        np.testing.assert_allclose(clock.update(q, q), [.01, 0, 1, .01, 0, 1])
        state = np.tile(q, (4, 1))
        np.testing.assert_array_equal(episode_event_features(state, state),
                                      episode_event_features(state, state))

    def test_owns_measurement_history_and_returned_features(self):
        clock, q, target = EventClock(), sample(0, 0), sample(0, 0)
        clock.update(q)
        clock.update(q, target)[:] = 99
        q[0] += 1
        result = clock.update(q, target)
        np.testing.assert_allclose(result, [.02, 0, 1, .02, .01, 1])

    def test_invalid_states_and_targets_do_not_advance_clock(self):
        bad = [np.zeros(15), np.zeros(17), np.zeros((2, 8)), np.zeros((1, 16)),
               np.zeros(16, dtype=complex), ['x'] * 16, None]
        for value in (np.nan, np.inf, -np.inf):
            for index in (0, 7, 14, 15):
                q = sample()
                q[index] = value
                bad.append(q)
        q, clock = sample(0, 0), EventClock()
        clock.update(q)
        for value in bad:
            with self.subTest(value=str(value)):
                with self.assertRaises(ValueError):
                    clock.update(value, q)
                if value is not None:
                    with self.assertRaises(ValueError):
                        clock.update(q, value)
        np.testing.assert_allclose(clock.update(q, q), [.01, 0, 1, .01, 0, 1])

    def test_invalid_parameters(self):
        bad = dict(dt=[0, -1], still_velocity_rad_s=[-.01],
                   max_seconds=[0, -1], close_threshold=[-.01, 1.01])
        for name, values in bad.items():
            for value in values + [np.nan, np.inf, -np.inf, [.1], 1j, None]:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    EventClock(**{name: value})

    def test_episode_validation_empty_and_single_frame(self):
        empty = episode_event_features(np.empty((0, 16)), np.empty((0, 16)))
        self.assertEqual(empty.shape, (0, 6))
        self.assertEqual(empty.dtype, np.float32)
        np.testing.assert_array_equal(episode_event_features([sample(0, 1)], [sample(1, 0)]),
                                      [[0, 0, 1, 0, 0, 0]])
        state = np.zeros((2, 16))
        for bad in (np.zeros(16), np.zeros((2, 15)), np.zeros((2, 2, 8)),
                    np.full((2, 16), np.nan), np.full((2, 16), np.inf)):
            with self.assertRaises(ValueError):
                episode_event_features(bad, state)
            with self.assertRaises(ValueError):
                episode_event_features(state, bad)
        with self.assertRaises(ValueError):
            episode_event_features(state, np.zeros((3, 16)))
        with self.assertRaises(ValueError):
            episode_event_features(np.empty((0, 16)), np.empty((0, 16)), dt=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
