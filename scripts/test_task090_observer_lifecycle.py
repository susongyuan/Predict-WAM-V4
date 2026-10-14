#!/usr/bin/env python3
"""CPU-only observer lifecycle regressions; AST policy entry, real gate, stub measurements.

No policy/model constructor, encoder, GPU, simulator, recorded TRAIN window or dense module
is imported. The tracking stub tests ownership/lifecycle, NOT geometric tracking quality.
Run from any cwd: python /absolute/path/scripts/test_task090_observer_lifecycle.py
"""
import ast
import copy
import hashlib
import json
import re
import sys
import time
import unittest
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.grounding import CandidateProgram, Step
from harness.orchestrator import Contract, Directive
from harness.task090_predicates import Task090Postconditions

SOURCE = ROOT / "harness/policy_server.py"
RGB = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
DEPTH = "robot_r1::robot_r1:zed_link:Camera:0::depth_linear"
POSE = "robot_r1::cam_rel_poses"
PROPRIO = "robot_r1::proprio"
PRIV = "privbase_nav::ashcan.n.01_1::ifzxzj"
METHODS = {"_onboard_observation_request", "_reset_onboard_measurement", "_log_task090_measurement",
           "_policy_observation", "_split_navigation_observation", "_find", "_contract_id",
           "_task090_post_update", "_act", "_act_orchestrated", "_return_action", "reset"}


def load_policy_methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WamPolicy")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert {n.name for n in cls.body} == METHODS
    ns = {"np": np, "re": re, "time": time,
          "torch": NS(no_grad=lambda: lambda f: f, from_numpy=np.asarray),
          "hold_baseline": lambda state: state[:, None, :23].copy()}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["WamPolicy"]


Policy = load_policy_methods()


def pick(obj="trash_can"):
    return Step("pick up from", (obj, "floors"))


def nav(obj="trash_can"):
    return Step("move to", (obj,))


def place(obj="trash_can"):
    return Step("place on next to", (obj, "floors", "furniture_sink"))


def observation(frame=1):
    state = np.arange(61, dtype=np.float32) / 100
    camera = np.zeros(7, np.float32)
    camera[6] = 1
    return {RGB: np.full((8, 8, 3), frame % 255, np.uint8),
            DEPTH: np.full((8, 8), 1 + frame / 1000, np.float32),
            POSE: camera, PROPRIO: state}


class TrackingStub:
    """An immutable reference bank plus contract-local candidate credit; not a tracker."""
    max_frame_gap = 1
    requires_applied_action = True

    def __init__(self, reply="normal", mutate=False):
        self.reply, self.mutate = reply, mutate
        self.calls, self.contract_resets = [], []
        self.resets = 0
        self.reset()

    def reset(self):
        self.resets += 1
        self.reference = self.track_id = None
        self.candidate_start, self.candidate_count = None, 0

    def reset_contract(self, contract=None):
        self.contract_resets.append(contract)
        self.candidate_start, self.candidate_count = None, 0

    def update(self, obs, state, **kwargs):
        self.calls.append({"keys": set(obs), "state": state.copy(), **copy.deepcopy(kwargs)})
        if self.reference is None:
            self.reference = np.array([[1., 2., 3.], [4., 5., 6.]])
            self.track_id = f"stub:{self.resets}:{kwargs['episode']}:{kwargs['object_id']}"
        self.candidate_start = kwargs['step'] if self.candidate_start is None else self.candidate_start
        self.candidate_count += 1
        if self.mutate:
            state[:] = -777
            for value in obs.values():
                value[...] = 0
            for feature in kwargs['head_features']:
                feature[...] = -777
            if kwargs.get('action') is not None:
                kwargs['action'][:] = 777
        if self.reply == "raise":
            raise RuntimeError("synthetic_producer_error")
        if self.reply == "claim":
            return {"status": "success", "verified": True, "admitted": True}
        if self.reply == "malformed":
            return None
        return {"status": "measurements_available", "verified": False, "admitted": False,
                "candidate_grasp": self.reply == "candidate", "track_id": self.track_id,
                "candidate_observations": self.candidate_count}


class LegacyStub(TrackingStub):
    reset_contract = None

    def __init__(self):
        self.nested_resets = 0
        self.measure = NS(reset=self.nested_reset)
        super().__init__()

    def nested_reset(self):
        self.nested_resets += 1


def policy(steps=None, producer=True):
    p = Policy()
    p.episode, p.step_in_episode = 2, 1
    p.onboard_measurement = TrackingStub() if producer is True else producer or None
    p.onboard_measurement_every = 5  # old logging interval cannot drop fresh navigation/HOLD frames
    p._onboard_measurement_binding = p._onboard_measurement_contract = None
    p._onboard_measurement_last_frame, p._onboard_measurement_seen = None, set()
    p.rows = []
    p._emit = p.rows.append
    p._last_action = np.linspace(-1, 1, 23, dtype=np.float32)
    p.queue, p._queue_model = [], "main"
    p.orch = NS(candidates=[CandidateProgram("synthetic", steps or [nav(), pick(), place()])],
                index=0, status="running", contract=None, events=[], closure=NS(state=None))
    p._grounded = True
    p.postcondition_mode = "task090_observed"
    p.post_task1 = Task090Postconditions()
    p.perception = NS(detect=lambda obs, state: ([], None))
    p._active_contract = None
    p._last_post_t1 = None
    p._prev_ub = None
    p.model = None
    p._last_pixels = np.zeros((3, 8, 8), np.float32)
    p._route_model = lambda: None
    p._record_v3_frame = lambda *args: None
    p._record_proprio = lambda *args: None
    p._preprocess = lambda obs: (np.array(obs[PROPRIO], copy=True), np.array(obs[RGB], copy=True))
    p._encode = lambda pixels: ([np.array(pixels, dtype=np.float32)], np.array([1., 0.]))
    p.stats = {"steps": 0, "holds": 0, "chunks": 0, "infer_ms": []}
    p._bounded_chunk = lambda chunk: chunk
    p._proprio_input = lambda model, state: state
    p._ensure_resident = lambda *args: None
    p._sample = lambda feats, state, cond: np.full((1, 23), np.mean(feats[0]) + np.mean(state), np.float32)
    p._apply_base_control = lambda chunk, skill: chunk
    p._apply_arm_gain = lambda chunk, state, skill: chunk
    p._proprio_log = p._v3_chunk_log = lambda model: {}
    p.chunk_id, p.exec_horizon, p.base_control_mode = 0, 1, "wam"
    p.tick_inputs = []

    def tick(cls, step, **kwargs):
        p.tick_inputs.append(copy.deepcopy(kwargs))
        c = p.orch.contract
        return Directive("HOLD" if p.orch.status == "held" else "EXECUTE",
                         skill=None if c is None else c.step.skill, condition=np.array([1.]))
    p.orch.tick = tick
    return p


def contract(p, index, attempts=0, started=0):
    p.orch.index = index
    c = Contract(p.orch.candidates[0].steps[index], index, None, 100,
                 arm=NS(arm_id="cpu-stub"), started_step=started, attempts=attempts)
    p.orch.contract = c
    p._active_contract = p._contract_id(c)
    if not c.is_navigation:
        p.post_task1.start(c.key, c.step.skill, c.step.objects, observation()[PROPRIO],
                          p.step_in_episode, episode=p.episode)
    return c


def observe(p, frame=None, obs=None, state=None, features=None):
    obs = observation(p.step_in_episode if frame is None else frame) if obs is None else obs
    p._log_task090_measurement(obs, obs.get(PROPRIO) if state is None else state,
                              [np.ones((1, 6, 4), np.float32)] if features is None else features)
    return p.rows[-1]["measurement"]


class Requests(unittest.TestCase):
    def test_navigation_uses_immediate_manipulation_not_destination(self):
        p = policy([nav("furniture_sink"), place()])
        req = p._onboard_observation_request()
        self.assertEqual(req['object_id'], 'trash_can')
        self.assertEqual(req['measurement_contract_key'], place().key())
        self.assertFalse(req['observed_identity'])

    def test_current_manipulation_is_explicit_intent(self):
        p = policy([pick("half"), place()])
        self.assertEqual(p._onboard_observation_request()['object_id'], 'half')

    def test_no_lookahead_across_another_navigation(self):
        for steps in ([nav()], [nav(), nav(), pick()]):
            with self.subTest(steps=steps):
                self.assertIsNone(policy(steps)._onboard_observation_request())

    def test_agreeing_interpretations_are_allowed(self):
        p = policy()
        p.orch.candidates.append(CandidateProgram("other wording", [nav(), pick(), pick('half')]))
        self.assertEqual(p._onboard_observation_request()['object_id'], 'trash_can')

    def test_divergent_targets_are_unknown_even_on_shared_navigation(self):
        p = policy()
        p.orch.candidates.append(CandidateProgram("other target", [nav(), pick('half')]))
        self.assertIsNone(p._onboard_observation_request())
        self.assertEqual(observe(p)['status'], 'unknown')
        self.assertFalse(p.onboard_measurement.calls)

    def test_instance_tags_are_not_collapsed_by_category_key(self):
        p = policy([nav('trash_can@1'), pick('trash_can@1')])
        p.orch.candidates.append(CandidateProgram("other instance", [nav('trash_can@2'), pick('trash_can@2')]))
        self.assertIsNone(p._onboard_observation_request())

    def test_reference_disagreement_is_not_guessed(self):
        p = policy([pick()])
        p.orch.candidates.append(CandidateProgram("other support", [Step('pick up from', ('trash_can', 'countertop'))]))
        self.assertIsNone(p._onboard_observation_request())

    def test_missing_empty_malformed_and_finished_requests(self):
        for steps in ([], [Step('pick up from', ())], [Step('pick up from', ('',))],
                      [Step('pick up from', (None,))]):
            p = policy()
            p.orch.candidates[0].steps = steps
            self.assertIsNone(p._onboard_observation_request())
        for status in ('done', 'no_candidates', 'invalid'):
            p = policy(); p.orch.status = status
            self.assertIsNone(p._onboard_observation_request())
        p = policy(); p.orch.candidates = []
        self.assertIsNone(p._onboard_observation_request())

    def test_stale_contract_cannot_override_program_target(self):
        p = policy(); c = contract(p, 0)
        c.index = 1
        self.assertIsNone(p._onboard_observation_request())
        c.index = 0; c.step = nav('half')
        self.assertIsNone(p._onboard_observation_request())

    def test_broken_request_is_contained_as_unknown(self):
        p = policy(); p.orch.candidates = [object()]
        self.assertEqual(observe(p)['status'], 'unknown')
        self.assertFalse(p.onboard_measurement.calls)


class Lifecycle(unittest.TestCase):
    def test_navigation_manipulation_and_hold_share_original_track(self):
        p = policy(); contract(p, 0)
        first = observe(p)
        producer = p.onboard_measurement
        reference, resets = producer.reference, producer.resets
        p.step_in_episode += 1; contract(p, 1)
        second = observe(p)
        self.assertEqual(producer.candidate_count, 1)  # new contract only, not old credit
        n = len(producer.contract_resets)
        p.step_in_episode += 1; p.orch.status = 'held'; p.orch.contract = None
        third = observe(p)
        self.assertEqual(len(producer.contract_resets), n)
        self.assertEqual([x['track_id'] for x in (first, second, third)], [first['track_id']] * 3)
        self.assertIs(producer.reference, reference)
        self.assertEqual(producer.resets, resets)
        self.assertEqual([c['step'] for c in producer.calls], [1, 2, 3])

    def test_same_target_place_and_retry_reset_only_candidate_counts(self):
        p = policy(); contract(p, 1); observe(p)
        producer = p.onboard_measurement; ref = producer.reference; track = producer.track_id
        for index, attempt in ((2, 0), (2, 1)):
            p.step_in_episode += 1; contract(p, index, attempts=attempt, started=p.step_in_episode)
            observe(p)
            self.assertEqual(producer.candidate_count, 1)
            self.assertIs(producer.reference, ref)
            self.assertEqual(producer.track_id, track)

    def test_legacy_contract_reset_preserves_reference_and_nested_identity(self):
        producer = LegacyStub(); p = policy(producer=producer)
        contract(p, 0); observe(p)
        ref, resets, nested = producer.reference, producer.resets, producer.nested_resets
        p.step_in_episode += 1; contract(p, 1); observe(p)
        self.assertIs(producer.reference, ref)
        self.assertEqual(producer.resets, resets)
        self.assertEqual(producer.nested_resets, nested)
        self.assertEqual(producer.candidate_count, 1)

    def test_real_v2_contract_reset_and_full_nested_reset(self):
        from harness.task090_rigid_observation_v2 import Task090RigidObservation
        from harness.task090_observation import Task090ObservationProducer
        producer = Task090RigidObservation.__new__(Task090RigidObservation)
        producer.measure = Task090ObservationProducer.__new__(Task090ObservationProducer)
        producer.measure.reset(); producer.reset()
        producer.reference = ref = np.ones((6, 3))
        producer.measure.track_id = 'original-candidate'
        producer.measure.seen.add('old-frame')
        producer.candidate_start, producer.candidate_count = 1, 50
        p = policy(producer=producer)
        p._reset_onboard_measurement(contract_only=True, contract='new-contract')
        self.assertIs(producer.reference, ref)
        self.assertEqual(producer.measure.track_id, 'original-candidate')
        self.assertEqual(producer.candidate_count, 0)
        p._reset_onboard_measurement()
        self.assertIsNone(producer.reference)
        self.assertIsNone(producer.measure.track_id)
        self.assertFalse(producer.measure.seen)

    def test_different_target_resets_identity(self):
        p = policy([pick(), pick('half')]); contract(p, 0); observe(p)
        old = p.onboard_measurement.track_id
        p.step_in_episode += 1; contract(p, 1); observe(p)
        self.assertNotEqual(p.onboard_measurement.track_id, old)
        self.assertEqual(p.onboard_measurement.calls[-1]['object_id'], 'half')

    def test_episode_change_resets_even_with_same_pixels_and_target(self):
        p = policy(); observe(p); old = p.onboard_measurement.track_id
        p.episode += 1; p.step_in_episode = 1; observe(p)
        self.assertNotEqual(p.onboard_measurement.track_id, old)
        self.assertEqual(len(p.onboard_measurement.calls), 2)

    def test_policy_reset_clears_observer_and_replay_history(self):
        p = policy(producer=LegacyStub()); observe(p)
        p.finish_episode = lambda: None
        p._reset_v3_contexts = lambda *args: None
        p._proprio_buf, p._proprio_frames = deque([1]), 1
        p.program = None
        p.reset()
        self.assertEqual((p.episode, p.step_in_episode), (3, 0))
        self.assertIsNone(p.onboard_measurement.reference)
        self.assertIsNone(p._onboard_measurement_binding)
        self.assertIsNone(p._onboard_measurement_last_frame)
        self.assertFalse(p._onboard_measurement_seen)
        observe(p)
        self.assertEqual(p._onboard_measurement_binding, (3, 'trash_can'))

    def test_ambiguity_invalidates_old_binding_without_guessing(self):
        p = policy(); observe(p)
        p.step_in_episode += 1
        p.orch.candidates.append(CandidateProgram('different', [nav(), pick('half')]))
        self.assertEqual(observe(p)['status'], 'unknown')
        self.assertIsNone(p._onboard_measurement_binding)
        self.assertIsNone(p.onboard_measurement.reference)
        self.assertEqual(len(p.onboard_measurement.calls), 1)

    def test_disabled_producer_has_no_diagnostic_or_control_effect(self):
        p = policy(producer=None)
        p._log_task090_measurement({}, np.zeros(61), [])
        self.assertFalse(p.rows)


class FreshnessAndIsolation(unittest.TestCase):
    def unknown(self, result):
        self.assertEqual(result['status'], 'unknown')
        self.assertIs(result['verified'], False)
        self.assertIs(result['admitted'], False)

    def test_duplicate_step_never_reupdates(self):
        p = policy(); observe(p)
        self.unknown(observe(p, frame=2))
        self.assertEqual(len(p.onboard_measurement.calls), 1)
        self.assertEqual(p.onboard_measurement.candidate_count, 0)

    def test_duplicate_rgbd_with_new_step_is_unknown(self):
        p = policy(); observe(p)
        p.step_in_episode += 1
        result = observe(p, frame=1)
        self.unknown(result)
        self.assertIn('repeated_sensor_frame', result['reason'])
        self.assertEqual(len(p.onboard_measurement.calls), 1)

    def test_target_change_does_not_launder_replayed_pixels(self):
        p = policy([pick(), pick('half')]); contract(p, 0); observe(p)
        p.step_in_episode += 1; contract(p, 1)
        self.unknown(observe(p, frame=1))
        self.assertEqual(len(p.onboard_measurement.calls), 1)

    def test_stale_gap_is_unknown_and_preserves_bank_not_credit(self):
        p = policy(); observe(p); ref = p.onboard_measurement.reference
        p.step_in_episode += 3
        result = observe(p)
        self.unknown(result)
        self.assertIn('stale_observation_gap', result['reason'])
        self.assertIs(p.onboard_measurement.reference, ref)
        self.assertEqual(p.onboard_measurement.candidate_count, 0)

    def test_invalid_clock_or_episode_regression_is_unknown(self):
        for episode, step in ((True, 2), (2, -1), (1, 2), (2, 0)):
            p = policy(); observe(p); p.episode, p.step_in_episode = episode, step
            self.unknown(observe(p, obs=observation(2)))
            self.assertEqual(len(p.onboard_measurement.calls), 1)

    def test_missing_and_bad_sensor_inputs_are_unknown(self):
        cases = [{}, {k: v for k, v in observation().items() if k != DEPTH}]
        for key, value in ((RGB, np.ones((8, 8))), (DEPTH, np.zeros((8, 8))),
                           (DEPTH, np.ones((3, 4))), (POSE, np.zeros(7)),
                           (PROPRIO, np.full(61, np.nan))):
            obs = observation(); obs[key] = value; cases.append(obs)
        for obs in cases:
            with self.subTest(keys=list(obs)):
                p = policy(); self.unknown(observe(p, obs=obs))
                self.assertFalse(p.onboard_measurement.calls)

    def test_forbidden_payloads_never_reach_producer(self):
        p = policy(); obs = observation()
        for key in (PRIV, 'object_scope', 'object_poses', 'reward', 'success', 'goal_literals',
                    'q_score', 'robot_r1::reward', 'robot_r1::robot_r1:zed_link:Camera:0::gt', 3):
            obs[key] = object()
        observe(p, obs=obs)
        self.assertEqual(p.onboard_measurement.calls[0]['keys'], {RGB, DEPTH, POSE, PROPRIO})

    def test_producer_mutations_cannot_change_shared_inputs_or_last_action(self):
        p = policy(producer=TrackingStub(mutate=True))
        obs = observation(); saved = copy.deepcopy(obs)
        features = [np.ones((1, 6, 4), np.float32)]; original_features = copy.deepcopy(features)
        previous_action = p._last_action.copy()
        observe(p, obs=obs, features=features)
        for key in obs:
            np.testing.assert_array_equal(obs[key], saved[key])
        np.testing.assert_array_equal(features, original_features)
        np.testing.assert_array_equal(p._last_action, previous_action)

    def test_errors_and_forged_success_are_unknown(self):
        for reply in ('raise', 'claim', 'malformed'):
            p = policy(producer=TrackingStub(reply=reply))
            self.unknown(observe(p))
            self.assertEqual(p.onboard_measurement.candidate_count, 0)

    def test_policy_action_and_real_gate_are_invariant_to_diagnostic_output(self):
        outcomes = []
        for reply in (None, 'normal', 'candidate', 'claim', 'raise', 'malformed'):
            p = policy([pick()], producer=None if reply is None else TrackingStub(reply, mutate=True))
            contract(p, 0)
            first = p._act(observation())
            p.orch.status = 'held'; p.orch.contract = None
            held = p._act(observation(2))
            outcomes.append((first, held, p.tick_inputs, p.orch.index, p._last_post_t1))
            self.assertFalse(p._last_post_t1['verified'])
            self.assertFalse(p._last_post_t1['admitted'])
            self.assertEqual(p.stats['chunks'], 1)
            self.assertEqual(p.stats['holds'], 1)
        for first, held, ticks, index, post in outcomes[1:]:
            np.testing.assert_array_equal(first, outcomes[0][0])
            np.testing.assert_array_equal(held, outcomes[0][1])
            self.assertEqual(index, outcomes[0][3])
            self.assertEqual(post, outcomes[0][4])
            for actual, baseline in zip(ticks, outcomes[0][2]):
                np.testing.assert_array_equal(actual.pop('base_velocity'), baseline['base_velocity'])
                self.assertEqual(actual, {k: v for k, v in baseline.items() if k != 'base_velocity'})

    def test_real_policy_entry_updates_navigation_operation_hold_and_strips_gt(self):
        p = policy(); contract(p, 0)
        packet = object(); seen = []
        p.privbase = NS(finished=False, prepare=lambda value, step: seen.append((value, step)),
                        evidence=lambda *args: None)
        module = ModuleType('harness.privileged_base_navigation'); module.KEY = PRIV
        with patch.dict(sys.modules, {'harness.privileged_base_navigation': module}):
            obs = observation(); obs[PRIV] = packet; obs['reward'] = object()
            p._act(obs)
            ref = p.onboard_measurement.reference
            contract(p, 1); p._act(observation(2))
            p.orch.status = 'held'; p.orch.contract = None; p.privbase.finished = True
            p._act(observation(3))
        self.assertEqual(seen, [(packet, 1)])
        self.assertIs(p.onboard_measurement.reference, ref)
        self.assertEqual(len(p.onboard_measurement.calls), 3)
        self.assertNotIn(PRIV, p.onboard_measurement.calls[0]['keys'])
        self.assertFalse(p._last_post_t1['verified'])

    def test_observation_is_after_grounding_before_tick_and_hold_return(self):
        p = policy(); p._grounded = False; p.orch.candidates = []; p.orch.status = 'held'
        def ground(rgb):
            p._grounded = True
            p.orch.candidates = [CandidateProgram('grounded intent', [nav(), pick()])]
        p._ground = ground
        p._act(observation())
        self.assertEqual(len(p.onboard_measurement.calls), 1)
        self.assertEqual(p.onboard_measurement.calls[0]['object_id'], 'trash_can')
        self.assertEqual(p.stats['holds'], 1)

    def test_no_gpu_model_or_dense_module_loaded(self):
        self.assertNotIn('torch', sys.modules)
        for name in sys.modules:
            self.assertFalse(name.startswith(('transformers', 'timm', 'isaacsim', 'omnigibson')))
            self.assertNotIn('dense_body', name)


if __name__ == '__main__':
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = [SOURCE, Path(__file__).resolve(), ROOT / 'harness/task090_predicates.py',
               ROOT / 'harness/task090_rigid_observation_v2.py', ROOT / 'harness/task090_observation.py',
               ROOT / 'harness/grounding.py', ROOT / 'harness/orchestrator.py',
               ROOT / 'harness/task1_perception_adapter.py']
    print(json.dumps({'scope': 'CPU synthetic lifecycle and control-boundary regressions; not physical closed loop',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'passed': result.wasSuccessful(),
        'elapsed_seconds': time.monotonic() - start, 'python': sys.executable,
        'model_loaded': False, 'gpu_used': False, 'simulator_used': False,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
