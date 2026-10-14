"""CPU state-machine tests with explicit FAKE admission, never physical validation."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.core import Memory, Scope, context_fingerprint
from harness.runtime import Observation, Outcome
from harness.robotwin_task_supervisor import (
    CapabilityCandidate, Directive, GroundingResult, ModelContract, ProviderBinding,
    TaskSupervisor, UserReply, VisibleResult,
)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.target = CapabilityCandidate('put_object_cabinet', 'track-object-1')
        self.other = CapabilityCandidate('put_object_cabinet', 'track-object-2')
        self.scope = Scope('put_object_cabinet', 'put_object_cabinet',
            context_fingerprint({'sensor_setup': 'fake_three_rgb', 'workspace': 'cpu_only'}),
            'fake-aloha-v1', 'fake-native14-v1', 'fake-30k-sha', 'fake-grounder-v1', 'development', 'fake-run')
        self.contract = ModelContract(self.target.capability, self.scope.wam_version,
                                      self.scope.robot_version, self.scope.controller_version)
        self.required = ('object_inside', 'released', 'stable')
        self.ground_calls, self.visible_calls, self.recovery_calls = [], [], []
        self.ground_edit = self.visible_edit = lambda r: r
        self.grounder = ProviderBinding('fake-grounder', 'v1', self.scope.key(), self.ground_callback,
                                       trusted=True, admitted=True)
        self.verifier = ProviderBinding('fake-independent-sensor', 'v1', self.scope.key(), self.visible_callback,
                                       trusted=True, admitted=True, independent=True)
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = Memory(Path(self.tmp.name) / 'ledger.sqlite', split_id=self.scope.split_id, run_id=self.scope.run_id)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.memory.close)
        self.s = self.make()
        self.s.begin_episode('ep1', self.target)
        self.observe()

    def make(self, **overrides):
        args = dict(contract=self.contract, scope=self.scope, required_predicates=self.required,
                    max_age_s=10., ask_timeout_s=3., grounder=self.grounder, verifier=self.verifier,
                    memory=self.memory, clock=lambda: self.now,
                    request_reobserve=self.recovery_callback, request_replan=self.recovery_callback)
        args.update(overrides)
        return TaskSupervisor(**args)

    def observe(self, oid='obs0', *, at=None, sensors=None, visible=None, supervisor=None):
        s = supervisor or self.s
        return s.observe(Observation(oid, self.now if at is None else at,
            sensors if sensors is not None else {'rgb': {'head_camera': 'fake_pixels'},
                                                 'proprioception': {'commanded_joint_gripper_state': [0.] * 14}}),
            visible_objects=(self.target.object_id, self.other.object_id) if visible is None else visible)

    def ground_callback(self, frame):
        self.ground_calls.append(frame)
        return self.ground_edit(GroundingResult('fake-grounder', 'v1', self.scope.key(), frame.episode_id,
            frame.observation.observation_id, frame.observation.observed_at, 'execute', (self.target,)))

    def visible_callback(self, frame):
        self.visible_calls.append(frame)
        return self.visible_edit(VisibleResult('fake-independent-sensor', 'v1', self.scope.key(), frame.episode_id,
            frame.observation.observation_id, frame.observation.observed_at,
            Outcome(dict.fromkeys(self.required), False, 'fake_independent_sensor:frame',
                    'perception_predicates', frame.observation.observation_id), candidate=self.target))

    def recovery_callback(self, request):
        self.recovery_calls.append(request)
        return True

    def rows(self):
        return self.memory.recall(self.scope, policy_only=True)

    def set_outcome(self, values, *, terminal=True, recovery=None, permitted=None):
        self.visible_edit = lambda r: replace(r, outcome=replace(r.outcome, predicates=values, terminal=terminal),
                                             recovery=recovery, recovery_permitted=permitted)

    def success(self):
        self.set_outcome(dict.fromkeys(self.required, True))
        self.s.ground()
        return self.s.evaluate_visible()

    def failure(self, operation='replan', permitted=True):
        self.s.ground()
        self.set_outcome(dict.fromkeys(self.required, False), recovery=operation, permitted=permitted)
        return self.s.evaluate_visible()

    def ask(self):
        self.ground_edit = lambda r: replace(r, status='ask', candidates=(self.target, self.other), visible_ambiguity=True)
        return self.s.ground()

    def reply(self, directive, candidate=None, **changes):
        r = UserReply(directive.question_id, 'ep1', 'obs0', candidate or self.target, self.now)
        return self.s.submit_reply(replace(r, **changes))

    def test_exact_model_contract_no_risk_or_condition_output(self):
        d = self.s.ground()
        self.assertEqual((d.mode, d.candidate), ('EXECUTE', self.target))
        self.assertEqual(self.contract.condition_mode, 'zero_single_task')
        self.assertEqual(self.contract.task_extent, 'whole_task')
        self.assertEqual(self.contract.state_semantics, 'commanded_joint_gripper_state')
        for name in ('condition', 'risk_upper', 'calibration_id', 'actions'):
            self.assertFalse(hasattr(d, name))
        for kw in ({'condition_mode': 'vtt'}, {'state_dim': 61}, {'action_dim': 23},
                   {'execution_horizon': 1}, {'task_extent': 'skill'}, {'rgb_views': ('head_camera',)},
                   {'state_semantics': 'actual_qpos'}, {'condition_dim': 1024.}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.make(contract=replace(self.contract, **kw))

    def test_scope_model_and_provider_version_preflight(self):
        for kw in ({'wam_version': 'other'}, {'controller_version': 'other'}, {'robot_version': 'other'},
                   {'skill_id': 'pick'}):
            with self.assertRaises(ValueError):
                self.make(scope=replace(self.scope, **kw))
        with self.assertRaises(ValueError):
            self.make(grounder=replace(self.grounder, scope_key='wrong'))
        other = Memory(Path(self.tmp.name) / 'other.sqlite', split_id='other', run_id='other')
        self.addCleanup(other.close)
        with self.assertRaises(ValueError):
            self.make(memory=other)

    def test_unknown_unsupported_and_wrong_grounding_ids_hold(self):
        variants = ({'status': 'unknown'}, {'status': 'unsupported'}, {'status': 'stale'},
                    {'episode_id': 'other'}, {'observation_id': 'other'}, {'observed_at': 99.},
                    {'provider_id': 'other'}, {'provider_version': 'other'}, {'scope_key': 'other'},
                    {'candidates': ()}, {'candidates': (self.other,)})
        for i, kw in enumerate(variants):
            self.now += 1
            self.observe(f'obs{i+1}')
            self.ground_edit = lambda r, kw=kw: replace(r, **kw)
            with self.subTest(kw=kw):
                self.assertEqual(self.s.ground().mode, 'HOLD')
        self.assertEqual(self.rows(), [])

    def test_unadmitted_vlm_cannot_self_promote(self):
        self.s.grounder = replace(self.grounder, admitted=False)
        self.ground_edit = lambda r: {'trusted': True, 'admitted': True, 'verified': True, 'status': 'execute'}
        self.assertEqual(self.s.ground().mode, 'HOLD')
        log = [e for e in self.s.events if e['kind'] == 'provider_advisory']
        self.assertTrue(log[0]['advisory_only'])
        self.assertTrue(log[0]['payload']['verified'])
        self.assertEqual(self.rows(), [])

    def test_two_visible_distinct_decisions_only_can_ask(self):
        cases = ((self.target,), (self.target, self.target), (self.target, self.other, self.target),
                 (self.target, CapabilityCandidate(self.target.capability, 'invisible')))
        for i, candidates in enumerate(cases):
            self.now += 1
            self.observe(f'obs{i+1}')
            self.ground_edit = lambda r, c=candidates: replace(r, status='ask', candidates=c, visible_ambiguity=True)
            self.assertEqual(self.s.ground().mode, 'HOLD')
        self.now += 1
        self.observe('last')
        self.ground_edit = lambda r: replace(r, status='ask', candidates=(self.target, self.other), visible_ambiguity=False)
        self.assertEqual(self.s.ground().mode, 'HOLD')

    def test_ask_requires_external_reply_and_is_not_automatic(self):
        d = self.ask()
        self.assertEqual(d.mode, 'ASK')
        self.assertIsNone(d.candidate)
        self.assertEqual(self.s.decision().mode, 'ASK')
        self.assertEqual(self.reply(d).mode, 'EXECUTE')
        self.assertEqual(self.reply(d).mode, 'HOLD')
        self.assertEqual(self.recovery_calls, [])
        self.assertEqual(self.rows(), [])

    def test_selected_other_object_does_not_retarget_zero_policy(self):
        d = self.ask()
        response = self.reply(d, self.other)
        self.assertEqual(response.mode, 'HOLD')
        self.assertEqual(response.reason, 'unsupported_capability_or_object_binding')

    def test_different_capability_is_a_decision_but_not_executable(self):
        alternative = CapabilityCandidate('open_other_container', self.target.object_id)
        self.ground_edit = lambda r: replace(r, status='ask', candidates=(self.target, alternative), visible_ambiguity=True)
        d = self.s.ground()
        self.assertEqual(d.mode, 'ASK')
        self.assertEqual(self.reply(d, alternative).mode, 'HOLD')

    def test_expired_or_wrong_external_reply_hold(self):
        for changes in ({'question_id': 'wrong'}, {'episode_id': 'other'}, {'observation_id': 'other'},
                        {'replied_at': 99.}, {'replied_at': 200.}, {'replied_at': float('nan')}):
            s = self.make()
            s.begin_episode('ep1', self.target)
            self.observe(supervisor=s)
            old, self.s = self.s, s
            d = self.ask()
            self.assertEqual(self.reply(d, **changes).mode, 'HOLD')
            self.s = old
        d = self.ask()
        self.now = d.expires_at
        self.assertEqual(self.reply(d).mode, 'HOLD')

    def test_new_observation_revokes_pending_question(self):
        d = self.ask()
        self.now += 1
        self.observe('obs1')
        self.assertEqual(self.reply(d).mode, 'HOLD')
        self.assertEqual(self.rows(), [])

    def test_question_budget_and_duplicate_grounding(self):
        self.ask()
        self.assertEqual(self.s.ground().mode, 'HOLD')
        self.now += 1
        self.observe('obs1')
        self.assertEqual(self.s.ground().mode, 'HOLD')

    def test_stale_observation_or_slow_provider_never_authorizes(self):
        self.s.ground()
        self.now += 11
        self.assertEqual(self.s.decision().mode, 'HOLD')
        self.assertEqual(self.s.evaluate_visible().mode, 'HOLD')
        self.assertEqual(self.visible_calls, [])
        self.observe('obs1')
        def slow(r):
            self.now += 11
            return r
        self.ground_edit = slow
        self.assertEqual(self.s.ground().mode, 'HOLD')

    def test_ingress_repeated_id_nonmonotonic_and_gt_fail_closed(self):
        self.s.ground()
        self.assertEqual(self.observe().mode, 'HOLD')
        self.assertEqual(self.s.decision().mode, 'HOLD')
        for sensors in ({'gt': True}, {'rgb': {'nested': [{'native_success': True}]}},
                        {'proprioception': {'source': 'evaluation_gt.jsonl'}}):
            self.now += 1
            self.assertEqual(self.observe(str(self.now), sensors=sensors).mode, 'HOLD')
        self.assertEqual(self.rows(), [])

    def test_private_provider_output_never_used_or_logged_as_visible(self):
        self.ground_edit = lambda r: {'status': 'execute', 'nested': [{'source': 'simulator_predicates'}]}
        self.assertEqual(self.s.ground().mode, 'HOLD')
        self.assertNotIn('simulator_predicates', repr(self.s.events))

    def test_default_unknown_and_nonterminal_do_not_write_memory(self):
        self.s.ground()
        self.assertEqual(self.s.evaluate_visible().mode, 'EXECUTE')
        self.assertEqual(self.rows(), [])
        self.now += 1
        self.observe('obs1')
        self.set_outcome(dict.fromkeys(self.required, True), terminal=False)
        self.assertEqual(self.s.evaluate_visible().mode, 'EXECUTE')
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.s.request_replan().mode, 'HOLD')
        self.assertEqual(self.recovery_calls, [])

    def test_partial_unknown_even_with_false_is_not_complete_memory(self):
        self.s.ground()
        self.set_outcome(dict(object_inside=False, released=None, stable=True))
        self.s.evaluate_visible()
        self.assertEqual(self.rows(), [])
        self.s.request_reobserve()
        self.assertEqual(self.recovery_calls, [])

    def test_missing_extra_or_nonboolean_predicates_never_write(self):
        for i, predicates in enumerate(({'object_inside': True}, dict.fromkeys(self.required + ('extra',), True),
                                        dict(object_inside=1, released=True, stable=True))):
            self.now += 1
            self.observe(f'obs{i+1}')
            self.s.ground()
            self.set_outcome(predicates)
            self.assertEqual(self.s.evaluate_visible().mode, 'HOLD')
        self.assertEqual(self.rows(), [])

    def test_only_independent_trusted_admitted_sensor_can_complete(self):
        for kw in ({'trusted': False}, {'admitted': False}, {'independent': False}):
            self.now += 1
            self.observe(str(self.now))
            self.s.ground()
            self.s.verifier = replace(self.verifier, **kw)
            self.set_outcome(dict.fromkeys(self.required, True))
            d = self.s.evaluate_visible()
            self.assertNotEqual(d.mode, 'DONE')
            self.assertIsNone(d.verified_outcome)
            self.assertEqual(self.rows(), [])

    def test_outcome_wrong_identity_scope_or_object_rejected(self):
        for i, kw in enumerate(({'provider_version': 'wrong'}, {'provider_id': 'other'}, {'scope_key': 'wrong'},
                               {'episode_id': 'ep2'}, {'observation_id': 'other'}, {'observed_at': 0.},
                               {'candidate': self.other})):
            self.now += 1
            self.observe(f'o{i}')
            self.s.ground()
            self.visible_edit = lambda r, kw=kw: replace(r, **kw, outcome=replace(r.outcome,
                                terminal=True, predicates=dict.fromkeys(self.required, True)))
            self.assertEqual(self.s.evaluate_visible().mode, 'HOLD')
        self.assertEqual(self.rows(), [])

    def test_native_score_and_spoofed_source_never_sensor_success(self):
        self.s.ground()
        self.visible_edit = lambda r: replace(r, outcome=Outcome(dict.fromkeys(self.required, True), True,
            'native_success', 'controller_postconditions', r.observation_id))
        self.assertEqual(self.s.evaluate_visible().mode, 'HOLD')
        self.assertEqual(self.rows(), [])

    def test_independent_complete_success_recorded_once_and_latched(self):
        result = self.success()
        self.assertEqual(result.mode, 'DONE')
        self.assertEqual(result.verified_outcome, 'success')
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]['observation_id'], 'obs0')
        self.assertEqual(self.s.evaluate_visible(), result)
        self.assertEqual(self.s.submit_reply(None), result)
        self.assertEqual(self.s.request_replan(), result)
        self.assertEqual(self.observe(), result)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(self.visible_calls), 1)

    def test_failure_replan_fresh_grounding_then_linked_recovery(self):
        self.assertEqual(self.failure().verified_outcome, 'failure')
        self.assertEqual(self.rows()[0]['outcome'], 'failure')
        failure_id = self.rows()[0]['id']
        d = self.s.request_replan()
        self.assertEqual(d.reason, 'awaiting_fresh_observation_after_replan')
        self.assertEqual(len(self.recovery_calls), 1)
        self.assertEqual(self.recovery_calls[0].operation, 'replan')
        self.assertEqual(self.s.ground().mode, 'HOLD')
        self.assertEqual(self.s.evaluate_visible().mode, 'HOLD')
        self.now += 1
        self.observe('obs1')
        self.s.ground()
        self.set_outcome(dict.fromkeys(self.required, True))
        self.assertEqual(self.s.evaluate_visible().mode, 'DONE')
        self.assertEqual(self.rows()[0]['outcome'], 'recovery')
        self.assertEqual(self.rows()[0]['recovery_of'], failure_id)

    def test_reobserve_callback_no_client_or_history_reset(self):
        history = ['episode-start', 'obs0']
        callback = Mock(side_effect=lambda req: self.recovery_callback(req))
        self.s._callbacks['reobserve'] = callback
        self.failure('reobserve')
        self.assertIn('awaiting_fresh_observation', self.s.request_reobserve().reason)
        callback.assert_called_once()
        self.assertEqual(history, ['episode-start', 'obs0'])
        self.assertFalse(hasattr(self.s, 'client'))
        self.assertFalse(hasattr(self.s, 'reset'))
        self.assertEqual(self.s.contract.execution_horizon, 8)

    def test_recovery_needs_permission_and_matching_operation(self):
        self.failure(permitted=None)
        self.assertEqual(self.s.request_replan().reason, 'recovery_not_authorized')
        self.assertEqual(self.recovery_calls, [])
        self.s.begin_episode('ep2', self.target)
        self.now += 1
        self.observe('obs2')
        self.failure('reobserve')
        self.assertEqual(self.s.request_replan().reason, 'recovery_not_authorized')
        self.assertEqual(self.s.request_reobserve().reason, 'recovery_not_authorized')
        self.assertEqual(self.recovery_calls, [])

    def test_unavailable_callback_does_not_pretend_to_replan(self):
        self.s._callbacks['replan'] = None
        self.failure()
        self.assertEqual(self.s.request_replan().reason, 'capability_unavailable:replan')
        self.assertEqual(self.s.request_replan().reason, 'recovery_not_authorized')
        self.assertEqual(self.recovery_calls, [])

    def test_callback_exception_no_retry_or_auto_execution(self):
        callback = Mock(side_effect=RuntimeError('clear unsupported'))
        self.s._callbacks['replan'] = callback
        self.failure()
        self.assertEqual(self.s.request_replan().reason, 'recovery_callback_failed_no_retry')
        self.s.request_replan()
        callback.assert_called_once()
        self.now += 1
        self.observe('obs1')
        self.assertEqual(self.s.ground().mode, 'HOLD')
        self.assertEqual(len(self.rows()), 1)

    def test_false_or_expired_recovery_ack_does_not_resume(self):
        self.s._callbacks['replan'] = lambda r: False
        self.failure()
        self.assertIn('not_acknowledged', self.s.request_replan().reason)
        self.assertEqual(self.s.ground().mode, 'HOLD')

    def test_recovery_permit_stale_revoked_or_new_observation_rejected(self):
        self.failure()
        self.s.verifier = replace(self.verifier, admitted=False)
        self.assertEqual(self.s.request_replan().reason, 'recovery_not_authorized')
        self.assertEqual(self.recovery_calls, [])

    def test_duplicate_failure_evidence_not_written_twice(self):
        self.failure()
        self.assertEqual(self.s.evaluate_visible().mode, 'HOLD')
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(self.visible_calls), 1)

    def test_readonly_memory_and_scope_isolation(self):
        ro = Memory(Path(self.tmp.name) / 'ledger.sqlite', split_id=self.scope.split_id, run_id=self.scope.run_id, readonly=True)
        self.addCleanup(ro.close)
        s = self.make(memory=ro)
        s.begin_episode('ro-ep', self.target)
        self.observe(supervisor=s)
        s.ground()
        self.set_outcome(dict.fromkeys(self.required, True))
        self.assertEqual(s.evaluate_visible().mode, 'DONE')
        self.assertEqual(self.rows(), [])

    def test_memory_write_failure_not_retried_on_same_observation(self):
        self.s.ground()
        self.set_outcome(dict.fromkeys(self.required, True))
        with patch.object(self.memory, 'record', side_effect=OSError('disk')) as record:
            self.assertEqual(self.s.evaluate_visible().reason, 'memory_write_failed_no_retry')
            self.s.evaluate_visible()
            record.assert_called_once()
        self.assertEqual(self.rows(), [])

    def test_two_unsupported_choices_do_not_create_a_fake_ask(self):
        a = CapabilityCandidate('unsupported_a', self.target.object_id)
        b = CapabilityCandidate('unsupported_b', self.other.object_id)
        self.ground_edit = lambda r: replace(r, status='ask', candidates=(a, b), visible_ambiguity=True)
        self.assertEqual(self.s.ground().mode, 'HOLD')

    def test_inner_outcome_observation_and_kind_bound(self):
        self.s.ground()
        self.visible_edit = lambda r: replace(r, outcome=replace(r.outcome, observation_id='other',
                    predicates=dict.fromkeys(self.required, True), terminal=True))
        self.assertEqual(self.s.evaluate_visible().mode, 'HOLD')
        self.assertEqual(self.rows(), [])

    def test_freshness_rechecked_after_slow_sensor_callback(self):
        self.s.ground()
        def delayed(r):
            self.now += 11
            return replace(r, outcome=replace(r.outcome, predicates=dict.fromkeys(self.required, True), terminal=True))
        self.visible_edit = delayed
        self.assertEqual(self.s.evaluate_visible().mode, 'HOLD')
        self.assertEqual(self.rows(), [])

    def test_passive_observation_does_not_rerun_grounder_or_change_model(self):
        self.assertEqual(self.s.ground().mode, 'EXECUTE')
        contract = self.s.contract
        self.now += 1
        self.assertEqual(self.observe('obs1').mode, 'EXECUTE')
        self.assertEqual(len(self.ground_calls), 1)
        self.assertIs(self.s.contract, contract)
        self.assertEqual(self.s.decision().expires_at, self.now + 10.)

    def test_new_frame_consumes_old_recovery_permit(self):
        self.failure()
        self.now += 1
        self.observe('obs1')
        self.assertEqual(self.s.request_replan().reason, 'recovery_not_authorized')
        self.assertEqual(self.recovery_calls, [])

    def test_slow_recovery_ack_cannot_resume(self):
        def slow_ack(request):
            self.now += 11
            return True
        self.s._callbacks['replan'] = slow_ack
        self.failure()
        self.assertIn('stale', self.s.request_replan().reason)
        self.assertEqual(self.s.ground().mode, 'HOLD')
        self.assertEqual(len(self.rows()), 1)

    def test_recovery_budget_is_per_episode(self):
        self.failure()
        self.s.request_replan()
        self.now += 1
        self.observe('obs1')
        self.failure()
        self.assertEqual(self.s.request_replan().reason, 'recovery_not_authorized')
        self.assertEqual(len(self.recovery_calls), 1)
        self.assertEqual(len(self.rows()), 2)

    def test_unknown_and_provider_absence_do_not_mean_success(self):
        self.s.ground()
        self.s.verifier = None
        d = self.s.evaluate_visible()
        self.assertEqual(d.mode, 'EXECUTE')
        self.assertIsNone(d.verified_outcome)
        self.assertEqual(self.rows(), [])

    def test_invalid_ingress_cannot_erase_monotonic_watermark(self):
        self.now += 2
        self.observe('obs1')
        self.observe('invalid', sensors={'gt': True})
        self.assertEqual(self.observe('old_unseen_id', at=101.).mode, 'HOLD')
        self.assertEqual(self.s.ground().mode, 'HOLD')
        self.assertEqual(self.ground_calls, [])

    def test_cpu_only_no_policy_simulator_imports(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('sapien', sys.modules)


if __name__ == '__main__':
    unittest.main()
