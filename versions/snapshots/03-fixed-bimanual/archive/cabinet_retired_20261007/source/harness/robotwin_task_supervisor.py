"""Single-threaded CPU whole-task supervisor, not a policy or admission authority.

Host supplies independently reviewed ProviderBinding objects and sensor ingress.
Scope.task_id and Scope.skill_id both name the whole-task contract capability.
Provider booleans are host configuration, NEVER accepted from VLM result payloads.
No providers are trusted by default. Python callbacks are not sandboxed: the host
must keep evaluator/GT handles out of them. No model/client/reset API is retained.

Flow: begin_episode -> observe -> ground -> [submit_reply] -> evaluate_visible.
Call decision() immediately before dispatch; its EXECUTE means task eligibility,
not calibrated physical safety. The transparent adapter still owns transactions.
Callbacks request_reobserve/replan only request passive sensing / queue invalidation;
they must not reset history, execute a new skill, or change the zero condition.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import math
import time
from typing import Callable

from harness.core import Memory, Scope, verified_outcome
from harness.runtime import Observation, Outcome
from harness.robotwin_system_adapter import reject_gt


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class ModelContract:
    capability: str
    wam_version: str
    robot_version: str
    controller_version: str
    task_extent: str = 'whole_task'
    condition_mode: str = 'zero_single_task'
    condition_dim: int = 1024
    state_dim: int = 14
    action_dim: int = 14
    rgb_views: tuple = ('head_camera', 'left_camera', 'right_camera')
    state_semantics: str = 'commanded_joint_gripper_state'
    execution_horizon: int = 8

    def validate(self, scope):
        expected = ('whole_task', 'zero_single_task', 1024, 14, 14,
                    ('head_camera', 'left_camera', 'right_camera'),
                    'commanded_joint_gripper_state', 8)
        actual = (self.task_extent, self.condition_mode, self.condition_dim, self.state_dim,
                  self.action_dim, self.rgb_views, self.state_semantics, self.execution_horizon)
        if actual != expected or any(type(x) is not int for x in
                                     (self.condition_dim, self.state_dim, self.action_dim, self.execution_horizon)):
            raise ValueError('only whole-task zero-conditioned native14/three-RGB/exec8 is supported')
        if not isinstance(scope, Scope):
            raise TypeError('core.Scope required')
        scope.key()
        if (scope.task_id != self.capability or scope.skill_id != self.capability
                or any(not _text(getattr(self, k)) or getattr(self, k) != getattr(scope, k)
                       for k in ('wam_version', 'robot_version', 'controller_version'))):
            raise ValueError('model/scope/version mismatch; skill_id is the whole-task capability')


@dataclass(frozen=True)
class CapabilityCandidate:
    capability: str
    object_id: str  # Sensor-bound instance/track, never a category-only alias or GT actor ID.


@dataclass(frozen=True)
class SensorFrame:
    episode_id: str
    observation: Observation
    visible_objects: tuple
    scope_key: str
    execution_candidate: CapabilityCandidate


@dataclass(frozen=True)
class ProviderBinding:
    provider_id: str
    version: str
    scope_key: str
    callback: Callable
    trusted: bool = False
    admitted: bool = False
    independent: bool = False  # Required for completion/recovery/memory, not a VLM self-report.


@dataclass(frozen=True)
class GroundingResult:
    provider_id: str
    provider_version: str
    scope_key: str
    episode_id: str
    observation_id: str
    observed_at: float
    status: str  # execute | ask | unsupported; anything else is unknown/HOLD.
    candidates: tuple = ()
    visible_ambiguity: bool = False


@dataclass(frozen=True)
class VisibleResult:
    provider_id: str
    provider_version: str
    scope_key: str
    episode_id: str
    observation_id: str
    observed_at: float
    outcome: Outcome
    recovery: str | None = None  # reobserve | replan; only after complete verified failure.
    recovery_permitted: bool | None = None  # Independent provider's explicit evidence, not a risk score.
    candidate: CapabilityCandidate | None = None


@dataclass(frozen=True)
class UserReply:
    question_id: str
    episode_id: str
    observation_id: str
    candidate: CapabilityCandidate
    replied_at: float  # Supplied by the external user-reply ingress on the same monotonic clock.


@dataclass(frozen=True)
class RecoveryRequest:
    episode_id: str
    observation_id: str
    capability: str
    object_id: str
    operation: str
    scope_key: str


@dataclass(frozen=True)
class Directive:
    mode: str
    reason: str
    candidate: CapabilityCandidate | None = None
    question_id: str | None = None
    choices: tuple = ()
    expires_at: float | None = None
    verified_outcome: str | None = None
    memory_record_id: int | None = None


class TaskSupervisor:
    """One whole-task capability per episode; no skill decomposition or condition vector.

    execution_candidate is host-pinned applicability, not a VLM-selected target for
    zero-conditioned WAM. Picking any other capability/object results in HOLD.
    required_predicates is the host's COMPLETE independently reviewed task goal set.
    Its definitions and provider admission must be established outside this module.
    """
    def __init__(self, *, contract, scope, required_predicates, max_age_s, ask_timeout_s,
                 grounder=None, verifier=None, memory=None, request_reobserve=None,
                 request_replan=None, max_questions=1, max_recoveries=1, clock=time.monotonic):
        contract.validate(scope)
        if (not required_predicates or not isinstance(required_predicates, tuple)
                or any(not _text(k) for k in required_predicates)
                or len(set(required_predicates)) != len(required_predicates)):
            raise ValueError('nonempty unique complete predicate tuple required')
        reject_gt(dict.fromkeys(required_predicates))
        if any(not _finite(x) or x <= 0 for x in (max_age_s, ask_timeout_s)):
            raise ValueError('finite positive time limits required')
        if any(type(n) is not int or n < 0 for n in (max_questions, max_recoveries)):
            raise ValueError('nonnegative integer budgets required')
        for provider in (grounder, verifier):
            if provider is not None and (not isinstance(provider, ProviderBinding)
                    or not callable(provider.callback) or not _text(provider.provider_id) or not _text(provider.version)
                    or provider.scope_key != scope.key()):
                raise ValueError('provider registration/scope mismatch')
        if memory is not None:
            if not isinstance(memory, Memory):
                raise TypeError('core.Memory required')
            memory.recall(scope, policy_only=True, limit=1)  # Preflight split/run only, never condition retrieval.
        if any(cb is not None and not callable(cb) for cb in (request_reobserve, request_replan)):
            raise TypeError('recovery callbacks must be callable or absent')
        self.contract, self.scope, self.required = contract, scope, required_predicates
        self.grounder, self.verifier, self.memory, self.clock = grounder, verifier, memory, clock
        self.max_age, self.ask_timeout = max_age_s, ask_timeout_s
        self.max_questions, self.max_recoveries = max_questions, max_recoveries
        self._callbacks = {'reobserve': request_reobserve, 'replan': request_replan}
        self.events, self._episodes = [], set()
        self._episode = self._frame = self._question = self._selected = None
        self._done, self._failed_attempt, self._recovery = False, False, None
        self._directive = Directive('HOLD', 'episode_not_started')

    def _hold(self, reason):
        self._directive = Directive('HOLD', reason)
        return self._directive

    def _advisory(self, kind, reason):
        self.events.append({'kind': kind, 'reason': reason, 'advisory_only': True,
                            'episode_id': self._episode})
        return self.decision()

    def _fresh(self):
        if self._frame is None:
            return False
        now, then = self.clock(), self._frame.observation.observed_at
        return _finite(now) and 0 <= now - then <= self.max_age

    @staticmethod
    def _candidate_valid(candidate):
        return (isinstance(candidate, CapabilityCandidate) and _text(candidate.capability)
                and _text(candidate.object_id))

    def begin_episode(self, episode_id, execution_candidate):
        if not _text(episode_id) or episode_id in self._episodes:
            raise ValueError('unique episode ID required; no policy reset is performed')
        if not self._candidate_valid(execution_candidate) or execution_candidate.capability != self.contract.capability:
            raise ValueError('execution binding must match the whole-task capability')
        self._episodes.add(episode_id)
        self._episode, self._execution_candidate = episode_id, execution_candidate
        self._frame = self._question = self._selected = self._recovery = None
        self._done = self._failed_attempt = self._recovered = False
        self._failure_record = None
        self._seen, self._grounded, self._evaluated = set(), set(), set()
        self._questions = self._recoveries = 0
        self._await_new_frame = False
        self._last_observed_at = None
        return self._hold('awaiting_grounding')

    def observe(self, observation, *, visible_objects):
        if self._done:
            return self.decision()
        self._question = None  # New ingress invalidates a pending question/reply, even if malformed.
        try:
            reject_gt(observation)
            if (self._episode is None or self._done or not isinstance(observation, Observation)
                    or not _text(observation.observation_id) or observation.observation_id in self._seen
                    or not _finite(observation.observed_at) or not _finite(self.clock())
                    or not 0 <= self.clock() - observation.observed_at <= self.max_age
                    or self._last_observed_at is not None and observation.observed_at <= self._last_observed_at
                    or not isinstance(observation.sensors, Mapping) or not observation.sensors
                    or not set(observation.sensors) <= {'rgb', 'depth', 'proprioception'}
                    or not isinstance(visible_objects, tuple) or any(not _text(o) for o in visible_objects)
                    or len(set(visible_objects)) != len(visible_objects)):
                raise ValueError('invalid/stale sensor ingress')
            self._frame = SensorFrame(self._episode, deepcopy(observation), visible_objects, self.scope.key(), self._execution_candidate)
            self._seen.add(observation.observation_id)
            self._last_observed_at = observation.observed_at
            self._await_new_frame = False
            self._recovery = None  # A recovery permit never migrates to another observation.
            if self._selected is not None and not self._failed_attempt and self._directive.mode == 'EXECUTE':
                return self._execute_eligible(self._selected)
            return self._hold('awaiting_grounding' if not self._failed_attempt else 'verified_failure')
        except (ValueError, TypeError):
            self._frame = None
            self._selected = None
            return self._hold('invalid_or_stale_observation')

    def decision(self):
        if self._done:
            return self._directive
        if not self._fresh():
            self._question = None
            return self._hold('observation_unknown_or_stale')
        if self._question is not None and self.clock() >= self._question['deadline']:
            self._question = None
            return self._hold('reply_expired')
        return self._directive

    def _bound(self, result, provider):
        return (self._fresh() and result.provider_id == provider.provider_id
                and result.provider_version == provider.version and result.scope_key == self.scope.key()
                and provider.scope_key == self.scope.key() and result.episode_id == self._episode
                and result.observation_id == self._frame.observation.observation_id
                and type(result.observed_at) in (int, float)
                and result.observed_at == self._frame.observation.observed_at)

    def _call(self, provider, result_type, independent=False):
        if provider is None:
            return None, 'provider_unavailable'
        if provider.scope_key != self.scope.key():
            return None, 'provider_scope_mismatch'
        try:
            result = provider.callback(deepcopy(self._frame))  # No caller-supplied result may self-authorize.
            reject_gt(result)
        except Exception:
            return None, 'provider_error_or_forbidden_gt'
        if (provider.trusted is not True or provider.admitted is not True
                or independent and provider.independent is not True):
            self.events.append({'kind': 'provider_advisory', 'advisory_only': True,
                                'provider_id': provider.provider_id,
                                'payload': asdict(result) if is_dataclass(result) else deepcopy(result)})
            return None, 'unadmitted_provider_advisory'
        if not isinstance(result, result_type) or not self._bound(result, provider):
            return None, 'unknown_stale_or_mismatched_provider_result'
        return result, None

    def _execute_eligible(self, candidate):
        if candidate != self._execution_candidate:
            return self._hold('unsupported_capability_or_object_binding')
        if self._failed_attempt or self._await_new_frame or not self._fresh():
            return self._hold('recovery_or_fresh_observation_required')
        self._selected = candidate
        self._directive = Directive('EXECUTE', 'whole_task_eligible_not_safety_verified', candidate=candidate,
                                    expires_at=self._frame.observation.observed_at + self.max_age)
        return self._directive

    def ground(self):
        if self._done or self._failed_attempt or self._await_new_frame or not self._fresh():
            return self.decision() if self._done else self._hold('grounding_not_ready')
        oid = self._frame.observation.observation_id
        if oid in self._grounded:
            return self._hold('grounding_already_consumed')
        self._grounded.add(oid)
        self._question, self._selected = None, None
        self._hold('grounding_unknown')
        result, error = self._call(self.grounder, GroundingResult)
        if error:
            self._advisory('grounding', error)
            return self._hold(error)
        if result.status == 'unsupported':
            return self._hold('unsupported')
        candidates = result.candidates
        if (not isinstance(candidates, tuple) or any(not self._candidate_valid(c) for c in candidates)
                or any(c.object_id not in self._frame.visible_objects for c in candidates)):
            return self._hold('unknown_or_nonvisible_candidates')
        if result.status == 'execute' and len(candidates) == 1:
            return self._execute_eligible(candidates[0])
        if (result.status != 'ask' or len(candidates) != 2 or candidates[0] == candidates[1]
                or self._execution_candidate not in candidates  # Two unsupported choices are the same HOLD decision.
                or result.visible_ambiguity is not True or self._questions >= self.max_questions):
            return self._hold('unknown_or_nonactionable_ambiguity')
        now = self.clock()
        self._questions += 1
        qid = f'{self._episode}:{oid}:{self._questions}'
        deadline = min(now + self.ask_timeout, self._frame.observation.observed_at + self.max_age)
        self._question = dict(id=qid, choices=candidates, created=now, deadline=deadline)
        self._directive = Directive('ASK', 'visible_decision_ambiguity', question_id=qid,
                                    choices=candidates, expires_at=deadline)
        return self._directive

    def submit_reply(self, reply):
        if self._done:
            return self.decision()
        self.decision()
        q, self._question = self._question, None  # External reply is single use, including invalid/expired.
        if (q is None or not isinstance(reply, UserReply) or not self._fresh()
                or reply.question_id != q['id'] or reply.episode_id != self._episode
                or reply.observation_id != self._frame.observation.observation_id
                or reply.candidate not in q['choices'] or not _finite(reply.replied_at)
                or not q['created'] <= reply.replied_at <= self.clock() < q['deadline']):
            return self._hold('missing_invalid_or_expired_user_reply')
        return self._execute_eligible(reply.candidate)

    def evaluate_visible(self):
        if self._done:
            return self.decision()
        if self._selected is None or self._failed_attempt or self._await_new_frame or not self._fresh():
            return self._hold('visible_evaluation_not_ready')
        oid = self._frame.observation.observation_id
        if oid in self._evaluated:
            return self._hold('visible_result_already_consumed')
        self._evaluated.add(oid)
        result, error = self._call(self.verifier, VisibleResult, independent=True)
        if error:
            if error == 'unadmitted_provider_advisory' or error == 'provider_unavailable':
                return self._advisory('visible_outcome', error)
            return self._hold(error)
        outcome = result.outcome
        if (result.candidate != self._selected or not isinstance(outcome, Outcome) or outcome.observation_id != oid
                or outcome.verifier_kind not in {'perception_predicates', 'controller_postconditions'}
                or not _text(outcome.source) or not isinstance(outcome.predicates, Mapping)
                or set(outcome.predicates) != set(self.required)
                or type(outcome.terminal) is not bool
                or any(v is not None and type(v) is not bool for v in outcome.predicates.values())):
            return self._hold('invalid_or_incomplete_visible_outcome')
        if not outcome.terminal or any(v is None for v in outcome.predicates.values()):
            return self._advisory('visible_outcome', 'unknown_or_nonterminal_no_memory')
        verified = verified_outcome(outcome.predicates, terminal=True)
        record_id = None
        if self.memory is not None and not self.memory.readonly:
            try:
                self.contract.validate(self.scope)
                record_id = self.memory.record(self.scope, predicates=dict(outcome.predicates), terminal=True,
                    episode_id=self._episode, source=outcome.source, verifier_kind=outcome.verifier_kind,
                    observation_id=oid, recovery_of=self._failure_record if self._recovered and verified == 'success' else None)
            except Exception:
                return self._hold('memory_write_failed_no_retry')
        if verified == 'success':
            self._done = True
            self._directive = Directive('DONE', 'independent_complete_visible_outcome', verified_outcome=verified,
                                        memory_record_id=record_id)
        else:
            self._failed_attempt = True
            self._failure_record = record_id
            if result.recovery in ('reobserve', 'replan') and result.recovery_permitted is True:
                self._recovery = (result.recovery, oid, self.verifier)
            self._directive = Directive('HOLD', 'independent_complete_visible_failure', verified_outcome=verified,
                                        memory_record_id=record_id)
        return self._directive

    def _request_recovery(self, operation):
        if self._done:
            return self.decision()
        permit, self._recovery = self._recovery, None  # One attempt even on unavailable/failed callback.
        if (self._done or not self._fresh() or permit != (operation, self._frame.observation.observation_id, self.verifier)
                or not self._failed_attempt or self._recoveries >= self.max_recoveries):
            return self._hold('recovery_not_authorized')
        callback = self._callbacks[operation]
        if callback is None:
            return self._hold('capability_unavailable:' + operation)
        self._recoveries += 1
        self._await_new_frame = True
        request = RecoveryRequest(self._episode, self._frame.observation.observation_id,
                                  self.contract.capability, self._selected.object_id, operation, self.scope.key())
        try:
            ack = callback(request)
        except Exception:
            return self._hold('recovery_callback_failed_no_retry')
        if ack is not True or not self._fresh():
            return self._hold('recovery_callback_not_acknowledged_or_stale')
        self._selected = None
        self._failed_attempt, self._recovered = False, True
        return self._hold('awaiting_fresh_observation_after_' + operation)

    def request_reobserve(self):
        return self._request_recovery('reobserve')

    def request_replan(self):
        return self._request_recovery('replan')
