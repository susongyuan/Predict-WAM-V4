"""Single-step orchestration with mandatory external model/robot adapters.

No WAM, VLM, physical validator or controller is implemented here. Adapter
contracts are documented in README.md; synthetic checks are not robot results.
"""

from dataclasses import asdict, dataclass, replace
import time
from typing import Mapping

from .core import Candidate, Decision, Evidence, GateConfig, Memory, Scope, _number, _text, decide, verified_outcome


VISIBLE_VERIFIERS = {"perception_predicates", "controller_postconditions"}
SENSOR_CHANNELS = {"rgb", "depth", "proprioception"}


@dataclass(frozen=True)
class Observation:
    observation_id: str
    observed_at: float
    sensors: Mapping  # Only permitted sensor channels, never simulator state.


@dataclass(frozen=True)
class Outcome:
    predicates: Mapping[str, bool | None]
    terminal: bool  # The complete skill attempt has ended.
    source: str
    verifier_kind: str
    observation_id: str


@dataclass(frozen=True)
class RunResult:
    status: str
    reason: str
    steps: int
    questions: int
    events: tuple[dict, ...]


def run_episode(adapter, *, scope: Scope, episode_id: str, instruction: str,
                memory: Memory, config: GateConfig, action_low, action_high,
                max_steps=100, max_cycles=200, max_retries=3, ask_budget=2,
                ask_timeout_s=30.0, clock=time.monotonic) -> RunResult:
    """Run one bounded episode; every accepted command is exactly one step.

    Adapter callbacks are synchronous and must obey their own driver timeouts.
    execute() must atomically check observation ID/deadline and return a bool;
    hold() must stop motion using measured controller state, not zero targets.
    Exceptions propagate after hold + queue clearing. No adapter has a default.
    """
    methods = ("observe", "propose", "verify", "execute", "hold", "clear_actions", "ask", "outcome")
    if any(not callable(getattr(adapter, name, None)) for name in methods):
        raise TypeError("Adapter must implement " + ", ".join(methods))
    if (not isinstance(config, GateConfig) or config.commit_steps != 1
            or any(type(n) is not int or n <= 0 for n in (max_steps, max_cycles, max_retries))
            or type(ask_budget) is not int or ask_budget < 0
            or not _number(ask_timeout_s) or ask_timeout_s <= 0
            or not _text(episode_id) or not _text(instruction)):
        raise ValueError("Runtime requires one-step commits, finite budgets and episode/instruction")
    # Validate physical bounds, calibration and the ledger's split/run before motion.
    decide([], action_low=action_low, action_high=action_high, observation_id="preflight",
           now=clock(), config=config, ask_remaining=ask_budget)
    memory.recall(scope, policy_only=True)
    steps = questions = retries = 0
    events, seen_observations = [], set()
    clarification = failure_id = None
    reason = "episode_start"

    def stop(why):
        try:
            adapter.hold(why)
        finally:
            adapter.clear_actions()

    def result(status, why):
        return RunResult(status, why, steps, questions, tuple(events))

    def observe(why):
        observation = adapter.observe(why)
        if (not isinstance(observation, Observation) or not _text(observation.observation_id)
                or not _number(observation.observed_at) or not isinstance(observation.sensors, Mapping)
                or not observation.sensors or not set(observation.sensors) <= SENSOR_CHANNELS):
            raise ValueError("Observation requires a fresh ID/time and only rgb/depth/proprioception")
        if observation.observation_id in seen_observations:
            raise ValueError("Sensor adapter reused an observation ID")
        seen_observations.add(observation.observation_id)
        return observation

    def gate(candidates, observation):
        return decide(candidates, action_low=action_low, action_high=action_high,
                      observation_id=observation.observation_id, now=clock(),
                      config=config, ask_remaining=ask_budget - questions)

    try:
        for cycle in range(max_cycles):
            stop(reason)
            if steps >= max_steps:
                return result("HOLD", "step_budget_exhausted")
            observation = observe(reason)
            age = clock() - observation.observed_at
            if not 0 <= age <= config.max_age_s:
                decision = Decision("OBSERVE", "stale_sensor_observation")
            else:
                proposals = adapter.propose(observation, instruction, clarification,
                                            memory.recall(scope, policy_only=True))
                # A model's self-declared evidence must never authorize its own action.
                unknown = Evidence(None, 0, None, None, None, "")
                if (not isinstance(proposals, (tuple, list))
                        or any(not isinstance(c, Candidate) for c in proposals)):
                    decision = Decision("HOLD", "invalid_candidates")
                else:
                    candidates = [replace(c, evidence=unknown) for c in proposals]
                    decision = gate(candidates, observation)
                    if any(c.observed_at != observation.observed_at for c in candidates):
                        decision = Decision("OBSERVE", "candidate_sensor_time_mismatch")
                    if decision.reason == "unknown_or_uncalibrated_physical_evidence":
                        checked = []
                        for candidate in candidates:
                            evidence = adapter.verify(observation, candidate)
                            if (not isinstance(evidence, Evidence)
                                    or evidence.verifier_kind not in VISIBLE_VERIFIERS):
                                raise ValueError("Physical evidence must come from a policy-visible verifier")
                            checked.append(replace(candidate, evidence=evidence))
                        decision = gate(checked, observation)
                adapter.clear_actions()  # Discard any WAM queue created while proposing.
            event = {"cycle": cycle, "mode": decision.mode, "reason": decision.reason,
                     "observation_id": observation.observation_id}
            events.append(event)
            if decision.mode == "EXECUTE":
                if clock() > decision.expires_at:
                    reason = "expired_before_dispatch"
                    event.update(mode="OBSERVE", reason=reason)
                else:
                    accepted = adapter.execute(decision.actions[0],
                                               observation_id=decision.observation_id,
                                               expires_at=decision.expires_at)
                    if type(accepted) is not bool:
                        raise TypeError("execute must acknowledge one step with a boolean")
                    if not accepted:
                        reason = "controller_rejected_stale_or_unsafe_command"
                        event.update(mode="REPLAN", reason=reason)
                    else:
                        steps += 1
                        stop("verify_postconditions")
                        after = observe("postconditions")
                        if not 0 <= clock() - after.observed_at <= config.max_age_s:
                            return result("HOLD", "stale_postcondition_observation")
                        outcome = adapter.outcome(observation, decision.actions[0], after)
                        if (not isinstance(outcome, Outcome) or not _text(outcome.source)
                                or outcome.verifier_kind not in VISIBLE_VERIFIERS
                                or outcome.observation_id != after.observation_id):
                            raise ValueError("Outcome must be bound to the visible post-step observation")
                        verified = verified_outcome(outcome.predicates, terminal=outcome.terminal)
                        event["outcome"] = {**asdict(outcome), "verified": verified}
                        if verified:
                            record_id = None
                            if not memory.readonly:
                                record_id = memory.record(scope, **asdict(outcome), episode_id=episode_id,
                                                          recovery_of=failure_id if verified == "success" else None)
                            event["memory_record_id"] = record_id
                            if verified == "success":
                                return result("SUCCESS", "verified_goal_postconditions")
                            failure_id = record_id
                            reason = "verified_failure_replan"
                        elif outcome.terminal:
                            return result("HOLD", "terminal_outcome_unverified")
                        else:
                            retries = 0
                            reason = "fresh_observation_after_step"
                            continue
            elif decision.mode == "ASK":
                stop("await_clarification")
                questions += 1
                deadline = clock() + ask_timeout_s
                try:
                    answer = adapter.ask(decision.choices, deadline=deadline)
                except TimeoutError:
                    return result("HOLD", "clarification_timeout")
                if clock() >= deadline or answer is None:
                    return result("HOLD", "clarification_timeout")
                if answer not in decision.choices:
                    return result("HOLD", "clarification_not_a_current_choice")
                clarification, reason = answer, "clarified_reobserve"
                event["answer"] = answer
                continue  # No pre-answer action can survive; acquire new sensors.
            elif decision.mode == "HOLD":
                return result("HOLD", decision.reason)
            else:
                reason = decision.reason
            retries += 1
            if retries >= max_retries:
                return result("HOLD", "retry_budget_exhausted")
        return result("HOLD", "cycle_budget_exhausted")
    finally:
        stop("episode_end")
