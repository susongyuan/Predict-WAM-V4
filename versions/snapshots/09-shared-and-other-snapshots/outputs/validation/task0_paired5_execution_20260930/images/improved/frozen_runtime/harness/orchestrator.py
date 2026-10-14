"""Skill-contract orchestrator: the runtime glue between grounding (VLM), execution (WAM), completion
predicates (VTT candidate closure / navigation precondition / independent manipulation postcondition),
bounded asking, and condition-space memory.

One idea, four uses of the same frozen-encoder vector space (docs/system_design.md §3.6, §4):
  * grounding   : VLM -> ordered skill steps (skill, objects); every step maps to a VTT class prototype
  * execution   : the WAM is conditioned on ONE 1024-d condition vector per step (prototype or a memory VTT)
  * completion  : VTT alignment only nominates a manipulation closure; advancing also requires an independent
                  policy-visible postcondition. Navigation closes when CLS(t) reaches the NEXT precondition.
  * memory      : verified realized VTTs become new condition "arms" for the same (skill, objects, scene);
                  a bandit over arms is the self-improvement loop. No weight is updated. Because training used
                  instance VTTs with probability 0.2, memory inputs use the same representation; transfer still needs validation.

Asking is bounded and action-level: the user is asked only when the surviving interpretations disagree on
the NEXT step (grounding.divergence); the shared prefix is executed first. Without a remaining ask budget the
orchestrator HOLDs (no motion) rather than guessing.

Nothing here touches the simulator's ground truth or the BDDL goal; all predicates come from the policy's own
camera features.  Pure Python + numpy; CPU self-test: `python3 -m harness.orchestrator`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import random
from pathlib import Path

import numpy as np

from harness.grounding import CandidateProgram, Catalog, Step, compile_programs, divergence

ROOT = Path(__file__).resolve().parents[1]
NAV_SKILL = "move to"


# ---- condition-space memory: a bandit over verified VTT arms ------------------------------------------------
@dataclass
class Arm:
    arm_id: str
    vector: np.ndarray
    source: str                      # "prototype" | "memory:<run>/ep<k>#<index>"
    successes: int = 0
    failures: int = 0
    support_count: int = 0             # evidence used to construct the condition, not executions using it

    def mean(self, prior=(1.0, 1.0)):
        return (self.successes + prior[0]) / (self.successes + self.failures + sum(prior))


class ConditionBank:
    """Arms per contract key "skill(objects)". `frozen=True` (report profile) never adds arms and picks the
    best posterior mean; online mode adds verified realized VTTs and explores with Thompson sampling."""

    def __init__(self, prototypes: dict, *, frozen=False, seed=0, min_arm_norm=1e-3, max_arms=8, train_targets=None):
        self.arms: dict[str, list[Arm]] = {}
        self.prototypes = prototypes         # skill name -> np.ndarray(1024)
        self.frozen = frozen
        self.rng = random.Random(seed)
        self.min_arm_norm = min_arm_norm
        self.max_arms = max_arms
        self.log: list[dict] = []
        self.targets, self.target_budgets = None, {}
        if train_targets is not None:
            if (not isinstance(train_targets, dict) or train_targets.get("format") != "train-contract-vtt-v1"
                    or train_targets.get("source_split") != "train" or train_targets.get("uses_validation_vectors") is not False
                    or train_targets.get("uses_validation_scalars") is not False
                    or not isinstance(train_targets.get("contracts"), dict) or not isinstance(train_targets.get("navigation"), dict)):
                raise ValueError("Training targets must be a train-only train-contract-vtt-v1 artifact")
            rows = dict(train_targets["contracts"])
            for key, following in train_targets["navigation"].items():
                if not key.startswith("move to(") or not isinstance(following, dict):
                    raise ValueError("Invalid navigation training targets")
                rows.update({self.storage_key(key, NAV_SKILL, nxt): row for nxt, row in following.items()})
            self.targets = {}
            for key, row in rows.items():
                skill = key.partition("(")[0]
                if skill not in prototypes or not isinstance(row, dict):
                    raise ValueError(f"Invalid training target: {key}")
                vector = np.asarray(row.get("vector"), np.float32)
                if (vector.shape != np.asarray(prototypes[skill]).shape or vector.shape != (train_targets.get("dimension"),)
                        or not np.isfinite(vector).all() or np.linalg.norm(vector) < min_arm_norm
                        or type(row.get("train_count")) is not int or row["train_count"] < train_targets.get("min_train_segments", 5)):
                    raise ValueError(f"Invalid training target vector/support: {key}")
                self.targets[key] = vector
                if "median_frames" in row:
                    median = row["median_frames"]
                    if type(median) not in (int, float) or not math.isfinite(median) or median <= 0:
                        raise ValueError(f"Invalid training target duration: {key}")
                    self.target_budgets[key] = median

    @staticmethod
    def storage_key(key, skill, next_key=None):
        return f"{key} -> {next_key}" if skill == NAV_SKILL and next_key is not None else key

    def _default(self, key, skill):
        if self.targets is None:
            return np.asarray(self.prototypes[skill], np.float32), "prototype"
        if key not in self.targets:
            raise ValueError(f"missing_training_condition:{key}")
        return self.targets[key], "train-contract-vtt-v1"

    def _key_arms(self, key: str, skill: str, next_key=None):
        key = self.storage_key(key, skill, next_key)
        if key not in self.arms:
            proto, source = self._default(key, skill)
            # Priors live in mean()/Thompson sampling; they must not appear as observed execution wins.
            self.arms[key] = [Arm("prototype", proto, source)]
        return self.arms[key]

    def select(self, key: str, skill: str, exclude=(), *, next_key=None):
        available = self._key_arms(key, skill, next_key)
        arms = [a for a in available if a.arm_id not in exclude] or available
        if self.frozen or len(arms) == 1:
            return max(arms, key=lambda a: (a.mean(), a.arm_id == "prototype"))
        draws = [(self.rng.betavariate(a.successes + 1, a.failures + 1), a) for a in arms]
        return max(draws, key=lambda d: d[0])[1]

    def update(self, key: str, arm_id: str, success: bool):
        if self.frozen:
            return
        for a in self.arms.get(key, []):
            if a.arm_id == arm_id:
                a.successes += int(success)
                a.failures += int(not success)
                self.log.append({"key": key, "arm": arm_id, "success": success})
                return
        raise KeyError(arm_id)

    def add_memory_arm(self, key: str, skill: str, vector, source: str, verified: bool):
        """Promote a verified realized transition into a new arm. Unverified -> ignored (not even stored)."""
        if self.frozen or verified is not True:
            return None
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != np.asarray(self.prototypes[skill]).shape or not np.isfinite(v).all() or float(np.linalg.norm(v)) < self.min_arm_norm:
            return None
        arms = self._key_arms(key, skill)
        if len(arms) >= self.max_arms:      # keep the prototype + best memory arms
            arms.sort(key=lambda a: (a.arm_id == "prototype", a.mean()), reverse=True)
            del arms[self.max_arms - 1:]
        used = {a.arm_id for a in arms}
        arm_id = next(f"memory{i}" for i in range(self.max_arms + 1) if f"memory{i}" not in used)
        arm = Arm(arm_id, v, source, support_count=1)
        arms.append(arm)
        return arm.arm_id

    def export(self):
        return {k: [{"arm": a.arm_id, "source": a.source, "successes": a.successes, "failures": a.failures,
                     "support_count": a.support_count, "vector": a.vector.tolist()} for a in arms] for k, arms in self.arms.items()}

    def load(self, data: dict):
        """Load an already scope-checked export; a prototype always remains available for retry."""
        if not isinstance(data, dict):
            raise ValueError("Condition-bank arms must be a mapping")
        loaded = {}
        for key, rows in data.items():
            skill = key.partition("(")[0]
            if skill not in self.prototypes or not key.endswith(")") or not isinstance(rows, list) or not rows:
                raise ValueError(f"Invalid condition-bank key/rows: {key}")
            proto, source = self._default(key, skill)
            arms = []
            for r in rows:
                v = np.asarray(r["vector"], np.float32)
                if (v.shape != proto.shape or not np.isfinite(v).all() or np.linalg.norm(v) < self.min_arm_norm
                        or not isinstance(r["arm"], str) or not r["arm"]
                        or not isinstance(r["source"], str) or not r["source"]
                        or any(type(r[k]) is not int or r[k] < 0 for k in ("successes", "failures"))
                        or type(r.get("support_count", 0)) is not int or r.get("support_count", 0) < 0):
                    raise ValueError(f"Invalid condition-bank arm: {key}")
                if r["arm"] == "prototype" and (r["source"] != source or not np.array_equal(v, proto)):
                    raise ValueError(f"Prototype does not match current model: {key}")
                arms.append(Arm(r["arm"], v, r["source"], r["successes"], r["failures"], r.get("support_count", 0)))
            if len({a.arm_id for a in arms}) != len(arms):
                raise ValueError(f"Duplicate condition-bank arm: {key}")
            if not any(a.arm_id == "prototype" for a in arms):
                arms.insert(0, Arm("prototype", proto, source))
            if len(arms) > self.max_arms:
                raise ValueError(f"Too many condition-bank arms: {key}")
            loaded[key] = arms
        self.arms.update(loaded)


# ---- contracts and the step state machine -----------------------------------------------------------------
@dataclass
class Contract:
    step: Step
    index: int
    precondition_key: str | None       # navigation: key of the next manipulation step, else None
    budget_steps: int
    arm_key: str | None = None        # navigation arms also bind the following manipulation target
    arm: Arm | None = None
    started_step: int | None = None
    attempts: int = 0
    tried_arms: list = field(default_factory=list)
    arrival_streak: int = 0
    settling: bool = False
    closure_candidate_emitted: bool = False
    # Irreversible, onboard-unobservable contact skills (Task 0 press): set from candidate evidence of this attempt.
    candidate_kind: bool = False       # at least one candidate-kind evidence record was seen for this attempt
    retry_safe: bool | None = None     # possible contact is sticky; later evidence cannot erase it

    @property
    def key(self):
        return self.step.key()

    @property
    def is_navigation(self):
        return self.step.skill == NAV_SKILL


@dataclass
class Directive:
    """What the policy server does at this control step."""
    mode: str                          # EXECUTE | HOLD | ASK | DONE
    skill: str | None = None
    condition: np.ndarray | None = None
    new_chunk_required: bool = False
    question: tuple = ()
    reason: str = ""
    question_id: int | None = None


class Orchestrator:
    def __init__(self, *, catalog: Catalog, bank: ConditionBank, closure_monitor, precondition_monitor,
                 budgets: dict, ask_budget=2, max_attempts=2, budget_factor=1.5, responder=None, replanner=None,
                 navigation_settle_steps=15, navigation_linear_speed_max=0.05,
                 navigation_yaw_speed_max=0.05, reapproach_on_retry=False, dock_score_min=None,
                 require_manipulation_postcondition=True, navigation_completion="precondition",
                 manipulation_completion="vtt_and_postcondition", rollback_on_release_failure=False, max_rollbacks=2,
                 min_budget_steps=None, candidate_advance_kinds=(), irreversible_skills=(),
                 reapproach_fallback_in_place=False, candidate_outcome_policy="hold"):
        """closure_monitor: harness.vtt_monitor.VTTClosureMonitor-like (start(skill_id, cls), update(cls, cfg)->verdict...)
        precondition_monitor: object with arrived(cls, key)->(verdict, cos, margin) or None (navigation on budget only)
        budgets: "skill(objects)" -> median demo frames; responder(choices)->index|None.
        replanner(failed_key, candidates, failed_step=...) returns full programs preserving the completed prefix, or None.
        navigation_completion: "precondition" (CLS/patch appearance + settle) or "docking" (a policy-visible docking
        controller reports `docked` for this contract key, then the measured base velocity settles).
        manipulation_completion: "vtt_and_postcondition" (VTT candidate AND verified postcondition) or "postcondition"
        (a verified, calibrated postcondition alone decides; a verified negative outcome `failed=True` ends the attempt).
        rollback_on_release_failure: a verified `place in` failure whose object is no longer held (released outside the
        container / not in hand) rewinds to that object's pick-up, so recovery re-approaches and re-grasps it.
        candidate_advance_kinds (historical name, default none): accepted unobservable candidate kinds.
        candidate_outcome_policy="hold" (default) stops at a satisfied candidate without completing the contract;
        independent verification/safe recovery requires a separately admitted path, not another action or blind retry.
        Explicit "legacy_advance" preserves historical unverified advancement for reproduction only; its event
        success is None, never True, and it cannot update reliability or memory.
        irreversible_skills (default none; Task 0: ("press",)): skills whose effect cannot be observed onboard and
        would be undone by a repetition (a second press toggles the radio off). Such a contract (or any contract that
        produced candidate-kind evidence) is never started twice in an episode once it advanced, and after a failed
        attempt it is retried only when the attempt's latest evidence says retry_safe=True (no possible contact);
        otherwise the orchestrator HOLDs with reason irreversible_outcome_unknown instead of re-executing blindly.
        reapproach_fallback_in_place (default False; Task 0): when reapproach_on_retry cannot compile a re-approach only
        because that approach was never trained (Task 0: move to(coffee_table) before a place-on retry), retry the
        contract in place instead of holding with invalid_skill_program (the retry is still bounded by max_attempts)."""
        self.catalog, self.bank = catalog, bank
        self.closure, self.precond = closure_monitor, precondition_monitor
        self.budgets, self.budget_factor = budgets, budget_factor
        self.ask_budget, self.max_attempts = ask_budget, max_attempts
        self.responder, self.replanner = responder, replanner
        self.navigation_settle_steps = int(navigation_settle_steps)
        self.navigation_linear_speed_max = float(navigation_linear_speed_max)
        self.navigation_yaw_speed_max = float(navigation_yaw_speed_max)
        self.reapproach_on_retry = bool(reapproach_on_retry)
        self.require_manipulation_postcondition = bool(require_manipulation_postcondition)
        if navigation_completion not in ("precondition", "docking"):
            raise ValueError("navigation_completion must be precondition or docking")
        if manipulation_completion not in ("vtt_and_postcondition", "postcondition"):
            raise ValueError("manipulation_completion must be vtt_and_postcondition or postcondition")
        self.navigation_completion, self.manipulation_completion = navigation_completion, manipulation_completion
        self.rollback_on_release_failure, self.max_rollbacks = bool(rollback_on_release_failure), int(max_rollbacks)
        # skill -> minimum contract budget (steps); e.g. a docking search needs a full in-place scan
        self.min_budget_steps = {str(k): int(v) for k, v in (min_budget_steps or {}).items()}
        if candidate_outcome_policy not in ("hold", "legacy_advance"):
            raise ValueError("candidate_outcome_policy must be hold or legacy_advance")
        self.candidate_outcome_policy = candidate_outcome_policy
        self.candidate_advance_kinds = tuple(str(k) for k in (candidate_advance_kinds or ()))
        self.irreversible_skills = tuple(str(k) for k in (irreversible_skills or ()))
        self.reapproach_fallback_in_place = bool(reapproach_fallback_in_place)
        if any(k in ("perception_predicates", "controller_postconditions") for k in self.candidate_advance_kinds):
            raise ValueError("candidate_advance_kinds must not name a verified evidence kind")
        # None disables the monocular dock sub-gate (falls back to the CLS-cosine + settle arrival only).
        self.dock_score_min = None if dock_score_min is None else float(dock_score_min)
        if (self.navigation_settle_steps < 1 or self.navigation_linear_speed_max <= 0
                or self.navigation_yaw_speed_max <= 0):
            raise ValueError("Navigation settle thresholds must be positive")
        self.events: list[dict] = []
        self._question_serial = 0
        self.reset([])

    # -- episode lifecycle
    def reset(self, candidates: list[CandidateProgram], visible_objects=None):
        self.visible_objects = list(self.catalog.categories if visible_objects is None else visible_objects)
        compiled, self.program_rejections = compile_programs(candidates, self.catalog, self.visible_objects)
        self.candidates = [] if self.program_rejections else compiled
        self.index = 0
        self.asks_left = self.ask_budget
        self.contract: Contract | None = None
        self.realized: list[dict] = []
        self.completed_steps: list[Step] = []
        # A manipulation retry can insert a new navigation contract. Keep the logical attempt count and tried
        # condition arms across that insertion so recovery stays bounded instead of retrying forever.
        self._manip_retry_state: dict[tuple[int, str], dict] = {}
        self._rollbacks: dict[str, int] = {}
        # irreversible contract key -> how its (only) execution ended; blocks any re-execution in this episode
        self._irreversible_attempted: dict[str, str] = {}
        self.status = "running" if self.candidates else "no_candidates"
        self.pending_hold_reason = (None if self.candidates else
                                    "invalid_skill_program" if self.program_rejections else "no_grounded_program")
        self.pending_question = ()
        self.question_id = None
        if self.program_rejections:
            self._emit(event="program_rejected", reasons=self.program_rejections)

    def answer(self, answer_index, *, question_id) -> bool:
        """Select only a currently pending choice; stale/invalid replies never resume motion."""
        if (self.status != "awaiting_answer" or type(question_id) is not int or question_id != self.question_id
                or type(answer_index) is not int or not 0 <= answer_index < len(self.candidates)):
            return False
        compiled, rejected = compile_programs([self.candidates[answer_index]], self.catalog, self.visible_objects,
                                              completed_prefix=self.completed_steps)
        if rejected:
            self.program_rejections = rejected
            self.status, self.pending_hold_reason = "held", "invalid_skill_program"
            self._emit(event="program_rejected", reasons=rejected)
            return False
        self._emit(event="answer", index=self.index, question_id=question_id, answer=answer_index,
                   interpretation=self.candidates[answer_index].interpretation)
        self.candidates = compiled
        self.pending_question, self.question_id = (), None
        self.status, self.pending_hold_reason = "running", None
        return True

    def _emit(self, **rec):
        self.events.append(rec)

    def _program(self):
        return self.candidates[0].steps if self.candidates else []

    def _navigation_precondition(self, candidate, index):
        # Consecutive navigation steps cannot borrow a later target's manipulation predicate.
        nxt = candidate.steps[index + 1] if index + 1 < len(candidate.steps) else None
        return nxt.key() if nxt is not None and nxt.skill != NAV_SKILL else None

    def _make_contract(self, index):
        steps = self._program()
        step = steps[index]
        pre_key = None
        if step.skill == NAV_SKILL:
            pre_key = self._navigation_precondition(self.candidates[0], index)
        arm_key = self.bank.storage_key(step.key(), step.skill, pre_key)
        median = self.bank.target_budgets.get(arm_key, self.budgets.get(step.key(), self.budgets.get(step.skill, 900)))
        contract = Contract(step, index, pre_key, max(int(median * self.budget_factor), self.min_budget_steps.get(step.skill, 0)),
                            arm_key=arm_key)
        if step.skill != NAV_SKILL:
            state = self._manip_retry_state.get(self._manip_retry_key(step))
            if state is not None:
                contract.attempts = state["attempts"]
                contract.tried_arms = list(state["tried_arms"])
        return contract

    def _manip_retry_key(self, step):
        # Navigation recovery does not advance the manipulation phase, so the key survives inserted approaches.
        phase = sum(done.skill != NAV_SKILL for done in self.completed_steps)
        return phase, step.key()

    def _start_contract(self, contract: Contract, cls_now, step_now, exclude=()):
        if contract.key in getattr(self, "_irreversible_attempted", {}):
            # Defense in depth for irreversible, unobservable skills: no recovery/replan path may execute it again.
            reason = "irreversible_skill_already_attempted"
            self.status, self.pending_hold_reason = "held", reason
            self._emit(event="hold", index=contract.index, key=contract.key, reason=reason, step=step_now,
                       previous_outcome=self._irreversible_attempted[contract.key])
            return False
        try:
            contract.arm = self.bank.select(contract.key, contract.step.skill, exclude=exclude, next_key=contract.precondition_key)
        except ValueError as error:
            self.status, self.pending_hold_reason = "held", str(error)
            self._emit(event="hold", index=contract.index, key=contract.key, reason=str(error), step=step_now)
            return False
        contract.tried_arms.append(contract.arm.arm_id)
        contract.started_step = step_now
        contract.attempts += 1
        if not contract.is_navigation:
            self._manip_retry_state[self._manip_retry_key(contract.step)] = {
                "attempts": contract.attempts, "tried_arms": list(contract.tried_arms)}
        self.closure.start(contract.step.skill, cls_now, condition=contract.arm.vector)
        self._emit(event="contract_start", index=contract.index, key=contract.key, arm=contract.arm.arm_id,
                   arm_key=contract.arm_key, condition_source=contract.arm.source,
                   attempt=contract.attempts, budget=contract.budget_steps, precondition=contract.precondition_key, step=step_now)
        return True

    # -- the per-control-step call
    def tick(self, cls_now, step_now: int, base_velocity=None, dock_score=None, dock_threshold=None,
             postcondition=None, motion=None, navigation_evidence=None) -> Directive:
        if self.status != "running":
            return Directive("DONE" if self.status == "done" else "HOLD", question=self.pending_question,
                             question_id=self.question_id, reason=self.pending_hold_reason or self.status)
        # 1) Ask when this skill or the predicate needed to verify its navigation differs.
        prefix, choices = divergence(self.candidates, self.index)
        nav_ambiguity = (prefix > 0 and self._program()[self.index].skill == NAV_SKILL
                         and len({self._navigation_precondition(c, self.index) for c in self.candidates}) > 1)
        if nav_ambiguity:
            choices = tuple(c.interpretation for c in self.candidates)
        if prefix == 0 and choices or nav_ambiguity:
            if self.asks_left <= 0:
                self.pending_hold_reason = "ambiguous_without_ask_budget"
                self.status = "held"
                self._emit(event="hold", index=self.index, reason=self.pending_hold_reason, step=step_now)
                return Directive("HOLD", question=choices, reason=self.pending_hold_reason)
            self.asks_left -= 1
            self._question_serial += 1
            self.pending_question, self.question_id = choices, self._question_serial
            self.status, self.pending_hold_reason = "awaiting_answer", "awaiting_answer"
            self._emit(event="ask", index=self.index, choices=list(choices), question_id=self.question_id, step=step_now)
            answer = self.responder(choices) if self.responder is not None else None
            if not self.answer(answer, question_id=self.question_id):
                if self.status == "held":
                    return Directive("HOLD", reason=self.pending_hold_reason)
                return Directive("ASK", question=choices, question_id=self.question_id, reason="intent_changes_next_skill")

        steps = self._program()
        if self.index >= len(steps):
            self.status = "done"
            return Directive("DONE", reason="program_complete")

        # 2) contract bookkeeping
        new_chunk = False
        if self.contract is None or self.contract.index != self.index:
            self.contract = self._make_contract(self.index)
            if not self._start_contract(self.contract, cls_now, step_now):
                return Directive("HOLD", reason=self.pending_hold_reason)
            new_chunk = True
        c = self.contract
        elapsed = step_now - c.started_step

        # Contact occurrence and task completion are different facts. This gate precedes BOTH
        # manipulation completion modes, so a VTT closure cannot bypass an uncertain irreversible effect.
        candidate = None if c.is_navigation else self._candidate_evidence(
            postcondition, key=c.key, observation_id=step_now)
        if candidate is not None:
            c.candidate_kind = True
            c.retry_safe = c.retry_safe is not False and candidate.get("retry_safe") is True
            if candidate["satisfied"] and self.candidate_outcome_policy == "hold":
                reason = "candidate_requires_independent_verification"
                self._irreversible_attempted[c.key] = reason
                self._finish(c, cls_now, step_now, None, {
                    "predicate": reason, "verified": False, "outcome_observable": False,
                    "candidate_occurred": True, "postcondition": dict(candidate)})
                self.pending_hold_reason, self.status = reason, "held"
                self.contract = None
                self._manip_retry_state.pop(self._manip_retry_key(c.step), None)
                self._emit(event="hold", index=self.index, key=c.key, reason=reason, step=step_now,
                           candidate_occurred=True, verified=False, success=None)
                return Directive("HOLD", reason=reason)

        # 3) completion predicate
        verdict, info = self._completion(c, cls_now, elapsed, base_velocity=base_velocity, dock_score=dock_score,
                                         dock_threshold=dock_threshold, postcondition=postcondition, motion=motion,
                                         navigation_evidence=navigation_evidence)
        if verdict == "settle":
            if not c.settling:
                c.settling = True
                self._emit(event="navigation_settle", index=c.index, key=c.key, step=step_now,
                           speed=list(info["base_velocity"]), required_steps=self.navigation_settle_steps)
            # This is a transient controller hold, not a declared failure or a satisfied precondition.
            return Directive("HOLD", reason="navigation_settle")
        if verdict is True:
            self._finish(c, cls_now, step_now, True, info)
            if not c.is_navigation:
                self._manip_retry_state.pop(self._manip_retry_key(c.step), None)
            if self._irreversible(c):
                self._irreversible_attempted[c.key] = str(info.get("predicate"))
            self.completed_steps.append(c.step)
            self.index += 1
            self.contract = None
            # Re-enter the ambiguity gate before ANY action under the next skill.
            return self.tick(cls_now, step_now, base_velocity=base_velocity)
        elif verdict is False:
            self._finish(c, cls_now, step_now, False, info)
            if self._rollback_release_failure(c, info, step_now):
                return self.tick(cls_now, step_now, base_velocity=base_velocity)
            if self._irreversible(c) and c.retry_safe is not True:
                # The attempt may have had its (unobservable) effect: repeating it could undo it. Fail closed.
                reason = "irreversible_outcome_unknown"
                self._irreversible_attempted[c.key] = reason
                self.pending_hold_reason, self.status = reason, "held"
                self.contract = None
                self._manip_retry_state.pop(self._manip_retry_key(c.step), None)
                self._emit(event="hold", index=self.index, key=c.key, reason=reason, step=step_now,
                           retry_safe=c.retry_safe, attempts=c.attempts)
                return Directive("HOLD", reason=reason)
            if c.attempts < self.max_attempts:
                if not c.is_navigation and self.reapproach_on_retry:
                    compiled, rejected = compile_programs(self.candidates, self.catalog, self.visible_objects,
                                                          completed_prefix=self.completed_steps, restart=True)
                    if not rejected and compiled:
                        old_index = self.index
                        self.candidates, self.contract = compiled, None
                        self._emit(event="recovery_reapproach", index=old_index, key=c.key, step=step_now,
                                   next_key=self._program()[self.index].key(), attempt=c.attempts + 1)
                        return self.tick(cls_now, step_now, base_velocity=base_velocity)
                    if (self.reapproach_fallback_in_place and rejected
                            and all(str(r).startswith("untrained_navigation:") for r in rejected)):
                        self._emit(event="recovery_in_place", index=self.index, key=c.key, step=step_now,
                                   reasons=list(rejected), attempt=c.attempts + 1)
                    else:
                        self.program_rejections = rejected or ["empty_recovery_program"]
                        self.pending_hold_reason, self.status = "invalid_skill_program", "held"
                        self.contract = None
                        self._emit(event="program_rejected", reasons=self.program_rejections, step=step_now)
                        self._emit(event="hold", index=self.index, key=c.key, reason=self.pending_hold_reason, step=step_now)
                        return Directive("HOLD", reason=self.pending_hold_reason)
                if not self._start_contract(c, cls_now, step_now, exclude=c.tried_arms):
                    return Directive("HOLD", reason=self.pending_hold_reason)
                new_chunk = True
            else:
                replan = self.replanner(c.key, self.candidates, failed_step=c.step) if self.replanner else None
                if replan:
                    compiled, rejected = compile_programs(replan, self.catalog, self.visible_objects,
                                                          completed_prefix=self.completed_steps, restart=True)
                    if not rejected:
                        self._emit(event="replan", index=self.index, key=c.key, step=step_now, n_candidates=len(compiled))
                        self.candidates, self.contract = compiled, None
                        return self.tick(cls_now, step_now, base_velocity=base_velocity)
                    self.program_rejections = rejected
                    self._emit(event="program_rejected", reasons=rejected, step=step_now)
                self.pending_hold_reason = "invalid_skill_program" if self.program_rejections else "required_skill_failed"
                self.status = "held"
                self.contract = None
                if not c.is_navigation:
                    self._manip_retry_state.pop(self._manip_retry_key(c.step), None)
                self._emit(event="hold", index=self.index, key=c.key, reason=self.pending_hold_reason, step=step_now)
                return Directive("HOLD", reason=self.pending_hold_reason)
        return Directive("EXECUTE", skill=c.step.skill, condition=c.arm.vector, new_chunk_required=new_chunk)

    def _rollback_release_failure(self, c: Contract, info: dict, step_now: int) -> bool:
        """A verified `place in` failure with the object out of the hand: rewind to that object's pick-up (a new
        approach is compiled in), bounded by max_rollbacks per object. Returns True when the program was rewound."""
        reason = (info.get("postcondition") or {}).get("reason")
        if (not self.rollback_on_release_failure or c.is_navigation or c.step.skill != "place in"
                or reason not in ("released_outside_container", "precondition_object_not_in_hand")):
            return False
        obj = c.step.objects[0]
        if self._rollbacks.get(obj, 0) >= self.max_rollbacks:
            return False
        picks = [i for i, s in enumerate(self.completed_steps) if s.skill == "pick up from" and s.objects[0] == obj]
        if not picks:
            return False
        prefix = self.completed_steps[:picks[-1]]
        compiled, rejected = compile_programs(self.candidates, self.catalog, self.visible_objects,
                                              completed_prefix=prefix, restart=True)
        if rejected or not compiled:
            self._emit(event="program_rejected", reasons=rejected or ["empty_rollback_program"], step=step_now)
            return False
        self._rollbacks[obj] = self._rollbacks.get(obj, 0) + 1
        self._emit(event="recovery_rollback", index=self.index, key=c.key, object=obj, reason=reason, step=step_now,
                   rewind_to=len(prefix), rollbacks=self._rollbacks[obj])
        self.candidates, self.contract = compiled, None
        self.completed_steps, self.index = prefix, len(prefix)
        self._manip_retry_state = {}
        return True

    @staticmethod
    def _postcondition_evidence(postcondition, *, key, observation_id):
        """Validate policy-visible postcondition evidence; malformed/self-declared evidence fails closed."""
        if not isinstance(postcondition, dict) or postcondition.get("verified") is not True:
            return None
        if postcondition.get("verifier_kind") not in ("perception_predicates", "controller_postconditions"):
            return None
        if not isinstance(postcondition.get("source"), str) or not postcondition["source"]:
            return None
        if (postcondition["source"].startswith("patch_postcondition_gate:")
                or postcondition.get("key") != key
                or type(postcondition.get("observation_id")) is not int
                or postcondition["observation_id"] != observation_id
                or type(postcondition.get("satisfied")) is not bool):
            return None
        return postcondition

    def _irreversible(self, c: Contract):
        return not c.is_navigation and (c.step.skill in self.irreversible_skills or c.candidate_kind)

    def _candidate_evidence(self, postcondition, *, key, observation_id):
        """Candidate evidence of an unobservable outcome (e.g. press contact): accepted only for configured kinds, only
        when it does not claim verification and states outcome_observable=False; same identity checks as verified
        evidence. It stops execution by default; explicit legacy mode may advance but never counts as verified."""
        if not self.candidate_advance_kinds or not isinstance(postcondition, dict):
            return None
        if (postcondition.get("verifier_kind") not in self.candidate_advance_kinds
                or postcondition.get("verified") is not False or postcondition.get("outcome_observable") is not False):
            return None
        if (not isinstance(postcondition.get("source"), str) or not postcondition["source"]
                or postcondition["source"].startswith("patch_postcondition_gate:")
                or postcondition.get("key") != key
                or type(postcondition.get("observation_id")) is not int
                or postcondition["observation_id"] != observation_id
                or type(postcondition.get("satisfied")) is not bool):
            return None
        return postcondition

    def _completion(self, c: Contract, cls_now, elapsed, *, base_velocity=None, dock_score=None, dock_threshold=None,
                    postcondition=None, motion=None, navigation_evidence=None):
        if c.is_navigation and self.navigation_completion == "docking":
            ev = navigation_evidence if isinstance(navigation_evidence, dict) else {}
            docked = ev.get("docked") is True and ev.get("key") == c.key and ev.get("observation_id") == c.started_step + elapsed
            if docked:
                velocity = np.asarray(base_velocity if base_velocity is not None else [np.inf] * 3, np.float32)
                stable = (velocity.shape == (3,) and np.isfinite(velocity).all()
                          and float(np.linalg.norm(velocity[:2])) <= self.navigation_linear_speed_max
                          and abs(float(velocity[2])) <= self.navigation_yaw_speed_max)
                c.arrival_streak = c.arrival_streak + 1 if stable else 0
                info = {"predicate": "docking_settle", "base_velocity": velocity.tolist(), "stable_steps": c.arrival_streak,
                        "dock_error": ev.get("error_norm"), "dock_target": ev.get("dock_target"), "verified": False}
                if c.arrival_streak >= self.navigation_settle_steps:
                    info["predicate"] = "docking"
                    return True, info
                return "settle", info
            c.arrival_streak, c.settling = 0, False
            if elapsed >= c.budget_steps:
                return False, {"predicate": "budget", "elapsed": elapsed, "docking_mode": ev.get("mode")}
            return None, {}
        if not c.is_navigation and self.manipulation_completion == "postcondition":
            cfg = self.closure.config_for(c.step.skill, max_steps=c.budget_steps)
            _, progress, alignment = self.closure.update(cls_now, cfg, motion=motion)   # logged; never decides
            evidence = self._postcondition_evidence(postcondition, key=c.key, observation_id=c.started_step + elapsed)
            candidate = self._candidate_evidence(postcondition, key=c.key, observation_id=c.started_step + elapsed)
            if candidate is not None:
                c.candidate_kind = True
            accepted = evidence if evidence is not None else candidate
            if accepted is not None and (candidate is not None or c.step.skill in self.irreversible_skills):
                # Once evidence permits possible contact, later records cannot make a blind retry safe.
                c.retry_safe = c.retry_safe is not False and accepted.get("retry_safe") is True
            info = {"predicate": "calibrated_postcondition", "progress": progress, "alignment": alignment,
                    "verified": evidence is not None and evidence["satisfied"]}
            if isinstance(postcondition, dict):
                info["postcondition"] = {k: postcondition.get(k) for k in
                                         ("satisfied", "failed", "reason", "verified", "source", "verifier_kind",
                                          "observation_id", "key", "detail")}
                info["postcondition"]["accepted_as_evidence"] = evidence is not None
                for k in ("outcome_observable", "retry_safe"):     # candidate-kind predicates only (Task 0 press)
                    if k in postcondition:
                        info["postcondition"][k] = postcondition[k]
                if candidate is not None:
                    info["postcondition"]["accepted_as_candidate"] = True
            if evidence is not None and evidence["satisfied"] is True:
                return True, info
            if candidate is not None and candidate["satisfied"] is True:
                # Only the explicitly selected legacy path can reach unverified advancement.
                info.update(predicate="unverified_outcome_candidate", verified=False, outcome_observable=False,
                            candidate_occurred=True)
                return (True if self.candidate_outcome_policy == "legacy_advance" else "candidate_hold"), info
            if evidence is not None and postcondition.get("failed") is True:
                info.update(verified=True, predicate="calibrated_postcondition_failed")
                return False, info
            if candidate is not None and candidate.get("failed") is True:
                info.update(verified=False, predicate="candidate_postcondition_failed")
                return False, info
            if elapsed >= c.budget_steps:
                info.update(predicate="budget_without_verified_postcondition", elapsed=elapsed, verified=False)
                return False, info
            return None, info
        if c.is_navigation:
            if self.precond is not None and c.precondition_key is not None:
                verdict, cos, margin = self.precond.arrived(cls_now, c.precondition_key)
                # Monocular dock sub-gate: the DINO-CLS cosine cannot separate a good trunk-facing dock from an
                # oblique mis-dock (both ~0.87-0.95), so when a dock_score is supplied require it to clear the
                # offline-validated threshold before arrival counts. A missing/NaN score (car not seen) is treated
                # as not-yet-docked, keeping the robot navigating rather than declaring a false arrival.
                # `dock_threshold` (per-key, from the generic PatchArrivalGate) overrides the global dock_score_min.
                thr = dock_threshold if dock_threshold is not None else self.dock_score_min
                dock_ok = thr is None or (dock_score is not None and np.isfinite(dock_score) and dock_score >= thr)
                if verdict is True and dock_ok and elapsed >= max(30, c.budget_steps // 10):
                    velocity = np.asarray(base_velocity if base_velocity is not None else [np.inf] * 3, np.float32)
                    stable = (velocity.shape == (3,) and np.isfinite(velocity).all()
                              and float(np.linalg.norm(velocity[:2])) <= self.navigation_linear_speed_max
                              and abs(float(velocity[2])) <= self.navigation_yaw_speed_max)
                    c.arrival_streak = c.arrival_streak + 1 if stable else 0
                    if c.arrival_streak < self.navigation_settle_steps:
                        return "settle", {"predicate": "precondition_settle", "cos": cos, "margin": margin,
                                          "base_velocity": velocity.tolist(), "stable_steps": c.arrival_streak}
                    return True, {"predicate": "precondition", "cos": cos, "margin": margin,
                                  "base_velocity": velocity.tolist(), "stable_steps": c.arrival_streak,
                                  "dock_score": (None if dock_score is None or not np.isfinite(dock_score) else float(dock_score)),
                                  "dock_threshold": (None if thr is None else float(thr))}
                c.arrival_streak = 0
                c.settling = False
            if elapsed >= c.budget_steps:
                return False, {"predicate": "budget", "elapsed": elapsed}
            return None, {}
        cfg = self.closure.config_for(c.step.skill, max_steps=c.budget_steps)
        vtt_verdict, progress, alignment = self.closure.update(cls_now, cfg, motion=motion)
        evidence = self._postcondition_evidence(postcondition, key=c.key, observation_id=c.started_step + elapsed)
        info = {"predicate": "vtt_candidate_and_postcondition", "vtt_candidate": vtt_verdict is True,
                "progress": progress, "alignment": alignment, "verified": evidence is not None and evidence["satisfied"],
                "motion_window_path": getattr(self.closure, "last_motion_path", None)}
        if isinstance(postcondition, dict):
            info["postcondition"] = {k: postcondition.get(k) for k in
                                     ("satisfied", "verified", "source", "verifier_kind", "observation_id", "key")}
            info["postcondition"]["accepted_as_evidence"] = evidence is not None
        # Enforce the contract budget outside VTTClosureMonitor: a persistent visual false positive otherwise
        # wins over its `elif max_steps` branch forever.
        if elapsed >= c.budget_steps:
            info.update(predicate="budget_without_verified_postcondition", elapsed=elapsed, verified=False)
            return False, info
        if vtt_verdict is True:
            if not c.closure_candidate_emitted:
                c.closure_candidate_emitted = True
                self._emit(event="closure_candidate", index=c.index, key=c.key, step=c.started_step + elapsed,
                           progress=progress, alignment=alignment,
                           postcondition=(None if evidence is None else info["postcondition"]))
            if not self.require_manipulation_postcondition:
                info.update(predicate="vtt_closure_unverified", verified=False)
                return True, info
            if evidence is not None and evidence["satisfied"] is True:
                return True, info
            return None, info
        if vtt_verdict is False:
            info.update(predicate="vtt_stall_or_budget", verified=False)
            return False, info
        return None, info

    def _finish(self, c: Contract, cls_now, step_now, success: bool | None, info: dict):
        # Legacy advancement is a control-flow decision, not a successful manipulation outcome.
        advanced_unverified = success is True and info.get("candidate_occurred") is True and not info.get("verified")
        if advanced_unverified:
            info = {**info, "legacy_program_advanced": True}
            success = None
        vtt = np.asarray(cls_now, np.float32) - np.asarray(self.closure.state.start_cls, np.float32)
        if info.get("verified") is True:
            self.bank.update(c.arm_key, c.arm.arm_id, success)
        self.realized.append({"index": c.index, "key": c.key, "arm": c.arm.arm_id, "arm_key": c.arm_key, "success": success,
                              "skill": c.step.skill, "objects": list(c.step.objects), "closure": success,
                              "steps": step_now - c.started_step, "end_step": step_now, "info": info, "vtt": vtt.tolist()})
        self._emit(event="contract_end", index=c.index, key=c.key, arm=c.arm.arm_id, arm_key=c.arm_key, success=success,
                   steps=step_now - c.started_step, step=step_now, **{k: v for k, v in info.items() if k != "vtt"})
        if success and not c.is_navigation:
            # VTT alignment is a completion heuristic, not independent evidence of task success.
            self.bank.add_memory_arm(c.key, c.step.skill, vtt, f"realized#{c.index}@{step_now}",
                                     verified=info.get("verified") is True)

    def flush_open_contract(self, cls_now, step_now):
        """Episode ended (evaluator reset): the open contract is unverified -> recorded as unknown, never promoted."""
        if self.contract is not None and self.contract.started_step is not None and self.closure.state is not None:
            vtt = np.asarray(cls_now, np.float32) - np.asarray(self.closure.state.start_cls, np.float32)
            self.realized.append({"index": self.contract.index, "key": self.contract.key, "arm": self.contract.arm.arm_id,
                                  "arm_key": self.contract.arm_key,
                                  "skill": self.contract.step.skill, "objects": list(self.contract.step.objects), "closure": None,
                                  "success": None, "steps": step_now - self.contract.started_step, "end_step": step_now,
                                  "info": {"predicate": "episode_end"}, "vtt": vtt.tolist()})


# ---- adapter over harness.vtt_monitor.VTTClosureMonitor: closure measured against the ARM actually used ----
class ClosureAdapter:
    """Progress/alignment of the realized transition against the condition vector the WAM was given (prototype
    or memory arm), with the per-class thresholds/stall rule of VTTClosureMonitor. Skill names in, not ids."""

    def __init__(self, monitor, configs: dict):
        self.monitor, self.configs = monitor, configs      # configs: class_id -> ClosureConfig
        self.state = None
        self._unit = self._norm = None
        self._motion = []                                  # per-step upper-body joint motion (policy-visible proprio)
        self.last_motion_path = None

    def start(self, skill, cls_now, condition=None):
        cid = int(self.monitor.classes[skill])
        self.state = self.monitor.start(cid, cls_now)
        v = np.asarray(condition if condition is not None else self.monitor.proto[cid], np.float64)
        self._norm = float(np.linalg.norm(v)); self._unit = v / (self._norm + 1e-8)
        self._motion, self.last_motion_path = [], None

    def window_path(self, cfg):
        """Upper-body joint path over the last cfg.motion_window steps, or None until the window is full."""
        w = int(cfg.motion_window)
        if len(self._motion) < w:
            return None
        return float(sum(self._motion[-w:]))

    def config_for(self, skill, max_steps):
        cfg = self.configs[int(self.monitor.classes[skill])]
        cfg.max_steps = int(max_steps)
        return cfg

    def signals(self, cls_now):
        s = self.state
        r = np.asarray(cls_now, np.float64) - s.start_cls
        rn = float(np.linalg.norm(r))
        return float(r @ self._unit / (self._norm + 1e-8)), (float(r @ self._unit / (rn + 1e-8)) if rn > 1e-6 else 0.0)

    def update(self, cls_now, cfg, motion=None):
        """motion: this step's upper-body joint motion (L1, rad) from the robot's own proprio, or None if unknown.
        A VTT plateau is a stall only if the robot is also idle over cfg.motion_window steps (cfg.motion_eps set and
        motion supplied); without motion evidence the legacy VTT-only rule applies."""
        s = self.state
        s.steps += 1
        progress, alignment = self.signals(cls_now)
        if progress > s.best_progress + cfg.stall_delta:
            s.best_progress, s.best_step = progress, s.steps
        s.history.append((s.steps, round(progress, 4), round(alignment, 4)))
        if motion is not None and np.isfinite(motion):
            self._motion.append(float(motion))
            if len(self._motion) > int(cfg.motion_window):
                del self._motion[:-int(cfg.motion_window)]
        self.last_motion_path = self.window_path(cfg) if self._motion else None
        if s.steps >= cfg.min_steps and progress >= cfg.done_progress and alignment >= cfg.done_alignment:
            return True, progress, alignment
        if cfg.max_steps is not None and s.steps >= cfg.max_steps:
            return False, progress, alignment
        vtt_plateau = s.steps - s.best_step >= cfg.stall_steps and s.steps >= cfg.min_steps
        if vtt_plateau:
            if cfg.motion_eps is None or not self._motion:
                return False, progress, alignment            # legacy rule (no motion evidence available)
            if self.last_motion_path is not None and self.last_motion_path <= cfg.motion_eps:
                return False, progress, alignment            # plateau AND idle robot -> stalled
        return None, progress, alignment


# ---- helpers for the policy server ---------------------------------------------------------------------------
def fixed_program_candidates(path=ROOT / "configs/task013_fixed_schedule.json"):
    prog = json.loads(Path(path).read_text())
    steps = [Step(p["skill"], tuple(p["objects"])) for p in prog["program"]]
    budgets = {s.key(): p["budget_frames"] for s, p in zip(steps, prog["program"])}
    return [CandidateProgram("fixed training program", steps)], budgets


def budgets_from_program(path=ROOT / "configs/task013_skill_program.json"):
    prog = json.loads(Path(path).read_text())
    out = {}
    for row in prog["skills"]:
        out[Step(row["skill"], tuple(row["objects"])).key()] = row["median_frames"]
        out.setdefault(row["skill"], row["median_frames"])
    return out


# ---- self-test ----------------------------------------------------------------------------------------------
class _FakeClosure:
    """Deterministic closure stand-in: manipulation closes after `close_after` steps with the prototype arm,
    fails (budget) with a 'bad' arm. Mirrors VTTClosureMonitor's start/update/state/config_for surface."""
    class State:
        def __init__(self, cls): self.start_cls = np.asarray(cls, np.float32)
    class Cfg:
        def __init__(self, max_steps): self.max_steps = max_steps

    def __init__(self, close_after=40, fail_arms=()):
        self.close_after, self.fail_arms, self.state, self.t, self.cond = close_after, set(fail_arms), None, 0, None

    def start(self, skill, cls, condition=None):
        self.state, self.t, self.cond = self.State(cls), 0, condition

    def config_for(self, skill, max_steps):
        return self.Cfg(max_steps)

    def update(self, cls, cfg, motion=None):
        self.t += 1
        bad = self.cond is not None and float(self.cond[0]) < 0
        if not bad and self.t >= self.close_after:
            return True, 1.0, 0.9
        if self.t >= cfg.max_steps:
            return False, 0.1, 0.1
        return None, 0.2, 0.3


class _FakePrecond:
    def __init__(self, arrive_after): self.arrive_after, self.t = arrive_after, 0
    def arrived(self, cls, key):
        self.t += 1
        return (True if self.t >= self.arrive_after else None), 0.5, 0.1


def _self_test():
    cat = Catalog()
    protos = {name: np.full(1024, 0.1, np.float32) for name in cat.patterns}
    cands, budgets = fixed_program_candidates()
    # 1) unambiguous program: navigation closes by precondition, manipulation by closure; no unverified promotion
    bank = ConditionBank(protos, seed=1)
    orch = Orchestrator(catalog=cat, bank=bank, closure_monitor=_FakeClosure(close_after=20), precondition_monitor=_FakePrecond(45),
                        budgets=budgets, budget_factor=1.0, responder=lambda ch: 0,
                        require_manipulation_postcondition=False)
    orch.reset(cands)
    d = None
    for t in range(6000):
        d = orch.tick(np.full(1024, 0.3 + 0.001 * t, np.float32), t, base_velocity=np.zeros(3, np.float32))
        if d.mode == "DONE":
            break
    assert d.mode == "DONE" and orch.index == 16, (d, orch.index)
    ends = [e for e in orch.events if e["event"] == "contract_end"]
    assert len(ends) == 16 and all(e["success"] for e in ends), ends[:3]
    assert "open lid(car)" in bank.arms and len(bank.arms["open lid(car)"]) == 1  # heuristic closure cannot certify memory
    assert all(len(arms) == 1 for key, arms in bank.arms.items() if key.startswith("move to(car) -> "))
    # 2) ambiguity: two candidates share a 2-step prefix, ask happens exactly at step index 2, answer picks candidate 1
    common = [Step("move to", ("car",)), Step("open lid", ("car",))]
    a = CandidateProgram("camera first", common + [Step("move to", ("digital",)), Step("pick up from", ("digital", "coffee_table"))])
    b = CandidateProgram("racket first", common + [Step("move to", ("tennis",)), Step("pick up from", ("tennis", "coffee_table"))])
    asked = []
    orch2 = Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(20), precondition_monitor=_FakePrecond(45),
                         budgets=budgets, budget_factor=1.0, responder=lambda ch: asked.append(ch) or 1,
                         require_manipulation_postcondition=False)
    orch2.reset([a, b])
    for t in range(3000):
        if orch2.tick(np.zeros(1024, np.float32), t, base_velocity=np.zeros(3, np.float32)).mode == "DONE":
            break
    ask = [e for e in orch2.events if e["event"] == "ask"]
    assert len(ask) == 1 and ask[0]["index"] == 2 and asked == [("camera first", "racket first")], ask
    assert orch2.candidates[0].interpretation == "racket first" and orch2.index == 4
    # 3) no ask budget -> HOLD at divergence, no motion, nothing executed beyond the shared prefix
    orch3 = Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(20), precondition_monitor=_FakePrecond(45),
                         budgets=budgets, budget_factor=1.0, ask_budget=0, responder=lambda ch: 0,
                         require_manipulation_postcondition=False)
    orch3.reset([a, b])
    modes = [orch3.tick(np.zeros(1024, np.float32), t, base_velocity=np.zeros(3, np.float32)).mode for t in range(400)]
    assert modes[-1] == "HOLD" and "EXECUTE" in modes and orch3.index == 2, (modes[-1], orch3.index)
    assert not any(e["event"] == "contract_start" and e["index"] >= 2 for e in orch3.events)
    # 4) failure -> retry with a different arm; closure alone never updates verified reliability statistics
    bank4 = ConditionBank(protos, seed=3)
    bank4.add_memory_arm("open lid(car)", "open lid", np.full(1024, -1.0, np.float32), "memory:bad", verified=True)
    bank4.arms["open lid(car)"][1].successes = 50          # bias Thompson towards the bad arm first
    orch4 = Orchestrator(catalog=cat, bank=bank4, closure_monitor=_FakeClosure(20), precondition_monitor=_FakePrecond(45),
                         budgets=budgets, budget_factor=0.2, responder=lambda ch: 0,
                         require_manipulation_postcondition=False)
    orch4.reset([CandidateProgram("lid only", [Step("open lid", ("car",))])])
    for t in range(3000):
        if orch4.tick(np.zeros(1024, np.float32), t, base_velocity=np.zeros(3, np.float32)).mode == "DONE":
            break
    ends4 = [e for e in orch4.events if e["event"] == "contract_end" and e["key"] == "open lid(car)"]
    assert [e["success"] for e in ends4] == [False, True] and ends4[0]["arm"] != ends4[1]["arm"], ends4
    assert bank4.arms["open lid(car)"][1].failures == 0
    frozen = ConditionBank(protos, frozen=True)
    frozen.select("open lid(car)", "open lid")
    before = frozen.export()
    frozen.update("open lid(car)", "prototype", False)
    assert frozen.export() == before
    assert frozen.add_memory_arm("open lid(car)", "open lid", np.ones(1024), "x", verified=True) is None
    assert ConditionBank(protos).add_memory_arm("open lid(car)", "open lid", np.ones(1024), "x", verified=False) is None
    # 5) export/load round trip
    b5 = ConditionBank(protos); b5.load(bank4.export()); assert len(b5.arms["open lid(car)"]) == 2
    # 6) Different program lengths are an intent choice; completion cannot select the short program silently.
    short = CandidateProgram("stop after lid", common)
    orch3.reset([short, a])
    for t in range(400):
        d = orch3.tick(np.zeros(1024, np.float32), t, base_velocity=np.zeros(3, np.float32))
    assert d.mode == "HOLD" and orch3.index == 2
    # 7) A missing navigation verifier times out; required failures never fall through to the next skill.
    failed = Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(20),
                          precondition_monitor=None, budgets={"move to(car)": 2}, max_attempts=2, budget_factor=1,
                          require_manipulation_postcondition=False)
    failed.reset([short])
    for t in range(10):
        d = failed.tick(np.zeros(1024, np.float32), t)
    assert d.mode == "HOLD" and failed.index == 0 and d.reason == "required_skill_failed"
    # 8) Eviction must not give two live arms the same identity.
    capped = ConditionBank(protos, max_arms=3)
    for i in range(12):
        capped.add_memory_arm("open lid(car)", "open lid", np.ones(1024), str(i), verified=True)
        arms = capped.arms["open lid(car)"]
        assert len({a.arm_id for a in arms}) == len(arms) <= 3
    # 9) A pending asynchronous question is emitted once, blocks motion, and rejects stale/invalid replies.
    pending = Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(2),
                           precondition_monitor=None, budgets=budgets, ask_budget=1,
                           require_manipulation_postcondition=False)
    pending.reset([CandidateProgram("open", [Step("open lid", ("car",))]),
                   CandidateProgram("close", [Step("close lid", ("car",))])])
    d = pending.tick(np.zeros(1024), 0)
    assert d.mode == "ASK" and pending.asks_left == 0 and pending.contract is None
    assert pending.tick(np.zeros(1024), 1).mode == "HOLD" and pending.asks_left == 0
    assert not pending.answer(True, question_id=d.question_id)
    assert not pending.answer(1, question_id=d.question_id + 1)
    assert pending.answer(1, question_id=d.question_id)
    resumed = pending.tick(np.zeros(1024), 2)
    assert resumed.mode == "EXECUTE" and resumed.skill == "move to" and resumed.new_chunk_required
    assert pending.candidates[0].steps == [Step("move to", ("car",)), Step("close lid", ("car",))]
    assert not pending.answer(0, question_id=d.question_id)
    # 10) Visual arrival alone cannot advance navigation: measured base velocity must be present and stable.
    settle = Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(20),
                          precondition_monitor=_FakePrecond(1),
                          budgets={"move to(car)": 100, "open lid(car)": 20}, budget_factor=1,
                          navigation_settle_steps=2, require_manipulation_postcondition=False)
    settle.reset([CandidateProgram("settle", common)])
    for t in range(40):
        d = settle.tick(np.zeros(1024, np.float32), t)
    assert settle.index == 0 and d.mode == "HOLD" and d.reason == "navigation_settle"
    assert settle.tick(np.zeros(1024, np.float32), 40, base_velocity=np.zeros(3, np.float32)).reason == "navigation_settle"
    d = settle.tick(np.zeros(1024, np.float32), 41, base_velocity=np.zeros(3, np.float32))
    assert settle.index == 1 and d.mode == "EXECUTE" and d.skill == "open lid"
    # 11) An enabled manipulation retry first inserts a fresh approach and keeps the retry count bounded.
    recovery = Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(999),
                            precondition_monitor=_FakePrecond(1),
                            budgets={"move to(car)": 100, "open lid(car)": 2}, budget_factor=1,
                            max_attempts=2, navigation_settle_steps=1, reapproach_on_retry=True,
                            require_manipulation_postcondition=False)
    recovery.reset([CandidateProgram("recover", common)])
    for t in range(80):
        d = recovery.tick(np.zeros(1024, np.float32), t, base_velocity=np.zeros(3, np.float32))
        if any(e["event"] == "recovery_reapproach" for e in recovery.events):
            break
    assert recovery.index == 1 and recovery.contract.is_navigation
    assert recovery.candidates[0].steps == [common[0], common[0], common[1]]
    assert recovery._manip_retry_state[(0, "open lid(car)")]["attempts"] == 1
    before = b5.export()
    try:
        b5.load({"open lid(car)": [{"arm": "memory0", "source": "test", "successes": 0, "failures": 0, "vector": [1.0]}]})
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong-dimensional memory was accepted")
    assert b5.export() == before
    # 12) Production-safe manipulation mode: VTT alone is only a candidate and cannot advance. A complete,
    # policy-visible independent evidence record is required; missing evidence fails closed at the budget.
    lid_program = CandidateProgram("lid", [Step("move to", ("car",)), Step("open lid", ("car",))])
    guarded = Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(2),
                           precondition_monitor=_FakePrecond(1),
                           budgets={"move to(car)": 100, "open lid(car)": 5}, max_attempts=1, budget_factor=1,
                           navigation_settle_steps=1)
    guarded.reset([lid_program])
    for t in range(50):
        d = guarded.tick(np.zeros(1024, np.float32), t, base_velocity=np.zeros(3, np.float32))
    assert d.mode == "HOLD" and guarded.index == 1 and d.reason == "required_skill_failed"
    assert any(e["event"] == "closure_candidate" for e in guarded.events)
    assert not any(e["event"] == "contract_end" and e["key"] == "open lid(car)" and e["success"]
                   for e in guarded.events)
    # Regression for the formerly trusted patch END gate: an appearance match must still time out, not win.
    guarded.reset([lid_program])
    for t in range(50):
        d = guarded.tick(np.full(1024, t * 0.01, np.float32), t, base_velocity=np.zeros(3, np.float32),
                         postcondition={"satisfied": True, "verified": True, "source": "patch_postcondition_gate:v1",
                                        "verifier_kind": "perception_predicates", "observation_id": t, "key": "open lid(car)"})
    assert d.mode == "HOLD" and guarded.index == 1
    assert not any(a.successes or a.support_count for arms in guarded.bank.arms.values() for a in arms)
    verified = Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(2),
                            precondition_monitor=_FakePrecond(1),
                            budgets={"move to(car)": 100, "open lid(car)": 20}, max_attempts=1, budget_factor=1,
                            navigation_settle_steps=1)
    verified.reset([lid_program])
    evidence = {"satisfied": True, "verified": True, "source": "synthetic://visible-postcondition",
                "verifier_kind": "perception_predicates", "observation_id": 2, "key": "open lid(car)"}
    # Even a historically 'verified' patch score, wrong target or stale observation cannot certify a skill.
    for invalid in ({**evidence, "source": "patch_postcondition_gate:v1"},
                    {**evidence, "key": "close lid(car)"}, {**evidence, "observation_id": 1}):
        assert verified._postcondition_evidence(invalid, key="open lid(car)", observation_id=2) is None
    assert verified._postcondition_evidence(evidence, key="open lid(car)", observation_id=2) is evidence
    for t in range(50):
        evidence["observation_id"] = t
        d = verified.tick(np.full(1024, t * 0.01, np.float32), t, base_velocity=np.zeros(3, np.float32),
                          postcondition=evidence)
        if d.mode == "DONE":
            break
    assert d.mode == "DONE" and verified.index == 2
    assert any(e["event"] == "contract_end" and e["key"] == "open lid(car)" and e["success"]
               and e.get("verified") is True for e in verified.events)
    memories = [a for a in verified.bank.arms["open lid(car)"] if a.arm_id != "prototype"]
    assert len(memories) == 1 and memories[0].successes == 0 and memories[0].support_count == 1
    # 13) Irreversible skill with an onboard-unobservable outcome (Task 0 press; generic mechanism shown on the
    # Task-13 catalog): a satisfied contact CANDIDATE advances unverified (no reliability/memory update) and the
    # skill is never executed twice; a failed attempt is retried only if its evidence says retry_safe.
    def irreversible(max_attempts=2, **kw):
        o = Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(2),
                         precondition_monitor=_FakePrecond(1), budgets={"move to(car)": 100, "open lid(car)": 20},
                         max_attempts=max_attempts, budget_factor=1, navigation_settle_steps=1,
                         manipulation_completion="postcondition", candidate_advance_kinds=("contact_candidate",),
                         irreversible_skills=("open lid",), **kw)
        o.reset([lid_program])
        return o

    def cand(t, satisfied=False, retry_safe=False, **extra):
        return {"satisfied": satisfied, "failed": False, "verified": False, "verifier_kind": "contact_candidate",
                "outcome_observable": False, "retry_safe": retry_safe, "source": "synthetic://contact",
                "observation_id": t, "key": "open lid(car)", **extra}
    adv = irreversible(candidate_outcome_policy="legacy_advance")
    for t in range(60):
        d = adv.tick(np.zeros(1024, np.float32), t, base_velocity=np.zeros(3, np.float32),
                     postcondition=cand(t, satisfied=t >= 10))
        if d.mode == "DONE":
            break
    ends = [e for e in adv.events if e["event"] == "contract_end" and e["key"] == "open lid(car)"]
    assert d.mode == "DONE" and len(ends) == 1 and ends[0]["success"] is None and ends[0]["verified"] is False, ends
    assert ends[0]["predicate"] == "unverified_outcome_candidate" and ends[0]["outcome_observable"] is False
    assert sum(e["event"] == "contract_start" and e["key"] == "open lid(car)" for e in adv.events) == 1
    assert not any(a.successes or a.failures or a.support_count for arms in adv.bank.arms.values() for a in arms)
    assert adv._irreversible_attempted == {"open lid(car)": "unverified_outcome_candidate"}
    # a replayed/recovered program cannot execute it again in this episode
    again = Contract(Step("open lid", ("car",)), 1, None, 20, arm_key="open lid(car)")
    assert not adv._start_contract(again, np.zeros(1024), 99) and adv.pending_hold_reason == "irreversible_skill_already_attempted"
    # (b) timed out with possible contact -> HOLD, no second attempt although max_attempts allows one
    unk = irreversible()
    for t in range(80):
        d = unk.tick(np.zeros(1024, np.float32), t, base_velocity=np.zeros(3, np.float32), postcondition=cand(t))
    assert d.mode == "HOLD" and unk.pending_hold_reason == "irreversible_outcome_unknown", (d, unk.pending_hold_reason)
    assert sum(e["event"] == "contract_start" and e["key"] == "open lid(car)" for e in unk.events) == 1
    # (c) the attempt provably had no contact (retry_safe) -> the ordinary bounded retry applies
    safe = irreversible()
    for t in range(120):
        d = safe.tick(np.zeros(1024, np.float32), t, base_velocity=np.zeros(3, np.float32), postcondition=cand(t, retry_safe=True))
    starts = [e for e in safe.events if e["event"] == "contract_start" and e["key"] == "open lid(car)"]
    assert d.mode == "HOLD" and [e["attempt"] for e in starts] == [1, 2] and safe.pending_hold_reason == "required_skill_failed"
    # (d) candidates cannot claim verification, observability, or a verified evidence kind
    for bad in (cand(3, verified=True), cand(3, outcome_observable=True), cand(3, key="close lid(car)"), cand(4)):
        assert adv._candidate_evidence(bad, key="open lid(car)", observation_id=3) is None
    assert adv._candidate_evidence(cand(3), key="open lid(car)", observation_id=3) is not None
    assert Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(2), precondition_monitor=None,
                        budgets=budgets)._candidate_evidence(cand(3), key="open lid(car)", observation_id=3) is None
    try:
        Orchestrator(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(2), precondition_monitor=None,
                     budgets=budgets, candidate_advance_kinds=("controller_postconditions",))
    except ValueError:
        pass
    else:
        raise AssertionError("a verified evidence kind was accepted as an unverified candidate kind")
    print("PASS (synthetic, CPU): navigation settling, bounded recovery and asking; VTT manipulation closure "
          "fails closed without independent postcondition evidence; verified evidence advances and may update memory; "
          "unobservable irreversible candidates advance unverified and are never re-executed blindly")


if __name__ == "__main__":
    _self_test()
