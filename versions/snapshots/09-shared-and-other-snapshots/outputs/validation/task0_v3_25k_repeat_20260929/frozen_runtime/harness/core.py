"""A decision kernel, not a robot controller or a WAM/VLM implementation.

Inputs must come from adapters that bind evidence to the current observation,
controller, models and calibration. See ``scripts/check_harness.py`` for use.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Mapping, Sequence


ACTION_DIM = 23


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _text(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class Evidence:
    executable: bool | None
    checked_steps: int
    risk_upper: float | None
    prediction_residual: float | None  # None when no trajectory/horizon-matched prediction exists.
    calibration_id: str | None
    source: str
    verifier_kind: str | None = None  # Runtime requires policy-visible provenance.


@dataclass(frozen=True)
class Candidate:
    interpretation: str
    actions: tuple[tuple[float, ...], ...]
    action_keys: tuple[str, ...]  # Grounded intended effect of each prefix step.
    observation_id: str
    observed_at: float  # Same monotonic clock as decide(now=...).
    evidence: Evidence


@dataclass(frozen=True)
class GateConfig:
    calibration_id: str
    max_risk: float
    max_residual: float
    max_age_s: float
    commit_steps: int = 1


@dataclass(frozen=True)
class Decision:
    mode: str
    reason: str
    actions: tuple[tuple[float, ...], ...] = ()
    choices: tuple[str, ...] = ()
    observation_id: str = ""
    expires_at: float | None = None


def decide(candidates: Sequence[Candidate], *, action_low: Sequence[float],
           action_high: Sequence[float], observation_id: str, now: float,
           config: GateConfig, ask_remaining: int) -> Decision:
    """Return EXECUTE / ASK / OBSERVE / REPLAN / HOLD without causing motion.

    Supply every still-plausible interpretation, once each. Unknown feasibility
    never removes an interpretation. Bounds are in the controller's explicit
    action units after denormalization; there is no implicit [-1, 1] assumption.
    Calibration must be fitted on allowed development data before evaluation.
    The runner owns the per-episode question counter and decrements it on ASK.
    ASK and HOLD contain no commands, never a zero vector (zero joint targets
    are not a safe hold). The controller must generate its own hold from state.
    """
    if (not isinstance(config, GateConfig) or not _text(config.calibration_id) or not _number(config.max_risk)
            or not 0 <= config.max_risk <= 1 or not _number(config.max_residual)
            or config.max_residual < 0 or not _number(config.max_age_s)
            or config.max_age_s <= 0 or type(config.commit_steps) is not int
            or not 1 <= config.commit_steps <= 4):
        raise ValueError("Invalid gate calibration/thresholds; commit_steps must be 1..4")
    if (not isinstance(action_low, (tuple, list)) or not isinstance(action_high, (tuple, list))
            or len(action_low) != ACTION_DIM or len(action_high) != ACTION_DIM
            or any(not _number(lo) or not _number(hi) or lo > hi
                   for lo, hi in zip(action_low, action_high))):
        raise ValueError("Provide finite, ordered controller bounds for all 23 actions")
    if not _number(now) or not _text(observation_id) or type(ask_remaining) is not int or ask_remaining < 0:
        raise ValueError("A finite monotonic time and current observation ID are required")
    if (not isinstance(candidates, (tuple, list))
            or any(not isinstance(c, Candidate) or not isinstance(c.evidence, Evidence) for c in candidates)):
        return Decision("HOLD", "invalid_candidates")
    if not candidates:
        return Decision("HOLD", "no_candidates")
    interpretations = [c.interpretation for c in candidates]
    if (any(not _text(value) for value in interpretations)
            or len(set(interpretations)) != len(interpretations)):
        return Decision("HOLD", "invalid_or_duplicate_interpretation")

    # Reject the whole set; dropping an invalid candidate could hide ambiguity.
    for candidate in candidates:
        if (not isinstance(candidate.actions, (tuple, list)) or not candidate.actions
                or not isinstance(candidate.action_keys, (tuple, list))
                or len(candidate.actions) != len(candidate.action_keys)
                or any(not _text(key) for key in candidate.action_keys)
                or any(not isinstance(action, (tuple, list)) or len(action) != ACTION_DIM or any(
                    not _number(value) or not lo <= value <= hi
                    for value, lo, hi in zip(action, action_low, action_high)
                ) for action in candidate.actions)):
            return Decision("HOLD", "invalid_action_chunk")
        if not _number(candidate.observed_at):
            return Decision("HOLD", "invalid_observation_time")
        if (candidate.observation_id != observation_id
                or not 0 <= now - candidate.observed_at <= config.max_age_s):
            return Decision("OBSERVE", "stale_or_mismatched_observation")
        ev = candidate.evidence
        if (ev.executable is not None and type(ev.executable) is not bool
                or type(ev.checked_steps) is not int
                or not 0 <= ev.checked_steps <= len(candidate.actions)
                or ev.risk_upper is not None and (
                    not _number(ev.risk_upper) or not 0 <= ev.risk_upper <= 1)
                or ev.prediction_residual is not None and (
                    not _number(ev.prediction_residual) or ev.prediction_residual < 0)):
            return Decision("HOLD", "invalid_safety_evidence")

    for candidate in candidates:
        ev = candidate.evidence
        if ev.executable is False:
            return Decision("REPLAN", "known_infeasible_interpretation")
        if (ev.executable is None or ev.checked_steps == 0 or not _text(ev.source)
                or ev.calibration_id != config.calibration_id
                or ev.risk_upper is None):
            return Decision("OBSERVE", "unknown_or_uncalibrated_physical_evidence")
        if ev.risk_upper > config.max_risk or (ev.prediction_residual is not None
                                               and ev.prediction_residual > config.max_residual):
            return Decision("REPLAN", "risk_or_prediction_residual_exceeded")

    steps = min(config.commit_steps, *(c.evidence.checked_steps for c in candidates))
    first = candidates[0]
    common = []
    for index in range(steps):
        if any(c.action_keys[index] != first.action_keys[index]
               or tuple(c.actions[index]) != tuple(first.actions[index]) for c in candidates):
            break
        common.append(tuple(first.actions[index]))
    if common:
        return Decision("EXECUTE", "verified_common_prefix", tuple(common),
                        observation_id=observation_id,
                        expires_at=min(c.observed_at + config.max_age_s for c in candidates))
    if len({c.action_keys[0] for c in candidates}) > 1:
        if ask_remaining == 0:
            return Decision("HOLD", "clarification_budget_exhausted")
        return Decision("ASK", "intent_changes_safe_next_action", choices=tuple(interpretations))
    # ponytail: exact equality is conservative; add a verified shared-action solver
    # when different trajectories with the same intended effect are common.
    return Decision("REPLAN", "equivalent_effect_needs_shared_verified_action")


def context_fingerprint(context: Mapping) -> str:
    """Hash a canonical task/scene context signature, not an embedding guess."""
    if not isinstance(context, Mapping):
        raise ValueError("Context must be a JSON mapping")
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class Scope:
    task_id: str
    skill_id: str
    context_fingerprint: str
    robot_version: str
    controller_version: str
    wam_version: str
    vlm_version: str
    split_id: str
    run_id: str

    def key(self):
        if any(not _text(value) for value in asdict(self).values()):
            raise ValueError("All memory scope fields are required")
        if (len(self.context_fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in self.context_fingerprint)):
            raise ValueError("Use context_fingerprint() for the context signature")
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def verified_outcome(predicates: Mapping[str, bool | None], *, terminal: bool) -> str | None:
    """Unknown predicates cannot certify success; only completed attempts persist.

    The caller supplies the complete goal/postcondition set from a trusted
        observation/controller verifier. ``terminal`` ends a skill attempt, not
    necessarily its environment episode. This validates structure, not truth.
    """
    if (type(terminal) is not bool or not isinstance(predicates, Mapping) or not predicates
            or any(not _text(key) or value is not None and type(value) is not bool
                   for key, value in predicates.items())):
        raise ValueError("Predicates must be a nonempty map of names to True/False/None")
    if terminal and all(value is True for value in predicates.values()):
        return "success"
    if terminal and any(value is False for value in predicates.values()):
        return "failure"
    return None


class Memory:
    """An evidence ledger pinned to one split/run; exact-scope retrieval only.

    Open frozen evaluations with readonly=True (SQLite mode=ro). Online runs
    need separate files. Cross-split promotion/skill synthesis is not provided.
    """

    def __init__(self, path: str | Path, *, split_id: str, run_id: str, readonly=False):
        if not _text(split_id) or not _text(run_id) or type(readonly) is not bool:
            raise ValueError("split_id, run_id and a boolean readonly are required")
        self.split_id, self.run_id, self.readonly = split_id, run_id, readonly
        path = Path(path).resolve()
        self.connection = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"),
                                          uri=True, timeout=1)
        self.connection.row_factory = sqlite3.Row
        identity = json.dumps([split_id, run_id])
        try:
            if not readonly:
                with self.connection:
                    self.connection.execute("CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
                    self.connection.execute("INSERT OR IGNORE INTO identity VALUES (1, ?)", (identity,))
                    if self.connection.execute("SELECT value FROM identity WHERE id=1").fetchone()[0] != identity:
                        raise ValueError("Memory database belongs to a different split/run")
                    self.connection.execute("""CREATE TABLE IF NOT EXISTS evidence (
                        id INTEGER PRIMARY KEY, scope TEXT NOT NULL, episode_id TEXT NOT NULL,
                        outcome TEXT NOT NULL CHECK(outcome IN ('success','failure','recovery')),
                        predicates TEXT NOT NULL, source TEXT NOT NULL, verifier_kind TEXT NOT NULL,
                        observation_id TEXT NOT NULL, recorded_at REAL NOT NULL,
                        recovery_of INTEGER REFERENCES evidence(id))""")
                    self.connection.execute("CREATE INDEX IF NOT EXISTS evidence_scope ON evidence(scope, id)")
            if self.connection.execute("SELECT value FROM identity WHERE id=1").fetchone()[0] != identity:
                raise ValueError("Memory database belongs to a different split/run")
        except Exception:
            self.connection.close()
            raise

    def _scope_key(self, scope):
        if scope.split_id != self.split_id or scope.run_id != self.run_id:
            raise ValueError("Cannot read/write a different split/run")
        return scope.key()

    def record(self, scope: Scope, *, predicates: Mapping[str, bool | None], terminal: bool,
               episode_id: str, source: str, verifier_kind: str, observation_id: str,
               recovery_of: int | None = None) -> int:
        """Store verified outcomes, including failures; guesses raise ValueError.

        source names the auditable evidence artifact. Recovery requires a prior
        failure in the same episode and exact scope, followed by verified success.
        """
        if self.readonly:
            raise PermissionError("Frozen memory is read-only")
        key = self._scope_key(scope)
        outcome = verified_outcome(predicates, terminal=terminal)
        if outcome is None:
            raise ValueError("Unverified outcomes and temporary observations are not memory")
        if (any(not _text(value) for value in (episode_id, source, observation_id))
                or verifier_kind not in ("perception_predicates", "simulator_predicates", "controller_postconditions")):
            raise ValueError("Auditable perception/controller verification provenance is required")
        if recovery_of is not None:
            if type(recovery_of) is not int or recovery_of <= 0:
                raise ValueError("recovery_of must be a positive evidence record ID")
            failure = self.connection.execute(
                "SELECT scope, episode_id, outcome FROM evidence WHERE id=?", (recovery_of,)).fetchone()
            if (outcome != "success" or failure is None
                    or tuple(failure) != (key, episode_id, "failure")):
                raise ValueError("Recovery must link to a verified failure in this episode/scope")
            outcome = "recovery"
        with self.connection:
            cursor = self.connection.execute("""INSERT INTO evidence
                (scope, episode_id, outcome, predicates, source, verifier_kind,
                 observation_id, recorded_at, recovery_of) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (key, episode_id, outcome, json.dumps(dict(predicates), sort_keys=True),
                 source, verifier_kind, observation_id, time.time(), recovery_of))
        return cursor.lastrowid

    def recall(self, scope: Scope, *, limit=20, policy_only=False) -> list[dict]:
        """Policy retrieval excludes simulator truth before applying the limit."""
        key = self._scope_key(scope)
        if type(limit) is not int or not 1 <= limit <= 100 or type(policy_only) is not bool:
            raise ValueError("limit must be 1..100 and policy_only must be boolean")
        visible = " AND verifier_kind IN ('perception_predicates','controller_postconditions')" if policy_only else ""
        rows = self.connection.execute(
            "SELECT * FROM evidence WHERE scope=?" + visible + " ORDER BY id DESC LIMIT ?", (key, limit)).fetchall()
        return [{**dict(row), "scope": json.loads(row["scope"]),
                 "predicates": json.loads(row["predicates"])} for row in rows]

    def close(self):
        self.connection.close()
