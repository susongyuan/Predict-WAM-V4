#!/usr/bin/env python3
"""Run without models/GPU: python3 scripts/check_harness.py."""

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.core import Candidate, Evidence, GateConfig, Memory, Scope, context_fingerprint, decide, verified_outcome
from harness.runtime import Observation, Outcome, run_episode


def raises(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


def check_runtime(directory, scope, config, action):
    class SyntheticAdapter:
        """Deterministic fake drivers, explicitly not WAM/VLM/robot behavior."""

        def __init__(self, mode="recover"):
            self.mode, self.now, self.observations = mode, 10.0, 0
            self.queue, self.commands, self.prompts, self.recalled = ["old-action"], [], [], []
            self.holds = []

        def hold(self, reason):
            self.holds.append(reason)

        def clear_actions(self):
            self.queue.clear()

        def observe(self, reason):
            self.observations += 1
            self.now += 0.01
            self.current_id = f"synthetic-{self.observations}"
            sensors = {"proprioception": (0.0,) * 61}
            if self.mode == "hidden_observation":
                sensors["bddl_truth"] = {"goal": True}
            if self.mode == "reused_observation":
                self.current_id = "reused"
            return Observation(self.current_id, self.now, sensors)

        def propose(self, observation, instruction, clarification, memories):
            assert self.queue == []
            self.recalled.append(memories)
            self.queue.extend(["new-but-uncommitted-action"] * 32)
            self.prompts.append((observation.observation_id, clarification))
            # Deliberately forged model safety evidence: runtime must replace it.
            forged = Evidence(True, 2, 0.0, 0.0, config.calibration_id, "untrusted-model")
            red = Candidate("red", (action, action), ("red-grasp", "carry"),
                            observation.observation_id, observation.observed_at, forged)
            blue = replace(red, interpretation="blue", actions=((0.2,) + action[1:], action),
                           action_keys=("blue-grasp", "carry"))
            return [blue] if clarification or self.mode not in ("recover", "timeout", "bad_answer") else [red, blue]

        def verify(self, observation, candidate):
            assert candidate.evidence.executable is None  # The model did not certify itself.
            if self.mode == "slow_verifier":
                self.now += 1.0
            return Evidence(None if self.mode == "unknown" else True, 2, 0.01, None,
                            config.calibration_id, "synthetic://physical-verifier",
                            "simulator_predicates" if self.mode == "hidden_evidence" else "controller_postconditions")

        def execute(self, command, *, observation_id, expires_at):
            assert self.queue == [] and len(command) == 23
            assert observation_id == self.current_id and self.now <= expires_at
            if self.mode == "reject":
                return False
            self.commands.append((observation_id, command))
            return True

        def ask(self, choices, *, deadline):
            assert self.queue == [] and not self.commands
            assert self.holds[-1] == "await_clarification"
            if self.mode == "timeout":
                self.now = deadline + 1
            return "not-a-choice" if self.mode == "bad_answer" else "blue"

        def outcome(self, before, command, after):
            success = self.mode != "recover" or len(self.commands) > 1
            predicates = {"goal": None if self.mode == "unverified" else success}
            return Outcome(predicates, self.mode != "nonterminal", "synthetic://postcondition",
                           "simulator_predicates" if self.mode == "hidden_outcome" else "perception_predicates",
                           after.observation_id)

    def run(adapter, memory, **options):
        return run_episode(adapter, scope=scope, episode_id="runtime-ep", instruction="拿杯子",
                           memory=memory, config=replace(config, commit_steps=1),
                           action_low=(-2.0,) * 23, action_high=(2.0,) * 23,
                           clock=lambda: adapter.now, **options)

    path = Path(directory) / "runtime.sqlite"
    memory = Memory(path, split_id=scope.split_id, run_id=scope.run_id)
    memory.record(scope, predicates={"hidden_goal": True}, terminal=True, episode_id="debug-only",
                  source="synthetic://hidden-truth", verifier_kind="simulator_predicates", observation_id="debug-obs")
    adapter = SyntheticAdapter()
    result = run(adapter, memory)
    assert (result.status, result.steps, result.questions) == ("SUCCESS", 2, 1)
    assert adapter.prompts[0][1] is None and adapter.prompts[1][1] == "blue"
    assert len({item[0] for item in adapter.prompts}) == len(adapter.prompts)
    assert [row["outcome"] for row in adapter.recalled[-1]] == ["failure"]
    assert all(row["verifier_kind"] != "simulator_predicates" for rows in adapter.recalled for row in rows)
    rows = memory.recall(scope, policy_only=True)
    assert [row["outcome"] for row in rows] == ["recovery", "failure"]
    assert memory.recall(scope, policy_only=True, limit=1)[0]["outcome"] == "recovery"
    assert len(memory.recall(scope)) == 3 and adapter.queue == [] and adapter.holds[-1] == "episode_end"

    for mode, expected in (("unknown", "retry_budget_exhausted"), ("slow_verifier", "retry_budget_exhausted"),
                           ("reject", "retry_budget_exhausted"), ("timeout", "clarification_timeout"),
                           ("bad_answer", "clarification_not_a_current_choice")):
        adapter = SyntheticAdapter(mode)
        result = run(adapter, memory)
        assert result.status == "HOLD" and result.reason == expected, (mode, result)
        assert result.steps == 0 and adapter.commands == [] and adapter.queue == []
        assert result.questions == (1 if mode in ("timeout", "bad_answer") else 0)
        assert adapter.observations <= 3 and adapter.holds[-1] == "episode_end"
    for mode in ("hidden_observation", "hidden_evidence", "hidden_outcome", "reused_observation"):
        adapter = SyntheticAdapter(mode)
        raises(ValueError, run, adapter, memory)
        assert adapter.queue == [] and adapter.holds[-1] == "episode_end"
    adapter = SyntheticAdapter("unverified")
    assert run(adapter, memory).reason == "terminal_outcome_unverified"
    adapter = SyntheticAdapter("nonterminal")
    assert run(adapter, memory, max_steps=2).reason == "step_budget_exhausted"
    adapter = SyntheticAdapter()
    assert run(adapter, memory, ask_budget=0).reason == "clarification_budget_exhausted"
    adapter = SyntheticAdapter()
    adapter.verify = None
    raises(TypeError, run, adapter, memory)
    count = len(memory.recall(scope))
    memory.close()
    frozen = Memory(path, split_id=scope.split_id, run_id=scope.run_id, readonly=True)
    adapter = SyntheticAdapter("success")
    result = run(adapter, frozen)
    assert result.status == "SUCCESS" and result.events[-1]["memory_record_id"] is None
    assert len(frozen.recall(scope)) == count
    frozen.close()


def main():
    # Synthetic evidence tests the kernel, not WAM quality or simulator success.
    action = (0.0,) * 23
    moved = (0.2,) + action[1:]
    evidence = Evidence(True, 2, 0.01, 0.02, "dev-calibration-v1", "synthetic-check")
    candidate = Candidate("拿红杯", (action, moved), ("approach", "red-grasp"), "obs-1", 10.0, evidence)
    config = GateConfig("dev-calibration-v1", 0.05, 0.1, 0.5, commit_steps=2)
    arguments = dict(action_low=(-2.0,) * 23, action_high=(2.0,) * 23,
                     observation_id="obs-1", now=10.1, config=config, ask_remaining=1)
    gate = lambda items: decide(items, **arguments)
    other = replace(candidate, interpretation="拿蓝杯", action_keys=("approach", "blue-grasp"))
    decision = gate([candidate, other])
    assert decision.mode == "EXECUTE" and decision.actions == (action,)
    assert decision.expires_at == 10.5 and decision.observation_id == "obs-1"
    assert gate([candidate]).actions == candidate.actions
    one_step = dict(arguments, config=replace(config, commit_steps=1))
    assert len(decide([candidate], **one_step).actions) == 1
    divergent = replace(other, actions=(moved, moved), action_keys=("blue-approach", "blue-grasp"))
    assert gate([candidate, divergent]).mode == "ASK"
    no_questions = decide([candidate, divergent], **dict(arguments, ask_remaining=0))
    assert no_questions.mode == "HOLD" and no_questions.actions == ()
    equivalent = replace(divergent, action_keys=candidate.action_keys)
    assert gate([candidate, equivalent]).mode == "REPLAN"  # Never ask about sampling noise.
    assert gate([candidate, replace(other, evidence=replace(evidence, executable=None))]).mode == "OBSERVE"
    assert gate([candidate, replace(other, evidence=replace(evidence, executable=False))]).mode == "REPLAN"
    # A new episode has no matched future observation. It must not deadlock,
    # while missing physical risk evidence must still block motion.
    assert gate([replace(candidate, evidence=replace(evidence, prediction_residual=None))]).mode == "EXECUTE"
    for ev in (replace(evidence, calibration_id=None), replace(evidence, risk_upper=None),
               replace(evidence, checked_steps=0),
               replace(evidence, source=""), replace(evidence, calibration_id="wrong-version")):
        assert gate([replace(candidate, evidence=ev)]).mode == "OBSERVE"
    for ev in (replace(evidence, risk_upper=0.2), replace(evidence, prediction_residual=0.5)):
        assert gate([replace(candidate, evidence=ev)]).mode == "REPLAN"
    for ev in (replace(evidence, risk_upper=float("nan")), replace(evidence, executable=1),
               replace(evidence, checked_steps=3), replace(evidence, prediction_residual=float("inf"))):
        assert gate([replace(candidate, evidence=ev)]).mode == "HOLD"
    for bad_action in (None, (0.0,) * 22, (float("nan"),) + action[1:],
                       (float("inf"),) + action[1:], (3.0,) + action[1:],
                       (10 ** 1000,) + action[1:], (True,) + action[1:]):
        assert gate([replace(candidate, actions=(bad_action, moved))]).mode == "HOLD"
    assert gate([replace(candidate, actions=((1.5,) * 23, moved))]).mode == "EXECUTE"
    for timestamp in (8.0, 11.0):
        assert gate([replace(candidate, observed_at=timestamp)]).mode == "OBSERVE"
    assert gate([replace(candidate, observation_id="old")]).mode == "OBSERVE"
    assert gate([replace(candidate, observed_at=float("nan"))]).mode == "HOLD"
    assert gate([]).mode == "HOLD" and gate([candidate, candidate]).mode == "HOLD"
    assert gate([{}]).mode == "HOLD" and gate(None).mode == "HOLD"
    assert gate([replace(candidate, actions=None)]).mode == "HOLD"
    raises(ValueError, decide, [candidate], **dict(arguments, action_low=(0.0,) * 22))
    raises(ValueError, decide, [candidate], **dict(arguments, action_low=(3.0,) * 23))
    raises(ValueError, decide, [candidate], **dict(arguments, config=replace(config, commit_steps=5)))

    assert verified_outcome({"goal": None}, terminal=True) is None
    assert verified_outcome({"goal": True}, terminal=False) is None
    assert verified_outcome({"goal": False}, terminal=False) is None
    assert verified_outcome({"goal": False}, terminal=True) == "failure"
    assert verified_outcome({"goal": True, "grasp": None}, terminal=True) is None
    raises(ValueError, verified_outcome, {"goal": 1}, terminal=True)
    scope = Scope("task-013", "carry", context_fingerprint({"scene": "house-a", "target": "bag"}),
                  "r1pro-v1", "controller-v1", "wam-weights-sha", "vlm-weights-sha", "dev", "run-001")
    provenance = dict(episode_id="ep-1", source="synthetic://check/step-3",
                      verifier_kind="simulator_predicates", observation_id="obs-3")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "run-001.sqlite"
        memory = Memory(path, split_id="dev", run_id="run-001")
        raises(ValueError, memory.record, scope, predicates={"goal": None}, terminal=True, **provenance)
        raises(ValueError, memory.record, scope, predicates={"goal": True}, terminal=False, **provenance)
        raises(ValueError, memory.record, scope, predicates={"goal": False}, terminal=False, **provenance)
        raises(ValueError, memory.record, scope, predicates={"goal": True}, terminal=True,
               **dict(provenance, verifier_kind="vlm_guess"))
        failure_id = memory.record(scope, predicates={"goal": False}, terminal=True, **provenance)
        raises(ValueError, memory.record, scope, predicates={"goal": True}, terminal=True,
               recovery_of=failure_id, **dict(provenance, episode_id="other-episode"))
        recovery_id = memory.record(scope, predicates={"goal": True}, terminal=True,
                                    recovery_of=failure_id, **dict(provenance, observation_id="obs-4"))
        rows = memory.recall(scope)
        assert [row["outcome"] for row in rows] == ["recovery", "failure"]
        assert rows[0]["id"] == recovery_id and rows[0]["recovery_of"] == failure_id
        assert rows[1]["predicates"] == {"goal": False}
        for field in ("task_id", "skill_id", "robot_version", "controller_version", "wam_version", "vlm_version"):
            assert memory.recall(replace(scope, **{field: "different"})) == []
        assert memory.recall(replace(scope, context_fingerprint=context_fingerprint({"scene": "other"}))) == []
        raises(ValueError, memory.recall, replace(scope, split_id="official"))
        raises(ValueError, memory.recall, replace(scope, run_id="run-002"))
        raises(ValueError, Memory, path, split_id="official", run_id="run-001")
        raises(ValueError, Memory, path, split_id="dev", run_id="run-002")
        memory.close()
        frozen = Memory(path, split_id="dev", run_id="run-001", readonly=True)
        assert len(frozen.recall(scope)) == 2
        raises(PermissionError, frozen.record, scope, predicates={"goal": True}, terminal=True, **provenance)
        raises(ValueError, Memory, path, split_id="official", run_id="run-001", readonly=True)
        frozen.close()
        isolated = Memory(Path(directory) / "run-002.sqlite", split_id="dev", run_id="run-002")
        assert isolated.recall(replace(scope, run_id="run-002")) == []
        isolated.close()
        check_runtime(directory, scope, config, action)
    print("PASS (synthetic CPU only): gates, one-step runtime, fresh observations/queues, clarification budgets, "
          "bounded recovery, visible-only evidence, frozen and isolated memory")


if __name__ == "__main__":
    main()
