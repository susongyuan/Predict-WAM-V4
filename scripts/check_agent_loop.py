#!/usr/bin/env python3
"""Synthetic check: grounding -> shared prefix -> runtime ASK -> async answer -> VTT closure -> reviewed memory.
CPU only, mock VLM, synthetic features and postcondition labels; not a robot result or a calibrated safety test.

It exercises exactly the code paths the closed-loop runtime will use (harness.grounding, harness.core.decide,
harness.vtt_monitor, harness.memory_vtt) so their contracts are checked together, not one by one.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.core import Candidate, Evidence, GateConfig, decide  # noqa: E402
from harness.grounding import CandidateProgram, Catalog, MockBackend, Step, compile_programs, divergence, ground  # noqa: E402
from harness.memory_vtt import export, promote, validate_context  # noqa: E402
from harness.orchestrator import ClosureAdapter, ConditionBank, Orchestrator  # noqa: E402
from harness.vtt_monitor import ClosureConfig, VTTClosureMonitor  # noqa: E402


def check_program_contracts(cat, visible):
    # Replay the actual saved Qwen response, without loading a VLM or changing its historical FAIL result.
    source = ROOT / "outputs/grounding/qwen3b_promptfix_single.json"
    recorded = json.loads(source.read_text())
    replay, rejected, _ = ground(MockBackend({recorded["instruction"]: recorded["raw_text"]}),
                                 recorded["instruction"], recorded["visible_objects"], cat)
    assert not rejected and [s.key() for s in replay[0].steps] == recorded["expected_program"]

    def compile_steps(steps, **kwargs):
        candidates, errors = compile_programs([CandidateProgram("contract check", steps,
                                               ["robot is already at the car"])], cat, visible, **kwargs)
        assert not errors, errors
        return candidates[0].steps

    nav, lid, close = Step("move to", ("car",)), Step("open lid", ("car",)), Step("close lid", ("car",))
    assert compile_steps([lid, close]) == [nav, lid, close]  # prose cannot certify proximity; no duplicate approach
    from harness.orchestrator import fixed_program_candidates
    fixed, _ = fixed_program_candidates()
    assert compile_steps(fixed[0].steps) == fixed[0].steps  # toy box in car: no spurious car -> box -> car detour
    assert compile_steps(compile_steps([lid, close])) == [nav, lid, close]
    box, pickup = Step("move to", ("toy_box",)), Step("pick up from", ("toy_box", "floors"))
    putbox = Step("place in", ("toy_box", "car"))
    assert compile_steps([nav, lid, box, pickup, putbox]) == [nav, lid, box, pickup, nav, putbox]
    putcam = Step("place in", ("digital", "toy_box"))
    assert compile_steps([nav, lid, putcam, close]) == [nav, lid, box, putcam, nav, close]
    assert compile_steps([nav, lid, close], completed_prefix=[nav, lid]) == [nav, lid, close]
    assert compile_steps([nav, lid, close], completed_prefix=[nav, lid], restart=True) == [nav, lid, nav, close]
    assert compile_programs([CandidateProgram("changed past", [box, pickup])], cat, visible,
                            completed_prefix=[nav])[1] == ["changed_completed_prefix"]
    assert compile_programs([CandidateProgram("skip failed step", [nav])], cat, visible,
                            completed_prefix=[nav], restart=True)[1] == ["empty_replan_suffix"]
    assert compile_programs([CandidateProgram("navigate to carried box", [box, pickup, box])], cat, visible)[1]
    assert compile_programs([CandidateProgram("ambiguous binding", [lid])], cat, visible + ["car"])[1] \
        == ["ambiguous_object_binding:car"]
    invalid = CandidateProgram("unsupported destination", [Step("place in", ("digital", "car"))])
    assert compile_programs([invalid], cat, visible)[1] == ["untrained_skill:place in(digital,car)"]
    missing_nav = Catalog()
    missing_nav.patterns["move to"] = missing_nav.patterns["move to"] - {("car",)}
    assert compile_programs([CandidateProgram("open", [lid])], missing_nav, visible)[1] == ["untrained_navigation:move to(car)"]
    missing_nav.patterns["invented"] = {("car",)}
    assert compile_programs([CandidateProgram("unknown contract", [Step("invented", ("car",))])], missing_nav, visible)[1] \
        == ["undefined_skill_contract:invented(car)"]
    duplicates, errors = compile_programs([CandidateProgram("implicit approach", [lid]),
                                           CandidateProgram("explicit approach", [nav, lid])], cat, visible)
    assert not errors and len(duplicates) == 1

    # Reset, asynchronous answers and replanners all pass through the same contract boundary.
    from harness.orchestrator import _FakeClosure, _FakePrecond
    prototypes = {name: np.ones(1024, np.float32) for name in cat.patterns}
    def runtime(replanner=None):
        return Orchestrator(catalog=cat, bank=ConditionBank(prototypes), closure_monitor=_FakeClosure(50),
                            precondition_monitor=_FakePrecond(1), budgets={"move to": 50, "open lid": 1},
                            max_attempts=1, budget_factor=1, replanner=replanner,
                            require_manipulation_postcondition=False)
    blocked = runtime()
    blocked.reset([CandidateProgram("valid", [lid]), invalid])
    assert blocked.tick(np.zeros(1024), 0).mode == "HOLD" and blocked.contract is None
    ambiguous = runtime()
    ambiguous.reset([CandidateProgram("open", [lid]), CandidateProgram("close", [close])])
    question = ambiguous.tick(np.zeros(1024), 0)
    assert question.mode == "ASK" and ambiguous.contract is None  # cannot use the first candidate's arrival predicate
    ambiguous.candidates[1].steps = invalid.steps
    assert not ambiguous.answer(1, question_id=question.question_id)
    assert ambiguous.tick(np.zeros(1024), 1).mode == "HOLD"
    for bad_replan in (False, True):
        replacement = CandidateProgram("changed past" if bad_replan else "retry full program",
                                       [box, pickup] if bad_replan else [nav, lid])
        orch = runtime(lambda key, candidates: [replacement])
        orch.reset([CandidateProgram("open", [lid])])
        for t in range(61):
            directive = orch.tick(np.zeros(1024), t, base_velocity=np.zeros(3, np.float32))
        assert orch.completed_steps == [nav] and orch.index == 1
        if bad_replan:
            assert directive.mode == "HOLD" and orch.program_rejections == ["changed_completed_prefix"]
        else:
            assert directive.mode == "EXECUTE" and orch.contract.step == nav
            assert orch.candidates[0].steps == [nav, nav, lid]

    # Same navigation object, different following manipulation: separate defaults and posterior statistics.
    def target(value, duration):
        return {"train_count": 5, "vector": np.full(1024, value).tolist(), "median_frames": duration}
    targets = {"format": "train-contract-vtt-v1", "source_split": "train", "dimension": 1024,
               "uses_validation_vectors": False, "uses_validation_scalars": False,
               "contracts": {lid.key(): target(0.1, 30), putbox.key(): target(0.2, 40)},
               "navigation": {nav.key(): {lid.key(): target(1.0, 51), putbox.key(): target(2.0, 71)}}}
    bank = ConditionBank(prototypes, train_targets=targets)
    opening = bank.select(nav.key(), nav.skill, next_key=lid.key())
    placing = bank.select(nav.key(), nav.skill, next_key=putbox.key())
    assert opening is not placing and not np.array_equal(opening.vector, placing.vector)
    bank.update(bank.storage_key(nav.key(), nav.skill, lid.key()), opening.arm_id, False)
    assert opening.failures == 1 and placing.failures == 0 and placing.source == "train-contract-vtt-v1"
    clone = ConditionBank(prototypes, train_targets=targets)
    clone.load(bank.export())
    assert clone.export() == bank.export()
    targeted = runtime()
    targeted.bank = bank
    targeted.reset([CandidateProgram("place box", [putbox])])
    directive = targeted.tick(np.zeros(1024), 0)
    assert directive.new_chunk_required and targeted.contract.budget_steps == 71
    assert targeted.contract.arm_key == "move to(car) -> place in(toy_box,car)"
    targeted.reset([CandidateProgram("close", [close])])
    assert targeted.tick(np.zeros(1024), 0).mode == "HOLD"  # missing exact phase never falls back to an unrelated average

    # The real policy adapter discards actions queued under a previous target on a new contract.
    from harness.policy_server import WamPolicy
    targeted.reset([CandidateProgram("place box", [putbox])])
    fake = WamPolicy.__new__(WamPolicy)
    fake.orch, fake._grounded, fake.step_in_episode, fake.episode = targeted, True, 1, 1
    fake.base_control_mode = "wam_whole_body"
    fake.queue, fake.exec_horizon, fake.chunk_id = [np.full(23, 99.0)], 2, 0
    fake.stats = {"steps": 0, "holds": 0, "chunks": 0, "infer_ms": []}
    sampled = []
    def sample(feats, state, condition):
        sampled.append(condition.numpy().copy())
        return np.full((32, 23), 0.2, np.float32)
    fake._sample, fake._bounded_chunk, fake._emit = sample, lambda chunk: chunk, lambda row: None
    fake._find = lambda obs, *keys: np.zeros((2, 2, 3), np.uint8)
    with tempfile.TemporaryDirectory() as directory:
        fake.log_dir = Path(directory)
        action = fake._act_orchestrated({}, np.zeros(61, np.float32), None, np.zeros(1024), time.monotonic())
    assert np.allclose(action.numpy(), 0.2) and len(sampled) == 1 and np.array_equal(sampled[0][0], placing.vector)
    assert len(fake.queue) == 1 and np.allclose(fake.queue[0], 0.2)
    probe = np.arange(32 * 23, dtype=np.float32).reshape(32, 23)
    original = probe.copy()
    fake.base_control_mode = "hold_during_manipulation"
    locked = fake._apply_base_control(probe, "open lid")
    assert np.array_equal(locked[:, :3], np.zeros((32, 3), np.float32))
    assert np.array_equal(locked[:, 3:], probe[:, 3:]) and np.array_equal(probe, original)
    assert fake._apply_base_control(probe, "move to") is probe
    # An optional open-lid diagnostic must not silently alter subsequent grasp/place/door skills.
    fake.open_lid_arm_gain = 2.0
    fake.action_low, fake.action_high = np.full(23, -1e6), np.full(23, 1e6)
    for other_skill in ("move to", "pick up from", "place in", "open door", "close lid"):
        assert fake._apply_arm_gain(probe, np.zeros(61), other_skill) is probe
    changed = fake._apply_arm_gain(probe, np.zeros(61), "open lid")
    np.testing.assert_allclose(changed[:, 7:14], 2 * probe[:, 7:14])
    np.testing.assert_array_equal(changed[:, :3], probe[:, :3])

    return {"source": str(source.relative_to(ROOT)), "instruction": recorded["instruction"],
            "historical_model_program": recorded["candidates"][0]["program"],
            "compiled_program": [s.key() for s in replay[0].steps], "rejections": rejected, "pass": True,
            "scope": "Deterministic replay of recorded raw VLM output plus synthetic contract/runtime checks; no new VLM inference or robot execution."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-report", type=Path, help="optional deterministic replay evidence JSON")
    options = parser.parse_args()
    cat = Catalog()
    visible = ["car", "door", "toy_box", "floors", "digital", "tennis", "coffee_table"]
    replay_report = check_program_contracts(cat, visible)
    common = [{"skill": "move to", "objects": ["car"]}, {"skill": "open lid", "objects": ["car"]}]
    cam = [{"skill": "move to", "objects": ["digital"]}, {"skill": "pick up from", "objects": ["digital", "coffee_table"]}]
    rac = [{"skill": "move to", "objects": ["tennis"]}, {"skill": "pick up from", "objects": ["tennis", "coffee_table"]}]
    tail = [{"skill": "move to", "objects": ["car"]}, {"skill": "place in", "objects": ["tennis", "car"]}, {"skill": "place in", "objects": ["digital", "toy_box"]}, {"skill": "close lid", "objects": ["car"]}]
    vlm = MockBackend({"pack the small things": json.dumps({"candidates": [
        {"interpretation": "camera first", "program": common + cam + rac + tail},
        {"interpretation": "racket first", "program": common + rac + cam + tail}]})})

    # 1. grounding: two interpretations that share a 2-step prefix
    cands, rejected, _ = ground(vlm, "pack the small things into the car", visible, cat)
    assert len(cands) == 2 and rejected == []
    prefix, choices = divergence(cands, 0)
    assert prefix == 2

    # 2. gate: while the next skill is shared, no question is asked even though two interpretations exist
    bounds = json.loads((ROOT / "configs/r1pro_action_bounds.json").read_text()) if (ROOT / "configs/r1pro_action_bounds.json").exists() else None
    low = tuple(bounds["action_low"]) if bounds else (-3.0,) * 23
    high = tuple(bounds["action_high"]) if bounds else (3.0,) * 23
    mid = tuple((l + h) / 2 for l, h in zip(low, high))
    ev = Evidence(True, 1, 0.01, None, "dev-cal-v0", "mock-verifier", "controller_postconditions")
    cfg = GateConfig("dev-cal-v0", 0.05, 0.1, 0.5, commit_steps=1)
    step = 0
    keys = [c.next_key(step) for c in cands]
    cs = [Candidate(c.interpretation, (mid,), (k,), "obs-1", 1.0, ev) for c, k in zip(cands, keys)]
    d = decide(cs, action_low=low, action_high=high, observation_id="obs-1", now=1.1, config=cfg, ask_remaining=2)
    assert d.mode == "EXECUTE", d
    # 3. at the divergence step, the same gate asks; with no budget it holds
    step = prefix
    keys = [c.next_key(step) for c in cands]
    cs = [Candidate(c.interpretation, (mid,), (k,), "obs-2", 2.0, ev) for c, k in zip(cands, keys)]
    d = decide(cs, action_low=low, action_high=high, observation_id="obs-2", now=2.1, config=cfg, ask_remaining=2)
    assert d.mode == "ASK" and set(d.choices) == {"camera first", "racket first"}, d
    assert decide(cs, action_low=low, action_high=high, observation_id="obs-2", now=2.1, config=cfg, ask_remaining=0).mode == "HOLD"
    # 4. answer -> single program -> execute
    chosen = [c for c in cands if c.interpretation == "racket first"]
    d = decide([Candidate(chosen[0].interpretation, (mid,), (chosen[0].next_key(step),), "obs-3", 3.0, ev)],
               action_low=low, action_high=high, observation_id="obs-3", now=3.1, config=cfg, ask_remaining=1)
    assert d.mode == "EXECUTE"

    # 5. VTT closure on a synthetic trajectory that moves along the 'open lid' prototype direction
    mon = VTTClosureMonitor()
    lid = mon.classes["open lid"]
    rng = np.random.default_rng(0)
    start = rng.normal(size=1024)
    mon.start(lid, start)
    cfgc = ClosureConfig(done_progress=0.7, done_alignment=0.6, stall_steps=50, min_steps=5, max_steps=400)
    verdicts = []
    for t in range(1, 60):
        cls = start + mon.proto[lid] * (t / 40) + rng.normal(scale=0.02, size=1024)
        verdicts.append(mon.update(cls, cfgc)[0])
    assert True in verdicts and verdicts.index(True) >= 5, verdicts[:10]
    # stall: no movement -> False after stall_steps
    mon.start(lid, start)
    v = None
    for t in range(80):
        v = mon.update(start + rng.normal(scale=0.001, size=1024), cfgc)[0]
    assert v is False
    # 6. Exercise the actual state machine (the direct gate test above alone does not test asking/resuming).
    class PerceptionStub:
        def arrived(self, cls_now, key):
            return True, 1.0, 1.0

    prototypes = {name: mon.proto[cid].astype(np.float32) for name, cid in mon.classes.items()}
    configs = {cid: ClosureConfig(0.7, 0.6, stall_steps=50, min_steps=5) for cid in mon.classes.values()}
    bank = ConditionBank(prototypes)
    orch = Orchestrator(catalog=cat, bank=bank, closure_monitor=ClosureAdapter(mon, configs),
                        precondition_monitor=PerceptionStub(), budgets={name: 200 for name in mon.classes},
                        ask_budget=1, budget_factor=1, require_manipulation_postcondition=False)
    orch.reset(cands)
    cls = np.zeros(1024, np.float32)
    for t in range(1000):
        if orch.contract is not None:
            cls = orch.closure.state.start_cls + orch.contract.arm.vector * ((t - orch.contract.started_step) / 40)
        directive = orch.tick(cls, t, base_velocity=np.zeros(3, np.float32))
        if directive.mode == "ASK":
            break
    assert directive.mode == "ASK" and orch.index == prefix and orch.contract is None
    assert all(e["index"] < prefix for e in orch.events if e["event"] == "contract_start")
    for wait in range(1, 4):
        assert orch.tick(cls, t + wait, base_velocity=np.zeros(3, np.float32)).mode == "HOLD"
    assert orch.asks_left == 0 and len([e for e in orch.events if e["event"] == "ask"]) == 1
    assert not orch.answer(1, question_id=directive.question_id + 1)
    assert not orch.answer(-1, question_id=directive.question_id)
    assert orch.answer(1, question_id=directive.question_id)
    for current in range(t + 4, t + 1000):
        if orch.contract is not None:
            cls = orch.closure.state.start_cls + orch.contract.arm.vector * ((current - orch.contract.started_step) / 40)
        directive = orch.tick(cls, current, base_velocity=np.zeros(3, np.float32))
        if directive.mode == "DONE":
            break
    assert directive.mode == "DONE" and orch.candidates[0].interpretation == "racket first"
    assert all(len(arms) == 1 and arms[0].successes == 0 for arms in bank.arms.values())

    # 7. Heuristic closure cannot promote memory. Independently supplied synthetic reviews can; the
    # resulting export is keyed by object bindings and explicitly scoped, then reloads in a frozen bank.
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        rows = [{"episode": 1, **r} for r in orch.realized]
        (run / "realized_vtt.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        args = argparse.Namespace(run_dir=tmp, db=str(run / "ledger.sqlite"), split="development", run="synthetic",
                                  verifier="perception_predicates", scene_context='{"scene":"synthetic"}',
                                  versions=None, wam_version="synthetic-wam", task="task-013")
        assert promote(args)["promoted"] == 0
        reviewed = {f"{r['index']}@{r['end_step']}": "done" for r in rows if r["skill"] != "move to"}
        (run / "closure_labels.json").write_text(json.dumps({"1": reviewed}))
        assert promote(args)["promoted"] == len(reviewed)
        args.stores, args.out = [str(run / "memory_vtt_store.jsonl")], str(run / "conditions.json")
        data = export(args)
        assert "pick up from(tennis,coffee_table)" in data["conditions"]
        assert "pick up from(digital,coffee_table)" in data["conditions"]
        validate_context(data["context"], data["context"])
        frozen = ConditionBank(prototypes, frozen=True)
        frozen.load(data["arms"])
        assert all({a.arm_id for a in arms} == {"prototype", "memory0"} for arms in frozen.arms.values())
        before = frozen.export()
        frozen.update("open lid(car)", "memory0", True)
        assert frozen.export() == before
        assert frozen.select("open lid(car)", "open lid").arm_id == "prototype"  # exported mean is an untried arm

    # 8. The real server's control-message handler must bind replies to the current episode AND question.
    from harness.policy_server import WamPolicy, control_message
    orch.reset([CandidateProgram(c.interpretation, c.steps[prefix:]) for c in cands])
    pending = orch.tick(cls, 0)
    assert pending.mode == "ASK"
    responses = []
    fake = SimpleNamespace(orch=orch, episode=2, step_in_episode=0, queue=["stale-action"], _emit=responses.append)
    status = control_message(fake, {"status": True})
    assert status["episode"] == 2 and status["question_id"] == pending.question_id and len(status["choices"]) == 2
    reply = {"episode": 2, "question_id": pending.question_id, "answer": 1}
    assert control_message(fake, {"clarification": {**reply, "source": "oracle"}})["accepted"] is False
    assert control_message(fake, {"clarification": {**reply, "episode": 1}})["accepted"] is False
    assert control_message(fake, {"clarification": {**reply, "question_id": pending.question_id - 1}})["accepted"] is False
    assert fake.queue == ["stale-action"] and orch.status == "awaiting_answer"
    assert control_message(fake, {"clarification": reply})["accepted"] is True and fake.queue == []
    assert control_message(fake, {"clarification": reply})["accepted"] is False
    assert control_message(fake, {"status": True})["question_id"] is None
    assert any(r["accepted"] and r["source"] == "human" for r in responses)
    # The final partial skill must persist exactly once and never become a successful memory.
    final = WamPolicy.__new__(WamPolicy)
    final.program = ["synthetic"]
    final.monitor = SimpleNamespace(state=SimpleNamespace(steps=7))
    final._last_cls = cls
    final._orch_finish_episode = lambda: None
    recorded = []
    final._record_realized = lambda vector, verdict: recorded.append(verdict)
    final.finish_episode()
    final.finish_episode()
    assert recorded == [None] and final.monitor.state is None
    from scripts.probe_vlm_grounding import judge
    assert judge("c", [], ["untrained_skill:place in(digital,car)"], [])[0]
    substituted = CandidateProgram("changed destination", [Step("place in", ("digital", "toy_box"))])
    assert not judge("c", [substituted], [], [])[0]
    if options.replay_report:
        options.replay_report.parent.mkdir(parents=True, exist_ok=True)
        options.replay_report.write_text(json.dumps(replay_report, indent=2) + "\n")
    print("PASS (synthetic, CPU only): core evidence gate; real orchestrator shared-prefix EXECUTE -> ASK once -> "
          "pending HOLD -> stale reply rejected -> answer resumes -> closure; reviewed SQLite memory -> scoped frozen bank; "
          "server clarification rejects stale/replayed IDs and clears queued actions; saved Qwen replay and navigation contracts")


if __name__ == "__main__":
    main()
