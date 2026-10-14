#!/usr/bin/env python3
"""CPU/stdlib-only regression: no model, simulator, GPU, systemd or historical-run writes.

    python3 -B scripts/check_evaluation_evidence.py
"""
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import behavior_closed_loop as launch
import summarize_closed_loop as summary

ROOT = Path(__file__).resolve().parents[1]
OG = ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson"


def definitions(path, names, namespace):
    """Execute only selected pure definitions; never import the simulator's package initializers."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Invalid evidence was accepted")


def summarize(run):
    with redirect_stdout(io.StringIO()):
        return summary.main(run, write=False)


def main():
    constants = {}
    definitions(OG / "eval/utils/eval_utils.py", {"DEFAULT_EVAL_SEED", "NUM_TEST_INSTANCES", "NUM_PUBLIC_TEST_INSTANCES",
                                                "NUM_HIDDEN_TEST_INSTANCES", "TEST_INSTANCE_IDS"}, constants)
    definitions(OG / "eval/evaluator.py", {"EVAL_MODES", "resolve_instance_ids"}, constants)
    resolve = constants["resolve_instance_ids"]
    task, indices = launch.CONFIG["task"]["name"], [10, 11]
    instance_ids = resolve(task, indices, "public_test")  # No hard-coded simulator instance ID.

    # Exercise the actual logging/status methods with tiny Python objects and a fake ordered websocket.
    evaluator_tree = ast.parse((OG / "eval/evaluator.py").read_text())
    evaluator_class = next(n for n in evaluator_tree.body if isinstance(n, ast.ClassDef) and n.name == "Evaluator")
    method_names = {"start_joint_log", "_write_joint_log", "stop_joint_log", "policy_evidence"}
    method_ns = {"os": os, "json": json, "is_system_bddl_inst": lambda _: False}
    exec(compile(ast.Module(body=[n for n in evaluator_class.body if isinstance(n, ast.FunctionDef)
                                 and n.name in method_names], type_ignores=[]), "evaluator_methods", "exec"), method_ns)
    status_messages, seed_calls = [], []

    class FakeEvaluator:
        def __init__(self, cfg):
            self.cfg, self.episode, self.steps, self.instance = cfg, 0, 0, None
            self._joint_log, self._joint_log_entities, self._joint_log_step = None, {}, 0
            entity = SimpleNamespace(n_joints=1, name="car", joints={"j_trunk": None}, joint_lower_limits=[0.0],
                                     joint_upper_limits=[2.53], get_joint_positions=lambda: [instance_ids.index(self.instance) / 10 + 0.1])
            self.env = SimpleNamespace(task=SimpleNamespace(success=False, object_scope={"car": entity}))
            self.metrics = [SimpleNamespace(aggregate=lambda env: {"q_score": {"final": 0.0}})]
            client = SimpleNamespace(
                get_server_metadata=lambda: {"policy": "lilawam_behavior_task13", "seed": cfg["seed"],
                                             "rng_reset_behavior": "Generator continues across episode resets"},
                _packer=SimpleNamespace(pack=lambda x: x),
                _ws=SimpleNamespace(send=lambda msg: status_messages.append(msg),
                                    recv=lambda **kw: {"episode": self.episode, "step": self.steps}))
            self.policy = SimpleNamespace(policy=client)

        def reset(self): self.episode += 1; self.steps = 0
        def load_task_instance(self, instance): self.instance = instance
        def step(self): self.steps += 1; self._write_joint_log(); return False, True
        def __enter__(self): return self
        def __exit__(self, *args): self.stop_joint_log()

    for name in method_names:
        setattr(FakeEvaluator, name, method_ns[name])
    eval_ns = {"argparse": argparse, "json": json, "os": os, "Path": Path, "__doc__": "CPU test",
               "logger": logging.getLogger("eval_cpu_check"), "gm": SimpleNamespace(), "Evaluator": FakeEvaluator,
               "OmegaConf": SimpleNamespace(create=lambda x: x, load=lambda _: {}),
               "DEFAULT_EVAL_SEED": constants["DEFAULT_EVAL_SEED"], "resolve_instance_ids": resolve,
               "seed_everything": lambda seed: seed_calls.append(seed) or seed}
    definitions(OG / "eval/eval.py", {"parse_args", "main"}, eval_ns)

    def fake_eval(argv):
        network = SimpleNamespace(unpackb=lambda x: x)
        with patch.dict(sys.modules, {"omnigibson.eval.utils.network_utils": network}), patch.object(sys, "argv", ["eval.py", *argv]):
            eval_ns["main"]()

    with tempfile.TemporaryDirectory(prefix="evaluation-evidence-cpu-") as directory:
        run = Path(directory) / "evidence"
        fake_eval(["--task-name", task, "--instance-indices", *map(str, indices), "--output-dir", str(run / "eval"),
                   "--seed", "23", "--log-object-joints"])
        assert seed_calls == [23] and all(msg == {"status": True} for msg in status_messages)
        expected = launch.planned_episodes(run, task, indices, 23)
        assert [r["instance_id"] for r in expected] == instance_ids
        assert len(launch.completed_results(run, expected)) == 2
        rejected(lambda: launch.planned_episodes(run, task, indices, 24))
        results = sorted((run / "eval/json").glob("*.json"))
        original = results[0].read_text()
        first = json.loads(original)
        for field, value in (("task", "wrong_task"), ("instance_id", instance_ids[0] + 1000), ("rollout_id", 7),
                             ("steps", 0), ("success", "false"), ("q_score", {"final": 2.0})):
            results[0].write_text(json.dumps({**first, field: value}))
            rejected(lambda: launch.completed_results(run, expected))
        results[0].write_text(original)
        duplicate = run / "eval/json/duplicate.json"
        duplicate.write_text(original)
        rejected(lambda: launch.completed_results(run, expected))
        duplicate.unlink()
        results[0].unlink()
        rejected(lambda: launch.completed_results(run, expected))
        results[0].write_text(original)
        print("PASS: existing instance mapping; exact task/instance/rollout set; missing/duplicate/invalid results rejected")

        episodes = [json.loads(p.read_text()) for p in results]
        events = [{"event": "orch_contract_end", "episode": e["policy_episode"], "key": "open lid(car)", "step": 1,
                   "success": True, "verified": True,
                   "postcondition": {"source": "patch_postcondition_gate:v1", "verifier_kind": "perception_predicates",
                                     "satisfied": True, "observation_id": 1, "accepted_as_evidence": False}} for e in episodes]
        trace = run / "policy_trace_fake.jsonl"
        jsonl(trace, events)
        out = summarize(run)
        assert not (run / "summary.json").exists()
        assert out["policy"]["verified_manipulation_successes"] is None
        assert len(out["policy"]["perception_candidates"]) == 2 and out["record_status"]["task_failures"] == 2
        for i, record in enumerate(out["object_joints"].values()):
            audit = record["perception_candidate_audit"]
            assert len(audit) == 1 and audit[0]["episode"] == episodes[i]["policy_episode"]
            assert audit[0]["object_joints_at_step"]["car"] == [i / 10 + 0.1]
            assert audit[0]["physical_verified"] is None
            assert audit[0]["postcondition"]["accepted_as_evidence"] is False
        logs = sorted((run / "eval/object_joints").glob("*.jsonl"))
        raw_log = logs[0].read_text()
        joint_rows = [json.loads(line) for line in raw_log.splitlines()]
        # A non-contiguous step is looked up by its explicit ID, not by a list offset.
        sparse = [dict(row, step=7) if row.get("step") == 1 else row for row in joint_rows]
        jsonl(logs[0], sparse)
        results[0].write_text(json.dumps({**first, "steps": 7}))
        events[0]["step"] = events[0]["postcondition"]["observation_id"] = 7
        jsonl(trace, events)
        assert summarize(run)["policy"]["perception_candidates"][0]["evidence_status"] == "matched_raw_joint_evidence"
        # An out-of-range event must not clamp to the final available joint sample.
        events[0]["step"] = events[0]["postcondition"]["observation_id"] = 99
        jsonl(trace, events)
        audit = next(iter(summarize(run)["object_joints"].values()))["perception_candidate_audit"][0]
        assert audit["evidence_status"] == "unknown" and audit["object_joints_at_step"] is None
        events[0]["step"] = events[0]["postcondition"]["observation_id"] = 1
        jsonl(trace, events)
        results[0].write_text(original)
        jsonl(logs[0], [r for r in joint_rows if r.get("step") != 1])
        assert summarize(run)["policy"]["perception_candidates"][0]["evidence_reason"] == "exact_joint_step_missing"
        jsonl(logs[0], [*joint_rows, joint_rows[2]])
        assert summarize(run)["policy"]["perception_candidates"][0]["evidence_reason"] == "invalid_or_duplicate_joint_steps"
        logs[0].write_text(raw_log)
        events[0]["postcondition"]["observation_id"] = 0
        jsonl(trace, events)
        assert summarize(run)["policy"]["perception_candidates"][0]["evidence_reason"] == "postcondition_observation_step_mismatch"
        events[0]["postcondition"]["observation_id"] = 1
        jsonl(trace, events)
        joint_rows[0].pop("episode_identity")
        jsonl(logs[0], joint_rows)
        unknown = summarize(run)
        assert next(iter(unknown["object_joints"].values()))["perception_candidate_audit"] == []
        assert unknown["policy"]["perception_candidates"][0]["evidence_status"] == "unknown"
        print("PASS: dual-episode binding; exact/sparse/missing/duplicate/out-of-range steps; legacy patch claims remain perception-only")

        artifact = Path(directory) / "checkpoint.pt"
        artifact.write_text("synthetic checkpoint; never loaded")
        model_cfg = {"dataset": {"stats_path": "data/behavior-1k/2026-challenge-demos/train_stats_task013.json"},
                     "model": {"vision_encoder": {"checkpoint_path": "models/dinov3-vitl16-pretrain-lvd1689m"}}}
        args = SimpleNamespace(checkpoint=str(artifact), program=None, memory_vtt=None,
                               orchestrate=json.dumps({"arrival_gate": "patch", "postcondition_gate": "patch",
                                                       "train_targets": "data/skill_vtt_task013/contract_targets.json"}))
        paths = launch.runtime_artifact_paths(args, ["--robot-config", str(OG / "eval/r1pro.yaml")], model_cfg)
        for name in ("data/skill_vtt_task013/contract_targets.json", "data/skill_vtt_task013/precond_patch_prototypes.npz",
                     "data/skill_vtt_task013/postcond_patch_meta.json", "data/behavior-1k/2026-challenge-demos/split_task_013.json",
                     "harness/policy_server.py", "harness/orchestrator.py"):
            assert (ROOT / name).resolve() in paths
        assert (OG / "eval/r1pro.yaml").resolve() in paths

        # Run the actual launcher main with process boundaries mocked: argv and the written manifest are exercised.
        for label, seed, mutate in (("default", 0, None), ("explicit", 41, None), ("changed", 41, "artifact"), ("bad-seed", 41, "seed")):
            artifact.write_text("synthetic checkpoint; never loaded")
            def guarded(argv, environment, output_dir, timeout):
                fake_eval(argv[3:])
                if mutate == "artifact":
                    artifact.write_text("changed during fake evaluation")
                if mutate == "seed":
                    (output_dir / "eval/policy_metadata.json").write_text(json.dumps({"effective_policy_seed": seed + 1}))
                return 0
            argv = ["behavior_closed_loop.py", "--profile", "development", "--instances", "10", "--run-name", label,
                    "--checkpoint", str(artifact), "--policy-device", "cpu"]
            if label != "default": argv += ["--seed", str(seed)]
            with patch.object(sys, "argv", argv), patch.object(launch, "ROOT", Path(directory)), \
                 patch.object(launch, "collect", return_value={"blockers": []}), \
                 patch.object(launch, "checkpoint_config", return_value=model_cfg), \
                 patch.object(launch, "runtime_artifact_paths", return_value=[artifact]), \
                 patch.object(launch, "run_guarded", side_effect=guarded), \
                 patch.object(launch.threading, "Thread"), patch.object(launch.subprocess, "run"), \
                 patch.object(launch.subprocess, "Popen", side_effect=AssertionError("A real process must not start")), \
                 redirect_stdout(io.StringIO()):
                try: launch.main()
                except SystemExit as exc: assert exc.code == (2 if mutate else 0), exc.code
                else: raise AssertionError("Launcher did not report its exit code")
            manifest = json.loads((Path(directory) / "outputs/behavior/closed_loop" / label / "manifest.json").read_text())
            for name in ("evaluator_argv", "policy_argv"):
                assert manifest[name][manifest[name].index("--seed") + 1] == str(seed)
            assert manifest["seed_behavior"]["physics_rendering_fully_deterministic"] is False
            assert "reset does not reseed" in manifest["seed_behavior"]["policy"]
            assert manifest["evaluation_complete"] is (mutate is None)
            if mutate is None:
                assert manifest["effective_policy_seed"] == manifest["effective_evaluator_seed"] == seed
                assert manifest["artifact_sha256"][str(artifact)] == launch.sha256_file(artifact)
            elif mutate == "artifact": assert manifest["artifact_changes"] == [str(artifact)]
        print("PASS: default/explicit seed argv + effective seed receipts/manifest; artifact hashes and mutation/mismatched-seed rejection")
    assert "torch" not in sys.modules and "omnigibson" not in sys.modules
    print("PASS: CPU-only; no model/simulator/GPU imports; only temporary test outputs were written")


if __name__ == "__main__":
    main()
