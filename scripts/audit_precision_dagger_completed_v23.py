#!/usr/bin/env python3
"""Independent CPU-only completion audit; no new simulator attempts or training."""
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_zhuhai_precision_fixed_attempts as e
from scripts import eval_precision_dagger_70k80k_fixed5_v19 as evaluator
from scripts.analyze_precision_dagger_milestones_v19 import analyze
from harness.zhuhai_dagger_training30k_v14 import validate_quota

BASE = ROOT / "outputs/zhuhai2026_tasks12"
VERIFY = BASE / "verification/dagger_firstgrasp_diagnostic_20261011_v21"


def binding(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=e.sha256_file(path), size=path.stat().st_size)


def main():
    output = VERIFY / "completed_training_and_tests_audit_v23.json"
    e.require(not output.exists(), "Preserve the previous audit")
    plan_path = VERIFY / "conditional_plan_v23.json"
    plan = e.read_json(plan_path)
    pipe = Path(plan["out"])
    status = e.read_json(pipe / "status.json")
    receipt = e.read_json(pipe / "final_receipt.json")
    e.require(status["status"] == status["stage"] == "completed", "Pipeline not complete")
    e.require(receipt["status"] == "completed" and receipt["final_step"] == 80000
              and receipt["additional_optimizer_updates"] == 30000
              and receipt["new_extra_DR_episodes"] == 50
              and receipt["total_corrections"] == 120 and receipt["tests_each"] == 5,
              "Requested workflow quotas differ")
    e.require(plan["test_milestones"] == [70000, 80000] and plan["tests_each"] == 5,
              "Unexpected test milestone")
    e.require(plan["record_video"] is False and plan["third_view"] is False,
              "Recording scope differs")
    for path, expected in {**plan["source_files"], **plan["bound_inputs"]}.items():
        e.require(e.sha256_file(path) == expected, "Preserved source/input changed: " + path)
    print("frozen source and prerequisite bindings verified", flush=True)

    training = Path(plan["training"])
    tm = e.read_json(training / "status.json")
    e.require(evaluator.training_finished(tm), "Final training quota/integrity status differs")
    integrity_path = VERIFY / "final80k_cpu_integrity_v23.json"
    integrity = e.read_json(integrity_path)
    e.require(integrity["step"] == 80000 and integrity["optimizer_states"] == 323
              and integrity["optimizer_steps"] == [80000]
              and integrity["rng_complete"] is True
              and integrity["model_and_full_weights_identical"] is True
              and integrity["total_stage_updates"] == 30000
              and integrity["expert_weights_all_logs"] == [1, 1, 1, 3]
              and integrity["frozen_DINO_all_logs"] is True,
              "Independent full-state verification incomplete")
    checkpoints = {}
    for name, proof in [("checkpoint_last.pt", integrity["full_receipt"]),
                        ("checkpoint_step_080000.pt", integrity["model_receipt"])]:
        path = training / "train" / name
        actual = binding(path)
        e.require(actual["size"] == proof["size"] and actual["sha256"] == proof["sha256"],
                  "Verified final checkpoint changed")
        checkpoints[name] = actual

    collection = Path(plan["collection"])
    cm = e.read_json(collection / "manifest.json")
    validate_quota(cm)
    e.require(cm["active_participants"] == [] and cm["third_view"] is False,
              "Collection still active or extra view exists")
    cp = e.read_json(collection / "plan.json")
    initial_analysis_path = Path(cp["failure_analysis"])
    e.require(e.sha256_file(initial_analysis_path) == cp["failure_analysis_sha256"],
              "Original failure analysis changed")
    initial = e.read_json(initial_analysis_path)
    original = {}
    for step, summary_binding in initial["input_summaries"].items():
        path = Path(summary_binding["path"])
        e.require(e.sha256_file(path) == summary_binding["sha256"], "Original fixed10 cohort changed")
        cohort = e.read_json(path)
        e.require(cohort["status"] == "completed" and len(cohort["episodes"]) == 10,
                  "Original fixed10 incomplete")
        for row in cohort["episodes"]:
            e.require(e.sha256_file(row["raw_result_path"]) == row["raw_result_sha256"],
                      "Original physical attempt changed")
        original[step] = dict(summary=summary_binding, full_successes=cohort["full_successes"], tests=10)

    expert_audit_path = VERIFY / "expert_duration_half_weight_audit_v22.json"
    expert_audit = e.read_json(expert_audit_path)
    e.require(expert_audit["new_episodes"] == 50 and expert_audit["all_mask_checks_passed"] is True,
              "Expert boundary-mask audit incomplete")
    contract = e.read_json(training / "train/manifest.json")["contract"]
    weights = contract["dagger_training"]["action_loss_weighting"]
    e.require(weights["verified_recovery_action_weight"] == 3
              and weights["original_action_weight"] == 1
              and weights["invalid_and_policy_actions_weight"] == 0
              and contract["dagger_training"]["original_splits_unchanged"] is True,
              "Requested expert weighting or data split changed")
    print("80k checkpoint, 50 DR additions, and original45k/50k evidence verified", flush=True)

    paths, cohorts, episode_audits = {}, {}, {}
    for step in (70000, 80000):
        directory = Path(plan["evaluation_outs"][str(step)])
        path = directory / "summary.json"
        paths[step] = path
        ep = e.read_json(directory / "plan.json")
        cohort = e.read_json(path)
        evaluator.verify(ep)
        metadata, contract = evaluator.metadata_milestone(ep["checkpoint"]["path"], step)
        e.require(metadata == ep["checkpoint"] == cohort["checkpoint"], "Checkpoint binding changed")
        e.require(cohort["status"] == "completed" and cohort["planned_episodes"] == 5
                  and cohort["checkpoint_step"] == step
                  and cohort["seeds"] == list(evaluator.SEEDS)
                  and [r["seed"] for r in cohort["episodes"]] == list(evaluator.SEEDS),
                  "Exact requested paired five-test cohort missing")
        e.require(cohort["record_video"] is False and cohort["third_view"] is False
                  and cohort["randomized"] is False, "Evaluation protocol changed")
        e.require(len(list(directory.glob("attempt_seed_*.json"))) == 5
                  and len(list((directory / e.TASK).glob("seed_*"))) == 5,
                  "Unexpected or duplicate attempt outputs")
        e.require(not any(directory.rglob("*.mp4")), "Unexpected recorded evaluation video")
        rows = []
        source_hash = ep["extension_sources"][str(Path(evaluator.evaluator.__file__).resolve())]
        for admitted in cohort["episodes"]:
            reclassified = evaluator.classify(Path(admitted["raw_result_path"]), metadata, contract, source_hash)
            e.require(reclassified == admitted, "Independent archive/score/braking admission differs")
            raw = e.read_json(admitted["raw_result_path"])
            e.require(raw["system_mode"] == "pure_wam", "Full-stack intervention changed baseline")
            code = e.read_json(directory / f'exit_seed_{admitted["seed"]}.json')["returncode"]
            expected = (0, 2) if admitted["status"] in ("physics_safety_abort", "control_safety_abort") else (0,)
            e.require(code in expected and admitted["attempt_count"] == 1, "Exit/attempt count differs")
            rows.append(dict(seed=admitted["seed"], status=admitted["status"],
                             control_steps=admitted["control_steps"], score=admitted["observed_base_score"],
                             full_task_success=admitted["full_task_success"],
                             verified_evidence_files=len(admitted["evidence_files"]),
                             raw_result_sha256=admitted["raw_result_sha256"],
                             independent_scoring_and_deterministic_braking_replay=True))
            print(f"{step} seed{admitted['seed']}: full archive and deterministic command replay verified", flush=True)
        e.require(cohort["full_successes"] == sum(r["full_task_success"] for r in rows)
                  and cohort["success_rate"] == cohort["full_successes"] / 5,
                  "Aggregate success count differs")
        episode_audits[str(step)] = rows
        cohorts[str(step)] = binding(path)

    analysis_path = pipe / "failure_analysis_70k80k.json"
    stored = e.read_json(analysis_path)
    fresh = analyze(paths, steps=(70000, 80000), count=5)
    for key in ("milestones", "paired_cases", "input_summaries"):
        e.require(fresh[key] == stored[key], "Independent trajectory-progress analysis differs")
    value = dict(schema="precision-dagger-requested-workflow-completion-audit-v23",
                 time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                 status="passed", training_step=80000, additional_updates=30000,
                 expert_action_weight=3, original_action_weight=1,
                 train_contract= binding(training / "train/manifest.json"),
                 final_checkpoint_integrity=binding(integrity_path), checkpoints=checkpoints,
                 original45k50k=original, extra_DR_episodes=50, total_corrections=120,
                 collection_manifest=binding(collection / "manifest.json"),
                 expert_boundary_mask_audit=binding(expert_audit_path),
                 paired_source_seeds=list(evaluator.SEEDS), cohorts=cohorts,
                 episode_audits=episode_audits, failure_analysis=binding(analysis_path),
                 record_video=False, third_view=False, pure_wam=True,
                 all_original_evidence_preserved=True,
                 verified_frozen_source_count=len(plan["source_files"]),
                 verified_prerequisite_count=len(plan["bound_inputs"]),
                 auditor_sha256=e.sha256_file(__file__), plan=binding(plan_path),
                 final_receipt=binding(pipe / "final_receipt.json"))
    with output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(audit=str(output), status="passed", full_successes={
        step: stored["milestones"][step]["full_successes"] for step in ("70000", "80000")})))


if __name__ == "__main__":
    main()
