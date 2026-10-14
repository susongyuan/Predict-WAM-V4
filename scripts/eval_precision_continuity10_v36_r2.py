#!/usr/bin/env python3
"""Ten preregistered paired scenes with C1/fine250 execution; first case observer video explicitly requested."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import precision_continuity_evidence_v36 as e
from scripts import eval_zhuhai_precision_continuity_v36 as runtime
from scripts import eval_precision_dagger_70k80k_fixed5_v19 as old
from scripts import precision_continuity_physics_abort_v36 as contact
from scripts import gpu_lock, run_zhuhai_task1_dr200_concurrent as admission
from scripts.analyze_precision_dagger_milestones_v19 import episode, ORDER
from scripts.progress1_log import log
from harness import zhuhai_chunk_continuity_v36 as braking

BASE = ROOT / "outputs/zhuhai2026_tasks12"
REFERENCE = BASE / "wam_precision_v2_task2_dagger_80k_fixed5_20261011_v23"
SEEDS = tuple(range(932106, 932116))
SCHEMA = "precision-80k-continuity10-paired-source-v36"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def classify_noncontact(path, metadata, frozen, source_hash):
    """Same immutable official-braking admission checks, with explicit new seed set."""
    import numpy as np
    path = Path(path)
    raw = e.read_json(path)
    abort = raw.get("status") == "infrastructure_error"
    if abort:
        prefix = f"InfrastructureError: wrong worker reply/ID, expected action/{e.TASK}_seed_{raw.get('seed')}/{raw.get('control_steps')}: "
        error = raw.get("error", "")
        e.require(error.startswith(prefix), "Unknown infrastructure error; preserve and HALT")
        reply = ast.literal_eval(error[len(prefix):])
        e.require(reply == dict(version=1, kind="infrastructure_error", episode="startup", step=0,
                               error="ValueError: " + braking.SIGNATURE), "Unknown worker rejection")
    else:
        e.require(raw.get("status") == "completed" and not raw.get("error"), "Nonterminal/infrastructure result")
    seed = raw.get("seed")
    e.require(seed in SEEDS and 0 < raw.get("control_steps", 0) <= 2400, "Invalid additional physical attempt")
    changed = runtime.overlay(frozen)
    binding = e.validate_binding(raw, metadata["step"], seed, metadata["sha256"], changed)
    e.require(binding.get("evaluation_version") == braking.VERSION
              and binding.get("evaluation_source_sha256") == source_hash
              and binding.get("training_contract_sha256") == e.digest(frozen), "Runtime overlay differs")
    terminal = raw.get("last_rule_evidence" if abort else "final_rule_evidence")
    score = e.validate_rule(terminal)
    if abort:
        e.require(raw.get("full_task_success") is None and raw.get("base_score") is None
                  and score["full_task_success"] is False, "Abort score changed")
    else:
        e.require(raw.get("full_task_success") == score["full_task_success"]
                  and raw.get("base_score") == score["base_score"], "Raw/physical score differs")
        e.require((score["full_task_success"] and raw.get("termination") == "full_rule_success")
                  or (not score["full_task_success"] and raw.get("termination") == "fixed_step_limit"
                      and raw["control_steps"] == 2400), "Invalid termination/budget")
    files = e.archive_evidence(path.parent, raw, binding, terminal)
    from scripts.precision_continuity_replay_v36 import replay,video_evidence
    replayed=replay(path.parent,changed['action_postprocessing'])
    files.update(video_evidence(path.parent,raw))
    if abort:
        e.require(replayed['terminal_braking_interval_infeasible'], 'Unproven terminal fine-command rejection')
    return dict(task=e.TASK, seed=seed, status="control_safety_abort" if abort else "completed",
                full_task_success=False if abort else raw["full_task_success"], base_score=raw["base_score"],
                observed_base_score=score["base_score"], control_steps=raw["control_steps"], attempt_count=1,
                raw_result_path=str(path.resolve()), raw_result_sha256=e.sha256_file(path), evidence_files=files,
                **{k:v for k,v in replayed.items() if k!='terminal_braking_interval_infeasible'})


def classify(path, metadata, frozen, source_hash):
    return contact.classify(path, metadata, frozen, source_hash, SEEDS, classify_noncontact)


def make_plan(out):
    reference = e.read_json(REFERENCE / "plan.json")
    prior = e.read_json(REFERENCE / "summary.json")
    e.require(prior["status"] == "completed" and len(prior["episodes"]) == 5, "Original five incomplete")
    old.verify(reference)
    metadata, contract = old.metadata_milestone(reference["checkpoint"]["path"], 80000)
    e.require(metadata == reference["checkpoint"] == prior["checkpoint"], "Actual80k model changed")
    e.require(not set(SEEDS) & set(prior["seeds"]), "Do not replay the five old failures")
    out = Path(out).resolve()
    e.require(BASE in out.parents and not out.exists(), "Fresh continuity10 output required")
    continuation = e.read_json(BASE / "verification/dagger_firstgrasp_diagnostic_20261011_v21/conditional_plan_v23.json")
    plan = deepcopy(reference)
    plan.update(schema=SCHEMA, out=str(out), seeds=list(SEEDS), planned_episodes=10,
                time=now(), original_five=dict(path=str(REFERENCE / "summary.json"),
                                               sha256=e.sha256_file(REFERENCE / "summary.json")),
                interpretation="10 preregistered first-ten80k extra20 seeds under changed execution protocol, paired development comparison; first-case requested observer render differs; no heldout or pure smoothing causal claim",
                gpu_lock_owner="zhuhai-continuity10-20261012-v36",
                systemd_unit="p2-zhuhai-continuity10-20261012-v36-r2.service",
                user_authorization="先修复整臂动作块连续性后测试十次；第一例真实第三视角录像，其余默认不录像；不追加训练采集",
                runtime_environment=continuation["runtime_environment"])
    paired_path=BASE/'wam_precision_v2_task2_80k_extra20_20261011_v24/summary.json'
    paired=e.read_json(paired_path)
    e.require(paired['status']=='completed' and len(paired['episodes'])==20 and paired['checkpoint']==metadata, 'Original20 paired baseline incomplete/changed')
    plan['paired_original20']=dict(path=str(paired_path),sha256=e.sha256_file(paired_path),
        paired_seeds=list(SEEDS),scores=[row['observed_base_score'] for row in paired['episodes'][:10]],
        full_successes=sum(row['full_task_success'] for row in paired['episodes'][:10]))
    plan['action_postprocessing']=braking.protocol()
    plan['action_postprocessing_sha256']=e.digest(plan['action_postprocessing'])
    plan['recording']=dict(requested=True,recorded_seeds=[932106],selection='first preregistered case',
        policy_inputs=['head_camera','left_camera','right_camera'],observer_is_policy_input=False)
    for name in ('scripts/eval_precision_continuity10_v36_r2.py','scripts/eval_zhuhai_precision_continuity_v36.py',
                 'scripts/precision_continuity_evidence_v36.py','scripts/precision_continuity_physics_abort_v36.py',
                 'scripts/precision_continuity_replay_v36.py','scripts/test_precision_continuity_v36.py',
                 'harness/zhuhai_chunk_continuity_v36.py','harness/zhuhai_observer_video_v36.py'):
        source=ROOT/name;plan['extension_sources'][str(source)]=e.sha256_file(source)
    plan['child_env']['P2_CONTINUITY_THIRD_VIEW_SEED']='932106' 
    plan["child_env"].update(plan["runtime_environment"])
    plan["commands"] = [dict(seed=s, out=str(out / e.TASK / f"seed_{s}"), command=[
        str(e.PYTHON), "-B", str(Path(runtime.__file__).resolve()), "--checkpoint", metadata["path"],
        "--out", str(out / e.TASK / f"seed_{s}"), "--task", e.TASK,
        "--gpu-lock-owner", plan["gpu_lock_owner"], "--episode-seed", str(s),
        "--max-steps", "2400", "--wall-seconds", "3600", "--startup-timeout", "180", "--policy-timeout", "60"])
        for s in SEEDS]
    return plan, contract


def verify(plan):
    old.verify(plan)
    e.require(e.sha256_file(plan['paired_original20']['path'])==plan['paired_original20']['sha256'],'Paired original20 changed')
    e.require(e.sha256_file(plan["original_five"]["path"]) == plan["original_five"]["sha256"], "Original five changed")


def run_once(item, plan, out):
    env = dict(os.environ, **plan["child_env"])
    with (out / f"seed_{item['seed']}.log").open("xb") as log_file:
        child = subprocess.Popen(item["command"], cwd=ROOT, env=env, stdout=log_file,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        e.write_json(out / "active_participant.json", dict(pid=child.pid, pgid=child.pid,
                                                          seed=item["seed"], command=item["command"], time=now()))
        started = time.monotonic()
        bad_nvml = 0
        try:
            while child.poll() is None:
                e.require(time.monotonic() - started < 3840, "Physical episode wall deadline")
                file = Path(item["out"]) / "result.json"
                state = e.read_json(file) if file.exists() else {}
                age = time.time() - file.stat().st_mtime if file.exists() else time.monotonic() - started
                e.require(not (state.get("status") in (None, "starting") and age > 300), "Startup exceeds300s")
                e.require(not (state.get("status") == "running" and age > 600), "Action heartbeat stale600s")
                try:
                    health = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                            env=env, capture_output=True, text=True, timeout=10)
                    healthy = health.returncode == 0 and health.stdout.strip() == "595.91.07"
                except subprocess.TimeoutExpired:
                    healthy = False
                bad_nvml = 0 if healthy else bad_nvml + 1
                e.require(bad_nvml < 3, "Pinned NVML failed3 consecutive checks")
                e.write_json(out / "watchdog_status.json", dict(time=now(), status="watching", seed=item["seed"],
                    pid=child.pid, phase=state.get("status"), control_steps=state.get("control_steps"),
                    heartbeat_age_seconds=age, pinned_driver_healthy=healthy,
                    startup_limit_seconds=300, action_heartbeat_limit_seconds=600))
                time.sleep(30)
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def wilson(successes, n):
    z = 1.959963984540054
    p = successes/n
    center = (p + z*z/(2*n))/(1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def failure_analysis(out, cohort):
    rows = [episode(r) for r in cohort["episodes"]]
    return dict(schema="80k-continuity10-scored-trajectory-analysis-v36", time=now(),
                tests=10, full_successes=cohort["full_successes"],
                full_success_wilson95=wilson(cohort["full_successes"], 10),
                scores=[r["observed_base_score"] for r in rows],
                failure_stages=dict(Counter(r["failure_stage"] for r in rows if not r["full_task_success"])),
                terminal_statuses=dict(Counter(r["status"] for r in rows)),
                per_object={k:dict(verified_grasp_cases=sum(r["objects"][k]["ever_verified_grasp"] for r in rows),
                                   valid_placement_cases=sum(r["objects"][k]["ever_valid_placement"] for r in rows)) for k in ORDER},
                episodes=rows, summary_path=str(out/"summary.json"), summary_sha256=e.sha256_file(out/"summary.json"),
                scope="10 paired source development scenes; fixed weights, two execution continuity changes, first-case requested observer; not isolated component causality or heldout generalization")


def execute(plan, contract):
    out = Path(plan["out"])
    out.mkdir(exist_ok=False)
    e.write_json(out / "plan.json", plan)
    rows, held = [], False

    def publish(status, error=None):
        value = dict(schema=SCHEMA, status=status, time=now(), checkpoint=plan["checkpoint"],
                     checkpoint_step=80000, checkpoint_sha256=plan["checkpoint"]["sha256"],
                     planned_episodes=10, seeds=list(SEEDS), episodes=rows,
                     full_successes=sum(r["full_task_success"] for r in rows),
                     success_rate=sum(r["full_task_success"] for r in rows)/10 if status=="completed" else None,
                     evaluation_version=braking.VERSION, action_postprocessing_sha256=plan["action_postprocessing_sha256"],
                     record_video=True, third_view=True, recorded_seeds=[932106], randomized=False, original_five=plan["original_five"],
                     interpretation=plan["interpretation"])
        if error:
            value["error"] = error
        e.write_json(out / "summary.json", value)
        return value

    publish("waiting_for_exclusive_gpu")
    try:
        verify(plan)
        deadline = time.monotonic() + 21600
        while gpu_lock.owner() is not None or admission.gpu_jobs(admission.processes()):
            e.require(time.monotonic() < deadline, "Continuity10 GPU wait exceeded6h")
            time.sleep(10)
        resource = admission.scheduler_module().collection_resource_admission()
        e.require(admission.scheduler_module().resource_ready(resource, expansion=True), "GPU/RAM admission insufficient")
        gpu_lock.acquire(plan["gpu_lock_owner"], max_wait=0)
        held = True
        e.require(not admission.gpu_jobs(admission.processes()), "Foreign GPU participant appeared")
        with Path(plan["checkpoint"]["path"]).open("rb") as locked:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
            publish("evaluating")
            log("80k连续性十测：932106–932115预注册配对；C1/250Hz官方制动，严格原评分，首例第三视角录像，其余无录像/额外相机；不重试有效失败。", "root")
            for item in plan["commands"]:
                verify(plan)
                e.verify_external_owner(plan["gpu_lock_owner"])
                target = Path(item["out"])
                target.parent.mkdir(exist_ok=True)
                e.require(not target.exists(), "No repeated physical attempt")
                e.write_json(out / f"attempt_seed_{item['seed']}.json", item)
                code = run_once(item, plan, out)
                e.write_json(out / f"exit_seed_{item['seed']}.json", dict(returncode=code))
                row = classify(target / "result.json", plan["checkpoint"], contract,
                               plan["extension_sources"][str(Path(runtime.__file__).resolve())])
                e.require(code in (0,2) if row["status"] in ("control_safety_abort","physics_safety_abort") else code==0,
                          "Result/exit mismatch")
                rows.append(row)
                publish("evaluating")
                log(f"80k连续性测试{len(rows)}/10 seed{item['seed']}：{row['status']}，得分{row['observed_base_score']}，整任务成功{row['full_task_success']}。", "root")
            verify(plan)
            cohort = publish("completed")
        e.write_json(out / "failure_analysis.json", failure_analysis(out, cohort))
        e.write_json(out / "final_receipt.json", dict(status="completed", time=now(), tests=10,
            full_successes=cohort["full_successes"], scores=[r["observed_base_score"] for r in rows],
            prior_five_preserved=True, attempts_not_retried=True, analysis=str(out/"failure_analysis.json")))
        return 0
    except BaseException as error:
        publish("halted", f"{type(error).__name__}: {error}")
        e.write_json(out / "failure_guard.json", dict(time=now(), error=f"{type(error).__name__}: {error}",
                                                      all_attempts_preserved=True, retries=0))
        raise
    finally:
        if held:
            jobs = admission.gpu_jobs(admission.processes())
            if jobs:
                e.write_json(out / "owner_retained_guard.json", dict(owner=plan["gpu_lock_owner"], jobs=jobs))
            else:
                gpu_lock.release(plan["gpu_lock_owner"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan, contract = make_plan(args.out)
    if args.execute:
        raise SystemExit(execute(plan, contract))
    print(json.dumps(dict(checkpoint=plan["checkpoint"], seeds=plan["seeds"], out=plan["out"],
                          record_video=True, third_view=True, recorded_seeds=[932106], source_count=len(plan["extension_sources"]))))
