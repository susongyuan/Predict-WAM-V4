#!/usr/bin/env python3
"""Reaudit the only completed v36 case without editing its halted summary or guard."""
from pathlib import Path
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import eval_precision_continuity10_v36_r2 as run
from scripts import precision_continuity_replay_v36 as old_replay
from scripts import precision_continuity_replay_audit_v37 as fixed
from scripts import precision_continuity_evidence_v36 as e
from scripts import gpu_lock
from scripts.progress1_log import log

def main():
    out=ROOT/'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_continuity10_20261012_v36_r2'
    dest=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24'
    target=dest/'continuity_firstcase_independent_audit_v37.json'
    e.require(not target.exists(),'preserve previous independent audit')
    plan=e.read_json(out/'plan.json');run.verify(plan)
    _,contract=run.old.metadata_milestone(plan['checkpoint']['path'],80000)
    baseline={name:e.sha256_file(out/name) for name in ('summary.json','failure_guard.json','plan.json')}
    e.require(len(list(out.glob('attempt_seed_*.json')))==1,'unexpected additional attempts')
    e.require(e.read_json(out/'exit_seed_932106.json')=={'returncode':0},'physical evaluation did not exit normally')
    old_replay.replay=fixed.replay;old_replay.video_evidence=fixed.video_evidence
    directory=out/e.TASK/'seed_932106'
    row=run.classify(directory/'result.json',plan['checkpoint'],contract,
        plan['extension_sources'][str(Path(run.runtime.__file__).resolve())])
    raw=e.read_json(directory/'result.json')
    e.require(row['control_steps']==2400 and row['observed_base_score']==0 and not row['full_task_success'],
        'unexpected first-case result')
    e.require(gpu_lock.owner() is None,'GPU owner still active')
    for name,sha in baseline.items():e.require(e.sha256_file(out/name)==sha,'old summary/guard/plan changed')
    receipt=e.read_json(directory/'third_person_video_receipt.json')
    export=ROOT/'success video/80k_C1_only_seed932106_third_person_full.mp4'
    e.require(not export.exists(),'preserve previous video')
    shutil.copy2(directory/'third_person_full.mp4',export)
    e.require(e.sha256_file(export)==receipt['video_sha256'],'video export differs')
    value=dict(schema='v36-firstcase-independent-audit-v37',status='passed',episodes=[row],
        summary_status='halted_original_queue_preserved',old_summary_guard_plan_sha256=baseline,
        actual_physical_attempts=1,canceled_remaining_attempts=9,checkpoint=plan['checkpoint'],
        action_postprocessing=plan['action_postprocessing'],gpu_owner_released=True,
        cause='archive uses float32 exact cast; old auditor required float64 equality',
        video_binding='separate source-hashed close receipt; original raw result lacks post-close annotation',
        original_raw_result_preserved=True,full_video_decode_verified=True,
        exported_video=str(export),recording_receipt=receipt,
        user_scope='diagnose and improve generation/task progress, then one improved third-person trial; no more ten-test queue',
        audit_source_sha256=e.sha256_file(Path(__file__)),
        replay_audit_source_sha256=e.sha256_file(Path(fixed.__file__)))
    e.write_json(target,value);e.write_json(export.with_suffix('.provenance.json'),value)
    # A separate verified snapshot allows existing read-only analyses to run.
    snapshot=dest/'continuity_firstcase_cohort_v37.json'
    e.write_json(snapshot,dict(status='completed',planned_episodes=1,seeds=[932106],episodes=[row],
        checkpoint=plan['checkpoint'],full_successes=0,source_plan=str(out/'plan.json'),
        scope='one completed preliminary C1/fine250 case; not a ten-case success rate'))
    log('纠正前条SIGINT记录：systemctl选项被拒且unit早已自动halted，未发信号。实测首例2400步0分；新独立核验确认float32精确归档与全部60000派发、安全/制动/录像PASS，保留原summary/guard，不续10测。','root')
    print(json.dumps({k:value[k] for k in ('status','actual_physical_attempts','cause','exported_video')},ensure_ascii=False))

if __name__=='__main__':main()
