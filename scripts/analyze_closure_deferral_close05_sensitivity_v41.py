#!/usr/bin/env python3
"""Supplementary <=0.5 closure-onset sensitivity for the v41 deferral analysis.

Recomputes, from the same archived raw chunks, the first index whose left/right
gripper value is <=0.5 (looser onset than the audit threshold 0.2), per replan
and per seed, plus counts of chunks where the <=0.5 onset exists but <=0.2 never
appears anywhere in the 32-step chunk, and dispatched <=0.5 close-run counts.
Read-only, separate receipt from the primary v41 deliverables.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CLOSE_ONSET = 0.5
CLOSE_AUDIT = 0.2
EXEC = 8
HORIZON = 32
GRIPPERS = {'left': 7, 'right': 15}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def require(flag, message):
    if not flag:
        raise RuntimeError(message)


def runs(flags):
    result, start = [], None
    for i, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            result.append([start, i - 1])
            start = None
    return result


def first_le(values, threshold):
    for i, v in enumerate(values):
        if v <= threshold:
            return i
    return None


def analyze(label, row):
    directory = Path(row['raw_result_path']).parent
    action_bytes = (directory / 'actions.jsonl').read_bytes()
    require(sha_bytes(action_bytes) == row['evidence_files']['actions.jsonl']['sha256'],
            f'{label} seed {row["seed"]}: actions hash differs')
    actions = [json.loads(line) for line in action_bytes.splitlines()]
    sent = [[a['dispatched_action'][7], a['dispatched_action'][15]] for a in actions]
    result = dict(cohort=label, seed=row['seed'], replans=0, left={}, right={})
    stats = {side: dict(none_05=0, exec_05=0, tail_05=0, onset_05_no_02_anywhere=0,
                        onset_05_in_exec_no_02_anywhere=0, onset_05_exec_close_02_tail=0,
                        hist_05=Counter()) for side in GRIPPERS}
    for a in actions:
        if not a['policy']['replanned']:
            continue
        result['replans'] += 1
        raw = a['policy']['raw_denormalized_chunk']
        for side, g in GRIPPERS.items():
            column = [v[g] for v in raw]
            i05 = first_le(column, CLOSE_ONSET)
            i02 = first_le(column, CLOSE_AUDIT)
            entry = stats[side]
            if i05 is None:
                entry['none_05'] += 1
            else:
                entry['hist_05'][str(i05 + 1)] += 1
                if i05 < EXEC:
                    entry['exec_05'] += 1
                else:
                    entry['tail_05'] += 1
            if i05 is not None and i02 is None:
                entry['onset_05_no_02_anywhere'] += 1
                if i05 < EXEC:
                    entry['onset_05_in_exec_no_02_anywhere'] += 1
            if i05 is not None and i02 is not None and i05 < EXEC <= i02:
                entry['onset_05_exec_close_02_tail'] += 1
    for side, g in GRIPPERS.items():
        entry = stats[side]
        dispatched = runs([v[0 if side == 'left' else 1] <= CLOSE_ONSET for v in sent])
        result[side] = dict(
            none_05=entry['none_05'], exec_05=entry['exec_05'], tail_05=entry['tail_05'],
            onset_05_no_02_anywhere=entry['onset_05_no_02_anywhere'],
            onset_05_in_exec_no_02_anywhere=entry['onset_05_in_exec_no_02_anywhere'],
            onset_05_exec_close_02_tail=entry['onset_05_exec_close_02_tail'],
            first_onset_index_1based_histogram=dict(sorted(entry['hist_05'].items(),
                                                           key=lambda kv: int(kv[0]))),
            dispatched_onset_runs_le_05=len(dispatched),
            dispatched_onset_first_step=dispatched[0][0] if dispatched else None)
    return result


def main():
    base = Path('/home/user/Robotic/project2/outputs/zhuhai2026_tasks12')
    verification = base / 'verification/dagger80k_extra20_20261011_v24'
    sources = []
    snap = (base / 'wam_precision_v2_task2_80k_extra20_20261011_v24/summary.json').read_bytes()
    extra = json.loads(snap)
    sources.append(dict(cohort='extra20', path=str((base / 'wam_precision_v2_task2_80k_extra20_20261011_v24/summary.json').resolve()),
                        sha256=sha_bytes(snap)))
    rows = [('extra20', r) for r in extra['episodes']]
    snap = (verification / 'continuity_firstcase_cohort_v37.json').read_bytes()
    v36 = json.loads(snap)
    sources.append(dict(cohort='v36_C1', path=str((verification / 'continuity_firstcase_cohort_v37.json').resolve()),
                        sha256=sha_bytes(snap)))
    rows += [('v36_C1', r) for r in v36['episodes'] if r['seed'] == 932106]
    snap = (base / 'wam_precision_v2_task2_rtc_single_20261012_v37/summary.json').read_bytes()
    v37 = json.loads(snap)
    sources.append(dict(cohort='v37_RTC', path=str((base / 'wam_precision_v2_task2_rtc_single_20261012_v37/summary.json').resolve()),
                        sha256=sha_bytes(snap)))
    rows += [('v37_RTC', r) for r in v37['episodes'] if r['seed'] == 932106]
    per_seed = [analyze(label, row) for label, row in rows]
    extra_rows = [r for r in per_seed if r['cohort'] == 'extra20']
    aggregate = {}
    for side in GRIPPERS:
        replans = sum(r['replans'] for r in extra_rows)
        none = sum(r[side]['none_05'] for r in extra_rows)
        exec_count = sum(r[side]['exec_05'] for r in extra_rows)
        tail = sum(r[side]['tail_05'] for r in extra_rows)
        hist = Counter()
        for r in extra_rows:
            hist.update(r[side]['first_onset_index_1based_histogram'])
        aggregate[side] = dict(
            replans=replans, none_05=none, exec_05=exec_count, tail_05=tail,
            conditional_p_onset_le_8=(exec_count / (replans - none)) if replans > none else None,
            first_onset_index_1based_histogram=dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
            onset_05_no_02_anywhere=sum(r[side]['onset_05_no_02_anywhere'] for r in extra_rows),
            onset_05_in_exec_no_02_anywhere=sum(r[side]['onset_05_in_exec_no_02_anywhere'] for r in extra_rows),
            onset_05_exec_close_02_tail=sum(r[side]['onset_05_exec_close_02_tail'] for r in extra_rows),
            seeds_with_any_dispatched_onset=sum(r[side]['dispatched_onset_runs_le_05'] > 0 for r in extra_rows),
            total_dispatched_onset_runs=sum(r[side]['dispatched_onset_runs_le_05'] for r in extra_rows))
    out = verification / 'closure_deferral_v41_close05_sensitivity.json'
    require(not out.exists(), 'preserve existing sensitivity receipt')
    value = dict(schema='closure-onset-0.5-sensitivity-v41',
                 time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                 thresholds=dict(onset_le=CLOSE_ONSET, audit_close_le=CLOSE_AUDIT,
                                 executed_prefix=EXEC, chunk_horizon=HORIZON),
                 note='Supplementary receipt; the primary deliverables remain closure_deferral_v41.json/.csv/.svg with threshold 0.2.',
                 sources=sources, per_seed=per_seed, aggregate_extra20=aggregate,
                 analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with out.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(out=str(out.resolve()), aggregate=aggregate), ensure_ascii=False))


if __name__ == '__main__':
    main()
