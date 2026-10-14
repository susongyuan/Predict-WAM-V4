#!/usr/bin/env python3
"""Infrastructure repair of the original twenty assigned seeds, never resampling."""
import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_dustbin_pure20 as baseline
from scripts import run_robotwin_checkpoint_queue as queue
from scripts.run_robotwin_posttrain_eval_pipeline import training_gate
from harness.robotwin_policy_worker import atomic_json, identity, sha256, training_contract

OLD = ROOT / 'outputs/runs/put_bottles_dustbin/pure25k_fresh20_20261007'
OUT = ROOT / 'outputs/runs/put_bottles_dustbin/pure25k_fixed20_offscreen_repair_20261007'


def build(out):
    old = baseline.read(OLD / 'manifest.json')
    state = baseline.read(OLD / 'queue_status.json')
    admitted = baseline.read(OLD / 'training_admission.json')
    if (old.get('seeds') != baseline.SEEDS or state.get('status') != 'stopped'
            or any(r['status'] == 'valid_terminal' for r in state.get('jobs', []))):
        raise ValueError('repair requires the original stopped, unscored assignment')
    unit = queue.unit_state('p2-dustbin-pure20-suite-20261007')
    if unit.get('MainPID') != '0' or unit.get('SubState') == 'running':
        raise ValueError('original parent still running')
    ready, gate = training_gate(baseline.TRAIN, baseline.TRAIN_UNIT)
    if not ready:
        raise ValueError('training/GPU gate not ready: ' + str(gate))
    contract, cache, _, receipt = training_contract(baseline.TRAIN, baseline.CKPT)
    if (receipt != old['checkpoint_receipt'] or receipt != admitted['checkpoint_receipt']
            or sha256(baseline.CKPT) != receipt['sha256']
            or contract['config']['dataset']['task_name'] != baseline.TASK
            or contract['config_sha256'] != sha256(baseline.CONFIG)):
        raise ValueError('original frozen model/config differs')
    dataset_seeds, _ = baseline.dataset_seed_inventory(cache)
    if set(baseline.SEEDS) & (dataset_seeds | set(admitted['excluded_seeds'])):
        raise ValueError('original assignment overlaps preregistered exclusions')
    if len(old['jobs']) != 20 or [j['seed'] for j in old['jobs']] != baseline.SEEDS:
        raise ValueError('original job order changed')
    jobs = []
    for j in old['jobs']:
        if (j['system_mode'] != 'pure' or j['checkpoint_step'] != 25000
                or j['max_steps'] != 1700 or j['execution_mode'] != 'native8'
                or j['camera_readback'] != 'serial' or j['settle_steps'] != 0
                or not j['record_onboard']):
            raise ValueError('original pure protocol mismatch')
        jobs.append(dict(j, unit=f"p2-dustbin-pure20-offscreen-{j['seed']}-20261007",
                         run_dir=str(out / f"seed_{j['seed']}"),
                         evaluator_entry='scripts/eval_robotwin_offscreen.py'))
    # Conservative freeze of existing harness, simulator and evaluator sources.
    # New, separately imported full-system providers can be developed in parallel.
    sources = [Path(p) for p in old['source_sha256']]
    sources += [Path(__file__), ROOT / 'scripts/eval_robotwin_offscreen.py']
    source_hashes = {str(p): sha256(p) for p in sources}
    changed_sources = {p: dict(original_sha256=h, repair_sha256=source_hashes[p])
                       for p, h in old['source_sha256'].items() if source_hashes[p] != h}
    manifest = dict(old, protocol='dustbin_pure25k_fixed20_offscreen_repair_v2',
                    created_at=baseline.now(), jobs=jobs, source_sha256=source_hashes,
                    repair_of=identity(OLD / 'manifest.json'),
                    repair_queue_evidence=identity(OLD / 'queue_status.json'),
                    original_attempts=state['jobs'],
                    source_changes_since_original=changed_sources,
                    protocol_deviation='Original first episode wrote native failure but exited 1 with X11 BadWindow; second was canceled for diagnosis. Neither is scored. All twenty original seeds are rerun once after removing display variables before simulator imports. No score-selected seed/attempt replacement. Source changes are recorded individually: task/count arguments in conditional VTT paths, non-pure VTT/bridge definitions, and evaluator-entry selection in queue. Pure-WAM model, normalization, renderer, RGB preprocessing, action sampling, execution horizon, budget and native scoring are unchanged; post-repair action evidence is verified separately.',
                    baseline_freeze_scope='Original 155-source archive plus repair runner/offscreen wrapper. Separate newly added full-system files are outside pure runtime; shared frozen files may not change during this queue.')
    evidence_paths = [OLD / 'completion.json']
    for attempt in state['jobs']:
        directory = Path(attempt['run_dir'])
        evidence_paths += [OLD / (attempt['unit'] + '_receipt.json')]
        evidence_paths += list(directory.glob('result.json')) + list(directory.glob('*/actions.jsonl'))
    manifest['original_attempt_evidence'] = [identity(p) for p in evidence_paths if p.is_file()]
    manifest['attempt_selection'] = 'Exactly one valid terminal per original seed; reject duplicate valid attempts. Never replay valid success or failure. One same-seed init retry; no further infrastructure repair within this suite.'
    manifest['estimated_wall_hours'] = [3, 8]
    admitted = dict(admitted, repair_training_gate=gate,
                    repaired_at=baseline.now(), original_admission=identity(OLD / 'training_admission.json'))
    queue.validate_manifest(manifest)
    return manifest, admitted


def summarize(out, manifest):
    state = baseline.read(out / 'queue_status.json')
    for seed in manifest['seeds']:
        if sum(r['seed'] == seed and r['status'] == 'valid_terminal' for r in state.get('jobs', [])) > 1:
            raise ValueError('duplicate valid terminal for seed ' + str(seed))
    result = baseline.summarize(out, manifest)
    result.update(assignment='original fixed twenty seeds; infrastructure repair',
                  original_archive=str(OLD), original_scored_episodes=0)
    atomic_json(out / 'summary.json', result)
    report = out / 'report.md'
    text = report.read_text().replace('25k纯WAM：20个全新种子', '25k纯WAM：原定20种子离屏修复续跑')
    text += '\n原队列原封保留：921001跑满1700步，原生结果false，但X11退出码1所以未评分；921002中断未评分。修复队列统一重跑原种子，不选择更好的结果。源码变更、原尝试SHA与修复偏差见manifest。\n'
    baseline.atomic_text(report, text)
    return result


def run(out):
    manifest, admitted = build(out)
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'manifest.json'
    atomic_json(path, manifest)
    atomic_json(out / 'training_admission.json', admitted)
    stop = threading.Event()
    errors = []
    def report():
        while not stop.wait(30):
            try:
                summarize(out, manifest)
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
    summarize(out, manifest)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    code, error = 1, None
    try:
        code = queue.run_manifest(manifest, path,
            before_job=lambda: baseline.verify_admission(manifest, admitted))
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        stop.set()
        thread.join(timeout=15)
        summary = summarize(out, manifest)
        changed = [p for p, h in manifest['source_sha256'].items() if sha256(p) != h]
        if changed:
            summary.update(verified_complete=False, source_changes_during_run=changed)
            atomic_json(out / 'summary.json', summary)
        if not summary['verified_complete']:
            code = 1
        atomic_json(out / 'completion.json', dict(exit_code=code, error=error,
            finished_at=baseline.now(), verified_complete=summary['verified_complete'],
            reporting_errors=errors, original_archive=str(OLD)))
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    baseline.bus_environment()
    if args.execute:
        raise SystemExit(run(OUT))
    manifest, _ = build(OUT)
    print(json.dumps(dict(protocol=manifest['protocol'], seeds=manifest['seeds'],
                         archived_original_attempts=len(manifest['original_attempts']),
                         evaluator_entry=manifest['jobs'][0]['evaluator_entry']), indent=2))
