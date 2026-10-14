#!/usr/bin/env python3
"""Summarize the diagnostic-advance batch: what the program did, and what the EVALUATOR independently scored.

This reads evaluator-privileged object poses for DIAGNOSIS ONLY. Nothing here is policy input, and a
diagnostic_advance is never a verified success.
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'outputs/behavior/closed_loop'


def trace_events(run):
    rows = []
    for path in sorted(run.glob('policy_trace*.jsonl')):
        for line in path.open():
            try: rows.append(json.loads(line))
            except ValueError: pass
    return rows


def bin_motion(run):
    files = sorted((run / 'eval/object_poses').glob('*.jsonl'))
    if not files:
        return None
    recs = []
    for line in files[0].open():
        d = json.loads(line)
        if d.get('type') in ('header', 'end'):
            continue
        recs.append(d)
    if not recs:
        return None
    steps = np.array([r['step'] for r in recs])
    key = next((k for k in recs[0]['objects'] if 'ashcan' in k or 'trash' in k), None)
    if key is None:
        return None
    az = np.array([r['objects'][key][2] for r in recs])
    aq = np.array([r['objects'][key][3:] for r in recs]); q0 = aq[0]
    rot = np.array([2*np.degrees(np.arccos(np.clip(abs(np.dot(q/np.linalg.norm(q), q0)), -1, 1))) for q in aq])
    base = float(recs[0]['manipulation']['assisted_grasp'].get('aabb_min_z') or 0.0)
    minz, active = [], []
    for r in recs:
        g = r['manipulation']['assisted_grasp']['right']
        z = g.get('aabb_min_z')
        minz.append(np.nan if z is None else float(z))
        active.append(bool(g.get('constraint_active')))
    minz = np.array(minz); active = np.array(active)
    best = cur = 0
    for i in range(len(minz)):
        cur = cur + 1 if (active[i] and np.isfinite(minz[i]) and minz[i] >= base + 0.03) else 0
        best = max(best, cur)
    return dict(steps=int(steps.max()), baseline_min_z=base,
                peak_min_z=float(np.nanmax(minz)) if np.isfinite(minz).any() else None,
                peak_rot_deg=float(rot.max()), final_rot_deg=float(rot[-1]),
                longest_assisted_lift_steps=int(best),
                constraint_active_steps=int(active.sum()))


def summarize(batch: Path):
    out = []
    for launch in sorted(batch.glob('seed*_launch.json')):
        argv = json.loads(launch.read_text())['argv']
        name = argv[argv.index('--run-name') + 1]
        seed = int(argv[argv.index('--seed') + 1])
        run = RUNS / name
        row = {'seed': seed, 'run': name, 'run_dir_exists': run.is_dir()}
        if not run.is_dir():
            out.append(row); continue
        ev = trace_events(run)
        row['contracts'] = [{'event': e['event'], 'step': e.get('step'), 'key': e.get('key'),
                             'steps': e.get('steps'), 'success': e.get('success'),
                             'predicate': e.get('predicate')}
                            for e in ev if e.get('event') in
                            ('orch_contract_start', 'orch_contract_end', 'orch_hold', 'orch_diagnostic_advance')]
        row['n_diagnostic_advance'] = sum(1 for e in ev if e.get('event') == 'orch_diagnostic_advance')
        js = sorted((run / 'eval/json').glob('*.json'))
        if js:
            sc = json.loads(js[0].read_text())
            row['evaluator'] = {k: sc.get(k) for k in ('success', 'steps', 'truncated', 'terminated')}
            row['evaluator']['q_score'] = (sc.get('q_score') or {}).get('final')
        row['bin'] = bin_motion(run)
        out.append(row)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True, type=Path)
    ap.add_argument('--json-out', type=Path, default=None)
    a = ap.parse_args()
    rows = summarize(a.batch)
    print(json.dumps(rows, indent=1, ensure_ascii=False))
    if a.json_out:
        a.json_out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
