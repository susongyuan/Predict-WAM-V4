#!/usr/bin/env python3
"""Read-only event bridge for the existing Task0 stage monitor; no GPU or control actions."""
import json
from pathlib import Path
import time

ROOT = Path('/home/user/Robotic/project2')
STAGES = ROOT / 'outputs/training/task0_v3_30k_20260929_stages'
last = None
last_error = None
while True:
    try:
        data = json.loads((STAGES / 'monitor_status.json').read_text())
        age = time.time() - data['observed_at']
        train = data.get('training') or {}
        step = train.get('observed_step')
        best = data.get('best_task') or {}
        errors = list(data.get('errors') or [])
        controller = data.get('controller') or {}
        if age > 180:
            errors.append('monitor_snapshot_stale_over_180s')
        if controller.get('ActiveState') not in (None, 'active') and not data.get('terminal'):
            errors.append('controller_not_active')
        key = (data.get('phase'), data.get('status', {}).get('step_target'),
               step // 1000 if isinstance(step, int) else None,
               best.get('candidate_count'), best.get('has_successful_checkpoint'), tuple(errors))
        if key != last:
            print(json.dumps({'phase': data.get('phase'), 'step': step,
                              'target': data.get('status', {}).get('step_target'),
                              'completed_candidates': best.get('candidate_count'),
                              'has_successful_checkpoint': best.get('has_successful_checkpoint'),
                              'errors': errors, 'terminal': data.get('terminal'),
                              'budget_complete': data.get('budget_complete')}, ensure_ascii=False), flush=True)
            last = key
        if data.get('terminal'):
            break
        last_error = None
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if error != last_error:
            print(json.dumps({'event': 'monitor_read_error', 'error': error}), flush=True)
            last_error = error
    time.sleep(10)
