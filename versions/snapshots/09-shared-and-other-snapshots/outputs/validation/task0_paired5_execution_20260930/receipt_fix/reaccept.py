#!/usr/bin/env python3
"""Targeted CPU reacceptance after the completed-receipt fix; no GPU calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'receipt_fix'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = json.loads((OUT / 'source_before/acceptance.json').read_text())
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
results = []
for name, command in [
    ('safety_reaccept', [sys.executable, '-B', str(BASE / 'runner/test_cpu.py')]),
    ('original_reaccept', [sys.executable, '-B', str(BASE / 'runner/paired5.py'), '--variant', 'original', '--seeds', '1', '--dry-run']),
    ('improved_reaccept', [sys.executable, '-B', str(BASE / 'runner/paired5.py'), '--variant', 'improved', '--seeds', '0', '--dry-run']),
]:
    started = time.monotonic()
    result = subprocess.run(command, env=env, cwd=BASE, capture_output=True, text=True, timeout=180)
    log = OUT / (name + '.log')
    with log.open('x') as f:
        f.write(result.stdout + result.stderr)
    if result.returncode:
        raise SystemExit(f'{name} failed: {log}')
    if 'reaccept' in name and name != 'safety_reaccept':
        payload = json.loads(result.stdout)
        assert payload['gpu_or_simulator_started'] is False
        assert payload['code_integrity_sha256'] == sha(BASE / 'runner/code_integrity.json')
    results.append({'name': name, 'returncode': 0, 'seconds': time.monotonic()-started,
                    'log': str(log), 'sha256': sha(log)})
old.update(code_integrity_sha256=sha(BASE / 'runner/code_integrity.json'), runner_safety_tests=23,
           results=results, receipt_reuse_fix_cpu_accepted=True,
           previous_acceptance=str(OUT / 'source_before/acceptance.json'),
           reviewer_scope='Main-session targeted CPU reacceptance after independent finding; no new physical trial or claim of post-fix independent review.')
with (OUT / 'reacceptance.json').open('x') as f:
    json.dump(old, f, indent=2)
(BASE / 'cpu/acceptance.json').write_text(json.dumps(old, indent=2)+'\n')
print(json.dumps({'cpu_accepted': True, 'gpu_started': False, 'results': results,
                  'code_integrity_sha256': old['code_integrity_sha256']}, indent=2))
