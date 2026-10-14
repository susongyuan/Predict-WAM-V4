#!/usr/bin/env python3
"""Finalize new CPU preparation only; refuses after any execution ticket exists."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
OLD=BASE.parent/'task0_v3_25k_repeat_20260929'

def main():
    if list((BASE/'state').glob('*/seed*/pending.json')): raise SystemExit('Execution began; seal immutable')
    files=[p for p in (BASE/'runner').glob('*.py')]
    files += [BASE/'authorization_supplement.json',BASE/'assembly.json',BASE/'plan_snapshot/protocol.json',BASE/'plan_snapshot/preflight.json']
    for variant in ('original','improved'):
        image=BASE/'images'/variant
        files += list((image/'runner').glob('*.py'))+list((image/'runner').glob('*.json'))
        files += list((image/'frozen_runtime/.conda-envs').glob('*/bin/python'))
    files += [OLD/'runner/repeat.py',OLD/'runner/bootstrap.py',OLD/'runner/freeze_inventory.json']
    files += [Path(p) for p in json.loads((BASE/'authorization_supplement.json').read_text())['bound_files']]
    mapping={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    path=BASE/'runner/code_integrity.json'
    if path.exists():
        before=path.read_bytes()
        archive=BASE/'cpu'/'seal_history'/f'{hashlib.sha256(before).hexdigest()}.json'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists(): archive.write_bytes(before)
    path.write_text(json.dumps(mapping,indent=2)+'\n')
    print(json.dumps({'sealed_small_files':len(mapping),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))

if __name__=='__main__': main()
