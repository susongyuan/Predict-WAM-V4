#!/usr/bin/env python3
"""Read-only wall occupancy sample of an already-running evaluator journal."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness.robotwin_policy_worker import atomic_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-dir',required=True);p.add_argument('--out',required=True)
p.add_argument('--seconds',type=float,default=30);args=p.parse_args()
if not 1<=args.seconds<=60:raise ValueError('bounded read-only sampling window')
path=Path(args.run_dir)/'live_state.json';out=Path(args.out)
if out.exists():raise ValueError('refuse existing sample receipt')
counts=Counter();sequences=set();steps=set();start=time.monotonic()
while time.monotonic()-start<args.seconds:
    try:
        state=json.loads(path.read_text());counts[(state.get('phase'),state.get('status'))]+=1
        sequences.add(state.get('sequence'));steps.add(state.get('step'))
    except (OSError,ValueError):counts[('read_unavailable','unknown')]+=1
    time.sleep(.025)
result=dict(elapsed_seconds=time.monotonic()-start,samples=sum(counts.values()),
            unique_sequences=len(sequences),observed_steps=sorted(s for s in steps if type(s)is int),
            occupancy=[dict(phase=k[0],status=k[1],samples=v,fraction=v/sum(counts.values()))
                       for k,v in counts.most_common()],
            source=str(path.resolve()),read_only=True,
            limitation='Approximate wall occupancy of the last journal phase, not exclusive call timings or CPU/GPU profiling. Completed state can include unjournaled work until next call. No model, simulator, protocol or running process was changed.')
atomic_json(out,result);print(json.dumps(result,indent=2))
