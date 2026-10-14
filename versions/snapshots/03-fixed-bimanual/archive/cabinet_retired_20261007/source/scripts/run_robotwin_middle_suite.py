"""Durable suite supervisor with mandatory real interface receipts and report."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.robotwin_policy_worker import atomic_json, sha256
from scripts.run_robotwin_checkpoint_queue import main as run_queue
from scripts.summarize_robotwin_middle import summarize

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args()
    path=Path(a.manifest).resolve();manifest=json.loads(path.read_text())
    for artifact in manifest['admission_receipts']:
        receipt=json.loads(Path(artifact['path']).read_text())
        if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('interface receipt changed')
        if receipt.get('status')!='completed':raise ValueError('real interface check did not complete')
    try:
        code=run_queue(['--manifest',str(path),'--execute'])
    finally:
        report=summarize(path)
        atomic_json(path.parent/'suite_finished.json',{'groups':report['groups'],
            'task_verifier_admitted':False,'all_planned_valid':all(g['valid']==g['planned'] for g in report['groups'].values())})
    return code
if __name__=='__main__':raise SystemExit(main())
