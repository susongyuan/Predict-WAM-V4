#!/usr/bin/env python3
"""Physical repair probes on preserved failed seeds; never count toward quota."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import collect_isaac_zhuhai_recycle as c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',required=True,type=int);ap.add_argument('--source-parent',required=True,type=int);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--gpu-lock-owner',required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/zhuhai2026_tasks12/production_20261009_v1/zhuhai_task2/source/manifest.json'
    supplied=json.loads(source.read_text());parent=next(x for x in supplied['attempts'] if x['seed']==args.source_parent and x['status']=='success')
    if c.sha256(parent['data_path'])!=parent['data_sha256']:raise ValueError('Parent data changed')
    config=c.parser().parse_args(['--task','zhuhai_task2','--mode','randomized','--count','100','--seed-start',str(args.seed),'--out',str(args.out),'--source-manifest',str(source),'--gpu-lock-owner',args.gpu_lock_owner])
    contract=c.contract(config);contract.update(source_probe=True,collection_scope='expert_repair_probe_not_production',expert_variant=dict(path=str(ROOT/'harness/isaac_zhuhai_task2_expert_repair_v2.py'),sha256=c.sha256(ROOT/'harness/isaac_zhuhai_task2_expert_repair_v2.py')),probe_implementation=dict(path=str(Path(__file__).resolve()),sha256=c.sha256(__file__)))
    c.write_once(args.out/'plan.json',dict(contract=contract,failed_seed=args.seed,source_parent=args.source_parent,admitted_to_production=False))
    from harness.isaac_zhuhai_task2_expert_repair_v2 import SmoothExpert
    from harness import isaac_zhuhai_expert as original
    original.SmoothExpert=SmoothExpert
    from harness import isaac_zhuhai_runtime as runtime
    class ProbeEnv(runtime.IsaacZhuhaiEnv):
        def close(self):
            self._clear_scene()
            manifest=args.out/'manifest.json';status=json.loads(manifest.read_text())['status'] if manifest.exists() else 'exception'
            code=0 if status=='complete' else 2
            if self.app is not None:self.app.close(exit_code=code);self.app=None
    runtime.IsaacZhuhaiEnv=ProbeEnv
    config.count=1;config.max_attempts=1;config.max_consecutive_failures=3
    return c.collect(config,contract,[parent])
if __name__=='__main__':raise SystemExit(main())
