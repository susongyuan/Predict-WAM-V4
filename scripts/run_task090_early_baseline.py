#!/usr/bin/env python3
"""Serial existing-environment launcher; no installation, alternate inputs or retries."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_task090_early_baseline import RUN_ROOT, sha, write_json
SCRIPT = ROOT/'scripts/check_task090_early_baseline.py'
TEST = ROOT/'scripts/test_task090_early_baseline_entry.py'
PYTHONS = {phase:ROOT/f'.conda-envs/{env}/bin/python' for phase,env in
           (('prepare','behavior'),('features','lilawam'),('track','behavior'))}


def launch(command, out, rss_gib):
    """Monitor only our live child; parent and all workloads are serial."""
    started=time.monotonic()
    peak_rss=peak_ram=peak_cpu=0.
    stopped=None
    psutil.cpu_percent(interval=None)
    with (out/'launcher.log').open('x') as log:
        child=subprocess.Popen([str(x) for x in command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                peak_rss=max(peak_rss,proc.memory_info().rss)
                peak_ram=max(peak_ram,psutil.virtual_memory().percent)
                peak_cpu=max(peak_cpu,psutil.cpu_percent(interval=None))
                if peak_rss >= rss_gib*1024**3 or peak_ram >= 85 or peak_cpu >= 85 or time.monotonic()-started >= 900:
                    stopped='declared CPU/RAM/RSS/wall guard'
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait()
                    break
            except psutil.NoSuchProcess:
                break
            try:
                child.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
        rc=child.wait()
    record=dict(command=list(map(str,command)),pid=child.pid,parent_pid=os.getpid(),returncode=rc,
        exited=True,wall_seconds=time.monotonic()-started,peak_sampled_rss_mib=peak_rss/1024**2,
        peak_sampled_system_cpu_percent=peak_cpu,peak_sampled_system_ram_percent=peak_ram,
        guard_stop=stopped,verified=False,admitted=False)
    write_json(out/'process_exit.json',record)
    print(json.dumps(record),flush=True)
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--preflight',type=Path,help='reuse a completed check-only launcher directory')
    ap.add_argument('--prepared',type=Path,help='reuse completed exact prepared frames without decoding again')
    ap.add_argument('--features',type=Path,help='reuse completed exact CPU feature without another forward')
    args=ap.parse_args()
    out=args.out.resolve()
    if RUN_ROOT.resolve() not in out.parents:
        ap.error('--out must be a new directory inside early_baseline_run02')
    if args.features is not None and args.prepared is None:
        ap.error('--features requires its matching --prepared')
    out.mkdir(parents=True,exist_ok=False)
    process=psutil.Process()
    process.cpu_affinity([min(process.cpu_affinity())])
    started=time.monotonic()
    records=[]
    phase_outputs=[]
    failure=None
    write_json(out/'plan.json',dict(driver_pid=os.getpid(),source_sha256={str(p):sha(p) for p in (SCRIPT,TEST,Path(__file__))},
        python_paths={p:str(v) for p,v in PYTHONS.items()},forward_budget=1,cpu_threads=1,cpu_ram_limit_percent=85,
        rss_limit_gib={'prepare':2,'features':6,'track':2},check_only=args.check_only,
        reuse_prepared=None if args.prepared is None else str(args.prepared),
        reuse_features=None if args.features is None else str(args.features),verified=False,admitted=False))
    try:
        preflight=args.preflight
        if preflight is None:
            test_out=out/'cpu_tests';test_out.mkdir()
            rec=launch([PYTHONS['prepare'],'-B',TEST],test_out,2);records.append(rec)
            assert rec['returncode']==0 and not rec['guard_stop'], 'CPU entry tests failed'
            preflight=out/'preflight'
            preflight.mkdir()
            for phase in PYTHONS:
                monitor=out/('preflight_'+phase);monitor.mkdir()
                dest=preflight/phase
                rec=launch([PYTHONS[phase],'-B',SCRIPT,'--stage',phase,'--out',dest,'--check-only'],monitor,6 if phase=='features' else 2)
                records.append(rec);phase_outputs.append(dest)
                assert rec['returncode']==0 and not rec['guard_stop'], 'preflight failed: '+phase
        else:
            # The workers additionally pin all three check results to their exact source SHA.
            prior=json.loads((preflight.parent/'result.json').read_text())
            assert prior['status']=='checks_complete_no_workload' and prior['counts']=={}
        if not args.check_only:
            prepared,features=args.prepared,args.features
            for phase in PYTHONS:
                if (phase=='prepare' and prepared is not None) or (phase=='features' and features is not None):
                    continue
                monitor=out/(phase+'_process');monitor.mkdir()
                dest=out/phase
                cmd=[PYTHONS[phase],'-B',SCRIPT,'--stage',phase,'--out',dest,'--preflight',preflight]
                if phase in ('features','track'):
                    cmd+=['--prepared',prepared]
                if phase=='track':
                    cmd+=['--features',features]
                rec=launch(cmd,monitor,6 if phase=='features' else 2)
                records.append(rec);phase_outputs.append(dest)
                assert rec['returncode']==0 and not rec['guard_stop'], 'stage failed: '+phase
                if phase=='prepare':prepared=dest
                if phase=='features':features=dest
    except BaseException:
        failure=traceback.format_exc()
    counts=Counter()
    stage_resources=[]
    for dest in phase_outputs:
        path=dest/'stage_result.json'
        if not path.exists():path=dest/'failure.json'
        if path.exists():
            report=json.loads(path.read_text())
            counts.update(report['resources']['counts'])
            stage_resources.append(dict(path=str(path),**report['resources']))
    own=resource.getrusage(resource.RUSAGE_SELF)
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    write_json(out/'result.json',dict(status='failed_preserved_no_retry' if failure else
        ('checks_complete_no_workload' if args.check_only else 'completed_diagnostic_not_admission'),
        driver_pid=os.getpid(),failure=failure,counts=dict(counts),processes=records,stage_resources=stage_resources,
        wall_seconds=time.monotonic()-started,cpu_seconds_parent_and_children=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,
        peak_rss_mib_max_process=max(own.ru_maxrss,children.ru_maxrss)/1024,
        resource_semantics='whole launcher invocation including checks; serial peak is max, not sum; reused work not counted again',
        prior_preflight=None if args.preflight is None else str(args.preflight),
        preflight=str(preflight),verified=False,admitted=False,gpu_used=False))
    return 1 if failure else 0


if __name__=='__main__':
    sys.exit(main())
