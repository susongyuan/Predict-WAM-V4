#!/usr/bin/env python3
"""Sequential 2/3/4-worker capacity trials with complete physical probe episodes.

Uses the frozen collector and separate probe outputs. Does not edit production
contracts, close applications, clear caches, or count probes toward quotas.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.collect_isaac_zhuhai import write_once, validate_manifest, sha256
from scripts.run_zhuhai_pipeline import atomic, verify_plan_sources
from scripts.zhuhai_resource_watch import memory_info, gpu_info
from scripts.gpu_lock import owner

GIB=1024**3


def proc_memory(pid):
    try:
        result={}
        for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
            if ':' in line:
                key,value=line.split(':',1); parts=value.split()
                if parts and parts[0].isdigit():result[key]=int(parts[0])*1024
        stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return dict(pid=pid,rss_bytes=result.get('Rss',0),pss_bytes=result.get('Pss',0),
                    swap_pss_bytes=result.get('SwapPss',0),cpu_ticks=int(stat[11])+int(stat[12]),state=stat[0])
    except (OSError,ValueError):return dict(pid=pid,exited=True)


def vm_counters():
    values={line.split()[0]:int(line.split()[1]) for line in Path('/proc/vmstat').read_text().splitlines()}
    pressure={}
    for kind in ('memory','io','cpu'):
        rows={}
        for line in Path(f'/proc/pressure/{kind}').read_text().splitlines():
            parts=line.split();rows[parts[0]]={k:float(v) for k,v in (p.split('=') for p in parts[1:])}
        pressure[kind]=rows
    return dict(swap_in_pages=values['pswpin'],swap_out_pages=values['pswpout'],pressure=pressure)


def production_status(source_root):
    result={}
    for i in (1,2):
        task=f'zhuhai_task{i}'
        m=json.loads((source_root/task/'source/manifest.json').read_text())
        successes=[r for r in m['attempts'] if r['status']=='success']
        result[task]=dict(status=m['status'],success_count=m['success_count'],
            recent_completed=[dict(seed=r['seed'],started_unix=r['started_unix'],finished_unix=r['finished_unix'],
                wall_seconds=r['finished_unix']-r['started_unix'],physics_seconds=r['accepted_physics_times'][-1]-r['accepted_physics_times'][0])
                for r in successes[-12:]])
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--gpu-lock-owner',required=True)
    ap.add_argument('--original-pids',nargs=2,type=int,required=True)
    args=ap.parse_args();args.out=args.out.resolve();args.source_root=args.source_root.resolve()
    if owner()!=args.gpu_lock_owner:raise RuntimeError('Existing production owner required')
    verify_plan_sources(json.loads((args.source_root/'plan.json').read_text()))
    for pid in args.original_pids:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if str(ROOT/'scripts/collect_isaac_zhuhai.py') not in command or str(args.source_root) not in ' '.join(command):
            raise RuntimeError('Original worker identity mismatch')
    state=dict(status='baseline_2_workers',controller_pid=os.getpid(),started_unix=time.time(),phases=[])
    children=[];latest_gpu={};gpu_at=0;previous=None
    minimum_ram=5.25*GIB
    # Original immutable collectors pause at 85% swap or 85% RAM. These
    # absolute margins protect those writers; no guard uses swap *increase*.
    maximum_swap_fraction=.82
    def persist():atomic(args.out/'status.json',state)
    def sample(phase):
        nonlocal latest_gpu,gpu_at,previous
        now=time.time();ram=memory_info();vm=vm_counters()
        if now-gpu_at>=5:latest_gpu=gpu_info();gpu_at=now
        processes=[proc_memory(pid) for pid in args.original_pids]+[proc_memory(c['proc'].pid) for c in children if c['proc'].poll() is None]
        value=dict(time_unix=now,phase=phase,ram=ram,gpu=latest_gpu,vm=vm,processes=processes)
        if previous:
            dt=now-previous['time_unix'];page=os.sysconf('SC_PAGE_SIZE')
            value['swap_in_bytes_per_second']=(vm['swap_in_pages']-previous['vm']['swap_in_pages'])*page/dt
            value['swap_out_bytes_per_second']=(vm['swap_out_pages']-previous['vm']['swap_out_pages'])*page/dt
        previous=value
        with (args.out/'resource_samples.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        state.update(last_sample=value,checked_unix=now);persist()
        reasons=[]
        if ram['available_bytes']<minimum_ram:reasons.append('available_RAM_below_5.25GiB_before_original_85pct_pause')
        if ram['swap_total_bytes'] and ram['swap_used_bytes']/ram['swap_total_bytes']>=maximum_swap_fraction:
            reasons.append('absolute_swap_82pct_before_original_85pct_pause')
        for device in latest_gpu.get('devices',[]):
            if device['total_bytes']-device['used_bytes']<GIB:reasons.append('GPU_free_below_1GiB')
        for row in processes[:2]:
            if row.get('exited') or row.get('state') in ('T','Z'):reasons.append('original_worker_not_running')
        return value,reasons
    def launch(name,task,seed):
        target=args.out/name;target.mkdir(exist_ok=False)
        command=[str(ROOT/'.conda-envs/behavior6/bin/python'),'-B',str(ROOT/'scripts/collect_isaac_zhuhai.py'),
            '--task',task,'--mode','source','--count','1','--source-probe','--max-attempts','1','--seed-start',str(seed),
            '--out',str(target/'probe'),'--gpu-lock-owner',args.gpu_lock_owner]
        env=os.environ.copy();env.update(OMNI_KIT_ACCEPT_EULA='YES',LD_PRELOAD=str(ROOT/'.conda-envs/behavior6/lib/libstdc++.so.6'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('CUDA_VISIBLE_DEVICES',None)
        with (target/'process.log').open('x') as f:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        child=dict(name=name,task=task,seed=seed,target=target,proc=proc,started_unix=time.time())
        children.append(child)
        write_once(target/'launch.json',dict(pid=proc.pid,command=command,started_unix=child['started_unix'],scope='capacity_probe_not_formal_data',record_video=False))
        return child
    def stop_trials(reason):
        for child in reversed(children):
            proc=child['proc']
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM);child['stop_reason']=reason
        for child in children:
            child['proc'].wait()
    def outcome(child):
        proc=child['proc'];entry=dict(name=child['name'],pid=proc.pid,exit_code=proc.returncode,
            wall_seconds=time.time()-child['started_unix'],stop_reason=child.get('stop_reason'),success=False)
        path=child['target']/'probe/manifest.json'
        if path.exists():
            m=json.loads(path.read_text());entry.update(manifest_status=m['status'],successful_episodes=m.get('success_count',0))
            if proc.returncode==0:
                rows=validate_manifest(path,task=child['task'],mode='source',count=1)
                row=rows[0];entry.update(success=True,physics_seconds=row['accepted_physics_times'][-1]-row['accepted_physics_times'][0],
                    recorded_episode_wall_seconds=row['finished_unix']-row['started_unix'],observation_count=row['observation_count'],manifest_sha256=sha256(path))
        write_once(child['target']/'result.json',entry)
        return entry
    def run_phase(label,phase_children,spawn_fourth=False):
        phase=dict(name=label,started_unix=time.time(),initial_production=production_status(args.source_root),guard_reasons=[],samples=0)
        state['phases'].append(phase);state['status']=label;persist()
        deadline=time.monotonic()+900;fourth_started=False;ready_at=None
        while True:
            value,reasons=sample(label);phase['samples']+=1
            if reasons:
                phase['guard_reasons']=reasons;stop_trials(';'.join(reasons));break
            if time.monotonic()>deadline:
                phase['guard_reasons']=['phase_900s_timeout'];stop_trials('phase_900s_timeout');break
            if spawn_fourth and not fourth_started:
                first=phase_children[0]
                # A growing actual onboard HDF means the third Isaac process
                # has finished initialization and entered physical collection.
                ready=any((first['target']/'probe/data').glob('*.partial'))
                if ready and ready_at is None:ready_at=time.monotonic()
                if first['proc'].poll() is not None:
                    phase['guard_reasons']=['third_restart_exited_before_fourth'];break
                if ready_at and time.monotonic()-ready_at>=15:
                    phase_children.append(launch('four_worker_probe4','zhuhai_task2',964302));fourth_started=True
            if all(c['proc'].poll() is not None for c in phase_children) and (not spawn_fourth or fourth_started):break
            time.sleep(1)
        phase.update(finished_unix=time.time(),final_production=production_status(args.source_root),
                     outcomes=[outcome(c) for c in phase_children],fourth_started=fourth_started)
        persist();return not phase['guard_reasons'] and all(x['success'] for x in phase['outcomes'])
    write_once(args.out/'intent.json',dict(schema='zhuhai-worker-scaling-v2',source_root=str(args.source_root),
        original_worker_pids=args.original_pids,gpu_lock_owner=args.gpu_lock_owner,source_quotas_modified=False,
        execution_code_sha256=sha256(__file__),record_video=False,third_view=False,
        guards=dict(minimum_available_ram_bytes=minimum_ram,maximum_absolute_swap_fraction=maximum_swap_fraction,
                    minimum_GPU_free_bytes=GIB,no_swap_delta_guard=True,reason='margin before frozen formal collector resource pause'),
        startup_and_complete_episode=True))
    try:
        state['baseline_production']=production_status(args.source_root)
        for _ in range(20):sample('baseline_2_workers');time.sleep(1)
        third=launch('three_worker_probe3','zhuhai_task2',964201)
        passed3=run_phase('three_workers',[third])
        state['three_workers_passed']=passed3;persist()
        if passed3:
            # The first probe has exited; restart worker3 and only then add
            # worker4, so this actually tests four simultaneous Isaac workers.
            restart=launch('four_worker_probe3','zhuhai_task1',964301)
            state['four_workers_passed']=run_phase('four_workers',[restart],spawn_fourth=True)
        else:state['four_workers_passed']=None
        state.update(status='completed',finished_unix=time.time(),final_production=production_status(args.source_root))
        persist();write_once(args.out/'result.json',state)
        print(json.dumps(dict(status=state['status'],three_workers_passed=state['three_workers_passed'],four_workers_passed=state['four_workers_passed'])),flush=True)
        return 0
    except BaseException as exc:
        stop_trials(f'{type(exc).__name__}: {exc}')
        state.update(status='failed_controller',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());persist()
        raise


if __name__=='__main__':raise SystemExit(main())
