#!/usr/bin/env python3
"""Bounded sequential one-scene-per-process expert collection, never test policies."""
import os,sys,json,time,subprocess,argparse,signal,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gpu_lock import owner
from harness.robotwin_policy_worker import atomic_json
STOP=False
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=100);p.add_argument('--max-candidates',type=int,default=300);p.add_argument('--gpu-lock-owner',required=True);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);assert owner()==a.gpu_lock_owner
    state={'status':'running','task':'pick_apple_messy','backend':'robotwin1','requested_episodes':a.episodes,'max_candidates':a.max_candidates,'accepted':[],'attempts':[],'current_seed':None,'policy_testing':False,'started_unix':time.time()}
    def save():atomic_json(out/'status.json',state)
    def stop(sig,frame):
        global STOP;STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save();failures=0
    try:
        for seed in range(a.max_candidates):
            if STOP:raise InterruptedError('user stopped collection')
            if len(state['accepted'])>=a.episodes:break
            assert owner()==a.gpu_lock_owner,'GPU ownership changed'
            mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:'))}
            if 1-mem['MemAvailable']/mem['MemTotal']>=.85:raise RuntimeError('RAM guard >=85%')
            ep=len(state['accepted']);trial=out/f'seed_{seed:06d}';state['current_seed']=seed;save()
            cmd=[str(ROOT/'.venvs/robotwin-sim/bin/python'),'-u','-B',str(ROOT/'scripts/collect_robotwin1_apple.py'),'--out',str(trial),'--seed',str(seed),'--episode-id',str(ep),'--gpu-lock-owner',a.gpu_lock_owner]
            with (out/f'seed_{seed:06d}.log').open('x') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.monotonic()
                try:
                    while proc.poll() is None:
                        if STOP or time.monotonic()-start>240:
                            os.killpg(proc.pid,signal.SIGTERM)
                            try:proc.wait(timeout=15)
                            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                            raise TimeoutError('collection candidate timeout or operator stop')
                        time.sleep(.5)
                finally:
                    if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=15)
            try:r=json.loads((trial/'result.json').read_text())
            except (OSError,ValueError):r={'status':'infrastructure_incomplete'}
            row={'seed':seed,'episode_id':ep,'run_dir':str(trial),'returncode':proc.returncode,'result_status':r['status']}
            state['attempts'].append(row)
            if proc.returncode==0 and r['status']=='accepted':
                state['accepted'].append({**row,'hdf5':str(trial/f'episode_{ep:07d}.hdf5'),'sha256':r['sha256'],'frames':r['frames'],'first_frame_visibility':r['first_frame_visibility']});failures=0
            elif r['status']=='rejected_expert_or_visibility':failures=0
            else:failures+=1
            save();print(json.dumps({'seed':seed,'accepted':len(state['accepted']),'target':a.episodes,'status':r['status']}),flush=True)
            if failures>=3:raise RuntimeError('three consecutive infrastructure failures')
        if len(state['accepted'])!=a.episodes:raise RuntimeError('candidate budget exhausted before requested valid expert episodes')
        state['status']='complete';state['current_seed']=None;state['completed_unix']=time.time();save();return 0
    except BaseException as e:
        state.update(status='stopped' if STOP else 'failed',error=f'{type(e).__name__}: {e}',finished_unix=time.time());save();raise
if __name__=='__main__':raise SystemExit(main())
