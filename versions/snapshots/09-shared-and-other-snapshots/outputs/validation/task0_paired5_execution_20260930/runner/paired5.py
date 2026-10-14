#!/usr/bin/env python3
"""Paired5 single-dispatch executor. Default CPU dry-run. --execute launches at most ONE new seed.
Completed seeds are reused; partial/error directories permanently block automatic relaunch.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/user/Robotic/project2')
OLD = PROJECT / 'outputs/validation/task0_v3_25k_repeat_20260929'
STATE = BASE / 'state'
from admission import authorize, AUTH_SHA, GPU_LOCK, require
_PERMIT = object()

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()
def stat(p):
    s=Path(p).stat()
    return {k:getattr(s,k) for k in ('st_size','st_mtime_ns','st_ctime_ns','st_ino','st_dev')}
def emit(event, **fields): print(json.dumps({'event':event, **fields}),flush=True)

def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

# Reuse result validation, video validation, atomic receipts, CPU imports, and lock protocol.
legacy = module(OLD / 'runner/repeat.py', 'paired5_legacy')
publish_once = legacy.publish_once

def name(variant, seed): return f'task0_paired5_{variant}_20260930_s025000_dev10_seed{seed}_k1'
def image(variant): return BASE / 'images' / variant / 'frozen_runtime'
def run_path(variant, seed):
    p = image(variant) / 'outputs/behavior/closed_loop' / name(variant,seed)
    require(p.resolve()==p, 'Output symlink/escape refused')
    return p

def selected(variant, seeds):
    require(variant in ('original','improved'), 'Invalid variant')
    require(seeds and len(set(seeds))==len(seeds) and seeds==sorted(seeds), 'Empty/duplicate/unsorted seeds refused')
    require(all(type(s) is int and 0<=s<=4 for s in seeds), 'Only paired seeds0..4 allowed')

def verify_code():
    path=BASE / 'runner/code_integrity.json'
    for p,d in read(path).items(): require(sha(p)==d, 'Runner/plan integrity drift: '+p)
    return sha(path)

def configure(variant):
    r=module(OLD / 'runner/repeat.py', 'paired5_'+variant)
    r.BASE=image(variant).parent; r.FROZEN=image(variant); r.RUNNER=r.BASE / 'runner'
    r.PROJECT=PROJECT; r.PROTOCOL=r.RUNNER / 'runtime_protocol.json'; r.INVENTORY=r.RUNNER / 'freeze_inventory.json'
    r.validate_seeds=lambda seeds:selected(variant,seeds)
    r.run_name=lambda seed:name(variant,seed)
    r.run_path=lambda seed:run_path(variant,seed)
    r.expected_artifacts=lambda seed:{str(Path(row['runtime']).resolve()):row['sha256']
                                     for row in read(r.INVENTORY)['files'] if row['baseline_bound'] is True}
    return r

def verify_image(variant):
    r=configure(variant); inv=read(r.INVENTORY)
    require(sha(inv['baseline_manifest'])==inv['baseline_manifest_sha256'], 'Historical manifest drift')
    original=read(OLD / 'runner/freeze_inventory.json')
    old_rows={x['relative']:x for x in original['files']}
    deltas={x['relative']:x['sha256'] for x in read(BASE/'plan_snapshot/protocol.json')['groups']['improved']['planned_source_deltas']}
    checks=0; large=[]
    for row in inv['files']:
        p=Path(row['runtime'])
        expected=deltas[row['relative']] if variant=='improved' and row['relative'] in deltas else old_rows[row['relative']]['sha256']
        require(row['sha256']==expected, 'Unregistered runtime delta')
        if row['copied']: require(p.resolve()==p and p.is_relative_to(r.FROZEN), 'Runtime source path escape')
        if p.stat().st_size >= 16*1024**2:
            require(stat(p)==row['assembly_stat'], 'Large payload stat changed since assembly')
            large.append({'path':str(p), 'historical_sha256':expected, 'stat_matches':True, 'fresh_content_hash':False})
        else:
            require(sha(p)==expected, 'Runtime/shared small file SHA drift: '+str(p)); checks+=1
    for rel,target in inv['shared_links'].items():
        require((r.FROZEN/rel).is_symlink() and (r.FROZEN/rel).resolve()==Path(target), 'Shared mount changed')
    derived=read(r.FROZEN/'data/skill_vtt_task000/contract_targets.json')['source_sha256']
    for rel,digest in derived.items(): require(sha(r.FROZEN/rel)==digest, 'Target source drift')
    return {'small_files_hashed':checks, 'large_payloads':large, 'variant':variant,
            'large_content_validation':'one fresh hash per payload inside each executing launcher, before GPU evaluator startup; stat-checked thereafter'}

def execute_environment():
    return {**os.environ, 'CUDA_VISIBLE_DEVICES':'0', 'PYTHONDONTWRITEBYTECODE':'1',
            'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'}

def cpu_probe(variant):
    r=configure(variant)
    payload=r.cpu_probe()
    expected_root=str(image(variant))
    require(payload['root']==expected_root, 'CPU import root differs from isolated image')
    for p in payload['modules'].values(): require(Path(p).is_relative_to(image(variant)), 'CPU imported live project')
    code="""import json,os,threading,torch,sys,inspect
import behavior_preflight as p, behavior_resources as r, behavior_closed_loop as c
assert not torch.cuda.is_initialized()
assert os.environ['CUDA_VISIBLE_DEVICES']==''
x={'memory_used_percent':84,'cpu_used_percent':84,'gpu':[{'vram_total_mib':16311,'vram_used_mib':15299,'utilization_percent':100}]}
assert r.violation(x) is None
x['gpu'][0]['vram_used_mib']=15300
assert r.violation(x)=='global VRAM threshold reached'
x['gpu'][0]['vram_used_mib']=14580
x['memory_used_percent']=85
assert r.violation(x)=='global RAM threshold reached'
x['memory_used_percent']=84;x['cpu_used_percent']=85
assert r.violation(x)=='global CPU threshold reached'
assert c.wait_for_policy_capacity(threading.Event(),'cuda',timeout=.01,sample=lambda:{'gpu':[{'vram_total_mib':16311,'vram_used_mib':13000}]})
parsed={}
def stop_before_hardware():
    a=inspect.currentframe().f_back.f_locals['args']
    parsed.update(vars(a))
    assert a.profile=='development' and a.instances==[10] and a.exec_horizon==8 and a.samples==1 and a.num_steps==10
    assert a.write_video and a.video_width==1280 and a.video_height==720 and a.video_view=='viewer'
    assert a.log_object_poses and a.env_wrapper=='harness.head_rgbd_wrapper.HeadRGBDWrapper' and a.policy_device=='cuda'
    assert a.max_steps is None and a.timeout==3600 and a.budget_factor==1.5
    raise RuntimeError('EXPECTED_CPU_PARSE_STOP')
c.collect=stop_before_hardware
sys.argv=json.loads(sys.argv[1])[2:]
try: c.main()
except RuntimeError as e:
    assert str(e)=='EXPECTED_CPU_PARSE_STOP'
else: raise AssertionError('CPU parser probe did not stop')
print(json.dumps({'parsed_actual_launcher_arguments':parsed,'cpu_cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'resource_policy':p.CONFIG['resource_policy'],'root':str(p.ROOT),'gpu_peak100_allowed':True,'vram15300_rejected':True,'cpu_ram85_rejected':True,'cuda_initialized':False}))
"""
    result=subprocess.run([str(image(variant)/'.conda-envs/lilawam/bin/python'),'-B','-c',code,json.dumps(r.command(1 if variant=='original' else 0))],
                          cwd=PROJECT,env=r.cpu_env(),capture_output=True,text=True,timeout=120,check=True)
    payload['effective_resource_probe']=json.loads(result.stdout)
    payload['execute_cuda_visible_devices']='0 (receipt and live exclusive ticket required; systemd wrappers set explicitly)'
    return payload

def historical_seed0():
    plan=read(BASE/'plan_snapshot/protocol.json')
    row=next(r for r in plan['runs'] if r['id']=='original_seed0')
    require(row['status']=='completed' and row['reused'] is True, 'Invalid historical seed0')
    # Small source-bound evidence already checked in preflight; do not decode/hash the old video again.
    pre=read(BASE/'plan_snapshot/preflight.json')
    manifest=Path(read(OLD/'runner/freeze_inventory.json')['baseline_manifest'])
    require(sha(manifest)==read(OLD/'runner/freeze_inventory.json')['baseline_manifest_sha256'], 'Historical seed0 changed')
    return row

def receipt_state(variant, seed):
    if variant=='original' and seed==0: return 'complete',historical_seed0()
    directory=STATE/variant/f'seed{seed}'; complete=directory/'complete.json'
    if complete.exists():
        require(not (directory/'infrastructure_error.json').exists(), 'Complete/error receipt conflict; no rerun')
        p=read(complete)
        require(p.get('status')=='complete' and p.get('variant')==variant and p.get('seed')==seed, 'Invalid complete receipt')
        require(p.get('code_integrity_sha256')==sha(BASE/'runner/code_integrity.json'), 'Completed run source binding drift')
        path=run_path(variant,seed)
        require(p.get('authorization_sha256')==AUTH_SHA and p.get('run_dir')==str(path), 'Completed run identity drift')
        require(path.is_dir(), 'Completed run directory missing')
        require(type(p.get('success')) is bool and type(p.get('steps')) is int and p['steps']>0
                and type(p.get('q_score')) in (int,float), 'Completed result fields missing or invalid')
        require(isinstance(p.get('video'),dict) and p['video'].get('all_verified') is True,
                'Completed video verification missing')
        require(isinstance(p.get('exit_confirmation'),list) and len(p['exit_confirmation'])==2,
                'Completed child-exit confirmation missing')
        evidence=p.get('evidence_sha256')
        require(isinstance(evidence,dict) and evidence, 'Completed evidence inventory empty or missing')
        for rel,digest in evidence.items():
            target=path/rel
            require(target.resolve().is_relative_to(path), 'Evidence escape')
            require(sha(target)==digest, 'Completed evidence changed: '+str(target))
        # Reuse the actual manifest/episode/import/video-presence validator, without
        # decoding the movie again. Exact inventory equality rejects omitted evidence.
        current=validate_result(variant,seed,decode=False)
        require(any(rel.startswith('eval/videos/') and rel.endswith('.mp4') for rel in current['evidence_sha256']),
                'Completed video file missing')
        for key,value in current.items():
            if key!='video':
                require(key in p and p[key]==value, 'Completed result/evidence mismatch: '+key)
        return 'complete',p
    require(not directory.exists(), f'Pending/error/incomplete receipt exists; no rerun: {directory}')
    require(not run_path(variant,seed).exists(), 'Existing output refuses overwrite')
    return 'new',None

def workers():
    found=[]
    targets={'train_wam_v3.py','train_lilawam_behavior.py','cache_dino_features.py','behavior_closed_loop.py',
             'replay_demo_actions_feasibility.py','policy_server.py','omnigibson.eval.eval'}
    import re
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try: args=[a.decode(errors='replace') for a in p.read_bytes().split(b'\0') if a]
        except (OSError,ProcessLookupError): continue
        if args and re.fullmatch(r'python(?:\d+(?:\.\d+)*)?',Path(args[0]).name):
            if any(Path(a).name in targets for a in args[1:]): found.append({'pid':int(p.parent.name),'argv':args})
    return found

@contextmanager
def dispatch_lock():
    STATE.mkdir(exist_ok=True)
    with (STATE/'dispatch.flock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def require_permit(permit):
    require(permit is _PERMIT, 'GPU launch requires explicit --execute controller permit')
    authorize(); verify_code()
    accepted=read(BASE/'cpu/acceptance.json')
    require(accepted['code_integrity_sha256']==sha(BASE/'runner/code_integrity.json'), 'CPU acceptance is stale')
    require(accepted['variants']=={'original':'cpu_accepted','improved':'cpu_accepted'}, 'Both paired images need CPU acceptance')
    require(accepted['gpu_started'] is False and accepted['safety_tests_passed'] is True, 'Invalid CPU acceptance')

def acquire_gpu(owner, permit=None):
    require_permit(permit)
    require(not workers(), 'GPU training/simulation worker already running')
    require(not GPU_LOCK.exists() and not GPU_LOCK.is_symlink(), 'GPU lock present; no wait, no steal')
    lock=module(image('original')/'scripts/gpu_lock.py','paired5_gpu_lock')
    lock.acquire(owner,max_wait=0,poll=.25,max_used_mib=4000)
    require(not workers(), 'GPU worker appeared; retaining acquired lock')
    return lock

def assert_exited(pid):
    require(not (Path('/proc')/str(pid)).exists(), 'Launcher PID still exists')
    result=[]
    for unit in (f'wam-policy-{pid}',f'behavior-run-{pid}'):
        r=subprocess.run(['systemctl','--user','show',unit,'--property=LoadState,ActiveState,SubState,MainPID,ControlGroup'],
                         capture_output=True,text=True,timeout=20)
        props=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
        require(props.get('LoadState')=='not-found' or (r.returncode==0 and props.get('ActiveState') in ('inactive','failed')
                and props.get('MainPID')=='0'), 'Child unit exit not confirmed: '+unit)
        group=props.get('ControlGroup')
        if group:
            cg=Path('/sys/fs/cgroup')/group.lstrip('/')
            for path in cg.rglob('cgroup.procs'):
                require(not path.read_text().strip(), 'Child cgroup still populated')
        result.append({'unit':unit,**props})
    require(not workers(), 'A GPU worker still exists; lock retained')
    return result

def validate_result(variant, seed, *, decode=True):
    r=configure(variant)
    receipt=r.validate_result(run_path(variant,seed),seed,decode=decode)
    imports=[read(p) for p in (run_path(variant,seed)/'runtime_imports').glob('*.json')]
    require(len(imports)>=3, 'Missing launcher/policy/evaluator import receipts')
    require({'lilawam','behavior6'} <= {p['environment'] for p in imports}, 'Incomplete interpreter receipts')
    for p in imports:
        require(p['cuda_visible_devices']=='0' and p['cpu_probe'] is False, 'Actual GPU environment was CPU-masked')
        require(p['authorization_sha256']==AUTH_SHA and p['variant']==variant, 'Runtime receipt drift')
        require(p['bootstrap_sha256']==sha(r.RUNNER/'bootstrap.py'), 'Runtime bootstrap drift')
        require(p['resource_adapter_sha256']==sha(BASE/'runner/admission.py'), 'Resource overlay drift')
        require(p['cwd']==str(image(variant)) and p['head_resolution']==720 and p['wrist_resolution']==224, 'Runtime/camera drift')
        for source in p['modules'].values(): require(Path(source).is_relative_to(image(variant)), 'Live mutable source imported')
    if variant=='improved':
        metadata=read(run_path(variant,seed)/'eval/policy_metadata.json')
        require(metadata.get('contract_safety_version')=='candidate_hold_release_scope_v1', 'Improved safety metadata missing')
    large=read(run_path(variant,seed)/'large_payload_verification.json')
    require(len(large)==2 and all(p['fresh_content_hash'] is True for p in large.values()), 'Missing initial large payload verification')
    receipt.update(variant=variant, benchmark_toggle_success=receipt['success'], benchmark_q_final=receipt['q_score'],
        full_pick_press_return_completed=None, navigate_independently_verified=None, pick_completed_assisted=None,
        press_toggled_on=None, return_stably_released=None,
        full_task_adjudication='pending independent four-stage trace/privileged-evidence audit; never inferred from Q or HOLD',
        authorization_sha256=AUTH_SHA, code_integrity_sha256=sha(BASE/'runner/code_integrity.json'))
    return receipt

def run_one(variant, seed, owner, permit=None):
    require_permit(permit)
    require(not (variant=='original' and seed==0), 'Original seed0 cannot be launched')
    status,previous=receipt_state(variant,seed)
    if status=='complete': return previous
    require(GPU_LOCK.read_text().strip()==owner, 'Wrong GPU lock owner')
    verify_image(variant)
    directory=STATE/variant/f'seed{seed}'; directory.mkdir(parents=True,exist_ok=False)
    r=configure(variant)
    pending={'variant':variant,'seed':seed,'status':'pending','run_dir':str(run_path(variant,seed)),
             'argv':r.command(seed),'controller_pid':os.getpid(),'lock_owner':owner,'authorization_sha256':AUTH_SHA,
             'code_integrity_sha256':sha(BASE/'runner/code_integrity.json'),'actual_launch_environment':{'CUDA_VISIBLE_DEVICES':'0'}}
    publish_once(directory/'pending.json',pending)
    child=None
    try:
        with (directory/'launcher.log').open('x') as log:
            child=subprocess.Popen(r.command(seed),cwd=image(variant),env=execute_environment(),stdout=log,stderr=subprocess.STDOUT)
            publish_once(directory/'process.json',{'controller_pid':os.getpid(),'launcher_pid':child.pid,
                         'child_units':[f'wam-policy-{child.pid}',f'behavior-run-{child.pid}'],
                         'stop_controller':f'kill -TERM {os.getpid()}', 'lock_owner':owner})
            emit('single_seed_started',**pending,launcher_pid=child.pid)
            rc=child.wait()
        require(rc==0, f'Launcher infrastructure error exit {rc}')
        exited=assert_exited(child.pid)
        verify_code(); verify_image(variant)
        receipt=validate_result(variant,seed)
        receipt['exit_confirmation']=exited
        publish_once(directory/'complete.json',receipt)
        return receipt
    except BaseException as error:
        if child is not None:
            for unit in (f'wam-policy-{child.pid}',f'behavior-run-{child.pid}'):
                try: subprocess.run(['systemctl','--user','stop',unit],capture_output=True,timeout=30)
                except (OSError,subprocess.SubprocessError): pass
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        confirmation=None
        try:
            if child is not None: confirmation=assert_exited(child.pid)
        except Exception as stop_error: confirmation={'unconfirmed':repr(stop_error)}
        publish_once(directory/'infrastructure_error.json',{**pending,'status':'infrastructure_error','error':repr(error),
                     'exit_confirmation':confirmation,'gpu_lock_retained':True,'auto_retry_allowed':False})
        raise

def dry_run(variant,seeds):
    selected(variant,seeds); authorize()
    code=verify_code(); provenance=verify_image(variant)
    runs=[{'variant':variant,'seed':s,'state':receipt_state(variant,s)[0],
           'argv':None if variant=='original' and s==0 else configure(variant).command(s),
           'run_dir':historical_seed0()['run_dir'] if variant=='original' and s==0 else str(run_path(variant,s))} for s in seeds]
    return {'status':'cpu_prepared_not_launched','gpu_or_simulator_started':False,'nvidia_smi_called':False,
            'code_integrity_sha256':code,'provenance':provenance,'cpu_probe':cpu_probe(variant),'runs':runs,
            'execute_boundary':'main-session explicit --execute; at most one NEW seed; live receipt, exclusive GPU lock, startup checks; no GPU training/simulation concurrency',
            'improved_boundary':'CPU-validated candidate HOLD only; no independent result verifier or automatic safe release admitted'}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true'); mode.add_argument('--dry-run',action='store_true'); mode.add_argument('--status',action='store_true')
    parser.add_argument('--variant',choices=('original','improved'),default='original')
    parser.add_argument('--seeds',nargs='+',type=int)
    args=parser.parse_args(argv)
    seeds=args.seeds if args.seeds is not None else ([1] if args.variant=='original' else [0])
    selected(args.variant,seeds)
    if args.status:
        verify_code()
        print(json.dumps([{'variant':args.variant,'seed':s,'state':receipt_state(args.variant,s)[0]} for s in seeds],indent=2)); return
    if not args.execute:
        print(json.dumps(dry_run(args.variant,seeds),indent=2)); return
    authorize(); verify_code()
    with dispatch_lock():
        outstanding=[s for s in seeds if receipt_state(args.variant,s)[0]!='complete']
        require(len(outstanding)<=1, 'Single-dispatch only: select at most one NEW seed; no auto chain')
        if not outstanding:
            emit('all_selected_complete_skipped',variant=args.variant,seeds=seeds,gpu_started=False); return
        # CPU readiness precedes any GPU query/lock. Failure leaves no pending episode.
        dry_run(args.variant,seeds)
        owner=f'task0-paired5-{args.variant}-seed{outstanding[0]}-{os.getpid()}'
        def stopped(signum,frame): raise InterruptedError(f'Controller received signal {signum}')
        signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
        lock=acquire_gpu(owner,_PERMIT)
        try:
            receipt=run_one(args.variant,outstanding[0],owner,_PERMIT)
            require(not workers(), 'GPU workers remain; refusing release')
            require(lock.release(owner), 'GPU lock ownership changed; refusing release')
            emit('single_seed_complete',variant=args.variant,seed=outstanding[0],success=receipt['success'],
                 q_score=receipt['q_score'],full_task=receipt['full_pick_press_return_completed'],gpu_lock_released=True)
        except BaseException:
            emit('single_seed_halted',owner=owner,gpu_lock_retained=True,automatic_retry=False); raise

if __name__=='__main__':
    try: main()
    except Exception as error:
        emit('controller_blocked',error=repr(error)); raise SystemExit(2)
