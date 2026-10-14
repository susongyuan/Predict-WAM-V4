#!/usr/bin/env python3
"""GPU-serial, resumable orchestration: 5k training segments -> fixed dev physical task -> next segment.
No policy success fabrication; physical failure does not early-stop training; infrastructure failure stops chain.
Start only with reviewed approval and independent NEW output prefix. --plan-only never launches GPU work.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',required=True);p.add_argument('--config',required=True);p.add_argument('--cache-ledger',required=True);p.add_argument('--approval',required=True);p.add_argument('--budget',type=int,choices=[30000,50000],default=30000);p.add_argument('--instance',type=int,default=10);p.add_argument('--resume-source');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
 if not a.prefix.replace('_','').replace('-','').isalnum():raise ValueError('safe prefix required')
 if a.instance not in range(10,20):raise ValueError('development instances only, never report')
 out=ROOT/'outputs/training'/f'{a.prefix}_stages';out.mkdir(exist_ok=False)
 plan={'version':1,'created_unix':time.time(),'total_optimizer_steps':a.budget,'checkpoints':list(range(5000,a.budget+1,5000)),
  'training':{'config':a.config,'config_sha256':digest(ROOT/a.config),'approval':a.approval,'cache_ledger':a.cache_ledger,
   'resume_source':a.resume_source,'optimizer_resume':True,'early_stopping':False},
  'physical':{'profile':'development','instance':a.instance,'seed':0,'samples':1,'exec_horizon':8,'task_id':0,
   'checkpoint_rule':'exact interval checkpoint, NOT retrospectively best','attempts_per_interval':1,'video':[1280,720],
   'success':'only evaluator success/BDDL and final Q; no timeout or program-end success','fixed_program':True,
   'scope':'development checkpoint tracking, not final report or VLM/asking/generalization validation'},
  'resource':'exclusive GPU; unit exits and lock releases before physical test','physical_failure':'record and continue unchanged training',
  'infra_failure':'stop chain, preserve last and all failure evidence','no_test_feedback_to_training':True,'automatic_extension':False}
 write(out/'plan.json',plan)
 if a.plan_only:print(json.dumps(plan,indent=2));return 0
 if not (ROOT/a.approval).is_file():raise ValueError('reviewed approval missing')
 resume=a.resume_source
 for step in plan['checkpoints']:
  name=f'{a.prefix}_s{step:06d}';unit=f'wam-train-{name}.service';physical=f'{a.prefix}_s{step:06d}_dev{a.instance}_k1'
  row={'phase':'launch_training','step_target':step,'training_name':name,'unit':unit,'physical_name':physical,'time':time.time()}
  write(out/'status.json',row)
  env={**os.environ,'NAME':name,'MODE':'long','MAX_STEPS':str(step),'CONFIG':a.config,'CACHE_LEDGER':a.cache_ledger,'APPROVAL':a.approval,'P2_TASK_ID':'0'}
  for k in ['RESUME','INIT','RUN_NAME']:env.pop(k,None)
  if resume:env['RESUME']=resume
  subprocess.run([str(ROOT/'scripts/run_wam_v3_unit.sh')],cwd=ROOT,env=env,check=True)
  while True:
   state=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],capture_output=True,text=True).stdout.strip()
   if state not in ('active','activating','deactivating','reloading'):break
   time.sleep(15)
  run=ROOT/'outputs/training'/name
  summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else {}
  status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
  if status.get('status')!='budget_complete' or summary.get('completed_updates')!=step:
   write(out/'status.json',{**row,'phase':'stopped_training_failure','observed_status':status});return 2
  deadline=time.time()+90
  while Path('/tmp/project2_gpu.lock').exists() and time.time()<deadline:time.sleep(2)
  if Path('/tmp/project2_gpu.lock').exists():raise RuntimeError('GPU lock not released; refusing simulator overlap')
  ck=run/f'checkpoint_step_{step:06d}.pt'
  if not ck.is_file():raise FileNotFoundError('exact interval checkpoint absent: '+str(ck))
  row.update(phase='physical_test',checkpoint=str(ck),checkpoint_sha256=digest(ck));write(out/'status.json',row)
  env={**os.environ,'NAME':physical,'CKPT':str(ck),'INST':str(a.instance),'PROFILE':'development','SAMPLES':'1','EXEC':'8','VIDEO_W':'1280','VIDEO_H':'720'}
  for k in ['DRY_RUN','MAXSTEPS','ORCH_EXTRA']:env.pop(k,None)
  with (out/f'physical_{step:06d}.log').open('x') as f:
   proc=subprocess.Popen([str(ROOT/'scripts/run_task0_closed_loop.sh')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
   write(out/'status.json',{**row,'physical_launcher_pid':proc.pid})
   rc=proc.wait()
  phys=ROOT/'outputs/behavior/closed_loop'/physical
  subprocess.run([sys.executable,str(ROOT/'scripts/summarize_closed_loop.py'),str(phys)],cwd=ROOT,check=True)
  man=json.loads((phys/'manifest.json').read_text()) if (phys/'manifest.json').exists() else {}
  complete=man.get('evaluation_complete') is True and man.get('official_score_valid') is True and rc==0
  row.update(phase='physical_complete' if complete else 'stopped_physical_infrastructure_failure',returncode=rc,physical_manifest=man,
    physical_success_not_inferred=True,time=time.time());write(out/f'stage_{step:06d}.json',row);write(out/'status.json',row)
  if not complete:return 3
  resume=str(run/'checkpoint_last.pt')
  # Explicitly ignore physical success/failure for optimizer/model selection; no changed samples or loss.
 write(out/'status.json',{'phase':'budget_complete','steps':a.budget,'automatic_extension':False,'task_capability_claimed':False,'time':time.time()});return 0
if __name__=='__main__':raise SystemExit(main())
