#!/usr/bin/env python3
"""Bounded TRAIN-only atomic visual probe: exactly 18 static + 2 temporal calls.

Prepare before GPU acquisition. Same pinned 3B BF16, 2 torch threads, 192 tokens.
One process, no retries, externally held only during this job, total <=180 seconds.
Does not modify ground/worker/eval/progress or existing results.
"""
from __future__ import annotations
import argparse
import copy
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness import robotwin_vlm_worker as v
from harness import robotwin_atomic_visual as a
from scripts.probe_robotwin_vlm_train_followup import tensor_audit

OUT=ROOT/'outputs/validation/robotwin_atomic_visual_train_20261006'
OLD=ROOT/'outputs/validation/robotwin_visible_probe_gpu_20261006_0420'
TRAIN=ROOT/'outputs/validation/robotwin_visible_probe_train_20261006'
PYTHON=ROOT/'.conda-envs/lilawam/bin/python'
OWNER='p2-atomic-visual-train-20261006'
LIMIT=180
CASES=tuple(f'episode_{ep:07d}_{part}' for ep in (0,1) for part in ('first','middle','last'))
# Pre-call visible-image references only; not read by the atomic module/model.
REFERENCES={
 'episode_0000000_first':{'state':'false','view':'cam_head','evidence':'Visible wooden front is aligned/closed, no cavity or extended sidewall.'},
 'episode_0000000_last':{'state':'true','view':'cam_head','evidence':'Extended front panel with exposed pale interior floor and sidewalls behind it.'},
 'episode_0000001_first':{'state':'false','view':'cam_head','evidence':'Clearly aligned wooden drawer fronts and handles, no visible opening.'},
 'episode_0000001_last':{'state':'true','view':'cam_head','evidence':'Front panel projects forward; pale open drawer floor and sidewalls are visible.'},
}

def now():return dt.datetime.now().astimezone().isoformat(timespec='milliseconds')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
 tmp=Path(path).with_suffix(Path(path).suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def env():
 os.environ.update({'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                    'OPENBLAS_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
 sys.dont_write_bytecode=True

def source_request(case):
 path=OLD/case/'request.json';req=json.loads(path.read_text());decoded=v.decode_request(req)
 old=json.loads((OLD/case/'response.json').read_text())['response']
 if decoded['previous'] is not None or old['binding']!=v.request_binding(decoded):raise RuntimeError('historical_binding_mismatch')
 montage=v.montage(decoded['current'])
 if old['montage_pixel_sha256']!=[v.digest(montage.tobytes())]:raise RuntimeError('historical_pixel_mismatch')
 return req,decoded

def case_request(case,cache):
 req={'version':1,'id':case['id'],'predicate':case['predicate'],'current':copy.deepcopy(cache[case['current']]['current'])}
 if case.get('previous'):req['previous']=copy.deepcopy(cache[case['previous']]['current'])
 a.decode_request(req)
 return req

def prepare():
 if (OUT/'protocol.json').exists():raise RuntimeError('protocol_exists_no_overwrite')
 OUT.mkdir(parents=True,exist_ok=True)
 tests=json.loads((OUT/'cpu_tests.json').read_text())
 if tests['failures'] or tests['errors'] or tests['gpu_used']:raise RuntimeError('CPU_tests_not_passed')
 cache={};input_ledger={}
 for cid in CASES:
  req,decoded=source_request(cid);cache[cid]=req
  input_ledger[cid]={'source_request':str(OLD/cid/'request.json'),'source_request_sha256':sha(OLD/cid/'request.json'),
                     'native_montage':str(TRAIN/cid/'native3rgb_montage.png'),
                     'binding':v.request_binding(decoded),'montage_pixel_sha256':v.digest(v.montage(decoded['current']).tobytes())}
 refs=copy.deepcopy(REFERENCES)
 for cid,item in refs.items():
  path=TRAIN/cid/'cam_head.png';item.update(image=str(path),image_sha256=sha(path))
 atomic(OUT/'pre_call_drawer_annotations.json',{'created_at':now(),'scope':'ONLY four unambiguous TRAIN drawer_open first/last frames',
   'method':'Pre-call direct image inspection; no simulator GT. Not blind to earlier reports; no labels for middle frames or other predicates.',
   'annotations':refs,'model_input':False})
 cases=[{'id':f'{cid}__{pred}','predicate':pred,'current':cid} for cid in CASES for pred in a.STATIC_PREDICATES]
 cases += [{'id':f'episode_{ep:07d}__visible_release','predicate':'visible_release','current':f'episode_{ep:07d}_last',
            'previous':f'episode_{ep:07d}_middle'} for ep in (0,1)]
 assert len(cases)==20 and len(cases)<=24
 files=[Path(__file__).resolve(),Path(a.__file__).resolve(),Path(v.__file__).resolve(),ROOT/'scripts/gpu_lock.py',
        ROOT/'scripts/probe_robotwin_vlm_train_followup.py']
 protocol={'schema':'TRAIN_atomic_visual_probe_v1','registered_at':now(),'before_gpu_lock_or_model':True,'status':v.STATUS,
  'cases':cases,'calls_planned':20,'calls_absolute_max':24,'retries':0,'hard_total_s':180,'cleanup_reserve_s':5,
  'model_id':v.MODEL_ID,'revision':v.REVISION,'dtype':'bf16','max_new_tokens':192,'do_sample':False,'torch_threads':2,
  'owner':OWNER,'lock_policy':'acquire only when immediately free via gpu_lock.acquire(max_wait=0); do not stop any other job',
  'annotation_sha256':sha(OUT/'pre_call_drawer_annotations.json'),'scored_labels':'only four pre-call drawer_open references',
  'prompts':{pred:{'version':a.VERSIONS[pred],'sha256':v.digest(a.PROMPTS[pred].encode()),'text':a.PROMPTS[pred],
                   'claim_scope':a.SCOPES[pred]} for pred in a.PREDICATES},
  'input_ledger':input_ledger,'source_hashes':{str(path):sha(path) for path in files},
  'resource_policy':'parent system RAM85%; existing backend pre/post RAM85% and VRAM15300MiB; external total timeout',
  'independence':'Every generate uses only its own atomic prompt and current[/previous] images; no prior answers/chat or other-predicate labels.',
  'excluded_model_inputs':['manual annotations','TRAIN episode/frame numbers','GT','gripper commands','phase','other answers','file paths'],
  'no_integration':'New module only. No edits to existing worker/ground/eval/progress; no robot or queue actions.',
  'read_scope':'Only existing six TRAIN RGB requests and matching receipts, fixed local weights; no HDF or other splits.'}
 atomic(OUT/'protocol.json',protocol)
 for case in cases:
  d=OUT/case['id'];d.mkdir();req=case_request(case,cache)
  atomic(d/'request_binding.json',{'id':case['id'],'predicate':case['predicate'],'source_current':case['current'],
    'source_previous':case.get('previous'),'binding':v.request_binding(a.decode_request(req))})
  (d/'prompt.txt').write_text(a.PROMPTS[case['predicate']])
 atomic(OUT/'prepared.json',{'protocol_sha256':sha(OUT/'protocol.json'),'model_loaded':False,'gpu_called':False})
 print('PREPARED '+str(OUT/'protocol.json')+' CALLS=20',flush=True)

def child():
 parent=os.getppid()
 if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0 or parent==1 or os.getppid()!=parent:
  raise RuntimeError('parent_death_guard_failed')
 env();protocol=json.loads((OUT/'protocol.json').read_text())
 atomic(OUT/'phase.json',{'phase':'load_model','pid':os.getpid(),'at':now()})
 backend=v.GpuBackend(OWNER,'bf16')
 atomic(OUT/'ready.json',{'status':v.STATUS,'backend':v.ready_message(backend),'atomic_api_version':a.API_VERSION,
                         'predicates':list(a.PREDICATES),'prompt_versions':a.VERSIONS})
 active={'id':None};actual_generate=backend.model.generate
 def wrapped_generate(**kwargs):
  tensors={k:kwargs[k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
  stats=tensor_audit(tensors,backend)
  stats.update(at=now(),max_new_tokens=kwargs['max_new_tokens'],do_sample=kwargs['do_sample'],captured_from='actual_generate_kwargs')
  assert stats['max_new_tokens']==192 and stats['do_sample'] is False
  atomic(OUT/active['id']/'token_audit.json',stats)
  return actual_generate(**kwargs)
 backend.model.generate=wrapped_generate  # only this isolated new model instance
 adapter=a.AtomicVisualVerifier(backend);cache={cid:source_request(cid)[0] for cid in CASES};results=[]
 for case in protocol['cases']:
  cid=case['id'];active['id']=cid
  atomic(OUT/'phase.json',{'phase':'query','case_id':cid,'at':now(),'pid':os.getpid(),'calls_completed':len(results)})
  req=case_request(case,cache)
  reply=adapter.query(req)
  assert reply['binding']==v.request_binding(a.decode_request(req))
  assert reply['prompt_sha256']==protocol['prompts'][case['predicate']]['sha256']
  atomic(OUT/cid/'result.json',reply);(OUT/cid/'raw_text.txt').write_text(reply['raw_text'])
  results.append(reply)
  atomic(OUT/'child_result.json',{'status':v.STATUS,'calls':len(results),'completed':len(results)==20,'records':results})
  print(f'CASE_DONE {cid} {reply["outcome"]} state={None if reply["parsed"] is None else reply["parsed"]["state"]}',flush=True)
  if reply['outcome']=='runtime_error':return 2
 return 0

def run():
 if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError('lilawam_required')
 if (OUT/'supervisor.json').exists():raise RuntimeError('no_automatic_retry')
 protocol=json.loads((OUT/'protocol.json').read_text())
 if sha(OUT/'protocol.json')!=json.loads((OUT/'prepared.json').read_text())['protocol_sha256']:raise RuntimeError('protocol_changed')
 for path,h in protocol['source_hashes'].items():
  if sha(path)!=h:raise RuntimeError('source_changed')
 if sha(OUT/'pre_call_drawer_annotations.json')!=protocol['annotation_sha256']:raise RuntimeError('annotations_changed')
 from scripts import gpu_lock
 acquired=False;process=None;stop=[None];reason=None;started=time.monotonic()
 result={'status':v.STATUS,'outcome':'running','supervisor_pid':os.getpid(),'started_at':now()}
 atomic(OUT/'supervisor.json',result)
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,'SIGTERM'));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,'SIGINT'))
 try:
  wait=gpu_lock.acquire(OWNER,max_wait=0);acquired=True;v.check_gpu_owner(OWNER,exclusive=True)
  result.update(owner=OWNER,acquired_at=now(),lock_wait_s=wait)
  env()
  with (OUT/'child.log').open('wb') as log:
   process=subprocess.Popen([str(PYTHON),'-B',str(Path(__file__).resolve()),'--child'],cwd=ROOT,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   result['child_pid']=process.pid;atomic(OUT/'supervisor.json',result)
   print(f'SUPERVISOR_PID={os.getpid()} CHILD_PID={process.pid} OWNER={OWNER} OUTPUT={OUT}',flush=True)
   while process.poll() is None:
    memory=v.cpu_resources();elapsed=time.monotonic()-started
    append(OUT/'resources.jsonl',{'at':now(),'elapsed_s':elapsed,**memory})
    if memory['system_ram_percent']>=85:reason='RAM85'
    if elapsed>=LIMIT-5:reason=reason or 'hard_budget_cleanup'
    reason=reason or stop[0]
    if reason:
     os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3);break
    time.sleep(.2)
   result['child_exit_code']=process.wait(timeout=3)
  result['outcome']='completed' if result['child_exit_code']==0 else 'failed'
 except Exception as exc:
  result.update(outcome='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  if process is not None and process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
  result['own_lock_released']=gpu_lock.release(OWNER) if acquired else False
  result.update(finished_at=now(),total_s=time.monotonic()-started,stop_reason=reason)
  if (OUT/'child_result.json').exists():result['child_result']=json.loads((OUT/'child_result.json').read_text())
  result['not_completed']=[c['id'] for c in protocol['cases'] if not (OUT/c['id']/'result.json').exists()]
  atomic(OUT/'result.json',result);atomic(OUT/'supervisor.json',result)
  print(json.dumps({k:result[k] for k in ('outcome','total_s','own_lock_released','not_completed')}),flush=True)
 return 0 if result['outcome']=='completed' else 2

def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--child',action='store_true');x=p.parse_args()
 if x.prepare:prepare();return 0
 if x.run:return run()
 return child()
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception:
  if OUT.exists():atomic(OUT/f'failure_{os.getpid()}.json',{'at':now(),'traceback':traceback.format_exc()})
  raise
