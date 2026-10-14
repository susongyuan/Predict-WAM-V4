#!/usr/bin/env python3
"""One registered 7B NF4/double-quant TRAIN comparison: 20 atomic + 3 ground.

--prepare is CPU-only. --run acquires only an immediately free project GPU lock,
supervises one child for <=240s, monitors RAM85%/VRAM15300MiB and releases its own
lock. No installation, download, fallback precision/model, retries, source edits,
GT, policy execution or evaluation integration.
"""
from __future__ import annotations
import argparse
import collections
import copy
import ctypes
import datetime as dt
import hashlib
import importlib.metadata
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
from scripts import probe_robotwin_atomic_visual_train as old_atomic
from scripts.probe_robotwin_vlm_train_followup import tensor_audit

OUT=ROOT/'outputs/validation/robotwin_atomic_visual_7b_train_20261006'
ATOMIC=ROOT/'outputs/validation/robotwin_atomic_visual_train_20261006'
GROUND=ROOT/'outputs/validation/robotwin_vlm_train_followup_20261006'
PYTHON=ROOT/'.conda-envs/lilawam/bin/python'
MODEL_ID='Qwen/Qwen2.5-VL-7B-Instruct'
REVISION='cc594898137f460bfe9f0759e9844b3ce807cfb5'
MODEL=Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots')/REVISION
OWNER='p2-atomic-7b-train-20261006'
LIMIT=240
GROUND_IDS=('ground_explicit','ground_ambiguous','ground_unsupported')
QUANT={'load_in_4bit':True,'bnb_4bit_quant_type':'nf4','bnb_4bit_use_double_quant':True,
       'bnb_4bit_compute_dtype':'bfloat16','bnb_4bit_quant_storage':'uint8',
       'non_quantized_requested_dtype':'bfloat16','cpu_offload':False}


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

def identity(path,content=False):
 p=Path(path);s=p.stat();d={'path':str(p),'resolved_path':str(p.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}
 if content:d['sha256']=sha(p)
 return d

def prepare():
 if OUT.exists():raise RuntimeError('new_output_required_no_overwrite')
 OUT.mkdir(parents=True)
 if 'torch' in sys.modules:raise RuntimeError('prepare_must_not_import_torch')
 ap=json.loads((ATOMIC/'protocol.json').read_text());gp=json.loads((GROUND/'protocol.json').read_text())
 assert len(ap['cases'])==20
 for pred in a.PREDICATES:
  if a.PROMPTS[pred]!=ap['prompts'][pred]['text'] or a.VERSIONS[pred]!=ap['prompts'][pred]['version']:
   raise RuntimeError('atomic_prompt_version_changed')
 if v.GROUND_PROMPT!=gp['ground_prompt_unchanged']:raise RuntimeError('ground_prompt_changed')
 cases=[{'id':c['id'],'kind':'atomic','atomic_case':copy.deepcopy(c),
         'baseline_result':str(ATOMIC/c['id']/'result.json'),'baseline_token_audit':str(ATOMIC/c['id']/'token_audit.json')} for c in ap['cases']]
 for cid in GROUND_IDS:
  case=next(c for c in gp['cases'] if c['id']==cid)
  request=json.loads((GROUND/cid/'request.json').read_text());v.decode_request(request)
  if request['instruction']!=case['instruction']:raise RuntimeError('ground_instruction_changed')
  cases.append({'id':cid,'kind':'ground','instruction':case['instruction'],
   'request_file':str(GROUND/cid/'request.json'),'baseline_result':str(GROUND/cid/'result.json'),
   'baseline_token_audit':str(GROUND/cid/'actual_input_audit.json')})
 assert len(cases)==23
 indexes=json.loads((MODEL/'model.safetensors.index.json').read_text())
 shards=sorted(set(indexes['weight_map'].values()))
 if len(shards)!=5 or any(Path(name).name!=name for name in shards):raise RuntimeError('unexpected_local_shards')
 model_files=shards+['config.json','generation_config.json','model.safetensors.index.json','preprocessor_config.json',
                    'tokenizer_config.json','tokenizer.json','chat_template.json','vocab.json','merges.txt']
 model_inventory=[identity(MODEL/name,not name.endswith('.safetensors')) for name in model_files]
 packages={name:importlib.metadata.version(name) for name in ('torch','transformers','bitsandbytes','accelerate','Pillow')}
 if packages['bitsandbytes']!='0.50.2' or packages['accelerate']!='1.15.0':
  raise RuntimeError('declared_environment_version_mismatch_no_install')
 input_ledger={}
 for cid in old_atomic.CASES:
  req,decoded=old_atomic.source_request(cid)
  meta=ap['input_ledger'][cid]
  if v.request_binding(decoded)!=meta['binding']:raise RuntimeError('TRAIN_binding_changed')
  if sha(meta['source_request'])!=meta['source_request_sha256']:raise RuntimeError('TRAIN_pixels_changed')
  input_ledger[cid]=copy.deepcopy(meta)
 files=[Path(__file__).resolve(),Path(a.__file__).resolve(),Path(v.__file__).resolve(),Path(old_atomic.__file__).resolve(),
        ROOT/'scripts/probe_robotwin_vlm_train_followup.py',ROOT/'scripts/gpu_lock.py']
 frozen=[ATOMIC/'protocol.json',ATOMIC/'result.json',ATOMIC/'pre_call_drawer_annotations.json',GROUND/'protocol.json',GROUND/'result.json']
 frozen += [Path(c['baseline_result']) for c in cases]+[Path(c['baseline_token_audit']) for c in cases]
 frozen += [Path(c['request_file']) for c in cases if c['kind']=='ground']
 p={'schema':'7B_quantized_TRAIN_atomic_comparison_v1','registered_at':now(),'before_gpu_lock_or_model':True,
  'status':v.STATUS,'model_id':MODEL_ID,'revision':REVISION,'local_snapshot':str(MODEL),'model_inventory':model_inventory,
  'quantization':QUANT,'requested_device_map':{'':'cuda:0'},'packages_metadata':packages,'torch_threads':2,'torch_interop_threads':1,
  'max_new_tokens':192,'do_sample':False,'fresh_each_request':True,'calls':23,'atomic_calls':20,'ground_calls':3,
  'cases':cases,'owner':OWNER,'hard_total_s':240,'cleanup_reserve_s':5,'retries':0,
  'lock_policy':'gpu_lock.acquire(max_wait=0), explicit exclusive check; never stop another job',
  'resource_policy':{'system_ram_abort_percent':85,'device_vram_abort_mib':15300,'RAM_poll_s':0.2,'nvidia_poll_s':1.0,
                     'parent_hard_stop_cleanup_s':235,'external_hard_stop_s':240},
  'atomic_prompts':ap['prompts'],'ground_prompt':v.GROUND_PROMPT,'input_ledger':input_ledger,
  'annotations':{'file':str(ATOMIC/'pre_call_drawer_annotations.json'),'sha256':sha(ATOMIC/'pre_call_drawer_annotations.json'),
                 'scope':'reuse same four pre-call visible drawer labels; other predicates have NO independent accuracy labels'},
  'source_hashes':{str(path):sha(path) for path in files},'old_artifact_hashes':{str(path):sha(path) for path in frozen},
  'exclusions':'No GT, HDF, VAL/heldout/dev, gripper commands, episode/frame IDs or other responses in model inputs. No old source/results/progress edits.',
  'comparison':'3B BF16 vs 7B NF4 double quant BF16 compute; size and quantization both change, not isolated size causality.',
  'failure_policy':'Missing backend/kernel/API, wrong actual quantization, unsupported BF16, OOM or input mismatch is a recorded failure. No installation, download, dtype/model fallback, retry or repairing model output.'}
 atomic(OUT/'protocol.json',p)
 for c in cases:
  d=OUT/c['id'];d.mkdir()
  prompt=a.PROMPTS[c['atomic_case']['predicate']] if c['kind']=='atomic' else v.GROUND_PROMPT+json.dumps(c['instruction'],ensure_ascii=False)
  (d/'prompt.txt').write_text(prompt)
 atomic(OUT/'prepared.json',{'protocol_sha256':sha(OUT/'protocol.json'),'torch_imported':False,'model_loaded':False,'gpu_called':False})
 print('PREPARED '+str(OUT/'protocol.json')+' CALLS=23 QUANT=NF4_doublequant_BF16',flush=True)


class Quantized7BBackend(v.GpuBackend):
 """New subclass, replacing ONLY construction; reuse frozen generation/guards."""
 def __init__(self,owner):
  self.owner=owner;self.dtype_name='bf16_compute_nf4_double_quant'
  v.check_gpu_owner(owner);v.configure_runtime()
  if v.cpu_resources()['system_ram_percent']>=85:raise RuntimeError('RAM85_before_load')
  import torch
  from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
  import bitsandbytes as bnb
  self.torch=torch;torch.set_num_threads(2);torch.set_num_interop_threads(1)
  v.check_gpu_owner(owner)
  if not torch.cuda.is_available():raise RuntimeError('CUDA_unavailable_no_fallback')
  if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16_unsupported_no_fallback')
  self.check_resources()
  quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                          bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_storage=torch.uint8,
                          llm_int8_enable_fp32_cpu_offload=False)
  atomic(OUT/'requested_quantization.json',quant.to_dict())
  begin=time.monotonic()
  self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL),device_map={'':'cuda:0'},
     dtype=torch.bfloat16,quantization_config=quant,local_files_only=True,trust_remote_code=False).eval()
  self.processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
  if not getattr(self.model,'is_loaded_in_4bit',False):raise RuntimeError('not_actually_loaded_in_4bit')
  if any(p.device.type!='cuda' or p.device.index!=0 for p in self.model.parameters()):raise RuntimeError('CPU_or_other_device_offload_not_allowed')
  layers=[]
  for name,module in self.model.named_modules():
   if isinstance(module,bnb.nn.Linear4bit):
    weight=module.weight;qs=getattr(weight,'quant_state',None)
    item={'name':name,'compute_dtype':str(module.compute_dtype),'quant_type':getattr(weight,'quant_type',None),
          'compress_statistics':getattr(weight,'compress_statistics',None),'storage_dtype':str(weight.dtype),
          'quant_state_present':qs is not None,'nested':getattr(qs,'nested',None)}
    layers.append(item)
    if module.compute_dtype!=torch.bfloat16 or item['quant_type']!='nf4' or item['compress_statistics'] is not True or item['nested'] is not True:
     raise RuntimeError('actual_4bit_configuration_mismatch: '+str(item))
  if not layers:raise RuntimeError('no_Linear4bit_modules_found')
  actual={'is_loaded_in_4bit':True,'linear4bit_count':len(layers),'layers':layers,
    'parameter_dtype_counts':dict(collections.Counter(str(p.dtype) for p in self.model.parameters())),
    'parameter_devices':sorted({str(p.device) for p in self.model.parameters()}),
    'hf_device_map':getattr(self.model,'hf_device_map',None),'memory_footprint_bytes':self.model.get_memory_footprint(),
    'requested_quantization_config':quant.to_dict()}
  atomic(OUT/'actual_quantization.json',actual)
  self.metadata={'id':MODEL_ID,'revision':REVISION,'device':'cuda:0','dtype':'NF4_4bit_double_quant_BF16_compute',
    'quantization':quant.to_dict(),'linear4bit_count':len(layers),'model_class':type(self.model).__name__,
    'processor_class':type(self.processor).__name__,'image_processor_class':type(self.processor.image_processor).__name__,
    'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
    'bitsandbytes':importlib.metadata.version('bitsandbytes'),'accelerate':importlib.metadata.version('accelerate'),
    'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
    'config_sha256':sha(MODEL/'config.json'),'load_s':time.monotonic()-begin,
    'gpu_name':torch.cuda.get_device_name(0),'gpu_capability':list(torch.cuda.get_device_capability(0))}
  self.guard()


def child():
 parent=os.getppid()
 if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0 or parent==1 or os.getppid()!=parent:
  raise RuntimeError('parent_death_guard_failed')
 env();protocol=json.loads((OUT/'protocol.json').read_text())
 atomic(OUT/'phase.json',{'phase':'quantized_model_load','pid':os.getpid(),'at':now()})
 backend=Quantized7BBackend(OWNER)
 atomic(OUT/'ready.json',{'status':v.STATUS,'model':backend.metadata,'resources':backend.resources(),
                        'max_new_tokens':192,'calls_planned':23})
 active={'case':None};original_generate=backend.model.generate
 def audited_generate(**kwargs):
  case=active['case'];inputs={k:kwargs[k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
  audit=tensor_audit(inputs,backend);prior=json.loads(Path(case['baseline_token_audit']).read_text())
  audit.update(at=now(),captured_from='actual_generate_kwargs',max_new_tokens=kwargs['max_new_tokens'],do_sample=kwargs['do_sample'],
    baseline_pixel_tensor_equal=audit['pixel_tensor_sha256_float32']==prior['pixel_tensor_sha256_float32'],
    baseline_input_token_ids_equal=audit['input_token_ids']==prior['input_token_ids'])
  atomic(OUT/case['id']/'token_audit.json',audit)
  if not audit['baseline_pixel_tensor_equal'] or audit['image_grid_thw']!=prior['image_grid_thw']:
   raise RuntimeError('preprocessing_differs_from_frozen_3B_inputs_no_silent_override')
  if kwargs['max_new_tokens']!=192 or kwargs['do_sample'] is not False:raise RuntimeError('generation_budget_changed')
  return original_generate(**kwargs)
 backend.model.generate=audited_generate  # isolated 7B instance, never old worker/source
 adapter=a.AtomicVisualVerifier(backend);ground_service=v.Service(backend,backend.guard)
 cache={cid:old_atomic.source_request(cid)[0] for cid in old_atomic.CASES};records=[]
 for case in protocol['cases']:
  cid=case['id'];active['case']=case;directory=OUT/cid
  atomic(OUT/'phase.json',{'phase':'generate','case_id':cid,'at':now(),'pid':os.getpid(),'completed':len(records)})
  started=time.monotonic()
  if case['kind']=='atomic':
   request=old_atomic.case_request(case['atomic_case'],cache)
   response=adapter.query(request)
   baseline=json.loads(Path(case['baseline_result']).read_text())
  else:
   request=json.loads(Path(case['request_file']).read_text())
   response=ground_service.handle(request)
   baseline=json.loads(Path(case['baseline_result']).read_text())['response']
  if response['binding']!=baseline['binding'] or response.get('prompt_sha256')!=baseline['prompt_sha256']:
   raise RuntimeError('case_binding_or_prompt_mismatch')
  atomic(directory/'request_binding.json',{'id':cid,'kind':case['kind'],'binding':response['binding']})
  result={'case_id':cid,'kind':case['kind'],'response':response,'roundtrip_s':time.monotonic()-started}
  atomic(directory/'result.json',result);(directory/'raw_text.txt').write_text(response['raw_text'])
  records.append(result);atomic(OUT/'child_result.json',{'status':v.STATUS,'records':records,'calls':len(records),'completed':len(records)==23})
  print('CASE_DONE '+cid+' '+response['outcome'],flush=True)
  if response['outcome']=='runtime_error':return 2
 return 0


def monitor(pid=None,query_gpu=False):
 sample={'at':now(),**v.cpu_resources()}
 if pid:
  try:
   for line in Path(f'/proc/{pid}/status').read_text().splitlines():
    key,_,value=line.partition(':')
    if key in ('VmRSS','VmHWM','VmSwap'):sample[key+'_bytes']=int(value.split()[0])*1024
  except FileNotFoundError:pass
 if query_gpu:
  result=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,timeout=3,check=True)
  used,total=[float(x.strip()) for x in result.stdout.strip().splitlines()[0].split(',')]
  sample.update(nvidia_used_mib=used,nvidia_total_mib=total)
 return sample


def run():
 if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError('existing_lilawam_required')
 if (OUT/'supervisor.json').exists():raise RuntimeError('run_already_started_no_retry')
 started=time.monotonic();p=json.loads((OUT/'protocol.json').read_text())
 if sha(OUT/'protocol.json')!=json.loads((OUT/'prepared.json').read_text())['protocol_sha256']:raise RuntimeError('protocol_changed')
 for path,h in {**p['source_hashes'],**p['old_artifact_hashes']}.items():
  if sha(path)!=h:raise RuntimeError('source_or_baseline_changed: '+path)
 for entry in p['model_inventory']:
  if identity(entry['path'],'sha256' in entry)!=entry:raise RuntimeError('local_model_identity_changed')
 from scripts import gpu_lock
 acquired=False;process=None;cancel=[None];reason=None
 result={'status':v.STATUS,'outcome':'running','supervisor_pid':os.getpid(),'started_at':now()}
 atomic(OUT/'supervisor.json',result)
 signal.signal(signal.SIGTERM,lambda *_:cancel.__setitem__(0,'SIGTERM'));signal.signal(signal.SIGINT,lambda *_:cancel.__setitem__(0,'SIGINT'))
 try:
  if v.cpu_resources()['system_ram_percent']>=85:raise RuntimeError('RAM85_before_lock')
  wait=gpu_lock.acquire(OWNER,max_wait=0);acquired=True;v.check_gpu_owner(OWNER,exclusive=True)
  result.update(owner=OWNER,lock_acquired_at=now(),lock_wait_s=wait)
  env()
  with (OUT/'child.log').open('wb') as log:
   process=subprocess.Popen([str(PYTHON),'-B',str(Path(__file__).resolve()),'--child'],cwd=ROOT,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   result['child_pid']=process.pid;atomic(OUT/'supervisor.json',result)
   print(f'SUPERVISOR_PID={os.getpid()} CHILD_PID={process.pid} OWNER={OWNER} OUTPUT={OUT}',flush=True)
   next_gpu=0.0
   while process.poll() is None:
    t=time.monotonic();gpu_query=t>=next_gpu
    sample=monitor(process.pid,gpu_query);sample['elapsed_s']=t-started
    if gpu_query:next_gpu=time.monotonic()+1
    append(OUT/'resources.jsonl',sample)
    if sample['system_ram_percent']>=85:reason='RAM85'
    if sample.get('nvidia_used_mib',0)>=15300:reason=reason or 'VRAM15300'
    if time.monotonic()-started>=LIMIT-5:reason=reason or 'hard_timeout_cleanup'
    reason=reason or cancel[0]
    if reason:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3);break
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
  result['not_completed']=[c['id'] for c in p['cases'] if not (OUT/c['id']/'result.json').exists()]
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
  if OUT.exists():atomic(OUT/f'failure_{os.getpid()}.json',{'at':now(),'traceback':traceback.format_exc(),'fallback_attempted':False})
  raise
