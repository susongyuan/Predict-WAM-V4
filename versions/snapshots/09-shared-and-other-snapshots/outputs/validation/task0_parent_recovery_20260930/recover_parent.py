import pathlib,json,hashlib
ROOT=pathlib.Path('/home/user/Robotic/project2')
OUT=ROOT/'outputs/validation/task0_parent_recovery_20260930'
LOG=pathlib.Path('/home/user/.claude/projects/-home-user-Robotic-project2')
TARGET='fba00fe6f1843fa2141296f95a9fb15225d1604419320a74da4089bf72dfb8d2'
def sha(b):return hashlib.sha256(b).hexdigest()
edits=[]; results={}; scanned=[]
for p in LOG.rglob('*.jsonl'):
 if 'wf_59a5efc1-2f3' in str(p):continue
 scanned.append(str(p))
 for n,l in enumerate(p.open(),1):
  try:d=json.loads(l)
  except ValueError:continue
  cs=d.get('message',{}).get('content',[])
  if not isinstance(cs,list):continue
  for c in cs:
   if not isinstance(c,dict):continue
   if c.get('type')=='tool_result':results[c.get('tool_use_id')]=c
   if c.get('type')=='tool_use' and c.get('name') in ('Edit','Write'):
    a=c.get('input',{})
    if a.get('file_path','').endswith('/scripts/train_wam_v3.py'):
     edits.append(dict(log=str(p),line=n,timestamp=d.get('timestamp'),tool=c['name'],id=c['id'],input=a))
edits.sort(key=lambda x:x['timestamp'])
for e in edits:
 r=results.get(e['id'],{})
 e['result_is_error']=r.get('is_error',False)
 e['result_present']=bool(r)
 # Only save matching trainer's tool result, never neighboring messages.
 e['result']=r.get('content')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'trainer_edit_history.json').write_text(json.dumps(edits,indent=2))
sources=[ROOT/'scripts/train_wam_v3.py',ROOT/'outputs/validation/task0_continuation_20k_20260929/backups/train_wam_v3.py.concurrent417481',ROOT/'outputs/validation/task0_continuation_recovery_20260929_1828/train_wam_v3.current.py']
report=dict(target_sha256=TARGET,logs_scanned=len(scanned),edit_count=len(edits),attempts=[],status='not_recovered')
for src in sources:
 b=src.read_bytes(); s=b.decode(); steps=[]
 attempt=dict(source=str(src),source_sha256=sha(b),steps=steps);report['attempts'].append(attempt)
 for e in reversed(edits):
  if e['tool']!='Edit' or e['result_is_error']:continue
  a=e['input'];old=a['old_string'];new=a['new_string'];count=s.count(new)
  st=dict(tool_use_id=e['id'],log=e['log'],line=e['line'],before_sha256=sha(s.encode()),new_string_occurrences=count)
  if count==1 or (count>0 and a.get('replace_all')):
   s=s.replace(new,old) if a.get('replace_all') else s.replace(new,old,1)
   st['action']='reverse_exact_new_to_old';st['after_sha256']=sha(s.encode())
  else:st['action']='not_applicable'
  steps.append(st)
  if sha(s.encode())==TARGET:
   path=OUT/'train_wam_v3.py.parent30k';path.write_bytes(s.encode())
   report.update(status='recovered',candidate_path=str(path),candidate_sha256=sha(path.read_bytes()),successful_source=str(src))
   break
 attempt['final_sha256']=sha(s.encode())
 print(src.name,attempt['source_sha256'],attempt['final_sha256'],[(st['line'],st['action']) for st in steps])
 if report['status']=='recovered':break
(OUT/'source_recovery.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status'])
