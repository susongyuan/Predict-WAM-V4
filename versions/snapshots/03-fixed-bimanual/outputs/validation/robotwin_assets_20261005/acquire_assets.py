#!/usr/bin/env python3
"""Fixed-revision asset acquisition only. Never imports or runs upstream code."""
import datetime, hashlib, json, os, pathlib, shutil, stat, subprocess, sys, time, traceback, urllib.request, zipfile

ROOT = pathlib.Path('/home/user/Robotic/project2')
BASE = ROOT / 'data/benchmarks/robotwin2/assets'
LOG = ROOT / 'outputs/validation/robotwin_assets_20261005'
REV = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'
REPO = 'TianxingChen/RoboTwin2.0'
API = f'https://huggingface.co/api/datasets/{REPO}'
ORIGIN = f'https://huggingface.co/datasets/{REPO}'
EXPECTED = {
 'embodiments.zip': (219847741, '85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789'),
 'objects.zip': (3737778549, '6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27'),
 'background_texture.zip': (10970687027, '54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070'),
}
GIB = 1024**3
CHUNK = 1024**2
BUDGET = 80 * GIB
MANIFEST = LOG / 'asset_manifest.json'
TRANS = BASE / '.transfers'
STAGING = BASE / '.staging'
for p in (BASE, LOG, TRANS, STAGING):
 p.mkdir(parents=True, exist_ok=True)
 if p.is_symlink(): raise RuntimeError(f'Refusing symlink work directory: {p}')

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def event(kind, **data):
 row = {'time': now(), 'event': kind, **data}
 line = json.dumps(row, ensure_ascii=False)
 print(line, flush=True)
 with (LOG / 'download.jsonl').open('a') as f: f.write(line + '\n')

def atomic_json(path, obj):
 tmp = path.with_name(path.name + '.tmp')
 with tmp.open('w') as f: json.dump(obj, f, indent=2, ensure_ascii=False); f.write('\n')
 os.replace(tmp, path)

if MANIFEST.exists():
 m = json.loads(MANIFEST.read_text())
 if m.get('repository') != REPO or m.get('revision') != REV: raise RuntimeError('Existing manifest has different provenance')
 if m.get('status') == 'complete':
  raise RuntimeError('Complete manifest exists; independently inspect receipt before any change')
else:
 m = {'schema_version': 1, 'repository': REPO, 'repository_url': ORIGIN, 'revision': REV,
      'created_at': now(), 'status': 'partial', 'asset_root': str(BASE), 'archives': {},
      'scope': 'Assets only; no training, GPU, renderer, simulation, package installation or downloaded-code execution.',
      'distribution_note': 'Full object library and background textures retained for randomized evaluation; no clean-to-randomized substitution. Asset preparation is not proof that the environment runs.',
      'selection': 'embodiments: only case-insensitive Aloha-AgileX subtree and all archive license/notice/copyright files; objects/background_texture: all members.',
      'limits': {'max_uncompressed_bytes_per_archive': BUDGET, 'max_total_uncompressed_bytes': 160*GIB, 'max_file_bytes': 16*GIB, 'max_archive_members': 500000, 'cpu_percent_ceiling': 85, 'ram_percent_ceiling': 85, 'stream_buffer_bytes': CHUNK},
      'symlinks': [], 'generated_configs': [], 'errors': []}

def save():
 m['updated_at'] = now()
 atomic_json(MANIFEST, m)

last_cpu = None
last_guard = 0.0
last_wait_log = 0.0

def resources():
 global last_cpu
 with open('/proc/meminfo') as f: mem = {a[0].rstrip(':'): int(a[1]) for a in (x.split() for x in f)}
 ram = 100*(1-mem['MemAvailable']/mem['MemTotal'])
 with open('/proc/stat') as f: vals = [int(x) for x in f.readline().split()[1:9]]
 total = sum(vals); idle = vals[3]+vals[4]
 cpu = 0 if last_cpu is None or total == last_cpu[0] else 100*(1-(idle-last_cpu[1])/(total-last_cpu[0]))
 last_cpu = (total,idle)
 return cpu,ram

def guard(force=False):
 global last_guard,last_wait_log
 t=time.monotonic()
 if not force and t-last_guard < 2: return
 cpu,ram=resources(); last_guard=t
 while cpu > 85 or ram > 85:
  if time.monotonic()-last_wait_log > 30:
   event('resource_pause',cpu_percent=round(cpu,2),ram_percent=round(ram,2)); last_wait_log=time.monotonic()
  time.sleep(5)
  cpu,ram=resources(); last_guard=time.monotonic()

try: os.nice(15)
except OSError: pass

def remote_bytes(url, limit=4*1024**2):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RoboTwin-asset-integrity/1.0'}),timeout=90) as r:
  b=r.read(limit+1)
  if len(b)>limit: raise RuntimeError('Metadata response too large')
  return b

def digest(path):
 h=hashlib.sha256(); n=0
 with path.open('rb') as f:
  while True:
   guard(); b=f.read(CHUNK)
   if not b: break
   h.update(b); n+=len(b)
 return n,h.hexdigest()

def download(name, row):
 size,sha=EXPECTED[name]
 complete=TRANS/name
 partial=TRANS/(name+'.partial')
 if complete.exists():
  actual,h=digest(complete)
  if (actual,h)!=(size,sha): raise RuntimeError(f'Existing transfer checksum mismatch: {name}')
  event('reuse_verified_transfer',name=name,bytes=actual)
  row.update(actual_bytes=actual,actual_sha256=h,sha256_verified=True,status='download_verified');save()
  return complete
 for attempt in range(1,9):
  guard(True)
  pos=partial.stat().st_size if partial.exists() else 0
  if pos>size: raise RuntimeError('Partial exceeds expected length')
  if pos==size: break
  h=hashlib.sha256()
  if pos:
   with partial.open('rb') as f:
    for b in iter(lambda:f.read(CHUNK),b''): guard(); h.update(b)
  url=row['url']+f'?download=true&asset_integrity_attempt={time.time_ns()}'
  headers={'User-Agent':'RoboTwin-asset-integrity/1.0','Accept-Encoding':'identity'}
  if pos: headers['Range']=f'bytes={pos}-'
  start=time.monotonic(); next_report=pos+256*CHUNK
  try:
   event('download_start',name=name,attempt=attempt,resume_bytes=pos)
   with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as r:
    if pos and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {pos}-')):
     raise RuntimeError('Server did not honor resume Range; preserved partial')
    if not pos and r.status not in (200,206): raise RuntimeError(f'Unexpected HTTP status {r.status}')
    remaining=r.headers.get('Content-Length')
    if remaining is not None and int(remaining)!=size-pos: raise RuntimeError('Content-Length differs from official LFS length')
    with partial.open('ab' if pos else 'wb') as f:
     while True:
      guard(); b=r.read(CHUNK)
      if not b: break
      if pos+len(b)>size: raise RuntimeError('Download exceeds expected length')
      f.write(b);h.update(b);pos+=len(b)
      if pos>=next_report:
       row.update(status='downloading',partial_bytes=pos);save()
       event('download_progress',name=name,bytes=pos,total=size,elapsed_seconds=round(time.monotonic()-start,1))
       next_report=pos+256*CHUNK
     f.flush();os.fsync(f.fileno())
   if pos!=size: raise RuntimeError(f'Truncated download {pos}/{size}')
   actual_sha=h.hexdigest()
   if actual_sha!=sha: raise RuntimeError(f'LFS SHA-256 mismatch {actual_sha}')
   break
  except Exception as e:
   row.update(status='partial',partial_bytes=partial.stat().st_size if partial.exists() else 0,last_download_error=str(e));save()
   event('download_attempt_error',name=name,attempt=attempt,error=str(e))
   if attempt==8 or 'SHA-256 mismatch' in str(e): raise
   time.sleep(min(30,attempt*3))
 else: raise RuntimeError('Download retries exhausted')
 actual,actual_sha=digest(partial)
 row.update(actual_bytes=actual,actual_sha256=actual_sha)
 if (actual,actual_sha)!=(size,sha): raise RuntimeError(f'Final LFS verification failed: {name}')
 os.replace(partial,complete)
 row.update(status='download_verified',sha256_verified=True,transfer_path=str(complete));save()
 event('download_verified',name=name,bytes=actual,sha256=actual_sha)
 return complete

def license_file(parts):
 return any(token in parts[-1].lower() for token in ('license','licence','copyright','copying','notice'))

def safe_member(info):
 name=info.filename
 if not name or '\x00' in name or '\\' in name or name.startswith('/') or ':' in name:
  raise RuntimeError(f'Unsafe ZIP name {name!r}')
 parts=name.rstrip('/').split('/')
 if any(p in ('','.', '..') for p in parts): raise RuntimeError(f'Unsafe ZIP path {name!r}')
 mode=(info.external_attr >> 16) & 0xffff
 kind=stat.S_IFMT(mode)
 if kind not in (0,stat.S_IFREG,stat.S_IFDIR): raise RuntimeError(f'Symlink or special ZIP entry {name!r}')
 if info.flag_bits & 1: raise RuntimeError(f'Encrypted ZIP entry {name!r}')
 if info.file_size>16*GIB: raise RuntimeError('ZIP individual file budget exceeded')
 if info.file_size>CHUNK and info.file_size>2000*max(1,info.compress_size): raise RuntimeError('ZIP expansion ratio budget exceeded')
 return parts

def extract(name,row,archive):
 rootname=name[:-4]
 stage=STAGING/(rootname+'-'+REV[:12])
 installed=BASE/rootname
 # Never infer completion from a directory without a receipt.
 if installed.exists():
  if row.get('extraction_status')!='complete': raise RuntimeError(f'Existing destination without complete receipt: {installed}')
  receipt=pathlib.Path(row['file_inventory'])
  verified=0
  with receipt.open() as f:
   for line in f:
    item=json.loads(line); p=BASE/item['path']
    if p.is_symlink() or not p.is_file(): raise RuntimeError(f'Installed asset missing/nonregular: {p}')
    n,h=digest(p)
    if (n,h)!=(item['bytes'],item['sha256']): raise RuntimeError(f'Installed asset changed: {p}')
    verified+=1
  event('reuse_verified_extraction',name=name,files=verified)
  return
 stage.mkdir(exist_ok=True)
 inventory=LOG/(rootname+'_files.jsonl')
 with zipfile.ZipFile(archive) as z:
  infos=z.infolist()
  if len(infos)>500000: raise RuntimeError('ZIP member budget exceeded')
  seen=set();selected=[];excluded=[];total=selected_bytes=0;licenses=[]
  for i in infos:
   parts=safe_member(i)
   key='/'.join(parts)
   if key in seen: raise RuntimeError(f'Duplicate ZIP member: {key}')
   seen.add(key)
   if parts[0] not in (rootname,'__MACOSX'): raise RuntimeError(f'Unexpected root {parts[0]!r} in {name}')
   metadata_only=parts[0]=='__MACOSX' or parts[-1]=='.DS_Store'
   keep=not metadata_only and (name!='embodiments.zip' or any(p.lower()=='aloha-agilex' for p in parts) or license_file(parts))
   row['excluded_metadata_note']='Apple __MACOSX resource forks and .DS_Store are excluded; safe-path and full streamed CRC checks still apply. Other exclusions are non-Aloha robot assets and non-license root metadata.'
   total+=i.file_size
   if keep: selected.append(i); selected_bytes+=i.file_size
   else: excluded.append(i.filename)
   if license_file(parts) and not i.is_dir(): licenses.append(i.filename)
  if total>BUDGET: raise RuntimeError('ZIP uncompressed budget exceeded')
  previous=sum(v.get('declared_uncompressed_bytes',0) for k,v in m['archives'].items() if k!=name)
  if total+previous>160*GIB: raise RuntimeError('Total uncompressed budget exceeded')
  if shutil.disk_usage(BASE).free < selected_bytes+20*GIB: raise RuntimeError('Insufficient extraction budget/headroom')
  if name=='embodiments.zip' and not any('aloha-agilex' in [p.lower() for p in safe_member(i)] and not i.is_dir() for i in selected):
   raise RuntimeError('Aloha-AgileX subtree not found')
  row.update(status='extracting',declared_uncompressed_bytes=total,selected_uncompressed_bytes=selected_bytes,
             member_count=len(infos),selected_member_count=len(selected),excluded_member_count=len(excluded),license_members=licenses,
             extraction_stage=str(stage),file_inventory=str(inventory),safe_zip_preflight=True)
  atomic_json(LOG/(rootname+'_excluded.json'),excluded)
  row['excluded_member_list']=str(LOG/(rootname+'_excluded.json'))
  save();event('zip_preflight_passed',name=name,total_members=len(infos),selected_members=len(selected),excluded_members=len(excluded),uncompressed_bytes=total,selected_bytes=selected_bytes,licenses=licenses)
  selected_names={i.filename for i in selected};written=files=crc_count=0;next_report=GIB
  with inventory.open('w') as out:
   for i in infos:
    guard();parts=safe_member(i);keep=i.filename in selected_names
    if i.is_dir():
     if keep: (stage/pathlib.Path(*parts)).mkdir(parents=True,exist_ok=True)
     continue
    target=stage/pathlib.Path(*parts)
    f=None
    if keep:
     target.parent.mkdir(parents=True,exist_ok=True)
     if target.is_symlink(): raise RuntimeError('Existing staging symlink')
     f=target.open('wb')
    h=hashlib.sha256();n=0
    try:
     with z.open(i) as src:
      while True:
       guard();b=src.read(CHUNK)
       if not b:break
       n+=len(b)
       if n>i.file_size: raise RuntimeError('ZIP member exceeded declared size')
       if f: f.write(b);h.update(b)
     if n!=i.file_size: raise RuntimeError('ZIP member length mismatch')
    finally:
     if f:f.close()
    # ZipExtFile verifies CRC at EOF; excluded robot members are streamed/discarded too.
    crc_count+=1
    if keep:
     os.chmod(target,0o644)
     files+=1;written+=n
     out.write(json.dumps({'path':i.filename,'bytes':n,'sha256':h.hexdigest(),'zip_crc32':f'{i.CRC:08x}'})+'\n')
    if written>=next_report:
     event('extraction_progress',name=name,written_bytes=written,files=files,crc_checked_files=crc_count)
     next_report=written+GIB
  if written!=selected_bytes: raise RuntimeError('Extracted total differs from selected total')
 if not (stage/rootname).is_dir(): raise RuntimeError('Missing extraction root')
 os.replace(stage/rootname,installed);stage.rmdir()
 row.update(status='extracted_verified',extraction_status='complete',extracted_bytes=written,extracted_files=files,
            crc_checked_files=crc_count,crc_all_archive_files_verified=True,extraction_path=str(installed));save()
 event('extraction_verified',name=name,bytes=written,files=files,crc_checked_files=crc_count)

try:
 guard(True);save()
 tree=json.loads(remote_bytes(API+f'/tree/{REV}?recursive=false&expand=false'))
 atomic_json(LOG/'upstream_root_metadata.json',tree)
 info=json.loads(remote_bytes(API+f'/revision/{REV}'))
 atomic_json(LOG/'upstream_dataset_metadata.json',info)
 if info.get('sha')!=REV: raise RuntimeError('Dataset API revision mismatch')
 readme=remote_bytes(ORIGIN+f'/raw/{REV}/README.md')
 (LOG/'upstream_README.txt').write_bytes(readme)
 m['data_license_metadata']=info.get('cardData',{})
 m['data_license_note']='Preserve all archive license/notice/copyright files. RoboTwin source MIT license does not establish licensing for every third-party asset; consult archive notices and upstream metadata.'
 event('provenance',revision=info.get('sha'),card_data=info.get('cardData',{}),root_readme=readme.decode('utf-8',errors='replace'))
 source_license=ROOT/'third_party/RoboTwin/LICENSE'
 provenance=BASE/'provenance';provenance.mkdir(exist_ok=True)
 shutil.copyfile(source_license,provenance/'LICENSE.RoboTwin.source')
 m['source_license']={'original':str(source_license),'preserved_at':str(provenance/'LICENSE.RoboTwin.source'),'sha256':digest(source_license)[1],'applies_to':'RoboTwin source; not blanket relicensing of downloaded assets'}
 for name,(size,sha) in EXPECTED.items():
  entry=next(x for x in tree if x.get('path')==name)
  if (entry['size'],entry['lfs']['size'],entry['lfs']['oid'])!=(size,size,sha): raise RuntimeError(f'Official metadata mismatch: {name}')
  row=m['archives'].setdefault(name,{})
  row.update(repository=REPO,revision=REV,url=ORIGIN+f'/resolve/{REV}/{name}',expected_bytes=size,expected_sha256=sha,upstream_lfs=entry['lfs'])
  save()
  archive=download(name,row)
  extract(name,row,archive)
 # Pure textual template conversion, identical substitutions to upstream script.
 templates=sorted((BASE/'embodiments').rglob('*_tmp.yml'))
 if not templates: raise RuntimeError('No embodiment configuration templates found')
 for template in templates:
  target=template.with_name(template.name.replace('_tmp.yml','.yml'))
  content=template.read_text();generated=content.replace('${ASSETS_PATH}',str(BASE.parent)).replace('$ASSETS_PATH',str(BASE.parent))
  if target.exists():
   if target.read_text()!=generated: raise RuntimeError(f'Non-generated/conflicting asset config exists; not overwritten: {target}')
  else: target.write_text(generated)
  item={'template':str(template),'path':str(target),'sha256':digest(target)[1],'substitution_ASSETS_PATH':str(BASE.parent),'upstream_rule_source':str(ROOT/'third_party/RoboTwin/scripts/update_embodiment_config_path.py')}
  if not any(x['path']==item['path'] for x in m['generated_configs']):m['generated_configs'].append(item)
  save()
 for name in EXPECTED:
  rootname=name[:-4];target=BASE/rootname;link=ROOT/'third_party/RoboTwin/assets'/rootname
  if os.path.lexists(link):
   if not link.is_symlink() or link.resolve()!=target.resolve():raise RuntimeError(f'Conflicting asset link: {link}')
  else:link.symlink_to(target,target_is_directory=True)
  item={'path':str(link),'target':str(target)}
  if item not in m['symlinks']:m['symlinks'].append(item)
  save()
 config_dir=ROOT/'third_party/RoboTwin/assets/embodiments/aloha-agilex'
 if not config_dir.is_dir():raise RuntimeError(f'Required configured robot path missing: {config_dir}')
 m['status']='verified_cleanup_pending';save()
 for name,row in m['archives'].items():
  path=TRANS/name
  if path.exists():path.unlink()
  row.update(transfer_deleted_after_all_verifications=True,status='complete')
  save()
 for path in (TRANS,STAGING):
  if path.exists() and not any(path.iterdir()):path.rmdir()
 m['status']='complete';m['completed_at']=now();save()
 event('complete',manifest=str(MANIFEST),asset_root=str(BASE),extracted_bytes=sum(x['extracted_bytes'] for x in m['archives'].values()),generated_configs=len(m['generated_configs']),symlinks=m['symlinks'])
except BaseException as e:
 m['status']='partial';m['errors'].append({'time':now(),'type':type(e).__name__,'message':str(e)});save()
 event('failed_partial_preserved',error=str(e),traceback=traceback.format_exc(),manifest=str(MANIFEST))
 sys.exit(1)
