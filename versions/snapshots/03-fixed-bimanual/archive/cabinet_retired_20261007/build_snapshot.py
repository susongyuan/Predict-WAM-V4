#!/usr/bin/env python3
"""CPU-only archival snapshot. Never removes or rewrites source files."""
from pathlib import Path
import json,hashlib,gzip,shutil,datetime,collections,os
ROOT=Path('/home/user/Robotic/project2'); os.chdir(ROOT)
A=ROOT/'archive/cabinet_retired_20261007'; SRC=A/'source'; SRC.mkdir(exist_ok=True)
text={'.json','.jsonl','.yaml','.yml','.toml','.md','.txt','.csv','.tsv','.log','.sh','.py','.sha256','.ini','.cfg'}
vd=Path('outputs/validation'); prefixes=('robotwin_','cabinet_','middle_','integrated_','system_comparison_','checkpoint_sweep','resume_20261005','feedback_','target_','train32_','train455_','vtt_retrieval_','wam_grasp_','wam_visual_','wrist_')
roots=[Path('outputs/runs/put_object_cabinet'),Path('data/benchmarks/robotwin2/put_object_cabinet_500_v1'),Path('data/benchmarks/robotwin2/put_object_cabinet_randomized'),Path('data/robotwin_features')]
roots += [p for p in vd.iterdir() if p.is_dir() and p.name.startswith(prefixes)]
roots += [p for p in Path('outputs/training').iterdir() if p.is_dir() and p.name.startswith(('cabinet_','wrist_target_'))]
roots += [Path('outputs/visualizations/system10_success_20261006'),Path('outputs/visualizations/wam_native_success_20261005')]
# Source/config snapshot and retained initialization's small metadata, never weights.
roots += [Path('scripts'),Path('harness'),Path('configs'),Path('outputs/training/task090_v3_50k_20260930_s050000')]
roots += [p for p in Path('docs').glob('*.md') if p.name not in {'current_status.md'}]
files={}
for root in roots:
 if not root.exists():continue
 for p in ([root] if root.is_file() else root.rglob('*')):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in str(p):files[str(p)]=p
records=[];inv=[]
success_names={'dev_30k_isolated700_seed10002_20261005_2041','system10_transparent_seed10014_20261006'}
partial='middle10_vtt_only_seed10012_20261006'
def write(p,mode='copy',payload=None):
 b=p.read_bytes() if payload is None else payload
 dst=SRC/p
 if mode=='gzip':dst=Path(str(dst)+'.gz')
 if mode=='excerpt':dst=Path(str(dst)+'.first_last.jsonl')
 dst.parent.mkdir(parents=True,exist_ok=True)
 if mode=='gzip':
  with gzip.GzipFile(filename=str(dst),mode='wb',compresslevel=3,mtime=0) as g:g.write(b)
 else:dst.write_bytes(b)
 records.append({'source':str(p),'archived':str(dst.relative_to(A)), 'mode':mode,'source_bytes':p.stat().st_size,'preserved_bytes':len(b),'preserved_sha256':hashlib.sha256(b).hexdigest(),'archive_bytes':dst.stat().st_size,'archive_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
for s,p in sorted(files.items()):
 n=p.stat().st_size
 inv.append({'path':s,'bytes':n,'mtime_ns':p.stat().st_mtime_ns})
 special=any(x in p.parts for x in success_names|{partial})
 if p.suffix.lower() in text or p.name in {'requirements.txt','Dockerfile'}:
  if p.name=='evaluation_gt.jsonl' and not special:
   with p.open('rb') as f:
    head=[];tail=collections.deque(maxlen=5)
    for i,line in enumerate(f):
     if i<2:head.append(line)
     else:tail.append(line)
   write(p,'excerpt',b''.join(head+list(tail)))
  elif p.suffix=='.log' and n>2*1024**2:
   with p.open('rb') as f:f.seek(-1024**2,2);b=f.read()
   write(p,'excerpt',b)
  elif n>2*1024**2:write(p,'gzip')
  else:write(p)
 elif special and (p.name=='third_view.mp4' or p.name.startswith('initial_') and p.suffix=='.png'):
  write(p)
# A few exact original onboard observations (not complete streams).
for name in sorted(success_names|{partial}):
 for ep in Path('outputs/runs/put_object_cabinet',name).glob('episode_*'):
  frames=sorted((ep/'onboard').glob('observation_*.npz'))
  if frames:
   picked={frames[0],frames[-1]}
   if name in success_names:
    term=324 if '10014' in name else 304
    picked.update(p for p in frames if p.stem in {f'observation_{term-1:06d}',f'observation_{term:06d}'})
   for p in sorted(picked):write(p)
# Full pure-WAM GIF is compact; system video + selected onboard observations suffice.
p=Path('outputs/visualizations/wam_native_success_20261005/ckpt30000_seed10002_native_success_full.gif')
if p.exists():write(p)
(A/'source_manifest.json').write_text(json.dumps({'created_at':datetime.datetime.now().astimezone().isoformat(),'scope_roots':[str(p) for p in roots], 'note':'Metadata/source only; no large checkpoints/raw datasets/features. GT streams are first2+last5 except two representative successes and user-cancelled run. Excerpt hashes cover preserved bytes, not full source.','files':records},ensure_ascii=False,indent=2))
(A/'source_inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2))
errs=[]
for r in records:
 p=A/r['archived'];b=gzip.decompress(p.read_bytes()) if r['mode']=='gzip' else p.read_bytes()
 if len(b)!=r['preserved_bytes'] or hashlib.sha256(b).hexdigest()!=r['preserved_sha256']:errs.append(str(p))
summary={'files_preserved':len(records),'archive_bytes':sum(r['archive_bytes'] for r in records),'source_files_inventoried':len(inv),'verification_errors':errs}
(A/'snapshot_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
