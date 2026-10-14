#!/usr/bin/env python3
"""Snapshot existing WAM artifacts and publish resumable, verified GitHub Release assets.
Does not execute training, evaluation, or deserialize checkpoints.
"""
import argparse, hashlib, json, os, re, subprocess, tarfile, time, shutil, gzip
from pathlib import Path

REPO='susongyuan/Predict-WAM-V4'
TAG='wam-history-artifacts-20261014'
GH=shutil.which('gh') or str(Path.home()/'.local/bin/gh')
BLOCK=4*1024*1024
PART=512*1024*1024
BAD_PARTS={'.git','__pycache__','.pytest_cache','appdata','cache','Cache','cache2','nvidia_5959107','nvidia_5959902','local_driver_libs','site-packages','node_modules'}
TEXT={'.py','.md','.json','.jsonl','.yaml','.yml','.txt','.csv','.log','.stdout','.stderr','.sh','.patch','.sha256','.xml','.rst'}
EVIDENCE={'.npz','.npy','.png','.jpg','.jpeg','.mp4','.gif','.gz','.zip','.svg','.html'}
SECRET=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}')

def write_json(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def gh(*args):
 p=subprocess.run([GH,*args],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[-2000:])
 return p.stdout

class SplitWriter:
 def __init__(self,base):self.base=base;self.index=0;self.stream=None;self.parts=[];self.pos=0
 def open(self):
  self.path=Path(str(self.base)+f'.part{self.index:04d}');self.stream=self.path.open('wb');self.hash=hashlib.sha256();self.size=0;self.index+=1
 def write(self,data):
  offset=0
  while offset<len(data):
   if self.stream is None:self.open()
   n=min(PART-self.size,len(data)-offset);chunk=data[offset:offset+n];self.stream.write(chunk);self.hash.update(chunk);self.size+=n;self.pos+=n;offset+=n
   if self.size==PART:self.finish()
  return len(data)
 def tell(self):return self.pos
 def flush(self):
  if self.stream:self.stream.flush()
 def finish(self):
  if self.stream:
   self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.stream=None
   self.parts.append(dict(name=self.path.name,bytes=self.size,sha256=self.hash.hexdigest()))
 def close(self):self.finish()

class HashReader:
 def __init__(self,fp):self.fp=fp;self.hash=hashlib.sha256()
 def read(self,n=-1):
  b=self.fp.read(n);self.hash.update(b);return b

def plan(root):
 groups={x:[] for x in ['priority-weights','metadata','datasets','remaining-weights','observations-and-results','historical-sources']};excluded=[]
 roots=['outputs/training','outputs/runs','outputs/pipelines','outputs/behavior','outputs/validation','outputs/zhuhai2026_tasks12','outputs/verification','outputs/construction','outputs/architecture','outputs/archive','archive','docs','data/isaac_zhuhai','data/zhuhai_hjepa_preparation_20261009']
 priority=['outputs/training/task090_v3_50k_20260930_s050000/checkpoint_step_050000.pt','outputs/runs/put_bottles_dustbin/wam_30k_20261007/train/checkpoint_step_025000.pt','outputs/runs/put_bottles_dustbin/wam_30k_20261007/train/checkpoint_step_030000.pt','outputs/runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/train/checkpoint_step_050000.pt','outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger30k_diagnostic_20261011_v23/train/checkpoint_step_080000.pt','outputs/zhuhai2026_tasks12/wam_precision_v4_task2_future16_50k_20261013_b23_fast/train/checkpoint_step_127500.pt','outputs/zhuhai2026_tasks12/wam_precision_v4_task2_future16_50k_20261013_b23_fast/train/checkpoint_step_130000.pt']
 for base in roots:
  for d,ds,fs in os.walk(root/base,followlinks=False):
   ds[:]=sorted(x for x in ds if x not in BAD_PARTS and not x.startswith('nvidia_'))
   for name in sorted(fs):
    p=Path(d)/name;r=str(p.relative_to(root));ext=p.suffix.lower()
    if p.is_symlink() or not p.is_file():continue
    if name in ['.env','credentials','token','id_rsa','id_ed25519'] or ext in ['.key','.pem','.lock','.pid']:
     excluded.append(dict(path=r,reason='credential-or-runtime-file'));continue
    st=p.stat()
    if ext in TEXT:
     # Scan all text in bounded blocks; never emit matched secrets.
     secret=False;tail=b''
     with p.open('rb') as f:
      while b:=f.read(BLOCK):
       if SECRET.search(tail+b):secret=True;break
       tail=b[-256:]
     if secret:excluded.append(dict(path=r,reason='secret-pattern'));continue
     cat='historical-sources' if ext in ['.py','.sh','.patch'] else 'metadata'
    elif ext in ['.pt','.pth','.ckpt']:cat='priority-weights' if r in priority else 'remaining-weights'
    elif ext in ['.hdf5','.h5']:cat='datasets'
    elif ext in EVIDENCE:cat='observations-and-results'
    else:
     excluded.append(dict(path=r,reason='not-in-artifact-allowlist'));continue
    groups[cat].append(dict(path=r,bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev))
 for n in ['README.md','progress.md','progress1.md','AGENTS.md']:
  p=root/n;st=p.stat();groups['metadata'].append(dict(path=n,bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev))
 groups['priority-weights'].sort(key=lambda r:priority.index(r['path']))
 return dict(groups=groups,excluded=excluded,external_data_policy='Upstream BEHAVIOR/RoboTwin data, simulator assets and pretrained encoders are referenced by provenance, not silently relicensed or mirrored. Generated project demonstrations and retained evaluation observations are included. Deleted historical weights cannot be reconstructed.')

def main():
 a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--staging',type=Path,required=True);a.add_argument('--publish',action='store_true');args=a.parse_args()
 args.staging.mkdir(parents=True,exist_ok=True);state_path=args.staging/'status.json'
 def status(**kw):
  old=json.loads(state_path.read_text()) if state_path.exists() else {};old.update(kw,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write_json(state_path,old)
 try:
  status(status='inventory',error=None)
  pp=args.staging/'plan.json'
  if not pp.exists():write_json(pp,plan(args.source))
  inventory=json.loads(pp.read_text());summary={k:dict(files=len(v),bytes=sum(r['bytes'] for r in v)) for k,v in inventory['groups'].items()};status(summary=summary,status='packaging')
  if args.publish:
   exists=subprocess.run([GH,'release','view',TAG,'--repo',REPO],capture_output=True).returncode==0
   if not exists:
    notes=args.staging/'release-notes.md';notes.write_text('''# WAM 历史代码、数据、权重与结果\n\n原始证据快照发布。各版本/任务索引见仓库 history/EVOLUTION.md。文件按类别切分：同一类别全部 part 按编号拼接后得到 tar；请先核验 manifest 的 SHA256 再解包。\n\n发布过程先使用 draft，全部上传核验后才正式发布。包括失败与未完成记录，历史计划/授权不是现在的运行指令。旧移动任务已删除的权重不能恢复。上游数据/资产仍按来源和原许可证下载。\n''')
    gh('release','create',TAG,'--repo',REPO,'--target','main','--draft','--title','WAM evolution: retained datasets, checkpoints and experiment evidence','--notes-file',str(notes))
  order=['historical-sources','priority-weights','metadata','datasets','remaining-weights','observations-and-results']
  for cat in order:
   rows=inventory['groups'][cat]
   manifest_path=args.staging/f'{cat}.manifest.json';compressed=cat in ('metadata','historical-sources');base=args.staging/(cat+('.tar.gz' if compressed else '.tar'))
   if manifest_path.exists():manifest=json.loads(manifest_path.read_text())
   else:
    # Incomplete category is never treated as a valid previous archive.
    for p in args.staging.glob(cat+'.tar*.part*'):p.unlink()
    writer=SplitWriter(base);records=[];inodes={};start=time.time()
    payload=gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) if compressed else writer
    with tarfile.open(fileobj=payload,mode='w|',format=tarfile.PAX_FORMAT) as tar:
     for i,row in enumerate(rows):
      snapshot=args.staging/'source-snapshot'/row['path']
      p=snapshot if snapshot.is_file() else args.source/row['path'];st=p.stat()
      if (st.st_size,st.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):raise RuntimeError('Source changed since inventory: '+row['path'])
      ti=tar.gettarinfo(str(p),arcname=row['path']);ti.uid=ti.gid=0;ti.uname=ti.gname='';ti.mode=0o755 if p.suffix=='.sh' else 0o644
      key=(st.st_dev,st.st_ino)
      if key in inodes:
       original=inodes[key];ti.type=tarfile.LNKTYPE;ti.linkname=original['path'];ti.size=0;tar.addfile(ti);digest=original['sha256'];alias=original['path']
      else:
       with p.open('rb') as f:reader=HashReader(f);tar.addfile(ti,reader);digest=reader.hash.hexdigest()
       alias=None;inodes[key]=dict(path=row['path'],sha256=digest)
      records.append(dict(path=row['path'],bytes=row['bytes'],sha256=digest,hardlink_to=alias))
      if i%50==0 or row['bytes']>100*1024**2:
       status(status='packaging',category=cat,files_done=i+1,files_total=len(rows),archive_bytes=writer.tell(),elapsed_seconds=round(time.time()-start))
    if compressed:payload.close()
    writer.close();manifest=dict(category=cat,format='concatenate parts in lexical order to a single '+('tar.gz' if compressed else 'tar'),parts=writer.parts,files=records,source_root=str(args.source),snapshot_time=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write_json(manifest_path,manifest)
   if args.publish:
    remote=next(x for x in json.loads(gh('api',f'repos/{REPO}/releases')) if x['tag_name']==TAG)
    release_id=remote['id']
    def assets():
     pages=json.loads(gh('api','--paginate','--slurp',f'repos/{REPO}/releases/{release_id}/assets?per_page=100'))
     return {x['name']:x for page in pages for x in page}
    uploaded=0
    for part in manifest['parts']:
     expected='sha256:'+part['sha256'];local=args.staging/part['name']
     for retry in range(4):
      old=assets().get(part['name'])
      if old:
       if old.get('state')=='uploaded':
        if old['size']!=part['bytes'] or old.get('digest')!=expected:raise RuntimeError('Remote completed asset differs: '+part['name'])
        break
       # Only incomplete assets from an interrupted transfer may be discarded.
       gh('api','--method','DELETE',f"repos/{REPO}/releases/assets/{old['id']}")
      status(status='uploading',category=cat,asset=part['name'],asset_bytes=part['bytes'],parts_done=uploaded,parts_total=len(manifest['parts']),retry=retry,upload_bytes_read=0)
      started=time.time()
      with (args.staging/'last-upload.log').open('w+') as log:
       proc=subprocess.Popen([GH,'release','upload',TAG,str(local),'--repo',REPO],stdout=log,stderr=log)
       while proc.poll() is None:
        transferred=None
        try:
         for fd in Path(f'/proc/{proc.pid}/fd').iterdir():
          if str(fd.readlink())==str(local):
           transferred=int(Path(f'/proc/{proc.pid}/fdinfo/{fd.name}').read_text().splitlines()[0].split()[1]);break
        except (OSError,ValueError):pass
        status(upload_elapsed_seconds=round(time.time()-started),upload_bytes_read=transferred)
        if time.time()-started>7200:proc.kill();proc.wait();break
        time.sleep(10)
       rc=proc.wait()
      if rc==0:
       item=assets().get(part['name'])
       if not item or item['size']!=part['bytes'] or item.get('digest')!=expected:raise RuntimeError('Post-upload SHA verification failed: '+part['name'])
       break
      if retry==3:raise RuntimeError('Upload failed after four attempts; see last-upload.log: '+part['name'])
      time.sleep(10)
     uploaded+=1;status(parts_done=uploaded,parts_total=len(manifest['parts']))
    existing=assets()
    if manifest_path.name not in existing:gh('release','upload',TAG,str(manifest_path),'--repo',REPO)
    status(status='category_verified',category=cat)
  if args.publish:
   gh('release','upload',TAG,str(pp),'--repo',REPO,'--clobber')
   gh('release','edit',TAG,'--repo',REPO,'--draft=false','--latest=false')
   publication=dict(status='complete',release_url=f'https://github.com/{REPO}/releases/tag/{TAG}',categories=summary,verified_remote_sha256=True,completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),limitations=inventory['external_data_policy'])
   repo_root=Path(__file__).resolve().parents[1]
   write_json(repo_root/'history/publication-status.json',publication)
   subprocess.run(['git','-C',str(repo_root),'add','history/publication-status.json'],check=True)
   subprocess.run(['git','-C',str(repo_root),'commit','--only','history/publication-status.json','-m','Record completed and SHA256-verified WAM history artifact publication','--quiet'],check=True)
   subprocess.run(['git','-C',str(repo_root),'push','origin','main'],check=True)
  status(status='complete',release_url=f'https://github.com/{REPO}/releases/tag/{TAG}')
 except BaseException as exc:status(status='failed',error=f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':main()
