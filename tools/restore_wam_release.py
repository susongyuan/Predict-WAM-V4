#!/usr/bin/env python3
"""Verify split release archives and restore paths without replacing existing files."""
import argparse,hashlib,json,os,tarfile
from pathlib import Path

class PartsReader:
 def __init__(self,paths):self.paths=iter(paths);self.current=None
 def read(self,n=-1):
  if n<0:raise ValueError('Bounded reads only')
  chunks=[];remaining=n
  while remaining:
   if self.current is None:
    try:self.current=next(self.paths).open('rb')
    except StopIteration:break
   chunk=self.current.read(remaining)
   if chunk:chunks.append(chunk);remaining-=len(chunk)
   else:self.current.close();self.current=None
  return b''.join(chunks)
 def close(self):
  if self.current:self.current.close()

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()

def main():
 a=argparse.ArgumentParser();a.add_argument('--assets',type=Path,required=True);a.add_argument('--category',required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
 manifest=json.loads((args.assets/(args.category+'.manifest.json')).read_text());parts=[]
 for p in manifest['parts']:
  if Path(p['name']).name!=p['name']:raise ValueError('Unsafe asset name')
  path=args.assets/p['name']
  if path.stat().st_size!=p['bytes'] or digest(path)!=p['sha256']:raise ValueError('Part verification failed: '+p['name'])
  parts.append(path)
 target=args.out.resolve();target.mkdir(parents=True,exist_ok=True)
 def safe(name):
  path=(target/name).resolve()
  if not path.is_relative_to(target) or path==target:raise ValueError('Unsafe archive path')
  return path
 known={r['path']:r for r in manifest['files']};seen=set();stream=PartsReader(parts)
 try:
  with tarfile.open(fileobj=stream,mode='r|*') as tar:
   for member in tar:
    if member.name not in known or member.name in seen:raise ValueError('Unexpected/duplicate archive member')
    path=safe(member.name)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if member.islnk():
     original=safe(member.linkname)
     if not original.is_file() or member.linkname not in seen:raise ValueError('Missing hardlink source')
     if known[member.linkname]['sha256']!=known[member.name]['sha256']:raise ValueError('Hardlink hash mismatch')
     os.link(original,path)
    elif member.isfile():
     h=hashlib.sha256()
     with tar.extractfile(member) as f,path.open('xb') as out:
      while b:=f.read(4*1024*1024):out.write(b);h.update(b)
     if h.hexdigest()!=known[member.name]['sha256']:raise ValueError('File verification failed: '+member.name)
    else:raise ValueError('Unsupported archive member')
    if path.stat().st_size!=known[member.name]['bytes']:raise ValueError('File length mismatch')
    seen.add(member.name)
 finally:stream.close()
 if seen!=set(known):raise ValueError('Incomplete archive')
 print(json.dumps({'status':'verified_restored','files':len(seen),'out':str(target)}))

if __name__=='__main__':main()
