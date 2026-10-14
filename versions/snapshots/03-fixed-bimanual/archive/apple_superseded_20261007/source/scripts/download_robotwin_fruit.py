#!/usr/bin/env python3
"""CPU-only, pinned official-linked apple candidate; does not relabel it grab_fruit.
Download/verify/extract with bounded RAM. Never touches existing task datasets.
"""
import hashlib,json,shutil,stat,time,urllib.request,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/benchmarks/robotwin_fruit/pick_apple_messy_early_raw'
REPO='YaoMarkMu/RoboTwin-Earlyversion'
REV='53fda493155c73c4a1829a6638535f787ca11009'
ARCHIVE='pick_apple_messy.zip'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def get(u):return urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'robotwin-fruit-audit/1'}),timeout=90)
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 api=f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}?limit=1000'
 with get(api) as r:entries=json.load(r)
 entry=next(x for x in entries if x['path']==ARCHIVE)
 size=entry['size']; expected=entry['lfs']['oid']
 assert size<2*1024**3 and shutil.disk_usage(OUT).free>4*size
 url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{ARCHIVE}'
 receipt={'schema_version':1,'task':'pick_apple_messy','match_status':'closest_verified_grasp_only_candidate_not_user_confirmed','repo':REPO,'revision':REV,'url':url,'archive':ARCHIVE,'upstream_file':entry,'expected_bytes':size,'expected_sha256':expected,'gpu_jobs':0,'status':'downloading','official_link':'https://github.com/RoboTwin-Platform/RoboTwin/blob/f14a315ac4c9c72bfea05b523874f787080dfcf2/README.md','official_link_original_dataset':'https://huggingface.co/datasets/YaoMarkMu/robotwin_dataset','started_unix':time.time()}
 def save(): (OUT/'SOURCE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 save()
 try:
  path=OUT/ARCHIVE
  with get(url) as r,path.open('xb') as f:
   while b:=r.read(2**20):f.write(b)
  assert path.stat().st_size==size and sha(path)==expected
  receipt.update(status='archive_verified',sha256=expected,received_bytes=size);save()
  with zipfile.ZipFile(path) as z:
   infos=z.infolist(); assert sum(x.file_size for x in infos)<8*1024**3
   names=set()
   for i in infos:
    p=PurePosixPath(i.filename)
    assert not(p.is_absolute() or '..' in p.parts or '\\' in i.filename or stat.S_ISLNK(i.external_attr>>16) or i.filename in names)
    names.add(i.filename)
   (OUT/'archive_members.json').write_text(json.dumps([{'path':i.filename,'bytes':i.file_size,'crc32':f'{i.CRC:08x}'} for i in infos],indent=2)+'\n')
   for i in infos:
    p=OUT/'source'/i.filename
    if i.is_dir():p.mkdir(parents=True,exist_ok=True);continue
    p.parent.mkdir(parents=True,exist_ok=True)
    with z.open(i) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,2**20)
  receipt.update(status='complete',zip_crc_verified=True,transport_archive_removed=False,extracted_bytes=sum(i.file_size for i in infos),members=len(infos),elapsed_seconds=time.time()-receipt['started_unix']);save()
  print(json.dumps(receipt,indent=2),flush=True)
 except Exception as e:
  receipt.update(status='failed',error=repr(e));save();raise
if __name__=='__main__':main()
