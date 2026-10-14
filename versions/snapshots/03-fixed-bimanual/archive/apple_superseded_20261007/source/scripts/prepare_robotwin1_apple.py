#!/usr/bin/env python3
"""Download pinned official RoboTwin1 code and two small official assets; no jobs executed."""
from pathlib import Path,PurePosixPath
import urllib.request,json,hashlib,tarfile,zipfile,shutil,concurrent.futures,stat
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'third_party/RoboTwin1'
OUT=ROOT/'outputs/verification/robotwin_fruit_20261007/robotwin1_install'
REV='65a6f5bb74794ba193c6bf17d8eae3b74fc2df70'
AREV='b59d5b9c07e1653d2114197ed6d4b18d228e5c01'
ITEMS=[('source.tar.gz',f'https://codeload.github.com/RoboTwin-Platform/RoboTwin/tar.gz/{REV}',None),('aloha_urdf.zip',f'https://huggingface.co/datasets/ZanxinChen/RoboTwin_asset/resolve/{AREV}/aloha_urdf.zip','6cd1f24cc29a6ecaa384fbeac3ddf5373a679132a59272369b4967beab50fec7'),('main_models.zip',f'https://huggingface.co/datasets/ZanxinChen/RoboTwin_asset/resolve/{AREV}/main_models.zip','33d434c27252bb0d320d98ecf7df603f89daf958df5f8a64992815ef27d160ae')]
def download(item):
 name,url,expected=item;p=OUT/name;h=hashlib.sha256()
 if not p.exists():
  with urllib.request.urlopen(url,timeout=90) as src,p.with_suffix(p.suffix+'.part').open('wb') as dst:
   while b:=src.read(2**20):dst.write(b)
  p.with_suffix(p.suffix+'.part').replace(p)
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 if expected and h.hexdigest()!=expected:raise ValueError('SHA mismatch '+name)
 print(name,p.stat().st_size,h.hexdigest(),flush=True)
 return dict(file=name,url=url,sha256=h.hexdigest(),size=p.stat().st_size,lfs_sha256_verified=bool(expected))
def safe(name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError(name)
 return p
def main():
 OUT.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
 records=list(concurrent.futures.ThreadPoolExecutor(3).map(download,ITEMS))
 with tarfile.open(OUT/'source.tar.gz') as tar:
  for m in tar:
   p=safe(m.name);parts=p.parts[1:]
   if not parts:continue
   rel=PurePosixPath(*parts)
   if parts[0] not in ('envs','task_config','script','third_party','LICENSE','README.md','INSTALLATION.md','CONFIG_TUTORIAL.md'):continue
   if m.issym() or m.islnk():continue
   dst=DEST/str(rel)
   if m.isdir():dst.mkdir(parents=True,exist_ok=True)
   elif m.isfile():
    dst.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(m) as src,dst.open('wb') as f:shutil.copyfileobj(src,f)
 for name,_,_ in ITEMS[1:]:
  with zipfile.ZipFile(OUT/name) as z:
   for m in z.infolist():
    p=safe(m.filename)
    if '__MACOSX' in p.parts or p.name=='.DS_Store':continue
    if stat.S_ISLNK(m.external_attr>>16):raise ValueError('symlink '+str(p))
    dst=DEST/str(p)
    if m.is_dir():dst.mkdir(parents=True,exist_ok=True)
    else:
     dst.parent.mkdir(parents=True,exist_ok=True)
     with z.open(m) as src,dst.open('wb') as f:shutil.copyfileobj(src,f,2**20)
 receipt={'status':'downloaded_not_executed','official_code_revision':REV,'official_assets_revision':AREV,'files':records,'root':str(DEST),'task':'pick_apple_messy','known_deviation':'none yet; runtime compatibility pending'}
 (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(DEST/'DOWNLOAD_PROVENANCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('EXTRACTED',str(DEST),flush=True)
if __name__=='__main__':main()
