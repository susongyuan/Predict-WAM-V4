from pathlib import Path
import json,hashlib,shutil,os,datetime
R=Path('/home/user/Robotic/project2');A=R/'archive/apple_superseded_20261007'
roots=['outputs/runs/pick_apple_messy','outputs/pipelines/robotwin1_apple_20261007','outputs/verification/robotwin_fruit_20261007','data/benchmarks/robotwin_fruit','third_party/RoboTwin1','data/robotwin1_apple_features','data/benchmarks/robotwin1/pick_apple_messy_100_rgb_v1']
text={'.json','.jsonl','.md','.yml','.yaml','.py','.txt','.log'};records=[]
for rel in roots:
 root=R/rel
 if not root.exists():continue
 for p in root.rglob('*'):
  if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
  keep=(p.suffix in text and p.stat().st_size<2*1024**2 and '/models/' not in str(p) and '/aloha_maniskill_sim/' not in str(p))
  keep=keep or ('native_firstframe_910000' in p.parts and p.suffix=='.png')
  if not keep:continue
  dst=A/'source'/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
  records.append({'source':str(p.relative_to(R)),'archive':str(dst.relative_to(A)),'size':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
# Quarantine apple-specific runnable drafts; keep generic trainer/posttrain queue.
code=['scripts/prepare_robotwin1_apple.py','scripts/probe_robotwin1_apple.py','scripts/collect_robotwin1_apple.py','scripts/batch_collect_robotwin1_apple.py','scripts/run_robotwin1_apple_schedule.py','scripts/test_robotwin1_apple_schedule.py','scripts/audit_robotwin_fruit_candidate.py','scripts/download_robotwin_fruit.py','harness/robotwin1_expert_planner.py','harness/robotwin1_wam_control.py']
for rel in code:
 p=R/rel
 if p.exists():
  dst=A/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);records.append({'source':rel,'archive':str(dst.relative_to(A)),'size':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
for r in records:assert hashlib.sha256((A/r['archive']).read_bytes()).hexdigest()==r['sha256']
receipt={'reason':'User superseded grasp-only apple task with disposal/container benchmark search; no WAM training or checkpoint tests were started. Expert successes are not policy successes.','saved_files':len(records),'saved_bytes':sum(x['size'] for x in records),'files':records,'removed':[],'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'other_session_touched':False}
(A/'preservation_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
for rel in roots+code:
 p=R/rel
 if not p.exists():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(R)
 size=sum(x.stat().st_size for x in p.rglob('*') if x.is_file() and not x.is_symlink()) if p.is_dir() else p.stat().st_size
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
 receipt['removed'].append({'path':rel,'logical_bytes':size})
receipt.update(status='complete',removed_logical_bytes=sum(x['logical_bytes'] for x in receipt['removed']))
(A/'cleanup_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','removed')},ensure_ascii=False,indent=2))
